from __future__ import annotations

import copy
import json

import pytest

import analyze_qwen36_fp8_kv_capacity_v79b as analysis


def _accepted_directory():
    return analysis.RUN_ROOT / analysis.V79B_ACCEPTED_RUN


def _accepted_receipt(gpu: int = 0):
    return json.loads(
        (_accepted_directory() / f"gpu_{gpu}.json").read_text(encoding="ascii")
    )


def _reseal(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = analysis.canonical_sha256(value)
    return value


def test_static_contract_binds_v79b_preregistration_and_exact_sources():
    result = analysis.validate_static_contract()
    prereg = result["preregistrations"][
        str(analysis.PREREG_V79B.relative_to(analysis.ROOT))
    ]
    assert prereg["content_sha256"] == (
        "7669c2f720f2a0d17e976de42cc5b7c08fba60a3251175a62eddf05de2dc1b5d"
    )
    assert prereg["file_sha256"] == (
        "8e1940db5134bb77ef9959d10b4eec5d43fab4e8653d62733b42939b5fd7300f"
    )
    assert result["sealed_sources"]["bundle_sha256"] == (
        "14fc85abb7b88eaa4def27d6d190c33b76e956cc453a7b46975a96b9487484f2"
    )


def test_finalized_analysis_matches_bound_capacity_performance_and_memory():
    result = analysis.analyze_finalized()
    assert result["passed"] is True
    assert result["content_sha256_before_self_field"] == (
        "dc2c3f47f28bd74bec0ddb385652c6263d38328f637f31c6769b1e48277ed46a"
    )
    assert result["capacity"]["v79b_capacity_matched_fp8_attention_kv_tokens"] == 162_304
    assert result["capacity"]["v79b_margin_over_v76_tokens"] == 4_608
    assert result["performance"]["v79b_cleanup_accepted_run_median_seconds"] == pytest.approx(
        49.10028860197053
    )
    assert result["performance"]["v79_all_same_model_four_replicates"][
        "combined_actor_median_seconds"
    ] == pytest.approx(48.984061275958084)
    assert result["vram"]["external_peak_memory_used_mib"] == {
        "v76_r7": 50_858,
        "v78_r3": 50_856,
        "v79b_r5_including_pynvml_observer": 49_994,
    }
    assert result["vram"]["v79b_peak_attributed_actor_process_mib"] == 49_384


def test_only_r5_has_corrected_external_cleanup_acceptance():
    result = analysis.analyze_finalized()
    assert result["evidence_classification"][
        "only_fully_preregistered_cleanup_accepted_run"
    ] == analysis.V79B_ACCEPTED_RUN
    assert result["v79b_accepted_run"]["telemetry"][
        "trailing_external_cleanup_batches"
    ] == [129, 130, 131]
    assert all(
        run["telemetry"]["external_cleanup_gate_passed"] is False
        and run["eligible_for_v79b_full_runtime_acceptance"] is False
        for run in result["v79_diagnostic_runs"]
    )


def test_paired_hash_drift_matrix_keeps_reference_nondeterminism_visible():
    drift = analysis.analyze_finalized()["paired_token_hash_drift"]
    assert drift["v76_r7_vs_v78_r3"]["differing_rows_total"] == 422
    assert drift["v76_r7_vs_v79b_r5"]["differing_rows_total"] == 430
    assert drift["v78_r3_vs_v79b_r5"]["differing_rows_total"] == 72
    assert drift["v78_r3_vs_v79b_r5"][
        "differing_rows_by_call_sum_gpus"
    ] == [22, 0, 0, 16, 15, 0, 0, 19]
    assert drift["v78_r3_vs_v79b_r5"]["compared_rows_total"] == 2_176


def test_actor_self_hash_and_resealed_runtime_tamper_fail_closed():
    receipt = _accepted_receipt()
    receipt["runtime"]["gpu_memory_utilization"] = 0.486
    with pytest.raises(RuntimeError, match="self hash"):
        analysis.validate_v79_actor(receipt, 0, "tampered")
    with pytest.raises(RuntimeError, match="precision/KV"):
        analysis.validate_v79_actor(_reseal(receipt), 0, "resealed-tampered")


def test_hash_only_output_counter_tamper_fails_even_when_resealed():
    receipt = _accepted_receipt()
    receipt["candidate_within_state_changed_rows"] = 1
    with pytest.raises(RuntimeError, match="derived output counters"):
        analysis.validate_v79_actor(_reseal(receipt), 0, "counter-tampered")


def test_forbidden_log_and_missing_backend_fail_closed(tmp_path):
    source = (_accepted_directory() / "gpu_0.log").read_text(encoding="utf-8")
    forbidden = tmp_path / "forbidden.log"
    forbidden.write_text(source + "\nfalling back\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden"):
        analysis.validate_log(forbidden, analysis.V79_TOKENS)
    missing = tmp_path / "missing.log"
    missing.write_text(
        source.replace("Using TRITON_ATTN attention backend", "attention backend"),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="required log"):
        analysis.validate_log(missing, analysis.V79_TOKENS)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda rows: rows[0].__setitem__("foreign_compute_pids", [999]), "telemetry contract"),
        (lambda rows: rows[0].__setitem__("hbm_bytes_per_second_inferred", True), "telemetry contract"),
        (lambda rows: rows.pop(), "incomplete four-GPU"),
        (
            lambda rows: [
                row.__setitem__("cleanup_nvidia_smi_memory_used_mib", 5)
                for row in rows
                if row["batch_index"] >= 129
            ],
            "cleanup gate",
        ),
    ],
)
def test_telemetry_foreign_pid_framing_inference_and_cleanup_fail_closed(
    tmp_path, mutation, message
):
    source = _accepted_directory() / "gpu_telemetry_v79.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="ascii").splitlines()]
    mutation(rows)
    path = tmp_path / "telemetry.jsonl"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="ascii",
    )
    pid_map = analysis._read_pid_map(_accepted_directory() / "actor_pids.csv")
    with pytest.raises(RuntimeError, match=message):
        analysis.validate_v79_telemetry(
            path, pid_map, require_external_cleanup=True
        )


def test_persisted_analysis_is_canonical_and_current():
    result = analysis.analyze_finalized()
    persisted = analysis.OUTPUT.read_text(encoding="ascii")
    assert persisted == analysis.render(result)
    body = copy.deepcopy(json.loads(persisted))
    claimed = body.pop("content_sha256_before_self_field")
    assert claimed == analysis.canonical_sha256(body)
