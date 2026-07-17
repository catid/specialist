from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import analyze_qwen36_v73b_lora_calibration as analysis


def _artifacts():
    return {
        "traffic": analysis._load_json(
            analysis.RUN / "exact_audit_traffic_v73b.json"
        ),
        "host_summary": analysis._load_json(
            analysis.RUN / "host_process_summary_v73b.json"
        ),
        "report": analysis._load_json(
            analysis.RUN / "mirrored_calibration_report_v73b.json"
        ),
        "population": analysis._load_json(
            analysis.RUN / "mirrored_population_v73b.json"
        ),
        "update": analysis._load_json(
            analysis.RUN / "pair_difference_update_v73b.json"
        ),
        "equivalence": analysis._load_json(
            analysis.RUN / "same_live_equivalence_v73b.json"
        ),
        "control_report": analysis._load_json(
            analysis.V66D_RUN / "mirrored_calibration_report_v66d.json"
        ),
        "control_population": analysis._load_json(
            analysis.V66D_RUN / "mirrored_population_v66d.json"
        ),
        "control_update": analysis._load_json(
            analysis.V66D_RUN / "pair_difference_update_v66d.json"
        ),
    }


def _actor_host(artifacts=None):
    if artifacts is None:
        artifacts = _artifacts()
    actor = analysis.validate_actor_receipts(
        analysis.RUN / "actor_cuda_work_receipts_v73b.jsonl",
        control_path=(
            analysis.V66D_RUN / "actor_cuda_work_receipts_v66d.jsonl"
        ),
    )
    host = analysis.validate_host_telemetry(
        analysis.RUN / "host_process_samples_v73b.jsonl",
        artifacts["host_summary"],
        actor["bindings"],
    )
    return actor, host


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="ascii",
    )


def _reseal_row(row, field):
    row = copy.deepcopy(row)
    row.pop(field, None)
    row[field] = analysis.canonical_sha256(row)
    return row


def test_static_inventory_hashes_attempt_prereg_and_every_json_self_hash():
    result = analysis.validate_static_identity()
    assert result["all_json_artifact_self_hashes_valid"] is True
    assert result["failure_artifact_absent"] is True
    assert result["run_inventory"]["file_count"] == 9
    assert result["preregistration"]["file_sha256"] == (
        "9c5ce43c36e08e038ee33e86380ab6c287ae1b4bcba4c80def623daeb00f7ed9"
    )


def test_finalized_acceptance_has_exact_traffic_host_and_cleanup():
    result = analysis.analyze_finalized()
    assert result["passed"] is True
    assert result["status"] == "accepted_same_live_v71_v72_lora_gate"
    assert result["traffic"]["saved_d2h_bytes"] == 14_121_664_512
    assert result["traffic"]["saved_d2h_percent"] == pytest.approx(
        76.99477421948279
    )
    assert result["host_state_residency"]["post_install"][
        "banks_per_actor"
    ] == 1
    assert result["host_state_residency"]["update_executed"][
        "banks_per_actor"
    ] == 2
    assert result["host_state_residency"]["post_abort"][
        "banks_per_actor"
    ] == 1
    assert result["host_state_residency"]["final_quiescent"][
        "banks_per_actor"
    ] == 1
    assert result["host_process"]["maximum_major_fault_delta"] == 0
    assert result["cleanup"]["all_four_compute_process_lists_empty"] is True


def test_same_live_compilers_and_four_actor_candidate_are_exact():
    result = analysis.analyze_finalized()
    assert result["same_live"][
        "canonical_and_independent_compilers_whole_mapping_exact"
    ] is True
    assert result["same_live"]["coefficient_sha256"] == (
        "005182fc01f44066ce9728cbefcaca905b08c79cb1d59d39532bc9d154c3bc14"
    )
    assert result["same_live"]["four_actor_live_identity_consensus"] is True
    assert result["same_live"]["historical_reward_bits_required"] is False
    assert result["same_live"]["historical_reward_drift"][
        "acceptance_threshold_applied"
    ] is False


def test_actor_receipt_tamper_and_resealed_work_drift_fail_closed(tmp_path):
    rows = analysis._load_jsonl(
        analysis.RUN / "actor_cuda_work_receipts_v73b.jsonl"
    )
    rows[0]["cuda_event"]["elapsed_ms"] += 1
    path = tmp_path / "actors.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(RuntimeError, match="self hash"):
        analysis.validate_actor_receipts(path)

    rows = analysis._load_jsonl(
        analysis.RUN / "actor_cuda_work_receipts_v73b.jsonl"
    )
    rows[0]["work_id"] = "f" * 64
    rows[0] = _reseal_row(rows[0], "receipt_sha256")
    _write_jsonl(path, rows)
    with pytest.raises(RuntimeError, match="differs from V66d"):
        analysis.validate_actor_receipts(
            path,
            control_path=(
                analysis.V66D_RUN / "actor_cuda_work_receipts_v66d.jsonl"
            ),
        )


def test_host_sample_tamper_and_resealed_fault_regression_fail_closed(tmp_path):
    rows = analysis._load_jsonl(
        analysis.RUN / "host_process_samples_v73b.jsonl"
    )
    bindings = _actor_host()[0]["bindings"]
    summary = _artifacts()["host_summary"]
    rows[0]["process"]["status_bytes"]["VmRSS"] += 4096
    path = tmp_path / "host.jsonl"
    _write_jsonl(path, rows)
    with pytest.raises(RuntimeError, match="sample changed"):
        analysis.validate_host_telemetry(path, summary, bindings)

    rows = analysis._load_jsonl(
        analysis.RUN / "host_process_samples_v73b.jsonl"
    )
    rank_zero = [
        index for index, row in enumerate(rows)
        if row["binding"]["actor_rank"] == 0
    ]
    index = rank_zero[-1]
    rows[index]["process"]["minor_faults"] = 0
    rows[index] = _reseal_row(rows[index], "row_sha256")
    _write_jsonl(path, rows)
    with pytest.raises(RuntimeError, match="moved backwards"):
        analysis.validate_host_telemetry(path, summary, bindings)


def test_resealed_reward_or_compiler_receipt_drift_fails_semantics():
    artifacts = _artifacts()
    actor, host = _actor_host(artifacts)
    artifacts["population"]["signed_rewards"][0]["reward"] += 1e-9
    with pytest.raises(RuntimeError, match="population completion"):
        analysis.validate_semantics(artifacts, actor, host)

    artifacts = _artifacts()
    actor, host = _actor_host(artifacts)
    compiler = artifacts["update"]["same_live_compiler_equivalence"]
    compiler["whole_result_mapping_exact"] = False
    compiler["equivalence_sha256"] = analysis.canonical_sha256({
        key: value for key, value in compiler.items()
        if key != "equivalence_sha256"
    })
    with pytest.raises(RuntimeError, match="same-live equivalence"):
        analysis.validate_semantics(artifacts, actor, host)


def test_traffic_and_residency_drift_fail_even_when_outer_object_resealed():
    artifacts = _artifacts()
    actor, host = _actor_host(artifacts)
    artifacts["traffic"]["per_actor_expected_and_observed"][
        "base_d2h_bytes"
    ] += 1
    with pytest.raises(RuntimeError, match="traffic receipt"):
        analysis.validate_semantics(artifacts, actor, host)

    artifacts = _artifacts()
    actor, host = _actor_host(artifacts)
    artifacts["traffic"]["host_state_residency"]["post_abort"][
        "unique_owned_bank_count_per_actor"
    ] = 2
    with pytest.raises(RuntimeError, match="residency"):
        analysis.validate_semantics(artifacts, actor, host)

    artifacts = _artifacts()
    actor, host = _actor_host(artifacts)
    artifacts["traffic"]["host_state_residency"]["post_abort"][
        "receipt_sha256_by_rank"
    ][0] = "f" * 64
    with pytest.raises(RuntimeError, match="abort receipt identity"):
        analysis.validate_semantics(artifacts, actor, host)


def test_rank_local_acceptance_or_four_wave_attribution_drift_fails():
    artifacts = _artifacts()
    actor, host = _actor_host(artifacts)
    tokens = artifacts["traffic"]["population_acceptance"]["tokens"]
    tokens[-1] = tokens[0]
    with pytest.raises(RuntimeError, match="rank-local"):
        analysis.validate_semantics(artifacts, actor, host)

    artifacts = _artifacts()
    actor, host = _actor_host(artifacts)
    artifacts["report"]["gpu_waves"]["by_wave"]["2"]["3"][
        "useful_work_attributed"
    ] = False
    with pytest.raises(RuntimeError, match="four-wave"):
        analysis.validate_semantics(artifacts, actor, host)


def test_cleanup_or_final_authority_drift_fails_closed():
    report = _artifacts()["report"]
    report["cleanup"]["after"][0]["state"] = "CREATED"
    with pytest.raises(RuntimeError, match="cleanup"):
        analysis.validate_cleanup(report)

    artifacts = _artifacts()
    actor, host = _actor_host(artifacts)
    artifacts["report"]["checkpoint_snapshot_or_promotion_performed"] = True
    with pytest.raises(RuntimeError, match="no-commit"):
        analysis.validate_semantics(artifacts, actor, host)


def test_timing_scopes_do_not_claim_generation_speedup():
    result = analysis.analyze_finalized()["timing"]
    common = result["common_scope_sampled_epoch_windows"]
    assert common["candidate_materialization"][
        "observed_reduction_fraction"
    ] == pytest.approx(0.5633110374044328)
    assert common["exact_restore"]["observed_reduction_fraction"] == pytest.approx(
        0.43468903907364354
    )
    assert result["actor_cuda_generation"][
        "critical_path_sum_delta_fraction"
    ] == pytest.approx(0.04009808521235114)
    assert result["actor_cuda_generation"]["generation_speedup_observed"] is False
    assert result["interpretation"][
        "do_not_attribute_overall_gain_to_model_inference"
    ] is True


def test_task_disposition_does_not_overclose_dense_or_roofline_work():
    disposition = analysis.analyze_finalized()["task_disposition"]
    assert disposition["specialist-0j5.29"].startswith("close")
    assert disposition["specialist-0j5.21"].startswith("close")
    assert disposition["specialist-0j5.14"].startswith("keep_open")
    assert disposition["specialist-0j5.19"].startswith("keep_open")
    assert analysis.analyze_finalized()["authority"][
        "dense_fullweight_ownership_inferred_from_lora"
    ] is False


def test_persisted_outputs_are_canonical_and_current():
    value = analysis.analyze_finalized()
    assert analysis.OUTPUT.read_text(encoding="ascii") == analysis.render_json(value)
    assert analysis.MARKDOWN.read_text(encoding="ascii") == (
        analysis.render_markdown(value)
    )
    body = copy.deepcopy(json.loads(analysis.OUTPUT.read_text(encoding="ascii")))
    claimed = body.pop("content_sha256_before_self_field")
    assert claimed == analysis.canonical_sha256(body)
