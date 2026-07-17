from __future__ import annotations

import copy
import json

import pytest

import run_qwen36_adapter_transport_paired_v81a as runner


def _seal(value):
    result = dict(value)
    result["content_sha256_before_self_field"] = (
        runner.canonical_sha256_v81a(result)
    )
    return result


def _preregistration(tmp_path, monkeypatch):
    value = _seal({
        "schema": "qwen36-adapter-transport-cuda-preregistration-v81a",
        "status": "sealed_cpu_cuda_integration_before_live_gate",
        "authority": {
            "gpu_launch": False,
            "quality_or_hpo_promotion": False,
        },
    })
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, sort_keys=True), encoding="ascii")
    monkeypatch.setattr(runner, "PREREGISTRATION", path)
    return value, path


def _gate(tmp_path, preregistration, prereg_path):
    bindings = []
    for index in range(3):
        path = tmp_path / f"v73c-evidence-{index}.json"
        path.write_text(f"evidence-{index}\n", encoding="ascii")
        bindings.append({
            "path": str(path),
            "file_sha256": runner.file_sha256_v81a(path),
        })
    return _seal({
        "schema": runner.GATE_SCHEMA_V81A,
        "status": "sealed_after_v73c_phase_evidence",
        "gpu_launch_authorized": True,
        "systems_only_nonpromotable": True,
        "quality_evaluation_authorized": False,
        "hpo_or_recipe_promotion_authorized": False,
        "checkpoint_or_model_update_authorized": False,
        "protected_content_access_authorized": False,
        "quarantined_v1_resolved": False,
        "v2_bound_successor_required_for_quality": True,
        "preregistration": {
            "path": str(prereg_path),
            "file_sha256": runner.file_sha256_v81a(prereg_path),
            "content_sha256": preregistration[
                "content_sha256_before_self_field"
            ],
        },
        "source_commit": "a" * 40,
        "v73c_phase_evidence": {
            "bead": "specialist-0j5.32",
            "complete": True,
            "timeline_profile_passed": True,
            "all_four_physical_gpus": [0, 1, 2, 3],
            "all_exact_phases_nonoverlapping": True,
            "materialize_restore_update_abort_cleanup_covered": True,
            "final_idle": True,
            "protected_content_opened": False,
            "profiled_timing_used_as_speed_claim": False,
            "bindings": bindings,
        },
        "pair_schedule": [list(item) for item in runner.PAIR_ORDER_V81A],
        "physical_gpu_ids": [0, 1, 2, 3],
    })


def _arm_receipt(row, gate_identity, *, deterministic_suffix=""):
    challenger = row["arm"] == runner.CHALLENGER_ARM_V81A
    elapsed = 80 if challenger else 100
    hbm = 70.0 if challenger else 100.0
    exact = {
        "v71_exact_audits_passed": True,
        "v72_one_two_one_ownership_passed": True,
        "same_live_dual_compiler_exact": True,
        "reward_values_cross_run_bitwise_oracle": False,
        "update_aborted_without_commit": True,
        "restore_or_poison_passed": True,
        "final_master_exact": True,
        "final_runtime_exact": True,
        "population_plan_sha256": "1" * 64,
        "candidate_projection_set_sha256": (
            deterministic_suffix or "2" * 64
        ),
        "candidate_audit_set_sha256": (
            "3" if challenger else "4"
        ) * 64,
        "same_live_reward_sha256": (
            "5" if challenger else "6"
        ) * 64,
        "same_live_update_sha256": (
            "7" if challenger else "8"
        ) * 64,
        "final_master_sha256": "9" * 64,
        "final_runtime_sha256": "a" * 64,
    }
    return _seal({
        "schema": runner.ARM_RECEIPT_SCHEMA_V81A,
        "pair_index": row["pair_index"],
        "pair_id": row["pair_id"],
        "order_position": row["order_position"],
        "arm": row["arm"],
        "gate_content_sha256": gate_identity,
        "physical_gpu_ids": [0, 1, 2, 3],
        "systems_only_nonpromotable": True,
        "protected_content_opened": False,
        "checkpoint_model_update_or_promotion_performed": False,
        "exact_lifecycle": exact,
        "transport": {
            "transition_count": 2,
            "runtime_view_count": 82,
            "h2d_bytes_per_transition": 9_842_688,
            "h2d_calls_per_transition": 82,
            "maximum_view_bytes": 524_288,
            "temporary_device_publication_staging_bytes_per_transition": (
                0 if challenger else 524_288
            ),
            "d2d_bytes_per_transition": 0 if challenger else 9_842_688,
            "pinned_host_bank_count": 1 if challenger else 0,
            "pinned_host_bank_bytes": 9_842_688 if challenger else 0,
            "actual_bank_is_pinned": challenger,
            "event_fenced_before_audit_and_generation": challenger,
        },
        "measurements": {
            "phase_source": "sealed_v73c_method",
            "all_four_useful_gpu_attribution": True,
            "profiler_overhead_used_as_speed_claim": False,
            "by_gpu": {
                str(gpu): {
                    "physical_gpu_id": gpu,
                    "actor_pid": 1000 + gpu,
                    "transition_elapsed_ns": [elapsed, elapsed],
                    "peak_allocated_vram_bytes": 1_000_000,
                    "peak_reserved_vram_bytes": 2_000_000,
                    "pcie_rx_bytes": 9_842_688,
                    "pcie_tx_bytes": 9_842_688,
                    "hbm_activity": {
                        "metric_kind": "tool_defined_bandwidth_sample",
                        "value": hbm,
                        "units": "tool_defined_units",
                        "exact_bytes": False,
                    },
                    "useful_cuda_event_elapsed_ns": 50,
                }
                for gpu in range(4)
            },
        },
        "cleanup": {
            "all_four_actor_final_idle": True,
            "all_four_gpu_process_lists_empty": True,
            "ray_shutdown": True,
            "pinned_banks_released": challenger,
            "failure": None,
        },
    })


def test_gate_requires_exact_v73c_phase_evidence_and_systems_only_authority(
    tmp_path, monkeypatch,
):
    prereg, path = _preregistration(tmp_path, monkeypatch)
    gate = _gate(tmp_path, prereg, path)
    validated = runner.validate_launch_gate_v81a(gate, prereg)
    assert validated["gate_content_sha256"] == gate[
        "content_sha256_before_self_field"
    ]
    assert len(validated["v73c_bound_files"]) == 3
    assert validated["systems_only_nonpromotable"] is True
    assert validated["quarantined_v1_resolved"] is False

    changed = copy.deepcopy(gate)
    changed["quality_evaluation_authorized"] = True
    changed = _seal({
        key: value for key, value in changed.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="widened"):
        runner.validate_launch_gate_v81a(changed, prereg)


@pytest.mark.parametrize("surface", runner.FORBIDDEN_V1_SURFACES_V81A)
def test_quarantined_v1_is_rejected_recursively(surface):
    with pytest.raises(RuntimeError, match="quarantined V1"):
        runner.reject_quarantined_v1_surface_v81a({
            "nested": [{"path": f"/home/catid/specialist/{surface}"}],
        })


def test_schedule_has_four_exactly_counterbalanced_four_gpu_pairs(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "PAIR_ROOT", tmp_path / "pairs")
    rows = runner.prospective_schedule_v81a()
    assert len(rows) == 8
    assert runner.PAIR_ORDER_V81A == (
        (runner.CONTROL_ARM_V81A, runner.CHALLENGER_ARM_V81A),
        (runner.CHALLENGER_ARM_V81A, runner.CONTROL_ARM_V81A),
        (runner.CONTROL_ARM_V81A, runner.CHALLENGER_ARM_V81A),
        (runner.CHALLENGER_ARM_V81A, runner.CONTROL_ARM_V81A),
    )
    assert {row["pair_index"] for row in rows} == {0, 1, 2, 3}
    assert all(row["physical_gpu_ids"] == [0, 1, 2, 3] for row in rows)
    assert len({row["run_directory"] for row in rows}) == 8


def test_arm_receipt_requires_exact_bytes_events_four_gpus_and_cleanup(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(runner, "PAIR_ROOT", tmp_path / "pairs")
    row = runner.prospective_schedule_v81a()[1]
    receipt = _arm_receipt(row, "b" * 64)
    assert runner.validate_arm_receipt_v81a(
        receipt, row, "b" * 64
    )["transport"]["actual_bank_is_pinned"] is True
    changed = copy.deepcopy(receipt)
    changed["transport"]["d2d_bytes_per_transition"] = 1
    changed = _seal({
        key: value for key, value in changed.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="byte/event ledger"):
        runner.validate_arm_receipt_v81a(changed, row, "b" * 64)


def test_pair_analysis_allows_arm_local_rewards_but_pairs_deterministic_state(
    monkeypatch, tmp_path,
):
    monkeypatch.setattr(runner, "PAIR_ROOT", tmp_path / "pairs")
    rows = runner.prospective_schedule_v81a()
    receipts = [_arm_receipt(row, "b" * 64) for row in rows]
    report = runner.analyze_paired_receipts_v81a(receipts, "b" * 64)
    assert report["pair_count"] == 4
    assert report["systems_metrics"][
        "control_over_challenger_transition_geomean"
    ] == pytest.approx(1.25)
    assert report["systems_metrics"][
        "control_over_challenger_hbm_activity_geomean"
    ] == pytest.approx(100 / 70)
    assert report["systems_metrics"]["systems_candidate_pass"] is True
    assert all(
        item["cross_run_reward_float_equality_required"] is False
        for item in report["pairs"]
    )
    assert report["decision_authority"][
        "direct_worker_or_recipe_promotion_authorized"
    ] is False
    assert report["decision_authority"][
        "v2_bound_quality_successor_may_be_preregistered"
    ] is True


def test_pair_analysis_rejects_deterministic_projection_drift(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "PAIR_ROOT", tmp_path / "pairs")
    rows = runner.prospective_schedule_v81a()
    receipts = [_arm_receipt(row, "b" * 64) for row in rows]
    target = next(
        index for index, row in enumerate(rows)
        if row["pair_index"] == 0 and row["arm"] == runner.CHALLENGER_ARM_V81A
    )
    receipts[target] = _arm_receipt(
        rows[target], "b" * 64, deterministic_suffix="f" * 64
    )
    with pytest.raises(RuntimeError, match="deterministic paired"):
        runner.analyze_paired_receipts_v81a(receipts, "b" * 64)


def test_injected_executor_runs_only_after_gate_and_validates_every_arm(
    tmp_path, monkeypatch,
):
    prereg, path = _preregistration(tmp_path, monkeypatch)
    gate = _gate(tmp_path, prereg, path)
    monkeypatch.setattr(runner, "PAIR_ROOT", tmp_path / "fresh-pairs")
    seen = []

    def executor(row):
        seen.append((row["pair_index"], row["arm"]))
        return _arm_receipt(
            row, gate["content_sha256_before_self_field"]
        )

    receipts = runner.execute_prospective_schedule_v81a(
        gate, prereg, executor
    )
    assert len(receipts) == 8
    assert seen == [
        (row["pair_index"], row["arm"])
        for row in runner.prospective_schedule_v81a()
    ]


def test_current_cli_execute_is_hard_blocked_before_any_executor(
    tmp_path, monkeypatch,
):
    prereg, path = _preregistration(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="cannot execute"):
        runner.main([
            "--execute",
            "--preregistration", str(path),
            "--preregistration-file-sha256", runner.file_sha256_v81a(path),
            "--preregistration-content-sha256",
            prereg["content_sha256_before_self_field"],
        ])
