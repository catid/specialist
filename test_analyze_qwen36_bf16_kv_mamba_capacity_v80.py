import copy
import json

import pytest

import analyze_qwen36_bf16_kv_mamba_capacity_v80 as v80


def test_v80_analysis_and_report_are_current_and_self_hashed():
    value = v80.analyze_finalized_v80()
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field")
    assert claimed == v80.canonical_sha256_v80(body)
    assert v80.OUTPUT.read_text(encoding="ascii") == v80.render_json_v80(value)
    assert v80.REPORT.read_text(encoding="utf-8") == v80.render_report_v80(value)


def test_v80_static_contract_binds_preregs_sources_and_run_bundles():
    value = v80.analyze_finalized_v80()
    static = value["static_contract"]
    assert set(static["preregistrations"]) == set(v80.PREREGISTRATIONS)
    assert {row["path"]: row["sha256"] for row in static["sealed_executable_sources"]["files"]} == v80.SOURCE_SHA256
    assert value["v80_r1"]["artifact_inventory"]["bundle_sha256"] == v80.RUN_BUNDLES[v80.V80_RUN]
    assert value["controls"]["v79b_r5"]["artifact_inventory"]["bundle_sha256"] == v80.RUN_BUNDLES[v80.V79B_RUN]
    assert value["controls"]["v78c_r1"]["artifact_inventory"]["bundle_sha256"] == v80.RUN_BUNDLES[v80.V78C_RUN]


def test_v80_literal_parent_gate_fails_only_torch_process_group_clause():
    gates = v80.analyze_finalized_v80()["parent_gate_evaluation"]
    results = gates["data_free_gate_results"]
    assert results["torch_process_group_destroyed_per_actor_literal_true"] is False
    assert all(
        passed
        for name, passed in results.items()
        if name != "torch_process_group_destroyed_per_actor_literal_true"
    )
    assert gates["literal_parent_data_free_contract_passed"] is False
    assert gates["literal_failure"]["receipt_values_by_gpu"] == [False] * 4
    assert gates["literal_failure"]["external_process_cleanup_still_passed"] is True


def test_v80_capacity_runtime_and_control_comparisons_are_exact():
    value = v80.analyze_finalized_v80()
    capacity = value["capacity"]
    performance = value["performance"]
    assert capacity["v80_tokens_per_actor"] == 162669
    assert capacity["v80_full_2048_token_contexts_floor"] == 79
    assert capacity["v80_margin_over_parent_minimum_tokens"] == 877
    assert capacity["v80_minus_v79b_tokens"] == 365
    assert performance["v80_r1_actor_median_seconds"] == pytest.approx(48.40858003747417)
    assert performance["v80_ratio_to_v78c_r1"] < 1
    assert performance["v80_ratio_to_v79b_r5"] < 1
    assert performance["v80_runtime_gate_passed"] is True
    assert performance["single_replicate_timing_not_a_promotion_claim"] is True


def test_v80_vram_activity_pcie_and_external_cleanup_pass():
    value = v80.analyze_finalized_v80()
    telemetry = value["v80_r1"]["telemetry"]
    memory = value["vram_and_bandwidth"]
    assert telemetry["row_count"] == 512
    assert telemetry["complete_four_gpu_batches"] == 128
    assert telemetry["all_four_gpus_useful"] is True
    assert telemetry["trailing_external_cleanup_batches"] == [125, 126, 127]
    assert telemetry["external_cleanup_gate_passed"] is True
    assert all(row["peak_memory_used_mib"] == 49366 for row in telemetry["per_gpu"].values())
    assert all(row["peak_attributed_process_memory_mib"] == 48756 for row in telemetry["per_gpu"].values())
    assert memory["v80_peak_memory_gate_passed"] is True
    assert memory["sampled_pcie_integrals"]["rx_bytes_sum_gpus"] > 0
    assert memory["sampled_pcie_integrals"]["tx_bytes_sum_gpus"] > 0
    assert memory["hbm_bandwidth_bytes_per_second"] is None
    assert memory["hbm_bandwidth_not_inferred"] is True


def test_v80_token_hash_drift_keeps_candidate_and_reference_behavior_visible():
    value = v80.analyze_finalized_v80()
    drift78 = value["paired_token_hash_drift"]["v78c_r1_vs_v80_r1"]
    drift79 = value["paired_token_hash_drift"]["v79b_r5_vs_v80_r1"]
    assert drift78["differing_rows_total"] == 46
    assert drift78["differing_rows_by_call_sum_gpus"] == [14, 0, 0, 13, 8, 0, 0, 11]
    assert drift79["differing_rows_total"] == 429
    assert drift79["differing_rows_by_call_sum_gpus"] == [31, 84, 84, 19, 18, 84, 84, 25]
    assert value["parent_gate_evaluation"]["reference_repeat_changed_rows_by_gpu"] == {"0": 6, "1": 5, "2": 2, "3": 6}
    assert value["parent_gate_evaluation"]["candidate_repeat_exact_all_actors"] is True


def test_v80_analyzer_has_no_data_gpu_or_promotion_authority():
    value = v80.analyze_finalized_v80()
    authority = value["authority"]
    assert authority == {
        "cpu_sealed_file_analysis_only": True,
        "dataset_prompt_generated_text_or_protected_data_opened": False,
        "gpu_or_model_launch_performed_by_analyzer": False,
        "model_adapter_training_or_checkpoint_update_performed": False,
        "scored_training_checkpoint_or_runtime_promotion_authorized": False,
    }
    assert value["candidate_parent_contract_passed"] is False
    assert value["promotion_authorized"] is False
    assert value["parent_gate_evaluation"]["source_disjoint_semantic_gate_run"] is False
    assert value["parent_gate_evaluation"]["protected_one_shot_ood_gate_run"] is False


def test_v80b_is_prospective_exactly_two_and_unchanged():
    prospective = v80.analyze_finalized_v80()["prospective_v80b"]
    assert prospective["exactly_two_runs"] == [
        "v80_bf16_kv_mamba_capacity_0479_r2",
        "v80_bf16_kv_mamba_capacity_0479_r3",
    ]
    assert len(prospective["exact_commands"]) == 2
    assert prospective["parent_runtime_and_thresholds_unchanged"] is True
    assert prospective["r1_observation_disclosed"] is True
    assert prospective["promotion_forbidden"] is True


def test_v80_fails_closed_if_sealed_run_bundle_identity_changes(monkeypatch):
    changed = dict(v80.RUN_BUNDLES)
    changed[v80.V80_RUN] = "0" * 64
    monkeypatch.setattr(v80, "RUN_BUNDLES", changed)
    with pytest.raises(RuntimeError, match="sealed run bundle changed"):
        v80.analyze_finalized_v80()


def test_v80_fails_closed_if_parent_prereg_identity_changes(monkeypatch):
    changed = copy.deepcopy(v80.PREREGISTRATIONS)
    changed["qwen36_bf16_kv_mamba_capacity_matched_v80.json"]["file_sha256"] = "0" * 64
    monkeypatch.setattr(v80, "PREREGISTRATIONS", changed)
    with pytest.raises(RuntimeError, match="preregistration file changed"):
        v80.analyze_finalized_v80()


def test_v80_report_states_failure_and_exact_commands():
    report = v80.render_report_v80(v80.analyze_finalized_v80())
    assert "literal parent data-free contract therefore **fails**" in report
    assert "external cleanup evidence is independently strong" in report
    assert "v80_bf16_kv_mamba_capacity_0479_r2" in report
    assert "v80_bf16_kv_mamba_capacity_0479_r3" in report
    assert "do not tune thresholds" in report


def test_v80_json_contains_no_nonfinite_values():
    encoded = v80.render_json_v80(v80.analyze_finalized_v80())
    decoded = json.loads(encoded)
    assert decoded["schema"] == v80.SCHEMA
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
