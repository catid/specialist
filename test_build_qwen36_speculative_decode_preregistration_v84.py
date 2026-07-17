import copy
import json

import pytest

import build_qwen36_speculative_decode_preregistration_v84 as v84


def test_v84_generated_artifacts_are_current_and_self_hashed():
    value = v84.build_preregistration_v84()
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field")
    assert claimed == v84.canonical_sha256_v84(body)
    assert v84.OUTPUT.read_text(encoding="ascii") == v84.render_json_v84(value)
    assert v84.REPORT.read_text(encoding="utf-8") == v84.render_report_v84(value)


def test_v84_is_cpu_only_and_live_launch_is_blocked():
    authority = v84.build_preregistration_v84()["authority"]
    assert authority == {
        "cpu_source_and_metadata_inspection_only": True,
        "torch_vllm_or_ray_imported_by_builder": False,
        "gpu_or_model_launch_performed": False,
        "dataset_prompt_protected_or_ood_rows_opened": False,
        "model_adapter_checkpoint_or_site_package_modified": False,
        "live_launch_authorized": False,
        "hpo_quality_or_promotion_authorized": False,
    }


def test_v84_binds_native_qwen_mtp_ngram_and_lora_support():
    source = v84.build_preregistration_v84()["installed_support_evidence"]
    assert source["vllm_version"] == "0.25.0"
    assert source["qwen3_5_moe_mtp_class_present"] is True
    assert source["target_checkpoint_mtp_autodetection_present"] is True
    assert source["model_free_cpu_ngram_proposer_present"] is True
    assert source["punica_lora_allocates_for_speculative_sample_rows"] is True
    assert source["source_presence_is_not_live_compatibility_or_speed_evidence"] is True


def test_v84_charges_checkpoint_native_mtp_residency():
    inv = v84.build_preregistration_v84()["checkpoint_mtp_inventory"]
    assert inv["checkpoint_native_mtp_layer_count"] == 1
    assert inv["baseline_live_mtp_named_parameter_count_per_actor"] == 0
    assert inv["bf16_mtp"]["tensor_count"] == 19
    assert inv["bf16_mtp"]["logical_bytes"] == 1_689_281_536
    assert inv["serialized_fp8_mtp"][
        "tensor_count_including_quantization_metadata"
    ] == 1560
    assert inv["serialized_fp8_mtp"]["logical_bytes"] == 853_668_480
    assert inv["serialized_fp8_mtp"]["physical_file_bytes"] == 853_860_608
    assert inv[
        "incremental_live_vram_must_be_measured_not_assumed_from_checkpoint_bytes"
    ] is True


def test_v84_arms_start_model_free_then_checkpoint_native():
    arms = v84.build_preregistration_v84()["prospective_systems_arms"]
    assert [arm["arm_id"] for arm in arms] == [
        "target_only_control",
        "cpu_ngram_k4_n2to5",
        "checkpoint_mtp_k1",
        "checkpoint_mtp_k3",
    ]
    assert arms[0]["speculative_config"] is None
    assert arms[1]["incremental_draft_checkpoint_bytes"] == 0
    assert arms[1]["neural_draft_model_loaded"] is False
    assert arms[2]["checkpoint_native_mtp_layer_replays_per_step"] == 1
    assert arms[3]["checkpoint_native_mtp_layer_replays_per_step"] == 3
    assert arms[3]["source_warns_acceptance_may_fall"] is True


def test_v84_does_not_misuse_one_token_v73d_as_speculative_evidence():
    gate = v84.build_preregistration_v84()["applicability_gate"]
    assert gate["v73b_v73d_max_tokens"] == 1
    assert gate["v73b_v73d_cannot_measure_multistep_acceptance"] is True
    assert gate["production_hpo_cli_default_max_tokens"] == 32
    assert gate[
        "close_not_applicable_if_final_training_uses_one_token_teacher_forced_scoring"
    ] is True
    workload = v84.build_preregistration_v84()["common_systems_workload"]
    assert workload["source"] == "deterministic_content_free_token_id_panel"
    assert workload["max_tokens"] == workload["min_tokens"] == 32
    assert workload["no_prompt_answer_or_semantic_output_persisted"] is True


def test_v84_candidate_cache_and_restore_isolation_fail_closed():
    isolation = v84.build_preregistration_v84()["candidate_isolation"]
    assert all(isolation.values())
    assert isolation[
        "prefix_cache_disabled_or_reset_after_every_in_place_adapter_change"
    ] is True
    assert isolation[
        "draft_and_target_kv_or_mamba_state_never_cross_candidate_identity"
    ] is True
    assert isolation["silent_non_speculative_fallback_fails_closed"] is True


def test_v84_measures_bandwidth_mechanism_without_inferring_hbm_from_utilization():
    measurement = v84.build_preregistration_v84()["measurement"]
    assert measurement["all_physical_gpu_ids_exact"] == [0, 1, 2, 3]
    assert measurement["counterbalanced_pairs_per_arm_min"] == 3
    assert measurement["target_forward_pass_count"] is True
    assert measurement["target_forwards_per_accepted_output_token"] is True
    assert measurement[
        "exact_hbm_bytes_require_privileged_counter_or_equivalent_primary_measurement"
    ] is True
    assert measurement["utilization_percent_is_not_hbm_bytes"] is True
    assert measurement["profiler_run_cannot_establish_throughput"] is True
    assert measurement["unprofiled_timing_control_required"] is True


def test_v84_requires_mechanism_speed_and_later_quality_gates():
    value = v84.build_preregistration_v84()
    systems = value["systems_selection"]
    assert systems[
        "output_token_hashes_must_match_target_only_under_greedy_contract"
    ] is True
    assert systems["point_end_to_end_rollouts_per_gpu_second_must_exceed_control"]
    assert systems["accepted_draft_tokens_must_be_nonzero"]
    assert systems["target_forward_passes_per_output_token_must_fall"]
    assert systems["systems_winner_is_not_quality_or_training_promotion"]
    quality = value["later_quality_gate"]
    assert quality["source_disjoint_multi_item_qa_dev_required"]
    assert quality["one_shot_protected_ood_only_after_recipe_freeze"]
    assert quality["validation_and_ood_noninferiority_required"]
    assert quality["promotion_default"] is False


def test_v84_rejects_external_draft_and_semantic_systems_screen():
    rejected = {
        row["arm"]: row
        for row in v84.build_preregistration_v84()[
            "explicitly_rejected_in_first_screen"
        ]
    }
    assert set(rejected) == {
        "external_neural_draft_model",
        "ngram_gpu",
        "semantic_or_protected_data_during_systems_screen",
    }
    assert "unbounded model residency" in rejected["external_neural_draft_model"][
        "reason"
    ]


def test_v84_fails_closed_if_pinned_source_changes(monkeypatch):
    changed = dict(v84.PINNED_FILES)
    path, _ = changed["vllm/config/speculative.py"]
    changed["vllm/config/speculative.py"] = (path, "0" * 64)
    monkeypatch.setattr(v84, "PINNED_FILES", changed)
    with pytest.raises(RuntimeError, match="pinned source changed"):
        v84.build_preregistration_v84()


def test_v84_json_has_no_nonfinite_values():
    encoded = v84.render_json_v84(v84.build_preregistration_v84())
    assert json.loads(encoded)["schema"] == v84.SCHEMA
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
