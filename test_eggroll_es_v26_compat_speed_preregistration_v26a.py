#!/usr/bin/env python3
"""CPU-only tests for the future V26A compatibility/speed preregistration."""

import json

import numpy as np
import pytest

import eggroll_es_v26_compat_speed_preregistration_v26a as prereg


def test_v26a_preregistration_is_deterministic_train_only_and_authority_narrow():
    first = prereg.build_preregistration_v26a()
    second = prereg.build_preregistration_v26a()
    assert first == second
    assert prereg.validate_preregistration_v26a(first) == first
    assert first["strict_train_only"] is True
    assert first["contains_validation_ood_heldout_or_benchmark_content"] is False
    assert first["train_request_contract"][
        "contains_validation_ood_heldout_or_benchmark_content"
    ] is False
    assert first["content_sha256_before_self_field"] == prereg.canonical_sha256({
        key: value
        for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })
    gate = first["gate"]
    assert gate["pass_authority"] == (
        "authorize_only_a_separate_fresh_preregistered_train_only_training_A_B"
    )
    for key in (
        "direct_model_adoption_authorized", "model_update_authorized",
        "checkpoint_write_authorized", "evaluation_authorized",
        "dataset_promotion_authorized",
    ):
        assert gate[key] is False
    assert first["required_runtime_adapter"][
        "gpu_idle_check_after_cpu_disk_audits_immediately_before_attempt_claim"
    ] is True


def test_v26a_model_and_two_wave_counterbalanced_mapping_are_exact():
    value = prereg.build_preregistration_v26a()
    models = value["model_contract"]
    assert models["fp8"]["weight_shard_manifest"] == (
        prereg.FP8_WEIGHT_SHARD_MANIFEST_V26A
    )
    assert models["hybrid_v26"]["weight_shard_manifest"] == (
        prereg.HYBRID_WEIGHT_SHARD_MANIFEST_V26A
    )
    assert models["hybrid_v26"]["target_key_count"] == 63_939
    assert models["hybrid_v26"]["only_routed_experts_remain_fp8"] is True
    assert models["hybrid_v26"]["all_non_routed_pathways_exact_bf16"] is True
    assert tuple(value["waves"]) == prereg.WAVE_ORDER_V26A
    assert tuple(
        value["waves"]["wave_a"]["gpu_assignments"][str(gpu_id)]["backend"]
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A
    ) == (
        "full_fp8", "fp8_routed_bf16_backbone_v26",
        "full_fp8", "fp8_routed_bf16_backbone_v26",
    )
    assert tuple(
        value["waves"]["wave_b"]["gpu_assignments"][str(gpu_id)]["backend"]
        for gpu_id in prereg.PHYSICAL_GPU_IDS_V26A
    ) == (
        "fp8_routed_bf16_backbone_v26", "full_fp8",
        "fp8_routed_bf16_backbone_v26", "full_fp8",
    )
    assert value["pairing"]["fp8_first_physical_gpu_ids"] == [0, 2]
    assert value["pairing"]["hybrid_first_physical_gpu_ids"] == [1, 3]
    assert tuple(value["pairing"]["physical_gpu_pairs"]) == (
        "physical_gpu_0", "physical_gpu_1", "physical_gpu_2", "physical_gpu_3",
    )
    assert value["runtime_contract"][
        "one_actor_and_one_complete_model_replica_per_physical_gpu_per_wave"
    ] is True
    assert value["runtime_contract"][
        "fresh_ray_actors_engines_and_model_loads_between_waves"
    ] is True
    assert value["runtime_contract"][
        "physical_gpu_identity_bound_by_nvml_index_uuid_and_pci_bus_across_waves"
    ] is True
    assert value["runtime_contract"][
        "placement_groups_bound_by_lightweight_string_int_canonicalized_gpu_probe"
    ] is True
    assert value["runtime_contract"][
        "placement_group_creation_order_never_used_as_physical_gpu_identity"
    ] is True
    assert value["runtime_contract"][
        "placement_groups_are_non_detached_and_removed_after_each_wave"
    ] is True
    assert value["runtime_contract"][
        "model_load_initialization_and_teardown_time_excluded"
    ] is True
    assert value["runtime_contract"][
        "interwave_async_cleanup_idle_poll_timeout_seconds"
    ] == 30.0
    assert value["runtime_contract"][
        "interwave_async_cleanup_idle_poll_interval_seconds"
    ] == 0.5
    assert value["runtime_contract"][
        "total_generation_requests_all_engines_all_waves"
    ] == 22_400


def test_v26a_tolerances_are_preregistered_task_level_and_not_posthoc():
    analysis = prereg.build_preregistration_v26a()["equivalence_analysis"]
    assert analysis["thresholds"] == prereg.EQUIVALENCE_THRESHOLDS_V26A
    assert analysis["threshold_justification"][
        "no_threshold_selected_after_outputs"
    ] is True
    assert analysis["exact_repeatability_within_each_wave_gpu_cell_required"] is True
    assert analysis[
        "same_backend_score_arrays_and_token_ids_exact_across_all_four_appearances"
    ] is True


def test_v26a_timing_bootstrap_draw_plan_is_exact_paired_and_fresh():
    first, first_sha = prereg.timing_bootstrap_draw_plan_v26a()
    second, second_sha = prereg.timing_bootstrap_draw_plan_v26a()
    assert first.shape == (20_000, 7)
    assert first.dtype == np.int64
    assert np.array_equal(first, second)
    assert first_sha == second_sha
    assert first_sha == prereg.build_preregistration_v26a()[
        "speed_and_memory_analysis"
    ]["paired_bootstrap_draw_plan_sha256"]
    assert np.all((first >= 0) & (first < 7))


def test_v26a_speed_gate_is_per_physical_gpu_and_exposes_order_effects():
    value = prereg.build_preregistration_v26a()
    analysis = value["speed_and_memory_analysis"]
    assert analysis["each_physical_gpu_speed_pair_gated_independently"] is True
    assert analysis["all_eight_wave_gpu_cells_peak_vram_gated"] is True
    assert analysis[
        "fp8_first_and_hybrid_first_order_strata_reported_separately"
    ] is True
    assert analysis["model_load_time_excluded_from_throughput"] is True
    assert prereg.SPEED_FAMILYWISE_QUANTILE_V26A == pytest.approx(0.0125)


def test_v26a_preregistration_validation_fails_closed():
    changed = json.loads(json.dumps(prereg.build_preregistration_v26a()))
    changed["gate"]["direct_model_adoption_authorized"] = True
    changed["content_sha256_before_self_field"] = prereg.canonical_sha256({
        key: value
        for key, value in changed.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.validate_preregistration_v26a(changed)
