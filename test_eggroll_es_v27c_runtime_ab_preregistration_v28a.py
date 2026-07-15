#!/usr/bin/env python3
"""CPU-only tests for the V28A V27C task-runtime preregistration."""

import copy

import numpy as np
import pytest

import eggroll_es_v27c_runtime_ab_preregistration_v28a as prereg


def test_v28a_preregistration_is_deterministic_self_sealed_and_train_only():
    first = prereg.build_preregistration_v28a()
    second = prereg.build_preregistration_v28a()
    assert first == second
    assert prereg.validate_preregistration_v28a(first) == first
    assert first["content_sha256_before_self_field"] == prereg.canonical_sha256({
        key: value for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })
    assert first["strict_train_only"] is True
    assert first["contains_dataset_rows_questions_answers_or_document_content"] is False
    assert first["contains_validation_heldout_ood_or_benchmark_content"] is False
    assert first["train_request_contract"][
        "contains_validation_heldout_ood_or_benchmark_content"
    ] is False


def test_v28a_positive_evidence_model_and_exact_committed_table_are_bound():
    value = prereg.build_preregistration_v28a()
    assert value["positive_evidence_commit"] == (
        "76e3b10730e5a7be9c5f2c298bc75095abf5a9c8"
    )
    assert value["authorization_basis"][
        "authorizes_only_this_separate_train_only_runtime_ab_preregistration"
    ] is True
    assert value["model_contract"]["weight_shard_manifest"] == (
        prereg.MODEL_WEIGHT_SHARD_MANIFEST_V28A
    )
    assert value["model_contract"]["metadata_file_sha256"] == (
        prereg.MODEL_METADATA_SHA256_V28A
    )
    seal = value["model_contract"]["committed_full_shard_model_seal"]
    assert seal["commit"] == prereg.MODEL_SEAL_COMMIT_V28A
    assert seal["content_sha256"] == prereg.MODEL_SEAL_CONTENT_SHA256_V28A
    assert seal["base_all_files_fingerprint_sha256"] == (
        prereg.MODEL_BASE_ALL_FILES_FINGERPRINT_SHA256_V28A
    )
    assert len(seal["base_shards"]) == 26
    table = value["tuned_table_contract"]
    assert table["commit"] == prereg.TUNED_TABLE_COMMIT_V28A
    assert table["file_sha256"] == prereg.TUNED_TABLE_FILE_SHA256_V28A
    assert table["content_sha256"] == prereg.TUNED_TABLE_CONTENT_SHA256_V28A
    assert table["directory_contains_exactly_one_file"] is True


def test_v28a_two_wave_assignment_is_counterbalanced_on_every_physical_gpu():
    value = prereg.build_preregistration_v28a()
    assert tuple(value["waves"]) == ("wave_a", "wave_b")
    for gpu_id in prereg.PHYSICAL_GPU_IDS_V28A:
        observed = {
            value["waves"][wave]["gpu_assignments"][str(gpu_id)]["arm"]
            for wave in prereg.WAVE_ORDER_V28A
        }
        assert observed == {"default_empty", "v27c_tuned"}
    assert value["pairing"]["default_first_physical_gpu_ids"] == [0, 2]
    assert value["pairing"]["tuned_first_physical_gpu_ids"] == [1, 3]
    runtime = value["runtime_contract"]
    assert runtime[
        "placement_groups_discovered_by_lightweight_string_int_gpu_probe"
    ] is True
    assert runtime["placement_group_creation_order_never_defines_physical_gpu"] is True
    assert runtime[
        "nvml_index_uuid_pci_and_total_memory_bound_across_both_waves"
    ] is True
    assert runtime["runtime_versions"] == {
        "ray": "2.56.0", "torch": "2.11.0+cu130", "vllm": "0.25.0",
    }
    assert runtime["max_model_len"] == 2048
    assert runtime["enable_prefix_caching"] is False
    assert runtime["total_generation_requests_all_engines_all_waves"] == 29_120
    assert runtime["interwave_cleanup_poll_timeout_seconds"] == 30.0
    assert runtime["interwave_cleanup_poll_interval_seconds"] == 0.5
    assert runtime["interwave_poll_preserves_exact_physical_gpu_identity"] is True
    assert runtime["final_cleanup_poll_timeout_seconds"] == 30.0
    assert runtime["final_cleanup_poll_interval_seconds"] == 0.5
    assert runtime["final_poll_preserves_exact_physical_gpu_identity"] is True
    assert runtime[
        "successful_report_requires_all_four_gpus_idle_after_wave_b_cleanup"
    ] is True


def test_v28a_equivalence_is_exact_and_performance_thresholds_are_preselected():
    value = prereg.build_preregistration_v28a()
    equivalence = value["exact_output_equivalence"]
    assert all(item is True for item in equivalence.values())
    performance = value["performance_analysis"]
    assert performance["per_gpu_familywise_lower_quantile"] == pytest.approx(0.0125)
    assert performance["per_gpu_familywise_upper_quantile"] == pytest.approx(0.9875)
    assert performance["per_gpu_throughput_lcb_nonregression_ratio"] == 0.98
    assert performance["global_observed_median_throughput_improvement_ratio"] == 1.01
    assert performance["per_gpu_peak_vram_ratio_familywise_ucb_max"] == 1.02
    assert performance["threshold_justification"]["selected_before_outputs"] is True


def test_v28a_bootstrap_draw_plan_is_deterministic_and_hierarchical():
    repetitions_a, gpus_a, sha_a = prereg.bootstrap_draw_plan_v28a()
    repetitions_b, gpus_b, sha_b = prereg.bootstrap_draw_plan_v28a()
    assert repetitions_a.shape == (20_000, 9)
    assert gpus_a.shape == (20_000, 4)
    assert np.array_equal(repetitions_a, repetitions_b)
    assert np.array_equal(gpus_a, gpus_b)
    assert sha_a == sha_b
    assert sha_a == prereg.build_preregistration_v28a()[
        "performance_analysis"
    ]["bootstrap_draw_plan_sha256"]


def test_v28a_gate_authority_is_narrow_and_validation_fails_closed():
    value = prereg.build_preregistration_v28a()
    gate = value["gate"]
    assert gate["pass_authority"] == (
        "authorize_only_a_separately_frozen_train_only_training_recipe_A_B"
    )
    for key in (
        "direct_recipe_adoption_authorized", "model_update_authorized",
        "checkpoint_write_authorized", "evaluation_authorized",
        "dataset_promotion_authorized",
    ):
        assert gate[key] is False
    changed = copy.deepcopy(value)
    changed["gate"]["direct_recipe_adoption_authorized"] = True
    changed["content_sha256_before_self_field"] = prereg.canonical_sha256({
        key: item for key, item in changed.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.validate_preregistration_v28a(changed)
