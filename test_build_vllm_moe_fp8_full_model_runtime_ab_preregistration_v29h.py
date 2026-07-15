#!/usr/bin/env python3
"""CPU-only tests for the V29H serialized-FP8 full-model A/B preregistration."""

import copy

import numpy as np
import pytest

import build_vllm_moe_fp8_full_model_runtime_ab_preregistration_v29h as prereg


def test_v29h_is_deterministic_self_sealed_and_content_closed():
    first = prereg.build_preregistration_v29h()
    second = prereg.build_preregistration_v29h()
    assert first == second
    assert prereg.validate_preregistration_v29h(first) == first
    assert first["content_sha256_before_self_field"] == prereg.canonical_sha256({
        key: value for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })
    assert first["strict_synthetic_train_runtime_only"] is True
    assert first["contains_dataset_rows_questions_answers_or_document_content"] is False
    assert first["contains_validation_heldout_ood_or_benchmark_content"] is False


def test_v29h_binds_v29f_v29e_and_exact_selected_table():
    value = prereg.build_preregistration_v29h()
    basis = value["authorization_basis"]
    assert basis["v29f_evidence_commit"] == (
        "5fe27ca6eb297a10c7a26aefc51b7e3dde28aa47"
    )
    assert basis["v29f_evidence_file_sha256"] == (
        "2fe8c0a2f244ac1d2594904d21e1b28b1383274f05c341cc839b06c7940b7258"
    )
    assert basis["v29f_evidence_content_sha256"] == (
        "1d3bd12b9b447303fb4f46083567c1ae7db723dddee261fbf36a76928de3a7f9"
    )
    table = value["selected_table_contract"]
    assert table["file_sha256"] == (
        "1a4ed0f44c6d7cc788baecd073107b4634db4c769f0820d10174f61117b25618"
    )
    assert table["content_sha256"] == (
        "d4a49735ccfd094d6e5a3ee763eca99ed355a51fd11a7c835bfecf9fafeaa50d"
    )
    assert table["loaded_config_sha256"] == (
        "ebf00590ac51e66e52f5e99b933d1be72703fbbcc809cc2d585eca8d6b0c0a5d"
    )
    assert set(table["exact_configs"]) == {"256", "512", "1024", "2048"}


def test_v29h_binds_serialized_block_fp8_full_model():
    model = prereg.build_preregistration_v29h()["serialized_fp8_model_contract"]
    assert model["path"].endswith("models/Qwen3.6-35B-A3B-FP8")
    assert model["config_sha256"] == prereg.MODEL_CONFIG_SHA256_V29H
    assert model["index_sha256"] == prereg.MODEL_INDEX_SHA256_V29H
    assert model["weight_shards"]["file_count"] == 42
    assert model["all_files"]["file_count"] == 56
    assert model["quantization"] == {
        "quant_method": "fp8", "format": "e4m3",
        "activation_scheme": "dynamic", "weight_block_size": [128, 128],
        "activation_dtype": "bfloat16",
    }


def test_v29h_schedule_is_eight_pair_counterbalanced_and_fresh():
    value = prereg.build_preregistration_v29h()
    schedule = value["schedule"]
    orders = [tuple(item["arm_order"])
              for item in schedule["paired_counterbalanced_schedule"]]
    assert len(orders) == 8
    assert orders.count(("default_empty", "v29_selected_tuned")) == 4
    assert orders.count(("v29_selected_tuned", "default_empty")) == 4
    assert len(set(schedule["fixed_pair_seeds"])) == 8
    assert schedule["fresh_four_tp1_engine_group_per_arm"] is True
    assert schedule["fresh_ray_runtime_and_full_model_loads_per_arm"] is True


def test_v29h_budget_recomputes_exactly_without_dataset_requests():
    value = prereg.build_preregistration_v29h()
    budget = value["request_budget"]
    assert budget["fresh_four_engine_groups"] == 8 * 2 == 16
    assert budget["full_model_tp1_loads"] == 16 * 4 == 64
    assert budget["generation_calls_per_engine_per_group"] == 1 + 2 + 7 == 10
    assert budget["total_generation_requests"] == 16 * 4 * 10 == 640
    assert budget["total_prompt_tokens"] == 16 * 10 * (256 + 512 + 1024 + 2048)
    assert budget["total_prompt_tokens"] == 614_400
    assert budget["total_generated_tokens"] == 640
    assert budget["synchronized_activity_witnesses"] == 64
    requests = value["synthetic_request_contract"]
    assert requests["dataset_files_tokenizers_decoded_text_or_semantic_content_opened"] is False
    assert requests["raw_token_ids_or_decoded_text_persisted"] is False


def test_v29h_bootstrap_is_fixed_hierarchical_and_ten_endpoint_corrected():
    pairs_a, calls_a, sha_a = prereg.bootstrap_draw_plan_v29h()
    pairs_b, calls_b, sha_b = prereg.bootstrap_draw_plan_v29h()
    assert pairs_a.shape == (50_000, 8)
    assert calls_a.shape == (50_000, 7)
    assert np.array_equal(pairs_a, pairs_b)
    assert np.array_equal(calls_a, calls_b)
    assert sha_a == sha_b
    stats = prereg.build_preregistration_v29h()["statistical_contract"]
    assert stats["bootstrap_draw_plan_sha256"] == sha_a
    assert len(stats["endpoints"]) == 10
    assert stats["per_endpoint_one_sided_alpha"] == pytest.approx(0.005)
    assert stats["global_latency_point_ratio_min"] == 1.002
    assert stats["global_latency_lcb_ratio_min"] == 0.99
    assert stats["vram_ucb_ratio_max"] == 1.02


def test_v29h_runtime_contract_forces_triton_fp8_tp1_and_exact_source():
    runtime = prereg.build_preregistration_v29h()["runtime_contract"]
    assert runtime["moe_backend"] == "triton"
    assert runtime["quantization_from_serialized_checkpoint"] == "fp8_block_128x128"
    assert runtime["engines_per_group"] == 4
    assert runtime["tensor_parallel_size_per_engine"] == 1
    assert runtime["exact_config_source_verified_inside_every_actor"] is True
    assert runtime["simultaneous_all_four_pid_bound_positive_activity_required_per_group"] is True


def test_v29h_authority_is_narrow_and_fails_closed():
    value = prereg.build_preregistration_v29h()
    authority = value["gate_and_authority"]
    assert authority["pass_authority"] == (
        "authorize_only_exact_v29_table_in_a_separately_frozen_"
        "serialized_fp8_train_only_recipe_ab"
    )
    forbidden = (
        "direct_table_or_recipe_adoption_authorized",
        "model_update_or_training_authorized", "checkpoint_write_authorized",
        "dataset_promotion_authorized",
        "evaluation_validation_heldout_ood_or_benchmark_access_authorized",
        "nontrain_runtime_reuse_authorized", "bf16_table_reuse_authorized",
    )
    assert all(authority[key] is False for key in forbidden)
    changed = copy.deepcopy(value)
    changed["gate_and_authority"]["model_update_or_training_authorized"] = True
    changed["content_sha256_before_self_field"] = prereg.canonical_sha256({
        key: item for key, item in changed.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.validate_preregistration_v29h(changed)
