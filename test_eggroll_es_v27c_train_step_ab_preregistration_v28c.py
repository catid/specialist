#!/usr/bin/env python3
"""CPU-only tests for the V28C BF16 ES train-step A/B preregistration."""

import copy

import numpy as np
import pytest

import eggroll_es_v27c_train_step_ab_preregistration_v28c as prereg


def test_v28c_is_deterministic_self_sealed_and_train_only():
    first = prereg.build_preregistration_v28c()
    second = prereg.build_preregistration_v28c()
    assert first == second
    assert prereg.validate_preregistration_v28c(first) == first
    assert first["content_sha256_before_self_field"] == prereg.canonical_sha256({
        key: value for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })
    assert first["strict_train_only"] is True
    assert first["contains_dataset_rows_questions_answers_or_document_content"] is False
    assert first["contains_validation_heldout_ood_or_benchmark_content"] is False


def test_v28c_binds_exact_v28b_authority_and_exact_v27c_table():
    value = prereg.build_preregistration_v28c()
    assert value["authorization_basis"] == {
        "path": str(prereg.POSITIVE_EVIDENCE_PATH_V28C),
        "commit": "fe96a88cc42ed7cae4e57ccb544b26942553421d",
        "file_sha256": "034b34166324359687398dd2825a0a602e444360144fc645c51e5e399e972041",
        "content_sha256": "d2601ec9636fd1df100018bc96d74adbdbc9fd2d4b1e0415cb20df683ae0326f",
        "v28b_gate_passed": True,
        "v28b_pass_authority": (
            "authorize_only_a_separately_frozen_train_only_training_recipe_A_B"
        ),
    }
    table = value["table_contract"]
    assert table["file_sha256"] == (
        "128806798a5bf8a961a5bd0bc8765c82e8b73a116e6c7411e7aeba5522667562"
    )
    assert table["content_sha256"] == (
        "a4f82f53b037f766536013bdc10c8ca1e49873603a8f44972ef8007ed406de84"
    )
    assert table["batch_keys"] == [256, 512, 1024, 2048]


def test_v28c_is_bf16_only_and_rejects_fp8_authority():
    value = prereg.build_preregistration_v28c()
    assert value["model_contract"]["dtype"] == "bfloat16"
    assert value["model_contract"]["quantization_config"] is None
    assert value["model_contract"]["fp8_or_other_quantized_model_paths_authorized"] is False
    assert value["runtime_contract"]["moe_backend"] == "triton"
    assert value["gate_and_authority"]["fp8_reuse_authorized"] is False


def test_v28c_counterbalance_and_fresh_engine_group_cardinality_are_fixed():
    value = prereg.build_preregistration_v28c()
    design = value["paired_counterbalanced_design"]
    orders = [tuple(design["pairs"][pair]["arm_order"]) for pair in design["pair_order"]]
    assert len(orders) == 12
    assert orders.count(("default_empty", "v27c_tuned")) == 6
    assert orders.count(("v27c_tuned", "default_empty")) == 6
    runtime = value["runtime_contract"]
    assert runtime["fresh_engine_group_count"] == 24
    assert runtime["engines_per_group"] == 4
    assert runtime["tensor_parallel_size_per_engine"] == 1
    assert runtime["all_four_gpus_simultaneously_active_during_every_estimator_required"] is True


def test_v28c_request_cardinality_recomputes_from_v13_mechanics():
    value = prereg.build_preregistration_v28c()
    budget = value["request_budget"]
    assert budget["requests_per_arm"] == 2 * 280 + 16 * 4 * 280 == 18_480
    assert budget["requests_per_pair"] == 36_960
    assert budget["total_generation_requests"] == 12 * 36_960 == 443_520
    train = value["frozen_train_step_contract"]
    assert train["panel_names"] == [
        "optimization_0", "optimization_1", "optimization_2",
        "train_screen_0", "train_screen_1",
    ]
    assert train["population_size"] == 32
    assert train["alpha"] == 0.0
    assert train["model_update_applied"] is False


def test_v28c_bootstrap_plan_is_fixed_paired_and_three_endpoint_corrected():
    first, first_sha = prereg.bootstrap_draw_plan_v28c()
    second, second_sha = prereg.bootstrap_draw_plan_v28c()
    assert first.shape == (50_000, 12)
    assert np.array_equal(first, second)
    assert first_sha == second_sha
    analysis = prereg.build_preregistration_v28c()["performance_analysis"]
    assert analysis["bootstrap_draw_plan_sha256"] == first_sha
    assert len(analysis["inferential_endpoints"]) == 3
    assert analysis["lower_quantile"] == pytest.approx(0.05 / 3)
    assert analysis["upper_quantile"] == pytest.approx(1.0 - 0.05 / 3)
    assert analysis["speed_point_ratio_min"] == 1.01
    assert analysis["speed_lcb_ratio_strictly_greater_than"] == 1.0
    assert analysis["memory_point_ratio_max"] == 1.01
    assert analysis["memory_ucb_ratio_max"] == 1.02


def test_v28c_pass_authority_is_narrow_and_validation_fails_closed():
    value = prereg.build_preregistration_v28c()
    authority = value["gate_and_authority"]
    assert authority["pass_authority"] == (
        "authorize_only_exact_v27c_table_in_a_separately_frozen_"
        "bf16_train_only_training_recipe"
    )
    forbidden = (
        "direct_recipe_adoption_authorized", "model_update_authorized",
        "checkpoint_write_authorized", "evaluation_authorized",
        "dataset_promotion_authorized",
        "validation_heldout_ood_or_benchmark_access_authorized",
        "nontrain_runtime_reuse_authorized", "fp8_reuse_authorized",
    )
    assert all(authority[key] is False for key in forbidden)
    changed = copy.deepcopy(value)
    changed["gate_and_authority"]["model_update_authorized"] = True
    changed["content_sha256_before_self_field"] = prereg.canonical_sha256({
        key: item for key, item in changed.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="contract changed"):
        prereg.validate_preregistration_v28c(changed)


def test_v28c_persistence_excludes_all_sensitive_or_raw_values():
    value = prereg.build_preregistration_v28c()
    persistence = value["persistence_contract"]
    assert persistence["compact_aggregate_ratios_bounds_commitments_and_booleans_only"] is True
    assert persistence["rows_prompts_answers_text_token_ids_raw_coefficients_diagnostics"] is False
    assert persistence["raw_timings_memory_samples_pids_bootstrap_draws"] is False
