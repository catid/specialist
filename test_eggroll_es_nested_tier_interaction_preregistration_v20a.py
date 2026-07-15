#!/usr/bin/env python3
"""Offline tests for the V20A nested tier-interaction preregistration."""

import copy
import json

import pytest

import eggroll_es_nested_tier_interaction_preregistration_v20a as prereg_v20a


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_v20a_preregistration_is_persisted_and_binds_both_negative_results():
    value = prereg_v20a.build_preregistration_v20a()
    assert json.loads(prereg_v20a.OUTPUT_PATH_V20A.read_text()) == value
    assert prereg_v20a.file_sha256(prereg_v20a.OUTPUT_PATH_V20A) == (
        "1a7f372bf6f2af6606acc4e6d4adbf9815b96931a532aa07348335a1416d6963"
    )
    assert value["content_sha256_before_self_field"] == (
        "9ce64a2d9cc91da2dd83aeb0d1e5adf3c0a3216e69613e6a61683c001909345e"
    )
    v18 = value["immutable_inputs"]["v18a_negative_evidence"]
    v19 = value["immutable_inputs"]["v19a_negative_evidence"]
    assert v18["file_sha256"] == (
        "135bfc1b3bbfa4a6f8605f758e2b84005bf574f525291750ba635db75cc4eb1b"
    )
    assert v19["commit"] == "f3a09393b6d13754310eeb2ffc131791c557fde3"
    assert v19["file_sha256"] == (
        "e91ae9f14d87ee97ce34fd7e5dcfaa27a0f1ef65abeba0582e0bf0cb2c8f5267"
    )
    assert v19["content_sha256"] == (
        "84895cba0c6b79adc1b6ae59680632694bc1e6cb29bf90211cdf8c7f3665040f"
    )
    assert v18["preregistered_gate_passed"] is False
    assert v19["preregistered_gate_passed"] is False


def test_v20a_freezes_new_basis_distinct_from_v18_and_v19():
    basis = prereg_v20a.validate_perturbation_basis_v20a()
    assert basis["basis_seed"] == 20260731
    assert len(basis["seeds"]) == len(set(basis["seeds"])) == 32
    assert prereg_v20a.canonical_sha256(basis["seeds"]) == (
        prereg_v20a.PERTURBATION_SEED_LIST_SHA256_V20A
    )
    assert prereg_v20a.canonical_sha256(basis) == (
        prereg_v20a.PERTURBATION_BASIS_SHA256_V20A
    )
    recipe = prereg_v20a.build_preregistration_v20a()["frozen_recipe"]
    assert recipe["perturbation_basis"] == basis
    assert recipe["perturbation_basis_sha256"] not in {
        recipe["prior_v18a_basis_content_sha256"],
        recipe["prior_v19a_basis_content_sha256"],
    }
    assert recipe["prior_basis_reuse_allowed"] is False
    assert recipe["basis_generation_or_selection_at_launch_allowed"] is False


def test_v20a_preregisters_nested_arms_exact_frame_and_raw_requests():
    value = prereg_v20a.build_preregistration_v20a()
    assert {
        arm: contract["active_patch_tiers"]
        for arm, contract in value["arms"].items()
    } == {
        "production_only": [],
        "patch_tier_2_only": [2],
        "patch_tiers_2_3": [2, 3],
        "patch_all_tiers": [1, 2, 3],
    }
    assert value["population"]["arm_population_denominators"] == {
        "production_only": 272,
        "patch_tier_2_only": 279,
        "patch_tiers_2_3": 287,
        "patch_all_tiers": 295,
    }
    panels = value["panels"]
    assert len(panels["names"]) == 10
    assert len(panels["optimization"]) == 6
    assert len(panels["train_only_screens"]) == 4
    assert panels["v19_overlap_theoretical_minimum_attained"] is True
    assert panels["all_32_v19_unused_base_components_selected"] is True
    runtime = value["runtime_request_contract"]
    assert runtime["raw_arm_request_coverage_is_authoritative"] is True
    assert runtime["requests_per_engine_per_arm_per_signed_wave"] == {
        "production_only": 240,
        "patch_tier_2_only": 250,
        "patch_tiers_2_3": 260,
        "patch_all_tiers": 270,
    }
    assert runtime["requests_per_engine_per_signed_wave_all_arms"] == 1020
    assert runtime["engine_count"] == 4
    assert runtime["tp_per_engine"] == 1


def test_v20a_preregisters_five_nested_contrasts_and_60_familywise_endpoints():
    value = prereg_v20a.build_preregistration_v20a()
    orders = value["common_random_numbers"]["signed_wave_arm_orders"]
    assert len(orders) == 16
    for sign in ("plus", "minus"):
        signed = [entry for entry in orders if entry["sign"] == sign]
        for arm in prereg_v20a.ARM_ORDER_V20A:
            assert sorted(entry["arm_order"].index(arm) for entry in signed) == [
                0, 0, 1, 1, 2, 2, 3, 3,
            ]
    analysis = value["analysis"]
    assert set(analysis["contrasts"]) == {
        "tier2_vs_production",
        "tiers2_3_vs_production",
        "all_tiers_vs_production",
        "conditional_tier3_after_tier2",
        "conditional_tier1_after_tiers2_3",
    }
    assert analysis["hypothesis_count"] == 60
    assert len(analysis["endpoint_contract"]) == 60
    bootstrap = analysis["bootstrap"]
    assert bootstrap["repetitions"] == 50_000
    assert bootstrap["one_sided_quantile"] == 0.05 / 60
    assert bootstrap["fixed_panel_identities_every_replicate"] is True
    assert bootstrap["whole_panel_block_resampling_used"] is False
    assert "shared_by_every_arm" in bootstrap["candidate_only_resampling"]
    assert bootstrap["persist_per_unit_scores_or_bootstrap_draws"] is False


def test_v20a_union_planner_is_bound_but_not_authorized_without_live_gate():
    value = prereg_v20a.build_preregistration_v20a()
    planner = value["immutable_inputs"]["proposed_union_request_planner"]
    assert planner["commit"] == "cc342be6374b1cb57479119c0789ad67f19467e0"
    assert planner["v19_raw_request_proof_count"] == 990
    assert planner["v19_global_unique_request_proof_count"] == 440
    assert planner["v20_exact_unique_request_count_frozen_here"] is False
    assert planner["changes_statistical_estimand"] is False
    assert planner["execution_authorized"] is False
    union = value["runtime_request_contract"]["proposed_token_hash_union_scorer"]
    assert union["exact_unique_request_count_frozen_here"] is False
    assert union["reference_wave_bit_exact_old_vs_union_required"] is True
    assert union["perturbed_signed_wave_bit_exact_old_vs_union_required"] is True
    assert union["all_per_arm_scores_and_dense_commitments_must_match"] is True
    assert union["execution_authorized_by_this_preregistration"] is False
    assert union["failure_action"] == "retain_raw_arm_request_execution"


def test_v20a_is_train_only_alpha_zero_and_opens_no_authority():
    value = prereg_v20a.build_preregistration_v20a()
    recipe = value["frozen_recipe"]
    assert recipe["model"].endswith("Qwen3.6-35B-A3B")
    assert recipe["layers"] == [20, 21, 22, 23]
    assert recipe["sigma"] == 0.0003
    assert recipe["alpha"] == 0.0
    assert recipe["model_update_allowed"] is False
    assert recipe["checkpoint_write_allowed"] is False
    assert recipe["evaluation_surfaces_opened"] is False
    assert recipe["dataset_promotion_allowed"] is False
    assert value["scoring"]["objective_change_allowed_in_v20a"] is False
    token = value["immutable_inputs"]["token_length_audit"]
    assert token["over_frozen_1024_total_token_cap_count"] == 0
    assert token["observed_combined_token_max"] == 144
    gate = value["attribution_gate"]
    assert gate["this_result_can_directly_promote_dataset"] is False
    assert gate["this_result_can_directly_update_model"] is False
    assert gate["this_result_can_directly_open_evaluation"] is False
    assert value["firewall"]["gpu_launch_allowed"] is False
    assert value["firewall"][
        "heldout_validation_ood_eval_or_benchmark_content_opened"
    ] is False
    keys = {str(key).lower() for key in _all_keys(value)}
    assert not keys & {"question", "answer", "prompt", "response", "text", "items"}


def test_v20a_preregistration_is_exclusive_and_fail_closed(tmp_path, monkeypatch):
    value = prereg_v20a.build_preregistration_v20a()
    tampered = copy.deepcopy(value)
    tampered["attribution_gate"]["this_result_can_directly_update_model"] = True
    tampered["content_sha256_before_self_field"] = prereg_v20a.canonical_sha256({
        key: item
        for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="preregistration changed"):
        prereg_v20a.validate_preregistration_v20a(tampered)

    output = (tmp_path / "immutable.json").resolve()
    monkeypatch.setattr(prereg_v20a, "OUTPUT_PATH_V20A", output)
    prereg_v20a._exclusive_write(output, value)
    with pytest.raises(RuntimeError, match="already exists"):
        prereg_v20a._exclusive_write(output, value)
