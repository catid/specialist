#!/usr/bin/env python3
"""Offline tests for the V19A disjoint-tier attribution preregistration."""

import copy
import json

import pytest

import eggroll_es_disjoint_tier_attribution_preregistration_v19a as prereg_v19a


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_v19a_preregistration_is_persisted_and_binds_v18a_negative_evidence():
    value = prereg_v19a.build_preregistration_v19a()
    assert json.loads(prereg_v19a.OUTPUT_PATH_V19A.read_text()) == value
    assert prereg_v19a.file_sha256(prereg_v19a.OUTPUT_PATH_V19A) == (
        "cd65f6127ce8d4b401d837bb23c424f0860259ab6920e25a68a4098f1798a987"
    )
    assert value["content_sha256_before_self_field"] == (
        "db2036da7c9b382a542d7c002f5af8a76bbe1101d342fad8bfdc8e3a67b6c997"
    )
    evidence = value["immutable_inputs"]["v18a_negative_evidence"]
    assert evidence["commit"] == (
        "f4fc90fe28e05d9ee4c94e98ce25419d6336ff87"
    )
    assert evidence["file_sha256"] == (
        "135bfc1b3bbfa4a6f8605f758e2b84005bf574f525291750ba635db75cc4eb1b"
    )
    assert evidence["content_sha256"] == (
        "9fc09b8125e7c945d75b5f2c60d06961a320dc8802e033838b6efa57d29f0d42"
    )
    assert evidence["source_launch_attempt_file_sha256"] == (
        "44340402296a814e1890520bc8f10e2df2364df023a30a9f3b9e68c15282b02d"
    )
    assert evidence["source_compact_report_file_sha256"] == (
        "efcc2e2c2317a072937d4bc34c27ed2b22d6e33cc0ae0e7c5873aba457b4f14e"
    )


def test_v19a_freezes_exact_fresh_non_v18a_basis_now():
    basis = prereg_v19a.validate_perturbation_basis_v19a()
    assert basis["basis_seed"] == 20260727
    assert len(basis["seeds"]) == len(set(basis["seeds"])) == 32
    assert prereg_v19a.canonical_sha256(basis["seeds"]) == (
        prereg_v19a.PERTURBATION_SEED_LIST_SHA256_V19A
    )
    assert prereg_v19a.canonical_sha256(basis) == (
        prereg_v19a.PERTURBATION_BASIS_SHA256_V19A
    )
    value = prereg_v19a.build_preregistration_v19a()
    recipe = value["frozen_recipe"]
    assert recipe["perturbation_basis"] == basis
    assert recipe["perturbation_basis_sha256"] != (
        recipe["prior_v18a_basis_content_sha256"]
    )
    assert recipe["prior_v18a_basis_reuse_allowed"] is False
    assert recipe["basis_generation_or_selection_at_launch_allowed"] is False
    required = value["required_next_artifacts"]
    assert required[
        "runtime_must_bind_exact_preregistered_basis_commitment"
    ] is True
    assert required[
        "runtime_must_bind_exact_preregistered_seed_list_commitment"
    ] is True


def test_v19a_preregisters_exact_disjoint_arms_panels_and_request_count():
    value = prereg_v19a.build_preregistration_v19a()
    assert set(value["arms"]) == set(prereg_v19a.ARM_ORDER_V19A)
    assert {
        arm: contract["active_patch_tier"]
        for arm, contract in value["arms"].items()
    } == {
        "production_only": None,
        "patch_tier_1_only": 1,
        "patch_tier_2_only": 2,
        "patch_tier_3_only": 3,
    }
    assert value["population"]["eligible_patch_population_per_tier"] == {
        "1": 75, "2": 75, "3": 75,
    }
    assert value["population"]["arm_population_denominators"] == {
        "production_only": 272,
        "patch_tier_1_only": 280,
        "patch_tier_2_only": 279,
        "patch_tier_3_only": 280,
    }
    panels = value["panels"]
    assert len(panels["names"]) == 10
    assert len(panels["optimization"]) == 6
    assert len(panels["train_only_screens"]) == 4
    assert panels["base_category_quota_per_panel"] == {
        category: 6 for category in prereg_v19a.frame_v19a.BASE_CATEGORIES_V19A
    }
    assert panels["production_topic_quota_per_panel"] == {
        "safety_consent": 3,
        "technique": 8,
        "equipment_material": 2,
        "resources_general": 11,
    }
    runtime = value["runtime_request_contract"]
    assert runtime["requests_per_engine_per_arm_per_signed_wave"] == {
        "production_only": 240,
        "patch_tier_1_only": 250,
        "patch_tier_2_only": 250,
        "patch_tier_3_only": 250,
    }
    assert runtime["requests_per_engine_per_signed_wave_all_arms"] == 990


def test_v19a_preregisters_same_role_bootstrap_and_36_familywise_endpoints():
    value = prereg_v19a.build_preregistration_v19a()
    orders = value["common_random_numbers"]["signed_wave_arm_orders"]
    assert len(orders) == 16
    for sign in ("plus", "minus"):
        signed = [entry for entry in orders if entry["sign"] == sign]
        for arm in prereg_v19a.ARM_ORDER_V19A:
            assert sorted(entry["arm_order"].index(arm) for entry in signed) == [
                0, 0, 1, 1, 2, 2, 3, 3,
            ]
    analysis = value["analysis"]
    assert analysis["hypothesis_count"] == 36
    assert len(analysis["endpoint_contract"]) == 36
    assert analysis["panel_endpoint_geometry"] == {
        "optimization_pairwise_pairs": 15,
        "aggregate_to_optimization_panels": 6,
        "train_only_screen_panels": 4,
        "median_and_worst_for_each_metric_family": True,
    }
    bootstrap = analysis["bootstrap"]
    assert bootstrap["repetitions"] == 50_000
    assert bootstrap["one_sided_quantile"] == 0.05 / 36
    assert bootstrap["fixed_panel_identities_every_replicate"] is True
    assert bootstrap["whole_panel_block_resampling_used"] is False
    assert "same_role_panel" in bootstrap["candidate_only_resampling"]
    assert "shared_across_all_three_patch_arms" in bootstrap[
        "candidate_only_resampling"
    ]
    assert bootstrap["persist_per_unit_scores_or_bootstrap_draws"] is False


def test_v19a_is_attribution_only_token_safe_and_opens_no_authority():
    value = prereg_v19a.build_preregistration_v19a()
    recipe = value["frozen_recipe"]
    assert recipe["layers"] == [20, 21, 22, 23]
    assert recipe["sigma"] == 0.0003
    assert recipe["alpha"] == 0.0
    assert recipe["model_update_allowed"] is False
    assert recipe["checkpoint_write_allowed"] is False
    assert recipe["evaluation_surfaces_opened"] is False
    assert value["scoring"]["objective_change_allowed_in_v19a"] is False
    token = value["immutable_inputs"]["token_length_audit"]
    assert token["over_frozen_1024_total_token_cap_count"] == 0
    assert token["observed_combined_token_max"] == 144
    gate = value["attribution_gate"]
    assert gate["this_result_can_directly_promote_dataset"] is False
    assert gate["this_result_can_directly_update_model"] is False
    assert gate["this_result_can_directly_open_evaluation"] is False
    assert "separate_fresh_basis_train_only_confirmation" in gate[
        "passing_tier_decision"
    ]
    assert value["firewall"]["gpu_launch_allowed"] is False
    assert value["firewall"][
        "heldout_validation_ood_eval_or_benchmark_content_opened"
    ] is False
    keys = {str(key).lower() for key in _all_keys(value)}
    assert not keys & {"question", "answer", "prompt", "response", "text", "items"}


def test_v19a_preregistration_is_exclusive_and_rejects_resealed_tampering(
    tmp_path, monkeypatch,
):
    value = prereg_v19a.build_preregistration_v19a()
    tampered = copy.deepcopy(value)
    tampered["attribution_gate"]["this_result_can_directly_update_model"] = True
    tampered["content_sha256_before_self_field"] = prereg_v19a.canonical_sha256({
        key: item
        for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="preregistration changed"):
        prereg_v19a.validate_preregistration_v19a(tampered)

    output = (tmp_path / "immutable.json").resolve()
    monkeypatch.setattr(prereg_v19a, "OUTPUT_PATH_V19A", output)
    prereg_v19a._exclusive_write(output, value)
    with pytest.raises(RuntimeError, match="already exists"):
        prereg_v19a._exclusive_write(output, value)
