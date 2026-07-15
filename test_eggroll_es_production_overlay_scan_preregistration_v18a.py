#!/usr/bin/env python3
"""Offline tests for the authoritative V18A patch preregistration."""

import copy
import json

import pytest

import eggroll_es_production_overlay_scan_preregistration_v18a as prereg_v18a


def test_v18a_preregistration_is_persisted_and_binds_frozen_artifacts():
    value = prereg_v18a.build_preregistration_v18a()
    assert json.loads(prereg_v18a.OUTPUT_PATH_V18A.read_text()) == value
    assert value["immutable_inputs"]["candidate"]["rows"] == 519
    assert value["immutable_inputs"]["candidate"]["file_sha256"] == (
        prereg_v18a.frame_v18a.CANDIDATE_SHA256_V18A
    )
    flow = value["immutable_inputs"]["flow_certificate"]
    assert flow["file_sha256"] == prereg_v18a.FLOW_CERTIFICATE_FILE_SHA256_V18A
    assert flow["content_sha256"] == (
        prereg_v18a.FLOW_CERTIFICATE_CONTENT_SHA256_V18A
    )
    assert flow["solver_status"] == 0
    assert flow["solver_mip_gap"] == 0.0
    assert flow["quota_relaxation_used"] is False


def test_v18a_preregisters_exact_patch_semantics_and_populations():
    value = prereg_v18a.build_preregistration_v18a()
    semantics = value["patch_semantics"]
    assert semantics["one_representative_per_joint_component_per_arm"] is True
    assert semantics["ambiguous_shared_url_candidate_topic"] == "technique"
    assert semantics["ambiguous_pair_action"] == "always_production_fallback"
    assert semantics[
        "paired_candidate_and_production_both_present_same_arm"
    ] is False
    assert semantics["same_arm_paired_duplicate_count"] == 0
    population = value["population"]
    assert population["eligible_patch_components"] == 225
    assert population["cumulative_patch_population_by_arm"] == {
        "production_only": 0,
        "patch_one_third": 75,
        "patch_two_thirds": 150,
        "patch_full": 225,
    }
    assert population["arm_population_denominators"] == {
        "production_only": 272,
        "patch_one_third": 280,
        "patch_two_thirds": 287,
        "patch_full": 295,
    }
    assert value["panels"]["arm_requests_per_panel"] == {
        "production_only": 52,
        "patch_one_third": 53,
        "patch_two_thirds": 54,
        "patch_full": 55,
    }


def test_v18a_preregisters_subset_ht_crn_bootstrap_and_36_familywise_gates():
    value = prereg_v18a.build_preregistration_v18a()
    scoring = value["scoring"]
    assert scoring["objective_change_allowed_in_v18a"] is False
    assert "objective_change_allowed_in_v17a" not in scoring
    token_lengths = scoring["token_length_audit"]
    assert set(token_lengths["tokenizer_boundary_mismatch_count"]) == {
        "candidate_v298", "production",
    }
    assert token_lengths[
        "combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max"
    ]["candidate_v298"] == [67, 91, 104, 129, 144]
    assert token_lengths["answer_token_quantiles_p50_p90_p95_p99_max"][
        "candidate_v298"
    ] == [19, 42, 52, 74, 86]
    estimator = value["estimator"]
    assert estimator["candidate_ht_targets_only_active_sealed_subset"] is True
    assert estimator["candidate_ht_never_upweights_to_all_226"] is True
    assert estimator["plain_request_mean_used"] is False
    assert estimator["shared_component_counted_once_per_arm"] is True
    orders = value["common_random_numbers"]["signed_wave_arm_orders"]
    assert len(orders) == 16
    for sign in ("plus", "minus"):
        signed = [item for item in orders if item["sign"] == sign]
        for arm in prereg_v18a.ARM_ORDER_V18A:
            assert sorted(item["arm_order"].index(arm) for item in signed) == [
                0, 0, 1, 1, 2, 2, 3, 3,
            ]
    analysis = value["analysis"]
    assert analysis["hypothesis_count"] == 36
    assert len(analysis["endpoint_contract"]) == 36
    bootstrap = analysis["bootstrap"]
    assert bootstrap["repetitions"] == 50_000
    assert bootstrap["one_sided_quantile"] == 0.05 / 36
    assert bootstrap["fixed_panel_identities_every_replicate"] is True
    assert bootstrap["whole_panel_block_resampling_used"] is False
    assert "q1" in bootstrap["candidate_only_resampling"]
    assert "same_role_panels" in bootstrap["candidate_only_resampling"]
    assert bootstrap[
        "persist_per_unit_scores_or_bootstrap_draws"
    ] is False

    panels = value["panels"]
    assert panels[
        "screens_scored_every_direction_but_excluded_from_optimization_or_update"
    ] is True
    assert "screens_excluded_from_every_direction" not in panels

    runtime = value["runtime_estimate"]
    assert runtime["requests_per_engine_per_arm_per_signed_wave"] == {
        "production_only": 260,
        "patch_one_third": 265,
        "patch_two_thirds": 270,
        "patch_full": 275,
    }
    assert runtime["requests_per_engine_per_signed_wave_all_arms"] == 1_070
    assert runtime["v17a_requests_per_engine_per_signed_wave"] == 380
    assert runtime["relative_prompt_request_count_vs_v17a"] == 1_070 / 380
    assert runtime["estimated_wall_minutes_four_gpus"] == [45, 90]
    assert runtime["estimated_aggregate_gpu_hours"] == [3.0, 6.0]


def test_v18a_freezes_v13_middle_late_default_triton_and_no_authority():
    value = prereg_v18a.build_preregistration_v18a()
    recipe = value["frozen_recipe"]
    assert recipe["layers"] == [20, 21, 22, 23]
    assert recipe["sigma"] == 0.0003
    assert recipe["alpha"] == 0.0
    assert recipe["population_size"] == 32
    assert recipe["moe_backend"]["moe_backend"] == "default_triton"
    assert set(recipe["moe_backend"]["override_environment"].values()) == {None}
    assert recipe["model_update_allowed"] is False
    assert recipe["checkpoint_write_allowed"] is False
    assert value["promotion_gate"]["dataset_promotion_authorized"] is False
    assert value["promotion_gate"]["model_update_authorized"] is False
    assert value["promotion_gate"]["evaluation_authorized"] is False
    assert value["required_next_artifacts"][
        "runtime_launch_authorized_by_this_preregistration"
    ] is False
    assert value["firewall"]["gpu_launch_allowed"] is False


def test_v18a_preregistration_rejects_resealed_authorization_tampering():
    value = prereg_v18a.build_preregistration_v18a()
    tampered = copy.deepcopy(value)
    tampered["firewall"]["gpu_launch_allowed"] = True
    tampered["content_sha256_before_self_field"] = prereg_v18a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="preregistration changed"):
        prereg_v18a.validate_preregistration_v18a(tampered)
