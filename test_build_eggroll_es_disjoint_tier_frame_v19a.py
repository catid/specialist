#!/usr/bin/env python3
"""Aggregate-only tests for the V19A disjoint-tier frame."""

import copy
import json

import pytest

import build_eggroll_es_disjoint_tier_frame_v19a as frame_v19a


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_v19a_frame_is_deterministic_persisted_and_exactly_bound():
    first = frame_v19a.build_certificate_v19a()
    second = frame_v19a.build_certificate_v19a()
    persisted = json.loads(frame_v19a.OUTPUT_PATH_V19A.read_text())
    assert first == second == persisted
    assert frame_v19a.file_sha256(frame_v19a.OUTPUT_PATH_V19A) == (
        "50820c67844a7b11c92bc0bbaa9c594c683440a65cb882de244216a77dca5fed"
    )
    assert first["content_sha256_before_self_field"] == (
        "7ad195a55b1e51268dfba1cddb43f869b014bdb1ef329f5d73c8246ac6cbff58"
    )
    assert first["inputs"]["v18a_frame_certificate"] == {
        "path": str(frame_v19a.frame_v18a.OUTPUT_PATH_V18A),
        "file_sha256": frame_v19a.FRAME_CERTIFICATE_V18A_FILE_SHA256_V19A,
        "content_sha256": frame_v19a.FRAME_CERTIFICATE_V18A_CONTENT_SHA256_V19A,
    }


def test_v19a_exact_ten_panel_flow_has_no_relaxation():
    value = frame_v19a.build_certificate_v19a()
    flow = value["constrained_base_flow"]
    solver = flow["solver"]
    assert solver["success"] is True
    assert solver["status"] == 0
    assert solver["mip_gap"] == 0.0
    assert solver["variable_count"] == 2_720
    assert solver["constraint_count"] == 352
    assert solver["quota_relaxation_used"] is False
    assert solver["fallback_solver_used"] is False
    assert flow["quota_seed_or_solver_relaxation_used"] is False
    assert flow["panel_count"] == 10
    assert flow["optimization_panel_count"] == 6
    assert flow["train_only_screen_panel_count"] == 4
    assert flow["selected_base_components"] == 240
    assert flow["globally_panel_disjoint_base_components"] is True
    for panel in value["panels"]:
        assert panel["base_components"] == 24
        assert panel["base_category_counts"] == {
            category: 6 for category in frame_v19a.BASE_CATEGORIES_V19A
        }
        assert panel["production_topic_counts"] == {
            "safety_consent": 3,
            "technique": 8,
            "equipment_material": 2,
            "resources_general": 11,
        }


def test_v19a_candidate_only_assignment_exhausts_before_reuse():
    value = frame_v19a.build_certificate_v19a()
    assignment = value["candidate_only_assignment"]
    assert assignment["every_unique_item_before_deterministic_reuse"] is True
    assert assignment["total_panel_assignments"] == 30
    assert assignment["unique_candidate_only_components"] == 23
    assert assignment["reuse_assignments_after_exhaustion"] == 7
    assert {
        tier: (
            summary["population"],
            summary["unique_components_used"],
            summary["reuse_assignments_after_exhaustion"],
        )
        for tier, summary in assignment["tier_summary"].items()
    } == {
        "1": (8, 8, 2),
        "2": (7, 7, 3),
        "3": (8, 8, 2),
    }
    assert all(
        summary["all_unique_assigned_before_first_reuse"] is True
        for summary in assignment["tier_summary"].values()
    )
    frame, tiers = frame_v19a.load_frozen_frame_v19a()
    reconstructed = frame_v19a.assign_candidate_only_v19a(frame, tiers)
    for tier, population in frame_v19a.CANDIDATE_ONLY_TIER_POPULATIONS_V19A.items():
        sequence = [
            reconstructed["assignments"][panel][tier]
            for panel in frame_v19a.PANEL_NAMES_V19A
        ]
        eligible = {
            index
            for index, assigned_tier in tiers["candidate_only_layer"].items()
            if assigned_tier == tier
        }
        assert len(set(sequence[:population])) == population
        assert set(sequence[:population]) == eligible


def test_v19a_arms_activate_only_one_disjoint_tier_without_duplicates():
    value = frame_v19a.build_certificate_v19a()
    expected_sides = {
        "production_only": (24, 0),
        "patch_tier_1_only": (18, 7),
        "patch_tier_2_only": (18, 7),
        "patch_tier_3_only": (18, 7),
    }
    expected_tier = {
        "production_only": None,
        "patch_tier_1_only": 1,
        "patch_tier_2_only": 2,
        "patch_tier_3_only": 3,
    }
    assert value["arm_order"] == list(frame_v19a.ARM_ORDER_V19A)
    assert value["disjoint_patch_tiers"][
        "no_arm_activates_more_than_one_patch_tier"
    ] is True
    for panel in value["panels"]:
        for arm, contract in panel["arms"].items():
            assert contract["active_patch_tier"] == expected_tier[arm]
            assert contract["requests"] == frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm]
            assert (
                contract["production_representatives"],
                contract["candidate_representatives"],
            ) == expected_sides[arm]
            assert contract["same_arm_paired_duplicate_count"] == 0


def test_v19a_estimand_uses_exact_ht_weights_and_denominators():
    estimator = frame_v19a.build_certificate_v19a()["estimand"]
    assert estimator["arm_population_denominators"] == {
        "production_only": 272,
        "patch_tier_1_only": 280,
        "patch_tier_2_only": 279,
        "patch_tier_3_only": 280,
    }
    assert estimator["arm_requests_per_panel"] == {
        "production_only": 24,
        "patch_tier_1_only": 25,
        "patch_tier_2_only": 25,
        "patch_tier_3_only": 25,
    }
    assert {
        key: item["horvitz_thompson_weight"]
        for key, item in estimator["base_population_ht_strata"].items()
    } == {
        "paired_tier_1": 67 / 6,
        "paired_tier_2": 68 / 6,
        "paired_tier_3": 67 / 6,
        "fallback": 70 / 6,
    }
    assert {
        key: item["horvitz_thompson_weight"]
        for key, item in estimator["candidate_only_ht_strata"].items()
    } == {"1": 8.0, "2": 7.0, "3": 8.0}
    assert estimator["plain_request_mean_used"] is False
    assert estimator["same_arm_paired_duplicate_count"] == 0


def test_v19a_certificate_is_content_free_and_fail_closed(tmp_path):
    value = frame_v19a.build_certificate_v19a()
    keys = {str(key).lower() for key in _all_keys(value)}
    assert not keys & {"question", "answer", "prompt", "response", "text", "items"}
    assert value["firewall"][
        "contains_question_answer_prompt_response_or_row_content"
    ] is False
    assert value["firewall"][
        "heldout_validation_ood_eval_or_benchmark_content_opened"
    ] is False
    tampered = copy.deepcopy(value)
    tampered["constrained_base_flow"]["solver"]["quota_relaxation_used"] = True
    tampered["content_sha256_before_self_field"] = frame_v19a.canonical_sha256({
        key: item
        for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="certificate changed"):
        frame_v19a.validate_certificate_v19a(tampered)

    output = tmp_path / "immutable.json"
    frame_v19a._exclusive_write(output, value)
    with pytest.raises(RuntimeError, match="already exists"):
        frame_v19a._exclusive_write(output, value)
