#!/usr/bin/env python3
"""Aggregate-only tests for the V20A nested interaction frame."""

import copy
import json

import pytest

import build_eggroll_es_nested_tier_interaction_frame_v20a as frame_v20a


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_v20a_frame_is_deterministic_persisted_and_bound():
    first = frame_v20a.build_certificate_v20a()
    second = frame_v20a.build_certificate_v20a()
    persisted = json.loads(frame_v20a.OUTPUT_PATH_V20A.read_text())
    assert first == second == persisted
    assert frame_v20a.file_sha256(frame_v20a.OUTPUT_PATH_V20A) == (
        "cd2cb7134716602f1d51fe4049c95c49c33ddee429dcc6e752e2cb0b4196f444"
    )
    assert first["content_sha256_before_self_field"] == (
        "1199f2bcb4cf3c6f394c2faf2edf14cb9ab74323c47614118511d768197c8078"
    )
    assert first["inputs"]["v19a_frame_certificate"]["file_sha256"] == (
        frame_v20a.FRAME_CERTIFICATE_V19A_FILE_SHA256_V20A
    )


def test_v20a_exact_flow_attains_cardinality_overlap_lower_bound():
    value = frame_v20a.build_certificate_v20a()
    flow = value["constrained_base_flow"]
    solver = flow["solver"]
    overlap = flow["v19_overlap"]
    assert solver["success"] is True
    assert solver["status"] == 0
    assert solver["mip_gap"] == 0.0
    assert solver["variable_count"] == 2_720
    assert solver["constraint_count"] == 353
    assert solver["quota_relaxation_used"] is False
    assert solver["fallback_solver_used"] is False
    assert flow["selected_base_components"] == 240
    assert flow["globally_panel_disjoint_base_components"] is True
    assert overlap["base_population"] == 272
    assert overlap["v19_selected"] == overlap["v20_selected"] == 240
    assert overlap["cardinality_lower_bound"] == 240 + 240 - 272 == 208
    assert overlap["achieved_v19_overlap"] == 208
    assert overlap["maximum_fresh_available"] == overlap["achieved_fresh"] == 32
    assert overlap["all_v19_unused_components_selected"] is True
    assert overlap["theoretical_minimum_overlap_attained"] is True
    for panel in value["panels"]:
        assert panel["base_components"] == 24
        assert panel["base_category_counts"] == {
            category: 6 for category in frame_v20a.BASE_CATEGORIES_V20A
        }
        assert panel["production_topic_counts"] == {
            "safety_consent": 3,
            "technique": 8,
            "equipment_material": 2,
            "resources_general": 11,
        }


def test_v20a_nested_arms_preserve_base_identity_and_expected_sides():
    value = frame_v20a.build_certificate_v20a()
    expected = {
        "production_only": ([], 24, 0, 24, 272),
        "patch_tier_2_only": ([2], 18, 7, 25, 279),
        "patch_tiers_2_3": ([2, 3], 12, 14, 26, 287),
        "patch_all_tiers": ([1, 2, 3], 6, 21, 27, 295),
    }
    assert value["arm_order"] == list(frame_v20a.ARM_ORDER_V20A)
    nested = value["nested_patch_arms"]
    assert nested["tier2_shared_by_last_three_arms"] is True
    assert nested["tier3_shared_by_last_two_arms"] is True
    assert nested["tier1_added_only_after_tiers_2_3"] is True
    for panel in value["panels"]:
        base_roots = {
            arm["shared_base_joint_component_identity_root_sha256"]
            for arm in panel["arms"].values()
        }
        assert base_roots == {
            panel["selected_base_joint_component_identity_root_sha256"]
        }
        for arm, contract in panel["arms"].items():
            tiers, production, candidate, requests, denominator = expected[arm]
            assert contract["active_patch_tiers"] == tiers
            assert contract["production_representatives"] == production
            assert contract["candidate_representatives"] == candidate
            assert contract["requests"] == requests
            assert contract["population_denominator"] == denominator
            assert contract["same_arm_paired_duplicate_count"] == 0


def test_v20a_candidate_exhaustion_and_ht_denominators_are_exact():
    value = frame_v20a.build_certificate_v20a()
    assignment = value["candidate_only_assignment"]
    assert assignment["every_unique_item_before_deterministic_reuse"] is True
    assert assignment["total_panel_tier_assignments"] == 30
    assert assignment["unique_candidate_only_components"] == 23
    assert assignment["reuse_assignments_after_exhaustion"] == 7
    assert {
        tier: (
            summary["population"],
            summary["unique_components_used"],
            summary["reuse_assignments_after_exhaustion"],
        )
        for tier, summary in assignment["tier_summary"].items()
    } == {"1": (8, 8, 2), "2": (7, 7, 3), "3": (8, 8, 2)}
    estimator = value["estimand"]
    assert estimator["arm_population_denominators"] == {
        "production_only": 272,
        "patch_tier_2_only": 279,
        "patch_tiers_2_3": 287,
        "patch_all_tiers": 295,
    }
    assert estimator["arm_requests_per_panel"] == {
        "production_only": 24,
        "patch_tier_2_only": 25,
        "patch_tiers_2_3": 26,
        "patch_all_tiers": 27,
    }
    assert {
        tier: item["active_for_arms"]
        for tier, item in estimator["candidate_only_ht_strata"].items()
    } == {
        "1": ["patch_all_tiers"],
        "2": ["patch_tier_2_only", "patch_tiers_2_3", "patch_all_tiers"],
        "3": ["patch_tiers_2_3", "patch_all_tiers"],
    }
    assert estimator["plain_request_mean_used"] is False


def test_v20a_certificate_is_content_free_and_fail_closed(tmp_path):
    value = frame_v20a.build_certificate_v20a()
    keys = {str(key).lower() for key in _all_keys(value)}
    assert not keys & {"question", "answer", "prompt", "response", "text", "items"}
    assert value["firewall"][
        "contains_question_answer_prompt_response_or_row_content"
    ] is False
    assert value["firewall"][
        "heldout_validation_ood_eval_or_benchmark_content_opened"
    ] is False
    tampered = copy.deepcopy(value)
    tampered["constrained_base_flow"]["v19_overlap"][
        "theoretical_minimum_overlap_attained"
    ] = False
    tampered["content_sha256_before_self_field"] = frame_v20a.canonical_sha256({
        key: item
        for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="certificate changed"):
        frame_v20a.validate_certificate_v20a(tampered)

    output = tmp_path / "immutable.json"
    frame_v20a._exclusive_write(output, value)
    with pytest.raises(RuntimeError, match="already exists"):
        frame_v20a._exclusive_write(output, value)
