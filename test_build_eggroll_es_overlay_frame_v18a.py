#!/usr/bin/env python3
"""Aggregate-only tests for the corrected V18A production patch frame."""

import copy
import json

import pytest

import build_eggroll_es_overlay_frame_v18a as frame_v18a


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_v18a_flow_certificate_reruns_deterministically_and_is_persisted():
    first = frame_v18a.build_certificate_v18a()
    second = frame_v18a.build_certificate_v18a()
    persisted = json.loads(frame_v18a.OUTPUT_PATH_V18A.read_text())
    assert first == second == persisted
    assert frame_v18a.file_sha256(frame_v18a.OUTPUT_PATH_V18A) == (
        "0887f936fd00d205fab46d490810732d16ddc2b34d96fdc507d43c20dec60f8e"
    )
    assert first["content_sha256_before_self_field"] == (
        "6a844851cc4e5a07c08b338c5e48f3b5ab58dbeb10c765570f3fa02f20c77b3b"
    )


def test_v18a_patch_frame_has_exact_safe_relationship_classes():
    value = frame_v18a.build_certificate_v18a()
    joint = value["joint_frame"]
    assert joint["joint_component_count"] == 295
    assert joint["relation_counts"] == {
        "paired": 203,
        "candidate_only": 23,
        "production_only": 69,
    }
    assert joint["match_class_counts"] == {
        "shared_document": 202,
        "shared_url_without_shared_document": 1,
        "candidate_only": 23,
        "production_only": 69,
    }
    patch = value["safe_patch_population"]
    assert patch["eligible_patch_components"] == 225
    assert patch["paired_tier_populations"] == {"1": 67, "2": 68, "3": 67}
    assert patch["candidate_only_layer_populations"] == {
        "1": 8, "2": 7, "3": 8,
    }
    assert patch["cumulative_patch_population_by_arm"] == {
        "production_only": 0,
        "patch_one_third": 75,
        "patch_two_thirds": 150,
        "patch_full": 225,
    }
    assert patch["excluded_ambiguous_candidate_topic"] == "technique"
    assert patch["excluded_ambiguous_production_topic"] == "resources_general"
    assert patch["excluded_ambiguous_always_uses"] == "production_representative"


def test_v18a_exact_flow_has_no_relaxation_and_every_panel_matches_quotas():
    value = frame_v18a.build_certificate_v18a()
    flow = value["constrained_flow"]
    assert flow["solver"]["status"] == 0
    assert flow["solver"]["success"] is True
    assert flow["solver"]["mip_gap"] == 0.0
    assert flow["solver"]["quota_relaxation_used"] is False
    assert flow["solver"]["fallback_solver_used"] is False
    assert flow["quota_seed_or_solver_relaxation_used"] is False
    assert flow["selected_base_components"] == 260
    assert flow["selected_candidate_only_components"] == 15
    expected_sides = {
        "production_only": (52, 0),
        "patch_one_third": (39, 14),
        "patch_two_thirds": (26, 28),
        "patch_full": (13, 42),
    }
    for panel in value["panels"]:
        assert panel["base_category_counts"] == {
            category: 13 for category in frame_v18a.BASE_CATEGORIES_V18A
        }
        assert panel["production_topic_counts"] == {
            "safety_consent": 7,
            "technique": 17,
            "equipment_material": 4,
            "resources_general": 24,
        }
        assert panel["candidate_only_layer_counts"] == {"1": 1, "2": 1, "3": 1}
        for arm, requests in frame_v18a.ARM_REQUESTS_PER_PANEL_V18A.items():
            contract = panel["arms"][arm]
            assert contract["requests"] == requests
            assert (
                contract["production_representatives"],
                contract["candidate_representatives"],
            ) == expected_sides[arm]
            assert contract["same_arm_paired_duplicate_count"] == 0


def test_v18a_estimand_uses_subset_specific_ht_and_exact_denominators():
    value = frame_v18a.build_certificate_v18a()
    estimator = value["estimand"]
    assert estimator["arm_population_denominators"] == {
        "production_only": 272,
        "patch_one_third": 280,
        "patch_two_thirds": 287,
        "patch_full": 295,
    }
    assert estimator["arm_requests_per_panel"] == {
        "production_only": 52,
        "patch_one_third": 53,
        "patch_two_thirds": 54,
        "patch_full": 55,
    }
    assert {
        key: item["horvitz_thompson_weight"]
        for key, item in estimator["base_population_ht_strata"].items()
    } == {
        "paired_tier_1": 67 / 13,
        "paired_tier_2": 68 / 13,
        "paired_tier_3": 67 / 13,
        "fallback": 70 / 13,
    }
    assert {
        key: item["horvitz_thompson_weight"]
        for key, item in estimator["candidate_only_ht_strata"].items()
    } == {"1": 8.0, "2": 7.0, "3": 8.0}
    assert estimator["candidate_ht_never_targets_all_226"] is True
    assert estimator["plain_request_mean_used"] is False
    assert estimator["same_arm_paired_duplicate_count"] == 0


def test_v18a_certificate_is_content_free_and_rejects_resealed_tampering():
    value = frame_v18a.build_certificate_v18a()
    keys = {key.lower() for key in _all_keys(value)}
    assert not keys & {"question", "answer", "prompt", "response", "text", "items"}
    assert value["firewall"][
        "contains_question_answer_prompt_response_or_row_content"
    ] is False
    assert value["firewall"][
        "heldout_validation_ood_eval_or_benchmark_content_opened"
    ] is False
    tampered = copy.deepcopy(value)
    tampered["constrained_flow"]["solver"]["quota_relaxation_used"] = True
    tampered["content_sha256_before_self_field"] = frame_v18a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="certificate changed"):
        frame_v18a.validate_certificate_v18a(tampered)
