#!/usr/bin/env python3
"""Offline tests for the V21A production-plus-v331 patch frame."""

import copy
import json

import pytest

import build_eggroll_es_production_v331_patch_frame_v21a as frame_v21a


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


@pytest.fixture(scope="module")
def rebuilt_v21a():
    return frame_v21a.build_certificate_v21a()


def test_v21a_binds_exact_commit_candidate_and_frozen_production_base(rebuilt_v21a):
    certificate = rebuilt_v21a
    candidate = certificate["inputs"]["candidate_v331"]
    assert candidate["source_commit"] == (
        "9d31e3407dd96f80011bdb2202e8aa9689b1d193"
    )
    assert candidate["file_sha256"] == (
        "29889e18fd7a8bf313021ec4e8ad18adcc1752529fba247fdbde006513f7866b"
    )
    assert candidate["ongoing_curation_used"] is False
    assert certificate["constrained_production_base"][
        "globally_panel_disjoint_base_components"
    ] is True
    assert certificate["constrained_production_base"][
        "selected_base_components"
    ] == 240


def test_v21a_joint_frame_finds_exact_candidate_only_topic_breadth(rebuilt_v21a):
    joint = rebuilt_v21a["joint_frame"]
    assert joint["joint_component_count"] == 326
    assert joint["relation_counts"] == {
        "paired": 206,
        "candidate_only": 54,
        "production_only": 66,
    }
    assert joint["candidate_only_topic_populations"] == {
        "safety_consent": 18,
        "technique": 18,
        "equipment_material": 10,
        "resources_general": 8,
    }


def test_v21a_minimal_q6_exhausts_all_unique_before_six_reuses(rebuilt_v21a):
    assignment = rebuilt_v21a["candidate_only_assignment"]
    assert assignment["quota_per_panel_by_topic"] == {
        "safety_consent": 2,
        "technique": 2,
        "equipment_material": 1,
        "resources_general": 1,
    }
    assert assignment["total_panel_assignments"] == 60
    assert assignment["unique_candidate_only_components"] == 54
    assert assignment["reuse_assignments_after_exhaustion"] == 6
    assert assignment["minimal_uniform_per_panel_quota_for_complete_topic_breadth"] is True
    assert assignment["inherited_q3_rejected_as_insufficient"] is True
    assert assignment["role_summary"] == {
        "optimization": {
            "panel_count": 6,
            "assignment_count": 36,
            "within_role_unique_component_count": 36,
            "reuse_assignment_count": 0,
            "new_global_unique_component_count": 36,
            "topic_assignment_counts": {
                "safety_consent": 12,
                "technique": 12,
                "equipment_material": 6,
                "resources_general": 6,
            },
        },
        "train_only_screen": {
            "panel_count": 4,
            "assignment_count": 24,
            "within_role_unique_component_count": 23,
            "reuse_assignment_count": 6,
            "new_global_unique_component_count": 18,
            "topic_assignment_counts": {
                "safety_consent": 8,
                "technique": 8,
                "equipment_material": 4,
                "resources_general": 4,
            },
        },
    }


def test_v21a_panels_are_exact_full_merge_not_candidate_replacement(rebuilt_v21a):
    for panel in rebuilt_v21a["panels"]:
        assert panel["base_category_counts"] == {
            category: 6 for category in frame_v21a.BASE_CATEGORIES_V21A
        }
        assert panel["production_topic_counts"] == (
            frame_v21a.PRODUCTION_TOPIC_QUOTAS_V21A
        )
        assert panel["candidate_only_topic_counts"] == (
            frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
        )
        production = panel["arms"]["production_only"]
        merged = panel["arms"]["production_plus_v331_patch"]
        assert (production["requests"], merged["requests"]) == (24, 30)
        assert production["production_representatives"] == 24
        assert merged["production_representatives"] == 24
        assert merged["candidate_only_representatives"] == 6
        assert production["paired_candidate_replacements"] == 0
        assert merged["paired_candidate_replacements"] == 0


def test_v21a_certificate_is_deterministic_compact_and_no_authority(rebuilt_v21a):
    artifact = json.loads(frame_v21a.OUTPUT_PATH_V21A.read_text())
    assert artifact == rebuilt_v21a
    assert frame_v21a.validate_certificate_v21a(artifact) == artifact
    keys = {str(key).lower() for key in _all_keys(artifact)}
    assert not keys & {
        "question", "questions", "answer", "answers", "prompt", "prompts",
        "response", "responses", "prompt_token_ids", "row_content",
    }
    assert artifact["firewall"] == {
        "contains_question_answer_prompt_response_token_or_row_content": False,
        "contains_evaluation_content": False,
        "heldout_validation_ood_eval_or_benchmark_content_opened": False,
        "depends_on_v20_runtime_result": False,
        "runtime_launch_authorized": False,
        "gpu_launch_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
    }


def test_v21a_certificate_rejects_candidate_authority_and_quota_tampering(rebuilt_v21a):
    tampered = copy.deepcopy(rebuilt_v21a)
    tampered["candidate_only_assignment"]["reuse_assignments_after_exhaustion"] = 7
    tampered["firewall"]["runtime_launch_authorized"] = True
    tampered["content_sha256_before_self_field"] = frame_v21a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="frame certificate changed"):
        frame_v21a.validate_certificate_v21a(tampered)
