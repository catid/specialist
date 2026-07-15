#!/usr/bin/env python3
"""Offline fail-closed tests for the V22A exact-v341 replacement frame."""

import copy
import json

import pytest

import build_eggroll_es_v341_matched_replacement_frame_v22a as frame_v22a


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


@pytest.fixture(scope="module")
def rebuilt_v22a():
    return frame_v22a.build_certificate_v22a()


def test_v22a_binds_exact_v341_seal_and_frozen_v19_panel_source(rebuilt_v22a):
    candidate = rebuilt_v22a["inputs"]["candidate_v341"]
    assert candidate["source_commit"] == (
        "162e39408f4af0feee694dd4c128e9bb10dac057"
    )
    assert candidate["sealed_snapshot_commit"] == (
        "01ca5809321d00a904f5bb8fff1b3d8dd402db25"
    )
    assert candidate["rows"] == 528
    assert candidate["file_sha256"] == frame_v22a.candidate_v341.V341_SHA256
    assert candidate["ongoing_curation_used"] is False
    assert rebuilt_v22a["inputs"]["frozen_v19a_panel_source"][
        "via_v21a_frame_commit"
    ] == frame_v22a.V21A_FRAME_COMMIT_V22A


def test_v22a_records_preregistration_time_206_to_205_correction(rebuilt_v22a):
    correction = rebuilt_v22a["preregistration_time_correction"]
    assert correction["prior_assumed_relation_counts"] == {
        "paired": 206, "candidate_only": 54, "production_only": 66,
    }
    assert correction["exact_v341_relation_counts"] == {
        "paired": 205, "candidate_only": 54, "production_only": 67,
    }
    assert correction["affected_frozen_sample_panel"] == "train_screen_0"
    assert correction["prior_expected_sampled_replacements"] == 185
    assert correction["exact_v341_sampled_replacements"] == 184
    assert correction["fabricated_or_substituted_candidate_representative"] is False
    assert correction["correction_made_before_runtime_or_result_observation"] is True
    assert correction["post_result_adaptation"] is False


def test_v22a_joint_frame_and_candidate_only_exclusion_are_exact(rebuilt_v22a):
    joint = rebuilt_v22a["joint_frame"]
    assert joint["candidate_component_count"] == 259
    assert joint["production_component_count"] == 272
    assert joint["joint_component_count"] == 326
    assert joint["relation_counts"] == {
        "paired": 205, "candidate_only": 54, "production_only": 67,
    }
    assert joint["match_class_counts"] == {
        "shared_document": 203,
        "shared_url_without_shared_document": 2,
        "candidate_only": 54,
        "production_only": 67,
    }
    assert rebuilt_v22a["estimand"]["candidate_only_additions_excluded"] == 54


def test_v22a_exact_same_240_components_panels_weights_and_denominator(rebuilt_v22a):
    base = rebuilt_v22a["frozen_production_base"]
    estimand = rebuilt_v22a["estimand"]
    assert base["population_component_count"] == 272
    assert base["sampled_component_count"] == 240
    assert base["globally_panel_disjoint"] is True
    assert estimand["arm_population_denominators"] == {
        "production_control": 272,
        "v341_matched_replacement": 272,
    }
    assert estimand["arm_requests_per_panel"] == {
        "production_control": 24,
        "v341_matched_replacement": 24,
    }
    assert estimand[
        "same_components_panels_ht_weights_and_denominator_both_arms"
    ] is True
    assert estimand["base_population_ht_strata"] == {
        category: {
            "population": population,
            "per_panel_quota": 6,
            "horvitz_thompson_weight": population / 6,
        }
        for category, population in (
            frame_v22a.BASE_CATEGORY_POPULATIONS_V22A.items()
        )
    }


def test_v22a_replaces_all_184_sampled_pairs_and_keeps_56_actual_production_only(
    rebuilt_v22a,
):
    assert rebuilt_v22a["frozen_production_base"][
        "matched_candidate_replacement_count"
    ] == 184
    assert rebuilt_v22a["frozen_production_base"][
        "unchanged_production_only_count"
    ] == 56
    assert sum(
        panel["matched_candidate_replacement_count"]
        for panel in rebuilt_v22a["panels"]
    ) == 184
    assert sum(
        panel["unchanged_production_only_count"]
        for panel in rebuilt_v22a["panels"]
    ) == 56
    for panel in rebuilt_v22a["panels"]:
        name = panel["name"]
        assert panel["matched_candidate_replacement_count"] == (
            frame_v22a.EXPECTED_REPLACEMENTS_BY_PANEL_V22A[name]
        )
        control = panel["arms"]["production_control"]
        treatment = panel["arms"]["v341_matched_replacement"]
        assert control["requests"] == treatment["requests"] == 24
        assert control["population_denominator"] == (
            treatment["population_denominator"]
        ) == 272
        assert control["active_component_identity_root_sha256"] == (
            treatment["active_component_identity_root_sha256"]
        )
        assert treatment["candidate_representatives"] == (
            frame_v22a.EXPECTED_REPLACEMENTS_BY_PANEL_V22A[name]
        )
        assert treatment["production_representatives"] == (
            frame_v22a.EXPECTED_UNCHANGED_BY_PANEL_V22A[name]
        )


def test_v22a_isolated_projection_and_artifact_are_deterministic_and_row_free(
    rebuilt_v22a,
):
    projection = frame_v22a.run_firewalled_projection_v22a()
    assert projection["frozen_production_base"][
        "matched_candidate_replacement_count"
    ] == 184
    artifact = json.loads(frame_v22a.OUTPUT_PATH_V22A.read_text())
    assert artifact == rebuilt_v22a
    assert frame_v22a.validate_certificate_v22a(artifact) == artifact
    keys = {str(key).lower() for key in _all_keys(artifact)}
    assert not keys & {
        "question", "questions", "question_text", "answer", "answers",
        "prompt", "prompts", "response", "responses", "prompt_token_ids",
        "row", "rows_content", "row_content", "text", "document", "url",
    }
    assert artifact["firewall"] == {
        "foreground_builder_opened_or_parsed_jsonl_rows": False,
        "isolated_projection_worker_read_train_rows": True,
        "projection_contains_question_answer_prompt_response_or_row_content": False,
        "contains_evaluation_content": False,
        "heldout_validation_ood_eval_or_benchmark_content_opened": False,
        "runtime_launch_authorized": False,
        "gpu_launch_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["joint_frame"]["relation_counts"].update(
            {"paired": 206, "production_only": 66}
        ),
        lambda value: value["preregistration_time_correction"].update(
            {"fabricated_or_substituted_candidate_representative": True}
        ),
        lambda value: value["frozen_production_base"].update(
            {"matched_candidate_replacement_count": 185}
        ),
        lambda value: value["panels"][0]["arms"][
            "v341_matched_replacement"
        ].update({"requests": 23}),
        lambda value: value["panels"][0].update(
            {"treatment_representative_assignment_root_sha256": "0" * 64}
        ),
        lambda value: value["firewall"].update(
            {"runtime_launch_authorized": True}
        ),
    ),
)
def test_v22a_rejects_relation_fabrication_sample_arm_and_authority_tampering(
    rebuilt_v22a, mutation,
):
    tampered = copy.deepcopy(rebuilt_v22a)
    mutation(tampered)
    tampered["content_sha256_before_self_field"] = frame_v22a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="frame changed"):
        frame_v22a.validate_certificate_v22a(tampered)
