#!/usr/bin/env python3
"""Offline fail-closed tests for the V22A matched-replacement preregistration."""

import copy
import json

import pytest

import eggroll_es_v341_matched_replacement_preregistration_v22a as prereg_v22a


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
    return prereg_v22a.build_preregistration_v22a()


def test_v22a_prereg_binds_exact_corrected_frame_and_v341_candidate(rebuilt_v22a):
    assert rebuilt_v22a["inputs"]["frame"] == {
        "commit": prereg_v22a.FRAME_COMMIT_V22A,
        "path": str(prereg_v22a.frame_v22a.OUTPUT_PATH_V22A),
        "file_sha256": prereg_v22a.FRAME_CERTIFICATE_SHA256_V22A,
        "content_sha256": prereg_v22a.FRAME_CERTIFICATE_CONTENT_SHA256_V22A,
    }
    candidate = rebuilt_v22a["inputs"]["candidate"]
    assert candidate["source_commit"] == (
        "162e39408f4af0feee694dd4c128e9bb10dac057"
    )
    assert candidate["file_sha256"] == (
        "40f4b73c25cccfddc49da039b40483b469b9858deec9d00ca399cb490f5aa47a"
    )
    correction = rebuilt_v22a["preregistration_time_correction"]
    assert correction["exact_v341_relation_counts"] == {
        "paired": 205, "candidate_only": 54, "production_only": 67,
    }
    assert correction["exact_v341_sampled_replacements"] == 184
    assert correction["fabricated_or_substituted_candidate_representative"] is False
    assert correction["post_result_adaptation"] is False


def test_v22a_prereg_freezes_equal_population_matched_replacement_estimand(
    rebuilt_v22a,
):
    frame = rebuilt_v22a["frame_contract"]
    assert frame["production_population_components"] == 272
    assert frame["globally_disjoint_sampled_components"] == 240
    assert frame["requests_per_panel_both_arms"] == 24
    assert frame["population_denominator_both_arms"] == 272
    assert frame["sampled_matched_candidate_replacements"] == 184
    assert frame["sampled_production_only_unchanged"] == 56
    assert frame["candidate_only_components_excluded"] == 54
    assert rebuilt_v22a["arms"] == {
        "production_control": {
            "requests_per_panel": 24,
            "requests_per_engine": 240,
            "population_denominator": 272,
        },
        "v341_matched_replacement": {
            "requests_per_panel": 24,
            "requests_per_engine": 240,
            "population_denominator": 272,
        },
    }
    assert rebuilt_v22a["estimator"][
        "same_component_draws_and_ht_coefficients_both_arms"
    ] is True
    assert rebuilt_v22a["estimator"]["candidate_only_strata_present"] is False


def test_v22a_has_fresh_64_direction_basis_distinct_from_v18_through_v21(
    rebuilt_v22a,
):
    basis = prereg_v22a.validate_perturbation_basis_v22a()
    recipe = rebuilt_v22a["frozen_recipe"]
    assert len(basis["seeds"]) == len(set(basis["seeds"])) == 64
    assert prereg_v22a.canonical_sha256(basis) == (
        prereg_v22a.PERTURBATION_BASIS_SHA256_V22A
    )
    assert recipe["perturbation_basis_sha256"] not in set(
        prereg_v22a.PRIOR_BASIS_CONTENT_SHA256.values()
    )
    assert recipe["prior_v18_v19_v20_v21_basis_reuse_allowed"] is False
    assert recipe["model_family"] == "Qwen3.6-35B-A3B"
    assert recipe["layers"] == [20, 21, 22, 23]
    assert recipe["sigma"] == 0.0003
    assert recipe["alpha"] == 0.0


def test_v22a_direction_schedule_and_equal_480_request_wave_are_exact(rebuilt_v22a):
    raw = rebuilt_v22a["paired_raw_schedule"]
    assert raw["terminology"] == {
        "antithetic_direction_count": 64,
        "signs_per_direction": 2,
        "signed_direction_evaluation_count": 128,
        "directions_per_resident_signed_wave": 4,
        "resident_signed_wave_count": 32,
        "engine_count": 4,
        "identity": "32 resident signed waves x 4 engines = 128 signed direction evaluations",
    }
    schedule = raw["resident_signed_wave_schedule"]
    assert len(schedule) == 32
    assert sum(len(item["engine_direction_seeds"]) for item in schedule) == 128
    assert raw["requests_per_engine_per_resident_signed_wave"] == 480
    assert raw["requests_by_arm_per_engine_per_resident_signed_wave"] == {
        "production_control": 240,
        "v341_matched_replacement": 240,
    }
    assert raw["requests_per_engine_all_resident_signed_waves"] == 15_360
    assert raw["requests_all_engines_all_resident_signed_waves"] == 61_440
    assert raw["dense_result_commitment_count"] == 2_560
    for sign in ("plus", "minus"):
        signed = [item for item in schedule if item["sign"] == sign]
        for arm in prereg_v22a.frame_v22a.ARM_ORDER_V22A:
            assert sorted(item["resident_arm_order"].index(arm) for item in signed) == (
                [0] * 8 + [1] * 8
            )


def test_v22a_freezes_one_contrast_twelve_endpoints_and_50k_paired_gate(
    rebuilt_v22a,
):
    analysis = rebuilt_v22a["analysis"]
    bootstrap = analysis["bootstrap"]
    assert analysis["contrast"]["name"] == (
        "v341_matched_replacement_vs_production"
    )
    assert analysis["endpoint_names"] == list(prereg_v22a.ENDPOINT_NAMES_V22A)
    assert analysis["endpoint_count"] == analysis["hypothesis_count"] == 12
    assert bootstrap["repetitions"] == 50_000
    assert bootstrap["one_sided_quantile"] == 0.05 / 12
    assert bootstrap["paired_same_draws_both_arms"] is True
    assert bootstrap["candidate_only_resampling_present"] is False
    assert bootstrap["whole_panel_block_resampling_used"] is False
    gate = rebuilt_v22a["compatibility_gate"]
    assert gate["noninferiority_margin"] == 0.0
    assert gate["observed_treatment_minus_control_nonnegative_all_twelve"] is True
    assert gate["zero_margin_familywise_lcb_nonnegative_all_twelve"] is True


def test_v22a_raw_only_prior_role_and_no_authority_are_explicit(rebuilt_v22a):
    raw = rebuilt_v22a["paired_raw_schedule"]
    assert raw["raw_arm_scoring_only"] is True
    assert raw["union_scoring_authorized"] is False
    assert raw["union_planner_called"] is False
    prior = rebuilt_v22a["prior_experiment_role"]
    assert prior[
        "v21_gate_decision_motivated_excluding_candidate_only_expansion"
    ] is True
    assert prior["v21_numeric_endpoint_values_used_to_tune_v22_recipe_or_gate"] is False
    assert prior["v22_runtime_result_observed"] is False
    assert rebuilt_v22a["firewall"] == {
        "offline_frame_and_preregistration_only": True,
        "contains_question_answer_prompt_response_token_or_row_content": False,
        "heldout_validation_ood_eval_or_benchmark_content_opened": False,
        "gpu_launch_authorized": False,
        "runtime_launch_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
    }


def test_v22a_artifact_is_deterministic_compact_and_has_no_row_qa_keys(rebuilt_v22a):
    artifact = json.loads(prereg_v22a.OUTPUT_PATH_V22A.read_text())
    assert artifact == rebuilt_v22a
    assert prereg_v22a.validate_preregistration_v22a(artifact) == artifact
    keys = {str(key).lower() for key in _all_keys(artifact)}
    assert not keys & {
        "question", "questions", "question_text", "answer", "answers",
        "prompt", "prompts", "response", "responses", "prompt_token_ids",
        "row", "row_content", "rows_content", "text", "document", "url",
        "unit_scores", "bootstrap_draws", "bootstrap_replicates",
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["preregistration_time_correction"].update(
            {"exact_v341_sampled_replacements": 185}
        ),
        lambda value: value["inputs"]["candidate"].update(
            {"file_sha256": "0" * 64}
        ),
        lambda value: value["frame_contract"].update(
            {"candidate_only_components_excluded": 53}
        ),
        lambda value: value["arms"]["v341_matched_replacement"].update(
            {"population_denominator": 326}
        ),
        lambda value: value["frozen_recipe"].update(
            {"perturbation_basis_sha256": prereg_v22a.PRIOR_BASIS_CONTENT_SHA256["v21a"]}
        ),
        lambda value: value["paired_raw_schedule"].update(
            {"requests_per_engine_per_resident_signed_wave": 540}
        ),
        lambda value: value["paired_raw_schedule"].update(
            {"exact_restore_and_verify_once_after_both_arms_each_wave": False}
        ),
        lambda value: value["analysis"]["bootstrap"].update(
            {"one_sided_quantile": 0.05 / 11}
        ),
        lambda value: value["compatibility_gate"].update(
            {"noninferiority_margin": float("inf")}
        ),
        lambda value: value["firewall"].update(
            {"runtime_launch_authorized": True}
        ),
    ),
)
def test_v22a_rejects_frame_basis_schedule_gate_and_authority_tampering(
    rebuilt_v22a, mutation,
):
    tampered = copy.deepcopy(rebuilt_v22a)
    mutation(tampered)
    tampered["content_sha256_before_self_field"] = prereg_v22a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="preregistration changed"):
        prereg_v22a.validate_preregistration_v22a(tampered)
