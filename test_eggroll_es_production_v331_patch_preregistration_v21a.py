#!/usr/bin/env python3
"""Offline fail-closed tests for the V21A v331 patch preregistration."""

import copy
import json

import pytest

import eggroll_es_production_v331_patch_preregistration_v21a as prereg_v21a


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
    return prereg_v21a.build_preregistration_v21a()


def test_v21a_prereg_binds_corrected_frame_and_commit_derived_candidate(rebuilt_v21a):
    assert rebuilt_v21a["inputs"]["frame"] == {
        "commit": "32bd189372a14b0c7db155d8986dcc6d4928b93b",
        "path": str(prereg_v21a.frame_v21a.OUTPUT_PATH_V21A),
        "file_sha256": prereg_v21a.FRAME_CERTIFICATE_SHA256_V21A,
        "content_sha256": prereg_v21a.FRAME_CERTIFICATE_CONTENT_SHA256_V21A,
    }
    candidate = rebuilt_v21a["inputs"]["candidate"]
    assert candidate["source_commit"] == (
        "9d31e3407dd96f80011bdb2202e8aa9689b1d193"
    )
    assert candidate["file_sha256"] == (
        "29889e18fd7a8bf313021ec4e8ad18adcc1752529fba247fdbde006513f7866b"
    )
    assert rebuilt_v21a["independence"]["v20_runtime_result_read_or_used"] is False


def test_v21a_prereg_freezes_full_merge_and_corrected_role_overlap(rebuilt_v21a):
    frame = rebuilt_v21a["frame_contract"]
    assert frame["candidate_only_global_unique_components"] == 54
    assert frame["candidate_only_total_assignments"] == 60
    assert frame["optimization_assignments_all_unique"] == 36
    assert frame["train_screen_assignments_all_unique"] == 24
    assert frame["cross_role_overlap_count"] == 6
    assert frame["cross_role_overlap_by_topic_explicit_zero"] == {
        "safety_consent": 2,
        "technique": 2,
        "equipment_material": 0,
        "resources_general": 2,
    }
    assert frame["full_merge_keeps_all_sampled_production_representatives"] is True
    assert frame["paired_candidate_replacements"] == 0


def test_v21a_prereg_has_fresh_64_direction_basis_distinct_from_v18_v19_v20(rebuilt_v21a):
    basis = prereg_v21a.validate_perturbation_basis_v21a()
    recipe = rebuilt_v21a["frozen_recipe"]
    assert len(basis["seeds"]) == len(set(basis["seeds"])) == 64
    assert prereg_v21a.canonical_sha256(basis) == (
        prereg_v21a.PERTURBATION_BASIS_SHA256_V21A
    )
    assert recipe["perturbation_basis_sha256"] not in set(
        prereg_v21a.PRIOR_BASIS_CONTENT_SHA256.values()
    )
    assert recipe["prior_v18_v19_v20_basis_reuse_allowed"] is False
    assert recipe["model_family"] == "Qwen3.6-35B-A3B"
    assert recipe["layers"] == [20, 21, 22, 23]
    assert recipe["alpha"] == 0.0


def test_v21a_direction_wave_terminology_and_paired_raw_schedule_are_unambiguous(
    rebuilt_v21a,
):
    raw = rebuilt_v21a["paired_raw_schedule"]
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
    assert raw["population_wave_count"] == 16
    assert raw["requests_per_engine_per_resident_signed_wave"] == 540
    assert raw["requests_per_engine_all_resident_signed_waves"] == 17_280
    assert raw["dense_result_commitment_count"] == 2_560
    assert raw["same_perturbation_basis_and_direction_for_both_arms"] is True
    for sign in ("plus", "minus"):
        signed = [item for item in schedule if item["sign"] == sign]
        for arm in prereg_v21a.frame_v21a.ARM_ORDER_V21A:
            assert sorted(item["resident_arm_order"].index(arm) for item in signed) == (
                [0] * 8 + [1] * 8
            )


def test_v21a_prereg_freezes_one_contrast_12_endpoints_and_50k_paired_bootstrap(
    rebuilt_v21a,
):
    analysis = rebuilt_v21a["analysis"]
    bootstrap = analysis["bootstrap"]
    assert analysis["endpoint_names"] == list(prereg_v21a.ENDPOINT_NAMES_V21A)
    assert analysis["endpoint_count"] == analysis["hypothesis_count"] == 12
    assert bootstrap["repetitions"] == 50_000
    assert bootstrap["one_sided_quantile"] == 0.05 / 12
    assert bootstrap["paired_same_draws_both_arms"] is True
    assert bootstrap["whole_panel_block_resampling_used"] is False
    assert rebuilt_v21a["compatibility_gate"]["noninferiority_margin"] == 0.0


def test_v21a_prereg_raw_only_union_forbidden_and_no_authority(rebuilt_v21a):
    evidence = rebuilt_v21a["inputs"]["union_failure_evidence"]
    raw = rebuilt_v21a["paired_raw_schedule"]
    assert evidence["commit"] == prereg_v21a.UNION_FAILURE_COMMIT_V21A
    assert evidence["raw_scoring_authoritative"] is True
    assert evidence["union_scoring_authorized"] is False
    assert raw["raw_arm_scoring_only"] is True
    assert raw["union_scoring_authorized"] is False
    assert raw["union_planner_called"] is False
    assert rebuilt_v21a["firewall"] == {
        "offline_frame_and_preregistration_only": True,
        "heldout_validation_ood_eval_or_benchmark_content_opened": False,
        "gpu_launch_authorized": False,
        "runtime_launch_authorized": False,
        "model_update_authorized": False,
        "checkpoint_write_authorized": False,
        "evaluation_authorized": False,
        "dataset_promotion_authorized": False,
    }


def test_v21a_prereg_is_deterministic_compact_and_contains_no_train_rows(rebuilt_v21a):
    artifact = json.loads(prereg_v21a.OUTPUT_PATH_V21A.read_text())
    assert artifact == rebuilt_v21a
    assert prereg_v21a.validate_preregistration_v21a(artifact) == artifact
    keys = {str(key).lower() for key in _all_keys(artifact)}
    assert not keys & {
        "question_text", "questions", "answer", "answers", "prompt", "prompts",
        "response", "responses", "prompt_token_ids", "row_content", "unit_scores",
    }


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["paired_raw_schedule"]["terminology"].update(
            {"signed_direction_evaluation_count": 64}
        ),
        lambda value: value["paired_raw_schedule"].update(
            {"union_scoring_authorized": True}
        ),
        lambda value: value["frame_contract"].update(
            {"train_screen_assignments_all_unique": 23}
        ),
        lambda value: value["firewall"].update({"runtime_launch_authorized": True}),
    ),
)
def test_v21a_prereg_rejects_direction_union_frame_and_authority_tampering(
    rebuilt_v21a, mutation,
):
    tampered = copy.deepcopy(rebuilt_v21a)
    mutation(tampered)
    tampered["content_sha256_before_self_field"] = prereg_v21a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="preregistration changed"):
        prereg_v21a.validate_preregistration_v21a(tampered)
