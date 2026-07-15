#!/usr/bin/env python3
"""Offline-only V22A exact-v341 matched-replacement preregistration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_v341_matched_replacement_frame_v22a as frame_v22a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V22A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_PRODUCTION_V341_MATCHED_REPLACEMENT_V22A_PREREGISTRATION.json"
).resolve()
FRAME_COMMIT_V22A = "1658798493838efbafeaa506940f48350959da95"
UNION_FAILURE_COMMIT_V22A = "a276bc8b982a04a608535a7c9ccda7e9dc77913f"
LAYER_PLAN_COMMIT_V22A = "a4cfdb6726d4381f5d280b47ca4866569c35bd3b"

FRAME_BUILDER_SHA256_V22A = (
    "8aed3ec9b64ff54eea5217f05927400f970b46c2486deb9a8274ec39226fb2ae"
)
FRAME_TEST_SHA256_V22A = (
    "46ab8a545f01ee700d6162fff1825d54517eb0f43d8e58482cdf30f4bb2ed953"
)
FRAME_CERTIFICATE_SHA256_V22A = (
    "a5e2d4384d40db3a4737c1c7d1a3859437658a54ad98b2448d36200cbb5d9683"
)
FRAME_CERTIFICATE_CONTENT_SHA256_V22A = (
    "f34a003a77e9e978e244f8bced8d6550675603d891f386684e288958871cc5ce"
)
UNION_FAILURE_EVIDENCE_PATH_V22A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V20A_UNION_EQUIVALENCE_FAILURE_EVIDENCE.json"
).resolve()
UNION_FAILURE_EVIDENCE_SHA256_V22A = (
    "d84fbd7b373b284f3c6a2ffbf1bd52431818ec4e6c60b4f3b320d36b581c9021"
)
UNION_FAILURE_EVIDENCE_CONTENT_SHA256_V22A = (
    "2caa4e1658ae794a3193ef28bb3393b966f1d329d574bc8d0b2534e3fb1f3302"
)
LAYER_PLAN_PATH_V22A = (
    ROOT / "experiments/layer_plans/middle_late_dense_v6.json"
).resolve()
LAYER_PLAN_FILE_SHA256_V22A = (
    "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df"
)
LAYER_PLAN_SHA256_V22A = (
    "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9"
)
MODEL_CONFIG_SHA256_V22A = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
MODEL_PATH_V22A = (ROOT / "models/Qwen3.6-35B-A3B").resolve()

EXPERIMENT_NAME_V22A = "production_v341_matched_replacement_v22a"
ENDPOINT_NAMES_V22A = (
    "optimization_pairwise_cosine_median",
    "optimization_pairwise_cosine_worst",
    "optimization_pairwise_sign_agreement_median",
    "optimization_pairwise_sign_agreement_worst",
    "aggregate_to_optimization_cosine_median",
    "aggregate_to_optimization_cosine_worst",
    "aggregate_to_optimization_sign_agreement_median",
    "aggregate_to_optimization_sign_agreement_worst",
    "train_screen_cosine_median",
    "train_screen_cosine_worst",
    "train_screen_sign_agreement_median",
    "train_screen_sign_agreement_worst",
)
FRESH_PERTURBATION_BASIS_SEED_V22A = 20260823
PERTURBATION_SEEDS_V22A = [
    929171943, 44283976, 19243918, 188077630,
    563254792, 260615822, 67083744, 501482804,
    1005030771, 1035223403, 400750884, 186689367,
    1042064539, 858149574, 757977916, 319071754,
    1027554276, 887857539, 830973270, 948610724,
    431976182, 703103948, 1041650540, 1046296268,
    796304359, 424364588, 1003116963, 325094061,
    400710689, 982893400, 285495853, 449714629,
    874681403, 988405411, 633088297, 120320982,
    702096201, 262518360, 868600344, 376841028,
    849056904, 555898751, 338012560, 929943581,
    197147603, 19659205, 221620274, 281385504,
    894510778, 696381376, 518777439, 177362306,
    849531214, 889116738, 921829666, 76728472,
    934240095, 541986188, 41149524, 159169717,
    420088791, 848969700, 516777682, 954243503,
]
PERTURBATION_SEED_LIST_SHA256_V22A = (
    "9faecdc81492052a6c466b0e986df9e31be0c0fccf24687a96ed604f2ef0f553"
)
PERTURBATION_BASIS_SHA256_V22A = (
    "f68624388ac0549ac82ba3d1e64a317233c42f900502a6f5c6d6f07071b4c60e"
)
PRIOR_BASIS_CONTENT_SHA256 = {
    "v18a": "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11",
    "v19a": "d4e46d7d51d5c82cfc981dad3b33db8a1766c70ad570ef931b12550d1bc7bf6c",
    "v20a": "b6d667c2f125f9d0be4d74ef536af03546fecb6c03f2838679f5a315a1ec9852",
    "v21a": "65970861cd06b53e52cf848b2c8b8961160bf9c68f6b1b9f4935a88ba8d314d2",
}
BOOTSTRAP_SEED_V22A = 20260824
BOOTSTRAP_REPETITIONS_V22A = 50_000
FAMILYWISE_ALPHA_V22A = 0.05
HYPOTHESIS_COUNT_V22A = 12

canonical_sha256 = frame_v22a.canonical_sha256
file_sha256 = frame_v22a.file_sha256


def _without_self(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file(path: Path, commit: str, digest: str) -> None:
    path = Path(path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    raw = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v22a preregistered artifact changed: {relative}")


def perturbation_basis_v22a() -> dict:
    return {
        "schema": "eggroll-es-v341-matched-replacement-perturbation-basis-v22a",
        "basis_seed": FRESH_PERTURBATION_BASIS_SEED_V22A,
        "population_size": 64,
        "seeds": list(PERTURBATION_SEEDS_V22A),
    }


def validate_perturbation_basis_v22a() -> dict:
    regenerated = np.random.default_rng(
        seed=FRESH_PERTURBATION_BASIS_SEED_V22A
    ).integers(0, 2**30, size=64, dtype=np.int64).tolist()
    basis = perturbation_basis_v22a()
    if (
        regenerated != PERTURBATION_SEEDS_V22A
        or len(set(regenerated)) != 64
        or canonical_sha256(regenerated) != PERTURBATION_SEED_LIST_SHA256_V22A
        or canonical_sha256(basis) != PERTURBATION_BASIS_SHA256_V22A
        or PERTURBATION_BASIS_SHA256_V22A
        in set(PRIOR_BASIS_CONTENT_SHA256.values())
    ):
        raise RuntimeError("v22a fresh 64-direction perturbation basis changed")
    return basis


def signed_wave_schedule_v22a() -> list[dict]:
    """Return 32 four-engine waves covering 64 antithetic directions."""
    schedule = []
    arms = list(frame_v22a.ARM_ORDER_V22A)
    for population_wave, start in enumerate(range(0, 64, 4)):
        seeds = PERTURBATION_SEEDS_V22A[start:start + 4]
        for sign_index, sign in enumerate(("plus", "minus")):
            first = (population_wave + sign_index) % 2
            order = arms[first:] + arms[:first]
            schedule.append({
                "resident_signed_wave_index": 2 * population_wave + sign_index,
                "population_wave_index": population_wave,
                "sign": sign,
                "negate": sign == "minus",
                "engine_direction_seeds": list(seeds),
                "resident_arm_order": order,
                "score_both_arms_before_restore": True,
                "restore_once_after_both_arms": True,
            })
    if (
        len(schedule) != 32
        or sum(len(item["engine_direction_seeds"]) for item in schedule) != 128
        or any(set(item["resident_arm_order"]) != set(arms) for item in schedule)
    ):
        raise RuntimeError("v22a resident signed-wave schedule changed")
    for sign in ("plus", "minus"):
        signed = [item for item in schedule if item["sign"] == sign]
        for arm in arms:
            if sorted(item["resident_arm_order"].index(arm) for item in signed) != (
                [0] * 8 + [1] * 8
            ):
                raise RuntimeError("v22a two-arm signed-wave balance changed")
    return schedule


def load_bound_inputs_v22a() -> tuple[dict, dict, dict]:
    for path, commit, digest in (
        (Path(frame_v22a.__file__), FRAME_COMMIT_V22A, FRAME_BUILDER_SHA256_V22A),
        (
            ROOT / "test_build_eggroll_es_v341_matched_replacement_frame_v22a.py",
            FRAME_COMMIT_V22A,
            FRAME_TEST_SHA256_V22A,
        ),
        (
            frame_v22a.OUTPUT_PATH_V22A,
            FRAME_COMMIT_V22A,
            FRAME_CERTIFICATE_SHA256_V22A,
        ),
        (
            UNION_FAILURE_EVIDENCE_PATH_V22A,
            UNION_FAILURE_COMMIT_V22A,
            UNION_FAILURE_EVIDENCE_SHA256_V22A,
        ),
        (
            LAYER_PLAN_PATH_V22A,
            LAYER_PLAN_COMMIT_V22A,
            LAYER_PLAN_FILE_SHA256_V22A,
        ),
    ):
        _verify_commit_file(path, commit, digest)
    frame = json.loads(frame_v22a.OUTPUT_PATH_V22A.read_text())
    frame_v22a.validate_certificate_v22a(frame)
    evidence = json.loads(UNION_FAILURE_EVIDENCE_PATH_V22A.read_text())
    layer_plan = json.loads(LAYER_PLAN_PATH_V22A.read_text())
    if (
        frame["content_sha256_before_self_field"]
        != FRAME_CERTIFICATE_CONTENT_SHA256_V22A
        or evidence.get("schema")
        != "eggroll-es-union-equivalence-failure-evidence-v20a"
        or evidence.get("content_sha256_before_self_field")
        != UNION_FAILURE_EVIDENCE_CONTENT_SHA256_V22A
        or evidence.get("decision", {}).get(
            "raw_arm_scoring_remains_authoritative"
        ) is not True
        or evidence.get("decision", {}).get(
            "union_scoring_authorized_for_v20a"
        ) is not False
        or layer_plan.get("plan_sha256") != LAYER_PLAN_SHA256_V22A
        or layer_plan.get("model_config_sha256") != MODEL_CONFIG_SHA256_V22A
        or layer_plan.get("layers") != [20, 21, 22, 23]
        or layer_plan.get("num_units") != 35
    ):
        raise RuntimeError("v22a preregistered input content changed")
    return frame, evidence, layer_plan


def _scoring_contract_v22a() -> dict:
    return {
        "objective": "per_example_mean_full_gold_answer_token_logprob",
        "full_gold_answer_in_scoring_sequence": True,
        "prompt_logprobs": 1,
        "scored_positions": "all_aligned_answer_tokens_only",
        "max_tokens": 1,
        "max_tokens_role": "dummy_generation_trigger_not_answer_cap",
        "frozen_max_total_prompt_answer_tokens": 1024,
        "detokenize": False,
        "eos_scored": False,
        "autoregressive_accuracy_or_token_f1_used": False,
        "objective_change_allowed": False,
        "v341_token_boundary_audit_required_before_any_future_runtime": True,
        "this_offline_preregistration_authorizes_runtime": False,
    }


def build_preregistration_v22a() -> dict:
    frame, evidence, layer_plan = load_bound_inputs_v22a()
    basis = validate_perturbation_basis_v22a()
    schedule = signed_wave_schedule_v22a()
    value = {
        "schema": "eggroll-es-v341-matched-replacement-preregistration-v22a",
        "status": "preregistered_offline_only_before_v22_runtime_no_v22_result_dependency",
        "experiment_name": EXPERIMENT_NAME_V22A,
        "scientific_objective": (
            "attribute_exact_v341_manual_representative_changes_on_the_same_"
            "frozen_production_component_sample_without_candidate_only_expansion"
        ),
        "inputs": {
            "frame": {
                "commit": FRAME_COMMIT_V22A,
                "path": str(frame_v22a.OUTPUT_PATH_V22A),
                "file_sha256": FRAME_CERTIFICATE_SHA256_V22A,
                "content_sha256": FRAME_CERTIFICATE_CONTENT_SHA256_V22A,
            },
            "candidate": copy.deepcopy(frame["inputs"]["candidate_v341"]),
            "production": copy.deepcopy(frame["inputs"]["production"]),
            "union_failure_evidence": {
                "commit": UNION_FAILURE_COMMIT_V22A,
                "path": str(UNION_FAILURE_EVIDENCE_PATH_V22A),
                "file_sha256": UNION_FAILURE_EVIDENCE_SHA256_V22A,
                "content_sha256": UNION_FAILURE_EVIDENCE_CONTENT_SHA256_V22A,
                "raw_scoring_authoritative": evidence["decision"][
                    "raw_arm_scoring_remains_authoritative"
                ],
                "union_scoring_authorized": False,
            },
        },
        "preregistration_time_correction": copy.deepcopy(
            frame["preregistration_time_correction"]
        ),
        "frame_contract": {
            "panel_names": list(frame_v22a.PANEL_NAMES_V22A),
            "optimization_panels": list(frame_v22a.OPTIMIZATION_PANELS_V22A),
            "train_only_screen_panels": list(frame_v22a.TRAIN_SCREEN_PANELS_V22A),
            "production_population_components": 272,
            "globally_disjoint_sampled_components": 240,
            "requests_per_panel_both_arms": 24,
            "population_denominator_both_arms": 272,
            "base_category_quota_per_panel": {
                category: frame_v22a.BASE_CATEGORY_QUOTA_V22A
                for category in frame_v22a.BASE_CATEGORIES_V22A
            },
            "production_topic_quota_per_panel": copy.deepcopy(
                frame_v22a.PRODUCTION_TOPIC_QUOTAS_V22A
            ),
            "exact_v341_joint_relation_counts": copy.deepcopy(
                frame["joint_frame"]["relation_counts"]
            ),
            "sampled_matched_candidate_replacements": 184,
            "sampled_production_only_unchanged": 56,
            "candidate_only_components_excluded": 54,
            "same_components_panels_ht_weights_and_denominator_both_arms": True,
            "fabricated_or_substituted_candidate_representative": False,
        },
        "arms": {
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
        },
        "estimator": {
            "shared_base_ht_strata": copy.deepcopy(
                frame["estimand"]["base_population_ht_strata"]
            ),
            "same_component_draws_and_ht_coefficients_both_arms": True,
            "candidate_representative_only_for_exact_paired_component": True,
            "production_representative_for_actual_production_only_component": True,
            "candidate_only_strata_present": False,
            "plain_request_mean_used": False,
            "same_component_counted_once_per_arm": True,
        },
        "frozen_recipe": {
            "model": str(MODEL_PATH_V22A),
            "model_family": "Qwen3.6-35B-A3B",
            "layers": [20, 21, 22, 23],
            "layer_plan": {
                "commit": LAYER_PLAN_COMMIT_V22A,
                "path": str(LAYER_PLAN_PATH_V22A),
                "file_sha256": LAYER_PLAN_FILE_SHA256_V22A,
                "plan_sha256": LAYER_PLAN_SHA256_V22A,
                "model_config_sha256": MODEL_CONFIG_SHA256_V22A,
            },
            "sigma": 0.0003,
            "alpha": 0.0,
            "population_size": 64,
            "perturbation_basis": basis,
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V22A,
            "perturbation_seed_list_sha256": PERTURBATION_SEED_LIST_SHA256_V22A,
            "prior_basis_content_sha256": PRIOR_BASIS_CONTENT_SHA256,
            "prior_v18_v19_v20_v21_basis_reuse_allowed": False,
            "basis_generator": (
                "numpy.default_rng(basis_seed).integers(0,2**30,size=64,dtype=int64)"
            ),
            "basis_generation_or_selection_at_runtime_allowed": False,
            "hardware": {
                "engine_count": 4,
                "tensor_parallel_per_engine": 1,
                "gpu_ids": [0, 1, 2, 3],
                "all_four_engines_every_resident_signed_wave": True,
            },
            "moe_backend": {
                "backend": "default_triton",
                "all_backend_override_environment_variables_must_be_unset": True,
                "v16_task_ab_decision_retained": "default_triton",
            },
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_surfaces_opened": False,
        },
        "paired_raw_schedule": {
            "terminology": {
                "antithetic_direction_count": 64,
                "signs_per_direction": 2,
                "signed_direction_evaluation_count": 128,
                "directions_per_resident_signed_wave": 4,
                "resident_signed_wave_count": 32,
                "engine_count": 4,
                "identity": (
                    "32 resident signed waves x 4 engines = 128 signed direction evaluations"
                ),
            },
            "population_wave_count": 16,
            "resident_signed_wave_schedule": schedule,
            "same_resident_perturbation_scores_both_arms_before_restore": True,
            "same_perturbation_basis_and_direction_for_both_arms": True,
            "arm_order_balanced_within_each_sign": True,
            "each_arm_appears_in_each_position_eight_times_per_sign": True,
            "exact_restore_and_verify_once_after_both_arms_each_wave": True,
            "partial_waves_allowed": False,
            "requests_per_engine_per_resident_signed_wave": 480,
            "requests_by_arm_per_engine_per_resident_signed_wave": {
                "production_control": 240,
                "v341_matched_replacement": 240,
            },
            "requests_per_engine_all_resident_signed_waves": 15_360,
            "requests_all_engines_all_resident_signed_waves": 61_440,
            "signed_direction_evaluations_all_engines": 128,
            "dense_result_commitment_count": 2_560,
            "raw_arm_scoring_only": True,
            "union_scoring_authorized": False,
            "union_planner_called": False,
        },
        "scoring": _scoring_contract_v22a(),
        "analysis": {
            "contrast": {
                "name": "v341_matched_replacement_vs_production",
                "treatment": "v341_matched_replacement",
                "control": "production_control",
                "treatment_minus_control": True,
            },
            "endpoint_names": list(ENDPOINT_NAMES_V22A),
            "endpoint_count": 12,
            "hypothesis_count": 12,
            "panel_endpoint_geometry": {
                "optimization_pairwise_pairs": 15,
                "aggregate_to_optimization_panels": 6,
                "train_only_screen_panels": 4,
                "median_and_worst_for_each_metric_family": True,
            },
            "bootstrap": {
                "seed": BOOTSTRAP_SEED_V22A,
                "repetitions": BOOTSTRAP_REPETITIONS_V22A,
                "familywise_alpha": FAMILYWISE_ALPHA_V22A,
                "one_sided_quantile": FAMILYWISE_ALPHA_V22A / 12,
                "multiplicity": "Bonferroni_over_one_contrast_times_twelve_endpoints",
                "paired_same_draws_both_arms": True,
                "within_panel_base_resampling": (
                    "resample_six_components_within_each_of_four_base_HT_strata_"
                    "using_identical_draws_and_coefficients_both_arms"
                ),
                "candidate_only_resampling_present": False,
                "whole_panel_block_resampling_used": False,
                "recompute_HT_arm_totals_exact_shared_denominator_coefficients_"
                "aggregate_and_all_nonlinear_endpoints_each_replicate": True,
                "persist_per_unit_scores_bootstrap_replicates_or_draws": False,
            },
        },
        "compatibility_gate": {
            "observed_treatment_minus_control_nonnegative_all_twelve": True,
            "zero_margin_familywise_lcb_nonnegative_all_twelve": True,
            "noninferiority_margin": 0.0,
            "all_panel_spreads_nonzero": True,
            "all_runtime_identity_restoration_boundary_backend_and_raw_audits": True,
            "no_pass_decision": "retain_production_dataset_and_v13_recipe",
            "pass_decision": (
                "authorize_only_a_separate_fresh_basis_train_only_confirmation_"
                "preregistration"
            ),
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        },
        "prior_experiment_role": {
            "v21_gate_decision_motivated_excluding_candidate_only_expansion": True,
            "v21_numeric_endpoint_values_used_to_tune_v22_recipe_or_gate": False,
            "v22_runtime_result_observed": False,
            "ongoing_curation_after_v341_source_commit_used": False,
        },
        "firewall": {
            "offline_frame_and_preregistration_only": True,
            "contains_question_answer_prompt_response_token_or_row_content": False,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "gpu_launch_authorized": False,
            "runtime_launch_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_preregistration_v22a(value)


def validate_preregistration_v22a(value: dict) -> dict:
    frame_contract = value.get("frame_contract", {})
    correction = value.get("preregistration_time_correction", {})
    arms = value.get("arms", {})
    estimator = value.get("estimator", {})
    recipe = value.get("frozen_recipe", {})
    raw = value.get("paired_raw_schedule", {})
    terms = raw.get("terminology", {})
    analysis = value.get("analysis", {})
    bootstrap = analysis.get("bootstrap", {})
    gate = value.get("compatibility_gate", {})
    prior = value.get("prior_experiment_role", {})
    firewall = value.get("firewall", {})
    expected_correction = {
        "cause": "exact_v341_correctness_removal_eliminated_one_v331_pair",
        "prior_assumed_relation_counts": {
            "paired": 206, "candidate_only": 54, "production_only": 66,
        },
        "exact_v341_relation_counts": {
            "paired": 205, "candidate_only": 54, "production_only": 67,
        },
        "affected_frozen_sample_panel": "train_screen_0",
        "prior_expected_sampled_replacements": 185,
        "exact_v341_sampled_replacements": 184,
        "fabricated_or_substituted_candidate_representative": False,
        "correction_made_before_runtime_or_result_observation": True,
        "post_result_adaptation": False,
    }
    expected_frame_contract = {
        "panel_names": list(frame_v22a.PANEL_NAMES_V22A),
        "optimization_panels": list(frame_v22a.OPTIMIZATION_PANELS_V22A),
        "train_only_screen_panels": list(frame_v22a.TRAIN_SCREEN_PANELS_V22A),
        "production_population_components": 272,
        "globally_disjoint_sampled_components": 240,
        "requests_per_panel_both_arms": 24,
        "population_denominator_both_arms": 272,
        "base_category_quota_per_panel": {
            category: frame_v22a.BASE_CATEGORY_QUOTA_V22A
            for category in frame_v22a.BASE_CATEGORIES_V22A
        },
        "production_topic_quota_per_panel": frame_v22a.PRODUCTION_TOPIC_QUOTAS_V22A,
        "exact_v341_joint_relation_counts": {
            "paired": 205, "candidate_only": 54, "production_only": 67,
        },
        "sampled_matched_candidate_replacements": 184,
        "sampled_production_only_unchanged": 56,
        "candidate_only_components_excluded": 54,
        "same_components_panels_ht_weights_and_denominator_both_arms": True,
        "fabricated_or_substituted_candidate_representative": False,
    }
    expected_estimator = {
        "shared_base_ht_strata": {
            category: {
                "population": population,
                "per_panel_quota": frame_v22a.BASE_CATEGORY_QUOTA_V22A,
                "horvitz_thompson_weight": (
                    population / frame_v22a.BASE_CATEGORY_QUOTA_V22A
                ),
            }
            for category, population in (
                frame_v22a.BASE_CATEGORY_POPULATIONS_V22A.items()
            )
        },
        "same_component_draws_and_ht_coefficients_both_arms": True,
        "candidate_representative_only_for_exact_paired_component": True,
        "production_representative_for_actual_production_only_component": True,
        "candidate_only_strata_present": False,
        "plain_request_mean_used": False,
        "same_component_counted_once_per_arm": True,
    }
    expected_bootstrap = {
        "seed": BOOTSTRAP_SEED_V22A,
        "repetitions": BOOTSTRAP_REPETITIONS_V22A,
        "familywise_alpha": FAMILYWISE_ALPHA_V22A,
        "one_sided_quantile": FAMILYWISE_ALPHA_V22A / 12,
        "multiplicity": "Bonferroni_over_one_contrast_times_twelve_endpoints",
        "paired_same_draws_both_arms": True,
        "within_panel_base_resampling": (
            "resample_six_components_within_each_of_four_base_HT_strata_"
            "using_identical_draws_and_coefficients_both_arms"
        ),
        "candidate_only_resampling_present": False,
        "whole_panel_block_resampling_used": False,
        "recompute_HT_arm_totals_exact_shared_denominator_coefficients_"
        "aggregate_and_all_nonlinear_endpoints_each_replicate": True,
        "persist_per_unit_scores_bootstrap_replicates_or_draws": False,
    }
    if (
        value.get("schema")
        != "eggroll-es-v341-matched-replacement-preregistration-v22a"
        or value.get("status")
        != "preregistered_offline_only_before_v22_runtime_no_v22_result_dependency"
        or value.get("experiment_name") != EXPERIMENT_NAME_V22A
        or value.get("scientific_objective") != (
            "attribute_exact_v341_manual_representative_changes_on_the_same_"
            "frozen_production_component_sample_without_candidate_only_expansion"
        )
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("inputs", {}).get("frame") != {
            "commit": FRAME_COMMIT_V22A,
            "path": str(frame_v22a.OUTPUT_PATH_V22A),
            "file_sha256": FRAME_CERTIFICATE_SHA256_V22A,
            "content_sha256": FRAME_CERTIFICATE_CONTENT_SHA256_V22A,
        }
        or value.get("inputs", {}).get("candidate") != {
            "source_commit": frame_v22a.candidate_v341.V341_SOURCE_COMMIT,
            "sealed_snapshot_commit": frame_v22a.CANDIDATE_SEAL_COMMIT_V22A,
            "path": str(frame_v22a.candidate_v341.OUTPUT_PATH_V341),
            "rows": frame_v22a.candidate_v341.V341_ROWS,
            "file_sha256": frame_v22a.candidate_v341.V341_SHA256,
            "manifest_path": str(frame_v22a.candidate_v341.MANIFEST_PATH_V341),
            "manifest_file_sha256": frame_v22a.CANDIDATE_MANIFEST_SHA256_V22A,
            "manifest_content_sha256": (
                frame_v22a.CANDIDATE_MANIFEST_CONTENT_SHA256_V22A
            ),
            "ongoing_curation_used": False,
        }
        or value.get("inputs", {}).get("production") != {
            "commit": frame_v22a.PRODUCTION_COMMIT_V22A,
            "path": str(frame_v22a.frame_v21a.frame_v18a.PRODUCTION_PATH_V18A),
            "rows": 784,
            "file_sha256": frame_v22a.PRODUCTION_SHA256_V22A,
        }
        or value.get("inputs", {}).get("union_failure_evidence") != {
            "commit": UNION_FAILURE_COMMIT_V22A,
            "path": str(UNION_FAILURE_EVIDENCE_PATH_V22A),
            "file_sha256": UNION_FAILURE_EVIDENCE_SHA256_V22A,
            "content_sha256": UNION_FAILURE_EVIDENCE_CONTENT_SHA256_V22A,
            "raw_scoring_authoritative": True,
            "union_scoring_authorized": False,
        }
        or correction != expected_correction
        or frame_contract != expected_frame_contract
        or arms != {
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
        or estimator != expected_estimator
        or recipe.get("model") != str(MODEL_PATH_V22A)
        or recipe.get("model_family") != "Qwen3.6-35B-A3B"
        or recipe.get("layers") != [20, 21, 22, 23]
        or recipe.get("layer_plan") != {
            "commit": LAYER_PLAN_COMMIT_V22A,
            "path": str(LAYER_PLAN_PATH_V22A),
            "file_sha256": LAYER_PLAN_FILE_SHA256_V22A,
            "plan_sha256": LAYER_PLAN_SHA256_V22A,
            "model_config_sha256": MODEL_CONFIG_SHA256_V22A,
        }
        or recipe.get("sigma") != 0.0003
        or recipe.get("alpha") != 0.0
        or recipe.get("population_size") != 64
        or recipe.get("perturbation_basis") != validate_perturbation_basis_v22a()
        or recipe.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V22A
        or recipe.get("perturbation_seed_list_sha256")
        != PERTURBATION_SEED_LIST_SHA256_V22A
        or recipe.get("prior_basis_content_sha256")
        != PRIOR_BASIS_CONTENT_SHA256
        or recipe.get("prior_v18_v19_v20_v21_basis_reuse_allowed") is not False
        or recipe.get("basis_generator") != (
            "numpy.default_rng(basis_seed).integers(0,2**30,size=64,dtype=int64)"
        )
        or recipe.get("basis_generation_or_selection_at_runtime_allowed") is not False
        or recipe.get("hardware") != {
            "engine_count": 4,
            "tensor_parallel_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_four_engines_every_resident_signed_wave": True,
        }
        or recipe.get("moe_backend") != {
            "backend": "default_triton",
            "all_backend_override_environment_variables_must_be_unset": True,
            "v16_task_ab_decision_retained": "default_triton",
        }
        or recipe.get("model_update_allowed") is not False
        or recipe.get("checkpoint_write_allowed") is not False
        or recipe.get("evaluation_surfaces_opened") is not False
        or terms != {
            "antithetic_direction_count": 64,
            "signs_per_direction": 2,
            "signed_direction_evaluation_count": 128,
            "directions_per_resident_signed_wave": 4,
            "resident_signed_wave_count": 32,
            "engine_count": 4,
            "identity": (
                "32 resident signed waves x 4 engines = 128 signed direction evaluations"
            ),
        }
        or raw.get("population_wave_count") != 16
        or raw.get("resident_signed_wave_schedule") != signed_wave_schedule_v22a()
        or raw.get("requests_per_engine_per_resident_signed_wave") != 480
        or raw.get("requests_by_arm_per_engine_per_resident_signed_wave") != {
            "production_control": 240,
            "v341_matched_replacement": 240,
        }
        or raw.get("requests_per_engine_all_resident_signed_waves") != 15_360
        or raw.get("requests_all_engines_all_resident_signed_waves") != 61_440
        or raw.get("dense_result_commitment_count") != 2_560
        or raw.get("signed_direction_evaluations_all_engines") != 128
        or raw.get("same_resident_perturbation_scores_both_arms_before_restore")
        is not True
        or raw.get("same_perturbation_basis_and_direction_for_both_arms") is not True
        or raw.get("arm_order_balanced_within_each_sign") is not True
        or raw.get("each_arm_appears_in_each_position_eight_times_per_sign") is not True
        or raw.get("exact_restore_and_verify_once_after_both_arms_each_wave")
        is not True
        or raw.get("partial_waves_allowed") is not False
        or raw.get("raw_arm_scoring_only") is not True
        or raw.get("union_scoring_authorized") is not False
        or raw.get("union_planner_called") is not False
        or value.get("scoring") != _scoring_contract_v22a()
        or analysis.get("contrast") != {
            "name": "v341_matched_replacement_vs_production",
            "treatment": "v341_matched_replacement",
            "control": "production_control",
            "treatment_minus_control": True,
        }
        or analysis.get("endpoint_names") != list(ENDPOINT_NAMES_V22A)
        or analysis.get("endpoint_count") != 12
        or analysis.get("hypothesis_count") != 12
        or analysis.get("panel_endpoint_geometry") != {
            "optimization_pairwise_pairs": 15,
            "aggregate_to_optimization_panels": 6,
            "train_only_screen_panels": 4,
            "median_and_worst_for_each_metric_family": True,
        }
        or bootstrap != expected_bootstrap
        or gate != {
            "observed_treatment_minus_control_nonnegative_all_twelve": True,
            "zero_margin_familywise_lcb_nonnegative_all_twelve": True,
            "noninferiority_margin": 0.0,
            "all_panel_spreads_nonzero": True,
            "all_runtime_identity_restoration_boundary_backend_and_raw_audits": True,
            "no_pass_decision": "retain_production_dataset_and_v13_recipe",
            "pass_decision": (
                "authorize_only_a_separate_fresh_basis_train_only_confirmation_"
                "preregistration"
            ),
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        }
        or prior.get(
            "v21_gate_decision_motivated_excluding_candidate_only_expansion"
        ) is not True
        or prior.get("v21_numeric_endpoint_values_used_to_tune_v22_recipe_or_gate")
        is not False
        or prior.get("v22_runtime_result_observed") is not False
        or prior.get("ongoing_curation_after_v341_source_commit_used") is not False
        or firewall != {
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
    ):
        raise RuntimeError("v22a exact-v341 matched-replacement preregistration changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v22a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V22A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V22A:
        raise ValueError("v22a preregistration output path changed")
    value = build_preregistration_v22a()
    _exclusive_write(OUTPUT_PATH_V22A, value)
    result = {
        "schema": "eggroll-es-v341-matched-replacement-preregistration-build-v22a",
        "output": str(OUTPUT_PATH_V22A),
        "file_sha256": file_sha256(OUTPUT_PATH_V22A),
        "content_sha256": value["content_sha256_before_self_field"],
        "antithetic_direction_count": 64,
        "signed_direction_evaluation_count": 128,
        "resident_signed_wave_count": 32,
        "requests_per_engine_per_resident_signed_wave": 480,
        "runtime_launch_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
