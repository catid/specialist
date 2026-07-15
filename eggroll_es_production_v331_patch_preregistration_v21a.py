#!/usr/bin/env python3
"""Offline-only V21A production-plus-v331 patch compatibility preregistration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_production_v331_patch_frame_v21a as frame_v21a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V21A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_PRODUCTION_V331_PATCH_COMPATIBILITY_V21A_PREREGISTRATION.json"
).resolve()
FRAME_COMMIT_V21A = "32bd189372a14b0c7db155d8986dcc6d4928b93b"
UNION_FAILURE_COMMIT_V21A = "a276bc8b982a04a608535a7c9ccda7e9dc77913f"
LAYER_PLAN_COMMIT_V21A = "a4cfdb6726d4381f5d280b47ca4866569c35bd3b"

FRAME_BUILDER_SHA256_V21A = (
    "31a39322e325ff29007824d7b4165de67a34b5ac5fd88935e5991dfb2e3cb0b6"
)
FRAME_TEST_SHA256_V21A = (
    "6f212f5996fa21603a006b34a261318a1e24e0347204d1a1b3357a23afe8fc81"
)
FRAME_CERTIFICATE_SHA256_V21A = (
    "9dea3ff4eb970087a17daf2dbbeb1a0f49985330b515160fcb0c0db87216fee9"
)
FRAME_CERTIFICATE_CONTENT_SHA256_V21A = (
    "59e3352aabe851e31cf9e5ee47559051373f397d7354d3125ace821c65b118c6"
)
UNION_FAILURE_EVIDENCE_PATH_V21A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V20A_UNION_EQUIVALENCE_FAILURE_EVIDENCE.json"
).resolve()
UNION_FAILURE_EVIDENCE_SHA256_V21A = (
    "d84fbd7b373b284f3c6a2ffbf1bd52431818ec4e6c60b4f3b320d36b581c9021"
)
UNION_FAILURE_EVIDENCE_CONTENT_SHA256_V21A = (
    "2caa4e1658ae794a3193ef28bb3393b966f1d329d574bc8d0b2534e3fb1f3302"
)
LAYER_PLAN_PATH_V21A = (
    ROOT / "experiments/layer_plans/middle_late_dense_v6.json"
).resolve()
LAYER_PLAN_FILE_SHA256_V21A = (
    "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df"
)
LAYER_PLAN_SHA256_V21A = (
    "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9"
)
MODEL_CONFIG_SHA256_V21A = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
MODEL_PATH_V21A = (ROOT / "models/Qwen3.6-35B-A3B").resolve()

EXPERIMENT_NAME_V21A = "production_v331_full_merged_patch_compatibility_v21a"
ENDPOINT_NAMES_V21A = (
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
FRESH_PERTURBATION_BASIS_SEED_V21A = 20260821
PERTURBATION_SEEDS_V21A = [
    415818684, 121142923, 313141107, 842241830,
    1001705657, 656692085, 497299778, 312345315,
    736130089, 597350828, 996298348, 652018116,
    449462007, 1058657574, 639434074, 270290775,
    189489956, 643072826, 144971429, 493554946,
    667043242, 654490629, 500304739, 234981491,
    520229966, 342772345, 685911883, 753401212,
    369356405, 136167865, 1060842542, 763347154,
    587678018, 879574783, 253256859, 620671297,
    149580036, 525705663, 893800412, 1046426576,
    918951458, 497856027, 714126995, 592400187,
    354778584, 716916757, 34757589, 1042511358,
    651607243, 1072171037, 959939420, 510760145,
    537017862, 493649905, 938646516, 128457608,
    409938263, 902812513, 877928459, 139391271,
    713871640, 618310447, 347662803, 742131867,
]
PERTURBATION_SEED_LIST_SHA256_V21A = (
    "b8456790fa704e10a50e332bc22bfeb7f981bdfa40c206494dcab8df2f1e9062"
)
PERTURBATION_BASIS_SHA256_V21A = (
    "65970861cd06b53e52cf848b2c8b8961160bf9c68f6b1b9f4935a88ba8d314d2"
)
PRIOR_BASIS_CONTENT_SHA256 = {
    "v18a": "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11",
    "v19a": "d4e46d7d51d5c82cfc981dad3b33db8a1766c70ad570ef931b12550d1bc7bf6c",
    "v20a": "b6d667c2f125f9d0be4d74ef536af03546fecb6c03f2838679f5a315a1ec9852",
}
BOOTSTRAP_SEED_V21A = 20260822
BOOTSTRAP_REPETITIONS_V21A = 50_000
FAMILYWISE_ALPHA_V21A = 0.05
HYPOTHESIS_COUNT_V21A = 12

canonical_sha256 = frame_v21a.canonical_sha256
file_sha256 = frame_v21a.file_sha256


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
        raise RuntimeError(f"v21a preregistered artifact changed: {relative}")


def perturbation_basis_v21a() -> dict:
    return {
        "schema": "eggroll-es-production-v331-patch-perturbation-basis-v21a",
        "basis_seed": FRESH_PERTURBATION_BASIS_SEED_V21A,
        "population_size": 64,
        "seeds": list(PERTURBATION_SEEDS_V21A),
    }


def validate_perturbation_basis_v21a() -> dict:
    regenerated = np.random.default_rng(
        seed=FRESH_PERTURBATION_BASIS_SEED_V21A
    ).integers(0, 2**30, size=64, dtype=np.int64).tolist()
    basis = perturbation_basis_v21a()
    if (
        regenerated != PERTURBATION_SEEDS_V21A
        or len(set(regenerated)) != 64
        or canonical_sha256(regenerated) != PERTURBATION_SEED_LIST_SHA256_V21A
        or canonical_sha256(basis) != PERTURBATION_BASIS_SHA256_V21A
        or PERTURBATION_BASIS_SHA256_V21A
        in set(PRIOR_BASIS_CONTENT_SHA256.values())
    ):
        raise RuntimeError("v21a fresh 64-direction perturbation basis changed")
    return basis


def signed_wave_schedule_v21a() -> list[dict]:
    """Return 32 four-engine resident waves covering 64 antithetic directions."""
    schedule = []
    arms = list(frame_v21a.ARM_ORDER_V21A)
    for population_wave, start in enumerate(range(0, 64, 4)):
        seeds = PERTURBATION_SEEDS_V21A[start:start + 4]
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
        raise RuntimeError("v21a resident signed-wave schedule changed")
    for sign in ("plus", "minus"):
        signed = [item for item in schedule if item["sign"] == sign]
        for arm in arms:
            if sorted(item["resident_arm_order"].index(arm) for item in signed) != (
                [0] * 8 + [1] * 8
            ):
                raise RuntimeError("v21a two-arm signed-wave balance changed")
    return schedule


def load_bound_inputs_v21a() -> tuple[dict, dict, dict]:
    for path, commit, digest in (
        (Path(frame_v21a.__file__), FRAME_COMMIT_V21A, FRAME_BUILDER_SHA256_V21A),
        (
            ROOT / "test_build_eggroll_es_production_v331_patch_frame_v21a.py",
            FRAME_COMMIT_V21A,
            FRAME_TEST_SHA256_V21A,
        ),
        (
            frame_v21a.OUTPUT_PATH_V21A,
            FRAME_COMMIT_V21A,
            FRAME_CERTIFICATE_SHA256_V21A,
        ),
        (
            UNION_FAILURE_EVIDENCE_PATH_V21A,
            UNION_FAILURE_COMMIT_V21A,
            UNION_FAILURE_EVIDENCE_SHA256_V21A,
        ),
        (
            LAYER_PLAN_PATH_V21A,
            LAYER_PLAN_COMMIT_V21A,
            LAYER_PLAN_FILE_SHA256_V21A,
        ),
    ):
        _verify_commit_file(path, commit, digest)
    frame = json.loads(frame_v21a.OUTPUT_PATH_V21A.read_text())
    frame_v21a.validate_certificate_v21a(frame)
    evidence = json.loads(UNION_FAILURE_EVIDENCE_PATH_V21A.read_text())
    layer_plan = json.loads(LAYER_PLAN_PATH_V21A.read_text())
    if (
        frame["content_sha256_before_self_field"]
        != FRAME_CERTIFICATE_CONTENT_SHA256_V21A
        or evidence.get("schema")
        != "eggroll-es-union-equivalence-failure-evidence-v20a"
        or evidence.get("content_sha256_before_self_field")
        != UNION_FAILURE_EVIDENCE_CONTENT_SHA256_V21A
        or evidence.get("decision", {}).get(
            "raw_arm_scoring_remains_authoritative"
        ) is not True
        or evidence.get("decision", {}).get(
            "union_scoring_authorized_for_v20a"
        ) is not False
        or layer_plan.get("plan_sha256") != LAYER_PLAN_SHA256_V21A
        or layer_plan.get("model_config_sha256") != MODEL_CONFIG_SHA256_V21A
        or layer_plan.get("layers") != [20, 21, 22, 23]
        or layer_plan.get("num_units") != 35
    ):
        raise RuntimeError("v21a preregistered input content changed")
    return frame, evidence, layer_plan


def build_preregistration_v21a() -> dict:
    frame, evidence, layer_plan = load_bound_inputs_v21a()
    perturbation_basis = validate_perturbation_basis_v21a()
    schedule = signed_wave_schedule_v21a()
    frame_overlap = frame["candidate_only_assignment"]["cross_role_overlap"]
    explicit_overlap = {
        topic: frame_overlap["topic_counts"].get(topic, 0)
        for topic in frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
    }
    if explicit_overlap != {
        "safety_consent": 2,
        "technique": 2,
        "equipment_material": 0,
        "resources_general": 2,
    }:
        raise RuntimeError("v21a explicit zero-equipment overlap convention changed")
    value = {
        "schema": "eggroll-es-production-v331-patch-preregistration-v21a",
        "status": "preregistered_offline_only_no_runtime_or_result_dependency",
        "experiment_name": EXPERIMENT_NAME_V21A,
        "question": (
            "Does adding the complete conflict-unit-deduplicated v331-only patch "
            "to the frozen production dataset preserve or improve all twelve "
            "train-only V13 stability endpoints under paired raw ES scoring?"
        ),
        "inputs": {
            "frame": {
                "commit": FRAME_COMMIT_V21A,
                "path": str(frame_v21a.OUTPUT_PATH_V21A),
                "file_sha256": FRAME_CERTIFICATE_SHA256_V21A,
                "content_sha256": FRAME_CERTIFICATE_CONTENT_SHA256_V21A,
            },
            "candidate": copy.deepcopy(frame["inputs"]["candidate_v331"]),
            "production": copy.deepcopy(frame["inputs"]["production"]),
            "union_failure_evidence": {
                "commit": UNION_FAILURE_COMMIT_V21A,
                "path": str(UNION_FAILURE_EVIDENCE_PATH_V21A),
                "file_sha256": UNION_FAILURE_EVIDENCE_SHA256_V21A,
                "content_sha256": UNION_FAILURE_EVIDENCE_CONTENT_SHA256_V21A,
                "raw_scoring_authoritative": evidence["decision"][
                    "raw_arm_scoring_remains_authoritative"
                ],
                "union_scoring_authorized": False,
            },
        },
        "frame_contract": {
            "panel_names": list(frame_v21a.PANEL_NAMES_V21A),
            "optimization_panels": list(frame_v21a.OPTIMIZATION_PANELS_V21A),
            "train_only_screen_panels": list(frame_v21a.TRAIN_SCREEN_PANELS_V21A),
            "globally_disjoint_production_base_components": 240,
            "base_requests_per_panel": 24,
            "base_category_quota_per_panel": {
                category: 6 for category in frame_v21a.BASE_CATEGORIES_V21A
            },
            "production_topic_quota_per_panel": copy.deepcopy(
                frame_v21a.PRODUCTION_TOPIC_QUOTAS_V21A
            ),
            "candidate_only_topic_population": copy.deepcopy(
                frame_v21a.CANDIDATE_ONLY_TOPIC_POPULATIONS_V21A
            ),
            "candidate_only_topic_quota_per_panel": copy.deepcopy(
                frame_v21a.CANDIDATE_ONLY_TOPIC_QUOTAS_V21A
            ),
            "candidate_only_requests_per_panel": 6,
            "candidate_only_global_unique_components": 54,
            "candidate_only_total_assignments": 60,
            "candidate_only_reuses_after_exhaustion": 6,
            "optimization_assignments_all_unique": 36,
            "train_screen_assignments_all_unique": 24,
            "cross_role_overlap_count": 6,
            "cross_role_overlap_by_topic_explicit_zero": explicit_overlap,
            "full_merge_keeps_all_sampled_production_representatives": True,
            "paired_candidate_replacements": 0,
        },
        "arms": {
            "production_only": {
                "requests_per_panel": 24,
                "requests_per_engine": 240,
                "population_denominator": 272,
            },
            "production_plus_v331_patch": {
                "requests_per_panel": 30,
                "requests_per_engine": 300,
                "population_denominator": 326,
            },
        },
        "estimator": {
            "base_ht_strata": copy.deepcopy(
                frame["estimand"]["base_population_ht_strata"]
            ),
            "candidate_only_ht_strata": copy.deepcopy(
                frame["estimand"]["candidate_only_ht_strata"]
            ),
            "plain_request_mean_used": False,
            "full_merge_is_additive_not_candidate_substitution": True,
            "same_component_counted_once_per_arm": True,
        },
        "frozen_recipe": {
            "model": str(MODEL_PATH_V21A),
            "model_family": "Qwen3.6-35B-A3B",
            "layers": [20, 21, 22, 23],
            "layer_plan": {
                "commit": LAYER_PLAN_COMMIT_V21A,
                "path": str(LAYER_PLAN_PATH_V21A),
                "file_sha256": LAYER_PLAN_FILE_SHA256_V21A,
                "plan_sha256": LAYER_PLAN_SHA256_V21A,
                "model_config_sha256": MODEL_CONFIG_SHA256_V21A,
            },
            "sigma": 0.0003,
            "alpha": 0.0,
            "population_size": 64,
            "perturbation_basis": perturbation_basis,
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V21A,
            "perturbation_seed_list_sha256": (
                PERTURBATION_SEED_LIST_SHA256_V21A
            ),
            "prior_basis_content_sha256": PRIOR_BASIS_CONTENT_SHA256,
            "prior_v18_v19_v20_basis_reuse_allowed": False,
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
                "identity": "32 resident signed waves x 4 engines = 128 signed direction evaluations",
            },
            "population_wave_count": 16,
            "resident_signed_wave_schedule": schedule,
            "same_resident_perturbation_scores_both_arms_before_restore": True,
            "same_perturbation_basis_and_direction_for_both_arms": True,
            "arm_order_balanced_within_each_sign": True,
            "each_arm_appears_in_each_position_eight_times_per_sign": True,
            "exact_restore_and_verify_once_after_both_arms_each_wave": True,
            "partial_waves_allowed": False,
            "requests_per_engine_per_resident_signed_wave": 540,
            "requests_by_arm_per_engine_per_resident_signed_wave": {
                "production_only": 240,
                "production_plus_v331_patch": 300,
            },
            "requests_per_engine_all_resident_signed_waves": 17_280,
            "signed_direction_evaluations_all_engines": 128,
            "dense_result_commitment_count": 2_560,
            "raw_arm_scoring_only": True,
            "union_scoring_authorized": False,
            "union_planner_called": False,
        },
        "scoring": {
            "objective": "per_example_mean_full_gold_answer_token_logprob",
            "prompt_contains_full_gold_answer": True,
            "prompt_logprobs": 1,
            "scored_positions": "all_aligned_answer_tokens_only",
            "max_tokens": 1,
            "max_tokens_role": "dummy_generation_trigger_not_answer_cap",
            "frozen_max_total_prompt_answer_tokens": 1024,
            "detokenize": False,
            "eos_scored": False,
            "autoregressive_accuracy_or_token_f1_used": False,
            "objective_change_allowed": False,
            "v331_token_boundary_audit_required_before_any_future_runtime": True,
            "this_offline_preregistration_authorizes_runtime": False,
        },
        "analysis": {
            "contrast": {
                "name": "production_plus_v331_patch_vs_production",
                "treatment": "production_plus_v331_patch",
                "control": "production_only",
                "treatment_minus_control": True,
            },
            "endpoint_names": list(ENDPOINT_NAMES_V21A),
            "endpoint_count": 12,
            "hypothesis_count": 12,
            "panel_endpoint_geometry": {
                "optimization_pairwise_pairs": 15,
                "aggregate_to_optimization_panels": 6,
                "train_only_screen_panels": 4,
                "median_and_worst_for_each_metric_family": True,
            },
            "bootstrap": {
                "seed": BOOTSTRAP_SEED_V21A,
                "repetitions": BOOTSTRAP_REPETITIONS_V21A,
                "familywise_alpha": FAMILYWISE_ALPHA_V21A,
                "one_sided_quantile": FAMILYWISE_ALPHA_V21A / 12,
                "multiplicity": "Bonferroni_over_one_contrast_times_twelve_endpoints",
                "paired_same_draws_both_arms": True,
                "within_panel_base_resampling": (
                    "resample_six_components_within_each_of_four_base_HT_strata_"
                    "using_shared_draws_both_arms"
                ),
                "candidate_only_resampling": (
                    "within_each_fixed_panel_role_and_topic_resample_observed_"
                    "candidate_only_slots_with_fixed_draws_for_merged_arm"
                ),
                "whole_panel_block_resampling_used": False,
                "recompute_HT_arm_totals_exact_denominators_coefficients_"
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
        "independence": {
            "v20_runtime_result_read_or_used": False,
            "v20_runtime_result_can_change_question_recipe_or_gate": False,
            "ongoing_curation_after_source_commit_used": False,
        },
        "firewall": {
            "offline_frame_and_preregistration_only": True,
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
    return validate_preregistration_v21a(value)


def validate_preregistration_v21a(value: dict) -> dict:
    recipe = value.get("frozen_recipe", {})
    raw = value.get("paired_raw_schedule", {})
    terms = raw.get("terminology", {})
    analysis = value.get("analysis", {})
    bootstrap = analysis.get("bootstrap", {})
    frame_contract = value.get("frame_contract", {})
    firewall = value.get("firewall", {})
    if (
        value.get("schema")
        != "eggroll-es-production-v331-patch-preregistration-v21a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("inputs", {}).get("frame", {}).get("commit")
        != FRAME_COMMIT_V21A
        or value.get("inputs", {}).get("frame", {}).get("content_sha256")
        != FRAME_CERTIFICATE_CONTENT_SHA256_V21A
        or frame_contract.get("train_screen_assignments_all_unique") != 24
        or frame_contract.get("cross_role_overlap_count") != 6
        or frame_contract.get("cross_role_overlap_by_topic_explicit_zero") != {
            "safety_consent": 2,
            "technique": 2,
            "equipment_material": 0,
            "resources_general": 2,
        }
        or frame_contract.get("paired_candidate_replacements") != 0
        or recipe.get("model_family") != "Qwen3.6-35B-A3B"
        or recipe.get("layers") != [20, 21, 22, 23]
        or recipe.get("sigma") != 0.0003
        or recipe.get("alpha") != 0.0
        or recipe.get("population_size") != 64
        or recipe.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V21A
        or recipe.get("perturbation_seed_list_sha256")
        != PERTURBATION_SEED_LIST_SHA256_V21A
        or recipe.get("prior_v18_v19_v20_basis_reuse_allowed") is not False
        or recipe.get("hardware") != {
            "engine_count": 4,
            "tensor_parallel_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "all_four_engines_every_resident_signed_wave": True,
        }
        or terms != {
            "antithetic_direction_count": 64,
            "signs_per_direction": 2,
            "signed_direction_evaluation_count": 128,
            "directions_per_resident_signed_wave": 4,
            "resident_signed_wave_count": 32,
            "engine_count": 4,
            "identity": "32 resident signed waves x 4 engines = 128 signed direction evaluations",
        }
        or raw.get("population_wave_count") != 16
        or raw.get("resident_signed_wave_schedule") != signed_wave_schedule_v21a()
        or raw.get("requests_per_engine_per_resident_signed_wave") != 540
        or raw.get("requests_per_engine_all_resident_signed_waves") != 17_280
        or raw.get("dense_result_commitment_count") != 2_560
        or raw.get("same_perturbation_basis_and_direction_for_both_arms") is not True
        or raw.get("raw_arm_scoring_only") is not True
        or raw.get("union_scoring_authorized") is not False
        or raw.get("union_planner_called") is not False
        or analysis.get("endpoint_names") != list(ENDPOINT_NAMES_V21A)
        or analysis.get("endpoint_count") != 12
        or analysis.get("hypothesis_count") != 12
        or bootstrap.get("repetitions") != 50_000
        or bootstrap.get("one_sided_quantile") != 0.05 / 12
        or bootstrap.get("paired_same_draws_both_arms") is not True
        or bootstrap.get("whole_panel_block_resampling_used") is not False
        or value.get("compatibility_gate", {}).get("noninferiority_margin") != 0.0
        or value.get("independence", {}).get("v20_runtime_result_read_or_used")
        is not False
        or firewall.get("offline_frame_and_preregistration_only") is not True
        or firewall.get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
        or any(firewall.get(key) is not False for key in (
            "gpu_launch_authorized", "runtime_launch_authorized",
            "model_update_authorized", "checkpoint_write_authorized",
            "evaluation_authorized", "dataset_promotion_authorized",
        ))
    ):
        raise RuntimeError("v21a production-v331 patch preregistration changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v21a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V21A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V21A:
        raise ValueError("v21a preregistration output path changed")
    value = build_preregistration_v21a()
    _exclusive_write(OUTPUT_PATH_V21A, value)
    result = {
        "schema": "eggroll-es-production-v331-patch-preregistration-build-v21a",
        "output": str(OUTPUT_PATH_V21A),
        "file_sha256": file_sha256(OUTPUT_PATH_V21A),
        "content_sha256": value["content_sha256_before_self_field"],
        "antithetic_direction_count": 64,
        "signed_direction_evaluation_count": 128,
        "resident_signed_wave_count": 32,
        "runtime_launch_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
