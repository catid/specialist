#!/usr/bin/env python3
"""Preregister the train-only V19A disjoint-third attribution experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_disjoint_tier_frame_v19a as frame_v19a
import eggroll_es_production_overlay_scan_preregistration_v18a as prereg_v18a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V19A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_PRODUCTION_PATCH_DISJOINT_TIER_ATTRIBUTION_V19A_PREREGISTRATION.json"
).resolve()
V18A_NEGATIVE_EVIDENCE_PATH_V19A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V18A_PRODUCTION_PATCH_COMPAT_NEGATIVE_EVIDENCE.json"
).resolve()
V18A_NEGATIVE_EVIDENCE_BUILDER_PATH_V19A = (
    ROOT / "build_eggroll_es_v18a_compact_evidence.py"
).resolve()
V18A_PREREGISTRATION_PATH_V19A = prereg_v18a.OUTPUT_PATH_V18A

EXPERIMENT_NAME_V19A = (
    "snapshot_v298_production_patch_v19a_disjoint_thirds_"
    "10x24_plus1_alpha_zero_middle_late_fresh_basis"
)
V18A_NEGATIVE_EVIDENCE_COMMIT_V19A = (
    "f4fc90fe28e05d9ee4c94e98ce25419d6336ff87"
)
V18A_NEGATIVE_EVIDENCE_FILE_SHA256_V19A = (
    "135bfc1b3bbfa4a6f8605f758e2b84005bf574f525291750ba635db75cc4eb1b"
)
V18A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V19A = (
    "9fc09b8125e7c945d75b5f2c60d06961a320dc8802e033838b6efa57d29f0d42"
)
V18A_NEGATIVE_EVIDENCE_BUILDER_SHA256_V19A = (
    "2c8bffd316b07e53c481e5f4ad9d44132b10d21da891746878a1e400c668c427"
)
V18A_PREREGISTRATION_FILE_SHA256_V19A = (
    "deb3522b5bcb32639702390450344b0e71f0820d418b843c86bfcaea21431b85"
)
V18A_PREREGISTRATION_CONTENT_SHA256_V19A = (
    "72509385d9df9fe2ce36a81c065dae995268cfcb3708eb10cd76ce39f9e77151"
)
V18A_PREREGISTRATION_BUILDER_SHA256_V19A = (
    "683dace9ae2328dcf0d8c79e5a5b69ecdcc62dc55586f64376e4dc8fdcb4819f"
)
FRAME_BUILDER_FILE_SHA256_V19A = (
    "4de43c263f580143d1b1f09a67240b8a05df7d45b419b19a211b549bf85b2052"
)
FRAME_CERTIFICATE_FILE_SHA256_V19A = (
    "50820c67844a7b11c92bc0bbaa9c594c683440a65cb882de244216a77dca5fed"
)
FRAME_CERTIFICATE_CONTENT_SHA256_V19A = (
    "7ad195a55b1e51268dfba1cddb43f869b014bdb1ef329f5d73c8246ac6cbff58"
)
TOKEN_AUDIT_BUILDER_FILE_SHA256_V19A = (
    "9ff344ce001673f21a3782c813f2545f7503c19f8f0cc6be7d57ae64bfc8e7f3"
)
TOKEN_AUDIT_FILE_SHA256_V19A = (
    "df1d3810e988c3ece4ef921643ffe226fa7bb7f2f91edf2895865afe78c7ee6f"
)
TOKEN_AUDIT_CONTENT_SHA256_V19A = (
    "8157f64794cfe34f50dc795bed338f4109aa84addb38f114932b95f3d7598329"
)

ARM_ORDER_V19A = frame_v19a.ARM_ORDER_V19A
ENDPOINT_NAMES_V19A = tuple(prereg_v18a.ENDPOINT_NAMES_V18A)
HYPOTHESIS_COUNT_V19A = 3 * len(ENDPOINT_NAMES_V19A)
FRESH_PERTURBATION_BASIS_SEED_V19A = 20260727
PERTURBATION_SEEDS_V19A = [
    1023521392, 162576173, 257459565, 1058358553,
    442126759, 904366571, 1058488684, 513115919,
    325573687, 428094410, 582904204, 368781717,
    876647537, 719888447, 1022025974, 419559184,
    410765061, 1049554684, 823900113, 817367952,
    975639498, 534598256, 771610715, 406720095,
    293702403, 673324180, 537324430, 619320720,
    470707518, 961353, 744189923, 521892147,
]
PERTURBATION_SEED_LIST_SHA256_V19A = (
    "05ab5fd8732861b12dd351d74c8f83318fb50810ebb1d497f4e69500ef12568f"
)
PERTURBATION_BASIS_SHA256_V19A = (
    "d4e46d7d51d5c82cfc981dad3b33db8a1766c70ad570ef931b12550d1bc7bf6c"
)
BOOTSTRAP_SEED_V19A = 20260728
BOOTSTRAP_REPETITIONS_V19A = 50_000
FAMILYWISE_ALPHA_V19A = 0.05

canonical_sha256 = prereg_v18a.canonical_sha256
file_sha256 = prereg_v18a.file_sha256


def _without_self(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_commit_file(path: Path, commit: str, digest: str) -> None:
    relative = Path(path).resolve().relative_to(ROOT).as_posix()
    raw = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT
    )
    if hashlib.sha256(raw).hexdigest() != digest or file_sha256(path) != digest:
        raise RuntimeError(f"v19a committed evidence binding changed: {relative}")


def _validate_v18a_negative_evidence(value: dict) -> dict:
    if (
        value.get("schema")
        != "eggroll-es-production-patch-compat-negative-evidence-v18a"
        or value.get("status")
        != "valid_completed_run_preregistered_gate_failed"
        or value.get("content_sha256_before_self_field")
        != V18A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V19A
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("gate_summary", {}).get("any_patch_passed") is not False
        or value.get("gate_summary", {}).get("bootstrap_pass_counts")
        != {
            "patch_one_third": 0,
            "patch_two_thirds": 0,
            "patch_full": 0,
        }
        or value.get("gate_summary", {}).get("observed_pass_counts")
        != {
            "patch_one_third": 5,
            "patch_two_thirds": 8,
            "patch_full": 7,
        }
        or value.get("decision", {}).get("retain_dataset") != "production"
        or value.get("decision", {}).get("retain_recipe")
        != "v13_middle_late_layers_20_23"
        or value.get("decision", {}).get("dataset_promotion_authorized")
        is not False
        or value.get("decision", {}).get("model_update_authorized") is not False
        or value.get("decision", {}).get("evaluation_authorized") is not False
        or value.get("contains_response_vectors_or_row_content") is not False
        or value.get("contains_validation_ood_heldout_or_benchmark_content")
        is not False
    ):
        raise RuntimeError("v19a V18A negative evidence binding changed")
    return value


def load_bound_inputs_v19a() -> tuple[dict, dict, dict, dict]:
    expected = {
        Path(frame_v19a.__file__).resolve(): FRAME_BUILDER_FILE_SHA256_V19A,
        frame_v19a.OUTPUT_PATH_V19A: FRAME_CERTIFICATE_FILE_SHA256_V19A,
        Path(prereg_v18a.__file__).resolve(): (
            V18A_PREREGISTRATION_BUILDER_SHA256_V19A
        ),
        V18A_PREREGISTRATION_PATH_V19A: (
            V18A_PREREGISTRATION_FILE_SHA256_V19A
        ),
        V18A_NEGATIVE_EVIDENCE_BUILDER_PATH_V19A: (
            V18A_NEGATIVE_EVIDENCE_BUILDER_SHA256_V19A
        ),
        V18A_NEGATIVE_EVIDENCE_PATH_V19A: (
            V18A_NEGATIVE_EVIDENCE_FILE_SHA256_V19A
        ),
        Path(prereg_v18a.token_audit_v18a.__file__).resolve(): (
            TOKEN_AUDIT_BUILDER_FILE_SHA256_V19A
        ),
        prereg_v18a.token_audit_v18a.OUTPUT_PATH_V18A: (
            TOKEN_AUDIT_FILE_SHA256_V19A
        ),
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v19a immutable train-only artifact binding changed")
    _verify_commit_file(
        V18A_NEGATIVE_EVIDENCE_PATH_V19A,
        V18A_NEGATIVE_EVIDENCE_COMMIT_V19A,
        V18A_NEGATIVE_EVIDENCE_FILE_SHA256_V19A,
    )
    _verify_commit_file(
        V18A_NEGATIVE_EVIDENCE_BUILDER_PATH_V19A,
        V18A_NEGATIVE_EVIDENCE_COMMIT_V19A,
        V18A_NEGATIVE_EVIDENCE_BUILDER_SHA256_V19A,
    )
    flow = json.loads(frame_v19a.OUTPUT_PATH_V19A.read_text(encoding="utf-8"))
    frame_v19a.validate_certificate_v19a(flow)
    evidence = _validate_v18a_negative_evidence(json.loads(
        V18A_NEGATIVE_EVIDENCE_PATH_V19A.read_text(encoding="utf-8")
    ))
    v18a_preregistration = json.loads(
        V18A_PREREGISTRATION_PATH_V19A.read_text(encoding="utf-8")
    )
    token_audit = json.loads(
        prereg_v18a.token_audit_v18a.OUTPUT_PATH_V18A.read_text(
            encoding="utf-8"
        )
    )
    prereg_v18a.token_audit_v18a.validate_audit_v18a(token_audit)
    if (
        flow.get("content_sha256_before_self_field")
        != FRAME_CERTIFICATE_CONTENT_SHA256_V19A
        or v18a_preregistration.get("content_sha256_before_self_field")
        != V18A_PREREGISTRATION_CONTENT_SHA256_V19A
        or v18a_preregistration.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(v18a_preregistration))
        or token_audit.get("content_sha256_before_self_field")
        != TOKEN_AUDIT_CONTENT_SHA256_V19A
    ):
        raise RuntimeError("v19a bound artifact content changed")
    return flow, evidence, v18a_preregistration, token_audit


def signed_wave_arm_orders_v19a() -> list[dict]:
    arms = list(ARM_ORDER_V19A)
    orders = []
    for population_wave in range(8):
        for sign_index, sign in enumerate(("plus", "minus")):
            rotation = (
                population_wave if sign == "plus" else population_wave + 2
            ) % len(arms)
            order = arms[rotation:] + arms[:rotation]
            orders.append({
                "signed_wave_index": 2 * population_wave + sign_index,
                "population_wave_index": population_wave,
                "sign": sign,
                "arm_order": order,
            })
    for sign in ("plus", "minus"):
        signed = [entry for entry in orders if entry["sign"] == sign]
        for arm in arms:
            positions = [entry["arm_order"].index(arm) for entry in signed]
            if sorted(positions) != [0, 0, 1, 1, 2, 2, 3, 3]:
                raise RuntimeError("v19a signed arm-position balance changed")
    return orders


def _scoring_contract_v19a(v18a_preregistration: dict) -> dict:
    scoring = copy.deepcopy(v18a_preregistration["scoring"])
    if scoring.pop("objective_change_allowed_in_v18a", None) is not False:
        raise RuntimeError("v19a inherited scoring contract changed")
    scoring["objective_change_allowed_in_v19a"] = False
    scoring["frozen_max_total_tokens"] = 1024
    return scoring


def perturbation_basis_v19a() -> dict:
    return {
        "schema": "eggroll-es-disjoint-tier-perturbation-basis-v19a",
        "basis_seed": FRESH_PERTURBATION_BASIS_SEED_V19A,
        "population_size": 32,
        "seeds": list(PERTURBATION_SEEDS_V19A),
    }


def validate_perturbation_basis_v19a() -> dict:
    regenerated = np.random.default_rng(
        seed=FRESH_PERTURBATION_BASIS_SEED_V19A
    ).integers(0, 2**30, size=32, dtype=np.int64).tolist()
    basis = perturbation_basis_v19a()
    if (
        regenerated != PERTURBATION_SEEDS_V19A
        or len(set(regenerated)) != 32
        or canonical_sha256(regenerated)
        != PERTURBATION_SEED_LIST_SHA256_V19A
        or canonical_sha256(basis) != PERTURBATION_BASIS_SHA256_V19A
        or PERTURBATION_BASIS_SHA256_V19A
        == prereg_v18a.prereg_v17a.anchor_v13.PERTURBATION_BASIS_SHA256_V13
        or PERTURBATION_SEED_LIST_SHA256_V19A
        == canonical_sha256(
            prereg_v18a.prereg_v17a.anchor_v13.PERTURBATION_SEEDS_V13
        )
    ):
        raise RuntimeError("v19a fresh perturbation basis changed")
    return basis


def build_preregistration_v19a() -> dict:
    flow, evidence, v18a_preregistration, token_audit = load_bound_inputs_v19a()
    comparisons = ARM_ORDER_V19A[1:]
    endpoint_contract = {
        f"{arm}__{endpoint}": {
            "arm": arm,
            "control": "production_only",
            "endpoint": endpoint,
            "noninferiority_margin": 0.0,
        }
        for arm in comparisons
        for endpoint in ENDPOINT_NAMES_V19A
    }
    arms = {}
    for arm in ARM_ORDER_V19A:
        tier = frame_v19a.ARM_ACTIVE_TIER_V19A[arm]
        arms[arm] = {
            "active_patch_tier": tier,
            "active_patch_population_components": 0 if tier is None else 75,
            "active_population_components": (
                frame_v19a.ARM_POPULATIONS_V19A[arm]
            ),
            "requests_per_panel": frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm],
            "paired_tiers_using_candidate_representative": (
                [] if tier is None else [tier]
            ),
            "paired_tiers_using_production_representative": (
                [1, 2, 3]
                if tier is None
                else [other for other in (1, 2, 3) if other != tier]
            ),
            "active_candidate_only_tiers": [] if tier is None else [tier],
            "fallback_always_uses_production_representative": True,
            "same_arm_paired_duplicate_count": 0,
        }
    prior_recipe = v18a_preregistration["frozen_recipe"]
    scoring = _scoring_contract_v19a(v18a_preregistration)
    perturbation_basis = validate_perturbation_basis_v19a()
    value = {
        "schema": "eggroll-es-disjoint-tier-attribution-preregistration-v19a",
        "status": (
            "preregistered_attribution_only_fresh_basis_sealed_runtime_not_authorized"
        ),
        "experiment_name": EXPERIMENT_NAME_V19A,
        "motivation": {
            "bound_v18a_result": evidence["status"],
            "v18a_any_patch_passed": False,
            "v18a_observed_pass_counts": evidence["gate_summary"][
                "observed_pass_counts"
            ],
            "v18a_bootstrap_pass_counts": evidence["gate_summary"][
                "bootstrap_pass_counts"
            ],
            "design_response": (
                "attribute_nonmonotonic_nested_patch_behavior_to_three_"
                "disjoint_tiers_with_more_fixed_train_only_panels"
            ),
            "quality_promotion_or_update_claim": False,
        },
        "immutable_inputs": {
            "candidate_v298": {
                "path": str(frame_v19a.frame_v18a.CANDIDATE_PATH_V18A),
                "rows": frame_v19a.frame_v18a.CANDIDATE_ROWS_V18A,
                "file_sha256": frame_v19a.frame_v18a.CANDIDATE_SHA256_V18A,
                "newer_candidate_discovery_allowed": False,
            },
            "production": {
                "path": str(frame_v19a.frame_v18a.PRODUCTION_PATH_V18A),
                "rows": frame_v19a.frame_v18a.PRODUCTION_ROWS_V18A,
                "file_sha256": frame_v19a.frame_v18a.PRODUCTION_SHA256_V18A,
            },
            "v19a_flow_certificate": {
                "path": str(frame_v19a.OUTPUT_PATH_V19A),
                "file_sha256": FRAME_CERTIFICATE_FILE_SHA256_V19A,
                "content_sha256": FRAME_CERTIFICATE_CONTENT_SHA256_V19A,
                "builder_path": str(Path(frame_v19a.__file__).resolve()),
                "builder_file_sha256": FRAME_BUILDER_FILE_SHA256_V19A,
                "solver_status": flow["constrained_base_flow"]["solver"][
                    "status"
                ],
                "solver_mip_gap": flow["constrained_base_flow"]["solver"][
                    "mip_gap"
                ],
                "quota_relaxation_used": False,
            },
            "v18a_negative_evidence": {
                "commit": V18A_NEGATIVE_EVIDENCE_COMMIT_V19A,
                "path": str(V18A_NEGATIVE_EVIDENCE_PATH_V19A),
                "file_sha256": V18A_NEGATIVE_EVIDENCE_FILE_SHA256_V19A,
                "content_sha256": V18A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V19A,
                "builder_path": str(V18A_NEGATIVE_EVIDENCE_BUILDER_PATH_V19A),
                "builder_file_sha256": (
                    V18A_NEGATIVE_EVIDENCE_BUILDER_SHA256_V19A
                ),
                "source_launch_attempt_file_sha256": evidence[
                    "input_artifacts"
                ]["launch_attempt"]["file_sha256"],
                "source_compact_report_file_sha256": evidence[
                    "input_artifacts"
                ]["compact_report"]["file_sha256"],
                "preregistered_gate_passed": False,
            },
            "v18a_scoring_preregistration": {
                "path": str(V18A_PREREGISTRATION_PATH_V19A),
                "file_sha256": V18A_PREREGISTRATION_FILE_SHA256_V19A,
                "content_sha256": V18A_PREREGISTRATION_CONTENT_SHA256_V19A,
                "builder_file_sha256": (
                    V18A_PREREGISTRATION_BUILDER_SHA256_V19A
                ),
            },
            "token_length_audit": {
                "path": str(prereg_v18a.token_audit_v18a.OUTPUT_PATH_V18A),
                "file_sha256": TOKEN_AUDIT_FILE_SHA256_V19A,
                "content_sha256": TOKEN_AUDIT_CONTENT_SHA256_V19A,
                "builder_file_sha256": TOKEN_AUDIT_BUILDER_FILE_SHA256_V19A,
                "candidate_label": "candidate_v298",
                "tokenizer_boundary_mismatch_count": 0,
                "over_frozen_1024_total_token_cap_count": 0,
                "observed_combined_token_max": token_audit["sources"][
                    "candidate_v298"
                ]["combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max"][-1],
            },
        },
        "patch_semantics": {
            "one_representative_per_joint_component_per_arm": True,
            "each_patch_arm_activates_exactly_one_disjoint_tier": True,
            "active_paired_tier_uses": "candidate_instead_of_production",
            "inactive_paired_tiers_use": "production",
            "active_candidate_only_tier_adds": "one_q1_observation_per_panel",
            "inactive_candidate_only_tiers_add": "nothing",
            "fallback_components_use": "production_every_arm",
            "ambiguous_pair_action": "always_production_fallback",
            "paired_candidate_and_production_both_present_same_arm": False,
            "same_arm_paired_duplicate_count": 0,
        },
        "population": {
            "joint_components": 295,
            "production_base_components": 272,
            "eligible_shared_document_paired_components": 202,
            "candidate_only_components": 23,
            "paired_tier_populations": {"1": 67, "2": 68, "3": 67},
            "candidate_only_tier_populations": {"1": 8, "2": 7, "3": 8},
            "eligible_patch_population_per_tier": {"1": 75, "2": 75, "3": 75},
            "arm_population_denominators": copy.deepcopy(
                frame_v19a.ARM_POPULATIONS_V19A
            ),
        },
        "panels": {
            "names": list(frame_v19a.PANEL_NAMES_V19A),
            "optimization": list(frame_v19a.OPTIMIZATION_PANELS_V19A),
            "train_only_screens": list(frame_v19a.TRAIN_SCREEN_PANELS_V19A),
            "base_category_quota_per_panel": {
                category: frame_v19a.BASE_CATEGORY_QUOTA_V19A
                for category in frame_v19a.BASE_CATEGORIES_V19A
            },
            "production_topic_quota_per_panel": copy.deepcopy(
                frame_v19a.PRODUCTION_TOPIC_QUOTAS_V19A
            ),
            "candidate_only_own_tier_quota_per_patch_arm_panel": 1,
            "arm_requests_per_panel": copy.deepcopy(
                frame_v19a.ARM_REQUESTS_PER_PANEL_V19A
            ),
            "globally_disjoint_base_components": True,
            "candidate_only_every_unique_before_deterministic_reuse": True,
            "candidate_only_reuse_assignments_after_exhaustion": 7,
            "fixed_side_representatives_every_direction_sign_and_arm": True,
            "screens_scored_every_direction_but_excluded_from_optimization_"
            "or_update": True,
            "infeasible_action": (
                "abort_v19a_without_quota_seed_grouping_or_solver_fallback"
            ),
        },
        "arms": arms,
        "estimator": {
            "base_ht_strata": copy.deepcopy(
                flow["estimand"]["base_population_ht_strata"]
            ),
            "candidate_only_ht_strata": copy.deepcopy(
                flow["estimand"]["candidate_only_ht_strata"]
            ),
            "arm_total": flow["estimand"]["arm_total"],
            "arm_mean": flow["estimand"]["arm_mean"],
            "arm_population_denominators": copy.deepcopy(
                frame_v19a.ARM_POPULATIONS_V19A
            ),
            "candidate_ht_targets_only_active_sealed_tier": True,
            "candidate_ht_never_upweights_to_all_226": True,
            "plain_request_mean_used": False,
            "shared_component_counted_once_per_arm": True,
        },
        "frozen_recipe": {
            "model": prior_recipe["model"],
            "layers": [20, 21, 22, 23],
            "layer_plan": copy.deepcopy(prior_recipe["layer_plan"]),
            "sigma": 0.0003,
            "alpha": 0.0,
            "population_size": 32,
            "perturbation_basis": perturbation_basis,
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V19A,
            "perturbation_seed_list_sha256": (
                PERTURBATION_SEED_LIST_SHA256_V19A
            ),
            "prior_v18a_basis_content_sha256": prior_recipe[
                "perturbation_basis_sha256"
            ],
            "prior_v18a_basis_reuse_allowed": False,
            "basis_generator": (
                "numpy.default_rng(basis_seed).integers(0,2**30,size=32,"
                "dtype=int64)"
            ),
            "basis_generation_or_selection_at_launch_allowed": False,
            "moe_backend": copy.deepcopy(prior_recipe["moe_backend"]),
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_surfaces_opened": False,
        },
        "common_random_numbers": {
            "same_resident_perturbation_scores_all_four_arms_before_restore": True,
            "signed_wave_arm_orders": signed_wave_arm_orders_v19a(),
            "each_arm_appears_each_position_twice_per_sign": True,
            "exact_restore_and_verify_once_after_all_four_arms_each_sign": True,
            "pre_post_base_probe_all_four_arms_must_match": True,
            "all_four_tp1_engines_every_signed_wave": True,
            "partial_waves_allowed": False,
        },
        "scoring": scoring,
        "analysis": {
            "metric_families": list(prereg_v18a.prereg_v17a.METRIC_FAMILIES_V17A),
            "endpoint_names": list(ENDPOINT_NAMES_V19A),
            "patch_to_production_comparisons": list(comparisons),
            "endpoint_contract": endpoint_contract,
            "hypothesis_count": HYPOTHESIS_COUNT_V19A,
            "panel_endpoint_geometry": {
                "optimization_pairwise_pairs": 15,
                "aggregate_to_optimization_panels": 6,
                "train_only_screen_panels": 4,
                "median_and_worst_for_each_metric_family": True,
            },
            "bootstrap": {
                "seed": BOOTSTRAP_SEED_V19A,
                "repetitions": BOOTSTRAP_REPETITIONS_V19A,
                "familywise_alpha": FAMILYWISE_ALPHA_V19A,
                "one_sided_quantile": (
                    FAMILYWISE_ALPHA_V19A / HYPOTHESIS_COUNT_V19A
                ),
                "multiplicity": (
                    "Bonferroni_over_three_disjoint_tier_arms_times_twelve_"
                    "stability_endpoints"
                ),
                "fixed_panel_identities_every_replicate": True,
                "within_panel_base_resampling": (
                    "resample_six_components_within_each_of_four_base_HT_"
                    "strata_using_shared_uniforms_all_arms"
                ),
                "candidate_only_resampling": (
                    "for_each_fixed_target_panel_draw_one_same_role_panel_"
                    "position_with_a_uniform_index_shared_across_all_three_"
                    "patch_arms_then_use_each_arms_own_tier_q1_observation"
                ),
                "whole_panel_block_resampling_used": False,
                "recompute_HT_arm_totals_exact_denominator_coefficients_"
                "aggregate_and_all_nonlinear_endpoints_each_replicate": True,
                "persist_per_unit_scores_or_bootstrap_draws": False,
            },
        },
        "attribution_gate": {
            "all_rules_conjunctive_within_each_tier_arm": True,
            "observed_patch_not_below_production_all_twelve": True,
            "every_zero_margin_familywise_lcb_nonnegative_all_twelve": True,
            "all_panel_spreads_nonzero": True,
            "all_runtime_identity_restoration_boundary_and_backend_audits": True,
            "no_passing_tier_decision": "retain_production_dataset_and_v13_recipe",
            "passing_tier_decision": (
                "authorize_only_a_separate_fresh_basis_train_only_"
                "confirmation_preregistration_for_that_tier"
            ),
            "this_result_can_directly_promote_dataset": False,
            "this_result_can_directly_update_model": False,
            "this_result_can_directly_open_evaluation": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        },
        "runtime_request_contract": {
            "requests_per_engine_per_arm_per_signed_wave": {
                "production_only": 240,
                "patch_tier_1_only": 250,
                "patch_tier_2_only": 250,
                "patch_tier_3_only": 250,
            },
            "requests_per_engine_per_signed_wave_all_arms": 990,
            "all_four_gpus_required": True,
        },
        "required_next_artifacts": {
            "separate_committed_runtime_adapter_required": True,
            "runtime_must_bind_exact_preregistered_basis_commitment": True,
            "runtime_must_bind_exact_preregistered_seed_list_commitment": True,
            "basis_generation_or_selection_at_launch_allowed": False,
            "runtime_adapter_must_implement_disjoint_tier_substitution": True,
            "runtime_adapter_must_recompute_tier_specific_HT": True,
            "runtime_launch_authorized_by_this_preregistration": False,
        },
        "firewall": {
            "train_only": True,
            "manual_candidate_seal_exactly_v298": True,
            "newer_candidate_discovery_allowed": False,
            "contains_question_answer_prompt_response_or_row_content": False,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "per_example_content_persisted": False,
            "model_update_checkpoint_or_evaluation_allowed": False,
            "gpu_launch_allowed": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    validate_preregistration_v19a(value)
    return value


def validate_preregistration_v19a(value: dict) -> dict:
    recipe = value.get("frozen_recipe", {})
    analysis = value.get("analysis", {})
    runtime = value.get("runtime_request_contract", {})
    token_audit = value.get("immutable_inputs", {}).get("token_length_audit", {})
    if (
        value.get("schema")
        != "eggroll-es-disjoint-tier-attribution-preregistration-v19a"
        or value.get("status")
        != (
            "preregistered_attribution_only_fresh_basis_sealed_"
            "runtime_not_authorized"
        )
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or set(value.get("arms", {})) != set(ARM_ORDER_V19A)
        or value.get("population", {}).get("arm_population_denominators")
        != frame_v19a.ARM_POPULATIONS_V19A
        or value.get("panels", {}).get("arm_requests_per_panel")
        != frame_v19a.ARM_REQUESTS_PER_PANEL_V19A
        or len(value.get("panels", {}).get("names", [])) != 10
        or len(value.get("panels", {}).get("optimization", [])) != 6
        or len(value.get("panels", {}).get("train_only_screens", [])) != 4
        or analysis.get("hypothesis_count") != 36
        or len(analysis.get("endpoint_contract", {})) != 36
        or analysis.get("bootstrap", {}).get(
            "fixed_panel_identities_every_replicate"
        ) is not True
        or analysis.get("bootstrap", {}).get("whole_panel_block_resampling_used")
        is not False
        or recipe.get("layers") != [20, 21, 22, 23]
        or recipe.get("sigma") != 0.0003
        or recipe.get("alpha") != 0.0
        or recipe.get("population_size") != 32
        or recipe.get("perturbation_basis") != perturbation_basis_v19a()
        or recipe.get("perturbation_basis_sha256")
        != PERTURBATION_BASIS_SHA256_V19A
        or recipe.get("perturbation_seed_list_sha256")
        != PERTURBATION_SEED_LIST_SHA256_V19A
        or recipe.get("prior_v18a_basis_reuse_allowed") is not False
        or recipe.get("basis_generation_or_selection_at_launch_allowed")
        is not False
        or runtime.get("requests_per_engine_per_arm_per_signed_wave")
        != {
            "production_only": 240,
            "patch_tier_1_only": 250,
            "patch_tier_2_only": 250,
            "patch_tier_3_only": 250,
        }
        or runtime.get("requests_per_engine_per_signed_wave_all_arms") != 990
        or token_audit.get("over_frozen_1024_total_token_cap_count") != 0
        or token_audit.get("observed_combined_token_max") != 144
        or value.get("scoring", {}).get("objective_change_allowed_in_v19a")
        is not False
        or value.get("attribution_gate", {}).get(
            "this_result_can_directly_promote_dataset"
        ) is not False
        or value.get("attribution_gate", {}).get(
            "this_result_can_directly_update_model"
        ) is not False
        or value.get("attribution_gate", {}).get(
            "this_result_can_directly_open_evaluation"
        ) is not False
        or value.get("required_next_artifacts", {}).get(
            "runtime_launch_authorized_by_this_preregistration"
        ) is not False
        or value.get("firewall", {}).get("gpu_launch_allowed") is not False
        or value.get("firewall", {}).get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
        or value.get("firewall", {}).get(
            "contains_question_answer_prompt_response_or_row_content"
        ) is not False
    ):
        raise RuntimeError("v19a disjoint-tier attribution preregistration changed")
    prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(value)
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    if path.resolve() != OUTPUT_PATH_V19A:
        raise ValueError("v19a preregistration output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v19a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V19A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V19A:
        raise ValueError("v19a preregistration output path changed")
    value = build_preregistration_v19a()
    _exclusive_write(OUTPUT_PATH_V19A, value)
    result = {
        "schema": "eggroll-es-disjoint-tier-preregistration-write-v19a",
        "path": str(OUTPUT_PATH_V19A),
        "file_sha256": file_sha256(OUTPUT_PATH_V19A),
        "content_sha256": value["content_sha256_before_self_field"],
        "runtime_launch_authorized": False,
        "model_update_authorized": False,
        "evaluation_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
