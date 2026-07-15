#!/usr/bin/env python3
"""Preregister the offline V20A nested tier-interaction train experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

import build_eggroll_es_nested_tier_interaction_frame_v20a as frame_v20a
import eggroll_es_disjoint_tier_attribution_preregistration_v19a as prereg_v19a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V20A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_PRODUCTION_PATCH_NESTED_TIER_INTERACTION_V20A_PREREGISTRATION.json"
).resolve()
V19A_NEGATIVE_EVIDENCE_PATH_V20A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V19A_DISJOINT_TIER_ATTRIBUTION_NEGATIVE_EVIDENCE.json"
).resolve()
V19A_NEGATIVE_EVIDENCE_BUILDER_PATH_V20A = (
    ROOT / "build_eggroll_es_v19a_compact_evidence.py"
).resolve()
UNION_PLANNER_PATH_V20A = (
    ROOT / "eggroll_es_union_request_plan_v20a.py"
).resolve()
UNION_PLANNER_TEST_PATH_V20A = (
    ROOT / "test_eggroll_es_union_request_plan_v20a.py"
).resolve()

EXPERIMENT_NAME_V20A = (
    "snapshot_v298_production_patch_v20a_nested_tier_interactions_"
    "10x24_plus_nested_q1_alpha_zero_middle_late_fresh_basis"
)
FRAME_BUILDER_FILE_SHA256_V20A = (
    "f1fc7dcea9780c27bd490caab4821f0ec8339ad748bdedb3cb2609e404743a76"
)
FRAME_CERTIFICATE_FILE_SHA256_V20A = (
    "cd2cb7134716602f1d51fe4049c95c49c33ddee429dcc6e752e2cb0b4196f444"
)
FRAME_CERTIFICATE_CONTENT_SHA256_V20A = (
    "1199f2bcb4cf3c6f394c2faf2edf14cb9ab74323c47614118511d768197c8078"
)
V19A_PREREGISTRATION_FILE_SHA256_V20A = (
    "cd65f6127ce8d4b401d837bb23c424f0860259ab6920e25a68a4098f1798a987"
)
V19A_PREREGISTRATION_CONTENT_SHA256_V20A = (
    "db2036da7c9b382a542d7c002f5af8a76bbe1101d342fad8bfdc8e3a67b6c997"
)
V19A_PREREGISTRATION_BUILDER_SHA256_V20A = (
    "e684da1e53a27c535eb441d0ce8109b8aadff9bad2f49117ea27b97c24d41365"
)
V19A_NEGATIVE_EVIDENCE_COMMIT_V20A = (
    "f3a09393b6d13754310eeb2ffc131791c557fde3"
)
V19A_NEGATIVE_EVIDENCE_FILE_SHA256_V20A = (
    "e91ae9f14d87ee97ce34fd7e5dcfaa27a0f1ef65abeba0582e0bf0cb2c8f5267"
)
V19A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V20A = (
    "84895cba0c6b79adc1b6ae59680632694bc1e6cb29bf90211cdf8c7f3665040f"
)
V19A_NEGATIVE_EVIDENCE_BUILDER_SHA256_V20A = (
    "a2753d87b6fabcb57f372668770187d8a44bb9bcc7e2177d32347d2452cf7516"
)
UNION_PLANNER_COMMIT_V20A = (
    "cc342be6374b1cb57479119c0789ad67f19467e0"
)
UNION_PLANNER_FILE_SHA256_V20A = (
    "b0a4292641929f551ebbe0aa3756df5b90240e622e30352ef52198d9fc90ef2f"
)
UNION_PLANNER_TEST_FILE_SHA256_V20A = (
    "b4bc22202ddc0a7892f2fcd59de7a3e79e0b3021cc4a138c2c400fc89b21bc2e"
)

ARM_ORDER_V20A = frame_v20a.ARM_ORDER_V20A
ENDPOINT_NAMES_V20A = tuple(prereg_v19a.ENDPOINT_NAMES_V19A)
CONTRASTS_V20A = {
    "tier2_vs_production": {
        "treatment": "patch_tier_2_only",
        "control": "production_only",
        "purpose": "absolute_tier2_effect",
    },
    "tiers2_3_vs_production": {
        "treatment": "patch_tiers_2_3",
        "control": "production_only",
        "purpose": "absolute_tiers2_3_effect",
    },
    "all_tiers_vs_production": {
        "treatment": "patch_all_tiers",
        "control": "production_only",
        "purpose": "same_frame_full_patch_recovery",
    },
    "conditional_tier3_after_tier2": {
        "treatment": "patch_tiers_2_3",
        "control": "patch_tier_2_only",
        "purpose": "conditional_tier3_effect_around_strongest_tier2",
    },
    "conditional_tier1_after_tiers2_3": {
        "treatment": "patch_all_tiers",
        "control": "patch_tiers_2_3",
        "purpose": "conditional_tier1_effect_after_tiers2_3",
    },
}
HYPOTHESIS_COUNT_V20A = len(CONTRASTS_V20A) * len(ENDPOINT_NAMES_V20A)
FRESH_PERTURBATION_BASIS_SEED_V20A = 20260731
PERTURBATION_SEEDS_V20A = [
    29271574, 953929159, 690861080, 85830537,
    25073372, 319470708, 1026263659, 342197043,
    886784214, 1071391876, 204609048, 559044873,
    660097328, 217474847, 125965093, 1016225387,
    983881699, 778438174, 866862373, 543868786,
    874123924, 390964630, 1072760879, 200493881,
    349819859, 1055438193, 452508913, 126412412,
    505616267, 110496880, 42318870, 567755038,
]
PERTURBATION_SEED_LIST_SHA256_V20A = (
    "9dde42844ecbfb038ff25ad576a6f8c691e3a0473a6d24bc8c63bdefd600ab94"
)
PERTURBATION_BASIS_SHA256_V20A = (
    "b6d667c2f125f9d0be4d74ef536af03546fecb6c03f2838679f5a315a1ec9852"
)
BOOTSTRAP_SEED_V20A = 20260801
BOOTSTRAP_REPETITIONS_V20A = 50_000
FAMILYWISE_ALPHA_V20A = 0.05
V18A_PERTURBATION_BASIS_SHA256_V20A = (
    "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11"
)


canonical_sha256 = prereg_v19a.canonical_sha256
file_sha256 = prereg_v19a.file_sha256


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
        raise RuntimeError(f"v20a committed evidence binding changed: {relative}")


def _validate_v19a_negative_evidence(value: dict) -> dict:
    if (
        value.get("schema")
        != "eggroll-es-disjoint-tier-attribution-negative-evidence-v19a"
        or value.get("status")
        != "valid_completed_run_preregistered_gate_failed"
        or value.get("content_sha256_before_self_field")
        != V19A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V20A
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("gate_summary", {}).get("any_tier_passed") is not False
        or value.get("gate_summary", {}).get("observed_pass_counts")
        != {
            "patch_tier_1_only": 2,
            "patch_tier_2_only": 5,
            "patch_tier_3_only": 4,
        }
        or value.get("gate_summary", {}).get("bootstrap_pass_counts")
        != {
            "patch_tier_1_only": 0,
            "patch_tier_2_only": 0,
            "patch_tier_3_only": 0,
        }
        or value.get("decision", {}).get("retain_dataset") != "production"
        or value.get("decision", {}).get("retain_recipe")
        != "v13_middle_late_layers_20_23"
        or value.get("decision", {}).get("dataset_promotion_authorized") is not False
        or value.get("decision", {}).get("model_update_authorized") is not False
        or value.get("decision", {}).get("evaluation_authorized") is not False
        or value.get("contains_response_vectors_or_row_content") is not False
        or value.get("contains_validation_ood_heldout_or_benchmark_content")
        is not False
    ):
        raise RuntimeError("v20a V19A negative evidence binding changed")
    return value


def load_bound_inputs_v20a() -> tuple[dict, dict, dict, dict, dict]:
    expected = {
        Path(frame_v20a.__file__).resolve(): FRAME_BUILDER_FILE_SHA256_V20A,
        frame_v20a.OUTPUT_PATH_V20A: FRAME_CERTIFICATE_FILE_SHA256_V20A,
        Path(prereg_v19a.__file__).resolve(): (
            V19A_PREREGISTRATION_BUILDER_SHA256_V20A
        ),
        prereg_v19a.OUTPUT_PATH_V19A: V19A_PREREGISTRATION_FILE_SHA256_V20A,
        V19A_NEGATIVE_EVIDENCE_BUILDER_PATH_V20A: (
            V19A_NEGATIVE_EVIDENCE_BUILDER_SHA256_V20A
        ),
        V19A_NEGATIVE_EVIDENCE_PATH_V20A: V19A_NEGATIVE_EVIDENCE_FILE_SHA256_V20A,
        prereg_v19a.V18A_NEGATIVE_EVIDENCE_BUILDER_PATH_V19A: (
            prereg_v19a.V18A_NEGATIVE_EVIDENCE_BUILDER_SHA256_V19A
        ),
        prereg_v19a.V18A_NEGATIVE_EVIDENCE_PATH_V19A: (
            prereg_v19a.V18A_NEGATIVE_EVIDENCE_FILE_SHA256_V19A
        ),
        prereg_v19a.prereg_v18a.token_audit_v18a.OUTPUT_PATH_V18A: (
            prereg_v19a.TOKEN_AUDIT_FILE_SHA256_V19A
        ),
        UNION_PLANNER_PATH_V20A: UNION_PLANNER_FILE_SHA256_V20A,
        UNION_PLANNER_TEST_PATH_V20A: UNION_PLANNER_TEST_FILE_SHA256_V20A,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v20a immutable train-only artifact binding changed")
    _verify_commit_file(
        V19A_NEGATIVE_EVIDENCE_PATH_V20A,
        V19A_NEGATIVE_EVIDENCE_COMMIT_V20A,
        V19A_NEGATIVE_EVIDENCE_FILE_SHA256_V20A,
    )
    _verify_commit_file(
        V19A_NEGATIVE_EVIDENCE_BUILDER_PATH_V20A,
        V19A_NEGATIVE_EVIDENCE_COMMIT_V20A,
        V19A_NEGATIVE_EVIDENCE_BUILDER_SHA256_V20A,
    )
    _verify_commit_file(
        prereg_v19a.V18A_NEGATIVE_EVIDENCE_PATH_V19A,
        prereg_v19a.V18A_NEGATIVE_EVIDENCE_COMMIT_V19A,
        prereg_v19a.V18A_NEGATIVE_EVIDENCE_FILE_SHA256_V19A,
    )
    _verify_commit_file(
        UNION_PLANNER_PATH_V20A,
        UNION_PLANNER_COMMIT_V20A,
        UNION_PLANNER_FILE_SHA256_V20A,
    )
    _verify_commit_file(
        UNION_PLANNER_TEST_PATH_V20A,
        UNION_PLANNER_COMMIT_V20A,
        UNION_PLANNER_TEST_FILE_SHA256_V20A,
    )
    flow = json.loads(frame_v20a.OUTPUT_PATH_V20A.read_text(encoding="utf-8"))
    frame_v20a.validate_certificate_v20a(flow)
    evidence_v19a = _validate_v19a_negative_evidence(json.loads(
        V19A_NEGATIVE_EVIDENCE_PATH_V20A.read_text(encoding="utf-8")
    ))
    evidence_v18a = prereg_v19a._validate_v18a_negative_evidence(json.loads(
        prereg_v19a.V18A_NEGATIVE_EVIDENCE_PATH_V19A.read_text(encoding="utf-8")
    ))
    prior = json.loads(prereg_v19a.OUTPUT_PATH_V19A.read_text(encoding="utf-8"))
    prereg_v19a.validate_preregistration_v19a(prior)
    token_audit = json.loads(
        prereg_v19a.prereg_v18a.token_audit_v18a.OUTPUT_PATH_V18A.read_text(
            encoding="utf-8"
        )
    )
    prereg_v19a.prereg_v18a.token_audit_v18a.validate_audit_v18a(token_audit)
    if (
        flow.get("content_sha256_before_self_field")
        != FRAME_CERTIFICATE_CONTENT_SHA256_V20A
        or prior.get("content_sha256_before_self_field")
        != V19A_PREREGISTRATION_CONTENT_SHA256_V20A
        or token_audit.get("content_sha256_before_self_field")
        != prereg_v19a.TOKEN_AUDIT_CONTENT_SHA256_V19A
    ):
        raise RuntimeError("v20a bound artifact content changed")
    return flow, evidence_v18a, evidence_v19a, prior, token_audit


def signed_wave_arm_orders_v20a() -> list[dict]:
    arms = list(ARM_ORDER_V20A)
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
                raise RuntimeError("v20a signed arm-position balance changed")
    return orders


def _scoring_contract_v20a(prior: dict) -> dict:
    scoring = copy.deepcopy(prior["scoring"])
    if scoring.pop("objective_change_allowed_in_v19a", None) is not False:
        raise RuntimeError("v20a inherited scoring contract changed")
    scoring["objective_change_allowed_in_v20a"] = False
    scoring["frozen_max_total_tokens"] = 1024
    return scoring


def perturbation_basis_v20a() -> dict:
    return {
        "schema": "eggroll-es-nested-tier-interaction-perturbation-basis-v20a",
        "basis_seed": FRESH_PERTURBATION_BASIS_SEED_V20A,
        "population_size": 32,
        "seeds": list(PERTURBATION_SEEDS_V20A),
    }


def validate_perturbation_basis_v20a() -> dict:
    regenerated = np.random.default_rng(
        seed=FRESH_PERTURBATION_BASIS_SEED_V20A
    ).integers(0, 2**30, size=32, dtype=np.int64).tolist()
    basis = perturbation_basis_v20a()
    v18a_basis = V18A_PERTURBATION_BASIS_SHA256_V20A
    v19a_basis = prereg_v19a.PERTURBATION_BASIS_SHA256_V19A
    if (
        regenerated != PERTURBATION_SEEDS_V20A
        or len(set(regenerated)) != 32
        or canonical_sha256(regenerated) != PERTURBATION_SEED_LIST_SHA256_V20A
        or canonical_sha256(basis) != PERTURBATION_BASIS_SHA256_V20A
        or PERTURBATION_BASIS_SHA256_V20A in {v18a_basis, v19a_basis}
        or set(regenerated) == set(prereg_v19a.PERTURBATION_SEEDS_V19A)
    ):
        raise RuntimeError("v20a fresh perturbation basis changed")
    return basis


def build_preregistration_v20a() -> dict:
    flow, evidence_v18a, evidence_v19a, prior, token_audit = (
        load_bound_inputs_v20a()
    )
    endpoint_contract = {
        f"{contrast}__{endpoint}": {
            "contrast": contrast,
            "treatment": contract["treatment"],
            "control": contract["control"],
            "endpoint": endpoint,
            "noninferiority_margin": 0.0,
        }
        for contrast, contract in CONTRASTS_V20A.items()
        for endpoint in ENDPOINT_NAMES_V20A
    }
    arms = {}
    for arm in ARM_ORDER_V20A:
        active_tiers = frame_v20a.ARM_ACTIVE_TIERS_V20A[arm]
        arms[arm] = {
            "active_patch_tiers": list(active_tiers),
            "active_patch_population_components": sum(
                frame_v20a.PATCH_TIER_POPULATIONS_V20A[tier]
                for tier in active_tiers
            ),
            "active_population_components": frame_v20a.ARM_POPULATIONS_V20A[arm],
            "requests_per_panel": frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm],
            "paired_tiers_using_candidate_representative": list(active_tiers),
            "paired_tiers_using_production_representative": [
                tier for tier in (1, 2, 3) if tier not in active_tiers
            ],
            "active_candidate_only_tiers": list(active_tiers),
            "fallback_always_uses_production_representative": True,
            "same_arm_paired_duplicate_count": 0,
        }
    prior_recipe = prior["frozen_recipe"]
    scoring = _scoring_contract_v20a(prior)
    perturbation_basis = validate_perturbation_basis_v20a()
    value = {
        "schema": "eggroll-es-nested-tier-interaction-preregistration-v20a",
        "status": (
            "preregistered_attribution_only_fresh_frame_fresh_basis_"
            "runtime_not_authorized"
        ),
        "experiment_name": EXPERIMENT_NAME_V20A,
        "motivation": {
            "bound_v18a_result": evidence_v18a["status"],
            "v18a_observed_pass_counts": evidence_v18a["gate_summary"][
                "observed_pass_counts"
            ],
            "v18a_bootstrap_pass_counts": evidence_v18a["gate_summary"][
                "bootstrap_pass_counts"
            ],
            "bound_v19a_result": evidence_v19a["status"],
            "v19a_observed_pass_counts": evidence_v19a["gate_summary"][
                "observed_pass_counts"
            ],
            "v19a_bootstrap_pass_counts": evidence_v19a["gate_summary"][
                "bootstrap_pass_counts"
            ],
            "v19a_strongest_nominal_tier": "patch_tier_2_only",
            "design_response": (
                "test_nested_tier_interactions_to_explain_v18_full_patch_"
                "recovery_and_isolate_tier3_after_tier2_then_tier1_after_"
                "tiers2_3_on_one_maximally_fresh_train_only_frame"
            ),
            "quality_promotion_or_update_claim": False,
        },
        "immutable_inputs": {
            "candidate_v298": {
                "path": str(frame_v20a.frame_v19a.frame_v18a.CANDIDATE_PATH_V18A),
                "rows": frame_v20a.frame_v19a.frame_v18a.CANDIDATE_ROWS_V18A,
                "file_sha256": frame_v20a.frame_v19a.frame_v18a.CANDIDATE_SHA256_V18A,
                "newer_candidate_discovery_allowed": False,
            },
            "production": {
                "path": str(frame_v20a.frame_v19a.frame_v18a.PRODUCTION_PATH_V18A),
                "rows": frame_v20a.frame_v19a.frame_v18a.PRODUCTION_ROWS_V18A,
                "file_sha256": frame_v20a.frame_v19a.frame_v18a.PRODUCTION_SHA256_V18A,
            },
            "v20a_flow_certificate": {
                "path": str(frame_v20a.OUTPUT_PATH_V20A),
                "file_sha256": FRAME_CERTIFICATE_FILE_SHA256_V20A,
                "content_sha256": FRAME_CERTIFICATE_CONTENT_SHA256_V20A,
                "builder_path": str(Path(frame_v20a.__file__).resolve()),
                "builder_file_sha256": FRAME_BUILDER_FILE_SHA256_V20A,
                "solver_status": flow["constrained_base_flow"]["solver"]["status"],
                "solver_mip_gap": flow["constrained_base_flow"]["solver"]["mip_gap"],
                "quota_relaxation_used": False,
                "v19_overlap_cardinality_lower_bound": 208,
                "v19_overlap_achieved": 208,
                "fresh_base_components_achieved": 32,
            },
            "v18a_negative_evidence": {
                "commit": prereg_v19a.V18A_NEGATIVE_EVIDENCE_COMMIT_V19A,
                "path": str(prereg_v19a.V18A_NEGATIVE_EVIDENCE_PATH_V19A),
                "file_sha256": prereg_v19a.V18A_NEGATIVE_EVIDENCE_FILE_SHA256_V19A,
                "content_sha256": prereg_v19a.V18A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V19A,
                "preregistered_gate_passed": False,
            },
            "v19a_negative_evidence": {
                "commit": V19A_NEGATIVE_EVIDENCE_COMMIT_V20A,
                "path": str(V19A_NEGATIVE_EVIDENCE_PATH_V20A),
                "file_sha256": V19A_NEGATIVE_EVIDENCE_FILE_SHA256_V20A,
                "content_sha256": V19A_NEGATIVE_EVIDENCE_CONTENT_SHA256_V20A,
                "builder_path": str(V19A_NEGATIVE_EVIDENCE_BUILDER_PATH_V20A),
                "builder_file_sha256": V19A_NEGATIVE_EVIDENCE_BUILDER_SHA256_V20A,
                "source_launch_attempt_file_sha256": evidence_v19a[
                    "input_artifacts"
                ]["launch_attempt"]["file_sha256"],
                "source_compact_report_file_sha256": evidence_v19a[
                    "input_artifacts"
                ]["compact_report"]["file_sha256"],
                "preregistered_gate_passed": False,
            },
            "v19a_preregistration": {
                "path": str(prereg_v19a.OUTPUT_PATH_V19A),
                "file_sha256": V19A_PREREGISTRATION_FILE_SHA256_V20A,
                "content_sha256": V19A_PREREGISTRATION_CONTENT_SHA256_V20A,
                "builder_file_sha256": V19A_PREREGISTRATION_BUILDER_SHA256_V20A,
            },
            "token_length_audit": {
                "path": str(
                    prereg_v19a.prereg_v18a.token_audit_v18a.OUTPUT_PATH_V18A
                ),
                "file_sha256": prereg_v19a.TOKEN_AUDIT_FILE_SHA256_V19A,
                "content_sha256": prereg_v19a.TOKEN_AUDIT_CONTENT_SHA256_V19A,
                "candidate_label": "candidate_v298",
                "tokenizer_boundary_mismatch_count": 0,
                "over_frozen_1024_total_token_cap_count": 0,
                "observed_combined_token_max": token_audit["sources"][
                    "candidate_v298"
                ]["combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max"][-1],
            },
            "proposed_union_request_planner": {
                "commit": UNION_PLANNER_COMMIT_V20A,
                "path": str(UNION_PLANNER_PATH_V20A),
                "file_sha256": UNION_PLANNER_FILE_SHA256_V20A,
                "test_path": str(UNION_PLANNER_TEST_PATH_V20A),
                "test_file_sha256": UNION_PLANNER_TEST_FILE_SHA256_V20A,
                "v19_raw_request_proof_count": 990,
                "v19_global_unique_request_proof_count": 440,
                "v19_raw_to_unique_ratio": 2.25,
                "v20_exact_unique_request_count_frozen_here": False,
                "changes_statistical_estimand": False,
                "execution_authorized": False,
            },
        },
        "patch_semantics": {
            "one_representative_per_joint_component_per_arm": True,
            "arms_are_strictly_nested_by_active_tier_set": True,
            "nested_active_tier_sets": {
                arm: list(tiers) for arm, tiers in frame_v20a.ARM_ACTIVE_TIERS_V20A.items()
            },
            "active_paired_tiers_use": "candidate_instead_of_production",
            "inactive_paired_tiers_use": "production",
            "each_active_candidate_only_tier_adds": "one_q1_observation_per_panel",
            "fallback_components_use": "production_every_arm",
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
                frame_v20a.ARM_POPULATIONS_V20A
            ),
        },
        "panels": {
            "names": list(frame_v20a.PANEL_NAMES_V20A),
            "optimization": list(frame_v20a.OPTIMIZATION_PANELS_V20A),
            "train_only_screens": list(frame_v20a.TRAIN_SCREEN_PANELS_V20A),
            "base_category_quota_per_panel": {
                category: frame_v20a.BASE_CATEGORY_QUOTA_V20A
                for category in frame_v20a.BASE_CATEGORIES_V20A
            },
            "production_topic_quota_per_panel": copy.deepcopy(
                frame_v20a.PRODUCTION_TOPIC_QUOTAS_V20A
            ),
            "arm_requests_per_panel": copy.deepcopy(
                frame_v20a.ARM_REQUESTS_PER_PANEL_V20A
            ),
            "globally_disjoint_base_components_within_v20a": True,
            "v19_overlap_theoretical_minimum_attained": True,
            "all_32_v19_unused_base_components_selected": True,
            "candidate_only_every_unique_before_deterministic_reuse": True,
            "fixed_side_representatives_every_direction_sign_and_arm": True,
            "screens_scored_every_direction_but_excluded_from_optimization_or_update": True,
            "infeasible_action": "abort_v20a_without_relaxation_or_fallback",
        },
        "arms": arms,
        "estimator": {
            "base_ht_strata": copy.deepcopy(flow["estimand"]["base_population_ht_strata"]),
            "candidate_only_ht_strata": copy.deepcopy(
                flow["estimand"]["candidate_only_ht_strata"]
            ),
            "arm_total": flow["estimand"]["arm_total"],
            "arm_mean": flow["estimand"]["arm_mean"],
            "arm_population_denominators": copy.deepcopy(
                frame_v20a.ARM_POPULATIONS_V20A
            ),
            "candidate_ht_targets_only_active_sealed_tiers": True,
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
            "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V20A,
            "perturbation_seed_list_sha256": PERTURBATION_SEED_LIST_SHA256_V20A,
            "prior_v18a_basis_content_sha256": (
                V18A_PERTURBATION_BASIS_SHA256_V20A
            ),
            "prior_v19a_basis_content_sha256": prereg_v19a.PERTURBATION_BASIS_SHA256_V19A,
            "prior_basis_reuse_allowed": False,
            "basis_generator": (
                "numpy.default_rng(basis_seed).integers(0,2**30,size=32,dtype=int64)"
            ),
            "basis_generation_or_selection_at_launch_allowed": False,
            "moe_backend": copy.deepcopy(prior_recipe["moe_backend"]),
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_surfaces_opened": False,
            "dataset_promotion_allowed": False,
        },
        "common_random_numbers": {
            "same_resident_perturbation_scores_all_four_arms_before_restore": True,
            "signed_wave_arm_orders": signed_wave_arm_orders_v20a(),
            "each_arm_appears_each_position_twice_per_sign": True,
            "exact_restore_and_verify_once_after_all_four_arms_each_sign": True,
            "pre_post_base_probe_all_four_arms_must_match": True,
            "all_four_tp1_engines_every_signed_wave": True,
            "gpu_ids": [0, 1, 2, 3],
            "tp_per_engine": 1,
            "partial_waves_allowed": False,
        },
        "scoring": scoring,
        "analysis": {
            "metric_families": list(
                prereg_v19a.prereg_v18a.prereg_v17a.METRIC_FAMILIES_V17A
            ),
            "endpoint_names": list(ENDPOINT_NAMES_V20A),
            "contrasts": copy.deepcopy(CONTRASTS_V20A),
            "endpoint_contract": endpoint_contract,
            "hypothesis_count": HYPOTHESIS_COUNT_V20A,
            "panel_endpoint_geometry": {
                "optimization_pairwise_pairs": 15,
                "aggregate_to_optimization_panels": 6,
                "train_only_screen_panels": 4,
                "median_and_worst_for_each_metric_family": True,
            },
            "bootstrap": {
                "seed": BOOTSTRAP_SEED_V20A,
                "repetitions": BOOTSTRAP_REPETITIONS_V20A,
                "familywise_alpha": FAMILYWISE_ALPHA_V20A,
                "one_sided_quantile": FAMILYWISE_ALPHA_V20A / HYPOTHESIS_COUNT_V20A,
                "multiplicity": (
                    "Bonferroni_over_five_preregistered_nested_contrasts_"
                    "times_twelve_stability_endpoints"
                ),
                "fixed_panel_identities_every_replicate": True,
                "within_panel_base_resampling": (
                    "resample_six_components_within_each_of_four_base_HT_"
                    "strata_using_shared_uniforms_all_arms"
                ),
                "candidate_only_resampling": (
                    "for_each_fixed_target_panel_and_tier_draw_one_same_role_"
                    "panel_position_with_uniforms_shared_by_every_arm_that_"
                    "contains_that_nested_tier"
                ),
                "whole_panel_block_resampling_used": False,
                "recompute_HT_arm_totals_exact_denominator_coefficients_"
                "aggregate_and_all_nonlinear_endpoints_each_replicate": True,
                "persist_per_unit_scores_or_bootstrap_draws": False,
            },
        },
        "attribution_gate": {
            "all_rules_conjunctive_within_each_contrast": True,
            "observed_treatment_not_below_control_all_twelve": True,
            "every_zero_margin_familywise_lcb_nonnegative_all_twelve": True,
            "all_panel_spreads_nonzero": True,
            "all_runtime_identity_restoration_boundary_and_backend_audits": True,
            "no_passing_design_decision": "retain_production_dataset_and_v13_recipe",
            "passing_design_decision": (
                "authorize_only_a_separate_fresh_basis_train_only_confirmation_"
                "preregistration_never_direct_update_eval_or_promotion"
            ),
            "this_result_can_directly_promote_dataset": False,
            "this_result_can_directly_update_model": False,
            "this_result_can_directly_open_evaluation": False,
            "dataset_promotion_authorized": False,
            "model_update_authorized": False,
            "evaluation_authorized": False,
        },
        "runtime_request_contract": {
            "raw_arm_request_coverage_is_authoritative": True,
            "requests_per_engine_per_arm_per_signed_wave": {
                "production_only": 240,
                "patch_tier_2_only": 250,
                "patch_tiers_2_3": 260,
                "patch_all_tiers": 270,
            },
            "requests_per_engine_per_signed_wave_all_arms": 1020,
            "all_four_gpus_required": True,
            "engine_count": 4,
            "tp_per_engine": 1,
            "proposed_token_hash_union_scorer": {
                "status": "not_authorized_requires_separate_equivalence_gate",
                "deduplicate_only_identical_tokenized_requests_across_arms": True,
                "planner_commit": UNION_PLANNER_COMMIT_V20A,
                "planner_file_sha256": UNION_PLANNER_FILE_SHA256_V20A,
                "exact_unique_request_count_frozen_here": False,
                "runtime_adapter_must_recompute_and_commit_exact_union": True,
                "reference_wave_bit_exact_old_vs_union_required": True,
                "perturbed_signed_wave_bit_exact_old_vs_union_required": True,
                "all_per_arm_scores_and_dense_commitments_must_match": True,
                "failure_action": "retain_raw_arm_request_execution",
                "execution_authorized_by_this_preregistration": False,
            },
        },
        "required_next_artifacts": {
            "separate_committed_runtime_adapter_required": True,
            "runtime_must_bind_exact_preregistered_basis_commitment": True,
            "runtime_must_bind_exact_preregistered_seed_list_commitment": True,
            "basis_generation_or_selection_at_launch_allowed": False,
            "runtime_adapter_must_implement_nested_tier_substitution": True,
            "runtime_adapter_must_recompute_nested_arm_specific_HT": True,
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
    validate_preregistration_v20a(value)
    return value


def validate_preregistration_v20a(value: dict) -> dict:
    recipe = value.get("frozen_recipe", {})
    analysis = value.get("analysis", {})
    runtime = value.get("runtime_request_contract", {})
    token = value.get("immutable_inputs", {}).get("token_length_audit", {})
    if (
        value.get("schema")
        != "eggroll-es-nested-tier-interaction-preregistration-v20a"
        or value.get("status")
        != (
            "preregistered_attribution_only_fresh_frame_fresh_basis_"
            "runtime_not_authorized"
        )
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or set(value.get("arms", {})) != set(ARM_ORDER_V20A)
        or value.get("population", {}).get("arm_population_denominators")
        != frame_v20a.ARM_POPULATIONS_V20A
        or value.get("panels", {}).get("arm_requests_per_panel")
        != frame_v20a.ARM_REQUESTS_PER_PANEL_V20A
        or value.get("panels", {}).get(
            "v19_overlap_theoretical_minimum_attained"
        ) is not True
        or len(value.get("panels", {}).get("names", [])) != 10
        or len(value.get("panels", {}).get("optimization", [])) != 6
        or len(value.get("panels", {}).get("train_only_screens", [])) != 4
        or set(analysis.get("contrasts", {})) != set(CONTRASTS_V20A)
        or analysis.get("hypothesis_count") != 60
        or len(analysis.get("endpoint_contract", {})) != 60
        or analysis.get("bootstrap", {}).get("fixed_panel_identities_every_replicate")
        is not True
        or analysis.get("bootstrap", {}).get("whole_panel_block_resampling_used")
        is not False
        or recipe.get("layers") != [20, 21, 22, 23]
        or recipe.get("sigma") != 0.0003
        or recipe.get("alpha") != 0.0
        or recipe.get("population_size") != 32
        or recipe.get("perturbation_basis") != perturbation_basis_v20a()
        or recipe.get("perturbation_basis_sha256") != PERTURBATION_BASIS_SHA256_V20A
        or recipe.get("perturbation_seed_list_sha256")
        != PERTURBATION_SEED_LIST_SHA256_V20A
        or recipe.get("prior_basis_reuse_allowed") is not False
        or recipe.get("basis_generation_or_selection_at_launch_allowed") is not False
        or recipe.get("model_update_allowed") is not False
        or recipe.get("checkpoint_write_allowed") is not False
        or recipe.get("evaluation_surfaces_opened") is not False
        or recipe.get("dataset_promotion_allowed") is not False
        or runtime.get("requests_per_engine_per_arm_per_signed_wave")
        != {
            "production_only": 240,
            "patch_tier_2_only": 250,
            "patch_tiers_2_3": 260,
            "patch_all_tiers": 270,
        }
        or runtime.get("requests_per_engine_per_signed_wave_all_arms") != 1020
        or runtime.get("engine_count") != 4
        or runtime.get("tp_per_engine") != 1
        or runtime.get("raw_arm_request_coverage_is_authoritative") is not True
        or runtime.get("proposed_token_hash_union_scorer", {}).get(
            "execution_authorized_by_this_preregistration"
        ) is not False
        or runtime.get("proposed_token_hash_union_scorer", {}).get(
            "exact_unique_request_count_frozen_here"
        ) is not False
        or token.get("over_frozen_1024_total_token_cap_count") != 0
        or token.get("observed_combined_token_max") != 144
        or value.get("scoring", {}).get("objective_change_allowed_in_v20a")
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
        raise RuntimeError("v20a nested tier interaction preregistration changed")
    prereg_v19a.prereg_v18a.prereg_v17a.validate_persisted_sha256_fields_v17a(
        value
    )
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    if path.resolve() != OUTPUT_PATH_V20A:
        raise ValueError("v20a preregistration output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v20a preregistration already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V20A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V20A:
        raise ValueError("v20a preregistration output path changed")
    value = build_preregistration_v20a()
    _exclusive_write(OUTPUT_PATH_V20A, value)
    result = {
        "schema": "eggroll-es-nested-tier-interaction-preregistration-write-v20a",
        "path": str(OUTPUT_PATH_V20A),
        "file_sha256": file_sha256(OUTPUT_PATH_V20A),
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
