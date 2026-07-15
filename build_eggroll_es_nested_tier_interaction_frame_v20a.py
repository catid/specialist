#!/usr/bin/env python3
"""Build the aggregate-only V20A nested tier-interaction train frame."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import build_eggroll_es_disjoint_tier_frame_v19a as frame_v19a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V20A = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v20a/"
    "production_patch_nested_interaction_panels_v20a.json"
).resolve()

FRAME_V19A_FILE_SHA256_V20A = (
    "4de43c263f580143d1b1f09a67240b8a05df7d45b419b19a211b549bf85b2052"
)
FRAME_CERTIFICATE_V19A_FILE_SHA256_V20A = (
    "50820c67844a7b11c92bc0bbaa9c594c683440a65cb882de244216a77dca5fed"
)
FRAME_CERTIFICATE_V19A_CONTENT_SHA256_V20A = (
    "7ad195a55b1e51268dfba1cddb43f869b014bdb1ef329f5d73c8246ac6cbff58"
)

PANEL_NAMES_V20A = frame_v19a.PANEL_NAMES_V19A
OPTIMIZATION_PANELS_V20A = frame_v19a.OPTIMIZATION_PANELS_V19A
TRAIN_SCREEN_PANELS_V20A = frame_v19a.TRAIN_SCREEN_PANELS_V19A
ARM_ORDER_V20A = (
    "production_only",
    "patch_tier_2_only",
    "patch_tiers_2_3",
    "patch_all_tiers",
)
ARM_ACTIVE_TIERS_V20A = {
    "production_only": (),
    "patch_tier_2_only": (2,),
    "patch_tiers_2_3": (2, 3),
    "patch_all_tiers": (1, 2, 3),
}
BASE_CATEGORIES_V20A = frame_v19a.BASE_CATEGORIES_V19A
BASE_CATEGORY_POPULATIONS_V20A = frame_v19a.BASE_CATEGORY_POPULATIONS_V19A
CANDIDATE_ONLY_TIER_POPULATIONS_V20A = (
    frame_v19a.CANDIDATE_ONLY_TIER_POPULATIONS_V19A
)
PAIRED_TIER_POPULATIONS_V20A = frame_v19a.PAIRED_TIER_POPULATIONS_V19A
PATCH_TIER_POPULATIONS_V20A = frame_v19a.PATCH_TIER_POPULATIONS_V19A
PRODUCTION_TOPIC_QUOTAS_V20A = frame_v19a.PRODUCTION_TOPIC_QUOTAS_V19A
BASE_CATEGORY_QUOTA_V20A = 6
ARM_REQUESTS_PER_PANEL_V20A = {
    "production_only": 24,
    "patch_tier_2_only": 25,
    "patch_tiers_2_3": 26,
    "patch_all_tiers": 27,
}
ARM_POPULATIONS_V20A = {
    "production_only": 272,
    "patch_tier_2_only": 279,
    "patch_tiers_2_3": 287,
    "patch_all_tiers": 295,
}
FLOW_SEED_V20A = 20260729
CANDIDATE_ASSIGNMENT_SEED_V20A = 20260730
BASE_POPULATION_V20A = 272
BASE_SELECTION_COUNT_V20A = 240
V19_BASE_SELECTION_COUNT_V20A = 240
THEORETICAL_MINIMUM_V19_OVERLAP_V20A = 208
THEORETICAL_MAXIMUM_FRESH_BASES_V20A = 32


canonical_sha256 = frame_v19a.canonical_sha256
file_sha256 = frame_v19a.file_sha256
identity_root_sha256 = frame_v19a.identity_root_sha256


def _without_self(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_bound_v19a_frame() -> dict:
    expected = {
        Path(frame_v19a.__file__).resolve(): FRAME_V19A_FILE_SHA256_V20A,
        frame_v19a.OUTPUT_PATH_V19A: FRAME_CERTIFICATE_V19A_FILE_SHA256_V20A,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v20a frozen V19A frame binding changed")
    certificate = json.loads(
        frame_v19a.OUTPUT_PATH_V19A.read_text(encoding="utf-8")
    )
    frame_v19a.validate_certificate_v19a(certificate)
    if (
        certificate.get("content_sha256_before_self_field")
        != FRAME_CERTIFICATE_V19A_CONTENT_SHA256_V20A
    ):
        raise RuntimeError("v20a frozen V19A frame content changed")
    return certificate


def load_frozen_frame_v20a() -> tuple[dict, dict]:
    """Reconstruct the frozen train-side joint frame and disjoint tiers."""
    _verify_bound_v19a_frame()
    return frame_v19a.load_frozen_frame_v19a()


def reconstruct_v19_base_indices_v20a(frame: dict, tiers: dict) -> set[int]:
    """Rebuild the sealed V19A selection without persisting row content."""
    flow = frame_v19a.solve_base_flow_v19a(frame, tiers)
    selected = {
        index
        for panel in PANEL_NAMES_V20A
        for index in flow["selected_base"][panel]
    }
    if len(selected) != V19_BASE_SELECTION_COUNT_V20A:
        raise RuntimeError("v20a reconstructed V19A base selection changed")
    return selected


def _objective_cost_v20a(value) -> float:
    digest = canonical_sha256({
        "schema": "eggroll-es-nested-tier-interaction-flow-objective-v20a",
        "seed": FLOW_SEED_V20A,
        "value": value,
    })
    return int(digest[:12], 16) / float(16**12)


def solve_base_flow_v20a(frame: dict, tiers: dict) -> dict:
    """Solve ten exact panels while attaining the overlap lower bound."""
    components = frame["joint_components"]
    production_units = frame["side_units"]["production"]
    base_indices = sorted(
        tiers["base_category"], key=lambda index: components[index]["joint_id"]
    )
    prior_selected = reconstruct_v19_base_indices_v20a(frame, tiers)
    fresh_indices = set(base_indices) - prior_selected
    theoretical_minimum = max(
        0,
        BASE_SELECTION_COUNT_V20A
        + V19_BASE_SELECTION_COUNT_V20A
        - len(base_indices),
    )
    if (
        len(base_indices) != BASE_POPULATION_V20A
        or len(fresh_indices) != THEORETICAL_MAXIMUM_FRESH_BASES_V20A
        or theoretical_minimum != THEORETICAL_MINIMUM_V19_OVERLAP_V20A
    ):
        raise RuntimeError("v20a overlap cardinality proof changed")

    variable_index = {}
    objective = []
    for component_index in base_indices:
        for panel_index, panel in enumerate(PANEL_NAMES_V20A):
            variable_index[(component_index, panel_index)] = len(objective)
            objective.append(_objective_cost_v20a({
                "joint_id": components[component_index]["joint_id"],
                "panel": panel,
            }))

    constraints = []
    lower = []
    upper = []

    def add_constraint(values, low, high):
        constraints.append(values)
        lower.append(low)
        upper.append(high)

    for component_index in base_indices:
        add_constraint({
            variable_index[(component_index, panel_index)]: 1.0
            for panel_index in range(len(PANEL_NAMES_V20A))
        }, 0.0, 1.0)
    for panel_index in range(len(PANEL_NAMES_V20A)):
        for category in BASE_CATEGORIES_V20A:
            add_constraint({
                variable_index[(component_index, panel_index)]: 1.0
                for component_index in base_indices
                if tiers["base_category"][component_index] == category
            }, BASE_CATEGORY_QUOTA_V20A, BASE_CATEGORY_QUOTA_V20A)
        for topic, quota in PRODUCTION_TOPIC_QUOTAS_V20A.items():
            add_constraint({
                variable_index[(component_index, panel_index)]: 1.0
                for component_index in base_indices
                if production_units[
                    components[component_index]["production_unit"]
                ]["stratum"] == topic
            }, quota, quota)
    add_constraint({
        variable_index[(component_index, panel_index)]: 1.0
        for component_index in fresh_indices
        for panel_index in range(len(PANEL_NAMES_V20A))
    }, len(fresh_indices), len(fresh_indices))

    row_indices = []
    column_indices = []
    coefficients = []
    for row_index, row in enumerate(constraints):
        for column_index, coefficient in row.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            coefficients.append(coefficient)
    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(constraints), len(objective)),
    ).tocsr()
    result = milp(
        c=np.asarray(objective),
        integrality=np.ones(len(objective)),
        bounds=Bounds(np.zeros(len(objective)), np.ones(len(objective))),
        constraints=LinearConstraint(
            matrix, np.asarray(lower), np.asarray(upper)
        ),
        options={"presolve": True, "time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.status != 0 or result.x is None:
        raise RuntimeError(
            "v20a exact maximum-freshness flow infeasible: "
            f"status={result.status}; quotas were not weakened"
        )
    rounded = np.rint(result.x)
    if np.max(np.abs(result.x - rounded)) > 1e-7:
        raise RuntimeError("v20a exact flow returned fractional values")
    selected = {panel: [] for panel in PANEL_NAMES_V20A}
    for (component_index, panel_index), column_index in variable_index.items():
        if rounded[column_index] > 0.5:
            selected[PANEL_NAMES_V20A[panel_index]].append(component_index)
    selected_global = {
        index for values in selected.values() for index in values
    }
    overlap = len(selected_global & prior_selected)
    fresh = len(selected_global - prior_selected)
    if (
        len(selected_global) != BASE_SELECTION_COUNT_V20A
        or overlap != THEORETICAL_MINIMUM_V19_OVERLAP_V20A
        or fresh != THEORETICAL_MAXIMUM_FRESH_BASES_V20A
        or not fresh_indices.issubset(selected_global)
    ):
        raise RuntimeError("v20a maximum-freshness overlap contract changed")
    return {
        "selected_base": selected,
        "prior_selected": prior_selected,
        "fresh_available": fresh_indices,
        "solver": {
            "implementation": "scipy.optimize.milp_highs_binary_constrained_flow",
            "scipy_version": scipy.__version__,
            "success": bool(result.success),
            "status": int(result.status),
            "mip_gap": float(result.mip_gap),
            "objective": float(result.fun),
            "variable_count": len(objective),
            "constraint_count": len(constraints),
            "quota_relaxation_used": False,
            "fallback_solver_used": False,
        },
        "overlap": {
            "base_population": len(base_indices),
            "v19_selected": len(prior_selected),
            "v20_selected": len(selected_global),
            "cardinality_lower_bound": theoretical_minimum,
            "achieved_v19_overlap": overlap,
            "maximum_fresh_available": len(fresh_indices),
            "achieved_fresh": fresh,
            "all_v19_unused_components_selected": True,
            "theoretical_minimum_overlap_attained": True,
            "v19_selected_identity_root_sha256": identity_root_sha256(
                components[index]["joint_id"] for index in prior_selected
            ),
            "v20_selected_identity_root_sha256": identity_root_sha256(
                components[index]["joint_id"] for index in selected_global
            ),
            "fresh_identity_root_sha256": identity_root_sha256(
                components[index]["joint_id"] for index in fresh_indices
            ),
        },
    }


def assign_candidate_only_v20a(frame: dict, tiers: dict) -> dict:
    """Exhaust each candidate-only tier before deterministic reuse."""
    components = frame["joint_components"]
    assignments = {panel: {} for panel in PANEL_NAMES_V20A}
    summary = {}
    for tier in (1, 2, 3):
        indices = [
            index
            for index, assigned_tier in tiers["candidate_only_layer"].items()
            if assigned_tier == tier
        ]
        unique_order = sorted(indices, key=lambda index: canonical_sha256({
            "schema": "eggroll-es-candidate-exhaustion-order-v20a",
            "seed": CANDIDATE_ASSIGNMENT_SEED_V20A,
            "tier": tier,
            "joint_id": components[index]["joint_id"],
        }))
        reuse_order = sorted(indices, key=lambda index: canonical_sha256({
            "schema": "eggroll-es-candidate-reuse-order-v20a",
            "seed": CANDIDATE_ASSIGNMENT_SEED_V20A,
            "tier": tier,
            "joint_id": components[index]["joint_id"],
        }))
        reuse_count = len(PANEL_NAMES_V20A) - len(unique_order)
        sequence = unique_order + reuse_order[:reuse_count]
        if (
            len(sequence) != len(PANEL_NAMES_V20A)
            or set(sequence[:len(unique_order)]) != set(indices)
        ):
            raise RuntimeError("v20a candidate-only exhaustion changed")
        for panel, component_index in zip(PANEL_NAMES_V20A, sequence):
            assignments[panel][tier] = component_index
        summary[str(tier)] = {
            "population": len(indices),
            "panel_assignments": len(sequence),
            "unique_components_used": len(set(sequence)),
            "reuse_assignments_after_exhaustion": reuse_count,
            "all_unique_assigned_before_first_reuse": True,
            "ordered_assignment_identity_sha256": canonical_sha256([
                components[index]["joint_id"] for index in sequence
            ]),
        }
    return {"assignments": assignments, "summary": summary}


def _representative_for_arm_v20a(
    component_index: int,
    active_tiers: tuple[int, ...],
    frame: dict,
    tiers: dict,
) -> tuple[str, str]:
    component = frame["joint_components"][component_index]
    if component["relation"] == "candidate_only":
        if tiers["candidate_only_layer"].get(component_index) not in active_tiers:
            raise RuntimeError("v20a arm received an inactive candidate-only tier")
        side = "candidate"
    elif tiers["paired_tier"].get(component_index) in active_tiers:
        side = "candidate"
    else:
        side = "production"
    unit = frame["side_units"][side][component[f"{side}_unit"]]
    return side, unit["representative_row_sha256"]


def materialize_panel_contracts_v20a(
    frame: dict,
    tiers: dict,
    base_flow: dict,
    candidate_assignment: dict,
) -> list[dict]:
    """Materialize aggregate identities and nested paired-arm commitments."""
    components = frame["joint_components"]
    production_units = frame["side_units"]["production"]
    summaries = []
    selected_global = []
    for panel in PANEL_NAMES_V20A:
        base = base_flow["selected_base"][panel]
        base_categories = Counter(tiers["base_category"][index] for index in base)
        topics = Counter(
            production_units[components[index]["production_unit"]]["stratum"]
            for index in base
        )
        if (
            base_categories != {
                category: BASE_CATEGORY_QUOTA_V20A
                for category in BASE_CATEGORIES_V20A
            }
            or topics != PRODUCTION_TOPIC_QUOTAS_V20A
        ):
            raise RuntimeError("v20a solved panel violates an exact quota")
        selected_global.extend(base)
        base_root = identity_root_sha256(
            components[index]["joint_id"] for index in base
        )
        arms = {}
        for arm in ARM_ORDER_V20A:
            active_tiers = ARM_ACTIVE_TIERS_V20A[arm]
            active = list(base) + [
                candidate_assignment["assignments"][panel][tier]
                for tier in active_tiers
            ]
            representatives = []
            representative_sides = Counter()
            for component_index in active:
                side, row_digest = _representative_for_arm_v20a(
                    component_index, active_tiers, frame, tiers
                )
                representative_sides[side] += 1
                representatives.append(
                    f"{components[component_index]['joint_id']}:{side}:{row_digest}"
                )
            if (
                len(active) != ARM_REQUESTS_PER_PANEL_V20A[arm]
                or len({components[index]["joint_id"] for index in active})
                != len(active)
            ):
                raise RuntimeError("v20a arm request or duplicate contract changed")
            arms[arm] = {
                "active_patch_tiers": list(active_tiers),
                "requests": len(active),
                "population_denominator": ARM_POPULATIONS_V20A[arm],
                "production_representatives": representative_sides["production"],
                "candidate_representatives": representative_sides["candidate"],
                "shared_base_joint_component_identity_root_sha256": base_root,
                "active_joint_component_identity_root_sha256": (
                    identity_root_sha256(
                        components[index]["joint_id"] for index in active
                    )
                ),
                "representative_assignment_root_sha256": (
                    identity_root_sha256(representatives)
                ),
                "same_arm_paired_duplicate_count": 0,
            }
        assigned = {
            str(tier): components[
                candidate_assignment["assignments"][panel][tier]
            ]["joint_id"]
            for tier in (1, 2, 3)
        }
        summaries.append({
            "name": panel,
            "role": (
                "optimization"
                if panel in OPTIMIZATION_PANELS_V20A
                else "train_only_screen"
            ),
            "base_components": len(base),
            "base_category_counts": dict(base_categories),
            "production_topic_counts": dict(topics),
            "selected_base_joint_component_identity_root_sha256": base_root,
            "candidate_only_tier_assignment_sha256": canonical_sha256(assigned),
            "nested_arm_identity_contract_sha256": canonical_sha256({
                arm: {
                    "active_tiers": list(ARM_ACTIVE_TIERS_V20A[arm]),
                    "base_root": arms[arm][
                        "shared_base_joint_component_identity_root_sha256"
                    ],
                    "active_root": arms[arm][
                        "active_joint_component_identity_root_sha256"
                    ],
                }
                for arm in ARM_ORDER_V20A
            }),
            "arms": arms,
        })
    if len(selected_global) != 240 or len(set(selected_global)) != 240:
        raise RuntimeError("v20a base panels are not globally disjoint")
    return summaries


def build_certificate_v20a() -> dict:
    bound_v19a = _verify_bound_v19a_frame()
    frame, tiers = load_frozen_frame_v20a()
    base_flow = solve_base_flow_v20a(frame, tiers)
    candidate_assignment = assign_candidate_only_v20a(frame, tiers)
    panels = materialize_panel_contracts_v20a(
        frame, tiers, base_flow, candidate_assignment
    )
    components = frame["joint_components"]
    candidate_sequence = [
        {
            "panel": panel,
            "tier": tier,
            "joint_id": components[
                candidate_assignment["assignments"][panel][tier]
            ]["joint_id"],
        }
        for panel in PANEL_NAMES_V20A
        for tier in (1, 2, 3)
    ]
    certificate = {
        "schema": "eggroll-es-nested-tier-interaction-flow-certificate-v20a",
        "status": "feasible_train_only_attribution_no_runtime_authorization",
        "arm_order": list(ARM_ORDER_V20A),
        "inputs": {
            "candidate_v298": {
                "path": str(frame_v19a.frame_v18a.CANDIDATE_PATH_V18A),
                "rows": frame_v19a.frame_v18a.CANDIDATE_ROWS_V18A,
                "file_sha256": frame_v19a.frame_v18a.CANDIDATE_SHA256_V18A,
                "manifest_path": str(
                    frame_v19a.frame_v18a.CANDIDATE_MANIFEST_PATH_V18A
                ),
                "manifest_file_sha256": (
                    frame_v19a.frame_v18a.CANDIDATE_MANIFEST_SHA256_V18A
                ),
            },
            "production": {
                "path": str(frame_v19a.frame_v18a.PRODUCTION_PATH_V18A),
                "rows": frame_v19a.frame_v18a.PRODUCTION_ROWS_V18A,
                "file_sha256": frame_v19a.frame_v18a.PRODUCTION_SHA256_V18A,
            },
            "v19a_frame_builder": {
                "path": str(Path(frame_v19a.__file__).resolve()),
                "file_sha256": FRAME_V19A_FILE_SHA256_V20A,
            },
            "v19a_frame_certificate": {
                "path": str(frame_v19a.OUTPUT_PATH_V19A),
                "file_sha256": FRAME_CERTIFICATE_V19A_FILE_SHA256_V20A,
                "content_sha256": FRAME_CERTIFICATE_V19A_CONTENT_SHA256_V20A,
            },
        },
        "joint_frame": {
            "joint_component_count": 295,
            "relation_counts": frame["relation_counts"],
            "match_class_counts": frame["match_counts"],
            "production_base_components": 272,
            "eligible_shared_document_paired_components": 202,
            "candidate_only_components": 23,
            "ambiguous_pair_action": "always_production_fallback",
            "v19a_joint_component_identity_root_sha256": bound_v19a[
                "joint_frame"
            ]["v18a_joint_component_identity_root_sha256"],
        },
        "nested_patch_arms": {
            "active_tiers": {
                arm: list(tiers_active)
                for arm, tiers_active in ARM_ACTIVE_TIERS_V20A.items()
            },
            "tier2_shared_by_last_three_arms": True,
            "tier3_shared_by_last_two_arms": True,
            "tier1_added_only_after_tiers_2_3": True,
            "conditional_tier3_contrast": "patch_tiers_2_3_minus_patch_tier_2_only",
            "conditional_tier1_contrast": "patch_all_tiers_minus_patch_tiers_2_3",
            "all_patch_contrast": "patch_all_tiers_minus_production_only",
            "tier_membership_reused_exactly_from_v18a_v19a": True,
            "paired_tier_populations": {
                str(key): value for key, value in PAIRED_TIER_POPULATIONS_V20A.items()
            },
            "candidate_only_tier_populations": {
                str(key): value
                for key, value in CANDIDATE_ONLY_TIER_POPULATIONS_V20A.items()
            },
        },
        "constrained_base_flow": {
            "seed": FLOW_SEED_V20A,
            "solver": base_flow["solver"],
            "panel_count": 10,
            "optimization_panel_count": 6,
            "train_only_screen_panel_count": 4,
            "base_category_quota_per_panel": {
                category: BASE_CATEGORY_QUOTA_V20A
                for category in BASE_CATEGORIES_V20A
            },
            "production_topic_quota_per_panel": PRODUCTION_TOPIC_QUOTAS_V20A,
            "selected_base_components": 240,
            "globally_panel_disjoint_base_components": True,
            "quota_seed_or_solver_relaxation_used": False,
            "infeasible_action": "abort_v20a_without_relaxation_or_fallback",
            "v19_overlap": base_flow["overlap"],
        },
        "candidate_only_assignment": {
            "seed": CANDIDATE_ASSIGNMENT_SEED_V20A,
            "quota_per_active_tier_per_panel": 1,
            "tier_summary": candidate_assignment["summary"],
            "every_unique_item_before_deterministic_reuse": True,
            "total_panel_tier_assignments": 30,
            "unique_candidate_only_components": 23,
            "reuse_assignments_after_exhaustion": 7,
            "ordered_assignment_sha256": canonical_sha256(candidate_sequence),
        },
        "estimand": {
            "semantics": (
                "nested_tiers_substitute_candidate_for_their_paired_base_"
                "component_and_add_one_q1_candidate_only_observation_per_"
                "active_tier_while_all_inactive_components_use_production"
            ),
            "arm_population_denominators": ARM_POPULATIONS_V20A,
            "arm_requests_per_panel": ARM_REQUESTS_PER_PANEL_V20A,
            "base_population_ht_strata": {
                category: {
                    "population": population,
                    "per_panel_quota": BASE_CATEGORY_QUOTA_V20A,
                    "horvitz_thompson_weight": population / BASE_CATEGORY_QUOTA_V20A,
                }
                for category, population in BASE_CATEGORY_POPULATIONS_V20A.items()
            },
            "candidate_only_ht_strata": {
                str(tier): {
                    "population": population,
                    "per_active_arm_panel_quota": 1,
                    "horvitz_thompson_weight": float(population),
                    "active_for_arms": [
                        arm for arm in ARM_ORDER_V20A
                        if tier in ARM_ACTIVE_TIERS_V20A[arm]
                    ],
                }
                for tier, population in CANDIDATE_ONLY_TIER_POPULATIONS_V20A.items()
            },
            "arm_total": (
                "sum_HT_weight_times_selected_active_component_score_over_"
                "the_arm_specific_base_and_candidate_only_strata"
            ),
            "arm_mean": "arm_total_divided_by_exact_arm_population_denominator",
            "plain_request_mean_used": False,
            "shared_component_counted_once_per_arm": True,
            "same_arm_paired_duplicate_count": 0,
        },
        "panels": panels,
        "firewall": {
            "contains_question_answer_prompt_response_or_row_content": False,
            "contains_evaluation_content": False,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "runtime_launch_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "dataset_promotion_authorized": False,
        },
    }
    certificate["content_sha256_before_self_field"] = canonical_sha256(certificate)
    validate_certificate_v20a(certificate)
    return certificate


def validate_certificate_v20a(value: dict) -> dict:
    flow = value.get("constrained_base_flow", {})
    solver = flow.get("solver", {})
    overlap = flow.get("v19_overlap", {})
    candidate = value.get("candidate_only_assignment", {})
    panels = value.get("panels", [])
    if (
        value.get("schema")
        != "eggroll-es-nested-tier-interaction-flow-certificate-v20a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("arm_order") != list(ARM_ORDER_V20A)
        or value.get("estimand", {}).get("arm_population_denominators")
        != ARM_POPULATIONS_V20A
        or value.get("estimand", {}).get("arm_requests_per_panel")
        != ARM_REQUESTS_PER_PANEL_V20A
        or solver.get("status") != 0
        or solver.get("success") is not True
        or not math.isclose(solver.get("mip_gap", math.inf), 0.0, abs_tol=1e-15)
        or solver.get("quota_relaxation_used") is not False
        or solver.get("fallback_solver_used") is not False
        or flow.get("quota_seed_or_solver_relaxation_used") is not False
        or overlap.get("cardinality_lower_bound")
        != THEORETICAL_MINIMUM_V19_OVERLAP_V20A
        or overlap.get("achieved_v19_overlap")
        != THEORETICAL_MINIMUM_V19_OVERLAP_V20A
        or overlap.get("achieved_fresh") != THEORETICAL_MAXIMUM_FRESH_BASES_V20A
        or overlap.get("all_v19_unused_components_selected") is not True
        or overlap.get("theoretical_minimum_overlap_attained") is not True
        or candidate.get("every_unique_item_before_deterministic_reuse") is not True
        or candidate.get("unique_candidate_only_components") != 23
        or candidate.get("reuse_assignments_after_exhaustion") != 7
        or len(panels) != 10
        or tuple(panel.get("name") for panel in panels) != PANEL_NAMES_V20A
        or any(
            panel.get("base_category_counts") != {
                category: BASE_CATEGORY_QUOTA_V20A
                for category in BASE_CATEGORIES_V20A
            }
            or panel.get("production_topic_counts") != PRODUCTION_TOPIC_QUOTAS_V20A
            or set(panel.get("arms", {})) != set(ARM_ORDER_V20A)
            or len({
                arm.get("shared_base_joint_component_identity_root_sha256")
                for arm in panel.get("arms", {}).values()
            }) != 1
            or any(
                arm.get("same_arm_paired_duplicate_count") != 0
                for arm in panel.get("arms", {}).values()
            )
            for panel in panels
        )
        or value.get("firewall", {}).get("runtime_launch_authorized") is not False
        or value.get("firewall", {}).get("model_update_authorized") is not False
        or value.get("firewall", {}).get("checkpoint_write_authorized") is not False
        or value.get("firewall", {}).get("dataset_promotion_authorized") is not False
        or value.get("firewall", {}).get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
    ):
        raise RuntimeError("v20a nested interaction flow certificate changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v20a flow certificate already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V20A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V20A:
        raise ValueError("v20a flow certificate output path changed")
    certificate = build_certificate_v20a()
    _exclusive_write(OUTPUT_PATH_V20A, certificate)
    result = {
        "schema": "eggroll-es-nested-tier-interaction-flow-build-v20a",
        "output": str(OUTPUT_PATH_V20A),
        "file_sha256": file_sha256(OUTPUT_PATH_V20A),
        "content_sha256": certificate["content_sha256_before_self_field"],
        "solver_status": certificate["constrained_base_flow"]["solver"]["status"],
        "solver_mip_gap": certificate["constrained_base_flow"]["solver"]["mip_gap"],
        "achieved_v19_overlap": certificate["constrained_base_flow"][
            "v19_overlap"
        ]["achieved_v19_overlap"],
        "runtime_launch_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
