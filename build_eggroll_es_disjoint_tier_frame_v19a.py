#!/usr/bin/env python3
"""Build the aggregate-only V19A disjoint-tier attribution frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import build_eggroll_es_overlay_frame_v18a as frame_v18a


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V19A = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v19a/"
    "production_patch_disjoint_tier_panels_v19a.json"
).resolve()

FRAME_V18A_FILE_SHA256_V19A = (
    "73d7c71233946996d64d776fb68c31609ad512379a15a75fae46dd0d8c395ba0"
)
FRAME_CERTIFICATE_V18A_FILE_SHA256_V19A = (
    "0887f936fd00d205fab46d490810732d16ddc2b34d96fdc507d43c20dec60f8e"
)
FRAME_CERTIFICATE_V18A_CONTENT_SHA256_V19A = (
    "6a844851cc4e5a07c08b338c5e48f3b5ab58dbeb10c765570f3fa02f20c77b3b"
)

PANEL_NAMES_V19A = tuple(
    [f"optimization_{index}" for index in range(6)]
    + [f"train_screen_{index}" for index in range(4)]
)
OPTIMIZATION_PANELS_V19A = PANEL_NAMES_V19A[:6]
TRAIN_SCREEN_PANELS_V19A = PANEL_NAMES_V19A[6:]
ARM_ORDER_V19A = (
    "production_only",
    "patch_tier_1_only",
    "patch_tier_2_only",
    "patch_tier_3_only",
)
ARM_ACTIVE_TIER_V19A = {
    "production_only": None,
    "patch_tier_1_only": 1,
    "patch_tier_2_only": 2,
    "patch_tier_3_only": 3,
}
BASE_CATEGORIES_V19A = frame_v18a.BASE_CATEGORIES_V18A
BASE_CATEGORY_POPULATIONS_V19A = frame_v18a.BASE_CATEGORY_POPULATIONS_V18A
CANDIDATE_ONLY_TIER_POPULATIONS_V19A = (
    frame_v18a.CANDIDATE_ONLY_LAYER_POPULATIONS_V18A
)
PAIRED_TIER_POPULATIONS_V19A = {1: 67, 2: 68, 3: 67}
PATCH_TIER_POPULATIONS_V19A = {1: 75, 2: 75, 3: 75}
PRODUCTION_TOPIC_QUOTAS_V19A = {
    "safety_consent": 3,
    "technique": 8,
    "equipment_material": 2,
    "resources_general": 11,
}
BASE_CATEGORY_QUOTA_V19A = 6
CANDIDATE_ONLY_TIER_QUOTA_V19A = 1
ARM_REQUESTS_PER_PANEL_V19A = {
    "production_only": 24,
    "patch_tier_1_only": 25,
    "patch_tier_2_only": 25,
    "patch_tier_3_only": 25,
}
ARM_POPULATIONS_V19A = {
    "production_only": 272,
    "patch_tier_1_only": 280,
    "patch_tier_2_only": 279,
    "patch_tier_3_only": 280,
}
FLOW_SEED_V19A = 20260725
CANDIDATE_ASSIGNMENT_SEED_V19A = 20260726


canonical_sha256 = frame_v18a.canonical_sha256
file_sha256 = frame_v18a.file_sha256
identity_root_sha256 = frame_v18a.identity_root_sha256


def _without_self(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _verify_bound_v18a_frame() -> dict:
    expected = {
        Path(frame_v18a.__file__).resolve(): FRAME_V18A_FILE_SHA256_V19A,
        frame_v18a.OUTPUT_PATH_V18A: (
            FRAME_CERTIFICATE_V18A_FILE_SHA256_V19A
        ),
        frame_v18a.CANDIDATE_PATH_V18A: frame_v18a.CANDIDATE_SHA256_V18A,
        frame_v18a.CANDIDATE_MANIFEST_PATH_V18A: (
            frame_v18a.CANDIDATE_MANIFEST_SHA256_V18A
        ),
        frame_v18a.PRODUCTION_PATH_V18A: frame_v18a.PRODUCTION_SHA256_V18A,
        Path(frame_v18a.sampler_v13.__file__).resolve(): (
            frame_v18a.SAMPLER_SHA256_V18A
        ),
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v19a frozen V18A frame or train input changed")
    certificate = json.loads(
        frame_v18a.OUTPUT_PATH_V18A.read_text(encoding="utf-8")
    )
    frame_v18a.validate_certificate_v18a(certificate)
    if (
        certificate.get("content_sha256_before_self_field")
        != FRAME_CERTIFICATE_V18A_CONTENT_SHA256_V19A
    ):
        raise RuntimeError("v19a bound V18A frame content changed")
    return certificate


def load_frozen_frame_v19a() -> tuple[dict, dict]:
    """Reconstruct only the frozen train-side V18A joint frame and tiers."""
    _verify_bound_v18a_frame()
    candidate, production = frame_v18a.load_bound_rows_v18a()
    frame = frame_v18a.build_joint_frame_v18a(candidate, production)
    tiers = frame_v18a.assign_patch_layers_v18a(frame)
    return frame, tiers


def _objective_cost_v19a(value) -> float:
    digest = canonical_sha256({
        "schema": "eggroll-es-disjoint-tier-flow-objective-v19a",
        "seed": FLOW_SEED_V19A,
        "value": value,
    })
    return int(digest[:12], 16) / float(16**12)


def solve_base_flow_v19a(frame: dict, tiers: dict) -> dict:
    """Solve ten globally-disjoint base panels without relaxing quotas."""
    components = frame["joint_components"]
    production_units = frame["side_units"]["production"]
    base_indices = sorted(
        tiers["base_category"], key=lambda index: components[index]["joint_id"]
    )
    variable_index = {}
    objective = []
    for component_index in base_indices:
        for panel_index, panel in enumerate(PANEL_NAMES_V19A):
            variable_index[(component_index, panel_index)] = len(objective)
            objective.append(_objective_cost_v19a({
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
            for panel_index in range(len(PANEL_NAMES_V19A))
        }, 0.0, 1.0)
    for panel_index in range(len(PANEL_NAMES_V19A)):
        for category in BASE_CATEGORIES_V19A:
            add_constraint({
                variable_index[(component_index, panel_index)]: 1.0
                for component_index in base_indices
                if tiers["base_category"][component_index] == category
            }, BASE_CATEGORY_QUOTA_V19A, BASE_CATEGORY_QUOTA_V19A)
        for topic, quota in PRODUCTION_TOPIC_QUOTAS_V19A.items():
            add_constraint({
                variable_index[(component_index, panel_index)]: 1.0
                for component_index in base_indices
                if production_units[
                    components[component_index]["production_unit"]
                ]["stratum"] == topic
            }, quota, quota)

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
            "v19a exact topic-by-tier flow infeasible: "
            f"status={result.status}; quotas were not weakened"
        )
    rounded = np.rint(result.x)
    if np.max(np.abs(result.x - rounded)) > 1e-7:
        raise RuntimeError("v19a exact flow returned fractional values")
    selected = {panel: [] for panel in PANEL_NAMES_V19A}
    for (component_index, panel_index), column_index in variable_index.items():
        if rounded[column_index] > 0.5:
            selected[PANEL_NAMES_V19A[panel_index]].append(component_index)
    return {
        "selected_base": selected,
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
    }


def assign_candidate_only_v19a(frame: dict, tiers: dict) -> dict:
    """Exhaust each tier's unique candidate-only pool before reuse."""
    components = frame["joint_components"]
    by_tier = {
        tier: [
            index
            for index, assigned_tier in tiers["candidate_only_layer"].items()
            if assigned_tier == tier
        ]
        for tier in (1, 2, 3)
    }
    assignments = {panel: {} for panel in PANEL_NAMES_V19A}
    summary = {}
    for tier, indices in by_tier.items():
        unique_order = sorted(
            indices,
            key=lambda index: canonical_sha256({
                "schema": "eggroll-es-candidate-exhaustion-order-v19a",
                "seed": CANDIDATE_ASSIGNMENT_SEED_V19A,
                "tier": tier,
                "joint_id": components[index]["joint_id"],
            }),
        )
        reuse_order = sorted(
            indices,
            key=lambda index: canonical_sha256({
                "schema": "eggroll-es-candidate-reuse-order-v19a",
                "seed": CANDIDATE_ASSIGNMENT_SEED_V19A,
                "tier": tier,
                "joint_id": components[index]["joint_id"],
            }),
        )
        reuse_count = len(PANEL_NAMES_V19A) - len(unique_order)
        sequence = unique_order + reuse_order[:reuse_count]
        if (
            len(sequence) != len(PANEL_NAMES_V19A)
            or len(set(sequence[: len(unique_order)])) != len(unique_order)
            or set(sequence[: len(unique_order)]) != set(indices)
        ):
            raise RuntimeError("v19a candidate-only exhaustion contract changed")
        for panel, component_index in zip(PANEL_NAMES_V19A, sequence):
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


def _representative_for_arm_v19a(
    component_index: int,
    arm: str,
    frame: dict,
    tiers: dict,
) -> tuple[str, str]:
    component = frame["joint_components"][component_index]
    active_tier = ARM_ACTIVE_TIER_V19A[arm]
    if component["relation"] == "candidate_only":
        if tiers["candidate_only_layer"].get(component_index) != active_tier:
            raise RuntimeError("v19a arm received another tier's candidate-only row")
        side = "candidate"
    elif (
        active_tier is not None
        and tiers["paired_tier"].get(component_index) == active_tier
    ):
        side = "candidate"
    else:
        side = "production"
    unit = frame["side_units"][side][component[f"{side}_unit"]]
    return side, unit["representative_row_sha256"]


def materialize_panel_contracts_v19a(
    frame: dict,
    tiers: dict,
    base_flow: dict,
    candidate_assignment: dict,
) -> list[dict]:
    """Materialize only aggregate identities; never persist row content."""
    components = frame["joint_components"]
    production_units = frame["side_units"]["production"]
    summaries = []
    selected_base_global = []
    for panel in PANEL_NAMES_V19A:
        base = base_flow["selected_base"][panel]
        base_categories = Counter(tiers["base_category"][index] for index in base)
        topics = Counter(
            production_units[components[index]["production_unit"]]["stratum"]
            for index in base
        )
        if (
            base_categories
            != {
                category: BASE_CATEGORY_QUOTA_V19A
                for category in BASE_CATEGORIES_V19A
            }
            or topics != PRODUCTION_TOPIC_QUOTAS_V19A
        ):
            raise RuntimeError("v19a solved panel violates an exact quota")
        selected_base_global.extend(base)
        arms = {}
        for arm in ARM_ORDER_V19A:
            active_tier = ARM_ACTIVE_TIER_V19A[arm]
            active = list(base)
            if active_tier is not None:
                active.append(candidate_assignment["assignments"][panel][active_tier])
            representatives = []
            representative_sides = Counter()
            for component_index in active:
                side, row_sha256 = _representative_for_arm_v19a(
                    component_index, arm, frame, tiers
                )
                representative_sides[side] += 1
                representatives.append(
                    f"{components[component_index]['joint_id']}:{side}:{row_sha256}"
                )
            if (
                len(active) != ARM_REQUESTS_PER_PANEL_V19A[arm]
                or len({components[index]["joint_id"] for index in active})
                != len(active)
            ):
                raise RuntimeError("v19a arm request or duplicate contract changed")
            arms[arm] = {
                "active_patch_tier": active_tier,
                "requests": len(active),
                "population_denominator": ARM_POPULATIONS_V19A[arm],
                "production_representatives": representative_sides["production"],
                "candidate_representatives": representative_sides["candidate"],
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
        candidate_components = {
            str(tier): components[
                candidate_assignment["assignments"][panel][tier]
            ]["joint_id"]
            for tier in (1, 2, 3)
        }
        summaries.append({
            "name": panel,
            "role": (
                "optimization"
                if panel in OPTIMIZATION_PANELS_V19A
                else "train_only_screen"
            ),
            "base_components": len(base),
            "base_category_counts": dict(base_categories),
            "production_topic_counts": dict(topics),
            "candidate_only_tier_quota": {"1": 1, "2": 1, "3": 1},
            "selected_base_joint_component_identity_root_sha256": (
                identity_root_sha256(
                    components[index]["joint_id"] for index in base
                )
            ),
            "candidate_only_tier_assignment_sha256": canonical_sha256(
                candidate_components
            ),
            "arms": arms,
        })
    if len(selected_base_global) != 240 or len(set(selected_base_global)) != 240:
        raise RuntimeError("v19a base panels are not globally disjoint")
    return summaries


def build_certificate_v19a() -> dict:
    bound_v18a = _verify_bound_v18a_frame()
    frame, tiers = load_frozen_frame_v19a()
    base_flow = solve_base_flow_v19a(frame, tiers)
    candidate_assignment = assign_candidate_only_v19a(frame, tiers)
    panels = materialize_panel_contracts_v19a(
        frame, tiers, base_flow, candidate_assignment
    )
    components = frame["joint_components"]
    candidate_assignment_sequence = [
        {
            "panel": panel,
            "tier": tier,
            "joint_id": components[
                candidate_assignment["assignments"][panel][tier]
            ]["joint_id"],
        }
        for panel in PANEL_NAMES_V19A
        for tier in (1, 2, 3)
    ]
    certificate = {
        "schema": "eggroll-es-disjoint-tier-flow-certificate-v19a",
        "status": "feasible_train_only_attribution_no_runtime_authorization",
        "arm_order": list(ARM_ORDER_V19A),
        "inputs": {
            "candidate_v298": {
                "path": str(frame_v18a.CANDIDATE_PATH_V18A),
                "rows": frame_v18a.CANDIDATE_ROWS_V18A,
                "file_sha256": frame_v18a.CANDIDATE_SHA256_V18A,
                "manifest_path": str(frame_v18a.CANDIDATE_MANIFEST_PATH_V18A),
                "manifest_file_sha256": (
                    frame_v18a.CANDIDATE_MANIFEST_SHA256_V18A
                ),
            },
            "production": {
                "path": str(frame_v18a.PRODUCTION_PATH_V18A),
                "rows": frame_v18a.PRODUCTION_ROWS_V18A,
                "file_sha256": frame_v18a.PRODUCTION_SHA256_V18A,
            },
            "v18a_frame_builder": {
                "path": str(Path(frame_v18a.__file__).resolve()),
                "file_sha256": FRAME_V18A_FILE_SHA256_V19A,
            },
            "v18a_frame_certificate": {
                "path": str(frame_v18a.OUTPUT_PATH_V18A),
                "file_sha256": FRAME_CERTIFICATE_V18A_FILE_SHA256_V19A,
                "content_sha256": (
                    FRAME_CERTIFICATE_V18A_CONTENT_SHA256_V19A
                ),
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
            "v18a_joint_component_identity_root_sha256": bound_v18a[
                "joint_frame"
            ]["joint_component_identity_root_sha256"],
        },
        "disjoint_patch_tiers": {
            "tier_membership_reused_exactly_from_v18a": True,
            "v18a_prefix_seed": frame_v18a.PREFIX_SEED_V18A,
            "paired_tier_populations": {
                str(key): value
                for key, value in PAIRED_TIER_POPULATIONS_V19A.items()
            },
            "candidate_only_tier_populations": {
                str(key): value
                for key, value in CANDIDATE_ONLY_TIER_POPULATIONS_V19A.items()
            },
            "eligible_patch_population_per_tier": {
                str(key): value
                for key, value in PATCH_TIER_POPULATIONS_V19A.items()
            },
            "no_arm_activates_more_than_one_patch_tier": True,
        },
        "constrained_base_flow": {
            "seed": FLOW_SEED_V19A,
            "solver": base_flow["solver"],
            "panel_count": 10,
            "optimization_panel_count": 6,
            "train_only_screen_panel_count": 4,
            "base_category_quota_per_panel": {
                category: BASE_CATEGORY_QUOTA_V19A
                for category in BASE_CATEGORIES_V19A
            },
            "production_topic_quota_per_panel": PRODUCTION_TOPIC_QUOTAS_V19A,
            "selected_base_components": 240,
            "globally_panel_disjoint_base_components": True,
            "quota_seed_or_solver_relaxation_used": False,
            "infeasible_action": (
                "abort_v19a_without_quota_seed_grouping_or_solver_fallback"
            ),
        },
        "candidate_only_assignment": {
            "seed": CANDIDATE_ASSIGNMENT_SEED_V19A,
            "quota_per_active_tier_per_panel": 1,
            "tier_summary": candidate_assignment["summary"],
            "every_unique_item_before_deterministic_reuse": True,
            "total_panel_assignments": 30,
            "unique_candidate_only_components": 23,
            "reuse_assignments_after_exhaustion": 7,
            "ordered_assignment_sha256": canonical_sha256(
                candidate_assignment_sequence
            ),
        },
        "estimand": {
            "semantics": (
                "one_representative_per_joint_component_candidate_substitutes_"
                "only_its_disjoint_paired_tier_candidate_only_adds_its_tier_"
                "q1_and_all_other_base_components_remain_production"
            ),
            "arm_population_denominators": ARM_POPULATIONS_V19A,
            "arm_requests_per_panel": ARM_REQUESTS_PER_PANEL_V19A,
            "base_population_ht_strata": {
                category: {
                    "population": population,
                    "per_panel_quota": BASE_CATEGORY_QUOTA_V19A,
                    "horvitz_thompson_weight": (
                        population / BASE_CATEGORY_QUOTA_V19A
                    ),
                }
                for category, population in BASE_CATEGORY_POPULATIONS_V19A.items()
            },
            "candidate_only_ht_strata": {
                str(tier): {
                    "population": population,
                    "per_active_arm_panel_quota": 1,
                    "horvitz_thompson_weight": float(population),
                    "active_only_for_arm": f"patch_tier_{tier}_only",
                }
                for tier, population in (
                    CANDIDATE_ONLY_TIER_POPULATIONS_V19A.items()
                )
            },
            "arm_total": (
                "sum_HT_weight_times_selected_active_component_score_over_"
                "the_arm_specific_base_and_candidate_only_strata"
            ),
            "arm_mean": "arm_total_divided_by_exact_arm_population_denominator",
            "candidate_ht_never_targets_all_226": True,
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
    certificate["content_sha256_before_self_field"] = canonical_sha256(
        certificate
    )
    validate_certificate_v19a(certificate)
    return certificate


def validate_certificate_v19a(value: dict) -> dict:
    solver = value.get("constrained_base_flow", {}).get("solver", {})
    candidate = value.get("candidate_only_assignment", {})
    panels = value.get("panels", [])
    if (
        value.get("schema")
        != "eggroll-es-disjoint-tier-flow-certificate-v19a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("arm_order") != list(ARM_ORDER_V19A)
        or value.get("estimand", {}).get("arm_population_denominators")
        != ARM_POPULATIONS_V19A
        or value.get("estimand", {}).get("arm_requests_per_panel")
        != ARM_REQUESTS_PER_PANEL_V19A
        or solver.get("status") != 0
        or solver.get("success") is not True
        or not math.isclose(solver.get("mip_gap", math.inf), 0.0, abs_tol=1e-15)
        or solver.get("quota_relaxation_used") is not False
        or solver.get("fallback_solver_used") is not False
        or value.get("constrained_base_flow", {}).get(
            "quota_seed_or_solver_relaxation_used"
        ) is not False
        or candidate.get("every_unique_item_before_deterministic_reuse")
        is not True
        or candidate.get("unique_candidate_only_components") != 23
        or candidate.get("reuse_assignments_after_exhaustion") != 7
        or len(panels) != 10
        or tuple(panel.get("name") for panel in panels) != PANEL_NAMES_V19A
        or any(
            panel.get("base_category_counts")
            != {
                category: BASE_CATEGORY_QUOTA_V19A
                for category in BASE_CATEGORIES_V19A
            }
            or panel.get("production_topic_counts")
            != PRODUCTION_TOPIC_QUOTAS_V19A
            or set(panel.get("arms", {})) != set(ARM_ORDER_V19A)
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
        raise RuntimeError("v19a disjoint-tier flow certificate changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v19a flow certificate already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V19A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V19A:
        raise ValueError("v19a flow certificate output path changed")
    certificate = build_certificate_v19a()
    _exclusive_write(OUTPUT_PATH_V19A, certificate)
    result = {
        "schema": "eggroll-es-disjoint-tier-flow-build-v19a",
        "output": str(OUTPUT_PATH_V19A),
        "file_sha256": file_sha256(OUTPUT_PATH_V19A),
        "content_sha256": certificate["content_sha256_before_self_field"],
        "solver_status": certificate["constrained_base_flow"]["solver"]["status"],
        "solver_mip_gap": certificate["constrained_base_flow"]["solver"]["mip_gap"],
        "runtime_launch_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
