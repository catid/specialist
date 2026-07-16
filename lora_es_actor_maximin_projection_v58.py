#!/usr/bin/env python3
"""Deterministic 24-row actor-aware maximin projection for V58."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import lora_es_fragile_maximin_projection_v55b as v55b
import lora_es_nested_population_v52 as v52


ROOT = Path(__file__).resolve().parent
V53_SELECTED_ARM = ROOT / "experiments/eggroll_es_hpo/runs/v53_lora_es_sigma_discrimination/sigma_0p0048_arm_v53.json"
V53_SELECTED_ARM_FILE_SHA256 = "f73adf79eb1cc10315a088568cd68816a5233fdb46217bb5ed55e3c0be778659"
V53_SELECTED_ARM_CONTENT_SHA256 = "f60eb4f5c66a3f26571c07111ec1ccf9c26ecd8121e3c160007c5ad7220ae486"
SCALE_ORDER_V58 = (0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625)
ROW_COUNT_V58 = 24
COLUMN_COUNT_V58 = 16
MINIMUM_MARGIN_V58 = 0.24
SOLVER_FTOL_V58 = 1e-13
SOLVER_MAXITER_V58 = 5000
CERTIFICATION_TOLERANCE_V58 = 1e-10


def _nested(value: dict, path: tuple[str, ...]) -> float:
    for key in path:
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("v58 objective is non-finite")
    return result


def _load_arm_v58() -> dict:
    if v52.file_sha256_v52(V53_SELECTED_ARM) != V53_SELECTED_ARM_FILE_SHA256:
        raise RuntimeError("v58 sealed V53 arm file changed")
    value = json.loads(V53_SELECTED_ARM.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != V53_SELECTED_ARM_CONTENT_SHA256
        or v52.canonical_sha256_v52(compact)
        != V53_SELECTED_ARM_CONTENT_SHA256
    ):
        raise RuntimeError("v58 sealed V53 arm content changed")
    return value


def _actor_objective_v58(
    arm: dict, path: tuple[str, ...], actor_rank: int,
) -> dict:
    sign_scores = {}
    for sign in ("plus", "minus"):
        rows = []
        for direction in range(COLUMN_COUNT_V58):
            score = _nested(
                arm["signed_scores"][sign][direction][actor_rank], path,
            )
            # Repeating one actor four times deliberately reuses the sealed
            # centered-rank implementation without changing its math.
            rows.append([score] * v52.ACTORS_V52)
        sign_scores[sign] = rows
    result = v52.objective_coefficients_v52(sign_scores)
    coefficients = np.asarray(result["coefficients"], dtype=np.float64)
    norm = float(np.linalg.norm(coefficients))
    if result["zero_spread"] is True or not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("v58 actor objective has zero/nonfinite spread")
    return {
        "schema": "actor-centered-rank-objective-v58",
        "actor_rank": actor_rank,
        "source": result,
        "coefficients": coefficients.tolist(),
        "coefficient_l2_norm": norm,
        "normalized_coefficients": (coefficients / norm).tolist(),
        "content_sha256": v52.canonical_sha256_v52({
            "actor_rank": actor_rank,
            "coefficients": coefficients.tolist(),
            "coefficient_l2_norm": norm,
            "normalized_coefficients": (coefficients / norm).tolist(),
        }),
    }


def _solve_maximin_v58(matrix: np.ndarray, old_direction: np.ndarray) -> dict:
    if matrix.shape != (ROW_COUNT_V58, COLUMN_COUNT_V58):
        raise RuntimeError("v58 normalized objective matrix shape changed")

    gram = matrix @ matrix.T

    def objective(weights: np.ndarray) -> float:
        return 0.5 * float(weights @ gram @ weights)

    def objective_jacobian(weights: np.ndarray) -> np.ndarray:
        return gram @ weights

    def simplex(weights: np.ndarray) -> float:
        return float(np.sum(weights) - 1.0)

    def simplex_jacobian(_weights: np.ndarray) -> np.ndarray:
        return np.ones(ROW_COUNT_V58, dtype=np.float64)

    starts = [np.ones(ROW_COUNT_V58, dtype=np.float64) / ROW_COUNT_V58]
    starts.extend(np.eye(ROW_COUNT_V58, dtype=np.float64)[index]
                  for index in range(ROW_COUNT_V58))
    receipts = []
    candidates = []
    for index, initial in enumerate(starts):
        solved = minimize(
            objective, initial, jac=objective_jacobian, method="SLSQP",
            bounds=[(0.0, None)] * ROW_COUNT_V58,
            constraints=[{
                "type": "eq", "fun": simplex, "jac": simplex_jacobian,
            }],
            options={
                "ftol": 1e-15,
                "maxiter": SOLVER_MAXITER_V58,
                "disp": False,
            },
        )
        weights = np.asarray(solved.x, dtype=np.float64)
        dual_vector = matrix.T @ weights
        dual_bound = float(np.linalg.norm(dual_vector))
        receipt = {
            "start_index": index,
            "success": bool(solved.success),
            "status": int(solved.status),
            "iterations": int(solved.nit),
            "dual_bound": dual_bound,
            "simplex_residual": abs(float(np.sum(weights) - 1.0)),
            "minimum_weight": float(np.min(weights)),
        }
        receipts.append(receipt)
        if solved.success:
            candidates.append((dual_bound, index, solved))
    if not candidates:
        raise RuntimeError("v58 dual SLSQP produced no successful candidate")
    _dual_bound, best_index, best = min(
        candidates, key=lambda item: (item[0], item[1]),
    )
    weights = np.asarray(best.x, dtype=np.float64)
    dual_vector = matrix.T @ weights
    dual_bound = float(np.linalg.norm(dual_vector))
    direction = dual_vector / dual_bound
    margins = matrix @ direction
    minimum = float(np.min(margins))
    norm = float(np.linalg.norm(direction))
    positive = [index for index, value in enumerate(weights) if value > 1e-9]
    inactive = [index for index in range(ROW_COUNT_V58) if index not in positive]
    simplex_residual = abs(float(np.sum(weights) - 1.0))
    primal_dual_gap = dual_bound - minimum
    complementarity = float(np.max(np.abs(
        weights * (margins - dual_bound)
    )))
    inactive_gap = float(
        np.min(margins[inactive]) - minimum if inactive else math.inf
    )
    if (
        len(receipts) != 25
        or sum(item["success"] for item in receipts) != 25
        or not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=CERTIFICATION_TOLERANCE_V58)
        or minimum <= MINIMUM_MARGIN_V58
        or float(np.min(weights)) < -1e-12
        or simplex_residual > 1e-12
        or primal_dual_gap < -1e-10 or primal_dual_gap > 1e-6
        or complementarity > 1e-7
        or len(positive) != 10
        or inactive_gap <= 1e-3
    ):
        raise RuntimeError("v58 convex-dual maximin certificate failed")
    return {
        "solver": "scipy.optimize.minimize/SLSQP",
        "objective": "minimize ||A.T@lambda|| over lambda>=0, sum(lambda)=1",
        "convex_dual_of": "max t subject to A@u >= t and ||u||_2 <= 1",
        "global_upper_bound_certificate": True,
        "ftol": 1e-15,
        "maxiter": SOLVER_MAXITER_V58,
        "deterministic_start_order": "uniform_simplex, simplex_vertices_0_through_23",
        "start_count": len(receipts),
        "successful_start_count": sum(item["success"] for item in receipts),
        "selected_start_index": best_index,
        "start_receipts": receipts,
        "dual_weights": weights.tolist(),
        "dual_weights_sha256": v52.canonical_sha256_v52(weights.tolist()),
        "dual_simplex_residual": simplex_residual,
        "minimum_dual_weight": float(np.min(weights)),
        "positive_dual_weight_indices_threshold_1e_9": positive,
        "positive_dual_weight_count": len(positive),
        "dual_upper_bound": dual_bound,
        "direction": direction.tolist(),
        "direction_norm": norm,
        "margins": margins.tolist(),
        "primal_lower_bound": minimum,
        "minimum_margin": minimum,
        "required_minimum_margin": MINIMUM_MARGIN_V58,
        "primal_dual_gap": primal_dual_gap,
        "required_maximum_primal_dual_gap": 1e-6,
        "maximum_complementarity_residual": complementarity,
        "required_maximum_complementarity_residual": 1e-7,
        "minimum_inactive_margin_gap": inactive_gap,
        "all_24_primal_margins_positive": bool(np.all(margins > 0.0)),
    }


def actor_maximin_projection_v58() -> dict:
    arm = _load_arm_v58()
    objective_names = list(v55b.OBJECTIVE_PATHS_V55B)
    if objective_names != sorted(objective_names) or len(objective_names) != 6:
        raise RuntimeError("v58 objective order changed")
    objectives = {}
    rows = []
    row_order = []
    for name in objective_names:
        actor_rows = []
        for actor_rank in range(v52.ACTORS_V52):
            objective = _actor_objective_v58(
                arm, v55b.OBJECTIVE_PATHS_V55B[name], actor_rank,
            )
            actor_rows.append(objective)
            rows.append(objective["normalized_coefficients"])
            row_order.append({"objective": name, "actor_rank": actor_rank})
        objectives[name] = actor_rows
    matrix = np.asarray(rows, dtype=np.float64)
    rank = int(np.linalg.matrix_rank(matrix))
    old = np.asarray(
        v55b.maximin_projection_v55b()["direction"], dtype=np.float64,
    )
    old_margins = matrix @ old
    if rank != COLUMN_COUNT_V58:
        raise RuntimeError("v58 actor objective matrix lost full column rank")
    solution = _solve_maximin_v58(matrix, old)
    direction = np.asarray(solution["direction"], dtype=np.float64)
    cosine = float(np.dot(direction, old))
    if cosine >= 0.99 or float(np.min(old_margins)) >= 0.0:
        raise RuntimeError("v58 direction is not information-gaining")
    result = {
        "schema": "actor-aware-24-row-maximin-projection-v58",
        "population_size": 16,
        "sigma": 0.0048,
        "seeds": list(v52.P16_SEEDS_V52),
        "objective_order": objective_names,
        "actor_order": list(range(v52.ACTORS_V52)),
        "row_order": row_order,
        "actor_objectives": objectives,
        "normalized_objective_matrix": matrix.tolist(),
        "normalized_objective_matrix_sha256": v52.canonical_sha256_v52(
            matrix.tolist()
        ),
        "matrix_shape": list(matrix.shape),
        "matrix_rank": rank,
        "solution": solution,
        "direction": solution["direction"],
        "direction_sha256": v52.canonical_sha256_v52(solution["direction"]),
        "minimum_actor_objective_margin": solution["minimum_margin"],
        "maximin_margin": solution["minimum_margin"],
        "all_24_actor_objective_margins_positive": all(
            value > 0.0 for value in solution["margins"]
        ),
        "v55b_direction_cosine": cosine,
        "v55b_actor_objective_margins": old_margins.tolist(),
        "v55b_minimum_actor_objective_margin": float(np.min(old_margins)),
        "genuinely_different_from_v55b": cosine < 0.99,
        "reference_norm": float(
            v55b.maximin_projection_v55b()["reference_norm"]
        ),
        "source_signed_scores_file_sha256": V53_SELECTED_ARM_FILE_SHA256,
        "source_signed_scores_content_sha256": V53_SELECTED_ARM_CONTENT_SHA256,
    }
    result["content_sha256"] = v52.canonical_sha256_v52(result)
    return result


def scale_plans_v58(projection: dict) -> list[dict]:
    if projection != actor_maximin_projection_v58():
        raise RuntimeError("v58 projection input changed")
    direction = np.asarray(projection["direction"], dtype=np.float64)
    reference = float(projection["reference_norm"])
    result = []
    for ratio in SCALE_ORDER_V58:
        coefficients = (direction * reference * ratio).tolist()
        actual = float(np.linalg.norm(coefficients) / reference)
        if not math.isclose(actual, ratio, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("v58 scale norm changed")
        result.append({
            "target_norm_ratio": ratio,
            "coefficients": coefficients,
            "coefficient_sha256": v52.canonical_sha256_v52({
                "seeds": list(v52.P16_SEEDS_V52),
                "coefficients": coefficients,
            }),
            "actual_norm_ratio": actual,
            "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
            "all_24_positive_margins_preserved_by_positive_scaling": True,
            "original_nine_calibrated_endpoint_gates_unchanged": True,
        })
    return result
