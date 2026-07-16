#!/usr/bin/env python3
"""Fragile-F1-priority constrained actor projection for V59."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

import lora_es_actor_maximin_projection_v58 as v58
import lora_es_fragile_maximin_projection_v55b as v55b
import lora_es_nested_population_v52 as v52


SCALE_ORDER_V59 = v58.SCALE_ORDER_V58
NONFRAGILE_MARGIN_FLOOR_V59 = 0.5 * v55b.MINIMUM_MAXIMIN_MARGIN_V55B
MINIMUM_FRAGILE_MARGIN_V59 = 0.47
SOLVER_FTOL_V59 = 1e-14
SOLVER_MAXITER_V59 = 10000
CERTIFICATION_TOLERANCE_V59 = 1e-8
FRAGILE_OBJECTIVE_V59 = "fragile_generation_f1"


def _solve_v59(matrix: np.ndarray, row_order: list[dict], v58_direction: np.ndarray) -> dict:
    fragile_indices = [
        index for index, row in enumerate(row_order)
        if row["objective"] == FRAGILE_OBJECTIVE_V59
    ]
    other_indices = [
        index for index in range(len(row_order))
        if index not in fragile_indices
    ]
    if fragile_indices != [4, 5, 6, 7] or len(other_indices) != 20:
        raise RuntimeError("v59 objective row partition changed")
    fragile = matrix[fragile_indices]
    other = matrix[other_indices]

    def objective(values: np.ndarray) -> float:
        return -float(values[-1])

    def objective_jacobian(_values: np.ndarray) -> np.ndarray:
        return np.concatenate([np.zeros(16), [-1.0]])

    def fragile_constraints(values: np.ndarray) -> np.ndarray:
        return fragile @ values[:16] - values[-1]

    def fragile_jacobian(_values: np.ndarray) -> np.ndarray:
        return np.hstack([fragile, -np.ones((4, 1))])

    def other_constraints(values: np.ndarray) -> np.ndarray:
        return other @ values[:16] - NONFRAGILE_MARGIN_FLOOR_V59

    def other_jacobian(_values: np.ndarray) -> np.ndarray:
        return np.hstack([other, np.zeros((20, 1))])

    def norm_constraint(values: np.ndarray) -> float:
        return 1.0 - float(np.dot(values[:16], values[:16]))

    def norm_jacobian(values: np.ndarray) -> np.ndarray:
        return np.concatenate([-2.0 * values[:16], [0.0]])

    constraints = [
        {"type": "ineq", "fun": fragile_constraints, "jac": fragile_jacobian},
        {"type": "ineq", "fun": other_constraints, "jac": other_jacobian},
        {"type": "ineq", "fun": norm_constraint, "jac": norm_jacobian},
    ]
    starts = [v58_direction.copy(), fragile.mean(axis=0)]
    starts.extend(row.copy() for row in fragile)
    receipts = []
    solved_values = []
    for index, raw in enumerate(starts):
        unit = raw / np.linalg.norm(raw)
        initial = np.concatenate([unit, [float(np.min(fragile @ unit))]])
        solved = minimize(
            objective, initial, jac=objective_jacobian, method="SLSQP",
            constraints=constraints,
            options={
                "ftol": SOLVER_FTOL_V59,
                "maxiter": SOLVER_MAXITER_V59,
                "disp": False,
            },
        )
        direction = np.asarray(solved.x[:16], dtype=np.float64)
        fragile_margins = fragile @ direction
        other_margins = other @ direction
        receipt = {
            "start_index": index,
            "success": bool(solved.success),
            "status": int(solved.status),
            "iterations": int(solved.nit),
            "fragile_minimum_margin": float(np.min(fragile_margins)),
            "other_minimum_margin": float(np.min(other_margins)),
            "direction_norm": float(np.linalg.norm(direction)),
        }
        receipts.append(receipt)
        solved_values.append(solved)
    if (
        len(receipts) != 6
        or any(item["success"] is not True for item in receipts)
        or max(item["fragile_minimum_margin"] for item in receipts)
        - min(item["fragile_minimum_margin"] for item in receipts) > 1e-10
        or max(item["other_minimum_margin"] for item in receipts)
        - min(item["other_minimum_margin"] for item in receipts) > 1e-10
    ):
        raise RuntimeError("v59 deterministic multi-start agreement failed")
    # Select the V58-direction start deterministically; all six starts agree.
    solved = solved_values[0]
    direction = np.asarray(solved.x[:16], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    fragile_margins = fragile @ direction
    other_margins = other @ direction
    all_margins = matrix @ direction
    multipliers = np.asarray(solved.multipliers, dtype=np.float64)
    if multipliers.shape != (25,):
        raise RuntimeError("v59 SLSQP multiplier inventory changed")
    fragile_multipliers = multipliers[:4]
    other_multipliers = multipliers[4:24]
    norm_multiplier = float(multipliers[24])
    values = np.concatenate([
        direction, [float(np.min(fragile_margins))],
    ])
    fragile_values = fragile_constraints(values)
    other_values = other_constraints(values)
    norm_value = norm_constraint(values)
    stationarity = objective_jacobian(values) - (
        fragile_jacobian(values).T @ fragile_multipliers
        + other_jacobian(values).T @ other_multipliers
        + norm_jacobian(values) * norm_multiplier
    )
    complementarity = np.concatenate([
        fragile_multipliers * fragile_values,
        other_multipliers * other_values,
        [norm_multiplier * norm_value],
    ])
    maximum_stationarity = float(np.max(np.abs(stationarity)))
    maximum_complementarity = float(np.max(np.abs(complementarity)))
    minimum_fragile = float(np.min(fragile_margins))
    minimum_other = float(np.min(other_margins))
    objective_minima = {}
    for objective in sorted({row["objective"] for row in row_order}):
        indices = [
            index for index, row in enumerate(row_order)
            if row["objective"] == objective
        ]
        objective_minima[objective] = float(np.min(all_margins[indices]))
    if (
        not math.isclose(
            NONFRAGILE_MARGIN_FLOOR_V59, 0.15,
            rel_tol=0.0, abs_tol=1e-15,
        )
        or minimum_fragile < MINIMUM_FRAGILE_MARGIN_V59
        or minimum_other < NONFRAGILE_MARGIN_FLOOR_V59 - 1e-12
        or not math.isclose(
            float(np.linalg.norm(direction)), 1.0,
            rel_tol=0.0, abs_tol=1e-12,
        )
        or float(np.min(multipliers)) < -1e-12
        or abs(float(np.sum(fragile_multipliers)) - 1.0) > 1e-9
        or maximum_stationarity > 1e-8
        or maximum_complementarity > 1e-8
        or any(value < NONFRAGILE_MARGIN_FLOOR_V59 - 1e-12
               for name, value in objective_minima.items()
               if name != FRAGILE_OBJECTIVE_V59)
    ):
        raise RuntimeError("v59 convex KKT certificate failed")
    return {
        "solver": "scipy.optimize.minimize/SLSQP",
        "convex_program": (
            "maximize t subject to four fragile-F1 actor margins >= t, "
            "other 20 actor-objective margins >= 0.15, ||u||_2 <= 1"
        ),
        "global_optimality_certificate": (
            "convex feasible set plus primal feasibility, nonnegative KKT "
            "multipliers, stationarity, and complementarity"
        ),
        "ftol": SOLVER_FTOL_V59,
        "maxiter": SOLVER_MAXITER_V59,
        "start_order": "v58_direction, fragile_row_mean, fragile_actor_rows_0_to_3",
        "start_receipts": receipts,
        "selected_start_index": 0,
        "direction": direction.tolist(),
        "direction_norm": float(np.linalg.norm(direction)),
        "fragile_row_indices": fragile_indices,
        "other_row_indices": other_indices,
        "fragile_margins": fragile_margins.tolist(),
        "other_margins": other_margins.tolist(),
        "all_24_margins": all_margins.tolist(),
        "minimum_fragile_margin": minimum_fragile,
        "required_minimum_fragile_margin": MINIMUM_FRAGILE_MARGIN_V59,
        "minimum_other_margin": minimum_other,
        "required_other_margin_floor": NONFRAGILE_MARGIN_FLOOR_V59,
        "other_margin_floor_derivation": (
            "exactly one half of V55B MINIMUM_MAXIMIN_MARGIN_V55B=0.3"
        ),
        "per_objective_minimum_margins": objective_minima,
        "kkt_multipliers": multipliers.tolist(),
        "minimum_kkt_multiplier": float(np.min(multipliers)),
        "fragile_multiplier_sum": float(np.sum(fragile_multipliers)),
        "norm_multiplier": norm_multiplier,
        "maximum_stationarity_residual": maximum_stationarity,
        "maximum_complementarity_residual": maximum_complementarity,
    }


def fragile_priority_projection_v59() -> dict:
    prior = v58.actor_maximin_projection_v58()
    matrix = np.asarray(prior["normalized_objective_matrix"], dtype=np.float64)
    direction58 = np.asarray(prior["direction"], dtype=np.float64)
    solution = _solve_v59(matrix, prior["row_order"], direction58)
    direction = np.asarray(solution["direction"], dtype=np.float64)
    cosine = float(np.dot(direction, direction58))
    if cosine >= 0.99 or cosine <= 0.8:
        raise RuntimeError("v59 direction novelty changed")
    result = {
        "schema": "fragile-f1-priority-constrained-actor-projection-v59",
        "population_size": 16,
        "sigma": 0.0048,
        "seeds": list(v52.P16_SEEDS_V52),
        "source_v58_projection_content_sha256": prior["content_sha256"],
        "source_matrix_sha256": prior["normalized_objective_matrix_sha256"],
        "row_order": prior["row_order"],
        "solution": solution,
        "direction": solution["direction"],
        "direction_sha256": v52.canonical_sha256_v52(solution["direction"]),
        "maximin_margin": solution["minimum_fragile_margin"],
        "minimum_fragile_margin": solution["minimum_fragile_margin"],
        "minimum_other_margin": solution["minimum_other_margin"],
        "v58_direction_cosine": cosine,
        "genuinely_different_from_v58": cosine < 0.99,
        "reference_norm": float(prior["reference_norm"]),
        "source_signed_scores_file_sha256": prior[
            "source_signed_scores_file_sha256"
        ],
        "source_signed_scores_content_sha256": prior[
            "source_signed_scores_content_sha256"
        ],
    }
    result["content_sha256"] = v52.canonical_sha256_v52(result)
    return result


def scale_plans_v59(projection: dict) -> list[dict]:
    if projection != fragile_priority_projection_v59():
        raise RuntimeError("v59 projection input changed")
    direction = np.asarray(projection["direction"], dtype=np.float64)
    reference = float(projection["reference_norm"])
    result = []
    for ratio in SCALE_ORDER_V59:
        coefficients = (direction * reference * ratio).tolist()
        actual = float(np.linalg.norm(coefficients) / reference)
        if not math.isclose(actual, ratio, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("v59 scale norm changed")
        result.append({
            "target_norm_ratio": ratio,
            "coefficients": coefficients,
            "coefficient_sha256": v52.canonical_sha256_v52({
                "seeds": list(v52.P16_SEEDS_V52),
                "coefficients": coefficients,
            }),
            "actual_norm_ratio": actual,
            "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
            "fragile_priority_and_other_positive_margins_preserved": True,
            "original_nine_calibrated_endpoint_gates_unchanged": True,
        })
    return result
