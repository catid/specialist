#!/usr/bin/env python3
"""Focused fail-closed tests for V58 actor-aware maximin HPO."""

from __future__ import annotations

import math

import numpy as np

import lora_es_actor_maximin_projection_v58 as v58


def test_v58_convex_dual_certificate() -> None:
    projection = v58.actor_maximin_projection_v58()
    solution = projection["solution"]
    assert projection["direction_sha256"] == (
        "710e824251342c7d4797f0a9ea3322a179614e8ac270c3a16d4b324461e75a0c"
    )
    assert projection["matrix_shape"] == [24, 16]
    assert projection["matrix_rank"] == 16
    assert solution["successful_start_count"] == 25
    assert solution["positive_dual_weight_count"] == 10
    assert solution["primal_lower_bound"] > 0.244
    assert 0.0 <= solution["primal_dual_gap"] <= 1e-6
    assert solution["maximum_complementarity_residual"] <= 1e-7
    assert solution["all_24_primal_margins_positive"] is True


def test_v58_direction_is_unit_and_improves_worst_actor_margin() -> None:
    projection = v58.actor_maximin_projection_v58()
    direction = np.asarray(projection["direction"], dtype=np.float64)
    assert math.isclose(np.linalg.norm(direction), 1.0, rel_tol=0, abs_tol=1e-10)
    assert projection["v55b_minimum_actor_objective_margin"] < 0.0
    assert projection["minimum_actor_objective_margin"] > 0.24
    assert projection["v55b_direction_cosine"] < 0.99


def test_v58_scale_grid_is_exact_and_descending() -> None:
    projection = v58.actor_maximin_projection_v58()
    plans = v58.scale_plans_v58(projection)
    assert [item["target_norm_ratio"] for item in plans] == [
        0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625,
    ]
    assert all(
        item["original_nine_calibrated_endpoint_gates_unchanged"] is True
        for item in plans
    )
    assert len({item["coefficient_sha256"] for item in plans}) == 6


def test_v58_row_order_covers_six_objectives_by_four_actors() -> None:
    projection = v58.actor_maximin_projection_v58()
    rows = projection["row_order"]
    assert len(rows) == 24
    assert {
        (row["objective"], row["actor_rank"]) for row in rows
    } == {
        (objective, actor)
        for objective in projection["objective_order"]
        for actor in range(4)
    }
