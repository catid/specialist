#!/usr/bin/env python3
"""Focused fail-closed tests for V59 fragile-priority train-only HPO."""

from __future__ import annotations

import math

import numpy as np

import lora_es_fragile_maximin_projection_v55b as v55b
import lora_es_fragile_priority_projection_v59 as v59
import run_lora_es_fragile_priority_v59 as runner59


def test_v59_direction_and_convex_kkt_certificate() -> None:
    projection = v59.fragile_priority_projection_v59()
    solution = projection["solution"]
    assert projection["direction_sha256"] == (
        "f80db1bde940053c93e559c9305dc750e82f5231d8df4f52fcf1b85c06dc0522"
    )
    assert solution["minimum_fragile_margin"] >= 0.47
    assert solution["minimum_other_margin"] >= 0.15 - 1e-12
    assert solution["maximum_stationarity_residual"] <= 1e-8
    assert solution["maximum_complementarity_residual"] <= 1e-8
    assert solution["minimum_kkt_multiplier"] >= -1e-12
    assert math.isclose(
        solution["fragile_multiplier_sum"], 1.0,
        rel_tol=0.0, abs_tol=1e-9,
    )


def test_v59_floor_is_preregistered_half_v55b_margin() -> None:
    assert v59.NONFRAGILE_MARGIN_FLOOR_V59 == 0.15
    assert v59.NONFRAGILE_MARGIN_FLOOR_V59 == (
        0.5 * v55b.MINIMUM_MAXIMIN_MARGIN_V55B
    )


def test_v59_multistart_agrees_and_actor_rows_are_partitioned() -> None:
    solution = v59.fragile_priority_projection_v59()["solution"]
    receipts = solution["start_receipts"]
    assert len(receipts) == 6
    assert all(item["success"] is True for item in receipts)
    assert max(item["fragile_minimum_margin"] for item in receipts) - min(
        item["fragile_minimum_margin"] for item in receipts
    ) <= 1e-10
    assert solution["fragile_row_indices"] == [4, 5, 6, 7]
    assert len(solution["other_row_indices"]) == 20


def test_v59_direction_is_unit_and_distinct_from_v58() -> None:
    projection = v59.fragile_priority_projection_v59()
    assert math.isclose(
        np.linalg.norm(projection["direction"]), 1.0,
        rel_tol=0.0, abs_tol=1e-12,
    )
    assert 0.8 < projection["v58_direction_cosine"] < 0.99


def test_v59_exact_matched_scale_grid_and_gate_contract() -> None:
    projection = v59.fragile_priority_projection_v59()
    plans = v59.scale_plans_v59(projection)
    assert [item["target_norm_ratio"] for item in plans] == [
        0.5, 0.375, 0.25, 0.1875, 0.125, 0.0625,
    ]
    assert len({item["coefficient_sha256"] for item in plans}) == 6
    assert all(
        item["original_nine_calibrated_endpoint_gates_unchanged"] is True
        for item in plans
    )


def test_v59_all_nonfragile_objectives_keep_floor() -> None:
    minima = v59.fragile_priority_projection_v59()["solution"][
        "per_objective_minimum_margins"
    ]
    assert minima["fragile_generation_f1"] >= 0.47
    assert all(
        margin >= 0.15 - 1e-12
        for objective, margin in minima.items()
        if objective != "fragile_generation_f1"
    )


def test_v59_scale_path_accepts_first_pass_prefix() -> None:
    rows = [
        {"target_norm_ratio": ratio}
        for ratio in [0.5, 0.375, 0.25]
    ]
    assert runner59._validated_scale_path_v59(rows, 0.25) == [
        0.5, 0.375, 0.25,
    ]


def test_v59_scale_path_requires_full_grid_for_all_fail() -> None:
    rows = [
        {"target_norm_ratio": ratio}
        for ratio in [0.5, 0.375, 0.25]
    ]
    try:
        runner59._validated_scale_path_v59(rows, None)
    except RuntimeError as error:
        assert str(error) == "v59 all-fail scale path ended early"
    else:
        raise AssertionError("v59 accepted a truncated all-fail scale path")
