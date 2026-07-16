#!/usr/bin/env python3
"""Prospective high-rep exact-64 alpha-zero calibration contracts (V65B)."""

from __future__ import annotations

import hashlib
import math

import numpy as np

import lora_es_ranking64_alpha_zero_calibration_v65a as v65a


ROWS_V65B = 64
ACTORS_V65B = 4
WARMUP_PERIODS_V65B = 8
SCORED_PERIODS_V65B = 72
BLOCKS_V65B = 36
SUPERBLOCKS_V65B = 18
PAIRS_PER_ACTOR_V65B = BLOCKS_V65B
PAIRED_REPLICAS_PER_UNIT_V65B = ACTORS_V65B * BLOCKS_V65B
FORWARD_SUBSETS_V65B = (
    (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
)
TEMPORAL_PASS_BLOCKS_V65B = {
    "early_position": tuple(range(0, BLOCKS_V65B, 2)),
    "late_position": tuple(range(1, BLOCKS_V65B, 2)),
}
RUN_HALF_BLOCKS_V65B = {
    "first_run_half": tuple(range(0, BLOCKS_V65B // 2)),
    "second_run_half": tuple(range(BLOCKS_V65B // 2, BLOCKS_V65B)),
}
TEMPORAL_EPOCH_BLOCKS_V65B = {
    f"epoch_{index}": tuple(range(index * 6, (index + 1) * 6))
    for index in range(6)
}
COMMON_GENERATION_SEED_V65B = v65a.COMMON_GENERATION_SEED_V65A
GENERATION_PARAMS_WITHOUT_SEED_V65B = dict(
    v65a.GENERATION_PARAMS_WITHOUT_SEED_V65A
)
ENGINE_CONTROLS_V65B = dict(v65a.ENGINE_CONTROLS_V65A)
METRIC_ORDER_V65B = tuple(v65a.METRIC_ORDER_V65A)
COMPOSITE_WEIGHTS_V65B = dict(v65a.COMPOSITE_WEIGHTS_V65A)
BOOTSTRAP_REPLICATES_V65B = 65_536
BOOTSTRAP_SEED_V65B = 2_026_071_613
BOOTSTRAP_ALPHA_V65B = v65a.ONE_SIDED_ALPHA_V65A
BOOTSTRAP_INDEX_MATRIX_SHA256_V65B = (
    "f2b8a47138660b542e574431d143bad3c24ddeef0b2d510260c7bbaf71199966"
)
MAX_PRIMARY_CI_HALFWIDTH_V65B = v65a.MAX_PRIMARY_CI_HALFWIDTH_V65A
MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65B = (
    v65a.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65A
)
R1_PLANNING_OBSERVATIONS_V65B = {
    "generated_f1_halfwidth": 0.0018032639829489152,
    "generated_f1_limit": 0.000773822590292528,
    "pair_0_generated_f1_halfwidth": 0.00285893530557547,
    "pair_0_joint_composite_halfwidth": 0.0029545548673513768,
    "pair_0_joint_composite_null_radius": 0.005444770775652155,
    "pair_1_joint_composite_null_radius": 0.0016462423942832222,
    "paired_replicas_per_unit": 8,
    "maximum_actor_leave_one_out_shift": 0.0012119648781783704,
    "naive_iid_minimum_paired_replicas": 43.44359896859038,
    "prospective_random_effects_halfwidth_at_144": 0.0006163854036316198,
    "prospective_pair_0_generated_f1_halfwidth_at_72": (
        0.0006738575138486831
    ),
    "prospective_pair_0_joint_halfwidth_at_72": 0.000696395260697293,
}
FUTURE_V65_NULL_BOUND_TRANSFER_V65B = {
    "B_F": "primary_cluster_bootstrap.intervals.generated_f1_delta.null_radius",
    "B_C": "primary_cluster_bootstrap.intervals.joint_composite.null_radius",
    "B_S": "primary_cluster_bootstrap.intervals.stability_improvement.null_radius",
    "B_C_pass": "primary_cluster_bootstrap.B_C_pass",
}


def _forward_actors_v65b(block_index: int) -> frozenset[int]:
    if type(block_index) is not int or block_index not in range(BLOCKS_V65B):
        raise ValueError("v65b block index changed")
    superblock = block_index // 2
    base = frozenset(FORWARD_SUBSETS_V65B[superblock % 6])
    return base if block_index % 2 == 0 else frozenset(range(4)) - base


def label_v65b(actor_rank: int, period_index: int) -> str:
    if (
        type(actor_rank) is not int or actor_rank not in range(ACTORS_V65B)
        or type(period_index) is not int
        or period_index not in range(SCORED_PERIODS_V65B)
    ):
        raise ValueError("v65b label coordinate changed")
    block = period_index // 2
    forward = actor_rank in _forward_actors_v65b(block)
    first = period_index % 2 == 0
    return "candidate" if forward == first else "reference"


LABEL_PLAN_V65B = {
    str(actor): [label_v65b(actor, period) for period in range(72)]
    for actor in range(4)
}


def validate_schedule_v65b() -> dict:
    if (
        set(LABEL_PLAN_V65B) != {"0", "1", "2", "3"}
        or any(
            not isinstance(LABEL_PLAN_V65B[str(actor)], list)
            or len(LABEL_PLAN_V65B[str(actor)]) != SCORED_PERIODS_V65B
            or LABEL_PLAN_V65B[str(actor)] != [
                label_v65b(actor, period)
                for period in range(SCORED_PERIODS_V65B)
            ]
            for actor in range(ACTORS_V65B)
        )
    ):
        raise RuntimeError("v65b sealed label plan changed")
    for period in range(SCORED_PERIODS_V65B):
        labels = [LABEL_PLAN_V65B[str(actor)][period] for actor in range(4)]
        if labels.count("candidate") != 2 or labels.count("reference") != 2:
            raise RuntimeError("v65b simultaneous label balance changed")
    actor_orders = {}
    observed_early_subsets = []
    for superblock in range(SUPERBLOCKS_V65B):
        observed_early_subsets.append(tuple(sorted(
            _forward_actors_v65b(2 * superblock)
        )))
    if any(
        observed_early_subsets.count(subset) != 3
        for subset in FORWARD_SUBSETS_V65B
    ):
        raise RuntimeError("v65b six-subset crossover frequency changed")
    observed_all_subsets = [
        tuple(sorted(_forward_actors_v65b(block))) for block in range(36)
    ]
    observed_late_subsets = [
        tuple(sorted(_forward_actors_v65b(2 * superblock + 1)))
        for superblock in range(SUPERBLOCKS_V65B)
    ]
    if any(
        observed_all_subsets.count(subset) != 6
        for subset in FORWARD_SUBSETS_V65B
    ) or any(
        observed_late_subsets.count(subset) != 3
        for subset in FORWARD_SUBSETS_V65B
    ):
        raise RuntimeError("v65b full forward-subset frequency changed")
    for blocks in RUN_HALF_BLOCKS_V65B.values():
        selected = [observed_all_subsets[block] for block in blocks]
        if any(selected.count(subset) != 3 for subset in FORWARD_SUBSETS_V65B):
            raise RuntimeError("v65b run-half subset balance changed")
    for blocks in TEMPORAL_EPOCH_BLOCKS_V65B.values():
        selected = [observed_all_subsets[block] for block in blocks]
        if any(selected.count(subset) != 1 for subset in FORWARD_SUBSETS_V65B):
            raise RuntimeError("v65b temporal-epoch subset balance changed")
    for actor in range(4):
        orders = []
        for block in range(BLOCKS_V65B):
            pair = tuple(
                LABEL_PLAN_V65B[str(actor)][2 * block:2 * block + 2]
            )
            if pair not in (("candidate", "reference"),
                            ("reference", "candidate")):
                raise RuntimeError("v65b adjacent pair changed")
            orders.append("candidate_first" if pair[0] == "candidate"
                          else "reference_first")
        if orders.count("candidate_first") != 18:
            raise RuntimeError("v65b actor order balance changed")
        for blocks in RUN_HALF_BLOCKS_V65B.values():
            selected = [orders[block] for block in blocks]
            if (
                selected.count("candidate_first") != 9
                or selected.count("reference_first") != 9
            ):
                raise RuntimeError("v65b actor run-half order balance changed")
        for blocks in TEMPORAL_EPOCH_BLOCKS_V65B.values():
            selected = [orders[block] for block in blocks]
            if (
                selected.count("candidate_first") != 3
                or selected.count("reference_first") != 3
            ):
                raise RuntimeError("v65b actor epoch order balance changed")
        for position, selected in (
            ("early", orders[0::2]), ("late", orders[1::2]),
        ):
            if (
                selected.count("candidate_first") != 9
                or selected.count("reference_first") != 9
            ):
                raise RuntimeError(
                    f"v65b actor {position}-position balance changed"
                )
        actor_orders[str(actor)] = orders
    return {
        "schema": "v65b-fixed-six-subset-crossover-schedule",
        "scored_periods": 72,
        "adjacent_blocks": 36,
        "four_period_superblocks": 18,
        "every_period_two_candidate_two_reference": True,
        "each_actor_candidate_first_blocks": 18,
        "each_actor_reference_first_blocks": 18,
        "six_forward_actor_subsets_each_used_three_times_per_position": True,
        "six_forward_actor_subsets_each_used_six_times_overall": True,
        "six_forward_actor_subsets_each_used_three_times_per_run_half": True,
        "six_forward_actor_subsets_each_used_once_per_temporal_epoch": True,
        "each_actor_candidate_and_reference_first_nine_times_per_position": True,
        "each_actor_candidate_and_reference_first_nine_times_per_run_half": True,
        "each_actor_candidate_and_reference_first_three_times_per_temporal_epoch": True,
        "actor_orders": actor_orders,
    }


def frozen_bootstrap_indices_v65b() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED_V65B))
    result = rng.integers(
        0, ROWS_V65B,
        size=(BOOTSTRAP_REPLICATES_V65B, ROWS_V65B),
        dtype=np.int64,
    )
    if hashlib.sha256(
        result.astype("<i8", copy=False).tobytes(order="C")
    ).hexdigest() != BOOTSTRAP_INDEX_MATRIX_SHA256_V65B:
        raise RuntimeError("v65b frozen bootstrap matrix changed")
    return result


def validate_scored_periods_v65b(scored_periods: object) -> np.ndarray:
    if not isinstance(scored_periods, list) or len(scored_periods) != 72:
        raise ValueError("v65b requires exactly 72 scored periods")
    tensor = np.empty((64, 72, 4, 3), dtype=np.float64)
    identities = None
    for period, actors in enumerate(scored_periods):
        if not isinstance(actors, list) or len(actors) != 4:
            raise ValueError("v65b actor coverage changed")
        for actor, metrics in enumerate(actors):
            if not isinstance(metrics, list) or len(metrics) != 64:
                raise ValueError("v65b unit coverage changed")
            current = []
            for unit, metric in enumerate(metrics):
                if (
                    not isinstance(metric, dict)
                    or set(metric) != {
                        "request_index", "row_sha256", "unit_identity_sha256",
                        "f1", "exact", "nonzero",
                    }
                    or type(metric["request_index"]) is not int
                    or metric["request_index"] != unit
                    or not isinstance(metric["row_sha256"], str)
                    or len(metric["row_sha256"]) != 64
                    or any(char not in "0123456789abcdef"
                           for char in metric["row_sha256"])
                    or not isinstance(metric["unit_identity_sha256"], str)
                    or len(metric["unit_identity_sha256"]) != 64
                    or any(char not in "0123456789abcdef"
                           for char in metric["unit_identity_sha256"])
                ):
                    raise ValueError("v65b metric identity changed")
                f1, exact, nonzero = v65a._metric_v65a(metric)
                tensor[unit, period, actor] = (f1, exact, nonzero)
                current.append((metric["row_sha256"],
                                metric["unit_identity_sha256"]))
            if identities is None:
                identities = current
            elif current != identities:
                raise ValueError("v65b metric identity order changed")
    if (
        identities is None
        or len({value[0] for value in identities}) != ROWS_V65B
        or len({value[1] for value in identities}) != ROWS_V65B
        or len(set(identities)) != ROWS_V65B
    ):
        raise ValueError("v65b metric identities are not 64 unique units")
    if not np.isfinite(tensor).all():
        raise ValueError("v65b non-finite metric tensor")
    return tensor


def paired_state_metrics_v65b(
    scored: object,
) -> tuple[np.ndarray, np.ndarray]:
    array = (validate_scored_periods_v65b(scored)
             if not isinstance(scored, np.ndarray)
             else np.asarray(scored, dtype=np.float64))
    if array.shape != (64, 72, 4, 3) or not np.isfinite(array).all():
        raise ValueError("v65b scored metric tensor changed")
    reference = np.empty((64, 4, 36, 3), dtype=np.float64)
    candidate = np.empty_like(reference)
    for actor in range(4):
        for block in range(36):
            left, right = 2 * block, 2 * block + 1
            labels = LABEL_PLAN_V65B[str(actor)][left:right + 1]
            candidate_period = left if labels[0] == "candidate" else right
            reference_period = right if candidate_period == left else left
            candidate[:, actor, block] = array[:, candidate_period, actor]
            reference[:, actor, block] = array[:, reference_period, actor]
    return reference, candidate


def paired_replicas_v65b(scored: object) -> np.ndarray:
    reference, candidate = paired_state_metrics_v65b(scored)
    return candidate - reference


def _instability_v65b(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape[0] != 64 or array.shape[-1] != 3:
        raise ValueError("v65b instability tensor changed")
    array = array.reshape(64, -1, 3)
    replicas = array.shape[1]
    if replicas not in (24, 72, 144):
        raise ValueError("v65b instability replica count changed")
    f1 = array[..., 0]
    # Mean absolute pairwise difference without materializing O(R^2) arrays.
    ordered = np.sort(f1, axis=1)
    coefficients = 2 * np.arange(replicas) - replicas + 1
    mean_pairwise = (
        2.0 * np.sum(ordered * coefficients, axis=1)
        / (replicas * (replicas - 1))
    )
    exact_disagreement = (array[..., 1].max(1) != array[..., 1].min(1))
    nonzero_disagreement = (array[..., 2].max(1) != array[..., 2].min(1))
    weights = v65a.population65.design61.STABILITY_WEIGHTS_V61
    return (
        weights["mean_pairwise_absolute_f1_delta"] * mean_pairwise
        + weights["exact_label_disagreement"] * exact_disagreement
        + weights["nonzero_label_disagreement"] * nonzero_disagreement
    )


def _interval_v65b(values: np.ndarray, indices: np.ndarray) -> dict:
    unit_values = np.asarray(values, dtype=np.float64)
    if unit_values.shape != (ROWS_V65B,) or not np.isfinite(unit_values).all():
        raise ValueError("v65b bootstrap unit vector changed")
    samples = np.mean(unit_values[indices], axis=1)
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    lower_index = int(math.floor(
        BOOTSTRAP_ALPHA_V65B * (len(ordered) - 1)
    ))
    upper_index = len(ordered) - 1 - lower_index
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    return {
        "point": float(np.mean(unit_values)),
        "lcb": lower,
        "ucb": upper,
        "halfwidth": 0.5 * (upper - lower),
        "contains_zero": lower <= 0.0 <= upper,
        "null_radius": max(abs(lower), abs(upper)),
    }


def _component_units_v65b(
    reference: np.ndarray, candidate: np.ndarray,
) -> dict[str, np.ndarray]:
    paired = candidate - reference
    axes = tuple(range(1, paired.ndim - 1))
    metric = paired.mean(axis=axes)
    stability = _instability_v65b(reference) - _instability_v65b(candidate)
    composite = (
        COMPOSITE_WEIGHTS_V65B["f1_delta"] * metric[:, 0]
        + COMPOSITE_WEIGHTS_V65B["nonzero_delta"] * metric[:, 2]
        + COMPOSITE_WEIGHTS_V65B["stability_improvement"] * stability
    )
    return {
        "generated_f1_delta": metric[:, 0],
        "generated_nonzero_delta": metric[:, 2],
        "stability_improvement": stability,
        "joint_composite": composite,
        "generated_exact_delta_diagnostic": metric[:, 1],
    }


def _group_intervals_v65b(
    reference: np.ndarray, candidate: np.ndarray,
    groups: dict[str, tuple[int, ...]], indices: np.ndarray,
) -> dict:
    result = {}
    for name, blocks in groups.items():
        components = _component_units_v65b(
            reference[:, :, blocks, :], candidate[:, :, blocks, :],
        )
        result[name] = {
            metric: _interval_v65b(values, indices)
            for metric, values in components.items()
            if metric in ("generated_f1_delta", "joint_composite",
                          "stability_improvement")
        }
    return result


def cluster_bootstrap_v65b(
    reference_replicas: object, candidate_replicas: object,
    *, bootstrap_indices: np.ndarray | None = None,
) -> dict:
    reference = np.asarray(reference_replicas, dtype=np.float64)
    candidate = np.asarray(candidate_replicas, dtype=np.float64)
    if (
        reference.shape != (64, 4, 36, 3)
        or candidate.shape != reference.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("v65b paired-state tensor changed")
    if bootstrap_indices is None:
        indices = frozen_bootstrap_indices_v65b()
    else:
        supplied = np.asarray(bootstrap_indices)
        if supplied.dtype.kind not in "iu":
            raise ValueError("v65b bootstrap indices must be integer typed")
        indices = supplied.astype(np.int64, copy=False)
    if (
        indices.shape != (BOOTSTRAP_REPLICATES_V65B, 64)
        or (indices < 0).any() or (indices >= 64).any()
        or hashlib.sha256(indices.astype("<i8", copy=False).tobytes(
            order="C"
        )).hexdigest() != BOOTSTRAP_INDEX_MATRIX_SHA256_V65B
    ):
        raise ValueError("v65b bootstrap matrix changed")
    components = _component_units_v65b(reference, candidate)
    intervals = {
        metric: _interval_v65b(values, indices)
        for metric, values in components.items()
    }
    passes = _group_intervals_v65b(
        reference, candidate, TEMPORAL_PASS_BLOCKS_V65B, indices,
    )
    epochs = _group_intervals_v65b(
        reference, candidate, TEMPORAL_EPOCH_BLOCKS_V65B, indices,
    )
    halves = _group_intervals_v65b(
        reference, candidate, RUN_HALF_BLOCKS_V65B, indices,
    )
    paired = candidate - reference
    forward = []
    reverse = []
    for actor in range(4):
        for block in range(36):
            target = forward if actor in _forward_actors_v65b(block) else reverse
            target.append(paired[:, actor, block, 0])
    orientation_units = np.mean(np.stack(forward, axis=1), axis=1) - np.mean(
        np.stack(reverse, axis=1), axis=1,
    )
    pass_difference_units = (
        _component_units_v65b(
            reference[:, :, TEMPORAL_PASS_BLOCKS_V65B["early_position"], :],
            candidate[:, :, TEMPORAL_PASS_BLOCKS_V65B["early_position"], :],
        )["joint_composite"]
        - _component_units_v65b(
            reference[:, :, TEMPORAL_PASS_BLOCKS_V65B["late_position"], :],
            candidate[:, :, TEMPORAL_PASS_BLOCKS_V65B["late_position"], :],
        )["joint_composite"]
    )
    b_c_pass = max(
        value["joint_composite"]["null_radius"] for value in passes.values()
    )
    full_f1 = float(paired[..., 0].mean())
    superblock_loo = []
    for omitted in range(SUPERBLOCKS_V65B):
        omitted_blocks = {2 * omitted, 2 * omitted + 1}
        keep = [block for block in range(36) if block not in omitted_blocks]
        superblock_loo.append(float(paired[:, :, keep, 0].mean()))
    return {
        "schema": "v65b-ranking64-high-rep-cluster-bootstrap",
        "units": 64,
        "actors": 4,
        "blocks_per_actor": 36,
        "paired_replicas_per_unit_preserved_and_averaged": 144,
        "resampled_axis": "conflict_unit_only",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES_V65B,
        "bootstrap_seed": BOOTSTRAP_SEED_V65B,
        "bootstrap_index_matrix_sha256": BOOTSTRAP_INDEX_MATRIX_SHA256_V65B,
        "one_sided_alpha": BOOTSTRAP_ALPHA_V65B,
        "joint_composite_weights": dict(COMPOSITE_WEIGHTS_V65B),
        "intervals": intervals,
        "temporal_pass_intervals": passes,
        "six_epoch_intervals": epochs,
        "six_epoch_intervals_are_sealed_non_gating_diagnostics": True,
        "run_half_intervals": halves,
        "orientation_effect_interval": _interval_v65b(
            orientation_units, indices,
        ),
        "early_minus_late_joint_interval": _interval_v65b(
            pass_difference_units, indices,
        ),
        "superblock_influence": {
            "schema": "v65b-four-period-superblock-influence",
            "full_f1_point": full_f1,
            "leave_one_superblock_out_f1_points": superblock_loo,
            "maximum_absolute_leave_one_superblock_out_shift": float(max(
                abs(value - full_f1) for value in superblock_loo
            )),
            "superblocks_are_not_treated_as_extra_conflict_units": True,
            "sealed_non_gating_diagnostic": True,
        },
        "B_C_pass": float(b_c_pass),
        "B_C_pass_definition": (
            "max(early_position.joint_composite.null_radius,"
            "late_position.joint_composite.null_radius)"
        ),
        "joint_distribution_bootstrapped_before_quantiles": True,
        "exact_is_numeric_diagnostic_not_gate": True,
    }


def actor_influence_v65b(paired_replicas: object) -> dict:
    paired = np.asarray(paired_replicas, dtype=np.float64)
    if paired.shape != (64, 4, 36, 3) or not np.isfinite(paired).all():
        raise ValueError("v65b actor tensor changed")
    full = float(paired[..., 0].mean())
    values = [float(paired[:, [a for a in range(4) if a != omitted], :, 0].mean())
              for omitted in range(4)]
    return {
        "schema": "v65b-ranking64-actor-influence",
        "full_four_actor_f1_point": full,
        "leave_one_actor_out_f1_points": values,
        "maximum_absolute_leave_one_actor_out_shift": float(
            max(abs(value - full) for value in values)
        ),
    }


GATE_CHECK_KEYS_V65B = frozenset({
    "generated_f1_primary_interval_contains_zero",
    "joint_composite_interval_contains_zero",
    "stability_improvement_interval_contains_zero",
    "generated_f1_primary_ci_halfwidth_within_fixed_limit",
    "actor_leave_one_out_shift_within_fixed_limit",
    "both_temporal_pass_f1_intervals_contain_zero",
    "both_temporal_pass_joint_intervals_contain_zero",
    "both_temporal_pass_f1_halfwidths_within_fixed_limit",
    "orientation_effect_interval_contains_zero",
    "early_minus_late_joint_interval_contains_zero",
    "both_run_half_f1_joint_and_stability_intervals_contain_zero",
    "both_run_half_f1_halfwidths_within_fixed_limit",
})


def _valid_interval_v65b(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "point", "lcb", "ucb", "halfwidth", "contains_zero", "null_radius",
    }:
        return False
    numeric = [
        value.get(key) for key in ("point", "lcb", "ucb", "halfwidth",
                                   "null_radius")
    ]
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float))
        or not math.isfinite(float(item)) for item in numeric
    ):
        return False
    point, lower, upper, halfwidth, radius = map(float, numeric)
    return (
        lower <= upper
        and halfwidth >= 0.0
        and radius >= 0.0
        and halfwidth == 0.5 * (upper - lower)
        and radius == max(abs(lower), abs(upper))
        and type(value.get("contains_zero")) is bool
        and value["contains_zero"] is (lower <= 0.0 <= upper)
        and math.isfinite(point)
    )


def _validate_gate_inputs_v65b(primary: object, actor: object) -> None:
    passes = (
        primary.get("temporal_pass_intervals")
        if isinstance(primary, dict) else None
    )
    halves = (
        primary.get("run_half_intervals")
        if isinstance(primary, dict) else None
    )
    if (
        not isinstance(primary, dict)
        or primary.get("schema")
        != "v65b-ranking64-high-rep-cluster-bootstrap"
        or not isinstance(actor, dict)
        or actor.get("schema") != "v65b-ranking64-actor-influence"
        or not isinstance(passes, dict)
        or not isinstance(halves, dict)
        or set(passes)
        != set(TEMPORAL_PASS_BLOCKS_V65B)
        or set(halves)
        != set(RUN_HALF_BLOCKS_V65B)
    ):
        raise ValueError("v65b gate input schema changed")
    intervals = primary.get("intervals", {})
    required_metrics = {
        "generated_f1_delta", "joint_composite", "stability_improvement",
    }
    if (
        not isinstance(intervals, dict)
        or not required_metrics.issubset(intervals)
        or any(not _valid_interval_v65b(intervals.get(metric))
               for metric in required_metrics)
        or not _valid_interval_v65b(primary.get("orientation_effect_interval"))
        or not _valid_interval_v65b(
            primary.get("early_minus_late_joint_interval")
        )
    ):
        raise ValueError("v65b primary gate interval changed")
    for groups in (
        primary["temporal_pass_intervals"], primary["run_half_intervals"],
    ):
        if any(
            not isinstance(value, dict) or set(value) != required_metrics
            or any(not _valid_interval_v65b(value.get(metric))
                   for metric in required_metrics)
            for value in groups.values()
        ):
            raise ValueError("v65b grouped gate interval changed")
    influence = actor.get("maximum_absolute_leave_one_actor_out_shift")
    if (
        isinstance(influence, bool) or not isinstance(influence, (int, float))
        or not math.isfinite(float(influence)) or float(influence) < 0.0
    ):
        raise ValueError("v65b actor influence gate changed")


def gate_v65b(primary: dict, actor: dict) -> dict:
    _validate_gate_inputs_v65b(primary, actor)
    intervals = primary["intervals"]
    passes = primary["temporal_pass_intervals"]
    halves = primary["run_half_intervals"]
    checks = {
        "generated_f1_primary_interval_contains_zero":
            intervals["generated_f1_delta"]["contains_zero"] is True,
        "joint_composite_interval_contains_zero":
            intervals["joint_composite"]["contains_zero"] is True,
        "stability_improvement_interval_contains_zero":
            intervals["stability_improvement"]["contains_zero"] is True,
        "generated_f1_primary_ci_halfwidth_within_fixed_limit":
            intervals["generated_f1_delta"]["halfwidth"]
            <= MAX_PRIMARY_CI_HALFWIDTH_V65B,
        "actor_leave_one_out_shift_within_fixed_limit":
            actor["maximum_absolute_leave_one_actor_out_shift"]
            <= MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65B,
        "both_temporal_pass_f1_intervals_contain_zero": all(
            value["generated_f1_delta"]["contains_zero"] is True
            for value in passes.values()
        ),
        "both_temporal_pass_joint_intervals_contain_zero": all(
            value["joint_composite"]["contains_zero"] is True
            for value in passes.values()
        ),
        "both_temporal_pass_f1_halfwidths_within_fixed_limit": all(
            value["generated_f1_delta"]["halfwidth"]
            <= MAX_PRIMARY_CI_HALFWIDTH_V65B for value in passes.values()
        ),
        "orientation_effect_interval_contains_zero":
            primary["orientation_effect_interval"]["contains_zero"] is True,
        "early_minus_late_joint_interval_contains_zero":
            primary["early_minus_late_joint_interval"]["contains_zero"] is True,
        "both_run_half_f1_joint_and_stability_intervals_contain_zero": all(
            value[metric]["contains_zero"] is True
            for value in halves.values()
            for metric in (
                "generated_f1_delta", "joint_composite",
                "stability_improvement",
            )
        ),
        "both_run_half_f1_halfwidths_within_fixed_limit": all(
            value["generated_f1_delta"]["halfwidth"]
            <= MAX_PRIMARY_CI_HALFWIDTH_V65B for value in halves.values()
        ),
    }
    return {
        "schema": "v65b-ranking64-high-rep-alpha-zero-gate",
        "maximum_primary_ci_halfwidth": MAX_PRIMARY_CI_HALFWIDTH_V65B,
        "maximum_actor_leave_one_out_shift":
            MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65B,
        "checks": checks,
        "passed": all(checks.values()),
        "thresholds_relaxed_or_reinterpreted_after_r1": False,
        "exact_or_sentinel_gate_applied": False,
        "success_directly_authorizes_v65_population": False,
    }


def future_v65_null_bounds_v65b(
    primary: dict, gate: dict, actor: dict,
) -> dict:
    checks = gate.get("checks", {}) if isinstance(gate, dict) else {}
    if (
        not isinstance(primary, dict)
        or primary.get("schema")
        != "v65b-ranking64-high-rep-cluster-bootstrap"
        or not isinstance(primary.get("intervals"), dict)
        or any(
            not _valid_interval_v65b(
                primary["intervals"].get(metric)
            ) for metric in (
                "generated_f1_delta", "joint_composite",
                "stability_improvement",
            )
        )
        or not isinstance(gate, dict)
        or gate.get("schema") != "v65b-ranking64-high-rep-alpha-zero-gate"
        or set(checks) != GATE_CHECK_KEYS_V65B
        or any(type(value) is not bool for value in checks.values())
        or type(gate.get("passed")) is not bool
        or gate.get("passed") is not all(checks.values())
        or gate.get("thresholds_relaxed_or_reinterpreted_after_r1") is not False
        or gate.get("maximum_primary_ci_halfwidth")
        != MAX_PRIMARY_CI_HALFWIDTH_V65B
        or gate.get("maximum_actor_leave_one_out_shift")
        != MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65B
        or gate.get("exact_or_sentinel_gate_applied") is not False
        or gate.get("success_directly_authorizes_v65_population") is not False
    ):
        raise ValueError("v65b future gate or primary schema changed")
    linked_gate = gate_v65b(primary, actor)
    if gate != linked_gate:
        raise ValueError(
            "v65b future gate is not linked to primary and actor observations"
        )
    intervals = primary["intervals"]
    bounds = {
        "B_F": intervals["generated_f1_delta"]["null_radius"],
        "B_C": intervals["joint_composite"]["null_radius"],
        "B_S": intervals["stability_improvement"]["null_radius"],
        "B_C_pass": primary.get("B_C_pass"),
    }
    expected_b_c_pass = max(
        primary["temporal_pass_intervals"][position]["joint_composite"][
            "null_radius"
        ]
        for position in TEMPORAL_PASS_BLOCKS_V65B
    )
    if (
        any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0.0
            for value in bounds.values()
        )
        or float(bounds["B_C_pass"]) != float(expected_b_c_pass)
    ):
        raise ValueError("v65b future bound observation changed")
    return {
        "schema": "v65b-to-future-v65-null-bound-observation",
        "outcome_independent_field_mapping": dict(
            FUTURE_V65_NULL_BOUND_TRANSFER_V65B
        ),
        "bounds": {key: float(value) for key, value in bounds.items()},
        "required_future_v65_spread_gates": {
            "pooled_joint_composite": "spread_strictly_greater_than_2*B_C",
            "each_temporal_pass_joint_composite":
                "spread_strictly_greater_than_2*B_C_pass",
            "generated_f1_when_used": "spread_strictly_greater_than_2*B_F",
            "stability_when_used": "spread_strictly_greater_than_2*B_S",
            "stability_coefficient_when_gate_not_met": 0.0,
        },
        "eligible_for_future_separate_preregistration": gate["passed"] is True,
        "v65_population_launch_authorized": False,
    }


def analyze_scored_periods_v65b(scored_periods: object) -> dict:
    validate_schedule_v65b()
    scored = validate_scored_periods_v65b(scored_periods)
    reference, candidate = paired_state_metrics_v65b(scored)
    primary = cluster_bootstrap_v65b(reference, candidate)
    actor = actor_influence_v65b(candidate - reference)
    gate = gate_v65b(primary, actor)
    return {
        "schema": "v65b-ranking64-high-rep-alpha-zero-analysis",
        "status": "complete_numeric_only_high_rep_calibration",
        "rows": 64,
        "actors": 4,
        "warmup_periods_excluded_from_every_metric": 8,
        "scored_periods": 72,
        "adjacent_blocks": 36,
        "paired_replicas_per_unit": 144,
        "schedule": validate_schedule_v65b(),
        "primary_cluster_bootstrap": primary,
        "actor_influence": actor,
        "future_v65_null_bound_observation": future_v65_null_bounds_v65b(
            primary, gate, actor,
        ),
        "required_alpha_zero_gate": gate,
        "r1_used_for_prospective_sample_size_planning_only": True,
        "r1_threshold_relaxation_or_outcome_reinterpretation": False,
        "v65_population_launch_authorized": False,
    }
