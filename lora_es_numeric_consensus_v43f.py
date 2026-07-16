#!/usr/bin/env python3
"""Deterministic train-score calibration and consensus statistics for V43F.

The functions in this module are CPU-only.  They operate on already-computed
per-conflict-unit score aggregates and never open a dataset, model, evaluation
artifact, or GPU.  V43F seals the constants below in its preregistration.
"""

from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations

import numpy as np


ENGINE_COUNT_V43F = 4
POPULATION_SIZE_V43F = 8
CALIBRATION_WARMUPS_V43F = 2
CALIBRATION_REPEATS_V43F = 8
POST_UPDATE_REPEATS_V43F = 8
BOOTSTRAP_RESAMPLES_V43F = 10_000
BOOTSTRAP_CONFIDENCE_V43F = 0.99
CALIBRATION_BOOTSTRAP_SEED_V43F = 2_026_071_543_601
POST_UPDATE_BOOTSTRAP_SEED_V43F = 2_026_071_543_602
CALIBRATION_NOISE_SEED_V43F = 1_906_431_947
RELIABILITY_MINIMUM_V43F = 0.8
CALIBRATION_TO_RESPONSE_MAX_FRACTION_V43F = 0.25
EXPECTED_CONFLICT_UNITS_V43F = 208


def canonical_sha256_v43f(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def calibration_sigma_v43f(alpha: float, population_size: int) -> float:
    alpha = float(alpha)
    population_size = int(population_size)
    if not math.isfinite(alpha) or alpha <= 0.0 or population_size <= 0:
        raise ValueError("v43f calibration sigma requires alpha>0 and population>0")
    return alpha / math.sqrt(population_size)


def balanced_cyclic_assignments_v43f(wave_start: int) -> list[list[dict]]:
    """Return two balanced actor assignments for one four-direction wave."""
    wave_start = int(wave_start)
    if wave_start < 0 or wave_start % ENGINE_COUNT_V43F != 0:
        raise ValueError("v43f wave start must be a nonnegative multiple of four")
    result = []
    for replicate in range(2):
        result.append([
            {
                "replicate": replicate,
                "actor_rank": actor,
                "direction_index": wave_start + (
                    (actor + replicate) % ENGINE_COUNT_V43F
                ),
            }
            for actor in range(ENGINE_COUNT_V43F)
        ])
    directions = [
        item["direction_index"] for assignment in result for item in assignment
    ]
    if (
        set(directions) != set(range(wave_start, wave_start + ENGINE_COUNT_V43F))
        or any(directions.count(index) != 2 for index in set(directions))
        or any(
            len({
                item["actor_rank"]
                for assignment in result for item in assignment
                if item["direction_index"] == index
            }) != 2
            for index in set(directions)
        )
    ):
        raise RuntimeError("v43f cyclic population assignment changed")
    return result


def score_records_to_arrays_v43f(records: list[dict]) -> dict:
    """Validate retained repeats and return R x A x U unit-score arrays."""
    if not isinstance(records, list) or not records:
        raise ValueError("v43f score records are empty")
    repeats = sorted(int(item["repeat_index"]) for item in records)
    if repeats != list(range(len(records))):
        raise ValueError("v43f retained repeat indices are not contiguous")
    first_actors = sorted(records[0]["actors"], key=lambda item: item["actor_rank"])
    actor_ranks = [int(item["actor_rank"]) for item in first_actors]
    if actor_ranks != list(range(ENGINE_COUNT_V43F)):
        raise ValueError("v43f score records require actor ranks 0..3")
    first_units = first_actors[0]["units"]
    unit_ids = [str(item["unit_identity_sha256"]) for item in first_units]
    row_counts = np.asarray([int(item["row_count"]) for item in first_units], dtype=np.float64)
    if (
        len(unit_ids) != len(set(unit_ids))
        or len(unit_ids) == 0
        or np.any(row_counts <= 0)
    ):
        raise ValueError("v43f unit inventory is invalid")
    values = np.empty(
        (len(records), ENGINE_COUNT_V43F, len(unit_ids)), dtype=np.float64,
    )
    dense_hashes: list[list[str]] = []
    for repeat_index, repeat in enumerate(records):
        actors = sorted(repeat["actors"], key=lambda item: item["actor_rank"])
        if [int(item["actor_rank"]) for item in actors] != actor_ranks:
            raise ValueError("v43f actor inventory changed between repeats")
        repeat_hashes = []
        for actor_index, actor in enumerate(actors):
            units = actor["units"]
            if (
                [str(item["unit_identity_sha256"]) for item in units] != unit_ids
                or [int(item["row_count"]) for item in units]
                != row_counts.astype(np.int64).tolist()
            ):
                raise ValueError("v43f unit inventory changed between actor scores")
            unit_values = [float(item["mean_answer_token_logprob"]) for item in units]
            if not all(math.isfinite(value) for value in unit_values):
                raise ValueError("v43f unit score is non-finite")
            values[repeat_index, actor_index, :] = unit_values
            repeat_hashes.append(str(actor["aggregate"]["dense_result_sha256"]))
        dense_hashes.append(repeat_hashes)
    return {
        "values": values,
        "row_counts": row_counts,
        "unit_ids": unit_ids,
        "actor_ranks": actor_ranks,
        "dense_hashes": dense_hashes,
    }


def _metric_values_v43f(
    values: np.ndarray, row_counts: np.ndarray,
) -> dict[str, np.ndarray]:
    equal = values.mean(axis=-1, dtype=np.float64)
    row = (
        values * row_counts.reshape((1,) * (values.ndim - 1) + (-1,))
    ).sum(axis=-1, dtype=np.float64) / float(row_counts.sum())
    return {"equal_unit_mean": equal, "unweighted_row_mean": row}


def calibration_bootstrap_bounds_v43f(
    records: list[dict],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES_V43F,
    seed: int = CALIBRATION_BOOTSTRAP_SEED_V43F,
    confidence: float = BOOTSTRAP_CONFIDENCE_V43F,
    batch_size: int = 128,
) -> dict:
    """Cluster-bootstrap simultaneous actor-spread bounds from calibration.

    Each bootstrap draw resamples conflict units.  The statistic is the maximum
    actor range over all retained repeats.  A Bonferroni-adjusted per-metric
    quantile provides at least the requested familywise confidence over the two
    reported metrics.
    """
    arrays = score_records_to_arrays_v43f(records)
    values = arrays["values"]
    row_counts = arrays["row_counts"]
    resamples = int(resamples)
    batch_size = int(batch_size)
    confidence = float(confidence)
    if resamples <= 0 or batch_size <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("v43f bootstrap configuration is invalid")
    observed_metrics = _metric_values_v43f(values, row_counts)
    observed = {
        metric: {
            "maximum_repeat_actor_spread": float(
                np.max(np.ptp(metric_values, axis=1))
            ),
            "repeat_actor_spreads": [
                float(item) for item in np.ptp(metric_values, axis=1)
            ],
        }
        for metric, metric_values in observed_metrics.items()
    }
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    samples = {
        "equal_unit_mean": np.empty(resamples, dtype=np.float64),
        "unweighted_row_mean": np.empty(resamples, dtype=np.float64),
    }
    unit_count = values.shape[-1]
    offset = 0
    while offset < resamples:
        count = min(batch_size, resamples - offset)
        indices = rng.integers(0, unit_count, size=(count, unit_count), dtype=np.int64)
        selected = values[:, :, indices]
        equal = selected.mean(axis=-1, dtype=np.float64)
        selected_counts = row_counts[indices]
        row = (
            selected * selected_counts[None, None, :, :]
        ).sum(axis=-1, dtype=np.float64) / selected_counts.sum(axis=-1)[None, None, :]
        samples["equal_unit_mean"][offset:offset + count] = np.max(
            np.ptp(equal, axis=1), axis=0,
        )
        samples["unweighted_row_mean"][offset:offset + count] = np.max(
            np.ptp(row, axis=1), axis=0,
        )
        offset += count
    per_metric_quantile = 1.0 - (1.0 - confidence) / 2.0
    bounds = {}
    for metric in ("equal_unit_mean", "unweighted_row_mean"):
        upper = float(np.quantile(
            samples[metric], per_metric_quantile, method="higher",
        ))
        bounds[metric] = {
            "upper_actor_spread": upper,
            "observed_maximum_repeat_actor_spread": observed[metric][
                "maximum_repeat_actor_spread"
            ],
            "observed_repeat_actor_spreads": observed[metric][
                "repeat_actor_spreads"
            ],
        }
    result = {
        "schema": "lora-es-numeric-calibration-bootstrap-v43f",
        "method": "conflict_unit_cluster_bootstrap_max_repeat_actor_range",
        "rng": "numpy.random.PCG64",
        "seed": int(seed),
        "resamples": resamples,
        "familywise_confidence": confidence,
        "metric_count": 2,
        "per_metric_quantile": per_metric_quantile,
        "retained_repeats": values.shape[0],
        "actors": values.shape[1],
        "conflict_units": unit_count,
        "bounds": bounds,
        "dense_hash_inventory_sha256": canonical_sha256_v43f(arrays["dense_hashes"]),
    }
    result["content_sha256"] = canonical_sha256_v43f(result)
    return result


def reliability_gate_v43f(
    central_replicates: list[list[float]],
    calibration_equal_unit_bound: float,
    *,
    minimum_reliability: float = RELIABILITY_MINIMUM_V43F,
    maximum_bound_fraction: float = CALIBRATION_TO_RESPONSE_MAX_FRACTION_V43F,
) -> dict:
    if (
        len(central_replicates) != POPULATION_SIZE_V43F
        or any(len(item) != 2 for item in central_replicates)
    ):
        raise ValueError("v43f reliability requires two replicates for eight directions")
    values = np.asarray(central_replicates, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("v43f central response is non-finite")
    means = values.mean(axis=1, dtype=np.float64)
    observed_variance = float(np.mean((means - means.mean()) ** 2))
    noise_variance = float(np.mean((values[:, 0] - values[:, 1]) ** 2) / 2.0)
    mean_noise_variance = noise_variance / 2.0
    signal_variance = max(0.0, observed_variance - mean_noise_variance)
    reliability = (
        signal_variance / observed_variance if observed_variance > 0.0 else 0.0
    )
    response_std = math.sqrt(observed_variance)
    bound = float(calibration_equal_unit_bound)
    bound_fraction = bound / response_std if response_std > 0.0 else math.inf
    passed = bool(
        math.isfinite(bound)
        and bound >= 0.0
        and reliability >= float(minimum_reliability)
        and bound_fraction <= float(maximum_bound_fraction)
    )
    result = {
        "schema": "replicated-central-response-reliability-v43f",
        "central_replicates": values.tolist(),
        "central_response": means.tolist(),
        "observed_central_response_variance": observed_variance,
        "central_response_std": response_std,
        "single_replicate_noise_variance": noise_variance,
        "mean_of_two_noise_variance": mean_noise_variance,
        "estimated_signal_variance": signal_variance,
        "reliability": reliability,
        "minimum_reliability": float(minimum_reliability),
        "calibration_equal_unit_bound": bound,
        "calibration_bound_fraction_of_response_std": bound_fraction,
        "maximum_bound_fraction": float(maximum_bound_fraction),
        "passed": passed,
    }
    result["content_sha256"] = canonical_sha256_v43f(result)
    return result


def post_update_consensus_v43f(
    records: list[dict],
    calibration_bounds: dict,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES_V43F,
    seed: int = POST_UPDATE_BOOTSTRAP_SEED_V43F,
    confidence: float = BOOTSTRAP_CONFIDENCE_V43F,
    batch_size: int = 128,
) -> dict:
    """Test all actor pairs for equivalence inside preregistered bounds."""
    arrays = score_records_to_arrays_v43f(records)
    values = arrays["values"]
    row_counts = arrays["row_counts"]
    resamples = int(resamples)
    confidence = float(confidence)
    pairs = list(combinations(range(ENGINE_COUNT_V43F), 2))
    metric_values = _metric_values_v43f(values, row_counts)
    actor_means = {
        metric: metric_array.mean(axis=0, dtype=np.float64)
        for metric, metric_array in metric_values.items()
    }
    observed_spreads = {
        metric: float(np.ptp(means)) for metric, means in actor_means.items()
    }
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    unit_count = values.shape[-1]
    repeat_count = values.shape[0]
    distributions = {
        metric: np.empty((resamples, len(pairs)), dtype=np.float64)
        for metric in metric_values
    }
    offset = 0
    while offset < resamples:
        count = min(int(batch_size), resamples - offset)
        unit_indices = rng.integers(
            0, unit_count, size=(count, unit_count), dtype=np.int64,
        )
        repeat_indices = rng.integers(
            0, repeat_count, size=(count, repeat_count), dtype=np.int64,
        )
        selected = values[:, :, unit_indices].transpose(2, 0, 1, 3)
        selected = np.take_along_axis(
            selected,
            repeat_indices[:, :, None, None],
            axis=1,
        ).mean(axis=1, dtype=np.float64)
        selected_counts = row_counts[unit_indices]
        boot_metrics = {
            "equal_unit_mean": selected.mean(axis=-1, dtype=np.float64),
            "unweighted_row_mean": (
                selected * selected_counts[:, None, :]
            ).sum(axis=-1, dtype=np.float64) / selected_counts.sum(axis=-1)[:, None],
        }
        for metric, actor_values in boot_metrics.items():
            for pair_index, (left, right) in enumerate(pairs):
                distributions[metric][offset:offset + count, pair_index] = (
                    actor_values[:, left] - actor_values[:, right]
                )
        offset += count
    family_count = len(pairs) * len(metric_values)
    tail = (1.0 - confidence) / (2.0 * family_count)
    pairwise = {}
    all_intervals_inside = True
    all_spreads_inside = True
    for metric, means in actor_means.items():
        bound = float(calibration_bounds[metric]["upper_actor_spread"])
        entries = []
        for pair_index, (left, right) in enumerate(pairs):
            lower = float(np.quantile(
                distributions[metric][:, pair_index], tail, method="lower",
            ))
            upper = float(np.quantile(
                distributions[metric][:, pair_index], 1.0 - tail, method="higher",
            ))
            inside = lower >= -bound and upper <= bound
            all_intervals_inside = all_intervals_inside and inside
            entries.append({
                "actors": [left, right],
                "observed_difference": float(means[left] - means[right]),
                "simultaneous_ci": [lower, upper],
                "equivalence_margin": bound,
                "inside_equivalence_margin": bool(inside),
            })
        spread_inside = observed_spreads[metric] <= bound
        all_spreads_inside = all_spreads_inside and spread_inside
        pairwise[metric] = {
            "actor_means": [float(value) for value in means],
            "actor_mean_spread": observed_spreads[metric],
            "equivalence_margin": bound,
            "spread_inside_equivalence_margin": bool(spread_inside),
            "pairs": entries,
        }
    result = {
        "schema": "post-update-cross-actor-equivalence-v43f",
        "method": "paired_conflict_unit_and_repeat_bootstrap_bonferroni",
        "rng": "numpy.random.PCG64",
        "seed": int(seed),
        "resamples": resamples,
        "familywise_confidence": confidence,
        "comparison_family_count": family_count,
        "per_tail_probability": tail,
        "retained_repeats": repeat_count,
        "actors": ENGINE_COUNT_V43F,
        "conflict_units": unit_count,
        "metrics": pairwise,
        "all_pairwise_intervals_inside_calibrated_margins": bool(
            all_intervals_inside
        ),
        "all_actor_mean_spreads_inside_calibrated_margins": bool(
            all_spreads_inside
        ),
        "passed": bool(all_intervals_inside and all_spreads_inside),
        "dense_hash_inventory_sha256": canonical_sha256_v43f(arrays["dense_hashes"]),
    }
    result["content_sha256"] = canonical_sha256_v43f(result)
    return result
