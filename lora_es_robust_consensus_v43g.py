#!/usr/bin/env python3
"""CPU-only robust population statistics for train-only LoRA-ES V43G.

V43G keeps V43F's fixed-panel calibration and post-update equivalence code,
but does not use a conflict-unit bootstrap as a scale estimate for a finite
panel ES response.  Population reliability is instead measured directly from
a complete four-actor block for every antithetic direction.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

import lora_es_numeric_consensus_v43f as v43f


ENGINE_COUNT_V43G = 4
POPULATION_SIZE_V43G = 8
SIGNED_REPLICATES_V43G = 4
CALIBRATION_WARMUPS_V43G = v43f.CALIBRATION_WARMUPS_V43F
CALIBRATION_REPEATS_V43G = v43f.CALIBRATION_REPEATS_V43F
POST_UPDATE_REPEATS_V43G = v43f.POST_UPDATE_REPEATS_V43F
BOOTSTRAP_RESAMPLES_V43G = v43f.BOOTSTRAP_RESAMPLES_V43F
BOOTSTRAP_CONFIDENCE_V43G = v43f.BOOTSTRAP_CONFIDENCE_V43F
CALIBRATION_BOOTSTRAP_SEED_V43G = 2_026_071_543_701
POST_UPDATE_BOOTSTRAP_SEED_V43G = 2_026_071_543_702
CALIBRATION_NOISE_SEED_V43G = v43f.CALIBRATION_NOISE_SEED_V43F
RELIABILITY_MINIMUM_V43G = 0.8
SPLIT_HALF_SPEARMAN_MINIMUM_V43G = 0.7
EXPECTED_CONFLICT_UNITS_V43G = v43f.EXPECTED_CONFLICT_UNITS_V43F

# This is a historical catastrophic-divergence ceiling, not a response-scale
# estimate.  It was sealed by V43F before its population was evaluated.
V43F_HISTORICAL_EQUAL_UNIT_BOUND = 0.001802103667415178

# Compatibility aliases used by the inherited calibration/equivalence path.
calibration_sigma_v43g = v43f.calibration_sigma_v43f
canonical_sha256_v43g = v43f.canonical_sha256_v43f


def calibration_bootstrap_bounds_v43g(records: list[dict], **kwargs) -> dict:
    kwargs.setdefault("seed", CALIBRATION_BOOTSTRAP_SEED_V43G)
    return v43f.calibration_bootstrap_bounds_v43f(records, **kwargs)


def post_update_consensus_v43g(
    records: list[dict], calibration_bounds: dict, **kwargs,
) -> dict:
    kwargs.setdefault("seed", POST_UPDATE_BOOTSTRAP_SEED_V43G)
    return v43f.post_update_consensus_v43f(
        records, calibration_bounds, **kwargs,
    )


def complete_actor_assignments_v43g(direction_index: int) -> list[dict]:
    """Assign one direction to every actor, one paired replicate per actor."""
    direction_index = int(direction_index)
    if not 0 <= direction_index < POPULATION_SIZE_V43G:
        raise ValueError("v43g direction index is outside the sealed population")
    return [
        {
            "direction_index": direction_index,
            "replicate": actor,
            "actor_rank": actor,
        }
        for actor in range(ENGINE_COUNT_V43G)
    ]


def _average_tie_ranks(values: Iterable[float]) -> list[float]:
    values = [float(value) for value in values]
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = 0.5 * (start + stop - 1)
        for position in range(start, stop):
            result[order[position]] = rank
        start = stop
    return result


def centered_ranks_v43g(values: Iterable[float]) -> list[float]:
    """Map finite fitnesses to [-0.5, 0.5], averaging exact ties."""
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("v43g fitness contains a non-finite value")
    if len(values) <= 1:
        return [0.0] * len(values)
    ranks = _average_tie_ranks(values)
    return [rank / (len(values) - 1) - 0.5 for rank in ranks]


def _median_four(values: Iterable[float]) -> float:
    values = sorted(float(value) for value in values)
    if len(values) != SIGNED_REPLICATES_V43G:
        raise ValueError("v43g robust location requires exactly four actors")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("v43g robust location contains a non-finite value")
    return 0.5 * (values[1] + values[2])


def robust_population_v43g(equal_unit_sign_scores: dict) -> dict:
    """Form median signed scores and standard EGGROLL centered-rank utilities."""
    if set(equal_unit_sign_scores) != {"plus", "minus"}:
        raise ValueError("v43g signed score inventory changed")
    arrays = {
        sign: np.asarray(equal_unit_sign_scores[sign], dtype=np.float64)
        for sign in ("plus", "minus")
    }
    expected = (POPULATION_SIZE_V43G, SIGNED_REPLICATES_V43G)
    if any(values.shape != expected for values in arrays.values()):
        raise ValueError("v43g signed score matrix must be 8 directions by 4 actors")
    if not all(np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("v43g signed score matrix contains a non-finite value")
    central_replicates = 0.5 * (arrays["plus"] - arrays["minus"])
    robust_signed = {
        sign: [_median_four(row) for row in arrays[sign]]
        for sign in ("plus", "minus")
    }
    robust_central = [_median_four(row) for row in central_replicates]
    signed_fitness = robust_signed["plus"] + robust_signed["minus"]
    utilities = centered_ranks_v43g(signed_fitness)
    plus_utilities = utilities[:POPULATION_SIZE_V43G]
    minus_utilities = utilities[POPULATION_SIZE_V43G:]
    coefficients = [
        plus - minus
        for plus, minus in zip(plus_utilities, minus_utilities, strict=True)
    ]
    return {
        "schema": "robust-centered-rank-population-v43g",
        "central_replicates": central_replicates.tolist(),
        "robust_central_response": robust_central,
        "robust_signed_scores": robust_signed,
        "signed_centered_rank_utilities": {
            "plus": plus_utilities,
            "minus": minus_utilities,
        },
        "coefficients": coefficients,
        "zero_utility_update": all(value == 0.0 for value in coefficients),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(dtype=np.float64)
    right = right - right.mean(dtype=np.float64)
    denominator = math.sqrt(
        float(np.mean(left * left)) * float(np.mean(right * right))
    )
    return float(np.mean(left * right) / denominator) if denominator > 0.0 else 0.0


def reliability_gate_v43g(
    central_replicates: list[list[float]],
    fresh_calibration_observed_maximum: float,
    *,
    minimum_reliability: float = RELIABILITY_MINIMUM_V43G,
    minimum_split_half_spearman: float = SPLIT_HALF_SPEARMAN_MINIMUM_V43G,
    historical_calibration_ceiling: float = V43F_HISTORICAL_EQUAL_UNIT_BOUND,
) -> dict:
    """Gate a four-actor population without fitting thresholds to that run."""
    values = np.asarray(central_replicates, dtype=np.float64)
    if values.shape != (POPULATION_SIZE_V43G, SIGNED_REPLICATES_V43G):
        raise ValueError("v43g reliability requires an 8 by 4 central matrix")
    if not np.isfinite(values).all():
        raise ValueError("v43g central response contains a non-finite value")
    means = values.mean(axis=1, dtype=np.float64)
    observed_variance = float(np.mean((means - means.mean()) ** 2))
    within_direction_variances = np.var(values, axis=1, ddof=1, dtype=np.float64)
    single_replicate_noise = float(within_direction_variances.mean())
    mean_of_four_noise = single_replicate_noise / SIGNED_REPLICATES_V43G
    signal_variance = max(0.0, observed_variance - mean_of_four_noise)
    reliability = signal_variance / observed_variance if observed_variance > 0.0 else 0.0

    left = values[:, :2].mean(axis=1, dtype=np.float64)
    right = values[:, 2:].mean(axis=1, dtype=np.float64)
    left_ranks = np.asarray(_average_tie_ranks(left.tolist()), dtype=np.float64)
    right_ranks = np.asarray(_average_tie_ranks(right.tolist()), dtype=np.float64)
    split_spearman = _correlation(left_ranks, right_ranks)
    split_pearson = _correlation(left, right)

    fresh_bound = float(fresh_calibration_observed_maximum)
    historical_ceiling = float(historical_calibration_ceiling)
    calibration_safe = bool(
        math.isfinite(fresh_bound)
        and fresh_bound >= 0.0
        and fresh_bound <= historical_ceiling
    )
    passed = bool(
        reliability >= float(minimum_reliability)
        and split_spearman >= float(minimum_split_half_spearman)
        and calibration_safe
    )
    result = {
        "schema": "complete-actor-block-reliability-v43g",
        "central_replicates": values.tolist(),
        "actor_mean_central_response": means.tolist(),
        "observed_actor_mean_response_variance": observed_variance,
        "single_replicate_noise_variance": single_replicate_noise,
        "mean_of_four_noise_variance": mean_of_four_noise,
        "estimated_signal_variance": signal_variance,
        "reliability": reliability,
        "minimum_reliability": float(minimum_reliability),
        "split_half_actor_groups": [[0, 1], [2, 3]],
        "split_half_spearman": split_spearman,
        "split_half_pearson": split_pearson,
        "minimum_split_half_spearman": float(minimum_split_half_spearman),
        "fresh_calibration_observed_maximum_actor_spread": fresh_bound,
        "historical_v43f_calibration_ceiling": historical_ceiling,
        "fresh_calibration_inside_historical_ceiling": calibration_safe,
        "passed": passed,
    }
    result["content_sha256"] = canonical_sha256_v43g(result)
    return result


def predicted_reliability_v43g(
    signal_variance_v43f: float,
    single_replicate_noise_variance_v43f: float,
    *,
    sigma_multiplier: float = 2.0,
    replicates: int = SIGNED_REPLICATES_V43G,
) -> float:
    """Linear-response planning projection; never an acceptance statistic."""
    signal = float(signal_variance_v43f) * float(sigma_multiplier) ** 2
    noise = float(single_replicate_noise_variance_v43f) / int(replicates)
    return signal / (signal + noise) if signal + noise > 0.0 else 0.0


def diagnose_v43f_artifacts_v43g(
    calibration_artifact: dict, reliability_artifact: dict,
) -> dict:
    """Derive the sealed V43G design evidence from V43F train-only aggregates."""
    if (
        calibration_artifact.get("schema")
        != "matched-lora-es-numeric-calibration-v43f"
        or reliability_artifact.get("schema")
        != "matched-lora-es-population-reliability-v43f"
        or calibration_artifact.get(
            "shadow_dev_external_eval_ood_or_holdout_opened"
        ) is not False
        or reliability_artifact.get(
            "shadow_dev_external_eval_ood_or_holdout_opened"
        ) is not False
    ):
        raise ValueError("v43g diagnostics require sealed train-only V43F artifacts")
    gate = reliability_artifact["reliability_gate"]
    central = np.asarray(gate["central_replicates"], dtype=np.float64)
    if central.shape != (8, 2) or not np.isfinite(central).all():
        raise ValueError("v43f central replicate evidence changed")

    calibration_scores = np.asarray([
        [
            float(actor["aggregate"]["equal_unit_mean"])
            for actor in sorted(record["actors"], key=lambda item: item["actor_rank"])
        ]
        for record in calibration_artifact["records"]
    ], dtype=np.float64)
    if calibration_scores.shape != (8, 4):
        raise ValueError("v43f calibration aggregate evidence changed")
    repeat_pattern_correlations = [
        _correlation(calibration_scores[left], calibration_scores[right])
        for left in range(8) for right in range(left)
    ]
    disagreements = np.abs(central[:, 0] - central[:, 1])
    largest = sorted(range(8), key=lambda index: (-disagreements[index], index))[:3]

    def two_replicate_reliability(values: np.ndarray) -> float:
        means = values.mean(axis=1, dtype=np.float64)
        observed = float(np.mean((means - means.mean()) ** 2))
        noise = float(np.mean((values[:, 0] - values[:, 1]) ** 2) / 2.0)
        signal = max(0.0, observed - noise / 2.0)
        return signal / observed if observed > 0.0 else 0.0

    calibration_bound = float(
        calibration_artifact["bootstrap"]["bounds"]["equal_unit_mean"]
        ["upper_actor_spread"]
    )
    calibration_observed = float(
        calibration_artifact["bootstrap"]["bounds"]["equal_unit_mean"]
        ["observed_maximum_repeat_actor_spread"]
    )
    response_std = float(gate["central_response_std"])
    projected = predicted_reliability_v43g(
        gate["estimated_signal_variance"],
        gate["single_replicate_noise_variance"],
    )
    result = {
        "schema": "v43f-train-only-failure-diagnostic-v43g",
        "v43f_reliability": float(gate["reliability"]),
        "v43f_calibration_bound_fraction": float(
            gate["calibration_bound_fraction_of_response_std"]
        ),
        "central_response_std": response_std,
        "calibration_bootstrap_upper_actor_spread": calibration_bound,
        "calibration_observed_maximum_actor_spread": calibration_observed,
        "bootstrap_to_observed_spread_ratio": (
            calibration_bound / calibration_observed
        ),
        "observed_spread_to_response_std": calibration_observed / response_std,
        "mean_calibration_repeat_actor_pattern_correlation": float(
            np.mean(repeat_pattern_correlations)
        ),
        "largest_replicate_disagreement_direction_indices": largest,
        "largest_replicate_disagreements": [
            float(disagreements[index]) for index in largest
        ],
        "leave_direction_3_out_reliability": two_replicate_reliability(
            np.delete(central, 3, axis=0)
        ),
        "v43g_sigma_multiplier": 2.0,
        "v43g_signed_replicates": SIGNED_REPLICATES_V43G,
        "linear_response_projected_v43g_reliability": projected,
        "interpretation": {
            "exact_state_failure_observed": False,
            "fixed_actor_bias_supported": False,
            "transient_heavy_tail_evaluation_noise_supported": True,
            "cluster_bootstrap_used_as_finite_panel_response_scale": True,
            "weakening_v43f_gate_alone_supported": False,
        },
    }
    result["content_sha256"] = canonical_sha256_v43g(result)
    return result
