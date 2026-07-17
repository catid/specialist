#!/usr/bin/env python3
"""Fail-closed CPU reward shaping for a compute-matched mirrored-ES ablation.

This module deliberately imports no dataset, model, torch, Ray, vLLM, or GPU
code.  It consumes an already-scored, train-only mirrored population and
compiles four preregistered views of exactly the same rewards:

* raw signed rewards;
* z-scores computed independently inside each prompt/CRN group;
* centered midranks computed independently inside each prompt/CRN group; and
* direct antithetic pair differences.

One call is exactly one population boundary.  This is stricter than merely
including the boundary in a grouping key: an accidental train/dev, seed,
generation, or population mixture fails rather than silently producing
multiple adaptive statistics.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SCHEMA_V1 = "specialist-prompt-local-reward-shaping-v1"
ALLOWED_DATASET_ROLE_V1 = "train"
ZSCORE_EPSILON_V1 = 1e-8
METHODS_V1 = (
    "raw_rewards",
    "within_prompt_centered_zscore",
    "within_prompt_centered_rank",
    "antithetic_pair_difference",
)
POPULATION_FIELDS_V1 = (
    "dataset_role",
    "training_seed",
    "generation_index",
    "population_id",
    "evaluation_contract_sha256",
)
PROMPT_GROUP_FIELDS_V1 = (
    "prompt_group_id",
    "repeat_index",
    "evaluation_seed",
)
RECORD_FIELDS_V1 = frozenset(
    POPULATION_FIELDS_V1
    + PROMPT_GROUP_FIELDS_V1
    + ("direction_index", "direction_seed", "sign", "reward")
)


def canonical_sha256_v1(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _exact_nonnegative_int_v1(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"reward-shaping {label} must be an exact integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"reward-shaping {label} must be nonnegative")
    return result


def _opaque_identity_v1(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError(f"reward-shaping {label} must be a nonempty short string")
    try:
        value.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError(f"reward-shaping {label} must be opaque ASCII") from error
    if any(character.isspace() for character in value):
        raise ValueError(f"reward-shaping {label} may not contain whitespace")
    return value


def _sha256_identity_v1(value: Any, label: str) -> str:
    result = _opaque_identity_v1(value, label)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError(f"reward-shaping {label} must be lowercase SHA-256")
    return result


def _finite_float_v1(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"reward-shaping {label} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"reward-shaping {label} must be finite")
    return 0.0 if result == 0.0 else result


def _finite_l2_v1(values: Sequence[float], label: str) -> float:
    result = math.hypot(*values)
    if not math.isfinite(result):
        raise RuntimeError(f"reward-shaping {label} overflowed")
    return result


def _direction_seeds_v1(direction_seeds: Sequence[int]) -> tuple[int, ...]:
    if isinstance(direction_seeds, (str, bytes)):
        raise ValueError("reward-shaping direction seeds must be a sequence")
    result = tuple(
        _exact_nonnegative_int_v1(value, "direction seed")
        for value in direction_seeds
    )
    if len(result) < 2 or len(set(result)) != len(result):
        raise ValueError(
            "reward-shaping requires at least two unique direction seeds"
        )
    return result


def centered_ranks_v1(values: Iterable[float]) -> list[float]:
    """Map finite values to [-0.5, 0.5] using exact-tie midranks."""
    values = [_finite_float_v1(value, "rank value") for value in values]
    if len(values) <= 1:
        return [0.0] * len(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        midrank = 0.5 * (start + stop - 1)
        for position in range(start, stop):
            ranks[order[position]] = midrank
        start = stop
    denominator = len(values) - 1
    result = [rank / denominator - 0.5 for rank in ranks]
    return [0.0 if value == 0.0 else value for value in result]


def centered_zscores_v1(values: Iterable[float]) -> tuple[list[float], dict]:
    """Return population z-scores with the current EGGROLL ``+1e-8`` guard."""
    values = [_finite_float_v1(value, "z-score value") for value in values]
    if not values:
        raise ValueError("reward-shaping z-score group must not be empty")
    try:
        # Divide before summing so a representable mean does not overflow just
        # because the unnormalized total is outside binary64.
        mean = math.fsum(value / len(values) for value in values)
    except OverflowError as error:
        raise RuntimeError("reward-shaping z-score mean overflowed") from error
    centered = [value - mean for value in values]
    if not all(math.isfinite(value) for value in centered):
        raise RuntimeError("reward-shaping z-score centering overflowed")
    scale = max((abs(value) for value in centered), default=0.0)
    if scale == 0.0:
        variance = 0.0
        standard_deviation = 0.0
        result = [0.0] * len(values)
    else:
        normalized_variance = math.fsum(
            (value / scale) ** 2 for value in centered
        ) / len(values)
        standard_deviation = scale * math.sqrt(normalized_variance)
        variance = standard_deviation * standard_deviation
        if not math.isfinite(variance) or not math.isfinite(standard_deviation):
            raise RuntimeError("reward-shaping z-score variance overflowed")
        denominator = standard_deviation + ZSCORE_EPSILON_V1
        if not math.isfinite(denominator):
            raise RuntimeError("reward-shaping z-score denominator overflowed")
        result = [value / denominator for value in centered]
        result = [0.0 if value == 0.0 else value for value in result]
    return result, {
        "mean": mean,
        "population_variance": variance,
        "population_standard_deviation": standard_deviation,
        "denominator_epsilon": ZSCORE_EPSILON_V1,
        "zero_spread": standard_deviation == 0.0,
    }


def _validate_population_v1(
    records: Sequence[Mapping[str, Any]],
    direction_seeds: Sequence[int],
    evaluation_contract_sha256: str,
) -> tuple[dict, list[dict], tuple[int, ...]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("reward-shaping records must be a sequence")
    if not records:
        raise ValueError("reward-shaping population must not be empty")
    seeds = _direction_seeds_v1(direction_seeds)
    expected_contract = _sha256_identity_v1(
        evaluation_contract_sha256, "evaluation contract"
    )
    normalized: list[dict] = []
    population_boundaries = set()
    for position, source in enumerate(records):
        if not isinstance(source, Mapping) or set(source) != RECORD_FIELDS_V1:
            raise ValueError(
                "reward-shaping record fields changed; semantic or holdout "
                f"payloads are forbidden (record {position})"
            )
        role = source["dataset_role"]
        if role != ALLOWED_DATASET_ROLE_V1:
            raise ValueError("reward-shaping adaptation accepts train records only")
        contract = _sha256_identity_v1(
            source["evaluation_contract_sha256"], "record evaluation contract"
        )
        if contract != expected_contract:
            raise RuntimeError("reward-shaping evaluation contract identity changed")
        direction_index = _exact_nonnegative_int_v1(
            source["direction_index"], "direction index"
        )
        if direction_index >= len(seeds):
            raise ValueError("reward-shaping direction index is outside population")
        direction_seed = _exact_nonnegative_int_v1(
            source["direction_seed"], "record direction seed"
        )
        if direction_seed != seeds[direction_index]:
            raise RuntimeError("reward-shaping direction seed identity changed")
        sign = source["sign"]
        if isinstance(sign, bool) or not isinstance(sign, numbers.Integral):
            raise ValueError("reward-shaping sign must be the exact integer +/-1")
        sign = int(sign)
        if sign not in (-1, 1):
            raise ValueError("reward-shaping sign must be +/-1")
        item = {
            "dataset_role": role,
            "training_seed": _exact_nonnegative_int_v1(
                source["training_seed"], "training seed"
            ),
            "generation_index": _exact_nonnegative_int_v1(
                source["generation_index"], "generation index"
            ),
            "population_id": _opaque_identity_v1(
                source["population_id"], "population identity"
            ),
            "evaluation_contract_sha256": contract,
            "prompt_group_id": _opaque_identity_v1(
                source["prompt_group_id"], "prompt group identity"
            ),
            "repeat_index": _exact_nonnegative_int_v1(
                source["repeat_index"], "repeat index"
            ),
            "evaluation_seed": _exact_nonnegative_int_v1(
                source["evaluation_seed"], "evaluation seed"
            ),
            "direction_index": direction_index,
            "direction_seed": direction_seed,
            "sign": sign,
            "reward": _finite_float_v1(source["reward"], "reward"),
        }
        population_boundaries.add(tuple(item[field] for field in POPULATION_FIELDS_V1))
        normalized.append(item)
    if len(population_boundaries) != 1:
        raise ValueError(
            "reward-shaping call crossed a role/seed/generation/population boundary"
        )
    boundary_tuple = next(iter(population_boundaries))
    boundary = dict(zip(POPULATION_FIELDS_V1, boundary_tuple, strict=True))
    return boundary, normalized, seeds


def _complete_prompt_groups_v1(
    records: Sequence[dict], direction_seeds: tuple[int, ...]
) -> list[tuple[dict, dict[tuple[int, int], float]]]:
    groups: dict[tuple[Any, ...], dict[tuple[int, int], float]] = {}
    for item in records:
        group_key = tuple(item[field] for field in PROMPT_GROUP_FIELDS_V1)
        candidate_key = (item["direction_index"], item["sign"])
        group = groups.setdefault(group_key, {})
        if candidate_key in group:
            raise ValueError("reward-shaping prompt group has a duplicate candidate")
        group[candidate_key] = item["reward"]
    expected = {
        (direction_index, sign)
        for direction_index in range(len(direction_seeds))
        for sign in (1, -1)
    }
    result = []
    for group_key in sorted(groups):
        observed = groups[group_key]
        if set(observed) != expected:
            raise RuntimeError(
                "reward-shaping prompt group does not contain one complete "
                "mirrored population"
            )
        identity = dict(zip(PROMPT_GROUP_FIELDS_V1, group_key, strict=True))
        result.append((identity, observed))
    return result


def _shape_prompt_group_v1(
    identity: dict,
    rewards: dict[tuple[int, int], float],
    direction_seeds: tuple[int, ...],
    method: str,
) -> dict:
    candidate_keys = [
        (direction_index, sign)
        for direction_index in range(len(direction_seeds))
        for sign in (1, -1)
    ]
    raw = [rewards[key] for key in candidate_keys]
    statistics: dict[str, Any]
    if method == "raw_rewards":
        utilities: list[float] | None = list(raw)
        statistics = {
            "location_or_scale_estimated": False,
            "zero_spread": len(set(raw)) == 1,
        }
    elif method == "within_prompt_centered_zscore":
        utilities, statistics = centered_zscores_v1(raw)
    elif method == "within_prompt_centered_rank":
        utilities = centered_ranks_v1(raw)
        statistics = {
            "exact_ties_receive_identical_midrank": True,
            "zero_spread": len(set(raw)) == 1,
        }
    elif method == "antithetic_pair_difference":
        utilities = None
        statistics = {
            "candidate_utility_materialized": False,
            "direct_pair_reduction": True,
            "zero_spread": len(set(raw)) == 1,
        }
    else:
        raise ValueError(f"unknown reward-shaping method: {method!r}")

    utility_lookup = (
        dict(zip(candidate_keys, utilities, strict=True))
        if utilities is not None else None
    )
    pairs = []
    for direction_index, seed in enumerate(direction_seeds):
        plus = rewards[(direction_index, 1)]
        minus = rewards[(direction_index, -1)]
        if utility_lookup is None:
            coefficient = plus - minus
            plus_utility = None
            minus_utility = None
        else:
            plus_utility = utility_lookup[(direction_index, 1)]
            minus_utility = utility_lookup[(direction_index, -1)]
            coefficient = plus_utility - minus_utility
        if not math.isfinite(coefficient):
            raise RuntimeError("reward-shaping pair coefficient overflowed")
        pairs.append({
            "direction_index": direction_index,
            "direction_seed": seed,
            "reward_plus": plus,
            "reward_minus": minus,
            "utility_plus": plus_utility,
            "utility_minus": minus_utility,
            "coefficient": 0.0 if coefficient == 0.0 else coefficient,
        })
    result = {
        "group": identity,
        "candidate_count": len(raw),
        "statistics": statistics,
        "pairs": pairs,
    }
    result["group_content_sha256"] = canonical_sha256_v1(result)
    return result


def shape_reward_population_v1(
    records: Sequence[Mapping[str, Any]],
    direction_seeds: Sequence[int],
    evaluation_contract_sha256: str,
    method: str,
) -> dict:
    """Compile one complete train population into direction coefficients."""
    if method not in METHODS_V1:
        raise ValueError(f"unknown reward-shaping method: {method!r}")
    boundary, normalized, seeds = _validate_population_v1(
        records, direction_seeds, evaluation_contract_sha256
    )
    prompt_groups = _complete_prompt_groups_v1(normalized, seeds)
    shaped_groups = [
        _shape_prompt_group_v1(identity, rewards, seeds, method)
        for identity, rewards in prompt_groups
    ]
    coefficients = []
    for direction_index in range(len(seeds)):
        values = [
            group["pairs"][direction_index]["coefficient"]
            for group in shaped_groups
        ]
        try:
            coefficient = math.fsum(value / len(values) for value in values)
        except OverflowError as error:
            raise RuntimeError(
                "reward-shaping prompt aggregation overflowed"
            ) from error
        if not math.isfinite(coefficient):
            raise RuntimeError("reward-shaping prompt aggregation overflowed")
        coefficients.append(0.0 if coefficient == 0.0 else coefficient)
    coefficient_l2 = _finite_l2_v1(coefficients, "coefficient L2")
    result = {
        "schema": SCHEMA_V1,
        "method": method,
        "population_boundary": boundary,
        "statistical_group_fields": list(PROMPT_GROUP_FIELDS_V1),
        "statistics_never_pool_across_prompt_groups": True,
        "one_population_boundary_per_call": True,
        "direction_seeds": list(seeds),
        "signed_candidates_per_prompt_group": 2 * len(seeds),
        "prompt_group_count": len(shaped_groups),
        "candidate_reward_count": len(normalized),
        "prompt_groups": shaped_groups,
        "direction_coefficients": coefficients,
        "coefficient_l2": coefficient_l2,
        "zero_update": coefficient_l2 == 0.0,
        "protected_semantics_opened": False,
    }
    result["content_sha256"] = canonical_sha256_v1(result)
    return result


def compare_reward_shaping_v1(
    records: Sequence[Mapping[str, Any]],
    direction_seeds: Sequence[int],
    evaluation_contract_sha256: str,
) -> dict:
    """Compile every arm from the same immutable candidate reward tensor."""
    arms = {
        method: shape_reward_population_v1(
            records, direction_seeds, evaluation_contract_sha256, method
        )
        for method in METHODS_V1
    }
    raw = arms["raw_rewards"]["direction_coefficients"]
    paired = arms["antithetic_pair_difference"]["direction_coefficients"]
    if raw != paired:
        raise RuntimeError(
            "raw signed mirrored rewards lost pair-difference equivalence"
        )
    cosines = {}
    for left_index, left_name in enumerate(METHODS_V1):
        for right_name in METHODS_V1[left_index + 1:]:
            cosines[f"{left_name}__vs__{right_name}"] = coefficient_cosine_v1(
                arms[left_name]["direction_coefficients"],
                arms[right_name]["direction_coefficients"],
            )
    result = {
        "schema": "specialist-reward-shaping-comparison-v1",
        "methods": list(METHODS_V1),
        "same_candidate_rewards_for_every_method": True,
        "arms": arms,
        "pairwise_direction_cosines": cosines,
        "raw_and_direct_pair_coefficients_exactly_equal": True,
        "raw_and_direct_pair_are_one_gradient_estimator_not_two_independent_arms": True,
    }
    result["content_sha256"] = canonical_sha256_v1(result)
    return result


def coefficient_cosine_v1(left: Sequence[float], right: Sequence[float]) -> float | None:
    left = [_finite_float_v1(value, "left coefficient") for value in left]
    right = [_finite_float_v1(value, "right coefficient") for value in right]
    if len(left) != len(right) or not left:
        raise ValueError("reward-shaping coefficient vectors must align")
    left_scale = max(abs(value) for value in left)
    right_scale = max(abs(value) for value in right)
    if left_scale == 0.0 or right_scale == 0.0:
        return None
    left_scaled = [value / left_scale for value in left]
    right_scaled = [value / right_scale for value in right]
    left_norm = _finite_l2_v1(left_scaled, "scaled left coefficient L2")
    right_norm = _finite_l2_v1(right_scaled, "scaled right coefficient L2")
    cosine = math.fsum(
        a * b for a, b in zip(left_scaled, right_scaled, strict=True)
    ) / (
        left_norm * right_norm
    )
    # A correctly computed cosine can exceed one by a few ulps after the
    # separate norm products.  Keep the public diagnostic in its exact range.
    return max(-1.0, min(1.0, cosine))


def _record_identity_v1(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in sorted(RECORD_FIELDS_V1 - {"reward"}))


def outlier_sensitivity_v1(
    clean_records: Sequence[Mapping[str, Any]],
    contaminated_records: Sequence[Mapping[str, Any]],
    direction_seeds: Sequence[int],
    evaluation_contract_sha256: str,
) -> dict:
    """Compare one clean tensor with the same tensor after one reward outlier."""
    clean_by_id = {_record_identity_v1(item): item for item in clean_records}
    contaminated_by_id = {
        _record_identity_v1(item): item for item in contaminated_records
    }
    if (
        len(clean_by_id) != len(clean_records)
        or len(contaminated_by_id) != len(contaminated_records)
        or set(clean_by_id) != set(contaminated_by_id)
    ):
        raise ValueError("outlier diagnostic candidate identities changed")
    changed = [
        identity for identity in clean_by_id
        if clean_by_id[identity].get("reward")
        != contaminated_by_id[identity].get("reward")
    ]
    if len(changed) != 1:
        raise ValueError("outlier diagnostic requires exactly one changed reward")
    clean = compare_reward_shaping_v1(
        clean_records, direction_seeds, evaluation_contract_sha256
    )
    contaminated = compare_reward_shaping_v1(
        contaminated_records, direction_seeds, evaluation_contract_sha256
    )
    method_diagnostics = {}
    for method in METHODS_V1:
        left = clean["arms"][method]
        right = contaminated["arms"][method]
        left_groups = {
            tuple(group["group"].items()): group["group_content_sha256"]
            for group in left["prompt_groups"]
        }
        right_groups = {
            tuple(group["group"].items()): group["group_content_sha256"]
            for group in right["prompt_groups"]
        }
        unchanged_group_count = sum(
            left_groups[key] == right_groups[key] for key in left_groups
        )
        delta = [
            after - before
            for before, after in zip(
                left["direction_coefficients"],
                right["direction_coefficients"],
                strict=True,
            )
        ]
        method_diagnostics[method] = {
            "clean_to_contaminated_cosine": coefficient_cosine_v1(
                left["direction_coefficients"], right["direction_coefficients"]
            ),
            "coefficient_l2_delta": _finite_l2_v1(
                delta, "outlier coefficient L2 delta"
            ),
            "unchanged_prompt_group_count": unchanged_group_count,
            "total_prompt_group_count": len(left_groups),
            "only_contaminated_prompt_group_changed": (
                unchanged_group_count == len(left_groups) - 1
            ),
        }
    result = {
        "schema": "specialist-reward-shaping-single-outlier-diagnostic-v1",
        "changed_candidate_identity_sha256": canonical_sha256_v1(changed[0]),
        "method_diagnostics": method_diagnostics,
        "raw_and_direct_pair_remain_exactly_equivalent": (
            clean["arms"]["raw_rewards"]["direction_coefficients"]
            == clean["arms"]["antithetic_pair_difference"][
                "direction_coefficients"
            ]
            and contaminated["arms"]["raw_rewards"]["direction_coefficients"]
            == contaminated["arms"]["antithetic_pair_difference"][
                "direction_coefficients"
            ]
        ),
    }
    result["content_sha256"] = canonical_sha256_v1(result)
    return result


def multi_seed_stability_v1(seed_outputs: Mapping[int, Mapping[str, Any]]) -> dict:
    """Report pairwise direction stability after each seed is shaped alone."""
    if not isinstance(seed_outputs, Mapping) or len(seed_outputs) < 2:
        raise ValueError("reward-shaping stability requires at least two seeds")
    normalized = {}
    for raw_seed, output in seed_outputs.items():
        seed = _exact_nonnegative_int_v1(raw_seed, "stability seed")
        if seed in normalized or not isinstance(output, Mapping):
            raise ValueError("reward-shaping stability seed inventory changed")
        if output.get("schema") != SCHEMA_V1:
            raise ValueError("reward-shaping stability output schema changed")
        sealed = dict(output)
        observed_content_sha = sealed.pop("content_sha256", None)
        if (
            not isinstance(observed_content_sha, str)
            or canonical_sha256_v1(sealed) != observed_content_sha
        ):
            raise RuntimeError("reward-shaping stability output identity changed")
        if output.get("population_boundary", {}).get("training_seed") != seed:
            raise ValueError("reward-shaping stability seed label changed")
        normalized[seed] = output
    methods = {item.get("method") for item in normalized.values()}
    direction_seeds = {
        tuple(item.get("direction_seeds", [])) for item in normalized.values()
    }
    contracts = {
        item.get("population_boundary", {}).get("evaluation_contract_sha256")
        for item in normalized.values()
    }
    generations = {
        item.get("population_boundary", {}).get("generation_index")
        for item in normalized.values()
    }
    group_counts = {item.get("prompt_group_count") for item in normalized.values()}
    reward_counts = {item.get("candidate_reward_count") for item in normalized.values()}
    if (
        len(methods) != 1
        or len(direction_seeds) != 1
        or len(contracts) != 1
        or len(generations) != 1
        or len(group_counts) != 1
        or len(reward_counts) != 1
    ):
        raise ValueError("reward-shaping stability outputs do not share an arm")
    pairs = []
    ordered_seeds = sorted(normalized)
    for left_index, left_seed in enumerate(ordered_seeds):
        for right_seed in ordered_seeds[left_index + 1:]:
            pairs.append({
                "left_training_seed": left_seed,
                "right_training_seed": right_seed,
                "direction_cosine": coefficient_cosine_v1(
                    normalized[left_seed]["direction_coefficients"],
                    normalized[right_seed]["direction_coefficients"],
                ),
            })
    finite = [item["direction_cosine"] for item in pairs
              if item["direction_cosine"] is not None]
    result = {
        "schema": "specialist-reward-shaping-multi-seed-stability-v1",
        "method": next(iter(methods)),
        "training_seeds": ordered_seeds,
        "direction_seeds": list(next(iter(direction_seeds))),
        "pairwise": pairs,
        "minimum_nonzero_direction_cosine": min(finite) if finite else None,
        "mean_nonzero_direction_cosine": (
            math.fsum(finite) / len(finite) if finite else None
        ),
        "zero_vector_pair_count": len(pairs) - len(finite),
    }
    result["content_sha256"] = canonical_sha256_v1(result)
    return result
