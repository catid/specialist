#!/usr/bin/env python3
"""Pure contracts for the train-only LoRA-ES sigma discrimination V53 run."""

from __future__ import annotations

import math

import lora_es_nested_population_v52 as v52


SIGMAS_V53 = (0.0012, 0.0024, 0.0048)
POPULATION_SIZE_V53 = 16
SEEDS_V53 = v52.P16_SEEDS_V52
ACTORS_V53 = v52.ACTORS_V52
MINIMUM_RELIABILITY_V53 = 0.8
MINIMUM_SPLIT_HALF_SPEARMAN_V53 = 0.7
HISTORICAL_CALIBRATION_CEILING_V53 = 0.001802103667415178


def canonical_sha256_v53(value: object) -> str:
    return v52.canonical_sha256_v52(value)


def state_derivations_v53(sigma: float) -> list[dict]:
    sigma = float(sigma)
    if sigma not in SIGMAS_V53:
        raise ValueError("v53 sigma is not preregistered")
    result = []
    for direction, seed in enumerate(SEEDS_V53):
        for label, sign in (("plus", 1), ("minus", -1)):
            result.append({
                "state_index": len(result),
                "direction": direction,
                "label": label,
                "sign": sign,
                "seed": seed,
                "sigma": sigma,
                "master_sha256": v52.MASTER_SHA256_V52,
                "derivation": "V41A antithetic candidate from pinned FP32 master",
                "candidate_identity_policy": (
                    "four-actor GPU-runtime consensus required before scoring"
                ),
                "runtime_identity_policy": (
                    "four-actor BF16-runtime consensus required before scoring"
                ),
            })
    return result


def arm_contracts_v53() -> list[dict]:
    return [{
        "arm_index": index,
        "sigma": sigma,
        "population_size": POPULATION_SIZE_V53,
        "seeds": list(SEEDS_V53),
        "state_derivations": state_derivations_v53(sigma),
        "state_derivation_inventory_sha256": canonical_sha256_v53(
            state_derivations_v53(sigma)
        ),
    } for index, sigma in enumerate(SIGMAS_V53)]


def select_smallest_passing_sigma_v53(receipts: list[dict]) -> float | None:
    if not 0 <= len(receipts) <= len(SIGMAS_V53):
        raise ValueError("v53 sigma receipt coverage changed")
    for index, receipt in enumerate(receipts):
        if (
            receipt.get("sigma") != SIGMAS_V53[index]
            or receipt.get("population_size") != POPULATION_SIZE_V53
            or not isinstance(receipt.get("passed"), bool)
        ):
            raise ValueError("v53 sigma receipt order or schema changed")
        if receipt["passed"]:
            if index != len(receipts) - 1:
                raise ValueError("v53 evaluated an arm after the first pass")
            return SIGMAS_V53[index]
    return None


def reliability_gate_v53(
    central_replicates: list[list[float]], fresh_maximum: float,
) -> dict:
    result = v52.reliability_gate_v52(central_replicates, fresh_maximum)
    if (
        result["population_size"] != POPULATION_SIZE_V53
        or result["minimum_reliability"] != MINIMUM_RELIABILITY_V53
        or result["minimum_split_half_spearman"]
        != MINIMUM_SPLIT_HALF_SPEARMAN_V53
        or result["historical_calibration_ceiling"]
        != HISTORICAL_CALIBRATION_CEILING_V53
        or result["passed"] is not bool(
            result["reliability"] >= MINIMUM_RELIABILITY_V53
            and result["split_half_spearman"]
            >= MINIMUM_SPLIT_HALF_SPEARMAN_V53
            and result["fresh_calibration_inside_historical_ceiling"]
            and result[
                "estimated_signal_standard_deviation_clears_fresh_"
                "calibration_maximum"
            ]
        )
        or not math.isfinite(result["estimated_signal_standard_deviation"])
    ):
        raise RuntimeError("v53 inherited reliability contract changed")
    result = dict(result)
    result["schema"] = "sigma-discrimination-reliability-v53"
    result["content_sha256"] = canonical_sha256_v53({
        key: value for key, value in result.items() if key != "content_sha256"
    })
    return result

