#!/usr/bin/env python3
"""Train-only conservative scale plans and actor-robust gate for V57."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

import lora_es_fragile_maximin_projection_v55b as v55b
import lora_es_nested_population_v52 as v52


SCALE_ORDER_V57 = (0.21875, 0.1875, 0.15625, 0.125, 0.09375, 0.0625)
ACTOR_ROBUST_METRICS_V57 = {
    "domain_point_improvement": ("domain", True),
    "prose_lm_noninferiority": ("prose_lm", False),
    "qa_logprob_noninferiority": ("qa_answer_logprob", False),
    "qa_generation_f1_noninferiority": ("qa_generation_f1", False),
    "qa_generation_exact_noninferiority": ("qa_generation_exact", False),
    "qa_generation_nonzero_noninferiority": (
        "qa_generation_nonzero", False,
    ),
}


def scale_plans_v57(projection: dict) -> list[dict]:
    """Scale the sealed V55B direction at preregistered sub-.25 ratios."""
    if projection != v55b.maximin_projection_v55b():
        raise RuntimeError("v57 projection input changed")
    direction = np.asarray(projection["direction"], dtype=np.float64)
    reference = float(projection["reference_norm"])
    result = []
    for ratio in SCALE_ORDER_V57:
        coefficients = (direction * reference * ratio).tolist()
        actual = float(np.linalg.norm(coefficients) / reference)
        if not math.isclose(actual, ratio, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("v57 scale norm changed")
        result.append({
            "target_norm_ratio": ratio,
            "coefficients": coefficients,
            "coefficient_sha256": v52.canonical_sha256_v52({
                "seeds": list(v52.P16_SEEDS_V52),
                "coefficients": coefficients,
            }),
            "actual_norm_ratio": actual,
            "coefficient_l2_norm": float(np.linalg.norm(coefficients)),
            "strictly_below_disqualified_v55b_ratio": ratio < 0.25,
            "six_strict_interior_objective_margins_preserved": True,
        })
    return result


def tighten_gate_v57(original_gate: Callable, *args, **kwargs) -> dict:
    """Require every actor's paired train delta to clear zero."""
    source = original_gate(*args, **kwargs)
    gate = dict(source)
    checks = dict(gate.get("checks", {}))
    if set(checks) != set(v52.TRAIN_GATE_NAMES_V52):
        raise RuntimeError("v57 inherited train gate inventory changed")
    metrics = gate.get("metrics", {})
    receipts = {}
    for check, (metric, strict) in ACTOR_ROBUST_METRICS_V57.items():
        deltas = metrics.get(metric, {}).get("paired_actor_deltas")
        if (
            not isinstance(deltas, list) or len(deltas) != v52.ACTORS_V52
            or any(not isinstance(value, (int, float)) for value in deltas)
        ):
            raise RuntimeError(f"v57 actor delta coverage changed: {metric}")
        actor_pass = all(
            value > 0.0 if strict else value >= 0.0 for value in deltas
        )
        inherited = checks[check] is True
        checks[check] = bool(inherited and actor_pass)
        receipts[check] = {
            "metric": metric,
            "paired_actor_deltas": deltas,
            "comparison": "> 0" if strict else ">= 0",
            "inherited_check_passed": inherited,
            "all_four_actor_deltas_passed": actor_pass,
            "passed": checks[check],
        }
    gate["checks"] = checks
    gate["passed"] = all(checks.values())
    gate["actor_robustness_v57"] = {
        "schema": "four-actor-paired-train-robustness-v57",
        "policy": (
            "all four paired actor deltas must be positive for domain and "
            "nonnegative for the five inherited noninferiority metrics"
        ),
        "receipts": receipts,
        "passed": all(item["passed"] for item in receipts.values()),
    }
    gate.pop("content_sha256", None)
    gate["content_sha256"] = v52.canonical_sha256_v52(gate)
    return gate
