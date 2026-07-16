#!/usr/bin/env python3
"""Deterministic six-objective maximin projection over sealed V53 scores."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import lora_es_nested_population_v52 as v52


ROOT = Path(__file__).resolve().parent
V53_SELECTED_ARM = ROOT / "experiments/eggroll_es_hpo/runs/v53_lora_es_sigma_discrimination/sigma_0p0048_arm_v53.json"
V54_PROJECTION = ROOT / "experiments/eggroll_es_hpo/runs/v54_lora_es_selected_score_backtracking/nested_population_v52.json"
V53_SELECTED_ARM_FILE_SHA256 = "f73adf79eb1cc10315a088568cd68816a5233fdb46217bb5ed55e3c0be778659"
V53_SELECTED_ARM_CONTENT_SHA256 = "f60eb4f5c66a3f26571c07111ec1ccf9c26ecd8121e3c160007c5ad7220ae486"
V54_PROJECTION_FILE_SHA256 = "74c7cffc2b47966a67a440ec63948970999dd5f6231b9dcc480cbc51c6a6c0e4"
V54_PROJECTION_CONTENT_SHA256 = "7cb96e4e32a71417afd1a3c36af647342e914cef1ea211f0aedc5e43a12066cc"
OBJECTIVE_PATHS_V55B = {
    "domain": ("domain", "aggregate", "equal_unit_mean"),
    "fragile_generation_f1": (
        "fragile_generation", "equal_conflict_unit_mean_f1",
    ),
    "fragile_generation_nonzero": (
        "fragile_generation", "nonzero_count",
    ),
    "prose_lm": ("prose_lm", "mean_token_logprob"),
    "qa_answer_logprob": (
        "qa_answer_logprob", "mean_example_logprob",
    ),
    "qa_generation_f1": ("qa_generation", "mean_f1"),
}
ZERO_SPREAD_ENDPOINT_PATHS_V55B = {
    "fragile_generation_exact": ("fragile_generation", "exact_count"),
    "qa_generation_exact": ("qa_generation", "exact_count"),
    "qa_generation_nonzero": ("qa_generation", "nonzero_count"),
}
GRAM_RANK_V55B = 6
GRAM_CONDITION_CEILING_V55B = 10.0
MINIMUM_DUAL_LAMBDA_V55B = 0.1
MINIMUM_MAXIMIN_MARGIN_V55B = 0.3
EQUALITY_RESIDUAL_TOLERANCE_V55B = 1e-12
SCALE_ORDER_V55B = v52.SCALE_ORDER_V52


def _nested(value: dict, path: tuple[str, ...]) -> float:
    for key in path:
        value = value[key]
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("v55b objective is non-finite")
    return result


def _objective(arm: dict, path: tuple[str, ...]) -> dict:
    sign_scores = {
        sign: [[
            _nested(arm["signed_scores"][sign][direction][actor], path)
            for actor in range(v52.ACTORS_V52)
        ] for direction in range(16)]
        for sign in ("plus", "minus")
    }
    return v52.objective_coefficients_v52(sign_scores)


def _load_self_hashed(path: Path, file_sha: str, content_sha: str) -> dict:
    if v52.file_sha256_v52(path) != file_sha:
        raise RuntimeError(f"v55b sealed input file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or v52.canonical_sha256_v52(compact) != content_sha
    ):
        raise RuntimeError(f"v55b sealed input content changed: {path}")
    return value


def maximin_projection_v55b() -> dict:
    """Return a full-rank positive-dual maximin/KKT certificate."""
    arm = _load_self_hashed(
        V53_SELECTED_ARM, V53_SELECTED_ARM_FILE_SHA256,
        V53_SELECTED_ARM_CONTENT_SHA256,
    )
    source = _load_self_hashed(
        V54_PROJECTION, V54_PROJECTION_FILE_SHA256,
        V54_PROJECTION_CONTENT_SHA256,
    )
    objectives = {
        name: _objective(arm, path)
        for name, path in OBJECTIVE_PATHS_V55B.items()
    }
    zero_spread = {
        name: _objective(arm, path)
        for name, path in ZERO_SPREAD_ENDPOINT_PATHS_V55B.items()
    }
    if (
        any(item["zero_spread"] for item in objectives.values())
        or any(not item["zero_spread"] for item in zero_spread.values())
    ):
        raise RuntimeError("v55b signed-score spread inventory changed")
    names = list(OBJECTIVE_PATHS_V55B)
    if names != sorted(names):
        raise RuntimeError("v55b objective order is not canonical")
    rows = np.stack([
        np.asarray(objectives[name]["coefficients"], dtype=np.float64)
        for name in names
    ])
    norms = np.linalg.norm(rows, axis=1)
    normalized = rows / norms[:, None]
    gram = normalized @ normalized.T
    rank = int(np.linalg.matrix_rank(gram))
    condition = float(np.linalg.cond(gram))
    dual_lambda = np.linalg.solve(
        gram, np.ones(len(names), dtype=np.float64),
    )
    raw_direction = normalized.T @ dual_lambda
    raw_norm = float(np.linalg.norm(raw_direction))
    direction = raw_direction / raw_norm
    margins = normalized @ direction
    margin = float(1.0 / math.sqrt(float(np.sum(dual_lambda))))
    maximum_residual = float(np.max(np.abs(margins - margin)))
    reference = source["arms"]["p16"]["projection"]["projection"][
        "diagnostics"
    ]["unconstrained_domain_norm"]
    reference = float(reference)
    old = np.asarray(
        source["arms"]["p16"]["projection"]["coefficients"],
        dtype=np.float64,
    )
    direction_cosine_vs_v54 = float(
        np.dot(direction, old) / np.linalg.norm(old)
    )
    if (
        rank != GRAM_RANK_V55B
        or condition > GRAM_CONDITION_CEILING_V55B
        or float(np.min(dual_lambda)) <= MINIMUM_DUAL_LAMBDA_V55B
        or margin <= MINIMUM_MAXIMIN_MARGIN_V55B
        or maximum_residual > EQUALITY_RESIDUAL_TOLERANCE_V55B
        or not math.isclose(
            float(np.linalg.norm(direction)), 1.0,
            rel_tol=0.0, abs_tol=EQUALITY_RESIDUAL_TOLERANCE_V55B,
        )
        or not np.all(margins > MINIMUM_MAXIMIN_MARGIN_V55B)
        or not math.isclose(
            raw_norm, math.sqrt(float(np.sum(dual_lambda))),
            rel_tol=0.0, abs_tol=EQUALITY_RESIDUAL_TOLERANCE_V55B,
        )
        or not math.isfinite(reference) or reference <= 0.0
    ):
        raise RuntimeError("v55b maximin/KKT certificate failed closed")
    result = {
        "schema": "six-objective-maximin-kkt-projection-v55b",
        "population_size": 16,
        "sigma": 0.0048,
        "seeds": list(v52.P16_SEEDS_V52),
        "objective_order": names,
        "objective_fitness": objectives,
        "zero_spread_endpoint_only_metrics": zero_spread,
        "normalized_objective_matrix": normalized.tolist(),
        "normalized_objective_matrix_sha256": v52.canonical_sha256_v52(
            normalized.tolist()
        ),
        "gram_matrix": gram.tolist(),
        "gram_matrix_sha256": v52.canonical_sha256_v52(gram.tolist()),
        "gram_rank": rank,
        "gram_condition": condition,
        "gram_condition_ceiling": GRAM_CONDITION_CEILING_V55B,
        "dual_lambda": dual_lambda.tolist(),
        "minimum_dual_lambda": float(np.min(dual_lambda)),
        "required_minimum_dual_lambda": MINIMUM_DUAL_LAMBDA_V55B,
        "direction": direction.tolist(),
        "direction_sha256": v52.canonical_sha256_v52(direction.tolist()),
        "objective_margins": {
            name: float(value) for name, value in zip(
                names, margins.tolist(), strict=True,
            )
        },
        "maximin_margin": margin,
        "minimum_maximin_margin": MINIMUM_MAXIMIN_MARGIN_V55B,
        "maximum_equal_margin_residual": maximum_residual,
        "equal_margin_residual_tolerance": EQUALITY_RESIDUAL_TOLERANCE_V55B,
        "kkt_certificate": {
            "closed_form": "lambda=solve(A*A.T,ones); u=A.T*lambda/||A.T*lambda||",
            "full_row_rank": True,
            "strictly_positive_dual": True,
            "all_six_constraints_active_at_equal_positive_margin": True,
            "primal_unit_norm": True,
            "strong_duality_verified": True,
        },
        "reference_norm": reference,
        "reference_norm_contract": (
            "exact V54 unconstrained primary norm; V55B changes direction "
            "while matching every V54 scale's coefficient norm"
        ),
        "direction_cosine_vs_v54": direction_cosine_vs_v54,
        "genuinely_different_from_v54": direction_cosine_vs_v54 < 0.99,
    }
    result["content_sha256"] = v52.canonical_sha256_v52(result)
    return result


def scale_plans_v55b(projection: dict) -> list[dict]:
    if projection != maximin_projection_v55b():
        raise RuntimeError("v55b projection input changed")
    direction = np.asarray(projection["direction"], dtype=np.float64)
    reference = float(projection["reference_norm"])
    source = _load_self_hashed(
        V54_PROJECTION, V54_PROJECTION_FILE_SHA256,
        V54_PROJECTION_CONTENT_SHA256,
    )
    v54_plans = source["arms"]["p16"]["scale_plans"]
    if [item["target_norm_ratio"] for item in v54_plans] != list(
        SCALE_ORDER_V55B
    ):
        raise RuntimeError("v55b V54 reference scale order changed")
    result = []
    for ratio, v54_plan in zip(
        SCALE_ORDER_V55B, v54_plans, strict=True,
    ):
        coefficients = (direction * reference * ratio).tolist()
        actual = float(np.linalg.norm(coefficients) / reference)
        new_norm = float(np.linalg.norm(coefficients))
        v54_norm = float(np.linalg.norm(np.asarray(
            v54_plan["coefficients"], dtype=np.float64,
        )))
        if (
            not math.isclose(actual, ratio, rel_tol=0.0, abs_tol=1e-12)
            or not math.isclose(
                new_norm, v54_norm, rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise RuntimeError("v55b scale norm changed")
        result.append({
            "target_norm_ratio": ratio,
            "coefficients": coefficients,
            "coefficient_sha256": v52.canonical_sha256_v52({
                "seeds": list(v52.P16_SEEDS_V52),
                "coefficients": coefficients,
            }),
            "actual_norm_ratio": actual,
            "coefficient_l2_norm": new_norm,
            "v54_reference_coefficient_l2_norm": v54_norm,
            "v54_reference_coefficient_sha256": v54_plan[
                "coefficient_sha256"
            ],
            "coefficient_l2_norm_exactly_matches_v54_same_ratio": True,
            "six_strict_interior_objective_margins_preserved_by_positive_"
            "scaling": True,
            "all_nine_endpoint_gates_unchanged": True,
        })
    return result
