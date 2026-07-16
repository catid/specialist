#!/usr/bin/env python3
"""CPU-only multi-anchor trust-region projection for LoRA EGGROLL-ES V43H.

Every objective is first converted to the same antithetic centered-rank seed
basis.  The domain vector is then projected onto the intersection of the
non-degrading half-spaces induced by all required training anchors.  Unlike a
sequence of one-anchor projections, the simultaneous projection cannot repair
one anchor by silently breaking another.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

import lora_es_robust_consensus_v43g as robust_v43g


POPULATION_SIZE_V43H = 8
SIGNED_REPLICATES_V43H = 4
TRUST_REGION_NORM_RATIO_V43H = 0.5
FEASIBILITY_TOLERANCE_V43H = 1e-12


def canonical_sha256_v43h(value: object) -> str:
    return robust_v43g.canonical_sha256_v43g(value)


def objective_coefficients_v43h(equal_unit_sign_scores: dict) -> dict:
    """Return robust EGGROLL utilities for one four-actor objective."""
    result = robust_v43g.robust_population_v43g(equal_unit_sign_scores)
    return {
        "schema": "multi-anchor-objective-centered-ranks-v43h",
        "robust_signed_scores": result["robust_signed_scores"],
        "signed_centered_rank_utilities": result[
            "signed_centered_rank_utilities"
        ],
        "coefficients": result["coefficients"],
        "zero_spread": result["zero_utility_update"],
    }


def _finite_vector(name: str, values, expected: int | None = None) -> np.ndarray:
    values = np.asarray([float(value) for value in values], dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"v43h {name} must be a one-dimensional population vector")
    if expected is not None and len(values) != expected:
        raise ValueError(f"v43h {name} population size changed")
    if not np.isfinite(values).all():
        raise ValueError(f"v43h {name} contains a non-finite coefficient")
    return values


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else None


def _project_halfspace_intersection(
    domain: np.ndarray, anchors: np.ndarray, tolerance: float,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Euclidean projection onto {x: anchors @ x >= 0} by active sets."""
    anchor_count = anchors.shape[0]
    candidates: list[tuple[float, tuple[int, ...], np.ndarray]] = []

    def consider(candidate: np.ndarray, active: tuple[int, ...]) -> None:
        if not np.isfinite(candidate).all():
            return
        if np.all(anchors @ candidate >= -tolerance):
            distance = float(np.dot(candidate - domain, candidate - domain))
            candidates.append((distance, active, candidate))

    consider(domain.copy(), ())
    # Zero is always feasible and makes the procedure fail closed even when
    # anchor gradients are duplicate, opposed, or rank deficient.
    consider(np.zeros_like(domain), tuple(range(anchor_count)))
    for size in range(1, anchor_count + 1):
        for active in itertools.combinations(range(anchor_count), size):
            selected = anchors[list(active)]
            gram = selected @ selected.T
            rhs = -(selected @ domain)
            multipliers, _, _, _ = np.linalg.lstsq(gram, rhs, rcond=None)
            if np.any(multipliers < -tolerance):
                continue
            candidate = domain + selected.T @ multipliers
            if np.max(np.abs(selected @ candidate), initial=0.0) > 1e-9:
                continue
            consider(candidate, active)
    if not candidates:
        raise RuntimeError("v43h anchor cone projection found no feasible point")
    # Active-index order breaks numerically exact distance ties deterministically.
    _, active, candidate = min(candidates, key=lambda item: (item[0], item[1]))
    return candidate, active


def project_multi_anchor_trust_region_v43h(
    domain_coefficients,
    anchor_coefficients: dict[str, list[float]],
    *,
    max_norm_ratio: float = TRUST_REGION_NORM_RATIO_V43H,
    tolerance: float = FEASIBILITY_TOLERANCE_V43H,
) -> dict:
    """Project one domain vector against every required anchor simultaneously.

    Constraints are first-order non-degradation constraints, ``c dot a >= 0``.
    The Euclidean cone projection minimizes the change from the domain update.
    A final homogeneous trust-region scaling caps the coefficient norm while
    preserving every constraint.  Any zero-spread required anchor fails closed.
    """
    domain = _finite_vector("domain", domain_coefficients)
    if not isinstance(anchor_coefficients, dict) or not anchor_coefficients:
        raise ValueError("v43h requires at least one named anchor")
    if tuple(sorted(anchor_coefficients)) != tuple(anchor_coefficients):
        raise ValueError("v43h anchor names must be in canonical sorted order")
    if (
        not math.isfinite(max_norm_ratio)
        or max_norm_ratio <= 0.0
        or max_norm_ratio > 1.0
    ):
        raise ValueError("v43h trust-region norm ratio must be in (0, 1]")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("v43h feasibility tolerance must be positive")

    anchors = {
        name: _finite_vector(name, values, expected=len(domain))
        for name, values in anchor_coefficients.items()
    }
    domain_norm = float(np.linalg.norm(domain))
    zero_anchors = [name for name, values in anchors.items() if np.linalg.norm(values) == 0.0]
    before = {name: _cosine(domain, values) for name, values in anchors.items()}
    if domain_norm == 0.0 or zero_anchors:
        coefficients = np.zeros_like(domain)
        decision = "skip_no_domain_spread" if domain_norm == 0.0 else "skip_required_anchor_no_spread"
        result = {
            "schema": "eggroll-es-multi-anchor-trust-region-v43h",
            "coefficients": coefficients.tolist(),
            "diagnostics": {
                "decision": decision,
                "anchor_order": list(anchors),
                "zero_spread_anchors": zero_anchors,
                "anchor_cosines_before": before,
                "anchor_cosines_after": {name: None for name in anchors},
                "anchor_directional_derivatives_after": {name: 0.0 for name in anchors},
                "active_anchor_indices": [],
                "active_anchor_names": [],
                "unconstrained_domain_norm": domain_norm,
                "projected_pretrust_norm": 0.0,
                "trust_region_norm_ratio": float(max_norm_ratio),
                "update_norm_ratio": 0.0,
                "all_anchor_halfspaces_satisfied": True,
            },
        }
        result["content_sha256"] = canonical_sha256_v43h(result)
        return result

    names = list(anchors)
    normalized = np.stack([
        anchors[name] / np.linalg.norm(anchors[name]) for name in names
    ])
    projected, active = _project_halfspace_intersection(
        domain, normalized, float(tolerance),
    )
    projected_norm = float(np.linalg.norm(projected))
    maximum_norm = float(max_norm_ratio) * domain_norm
    if projected_norm > maximum_norm:
        projected = projected * (maximum_norm / projected_norm)
    final_norm = float(np.linalg.norm(projected))
    derivatives = {
        name: float(np.dot(projected, values)) for name, values in anchors.items()
    }
    satisfied = all(value >= -float(tolerance) for value in derivatives.values())
    if not satisfied:
        raise RuntimeError("v43h trust-region scaling violated an anchor halfspace")
    result = {
        "schema": "eggroll-es-multi-anchor-trust-region-v43h",
        "coefficients": projected.tolist(),
        "diagnostics": {
            "decision": "project_and_trust_region",
            "anchor_order": names,
            "zero_spread_anchors": [],
            "anchor_cosines_before": before,
            "anchor_cosines_after": {
                name: _cosine(projected, values) for name, values in anchors.items()
            },
            "anchor_directional_derivatives_after": derivatives,
            "active_anchor_indices": list(active),
            "active_anchor_names": [names[index] for index in active],
            "unconstrained_domain_norm": domain_norm,
            "projected_pretrust_norm": projected_norm,
            "trust_region_norm_ratio": float(max_norm_ratio),
            "update_norm_ratio": final_norm / domain_norm,
            "all_anchor_halfspaces_satisfied": satisfied,
        },
    }
    result["content_sha256"] = canonical_sha256_v43h(result)
    return result
