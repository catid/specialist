"""Coefficient-space general-language constraint for EGGROLL-ES.

The upstream trainer represents an ES update as a linear combination of the
same seeded noise vectors used to evaluate the population.  A domain objective
and a teacher-forced language-model anchor can therefore be combined without
materializing either enormous gradient: their standardized population scores
are the coordinates of the two estimated gradients in that shared noise
basis.

This module deliberately contains no model or vLLM code.  It is the small,
CPU-only part of an anchored trainer and can be tested independently of a
hash-pinned experiment.
"""

import math


def _finite_scores(name, values):
    values = [float(value) for value in values]
    if len(values) < 2:
        raise ValueError(
            f"{name} scores require at least two population members"
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} scores must all be finite")
    return values


def _standardize(values, epsilon):
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    if std <= epsilon:
        return [0.0] * len(values), mean, std
    # Match upstream's stabilizer rather than silently changing its reward
    # shaping when this helper is integrated into the trainer.
    return [
        (value - mean) / (std + epsilon)
        for value in values
    ], mean, std


def _dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def _norm(values):
    return math.sqrt(_dot(values, values))


def project_anchor_safe_coefficients(
    domain_scores,
    anchor_scores,
    *,
    min_anchor_cosine=0.0,
    epsilon=1e-8,
):
    """Return ES coefficients constrained to a non-degrading anchor direction.

    Both score lists must be ordered by the same perturbation seeds and higher
    must be better.  ``anchor_scores`` should be token-weighted mean selected-
    token log probabilities on a *training-only* general-prose minibatch.

    Let ``d`` and ``a`` be the separately standardized domain and anchor score
    vectors.  If their cosine is below ``min_anchor_cosine``, this computes the
    minimum change ``d + lambda * a`` that reaches the requested cone (the
    non-degrading half-space when the requested cosine is zero).
    The result is never allowed a larger coefficient norm than ``d``.  It must
    be sent directly to ``update_weights_from_seeds``; standardizing it again
    would erase the deliberate reduction in an unsafe update's step length.

    A zero-spread anchor fails closed by returning a zero update, since the
    sampled population contains no evidence that any proposed direction
    preserves the anchor.  A zero-spread domain also returns a zero update,
    matching the effective behavior of upstream z-score shaping.
    """
    domain_scores = _finite_scores("domain", domain_scores)
    anchor_scores = _finite_scores("anchor", anchor_scores)
    if len(domain_scores) != len(anchor_scores):
        raise ValueError("domain and anchor score counts differ")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if (
        not math.isfinite(min_anchor_cosine)
        or min_anchor_cosine < 0.0
        or min_anchor_cosine >= 1.0
    ):
        raise ValueError("minimum anchor cosine must be in [0, 1)")

    domain, domain_mean, domain_std = _standardize(domain_scores, epsilon)
    anchor, anchor_mean, anchor_std = _standardize(anchor_scores, epsilon)
    domain_norm = _norm(domain)
    anchor_norm = _norm(anchor)

    diagnostics = {
        "schema": "eggroll-es-anchor-projection-v1",
        "population_size": len(domain),
        "domain_mean": domain_mean,
        "domain_std": domain_std,
        "anchor_mean": anchor_mean,
        "anchor_std": anchor_std,
        "min_anchor_cosine": float(min_anchor_cosine),
        "projection_lambda": 0.0,
        "domain_anchor_cosine_before": None,
        "domain_anchor_cosine_after": None,
        "update_norm_ratio": 0.0,
    }

    if domain_norm == 0.0:
        diagnostics["decision"] = "skip_no_domain_spread"
        return {
            "coefficients": [0.0] * len(domain),
            "diagnostics": diagnostics,
        }
    if anchor_norm == 0.0:
        diagnostics["decision"] = "skip_no_anchor_spread"
        return {
            "coefficients": [0.0] * len(domain),
            "diagnostics": diagnostics,
        }

    before_dot = _dot(domain, anchor)
    before_cosine = before_dot / (domain_norm * anchor_norm)
    diagnostics["domain_anchor_cosine_before"] = before_cosine
    if before_cosine >= min_anchor_cosine:
        coefficients = domain
        diagnostics["decision"] = "accept_domain_direction"
    else:
        # Decompose d into its component along the unit anchor direction and
        # an orthogonal component.  Adding lambda*a changes only the former.
        # To obtain cosine m, its required positive parallel magnitude is
        # m * ||d_perp|| / sqrt(1 - m**2).  This is a cone constraint for
        # m > 0, rather than the linear dot-product constraint used at m = 0.
        parallel = before_dot / anchor_norm
        perpendicular_squared = max(
            0.0, domain_norm * domain_norm - parallel * parallel,
        )
        perpendicular = math.sqrt(perpendicular_squared)
        target_parallel = (
            min_anchor_cosine * perpendicular
            / math.sqrt(1.0 - min_anchor_cosine * min_anchor_cosine)
        )
        projection_lambda = (target_parallel - parallel) / anchor_norm
        coefficients = [
            domain_value + projection_lambda * anchor_value
            for domain_value, anchor_value in zip(domain, anchor)
        ]
        diagnostics["projection_lambda"] = projection_lambda
        diagnostics["decision"] = "project_to_anchor_cone"

        # A positive requested margin can increase the raw vector norm.  Keep
        # alpha as an upper bound on update scale.  Uniform scaling preserves
        # the achieved cosine and hence the cone constraint.
        coefficient_norm = _norm(coefficients)
        if coefficient_norm > domain_norm:
            scale = domain_norm / coefficient_norm
            coefficients = [value * scale for value in coefficients]

    coefficient_norm = _norm(coefficients)
    if coefficient_norm > 0.0:
        after_cosine = _dot(coefficients, anchor) / (
            coefficient_norm * anchor_norm
        )
        # Roundoff around a tangent projection can produce a tiny negative
        # value; record the actual result and let tests enforce a tight bound.
        diagnostics["domain_anchor_cosine_after"] = after_cosine
    diagnostics["update_norm_ratio"] = coefficient_norm / domain_norm
    return {"coefficients": coefficients, "diagnostics": diagnostics}
