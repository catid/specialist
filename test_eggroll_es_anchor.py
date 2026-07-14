import math

import pytest

from eggroll_es_anchor import project_anchor_safe_coefficients


def dot(left, right):
    return math.fsum(a * b for a, b in zip(left, right))


def norm(values):
    return math.sqrt(dot(values, values))


def test_aligned_anchor_leaves_upstream_domain_coefficients_unchanged():
    result = project_anchor_safe_coefficients(
        [0.0, 1.0, 2.0, 3.0],
        [10.0, 11.0, 12.0, 13.0],
    )
    expected_mean = 1.5
    expected_std = math.sqrt(1.25)
    expected = [
        (value - expected_mean) / (expected_std + 1e-8)
        for value in [0.0, 1.0, 2.0, 3.0]
    ]
    assert result["coefficients"] == pytest.approx(expected)
    assert result["diagnostics"]["decision"] == "accept_domain_direction"
    assert result["diagnostics"]["update_norm_ratio"] == pytest.approx(1.0)


def test_conflicting_direction_is_projected_with_no_larger_step():
    scores = [0.0, 3.0, 1.0, 2.0, 4.0]
    anchors = [4.0, 0.0, 3.0, 1.0, 2.0]
    unconstrained = project_anchor_safe_coefficients(scores, scores)[
        "coefficients"
    ]
    anchor_direction = project_anchor_safe_coefficients(anchors, anchors)[
        "coefficients"
    ]
    result = project_anchor_safe_coefficients(
        scores, anchors, min_anchor_cosine=0.1,
    )
    coefficients = result["coefficients"]
    cosine = dot(coefficients, anchor_direction) / (
        norm(coefficients) * norm(anchor_direction)
    )
    assert result["diagnostics"]["decision"] == (
        "project_to_anchor_cone"
    )
    assert cosine >= 0.1 - 1e-12
    assert norm(coefficients) <= norm(unconstrained) + 1e-12
    assert math.fsum(coefficients) == pytest.approx(0.0, abs=1e-12)


def test_exact_opposition_at_zero_margin_safely_cancels_update():
    result = project_anchor_safe_coefficients(
        [0.0, 1.0, 2.0, 3.0],
        [3.0, 2.0, 1.0, 0.0],
        min_anchor_cosine=0.0,
    )
    assert result["coefficients"] == pytest.approx([0.0] * 4, abs=1e-12)
    assert result["diagnostics"]["update_norm_ratio"] == pytest.approx(
        0.0, abs=1e-12,
    )


def test_positive_cosine_constraint_uses_cone_not_only_dot_threshold():
    # These centered score vectors are orthogonal.  A linear dot threshold of
    # m*||d||*||a|| would produce cosine m/sqrt(1+m**2), which is too small.
    domain = [1.0, -1.0, 1.0, -1.0]
    anchor = [1.0, 1.0, -1.0, -1.0]
    anchor_direction = project_anchor_safe_coefficients(anchor, anchor)[
        "coefficients"
    ]
    result = project_anchor_safe_coefficients(
        domain, anchor, min_anchor_cosine=0.8,
    )
    coefficients = result["coefficients"]
    cosine = dot(coefficients, anchor_direction) / (
        norm(coefficients) * norm(anchor_direction)
    )
    assert cosine == pytest.approx(0.8)


def test_zero_spread_anchor_fails_closed():
    result = project_anchor_safe_coefficients(
        [0.0, 1.0, 2.0, 3.0],
        [5.0, 5.0, 5.0, 5.0],
    )
    assert result["coefficients"] == [0.0] * 4
    assert result["diagnostics"]["decision"] == "skip_no_anchor_spread"


@pytest.mark.parametrize(
    "domain,anchor,kwargs,message",
    [
        ([1.0], [1.0], {}, "at least two"),
        ([1.0, 2.0], [1.0, 2.0, 3.0], {}, "counts differ"),
        ([1.0, math.nan], [1.0, 2.0], {}, "finite"),
        ([1.0, 2.0], [1.0, 2.0], {"epsilon": 0.0}, "epsilon"),
        (
            [1.0, 2.0], [1.0, 2.0], {"min_anchor_cosine": 1.0},
            "cosine",
        ),
    ],
)
def test_invalid_projection_inputs_fail_closed(
    domain, anchor, kwargs, message,
):
    with pytest.raises(ValueError, match=message):
        project_anchor_safe_coefficients(domain, anchor, **kwargs)
