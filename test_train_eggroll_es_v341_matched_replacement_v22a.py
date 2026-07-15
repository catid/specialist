#!/usr/bin/env python3
"""Synthetic train-only tests for V22A panel and estimator mechanics."""

import copy

import numpy as np
import pytest

import train_eggroll_es_v341_matched_replacement_v22a as mechanics_v22a


@pytest.fixture(scope="module")
def panel_bundle_v22a():
    return mechanics_v22a.load_panel_bundle_v22a()


@pytest.fixture(scope="module")
def synthetic_unit_scores_v22a():
    return {
        arm: np.random.default_rng(20260826 + arm_index).normal(
            size=(10, 2, 64, 24)
        )
        for arm_index, arm in enumerate(mechanics_v22a.ARMS_V22A)
    }


@pytest.fixture(scope="module")
def bootstrap_v22a(synthetic_unit_scores_v22a, panel_bundle_v22a):
    return mechanics_v22a.paired_stratified_bootstrap_v22a(
        synthetic_unit_scores_v22a, panel_bundle_v22a, chunk_size=257
    )


def test_v22a_materializes_exact_matched_replacement_without_candidate_only(
    panel_bundle_v22a,
):
    bundle = panel_bundle_v22a
    assert bundle == mechanics_v22a.load_panel_bundle_v22a()
    assert bundle["contains_evaluation_content"] is False
    assert bundle["persisted_to_disk"] is False
    assert bundle["replacement_counts"] == {
        "sampled_candidate_representatives": 184,
        "sampled_production_representatives_unchanged": 56,
        "candidate_only_components_included": 0,
    }
    for panel in mechanics_v22a.PANEL_NAMES_V22A:
        control = bundle["panels"][panel]["arms"]["production_control"]
        treatment = bundle["panels"][panel]["arms"]["v341_matched_replacement"]
        expected = mechanics_v22a.frame_v22a.EXPECTED_REPLACEMENTS_BY_PANEL_V22A[
            panel
        ]
        assert len(control["joint_ids"]) == len(treatment["joint_ids"]) == 24
        assert treatment["joint_ids"] == control["joint_ids"]
        assert treatment["ht_strata"] == control["ht_strata"]
        assert treatment["weights"] == control["weights"]
        assert control["representative_sides"] == ["production"] * 24
        assert treatment["representative_sides"].count("candidate") == expected
        assert treatment["representative_sides"].count("production") == 24 - expected


def test_v22a_materialized_ht_strata_and_denominators_are_identical(panel_bundle_v22a):
    for panel in mechanics_v22a.PANEL_NAMES_V22A:
        for arm in mechanics_v22a.ARMS_V22A:
            batch = panel_bundle_v22a["panels"][panel]["arms"][arm]
            assert sum(batch["weights"]) == pytest.approx(272.0, abs=1e-12)
            assert {
                name: batch["ht_strata"].count(name)
                for name in mechanics_v22a.BASE_CATEGORIES_V22A
            } == {name: 6 for name in mechanics_v22a.BASE_CATEGORIES_V22A}
            assert not any(
                value.startswith("candidate_only") for value in batch["ht_strata"]
            )


def test_v22a_observed_estimator_has_two_equal_arms_ten_panels_64_directions(
    synthetic_unit_scores_v22a, panel_bundle_v22a,
):
    panel_scores = mechanics_v22a.observed_panel_scores_v22a(
        synthetic_unit_scores_v22a, panel_bundle_v22a
    )
    assert panel_scores.shape == (2, 10, 2, 64)
    observed = mechanics_v22a.recompute_observed_endpoints_v22a(
        synthetic_unit_scores_v22a, panel_bundle_v22a
    )
    assert set(observed["arms"]) == set(mechanics_v22a.ARMS_V22A)
    for arm in mechanics_v22a.ARMS_V22A:
        assert set(observed["arms"][arm]["endpoint_values"]) == set(
            mechanics_v22a.ENDPOINT_NAMES_V22A
        )
        assert observed["arms"][arm]["all_panel_spreads_nonzero"] is True


def test_v22a_bootstrap_is_exact_paired_50k_one_contrast_twelve_endpoints(
    bootstrap_v22a,
):
    assert bootstrap_v22a["repetitions"] == 50_000
    assert bootstrap_v22a["one_sided_quantile"] == 0.05 / 12
    assert bootstrap_v22a["quantile_method"] == "linear"
    assert bootstrap_v22a["paired_same_draws_both_arms"] is True
    assert bootstrap_v22a[
        "same_ht_coefficients_and_denominator_both_arms"
    ] is True
    assert bootstrap_v22a["candidate_only_resampling_present"] is False
    comparison = bootstrap_v22a["comparison"]
    assert comparison["treatment"] == "v341_matched_replacement"
    assert comparison["control"] == "production_control"
    assert set(comparison["endpoints"]) == set(mechanics_v22a.ENDPOINT_NAMES_V22A)
    for item in comparison["endpoints"].values():
        assert set(item) == {
            "treatment_minus_control", "familywise_lcb", "noninferiority_margin",
        }
        assert item["noninferiority_margin"] == 0.0


def test_v22a_bootstrap_is_chunk_invariant(
    synthetic_unit_scores_v22a, panel_bundle_v22a, bootstrap_v22a,
):
    second = mechanics_v22a.paired_stratified_bootstrap_v22a(
        synthetic_unit_scores_v22a, panel_bundle_v22a, chunk_size=503
    )
    assert second == bootstrap_v22a


def _runtime_integrity(passed=True):
    value = {
        key: True for key in mechanics_v22a.RUNTIME_INTEGRITY_KEYS_V22A
    }
    value["union_planner_called"] = False
    if not passed:
        value["pre_post_raw_reference_probes_equal"] = False
        value["all_integrity_audits_passed"] = False
    return value


def _summary(bootstrap, passed_integrity=True):
    return {
        "arms": {
            arm: {
                "all_panel_spreads_nonzero": True,
                "endpoint_values": {
                    name: 0.0 for name in mechanics_v22a.ENDPOINT_NAMES_V22A
                },
            }
            for arm in mechanics_v22a.ARMS_V22A
        },
        "paired_bootstrap": copy.deepcopy(bootstrap),
        "runtime_integrity": _runtime_integrity(passed_integrity),
    }


def test_v22a_gate_is_exact_zero_margin_conjunctive_and_never_mutates(bootstrap_v22a):
    passing = copy.deepcopy(bootstrap_v22a)
    for item in passing["comparison"]["endpoints"].values():
        item["treatment_minus_control"] = 0.0
        item["familywise_lcb"] = 0.0
    gate = mechanics_v22a.evaluate_compatibility_gate_v22a(_summary(passing))
    assert gate["compatibility_gate_passed"] is True
    assert gate["observed_pass_count"] == gate["bootstrap_pass_count"] == 12
    assert gate["dataset_promotion_authorized"] is False
    assert gate["model_update_authorized"] is False
    assert gate["evaluation_authorized"] is False
    failing = copy.deepcopy(passing)
    next(iter(failing["comparison"]["endpoints"].values()))[
        "familywise_lcb"
    ] = -np.finfo(np.float64).eps
    assert mechanics_v22a.evaluate_compatibility_gate_v22a(
        _summary(failing)
    )["compatibility_gate_passed"] is False
    integrity_failure = mechanics_v22a.evaluate_compatibility_gate_v22a(
        _summary(passing, passed_integrity=False)
    )
    assert integrity_failure["compatibility_gate_passed"] is False
    assert integrity_failure["all_runtime_integrity_audits_passed"] is False


def test_v22a_rejects_score_shapes_panel_pairing_and_gate_coverage(
    synthetic_unit_scores_v22a, panel_bundle_v22a, bootstrap_v22a,
):
    wrong = copy.deepcopy(synthetic_unit_scores_v22a)
    wrong["production_control"] = wrong["production_control"][..., :-1]
    with pytest.raises(RuntimeError, match="score tensor changed"):
        mechanics_v22a.observed_panel_scores_v22a(wrong, panel_bundle_v22a)
    panel = copy.deepcopy(panel_bundle_v22a)
    panel["panels"]["optimization_0"]["arms"]["production_control"][
        "questions"
    ].pop()
    with pytest.raises(RuntimeError, match="panel batch changed"):
        mechanics_v22a.validate_panel_bundle_v22a(panel)
    incomplete = copy.deepcopy(bootstrap_v22a)
    incomplete["comparison"]["endpoints"].pop(next(iter(
        incomplete["comparison"]["endpoints"]
    )))
    with pytest.raises(RuntimeError, match="gate input coverage changed"):
        mechanics_v22a.evaluate_compatibility_gate_v22a(_summary(incomplete))


@pytest.mark.parametrize(
    "tamper",
    (
        "positive_infinity", "nonzero_margin", "comparison_name",
        "quantile_method", "draw_plan", "endpoint_extra_key", "delta_mismatch",
        "candidate_resampling", "integrity_extra_key",
    ),
)
def test_v22a_gate_rejects_exact_contract_tampering(bootstrap_v22a, tamper):
    value = copy.deepcopy(bootstrap_v22a)
    for item in value["comparison"]["endpoints"].values():
        item["treatment_minus_control"] = 0.0
        item["familywise_lcb"] = 0.0
    first = next(iter(value["comparison"]["endpoints"].values()))
    if tamper == "positive_infinity":
        first["familywise_lcb"] = float("inf")
    elif tamper == "nonzero_margin":
        first["noninferiority_margin"] = np.finfo(np.float64).eps
    elif tamper == "comparison_name":
        value["comparison"]["name"] = "changed"
    elif tamper == "quantile_method":
        value["quantile_method"] = "nearest"
    elif tamper == "draw_plan":
        value["draw_plan_content_sha256"] = "0" * 64
    elif tamper == "endpoint_extra_key":
        first["extra"] = False
    elif tamper == "delta_mismatch":
        first["treatment_minus_control"] = np.finfo(np.float64).eps
    elif tamper == "candidate_resampling":
        value["candidate_only_resampling_present"] = True
    summary = _summary(value)
    if tamper == "integrity_extra_key":
        summary["runtime_integrity"]["extra"] = True
    with pytest.raises(RuntimeError, match="gate input coverage changed"):
        mechanics_v22a.evaluate_compatibility_gate_v22a(summary)
