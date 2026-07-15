#!/usr/bin/env python3
"""Synthetic train-only tests for V21A panel and estimator mechanics."""

import copy

import numpy as np
import pytest

import train_eggroll_es_production_v331_patch_v21a as mechanics_v21a


@pytest.fixture(scope="module")
def panel_bundle_v21a():
    return mechanics_v21a.load_panel_bundle_v21a()


@pytest.fixture(scope="module")
def synthetic_unit_scores_v21a():
    result = {}
    for arm_index, arm in enumerate(mechanics_v21a.ARMS_V21A):
        shape = (
            10, 2, 64,
            mechanics_v21a.frame_v21a.ARM_REQUESTS_PER_PANEL_V21A[arm],
        )
        result[arm] = np.random.default_rng(20260824 + arm_index).normal(
            size=shape
        )
    return result


@pytest.fixture(scope="module")
def bootstrap_v21a(synthetic_unit_scores_v21a, panel_bundle_v21a):
    return mechanics_v21a.paired_stratified_bootstrap_v21a(
        synthetic_unit_scores_v21a, panel_bundle_v21a, chunk_size=257
    )


def test_v21a_materializes_exact_full_merge_without_replacing_production(
    panel_bundle_v21a,
):
    bundle = panel_bundle_v21a
    assert bundle == mechanics_v21a.load_panel_bundle_v21a()
    assert bundle["contains_evaluation_content"] is False
    assert bundle["persisted_to_disk"] is False
    assert bundle["token_boundary_audit_required_before_any_scoring"] is True
    for panel in mechanics_v21a.PANEL_NAMES_V21A:
        production = bundle["panels"][panel]["arms"]["production_only"]
        merged = bundle["panels"][panel]["arms"][
            "production_plus_v331_patch"
        ]
        assert len(production["joint_ids"]) == 24
        assert len(merged["joint_ids"]) == 30
        assert merged["joint_ids"][:24] == production["joint_ids"]
        assert merged["row_sha256"][:24] == production["row_sha256"]
        assert merged["representative_sides"] == ["production"] * 24 + [
            "candidate"
        ] * 6


def test_v21a_materialized_ht_strata_have_exact_weights(panel_bundle_v21a):
    for panel in mechanics_v21a.PANEL_NAMES_V21A:
        for arm in mechanics_v21a.ARMS_V21A:
            batch = panel_bundle_v21a["panels"][panel]["arms"][arm]
            assert sum(batch["weights"]) == pytest.approx(
                mechanics_v21a.frame_v21a.ARM_POPULATIONS_V21A[arm], abs=1e-12
            )
            counts = {
                name: batch["ht_strata"].count(name)
                for name in mechanics_v21a.BASE_CATEGORIES_V21A
            }
            assert counts == {name: 6 for name in mechanics_v21a.BASE_CATEGORIES_V21A}
            candidate = [
                value for value in batch["ht_strata"]
                if value.startswith("candidate_only_topic:")
            ]
            assert len(candidate) == (0 if arm == "production_only" else 6)


def test_v21a_observed_estimator_has_two_arms_ten_panels_and_64_directions(
    synthetic_unit_scores_v21a, panel_bundle_v21a,
):
    panel_scores = mechanics_v21a.observed_panel_scores_v21a(
        synthetic_unit_scores_v21a, panel_bundle_v21a
    )
    assert panel_scores.shape == (2, 10, 2, 64)
    observed = mechanics_v21a.recompute_observed_endpoints_v21a(
        synthetic_unit_scores_v21a, panel_bundle_v21a
    )
    assert set(observed["arms"]) == set(mechanics_v21a.ARMS_V21A)
    for arm in mechanics_v21a.ARMS_V21A:
        assert set(observed["arms"][arm]["endpoint_values"]) == set(
            mechanics_v21a.ENDPOINT_NAMES_V21A
        )
        assert observed["arms"][arm]["all_panel_spreads_nonzero"] is True


def test_v21a_paired_bootstrap_is_exact_50k_one_contrast_twelve_endpoints(
    bootstrap_v21a,
):
    assert bootstrap_v21a["repetitions"] == 50_000
    assert bootstrap_v21a["one_sided_quantile"] == 0.05 / 12
    assert bootstrap_v21a["quantile_method"] == "linear"
    assert bootstrap_v21a["paired_same_draws_both_arms"] is True
    assert bootstrap_v21a["whole_panel_block_resampling_used"] is False
    comparison = bootstrap_v21a["comparison"]
    assert comparison["treatment"] == "production_plus_v331_patch"
    assert comparison["control"] == "production_only"
    assert set(comparison["endpoints"]) == set(mechanics_v21a.ENDPOINT_NAMES_V21A)
    for item in comparison["endpoints"].values():
        assert set(item) == {
            "treatment_minus_control", "familywise_lcb", "noninferiority_margin"
        }
        assert item["noninferiority_margin"] == 0.0


def test_v21a_bootstrap_is_chunk_invariant(
    synthetic_unit_scores_v21a, panel_bundle_v21a, bootstrap_v21a,
):
    second = mechanics_v21a.paired_stratified_bootstrap_v21a(
        synthetic_unit_scores_v21a, panel_bundle_v21a, chunk_size=503
    )
    assert second == bootstrap_v21a


def _summary(bootstrap, passed_integrity=True):
    return {
        "arms": {
            arm: {
                "all_panel_spreads_nonzero": True,
                "endpoint_values": {
                    name: 0.0 for name in mechanics_v21a.ENDPOINT_NAMES_V21A
                },
            }
            for arm in mechanics_v21a.ARMS_V21A
        },
        "paired_bootstrap": copy.deepcopy(bootstrap),
        "runtime_integrity": {
            "all_integrity_audits_passed": passed_integrity,
        },
    }


def test_v21a_gate_is_zero_margin_conjunctive_and_never_authorizes_mutation(
    bootstrap_v21a,
):
    passing_bootstrap = copy.deepcopy(bootstrap_v21a)
    for item in passing_bootstrap["comparison"]["endpoints"].values():
        item["treatment_minus_control"] = 0.0
        item["familywise_lcb"] = 0.0
    gate = mechanics_v21a.evaluate_compatibility_gate_v21a(
        _summary(passing_bootstrap)
    )
    assert gate["compatibility_gate_passed"] is True
    assert gate["observed_pass_count"] == gate["bootstrap_pass_count"] == 12
    assert gate["dataset_promotion_authorized"] is False
    assert gate["model_update_authorized"] is False
    assert gate["evaluation_authorized"] is False
    failing = copy.deepcopy(passing_bootstrap)
    first = next(iter(failing["comparison"]["endpoints"].values()))
    first["familywise_lcb"] = -np.finfo(np.float64).eps
    assert mechanics_v21a.evaluate_compatibility_gate_v21a(
        _summary(failing)
    )["compatibility_gate_passed"] is False


def test_v21a_rejects_shapes_panel_tampering_and_gate_coverage(
    synthetic_unit_scores_v21a, panel_bundle_v21a, bootstrap_v21a,
):
    wrong = copy.deepcopy(synthetic_unit_scores_v21a)
    wrong["production_only"] = wrong["production_only"][..., :-1]
    with pytest.raises(RuntimeError, match="score tensor changed"):
        mechanics_v21a.observed_panel_scores_v21a(wrong, panel_bundle_v21a)
    panel = copy.deepcopy(panel_bundle_v21a)
    panel["panels"]["optimization_0"]["arms"]["production_only"][
        "questions"
    ].pop()
    with pytest.raises(RuntimeError, match="panel batch changed"):
        mechanics_v21a.validate_panel_bundle_v21a(panel)
    incomplete = copy.deepcopy(bootstrap_v21a)
    incomplete["comparison"]["endpoints"].pop(next(iter(
        incomplete["comparison"]["endpoints"]
    )))
    with pytest.raises(RuntimeError, match="gate input coverage changed"):
        mechanics_v21a.evaluate_compatibility_gate_v21a(_summary(incomplete))


@pytest.mark.parametrize("tamper", (
    "positive_infinity", "nonzero_margin", "comparison_name",
    "quantile_method", "draw_plan", "endpoint_extra_key", "delta_mismatch",
))
def test_v21a_gate_rejects_exact_contract_tampering(bootstrap_v21a, tamper):
    value = copy.deepcopy(bootstrap_v21a)
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
    with pytest.raises(RuntimeError, match="gate input coverage changed"):
        mechanics_v21a.evaluate_compatibility_gate_v21a(_summary(value))
