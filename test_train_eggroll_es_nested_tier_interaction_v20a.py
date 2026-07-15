#!/usr/bin/env python3
"""Pure synthetic tests for V20A nested tier attribution mechanics."""

import copy

import numpy as np
import pytest

import train_eggroll_es_nested_tier_interaction_v20a as mechanics_v20a


@pytest.fixture(scope="module")
def panel_bundle_v20a():
    return mechanics_v20a.load_panel_bundle_v20a()


@pytest.fixture(scope="module")
def synthetic_unit_scores_v20a():
    result = {}
    for arm_index, arm in enumerate(mechanics_v20a.ARMS_V20A):
        shape = (
            10,
            2,
            32,
            mechanics_v20a.frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm],
        )
        result[arm] = np.random.default_rng(20260810 + arm_index).normal(
            size=shape
        )
    return result


@pytest.fixture(scope="module")
def bootstrap_v20a(synthetic_unit_scores_v20a, panel_bundle_v20a):
    return mechanics_v20a.paired_stratified_bootstrap_v20a(
        synthetic_unit_scores_v20a,
        panel_bundle_v20a,
        chunk_size=257,
    )


def _all_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _all_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_keys(item)


def test_v20a_rematerializes_exact_commit_bound_bundle(panel_bundle_v20a):
    bundle = panel_bundle_v20a
    assert bundle == mechanics_v20a.load_panel_bundle_v20a()
    assert bundle["content_sha256_before_self_field"] == (
        "bbf1d592799ba30e4506c4e5abe9851bc3673e619ed29f65d6054c8b150681bd"
    )
    assert bundle["preregistration"] == {
        "file_sha256": mechanics_v20a.PREREGISTRATION_FILE_SHA256_V20A,
        "content_sha256": mechanics_v20a.PREREGISTRATION_CONTENT_SHA256_V20A,
        "sealed_commit": mechanics_v20a.SEALED_FOUNDATION_COMMIT_V20A,
    }
    assert bundle["frame"] == {
        "file_sha256": mechanics_v20a.FRAME_CERTIFICATE_FILE_SHA256_V20A,
        "content_sha256": mechanics_v20a.FRAME_CERTIFICATE_CONTENT_SHA256_V20A,
        "sealed_commit": mechanics_v20a.SEALED_FOUNDATION_COMMIT_V20A,
    }
    assert bundle["draw_plan"]["content_sha256"] == (
        mechanics_v20a.DRAW_PLAN_CONTENT_SHA256_V20A
    )
    assert bundle["contains_evaluation_content"] is False
    assert bundle["persisted_to_disk"] is False


def test_v20a_bundle_has_exact_nested_substitution_and_no_duplicates(
    panel_bundle_v20a,
):
    expected_candidate_sides = {
        "production_only": 0,
        "patch_tier_2_only": 7,
        "patch_tiers_2_3": 14,
        "patch_all_tiers": 21,
    }
    for panel in mechanics_v20a.PANEL_NAMES_V20A:
        arms = panel_bundle_v20a["panels"][panel]["arms"]
        production_base = arms["production_only"]["joint_ids"]
        assert len(production_base) == len(set(production_base)) == 24
        for arm in mechanics_v20a.ARMS_V20A:
            batch = arms[arm]
            active_tiers = mechanics_v20a.frame_v20a.ARM_ACTIVE_TIERS_V20A[arm]
            assert batch["joint_ids"][:24] == production_base
            assert len(batch["joint_ids"]) == len(set(batch["joint_ids"]))
            assert len(batch["joint_ids"]) == (
                mechanics_v20a.frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm]
            )
            assert sum(batch["weights"]) == pytest.approx(
                mechanics_v20a.frame_v20a.ARM_POPULATIONS_V20A[arm],
                abs=1e-12,
            )
            assert batch["representative_sides"].count("candidate") == (
                expected_candidate_sides[arm]
            )
            assert [
                item for item in batch["ht_strata"]
                if item.startswith("candidate_only_tier_")
            ] == [f"candidate_only_tier_{tier}" for tier in active_tiers]


def test_v20a_ht_recomputation_uses_exact_nested_denominators(panel_bundle_v20a):
    stratum_values = {
        "paired_tier_1": 1.0,
        "paired_tier_2": 2.0,
        "paired_tier_3": 3.0,
        "fallback": 4.0,
        "candidate_only_tier_1": 5.0,
        "candidate_only_tier_2": 6.0,
        "candidate_only_tier_3": 7.0,
    }
    unit_scores = {}
    expected = {}
    for arm in mechanics_v20a.ARMS_V20A:
        requests = mechanics_v20a.frame_v20a.ARM_REQUESTS_PER_PANEL_V20A[arm]
        values = np.empty((10, 2, 32, requests), dtype=np.float64)
        arm_expected = []
        for panel_index, panel in enumerate(mechanics_v20a.PANEL_NAMES_V20A):
            batch = panel_bundle_v20a["panels"][panel]["arms"][arm]
            row_values = np.asarray([
                stratum_values[stratum] for stratum in batch["ht_strata"]
            ])
            values[panel_index] = row_values[np.newaxis, np.newaxis, :]
            arm_expected.append(
                np.dot(row_values, batch["weights"])
                / mechanics_v20a.frame_v20a.ARM_POPULATIONS_V20A[arm]
            )
        unit_scores[arm] = values
        expected[arm] = np.asarray(arm_expected)
    observed = mechanics_v20a.observed_panel_scores_v20a(
        unit_scores, panel_bundle_v20a
    )
    assert observed.shape == (4, 10, 2, 32)
    for arm_index, arm in enumerate(mechanics_v20a.ARMS_V20A):
        assert np.allclose(
            observed[arm_index],
            np.broadcast_to(expected[arm][:, None, None], (10, 2, 32)),
            rtol=0.0,
            atol=1e-15,
        )


def test_v20a_endpoint_geometry_is_six_four_and_twelve(
    synthetic_unit_scores_v20a,
    panel_bundle_v20a,
):
    panel_scores = mechanics_v20a.observed_panel_scores_v20a(
        synthetic_unit_scores_v20a, panel_bundle_v20a
    )
    analyzed = mechanics_v20a._endpoint_arrays_v20a(
        panel_scores[:, np.newaxis, ...]
    )
    assert analyzed["coefficients"].shape == (4, 1, 10, 32)
    assert analyzed["aggregate"].shape == (4, 1, 32)
    assert {name: values.shape[-1] for name, values in analyzed["families"].items()} == {
        "optimization_pairwise_cosine": 15,
        "optimization_pairwise_sign_agreement": 15,
        "aggregate_to_optimization_cosine": 6,
        "aggregate_to_optimization_sign_agreement": 6,
        "train_screen_cosine": 4,
        "train_screen_sign_agreement": 4,
    }
    observed = mechanics_v20a.recompute_observed_endpoints_v20a(
        synthetic_unit_scores_v20a, panel_bundle_v20a
    )
    assert set(observed["arms"]) == set(mechanics_v20a.ARMS_V20A)
    assert all(
        len(contract["endpoint_values"]) == 12
        and contract["all_panel_spreads_nonzero"] is True
        for contract in observed["arms"].values()
    )


def test_v20a_exact_50000_bootstrap_has_five_contrasts_and_60_bounds(
    bootstrap_v20a,
):
    assert bootstrap_v20a["seed"] == 20260801
    assert bootstrap_v20a["repetitions"] == 50_000
    assert bootstrap_v20a["one_sided_quantile"] == 0.05 / 60
    assert bootstrap_v20a["whole_panel_block_resampling_used"] is False
    assert bootstrap_v20a["draw_plan_content_sha256"] == (
        mechanics_v20a.DRAW_PLAN_CONTENT_SHA256_V20A
    )
    assert set(bootstrap_v20a["comparisons"]) == set(
        mechanics_v20a.CONTRASTS_V20A
    )
    assert sum(
        len(endpoints)
        for endpoints in bootstrap_v20a["comparisons"].values()
    ) == 60


def test_v20a_bootstrap_is_chunk_invariant(
    bootstrap_v20a,
    synthetic_unit_scores_v20a,
    panel_bundle_v20a,
):
    second = mechanics_v20a.paired_stratified_bootstrap_v20a(
        synthetic_unit_scores_v20a,
        panel_bundle_v20a,
        chunk_size=509,
    )
    assert second == bootstrap_v20a


def test_v20a_gate_is_conjunctive_and_only_authorizes_confirmation(
    bootstrap_v20a,
    synthetic_unit_scores_v20a,
    panel_bundle_v20a,
):
    observed = mechanics_v20a.recompute_observed_endpoints_v20a(
        synthetic_unit_scores_v20a, panel_bundle_v20a
    )
    passing = copy.deepcopy(bootstrap_v20a)
    for endpoints in passing["comparisons"].values():
        for endpoint in endpoints.values():
            endpoint["treatment_minus_control"] = 0.1
            endpoint["familywise_lcb"] = 0.01
    summary = {
        "arms": observed["arms"],
        "paired_bootstrap": passing,
        "runtime_integrity": {"all_integrity_audits_passed": True},
    }
    gate = mechanics_v20a.evaluate_attribution_gate_v20a(summary)
    assert gate["all_five_contrasts_passed"] is True
    assert "only_separate_fresh_basis_train_only_confirmation" in gate["decision"]
    assert gate["dataset_promotion_authorized"] is False
    assert gate["model_update_authorized"] is False
    assert gate["evaluation_authorized"] is False
    failing = copy.deepcopy(summary)
    endpoint = mechanics_v20a.prereg_v20a.ENDPOINT_NAMES_V20A[0]
    failing["paired_bootstrap"]["comparisons"][
        "conditional_tier3_after_tier2"
    ][endpoint]["familywise_lcb"] = -1e-9
    failed = mechanics_v20a.evaluate_attribution_gate_v20a(failing)
    assert failed["contrasts"]["conditional_tier3_after_tier2"][
        "preregistered_contrast_gate_passed"
    ] is False


def test_v20a_compact_summary_persists_no_scores_draws_or_content(
    monkeypatch,
    bootstrap_v20a,
    synthetic_unit_scores_v20a,
    panel_bundle_v20a,
):
    monkeypatch.setattr(
        mechanics_v20a,
        "paired_stratified_bootstrap_v20a",
        lambda *_args, **_kwargs: copy.deepcopy(bootstrap_v20a),
    )
    summary = mechanics_v20a.build_compact_estimator_summary_v20a(
        synthetic_unit_scores_v20a, panel_bundle_v20a
    )
    assert summary["persisted_response_vectors_or_row_content"] is False
    assert summary["bootstrap_draws_persisted"] is False
    assert summary["unit_scores_persisted"] is False
    keys = {str(key).lower() for key in _all_keys(summary)}
    assert not keys & {
        "unit_scores", "bootstrap_draws", "questions", "answers",
        "prompt", "response", "row_content",
    }


def test_v20a_bundle_validation_rejects_resealed_tampering(panel_bundle_v20a):
    tampered = copy.deepcopy(panel_bundle_v20a)
    tampered["panels"]["optimization_0"]["arms"]["patch_all_tiers"][
        "weights"
    ][0] += 1.0
    tampered["content_sha256_before_self_field"] = mechanics_v20a.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="panel bundle changed"):
        mechanics_v20a.validate_panel_bundle_v20a(tampered)
