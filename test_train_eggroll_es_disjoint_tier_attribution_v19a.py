#!/usr/bin/env python3
"""Pure synthetic tests for V19A disjoint-tier trainer mechanics."""

import copy

import numpy as np
import pytest

import train_eggroll_es_disjoint_tier_attribution_v19a as mechanics_v19a


@pytest.fixture(scope="module")
def panel_bundle_v19a():
    return mechanics_v19a.load_panel_bundle_v19a()


@pytest.fixture(scope="module")
def synthetic_unit_scores_v19a():
    result = {}
    for arm_index, arm in enumerate(mechanics_v19a.ARMS_V19A):
        shape = (
            10,
            2,
            32,
            mechanics_v19a.frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm],
        )
        result[arm] = np.random.default_rng(20260800 + arm_index).normal(
            size=shape
        )
    return result


@pytest.fixture(scope="module")
def bootstrap_v19a(synthetic_unit_scores_v19a, panel_bundle_v19a):
    return mechanics_v19a.paired_stratified_bootstrap_v19a(
        synthetic_unit_scores_v19a,
        panel_bundle_v19a,
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


def test_v19a_rematerializes_exact_commit_bound_in_memory_bundle(
    panel_bundle_v19a,
):
    bundle = panel_bundle_v19a
    assert bundle == mechanics_v19a.load_panel_bundle_v19a()
    assert bundle["content_sha256_before_self_field"] == (
        "b9bfc1868f5e2a6f54cd9531e0b759872020d2bc8fb9e8a8a287b548293d4f06"
    )
    assert bundle["preregistration"] == {
        "file_sha256": mechanics_v19a.PREREGISTRATION_FILE_SHA256_V19A,
        "content_sha256": mechanics_v19a.PREREGISTRATION_CONTENT_SHA256_V19A,
        "sealed_commit": mechanics_v19a.SEALED_PREREG_COMMIT_V19A,
    }
    assert bundle["frame"] == {
        "file_sha256": mechanics_v19a.FRAME_CERTIFICATE_FILE_SHA256_V19A,
        "content_sha256": mechanics_v19a.FRAME_CERTIFICATE_CONTENT_SHA256_V19A,
        "sealed_commit": mechanics_v19a.SEALED_PREREG_COMMIT_V19A,
    }
    assert bundle["token_audit"] == {
        "content_sha256": mechanics_v19a.prereg_v19a.TOKEN_AUDIT_CONTENT_SHA256_V19A,
        "frozen_total_token_cap": 1024,
        "over_cap_count": 0,
        "observed_combined_token_max": 144,
    }
    assert bundle["contains_evaluation_content"] is False
    assert bundle["persisted_to_disk"] is False


def test_v19a_bundle_has_exact_own_tier_substitution_and_no_duplicates(
    panel_bundle_v19a,
):
    for panel in mechanics_v19a.PANEL_NAMES_V19A:
        arms = panel_bundle_v19a["panels"][panel]["arms"]
        production_base = arms["production_only"]["joint_ids"]
        assert len(production_base) == len(set(production_base)) == 24
        for arm in mechanics_v19a.ARMS_V19A:
            batch = arms[arm]
            active_tier = mechanics_v19a.frame_v19a.ARM_ACTIVE_TIER_V19A[arm]
            assert batch["joint_ids"][:24] == production_base
            assert len(batch["joint_ids"]) == len(set(batch["joint_ids"]))
            assert len(batch["joint_ids"]) == (
                mechanics_v19a.frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm]
            )
            assert sum(batch["weights"]) == pytest.approx(
                mechanics_v19a.frame_v19a.ARM_POPULATIONS_V19A[arm],
                abs=1e-12,
            )
            expected_candidate_sides = 0 if active_tier is None else 7
            assert batch["representative_sides"].count("candidate") == (
                expected_candidate_sides
            )
            for stratum, side in zip(
                batch["ht_strata"][:24],
                batch["representative_sides"][:24],
            ):
                assert side == (
                    "candidate"
                    if stratum == f"paired_tier_{active_tier}"
                    else "production"
                )
            candidate_strata = [
                item
                for item in batch["ht_strata"]
                if item.startswith("candidate_only_tier_")
            ]
            assert candidate_strata == (
                []
                if active_tier is None
                else [f"candidate_only_tier_{active_tier}"]
            )


def test_v19a_unit_score_contract_and_ht_recomputation_are_exact(
    panel_bundle_v19a,
):
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
    for arm in mechanics_v19a.ARMS_V19A:
        requests = mechanics_v19a.frame_v19a.ARM_REQUESTS_PER_PANEL_V19A[arm]
        values = np.empty((10, 2, 32, requests), dtype=np.float64)
        arm_expected = []
        for panel_index, panel in enumerate(mechanics_v19a.PANEL_NAMES_V19A):
            batch = panel_bundle_v19a["panels"][panel]["arms"][arm]
            row_values = np.asarray([
                stratum_values[stratum] for stratum in batch["ht_strata"]
            ])
            values[panel_index] = row_values[np.newaxis, np.newaxis, :]
            arm_expected.append(
                np.dot(row_values, batch["weights"])
                / mechanics_v19a.frame_v19a.ARM_POPULATIONS_V19A[arm]
            )
        unit_scores[arm] = values
        expected[arm] = np.asarray(arm_expected)
    observed = mechanics_v19a.observed_panel_scores_v19a(
        unit_scores, panel_bundle_v19a
    )
    assert observed.shape == (4, 10, 2, 32)
    for arm_index, arm in enumerate(mechanics_v19a.ARMS_V19A):
        assert np.allclose(
            observed[arm_index],
            np.broadcast_to(expected[arm][:, np.newaxis, np.newaxis], (10, 2, 32)),
            rtol=0.0,
            atol=1e-15,
        )

    malformed = copy.deepcopy(unit_scores)
    malformed["patch_tier_2_only"] = malformed["patch_tier_2_only"][..., :-1]
    with pytest.raises(RuntimeError, match="score tensor changed"):
        mechanics_v19a.observed_panel_scores_v19a(malformed, panel_bundle_v19a)


def test_v19a_observed_endpoint_geometry_is_six_four_and_twelve(
    synthetic_unit_scores_v19a,
    panel_bundle_v19a,
):
    panel_scores = mechanics_v19a.observed_panel_scores_v19a(
        synthetic_unit_scores_v19a, panel_bundle_v19a
    )
    analyzed = mechanics_v19a._endpoint_arrays_v19a(
        panel_scores[:, np.newaxis, ...]
    )
    assert analyzed["coefficients"].shape == (4, 1, 10, 32)
    assert analyzed["aggregate"].shape == (4, 1, 32)
    assert {
        name: values.shape[-1]
        for name, values in analyzed["families"].items()
    } == {
        "optimization_pairwise_cosine": 15,
        "optimization_pairwise_sign_agreement": 15,
        "aggregate_to_optimization_cosine": 6,
        "aggregate_to_optimization_sign_agreement": 6,
        "train_screen_cosine": 4,
        "train_screen_sign_agreement": 4,
    }
    assert set(analyzed["endpoints"]) == set(
        mechanics_v19a.prereg_v19a.ENDPOINT_NAMES_V19A
    )
    observed = mechanics_v19a.recompute_observed_endpoints_v19a(
        synthetic_unit_scores_v19a, panel_bundle_v19a
    )
    assert set(observed["arms"]) == set(mechanics_v19a.ARMS_V19A)
    assert all(
        len(contract["endpoint_values"]) == 12
        and contract["all_panel_spreads_nonzero"] is True
        for contract in observed["arms"].values()
    )


def test_v19a_exact_50000_bootstrap_is_reproducible_and_chunk_invariant(
    bootstrap_v19a,
    synthetic_unit_scores_v19a,
    panel_bundle_v19a,
):
    second = mechanics_v19a.paired_stratified_bootstrap_v19a(
        synthetic_unit_scores_v19a,
        panel_bundle_v19a,
        chunk_size=509,
    )
    assert bootstrap_v19a == second
    assert bootstrap_v19a["seed"] == 20260728
    assert bootstrap_v19a["repetitions"] == 50_000
    assert bootstrap_v19a["one_sided_quantile"] == 0.05 / 36
    assert bootstrap_v19a["whole_panel_block_resampling_used"] is False
    assert bootstrap_v19a["draw_plan_content_sha256"] == (
        mechanics_v19a.BOOTSTRAP_DRAW_PLAN_CONTENT_SHA256_V19A
    )
    assert set(bootstrap_v19a["comparisons"]) == set(
        mechanics_v19a.ARMS_V19A[1:]
    )
    assert all(
        len(endpoints) == 12
        for endpoints in bootstrap_v19a["comparisons"].values()
    )


def test_v19a_gate_is_conjunctive_and_can_only_authorize_confirmation(
    bootstrap_v19a,
    synthetic_unit_scores_v19a,
    panel_bundle_v19a,
):
    observed = mechanics_v19a.recompute_observed_endpoints_v19a(
        synthetic_unit_scores_v19a, panel_bundle_v19a
    )
    passing_bootstrap = copy.deepcopy(bootstrap_v19a)
    for endpoints in passing_bootstrap["comparisons"].values():
        for endpoint in endpoints.values():
            endpoint["patch_minus_production"] = 0.1
            endpoint["familywise_lcb"] = 0.01
    summary = {
        "arms": observed["arms"],
        "paired_bootstrap": passing_bootstrap,
        "runtime_integrity": {"all_integrity_audits_passed": True},
    }
    gate = mechanics_v19a.evaluate_attribution_gate_v19a(summary)
    assert gate["any_tier_passed"] is True
    assert gate["passing_tiers"] == list(mechanics_v19a.ARMS_V19A[1:])
    assert "only_separate_fresh_basis_train_only_confirmation" in gate["decision"]
    assert gate["dataset_promotion_authorized"] is False
    assert gate["model_update_authorized"] is False
    assert gate["evaluation_authorized"] is False

    failing = copy.deepcopy(summary)
    endpoint = mechanics_v19a.prereg_v19a.ENDPOINT_NAMES_V19A[0]
    failing["paired_bootstrap"]["comparisons"]["patch_tier_2_only"][endpoint][
        "familywise_lcb"
    ] = -1e-9
    failed_gate = mechanics_v19a.evaluate_attribution_gate_v19a(failing)
    assert failed_gate["arms"]["patch_tier_2_only"][
        "preregistered_attribution_gate_passed"
    ] is False


def test_v19a_compact_summary_persists_no_scores_draws_or_row_content(
    monkeypatch,
    bootstrap_v19a,
    synthetic_unit_scores_v19a,
    panel_bundle_v19a,
):
    monkeypatch.setattr(
        mechanics_v19a,
        "paired_stratified_bootstrap_v19a",
        lambda *_args, **_kwargs: copy.deepcopy(bootstrap_v19a),
    )
    summary = mechanics_v19a.build_compact_estimator_summary_v19a(
        synthetic_unit_scores_v19a, panel_bundle_v19a
    )
    assert summary["persisted_response_vectors_or_row_content"] is False
    assert summary["bootstrap_draws_persisted"] is False
    assert summary["unit_scores_persisted"] is False
    assert summary["runtime_integrity_required_before_gate"] is True
    assert summary["content_sha256_before_self_field"] == (
        mechanics_v19a.canonical_sha256({
            key: item
            for key, item in summary.items()
            if key != "content_sha256_before_self_field"
        })
    )
    keys = {str(key).lower() for key in _all_keys(summary)}
    assert not keys & {
        "unit_scores",
        "bootstrap_draws",
        "questions",
        "answers",
        "prompt",
        "response",
        "row_content",
    }


def test_v19a_bundle_validation_rejects_resealed_tampering(panel_bundle_v19a):
    tampered = copy.deepcopy(panel_bundle_v19a)
    tampered["panels"]["optimization_0"]["arms"]["patch_tier_3_only"][
        "weights"
    ][0] += 1.0
    tampered["content_sha256_before_self_field"] = mechanics_v19a.canonical_sha256({
        key: item
        for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="panel bundle changed"):
        mechanics_v19a.validate_panel_bundle_v19a(tampered)
