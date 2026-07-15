#!/usr/bin/env python3
"""Offline tests for the pure V17A paired trainer-side mechanics."""

import copy
import json
import math

import numpy as np
import pytest

import train_eggroll_es_paired_data_compat_v17a as trainer_v17a


def synthetic_unit_scores_v17a():
    """Nondegenerate paired scores with identical production/candidate arms."""
    scores = np.empty((2, 5, 2, 32, 38), dtype=np.float64)
    for arm_index in range(2):
        for panel_index in range(5):
            for sign_index in range(2):
                sign = 1.0 if sign_index == 0 else -1.0
                for direction_index in range(32):
                    for unit_index in range(38):
                        response = (
                            (direction_index - 15.5)
                            * (1.0 + 0.03 * panel_index)
                            + 0.001 * unit_index
                            * ((direction_index % 5) - 2)
                        )
                        scores[
                            arm_index, panel_index, sign_index,
                            direction_index, unit_index,
                        ] = -1.0 + 0.002 * unit_index + sign * response
    return scores


def reseal(value):
    value["content_sha256_before_self_field"] = trainer_v17a.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
    return value


def test_v17a_materializes_exact_fixed_paired_batches_without_eval_content():
    first = trainer_v17a.load_paired_panel_bundle_v17a()
    second = trainer_v17a.load_paired_panel_bundle_v17a()
    assert first == second
    assert first["content_sha256_before_self_field"] == (
        trainer_v17a.PAIRED_PANEL_BUNDLE_CONTENT_SHA256_V17A
    )
    assert first["contains_evaluation_content"] is False
    assert tuple(first["panels"]) == trainer_v17a.PANEL_NAMES_V17A
    assert sum(len(panel["unit_ids"]) for panel in first["panels"].values()) == 190
    assert len({
        unit_id
        for panel in first["panels"].values()
        for unit_id in panel["unit_ids"]
    }) == 190
    for panel in first["panels"].values():
        assert len(panel["unit_ids"]) == 38
        assert set(panel["arms"]) == set(trainer_v17a.ARMS_V17A)
        assert math.isclose(sum(panel["weights"]), 195.0, abs_tol=1e-12)
        assert {
            arm: panel["arms"][arm]["ordered_row_identity_sha256"]
            for arm in trainer_v17a.ARMS_V17A
        } == {
            arm: second["panels"][panel["name"]]["arms"][arm][
                "ordered_row_identity_sha256"
            ]
            for arm in trainer_v17a.ARMS_V17A
        }

    tampered = copy.deepcopy(first)
    tampered["panels"]["optimization_0"]["weights"][0] += 1e-6
    reseal(tampered)
    with pytest.raises(RuntimeError, match="bundle changed|contract changed"):
        trainer_v17a.validate_paired_panel_bundle_v17a(tampered)


def test_v17a_resident_schedule_pairs_both_arms_before_each_exact_restore():
    schedule = trainer_v17a.resident_signed_wave_schedule_v17a()
    assert len(schedule) == 16
    assert [item["sign"] for item in schedule] == ["plus", "minus"] * 8
    assert all(len(item["engine_seeds"]) == 4 for item in schedule)
    assert all(item["restore_after_both_arms"] is True for item in schedule)
    assert [item["resident_arm_order"] for item in schedule] == [
        list(trainer_v17a.ARMS_V17A) if index % 2 == 0
        else list(reversed(trainer_v17a.ARMS_V17A))
        for index in range(16)
    ]
    assert sum(
        item["resident_arm_order"][0] == "production" for item in schedule
    ) == 8
    assert sum(
        item["resident_arm_order"][0] == "candidate_v283" for item in schedule
    ) == 8

    changed = list(trainer_v17a.anchor_v13.PERTURBATION_SEEDS_V13)
    changed[0] += 1
    with pytest.raises(RuntimeError, match="basis changed"):
        trainer_v17a.resident_signed_wave_schedule_v17a(changed)

    events = []
    item = schedule[1]
    captures = trainer_v17a.execute_paired_resident_signed_wave_v17a(
        item,
        perturb=lambda seeds, negate: events.append(("perturb", seeds, negate)),
        score_arm=lambda arm: events.append(("score", arm)) or f"{arm}-score",
        restore=lambda: events.append(("restore",)),
    )
    assert events == [
        ("perturb", item["engine_seeds"], True),
        ("score", "candidate_v283"),
        ("score", "production"),
        ("restore",),
    ]
    assert tuple(captures) == ("candidate_v283", "production")

    failed_events = []
    with pytest.raises(RuntimeError, match="synthetic arm failure"):
        trainer_v17a.execute_paired_resident_signed_wave_v17a(
            schedule[0],
            perturb=lambda _seeds, _negate: failed_events.append("perturb"),
            score_arm=lambda arm: (
                (_ for _ in ()).throw(RuntimeError("synthetic arm failure"))
                if arm == "candidate_v283" else failed_events.append(arm)
            ),
            restore=lambda: failed_events.append("restore"),
        )
    assert failed_events == ["perturb", "production", "restore"]


def test_v17a_observed_endpoints_recompute_from_weighted_unit_scores():
    bundle = trainer_v17a.load_paired_panel_bundle_v17a()
    scores = synthetic_unit_scores_v17a()
    panel_scores = trainer_v17a.observed_panel_scores_v17a(scores, bundle)
    weights = np.asarray([
        bundle["panels"][name]["weights"]
        for name in trainer_v17a.PANEL_NAMES_V17A
    ])
    expected = np.einsum("apsdu,pu->apsd", scores, weights) / 195.0
    np.testing.assert_allclose(panel_scores, expected, rtol=0.0, atol=1e-12)

    observed = trainer_v17a.recompute_observed_endpoints_v17a(scores, bundle)
    production = observed["versions"]["production"]
    candidate = observed["versions"]["candidate_v283"]
    assert production["all_panel_spreads_nonzero"] is True
    assert candidate["all_panel_spreads_nonzero"] is True
    assert production["endpoint_values"] == candidate["endpoint_values"]
    assert set(production["endpoint_values"]) == set(
        trainer_v17a.prereg_v17a.ENDPOINT_CONTRACT_V17A
    )

    central = 0.5 * (panel_scores[0, :, 0] - panel_scores[0, :, 1])
    coefficients = [
        trainer_v17a.anchor_v13._standardize_v13(values.tolist())[0]
        for values in central
    ]
    pairwise = [
        trainer_v17a.anchor_v13._cosine_v13(coefficients[left], coefficients[right])
        for left, right in ((0, 1), (0, 2), (1, 2))
    ]
    assert production["endpoint_values"][
        "optimization_pairwise_cosine_median"
    ] == pytest.approx(float(np.median(pairwise)), abs=1e-12)
    assert production["endpoint_values"][
        "optimization_pairwise_cosine_worst"
    ] == pytest.approx(min(pairwise), abs=1e-12)

    zero_spread = np.ones_like(scores)
    with pytest.raises(RuntimeError, match="spread is zero"):
        trainer_v17a.recompute_observed_endpoints_v17a(zero_spread, bundle)


def test_v17a_exact_20k_bootstrap_is_paired_stratified_and_compact():
    bundle = trainer_v17a.load_paired_panel_bundle_v17a()
    scores = synthetic_unit_scores_v17a()
    bootstrap = trainer_v17a.paired_stratified_bootstrap_v17a(scores, bundle)
    assert set(bootstrap) == {
        "seed", "repetitions", "one_sided_quantile", "endpoints",
    }
    assert bootstrap["seed"] == trainer_v17a.prereg_v17a.BOOTSTRAP_SEED_V17A
    assert bootstrap["repetitions"] == 20_000
    assert bootstrap["one_sided_quantile"] == 0.05 / 12
    assert set(bootstrap["endpoints"]) == set(
        trainer_v17a.prereg_v17a.ENDPOINT_CONTRACT_V17A
    )
    for endpoint in bootstrap["endpoints"].values():
        assert set(endpoint) == {
            "candidate_minus_production", "familywise_lcb",
            "noninferiority_margin",
        }
        assert endpoint == {
            "candidate_minus_production": 0.0,
            "familywise_lcb": 0.0,
            "noninferiority_margin": 0.0,
        }
    serialized = json.dumps(bootstrap, sort_keys=True)
    assert "unit_scores" not in serialized
    assert "replicates" not in serialized


def test_v17a_trainer_mechanics_expose_no_launch_update_or_eval_entrypoint():
    for name in (
        "main", "run", "load_trainer", "make_trainer", "apply_seed_coefficients",
        "train_step", "fit", "eval_step", "save_checkpoint",
    ):
        assert not hasattr(trainer_v17a, name)
