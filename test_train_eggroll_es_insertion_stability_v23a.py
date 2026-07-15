#!/usr/bin/env python3
"""CPU-only tests for V23A compact estimator mechanics."""

import copy

import numpy as np
import pytest

import train_eggroll_es_insertion_stability_v23a as mechanics_v23a


def synthetic_scores():
    generator = np.random.default_rng(20260829)
    unit = generator.normal(size=(5, 2, 32, 56))
    reference = generator.normal(loc=-2.0, scale=0.2, size=(5, 56))
    return (
        {arm: unit.copy() for arm in mechanics_v23a.ARMS_V23A},
        {
            arm: reference.copy() + (0.0 if index == 0 else 0.1)
            for index, arm in enumerate(mechanics_v23a.ARMS_V23A)
        },
    )


def passing_integrity():
    return {
        arm: {
            key: False if key == "union_planner_called" else True
            for key in mechanics_v23a.RUNTIME_INTEGRITY_KEYS_V23A
        } for arm in mechanics_v23a.ARMS_V23A
    }


@pytest.fixture(scope="module")
def compact():
    unit, reference = synthetic_scores()
    panel_bundle = mechanics_v23a.load_panel_bundle_v23a()
    return mechanics_v23a.build_compact_estimator_summary_v23a(
        unit, reference, panel_bundle
    )


def test_v23a_compact_estimator_has_exact_arms_comparisons_and_no_raw_arrays(compact):
    assert tuple(compact["arms"]) == mechanics_v23a.ARMS_V23A
    assert set(compact["comparisons"]) == set(mechanics_v23a.CANDIDATE_ARMS_V23A)
    for comparison in compact["comparisons"].values():
        assert comparison["endpoint_count"] == 16
        assert set(comparison["endpoints"]) == set(mechanics_v23a.ALL_ENDPOINTS_V23A)
    assert compact["persisted_response_vectors_or_row_content"] is False
    assert compact["unit_scores_persisted"] is False
    assert compact["bootstrap_draws_or_replicates_persisted"] is False
    assert compact["bootstrap"] == {
        "seed": 20260827,
        "repetitions": 50_000,
        "one_sided_familywise_quantile": 0.05 / 48,
        "family_hypothesis_count": 48,
        "quantile_method": "linear",
        "draw_plan_content_sha256": compact["bootstrap"][
            "draw_plan_content_sha256"
        ],
        "paired_same_direction_and_row_draws_all_arms": True,
        "draw_arrays_persisted": False,
    }


def test_v23a_identical_gradient_arms_and_better_reference_pass_all_48(compact):
    summary = copy.deepcopy(compact)
    summary["runtime_integrity"] = passing_integrity()
    gate = mechanics_v23a.evaluate_gate_v23a(summary)
    assert gate["family_hypothesis_count"] == 48
    assert gate["passing_location_count"] == 3
    assert gate["compatibility_gate_passed"] is True
    assert gate["selected_location_for_confirmation"] in (
        mechanics_v23a.CANDIDATE_ARMS_V23A
    )
    assert gate["decision"] == (
        "authorize_only_separate_fresh_basis_train_only_confirmation"
    )
    assert all(gate[key] is False for key in (
        "direct_model_update_authorized", "checkpoint_write_authorized",
        "evaluation_authorized", "dataset_promotion_authorized",
    ))


def test_v23a_one_reference_failure_blocks_that_location(compact):
    summary = copy.deepcopy(compact)
    summary["runtime_integrity"] = passing_integrity()
    arm = "insert_front_e005"
    summary["comparisons"][arm]["endpoints"][
        "unperturbed_reward_delta_worst"
    ]["familywise_lcb"] = -1e-9
    gate = mechanics_v23a.evaluate_gate_v23a(summary)
    assert gate["location_results"][arm]["reference_pass_count"] == 3
    assert gate["location_results"][arm]["location_passed"] is False
    assert gate["passing_location_count"] == 2


def test_v23a_integrity_failure_is_valid_but_cannot_advance(compact):
    summary = copy.deepcopy(compact)
    summary["runtime_integrity"] = passing_integrity()
    summary["runtime_integrity"]["base_middle_late"][
        "exact_selected_reference_restored_every_signed_wave"
    ] = False
    gate = mechanics_v23a.evaluate_gate_v23a(summary)
    assert gate["compatibility_gate_passed"] is False
    assert gate["passing_location_count"] == 0
    assert gate["decision"] == "retain_v13_base_middle_late_recipe"


def test_v23a_gate_rejects_missing_endpoint_or_invalid_integrity(compact):
    summary = copy.deepcopy(compact)
    summary["runtime_integrity"] = passing_integrity()
    del summary["comparisons"]["insert_back_e005"]["endpoints"][
        mechanics_v23a.ALL_ENDPOINTS_V23A[0]
    ]
    with pytest.raises(RuntimeError, match="endpoint coverage"):
        mechanics_v23a.evaluate_gate_v23a(summary)
    summary = copy.deepcopy(compact)
    summary["runtime_integrity"] = passing_integrity()
    summary["runtime_integrity"]["insert_middle_e005"]["extra"] = True
    with pytest.raises(RuntimeError, match="runtime integrity"):
        mechanics_v23a.evaluate_gate_v23a(summary)


def test_v23a_draw_plan_is_exact_paired_and_not_persisted():
    bundle = mechanics_v23a.load_panel_bundle_v23a()
    certificate, directions, rows = mechanics_v23a.bootstrap_draw_plan_v23a(bundle)
    assert directions.shape == (50_000, 32)
    assert directions.dtype == np.uint8
    assert set(rows) == set(mechanics_v23a.PANEL_NAMES_V23A)
    assert certificate["paired_same_direction_draws_all_arms"] is True
    assert certificate["paired_same_stratified_row_draws_all_arms"] is True
    assert certificate["draw_arrays_persisted"] is False
    assert certificate["content_sha256_before_self_field"] == (
        mechanics_v23a.canonical_sha256({
            key: value for key, value in certificate.items()
            if key != "content_sha256_before_self_field"
        })
    )


@pytest.mark.parametrize("arm", mechanics_v23a.ARMS_V23A)
def test_v23a_rejects_wrong_raw_score_geometry(arm):
    unit, reference = synthetic_scores()
    unit[arm] = unit[arm][:, :, :, :-1]
    with pytest.raises(RuntimeError, match="geometry changed"):
        mechanics_v23a.build_compact_estimator_summary_v23a(
            unit, reference, mechanics_v23a.load_panel_bundle_v23a()
        )
