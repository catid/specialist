import copy
import json

import numpy as np
import pytest

import train_eggroll_es_v401_replacement_fraction_v34b as mechanics


def _synthetic_scores(seed=34):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(2, 5, 2, 64, 39))


def _integrity(value=True):
    result = {key: value for key in mechanics.RUNTIME_INTEGRITY_KEYS}
    result["all_integrity_audits_passed"] = value
    return result


def test_hardened_preregistration_and_transient_panel_bundle():
    prereg = mechanics.load_hardened_preregistration()
    assert prereg["content_sha256_before_self_field"] == mechanics.PREREGISTRATION_CONTENT_SHA256
    bundle = mechanics.materialize_paired_panel_bundle()
    assert mechanics.validate_panel_bundle(bundle) is bundle
    assert bundle["content_sha256_before_self_field"] == mechanics.PANEL_BUNDLE_CONTENT_SHA256
    assert bundle["contains_validation_ood_heldout_or_benchmark_content"] is False


def test_resident_wave_restores_on_success_and_failure():
    item = mechanics.resident_signed_wave_schedule()[0]
    events = []
    captures = mechanics.execute_resident_signed_wave(
        item,
        perturb=lambda seeds, negate: events.append(("perturb", len(seeds), negate)),
        score_source=lambda source: events.append(("score", source)) or source,
        restore=lambda: events.append(("restore",)),
    )
    assert tuple(captures) == tuple(item["resident_source_order"])
    assert events[-1] == ("restore",)
    events.clear()
    with pytest.raises(ZeroDivisionError):
        mechanics.execute_resident_signed_wave(
            item,
            perturb=lambda seeds, negate: None,
            score_source=lambda source: 1 / 0,
            restore=lambda: events.append(("restore",)),
        )
    assert events == [("restore",)]


def test_convex_fraction_algebra_adds_no_score_requests():
    rng = np.random.default_rng(1)
    sources = rng.normal(size=(2, 7, 5, 2, 64))
    result = mechanics.convex_fraction_panel_scores(sources)
    assert result.shape == (6, 7, 5, 2, 64)
    assert np.array_equal(result[0], sources[0])
    assert np.array_equal(result[-1], sources[1])
    assert np.allclose(result[3], 0.8 * sources[0] + 0.2 * sources[1])


def test_endpoint_geometry_and_central_response_interpolation():
    rng = np.random.default_rng(2)
    sources = rng.normal(size=(2, 3, 5, 2, 64))
    arms = mechanics.convex_fraction_panel_scores(sources)
    analyzed = mechanics.endpoint_arrays(arms)
    assert set(analyzed["endpoints"]) == set(mechanics.prereg_v34b.ENDPOINTS)
    assert all(value.shape == (6, 3) for value in analyzed["endpoints"].values())
    production_central = 0.5 * (sources[0, :, :, 0] - sources[0, :, :, 1])
    candidate_central = 0.5 * (sources[1, :, :, 0] - sources[1, :, :, 1])
    assert np.allclose(
        analyzed["central"][2],
        0.9 * production_central + 0.1 * candidate_central,
    )


def test_small_fixed_sequence_bootstrap_is_paired_and_compact():
    bundle = mechanics.materialize_paired_panel_bundle()
    result = mechanics.analyze_fixed_sequence_impl(
        _synthetic_scores(),
        bundle,
        repetitions=128,
        expected_draw_plan_sha256=None,
    )
    assert result["bootstrap"]["repetitions"] == 128
    assert 1 <= len(result["tested_fractions"]) <= 5
    assert result["fraction_specific_model_requests"] == 0
    assert result["tested_fractions"][0]["fraction"] == 0.05
    encoded = json.dumps(result).lower()
    for forbidden in ('"unit_scores"', '"coefficients"', '"bootstrap_draws"', '"questions"', '"answers"'):
        assert forbidden not in encoded


def test_gate_stops_at_first_failure_and_never_promotes_directly():
    endpoints = {
        name: {
            "fraction_minus_production": -0.01,
            "familywise_lcb": -0.02,
            "noninferiority_margin": 0.0,
        }
        for name in mechanics.prereg_v34b.ENDPOINTS
    }
    analysis = {
        "schema": "eggroll-es-v401-replacement-fraction-fixed-sequence-analysis-v34b",
        "bootstrap": {},
        "production_compact_estimator_sha256": "0" * 64,
        "tested_fractions": [{
            "fraction": 0.05,
            "all_12_point_deltas_nonnegative": False,
            "all_12_familywise_lcbs_nonnegative": False,
            "pass": False,
            "endpoints": endpoints,
            "fraction_compact_estimator_sha256": "1" * 64,
        }],
        "untested_fractions_after_first_failure": [0.1, 0.2, 0.4, 1.0],
        "stopped_at_first_failure": True,
        "largest_consecutively_passing_fraction": 0.0,
        "fraction_specific_model_requests": 0,
        "persisted_response_vectors_unit_scores_coefficients_or_draws": False,
    }
    analysis["content_sha256_before_self_field"] = mechanics.canonical_sha256(analysis)
    summary = {
        "schema": "eggroll-es-v401-replacement-fraction-compact-summary-v34b",
        "preregistration_content_sha256": mechanics.PREREGISTRATION_CONTENT_SHA256,
        "frame_content_sha256": mechanics.prereg_v34b.FRAME_CONTENT_SHA256,
        "runtime_integrity": _integrity(),
        "fixed_sequence_analysis": analysis,
        "contains_dataset_rows_questions_answers_document_or_eval_content": False,
        "contains_unit_scores_response_vectors_coefficients_bootstrap_draws_or_replicates": False,
    }
    summary["content_sha256_before_self_field"] = mechanics.canonical_sha256(summary)
    gate = mechanics.evaluate_gate(summary)
    assert gate["largest_consecutively_passing_fraction"] == 0.0
    assert gate["decision"] == "retain_production_no_fraction_authorized"
    assert gate["direct_dataset_promotion_authorized"] is False
    assert gate["model_update_authorized"] is False


def test_integrity_fails_closed():
    assert mechanics.validate_runtime_integrity(_integrity())
    changed = _integrity()
    changed["all_four_tp1_engines_every_signed_wave"] = False
    with pytest.raises(RuntimeError):
        mechanics.validate_runtime_integrity(changed)
    changed = _integrity()
    changed["unexpected"] = True
    with pytest.raises(RuntimeError):
        mechanics.validate_runtime_integrity(changed)
