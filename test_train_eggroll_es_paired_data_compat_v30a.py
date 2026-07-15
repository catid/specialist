import json

import numpy as np
import pytest

import train_eggroll_es_paired_data_compat_v30a as mechanics


def _bundle():
    return mechanics.load_paired_panel_bundle_v30a()


def _identical_scores():
    values = np.empty((2, 5, 2, 32, 39), dtype=np.float64)
    direction = np.linspace(-1.0, 1.0, 32)
    for version in range(2):
        for panel in range(5):
            panel_signal = np.sin(direction * (panel + 1)) + direction * 0.1
            values[version, panel, 0] = panel_signal[:, None] + panel * 0.01
            values[version, panel, 1] = -panel_signal[:, None] + panel * 0.01
    return values


def test_v30a_materialized_frame_is_exact_5x39_and_content_sealed():
    bundle = _bundle()
    assert bundle["content_sha256_before_self_field"] == (
        "c2c18d07399b04d48c1e96b299fde88b73f83bf549a272f0d087d3bfc1502e80"
    )
    assert tuple(bundle["panels"]) == mechanics.PANEL_NAMES_V30A
    assert [len(panel["unit_ids"]) for panel in bundle["panels"].values()] == [39] * 5
    assert len({
        unit for panel in bundle["panels"].values() for unit in panel["unit_ids"]
    }) == 195
    anchors = [
        anchor
        for panel in bundle["panels"].values()
        for anchor in panel["pairing_anchors"]
    ]
    assert anchors.count("shared_document") == 193
    assert anchors.count("joint_component_cross_side_link") == 2
    assert bundle["frame"]["reserve_paired_units"] == 10
    assert bundle["contains_validation_ood_heldout_or_benchmark_content"] is False


def test_v30a_schedule_is_fresh_32_direction_16_wave_and_alternating():
    schedule = mechanics.resident_signed_wave_schedule_v30a()
    assert len(schedule) == 16
    assert [item["sign"] for item in schedule] == ["plus", "minus"] * 8
    assert [item["resident_version_order"] for item in schedule] == [
        ["production", "candidate_v389"],
        ["candidate_v389", "production"],
    ] * 8
    directions = [
        direction
        for item in schedule[::2]
        for direction in item["engine_direction_indices"]
    ]
    assert directions == list(range(32))
    assert all(item["restore_after_both_versions"] is True for item in schedule)


def test_v30a_resident_wave_perturbs_once_scores_both_then_restores():
    events = []
    schedule_item = mechanics.resident_signed_wave_schedule_v30a()[1]
    captures = mechanics.execute_paired_resident_signed_wave_v30a(
        schedule_item,
        perturb=lambda seeds, negate: events.append(("perturb", seeds, negate)),
        score_version=lambda version: events.append(("score", version)) or version,
        restore=lambda: events.append(("restore",)),
    )
    assert list(captures) == ["candidate_v389", "production"]
    assert events[0][0] == "perturb"
    assert [event[0] for event in events] == ["perturb", "score", "score", "restore"]
    assert events[0][1] == schedule_item["engine_direction_seeds"]
    assert events[0][2] is True


def test_v30a_exact_50k_draw_plan_hash_matches_preregistration():
    draws, digest = mechanics._bootstrap_draw_plan_v30a(
        _bundle(), 50_000, mechanics.EXPECTED_DRAW_PLAN_SHA256_V30A,
    )
    assert digest == "dbea000043a713114c150d07e11b813d1cbf00dcca9c418527d9a81c94b94ad5"
    assert sum(
        item["draw"].nbytes for panel in draws.values() for item in panel.values()
    ) == 78_000_000


def test_v30a_synthetic_paired_bootstrap_is_compact_and_has_12_zero_deltas():
    result = mechanics._paired_stratified_bootstrap_impl(
        _identical_scores(), _bundle(), 32, None,
    )
    assert result["repetitions"] == 32
    assert len(result["endpoints"]) == 12
    assert result["raw_draws_or_replicates_persisted"] is False
    for endpoint in result["endpoints"].values():
        assert endpoint["candidate_v389_minus_production"] == pytest.approx(0.0)
        assert endpoint["familywise_lcb"] == pytest.approx(0.0)
        assert endpoint["noninferiority_margin"] == 0.0
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in ("unit_scores", "bootstrap_draws", "bootstrap_replicates"):
        assert f'"{forbidden}"' not in serialized


def test_v30a_gate_is_conjunctive_and_authority_is_narrow():
    endpoints = {
        name: {
            "candidate_v389_minus_production": 0.0,
            "familywise_lcb": 0.0,
            "noninferiority_margin": 0.0,
        }
        for name in mechanics.prereg_v30a.ENDPOINTS
    }
    summary = {
        "runtime_integrity": {"all_integrity_audits_passed": True},
        "paired_bootstrap": {"endpoints": endpoints},
    }
    passed = mechanics.evaluate_candidate_v30a(summary)
    assert passed["pass"] is True
    assert passed["decision"] == (
        "authorize_only_separate_full-v389_train-only_HPO_preregistration"
    )
    assert passed["dataset_promotion_authorized"] is False
    assert passed["model_update_authorized"] is False
    failed_summary = json.loads(json.dumps(summary))
    failed_summary["paired_bootstrap"]["endpoints"][
        mechanics.prereg_v30a.ENDPOINTS[0]
    ]["familywise_lcb"] = -1e-12
    failed = mechanics.evaluate_candidate_v30a(failed_summary)
    assert failed["pass"] is False
    assert failed["decision"] == "retain_production_dataset_and_v13_recipe"


def test_v30a_score_geometry_and_bundle_identity_fail_closed():
    with pytest.raises(RuntimeError, match="score tensor changed"):
        mechanics.observed_panel_scores_v30a(np.zeros((2, 5, 2, 32, 38)), _bundle())
    changed = _bundle()
    changed["content_sha256_before_self_field"] = "0" * 64
    with pytest.raises(RuntimeError, match="bundle changed"):
        mechanics.validate_paired_panel_bundle_v30a(changed)
