from __future__ import annotations

import pytest

import build_category_balanced_selection_contract_v1 as balanced


def test_static_targets_are_exact_on_every_selection_axis():
    balanced.validate_static_targets()
    sources = balanced.source_targets()
    categories = balanced.category_targets()
    assert len(sources) == 29
    assert sum(sources.values()) == 740_847
    assert categories == {
        "lineage_people_history": 148_169,
        "tying_knots_frictions_technique": 120_000,
        "rigging_hardpoints_uplines_mechanics": 200_000,
        "safety_anatomy_consent_risk": 150_000,
        "materials_inspection_care_equipment": 122_678,
    }
    assert sum(balanced.TASK_SUBTYPE_TARGETS.values()) == 740_847
    assert sum(balanced.TASK_FAMILY_TARGETS.values()) == 740_847
    assert sum(balanced.GENERATION_MODE_TARGETS.values()) == 740_847
    assert sources["shibari_atlas"] / 740_847 <= 0.20
    assert min(sources.values()) >= balanced.MINIMUM_ACCEPTED_TOKENS_PER_SOURCE


def test_duplicate_source_assignment_fails_closed(monkeypatch: pytest.MonkeyPatch):
    mutated = {
        **balanced.CATEGORY_SOURCE_TARGETS,
        "synthetic_duplicate": {"shibari_atlas": 1},
    }
    monkeypatch.setattr(balanced, "CATEGORY_SOURCE_TARGETS", mutated)
    with pytest.raises(RuntimeError, match="multiple primary categories"):
        balanced.source_targets()


def test_materialized_contract_audits_skew_and_forbids_silent_reallocation():
    contract = balanced.build(check=True)
    audit = contract["current_source_group_weighted_plan_audit"]
    assert audit["largest_source"] == "shibari_atlas"
    assert audit["largest_source_fraction"] == pytest.approx(0.8951497407696866)
    assert contract["anti_dominance_gates"]["shibari_atlas_max_fraction_of_generated"] <= 0.20
    assert contract["accepted_token_targets"]["all_axes_must_hold_simultaneously"] is True
    assert contract["exact_budget_solver"][
        "silent_cross_source_or_cross_category_reallocation_allowed"
    ] is False
    assert contract["candidate_pool_contract"][
        "category_balanced_technical_deficit_pass_required"
    ] is True
    assert contract["training_launch_authorized"] is False
