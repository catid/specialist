#!/usr/bin/env python3

import inspect
import json

import pytest

import build_eggroll_es_v23a_negative_evidence_r3 as evidence_r3


def test_completed_v23a_r3_is_exact_and_gate_recomputes():
    _attempt, report = evidence_r3.validate_completed_v23a_r3()
    assert report["gate"]["location_results"] == evidence_r3.EXPECTED_LOCATION_RESULTS
    assert report["gate"]["passing_location_count"] == 0
    assert report["gate"]["selected_location_for_confirmation"] is None


def test_compact_evidence_is_deterministic_and_self_sealed():
    frozen = json.loads(evidence_r3.OUTPUT_PATH.read_text())
    built = evidence_r3.build_evidence()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == evidence_r3.canonical_sha256(
        {
            key: value for key, value in frozen.items()
            if key != "content_sha256_before_self_field"
        }
    )


def test_negative_decision_is_fail_closed_and_scoped():
    evidence = evidence_r3.build_evidence()
    assert evidence["aggregate_gate"]["passing_location_count"] == 0
    assert evidence["aggregate_gate"]["selected_location_for_confirmation"] is None
    assert evidence["closure"]["confirmation_authorized"] is False
    assert evidence["closure"]["evaluation_authorized"] is False
    assert evidence["closure"]["model_update_authorized"] is False
    assert evidence["closure"]["retained_recipe"] == "v13_base_middle_late"
    assert len(evidence["closure"]["closed_hypotheses"]) == 3
    assert "does not generalize" in evidence["scope_note"]


def test_evidence_excludes_raw_and_nontrain_surfaces():
    evidence = evidence_r3.build_evidence()
    assert evidence[
        "contains_response_vectors_unit_scores_bootstrap_draws_or_replicates"
    ] is False
    assert evidence[
        "contains_dataset_rows_questions_answers_or_document_content"
    ] is False
    assert evidence[
        "contains_validation_ood_heldout_or_benchmark_content"
    ] is False
    serialized = json.dumps(evidence, sort_keys=True)
    for forbidden in (
        '"comparisons"', '"unit_scores"', '"bootstrap_draws"',
        '"questions"', '"answers"', '"heldout_results"', '"ood_results"',
    ):
        assert forbidden not in serialized
    source = inspect.getsource(evidence_r3)
    for forbidden_call in ("load_frozen_train(", "load_panel_bundle_v13("):
        assert forbidden_call not in source


def test_exclusive_write_rejects_reuse(tmp_path):
    output = tmp_path / "evidence.json"
    evidence_r3.exclusive_write(output, {"bound": True})
    with pytest.raises(ValueError, match="already exists"):
        evidence_r3.exclusive_write(output, {"bound": True})
