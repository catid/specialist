#!/usr/bin/env python3

import inspect
import json

import pytest

import build_eggroll_es_v25a_paired_promising_unconfirmed_evidence_r1 as evidence_r1


def test_completed_v25a_r1_is_exact_integrity_passed_and_gate_failed():
    _attempt, report = evidence_r1.validate_completed_v25a_r1()
    assert report["gate"]["pass"] is False
    assert report["gate"]["all_runtime_integrity_audits_passed"] is True
    assert report["gate"]["all_12_familywise_lcbs_nonnegative"] is False


def test_compact_evidence_is_deterministic_and_self_sealed():
    frozen = json.loads(evidence_r1.OUTPUT_PATH.read_text())
    built = evidence_r1.build_evidence()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == evidence_r1.canonical_sha256({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })


def test_result_is_promising_but_not_recast_as_confirmation():
    evidence = evidence_r1.build_evidence()
    result = evidence["aggregate_result"]
    assert result["observed_candidate_minus_production_nonnegative"] == 12
    assert result["observed_candidate_minus_production_positive"] == 11
    assert result["familywise_lcbs_nonnegative"] == 0
    assert evidence["decision"]["global_gate_passed"] is False
    assert evidence["decision"]["confirmation_authorized_by_this_gate"] is False
    assert evidence["decision"]["evaluation_authorized"] is False
    assert evidence["decision"]["model_update_authorized"] is False


def test_evidence_excludes_raw_rows_and_nontrain_surfaces():
    evidence = evidence_r1.build_evidence()
    assert evidence["contains_response_vectors_unit_scores_or_bootstrap_replicates"] is False
    assert evidence["contains_dataset_rows_questions_answers_or_document_content"] is False
    assert evidence["contains_validation_ood_heldout_or_benchmark_content"] is False
    serialized = json.dumps(evidence, sort_keys=True)
    for forbidden in (
        '"unit_scores"', '"bootstrap_draws"', '"questions"', '"answers"',
        '"heldout_results"', '"ood_results"',
    ):
        assert forbidden not in serialized
    source = inspect.getsource(evidence_r1)
    for forbidden_call in ("load_frozen_train(", "load_panel_bundle_v13("):
        assert forbidden_call not in source


def test_exclusive_write_rejects_reuse(tmp_path):
    output = tmp_path / "evidence.json"
    evidence_r1.exclusive_write(output, {"bound": True})
    with pytest.raises(FileExistsError):
        evidence_r1.exclusive_write(output, {"bound": True})
