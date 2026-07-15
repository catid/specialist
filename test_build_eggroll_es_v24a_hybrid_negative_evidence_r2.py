#!/usr/bin/env python3

import inspect
import json

import pytest

import build_eggroll_es_v24a_hybrid_negative_evidence_r2 as evidence_r2


def test_completed_v24a_r1_is_exact_and_failed_only_quality_speed_gate():
    _attempt, report = evidence_r2.validate_completed_v24a_r1()
    estimator = report["estimator_and_gate"]
    assert estimator["global_pass"] is False
    for pair in ("pair_a", "pair_b"):
        assert estimator["pairs"][pair]["memory_pass"] is True
        assert estimator["pairs"][pair]["quality_pass"] is False
        assert estimator["pairs"][pair]["runtime_integrity_pass"] is True


def test_compact_evidence_is_deterministic_and_self_sealed():
    frozen = json.loads(evidence_r2.OUTPUT_PATH.read_text())
    built = evidence_r2.build_evidence()
    assert built == frozen
    assert frozen["content_sha256_before_self_field"] == evidence_r2.canonical_sha256(
        {
            key: value for key, value in frozen.items()
            if key != "content_sha256_before_self_field"
        }
    )


def test_negative_decision_is_fail_closed_but_preserves_memory_finding():
    evidence = evidence_r2.build_evidence()
    assert evidence["memory_result"]["reduction"] == pytest.approx(
        0.48187173896726543
    )
    assert evidence["memory_result"]["both_pairs_passed"] is True
    assert evidence["quality_result"]["familywise_endpoints_passed_per_pair"] == {
        "pair_a": 0, "pair_b": 0,
    }
    assert evidence["decision"]["global_gate_passed"] is False
    assert evidence["decision"]["confirmation_authorized"] is False
    assert evidence["decision"]["evaluation_authorized"] is False
    assert evidence["decision"]["model_update_authorized"] is False
    assert evidence["decision"]["retain_backend"].startswith("bf16_")


def test_evidence_excludes_raw_and_nontrain_surfaces():
    evidence = evidence_r2.build_evidence()
    assert evidence[
        "contains_response_vectors_unit_scores_timing_vectors_or_bootstrap_replicates"
    ] is False
    assert evidence[
        "contains_dataset_rows_questions_answers_or_document_content"
    ] is False
    assert evidence[
        "contains_validation_ood_heldout_or_benchmark_content"
    ] is False
    serialized = json.dumps(evidence, sort_keys=True)
    for forbidden in (
        '"unit_scores"', '"timing_vectors"', '"bootstrap_draws"',
        '"questions"', '"answers"', '"heldout_results"', '"ood_results"',
    ):
        assert forbidden not in serialized
    source = inspect.getsource(evidence_r2)
    for forbidden_call in ("load_frozen_train(", "load_panel_bundle_v13("):
        assert forbidden_call not in source


def test_exclusive_write_rejects_reuse(tmp_path):
    output = tmp_path / "evidence.json"
    evidence_r2.exclusive_write(output, {"bound": True})
    with pytest.raises(ValueError, match="already exists"):
        evidence_r2.exclusive_write(output, {"bound": True})
