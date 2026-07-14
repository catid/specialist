#!/usr/bin/env python3

import inspect
import json

import pytest

import build_eggroll_es_v14b_evidence_v15 as evidence_v15


def test_v15_compact_negative_evidence_is_exact_and_deterministic():
    frozen = json.loads(evidence_v15.OUTPUT_PATH_V15.read_text())
    built = evidence_v15.build_evidence_v15()
    assert built == frozen
    assert evidence_v15._file_sha256(evidence_v15.OUTPUT_PATH_V15) == (
        "735ad52b6395700feb4e8a3dccab165f9b79e620a53918d96e0a26979f58224c"
    )
    assert frozen["content_sha256_before_self_field"] == (
        "440504e6c81673ea8de89f336d587a0c57408ea21d3a925ed73b73ecfbeaa7b8"
    )
    assert frozen["content_sha256_before_self_field"] == evidence_v15._canonical({
        key: value for key, value in frozen.items()
        if key != "content_sha256_before_self_field"
    })


def test_v15_binds_exact_attempt_report_source_and_clean_runtime():
    evidence = evidence_v15.build_evidence_v15()
    assert evidence["v14b_attempt"] == {
        "path": str(evidence_v15.ATTEMPT_PATH_V15),
        "file_sha256": evidence_v15.ATTEMPT_FILE_SHA256_V15,
        "content_sha256": evidence_v15.ATTEMPT_CONTENT_SHA256_V15,
        "source_commit": evidence_v15.SOURCE_COMMIT_V15,
        "source_bundle_content_sha256": (
            evidence_v15.SOURCE_BUNDLE_CONTENT_SHA256_V15
        ),
    }
    assert evidence["v14b_report"]["file_sha256"] == (
        evidence_v15.REPORT_FILE_SHA256_V15
    )
    runtime = evidence["runtime_integrity"]
    assert runtime["alpha"] == 0.0
    assert runtime["model_update_applied"] is False
    assert runtime["application_count"] == 0
    assert runtime["all_integrity_audits_passed"] is True
    assert runtime["hardware"] == {
        "engine_count": 4, "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3], "population_waves": 8,
        "signed_waves": 16, "partial_waves": 0,
        "all_engines_generate_every_signed_wave": True,
    }
    assert runtime["moe_backend"] == "triton"
    assert runtime["tuned_config_folder"] is None


def test_v15_recomputes_exact_failed_gate_and_rules():
    _attempt, report, candidate, gate = evidence_v15.validate_v14b_negative_v15()
    assert candidate == report["diagnostic"]["analysis"]["candidate_summary"]
    assert gate == report["diagnostic"]["analysis"]["promotion_gate"]
    assert candidate["stability"] == evidence_v15.EXPECTED_STABILITY_V15
    assert candidate["all_panel_spreads_nonzero"] is True
    assert candidate["all_integrity_audits_passed"] is True
    assert candidate["robust_aggregate"]["coefficient_sha256"] == (
        evidence_v15.AGGREGATE_COEFFICIENT_SHA256_V15
    )
    assert gate["eligible_for_train_only_estimator_confirmation"] is False
    assert gate["eligible_for_model_update"] is False
    assert gate["eligible_to_open_evaluation"] is False
    assert evidence_v15.build_evidence_v15()["aggregate_gate"][
        "failed_rules"
    ] == evidence_v15.EXPECTED_FAILED_RULES_V15


def test_v15_contains_no_response_vectors_dense_hashes_or_source_rows():
    evidence = evidence_v15.build_evidence_v15()
    assert evidence["contains_response_vectors_or_dense_result_hashes"] is False
    assert evidence[
        "contains_source_rows_questions_answers_or_document_content"
    ] is False
    serialized = json.dumps(evidence, sort_keys=True)
    for forbidden in (
        "full_frame_sign_scores", "matched56_sign_scores",
        "complement_sign_scores", "dense_result_sha256",
        "document_sha256s", '"questions"', '"answers"', '"row_index"',
    ):
        assert forbidden not in serialized
    source = inspect.getsource(evidence_v15)
    for forbidden_call in (
        "load_panel_bundle_v14b(", "materialize_sampler_v14b(",
        "load_frozen_train(",
    ):
        assert forbidden_call not in source


def test_v15_decision_retains_v13_and_prohibits_every_release_surface():
    evidence = evidence_v15.build_evidence_v15()
    assert evidence["decision"] == {
        "sampler": "retain_v13",
        "fresh_basis_k2_confirmation_authorized": False,
        "evaluation_surface_opened_or_authorized": False,
        "model_update_applied_or_authorized": False,
        "reason": "v14b_failed_its_preregistered_conjunctive_gate",
    }
    assert evidence["aggregate_gate"][
        "eligible_for_fresh_basis_confirmation"
    ] is False
    assert "authorizes no follow-up experiment" in evidence[
        "next_step_constraint"
    ]


def test_v15_changed_artifact_fails_and_output_write_is_exclusive(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(evidence_v15, "REPORT_FILE_SHA256_V15", "0" * 64)
    with pytest.raises(RuntimeError, match="file identity changed"):
        evidence_v15.validate_v14b_negative_v15()
    output = tmp_path / "compact.json"
    evidence_v15._exclusive_write(output, {"bound": True})
    with pytest.raises(ValueError, match="already exists"):
        evidence_v15._exclusive_write(output, {"bound": True})
