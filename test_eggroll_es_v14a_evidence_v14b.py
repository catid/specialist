#!/usr/bin/env python3

import inspect
import json

import pytest

import build_eggroll_es_v14a_evidence_v14b as evidence_v14b


def test_v14b_compact_negative_evidence_is_exact_and_deterministic():
    frozen = json.loads(evidence_v14b.OUTPUT_PATH_V14B.read_text())
    built = evidence_v14b.build_evidence_v14b()
    assert built == frozen
    assert evidence_v14b._file_sha256(evidence_v14b.OUTPUT_PATH_V14B) == (
        "9329c09fec5d76e209cff36ac80ffbbec69f53fe7bae9edcc069421d626cc9e9"
    )
    assert frozen["content_sha256_before_self_field"] == (
        "ee4ded3d974dfd0becaedb1007f96888e133db51e62130d2844ab9c25e2ccf2b"
    )
    assert frozen["content_sha256_before_self_field"] == (
        evidence_v14b._canonical({
            key: value for key, value in frozen.items()
            if key != "content_sha256_before_self_field"
        })
    )


def test_v14b_binds_exact_attempt_report_source_and_runtime_integrity():
    evidence = evidence_v14b.build_evidence_v14b()
    assert evidence["v14a_attempt"] == {
        "path": str(evidence_v14b.ATTEMPT_PATH_V14B),
        "file_sha256": evidence_v14b.ATTEMPT_FILE_SHA256_V14B,
        "content_sha256": evidence_v14b.ATTEMPT_CONTENT_SHA256_V14B,
        "source_commit": evidence_v14b.SOURCE_COMMIT_V14B,
        "source_bundle_content_sha256": (
            evidence_v14b.SOURCE_BUNDLE_CONTENT_SHA256_V14B
        ),
    }
    assert evidence["v14a_report"]["file_sha256"] == (
        evidence_v14b.REPORT_FILE_SHA256_V14B
    )
    assert evidence["v14a_report"]["content_sha256"] == (
        evidence_v14b.REPORT_CONTENT_SHA256_V14B
    )
    assert evidence["v14a_report"]["implementation_bundle_sha256"] == (
        evidence_v14b.IMPLEMENTATION_BUNDLE_SHA256_V14B
    )
    runtime = evidence["runtime_integrity"]
    assert runtime["alpha"] == 0.0
    assert runtime["model_update_applied"] is False
    assert runtime["application_count"] == 0
    assert runtime["identity_audit_passed"] is True
    assert runtime["pre_post_base_probe_equal"] is True
    assert runtime["hardware"] == {
        "engine_count": 4, "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3], "population_waves": 8,
        "signed_waves": 16, "partial_waves": 0,
        "all_engines_generate_every_signed_wave": True,
    }


def test_v14b_recomputes_the_failed_conjunctive_gate_from_aggregates():
    _attempt, report, candidate, gate = (
        evidence_v14b.validate_v14a_negative_v14b()
    )
    assert candidate == report["diagnostic"]["analysis"]["candidate_summary"]
    assert gate == report["diagnostic"]["analysis"]["promotion_gate"]
    assert candidate["stability"] == evidence_v14b.EXPECTED_STABILITY_V14B
    assert candidate["all_panel_spreads_nonzero"] is True
    assert candidate["robust_aggregate"]["coefficient_sha256"] == (
        evidence_v14b.AGGREGATE_COEFFICIENT_SHA256_V14B
    )
    assert gate["eligible_for_train_only_sampler_adoption"] is False
    assert gate["eligible_for_model_update"] is False
    evidence = evidence_v14b.build_evidence_v14b()
    assert evidence["aggregate_gate"]["failed_rules"] == [
        "matched56_pairwise_cosine.median_passed",
        "matched56_pairwise_cosine.worst_passed",
        "full_to_matched56_optimization_cosine.median_passed",
        "full_to_matched56_optimization_cosine.worst_passed",
        "full_to_matched56_optimization_sign_agreement.median_passed",
    ]
    assert evidence["decision"] == {
        "sampler": "retain_v13",
        "row_draw_iteration_1_confirmation_authorized": False,
        "evaluation_surface_opened": False,
        "model_update_authorized": False,
        "reason": "v14a_failed_its_preregistered_conjunctive_gate",
    }


def test_v14b_evidence_contains_no_response_vectors_dense_hashes_or_rows():
    evidence = evidence_v14b.build_evidence_v14b()
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
    source = inspect.getsource(evidence_v14b)
    for forbidden_call in (
        "load_panel_bundle_v14a(", "materialize_panels_v14a(",
        "load_frozen_train(",
    ):
        assert forbidden_call not in source


def test_v14b_recommends_one_cost_bounded_k2_row_mean_hypothesis():
    recommendation = evidence_v14b.build_evidence_v14b()[
        "next_train_only_estimator_recommendation"
    ]
    assert recommendation["name"] == (
        "fresh_two_distinct_rows_per_multiline_document_mean"
    )
    assert recommendation["hypothesis_count"] == 1
    assert recommendation["full_frame_documents"] == 310
    assert recommendation["single_row_documents"] == 139
    assert recommendation["multirow_documents"] == 171
    assert recommendation["distinct_rows_per_multirow_document"] == 2
    assert recommendation["unique_train_prompts_per_direction_sign"] == 481
    assert "k in {1,2,3,all}" in recommendation[
        "why_not_multiplicity_search_now"
    ]
    assert "multiple-testing" in recommendation[
        "why_not_multiplicity_search_now"
    ]
    assert recommendation["current_authorization"].startswith(
        "recommendation only"
    )


def test_v14b_rejects_changed_source_artifact_and_writes_exclusively(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(evidence_v14b, "REPORT_FILE_SHA256_V14B", "0" * 64)
    with pytest.raises(RuntimeError, match="file identity changed"):
        evidence_v14b.validate_v14a_negative_v14b()
    output = tmp_path / "compact.json"
    evidence_v14b._exclusive_write(output, {"bound": True})
    with pytest.raises(ValueError, match="already exists"):
        evidence_v14b._exclusive_write(output, {"bound": True})
