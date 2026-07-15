#!/usr/bin/env python3
"""Focused aggregate-only tests for V22A negative evidence."""

import copy
import json

import pytest

import build_eggroll_es_v341_matched_replacement_negative_evidence_v22a as evidence


EXPECTED_CONTENT_SHA256 = (
    "67eb3c39aa8d13a8924a2880c04f021f8ad27b9fcfb82eba9731ab195f5d3318"
)
EXPECTED_FILE_SHA256 = (
    "408d48a8eb9a63da9f12062d94b5249c88cc04d254eb553230e23cae516a1955"
)


def load_documents():
    return tuple(
        json.loads(path.read_text(encoding="utf-8"))
        for path in (
            evidence.FAILED_ATTEMPT_PATH_V22A,
            evidence.COMPLETE_ATTEMPT_PATH_V22A,
            evidence.REPORT_PATH_V22A,
        )
    )


def reseal(value):
    value["content_sha256_before_self_field"] = evidence.canonical_sha256(
        evidence._without_self(value)
    )


def test_v22a_evidence_is_exact_deterministic_and_persisted():
    built = evidence.build_evidence_v22a(validate_launch_sources=True)
    persisted = json.loads(evidence.OUTPUT_PATH_V22A.read_text(encoding="utf-8"))
    assert built == persisted
    assert built["content_sha256_before_self_field"] == EXPECTED_CONTENT_SHA256
    assert evidence.file_sha256(evidence.OUTPUT_PATH_V22A) == EXPECTED_FILE_SHA256
    assert built["inputs"]["durable_cardinality"] == {
        "attempt_count": 2,
        "failed_attempt_count": 1,
        "completed_attempt_count": 1,
        "report_count": 1,
        "checkpoint_file_count": 0,
        "evaluation_file_count": 0,
    }
    assert built["source_provenance"]["source_file_count"] == 69
    assert built["execution"]["requests_all_engines_all_signed_waves"] == 61_440
    assert built["analysis"]["observed_pass_count"] == 5
    assert built["analysis"]["bootstrap_pass_count"] == 0
    assert built["recomputed_gate"]["compatibility_gate_passed"] is False


def test_v22a_run_documents_validate_independently_without_launch_file_reads():
    failed, complete, report = load_documents()
    validated = evidence.validate_run_documents_v22a(
        failed, complete, report,
        require_frozen_hashes=True, validate_launch_sources=False,
    )
    assert validated["gate"]["decision"] == "retain_production_dataset_and_v13_recipe"
    assert validated["forbidden_content_keys_found"] == []


@pytest.mark.parametrize(
    ("target", "key", "value", "message"),
    (
        ("audit", "requests_all_engines_all_signed_waves", 61_439, "runtime audit"),
        ("gate", "compatibility_gate_passed", True, "persisted gate"),
        ("summary", "row_content", [], "compact summary"),
        ("complete", "model_update_applied", True, "no-mutation authority"),
        ("failed", "report_exists_after_attempt", True, "fail-closed attempt"),
    ),
)
def test_v22a_rejects_resealed_run_tampering(target, key, value, message):
    failed, complete, report = load_documents()
    selected = {
        "audit": report["runtime_audit"],
        "gate": report["gate"],
        "summary": report["summary"],
        "complete": complete,
        "failed": failed,
    }[target]
    selected[key] = value
    if target in {"audit", "summary"}:
        reseal(selected)
    if target in {"audit", "gate", "summary"}:
        reseal(report)
        complete["report_binding"]["content_sha256"] = report[
            "content_sha256_before_self_field"
        ]
        reseal(complete)
    elif target == "complete":
        reseal(complete)
    else:
        reseal(failed)
    with pytest.raises(RuntimeError, match=message):
        evidence.validate_run_documents_v22a(
            failed, complete, report,
            require_frozen_hashes=False, validate_launch_sources=False,
        )


def test_v22a_rejects_resealed_source_provenance_tampering():
    failed, complete, report = load_documents()
    source = copy.deepcopy(complete["source_provenance"])
    source["git_head"] = "0" * 40
    reseal(source)
    failed["source_provenance"] = copy.deepcopy(source)
    complete["source_provenance"] = source
    reseal(failed)
    reseal(complete)
    with pytest.raises(RuntimeError, match="source provenance"):
        evidence.validate_run_documents_v22a(
            failed, complete, report,
            require_frozen_hashes=False, validate_launch_sources=False,
        )


def test_v22a_evidence_rejects_resealed_decision_tampering():
    value = json.loads(evidence.OUTPUT_PATH_V22A.read_text(encoding="utf-8"))
    value["decision"]["candidate_v341_replacement_promotion_authorized"] = True
    reseal(value)
    with pytest.raises(RuntimeError, match="aggregate-only negative evidence"):
        evidence.validate_evidence_v22a(value)


def test_v22a_evidence_contains_no_forbidden_detail_keys():
    value = json.loads(evidence.OUTPUT_PATH_V22A.read_text(encoding="utf-8"))
    assert not (
        evidence.FORBIDDEN_CONTENT_KEYS_V22A
        & set(evidence._recursive_keys(value))
    )
    assert value["aggregate_only_audit"] == {
        "forbidden_content_keys_found": [],
        "contains_row_question_answer_prompt_token_response_or_unit_scores": False,
        "contains_holdout_validation_ood_or_benchmark_content": False,
        "bootstrap_draws_or_replicates_persisted": False,
        "checkpoint_or_evaluation_files_created": False,
        "union_planner_called": False,
        "model_update_checkpoint_evaluation_or_promotion_applied": False,
    }


def test_v22a_evidence_write_is_scoped_and_immutable(tmp_path, monkeypatch):
    value = json.loads(evidence.OUTPUT_PATH_V22A.read_text(encoding="utf-8"))
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(evidence, "OUTPUT_PATH_V22A", output)
    evidence._exclusive_write(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    with pytest.raises(RuntimeError, match="already exists"):
        evidence._exclusive_write(output, value)
    with pytest.raises(ValueError, match="output path changed"):
        evidence._exclusive_write(tmp_path / "other.json", value)
