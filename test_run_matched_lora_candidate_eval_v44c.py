#!/usr/bin/env python3

import json

import pytest

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v44c as subject


def test_explicit_only_source_fields_are_accepted_v44c():
    row = {"question": "What is the fact?", "answer": "The source answer."}
    assert subject.qa_pair_from_record_v44c(row) == (
        "What is the fact?", "The source answer."
    )


def test_strict_serialization_must_agree_with_explicit_fields_v44c():
    row = {
        "question": "What is the fact?", "answer": "The source answer.",
        "text": "Question: What is the fact?\nAnswer: The source answer.",
    }
    assert subject.qa_pair_from_record_v44c(row) == (
        "What is the fact?", "The source answer."
    )
    row["text"] = "Question: What is the fact?\nAnswer: A different answer."
    with pytest.raises(ValueError, match="disagree"):
        subject.qa_pair_from_record_v44c(row)


@pytest.mark.parametrize("row,match", [
    ({"question": "Only one"}, "only one"),
    ({"answer": "Only one"}, "only one"),
    ({"question": "Q", "answer": "<think>bad</think>"}, "protocol"),
    ({"question": "Q", "answer": "A", "text": "arbitrary prose"}, "not a supported"),
    ({"text": 7}, "must be a string"),
    ({}, "neither explicit"),
])
def test_parser_fails_closed_v44c(row, match):
    with pytest.raises(ValueError, match=match):
        subject.qa_pair_from_record_v44c(row)


def test_all_authorized_schemas_preflight_once_v44c():
    value = subject.offline_authorized_schema_audit_v44c()
    assert value["status"] == "complete_before_model_creation"
    assert value["all_four_authorized_inputs_parsed_once"] is True
    assert value["holdout_or_heldout_opened"] is False
    assert value["semantic_access_counts"] == {
        "split_manifest": 1, "shadow": 1, "ood_qa": 1, "ood_prose": 1,
    }
    receipts = {item["label"]: item for item in value["schema_receipts"]}
    assert receipts["shadow"]["rows"] == 83
    assert receipts["ood_qa"]["rows"] == 24
    assert receipts["ood_qa"]["serialization"] == (
        "explicit_question_answer_fields_no_text_field"
    )
    assert receipts["ood_prose"]["rows"] == 16


def test_v44b_failure_consumed_no_holdout_and_persisted_no_metrics_v44c():
    value = subject.failed_launch_provenance_v44c()
    assert value["known_protected_semantic_access_order"] == [
        "split_manifest", "shadow", "ood_qa"
    ]
    assert value["ood_prose_known_unread_by_control_flow"] is True
    assert value["heldout_or_holdout_opened"] is False
    assert value["raw_item_artifact_persisted"] is False
    assert value["aggregate_report_persisted"] is False
    assert value["shadow_metrics_persisted_or_observed"] is False


def test_preflight_hook_requires_exact_preregistered_aggregate_v44c():
    expected = subject.offline_authorized_schema_audit_v44c()
    firewall = core.SingleSemanticAccessV44A(core.PROTECTED_INPUTS_V44A)
    result = subject.protected_preflight_v44c(
        firewall, {"cpu_preflight_expected": expected}
    )
    assert result["status"] == "complete_before_model_creation"
    firewall = core.SingleSemanticAccessV44A(core.PROTECTED_INPUTS_V44A)
    with pytest.raises(RuntimeError, match="identity changed"):
        subject.protected_preflight_v44c(
            firewall, {"cpu_preflight_expected": {}}
        )


def test_v44c_paths_and_no_holdout_binding_v44c():
    paths = {subject.RUN_DIR, subject.ATTEMPT, subject.RAW,
             subject.GPU_LOG, subject.REPORT}
    # Path validation must remain repeatable after the preregistered run has
    # produced its artifacts.  Freshness is enforced by the launcher's
    # attempt-marker interlock, not by a stateful unit-test assertion.
    assert subject.RUN_DIR.name.startswith("v44c_")
    assert subject.ATTEMPT.name.startswith(".v44c_")
    assert subject.RAW.parent == subject.RUN_DIR
    assert subject.GPU_LOG.parent == subject.RUN_DIR
    assert subject.REPORT.parent.name == "eval_reports"
    assert all("holdout" not in str(path).casefold() for path in paths)
    protected = [item["path"] for item in core.PROTECTED_INPUTS_V44A.values()]
    assert all("holdout" not in path.casefold() for path in protected)


def test_builder_and_loader_preserve_honest_schema_audit_receipt_v44c(
    tmp_path, monkeypatch
):
    import build_matched_lora_candidate_eval_preregistration_v44c as builder

    value = builder.build()
    assert value["schema_only_authorized_inputs_opened_before_preregistration"]
    assert value["evaluation_metrics_observed_before_v44c_preregistration"] is False
    assert value["heldout_or_holdout_opened_during_schema_audit"] is False
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(candidate):
        assert str(candidate) not in protected
        return original(candidate)

    monkeypatch.setattr(core, "file_sha256", guarded)
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": original(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    loaded = subject.load_preregistration_v44c(args)
    assert loaded["cpu_preflight_expected"] == value["cpu_preflight_expected"]
