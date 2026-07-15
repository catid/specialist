#!/usr/bin/env python3
"""Aggregate-only tamper tests for V21A negative compatibility evidence."""

import copy
import json

import pytest

import build_eggroll_es_production_v331_patch_negative_evidence_v21a as evidence_v21a


def load_documents():
    return (
        json.loads(evidence_v21a.ATTEMPT_PATH_V21A.read_text(encoding="utf-8")),
        json.loads(evidence_v21a.REPORT_PATH_V21A.read_text(encoding="utf-8")),
    )


def reseal_documents(attempt, report):
    for key in ("recipe", "summary", "runtime_audit"):
        if isinstance(report.get(key), dict):
            report[key]["content_sha256_before_self_field"] = (
                evidence_v21a.canonical_sha256(
                    evidence_v21a._without_self(report[key])
                )
            )
    source = attempt.get("source_provenance")
    if isinstance(source, dict):
        source["content_sha256_before_self_field"] = (
            evidence_v21a.canonical_sha256(evidence_v21a._without_self(source))
        )
    attempt["recipe"] = copy.deepcopy(report["recipe"])
    report["content_sha256_before_self_field"] = evidence_v21a.canonical_sha256(
        evidence_v21a._without_self(report)
    )
    attempt["report_binding"]["content_sha256"] = report[
        "content_sha256_before_self_field"
    ]
    attempt["content_sha256_before_self_field"] = evidence_v21a.canonical_sha256(
        evidence_v21a._without_self(attempt)
    )


def test_v21a_persisted_evidence_is_exact_deterministic_and_aggregate_only():
    persisted = json.loads(evidence_v21a.OUTPUT_PATH_V21A.read_text(encoding="utf-8"))
    rebuilt = evidence_v21a.build_evidence_v21a(validate_launch_sources=True)
    assert persisted == rebuilt
    assert evidence_v21a.validate_evidence_v21a(persisted) == persisted
    assert evidence_v21a.file_sha256(evidence_v21a.OUTPUT_PATH_V21A) == (
        "ff457da7630ed22300135d5ec708cb026d126f828cd31175bd523686abac5f69"
    )
    assert persisted["content_sha256_before_self_field"] == (
        "0060d3853b4e15f3668671b6b752fb62a3e4c632fde380bf981063430b44a797"
    )
    assert persisted["analysis"]["observed_pass_count"] == 3
    assert persisted["analysis"]["bootstrap_pass_count"] == 0
    assert persisted["recomputed_gate"]["compatibility_gate_passed"] is False
    assert persisted["recomputed_gate"]["decision"] == (
        "retain_production_dataset_and_v13_recipe"
    )
    assert not (
        evidence_v21a.FORBIDDEN_CONTENT_KEYS_V21A
        & set(evidence_v21a._recursive_keys(persisted))
    )


def test_v21a_exact_run_has_one_attempt_one_report_and_56_launch_sources():
    attempt, report = evidence_v21a._load_exact_run_files_v21a()
    validated = evidence_v21a.validate_run_documents_v21a(
        attempt, report, require_frozen_hashes=True, validate_launch_sources=True
    )
    assert validated["source_audit"]["git_head"] == (
        evidence_v21a.LAUNCH_GIT_HEAD_V21A
    )
    assert validated["source_audit"]["source_file_count"] == 56
    assert validated["source_audit"]["all_source_files_match_launch_head"] is True
    assert validated["gate"]["observed_pass_count"] == 3
    assert validated["gate"]["bootstrap_pass_count"] == 0


@pytest.mark.parametrize(
    "tamper",
    (
        "summary_self", "observed_delta", "lcb_nonfinite", "nonzero_margin",
        "bootstrap_repetitions", "bootstrap_quantile", "gate_count",
        "signed_waves", "request_total", "commitment_count", "union_called",
        "model_update", "forbidden_content", "report_binding",
    ),
)
def test_v21a_documents_reject_aggregate_runtime_and_authority_tampering(tamper):
    attempt, report = load_documents()
    summary = report["summary"]
    audit = report["runtime_audit"]
    first = summary["paired_bootstrap"]["comparison"]["endpoints"][
        evidence_v21a.ENDPOINT_NAMES_V21A[0]
    ]
    if tamper == "summary_self":
        summary["content_sha256_before_self_field"] = "0" * 64
    elif tamper == "observed_delta":
        first["treatment_minus_control"] += 0.001
        reseal_documents(attempt, report)
    elif tamper == "lcb_nonfinite":
        first["familywise_lcb"] = float("inf")
        reseal_documents(attempt, report)
    elif tamper == "nonzero_margin":
        first["noninferiority_margin"] = 0.001
        reseal_documents(attempt, report)
    elif tamper == "bootstrap_repetitions":
        summary["paired_bootstrap"]["repetitions"] = 49_999
        reseal_documents(attempt, report)
    elif tamper == "bootstrap_quantile":
        summary["paired_bootstrap"]["one_sided_quantile"] = 0.05
        reseal_documents(attempt, report)
    elif tamper == "gate_count":
        report["gate"]["observed_pass_count"] = 4
        reseal_documents(attempt, report)
    elif tamper == "signed_waves":
        audit["signed_wave_count"] = 31
        reseal_documents(attempt, report)
    elif tamper == "request_total":
        audit["requests_per_engine_all_signed_waves"] = 17_279
        reseal_documents(attempt, report)
    elif tamper == "commitment_count":
        audit["dense_result_commitment_count"] = 2_559
        reseal_documents(attempt, report)
    elif tamper == "union_called":
        report["union_planner_called"] = True
        reseal_documents(attempt, report)
    elif tamper == "model_update":
        report["model_update_applied"] = True
        reseal_documents(attempt, report)
    elif tamper == "forbidden_content":
        audit["questions"] = ["forbidden"]
        reseal_documents(attempt, report)
    elif tamper == "report_binding":
        attempt["report_binding"]["file_sha256"] = "0" * 64
        attempt["content_sha256_before_self_field"] = (
            evidence_v21a.canonical_sha256(evidence_v21a._without_self(attempt))
        )
    with pytest.raises(RuntimeError):
        evidence_v21a.validate_run_documents_v21a(
            attempt, report, require_frozen_hashes=False,
            validate_launch_sources=False,
        )


def test_v21a_rejects_a_fully_resealed_false_launch_source_binding():
    attempt, report = load_documents()
    key = next(iter(report["implementation"]["files"]))
    false_digest = "0" * 64
    report["implementation"]["files"][key]["file_sha256"] = false_digest
    report["implementation"]["bundle_sha256"] = evidence_v21a.canonical_sha256(
        report["implementation"]["files"]
    )
    bundle = report["implementation"]["bundle_sha256"]
    report["recipe"]["implementation_bundle_sha256"] = bundle
    attempt["source_provenance"]["files"][key]["file_sha256"] = false_digest
    attempt["source_provenance"]["implementation_bundle_sha256"] = bundle
    reseal_documents(attempt, report)
    with pytest.raises(RuntimeError, match="launch source file digest changed"):
        evidence_v21a.validate_run_documents_v21a(
            attempt, report, require_frozen_hashes=False,
            validate_launch_sources=True,
        )


@pytest.mark.parametrize(
    "tamper",
    ("decision", "source_count", "endpoint_margin", "forbidden_audit", "self"),
)
def test_v21a_evidence_rejects_compact_artifact_tampering(tamper):
    value = json.loads(evidence_v21a.OUTPUT_PATH_V21A.read_text(encoding="utf-8"))
    if tamper == "decision":
        value["decision"]["compatibility_gate_passed"] = True
    elif tamper == "source_count":
        value["source_provenance"]["source_file_count"] = 55
    elif tamper == "endpoint_margin":
        first = next(iter(value["analysis"]["endpoints"].values()))
        first["noninferiority_margin"] = 0.001
    elif tamper == "forbidden_audit":
        value["aggregate_only_audit"]["forbidden_content_keys_found"] = [
            "questions"
        ]
    elif tamper == "self":
        value["content_sha256_before_self_field"] = "0" * 64
    if tamper != "self":
        value["content_sha256_before_self_field"] = evidence_v21a.canonical_sha256(
            evidence_v21a._without_self(value)
        )
    with pytest.raises(RuntimeError, match="negative evidence changed"):
        evidence_v21a.validate_evidence_v21a(value)
