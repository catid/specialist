#!/usr/bin/env python3

import copy
import json

import pytest

import build_vllm_moe_fp8_selected_table_v29d_failure_evidence_v29e as evidence


def test_v29e_failure_evidence_binds_exact_failed_attempt_and_v29d_commit():
    value = evidence.build_failure_evidence_v29e()
    assert value["failed_attempt"]["file_sha256"] == (
        "1dabc51d07b0728f2f0492a4404fdd3b8ab61127d26d2c5f693dd8bcca4bb08f"
    )
    assert value["failed_attempt"]["content_sha256"] == (
        "3d7f133610cb8eca9a0ce5ffb930170f627b746d18ff56fce04f73c9c0a76142"
    )
    assert value["contracts"]["v29d_preregistration_commit"] == (
        "23d6bd0f7d90a7438488e55eb796928b9a0bfd31"
    )
    assert value["failed_attempt"]["failure_message_sha256"] == (
        "c0dd55cfb46132f2d48cdcaea5cb128d93f183d60e816fd9eff93b5a2b7c97c8"
    )


def test_v29e_failure_is_observability_only_final_idle_and_no_report_or_adoption():
    value = evidence.build_failure_evidence_v29e()
    boundary = value["failure_boundary"]
    assert boundary["activity_observability_gate_failed"] is True
    assert boundary["kernel_statistical_evaluation_completed"] is False
    assert boundary["evaluation_report_written"] is False
    assert boundary["final_idle_certificate_present"] is True
    assert boundary["direct_table_adoption_or_action_taken"] is False
    assert value["decision"]["selected_table_adoption_authorized"] is False
    assert value["decision"]["authorize_only_exact_observability_retry_preregistration"] is True


def test_v29e_changed_attempt_bytes_fail_closed(monkeypatch, tmp_path):
    changed = tmp_path / evidence.ATTEMPT_PATH_V29E.name
    changed.write_bytes(evidence.ATTEMPT_PATH_V29E.read_bytes() + b"\n")
    monkeypatch.setattr(evidence, "ATTEMPT_PATH_V29E", changed)
    with pytest.raises(RuntimeError, match="attempt file identity changed"):
        evidence.validate_v29d_failure_v29e()


def test_v29e_report_presence_fails_closed(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(evidence, "REPORT_PATH_V29E", report)
    with pytest.raises(RuntimeError, match="report to be absent"):
        evidence.validate_v29d_failure_v29e()


def test_v29e_attempt_semantic_tamper_fails_self_hash():
    attempt = json.loads(evidence.ATTEMPT_PATH_V29E.read_text(encoding="utf-8"))
    changed = copy.deepcopy(attempt)
    changed["direct_action_taken"] = True
    with pytest.raises(RuntimeError, match="attempt self hash changed"):
        evidence._verify_self(
            changed, evidence.ATTEMPT_CONTENT_SHA256_V29E, "V29D attempt",
        )


def test_v29e_compact_failure_evidence_rejects_raw_payload_keys():
    for key in ("question", "responses", "timing_vectors", "raw_pids", "traceback"):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            evidence._assert_compact_v29e({key: []})


def test_v29e_failure_evidence_build_is_deterministic_and_dry(capsys):
    first = evidence.build_failure_evidence_v29e()
    second = evidence.main(["--dry-run"])
    output = json.loads(capsys.readouterr().out)
    assert first == second
    assert output["content_sha256"] == first["content_sha256_before_self_field"]
    assert output["gpu_launched"] is False
