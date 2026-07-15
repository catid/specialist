import json
from pathlib import Path

import pytest

import build_eggroll_es_v23a_probe_context_failure_evidence_r3 as evidence_r3


MAIN_RUNS = Path("/home/catid/specialist/experiments/eggroll_es_hpo/runs")
MAIN_ATTEMPT = MAIN_RUNS / evidence_r3.ATTEMPT_RELATIVE_PATH_R3.split("runs/", 1)[1]
MAIN_REPORT = MAIN_RUNS / evidence_r3.REPORT_RELATIVE_PATH_R3.split("runs/", 1)[1]


def test_v23a_r3_evidence_exactly_rebuilds_compact_r2_failure():
    rebuilt = evidence_r3.build_probe_context_failure_evidence_r3(
        MAIN_ATTEMPT, MAIN_REPORT
    )
    persisted = json.loads(evidence_r3.OUTPUT_PATH_R3.read_text(encoding="utf-8"))
    assert rebuilt == persisted
    assert rebuilt["failed_attempt"]["file_sha256"] == (
        evidence_r3.EXPECTED_ATTEMPT_FILE_SHA256_R3
    )
    assert rebuilt["failed_attempt"]["compact_report_absent"] is True
    boundary = rebuilt["proven_control_flow_boundary"]
    assert boundary["all_64_selected_restore_verifications_completed"] is True
    assert boundary[
        "full_selected_and_unselected_population_boundary_audit_completed"
    ] is True
    diagnosis = rebuilt["probe_context_diagnosis"]
    assert diagnosis["pre_probe_source"].endswith("280_request_batch")
    assert diagnosis["post_probe_source"].endswith("1_request_batch")
    assert diagnosis["generation_batch_shape_and_order_same"] is False
    assert rebuilt["traceback_or_model_repr_persisted"] is False
    assert rebuilt["row_response_or_score_content_persisted"] is False
    assert not (
        evidence_r3.FORBIDDEN_KEYS_R3
        & set(evidence_r3._recursive_keys(rebuilt))
    )


def test_v23a_r3_evidence_rejects_attempt_mutation_or_report(tmp_path):
    attempt = json.loads(MAIN_ATTEMPT.read_text(encoding="utf-8"))
    attempt["model_update_applied"] = True
    changed = tmp_path / "attempt.json"
    changed.write_text(json.dumps(attempt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence changed"):
        evidence_r3.build_probe_context_failure_evidence_r3(
            changed, tmp_path / "absent.json"
        )
    report = tmp_path / "report.json"
    report.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="evidence changed"):
        evidence_r3.build_probe_context_failure_evidence_r3(MAIN_ATTEMPT, report)


def test_v23a_r3_evidence_forbids_verbose_or_score_payloads():
    for key in ("traceback", "message", "question", "responses", "score_arrays"):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            evidence_r3._assert_compact({key: "not allowed"})
