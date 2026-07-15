import json
from pathlib import Path

import pytest

import build_eggroll_es_v24a_memory_failure_evidence_r1 as evidence


MAIN = Path("/home/catid/specialist")
ATTEMPT = MAIN / evidence.ATTEMPT_RELATIVE_PATH_R1
REPORT = MAIN / evidence.REPORT_RELATIVE_PATH_R1
MODEL_RUNNER = MAIN / evidence.VLLM_MODEL_RUNNER_RELATIVE_PATH_R1
GPU_WORKER = MAIN / evidence.VLLM_GPU_WORKER_RELATIVE_PATH_R1
ORIGINAL_RUNTIME = MAIN / evidence.ORIGINAL_RUNTIME_RELATIVE_PATH_R1


def _build():
    return evidence.build_memory_failure_evidence_r1(
        ATTEMPT, REPORT, MODEL_RUNNER, GPU_WORKER, ORIGINAL_RUNTIME,
    )


def test_v24a_r1_rebuilds_compact_memory_failure_evidence_exactly():
    rebuilt = _build()
    persisted = json.loads(evidence.OUTPUT_PATH_R1.read_text(encoding="utf-8"))
    assert rebuilt == persisted
    assert rebuilt["failed_attempt"]["failure_type"] == "SystemExit"
    assert rebuilt["failed_attempt"]["compact_report_absent"] is True
    assert rebuilt["invalid_original_memory_endpoint"][
        "forty_percent_reduction_gate_capable"
    ] is False
    assert rebuilt["authoritative_replacement_endpoint"][
        "assignment_semantics"
    ] == "self.model_memory_usage = m.consumed_memory"
    assert rebuilt["retry_scope"][
        "abort_before_reference_a_b_or_first_perturbation_on_failure"
    ] is True
    assert rebuilt["authority"]["dataset_content_inspected"] is False


def test_v24a_r1_evidence_fails_closed_on_source_or_attempt_drift(tmp_path):
    changed_source = tmp_path / "gpu_model_runner.py"
    changed_source.write_bytes(MODEL_RUNNER.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="evidence changed"):
        evidence.build_memory_failure_evidence_r1(
            ATTEMPT, REPORT, changed_source, GPU_WORKER, ORIGINAL_RUNTIME,
        )
    changed_attempt = tmp_path / "attempt.json"
    changed_attempt.write_bytes(ATTEMPT.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="evidence changed"):
        evidence.build_memory_failure_evidence_r1(
            changed_attempt, REPORT, MODEL_RUNNER, GPU_WORKER, ORIGINAL_RUNTIME,
        )
