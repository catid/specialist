import copy
import json
from pathlib import Path

import pytest

import build_vllm_moe_fp8_tuning_failure_evidence_v29a as evidence


ROOT = Path("/home/catid/specialist")
ATTEMPT = ROOT / evidence.ATTEMPT_RELATIVE_PATH_V29A
REPORT = ROOT / evidence.REPORT_RELATIVE_PATH_V29A
OUTPUT_DIRECTORY = ROOT / evidence.OUTPUT_DIRECTORY_RELATIVE_PATH_V29A
RUNTIME = ROOT / evidence.RUNTIME_RELATIVE_PATH_V29A


def _build(
    attempt=ATTEMPT,
    report=REPORT,
    output_directory=OUTPUT_DIRECTORY,
    runtime=RUNTIME,
):
    return evidence.build_failure_evidence_v29a(
        attempt, report, output_directory, runtime,
    )


def _write_resealed_attempt(tmp_path, mutate):
    value = json.loads(ATTEMPT.read_text(encoding="utf-8"))
    mutate(value)
    value["content_sha256_before_self_field"] = evidence.prereg.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
    path = tmp_path / "attempt.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def test_v29a_failure_evidence_rebuilds_exactly_and_contains_only_hash_failure():
    rebuilt = _build()
    persisted = json.loads(evidence.OUTPUT_PATH_V29A.read_text(encoding="utf-8"))
    assert rebuilt == persisted
    failure = rebuilt["failed_attempt"]["failure_integrity"]
    assert set(failure) == {
        "exception_class", "message_sha256",
        "raw_message_or_traceback_persisted",
    }
    assert failure["exception_class"] == "ServerUnavailable"
    assert failure["raw_message_or_traceback_persisted"] is False
    assert rebuilt["failure_boundary"][
        "failure_before_official_tune_future_submission"
    ] is True
    assert rebuilt["failure_boundary"]["all_four_gpus_finally_idle"] is True
    assert rebuilt["closed_surfaces"]["evaluation_or_dataset_surface_opened"] is False
    assert "message" not in set(evidence._recursive_keys(rebuilt))
    assert "traceback" not in set(evidence._recursive_keys(rebuilt))


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value["failure"].__setitem__(
            "exception_class", "DifferentFailure"
        ),
        lambda value: value["failure"].__setitem__("message_sha256", "0" * 64),
        lambda value: value.__setitem__(
            "final_idle_certificate_sha256", "0" * 64
        ),
        lambda value: value.__setitem__("selected_table_written", True),
        lambda value: value.__setitem__(
            "dataset_train_evaluation_validation_heldout_ood_or_benchmark_surface_opened",
            True,
        ),
    ),
)
def test_v29a_failure_evidence_rejects_even_resealed_attempt_mutations(
    tmp_path, monkeypatch, mutate,
):
    changed = _write_resealed_attempt(tmp_path, mutate)
    monkeypatch.setattr(
        evidence,
        "EXPECTED_ATTEMPT_FILE_SHA256_V29A",
        evidence.file_sha256(changed),
    )
    monkeypatch.setattr(
        evidence,
        "EXPECTED_ATTEMPT_CONTENT_SHA256_V29A",
        json.loads(changed.read_text())["content_sha256_before_self_field"],
    )
    with pytest.raises(RuntimeError, match="evidence changed"):
        _build(attempt=changed)


def test_v29a_failure_evidence_rejects_runtime_or_output_surface_drift(tmp_path):
    changed_runtime = tmp_path / "runtime.py"
    changed_runtime.write_bytes(RUNTIME.read_bytes() + b"\n")
    with pytest.raises(RuntimeError, match="evidence changed"):
        _build(runtime=changed_runtime)
    changed_output = tmp_path / "output"
    changed_output.mkdir()
    (changed_output / "unexpected.json").write_text("{}")
    with pytest.raises(RuntimeError, match="evidence changed"):
        _build(output_directory=changed_output)


def test_v29a_failure_evidence_compact_gate_rejects_raw_payload_keys():
    for forbidden in ("message", "traceback", "compiler_log", "search_results"):
        with pytest.raises(RuntimeError, match="forbidden keys"):
            evidence._assert_compact_v29a({forbidden: "not allowed"})
