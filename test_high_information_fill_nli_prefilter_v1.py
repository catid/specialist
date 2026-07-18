from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

import run_high_information_fill_nli_prefilter_v1 as fill_nli
import run_high_information_nli_prefilter_v1 as primary_nli


def packet(*, negative: bool = False) -> dict:
    return {
        "packet_id": "semantic-packet-v1:" + "1" * 64,
        "candidate_example_id": "candidate-synthetic",
        "request_id": "request-synthetic",
        "source_group_id": "group-synthetic",
        "generation_mode": (
            "calibrated_hard_negative" if negative else "positive"
        ),
        "evidence_quotes": ["Synthetic evidence supports the bounded fact."],
        "answer": (
            "The source cannot establish the requested claim."
            if negative
            else "The bounded fact is supported."
        ),
    }


def synthetic_fill_contract(shard_index: int = 0) -> dict:
    candidate_path, report_path = fill_nli.generation_pass.generation_paths(
        fill_nli.PASS_ID, shard_index
    )
    value = {
        "schema": "high-information-generation-pass-contract-v1",
        "generation_pass_id": fill_nli.PASS_ID,
        "gpu_shard": shard_index,
        "candidate_path": fill_nli.corpus.relative(candidate_path),
        "candidate_file_sha256": "1" * 64,
        "generation_report_path": fill_nli.corpus.relative(report_path),
        "generation_report_file_sha256": "2" * 64,
        "generation_report_self_sha256": "3" * 64,
        "worker_path": "run_high_information_candidate_fill_shard_v1.py",
        "worker_file_sha256": fill_nli.corpus.file_sha256(
            fill_nli.corpus.ROOT / "run_high_information_candidate_fill_shard_v1.py"
        ),
        "runtime_worker_receipts": fill_nli._expected_fill_runtime_receipts(),
        "temperature": fill_nli.FILL_TEMPERATURE,
        "top_p": fill_nli.FILL_TOP_P,
        "seed_scheme": fill_nli.FILL_SEED_SCHEME,
        "prompt_spec_sha256": "4" * 64,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    value["content_sha256_before_self_field"] = fill_nli._self_address(value)
    return value


def test_outputs_are_separate_from_primary_nli_lane():
    for shard in range(4):
        fill_paths = set(fill_nli.output_paths(shard).values())
        primary_paths = set(primary_nli.shard_paths(shard).values())
        assert not fill_paths & primary_paths
        assert all(fill_nli.PASS_SLUG in path.name for path in fill_paths)


def test_positive_result_reuses_nli_semantics_but_binds_fill_pass():
    source = packet()
    result = fill_nli.make_result(
        source,
        probabilities={"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
        input_token_count=42,
        input_truncated=False,
        run_contract_sha256="a" * 64,
        generation_pass_contract_sha256="b" * 64,
    )
    fill_nli.validate_result(result, source, "a" * 64, "b" * 64)
    assert result["verdict"] == "pass"
    assert result["generation_pass_id"] == fill_nli.PASS_ID
    assert result["semantic_verification_completed"] is False
    assert result["eligible_for_training"] is False

    tampered = copy.deepcopy(result)
    tampered["generation_pass_id"] = "primary-generation-v1"
    tampered["content_sha256_before_self_field"] = fill_nli._self_address(tampered)
    with pytest.raises(RuntimeError, match="identity changed"):
        fill_nli.validate_result(tampered, source, "a" * 64, "b" * 64)


def test_hard_negative_remains_not_applicable_and_training_ineligible():
    source = packet(negative=True)
    result = fill_nli.make_result(
        source,
        probabilities=None,
        input_token_count=None,
        input_truncated=False,
        run_contract_sha256="a" * 64,
        generation_pass_contract_sha256="b" * 64,
    )
    fill_nli.validate_result(result, source, "a" * 64, "b" * 64)
    assert result["verdict"] == "not_applicable"
    assert result["eligible_for_training"] is False
    assert result["manual_review_reasons"] == [
        "hard_negative_requires_absence_or_false_premise_verifier"
    ]


def test_fill_contract_requires_exact_config_and_both_runtime_receipts():
    contract = synthetic_fill_contract()
    fill_nli.validate_fill_generation_contract(contract, copy.deepcopy(contract), 0)
    assert len(contract["runtime_worker_receipts"]) == 2

    drifted = copy.deepcopy(contract)
    drifted["temperature"] = 0.3
    drifted["content_sha256_before_self_field"] = fill_nli._self_address(drifted)
    with pytest.raises(RuntimeError, match="differs from runtime receipt"):
        fill_nli.validate_fill_generation_contract(drifted, contract, 0)

    missing_receipt = copy.deepcopy(contract)
    missing_receipt["runtime_worker_receipts"] = missing_receipt[
        "runtime_worker_receipts"
    ][:1]
    missing_receipt["content_sha256_before_self_field"] = fill_nli._self_address(
        missing_receipt
    )
    with pytest.raises(RuntimeError, match="differs from runtime receipt"):
        fill_nli.validate_fill_generation_contract(missing_receipt, contract, 0)


def test_run_contract_binds_fill_model_runtime_and_planned_outputs():
    pass_contract = synthetic_fill_contract()
    summary = {
        "generation_pass_contract": pass_contract,
        "report_path": "synthetic/structural.jsonl",
        "report_file_sha256": "5" * 64,
        "content_sha256_before_self_field": "6" * 64,
    }
    model_receipts = {
        "synthetic.model": {
            "snapshot_blob_id": "blob",
            "file_sha256": "7" * 64,
            "file_bytes": 10,
            "required_for_local_runtime": True,
        }
    }
    paths = fill_nli.output_paths(0)
    contract = fill_nli.build_run_contract(
        shard_index=0,
        structural_summary=summary,
        model_receipts=model_receipts,
        paths=paths,
        smoke=False,
    )
    assert contract["generation_pass"]["id"] == fill_nli.PASS_ID
    assert contract["generation_pass"]["runtime_worker_receipts"] == pass_contract[
        "runtime_worker_receipts"
    ]
    assert len(contract["generation_pass"]["runtime_worker_receipts"]) == 2
    assert contract["model"]["file_receipts"] == model_receipts
    assert contract["runtime"]["torch_version"] == fill_nli.TORCH_VERSION
    assert contract["runtime"]["python_executable"] == str(
        fill_nli.RUNTIME_PYTHON_EXECUTABLE
    )
    assert contract["runtime"]["ld_library_path"] == str(
        fill_nli.CU13_LIBRARY_PATH
    )
    assert contract["planned_outputs"]["receipt"].endswith(".receipt.json")
    assert contract["training_rows_emitted"] is False
    assert contract["content_sha256_before_self_field"] == fill_nli._self_address(
        contract
    )


def test_model_snapshot_rejects_matching_link_with_tampered_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "expected-blob"
    blob.write_bytes(b"tampered")
    (tmp_path / "config.json").symlink_to(blob)
    monkeypatch.setattr(fill_nli, "MODEL_DIRECTORY", tmp_path.resolve())
    monkeypatch.setattr(
        fill_nli, "MODEL_BLOB_RECEIPTS", {"config.json": "expected-blob"}
    )
    monkeypatch.setattr(
        fill_nli,
        "MODEL_FILE_SHA256",
        {"config.json": hashlib.sha256(b"expected").hexdigest()},
    )
    with pytest.raises(RuntimeError, match="model file content changed"):
        fill_nli.validate_model_snapshot(tmp_path)


def test_runtime_environment_rejects_a_different_python(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(fill_nli.sys, "executable", "/synthetic/wrong/python")
    with pytest.raises(RuntimeError, match="es-at-scale Python environment"):
        fill_nli.validate_runtime_environment()


def test_output_receipt_binds_output_and_report_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fill_nli.corpus, "ROOT", tmp_path.resolve())
    paths = {
        name: tmp_path / f"synthetic.{name}"
        for name in ("partial", "output", "report", "receipt", "telemetry")
    }
    report = {
        "schema": fill_nli.REPORT_SCHEMA,
        "run_contract": {"content_sha256_before_self_field": "a" * 64},
        "training_rows_emitted": False,
    }
    report["content_sha256_before_self_field"] = fill_nli._self_address(report)
    output_payload = b'{"synthetic":true}\n'
    report_payload = (
        fill_nli.json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode()
    receipt = fill_nli.build_output_receipt(
        shard_index=0,
        paths=paths,
        output_payload=output_payload,
        report_payload=report_payload,
        report=report,
    )
    assert receipt["output_sha256"] == hashlib.sha256(output_payload).hexdigest()
    assert receipt["report_file_sha256"] == hashlib.sha256(
        report_payload
    ).hexdigest()
    assert receipt["training_rows_emitted"] is False
    assert receipt["content_sha256_before_self_field"] == fill_nli._self_address(
        receipt
    )
