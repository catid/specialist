#!/usr/bin/env python3
"""Run an isolated, pass-aware NLI prefilter over the quality fill pass.

This worker deliberately does not share output paths with the primary NLI or
semantic-judge lanes.  It reuses the pinned NLI scoring semantics while binding
every result to the quality-deficit-fill generation-pass contract, both fill
runtime worker receipts, exact sampling configuration, structural-review
identity, model file contents, runtime implementations, and sealed output and
report hashes.  NLI is never semantic acceptance and every row remains
training-ineligible.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence

import build_high_information_domain_corpus_v1 as corpus
import run_high_information_nli_prefilter_v1 as nli
import verify_high_information_candidates_v1 as structural
import verify_high_information_generation_pass_v1 as generation_pass
import verify_high_information_semantic_decisions_v1 as semantic


PASS_ID = "quality-deficit-fill-v1"
PASS_SLUG = "quality_deficit_fill_v1"
RESULT_SCHEMA = "high-information-pass-aware-fill-nli-result-v1"
RUN_CONTRACT_SCHEMA = "high-information-pass-aware-fill-nli-run-contract-v1"
REPORT_SCHEMA = "high-information-pass-aware-fill-nli-report-v1"
RECEIPT_SCHEMA = "high-information-pass-aware-fill-nli-output-receipt-v1"
TELEMETRY_SCHEMA = "high-information-pass-aware-fill-nli-telemetry-v1"

# The blob identities are the snapshot link targets; the SHA-256 map binds the
# bytes actually loaded by the tokenizer and model runtime, including files
# whose Hugging Face cache IDs use Git SHA-1 rather than content SHA-256.
MODEL_ID = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
MODEL_REVISION = "b3546ea6b0346eb6f8d5d68b13c7dc6d0376b3d7"
MODEL_DIRECTORY = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--MoritzLaurer--DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    / "snapshots"
    / MODEL_REVISION
).resolve()
MODEL_BLOB_RECEIPTS = {
    "config.json": "4132f0331b3d94b85de11ade18e48e0dae07d2d2",
    "tokenizer_config.json": "60b134d3ec353ffa99d215bd626f91be3c7ac7de",
    "tokenizer.json": "08e1fd9af8a65c4ab19cb0b3f6e1a83ee911f8fb",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "model.safetensors": "c03cd208bf920b4fbbb182a0535859a566e8acc3477d2b536bf87b769978524b",
}
MODEL_FILE_SHA256 = {
    "config.json": "7f9d420b616691c5b575bab8839719aefbddc656d8e1316519414b097ce43e3d",
    "tokenizer_config.json": "556a192d83e3edb45a09a6207095dd9295c3b2279062e5cc430feac4cb7a0817",
    "tokenizer.json": "7aa118770f066a74530d161c7d0b994d0629cc0ff3a0df213f184192773f960a",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "model.safetensors": "c03cd208bf920b4fbbb182a0535859a566e8acc3477d2b536bf87b769978524b",
}

TRANSFORMERS_VERSION = "5.13.1"
TORCH_VERSION = "2.11.0+cu130"
DTYPE = "bfloat16"
RUNTIME_PYTHON_EXECUTABLE = (
    corpus.ROOT / "es-at-scale/.venv/bin/python"
).absolute()
RUNTIME_VIRTUAL_ENV = (corpus.ROOT / "es-at-scale/.venv").absolute()
CU13_LIBRARY_PATH = (
    corpus.ROOT
    / "es-at-scale/.venv/lib/python3.12/site-packages/nvidia/cu13/lib"
).resolve()
MAX_LENGTH = 512
ENTAILMENT_MINIMUM = 0.70
CONTRADICTION_MAXIMUM = 0.10
CONTRADICTION_FAILURE = 0.50
FILL_TEMPERATURE = 0.4
FILL_TOP_P = 0.9
FILL_SEED_SCHEME = (
    "sha256('quality-deficit-fill-v1:' + request_id) first 32 bits"
)

RESULT_FIELDS = {
    "schema",
    "packet_id",
    "candidate_example_id",
    "request_id",
    "source_group_id",
    "generation_mode",
    "generation_pass_id",
    "generation_pass_contract_sha256",
    "model",
    "premise_construction",
    "premise_sha256",
    "hypothesis_sha256",
    "input_token_count_before_truncation",
    "max_length",
    "input_truncated",
    "probabilities",
    "thresholds",
    "run_contract_sha256",
    "verdict",
    "manual_review_required",
    "manual_review_reasons",
    "semantic_verification_completed",
    "eligible_for_training",
    "content_sha256_before_self_field",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _self_address(value: dict) -> str:
    unsigned = dict(value)
    unsigned.pop("content_sha256_before_self_field", None)
    return corpus.canonical_sha256(unsigned)


def _require_self_address(value: dict, label: str) -> None:
    if value.get("content_sha256_before_self_field") != _self_address(value):
        raise RuntimeError(f"{label} content address changed")


def _regular_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if path.is_symlink() or not resolved.is_file():
        raise RuntimeError(f"{label} must be a non-symlink regular file")
    return resolved


def validate_reused_nli_semantics() -> dict[str, Any]:
    observed = {
        "model_id": nli.MODEL_ID,
        "model_revision": nli.MODEL_REVISION,
        "model_blob_receipts": nli.MODEL_BLOB_RECEIPTS,
        "transformers_version": nli.TRANSFORMERS_VERSION,
        "torch_version": nli.TORCH_VERSION,
        "max_length": nli.MAX_LENGTH,
        "entailment_minimum": nli.ENTAILMENT_MINIMUM,
        "contradiction_maximum": nli.CONTRADICTION_MAXIMUM,
        "contradiction_failure": nli.CONTRADICTION_FAILURE,
        "premise_construction": "ordered_exact_evidence_quotes_joined_by_newline",
    }
    expected = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_blob_receipts": MODEL_BLOB_RECEIPTS,
        "transformers_version": TRANSFORMERS_VERSION,
        "torch_version": TORCH_VERSION,
        "max_length": MAX_LENGTH,
        "entailment_minimum": ENTAILMENT_MINIMUM,
        "contradiction_maximum": CONTRADICTION_MAXIMUM,
        "contradiction_failure": CONTRADICTION_FAILURE,
        "premise_construction": "ordered_exact_evidence_quotes_joined_by_newline",
    }
    if observed != expected:
        raise RuntimeError("reused NLI semantics changed")
    return observed


def validate_model_snapshot(model_directory: Path) -> dict[str, dict[str, Any]]:
    model_directory = model_directory.expanduser().resolve()
    if model_directory != MODEL_DIRECTORY:
        raise RuntimeError("fill NLI requires the pinned local model revision")
    if set(MODEL_BLOB_RECEIPTS) != set(MODEL_FILE_SHA256):
        raise RuntimeError("fill NLI model receipt coverage changed")
    observed: dict[str, dict[str, Any]] = {}
    for name, expected_blob in MODEL_BLOB_RECEIPTS.items():
        path = model_directory / name
        if not path.is_symlink() or not path.is_file():
            raise RuntimeError(f"pinned fill NLI snapshot file is missing: {name}")
        target = Path(os.readlink(path)).name
        if target != expected_blob:
            raise RuntimeError(f"pinned fill NLI snapshot link changed: {name}")
        digest = corpus.file_sha256(path)
        if digest != MODEL_FILE_SHA256[name]:
            raise RuntimeError(f"pinned fill NLI model file content changed: {name}")
        observed[name] = {
            "snapshot_blob_id": target,
            "file_sha256": digest,
            "file_bytes": path.stat().st_size,
            "required_for_local_runtime": True,
        }
    return observed


def validate_runtime_environment() -> dict[str, str]:
    python_executable = Path(sys.executable).absolute()
    if python_executable != RUNTIME_PYTHON_EXECUTABLE:
        raise RuntimeError(
            "fill NLI requires the pinned es-at-scale Python environment"
        )
    virtual_env = Path(sys.prefix).absolute()
    if virtual_env != RUNTIME_VIRTUAL_ENV:
        raise RuntimeError("fill NLI virtual-environment identity changed")
    observed_library_path = os.environ.get("LD_LIBRARY_PATH")
    if observed_library_path != str(CU13_LIBRARY_PATH):
        raise RuntimeError("fill NLI requires the pinned cu13 LD_LIBRARY_PATH")
    import torch
    import transformers
    package_versions = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
    }
    if package_versions != {
        "transformers": TRANSFORMERS_VERSION,
        "torch": TORCH_VERSION,
    }:
        raise RuntimeError("fill NLI runtime package version changed")
    return {
        "python_executable": str(python_executable),
        "virtual_environment": str(virtual_env),
        "ld_library_path": observed_library_path,
        "transformers_version": package_versions["transformers"],
        "torch_version": package_versions["torch"],
        "dtype": DTYPE,
    }


def output_paths(shard_index: int, *, smoke: bool = False) -> dict[str, Path]:
    if shard_index not in range(4):
        raise ValueError("fill NLI shard must be 0-3")
    suffix = ".smoke" if smoke else ""
    stem = f"nli_prefilter_{PASS_SLUG}_gpu{shard_index}{suffix}"
    return {
        "partial": corpus.OUTPUT_DIR / f"{stem}.partial.jsonl",
        "output": corpus.OUTPUT_DIR / f"{stem}.jsonl",
        "report": corpus.OUTPUT_DIR / f"{stem}.report.json",
        "receipt": corpus.OUTPUT_DIR / f"{stem}.receipt.json",
        "telemetry": corpus.OUTPUT_DIR / f"{stem}.telemetry.json",
    }


def structural_paths(shard_index: int) -> tuple[Path, Path]:
    return generation_pass.structural_paths(PASS_ID, shard_index)


def _expected_fill_runtime_receipts() -> list[dict[str, str]]:
    spec = generation_pass.PASS_SPECS.get(PASS_ID)
    if spec is None:
        raise RuntimeError("quality fill generation pass is no longer registered")
    return [
        {
            "path": corpus.relative(Path(path).resolve()),
            "file_sha256": corpus.file_sha256(Path(path).resolve()),
        }
        for path in spec["runtime_workers"]
    ]


def validate_fill_generation_contract(
    contract: dict, expected_contract: dict, shard_index: int
) -> None:
    _require_self_address(contract, "fill generation-pass contract")
    if contract != expected_contract:
        raise RuntimeError("fill generation-pass contract differs from runtime receipt")
    candidate_path, report_path = generation_pass.generation_paths(PASS_ID, shard_index)
    expected = {
        "schema": "high-information-generation-pass-contract-v1",
        "generation_pass_id": PASS_ID,
        "gpu_shard": shard_index,
        "candidate_path": corpus.relative(candidate_path),
        "generation_report_path": corpus.relative(report_path),
        "worker_path": "run_high_information_candidate_fill_shard_v1.py",
        "runtime_worker_receipts": _expected_fill_runtime_receipts(),
        "temperature": FILL_TEMPERATURE,
        "top_p": FILL_TOP_P,
        "seed_scheme": FILL_SEED_SCHEME,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise RuntimeError(f"fill generation-pass field changed: {key}")
    if len(contract["runtime_worker_receipts"]) != 2:
        raise RuntimeError("fill generation pass must bind both runtime workers")
    for field in (
        "candidate_file_sha256",
        "generation_report_file_sha256",
        "generation_report_self_sha256",
        "worker_file_sha256",
        "prompt_spec_sha256",
        "content_sha256_before_self_field",
    ):
        value = contract.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise RuntimeError(f"fill generation-pass identity missing: {field}")


def load_structural_packets(shard_index: int) -> tuple[list[dict], dict]:
    review_path, summary_path = structural_paths(shard_index)
    if any(path.is_symlink() or not path.is_file() for path in (review_path, summary_path)):
        raise RuntimeError("fill structural review output/summary pair is incomplete")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require_self_address(summary, "fill structural summary")
    (
        candidates,
        requests,
        _,
        expected_pass_contract,
        contexts,
        request_index,
    ) = generation_pass.load_generation_pass(PASS_ID, shard_index)
    observed_pass_contract = summary.get("generation_pass_contract")
    if not isinstance(observed_pass_contract, dict):
        raise RuntimeError("fill structural summary lacks generation-pass contract")
    validate_fill_generation_contract(
        observed_pass_contract, expected_pass_contract, shard_index
    )
    review_payload = review_path.read_bytes()
    if (
        summary.get("schema")
        != "high-information-pass-aware-structural-review-v1"
        or summary.get("generation_pass_id") != PASS_ID
        or summary.get("gpu_shard") != shard_index
        or summary.get("requests_with_candidate_records") != len(requests)
        or summary.get("report_path") != corpus.relative(review_path)
        or summary.get("report_file_sha256")
        != corpus.sha256_bytes(review_payload)
        or summary.get("semantic_verification_completed") is not False
        or summary.get("accepted_training_rows_emitted") is not False
    ):
        raise RuntimeError("fill structural summary contract changed")

    rows = [
        json.loads(line)
        for line in review_payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(requests) or len(candidates) != len(requests):
        raise RuntimeError("fill structural review/request coverage changed")
    packets: list[dict] = []
    seen: set[str] = set()
    pass_contract_sha256 = observed_pass_contract[
        "content_sha256_before_self_field"
    ]
    for row, candidate_record, request in zip(
        rows, candidates, requests, strict=True
    ):
        _require_self_address(row, "fill structural review row")
        if (
            row.get("schema")
            != "high-information-pass-aware-structural-review-row-v1"
            or row.get("request_id") != request["request_id"]
            or request_index.get(request["request_id"]) != request
            or request.get("gpu_shard") != shard_index
            or row.get("generation_pass_id") != PASS_ID
            or row.get("generation_pass_contract_sha256")
            != pass_contract_sha256
            or row.get("candidate_record_sha256")
            != candidate_record["content_sha256_before_self_field"]
            or row.get("status")
            not in {
                "structurally_valid_semantic_verification_pending",
                "partially_or_fully_rejected",
                "rejected",
            }
        ):
            raise RuntimeError("fill structural row lineage changed")
        if row.get("status") == "rejected" and row.get(
            "structurally_valid_examples"
        ):
            raise RuntimeError("rejected fill structural row contains valid examples")
        context = contexts[request["source_context_id"]]
        for candidate in row.get("structurally_valid_examples", []):
            packet = semantic.build_packet(candidate, request, context)
            candidate_id = packet["candidate_example_id"]
            if candidate_id in seen:
                raise RuntimeError("fill structural review duplicates candidate identity")
            seen.add(candidate_id)
            packets.append(packet)
    if len(packets) != summary.get("structurally_valid_examples"):
        raise RuntimeError("fill structural summary example count changed")
    packets.sort(key=lambda value: (value["request_id"], value["candidate_example_id"]))
    return packets, summary


def _base_projection(row: dict) -> dict:
    projection = dict(row)
    projection.pop("generation_pass_id", None)
    projection.pop("generation_pass_contract_sha256", None)
    projection["schema"] = nli.RESULT_SCHEMA
    projection["content_sha256_before_self_field"] = nli._self_address(projection)
    return projection


def make_result(
    packet: dict,
    *,
    probabilities: dict[str, float] | None,
    input_token_count: int | None,
    input_truncated: bool,
    run_contract_sha256: str,
    generation_pass_contract_sha256: str,
) -> dict:
    validate_reused_nli_semantics()
    row = nli.make_result(
        packet,
        probabilities=probabilities,
        input_token_count=input_token_count,
        input_truncated=input_truncated,
        run_contract_sha256=run_contract_sha256,
    )
    row["schema"] = RESULT_SCHEMA
    row["generation_pass_id"] = PASS_ID
    row["generation_pass_contract_sha256"] = generation_pass_contract_sha256
    row["content_sha256_before_self_field"] = _self_address(row)
    return row


def validate_result(
    row: dict,
    packet: dict,
    run_contract_sha256: str,
    generation_pass_contract_sha256: str,
) -> None:
    _require_self_address(row, "pass-aware fill NLI result")
    if set(row) != RESULT_FIELDS:
        raise RuntimeError("pass-aware fill NLI result fields changed")
    if (
        row.get("schema") != RESULT_SCHEMA
        or row.get("generation_pass_id") != PASS_ID
        or row.get("generation_pass_contract_sha256")
        != generation_pass_contract_sha256
        or row.get("run_contract_sha256") != run_contract_sha256
        or row.get("semantic_verification_completed") is not False
        or row.get("eligible_for_training") is not False
    ):
        raise RuntimeError("pass-aware fill NLI result identity changed")
    nli.validate_result(_base_projection(row), packet, run_contract_sha256)


def _load_partial(
    path: Path,
    packets: Sequence[dict],
    run_contract_sha256: str,
    generation_pass_contract_sha256: str,
) -> list[dict]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("fill NLI partial output is not a regular file")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) > len(packets):
        raise RuntimeError("fill NLI partial output exceeds packet shard")
    for row, packet in zip(rows, packets, strict=False):
        validate_result(
            row,
            packet,
            run_contract_sha256,
            generation_pass_contract_sha256,
        )
    return rows


def _implementation_receipts() -> list[dict[str, str]]:
    paths = (
        Path(__file__).resolve(),
        Path(nli.__file__).resolve(),
        Path(generation_pass.__file__).resolve(),
        Path(structural.__file__).resolve(),
        Path(semantic.__file__).resolve(),
    )
    return [
        {"path": corpus.relative(path), "file_sha256": corpus.file_sha256(path)}
        for path in paths
    ]


def build_run_contract(
    *,
    shard_index: int,
    structural_summary: dict,
    model_receipts: dict[str, dict[str, Any]],
    paths: dict[str, Path],
    smoke: bool,
    runtime_environment: dict[str, str] | None = None,
) -> dict:
    validate_reused_nli_semantics()
    if runtime_environment is None:
        runtime_environment = validate_runtime_environment()
    pass_contract = structural_summary["generation_pass_contract"]
    contract = {
        "schema": RUN_CONTRACT_SCHEMA,
        "generation_pass": {
            "id": PASS_ID,
            "contract_sha256": pass_contract[
                "content_sha256_before_self_field"
            ],
            "candidate_output_sha256": pass_contract["candidate_file_sha256"],
            "generation_report_file_sha256": pass_contract[
                "generation_report_file_sha256"
            ],
            "generation_report_self_sha256": pass_contract[
                "generation_report_self_sha256"
            ],
            "runtime_worker_receipts": pass_contract["runtime_worker_receipts"],
            "temperature": pass_contract["temperature"],
            "top_p": pass_contract["top_p"],
            "seed_scheme": pass_contract["seed_scheme"],
            "prompt_spec_sha256": pass_contract["prompt_spec_sha256"],
        },
        "gpu_shard": shard_index,
        "smoke": smoke,
        "structural_review": {
            "path": structural_summary["report_path"],
            "file_sha256": structural_summary["report_file_sha256"],
            "summary_self_sha256": structural_summary[
                "content_sha256_before_self_field"
            ],
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "file_receipts": model_receipts,
        },
        "runtime": runtime_environment,
        "nli_semantics": {
            "premise_construction": (
                "ordered_exact_evidence_quotes_joined_by_newline"
            ),
            "max_length": MAX_LENGTH,
            "thresholds": {
                "entailment_minimum": ENTAILMENT_MINIMUM,
                "contradiction_maximum": CONTRADICTION_MAXIMUM,
                "contradiction_failure": CONTRADICTION_FAILURE,
            },
            "hard_negative_policy": (
                "not_applicable_requires_absence_or_false_premise_verifier"
            ),
        },
        "implementation_receipts": _implementation_receipts(),
        "planned_outputs": {
            key: corpus.relative(value) for key, value in paths.items()
        },
        "output_receipt_schema": RECEIPT_SCHEMA,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    contract["content_sha256_before_self_field"] = _self_address(contract)
    return contract


def _payload(rows: Sequence[dict]) -> bytes:
    return corpus.jsonl_payload(rows)


def _telemetry(
    *, shard_index: int, gpu_index: int, packets: int, completed: int, status: str
) -> dict:
    return {
        "schema": TELEMETRY_SCHEMA,
        "updated_at": utc_now(),
        "status": status,
        "pid": os.getpid(),
        "generation_pass_id": PASS_ID,
        "gpu_shard": shard_index,
        "physical_gpu_index": gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packets": packets,
        "completed_packets": completed,
        "training_rows_emitted": False,
    }


def _prepare(args: argparse.Namespace) -> tuple[list[dict], dict, dict, dict]:
    validate_reused_nli_semantics()
    runtime_environment = validate_runtime_environment()
    model_receipts = validate_model_snapshot(args.model_directory)
    packets, structural_summary = load_structural_packets(args.shard_index)
    if args.smoke:
        positive = next((item for item in packets if nli.nli_inputs(item)), None)
        negative = next((item for item in packets if nli.nli_inputs(item) is None), None)
        packets = [item for item in (positive, negative) if item is not None]
    paths = output_paths(args.shard_index, smoke=args.smoke)
    contract = build_run_contract(
        shard_index=args.shard_index,
        structural_summary=structural_summary,
        model_receipts=model_receipts,
        paths=paths,
        smoke=args.smoke,
        runtime_environment=runtime_environment,
    )
    return packets, structural_summary, paths, contract


def preflight(args: argparse.Namespace) -> dict:
    packets, structural_summary, paths, contract = _prepare(args)
    pass_contract = structural_summary["generation_pass_contract"]
    return {
        "status": "fill_nli_preflight_complete_no_gpu_launch",
        "generation_pass_id": PASS_ID,
        "gpu_shard": args.shard_index,
        "packets": len(packets),
        "positive_packets": sum(nli.nli_inputs(packet) is not None for packet in packets),
        "hard_negative_packets": sum(
            nli.nli_inputs(packet) is None for packet in packets
        ),
        "fill_runtime_worker_receipts": pass_contract["runtime_worker_receipts"],
        "generation_pass_contract_sha256": pass_contract[
            "content_sha256_before_self_field"
        ],
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "run_contract": contract,
        "planned_outputs": {
            key: corpus.relative(value) for key, value in paths.items()
        },
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }


def build_output_receipt(
    *,
    shard_index: int,
    paths: dict[str, Path],
    output_payload: bytes,
    report_payload: bytes,
    report: dict,
) -> dict:
    _require_self_address(report, "pass-aware fill NLI report")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "sealed_complete_semantic_pending",
        "generation_pass_id": PASS_ID,
        "gpu_shard": shard_index,
        "output": corpus.relative(paths["output"]),
        "output_sha256": corpus.sha256_bytes(output_payload),
        "report": corpus.relative(paths["report"]),
        "report_file_sha256": corpus.sha256_bytes(report_payload),
        "report_self_sha256": report["content_sha256_before_self_field"],
        "run_contract_sha256": report["run_contract"][
            "content_sha256_before_self_field"
        ],
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    receipt["content_sha256_before_self_field"] = _self_address(receipt)
    return receipt


def run(args: argparse.Namespace) -> dict:
    packets, structural_summary, paths, contract = _prepare(args)
    if any(paths[name].exists() for name in ("output", "report", "receipt")):
        raise RuntimeError(
            "sealed fill NLI output exists; use --check-output instead of overwriting"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        raise RuntimeError("fill NLI requires exactly its assigned physical GPU")

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if transformers.__version__ != TRANSFORMERS_VERSION or torch.__version__ != TORCH_VERSION:
        raise RuntimeError("fill NLI runtime package version changed")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_directory, local_files_only=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_directory,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).eval().to("cuda:0")
    labels = {
        int(index): label.casefold() for index, label in model.config.id2label.items()
    }
    if labels != {0: "entailment", 1: "neutral", 2: "contradiction"}:
        raise RuntimeError("pinned fill NLI label order changed")

    started_at = utc_now()
    run_contract_sha256 = contract["content_sha256_before_self_field"]
    pass_contract_sha256 = structural_summary["generation_pass_contract"][
        "content_sha256_before_self_field"
    ]
    rows = _load_partial(
        paths["partial"], packets, run_contract_sha256, pass_contract_sha256
    )
    for batch_start in range(len(rows), len(packets), args.batch_size):
        batch_started = time.monotonic()
        batch_packets = packets[batch_start : batch_start + args.batch_size]
        positive_packets = [
            packet for packet in batch_packets if nli.nli_inputs(packet) is not None
        ]
        inference: dict[str, tuple[dict[str, float], int, bool]] = {}
        if positive_packets:
            pairs = [nli.nli_inputs(packet) for packet in positive_packets]
            assert all(pair is not None for pair in pairs)
            premises = [pair[0] for pair in pairs if pair is not None]
            hypotheses = [pair[1] for pair in pairs if pair is not None]
            untruncated = tokenizer(
                premises,
                hypotheses,
                add_special_tokens=True,
                truncation=False,
                return_length=True,
            )
            lengths = [int(value) for value in untruncated["length"]]
            encoded = tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation="only_first",
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to("cuda:0") for key, value in encoded.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1)
            for packet, vector, length in zip(
                positive_packets, probabilities.cpu().tolist(), lengths, strict=True
            ):
                inference[packet["candidate_example_id"]] = (
                    {
                        labels[index]: float(value)
                        for index, value in enumerate(vector)
                    },
                    length,
                    length > MAX_LENGTH,
                )
        for packet in batch_packets:
            values = inference.get(packet["candidate_example_id"])
            row = make_result(
                packet,
                probabilities=values[0] if values else None,
                input_token_count=values[1] if values else None,
                input_truncated=values[2] if values else False,
                run_contract_sha256=run_contract_sha256,
                generation_pass_contract_sha256=pass_contract_sha256,
            )
            validate_result(
                row, packet, run_contract_sha256, pass_contract_sha256
            )
            rows.append(row)
        _atomic_write(paths["partial"], _payload(rows))
        telemetry = _telemetry(
            shard_index=args.shard_index,
            gpu_index=args.gpu_index,
            packets=len(packets),
            completed=len(rows),
            status="running",
        )
        telemetry["last_batch_seconds"] = time.monotonic() - batch_started
        _atomic_write(paths["telemetry"], corpus.canonical_bytes(telemetry))

    output_payload = _payload(rows)
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete_semantic_pending",
        "started_at": started_at,
        "completed_at": utc_now(),
        "generation_pass_id": PASS_ID,
        "gpu_shard": args.shard_index,
        "physical_gpu_index": args.gpu_index,
        "smoke": args.smoke,
        "packets": len(packets),
        "pass": sum(row["verdict"] == "pass" for row in rows),
        "fail": sum(row["verdict"] == "fail" for row in rows),
        "uncertain": sum(row["verdict"] == "uncertain" for row in rows),
        "not_applicable": sum(
            row["verdict"] == "not_applicable" for row in rows
        ),
        "manual_review_required": sum(
            row["manual_review_required"] for row in rows
        ),
        "generation_pass_contract_sha256": pass_contract_sha256,
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "output": corpus.relative(paths["output"]),
        "output_sha256": corpus.sha256_bytes(output_payload),
        "output_receipt": corpus.relative(paths["receipt"]),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "model_file_receipts": contract["model"]["file_receipts"],
        "implementation_receipts": contract["implementation_receipts"],
        "run_contract": contract,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    report["content_sha256_before_self_field"] = _self_address(report)
    report_payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    receipt = build_output_receipt(
        shard_index=args.shard_index,
        paths=paths,
        output_payload=output_payload,
        report_payload=report_payload,
        report=report,
    )
    _atomic_write(paths["output"], output_payload)
    _atomic_write(paths["report"], report_payload)
    _atomic_write(paths["receipt"], (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"))
    _atomic_write(paths["telemetry"], corpus.canonical_bytes(_telemetry(
        shard_index=args.shard_index,
        gpu_index=args.gpu_index,
        packets=len(packets),
        completed=len(rows),
        status="complete_semantic_pending",
    )))
    return report


def validate_completed_output(args: argparse.Namespace) -> dict:
    packets, structural_summary, paths, expected_contract = _prepare(args)
    for name in ("output", "report", "receipt"):
        _regular_file(paths[name], f"sealed fill NLI {name}")
    output_payload = paths["output"].read_bytes()
    report_payload = paths["report"].read_bytes()
    report = json.loads(report_payload)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    _require_self_address(report, "pass-aware fill NLI report")
    _require_self_address(receipt, "pass-aware fill NLI output receipt")
    pass_contract_sha256 = structural_summary["generation_pass_contract"][
        "content_sha256_before_self_field"
    ]
    run_contract_sha256 = expected_contract["content_sha256_before_self_field"]
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status") != "complete_semantic_pending"
        or report.get("generation_pass_id") != PASS_ID
        or report.get("gpu_shard") != args.shard_index
        or report.get("smoke") != args.smoke
        or report.get("packets") != len(packets)
        or report.get("generation_pass_contract_sha256")
        != pass_contract_sha256
        or report.get("structural_review_sha256")
        != structural_summary["report_file_sha256"]
        or report.get("output") != corpus.relative(paths["output"])
        or report.get("output_sha256") != corpus.sha256_bytes(output_payload)
        or report.get("output_receipt") != corpus.relative(paths["receipt"])
        or report.get("run_contract") != expected_contract
        or report.get("semantic_verification_completed") is not False
        or report.get("training_rows_emitted") is not False
    ):
        raise RuntimeError("pass-aware fill NLI report contract changed")
    rows = [
        json.loads(line)
        for line in output_payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(packets):
        raise RuntimeError("pass-aware fill NLI output row count changed")
    for row, packet in zip(rows, packets, strict=True):
        validate_result(
            row, packet, run_contract_sha256, pass_contract_sha256
        )
    expected_receipt = build_output_receipt(
        shard_index=args.shard_index,
        paths=paths,
        output_payload=output_payload,
        report_payload=report_payload,
        report=report,
    )
    if receipt != expected_receipt:
        raise RuntimeError("pass-aware fill NLI output receipt changed")
    return {
        "status": "sealed_fill_nli_output_valid_semantic_pending",
        "generation_pass_id": PASS_ID,
        "gpu_shard": args.shard_index,
        "packets": len(rows),
        "output": corpus.relative(paths["output"]),
        "output_sha256": receipt["output_sha256"],
        "report": corpus.relative(paths["report"]),
        "report_file_sha256": receipt["report_file_sha256"],
        "report_self_sha256": receipt["report_self_sha256"],
        "receipt": corpus.relative(paths["receipt"]),
        "receipt_file_sha256": corpus.file_sha256(paths["receipt"]),
        "run_contract_sha256": run_contract_sha256,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--shard-index", required=True, type=int, choices=range(4))
    result.add_argument("--gpu-index", type=int)
    result.add_argument("--model-directory", type=Path, default=MODEL_DIRECTORY)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--smoke", action="store_true")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--check-plan", action="store_true")
    modes.add_argument("--check-output", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.batch_size <= 512:
        raise ValueError("fill NLI batch size must be 1-512")
    if args.check_plan:
        value = preflight(args)
    elif args.check_output:
        value = validate_completed_output(args)
    else:
        if args.gpu_index is None or args.gpu_index != args.shard_index:
            raise ValueError("fill NLI shard 0-3 must use its matching GPU")
        value = run(args)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
