#!/usr/bin/env python3
"""Run an isolated two-pass semantic judge over quality-deficit-fill-v1.

The worker imports the primary judge's parsing, prompt, validation, and gate
aggregation functions, but pins their implementation bytes and writes only to
fill-specific paths.  It will not preflight or launch until the corresponding
fill NLI output, report, and receipt have passed their sealed-output validator.
Every nested result remains ineligible for training.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence

import build_high_information_domain_corpus_v1 as corpus
import high_information_semantic_facets_v1 as facets
import run_high_information_fill_nli_prefilter_v1 as fill_nli
import run_high_information_nli_prefilter_v1 as primary_nli
import run_high_information_semantic_judge_shard_v1 as primary_judge
import verify_high_information_candidates_v1 as structural
import verify_high_information_generation_pass_v1 as generation_pass
import verify_high_information_semantic_decisions_v1 as semantic


PASS_ID = "quality-deficit-fill-v1"
PASS_SLUG = "quality_deficit_fill_v1"
RESULT_SCHEMA = "high-information-pass-aware-fill-semantic-judge-result-v1"
RECORD_SCHEMA = "high-information-pass-aware-fill-semantic-judge-request-v1"
RUN_CONTRACT_SCHEMA = (
    "high-information-pass-aware-fill-semantic-judge-run-contract-v1"
)
REPORT_SCHEMA = "high-information-pass-aware-fill-semantic-judge-report-v1"
RECEIPT_SCHEMA = (
    "high-information-pass-aware-fill-semantic-judge-output-receipt-v1"
)
TELEMETRY_SCHEMA = "high-information-pass-aware-fill-semantic-judge-telemetry-v1"

PRIMARY_JUDGE_FILE_SHA256 = (
    "0104c58066a08ee5d9d1b05020dfe26cef8734e9a185c7f32b29972a26567935"
)
FACET_IMPLEMENTATION_SHA256 = (
    "880727323764910736d2d86d917b8bcf1783c63ec226c990653c6be9a42f9140"
)
PRIMARY_NLI_FILE_SHA256 = (
    "b346f049e9d4520fa2c7c7a004574eb7132b6cbd1c843347320d587c66b3f0ea"
)
FILL_NLI_FILE_SHA256 = (
    "0cfcf91653e9b0b61e4728e5d91dbeccde0baaf609c4823743a18c3cf5397fa1"
)
SEMANTIC_VERIFIER_FILE_SHA256 = (
    "8700b28d6ef1edbe0ed1b6e5be4642ec1c5b3bdbe1b9267b4b874c0a4eb25dc4"
)
GENERATION_PASS_VERIFIER_FILE_SHA256 = (
    "c8e47c19c1a15c10cb7351f07c45f9e7d4d8efdd5ec7345b9c2d1f0ba6ad2bc5"
)
GUIDED_SCHEMA_SHA256 = (
    "04565b2c63344d3c5e49db85b382a0fc8ec5b9045140ceabc157a28cf7ebdbc1"
)
SEMANTIC_PROTOCOL_SHA256 = (
    "6ae7b69e77bc64e294af1daa36786b05d867afb2ec8aee34eda76e9410eb9bfc"
)
CANDIDATE_GROUPING = "singleton_sorted_within_request_v1"

NLI_LAUNCH_CONTRACT_PATH = (
    corpus.OUTPUT_DIR
    / "nli_prefilter_quality_deficit_fill_v1.launch_contract.json"
)
NLI_LAUNCH_CONTRACT_FILE_SHA256 = (
    "0bf80a57ad23fbb0637794e013f946d2bdb0188573083ef24571ab83947d4a1b"
)
NLI_LAUNCH_CONTRACT_SELF_SHA256 = (
    "2def05349d30de388c791bdc74010bbeb83ac6730b104947c8c59aef29a55cb9"
)
NLI_LAUNCH_BATCH_SIZE = 64

MODEL_ID = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
MODEL_REVISION = "95a6d26c4bfb886c58daf9d3f7332c857cb27b43"
MODEL_DIRECTORY = primary_judge.MODEL_DIRECTORY
MODEL_BLOB_RECEIPTS = primary_judge.MODEL_BLOB_RECEIPTS
RUNTIME_MODEL_FILE_SHA256 = primary_judge.RUNTIME_MODEL_FILE_SHA256
VLLM_VERSION = "0.25.0"
MISTRAL_COMMON_VERSION = "1.11.5"
TRANSFORMERS_VERSION = "5.13.1"
TORCH_VERSION = "2.11.0+cu130"
DTYPE = "bfloat16"
RUNTIME_PYTHON_EXECUTABLE = fill_nli.RUNTIME_PYTHON_EXECUTABLE
RUNTIME_VIRTUAL_ENV = fill_nli.RUNTIME_VIRTUAL_ENV
CU13_LIBRARY_PATH = fill_nli.CU13_LIBRARY_PATH
MAX_NUM_SEQS = 32
DEFAULT_REQUEST_BATCH_SIZE = 16
DEFAULT_MAX_MODEL_LEN = 16_384
DEFAULT_MAX_TOKENS = 3_072

PRIMARY_RECORD_FIELDS = {
    "schema",
    "request_id",
    "source_group_id",
    "candidate_example_ids",
    "pass_output_sha256s",
    "normalized_pass_outputs",
    "pass_validation_errors",
    "invalid_raw_pass_outputs",
    "results",
    "run_contract_sha256",
    "semantic_verification_completed",
    "training_rows_emitted",
    "content_sha256_before_self_field",
}
FILL_RECORD_FIELDS = PRIMARY_RECORD_FIELDS | {
    "generation_pass_id",
    "generation_pass_contract_sha256",
    "nli_output_sha256",
    "nli_output_receipt_self_sha256",
}
FILL_RESULT_EXTRA_FIELDS = {
    "generation_pass_id",
    "generation_pass_contract_sha256",
    "nli_output_receipt_self_sha256",
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


def _expected_protocol() -> dict[str, Any]:
    return {
        "candidate_grouping": CANDIDATE_GROUPING,
        "passes": list(primary_judge.PASS_NAMES),
        "model_gates": list(primary_judge.MODEL_GATES),
        "evidence_required_pass_gates": sorted(
            primary_judge.EVIDENCE_REQUIRED_PASS_GATES
        ),
        "all_gates": list(primary_judge.ALL_GATES),
        "failure_codes": list(primary_judge.FAILURE_CODES),
        "guided_schema_sha256": corpus.canonical_sha256(
            primary_judge.GUIDED_SCHEMA
        ),
    }


def validate_reused_judge_semantics() -> dict[str, Any]:
    """Pin every imported semantic implementation and public protocol value."""

    implementation_expectations = {
        Path(primary_judge.__file__).resolve(): PRIMARY_JUDGE_FILE_SHA256,
        Path(facets.__file__).resolve(): FACET_IMPLEMENTATION_SHA256,
        Path(primary_nli.__file__).resolve(): PRIMARY_NLI_FILE_SHA256,
        Path(fill_nli.__file__).resolve(): FILL_NLI_FILE_SHA256,
        Path(semantic.__file__).resolve(): SEMANTIC_VERIFIER_FILE_SHA256,
        Path(generation_pass.__file__).resolve(): (
            GENERATION_PASS_VERIFIER_FILE_SHA256
        ),
    }
    receipts = []
    for path, expected in implementation_expectations.items():
        observed = corpus.file_sha256(path)
        if observed != expected:
            raise RuntimeError(f"pinned semantic implementation changed: {path.name}")
        receipts.append({"path": corpus.relative(path), "file_sha256": observed})

    protocol = _expected_protocol()
    observed = {
        "pass_id": fill_nli.PASS_ID,
        "model_id": primary_judge.MODEL_ID,
        "model_revision": primary_judge.MODEL_REVISION,
        "model_blob_receipts": primary_judge.MODEL_BLOB_RECEIPTS,
        "runtime_model_file_sha256": primary_judge.RUNTIME_MODEL_FILE_SHA256,
        "vllm_version": primary_judge.VLLM_VERSION,
        "guided_schema_sha256": corpus.canonical_sha256(
            primary_judge.GUIDED_SCHEMA
        ),
        "semantic_protocol_sha256": corpus.canonical_sha256(protocol),
    }
    expected = {
        "pass_id": PASS_ID,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_blob_receipts": MODEL_BLOB_RECEIPTS,
        "runtime_model_file_sha256": RUNTIME_MODEL_FILE_SHA256,
        "vllm_version": VLLM_VERSION,
        "guided_schema_sha256": GUIDED_SCHEMA_SHA256,
        "semantic_protocol_sha256": SEMANTIC_PROTOCOL_SHA256,
    }
    if observed != expected:
        raise RuntimeError("reused primary judge semantics changed")
    return {
        "implementation_receipts": receipts,
        "protocol": protocol,
        **observed,
    }


def validate_runtime_environment() -> dict[str, str]:
    """Require the exact es-at-scale Python and package/runtime identities."""

    base = fill_nli.validate_runtime_environment()
    if (
        Path(sys.executable).absolute() != RUNTIME_PYTHON_EXECUTABLE
        or Path(sys.prefix).absolute() != RUNTIME_VIRTUAL_ENV
        or os.environ.get("LD_LIBRARY_PATH") != str(CU13_LIBRARY_PATH)
        or base.get("transformers_version") != TRANSFORMERS_VERSION
        or base.get("torch_version") != TORCH_VERSION
    ):
        raise RuntimeError("fill semantic judge runtime identity changed")
    package_versions = {
        "vllm": metadata.version("vllm"),
        "mistral_common": metadata.version("mistral-common"),
    }
    if package_versions != {
        "vllm": VLLM_VERSION,
        "mistral_common": MISTRAL_COMMON_VERSION,
    }:
        raise RuntimeError("fill semantic judge runtime package version changed")
    return {
        **base,
        "vllm_version": package_versions["vllm"],
        "mistral_common_version": package_versions["mistral_common"],
    }


def validate_model_snapshot(model_directory: Path) -> dict[str, dict[str, Any]]:
    validate_reused_judge_semantics()
    receipts = primary_judge.validate_model_snapshot(model_directory)
    for name, expected_sha256 in RUNTIME_MODEL_FILE_SHA256.items():
        receipt = receipts.get(name)
        if (
            not isinstance(receipt, dict)
            or receipt.get("runtime_loaded") is not True
            or receipt.get("file_sha256") != expected_sha256
            or not isinstance(receipt.get("file_bytes"), int)
            or receipt["file_bytes"] <= 0
        ):
            raise RuntimeError("pinned Mistral runtime-byte receipt changed")
    return receipts


def output_paths(shard_index: int, *, smoke: bool = False) -> dict[str, Path]:
    if shard_index not in range(4):
        raise ValueError("fill semantic judge shard must be 0-3")
    suffix = ".smoke" if smoke else ""
    stem = f"semantic_judge_{PASS_SLUG}_gpu{shard_index}{suffix}"
    return {
        "partial": corpus.OUTPUT_DIR / f"{stem}.partial.jsonl",
        "output": corpus.OUTPUT_DIR / f"{stem}.jsonl",
        "report": corpus.OUTPUT_DIR / f"{stem}.report.json",
        "receipt": corpus.OUTPUT_DIR / f"{stem}.receipt.json",
        "telemetry": corpus.OUTPUT_DIR / f"{stem}.telemetry.json",
    }


def groups_by_request(packets: Sequence[dict]) -> list[list[dict]]:
    """Reuse and enforce singleton, request/candidate-sorted grouping."""

    groups = primary_judge.groups_by_request(packets)
    expected = sorted(
        packets,
        key=lambda value: (value["request_id"], value["candidate_example_id"]),
    )
    if (
        any(len(group) != 1 for group in groups)
        or [group[0]["candidate_example_id"] for group in groups]
        != [packet["candidate_example_id"] for packet in expected]
    ):
        raise RuntimeError("singleton-sorted candidate grouping changed")
    return groups


def validate_nli_launch_contract() -> tuple[dict, dict[str, str]]:
    path = _regular_file(NLI_LAUNCH_CONTRACT_PATH, "fill NLI launch contract")
    file_sha256 = corpus.file_sha256(path)
    if file_sha256 != NLI_LAUNCH_CONTRACT_FILE_SHA256:
        raise RuntimeError("fill NLI launch-contract file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require_self_address(value, "fill NLI launch contract")
    if (
        value.get("schema")
        != "high-information-pass-aware-fill-nli-launch-contract-v1"
        or value.get("status") != "ready_for_explicit_four_gpu_nli_launch"
        or value.get("content_sha256_before_self_field")
        != NLI_LAUNCH_CONTRACT_SELF_SHA256
        or value.get("batch_size") != NLI_LAUNCH_BATCH_SIZE
        or value.get("generation_pass", {}).get("id") != PASS_ID
        or len(value.get("shards", [])) != 4
        or value.get("policy", {}).get("training_rows_emitted") is not False
    ):
        raise RuntimeError("fill NLI launch-contract semantics changed")
    return value, {
        "path": corpus.relative(path),
        "file_sha256": file_sha256,
        "content_sha256_before_self_field": value[
            "content_sha256_before_self_field"
        ],
    }


def _fill_nli_args(shard_index: int) -> argparse.Namespace:
    return argparse.Namespace(
        shard_index=shard_index,
        gpu_index=None,
        model_directory=fill_nli.MODEL_DIRECTORY,
        batch_size=NLI_LAUNCH_BATCH_SIZE,
        smoke=False,
        check_plan=False,
        check_output=True,
    )


def load_fill_nli_results(
    shard_index: int, packets: Sequence[dict]
) -> tuple[dict[str, dict], dict[str, Any]]:
    """Validate the sealed fill NLI trio before exposing any rows to judging."""

    launch_contract, launch_receipt = validate_nli_launch_contract()
    completion = fill_nli.validate_completed_output(_fill_nli_args(shard_index))
    paths = fill_nli.output_paths(shard_index, smoke=False)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    _require_self_address(report, "sealed fill NLI report")
    _require_self_address(receipt, "sealed fill NLI receipt")
    launch_shards = {
        item.get("gpu_shard"): item for item in launch_contract["shards"]
    }
    launch_shard = launch_shards.get(shard_index)
    if not isinstance(launch_shard, dict):
        raise RuntimeError("fill NLI launch contract omits judge shard")
    if (
        launch_shard.get("run_contract_sha256")
        != completion["run_contract_sha256"]
        or launch_shard.get("planned_outputs", {}).get("output")
        != corpus.relative(paths["output"])
        or launch_shard.get("planned_outputs", {}).get("report")
        != corpus.relative(paths["report"])
        or launch_shard.get("planned_outputs", {}).get("receipt")
        != corpus.relative(paths["receipt"])
        or receipt.get("output_sha256") != completion["output_sha256"]
        or receipt.get("report_file_sha256")
        != completion["report_file_sha256"]
        or receipt.get("report_self_sha256")
        != completion["report_self_sha256"]
        or receipt.get("run_contract_sha256")
        != completion["run_contract_sha256"]
        or completion.get("training_rows_emitted") is not False
    ):
        raise RuntimeError("sealed fill NLI launch/output lineage changed")

    rows = [
        json.loads(line)
        for line in paths["output"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(packets):
        raise RuntimeError("sealed fill NLI row count differs from fill packets")
    pass_contract_sha256 = report["generation_pass_contract_sha256"]
    run_contract_sha256 = completion["run_contract_sha256"]
    index: dict[str, dict] = {}
    for row, packet in zip(rows, packets, strict=True):
        fill_nli.validate_result(
            row, packet, run_contract_sha256, pass_contract_sha256
        )
        candidate_id = row["candidate_example_id"]
        if candidate_id in index:
            raise RuntimeError("sealed fill NLI output duplicates candidate identity")
        index[candidate_id] = row
    identity = {
        "launch_contract": launch_receipt,
        "worker_file_sha256": FILL_NLI_FILE_SHA256,
        "primary_nli_file_sha256": PRIMARY_NLI_FILE_SHA256,
        "result_schema": fill_nli.RESULT_SCHEMA,
        "output": completion["output"],
        "output_sha256": completion["output_sha256"],
        "report": completion["report"],
        "report_file_sha256": completion["report_file_sha256"],
        "report_self_sha256": completion["report_self_sha256"],
        "receipt": completion["receipt"],
        "receipt_file_sha256": completion["receipt_file_sha256"],
        "receipt_self_sha256": receipt["content_sha256_before_self_field"],
        "run_contract_sha256": completion["run_contract_sha256"],
        "generation_pass_contract_sha256": pass_contract_sha256,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    return index, identity


def prompt_receipts(groups: Sequence[Sequence[dict]]) -> dict[str, Any]:
    receipts = []
    for group in groups:
        candidate_ids = [packet["candidate_example_id"] for packet in group]
        for pass_name in primary_judge.PASS_NAMES:
            messages = primary_judge.judge_messages(group, pass_name)
            if [item.get("role") for item in messages] != ["system", "user"]:
                raise RuntimeError("fill semantic judge prompt roles changed")
            encoded = [item["content"].encode("utf-8") for item in messages]
            receipts.append({
                "request_id": group[0]["request_id"],
                "source_group_id": group[0]["source_group_id"],
                "candidate_example_ids": candidate_ids,
                "pass_name": pass_name,
                "messages_sha256": corpus.canonical_sha256(messages),
                "system_message_sha256": corpus.sha256_bytes(encoded[0]),
                "system_message_bytes": len(encoded[0]),
                "user_message_sha256": corpus.sha256_bytes(encoded[1]),
                "user_message_bytes": len(encoded[1]),
            })
    expected = len(groups) * len(primary_judge.PASS_NAMES)
    if len(receipts) != expected:
        raise RuntimeError("fill semantic judge prompt coverage changed")
    identities = [
        (
            item["request_id"],
            tuple(item["candidate_example_ids"]),
            item["pass_name"],
        )
        for item in receipts
    ]
    if len(identities) != len(set(identities)):
        raise RuntimeError("fill semantic judge prompt identity duplicated")
    return {
        "schema": "fill-semantic-judge-rendered-prompt-receipt-root-v1",
        "candidate_grouping": CANDIDATE_GROUPING,
        "rendered_prompts": len(receipts),
        "receipts_sha256": corpus.canonical_sha256(receipts),
        "ordered_receipt_fields": [
            "request_id",
            "source_group_id",
            "candidate_example_ids",
            "pass_name",
            "messages_sha256",
            "system_message_sha256",
            "system_message_bytes",
            "user_message_sha256",
            "user_message_bytes",
        ],
        "every_rendered_system_and_user_message_bound": True,
        "prompt_text_persisted": False,
    }


def structural_receipt(shard_index: int, summary: dict) -> dict[str, Any]:
    review_path, summary_path = fill_nli.structural_paths(shard_index)
    _regular_file(review_path, "fill structural review")
    _regular_file(summary_path, "fill structural summary")
    if summary.get("report_file_sha256") != corpus.file_sha256(review_path):
        raise RuntimeError("fill structural review receipt changed")
    return {
        "review_path": corpus.relative(review_path),
        "review_file_sha256": summary["report_file_sha256"],
        "summary_path": corpus.relative(summary_path),
        "summary_file_sha256": corpus.file_sha256(summary_path),
        "summary_self_sha256": summary["content_sha256_before_self_field"],
        "structurally_valid_examples": summary["structurally_valid_examples"],
    }


def implementation_receipts() -> list[dict[str, str]]:
    pinned = validate_reused_judge_semantics()["implementation_receipts"]
    dynamic_paths = (Path(__file__).resolve(), Path(structural.__file__).resolve())
    dynamic = [
        {"path": corpus.relative(path), "file_sha256": corpus.file_sha256(path)}
        for path in dynamic_paths
    ]
    return dynamic + pinned


def build_run_contract(
    *,
    shard_index: int,
    structural_summary: dict,
    nli_identity: dict[str, Any],
    model_receipts: dict[str, dict[str, Any]],
    runtime_environment: dict[str, str],
    paths: dict[str, Path],
    args: argparse.Namespace,
    rendered_prompts: dict[str, Any],
    prompt_statistics: dict[str, Any],
) -> dict:
    reused = validate_reused_judge_semantics()
    pass_contract = structural_summary["generation_pass_contract"]
    if (
        pass_contract.get("generation_pass_id") != PASS_ID
        or nli_identity.get("generation_pass_contract_sha256")
        != pass_contract.get("content_sha256_before_self_field")
    ):
        raise RuntimeError("fill generation-pass contract lineage changed")
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
            "runtime_worker_receipts": pass_contract[
                "runtime_worker_receipts"
            ],
            "temperature": pass_contract["temperature"],
            "top_p": pass_contract["top_p"],
            "seed_scheme": pass_contract["seed_scheme"],
            "prompt_spec_sha256": pass_contract["prompt_spec_sha256"],
        },
        "gpu_shard": shard_index,
        "smoke": args.smoke,
        "structural_review": structural_receipt(shard_index, structural_summary),
        "sealed_fill_nli": nli_identity,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "snapshot_blob_receipts": MODEL_BLOB_RECEIPTS,
            "runtime_file_sha256": RUNTIME_MODEL_FILE_SHA256,
            "validated_file_receipts": model_receipts,
        },
        "runtime": runtime_environment,
        "judge_protocol": {
            **reused["protocol"],
            "semantic_protocol_sha256": SEMANTIC_PROTOCOL_SHA256,
            "primary_judge_result_schema": primary_judge.RESULT_SCHEMA,
            "primary_judge_record_schema": primary_judge.RECORD_SCHEMA,
            "fill_result_schema": RESULT_SCHEMA,
            "fill_record_schema": RECORD_SCHEMA,
            "rendered_prompt_receipts": rendered_prompts,
        },
        "guided_schema": {
            "schema": primary_judge.GUIDED_SCHEMA,
            "sha256": GUIDED_SCHEMA_SHA256,
        },
        "inference": {
            "dtype": DTYPE,
            "temperature": 0.0,
            "pass_seeds": {
                name: 1009 + index
                for index, name in enumerate(primary_judge.PASS_NAMES)
            },
            "request_batch_size": args.request_batch_size,
            "max_model_len": args.max_model_len,
            "max_tokens": args.max_tokens,
            "max_num_seqs": MAX_NUM_SEQS,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
            "prompt_statistics": prompt_statistics,
        },
        "implementation_receipts": implementation_receipts(),
        "planned_outputs": {
            key: corpus.relative(value) for key, value in paths.items()
        },
        "output_receipt_schema": RECEIPT_SCHEMA,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    contract["content_sha256_before_self_field"] = _self_address(contract)
    return contract


def _base_result_projection(result: dict) -> dict:
    projection = dict(result)
    for field in FILL_RESULT_EXTRA_FIELDS:
        projection.pop(field, None)
    projection["schema"] = primary_judge.RESULT_SCHEMA
    return projection


def _base_record_projection(record: dict) -> dict:
    projection = dict(record)
    for field in (
        "generation_pass_id",
        "generation_pass_contract_sha256",
        "nli_output_sha256",
        "nli_output_receipt_self_sha256",
    ):
        projection.pop(field, None)
    projection["schema"] = primary_judge.RECORD_SCHEMA
    projection["results"] = [
        _base_result_projection(result) for result in record["results"]
    ]
    projection["content_sha256_before_self_field"] = primary_judge._self_address(
        projection
    )
    return projection


def make_record(
    group: Sequence[dict],
    pass_outputs: dict[str, dict],
    nli_index: dict[str, dict],
    *,
    run_contract_sha256: str,
    generation_pass_contract_sha256: str,
    nli_output_sha256: str,
    nli_output_receipt_self_sha256: str,
) -> dict:
    validate_reused_judge_semantics()
    record = primary_judge.make_record(
        group,
        pass_outputs,
        nli_index,
        run_contract_sha256=run_contract_sha256,
    )
    record["schema"] = RECORD_SCHEMA
    record["generation_pass_id"] = PASS_ID
    record["generation_pass_contract_sha256"] = (
        generation_pass_contract_sha256
    )
    record["nli_output_sha256"] = nli_output_sha256
    record["nli_output_receipt_self_sha256"] = (
        nli_output_receipt_self_sha256
    )
    for result in record["results"]:
        result["schema"] = RESULT_SCHEMA
        result["generation_pass_id"] = PASS_ID
        result["generation_pass_contract_sha256"] = (
            generation_pass_contract_sha256
        )
        result["nli_output_receipt_self_sha256"] = (
            nli_output_receipt_self_sha256
        )
        result["eligible_for_training"] = False
    record["content_sha256_before_self_field"] = _self_address(record)
    return record


def validate_record(
    record: dict,
    group: Sequence[dict],
    run_contract_sha256: str,
    generation_pass_contract_sha256: str,
    nli_output_sha256: str,
    nli_output_receipt_self_sha256: str,
) -> None:
    _require_self_address(record, "pass-aware fill semantic judge record")
    if (
        set(record) != FILL_RECORD_FIELDS
        or record.get("schema") != RECORD_SCHEMA
        or record.get("generation_pass_id") != PASS_ID
        or record.get("generation_pass_contract_sha256")
        != generation_pass_contract_sha256
        or record.get("nli_output_sha256") != nli_output_sha256
        or record.get("nli_output_receipt_self_sha256")
        != nli_output_receipt_self_sha256
        or record.get("run_contract_sha256") != run_contract_sha256
        or record.get("semantic_verification_completed") is not False
        or record.get("training_rows_emitted") is not False
    ):
        raise RuntimeError("pass-aware fill semantic judge record changed")
    for result in record.get("results", []):
        if (
            result.get("schema") != RESULT_SCHEMA
            or result.get("generation_pass_id") != PASS_ID
            or result.get("generation_pass_contract_sha256")
            != generation_pass_contract_sha256
            or result.get("nli_output_receipt_self_sha256")
            != nli_output_receipt_self_sha256
            or result.get("eligible_for_training") is not False
        ):
            raise RuntimeError("fill semantic result became training-eligible")
    primary_judge.validate_record(
        _base_record_projection(record), group, run_contract_sha256
    )


def _payload(rows: Sequence[dict]) -> bytes:
    return corpus.jsonl_payload(rows)


def _load_partial(
    path: Path,
    groups: Sequence[Sequence[dict]],
    contract_sha256: str,
    generation_pass_contract_sha256: str,
    nli_output_sha256: str,
    nli_receipt_self_sha256: str,
) -> list[dict]:
    if not path.exists():
        return []
    _regular_file(path, "fill semantic judge partial")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) > len(groups):
        raise RuntimeError("fill semantic judge partial exceeds request groups")
    for row, group in zip(rows, groups, strict=False):
        validate_record(
            row,
            group,
            contract_sha256,
            generation_pass_contract_sha256,
            nli_output_sha256,
            nli_receipt_self_sha256,
        )
    return rows


def _telemetry(
    *,
    shard_index: int,
    gpu_index: int,
    groups: int,
    completed: int,
    candidates: int,
    run_contract_sha256: str,
    status: str,
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
        "request_groups": groups,
        "completed_request_groups": completed,
        "candidates_completed": candidates,
        "run_contract_sha256": run_contract_sha256,
        "training_rows_emitted": False,
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    validate_reused_judge_semantics()
    packets, structural_summary = fill_nli.load_structural_packets(
        args.shard_index
    )
    # This call is deliberately before any Mistral-byte hashing.  Missing or
    # unsealed fill NLI artifacts block the lane immediately.
    nli_index, nli_identity = load_fill_nli_results(args.shard_index, packets)
    runtime_environment = validate_runtime_environment()
    model_receipts = validate_model_snapshot(args.model_directory)
    groups = groups_by_request(packets)
    if args.smoke:
        groups = groups[:1]
    rendered_prompts = prompt_receipts(groups)
    statistics = primary_judge.mistral_prompt_statistics(groups, args.max_tokens)
    if statistics["maximum_prompt_plus_output_budget"] > args.max_model_len:
        raise RuntimeError("fill semantic prompt plus output exceeds max model length")
    paths = output_paths(args.shard_index, smoke=args.smoke)
    contract = build_run_contract(
        shard_index=args.shard_index,
        structural_summary=structural_summary,
        nli_identity=nli_identity,
        model_receipts=model_receipts,
        runtime_environment=runtime_environment,
        paths=paths,
        args=args,
        rendered_prompts=rendered_prompts,
        prompt_statistics=statistics,
    )
    return {
        "packets": packets,
        "structural_summary": structural_summary,
        "nli_index": nli_index,
        "nli_identity": nli_identity,
        "groups": groups,
        "paths": paths,
        "contract": contract,
        "prompt_statistics": statistics,
    }


def preflight(args: argparse.Namespace) -> dict:
    prepared = _prepare(args)
    return {
        "status": "fill_semantic_judge_preflight_complete_no_gpu_launch",
        "generation_pass_id": PASS_ID,
        "gpu_shard": args.shard_index,
        "request_groups": len(prepared["groups"]),
        "candidates": sum(len(group) for group in prepared["groups"]),
        "sealed_fill_nli": prepared["nli_identity"],
        "run_contract": prepared["contract"],
        "planned_outputs": {
            key: corpus.relative(value)
            for key, value in prepared["paths"].items()
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
    _require_self_address(report, "pass-aware fill semantic judge report")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "sealed_complete_manual_and_global_selection_pending",
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
    prepared = _prepare(args)
    packets = prepared["packets"]
    groups = prepared["groups"]
    nli_index = prepared["nli_index"]
    nli_identity = prepared["nli_identity"]
    structural_summary = prepared["structural_summary"]
    paths = prepared["paths"]
    contract = prepared["contract"]
    if any(paths[name].exists() for name in ("output", "report", "receipt")):
        raise RuntimeError(
            "sealed fill semantic output exists; use --check-output instead of overwriting"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        raise RuntimeError(
            "fill semantic judge requires exactly its assigned physical GPU"
        )

    from vllm import LLM, SamplingParams, __version__ as vllm_version
    from vllm.sampling_params import StructuredOutputsParams

    if vllm_version != VLLM_VERSION:
        raise RuntimeError("fill semantic judge requires vLLM 0.25.0 exactly")
    engine = LLM(
        model=str(args.model_directory),
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        dtype=DTYPE,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=args.enforce_eager,
        seed=17,
    )
    structured = StructuredOutputsParams(
        json=primary_judge.GUIDED_SCHEMA,
        disable_additional_properties=True,
    )
    started_at = utc_now()
    contract_sha256 = contract["content_sha256_before_self_field"]
    pass_contract_sha256 = structural_summary["generation_pass_contract"][
        "content_sha256_before_self_field"
    ]
    nli_output_sha256 = nli_identity["output_sha256"]
    nli_receipt_self_sha256 = nli_identity["receipt_self_sha256"]
    rows = _load_partial(
        paths["partial"],
        groups,
        contract_sha256,
        pass_contract_sha256,
        nli_output_sha256,
        nli_receipt_self_sha256,
    )
    for batch_start in range(len(rows), len(groups), args.request_batch_size):
        batch_started = time.monotonic()
        batch_groups = groups[batch_start : batch_start + args.request_batch_size]
        conversations = []
        parameters = []
        mapping = []
        for group_index, group in enumerate(batch_groups):
            for pass_index, pass_name in enumerate(primary_judge.PASS_NAMES):
                conversations.append(primary_judge.judge_messages(group, pass_name))
                parameters.append(SamplingParams(
                    temperature=0.0,
                    seed=1009 + pass_index,
                    max_tokens=args.max_tokens,
                    structured_outputs=structured,
                ))
                mapping.append((group_index, pass_name))
        outputs = engine.chat(
            conversations, sampling_params=parameters, use_tqdm=False
        )
        if len(outputs) != len(mapping):
            raise RuntimeError("fill semantic judge output count changed")
        parsed_by_group: list[dict[str, dict]] = [dict() for _ in batch_groups]
        for output, (group_index, pass_name) in zip(outputs, mapping, strict=True):
            if len(output.outputs) != 1:
                raise RuntimeError("fill semantic judge output multiplicity changed")
            parsed_by_group[group_index][pass_name] = (
                primary_judge.parse_judge_output(output.outputs[0].text)
            )
        for group, pass_outputs in zip(
            batch_groups, parsed_by_group, strict=True
        ):
            record = make_record(
                group,
                pass_outputs,
                nli_index,
                run_contract_sha256=contract_sha256,
                generation_pass_contract_sha256=pass_contract_sha256,
                nli_output_sha256=nli_output_sha256,
                nli_output_receipt_self_sha256=nli_receipt_self_sha256,
            )
            validate_record(
                record,
                group,
                contract_sha256,
                pass_contract_sha256,
                nli_output_sha256,
                nli_receipt_self_sha256,
            )
            rows.append(record)
        _atomic_write(paths["partial"], _payload(rows))
        telemetry = _telemetry(
            shard_index=args.shard_index,
            gpu_index=args.gpu_index,
            groups=len(groups),
            completed=len(rows),
            candidates=sum(len(row["results"]) for row in rows),
            run_contract_sha256=contract_sha256,
            status="running",
        )
        telemetry["last_batch_seconds"] = time.monotonic() - batch_started
        _atomic_write(paths["telemetry"], corpus.canonical_bytes(telemetry))

    output_payload = _payload(rows)
    results = [result for row in rows for result in row["results"]]
    report = {
        "schema": REPORT_SCHEMA,
        "status": "complete_manual_and_global_selection_pending",
        "started_at": started_at,
        "completed_at": utc_now(),
        "generation_pass_id": PASS_ID,
        "gpu_shard": args.shard_index,
        "physical_gpu_index": args.gpu_index,
        "smoke": args.smoke,
        "request_groups": len(groups),
        "candidates": len(results),
        "judge_consensus_passed": sum(
            item["judge_consensus_passed"] for item in results
        ),
        "manual_review_required": sum(
            item["manual_review_required"] for item in results
        ),
        "generation_pass_contract_sha256": pass_contract_sha256,
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "nli_output_sha256": nli_output_sha256,
        "nli_output_receipt_self_sha256": nli_receipt_self_sha256,
        "output": corpus.relative(paths["output"]),
        "output_sha256": corpus.sha256_bytes(output_payload),
        "output_receipt": corpus.relative(paths["receipt"]),
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
        groups=len(groups),
        completed=len(rows),
        candidates=len(results),
        run_contract_sha256=contract_sha256,
        status="complete_manual_and_global_selection_pending",
    )))
    return report


def validate_completed_output(args: argparse.Namespace) -> dict:
    prepared = _prepare(args)
    groups = prepared["groups"]
    paths = prepared["paths"]
    contract = prepared["contract"]
    nli_identity = prepared["nli_identity"]
    structural_summary = prepared["structural_summary"]
    for name in ("output", "report", "receipt"):
        _regular_file(paths[name], f"sealed fill semantic judge {name}")
    output_payload = paths["output"].read_bytes()
    report_payload = paths["report"].read_bytes()
    report = json.loads(report_payload)
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    _require_self_address(report, "pass-aware fill semantic judge report")
    _require_self_address(receipt, "pass-aware fill semantic judge receipt")
    contract_sha256 = contract["content_sha256_before_self_field"]
    pass_contract_sha256 = structural_summary["generation_pass_contract"][
        "content_sha256_before_self_field"
    ]
    nli_output_sha256 = nli_identity["output_sha256"]
    nli_receipt_self_sha256 = nli_identity["receipt_self_sha256"]
    rows = [
        json.loads(line)
        for line in output_payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != len(groups):
        raise RuntimeError("fill semantic judge output row count changed")
    for row, group in zip(rows, groups, strict=True):
        validate_record(
            row,
            group,
            contract_sha256,
            pass_contract_sha256,
            nli_output_sha256,
            nli_receipt_self_sha256,
        )
    results = [result for row in rows for result in row["results"]]
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("status")
        != "complete_manual_and_global_selection_pending"
        or report.get("generation_pass_id") != PASS_ID
        or report.get("gpu_shard") != args.shard_index
        or report.get("smoke") != args.smoke
        or report.get("request_groups") != len(groups)
        or report.get("candidates") != len(results)
        or report.get("generation_pass_contract_sha256")
        != pass_contract_sha256
        or report.get("structural_review_sha256")
        != structural_summary["report_file_sha256"]
        or report.get("nli_output_sha256") != nli_output_sha256
        or report.get("nli_output_receipt_self_sha256")
        != nli_receipt_self_sha256
        or report.get("output") != corpus.relative(paths["output"])
        or report.get("output_sha256") != corpus.sha256_bytes(output_payload)
        or report.get("output_receipt") != corpus.relative(paths["receipt"])
        or report.get("run_contract") != contract
        or report.get("semantic_verification_completed") is not False
        or report.get("training_rows_emitted") is not False
        or any(item.get("eligible_for_training") is not False for item in results)
    ):
        raise RuntimeError("pass-aware fill semantic judge report changed")
    expected_receipt = build_output_receipt(
        shard_index=args.shard_index,
        paths=paths,
        output_payload=output_payload,
        report_payload=report_payload,
        report=report,
    )
    if receipt != expected_receipt:
        raise RuntimeError("pass-aware fill semantic judge output receipt changed")
    return {
        "status": "sealed_fill_semantic_judge_output_valid_selection_pending",
        "generation_pass_id": PASS_ID,
        "gpu_shard": args.shard_index,
        "request_groups": len(rows),
        "candidates": len(results),
        "output": corpus.relative(paths["output"]),
        "output_sha256": receipt["output_sha256"],
        "report": corpus.relative(paths["report"]),
        "report_file_sha256": receipt["report_file_sha256"],
        "report_self_sha256": receipt["report_self_sha256"],
        "receipt": corpus.relative(paths["receipt"]),
        "receipt_file_sha256": corpus.file_sha256(paths["receipt"]),
        "receipt_self_sha256": receipt["content_sha256_before_self_field"],
        "run_contract_sha256": contract_sha256,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--shard-index", required=True, type=int, choices=range(4))
    result.add_argument("--gpu-index", type=int)
    result.add_argument("--model-directory", type=Path, default=MODEL_DIRECTORY)
    result.add_argument(
        "--request-batch-size", type=int, default=DEFAULT_REQUEST_BATCH_SIZE
    )
    result.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    result.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    result.add_argument("--enforce-eager", action="store_true")
    result.add_argument("--smoke", action="store_true")
    modes = result.add_mutually_exclusive_group()
    modes.add_argument("--check-plan", action="store_true")
    modes.add_argument("--check-output", action="store_true")
    return result


def _validate_args(args: argparse.Namespace) -> None:
    if not 1 <= args.request_batch_size <= 16:
        raise ValueError("fill semantic judge request batch must be 1-16")
    if not 8_192 <= args.max_model_len <= 32_768:
        raise ValueError("fill semantic judge max model length is out of bounds")
    if not 1_024 <= args.max_tokens <= 4_096:
        raise ValueError("fill semantic judge output budget is out of bounds")
    if not 0.6 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("fill semantic judge GPU memory utilization is out of bounds")
    if not args.check_plan and not args.check_output:
        if args.gpu_index is None or args.gpu_index != args.shard_index:
            raise ValueError(
                "fill semantic judge shard 0-3 must use its matching GPU"
            )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    _validate_args(args)
    if args.check_plan:
        value = preflight(args)
    elif args.check_output:
        value = validate_completed_output(args)
    else:
        value = run(args)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
