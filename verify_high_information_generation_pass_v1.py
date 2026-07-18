#!/usr/bin/env python3
"""Pass-aware structural verification for high-information candidate pools.

Generation passes intentionally reuse request IDs.  This verifier binds each
pool to an explicit pass ID, exact candidate/report paths, implementation
receipt, sampling recipe, and per-row content address before emitting separate
structural-review artifacts.  It never merges passes and never emits training
rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import build_high_information_domain_corpus_v1 as corpus
import run_high_information_candidate_shard_v1 as primary_worker
import verify_high_information_candidates_v1 as structural


PRIMARY_PASS_ID = "primary-generation-v1"
FILL_PASS_ID = "quality-deficit-fill-v1"
PASS_SPECS = {
    PRIMARY_PASS_ID: {
        "candidate_stem": "generation_candidates_gpu{shard}",
        "worker": corpus.ROOT / "run_high_information_candidate_shard_v1.py",
        "runtime_workers": (
            corpus.ROOT / "run_high_information_candidate_shard_v1.py",
        ),
        "temperature": 0.3,
        "top_p": 0.9,
        "seed_scheme": "sha256(request_id) first 32 bits",
        "generator_generation_pass": None,
    },
    FILL_PASS_ID: {
        "candidate_stem": "generation_fill_candidates_gpu{shard}",
        "worker": corpus.ROOT / "run_high_information_candidate_fill_shard_v1.py",
        "runtime_workers": (
            corpus.ROOT / "run_high_information_candidate_fill_shard_v1.py",
            corpus.ROOT / "run_high_information_candidate_shard_v1.py",
        ),
        "temperature": 0.4,
        "top_p": 0.9,
        "seed_scheme": "sha256('quality-deficit-fill-v1:' + request_id) first 32 bits",
        "generator_generation_pass": FILL_PASS_ID,
    },
}


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


def pass_slug(pass_id: str) -> str:
    if pass_id == PRIMARY_PASS_ID:
        return "primary_v1"
    if pass_id == FILL_PASS_ID:
        return "quality_deficit_fill_v1"
    raise ValueError("unknown generation pass ID")


def generation_paths(pass_id: str, shard_index: int) -> tuple[Path, Path]:
    if pass_id not in PASS_SPECS or shard_index not in range(4):
        raise ValueError("unknown generation pass or shard")
    stem = PASS_SPECS[pass_id]["candidate_stem"].format(shard=shard_index)
    return (
        corpus.OUTPUT_DIR / f"{stem}.jsonl",
        corpus.OUTPUT_DIR / f"{stem}.report.json",
    )


def structural_paths(pass_id: str, shard_index: int) -> tuple[Path, Path]:
    slug = pass_slug(pass_id)
    stem = f"candidate_structural_review_{slug}_gpu{shard_index}"
    return (
        corpus.OUTPUT_DIR / f"{stem}.jsonl",
        corpus.OUTPUT_DIR / f"{stem}.summary.json",
    )


def expected_seed(pass_id: str, request_id: str) -> int:
    if pass_id == PRIMARY_PASS_ID:
        payload = request_id
    elif pass_id == FILL_PASS_ID:
        payload = f"{FILL_PASS_ID}:{request_id}"
    else:
        raise ValueError("unknown generation pass ID")
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def validate_generator(generator: Any, pass_id: str, request_id: str) -> None:
    spec = PASS_SPECS[pass_id]
    expected = {
        "model": "Qwen3.6-35B-A3B",
        "checkpoint": "sealed_local_base",
        "engine": "vllm-0.25.0",
        "dtype": "bfloat16",
        "temperature": spec["temperature"],
        "top_p": spec["top_p"],
        "seed": expected_seed(pass_id, request_id),
        "enable_thinking": False,
    }
    if spec["generator_generation_pass"] is not None:
        expected["generation_pass"] = spec["generator_generation_pass"]
    if generator != expected:
        raise RuntimeError("candidate generator does not match its generation pass")


def load_generation_pass(
    pass_id: str, shard_index: int
) -> tuple[list[dict], list[dict], dict, dict, dict[str, dict], dict[str, dict]]:
    if pass_id not in PASS_SPECS or shard_index not in range(4):
        raise ValueError("unknown generation pass or shard")
    prompt_spec, contexts, request_index = structural.load_plan()
    requests = sorted(
        (
            request
            for request in request_index.values()
            if request["gpu_shard"] == shard_index
        ),
        key=lambda value: value["request_id"],
    )
    candidate_path, report_path = generation_paths(pass_id, shard_index)
    if any(path.is_symlink() or not path.is_file() for path in (candidate_path, report_path)):
        raise RuntimeError("generation pass output/report pair is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require_self_address(report, "generation pass report")
    spec = PASS_SPECS[pass_id]
    worker = Path(spec["worker"]).resolve()
    runtime_workers = tuple(Path(path).resolve() for path in spec["runtime_workers"])
    candidate_payload = candidate_path.read_bytes()
    if (
        report.get("schema") != "high-information-candidate-shard-report-v1"
        or report.get("status") != "complete_unverified"
        or report.get("gpu_shard") != shard_index
        or report.get("physical_gpu_index") != shard_index
        or report.get("requests") != len(requests)
        or report.get("candidate_records") != len(requests)
        or report.get("output") != corpus.relative(candidate_path)
        or report.get("output_sha256") != corpus.sha256_bytes(candidate_payload)
        or report.get("worker_file_sha256") != corpus.file_sha256(worker)
        or report.get("vllm_version") != "0.25.0"
        or report.get("dtype") != "bfloat16"
        or report.get("tensor_parallel_size") != 1
        or report.get("generation_config") != "vllm"
        or report.get("model_files") != primary_worker.MODEL_FILE_SHA256
        or report.get("semantic_verification_completed") is not False
        or report.get("training_rows_emitted") is not False
    ):
        raise RuntimeError("generation pass report contract changed")
    candidates = [
        json.loads(line)
        for line in candidate_payload.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(candidates) != len(requests):
        raise RuntimeError("generation pass candidate count changed")
    for candidate, request in zip(candidates, requests, strict=True):
        errors = structural.validate_generated_candidate_envelope(candidate, request)
        if errors:
            raise RuntimeError(
                "generation pass envelope changed before structural review: "
                + ",".join(errors)
            )
        validate_generator(candidate.get("generator"), pass_id, request["request_id"])

    pass_contract = {
        "schema": "high-information-generation-pass-contract-v1",
        "generation_pass_id": pass_id,
        "gpu_shard": shard_index,
        "candidate_path": corpus.relative(candidate_path),
        "candidate_file_sha256": corpus.sha256_bytes(candidate_payload),
        "generation_report_path": corpus.relative(report_path),
        "generation_report_file_sha256": corpus.file_sha256(report_path),
        "generation_report_self_sha256": report["content_sha256_before_self_field"],
        "worker_path": corpus.relative(worker),
        "worker_file_sha256": corpus.file_sha256(worker),
        "runtime_worker_receipts": [
            {
                "path": corpus.relative(path),
                "file_sha256": corpus.file_sha256(path),
            }
            for path in runtime_workers
        ],
        "temperature": spec["temperature"],
        "top_p": spec["top_p"],
        "seed_scheme": spec["seed_scheme"],
        "prompt_spec_sha256": prompt_spec["content_sha256_before_self_field"],
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    pass_contract["content_sha256_before_self_field"] = _self_address(pass_contract)
    return candidates, requests, report, pass_contract, contexts, request_index


def _address_review_row(row: dict) -> dict:
    value = dict(row)
    value["schema"] = "high-information-pass-aware-structural-review-row-v1"
    value["content_sha256_before_self_field"] = _self_address(value)
    return value


def verify_pass(pass_id: str, shard_index: int) -> dict:
    (
        candidates,
        requests,
        _,
        pass_contract,
        contexts,
        request_index,
    ) = load_generation_pass(pass_id, shard_index)
    tokenizer = corpus.load_tokenizer()
    reports = []
    for candidate, request in zip(candidates, requests, strict=True):
        review = structural.verify_candidate_record(
            candidate,
            requests=request_index,
            contexts=contexts,
            tokenizer=tokenizer,
        )
        review["generation_pass_id"] = pass_id
        review["generation_pass_contract_sha256"] = pass_contract[
            "content_sha256_before_self_field"
        ]
        review["candidate_record_sha256"] = candidate[
            "content_sha256_before_self_field"
        ]
        reports.append(_address_review_row(review))

    report_path, summary_path = structural_paths(pass_id, shard_index)
    report_payload = corpus.jsonl_payload(reports)
    summary = {
        "schema": "high-information-pass-aware-structural-review-v1",
        "generation_pass_id": pass_id,
        "gpu_shard": shard_index,
        "generation_pass_contract": pass_contract,
        "requests_with_candidate_records": len(reports),
        "fully_structurally_valid_requests": sum(
            item["status"] == "structurally_valid_semantic_verification_pending"
            for item in reports
        ),
        "structurally_valid_examples": sum(
            len(item.get("structurally_valid_examples", [])) for item in reports
        ),
        "structurally_valid_assistant_tokens": sum(
            item.get("structurally_valid_assistant_tokens", 0) for item in reports
        ),
        "report_path": corpus.relative(report_path),
        "report_file_sha256": corpus.sha256_bytes(report_payload),
        "semantic_verification_completed": False,
        "accepted_training_rows_emitted": False,
    }
    summary["content_sha256_before_self_field"] = _self_address(summary)
    summary_payload = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _atomic_write(report_path, report_payload)
    _atomic_write(summary_path, summary_payload)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-pass-id", choices=sorted(PASS_SPECS), required=True)
    parser.add_argument("--shard-index", type=int, choices=range(4), required=True)
    parser.add_argument("--check-input", action="store_true")
    args = parser.parse_args()
    if args.check_input:
        candidates, requests, _, contract, _, _ = load_generation_pass(
            args.generation_pass_id, args.shard_index
        )
        value = {
            "status": "generation_pass_input_valid_semantic_pending",
            "generation_pass_id": args.generation_pass_id,
            "gpu_shard": args.shard_index,
            "candidates": len(candidates),
            "requests": len(requests),
            "generation_pass_contract_sha256": contract[
                "content_sha256_before_self_field"
            ],
            "training_rows_emitted": False,
        }
    else:
        value = verify_pass(args.generation_pass_id, args.shard_index)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
