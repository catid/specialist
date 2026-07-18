#!/usr/bin/env python3
"""Run a pinned, train-only NLI prefilter over structural candidate passes.

The prefilter is deliberately not a semantic acceptance mechanism.  It scores
positive answers against their exact cited source excerpts with a model that is
independent of the Qwen generator.  Hard negatives are left to the independent
semantic judge because textual entailment cannot establish absence from a
source.  Every output remains training-ineligible.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import build_high_information_domain_corpus_v1 as corpus
import verify_high_information_candidates_v1 as structural
import verify_high_information_semantic_decisions_v1 as semantic


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
TRANSFORMERS_VERSION = "5.13.1"
TORCH_VERSION = "2.11.0+cu130"
MAX_LENGTH = 512
ENTAILMENT_MINIMUM = 0.70
CONTRADICTION_MAXIMUM = 0.10
CONTRADICTION_FAILURE = 0.50
RESULT_SCHEMA = "high-information-nli-prefilter-result-v1"


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


def validate_model_snapshot(model_directory: Path) -> dict[str, str]:
    model_directory = model_directory.expanduser().resolve()
    if model_directory != MODEL_DIRECTORY:
        raise RuntimeError("NLI prefilter requires the pinned local model revision")
    observed = {}
    for name, expected_blob in MODEL_BLOB_RECEIPTS.items():
        path = model_directory / name
        if not path.is_symlink() or not path.is_file():
            raise RuntimeError(f"pinned NLI snapshot file is missing: {name}")
        target = os.readlink(path)
        observed_blob = Path(target).name
        if observed_blob != expected_blob:
            raise RuntimeError(f"pinned NLI snapshot file changed: {name}")
        observed[name] = observed_blob
    return observed


def shard_paths(shard_index: int, *, smoke: bool = False) -> dict[str, Path]:
    suffix = ".smoke" if smoke else ""
    stem = f"nli_prefilter_sealed_gpu{shard_index}{suffix}"
    return {
        "partial": corpus.OUTPUT_DIR / f"{stem}.partial.jsonl",
        "output": corpus.OUTPUT_DIR / f"{stem}.jsonl",
        "report": corpus.OUTPUT_DIR / f"{stem}.report.json",
        "telemetry": corpus.OUTPUT_DIR / f"{stem}.telemetry.json",
    }


def structural_paths(shard_index: int) -> tuple[Path, Path]:
    return (
        corpus.OUTPUT_DIR / f"candidate_structural_review_gpu{shard_index}.jsonl",
        corpus.OUTPUT_DIR
        / f"candidate_structural_review_gpu{shard_index}.summary.json",
    )


def load_structural_packets(
    shard_index: int,
    review_path: Path,
    summary_path: Path,
) -> tuple[list[dict], dict]:
    review_path = review_path.resolve()
    summary_path = summary_path.resolve()
    expected_review, expected_summary = structural_paths(shard_index)
    if review_path != expected_review.resolve() or summary_path != expected_summary.resolve():
        raise RuntimeError("NLI input must be the canonical structural shard output")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    _require_self_address(summary, "structural summary")
    if (
        summary.get("schema") != "high-information-candidate-structural-review-v1"
        or summary.get("report_path") != corpus.relative(review_path)
        or summary.get("report_file_sha256") != corpus.file_sha256(review_path)
        or summary.get("semantic_verification_completed") is not False
        or summary.get("accepted_training_rows_emitted") is not False
    ):
        raise RuntimeError("structural summary contract changed")

    _, contexts, requests = structural.load_plan()
    rows = [
        json.loads(line)
        for line in review_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != summary.get("requests_with_candidate_records"):
        raise RuntimeError("structural review row count changed")
    packets = []
    seen: set[str] = set()
    for row in rows:
        _require_self_address(row, "structural review row")
        candidates = row.get("structurally_valid_examples", [])
        if not candidates:
            continue
        if row.get("status") not in {
            "structurally_valid_semantic_verification_pending",
            "partially_or_fully_rejected",
        }:
            raise RuntimeError("structurally valid examples have an invalid row status")
        request_id = row.get("request_id")
        if request_id not in requests or requests[request_id]["gpu_shard"] != shard_index:
            raise RuntimeError("structural review crossed request-shard lineage")
        request = requests[request_id]
        context = contexts[request["source_context_id"]]
        for candidate in candidates:
            packet = semantic.build_packet(candidate, request, context)
            if packet["candidate_example_id"] in seen:
                raise RuntimeError("structural review duplicates candidate identity")
            seen.add(packet["candidate_example_id"])
            packets.append(packet)
    if len(packets) != summary.get("structurally_valid_examples"):
        raise RuntimeError("structural summary example count changed")
    packets.sort(key=lambda value: (value["request_id"], value["candidate_example_id"]))
    return packets, summary


def nli_inputs(packet: dict) -> tuple[str, str] | None:
    if packet["generation_mode"] == "calibrated_hard_negative":
        return None
    premise = "\n".join(packet["evidence_quotes"])
    hypothesis = packet["answer"]
    if not premise.strip() or not hypothesis.strip():
        raise RuntimeError("NLI packet has an empty premise or hypothesis")
    return premise, hypothesis


def classify_probabilities(
    probabilities: dict[str, float], *, input_truncated: bool
) -> tuple[str, bool, list[str]]:
    if set(probabilities) != {"entailment", "neutral", "contradiction"}:
        raise RuntimeError("NLI labels changed")
    if any(value < 0.0 or value > 1.0 for value in probabilities.values()):
        raise RuntimeError("NLI probability is outside [0, 1]")
    reasons = []
    if input_truncated:
        reasons.append("nli_input_truncated")
    if (
        probabilities["entailment"] >= ENTAILMENT_MINIMUM
        and probabilities["contradiction"] <= CONTRADICTION_MAXIMUM
        and not input_truncated
    ):
        return "pass", False, reasons
    if probabilities["contradiction"] >= CONTRADICTION_FAILURE:
        return "fail", False, reasons
    reasons.append("nli_confidence_between_accept_and_reject_thresholds")
    return "uncertain", True, reasons


def make_result(
    packet: dict,
    *,
    probabilities: dict[str, float] | None,
    input_token_count: int | None,
    input_truncated: bool,
    run_contract_sha256: str,
) -> dict:
    inputs = nli_inputs(packet)
    if inputs is None:
        verdict = "not_applicable"
        manual_review_required = False
        review_reasons = ["hard_negative_requires_absence_or_false_premise_verifier"]
        premise_sha256 = None
        hypothesis_sha256 = None
    else:
        if probabilities is None or input_token_count is None:
            raise RuntimeError("positive NLI result is missing model output")
        verdict, manual_review_required, review_reasons = classify_probabilities(
            probabilities, input_truncated=input_truncated
        )
        premise, hypothesis = inputs
        premise_sha256 = corpus.sha256_bytes(premise.encode("utf-8"))
        hypothesis_sha256 = corpus.sha256_bytes(hypothesis.encode("utf-8"))
    row = {
        "schema": RESULT_SCHEMA,
        "packet_id": packet["packet_id"],
        "candidate_example_id": packet["candidate_example_id"],
        "request_id": packet["request_id"],
        "source_group_id": packet["source_group_id"],
        "generation_mode": packet["generation_mode"],
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "torch_version": TORCH_VERSION,
            "dtype": "bfloat16",
        },
        "premise_construction": "ordered_exact_evidence_quotes_joined_by_newline",
        "premise_sha256": premise_sha256,
        "hypothesis_sha256": hypothesis_sha256,
        "input_token_count_before_truncation": input_token_count,
        "max_length": MAX_LENGTH,
        "input_truncated": input_truncated,
        "probabilities": probabilities,
        "thresholds": {
            "entailment_minimum": ENTAILMENT_MINIMUM,
            "contradiction_maximum": CONTRADICTION_MAXIMUM,
            "contradiction_failure": CONTRADICTION_FAILURE,
        },
        "run_contract_sha256": run_contract_sha256,
        "verdict": verdict,
        "manual_review_required": manual_review_required,
        "manual_review_reasons": review_reasons,
        "semantic_verification_completed": False,
        "eligible_for_training": False,
    }
    row["content_sha256_before_self_field"] = _self_address(row)
    return row


def validate_result(row: dict, packet: dict, run_contract_sha256: str) -> None:
    _require_self_address(row, "NLI result")
    if (
        row.get("schema") != RESULT_SCHEMA
        or row.get("packet_id") != packet["packet_id"]
        or row.get("candidate_example_id") != packet["candidate_example_id"]
        or row.get("request_id") != packet["request_id"]
        or row.get("source_group_id") != packet["source_group_id"]
        or row.get("model", {}).get("revision") != MODEL_REVISION
        or row.get("model")
        != {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "torch_version": TORCH_VERSION,
            "dtype": "bfloat16",
        }
        or row.get("premise_construction")
        != "ordered_exact_evidence_quotes_joined_by_newline"
        or row.get("max_length") != MAX_LENGTH
        or row.get("thresholds")
        != {
            "entailment_minimum": ENTAILMENT_MINIMUM,
            "contradiction_maximum": CONTRADICTION_MAXIMUM,
            "contradiction_failure": CONTRADICTION_FAILURE,
        }
        or row.get("run_contract_sha256") != run_contract_sha256
        or row.get("semantic_verification_completed") is not False
        or row.get("eligible_for_training") is not False
    ):
        raise RuntimeError("NLI result identity changed")


def _load_partial(
    path: Path, packets: Sequence[dict], run_contract_sha256: str
) -> list[dict]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) > len(packets):
        raise RuntimeError("NLI partial output exceeds packet shard")
    for row, packet in zip(rows, packets, strict=False):
        validate_result(row, packet, run_contract_sha256)
    return rows


def build_run_contract(
    *,
    shard_index: int,
    structural_summary: dict,
    model_receipts: dict[str, str],
) -> dict:
    contract = {
        "schema": "high-information-nli-prefilter-run-contract-v1",
        "gpu_shard": shard_index,
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "blob_receipts": model_receipts,
        },
        "runtime": {
            "transformers_version": TRANSFORMERS_VERSION,
            "torch_version": TORCH_VERSION,
            "dtype": "bfloat16",
        },
        "premise_construction": "ordered_exact_evidence_quotes_joined_by_newline",
        "max_length": MAX_LENGTH,
        "thresholds": {
            "entailment_minimum": ENTAILMENT_MINIMUM,
            "contradiction_maximum": CONTRADICTION_MAXIMUM,
            "contradiction_failure": CONTRADICTION_FAILURE,
        },
        "worker_file_sha256": corpus.file_sha256(Path(__file__).resolve()),
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
        "schema": "high-information-nli-prefilter-telemetry-v1",
        "updated_at": utc_now(),
        "status": status,
        "pid": os.getpid(),
        "gpu_shard": shard_index,
        "physical_gpu_index": gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "packets": packets,
        "completed_packets": completed,
    }


def preflight(args: argparse.Namespace) -> dict:
    model_receipts = validate_model_snapshot(args.model_directory)
    review, summary = structural_paths(args.shard_index)
    packets, structural_summary = load_structural_packets(
        args.shard_index, review, summary
    )
    contract = build_run_contract(
        shard_index=args.shard_index,
        structural_summary=structural_summary,
        model_receipts=model_receipts,
    )
    return {
        "status": "nli_preflight_complete_no_gpu_launch",
        "gpu_shard": args.shard_index,
        "packets": len(packets),
        "positive_packets": sum(nli_inputs(packet) is not None for packet in packets),
        "hard_negative_packets": sum(nli_inputs(packet) is None for packet in packets),
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "model_revision": MODEL_REVISION,
        "model_blob_receipts": model_receipts,
        "run_contract_sha256": contract["content_sha256_before_self_field"],
        "training_rows_emitted": False,
    }


def run(args: argparse.Namespace) -> dict:
    started_at = utc_now()
    model_receipts = validate_model_snapshot(args.model_directory)
    review_path, summary_path = structural_paths(args.shard_index)
    packets, structural_summary = load_structural_packets(
        args.shard_index, review_path, summary_path
    )
    contract = build_run_contract(
        shard_index=args.shard_index,
        structural_summary=structural_summary,
        model_receipts=model_receipts,
    )
    contract_sha256 = contract["content_sha256_before_self_field"]
    if args.smoke:
        positive = next((item for item in packets if nli_inputs(item) is not None), None)
        negative = next((item for item in packets if nli_inputs(item) is None), None)
        packets = [item for item in (positive, negative) if item is not None]
    paths = shard_paths(args.shard_index, smoke=args.smoke)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        raise RuntimeError("NLI prefilter requires exactly its assigned physical GPU")

    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if transformers.__version__ != TRANSFORMERS_VERSION or torch.__version__ != TORCH_VERSION:
        raise RuntimeError("NLI runtime package version changed")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_directory, local_files_only=True, use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_directory,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).eval().to("cuda:0")
    labels = {int(index): label.casefold() for index, label in model.config.id2label.items()}
    if labels != {0: "entailment", 1: "neutral", 2: "contradiction"}:
        raise RuntimeError("pinned NLI label order changed")

    rows = _load_partial(paths["partial"], packets, contract_sha256)
    completed = len(rows)
    for batch_start in range(completed, len(packets), args.batch_size):
        batch_started = time.monotonic()
        batch_packets = packets[batch_start : batch_start + args.batch_size]
        positive_packets = [
            packet for packet in batch_packets if nli_inputs(packet) is not None
        ]
        inference: dict[str, tuple[dict[str, float], int, bool]] = {}
        if positive_packets:
            pairs = [nli_inputs(packet) for packet in positive_packets]
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
                    {labels[index]: float(value) for index, value in enumerate(vector)},
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
                run_contract_sha256=contract_sha256,
            )
            validate_result(row, packet, contract_sha256)
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
    _atomic_write(paths["output"], output_payload)
    report = {
        "schema": "high-information-nli-prefilter-report-v1",
        "status": "complete_semantic_pending",
        "started_at": started_at,
        "completed_at": utc_now(),
        "gpu_shard": args.shard_index,
        "physical_gpu_index": args.gpu_index,
        "packets": len(packets),
        "pass": sum(row["verdict"] == "pass" for row in rows),
        "fail": sum(row["verdict"] == "fail" for row in rows),
        "uncertain": sum(row["verdict"] == "uncertain" for row in rows),
        "not_applicable": sum(row["verdict"] == "not_applicable" for row in rows),
        "manual_review_required": sum(row["manual_review_required"] for row in rows),
        "structural_review_sha256": structural_summary["report_file_sha256"],
        "output": corpus.relative(paths["output"]),
        "output_sha256": corpus.sha256_bytes(output_payload),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "model_blob_receipts": model_receipts,
        "worker_file_sha256": contract["worker_file_sha256"],
        "run_contract": contract,
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    report["content_sha256_before_self_field"] = _self_address(report)
    _atomic_write(paths["report"], (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"))
    _atomic_write(paths["telemetry"], corpus.canonical_bytes(_telemetry(
        shard_index=args.shard_index,
        gpu_index=args.gpu_index,
        packets=len(packets),
        completed=len(rows),
        status="complete_semantic_pending",
    )))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--shard-index", required=True, type=int)
    result.add_argument("--gpu-index", required=True, type=int)
    result.add_argument("--model-directory", type=Path, default=MODEL_DIRECTORY)
    result.add_argument("--batch-size", type=int, default=64)
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--check-plan", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.shard_index not in range(4) or args.gpu_index != args.shard_index:
        raise ValueError("NLI shard 0-3 must be pinned to the matching GPU")
    if not 1 <= args.batch_size <= 512:
        raise ValueError("NLI batch size must be 1-512")
    result = preflight(args) if args.check_plan else run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
