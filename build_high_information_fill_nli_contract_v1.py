#!/usr/bin/env python3
"""Build the sealed four-shard launch contract for the isolated fill NLI lane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_high_information_domain_corpus_v1 as corpus
import run_high_information_fill_nli_prefilter_v1 as fill_nli


DEFAULT_OUTPUT = (
    corpus.OUTPUT_DIR
    / "nli_prefilter_quality_deficit_fill_v1.launch_contract.json"
)
SCHEMA = "high-information-pass-aware-fill-nli-launch-contract-v1"
BATCH_SIZE = 64
PYTHON_EXECUTABLE = "es-at-scale/.venv/bin/python"
CU13_LIBRARY_PATH = (
    "/home/catid/specialist/es-at-scale/.venv/lib/python3.12/"
    "site-packages/nvidia/cu13/lib"
)


def safe_run_command(shard_index: int) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={shard_index} "
        f"LD_LIBRARY_PATH={CU13_LIBRARY_PATH} {PYTHON_EXECUTABLE} "
        "run_high_information_fill_nli_prefilter_v1.py "
        f"--shard-index {shard_index} --gpu-index {shard_index} "
        f"--batch-size {BATCH_SIZE}"
    )


def safe_preflight_command(shard_index: int) -> str:
    return (
        f"LD_LIBRARY_PATH={CU13_LIBRARY_PATH} {PYTHON_EXECUTABLE} "
        "run_high_information_fill_nli_prefilter_v1.py "
        f"--shard-index {shard_index} --check-plan"
    )


def safe_output_check_command(shard_index: int) -> str:
    return (
        f"LD_LIBRARY_PATH={CU13_LIBRARY_PATH} {PYTHON_EXECUTABLE} "
        "run_high_information_fill_nli_prefilter_v1.py "
        f"--shard-index {shard_index} --check-output"
    )


def build_contract() -> dict:
    shards = []
    shared_model = None
    shared_runtime = None
    shared_implementations = None
    shared_fill_receipts = None
    for shard_index in range(4):
        args = argparse.Namespace(
            shard_index=shard_index,
            gpu_index=None,
            model_directory=fill_nli.MODEL_DIRECTORY,
            batch_size=BATCH_SIZE,
            smoke=False,
            check_plan=True,
            check_output=False,
        )
        preflight = fill_nli.preflight(args)
        run_contract = preflight["run_contract"]
        model = run_contract["model"]
        runtime = run_contract["runtime"]
        implementations = run_contract["implementation_receipts"]
        fill_receipts = run_contract["generation_pass"][
            "runtime_worker_receipts"
        ]
        if shared_model is None:
            shared_model = model
            shared_runtime = runtime
            shared_implementations = implementations
            shared_fill_receipts = fill_receipts
        elif (
            model != shared_model
            or runtime != shared_runtime
            or implementations != shared_implementations
            or fill_receipts != shared_fill_receipts
        ):
            raise RuntimeError("fill NLI shared runtime receipts differ by shard")
        shards.append({
            "gpu_shard": shard_index,
            "packets": preflight["packets"],
            "positive_packets": preflight["positive_packets"],
            "hard_negative_packets": preflight["hard_negative_packets"],
            "generation_pass_contract_sha256": preflight[
                "generation_pass_contract_sha256"
            ],
            "structural_review_sha256": preflight[
                "structural_review_sha256"
            ],
            "run_contract_sha256": run_contract[
                "content_sha256_before_self_field"
            ],
            "planned_outputs": preflight["planned_outputs"],
            "preflight_command": safe_preflight_command(shard_index),
            "run_command": safe_run_command(shard_index),
            "post_run_output_check_command": safe_output_check_command(
                shard_index
            ),
            "post_run_receipt_required": True,
            "semantic_verification_completed": False,
            "training_rows_emitted": False,
        })
    assert shared_model is not None
    assert shared_runtime is not None
    assert shared_implementations is not None
    assert shared_fill_receipts is not None
    contract = {
        "schema": SCHEMA,
        "status": "ready_for_explicit_four_gpu_nli_launch",
        "generation_pass": {
            "id": fill_nli.PASS_ID,
            "temperature": fill_nli.FILL_TEMPERATURE,
            "top_p": fill_nli.FILL_TOP_P,
            "seed_scheme": fill_nli.FILL_SEED_SCHEME,
            "runtime_worker_receipts": shared_fill_receipts,
        },
        "model": shared_model,
        "runtime": shared_runtime,
        "runtime_environment": {
            "python_executable": PYTHON_EXECUTABLE,
            "ld_library_path": CU13_LIBRARY_PATH,
        },
        "nli_semantics": {
            "implementation": "run_high_information_nli_prefilter_v1.py",
            "max_length": fill_nli.MAX_LENGTH,
            "thresholds": {
                "entailment_minimum": fill_nli.ENTAILMENT_MINIMUM,
                "contradiction_maximum": fill_nli.CONTRADICTION_MAXIMUM,
                "contradiction_failure": fill_nli.CONTRADICTION_FAILURE,
            },
            "hard_negative_policy": (
                "not_applicable_requires_absence_or_false_premise_verifier"
            ),
        },
        "implementation_receipts": shared_implementations,
        "contract_builder_receipt": {
            "path": corpus.relative(Path(__file__).resolve()),
            "file_sha256": corpus.file_sha256(Path(__file__).resolve()),
        },
        "batch_size": BATCH_SIZE,
        "shards": shards,
        "post_run_hash_binding": {
            "receipt_schema": fill_nli.RECEIPT_SCHEMA,
            "required_fields": [
                "output_sha256",
                "report_file_sha256",
                "report_self_sha256",
                "run_contract_sha256",
                "content_sha256_before_self_field",
            ],
            "output_or_report_without_matching_receipt_is_valid": False,
        },
        "policy": {
            "active_primary_nli_or_judge_modified": False,
            "protected_development_final_holdout_ood_terminal_sources_read": False,
            "gpu_job_launched_by_contract_builder": False,
            "semantic_verification_completed": False,
            "training_rows_emitted": False,
        },
    }
    contract["content_sha256_before_self_field"] = corpus.canonical_sha256(contract)
    return contract


def contract_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--print-contract", action="store_true")
    result.add_argument("--check", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    value = build_contract()
    payload = contract_bytes(value)
    if args.check:
        if args.output.is_symlink() or not args.output.is_file():
            raise RuntimeError("fill NLI launch contract is missing")
        if args.output.read_bytes() != payload:
            raise RuntimeError("fill NLI launch contract changed")
    if args.print_contract:
        print(payload.decode(), end="")
    if not args.check and not args.print_contract:
        print(json.dumps({
            "status": value["status"],
            "output": corpus.relative(args.output),
            "content_sha256": value["content_sha256_before_self_field"],
            "shards": len(value["shards"]),
            "training_rows_emitted": False,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
