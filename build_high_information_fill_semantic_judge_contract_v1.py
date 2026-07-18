#!/usr/bin/env python3
"""Build the fail-closed launch scaffold for the isolated fill semantic judge.

No semantic-judge command is emitted until all four fill NLI shards have a
sealed output/report/receipt trio and each trio passes the fill NLI validator.
The blocked scaffold still binds the generation, structural, implementation,
model, runtime, schema, and per-rendered-prompt identities needed by the
eventual run contracts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import build_high_information_domain_corpus_v1 as corpus
import run_high_information_fill_nli_prefilter_v1 as fill_nli
import run_high_information_fill_semantic_judge_v1 as fill_judge
import run_high_information_semantic_judge_shard_v1 as primary_judge


DEFAULT_OUTPUT = (
    corpus.OUTPUT_DIR
    / "semantic_judge_quality_deficit_fill_v1.launch_contract.json"
)
SCHEMA = "high-information-pass-aware-fill-semantic-judge-launch-contract-v1"
PYTHON_EXECUTABLE = "es-at-scale/.venv/bin/python"
CU13_LIBRARY_PATH = str(fill_judge.CU13_LIBRARY_PATH)
REQUEST_BATCH_SIZE = fill_judge.DEFAULT_REQUEST_BATCH_SIZE
MAX_MODEL_LEN = fill_judge.DEFAULT_MAX_MODEL_LEN
MAX_TOKENS = fill_judge.DEFAULT_MAX_TOKENS
GPU_MEMORY_UTILIZATION = 0.90


def _args(shard_index: int) -> argparse.Namespace:
    return argparse.Namespace(
        shard_index=shard_index,
        gpu_index=None,
        model_directory=fill_judge.MODEL_DIRECTORY,
        request_batch_size=REQUEST_BATCH_SIZE,
        max_model_len=MAX_MODEL_LEN,
        max_tokens=MAX_TOKENS,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        enforce_eager=False,
        smoke=False,
        check_plan=True,
        check_output=False,
    )


def safe_run_command(shard_index: int) -> str:
    return (
        f"CUDA_VISIBLE_DEVICES={shard_index} "
        f"LD_LIBRARY_PATH={CU13_LIBRARY_PATH} {PYTHON_EXECUTABLE} "
        "run_high_information_fill_semantic_judge_v1.py "
        f"--shard-index {shard_index} --gpu-index {shard_index} "
        f"--request-batch-size {REQUEST_BATCH_SIZE} "
        f"--max-model-len {MAX_MODEL_LEN} --max-tokens {MAX_TOKENS} "
        f"--gpu-memory-utilization {GPU_MEMORY_UTILIZATION:.2f}"
    )


def safe_preflight_command(shard_index: int) -> str:
    return (
        f"LD_LIBRARY_PATH={CU13_LIBRARY_PATH} {PYTHON_EXECUTABLE} "
        "run_high_information_fill_semantic_judge_v1.py "
        f"--shard-index {shard_index} "
        f"--request-batch-size {REQUEST_BATCH_SIZE} "
        f"--max-model-len {MAX_MODEL_LEN} --max-tokens {MAX_TOKENS} "
        f"--gpu-memory-utilization {GPU_MEMORY_UTILIZATION:.2f} --check-plan"
    )


def safe_output_check_command(shard_index: int) -> str:
    return (
        f"LD_LIBRARY_PATH={CU13_LIBRARY_PATH} {PYTHON_EXECUTABLE} "
        "run_high_information_fill_semantic_judge_v1.py "
        f"--shard-index {shard_index} "
        f"--request-batch-size {REQUEST_BATCH_SIZE} "
        f"--max-model-len {MAX_MODEL_LEN} --max-tokens {MAX_TOKENS} "
        f"--gpu-memory-utilization {GPU_MEMORY_UTILIZATION:.2f} --check-output"
    )


def _required_nli_paths(shard_index: int) -> dict[str, str]:
    paths = fill_nli.output_paths(shard_index, smoke=False)
    return {
        name: corpus.relative(paths[name])
        for name in ("output", "report", "receipt")
    }


def _nli_trio_state(shard_index: int) -> dict[str, Any]:
    paths = fill_nli.output_paths(shard_index, smoke=False)
    state = {}
    for name in ("output", "report", "receipt"):
        path = paths[name]
        state[name] = {
            "path": corpus.relative(path),
            "regular_non_symlink_file_present": (
                not path.is_symlink() and path.is_file()
            ),
        }
    return {
        "gpu_shard": shard_index,
        "artifacts": state,
        "sealed_trio_present": all(
            value["regular_non_symlink_file_present"]
            for value in state.values()
        ),
    }


def _static_shard(shard_index: int) -> dict[str, Any]:
    packets, summary = fill_nli.load_structural_packets(shard_index)
    groups = fill_judge.groups_by_request(packets)
    pass_contract = summary["generation_pass_contract"]
    prompts = fill_judge.prompt_receipts(groups)
    planned_outputs = fill_judge.output_paths(shard_index, smoke=False)
    return {
        "gpu_shard": shard_index,
        "packets": len(packets),
        "request_groups": len(groups),
        "generation_pass": {
            "id": fill_judge.PASS_ID,
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
        "structural_review": fill_judge.structural_receipt(shard_index, summary),
        "rendered_prompt_receipts": prompts,
        "required_sealed_fill_nli": _required_nli_paths(shard_index),
        "planned_fill_judge_outputs": {
            key: corpus.relative(value) for key, value in planned_outputs.items()
        },
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }


def _ready_preflights(states: list[dict[str, Any]]) -> tuple[list[dict] | None, list[str]]:
    blockers = []
    for state in states:
        for name, artifact in state["artifacts"].items():
            if not artifact["regular_non_symlink_file_present"]:
                blockers.append(
                    f"missing_sealed_fill_nli_{name}:gpu{state['gpu_shard']}"
                )
    if blockers:
        return None, blockers
    if os.environ.get("LD_LIBRARY_PATH") != CU13_LIBRARY_PATH:
        return None, ["fill_judge_contract_builder_runtime_invalid"]
    preflights = []
    for shard_index in range(4):
        try:
            # Importing the official prompt stack pulls in cv2, whose package
            # initializer prepends its lib64 directory to LD_LIBRARY_PATH in
            # this long-lived process.  Every real command starts separately
            # from the exact cu13 environment; restore that already-validated
            # identity before validating the next shard in-process.
            os.environ["LD_LIBRARY_PATH"] = CU13_LIBRARY_PATH
            preflights.append(fill_judge.preflight(_args(shard_index)))
        except (OSError, RuntimeError, ValueError):
            blockers.append(f"fill_nli_or_judge_preflight_invalid:gpu{shard_index}")
            return None, blockers
    return preflights, []


def build_contract() -> dict:
    reused = fill_judge.validate_reused_judge_semantics()
    _, nli_launch_receipt = fill_judge.validate_nli_launch_contract()
    static_shards = [_static_shard(shard) for shard in range(4)]
    nli_states = [_nli_trio_state(shard) for shard in range(4)]
    preflights, blockers = _ready_preflights(nli_states)
    commands_released = preflights is not None
    if preflights is not None:
        for static, preflight in zip(static_shards, preflights, strict=True):
            run_contract = preflight["run_contract"]
            if (
                run_contract["gpu_shard"] != static["gpu_shard"]
                or run_contract["generation_pass"]["contract_sha256"]
                != static["generation_pass"]["contract_sha256"]
                or run_contract["judge_protocol"]["rendered_prompt_receipts"]
                != static["rendered_prompt_receipts"]
            ):
                raise RuntimeError("ready fill semantic preflight differs from scaffold")
            static["sealed_fill_nli"] = preflight["sealed_fill_nli"]
            static["run_contract_sha256"] = run_contract[
                "content_sha256_before_self_field"
            ]
            static["preflight_command"] = safe_preflight_command(
                static["gpu_shard"]
            )
            static["run_command"] = safe_run_command(static["gpu_shard"])
            static["post_run_output_check_command"] = safe_output_check_command(
                static["gpu_shard"]
            )
            static["post_run_receipt_required"] = True

    contract = {
        "schema": SCHEMA,
        "status": (
            "ready_for_explicit_four_gpu_fill_semantic_judge_launch"
            if commands_released
            else "blocked_pending_all_four_sealed_fill_nli_outputs"
        ),
        "commands_released": commands_released,
        "blockers": blockers,
        "generation_pass_id": fill_judge.PASS_ID,
        "fill_nli_launch_contract": nli_launch_receipt,
        "fill_judge_wrapper_receipt": {
            "path": corpus.relative(Path(fill_judge.__file__).resolve()),
            "file_sha256": corpus.file_sha256(
                Path(fill_judge.__file__).resolve()
            ),
        },
        "primary_judge_semantics": {
            "implementation_receipts": reused["implementation_receipts"],
            "guided_schema_sha256": reused["guided_schema_sha256"],
            "semantic_protocol_sha256": reused[
                "semantic_protocol_sha256"
            ],
            "protocol": reused["protocol"],
        },
        "pinned_mistral": {
            "id": fill_judge.MODEL_ID,
            "revision": fill_judge.MODEL_REVISION,
            "snapshot_blob_receipts": fill_judge.MODEL_BLOB_RECEIPTS,
            "runtime_file_sha256": fill_judge.RUNTIME_MODEL_FILE_SHA256,
        },
        "exact_runtime": {
            "python_executable": str(fill_judge.RUNTIME_PYTHON_EXECUTABLE),
            "virtual_environment": str(fill_judge.RUNTIME_VIRTUAL_ENV),
            "ld_library_path": str(fill_judge.CU13_LIBRARY_PATH),
            "transformers_version": fill_judge.TRANSFORMERS_VERSION,
            "torch_version": fill_judge.TORCH_VERSION,
            "vllm_version": fill_judge.VLLM_VERSION,
            "mistral_common_version": fill_judge.MISTRAL_COMMON_VERSION,
            "dtype": fill_judge.DTYPE,
        },
        "inference": {
            "passes": list(primary_judge.PASS_NAMES),
            "request_batch_size": REQUEST_BATCH_SIZE,
            "two_pass_sequences_per_full_batch": (
                REQUEST_BATCH_SIZE * len(primary_judge.PASS_NAMES)
            ),
            "max_model_len": MAX_MODEL_LEN,
            "max_tokens": MAX_TOKENS,
            "max_num_seqs": fill_judge.MAX_NUM_SEQS,
            "temperature": 0.0,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "guided_schema": primary_judge.GUIDED_SCHEMA,
            "guided_schema_sha256": fill_judge.GUIDED_SCHEMA_SHA256,
        },
        "nli_artifact_states": nli_states,
        "shards": static_shards,
        "eventual_exact_hash_binding": {
            "fill_nli_required_fields": [
                "output_sha256",
                "report_file_sha256",
                "report_self_sha256",
                "receipt_file_sha256",
                "receipt_self_sha256",
                "run_contract_sha256",
            ],
            "fill_judge_receipt_schema": fill_judge.RECEIPT_SCHEMA,
            "fill_judge_required_fields": [
                "output_sha256",
                "report_file_sha256",
                "report_self_sha256",
                "run_contract_sha256",
                "content_sha256_before_self_field",
            ],
            "output_or_report_without_matching_receipt_is_valid": False,
        },
        "contract_builder_receipt": {
            "path": corpus.relative(Path(__file__).resolve()),
            "file_sha256": corpus.file_sha256(Path(__file__).resolve()),
        },
        "policy": {
            "all_four_sealed_fill_nli_trios_required_before_any_command": True,
            "active_primary_nli_or_judge_outputs_modified": False,
            "gpu_job_launched_by_contract_builder": False,
            "protected_development_final_holdout_ood_terminal_incident_sources_read": False,
            "semantic_verification_completed": False,
            "training_rows_emitted": False,
        },
    }
    if not commands_released:
        forbidden = {
            "preflight_command",
            "run_command",
            "post_run_output_check_command",
        }
        if any(forbidden & set(shard) for shard in static_shards):
            raise RuntimeError("blocked fill semantic contract leaked a command")
    contract["content_sha256_before_self_field"] = corpus.canonical_sha256(
        contract
    )
    return contract


def contract_bytes(value: dict) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


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
            raise RuntimeError("fill semantic judge launch contract is missing")
        if args.output.read_bytes() != payload:
            raise RuntimeError("fill semantic judge launch contract changed")
    if args.print_contract:
        print(payload.decode("utf-8"), end="")
    if not args.check and not args.print_contract:
        print(json.dumps({
            "status": value["status"],
            "output": corpus.relative(args.output),
            "content_sha256": value["content_sha256_before_self_field"],
            "commands_released": value["commands_released"],
            "shards": len(value["shards"]),
            "training_rows_emitted": False,
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
