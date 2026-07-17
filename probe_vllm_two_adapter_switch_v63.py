#!/usr/bin/env python3
"""Data-free V434/V59 LoRA switching feasibility probe for V63.

Only token-count and token-ID SHA-256 receipts are persisted. No source dataset,
prompt text, generated text, token IDs, adapter update, or protected data is
opened or written.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import probe_vllm_synthetic_repeat_determinism_v62b as base


REFERENCE = base.ADAPTER
CANDIDATE = Path(
    "experiments/eggroll_es_hpo/runs/v59_lora_es_fragile_priority/"
    "selected_candidate_v59"
).resolve()
CALL_PLAN = ("reference", "candidate", "candidate", "reference") * 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", required=True)
    value.add_argument("--actor-label", required=True)
    value.add_argument("--reference-adapter", default=str(REFERENCE))
    value.add_argument("--candidate-adapter", default=str(CANDIDATE))
    value.add_argument("--graph", action="store_true")
    return value


def adapter_identity(path: Path) -> dict:
    weights = path / "adapter_model.safetensors"
    config = path / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise RuntimeError("two-adapter probe path is incomplete")
    return {
        "weights_sha256": base.file_sha256(weights),
        "config_sha256": base.file_sha256(config),
    }


def changed_rows(calls: list[dict]) -> int:
    return sum(
        len({call["rows"][row]["token_ids_sha256"] for call in calls}) > 1
        for row in range(base.ROWS)
    )


def shutdown_engine(engine: object) -> bool:
    """Release the in-process vLLM engine and its default torch PG."""
    engine.llm_engine.engine_core.shutdown()

    import torch.distributed as dist

    process_group_destroyed = dist.is_available() and dist.is_initialized()
    if process_group_destroyed:
        dist.destroy_process_group()
    return process_group_destroyed


def main() -> int:
    args = parser().parse_args()
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise RuntimeError("two-adapter probe requires Qwen GDN-compatible BI=false")
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(base.TUNED)
    reference = Path(args.reference_adapter).resolve()
    candidate = Path(args.candidate_adapter).resolve()
    identities = {
        "reference": adapter_identity(reference),
        "candidate": adapter_identity(candidate),
    }
    if identities["reference"] == identities["candidate"]:
        raise RuntimeError("two-adapter probe states must differ")
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("two-adapter probe requires a fresh output path")

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    prompts = [
        (
            f"Synthetic two-adapter switch probe item {index}. "
            + "alpha beta gamma delta epsilon zeta eta theta "
            * (8 + (index * 17) % 53)
            + "Give one concise neutral continuation."
        )
        for index in range(base.ROWS)
    ]
    engine = LLM(
        model=str(base.MODEL),
        tokenizer=str(base.MODEL),
        dtype="bfloat16",
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
        moe_backend="triton",
        gpu_memory_utilization=0.82,
        max_model_len=2048,
        max_num_seqs=68,
        scheduling_policy="fcfs",
        async_scheduling=False,
        enforce_eager=not args.graph,
        enable_prefix_caching=False,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        max_cpu_loras=2,
        limit_mm_per_prompt={"image": 0, "video": 0},
        mm_processor_cache_gb=0,
        skip_mm_profiling=True,
        generation_config="vllm",
        seed=202607241,
    )
    requests = {
        "reference": LoRARequest("reference-v434", 1, str(reference)),
        "candidate": LoRARequest("candidate-v59", 2, str(candidate)),
    }
    params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        detokenize=False,
        seed=2026071601,
    )

    started = time.monotonic()
    warmups = {}
    for label in ("reference", "candidate"):
        outputs = engine.generate(
            prompts, params, use_tqdm=False, lora_request=requests[label]
        )
        warmups[label] = base.canonical_sha256(base.token_hashes(outputs))

    calls = []
    for call_index, label in enumerate(CALL_PLAN):
        outputs = engine.generate(
            prompts, params, use_tqdm=False, lora_request=requests[label]
        )
        rows = base.token_hashes(outputs)
        calls.append({
            "call_index": call_index,
            "label": label,
            "rows": rows,
            "rows_sha256": base.canonical_sha256(rows),
        })
    reference_calls = [call for call in calls if call["label"] == "reference"]
    candidate_calls = [call for call in calls if call["label"] == "candidate"]
    between_state_differing_rows = sum(
        reference_calls[0]["rows"][row]["token_ids_sha256"]
        != candidate_calls[0]["rows"][row]["token_ids_sha256"]
        for row in range(base.ROWS)
    )
    wall_runtime_seconds = time.monotonic() - started
    process_group_destroyed = shutdown_engine(engine)
    del engine
    gc.collect()
    value = {
        "schema": "v63-synthetic-two-adapter-switch-feasibility-probe",
        "actor_label": args.actor_label,
        "adapter_identities": identities,
        "call_plan": list(CALL_PLAN),
        "runtime": {
            "VLLM_BATCH_INVARIANT": False,
            "async_scheduling": False,
            "scheduling_policy": "fcfs",
            "max_num_seqs": 68,
            "enforce_eager": not args.graph,
            "cuda_graphs_enabled": bool(args.graph),
            "max_loras": 1,
            "max_cpu_loras": 2,
            "vllm_version": "0.25.0",
        },
        "warmup_rows_sha256": warmups,
        "calls": calls,
        "reference_within_state_changed_rows": changed_rows(reference_calls),
        "candidate_within_state_changed_rows": changed_rows(candidate_calls),
        "between_state_differing_rows": between_state_differing_rows,
        "wall_runtime_seconds_excluding_model_load_and_cleanup": wall_runtime_seconds,
        "engine_shutdown_completed": True,
        "torch_process_group_destroyed": process_group_destroyed,
        "source_dataset_rows_opened": 0,
        "prompt_or_generation_text_persisted": False,
        "token_ids_persisted": False,
        "adapter_update_or_hpo_performed": False,
        "protected_ood_shadow_or_terminal_opened": False,
    }
    value["content_sha256_before_self_field"] = base.canonical_sha256(value)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "reference_changed": value["reference_within_state_changed_rows"],
        "candidate_changed": value["candidate_within_state_changed_rows"],
        "between_state_differing": value["between_state_differing_rows"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
