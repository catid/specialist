#!/usr/bin/env python3
"""Data-free repeated-generation determinism probe for the V62 evaluator.

The probe uses synthetic prompts only.  It persists hashes and lengths of token
IDs, never prompt text, generated text, or token IDs themselves.  It performs
one unscored full-shape warmup followed by four identical greedy calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


MODEL = Path("models/Qwen3.6-35B-A3B").resolve()
ADAPTER = Path(
    "experiments/eggroll_es_hpo/staged_adapters/"
    "v434_equal_sft_qwen35_vllm_namespace_v49d"
).resolve()
TUNED = Path(
    "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_bf16_tp1_exhaustive_v27c"
).resolve()
ROWS = 68
RECORDED_CALLS = 4


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", required=True)
    value.add_argument("--actor-label", required=True)
    value.add_argument("--graph", action="store_true")
    value.add_argument("--max-num-seqs", type=int, default=68)
    value.add_argument("--torch-deterministic", action="store_true")
    value.add_argument("--warmup-calls", type=int, default=1)
    return value


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def token_hashes(outputs: list) -> list[dict]:
    result = []
    for output in outputs:
        token_ids = [int(value) for value in output.outputs[0].token_ids]
        result.append({
            "token_count": len(token_ids),
            "token_ids_sha256": hashlib.sha256(
                json.dumps(token_ids, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
        })
    if len(result) != ROWS:
        raise RuntimeError("synthetic probe row count changed")
    return result


def main() -> int:
    args = parser().parse_args()
    if not 1 <= args.max_num_seqs <= ROWS:
        raise RuntimeError("--max-num-seqs must be between 1 and 68")
    if not 1 <= args.warmup_calls <= 8:
        raise RuntimeError("--warmup-calls must be between 1 and 8")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise RuntimeError("probe requires installed Qwen GDN-compatible BI=false")
    os.environ["VLLM_BATCH_INVARIANT"] = "0"
    os.environ["VLLM_TUNED_CONFIG_FOLDER"] = str(TUNED)

    if args.torch_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        torch.use_deterministic_algorithms(True, warn_only=False)

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    prompts = [
        (
            f"Synthetic deterministic evaluator probe item {index}. "
            + "alpha beta gamma delta epsilon zeta eta theta "
            * (8 + (index * 17) % 53)
            + "Give one concise neutral continuation."
        )
        for index in range(ROWS)
    ]
    engine = LLM(
        model=str(MODEL),
        tokenizer=str(MODEL),
        dtype="bfloat16",
        tensor_parallel_size=1,
        distributed_executor_backend="uni",
        moe_backend="triton",
        gpu_memory_utilization=0.82,
        max_model_len=2048,
        max_num_seqs=args.max_num_seqs,
        scheduling_policy="fcfs",
        async_scheduling=False,
        enforce_eager=not args.graph,
        enable_prefix_caching=False,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        max_cpu_loras=1,
        limit_mm_per_prompt={"image": 0, "video": 0},
        mm_processor_cache_gb=0,
        skip_mm_profiling=True,
        generation_config="vllm",
        seed=202607241,
    )
    request = LoRARequest("v434-synthetic-probe", 1, str(ADAPTER))
    params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        detokenize=False,
        seed=2026071601,
    )

    started = time.monotonic()
    warmup_receipt_hashes = []
    for _ in range(args.warmup_calls):
        warmup = engine.generate(
            prompts, params, use_tqdm=False, lora_request=request
        )
        warmup_receipt_hashes.append(canonical_sha256(token_hashes(warmup)))
    calls = []
    for index in range(RECORDED_CALLS):
        outputs = engine.generate(
            prompts, params, use_tqdm=False, lora_request=request
        )
        receipt = token_hashes(outputs)
        calls.append({
            "call_index": index,
            "rows": receipt,
            "rows_sha256": canonical_sha256(receipt),
        })

    per_row_distinct = [
        len({call["rows"][row]["token_ids_sha256"] for call in calls})
        for row in range(ROWS)
    ]
    value = {
        "schema": "v62b-synthetic-fullbatch-repeat-determinism-probe",
        "actor_label": args.actor_label,
        "synthetic_rows": ROWS,
        "unscored_fullbatch_warmups": args.warmup_calls,
        "recorded_calls": RECORDED_CALLS,
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
            "seed": 2026071601,
        },
        "runtime": {
            "VLLM_BATCH_INVARIANT": False,
            "async_scheduling": False,
            "max_num_seqs": args.max_num_seqs,
            "scheduling_policy": "fcfs",
            "enforce_eager": not args.graph,
            "cuda_graphs_enabled": bool(args.graph),
            "torch_deterministic_algorithms": bool(args.torch_deterministic),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
            "CUDA_DEVICE_MAX_CONNECTIONS": os.environ.get(
                "CUDA_DEVICE_MAX_CONNECTIONS"
            ),
            "CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"),
            "tensor_parallel_size": 1,
            "vllm_version": "0.25.0",
        },
        "warmup_call_rows_sha256": warmup_receipt_hashes,
        "warmup_rows_sha256": warmup_receipt_hashes[-1],
        "calls": calls,
        "within_actor_all_four_calls_identical": all(
            value == 1 for value in per_row_distinct
        ),
        "within_actor_changed_row_count": sum(
            value != 1 for value in per_row_distinct
        ),
        "within_actor_max_distinct_generations_per_row": max(per_row_distinct),
        "wall_runtime_seconds_including_warmup_excluding_model_load": (
            time.monotonic() - started
        ),
        "source_dataset_rows_opened": 0,
        "prompt_or_generation_text_persisted": False,
        "token_ids_persisted": False,
        "adapter_update_or_hpo_performed": False,
        "protected_ood_shadow_or_terminal_opened": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    output = Path(args.output).resolve()
    if output.exists():
        raise RuntimeError("synthetic probe requires a fresh output path")
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "within_actor_changed_row_count": value[
            "within_actor_changed_row_count"
        ],
        "within_actor_all_four_calls_identical": value[
            "within_actor_all_four_calls_identical"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
