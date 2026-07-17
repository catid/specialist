#!/usr/bin/env python3
"""Prospective data-free four-GPU FP32/BF16 collective preflight.

The script validates a completed V82 post-optimization bottleneck receipt
before importing torch.  It is intentionally not launchable from the static
preregistration alone.  The benchmark uses native tensor boundaries, three
registered synthetic seeds, and same-process AB/BA blocks; it opens no model
or dataset and cannot promote a training recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any

import eggroll_es_collective_compression_v82 as oracle


SCHEMA_V82 = "qwen36-collective-compression-data-free-preflight-v82"
PAIRS_PER_SEED_V82 = 6
ITERATIONS_PER_BLOCK_V82 = 8
WARMUP_ITERATIONS_V82 = 2
SAMPLE_ELEMENTS_PER_TENSOR_V82 = 16


def _load_json_no_duplicates_v82(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise RuntimeError("V82 gate receipt must be a JSON object")
    return value


def load_authorized_gate_v82(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("V82 requires a regular post-optimization gate receipt")
    value = _load_json_no_duplicates_v82(path)
    if not oracle.live_launch_authorized_v82(value):
        raise RuntimeError("V82 bottleneck receipt does not authorize a GPU probe")
    return value


def block_orders_v82(pair_count: int = PAIRS_PER_SEED_V82) -> tuple[tuple[str, str], ...]:
    if isinstance(pair_count, bool) or not isinstance(pair_count, int):
        raise TypeError("V82 pair count must be an integer")
    if pair_count < 2 or pair_count % 2:
        raise ValueError("V82 pair count must be positive, even, and at least two")
    return tuple(
        ("fp32_control", "bf16_error_feedback")
        if index % 2 == 0
        else ("bf16_error_feedback", "fp32_control")
        for index in range(pair_count)
    )


def summarize_blocks_v82(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    expected_count = (
        len(oracle.REGISTERED_SEEDS_V82) * PAIRS_PER_SEED_V82 * 2
    )
    if len(blocks) != expected_count:
        raise RuntimeError("V82 block count changed")
    by_seed = []
    cursor = 0
    for seed in oracle.REGISTERED_SEEDS_V82:
        ratios = []
        for pair_index, expected_order in enumerate(block_orders_v82()):
            pair = blocks[cursor:cursor + 2]
            cursor += 2
            if (
                [item.get("arm") for item in pair] != list(expected_order)
                or any(item.get("seed") != seed for item in pair)
                or any(item.get("pair_index") != pair_index for item in pair)
            ):
                raise RuntimeError("V82 block order changed")
            by_arm = {item["arm"]: item for item in pair}
            for item in pair:
                seconds = item.get("seconds")
                if (
                    not isinstance(seconds, float)
                    or not math.isfinite(seconds)
                    or seconds <= 0.0
                ):
                    raise RuntimeError("V82 block timing changed")
            ratios.append(
                by_arm["fp32_control"]["seconds"]
                / by_arm["bf16_error_feedback"]["seconds"]
            )
        by_seed.append({
            "seed": seed,
            "bf16_over_fp32_speed_median": statistics.median(ratios),
            "bf16_over_fp32_speed_geometric_mean": math.exp(
                math.fsum(math.log(value) for value in ratios) / len(ratios)
            ),
            "bf16_faster_pairs": sum(value > 1.0 for value in ratios),
            "fp32_faster_pairs": sum(value < 1.0 for value in ratios),
            "pair_count": len(ratios),
        })
    return {"by_seed": by_seed, "quality_promotion_authorized": False}


def _atomic_json_v82(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def execute_v82(gate: dict[str, Any], output: Path) -> int:
    # Delayed import is a safety property: an incomplete dependency gate fails
    # before torch can initialize CUDA or NCCL.
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    report_group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != oracle.WORLD_SIZE_V82:
        raise RuntimeError("V82 preflight requires exactly four ranks")

    device = torch.device(f"cuda:{local_rank}")
    maximum = oracle.MAX_TENSOR_ELEMENTS_V82
    accumulator = torch.empty(maximum, dtype=torch.float32, device=device)
    compressed = torch.empty(maximum, dtype=torch.bfloat16, device=device)
    host_fp32 = torch.empty(maximum, dtype=torch.float32, pin_memory=True)
    host_bf16 = torch.empty(maximum, dtype=torch.bfloat16, pin_memory=True)
    residual_current = [
        torch.zeros(elements, dtype=torch.float32, device=device)
        for elements in oracle.RUNTIME_PARAMETER_ELEMENTS_V82
    ]
    residual_pending = [torch.empty_like(value) for value in residual_current]
    blocks: list[dict[str, Any]] = []
    last_bf16_inputs: dict[int, tuple[float, int]] = {}

    def synthetic_scalar(seed: int, iteration: int, tensor_index: int) -> float:
        return float(
            (rank + 1) * 0.03125
            + ((seed % 997) + 1) * 1.0e-6
            + (iteration % 17) * 1.0e-5
            + (tensor_index % 11) * 1.0e-4
        )

    def iteration(arm: str, seed: int, ordinal: int) -> None:
        nonlocal residual_current, residual_pending
        for tensor_index, elements in enumerate(
            oracle.RUNTIME_PARAMETER_ELEMENTS_V82
        ):
            update = accumulator[:elements]
            scalar = synthetic_scalar(seed, ordinal, tensor_index)
            update.fill_(scalar)
            if arm == "fp32_control":
                dist.all_reduce(update)
                host_fp32[:elements].copy_(update, non_blocking=True)
            else:
                update.add_(residual_current[tensor_index])
                q = compressed[:elements]
                q.copy_(update)
                residual_pending[tensor_index].copy_(update)
                residual_pending[tensor_index].sub_(q)
                dist.all_reduce(q)
                host_bf16[:elements].copy_(q, non_blocking=True)
                last_bf16_inputs[tensor_index] = (scalar, ordinal)
        torch.cuda.synchronize()
        if arm == "bf16_error_feedback":
            residual_current, residual_pending = (
                residual_pending,
                residual_current,
            )

    try:
        for seed in oracle.REGISTERED_SEEDS_V82:
            for values in residual_current + residual_pending:
                values.zero_()
            torch.cuda.synchronize()
            for arm in ("fp32_control", "bf16_error_feedback"):
                for warmup in range(WARMUP_ITERATIONS_V82):
                    iteration(arm, seed, -WARMUP_ITERATIONS_V82 + warmup)
            for pair_index, order in enumerate(block_orders_v82()):
                for order_index, arm in enumerate(order):
                    dist.barrier()
                    torch.cuda.synchronize()
                    started = time.monotonic()
                    for block_iteration in range(ITERATIONS_PER_BLOCK_V82):
                        ordinal = (
                            pair_index * ITERATIONS_PER_BLOCK_V82
                            + block_iteration
                        )
                        iteration(arm, seed, ordinal)
                    elapsed = time.monotonic() - started
                    dist.barrier()
                    blocks.append({
                        "seed": seed,
                        "pair_index": pair_index,
                        "order_index": order_index,
                        "arm": arm,
                        "iterations": ITERATIONS_PER_BLOCK_V82,
                        "seconds": float(elapsed),
                        "updates_per_second": (
                            ITERATIONS_PER_BLOCK_V82 / float(elapsed)
                        ),
                    })

        # Data-free numerical audit of the last local compression transaction.
        residual_samples = []
        prior_samples = []
        for tensor_index, elements in enumerate(
            oracle.RUNTIME_PARAMETER_ELEMENTS_V82
        ):
            count = min(SAMPLE_ELEMENTS_PER_TENSOR_V82, elements)
            residual_samples.append(
                residual_current[tensor_index][:count].cpu().tolist()
            )
            prior_samples.append(
                residual_pending[tensor_index][:count].cpu().tolist()
            )
        maximum_error = 0.0
        maximum_bound = 0.0
        for tensor_index, (scalar, _ordinal) in last_bf16_inputs.items():
            for prior, observed in zip(
                prior_samples[tensor_index],
                residual_samples[tensor_index],
                strict=True,
            ):
                expected = oracle.prepare_bf16_element_v82(scalar, prior)
                if oracle.float32_bits_v82(observed) != oracle.float32_bits_v82(
                    expected["next_residual"]
                ):
                    raise RuntimeError("V82 GPU BF16 residual sample changed")
                maximum_error = max(maximum_error, expected["absolute_error"])
                maximum_bound = max(
                    maximum_bound, expected["absolute_error_bound"]
                )

        local = {
            "rank": rank,
            "gpu": local_rank,
            "gpu_name": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "nccl_version": list(torch.cuda.nccl.version()),
            "blocks": blocks,
            "summary": summarize_blocks_v82(blocks),
            "residual_audit": {
                "sample_elements_per_tensor": SAMPLE_ELEMENTS_PER_TENSOR_V82,
                "maximum_absolute_error": maximum_error,
                "maximum_registered_bound": maximum_bound,
                "all_finite": True,
                "local_conservation_bit_exact": True,
                "gpu_matches_cpu_oracle_bit_exact": True,
            },
            "cuda_max_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "cuda_max_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        }
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local, gathered, dst=0, group=report_group)
        if rank == 0:
            accounting = oracle.collective_byte_accounting_v82()
            body = {
                "schema": SCHEMA_V82,
                "gate_content_sha256": gate["content_sha256"],
                "backend": "nccl",
                "world_size": world_size,
                "physical_gpus": [0, 1, 2, 3],
                "registered_seeds": list(oracle.REGISTERED_SEEDS_V82),
                "pairs_per_seed": PAIRS_PER_SEED_V82,
                "iterations_per_block": ITERATIONS_PER_BLOCK_V82,
                "warmup_iterations_per_arm_per_seed": WARMUP_ITERATIONS_V82,
                "native_tensor_elements": list(
                    oracle.RUNTIME_PARAMETER_ELEMENTS_V82
                ),
                "byte_accounting": accounting,
                "link_bytes_measured": False,
                "link_byte_note": (
                    "The byte accounting is algorithmic payload/ring projection; "
                    "promotion still requires the gate's profiler and exact "
                    "post-run link/HBM receipt."
                ),
                "results": gathered,
                "data_model_or_protected_content_opened": False,
                "training_checkpoint_or_quality_promotion_performed": False,
            }
            report = {
                **body,
                "content_sha256": oracle.canonical_sha256_v82(body),
            }
            _atomic_json_v82(output, report)
    finally:
        dist.destroy_process_group(report_group)
        dist.destroy_process_group()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gate = load_authorized_gate_v82(args.gate.resolve())
    return execute_v82(gate, args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
