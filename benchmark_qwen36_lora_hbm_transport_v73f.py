#!/usr/bin/env python3
"""Exploratory four-GPU LoRA transport/materialization microbenchmark.

This systems-only probe reads the unprotected staged V434 LoRA adapter and
compares two H2D publication paths plus tensorwise versus flat GPU
perturbation/materialization.  It does not load a model or any dataset and is
not promotion evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file

import qwen36_v73e_exact_phase_profiler_contract as contract


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v73f_qwen36_lora_hbm_transport_exploratory.json"
)


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _measure(device: torch.device, iterations: int, operation) -> dict:
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    end.synchronize()
    wall_end = time.perf_counter_ns()
    cuda_ms = float(start.elapsed_time(end))
    _require(math.isfinite(cuda_ms) and cuda_ms > 0.0, "invalid CUDA timing")
    return {
        "iterations": iterations,
        "cuda_elapsed_ms": cuda_ms,
        "wall_elapsed_ms": (wall_end - wall_start) / 1_000_000.0,
    }


def _rank_probe(args) -> dict:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _require(world_size == 4, "V73F requires exactly four ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")

    observed_weights_sha = contract.file_sha256(contract.STAGED_ADAPTER_WEIGHTS)
    _require(
        observed_weights_sha == contract.STAGED_ADAPTER_WEIGHTS_SHA256,
        "staged adapter identity changed",
    )
    state = load_file(str(contract.STAGED_ADAPTER_WEIGHTS), device="cpu")
    keys = sorted(state)
    pageable = [state[key].float().contiguous() for key in keys]
    _require(
        len(pageable) == contract.STAGED_TENSOR_COUNT
        and sum(item.numel() for item in pageable)
        == contract.STAGED_ELEMENT_COUNT,
        "staged adapter geometry changed",
    )
    pinned = [item.pin_memory() for item in pageable]
    destinations = [
        torch.empty_like(item, device=device, dtype=torch.bfloat16)
        for item in pageable
    ]
    master = [item.to(device=device, dtype=torch.float32) for item in pageable]
    generator = torch.Generator(device=device)
    generator.manual_seed(73_000 + rank)
    noise = [
        torch.randn(
            item.shape,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        for item in master
    ]
    workspace = [torch.empty_like(item) for item in master]
    flat_master = torch.cat([item.reshape(-1) for item in master])
    flat_noise = torch.cat([item.reshape(-1) for item in noise])
    flat_workspace = torch.empty_like(flat_master)
    flat_destination = torch.empty(
        flat_master.numel(), dtype=torch.bfloat16, device=device
    )

    def pageable_h2d():
        for source, target in zip(pageable, destinations, strict=True):
            target.copy_(source, non_blocking=False)

    def pinned_h2d():
        for source, target in zip(pinned, destinations, strict=True):
            target.copy_(source, non_blocking=True)

    def tensorwise_materialize():
        for base, epsilon, temporary, target in zip(
            master, noise, workspace, destinations, strict=True
        ):
            torch.add(base, epsilon, alpha=args.sigma, out=temporary)
            target.copy_(temporary)

    def flat_materialize():
        torch.add(
            flat_master,
            flat_noise,
            alpha=args.sigma,
            out=flat_workspace,
        )
        flat_destination.copy_(flat_workspace)

    for _ in range(args.warmup):
        pageable_h2d()
        pinned_h2d()
        tensorwise_materialize()
        flat_materialize()
    torch.cuda.synchronize(device)
    dist.barrier()

    repeats = []
    for repeat in range(args.repeats):
        transport_order = (
            ("pageable_sync_h2d", pageable_h2d),
            ("pinned_nonblocking_h2d", pinned_h2d),
        )
        materialize_order = (
            ("tensorwise_gpu_materialize", tensorwise_materialize),
            ("flat_gpu_materialize", flat_materialize),
        )
        if (rank + repeat) % 2:
            transport_order = tuple(reversed(transport_order))
            materialize_order = tuple(reversed(materialize_order))
        row = {"repeat": repeat, "arm_order": []}
        for name, operation in transport_order:
            dist.barrier()
            row["arm_order"].append(name)
            row[name] = _measure(device, args.transport_iterations, operation)
        for name, operation in materialize_order:
            dist.barrier()
            row["arm_order"].append(name)
            row[name] = _measure(
                device, args.materialize_iterations, operation
            )
        repeats.append(row)

    tensorwise_materialize()
    tensorwise_flat = torch.cat([item.reshape(-1) for item in destinations])
    flat_materialize()
    exact_flat_match = bool(torch.equal(tensorwise_flat, flat_destination))
    checksum = float(flat_destination.float().sum().item())
    _require(exact_flat_match and math.isfinite(checksum), "arm output mismatch")
    result = {
        "rank": rank,
        "physical_gpu_id": local_rank,
        "gpu_name": torch.cuda.get_device_name(device),
        "staged_adapter_weights_sha256": observed_weights_sha,
        "tensor_count": len(pageable),
        "element_count": sum(item.numel() for item in pageable),
        "fp32_bytes_per_transition": sum(item.nbytes for item in pageable),
        "bf16_bytes_per_transition": sum(item.numel() * 2 for item in pageable),
        "flat_and_tensorwise_output_exact": exact_flat_match,
        "aggregate_output_checksum": checksum,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "repeats": repeats,
    }
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(result, gathered, dst=0)
    dist.barrier()
    dist.destroy_process_group()
    return {"rank_result": result, "gathered": gathered}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--transport-iterations", type=int, default=5_000)
    parser.add_argument("--materialize-iterations", type=int, default=20_000)
    parser.add_argument("--sigma", type=float, default=0.0006)
    args = parser.parse_args(argv)
    _require(
        args.warmup > 0
        and args.repeats >= 3
        and args.transport_iterations > 0
        and args.materialize_iterations > 0
        and math.isfinite(args.sigma)
        and args.sigma > 0.0,
        "invalid V73F benchmark arguments",
    )
    output = Path(args.output).resolve()
    _require(not output.exists(), f"V73F output path is not fresh: {output}")
    probe = _rank_probe(args)
    if int(os.environ["RANK"]) == 0:
        body = {
            "schema": "qwen36-v73f-lora-hbm-transport-exploratory-v1",
            "status": "complete_systems_only_nonpromotable",
            "authority": {
                "model_or_dataset_loaded": False,
                "protected_evaluation_opened": False,
                "quality_hpo_or_recipe_promotion_authorized": False,
                "checkpoint_or_model_update_performed": False,
            },
            "world_size": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "transport_iterations": args.transport_iterations,
            "materialize_iterations": args.materialize_iterations,
            "sigma": args.sigma,
            "ranks": probe["gathered"],
        }
        body["content_sha256_before_self_field"] = canonical_sha256(body)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(
                (json.dumps(body, indent=2, sort_keys=True) + "\n").encode(
                    "ascii"
                )
            )
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps({
            "output": str(output),
            "file_sha256": contract.file_sha256(output),
            "content_sha256": body["content_sha256_before_self_field"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
