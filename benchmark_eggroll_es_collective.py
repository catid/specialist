#!/usr/bin/env python3
"""Benchmark the four-GPU EGGROLL-ES coefficient-update collective."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
import torch.distributed as dist


MIDDLE_LATE_RUNTIME_PARAMETER_ELEMENTS = (
    (131_072, 25_165_824, 8_388_608, 524_288, 1_048_576, 2_097_152) * 3
    + (18_874_368, 8_388_608, 524_288, 1_048_576, 2_097_152)
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", type=int, default=142_999_552)
    parser.add_argument("--bucket-elements", type=int, default=0)
    parser.add_argument(
        "--pattern", choices=("flat", "middle_late_runtime_v1"),
        default="flat",
    )
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}"),
    )
    report_group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    element_sizes = (
        MIDDLE_LATE_RUNTIME_PARAMETER_ELEMENTS
        if args.pattern == "middle_late_runtime_v1"
        else (args.elements,)
    )
    if sum(element_sizes) != args.elements:
        raise ValueError("pattern element total does not match --elements")
    tensors = [
        torch.full(
            (elements,), rank + 1, dtype=torch.float32,
            device=f"cuda:{local_rank}",
        )
        for elements in element_sizes
    ]
    bucket_elements = args.bucket_elements or max(element_sizes)
    if bucket_elements <= 0 or bucket_elements > args.elements:
        raise ValueError("bucket elements must be in [1, elements]")

    def all_reduce_tensor() -> None:
        for tensor in tensors:
            for start in range(0, tensor.numel(), bucket_elements):
                dist.all_reduce(tensor[start:start + bucket_elements])

    def fill_tensors() -> None:
        for tensor in tensors:
            tensor.fill_(rank + 1)

    for _ in range(args.warmup):
        fill_tensors()
        all_reduce_tensor()
    torch.cuda.synchronize()
    dist.barrier()

    started = time.monotonic()
    iterations = 0
    while time.monotonic() - started < args.seconds:
        fill_tensors()
        all_reduce_tensor()
        torch.cuda.synchronize()
        iterations += 1
    elapsed = time.monotonic() - started

    bytes_per_collective = args.elements * tensors[0].element_size()
    algorithm_gib_per_second = (
        bytes_per_collective * iterations / elapsed / (2**30)
    )
    bus_gib_per_second = (
        algorithm_gib_per_second * 2 * (world_size - 1) / world_size
    )
    local = {
        "rank": rank,
        "gpu": local_rank,
        "iterations": iterations,
        "seconds": elapsed,
        "algorithm_gib_per_second": algorithm_gib_per_second,
        "bus_gib_per_second": bus_gib_per_second,
        "checksum": float(sum(
            tensor[:min(1024, tensor.numel())].sum() for tensor in tensors
        ).cpu()),
    }
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0, group=report_group)
    if rank == 0:
        print(json.dumps({
            "schema": "eggroll-es-four-gpu-collective-benchmark-v1",
            "backend": "nccl",
            "world_size": world_size,
            "elements": args.elements,
            "pattern": args.pattern,
            "runtime_parameter_count": len(element_sizes),
            "runtime_parameter_elements_sha256": hashlib.sha256(
                json.dumps(element_sizes, separators=(",", ":")).encode()
            ).hexdigest(),
            "bucket_elements": bucket_elements,
            "collective_count_per_iteration": sum(
                (elements + bucket_elements - 1) // bucket_elements
                for elements in element_sizes
            ),
            "bytes_per_rank": bytes_per_collective,
            "duration_target_seconds": args.seconds,
            "results": gathered,
        }, sort_keys=True))
    dist.destroy_process_group(report_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
