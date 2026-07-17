#!/usr/bin/env python3
"""In-process paired four-GPU collective-layout benchmark.

Both the real 23-tensor layout and a single flat tensor stay allocated in the
same four-rank process.  Fixed-iteration blocks alternate AB/BA order, avoiding
the process-to-process throughput regime that invalidated earlier comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import time

import torch
import torch.distributed as dist

from benchmark_eggroll_es_collective import (
    MIDDLE_LATE_RUNTIME_PARAMETER_ELEMENTS,
)


ELEMENTS_V68 = sum(MIDDLE_LATE_RUNTIME_PARAMETER_ELEMENTS)
BYTES_V68 = ELEMENTS_V68 * 4


def block_orders_v68(pair_count: int) -> tuple[tuple[str, str], ...]:
    if isinstance(pair_count, bool) or not isinstance(pair_count, int):
        raise TypeError("pair count must be an integer")
    if pair_count < 2 or pair_count % 2:
        raise ValueError("pair count must be positive, even, and at least two")
    return tuple(
        ("native", "flat") if index % 2 == 0 else ("flat", "native")
        for index in range(pair_count)
    )


def geometric_mean_v68(values) -> float:
    values = tuple(float(value) for value in values)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def paired_summary_v68(blocks: list[dict], pair_count: int) -> dict:
    orders = block_orders_v68(pair_count)
    if len(blocks) != pair_count * 2:
        raise RuntimeError("paired block count changed")
    pairs = []
    for pair_index, expected_order in enumerate(orders):
        selected = blocks[pair_index * 2:(pair_index + 1) * 2]
        if tuple(item.get("layout") for item in selected) != expected_order:
            raise RuntimeError("paired layout order changed")
        by_layout = {item["layout"]: item for item in selected}
        for item in selected:
            if (
                item.get("pair_index") != pair_index
                or not isinstance(item.get("seconds"), float)
                or not math.isfinite(item["seconds"])
                or item["seconds"] <= 0
                or not isinstance(item.get("iterations"), int)
                or item["iterations"] <= 0
            ):
                raise RuntimeError("paired block timing changed")
        native = by_layout["native"]["seconds"]
        flat = by_layout["flat"]["seconds"]
        pairs.append({
            "pair_index": pair_index,
            "order": list(expected_order),
            "native_seconds": native,
            "flat_seconds": flat,
            "native_over_flat_speed": flat / native,
        })
    ratios = [item["native_over_flat_speed"] for item in pairs]
    return {
        "pairs": pairs,
        "native_over_flat_geometric_speed": geometric_mean_v68(ratios),
        "native_over_flat_median_speed": statistics.median(ratios),
        "native_faster_pair_count": sum(value > 1 for value in ratios),
        "flat_faster_pair_count": sum(value < 1 for value in ratios),
        "exact_tie_pair_count": sum(value == 1 for value in ratios),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--iterations-per-block", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    orders = block_orders_v68(args.pairs)
    if args.iterations_per_block <= 0 or args.warmup <= 0:
        raise ValueError("iterations and warmup must be positive")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}"),
    )
    report_group = dist.new_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 4:
        raise RuntimeError("paired collective benchmark requires four ranks")

    tensors = {
        "native": [
            torch.empty(
                elements,
                dtype=torch.float32,
                device=f"cuda:{local_rank}",
            )
            for elements in MIDDLE_LATE_RUNTIME_PARAMETER_ELEMENTS
        ],
        "flat": [
            torch.empty(
                ELEMENTS_V68,
                dtype=torch.float32,
                device=f"cuda:{local_rank}",
            )
        ],
    }

    def iteration(layout: str) -> None:
        for tensor in tensors[layout]:
            tensor.fill_(rank + 1)
            dist.all_reduce(tensor)

    for layout in ("native", "flat"):
        for _ in range(args.warmup):
            iteration(layout)
        torch.cuda.synchronize()
        dist.barrier()

    blocks = []
    block_index = 0
    for pair_index, order in enumerate(orders):
        for order_index, layout in enumerate(order):
            dist.barrier()
            torch.cuda.synchronize()
            started = time.monotonic()
            for _ in range(args.iterations_per_block):
                iteration(layout)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - started
            dist.barrier()
            blocks.append({
                "block_index": block_index,
                "pair_index": pair_index,
                "order_index": order_index,
                "layout": layout,
                "iterations": args.iterations_per_block,
                "seconds": elapsed,
                "algorithm_gib_per_second": (
                    BYTES_V68 * args.iterations_per_block / elapsed / (2**30)
                ),
            })
            block_index += 1

    expected_checksum = 10.0 * 1024
    checksums = {
        layout: float(sum(
            tensor[:min(1024, tensor.numel())].sum() for tensor in values
        ).cpu())
        for layout, values in tensors.items()
    }
    if (
        checksums["flat"] != expected_checksum
        or checksums["native"]
        != expected_checksum * len(MIDDLE_LATE_RUNTIME_PARAMETER_ELEMENTS)
    ):
        raise RuntimeError("paired collective checksum changed")
    local = {
        "rank": rank,
        "gpu": local_rank,
        "blocks": blocks,
        "summary": paired_summary_v68(blocks, args.pairs),
        "checksums": checksums,
    }
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local, gathered, dst=0, group=report_group)
    if rank == 0:
        print(json.dumps({
            "schema": "eggroll-es-in-process-paired-collective-layout-v68",
            "backend": "nccl",
            "world_size": world_size,
            "elements_per_rank": ELEMENTS_V68,
            "bytes_per_layout_per_rank": BYTES_V68,
            "both_layouts_resident_bytes_per_rank": BYTES_V68 * 2,
            "pair_count": args.pairs,
            "iterations_per_block": args.iterations_per_block,
            "warmup_iterations_per_layout": args.warmup,
            "orders": [list(value) for value in orders],
            "runtime_parameter_count": len(
                MIDDLE_LATE_RUNTIME_PARAMETER_ELEMENTS
            ),
            "runtime_parameter_elements_sha256": hashlib.sha256(
                json.dumps(
                    MIDDLE_LATE_RUNTIME_PARAMETER_ELEMENTS,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "results": gathered,
            "data_or_model_opened": False,
        }, sort_keys=True))
    dist.destroy_process_group(report_group)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
