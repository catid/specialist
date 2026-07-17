#!/usr/bin/env python3
"""Fail-closed summary of the V73F four-GPU systems microbenchmarks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUTS = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v73f_qwen36_lora_hbm_transport_exploratory.json",
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v73f_qwen36_lora_hbm_transport_exploratory_rep2.json",
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v73f_qwen36_lora_hbm_transport_exploratory_rep3.json",
)
DEFAULT_OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_v73f_lora_hbm_transport_summary_20260717.json"
)
EXPECTED_SCHEMA = "qwen36-v73f-lora-hbm-transport-exploratory-v1"
EXPECTED_STATUS = "complete_systems_only_nonpromotable"
EXPECTED_AUTHORITY = {
    "model_or_dataset_loaded": False,
    "protected_evaluation_opened": False,
    "quality_hpo_or_recipe_promotion_authorized": False,
    "checkpoint_or_model_update_performed": False,
}
OPERATIONS = (
    "pageable_sync_h2d",
    "pinned_nonblocking_h2d",
    "tensorwise_gpu_materialize",
    "flat_gpu_materialize",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def geometric_mean(values: Iterable[float]) -> float:
    values = tuple(values)
    _require(bool(values) and all(value > 0.0 for value in values), "invalid geometric mean")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def distribution(values: Iterable[float]) -> dict[str, float | int]:
    values = tuple(values)
    _require(bool(values) and all(math.isfinite(value) for value in values), "invalid sample")
    return {
        "count": len(values),
        "geometric_mean": geometric_mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _load_and_validate(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(path.is_file() and not path.is_symlink(), f"input missing or symlinked: {path}")
    value = json.loads(path.read_text(encoding="ascii"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(claimed == canonical_sha256(body), f"self hash changed: {path}")
    _require(value.get("schema") == EXPECTED_SCHEMA, f"schema changed: {path}")
    _require(value.get("status") == EXPECTED_STATUS, f"status changed: {path}")
    _require(value.get("authority") == EXPECTED_AUTHORITY, f"authority changed: {path}")
    _require(value.get("world_size") == 4, f"world size changed: {path}")
    _require(value.get("physical_gpu_ids") == [0, 1, 2, 3], f"GPU inventory changed: {path}")
    ranks = value.get("ranks")
    _require(isinstance(ranks, list) and len(ranks) == 4, f"rank inventory changed: {path}")
    _require(
        sorted(rank.get("rank") for rank in ranks) == [0, 1, 2, 3]
        and sorted(rank.get("physical_gpu_id") for rank in ranks) == [0, 1, 2, 3],
        f"rank mapping changed: {path}",
    )
    _require(
        all(rank.get("flat_and_tensorwise_output_exact") is True for rank in ranks),
        f"materialization outputs differ: {path}",
    )
    identity = {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": file_sha256(path),
        "content_sha256": claimed,
        "transport_iterations": value.get("transport_iterations"),
        "materialize_iterations": value.get("materialize_iterations"),
    }
    return value, identity


def _summarize_run(value: dict[str, Any], identity: dict[str, Any]) -> dict[str, Any]:
    repeats = value.get("repeats")
    _require(isinstance(repeats, int) and repeats >= 3, "invalid repeat count")
    operation_ms_per_iteration = {name: [] for name in OPERATIONS}
    h2d_speedups = []
    materialize_speedups = []
    for rank in value["ranks"]:
        rows = rank.get("repeats")
        _require(isinstance(rows, list) and len(rows) == repeats, "rank repeats changed")
        for repeat_index, row in enumerate(rows):
            _require(row.get("repeat") == repeat_index, "repeat ordering changed")
            _require(sorted(row.get("arm_order", [])) == sorted(OPERATIONS), "arm order changed")
            per_iteration = {}
            for operation in OPERATIONS:
                measurement = row.get(operation)
                _require(isinstance(measurement, dict), f"missing operation: {operation}")
                iterations = measurement.get("iterations")
                elapsed = measurement.get("cuda_elapsed_ms")
                _require(
                    isinstance(iterations, int)
                    and iterations > 0
                    and isinstance(elapsed, (int, float))
                    and math.isfinite(elapsed)
                    and elapsed > 0.0,
                    f"invalid operation timing: {operation}",
                )
                per_iteration[operation] = float(elapsed) / iterations
                operation_ms_per_iteration[operation].append(per_iteration[operation])
            h2d_speedups.append(
                per_iteration["pageable_sync_h2d"]
                / per_iteration["pinned_nonblocking_h2d"]
            )
            materialize_speedups.append(
                per_iteration["tensorwise_gpu_materialize"]
                / per_iteration["flat_gpu_materialize"]
            )
    return {
        **identity,
        "rank_repeat_samples": 4 * repeats,
        "operation_cuda_ms_per_iteration": {
            name: distribution(values)
            for name, values in operation_ms_per_iteration.items()
        },
        "paired_speedup": {
            "pinned_nonblocking_over_pageable_sync_h2d": distribution(h2d_speedups),
            "flat_over_tensorwise_gpu_materialize": distribution(materialize_speedups),
        },
        "peak_allocated_bytes_max": max(rank["peak_allocated_bytes"] for rank in value["ranks"]),
        "peak_reserved_bytes_max": max(rank["peak_reserved_bytes"] for rank in value["ranks"]),
    }


def analyze(paths: list[Path]) -> dict[str, Any]:
    _require(len(paths) >= 3 and len(set(paths)) == len(paths), "three distinct inputs required")
    runs = []
    all_h2d = []
    all_materialize = []
    staged_hashes = set()
    for path in paths:
        value, identity = _load_and_validate(path)
        staged_hashes.update(rank.get("staged_adapter_weights_sha256") for rank in value["ranks"])
        summary = _summarize_run(value, identity)
        runs.append(summary)
        all_h2d.extend(
            row["pageable_sync_h2d"]["cuda_elapsed_ms"]
            / row["pageable_sync_h2d"]["iterations"]
            / (
                row["pinned_nonblocking_h2d"]["cuda_elapsed_ms"]
                / row["pinned_nonblocking_h2d"]["iterations"]
            )
            for rank in value["ranks"]
            for row in rank["repeats"]
        )
        all_materialize.extend(
            row["tensorwise_gpu_materialize"]["cuda_elapsed_ms"]
            / row["tensorwise_gpu_materialize"]["iterations"]
            / (
                row["flat_gpu_materialize"]["cuda_elapsed_ms"]
                / row["flat_gpu_materialize"]["iterations"]
            )
            for rank in value["ranks"]
            for row in rank["repeats"]
        )
    _require(len(staged_hashes) == 1 and None not in staged_hashes, "adapter identity changed")
    result = {
        "schema": "qwen36-v73f-lora-hbm-transport-summary-v1",
        "status": "complete_systems_only_nonpromotable",
        "authority": EXPECTED_AUTHORITY,
        "interpretation": {
            "end_to_end_training_speedup_measured": False,
            "quality_or_recipe_promotion_authorized": False,
            "supports_exact_phase_profile_followup": True,
        },
        "staged_adapter_weights_sha256": next(iter(staged_hashes)),
        "runs": runs,
        "aggregate_paired_speedup": {
            "pinned_nonblocking_over_pageable_sync_h2d": distribution(all_h2d),
            "flat_over_tensorwise_gpu_materialize": distribution(all_materialize),
        },
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", dest="inputs")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    inputs = [Path(item).resolve() for item in args.inputs] if args.inputs else list(DEFAULT_INPUTS)
    output = Path(args.output).resolve()
    _require(not output.exists(), f"output path is not fresh: {output}")
    result = analyze(inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write((json.dumps(result, indent=2, sort_keys=True) + "\n").encode("ascii"))
    print(json.dumps({"output": str(output), "file_sha256": file_sha256(output), "content_sha256": result["content_sha256_before_self_field"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
