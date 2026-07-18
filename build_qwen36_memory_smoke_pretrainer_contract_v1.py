#!/usr/bin/env python3
"""Seal the four synthetic Qwen3.6 expert-LoRA memory-smoke receipts.

This is deliberately a *pre-trainer* contract.  It proves that one complete
BF16 forward/backward/AdamW update fits for each planned adapter shape.  It
does not claim that the final packed mixed-format trainer, its dataloader, or
multi-step checkpoint/resume path has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parent
INPUT_DIRECTORY = (
    ROOT / "training_protocol/qwen36_memory_smoke_pretrainer_v1"
).resolve()
OUTPUT = (
    ROOT / "training_protocol/qwen36_memory_smoke_pretrainer_contract_v1.json"
).resolve()
SCHEMA = "specialist-qwen36-memory-smoke-pretrainer-contract-v1"
INPUTS = (
    ("gpu0_shared_r16_seq2048.json", 0, True, None, 16, 4_915_200),
    ("gpu1_routed_r2_shared_r16_seq2048.json", 1, False, 2, 16, 120_258_560),
    ("gpu2_routed_r4_shared_r16_seq2048.json", 2, False, 4, 16, 235_601_920),
    ("gpu3_routed_r4_shared_r16_seq2048.json", 3, False, 4, 16, 235_601_920),
)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite_nonnegative(value: Any, label: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), label)
    result = float(value)
    _require(result >= 0.0 and result < float("inf"), label)
    return result


def _validate_receipt(
    path: Path,
    gpu_index: int,
    shared_only: bool,
    routed_rank: int | None,
    shared_rank: int,
    trainable_elements: int,
) -> dict:
    _require(path.is_file() and not path.is_symlink(), f"unsafe receipt: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"receipt is not an object: {path}")
    _require(
        value.get("schema") == "qwen36-expert-lora-synthetic-memory-smoke-v1",
        f"receipt schema changed: {path}",
    )
    _require(value.get("authority") == {
        "synthetic_token_ids_only": True,
        "dataset_or_evaluation_source_opened": False,
        "checkpoint_weights_loaded": True,
        "optimizer_created": True,
        "synthetic_adapter_update_performed": True,
    }, f"receipt authority changed: {path}")
    gpu = value.get("gpu")
    _require(isinstance(gpu, dict), f"missing GPU receipt: {path}")
    _require(
        gpu.get("cuda_visible_devices") == str(gpu_index),
        f"wrong physical GPU assignment: {path}",
    )
    _require(gpu.get("compute_capability") == "12.0", f"wrong GPU CC: {path}")
    total = _finite_nonnegative(gpu.get("total_memory_gib"), "invalid total memory")
    _require(total >= 95.0, f"insufficient recorded GPU memory: {path}")

    configuration = value.get("configuration")
    _require(isinstance(configuration, dict), f"missing configuration: {path}")
    _require(configuration == {
        "dtype": "torch.bfloat16",
        "gradient_checkpointing": True,
        "gradient_checkpointing_use_reentrant": False,
        "hybrid_module_count": 30,
        "routed_rank": routed_rank,
        "sequence_length": 2048,
        "shared_only": shared_only,
        "shared_rank": shared_rank,
        "use_cache": False,
    }, f"configuration changed: {path}")
    scope = value.get("scope")
    _require(isinstance(scope, dict), f"missing trainable scope: {path}")
    _require(scope.get("trainable_elements") == trainable_elements,
             f"trainable element count changed: {path}")
    _require(scope.get("target_count") == (120 if shared_only else 200),
             f"target count changed: {path}")
    _require(scope.get("trainable_tensor_count") == (240 if shared_only else 400),
             f"trainable tensor count changed: {path}")
    for identity_name in ("preattach_identity_sha256", "postattach_identity_sha256"):
        _require(
            isinstance(scope.get(identity_name), str)
            and re.fullmatch(r"[0-9a-f]{64}", scope[identity_name]) is not None,
            f"invalid {identity_name}: {path}",
        )

    measurement = value.get("measurement")
    _require(isinstance(measurement, dict), f"missing measurement: {path}")
    _require(_finite_nonnegative(measurement.get("loss"), "invalid loss") > 0.0,
             f"nonpositive loss: {path}")
    _require(
        _finite_nonnegative(measurement.get("gradient_norm_before_clip"),
                            "invalid gradient norm") > 0.0,
        f"nonpositive gradient norm: {path}",
    )
    _require(_finite_nonnegative(measurement.get("step_seconds"),
                                 "invalid step seconds") > 0.0,
             f"nonpositive step duration: {path}")
    _require(_finite_nonnegative(measurement.get("tokens_per_second"),
                                 "invalid throughput") > 0.0,
             f"nonpositive throughput: {path}")
    stages = {}
    for stage_name in ("after_setup", "after_backward", "after_step"):
        stage = measurement.get(stage_name)
        _require(isinstance(stage, dict), f"missing {stage_name}: {path}")
        checked = {
            name: _finite_nonnegative(stage.get(name), f"invalid {stage_name}.{name}")
            for name in (
                "allocated_gib", "reserved_gib", "peak_allocated_gib",
                "peak_reserved_gib",
            )
        }
        _require(checked["reserved_gib"] + 1e-9 >= checked["allocated_gib"],
                 f"reserved below allocated: {path}")
        _require(checked["peak_reserved_gib"] + 1e-9 >= checked["peak_allocated_gib"],
                 f"peak reserved below peak allocated: {path}")
        stages[stage_name] = checked
    peak_reserved = max(stage["peak_reserved_gib"] for stage in stages.values())
    headroom = total - peak_reserved
    _require(headroom >= 8.0, f"less than 8 GiB operational headroom: {path}")
    return {
        "artifact": path.relative_to(ROOT).as_posix(),
        "artifact_bytes": path.stat().st_size,
        "artifact_sha256": file_sha256(path),
        "physical_gpu_index": gpu_index,
        "gpu_name": gpu["name"],
        "total_memory_gib": total,
        "configuration": configuration,
        "scope": scope,
        "measurements": {
            "loss": measurement["loss"],
            "gradient_norm_before_clip": measurement["gradient_norm_before_clip"],
            "step_seconds": measurement["step_seconds"],
            "tokens_per_second": measurement["tokens_per_second"],
            "peak_allocated_gib": max(
                stage["peak_allocated_gib"] for stage in stages.values()
            ),
            "peak_reserved_gib": peak_reserved,
            "headroom_gib": headroom,
        },
    }


def build() -> dict:
    receipts = []
    for filename, gpu, shared_only, routed_rank, shared_rank, trainable in INPUTS:
        receipts.append(_validate_receipt(
            INPUT_DIRECTORY / filename,
            gpu,
            shared_only,
            routed_rank,
            shared_rank,
            trainable,
        ))
    _require(
        [item["physical_gpu_index"] for item in receipts] == [0, 1, 2, 3],
        "four distinct physical GPU assignments were not sealed",
    )
    rank4 = [item for item in receipts if item["configuration"]["routed_rank"] == 4]
    _require(len(rank4) == 2, "rank-4 reproducibility needs two GPU receipts")
    rank4_peak_span = max(
        item["measurements"]["peak_reserved_gib"] for item in rank4
    ) - min(item["measurements"]["peak_reserved_gib"] for item in rank4)
    result = {
        "schema": SCHEMA,
        "authority": {
            "synthetic_token_ids_only": True,
            "dataset_or_evaluation_source_opened": False,
            "actual_checkpoint_weights_loaded": True,
            "actual_expert_lora_adapters_attached": True,
            "one_optimizer_update_per_receipt": True,
            "training_dataset_opened": False,
            "training_launched": False,
            "final_mixed_trainer_validated": False,
            "checkpoint_resume_validated": False,
        },
        "result": {
            "all_four_physical_gpus_passed": True,
            "sequence_length": 2048,
            "precision": "bfloat16",
            "maximum_peak_reserved_gib": max(
                item["measurements"]["peak_reserved_gib"] for item in receipts
            ),
            "minimum_headroom_gib": min(
                item["measurements"]["headroom_gib"] for item in receipts
            ),
            "minimum_operational_headroom_gate_gib": 8.0,
            "operational_headroom_gate_passed": True,
            "rank4_peak_reserved_span_gib_across_two_gpus": rank4_peak_span,
            "pretrainer_memory_smoke_passed": True,
            "final_trainer_memory_gate_pending": True,
        },
        "receipts": receipts,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build()
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if args.check:
        _require(OUTPUT.is_file(), "memory-smoke contract is missing")
        _require(OUTPUT.read_bytes() == payload, "memory-smoke contract is stale")
    else:
        _atomic_write(OUTPUT, payload)
    print(json.dumps({
        "contract": OUTPUT.relative_to(ROOT).as_posix(),
        "content_sha256": value["content_sha256_before_self_field"],
        "maximum_peak_reserved_gib": value["result"]["maximum_peak_reserved_gib"],
        "minimum_headroom_gib": value["result"]["minimum_headroom_gib"],
        "status": "checked" if args.check else "written",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
