#!/usr/bin/env python3
"""Right-size the V73 serialized-FP8 cache budget to expose VRAM savings.

V73 holds ``gpu_memory_utilization`` at 0.82, so vLLM converts the FP8 weight
savings into a much larger KV cache.  V74 changes only that fraction to 0.50;
the model, adapter pair, workload, eager kernels, and precision format remain
the V73 contract.  This is a data-free capacity diagnostic, not a scored arm.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import probe_vllm_quantized_adapter_switch_v73 as base


SCHEMA_V74 = "v74-qwen36-fp8-rightsized-vram-preflight"
BASE_GPU_MEMORY_UTILIZATION_V74 = 0.82
TARGET_GPU_MEMORY_UTILIZATION_V74 = 0.50


def validate_argv_v74(argv: list[str]) -> Path:
    arm_positions = [index for index, value in enumerate(argv)
                     if value == "--precision-arm"]
    output_positions = [index for index, value in enumerate(argv)
                        if value == "--output"]
    if (
        len(arm_positions) != 1
        or arm_positions[0] + 1 >= len(argv)
        or argv[arm_positions[0] + 1] != "fp8_serialized"
    ):
        raise RuntimeError("V74 requires exactly the serialized-FP8 arm")
    if len(output_positions) != 1 or output_positions[0] + 1 >= len(argv):
        raise RuntimeError("V74 requires exactly one --output")
    if "--graph" in argv:
        raise RuntimeError("V74 retains the eager V73 execution contract")
    return Path(argv[output_positions[0] + 1]).resolve()


def resolved_budget_v74(engine: object) -> dict:
    llm_engine = getattr(engine, "llm_engine", None)
    config = getattr(llm_engine, "vllm_config", None)
    cache = getattr(config, "cache_config", None)
    value = getattr(cache, "gpu_memory_utilization", None)
    if value != TARGET_GPU_MEMORY_UTILIZATION_V74:
        raise RuntimeError("V74 live engine resolved a different memory fraction")
    return {
        "gpu_memory_utilization": value,
        "resolved_from_live_engine": True,
    }


def upgraded_receipt_v74(value: dict, resolved: dict) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema") != base.SCHEMA_V73
        or value.get("precision_arm") != "fp8_serialized"
        or value.get("runtime", {}).get("resolved_quantization") != "fp8"
        or value.get("runtime", {}).get("enforce_eager") is not True
        or value.get("preflight_gates", {}).get(
            "scored_evaluation_or_training_authorized"
        ) is not False
    ):
        raise RuntimeError("V74 underlying V73 receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = dict(value)
    original.pop("content_sha256_before_self_field", None)
    if base.base.base.canonical_sha256(original) != claimed:
        raise RuntimeError("V74 underlying V73 receipt identity changed")
    if resolved != {
        "gpu_memory_utilization": TARGET_GPU_MEMORY_UTILIZATION_V74,
        "resolved_from_live_engine": True,
    }:
        raise RuntimeError("V74 live budget certificate changed")
    result = dict(original)
    result["schema"] = SCHEMA_V74
    result["runtime"] = dict(result["runtime"])
    result["runtime"]["gpu_memory_utilization"] = (
        TARGET_GPU_MEMORY_UTILIZATION_V74
    )
    result["resolved_memory_budget_certificate"] = dict(resolved)
    result["single_variable_change_from_v73_fp8"] = {
        "gpu_memory_utilization": [
            BASE_GPU_MEMORY_UTILIZATION_V74,
            TARGET_GPU_MEMORY_UTILIZATION_V74,
        ]
    }
    result["authority"] = {
        "data_free_capacity_and_switch_diagnostic_only": True,
        "scored_evaluation_training_checkpoint_or_promotion_allowed": False,
    }
    result["content_sha256_before_self_field"] = (
        base.base.base.canonical_sha256(result)
    )
    return result


def publish_v74(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.v74-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    output = validate_argv_v74(sys.argv[1:])
    import vllm

    original_llm = vllm.LLM
    observed: list[dict] = []

    def rightsized_llm(*args, **kwargs):
        if (
            kwargs.get("gpu_memory_utilization")
            != BASE_GPU_MEMORY_UTILIZATION_V74
            or kwargs.get("quantization") != "fp8"
        ):
            raise RuntimeError("V74 underlying V73 engine contract changed")
        kwargs = dict(kwargs)
        kwargs["gpu_memory_utilization"] = TARGET_GPU_MEMORY_UTILIZATION_V74
        engine = original_llm(*args, **kwargs)
        observed.append(resolved_budget_v74(engine))
        return engine

    vllm.LLM = rightsized_llm
    try:
        status = base.main()
    finally:
        vllm.LLM = original_llm
    if status != 0 or not output.is_file() or len(observed) != 1:
        raise RuntimeError("V74 underlying right-sized FP8 probe failed")
    upgraded = upgraded_receipt_v74(
        json.loads(output.read_text(encoding="utf-8")), observed[0]
    )
    publish_v74(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V74,
        "output": str(output),
        "content_sha256": upgraded["content_sha256_before_self_field"],
        "wall_runtime_seconds": upgraded[
            "wall_runtime_seconds_excluding_model_load_and_cleanup"
        ],
        "preflight_gates": upgraded["preflight_gates"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
