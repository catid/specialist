#!/usr/bin/env python3
"""Run the right-sized FP8 probe with explicit, non-fallback kernel choices."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import probe_vllm_fp8_rightsized_v74 as base


SCHEMA_V75 = "v75-qwen36-fp8-clean-environment-preflight"


def upgraded_receipt_v75(value: dict, environment: dict) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema") != base.SCHEMA_V74
        or value.get("runtime", {}).get("resolved_quantization") != "fp8"
        or value.get("runtime", {}).get("gpu_memory_utilization") != 0.50
        or value.get("authority", {}).get(
            "scored_evaluation_training_checkpoint_or_promotion_allowed"
        ) is not False
    ):
        raise RuntimeError("V75 underlying V74 receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = dict(value)
    original.pop("content_sha256_before_self_field", None)
    if base.base.base.base.canonical_sha256(original) != claimed:
        raise RuntimeError("V75 underlying V74 receipt identity changed")
    expected_environment = {
        "VLLM_USE_DEEP_GEMM": "0",
        "enable_flashinfer_autotune": False,
        "tuned_config_folder_was_fresh_and_empty": True,
        "explicit_cutlass_path_requested_without_runtime_fallback": True,
    }
    if environment != expected_environment:
        raise RuntimeError("V75 explicit kernel environment changed")
    result = dict(original)
    result["schema"] = SCHEMA_V75
    result["explicit_kernel_environment"] = dict(environment)
    result["runtime"] = dict(result["runtime"])
    result["runtime"]["enable_flashinfer_autotune"] = False
    result["runtime"]["starting_moe_tuning_table"] = "fresh_empty_default"
    result["log_gate"] = {
        "forbidden_fallback_fragments_must_be_checked_by_orchestrator": True,
        "forbidden_fallback_fragments_passed": None,
    }
    result["content_sha256_before_self_field"] = (
        base.base.base.base.canonical_sha256(result)
    )
    return result


def publish_v75(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.v75-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    output = base.validate_argv_v74(sys.argv[1:])
    prior_deep_gemm = os.environ.get("VLLM_USE_DEEP_GEMM")
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    import vllm

    original_llm = vllm.LLM
    original_tuned = base.base.base.base.TUNED
    observed: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="v75-empty-fp8-moe-") as folder:
        empty = Path(folder).resolve()
        if any(empty.iterdir()):
            raise RuntimeError("V75 tuning directory is not fresh and empty")

        def explicit_kernel_llm(*args, **kwargs):
            if (
                kwargs.get("quantization") != "fp8"
                or kwargs.get("gpu_memory_utilization") != 0.50
                or "enable_flashinfer_autotune" in kwargs
                or os.environ.get("VLLM_TUNED_CONFIG_FOLDER") != str(empty)
                or os.environ.get("VLLM_USE_DEEP_GEMM") != "0"
                or any(empty.iterdir())
            ):
                raise RuntimeError("V75 inherited engine/environment contract changed")
            kwargs = dict(kwargs)
            kwargs["enable_flashinfer_autotune"] = False
            engine = original_llm(*args, **kwargs)
            config = engine.llm_engine.vllm_config
            if (
                config.kernel_config.enable_flashinfer_autotune is not False
                or config.model_config.quantization != "fp8"
                or config.cache_config.gpu_memory_utilization != 0.50
            ):
                raise RuntimeError("V75 live engine ignored explicit kernel settings")
            observed.append({
                "VLLM_USE_DEEP_GEMM": "0",
                "enable_flashinfer_autotune": False,
                "tuned_config_folder_was_fresh_and_empty": True,
                "explicit_cutlass_path_requested_without_runtime_fallback": True,
            })
            return engine

        base.base.base.base.TUNED = empty
        vllm.LLM = explicit_kernel_llm
        try:
            status = base.main()
        finally:
            base.base.base.base.TUNED = original_tuned
            vllm.LLM = original_llm
            if prior_deep_gemm is None:
                os.environ.pop("VLLM_USE_DEEP_GEMM", None)
            else:
                os.environ["VLLM_USE_DEEP_GEMM"] = prior_deep_gemm
    if status != 0 or not output.is_file() or len(observed) != 1:
        raise RuntimeError("V75 underlying explicit-kernel probe failed")
    upgraded = upgraded_receipt_v75(
        json.loads(output.read_text(encoding="utf-8")), observed[0]
    )
    publish_v75(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V75,
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
