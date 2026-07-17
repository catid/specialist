#!/usr/bin/env python3
"""Data-free FP8 per-token-head KV-cache preflight on the V76 FP8 arm."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import probe_vllm_fp8_attested_v76 as base


SCHEMA_V78 = "v78-qwen36-fp8-per-token-head-kv-preflight"
KV_DTYPE_V78 = "fp8_per_token_head"


def upgraded_receipt_v78(value: dict, resolved: dict) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema") != base.SCHEMA_V76
        or value.get("precision_arm") != "fp8_serialized"
        or value.get("runtime", {}).get("gpu_memory_utilization") != 0.50
        or value.get("routed_fp8_runtime_attestation", {}).get(
            "fp8_moe_method_count"
        ) != 40
        or value.get("preflight_gates", {}).get(
            "scored_evaluation_or_training_authorized"
        ) is not False
    ):
        raise RuntimeError("V78 underlying V76 receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = dict(value)
    original.pop("content_sha256_before_self_field", None)
    if base.base.base.base.base.base.canonical_sha256(original) != claimed:
        raise RuntimeError("V78 underlying V76 receipt identity changed")
    if (
        resolved.get("cache_dtype") != KV_DTYPE_V78
        or resolved.get("calculate_kv_scales") is not False
        or resolved.get("kv_cache_dtype_skip_layers") != []
        or resolved.get("resolved_from_live_engine") is not True
    ):
        raise RuntimeError("V78 live KV-cache certificate changed")
    result = dict(original)
    result["schema"] = SCHEMA_V78
    result["runtime"] = dict(result["runtime"])
    result["runtime"]["kv_cache_dtype"] = KV_DTYPE_V78
    result["resolved_kv_cache_certificate"] = dict(resolved)
    result["single_variable_change_from_v76"] = {
        "kv_cache_dtype": ["auto_resolved_bfloat16", KV_DTYPE_V78],
    }
    result["preflight_gates"] = dict(result["preflight_gates"])
    result["preflight_gates"].update({
        "fp8_per_token_head_kv_resolved": True,
        "kv_token_semantic_and_ood_gate_pending": True,
        "scored_evaluation_or_training_authorized": False,
    })
    result["authority"] = {
        "data_free_kv_capacity_and_switch_diagnostic_only": True,
        "scored_evaluation_training_checkpoint_or_promotion_allowed": False,
    }
    result["content_sha256_before_self_field"] = (
        base.base.base.base.base.base.canonical_sha256(result)
    )
    return result


def publish_v78(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.v78-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    output = base.base.base.validate_argv_v74(sys.argv[1:])
    import vllm

    original_llm = vllm.LLM
    observed: list[dict] = []

    def fp8_kv_llm(*args, **kwargs):
        if (
            kwargs.get("quantization") != "fp8"
            or kwargs.get("gpu_memory_utilization") != 0.50
            or "kv_cache_dtype" in kwargs
            or "calculate_kv_scales" in kwargs
        ):
            raise RuntimeError("V78 underlying V76 engine contract changed")
        kwargs = dict(kwargs)
        kwargs["kv_cache_dtype"] = KV_DTYPE_V78
        kwargs["calculate_kv_scales"] = False
        engine = original_llm(*args, **kwargs)
        cache = engine.llm_engine.vllm_config.cache_config
        observed.append({
            "cache_dtype": cache.cache_dtype,
            "calculate_kv_scales": cache.calculate_kv_scales,
            "kv_cache_dtype_skip_layers": list(
                cache.kv_cache_dtype_skip_layers
            ),
            "mamba_cache_dtype": cache.mamba_cache_dtype,
            "mamba_ssm_cache_dtype": cache.mamba_ssm_cache_dtype,
            "resolved_from_live_engine": True,
        })
        return engine

    vllm.LLM = fp8_kv_llm
    try:
        status = base.main()
    finally:
        vllm.LLM = original_llm
    if status != 0 or not output.is_file() or len(observed) != 1:
        raise RuntimeError("V78 underlying FP8 KV-cache probe failed")
    upgraded = upgraded_receipt_v78(
        json.loads(output.read_text(encoding="utf-8")), observed[0]
    )
    publish_v78(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V78,
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
