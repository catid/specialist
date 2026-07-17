#!/usr/bin/env python3
"""Data-free BF16 Mamba-SSM cache isolation on the V76 BF16-KV control."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import probe_vllm_fp8_attested_v76 as base


SCHEMA_V78C = "v78c-qwen36-bf16-kv-bf16-mamba-ssm-preflight"


def upgraded_receipt_v78c(value: dict, resolved: dict) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema") != base.SCHEMA_V76
        or value.get("runtime", {}).get("gpu_memory_utilization") != 0.50
        or value.get("preflight_gates", {}).get(
            "scored_evaluation_or_training_authorized"
        ) is not False
    ):
        raise RuntimeError("V78c underlying V76 receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = dict(value)
    original.pop("content_sha256_before_self_field", None)
    canonical = base.base.base.base.base.base.canonical_sha256
    if canonical(original) != claimed:
        raise RuntimeError("V78c underlying V76 receipt identity changed")
    if resolved != {
        "cache_dtype": "auto",
        "mamba_ssm_cache_dtype": "bfloat16",
        "resolved_from_live_engine": True,
    }:
        raise RuntimeError("V78c live hybrid-cache certificate changed")
    result = dict(original)
    result["schema"] = SCHEMA_V78C
    result["runtime"] = dict(result["runtime"])
    result["runtime"]["mamba_ssm_cache_dtype"] = "bfloat16"
    result["resolved_hybrid_cache_certificate"] = dict(resolved)
    result["single_variable_change_from_v76"] = {
        "mamba_ssm_cache_dtype": ["float32", "bfloat16"],
    }
    result["preflight_gates"] = dict(result["preflight_gates"])
    result["preflight_gates"].update({
        "bf16_mamba_ssm_cache_resolved": True,
        "mamba_ssm_semantic_and_ood_gate_pending": True,
        "scored_evaluation_or_training_authorized": False,
    })
    result["authority"] = {
        "data_free_hybrid_cache_isolation_only": True,
        "scored_evaluation_training_checkpoint_or_promotion_allowed": False,
    }
    result["content_sha256_before_self_field"] = canonical(result)
    return result


def publish_v78c(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.v78c-{os.getpid()}")
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

    def bf16_mamba_ssm_llm(*args, **kwargs):
        if (
            kwargs.get("quantization") != "fp8"
            or kwargs.get("gpu_memory_utilization") != 0.50
            or "kv_cache_dtype" in kwargs
            or "mamba_ssm_cache_dtype" in kwargs
        ):
            raise RuntimeError("V78c underlying V76 engine contract changed")
        kwargs = dict(kwargs)
        kwargs["mamba_ssm_cache_dtype"] = "bfloat16"
        engine = original_llm(*args, **kwargs)
        cache = engine.llm_engine.vllm_config.cache_config
        observed.append({
            "cache_dtype": cache.cache_dtype,
            "mamba_ssm_cache_dtype": cache.mamba_ssm_cache_dtype,
            "resolved_from_live_engine": True,
        })
        return engine

    vllm.LLM = bf16_mamba_ssm_llm
    try:
        status = base.main()
    finally:
        vllm.LLM = original_llm
    if status != 0 or not output.is_file() or len(observed) != 1:
        raise RuntimeError("V78c underlying hybrid-cache probe failed")
    upgraded = upgraded_receipt_v78c(
        json.loads(output.read_text(encoding="utf-8")), observed[0]
    )
    publish_v78c(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V78C,
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
