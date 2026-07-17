#!/usr/bin/env python3
"""Data-free BF16 Mamba-SSM cache follow-up to the V78 FP8-KV arm."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import probe_vllm_fp8_kv_cache_v78 as base


SCHEMA_V78B = "v78b-qwen36-fp8-kv-bf16-mamba-ssm-preflight"
MAMBA_SSM_DTYPE_V78B = "bfloat16"


def upgraded_receipt_v78b(value: dict, resolved: dict) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema") != base.SCHEMA_V78
        or value.get("runtime", {}).get("kv_cache_dtype")
        != "fp8_per_token_head"
        or value.get("resolved_kv_cache_certificate", {}).get(
            "mamba_ssm_cache_dtype"
        ) != MAMBA_SSM_DTYPE_V78B
        or value.get("preflight_gates", {}).get(
            "scored_evaluation_or_training_authorized"
        ) is not False
    ):
        raise RuntimeError("V78b underlying V78 receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = dict(value)
    original.pop("content_sha256_before_self_field", None)
    canonical = base.base.base.base.base.base.base.canonical_sha256
    if canonical(original) != claimed:
        raise RuntimeError("V78b underlying V78 receipt identity changed")
    if resolved != {
        "mamba_ssm_cache_dtype": MAMBA_SSM_DTYPE_V78B,
        "resolved_from_live_engine": True,
    }:
        raise RuntimeError("V78b live Mamba cache certificate changed")
    result = dict(original)
    result["schema"] = SCHEMA_V78B
    result["runtime"] = dict(result["runtime"])
    result["runtime"]["mamba_ssm_cache_dtype"] = MAMBA_SSM_DTYPE_V78B
    result["resolved_mamba_ssm_cache_certificate"] = dict(resolved)
    parent_delta = result.pop("single_variable_change_from_v76", None)
    if parent_delta != {
        "kv_cache_dtype": ["auto_resolved_bfloat16", "fp8_per_token_head"]
    }:
        raise RuntimeError("V78b parent KV delta changed")
    result["single_variable_change_from_v78"] = {
        "mamba_ssm_cache_dtype": ["float32", MAMBA_SSM_DTYPE_V78B],
    }
    result["combined_changes_from_v76"] = {
        **parent_delta,
        "mamba_ssm_cache_dtype": ["float32", MAMBA_SSM_DTYPE_V78B],
    }
    result["preflight_gates"] = dict(result["preflight_gates"])
    result["preflight_gates"].update({
        "bf16_mamba_ssm_cache_resolved": True,
        "mamba_ssm_semantic_and_ood_gate_pending": True,
        "scored_evaluation_or_training_authorized": False,
    })
    result["authority"] = {
        "data_free_hybrid_cache_capacity_diagnostic_only": True,
        "scored_evaluation_training_checkpoint_or_promotion_allowed": False,
    }
    result["content_sha256_before_self_field"] = canonical(result)
    return result


def publish_v78b(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.v78b-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    output = base.base.base.base.validate_argv_v74(sys.argv[1:])
    import vllm

    original_llm = vllm.LLM
    observed: list[dict] = []

    def bf16_mamba_ssm_llm(*args, **kwargs):
        if (
            kwargs.get("kv_cache_dtype") != "fp8_per_token_head"
            or "mamba_ssm_cache_dtype" in kwargs
        ):
            raise RuntimeError("V78b underlying V78 engine contract changed")
        kwargs = dict(kwargs)
        kwargs["mamba_ssm_cache_dtype"] = MAMBA_SSM_DTYPE_V78B
        engine = original_llm(*args, **kwargs)
        cache = engine.llm_engine.vllm_config.cache_config
        observed.append({
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
        raise RuntimeError("V78b underlying BF16 Mamba-SSM probe failed")
    upgraded = upgraded_receipt_v78b(
        json.loads(output.read_text(encoding="utf-8")), observed[0]
    )
    publish_v78b(output, upgraded)
    print(json.dumps({
        "schema": SCHEMA_V78B,
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
