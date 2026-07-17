#!/usr/bin/env python3
"""Right-size V78 FP8 hybrid KV cache while retaining V76 capacity.

V79 reproduces the sealed V78 runtime contract from the V73 data-free probe,
then changes only ``gpu_memory_utilization`` from 0.500 to 0.485.  Rebuilding
the thin wrapper from V73 avoids falsifying V74's historical live-engine
certificate, which correctly records 0.500 for that prior run.  This remains
a data-free capacity and adapter-switch diagnostic; it cannot authorize scored
evaluation, training, checkpointing, or promotion.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import probe_vllm_fp8_attested_v76 as audit_base
import probe_vllm_quantized_adapter_switch_v73 as base


ROOT = Path(__file__).resolve().parent
PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fp8_kv_capacity_matched_v79.json"
)
PREREG_CONTENT_SHA256 = (
    "6c73ac0f6bf4019cdf297546e4315dc99b68d9549a24c03f4eaa9c8ebb589023"
)
PREREG_FILE_SHA256 = (
    "0e195c05fd72e36656ee6536d6656d932aac0028fbbc7983f688df9dc7b18753"
)

SCHEMA_V79 = "v79-qwen36-fp8-kv-capacity-matched-preflight"
BASE_GPU_MEMORY_UTILIZATION_V73 = 0.82
REFERENCE_GPU_MEMORY_UTILIZATION_V78 = 0.50
TARGET_GPU_MEMORY_UTILIZATION_V79 = 0.485
KV_DTYPE_V79 = "fp8_per_token_head"
MIN_KV_CAPACITY_TOKENS_V79 = 161_792
MIN_KV_MAX_CONCURRENCY_V79 = 79.0
V78_PROBE_SHA256 = (
    "916aa316494619030b6232d2596486ae43fc58709063b1045617c358b3073485"
)
V78_RUN_BUNDLE_SHA256 = (
    "e6df12c976910948c1026249b05fc065932169897aa5a09ff984b6d765385463"
)
PARAMETER_RESIDENCY_V79 = {
    "schema": "v76-live-named-parameter-residency",
    "components": {
        "language": {
            "device_counts": {"cuda:0": 813},
            "dtype_counts": {
                "torch.bfloat16": 303,
                "torch.float32": 270,
                "torch.float8_e4m3fn": 240,
            },
            "logical_bytes": 35_712_084_096,
            "parameter_count": 813,
            "parameter_names_sha256": (
                "a850f55c3f02ef904041d48b29f13af2d29834da200f92dcc9728760cb185b90"
            ),
        }
    },
    "named_parameters_remove_duplicate_default": True,
    "total_logical_bytes": 35_712_084_096,
    "total_parameter_count": 813,
}


def canonical_sha256_v79(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def validate_argv_v79(argv: list[str]) -> Path:
    arm_positions = [
        index for index, value in enumerate(argv) if value == "--precision-arm"
    ]
    output_positions = [
        index for index, value in enumerate(argv) if value == "--output"
    ]
    if (
        len(arm_positions) != 1
        or arm_positions[0] + 1 >= len(argv)
        or argv[arm_positions[0] + 1] != "fp8_serialized"
    ):
        raise RuntimeError("V79 requires exactly the serialized-FP8 arm")
    if len(output_positions) != 1 or output_positions[0] + 1 >= len(argv):
        raise RuntimeError("V79 requires exactly one --output")
    if "--graph" in argv:
        raise RuntimeError("V79 retains the eager V78 execution contract")
    output = Path(argv[output_positions[0] + 1]).resolve()
    if output.exists():
        raise RuntimeError("V79 requires a fresh output path")
    return output


def validate_static_v79() -> dict:
    if not PREREG.is_file() or PREREG.is_symlink():
        raise RuntimeError("V79 preregistration is missing or unsafe")
    if base.file_sha256_v73(PREREG) != PREREG_FILE_SHA256:
        raise RuntimeError("V79 preregistration file identity changed")
    value = json.loads(PREREG.read_text(encoding="ascii"))
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    if (
        claimed != PREREG_CONTENT_SHA256
        or canonical_sha256_v79(body) != claimed
        or value.get("schema")
        != "v79-qwen36-fp8-kv-capacity-matched-preregistration"
        or value.get("selected_runtime", {}).get("gpu_memory_utilization")
        != TARGET_GPU_MEMORY_UTILIZATION_V79
        or value.get("selected_runtime", {}).get("kv_cache_dtype")
        != KV_DTYPE_V79
        or value.get("live_acceptance", {}).get("capacity", {}).get(
            "minimum_tokens_per_actor"
        )
        != MIN_KV_CAPACITY_TOKENS_V79
        or value.get("authority", {}).get("scored_or_training_authority")
        is not False
    ):
        raise RuntimeError("V79 preregistration content or authority changed")
    return value


def rightsized_kwargs_v79(kwargs: dict) -> dict:
    """Validate the V73 call surface and apply the sealed V78+V79 delta."""
    if (
        kwargs.get("quantization") != "fp8"
        or kwargs.get("gpu_memory_utilization")
        != BASE_GPU_MEMORY_UTILIZATION_V73
        or kwargs.get("enforce_eager") is not True
        or kwargs.get("moe_backend") != "triton"
        or "kv_cache_dtype" in kwargs
        or "calculate_kv_scales" in kwargs
        or "enable_flashinfer_autotune" in kwargs
    ):
        raise RuntimeError("V79 underlying V73 engine contract changed")
    result = dict(kwargs)
    result.update(
        {
            "gpu_memory_utilization": TARGET_GPU_MEMORY_UTILIZATION_V79,
            "kv_cache_dtype": KV_DTYPE_V79,
            "calculate_kv_scales": False,
            "enable_flashinfer_autotune": False,
        }
    )
    return result


def resolved_runtime_v79(engine: object) -> dict:
    llm_engine = getattr(engine, "llm_engine", None)
    config = getattr(llm_engine, "vllm_config", None)
    cache = getattr(config, "cache_config", None)
    kernel = getattr(config, "kernel_config", None)
    model = getattr(config, "model_config", None)
    result = {
        "gpu_memory_utilization": getattr(
            cache, "gpu_memory_utilization", None
        ),
        "cache_dtype": getattr(cache, "cache_dtype", None),
        "calculate_kv_scales": getattr(cache, "calculate_kv_scales", None),
        "kv_cache_dtype_skip_layers": list(
            getattr(cache, "kv_cache_dtype_skip_layers", [])
        ),
        "mamba_cache_dtype": getattr(cache, "mamba_cache_dtype", None),
        "mamba_ssm_cache_dtype": getattr(
            cache, "mamba_ssm_cache_dtype", None
        ),
        "kv_cache_size_tokens": getattr(
            cache, "kv_cache_size_tokens", None
        ),
        "kv_cache_max_concurrency": getattr(
            cache, "kv_cache_max_concurrency", None
        ),
        "num_gpu_blocks": getattr(cache, "num_gpu_blocks", None),
        "block_size": getattr(cache, "block_size", None),
        "enable_flashinfer_autotune": getattr(
            kernel, "enable_flashinfer_autotune", None
        ),
        "model_quantization": getattr(model, "quantization", None),
        "resolved_from_live_engine": True,
    }
    if (
        result["gpu_memory_utilization"]
        != TARGET_GPU_MEMORY_UTILIZATION_V79
        or result["cache_dtype"] != KV_DTYPE_V79
        or result["calculate_kv_scales"] is not False
        or result["kv_cache_dtype_skip_layers"] != []
        or result["mamba_cache_dtype"] != "auto"
        or result["mamba_ssm_cache_dtype"] != "float32"
        or not isinstance(result["kv_cache_size_tokens"], int)
        or result["kv_cache_size_tokens"] < MIN_KV_CAPACITY_TOKENS_V79
        or not isinstance(result["kv_cache_max_concurrency"], (int, float))
        or result["kv_cache_max_concurrency"]
        < MIN_KV_MAX_CONCURRENCY_V79
        or not isinstance(result["num_gpu_blocks"], int)
        or result["num_gpu_blocks"] <= 0
        or not isinstance(result["block_size"], int)
        or result["block_size"] <= 0
        or result["enable_flashinfer_autotune"] is not False
        or result["model_quantization"] != "fp8"
    ):
        raise RuntimeError(
            "V79 live hybrid-cache/capacity certificate failed: "
            + json.dumps(result, sort_keys=True)
        )
    return result


def summarize_generation_timings_v79(rows: list[dict]) -> dict:
    call_plan = [
        "reference",
        "candidate",
        "candidate",
        "reference",
        "reference",
        "candidate",
        "candidate",
        "reference",
    ]
    if (
        not isinstance(rows, list)
        or len(rows) != 10
        or [row.get("label") for row in rows[:2]]
        != ["reference", "candidate"]
        or [row.get("label") for row in rows[2:]] != call_plan
        or any(
            row.get("call_index") != index
            or row.get("request_count") != 68
            or not isinstance(row.get("generated_token_count"), int)
            or row["generated_token_count"] <= 0
            or not isinstance(row.get("elapsed_seconds"), (int, float))
            or row["elapsed_seconds"] <= 0
            for index, row in enumerate(rows)
        )
    ):
        raise RuntimeError("V79 generation timing surface changed")
    measured = rows[2:]
    elapsed = sorted(float(row["elapsed_seconds"]) for row in measured)
    generated = sum(row["generated_token_count"] for row in measured)
    total_seconds = sum(row["elapsed_seconds"] for row in measured)
    p95_index = max(0, min(len(elapsed) - 1, (95 * len(elapsed) + 99) // 100 - 1))
    return {
        "schema": "v79-data-free-generation-performance",
        "warmup_call_count": 2,
        "measured_call_count": 8,
        "request_count_per_call": 68,
        "measured_generated_token_count": generated,
        "measured_generation_seconds_sum": total_seconds,
        "aggregate_generated_tokens_per_second": generated / total_seconds,
        "median_call_latency_seconds": (
            elapsed[len(elapsed) // 2 - 1] + elapsed[len(elapsed) // 2]
        )
        / 2,
        "p95_call_latency_seconds_nearest_rank": elapsed[p95_index],
        "max_call_latency_seconds": elapsed[-1],
        "calls": rows,
        "prompt_or_generation_text_persisted": False,
        "token_ids_persisted": False,
    }


def upgraded_receipt_v79(
    value: dict,
    resolved: dict,
    routed_audit: dict,
    generation_performance: dict,
) -> dict:
    if (
        not isinstance(value, dict)
        or value.get("schema") != base.SCHEMA_V73
        or value.get("precision_arm") != "fp8_serialized"
        or value.get("runtime", {}).get("gpu_memory_utilization")
        != BASE_GPU_MEMORY_UTILIZATION_V73
        or value.get("runtime", {}).get("resolved_quantization") != "fp8"
        or value.get("runtime", {}).get("enforce_eager") is not True
        or value.get("adapter_update_or_hpo_performed") is not False
        or value.get("source_dataset_rows_opened") != 0
        or value.get("protected_ood_shadow_or_terminal_opened") is not False
        or value.get("preflight_gates", {}).get(
            "scored_evaluation_or_training_authorized"
        )
        is not False
    ):
        raise RuntimeError("V79 underlying V73 receipt changed")
    claimed = value.get("content_sha256_before_self_field")
    original = copy.deepcopy(value)
    original.pop("content_sha256_before_self_field", None)
    if base.base.base.canonical_sha256(original) != claimed:
        raise RuntimeError("V79 underlying V73 receipt identity changed")

    resolved = dict(resolved)
    if (
        resolved.get("gpu_memory_utilization")
        != TARGET_GPU_MEMORY_UTILIZATION_V79
        or resolved.get("cache_dtype") != KV_DTYPE_V79
        or resolved.get("mamba_cache_dtype") != "auto"
        or resolved.get("mamba_ssm_cache_dtype") != "float32"
        or resolved.get("kv_cache_size_tokens", 0)
        < MIN_KV_CAPACITY_TOKENS_V79
        or resolved.get("resolved_from_live_engine") is not True
    ):
        raise RuntimeError("V79 resolved runtime certificate changed")
    routed_audit = audit_base.validate_runtime_audit_v76(routed_audit)
    if routed_audit.get("parameter_residency") != PARAMETER_RESIDENCY_V79:
        raise RuntimeError("V79 live parameter-residency identity changed")
    if generation_performance.get("schema") != (
        "v79-data-free-generation-performance"
    ):
        raise RuntimeError("V79 generation performance certificate changed")

    result = original
    result["schema"] = SCHEMA_V79
    result["runtime"] = dict(result["runtime"])
    result["runtime"].update(
        {
            "gpu_memory_utilization": TARGET_GPU_MEMORY_UTILIZATION_V79,
            "kv_cache_dtype": KV_DTYPE_V79,
            "enable_flashinfer_autotune": False,
            "starting_moe_tuning_table": "fresh_empty_default",
        }
    )
    result["resolved_memory_budget_certificate"] = {
        "gpu_memory_utilization": TARGET_GPU_MEMORY_UTILIZATION_V79,
        "resolved_from_live_engine": True,
    }
    result["resolved_kv_cache_certificate"] = resolved
    result["routed_fp8_runtime_attestation"] = routed_audit
    result["generation_performance"] = generation_performance
    result["deepgemm_ordering_workaround"] = {
        "schema": "v76-explicit-deepgemm-disable-ordering-workaround",
        "VLLM_USE_DEEP_GEMM": "0",
        "quant_config_use_deep_gemm_before_post_init": False,
        "upstream_source_modified": False,
    }
    result["explicit_kernel_environment"] = {
        "VLLM_USE_DEEP_GEMM": "0",
        "enable_flashinfer_autotune": False,
        "tuned_config_folder_was_fresh_and_empty": True,
        "routed_moe_backend_from_worker_attestation": "TRITON",
    }
    result["single_variable_change_from_v78"] = {
        "gpu_memory_utilization": [
            REFERENCE_GPU_MEMORY_UTILIZATION_V78,
            TARGET_GPU_MEMORY_UTILIZATION_V79,
        ]
    }
    result["v78_reference_identity"] = {
        "probe_sha256": V78_PROBE_SHA256,
        "run_bundle_sha256": V78_RUN_BUNDLE_SHA256,
        "unchanged_runtime_fields_are_bound_by_v79_preregistration": True,
    }
    result["preflight_gates"] = dict(result["preflight_gates"])
    result["preflight_gates"].update(
        {
            "fp8_per_token_head_kv_resolved": True,
            "mamba_ssm_cache_remains_float32": True,
            "kv_capacity_minimum_tokens_passed": True,
            "kv_capacity_minimum_tokens": MIN_KV_CAPACITY_TOKENS_V79,
            "external_four_gpu_log_gate_pending": True,
            "external_four_gpu_cleanup_idle_gate_pending": True,
            "paired_output_gate_pending": True,
            "source_disjoint_semantic_gate_pending": True,
            "protected_ood_gate_pending": True,
            "scored_evaluation_or_training_authorized": False,
        }
    )
    result["authority"] = {
        "data_free_kv_capacity_and_switch_diagnostic_only": True,
        "scored_evaluation_training_checkpoint_or_promotion_allowed": False,
    }
    result["preregistration_v79"] = {
        "file_sha256": PREREG_FILE_SHA256,
        "content_sha256": PREREG_CONTENT_SHA256,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v79(result)
    return result


def publish_v79(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.v79-{os.getpid()}")
    with temporary.open("x", encoding="ascii") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    output = validate_argv_v79(sys.argv[1:])
    validate_static_v79()

    prior_environment = {
        name: os.environ.get(name)
        for name in (
            "VLLM_USE_DEEP_GEMM",
            "VLLM_ALLOW_INSECURE_SERIALIZATION",
            "VLLM_TUNED_CONFIG_FOLDER",
            "VLLM_BATCH_INVARIANT",
        )
    }
    os.environ["VLLM_USE_DEEP_GEMM"] = "0"
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    import vllm
    from vllm.config import VllmConfig

    original_llm = vllm.LLM
    original_get_quant = VllmConfig._get_quantization_config
    original_tuned = base.base.base.TUNED
    observed: list[dict] = []
    audits: list[dict] = []
    timing_rows: list[dict] = []

    def explicit_get_quantization(model_config, load_config):
        quant = original_get_quant(model_config, load_config)
        if (
            getattr(model_config, "quantization", None) == "fp8"
            and os.environ.get("VLLM_USE_DEEP_GEMM") == "0"
            and hasattr(quant, "use_deep_gemm")
        ):
            quant.use_deep_gemm = False
        return quant

    with tempfile.TemporaryDirectory(prefix="v79-empty-fp8-moe-") as folder:
        empty = Path(folder).resolve()

        def capacity_matched_llm(*args, **kwargs):
            if (
                os.environ.get("VLLM_TUNED_CONFIG_FOLDER") != str(empty)
                or os.environ.get("VLLM_USE_DEEP_GEMM") != "0"
                or any(empty.iterdir())
            ):
                raise RuntimeError("V79 clean kernel environment changed")
            engine = original_llm(*args, **rightsized_kwargs_v79(kwargs))
            observed.append(resolved_runtime_v79(engine))
            values = engine.apply_model(audit_base.audit_fp8_routed_runtime_v76)
            if not isinstance(values, list) or len(values) != 1:
                raise RuntimeError("V79 apply_model worker cardinality changed")
            audits.append(audit_base.validate_runtime_audit_v76(values[0]))
            original_generate = engine.generate

            def timed_generate(*generate_args, **generate_kwargs):
                started = time.monotonic()
                outputs = original_generate(*generate_args, **generate_kwargs)
                elapsed = time.monotonic() - started
                request = generate_kwargs.get("lora_request")
                name = str(getattr(request, "lora_name", ""))
                label = (
                    "reference" if "reference" in name
                    else "candidate" if "candidate" in name
                    else None
                )
                token_count = sum(
                    len(candidate.token_ids)
                    for output_row in outputs
                    for candidate in output_row.outputs
                )
                timing_rows.append(
                    {
                        "call_index": len(timing_rows),
                        "label": label,
                        "request_count": len(outputs),
                        "generated_token_count": token_count,
                        "elapsed_seconds": elapsed,
                    }
                )
                return outputs

            engine.generate = timed_generate
            return engine

        base.base.base.TUNED = empty
        VllmConfig._get_quantization_config = staticmethod(
            explicit_get_quantization
        )
        vllm.LLM = capacity_matched_llm
        try:
            status = base.main()
        finally:
            vllm.LLM = original_llm
            VllmConfig._get_quantization_config = original_get_quant
            base.base.base.TUNED = original_tuned
            for name, prior in prior_environment.items():
                if prior is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = prior

    if (
        status != 0
        or not output.is_file()
        or len(observed) != 1
        or len(audits) != 1
        or len(timing_rows) != 10
    ):
        raise RuntimeError("V79 underlying capacity-matched probe failed")
    performance = summarize_generation_timings_v79(timing_rows)
    upgraded = upgraded_receipt_v79(
        json.loads(output.read_text(encoding="utf-8")),
        observed[0],
        audits[0],
        performance,
    )
    publish_v79(output, upgraded)
    print(
        json.dumps(
            {
                "schema": SCHEMA_V79,
                "output": str(output),
                "content_sha256": upgraded[
                    "content_sha256_before_self_field"
                ],
                "gpu_memory_utilization": TARGET_GPU_MEMORY_UTILIZATION_V79,
                "kv_cache_size_tokens": observed[0][
                    "kv_cache_size_tokens"
                ],
                "preflight_gates": upgraded["preflight_gates"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
