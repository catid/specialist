#!/usr/bin/env python3
"""Deterministic CPU-only postrun analysis for Qwen3.6 V80 r1.

The analyzer opens only sealed JSON/log/telemetry commitments from the
data-free probes.  It imports neither torch nor vLLM, does not open a dataset,
and does not launch a GPU.  Any input identity or contract mismatch fails
closed before a result is emitted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_bf16_kv_mamba_capacity_v80_postrun_20260717.json"
)
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_bf16_kv_mamba_capacity_v80_live_evidence_20260717.md"
)
RUN_ROOT = ROOT / "experiments/eggroll_es_hpo/runs"
PREREG_ROOT = ROOT / "experiments/eggroll_es_hpo/preregistrations"

SCHEMA = "v80-qwen36-bf16-kv-mamba-capacity-postrun-analysis"
GPU_IDS = (0, 1, 2, 3)
CALL_PLAN = [
    "reference", "candidate", "candidate", "reference",
    "reference", "candidate", "candidate", "reference",
]
ROWS_PER_CALL = 68
TOKENS_PER_ROW = 64
MEMORY_TOTAL_MIB = 97_887

V80_RUN = "v80_bf16_kv_mamba_capacity_0479_r1"
V79B_RUN = "v79b_fp8_kv_capacity_0485_r5_sealed_cleanup"
V78C_RUN = "v78c_bf16_kv_bf16_mamba_ssm_r1"

PREREGISTRATIONS = {
    "qwen36_bf16_kv_mamba_capacity_matched_v80.json": {
        "schema": "v80-qwen36-bf16-kv-mamba-capacity-matched-preregistration",
        "file_sha256": "be7de6c8ac1d59feef0ec0d0e289be85612644efbe72fdb3847fa34d5dd50aad",
        "content_sha256": "7527ed6fe0154a79ecc0de46b00af4601b0e3deaac184f2af094fba15740149a",
    },
    "qwen36_bf16_kv_mamba_confirmation_v80b.json": {
        "schema": "v80b-qwen36-bf16-kv-mamba-two-run-confirmation-preregistration",
        "file_sha256": "9d4c85f594946dc9f200b064ce79ee989cd218a17879c6786da85405d35d6724",
        "content_sha256": "8515a4680175a68233f2ff408bd0439daebf1a8a4c94a842c45e08ac9ec6b976",
    },
    "qwen36_fp8_kv_capacity_cleanup_v79b.json": {
        "schema": "v79b-qwen36-fp8-kv-cleanup-preregistration",
        "file_sha256": "8e1940db5134bb77ef9959d10b4eec5d43fab4e8653d62733b42939b5fd7300f",
        "content_sha256": "7669c2f720f2a0d17e976de42cc5b7c08fba60a3251175a62eddf05de2dc1b5d",
    },
}

SOURCE_SHA256 = {
    "probe_vllm_quantized_adapter_switch_v73.py": "43661c32cd8d06deef6d8e2f0d83d889b00f554748b94c3345e2b2052cac66a9",
    "probe_vllm_fp8_attested_v76.py": "a23d43ee5b6b334fdc58b93e0ce7e7d3fcf72ea4047549f3a0f4d5b715a3fc70",
    "probe_vllm_bf16_kv_mamba_bf16_v78c.py": "761857944064a0b21ff528971d3f497e4e67865679fa51a30d385cab65835dcb",
    "probe_vllm_bf16_kv_mamba_capacity_v80.py": "3679bfb1d7f1995701b8b96c100f41ac05fc8f3338977e5760f52aa1ef8009fd",
    "monitor_qwen36_fp8_kv_capacity_v79.py": "6035eb32f90815ed2a2d8734d9e9072123b8ecc70d74449a2710e94b673ed3df",
    "launch_qwen36_bf16_kv_mamba_capacity_v80.sh": "3a335c0e44a9b8130adec8c0e11f52d2246264aa0a47f5b173cbdb35093fc7ad",
}

RUN_BUNDLES = {
    V80_RUN: "73adc7ebe416d6065808cf918415d077989b1a064b47ef5132422b32da118e47",
    V79B_RUN: "17a4f99164cd17f8e55ddcd879460c99e356e34d8e90e017abbbd7597085dde7",
    V78C_RUN: "a9d82f71bb6beecc420be135737f2048fb770bee5d97309f9e90da4e31ef833f",
}
RUN_FILE_COUNTS = {V80_RUN: 11, V79B_RUN: 11, V78C_RUN: 9}
CAPACITY_TOKENS = {V80_RUN: 162_669, V79B_RUN: 162_304, V78C_RUN: 218_843}

OVERRIDE_WARNING = (
    "Qwen3.5 model specifies mamba_ssm_dtype='float32' in its config, but "
    "--mamba-ssm-cache-dtype='bfloat16' was passed. Using the user-specified value."
)
FORBIDDEN_LOG_FRAGMENTS = (
    "Auto-disabled DeepGemm",
    "DeepGEMM E8M0 enabled",
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "falling back",
    "fallback to",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256_v80(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256_v80(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _validate_self_hash(value: Mapping[str, Any], label: str) -> None:
    body = copy.deepcopy(dict(value))
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and claimed == canonical_sha256_v80(body),
        f"self hash changed: {label}",
    )


def _inventory(run: str) -> dict[str, Any]:
    directory = RUN_ROOT / run
    _require(
        directory.is_dir() and not directory.is_symlink(), f"run missing: {run}"
    )
    paths = sorted(directory.iterdir())
    _require(
        len(paths) == RUN_FILE_COUNTS[run]
        and all(path.is_file() and not path.is_symlink() for path in paths),
        f"run artifact cardinality/type changed: {run}",
    )
    rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256_v80(path),
        }
        for path in paths
    ]
    bundle = canonical_sha256_v80(rows)
    _require(bundle == RUN_BUNDLES[run], f"sealed run bundle changed: {run}")
    return {"file_count": len(rows), "bundle_sha256": bundle, "files": rows}


def validate_static_contract_v80() -> dict[str, Any]:
    preregs = {}
    for filename, expected in PREREGISTRATIONS.items():
        path = PREREG_ROOT / filename
        _require(
            path.is_file()
            and not path.is_symlink()
            and file_sha256_v80(path) == expected["file_sha256"],
            f"sealed preregistration file changed: {filename}",
        )
        value = _load_json(path)
        _validate_self_hash(value, filename)
        _require(
            value.get("schema") == expected["schema"]
            and value.get("content_sha256_before_self_field")
            == expected["content_sha256"],
            f"sealed preregistration content changed: {filename}",
        )
        preregs[filename] = dict(expected)

    sources = []
    for relative, expected in SOURCE_SHA256.items():
        path = ROOT / relative
        _require(
            path.is_file()
            and not path.is_symlink()
            and file_sha256_v80(path) == expected,
            f"sealed executable source changed: {relative}",
        )
        sources.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": expected}
        )

    parent = _load_json(
        PREREG_ROOT / "qwen36_bf16_kv_mamba_capacity_matched_v80.json"
    )
    v80b = _load_json(
        PREREG_ROOT / "qwen36_bf16_kv_mamba_confirmation_v80b.json"
    )
    _require(
        v80b.get("parent_v80", {}).get("selected_runtime_retained_exactly")
        == parent.get("selected_runtime")
        and v80b.get("parent_v80", {}).get(
            "live_acceptance_retained_exactly"
        )
        == parent.get("live_acceptance")
        and v80b.get("prospective_integrity", {}).get(
            "r1_was_observed_before_this_preregistration"
        )
        is True
        and v80b.get("prospective_integrity", {}).get(
            "threshold_tuning_after_r1_forbidden"
        )
        is True,
        "V80B did not retain/disclose the V80 parent contract exactly",
    )
    _require(
        v80b.get("sealed_executable_sources", {}).get("files") == sources,
        "V80B source inventory changed",
    )
    return {
        "preregistrations": preregs,
        "sealed_executable_sources": {
            "files": sources,
            "bundle_sha256": canonical_sha256_v80(sources),
        },
        "parent_selected_runtime_canonical_sha256": canonical_sha256_v80(
            parent["selected_runtime"]
        ),
        "parent_live_acceptance_canonical_sha256": canonical_sha256_v80(
            parent["live_acceptance"]
        ),
    }


def _validate_hash_rows(receipt: Mapping[str, Any], label: str) -> None:
    calls = receipt.get("calls")
    _require(
        isinstance(calls, list)
        and len(calls) == len(CALL_PLAN)
        and receipt.get("call_plan") == CALL_PLAN,
        f"call plan changed: {label}",
    )
    for call_index, (call, state) in enumerate(zip(calls, CALL_PLAN)):
        rows = call.get("rows") if isinstance(call, dict) else None
        _require(
            call.get("call_index") == call_index
            and call.get("label") == state
            and isinstance(rows, list)
            and len(rows) == ROWS_PER_CALL,
            f"call identity/cardinality changed: {label}:{call_index}",
        )
        for row in rows:
            _require(
                isinstance(row, dict)
                and set(row) == {"token_count", "token_ids_sha256"}
                and row.get("token_count") == TOKENS_PER_ROW
                and re.fullmatch(r"[0-9a-f]{64}", row["token_ids_sha256"])
                is not None,
                f"hash-only row contract changed: {label}:{call_index}",
            )
        _require(
            call.get("rows_sha256") == canonical_sha256_v80(rows),
            f"call row hash changed: {label}:{call_index}",
        )

    by_state = {
        state: [call for call in calls if call["label"] == state]
        for state in ("reference", "candidate")
    }
    changed = {
        state: sum(
            len({call["rows"][row]["token_ids_sha256"] for call in state_calls})
            > 1
            for row in range(ROWS_PER_CALL)
        )
        for state, state_calls in by_state.items()
    }
    between = sum(
        by_state["reference"][0]["rows"][row]["token_ids_sha256"]
        != by_state["candidate"][0]["rows"][row]["token_ids_sha256"]
        for row in range(ROWS_PER_CALL)
    )
    _require(
        receipt.get("reference_within_state_changed_rows") == changed["reference"]
        and receipt.get("candidate_within_state_changed_rows")
        == changed["candidate"]
        and receipt.get("between_state_differing_rows") == between,
        f"derived output counters changed: {label}",
    )


def _validate_generation_performance(
    value: Mapping[str, Any], schema: str, label: str
) -> dict[str, Any]:
    calls = value.get("calls")
    _require(
        value.get("schema") == schema
        and isinstance(calls, list)
        and len(calls) == 10
        and value.get("warmup_call_count") == 2
        and value.get("measured_call_count") == 8,
        f"generation timing cardinality changed: {label}",
    )
    _require(
        [row.get("call_index") for row in calls] == list(range(10))
        and [row.get("label") for row in calls[2:]] == CALL_PLAN,
        f"generation timing call plan changed: {label}",
    )
    for row in calls:
        elapsed = row.get("elapsed_seconds")
        _require(
            row.get("request_count") == ROWS_PER_CALL
            and row.get("generated_token_count") == ROWS_PER_CALL * TOKENS_PER_ROW
            and isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool)
            and math.isfinite(elapsed)
            and elapsed > 0,
            f"generation timing row changed: {label}",
        )
    measured = [row["elapsed_seconds"] for row in calls[2:]]
    elapsed_sum = sum(measured)
    token_count = len(measured) * ROWS_PER_CALL * TOKENS_PER_ROW
    p95 = sorted(measured)[math.ceil(0.95 * len(measured)) - 1]
    _require(
        value.get("measured_generated_token_count") == token_count
        and math.isclose(
            value.get("measured_generation_seconds_sum"),
            elapsed_sum,
            rel_tol=0,
            abs_tol=1e-12,
        )
        and math.isclose(
            value.get("aggregate_generated_tokens_per_second"),
            token_count / elapsed_sum,
            rel_tol=0,
            abs_tol=1e-9,
        )
        and math.isclose(
            value.get("median_call_latency_seconds"),
            statistics.median(measured),
            rel_tol=0,
            abs_tol=1e-12,
        )
        and math.isclose(
            value.get("p95_call_latency_seconds_nearest_rank"),
            p95,
            rel_tol=0,
            abs_tol=1e-12,
        )
        and math.isclose(
            value.get("max_call_latency_seconds"),
            max(measured),
            rel_tol=0,
            abs_tol=1e-12,
        ),
        f"generation timing summary changed: {label}",
    )
    return {
        "aggregate_generated_tokens_per_second": value[
            "aggregate_generated_tokens_per_second"
        ],
        "median_call_latency_seconds": value["median_call_latency_seconds"],
        "p95_call_latency_seconds_nearest_rank": value[
            "p95_call_latency_seconds_nearest_rank"
        ],
        "max_call_latency_seconds": value["max_call_latency_seconds"],
    }


def _validate_residency(receipt: Mapping[str, Any], label: str) -> dict[str, Any]:
    audit = receipt.get("routed_fp8_runtime_attestation", {})
    residency = audit.get("parameter_residency", {})
    language = residency.get("components", {}).get("language", {})
    records = audit.get("fp8_moe_records")
    _require(
        audit.get("fp8_moe_method_count") == 40
        and audit.get("fp8_quant_reference_count") == 80
        and audit.get("routed_like_module_count") == 40
        and audit.get("routed_like_without_fp8_method") == []
        and isinstance(records, list)
        and len(records) == 40
        and all(
            row.get("fp8_backend_name") == "TRITON"
            and row.get("experts_implementation_class") == "TritonExperts"
            and row.get("runtime_quant_wrapper_class") == "FusedMoEModularMethod"
            and row.get("weight_block_size") == [128, 128]
            and row.get("w13_dtype") == "torch.float8_e4m3fn"
            and row.get("w2_dtype") == "torch.float8_e4m3fn"
            for row in records
        )
        and residency.get("total_parameter_count") == 813
        and residency.get("total_logical_bytes") == 35_712_084_096
        and set(residency.get("components", {})) == {"language"}
        and language.get("parameter_names_sha256")
        == "a850f55c3f02ef904041d48b29f13af2d29834da200f92dcc9728760cb185b90",
        f"routed backend or residency changed: {label}",
    )
    return residency


def _validate_actor(run: str, gpu: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = RUN_ROOT / run / f"gpu_{gpu}.json"
    receipt = _load_json(path)
    label = f"{run}:gpu{gpu}"
    _validate_self_hash(receipt, label)
    _validate_hash_rows(receipt, label)
    _require(
        receipt.get("actor_label") == f"gpu-{gpu}"
        and receipt.get("source_dataset_rows_opened") == 0
        and receipt.get("protected_ood_shadow_or_terminal_opened") is False
        and receipt.get("adapter_update_or_hpo_performed") is False
        and receipt.get("prompt_or_generation_text_persisted") is False
        and receipt.get("token_ids_persisted") is False
        and receipt.get("engine_shutdown_completed") is True,
        f"actor authority/cleanup changed: {label}",
    )
    runtime = receipt.get("runtime", {})
    _require(
        runtime.get("resolved_quantization") == "fp8"
        and runtime.get("serialized_fp8_checkpoint") is True
        and runtime.get("starting_moe_tuning_table") == "fresh_empty_default"
        and runtime.get("vllm_version") == "0.25.0"
        and runtime.get("enforce_eager") is True
        and runtime.get("async_scheduling") is False
        and runtime.get("scheduling_policy") == "fcfs"
        and runtime.get("max_num_seqs") == ROWS_PER_CALL
        and runtime.get("max_loras") == 1
        and runtime.get("max_cpu_loras") == 2
        and runtime.get("enable_flashinfer_autotune") is False,
        f"common runtime changed: {label}",
    )
    precision = receipt.get("resolved_precision_certificate", {})
    _require(
        precision.get("model_quantization") == "fp8"
        and precision.get("quant_config_class") == "Fp8Config"
        and precision.get("weight_block_size") == [128, 128],
        f"precision certificate changed: {label}",
    )
    residency = _validate_residency(receipt, label)

    performance = None
    if run == V80_RUN:
        cache = receipt.get("resolved_hybrid_cache_certificate", {})
        _require(
            receipt.get("schema")
            == "v80-qwen36-bf16-kv-mamba-capacity-matched-preflight"
            and runtime.get("gpu_memory_utilization") == 0.479
            and runtime.get("mamba_ssm_cache_dtype") == "bfloat16"
            and cache.get("resolved_from_live_engine") is True
            and cache.get("gpu_memory_utilization") == 0.479
            and cache.get("cache_dtype") == "auto"
            and cache.get("mamba_cache_dtype") == "auto"
            and cache.get("mamba_ssm_cache_dtype") == "bfloat16"
            and cache.get("model_quantization") == "fp8"
            and cache.get("block_size") == 544
            and cache.get("num_gpu_blocks") == 556
            and cache.get("kv_cache_size_tokens") == CAPACITY_TOKENS[run]
            and math.isclose(
                cache.get("kv_cache_max_concurrency"),
                79.42857142857143,
                rel_tol=0,
                abs_tol=1e-12,
            )
            and receipt.get("preregistration_v80")
            == {
                "content_sha256": PREREGISTRATIONS[
                    "qwen36_bf16_kv_mamba_capacity_matched_v80.json"
                ]["content_sha256"],
                "file_sha256": PREREGISTRATIONS[
                    "qwen36_bf16_kv_mamba_capacity_matched_v80.json"
                ]["file_sha256"],
            }
            and receipt.get("single_variable_change_from_v78c")
            == {"gpu_memory_utilization": [0.5, 0.479]}
            and receipt.get("v78c_reference_identity", {}).get(
                "run_bundle_sha256"
            )
            == RUN_BUNDLES[V78C_RUN],
            f"V80 live cache/ancestry changed: {label}",
        )
        gates = receipt.get("preflight_gates", {})
        _require(
            gates.get("candidate_repeat_exact_at_token_hash_level") is True
            and gates.get("candidate_changes_output") is True
            and gates.get("reference_restore_exact_at_token_hash_level") is False
            and gates.get("kv_capacity_minimum_tokens_passed") is True
            and gates.get("bf16_attention_kv_auto_resolved") is True
            and gates.get("bf16_mamba_ssm_cache_resolved") is True
            and gates.get("scored_evaluation_or_training_authorized") is False
            and receipt.get("authority", {}).get(
                "scored_evaluation_training_checkpoint_or_promotion_allowed"
            )
            is False,
            f"V80 actor gate/authority changed: {label}",
        )
        performance = _validate_generation_performance(
            receipt.get("generation_performance", {}),
            "v80-data-free-generation-performance",
            label,
        )
    elif run == V79B_RUN:
        cache = receipt.get("resolved_kv_cache_certificate", {})
        _require(
            receipt.get("schema")
            == "v79-qwen36-fp8-kv-capacity-matched-preflight"
            and runtime.get("gpu_memory_utilization") == 0.485
            and runtime.get("kv_cache_dtype") == "fp8_per_token_head"
            and cache.get("gpu_memory_utilization") == 0.485
            and cache.get("cache_dtype") == "fp8_per_token_head"
            and cache.get("mamba_ssm_cache_dtype") == "float32"
            and cache.get("kv_cache_size_tokens") == CAPACITY_TOKENS[run]
            and cache.get("block_size") == 2048
            and cache.get("kv_cache_max_concurrency") == 79.25,
            f"V79B control changed: {label}",
        )
        performance = _validate_generation_performance(
            receipt.get("generation_performance", {}),
            "v79-data-free-generation-performance",
            label,
        )
    else:
        cache = receipt.get("resolved_hybrid_cache_certificate", {})
        _require(
            receipt.get("schema")
            == "v78c-qwen36-bf16-kv-bf16-mamba-ssm-preflight"
            and runtime.get("gpu_memory_utilization") == 0.5
            and runtime.get("mamba_ssm_cache_dtype") == "bfloat16"
            and cache
            == {
                "cache_dtype": "auto",
                "mamba_ssm_cache_dtype": "bfloat16",
                "resolved_from_live_engine": True,
            },
            f"V78c control changed: {label}",
        )

    wall = receipt.get("wall_runtime_seconds_excluding_model_load_and_cleanup")
    _require(
        isinstance(wall, (int, float))
        and not isinstance(wall, bool)
        and math.isfinite(wall)
        and wall > 0,
        f"actor wall runtime changed: {label}",
    )
    summary = {
        "gpu": gpu,
        "receipt_content_sha256": receipt["content_sha256_before_self_field"],
        "wall_runtime_seconds": wall,
        "reference_repeat_changed_rows": receipt[
            "reference_within_state_changed_rows"
        ],
        "candidate_repeat_changed_rows": receipt[
            "candidate_within_state_changed_rows"
        ],
        "between_state_differing_rows": receipt["between_state_differing_rows"],
        "engine_shutdown_completed": receipt["engine_shutdown_completed"],
        "torch_process_group_destroyed_receipt_value": receipt.get(
            "torch_process_group_destroyed"
        ),
        "generation_performance": performance,
    }
    return summary, receipt


def _validate_log(run: str, gpu: int) -> dict[str, Any]:
    path = RUN_ROOT / run / f"gpu_{gpu}.log"
    text = path.read_text(encoding="utf-8", errors="strict")
    backend = "FLASH_ATTN" if run in (V80_RUN, V78C_RUN) else "TRITON_ATTN"
    required = [
        "Using TRITON Fp8 MoE backend",
        f"Using {backend} attention backend",
        "Available KV cache memory:",
        "GPU KV cache size:",
        "Maximum concurrency for 2,048 tokens per request:",
        "Skipping FlashInfer autotune because it is disabled",
    ]
    if run in (V80_RUN, V78C_RUN):
        required.append(OVERRIDE_WARNING)
    _require(
        all(text.count(fragment) == 1 for fragment in required),
        f"required log cardinality changed: {run}:gpu{gpu}",
    )
    _require(
        all(fragment.lower() not in text.lower() for fragment in FORBIDDEN_LOG_FRAGMENTS),
        f"forbidden fallback/error in log: {run}:gpu{gpu}",
    )
    tokens = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    concurrency = re.findall(
        r"Maximum concurrency for 2,048 tokens per request: ([\d.]+)x", text
    )
    available = re.findall(r"Available KV cache memory: ([\d.]+) GiB", text)
    _require(
        len(tokens) == len(concurrency) == len(available) == 1
        and int(tokens[0].replace(",", "")) == CAPACITY_TOKENS[run],
        f"capacity log changed: {run}:gpu{gpu}",
    )
    return {
        "file_sha256": file_sha256_v80(path),
        "attention_backend": backend,
        "available_kv_gib": float(available[0]),
        "kv_cache_size_tokens": int(tokens[0].replace(",", "")),
        "maximum_concurrency": float(concurrency[0]),
        "required_fragment_counts": {
            fragment: text.count(fragment) for fragment in required
        },
        "forbidden_fragment_counts": {
            fragment: text.lower().count(fragment.lower())
            for fragment in FORBIDDEN_LOG_FRAGMENTS
        },
    }


def _read_pid_map(run: str) -> dict[int, int]:
    path = RUN_ROOT / run / "actor_pids.csv"
    lines = path.read_text(encoding="ascii").splitlines()
    _require(len(lines) == 4, f"PID map cardinality changed: {run}")
    result = {}
    for line in lines:
        fields = line.split(",")
        _require(len(fields) == 2, f"PID map row changed: {run}")
        gpu, pid = map(int, fields)
        _require(
            gpu in GPU_IDS and pid > 1 and gpu not in result,
            f"PID/GPU binding changed: {run}",
        )
        result[gpu] = pid
    _require(
        tuple(sorted(result)) == GPU_IDS and len(set(result.values())) == 4,
        f"PID map uniqueness changed: {run}",
    )
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        _require(bool(line), f"blank JSONL row: {path}:{line_number}")
        row = json.loads(line)
        _require(isinstance(row, dict), f"JSONL object required: {path}:{line_number}")
        rows.append(row)
    _require(bool(rows), f"empty telemetry: {path}")
    return rows


def _validate_new_telemetry(run: str) -> dict[str, Any]:
    path = RUN_ROOT / run / (
        "gpu_telemetry_v80.jsonl" if run == V80_RUN else "gpu_telemetry_v79.jsonl"
    )
    pid_map = _read_pid_map(run)
    rows = _load_jsonl(path)
    _require(
        [row.get("sequence") for row in rows] == list(range(len(rows))),
        f"telemetry sequence changed: {run}",
    )
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gpu = row.get("gpu")
        _require(
            row.get("schema") == "v79-four-gpu-kv-capacity-telemetry"
            and gpu in GPU_IDS
            and row.get("actor_root_pid") == pid_map[gpu]
            and row.get("memory_total_mib") == MEMORY_TOTAL_MIB
            and row.get("pcie_counters_supported") is True
            and isinstance(row.get("pcie_rx_kib_per_second"), int)
            and row["pcie_rx_kib_per_second"] >= 0
            and isinstance(row.get("pcie_tx_kib_per_second"), int)
            and row["pcie_tx_kib_per_second"] >= 0
            and row.get("hbm_bytes_per_second_inferred") is False
            and row.get("foreign_compute_pids") == [],
            f"telemetry row contract changed: {run}",
        )
        by_batch[row["batch_index"]].append(row)
        by_gpu[gpu].append(row)
    _require(
        sorted(by_batch) == list(range(len(by_batch))),
        f"telemetry batch sequence changed: {run}",
    )
    monotonic = []
    for batch_index, batch in sorted(by_batch.items()):
        _require(
            len(batch) == 4
            and sorted(row["gpu"] for row in batch) == list(GPU_IDS)
            and len({row["monotonic_ns"] for row in batch}) == 1
            and len({row["sampled_at_utc"] for row in batch}) == 1,
            f"incomplete four-GPU telemetry batch: {run}:{batch_index}",
        )
        monotonic.append(batch[0]["monotonic_ns"])
    intervals = [
        (right - left) / 1e9 for left, right in zip(monotonic, monotonic[1:])
    ]
    _require(
        bool(intervals)
        and all(math.isfinite(value) and 0 < value <= 1.0 for value in intervals),
        f"telemetry interval changed: {run}",
    )

    per_gpu = {}
    for gpu in GPU_IDS:
        gpu_rows = by_gpu[gpu]
        attributed = {
            pid
            for row in gpu_rows
            for pid in row.get("attributed_compute_pids", [])
        }
        ancestry = {
            pid
            for row in gpu_rows
            for pid in row.get("ancestry_attributed_compute_pids", [])
        }
        uuids = {row["gpu_uuid"] for row in gpu_rows}
        _require(
            len(attributed) == 1
            and attributed == ancestry
            and len(uuids) == 1
            and any(row["gpu_utilization_percent"] > 0 for row in gpu_rows)
            and any(row["memory_utilization_percent"] > 0 for row in gpu_rows),
            f"GPU not uniquely attributed/useful: {run}:gpu{gpu}",
        )
        peak = max(row["memory_used_mib"] for row in gpu_rows)
        process_peak = max(
            sum(row.get("process_memory_mib", {}).values()) for row in gpu_rows
        )
        per_gpu[str(gpu)] = {
            "actor_root_pid": pid_map[gpu],
            "attributed_compute_pids": sorted(attributed),
            "gpu_uuid": next(iter(uuids)),
            "sample_count": len(gpu_rows),
            "positive_gpu_utilization_samples": sum(
                row["gpu_utilization_percent"] > 0 for row in gpu_rows
            ),
            "positive_hbm_utilization_samples": sum(
                row["memory_utilization_percent"] > 0 for row in gpu_rows
            ),
            "peak_gpu_utilization_percent": max(
                row["gpu_utilization_percent"] for row in gpu_rows
            ),
            "peak_hbm_utilization_percent": max(
                row["memory_utilization_percent"] for row in gpu_rows
            ),
            "peak_memory_used_mib": peak,
            "peak_attributed_process_memory_mib": process_peak,
            "minimum_physical_headroom_mib": MEMORY_TOTAL_MIB - peak,
            "peak_power_draw_mw": max(row["power_draw_mw"] for row in gpu_rows),
        }

    rx_bytes = 0.0
    tx_bytes = 0.0
    for left_index, dt in enumerate(intervals):
        for row in by_batch[left_index]:
            rx_bytes += row["pcie_rx_kib_per_second"] * 1024.0 * dt
            tx_bytes += row["pcie_tx_kib_per_second"] * 1024.0 * dt

    cleanup_batches = []
    for batch_index, batch in sorted(by_batch.items()):
        if all(
            row.get("actor_root_alive") is False
            and row.get("compute_pids") == []
            and row.get("foreign_compute_pids") == []
            and row.get("cleanup_memory_gate_uses_external_nvidia_smi") is True
            and row.get("cleanup_nvidia_smi_gpu_utilization_percent") == 0
            and isinstance(row.get("cleanup_nvidia_smi_memory_used_mib"), int)
            and row["cleanup_nvidia_smi_memory_used_mib"] <= 4
            for row in batch
        ):
            cleanup_batches.append(batch_index)
    trailing = []
    for batch_index in reversed(sorted(by_batch)):
        if batch_index in cleanup_batches:
            trailing.append(batch_index)
        else:
            break
    trailing.reverse()
    _require(
        len(trailing) >= 3
        and trailing[-3:] == list(range(trailing[-1] - 2, trailing[-1] + 1)),
        f"external cleanup gate failed: {run}",
    )
    return {
        "file_sha256": file_sha256_v80(path),
        "row_count": len(rows),
        "complete_four_gpu_batches": len(by_batch),
        "sample_interval_seconds": {
            "minimum": min(intervals),
            "median": statistics.median(intervals),
            "maximum": max(intervals),
        },
        "per_gpu": per_gpu,
        "all_four_gpus_useful": True,
        "foreign_compute_pid_rows": 0,
        "sampled_pcie_integrals": {
            "claim": (
                "left-rectangle estimate across adjacent telemetry batches; "
                "not exact bytes and not HBM-bandwidth measurement"
            ),
            "rx_bytes_sum_gpus": round(rx_bytes),
            "tx_bytes_sum_gpus": round(tx_bytes),
        },
        "hbm_bandwidth_bytes_per_second": None,
        "hbm_bandwidth_reason": (
            "NVML memory-utilization percentages do not identify transferred bytes"
        ),
        "external_cleanup_accepted_batches": cleanup_batches,
        "trailing_external_cleanup_batches": trailing,
        "external_cleanup_gate_passed": True,
    }


def _validate_old_telemetry_v78c() -> dict[str, Any]:
    path = RUN_ROOT / V78C_RUN / "nvidia_smi_samples.log"
    lines = path.read_text(encoding="ascii").splitlines()
    _require(lines and len(lines) % 5 == 0, "V78c telemetry framing changed")
    per_gpu = {gpu: [] for gpu in GPU_IDS}
    for offset in range(0, len(lines), 5):
        _require(lines[offset].isdigit(), "V78c telemetry timestamp changed")
        seen = []
        for line in lines[offset + 1 : offset + 5]:
            fields = [field.strip() for field in line.split(",")]
            _require(len(fields) == 4, "V78c telemetry row changed")
            gpu, utilization, memory = map(int, fields[:3])
            power = float(fields[3])
            _require(
                gpu in GPU_IDS
                and 0 <= utilization <= 100
                and memory >= 0
                and power >= 0,
                "V78c telemetry value changed",
            )
            seen.append(gpu)
            per_gpu[gpu].append((utilization, memory, power))
        _require(sorted(seen) == list(GPU_IDS), "V78c telemetry GPU batch changed")
    _require(
        all(max(row[0] for row in rows) > 0 for rows in per_gpu.values()),
        "V78c did not use all GPUs",
    )
    return {
        "file_sha256": file_sha256_v80(path),
        "complete_four_gpu_batches": len(lines) // 5,
        "per_gpu": {
            str(gpu): {
                "peak_gpu_utilization_percent": max(row[0] for row in rows),
                "peak_memory_used_mib": max(row[1] for row in rows),
                "peak_power_draw_w": max(row[2] for row in rows),
            }
            for gpu, rows in per_gpu.items()
        },
    }


def _validate_run(run: str) -> dict[str, Any]:
    inventory = _inventory(run)
    actors = []
    receipts = []
    logs = []
    for gpu in GPU_IDS:
        actor, receipt = _validate_actor(run, gpu)
        actors.append(actor)
        receipts.append(receipt)
        logs.append(_validate_log(run, gpu))
    telemetry = (
        _validate_old_telemetry_v78c()
        if run == V78C_RUN
        else _validate_new_telemetry(run)
    )
    walls = [actor["wall_runtime_seconds"] for actor in actors]
    return {
        "run": run,
        "artifact_inventory": inventory,
        "actors": actors,
        "actor_wall_runtime_seconds_median": statistics.median(walls),
        "actor_wall_runtime_seconds_range": [min(walls), max(walls)],
        "logs": logs,
        "telemetry": telemetry,
        "receipts": receipts,
    }


def _different_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    _require(
        len(left["rows"]) == len(right["rows"]) == ROWS_PER_CALL,
        "paired drift row cardinality changed",
    )
    return sum(lrow != rrow for lrow, rrow in zip(left["rows"], right["rows"]))


def paired_drift_matrix_v80(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    _require(len(left) == len(right) == 4, f"paired actor count changed: {label}")
    by_gpu = {}
    by_call = [0] * len(CALL_PLAN)
    for gpu in GPU_IDS:
        _require(
            left[gpu].get("actor_label")
            == right[gpu].get("actor_label")
            == f"gpu-{gpu}",
            f"paired GPU identity changed: {label}",
        )
        values = []
        for call_index in range(len(CALL_PLAN)):
            differing = _different_rows(
                left[gpu]["calls"][call_index], right[gpu]["calls"][call_index]
            )
            values.append(differing)
            by_call[call_index] += differing
        by_gpu[str(gpu)] = values
    differing_total = sum(by_call)
    compared = len(GPU_IDS) * len(CALL_PLAN) * ROWS_PER_CALL
    return {
        "pair": label,
        "call_plan": CALL_PLAN,
        "differing_rows_by_gpu_and_call": by_gpu,
        "differing_rows_by_call_sum_gpus": by_call,
        "differing_rows_total": differing_total,
        "compared_rows_total": compared,
        "agreement_fraction": (compared - differing_total) / compared,
        "claim": (
            "token-hash commitment disagreement, not semantic quality; known "
            "reference-repeat nondeterminism remains visible"
        ),
    }


def _strip_receipts(run: dict[str, Any]) -> dict[str, Any]:
    result = dict(run)
    result.pop("receipts", None)
    return result


def analyze_finalized_v80() -> dict[str, Any]:
    static = validate_static_contract_v80()
    v80 = _validate_run(V80_RUN)
    v79b = _validate_run(V79B_RUN)
    v78c = _validate_run(V78C_RUN)

    v80_median = v80["actor_wall_runtime_seconds_median"]
    v79b_median = v79b["actor_wall_runtime_seconds_median"]
    v78c_median = v78c["actor_wall_runtime_seconds_median"]
    v80_telemetry = v80["telemetry"]
    v80_peak = max(
        row["peak_memory_used_mib"] for row in v80_telemetry["per_gpu"].values()
    )
    v80_process_peak = max(
        row["peak_attributed_process_memory_mib"]
        for row in v80_telemetry["per_gpu"].values()
    )
    v79b_peak = max(
        row["peak_memory_used_mib"] for row in v79b["telemetry"]["per_gpu"].values()
    )
    v78c_peak = max(
        row["peak_memory_used_mib"] for row in v78c["telemetry"]["per_gpu"].values()
    )

    drift = {
        "v78c_r1_vs_v80_r1": paired_drift_matrix_v80(
            v78c["receipts"], v80["receipts"], "v78c_r1_vs_v80_r1"
        ),
        "v79b_r5_vs_v80_r1": paired_drift_matrix_v80(
            v79b["receipts"], v80["receipts"], "v79b_r5_vs_v80_r1"
        ),
    }

    actor_shutdown = all(
        actor["engine_shutdown_completed"] is True for actor in v80["actors"]
    )
    actor_pg_literal = all(
        actor["torch_process_group_destroyed_receipt_value"] is True
        for actor in v80["actors"]
    )
    capacity_pass = all(
        log["kv_cache_size_tokens"] >= 161_792 for log in v80["logs"]
    )
    runtime_pass = v80_median <= 50.55591316620558 and v80_median / v78c_median <= 1.03
    memory_pass = all(
        row["peak_memory_used_mib"] <= 50_808
        and row["minimum_physical_headroom_mib"] >= 47_079
        for row in v80_telemetry["per_gpu"].values()
    )
    output_pass = all(
        actor["candidate_repeat_changed_rows"] == 0
        and actor["between_state_differing_rows"] > 0
        for actor in v80["actors"]
    )
    gate_results = {
        "cardinality_identity_hash_only_and_no_data_access": True,
        "capacity_live_field_log_and_minimum": capacity_pass,
        "hybrid_cache_precision_backend_residency_and_warning": True,
        "required_and_forbidden_logs": True,
        "runtime_and_memory": runtime_pass and memory_pass,
        "four_gpu_activity_and_pcie_telemetry": (
            v80_telemetry["all_four_gpus_useful"]
            and v80_telemetry["sampled_pcie_integrals"]["rx_bytes_sum_gpus"] >= 0
            and v80_telemetry["sampled_pcie_integrals"]["tx_bytes_sum_gpus"] >= 0
        ),
        "candidate_output_and_paired_hash_matrix": output_pass,
        "engine_shutdown_completed_per_actor": actor_shutdown,
        "external_three_batch_post_exit_cleanup": v80_telemetry[
            "external_cleanup_gate_passed"
        ],
        "torch_process_group_destroyed_per_actor_literal_true": actor_pg_literal,
    }
    _require(
        all(
            passed
            for gate, passed in gate_results.items()
            if gate != "torch_process_group_destroyed_per_actor_literal_true"
        ),
        "unexpected V80 parent data-free gate failure",
    )
    _require(
        actor_pg_literal is False
        and [
            actor["torch_process_group_destroyed_receipt_value"]
            for actor in v80["actors"]
        ]
        == [False] * 4,
        "expected literal Torch process-group mismatch changed",
    )

    v80b = _load_json(
        PREREG_ROOT / "qwen36_bf16_kv_mamba_confirmation_v80b.json"
    )
    exact_commands = [
        row["exact_command"] for row in v80b["confirmatory_runs"]
    ]
    result = {
        "schema": SCHEMA,
        "status": (
            "analysis_completed_literal_parent_torch_process_group_gate_failed_"
            "semantic_ood_pending_no_promotion"
        ),
        "authority": {
            "cpu_sealed_file_analysis_only": True,
            "dataset_prompt_generated_text_or_protected_data_opened": False,
            "gpu_or_model_launch_performed_by_analyzer": False,
            "model_adapter_training_or_checkpoint_update_performed": False,
            "scored_training_checkpoint_or_runtime_promotion_authorized": False,
        },
        "static_contract": static,
        "evidence_classification": {
            "v80_r1": (
                "observed_before_v80b; immutable data-free diagnostic; not "
                "eligible for threshold tuning or promotion"
            ),
            "v79b_r5": "sealed FP8-attention-KV cleanup-accepted control",
            "v78c_r1": "sealed BF16-hybrid-cache 0.500 parent control",
            "v80b_r2_r3": (
                "prospective exact-source exact-runtime confirmation runs; "
                "not launched by this analyzer"
            ),
        },
        "v80_r1": _strip_receipts(v80),
        "controls": {
            "v79b_r5": _strip_receipts(v79b),
            "v78c_r1": _strip_receipts(v78c),
        },
        "capacity": {
            "v80_tokens_per_actor": CAPACITY_TOKENS[V80_RUN],
            "v80_full_2048_token_contexts_floor": CAPACITY_TOKENS[V80_RUN] // 2048,
            "v80_remainder_tokens_after_full_contexts": CAPACITY_TOKENS[V80_RUN] % 2048,
            "parent_minimum_tokens": 161_792,
            "v80_margin_over_parent_minimum_tokens": CAPACITY_TOKENS[V80_RUN] - 161_792,
            "v80_margin_over_v76_tokens": CAPACITY_TOKENS[V80_RUN] - 157_696,
            "v79b_tokens_per_actor": CAPACITY_TOKENS[V79B_RUN],
            "v80_minus_v79b_tokens": CAPACITY_TOKENS[V80_RUN] - CAPACITY_TOKENS[V79B_RUN],
            "v80_ratio_to_v79b": CAPACITY_TOKENS[V80_RUN] / CAPACITY_TOKENS[V79B_RUN],
            "v78c_tokens_per_actor": CAPACITY_TOKENS[V78C_RUN],
            "v80_ratio_to_v78c": CAPACITY_TOKENS[V80_RUN] / CAPACITY_TOKENS[V78C_RUN],
        },
        "performance": {
            "v80_r1_actor_median_seconds": v80_median,
            "v79b_r5_actor_median_seconds": v79b_median,
            "v78c_r1_actor_median_seconds": v78c_median,
            "v80_ratio_to_v79b_r5": v80_median / v79b_median,
            "v80_ratio_to_v78c_r1": v80_median / v78c_median,
            "v80_preregistered_runtime_seconds_max": 50.55591316620558,
            "v80_preregistered_ratio_to_v78c_max": 1.03,
            "v80_runtime_gate_passed": runtime_pass,
            "single_replicate_timing_not_a_promotion_claim": True,
        },
        "vram_and_bandwidth": {
            "external_peak_memory_used_mib": {
                "v80_r1_including_pynvml_observer": v80_peak,
                "v79b_r5_including_pynvml_observer": v79b_peak,
                "v78c_r1_old_external_monitor": v78c_peak,
            },
            "v80_peak_attributed_actor_process_mib": v80_process_peak,
            "v80_external_savings_mib_vs_v79b_r5": v79b_peak - v80_peak,
            "v80_external_savings_mib_vs_v78c_r1": v78c_peak - v80_peak,
            "v80_peak_memory_gate_passed": memory_pass,
            "sampled_pcie_integrals": v80_telemetry["sampled_pcie_integrals"],
            "hbm_bandwidth_bytes_per_second": None,
            "hbm_bandwidth_not_inferred": True,
            "observer_caveat": (
                "new telemetry's total includes the in-process pynvml observer; "
                "the attributed actor-process peak is diagnostic, while the "
                "external total is used for the fail-closed gate"
            ),
        },
        "paired_token_hash_drift": drift,
        "parent_gate_evaluation": {
            "data_free_gate_results": gate_results,
            "all_data_free_gates_except_literal_torch_process_group_clause_passed": True,
            "literal_parent_data_free_contract_passed": False,
            "literal_failure": {
                "parent_clause": "torch_process_group_destroyed_per_actor=true",
                "receipt_values_by_gpu": [False] * 4,
                "probe_semantics": (
                    "the receipt field is true only when torch.distributed was "
                    "initialized and destroy_process_group() was called; TP1 may "
                    "have no initialized process group, but that semantic "
                    "reinterpretation was not preregistered"
                ),
                "external_process_cleanup_still_passed": True,
                "postrun_threshold_or_gate_reinterpretation_forbidden": True,
            },
            "reference_restore_exact": False,
            "reference_repeat_changed_rows_by_gpu": {
                str(actor["gpu"]): actor["reference_repeat_changed_rows"]
                for actor in v80["actors"]
            },
            "candidate_repeat_exact_all_actors": output_pass,
            "source_disjoint_semantic_gate_run": False,
            "protected_one_shot_ood_gate_run": False,
            "promotion_authorized": False,
        },
        "prospective_v80b": {
            "content_sha256": PREREGISTRATIONS[
                "qwen36_bf16_kv_mamba_confirmation_v80b.json"
            ]["content_sha256"],
            "exactly_two_runs": [
                "v80_bf16_kv_mamba_capacity_0479_r2",
                "v80_bf16_kv_mamba_capacity_0479_r3",
            ],
            "exact_commands": exact_commands,
            "parent_runtime_and_thresholds_unchanged": True,
            "r1_observation_disclosed": True,
            "promotion_forbidden": True,
        },
        "analysis_completed": True,
        "candidate_parent_contract_passed": False,
        "promotion_authorized": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v80(result)
    return result


def render_json_v80(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"


def render_report_v80(value: Mapping[str, Any]) -> str:
    capacity = value["capacity"]
    performance = value["performance"]
    memory = value["vram_and_bandwidth"]
    gates = value["parent_gate_evaluation"]
    v80 = value["v80_r1"]
    telemetry = v80["telemetry"]
    drift_v78 = value["paired_token_hash_drift"]["v78c_r1_vs_v80_r1"]
    drift_v79 = value["paired_token_hash_drift"]["v79b_r5_vs_v80_r1"]
    commands = value["prospective_v80b"]["exact_commands"]
    actor_peaks = [
        row["peak_memory_used_mib"] for row in telemetry["per_gpu"].values()
    ]
    actor_process_peaks = [
        row["peak_attributed_process_memory_mib"]
        for row in telemetry["per_gpu"].values()
    ]
    gate_lines = "\n".join(
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in gates["data_free_gate_results"].items()
    )
    return f"""# Qwen3.6 BF16 hybrid-cache V80 r1 live evidence

## Outcome

V80 r1 passes every preregistered data-free capacity, runtime, backend,
residency, log, four-GPU activity, PCIe-observation, VRAM, output, engine
shutdown, and external cleanup check except one literal clause.  The parent
requires `torch_process_group_destroyed_per_actor=true`, while all four actor
receipts contain `false`.  The probe sets this field to true only if a Torch
process group was initialized before `destroy_process_group()` was called;
TP1 may initialize no such group.  That is a plausible semantic explanation,
but it was not preregistered, so this report does not reinterpret the gate
after seeing r1.  The literal parent data-free contract therefore **fails**.

The external cleanup evidence is independently strong: trailing batches
{telemetry['trailing_external_cleanup_batches']} show dead actor roots, no
compute or foreign PIDs, exactly 0% GPU utilization, and at most 4 MiB from
external `nvidia-smi` on all four GPUs.  Semantic and protected OOD evaluation
remain unopened, reference restoration remains non-exact, and promotion is
not authorized.

## Sealed evidence

- V80 r1 bundle: `{v80['artifact_inventory']['bundle_sha256']}`
- V80 parent content: `{value['static_contract']['preregistrations']['qwen36_bf16_kv_mamba_capacity_matched_v80.json']['content_sha256']}`
- V80B prospective confirmation content: `{value['prospective_v80b']['content_sha256']}`
- Executable source bundle: `{value['static_contract']['sealed_executable_sources']['bundle_sha256']}`
- Analysis content: `{value['content_sha256_before_self_field']}`

No dataset, prompt text, generated text, token IDs, protected evaluation data,
model, or GPU API was opened by this analyzer.

## Capacity, speed, and memory

| Arm | Attention KV / Mamba SSM | Utilization | Tokens / actor | Full 2048-token contexts | Four-actor median |
|---|---|---:|---:|---:|---:|
| V80 r1 | auto→BF16 / BF16 | 0.479 | {capacity['v80_tokens_per_actor']:,} | {capacity['v80_full_2048_token_contexts_floor']} | {performance['v80_r1_actor_median_seconds']:.3f} s |
| V79B r5 | FP8 per-token-head / FP32 | 0.485 | {capacity['v79b_tokens_per_actor']:,} | {capacity['v79b_tokens_per_actor'] // 2048} | {performance['v79b_r5_actor_median_seconds']:.3f} s |
| V78c r1 | auto→BF16 / BF16 | 0.500 | {capacity['v78c_tokens_per_actor']:,} | {capacity['v78c_tokens_per_actor'] // 2048} | {performance['v78c_r1_actor_median_seconds']:.3f} s |

V80 exceeds its 161,792-token minimum by
{capacity['v80_margin_over_parent_minimum_tokens']:,} tokens and V76 by
{capacity['v80_margin_over_v76_tokens']:,}.  It has
{capacity['v80_minus_v79b_tokens']:,} more live cache tokens than V79B.  The
r1 median is {(1 - performance['v80_ratio_to_v78c_r1']) * 100:.2f}% faster than
V78c r1 and {(1 - performance['v80_ratio_to_v79b_r5']) * 100:.2f}% faster than
V79B r5, but a single replicate is not a promotion-quality timing estimate.

V80 external peak MiB by GPU is {actor_peaks}; attributed actor-process peaks
are {actor_process_peaks}.  Its worst external peak is
{memory['external_peak_memory_used_mib']['v80_r1_including_pynvml_observer']:,}
MiB, within the 50,808 MiB gate.  New-monitor totals include the in-process
NVML observer, so the external total is used for the fail-closed gate and the
attributed value is reported only as a diagnostic.  Sampled PCIe RX/TX totals
are {memory['sampled_pcie_integrals']['rx_bytes_sum_gpus']:,} /
{memory['sampled_pcie_integrals']['tx_bytes_sum_gpus']:,} bytes using the
preregistered left-rectangle approximation.  HBM bytes/s are not inferred
from NVML memory-utilization percentages.

## Parent data-free gate matrix

| Gate | Result |
|---|---|
{gate_lines}

Candidate repeats are exact on all four actors.  Reference repeat changed-row
counts are {list(gates['reference_repeat_changed_rows_by_gpu'].values())}; the
known nondeterminism remains visible.

## Paired token-hash drift

| Pair | Differing / compared rows | Agreement | Differences by call |
|---|---:|---:|---|
| V78c r1 vs V80 r1 | {drift_v78['differing_rows_total']} / {drift_v78['compared_rows_total']} | {drift_v78['agreement_fraction']:.2%} | {drift_v78['differing_rows_by_call_sum_gpus']} |
| V79B r5 vs V80 r1 | {drift_v79['differing_rows_total']} / {drift_v79['compared_rows_total']} | {drift_v79['agreement_fraction']:.2%} | {drift_v79['differing_rows_by_call_sum_gpus']} |

These are token-hash commitment comparisons, not semantic quality scores.

## Prospective V80B confirmations

V80B was sealed only after disclosing that r1 had been observed.  It copies
the parent runtime and all thresholds unchanged, admits exactly r2 and r3,
forbids replacement/exclusion after failure, and cannot authorize promotion.
The exact commands are:

```bash
{commands[0]}
{commands[1]}
```

Run them sequentially because each command uses all four GPUs.  Analyze both
independently; do not tune thresholds or reinterpret the literal process-group
clause from r1, r2, or r3.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()
    value = analyze_finalized_v80()
    json_payload = render_json_v80(value)
    report_payload = render_report_v80(value)
    output = Path(args.output).resolve()
    report = Path(args.report).resolve()
    if args.check:
        _require(output.is_file(), f"analysis JSON missing: {output}")
        _require(report.is_file(), f"analysis report missing: {report}")
        _require(
            output.read_text(encoding="ascii") == json_payload,
            f"analysis JSON stale: {output}",
        )
        _require(
            report.read_text(encoding="utf-8") == report_payload,
            f"analysis report stale: {report}",
        )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_payload, encoding="ascii")
        report.write_text(report_payload, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "report": str(report),
                "content_sha256": value["content_sha256_before_self_field"],
                "candidate_parent_contract_passed": value[
                    "candidate_parent_contract_passed"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
