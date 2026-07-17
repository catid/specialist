#!/usr/bin/env python3
"""Deterministic, CPU-only postrun analysis for Qwen3.6 V79/V79B.

The analyzer deliberately separates two evidence classes:

* V79 ``r1_fix``/``r2``/``r3`` are same-model performance replicates.  Their
  older monitor did not make the external ``nvidia-smi`` cleanup observation,
  so they cannot satisfy V79B cleanup acceptance.
* V79B ``r5`` is the only run launched after the corrected cleanup contract was
  sealed.  It is the only run eligible for full data-free runtime acceptance.

No torch, vLLM, NVML, dataset, model, or GPU module is imported.  Persisted
token IDs and text are neither required nor opened; comparisons use the
already-persisted token-count and SHA-256 commitments only.
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
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "qwen36-fp8-kv-capacity-postrun-analysis-v79b"
ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_fp8_kv_capacity_v79b_postrun_20260717.json"
)
PREREG_V79 = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fp8_kv_capacity_matched_v79.json"
)
PREREG_V79B = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fp8_kv_capacity_cleanup_v79b.json"
)
RUN_ROOT = ROOT / "experiments/eggroll_es_hpo/runs"

V79_DIAGNOSTIC_RUNS = (
    "v79_fp8_kv_capacity_0485_r1_fix",
    "v79_fp8_kv_capacity_0485_r2",
    "v79_fp8_kv_capacity_0485_r3",
)
V79B_ACCEPTED_RUN = "v79b_fp8_kv_capacity_0485_r5_sealed_cleanup"
V76_PERFORMANCE_RUNS = (
    "v76_fp8_attested_050_r5",
    "v76_fp8_attested_050_r6_residency",
    "v76_fp8_attested_050_r7_residency",
)
V78_PERFORMANCE_RUNS = (
    "v78_fp8_per_token_head_kv_r1",
    "v78_fp8_per_token_head_kv_r2",
    "v78_fp8_per_token_head_kv_r3",
)
V76_PAIRED_RUN = V76_PERFORMANCE_RUNS[-1]
V78_PAIRED_RUN = V78_PERFORMANCE_RUNS[-1]

EXPECTED_PREREGISTRATIONS = {
    str(PREREG_V79.relative_to(ROOT)): {
        "schema": "v79-qwen36-fp8-kv-capacity-matched-preregistration",
        "content_sha256": (
            "6c73ac0f6bf4019cdf297546e4315dc99b68d9549a24c03f4eaa9c8ebb589023"
        ),
        "file_sha256": (
            "0e195c05fd72e36656ee6536d6656d932aac0028fbbc7983f688df9dc7b18753"
        ),
    },
    str(PREREG_V79B.relative_to(ROOT)): {
        "schema": "v79b-qwen36-fp8-kv-cleanup-preregistration",
        "content_sha256": (
            "7669c2f720f2a0d17e976de42cc5b7c08fba60a3251175a62eddf05de2dc1b5d"
        ),
        "file_sha256": (
            "8e1940db5134bb77ef9959d10b4eec5d43fab4e8653d62733b42939b5fd7300f"
        ),
    },
}
EXPECTED_SOURCE_SHA256 = {
    "probe_vllm_fp8_kv_capacity_v79.py": (
        "6b72de1bd7d7878ba4183bae618108f8cd1cf997e33c7447ee2459700e15ff45"
    ),
    "monitor_qwen36_fp8_kv_capacity_v79.py": (
        "6035eb32f90815ed2a2d8734d9e9072123b8ecc70d74449a2710e94b673ed3df"
    ),
    "launch_qwen36_fp8_kv_capacity_v79.sh": (
        "4ca93e3a171787bb56613bf3648365ae96a355e28ec95f094b12d1982b6772df"
    ),
}
EXPECTED_PAIRED_BUNDLES = {
    V76_PAIRED_RUN: (
        "46cf5ab3e6d3688de25cfdcf101710a129fdba309a5f11a9404d17344848e5e6"
    ),
    V78_PAIRED_RUN: (
        "e6df12c976910948c1026249b05fc065932169897aa5a09ff984b6d765385463"
    ),
}
EXPECTED_RUN_BUNDLES = {
    "v76_fp8_attested_050_r5": (
        "5124652dc91af81de6e55c66d5eaa6b8a6b355a85f50369052d960eb5c028d87"
    ),
    "v76_fp8_attested_050_r6_residency": (
        "142fea7a45b62ec87d1d60c35f8819e017b79ac3a4004aa1fdb3e4882d775795"
    ),
    "v76_fp8_attested_050_r7_residency": (
        "46cf5ab3e6d3688de25cfdcf101710a129fdba309a5f11a9404d17344848e5e6"
    ),
    "v78_fp8_per_token_head_kv_r1": (
        "0897b1c80b8161171736b994e1e5e4a88728a19d39200f280ff7799552838c71"
    ),
    "v78_fp8_per_token_head_kv_r2": (
        "0bcfe80ff428348b06e9295bf3ed67df7acb674a7dd3a6a74b08cd79c7b9bb9c"
    ),
    "v78_fp8_per_token_head_kv_r3": (
        "e6df12c976910948c1026249b05fc065932169897aa5a09ff984b6d765385463"
    ),
    "v79_fp8_kv_capacity_0485_r1_fix": (
        "0b8e57a86fe1d3ad1a7e34de6072a54e27cd197ffa9b4a551079ab8563f0e587"
    ),
    "v79_fp8_kv_capacity_0485_r2": (
        "27934a9f678018c20a91e13ab7095633f366fe1827c0f68540d73436772451a5"
    ),
    "v79_fp8_kv_capacity_0485_r3": (
        "4cd37911ace8fe496ae50c0788772e61c9127654d27988493b550204de8b62af"
    ),
    "v79b_fp8_kv_capacity_0485_r5_sealed_cleanup": (
        "17a4f99164cd17f8e55ddcd879460c99e356e34d8e90e017abbbd7597085dde7"
    ),
}

ACTOR_SCHEMA_V79 = "v79-qwen36-fp8-kv-capacity-matched-preflight"
ACTOR_SCHEMA_V76 = "v76-qwen36-fp8-routed-runtime-attestation"
ACTOR_SCHEMA_V78 = "v78-qwen36-fp8-per-token-head-kv-preflight"
TELEMETRY_SCHEMA = "v79-four-gpu-kv-capacity-telemetry"
CALL_PLAN = [
    "reference", "candidate", "candidate", "reference",
    "reference", "candidate", "candidate", "reference",
]
GPU_IDS = (0, 1, 2, 3)
ROWS_PER_CALL = 68
TOKENS_PER_ROW = 64
V76_TOKENS = 157_696
V78_TOKENS = 198_656
V79_TOKENS = 162_304
MEMORY_TOTAL_MIB = 97_887

REQUIRED_LOG_FRAGMENTS = (
    "Using TRITON Fp8 MoE backend",
    "Using TRITON_ATTN attention backend",
    "Available KV cache memory:",
    "GPU KV cache size:",
    "Maximum concurrency for 2,048 tokens per request:",
    "Skipping FlashInfer autotune because it is disabled",
)
FORBIDDEN_LOG_FRAGMENTS = (
    "Auto-disabled DeepGemm",
    "DeepGEMM E8M0 enabled",
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "falling back",
    "fallback to",
    "please install bitsandbytes",
    "not supported by",
)
KNOWN_VISIBLE_WARNINGS = (
    "Using default MoE config",
    "No LoRA kernel configs found",
    "Using default LoRA kernel configs",
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
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="ascii"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _validate_self_hash(value: Mapping[str, Any], label: str) -> None:
    body = copy.deepcopy(dict(value))
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and claimed == canonical_sha256(body),
        f"self hash changed: {label}",
    )


def _inventory(directory: Path, expected_count: int | None = None) -> dict[str, Any]:
    directory = Path(directory).resolve()
    _require(directory.is_dir() and not directory.is_symlink(), f"missing run: {directory}")
    paths = sorted(directory.iterdir())
    _require(
        all(path.is_file() and not path.is_symlink() for path in paths),
        f"run contains a non-regular file: {directory}",
    )
    if expected_count is not None:
        _require(len(paths) == expected_count, f"run file count changed: {directory}")
    rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in paths
    ]
    return {
        "file_count": len(rows),
        "bundle_sha256": canonical_sha256(rows),
        "files": rows,
    }


def validate_static_contract(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    preregistrations = {}
    for relative, expected in EXPECTED_PREREGISTRATIONS.items():
        path = root / relative
        _require(file_sha256(path) == expected["file_sha256"], f"file hash changed: {relative}")
        value = _load_json(path)
        _validate_self_hash(value, relative)
        _require(
            value.get("schema") == expected["schema"]
            and value.get("content_sha256_before_self_field")
            == expected["content_sha256"],
            f"preregistration content changed: {relative}",
        )
        preregistrations[relative] = dict(expected)

    source_rows = []
    for relative, expected in EXPECTED_SOURCE_SHA256.items():
        path = root / relative
        actual = file_sha256(path)
        _require(actual == expected, f"sealed V79B source changed: {relative}")
        source_rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": actual})
    prereg_v79b = _load_json(root / PREREG_V79B.relative_to(ROOT))
    _require(
        prereg_v79b.get("sealed_sources", {}).get("files") == source_rows
        and prereg_v79b.get("sealed_sources", {}).get("bundle_sha256")
        == canonical_sha256(source_rows)
        and prereg_v79b.get("authority", {}).get(
            "scored_training_checkpoint_or_promotion_authorized"
        )
        is False,
        "V79B sealed source inventory or authority changed",
    )
    return {
        "preregistrations": preregistrations,
        "sealed_sources": {
            "files": source_rows,
            "bundle_sha256": canonical_sha256(source_rows),
        },
    }


def _validate_hash_rows(receipt: Mapping[str, Any], label: str) -> None:
    calls = receipt.get("calls")
    _require(isinstance(calls, list) and len(calls) == len(CALL_PLAN), f"call count changed: {label}")
    _require(receipt.get("call_plan") == CALL_PLAN, f"call plan changed: {label}")
    for call_index, (call, expected_label) in enumerate(zip(calls, CALL_PLAN)):
        _require(
            isinstance(call, dict)
            and call.get("call_index") == call_index
            and call.get("label") == expected_label,
            f"call identity changed: {label}:{call_index}",
        )
        rows = call.get("rows")
        _require(isinstance(rows, list) and len(rows) == ROWS_PER_CALL, f"row count changed: {label}")
        for row in rows:
            _require(
                isinstance(row, dict)
                and set(row) == {"token_count", "token_ids_sha256"}
                and row.get("token_count") == TOKENS_PER_ROW
                and isinstance(row.get("token_ids_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", row["token_ids_sha256"]) is not None,
                f"hash-only row changed: {label}",
            )
        _require(call.get("rows_sha256") == canonical_sha256(rows), f"row hash changed: {label}")

    by_label = {
        state: [call for call in calls if call["label"] == state]
        for state in ("reference", "candidate")
    }
    changed = {}
    for state, state_calls in by_label.items():
        changed[state] = sum(
            len({call["rows"][row]["token_ids_sha256"] for call in state_calls}) > 1
            for row in range(ROWS_PER_CALL)
        )
    between = sum(
        by_label["reference"][0]["rows"][row]["token_ids_sha256"]
        != by_label["candidate"][0]["rows"][row]["token_ids_sha256"]
        for row in range(ROWS_PER_CALL)
    )
    _require(
        receipt.get("reference_within_state_changed_rows") == changed["reference"]
        and receipt.get("candidate_within_state_changed_rows") == changed["candidate"]
        and receipt.get("between_state_differing_rows") == between,
        f"derived output counters changed: {label}",
    )


def _validate_generation_performance(value: Mapping[str, Any], label: str) -> None:
    calls = value.get("calls")
    _require(
        value.get("schema") == "v79-data-free-generation-performance"
        and isinstance(calls, list)
        and len(calls) == 10
        and value.get("warmup_call_count") == 2
        and value.get("measured_call_count") == 8,
        f"timing cardinality changed: {label}",
    )
    _require(
        [row.get("call_index") for row in calls] == list(range(10))
        and [row.get("label") for row in calls[2:]] == CALL_PLAN,
        f"timing call plan changed: {label}",
    )
    for row in calls:
        _require(
            row.get("request_count") == ROWS_PER_CALL
            and row.get("generated_token_count") == ROWS_PER_CALL * TOKENS_PER_ROW
            and isinstance(row.get("elapsed_seconds"), (int, float))
            and not isinstance(row.get("elapsed_seconds"), bool)
            and math.isfinite(row["elapsed_seconds"])
            and row["elapsed_seconds"] > 0,
            f"timing row changed: {label}",
        )
    measured = [row["elapsed_seconds"] for row in calls[2:]]
    total_seconds = sum(measured)
    total_tokens = 8 * ROWS_PER_CALL * TOKENS_PER_ROW
    nearest_rank_p95 = sorted(measured)[math.ceil(0.95 * len(measured)) - 1]
    _require(
        value.get("measured_generated_token_count") == total_tokens
        and math.isclose(value.get("measured_generation_seconds_sum"), total_seconds, rel_tol=0, abs_tol=1e-12)
        and math.isclose(value.get("aggregate_generated_tokens_per_second"), total_tokens / total_seconds, rel_tol=0, abs_tol=1e-9)
        and math.isclose(value.get("median_call_latency_seconds"), statistics.median(measured), rel_tol=0, abs_tol=1e-12)
        and math.isclose(value.get("p95_call_latency_seconds_nearest_rank"), nearest_rank_p95, rel_tol=0, abs_tol=1e-12)
        and math.isclose(value.get("max_call_latency_seconds"), max(measured), rel_tol=0, abs_tol=1e-12)
        and value.get("prompt_or_generation_text_persisted") is False
        and value.get("token_ids_persisted") is False,
        f"timing summary changed: {label}",
    )


def validate_v79_actor(receipt: Mapping[str, Any], gpu: int, label: str) -> dict[str, Any]:
    _validate_self_hash(receipt, label)
    _validate_hash_rows(receipt, label)
    _require(
        receipt.get("schema") == ACTOR_SCHEMA_V79
        and receipt.get("actor_label") == f"gpu-{gpu}"
        and receipt.get("source_dataset_rows_opened") == 0
        and receipt.get("protected_ood_shadow_or_terminal_opened") is False
        and receipt.get("adapter_update_or_hpo_performed") is False
        and receipt.get("prompt_or_generation_text_persisted") is False
        and receipt.get("token_ids_persisted") is False
        and receipt.get("engine_shutdown_completed") is True
        and receipt.get("authority", {}).get(
            "scored_evaluation_training_checkpoint_or_promotion_allowed"
        )
        is False,
        f"actor authority changed: {label}",
    )
    runtime = receipt.get("runtime", {})
    kv = receipt.get("resolved_kv_cache_certificate", {})
    precision = receipt.get("resolved_precision_certificate", {})
    _require(
        runtime.get("gpu_memory_utilization") == 0.485
        and runtime.get("kv_cache_dtype") == "fp8_per_token_head"
        and runtime.get("resolved_quantization") == "fp8"
        and runtime.get("serialized_fp8_checkpoint") is True
        and runtime.get("starting_moe_tuning_table") == "fresh_empty_default"
        and runtime.get("enforce_eager") is True
        and runtime.get("async_scheduling") is False
        and runtime.get("scheduling_policy") == "fcfs"
        and runtime.get("max_num_seqs") == ROWS_PER_CALL
        and runtime.get("max_loras") == 1
        and runtime.get("max_cpu_loras") == 2
        and runtime.get("enable_flashinfer_autotune") is False
        and kv.get("resolved_from_live_engine") is True
        and kv.get("gpu_memory_utilization") == 0.485
        and kv.get("cache_dtype") == "fp8_per_token_head"
        and kv.get("calculate_kv_scales") is False
        and kv.get("kv_cache_dtype_skip_layers") == []
        and kv.get("mamba_cache_dtype") == "auto"
        and kv.get("mamba_ssm_cache_dtype") == "float32"
        and kv.get("kv_cache_size_tokens") == V79_TOKENS
        and kv.get("kv_cache_max_concurrency") == 79.25
        and kv.get("block_size") == 2048
        and precision.get("model_quantization") == "fp8"
        and precision.get("quant_config_class") == "Fp8Config"
        and precision.get("weight_block_size") == [128, 128],
        f"live precision/KV certificate changed: {label}",
    )
    _require(
        receipt.get("preregistration_v79")
        == {
            "content_sha256": EXPECTED_PREREGISTRATIONS[
                str(PREREG_V79.relative_to(ROOT))
            ]["content_sha256"],
            "file_sha256": EXPECTED_PREREGISTRATIONS[
                str(PREREG_V79.relative_to(ROOT))
            ]["file_sha256"],
        }
        and receipt.get("single_variable_change_from_v78")
        == {"gpu_memory_utilization": [0.5, 0.485]}
        and receipt.get("v78_reference_identity", {}).get("run_bundle_sha256")
        == EXPECTED_PAIRED_BUNDLES[V78_PAIRED_RUN],
        f"V79 ancestry changed: {label}",
    )
    audit = receipt.get("routed_fp8_runtime_attestation", {})
    records = audit.get("fp8_moe_records")
    residency = audit.get("parameter_residency", {})
    language = residency.get("components", {}).get("language", {})
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
        f"TRITON routed-expert or text-only residency audit changed: {label}",
    )
    _validate_generation_performance(receipt.get("generation_performance", {}), label)
    gates = receipt.get("preflight_gates", {})
    _require(
        gates.get("candidate_repeat_exact_at_token_hash_level") is True
        and gates.get("candidate_changes_output") is True
        and gates.get("reference_restore_exact_at_token_hash_level") is False
        and gates.get("kv_capacity_minimum_tokens_passed") is True
        and gates.get("scored_evaluation_or_training_authorized") is False,
        f"actor gate state changed: {label}",
    )
    wall = receipt.get("wall_runtime_seconds_excluding_model_load_and_cleanup")
    _require(
        isinstance(wall, (int, float))
        and not isinstance(wall, bool)
        and math.isfinite(wall)
        and wall > 0,
        f"wall runtime changed: {label}",
    )
    return {
        "gpu": gpu,
        "receipt_content_sha256": receipt["content_sha256_before_self_field"],
        "wall_runtime_seconds": wall,
        "generated_tokens_per_second": receipt["generation_performance"][
            "aggregate_generated_tokens_per_second"
        ],
        "reference_repeat_changed_rows": receipt[
            "reference_within_state_changed_rows"
        ],
        "candidate_repeat_changed_rows": receipt[
            "candidate_within_state_changed_rows"
        ],
        "between_state_differing_rows": receipt["between_state_differing_rows"],
    }


def validate_log(path: Path, expected_tokens: int, require_triton_attn: bool = True) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="strict")
    if require_triton_attn:
        for fragment in REQUIRED_LOG_FRAGMENTS:
            _require(text.count(fragment) == 1, f"required log cardinality changed: {path}:{fragment}")
    for fragment in FORBIDDEN_LOG_FRAGMENTS:
        _require(fragment.lower() not in text.lower(), f"forbidden fallback/error log: {path}:{fragment}")
    match = re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    _require(len(match) == 1 and int(match[0].replace(",", "")) == expected_tokens, f"capacity log changed: {path}")
    concurrency = re.findall(
        r"Maximum concurrency for 2,048 tokens per request: ([\d.]+)x", text
    )
    _require(len(concurrency) == 1, f"concurrency log changed: {path}")
    return {
        "file_sha256": file_sha256(path),
        "kv_cache_size_tokens": expected_tokens,
        "maximum_concurrency": float(concurrency[0]),
        "required_fragment_counts": {
            fragment: text.count(fragment) for fragment in REQUIRED_LOG_FRAGMENTS
        }
        if require_triton_attn
        else {},
        "forbidden_fragment_counts": {
            fragment: text.lower().count(fragment.lower())
            for fragment in FORBIDDEN_LOG_FRAGMENTS
        },
        "known_visible_warning_counts": {
            fragment: text.count(fragment) for fragment in KNOWN_VISIBLE_WARNINGS
        },
    }


def _read_pid_map(path: Path) -> dict[int, int]:
    rows = Path(path).read_text(encoding="ascii").splitlines()
    _require(len(rows) == 4, f"actor PID map cardinality changed: {path}")
    result = {}
    for row in rows:
        fields = row.split(",")
        _require(len(fields) == 2, f"actor PID map row changed: {path}")
        gpu, pid = map(int, fields)
        _require(gpu in GPU_IDS and pid > 1 and gpu not in result, f"actor PID map changed: {path}")
        result[gpu] = pid
    _require(tuple(sorted(result)) == GPU_IDS and len(set(result.values())) == 4, f"PID/GPU binding changed: {path}")
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        _require(bool(line), f"blank JSONL row: {path}:{line_number}")
        value = json.loads(line)
        _require(isinstance(value, dict), f"JSONL object required: {path}:{line_number}")
        rows.append(value)
    _require(bool(rows), f"empty telemetry: {path}")
    return rows


def validate_v79_telemetry(
    path: Path,
    pid_map: Mapping[int, int],
    *,
    require_external_cleanup: bool,
) -> dict[str, Any]:
    rows = _load_jsonl(path)
    _require(
        [row.get("sequence") for row in rows] == list(range(len(rows))),
        f"telemetry sequence changed: {path}",
    )
    by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_gpu: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gpu = row.get("gpu")
        _require(
            row.get("schema") == TELEMETRY_SCHEMA
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
            f"telemetry contract changed: {path}",
        )
        by_batch[row["batch_index"]].append(row)
        by_gpu[gpu].append(row)
    _require(sorted(by_batch) == list(range(len(by_batch))), f"batch sequence changed: {path}")

    monotonic = []
    for batch_index, batch in sorted(by_batch.items()):
        _require(
            len(batch) == 4
            and sorted(row["gpu"] for row in batch) == list(GPU_IDS)
            and len({row["monotonic_ns"] for row in batch}) == 1
            and len({row["sampled_at_utc"] for row in batch}) == 1,
            f"incomplete four-GPU telemetry batch: {path}:{batch_index}",
        )
        monotonic.append(batch[0]["monotonic_ns"])
    intervals = [
        (right - left) / 1e9 for left, right in zip(monotonic, monotonic[1:])
    ]
    _require(
        bool(intervals)
        and all(math.isfinite(value) and 0 < value <= 1.0 for value in intervals),
        f"telemetry sample interval changed: {path}",
    )

    per_gpu = {}
    for gpu in GPU_IDS:
        gpu_rows = by_gpu[gpu]
        uuids = {row["gpu_uuid"] for row in gpu_rows}
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
        _require(
            len(uuids) == 1
            and len(attributed) == 1
            and attributed == ancestry
            and any(row["gpu_utilization_percent"] > 0 for row in gpu_rows)
            and any(row["memory_utilization_percent"] > 0 for row in gpu_rows),
            f"GPU was not uniquely attributed and useful: {path}:gpu{gpu}",
        )
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
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in gpu_rows),
            "peak_attributed_process_memory_mib": process_peak,
            "minimum_physical_headroom_mib": MEMORY_TOTAL_MIB
            - max(row["memory_used_mib"] for row in gpu_rows),
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
        accepted = all(
            row.get("actor_root_alive") is False
            and row.get("compute_pids") == []
            and row.get("foreign_compute_pids") == []
            and row.get("cleanup_memory_gate_uses_external_nvidia_smi") is True
            and row.get("cleanup_nvidia_smi_gpu_utilization_percent") == 0
            and isinstance(row.get("cleanup_nvidia_smi_memory_used_mib"), int)
            and row["cleanup_nvidia_smi_memory_used_mib"] <= 4
            for row in batch
        )
        if accepted:
            cleanup_batches.append(batch_index)
    trailing = []
    for batch_index in reversed(sorted(by_batch)):
        if batch_index in cleanup_batches:
            trailing.append(batch_index)
        else:
            break
    trailing.reverse()
    if require_external_cleanup:
        _require(
            len(trailing) >= 3
            and trailing[-3:] == list(range(trailing[-1] - 2, trailing[-1] + 1)),
            f"V79B external cleanup gate failed: {path}",
        )
    else:
        _require(
            not cleanup_batches,
            f"diagnostic run unexpectedly claims corrected V79B cleanup: {path}",
        )
    return {
        "file_sha256": file_sha256(path),
        "row_count": len(rows),
        "complete_four_gpu_batches": len(by_batch),
        "sample_interval_seconds": {
            "minimum": min(intervals),
            "median": statistics.median(intervals),
            "maximum": max(intervals),
        },
        "per_gpu": per_gpu,
        "foreign_compute_pid_rows": 0,
        "sampled_pcie_integrals": {
            "claim": (
                "left-rectangle estimate across adjacent telemetry batches; "
                "not an exact byte count or HBM-bandwidth measurement"
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
        "external_cleanup_gate_passed": len(trailing) >= 3,
    }


def _validate_v79_run(name: str, require_external_cleanup: bool) -> dict[str, Any]:
    directory = RUN_ROOT / name
    inventory = _inventory(directory, 11)
    _require(
        inventory["bundle_sha256"] == EXPECTED_RUN_BUNDLES[name],
        f"sealed V79 run bundle changed: {name}",
    )
    pid_map = _read_pid_map(directory / "actor_pids.csv")
    actors = []
    logs = []
    receipts = []
    for gpu in GPU_IDS:
        receipt_path = directory / f"gpu_{gpu}.json"
        receipt = _load_json(receipt_path)
        receipts.append(receipt)
        actors.append(validate_v79_actor(receipt, gpu, str(receipt_path)))
        logs.append(validate_log(directory / f"gpu_{gpu}.log", V79_TOKENS))
    telemetry = validate_v79_telemetry(
        directory / "gpu_telemetry_v79.jsonl",
        pid_map,
        require_external_cleanup=require_external_cleanup,
    )
    walls = [actor["wall_runtime_seconds"] for actor in actors]
    throughputs = [actor["generated_tokens_per_second"] for actor in actors]
    return {
        "run": name,
        "evidence_class": (
            "fully_preregistered_cleanup_accepted_v79b"
            if require_external_cleanup
            else "same_model_performance_diagnostic_old_cleanup_monitor"
        ),
        "eligible_for_v79b_full_runtime_acceptance": require_external_cleanup,
        "artifact_inventory": inventory,
        "actor_pid_map": {str(gpu): pid_map[gpu] for gpu in GPU_IDS},
        "actors": actors,
        "actor_wall_runtime_seconds_median": statistics.median(walls),
        "actor_wall_runtime_seconds_range": [min(walls), max(walls)],
        "actor_generated_tokens_per_second_median": statistics.median(throughputs),
        "logs": logs,
        "telemetry": telemetry,
        "receipts": receipts,
    }


def _validate_baseline_actor(
    receipt: Mapping[str, Any], gpu: int, schema: str, label: str
) -> dict[str, Any]:
    _validate_self_hash(receipt, label)
    _validate_hash_rows(receipt, label)
    _require(
        receipt.get("schema") == schema
        and receipt.get("actor_label") == f"gpu-{gpu}"
        and receipt.get("source_dataset_rows_opened") == 0
        and receipt.get("protected_ood_shadow_or_terminal_opened") is False
        and receipt.get("adapter_update_or_hpo_performed") is False
        and receipt.get("engine_shutdown_completed") is True
        and receipt.get("preflight_gates", {}).get(
            "scored_evaluation_or_training_authorized"
        )
        is False,
        f"baseline actor changed: {label}",
    )
    wall = receipt.get("wall_runtime_seconds_excluding_model_load_and_cleanup")
    _require(isinstance(wall, (int, float)) and wall > 0 and math.isfinite(wall), f"baseline wall runtime changed: {label}")
    return {
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
    }


def _read_old_monitor(path: Path) -> dict[str, Any]:
    lines = Path(path).read_text(encoding="ascii").splitlines()
    _require(len(lines) % 5 == 0 and lines, f"old monitor framing changed: {path}")
    per_gpu = {gpu: [] for gpu in GPU_IDS}
    for offset in range(0, len(lines), 5):
        _require(re.fullmatch(r"\d+", lines[offset]) is not None, f"old monitor timestamp changed: {path}")
        batch = []
        for line in lines[offset + 1:offset + 5]:
            fields = [field.strip() for field in line.split(",")]
            _require(len(fields) == 4, f"old monitor row changed: {path}")
            gpu, utilization, memory = map(int, fields[:3])
            power = float(fields[3])
            _require(gpu in GPU_IDS and 0 <= utilization <= 100 and memory >= 0 and power >= 0, f"old monitor values changed: {path}")
            batch.append(gpu)
            per_gpu[gpu].append((utilization, memory, power))
        _require(sorted(batch) == list(GPU_IDS), f"old monitor GPU coverage changed: {path}")
    return {
        "file_sha256": file_sha256(path),
        "complete_four_gpu_batches": len(lines) // 5,
        "per_gpu": {
            str(gpu): {
                "peak_gpu_utilization_percent": max(row[0] for row in per_gpu[gpu]),
                "peak_memory_used_mib": max(row[1] for row in per_gpu[gpu]),
                "peak_power_draw_w": max(row[2] for row in per_gpu[gpu]),
            }
            for gpu in GPU_IDS
        },
    }


def _validate_baseline_run(name: str, schema: str, expected_tokens: int) -> dict[str, Any]:
    directory = RUN_ROOT / name
    inventory = _inventory(directory, 9)
    _require(
        inventory["bundle_sha256"] == EXPECTED_RUN_BUNDLES[name],
        f"sealed baseline run bundle changed: {name}",
    )
    if name in EXPECTED_PAIRED_BUNDLES:
        _require(
            inventory["bundle_sha256"] == EXPECTED_PAIRED_BUNDLES[name],
            f"paired baseline bundle changed: {name}",
        )
    actors = []
    receipts = []
    logs = []
    for gpu in GPU_IDS:
        receipt_path = directory / f"gpu_{gpu}.json"
        receipt = _load_json(receipt_path)
        receipts.append(receipt)
        actors.append(_validate_baseline_actor(receipt, gpu, schema, str(receipt_path)))
        logs.append(
            validate_log(
                directory / f"gpu_{gpu}.log",
                expected_tokens,
                require_triton_attn=(schema == ACTOR_SCHEMA_V78),
            )
        )
    monitor = _read_old_monitor(directory / "nvidia_smi_samples.log")
    walls = [actor["wall_runtime_seconds"] for actor in actors]
    return {
        "run": name,
        "artifact_inventory": inventory,
        "actors": actors,
        "actor_wall_runtime_seconds_median": statistics.median(walls),
        "actor_wall_runtime_seconds_range": [min(walls), max(walls)],
        "logs": logs,
        "telemetry": monitor,
        "receipts": receipts,
    }


def _different_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> int:
    _require(len(left["rows"]) == len(right["rows"]) == ROWS_PER_CALL, "drift row cardinality changed")
    return sum(
        lrow != rrow for lrow, rrow in zip(left["rows"], right["rows"])
    )


def paired_drift_matrix(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    _require(len(left) == len(right) == 4, f"paired actor count changed: {label}")
    by_gpu = {}
    by_call = [0] * len(CALL_PLAN)
    for gpu in GPU_IDS:
        _require(left[gpu].get("actor_label") == right[gpu].get("actor_label") == f"gpu-{gpu}", f"paired GPU identity changed: {label}")
        values = []
        for call_index in range(len(CALL_PLAN)):
            count = _different_rows(
                left[gpu]["calls"][call_index], right[gpu]["calls"][call_index]
            )
            values.append(count)
            by_call[call_index] += count
        by_gpu[str(gpu)] = values
    differing = sum(by_call)
    compared = len(GPU_IDS) * len(CALL_PLAN) * ROWS_PER_CALL
    return {
        "pair": label,
        "call_plan": CALL_PLAN,
        "differing_rows_by_gpu_and_call": by_gpu,
        "differing_rows_by_call_sum_gpus": by_call,
        "differing_rows_total": differing,
        "compared_rows_total": compared,
        "agreement_fraction": (compared - differing) / compared,
        "claim": (
            "hash-commitment disagreement, not a semantic quality metric; "
            "known reference-repeat nondeterminism remains visible"
        ),
    }


def _performance_series(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        actor["wall_runtime_seconds"]
        for run in runs
        for actor in run["actors"]
    ]
    return {
        "run_medians_seconds": {
            run["run"]: run["actor_wall_runtime_seconds_median"] for run in runs
        },
        "actor_count": len(values),
        "combined_actor_median_seconds": statistics.median(values),
        "combined_actor_range_seconds": [min(values), max(values)],
    }


def _strip_receipts(run: dict[str, Any]) -> dict[str, Any]:
    result = dict(run)
    result.pop("receipts", None)
    return result


def analyze_finalized(root: Path = ROOT) -> dict[str, Any]:
    root = Path(root).resolve()
    _require(root == ROOT, "alternate roots are not accepted for finalized evidence")
    static = validate_static_contract(root)
    diagnostics = [
        _validate_v79_run(name, require_external_cleanup=False)
        for name in V79_DIAGNOSTIC_RUNS
    ]
    accepted = _validate_v79_run(V79B_ACCEPTED_RUN, require_external_cleanup=True)
    v76_runs = [
        _validate_baseline_run(name, ACTOR_SCHEMA_V76, V76_TOKENS)
        for name in V76_PERFORMANCE_RUNS
    ]
    v78_runs = [
        _validate_baseline_run(name, ACTOR_SCHEMA_V78, V78_TOKENS)
        for name in V78_PERFORMANCE_RUNS
    ]

    v76_paired = next(run for run in v76_runs if run["run"] == V76_PAIRED_RUN)
    v78_paired = next(run for run in v78_runs if run["run"] == V78_PAIRED_RUN)
    drift = {
        "v76_r7_vs_v78_r3": paired_drift_matrix(
            v76_paired["receipts"], v78_paired["receipts"], "v76_r7_vs_v78_r3"
        ),
        "v76_r7_vs_v79b_r5": paired_drift_matrix(
            v76_paired["receipts"], accepted["receipts"], "v76_r7_vs_v79b_r5"
        ),
        "v78_r3_vs_v79b_r5": paired_drift_matrix(
            v78_paired["receipts"], accepted["receipts"], "v78_r3_vs_v79b_r5"
        ),
    }

    v79_diagnostic_perf = _performance_series(diagnostics)
    v79_all_perf = _performance_series([*diagnostics, accepted])
    v76_perf = _performance_series(v76_runs)
    v78_perf = _performance_series(v78_runs)
    accepted_median = accepted["actor_wall_runtime_seconds_median"]
    v76_r7_median = v76_paired["actor_wall_runtime_seconds_median"]
    v78_r3_median = v78_paired["actor_wall_runtime_seconds_median"]

    accepted_peak = max(
        row["peak_memory_used_mib"]
        for row in accepted["telemetry"]["per_gpu"].values()
    )
    accepted_process_peak = max(
        row["peak_attributed_process_memory_mib"]
        for row in accepted["telemetry"]["per_gpu"].values()
    )
    v76_peak = max(
        row["peak_memory_used_mib"]
        for row in v76_paired["telemetry"]["per_gpu"].values()
    )
    v78_peak = max(
        row["peak_memory_used_mib"]
        for row in v78_paired["telemetry"]["per_gpu"].values()
    )

    result = {
        "schema": SCHEMA,
        "status": "data_free_runtime_gates_passed_semantic_ood_and_exact_restore_pending",
        "authority": {
            "dataset_or_protected_data_opened": False,
            "model_update_training_checkpoint_or_promotion_performed": False,
            "scored_training_checkpoint_or_layout_promotion_authorized": False,
        },
        "static_contract": static,
        "evidence_classification": {
            "same_model_performance_diagnostics": list(V79_DIAGNOSTIC_RUNS),
            "only_fully_preregistered_cleanup_accepted_run": V79B_ACCEPTED_RUN,
            "excluded_runs": {
                "v79_fp8_kv_capacity_0485_r1": "incomplete failed first attempt",
                "v79_fp8_kv_capacity_0485_r4_cleanup": (
                    "cleanup implementation preceded immutable V79B preregistration"
                ),
            },
        },
        "v79_diagnostic_runs": [_strip_receipts(run) for run in diagnostics],
        "v79b_accepted_run": _strip_receipts(accepted),
        "baseline_runs": {
            "v76": [_strip_receipts(run) for run in v76_runs],
            "v78": [_strip_receipts(run) for run in v78_runs],
        },
        "capacity": {
            "v76_bf16_attention_kv_tokens": V76_TOKENS,
            "v78_fp8_attention_kv_tokens": V78_TOKENS,
            "v79b_capacity_matched_fp8_attention_kv_tokens": V79_TOKENS,
            "v79b_minimum_gate_tokens": 161_792,
            "v79b_margin_over_minimum_gate_tokens": V79_TOKENS - 161_792,
            "v79b_margin_over_v76_tokens": V79_TOKENS - V76_TOKENS,
            "v79b_ratio_to_v76": V79_TOKENS / V76_TOKENS,
            "v79b_ratio_to_v78": V79_TOKENS / V78_TOKENS,
            "v79b_full_2048_token_contexts_floor": V79_TOKENS // 2048,
        },
        "performance": {
            "v76_three_replicates": v76_perf,
            "v78_three_replicates": v78_perf,
            "v79_old_monitor_three_replicates": v79_diagnostic_perf,
            "v79b_cleanup_accepted_run_median_seconds": accepted_median,
            "v79b_cleanup_accepted_ratio_to_v76_r7": accepted_median / v76_r7_median,
            "v79b_cleanup_accepted_ratio_to_v78_r3": accepted_median / v78_r3_median,
            "v79_all_same_model_four_replicates": v79_all_perf,
            "v79_all_ratio_to_v76_three_replicates": (
                v79_all_perf["combined_actor_median_seconds"]
                / v76_perf["combined_actor_median_seconds"]
            ),
            "v79_all_ratio_to_v78_three_replicates": (
                v79_all_perf["combined_actor_median_seconds"]
                / v78_perf["combined_actor_median_seconds"]
            ),
            "preregistered_v79b_runtime_gate_passed": (
                accepted_median <= 50.840854837201476
                and accepted_median / v78_r3_median <= 1.03
            ),
        },
        "vram": {
            "external_peak_memory_used_mib": {
                "v76_r7": v76_peak,
                "v78_r3": v78_peak,
                "v79b_r5_including_pynvml_observer": accepted_peak,
            },
            "v79b_external_peak_ratio_to_v76_r7": accepted_peak / v76_peak,
            "v79b_external_peak_ratio_to_v78_r3": accepted_peak / v78_peak,
            "v79b_external_peak_savings_mib_vs_v76_r7": v76_peak - accepted_peak,
            "v79b_external_peak_savings_mib_vs_v78_r3": v78_peak - accepted_peak,
            "v79b_peak_attributed_actor_process_mib": accepted_process_peak,
            "v79b_attributed_actor_savings_mib_vs_v78_r3_external_peak": (
                v78_peak - accepted_process_peak
            ),
            "v79b_monitor_overhead_caveat": (
                "V79B in-process NVML observation holds about 598 MiB after actor "
                "exit; compare the external total for the fail-closed gate and the "
                "attributed process peak only as a diagnostic estimate"
            ),
            "v79b_external_peak_gate_passed": accepted_peak <= 50_856,
            "v79b_minimum_physical_headroom_gate_passed": all(
                row["minimum_physical_headroom_mib"] >= 47_031
                for row in accepted["telemetry"]["per_gpu"].values()
            ),
        },
        "paired_token_hash_drift": drift,
        "output_and_promotion_gates": {
            "candidate_repeat_exact_all_v79_actors": all(
                actor["candidate_repeat_changed_rows"] == 0
                for run in [*diagnostics, accepted]
                for actor in run["actors"]
            ),
            "v79b_reference_repeat_changed_rows_by_gpu": {
                str(actor["gpu"]): actor["reference_repeat_changed_rows"]
                for actor in accepted["actors"]
            },
            "v79b_candidate_changes_output_by_gpu": {
                str(actor["gpu"]): actor["between_state_differing_rows"]
                for actor in accepted["actors"]
            },
            "reference_restore_exact": False,
            "source_disjoint_semantic_gate_run": False,
            "protected_one_shot_ood_gate_run": False,
            "promotion_authorized": False,
        },
        "passed": True,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def render(value: Mapping[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    result = analyze_finalized()
    output = Path(args.output).resolve()
    expected = render(result)
    if args.check:
        _require(output.is_file(), f"postrun analysis missing: {output}")
        _require(output.read_text(encoding="ascii") == expected, f"postrun analysis stale: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(expected, encoding="ascii")
    print(json.dumps({
        "output": str(output),
        "content_sha256": result["content_sha256_before_self_field"],
        "status": result["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
