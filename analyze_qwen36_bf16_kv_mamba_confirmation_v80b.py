#!/usr/bin/env python3
"""CPU-only aggregate analysis of the sealed Qwen3.6 V80B confirmations.

The analysis opens only data-free JSON, log, telemetry, and preregistration
artifacts.  It never opens a dataset, launches a model/GPU, or changes an
immutable parent result.  V83 is applied only as an additive, source-bound
interpretation of the legacy TP1 process-group receipt.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "experiments/eggroll_es_hpo/runs"
PREREG_ROOT = ROOT / "experiments/eggroll_es_hpo/preregistrations"
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_bf16_kv_mamba_confirmation_v80b_postrun_20260717.json"
)
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_bf16_kv_mamba_confirmation_v80b_postrun_20260717.md"
)

SCHEMA = "v80b-qwen36-bf16-kv-mamba-three-run-aggregate-analysis"
GPU_IDS = (0, 1, 2, 3)
RUNS = (
    "v80_bf16_kv_mamba_capacity_0479_r1",
    "v80_bf16_kv_mamba_capacity_0479_r2",
    "v80_bf16_kv_mamba_capacity_0479_r3",
)
RUN_BUNDLES = {
    RUNS[0]: "73adc7ebe416d6065808cf918415d077989b1a064b47ef5132422b32da118e47",
    RUNS[1]: "3ac1156ac629d8c27d70756c08fa98652c74351549e5fea09711473900b584bc",
    RUNS[2]: "cd5107654df8367284a5fc720c50c3007ce16513ae77b55e7c5ccc52af3c3d5f",
}
CALL_PLAN = [
    "reference",
    "candidate",
    "candidate",
    "reference",
    "reference",
    "candidate",
    "candidate",
    "reference",
]
ROWS_PER_CALL = 68
TOKENS_PER_ROW = 64
MEMORY_TOTAL_MIB = 97_887
CAPACITY_TOKENS = 162_669
CAPACITY_MINIMUM_TOKENS = 161_792
RUNTIME_MAX_SECONDS = 50.55591316620558
RUNTIME_RATIO_TO_V78_MAX = 1.03
PEAK_MEMORY_MAX_MIB = 50_808
MINIMUM_HEADROOM_MIB = 47_079

PARENT_ANALYZER = ROOT / "analyze_qwen36_bf16_kv_mamba_capacity_v80.py"
PARENT_ANALYZER_SHA256 = (
    "879f4bae9c1dadae1d0527cdfe2abb421b0b198bdf3b174fa180e93b36ca6cdb"
)
PARENT_RESULT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_bf16_kv_mamba_capacity_v80_postrun_20260717.json"
)
PARENT_RESULT_FILE_SHA256 = (
    "bc3d94666f0b5ec27b8ab427304204eca5480450c0076c54da77a14cc2a0cae2"
)
PARENT_RESULT_CONTENT_SHA256 = (
    "f59870ef504fcab5f80dd424c38f7e3741886add683a1059c270e8eb257d729f"
)
V79B_RESULT = ROOT / (
    "experiments/eggroll_es_hpo/qwen36_fp8_kv_capacity_v79b_postrun_20260717.json"
)
V79B_RESULT_FILE_SHA256 = (
    "fd458aa6f2416df73578f3153b99480534f8513d9cabaf95438aaf6fc346e775"
)
V79B_RESULT_CONTENT_SHA256 = (
    "dc2c3f47f28bd74bec0ddb385652c6263d38328f637f31c6769b1e48277ed46a"
)
V80B_PREREG = PREREG_ROOT / "qwen36_bf16_kv_mamba_confirmation_v80b.json"
V80B_PREREG_FILE_SHA256 = (
    "9d4c85f594946dc9f200b064ce79ee989cd218a17879c6786da85405d35d6724"
)
V80B_PREREG_CONTENT_SHA256 = (
    "8515a4680175a68233f2ff408bd0439daebf1a8a4c94a842c45e08ac9ec6b976"
)
V83_BUILDER = ROOT / "build_qwen36_tp1_process_group_cleanup_v83.py"
V83_BUILDER_SHA256 = (
    "5aa7252fa4e8bf998a7b9588d261df79fcc61910d61c0eb021c3ecc322e556ce"
)
V83_RESULT = PREREG_ROOT / "qwen36_tp1_process_group_cleanup_v83.json"
V83_RESULT_FILE_SHA256 = (
    "768d199915c2110fa1170af506147975a23cdfb87a19876e1f5735a716022dd1"
)
V83_RESULT_CONTENT_SHA256 = (
    "45957ac5f53004456862596baacc09d95f0179995436e745eea7dd970e1a91de"
)
LEGACY_SOURCE = ROOT / "probe_vllm_two_adapter_switch_v63.py"
LEGACY_SOURCE_SHA256 = (
    "115774a63f54480fa4796f24f5b47a82fda1c2a761db4cf3ae0b6b83e85165d6"
)

OVERRIDE_WARNING = (
    "Qwen3.5 model specifies mamba_ssm_dtype='float32' in its config, but "
    "--mamba-ssm-cache-dtype='bfloat16' was passed. Using the user-specified value."
)
REQUIRED_LOG_FRAGMENTS = (
    "Using TRITON Fp8 MoE backend",
    "Using FLASH_ATTN attention backend",
    OVERRIDE_WARNING,
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
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256_v80b(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256_v80b(path: Path) -> str:
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
        isinstance(claimed, str) and claimed == canonical_sha256_v80b(body),
        f"self hash changed: {label}",
    )


def _validate_bound_json(
    path: Path, file_sha256: str, content_sha256: str, schema: str
) -> dict[str, Any]:
    _require(
        path.is_file()
        and not path.is_symlink()
        and file_sha256_v80b(path) == file_sha256,
        f"sealed JSON file changed: {path.relative_to(ROOT)}",
    )
    value = _load_json(path)
    _validate_self_hash(value, str(path.relative_to(ROOT)))
    _require(
        value.get("schema") == schema
        and value.get("content_sha256_before_self_field") == content_sha256,
        f"sealed JSON content changed: {path.relative_to(ROOT)}",
    )
    return value


def _inventory(run: str) -> dict[str, Any]:
    directory = RUN_ROOT / run
    _require(
        directory.is_dir() and not directory.is_symlink(), f"run missing: {run}"
    )
    paths = sorted(directory.iterdir())
    _require(
        len(paths) == 11
        and all(path.is_file() and not path.is_symlink() for path in paths),
        f"run artifact cardinality/type changed: {run}",
    )
    rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256_v80b(path),
        }
        for path in paths
    ]
    bundle = canonical_sha256_v80b(rows)
    _require(bundle == RUN_BUNDLES[run], f"sealed run bundle changed: {run}")
    return {"file_count": 11, "bundle_sha256": bundle, "files": rows}


def _load_parent_helpers():
    _require(
        file_sha256_v80b(PARENT_ANALYZER) == PARENT_ANALYZER_SHA256,
        "sealed V80 parent analyzer changed",
    )
    return importlib.import_module("analyze_qwen36_bf16_kv_mamba_capacity_v80")


def _validate_static_contract(parent: Any) -> dict[str, Any]:
    parent_static = parent.validate_static_contract_v80()
    parent_result = _validate_bound_json(
        PARENT_RESULT,
        PARENT_RESULT_FILE_SHA256,
        PARENT_RESULT_CONTENT_SHA256,
        "v80-qwen36-bf16-kv-mamba-capacity-postrun-analysis",
    )
    _require(
        parent_result.get("candidate_parent_contract_passed") is False
        and parent_result.get("promotion_authorized") is False
        and parent_result.get("parent_gate_evaluation", {}).get(
            "literal_parent_data_free_contract_passed"
        )
        is False,
        "immutable V80 parent result changed",
    )
    controls = _validate_bound_json(
        V79B_RESULT,
        V79B_RESULT_FILE_SHA256,
        V79B_RESULT_CONTENT_SHA256,
        "qwen36-fp8-kv-capacity-postrun-analysis-v79b",
    )
    v80b = _validate_bound_json(
        V80B_PREREG,
        V80B_PREREG_FILE_SHA256,
        V80B_PREREG_CONTENT_SHA256,
        "v80b-qwen36-bf16-kv-mamba-two-run-confirmation-preregistration",
    )
    v83 = _validate_bound_json(
        V83_RESULT,
        V83_RESULT_FILE_SHA256,
        V83_RESULT_CONTENT_SHA256,
        "v83-qwen36-tp1-process-group-cleanup-amendment",
    )
    for path, expected, label in (
        (V83_BUILDER, V83_BUILDER_SHA256, "V83 builder"),
        (LEGACY_SOURCE, LEGACY_SOURCE_SHA256, "legacy shutdown source"),
    ):
        _require(
            path.is_file()
            and not path.is_symlink()
            and file_sha256_v80b(path) == expected,
            f"sealed {label} changed",
        )
    _require(
        [row["run"] for row in v80b["confirmatory_runs"]] == list(RUNS[1:])
        and [row["ordinal"] for row in v80b["confirmatory_runs"]] == [2, 3]
        and v80b["confirmation_analysis"][
            "evaluate_each_run_independently_against_unchanged_parent_gates"
        ]
        is True
        and v80b["confirmation_analysis"]["promotion_default"] is False
        and v80b["confirmation_analysis"][
            "semantic_and_protected_ood_gates_remain_pending"
        ]
        is True,
        "V80B prospective confirmation contract changed",
    )
    _require(
        v83["immutable_parent_result"]
        == {
            "literal_torch_process_group_destroyed_clause_passed": False,
            "reason": "the legacy field name does not match its source semantics",
            "result_rewritten": False,
        }
        and v83["legacy_source_binding"]["file_sha256"]
        == LEGACY_SOURCE_SHA256
        and v83["legacy_source_binding"]["actual_value_semantics"]
        == "dist_is_initialized_before_shutdown"
        and v83["additive_v83_verdict"]["cleanup_semantics_passed"] is True
        and v83["additive_v83_verdict"][
            "semantic_ood_and_promotion_gates_still_pending"
        ]
        is True,
        "V83 additive interpretation changed",
    )
    return {
        "parent_static_contract": parent_static,
        "bound_files": {
            "v80_parent_analyzer": {
                "path": str(PARENT_ANALYZER.relative_to(ROOT)),
                "file_sha256": PARENT_ANALYZER_SHA256,
            },
            "v80_parent_result": {
                "path": str(PARENT_RESULT.relative_to(ROOT)),
                "file_sha256": PARENT_RESULT_FILE_SHA256,
                "content_sha256": PARENT_RESULT_CONTENT_SHA256,
            },
            "v79b_control_result": {
                "path": str(V79B_RESULT.relative_to(ROOT)),
                "file_sha256": V79B_RESULT_FILE_SHA256,
                "content_sha256": V79B_RESULT_CONTENT_SHA256,
            },
            "v80b_preregistration": {
                "path": str(V80B_PREREG.relative_to(ROOT)),
                "file_sha256": V80B_PREREG_FILE_SHA256,
                "content_sha256": V80B_PREREG_CONTENT_SHA256,
            },
            "v83_additive_interpretation": {
                "path": str(V83_RESULT.relative_to(ROOT)),
                "file_sha256": V83_RESULT_FILE_SHA256,
                "content_sha256": V83_RESULT_CONTENT_SHA256,
                "builder_file_sha256": V83_BUILDER_SHA256,
                "legacy_source_file_sha256": LEGACY_SOURCE_SHA256,
            },
        },
        "v80_parent_result": parent_result,
        "v79b_controls": controls,
        "v80b_preregistration": v80b,
        "v83": v83,
    }


def _validate_actor(parent: Any, run: str, gpu: int) -> tuple[dict[str, Any], dict[str, Any]]:
    path = RUN_ROOT / run / f"gpu_{gpu}.json"
    receipt = _load_json(path)
    label = f"{run}:gpu{gpu}"
    parent._validate_self_hash(receipt, label)
    parent._validate_hash_rows(receipt, label)
    residency = parent._validate_residency(receipt, label)
    performance = parent._validate_generation_performance(
        receipt.get("generation_performance", {}),
        "v80-data-free-generation-performance",
        label,
    )
    runtime = receipt.get("runtime", {})
    cache = receipt.get("resolved_hybrid_cache_certificate", {})
    memory = receipt.get("resolved_memory_budget_certificate", {})
    precision = receipt.get("resolved_precision_certificate", {})
    gates = receipt.get("preflight_gates", {})
    _require(
        receipt.get("schema")
        == "v80-qwen36-bf16-kv-mamba-capacity-matched-preflight"
        and receipt.get("actor_label") == f"gpu-{gpu}"
        and receipt.get("source_dataset_rows_opened") == 0
        and receipt.get("protected_ood_shadow_or_terminal_opened") is False
        and receipt.get("adapter_update_or_hpo_performed") is False
        and receipt.get("prompt_or_generation_text_persisted") is False
        and receipt.get("token_ids_persisted") is False
        and receipt.get("engine_shutdown_completed") is True
        and receipt.get("torch_process_group_destroyed") is False,
        f"actor authority/legacy cleanup changed: {label}",
    )
    _require(
        runtime.get("gpu_memory_utilization") == 0.479
        and runtime.get("resolved_quantization") == "fp8"
        and runtime.get("serialized_fp8_checkpoint") is True
        and runtime.get("mamba_ssm_cache_dtype") == "bfloat16"
        and runtime.get("starting_moe_tuning_table") == "fresh_empty_default"
        and runtime.get("vllm_version") == "0.25.0"
        and runtime.get("enforce_eager") is True
        and runtime.get("async_scheduling") is False
        and runtime.get("scheduling_policy") == "fcfs"
        and runtime.get("max_num_seqs") == ROWS_PER_CALL
        and runtime.get("max_loras") == 1
        and runtime.get("max_cpu_loras") == 2
        and runtime.get("enable_flashinfer_autotune") is False,
        f"V80 runtime changed: {label}",
    )
    _require(
        precision.get("resolved_from_live_engine") is True
        and precision.get("model_quantization") == "fp8"
        and precision.get("quant_config_class") == "Fp8Config"
        and precision.get("weight_block_size") == [128, 128]
        and memory
        == {"gpu_memory_utilization": 0.479, "resolved_from_live_engine": True},
        f"V80 precision or memory certificate changed: {label}",
    )
    _require(
        cache.get("resolved_from_live_engine") is True
        and cache.get("gpu_memory_utilization") == 0.479
        and cache.get("cache_dtype") == "auto"
        and cache.get("mamba_cache_dtype") == "auto"
        and cache.get("mamba_ssm_cache_dtype") == "bfloat16"
        and cache.get("model_quantization") == "fp8"
        and cache.get("block_size") == 544
        and cache.get("num_gpu_blocks") == 556
        and cache.get("kv_cache_size_tokens") == CAPACITY_TOKENS
        and math.isclose(
            cache.get("kv_cache_max_concurrency"),
            79.42857142857143,
            rel_tol=0,
            abs_tol=1e-12,
        )
        and receipt.get("preregistration_v80")
        == {
            "content_sha256": "7527ed6fe0154a79ecc0de46b00af4601b0e3deaac184f2af094fba15740149a",
            "file_sha256": "be7de6c8ac1d59feef0ec0d0e289be85612644efbe72fdb3847fa34d5dd50aad",
        }
        and receipt.get("single_variable_change_from_v78c")
        == {"gpu_memory_utilization": [0.5, 0.479]}
        and receipt.get("v78c_reference_identity", {}).get("run_bundle_sha256")
        == "a9d82f71bb6beecc420be135737f2048fb770bee5d97309f9e90da4e31ef833f",
        f"V80 cache or ancestry changed: {label}",
    )
    _require(
        gates.get("candidate_repeat_exact_at_token_hash_level") is True
        and gates.get("candidate_changes_output") is True
        and gates.get("reference_restore_exact_at_token_hash_level") is False
        and gates.get("kv_capacity_minimum_tokens_passed") is True
        and gates.get("bf16_attention_kv_auto_resolved") is True
        and gates.get("bf16_mamba_ssm_cache_resolved") is True
        and gates.get("scored_evaluation_or_training_authorized") is False
        and receipt.get("authority")
        == {
            "data_free_hybrid_cache_capacity_diagnostic_only": True,
            "scored_evaluation_training_checkpoint_or_promotion_allowed": False,
        },
        f"V80 candidate/authority gate changed: {label}",
    )
    wall = receipt.get("wall_runtime_seconds_excluding_model_load_and_cleanup")
    _require(
        isinstance(wall, (int, float))
        and not isinstance(wall, bool)
        and math.isfinite(wall)
        and wall > 0,
        f"actor runtime changed: {label}",
    )
    return (
        {
            "gpu": gpu,
            "receipt_file_sha256": file_sha256_v80b(path),
            "receipt_content_sha256": receipt[
                "content_sha256_before_self_field"
            ],
            "wall_runtime_seconds": wall,
            "reference_repeat_changed_rows": receipt[
                "reference_within_state_changed_rows"
            ],
            "candidate_repeat_changed_rows": receipt[
                "candidate_within_state_changed_rows"
            ],
            "between_state_differing_rows": receipt[
                "between_state_differing_rows"
            ],
            "candidate_receipt_passed": (
                receipt["candidate_within_state_changed_rows"] == 0
                and receipt["between_state_differing_rows"] > 0
            ),
            "engine_shutdown_completed": True,
            "legacy_torch_process_group_field_value": False,
            "generation_performance": performance,
            "backend_receipt": {
                "routed_moe_backend": "TRITON",
                "attention_backend_expected_from_log": "FLASH_ATTN",
                "fp8_moe_method_count": 40,
                "parameter_count": residency["total_parameter_count"],
                "logical_bytes": residency["total_logical_bytes"],
                "passed": True,
            },
        },
        receipt,
    )


def _validate_log(run: str, gpu: int) -> dict[str, Any]:
    path = RUN_ROOT / run / f"gpu_{gpu}.log"
    log = path.read_text(encoding="utf-8", errors="strict")
    required_counts = {fragment: log.count(fragment) for fragment in REQUIRED_LOG_FRAGMENTS}
    forbidden_counts = {
        fragment: log.lower().count(fragment.lower())
        for fragment in FORBIDDEN_LOG_FRAGMENTS
    }
    _require(
        all(count == 1 for count in required_counts.values()),
        f"required backend/capacity log cardinality changed: {run}:gpu{gpu}",
    )
    _require(
        all(count == 0 for count in forbidden_counts.values()),
        f"forbidden fallback/error found: {run}:gpu{gpu}",
    )
    tokens = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
    concurrency = re.findall(
        r"Maximum concurrency for 2,048 tokens per request: ([\d.]+)x", log
    )
    available = re.findall(r"Available KV cache memory: ([\d.]+) GiB", log)
    _require(
        len(tokens) == len(concurrency) == len(available) == 1
        and int(tokens[0].replace(",", "")) == CAPACITY_TOKENS
        and math.isclose(float(concurrency[0]), 79.43, rel_tol=0, abs_tol=0.005),
        f"capacity log changed: {run}:gpu{gpu}",
    )
    return {
        "gpu": gpu,
        "file_sha256": file_sha256_v80b(path),
        "attention_backend": "FLASH_ATTN",
        "routed_moe_backend": "TRITON",
        "available_kv_gib": float(available[0]),
        "kv_cache_size_tokens": CAPACITY_TOKENS,
        "maximum_concurrency": float(concurrency[0]),
        "required_fragment_counts": required_counts,
        "forbidden_fragment_counts": forbidden_counts,
        "fallback_receipt_passed": True,
        "backend_log_receipt_passed": True,
    }


def _read_pid_map(run: str) -> dict[int, int]:
    lines = (RUN_ROOT / run / "actor_pids.csv").read_text(encoding="ascii").splitlines()
    _require(len(lines) == 4, f"PID map cardinality changed: {run}")
    result: dict[int, int] = {}
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


def _validate_telemetry(run: str) -> dict[str, Any]:
    path = RUN_ROOT / run / "gpu_telemetry_v80.jsonl"
    rows = _load_jsonl(path)
    pid_map = _read_pid_map(run)
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
            f"telemetry receipt changed: {run}",
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
            pid for row in gpu_rows for pid in row.get("attributed_compute_pids", [])
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
        "file_sha256": file_sha256_v80b(path),
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


def _validate_v83_run_binding(
    v83: Mapping[str, Any], run: str, inventory: Mapping[str, Any],
    actors: Sequence[Mapping[str, Any]], telemetry: Mapping[str, Any]
) -> dict[str, Any]:
    bound = v83.get("runs", {}).get(run, {})
    _require(
        bound.get("bundle_sha256") == inventory["bundle_sha256"]
        and bound.get("artifact_inventory") == inventory["files"],
        f"V83 run identity does not match sealed V80B run: {run}",
    )
    v83_actors = bound.get("actor_receipts")
    _require(
        isinstance(v83_actors, list)
        and len(v83_actors) == 4
        and all(
            v83_actor.get("gpu") == actor["gpu"]
            and v83_actor.get("receipt_file_sha256")
            == actor["receipt_file_sha256"]
            and v83_actor.get("receipt_content_sha256")
            == actor["receipt_content_sha256"]
            and v83_actor.get("legacy_field_value") is False
            and v83_actor.get("source_bound_interpretation")
            == {
                "destroy_process_group_attempted": False,
                "destroy_was_required": False,
                "process_group_initialized_before_shutdown": False,
            }
            for v83_actor, actor in zip(v83_actors, actors)
        ),
        f"V83 actor interpretation does not bind all four actors: {run}",
    )
    cleanup = bound.get("external_cleanup", {})
    _require(
        cleanup.get("telemetry_file_sha256") == telemetry["file_sha256"]
        and cleanup.get("minimum_consecutive_idle_batches") == 3
        and [row["batch_index"] for row in cleanup["accepted_final_batches"]]
        == telemetry["trailing_external_cleanup_batches"][-3:]
        and all(
            row["actor_roots_dead"] is True
            and row["compute_pids"] == []
            and row["foreign_compute_pids"] == []
            and row["external_memory_used_mib_max"] <= 4
            and row["external_gpu_utilization_percent_max"] == 0
            for row in cleanup["accepted_final_batches"]
        ),
        f"V83 external cleanup binding changed: {run}",
    )
    return {
        "legacy_field_values": [False] * 4,
        "process_group_initialized_before_shutdown": [False] * 4,
        "destroy_process_group_attempted": [False] * 4,
        "destroy_was_required": [False] * 4,
        "external_cleanup_final_batches": telemetry[
            "trailing_external_cleanup_batches"
        ][-3:],
        "additive_cleanup_semantics_passed": True,
        "immutable_parent_literal_result_rewritten": False,
    }


def _validate_run(parent: Any, v83: Mapping[str, Any], run: str) -> dict[str, Any]:
    inventory = _inventory(run)
    actors = []
    receipts = []
    logs = []
    for gpu in GPU_IDS:
        actor, receipt = _validate_actor(parent, run, gpu)
        actors.append(actor)
        receipts.append(receipt)
        logs.append(_validate_log(run, gpu))
    telemetry = _validate_telemetry(run)
    v83_binding = _validate_v83_run_binding(v83, run, inventory, actors, telemetry)
    walls = [actor["wall_runtime_seconds"] for actor in actors]
    peak = max(row["peak_memory_used_mib"] for row in telemetry["per_gpu"].values())
    process_peak = max(
        row["peak_attributed_process_memory_mib"]
        for row in telemetry["per_gpu"].values()
    )
    return {
        "run": run,
        "artifact_inventory": inventory,
        "actors": actors,
        "actor_wall_runtime_seconds_median": statistics.median(walls),
        "actor_wall_runtime_seconds_range": [min(walls), max(walls)],
        "capacity_tokens_per_actor": CAPACITY_TOKENS,
        "logs": logs,
        "telemetry": telemetry,
        "external_peak_memory_used_mib": peak,
        "peak_attributed_actor_process_memory_mib": process_peak,
        "v83_process_group_and_cleanup_binding": v83_binding,
        "receipts": receipts,
    }


def _strip_receipts(run: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(run)
    value.pop("receipts", None)
    return value


def _paired_drift(parent: Any, left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    return parent.paired_drift_matrix_v80(left, right, label)


def _control_comparison(controls: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    capacity = controls["capacity"]
    performance = controls["performance"]
    vram = controls["vram"]
    walls = [actor["wall_runtime_seconds"] for run in runs for actor in run["actors"]]
    v80_median = statistics.median(walls)
    v80_range = [min(walls), max(walls)]
    v80_peak = max(run["external_peak_memory_used_mib"] for run in runs)
    v80_process_peak = max(
        run["peak_attributed_actor_process_memory_mib"] for run in runs
    )
    control_rows = {
        "v76": {
            "capacity_tokens_per_actor": capacity["v76_bf16_attention_kv_tokens"],
            "combined_actor_median_seconds": performance["v76_three_replicates"][
                "combined_actor_median_seconds"
            ],
            "combined_actor_range_seconds": performance["v76_three_replicates"][
                "combined_actor_range_seconds"
            ],
            "external_peak_memory_used_mib": vram["external_peak_memory_used_mib"][
                "v76_r7"
            ],
        },
        "v78": {
            "capacity_tokens_per_actor": capacity["v78_fp8_attention_kv_tokens"],
            "combined_actor_median_seconds": performance["v78_three_replicates"][
                "combined_actor_median_seconds"
            ],
            "combined_actor_range_seconds": performance["v78_three_replicates"][
                "combined_actor_range_seconds"
            ],
            "external_peak_memory_used_mib": vram["external_peak_memory_used_mib"][
                "v78_r3"
            ],
        },
        "v79b": {
            "capacity_tokens_per_actor": capacity[
                "v79b_capacity_matched_fp8_attention_kv_tokens"
            ],
            "combined_actor_median_seconds": performance[
                "v79b_cleanup_accepted_run_median_seconds"
            ],
            "combined_actor_range_seconds": None,
            "external_peak_memory_used_mib": vram["external_peak_memory_used_mib"][
                "v79b_r5_including_pynvml_observer"
            ],
        },
    }
    _require(
        control_rows
        == {
            "v76": {
                "capacity_tokens_per_actor": 157_696,
                "combined_actor_median_seconds": 48.58432143297978,
                "combined_actor_range_seconds": [48.255868688982446, 49.15785783302272],
                "external_peak_memory_used_mib": 50_858,
            },
            "v78": {
                "capacity_tokens_per_actor": 198_656,
                "combined_actor_median_seconds": 49.10173862800002,
                "combined_actor_range_seconds": [48.49649476300692, 51.4921482350328],
                "external_peak_memory_used_mib": 50_856,
            },
            "v79b": {
                "capacity_tokens_per_actor": 162_304,
                "combined_actor_median_seconds": 49.10028860197053,
                "combined_actor_range_seconds": None,
                "external_peak_memory_used_mib": 49_994,
            },
        },
        "sealed V76/V78/V79B aggregate controls changed",
    )
    return {
        "control_source": {
            "path": str(V79B_RESULT.relative_to(ROOT)),
            "file_sha256": V79B_RESULT_FILE_SHA256,
            "content_sha256": V79B_RESULT_CONTENT_SHA256,
        },
        "v80b": {
            "actor_count": 12,
            "capacity_tokens_per_actor": CAPACITY_TOKENS,
            "combined_actor_median_seconds": v80_median,
            "combined_actor_range_seconds": v80_range,
            "run_medians_seconds": {
                run["run"]: run["actor_wall_runtime_seconds_median"] for run in runs
            },
            "external_peak_memory_used_mib": v80_peak,
            "run_external_peaks_mib": {
                run["run"]: run["external_peak_memory_used_mib"] for run in runs
            },
            "peak_attributed_actor_process_memory_mib": v80_process_peak,
        },
        "controls": control_rows,
        "capacity_delta_tokens": {
            name: CAPACITY_TOKENS - row["capacity_tokens_per_actor"]
            for name, row in control_rows.items()
        },
        "capacity_ratio": {
            name: CAPACITY_TOKENS / row["capacity_tokens_per_actor"]
            for name, row in control_rows.items()
        },
        "runtime_ratio": {
            name: v80_median / row["combined_actor_median_seconds"]
            for name, row in control_rows.items()
        },
        "external_peak_memory_savings_mib": {
            name: row["external_peak_memory_used_mib"] - v80_peak
            for name, row in control_rows.items()
        },
        "observer_caveat": (
            "V80B and V79B external totals include an in-process pynvml observer; "
            "V76/V78 use the older external monitor. Exact MiB differences are "
            "descriptive across monitor generations, not a causal attribution."
        ),
    }


def analyze_v80b() -> dict[str, Any]:
    parent = _load_parent_helpers()
    static = _validate_static_contract(parent)
    runs = [_validate_run(parent, static["v83"], run) for run in RUNS]
    v78c = parent._validate_run(parent.V78C_RUN)
    drift = {}
    for run in runs:
        drift[f"v78c_r1_vs_{run['run']}"] = _paired_drift(
            parent,
            v78c["receipts"],
            run["receipts"],
            f"v78c_r1_vs_{run['run']}",
        )
    for left, right in zip(runs, runs[1:]):
        key = f"{left['run']}_vs_{right['run']}"
        drift[key] = _paired_drift(
            parent, left["receipts"], right["receipts"], key
        )

    v78c_median = v78c["actor_wall_runtime_seconds_median"]
    independent = {}
    for run in runs:
        telemetry = run["telemetry"]
        actors = run["actors"]
        capacity_pass = all(
            log["kv_cache_size_tokens"] >= CAPACITY_MINIMUM_TOKENS
            for log in run["logs"]
        )
        runtime_pass = (
            run["actor_wall_runtime_seconds_median"] <= RUNTIME_MAX_SECONDS
            and run["actor_wall_runtime_seconds_median"] / v78c_median
            <= RUNTIME_RATIO_TO_V78_MAX
        )
        memory_pass = all(
            row["peak_memory_used_mib"] <= PEAK_MEMORY_MAX_MIB
            and row["minimum_physical_headroom_mib"] >= MINIMUM_HEADROOM_MIB
            for row in telemetry["per_gpu"].values()
        )
        output_pass = all(actor["candidate_receipt_passed"] for actor in actors)
        literal_pg_pass = all(
            actor["legacy_torch_process_group_field_value"] is True
            for actor in actors
        )
        gates = {
            "cardinality_identity_hash_only_and_no_data_access": True,
            "capacity_live_field_log_and_minimum": capacity_pass,
            "hybrid_cache_precision_backend_residency_and_warning": True,
            "required_and_forbidden_logs": True,
            "runtime_and_memory": runtime_pass and memory_pass,
            "four_gpu_activity_and_pcie_telemetry": telemetry[
                "all_four_gpus_useful"
            ],
            "candidate_output_and_paired_hash_matrix": output_pass,
            "engine_shutdown_completed_per_actor": all(
                actor["engine_shutdown_completed"] for actor in actors
            ),
            "external_three_batch_post_exit_cleanup": telemetry[
                "external_cleanup_gate_passed"
            ],
            "torch_process_group_destroyed_per_actor_literal_true": literal_pg_pass,
        }
        _require(
            all(
                passed
                for name, passed in gates.items()
                if name != "torch_process_group_destroyed_per_actor_literal_true"
            )
            and literal_pg_pass is False,
            f"unexpected independent parent gate result: {run['run']}",
        )
        independent[run["run"]] = {
            "data_free_gate_results": gates,
            "all_data_free_gates_except_literal_legacy_clause_passed": True,
            "immutable_literal_parent_contract_passed": False,
            "additive_v83_cleanup_semantics_passed": True,
            "immutable_parent_result_rewritten": False,
            "source_disjoint_semantic_gate_run": False,
            "protected_one_shot_ood_gate_run": False,
            "promotion_authorized": False,
        }

    comparison = _control_comparison(static["v79b_controls"], runs)
    receipt_matrix = []
    for run in runs:
        for actor, log in zip(run["actors"], run["logs"]):
            receipt_matrix.append(
                {
                    "run": run["run"],
                    "gpu": actor["gpu"],
                    "actor_receipt_validated": True,
                    "candidate_receipt_passed": actor["candidate_receipt_passed"],
                    "engine_shutdown_receipt_passed": actor[
                        "engine_shutdown_completed"
                    ],
                    "external_cleanup_receipt_passed": run["telemetry"][
                        "external_cleanup_gate_passed"
                    ],
                    "legacy_literal_process_group_clause_passed": False,
                    "v83_additive_process_group_cleanup_passed": True,
                    "fallback_receipt_passed": log["fallback_receipt_passed"],
                    "backend_receipt_passed": (
                        actor["backend_receipt"]["passed"]
                        and log["backend_log_receipt_passed"]
                    ),
                }
            )
    _require(
        len(receipt_matrix) == 12
        and all(
            row["actor_receipt_validated"]
            and row["candidate_receipt_passed"]
            and row["engine_shutdown_receipt_passed"]
            and row["external_cleanup_receipt_passed"]
            and row["v83_additive_process_group_cleanup_passed"]
            and row["fallback_receipt_passed"]
            and row["backend_receipt_passed"]
            and not row["legacy_literal_process_group_clause_passed"]
            for row in receipt_matrix
        ),
        "12-actor receipt matrix changed",
    )

    result = {
        "schema": SCHEMA,
        "status": (
            "three_sealed_runs_validated_additive_v83_cleanup_passed_"
            "immutable_parent_literal_result_preserved_semantic_ood_pending_"
            "no_promotion"
        ),
        "authority": {
            "cpu_sealed_file_analysis_only": True,
            "dataset_prompt_generated_text_or_protected_data_opened": False,
            "gpu_or_model_launch_performed_by_analyzer": False,
            "model_adapter_training_or_checkpoint_update_performed": False,
            "scored_training_checkpoint_or_runtime_promotion_authorized": False,
        },
        "static_contract": {
            "parent_static_contract": static["parent_static_contract"],
            "bound_files": static["bound_files"],
        },
        "evidence_classification": {
            RUNS[0]: "observed before V80B and retained as immutable parent evidence",
            RUNS[1]: "prospective V80B confirmation run 2",
            RUNS[2]: "prospective V80B confirmation run 3",
            "v83": (
                "additive source-bound TP1 cleanup interpretation; immutable "
                "V80 parent result is not rewritten"
            ),
            "controls": "sealed V79B aggregate values for V76/V78/V79B",
        },
        "runs": {run["run"]: _strip_receipts(run) for run in runs},
        "twelve_actor_receipt_matrix": receipt_matrix,
        "paired_token_hash_drift": drift,
        "independent_parent_gate_evaluation": independent,
        "combined_comparison": comparison,
        "aggregate_verdict": {
            "sealed_run_bundle_count": 3,
            "validated_actor_count": 12,
            "candidate_receipt_count": 12,
            "cleanup_receipt_count": 12,
            "fallback_receipt_count": 12,
            "backend_receipt_count": 12,
            "all_nonlegacy_data_free_receipts_passed": True,
            "immutable_parent_literal_contract_passed": False,
            "immutable_parent_result_rewritten": False,
            "additive_v83_cleanup_semantics_passed": True,
            "source_disjoint_semantic_gate_run": False,
            "protected_one_shot_ood_gate_run": False,
            "promotion_authorized": False,
        },
        "bandwidth_claim": {
            "pcie_integrals_are_estimates": True,
            "hbm_bandwidth_bytes_per_second": None,
            "hbm_bandwidth_not_inferred": True,
            "reason": (
                "NVML memory-utilization percentages do not measure HBM bytes/s"
            ),
        },
        "analysis_completed": True,
        "candidate_parent_contract_passed": False,
        "promotion_authorized": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v80b(result)
    return result


def render_json_v80b(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False
    ) + "\n"


def render_report_v80b(value: Mapping[str, Any]) -> str:
    comparison = value["combined_comparison"]
    v80b = comparison["v80b"]
    controls = comparison["controls"]
    run_lines = []
    for run_name, run in value["runs"].items():
        gate = value["independent_parent_gate_evaluation"][run_name]
        run_lines.append(
            f"| {run_name} | {run['artifact_inventory']['bundle_sha256']} | "
            f"{run['actor_wall_runtime_seconds_median']:.3f} s | "
            f"{run['external_peak_memory_used_mib']:,} MiB | "
            f"PASS | FAIL | PASS |"
        )
        _require(
            gate["all_data_free_gates_except_literal_legacy_clause_passed"] is True,
            f"report received failed nonlegacy gate: {run_name}",
        )
    comparison_lines = []
    rows = {"V80B": v80b, **{name.upper(): row for name, row in controls.items()}}
    for name, row in rows.items():
        capacity = row["capacity_tokens_per_actor"]
        median = row["combined_actor_median_seconds"]
        peak = row["external_peak_memory_used_mib"]
        actor_count = row.get("actor_count", "sealed control")
        comparison_lines.append(
            f"| {name} | {actor_count} | {capacity:,} | {capacity // 2048} | "
            f"{median:.3f} s | {peak:,} MiB |"
        )
    deltas = comparison["capacity_delta_tokens"]
    runtime = comparison["runtime_ratio"]
    savings = comparison["external_peak_memory_savings_mib"]
    return f"""# Qwen3.6 V80B three-run aggregate evidence

## Outcome

All three sealed V80 bundles and all 12 actor, candidate, cleanup, fallback,
and backend receipt sets validate.  Each run independently passes every V80
data-free parent gate except the immutable literal requirement that
`torch_process_group_destroyed` be true.  That historical result remains a
failure and is not rewritten.

V83 binds the exact shutdown source and the same 12 receipts: the legacy field
actually recorded `dist.is_initialized()` before shutdown.  All values are
false, so TP1 never initialized a process group and no destroy call was
required.  Every run independently ends with at least three external batches
showing dead actor roots, no compute or foreign PID, 0% GPU utilization, and at
most 4 MiB.  The additive cleanup interpretation therefore passes.  Semantic
and protected OOD evaluation remain unopened, so promotion remains false.

| Run | sealed bundle SHA-256 | four-actor median | worst external peak | nonlegacy gates | literal parent PG | additive V83 cleanup |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(run_lines)}

## Timing, capacity, and VRAM

| Arm | actors | cache tokens / actor | full 2048 contexts | actor median | external peak |
|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison_lines)}

Across 12 V80B actors, the median is
{v80b['combined_actor_median_seconds']:.3f} seconds and the range is
{v80b['combined_actor_range_seconds'][0]:.3f}-{v80b['combined_actor_range_seconds'][1]:.3f}
seconds.  Relative to the sealed control aggregates, V80B is
{(1-runtime['v76'])*100:.2f}% faster than V76,
{(1-runtime['v78'])*100:.2f}% faster than V78, and
{(1-runtime['v79b'])*100:.2f}% faster than the cleanup-accepted V79B run.

V80B provides {deltas['v76']:+,} tokens versus V76,
{deltas['v78']:+,} versus V78, and {deltas['v79b']:+,} versus V79B.  Its worst
external peak is {v80b['external_peak_memory_used_mib']:,} MiB, for descriptive
savings of {savings['v76']:+,} MiB versus V76, {savings['v78']:+,} MiB versus
V78, and {savings['v79b']:+,} MiB versus V79B.  V80B and V79B totals include an
in-process NVML observer, while V76/V78 use an older external monitor, so exact
cross-generation MiB differences are descriptive rather than causal.

PCIe RX/TX values in the machine report are left-rectangle estimates from
sampled rates.  HBM bandwidth is intentionally absent: NVML memory-utilization
percentages do not measure bytes per second.

## Evidence boundaries

- V80B preregistration: `{V80B_PREREG_CONTENT_SHA256}`
- V83 additive interpretation: `{V83_RESULT_CONTENT_SHA256}`
- V79B aggregate control: `{V79B_RESULT_CONTENT_SHA256}`
- Aggregate analysis: `{value['content_sha256_before_self_field']}`
- Dataset, prompts, generated text, token IDs, protected data, model, and GPU
  access by this analyzer: none
- Semantic gate run: false
- Protected one-shot OOD gate run: false
- Promotion authorized: false
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = analyze_v80b()
    output = render_json_v80b(value)
    report = render_report_v80b(value)
    if args.check:
        _require(OUTPUT.read_text(encoding="ascii") == output, "V80B JSON is stale")
        _require(REPORT.read_text(encoding="ascii") == report, "V80B report is stale")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(output, encoding="ascii")
        REPORT.write_text(report, encoding="ascii")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "report": str(REPORT),
                "content_sha256": value["content_sha256_before_self_field"],
                "validated_actor_count": 12,
                "promotion_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
