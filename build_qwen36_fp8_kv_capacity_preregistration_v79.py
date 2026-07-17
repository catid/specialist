#!/usr/bin/env python3
"""Build the sealed CPU-only V79 hybrid-KV capacity preregistration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fp8_kv_capacity_matched_v79.json"
)
V76_RUN = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v76_fp8_attested_050_r7_residency"
)
V78_RUN = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v78_fp8_per_token_head_kv_r3"
)
HARDWARE_TELEMETRY = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v66d_lora_es_mirrored_crn_qwen36_calibration/"
    "gpu_activity_v66d.jsonl"
)
V73_PROBE = ROOT / "probe_vllm_quantized_adapter_switch_v73.py"
V76_PROBE = ROOT / "probe_vllm_fp8_attested_v76.py"
V78_PROBE = ROOT / "probe_vllm_fp8_kv_cache_v78.py"
LAUNCHER = ROOT / "launch_qwen36_fp8_kv_capacity_v79.sh"
MONITOR = ROOT / "monitor_qwen36_fp8_kv_capacity_v79.py"

SCHEMA = "v79-qwen36-fp8-kv-capacity-matched-preregistration"
V76_RUN_BUNDLE_SHA256 = (
    "46cf5ab3e6d3688de25cfdcf101710a129fdba309a5f11a9404d17344848e5e6"
)
V78_RUN_BUNDLE_SHA256 = (
    "e6df12c976910948c1026249b05fc065932169897aa5a09ff984b6d765385463"
)
HARDWARE_TELEMETRY_SHA256 = (
    "a31d9c4cfe6507ca642c061c14cdb40b8ebe35b6ea81783a2199df2bb3c0e475"
)
SOURCE_SHA256 = {
    "probe_vllm_quantized_adapter_switch_v73.py": (
        "43661c32cd8d06deef6d8e2f0d83d889b00f554748b94c3345e2b2052cac66a9"
    ),
    "probe_vllm_fp8_attested_v76.py": (
        "a23d43ee5b6b334fdc58b93e0ce7e7d3fcf72ea4047549f3a0f4d5b715a3fc70"
    ),
    "probe_vllm_fp8_kv_cache_v78.py": (
        "916aa316494619030b6232d2596486ae43fc58709063b1045617c358b3073485"
    ),
    "launch_qwen36_fp8_kv_capacity_v79.sh": (
        "4ca93e3a171787bb56613bf3648365ae96a355e28ec95f094b12d1982b6772df"
    ),
    "monitor_qwen36_fp8_kv_capacity_v79.py": (
        "30b67f5665aba0159c4d42c5c99ec15e6939f67fe0b5d065336b60cdc40f2d4e"
    ),
}

PHYSICAL_MEMORY_MIB = 97_887
REFERENCE_UTILIZATION = 0.500
TARGET_UTILIZATION = 0.485
REJECTED_NEXT_LOWER_UTILIZATION = 0.484
AVAILABLE_KV_GIB_AT_REFERENCE = 7.77
V76_CAPACITY_TOKENS = 157_696
V78_CAPACITY_TOKENS = 198_656
MAX_MODEL_LEN = 2_048
MIN_MARGIN_CONTEXTS = 2
MIN_CAPACITY_TOKENS = 161_792


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256_v79(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256_v79(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _validate_self_hash(value: dict[str, Any], path: Path) -> None:
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        claimed == canonical_sha256_v79(body),
        f"receipt self hash changed: {path}",
    )


def _run_inventory(root: Path, expected: str) -> dict[str, Any]:
    _require(root.is_dir() and not root.is_symlink(), f"missing run: {root}")
    rows = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": file_sha256_v79(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file()
    ]
    _require(len(rows) == 9, f"run file count changed: {root}")
    bundle = canonical_sha256_v79(rows)
    _require(bundle == expected, f"run inventory changed: {root}")
    return {"file_count": len(rows), "bundle_sha256": bundle, "files": rows}


def _log_scalar(text: str, pattern: str, label: str) -> str:
    matches = re.findall(pattern, text)
    _require(len(matches) == 1, f"{label} log cardinality changed")
    return matches[0]


def _read_run_v79(
    root: Path, expected_bundle: str, expected_schema: str, expected_tokens: int
) -> dict[str, Any]:
    inventory = _run_inventory(root, expected_bundle)
    receipts = []
    logs = []
    for gpu in range(4):
        receipt_path = root / f"gpu_{gpu}.json"
        log_path = root / f"gpu_{gpu}.log"
        value = _json(receipt_path)
        _validate_self_hash(value, receipt_path)
        _require(value.get("schema") == expected_schema, "actor schema changed")
        _require(
            value.get("runtime", {}).get("gpu_memory_utilization") == 0.5
            and value.get("precision_arm") == "fp8_serialized"
            and value.get("runtime", {}).get("resolved_quantization") == "fp8"
            and value.get("runtime", {}).get("enforce_eager") is True
            and value.get("runtime", {}).get("max_num_seqs") == 68
            and value.get("source_dataset_rows_opened") == 0
            and value.get("protected_ood_shadow_or_terminal_opened") is False
            and value.get("adapter_update_or_hpo_performed") is False
            and value.get("engine_shutdown_completed") is True
            and value.get("preflight_gates", {}).get(
                "scored_evaluation_or_training_authorized"
            )
            is False,
            "actor authority/runtime changed",
        )
        text = log_path.read_text(encoding="utf-8", errors="strict")
        tokens = int(
            _log_scalar(
                text,
                r"GPU KV cache size: ([\d,]+) tokens",
                "capacity",
            ).replace(",", "")
        )
        available = float(
            _log_scalar(
                text,
                r"Available KV cache memory: ([\d.]+) GiB",
                "available KV",
            )
        )
        concurrency = float(
            _log_scalar(
                text,
                r"Maximum concurrency for 2,048 tokens per request: "
                r"([\d.]+)x",
                "concurrency",
            )
        )
        _require(tokens == expected_tokens, "capacity changed")
        _require(available == AVAILABLE_KV_GIB_AT_REFERENCE, "KV GiB changed")
        _require(
            concurrency == expected_tokens / MAX_MODEL_LEN,
            "hybrid concurrency changed",
        )
        _require(
            "Using TRITON Fp8 MoE backend" in text
            and "Auto-disabled DeepGemm" not in text
            and "DeepGEMM E8M0 enabled" not in text
            and "Traceback (most recent call last)" not in text
            and "CUDA out of memory" not in text,
            "required/forbidden log gate changed",
        )
        receipts.append(value)
        logs.append(text)

    telemetry = _parse_legacy_nvml_v79(root / "nvidia_smi_samples.log")
    times = [
        value["wall_runtime_seconds_excluding_model_load_and_cleanup"]
        for value in receipts
    ]
    return {
        "inventory": inventory,
        "actor_count": len(receipts),
        "median_runtime_seconds": statistics.median(times),
        "actor_runtime_seconds": times,
        "available_kv_gib_per_actor": [AVAILABLE_KV_GIB_AT_REFERENCE] * 4,
        "capacity_tokens_per_actor": [expected_tokens] * 4,
        "max_concurrency_per_actor": [expected_tokens / MAX_MODEL_LEN] * 4,
        "telemetry": telemetry,
        "receipts": receipts,
    }


def _parse_legacy_nvml_v79(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii").splitlines()
    batches: list[list[tuple[int, int, int, float]]] = []
    current: list[tuple[int, int, int, float]] = []
    for line in lines:
        if "," not in line:
            if current:
                batches.append(current)
                current = []
            _require(line.isdigit(), "legacy NVML timestamp changed")
            continue
        fields = [item.strip() for item in line.split(",")]
        _require(len(fields) == 4, "legacy NVML row width changed")
        current.append(
            (int(fields[0]), int(fields[1]), int(fields[2]), float(fields[3]))
        )
    if current:
        batches.append(current)
    _require(len(batches) > 2, "legacy NVML sample count changed")
    _require(
        all([row[0] for row in batch] == [0, 1, 2, 3] for batch in batches),
        "legacy NVML GPU batches changed",
    )
    by_gpu = {
        gpu: [batch[gpu] for batch in batches]
        for gpu in range(4)
    }
    _require(
        all(max(row[1] for row in rows) > 0 for rows in by_gpu.values()),
        "not all GPUs have useful activity",
    )
    _require(
        all(
            row[1] == 0 and row[2] <= 4
            for batch in batches[-2:]
            for row in batch
        ),
        "cleanup-idle tail changed",
    )
    return {
        "sample_batch_count": len(batches),
        "all_four_gpus_useful": True,
        "max_gpu_utilization_percent": {
            str(gpu): max(row[1] for row in rows)
            for gpu, rows in by_gpu.items()
        },
        "peak_memory_used_mib": {
            str(gpu): max(row[2] for row in rows)
            for gpu, rows in by_gpu.items()
        },
        "cleanup_final_two_batches_zero_util_and_at_most_4_mib": True,
    }


def inspect_hardware_v79() -> dict[str, Any]:
    _require(
        file_sha256_v79(HARDWARE_TELEMETRY) == HARDWARE_TELEMETRY_SHA256,
        "hardware telemetry identity changed",
    )
    totals = set()
    gpu_ids = set()
    count = 0
    with HARDWARE_TELEMETRY.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            totals.add(row.get("memory_total_mib"))
            gpu_ids.add(row.get("gpu"))
            count += 1
    _require(
        count == 644
        and totals == {PHYSICAL_MEMORY_MIB}
        and gpu_ids == {0, 1, 2, 3},
        "sealed physical-memory evidence changed",
    )
    return {
        "path": str(HARDWARE_TELEMETRY.relative_to(ROOT)),
        "sha256": HARDWARE_TELEMETRY_SHA256,
        "row_count": count,
        "physical_gpu_ids": sorted(gpu_ids),
        "memory_total_mib_per_gpu": PHYSICAL_MEMORY_MIB,
    }


def projected_capacity_v79(utilization: float) -> dict[str, Any]:
    released_mib = (
        REFERENCE_UTILIZATION - utilization
    ) * PHYSICAL_MEMORY_MIB
    projected_available_gib = (
        AVAILABLE_KV_GIB_AT_REFERENCE - released_mib / 1024
    )
    raw_tokens = (
        V78_CAPACITY_TOKENS
        * projected_available_gib
        / AVAILABLE_KV_GIB_AT_REFERENCE
    )
    aligned_tokens = math.floor(raw_tokens / MAX_MODEL_LEN) * MAX_MODEL_LEN
    return {
        "gpu_memory_utilization": utilization,
        "released_budget_mib_vs_v78": released_mib,
        "projected_available_kv_gib": projected_available_gib,
        "projected_raw_tokens": raw_tokens,
        "projected_block_aligned_tokens": aligned_tokens,
        "projected_full_2048_token_contexts": aligned_tokens // MAX_MODEL_LEN,
        "projection_is_not_live_evidence": True,
    }


def derive_utilization_v79() -> dict[str, Any]:
    selected = projected_capacity_v79(TARGET_UTILIZATION)
    rejected = projected_capacity_v79(REJECTED_NEXT_LOWER_UTILIZATION)
    required = V76_CAPACITY_TOKENS + MIN_MARGIN_CONTEXTS * MAX_MODEL_LEN
    _require(required == MIN_CAPACITY_TOKENS, "capacity target changed")
    _require(
        selected["projected_block_aligned_tokens"] >= required
        and rejected["projected_block_aligned_tokens"] < required,
        "0.001-grid utilization selection changed",
    )
    return {
        "search_grid_increment": 0.001,
        "selection_rule": (
            "lowest grid point projecting at least two complete 2048-token "
            "contexts beyond the sealed V76 capacity; live capacity must "
            "independently pass"
        ),
        "v76_capacity_floor_tokens": V76_CAPACITY_TOKENS,
        "minimum_margin_contexts": MIN_MARGIN_CONTEXTS,
        "minimum_margin_tokens": MIN_MARGIN_CONTEXTS * MAX_MODEL_LEN,
        "minimum_live_capacity_tokens_per_actor": required,
        "selected": selected,
        "next_lower_rejected": rejected,
    }


def inspect_sources_v79() -> dict[str, Any]:
    paths = {
        V73_PROBE.name: V73_PROBE,
        V76_PROBE.name: V76_PROBE,
        V78_PROBE.name: V78_PROBE,
        LAUNCHER.name: LAUNCHER,
        MONITOR.name: MONITOR,
    }
    rows = []
    for name, path in paths.items():
        actual = file_sha256_v79(path)
        _require(actual == SOURCE_SHA256[name], f"source changed: {name}")
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )
    return {
        "files": rows,
        "bundle_sha256": canonical_sha256_v79(rows),
        "site_packages_modified": False,
    }


def _residency_identity(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        row.get("routed_fp8_runtime_attestation", {}).get(
            "parameter_residency"
        )
        for row in receipts
    ]
    _require(values and all(value == values[0] for value in values), "residency drift")
    expected = {
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
    _require(values[0] == expected, "parameter-residency identity changed")
    return expected


def build_preregistration_v79() -> dict[str, Any]:
    v76 = _read_run_v79(
        V76_RUN,
        V76_RUN_BUNDLE_SHA256,
        "v76-qwen36-fp8-routed-runtime-attestation",
        V76_CAPACITY_TOKENS,
    )
    v78 = _read_run_v79(
        V78_RUN,
        V78_RUN_BUNDLE_SHA256,
        "v78-qwen36-fp8-per-token-head-kv-preflight",
        V78_CAPACITY_TOKENS,
    )
    for receipt in v76["receipts"]:
        _require(
            receipt.get("resolved_kv_cache_certificate") is None,
            "V76 must retain auto/BF16 hybrid KV",
        )
    for receipt in v78["receipts"]:
        _require(
            receipt.get("resolved_kv_cache_certificate")
            == {
                "cache_dtype": "fp8_per_token_head",
                "calculate_kv_scales": False,
                "kv_cache_dtype_skip_layers": [],
                "mamba_cache_dtype": "auto",
                "mamba_ssm_cache_dtype": "float32",
                "resolved_from_live_engine": True,
            },
            "V78 hybrid-cache dtype identity changed",
        )
    residency = _residency_identity(v78["receipts"])
    hardware = inspect_hardware_v79()
    sources = inspect_sources_v79()
    derivation = derive_utilization_v79()
    v78_peak = max(v78["telemetry"]["peak_memory_used_mib"].values())
    expected_peak = (
        v78_peak
        - derivation["selected"]["released_budget_mib_vs_v78"]
    )

    value: dict[str, Any] = {
        "schema": SCHEMA,
        "bead": "specialist-0j5.24",
        "status": "cpu_preregistered_live_launch_not_performed",
        "authority": {
            "cpu_evidence_and_contract_build_only": True,
            "gpu_launch_performed_by_this_build": False,
            "dataset_or_protected_data_opened": False,
            "model_update_or_training_performed": False,
            "checkpoint_or_config_promotion_performed": False,
            "site_package_modification_performed": False,
            "scored_or_training_authority": False,
        },
        "single_variable_change_from_v78": {
            "gpu_memory_utilization": [
                REFERENCE_UTILIZATION,
                TARGET_UTILIZATION,
            ]
        },
        "selected_runtime": {
            "gpu_memory_utilization": TARGET_UTILIZATION,
            "kv_cache_dtype": "fp8_per_token_head",
            "calculate_kv_scales": False,
            "kv_cache_dtype_skip_layers": [],
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "float32",
            "attention_backend_required_by_log": "TRITON_ATTN",
            "routed_moe_backend": "TRITON",
            "quantization": "fp8",
            "enforce_eager": True,
            "enable_flashinfer_autotune": False,
            "max_num_seqs": 68,
            "max_model_len": MAX_MODEL_LEN,
            "max_loras": 1,
            "max_cpu_loras": 2,
            "scheduling_policy": "fcfs",
            "async_scheduling": False,
            "prefix_caching": False,
            "all_other_v78_engine_workload_adapter_and_seed_fields_unchanged": True,
        },
        "utilization_derivation": derivation,
        "sealed_evidence": {
            "hardware": hardware,
            "v76_bf16_kv": {
                key: value
                for key, value in v76.items()
                if key != "receipts"
            },
            "v78_fp8_per_token_head_r3": {
                key: value
                for key, value in v78.items()
                if key != "receipts"
            },
            "v78_r1_historical_only_not_launch_ancestry": {
                "run_bundle_sha256": (
                    "0897b1c80b8161171736b994e1e5e4a88728a19d39200f280ff7799552838c71"
                ),
                "median_runtime_seconds": 49.38608100148849,
                "capacity_tokens_per_actor": 198_656,
            },
            "current_parameter_residency": residency,
            "sources": sources,
        },
        "live_acceptance": {
            "cardinality_and_identity": {
                "physical_gpu_ids_exact": [0, 1, 2, 3],
                "one_unique_actor_pid_per_gpu": True,
                "four_self_hashed_v79_actor_receipts": True,
                "fresh_run_directory_required": True,
                "no_foreign_compute_pid_during_measurement": True,
            },
            "hybrid_cache": {
                "cache_dtype_exact": "fp8_per_token_head",
                "calculate_kv_scales_exact": False,
                "skip_layers_exact": [],
                "mamba_ssm_cache_dtype_exact": "float32",
                "attention_backend_log_exact": "TRITON_ATTN",
                "routed_moe_backend_exact": "TRITON",
                "parameter_residency_must_equal_sealed_v78_r3": True,
            },
            "capacity": {
                "minimum_tokens_per_actor": MIN_CAPACITY_TOKENS,
                "minimum_full_2048_token_contexts_per_actor": 79,
                "minimum_margin_tokens_over_v76": 4_096,
                "minimum_margin_fraction_over_v76": (
                    MIN_CAPACITY_TOKENS / V76_CAPACITY_TOKENS - 1
                ),
                "live_engine_field_and_actor_log_must_agree": True,
                "projection_does_not_satisfy_this_gate": True,
            },
            "performance_and_memory": {
                "median_runtime_ratio_to_v78_r3_max": 1.03,
                "median_runtime_seconds_max": (
                    v78["median_runtime_seconds"] * 1.03
                ),
                "per_actor_generated_tokens_per_second_required": True,
                "per_actor_median_p95_and_max_call_latency_required": True,
                "ten_timed_generation_calls_exact_two_warmup_eight_measured": True,
                "peak_memory_used_mib_per_actor_max": v78_peak,
                "projected_peak_memory_used_mib_not_a_gate": expected_peak,
                "minimum_physical_headroom_mib_per_actor": (
                    PHYSICAL_MEMORY_MIB - v78_peak
                ),
                "all_four_gpus_require_positive_gpu_utilization": True,
                "all_four_gpus_require_positive_memory_utilization": True,
                "report_peak_vram_power_gpu_and_memory_utilization": True,
                "sample_interval_seconds_max": 1.0,
                "pcie_rx_tx_kib_per_second_required_not_null": True,
                "sampled_pcie_rx_tx_byte_integrals_required": True,
                "pcie_byte_integrals_are_left_rectangle_estimates": True,
                "hbm_bytes_per_second_must_not_be_inferred_from_memory_utilization": True,
            },
            "output": {
                "call_plan_exact": [
                    "reference",
                    "candidate",
                    "candidate",
                    "reference",
                    "reference",
                    "candidate",
                    "candidate",
                    "reference",
                ],
                "candidate_changes_output": True,
                "candidate_repeat_exact_at_token_hash_level": True,
                "persist_only_token_counts_and_sha256_not_text_or_token_ids": True,
                "paired_v78_v79_hash_agreement_and_drift_matrix_required": True,
                "known_reference_repeat_nondeterminism_cannot_be_hidden": True,
            },
            "semantic": {
                "source_disjoint_paired_evaluation_required": True,
                "mean_delta_min": -0.001,
                "paired_95pct_lower_bound_min": -0.002,
                "no_new_safety_failure": True,
                "run_only_after_data_free_live_gates_pass": True,
            },
            "protected_ood": {
                "one_shot_after_selection_freeze": True,
                "paired_95pct_lower_bound_min": -0.005,
                "worst_stratum_point_delta_min": -0.01,
                "no_new_safety_failure": True,
                "no_retuning_after_open": True,
            },
            "logs": {
                "required_once_per_actor": [
                    "Using TRITON Fp8 MoE backend",
                    "Using TRITON_ATTN attention backend",
                    "Available KV cache memory:",
                    "GPU KV cache size:",
                    "Maximum concurrency for 2,048 tokens per request:",
                    "Skipping FlashInfer autotune because it is disabled",
                ],
                "forbidden": [
                    "Auto-disabled DeepGemm",
                    "DeepGEMM E8M0 enabled",
                    "Traceback (most recent call last)",
                    "CUDA out of memory",
                ],
                "known_default_moe_and_lora_warnings_reported_not_suppressed": True,
            },
            "cleanup": {
                "engine_shutdown_completed_per_actor": True,
                "torch_process_group_destroyed_per_actor": True,
                "minimum_consecutive_post_exit_idle_batches": 2,
                "post_exit_gpu_utilization_percent_exact": 0,
                "post_exit_memory_used_mib_max": 4,
                "no_v79_actor_pid_remaining": True,
            },
            "promotion": {
                "all_prior_gates_must_pass": True,
                "reference_restore_exactness_or_approved_semantic_resolution_required": True,
                "scored_training_checkpoint_or_layout_promotion_default": False,
            },
        },
        "launch": {
            "exact_command": (
                "RUN=/home/catid/specialist/experiments/eggroll_es_hpo/runs/"
                "v79_fp8_kv_capacity_0485_r1 "
                "bash /home/catid/specialist/"
                "launch_qwen36_fp8_kv_capacity_v79.sh"
            ),
            "launcher_path": str(LAUNCHER.relative_to(ROOT)),
            "launcher_sha256": SOURCE_SHA256[LAUNCHER.name],
            "telemetry_monitor_path": str(MONITOR.relative_to(ROOT)),
            "telemetry_monitor_sha256": SOURCE_SHA256[MONITOR.name],
            "telemetry_sample_interval_seconds": 0.5,
            "pcie_counter_policy": "required_fail_closed_if_unsupported",
            "launch_not_performed_by_builder": True,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v79(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build_preregistration_v79()
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.check:
        _require(OUTPUT.is_file(), "V79 preregistration is missing")
        _require(
            OUTPUT.read_text(encoding="ascii") == payload,
            "V79 preregistration is stale",
        )
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(payload, encoding="ascii")
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": value["status"],
                "content_sha256": value[
                    "content_sha256_before_self_field"
                ],
                "output": str(OUTPUT),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
