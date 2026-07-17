#!/usr/bin/env python3
"""Build the CPU-only V80 BF16-attention/BF16-Mamba capacity contract."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import build_qwen36_fp8_kv_capacity_preregistration_v79 as common


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_bf16_kv_mamba_capacity_matched_v80.json"
)
V76_RUN = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v76_fp8_attested_050_r7_residency"
)
V78C_RUN = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v78c_bf16_kv_bf16_mamba_ssm_r1"
)
V73_PROBE = ROOT / "probe_vllm_quantized_adapter_switch_v73.py"
V76_PROBE = ROOT / "probe_vllm_fp8_attested_v76.py"
V78C_PROBE = ROOT / "probe_vllm_bf16_kv_mamba_bf16_v78c.py"
MONITOR = ROOT / "monitor_qwen36_fp8_kv_capacity_v79.py"
LAUNCHER = ROOT / "launch_qwen36_bf16_kv_mamba_capacity_v80.sh"

SCHEMA = "v80-qwen36-bf16-kv-mamba-capacity-matched-preregistration"
V76_RUN_BUNDLE_SHA256 = (
    "46cf5ab3e6d3688de25cfdcf101710a129fdba309a5f11a9404d17344848e5e6"
)
V78C_RUN_BUNDLE_SHA256 = (
    "a9d82f71bb6beecc420be135737f2048fb770bee5d97309f9e90da4e31ef833f"
)
SOURCE_SHA256 = {
    "probe_vllm_quantized_adapter_switch_v73.py": (
        "43661c32cd8d06deef6d8e2f0d83d889b00f554748b94c3345e2b2052cac66a9"
    ),
    "probe_vllm_fp8_attested_v76.py": (
        "a23d43ee5b6b334fdc58b93e0ce7e7d3fcf72ea4047549f3a0f4d5b715a3fc70"
    ),
    "probe_vllm_bf16_kv_mamba_bf16_v78c.py": (
        "761857944064a0b21ff528971d3f497e4e67865679fa51a30d385cab65835dcb"
    ),
    "monitor_qwen36_fp8_kv_capacity_v79.py": (
        "228885d9854828a702d49e6c38a13b6d2189d6b4e026f081e9f4e39c2c0c4df2"
    ),
    "launch_qwen36_bf16_kv_mamba_capacity_v80.sh": (
        "3a335c0e44a9b8130adec8c0e11f52d2246264aa0a47f5b173cbdb35093fc7ad"
    ),
}

PHYSICAL_MEMORY_MIB = 97_887
REFERENCE_UTILIZATION = 0.500
TARGET_UTILIZATION = 0.479
REJECTED_NEXT_LOWER_UTILIZATION = 0.478
AVAILABLE_KV_GIB_AT_REFERENCE = 7.77
V76_CAPACITY_TOKENS = 157_696
V78C_CAPACITY_TOKENS = 218_843
MAX_MODEL_LEN = 2_048
MIN_CAPACITY_TOKENS = 161_792
OVERRIDE_WARNING = (
    "Qwen3.5 model specifies mamba_ssm_dtype='float32' in its config, but "
    "--mamba-ssm-cache-dtype='bfloat16' was passed. Using the "
    "user-specified value."
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _validate_self_hash(value: dict[str, Any], path: Path) -> None:
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        claimed == common.canonical_sha256_v79(body),
        f"receipt self hash changed: {path}",
    )


def _log_scalar(text: str, pattern: str, label: str) -> str:
    matches = re.findall(pattern, text)
    _require(len(matches) == 1, f"{label} log cardinality changed")
    return matches[0]


def _parse_v78c_nvml_v80(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="ascii").splitlines()
    batches: list[list[tuple[int, int, int, float]]] = []
    current: list[tuple[int, int, int, float]] = []
    for line in lines:
        if "," not in line:
            if current:
                batches.append(current)
                current = []
            _require(line.isdigit(), "V78c NVML timestamp changed")
            continue
        fields = [item.strip() for item in line.split(",")]
        _require(len(fields) == 4, "V78c NVML row width changed")
        current.append(
            (int(fields[0]), int(fields[1]), int(fields[2]), float(fields[3]))
        )
    if current:
        batches.append(current)
    _require(len(batches) > 2, "V78c NVML sample count changed")
    _require(
        all([row[0] for row in batch] == [0, 1, 2, 3] for batch in batches),
        "V78c NVML GPU batches changed",
    )
    by_gpu = {
        gpu: [batch[gpu] for batch in batches]
        for gpu in range(4)
    }
    _require(
        all(max(row[1] for row in rows) > 0 for rows in by_gpu.values()),
        "V78c did not use every GPU",
    )
    _require(
        all(
            row[1] == 0 and row[2] <= 5
            for batch in batches[-2:]
            for row in batch
        ),
        "V78c cleanup-idle tail changed",
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
        "cleanup_final_two_batches_zero_util_and_at_most_5_mib": True,
    }


def inspect_v78c_v80() -> dict[str, Any]:
    inventory = common._run_inventory(
        V78C_RUN, V78C_RUN_BUNDLE_SHA256
    )
    receipts = []
    times = []
    for gpu in range(4):
        receipt_path = V78C_RUN / f"gpu_{gpu}.json"
        log_path = V78C_RUN / f"gpu_{gpu}.log"
        value = _json(receipt_path)
        _validate_self_hash(value, receipt_path)
        _require(
            value.get("schema")
            == "v78c-qwen36-bf16-kv-bf16-mamba-ssm-preflight"
            and value.get("runtime", {}).get("gpu_memory_utilization") == 0.5
            and value.get("runtime", {}).get("mamba_ssm_cache_dtype")
            == "bfloat16"
            and value.get("runtime", {}).get("resolved_quantization") == "fp8"
            and value.get("runtime", {}).get("enforce_eager") is True
            and value.get("runtime", {}).get("max_num_seqs") == 68
            and value.get("source_dataset_rows_opened") == 0
            and value.get("protected_ood_shadow_or_terminal_opened") is False
            and value.get("adapter_update_or_hpo_performed") is False
            and value.get("engine_shutdown_completed") is True
            and value.get("resolved_hybrid_cache_certificate")
            == {
                "cache_dtype": "auto",
                "mamba_ssm_cache_dtype": "bfloat16",
                "resolved_from_live_engine": True,
            }
            and value.get("preflight_gates", {}).get(
                "scored_evaluation_or_training_authorized"
            )
            is False,
            "V78c actor identity/runtime changed",
        )
        text = log_path.read_text(encoding="utf-8", errors="strict")
        tokens = int(
            _log_scalar(
                text,
                r"GPU KV cache size: ([\d,]+) tokens",
                "V78c capacity",
            ).replace(",", "")
        )
        available = float(
            _log_scalar(
                text,
                r"Available KV cache memory: ([\d.]+) GiB",
                "V78c KV GiB",
            )
        )
        concurrency = float(
            _log_scalar(
                text,
                r"Maximum concurrency for 2,048 tokens per request: "
                r"([\d.]+)x",
                "V78c concurrency",
            )
        )
        _require(
            tokens == V78C_CAPACITY_TOKENS
            and available == AVAILABLE_KV_GIB_AT_REFERENCE
            and concurrency == 106.86,
            "V78c capacity evidence changed",
        )
        _require(
            text.count(OVERRIDE_WARNING) == 1
            and "Using FLASH_ATTN attention backend" in text
            and "Using TRITON Fp8 MoE backend" in text
            and "Auto-disabled DeepGemm" not in text
            and "DeepGEMM E8M0 enabled" not in text
            and "Traceback (most recent call last)" not in text
            and "CUDA out of memory" not in text,
            "V78c required/forbidden log gate changed",
        )
        receipts.append(value)
        times.append(value["wall_runtime_seconds_excluding_model_load_and_cleanup"])
    residency = common._residency_identity(receipts)
    return {
        "inventory": inventory,
        "actor_count": 4,
        "median_runtime_seconds": statistics.median(times),
        "actor_runtime_seconds": times,
        "available_kv_gib_per_actor": [7.77] * 4,
        "capacity_tokens_per_actor": [V78C_CAPACITY_TOKENS] * 4,
        "max_concurrency_per_actor": [106.86] * 4,
        "attention_backend": "FLASH_ATTN",
        "model_config_fp32_ssm_override_warning_actor_count": 4,
        "parameter_residency": residency,
        "telemetry": _parse_v78c_nvml_v80(
            V78C_RUN / "nvidia_smi_samples.log"
        ),
    }


def projected_capacity_v80(utilization: float) -> dict[str, Any]:
    released_mib = (
        REFERENCE_UTILIZATION - utilization
    ) * PHYSICAL_MEMORY_MIB
    projected_available_gib = (
        AVAILABLE_KV_GIB_AT_REFERENCE - released_mib / 1024
    )
    raw_tokens = (
        V78C_CAPACITY_TOKENS
        * projected_available_gib
        / AVAILABLE_KV_GIB_AT_REFERENCE
    )
    aligned_tokens = math.floor(raw_tokens / MAX_MODEL_LEN) * MAX_MODEL_LEN
    return {
        "gpu_memory_utilization": utilization,
        "released_budget_mib_vs_v78c": released_mib,
        "projected_available_kv_gib": projected_available_gib,
        "projected_raw_tokens": raw_tokens,
        "projected_complete_context_tokens": aligned_tokens,
        "projected_full_2048_token_contexts": aligned_tokens // MAX_MODEL_LEN,
        "projection_is_not_live_evidence": True,
    }


def derive_utilization_v80() -> dict[str, Any]:
    selected = projected_capacity_v80(TARGET_UTILIZATION)
    rejected = projected_capacity_v80(REJECTED_NEXT_LOWER_UTILIZATION)
    _require(
        selected["projected_complete_context_tokens"] >= MIN_CAPACITY_TOKENS
        and rejected["projected_complete_context_tokens"]
        < MIN_CAPACITY_TOKENS,
        "V80 grid selection changed",
    )
    return {
        "search_grid_increment": 0.001,
        "selection_rule": (
            "lowest grid point projecting at least 79 complete 2048-token "
            "contexts; live group-aware capacity must independently pass"
        ),
        "v76_capacity_floor_tokens": V76_CAPACITY_TOKENS,
        "minimum_live_capacity_tokens_per_actor": MIN_CAPACITY_TOKENS,
        "minimum_margin_tokens_over_v76": (
            MIN_CAPACITY_TOKENS - V76_CAPACITY_TOKENS
        ),
        "selected": selected,
        "next_lower_rejected": rejected,
    }


def inspect_sources_v80() -> dict[str, Any]:
    paths = {
        path.name: path
        for path in (
            V73_PROBE,
            V76_PROBE,
            V78C_PROBE,
            MONITOR,
            LAUNCHER,
        )
    }
    rows = []
    for name, path in paths.items():
        actual = common.file_sha256_v79(path)
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
        "bundle_sha256": common.canonical_sha256_v79(rows),
        "site_packages_modified": False,
    }


def build_preregistration_v80() -> dict[str, Any]:
    v76 = common._read_run_v79(
        V76_RUN,
        V76_RUN_BUNDLE_SHA256,
        "v76-qwen36-fp8-routed-runtime-attestation",
        V76_CAPACITY_TOKENS,
    )
    v78c = inspect_v78c_v80()
    hardware = common.inspect_hardware_v79()
    sources = inspect_sources_v80()
    derivation = derive_utilization_v80()
    v78c_peak = max(v78c["telemetry"]["peak_memory_used_mib"].values())
    projected_peak = (
        v78c_peak
        - derivation["selected"]["released_budget_mib_vs_v78c"]
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
        "single_variable_change_from_v78c": {
            "gpu_memory_utilization": [
                REFERENCE_UTILIZATION,
                TARGET_UTILIZATION,
            ]
        },
        "selected_runtime": {
            "gpu_memory_utilization": TARGET_UTILIZATION,
            "kv_cache_dtype": "auto",
            "attention_kv_effective_dtype": "bfloat16",
            "mamba_cache_dtype": "auto",
            "mamba_ssm_cache_dtype": "bfloat16",
            "attention_backend_required_by_log": "FLASH_ATTN",
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
            "model_config_recommends_mamba_ssm_float32": True,
            "user_override_warning_expected": True,
            "all_other_v78c_engine_workload_adapter_and_seed_fields_unchanged": True,
        },
        "utilization_derivation": derivation,
        "sealed_evidence": {
            "hardware": hardware,
            "v76_control": {
                key: item for key, item in v76.items() if key != "receipts"
            },
            "v78c_r1": v78c,
            "sources": sources,
        },
        "live_acceptance": {
            "cardinality_and_identity": {
                "physical_gpu_ids_exact": [0, 1, 2, 3],
                "one_unique_actor_root_pid_per_gpu": True,
                "compute_pids_must_be_actor_descendants": True,
                "four_self_hashed_v80_actor_receipts": True,
                "fresh_run_directory_required": True,
                "foreign_compute_pids_exact": [],
            },
            "hybrid_cache": {
                "cache_dtype_exact": "auto",
                "effective_attention_kv_dtype": "bfloat16",
                "mamba_cache_dtype_exact": "auto",
                "mamba_ssm_cache_dtype_exact": "bfloat16",
                "attention_backend_log_exact": "FLASH_ATTN",
                "routed_moe_backend_exact": "TRITON",
                "parameter_residency_must_equal_sealed_v78c_r1": True,
                "model_config_fp32_ssm_override_warning_required": True,
            },
            "capacity": {
                "minimum_tokens_per_actor": MIN_CAPACITY_TOKENS,
                "minimum_full_2048_token_contexts_per_actor": 79,
                "minimum_margin_tokens_over_v76": 4_096,
                "live_engine_field_and_actor_log_must_agree": True,
                "projection_does_not_satisfy_this_gate": True,
            },
            "performance_and_memory": {
                "median_runtime_ratio_to_v78c_r1_max": 1.03,
                "median_runtime_seconds_max": (
                    v78c["median_runtime_seconds"] * 1.03
                ),
                "per_actor_generated_tokens_per_second_required": True,
                "per_actor_median_p95_and_max_call_latency_required": True,
                "ten_timed_generation_calls_exact_two_warmup_eight_measured": True,
                "peak_memory_used_mib_per_actor_max": v78c_peak,
                "projected_peak_memory_used_mib_not_a_gate": projected_peak,
                "minimum_physical_headroom_mib_per_actor": (
                    PHYSICAL_MEMORY_MIB - v78c_peak
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
                "paired_v78c_v80_hash_agreement_and_drift_matrix_required": True,
                "known_reference_repeat_nondeterminism_cannot_be_hidden": True,
            },
            "semantic": {
                "source_disjoint_paired_evaluation_required": True,
                "mean_delta_min": -0.001,
                "paired_95pct_lower_bound_min": -0.002,
                "no_new_safety_failure": True,
                "explicit_mamba_ssm_override_requires_strict_review": True,
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
                    "Using FLASH_ATTN attention backend",
                    OVERRIDE_WARNING,
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
                "minimum_consecutive_post_exit_idle_batches": 3,
                "post_exit_gpu_utilization_percent_exact": 0,
                "post_exit_memory_used_mib_max": 4,
                "no_v80_actor_or_descendant_compute_pid_remaining": True,
            },
            "promotion": {
                "all_prior_gates_must_pass": True,
                "model_config_fp32_ssm_override_requires_explicit_acceptance": True,
                "reference_restore_exactness_or_approved_semantic_resolution_required": True,
                "scored_training_checkpoint_or_layout_promotion_default": False,
            },
        },
        "launch": {
            "exact_command": (
                "RUN=/home/catid/specialist/experiments/eggroll_es_hpo/runs/"
                "v80_bf16_kv_mamba_capacity_0479_r1 "
                "bash /home/catid/specialist/"
                "launch_qwen36_bf16_kv_mamba_capacity_v80.sh"
            ),
            "launcher_path": str(LAUNCHER.relative_to(ROOT)),
            "launcher_sha256": SOURCE_SHA256[LAUNCHER.name],
            "telemetry_monitor_path": str(MONITOR.relative_to(ROOT)),
            "telemetry_monitor_sha256": SOURCE_SHA256[MONITOR.name],
            "telemetry_sample_interval_seconds": 0.5,
            "telemetry_cleanup_consecutive_batches": 3,
            "telemetry_cleanup_timeout_seconds": 60,
            "pcie_counter_policy": "required_fail_closed_if_unsupported",
            "launch_not_performed_by_builder": True,
        },
    }
    value["content_sha256_before_self_field"] = (
        common.canonical_sha256_v79(value)
    )
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = build_preregistration_v80()
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.check:
        _require(OUTPUT.is_file(), "V80 preregistration is missing")
        _require(
            OUTPUT.read_text(encoding="ascii") == payload,
            "V80 preregistration is stale",
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
