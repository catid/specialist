#!/usr/bin/env python3
"""Build the V27C exact-MLIR-runtime-failure retry contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V27C_MOE_TUNING_MLIR_RUNTIME_RETRY_PREREGISTRATION.json"
)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def build_preregistration():
    value = {
        "schema": "vllm-moe-tuning-mlir-runtime-retry-preregistration-v27c",
        "status": "preregistered_before_retry_output_retry_not_launched",
        "failed_selection_attempt": {
            "v27b_preregistration_commit": (
                "90ee9dd8275348e1bc5abcc5bbbf0caceec83bdd"
            ),
            "v27b_patched_tuner_sha256": (
                "2a9c31063532c0572500d9ed19db7df18d82cd8dad85cefb41eab2a6269df25e"
            ),
            "exit_code": 1,
            "output_table_written": False,
            "output_file_count": 0,
            "complete_failure_log_sha256": (
                "35702f6db186adfb3c77fa39d1105cd4f15011784593635f3d90ad1a6702642d"
            ),
            "complete_failure_log_bytes": 361_846,
            "ray_cause_class": "RuntimeError",
            "ray_cause_message": "PassManager::run failed",
            "compiler_pass": "TritonGPURemoveLayoutConversions",
            "compiler_diagnostic": (
                "Failures have been detected while processing an MLIR pass pipeline"
            ),
            "failed_worker_progress_percent": 83,
            "other_worker_progress_percentages_at_failure": [83, 78, 64],
            "model_update_checkpoint_dataset_or_nontrain_surface_opened": False,
        },
        "retry_overlay": {
            "base_official_tuner_sha256": (
                "230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf"
            ),
            "v27b_patched_tuner_sha256": (
                "2a9c31063532c0572500d9ed19db7df18d82cd8dad85cefb41eab2a6269df25e"
            ),
            "v27c_patched_tuner_sha256": (
                "ec1cfe2aea31a3aed59fc2322b8697855329f1d8cdf8c5c4d0310655e6399ec7"
            ),
            "only_new_change": (
                "skip a built-in RuntimeError only when its message is exactly "
                "PassManager::run failed"
            ),
            "newly_skipped_exception_class": "builtins.RuntimeError",
            "newly_skipped_exact_message": "PassManager::run failed",
            "newly_skipped_exact_message_count": 1,
            "all_other_runtime_errors_reraised": True,
            "existing_out_of_resources_and_compilation_error_skips_unchanged": True,
            "benchmark_timing_config_generation_and_selection_unchanged": True,
        },
        "exact_retry": {
            "model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "model_config_sha256": (
                "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
            ),
            "batch_sizes": [256, 512, 1024, 2048],
            "search_space_configurations_per_batch": 1920,
            "selection_seed": 20260715,
            "tp_size": 1,
            "dtype": "auto_bfloat16",
            "gpu_ids": [0, 1, 2, 3],
            "fresh_output_directory": "/tmp/vllm_moe_tuned_v27c",
            "output_must_not_exist_before_retry": True,
        },
        "evaluation": {
            "retry_output_is_selection_not_evaluation": True,
            "output_must_be_committed_and_hash_frozen_before_evaluation": True,
            "evaluation_contract_commit": (
                "2572204429cf2b016d0a001086d309b582b97724"
            ),
            "evaluation_runtime_commit": "4a7a22b",
            "evaluation_gate_unchanged_from_v27a": True,
            "evaluation_not_launched_by_this_artifact": True,
        },
        "authority": {
            "retry_selection_launch_authorized": True,
            "direct_recipe_adoption_model_update_checkpoint_dataset_promotion": False,
            "validation_heldout_ood_or_benchmark_open": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


if __name__ == "__main__":
    print(json.dumps(build_preregistration(), indent=2, sort_keys=True))
