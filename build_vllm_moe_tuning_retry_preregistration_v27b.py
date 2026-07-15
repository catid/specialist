#!/usr/bin/env python3
"""Build the V27B compiler-failure-only retry contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V27B_MOE_TUNING_COMPILER_RETRY_PREREGISTRATION.json"
)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def build_preregistration():
    value = {
        "schema": "vllm-moe-tuning-compiler-retry-preregistration-v27b",
        "status": "preregistered_before_retry_output_retry_not_launched",
        "failed_selection_attempt": {
            "v27a_preregistration_commit": "2572204429cf2b016d0a001086d309b582b97724",
            "official_tuner_sha256": (
                "230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf"
            ),
            "exit_code": 1,
            "output_table_written": False,
            "workers_with_identical_compiler_error": 3,
            "worker_error_file_sha256": (
                "47ebb5c98e8db695bcc17103cd0846f6d1119e6ac1274621d277b7dc91541330"
            ),
            "worker_error_file_bytes": 12517,
            "error_pass": "TritonGPURemoveLayoutConversions",
            "error_target": "cuda:120",
            "error_class_to_admit_on_retry": "triton.compiler.errors.CompilationError",
            "model_update_checkpoint_dataset_or_nontrain_surface_opened": False,
        },
        "retry_overlay": {
            "base_official_tuner_sha256": (
                "230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf"
            ),
            "patched_tuner_sha256": (
                "2a9c31063532c0572500d9ed19db7df18d82cd8dad85cefb41eab2a6269df25e"
            ),
            "only_change": (
                "extend the existing invalid-config exception tuple from "
                "OutOfResources to OutOfResources plus CompilationError"
            ),
            "benchmark_timing_config_generation_and_selection_unchanged": True,
            "newly_skipped_exception_class_count": 1,
            "newly_skipped_exception_class": "triton.compiler.errors.CompilationError",
            "all_other_exceptions_fail_closed": True,
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
            "fresh_output_directory": "/tmp/vllm_moe_tuned_v27b",
            "output_must_not_exist_before_retry": True,
        },
        "evaluation": {
            "retry_output_is_selection_not_evaluation": True,
            "output_must_be_committed_and_hash_frozen_before_evaluation": True,
            "evaluation_contract_commit": "2572204429cf2b016d0a001086d309b582b97724",
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
