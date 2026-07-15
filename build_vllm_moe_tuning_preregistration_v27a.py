#!/usr/bin/env python3
"""Build the pre-output V27A MoE tuning evaluation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V27A_MOE_TUNING_EVALUATION_PREREGISTRATION.json"
)
OFFICIAL_TUNER_URL = (
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.25.0/"
    "benchmarks/kernels/benchmark_moe.py"
)
OFFICIAL_TUNER_SHA256 = (
    "230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf"
)
FUSED_MOE_SHA256 = (
    "72811a4e543cc6f415f184cb951b61522643cddc4d6456f61a2f8c1a53b2cf79"
)
MODEL_CONFIG_SHA256 = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
GPU_UUIDS = [
    "GPU-4c394fc5-b18f-6622-ca94-f7fbd7112927",
    "GPU-f10c2baf-536b-1d40-cd4b-25b202ae0ded",
    "GPU-04cde663-7c53-2f18-3ec4-1699820e2640",
    "GPU-972bf85d-1b32-2d1b-20f6-babc4c804999",
]
BATCH_SIZES = [256, 512, 1024, 2048]
EVALUATION_SEEDS = [20260716, 20260717, 20260718, 20260719, 20260720]


def canonical_sha256(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def build_preregistration():
    schedule = []
    for repetition, seed in enumerate(EVALUATION_SEEDS):
        order = ["default", "tuned"] if repetition % 2 == 0 else ["tuned", "default"]
        schedule.append({
            "repetition": repetition,
            "seed": seed,
            "arm_order": order,
            "batch_to_physical_gpu": {
                str(batch): gpu for batch, gpu in zip(BATCH_SIZES, range(4))
            },
        })
    value = {
        "schema": "vllm-moe-tuning-evaluation-preregistration-v27a",
        "status": "preregistered_before_tuning_output_observed_evaluation_not_launched",
        "selection_is_not_evaluation": True,
        "selection": {
            "state_at_preregistration": "official_tuner_running_output_not_observed",
            "official_source_url": OFFICIAL_TUNER_URL,
            "official_source_sha256": OFFICIAL_TUNER_SHA256,
            "vllm_version": "0.25.0",
            "torch_version": "2.11.0+cu130",
            "triton_version": "3.6.0",
            "installed_fused_moe_sha256": FUSED_MOE_SHA256,
            "model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "model_config_sha256": MODEL_CONFIG_SHA256,
            "model_geometry": {
                "experts": 256,
                "experts_per_token": 8,
                "hidden_size": 2048,
                "moe_intermediate_size": 512,
                "config_file_n_dimension": 512,
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
            },
            "batch_sizes": BATCH_SIZES,
            "search_space_configurations_per_batch": 1920,
            "selection_seed": 20260715,
            "temporary_output_directory": "/tmp/vllm_moe_tuned_v25",
            "expected_filename": (
                "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
                "Max-Q_Workstation_Edition.json"
            ),
            "tuned_output_must_be_frozen_by_hash_before_evaluation": True,
        },
        "hardware": {
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            "driver_version": "610.43.02",
            "gpu_uuids_in_physical_order": GPU_UUIDS,
            "exactly_four_one_gpu_ray_workers": True,
        },
        "evaluation": {
            "not_launched_by_this_artifact": True,
            "fresh_seeds": EVALUATION_SEEDS,
            "repetitions": 5,
            "schedule": schedule,
            "batch_sizes": BATCH_SIZES,
            "separate_fresh_process_and_ray_runtime_per_arm": True,
            "default_arm": {
                "vllm_tuned_config_folder": "fresh_empty_directory",
                "requires_generic_default_config_warning": True,
            },
            "tuned_arm": {
                "vllm_tuned_config_folder": "future_hash_frozen_tuned_directory",
                "requires_exact_tuned_config_load_log": True,
            },
            "common_command": {
                "tp_size": 1,
                "dtype": "auto_bfloat16",
                "tune": False,
                "official_num_iters_per_worker": 100,
                "raw_endpoint": "reported_kernel_time_microseconds",
            },
            "paired_endpoint": "default_kernel_time_divided_by_tuned_kernel_time",
            "summaries": {
                "per_batch": "median_of_five_paired_speedups",
                "global": "geometric_mean_of_four_per_batch_medians",
            },
            "pass_gate": {
                "each_batch_median_speedup_at_least": 1.0,
                "each_batch_tuned_faster_in_at_least_repetitions": 4,
                "global_geometric_mean_speedup_at_least": 1.03,
                "all_processes_exit_zero": True,
                "all_config_and_hardware_identities_exact": True,
            },
            "failure_decision": "discard_tuned_table_and_keep_vllm_generic_defaults",
        },
        "authority": {
            "pass_authorizes_only_separate_end_to_end_train_only_runtime_ab_preregistration": True,
            "direct_training_recipe_adoption_allowed": False,
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "dataset_promotion_allowed": False,
            "validation_heldout_ood_or_benchmark_open_allowed": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def exclusive_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    value = build_preregistration()
    if args.output:
        exclusive_write(args.output, value)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
