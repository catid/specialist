#!/usr/bin/env python3
"""Build the V27D Ray GPU-ID representation-only evaluation retry contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V27D_MOE_TUNING_EVALUATION_IDENTITY_RETRY_PREREGISTRATION.json"
)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def build_preregistration():
    value = {
        "schema": "vllm-moe-tuning-evaluation-identity-retry-preregistration-v27d",
        "status": "preregistered_before_retry_measurement_retry_not_launched",
        "failed_evaluation_attempt": {
            "evaluation_contract_commit": (
                "2572204429cf2b016d0a001086d309b582b97724"
            ),
            "evaluation_runtime_commit": (
                "4a7a22b922ad5ed5bbd63f64135cfe42c3f3a0fa"
            ),
            "implementation_bundle_sha256": (
                "2becd7d693c3601a148db040fe004e7f61a1dcb690f0ee875a2f21e00eda4b9c"
            ),
            "attempt_file_sha256": (
                "a4d74ad8185c6be6010d44aa79f47f00e579cb48fda367c4d46e8d63db2b6b79"
            ),
            "attempt_content_sha256": (
                "a095dbe10ea642a0d64fccc9fd2cebf3782a79585d47204fcac03d7b00af3d99"
            ),
            "complete_log_sha256": (
                "aabb6aa9cfbeaf4634889ff4a836646eaeff7ff04d37de4b4f7756824cfcc5e6"
            ),
            "complete_log_bytes": 2_257,
            "exit_code": 1,
            "failure_type": "RayTaskError(RuntimeError)",
            "failure_message": "V27A Ray physical-GPU assignment changed",
            "phase": "inside_first_arm_before_first_kernel_benchmark",
            "kernel_timing_count": 0,
            "report_written": False,
            "model_update_checkpoint_dataset_or_nontrain_surface_opened": False,
        },
        "identity_probe": {
            "ray_version": "2.56.0",
            "actor_count": 4,
            "one_actor_per_gpu": True,
            "physical_gpu_ids": [0, 1, 2, 3],
            "raw_ray_gpu_ids_in_physical_order": [["0"], ["1"], ["2"], ["3"]],
            "raw_ray_gpu_id_type": "str",
            "cuda_visible_devices_in_physical_order": ["0", "1", "2", "3"],
            "visible_device_matches_canonical_physical_id_all_four": True,
            "only_failed_comparison": "list[str] compared to list[int]",
            "probe_used_no_dataset_model_or_kernel_benchmark": True,
        },
        "retry_overlay": {
            "only_semantic_change": (
                "canonicalize an exact Ray GPU ID string in {0,1,2,3} to its integer "
                "physical ID before equality checks"
            ),
            "accepted_raw_types": ["int", "str"],
            "accepted_canonical_ids": [0, 1, 2, 3],
            "booleans_floats_padded_strings_uuids_and_other_values_rejected": True,
            "cuda_visible_devices_must_equal_canonical_physical_id": True,
            "one_actor_per_unique_physical_gpu_still_required": True,
            "all_four_physical_gpus_still_required": True,
            "batch_to_physical_gpu_schedule_unchanged": True,
            "benchmark_configs_iterations_seeds_arm_order_and_gate_unchanged": True,
            "fresh_attempt_name": (
                "s6_v27a_moe_tuned_vs_default_fresh_paired_kernel_"
                "evaluation_retry_r1"
            ),
        },
        "frozen_selection": {
            "table_file_sha256": (
                "128806798a5bf8a961a5bd0bc8765c82e8b73a116e6c7411e7aeba5522667562"
            ),
            "table_content_sha256": (
                "a4f82f53b037f766536013bdc10c8ca1e49873603a8f44972ef8007ed406de84"
            ),
            "selection_evidence_commit": "27f5aae",
            "selection_not_reopened": True,
        },
        "evaluation": {
            "base_contract_commit": "2572204429cf2b016d0a001086d309b582b97724",
            "fresh_seeds": [20260716, 20260717, 20260718, 20260719, 20260720],
            "repetitions": 5,
            "batch_sizes": [256, 512, 1024, 2048],
            "official_iterations_per_worker": 100,
            "pass_gate_unchanged": {
                "each_batch_median_speedup_at_least": 1.0,
                "each_batch_tuned_faster_repetitions_at_least": 4,
                "global_geometric_mean_speedup_at_least": 1.03,
            },
            "retry_not_launched_by_this_artifact": True,
        },
        "authority": {
            "evaluation_retry_authorized": True,
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
