#!/usr/bin/env python3
"""Build compact aggregate evidence for the passing V27D paired kernel gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V27D_MOE_TUNING_EVALUATION_POSITIVE_EVIDENCE.json"
)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def build_evidence():
    value = {
        "schema": "vllm-moe-tuning-evaluation-positive-evidence-v27d",
        "status": "valid_completed_paired_kernel_gate_passed",
        "contracts": {
            "base_evaluation_preregistration_commit": (
                "2572204429cf2b016d0a001086d309b582b97724"
            ),
            "identity_retry_preregistration_commit": (
                "1c35946ea688950b2ff577a8c24620acd2dc7ea4"
            ),
            "identity_retry_runtime_commit": "6987c3a",
            "identity_retry_preregistration_content_sha256": (
                "0a1f394a869deb3af75e8d2bd53043988909e6def0f969dc5254def761255eeb"
            ),
            "implementation_bundle_sha256": (
                "8a339a76b9fd6bb1535ce3dcfe0fb036e5ab57f42acf3cff32b4c3896af5da36"
            ),
            "selected_table_file_sha256": (
                "128806798a5bf8a961a5bd0bc8765c82e8b73a116e6c7411e7aeba5522667562"
            ),
            "selected_table_content_sha256": (
                "a4f82f53b037f766536013bdc10c8ca1e49873603a8f44972ef8007ed406de84"
            ),
            "official_tuner_sha256": (
                "230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf"
            ),
        },
        "artifacts": {
            "report": {
                "file_sha256": (
                    "94b322a7dc90f5d142ace058d9f3d682edb3349935614ed551fa105486934dfb"
                ),
                "content_sha256": (
                    "91b658edc89249020c3860bfd414bb33725e63e8419ce12c0fbee1dba751f6de"
                ),
            },
            "attempt": {
                "file_sha256": (
                    "048f2347b43e5b2646ca6088ceb2b5138f3948594ece827a9927cbf67d57e204"
                ),
                "content_sha256": (
                    "7453f80cc10c85e8b5ff39bfaf70f90f0b7c38c84497a42b75cc9e5fdeb7c85d"
                ),
                "status": "complete",
                "phase": "after_compact_report",
            },
            "complete_log_sha256": (
                "f5b5fe67f3745428adcecfc6015cdaa3aeb1f529cef09db76e8b61bc9facba74"
            ),
            "complete_log_bytes": 9_263,
        },
        "runtime_integrity": {
            "fresh_ray_arm_count": 10,
            "repetition_count": 5,
            "exactly_four_one_gpu_workers_per_arm": True,
            "all_four_physical_gpu_ids_canonicalized_and_verified_each_arm": True,
            "default_arm_generic_fallback_verified_each_repetition": True,
            "tuned_arm_exact_committed_table_load_verified_each_repetition": True,
            "alternating_arm_order_unchanged": True,
            "official_iterations_per_worker": 100,
            "all_processes_exit_zero": True,
        },
        "aggregate_result": {
            "batch_256": {
                "median_speedup": 1.055909609745163,
                "tuned_faster_repetitions": 5,
                "repetitions": 5,
                "pass": True,
            },
            "batch_512": {
                "median_speedup": 1.0246098951794156,
                "tuned_faster_repetitions": 5,
                "repetitions": 5,
                "pass": True,
            },
            "batch_1024": {
                "median_speedup": 1.1209752402917341,
                "tuned_faster_repetitions": 5,
                "repetitions": 5,
                "pass": True,
            },
            "batch_2048": {
                "median_speedup": 1.0155353235485465,
                "tuned_faster_repetitions": 5,
                "repetitions": 5,
                "pass": True,
            },
            "global_geometric_mean_speedup": 1.0534625119555665,
            "required_global_geometric_mean_speedup": 1.03,
            "all_per_batch_gates_passed": True,
            "global_gate_passed": True,
        },
        "decision": {
            "authorize_separate_end_to_end_train_only_runtime_ab_preregistration": True,
            "direct_recipe_adoption_authorized": False,
            "model_update_checkpoint_dataset_promotion_authorized": False,
            "validation_heldout_ood_or_benchmark_open_authorized": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


if __name__ == "__main__":
    print(json.dumps(build_evidence(), indent=2, sort_keys=True))
