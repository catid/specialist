#!/usr/bin/env python3
"""Build compact aggregate evidence for the failed-safe V26A hybrid gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V26A_FP8_ROUTED_EXPERTS_COMPAT_SPEED_EVIDENCE.json"
)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()


def build_evidence():
    equivalence = {
        "absolute_mean_signed_logprob_delta": 0.009502541939826211,
        "mean_absolute_logprob_delta": 0.15710689712839349,
        "root_mean_square_logprob_delta": 0.234610494684071,
        "p99_absolute_logprob_delta": 0.8093850884027765,
        "maximum_absolute_logprob_delta": 1.647368311882019,
        "row_logprob_pearson": 0.9951093302331562,
        "greedy_first_token_agreement": 0.9678571428571429,
    }
    thresholds = {
        "absolute_mean_signed_logprob_delta_max": 0.02,
        "mean_absolute_logprob_delta_max": 0.05,
        "root_mean_square_logprob_delta_max": 0.10,
        "p99_absolute_logprob_delta_max": 0.25,
        "maximum_absolute_logprob_delta_max": 0.75,
        "row_logprob_pearson_min": 0.995,
        "greedy_first_token_agreement_min": 0.99,
    }
    speed = {
        "physical_gpu_0": {
            "load_order": "fp8_first",
            "median_hybrid_over_fp8_throughput_ratio": 0.9991355523340215,
            "familywise_lower_confidence_bound": 0.9955526216752136,
        },
        "physical_gpu_1": {
            "load_order": "hybrid_first",
            "median_hybrid_over_fp8_throughput_ratio": 1.0018857120902784,
            "familywise_lower_confidence_bound": 0.9942020778675713,
        },
        "physical_gpu_2": {
            "load_order": "fp8_first",
            "median_hybrid_over_fp8_throughput_ratio": 0.9950546090740336,
            "familywise_lower_confidence_bound": 0.986977307944782,
        },
        "physical_gpu_3": {
            "load_order": "hybrid_first",
            "median_hybrid_over_fp8_throughput_ratio": 1.0107084554595036,
            "familywise_lower_confidence_bound": 1.0096913992642902,
        },
    }
    for value in speed.values():
        value.update({
            "noninferiority_ratio": 0.90,
            "speed_and_memory_pass": True,
        })
    value = {
        "schema": "eggroll-es-v26a-compat-speed-evidence",
        "status": "valid_completed_train_only_gate_failed_safe",
        "contracts": {
            "implementation_commit": (
                "6b338e0af96b54353754d15465487a30440edaad"
            ),
            "preregistration_file_sha256": (
                "ffe596314299b13e0691ee3eaea8df0f123dd06a767b768edeab5c0a6bf6a4cf"
            ),
            "preregistration_content_sha256": (
                "d8b3cc6f5837606f43c0fa26841b223a974d93fb4117c91fde553dae67af4767"
            ),
            "implementation_bundle_sha256": (
                "0f95c530e77247c54b3d1438c8d40d52ce7de159fc51cbf778a73bd90dfcb6b2"
            ),
            "runtime_recipe_content_sha256": (
                "677033c9d9efedf9d75612c405602f58ae8e22a7062f4ebe5fe90b24d8ad1b66"
            ),
        },
        "artifacts": {
            "report": {
                "file_sha256": (
                    "271821d0cfa0682fc1b5ea15b7ab65de569ddce55e0dc361119eeacd8fb9b003"
                ),
                "content_sha256": (
                    "94666feedd3eb51abe6293f6d56d605ce955e36f6838f4611264a10ced60b834"
                ),
            },
            "attempt": {
                "file_sha256": (
                    "683b14b99128f2eb7d6c565cb80148c4178e614ef86d0e23760fbb40fa1f018d"
                ),
                "content_sha256": (
                    "3b86d2e5b2915383ad75e85b700ae031ddda61d550042c21bc8656a9d4face03"
                ),
                "status": "complete",
                "phase": "after_actor_cleanup_and_compact_report",
            },
        },
        "runtime_integrity": {
            "wave_count": 2,
            "actors_per_wave": 4,
            "generation_calls_per_actor_per_wave": 10,
            "total_engine_generation_calls": 80,
            "total_generation_requests": 22_400,
            "processed_tokens_per_engine_call": 15_471,
            "physical_gpu_ids_discovered_and_verified": [0, 1, 2, 3],
            "same_backend_outputs_exact_across_four_appearances": True,
            "all_timed_outputs_equal_deterministic_reference": True,
            "fresh_model_loads_between_counterbalanced_waves": True,
            "interwave_cleanup_poll_count": 1,
            "interwave_cleanup_elapsed_milliseconds": 0,
            "all_runtime_integrity_audits_passed": True,
        },
        "aggregate_result": {
            "equivalence": {
                "metrics_identical_across_all_four_physical_gpu_pairs": True,
                "metrics": equivalence,
                "thresholds": thresholds,
                "passing_physical_gpu_pairs": 0,
                "required_passing_physical_gpu_pairs": 4,
                "pass": False,
            },
            "speed": speed,
            "all_four_speed_noninferiority_pairs_passed": True,
            "full_fp8_peak_nvml_fraction": 0.8474874089511375,
            "hybrid_peak_nvml_fraction": 0.8469970476161288,
            "maximum_allowed_peak_nvml_fraction": 0.95,
            "all_eight_peak_vram_cells_passed": True,
            "overall_gate_passed": False,
            "sole_failed_gate": "behavioral_equivalence",
        },
        "decision": {
            "retain_existing_full_fp8_model": True,
            "authorize_hybrid_training_ab": False,
            "direct_model_adoption_authorized": False,
            "model_update_or_checkpoint_authorized": False,
            "dataset_promotion_authorized": False,
            "validation_heldout_ood_or_benchmark_open_authorized": False,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    value = build_evidence()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return value


if __name__ == "__main__":
    main()
