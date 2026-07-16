#!/usr/bin/env python3
"""Seal V53 before any model, Ray, GPU, or protected-data access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as v52
import lora_es_sigma_discrimination_v53 as design


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments/eggroll_es_hpo/preregistrations/lora_es_sigma_discrimination_v53.json"
RUN_DIR = ROOT / "experiments/eggroll_es_hpo/runs/v53_lora_es_sigma_discrimination"
RETRY5_PREREG = ROOT / "experiments/eggroll_es_hpo/preregistrations/matched_lora_es_nested_p8_vs_p16_v52_retry5.json"
RETRY5_POPULATION = ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16_retry5/nested_population_v52.json"
RETRY5_FAILURE = ROOT / "experiments/eggroll_es_hpo/runs/v52_matched_lora_es_nested_p8_vs_p16_retry5/failure_v52.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_hash(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["content_sha256_before_self_field"]


def implementation_bindings() -> dict:
    return {name: {"path": str(ROOT / name), "file_sha256": file_sha256(ROOT / name)} for name in (
        "lora_es_sigma_discrimination_v53.py",
        "run_lora_es_sigma_discrimination_v53.py",
    )}


def build_v53() -> dict:
    retry5 = json.loads(RETRY5_POPULATION.read_text(encoding="utf-8"))
    p16 = retry5["arms"]["p16"]["reliability"]
    result = {
        "schema": "lora-es-sigma-discrimination-preregistration-v53",
        "status": "sealed_before_v53_model_ray_gpu_or_protected_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Find the smallest preregistered P16 perturbation sigma whose "
            "train-only signal is distinguishable from measured inference noise."
        ),
        "scientific_rationale": {
            "retry5_sigma": v52.SIGMA_V52,
            "retry5_p16_estimated_signal_standard_deviation": p16[
                "estimated_signal_standard_deviation"
            ],
            "retry5_fresh_maximum_actor_spread": p16[
                "fresh_calibration_observed_maximum_actor_spread"
            ],
            "retry5_p16_signal_to_noise_clearance_ratio": (
                p16["estimated_signal_standard_deviation"]
                / p16["fresh_calibration_observed_maximum_actor_spread"]
            ),
            "sigma_grid": list(design.SIGMAS_V53),
            "grid_reason": (
                "geometric doubling from 2x to 8x Retry5; under a local linear "
                "response approximation 0.0024 is the first grid point expected "
                "to clear Retry5 noise, with one bounded fallback"
            ),
            "linearity_not_assumed_for_acceptance": True,
            "result_authorizes_measurement_sigma_selection_only": True,
            "optimizer_step_quality_or_ood_safety_not_inferred": True,
        },
        "design": {
            "population_size": design.POPULATION_SIZE_V53,
            "fixed_seeds": list(design.SEEDS_V53),
            "arms": design.arm_contracts_v53(),
            "arm_order": "sigma_ascending",
            "adaptive_stop": "stop_after_first_arm_passing_every_fixed_gate",
            "maximum_arms": 3,
            "states_per_arm": 32,
            "maximum_states": 96,
            "actors": 4,
            "same_train_rows_generation_panel_sampling_and_actor_reducer": True,
            "one_shared_fresh_numeric_calibration_before_all_arms": True,
        },
        "fixed_gates": {
            "reliability_minimum": design.MINIMUM_RELIABILITY_V53,
            "split_half_spearman_minimum": design.MINIMUM_SPLIT_HALF_SPEARMAN_V53,
            "estimated_signal_std_strictly_greater_than_fresh_max_actor_spread": True,
            "fresh_max_actor_spread_at_or_below_historical_ceiling": design.HISTORICAL_CALIBRATION_CEILING_V53,
            "no_retroactive_threshold_grid_seed_or_order_changes": True,
            "all_32_candidates_unique_and_not_master_fp32": True,
            "all_32_candidates_unique_and_not_master_bf16": True,
            "four_actor_fp32_and_bf16_consensus_before_each_score": True,
            "each_candidate_direct_from_pinned_master": True,
            "plus_minus_antithetic_coverage_exact": True,
            "exact_master_restore_and_quiescence_after_each_arm": True,
        },
        "authorization": {
            "gpu_launch": True,
            "optimizer_update": False,
            "projection": False,
            "train_gate": False,
            "candidate_snapshot": False,
            "ood_shadow_benchmark_or_holdout": False,
            "protected_semantics": False,
            "sealed_holdout": False,
        },
        "fixed_recipe": {
            "model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "source": str(v52.SOURCE_V52),
            "staged": str(v52.STAGED_V52),
            "master_sha256": v52.MASTER_SHA256_V52,
            "master_runtime_values_sha256": v52.MASTER_RUNTIME_SHA256_V52,
            "dataset": str(v52.TRAIN_DATASET_V52),
            "dataset_sha256": v52.DATASET_SHA256_V52,
            "train_bundle_content_sha256": v52.TRAIN_BUNDLE_CONTENT_SHA256_V52,
            "membership_file_sha256": v52.MEMBERSHIP_SHA256_V52,
            "generation_panel_file_sha256": v52.SUBSET_FILE_SHA256_V52,
            "request_order_sha256": v52.REQUEST_ORDER_SHA256_V52,
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
            "physical_gpu_ids": [0, 1, 2, 3],
            "worker_extension": "eggroll_es_worker_lora_v52.LoRAAdapterStateWorkerExtensionV52",
        },
        "runtime": dict(v52.RUNTIME_V52),
        "sealed_retry5_evidence": {
            "preregistration": {"path": str(RETRY5_PREREG), "file_sha256": file_sha256(RETRY5_PREREG), "content_sha256": self_hash(RETRY5_PREREG)},
            "population": {"path": str(RETRY5_POPULATION), "file_sha256": file_sha256(RETRY5_POPULATION), "content_sha256": self_hash(RETRY5_POPULATION)},
            "failure": {"path": str(RETRY5_FAILURE), "file_sha256": file_sha256(RETRY5_FAILURE), "content_sha256": self_hash(RETRY5_FAILURE)},
        },
        "artifacts": {
            "attempt": str(RUN_DIR.parent / ".v53_lora_es_sigma_discrimination.attempt.json"),
            "run_directory": str(RUN_DIR),
            "population": str(RUN_DIR / "nested_population_v52.json"),
            "report": str(RUN_DIR / "sigma_discrimination_report_v53.json"),
            "gpu_log": str(RUN_DIR / "gpu_activity_v53.jsonl"),
            "coordinator_log": str(RUN_DIR.parent / "v53_lora_es_sigma_discrimination.coordinator.log"),
            "internal_fail_closed_receipt": str(RUN_DIR / "failure_v52.json"),
            "sealed_holdout_artifact": None,
        },
        "implementation_bindings": implementation_bindings(),
        "protected_semantics_opened": False,
        "sealed_holdout_opened": False,
    }
    result["content_sha256_before_self_field"] = design.canonical_sha256_v53(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = build_v53()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"path": str(args.output), "file_sha256": file_sha256(args.output), "content_sha256": result["content_sha256_before_self_field"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
