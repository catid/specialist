#!/usr/bin/env python3
"""Seal V51 direct-pinned-master transition microbenchmark before launch."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_transition_microbenchmark_v51 as planning
import run_lora_es_transition_microbenchmark_v51 as runtime


def build_v51() -> dict:
    parent = runtime._load_parent_v51()
    design = planning.build_design_v51()
    recipe = dict(parent["recipe"])
    recipe["worker_extension"] = runtime.WORKER_EXTENSION_V51
    result = {
        "schema": (
            "matched-lora-es-direct-pinned-master-transition-"
            "preregistration-v51"
        ),
        "status": "preregistered_before_train_only_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Measure and remove the repeated exact restore-to-master before "
            "the next antithetic materialization, while reconstructing every "
            "runtime state independently from the pinned canonical FP32 master."
        ),
        "gpu_launch_authorized": True,
        "microbenchmark_only": True,
        "optimizer_update_authorized": False,
        "quality_selection_or_promotion_authorized": False,
        "protected_semantic_access_authorized": False,
        "shadow_ood_holdout_or_benchmark_authorized": False,
        "parents": {
            "v48b_preregistration": {
                "path": str(runtime.PARENT_PREREGISTRATION),
                "file_sha256": runtime.PARENT_PREREGISTRATION_FILE_SHA256,
                "content_sha256": (
                    runtime.PARENT_PREREGISTRATION_CONTENT_SHA256
                ),
                "requests_objectives_and_sampling_inherited": True,
            },
        },
        "recipe": recipe,
        "design": design,
        "instrumentation": {
            "per_state_phases": [
                "materialize", "generate", "score", "restore", "drain",
            ],
            "states": 16,
            "actors_per_state": 4,
            "required_actor_phase_receipts": 320,
            "worker_clocks_are_duration_only_not_cross_process_comparable": True,
            "controller_observed_generation_and_drain_completion": True,
            "timing_does_not_change_objective_or_gate": True,
        },
        "transition_contract": {
            "state_order": "direction ascending; plus then minus",
            "sole_runtime_slot_retained": True,
            "immutable_fp32_master_retained": True,
            "every_candidate_sha256_precommitted": True,
            "every_runtime_values_sha256_precommitted": True,
            "direct_candidate_to_candidate_delta_forbidden": True,
            "intermediate_bf16_algebraic_restore_forbidden": True,
            "intermediate_exact_master_restores_eliminated": 15,
            "final_exact_master_restore_required_on_success": True,
            "emergency_exact_master_restore_required_on_failure": True,
            "final_all_rank_state_certificate_required": True,
        },
        "performance_interpretation": {
            "historical_v50_population_wall_seconds": design[
                "historical_train_only_baseline"
            ]["v50"]["wall_seconds"],
            "historical_v50_median_synchronized_idle_gap_seconds": design[
                "historical_train_only_baseline"
            ]["v50"]["synchronized_idle_gap_median_seconds"],
            "useful_speed_signal": (
                "at least 10% lower population wall and at least 20% lower "
                "median synchronized idle gap than sealed V50"
            ),
            "speed_signal_is_descriptive_not_quality_promotion": True,
            "independent_population_scores_are_not_required_bit_exact": True,
            "state_candidate_and_runtime_identities_are_required_bit_exact": True,
        },
        "required_gates": {
            "all_16x4x5_timing_receipts": True,
            "all_16_candidate_and_runtime_identities_exact": True,
            "all_candidates_reconstructed_from_pinned_master": True,
            "no_cumulative_candidate_transition": True,
            "15_intermediate_restores_elided": True,
            "final_all_four_exact_master_runtime_restore": True,
            "post_population_base_score_exact": True,
            "cpu_scorers_have_zero_gpu_visibility": True,
            "all_four_gpus_attributed_positive": True,
            "strict_cleanup_and_gpu_idle": True,
            "no_optimizer_update_or_snapshot": True,
        },
        "access_contract": {
            "runtime_train_paths_only": parent["access_contract"][
                "only_runtime_train_paths_may_open"
            ],
            "sealed_train_only_timing_sources": [
                str(planning.V48B_POPULATION),
                str(planning.V50_POPULATION),
                str(planning.V48B_GPU_LOG),
                str(planning.V50_GPU_LOG),
            ],
            "dry_run_reads_train_semantics": False,
            "dry_run_loads_model_or_gpu": False,
            "protected_eval_ood_holdout_or_benchmark_path_opened": False,
        },
        "artifacts": {
            "attempt": str(runtime.ATTEMPT),
            "run_directory": str(runtime.RUN_DIR),
            "report": str(runtime.REPORT),
            "failure": str(runtime.FAILURE),
            "gpu_log": str(runtime.GPU_LOG),
            "timing": str(runtime.TIMING_ARTIFACT),
            "population": str(runtime.POPULATION_ARTIFACT),
            "snapshot": None,
        },
        "implementation_bindings": runtime.implementation_bindings_v51(),
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    result["content_sha256_before_self_field"] = (
        planning.canonical_sha256_v51(result)
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", type=Path, default=runtime.PREREGISTRATION)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v51()
    runtime.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": planning.file_sha256_v51(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
