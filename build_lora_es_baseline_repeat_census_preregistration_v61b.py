#!/usr/bin/env python3
"""Seal V61B before any train-row, model, or GPU access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_baseline_repeat_census_v61b as analysis
import lora_es_nested_population_v52 as design_v52
import run_lora_es_baseline_repeat_census_v61b as runtime
import run_lora_es_baseline_census_v61a as runtime_v61a


def build_v61b() -> dict:
    value = {
        "schema": "v61b-v434-common-seed-repeat-census-preregistration",
        "status": "preregistered_before_v61b_train_model_or_gpu_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "train_semantic_access_authorized": True,
        "selection_update_or_promotion_authorized": False,
        "eval_ood_shadow_holdout_access_authorized": False,
        "purpose": (
            "Separate within-actor repeat variance from cross-actor variance by "
            "running two identical, strictly sequential greedy passes with one "
            "common seed at the unchanged V434 LoRA state."
        ),
        "scientific_scope": {
            "characterization_only": True,
            "candidate_comparison_or_ranking": False,
            "adapter_update_or_master_commit": False,
            "causal_variance_source_claim_authorized": False,
            "future_hpo_launch_authorized": False,
        },
        "access_contract": {
            "only_live_semantic_paths_may_open": [
                str(design_v52.TRAIN_DATASET_V52),
                str(design_v52.TRAIN_MEMBERSHIP_V52),
            ],
            "builder_reads_train_rows": False,
            "dry_run_reads_train_rows": False,
            "builder_or_runtime_opens_v61a_row_level_evidence": False,
            "builder_reads_model_or_adapter_artifacts": False,
            "dry_run_loads_model_or_gpu": False,
            "eval_ood_shadow_holdout_or_benchmark_opened": False,
        },
        "fixed_census_recipe": {
            "base_model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "train_dataset": str(design_v52.TRAIN_DATASET_V52),
            "train_dataset_file_sha256": design_v52.DATASET_SHA256_V52,
            "train_rows": analysis.TRAIN_ROWS_V61B,
            "train_conflict_units": analysis.TRAIN_CONFLICT_UNITS_V61B,
            "membership": str(design_v52.TRAIN_MEMBERSHIP_V52),
            "membership_file_sha256": design_v52.MEMBERSHIP_SHA256_V52,
            "membership_content_sha256": design_v52.MEMBERSHIP_CONTENT_SHA256_V52,
            "train_bundle_content_sha256": design_v52.TRAIN_BUNDLE_CONTENT_SHA256_V52,
            "source_v434": str(design_v52.SOURCE_V52),
            "source_weights_file_sha256": design_v52.SOURCE_WEIGHTS_SHA256_V52,
            "source_config_file_sha256": design_v52.SOURCE_CONFIG_SHA256_V52,
            "staged_v434": str(design_v52.STAGED_V52),
            "staged_weights_file_sha256": design_v52.STAGED_WEIGHTS_SHA256_V52,
            "staged_config_file_sha256": design_v52.STAGED_CONFIG_SHA256_V52,
            "staged_manifest_file_sha256": design_v52.STAGED_MANIFEST_FILE_SHA256_V52,
            "canonical_fp32_master_sha256": design_v52.MASTER_SHA256_V52,
            "bf16_runtime_values_sha256": design_v52.MASTER_RUNTIME_SHA256_V52,
            "worker_extension": runtime.WORKER_EXTENSION_V61B,
            "physical_gpu_ids": [0, 1, 2, 3],
            "actors": analysis.ACTORS_V61B,
            "tensor_parallel_size_per_actor": 1,
            "sequential_passes_per_actor": analysis.PASSES_V61B,
            "strict_pass_order": [0, 1],
            "all_actors_each_pass_receive_identical_ordered_448_rows": True,
            "pass_1_starts_only_after_all_pass_0_outputs_resolve": True,
            "common_generation_seed": analysis.COMMON_GENERATION_SEED_V61B,
            "generation_params_without_seed": dict(
                analysis.GENERATION_PARAMS_WITHOUT_SEED_V61B
            ),
            "f1_absolute_delta_thresholds": list(
                analysis.F1_ABSOLUTE_DELTA_THRESHOLDS_V61B
            ),
            "total_completions": 3584,
            "worst_case_decode_tokens": 3584 * analysis.MAX_GENERATION_TOKENS_V61B,
        },
        "analysis_contract": {
            "within_actor_pass_repeat": {
                "per_actor_absolute_f1_delta_threshold_counts": True,
                "per_actor_exact_label_disagreement": True,
                "per_actor_nonzero_label_disagreement": True,
                "conflict_units_with_any_row_actor_delta_threshold_counts": True,
            },
            "cross_actor_same_seed_each_pass": {
                "row_f1_range_threshold_counts": True,
                "pairwise_mean_absolute_f1_delta": True,
                "pairwise_exact_and_nonzero_label_disagreement": True,
                "conflict_unit_threshold_counts": True,
            },
            "v61a_distinct_seed_comparison": (
                "bound aggregate hashes and fixed counts only; no row-level reopen"
            ),
            "thresholds_or_interpretation_changed_after_outcomes": False,
        },
        "evidence_contract": {
            "row_fields": [
                "row_index", "row_sha256", "unit_identity_sha256", "row_count",
            ],
            "metric_fields": [
                "actor_rank", "pass_index", "generation_seed", "f1", "exact", "nonzero",
            ],
            "question_answer_generation_or_prediction_hash_persisted": False,
            "rows": 448, "conflict_units": 208,
            "actors_per_row_pass": 4, "passes_per_row": 2,
        },
        "bound_v61a_distinct_seed_aggregate": dict(
            analysis.V61A_BOUND_AGGREGATES
        ),
        "runtime": dict(design_v52.RUNTIME_V52),
        "required_python": str(design_v52.REQUIRED_PYTHON_V52),
        "implementation_bindings": runtime.implementation_bindings_v61b(),
        "artifacts": runtime._artifacts_v61b(),
        "required_gates": {
            "exclusive_idle_four_gpu_preflight": True,
            "exact_v434_inputs_and_master": True,
            "strictly_sequential_complete_passes": True,
            "448_completions_per_actor_per_pass": True,
            "all_four_gpus_attributed_positive": True,
            "numeric_hash_only_evidence": True,
            "strict_four_engine_cleanup_and_idle": True,
            "selection_update_promotion_and_protected_access_zero": True,
        },
        "raw_question_answer_or_generation_text_may_be_persisted": False,
        "protected_semantics_opened": False,
        "eval_ood_shadow_or_holdout_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    }
    value["content_sha256_before_self_field"] = analysis.canonical_sha256_v61b(value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.PREREGISTRATION))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists(): raise FileExistsError(output)
    value = build_v61b()
    runtime_v61a.atomic_json_v61a(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime_v61a.file_sha256_v61a(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "builder_train_rows_opened": 0,
        "builder_v61a_row_evidence_opened": False,
        "builder_model_or_gpu_accessed": False,
        "gpu_launch_authorized": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
