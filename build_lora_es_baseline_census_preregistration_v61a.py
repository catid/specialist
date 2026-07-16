#!/usr/bin/env python3
"""Seal V61A before any V434 train row, model, or GPU access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_baseline_census_strata_v61a as strata
import lora_es_nested_population_v52 as design_v52
import run_lora_es_baseline_census_v61a as runtime


OUTPUT = runtime.PREREGISTRATION


def build_v61a() -> dict:
    value = {
        "schema": "v61a-v434-train-baseline-census-preregistration",
        "status": "preregistered_before_v61a_model_gpu_or_train_row_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "train_semantic_access_authorized": True,
        "eval_ood_shadow_holdout_access_authorized": False,
        "candidate_selection_update_or_promotion_authorized": False,
        "purpose": (
            "Characterize exact, partial, difficult, and actor-unstable "
            "generation support at the unchanged V434 LoRA state so a later, "
            "separately preregistered V61 document-block/bootstrap HPO can use "
            "a nondegenerate train-only generation panel."
        ),
        "scientific_scope": {
            "characterization_only": True,
            "candidate_comparison_or_ranking": False,
            "adapter_update_or_master_commit": False,
            "later_panel_or_hpo_launch_authorized_by_this_preregistration": False,
            "complete_census_even_if_stable_exact_support_fails": True,
            "stable_exact_shortfall_only_disables_later_panel_hpo_eligibility": True,
        },
        "access_contract": {
            "only_live_semantic_paths_may_open": [
                str(design_v52.TRAIN_DATASET_V52),
                str(design_v52.TRAIN_MEMBERSHIP_V52),
            ],
            "preregistration_builder_reads_train_rows": False,
            "dry_run_reads_train_rows": False,
            "preregistration_builder_reads_model_or_adapter_artifacts": False,
            "dry_run_loads_model_or_gpu": False,
            "prose_qa_proxy_eval_ood_shadow_holdout_or_benchmark_opened": False,
        },
        "fixed_census_recipe": {
            "base_model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "train_dataset": str(design_v52.TRAIN_DATASET_V52),
            "train_dataset_file_sha256": design_v52.DATASET_SHA256_V52,
            "train_rows": strata.TRAIN_ROWS_V61A,
            "train_conflict_units": strata.TRAIN_CONFLICT_UNITS_V61A,
            "membership": str(design_v52.TRAIN_MEMBERSHIP_V52),
            "membership_file_sha256": design_v52.MEMBERSHIP_SHA256_V52,
            "membership_content_sha256": design_v52.MEMBERSHIP_CONTENT_SHA256_V52,
            "train_bundle_content_sha256": (
                design_v52.TRAIN_BUNDLE_CONTENT_SHA256_V52
            ),
            "source_v434": str(design_v52.SOURCE_V52),
            "source_weights_file_sha256": design_v52.SOURCE_WEIGHTS_SHA256_V52,
            "source_config_file_sha256": design_v52.SOURCE_CONFIG_SHA256_V52,
            "staged_v434": str(design_v52.STAGED_V52),
            "staged_weights_file_sha256": design_v52.STAGED_WEIGHTS_SHA256_V52,
            "staged_config_file_sha256": design_v52.STAGED_CONFIG_SHA256_V52,
            "staged_manifest_file_sha256": (
                design_v52.STAGED_MANIFEST_FILE_SHA256_V52
            ),
            "canonical_fp32_master_sha256": design_v52.MASTER_SHA256_V52,
            "bf16_runtime_values_sha256": design_v52.MASTER_RUNTIME_SHA256_V52,
            "worker_extension": runtime.WORKER_EXTENSION_V61A,
            "physical_gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
            "all_four_actors_score_identical_ordered_448_rows": True,
            "actor_generation_seeds": list(
                strata.ACTOR_GENERATION_SEEDS_V61A
            ),
            "generation_params_without_seed": dict(
                strata.GENERATION_PARAMS_WITHOUT_SEED_V61A
            ),
            "greedy_completions": (
                strata.TRAIN_ROWS_V61A * strata.ACTORS_V61A
            ),
            "worst_case_decode_tokens": (
                strata.TRAIN_ROWS_V61A * strata.ACTORS_V61A
                * strata.MAX_GENERATION_TOKENS_V61A
            ),
        },
        "frozen_stratification_and_partition": {
            "strata": list(strata.STRATA_V61A),
            "partial_f1_minimum": strata.PARTIAL_F1_MINIMUM_V61A,
            "actor_f1_stability_atol": strata.ACTOR_F1_STABILITY_ATOL_V61A,
            "representative_seed": strata.REPRESENTATIVE_SEED_V61A,
            "holdback_seed": strata.HOLDBACK_SEED_V61A,
            "selection_seed": strata.SELECTION_SEED_V61A,
            "holdback_fraction_per_stratum": strata.HOLDBACK_FRACTION_V61A,
            "stable_exact_fail_closed_minima": {
                "total": strata.MINIMUM_STABLE_EXACT_UNITS_V61A,
                "selection_pool": (
                    strata.MINIMUM_STABLE_EXACT_SELECTION_UNITS_V61A
                ),
                "holdback": strata.MINIMUM_STABLE_EXACT_HOLDBACK_UNITS_V61A,
            },
            "quota_or_threshold_relaxation_after_outcomes": False,
            "stable_exact_shortfall_action": (
                "persist_complete_content_free_census_and_strata_then_set_"
                "later_v61_hpo_authorized_false"
            ),
        },
        "evidence_contract": {
            "persisted_row_identity_fields": [
                "row_index", "row_sha256", "unit_identity_sha256", "row_count",
            ],
            "persisted_per_actor_fields": [
                "actor_rank", "generation_seed", "f1", "exact", "nonzero",
            ],
            "question_answer_generation_or_output_hash_persisted": False,
            "numeric_row_and_unit_manifests_sha256_persisted": True,
            "all_448_rows_and_208_units_required": True,
            "all_four_actor_metrics_per_row_required": True,
        },
        "runtime": dict(design_v52.RUNTIME_V52),
        "required_python": str(design_v52.REQUIRED_PYTHON_V52),
        "implementation_bindings": runtime.implementation_bindings_v61a(),
        "artifacts": runtime._expected_artifacts_v61a(),
        "required_gates": {
            "exclusive_idle_four_gpu_preflight": True,
            "exact_v434_artifact_identities": True,
            "exact_v434_master_same_on_all_four_actors": True,
            "448_completions_per_actor": True,
            "four_independent_fixed_actor_seeds": True,
            "all_four_gpus_attributed_positive": True,
            "numeric_hash_only_evidence": True,
            "strict_four_engine_cleanup_and_idle": True,
            "eval_ood_shadow_holdout_access_zero": True,
            "candidate_selection_update_commit_or_promotion_zero": True,
        },
        "raw_question_answer_or_generation_text_may_be_persisted": False,
        "protected_semantics_opened": False,
        "eval_ood_shadow_or_holdout_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    }
    value["content_sha256_before_self_field"] = (
        strata.canonical_sha256_v61a(value)
    )
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v61a()
    runtime.atomic_json_v61a(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.file_sha256_v61a(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "builder_train_rows_opened": 0,
        "builder_model_or_gpu_accessed": False,
        "gpu_launch_authorized": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
