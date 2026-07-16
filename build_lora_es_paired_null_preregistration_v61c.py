#!/usr/bin/env python3
"""Seal V61C before staged train semantics, model, or GPU access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as design_v52
import lora_es_paired_null_calibration_v61c as analysis
import lora_es_robust_paired_hpo_v61 as preview_design
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_paired_null_calibration_v61c as runtime


def build_v61c() -> dict:
    value = {
        "schema": "v61c-v434-identical-state-paired-evaluator-preregistration",
        "status": "preregistered_before_v61c_train_semantics_model_or_gpu_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "staged_train_semantic_access_authorized": True,
        "selection_hpo_update_or_promotion_authorized": False,
        "eval_ood_shadow_or_holdout_access_authorized": False,
        "purpose": (
            "Measure paired generated-F1 and teacher-forced answer-logprob null "
            "noise at one unchanged V434 LoRA state before choosing the V61 "
            "HPO primary fitness. Reference and candidate are counterbalanced "
            "labels for identical state executions."
        ),
        "scientific_scope": {
            "alpha_zero_identical_state_characterization_only": True,
            "candidate_comparison_selection_or_ranking": False,
            "perturbation_candidate_materialization_or_adapter_update": False,
            "future_hpo_launch_authorized_by_this_run": False,
            "causal_source_of_execution_nondeterminism_claim_authorized": False,
        },
        "adaptive_design_provenance": {
            "v61a_baseline_model_outcomes_used_for_train_only_stratification": True,
            "future_candidate_outcomes_used_for_panel_selection": False,
            "protected_or_holdback_outcomes_used": False,
            "train_only_adaptive_design": True,
        },
        "access_contract": {
            "only_live_semantic_paths_may_open": [
                str(runtime.STAGED_DATASET), str(runtime.STAGED_PANEL),
            ],
            "full_v52_train_or_membership_may_open_live": False,
            "builder_reads_staged_train_rows_or_panel": False,
            "dry_run_reads_staged_train_rows_or_panel": False,
            "builder_reads_model_or_adapter_artifacts": False,
            "dry_run_loads_model_or_gpu": False,
            "v61a_or_v61b_row_level_evidence_may_open": False,
            "untouched_holdback_units_or_documents_may_open": False,
            "eval_ood_shadow_terminal_or_benchmark_may_open": False,
        },
        "fixed_calibration_recipe": {
            "base_model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "staged_dataset": str(runtime.STAGED_DATASET),
            "staged_dataset_file_sha256": runtime.STAGED_DATASET_FILE_SHA256,
            "staged_panel": str(runtime.STAGED_PANEL),
            "staged_panel_file_sha256": runtime.STAGED_PANEL_FILE_SHA256,
            "staged_panel_content_sha256": runtime.STAGED_PANEL_CONTENT_SHA256,
            "corrected_preview_file_sha256": inputs_preview_file_sha256(),
            "corrected_preview_content_sha256": inputs_preview_content_sha256(),
            "rows": analysis.ROWS_V61C,
            "ranking_units": analysis.RANKING_UNITS_V61C,
            "exact_sentinel_units": analysis.EXACT_SENTINEL_UNITS_V61C,
            "holdback_units": 0,
            "holdback_documents": 0,
            "conflict_units_are_document_disjoint_from_holdback": True,
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
            "worker_extension": runtime.WORKER_EXTENSION_V61C,
            "physical_gpu_ids": [0, 1, 2, 3],
            "actors": analysis.ACTORS_V61C,
            "tensor_parallel_size_per_actor": 1,
            "sequential_periods": analysis.PERIODS_V61C,
            "label_plan": dict(analysis.LABEL_PLAN_V61C),
            "request_type_order": dict(analysis.REQUEST_TYPE_ORDER_V61C),
            "pair_periods": [list(pair) for pair in analysis.PAIR_PERIODS_V61C],
            "all_reference_and_candidate_labels_use_identical_v434_state": True,
            "exact_state_certificate_before_and_after_every_period": True,
            "common_generation_seed": analysis.COMMON_GENERATION_SEED_V61C,
            "generation_params_without_seed": dict(
                analysis.GENERATION_PARAMS_WITHOUT_SEED_V61C
            ),
            "teacher_forced_params_without_seed": dict(
                analysis.TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61C
            ),
            "generation_completions": 1088,
            "teacher_forced_requests": 1088,
            "alpha": 0.0,
            "sigma_or_direction": None,
            "adapter_update_or_candidate_materialization": False,
        },
        "paired_analysis_contract": {
            "pairing_keys": [
                "unit_identity_sha256", "actor_rank", "pair_index",
            ],
            "generation_f1_instability_bands": list(
                analysis.F1_INSTABILITY_BANDS_V61C
            ),
            "teacher_logprob_instability_bands": list(
                analysis.LOGPROB_INSTABILITY_BANDS_V61C
            ),
            "bootstrap_replicates": analysis.BOOTSTRAP_REPLICATES_V61C,
            "bootstrap_seed": analysis.BOOTSTRAP_SEED_V61C,
            "one_sided_alpha": analysis.BOOTSTRAP_ALPHA_V61C,
            "primary_conflict_unit_cluster_bootstrap": {
                "resampled_axis": "conflict_unit",
                "within_unit_actor_pair_replicas_preserved_and_averaged": 8,
                "estimand": (
                    "mean conflict-unit paired delta after averaging all four "
                    "actors and both paired periods within each unit"
                ),
                "used_for_logprob_primary_eligibility": True,
            },
            "future_single_replica_noise_diagnostic": {
                "resampled_axis": (
                    "conflict_unit_then_one_uniform_actor_pair_replica"
                ),
                "not_the_intended_eight_replica_hpo_estimator": True,
                "bootstrap_seed": (
                    analysis.SINGLE_REPLICA_DIAGNOSTIC_BOOTSTRAP_SEED_V61C
                ),
                "used_for_logprob_primary_eligibility": False,
            },
            "teacher_forced_answer_scoring": {
                "implementation": (
                    "prepare_gold_answer_items_v4 plus "
                    "score_gold_answer_outputs_v4"
                ),
                "prompt_logprobs": 1,
                "per_example_mean_answer_token_logprob": True,
                "equal_conflict_unit_primary_aggregation": True,
            },
            "teacher_logprob_primary_eligibility": {
                "primary_cluster_null_interval_contains_zero": True,
                "maximum_absolute_point_null": (
                    analysis.LOGPROB_PRIMARY_MAX_ABSOLUTE_POINT_NULL_V61C
                ),
                "maximum_primary_cluster_ci_halfwidth": (
                    analysis.LOGPROB_PRIMARY_MAX_CI_HALFWIDTH_V61C
                ),
                "maximum_same_label_repeat_mean_absolute_delta": (
                    analysis.LOGPROB_PRIMARY_MAX_REPEAT_MEAN_ABSOLUTE_DELTA_V61C
                ),
                "logprob_practical_effect_scale": (
                    analysis.LOGPROB_PRACTICAL_EFFECT_SCALE_V61C
                ),
                "generated_f1_practical_effect_scale": (
                    analysis.F1_PRACTICAL_EFFECT_SCALE_V61C
                ),
                "normalized_primary_cluster_logprob_ci_halfwidth_must_not_exceed_generated_f1": True,
                "all_checks_required": True,
            },
            "sparse_exact_sentinel": {
                "separate_from_primary_fitness": True,
                "any_individual_paired_exact_label_delta_fails": True,
                "aggregate_zero_sums_are_diagnostics_only": True,
                "plus_one_minus_one_cancellation_cannot_pass": True,
            },
            "thresholds_or_interpretation_changed_after_live_outcomes": False,
        },
        "bound_train_only_aggregate_evidence": {
            "v61a_strata_file_sha256": preview_design.V61A_STRATA_FILE_SHA256,
            "v61a_strata_content_sha256": preview_design.V61A_STRATA_CONTENT_SHA256,
            "v61b_evidence_file_sha256": preview_design.V61B_EVIDENCE_FILE_SHA256,
            "v61b_evidence_content_sha256": preview_design.V61B_EVIDENCE_CONTENT_SHA256,
            "v61b_analysis_file_sha256": preview_design.V61B_ANALYSIS_FILE_SHA256,
            "v61b_analysis_content_sha256": preview_design.V61B_ANALYSIS_CONTENT_SHA256,
            "v61b_report_file_sha256": preview_design.V61B_REPORT_FILE_SHA256,
            "v61b_report_content_sha256": preview_design.V61B_REPORT_CONTENT_SHA256,
            "row_level_evidence_opened_by_builder_or_runtime": False,
        },
        "evidence_contract": {
            "row_identity_fields": [
                "request_index", "row_sha256", "unit_identity_sha256", "role",
            ],
            "generation_metric_fields": ["f1", "exact", "nonzero"],
            "teacher_metric_fields": [
                "mean_answer_token_logprob", "answer_token_count",
                "numeric_example_sha256",
            ],
            "raw_prompt_answer_prediction_generation_or_token_ids_persisted": False,
            "rows": 68, "actors_per_row_period": 4, "periods_per_row": 4,
        },
        "runtime": dict(design_v52.RUNTIME_V52),
        "required_python": str(design_v52.REQUIRED_PYTHON_V52),
        "implementation_bindings": runtime.implementation_bindings_v61c(),
        "artifacts": runtime._artifacts_v61c(),
        "required_gates": {
            "exclusive_idle_four_gpu_preflight": True,
            "exact_v434_inputs_master_and_runtime_state": True,
            "all_four_gpus_attributed_positive": True,
            "all_four_periods_and_both_request_types_complete": True,
            "state_unchanged_before_and_after_every_period": True,
            "numeric_hash_only_evidence": True,
            "strict_four_engine_cleanup_and_idle": True,
            "selection_update_candidate_materialization_and_protected_access_zero": True,
        },
        "raw_question_answer_or_generation_text_may_be_persisted": False,
        "protected_semantics_opened": False,
        "eval_ood_shadow_or_holdout_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v61c(value)
    )
    return value


def inputs_preview_file_sha256() -> str:
    # Kept as helpers so prereg tests can verify the corrected provenance
    # without opening the generated preview.
    import build_lora_es_paired_null_inputs_v61c as input_builder
    return input_builder.PREVIEW_FILE_SHA256


def inputs_preview_content_sha256() -> str:
    import build_lora_es_paired_null_inputs_v61c as input_builder
    return input_builder.PREVIEW_CONTENT_SHA256


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.PREREGISTRATION))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v61c()
    runtime_v61a.atomic_json_v61a(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime_v61a.file_sha256_v61a(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "builder_staged_train_rows_or_panel_opened": False,
        "builder_model_or_gpu_accessed": False,
        "gpu_launch_authorized": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
