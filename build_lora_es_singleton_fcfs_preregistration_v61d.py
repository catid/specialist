#!/usr/bin/env python3
"""Seal matched V61D singleton-FCFS calibration before live access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as design_v52
import lora_es_singleton_fcfs_calibration_v61d as analysis
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_singleton_fcfs_calibration_v61d as runtime
import run_lora_es_paired_null_calibration_v61c as runtime_v61c


def build_v61d() -> dict:
    audit = runtime._read_support_audit_v61d()
    value = {
        "schema": "v61d-v434-singleton-fcfs-paired-evaluator-preregistration",
        "status": "preregistered_before_v61d_train_semantics_model_or_gpu_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "staged_train_semantic_access_authorized": True,
        "selection_hpo_update_or_promotion_authorized": False,
        "eval_ood_shadow_or_holdout_access_authorized": False,
        "purpose": (
            "Repeat V61C unchanged at the exact V434 state and identical panel, "
            "labels, periods, metrics, bootstrap, and frozen thresholds while "
            "holding each actor to synchronous singleton FCFS scheduling."
        ),
        "scientific_scope": {
            "matched_alpha_zero_identical_state_characterization_only": True,
            "only_scientific_change_from_v61c": "runtime_determinism_controls",
            "candidate_comparison_selection_ranking_or_update": False,
            "future_hpo_launch_authorized_by_this_run": False,
            "causal_source_of_execution_nondeterminism_claim_authorized": False,
        },
        "matched_v61c_contract": {
            "same_staged_rows_and_order": True,
            "same_ranking_and_exact_sentinel_roles": True,
            "same_label_plan_periods_and_request_type_order": True,
            "same_generation_and_teacher_forced_metrics": True,
            "same_primary_and_diagnostic_bootstrap": True,
            "same_frozen_effect_scales_and_eligibility_thresholds": True,
            "threshold_relaxation": False,
            "v61c_finalizer_file_sha256": runtime.V61C_FINALIZER_FILE_SHA256,
            "v61c_finalizer_content_sha256": runtime.V61C_FINALIZER_CONTENT_SHA256,
        },
        "access_contract": {
            "only_live_semantic_paths_may_open": [
                str(runtime_v61c.STAGED_DATASET), str(runtime_v61c.STAGED_PANEL),
            ],
            "full_v52_train_or_membership_may_open_live": False,
            "builder_or_dry_run_reads_staged_rows_or_panel": False,
            "builder_or_dry_run_loads_model_or_gpu": False,
            "v61a_v61b_or_v61c_row_level_evidence_may_open": False,
            "holdback_ood_shadow_terminal_or_benchmark_may_open": False,
        },
        "installed_vllm_support_audit": {
            "path": str(runtime.AUDIT),
            "file_sha256": runtime.AUDIT_FILE_SHA256,
            "content_sha256": runtime.AUDIT_CONTENT_SHA256,
            "status": audit["status"],
            "singleton_fcfs_controls_supported": True,
            "gpu_model_or_train_semantics_accessed": False,
        },
        "fixed_calibration_recipe": {
            "base_model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "staged_dataset": str(runtime_v61c.STAGED_DATASET),
            "staged_dataset_file_sha256": runtime_v61c.STAGED_DATASET_FILE_SHA256,
            "staged_panel": str(runtime_v61c.STAGED_PANEL),
            "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
            "staged_panel_content_sha256": runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
            "rows": analysis.ROWS_V61D,
            "ranking_units": analysis.RANKING_UNITS_V61D,
            "exact_sentinel_units": analysis.EXACT_SENTINEL_UNITS_V61D,
            "holdback_units": 0, "holdback_documents": 0,
            "source_v434": str(design_v52.SOURCE_V52),
            "source_weights_file_sha256": design_v52.SOURCE_WEIGHTS_SHA256_V52,
            "source_config_file_sha256": design_v52.SOURCE_CONFIG_SHA256_V52,
            "staged_v434": str(design_v52.STAGED_V52),
            "staged_weights_file_sha256": design_v52.STAGED_WEIGHTS_SHA256_V52,
            "staged_config_file_sha256": design_v52.STAGED_CONFIG_SHA256_V52,
            "staged_manifest_file_sha256": design_v52.STAGED_MANIFEST_FILE_SHA256_V52,
            "canonical_fp32_master_sha256": design_v52.MASTER_SHA256_V52,
            "bf16_runtime_values_sha256": design_v52.MASTER_RUNTIME_SHA256_V52,
            "worker_extension": runtime.WORKER_EXTENSION_V61D,
            "physical_gpu_ids": [0, 1, 2, 3], "actors": 4,
            "tensor_parallel_size_per_actor": 1, "sequential_periods": 4,
            "label_plan": dict(analysis.LABEL_PLAN_V61D),
            "request_type_order": dict(analysis.REQUEST_TYPE_ORDER_V61D),
            "pair_periods": [list(pair) for pair in analysis.PAIR_PERIODS_V61D],
            "common_generation_seed": analysis.COMMON_GENERATION_SEED_V61D,
            "generation_params_without_seed": dict(
                analysis.GENERATION_PARAMS_WITHOUT_SEED_V61D
            ),
            "teacher_forced_params_without_seed": dict(
                analysis.TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61D
            ),
            "runtime_determinism_controls": dict(analysis.RUNTIME_CONTROLS_V61D),
            "batch_composition_contract": (
                "at most one active sequence per actor under synchronous FCFS"
            ),
            "v61c_v27c_tuned_table_identity_retained": True,
            "global_batch_invariance_claimed": False,
            "generation_completions": 1088,
            "teacher_forced_requests": 1088,
            "alpha": 0.0, "sigma_or_direction": None,
            "adapter_update_or_candidate_materialization": False,
        },
        "unchanged_analysis_contract": {
            "generation_f1_instability_bands": list(
                analysis.F1_INSTABILITY_BANDS_V61D
            ),
            "teacher_logprob_instability_bands": list(
                analysis.LOGPROB_INSTABILITY_BANDS_V61D
            ),
            "bootstrap_replicates": analysis.BOOTSTRAP_REPLICATES_V61D,
            "bootstrap_seed": analysis.BOOTSTRAP_SEED_V61D,
            "single_replica_diagnostic_seed": (
                analysis.SINGLE_REPLICA_DIAGNOSTIC_BOOTSTRAP_SEED_V61D
            ),
            "one_sided_alpha": analysis.BOOTSTRAP_ALPHA_V61D,
            "primary_preserves_and_averages_all_actor_pair_replicas": 8,
            "generated_f1_practical_effect_scale": (
                analysis.F1_PRACTICAL_EFFECT_SCALE_V61D
            ),
            "teacher_logprob_practical_effect_scale": (
                analysis.LOGPROB_PRACTICAL_EFFECT_SCALE_V61D
            ),
            "teacher_max_absolute_point_null": (
                analysis.LOGPROB_PRIMARY_MAX_ABSOLUTE_POINT_NULL_V61D
            ),
            "teacher_max_ci_halfwidth": (
                analysis.LOGPROB_PRIMARY_MAX_CI_HALFWIDTH_V61D
            ),
            "teacher_max_repeat_mean_absolute_delta": (
                analysis.LOGPROB_PRIMARY_MAX_REPEAT_MEAN_ABSOLUTE_DELTA_V61D
            ),
            "teacher_normalized_ci_must_not_exceed_generated_f1": True,
            "any_individual_exact_sentinel_flip_fails": True,
            "thresholds_changed_after_v61c_outcomes": False,
        },
        "runtime": dict(design_v52.RUNTIME_V52),
        "required_python": str(design_v52.REQUIRED_PYTHON_V52),
        "implementation_bindings": runtime.implementation_bindings_v61d(),
        "artifacts": runtime._artifacts_v61d(),
        "required_gates": {
            "exclusive_idle_four_gpu_preflight": True,
            "all_actor_identities_verify_singleton_fcfs_eager_and_batch_invariant_false": True,
            "all_actor_identities_verify_exact_v27c_tuned_table": True,
            "exact_v434_state_before_and_after_every_period": True,
            "all_four_gpus_attributed_positive": True,
            "numeric_hash_only_evidence": True,
            "strict_four_engine_cleanup_and_idle": True,
            "selection_update_candidate_and_protected_access_zero": True,
        },
        "raw_question_answer_or_generation_text_may_be_persisted": False,
        "protected_semantics_opened": False,
        "eval_ood_shadow_or_holdout_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    }
    value["content_sha256_before_self_field"] = analysis.canonical_sha256_v61d(
        value
    )
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.PREREGISTRATION))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v61d()
    runtime_v61a.atomic_json_v61a(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime_v61a.file_sha256_v61a(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launch_authorized": True,
        "builder_train_model_or_gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
