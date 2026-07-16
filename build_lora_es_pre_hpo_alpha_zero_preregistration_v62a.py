#!/usr/bin/env python3
"""Seal the generation-only V62A alpha-zero calibration before live access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as design_v52
import lora_es_pre_hpo_alpha_zero_calibration_v62a as analysis
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_paired_null_calibration_v61c as runtime_v61c
import run_lora_es_pre_hpo_alpha_zero_calibration_v62a as runtime


def build_v62a() -> dict:
    support = runtime._read_support_audit_v62a()
    runtime._read_v62_methodology_v62a()
    value = {
        "schema": "v62a-v434-pre-hpo-alpha-zero-generation-preregistration",
        "status": "preregistered_before_train_semantics_model_or_gpu_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "specific_alpha_zero_calibration_gpu_launch_authorized": True,
        "support_audit_alone_authorizes_gpu_launch": False,
        "builder_or_dry_run_performed_gpu_launch": False,
        "hpo_population_update_or_candidate_authorized": False,
        "ood_shadow_holdout_or_protected_access_authorized": False,
        "purpose": (
            "Run the actual intended four-actor generation-only HPO evaluator "
            "at alpha zero and unchanged V434 state, then fail closed unless "
            "the three V62 null-calibration gates pass."
        ),
        "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
        "v62_numeric_audit_identities": dict(
            analysis.V62_NUMERIC_AUDIT_IDENTITIES
        ),
        "v62_preregistration_identities": dict(
            analysis.V62_PREREGISTRATION_IDENTITIES
        ),
        "scientific_scope": {
            "fresh_pre_hpo_alpha_zero_characterization_only": True,
            "reference_and_candidate_are_counterbalanced_aliases_for_identical_"
            "v434_state": True,
            "actual_intended_hpo_generation_runtime": True,
            "adapter_perturbation_or_update": False,
            "candidate_state_or_persistent_artifact": False,
            "hpo_population_selection_or_promotion": False,
            "calibration_success_itself_authorizes_hpo": False,
        },
        "installed_runtime_support_audit": {
            "path": str(runtime.SUPPORT_AUDIT),
            "file_sha256": runtime.SUPPORT_AUDIT_FILE_SHA256,
            "content_sha256": runtime.SUPPORT_AUDIT_CONTENT_SHA256,
            "status": support["status"],
            "pre_hpo_alpha_zero_runtime_supported": True,
            "support_audit_authorizes_gpu_launch": False,
            "model_train_semantics_or_gpu_accessed": False,
        },
        "access_contract": {
            "only_live_semantic_paths_may_open": [
                str(runtime_v61c.STAGED_DATASET),
                str(runtime_v61c.STAGED_PANEL),
            ],
            "full_train_membership_holdback_ood_shadow_terminal_or_"
            "protected_may_open": False,
            "builder_or_dry_run_reads_staged_rows_or_panel": False,
            "builder_or_dry_run_loads_model_or_gpu": False,
            "live_runtime_may_load_only_pinned_qwen36_v434_and_staged_68_rows": True,
            "raw_question_answer_or_generation_text_may_be_persisted": False,
            "numeric_hash_only_evidence_required": True,
        },
        "fixed_calibration_recipe": {
            "base_model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "adapter_state": "V434",
            "staged_dataset": str(runtime_v61c.STAGED_DATASET),
            "staged_dataset_file_sha256": (
                runtime_v61c.STAGED_DATASET_FILE_SHA256
            ),
            "staged_panel": str(runtime_v61c.STAGED_PANEL),
            "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
            "staged_panel_content_sha256": (
                runtime_v61c.STAGED_PANEL_CONTENT_SHA256
            ),
            "rows": 68,
            "ranking_units": 64,
            "exact_sentinel_units": 4,
            "same_call_ranking_plus_sentinel_rows": True,
            "holdback_units": 0,
            "holdback_documents": 0,
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
            "worker_extension": runtime.WORKER_EXTENSION_V62A,
            "physical_gpu_ids": [0, 1, 2, 3],
            "actors": 4,
            "tensor_parallel_size_per_actor": 1,
            "sequential_periods": 4,
            "counterbalanced_pairs_per_actor": 2,
            "replicas_per_conflict_unit": 8,
            "label_plan": dict(analysis.LABEL_PLAN_V62A),
            "pair_periods": [
                list(pair) for pair in analysis.PAIR_PERIODS_V62A
            ],
            "generation_only": True,
            "request_type_per_period": "generation",
            "common_generation_seed": analysis.COMMON_GENERATION_SEED_V62A,
            "generation_params_without_seed": dict(
                analysis.GENERATION_PARAMS_WITHOUT_SEED_V62A
            ),
            "teacher_forced_requests": 0,
            "teacher_forced_metric_computed": False,
            "runtime_determinism_controls": dict(
                analysis.RUNTIME_CONTROLS_V62A
            ),
            "submitted_requests_per_actor_call": 68,
            "active_sequence_limit_per_actor": 68,
            "v27c_tuned_table_runtime_identity_retained": True,
            "global_batch_invariance_claimed": False,
            "generation_completions": 1088,
            "alpha": 0.0,
            "sigma_or_direction": None,
            "adapter_update_candidate_or_hpo_performed": False,
        },
        "primary_numeric_estimator": {
            "metric": "paired_generated_f1_delta",
            "within_unit_aggregation": "arithmetic_mean_all_8_replicas",
            "bootstrap_resampled_axis": "conflict_unit",
            "within_unit_replicas_preserved": 8,
            "bootstrap_replicates": analysis.BOOTSTRAP_REPLICATES_V62A,
            "bootstrap_seed": analysis.BOOTSTRAP_SEED_V62A,
            "one_sided_alpha": analysis.ONE_SIDED_ALPHA_V62A,
            "actor_influence": (
                "maximum absolute shift from full four-actor point to each "
                "leave-one-actor-out point"
            ),
            "teacher_forced_logprob_role": "absent_not_computed",
        },
        "required_alpha_zero_gates": {
            "generated_f1_primary_interval_must_contain_zero": True,
            "maximum_primary_ci_halfwidth_inclusive": (
                analysis.MAX_PRIMARY_CI_HALFWIDTH_V62A
            ),
            "maximum_actor_leave_one_out_shift_inclusive": (
                analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
            ),
            "all_comparisons_inclusive": True,
            "failure_action": "fail_closed_before_hpo_population_or_update",
            "success_does_not_authorize_hpo_population_or_update": True,
        },
        "exact_sentinel_diagnostics": {
            "stable_unit_identity_sha256": list(
                analysis.STABLE_EXACT_UNIT_SHA256_V62A
            ),
            "stable_per_unit_statistic": (
                "strict_majority_exact_pass_count_at_least_5_of_8"
            ),
            "stable_aggregate_statistic": (
                "reference_and_candidate_exact_pass_totals_across_24_replicas"
            ),
            "actor_unstable_stress_unit_identity_sha256": (
                analysis.ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62A
            ),
            "actor_unstable_role": "diagnostic_stress_unit_only",
            "used_in_alpha_zero_gate": False,
            "any_single_flip_aborts": False,
            "any_per_unit_eight_of_eight_failure_aborts": False,
        },
        "runtime": dict(design_v52.RUNTIME_V52),
        "required_python": str(design_v52.REQUIRED_PYTHON_V52),
        "implementation_bindings": runtime.implementation_bindings_v62a(),
        "artifacts": runtime._artifacts_v62a(),
        "required_integrity_gates": {
            "exclusive_idle_four_gpu_preflight": True,
            "all_four_tp1_actor_identities_exact": True,
            "all_actor_identities_verify_sync_fcfs_eager_bi_false_max68": True,
            "all_actor_identities_verify_exact_v27c_tuned_table": True,
            "exact_v434_state_before_and_after_every_period": True,
            "same_68_requests_submitted_in_each_actor_period_call": True,
            "all_four_gpus_attributed_positive": True,
            "strict_four_engine_cleanup_and_final_idle": True,
            "numeric_hash_only_evidence": True,
            "update_candidate_hpo_and_protected_access_zero": True,
        },
        "raw_question_answer_or_generation_text_may_be_persisted": False,
        "protected_semantics_opened": False,
        "ood_shadow_holdout_or_terminal_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v62a(value)
    )
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(runtime.PREREGISTRATION))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v62a()
    runtime_v61a.atomic_json_v61a(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime_v61a.file_sha256_v61a(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "specific_alpha_zero_calibration_gpu_launch_authorized": True,
        "hpo_population_update_or_candidate_authorized": False,
        "builder_train_model_or_gpu_accessed": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
