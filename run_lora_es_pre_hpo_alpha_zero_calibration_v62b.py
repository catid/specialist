#!/usr/bin/env python3
"""Four-GPU outcome-agnostic V62B pre-HPO alpha-zero calibration.

The live path fixes four unscored warmup periods followed by 24 scored
periods.  Warmup generations are discarded without scoring or persistence.
Reference and candidate are counterbalanced aliases for one unchanged V434
state.  This module implements no update, candidate, HPO, or protected access.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import audit_vllm_pre_hpo_alpha_zero_support_v62b as support_audit
import lora_es_nested_population_v52 as design_v52
import lora_es_pre_hpo_alpha_zero_calibration_v62b as analysis
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_nested_population_v52 as runtime_v52
import run_lora_es_paired_null_calibration_v61c as runtime_v61c
import run_lora_es_pre_hpo_alpha_zero_calibration_v62a as runtime_v62a


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v62b_v434_pre_hpo_alpha_zero_generation_calibration"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
EVIDENCE = (RUN_DIR / "alpha_zero_evidence_v62b.json").resolve()
ANALYSIS = (RUN_DIR / "alpha_zero_analysis_v62b.json").resolve()
REPORT = (RUN_DIR / "alpha_zero_report_v62b.json").resolve()
FAILURE = (RUN_DIR / "failure_v62b.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v62b.jsonl").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v434_pre_hpo_alpha_zero_generation_calibration_v62b.json"
).resolve()
SUPPORT_AUDIT = support_audit.OUTPUT
SUPPORT_AUDIT_FILE_SHA256 = (
    "462941e6d1e8879b8bfaecaf4718e0fe371b0c9eaa57aa68cfde913133ba7c0f"
)
SUPPORT_AUDIT_CONTENT_SHA256 = (
    "659d3a8b4500f2d8bb16da3250c863989575182b1ce820c937fd66243fa687e7"
)
WORKER_EXTENSION_V62B = runtime_v62a.WORKER_EXTENSION_V62A


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--execute", action="store_true")
    return value


def _read_support_audit_v62b() -> dict:
    if support_audit.file_sha256_v62b(SUPPORT_AUDIT) != (
        SUPPORT_AUDIT_FILE_SHA256
    ):
        raise RuntimeError("v62b installed-runtime support audit file changed")
    value = json.loads(SUPPORT_AUDIT.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    projection = value.get("intended_evaluator_projection", {})
    if (
        value.get("content_sha256_before_self_field")
        != SUPPORT_AUDIT_CONTENT_SHA256
        or support_audit.canonical_sha256_v62b(compact)
        != SUPPORT_AUDIT_CONTENT_SHA256
        or value.get("schema")
        != "v62b-installed-vllm-pre-hpo-alpha-zero-support-audit"
        or value.get("status") != "supported"
        or value.get("pre_hpo_alpha_zero_runtime_supported") is not True
        or value.get("requested_runtime_controls")
        != analysis.RUNTIME_CONTROLS_V62B
        or value.get("support_audit_authorizes_gpu_launch") is not False
        or value.get("model_train_semantics_or_gpu_accessed") is not False
        or projection.get("unscored_warmup_periods")
        != analysis.WARMUP_PERIODS_V62B
        or projection.get("scored_periods") != analysis.SCORED_PERIODS_V62B
        or projection.get("total_sequential_periods")
        != analysis.TOTAL_PERIODS_V62B
        or projection.get("scored_replicas_per_conflict_unit")
        != analysis.REPLICAS_PER_UNIT_V62B
        or projection.get("warmup_outputs_scored_or_persisted") is not False
    ):
        raise RuntimeError("v62b installed-runtime support changed")
    observed = {
        key: support_audit.file_sha256_v62b(path)
        for key, path in support_audit.v62a.v61e.SOURCE_PATHS.items()
    }
    if observed != value.get("installed_vllm_source_file_sha256"):
        raise RuntimeError("v62b installed vLLM source identities changed")
    return value


def _read_v62_methodology_v62b() -> dict:
    value = runtime_v62a._read_v62_methodology_v62a()
    frozen = value["preregistration"][
        "fresh_pre_hpo_alpha_zero_calibration_gate"
    ]
    if (
        frozen.get("maximum_primary_ci_halfwidth")
        != analysis.MAX_PRIMARY_CI_HALFWIDTH_V62B
        or frozen.get("maximum_actor_leave_one_out_shift")
        != analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B
        or frozen.get("generated_f1_primary_interval_must_contain_zero")
        is not True
    ):
        raise RuntimeError("v62b inherited V62 methodology gate changed")
    return value


def implementation_bindings_v62b() -> dict:
    paths = {
        "runtime_v62b": Path(__file__).resolve(),
        "preregistration_builder_v62b": (
            ROOT / "build_lora_es_pre_hpo_alpha_zero_preregistration_v62b.py"
        ),
        "analysis_v62b": Path(analysis.__file__).resolve(),
        "tests_v62b": (
            ROOT / "test_lora_es_pre_hpo_alpha_zero_calibration_v62b.py"
        ),
        "support_audit_v62b": Path(support_audit.__file__).resolve(),
        "runtime_v62a": Path(runtime_v62a.__file__).resolve(),
        "analysis_v62a": Path(runtime_v62a.analysis.__file__).resolve(),
        "support_audit_v62a": Path(runtime_v62a.support_audit.__file__).resolve(),
        "method_numeric_audit_v62": (
            ROOT / "analyze_lora_es_generated_f1_gate_v62.py"
        ),
        "method_preregistration_builder_v62": (
            ROOT / "build_lora_es_generated_f1_gate_preregistration_v62.py"
        ),
        "method_tests_v62": ROOT / "test_lora_es_generated_f1_gate_v62.py",
        "runtime_v61c": Path(runtime_v61c.__file__).resolve(),
        "input_builder_v61c": Path(runtime_v61c.inputs.__file__).resolve(),
        "runtime_v61a": Path(runtime_v61a.__file__).resolve(),
        "design_v52": Path(design_v52.__file__).resolve(),
        "runtime_v52": Path(runtime_v52.__file__).resolve(),
        "anchor_v4": ROOT / "train_eggroll_es_specialist_anchor_v4.py",
        "worker_v52": ROOT / "eggroll_es_worker_lora_v52.py",
        "worker_v51": ROOT / "eggroll_es_worker_lora_v51.py",
        "worker_v41a": ROOT / "eggroll_es_worker_lora_v41a.py",
    }
    return {
        "code_file_sha256": {
            key: runtime_v61a.file_sha256_v61a(path)
            for key, path in paths.items()
        },
        "support_audit": {
            "file_sha256": SUPPORT_AUDIT_FILE_SHA256,
            "content_sha256": SUPPORT_AUDIT_CONTENT_SHA256,
        },
        "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
        "v62_numeric_audit": dict(analysis.V62_NUMERIC_AUDIT_IDENTITIES),
        "v62_preregistration": dict(analysis.V62_PREREGISTRATION_IDENTITIES),
        "pinned_v434_identities_without_reopening": (
            runtime_v61a.implementation_bindings_v61a()[
                "pinned_v52_artifact_identities_without_reopening"
            ]
        ),
        "staged_dataset_file_sha256": (
            runtime_v61c.STAGED_DATASET_FILE_SHA256
        ),
        "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": (
            runtime_v61c.STAGED_PANEL_CONTENT_SHA256
        ),
        "train_semantics_model_gpu_or_protected_paths_opened": False,
    }


def _artifacts_v62b() -> dict:
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "evidence": str(EVIDENCE),
        "analysis": str(ANALYSIS),
        "report": str(REPORT),
        "failure": str(FAILURE),
        "gpu_log": str(GPU_LOG),
    }


def access_contract_v62b() -> dict:
    return {
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
        "warmup_raw_output_or_generation_metric_may_be_persisted": False,
        "numeric_hash_only_evidence_required": True,
    }


def scientific_scope_v62b() -> dict:
    return {
        "fresh_pre_hpo_alpha_zero_characterization_only": True,
        "reference_and_candidate_are_counterbalanced_aliases_for_identical_"
        "v434_state": True,
        "actual_intended_hpo_generation_runtime": True,
        "outcome_agnostic_fixed_schedule": True,
        "unscored_warmup_excluded_from_every_metric": True,
        "adapter_perturbation_or_update": False,
        "candidate_state_or_persistent_artifact": False,
        "hpo_population_selection_or_promotion": False,
        "calibration_success_itself_authorizes_hpo": False,
    }


def installed_support_binding_v62b() -> dict:
    support = _read_support_audit_v62b()
    return {
        "path": str(SUPPORT_AUDIT),
        "file_sha256": SUPPORT_AUDIT_FILE_SHA256,
        "content_sha256": SUPPORT_AUDIT_CONTENT_SHA256,
        "status": support["status"],
        "pre_hpo_alpha_zero_runtime_supported": True,
        "support_audit_authorizes_gpu_launch": False,
        "model_train_semantics_or_gpu_accessed": False,
    }


def fixed_recipe_v62b() -> dict:
    return {
        "base_model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
        "adapter_state": "V434",
        "staged_dataset": str(runtime_v61c.STAGED_DATASET),
        "staged_dataset_file_sha256": runtime_v61c.STAGED_DATASET_FILE_SHA256,
        "staged_panel": str(runtime_v61c.STAGED_PANEL),
        "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": (
            runtime_v61c.STAGED_PANEL_CONTENT_SHA256
        ),
        "rows": analysis.ROWS_V62B,
        "ranking_units": analysis.RANKING_UNITS_V62B,
        "exact_sentinel_units": analysis.EXACT_SENTINEL_UNITS_V62B,
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
        "worker_extension": WORKER_EXTENSION_V62B,
        "physical_gpu_ids": [0, 1, 2, 3],
        "actors": analysis.ACTORS_V62B,
        "tensor_parallel_size_per_actor": 1,
        "unscored_warmup_periods": analysis.WARMUP_PERIODS_V62B,
        "warmup_counterbalanced_blocks": 1,
        "warmup_label_plan": dict(analysis.WARMUP_LABEL_PLAN_V62B),
        "warmup_generation_completions_discarded": (
            analysis.WARMUP_GENERATION_COMPLETIONS_V62B
        ),
        "warmup_outputs_scored_or_persisted": False,
        "warmup_adaptive_retry_drop_or_reorder": False,
        "scored_counterbalanced_blocks": analysis.SCORED_BLOCKS_V62B,
        "periods_per_counterbalanced_block": (
            analysis.PERIODS_PER_BLOCK_V62B
        ),
        "scored_sequential_periods": analysis.SCORED_PERIODS_V62B,
        "total_sequential_periods": analysis.TOTAL_PERIODS_V62B,
        "counterbalanced_pairs_per_actor": analysis.PAIRS_PER_ACTOR_V62B,
        "replicas_per_conflict_unit": analysis.REPLICAS_PER_UNIT_V62B,
        "scored_label_plan": dict(analysis.LABEL_PLAN_V62B),
        "pair_periods": [list(pair) for pair in analysis.PAIR_PERIODS_V62B],
        "candidate_after_reference_pairs_per_actor": 6,
        "candidate_before_reference_pairs_per_actor": 6,
        "all_scored_periods_included": True,
        "scored_adaptive_retry_drop_reorder_or_early_stop": False,
        "generation_only": True,
        "request_type_per_period": "generation",
        "common_generation_seed": analysis.COMMON_GENERATION_SEED_V62B,
        "generation_params_without_seed": dict(
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V62B
        ),
        "teacher_forced_requests": 0,
        "teacher_forced_metric_computed": False,
        "runtime_determinism_controls": dict(
            analysis.RUNTIME_CONTROLS_V62B
        ),
        "submitted_requests_per_actor_call": analysis.ROWS_V62B,
        "active_sequence_limit_per_actor": analysis.ROWS_V62B,
        "v27c_tuned_table_runtime_identity_retained": True,
        "global_batch_invariance_claimed": False,
        "scored_generation_completions": (
            analysis.SCORED_GENERATION_COMPLETIONS_V62B
        ),
        "total_generation_completions": (
            analysis.TOTAL_GENERATION_COMPLETIONS_V62B
        ),
        "alpha": 0.0,
        "sigma_or_direction": None,
        "adapter_update_candidate_or_hpo_performed": False,
    }


def primary_estimator_v62b() -> dict:
    return {
        "metric": "paired_generated_f1_delta",
        "warmup_role": "unscored_discarded_and_excluded_from_every_metric",
        "within_unit_aggregation": "arithmetic_mean_all_48_replicas",
        "bootstrap_resampled_axis": "conflict_unit",
        "within_unit_replicas_preserved": analysis.REPLICAS_PER_UNIT_V62B,
        "bootstrap_replicates": analysis.BOOTSTRAP_REPLICATES_V62B,
        "bootstrap_seed": analysis.BOOTSTRAP_SEED_V62B,
        "one_sided_alpha": analysis.ONE_SIDED_ALPHA_V62B,
        "actor_influence": (
            "maximum absolute shift from full four-actor point to each "
            "leave-one-actor-out point across all 12 pairs per actor"
        ),
        "teacher_forced_logprob_role": "absent_not_computed",
    }


def required_gates_v62b() -> dict:
    return {
        "generated_f1_primary_interval_must_contain_zero": True,
        "maximum_primary_ci_halfwidth_inclusive": (
            analysis.MAX_PRIMARY_CI_HALFWIDTH_V62B
        ),
        "maximum_actor_leave_one_out_shift_inclusive": (
            analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B
        ),
        "all_comparisons_inclusive": True,
        "failure_action": "fail_closed_before_hpo_population_or_update",
        "success_does_not_authorize_hpo_population_or_update": True,
    }


def sentinel_policy_v62b() -> dict:
    return {
        "stable_unit_identity_sha256": list(
            analysis.STABLE_EXACT_UNIT_SHA256_V62B
        ),
        "stable_per_unit_statistic": (
            "strict_majority_exact_pass_count_at_least_25_of_48"
        ),
        "stable_aggregate_statistic": (
            "reference_and_candidate_exact_pass_totals_across_144_replicas"
        ),
        "actor_unstable_stress_unit_identity_sha256": (
            analysis.ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62B
        ),
        "actor_unstable_role": "diagnostic_stress_unit_only",
        "used_in_alpha_zero_gate": False,
        "any_single_flip_aborts": False,
        "any_per_unit_all_replicas_failure_aborts": False,
    }


def integrity_gates_v62b() -> dict:
    return {
        "exclusive_idle_four_gpu_preflight": True,
        "all_four_tp1_actor_and_pid_identities_exact": True,
        "all_actor_identities_verify_sync_fcfs_eager_bi_false_max68": True,
        "all_actor_identities_verify_exact_v27c_tuned_table": True,
        "exact_v434_state_before_and_after_every_warmup_period": True,
        "exact_v434_state_before_and_after_every_scored_period": True,
        "exactly_four_actor_batches_and_68_rows_per_call": True,
        "warmup_outputs_discarded_without_scoring_or_persistence": True,
        "warmup_never_adaptively_retried_dropped_or_reordered": True,
        "all_24_scored_periods_included_without_adaptation_or_early_stop": True,
        "all_four_gpus_attributed_positive": True,
        "strict_four_engine_cleanup_and_final_idle": True,
        "numeric_hash_only_evidence": True,
        "update_candidate_hpo_and_protected_access_zero": True,
    }


def load_preregistration_v62b(args) -> dict:
    path = Path(args.preregistration).resolve()
    if runtime_v61a.file_sha256_v61a(path) != args.preregistration_sha256:
        raise RuntimeError("v62b preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    expected_keys = {
        "schema",
        "status",
        "created_at_utc",
        "specific_alpha_zero_calibration_gpu_launch_authorized",
        "support_audit_alone_authorizes_gpu_launch",
        "builder_or_dry_run_performed_gpu_launch",
        "hpo_population_update_or_candidate_authorized",
        "ood_shadow_holdout_or_protected_access_authorized",
        "purpose",
        "v62_methodology_commit",
        "v62_numeric_audit_identities",
        "v62_preregistration_identities",
        "scientific_scope",
        "installed_runtime_support_audit",
        "access_contract",
        "fixed_calibration_recipe",
        "primary_numeric_estimator",
        "required_alpha_zero_gates",
        "exact_sentinel_diagnostics",
        "runtime",
        "required_python",
        "implementation_bindings",
        "artifacts",
        "required_integrity_gates",
        "raw_question_answer_or_generation_text_may_be_persisted",
        "warmup_raw_output_or_generation_metric_may_be_persisted",
        "protected_semantics_opened",
        "ood_shadow_holdout_or_terminal_opened",
        "content_sha256_before_self_field",
    }
    if (
        set(value) != expected_keys
        or value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or analysis.canonical_sha256_v62b(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "v62b-v434-pre-hpo-alpha-zero-generation-preregistration"
        or value.get("status")
        != "preregistered_before_train_semantics_model_or_gpu_access"
        or value.get("specific_alpha_zero_calibration_gpu_launch_authorized")
        is not True
        or value.get("support_audit_alone_authorizes_gpu_launch") is not False
        or value.get("builder_or_dry_run_performed_gpu_launch") is not False
        or value.get("hpo_population_update_or_candidate_authorized") is not False
        or value.get("ood_shadow_holdout_or_protected_access_authorized")
        is not False
        or value.get("v62_methodology_commit") != analysis.V62_METHOD_COMMIT
        or value.get("v62_numeric_audit_identities")
        != analysis.V62_NUMERIC_AUDIT_IDENTITIES
        or value.get("v62_preregistration_identities")
        != analysis.V62_PREREGISTRATION_IDENTITIES
        or value.get("purpose") != (
            "Run a fixed outcome-agnostic four-actor generation-only HPO "
            "evaluator calibration at alpha zero and unchanged V434 state, "
            "discard one counterbalanced warmup block, score exactly six "
            "counterbalanced blocks, and fail closed unless all three "
            "unchanged V62 gates pass."
        )
        or value.get("scientific_scope") != scientific_scope_v62b()
        or value.get("installed_runtime_support_audit")
        != installed_support_binding_v62b()
        or value.get("access_contract") != access_contract_v62b()
        or value.get("fixed_calibration_recipe") != fixed_recipe_v62b()
        or value.get("primary_numeric_estimator") != primary_estimator_v62b()
        or value.get("required_alpha_zero_gates") != required_gates_v62b()
        or value.get("exact_sentinel_diagnostics") != sentinel_policy_v62b()
        or value.get("runtime") != design_v52.RUNTIME_V52
        or value.get("required_python") != str(design_v52.REQUIRED_PYTHON_V52)
        or value.get("implementation_bindings")
        != implementation_bindings_v62b()
        or value.get("artifacts") != _artifacts_v62b()
        or value.get("required_integrity_gates") != integrity_gates_v62b()
        or value.get("raw_question_answer_or_generation_text_may_be_persisted")
        is not False
        or value.get(
            "warmup_raw_output_or_generation_metric_may_be_persisted"
        ) is not False
        or value.get("protected_semantics_opened") is not False
        or value.get("ood_shadow_holdout_or_terminal_opened") is not False
    ):
        raise RuntimeError("v62b preregistration contract changed")
    _read_support_audit_v62b()
    _read_v62_methodology_v62b()
    return value


def engine_kwargs_v62b(v40a, precision="bfloat16") -> dict:
    value = runtime_v62a.engine_kwargs_v62a(v40a, precision)
    if {
        key: value.get(key) for key in (
            "enforce_eager",
            "async_scheduling",
            "max_num_seqs",
            "scheduling_policy",
        )
    } != {
        "enforce_eager": True,
        "async_scheduling": False,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
    }:
        raise RuntimeError("v62b inherited engine runtime changed")
    return value


def _make_trainer_v62b(prereg: dict, prior):
    saved_module = (
        runtime_v62a.EXPERIMENT,
        runtime_v62a.RUN_DIR,
        runtime_v62a.WORKER_EXTENSION_V62A,
    )
    runtime_v62a.EXPERIMENT = EXPERIMENT
    runtime_v62a.RUN_DIR = RUN_DIR
    runtime_v62a.WORKER_EXTENSION_V62A = WORKER_EXTENSION_V62B
    try:
        return runtime_v62a._make_trainer_v62a(prereg, prior)
    finally:
        (
            runtime_v62a.EXPERIMENT,
            runtime_v62a.RUN_DIR,
            runtime_v62a.WORKER_EXTENSION_V62A,
        ) = saved_module


def _generation_params_v62b():
    return runtime_v62a._generation_params_v62a()


def _lora_request_v62b(prior):
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "v434_pre_hpo_alpha_zero_evaluator_v62b",
        1,
        str(design_v52.STAGED_V52),
        base_model_name=str(prior.v40a.MODEL),
    )


def _validate_actor_identities_v62b(actor_ids: object, tuned_sha256: str) -> dict:
    expected_keys = {
        "schema",
        "pid",
        "physical_gpu_id",
        "cuda_visible_devices",
        "cuda_current_device",
        "VLLM_BATCH_INVARIANT",
        "async_scheduling",
        "max_num_seqs",
        "scheduling_policy",
        "scheduler_class",
        "enforce_eager",
        "tuned_folder",
        "tuned_table_content_sha256",
        "submitted_request_batch_size",
        "generation_only",
        "global_batch_invariance_claimed",
    }
    if not isinstance(actor_ids, list) or len(actor_ids) != analysis.ACTORS_V62B:
        raise RuntimeError("v62b requires exactly four actor identities")
    mapping = {}
    pids = set()
    for item in actor_ids:
        if not isinstance(item, dict):
            raise RuntimeError("v62b actor identity changed")
        gpu = item.get("physical_gpu_id")
        pid = item.get("pid")
        if (
            set(item) != expected_keys
            or item.get("schema") != "pre-hpo-alpha-zero-actor-identity-v62a"
            or isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or gpu not in (0, 1, 2, 3)
            or item.get("cuda_visible_devices") != str(gpu)
            or item.get("cuda_current_device") != 0
            or {
                key: item.get(key) for key in analysis.RUNTIME_CONTROLS_V62B
            } != analysis.RUNTIME_CONTROLS_V62B
            or item.get("scheduler_class") != "Scheduler"
            or item.get("tuned_folder") != str(
                design_v52.RUNTIME_V52["tuned_folder"]
            )
            or item.get("tuned_table_content_sha256") != tuned_sha256
            or item.get("submitted_request_batch_size") != analysis.ROWS_V62B
            or item.get("generation_only") is not True
            or item.get("global_batch_invariance_claimed") is not False
            or gpu in mapping
            or pid in pids
        ):
            raise RuntimeError("v62b actor identity or runtime control changed")
        mapping[gpu] = pid
        pids.add(pid)
    if set(mapping) != {0, 1, 2, 3} or len(pids) != analysis.ACTORS_V62B:
        raise RuntimeError("v62b actor GPU/PID mapping incomplete")
    return mapping


def _validate_batches_v62b(batches: object, period_kind: str) -> list:
    if (
        not isinstance(batches, list)
        or len(batches) != analysis.ACTORS_V62B
        or any(not isinstance(batch, list) for batch in batches)
        or any(len(batch) != analysis.ROWS_V62B for batch in batches)
    ):
        raise RuntimeError(
            f"v62b {period_kind} exact four-actor/68-row coverage changed"
        )
    return batches


def _state_receipt_v62b(
    period_kind: str,
    period_index: int,
    before: dict,
    after: dict,
    installed: dict,
) -> dict:
    if (
        period_kind not in ("unscored_warmup", "scored")
        or isinstance(period_index, bool)
        or not isinstance(period_index, int)
        or period_index < 0
        or before != after
        or before != installed
    ):
        raise RuntimeError(f"v62b V434 state changed across {period_kind} period")
    value = {
        "period_kind": period_kind,
        "period_index": period_index,
        "before": before,
        "after": after,
        "identical_v434_state": True,
    }
    validator_value = dict(value)
    validator_value["period_index"] = 0
    analysis._validate_state_receipts_v62b(
        [validator_value],
        period_kind=period_kind,
        expected_count=1,
    )
    return value


def _validate_postrun_integrity_v62b(
    gpu: object,
    cleanup: object,
    idle: object,
    pid_map: dict,
) -> None:
    if (
        not isinstance(gpu, dict)
        or gpu.get("all_four_attributed_positive") is not True
        or set(gpu.get("by_gpu", {})) != {"0", "1", "2", "3"}
        or any(
            gpu["by_gpu"][str(index)].get("expected_pid") != pid_map[index]
            or gpu["by_gpu"][str(index)].get("positive_samples", 0) <= 0
            for index in range(4)
        )
        or not isinstance(cleanup, dict)
        or cleanup.get("schema")
        != "eggroll-es-placement-group-cleanup-v38a"
        or cleanup.get("engine_kill_count") != 4
        or cleanup.get("placement_group_remove_count") != 4
        or cleanup.get("all_four_gcs_states_removed") is not True
        or idle != {"all_four_compute_process_lists_empty": True}
    ):
        raise RuntimeError("v62b utilization, cleanup, or final-idle gate changed")


def build_evidence_v62b(
    rows,
    scored_periods,
    warmup_state_receipts: list[dict],
    scored_state_receipts: list[dict],
) -> dict:
    if (
        len(rows) != analysis.ROWS_V62B
        or len(scored_periods) != analysis.SCORED_PERIODS_V62B
        or len(warmup_state_receipts) != analysis.WARMUP_PERIODS_V62B
        or len(scored_state_receipts) != analysis.SCORED_PERIODS_V62B
        or any(len(period) != analysis.ACTORS_V62B for period in scored_periods)
        or any(
            len(batch) != analysis.ROWS_V62B
            for period in scored_periods for batch in period
        )
    ):
        raise RuntimeError("v62b generation evidence coverage changed")
    evidence_rows = []
    for request_index, row in enumerate(rows):
        evidence_rows.append({
            "request_index": request_index,
            "row_sha256": row["row_sha256"],
            "unit_identity_sha256": row["unit_identity_sha256"],
            "role": row["role"],
            "scored_periods": [{
                "period_index": period_index,
                "request_type": "generation",
                "actors": [{
                    "actor_rank": actor_rank,
                    "label": analysis.LABEL_PLAN_V62B[str(actor_rank)][
                        period_index
                    ],
                    "generation": scored_periods[period_index][actor_rank][
                        request_index
                    ],
                } for actor_rank in range(analysis.ACTORS_V62B)],
            } for period_index in range(analysis.SCORED_PERIODS_V62B)],
        })
    value = {
        "schema": "v62b-pre-hpo-alpha-zero-generation-only-evidence",
        "status": (
            "complete_fixed_warmup_and_scored_alpha_zero_characterization"
        ),
        "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
        "v62_numeric_audit_identities": dict(
            analysis.V62_NUMERIC_AUDIT_IDENTITIES
        ),
        "v62_preregistration_identities": dict(
            analysis.V62_PREREGISTRATION_IDENTITIES
        ),
        "staged_dataset_file_sha256": runtime_v61c.STAGED_DATASET_FILE_SHA256,
        "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": (
            runtime_v61c.STAGED_PANEL_CONTENT_SHA256
        ),
        "canonical_fp32_master_sha256": design_v52.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": design_v52.MASTER_RUNTIME_SHA256_V52,
        "row_count": analysis.ROWS_V62B,
        "ranking_units": analysis.RANKING_UNITS_V62B,
        "exact_sentinel_units": analysis.EXACT_SENTINEL_UNITS_V62B,
        "actor_count": analysis.ACTORS_V62B,
        "unscored_warmup_period_count": analysis.WARMUP_PERIODS_V62B,
        "scored_period_count": analysis.SCORED_PERIODS_V62B,
        "total_period_count": analysis.TOTAL_PERIODS_V62B,
        "scored_blocks": analysis.SCORED_BLOCKS_V62B,
        "periods_per_block": analysis.PERIODS_PER_BLOCK_V62B,
        "pairs_per_actor": analysis.PAIRS_PER_ACTOR_V62B,
        "replicas_per_unit": analysis.REPLICAS_PER_UNIT_V62B,
        "warmup_label_plan": dict(analysis.WARMUP_LABEL_PLAN_V62B),
        "scored_label_plan": dict(analysis.LABEL_PLAN_V62B),
        "pair_periods": [list(pair) for pair in analysis.PAIR_PERIODS_V62B],
        "common_generation_seed": analysis.COMMON_GENERATION_SEED_V62B,
        "generation_params_without_seed": dict(
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V62B
        ),
        "runtime_determinism_controls": dict(
            analysis.RUNTIME_CONTROLS_V62B
        ),
        "warmup_state_receipts": warmup_state_receipts,
        "numeric_warmup_state_receipts_sha256": (
            analysis.canonical_sha256_v62b(warmup_state_receipts)
        ),
        "scored_state_receipts": scored_state_receipts,
        "numeric_scored_state_receipts_sha256": (
            analysis.canonical_sha256_v62b(scored_state_receipts)
        ),
        "rows": evidence_rows,
        "numeric_actor_period_manifest_sha256": (
            analysis.canonical_sha256_v62b(evidence_rows)
        ),
        "generation_only": True,
        "warmup_generation_completions_discarded": (
            analysis.WARMUP_GENERATION_COMPLETIONS_V62B
        ),
        "scored_generation_completions": (
            analysis.SCORED_GENERATION_COMPLETIONS_V62B
        ),
        "total_generation_completions": (
            analysis.TOTAL_GENERATION_COMPLETIONS_V62B
        ),
        "warmup_raw_outputs_persisted": False,
        "warmup_generation_metrics_computed_or_persisted": False,
        "warmup_adaptive_retry_drop_or_reorder_performed": False,
        "scored_period_adaptive_retry_drop_reorder_or_early_stop_performed": (
            False
        ),
        "teacher_forced_requests": 0,
        "alpha": 0.0,
        "adapter_update_candidate_or_hpo_performed": False,
        "holdback_ood_shadow_or_protected_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v62b(value)
    )
    analysis.validate_evidence_v62b(value)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v62b(args)
    if args.dry_run:
        if args.execute:
            raise RuntimeError("v62b dry-run and execute are mutually exclusive")
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "support_audit_file_sha256": SUPPORT_AUDIT_FILE_SHA256,
            "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
            "staged_train_rows_opened": 0,
            "unscored_warmup_periods": analysis.WARMUP_PERIODS_V62B,
            "scored_periods": analysis.SCORED_PERIODS_V62B,
            "scored_replicas_per_conflict_unit": (
                analysis.REPLICAS_PER_UNIT_V62B
            ),
            "warmup_outputs_scored_or_persisted": False,
            "generation_only": True,
            "teacher_forced_requests": 0,
            "model_or_gpu_loaded": False,
            "filesystem_writes": False,
            "hpo_update_candidate_or_protected_access": False,
        }, sort_keys=True))
        return 0
    if not args.execute:
        raise RuntimeError("v62b live path requires --execute")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise RuntimeError("v62b requires batch-invariant mode absent or false")
    runtime_v52.require_live_interpreter_v52()
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v62b requires fresh artifact paths")

    import run_lora_es_generation_boundary_v48b as v48b
    prior = v48b.v43i
    v40a = prior.v40a
    preflight = v40a.gpu_preflight()
    attempt = runtime_v61a.self_hashed_v61a({
        "schema": "v62b-pre-hpo-alpha-zero-generation-attempt",
        "status": "launching_specific_calibration_only",
        "phase": (
            "after_gpu_inventory_preflight_before_staged_train_semantics_"
            "model_load_or_gpu_compute"
        ),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "support_audit_file_sha256": SUPPORT_AUDIT_FILE_SHA256,
        "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
        "runtime_determinism_controls": dict(
            analysis.RUNTIME_CONTROLS_V62B
        ),
        "fixed_unscored_warmup_periods": analysis.WARMUP_PERIODS_V62B,
        "fixed_scored_periods": analysis.SCORED_PERIODS_V62B,
        "fixed_scored_replicas_per_conflict_unit": (
            analysis.REPLICAS_PER_UNIT_V62B
        ),
        "preflight": preflight,
        "gpu_inventory_preflight_performed": True,
        "model_loaded_or_gpu_compute_started": False,
        "hpo_update_candidate_or_protected_access": False,
    })
    runtime_v61a.atomic_json_v61a(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    try:
        adapter_ids = runtime_v61c._verify_adapter_artifacts_v61c()
        rows, panel = runtime_v61c.load_staged_inputs_v61c()
        if len(rows) != analysis.ROWS_V62B:
            raise RuntimeError("v62b exact staged row coverage changed")
        prompts = [v40a.base.specialist_template(row["question"]) for row in rows]
        if len(prompts) != analysis.ROWS_V62B:
            raise RuntimeError("v62b exact prompt coverage changed")
        v40a.base.set_seed(prior.GLOBAL_SEED)
        trainer, saved = _make_trainer_v62b(prereg, prior)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v62a.remote() for engine in trainer.engines
        ])
        pid_map = _validate_actor_identities_v62b(
            actor_ids,
            prereg["runtime"]["tuned_table_content_sha256"],
        )
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures),
            daemon=True,
        )
        monitor.start()
        request = _lora_request_v62b(prior)
        phase.value = "activate_v434_lora_slot"
        if v40a._rpc_all(trainer, "add_lora", (request,)) != [True] * 4:
            raise RuntimeError("v62b four-actor LoRA activation failed")
        phase.value = "install_exact_v434_master"
        installations = v40a._rpc_all(trainer, "install_adapter_state_v41a", (
            str(design_v52.SOURCE_WEIGHTS_V52),
            str(design_v52.SOURCE_CONFIG_V52),
            design_v52.SOURCE_WEIGHTS_SHA256_V52,
            design_v52.SOURCE_CONFIG_SHA256_V52,
        ))
        installed = runtime_v61c._assert_v434_certificates_v61c(
            v40a._rpc_all(trainer, "adapter_state_certificate_v52")
        )

        warmup_state_receipts = []
        for period_index in range(analysis.WARMUP_PERIODS_V62B):
            before = runtime_v61c._assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            phase.value = f"unscored_warmup_{period_index}_generation_all_actors"
            warmup_batches = _validate_batches_v62b(
                trainer._resolve([
                    engine.generate.remote(
                        prompts,
                        _generation_params_v62b(),
                        use_tqdm=False,
                        lora_request=request,
                    ) for engine in trainer.engines
                ]),
                "unscored_warmup",
            )
            after = runtime_v61c._assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            warmup_state_receipts.append(_state_receipt_v62b(
                "unscored_warmup", period_index, before, after, installed,
            ))
            del warmup_batches

        scored_periods = []
        scored_state_receipts = []
        for period_index in range(analysis.SCORED_PERIODS_V62B):
            before = runtime_v61c._assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            phase.value = f"scored_period_{period_index}_generation_all_actors"
            batches = _validate_batches_v62b(
                trainer._resolve([
                    engine.generate.remote(
                        prompts,
                        _generation_params_v62b(),
                        use_tqdm=False,
                        lora_request=request,
                    ) for engine in trainer.engines
                ]),
                "scored",
            )
            scored_periods.append([
                runtime_v61c.score_generation_batch_v61c(rows, batch, prior.fused)
                for batch in batches
            ])
            after = runtime_v61c._assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            scored_state_receipts.append(_state_receipt_v62b(
                "scored", period_index, before, after, installed,
            ))

        evidence = build_evidence_v62b(
            rows,
            scored_periods,
            warmup_state_receipts,
            scored_state_receipts,
        )
        null_analysis = analysis.build_analysis_v62b(evidence)
        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v62b GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        _validate_postrun_integrity_v62b(gpu, cleanup, idle, pid_map)

        runtime_v61a.atomic_json_v61a(EVIDENCE, evidence)
        runtime_v61a.atomic_json_v61a(ANALYSIS, null_analysis)
        gate = null_analysis["required_pre_hpo_gate"]
        report = runtime_v61a.self_hashed_v61a({
            "schema": "v62b-pre-hpo-alpha-zero-generation-report",
            "status": (
                "complete_gate_passed_hpo_still_unauthorized"
                if gate["passed"] else "complete_gate_failed_closed"
            ),
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": (
                args.preregistration_content_sha256
            ),
            "support_audit_file_sha256": SUPPORT_AUDIT_FILE_SHA256,
            "support_audit_content_sha256": SUPPORT_AUDIT_CONTENT_SHA256,
            "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
            "adapter_artifact_identities": adapter_ids,
            "panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
            "panel_content_sha256": runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
            "panel_document_block_audit": panel["document_block_audit"],
            "master_state_receipt": installed,
            "installations": installations,
            "warmup_state_receipts_sha256": evidence[
                "numeric_warmup_state_receipts_sha256"
            ],
            "scored_state_receipts_sha256": evidence[
                "numeric_scored_state_receipts_sha256"
            ],
            "actor_identities": actor_ids,
            "warmup": {
                "periods": analysis.WARMUP_PERIODS_V62B,
                "generation_completions_discarded": (
                    analysis.WARMUP_GENERATION_COMPLETIONS_V62B
                ),
                "raw_outputs_persisted": False,
                "generation_metrics_computed_or_persisted": False,
                "adaptive_retry_drop_or_reorder_performed": False,
            },
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
                "content_sha256": evidence[
                    "content_sha256_before_self_field"
                ],
                "rows": analysis.ROWS_V62B,
                "actors": analysis.ACTORS_V62B,
                "scored_periods": analysis.SCORED_PERIODS_V62B,
                "pairs_per_actor": analysis.PAIRS_PER_ACTOR_V62B,
                "replicas_per_conflict_unit": analysis.REPLICAS_PER_UNIT_V62B,
                "scored_generation_completions": (
                    analysis.SCORED_GENERATION_COMPLETIONS_V62B
                ),
                "total_generation_completions": (
                    analysis.TOTAL_GENERATION_COMPLETIONS_V62B
                ),
                "teacher_forced_requests": 0,
                "all_scored_periods_included_without_early_stop": True,
            },
            "analysis": {
                "path": str(ANALYSIS),
                "file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
                "content_sha256": null_analysis[
                    "content_sha256_before_self_field"
                ],
                "required_pre_hpo_gate": gate,
                "exact_sentinel_diagnostics": null_analysis[
                    "exact_sentinel_diagnostics"
                ],
            },
            "gpu_activity": gpu,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "gpu_log_file_sha256": runtime_v61a.file_sha256_v61a(GPU_LOG),
            "alpha": 0.0,
            "generation_only": True,
            "adapter_update_candidate_or_hpo_performed": False,
            "holdback_ood_shadow_or_protected_opened": False,
            "raw_question_answer_or_generation_text_persisted": False,
            "hpo_population_launch_authorized": False,
        })
        runtime_v61a.atomic_json_v61a(REPORT, report)
        print(json.dumps({
            "report_file_sha256": runtime_v61a.file_sha256_v61a(REPORT),
            "report_content_sha256": report[
                "content_sha256_before_self_field"
            ],
            "evidence_file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
            "evidence_content_sha256": evidence[
                "content_sha256_before_self_field"
            ],
            "analysis_file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
            "analysis_content_sha256": null_analysis[
                "content_sha256_before_self_field"
            ],
            "required_pre_hpo_gate_passed": gate["passed"],
            "unscored_warmup_periods": analysis.WARMUP_PERIODS_V62B,
            "scored_periods": analysis.SCORED_PERIODS_V62B,
            "scored_replicas_per_conflict_unit": (
                analysis.REPLICAS_PER_UNIT_V62B
            ),
            "generation_only": True,
            "teacher_forced_requests": 0,
            "all_four_gpus_attributed_positive": True,
            "hpo_population_launch_authorized": False,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        if not FAILURE.exists():
            failure = runtime_v61a._sanitize_failure_v61a(error)
            failure.update({
                "schema": "v62b-pre-hpo-alpha-zero-generation-failure",
                "runtime_determinism_controls": dict(
                    analysis.RUNTIME_CONTROLS_V62B
                ),
                "fixed_unscored_warmup_periods": analysis.WARMUP_PERIODS_V62B,
                "fixed_scored_periods": analysis.SCORED_PERIODS_V62B,
                "adaptive_retry_drop_reorder_or_early_stop_performed": False,
                "adapter_update_candidate_or_hpo_performed": False,
                "holdback_ood_shadow_or_protected_opened": False,
            })
            runtime_v61a.atomic_json_v61a(
                FAILURE,
                runtime_v61a.self_hashed_v61a(failure),
            )
        raise
    finally:
        if trainer is not None:
            try:
                v40a.base.close_trainer(trainer)
            except Exception:
                pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            prior.v40a.EXPERIMENT, prior.v40a.RUN_DIR, prior.v40a.WORKER_EXTENSION = (
                saved
            )


if __name__ == "__main__":
    raise SystemExit(main())
