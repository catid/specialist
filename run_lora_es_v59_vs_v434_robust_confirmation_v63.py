#!/usr/bin/env python3
"""Sealed four-GPU V63 V59-versus-V434 train-only confirmation runtime.

The builder and ``--dry-run`` paths are CPU-only and never open the staged
dataset, base model, or any GPU.  The explicit live path runs a fixed V62B
schedule with two immutable standard vLLM LoRA requests.  It performs no ES
step, optimizer update, HPO, promotion, or protected evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open

import lora_es_nested_population_v52 as design_v52
import lora_es_v59_vs_v434_robust_confirmation_v63 as analysis
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_nested_population_v52 as runtime_v52
import run_lora_es_paired_null_calibration_v61c as runtime_v61c
import run_lora_es_pre_hpo_alpha_zero_calibration_v62a as runtime_v62a
import run_lora_es_pre_hpo_alpha_zero_calibration_v62b as runtime_v62b


ROOT = Path(__file__).resolve().parent
BASE_MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
EXPERIMENT = "v63_v59_vs_v434_train_only_robust_confirmation"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
EVIDENCE = (RUN_DIR / "confirmation_evidence_v63.json").resolve()
ANALYSIS = (RUN_DIR / "confirmation_analysis_v63.json").resolve()
REPORT = (RUN_DIR / "confirmation_report_v63.json").resolve()
FAILURE = (RUN_DIR / "failure_v63.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v63.jsonl").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v59_vs_v434_train_only_robust_confirmation_v63.json"
).resolve()
V62B_FINALIZED = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v62b_v434_pre_hpo_alpha_zero_generation_calibration/"
    "alpha_zero_finalized_v62b.json"
).resolve()
REFERENCE_ADAPTER = design_v52.STAGED_V52
CANDIDATE_ADAPTER = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v59_lora_es_fragile_priority/selected_candidate_v59"
).resolve()
REFERENCE_WEIGHTS = (REFERENCE_ADAPTER / "adapter_model.safetensors").resolve()
REFERENCE_CONFIG = (REFERENCE_ADAPTER / "adapter_config.json").resolve()
CANDIDATE_WEIGHTS = (CANDIDATE_ADAPTER / "adapter_model.safetensors").resolve()
CANDIDATE_CONFIG = (CANDIDATE_ADAPTER / "adapter_config.json").resolve()
WORKER_EXTENSION_V63 = runtime_v62a.WORKER_EXTENSION_V62A
_ORIGINAL_ENGINE_KWARGS_V62A = runtime_v62a.engine_kwargs_v62a


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--execute", action="store_true")
    return value


def file_sha256_v63(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_v62b_eligibility_v63() -> dict:
    """Verify the sealed calibration eligibility without inheriting authority."""
    if file_sha256_v63(V62B_FINALIZED) != (
        analysis.V62B_FINALIZED_FILE_SHA256_V63
    ):
        raise RuntimeError("v63 V62B finalized artifact file changed")
    value = json.loads(V62B_FINALIZED.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    eligibility = value.get("calibration_eligibility_observation", {})
    frozen = value.get("frozen_non_authorization", {})
    gate = value.get("observed_numeric_outcome_without_authorization", {}).get(
        "required_pre_hpo_gate", {}
    )
    if (
        value.get("content_sha256_before_self_field")
        != analysis.V62B_FINALIZED_CONTENT_SHA256_V63
        or analysis.canonical_sha256_v63(compact)
        != analysis.V62B_FINALIZED_CONTENT_SHA256_V63
        or value.get("schema")
        != "v62b-pre-hpo-alpha-zero-independent-finalizer"
        or value.get("status")
        != "complete_numeric_only_eligibility_observed_hpo_unauthorized"
        or eligibility.get("eligible_for_later_separately_preregistered_hpo_work")
        is not True
        or eligibility.get("eligibility_is_not_launch_or_update_authority")
        is not True
        or eligibility.get("failed_gate_count") != 0
        or gate.get("passed") is not True
        or any(gate.get("checks", {}).get(name) is not True for name in (
            "null_primary_ci_contains_zero",
            "primary_ci_halfwidth_at_most_frozen_limit_inclusive",
            "actor_leave_one_out_shift_at_most_frozen_limit_inclusive",
        ))
        or gate.get("maximum_primary_ci_halfwidth_inclusive")
        != analysis.FROZEN_NULL_WIDTH_V63
        or frozen.get("finalizer_accepts_and_records_either_gate_outcome")
        is not True
        or frozen.get("hpo_population_launch_or_update_authorized") is not False
        or frozen.get("gpu_or_model_launch_authorized") is not False
        or frozen.get(
            "holdback_ood_shadow_terminal_or_protected_access_authorized"
        ) is not False
        or value.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v63 V62B finalized eligibility changed")
    return {
        "commit": analysis.V62B_FINALIZER_COMMIT_V63,
        "file_sha256": analysis.V62B_FINALIZED_FILE_SHA256_V63,
        "content_sha256": analysis.V62B_FINALIZED_CONTENT_SHA256_V63,
        "eligible_for_later_separately_preregistered_work": True,
        "launch_or_update_authority": False,
    }


def verify_adapter_artifacts_v63() -> dict[str, dict]:
    observed = {
        "reference": {
            "label": "reference",
            "adapter": "V434",
            "lora_int_id": analysis.REFERENCE_LORA_ID_V63,
            "weights_file_sha256": file_sha256_v63(REFERENCE_WEIGHTS),
            "config_file_sha256": file_sha256_v63(REFERENCE_CONFIG),
            "canonical_fp32_state_sha256": (
                analysis.REFERENCE_CANONICAL_STATE_SHA256_V63
            ),
            "runtime_bf16_values_sha256": (
                analysis.REFERENCE_RUNTIME_VALUES_SHA256_V63
            ),
        },
        "candidate": {
            "label": "candidate",
            "adapter": "V59",
            "lora_int_id": analysis.CANDIDATE_LORA_ID_V63,
            "weights_file_sha256": file_sha256_v63(CANDIDATE_WEIGHTS),
            "config_file_sha256": file_sha256_v63(CANDIDATE_CONFIG),
            "canonical_fp32_state_sha256": (
                analysis.CANDIDATE_CANONICAL_STATE_SHA256_V63
            ),
        },
    }
    if observed != analysis.expected_adapter_identities_v63():
        raise RuntimeError("v63 reference or candidate adapter file changed")
    with safe_open(CANDIDATE_WEIGHTS, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = list(handle.keys())
    if (
        metadata.get("schema") != "uncommitted-canonical-peft-fp32-v52"
        or metadata.get("candidate_sha256")
        != analysis.CANDIDATE_CANONICAL_STATE_SHA256_V63
        or metadata.get("manifest_sha256")
        != "e477cac0ed5fbed2cf106d0a5640648a9a533e683e94c06d4571e57cef5c85d7"
        or len(keys) != 70
    ):
        raise RuntimeError("v63 V59 canonical candidate metadata changed")
    return observed


def installed_two_adapter_support_v63() -> dict:
    """Static installed-API proof; it intentionally performs no GPU probe."""
    from vllm.lora.request import LoRARequest

    source = Path(inspect.getsourcefile(LoRARequest) or "").resolve()
    signature = str(inspect.signature(LoRARequest))
    reference = LoRARequest(
        "v434_reference_v63",
        analysis.REFERENCE_LORA_ID_V63,
        str(REFERENCE_ADAPTER),
        base_model_name=str(BASE_MODEL),
    )
    candidate = LoRARequest(
        "v59_candidate_v63",
        analysis.CANDIDATE_LORA_ID_V63,
        str(CANDIDATE_ADAPTER),
        base_model_name=str(BASE_MODEL),
    )
    request_projection = [{
        "lora_name": request.lora_name,
        "lora_int_id": request.lora_int_id,
        "lora_path": request.lora_path,
        "base_model_name": request.base_model_name,
    } for request in (reference, candidate)]
    if (
        "lora_int_id" not in signature
        or request_projection[0]["lora_int_id"] != 1
        or request_projection[1]["lora_int_id"] != 2
        or request_projection[0]["lora_path"] == request_projection[1]["lora_path"]
    ):
        raise RuntimeError("v63 installed LoRARequest API cannot bind IDs 1/2")
    return {
        "schema": "v63-installed-vllm-two-standard-lora-request-static-support",
        "status": "static_api_supported_live_switch_not_preclaimed",
        "lora_request_source_path": str(source),
        "lora_request_source_file_sha256": file_sha256_v63(source),
        "lora_request_signature": signature,
        "request_projection": request_projection,
        "engine_lora_capacity": {"max_loras": 1, "max_cpu_loras": 2},
        "sequential_on_demand_requests_required": True,
        "live_gpu_switch_probe_performed": False,
        "model_dataset_or_gpu_opened": False,
        "support_observation_alone_authorizes_launch": False,
    }


def implementation_bindings_v63() -> dict:
    paths = {
        "runtime_v63": Path(__file__).resolve(),
        "analysis_v63": Path(analysis.__file__).resolve(),
        "preregistration_builder_v63": (
            ROOT / "build_lora_es_v59_vs_v434_robust_confirmation_"
            "preregistration_v63.py"
        ),
        "finalizer_v63": (
            ROOT / "finalize_lora_es_v59_vs_v434_robust_confirmation_v63.py"
        ),
        "tests_v63": (
            ROOT / "test_lora_es_v59_vs_v434_robust_confirmation_v63.py"
        ),
        "analysis_v62b": Path(runtime_v62b.analysis.__file__).resolve(),
        "runtime_v62b": Path(runtime_v62b.__file__).resolve(),
        "runtime_v62a": Path(runtime_v62a.__file__).resolve(),
        "runtime_v61c": Path(runtime_v61c.__file__).resolve(),
        "runtime_v61a": Path(runtime_v61a.__file__).resolve(),
        "design_v52": Path(design_v52.__file__).resolve(),
        "runtime_v52": Path(runtime_v52.__file__).resolve(),
    }
    return {
        "code_file_sha256": {
            key: file_sha256_v63(path) for key, path in paths.items()
        },
        "v62b_finalized": verify_v62b_eligibility_v63(),
        "adapter_artifacts": verify_adapter_artifacts_v63(),
        "installed_two_adapter_support": installed_two_adapter_support_v63(),
        "staged_dataset_file_sha256": (
            runtime_v61c.STAGED_DATASET_FILE_SHA256
        ),
        "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": (
            runtime_v61c.STAGED_PANEL_CONTENT_SHA256
        ),
        "train_dataset_model_gpu_or_protected_paths_opened": False,
    }


def _artifacts_v63() -> dict:
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "evidence": str(EVIDENCE),
        "analysis": str(ANALYSIS),
        "report": str(REPORT),
        "failure": str(FAILURE),
        "gpu_log": str(GPU_LOG),
    }


def access_contract_v63() -> dict:
    return {
        "only_live_semantic_paths_may_open": [
            str(runtime_v61c.STAGED_DATASET),
            str(runtime_v61c.STAGED_PANEL),
        ],
        "builder_or_dry_run_reads_staged_rows_or_panel": False,
        "builder_or_dry_run_loads_base_model_or_gpu": False,
        "full_train_membership_holdback_ood_shadow_terminal_or_protected_may_open": False,
        "live_runtime_may_load_only_pinned_qwen36_v434_v59_and_staged_68_rows": True,
        "raw_question_answer_prompt_or_generation_text_may_be_persisted": False,
        "warmup_raw_output_or_metric_may_be_persisted": False,
        "numeric_hash_only_evidence_required": True,
        "optimizer_master_checkpoint_or_update_state_may_open": False,
    }


def scientific_scope_v63() -> dict:
    return {
        "train_only_v59_vs_v434_confirmation": True,
        "outcome_agnostic_fixed_schedule": True,
        "v62b_finalized_eligibility_verified_before_live_path": True,
        "unscored_warmup_excluded_from_every_metric": True,
        "two_distinct_immutable_standard_lora_requests": True,
        "reference_lora_id": analysis.REFERENCE_LORA_ID_V63,
        "candidate_lora_id": analysis.CANDIDATE_LORA_ID_V63,
        "adapter_perturbation_or_update": False,
        "hpo_population_selection_or_promotion": False,
        "median_consensus_or_best_of_selection": False,
        "confirmation_success_itself_authorizes_update_or_promotion": False,
    }


def fixed_recipe_v63() -> dict:
    return {
        "base_model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
        "reference": analysis.expected_adapter_identities_v63()["reference"],
        "candidate": analysis.expected_adapter_identities_v63()["candidate"],
        "reference_request_path": str(REFERENCE_ADAPTER),
        "candidate_request_path": str(CANDIDATE_ADAPTER),
        "staged_dataset": str(runtime_v61c.STAGED_DATASET),
        "staged_dataset_file_sha256": runtime_v61c.STAGED_DATASET_FILE_SHA256,
        "staged_panel": str(runtime_v61c.STAGED_PANEL),
        "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
        "rows": analysis.ROWS_V63,
        "ranking_units": analysis.RANKING_UNITS_V63,
        "exact_sentinel_units": analysis.EXACT_SENTINEL_UNITS_V63,
        "same_call_ranking_plus_sentinel_rows": True,
        "holdback_units": 0,
        "physical_gpu_ids": [0, 1, 2, 3],
        "actors": analysis.ACTORS_V63,
        "tensor_parallel_size_per_actor": 1,
        "unscored_warmup_periods": analysis.WARMUP_PERIODS_V63,
        "warmup_label_plan": dict(analysis.WARMUP_LABEL_PLAN_V63),
        "warmup_generation_completions_discarded": (
            analysis.WARMUP_GENERATION_COMPLETIONS_V63
        ),
        "scored_counterbalanced_blocks": analysis.SCORED_BLOCKS_V63,
        "periods_per_counterbalanced_block": analysis.PERIODS_PER_BLOCK_V63,
        "scored_sequential_periods": analysis.SCORED_PERIODS_V63,
        "total_sequential_periods": analysis.TOTAL_PERIODS_V63,
        "counterbalanced_pairs_per_actor": analysis.PAIRS_PER_ACTOR_V63,
        "replicas_per_conflict_unit": analysis.REPLICAS_PER_UNIT_V63,
        "scored_label_plan": dict(analysis.LABEL_PLAN_V63),
        "pair_periods": [list(pair) for pair in analysis.PAIR_PERIODS_V63],
        "candidate_after_reference_pairs_per_actor": 6,
        "candidate_before_reference_pairs_per_actor": 6,
        "all_scored_periods_included": True,
        "adaptive_retry_drop_reorder_or_early_stop": False,
        "generation_only": True,
        "common_generation_seed": analysis.COMMON_GENERATION_SEED_V63,
        "generation_params_without_seed": dict(
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V63
        ),
        "teacher_forced_requests": 0,
        "runtime_determinism_controls": dict(analysis.RUNTIME_CONTROLS_V63),
        "submitted_requests_per_actor_call": analysis.ROWS_V63,
        "active_sequence_limit_per_actor": analysis.ROWS_V63,
        "global_batch_invariance_claimed": False,
        "scored_generation_completions": (
            analysis.SCORED_GENERATION_COMPLETIONS_V63
        ),
        "total_generation_completions": analysis.TOTAL_GENERATION_COMPLETIONS_V63,
        "adapter_identity_verified_before_and_after_every_period": True,
        "sigma_or_alpha": None,
        "adapter_update_hpo_or_promotion_performed": False,
    }


def primary_estimator_v63() -> dict:
    return {
        "metric": "paired_generated_f1_candidate_minus_reference",
        "warmup_role": "unscored_discarded_and_excluded_from_every_metric",
        "within_unit_aggregation": "arithmetic_mean_all_48_replicas",
        "bootstrap_resampled_axis": "conflict_unit",
        "within_unit_replicas_preserved": analysis.REPLICAS_PER_UNIT_V63,
        "bootstrap_replicates": analysis.BOOTSTRAP_REPLICATES_V63,
        "bootstrap_seed": analysis.BOOTSTRAP_SEED_V63,
        "one_sided_alpha": analysis.ONE_SIDED_ALPHA_V63,
        "actor_influence": (
            "maximum absolute shift from the full four-actor point to each "
            "leave-one-actor-out point across all 12 pairs per actor"
        ),
        "robust_fitness": "one_sided_lcb_minus_maximum_actor_loo_shift",
        "median_consensus_or_best_of_selection": False,
        "teacher_forced_logprob_role": "absent_not_computed",
    }


def required_gates_v63() -> dict:
    return {
        "robust_fitness_lcb_minus_max_actor_loo_shift_strictly_positive": True,
        "point_improvement_minimum_inclusive": (
            analysis.MINIMUM_POINT_IMPROVEMENT_V63
        ),
        "point_improvement_definition": (
            "two_times_frozen_null_width_0.000773822590292528"
        ),
        "stable_exact_aggregate_candidate_at_least_reference_inclusive": True,
        "stable_exact_each_unit_candidate_at_least_reference_inclusive": True,
        "stable_nonzero_aggregate_candidate_at_least_reference_inclusive": True,
        "actor_unstable_stress_unit_is_diagnostic_only": True,
        "success_does_not_authorize_update_hpo_promotion_or_protected_access": True,
        "failure_action": "fail_closed_without_update_hpo_or_promotion",
    }


def integrity_gates_v63() -> dict:
    return {
        "v62b_finalized_file_and_content_hash_exact_before_live_path": True,
        "v59_and_v434_file_and_canonical_identities_exact_before_live_path": True,
        "exclusive_idle_four_gpu_preflight": True,
        "all_four_tp1_actor_and_pid_identities_exact": True,
        "all_actors_sync_fcfs_eager_bi_false_max68": True,
        "engine_max_loras_one_and_max_cpu_loras_two": True,
        "standard_lora_requests_have_distinct_exact_ids_one_and_two": True,
        "both_adapter_file_identities_before_and_after_every_period": True,
        "exactly_four_actor_batches_and_68_rows_per_call": True,
        "four_warmup_periods_discarded_without_scoring_or_persistence": True,
        "all_24_scored_periods_included_without_retry_drop_or_early_stop": True,
        "all_four_gpus_attributed_positive": True,
        "strict_four_engine_cleanup_and_final_idle": True,
        "numeric_hash_only_evidence": True,
        "update_hpo_master_checkpoint_and_protected_access_zero": True,
    }


def load_preregistration_v63(args) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256_v63(path) != args.preregistration_sha256:
        raise RuntimeError("v63 preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    expected_keys = {
        "schema", "status", "created_at_utc",
        "specific_v63_confirmation_gpu_launch_authorized",
        "eligibility_or_static_support_alone_authorizes_launch",
        "builder_or_dry_run_performed_gpu_launch",
        "update_hpo_candidate_promotion_or_protected_access_authorized",
        "purpose", "scientific_scope", "v62b_finalized_eligibility",
        "installed_two_adapter_static_support", "access_contract",
        "fixed_confirmation_recipe", "primary_numeric_estimator",
        "required_confirmation_gates", "runtime", "required_python",
        "implementation_bindings", "artifacts", "required_integrity_gates",
        "raw_question_answer_prompt_or_generation_text_may_be_persisted",
        "warmup_raw_output_or_generation_metric_may_be_persisted",
        "protected_semantics_opened", "ood_shadow_holdout_or_terminal_opened",
        "content_sha256_before_self_field",
    }
    if (
        set(value) != expected_keys
        or value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or analysis.canonical_sha256_v63(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "v63-v59-vs-v434-train-only-robust-confirmation-preregistration"
        or value.get("status")
        != "preregistered_before_train_semantics_model_or_gpu_access"
        or value.get("specific_v63_confirmation_gpu_launch_authorized") is not True
        or value.get("eligibility_or_static_support_alone_authorizes_launch")
        is not False
        or value.get("builder_or_dry_run_performed_gpu_launch") is not False
        or value.get(
            "update_hpo_candidate_promotion_or_protected_access_authorized"
        ) is not False
        or value.get("purpose") != (
            "Run one fixed train-only, four-actor, generation-only robust "
            "confirmation of immutable V59 against immutable V434 after the "
            "sealed V62B evaluator calibration; no result authorizes an "
            "update, HPO, promotion, or protected evaluation."
        )
        or value.get("scientific_scope") != scientific_scope_v63()
        or value.get("v62b_finalized_eligibility")
        != verify_v62b_eligibility_v63()
        or value.get("installed_two_adapter_static_support")
        != installed_two_adapter_support_v63()
        or value.get("access_contract") != access_contract_v63()
        or value.get("fixed_confirmation_recipe") != fixed_recipe_v63()
        or value.get("primary_numeric_estimator") != primary_estimator_v63()
        or value.get("required_confirmation_gates") != required_gates_v63()
        or value.get("runtime") != design_v52.RUNTIME_V52
        or value.get("required_python") != str(design_v52.REQUIRED_PYTHON_V52)
        or value.get("implementation_bindings") != implementation_bindings_v63()
        or value.get("artifacts") != _artifacts_v63()
        or value.get("required_integrity_gates") != integrity_gates_v63()
        or value.get(
            "raw_question_answer_prompt_or_generation_text_may_be_persisted"
        ) is not False
        or value.get(
            "warmup_raw_output_or_generation_metric_may_be_persisted"
        ) is not False
        or value.get("protected_semantics_opened") is not False
        or value.get("ood_shadow_holdout_or_terminal_opened") is not False
    ):
        raise RuntimeError("v63 preregistration contract changed")
    return value


def engine_kwargs_v63(v40a, precision="bfloat16") -> dict:
    value = _ORIGINAL_ENGINE_KWARGS_V62A(v40a, precision)
    value["max_loras"] = 1
    value["max_cpu_loras"] = 2
    expected = {
        "enforce_eager": True,
        "async_scheduling": False,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
        "max_loras": 1,
        "max_cpu_loras": 2,
    }
    if {key: value.get(key) for key in expected} != expected:
        raise RuntimeError("v63 engine runtime controls changed")
    return value


def _make_trainer_v63(prereg: dict, prior):
    saved_module = (
        runtime_v62a.EXPERIMENT,
        runtime_v62a.RUN_DIR,
        runtime_v62a.WORKER_EXTENSION_V62A,
        runtime_v62a.engine_kwargs_v62a,
    )
    runtime_v62a.EXPERIMENT = EXPERIMENT
    runtime_v62a.RUN_DIR = RUN_DIR
    runtime_v62a.WORKER_EXTENSION_V62A = WORKER_EXTENSION_V63
    runtime_v62a.engine_kwargs_v62a = engine_kwargs_v63
    try:
        return runtime_v62a._make_trainer_v62a(prereg, prior)
    finally:
        (
            runtime_v62a.EXPERIMENT,
            runtime_v62a.RUN_DIR,
            runtime_v62a.WORKER_EXTENSION_V62A,
            runtime_v62a.engine_kwargs_v62a,
        ) = saved_module


def _generation_params_v63():
    return runtime_v62a._generation_params_v62a()


def _lora_requests_v63(prior) -> dict:
    from vllm.lora.request import LoRARequest

    values = {
        "reference": LoRARequest(
            "v434_reference_v63",
            analysis.REFERENCE_LORA_ID_V63,
            str(REFERENCE_ADAPTER),
            base_model_name=str(prior.v40a.MODEL),
        ),
        "candidate": LoRARequest(
            "v59_candidate_v63",
            analysis.CANDIDATE_LORA_ID_V63,
            str(CANDIDATE_ADAPTER),
            base_model_name=str(prior.v40a.MODEL),
        ),
    }
    expected = analysis.expected_adapter_identities_v63()
    for label, request in values.items():
        if (
            request.lora_int_id != expected[label]["lora_int_id"]
            or request.lora_path
            != str(REFERENCE_ADAPTER if label == "reference" else CANDIDATE_ADAPTER)
        ):
            raise RuntimeError("v63 LoRA request identity changed")
    return values


def _assignments_v63(period_kind: str, period_index: int) -> list[dict]:
    plan = (
        analysis.WARMUP_LABEL_PLAN_V63
        if period_kind == "unscored_warmup"
        else analysis.LABEL_PLAN_V63
    )
    identities = analysis.expected_adapter_identities_v63()
    result = []
    for actor_rank in range(analysis.ACTORS_V63):
        label = plan[str(actor_rank)][period_index]
        item = identities[label]
        result.append({
            "actor_rank": actor_rank,
            "label": label,
            "adapter": item["adapter"],
            "lora_int_id": item["lora_int_id"],
            "weights_file_sha256": item["weights_file_sha256"],
            "config_file_sha256": item["config_file_sha256"],
            "canonical_fp32_state_sha256": item[
                "canonical_fp32_state_sha256"
            ],
        })
    return result


def _state_receipt_v63(
    period_kind: str,
    period_index: int,
    before: dict,
    after: dict,
) -> dict:
    value = {
        "period_kind": period_kind,
        "period_index": period_index,
        "before": before,
        "after": after,
        "actor_request_assignments": _assignments_v63(
            period_kind, period_index
        ),
        "both_adapter_files_exact_and_unchanged": before == after,
    }
    validator = dict(value)
    validator["period_index"] = 0
    validator["actor_request_assignments"] = _assignments_v63(period_kind, 0)
    analysis._validate_state_receipts_v63(
        [validator], period_kind=period_kind, expected_count=1
    )
    return value


def _validate_batches_v63(batches: object, period_kind: str) -> list:
    if (
        not isinstance(batches, list)
        or len(batches) != analysis.ACTORS_V63
        or any(not isinstance(batch, list) for batch in batches)
        or any(len(batch) != analysis.ROWS_V63 for batch in batches)
    ):
        raise RuntimeError(
            f"v63 {period_kind} exact four-actor/68-row coverage changed"
        )
    return batches


def _validate_actor_identities_v63(actor_ids: object, tuned_sha256: str) -> dict:
    return runtime_v62b._validate_actor_identities_v62b(
        actor_ids, tuned_sha256
    )


def _validate_postrun_integrity_v63(gpu, cleanup, idle, pid_map) -> None:
    runtime_v62b._validate_postrun_integrity_v62b(
        gpu, cleanup, idle, pid_map
    )


def build_evidence_v63(
    rows,
    scored_periods,
    warmup_state_receipts: list[dict],
    scored_state_receipts: list[dict],
) -> dict:
    if (
        len(rows) != analysis.ROWS_V63
        or len(scored_periods) != analysis.SCORED_PERIODS_V63
        or len(warmup_state_receipts) != analysis.WARMUP_PERIODS_V63
        or len(scored_state_receipts) != analysis.SCORED_PERIODS_V63
        or any(len(period) != analysis.ACTORS_V63 for period in scored_periods)
        or any(
            len(batch) != analysis.ROWS_V63
            for period in scored_periods for batch in period
        )
    ):
        raise RuntimeError("v63 generation evidence coverage changed")
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
                    "label": analysis.LABEL_PLAN_V63[str(actor_rank)][
                        period_index
                    ],
                    "generation": scored_periods[period_index][actor_rank][
                        request_index
                    ],
                } for actor_rank in range(analysis.ACTORS_V63)],
            } for period_index in range(analysis.SCORED_PERIODS_V63)],
        })
    value = {
        "schema": "v63-v59-vs-v434-train-only-generation-evidence",
        "status": "complete_fixed_confirmation_schedule",
        "v62b_finalized_artifact": verify_v62b_eligibility_v63(),
        "adapter_identities": verify_adapter_artifacts_v63(),
        "staged_dataset_file_sha256": runtime_v61c.STAGED_DATASET_FILE_SHA256,
        "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
        "row_count": analysis.ROWS_V63,
        "ranking_units": analysis.RANKING_UNITS_V63,
        "exact_sentinel_units": analysis.EXACT_SENTINEL_UNITS_V63,
        "actor_count": analysis.ACTORS_V63,
        "unscored_warmup_period_count": analysis.WARMUP_PERIODS_V63,
        "scored_period_count": analysis.SCORED_PERIODS_V63,
        "total_period_count": analysis.TOTAL_PERIODS_V63,
        "scored_blocks": analysis.SCORED_BLOCKS_V63,
        "periods_per_block": analysis.PERIODS_PER_BLOCK_V63,
        "pairs_per_actor": analysis.PAIRS_PER_ACTOR_V63,
        "replicas_per_unit": analysis.REPLICAS_PER_UNIT_V63,
        "warmup_label_plan": dict(analysis.WARMUP_LABEL_PLAN_V63),
        "scored_label_plan": dict(analysis.LABEL_PLAN_V63),
        "pair_periods": [list(pair) for pair in analysis.PAIR_PERIODS_V63],
        "common_generation_seed": analysis.COMMON_GENERATION_SEED_V63,
        "generation_params_without_seed": dict(
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V63
        ),
        "runtime_determinism_controls": dict(analysis.RUNTIME_CONTROLS_V63),
        "warmup_state_receipts": warmup_state_receipts,
        "numeric_warmup_state_receipts_sha256": (
            analysis.canonical_sha256_v63(warmup_state_receipts)
        ),
        "scored_state_receipts": scored_state_receipts,
        "numeric_scored_state_receipts_sha256": (
            analysis.canonical_sha256_v63(scored_state_receipts)
        ),
        "rows": evidence_rows,
        "numeric_actor_period_manifest_sha256": (
            analysis.canonical_sha256_v63(evidence_rows)
        ),
        "generation_only": True,
        "warmup_generation_completions_discarded": (
            analysis.WARMUP_GENERATION_COMPLETIONS_V63
        ),
        "scored_generation_completions": (
            analysis.SCORED_GENERATION_COMPLETIONS_V63
        ),
        "total_generation_completions": analysis.TOTAL_GENERATION_COMPLETIONS_V63,
        "warmup_raw_outputs_persisted": False,
        "warmup_generation_metrics_computed_or_persisted": False,
        "warmup_adaptive_retry_drop_or_reorder_performed": False,
        "scored_period_adaptive_retry_drop_reorder_or_early_stop_performed": False,
        "teacher_forced_requests": 0,
        "adapter_update_hpo_or_promotion_performed": False,
        "holdback_ood_shadow_or_protected_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v63(value)
    )
    analysis.validate_evidence_v63(value)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v63(args)
    if args.dry_run:
        if args.execute:
            raise RuntimeError("v63 dry-run and execute are mutually exclusive")
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "v62b_finalized_file_sha256": (
                analysis.V62B_FINALIZED_FILE_SHA256_V63
            ),
            "v62b_finalized_content_sha256": (
                analysis.V62B_FINALIZED_CONTENT_SHA256_V63
            ),
            "adapter_artifact_identities_verified": True,
            "staged_train_rows_or_panel_opened": 0,
            "unscored_warmup_periods": analysis.WARMUP_PERIODS_V63,
            "scored_periods": analysis.SCORED_PERIODS_V63,
            "scored_replicas_per_conflict_unit": analysis.REPLICAS_PER_UNIT_V63,
            "warmup_outputs_scored_or_persisted": False,
            "generation_only": True,
            "teacher_forced_requests": 0,
            "base_model_or_gpu_loaded": False,
            "filesystem_writes": False,
            "update_hpo_master_checkpoint_or_protected_access": False,
        }, sort_keys=True))
        return 0
    if not args.execute:
        raise RuntimeError("v63 live path requires --execute")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise RuntimeError("v63 requires batch-invariant mode absent or false")
    runtime_v52.require_live_interpreter_v52()
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v63 requires fresh artifact paths")

    import run_lora_es_generation_boundary_v48b as v48b
    prior = v48b.v43i
    v40a = prior.v40a
    preflight = v40a.gpu_preflight()
    attempt = runtime_v61a.self_hashed_v61a({
        "schema": "v63-v59-vs-v434-confirmation-attempt",
        "status": "launching_specific_train_only_confirmation",
        "phase": (
            "after_eligibility_adapter_and_gpu_preflight_before_staged_train_"
            "semantics_model_load_or_gpu_compute"
        ),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "v62b_finalized": verify_v62b_eligibility_v63(),
        "adapter_artifacts": verify_adapter_artifacts_v63(),
        "runtime_determinism_controls": dict(analysis.RUNTIME_CONTROLS_V63),
        "fixed_unscored_warmup_periods": analysis.WARMUP_PERIODS_V63,
        "fixed_scored_periods": analysis.SCORED_PERIODS_V63,
        "fixed_scored_replicas_per_conflict_unit": (
            analysis.REPLICAS_PER_UNIT_V63
        ),
        "preflight": preflight,
        "gpu_inventory_preflight_performed": True,
        "model_loaded_or_gpu_compute_started": False,
        "update_hpo_master_checkpoint_or_protected_access": False,
    })
    runtime_v61a.atomic_json_v61a(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    try:
        rows, panel = runtime_v61c.load_staged_inputs_v61c()
        if len(rows) != analysis.ROWS_V63:
            raise RuntimeError("v63 exact staged row coverage changed")
        prompts = [v40a.base.specialist_template(row["question"]) for row in rows]
        if len(prompts) != analysis.ROWS_V63:
            raise RuntimeError("v63 exact prompt coverage changed")
        v40a.base.set_seed(prior.GLOBAL_SEED)
        trainer, saved = _make_trainer_v63(prereg, prior)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v62a.remote() for engine in trainer.engines
        ])
        pid_map = _validate_actor_identities_v63(
            actor_ids, prereg["runtime"]["tuned_table_content_sha256"]
        )
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures),
            daemon=True,
        )
        monitor.start()
        requests = _lora_requests_v63(prior)

        warmup_state_receipts = []
        for period_index in range(analysis.WARMUP_PERIODS_V63):
            before = verify_adapter_artifacts_v63()
            phase.value = f"unscored_warmup_{period_index}_generation_all_actors"
            assignments = _assignments_v63("unscored_warmup", period_index)
            warmup_batches = _validate_batches_v63(
                trainer._resolve([
                    engine.generate.remote(
                        prompts,
                        _generation_params_v63(),
                        use_tqdm=False,
                        lora_request=requests[assignments[actor]["label"]],
                    )
                    for actor, engine in enumerate(trainer.engines)
                ]),
                "unscored_warmup",
            )
            after = verify_adapter_artifacts_v63()
            warmup_state_receipts.append(_state_receipt_v63(
                "unscored_warmup", period_index, before, after
            ))
            del warmup_batches

        scored_periods = []
        scored_state_receipts = []
        for period_index in range(analysis.SCORED_PERIODS_V63):
            before = verify_adapter_artifacts_v63()
            phase.value = f"scored_period_{period_index}_generation_all_actors"
            assignments = _assignments_v63("scored", period_index)
            batches = _validate_batches_v63(
                trainer._resolve([
                    engine.generate.remote(
                        prompts,
                        _generation_params_v63(),
                        use_tqdm=False,
                        lora_request=requests[assignments[actor]["label"]],
                    )
                    for actor, engine in enumerate(trainer.engines)
                ]),
                "scored",
            )
            scored_periods.append([
                runtime_v61c.score_generation_batch_v61c(rows, batch, prior.fused)
                for batch in batches
            ])
            after = verify_adapter_artifacts_v63()
            scored_state_receipts.append(_state_receipt_v63(
                "scored", period_index, before, after
            ))

        evidence = build_evidence_v63(
            rows, scored_periods, warmup_state_receipts, scored_state_receipts
        )
        confirmation = analysis.build_analysis_v63(evidence)
        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v63 GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        _validate_postrun_integrity_v63(gpu, cleanup, idle, pid_map)

        runtime_v61a.atomic_json_v61a(EVIDENCE, evidence)
        runtime_v61a.atomic_json_v61a(ANALYSIS, confirmation)
        gate = confirmation["required_confirmation_gate"]
        report = runtime_v61a.self_hashed_v61a({
            "schema": "v63-v59-vs-v434-train-only-confirmation-report",
            "status": (
                "complete_gate_passed_without_promotion_authority"
                if gate["passed"] else "complete_gate_failed_closed"
            ),
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "v62b_finalized": verify_v62b_eligibility_v63(),
            "adapter_artifact_identities": verify_adapter_artifacts_v63(),
            "two_standard_lora_requests": {
                "reference_id": 1,
                "candidate_id": 2,
                "max_loras": 1,
                "max_cpu_loras": 2,
                "sequential_period_switching": True,
            },
            "panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
            "panel_content_sha256": runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
            "panel_document_block_audit": panel["document_block_audit"],
            "actor_identities": actor_ids,
            "warmup_state_receipts_sha256": evidence[
                "numeric_warmup_state_receipts_sha256"
            ],
            "scored_state_receipts_sha256": evidence[
                "numeric_scored_state_receipts_sha256"
            ],
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": file_sha256_v63(EVIDENCE),
                "content_sha256": evidence["content_sha256_before_self_field"],
                "rows": analysis.ROWS_V63,
                "actors": analysis.ACTORS_V63,
                "scored_periods": analysis.SCORED_PERIODS_V63,
                "pairs_per_actor": analysis.PAIRS_PER_ACTOR_V63,
                "replicas_per_conflict_unit": analysis.REPLICAS_PER_UNIT_V63,
                "all_scored_periods_included_without_early_stop": True,
            },
            "analysis": {
                "path": str(ANALYSIS),
                "file_sha256": file_sha256_v63(ANALYSIS),
                "content_sha256": confirmation[
                    "content_sha256_before_self_field"
                ],
                "required_confirmation_gate": gate,
                "exact_sentinel_diagnostics": confirmation[
                    "exact_sentinel_diagnostics"
                ],
            },
            "gpu_activity": gpu,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "gpu_log_file_sha256": file_sha256_v63(GPU_LOG),
            "generation_only": True,
            "teacher_forced_requests": 0,
            "adaptive_retry_drop_reorder_or_early_stop_performed": False,
            "median_consensus_or_best_of_selection_performed": False,
            "adapter_update_hpo_master_checkpoint_or_promotion_performed": False,
            "holdback_ood_shadow_or_protected_opened": False,
            "raw_question_answer_or_generation_text_persisted": False,
            "result_authorizes_update_hpo_promotion_or_protected_access": False,
        })
        runtime_v61a.atomic_json_v61a(REPORT, report)
        print(json.dumps({
            "report_file_sha256": file_sha256_v63(REPORT),
            "report_content_sha256": report["content_sha256_before_self_field"],
            "evidence_file_sha256": file_sha256_v63(EVIDENCE),
            "evidence_content_sha256": evidence[
                "content_sha256_before_self_field"
            ],
            "analysis_file_sha256": file_sha256_v63(ANALYSIS),
            "analysis_content_sha256": confirmation[
                "content_sha256_before_self_field"
            ],
            "required_confirmation_gate_passed": gate["passed"],
            "unscored_warmup_periods": analysis.WARMUP_PERIODS_V63,
            "scored_periods": analysis.SCORED_PERIODS_V63,
            "scored_replicas_per_conflict_unit": analysis.REPLICAS_PER_UNIT_V63,
            "all_four_gpus_attributed_positive": True,
            "update_hpo_promotion_or_protected_access_authorized": False,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        if not FAILURE.exists():
            failure = runtime_v61a._sanitize_failure_v61a(error)
            failure.update({
                "schema": "v63-v59-vs-v434-confirmation-failure",
                "runtime_determinism_controls": dict(
                    analysis.RUNTIME_CONTROLS_V63
                ),
                "fixed_unscored_warmup_periods": analysis.WARMUP_PERIODS_V63,
                "fixed_scored_periods": analysis.SCORED_PERIODS_V63,
                "adaptive_retry_drop_reorder_or_early_stop_performed": False,
                "adapter_update_hpo_master_checkpoint_or_promotion_performed": False,
                "holdback_ood_shadow_or_protected_opened": False,
            })
            runtime_v61a.atomic_json_v61a(
                FAILURE, runtime_v61a.self_hashed_v61a(failure)
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
