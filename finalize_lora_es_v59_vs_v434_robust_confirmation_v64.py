#!/usr/bin/env python3
"""Outcome-agnostic numeric finalizer for a future sealed V64 live run.

The finalizer remains byte-identical to the implementation bound before the
live run. Postrun file/content hashes are required as command-line facts; the
immutable finalizer then verifies every file, numeric rebuild, report link,
GPU log, cleanup receipt, and non-authorization invariant. Tests exercise both
scientific outcomes and tampering without launching a model or GPU.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path

import lora_es_v59_vs_v434_robust_confirmation_v64 as analysis
import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = (runtime.RUN_DIR / "confirmation_finalized_v64.json").resolve()

FORBIDDEN_TEXT_KEYS_V64 = {
    "answer", "completion", "completion_text", "generated_text",
    "output_text", "outputs", "prediction", "prompt", "prompt_token_ids",
    "question", "raw_text", "response", "text", "token_ids",
}


@dataclass(frozen=True)
class SelfHashedSourceV64:
    path: Path
    file_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class FinalizerSourcesV64:
    preregistration: SelfHashedSourceV64
    attempt: SelfHashedSourceV64
    panel: SelfHashedSourceV64
    evidence: SelfHashedSourceV64
    analysis: SelfHashedSourceV64
    report: SelfHashedSourceV64
    gpu_log_path: Path
    gpu_log_file_sha256: str


def _require_sha256_v64(value: str | None, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v64 production outcome hash is not sealed: {name}")
    return value


def production_sources_v64(
    preregistration_file_sha256: str,
    preregistration_content_sha256: str,
    attempt_file_sha256: str,
    attempt_content_sha256: str,
    evidence_file_sha256: str,
    evidence_content_sha256: str,
    analysis_file_sha256: str,
    analysis_content_sha256: str,
    report_file_sha256: str,
    report_content_sha256: str,
    gpu_log_file_sha256: str,
) -> FinalizerSourcesV64:
    return FinalizerSourcesV64(
        preregistration=SelfHashedSourceV64(
            runtime.PREREGISTRATION,
            _require_sha256_v64(
                preregistration_file_sha256, "preregistration file"
            ),
            _require_sha256_v64(
                preregistration_content_sha256, "preregistration content"
            ),
        ),
        attempt=SelfHashedSourceV64(
            runtime.ATTEMPT,
            _require_sha256_v64(attempt_file_sha256, "attempt file"),
            _require_sha256_v64(attempt_content_sha256, "attempt content"),
        ),
        panel=SelfHashedSourceV64(
            runtime.runtime_v61c.STAGED_PANEL,
            runtime.runtime_v61c.STAGED_PANEL_FILE_SHA256,
            runtime.runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
        ),
        evidence=SelfHashedSourceV64(
            runtime.EVIDENCE,
            _require_sha256_v64(evidence_file_sha256, "evidence file"),
            _require_sha256_v64(evidence_content_sha256, "evidence content"),
        ),
        analysis=SelfHashedSourceV64(
            runtime.ANALYSIS,
            _require_sha256_v64(analysis_file_sha256, "analysis file"),
            _require_sha256_v64(analysis_content_sha256, "analysis content"),
        ),
        report=SelfHashedSourceV64(
            runtime.REPORT,
            _require_sha256_v64(report_file_sha256, "report file"),
            _require_sha256_v64(report_content_sha256, "report content"),
        ),
        gpu_log_path=runtime.GPU_LOG,
        gpu_log_file_sha256=_require_sha256_v64(
            gpu_log_file_sha256, "GPU log file"
        ),
    )


def _read_self_hashed_v64(source: SelfHashedSourceV64) -> dict:
    if runtime.file_sha256_v64(source.path) != source.file_sha256:
        raise RuntimeError(f"v64 finalizer input file changed: {source.path}")
    value = json.loads(source.path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != source.content_sha256
        or analysis.canonical_sha256_v64(compact) != source.content_sha256
    ):
        raise RuntimeError(
            f"v64 finalizer input content changed: {source.path}"
        )
    return value


def _verify_no_text_keys_v64(name: str, value: object) -> dict:
    found = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in FORBIDDEN_TEXT_KEYS_V64:
                    found.append(f"{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, name)
    if found:
        raise RuntimeError(f"v64 forbidden text-bearing key: {found[0]}")
    return {"source": name, "forbidden_text_key_count": 0}


def _verify_preregistration_v64(prereg: dict) -> dict:
    expected_keys = {
        "schema", "status", "created_at_utc",
        "specific_v64_confirmation_gpu_launch_authorized",
        "eligibility_or_static_support_alone_authorizes_launch",
        "builder_or_dry_run_performed_cuda_compute_launch",
        "update_hpo_candidate_promotion_or_protected_access_authorized",
        "purpose", "scientific_scope", "v62b_finalized_eligibility",
        "installed_two_adapter_static_support",
        "base_model_artifact_expectation", "access_contract",
        "fixed_confirmation_recipe", "primary_numeric_estimator",
        "required_confirmation_gates", "runtime", "required_python",
        "implementation_bindings", "artifacts", "required_integrity_gates",
        "raw_question_answer_prompt_or_generation_text_may_be_persisted",
        "warmup_raw_output_or_generation_metric_may_be_persisted",
        "protected_semantics_opened", "ood_shadow_holdout_or_terminal_opened",
        "content_sha256_before_self_field",
    }
    expected_bindings = runtime.implementation_bindings_v64()
    if (
        set(prereg) != expected_keys
        or prereg.get("schema")
        != "v64-v59-vs-v434-train-only-robust-confirmation-preregistration"
        or prereg.get("status")
        != "preregistered_before_train_semantics_model_or_cuda_compute"
        or prereg.get("specific_v64_confirmation_gpu_launch_authorized")
        is not True
        or prereg.get("eligibility_or_static_support_alone_authorizes_launch")
        is not False
        or prereg.get("builder_or_dry_run_performed_cuda_compute_launch")
        is not False
        or prereg.get(
            "update_hpo_candidate_promotion_or_protected_access_authorized"
        ) is not False
        or prereg.get("scientific_scope") != runtime.scientific_scope_v64()
        or prereg.get("v62b_finalized_eligibility")
        != runtime.verify_v62b_eligibility_v64()
        or prereg.get("installed_two_adapter_static_support")
        != runtime.installed_two_adapter_support_v64()
        or prereg.get("base_model_artifact_expectation")
        != runtime.base_model_artifact_expectation_v64()
        or prereg.get("access_contract") != runtime.access_contract_v64()
        or prereg.get("required_confirmation_gates")
        != runtime.required_gates_v64()
        or prereg.get("fixed_confirmation_recipe")
        != runtime.fixed_recipe_v64()
        or prereg.get("primary_numeric_estimator")
        != runtime.primary_estimator_v64()
        or prereg.get("runtime") != runtime.design_v52.RUNTIME_V52
        or prereg.get("required_python")
        != str(runtime.design_v52.REQUIRED_PYTHON_V52)
        or prereg.get("implementation_bindings") != expected_bindings
        or prereg.get("artifacts") != runtime._artifacts_v64()
        or prereg.get("required_integrity_gates")
        != runtime.integrity_gates_v64()
        or prereg.get(
            "raw_question_answer_prompt_or_generation_text_may_be_persisted"
        ) is not False
        or prereg.get(
            "warmup_raw_output_or_generation_metric_may_be_persisted"
        ) is not False
        or prereg.get("protected_semantics_opened") is not False
        or prereg.get("ood_shadow_holdout_or_terminal_opened") is not False
    ):
        raise RuntimeError("v64 finalizer preregistration changed")
    return {
        "fixed_recipe_exact": True,
        "fixed_gates_exact": True,
        "v62b_finalized_eligibility_exact": True,
        "implementation_bindings_exact": True,
        "immutable_prelaunch_finalizer_binding_exact": True,
    }


def _verify_attempt_v64(
    attempt: dict,
    prereg: dict,
    sources: FinalizerSourcesV64,
) -> dict:
    expected_keys = {
        "schema", "status", "phase", "started_at_utc",
        "preregistration_file_sha256", "preregistration_content_sha256",
        "v62b_finalized", "adapter_artifacts", "base_model_artifact_receipt",
        "runtime_determinism_controls", "fixed_unscored_warmup_periods",
        "fixed_scored_periods", "fixed_scored_replicas_per_conflict_unit",
        "preflight", "gpu_inventory_preflight_performed",
        "model_loaded_or_gpu_compute_started",
        "update_hpo_master_checkpoint_or_protected_access",
        "content_sha256_before_self_field",
    }
    preflight = attempt.get("preflight", {})
    memory = preflight.get("memory_used_mib", {})
    base_model_receipt = runtime.validate_base_model_artifact_receipt_v64(
        attempt.get("base_model_artifact_receipt"),
        prereg["base_model_artifact_expectation"],
    )
    if (
        set(attempt) != expected_keys
        or attempt.get("schema") != "v64-v59-vs-v434-confirmation-attempt"
        or attempt.get("status")
        != "launching_specific_train_only_confirmation"
        or attempt.get("phase") != (
            "after_base_model_byte_audit_eligibility_adapter_and_gpu_preflight_"
            "before_staged_train_semantics_model_load_or_gpu_compute"
        )
        or attempt.get("preregistration_file_sha256")
        != sources.preregistration.file_sha256
        or attempt.get("preregistration_content_sha256")
        != sources.preregistration.content_sha256
        or attempt.get("v62b_finalized")
        != runtime.verify_v62b_eligibility_v64()
        or attempt.get("adapter_artifacts")
        != analysis.expected_adapter_identities_v64()
        or attempt.get("base_model_artifact_receipt") != base_model_receipt
        or attempt.get("runtime_determinism_controls")
        != analysis.RUNTIME_CONTROLS_V64
        or attempt.get("fixed_unscored_warmup_periods")
        != analysis.WARMUP_PERIODS_V64
        or attempt.get("fixed_scored_periods") != analysis.SCORED_PERIODS_V64
        or attempt.get("fixed_scored_replicas_per_conflict_unit")
        != analysis.REPLICAS_PER_UNIT_V64
        or set(preflight) != {"compute_process_query_empty", "memory_used_mib"}
        or preflight.get("compute_process_query_empty") is not True
        or set(memory) != {"0", "1", "2", "3"}
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            or value < 0 or value > 2048
            for value in memory.values()
        )
        or attempt.get("gpu_inventory_preflight_performed") is not True
        or attempt.get("model_loaded_or_gpu_compute_started") is not False
        or attempt.get("update_hpo_master_checkpoint_or_protected_access")
        is not False
    ):
        raise RuntimeError("v64 exclusive-idle launch attempt changed")
    return {
        "attempt_file_and_content_hash_exact": True,
        "all_base_model_files_byte_hash_exact_before_attempt": True,
        "exclusive_idle_four_gpu_preflight_exact": True,
        "model_or_gpu_compute_not_started_before_attempt_receipt": True,
        "update_hpo_or_protected_access_before_attempt_receipt": False,
    }


def _actor_pid_map_v64(report: dict, prereg: dict) -> dict[int, int]:
    try:
        return runtime._validate_actor_identities_v64(
            report.get("actor_identities"),
            prereg["runtime"]["tuned_table_content_sha256"],
        )
    except Exception as error:
        raise RuntimeError("v64 actor runtime identity changed") from error


def _verify_cleanup_v64(cleanup: object, idle: object) -> dict:
    expected_cleanup_keys = {
        "schema", "driver_scoped_non_detached_by_construction",
        "engine_kill_count", "placement_group_remove_count", "before",
        "after", "all_four_gcs_states_removed",
    }
    expected_row_keys = {
        "placement_group_id", "strategy", "state", "bundles",
        "bundles_to_node_id",
    }
    if not isinstance(cleanup, dict):
        raise RuntimeError("v64 cleanup changed")
    before = cleanup.get("before", [])
    after = cleanup.get("after", [])
    before_ids = [item.get("placement_group_id") for item in before]
    after_ids = [item.get("placement_group_id") for item in after]
    if (
        set(cleanup) != expected_cleanup_keys
        or cleanup.get("schema") != "eggroll-es-placement-group-cleanup-v38a"
        or cleanup.get("driver_scoped_non_detached_by_construction") is not True
        or cleanup.get("engine_kill_count") != 4
        or cleanup.get("placement_group_remove_count") != 4
        or cleanup.get("all_four_gcs_states_removed") is not True
        or not isinstance(before, list) or len(before) != 4
        or not isinstance(after, list) or len(after) != 4
        or any(not isinstance(item, dict) or set(item) != expected_row_keys
               for item in [*before, *after])
        or len(set(before_ids)) != 4
        or before_ids != after_ids
        or any(item.get("strategy") != "PACK" for item in [*before, *after])
        or any(item.get("state") != "CREATED" for item in before)
        or any(item.get("state") != "REMOVED" for item in after)
        or idle != {"all_four_compute_process_lists_empty": True}
    ):
        raise RuntimeError("v64 cleanup changed")
    return {
        "engines_killed": 4,
        "placement_groups_removed": 4,
        "all_four_gcs_states_removed": True,
        "all_four_final_gpu_compute_process_lists_empty": True,
    }


def _gpu_summary_v64(
    path: Path,
    expected_sha256: str,
    report: dict,
    pid_map: dict[int, int],
) -> dict:
    if runtime.file_sha256_v64(path) != expected_sha256:
        raise RuntimeError("v64 GPU log changed")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    _verify_no_text_keys_v64("gpu_log", rows)
    expected_keys = {
        "compute_pids", "expected_pid", "foreign_compute_pids", "gpu",
        "memory_used_mib", "phase", "sampled_at_utc", "utilization_percent",
    }
    allowed_phases = {"setup"} | {
        f"unscored_warmup_{index}_generation_all_actors"
        for index in range(analysis.WARMUP_PERIODS_V64)
    } | {
        f"scored_period_{index}_generation_all_actors"
        for index in range(analysis.SCORED_PERIODS_V64)
    }
    required_phases = allowed_phases - {"setup"}
    if (
        not rows
        or len(rows) % analysis.ACTORS_V64 != 0
        or any(not isinstance(row, dict) or set(row) != expected_keys for row in rows)
        or any(row.get("phase") not in allowed_phases for row in rows)
        or any(
            not isinstance(row.get("sampled_at_utc"), str)
            or not row["sampled_at_utc"]
            or isinstance(row.get("utilization_percent"), bool)
            or not isinstance(row.get("utilization_percent"), int)
            or not 0 <= row["utilization_percent"] <= 100
            or isinstance(row.get("memory_used_mib"), bool)
            or not isinstance(row.get("memory_used_mib"), int)
            or row["memory_used_mib"] < 0
            for row in rows
        )
    ):
        raise RuntimeError("v64 GPU log row count or schema changed")
    for start in range(0, len(rows), analysis.ACTORS_V64):
        group = rows[start:start + analysis.ACTORS_V64]
        if (
            [row["gpu"] for row in group] != [0, 1, 2, 3]
            or len({row["sampled_at_utc"] for row in group}) != 1
        ):
            raise RuntimeError("v64 GPU log sampling cycle changed")
    by_gpu = {}
    samples_per_gpu = len(rows) // analysis.ACTORS_V64
    for gpu in range(analysis.ACTORS_V64):
        expected_pid = pid_map[gpu]
        selected = [row for row in rows if row.get("gpu") == gpu]
        positive_phases = {
            row["phase"] for row in selected
            if row["utilization_percent"] > 0
        }
        if (
            len(selected) != samples_per_gpu
            or {row["phase"] for row in selected}.issuperset(required_phases)
            is not True
            or any(row.get("expected_pid") != expected_pid for row in selected)
            or any(row.get("foreign_compute_pids") != [] for row in selected)
            or any(row.get("compute_pids") != [expected_pid] for row in selected)
            or not any(row["utilization_percent"] > 0 for row in selected)
            or positive_phases.issuperset(required_phases) is not True
        ):
            raise RuntimeError("v64 foreign, inactive, or mismatched GPU process")
        by_gpu[str(gpu)] = {
            "expected_pid": expected_pid,
            "samples": len(selected),
            "resident_samples": len(selected),
            "positive_samples": sum(
                row["utilization_percent"] > 0 for row in selected
            ),
            "mean_resident_utilization_percent": math.fsum(
                row["utilization_percent"] for row in selected
            ) / len(selected),
            "peak_utilization_percent": max(
                row["utilization_percent"] for row in selected
            ),
            "peak_memory_used_mib": max(
                row["memory_used_mib"] for row in selected
            ),
        }
    rebuilt = {"all_four_attributed_positive": True, "by_gpu": by_gpu}
    if report.get("gpu_activity") != rebuilt:
        raise RuntimeError("v64 reported GPU summary differs from numeric log")
    return {
        **rebuilt,
        "gpu_log_rows": len(rows),
        "samples_per_gpu": samples_per_gpu,
        "all_28_generation_phases_observed_on_every_gpu": True,
        "all_28_generation_phases_attributed_positive_on_every_gpu": True,
        "foreign_compute_process_observations": 0,
    }


def _verify_panel_projection_v64(
    panel: dict,
    evidence: dict,
    report: dict,
) -> dict:
    items = panel.get("items", [])
    evidence_rows = evidence.get("rows", [])
    panel_projection = [{
        "request_index": item.get("request_index"),
        "row_sha256": item.get("row_sha256"),
        "unit_identity_sha256": item.get("unit_identity_sha256"),
        "role": item.get("role"),
    } for item in items]
    evidence_projection = [{
        "request_index": item.get("request_index"),
        "row_sha256": item.get("row_sha256"),
        "unit_identity_sha256": item.get("unit_identity_sha256"),
        "role": item.get("role"),
    } for item in evidence_rows]
    if (
        panel.get("schema") != "v61c-paired-null-calibration-panel"
        or panel.get("status") != "sealed_cpu_only_before_v61c_preregistration"
        or panel.get("ranking_units") != analysis.RANKING_UNITS_V64
        or panel.get("exact_sentinel_units") != analysis.EXACT_SENTINEL_UNITS_V64
        or panel.get("question_answer_or_generation_text_persisted_in_panel")
        is not False
        or panel.get("protected_semantics_opened") is not False
        or panel.get("model_or_gpu_accessed") is not False
        or not isinstance(items, list) or len(items) != analysis.ROWS_V64
        or [item.get("request_index") for item in items]
        != list(range(analysis.ROWS_V64))
        or panel.get("request_order_row_sha256")
        != [item.get("row_sha256") for item in items]
        or panel.get("request_order_sha256")
        != runtime.PANEL_REQUEST_ORDER_SHA256_V64
        or analysis.canonical_sha256_v64(panel_projection)
        != runtime.PANEL_REQUEST_PROJECTION_SHA256_V64
        or analysis.canonical_sha256_v64(panel.get("document_block_audit"))
        != runtime.PANEL_DOCUMENT_BLOCK_AUDIT_SHA256_V64
        or panel_projection != evidence_projection
        or report.get("panel_document_block_audit")
        != panel.get("document_block_audit")
    ):
        raise RuntimeError("v64 sealed numeric panel projection changed")
    return {
        "rows": analysis.ROWS_V64,
        "ranking_units": analysis.RANKING_UNITS_V64,
        "exact_sentinel_units": analysis.EXACT_SENTINEL_UNITS_V64,
        "evidence_request_order_matches_sealed_numeric_panel": True,
        "document_block_audit_exact": True,
        "question_answer_or_generation_text_in_panel": False,
    }


def _verify_report_v64(
    report: dict,
    prereg: dict,
    attempt: dict,
    panel: dict,
    evidence: dict,
    stored_analysis: dict,
    sources: FinalizerSourcesV64,
) -> dict:
    gate = stored_analysis["required_confirmation_gate"]
    expected_status = (
        "complete_gate_passed_without_promotion_authority"
        if gate["passed"] else "complete_gate_failed_closed"
    )
    expected_report_keys = {
        "schema", "status", "started_at_utc", "completed_at_utc",
        "wall_runtime_seconds", "preregistration_file_sha256",
        "preregistration_content_sha256", "attempt", "v62b_finalized",
        "adapter_artifact_identities", "base_model_prelaunch_artifact_receipt",
        "base_model_postrun_artifact_receipt", "two_standard_lora_requests",
        "panel_file_sha256", "panel_content_sha256",
        "panel_document_block_audit", "actor_identities",
        "warmup_state_receipts_sha256", "scored_state_receipts_sha256",
        "evidence", "analysis", "gpu_activity", "cleanup", "final_gpu_idle",
        "gpu_log_file_sha256", "generation_only", "teacher_forced_requests",
        "adaptive_retry_drop_reorder_or_early_stop_performed",
        "median_consensus_or_best_of_selection_performed",
        "adapter_update_hpo_master_checkpoint_or_promotion_performed",
        "holdback_ood_shadow_or_protected_opened",
        "raw_question_answer_or_generation_text_persisted",
        "result_authorizes_update_hpo_promotion_or_protected_access",
        "content_sha256_before_self_field",
    }
    evidence_ref = report.get("evidence", {})
    analysis_ref = report.get("analysis", {})
    attempt_ref = report.get("attempt", {})
    expected_evidence_keys = {
        "path", "file_sha256", "content_sha256", "rows", "actors",
        "scored_periods", "pairs_per_actor", "replicas_per_conflict_unit",
        "all_scored_periods_included_without_early_stop",
    }
    expected_analysis_keys = {
        "path", "file_sha256", "content_sha256",
        "required_confirmation_gate", "exact_sentinel_diagnostics",
    }
    expected_attempt_keys = {"path", "file_sha256", "content_sha256"}
    base_model_prelaunch = runtime.validate_base_model_artifact_receipt_v64(
        report.get("base_model_prelaunch_artifact_receipt"),
        prereg["base_model_artifact_expectation"],
    )
    base_model_postrun = runtime.validate_base_model_artifact_receipt_v64(
        report.get("base_model_postrun_artifact_receipt"),
        prereg["base_model_artifact_expectation"],
    )
    pid_map = _actor_pid_map_v64(report, prereg)
    cleanup = _verify_cleanup_v64(
        report.get("cleanup"), report.get("final_gpu_idle")
    )
    gpu = _gpu_summary_v64(
        sources.gpu_log_path,
        sources.gpu_log_file_sha256,
        report,
        pid_map,
    )
    panel_verification = _verify_panel_projection_v64(panel, evidence, report)
    wall = report.get("wall_runtime_seconds")
    if (
        set(report) != expected_report_keys
        or report.get("schema")
        != "v64-v59-vs-v434-train-only-confirmation-report"
        or report.get("status") != expected_status
        or not isinstance(report.get("started_at_utc"), str)
        or report.get("started_at_utc") != attempt.get("started_at_utc")
        or not isinstance(report.get("completed_at_utc"), str)
        or isinstance(wall, bool) or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall)) or float(wall) <= 0.0
        or report.get("preregistration_file_sha256")
        != sources.preregistration.file_sha256
        or report.get("preregistration_content_sha256")
        != sources.preregistration.content_sha256
        or not isinstance(attempt_ref, dict)
        or set(attempt_ref) != expected_attempt_keys
        or Path(attempt_ref.get("path", "")).resolve() != sources.attempt.path
        or attempt_ref.get("file_sha256") != sources.attempt.file_sha256
        or attempt_ref.get("content_sha256") != sources.attempt.content_sha256
        or report.get("v62b_finalized")
        != runtime.verify_v62b_eligibility_v64()
        or report.get("adapter_artifact_identities")
        != analysis.expected_adapter_identities_v64()
        or base_model_prelaunch
        != attempt.get("base_model_artifact_receipt")
        or base_model_postrun != base_model_prelaunch
        or report.get("two_standard_lora_requests") != {
            "reference_id": analysis.REFERENCE_LORA_ID_V64,
            "candidate_id": analysis.CANDIDATE_LORA_ID_V64,
            "max_loras": 1,
            "max_cpu_loras": 2,
            "sequential_period_switching": True,
        }
        or report.get("panel_file_sha256")
        != prereg["fixed_confirmation_recipe"]["staged_panel_file_sha256"]
        or report.get("panel_content_sha256")
        != prereg["fixed_confirmation_recipe"]["staged_panel_content_sha256"]
        or report.get("warmup_state_receipts_sha256")
        != evidence.get("numeric_warmup_state_receipts_sha256")
        or report.get("scored_state_receipts_sha256")
        != evidence.get("numeric_scored_state_receipts_sha256")
        or not isinstance(evidence_ref, dict)
        or set(evidence_ref) != expected_evidence_keys
        or Path(evidence_ref.get("path", "")).resolve() != sources.evidence.path
        or evidence_ref.get("file_sha256") != sources.evidence.file_sha256
        or evidence_ref.get("content_sha256") != sources.evidence.content_sha256
        or evidence_ref.get("rows") != analysis.ROWS_V64
        or evidence_ref.get("actors") != analysis.ACTORS_V64
        or evidence_ref.get("scored_periods") != analysis.SCORED_PERIODS_V64
        or evidence_ref.get("pairs_per_actor") != analysis.PAIRS_PER_ACTOR_V64
        or evidence_ref.get("replicas_per_conflict_unit")
        != analysis.REPLICAS_PER_UNIT_V64
        or evidence_ref.get("all_scored_periods_included_without_early_stop")
        is not True
        or not isinstance(analysis_ref, dict)
        or set(analysis_ref) != expected_analysis_keys
        or Path(analysis_ref.get("path", "")).resolve() != sources.analysis.path
        or analysis_ref.get("file_sha256") != sources.analysis.file_sha256
        or analysis_ref.get("content_sha256") != sources.analysis.content_sha256
        or analysis_ref.get("required_confirmation_gate") != gate
        or analysis_ref.get("exact_sentinel_diagnostics")
        != stored_analysis["exact_sentinel_diagnostics"]
        or report.get("gpu_log_file_sha256")
        != sources.gpu_log_file_sha256
        or report.get("generation_only") is not True
        or report.get("teacher_forced_requests") != 0
        or report.get(
            "adaptive_retry_drop_reorder_or_early_stop_performed"
        ) is not False
        or report.get("median_consensus_or_best_of_selection_performed")
        is not False
        or report.get(
            "adapter_update_hpo_master_checkpoint_or_promotion_performed"
        ) is not False
        or report.get("holdback_ood_shadow_or_protected_opened") is not False
        or report.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or report.get(
            "result_authorizes_update_hpo_promotion_or_protected_access"
        ) is not False
        or evidence.get("content_sha256_before_self_field")
        != sources.evidence.content_sha256
    ):
        raise RuntimeError("v64 finalizer report or integrity changed")
    return {
        "actor_count": analysis.ACTORS_V64,
        "physical_gpu_ids": [0, 1, 2, 3],
        "unique_processes": analysis.ACTORS_V64,
        "actor_runtime_controls_exact": True,
        "standard_lora_request_ids_and_capacity_exact": True,
        "base_model_full_byte_hash_prelaunch_and_postrun_exact": True,
        "panel_hashes_exact": True,
        "sealed_numeric_panel_projection": panel_verification,
        "state_receipt_hash_crosslinks_exact": True,
        "evidence_coverage_and_analysis_crosslinks_exact": True,
        "gpu_activity_recomputed_from_numeric_log": gpu,
        "cleanup_and_final_idle": cleanup,
        "fixed_complete_schedule": True,
        "result_authority": False,
    }


def finalize_v64(sources: FinalizerSourcesV64) -> dict:
    prereg = _read_self_hashed_v64(sources.preregistration)
    attempt = _read_self_hashed_v64(sources.attempt)
    panel = _read_self_hashed_v64(sources.panel)
    evidence = _read_self_hashed_v64(sources.evidence)
    stored_analysis = _read_self_hashed_v64(sources.analysis)
    report = _read_self_hashed_v64(sources.report)
    no_text = {
        name: _verify_no_text_keys_v64(name, value)
        for name, value in (
            ("preregistration", prereg),
            ("attempt", attempt),
            ("panel", panel),
            ("evidence", evidence),
            ("analysis", stored_analysis),
            ("report", report),
        )
    }
    prereg_verification = _verify_preregistration_v64(prereg)
    attempt_verification = _verify_attempt_v64(attempt, prereg, sources)
    analysis.validate_evidence_v64(evidence)
    rebuilt = analysis.build_analysis_v64(evidence)
    if rebuilt != stored_analysis:
        raise RuntimeError("v64 stored analysis differs from exact numeric rebuild")
    report_verification = _verify_report_v64(
        report, prereg, attempt, panel, evidence, stored_analysis, sources
    )
    gate = copy.deepcopy(stored_analysis["required_confirmation_gate"])
    failed = [name for name, passed in gate["checks"].items() if not passed]
    value = {
        "schema": "v64-v59-vs-v434-independent-numeric-finalizer",
        "status": (
            "complete_gate_passed_without_authority"
            if gate["passed"] else "complete_gate_failed_closed"
        ),
        "source_hashes": {
            name: {
                "file_sha256": source.file_sha256,
                "content_sha256": source.content_sha256,
            }
            for name, source in (
                ("preregistration", sources.preregistration),
                ("attempt", sources.attempt),
                ("panel", sources.panel),
                ("evidence", sources.evidence),
                ("analysis", sources.analysis),
                ("report", sources.report),
            )
        },
        "observed_numeric_outcome_without_authorization": {
            "primary_generated_f1": copy.deepcopy(
                stored_analysis["primary_generated_f1"]
            ),
            "actor_influence": copy.deepcopy(
                stored_analysis["actor_influence"]
            ),
            "required_confirmation_gate": gate,
            "exact_sentinel_diagnostics": copy.deepcopy(
                stored_analysis["exact_sentinel_diagnostics"]
            ),
            "failed_gates": failed,
            "passed_gate_count": len(gate["checks"]) - len(failed),
            "failed_gate_count": len(failed),
        },
        "verification": {
            "preregistration": prereg_verification,
            "exclusive_idle_launch_attempt": attempt_verification,
            "report": report_verification,
            "analysis_exactly_rebuilt": True,
            "no_text_leakage": no_text,
            "all_file_and_self_hashes_verified": True,
        },
        "gpu_log_file_sha256": sources.gpu_log_file_sha256,
        "frozen_non_authorization": {
            "finalizer_accepts_and_records_either_gate_outcome": True,
            "outcome_assumed_before_read": False,
            "thresholds_changed_after_outcome": False,
            "failed_gate_reinterpreted_or_relaxed": False,
            "gpu_or_model_launch_authorized": False,
            "adapter_update_hpo_candidate_promotion_authorized": False,
            "holdback_ood_shadow_terminal_or_protected_access_authorized": False,
        },
        "raw_question_answer_prediction_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v64(value)
    )
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-file-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--attempt-file-sha256", required=True)
    parser.add_argument("--attempt-content-sha256", required=True)
    parser.add_argument("--evidence-file-sha256", required=True)
    parser.add_argument("--evidence-content-sha256", required=True)
    parser.add_argument("--analysis-file-sha256", required=True)
    parser.add_argument("--analysis-content-sha256", required=True)
    parser.add_argument("--report-file-sha256", required=True)
    parser.add_argument("--report-content-sha256", required=True)
    parser.add_argument("--gpu-log-file-sha256", required=True)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    result = finalize_v64(production_sources_v64(
        args.preregistration_file_sha256,
        args.preregistration_content_sha256,
        args.attempt_file_sha256,
        args.attempt_content_sha256,
        args.evidence_file_sha256,
        args.evidence_content_sha256,
        args.analysis_file_sha256,
        args.analysis_content_sha256,
        args.report_file_sha256,
        args.report_content_sha256,
        args.gpu_log_file_sha256,
    ))
    runtime.runtime_v61a.atomic_json_v61a(output, result)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.file_sha256_v64(output),
        "content_sha256": result["content_sha256_before_self_field"],
        "required_confirmation_gate_passed": result[
            "observed_numeric_outcome_without_authorization"
        ]["required_confirmation_gate"]["passed"],
        "update_hpo_promotion_or_protected_access_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
