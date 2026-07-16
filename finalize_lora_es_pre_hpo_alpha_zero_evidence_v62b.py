#!/usr/bin/env python3
"""Outcome-agnostic numeric finalizer for the sealed V62B live run."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import audit_vllm_pre_hpo_alpha_zero_support_v62b as support_audit
import lora_es_pre_hpo_alpha_zero_calibration_v62b as analysis
import run_lora_es_pre_hpo_alpha_zero_calibration_v62b as runtime


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v62b_v434_pre_hpo_alpha_zero_generation_calibration"
).resolve()
EVIDENCE = (RUN_DIR / "alpha_zero_evidence_v62b.json").resolve()
ANALYSIS = (RUN_DIR / "alpha_zero_analysis_v62b.json").resolve()
REPORT = (RUN_DIR / "alpha_zero_report_v62b.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v62b.jsonl").resolve()
OUTPUT = (RUN_DIR / "alpha_zero_finalized_v62b.json").resolve()
TESTS = (
    ROOT / "test_finalize_lora_es_pre_hpo_alpha_zero_evidence_v62b.py"
).resolve()

PREREGISTRATION_FILE_SHA256 = (
    "c3a367b54785265a7a40bc1fa215c59fb7106a603e0197bdd3d0abb3628fccac"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "cf3fd16916477e8f2ff381fdbb7e978554c7379cdd2512ba2cabaded32ca8bbb"
)
SUPPORT_AUDIT_FILE_SHA256 = runtime.SUPPORT_AUDIT_FILE_SHA256
SUPPORT_AUDIT_CONTENT_SHA256 = runtime.SUPPORT_AUDIT_CONTENT_SHA256
EVIDENCE_FILE_SHA256: str | None = (
    "84777f96c6b6fa21d8e58c983263f1ae25d3296370d1800aaebbe62f57acd0ea"
)
EVIDENCE_CONTENT_SHA256: str | None = (
    "6fcd0123a48e20795c90826ef837687366a3d218321db5981fe5ecc59a7cee6f"
)
ANALYSIS_FILE_SHA256: str | None = (
    "8d8e5f2da35723a6eefb315905bf333cd3ea700ef4c2687d5f47fc873f36e978"
)
ANALYSIS_CONTENT_SHA256: str | None = (
    "156db636638749c5aa4bda8f2080d9798fb3d826cbbebbeb606184619409e8c1"
)
REPORT_FILE_SHA256: str | None = (
    "4383af865d8359d5cb9529f6eaa5f1c7b003c88e3ecc73c8b52032c8a3bb4489"
)
REPORT_CONTENT_SHA256: str | None = (
    "c9d5e13d99a93878e1f5dbc94dd778e11fbb658670c89cc377b4d3043b7f4203"
)
GPU_LOG_FILE_SHA256: str | None = (
    "87807b20b8b20d7d11ae39f3cbee8e87e795b19eac1e1b92e8e5783604db7f74"
)

FORBIDDEN_TEXT_KEYS_V62B = {
    "answer",
    "completion",
    "completion_text",
    "generated_text",
    "output_text",
    "outputs",
    "prediction",
    "prompt",
    "prompt_token_ids",
    "question",
    "raw_text",
    "response",
    "text",
    "token_ids",
}


@dataclass(frozen=True)
class SelfHashedSourceV62B:
    path: Path
    file_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class FinalizerSourcesV62B:
    preregistration: SelfHashedSourceV62B
    support_audit: SelfHashedSourceV62B
    evidence: SelfHashedSourceV62B
    analysis: SelfHashedSourceV62B
    report: SelfHashedSourceV62B
    gpu_log_path: Path
    gpu_log_file_sha256: str


def file_sha256_v62b(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256_v62b(value: str | None, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v62b production outcome hash is not sealed: {name}")
    return value


def production_sources_v62b() -> FinalizerSourcesV62B:
    return FinalizerSourcesV62B(
        preregistration=SelfHashedSourceV62B(
            runtime.PREREGISTRATION,
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_CONTENT_SHA256,
        ),
        support_audit=SelfHashedSourceV62B(
            runtime.SUPPORT_AUDIT,
            SUPPORT_AUDIT_FILE_SHA256,
            SUPPORT_AUDIT_CONTENT_SHA256,
        ),
        evidence=SelfHashedSourceV62B(
            EVIDENCE,
            _require_sha256_v62b(EVIDENCE_FILE_SHA256, "evidence file"),
            _require_sha256_v62b(EVIDENCE_CONTENT_SHA256, "evidence content"),
        ),
        analysis=SelfHashedSourceV62B(
            ANALYSIS,
            _require_sha256_v62b(ANALYSIS_FILE_SHA256, "analysis file"),
            _require_sha256_v62b(ANALYSIS_CONTENT_SHA256, "analysis content"),
        ),
        report=SelfHashedSourceV62B(
            REPORT,
            _require_sha256_v62b(REPORT_FILE_SHA256, "report file"),
            _require_sha256_v62b(REPORT_CONTENT_SHA256, "report content"),
        ),
        gpu_log_path=GPU_LOG,
        gpu_log_file_sha256=_require_sha256_v62b(
            GPU_LOG_FILE_SHA256, "GPU log file"
        ),
    )


def _read_self_hashed_v62b(source: SelfHashedSourceV62B) -> dict:
    if file_sha256_v62b(source.path) != source.file_sha256:
        raise RuntimeError(f"v62b finalizer input file changed: {source.path}")
    value = json.loads(source.path.read_text(encoding="utf-8"))
    compact = {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != source.content_sha256
        or analysis.canonical_sha256_v62b(compact) != source.content_sha256
    ):
        raise RuntimeError(f"v62b finalizer input content changed: {source.path}")
    return value


def _verify_static_sources_v62b(prereg: dict, audit: dict, sources) -> dict:
    installed = {
        key: file_sha256_v62b(path)
        for key, path in support_audit.v62a.v61e.SOURCE_PATHS.items()
    }
    projection = audit.get("intended_evaluator_projection", {})
    if (
        prereg.get("schema")
        != "v62b-v434-pre-hpo-alpha-zero-generation-preregistration"
        or prereg.get("status")
        != "preregistered_before_train_semantics_model_or_gpu_access"
        or prereg.get("v62_methodology_commit") != analysis.V62_METHOD_COMMIT
        or prereg.get("v62_numeric_audit_identities")
        != analysis.V62_NUMERIC_AUDIT_IDENTITIES
        or prereg.get("v62_preregistration_identities")
        != analysis.V62_PREREGISTRATION_IDENTITIES
        or prereg.get("implementation_bindings")
        != runtime.implementation_bindings_v62b()
        or prereg.get("specific_alpha_zero_calibration_gpu_launch_authorized")
        is not True
        or prereg.get("support_audit_alone_authorizes_gpu_launch") is not False
        or prereg.get("builder_or_dry_run_performed_gpu_launch") is not False
        or prereg.get("hpo_population_update_or_candidate_authorized")
        is not False
        or prereg.get("ood_shadow_holdout_or_protected_access_authorized")
        is not False
        or prereg.get("installed_runtime_support_audit", {}).get("file_sha256")
        != sources.support_audit.file_sha256
        or prereg.get("installed_runtime_support_audit", {}).get(
            "content_sha256"
        ) != sources.support_audit.content_sha256
        or audit.get("schema")
        != "v62b-installed-vllm-pre-hpo-alpha-zero-support-audit"
        or audit.get("status") != "supported"
        or audit.get("installed_vllm_source_file_sha256") != installed
        or audit.get("requested_runtime_controls")
        != analysis.RUNTIME_CONTROLS_V62B
        or audit.get("pre_hpo_alpha_zero_runtime_supported") is not True
        or audit.get("support_audit_authorizes_gpu_launch") is not False
        or audit.get("model_train_semantics_or_gpu_accessed") is not False
        or projection.get("unscored_warmup_periods")
        != analysis.WARMUP_PERIODS_V62B
        or projection.get("scored_periods") != analysis.SCORED_PERIODS_V62B
        or projection.get("total_sequential_periods")
        != analysis.TOTAL_PERIODS_V62B
        or projection.get("scored_replicas_per_conflict_unit")
        != analysis.REPLICAS_PER_UNIT_V62B
        or projection.get("warmup_outputs_scored_or_persisted") is not False
    ):
        raise RuntimeError("v62b preregistration, support, or code binding changed")
    return {
        "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
        "v62_numeric_audit_identities": dict(
            analysis.V62_NUMERIC_AUDIT_IDENTITIES
        ),
        "v62_preregistration_identities": dict(
            analysis.V62_PREREGISTRATION_IDENTITIES
        ),
        "installed_vllm_source_identities_exact": True,
        "implementation_bindings_exact": True,
        "support_audit_did_not_authorize_launch": True,
    }


def _verify_no_text_keys_v62b(name: str, value: object) -> dict:
    found = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).lower()
                if normalized in FORBIDDEN_TEXT_KEYS_V62B:
                    found.append(f"{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, name)
    if found:
        raise RuntimeError(f"v62b forbidden text-bearing key observed: {found[0]}")
    return {
        "source": name,
        "forbidden_text_key_count": 0,
        "forbidden_exact_keys": sorted(FORBIDDEN_TEXT_KEYS_V62B),
    }


def _verify_analysis_rebuild_v62b(evidence: dict, stored: dict) -> dict:
    rebuilt = analysis.build_analysis_v62b(evidence)
    if rebuilt != stored:
        raise RuntimeError("v62b stored analysis differs from exact numeric rebuild")
    return rebuilt


def _verify_actor_identities_v62b(report: dict, prereg: dict) -> dict:
    actors = report.get("actor_identities", [])
    expected_tuned = prereg["runtime"]["tuned_table_content_sha256"]
    expected_folder = prereg["runtime"]["tuned_folder"]
    by_gpu = {}
    pids = set()
    for item in actors:
        gpu = item.get("physical_gpu_id")
        pid = item.get("pid")
        controls = {
            key: item.get(key) for key in analysis.RUNTIME_CONTROLS_V62B
        }
        if (
            gpu not in (0, 1, 2, 3)
            or not isinstance(pid, int)
            or pid <= 0
            or gpu in by_gpu
            or pid in pids
            or item.get("schema")
            != "pre-hpo-alpha-zero-actor-identity-v62a"
            or item.get("cuda_visible_devices") != str(gpu)
            or item.get("cuda_current_device") != 0
            or controls != analysis.RUNTIME_CONTROLS_V62B
            or item.get("enforce_eager") is not True
            or item.get("VLLM_BATCH_INVARIANT") is not False
            or item.get("max_num_seqs") != 68
            or item.get("scheduler_class") != "Scheduler"
            or item.get("submitted_request_batch_size") != 68
            or item.get("generation_only") is not True
            or item.get("global_batch_invariance_claimed") is not False
            or item.get("tuned_folder") != expected_folder
            or item.get("tuned_table_content_sha256") != expected_tuned
        ):
            raise RuntimeError("v62b actor runtime identity changed")
        by_gpu[gpu] = pid
        pids.add(pid)
    if len(actors) != 4 or set(by_gpu) != {0, 1, 2, 3}:
        raise RuntimeError("v62b actor GPU coverage changed")
    return {
        "actors": 4,
        "unique_processes": 4,
        "physical_gpu_ids": [0, 1, 2, 3],
        "enforce_eager_all_actors": True,
        "batch_invariance_all_actors": False,
        "max_num_seqs_all_actors": 68,
        "all_runtime_controls_exact": True,
        "all_v27c_tuned_table_identities_exact": True,
    }


def _verify_state_v62b(evidence: dict, report: dict, prereg: dict) -> dict:
    warmup = evidence.get("warmup_state_receipts", [])
    scored = evidence.get("scored_state_receipts", [])
    recipe = prereg["fixed_calibration_recipe"]
    all_receipts = [*warmup, *scored]
    if not all_receipts:
        raise RuntimeError("v62b V434 state receipts missing")
    master = all_receipts[0].get("before")
    if (
        len(warmup) != 4
        or len(scored) != 24
        or [item.get("period_index") for item in warmup] != list(range(4))
        or [item.get("period_index") for item in scored] != list(range(24))
        or any(
            item.get("identical_v434_state") is not True
            or item.get("before") != item.get("after")
            or item.get("before") != master
            or item.get("before", {}).get("canonical_fp32_master_sha256")
            != recipe["canonical_fp32_master_sha256"]
            or item.get("before", {}).get("bf16_runtime_values_sha256")
            != recipe["bf16_runtime_values_sha256"]
            for item in all_receipts
        )
        or evidence.get("numeric_warmup_state_receipts_sha256")
        != analysis.canonical_sha256_v62b(warmup)
        or evidence.get("numeric_scored_state_receipts_sha256")
        != analysis.canonical_sha256_v62b(scored)
        or report.get("warmup_state_receipts_sha256")
        != evidence.get("numeric_warmup_state_receipts_sha256")
        or report.get("scored_state_receipts_sha256")
        != evidence.get("numeric_scored_state_receipts_sha256")
        or report.get("master_state_receipt") != master
    ):
        raise RuntimeError("v62b identical V434 state receipts changed")
    return {
        "unscored_warmup_periods": 4,
        "scored_periods": 24,
        "total_periods": 28,
        "identical_before_after_periods": 28,
        "single_unchanged_master_receipt_across_all_periods": True,
        "canonical_fp32_master_sha256": recipe[
            "canonical_fp32_master_sha256"
        ],
        "bf16_runtime_values_sha256": recipe["bf16_runtime_values_sha256"],
    }


def _verify_fixed_schedule_v62b(
    evidence: dict,
    report: dict,
    prereg: dict,
    rebuilt: dict,
) -> dict:
    recipe = prereg.get("fixed_calibration_recipe", {})
    warmup = report.get("warmup", {})
    receipts = rebuilt.get("counterbalance_pair_receipts", [])
    if (
        recipe.get("actors") != 4
        or recipe.get("rows") != 68
        or recipe.get("unscored_warmup_periods") != 4
        or recipe.get("scored_sequential_periods") != 24
        or recipe.get("total_sequential_periods") != 28
        or recipe.get("scored_counterbalanced_blocks") != 6
        or recipe.get("counterbalanced_pairs_per_actor") != 12
        or recipe.get("replicas_per_conflict_unit") != 48
        or recipe.get("all_scored_periods_included") is not True
        or recipe.get("warmup_outputs_scored_or_persisted") is not False
        or recipe.get("warmup_adaptive_retry_drop_or_reorder") is not False
        or recipe.get("scored_adaptive_retry_drop_reorder_or_early_stop")
        is not False
        or evidence.get("unscored_warmup_period_count") != 4
        or evidence.get("scored_period_count") != 24
        or evidence.get("total_period_count") != 28
        or evidence.get("warmup_generation_completions_discarded") != 1088
        or evidence.get("scored_generation_completions") != 6528
        or evidence.get("total_generation_completions") != 7616
        or evidence.get("warmup_raw_outputs_persisted") is not False
        or evidence.get("warmup_generation_metrics_computed_or_persisted")
        is not False
        or evidence.get("warmup_adaptive_retry_drop_or_reorder_performed")
        is not False
        or evidence.get(
            "scored_period_adaptive_retry_drop_reorder_or_early_stop_performed"
        ) is not False
        or len(evidence.get("rows", [])) != 68
        or any(len(row.get("scored_periods", [])) != 24 for row in evidence["rows"])
        or len(receipts) != 48
        or warmup.get("periods") != 4
        or warmup.get("generation_completions_discarded") != 1088
        or warmup.get("raw_outputs_persisted") is not False
        or warmup.get("generation_metrics_computed_or_persisted") is not False
        or warmup.get("adaptive_retry_drop_or_reorder_performed") is not False
        or report.get("evidence", {}).get(
            "all_scored_periods_included_without_early_stop"
        ) is not True
        or rebuilt.get("unscored_warmup_excluded_from_every_metric") is not True
    ):
        raise RuntimeError("v62b fixed warmup or scored schedule changed")
    return {
        "rows": 68,
        "actors": 4,
        "unscored_warmup_periods": 4,
        "warmup_generation_completions_discarded": 1088,
        "scored_periods": 24,
        "scored_generation_completions": 6528,
        "total_generation_completions": 7616,
        "counterbalanced_pairs_per_actor": 12,
        "replicas_per_conflict_unit": 48,
        "all_scored_periods_included_without_early_stop": True,
        "warmup_outputs_scored_or_persisted": False,
        "adaptive_schedule_change_performed": False,
    }


def _gpu_summary_v62b(path: Path, expected_sha: str, report: dict) -> dict:
    if file_sha256_v62b(path) != expected_sha:
        raise RuntimeError("v62b GPU log changed")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    _verify_no_text_keys_v62b("gpu_log", rows)
    expected_keys = {
        "compute_pids",
        "expected_pid",
        "foreign_compute_pids",
        "gpu",
        "memory_used_mib",
        "phase",
        "sampled_at_utc",
        "utilization_percent",
    }
    if len(rows) != 2336 or any(set(row) != expected_keys for row in rows):
        raise RuntimeError("v62b GPU log row count or schema changed")
    actor_pids = {
        int(item["physical_gpu_id"]): int(item["pid"])
        for item in report.get("actor_identities", [])
    }
    if set(actor_pids) != {0, 1, 2, 3} or len(set(actor_pids.values())) != 4:
        raise RuntimeError("v62b GPU actor map changed")
    by_gpu = {}
    for gpu in range(4):
        expected_pid = actor_pids[gpu]
        selected = [item for item in rows if item.get("gpu") == gpu]
        if (
            len(selected) != 584
            or any(item.get("expected_pid") != expected_pid for item in selected)
            or any(item.get("foreign_compute_pids") != [] for item in selected)
            or any(item.get("compute_pids") != [expected_pid] for item in selected)
        ):
            raise RuntimeError("v62b foreign or mismatched GPU process observed")
        resident = [item for item in selected if expected_pid in item["compute_pids"]]
        if (
            len(resident) != 584
            or not any(item["utilization_percent"] > 0 for item in resident)
            or max(item["utilization_percent"] for item in resident) != 100
        ):
            raise RuntimeError("v62b GPU activity or 100-percent peak changed")
        by_gpu[str(gpu)] = {
            "expected_pid": expected_pid,
            "samples": len(selected),
            "resident_samples": len(resident),
            "positive_samples": sum(
                item["utilization_percent"] > 0 for item in resident
            ),
            "mean_resident_utilization_percent": math.fsum(
                item["utilization_percent"] for item in resident
            ) / len(resident),
            "peak_utilization_percent": max(
                item["utilization_percent"] for item in resident
            ),
            "peak_memory_used_mib": max(
                item["memory_used_mib"] for item in resident
            ),
        }
    rebuilt = {"all_four_attributed_positive": True, "by_gpu": by_gpu}
    if report.get("gpu_activity") != rebuilt:
        raise RuntimeError("v62b reported GPU summary differs from numeric log")
    return {
        **rebuilt,
        "gpu_log_rows": 2336,
        "samples_per_gpu": 584,
        "all_four_peak_utilization_percent": 100,
        "foreign_compute_process_observations": 0,
        "forbidden_text_key_count": 0,
    }


def _gate_observation_v62b(gate: dict) -> dict:
    checks = gate.get("checks", {})
    expected_keys = {
        "null_primary_ci_contains_zero",
        "primary_ci_halfwidth_at_most_frozen_limit_inclusive",
        "actor_leave_one_out_shift_at_most_frozen_limit_inclusive",
    }
    if (
        set(checks) != expected_keys
        or any(not isinstance(value, bool) for value in checks.values())
        or gate.get("passed") is not all(checks.values())
        or gate.get("maximum_primary_ci_halfwidth_inclusive")
        != analysis.MAX_PRIMARY_CI_HALFWIDTH_V62B
        or gate.get("maximum_actor_leave_one_out_shift_inclusive")
        != analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B
        or gate.get("failure_action")
        != "fail_closed_before_hpo_population_or_update"
        or gate.get("authorizes_hpo_population_or_update") is not False
    ):
        raise RuntimeError("v62b frozen gate semantics changed")
    failed = sorted(key for key, passed in checks.items() if not passed)
    return {
        "checks": copy.deepcopy(checks),
        "passed": gate["passed"],
        "passed_gate_count": len(checks) - len(failed),
        "failed_gate_count": len(failed),
        "failed_gates": failed,
        "maximum_primary_ci_halfwidth_inclusive": (
            analysis.MAX_PRIMARY_CI_HALFWIDTH_V62B
        ),
        "maximum_actor_leave_one_out_shift_inclusive": (
            analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B
        ),
        "failure_action": gate["failure_action"],
        "authorizes_hpo_population_or_update": False,
    }


def _verify_report_v62b(report: dict, rebuilt: dict, sources) -> dict:
    gate = rebuilt["required_pre_hpo_gate"]
    expected_status = (
        "complete_gate_passed_hpo_still_unauthorized"
        if gate["passed"]
        else "complete_gate_failed_closed"
    )
    cleanup = report.get("cleanup", {})
    before = cleanup.get("before", [])
    after = cleanup.get("after", [])
    evidence_ref = report.get("evidence", {})
    analysis_ref = report.get("analysis", {})
    if (
        report.get("schema") != "v62b-pre-hpo-alpha-zero-generation-report"
        or report.get("status") != expected_status
        or report.get("preregistration_file_sha256")
        != sources.preregistration.file_sha256
        or report.get("preregistration_content_sha256")
        != sources.preregistration.content_sha256
        or report.get("support_audit_file_sha256")
        != sources.support_audit.file_sha256
        or report.get("support_audit_content_sha256")
        != sources.support_audit.content_sha256
        or report.get("v62_methodology_commit") != analysis.V62_METHOD_COMMIT
        or evidence_ref.get("file_sha256") != sources.evidence.file_sha256
        or evidence_ref.get("content_sha256") != sources.evidence.content_sha256
        or evidence_ref.get("rows") != 68
        or evidence_ref.get("actors") != 4
        or evidence_ref.get("scored_periods") != 24
        or evidence_ref.get("pairs_per_actor") != 12
        or evidence_ref.get("replicas_per_conflict_unit") != 48
        or evidence_ref.get("scored_generation_completions") != 6528
        or evidence_ref.get("total_generation_completions") != 7616
        or evidence_ref.get("teacher_forced_requests") != 0
        or evidence_ref.get("all_scored_periods_included_without_early_stop")
        is not True
        or analysis_ref.get("file_sha256") != sources.analysis.file_sha256
        or analysis_ref.get("content_sha256") != sources.analysis.content_sha256
        or analysis_ref.get("required_pre_hpo_gate") != gate
        or analysis_ref.get("exact_sentinel_diagnostics")
        != rebuilt["exact_sentinel_diagnostics"]
        or report.get("gpu_log_file_sha256") != sources.gpu_log_file_sha256
        or report.get("generation_only") is not True
        or report.get("alpha") != 0.0
        or report.get("adapter_update_candidate_or_hpo_performed") is not False
        or report.get("holdback_ood_shadow_or_protected_opened") is not False
        or report.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or report.get("hpo_population_launch_authorized") is not False
        or cleanup.get("engine_kill_count") != 4
        or cleanup.get("placement_group_remove_count") != 4
        or cleanup.get("all_four_gcs_states_removed") is not True
        or len(before) != 4
        or any(item.get("state") != "CREATED" for item in before)
        or len(after) != 4
        or any(item.get("state") != "REMOVED" for item in after)
        or report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
    ):
        raise RuntimeError("v62b sealed report, hash chain, or cleanup changed")
    return {
        "reported_status_matches_observed_gate": True,
        "engines_killed": 4,
        "placement_groups_removed": 4,
        "all_four_gcs_states_removed": True,
        "all_four_final_gpu_compute_process_lists_empty": True,
        "no_hpo_update_candidate_or_protected_opening": True,
    }


def _performance_v62b(report: dict) -> dict:
    wall = float(report.get("wall_runtime_seconds", 0.0))
    if not math.isfinite(wall) or wall <= 0.0:
        raise RuntimeError("v62b wall runtime changed")
    return {
        "wall_runtime_seconds": wall,
        "unscored_warmup_generation_completions": 1088,
        "scored_generation_completions": 6528,
        "total_generation_completions": 7616,
        "teacher_forced_requests": 0,
        "total_generation_completions_per_second": 7616 / wall,
        "scored_generation_completions_per_second": 6528 / wall,
    }


def build_finalized_v62b(
    sources: FinalizerSourcesV62B | None = None,
) -> dict:
    sources = production_sources_v62b() if sources is None else sources
    prereg = _read_self_hashed_v62b(sources.preregistration)
    audit = _read_self_hashed_v62b(sources.support_audit)
    static = _verify_static_sources_v62b(prereg, audit, sources)
    evidence = _read_self_hashed_v62b(sources.evidence)
    stored_analysis = _read_self_hashed_v62b(sources.analysis)
    report = _read_self_hashed_v62b(sources.report)
    leakage = {
        name: _verify_no_text_keys_v62b(name, value)
        for name, value in (
            ("preregistration", prereg),
            ("support_audit", audit),
            ("evidence", evidence),
            ("analysis", stored_analysis),
            ("report", report),
        )
    }
    rebuilt = _verify_analysis_rebuild_v62b(evidence, stored_analysis)
    actors = _verify_actor_identities_v62b(report, prereg)
    states = _verify_state_v62b(evidence, report, prereg)
    schedule = _verify_fixed_schedule_v62b(
        evidence, report, prereg, rebuilt
    )
    report_check = _verify_report_v62b(report, rebuilt, sources)
    gpu = _gpu_summary_v62b(
        sources.gpu_log_path, sources.gpu_log_file_sha256, report
    )
    gate = _gate_observation_v62b(rebuilt["required_pre_hpo_gate"])
    value = {
        "schema": "v62b-pre-hpo-alpha-zero-independent-finalizer",
        "status": "complete_numeric_only_eligibility_observed_hpo_unauthorized",
        "source_hashes": {
            name: {
                "file_sha256": source.file_sha256,
                "content_sha256": source.content_sha256,
            }
            for name, source in (
                ("preregistration", sources.preregistration),
                ("support_audit", sources.support_audit),
                ("evidence", sources.evidence),
                ("analysis", sources.analysis),
                ("report", sources.report),
            )
        },
        "gpu_log_file_sha256": sources.gpu_log_file_sha256,
        "verification": {
            "all_file_and_self_hashes_verified": True,
            "static_preregistration_support_method_and_code_chain": static,
            "stored_analysis_exactly_equals_independent_numeric_rebuild": True,
            "no_text_leakage": leakage,
            "actor_runtime_identities": actors,
            "v434_state_receipts": states,
            "fixed_complete_warmup_and_scored_schedule": schedule,
            "gpu_activity_recomputed_from_numeric_log": gpu,
            "report_hash_chain_cleanup_and_idle": report_check,
        },
        "observed_numeric_outcome_without_authorization": {
            "primary_generated_f1": copy.deepcopy(
                rebuilt["primary_generated_f1"]
            ),
            "actor_influence": copy.deepcopy(rebuilt["actor_influence"]),
            "required_pre_hpo_gate": gate,
            "exact_sentinel_diagnostics": copy.deepcopy(
                rebuilt["exact_sentinel_diagnostics"]
            ),
            "performance": _performance_v62b(report),
        },
        "calibration_eligibility_observation": {
            "eligible_for_later_separately_preregistered_hpo_work": gate[
                "passed"
            ],
            "passed_gate_count": gate["passed_gate_count"],
            "failed_gate_count": gate["failed_gate_count"],
            "eligibility_is_not_launch_or_update_authority": True,
            "hpo_population_launch_or_update_authorized": False,
            "protected_access_authorized": False,
        },
        "frozen_non_authorization": {
            "outcome_assumed_before_read": False,
            "finalizer_accepts_and_records_either_gate_outcome": True,
            "thresholds_changed_after_outcome": False,
            "failed_gate_reinterpreted_or_relaxed": False,
            "hpo_population_launch_or_update_authorized": False,
            "candidate_materialization_or_promotion_authorized": False,
            "holdback_ood_shadow_terminal_or_protected_access_authorized": False,
            "gpu_or_model_launch_authorized": False,
        },
        "implementation_bindings": {
            "finalizer_file_sha256": file_sha256_v62b(
                Path(__file__).resolve()
            ),
            "tests_file_sha256": file_sha256_v62b(TESTS),
        },
        "raw_question_answer_prediction_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v62b(value)
    )
    return value


def _exclusive_write_v62b(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_finalized_v62b()
    _exclusive_write_v62b(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v62b(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "eligible_for_later_separately_preregistered_hpo_work": value[
            "calibration_eligibility_observation"
        ]["eligible_for_later_separately_preregistered_hpo_work"],
        "hpo_authorized": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
