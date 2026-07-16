#!/usr/bin/env python3
"""Outcome-agnostic numeric verifier for the sealed V61E calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import audit_vllm_fullbatch_fcfs_support_v61e as support_audit
import lora_es_fullbatch_fcfs_calibration_v61e as analysis
import run_lora_es_fullbatch_fcfs_calibration_v61e as runtime


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61e_v434_fullbatch_fcfs_paired_evaluator_calibration"
).resolve()
EVIDENCE = (RUN_DIR / "fullbatch_fcfs_null_evidence_v61e.json").resolve()
ANALYSIS = (RUN_DIR / "fullbatch_fcfs_null_analysis_v61e.json").resolve()
REPORT = (RUN_DIR / "fullbatch_fcfs_null_report_v61e.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v61e.jsonl").resolve()
OUTPUT = (RUN_DIR / "fullbatch_fcfs_finalized_v61e.json").resolve()
TESTS = (ROOT / "test_finalize_lora_es_fullbatch_fcfs_evidence_v61e.py").resolve()

V61C_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61c_v434_identical_state_paired_evaluator_calibration"
).resolve()
V61D_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61d_v434_singleton_fcfs_paired_evaluator_calibration"
).resolve()
V61C_FINALIZED = (V61C_DIR / "paired_null_finalized_v61c.json").resolve()
V61D_FINALIZED = (V61D_DIR / "singleton_fcfs_finalized_v61d.json").resolve()

PREREGISTRATION_FILE_SHA256 = (
    "1d784aed4b7734645ea6577d0ccef10b7428b43adf35eb2a5211a418afa9c764"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "d7fba56b19234d6e3e3242e63685049238bc75098224c297b4f7c64148475a71"
)
SUPPORT_AUDIT_FILE_SHA256 = runtime.AUDIT_FILE_SHA256
SUPPORT_AUDIT_CONTENT_SHA256 = runtime.AUDIT_CONTENT_SHA256

# Sealed only after the parent confirmed V61E exit 0 and four-GPU cleanup.
EVIDENCE_FILE_SHA256: str | None = (
    "8cd25a3f2f94175dba174199fca665209cbc7d98959ce26df7856cdd4e79507f"
)
EVIDENCE_CONTENT_SHA256: str | None = (
    "db145f887a90cd9383bef0d7caaeb22839e34af946c06c547ddc7c79fb564a66"
)
ANALYSIS_FILE_SHA256: str | None = (
    "37a97f560f9a622592bc387e7bd2d2220a6349c5d53ce6ba7361d7dd0aa7347c"
)
ANALYSIS_CONTENT_SHA256: str | None = (
    "33cc28c8b466fbce4895645aa944c15bfc37991284b1a61afa84202fe1ab1b1f"
)
REPORT_FILE_SHA256: str | None = (
    "5b88eef0f5044f915d735f21d81b429cb6b750638d5db87bfc698a23502fa5a0"
)
REPORT_CONTENT_SHA256: str | None = (
    "1ca436b1b6bcc6be6a96d3ce2bc4eccca40b7c789f31e656d62720505a182b7f"
)
GPU_LOG_FILE_SHA256: str | None = (
    "256ca30883c539ceb4417aaab429c60a5ced926c5833eb2c98a7bacb83547af9"
)


@dataclass(frozen=True)
class SelfHashedSourceV61E:
    path: Path
    file_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class FinalizerSourcesV61E:
    preregistration: SelfHashedSourceV61E
    support_audit: SelfHashedSourceV61E
    evidence: SelfHashedSourceV61E
    analysis: SelfHashedSourceV61E
    report: SelfHashedSourceV61E
    gpu_log_path: Path
    gpu_log_file_sha256: str
    v61c_finalized: SelfHashedSourceV61E
    v61d_finalized: SelfHashedSourceV61E


def file_sha256_v61e(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256_v61e(value: str | None, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v61e production outcome hash is not sealed: {name}")
    return value


def production_sources_v61e() -> FinalizerSourcesV61E:
    values = {
        "evidence_file": _require_sha256_v61e(
            EVIDENCE_FILE_SHA256, "EVIDENCE_FILE_SHA256"
        ),
        "evidence_content": _require_sha256_v61e(
            EVIDENCE_CONTENT_SHA256, "EVIDENCE_CONTENT_SHA256"
        ),
        "analysis_file": _require_sha256_v61e(
            ANALYSIS_FILE_SHA256, "ANALYSIS_FILE_SHA256"
        ),
        "analysis_content": _require_sha256_v61e(
            ANALYSIS_CONTENT_SHA256, "ANALYSIS_CONTENT_SHA256"
        ),
        "report_file": _require_sha256_v61e(
            REPORT_FILE_SHA256, "REPORT_FILE_SHA256"
        ),
        "report_content": _require_sha256_v61e(
            REPORT_CONTENT_SHA256, "REPORT_CONTENT_SHA256"
        ),
        "gpu_file": _require_sha256_v61e(
            GPU_LOG_FILE_SHA256, "GPU_LOG_FILE_SHA256"
        ),
    }
    matched = runtime.MATCHED_SOURCE_IDENTITIES_V61E
    return FinalizerSourcesV61E(
        preregistration=SelfHashedSourceV61E(
            runtime.PREREGISTRATION,
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_CONTENT_SHA256,
        ),
        support_audit=SelfHashedSourceV61E(
            runtime.AUDIT,
            SUPPORT_AUDIT_FILE_SHA256,
            SUPPORT_AUDIT_CONTENT_SHA256,
        ),
        evidence=SelfHashedSourceV61E(
            EVIDENCE, values["evidence_file"], values["evidence_content"]
        ),
        analysis=SelfHashedSourceV61E(
            ANALYSIS, values["analysis_file"], values["analysis_content"]
        ),
        report=SelfHashedSourceV61E(
            REPORT, values["report_file"], values["report_content"]
        ),
        gpu_log_path=GPU_LOG,
        gpu_log_file_sha256=values["gpu_file"],
        v61c_finalized=SelfHashedSourceV61E(
            V61C_FINALIZED,
            matched["v61c_finalizer"]["file_sha256"],
            matched["v61c_finalizer"]["content_sha256"],
        ),
        v61d_finalized=SelfHashedSourceV61E(
            V61D_FINALIZED,
            matched["v61d_finalizer"]["file_sha256"],
            matched["v61d_finalizer"]["content_sha256"],
        ),
    )


def _read_self_hashed_v61e(source: SelfHashedSourceV61E) -> dict:
    if file_sha256_v61e(source.path) != source.file_sha256:
        raise RuntimeError(f"v61e finalizer input file changed: {source.path}")
    value = json.loads(source.path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != source.content_sha256
        or analysis.canonical_sha256_v61e(compact) != source.content_sha256
    ):
        raise RuntimeError(f"v61e finalizer input content changed: {source.path}")
    return value


def _verify_static_sources_v61e(prereg: dict, audit: dict, sources) -> None:
    matched = prereg.get("matched_numeric_sources", {})
    if (
        prereg.get("schema")
        != "v61e-v434-fullbatch-fcfs-paired-evaluator-preregistration"
        or prereg.get("status")
        != "preregistered_before_v61e_train_semantics_model_or_gpu_access"
        or prereg.get("selection_hpo_update_or_promotion_authorized") is not False
        or prereg.get("eval_ood_shadow_or_holdout_access_authorized") is not False
        or prereg.get("implementation_bindings")
        != runtime.implementation_bindings_v61e()
        or prereg.get("installed_vllm_support_audit", {}).get("file_sha256")
        != sources.support_audit.file_sha256
        or prereg.get("installed_vllm_support_audit", {}).get("content_sha256")
        != sources.support_audit.content_sha256
        or matched != runtime.MATCHED_SOURCE_IDENTITIES_V61E
        or matched.get("v61c_finalizer", {}).get("file_sha256")
        != sources.v61c_finalized.file_sha256
        or matched.get("v61c_finalizer", {}).get("content_sha256")
        != sources.v61c_finalized.content_sha256
        or matched.get("v61d_finalizer", {}).get("file_sha256")
        != sources.v61d_finalized.file_sha256
        or matched.get("v61d_finalizer", {}).get("content_sha256")
        != sources.v61d_finalized.content_sha256
    ):
        raise RuntimeError("v61e preregistration, code, or prior-source binding changed")
    observed = {
        key: file_sha256_v61e(path)
        for key, path in support_audit.SOURCE_PATHS.items()
    }
    if (
        audit.get("schema")
        != "v61e-installed-vllm-fullbatch-fcfs-support-audit"
        or audit.get("status") != "supported"
        or audit.get("source_file_sha256") != observed
        or audit.get("requested_runtime_controls")
        != analysis.RUNTIME_CONTROLS_V61E
        or audit.get("fullbatch_fcfs_controls_supported") is not True
        or audit.get("batch_invariant_environment_resolved_false") is not True
        or audit.get("effective_request_batch_size") != 68
        or audit.get("gpu_model_or_train_semantics_accessed") is not False
    ):
        raise RuntimeError("v61e support audit binding changed")


def _verify_actor_identities_v61e(report: dict, prereg: dict) -> dict:
    actors = report.get("actor_identities", [])
    expected_tuned = prereg["runtime"]["tuned_table_content_sha256"]
    expected_folder = prereg["runtime"]["tuned_folder"]
    by_gpu = {}
    pids = set()
    for item in actors:
        gpu = item.get("physical_gpu_id")
        pid = item.get("pid")
        projection = {
            key: item.get(key) for key in analysis.RUNTIME_CONTROLS_V61E
        }
        if (
            gpu not in (0, 1, 2, 3)
            or not isinstance(pid, int)
            or pid <= 0
            or gpu in by_gpu
            or pid in pids
            or item.get("schema") != "fullbatch-fcfs-actor-identity-v61e"
            or item.get("cuda_visible_devices") != str(gpu)
            or item.get("cuda_current_device") != 0
            or projection != analysis.RUNTIME_CONTROLS_V61E
            or item.get("scheduler_class") != "Scheduler"
            or item.get("fullbatch_active_sequence_limit") != 68
            or item.get("effective_request_batch_size") != 68
            or item.get("global_batch_invariance_claimed") is not False
            or item.get("tuned_folder") != expected_folder
            or item.get("tuned_table_content_sha256") != expected_tuned
        ):
            raise RuntimeError("v61e actor runtime identity changed")
        by_gpu[gpu] = pid
        pids.add(pid)
    if len(actors) != 4 or set(by_gpu) != {0, 1, 2, 3}:
        raise RuntimeError("v61e actor GPU coverage changed")
    return {
        "actors": 4,
        "unique_processes": 4,
        "physical_gpu_ids": [0, 1, 2, 3],
        "all_runtime_controls_exact": True,
        "all_v27c_tuned_table_identities_exact": True,
    }


def _verify_state_v61e(evidence: dict, report: dict, prereg: dict) -> dict:
    receipts = evidence.get("state_receipts", [])
    recipe = prereg["fixed_calibration_recipe"]
    master = recipe["canonical_fp32_master_sha256"]
    runtime_sha = recipe["bf16_runtime_values_sha256"]
    if (
        len(receipts) != 4
        or [item.get("period_index") for item in receipts] != [0, 1, 2, 3]
        or any(
            item.get("identical_v434_state") is not True
            or item.get("before") != item.get("after")
            or item.get("before", {}).get("canonical_fp32_master_sha256")
            != master
            or item.get("before", {}).get("bf16_runtime_values_sha256")
            != runtime_sha
            for item in receipts
        )
        or evidence.get("numeric_state_receipts_sha256")
        != analysis.canonical_sha256_v61e(receipts)
        or report.get("state_receipts_sha256")
        != evidence.get("numeric_state_receipts_sha256")
        or report.get("master_state_receipt") != receipts[0].get("before")
    ):
        raise RuntimeError("v61e identical V434 state receipts changed")
    return {
        "periods": 4,
        "identical_before_after_periods": 4,
        "canonical_fp32_master_sha256": master,
        "bf16_runtime_values_sha256": runtime_sha,
    }


def _gpu_summary_v61e(path: Path, expected_sha: str, report: dict) -> dict:
    if file_sha256_v61e(path) != expected_sha:
        raise RuntimeError("v61e GPU log changed")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    actor_pids = {
        int(item["physical_gpu_id"]): int(item["pid"])
        for item in report.get("actor_identities", [])
    }
    if set(actor_pids) != {0, 1, 2, 3} or len(set(actor_pids.values())) != 4:
        raise RuntimeError("v61e GPU actor map changed")
    by_gpu = {}
    for gpu in range(4):
        expected_pid = actor_pids[gpu]
        selected = [item for item in rows if item.get("gpu") == gpu]
        if (
            not selected
            or any(item.get("expected_pid") != expected_pid for item in selected)
            or any(item.get("foreign_compute_pids") != [] for item in selected)
            or any(
                any(pid != expected_pid for pid in item.get("compute_pids", []))
                for item in selected
            )
        ):
            raise RuntimeError("v61e foreign or mismatched GPU process observed")
        resident = [item for item in selected if expected_pid in item["compute_pids"]]
        if not resident or not any(
            item["utilization_percent"] > 0 for item in resident
        ):
            raise RuntimeError("v61e GPU lacked attributed positive activity")
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
        raise RuntimeError("v61e reported GPU summary differs from numeric log")
    return {
        **rebuilt,
        "gpu_log_rows": len(rows),
        "foreign_compute_process_observations": 0,
    }


def _verify_report_v61e(report: dict, prereg: dict, sources) -> dict:
    cleanup = report.get("cleanup", {})
    after = cleanup.get("after", [])
    recipe = prereg["fixed_calibration_recipe"]
    if (
        report.get("schema") != "v61e-fullbatch-fcfs-paired-evaluator-report"
        or report.get("status")
        != "complete_matched_content_free_characterization_sealed"
        or report.get("preregistration_file_sha256")
        != sources.preregistration.file_sha256
        or report.get("preregistration_content_sha256")
        != sources.preregistration.content_sha256
        or report.get("support_audit_file_sha256")
        != sources.support_audit.file_sha256
        or report.get("support_audit_content_sha256")
        != sources.support_audit.content_sha256
        or report.get("runtime_determinism_controls")
        != analysis.RUNTIME_CONTROLS_V61E
        or report.get("matched_numeric_sources")
        != runtime.MATCHED_SOURCE_IDENTITIES_V61E
        or report.get("matched_v61c_panel_labels_metrics_bootstrap_thresholds")
        is not True
        or report.get("v61c_effective_request_batch_size") != 68
        or report.get("v61c_thresholds_relaxed_or_changed") is not False
        or report.get("panel_file_sha256")
        != recipe["staged_panel_file_sha256"]
        or report.get("panel_content_sha256")
        != recipe["staged_panel_content_sha256"]
        or report.get("prior_numeric_evidence_opened") is not False
        or report.get("alpha") != 0.0
        or report.get("adapter_update_or_candidate_materialization_performed")
        is not False
        or report.get("holdback_ood_shadow_or_protected_opened") is not False
        or report.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or report.get("hpo_selection_update_or_promotion_authorized") is not False
        or cleanup.get("engine_kill_count") != 4
        or cleanup.get("placement_group_remove_count") != 4
        or cleanup.get("all_four_gcs_states_removed") is not True
        or len(after) != 4
        or any(item.get("state") != "REMOVED" for item in after)
        or report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
    ):
        raise RuntimeError("v61e sealed report or cleanup contract changed")
    return {
        "engines_killed": 4,
        "placement_groups_removed": 4,
        "all_four_final_gpu_compute_process_lists_empty": True,
    }


def _scalar_v61e(value) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("v61e non-finite comparison value")
    return result


def _three_way_v61e(v61e, v61d, v61c) -> dict:
    current = _scalar_v61e(v61e)
    singleton = _scalar_v61e(v61d)
    baseline = _scalar_v61e(v61c)
    return {
        "v61e": current,
        "v61d": singleton,
        "v61c": baseline,
        "v61e_minus_v61d": current - singleton,
        "v61e_minus_v61c": current - baseline,
        "v61e_over_v61d": None if singleton == 0.0 else current / singleton,
        "v61e_over_v61c": None if baseline == 0.0 else current / baseline,
    }


def _ranking_comparison_v61e(rebuilt: dict, prior_c: dict, prior_d: dict) -> dict:
    current = rebuilt["ranking_bootstrap"]
    current_intervals = current["primary_conflict_unit_cluster_bootstrap"][
        "intervals"
    ]
    c_null = prior_c["ranking_primary_null"]
    d_comparison = prior_d[
        "ranking_point_and_primary_ci_halfwidth_comparison"
    ]
    result = {}
    for metric in (
        "generated_f1_delta",
        "generated_exact_delta",
        "generated_nonzero_delta",
        "teacher_forced_logprob_delta",
    ):
        result[metric] = {
            "point": _three_way_v61e(
                current["point"][metric],
                d_comparison[metric]["point"]["v61d"],
                c_null["point"][metric],
            ),
            "primary_ci_halfwidth": _three_way_v61e(
                current_intervals[metric]["halfwidth"],
                d_comparison[metric]["primary_ci_halfwidth"]["v61d"],
                c_null["raw_primary_cluster_intervals"][metric]["halfwidth"],
            ),
            "v61e_interval_contains_zero": current_intervals[metric][
                "contains_zero"
            ],
        }
    return result


def _repeat_summary_v61e(items: list[dict]) -> dict:
    if len(items) != 8:
        raise RuntimeError("v61e same-label repeat coverage changed")
    result = {}
    for metric, prefix in (
        ("generated_f1", "generated_f1"),
        ("teacher_logprob", "teacher_logprob"),
    ):
        means = [float(item[f"{prefix}_mean_absolute_delta"]) for item in items]
        maxima = [
            float(item[f"{prefix}_maximum_absolute_delta"]) for item in items
        ]
        if not np.isfinite(means + maxima).all():
            raise RuntimeError("v61e same-label repeat values changed")
        result[metric] = {
            "mean_of_actor_label_mean_absolute_deltas": float(np.mean(means)),
            "maximum_actor_label_mean_absolute_delta": max(means),
            "maximum_individual_absolute_delta": max(maxima),
        }
    return result


def _repeat_comparison_v61e(rebuilt: dict, prior_c: dict, prior_d: dict) -> dict:
    current = _repeat_summary_v61e(rebuilt["within_actor_same_label_repeat"])
    baseline = _repeat_summary_v61e(
        prior_c["ranking_metric_breakdown"]["within_actor_same_label_repeat"]
    )
    d_values = prior_d["same_label_repeat_drift_comparison"]
    result = {}
    for metric in ("generated_f1", "teacher_logprob"):
        result[metric] = {
            key: _three_way_v61e(
                current[metric][key], d_values[metric][key]["v61d"],
                baseline[metric][key],
            )
            for key in current[metric]
        }
    return result


def _sentinel_summary_v61e(evidence: dict, rebuilt: dict) -> dict:
    rows = analysis.validate_evidence_v61e(evidence)
    generation, teacher = analysis.v61c._metric_arrays_v61c(rows)
    delta, _, _ = analysis.v61c._paired_deltas_v61c(
        generation[64:], teacher[64:]
    )
    locations = np.argwhere(delta[..., 1] != 0.0)
    counts = Counter(int(location[0]) for location in locations)
    flips = int(len(locations))
    maximum = max(counts.values(), default=0)
    if (
        flips != rebuilt["exact_sentinel"][
            "nonzero_individual_paired_exact_delta_count"
        ]
        or rebuilt["exact_sentinel"]["passed"] != (flips == 0)
    ):
        raise RuntimeError("v61e exact-sentinel reconstruction changed")
    return {
        "passed": rebuilt["exact_sentinel"]["passed"],
        "individual_exact_flip_count": flips,
        "affected_unit_count": len(counts),
        "maximum_flips_in_one_unit": maximum,
        "fraction_of_flips_in_most_affected_unit": (
            0.0 if flips == 0 else maximum / flips
        ),
        "flip_count_histogram_over_affected_units": {
            str(key): value for key, value in sorted(Counter(counts.values()).items())
        },
        "flip_count_by_unit": {
            rows[64 + unit]["unit_identity_sha256"]: count
            for unit, count in sorted(counts.items())
        },
        "maximum_absolute_individual_exact_delta": float(
            np.max(np.abs(delta[..., 1]))
        ),
        "total_sentinel_units": 4,
    }


def _sentinel_comparison_v61e(current: dict, prior_c: dict, prior_d: dict) -> dict:
    c = prior_c["exact_sentinel"]
    d = prior_d["exact_sentinel_flip_and_concentration_comparison"]["v61d"]
    c_units = set(c["flip_count_by_unit"])
    current_units = set(current["flip_count_by_unit"])
    return {
        "v61e": current,
        "v61d": d,
        "v61c": {
            "passed": c["passed"],
            "individual_exact_flip_count": c["individual_exact_flip_count"],
            "affected_unit_count": c["units_with_any_exact_flip"],
            "maximum_flips_in_one_unit": max(
                c["flip_count_by_unit"].values(), default=0
            ),
            "maximum_absolute_individual_exact_delta": c[
                "maximum_absolute_individual_exact_delta"
            ],
            "total_sentinel_units": c["total_sentinel_units"],
        },
        "individual_exact_flip_count": _three_way_v61e(
            current["individual_exact_flip_count"],
            d["individual_exact_flip_count"],
            c["individual_exact_flip_count"],
        ),
        "affected_unit_hash_overlap_with_v61c": sorted(current_units & c_units),
        "affected_unit_hash_overlap_count_with_v61c": len(
            current_units & c_units
        ),
    }


def _performance_v61e(report: dict) -> dict:
    wall = _scalar_v61e(report.get("wall_runtime_seconds", 0.0))
    counts = report.get("evidence", {})
    generation = counts.get("generation_completions")
    teacher = counts.get("teacher_forced_requests")
    if wall <= 0.0 or generation != 1088 or teacher != 1088:
        raise RuntimeError("v61e matched runtime or request counts changed")
    total = generation + teacher
    return {
        "wall_runtime_seconds": wall,
        "generation_completions": generation,
        "teacher_forced_requests": teacher,
        "request_units": total,
        "request_units_per_second": total / wall,
    }


def _performance_comparison_v61e(report: dict, prior_d: dict) -> dict:
    current = _performance_v61e(report)
    previous = prior_d["wall_runtime_and_request_throughput_comparison"]
    singleton = previous["v61d"]
    baseline = previous["v61c"]
    return {
        "v61e": current,
        "v61d": singleton,
        "v61c": baseline,
        "wall_runtime_seconds": _three_way_v61e(
            current["wall_runtime_seconds"],
            singleton["wall_runtime_seconds"],
            baseline["wall_runtime_seconds"],
        ),
        "request_units_per_second": _three_way_v61e(
            current["request_units_per_second"],
            singleton["request_units_per_second"],
            baseline["request_units_per_second"],
        ),
        "matched_request_counts": True,
    }


def build_finalized_v61e(sources: FinalizerSourcesV61E | None = None) -> dict:
    sources = production_sources_v61e() if sources is None else sources
    prereg = _read_self_hashed_v61e(sources.preregistration)
    audit = _read_self_hashed_v61e(sources.support_audit)
    _verify_static_sources_v61e(prereg, audit, sources)
    evidence = _read_self_hashed_v61e(sources.evidence)
    stored_analysis = _read_self_hashed_v61e(sources.analysis)
    report = _read_self_hashed_v61e(sources.report)
    prior_c = _read_self_hashed_v61e(sources.v61c_finalized)
    prior_d = _read_self_hashed_v61e(sources.v61d_finalized)
    rebuilt = analysis.build_analysis_v61e(evidence)
    if rebuilt != stored_analysis:
        raise RuntimeError("v61e stored analysis differs from independent rebuild")
    matched = prereg["matched_numeric_sources"]
    if (
        report.get("evidence", {}).get("file_sha256")
        != sources.evidence.file_sha256
        or report.get("evidence", {}).get("content_sha256")
        != sources.evidence.content_sha256
        or report.get("analysis", {}).get("file_sha256")
        != sources.analysis.file_sha256
        or report.get("analysis", {}).get("content_sha256")
        != sources.analysis.content_sha256
        or report.get("gpu_log_file_sha256") != sources.gpu_log_file_sha256
        or prior_c.get("source_hashes", {}).get("evidence")
        != matched["v61c_evidence"]
        or prior_d.get("source_hashes", {}).get("evidence")
        != matched["v61d_evidence"]
    ):
        raise RuntimeError("v61e outcome or prior evidence hash chain changed")
    actors = _verify_actor_identities_v61e(report, prereg)
    states = _verify_state_v61e(evidence, report, prereg)
    cleanup = _verify_report_v61e(report, prereg, sources)
    gpu = _gpu_summary_v61e(
        sources.gpu_log_path, sources.gpu_log_file_sha256, report
    )
    sentinel = _sentinel_summary_v61e(evidence, rebuilt)
    value = {
        "schema": "v61e-fullbatch-fcfs-independent-finalizer",
        "status": "complete_numeric_only_matched_characterization_hpo_unauthorized",
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
                ("v61c_finalized", sources.v61c_finalized),
                ("v61d_finalized", sources.v61d_finalized),
            )
        },
        "gpu_log_file_sha256": sources.gpu_log_file_sha256,
        "verification": {
            "all_file_and_self_hashes_verified": True,
            "preregistration_support_code_and_prior_hash_chain_verified": True,
            "stored_analysis_exactly_equals_independent_numeric_rebuild": True,
            "actor_runtime_identities": actors,
            "v434_state_receipts": states,
            "gpu_activity_recomputed_from_numeric_log": gpu,
            "cleanup": cleanup,
        },
        "ranking_point_and_primary_ci_halfwidth_comparison": (
            _ranking_comparison_v61e(rebuilt, prior_c, prior_d)
        ),
        "same_label_repeat_drift_comparison": _repeat_comparison_v61e(
            rebuilt, prior_c, prior_d
        ),
        "exact_sentinel_flip_and_concentration_comparison": (
            _sentinel_comparison_v61e(sentinel, prior_c, prior_d)
        ),
        "wall_runtime_and_request_throughput_comparison": (
            _performance_comparison_v61e(report, prior_d)
        ),
        "observed_decision_inputs_without_authorization": {
            "v61e_teacher_forced_logprob_primary_eligible": rebuilt[
                "noise_scale_comparison"
            ]["teacher_forced_logprob_primary_eligible"],
            "v61e_exact_sentinel_passed": sentinel["passed"],
        },
        "frozen_non_authorization": {
            "outcome_assumed_before_read": False,
            "thresholds_changed_after_outcomes": False,
            "causal_source_of_nondeterminism_claimed": False,
            "hpo_update_selection_or_promotion_authorized": False,
            "holdback_ood_shadow_terminal_or_protected_access_authorized": False,
        },
        "implementation_bindings": {
            "finalizer_file_sha256": file_sha256_v61e(Path(__file__).resolve()),
            "tests_file_sha256": file_sha256_v61e(TESTS),
        },
        "raw_question_answer_prediction_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = analysis.canonical_sha256_v61e(
        value
    )
    return value


def _exclusive_write_v61e(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
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
    value = build_finalized_v61e()
    _exclusive_write_v61e(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v61e(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "hpo_authorized": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
