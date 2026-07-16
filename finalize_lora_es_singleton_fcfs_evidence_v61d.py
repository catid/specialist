#!/usr/bin/env python3
"""Independently verify numeric-only V61D evidence and compare it with V61C."""

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

import audit_vllm_singleton_fcfs_support_v61d as support_audit
import lora_es_singleton_fcfs_calibration_v61d as analysis
import run_lora_es_singleton_fcfs_calibration_v61d as runtime


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61d_v434_singleton_fcfs_paired_evaluator_calibration"
).resolve()
EVIDENCE = (RUN_DIR / "singleton_fcfs_null_evidence_v61d.json").resolve()
ANALYSIS = (RUN_DIR / "singleton_fcfs_null_analysis_v61d.json").resolve()
REPORT = (RUN_DIR / "singleton_fcfs_null_report_v61d.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v61d.jsonl").resolve()
OUTPUT = (RUN_DIR / "singleton_fcfs_finalized_v61d.json").resolve()
PREREGISTRATION = runtime.PREREGISTRATION
SUPPORT_AUDIT = runtime.AUDIT
V61C_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61c_v434_identical_state_paired_evaluator_calibration"
).resolve()
V61C_FINALIZED = (V61C_DIR / "paired_null_finalized_v61c.json").resolve()
V61C_REPORT = (V61C_DIR / "paired_null_report_v61c.json").resolve()
TESTS = (ROOT / "test_finalize_lora_es_singleton_fcfs_evidence_v61d.py").resolve()

PREREGISTRATION_FILE_SHA256 = (
    "cb30497f0d2d94deb0df63b82f6fde10c75c080b4864a28253caa190a4dc2084"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "96a55b09dddccebf6e78a784cc19b03ec8ef681c4e33ca3249190964d0598dfc"
)
SUPPORT_AUDIT_FILE_SHA256 = runtime.AUDIT_FILE_SHA256
SUPPORT_AUDIT_CONTENT_SHA256 = runtime.AUDIT_CONTENT_SHA256
V61C_FINALIZED_FILE_SHA256 = runtime.V61C_FINALIZER_FILE_SHA256
V61C_FINALIZED_CONTENT_SHA256 = runtime.V61C_FINALIZER_CONTENT_SHA256
V61C_REPORT_FILE_SHA256 = (
    "f50b665fde835a29f3ee928d808df7a83237fbac09050d1730b2974f9dca44a9"
)
V61C_REPORT_CONTENT_SHA256 = (
    "c2e2cfbf8d1cefbe15fbb122d5e5f6fb06b73b7b312b4e0ae6d36c4e8a16378b"
)

# Sealed only after the V61D runner completed and released all four GPUs.
EVIDENCE_FILE_SHA256: str | None = (
    "49be43e8a2e02093952bec7a0186f900fd64e3ec00057ece31e290a540c7044e"
)
EVIDENCE_CONTENT_SHA256: str | None = (
    "f07a24fcd5ae0cedf1703f1bf25a7e9b6ca3db900d4bd58cc7351a68ec795048"
)
ANALYSIS_FILE_SHA256: str | None = (
    "7d24774a5f21e08c8302aeab76b050bb48a4be87d4c370070a2c23f45d6e4a51"
)
ANALYSIS_CONTENT_SHA256: str | None = (
    "9b15b866940f8845aca0b25f0a3f1b90ad3d883c0294c35f37fa17deb959fc7b"
)
REPORT_FILE_SHA256: str | None = (
    "bf20128420930f2f1a19b0d60e81bb0233d57c10014fb098a7d2eccefebaf059"
)
REPORT_CONTENT_SHA256: str | None = (
    "3805e4656c8d60c220e13995c356dfd856203d268372b3521913b64e130d3935"
)
GPU_LOG_FILE_SHA256: str | None = (
    "8565da9f311310151f8c051f0df3d146e13f904db246fb2c199b7c0fa32c95fc"
)


@dataclass(frozen=True)
class SelfHashedSourceV61D:
    path: Path
    file_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class FinalizerSourcesV61D:
    preregistration: SelfHashedSourceV61D
    support_audit: SelfHashedSourceV61D
    evidence: SelfHashedSourceV61D
    analysis: SelfHashedSourceV61D
    report: SelfHashedSourceV61D
    gpu_log_path: Path
    gpu_log_file_sha256: str
    v61c_finalized: SelfHashedSourceV61D
    v61c_report: SelfHashedSourceV61D


def file_sha256_v61d(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256_v61d(value: str | None, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v61d production outcome hash is not sealed: {name}")
    return value


def production_sources_v61d() -> FinalizerSourcesV61D:
    """Return production bindings, failing before any live-output file access."""
    evidence_file = _require_sha256_v61d(
        EVIDENCE_FILE_SHA256, "EVIDENCE_FILE_SHA256"
    )
    evidence_content = _require_sha256_v61d(
        EVIDENCE_CONTENT_SHA256, "EVIDENCE_CONTENT_SHA256"
    )
    analysis_file = _require_sha256_v61d(
        ANALYSIS_FILE_SHA256, "ANALYSIS_FILE_SHA256"
    )
    analysis_content = _require_sha256_v61d(
        ANALYSIS_CONTENT_SHA256, "ANALYSIS_CONTENT_SHA256"
    )
    report_file = _require_sha256_v61d(REPORT_FILE_SHA256, "REPORT_FILE_SHA256")
    report_content = _require_sha256_v61d(
        REPORT_CONTENT_SHA256, "REPORT_CONTENT_SHA256"
    )
    gpu_file = _require_sha256_v61d(
        GPU_LOG_FILE_SHA256, "GPU_LOG_FILE_SHA256"
    )
    return FinalizerSourcesV61D(
        preregistration=SelfHashedSourceV61D(
            PREREGISTRATION,
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_CONTENT_SHA256,
        ),
        support_audit=SelfHashedSourceV61D(
            SUPPORT_AUDIT,
            SUPPORT_AUDIT_FILE_SHA256,
            SUPPORT_AUDIT_CONTENT_SHA256,
        ),
        evidence=SelfHashedSourceV61D(EVIDENCE, evidence_file, evidence_content),
        analysis=SelfHashedSourceV61D(ANALYSIS, analysis_file, analysis_content),
        report=SelfHashedSourceV61D(REPORT, report_file, report_content),
        gpu_log_path=GPU_LOG,
        gpu_log_file_sha256=gpu_file,
        v61c_finalized=SelfHashedSourceV61D(
            V61C_FINALIZED,
            V61C_FINALIZED_FILE_SHA256,
            V61C_FINALIZED_CONTENT_SHA256,
        ),
        v61c_report=SelfHashedSourceV61D(
            V61C_REPORT, V61C_REPORT_FILE_SHA256, V61C_REPORT_CONTENT_SHA256
        ),
    )


def _read_self_hashed_v61d(source: SelfHashedSourceV61D) -> dict:
    if file_sha256_v61d(source.path) != source.file_sha256:
        raise RuntimeError(f"v61d finalizer input file changed: {source.path}")
    value = json.loads(source.path.read_text(encoding="utf-8"))
    compact = {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != source.content_sha256
        or analysis.canonical_sha256_v61d(compact) != source.content_sha256
    ):
        raise RuntimeError(f"v61d finalizer input content changed: {source.path}")
    return value


def _verify_static_bindings_v61d(prereg: dict, audit: dict, sources) -> None:
    if (
        prereg.get("schema")
        != "v61d-v434-singleton-fcfs-paired-evaluator-preregistration"
        or prereg.get("status")
        != "preregistered_before_v61d_train_semantics_model_or_gpu_access"
        or prereg.get("selection_hpo_update_or_promotion_authorized") is not False
        or prereg.get("eval_ood_shadow_or_holdout_access_authorized") is not False
        or prereg.get("implementation_bindings")
        != runtime.implementation_bindings_v61d()
        or prereg.get("installed_vllm_support_audit", {}).get("file_sha256")
        != sources.support_audit.file_sha256
        or prereg.get("installed_vllm_support_audit", {}).get("content_sha256")
        != sources.support_audit.content_sha256
    ):
        raise RuntimeError("v61d preregistration or code bindings changed")
    observed_sources = {
        key: file_sha256_v61d(path)
        for key, path in support_audit.SOURCE_PATHS.items()
    }
    if (
        audit.get("schema")
        != "v61d-installed-vllm-singleton-fcfs-support-audit"
        or audit.get("status") != "supported"
        or audit.get("source_file_sha256") != observed_sources
        or audit.get("requested_runtime_controls")
        != analysis.RUNTIME_CONTROLS_V61D
        or audit.get("singleton_fcfs_controls_supported") is not True
        or audit.get("batch_invariant_environment_resolved_false") is not True
        or audit.get("global_batch_invariance_claimed") is not False
        or audit.get("gpu_model_or_train_semantics_accessed") is not False
    ):
        raise RuntimeError("v61d installed-vLLM support binding changed")


def _verify_actor_identities_v61d(report: dict, prereg: dict) -> dict:
    actors = report.get("actor_identities", [])
    expected_tuned = prereg.get("runtime", {}).get("tuned_table_content_sha256")
    expected_folder = prereg.get("runtime", {}).get("tuned_folder")
    if len(actors) != 4:
        raise RuntimeError("v61d actor identity count changed")
    pids = set()
    by_gpu = {}
    for item in actors:
        gpu = item.get("physical_gpu_id")
        pid = item.get("pid")
        projection = {
            key: item.get(key) for key in analysis.RUNTIME_CONTROLS_V61D
        }
        if (
            gpu not in (0, 1, 2, 3)
            or not isinstance(pid, int)
            or pid <= 0
            or gpu in by_gpu
            or pid in pids
            or item.get("schema") != "singleton-fcfs-actor-identity-v61d"
            or item.get("cuda_visible_devices") != str(gpu)
            or item.get("cuda_current_device") != 0
            or projection != analysis.RUNTIME_CONTROLS_V61D
            or item.get("scheduler_class") != "Scheduler"
            or item.get("singleton_active_sequence_limit") != 1
            or item.get("global_batch_invariance_claimed") is not False
            or item.get("tuned_folder") != expected_folder
            or item.get("tuned_table_content_sha256") != expected_tuned
        ):
            raise RuntimeError("v61d actor runtime identity changed")
        pids.add(pid)
        by_gpu[gpu] = pid
    if set(by_gpu) != {0, 1, 2, 3}:
        raise RuntimeError("v61d actor GPU coverage changed")
    return {
        "actors": 4,
        "physical_gpu_ids": [0, 1, 2, 3],
        "unique_processes": 4,
        "all_runtime_controls_exact": True,
        "all_v27c_tuned_table_identities_exact": True,
    }


def _verify_v434_receipts_v61d(evidence: dict, report: dict, prereg: dict) -> dict:
    receipts = evidence.get("state_receipts", [])
    recipe = prereg.get("fixed_calibration_recipe", {})
    master = recipe.get("canonical_fp32_master_sha256")
    runtime_sha = recipe.get("bf16_runtime_values_sha256")
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
        != analysis.canonical_sha256_v61d(receipts)
        or report.get("state_receipts_sha256")
        != evidence.get("numeric_state_receipts_sha256")
        or report.get("master_state_receipt") != receipts[0].get("before")
    ):
        raise RuntimeError("v61d identical V434 state receipts changed")
    return {
        "periods": 4,
        "identical_before_after_periods": 4,
        "canonical_fp32_master_sha256": master,
        "bf16_runtime_values_sha256": runtime_sha,
    }


def _gpu_summary_v61d(path: Path, expected_sha: str, report: dict) -> dict:
    if file_sha256_v61d(path) != expected_sha:
        raise RuntimeError("v61d GPU log changed")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    actor_pids = {
        int(item["physical_gpu_id"]): int(item["pid"])
        for item in report.get("actor_identities", [])
    }
    if set(actor_pids) != {0, 1, 2, 3} or len(set(actor_pids.values())) != 4:
        raise RuntimeError("v61d GPU actor map changed")
    by_gpu = {}
    for gpu in range(4):
        selected = [item for item in rows if item.get("gpu") == gpu]
        expected_pid = actor_pids[gpu]
        if (
            not selected
            or any(item.get("expected_pid") != expected_pid for item in selected)
            or any(item.get("foreign_compute_pids") != [] for item in selected)
            or any(
                any(pid != expected_pid for pid in item.get("compute_pids", []))
                for item in selected
            )
        ):
            raise RuntimeError("v61d foreign or mismatched GPU process observed")
        resident = [item for item in selected if expected_pid in item["compute_pids"]]
        if not resident or not any(
            item["utilization_percent"] > 0 for item in resident
        ):
            raise RuntimeError("v61d GPU lacked attributed positive activity")
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
        raise RuntimeError("v61d reported GPU summary differs from numeric log")
    return {
        **rebuilt,
        "gpu_log_rows": len(rows),
        "foreign_compute_process_observations": 0,
    }


def _verify_cleanup_v61d(report: dict) -> dict:
    cleanup = report.get("cleanup", {})
    after = cleanup.get("after", [])
    if (
        cleanup.get("engine_kill_count") != 4
        or cleanup.get("placement_group_remove_count") != 4
        or cleanup.get("all_four_gcs_states_removed") is not True
        or len(after) != 4
        or any(item.get("state") != "REMOVED" for item in after)
        or report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        )
        is not True
    ):
        raise RuntimeError("v61d engine, placement-group, or GPU cleanup changed")
    return {
        "engines_killed": 4,
        "placement_groups_removed": 4,
        "all_four_final_gpu_compute_process_lists_empty": True,
    }


def _scalar_comparison_v61d(v61d: float, v61c: float) -> dict:
    current = float(v61d)
    baseline = float(v61c)
    return {
        "v61d": current,
        "v61c": baseline,
        "v61d_minus_v61c": current - baseline,
        "v61d_over_v61c": None if baseline == 0.0 else current / baseline,
        "both_zero": current == 0.0 and baseline == 0.0,
    }


def _ranking_comparison_v61d(rebuilt: dict, prior: dict) -> dict:
    current_bootstrap = rebuilt["ranking_bootstrap"]
    current_intervals = current_bootstrap[
        "primary_conflict_unit_cluster_bootstrap"
    ]["intervals"]
    prior_null = prior["ranking_primary_null"]
    prior_intervals = prior_null["raw_primary_cluster_intervals"]
    result = {}
    for metric in (
        "generated_f1_delta",
        "generated_exact_delta",
        "generated_nonzero_delta",
        "teacher_forced_logprob_delta",
    ):
        result[metric] = {
            "point": _scalar_comparison_v61d(
                current_bootstrap["point"][metric], prior_null["point"][metric]
            ),
            "primary_ci_halfwidth": _scalar_comparison_v61d(
                current_intervals[metric]["halfwidth"],
                prior_intervals[metric]["halfwidth"],
            ),
            "v61d_interval_contains_zero": current_intervals[metric][
                "contains_zero"
            ],
        }
    return result


def _repeat_summary_v61d(items: list[dict]) -> dict:
    if len(items) != 8:
        raise RuntimeError("same-label repeat coverage changed")
    result = {}
    for name, prefix in (
        ("generated_f1", "generated_f1"),
        ("teacher_logprob", "teacher_logprob"),
    ):
        means = [float(item[f"{prefix}_mean_absolute_delta"]) for item in items]
        maxima = [
            float(item[f"{prefix}_maximum_absolute_delta"]) for item in items
        ]
        if not np.isfinite(means + maxima).all():
            raise RuntimeError("same-label repeat values changed")
        result[name] = {
            "actor_label_repeats": 8,
            "mean_of_actor_label_mean_absolute_deltas": float(np.mean(means)),
            "maximum_actor_label_mean_absolute_delta": max(means),
            "maximum_individual_absolute_delta": max(maxima),
        }
    return result


def _repeat_comparison_v61d(rebuilt: dict, prior: dict) -> dict:
    current = _repeat_summary_v61d(rebuilt["within_actor_same_label_repeat"])
    baseline = _repeat_summary_v61d(
        prior["ranking_metric_breakdown"]["within_actor_same_label_repeat"]
    )
    result = {}
    for metric in ("generated_f1", "teacher_logprob"):
        result[metric] = {
            key: _scalar_comparison_v61d(current[metric][key], baseline[metric][key])
            for key in (
                "mean_of_actor_label_mean_absolute_deltas",
                "maximum_actor_label_mean_absolute_delta",
                "maximum_individual_absolute_delta",
            )
        }
    return result


def _sentinel_summary_v61d(evidence: dict, rebuilt: dict) -> dict:
    rows = analysis.validate_evidence_v61d(evidence)
    generation, teacher = analysis.v61c._metric_arrays_v61c(rows)
    sentinel_delta, _, _ = analysis.v61c._paired_deltas_v61c(
        generation[64:], teacher[64:]
    )
    locations = np.argwhere(sentinel_delta[..., 1] != 0.0)
    counts = Counter(int(location[0]) for location in locations)
    flips = int(len(locations))
    maximum = max(counts.values(), default=0)
    passed = rebuilt["exact_sentinel"]["passed"]
    if (
        flips
        != rebuilt["exact_sentinel"][
            "nonzero_individual_paired_exact_delta_count"
        ]
        or passed != (flips == 0)
    ):
        raise RuntimeError("v61d exact-sentinel reconstruction changed")
    return {
        "passed": passed,
        "individual_exact_flip_count": flips,
        "affected_unit_count": len(counts),
        "total_sentinel_units": 4,
        "maximum_flips_in_one_unit": maximum,
        "fraction_of_flips_in_most_affected_unit": (
            0.0 if flips == 0 else maximum / flips
        ),
        "flip_count_histogram_over_affected_units": {
            str(key): value for key, value in sorted(Counter(counts.values()).items())
        },
        "maximum_absolute_individual_exact_delta": float(
            np.max(np.abs(sentinel_delta[..., 1]))
        ),
    }


def _sentinel_comparison_v61d(current: dict, prior: dict) -> dict:
    baseline = {
        "passed": prior["passed"],
        "individual_exact_flip_count": prior["individual_exact_flip_count"],
        "affected_unit_count": prior["units_with_any_exact_flip"],
        "total_sentinel_units": prior["total_sentinel_units"],
        "maximum_flips_in_one_unit": max(
            prior["flip_count_by_unit"].values(), default=0
        ),
        "fraction_of_flips_in_most_affected_unit": (
            0.0
            if prior["individual_exact_flip_count"] == 0
            else max(prior["flip_count_by_unit"].values(), default=0)
            / prior["individual_exact_flip_count"]
        ),
        "flip_count_histogram_over_affected_units": {
            str(key): value
            for key, value in prior[
                "flip_count_histogram_over_affected_units"
            ].items()
        },
        "maximum_absolute_individual_exact_delta": prior[
            "maximum_absolute_individual_exact_delta"
        ],
    }
    return {
        "v61d": current,
        "v61c": baseline,
        "individual_exact_flip_count": _scalar_comparison_v61d(
            current["individual_exact_flip_count"],
            baseline["individual_exact_flip_count"],
        ),
        "affected_unit_count": _scalar_comparison_v61d(
            current["affected_unit_count"], baseline["affected_unit_count"]
        ),
        "maximum_flips_in_one_unit": _scalar_comparison_v61d(
            current["maximum_flips_in_one_unit"],
            baseline["maximum_flips_in_one_unit"],
        ),
    }


def _performance_summary_v61d(report: dict) -> dict:
    wall = float(report.get("wall_runtime_seconds", 0.0))
    counts = report.get("evidence", {})
    generation = counts.get("generation_completions")
    teacher = counts.get("teacher_forced_requests")
    if (
        not math.isfinite(wall)
        or wall <= 0.0
        or generation != 1088
        or teacher != 1088
    ):
        raise RuntimeError("matched runtime or request counts changed")
    total = generation + teacher
    return {
        "wall_runtime_seconds": wall,
        "generation_completions": generation,
        "teacher_forced_requests": teacher,
        "request_units": total,
        "generation_completions_per_second": generation / wall,
        "teacher_forced_requests_per_second": teacher / wall,
        "request_units_per_second": total / wall,
    }


def _performance_comparison_v61d(current_report: dict, prior_report: dict) -> dict:
    current = _performance_summary_v61d(current_report)
    baseline = _performance_summary_v61d(prior_report)
    return {
        "v61d": current,
        "v61c": baseline,
        "wall_runtime_seconds": _scalar_comparison_v61d(
            current["wall_runtime_seconds"], baseline["wall_runtime_seconds"]
        ),
        "request_units_per_second": _scalar_comparison_v61d(
            current["request_units_per_second"],
            baseline["request_units_per_second"],
        ),
        "matched_request_counts": True,
    }


def _verify_report_contract_v61d(report: dict, prereg: dict, sources) -> None:
    recipe = prereg["fixed_calibration_recipe"]
    matched = prereg["matched_v61c_contract"]
    if (
        report.get("schema") != "v61d-singleton-fcfs-paired-evaluator-report"
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
        != analysis.RUNTIME_CONTROLS_V61D
        or report.get("matched_v61c_panel_labels_metrics_bootstrap_thresholds")
        is not True
        or report.get("v61c_thresholds_relaxed_or_changed") is not False
        or report.get("panel_file_sha256")
        != recipe.get("staged_panel_file_sha256")
        or report.get("panel_content_sha256")
        != recipe.get("staged_panel_content_sha256")
        or report.get("matched_v61c_finalizer", {}).get("file_sha256")
        != matched.get("v61c_finalizer_file_sha256")
        or report.get("matched_v61c_finalizer", {}).get("content_sha256")
        != matched.get("v61c_finalizer_content_sha256")
        or report.get("alpha") != 0.0
        or report.get("adapter_update_or_candidate_materialization_performed")
        is not False
        or report.get("holdback_ood_shadow_or_protected_opened") is not False
        or report.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or report.get("hpo_selection_update_or_promotion_authorized") is not False
    ):
        raise RuntimeError("v61d sealed report contract changed")


def build_finalized_v61d(sources: FinalizerSourcesV61D | None = None) -> dict:
    sources = production_sources_v61d() if sources is None else sources
    prereg = _read_self_hashed_v61d(sources.preregistration)
    audit = _read_self_hashed_v61d(sources.support_audit)
    _verify_static_bindings_v61d(prereg, audit, sources)
    evidence = _read_self_hashed_v61d(sources.evidence)
    stored_analysis = _read_self_hashed_v61d(sources.analysis)
    report = _read_self_hashed_v61d(sources.report)
    prior = _read_self_hashed_v61d(sources.v61c_finalized)
    prior_report = _read_self_hashed_v61d(sources.v61c_report)
    rebuilt = analysis.build_analysis_v61d(evidence)
    if rebuilt != stored_analysis:
        raise RuntimeError("v61d stored analysis differs from independent rebuild")
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
        or prior.get("source_hashes", {}).get("report", {}).get("file_sha256")
        != sources.v61c_report.file_sha256
        or prior.get("source_hashes", {}).get("report", {}).get("content_sha256")
        != sources.v61c_report.content_sha256
        or prereg.get("matched_v61c_contract", {}).get(
            "v61c_finalizer_file_sha256"
        )
        != sources.v61c_finalized.file_sha256
        or prereg.get("matched_v61c_contract", {}).get(
            "v61c_finalizer_content_sha256"
        )
        != sources.v61c_finalized.content_sha256
    ):
        raise RuntimeError("v61d outcome or matched V61C source hashes changed")
    _verify_report_contract_v61d(report, prereg, sources)
    actors = _verify_actor_identities_v61d(report, prereg)
    states = _verify_v434_receipts_v61d(evidence, report, prereg)
    gpu = _gpu_summary_v61d(
        sources.gpu_log_path, sources.gpu_log_file_sha256, report
    )
    cleanup = _verify_cleanup_v61d(report)
    sentinel = _sentinel_summary_v61d(evidence, rebuilt)
    value = {
        "schema": "v61d-singleton-fcfs-independent-finalizer",
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
                ("v61c_report", sources.v61c_report),
            )
        },
        "gpu_log_file_sha256": sources.gpu_log_file_sha256,
        "verification": {
            "all_file_and_self_hashes_verified": True,
            "preregistration_support_and_code_bindings_verified": True,
            "stored_analysis_exactly_equals_independent_numeric_rebuild": True,
            "actor_runtime_identities": actors,
            "v434_state_receipts": states,
            "gpu_activity_recomputed_from_numeric_log": gpu,
            "cleanup": cleanup,
            "matched_v61c_source_verified": True,
        },
        "ranking_point_and_primary_ci_halfwidth_comparison": (
            _ranking_comparison_v61d(rebuilt, prior)
        ),
        "same_label_repeat_drift_comparison": _repeat_comparison_v61d(
            rebuilt, prior
        ),
        "exact_sentinel_flip_and_concentration_comparison": (
            _sentinel_comparison_v61d(sentinel, prior["exact_sentinel"])
        ),
        "wall_runtime_and_request_throughput_comparison": (
            _performance_comparison_v61d(report, prior_report)
        ),
        "frozen_decision_state": {
            "v61d_teacher_forced_logprob_primary_eligible": rebuilt[
                "noise_scale_comparison"
            ]["teacher_forced_logprob_primary_eligible"],
            "v61d_exact_sentinel_passed": sentinel["passed"],
            "thresholds_changed_after_outcomes": False,
            "causal_source_of_nondeterminism_claimed": False,
            "hpo_update_selection_or_promotion_authorized": False,
            "holdback_ood_shadow_terminal_or_protected_access_authorized": False,
        },
        "implementation_bindings": {
            "finalizer_file_sha256": file_sha256_v61d(Path(__file__).resolve()),
            "tests_file_sha256": file_sha256_v61d(TESTS),
        },
        "raw_question_answer_prediction_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = analysis.canonical_sha256_v61d(
        value
    )
    return value


def _exclusive_write_v61d(path: Path, payload: bytes) -> None:
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
    value = build_finalized_v61d()
    _exclusive_write_v61d(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(
        json.dumps(
            {
                "path": str(output),
                "file_sha256": file_sha256_v61d(output),
                "content_sha256": value["content_sha256_before_self_field"],
                "hpo_authorized": False,
                "protected_semantics_opened": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
