#!/usr/bin/env python3
"""Outcome-agnostic numeric finalizer for the sealed V62A live run."""

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

import numpy as np

import audit_vllm_pre_hpo_alpha_zero_support_v62a as support_audit
import lora_es_pre_hpo_alpha_zero_calibration_v62a as analysis
import run_lora_es_pre_hpo_alpha_zero_calibration_v62a as runtime


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v62a_v434_pre_hpo_alpha_zero_generation_calibration"
).resolve()
EVIDENCE = (RUN_DIR / "alpha_zero_evidence_v62a.json").resolve()
ANALYSIS = (RUN_DIR / "alpha_zero_analysis_v62a.json").resolve()
REPORT = (RUN_DIR / "alpha_zero_report_v62a.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v62a.jsonl").resolve()
OUTPUT = (RUN_DIR / "alpha_zero_finalized_v62a.json").resolve()
TESTS = (
    ROOT / "test_finalize_lora_es_pre_hpo_alpha_zero_evidence_v62a.py"
).resolve()

PREREGISTRATION_FILE_SHA256 = (
    "8d6e0914f2e52757a3fe4873bcdfbcc4dc5be55f45409325e4236125463c6992"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "c1cdd1900406823ba15769fe3fb10b574129d9d3467e3e0d32905723df3d0d5e"
)
SUPPORT_AUDIT_FILE_SHA256 = runtime.SUPPORT_AUDIT_FILE_SHA256
SUPPORT_AUDIT_CONTENT_SHA256 = runtime.SUPPORT_AUDIT_CONTENT_SHA256
EVIDENCE_FILE_SHA256: str | None = (
    "5663a96e9591e8cec81f3b64c477e06e1267aab6ef49721793193cb90875ed0b"
)
EVIDENCE_CONTENT_SHA256: str | None = (
    "23d2197f9372207daa9cf06dde53d88c23d3c4d4da3f5e2f1c1b4b77a97245f0"
)
ANALYSIS_FILE_SHA256: str | None = (
    "59028a2ba0f589c2a3e4d643a52484db515cc53ef55f234bc4c78cecab3c3a3d"
)
ANALYSIS_CONTENT_SHA256: str | None = (
    "5237d5fd6e865ea6abb60f96d59305a1162c644675278d09a2388cbe69a35656"
)
REPORT_FILE_SHA256: str | None = (
    "209fd26337d067097e30ae4ce4117a303a83a7f6cc21e3a49fdc7659a95487b2"
)
REPORT_CONTENT_SHA256: str | None = (
    "4166a7382ac33c170b1c6df1a2b8a2af15ceecdebad588fbd3fccd8f9ad6c302"
)
GPU_LOG_FILE_SHA256: str | None = (
    "5e5579dc4f1c924777ec5259330a88360241920e3bc08e52a70273d2d9e430d2"
)
FORBIDDEN_TEXT_KEYS_V62A = {
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
class SelfHashedSourceV62A:
    path: Path
    file_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class FinalizerSourcesV62A:
    preregistration: SelfHashedSourceV62A
    support_audit: SelfHashedSourceV62A
    evidence: SelfHashedSourceV62A
    analysis: SelfHashedSourceV62A
    report: SelfHashedSourceV62A
    gpu_log_path: Path
    gpu_log_file_sha256: str


def file_sha256_v62a(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256_v62a(value: str | None, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v62a production outcome hash is not sealed: {name}")
    return value


def production_sources_v62a() -> FinalizerSourcesV62A:
    return FinalizerSourcesV62A(
        preregistration=SelfHashedSourceV62A(
            runtime.PREREGISTRATION,
            PREREGISTRATION_FILE_SHA256,
            PREREGISTRATION_CONTENT_SHA256,
        ),
        support_audit=SelfHashedSourceV62A(
            runtime.SUPPORT_AUDIT,
            SUPPORT_AUDIT_FILE_SHA256,
            SUPPORT_AUDIT_CONTENT_SHA256,
        ),
        evidence=SelfHashedSourceV62A(
            EVIDENCE,
            _require_sha256_v62a(EVIDENCE_FILE_SHA256, "evidence file"),
            _require_sha256_v62a(EVIDENCE_CONTENT_SHA256, "evidence content"),
        ),
        analysis=SelfHashedSourceV62A(
            ANALYSIS,
            _require_sha256_v62a(ANALYSIS_FILE_SHA256, "analysis file"),
            _require_sha256_v62a(ANALYSIS_CONTENT_SHA256, "analysis content"),
        ),
        report=SelfHashedSourceV62A(
            REPORT,
            _require_sha256_v62a(REPORT_FILE_SHA256, "report file"),
            _require_sha256_v62a(REPORT_CONTENT_SHA256, "report content"),
        ),
        gpu_log_path=GPU_LOG,
        gpu_log_file_sha256=_require_sha256_v62a(
            GPU_LOG_FILE_SHA256, "GPU log file"
        ),
    )


def _read_self_hashed_v62a(source: SelfHashedSourceV62A) -> dict:
    if file_sha256_v62a(source.path) != source.file_sha256:
        raise RuntimeError(f"v62a finalizer input file changed: {source.path}")
    value = json.loads(source.path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != source.content_sha256
        or analysis.canonical_sha256_v62a(compact) != source.content_sha256
    ):
        raise RuntimeError(f"v62a finalizer input content changed: {source.path}")
    return value


def _verify_static_sources_v62a(prereg: dict, audit: dict, sources) -> dict:
    installed = {
        key: file_sha256_v62a(path)
        for key, path in support_audit.v61e.SOURCE_PATHS.items()
    }
    if (
        prereg.get("schema")
        != "v62a-v434-pre-hpo-alpha-zero-generation-preregistration"
        or prereg.get("status")
        != "preregistered_before_train_semantics_model_or_gpu_access"
        or prereg.get("v62_methodology_commit") != analysis.V62_METHOD_COMMIT
        or prereg.get("v62_numeric_audit_identities")
        != analysis.V62_NUMERIC_AUDIT_IDENTITIES
        or prereg.get("v62_preregistration_identities")
        != analysis.V62_PREREGISTRATION_IDENTITIES
        or prereg.get("implementation_bindings")
        != runtime.implementation_bindings_v62a()
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
        != "v62a-installed-vllm-pre-hpo-alpha-zero-support-audit"
        or audit.get("status") != "supported"
        or audit.get("installed_vllm_source_file_sha256") != installed
        or audit.get("requested_runtime_controls")
        != analysis.RUNTIME_CONTROLS_V62A
        or audit.get("pre_hpo_alpha_zero_runtime_supported") is not True
        or audit.get("support_audit_authorizes_gpu_launch") is not False
        or audit.get("model_train_semantics_or_gpu_accessed") is not False
    ):
        raise RuntimeError("v62a preregistration, support, or code binding changed")
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
    }


def _verify_no_text_keys_v62a(name: str, value: object) -> dict:
    found = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                normalized = str(key).lower()
                if normalized in FORBIDDEN_TEXT_KEYS_V62A:
                    found.append(f"{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, name)
    if found:
        raise RuntimeError(f"v62a forbidden text-bearing key observed: {found[0]}")
    return {
        "source": name,
        "forbidden_text_key_count": 0,
        "forbidden_exact_keys": sorted(FORBIDDEN_TEXT_KEYS_V62A),
    }


def _verify_analysis_rebuild_v62a(evidence: dict, stored: dict) -> dict:
    rebuilt = analysis.build_analysis_v62a(evidence)
    if rebuilt != stored:
        raise RuntimeError("v62a stored analysis differs from exact numeric rebuild")
    return rebuilt


def _verify_actor_identities_v62a(report: dict, prereg: dict) -> dict:
    actors = report.get("actor_identities", [])
    expected_tuned = prereg["runtime"]["tuned_table_content_sha256"]
    expected_folder = prereg["runtime"]["tuned_folder"]
    by_gpu = {}
    pids = set()
    for item in actors:
        gpu = item.get("physical_gpu_id")
        pid = item.get("pid")
        controls = {
            key: item.get(key) for key in analysis.RUNTIME_CONTROLS_V62A
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
            or controls != analysis.RUNTIME_CONTROLS_V62A
            or item.get("scheduler_class") != "Scheduler"
            or item.get("submitted_request_batch_size") != 68
            or item.get("generation_only") is not True
            or item.get("global_batch_invariance_claimed") is not False
            or item.get("tuned_folder") != expected_folder
            or item.get("tuned_table_content_sha256") != expected_tuned
        ):
            raise RuntimeError("v62a actor runtime identity changed")
        by_gpu[gpu] = pid
        pids.add(pid)
    if len(actors) != 4 or set(by_gpu) != {0, 1, 2, 3}:
        raise RuntimeError("v62a actor GPU coverage changed")
    return {
        "actors": 4,
        "unique_processes": 4,
        "physical_gpu_ids": [0, 1, 2, 3],
        "all_runtime_controls_exact": True,
        "all_v27c_tuned_table_identities_exact": True,
    }


def _verify_state_v62a(evidence: dict, report: dict, prereg: dict) -> dict:
    receipts = evidence.get("state_receipts", [])
    recipe = prereg["fixed_calibration_recipe"]
    if (
        len(receipts) != 4
        or [item.get("period_index") for item in receipts] != [0, 1, 2, 3]
        or any(
            item.get("identical_v434_state") is not True
            or item.get("before") != item.get("after")
            or item.get("before", {}).get("canonical_fp32_master_sha256")
            != recipe["canonical_fp32_master_sha256"]
            or item.get("before", {}).get("bf16_runtime_values_sha256")
            != recipe["bf16_runtime_values_sha256"]
            for item in receipts
        )
        or evidence.get("numeric_state_receipts_sha256")
        != analysis.canonical_sha256_v62a(receipts)
        or report.get("state_receipts_sha256")
        != evidence.get("numeric_state_receipts_sha256")
        or report.get("master_state_receipt") != receipts[0].get("before")
    ):
        raise RuntimeError("v62a identical V434 state receipts changed")
    return {
        "periods": 4,
        "identical_before_after_periods": 4,
        "canonical_fp32_master_sha256": recipe[
            "canonical_fp32_master_sha256"
        ],
        "bf16_runtime_values_sha256": recipe["bf16_runtime_values_sha256"],
    }


def _gpu_summary_v62a(path: Path, expected_sha: str, report: dict) -> dict:
    if file_sha256_v62a(path) != expected_sha:
        raise RuntimeError("v62a GPU log changed")
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != 404:
        raise RuntimeError("v62a GPU log row count changed")
    actor_pids = {
        int(item["physical_gpu_id"]): int(item["pid"])
        for item in report.get("actor_identities", [])
    }
    if set(actor_pids) != {0, 1, 2, 3} or len(set(actor_pids.values())) != 4:
        raise RuntimeError("v62a GPU actor map changed")
    by_gpu = {}
    for gpu in range(4):
        expected_pid = actor_pids[gpu]
        selected = [item for item in rows if item.get("gpu") == gpu]
        if (
            len(selected) != 101
            or any(item.get("expected_pid") != expected_pid for item in selected)
            or any(item.get("foreign_compute_pids") != [] for item in selected)
            or any(
                any(pid != expected_pid for pid in item.get("compute_pids", []))
                for item in selected
            )
        ):
            raise RuntimeError("v62a foreign or mismatched GPU process observed")
        resident = [item for item in selected if expected_pid in item["compute_pids"]]
        if not resident or not any(
            item["utilization_percent"] > 0 for item in resident
        ):
            raise RuntimeError("v62a GPU lacked attributed positive activity")
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
        raise RuntimeError("v62a reported GPU summary differs from numeric log")
    return {
        **rebuilt,
        "gpu_log_rows": 404,
        "samples_per_gpu": 101,
        "foreign_compute_process_observations": 0,
    }


def _gate_observation_v62a(gate: dict) -> dict:
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
        != analysis.MAX_PRIMARY_CI_HALFWIDTH_V62A
        or gate.get("maximum_actor_leave_one_out_shift_inclusive")
        != analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
        or gate.get("failure_action")
        != "fail_closed_before_hpo_population_or_update"
        or gate.get("authorizes_hpo_population_or_update") is not False
    ):
        raise RuntimeError("v62a frozen gate semantics changed")
    failed = sorted(key for key, passed in checks.items() if not passed)
    return {
        "checks": copy.deepcopy(checks),
        "passed": gate["passed"],
        "passed_gate_count": len(checks) - len(failed),
        "failed_gate_count": len(failed),
        "failed_gates": failed,
        "failure_action": gate["failure_action"],
        "authorizes_hpo_population_or_update": False,
    }


def _verify_report_v62a(report: dict, prereg: dict, rebuilt: dict, sources) -> dict:
    gate = rebuilt["required_pre_hpo_gate"]
    expected_status = (
        "complete_gate_passed_hpo_still_unauthorized"
        if gate["passed"] else "complete_gate_failed_closed"
    )
    cleanup = report.get("cleanup", {})
    after = cleanup.get("after", [])
    evidence_ref = report.get("evidence", {})
    analysis_ref = report.get("analysis", {})
    if (
        report.get("schema") != "v62a-pre-hpo-alpha-zero-generation-report"
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
        or evidence_ref.get("periods") != 4
        or evidence_ref.get("generation_completions") != 1088
        or evidence_ref.get("teacher_forced_requests") != 0
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
        or len(after) != 4
        or any(item.get("state") != "REMOVED" for item in after)
        or report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
    ):
        raise RuntimeError("v62a sealed report, hash chain, or cleanup changed")
    return {
        "reported_status_matches_observed_gate": True,
        "engines_killed": 4,
        "placement_groups_removed": 4,
        "all_four_final_gpu_compute_process_lists_empty": True,
        "no_hpo_update_candidate_or_protected_opening": True,
    }


def _pair_diagnosis_v62a(evidence: dict) -> dict:
    rows = analysis.validate_evidence_v62a(evidence)
    generation = analysis._generation_array_v62a(rows)
    delta, receipts = analysis.paired_deltas_v62a(generation[:64])
    f1 = delta[..., 0]
    summaries = []
    for pair_index, label in enumerate(("first_or_cold_pair", "later_pair")):
        values = f1[:, :, pair_index]
        summaries.append({
            "pair_index": pair_index,
            "label": label,
            "periods": list(analysis.PAIR_PERIODS_V62A[pair_index]),
            "point_mean_f1_delta": float(np.mean(values)),
            "actor_mean_f1_deltas": [
                float(value) for value in np.mean(values, axis=0)
            ],
            "nonzero_replica_count": int(np.count_nonzero(values)),
            "units_with_any_nonzero_replica": int(np.sum(
                np.any(values != 0.0, axis=1)
            )),
            "positive_replica_count": int(np.sum(values > 0.0)),
            "negative_replica_count": int(np.sum(values < 0.0)),
        })
    return {
        "pairs": summaries,
        "later_minus_first_point_mean_f1_delta": (
            summaries[1]["point_mean_f1_delta"]
            - summaries[0]["point_mean_f1_delta"]
        ),
        "pair_receipts": receipts,
        "descriptive_observation": (
            "The first measured pair carries nearly all of the positive mean; "
            "the later pair mean is near zero."
        ),
        "cold_start_or_order_effect_claimed": False,
        "causal_interpretation_authorized": False,
        "used_to_change_or_relax_frozen_gates": False,
    }


def _performance_v62a(report: dict) -> dict:
    wall = float(report.get("wall_runtime_seconds", 0.0))
    if not math.isfinite(wall) or wall <= 0.0:
        raise RuntimeError("v62a wall runtime changed")
    return {
        "wall_runtime_seconds": wall,
        "generation_completions": 1088,
        "teacher_forced_requests": 0,
        "generation_completions_per_second": 1088 / wall,
    }


def build_finalized_v62a(
    sources: FinalizerSourcesV62A | None = None,
) -> dict:
    sources = production_sources_v62a() if sources is None else sources
    prereg = _read_self_hashed_v62a(sources.preregistration)
    audit = _read_self_hashed_v62a(sources.support_audit)
    static = _verify_static_sources_v62a(prereg, audit, sources)
    evidence = _read_self_hashed_v62a(sources.evidence)
    stored_analysis = _read_self_hashed_v62a(sources.analysis)
    report = _read_self_hashed_v62a(sources.report)
    leakage = {
        name: _verify_no_text_keys_v62a(name, value)
        for name, value in (
            ("evidence", evidence),
            ("analysis", stored_analysis),
            ("report", report),
        )
    }
    rebuilt = _verify_analysis_rebuild_v62a(evidence, stored_analysis)
    actors = _verify_actor_identities_v62a(report, prereg)
    states = _verify_state_v62a(evidence, report, prereg)
    report_check = _verify_report_v62a(
        report, prereg, rebuilt, sources
    )
    gpu = _gpu_summary_v62a(
        sources.gpu_log_path, sources.gpu_log_file_sha256, report
    )
    gate = _gate_observation_v62a(rebuilt["required_pre_hpo_gate"])
    value = {
        "schema": "v62a-pre-hpo-alpha-zero-independent-finalizer",
        "status": "complete_numeric_only_gate_observed_hpo_unauthorized",
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
            "performance": _performance_v62a(report),
        },
        "first_pair_vs_later_pair_descriptive_diagnosis": (
            _pair_diagnosis_v62a(evidence)
        ),
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
            "finalizer_file_sha256": file_sha256_v62a(
                Path(__file__).resolve()
            ),
            "tests_file_sha256": file_sha256_v62a(TESTS),
        },
        "raw_question_answer_prediction_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v62a(value)
    )
    return value


def _exclusive_write_v62a(path: Path, payload: bytes) -> None:
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
    value = build_finalized_v62a()
    _exclusive_write_v62a(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v62a(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "failed_gate_count": value[
            "observed_numeric_outcome_without_authorization"
        ]["required_pre_hpo_gate"]["failed_gate_count"],
        "hpo_authorized": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
