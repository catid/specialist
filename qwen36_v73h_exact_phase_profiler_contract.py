#!/usr/bin/env python3
"""V73H fresh-path contract for process-start Ray actor guard installation."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import qwen36_v73e_exact_phase_profiler_contract as _base
from qwen36_v73e_exact_phase_profiler_contract import *  # noqa: F403


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73h_exact_phase_profiler.json"
).resolve()
TARGET = (
    ROOT / "run_lora_es_v71_v72_profile_calibration_v73h.py"
).resolve()
LAUNCHER = (ROOT / "run_qwen36_v73h_exact_phase_profiler.py").resolve()
GUARD_DIRECTORY = (ROOT / "v73h_sitecustomize").resolve()
GUARD = (GUARD_DIRECTORY / "v73e_path_open_guard.py").resolve()
SITECUSTOMIZE = (GUARD_DIRECTORY / "sitecustomize.py").resolve()
WORKER = (ROOT / "eggroll_es_worker_lora_v73h.py").resolve()
BUILDER = (
    ROOT / "build_qwen36_v73h_exact_phase_profiler_preregistration.py"
).resolve()
CLOSURE_AUDITOR = (
    ROOT / "audit_v73h_systems_only_import_graph_v1.py"
).resolve()
REGRESSION_TEST = (
    ROOT / "test_qwen36_v73h_process_start_actor_guard.py"
).resolve()
WORKER_EXTENSION = (
    "eggroll_es_worker_lora_v73h.LoRAAdapterStateWorkerExtensionV73E"
)
ACTOR_GUARD_MECHANISM = "ray_actor_sitecustomize_pre_runtime_imports"
BASE_RUNTIME_CONTRACT = _base


def arm_artifacts_v73e(arm: str) -> dict[str, str]:
    if arm not in {"timeline", "hbm_metrics"}:
        raise ValueError(f"unsupported V73H profiler arm: {arm}")
    stem = f"v73h_{arm}_lora_es_content_free_qwen36_exact_phase"
    run = (ROOT / "experiments/eggroll_es_hpo/runs" / stem).resolve()
    profile = (ROOT / "experiments/eggroll_es_hpo/profiles" / stem).resolve()
    trace_base = profile / stem
    return {
        "application_attempt": str(run.parent / f".{stem}.attempt.json"),
        "run_directory": str(run),
        "gpu_log": str(run / "gpu_activity_v73h.jsonl"),
        "actor_cuda_work_log": str(run / "actor_cuda_work_receipts_v73h.jsonl"),
        "host_process_samples": str(run / "host_process_samples_v73h.jsonl"),
        "host_process_summary": str(run / "host_process_summary_v73h.json"),
        "population": str(run / "mirrored_population_v73h.json"),
        "update": str(run / "pair_difference_update_v73h.json"),
        "audit_traffic": str(run / "exact_audit_traffic_v73h.json"),
        "equivalence": str(run / "content_free_systems_consistency_v73h.json"),
        "phase_receipt": str(run / "exact_phase_ranges_v73h.json"),
        "path_guard_receipt": str(run / "systems_only_path_guard_v73h.json"),
        "report": str(run / "mirrored_calibration_report_v73h.json"),
        "failure": str(run / "failure_v73h.json"),
        "profile_directory": str(profile),
        "profile_attempt": str(profile.parent / f".{stem}.attempt.json"),
        "profile_failure": str(profile / "profile_failure_v73h.json"),
        "nsys_report": str(trace_base.with_suffix(".nsys-rep")),
        "sqlite_export": str(trace_base.with_suffix(".sqlite")),
        "profile_analysis": str(profile / "profile_analysis_v73h.json"),
        "nccl_debug_pattern": str(profile / "nccl_debug.*.log"),
    }


_base.OUTPUT = OUTPUT
_base.TARGET = TARGET
_base.LAUNCHER = LAUNCHER
_base.GUARD_DIRECTORY = GUARD_DIRECTORY
_base.GUARD = GUARD
_base.SITECUSTOMIZE = SITECUSTOMIZE
_base.WORKER = WORKER
_base.WORKER_EXTENSION = WORKER_EXTENSION
_base.arm_artifacts_v73e = arm_artifacts_v73e


def command_template_v73e(arm: str) -> list[str]:
    return _base.command_template_v73e(arm)


def expand_command_v73e(
    arm: str,
    preregistration_file_sha256: str,
    preregistration_content_sha256: str,
) -> list[str]:
    return _base.expand_command_v73e(
        arm,
        preregistration_file_sha256,
        preregistration_content_sha256,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_generated_preregistration_v73e(value: Mapping[str, Any]) -> None:
    original = copy.deepcopy(dict(value))
    body = copy.deepcopy(original)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and claimed == canonical_sha256(body),
        "V73H preregistration self hash changed",
    )
    bootstrap = original.get("ray_actor_guard_bootstrap", {})
    amendment = original.get("v73h_successor_amendment", {})
    bindings = amendment.get("bindings", {})
    predecessor = original.get("immutable_v73g_attempt_1_predecessor", {})
    _require(
        bootstrap.get("actor_mechanism") == ACTOR_GUARD_MECHANISM
        and bootstrap.get("actor_guard_installed_by_sitecustomize_before_runtime_imports")
        is True
        and bootstrap.get("worker_extension_resolved_after_process_start_guard")
        is True
        and bootstrap.get("actor_guard_install_deferred_until_worker_extension")
        is False
        and bootstrap.get("actor_guard_file_sha256") == file_sha256(GUARD)
        and amendment.get("schema")
        == "qwen36-v73h-process-start-actor-guard-amendment-v1"
        and amendment.get("only_runtime_logic_change")
        == (
            "actor_sitecustomize_installs_the_exact_actor_guard_before_"
            "application_and_vllm_runtime_imports"
        )
        and amendment.get(
            "actor_sitecustomize_requires_exact_flag_hash_and_inherited_controller_pid"
        )
        is True
        and amendment.get(
            "parent_and_historical_modules_imported_after_actor_guard_are_covered"
        )
        is True
        and amendment.get("worker_extension_validates_preinstalled_actor_guard_receipt")
        is True
        and amendment.get("controller_and_actor_guard_mechanisms_remain_distinct")
        is True
        and amendment.get("v73g_attempt_1_reuse_or_overwrite_authorized") is False
        and amendment.get("quality_hpo_or_promotion_authorized") is False
        and amendment.get("protected_dev_ood_or_holdout_opened") is False
        and bindings.get("successor_builder") == source_identity(BUILDER)
        and bindings.get("successor_contract")
        == source_identity(Path(__file__).resolve())
        and bindings.get("successor_target") == source_identity(TARGET)
        and bindings.get("successor_launcher") == source_identity(LAUNCHER)
        and bindings.get("successor_worker") == source_identity(WORKER)
        and bindings.get("successor_guard") == source_identity(GUARD)
        and bindings.get("successor_sitecustomize")
        == source_identity(SITECUSTOMIZE)
        and bindings.get("successor_closure_auditor")
        == source_identity(CLOSURE_AUDITOR)
        and bindings.get("successor_regression_tests")
        == source_identity(REGRESSION_TEST)
        and predecessor.get("status")
        == "immutable_failed_closed_real_ray_actor_preload_before_worker_guard",
        "V73H additive actor-guard amendment changed",
    )

    # Reuse every unchanged V73E structural validator after adapting only the
    # explicitly amended mechanism back to its compatibility value.
    compatible = copy.deepcopy(original)
    compatible["ray_actor_guard_bootstrap"]["actor_mechanism"] = (
        "ray_actor_worker_extension_pre_parent_import"
    )
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = canonical_sha256(compatible)
    _base.validate_generated_preregistration_v73e(compatible)
