#!/usr/bin/env python3
"""Additive V73G path and worker binding over the sealed V73E contract.

V73G deliberately retains V73E's content-free workload, schemas, phase names,
guard, and postrun acceptance logic.  It changes only fresh artifact paths and
the worker module containing the controller-bootstrap repair.
"""

from __future__ import annotations

from pathlib import Path

import qwen36_v73e_exact_phase_profiler_contract as _base
from qwen36_v73e_exact_phase_profiler_contract import *  # noqa: F403


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT
    / "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_v73g_exact_phase_profiler.json"
).resolve()
TARGET = (
    ROOT / "run_lora_es_v71_v72_profile_calibration_v73g.py"
).resolve()
LAUNCHER = (ROOT / "run_qwen36_v73g_exact_phase_profiler.py").resolve()
WORKER = (ROOT / "eggroll_es_worker_lora_v73g.py").resolve()
WORKER_EXTENSION = (
    "eggroll_es_worker_lora_v73g.LoRAAdapterStateWorkerExtensionV73E"
)
BASE_RUNTIME_CONTRACT = _base


def arm_artifacts_v73e(arm: str) -> dict[str, str]:
    if arm not in {"timeline", "hbm_metrics"}:
        raise ValueError(f"unsupported V73G profiler arm: {arm}")
    stem = f"v73g_{arm}_lora_es_content_free_qwen36_exact_phase"
    run = (ROOT / "experiments/eggroll_es_hpo/runs" / stem).resolve()
    profile = (ROOT / "experiments/eggroll_es_hpo/profiles" / stem).resolve()
    trace_base = profile / stem
    return {
        "application_attempt": str(run.parent / f".{stem}.attempt.json"),
        "run_directory": str(run),
        "gpu_log": str(run / "gpu_activity_v73g.jsonl"),
        "actor_cuda_work_log": str(run / "actor_cuda_work_receipts_v73g.jsonl"),
        "host_process_samples": str(run / "host_process_samples_v73g.jsonl"),
        "host_process_summary": str(run / "host_process_summary_v73g.json"),
        "population": str(run / "mirrored_population_v73g.json"),
        "update": str(run / "pair_difference_update_v73g.json"),
        "audit_traffic": str(run / "exact_audit_traffic_v73g.json"),
        "equivalence": str(run / "content_free_systems_consistency_v73g.json"),
        "phase_receipt": str(run / "exact_phase_ranges_v73g.json"),
        "path_guard_receipt": str(run / "systems_only_path_guard_v73g.json"),
        "report": str(run / "mirrored_calibration_report_v73g.json"),
        "failure": str(run / "failure_v73g.json"),
        "profile_directory": str(profile),
        "profile_attempt": str(profile.parent / f".{stem}.attempt.json"),
        "profile_failure": str(profile / "profile_failure_v73g.json"),
        "nsys_report": str(trace_base.with_suffix(".nsys-rep")),
        "sqlite_export": str(trace_base.with_suffix(".sqlite")),
        "profile_analysis": str(profile / "profile_analysis_v73g.json"),
        "nccl_debug_pattern": str(profile / "nccl_debug.*.log"),
    }


# The inherited functions resolve their globals in the V73E base module. Patch
# only the prospective path and worker surface before exposing those functions.
_base.OUTPUT = OUTPUT
_base.TARGET = TARGET
_base.LAUNCHER = LAUNCHER
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


validate_generated_preregistration_v73e = (
    _base.validate_generated_preregistration_v73e
)
