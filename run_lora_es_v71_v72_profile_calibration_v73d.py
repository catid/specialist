#!/usr/bin/env python3
"""Fresh-path V73D target for prospective exact-phase profiling.

The target reuses the sealed V73B workload and fail-closed semantics without
mutating or reading from its output directory.  It adds controller NVTX phase
ranges and a self-hashed phase receipt.  The process is not a launcher: live
execution requires an exact expanded-command attestation supplied by the
sealed parent profiler.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import qwen36_v73d_exact_phase_profiler_contract as builder
import run_lora_es_mirrored_calibration_v66 as v66
import build_lora_es_v71_v72_live_calibration_preregistration_v73 as v73_builder


def _sealed_historical_v73_contract_for_systems_runtime():
    path = builder.V73_PREREGISTRATION
    if builder.file_sha256(path) != builder.V73_PREREGISTRATION_FILE_SHA256:
        raise RuntimeError("V73D historical V73 systems contract changed")
    value = json.loads(path.read_text(encoding="ascii"))
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    if (
        claimed != builder.V73_PREREGISTRATION_CONTENT_SHA256
        or builder.canonical_sha256(body) != claimed
    ):
        raise RuntimeError("V73D historical V73 contract self hash changed")
    return value


# V73's legacy module computes endpoint constants by rebuilding V66 ancestry,
# which would open the quarantined historical contract.  Supply the already
# sealed V73 bytes for that import-time constant only; no quality authority is
# inherited or rebuilt.
v73_builder.build_preregistration_v73 = (
    _sealed_historical_v73_contract_for_systems_runtime
)

import run_lora_es_v71_v72_live_calibration_v73 as v73
import run_lora_es_v71_v72_same_live_calibration_v73b as v73b


ROOT = Path(__file__).resolve().parent
COMMAND_ATTESTATION_ENV = "SPECIALIST_V73D_EXPANDED_COMMAND_SHA256"
_PHASE_INSTANCE = None
_GUARD_EVIDENCE = None
_GUARD_FAILURE = None
_RAY_BOOTSTRAP_EVIDENCE = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="ascii"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _validate_self_hash(value: Mapping[str, Any], label: str) -> str:
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and claimed == builder.canonical_sha256(body),
        f"V73D self hash changed: {label}",
    )
    return claimed


def load_preregistration_v73d(args) -> dict[str, Any]:
    path = Path(args.preregistration).resolve()
    _require(
        builder.file_sha256(path) == args.preregistration_sha256,
        "V73D preregistration file identity changed",
    )
    value = _load_json(path)
    _validate_self_hash(value, "preregistration")
    builder.validate_generated_preregistration_v73d(value)
    _require(
        value.get("content_sha256_before_self_field")
        == args.preregistration_content_sha256
        and value.get("status")
        == "sealed_cpu_only_before_v73d_model_ray_gpu_or_protected_access",
        "V73D preregistration content changed",
    )
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    result = dict(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = builder.canonical_sha256(result)
    payload = (
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-v73d-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return result


def _rewrite_json(path: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(path)
    value.update(dict(updates))
    return _atomic_json(path, value)


def silence_target_streams_v73d() -> None:
    """Keep Nsight's mandatory ProcessStreams payload empty before workload."""
    sys.stdout.flush()
    sys.stderr.flush()
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        os.close(descriptor)


def _path_guard_v73d():
    module = importlib.import_module("v73d_path_open_guard")
    _require(
        module.installed()
        and module.installation_mechanism() == "controller_sitecustomize"
        and os.environ.get("SPECIALIST_V73D_SYSTEMS_ONLY_GUARD") == "1"
        and os.environ.get("SPECIALIST_V73D_CONTROLLER_GUARD_PID")
        == str(os.getpid())
        and Path(module.__file__).resolve() == builder.GUARD,
        "V73D systems-only path guard was not installed before imports",
    )
    return module


def _validate_guard_process_receipt_v73d(
    value: Mapping[str, Any], expected_mechanism: str
) -> dict[str, Any]:
    row = dict(value)
    claimed = row.pop("receipt_sha256", None)
    _require(
        claimed == builder.canonical_sha256(row)
        and row.get("schema")
        == "qwen36-v73d-systems-only-path-guard-process-v1"
        and row.get("installed_before_runtime_imports") is True
        and row.get("installation_mechanism") == expected_mechanism
        and row.get("installation_pid") == row.get("pid")
        and row.get("boundary_registry_file_sha256")
        == builder.BOUNDARY_REGISTRY_FILE_SHA256
        and row.get("boundary_registry_content_sha256")
        == builder.BOUNDARY_REGISTRY_CONTENT_SHA256
        and row.get("exact_path_identity_count") == 3
        and row.get("prefix_identity_count") == 2
        and row.get("successful_protected_opens") == 0
        and row.get("successful_protected_resolves") == 0
        and row.get("successful_protected_metadata") == 0
        and row.get("successful_protected_enumerations") == 0
        and row.get("protected_path_values_persisted") is False
        and row.get("quality_hpo_or_promotion_authorized") is False,
        "V73D systems-only process guard receipt changed",
    )
    return dict(value)


def _validate_worker_bootstrap_receipt_v73d(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(value)
    claimed = row.pop("receipt_sha256", None)
    pre_parent_guard = _validate_guard_process_receipt_v73d(
        row.get("pre_parent_guard_receipt", {}),
        "ray_actor_worker_extension_pre_parent_import",
    )
    guard = _validate_guard_process_receipt_v73d(
        row.get("guard_process_receipt", {}),
        "ray_actor_worker_extension_pre_parent_import",
    )
    _require(
        claimed == builder.canonical_sha256(row)
        and row.get("schema")
        == "qwen36-v73d-worker-bootstrap-receipt-v1"
        and row.get("pid") == guard.get("pid")
        and row.get("process_role") == "ray_actor_worker_extension"
        and row.get("bootstrap_mechanism")
        == "ray_actor_worker_extension_pre_parent_import"
        and row.get("guard_was_preinstalled") is False
        and row.get("parent_modules_absent_before_guard_install") is True
        and row.get(
            "historical_reference_modules_absent_before_guard_install"
        )
        is True
        and row.get("historical_reference_module_identity_count") == 3
        and isinstance(
            row.get("historical_reference_module_identity_set_sha256"), str
        )
        and row["historical_reference_module_identity_set_sha256"]
        == builder.HISTORICAL_REFERENCE_MODULE_IDENTITY_SET_SHA256
        and row.get("parent_module_count_after_guard_install") == 3
        and row.get("guard_source_sha256")
        == builder.file_sha256(builder.GUARD)
        and row.get("actor_bootstrap_env_exact") is True
        and row.get("pre_parent_guard_receipt_sha256")
        == pre_parent_guard.get("receipt_sha256")
        and pre_parent_guard.get("pid") == guard.get("pid")
        and row.get("quality_hpo_or_promotion_authorized") is False,
        "V73D Ray actor worker bootstrap receipt changed",
    )
    return dict(value)


def _guard_evidence_v73d(worker_rows) -> dict[str, Any]:
    _require(
        isinstance(_RAY_BOOTSTRAP_EVIDENCE, dict),
        "V73D Ray runtime-env bootstrap evidence is absent",
    )
    controller = _validate_guard_process_receipt_v73d(
        _path_guard_v73d().receipt(), "controller_sitecustomize"
    )
    workers = [
        _validate_worker_bootstrap_receipt_v73d(row) for row in worker_rows
    ]
    worker_guards = [row["guard_process_receipt"] for row in workers]
    _require(
        len(workers) == 4
        and len({row["pid"] for row in workers}) == 4
        and controller["pid"] not in {row["pid"] for row in workers}
        and len({row["exact_path_identity_set_sha256"] for row in worker_guards})
        == 1
        and worker_guards[0]["exact_path_identity_set_sha256"]
        == controller["exact_path_identity_set_sha256"]
        and len({row["prefix_identity_set_sha256"] for row in worker_guards})
        == 1
        and worker_guards[0]["prefix_identity_set_sha256"]
        == controller["prefix_identity_set_sha256"],
        "V73D systems-only worker guard coverage changed",
    )
    processes = [controller, *worker_guards]
    return {
        "schema": "qwen36-v73d-systems-only-path-guard-receipt-v1",
        "status": (
            "zero_successful_protected_open_resolve_metadata_or_"
            "enumeration_systems_only"
        ),
        "controller": controller,
        "workers": workers,
        "controller_bootstrap_mechanism": "controller_sitecustomize",
        "actor_bootstrap_mechanism": (
            "ray_actor_worker_extension_pre_parent_import"
        ),
        "mechanisms_are_distinct": True,
        "ray_job_runtime_env_bootstrap": copy.deepcopy(
            _RAY_BOOTSTRAP_EVIDENCE
        ),
        "process_count": len(processes),
        "successful_protected_opens": sum(
            row["successful_protected_opens"] for row in processes
        ),
        "successful_protected_resolves": sum(
            row["successful_protected_resolves"] for row in processes
        ),
        "successful_protected_metadata": sum(
            row["successful_protected_metadata"] for row in processes
        ),
        "successful_protected_enumerations": sum(
            row["successful_protected_enumerations"] for row in processes
        ),
        "denied_protected_open_attempts": sum(
            row["protected_open_attempts_denied"] for row in processes
        ),
        "denied_protected_resolve_attempts": sum(
            row["protected_resolve_attempts_denied"] for row in processes
        ),
        "protected_path_values_persisted": False,
        "quality_hpo_or_promotion_performed": False,
        "lineage_rehabilitation_performed": False,
    }


class _PhaseRangeLedgerV73D:
    def __init__(self, domain):
        self.domain = domain
        self.rows: list[dict[str, Any]] = []
        self.active: dict[str, Any] | None = None
        self.controller_pid = os.getpid()

    def open(self, phase: str, epoch: int) -> None:
        _require(self.active is None, "V73D NVTX phase range nested unexpectedly")
        _require(
            isinstance(phase, str) and phase in builder.PHASES,
            f"V73D unsupported phase: {phase}",
        )
        started_ns = time.monotonic_ns()
        self.domain.push_range(message=phase, color="blue")
        self.active = {
            "phase": phase,
            "epoch": int(epoch),
            "controller_pid": self.controller_pid,
            "started_monotonic_ns": started_ns,
        }

    def close(self) -> None:
        if self.active is None:
            return
        ended_ns = time.monotonic_ns()
        self.domain.pop_range()
        row = dict(self.active)
        row["ended_monotonic_ns"] = ended_ns
        row["elapsed_ns"] = ended_ns - row["started_monotonic_ns"]
        _require(row["elapsed_ns"] > 0, "V73D NVTX phase range was empty")
        self.rows.append(row)
        self.active = None


def phase_class_v73d(base_phase_class, domain):
    """Build the additive phase class; injectable for CPU-only tests."""

    class ExactPhaseHandshakeV73D(base_phase_class):
        def __init__(self, *args, **kwargs):
            global _PHASE_INSTANCE
            super().__init__(*args, **kwargs)
            _require(_PHASE_INSTANCE is None, "V73D phase instance duplicated")
            self.v73d_ledger = _PhaseRangeLedgerV73D(domain)
            _PHASE_INSTANCE = self
            phase, epoch = self.snapshot()
            self.v73d_ledger.open(phase, epoch)

        @property
        def value(self):
            return base_phase_class.value.fget(self)

        @value.setter
        def value(self, phase):
            current, _ = self.snapshot()
            if phase == current:
                return
            self.v73d_ledger.close()
            base_phase_class.value.fset(self, phase)
            observed, epoch = self.snapshot()
            _require(observed == phase, "V73D phase transition did not publish")
            self.v73d_ledger.open(observed, epoch)

        def close_v73d_range(self):
            self.v73d_ledger.close()

    ExactPhaseHandshakeV73D.__name__ = "ExactPhaseHandshakeV73D"
    return ExactPhaseHandshakeV73D


def _v73b_artifact_mapping(paths: Mapping[str, str]) -> dict[str, str]:
    return {
        "attempt": paths["application_attempt"],
        "run_directory": paths["run_directory"],
        "gpu_log": paths["gpu_log"],
        "actor_cuda_work_log": paths["actor_cuda_work_log"],
        "host_process_samples": paths["host_process_samples"],
        "host_process_summary": paths["host_process_summary"],
        "population": paths["population"],
        "update": paths["update"],
        "audit_traffic": paths["audit_traffic"],
        "equivalence": paths["equivalence"],
        "report": paths["report"],
        "failure": paths["failure"],
    }


def ray_actor_bootstrap_runtime_env_v73d(
    runtime_env: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copied Ray job runtime-env with the actor guard injected."""
    merged = copy.deepcopy(runtime_env or {})
    _require(
        isinstance(merged, dict),
        "V73D Ray runtime_env must be a mapping",
    )
    env_vars = copy.deepcopy(merged.pop("env_vars", {}) or {})
    _require(
        isinstance(env_vars, dict)
        and all(isinstance(key, str) and isinstance(value, str)
                for key, value in env_vars.items())
        and builder.ACTOR_BOOTSTRAP_ENV not in env_vars
        and builder.ACTOR_GUARD_SHA_ENV not in env_vars,
        "V73D Ray actor bootstrap env collided",
    )
    guard_sha256 = builder.file_sha256(builder.GUARD)
    env_vars.update({
        builder.ACTOR_BOOTSTRAP_ENV: "1",
        builder.ACTOR_GUARD_SHA_ENV: guard_sha256,
    })
    merged["env_vars"] = env_vars
    evidence = {
        "schema": "qwen36-v73d-ray-job-runtime-env-bootstrap-v1",
        "injected_before_ray_init": True,
        "job_env_merges_with_actor_specific_runtime_env": True,
        "actor_bootstrap_env_name": builder.ACTOR_BOOTSTRAP_ENV,
        "actor_guard_sha_env_name": builder.ACTOR_GUARD_SHA_ENV,
        "actor_guard_file_sha256": guard_sha256,
        "environment_values_persisted_beyond_fixed_flags_and_hash": False,
    }
    return merged, evidence


@contextmanager
def patched_live_v73d(
    preregistration: Mapping[str, Any],
    workload: Mapping[str, Any],
    control: Mapping[str, Any],
    arm: str,
    *,
    nvtx_domain=None,
):
    """Patch only the new process and restore every inherited global."""
    import ray
    import run_lora_topology_probe_v40a as v40a

    global _PHASE_INSTANCE, _GUARD_EVIDENCE, _GUARD_FAILURE
    global _RAY_BOOTSTRAP_EVIDENCE
    _require(_PHASE_INSTANCE is None, "V73D live patch is not reentrant")
    if nvtx_domain is None:
        import nvtx

        nvtx_domain = nvtx.Domain(builder.PHASE_DOMAIN)
    paths = preregistration["arms"][arm]["artifacts"]
    mapped = _v73b_artifact_mapping(paths)
    replacements = {
        "PREREGISTRATION": builder.OUTPUT,
        "RUN_DIR": Path(mapped["run_directory"]),
        "_ARTIFACTS": mapped,
        "ATTEMPT": Path(mapped["attempt"]),
        "GPU_LOG": Path(mapped["gpu_log"]),
        "GPU_WORK_LOG": Path(mapped["actor_cuda_work_log"]),
        "HOST_SAMPLES": Path(mapped["host_process_samples"]),
        "HOST_SUMMARY": Path(mapped["host_process_summary"]),
        "POPULATION": Path(mapped["population"]),
        "UPDATE": Path(mapped["update"]),
        "AUDIT_TRAFFIC": Path(mapped["audit_traffic"]),
        "EQUIVALENCE": Path(mapped["equivalence"]),
        "REPORT": Path(mapped["report"]),
        "FAILURE": Path(mapped["failure"]),
    }
    saved_v73b = {name: getattr(v73b, name) for name in replacements}
    base_phase = v73.CompatiblePhaseHandshakeV73
    base_abort = v73._execute_and_abort_nonzero_update_v73
    base_worker_extension = v73.WORKER_EXTENSION_V73
    base_cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a
    base_ray_init = ray.init
    phase_class = phase_class_v73d(base_phase, nvtx_domain)

    def ray_init_v73d(*args, **kwargs):
        global _RAY_BOOTSTRAP_EVIDENCE
        _require(
            _RAY_BOOTSTRAP_EVIDENCE is None,
            "V73D Ray bootstrap injection was invoked more than once",
        )
        runtime_env, evidence = ray_actor_bootstrap_runtime_env_v73d(
            kwargs.pop("runtime_env", {}) or {}
        )
        kwargs["runtime_env"] = runtime_env
        _RAY_BOOTSTRAP_EVIDENCE = evidence
        return base_ray_init(*args, **kwargs)

    def abort_then_final_audit_v73d(*args, **kwargs):
        value = base_abort(*args, **kwargs)
        context = v73._ACTIVE_CONTEXT_V73
        _require(context is not None and context.phase is not None,
                 "V73D abort phase context disappeared")
        context.phase.value = "post_abort_final_audit_all_actors"
        return value

    def cleanup_phase_v73d(trainer):
        global _GUARD_EVIDENCE, _GUARD_FAILURE
        context = v73._ACTIVE_CONTEXT_V73
        if context is not None and context.phase is not None:
            context.phase.value = "cleanup_all_actors"
        try:
            worker_rows = v73._rpc_all_v73(
                trainer, "systems_only_path_guard_receipt_v73d"
            )
            _GUARD_EVIDENCE = _guard_evidence_v73d(worker_rows)
        except BaseException as error:
            _GUARD_FAILURE = error
        return base_cleanup(trainer)

    for name, value in replacements.items():
        setattr(v73b, name, value)
    v73.CompatiblePhaseHandshakeV73 = phase_class
    v73.WORKER_EXTENSION_V73 = builder.WORKER_EXTENSION
    v73._execute_and_abort_nonzero_update_v73 = abort_then_final_audit_v73d
    v40a.cleanup_v38a.strict_close_trainer_v38a = cleanup_phase_v73d
    ray.init = ray_init_v73d
    try:
        with v73b.patched_live_v73b(workload, control) as context:
            yield context
    finally:
        if _PHASE_INSTANCE is not None:
            _PHASE_INSTANCE.close_v73d_range()
        v40a.cleanup_v38a.strict_close_trainer_v38a = base_cleanup
        ray.init = base_ray_init
        v73._execute_and_abort_nonzero_update_v73 = base_abort
        v73.CompatiblePhaseHandshakeV73 = base_phase
        v73.WORKER_EXTENSION_V73 = base_worker_extension
        for name, value in saved_v73b.items():
            setattr(v73b, name, value)


def phase_receipt_v73d(arm: str, *, complete: bool) -> dict[str, Any]:
    instance = _PHASE_INSTANCE
    rows = [] if instance is None else list(instance.v73d_ledger.rows)
    observed = [row["phase"] for row in rows]
    if complete:
        _require(
            observed == list(builder.PHASES),
            f"V73D phase sequence changed: {observed}",
        )
        _require(
            all(
                right["started_monotonic_ns"] >= left["ended_monotonic_ns"]
                for left, right in zip(rows, rows[1:])
            ),
            "V73D phase ranges overlapped",
        )
    result = {
        "schema": "eggroll-es-exact-phase-range-receipt-v73d",
        "arm": arm,
        "complete": bool(complete),
        "nvtx_domain": builder.PHASE_DOMAIN,
        "expected_phase_order": list(builder.PHASES),
        "observed_phase_order": observed,
        "phase_count": len(rows),
        "one_controller_pid": len({row["controller_pid"] for row in rows}) <= 1,
        "rows": rows,
        "contains_prompts_questions_answers_or_outputs": False,
    }
    return result


def _reference(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "file_sha256": builder.file_sha256(path),
        "content_sha256": value["content_sha256_before_self_field"],
    }


def gpu_time_accounting_v73d(
    legacy_compute: Mapping[str, Any] | None,
) -> dict[str, Any]:
    legacy = dict(legacy_compute or {})
    return {
        "schema": "qwen36-v73d-gpu-time-accounting-split-v1",
        "reserved_wall_gpu_seconds": legacy.get("charged_gpu_seconds"),
        "reserved_wall_source": "legacy_compute_ledger_charged_gpu_seconds",
        "legacy_estimated_model_allocation_or_residency_seconds_per_gpu": (
            legacy.get("model_allocation_or_residency_seconds_per_gpu")
        ),
        "legacy_residency_estimate_classification": (
            "diagnostic_not_directly_measured_not_accepted"
        ),
        "model_resident_gpu_seconds_measured": None,
        "useful_gpu_seconds_measured": None,
        "promotion_charged_gpu_seconds": 0,
        "reserved_wall_is_not_model_residency_or_useful_work": True,
        "profiled_event_time_is_not_reclassified_as_unprofiled_useful_time": (
            True
        ),
    }


def finalize_success_v73d(
    context,
    preregistration: Mapping[str, Any],
    arm: str,
    command_sha256: str,
) -> dict[str, Any]:
    paths = preregistration["arms"][arm]["artifacts"]
    _require(
        _GUARD_FAILURE is None
        and isinstance(_GUARD_EVIDENCE, dict)
        and isinstance(_RAY_BOOTSTRAP_EVIDENCE, dict),
        "V73D systems-only path guard evidence is incomplete",
    )
    guard_path = Path(paths["path_guard_receipt"])
    guard = _atomic_json(guard_path, _GUARD_EVIDENCE)
    phase_path = Path(paths["phase_receipt"])
    phase = _atomic_json(phase_path, phase_receipt_v73d(arm, complete=True))
    v73b.finalize_success_artifacts_v73b(context)

    population_path = Path(paths["population"])
    population = _rewrite_json(population_path, {
        "schema": "v73d-v71-v72-qwen36-population-evidence",
        "profile_arm": arm,
    })
    update_path = Path(paths["update"])
    update = _rewrite_json(update_path, {
        "schema": "v73d-v71-v72-qwen36-update-evidence",
        "profile_arm": arm,
        "population_content_sha256": population[
            "content_sha256_before_self_field"
        ],
    })
    equivalence_path = Path(paths["equivalence"])
    equivalence = _rewrite_json(equivalence_path, {
        "schema": "eggroll-es-same-live-equivalence-v73d",
        "profile_arm": arm,
    })
    report_path = Path(paths["report"])
    report = _load_json(report_path)
    gpu_time_accounting = gpu_time_accounting_v73d(
        report.get("compute_ledger", {})
    )
    report.update({
        "schema": "v73d-exact-phase-qwen36-calibration-report",
        "status": "complete_same_live_no_commit_awaiting_parent_trace_analysis",
        "beads": [
            "specialist-0j5.32",
            "specialist-nen.33",
            "specialist-nen.34",
        ],
        "profile_arm": arm,
        "expanded_profiler_command_sha256": command_sha256,
        "population": _reference(population_path, population),
        "nonzero_update": _reference(update_path, update),
        "same_live_equivalence": _reference(equivalence_path, equivalence),
        "exact_phase_receipt": _reference(phase_path, phase),
        "systems_only_path_guard": _reference(guard_path, guard),
        "ray_actor_guard_bootstrap": copy.deepcopy(
            _RAY_BOOTSTRAP_EVIDENCE
        ),
        "immutable_v73c_attempt_1_predecessor": preregistration[
            "immutable_v73c_attempt_1_predecessor"
        ],
        "gpu_time_accounting_policy": {
            "reserved_wall_gpu_seconds_reported_separately": True,
            "model_resident_gpu_seconds_require_direct_evidence": True,
            "useful_gpu_seconds_require_direct_evidence": True,
            "promotion_charged_gpu_seconds": 0,
        },
        "gpu_time_accounting": gpu_time_accounting,
        "parent_trace_output_finalizes_after_target_process_exit": True,
        "semantic_quality_selection_or_hpo_performed": False,
        "raw_prompts_questions_answers_or_outputs_persisted": False,
        "checkpoint_snapshot_or_promotion_performed": False,
        "protected_dev_ood_or_holdout_opened": False,
        "successful_protected_path_opens_or_resolves": 0,
        "successful_protected_open_resolve_metadata_or_enumeration": 0,
        "quality_hpo_promotion_or_lineage_rehabilitation_performed": False,
    })
    report = _atomic_json(report_path, report)

    attempt_path = Path(paths["application_attempt"])
    if attempt_path.is_file():
        _rewrite_json(attempt_path, {
            "schema": "v73d-exact-phase-qwen36-calibration-attempt",
            "status": "target_complete_parent_trace_analysis_pending",
            "profile_arm": arm,
            "expanded_profiler_command_sha256": command_sha256,
            "ray_actor_guard_bootstrap": copy.deepcopy(
                _RAY_BOOTSTRAP_EVIDENCE
            ),
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_or_promotion_authorized": False,
        })
    return report


def finalize_failure_v73d(
    preregistration: Mapping[str, Any],
    arm: str,
    command_sha256: str,
) -> None:
    paths = preregistration["arms"][arm]["artifacts"]
    run = Path(paths["run_directory"])
    if run.is_dir():
        phase_path = Path(paths["phase_receipt"])
        if not phase_path.exists():
            _atomic_json(phase_path, phase_receipt_v73d(arm, complete=False))
        guard_path = Path(paths["path_guard_receipt"])
        if not guard_path.exists() and isinstance(_GUARD_EVIDENCE, dict):
            _atomic_json(guard_path, _GUARD_EVIDENCE)
    failure_path = Path(paths["failure"])
    if failure_path.is_file():
        failure = _load_json(failure_path)
        legacy_compute = failure.get("compute_ledger", {})
        updates = {
            "schema": "v73d-exact-phase-qwen36-calibration-failure",
            "profile_arm": arm,
            "expanded_profiler_command_sha256": command_sha256,
            "historical_reward_floats_used_as_acceptance_gate": False,
            "same_live_equivalence_required": True,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_or_promotion_performed": False,
            "immutable_v73c_attempt_1_predecessor": preregistration[
                "immutable_v73c_attempt_1_predecessor"
            ],
            "ray_actor_guard_bootstrap": copy.deepcopy(
                _RAY_BOOTSTRAP_EVIDENCE
            ),
            "gpu_time_accounting": gpu_time_accounting_v73d(legacy_compute),
        }
        if Path(paths["phase_receipt"]).is_file():
            phase = _load_json(Path(paths["phase_receipt"]))
            updates["exact_phase_receipt"] = _reference(
                Path(paths["phase_receipt"]), phase
            )
        _rewrite_json(failure_path, updates)


def parser_v73d() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(builder.OUTPUT))
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--arm", choices=("timeline", "hbm_metrics"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv=None) -> int:
    args = parser_v73d().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("V73D target requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v73d(args)
    arm = preregistration["arms"][args.arm]
    expected_command = builder.expand_command_v73d(
        args.arm,
        args.preregistration_sha256,
        args.preregistration_content_sha256,
    )
    command_sha256 = builder.canonical_sha256(expected_command)
    if args.dry_run:
        print(json.dumps({
            "schema": preregistration["schema"],
            "arm": args.arm,
            "arm_status": arm["status"],
            "expected_artifacts": arm["artifacts"],
            "expanded_command_sha256": command_sha256,
            "phase_domain": builder.PHASE_DOMAIN,
            "phase_count": len(builder.PHASES),
            "train_semantics_model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "protected_dev_ood_or_holdout_opened": False,
        }, sort_keys=True))
        return 0
    _require(
        arm["launch_authorized_by_this_file_after_identity_checks"] is True,
        "V73D HBM metrics arm is blocked before model/Ray/GPU work",
    )
    _require(
        Path(sys.executable).absolute() == builder.REQUIRED_PYTHON,
        f"V73D target requires {builder.REQUIRED_PYTHON}",
    )
    _require(
        os.environ.get(COMMAND_ATTESTATION_ENV) == command_sha256,
        "V73D expanded profiler command attestation changed",
    )
    silence_target_streams_v73d()
    _path_guard_v73d()
    workload = _load_json(builder.V73B_PREREGISTRATION)
    control = v73.load_accepted_control_values_v73()
    try:
        with patched_live_v73d(
            preregistration, workload, copy.deepcopy(control), args.arm
        ) as context:
            result = v66.execute_v66(workload, args)
            _require(
                _GUARD_FAILURE is None and isinstance(_GUARD_EVIDENCE, dict),
                "V73D systems-only path guard receipt was not captured",
            )
            if _PHASE_INSTANCE is not None:
                _PHASE_INSTANCE.close_v73d_range()
            finalize_success_v73d(
                context, preregistration, args.arm, command_sha256
            )
            return result
    except BaseException:
        finalize_failure_v73d(preregistration, args.arm, command_sha256)
        raise
    finally:
        globals()["_PHASE_INSTANCE"] = None
        globals()["_GUARD_EVIDENCE"] = None
        globals()["_GUARD_FAILURE"] = None
        globals()["_RAY_BOOTSTRAP_EVIDENCE"] = None


if __name__ == "__main__":
    raise SystemExit(main())
