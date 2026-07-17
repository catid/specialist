#!/usr/bin/env python3
"""Fresh V66d mirrored calibration with race-free GPU work attribution.

V66d deliberately wraps the sealed V66 execution engine instead of editing
the historical V66/V66c implementation.  The wrapper substitutes a sampled
phase handshake, a V66d worker extension, and actor CUDA-event receipts while
retaining the already-live-proven mirrored ES/state/update protocol.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import build_lora_es_mirrored_calibration_preregistration_v66d as builder
import eggroll_es_gpu_telemetry_v66 as telemetry
import run_lora_es_mirrored_calibration_v66 as v66
import train_eggroll_es_specialist as base


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = builder.OUTPUT
RUN_DIR = builder.RUN
ATTEMPT = (
    RUN_DIR.parent
    / ".v66d_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
).resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v66d.jsonl").resolve()
GPU_WORK_LOG = (RUN_DIR / "actor_cuda_work_receipts_v66d.jsonl").resolve()
POPULATION = (RUN_DIR / "mirrored_population_v66d.json").resolve()
UPDATE = (RUN_DIR / "pair_difference_update_v66d.json").resolve()
REPORT = (RUN_DIR / "mirrored_calibration_report_v66d.json").resolve()
FAILURE = (RUN_DIR / "failure_v66d.json").resolve()
WORKER_EXTENSION_V66D = (
    "eggroll_es_worker_lora_v66d.LoRAAdapterStateWorkerExtensionV66D"
)
REQUIRED_WORKER_ENDPOINTS_V66D = (
    "install_adapter_state_v41a",
    "active_lora_slot_certificate_v66",
    "materialize_mirrored_adapter_v66",
    "begin_actor_gpu_work_v66d",
    "end_actor_gpu_work_v66d",
    "restore_mirrored_adapter_v66",
)


def artifacts_v66d():
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "gpu_log": str(GPU_LOG),
        "actor_cuda_work_log": str(GPU_WORK_LOG),
        "population": str(POPULATION),
        "update": str(UPDATE),
        "report": str(REPORT),
        "failure": str(FAILURE),
    }


def validate_lora_worker_contract_v66d():
    contract = base.validate_worker_state_mode(
        base.TRAINER_STATE_MODE_EXTERNAL_WORKER,
        WORKER_EXTENSION_V66D,
        REQUIRED_WORKER_ENDPOINTS_V66D,
    )
    extension = contract.pop("resolved_worker_extension")
    contract["resolved_class"] = f"{extension.__module__}.{extension.__qualname__}"
    contract["dense_full_weight_master_install_authorized"] = False
    return contract


def load_preregistration_v66d(args):
    path = Path(args.preregistration).resolve()
    if v66.file_sha256_v66(path) != args.preregistration_sha256:
        raise RuntimeError("v66d preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or v66.mirrored.canonical_sha256_v66(compact)
        != args.preregistration_content_sha256
        or value != builder.build_preregistration_v66d()
        or value.get("schema")
        != "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66d"
        or value.get("status")
        != "sealed_after_v66c_short_phase_telemetry_race_before_v66d_gpu_access"
        or value.get("artifacts") != artifacts_v66d()
    ):
        raise RuntimeError("v66d preregistration content changed")
    return value


class CompatiblePhaseHandshakeV66D(telemetry.PhaseHandshakeV66):
    """Expose the legacy ``.value`` API with a four-row sample barrier."""

    @property
    def value(self):
        return self.snapshot()[0]

    @value.setter
    def value(self, phase):
        current, _ = self.snapshot()
        if phase != current:
            self.transition(phase, timeout_seconds=15.0)


@dataclass(frozen=True)
class RuntimeGenerationHandleV66D:
    ref: Any
    assignment: dict


class LiveContextV66D:
    def __init__(self):
        self.bindings = None
        self.assignments = {}
        self.receipts = {}
        self.summary = None
        self.ready = threading.Event()

    def capture_bindings(self, actor, worker, legacy_validator):
        pid_map = legacy_validator(actor, worker)
        identities = []
        for rank, identity in enumerate(actor):
            identities.append({
                "actor_rank": rank,
                "worker_pid": identity["pid"],
                "physical_gpu_id": identity["physical_gpu_id"],
            })
        self.bindings = telemetry.canonical_actor_bindings_v66(identities)
        if telemetry.binding_by_gpu_v66(self.bindings) != {
            gpu: {
                "actor_rank": next(
                    item["actor_rank"] for item in self.bindings
                    if item["physical_gpu_id"] == gpu
                ),
                "worker_pid": pid,
                "physical_gpu_id": gpu,
            }
            for gpu, pid in pid_map.items()
        }:
            raise RuntimeError("v66d captured binding differs from legacy identity")
        return pid_map

    def register_assignment(self, assignment):
        work_id = telemetry.work_id_v66d(assignment)
        compact = {
            key: assignment[key]
            for key in (
                "wave_index", "engine_rank", "direction_seed", "sign",
                "pair_id", "evaluation_contract_sha256",
            )
        }
        prior = self.assignments.setdefault(work_id, compact)
        if prior != compact:
            raise RuntimeError("v66d work ID collision")
        return work_id, compact

    def validate_begin(self, receipt, assignment):
        work_id, compact = self.register_assignment(assignment)
        if self.bindings is None:
            raise RuntimeError("v66d bindings were not captured before work")
        binding = next(
            item for item in self.bindings
            if item["actor_rank"] == compact["engine_rank"]
        )
        required = {
            "schema", "work_id", "wave_index", "engine_rank",
            "direction_seed", "sign", "pair_id",
            "evaluation_contract_sha256", "worker_pid", "physical_gpu_id",
            "cuda_event_start_recorded",
            "output_or_token_cardinality_observed",
        }
        if (
            not isinstance(receipt, dict)
            or set(receipt) != required
            or receipt["schema"] != "eggroll-es-actor-cuda-work-begin-v66d"
            or receipt["work_id"] != work_id
            or any(receipt[key] != compact[key] for key in compact)
            or receipt["worker_pid"] != binding["worker_pid"]
            or receipt["physical_gpu_id"] != binding["physical_gpu_id"]
            or receipt["cuda_event_start_recorded"] is not True
            or receipt["output_or_token_cardinality_observed"] is not False
        ):
            raise RuntimeError("v66d actor CUDA begin receipt changed")

    def record_receipt(self, receipt, assignment):
        work_id, _ = self.register_assignment(assignment)
        if (
            not isinstance(receipt, dict)
            or receipt.get("work_id") != work_id
            or telemetry.seal_actor_work_receipt_v66d(receipt) != receipt
            or work_id in self.receipts
        ):
            raise RuntimeError("v66d actor CUDA end receipt changed or duplicated")
        self.receipts[work_id] = receipt
        payload = (
            json.dumps(
                receipt,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ) + "\n"
        ).encode("ascii")
        mode = "xb" if not GPU_WORK_LOG.exists() else "ab"
        with GPU_WORK_LOG.open(mode) as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())

    def ensure_summary(self, path, expected_pids):
        if self.summary is not None:
            return self.summary
        if self.bindings is None:
            raise RuntimeError("v66d bindings are unavailable")
        by_gpu = telemetry.binding_by_gpu_v66(self.bindings)
        if expected_pids != {
            gpu: item["worker_pid"] for gpu, item in by_gpu.items()
        }:
            raise RuntimeError("v66d summary PID map changed")
        if not GPU_WORK_LOG.is_file():
            raise RuntimeError("v66d actor CUDA work log is absent")
        logged = [
            json.loads(line)
            for line in GPU_WORK_LOG.read_text(encoding="ascii").splitlines()
            if line
        ]
        if logged != list(self.receipts.values()):
            raise RuntimeError("v66d durable actor work log differs from memory")
        summary = telemetry.summarize_mirrored_waves_v66d(
            telemetry.read_samples_v66(path),
            self.bindings,
            list(self.assignments.values()),
            logged,
            expected_request_outputs=64,
        )
        summary["actor_cuda_work_log"] = {
            "path": str(GPU_WORK_LOG),
            "file_sha256": v66.file_sha256_v66(GPU_WORK_LOG),
            "rows": len(logged),
        }
        self.summary = summary
        return summary

    def legacy_gpu_summary(self, path, expected_pids):
        summary = self.ensure_summary(path, expected_pids)
        by_gpu = {}
        for gpu in telemetry.GPU_IDS_V66:
            rows = [
                wave[str(gpu)] for wave in summary["by_wave"].values()
            ]
            by_gpu[str(gpu)] = {
                "expected_pid": rows[0]["worker_pid"],
                "samples": sum(item["acknowledged_phase_samples"] for item in rows),
                "resident_samples": sum(item["resident_phase_samples"] for item in rows),
                "positive_samples": sum(
                    item["positive_nvml_phase_samples"] for item in rows
                ),
                "peak_utilization_percent": max(
                    item["peak_gpu_utilization_percent"] for item in rows
                ),
                "peak_memory_used_mib": max(
                    item["peak_memory_used_mib"] for item in rows
                ),
                "actor_cuda_work_receipts": len(rows),
                "useful_work_attributed": True,
            }
        return {
            "all_four_attributed_positive": True,
            "attribution_contract": (
                "exact-phase four-row handshake plus actor CUDA event and output counts"
            ),
            "by_gpu": by_gpu,
        }


_ACTIVE_CONTEXT_V66D = None


class RayMirroredCallbacksV66D(v66._RayMirroredCallbacksV66):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if _ACTIVE_CONTEXT_V66D is None:
            raise RuntimeError("v66d live context is absent")
        self.context = _ACTIVE_CONTEXT_V66D

    def submit_materialize(self, assignment):
        self.context.register_assignment(assignment)
        return super().submit_materialize(assignment)

    def submit_evaluate(self, assignment, payload):
        if payload["contract"]["evaluation_contract_sha256"] != (
            assignment["evaluation_contract_sha256"]
        ):
            raise RuntimeError("v66d evaluation payload changed at submission")
        self.context.register_assignment(assignment)
        self.phase.value = (
            f"mirrored_wave_{assignment['wave_index']}_generation_all_actors"
        )
        engine = self.trainer.engines[assignment["engine_rank"]]
        begin_ref = engine.collective_rpc.remote(
            "begin_actor_gpu_work_v66d", args=(assignment,)
        )
        begin_values = self.trainer._resolve([begin_ref])
        if (
            len(begin_values) != 1
            or not isinstance(begin_values[0], list)
            or len(begin_values[0]) != 1
        ):
            raise RuntimeError("v66d incomplete actor CUDA begin RPC")
        self.context.validate_begin(begin_values[0][0], assignment)
        generation_ref = engine.generate.remote(
            payload["prompts"],
            self.sampling,
            use_tqdm=False,
            lora_request=self.prior._lora_request(),
        )
        return RuntimeGenerationHandleV66D(generation_ref, dict(assignment))

    @staticmethod
    def _output_cardinality(value):
        if not isinstance(value, list):
            raise RuntimeError("v66d generation output batch changed")
        try:
            result = {
                "request_outputs": len(value),
                "samples": sum(len(output.outputs) for output in value),
                "generated_tokens": sum(
                    len(sample.token_ids)
                    for output in value for sample in output.outputs
                ),
                "prompt_tokens": sum(
                    len(output.prompt_token_ids) for output in value
                ),
            }
        except (AttributeError, TypeError) as error:
            raise RuntimeError("v66d generation output cardinality changed") from error
        if (
            result["request_outputs"] != 64
            or result["samples"] != 64
            or result["generated_tokens"] != 64
            or result["prompt_tokens"] < 64
        ):
            raise RuntimeError("v66d generation output/token coverage changed")
        return result

    def resolve_one(self, handle):
        if not isinstance(handle, RuntimeGenerationHandleV66D):
            return super().resolve_one(handle)
        values = self.trainer._resolve([handle.ref])
        if len(values) != 1:
            raise RuntimeError("v66d Ray generation handle coverage changed")
        value = values[0]
        cardinality = self._output_cardinality(value)
        rank = handle.assignment["engine_rank"]
        end_ref = self.trainer.engines[rank].collective_rpc.remote(
            "end_actor_gpu_work_v66d",
            args=(handle.assignment, cardinality),
        )
        end_values = self.trainer._resolve([end_ref])
        if (
            len(end_values) != 1
            or not isinstance(end_values[0], list)
            or len(end_values[0]) != 1
        ):
            raise RuntimeError("v66d incomplete actor CUDA end RPC")
        self.context.record_receipt(end_values[0][0], handle.assignment)
        dense = self.prior.anchor_v4.score_gold_answer_outputs_v4(
            self.prepared["dense_items"], value
        )
        reward = float(dense["mean_example_mean_logprob"])
        if (
            dense["example_count"] != 64
            or dense["answer_token_count"]
            != self.prepared["answer_token_count_per_candidate"]
            or not math.isfinite(reward)
        ):
            raise RuntimeError("v66d answer-logprob reward coverage changed")
        return reward


@contextmanager
def patched_live_v66d():
    """Install V66d hooks only for the authorized live process."""
    import run_lora_topology_probe_v40a as v40a

    global _ACTIVE_CONTEXT_V66D
    context = LiveContextV66D()
    _ACTIVE_CONTEXT_V66D = context
    v66_names = {
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "GPU_LOG": GPU_LOG,
        "POPULATION": POPULATION,
        "UPDATE": UPDATE,
        "REPORT": REPORT,
        "FAILURE": FAILURE,
        "WORKER_EXTENSION_V66": WORKER_EXTENSION_V66D,
        "_RayMirroredCallbacksV66": RayMirroredCallbacksV66D,
    }
    saved_v66 = {name: getattr(v66, name) for name in v66_names}
    for name, value in v66_names.items():
        setattr(v66, name, value)
    saved_v40a = {
        "Phase": v40a.Phase,
        "validate_identities": v40a.validate_identities,
        "monitor_gpus": v40a.monitor_gpus,
        "summarize_gpu": v40a.summarize_gpu,
        "atomic_json": v40a.atomic_json,
    }
    original_validate = v40a.validate_identities
    original_atomic_json = v40a.atomic_json
    original_write = v66._write_self_hashed_v66
    original_wave_summary = v66._gpu_wave_summary_v66

    def validate_identities(actor, worker):
        return context.capture_bindings(actor, worker, original_validate)

    def monitor_gpus(stop, phase, expected_pids, path, failures):
        if context.bindings is None:
            raise RuntimeError("v66d monitor started without actor bindings")
        expected = {
            gpu: item["worker_pid"]
            for gpu, item in telemetry.binding_by_gpu_v66(
                context.bindings
            ).items()
        }
        if expected != expected_pids:
            raise RuntimeError("v66d monitor PID map changed")
        telemetry.monitor_gpus_v66(
            stop,
            phase,
            context.bindings,
            path,
            failures,
            context.ready,
            sample_interval_seconds=0.25,
        )

    def summarize_gpu(path, expected_pids):
        return context.legacy_gpu_summary(path, expected_pids)

    def summarize_waves(path, expected_pids):
        return context.ensure_summary(path, expected_pids)

    def atomic_json(path, value):
        if Path(path).resolve() == ATTEMPT:
            compact = {
                key: item for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }
            compact["schema"] = "v66d-mirrored-qwen36-calibration-attempt"
            compact["gpu_work_attribution"] = (
                "four-row phase handshake plus actor CUDA event/output receipt"
            )
            value = v40a.self_hashed(compact)
        return original_atomic_json(path, value)

    def write_v66d(path, value):
        value = dict(value)
        resolved = Path(path).resolve()
        if resolved == ATTEMPT:
            value["schema"] = "v66d-mirrored-qwen36-calibration-attempt"
            value["gpu_work_attribution"] = (
                "four-row phase handshake plus actor CUDA event/output receipt"
            )
        elif resolved == REPORT:
            value["schema"] = "v66d-mirrored-crn-qwen36-calibration-report"
            value["status"] = (
                "complete_nonzero_train_only_no_commit_actor_attributed"
            )
            value["beads"] = [
                "specialist-0j5.2", "specialist-0j5.12", "specialist-nen.25",
            ]
            value["actor_cuda_work_log"] = context.ensure_summary(
                GPU_LOG,
                {
                    item["physical_gpu_id"]: item["worker_pid"]
                    for item in context.bindings
                },
            )["actor_cuda_work_log"]
        elif resolved == FAILURE:
            value["schema"] = "v66d-mirrored-qwen36-calibration-failure"
            value["partial_actor_cuda_work_log"] = {
                "path": str(GPU_WORK_LOG),
                "exists": GPU_WORK_LOG.exists(),
                "rows": len(context.receipts),
            }
        return original_write(path, value)

    v40a.Phase = CompatiblePhaseHandshakeV66D
    v40a.validate_identities = validate_identities
    v40a.monitor_gpus = monitor_gpus
    v40a.summarize_gpu = summarize_gpu
    v40a.atomic_json = atomic_json
    v66._gpu_wave_summary_v66 = summarize_waves
    v66._write_self_hashed_v66 = write_v66d
    try:
        yield context
    finally:
        v66._write_self_hashed_v66 = original_write
        v66._gpu_wave_summary_v66 = original_wave_summary
        for name, value in saved_v40a.items():
            setattr(v40a, name, value)
        for name, value in saved_v66.items():
            setattr(v66, name, value)
        _ACTIVE_CONTEXT_V66D = None


def parser_v66d():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv=None):
    args = parser_v66d().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("v66d requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v66d(args)
    worker_contract = validate_lora_worker_contract_v66d()
    if args.dry_run:
        recipe = preregistration["fixed_recipe"]
        print(json.dumps({
            "schema": preregistration["schema"],
            "model": "Qwen3.6-35B-A3B",
            "four_tp1_engines": True,
            "direction_count": len(recipe["direction_seeds"]),
            "signed_population_size": 2 * len(recipe["direction_seeds"]),
            "train_only_rows_per_candidate": 64,
            "expected_artifacts": artifacts_v66d(),
            "worker_contract": worker_contract,
            "phase_transition_requires_four_gpu_sample_ack": True,
            "actor_cuda_event_and_output_receipt_per_candidate": True,
            "nvml_positive_sample_required_for_short_phase": False,
            "train_semantics_model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_or_promotion_authorized": False,
        }, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != v66.REQUIRED_PYTHON_V66:
        raise RuntimeError(
            f"v66d requires {v66.REQUIRED_PYTHON_V66}; observed {sys.executable}"
        )
    with patched_live_v66d():
        return v66.execute_v66(preregistration, args)


if __name__ == "__main__":
    raise SystemExit(main())
