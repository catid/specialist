#!/usr/bin/env python3
"""V66 mirrored LoRA worker with actor-attributed CUDA work receipts.

The start event is recorded only after the mirrored candidate is active.  The
end event is recorded only after controller-observed generation outputs have
been counted.  Receipts bind the signed candidate to the worker's intrinsic
PID and physical GPU, so a resident but idle process cannot satisfy the V66d
useful-work gate merely because device-wide NVML utilization is positive.
"""

from __future__ import annotations

import math
import time

import torch

from eggroll_es_gpu_telemetry_v66 import (
    WORK_RECEIPT_SCHEMA_V66D,
    seal_actor_work_receipt_v66d,
    work_id_v66d,
)
from eggroll_es_worker_lora_v66 import LoRAAdapterStateWorkerExtensionV66


def _coordinate_v66d(assignment):
    work_id = work_id_v66d(assignment)
    return work_id, {
        key: assignment[key]
        for key in (
            "wave_index", "engine_rank", "direction_seed", "sign",
            "pair_id", "evaluation_contract_sha256",
        )
    }


def _cardinality_v66d(value):
    required = {
        "request_outputs", "samples", "generated_tokens", "prompt_tokens",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("v66d generation cardinality schema changed")
    result = {}
    for key in sorted(required):
        item = value[key]
        if type(item) is not int or item <= 0:
            raise RuntimeError("v66d generation cardinality was not positive")
        result[key] = item
    if (
        result["samples"] != result["request_outputs"]
        or result["generated_tokens"] != result["samples"]
        or result["prompt_tokens"] < result["request_outputs"]
    ):
        raise RuntimeError("v66d generation cardinality did not prove work")
    return result


class LoRAAdapterStateWorkerExtensionV66D(
    LoRAAdapterStateWorkerExtensionV66,
):
    """Add start/end CUDA-event receipts around each actor generation."""

    def _device_v66d(self):
        device = torch.device(getattr(self, "device", "cuda:0"))
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("v66d actor CUDA work receipt requires CUDA")
        return device

    def begin_actor_gpu_work_v66d(self, assignment):
        self._require_not_poisoned_v66()
        if getattr(self, "_v66d_active_gpu_work", None) is not None:
            raise RuntimeError("v66d actor already has an active GPU work event")
        work_id, coordinate = _coordinate_v66d(assignment)
        active = getattr(self, "_v66_candidate_transaction", None)
        if (
            not isinstance(active, dict)
            or active.get("phase") != "candidate_active"
            or any(
                active.get(key) != coordinate[key]
                for key in (
                    "direction_seed", "sign", "pair_id",
                    "evaluation_contract_sha256",
                )
            )
        ):
            raise RuntimeError(
                "v66d CUDA event did not begin on the active signed candidate"
            )
        identity = self._intrinsic_worker_identity_v65b()
        device = self._device_v66d()
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        started_ns = time.monotonic_ns()
        self._v66d_active_gpu_work = {
            "work_id": work_id,
            "coordinate": coordinate,
            "identity": identity,
            "device": device,
            "start_event": start,
            "started_ns": started_ns,
        }
        return {
            "schema": "eggroll-es-actor-cuda-work-begin-v66d",
            "work_id": work_id,
            **coordinate,
            "worker_pid": identity["worker_pid"],
            "physical_gpu_id": identity["worker_physical_gpu_id"],
            "cuda_event_start_recorded": True,
            "output_or_token_cardinality_observed": False,
        }

    def end_actor_gpu_work_v66d(self, assignment, output_cardinality):
        self._require_not_poisoned_v66()
        work_id, coordinate = _coordinate_v66d(assignment)
        active = getattr(self, "_v66d_active_gpu_work", None)
        if (
            not isinstance(active, dict)
            or active.get("work_id") != work_id
            or active.get("coordinate") != coordinate
        ):
            raise RuntimeError("v66d actor GPU work event is missing or changed")
        cardinality = _cardinality_v66d(output_cardinality)
        end = None
        ended_ns = None
        try:
            # vLLM may use streams other than the worker's current stream.
            # Device synchronization makes the end timestamp follow all of
            # this actor's submitted generation kernels.
            torch.cuda.synchronize(active["device"])
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            end.synchronize()
            ended_ns = time.monotonic_ns()
            elapsed_ms = float(active["start_event"].elapsed_time(end))
            elapsed_ns = ended_ns - active["started_ns"]
            if (
                not math.isfinite(elapsed_ms)
                or elapsed_ms <= 0.0
                or elapsed_ns <= 0
            ):
                raise RuntimeError(
                    "v66d CUDA event elapsed time did not prove completed work"
                )
            identity = self._intrinsic_worker_identity_v65b()
            if identity != active["identity"]:
                raise RuntimeError("v66d worker identity changed during generation")
            receipt = {
                "schema": WORK_RECEIPT_SCHEMA_V66D,
                "work_id": work_id,
                **coordinate,
                "worker_pid": identity["worker_pid"],
                "physical_gpu_id": identity["worker_physical_gpu_id"],
                "cuda_event": {
                    "backend": "torch.cuda.Event",
                    "start_recorded": True,
                    "end_recorded": True,
                    "end_synchronized": True,
                    "elapsed_ms": elapsed_ms,
                    "worker_monotonic_elapsed_ns": elapsed_ns,
                },
                "output_cardinality": cardinality,
            }
            return seal_actor_work_receipt_v66d(receipt)
        finally:
            self._v66d_active_gpu_work = None

    def restore_mirrored_adapter_v66(
        self,
        expected_master_sha256,
        reason,
        expected_pair_id=None,
    ):
        # A failed/unknown generation RPC can strand only the telemetry event,
        # never the exact-master restoration.  Clear it before delegating to
        # the inherited unconditional reconstruction path.  The controller's
        # missing-receipt gate will still fail the run closed.
        orphaned = getattr(self, "_v66d_active_gpu_work", None) is not None
        if orphaned:
            try:
                torch.cuda.synchronize(
                    self._v66d_active_gpu_work["device"]
                )
            finally:
                self._v66d_active_gpu_work = None
        receipt = super().restore_mirrored_adapter_v66(
            expected_master_sha256,
            reason,
            expected_pair_id,
        )
        return {
            **receipt,
            "orphaned_v66d_gpu_work_event_cleared": orphaned,
        }

    def mirrored_adapter_state_certificate_v66(self):
        if getattr(self, "_v66d_active_gpu_work", None) is not None:
            raise RuntimeError("v66d final certificate found active GPU work")
        return {
            **super().mirrored_adapter_state_certificate_v66(),
            "v66d_gpu_work_event_quiescent": True,
        }
