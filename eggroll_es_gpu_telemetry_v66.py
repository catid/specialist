#!/usr/bin/env python3
"""Reusable four-GPU phase telemetry for specialist ES experiments.

The monitor binds controller actor rank, intrinsic worker PID, and physical GPU
without assuming rank equals GPU.  Phase changes use a sampling handshake so a
short operation cannot begin before the monitor has observed its label.  The
offline validator distinguishes useful GPU phases, resident CPU-bound phases,
and final idle cleanup while rejecting missing actors or foreign compute PIDs.
"""

from __future__ import annotations

import json
import hashlib
import math
import numbers
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_V66 = "eggroll-es-four-gpu-phase-telemetry-v66"
WORK_RECEIPT_SCHEMA_V66D = "eggroll-es-actor-cuda-work-receipt-v66d"
GPU_IDS_V66 = (0, 1, 2, 3)
PHASE_KINDS_V66 = {
    "useful_gpu",
    "resident_cpu_bound",
    "cleanup_idle",
}


def _exact_int_v66(value, label, *, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) < minimum
    ):
        raise RuntimeError(f"invalid {label}")
    return int(value)


def canonical_actor_bindings_v66(identities):
    """Seal any exact actor-to-GPU permutation as rank-ordered bindings."""
    if not isinstance(identities, (list, tuple)) or len(identities) != 4:
        raise RuntimeError("telemetry requires exactly four actor identities")
    rows = []
    for identity in identities:
        if not isinstance(identity, dict) or set(identity) != {
            "actor_rank", "worker_pid", "physical_gpu_id",
        }:
            raise RuntimeError("actor identity schema changed")
        rows.append({
            "actor_rank": _exact_int_v66(
                identity["actor_rank"], "actor rank",
            ),
            "worker_pid": _exact_int_v66(
                identity["worker_pid"], "worker PID", minimum=1,
            ),
            "physical_gpu_id": _exact_int_v66(
                identity["physical_gpu_id"], "physical GPU ID",
            ),
        })
    rows.sort(key=lambda row: row["actor_rank"])
    if (
        [row["actor_rank"] for row in rows] != list(GPU_IDS_V66)
        or {row["physical_gpu_id"] for row in rows} != set(GPU_IDS_V66)
        or len({row["worker_pid"] for row in rows}) != 4
    ):
        raise RuntimeError("actor rank/PID/GPU coverage is not one-to-one")
    return rows


def binding_by_gpu_v66(bindings):
    bindings = canonical_actor_bindings_v66(bindings)
    return {row["physical_gpu_id"]: row for row in bindings}


def validate_phase_contract_v66(contract):
    if not isinstance(contract, (list, tuple)) or not contract:
        raise RuntimeError("phase contract must be a nonempty list")
    result = []
    names = set()
    for item in contract:
        if not isinstance(item, dict) or set(item) != {
            "phase", "kind", "minimum_samples_per_gpu",
        }:
            raise RuntimeError("phase contract schema changed")
        phase = item["phase"]
        kind = item["kind"]
        if not isinstance(phase, str) or not phase or phase in names:
            raise RuntimeError("phase names must be unique nonempty strings")
        if kind not in PHASE_KINDS_V66:
            raise RuntimeError("phase kind is unsupported")
        names.add(phase)
        result.append({
            "phase": phase,
            "kind": kind,
            "minimum_samples_per_gpu": _exact_int_v66(
                item["minimum_samples_per_gpu"],
                "minimum samples per GPU", minimum=1,
            ),
        })
    if result[-1]["kind"] != "cleanup_idle":
        raise RuntimeError("last telemetry phase must prove cleanup idle")
    return result


class PhaseHandshakeV66:
    """Thread-safe phase label whose transitions wait for one monitor sample."""

    def __init__(self, initial="setup"):
        if not isinstance(initial, str) or not initial:
            raise ValueError("initial phase must be a nonempty string")
        self._condition = threading.Condition()
        self._phase = initial
        self._epoch = 0
        self._sampled_epoch = -1

    def snapshot(self):
        with self._condition:
            return self._phase, self._epoch

    def acknowledge(self, epoch, sampled_gpu_ids):
        epoch = _exact_int_v66(epoch, "phase epoch")
        if (
            not isinstance(sampled_gpu_ids, (list, tuple, set, frozenset))
            or set(sampled_gpu_ids) != set(GPU_IDS_V66)
            or len(sampled_gpu_ids) != len(GPU_IDS_V66)
            or any(type(gpu) is not int for gpu in sampled_gpu_ids)
        ):
            raise RuntimeError(
                "phase acknowledgement requires one complete four-GPU batch"
            )
        with self._condition:
            self._sampled_epoch = max(self._sampled_epoch, epoch)
            self._condition.notify_all()

    def transition(self, phase, *, timeout_seconds=10.0):
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a nonempty string")
        timeout_seconds = float(timeout_seconds)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
            raise ValueError("phase handshake timeout must be positive")
        with self._condition:
            self._epoch += 1
            epoch = self._epoch
            self._phase = phase
            self._condition.notify_all()
            if not self._condition.wait_for(
                lambda: self._sampled_epoch >= epoch,
                timeout=timeout_seconds,
            ):
                raise TimeoutError(
                    f"GPU monitor did not sample phase {phase!r}"
                )
        return {"phase": phase, "epoch": epoch, "sampled": True}


def _pcie_rate_v66(pynvml, handle, counter_name):
    getter = getattr(pynvml, "nvmlDeviceGetPcieThroughput", None)
    counter = getattr(pynvml, counter_name, None)
    if getter is None or counter is None:
        return None
    try:
        value = getter(handle, counter)
    except pynvml.NVMLError:
        return None
    return _exact_int_v66(value, counter_name)


def monitor_gpus_v66(
    stop,
    phase,
    bindings,
    path,
    failures,
    ready,
    *,
    sample_interval_seconds=0.25,
):
    """Sample NVML into a fresh JSONL artifact until ``stop`` is set."""
    sample_interval_seconds = float(sample_interval_seconds)
    if (
        not math.isfinite(sample_interval_seconds)
        or sample_interval_seconds <= 0.0
        or sample_interval_seconds > 1.0
    ):
        raise ValueError("GPU sample interval must be in (0, 1] seconds")
    by_gpu = binding_by_gpu_v66(bindings)
    try:
        import pynvml

        pynvml.nvmlInit()
        handles = {
            gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu)
            for gpu in GPU_IDS_V66
        }
        sequence = 0
        with Path(path).open("x", encoding="utf-8") as output:
            while not stop.is_set():
                sampled_at = datetime.now(timezone.utc).isoformat()
                monotonic_ns = time.monotonic_ns()
                current_phase, phase_epoch = phase.snapshot()
                sampled_gpus = []
                for gpu, handle in handles.items():
                    processes = pynvml.nvmlDeviceGetComputeRunningProcesses(
                        handle
                    )
                    process_memory = {
                        int(item.pid): int(item.usedGpuMemory // (1024 * 1024))
                        for item in processes
                    }
                    pids = sorted(process_memory)
                    expected_pid = by_gpu[gpu]["worker_pid"]
                    foreign = [pid for pid in pids if pid != expected_pid]
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    output.write(json.dumps({
                        "schema": SCHEMA_V66,
                        "sequence": sequence,
                        "sampled_at_utc": sampled_at,
                        "monotonic_ns": monotonic_ns,
                        "phase": current_phase,
                        "phase_epoch": phase_epoch,
                        "gpu": gpu,
                        "actor_rank": by_gpu[gpu]["actor_rank"],
                        "expected_pid": expected_pid,
                        "compute_pids": pids,
                        "process_memory_mib": process_memory,
                        "foreign_compute_pids": foreign,
                        "gpu_utilization_percent": int(utilization.gpu),
                        "memory_utilization_percent": int(utilization.memory),
                        "memory_used_mib": int(memory.used // (1024 * 1024)),
                        "memory_total_mib": int(memory.total // (1024 * 1024)),
                        "pcie_rx_kib_per_second": _pcie_rate_v66(
                            pynvml, handle, "NVML_PCIE_UTIL_RX_BYTES",
                        ),
                        "pcie_tx_kib_per_second": _pcie_rate_v66(
                            pynvml, handle, "NVML_PCIE_UTIL_TX_BYTES",
                        ),
                        "power_draw_mw": int(
                            pynvml.nvmlDeviceGetPowerUsage(handle)
                        ),
                    }, sort_keys=True) + "\n")
                    sequence += 1
                    sampled_gpus.append(gpu)
                output.flush()
                phase.acknowledge(phase_epoch, sampled_gpus)
                ready.set()
                stop.wait(sample_interval_seconds)
    except BaseException as error:
        failures.put(error)
    finally:
        ready.set()
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def read_samples_v66(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("GPU telemetry artifact is empty")
    return rows


def _validate_sample_v66(row, by_gpu):
    required = {
        "schema", "sequence", "sampled_at_utc", "monotonic_ns", "phase",
        "phase_epoch", "gpu", "actor_rank", "expected_pid", "compute_pids",
        "process_memory_mib", "foreign_compute_pids",
        "gpu_utilization_percent", "memory_utilization_percent",
        "memory_used_mib", "memory_total_mib", "pcie_rx_kib_per_second",
        "pcie_tx_kib_per_second", "power_draw_mw",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise RuntimeError("GPU telemetry sample schema changed")
    gpu = _exact_int_v66(row["gpu"], "sample GPU")
    if gpu not in by_gpu or row["schema"] != SCHEMA_V66:
        raise RuntimeError("GPU telemetry sample identity changed")
    binding = by_gpu[gpu]
    pids = row["compute_pids"]
    process_memory = row["process_memory_mib"]
    if (
        row["actor_rank"] != binding["actor_rank"]
        or row["expected_pid"] != binding["worker_pid"]
        or not isinstance(pids, list)
        or pids != sorted(set(pids))
        or any(type(pid) is not int or pid <= 0 for pid in pids)
        or not isinstance(process_memory, dict)
        or {int(pid) for pid in process_memory} != set(pids)
        or row["foreign_compute_pids"]
        != [pid for pid in pids if pid != binding["worker_pid"]]
        or row["foreign_compute_pids"]
    ):
        raise RuntimeError("GPU telemetry PID attribution failed")
    for key in (
        "sequence", "monotonic_ns", "phase_epoch",
        "gpu_utilization_percent", "memory_utilization_percent",
        "memory_used_mib", "memory_total_mib", "power_draw_mw",
    ):
        _exact_int_v66(row[key], key)
    if (
        row["gpu_utilization_percent"] > 100
        or row["memory_utilization_percent"] > 100
        or row["memory_used_mib"] > row["memory_total_mib"]
        or not isinstance(row["phase"], str)
        or not row["phase"]
    ):
        raise RuntimeError("GPU telemetry numeric or phase value changed")
    for key in ("pcie_rx_kib_per_second", "pcie_tx_kib_per_second"):
        if row[key] is not None:
            _exact_int_v66(row[key], key)
    return row


def _integrated_pcie_bytes_v66(rows, key):
    supported = [row for row in rows if row[key] is not None]
    if len(supported) < 2:
        return None
    total = 0.0
    for previous, current in zip(supported, supported[1:]):
        elapsed = max(
            0.0,
            (current["monotonic_ns"] - previous["monotonic_ns"]) / 1e9,
        )
        total += previous[key] * 1024.0 * elapsed
    return int(total)


def summarize_phases_v66(
    rows,
    bindings,
    phase_contract,
    *,
    idle_memory_limit_mib=16,
):
    """Validate useful work, resident CPU phases, and final cleanup."""
    by_gpu = binding_by_gpu_v66(bindings)
    contract = validate_phase_contract_v66(phase_contract)
    idle_memory_limit_mib = _exact_int_v66(
        idle_memory_limit_mib, "idle memory limit MiB",
    )
    rows = [_validate_sample_v66(row, by_gpu) for row in rows]
    known = {item["phase"] for item in contract}
    if any(row["phase"] not in known for row in rows):
        raise RuntimeError("GPU telemetry contains an undeclared phase")
    by_phase = {}
    for item in contract:
        phase = item["phase"]
        kind = item["kind"]
        per_gpu = {}
        for gpu in GPU_IDS_V66:
            selected = [
                row for row in rows
                if row["phase"] == phase and row["gpu"] == gpu
            ]
            if len(selected) < item["minimum_samples_per_gpu"]:
                raise RuntimeError(
                    f"GPU {gpu} has insufficient samples for phase {phase!r}"
                )
            expected_pid = by_gpu[gpu]["worker_pid"]
            resident = [
                row for row in selected
                if expected_pid in row["compute_pids"]
            ]
            if kind == "cleanup_idle":
                if any(row["compute_pids"] for row in selected) or any(
                    row["memory_used_mib"] > idle_memory_limit_mib
                    for row in selected
                ):
                    raise RuntimeError(
                        f"GPU {gpu} did not become idle in phase {phase!r}"
                    )
            elif not resident:
                raise RuntimeError(
                    f"GPU {gpu} lost its expected actor in phase {phase!r}"
                )
            useful = [
                row for row in resident
                if row["gpu_utilization_percent"] > 0
                or row["memory_utilization_percent"] > 0
                or (row["pcie_rx_kib_per_second"] or 0) > 0
                or (row["pcie_tx_kib_per_second"] or 0) > 0
            ]
            if kind == "useful_gpu" and not useful:
                raise RuntimeError(
                    f"GPU {gpu} lacked attributed useful work in phase {phase!r}"
                )
            per_gpu[str(gpu)] = {
                "actor_rank": by_gpu[gpu]["actor_rank"],
                "expected_pid": expected_pid,
                "samples": len(selected),
                "resident_samples": len(resident),
                "useful_samples": len(useful),
                "peak_gpu_utilization_percent": max(
                    row["gpu_utilization_percent"] for row in selected
                ),
                "peak_memory_utilization_percent": max(
                    row["memory_utilization_percent"] for row in selected
                ),
                "peak_memory_used_mib": max(
                    row["memory_used_mib"] for row in selected
                ),
                "integrated_pcie_rx_bytes": _integrated_pcie_bytes_v66(
                    selected, "pcie_rx_kib_per_second",
                ),
                "integrated_pcie_tx_bytes": _integrated_pcie_bytes_v66(
                    selected, "pcie_tx_kib_per_second",
                ),
            }
        by_phase[phase] = {"kind": kind, "by_gpu": per_gpu}
    return {
        "schema": "eggroll-es-four-gpu-phase-summary-v66",
        "passed": True,
        "all_four_gpus_bound_one_to_one": True,
        "all_useful_phases_positive_on_all_four_gpus": True,
        "all_resident_cpu_bound_phases_retained_all_actors": True,
        "final_cleanup_idle_on_all_four_gpus": True,
        "foreign_compute_process_observations": 0,
        "bindings": canonical_actor_bindings_v66(bindings),
        "phase_contract": contract,
        "by_phase": by_phase,
    }


def _canonical_sha256_v66d(value):
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def work_id_v66d(assignment):
    """Return the controller/worker shared identity for one signed evaluation."""
    if not isinstance(assignment, dict):
        raise RuntimeError("v66d work assignment must be a dictionary")
    required = {
        "wave_index", "engine_rank", "direction_seed", "sign", "pair_id",
        "evaluation_contract_sha256",
    }
    if not required.issubset(assignment):
        raise RuntimeError("v66d work assignment is incomplete")
    coordinate = {
        "wave_index": _exact_int_v66(
            assignment["wave_index"], "work wave index",
        ),
        "engine_rank": _exact_int_v66(
            assignment["engine_rank"], "work engine rank",
        ),
        "direction_seed": _exact_int_v66(
            assignment["direction_seed"], "work direction seed",
        ),
        "sign": assignment["sign"],
        "pair_id": assignment["pair_id"],
        "evaluation_contract_sha256": assignment[
            "evaluation_contract_sha256"
        ],
    }
    if (
        type(coordinate["sign"]) is not int
        or coordinate["sign"] not in (-1, 1)
        or type(coordinate["pair_id"]) is not str
        or len(coordinate["pair_id"]) != 64
        or type(coordinate["evaluation_contract_sha256"]) is not str
        or len(coordinate["evaluation_contract_sha256"]) != 64
    ):
        raise RuntimeError("v66d work assignment coordinate changed")
    return _canonical_sha256_v66d(coordinate)


def seal_actor_work_receipt_v66d(receipt):
    """Self-hash one actor receipt without mutating the caller's object."""
    if not isinstance(receipt, dict):
        raise RuntimeError("v66d actor work receipt must be a dictionary")
    result = dict(receipt)
    result.pop("receipt_sha256", None)
    result["receipt_sha256"] = _canonical_sha256_v66d(result)
    return result


def _expected_work_v66d(assignments):
    if not isinstance(assignments, (list, tuple)) or not assignments:
        raise RuntimeError("v66d expected work assignments are empty")
    expected = {}
    by_wave = {}
    counts_by_wave = {}
    for assignment in assignments:
        work_id = work_id_v66d(assignment)
        if work_id in expected:
            raise RuntimeError("v66d expected work assignment is duplicated")
        compact = {
            key: assignment[key]
            for key in (
                "wave_index", "engine_rank", "direction_seed", "sign",
                "pair_id", "evaluation_contract_sha256",
            )
        }
        expected[work_id] = compact
        by_wave.setdefault(compact["wave_index"], set()).add(
            compact["engine_rank"]
        )
        counts_by_wave[compact["wave_index"]] = (
            counts_by_wave.get(compact["wave_index"], 0) + 1
        )
    if (
        not by_wave
        or sorted(by_wave) != list(range(len(by_wave)))
        or any(ranks != set(GPU_IDS_V66) for ranks in by_wave.values())
        or any(count != 4 for count in counts_by_wave.values())
    ):
        raise RuntimeError("v66d expected work lacks a complete four-actor wave")
    return expected, by_wave


def validate_actor_work_receipts_v66d(
    receipts,
    bindings,
    assignments,
    *,
    expected_request_outputs=64,
):
    """Validate exact per-actor CUDA-event/output evidence for every candidate."""
    bindings = canonical_actor_bindings_v66(bindings)
    by_rank = {item["actor_rank"]: item for item in bindings}
    expected, _ = _expected_work_v66d(assignments)
    expected_request_outputs = _exact_int_v66(
        expected_request_outputs, "expected request outputs", minimum=1,
    )
    if not isinstance(receipts, (list, tuple)):
        raise RuntimeError("v66d actor work receipts must be a list")
    observed = {}
    required = {
        "schema", "receipt_sha256", "work_id", "wave_index", "engine_rank",
        "direction_seed", "sign", "pair_id", "evaluation_contract_sha256",
        "worker_pid", "physical_gpu_id", "cuda_event", "output_cardinality",
    }
    for receipt in receipts:
        if not isinstance(receipt, dict) or set(receipt) != required:
            raise RuntimeError("v66d actor work receipt schema changed")
        compact = dict(receipt)
        observed_sha = compact.pop("receipt_sha256")
        if (
            receipt["schema"] != WORK_RECEIPT_SCHEMA_V66D
            or type(observed_sha) is not str
            or observed_sha != _canonical_sha256_v66d(compact)
        ):
            raise RuntimeError("v66d actor work receipt hash changed")
        work_id = receipt["work_id"]
        if work_id not in expected or work_id in observed:
            raise RuntimeError("v66d actor work receipt coverage changed")
        assignment = expected[work_id]
        if any(
            receipt[key] != assignment[key]
            for key in (
                "wave_index", "engine_rank", "direction_seed", "sign",
                "pair_id", "evaluation_contract_sha256",
            )
        ):
            raise RuntimeError("v66d actor work receipt coordinate changed")
        rank = _exact_int_v66(receipt["engine_rank"], "receipt engine rank")
        binding = by_rank.get(rank)
        if (
            binding is None
            or receipt["worker_pid"] != binding["worker_pid"]
            or receipt["physical_gpu_id"] != binding["physical_gpu_id"]
        ):
            raise RuntimeError("v66d actor work receipt PID/GPU binding changed")
        cuda_event = receipt["cuda_event"]
        if not isinstance(cuda_event, dict) or set(cuda_event) != {
            "backend", "start_recorded", "end_recorded", "end_synchronized",
            "elapsed_ms", "worker_monotonic_elapsed_ns",
        }:
            raise RuntimeError("v66d CUDA event receipt schema changed")
        elapsed_ms = cuda_event["elapsed_ms"]
        if (
            cuda_event["backend"] != "torch.cuda.Event"
            or cuda_event["start_recorded"] is not True
            or cuda_event["end_recorded"] is not True
            or cuda_event["end_synchronized"] is not True
            or isinstance(elapsed_ms, bool)
            or not isinstance(elapsed_ms, numbers.Real)
            or not math.isfinite(float(elapsed_ms))
            or float(elapsed_ms) <= 0.0
            or _exact_int_v66(
                cuda_event["worker_monotonic_elapsed_ns"],
                "worker monotonic elapsed ns", minimum=1,
            ) <= 0
        ):
            raise RuntimeError("v66d CUDA event did not prove completed work")
        cardinality = receipt["output_cardinality"]
        if not isinstance(cardinality, dict) or set(cardinality) != {
            "request_outputs", "samples", "generated_tokens", "prompt_tokens",
        }:
            raise RuntimeError("v66d output cardinality receipt schema changed")
        request_outputs = _exact_int_v66(
            cardinality["request_outputs"], "request output count", minimum=1,
        )
        samples = _exact_int_v66(
            cardinality["samples"], "sample count", minimum=1,
        )
        generated_tokens = _exact_int_v66(
            cardinality["generated_tokens"], "generated token count", minimum=1,
        )
        prompt_tokens = _exact_int_v66(
            cardinality["prompt_tokens"], "prompt token count", minimum=1,
        )
        if (
            request_outputs != expected_request_outputs
            or samples != expected_request_outputs
            or generated_tokens != samples
            or prompt_tokens < request_outputs
        ):
            raise RuntimeError(
                "v66d resident actor lacked exact generation cardinality"
            )
        observed[work_id] = receipt
    if set(observed) != set(expected):
        raise RuntimeError("v66d actor work receipt matrix is incomplete")
    return [observed[work_id] for work_id in sorted(observed)]


def summarize_mirrored_waves_v66d(
    rows,
    bindings,
    assignments,
    receipts,
    *,
    expected_request_outputs=64,
):
    """Combine exact-label NVML rows with actor-side CUDA work receipts."""
    bindings = canonical_actor_bindings_v66(bindings)
    by_gpu = binding_by_gpu_v66(bindings)
    expected, waves = _expected_work_v66d(assignments)
    receipts = validate_actor_work_receipts_v66d(
        receipts,
        bindings,
        assignments,
        expected_request_outputs=expected_request_outputs,
    )
    receipt_by_work = {item["work_id"]: item for item in receipts}
    rows = [_validate_sample_v66(row, by_gpu) for row in rows]
    by_wave = {}
    for wave_index in sorted(waves):
        phase_name = f"mirrored_wave_{wave_index}_generation_all_actors"
        phase_rows = [row for row in rows if row["phase"] == phase_name]
        if len({row["phase_epoch"] for row in phase_rows}) != 1:
            raise RuntimeError(
                f"v66d phase {phase_name!r} lacks one acknowledged epoch"
            )
        sampled_batches = {}
        for row in phase_rows:
            sampled_batches.setdefault(row["monotonic_ns"], []).append(row)
        acknowledged_batches = [
            batch for batch in sampled_batches.values()
            if len(batch) == 4
            and {row["gpu"] for row in batch} == set(GPU_IDS_V66)
        ]
        if not acknowledged_batches:
            raise RuntimeError(
                f"v66d phase {phase_name!r} lacks one complete four-GPU sample batch"
            )
        per_gpu = {}
        assignments_for_wave = {
            item["engine_rank"]: (work_id, item)
            for work_id, item in expected.items()
            if item["wave_index"] == wave_index
        }
        for rank in GPU_IDS_V66:
            binding = next(
                item for item in bindings if item["actor_rank"] == rank
            )
            gpu = binding["physical_gpu_id"]
            selected = [row for row in phase_rows if row["gpu"] == gpu]
            resident = [
                row for row in selected
                if binding["worker_pid"] in row["compute_pids"]
            ]
            if not selected or not resident:
                raise RuntimeError(
                    f"v66d wave {wave_index} GPU {gpu} was not sampled resident"
                )
            work_id, _ = assignments_for_wave[rank]
            receipt = receipt_by_work[work_id]
            positive = [
                row for row in resident
                if row["gpu_utilization_percent"] > 0
                or row["memory_utilization_percent"] > 0
                or (row["pcie_rx_kib_per_second"] or 0) > 0
                or (row["pcie_tx_kib_per_second"] or 0) > 0
            ]
            per_gpu[str(gpu)] = {
                "actor_rank": rank,
                "worker_pid": binding["worker_pid"],
                "acknowledged_phase_samples": len(selected),
                "resident_phase_samples": len(resident),
                "positive_nvml_phase_samples": len(positive),
                "peak_gpu_utilization_percent": max(
                    row["gpu_utilization_percent"] for row in selected
                ),
                "peak_memory_used_mib": max(
                    row["memory_used_mib"] for row in selected
                ),
                "actor_cuda_work_receipt_sha256": receipt[
                    "receipt_sha256"
                ],
                "cuda_event_elapsed_ms": receipt["cuda_event"][
                    "elapsed_ms"
                ],
                "output_cardinality": receipt["output_cardinality"],
                "useful_work_attributed": True,
                "attribution_sources": (
                    ["actor_cuda_event_and_output_cardinality", "nvml"]
                    if positive
                    else ["actor_cuda_event_and_output_cardinality"]
                ),
            }
        by_wave[str(wave_index)] = per_gpu
    return {
        "schema": "eggroll-es-mirrored-four-gpu-work-summary-v66d",
        "passed": True,
        "waves": len(waves),
        "signed_candidates": len(expected),
        "sampling_handshake_acknowledged_all_four_rows_each_wave": True,
        "actor_cuda_event_receipt_for_every_candidate": True,
        "actor_pid_gpu_binding_exact": True,
        "nonzero_output_and_token_cardinality_every_candidate": True,
        "all_four_physical_gpus_useful_in_every_wave": True,
        "foreign_compute_process_observations": 0,
        "bindings": bindings,
        "by_wave": by_wave,
    }
