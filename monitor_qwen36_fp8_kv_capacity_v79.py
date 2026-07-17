#!/usr/bin/env python3
"""NVML monitor for the four-process V79 data-free preflight.

This process creates no CUDA context.  It samples at 2 Hz, attributes every
GPU compute PID to the corresponding launched actor through the Linux parent
process chain, records HBM utilization as a percentage (never inferred bytes),
and records the NVML PCIe RX/TX throughput counters needed for sampled byte
integration.  Unsupported PCIe counters fail closed when ``--require-pcie``
is selected by the sealed launcher.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_V79 = "v79-four-gpu-kv-capacity-telemetry"
GPU_IDS_V79 = (0, 1, 2, 3)
POST_EXIT_MEMORY_USED_MIB_MAX_V79 = 4


def read_actor_pids_v79(path: Path) -> dict[int, int]:
    rows: dict[int, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split(",")
        if len(fields) != 2:
            raise RuntimeError("V79 actor PID row width changed")
        gpu, pid = (int(value) for value in fields)
        if gpu in rows or gpu not in GPU_IDS_V79 or pid <= 0:
            raise RuntimeError("V79 actor PID identity changed")
        rows[gpu] = pid
    if set(rows) != set(GPU_IDS_V79) or len(set(rows.values())) != 4:
        raise RuntimeError("V79 requires four unique actor PIDs")
    return rows


def parent_pid_v79(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(
            encoding="ascii"
        ).splitlines():
            if line.startswith("PPid:"):
                return int(line.split(":", 1)[1].strip())
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    raise RuntimeError(f"V79 could not parse parent PID for {pid}")


def is_descendant_v79(pid: int, root_pid: int) -> bool:
    if pid <= 0 or root_pid <= 0:
        return False
    seen = set()
    current: int | None = pid
    while current is not None and current > 1 and current not in seen:
        if current == root_pid:
            return True
        seen.add(current)
        current = parent_pid_v79(current)
    return current == root_pid


def pcie_rate_v79(pynvml, handle, counter_name: str) -> int | None:
    getter = getattr(pynvml, "nvmlDeviceGetPcieThroughput", None)
    counter = getattr(pynvml, counter_name, None)
    if getter is None or counter is None:
        return None
    try:
        return int(getter(handle, counter))
    except pynvml.NVMLError:
        return None


def process_rows_v79(pynvml, handle) -> dict[int, int]:
    try:
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml.NVMLError as error:
        raise RuntimeError("V79 compute-process telemetry unavailable") from error
    return {
        int(item.pid): int(item.usedGpuMemory // (1024 * 1024))
        for item in processes
    }


def parse_nvidia_smi_cleanup_v79(text: str) -> dict[int, dict[str, int]]:
    result = {}
    for line in text.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3:
            raise RuntimeError("V79 cleanup nvidia-smi row width changed")
        gpu, memory_used_mib, utilization = (int(item) for item in fields)
        if gpu in result or gpu not in GPU_IDS_V79:
            raise RuntimeError("V79 cleanup nvidia-smi GPU identity changed")
        result[gpu] = {
            "memory_used_mib": memory_used_mib,
            "gpu_utilization_percent": utilization,
        }
    if set(result) != set(GPU_IDS_V79):
        raise RuntimeError("V79 cleanup nvidia-smi cardinality changed")
    return result


def nvidia_smi_cleanup_v79() -> dict[int, dict[str, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if completed.stderr.strip():
        raise RuntimeError("V79 cleanup nvidia-smi emitted stderr")
    return parse_nvidia_smi_cleanup_v79(completed.stdout)


def sample_batch_v79(
    pynvml,
    handles: dict[int, object],
    roots: dict[int, int],
    batch_index: int,
    *,
    require_pcie: bool,
    trusted_compute_pids: dict[int, set[int]] | None = None,
) -> list[dict]:
    sampled_at = datetime.now(timezone.utc).isoformat()
    monotonic_ns = time.monotonic_ns()
    rows = []
    for gpu in GPU_IDS_V79:
        handle = handles[gpu]
        root_pid = roots[gpu]
        process_memory = process_rows_v79(pynvml, handle)
        pids = sorted(process_memory)
        ancestry_attributed = [
            pid for pid in pids if is_descendant_v79(pid, root_pid)
        ]
        prior_attributed = sorted(
            set(pids)
            & (
                trusted_compute_pids.get(gpu, set())
                if trusted_compute_pids is not None
                else set()
            )
            - set(ancestry_attributed)
        )
        attributed = sorted(set(ancestry_attributed) | set(prior_attributed))
        if trusted_compute_pids is not None:
            trusted_compute_pids[gpu].update(attributed)
        foreign = [pid for pid in pids if pid not in attributed]
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        rx = pcie_rate_v79(pynvml, handle, "NVML_PCIE_UTIL_RX_BYTES")
        tx = pcie_rate_v79(pynvml, handle, "NVML_PCIE_UTIL_TX_BYTES")
        if require_pcie and (rx is None or tx is None):
            raise RuntimeError(
                f"V79 GPU {gpu} PCIe RX/TX counters are unsupported"
            )
        rows.append(
            {
                "schema": SCHEMA_V79,
                "sequence": batch_index * 4 + gpu,
                "batch_index": batch_index,
                "sampled_at_utc": sampled_at,
                "monotonic_ns": monotonic_ns,
                "gpu": gpu,
                "gpu_uuid": str(pynvml.nvmlDeviceGetUUID(handle)),
                "actor_root_pid": root_pid,
                "actor_root_alive": Path(f"/proc/{root_pid}").exists(),
                "compute_pids": pids,
                "process_memory_mib": process_memory,
                "attributed_compute_pids": attributed,
                "ancestry_attributed_compute_pids": ancestry_attributed,
                "prior_ancestry_attributed_compute_pids": prior_attributed,
                "foreign_compute_pids": foreign,
                "gpu_utilization_percent": int(utilization.gpu),
                "memory_utilization_percent": int(utilization.memory),
                "hbm_bytes_per_second_inferred": False,
                "memory_used_mib": int(memory.used // (1024 * 1024)),
                "memory_total_mib": int(memory.total // (1024 * 1024)),
                "pcie_rx_kib_per_second": rx,
                "pcie_tx_kib_per_second": tx,
                "pcie_counters_supported": rx is not None and tx is not None,
                "power_draw_mw": int(
                    pynvml.nvmlDeviceGetPowerUsage(handle)
                ),
            }
        )
    return rows


def cleanup_idle_batch_v79(rows: list[dict]) -> bool:
    return (
        len(rows) == len(GPU_IDS_V79)
        and {row.get("gpu") for row in rows} == set(GPU_IDS_V79)
        and all(
            row.get("actor_root_alive") is False
            and row.get("compute_pids") == []
            and row.get("cleanup_nvidia_smi_gpu_utilization_percent") == 0
            and isinstance(
                row.get("cleanup_nvidia_smi_memory_used_mib"), int
            )
            and row["cleanup_nvidia_smi_memory_used_mib"]
            <= POST_EXIT_MEMORY_USED_MIB_MAX_V79
            for row in rows
        )
    )


def run_monitor_v79(
    actor_pids: Path,
    output: Path,
    *,
    sample_interval_seconds: float,
    cleanup_batches: int,
    max_cleanup_wait_seconds: float,
    require_pcie: bool,
) -> None:
    if output.exists():
        raise RuntimeError("V79 telemetry output must be fresh")
    if not (0 < sample_interval_seconds <= 1.0):
        raise ValueError("V79 telemetry interval must be in (0, 1] seconds")
    if cleanup_batches < 2:
        raise ValueError("V79 requires at least two cleanup batches")
    if max_cleanup_wait_seconds <= 0:
        raise ValueError("V79 cleanup wait must be positive")
    roots = read_actor_pids_v79(actor_pids)

    import pynvml

    pynvml.nvmlInit()
    try:
        handles = {
            gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu)
            for gpu in GPU_IDS_V79
        }
        batch_index = 0
        cleanup_idle_count = 0
        cleanup_started: float | None = None
        trusted_compute_pids = {gpu: set() for gpu in GPU_IDS_V79}
        with output.open("x", encoding="utf-8") as handle:
            while True:
                rows = sample_batch_v79(
                    pynvml,
                    handles,
                    roots,
                    batch_index,
                    require_pcie=require_pcie,
                    trusted_compute_pids=trusted_compute_pids,
                )
                if any(row["foreign_compute_pids"] for row in rows):
                    raise RuntimeError("V79 observed a foreign GPU compute PID")
                roots_alive = any(
                    Path(f"/proc/{pid}").exists() for pid in roots.values()
                )
                cleanup_snapshot = (
                    None if roots_alive else nvidia_smi_cleanup_v79()
                )
                for row in rows:
                    snapshot = (
                        None
                        if cleanup_snapshot is None
                        else cleanup_snapshot[row["gpu"]]
                    )
                    row["cleanup_nvidia_smi_memory_used_mib"] = (
                        None if snapshot is None else snapshot["memory_used_mib"]
                    )
                    row["cleanup_nvidia_smi_gpu_utilization_percent"] = (
                        None
                        if snapshot is None
                        else snapshot["gpu_utilization_percent"]
                    )
                    row["cleanup_memory_gate_uses_external_nvidia_smi"] = (
                        snapshot is not None
                    )
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                batch_index += 1
                if roots_alive:
                    cleanup_idle_count = 0
                    cleanup_started = None
                else:
                    if cleanup_started is None:
                        cleanup_started = time.monotonic()
                    cleanup_idle_count = (
                        cleanup_idle_count + 1
                        if cleanup_idle_batch_v79(rows)
                        else 0
                    )
                    if cleanup_idle_count >= cleanup_batches:
                        break
                    if (
                        time.monotonic() - cleanup_started
                        > max_cleanup_wait_seconds
                    ):
                        raise RuntimeError(
                            "V79 GPUs did not reach the sealed cleanup-idle gate"
                        )
                time.sleep(sample_interval_seconds)
    finally:
        pynvml.nvmlShutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-pids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--cleanup-batches", type=int, default=3)
    parser.add_argument("--max-cleanup-wait-seconds", type=float, default=60.0)
    parser.add_argument("--require-pcie", action="store_true")
    args = parser.parse_args()
    run_monitor_v79(
        args.actor_pids.resolve(),
        args.output.resolve(),
        sample_interval_seconds=args.sample_interval_seconds,
        cleanup_batches=args.cleanup_batches,
        max_cleanup_wait_seconds=args.max_cleanup_wait_seconds,
        require_pcie=args.require_pcie,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
