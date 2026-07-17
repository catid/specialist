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
import time
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_V79 = "v79-four-gpu-kv-capacity-telemetry"
GPU_IDS_V79 = (0, 1, 2, 3)


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


def sample_batch_v79(
    pynvml,
    handles: dict[int, object],
    roots: dict[int, int],
    batch_index: int,
    *,
    require_pcie: bool,
) -> list[dict]:
    sampled_at = datetime.now(timezone.utc).isoformat()
    monotonic_ns = time.monotonic_ns()
    rows = []
    for gpu in GPU_IDS_V79:
        handle = handles[gpu]
        root_pid = roots[gpu]
        process_memory = process_rows_v79(pynvml, handle)
        pids = sorted(process_memory)
        attributed = [
            pid for pid in pids if is_descendant_v79(pid, root_pid)
        ]
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


def run_monitor_v79(
    actor_pids: Path,
    output: Path,
    *,
    sample_interval_seconds: float,
    cleanup_batches: int,
    require_pcie: bool,
) -> None:
    if output.exists():
        raise RuntimeError("V79 telemetry output must be fresh")
    if not (0 < sample_interval_seconds <= 1.0):
        raise ValueError("V79 telemetry interval must be in (0, 1] seconds")
    if cleanup_batches < 2:
        raise ValueError("V79 requires at least two cleanup batches")
    roots = read_actor_pids_v79(actor_pids)

    import pynvml

    pynvml.nvmlInit()
    try:
        handles = {
            gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu)
            for gpu in GPU_IDS_V79
        }
        batch_index = 0
        cleanup_remaining: int | None = None
        with output.open("x", encoding="utf-8") as handle:
            while True:
                rows = sample_batch_v79(
                    pynvml,
                    handles,
                    roots,
                    batch_index,
                    require_pcie=require_pcie,
                )
                if any(row["foreign_compute_pids"] for row in rows):
                    raise RuntimeError("V79 observed a foreign GPU compute PID")
                for row in rows:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                handle.flush()
                batch_index += 1

                roots_alive = any(
                    Path(f"/proc/{pid}").exists() for pid in roots.values()
                )
                if roots_alive:
                    cleanup_remaining = None
                elif cleanup_remaining is None:
                    cleanup_remaining = cleanup_batches - 1
                    if cleanup_remaining == 0:
                        break
                else:
                    cleanup_remaining -= 1
                    if cleanup_remaining == 0:
                        break
                time.sleep(sample_interval_seconds)
    finally:
        pynvml.nvmlShutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-pids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.5)
    parser.add_argument("--cleanup-batches", type=int, default=3)
    parser.add_argument("--require-pcie", action="store_true")
    args = parser.parse_args()
    run_monitor_v79(
        args.actor_pids.resolve(),
        args.output.resolve(),
        sample_interval_seconds=args.sample_interval_seconds,
        cleanup_batches=args.cleanup_batches,
        require_pcie=args.require_pcie,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
