#!/usr/bin/env python3
"""Fail-closed launcher and postrun analyzer for V73E exact-phase profiling."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sqlite3
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import qwen36_v73e_exact_phase_profiler_contract as builder


ROOT = Path(__file__).resolve().parent
COMMAND_ATTESTATION_ENV = "SPECIALIST_V73E_EXPANDED_COMMAND_SHA256"
NVTX_RANGE_TYPES = (59, 60, 70, 71)
NVTX_DOMAIN_CREATE = 75
GPU_IDS = (0, 1, 2, 3)
CANONICAL_FP32_TENSOR_COUNT = 70
CANONICAL_FP32_ELEMENT_COUNT = 4_528_128
LOGICAL_COLLECTIVE_PAYLOAD_BYTES_PER_RANK = 18_112_512
NOMINAL_RING_SCHEDULE_BYTES_PER_RANK = 27_168_768


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="ascii"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(
        Path(path).read_text(encoding="ascii").splitlines(), 1
    ):
        _require(bool(line), f"blank JSONL row: {path}:{number}")
        row = json.loads(line)
        _require(isinstance(row, dict), f"JSONL object required: {path}:{number}")
        rows.append(row)
    _require(bool(rows), f"empty JSONL artifact: {path}")
    return rows


def _validate_self_hash(value: Mapping[str, Any], label: str) -> str:
    body = copy.deepcopy(dict(value))
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and claimed == builder.canonical_sha256(body),
        f"V73E self hash changed: {label}",
    )
    return claimed


def load_preregistration_v73e(
    path: Path,
    file_sha256: str,
    content_sha256: str,
) -> dict[str, Any]:
    path = Path(path).resolve()
    _require(
        builder.file_sha256(path) == file_sha256,
        "V73E preregistration file identity changed",
    )
    value = _load_json(path)
    _validate_self_hash(value, "preregistration")
    builder.validate_generated_preregistration_v73e(value)
    _require(
        value.get("content_sha256_before_self_field") == content_sha256,
        "V73E preregistration content changed",
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
    temporary = path.with_name(f".{path.name}.tmp-v73e-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return result


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
        )
    }


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    safe = table.replace('"', '""')
    return {row[1] for row in connection.execute(f'PRAGMA table_info("{safe}")')}


def _require_columns(
    connection: sqlite3.Connection,
    table: str,
    required: Iterable[str],
) -> None:
    observed = _columns(connection, table)
    missing = set(required) - observed
    _require(not missing, f"V73E SQLite {table} columns absent: {sorted(missing)}")


def _decode_pid(global_id: int) -> int:
    return (int(global_id) >> 24) & 0x00FF_FFFF


def _string_ids(connection: sqlite3.Connection) -> dict[int, str]:
    return {
        int(row[0]): str(row[1])
        for row in connection.execute("SELECT id, value FROM StringIds")
    }


def _nvtx_rows(connection: sqlite3.Connection) -> tuple[list[dict], list[dict]]:
    required = {"NVTX_EVENTS", "StringIds"}
    _require(required.issubset(_tables(connection)), "V73E NVTX tables absent")
    _require_columns(
        connection,
        "NVTX_EVENTS",
        ("start", "end", "eventType", "globalTid", "domainId", "textId", "text"),
    )
    strings = _string_ids(connection)
    domains: dict[tuple[int, int], str] = {}
    for row in connection.execute(
        "SELECT domainId, globalTid, textId, text FROM NVTX_EVENTS "
        "WHERE eventType = ?",
        (NVTX_DOMAIN_CREATE,),
    ):
        name = strings.get(row[2], row[3])
        if name is not None:
            domains[(int(row[0]), _decode_pid(row[1]))] = str(name)

    phase_rows = []
    nccl_rows = []
    placeholders = ",".join("?" for _ in NVTX_RANGE_TYPES)
    query = (
        "SELECT start, end, eventType, globalTid, domainId, textId, text "
        f"FROM NVTX_EVENTS WHERE eventType IN ({placeholders}) ORDER BY start"
    )
    for row in connection.execute(query, NVTX_RANGE_TYPES):
        start, end = row[0], row[1]
        if start is None or end is None or int(end) <= int(start):
            continue
        pid = _decode_pid(row[3])
        domain = domains.get((int(row[4]), pid))
        text = strings.get(row[5], row[6])
        item = {
            "start_ns": int(start),
            "end_ns": int(end),
            "elapsed_ns": int(end) - int(start),
            "pid": pid,
            "tid": int(row[3]) & 0x00FF_FFFF,
            "domain": domain,
            "text": None if text is None else str(text),
        }
        if domain == builder.PHASE_DOMAIN:
            phase_rows.append(item)
        elif domain == "NCCL":
            nccl_rows.append(item)
    return phase_rows, nccl_rows


def _validate_phase_ranges(
    trace_rows: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    _validate_self_hash(receipt, "phase receipt")
    expected = list(builder.PHASES)
    _require(
        receipt.get("complete") is True
        and receipt.get("nvtx_domain") == builder.PHASE_DOMAIN
        and receipt.get("expected_phase_order") == expected
        and receipt.get("observed_phase_order") == expected
        and receipt.get("phase_count") == len(expected),
        "V73E application phase receipt changed",
    )
    selected = [dict(row) for row in trace_rows]
    _require(
        [row["text"] for row in selected] == expected,
        "V73E trace phase order changed",
    )
    _require(
        len({row["pid"] for row in selected}) == 1
        and len({row["tid"] for row in selected}) == 1
        and all(
            right["start_ns"] >= left["end_ns"]
            for left, right in zip(selected, selected[1:])
        ),
        "V73E trace phase ownership or overlap changed",
    )
    return selected


def _phase_for_event(
    start_ns: int,
    end_ns: int,
    phases: Sequence[Mapping[str, Any]],
    label: str,
) -> str:
    if start_ns == end_ns:
        # Allocation records are timestamped points.  Treat phase ranges as
        # half-open so a point exactly on a transition cannot match both
        # adjacent phases.
        matches = [
            phase["text"]
            for phase in phases
            if phase["start_ns"] <= start_ns < phase["end_ns"]
        ]
    else:
        matches = [
            phase["text"]
            for phase in phases
            if phase["start_ns"] <= start_ns and end_ns <= phase["end_ns"]
        ]
    _require(
        len(matches) == 1,
        f"V73E {label} crossed or fell outside exact phase ranges",
    )
    return str(matches[0])


def _gpu_events(
    connection: sqlite3.Connection,
    phases: Sequence[Mapping[str, Any]],
    physical_gpu_by_pid: Mapping[int, int],
) -> tuple[list[dict], list[dict]]:
    expected_pids = set(physical_gpu_by_pid)
    tables = _tables(connection)
    _require(
        "CUPTI_ACTIVITY_KIND_KERNEL" in tables
        and "CUPTI_ACTIVITY_KIND_MEMCPY" in tables
        and "ENUM_CUDA_MEMCPY_OPER" in tables,
        "V73E CUDA kernel/memcpy tables absent",
    )
    kernel_required = (
        "start", "end", "deviceId", "globalPid", "demangledName", "shortName"
    )
    memcpy_required = (
        "start", "end", "deviceId", "globalPid", "bytes", "copyKind"
    )
    _require_columns(connection, "CUPTI_ACTIVITY_KIND_KERNEL", kernel_required)
    _require_columns(connection, "CUPTI_ACTIVITY_KIND_MEMCPY", memcpy_required)
    _require_columns(connection, "ENUM_CUDA_MEMCPY_OPER", ("id", "label"))
    copy_labels = {
        int(row[0]): str(row[1])
        for row in connection.execute("SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER")
    }
    kernels = []
    strings = _string_ids(connection)
    for row in connection.execute(
        "SELECT start, end, deviceId, globalPid, demangledName, shortName FROM "
        "CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ):
        start, end, device, global_pid = map(int, row[:4])
        _require(end > start, "V73E CUDA kernel duration changed")
        pid = _decode_pid(global_pid)
        _require(pid in expected_pids, "V73E CUDA kernel PID is not an actor")
        demangled = strings.get(int(row[4]))
        short = strings.get(int(row[5]))
        _require(
            isinstance(demangled, str) and isinstance(short, str),
            "V73E CUDA kernel symbol identity changed",
        )
        kernels.append({
            "phase": _phase_for_event(start, end, phases, "CUDA kernel"),
            "start_ns": start,
            "end_ns": end,
            "elapsed_ns": end - start,
            "device_id": device,
            "cupti_device_id": device,
            "physical_gpu_id": physical_gpu_by_pid[pid],
            "pid": pid,
            "demangled_name": demangled,
            "short_name": short,
        })
    copies = []
    for row in connection.execute(
        "SELECT start, end, deviceId, globalPid, bytes, copyKind FROM "
        "CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start"
    ):
        start, end, device, global_pid, size, kind = map(int, row)
        _require(end > start and size >= 0, "V73E CUDA memcpy changed")
        pid = _decode_pid(global_pid)
        _require(pid in expected_pids, "V73E CUDA memcpy PID is not an actor")
        _require(kind in copy_labels, "V73E CUDA memcpy kind is unknown")
        copies.append({
            "phase": _phase_for_event(start, end, phases, "CUDA memcpy"),
            "start_ns": start,
            "end_ns": end,
            "elapsed_ns": end - start,
            "device_id": device,
            "cupti_device_id": device,
            "physical_gpu_id": physical_gpu_by_pid[pid],
            "pid": pid,
            "bytes": size,
            "copy_kind": copy_labels[kind],
        })
    _require(bool(kernels), "V73E CUDA kernel trace is empty")
    _require(bool(copies), "V73E CUDA memcpy trace is empty")
    _require(
        {row["physical_gpu_id"] for row in kernels} == set(GPU_IDS)
        and {row["pid"] for row in kernels} == expected_pids,
        "V73E CUDA kernel trace omitted a GPU",
    )
    return kernels, copies


def _allocation_events(
    connection: sqlite3.Connection,
    phases: Sequence[Mapping[str, Any]],
    physical_gpu_by_pid: Mapping[int, int],
) -> list[dict[str, Any]]:
    expected_pids = set(physical_gpu_by_pid)
    tables = _tables(connection)
    _require(
        "CUDA_GPU_MEMORY_USAGE_EVENTS" in tables
        and "ENUM_CUDA_DEV_MEM_EVENT_OPER" in tables,
        "V73E CUDA allocation tables absent",
    )
    _require_columns(
        connection,
        "CUDA_GPU_MEMORY_USAGE_EVENTS",
        ("start", "globalPid", "deviceId", "bytes", "memoryOperationType"),
    )
    _require_columns(
        connection, "ENUM_CUDA_DEV_MEM_EVENT_OPER", ("id", "label")
    )
    labels = {
        int(row[0]): str(row[1])
        for row in connection.execute(
            "SELECT id, label FROM ENUM_CUDA_DEV_MEM_EVENT_OPER"
        )
    }
    rows = []
    for row in connection.execute(
        "SELECT start, globalPid, deviceId, bytes, memoryOperationType "
        "FROM CUDA_GPU_MEMORY_USAGE_EVENTS ORDER BY start"
    ):
        start, global_pid, device, size, operation = map(int, row)
        pid = _decode_pid(global_pid)
        _require(pid in expected_pids, "V73E allocation PID is not an actor")
        _require(size >= 0 and operation in labels, "V73E allocation event changed")
        rows.append({
            "phase": _phase_for_event(start, start, phases, "allocation event"),
            "start_ns": start,
            "device_id": device,
            "cupti_device_id": device,
            "physical_gpu_id": physical_gpu_by_pid[pid],
            "pid": pid,
            "bytes": size,
            "operation": labels[operation],
        })
    _require(bool(rows), "V73E allocation event trace is empty")
    _require(
        {row["physical_gpu_id"] for row in rows} == set(GPU_IDS)
        and {row["pid"] for row in rows} == expected_pids,
        "V73E allocation event trace omitted an actor or physical GPU",
    )
    return rows


def _event_union_ns(rows: Sequence[Mapping[str, Any]]) -> int:
    intervals = sorted((int(row["start_ns"]), int(row["end_ns"])) for row in rows)
    if not intervals:
        return 0
    total = 0
    left, right = intervals[0]
    for start, end in intervals[1:]:
        if start <= right:
            right = max(right, end)
        else:
            total += right - left
            left, right = start, end
    return total + right - left


def _summarize_gpu_events(
    kernels: Sequence[Mapping[str, Any]],
    copies: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = {}
    for phase in builder.PHASES:
        per_gpu = {}
        for gpu in GPU_IDS:
            selected_k = [
                row for row in kernels
                if row["phase"] == phase and row["physical_gpu_id"] == gpu
            ]
            selected_c = [
                row for row in copies
                if row["phase"] == phase and row["physical_gpu_id"] == gpu
            ]
            per_gpu[str(gpu)] = {
                "kernel_count": len(selected_k),
                "kernel_sum_ns": sum(row["elapsed_ns"] for row in selected_k),
                "memcpy_count": len(selected_c),
                "memcpy_sum_ns": sum(row["elapsed_ns"] for row in selected_c),
                "memcpy_bytes": sum(row["bytes"] for row in selected_c),
                "gpu_event_union_ns": _event_union_ns([*selected_k, *selected_c]),
                "kernel_by_short_name": {
                    name: {
                        "count": sum(
                            row["short_name"] == name for row in selected_k
                        ),
                        "sum_ns": sum(
                            row["elapsed_ns"] for row in selected_k
                            if row["short_name"] == name
                        ),
                    }
                    for name in sorted({row["short_name"] for row in selected_k})
                },
                "memcpy_by_kind": {
                    kind: {
                        "count": sum(row["copy_kind"] == kind for row in selected_c),
                        "bytes": sum(
                            row["bytes"] for row in selected_c
                            if row["copy_kind"] == kind
                        ),
                    }
                    for kind in sorted({row["copy_kind"] for row in selected_c})
                },
            }
        result[phase] = per_gpu
    return result


def _explicit_named_kernel_evidence(
    kernels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    classifiers = {
        "nccl_explicit_symbol": ("nccl",),
        "pack_flatten_or_materialize_explicit_symbol": (
            "pack", "flatten", "materializ"
        ),
    }
    result = {}
    for label, tokens in classifiers.items():
        selected = [
            row for row in kernels
            if any(
                token in (
                    f"{row['short_name']} {row['demangled_name']}".lower()
                )
                for token in tokens
            )
        ]
        by_phase_gpu: dict[str, dict[str, Any]] = {}
        for phase in builder.PHASES:
            per_gpu = {}
            for gpu in GPU_IDS:
                rows = [
                    row for row in selected
                    if row["phase"] == phase and row["physical_gpu_id"] == gpu
                ]
                per_gpu[str(gpu)] = {
                    "count": len(rows),
                    "sum_ns": sum(row["elapsed_ns"] for row in rows),
                    "short_names": sorted({row["short_name"] for row in rows}),
                }
            by_phase_gpu[phase] = per_gpu
        result[label] = {
            "classification_available": bool(selected),
            "classifier": "case_insensitive_exported_kernel_symbol_token_only",
            "tokens": list(tokens),
            "missing_symbols_are_not_inferred": True,
            "event_count": len(selected),
            "by_phase_and_gpu": by_phase_gpu,
        }
    result["actual_nccl_link_path_or_link_only_time"] = {
        "available": False,
        "reason": "timeline_symbols_and_ranges_do_not_measure_link_only_time",
        "topology_capability_used_as_path_evidence": False,
    }
    return result


def _summarize_allocations(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for gpu in GPU_IDS:
        selected = [row for row in rows if row["physical_gpu_id"] == gpu]
        running = 0
        peak = 0
        by_phase: dict[str, dict[str, int]] = defaultdict(
            lambda: {"event_count": 0, "allocated_bytes": 0, "freed_bytes": 0}
        )
        for row in selected:
            label = row["operation"].lower()
            is_free = "free" in label or "dealloc" in label or "release" in label
            is_alloc = "alloc" in label and not is_free
            _require(is_alloc != is_free, f"V73E allocation label unsupported: {label}")
            phase = by_phase[row["phase"]]
            phase["event_count"] += 1
            if is_alloc:
                running += row["bytes"]
                phase["allocated_bytes"] += row["bytes"]
            else:
                running -= row["bytes"]
                phase["freed_bytes"] += row["bytes"]
            _require(running >= 0, "V73E traced allocation balance became negative")
            peak = max(peak, running)
        result[str(gpu)] = {
            "event_count": len(selected),
            "tool_traced_peak_live_allocation_bytes": peak,
            "final_live_allocation_bytes": running,
            "by_phase": dict(sorted(by_phase.items())),
            "scope": "CUDA allocation API events visible to Nsight Systems",
        }
    return result


_NCCL_BYTE_PATTERN = re.compile(
    r"(?:bytes|size)\s*[=:]\s*([0-9]+)", re.IGNORECASE
)


def _summarize_nccl(
    rows: Sequence[Mapping[str, Any]],
    phases: Sequence[Mapping[str, Any]],
    expected_pids: set[int],
) -> dict[str, Any]:
    _require(bool(rows), "V73E NCCL trace is empty")
    by_phase: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"event_count": 0, "sum_duration_ns": 0, "tool_exposed_bytes": 0}
    )
    texts = set()
    exposed = 0
    by_operation: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"event_count": 0, "sum_duration_ns": 0}
    )
    update_count_by_pid = {str(pid): 0 for pid in sorted(expected_pids)}
    for row in rows:
        _require(row["pid"] in expected_pids, "V73E NCCL PID is not an actor")
        phase = _phase_for_event(
            row["start_ns"], row["end_ns"], phases, "NCCL event"
        )
        item = by_phase[phase]
        item["event_count"] += 1
        item["sum_duration_ns"] += row["elapsed_ns"]
        text = row.get("text") or ""
        texts.add(text)
        operation = by_operation[text]
        operation["event_count"] += 1
        operation["sum_duration_ns"] += row["elapsed_ns"]
        if phase == "pair_difference_update_execute_all_actors":
            update_count_by_pid[str(row["pid"])] += 1
        matches = [int(value) for value in _NCCL_BYTE_PATTERN.findall(text)]
        item["tool_exposed_bytes"] += sum(matches)
        exposed += sum(matches)
    _require(
        by_phase["pair_difference_update_execute_all_actors"]["event_count"] > 0,
        "V73E update execute contains no NCCL trace",
    )
    return {
        "by_phase": dict(sorted(by_phase.items())),
        "event_text_set_sha256": builder.canonical_sha256(sorted(texts)),
        "by_exact_exported_operation_text": dict(sorted(by_operation.items())),
        "update_execute_api_range_count_by_actor_pid": update_count_by_pid,
        "canonical_tensor_count_hypothesis": CANONICAL_FP32_TENSOR_COUNT,
        "observed_exactly_70_api_ranges_per_actor": all(
            count == CANONICAL_FP32_TENSOR_COUNT
            for count in update_count_by_pid.values()
        ),
        "api_range_count_interpreted_as_python_call_count": False,
        "api_range_duration_interpreted_as_link_only_time": False,
        "tool_exposed_message_bytes": exposed,
        "tool_exposed_message_bytes_available": exposed > 0,
        "exact_logical_collective_payload_bytes_per_rank": (
            LOGICAL_COLLECTIVE_PAYLOAD_BYTES_PER_RANK
        ),
        "exact_logical_collective_payload_bytes_all_ranks": (
            4 * LOGICAL_COLLECTIVE_PAYLOAD_BYTES_PER_RANK
        ),
        "logical_payload_is_not_physical_link_bytes": True,
        "canonical_fp32_tensor_count": CANONICAL_FP32_TENSOR_COUNT,
        "canonical_fp32_element_count": CANONICAL_FP32_ELEMENT_COUNT,
        "nominal_ring_schedule_bytes_per_rank": (
            NOMINAL_RING_SCHEDULE_BYTES_PER_RANK
        ),
        "nominal_ring_schedule_is_not_measured_physical_link_bytes": True,
        "trace_mode": "legacy_NCCL_NVTX_API_ranges_for_installed_2.27.3",
        "physical_link_bytes_inferred_from_algorithm": False,
    }


def _validate_empty_process_streams(
    connection: sqlite3.Connection,
) -> dict[str, Any]:
    _require(
        "ProcessStreams" in _tables(connection),
        "V73E ProcessStreams table absent",
    )
    _require_columns(
        connection,
        "ProcessStreams",
        ("globalPid", "filenameId", "contentId"),
    )
    strings = _string_ids(connection)
    rows = []
    for global_pid, filename_id, content_id in connection.execute(
        "SELECT globalPid, filenameId, contentId FROM ProcessStreams"
    ):
        filename = strings.get(int(filename_id))
        content = strings.get(int(content_id))
        _require(
            isinstance(filename, str) and isinstance(content, str),
            "V73E process stream string identity changed",
        )
        _require(
            not content.strip(),
            "V73E target process stream was nonempty; trace rejected",
        )
        rows.append({
            "pid": _decode_pid(int(global_pid)),
            "filename_sha256": hashlib.sha256(
                filename.encode("utf-8")
            ).hexdigest(),
        })
    return {
        "stream_count": len(rows),
        "all_stream_content_empty": True,
        "stream_identity": rows,
        "prompts_questions_answers_or_outputs_persisted": False,
    }


_NCCL_EXPLICIT_TRANSPORT = re.compile(
    r"\bvia\s+([A-Za-z0-9_./+-]+)", re.IGNORECASE
)


def _analyze_nccl_debug_logs_v73e(
    pattern: str,
    expected_pids: set[int],
) -> dict[str, Any]:
    path = Path(pattern)
    _require("*" in path.name, "V73E NCCL debug pattern changed")
    files = sorted(path.parent.glob(path.name))
    _require(files, "V73E NCCL metadata logs are absent")
    observed_pids = set()
    logs = []
    explicit = []
    forbidden_markers = (
        "<|im_start|>", "<|im_end|>", "<think>", '"question"', '"answer"'
    )
    for file_path in files:
        _require(
            file_path.is_file() and not file_path.is_symlink(),
            "V73E NCCL metadata log is not a regular file",
        )
        match = re.search(r"\.([0-9]+)\.log$", file_path.name)
        _require(match is not None, "V73E NCCL metadata PID filename changed")
        pid = int(match.group(1))
        observed_pids.add(pid)
        content = file_path.read_text(encoding="utf-8")
        _require(content, "V73E NCCL metadata log is empty")
        lowered = content.casefold()
        _require(
            not any(marker.casefold() in lowered for marker in forbidden_markers),
            "V73E NCCL metadata log contains semantic payload marker",
        )
        lines = content.splitlines()
        for number, line in enumerate(lines, 1):
            if "NCCL INFO" not in line or " via " not in line:
                continue
            transport = _NCCL_EXPLICIT_TRANSPORT.search(line)
            if transport is None:
                continue
            explicit.append({
                "pid": pid,
                "line_number": number,
                "transport_token": transport.group(1),
                "line_sha256": hashlib.sha256(
                    line.encode("utf-8")
                ).hexdigest(),
            })
        logs.append({
            "pid": pid,
            "file_sha256": builder.file_sha256(file_path),
            "bytes": file_path.stat().st_size,
            "line_count": len(lines),
        })
    _require(
        expected_pids.issubset(observed_pids),
        "V73E NCCL metadata logs omitted an actor PID",
    )
    return {
        "debug_level": "INFO",
        "debug_subsystems": ["INIT", "GRAPH", "COLL"],
        "technical_metadata_logs": logs,
        "all_actor_pids_covered": True,
        "semantic_payload_markers_absent": True,
        "explicit_transport_evidence": explicit,
        "actual_transport_evidence_available": bool(explicit),
        "path_left_unresolved_when_explicit_line_absent": True,
        "topology_capability_used_as_path_evidence": False,
        "transport_token_used_to_infer_link_only_time_or_bytes": False,
    }


def analyze_sqlite_v73e(
    sqlite_path: Path,
    phase_receipt: Mapping[str, Any],
    physical_gpu_by_pid: Mapping[int, int],
) -> dict[str, Any]:
    expected_pids = set(physical_gpu_by_pid)
    path = Path(sqlite_path).resolve()
    _require(path.is_file() and not path.is_symlink(), "V73E SQLite export absent")
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        phase_trace, nccl = _nvtx_rows(connection)
        phases = _validate_phase_ranges(phase_trace, phase_receipt)
        kernels, copies = _gpu_events(
            connection, phases, physical_gpu_by_pid
        )
        allocations = _allocation_events(
            connection, phases, physical_gpu_by_pid
        )
        result = {
            "sqlite_file_sha256": builder.file_sha256(path),
            "sqlite_tables_sha256": builder.canonical_sha256(sorted(_tables(connection))),
            "exact_phase_ranges": phases,
            "cuda_by_phase_and_gpu": _summarize_gpu_events(kernels, copies),
            "explicit_named_kernel_evidence": (
                _explicit_named_kernel_evidence(kernels)
            ),
            "cuda_kernel_event_count": len(kernels),
            "cuda_memcpy_event_count": len(copies),
            "cuda_memcpy_bytes": sum(row["bytes"] for row in copies),
            "cuda_allocations": _summarize_allocations(allocations),
            "gpu_attribution": {
                "source": "exact_actor_receipt_pid_to_physical_gpu_mapping",
                "physical_gpu_by_actor_pid": {
                    str(pid): int(gpu)
                    for pid, gpu in sorted(physical_gpu_by_pid.items())
                },
                "cupti_device_ids_by_actor_pid": {
                    str(pid): sorted({
                        row["cupti_device_id"]
                        for row in [*kernels, *copies, *allocations]
                        if row["pid"] == pid
                    })
                    for pid in sorted(expected_pids)
                },
                "cupti_process_local_ordinal_assumed_physical": False,
                "all_four_physical_gpus_exact": True,
            },
            "nccl": _summarize_nccl(nccl, phases, expected_pids),
            "process_streams": _validate_empty_process_streams(connection),
            "hbm_dram_metrics": {
                "collected": False,
                "reason": "timeline_arm_excludes_privileged_gpu_metrics",
                "utilization_percent_converted_to_bytes": False,
            },
        }
    finally:
        connection.close()
    return result


def _validate_guard_process_receipt_v73e(
    value: Mapping[str, Any], expected_mechanism: str
) -> dict[str, Any]:
    row = dict(value)
    claimed = row.pop("receipt_sha256", None)
    _require(
        claimed == builder.canonical_sha256(row)
        and value.get("schema")
        == "qwen36-v73e-systems-only-path-guard-process-v1"
        and value.get("installed_before_runtime_imports") is True
        and value.get("installation_mechanism") == expected_mechanism
        and value.get("installation_pid") == value.get("pid")
        and value.get("boundary_registry_file_sha256")
        == builder.BOUNDARY_REGISTRY_FILE_SHA256
        and value.get("boundary_registry_content_sha256")
        == builder.BOUNDARY_REGISTRY_CONTENT_SHA256
        and value.get("exact_path_identity_count") == 3
        and value.get("prefix_identity_count") == 2
        and value.get("successful_protected_opens") == 0
        and value.get("successful_protected_resolves") == 0
        and value.get("successful_protected_metadata") == 0
        and value.get("successful_protected_enumerations") == 0
        and value.get("protected_path_values_persisted") is False
        and value.get("quality_hpo_or_promotion_authorized") is False,
        "V73E path guard process receipt changed",
    )
    return dict(value)


def _validate_actor_bootstrap_receipt_v73e(
    value: Mapping[str, Any], guard_source_sha256: str
) -> dict[str, Any]:
    row = dict(value)
    claimed = row.pop("receipt_sha256", None)
    pre_parent = _validate_guard_process_receipt_v73e(
        row.get("pre_parent_guard_receipt", {}),
        "ray_actor_worker_extension_pre_parent_import",
    )
    current = _validate_guard_process_receipt_v73e(
        row.get("guard_process_receipt", {}),
        "ray_actor_worker_extension_pre_parent_import",
    )
    staged = row.get("staged_inverse_install", {})
    inverse = staged.get("inverse_transform_proof", {})
    _require(
        claimed == builder.canonical_sha256(row)
        and value.get("schema")
        == "qwen36-v73e-worker-bootstrap-receipt-v1"
        and value.get("pid") == current.get("pid")
        and value.get("process_role") == "ray_actor_worker_extension"
        and value.get("bootstrap_mechanism")
        == "ray_actor_worker_extension_pre_parent_import"
        and value.get("guard_was_preinstalled") is False
        and value.get("parent_modules_absent_before_guard_install") is True
        and value.get(
            "historical_reference_modules_absent_before_guard_install"
        )
        is True
        and value.get("historical_reference_module_identity_count") == 3
        and value.get(
            "historical_reference_module_identity_set_sha256"
        )
        == builder.HISTORICAL_REFERENCE_MODULE_IDENTITY_SET_SHA256
        and value.get("parent_module_count_after_guard_install") == 3
        and value.get("guard_source_sha256") == guard_source_sha256
        and value.get("actor_bootstrap_env_exact") is True
        and value.get("pre_parent_guard_receipt_sha256")
        == pre_parent.get("receipt_sha256")
        and pre_parent.get("pid") == current.get("pid")
        and value.get("staged_inverse_install_complete") is True
        and staged.get("schema")
        == "qwen36-v73e-staged-inverse-install-v1"
        and staged.get("complete") is True
        and staged.get("staged_weights_sha256")
        == builder.STAGED_ADAPTER_WEIGHTS_SHA256
        and staged.get("staged_config_sha256")
        == builder.STAGED_ADAPTER_CONFIG_SHA256
        and staged.get("canonical_master_sha256")
        == builder.CANONICAL_MASTER_SHA256
        and staged.get("canonical_runtime_values_sha256")
        == builder.CANONICAL_RUNTIME_VALUES_SHA256
        and staged.get(
            "historical_protected_source_opened_resolved_statted_or_hashed"
        )
        is False
        and inverse.get("schema")
        == "qwen36-v73e-staged-inverse-transform-proof-v1"
        and inverse.get("operation") == "exact_prefix_inverse_only"
        and inverse.get("source_tensor_namespace") == builder.STAGED_TARGET_PREFIX
        and inverse.get("canonical_tensor_namespace")
        == builder.CANONICAL_SOURCE_PREFIX
        and inverse.get("tensor_count") == builder.STAGED_TENSOR_COUNT
        and inverse.get("element_count") == builder.STAGED_ELEMENT_COUNT
        and inverse.get("inverse_key_mapping_sha256")
        == builder.INVERSE_KEY_MAPPING_SHA256
        and inverse.get("staged_transform_identity_sha256")
        == builder.STAGED_TRANSFORM_IDENTITY_SHA256
        and value.get("quality_hpo_or_promotion_authorized") is False,
        "V73E Ray actor bootstrap receipt changed",
    )
    return dict(value)


def validate_target_artifacts_v73e(
    preregistration: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    paths = preregistration["arms"][arm]["artifacts"]
    report = _load_json(Path(paths["report"]))
    population = _load_json(Path(paths["population"]))
    update = _load_json(Path(paths["update"]))
    equivalence = _load_json(Path(paths["equivalence"]))
    traffic = _load_json(Path(paths["audit_traffic"]))
    host = _load_json(Path(paths["host_process_summary"]))
    phase = _load_json(Path(paths["phase_receipt"]))
    guard = _load_json(Path(paths["path_guard_receipt"]))
    attempt = _load_json(Path(paths["profile_attempt"]))
    for name, value in (
        ("report", report),
        ("population", population),
        ("update", update),
        ("equivalence", equivalence),
        ("traffic", traffic),
        ("host", host),
        ("phase", phase),
        ("path guard", guard),
        ("profile attempt", attempt),
    ):
        _validate_self_hash(value, name)
    rows = _load_jsonl(Path(paths["actor_cuda_work_log"]))
    _require(len(rows) == 16, "V73E actor receipt count changed")
    binding_sets = defaultdict(set)
    waves = defaultdict(set)
    for row in rows:
        body = dict(row)
        claimed = body.pop("receipt_sha256", None)
        _require(
            claimed == builder.canonical_sha256(body),
            "V73E actor receipt self hash changed",
        )
        rank = int(row["engine_rank"])
        waves[int(row["wave_index"])].add(rank)
        binding_sets[rank].add((
            int(row["worker_pid"]), int(row["physical_gpu_id"])
        ))
    _require(
        all(len(values) == 1 for values in binding_sets.values()),
        "V73E actor PID/GPU binding changed between waves",
    )
    bindings = {
        rank: {"pid": next(iter(values))[0], "gpu": next(iter(values))[1]}
        for rank, values in binding_sets.items()
    }
    _require(
        set(waves) == set(range(4))
        and all(waves[wave] == set(GPU_IDS) for wave in range(4))
        and set(bindings) == set(GPU_IDS)
        and {row["gpu"] for row in bindings.values()} == set(GPU_IDS)
        and len({row["pid"] for row in bindings.values()}) == 4,
        "V73E four-wave actor coverage changed",
    )
    guard_source_sha256 = builder.file_sha256(builder.GUARD)
    controller_guard = _validate_guard_process_receipt_v73e(
        guard.get("controller", {}), "controller_sitecustomize"
    )
    worker_bootstraps = [
        _validate_actor_bootstrap_receipt_v73e(row, guard_source_sha256)
        for row in guard.get("workers", [])
    ]
    worker_guards = [row["guard_process_receipt"] for row in worker_bootstraps]
    guard_processes = [controller_guard, *worker_guards]
    _require(
        len(guard_processes) == 5,
        "V73E path guard process coverage changed",
    )
    _require(
        {row["pid"] for row in worker_bootstraps}
        == {row["pid"] for row in bindings.values()}
        and guard["controller"]["pid"]
        not in {row["pid"] for row in worker_bootstraps}
        and guard.get("controller_bootstrap_mechanism")
        == "controller_sitecustomize"
        and guard.get("actor_bootstrap_mechanism")
        == "ray_actor_worker_extension_pre_parent_import"
        and guard.get("mechanisms_are_distinct") is True
        and guard.get("ray_job_runtime_env_bootstrap", {}).get("schema")
        == "qwen36-v73e-ray-job-runtime-env-bootstrap-v1"
        and guard.get("ray_job_runtime_env_bootstrap", {}).get(
            "injected_before_ray_init"
        )
        is True
        and guard.get("ray_job_runtime_env_bootstrap", {}).get(
            "job_env_merges_with_actor_specific_runtime_env"
        )
        is True
        and guard.get("ray_job_runtime_env_bootstrap", {}).get(
            "actor_guard_file_sha256"
        )
        == guard_source_sha256,
        "V73E path guard worker PID attribution changed",
    )
    expected_command = builder.expand_command_v73e(
        arm,
        attempt.get("preregistration_file_sha256", ""),
        preregistration["content_sha256_before_self_field"],
    )
    expected_command_sha = builder.canonical_sha256(expected_command)
    postrun_toolchain_identity = validate_prelaunch_toolchain_v73e(
        preregistration
    )
    accounting = report.get("gpu_time_accounting", {})
    legacy_compute = report.get("compute_ledger", {})
    input_receipt = report.get("content_free_input", {})
    input_body = dict(input_receipt)
    input_sha256 = input_body.pop("receipt_sha256", None)
    staged_contract = report.get("staged_only_adapter_bootstrap", {})
    _require(
        report.get("schema") == "v73e-exact-phase-qwen36-calibration-report"
        and report.get("status")
        == (
            "complete_content_free_self_consistent_no_commit_"
            "awaiting_parent_trace_analysis"
        )
        and report.get("profile_arm") == arm
        and report.get("semantic_quality_selection_or_hpo_performed") is False
        and report.get("raw_prompts_questions_answers_or_outputs_persisted") is False
        and report.get("checkpoint_snapshot_or_promotion_performed") is False
        and report.get("protected_dev_ood_or_holdout_opened") is False
        and report.get("successful_protected_path_opens_or_resolves") == 0
        and report.get(
            "successful_protected_open_resolve_metadata_or_enumeration"
        )
        == 0
        and report.get(
            "quality_hpo_promotion_or_lineage_rehabilitation_performed"
        )
        is False
        and guard.get("status")
        == (
            "zero_successful_protected_open_resolve_metadata_or_"
            "enumeration_systems_only"
        )
        and guard.get("process_count") == 5
        and len(guard.get("workers", [])) == 4
        and guard.get("successful_protected_opens") == 0
        and guard.get("successful_protected_resolves") == 0
        and guard.get("successful_protected_metadata") == 0
        and guard.get("successful_protected_enumerations") == 0
        and guard.get("protected_path_values_persisted") is False
        and guard.get("quality_hpo_or_promotion_performed") is False
        and guard.get("lineage_rehabilitation_performed") is False
        and report.get("immutable_v73d_attempt_1_predecessor")
        == preregistration.get("immutable_v73d_attempt_1_predecessor")
        and report.get("ray_actor_guard_bootstrap")
        == guard.get("ray_job_runtime_env_bootstrap")
        and input_receipt.get("schema")
        == "qwen36-v73e-content-free-systems-input-receipt-v1"
        and input_sha256 == builder.canonical_sha256(input_body)
        and input_receipt.get("content_free_token_panel_content_sha256")
        == builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        and input_receipt.get("request_count")
        == builder.CONTENT_FREE_TOKEN_PANEL_ROWS
        and input_receipt.get("unscored_prefix_tokens_per_candidate")
        == builder.CONTENT_FREE_UNSCORED_PREFIX_TOKENS_PER_CANDIDATE
        and input_receipt.get("answer_tokens_per_candidate")
        == builder.CONTENT_FREE_ANSWER_TOKENS_PER_CANDIDATE
        and input_receipt.get("total_tokens_per_candidate")
        == builder.CONTENT_FREE_TOTAL_TOKENS_PER_CANDIDATE
        and input_receipt.get("evaluation_contract_sha256")
        == builder.CONTENT_FREE_EVALUATION_CONTRACT_SHA256
        and input_receipt.get("qa_dev_or_other_semantic_dataset_used") is False
        and input_receipt.get("protected_semantics_opened") is False
        and staged_contract.get("schema")
        == "qwen36-v73e-staged-only-adapter-contract-v1"
        and staged_contract.get("status")
        == "sealed_staged_only_exact_inverse_verified"
        and staged_contract.get("canonical_master_identity", {}).get("sha256")
        == builder.CANONICAL_MASTER_SHA256
        and staged_contract.get("canonical_runtime_values_sha256")
        == builder.CANONICAL_RUNTIME_VALUES_SHA256
        and staged_contract.get(
            "historical_protected_source_opened_resolved_statted_or_hashed"
        )
        is False
        and accounting.get("schema")
        == "qwen36-v73e-gpu-time-accounting-split-v1"
        and accounting.get("reserved_wall_gpu_seconds")
        == legacy_compute.get("charged_gpu_seconds")
        and accounting.get(
            "legacy_estimated_model_allocation_or_residency_seconds_per_gpu"
        )
        == legacy_compute.get(
            "model_allocation_or_residency_seconds_per_gpu"
        )
        and accounting.get("legacy_residency_estimate_classification")
        == "diagnostic_not_directly_measured_not_accepted"
        and accounting.get("model_resident_gpu_seconds_measured") is None
        and accounting.get("useful_gpu_seconds_measured") is None
        and accounting.get("promotion_charged_gpu_seconds") == 0
        and accounting.get(
            "reserved_wall_is_not_model_residency_or_useful_work"
        )
        is True
        and report.get("expanded_profiler_command_sha256")
        == expected_command_sha
        and attempt.get("status")
        == "prelaunch_accepted_launching_fresh_no_commit_profile"
        and attempt.get("expanded_command") == expected_command
        and attempt.get("expanded_command_sha256") == expected_command_sha
        and attempt.get("prelaunch_toolchain_identity")
        == postrun_toolchain_identity
        and attempt.get("four_gpu_idle_preflight", {}).get("passed") is True
        and attempt.get("four_gpu_idle_preflight", {}).get(
            "no_compute_processes"
        )
        is True
        and attempt.get("four_gpu_idle_preflight", {}).get(
            "all_four_physical_gpus_exactly_attributed"
        )
        is True
        and attempt.get("four_gpu_idle_preflight", {}).get(
            "all_gpu_pairs_node_no_nvlink_all_numa_zero"
        )
        is True
        and attempt.get("four_gpu_idle_preflight", {}).get(
            "actual_nccl_transport_or_path_inferred_from_node"
        )
        is False
        and attempt.get("four_gpu_idle_preflight", {}).get(
            "all_cross_gpu_peer_read_write_ok_nvlink_not_supported"
        )
        is True
        and attempt.get("four_gpu_idle_preflight", {}).get(
            "capability_matrix_used_as_actual_path_evidence"
        )
        is False
        and report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        )
        is True,
        "V73E target authority or final idle changed",
    )
    population_consistency = population.get(
        "content_free_population_consistency", {}
    )
    update_consistency = update.get("content_free_update_consistency", {})
    compiler_consistency = update.get(
        "content_free_current_run_dual_compiler_consistency", {}
    )
    actor_consistency = equivalence.get("actor_work", {})
    _require(
        population.get("content_free_input") == input_receipt
        and population_consistency.get("schema")
        == "qwen36-v73e-content-free-population-consistency-v1"
        and population_consistency.get("plan_sha256")
        == builder.CONTENT_FREE_PLAN_SHA256
        and population_consistency.get("evaluation_contract_sha256")
        == builder.CONTENT_FREE_EVALUATION_CONTRACT_SHA256
        and population_consistency.get("candidate_count") == 16
        and population_consistency.get("candidate_identity_count") == 16
        and population_consistency.get(
            "historical_reward_population_or_semantic_authority_inherited"
        )
        is False
        and update_consistency.get("schema")
        == "qwen36-v73e-content-free-update-consistency-v1"
        and update_consistency.get("plan_sha256")
        == builder.CONTENT_FREE_PLAN_SHA256
        and update_consistency.get("signed_reward_sha256")
        == population_consistency.get("signed_reward_sha256")
        and update_consistency.get("canonical_and_independent_compilers_exact")
        is True
        and compiler_consistency.get("schema")
        == "qwen36-v73e-current-run-dual-compiler-consistency-v1"
        and compiler_consistency.get("current_run_signed_reward_sha256")
        == population_consistency.get("signed_reward_sha256")
        and compiler_consistency.get("coefficient_sha256")
        == update_consistency.get("coefficient_sha256")
        and compiler_consistency.get(
            "canonical_and_independent_compiler_whole_mapping_exact"
        )
        is True
        and equivalence.get("schema")
        == "qwen36-v73e-content-free-systems-consistency-v1"
        and equivalence.get("content_free_input") == input_receipt
        and equivalence.get("population") == population_consistency
        and equivalence.get("update") == update_consistency
        and equivalence.get("current_run_dual_compiler")
        == compiler_consistency
        and equivalence.get("accepted_historical_control_or_equivalence_present")
        is False
        and actor_consistency.get("schema")
        == "qwen36-v73e-content-free-actor-work-consistency-v1"
        and actor_consistency.get("receipt_count") == 16
        and actor_consistency.get("sealed_plan_sha256")
        == builder.CONTENT_FREE_PLAN_SHA256
        and update.get("all_four_abort_receipts_exact") is True
        and update.get("master_committed") is False
        and equivalence.get("master_committed") is False
        and traffic.get("exact_match") is True,
        "V73E content-free consistency, abort, or traffic semantics changed",
    )
    _validate_phase_ranges(
        [
            {
                "text": row["phase"],
                "pid": row["controller_pid"],
                "tid": 0,
                "start_ns": row["started_monotonic_ns"],
                "end_ns": row["ended_monotonic_ns"],
            }
            for row in phase["rows"]
        ],
        phase,
    )
    return {
        "report_content_sha256": report["content_sha256_before_self_field"],
        "population_content_sha256": population[
            "content_sha256_before_self_field"
        ],
        "update_content_sha256": update["content_sha256_before_self_field"],
        "phase_receipt_content_sha256": phase[
            "content_sha256_before_self_field"
        ],
        "path_guard_receipt_content_sha256": guard[
            "content_sha256_before_self_field"
        ],
        "actor_receipt_count": len(rows),
        "actor_pids": sorted(row["pid"] for row in bindings.values()),
        "actor_gpus": sorted(row["gpu"] for row in bindings.values()),
        "physical_gpu_by_actor_pid": {
            str(row["pid"]): row["gpu"] for row in bindings.values()
        },
        "expanded_profiler_command_sha256": expected_command_sha,
        "prelaunch_four_gpu_idle_preflight": attempt[
            "four_gpu_idle_preflight"
        ],
        "prelaunch_and_postrun_toolchain_identity": (
            postrun_toolchain_identity
        ),
        "phase_receipt": phase,
        "content_free_current_run_compilers_exact": True,
        "four_actor_update_identity_consensus": True,
        "exact_abort_no_commit_final_idle": True,
        "protected_or_semantic_evaluation_opened": False,
        "successful_protected_path_open_or_resolve_count": 0,
        "successful_protected_open_resolve_metadata_or_enumeration_count": 0,
        "quality_hpo_promotion_or_lineage_rehabilitation_performed": False,
    }


def _top_three(sqlite: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for phase, by_gpu in sqlite["cuda_by_phase_and_gpu"].items():
        union = sum(item["gpu_event_union_ns"] for item in by_gpu.values())
        copied = sum(item["memcpy_bytes"] for item in by_gpu.values())
        candidates.append({
            "phase": phase,
            "sum_gpu_event_union_ns_across_four_gpus": union,
            "cuda_memcpy_bytes": copied,
        })
    return sorted(
        candidates,
        key=lambda row: (
            row["sum_gpu_event_union_ns_across_four_gpus"],
            row["cuda_memcpy_bytes"],
            row["phase"],
        ),
        reverse=True,
    )[:3]


def analyze_profile_v73e(
    preregistration: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    _require(arm == "timeline", "current V73E analyzer accepts timeline arm only")
    paths = preregistration["arms"][arm]["artifacts"]
    target = validate_target_artifacts_v73e(preregistration, arm)
    report_path = Path(paths["nsys_report"])
    sqlite_path = Path(paths["sqlite_export"])
    _require(
        report_path.is_file() and not report_path.is_symlink(),
        "V73E Nsight report absent",
    )
    sqlite_result = analyze_sqlite_v73e(
        sqlite_path,
        target["phase_receipt"],
        {
            int(pid): int(gpu)
            for pid, gpu in target["physical_gpu_by_actor_pid"].items()
        },
    )
    nccl_debug = _analyze_nccl_debug_logs_v73e(
        paths["nccl_debug_pattern"], set(target["actor_pids"])
    )
    value = {
        "schema": "qwen36-v73e-exact-phase-profile-analysis-v1",
        "status": "timeline_accepted_hbm_metrics_still_permission_blocked",
        "arm": arm,
        "nsys_report": {
            "path": str(report_path.resolve()),
            "file_sha256": builder.file_sha256(report_path),
            "bytes": report_path.stat().st_size,
        },
        "sqlite": sqlite_result,
        "nccl_debug_transport_metadata": nccl_debug,
        "target_semantics": {
            key: value for key, value in target.items() if key != "phase_receipt"
        },
        "top_three_timeline_residuals": _top_three(sqlite_result),
        "ranking_scope": (
            "profiled CUDA event union and memcpy bytes; diagnostic only because "
            "the profiler changes timing"
        ),
        "unprofiled_timing_control": preregistration[
            "accepted_unprofiled_control"
        ],
        "profiled_timing_used_for_throughput_promotion": False,
        "hbm_dram_counter_arm_complete": False,
        "remaining_blocker": "RmProfilingAdminOnly=1/ERR_NVGPUCTRPERM",
        "protected_dev_ood_or_holdout_opened": False,
        "raw_prompts_questions_answers_or_outputs_persisted": False,
    }
    value["content_sha256_before_self_field"] = builder.canonical_sha256(value)
    return value


def _fresh_launch(paths: Mapping[str, str]) -> None:
    top_level = (
        "application_attempt",
        "run_directory",
        "profile_attempt",
        "profile_directory",
    )
    _require(
        all(not Path(paths[name]).exists() for name in top_level),
        "V73E requires fresh application/profile paths",
    )


def validate_prelaunch_toolchain_v73e(
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-hash every timeline executable/runtime input before any launch write."""

    toolchain = preregistration["toolchain"]

    def exact_source(
        expected_invocation: Path,
        sealed: Mapping[str, Any],
        label: str,
    ) -> dict[str, str]:
        invocation = Path(expected_invocation)
        sealed_path = Path(str(sealed.get("path", "")))
        _require(
            invocation.is_file()
            and sealed_path.is_file()
            and invocation.resolve() == sealed_path.resolve()
            and builder.file_sha256(invocation) == sealed.get("file_sha256"),
            f"V73E prelaunch tool identity changed: {label}",
        )
        return {
            "label": label,
            "resolved_path": str(invocation.resolve()),
            "file_sha256": str(sealed["file_sha256"]),
        }

    checked = [
        exact_source(
            builder.BASH,
            toolchain["launch_shell"]["bash"],
            "launch bash",
        ),
        exact_source(
            builder.BASHRC,
            toolchain["launch_shell"]["bashrc"],
            "launch bashrc",
        ),
        exact_source(
            builder.NVIDIA_SMI,
            toolchain["nvidia_smi"],
            "nvidia-smi",
        ),
        exact_source(builder.ENV, toolchain["environment_exec"], "env"),
        exact_source(
            builder.NSYS,
            toolchain["nsight_systems"],
            "nsight-systems invocation",
        ),
        exact_source(
            Path(toolchain["nsight_systems"]["native"]["path"]),
            toolchain["nsight_systems"]["native"],
            "nsight-systems native binary",
        ),
        exact_source(
            Path(toolchain["python_nvtx"]["package_init"]["path"]),
            toolchain["python_nvtx"]["package_init"],
            "python nvtx package",
        ),
        exact_source(
            Path(toolchain["python_nvtx"]["native_extension"]["path"]),
            toolchain["python_nvtx"]["native_extension"],
            "python nvtx native extension",
        ),
        exact_source(
            Path(toolchain["python_nvtx"]["nvtools_library"]["path"]),
            toolchain["python_nvtx"]["nvtools_library"],
            "nvToolsExt library",
        ),
        exact_source(
            Path(toolchain["nccl"]["package_metadata"]["path"]),
            toolchain["nccl"]["package_metadata"],
            "NCCL package metadata",
        ),
        exact_source(
            Path(toolchain["nccl"]["library"]["path"]),
            toolchain["nccl"]["library"],
            "NCCL library",
        ),
    ]
    python = toolchain["target_python"]
    _require(
        Path(str(python.get("invocation_path", ""))).absolute()
        == builder.REQUIRED_PYTHON,
        "V73E target Python invocation path changed",
    )
    checked.append(exact_source(
        builder.REQUIRED_PYTHON,
        python["resolved_file"],
        "target Python",
    ))
    return {
        "schema": "qwen36-v73e-prelaunch-toolchain-identity-v1",
        "passed": True,
        "checked_source_count": len(checked),
        "checked_sources": checked,
        "environment_values_persisted": False,
    }


def _run_text(command: Sequence[str]) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return completed.stdout.strip()


def four_gpu_idle_preflight_v73e(
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    topology = builder.gpu_topology_v73e()
    _require(
        topology == preregistration["physical_gpu_topology"],
        "V73E physical GPU topology changed before launch",
    )
    gpu_command = [
        str(builder.NVIDIA_SMI),
        "--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    process_command = [
        str(builder.NVIDIA_SMI),
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    rows = []
    for line in _run_text(gpu_command).splitlines():
        fields = [field.strip() for field in line.split(",")]
        _require(len(fields) == 5, "V73E idle GPU query shape changed")
        rows.append({
            "physical_gpu_id": int(fields[0]),
            "uuid": fields[1],
            "pci_bus_id": fields[2],
            "memory_used_mib": int(fields[3]),
            "utilization_percent": int(fields[4]),
        })
    identity = [
        {
            "physical_gpu_id": row["physical_gpu_id"],
            "uuid": row["uuid"],
            "pci_bus_id": row["pci_bus_id"],
        }
        for row in rows
    ]
    _require(
        identity == preregistration["physical_gpu_identity"],
        "V73E physical GPU attribution changed before launch",
    )
    raw_processes = _run_text(process_command)
    process_rows = []
    for line in raw_processes.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        _require(len(fields) == 4, "V73E compute process query shape changed")
        process_rows.append({
            "gpu_uuid": fields[0],
            "pid": int(fields[1]),
            "process_name": fields[2],
            "used_gpu_memory_mib": int(fields[3]),
        })
    no_compute_processes = not process_rows
    utilization_idle = all(row["utilization_percent"] == 0 for row in rows)
    driver_only_memory = all(row["memory_used_mib"] <= 32 for row in rows)
    passed = (
        len(rows) == 4
        and no_compute_processes
        and utilization_idle
        and driver_only_memory
        and topology["actual_nccl_transport_or_path_inferred_from_node"] is False
        and topology["capability_matrix_used_as_actual_path_evidence"] is False
    )
    return {
        "schema": "qwen36-v73e-four-gpu-idle-preflight-v1",
        "gpu_query": gpu_command,
        "compute_process_query": process_command,
        "physical_gpu_rows": rows,
        "physical_gpu_topology": topology,
        "compute_process_rows": process_rows,
        "all_four_physical_gpus_exactly_attributed": len(rows) == 4,
        "all_gpu_pairs_node_no_nvlink_all_numa_zero": (
            topology["all_off_diagonal_gpu_pairs_report_node"] is True
            and topology["nvlink_pair_label_present"] is False
            and topology["all_numa_affinity_zero"] is True
        ),
        "all_cross_gpu_peer_read_write_ok_nvlink_not_supported": (
            topology["all_cross_gpu_peer_read_status_ok"] is True
            and topology["all_cross_gpu_peer_write_status_ok"] is True
            and topology["all_cross_gpu_nvlink_status_not_supported"] is True
        ),
        "capability_matrix_used_as_actual_path_evidence": False,
        "actual_nccl_transport_or_path_inferred_from_node": False,
        "foreign_or_other_compute_process_count": len(process_rows),
        "no_compute_processes": no_compute_processes,
        "all_utilization_samples_zero": utilization_idle,
        "all_memory_is_driver_baseline_at_most_32_mib": driver_only_memory,
        "passed": passed,
    }


def profile_failure_receipt_v73e(
    preregistration: Mapping[str, Any],
    arm: str,
    returncode: int,
    expanded_command_sha256: str,
) -> dict[str, Any]:
    paths = preregistration["arms"][arm]["artifacts"]
    run_failure_path = Path(paths["failure"])
    run_failure_reference = None
    ray_bootstrap = None
    accounting = {
        "schema": "qwen36-v73e-gpu-time-accounting-split-v1",
        "reserved_wall_gpu_seconds": None,
        "reserved_wall_source": "legacy_compute_ledger_charged_gpu_seconds",
        "legacy_estimated_model_allocation_or_residency_seconds_per_gpu": None,
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
    if run_failure_path.is_file():
        run_failure = _load_json(run_failure_path)
        content_sha256 = _validate_self_hash(run_failure, "run failure")
        run_failure_reference = {
            "path": str(run_failure_path.resolve()),
            "file_sha256": builder.file_sha256(run_failure_path),
            "content_sha256": content_sha256,
        }
        ray_bootstrap = copy.deepcopy(
            run_failure.get("ray_actor_guard_bootstrap")
        )
        observed_accounting = run_failure.get("gpu_time_accounting")
        _require(
            isinstance(observed_accounting, dict)
            and observed_accounting.get("schema")
            == "qwen36-v73e-gpu-time-accounting-split-v1"
            and observed_accounting.get("model_resident_gpu_seconds_measured")
            is None
            and observed_accounting.get("useful_gpu_seconds_measured") is None
            and observed_accounting.get("promotion_charged_gpu_seconds") == 0
            and observed_accounting.get(
                "reserved_wall_is_not_model_residency_or_useful_work"
            )
            is True,
            "V73E run failure GPU accounting split changed",
        )
        accounting = copy.deepcopy(observed_accounting)
    return {
        "schema": "qwen36-v73e-profiler-launch-failure-v1",
        "status": "target_or_profiler_failed_closed",
        "arm": arm,
        "returncode": int(returncode),
        "expanded_command_sha256": expanded_command_sha256,
        "run_failure_exists": run_failure_reference is not None,
        "run_failure": run_failure_reference,
        "ray_actor_guard_bootstrap": ray_bootstrap,
        "immutable_v73d_attempt_1_predecessor": preregistration[
            "immutable_v73d_attempt_1_predecessor"
        ],
        "gpu_time_accounting": accounting,
        "protected_dev_ood_or_holdout_opened": False,
        "successful_protected_open_resolve_metadata_or_enumeration": 0,
        "checkpoint_snapshot_commit_or_promotion_performed": False,
    }


def execute_profile_v73e(
    preregistration: Mapping[str, Any],
    preregistration_file_sha256: str,
    arm: str,
) -> dict[str, Any]:
    arm_contract = preregistration["arms"][arm]
    _require(
        arm_contract["launch_authorized_by_this_file_after_identity_checks"] is True,
        "V73E HBM metrics arm blocked before directory creation or subprocess",
    )
    toolchain_identity = validate_prelaunch_toolchain_v73e(preregistration)
    paths = arm_contract["artifacts"]
    _fresh_launch(paths)
    command = builder.expand_command_v73e(
        arm,
        preregistration_file_sha256,
        preregistration["content_sha256_before_self_field"],
    )
    command_sha = builder.canonical_sha256(command)
    preflight = four_gpu_idle_preflight_v73e(preregistration)
    attempt = {
        "schema": "qwen36-v73e-profiler-launch-attempt-v1",
        "status": (
            "prelaunch_accepted_launching_fresh_no_commit_profile"
            if preflight["passed"]
            else "prelaunch_rejected_no_profiler_process_created"
        ),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "arm": arm,
        "expanded_command": command,
        "expanded_command_sha256": command_sha,
        "four_gpu_idle_preflight": preflight,
        "prelaunch_toolchain_identity": toolchain_identity,
        "preregistration_file_sha256": preregistration_file_sha256,
        "preregistration_content_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "output_paths": paths,
        "subprocess_shell_flag": False,
        "fixed_bashrc_exec_wrapper": True,
        "bashrc_file_sha256": preregistration["toolchain"]["launch_shell"][
            "bashrc"
        ]["file_sha256"],
        "force_overwrite": False,
        "protected_dev_ood_or_holdout_opened": False,
        "checkpoint_snapshot_commit_or_promotion_authorized": False,
    }
    _atomic_json(Path(paths["profile_attempt"]), attempt)
    _require(
        preflight["passed"] is True,
        "V73E requires all four GPUs idle and foreign-process-free before launch",
    )
    Path(paths["profile_directory"]).mkdir(mode=0o755)
    environment = dict(os.environ)
    environment[COMMAND_ATTESTATION_ENV] = command_sha
    completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
    if completed.returncode != 0:
        failure = profile_failure_receipt_v73e(
            preregistration, arm, completed.returncode, command_sha
        )
        _atomic_json(Path(paths["profile_failure"]), failure)
        raise RuntimeError(
            f"V73E profiler target failed with return code {completed.returncode}"
        )
    result = analyze_profile_v73e(preregistration, arm)
    _atomic_json(Path(paths["profile_analysis"]), result)
    return result


def parser_v73e() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(builder.OUTPUT))
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--arm", choices=("timeline", "hbm_metrics"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv=None) -> int:
    args = parser_v73e().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("V73E profiler requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v73e(
        Path(args.preregistration),
        args.preregistration_sha256,
        args.preregistration_content_sha256,
    )
    arm = preregistration["arms"][args.arm]
    command = builder.expand_command_v73e(
        args.arm,
        args.preregistration_sha256,
        args.preregistration_content_sha256,
    )
    if args.dry_run:
        print(json.dumps({
            "schema": preregistration["schema"],
            "arm": args.arm,
            "arm_status": arm["status"],
            "launch_authorized": arm[
                "launch_authorized_by_this_file_after_identity_checks"
            ],
            "expanded_command": command,
            "expanded_command_sha256": builder.canonical_sha256(command),
            "output_paths": arm["artifacts"],
            "model_dataset_ray_cuda_or_gpu_opened": False,
            "filesystem_writes": False,
            "protected_dev_ood_or_holdout_opened": False,
        }, sort_keys=True))
        return 0
    result = execute_profile_v73e(
        preregistration, args.preregistration_sha256, args.arm
    )
    print(json.dumps({
        "status": result["status"],
        "analysis": preregistration["arms"][args.arm]["artifacts"][
            "profile_analysis"
        ],
        "content_sha256": result["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
