#!/usr/bin/env python3
"""Prospective exact-FP32 LoRA collective coalescing contract (V83A).

This module is deliberately CPU-only.  It seals deterministic bucket layouts
for the accepted V72 70-tensor LoRA update and provides a synthetic transaction
used to prove packing, reduction, unpacking, candidate, and rollback behavior.
It is not imported by V72 and cannot execute on CUDA.  A future live adapter
must preserve the bound PyNccl-style call, stream/event ordering, and failure
semantics, and remains gated on separate V73D materiality evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch


SCHEMA_V83A = "eggroll-es-exact-fp32-collective-coalescing-v83a"
PLAN_SCHEMA_V83A = "qwen36-lora-fp32-coalescing-plan-v83a"
TRANSACTION_SCHEMA_V83A = "synthetic-fp32-coalesced-update-v83a"
SYNTHETIC_AUTHORITY_V83A = "synthetic_cpu_fake_four_rank_only_v83a"

WORLD_SIZE_V83A = 4
TENSOR_COUNT_V83A = 70
MODULE_COUNT_V83A = 35
TOTAL_ELEMENTS_V83A = 4_528_128
TOTAL_BYTES_V83A = 18_112_512
MAX_TENSOR_ELEMENTS_V83A = 262_144
FP32_BYTES_V83A = 4
NATIVE_COLLECTIVE_CALLS_V83A = 70
NATIVE_MAX_ACCUMULATOR_BYTES_V83A = 1_048_576

EXPECTED_SOURCE_MANIFEST_SHA256_V83A = (
    "e12f7199343477db3927bda67bf5f364030a47216be8aa2b30fc3b71c261da2b"
)
EXPECTED_ORDERED_KEY_SHA256_V83A = (
    "ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280"
)

# Greedy, source-order, no-split capacities.  The flat plan is the maximum;
# the smaller alternatives trade more calls for a bounded staging allocation.
BUCKET_CHOICES_V83A = (
    ("flat_all_18112512b", TOTAL_ELEMENTS_V83A),
    ("bounded_8mib", (8 << 20) // FP32_BYTES_V83A),
    ("bounded_4mib", (4 << 20) // FP32_BYTES_V83A),
    ("bounded_2mib", (2 << 20) // FP32_BYTES_V83A),
)
BUCKET_CAPACITY_ELEMENTS_V83A = dict(BUCKET_CHOICES_V83A)


def canonical_sha256_v83a(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_v83a(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _exact_int_v83a(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"V83A {label} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"V83A {label} must be >= {minimum}")
    return result


def _source_records_v83a(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("V83A source manifest must be a sequence")
    _require_v83a(
        len(records) == TENSOR_COUNT_V83A,
        "V83A source tensor count changed",
    )
    result: list[dict[str, Any]] = []
    fields = {"dtype", "elements", "key", "module", "ordinal", "role", "shape"}
    for ordinal, source in enumerate(records):
        _require_v83a(
            isinstance(source, Mapping) and set(source) == fields,
            f"V83A source fields changed at {ordinal}",
        )
        shape = source["shape"]
        elements = _exact_int_v83a(source["elements"], "source elements", 1)
        key = source["key"]
        module = source["module"]
        role = source["role"]
        _require_v83a(
            source["ordinal"] == ordinal
            and source["dtype"] == "float32"
            and isinstance(shape, list)
            and len(shape) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in shape
            )
            and math.prod(shape) == elements
            and isinstance(key, str)
            and isinstance(module, str)
            and role in ("A", "B")
            and key == f"base_model.model.{module}.lora_{role}.weight",
            f"V83A source record changed at {ordinal}",
        )
        result.append(
            {
                "ordinal": ordinal,
                "key": key,
                "module": module,
                "role": role,
                "shape": list(shape),
                "elements": elements,
                "dtype": "float32",
            }
        )
    keys = [row["key"] for row in result]
    _require_v83a(
        keys == sorted(keys)
        and len(set(keys)) == TENSOR_COUNT_V83A
        and len({row["module"] for row in result}) == MODULE_COUNT_V83A
        and sum(row["elements"] for row in result) == TOTAL_ELEMENTS_V83A
        and max(row["elements"] for row in result) == MAX_TENSOR_ELEMENTS_V83A
        and canonical_sha256_v83a(result)
        == EXPECTED_SOURCE_MANIFEST_SHA256_V83A
        and canonical_sha256_v83a(keys) == EXPECTED_ORDERED_KEY_SHA256_V83A,
        "V83A sealed ordered LoRA surface changed",
    )
    return result


def build_bucket_plan_v83a(
    records: Sequence[Mapping[str, Any]], choice: str
) -> dict[str, Any]:
    """Build one deterministic greedy, no-split bucket plan."""
    source = _source_records_v83a(records)
    if choice not in BUCKET_CAPACITY_ELEMENTS_V83A:
        raise ValueError(f"V83A unknown bucket choice: {choice}")
    capacity = BUCKET_CAPACITY_ELEMENTS_V83A[choice]
    _require_v83a(
        capacity >= MAX_TENSOR_ELEMENTS_V83A,
        "V83A bucket cannot hold the largest source tensor",
    )

    buckets: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    pending_elements = 0
    global_cursor = 0

    def close_bucket() -> None:
        nonlocal pending, pending_elements
        if not pending:
            return
        bucket_ordinal = len(buckets)
        bucket = {
            "ordinal": bucket_ordinal,
            "elements": pending_elements,
            "bytes": pending_elements * FP32_BYTES_V83A,
            "entries": pending,
        }
        bucket["content_sha256"] = canonical_sha256_v83a(bucket)
        buckets.append(bucket)
        pending = []
        pending_elements = 0

    for row in source:
        elements = row["elements"]
        if pending and pending_elements + elements > capacity:
            close_bucket()
        bucket_start = pending_elements
        entry = {
            "source_ordinal": row["ordinal"],
            "key": row["key"],
            "module": row["module"],
            "role": row["role"],
            "shape": row["shape"],
            "dtype": "float32",
            "elements": elements,
            "bytes": elements * FP32_BYTES_V83A,
            "global_start": global_cursor,
            "global_end": global_cursor + elements,
            "bucket_start": bucket_start,
            "bucket_end": bucket_start + elements,
        }
        pending.append(entry)
        pending_elements += elements
        global_cursor += elements
    close_bucket()

    maximum_bucket_elements = max(row["elements"] for row in buckets)
    maximum_bucket_bytes = maximum_bucket_elements * FP32_BYTES_V83A
    pack_hbm = TOTAL_ELEMENTS_V83A * 2 * FP32_BYTES_V83A
    unpack_hbm = TOTAL_ELEMENTS_V83A * 2 * FP32_BYTES_V83A
    body = {
        "schema": PLAN_SCHEMA_V83A,
        "choice": choice,
        "algorithm": "source_order_greedy_no_tensor_split",
        "capacity_elements": capacity,
        "capacity_bytes": capacity * FP32_BYTES_V83A,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256_V83A,
        "ordered_key_sha256": EXPECTED_ORDERED_KEY_SHA256_V83A,
        "world_size": WORLD_SIZE_V83A,
        "dtype": "float32",
        "tensor_count": TENSOR_COUNT_V83A,
        "module_count": MODULE_COUNT_V83A,
        "total_elements": TOTAL_ELEMENTS_V83A,
        "total_bytes": TOTAL_BYTES_V83A,
        "native_collective_calls": NATIVE_COLLECTIVE_CALLS_V83A,
        "coalesced_collective_calls": len(buckets),
        "collective_calls_eliminated": NATIVE_COLLECTIVE_CALLS_V83A - len(buckets),
        "maximum_bucket_elements": maximum_bucket_elements,
        "maximum_bucket_bytes": maximum_bucket_bytes,
        "buckets": buckets,
        "byte_accounting": {
            "fp32_payload_bytes_per_actor_per_update": TOTAL_BYTES_V83A,
            "nominal_ring_bus_bytes_per_actor_per_update": (
                TOTAL_BYTES_V83A * 2 * (WORLD_SIZE_V83A - 1) // WORLD_SIZE_V83A
            ),
            "network_payload_change_versus_native_bytes": 0,
            "sequential_reusable_gpu_staging_bytes": maximum_bucket_bytes,
            "native_maximum_gpu_accumulator_bytes": (
                NATIVE_MAX_ACCUMULATOR_BYTES_V83A
            ),
            "incremental_gpu_staging_bytes_versus_native_maximum": (
                maximum_bucket_bytes - NATIVE_MAX_ACCUMULATOR_BYTES_V83A
            ),
            "materialized_pack_read_plus_write_hbm_bytes": pack_hbm,
            "materialized_gpu_unpack_read_plus_write_hbm_bytes": unpack_hbm,
            "conservative_pack_plus_gpu_unpack_hbm_bytes": pack_hbm + unpack_hbm,
            "unchanged_d2h_source_hbm_read_bytes": TOTAL_BYTES_V83A,
            "direct_fill_extra_pack_or_gpu_unpack_hbm_bytes": 0,
            "direct_fill_zero_is_design_target_not_live_measurement": True,
            "nccl_internal_hbm_bytes_excluded": True,
        },
        "future_runtime_contract": {
            "communicator_expression": "self.inter_pg.all_reduce",
            "call_expression": (
                "self.inter_pg.all_reduce(bucket, out_tensor=bucket, stream=stream)"
            ),
            "in_place_fp32": True,
            "one_ordered_stream_for_fill_collective_scale_and_d2h": True,
            "event_recorded_after_scale": True,
            "event_synchronized_before_host_consumption_or_staging_reuse": True,
            "partial_candidate_never_committed": True,
            "failure_restores_original_master_or_terminally_poisons": True,
            "stale_transaction_or_bucket_generation_reuse_rejected": True,
        },
    }
    return {**body, "content_sha256": canonical_sha256_v83a(body)}


def _records_from_plan_v83a(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    buckets = value.get("buckets") if isinstance(value, Mapping) else None
    _require_v83a(isinstance(buckets, list), "V83A plan buckets are absent")
    records: list[dict[str, Any]] = []
    for bucket in buckets:
        _require_v83a(isinstance(bucket, Mapping), "V83A bucket changed")
        entries = bucket.get("entries")
        _require_v83a(isinstance(entries, list), "V83A bucket entries changed")
        for entry in entries:
            _require_v83a(isinstance(entry, Mapping), "V83A entry changed")
            records.append(
                {
                    "ordinal": entry.get("source_ordinal"),
                    "key": entry.get("key"),
                    "module": entry.get("module"),
                    "role": entry.get("role"),
                    "shape": entry.get("shape"),
                    "elements": entry.get("elements"),
                    "dtype": entry.get("dtype"),
                }
            )
    return records


def validate_bucket_plan_v83a(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("V83A bucket plan must be a mapping")
    records = _records_from_plan_v83a(value)
    expected = build_bucket_plan_v83a(records, str(value.get("choice")))
    _require_v83a(dict(value) == expected, "V83A bucket plan changed")
    return copy.deepcopy(expected)


def _tensor_raw_v83a(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().to(device="cpu").contiguous().view(torch.uint8)
    return value.numpy().tobytes()


def _tensor_sha256_v83a(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_raw_v83a(tensor)).hexdigest()


def mapping_identity_v83a(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    records = []
    for key, tensor in tensors.items():
        records.append(
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "elements": int(tensor.numel()),
                "sha256": _tensor_sha256_v83a(tensor),
            }
        )
    body = {
        "tensor_count": len(records),
        "elements": sum(row["elements"] for row in records),
        "bytes": sum(row["elements"] * FP32_BYTES_V83A for row in records),
        "tensors": records,
    }
    return {**body, "sha256": canonical_sha256_v83a(body)}


def _storage_span_v83a(tensor: torch.Tensor) -> tuple[int, int]:
    start = int(tensor.data_ptr())
    return start, start + int(tensor.numel() * tensor.element_size())


def _validate_tensor_mapping_v83a(
    tensors: Mapping[str, torch.Tensor],
    plan: Mapping[str, Any],
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, Mapping):
        raise TypeError(f"V83A {label} must be a mapping")
    entries = [entry for bucket in plan["buckets"] for entry in bucket["entries"]]
    keys = [entry["key"] for entry in entries]
    _require_v83a(list(tensors) == keys, f"V83A {label} key order changed")
    result = dict(tensors)
    spans: list[tuple[int, int, str]] = []
    for entry in entries:
        key = entry["key"]
        tensor = result[key]
        _require_v83a(
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and tensor.is_contiguous()
            and list(tensor.shape) == entry["shape"]
            and bool(torch.isfinite(tensor).all()),
            f"V83A {label} tensor changed: {key}",
        )
        start, end = _storage_span_v83a(tensor)
        for other_start, other_end, other_key in spans:
            _require_v83a(
                end <= other_start or start >= other_end,
                f"V83A {label} storage aliases: {other_key} / {key}",
            )
        spans.append((start, end, key))
    return result


class ExactFP32CoalescedUpdateV83A:
    """Synthetic CPU transaction that mirrors the prospective call boundary.

    CUDA tensors are rejected.  The class exists to make the future interface
    and transactional invariants executable under fake-four-rank tests; it is
    not a live training implementation.
    """

    def __init__(
        self,
        master: Mapping[str, torch.Tensor],
        plan: Mapping[str, Any],
        communicator: Any,
        stream: Any,
        event_factory: Callable[[], Any],
        *,
        authority: str,
    ) -> None:
        if authority != SYNTHETIC_AUTHORITY_V83A:
            raise RuntimeError("V83A live authority is absent")
        self.plan = validate_bucket_plan_v83a(plan)
        self.original_master = _validate_tensor_mapping_v83a(
            master, self.plan, "original master"
        )
        self.original_identity = mapping_identity_v83a(self.original_master)
        _require_v83a(
            getattr(communicator, "rank", None) in range(WORLD_SIZE_V83A)
            and getattr(communicator, "world_size", None) == WORLD_SIZE_V83A,
            "V83A fake communicator topology changed",
        )
        if stream is None or not callable(event_factory):
            raise ValueError("V83A stream and event factory are required")
        self.communicator = communicator
        self.stream = stream
        self.event_factory = event_factory
        self.phase = "ready"
        self.current_master = self.original_master
        self.pending_candidate: dict[str, torch.Tensor] | None = None
        self._execute_receipt: dict[str, Any] | None = None

    def _original_exact_v83a(self, boundary: str) -> dict[str, Any]:
        current = mapping_identity_v83a(self.original_master)
        _require_v83a(
            current == self.original_identity,
            f"V83A original master changed at {boundary}",
        )
        return current

    def execute_v83a(
        self,
        producer: Callable[[Mapping[str, Any]], torch.Tensor],
        *,
        scale: float,
    ) -> dict[str, Any]:
        if self.phase != "ready":
            raise RuntimeError("V83A stale transaction reuse rejected")
        if not callable(producer):
            raise TypeError("V83A producer must be callable")
        scale = float(scale)
        if not math.isfinite(scale) or scale == 0.0:
            raise ValueError("V83A scale must be finite and nonzero")
        self._original_exact_v83a("execute_preflight")
        self.phase = "executing"
        candidate: dict[str, torch.Tensor] = {}
        update_records: list[dict[str, Any]] = []
        bucket_receipts: list[dict[str, Any]] = []
        try:
            for bucket in self.plan["buckets"]:
                staging = torch.empty(
                    bucket["elements"], dtype=torch.float32, device="cpu"
                )
                staging_span = _storage_span_v83a(staging)
                seen: list[str] = []
                for entry in bucket["entries"]:
                    key = entry["key"]
                    source = producer(copy.deepcopy(entry))
                    _require_v83a(
                        isinstance(source, torch.Tensor)
                        and source.device.type == "cpu"
                        and source.dtype == torch.float32
                        and source.is_contiguous()
                        and list(source.shape) == entry["shape"]
                        and bool(torch.isfinite(source).all()),
                        f"V83A producer tensor changed: {key}",
                    )
                    source_span = _storage_span_v83a(source)
                    _require_v83a(
                        source_span[1] <= staging_span[0]
                        or source_span[0] >= staging_span[1],
                        f"V83A source aliases staging: {key}",
                    )
                    for master_tensor in self.original_master.values():
                        master_span = _storage_span_v83a(master_tensor)
                        _require_v83a(
                            source_span[1] <= master_span[0]
                            or source_span[0] >= master_span[1],
                            f"V83A source aliases original master: {key}",
                        )
                    staging[
                        entry["bucket_start"] : entry["bucket_end"]
                    ].copy_(source.reshape(-1))
                    seen.append(key)
                _require_v83a(
                    seen == [entry["key"] for entry in bucket["entries"]],
                    "V83A bucket fill coverage changed",
                )
                reduced = self.communicator.all_reduce(
                    staging, out_tensor=staging, stream=self.stream
                )
                _require_v83a(
                    isinstance(reduced, torch.Tensor)
                    and reduced.device.type == "cpu"
                    and reduced.dtype == torch.float32
                    and reduced.is_contiguous()
                    and reduced.shape == staging.shape
                    and reduced.data_ptr() == staging.data_ptr()
                    and reduced.storage_offset() == staging.storage_offset(),
                    "V83A PyNccl-style reduction return changed",
                )
                reduced.mul_(scale)
                event = self.event_factory()
                _require_v83a(
                    callable(getattr(event, "record", None))
                    and callable(getattr(event, "synchronize", None)),
                    "V83A event interface changed",
                )
                event.record(self.stream)
                event.synchronize()
                for entry in bucket["entries"]:
                    key = entry["key"]
                    host_delta = reduced[
                        entry["bucket_start"] : entry["bucket_end"]
                    ].reshape(entry["shape"]).clone().contiguous()
                    update_records.append(
                        {
                            "key": key,
                            "shape": list(host_delta.shape),
                            "dtype": str(host_delta.dtype),
                            "elements": int(host_delta.numel()),
                            "sha256": _tensor_sha256_v83a(host_delta),
                        }
                    )
                    candidate[key] = self.original_master[key].add(
                        host_delta
                    ).contiguous()
                bucket_receipts.append(
                    {
                        "ordinal": bucket["ordinal"],
                        "elements": bucket["elements"],
                        "keys": seen,
                        "collective_return_aliases_input": True,
                        "event_recorded_after_scale": True,
                        "event_synchronized_before_unpack": True,
                    }
                )
            self._original_exact_v83a("execute_postflight")
            candidate = _validate_tensor_mapping_v83a(
                candidate, self.plan, "pending candidate"
            )
            update_body = {
                "tensor_count": len(update_records),
                "elements": sum(row["elements"] for row in update_records),
                "bytes": sum(
                    row["elements"] * FP32_BYTES_V83A for row in update_records
                ),
                "tensors": update_records,
            }
            update_identity = {
                **update_body,
                "sha256": canonical_sha256_v83a(update_body),
            }
            candidate_identity = mapping_identity_v83a(candidate)
        except BaseException:
            candidate.clear()
            self.pending_candidate = None
            self.current_master = self.original_master
            self.phase = "poisoned"
            self._original_exact_v83a("failed_execute")
            raise
        self.pending_candidate = candidate
        self.phase = "executed"
        receipt = {
            "schema": TRANSACTION_SCHEMA_V83A,
            "executed": True,
            "synthetic_cpu_only": True,
            "rank": int(self.communicator.rank),
            "world_size": int(self.communicator.world_size),
            "choice": self.plan["choice"],
            "collective_dtype": "torch.float32",
            "collective_calls": len(bucket_receipts),
            "communicator_expression": "self.inter_pg.all_reduce",
            "call_expression": (
                "self.inter_pg.all_reduce(bucket, out_tensor=bucket, stream=stream)"
            ),
            "update_identity": update_identity,
            "candidate_identity": candidate_identity,
            "original_identity": self.original_identity,
            "bucket_receipts": bucket_receipts,
            "master_committed": False,
            "live_pynccl_executed": False,
        }
        receipt["content_sha256"] = canonical_sha256_v83a(receipt)
        self._execute_receipt = receipt
        return copy.deepcopy(receipt)

    def commit_provisional_v83a(self, expected_candidate_sha256: str) -> dict[str, Any]:
        if self.phase != "executed" or self.pending_candidate is None:
            raise RuntimeError("V83A candidate is not executable for commit")
        identity = mapping_identity_v83a(self.pending_candidate)
        _require_v83a(
            identity["sha256"] == str(expected_candidate_sha256)
            and self._execute_receipt is not None
            and identity == self._execute_receipt["candidate_identity"],
            "V83A candidate identity changed before commit",
        )
        self._original_exact_v83a("provisional_commit")
        self.current_master = self.pending_candidate
        self.phase = "committed_provisional"
        return {
            "schema": "synthetic-fp32-coalesced-provisional-commit-v83a",
            "committed": True,
            "candidate_identity": identity,
            "finalized": False,
        }

    def restore_v83a(self, reason: str) -> dict[str, Any]:
        if self.phase not in ("executed", "committed_provisional"):
            raise RuntimeError("V83A stale restore rejected")
        if not isinstance(reason, str) or not reason:
            raise ValueError("V83A restore reason is required")
        identity = self._original_exact_v83a("restore")
        self.current_master = self.original_master
        self.pending_candidate = None
        self.phase = "restored"
        return {
            "schema": "synthetic-fp32-coalesced-restore-v83a",
            "restored": True,
            "reason": reason,
            "current_master_is_original": self.current_master is self.original_master,
            "restored_identity": identity,
        }

    def finalize_v83a(self, expected_candidate_sha256: str) -> dict[str, Any]:
        if self.phase != "committed_provisional" or self.pending_candidate is None:
            raise RuntimeError("V83A provisional candidate is not finalizable")
        identity = mapping_identity_v83a(self.pending_candidate)
        _require_v83a(
            identity["sha256"] == str(expected_candidate_sha256),
            "V83A final candidate identity changed",
        )
        self.phase = "finalized"
        return {
            "schema": "synthetic-fp32-coalesced-finalized-v83a",
            "finalized": True,
            "candidate_identity": identity,
            "rollback_still_authorized": False,
        }
