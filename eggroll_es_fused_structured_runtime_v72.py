#!/usr/bin/env python3
"""CPU oracle for fused structured-noise writes into runtime LoRA views.

This module is production-shaped but intentionally CPU-only.  It consumes the
sealed absolute-index RNG in :mod:`structured_es_oracle_v1`, projects the
canonical 70-tensor FP32 LoRA surface directly into the packed 82-view BF16
runtime layout, and follows the V71 cheap-transition/exact-boundary lifecycle.

The implementation never materializes a whole noise, candidate, or update
surface.  A full pending FP32 master is allowed only as transaction-owned
output so that V71 commit/final rollback ownership remains possible.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

import eggroll_es_audit_contract_v71 as audit_v71
import structured_es_oracle_v1 as oracle


SCHEMA_V72 = "eggroll-es-fused-structured-runtime-v72"
PROJECTION_SCHEMA_V72 = "qwen36-lora-runtime-projection-v72"
CANDIDATE_SCHEMA_V72 = "fused-structured-runtime-candidate-v72"
UPDATE_SCHEMA_V72 = "streamed-structured-master-update-v72"
MAX_FINAL_UPDATE_ULP_V72 = 2
DEFAULT_CHUNK_ELEMENTS_V72 = oracle.CHUNK_ELEMENTS_V1

EXPECTED_ORACLE_FILE_SHA256_V72 = (
    "8fca35f89744f292ef0d9327f547196dd26f93268336f3fad4812a065f35f740"
)
EXPECTED_V71_AUDIT_FILE_SHA256_V72 = (
    "cc80ac0e1bf3c9db83e3275df16ea1479273d92a40240496163543643bd0eaa8"
)
EXPECTED_SOURCE_TENSORS_V72 = 70
EXPECTED_SOURCE_ELEMENTS_V72 = 4_528_128
EXPECTED_RUNTIME_VIEWS_V72 = 82
EXPECTED_RUNTIME_ELEMENTS_V72 = 4_921_344
EXPECTED_RUNTIME_BF16_BYTES_V72 = 9_842_688


def canonical_sha256_v72(value: Any) -> str:
    return oracle.canonical_sha256_v1(value)


def _exact_int_v72(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"v72 {label} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"v72 {label} must be >= {minimum}")
    return result


def _finite_float_v72(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"v72 {label} must be real")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"v72 {label} must be finite" + ("/positive" if positive else ""))
    return result


def _sha256_string_v72(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"v72 {label} must be lowercase SHA-256")
    return value


def _shape_v72(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"v72 {label} must be a rank-two shape")
    return (
        _exact_int_v72(value[0], f"{label} row", 1),
        _exact_int_v72(value[1], f"{label} column", 1),
    )


def _tensor_raw_v72(tensor: torch.Tensor) -> bytes:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        tensor = tensor.detach().to(device="cpu").contiguous()
    return tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_sha256_v72(tensor: torch.Tensor) -> str:
    return hashlib.sha256(_tensor_raw_v72(tensor)).hexdigest()


def _mapping_identity_v72(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    records = []
    for key in sorted(tensors):
        tensor = tensors[key]
        records.append({
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "elements": int(tensor.numel()),
            "sha256": _tensor_sha256_v72(tensor),
        })
    return {
        "tensor_count": len(records),
        "elements": sum(item["elements"] for item in records),
        "bytes": sum(int(tensors[item["key"]].element_size()) * item["elements"]
                     for item in records),
        "sha256": audit_v71.canonical_sha256_v71(records),
        "tensors": records,
    }


def _validate_assignments_v72(assignments: Sequence[Mapping[str, Any]]) -> list[dict]:
    fields = {
        "peft_key", "runtime_module", "runtime_shape", "segment_count",
        "segment_index", "side", "slice_index", "slot", "source_shape",
    }
    if (
        not isinstance(assignments, Sequence)
        or isinstance(assignments, (str, bytes))
        or not assignments
    ):
        raise ValueError("v72 runtime assignments are absent")
    result = []
    for index, item in enumerate(assignments):
        if not isinstance(item, Mapping) or set(item) != fields:
            raise ValueError(f"v72 assignment {index} fields changed")
        source_key = item["peft_key"]
        runtime_module = item["runtime_module"]
        side = item["side"]
        if (
            not isinstance(source_key, str) or not source_key
            or not isinstance(runtime_module, str) or not runtime_module
            or side not in {"A", "B"}
        ):
            raise ValueError("v72 assignment identity changed")
        source_shape = _shape_v72(item["source_shape"], "source shape")
        runtime_shape = _shape_v72(item["runtime_shape"], "runtime shape")
        segment_count = _exact_int_v72(item["segment_count"], "segment count", 1)
        segment_index = _exact_int_v72(item["segment_index"], "segment index")
        slice_index = _exact_int_v72(item["slice_index"], "slice index")
        slot = _exact_int_v72(item["slot"], "slot")
        if segment_index >= segment_count or slot != 0:
            raise ValueError("v72 assignment segment/slot changed")
        expected_suffix = f".lora_{side}.weight"
        if not source_key.endswith(expected_suffix):
            raise ValueError("v72 assignment side differs from PEFT key")
        result.append({
            "peft_key": source_key,
            "runtime_module": runtime_module,
            "runtime_shape": list(runtime_shape),
            "segment_count": segment_count,
            "segment_index": segment_index,
            "side": side,
            "slice_index": slice_index,
            "slot": slot,
            "source_shape": list(source_shape),
        })
    return result


def build_runtime_projection_manifest_v72(
    assignments: Sequence[Mapping[str, Any]],
    *,
    b_scale: float,
) -> dict[str, Any]:
    """Convert V41/V71 assignment certificates into direct-write spans."""
    rows = _validate_assignments_v72(assignments)
    b_scale = _finite_float_v72(b_scale, "B scale", positive=True)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        grouped[item["peft_key"]].append(item)
    source_tensors = []
    projections = []
    runtime_keys = set()
    for source_key in sorted(grouped):
        group = sorted(grouped[source_key], key=lambda item: item["segment_index"])
        side = group[0]["side"]
        source_shape = tuple(group[0]["source_shape"])
        segment_count = group[0]["segment_count"]
        if (
            any(tuple(item["source_shape"]) != source_shape for item in group)
            or any(item["side"] != side for item in group)
            or any(item["segment_count"] != segment_count for item in group)
            or [item["segment_index"] for item in group] != list(range(segment_count))
        ):
            raise RuntimeError(f"v72 source projection group changed: {source_key}")
        source_elements = math.prod(source_shape)
        source_tensors.append({
            "key": source_key,
            "side": side,
            "shape": list(source_shape),
            "elements": source_elements,
        })
        if side == "A":
            if any(tuple(item["runtime_shape"]) != source_shape for item in group):
                raise RuntimeError("v72 packed A must duplicate the full source")
            spans = [(0, source_elements) for _item in group]
        else:
            if any(item["runtime_shape"][1] != source_shape[1] for item in group):
                raise RuntimeError("v72 packed B column width changed")
            cursor = 0
            spans = []
            for item in group:
                count = math.prod(item["runtime_shape"])
                spans.append((cursor, cursor + count))
                cursor += count
            if cursor != source_elements:
                raise RuntimeError("v72 packed B segments do not cover the source")
        for item, (start, end) in zip(group, spans, strict=True):
            runtime_key = (
                f"{item['runtime_module']}|{side}|slice={item['slice_index']}|slot=0"
            )
            if runtime_key in runtime_keys:
                raise RuntimeError("v72 runtime projection key is duplicated")
            runtime_keys.add(runtime_key)
            if end - start != math.prod(item["runtime_shape"]):
                raise RuntimeError("v72 source/runtime projection extent changed")
            projections.append({
                "runtime_key": runtime_key,
                "runtime_module": item["runtime_module"],
                "source_key": source_key,
                "source_shape": list(source_shape),
                "side": side,
                "slice_index": item["slice_index"],
                "source_start": start,
                "source_end": end,
                "runtime_shape": list(item["runtime_shape"]),
                "value_scale": 1.0 if side == "A" else b_scale,
            })
    projections.sort(key=lambda item: item["runtime_key"])
    compact = {
        "schema": PROJECTION_SCHEMA_V72,
        "source_tensor_count": len(source_tensors),
        "source_elements": sum(item["elements"] for item in source_tensors),
        "runtime_view_count": len(projections),
        "runtime_elements": sum(math.prod(item["runtime_shape"]) for item in projections),
        "runtime_dtype": "torch.bfloat16",
        "runtime_bytes": sum(math.prod(item["runtime_shape"]) * 2 for item in projections),
        "b_scale": b_scale,
        "source_tensors": source_tensors,
        "projections": projections,
        "packed_a_full_duplication": True,
        "packed_b_disjoint_split": True,
    }
    return {
        **compact,
        "content_sha256_before_self_field": canonical_sha256_v72(compact),
    }


def validate_runtime_projection_manifest_v72(
    manifest: Mapping[str, Any],
    *,
    require_production_shape: bool = False,
) -> dict[str, Any]:
    fields = {
        "schema", "source_tensor_count", "source_elements", "runtime_view_count",
        "runtime_elements", "runtime_dtype", "runtime_bytes", "b_scale",
        "source_tensors", "projections", "packed_a_full_duplication",
        "packed_b_disjoint_split", "content_sha256_before_self_field",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != fields:
        raise ValueError("v72 projection manifest fields changed")
    compact = dict(manifest)
    claimed = compact.pop("content_sha256_before_self_field")
    if (
        manifest["schema"] != PROJECTION_SCHEMA_V72
        or claimed != canonical_sha256_v72(compact)
        or manifest["runtime_dtype"] != "torch.bfloat16"
        or manifest["packed_a_full_duplication"] is not True
        or manifest["packed_b_disjoint_split"] is not True
    ):
        raise RuntimeError("v72 projection manifest identity changed")
    b_scale = _finite_float_v72(manifest["b_scale"], "B scale", positive=True)
    sources = manifest["source_tensors"]
    projections = manifest["projections"]
    if not isinstance(sources, list) or not sources or not isinstance(projections, list):
        raise ValueError("v72 projection inventory is absent")
    source_map = {}
    for item in sources:
        if not isinstance(item, Mapping) or set(item) != {"key", "side", "shape", "elements"}:
            raise ValueError("v72 source inventory fields changed")
        key = item["key"]
        shape = _shape_v72(item["shape"], "source shape")
        if (
            not isinstance(key, str) or not key or key in source_map
            or item["side"] not in {"A", "B"}
            or item["elements"] != math.prod(shape)
        ):
            raise RuntimeError("v72 source inventory changed")
        source_map[key] = {**item, "shape": list(shape)}
    runtime_keys = set()
    by_source: dict[str, list[dict]] = defaultdict(list)
    runtime_elements = 0
    projection_fields = {
        "runtime_key", "runtime_module", "source_key", "source_shape", "side",
        "slice_index", "source_start", "source_end", "runtime_shape", "value_scale",
    }
    for item in projections:
        if not isinstance(item, Mapping) or set(item) != projection_fields:
            raise ValueError("v72 projection fields changed")
        runtime_key = item["runtime_key"]
        source_key = item["source_key"]
        if (
            not isinstance(runtime_key, str) or not runtime_key
            or runtime_key in runtime_keys or source_key not in source_map
            or not isinstance(item["runtime_module"], str)
            or not item["runtime_module"]
        ):
            raise RuntimeError("v72 projection key changed")
        runtime_keys.add(runtime_key)
        source = source_map[source_key]
        source_shape = _shape_v72(item["source_shape"], "projection source shape")
        runtime_shape = _shape_v72(item["runtime_shape"], "runtime shape")
        start = _exact_int_v72(item["source_start"], "source start")
        end = _exact_int_v72(item["source_end"], "source end", 1)
        slice_index = _exact_int_v72(item["slice_index"], "slice index")
        scale = _finite_float_v72(item["value_scale"], "projection scale", positive=True)
        expected_runtime_key = (
            f"{item['runtime_module']}|{item['side']}|slice={slice_index}|slot=0"
        )
        if (
            list(source_shape) != source["shape"]
            or item["side"] != source["side"]
            or runtime_key != expected_runtime_key
            or end <= start or end > source["elements"]
            or end - start != math.prod(runtime_shape)
            or scale != (1.0 if source["side"] == "A" else b_scale)
            or (
                source["side"] == "B"
                and (start % source_shape[1] != 0 or end % source_shape[1] != 0)
            )
        ):
            raise RuntimeError("v72 projection extent/scale changed")
        runtime_elements += math.prod(runtime_shape)
        by_source[source_key].append(dict(item))
    for key, source in source_map.items():
        group = by_source.get(key, [])
        if not group:
            raise RuntimeError("v72 source has no runtime projection")
        ranges = sorted((item["source_start"], item["source_end"]) for item in group)
        if source["side"] == "A":
            if any(item != (0, source["elements"]) for item in ranges):
                raise RuntimeError("v72 A projection is not full duplication")
        else:
            cursor = 0
            for start, end in ranges:
                if start != cursor:
                    raise RuntimeError("v72 B projection has a gap or overlap")
                cursor = end
            if cursor != source["elements"]:
                raise RuntimeError("v72 B projection coverage is incomplete")
    expected_counts = {
        "source_tensor_count": len(source_map),
        "source_elements": sum(item["elements"] for item in source_map.values()),
        "runtime_view_count": len(runtime_keys),
        "runtime_elements": runtime_elements,
        "runtime_bytes": runtime_elements * 2,
    }
    if any(manifest[key] != value for key, value in expected_counts.items()):
        raise RuntimeError("v72 projection aggregate changed")
    if require_production_shape and expected_counts != {
        "source_tensor_count": EXPECTED_SOURCE_TENSORS_V72,
        "source_elements": EXPECTED_SOURCE_ELEMENTS_V72,
        "runtime_view_count": EXPECTED_RUNTIME_VIEWS_V72,
        "runtime_elements": EXPECTED_RUNTIME_ELEMENTS_V72,
        "runtime_bytes": EXPECTED_RUNTIME_BF16_BYTES_V72,
    }:
        raise RuntimeError("v72 production Qwen LoRA projection shape changed")
    return dict(manifest)


def allocate_runtime_views_v72(manifest: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    plan = validate_runtime_projection_manifest_v72(manifest)
    return {
        item["runtime_key"]: torch.empty(item["runtime_shape"], dtype=torch.bfloat16)
        for item in plan["projections"]
    }


def _validate_master_v72(
    master: Mapping[str, torch.Tensor], manifest: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    plan = validate_runtime_projection_manifest_v72(manifest)
    shapes = {item["key"]: tuple(item["shape"]) for item in plan["source_tensors"]}
    if not isinstance(master, Mapping) or set(master) != set(shapes):
        raise RuntimeError("v72 FP32 master key coverage changed")
    result = dict(master)
    for key, shape in shapes.items():
        tensor = result[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu" or tensor.dtype != torch.float32
            or tuple(tensor.shape) != shape or not tensor.is_contiguous()
            or not bool(torch.isfinite(tensor).all())
        ):
            raise RuntimeError(f"v72 FP32 master tensor changed: {key}")
    return result


def _validate_runtime_views_v72(
    views: Mapping[str, torch.Tensor], manifest: Mapping[str, Any]
) -> dict[str, torch.Tensor]:
    plan = validate_runtime_projection_manifest_v72(manifest)
    shapes = {
        item["runtime_key"]: tuple(item["runtime_shape"])
        for item in plan["projections"]
    }
    if not isinstance(views, Mapping) or set(views) != set(shapes):
        raise RuntimeError("v72 runtime view key coverage changed")
    result = dict(views)
    pointers = set()
    for key, shape in shapes.items():
        tensor = result[key]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu" or tensor.dtype != torch.bfloat16
            or tuple(tensor.shape) != shape or not tensor.is_contiguous()
        ):
            raise RuntimeError(f"v72 runtime view tensor changed: {key}")
        pointer = (tensor.untyped_storage().data_ptr(), tensor.storage_offset())
        if pointer in pointers:
            raise RuntimeError("v72 runtime views alias each other")
        pointers.add(pointer)
    return result


def _source_layout_v72(manifest: Mapping[str, Any]) -> tuple[list[dict], dict[str, int]]:
    plan = validate_runtime_projection_manifest_v72(manifest)
    rows = []
    offsets = {}
    cursor = 0
    for item in sorted(plan["source_tensors"], key=lambda value: value["key"]):
        offsets[item["key"]] = cursor
        rows.append({**item, "global_start": cursor, "global_end": cursor + item["elements"]})
        cursor += item["elements"]
    if cursor != plan["source_elements"]:
        raise RuntimeError("v72 global source layout changed")
    return rows, offsets


def _iter_source_chunks_v72(
    source_layout: Sequence[Mapping[str, Any]],
    global_start: int,
    global_end: int,
    chunk_elements: int,
):
    chunk_elements = _exact_int_v72(chunk_elements, "chunk elements", 1)
    for item in source_layout:
        start = max(global_start, item["global_start"])
        end = min(global_end, item["global_end"])
        if end <= start:
            continue
        local_start = start - item["global_start"]
        local_end = end - item["global_start"]
        for chunk_start, chunk_end in oracle.chunk_ranges_v1(
            local_start, local_end, chunk_elements
        ):
            yield {
                "source_key": item["key"],
                "local_start": chunk_start,
                "local_end": chunk_end,
                "global_start": item["global_start"] + chunk_start,
                "global_end": item["global_start"] + chunk_end,
            }


def _exact_coverage_v72(ranges: Sequence[tuple[int, int]], total: int) -> bool:
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor or end <= start:
            return False
        cursor = end
    return cursor == total


def _factor_cache_bytes_v72(shape: Sequence[int], method: str, rank: int | None) -> int:
    if method == "iid_absolute_index":
        return 0
    rows, columns = _shape_v72(shape, "structured shape")
    rank = _exact_int_v72(rank, "structured rank", 1)
    return (rows + columns) * rank * 4


@dataclass
class _ExpectedRuntimeChunkV72:
    start: int
    end: int
    sha256: str


class FusedStructuredRuntimeV72:
    """Direct-write candidate transaction with exact restore-or-poison."""

    def __init__(
        self,
        master: Mapping[str, torch.Tensor],
        runtime_views: Mapping[str, torch.Tensor],
        manifest: Mapping[str, Any],
    ):
        self.manifest = validate_runtime_projection_manifest_v72(manifest)
        self.master = _validate_master_v72(master, self.manifest)
        self.runtime_views = _validate_runtime_views_v72(runtime_views, self.manifest)
        self._source_layout, self._source_offsets = _source_layout_v72(self.manifest)
        self._projections_by_source: dict[str, list[dict]] = defaultdict(list)
        for item in self.manifest["projections"]:
            self._projections_by_source[item["source_key"]].append(dict(item))
        self._runtime_links = {
            key: {
                "object_id": id(tensor),
                "storage_data_ptr": int(tensor.untyped_storage().data_ptr()),
                "storage_offset": int(tensor.storage_offset()),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
            for key, tensor in self.runtime_views.items()
        }
        self._master_cache = audit_v71.OwnedMasterIdentityCacheV71(self.master)
        self.phase = "initializing_canonical_runtime"
        self.poisoned = False
        self.poison_reason = None
        self._expected: dict[str, list[_ExpectedRuntimeChunkV72]] = defaultdict(list)
        self._runtime_written_ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._active = None
        self._world_size = None
        self._candidate_receipts = []
        self._ledger = {
            "candidate_runtime_write_bytes": 0,
            "restore_runtime_write_bytes": 0,
            "post_generation_exact_audit_d2h_bytes": 0,
            "restore_exact_audit_d2h_bytes": 0,
            "maximum_candidate_scratch_bytes": 0,
            "whole_surface_noise_elements_allocated": 0,
            "whole_surface_candidate_elements_allocated": 0,
        }
        self._materialize_canonical_initial_v72()

    def _require_healthy_v72(self):
        if self.poisoned:
            raise RuntimeError("v72 fused structured runtime is terminally poisoned")

    def _poison_v72(self, reason: str):
        self.poisoned = True
        self.poison_reason = str(reason)
        self.phase = "terminal_poison"
        if hasattr(self, "_transaction"):
            self._transaction.poison(str(reason))

    def _cheap_runtime_links_v72(self, phase: str) -> dict[str, Any]:
        self._require_healthy_v72()
        for key, tensor in self.runtime_views.items():
            observed = {
                "object_id": id(tensor),
                "storage_data_ptr": int(tensor.untyped_storage().data_ptr()),
                "storage_offset": int(tensor.storage_offset()),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
            }
            if observed != self._runtime_links[key]:
                self._poison_v72(f"runtime_link_drift:{key}")
                raise RuntimeError(f"v72 runtime object/storage link drifted: {key}")
        return {
            "schema": "eggroll-es-cheap-runtime-link-invariant-v72",
            "phase": str(phase),
            "passed": True,
            "runtime_view_count": len(self.runtime_views),
            "bytes_not_read_back": self.manifest["runtime_bytes"],
            "d2h_bytes": 0,
        }

    def _reset_expected_v72(self):
        self._expected = defaultdict(list)
        self._runtime_written_ranges = defaultdict(list)

    def _write_projected_chunk_v72(
        self,
        source_key: str,
        start: int,
        end: int,
        values: torch.Tensor,
        *,
        ledger_key: str,
    ) -> int:
        if (
            values.device.type != "cpu" or values.dtype != torch.float32
            or values.ndim != 1 or values.numel() != end - start
            or not bool(torch.isfinite(values).all())
        ):
            raise RuntimeError("v72 projected candidate chunk changed")
        written = 0
        for projection in self._projections_by_source[source_key]:
            overlap_start = max(start, projection["source_start"])
            overlap_end = min(end, projection["source_end"])
            if overlap_end <= overlap_start:
                continue
            source_offset = overlap_start - start
            runtime_start = overlap_start - projection["source_start"]
            runtime_end = runtime_start + (overlap_end - overlap_start)
            runtime_key = projection["runtime_key"]
            existing = self._runtime_written_ranges[runtime_key]
            if any(not (runtime_end <= left or runtime_start >= right)
                   for left, right in existing):
                self._poison_v72(f"runtime_overlap:{runtime_key}")
                raise RuntimeError("v72 runtime projection write overlaps")
            fp32 = values[source_offset:source_offset + (overlap_end - overlap_start)]
            if projection["value_scale"] != 1.0:
                fp32 = fp32.mul(projection["value_scale"])
            projected = fp32.to(dtype=torch.bfloat16).contiguous()
            self.runtime_views[runtime_key].view(-1)[runtime_start:runtime_end].copy_(projected)
            raw = _tensor_raw_v72(projected)
            self._expected[runtime_key].append(_ExpectedRuntimeChunkV72(
                runtime_start, runtime_end, hashlib.sha256(raw).hexdigest()
            ))
            existing.append((runtime_start, runtime_end))
            written += len(raw)
        self._ledger[ledger_key] += written
        return written

    def _validate_runtime_coverage_v72(self):
        for key, view in self.runtime_views.items():
            ranges = self._runtime_written_ranges.get(key, [])
            if not _exact_coverage_v72(ranges, view.numel()):
                self._poison_v72(f"runtime_coverage_incomplete:{key}")
                raise RuntimeError(f"v72 runtime projection coverage incomplete: {key}")

    def _audit_expected_runtime_v72(self, boundary: str) -> dict[str, Any]:
        self._cheap_runtime_links_v72(boundary)
        hashes = {}
        for key in sorted(self.runtime_views):
            view = self.runtime_views[key].view(-1)
            chunks = sorted(self._expected[key], key=lambda item: item.start)
            if not _exact_coverage_v72([(item.start, item.end) for item in chunks], view.numel()):
                self._poison_v72(f"expected_coverage_incomplete:{key}")
                raise RuntimeError("v72 expected runtime coverage changed")
            digest = hashlib.sha256()
            for item in chunks:
                raw = _tensor_raw_v72(view[item.start:item.end].contiguous())
                if hashlib.sha256(raw).hexdigest() != item.sha256:
                    self._poison_v72(f"runtime_content_mismatch:{key}")
                    raise RuntimeError(f"v72 runtime content differs from stream: {key}")
                digest.update(raw)
            hashes[key] = digest.hexdigest()
        identity = audit_v71.canonical_sha256_v71([
            {
                "key": key,
                "shape": list(self.runtime_views[key].shape),
                "dtype": str(self.runtime_views[key].dtype),
                "elements": int(self.runtime_views[key].numel()),
                "sha256": hashes[key],
            }
            for key in sorted(hashes)
        ])
        return {
            "schema": "eggroll-es-single-exact-runtime-readback-v72",
            "boundary": str(boundary),
            "runtime_view_count": len(hashes),
            "runtime_elements": self.manifest["runtime_elements"],
            "d2h_bytes": self.manifest["runtime_bytes"],
            "sha256_by_key": hashes,
            "runtime_identity_sha256": identity,
            "single_d2h_readback": True,
            "matches_streamed_expected_chunks": True,
        }

    def _write_master_range_v72(self, global_start: int, global_end: int, chunk_elements: int):
        for item in _iter_source_chunks_v72(
            self._source_layout, global_start, global_end, chunk_elements
        ):
            key = item["source_key"]
            values = self.master[key].view(-1)[item["local_start"]:item["local_end"]]
            self._write_projected_chunk_v72(
                key, item["local_start"], item["local_end"], values,
                ledger_key="restore_runtime_write_bytes",
            )
            yield item

    def _materialize_canonical_initial_v72(self):
        self._reset_expected_v72()
        for _item in self._write_master_range_v72(
            0, self.manifest["source_elements"], DEFAULT_CHUNK_ELEMENTS_V72
        ):
            pass
        self._validate_runtime_coverage_v72()
        audit = self._audit_expected_runtime_v72("initial_canonical_bind")
        self._canonical_runtime_sha256 = audit["runtime_identity_sha256"]
        self._canonical_runtime_hashes = audit["sha256_by_key"]
        self._runtime_registry = audit_v71.TensorInvariantRegistryV71(
            "runtime_lora_views", self.runtime_views,
            precomputed_sha256=self._canonical_runtime_hashes,
        )
        self._transaction = oracle.RuntimeTransactionOracleV1(
            self.manifest["source_elements"],
            self._master_cache.sha256,
            self._canonical_runtime_sha256,
        )
        # Initialization is not a candidate/restore operation in the byte ledger.
        self._ledger["restore_runtime_write_bytes"] = 0
        self.phase = "quiescent"
        self._reset_expected_v72()

    def begin_candidate_v72(
        self,
        *,
        method: str,
        seed: int,
        sigma: float,
        sign: int,
        structured_rank: int | None = None,
    ) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "quiescent":
            raise RuntimeError("v72 candidate began outside quiescent state")
        seed = _exact_int_v72(seed, "direction seed")
        sigma = _finite_float_v72(sigma, "sigma", positive=True)
        if isinstance(sign, bool) or sign not in (-1, 1):
            raise ValueError("v72 candidate sign must be exactly +/-1")
        identities = [
            oracle.noise_identity_v1(
                method, seed, item["key"], item["shape"], structured_rank
            )
            for item in self.manifest["source_tensors"]
        ]
        program = {
            "schema": "structured-es-runtime-noise-program-v72",
            "projection_manifest_sha256": self.manifest[
                "content_sha256_before_self_field"
            ],
            "method": method,
            "seed": seed,
            "sigma": sigma,
            "sign": sign,
            "structured_rank": structured_rank,
            "tensor_noise_identity_sha256": [
                item["identity_sha256"] for item in identities
            ],
            "rng_algorithm": oracle.RNG_ALGORITHM_V1,
            "absolute_indexed": True,
        }
        program["program_sha256"] = canonical_sha256_v72(program)
        try:
            self._master_cache.cached_identity(self.master, "candidate_begin")
            self._runtime_registry.cheap_certificate(
                self.runtime_views, "candidate_begin"
            )
        except BaseException:
            self._poison_v72("candidate_begin_invariant_failure")
            raise
        self._transaction.begin_candidate(program["program_sha256"])
        self._reset_expected_v72()
        self._active = program
        self._world_size = None
        self.phase = "materializing_candidate"
        return program

    def write_candidate_shard_v72(
        self,
        *,
        world_size: int,
        rank: int,
        chunk_elements: int = DEFAULT_CHUNK_ELEMENTS_V72,
    ) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "materializing_candidate" or not isinstance(self._active, dict):
            raise RuntimeError("v72 has no materializing candidate")
        world_size = _exact_int_v72(world_size, "world size", 1)
        rank = _exact_int_v72(rank, "shard rank")
        chunk_elements = _exact_int_v72(chunk_elements, "chunk elements", 1)
        if rank >= world_size:
            raise ValueError("v72 shard rank is outside world size")
        if self._world_size is None:
            self._world_size = world_size
        elif self._world_size != world_size:
            self._poison_v72("world_size_changed_mid_candidate")
            raise RuntimeError("v72 candidate world size changed")
        shard_start, shard_end = oracle.shard_range_v1(
            self.manifest["source_elements"], world_size, rank
        )
        chunks = 0
        runtime_bytes = 0
        maximum_scratch = 0
        for item in _iter_source_chunks_v72(
            self._source_layout, shard_start, shard_end, chunk_elements
        ):
            try:
                self._transaction.record_candidate_chunk(
                    item["global_start"], item["global_end"]
                )
            except BaseException:
                self._poison_v72("candidate_source_overlap_or_invalid_range")
                raise
            key = item["source_key"]
            count = item["local_end"] - item["local_start"]
            noise = oracle.noise_chunk_v1(
                self._active["method"], self._active["seed"], key,
                list(self.master[key].shape), item["local_start"], count,
                self._active["structured_rank"],
            )
            candidate = self.master[key].view(-1)[
                item["local_start"]:item["local_end"]
            ].add(noise, alpha=self._active["sign"] * self._active["sigma"])
            runtime_bytes += self._write_projected_chunk_v72(
                key, item["local_start"], item["local_end"], candidate,
                ledger_key="candidate_runtime_write_bytes",
            )
            factor = _factor_cache_bytes_v72(
                self.master[key].shape,
                self._active["method"],
                self._active["structured_rank"],
            )
            # Noise + candidate + optional scaled FP32 projection + BF16 staging.
            maximum_scratch = max(maximum_scratch, factor + count * 14)
            chunks += 1
        self._ledger["maximum_candidate_scratch_bytes"] = max(
            self._ledger["maximum_candidate_scratch_bytes"], maximum_scratch
        )
        return {
            "schema": "fused-structured-candidate-shard-v72",
            "rank": rank,
            "world_size": world_size,
            "global_start": shard_start,
            "global_end": shard_end,
            "chunks": chunks,
            "runtime_bytes_written": runtime_bytes,
            "maximum_scratch_bytes": maximum_scratch,
            "dense_noise_materialized": False,
            "dense_candidate_materialized": False,
        }

    def complete_candidate_write_v72(self) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "materializing_candidate":
            raise RuntimeError("v72 candidate write completion is out of order")
        try:
            self._transaction.finish_candidate()
        except BaseException:
            self._poison_v72("candidate_source_coverage_incomplete")
            raise
        self._validate_runtime_coverage_v72()
        self._cheap_runtime_links_v72("candidate_pre_generation")
        # V71 cheap checks include tensor versions.  Rebind the deliberate
        # write with a fail-closed content sentinel: an unexpected exact audit
        # before post-generation will reject, while the cheap pre-generation
        # edge can verify storage/version without an early D2H readback.
        provisional_hashes = {
            key: hashlib.sha256(
                (
                    "v72-provisional-content-sentinel\0"
                    f"{self._active['program_sha256']}\0{key}"
                ).encode("utf-8")
            ).hexdigest()
            for key in self.runtime_views
        }
        rebind = self._runtime_registry.rebind_controlled_write(
            self.runtime_views,
            "candidate_pre_generation_provisional",
            expected_sha256=provisional_hashes,
        )
        cheap = self._runtime_registry.cheap_certificate(
            self.runtime_views, "candidate_pre_generation"
        )
        self.phase = "candidate_materialized_provisional"
        return {
            "schema": "fused-structured-candidate-provisional-v72",
            "program_sha256": self._active["program_sha256"],
            "runtime_bytes_written": self.manifest["runtime_bytes"],
            "cheap_runtime_invariant": cheap,
            "provisional_version_rebind": rebind,
            "content_identity_is_fail_closed_sentinel_until_post_generation": True,
            "reward_provisional": True,
            "exact_post_generation_audit_required": True,
            "dense_noise_materialized": False,
            "dense_candidate_materialized": False,
        }

    def post_generation_exact_audit_v72(self) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "candidate_materialized_provisional":
            raise RuntimeError("v72 post-generation audit is out of order")
        try:
            self._runtime_registry.cheap_certificate(
                self.runtime_views, "candidate_post_generation_before_exact"
            )
        except BaseException:
            self._poison_v72("post_generation_version_invariant_failure")
            raise
        exact = self._audit_expected_runtime_v72("candidate_post_generation")
        self._runtime_registry.rebind_controlled_write(
            self.runtime_views,
            "candidate_post_generation",
            expected_sha256=exact["sha256_by_key"],
        )
        cheap = self._runtime_registry.cheap_certificate(
            self.runtime_views, "candidate_post_generation"
        )
        self._ledger["post_generation_exact_audit_d2h_bytes"] += exact["d2h_bytes"]
        compact = {
            "schema": CANDIDATE_SCHEMA_V72,
            "program": self._active,
            "projection_manifest_sha256": self.manifest[
                "content_sha256_before_self_field"
            ],
            "runtime_exact_audit": exact,
            "cheap_transition_invariant": cheap,
            "v71_boundary": "candidate_post_generation",
            "reward_accepted": False,
            "exact_restore_required": True,
            "terminal_poisoned": False,
        }
        receipt = {**compact, "content_sha256": canonical_sha256_v72(compact)}
        self._candidate_receipts.append(receipt["content_sha256"])
        self.phase = "candidate_active_audited"
        return receipt

    def begin_restore_v72(self) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase not in {
            "materializing_candidate", "candidate_materialized_provisional",
            "candidate_active_audited",
        }:
            raise RuntimeError("v72 restore began without uncertain/active candidate")
        self._transaction.begin_restore()
        self._reset_expected_v72()
        self._world_size = None
        self.phase = "restoring_canonical"
        return {
            "schema": "fused-structured-restore-begun-v72",
            "unknown_or_partial_candidate_allowed": True,
            "exact_restore_or_poison": True,
        }

    def write_restore_shard_v72(
        self,
        *,
        world_size: int,
        rank: int,
        chunk_elements: int = DEFAULT_CHUNK_ELEMENTS_V72,
    ) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "restoring_canonical":
            raise RuntimeError("v72 has no active canonical restore")
        world_size = _exact_int_v72(world_size, "world size", 1)
        rank = _exact_int_v72(rank, "shard rank")
        if rank >= world_size:
            raise ValueError("v72 restore shard rank is outside world size")
        if self._world_size is None:
            self._world_size = world_size
        elif self._world_size != world_size:
            self._poison_v72("world_size_changed_mid_restore")
            raise RuntimeError("v72 restore world size changed")
        start, end = oracle.shard_range_v1(
            self.manifest["source_elements"], world_size, rank
        )
        runtime_bytes = 0
        chunks = 0
        before = self._ledger["restore_runtime_write_bytes"]
        for item in self._write_master_range_v72(start, end, chunk_elements):
            try:
                self._transaction.record_restore_chunk(
                    item["global_start"], item["global_end"]
                )
            except BaseException:
                self._poison_v72("restore_source_overlap_or_invalid_range")
                raise
            chunks += 1
        runtime_bytes = self._ledger["restore_runtime_write_bytes"] - before
        return {
            "schema": "fused-structured-restore-shard-v72",
            "rank": rank,
            "world_size": world_size,
            "global_start": start,
            "global_end": end,
            "chunks": chunks,
            "runtime_bytes_written": runtime_bytes,
        }

    def finish_restore_v72(self) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "restoring_canonical":
            raise RuntimeError("v72 restore completion is out of order")
        try:
            self._validate_runtime_coverage_v72()
            exact = self._audit_expected_runtime_v72("restore")
            master = self._master_cache.exact_audit(self.master, "restore")
            if (
                master["sha256"] != self._master_cache.sha256
                or exact["runtime_identity_sha256"] != self._canonical_runtime_sha256
                or exact["sha256_by_key"] != self._canonical_runtime_hashes
            ):
                raise RuntimeError("v72 exact canonical identity changed")
            self._transaction.finish_restore(
                master["sha256"], exact["runtime_identity_sha256"]
            )
            self._runtime_registry.rebind_controlled_write(
                self.runtime_views, "restore",
                expected_sha256=exact["sha256_by_key"],
            )
        except BaseException as error:
            if not self.poisoned:
                self._poison_v72("restore_coverage_or_identity_failure")
            raise RuntimeError("v72 exact restore failed; runtime poisoned") from error
        self._ledger["restore_exact_audit_d2h_bytes"] += exact["d2h_bytes"]
        self.phase = "quiescent"
        self._active = None
        self._world_size = None
        self._reset_expected_v72()
        return {
            "schema": "fused-structured-exact-master-restore-v72",
            "restored": True,
            "master_identity_sha256": master["sha256"],
            "runtime_identity_sha256": exact["runtime_identity_sha256"],
            "single_exact_runtime_readback": exact,
            "v71_restore_boundary_compatible": True,
            "terminal_poisoned": False,
        }

    def repair_after_uncertain_candidate_v72(
        self,
        *,
        world_size: int = oracle.WORLD_SIZE_V1,
        chunk_elements: int = DEFAULT_CHUNK_ELEMENTS_V72,
        inject_failure_after_shards: int | None = None,
    ) -> dict[str, Any]:
        """Restore an unknown/partial write; test injection exercises poison."""
        self.begin_restore_v72()
        try:
            for ordinal, rank in enumerate(range(world_size)):
                if inject_failure_after_shards is not None and ordinal >= int(
                    inject_failure_after_shards
                ):
                    raise RuntimeError("injected restore RPC failure")
                self.write_restore_shard_v72(
                    world_size=world_size, rank=rank, chunk_elements=chunk_elements
                )
            return self.finish_restore_v72()
        except BaseException as error:
            if not self.poisoned:
                self._poison_v72("uncertain_candidate_restore_failed")
            raise RuntimeError(
                "v72 uncertain candidate could not be restored; runtime poisoned"
            ) from error

    def transaction_receipt_v72(self) -> dict[str, Any]:
        return {
            "schema": "fused-structured-runtime-transaction-receipt-v72",
            "phase": self.phase,
            "poisoned": self.poisoned,
            "poison_reason": self.poison_reason,
            "oracle_transaction": self._transaction.receipt(),
            "projection_manifest_sha256": self.manifest[
                "content_sha256_before_self_field"
            ],
            "candidate_exact_audit_sha256": list(self._candidate_receipts),
            "byte_ledger": dict(self._ledger),
            "whole_surface_noise_or_candidate_allocated": False,
        }


def _record_update_range_v72(
    ranges: list[tuple[int, int]], start: int, end: int, total: int
):
    if end <= start or start < 0 or end > total:
        raise RuntimeError("v72 update chunk range is invalid")
    if any(not (end <= left or start >= right) for left, right in ranges):
        raise RuntimeError("v72 update chunk range overlaps")
    ranges.append((start, end))


def _weighted_update_chunk_v72(
    *,
    shape: Sequence[int],
    method: str,
    seed_coefficients: Sequence[tuple[int, float]],
    tensor_key: str,
    sigma: float,
    start: int,
    count: int,
    structured_rank: int | None,
) -> torch.Tensor:
    accumulator = torch.zeros(count, dtype=torch.float32)
    population = len(seed_coefficients)
    for seed, coefficient in seed_coefficients:
        noise = oracle.noise_chunk_v1(
            method, seed, tensor_key, shape, start, count, structured_rank
        )
        accumulator.add_(noise, alpha=coefficient)
    accumulator.mul_(1.0 / (2.0 * population * sigma))
    if not bool(torch.isfinite(accumulator).all()):
        raise FloatingPointError("v72 streamed update became nonfinite")
    return accumulator


class StreamedMasterUpdateV72:
    """One-shot V71-compatible pending-master transaction."""

    def __init__(
        self,
        master: Mapping[str, torch.Tensor],
        manifest: Mapping[str, Any],
        *,
        method: str,
        seeds: Sequence[int],
        coefficients: Sequence[float],
        sigma: float,
        step_size: float,
        v71_update_acceptance_sha256: str,
        structured_rank: int | None = None,
    ):
        self.manifest = validate_runtime_projection_manifest_v72(manifest)
        self.original_master = _validate_master_v72(master, self.manifest)
        self.current_master = self.original_master
        self._original_cache = audit_v71.OwnedMasterIdentityCacheV71(
            self.original_master
        )
        if len(seeds) != len(coefficients) or not seeds:
            raise ValueError("v72 update seeds and coefficients must match")
        pairs = [
            (
                _exact_int_v72(seed, "direction seed"),
                _finite_float_v72(coefficient, "pair coefficient"),
            )
            for seed, coefficient in zip(seeds, coefficients, strict=True)
        ]
        pairs.sort(key=lambda item: item[0])
        if len({seed for seed, _coefficient in pairs}) != len(pairs):
            raise ValueError("v72 update direction seeds must be unique")
        self.method = method
        self.seed_coefficients = pairs
        self.sigma = _finite_float_v72(sigma, "sigma", positive=True)
        self.step_size = _finite_float_v72(step_size, "step size", positive=True)
        self.structured_rank = structured_rank
        # Validate every tensor/method combination before allocating output.
        for item in self.manifest["source_tensors"]:
            oracle.noise_identity_v1(
                method, pairs[0][0], item["key"], item["shape"], structured_rank
            )
        self.update_acceptance_sha256 = _sha256_string_v72(
            v71_update_acceptance_sha256, "V71 update acceptance"
        )
        self._source_layout, _offsets = _source_layout_v72(self.manifest)
        self.pending_master = {
            key: torch.empty_like(tensor) for key, tensor in self.original_master.items()
        }
        self.ranges: list[tuple[int, int]] = []
        self.world_size = None
        self.phase = "streaming_pending_master"
        self.poisoned = False
        self.poison_reason = None
        self.pending_identity = None
        self.rollback_master = None
        self._ledger = {
            "canonical_master_read_bytes": 0,
            "pending_master_write_bytes": 0,
            "pending_master_persistent_bytes": sum(
                tensor.numel() * tensor.element_size()
                for tensor in self.pending_master.values()
            ),
            "maximum_update_scratch_bytes": 0,
            "whole_surface_noise_elements_allocated": 0,
            "whole_surface_update_elements_allocated": 0,
        }

    def _require_healthy_v72(self):
        if self.poisoned:
            raise RuntimeError("v72 streamed master update is terminally poisoned")

    def _poison_v72(self, reason: str):
        self.poisoned = True
        self.poison_reason = str(reason)
        self.phase = "terminal_poison"

    def write_shard_v72(
        self,
        *,
        world_size: int,
        rank: int,
        chunk_elements: int = DEFAULT_CHUNK_ELEMENTS_V72,
    ) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "streaming_pending_master":
            raise RuntimeError("v72 update shard arrived in the wrong phase")
        world_size = _exact_int_v72(world_size, "world size", 1)
        rank = _exact_int_v72(rank, "shard rank")
        if rank >= world_size:
            raise ValueError("v72 update shard rank is outside world size")
        if self.world_size is None:
            self.world_size = world_size
        elif self.world_size != world_size:
            self._poison_v72("world_size_changed_mid_update")
            raise RuntimeError("v72 update world size changed")
        start, end = oracle.shard_range_v1(
            self.manifest["source_elements"], world_size, rank
        )
        chunks = 0
        maximum_scratch = 0
        bytes_written = 0
        try:
            for item in _iter_source_chunks_v72(
                self._source_layout, start, end, chunk_elements
            ):
                _record_update_range_v72(
                    self.ranges, item["global_start"], item["global_end"],
                    self.manifest["source_elements"],
                )
                key = item["source_key"]
                count = item["local_end"] - item["local_start"]
                update = _weighted_update_chunk_v72(
                    shape=list(self.original_master[key].shape),
                    method=self.method,
                    seed_coefficients=self.seed_coefficients,
                    tensor_key=key,
                    sigma=self.sigma,
                    start=item["local_start"],
                    count=count,
                    structured_rank=self.structured_rank,
                )
                candidate = self.original_master[key].view(-1)[
                    item["local_start"]:item["local_end"]
                ].add(update, alpha=self.step_size)
                self.pending_master[key].view(-1)[
                    item["local_start"]:item["local_end"]
                ].copy_(candidate)
                byte_count = count * 4
                self._ledger["canonical_master_read_bytes"] += byte_count
                self._ledger["pending_master_write_bytes"] += byte_count
                bytes_written += byte_count
                factor = _factor_cache_bytes_v72(
                    self.original_master[key].shape,
                    self.method,
                    self.structured_rank,
                )
                maximum_scratch = max(maximum_scratch, factor + count * 12)
                chunks += 1
        except BaseException:
            self.phase = "partial_update_fault_original_master_unchanged"
            raise
        self._ledger["maximum_update_scratch_bytes"] = max(
            self._ledger["maximum_update_scratch_bytes"], maximum_scratch
        )
        return {
            "schema": "streamed-structured-update-shard-v72",
            "rank": rank,
            "world_size": world_size,
            "global_start": start,
            "global_end": end,
            "chunks": chunks,
            "pending_master_bytes_written": bytes_written,
            "maximum_scratch_bytes": maximum_scratch,
            "dense_noise_or_update_materialized": False,
        }

    def finish_v72(self) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "streaming_pending_master":
            raise RuntimeError("v72 update finish is out of order")
        if not _exact_coverage_v72(self.ranges, self.manifest["source_elements"]):
            self.phase = "partial_update_fault_original_master_unchanged"
            raise RuntimeError("v72 update coverage has a gap, overlap, or duplicate")
        if any(not bool(torch.isfinite(tensor).all())
               for tensor in self.pending_master.values()):
            self.phase = "partial_update_fault_original_master_unchanged"
            raise FloatingPointError("v72 pending master is nonfinite")
        self.pending_identity = _mapping_identity_v72(self.pending_master)
        self.phase = "pending_master_complete_not_committed"
        compact = {
            "schema": UPDATE_SCHEMA_V72,
            "phase": self.phase,
            "projection_manifest_sha256": self.manifest[
                "content_sha256_before_self_field"
            ],
            "rng_algorithm": oracle.RNG_ALGORITHM_V1,
            "method": self.method,
            "structured_rank": self.structured_rank,
            "seed_coefficients": [list(item) for item in self.seed_coefficients],
            "sigma": self.sigma,
            "step_size": self.step_size,
            "v71_update_acceptance_sha256": self.update_acceptance_sha256,
            "original_master_sha256": self._original_cache.sha256,
            "pending_master_identity": self.pending_identity,
            "byte_ledger": dict(self._ledger),
            "dense_noise_materialized": False,
            "dense_update_materialized": False,
            "pending_master_is_transaction_output_not_scratch": True,
            "requires_v71_commit_and_final_exact_boundaries": True,
            "committed": False,
        }
        return {**compact, "content_sha256": canonical_sha256_v72(compact)}

    def abort_v72(self, reason: str) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase not in {
            "streaming_pending_master", "partial_update_fault_original_master_unchanged",
            "pending_master_complete_not_committed", "commit_provisional_awaiting_boundary",
            "commit_boundary_passed_awaiting_final",
        }:
            raise RuntimeError("v72 update abort is out of order")
        reason = str(reason)
        if not reason:
            raise ValueError("v72 update abort reason is empty")
        try:
            exact = self._original_cache.exact_audit(
                self.original_master, f"update_abort:{reason}"
            )
            if exact["sha256"] != self._original_cache.sha256:
                raise RuntimeError("v72 rollback master identity changed")
        except BaseException as error:
            self._poison_v72("update_rollback_identity_failure")
            raise RuntimeError("v72 update rollback failed; transaction poisoned") from error
        self.current_master = self.original_master
        self.pending_master = None
        self.rollback_master = None
        self.phase = "aborted_exact_original_restored"
        return {
            "schema": "streamed-structured-update-abort-v72",
            "reason": reason,
            "original_master_sha256": exact["sha256"],
            "original_master_exact_audit": exact,
            "pending_output_discarded": True,
            "terminal_poisoned": False,
        }

    def commit_provisional_v72(self, update_acceptance_sha256: str) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "pending_master_complete_not_committed":
            raise RuntimeError("v72 provisional commit is out of order")
        if _sha256_string_v72(update_acceptance_sha256, "update acceptance") != (
            self.update_acceptance_sha256
        ):
            raise RuntimeError("v72 update acceptance identity changed")
        self.rollback_master = self.original_master
        self.current_master = self.pending_master
        self.phase = "commit_provisional_awaiting_boundary"
        return {
            "schema": "streamed-structured-update-provisional-commit-v72",
            "pending_master_identity": self.pending_identity,
            "rollback_retained": True,
            "accepted": False,
            "v71_commit_boundary_required": True,
        }

    def accept_commit_boundary_v72(self, audit_sha256: str) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "commit_provisional_awaiting_boundary":
            raise RuntimeError("v72 commit boundary is out of order")
        audit_sha256 = _sha256_string_v72(audit_sha256, "V71 commit audit")
        self.phase = "commit_boundary_passed_awaiting_final"
        return {
            "schema": "streamed-structured-update-commit-boundary-v72",
            "v71_commit_boundary_audit_sha256": audit_sha256,
            "commit_accepted": True,
            "rollback_retained": True,
            "v71_final_boundary_required": True,
        }

    def finalize_v72(self, audit_sha256: str) -> dict[str, Any]:
        self._require_healthy_v72()
        if self.phase != "commit_boundary_passed_awaiting_final":
            raise RuntimeError("v72 final boundary is out of order")
        audit_sha256 = _sha256_string_v72(audit_sha256, "V71 final audit")
        final_identity = _mapping_identity_v72(self.current_master)
        if final_identity != self.pending_identity:
            self._poison_v72("final_master_identity_failure")
            raise RuntimeError("v72 final master identity changed; transaction poisoned")
        self.rollback_master = None
        self.original_master = self.current_master
        self._original_cache = audit_v71.OwnedMasterIdentityCacheV71(
            self.original_master, identity=final_identity
        )
        self.phase = "finalized"
        return {
            "schema": "streamed-structured-update-finalized-v72",
            "v71_final_boundary_audit_sha256": audit_sha256,
            "final_master_identity": final_identity,
            "rollback_released": True,
            "terminal_poisoned": False,
        }


def cpu_oracle_reference_update_v72(
    master: Mapping[str, torch.Tensor],
    manifest: Mapping[str, Any],
    *,
    method: str,
    seeds: Sequence[int],
    coefficients: Sequence[float],
    sigma: float,
    step_size: float,
    structured_rank: int | None = None,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS_V72,
) -> dict[str, torch.Tensor]:
    """Independent CPU oracle used only for correctness comparison."""
    plan = validate_runtime_projection_manifest_v72(manifest)
    source = _validate_master_v72(master, plan)
    pairs = sorted(zip(seeds, coefficients, strict=True), key=lambda item: item[0])
    result = {}
    for item in plan["source_tensors"]:
        key = item["key"]
        candidate = torch.empty_like(source[key]).view(-1)
        for chunk in oracle.streamed_weighted_update_chunks_v1(
            item["shape"], method,
            [pair[0] for pair in pairs], [pair[1] for pair in pairs],
            key, sigma, structured_rank=structured_rank,
            chunk_elements=chunk_elements,
        ):
            candidate[chunk["start"]:chunk["end"]].copy_(
                source[key].view(-1)[chunk["start"]:chunk["end"]].add(
                    chunk["update"], alpha=step_size
                )
            )
        result[key] = candidate.view_as(source[key])
    return result


def _ordered_float32_bits_v72(tensor: torch.Tensor) -> np.ndarray:
    values = tensor.detach().cpu().contiguous().numpy().view(np.uint32)
    wide = values.astype(np.uint64)
    negative = (wide & np.uint64(0x80000000)) != 0
    ordered = np.where(
        negative,
        np.uint64(0xFFFFFFFF) - wide,
        wide + np.uint64(0x80000000),
    )
    ordered[np.asarray(tensor.detach().cpu().contiguous().numpy() == 0)] = np.uint64(
        0x80000000
    )
    return ordered


def validate_update_ulp_v72(
    observed: Mapping[str, torch.Tensor],
    expected: Mapping[str, torch.Tensor],
    *,
    maximum_ulp: int = MAX_FINAL_UPDATE_ULP_V72,
) -> dict[str, Any]:
    maximum_ulp = _exact_int_v72(maximum_ulp, "maximum ULP")
    if not isinstance(observed, Mapping) or set(observed) != set(expected):
        raise RuntimeError("v72 update comparison key coverage changed")
    per_tensor = {}
    global_max = 0
    global_abs = 0.0
    for key in sorted(expected):
        left = observed[key]
        right = expected[key]
        if (
            not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
            or left.dtype != torch.float32 or right.dtype != torch.float32
            or left.shape != right.shape
            or not bool(torch.isfinite(left).all())
            or not bool(torch.isfinite(right).all())
        ):
            raise RuntimeError("v72 update comparison tensor changed")
        a = _ordered_float32_bits_v72(left)
        b = _ordered_float32_bits_v72(right)
        distance = np.maximum(a, b) - np.minimum(a, b)
        max_ulp = int(distance.max(initial=0))
        max_abs = float(torch.max(torch.abs(left - right)).item())
        per_tensor[key] = {"maximum_ulp": max_ulp, "maximum_abs": max_abs}
        global_max = max(global_max, max_ulp)
        global_abs = max(global_abs, max_abs)
    if global_max > maximum_ulp:
        raise RuntimeError("v72 final update exceeds the registered ULP gate")
    return {
        "schema": "streamed-structured-update-ulp-certificate-v72",
        "passed": True,
        "maximum_allowed_ulp": maximum_ulp,
        "observed_maximum_ulp": global_max,
        "observed_maximum_abs": global_abs,
        "per_tensor": per_tensor,
    }


def production_byte_ledger_v72(
    manifest: Mapping[str, Any],
    *,
    method: str,
    structured_rank: int | None,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS_V72,
) -> dict[str, Any]:
    plan = validate_runtime_projection_manifest_v72(
        manifest, require_production_shape=True
    )
    chunk_elements = _exact_int_v72(chunk_elements, "chunk elements", 1)
    maximum_chunk = min(
        chunk_elements, max(item["elements"] for item in plan["source_tensors"])
    )
    factor = max(
        _factor_cache_bytes_v72(item["shape"], method, structured_rank)
        for item in plan["source_tensors"]
    )
    source_fp32_bytes = plan["source_elements"] * 4
    runtime_bytes = plan["runtime_bytes"]
    candidate_scratch = factor + maximum_chunk * 14
    update_scratch = factor + maximum_chunk * 12
    if method == "iid_absolute_index":
        random_values = plan["source_elements"]
    else:
        rank = _exact_int_v72(structured_rank, "structured rank", 1)
        random_values = sum(
            sum(item["shape"]) * rank for item in plan["source_tensors"]
        )
    return {
        "schema": "qwen36-fused-structured-byte-ledger-v72",
        "projection_manifest_sha256": plan["content_sha256_before_self_field"],
        "method": method,
        "structured_rank": structured_rank,
        "chunk_elements": chunk_elements,
        "source_fp32_master_bytes": source_fp32_bytes,
        "runtime_bf16_bytes": runtime_bytes,
        "unique_random_values_per_direction": random_values,
        "maximum_factor_cache_bytes": factor,
        "candidate_scratch_ceiling_bytes": candidate_scratch,
        "weighted_update_scratch_ceiling_bytes": update_scratch,
        "candidate_direct_runtime_write_bytes": runtime_bytes,
        "candidate_post_generation_exact_readback_bytes": runtime_bytes,
        "restore_direct_runtime_write_bytes": runtime_bytes,
        "restore_exact_readback_bytes": runtime_bytes,
        "pending_master_transaction_output_bytes": source_fp32_bytes,
        "weighted_update_master_read_bytes": source_fp32_bytes,
        "weighted_update_pending_master_write_bytes": source_fp32_bytes,
        "eliminated_candidate_fp32_device_to_host_bytes": source_fp32_bytes,
        "eliminated_pre_generation_runtime_equality_readback_bytes": runtime_bytes,
        "per_16_candidate_direct_runtime_write_bytes": (
            oracle.SIGNED_CANDIDATES_PER_UPDATE_V1 * runtime_bytes
        ),
        "per_16_candidate_restore_runtime_write_bytes": (
            oracle.SIGNED_CANDIDATES_PER_UPDATE_V1 * runtime_bytes
        ),
        "whole_surface_noise_elements_allocated": 0,
        "whole_surface_candidate_elements_allocated": 0,
        "whole_surface_update_elements_allocated": 0,
        "pending_master_is_persistent_transaction_output_not_scratch": True,
        "post_generation_and_restore_exact_audits_retained": True,
    }
