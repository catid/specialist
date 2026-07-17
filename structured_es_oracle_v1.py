#!/usr/bin/env python3
"""Absolute-index correctness oracle for streamed structured ES noise.

The oracle is intentionally CPU-only.  It defines the values, identities,
coverage rules, scaling, scratch accounting, and transaction semantics a
future CUDA/vLLM worker must reproduce without ever allocating a dense noise
or candidate surface.  Production kernels may vectorize the computation, but
may not change its absolute-index domains or fixed ascending-rank reduction.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch


SCHEMA_V1 = "eggroll-structured-perturbation-oracle-v1"
MASK64_V1 = (1 << 64) - 1
RNG_ALGORITHM_V1 = "sha256-domain-splitmix64-box-muller-fp32-v1"
STRUCTURED_RANKS_V1 = (1, 4, 8, 16)
CHUNK_ELEMENTS_V1 = 16_384
WORLD_SIZE_V1 = 4
LORA_TENSORS_V1 = 70
LORA_MODULES_V1 = 35
LORA_ELEMENTS_V1 = 4_528_128
LORA_FP32_BYTES_V1 = 18_112_512
LORA_RUNTIME_BF16_BYTES_V1 = 9_842_688
LORA_MAX_TENSOR_ELEMENTS_V1 = 262_144
FULL_MODEL_TENSORS_V1 = 1_045
FULL_MODEL_ELEMENTS_V1 = 35_951_822_704
FULL_MODEL_BF16_BYTES_V1 = 71_903_645_408
FULL_MODEL_FP32_MASTER_BYTES_V1 = 143_807_290_816
FULL_MODEL_MAX_TENSOR_ELEMENTS_V1 = 536_870_912
DIRECTIONS_PER_UPDATE_V1 = 8
SIGNED_CANDIDATES_PER_UPDATE_V1 = 16
TRAIN_UNITS_PER_CANDIDATE_V1 = 64
ROLLOUTS_PER_UPDATE_V1 = 1_024
SIGMA_SCHEDULE_V1 = (0.0006, 0.0003)
UPDATE_BUDGET_RATIO_V1 = 0.0005
UPDATE_NORM_RELATIVE_TOLERANCE_V1 = 5.0e-5
REPLICATE_SEEDS_V1 = (1701, 1702, 1703)

OPTIMIZER_CONTRACT_FILE_SHA256_V1 = (
    "428d1de245a5cd5ad3cb976aa5312f6eda0874efb895d298bb5731a05f326924"
)
OPTIMIZER_CONTRACT_CONTENT_SHA256_V1 = (
    "e8c646b5929de49805421035bb56f2eca2ed2010f7d1fce6893f5b095303dbc9"
)
V66D_PREREG_FILE_SHA256_V1 = (
    "3269f7138d74266538cc3b0f31e1a904808f8f3751dde5a7a9456e93b13314b0"
)
V66D_PREREG_CONTENT_SHA256_V1 = (
    "2f8e23b643507b594b05719966da1b9bcc64a2b7f412021066df4c6418144531"
)
V66D_REPORT_FILE_SHA256_V1 = (
    "12a5e854856d28bd8439cf3ed004664317086f8d117ae08e78b59f857f6102bb"
)
V66D_REPORT_CONTENT_SHA256_V1 = (
    "87d1eca139ee0b766b15517c81459becd0369c9d5f7ffb78269fdfce977de684"
)
V66D_TELEMETRY_FILE_SHA256_V1 = (
    "a31d9c4cfe6507ca642c061c14cdb40b8ebe35b6ea81783a2199df2bb3c0e475"
)
V66D_ACTOR_LOG_FILE_SHA256_V1 = (
    "aa10617c347b7ce5449165580dd4eaa98bb5131cfde5fcf9cda1134b380390e0"
)
MODEL_INDEX_FILE_SHA256_V1 = (
    "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
)
MODEL_CONFIG_FILE_SHA256_V1 = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
LORA_MASTER_IDENTITY_SHA256_V1 = (
    "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
)


def canonical_sha256_v1(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _exact_int_v1(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"v1 {label} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"v1 {label} must be >= {minimum}")
    return result


def _finite_float_v1(value: Any, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"v1 {label} must be real")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"v1 {label} must be finite" + ("/positive" if positive else ""))
    return result


def _hex_sha_v1(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"v1 {label} must be lowercase SHA-256")
    return value


def _strict_fields_v1(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"v1 {label} fields changed")


def _tensor_shape_v1(shape: Sequence[int]) -> tuple[int, ...]:
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or len(shape) < 1
    ):
        raise ValueError("v1 tensor shape must have positive rank")
    return tuple(
        _exact_int_v1(dimension, f"shape dimension {index}", minimum=1)
        for index, dimension in enumerate(shape)
    )


def _shape_v1(shape: Sequence[int]) -> tuple[int, int]:
    dimensions = _tensor_shape_v1(shape)
    if len(dimensions) != 2:
        raise ValueError("v1 structured tensor shape must be rank two")
    return dimensions


def _float32_v1(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _splitmix64_v1(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & MASK64_V1
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64_V1
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64_V1
    return (value ^ (value >> 31)) & MASK64_V1


def _rng_key_v1(seed: int, tensor_key: str, domain: str) -> int:
    seed = _exact_int_v1(seed, "direction seed")
    if not isinstance(tensor_key, str) or not tensor_key:
        raise ValueError("v1 tensor key must be nonempty")
    if not isinstance(domain, str) or not domain:
        raise ValueError("v1 RNG domain must be nonempty")
    payload = (
        f"{RNG_ALGORITHM_V1}\0{seed}\0{tensor_key}\0{domain}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def absolute_normal_v1(seed: int, tensor_key: str, domain: str, index: int) -> float:
    """Return one FP32 normal determined only by its absolute ordinal."""
    index = _exact_int_v1(index, "absolute RNG index")
    if index >= (1 << 62):
        raise ValueError("v1 absolute RNG index exceeds the sealed counter domain")
    key = _rng_key_v1(seed, tensor_key, domain)
    first = _splitmix64_v1((key + 2 * index) & MASK64_V1)
    second = _splitmix64_v1((key + 2 * index + 1) & MASK64_V1)
    # Use 53 high bits and an open interval to avoid log(0).  The CPU oracle
    # rounds the result to FP32 exactly once.
    u1 = ((first >> 11) + 0.5) / float(1 << 53)
    u2 = ((second >> 11) + 0.5) / float(1 << 53)
    value = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return _float32_v1(value)


def noise_identity_v1(
    method: str,
    seed: int,
    tensor_key: str,
    shape: Sequence[int],
    structured_rank: int | None = None,
) -> dict[str, Any]:
    dimensions = _tensor_shape_v1(shape)
    seed = _exact_int_v1(seed, "direction seed")
    if not isinstance(tensor_key, str) or not tensor_key:
        raise ValueError("v1 tensor key must be nonempty")
    if method == "iid_absolute_index":
        if structured_rank is not None:
            raise ValueError("v1 IID noise must not declare a structured rank")
        rank = None
        scale = 1.0
    elif method == "structured_outer_product":
        rows, columns = _shape_v1(dimensions)
        rank = _exact_int_v1(structured_rank, "structured rank", minimum=1)
        if rank not in STRUCTURED_RANKS_V1 or rank > min(rows, columns):
            raise ValueError("v1 structured rank is not registered for this matrix")
        scale = 1.0 / math.sqrt(rank)
    else:
        raise ValueError("v1 perturbation method is not registered")
    result = {
        "schema": "structured-es-noise-identity-v1",
        "method": method,
        "rng_algorithm": RNG_ALGORITHM_V1,
        "seed": seed,
        "tensor_key": tensor_key,
        "shape": list(dimensions),
        "structured_rank": rank,
        "rank_scale": scale,
        "absolute_indexed": True,
    }
    result["identity_sha256"] = canonical_sha256_v1(result)
    return result


def noise_chunk_v1(
    method: str,
    seed: int,
    tensor_key: str,
    shape: Sequence[int],
    start: int,
    count: int,
    structured_rank: int | None = None,
) -> torch.Tensor:
    identity = noise_identity_v1(method, seed, tensor_key, shape, structured_rank)
    dimensions = identity["shape"]
    total = math.prod(dimensions)
    start = _exact_int_v1(start, "chunk start")
    count = _exact_int_v1(count, "chunk count", minimum=1)
    if start + count > total:
        raise ValueError("v1 noise chunk exceeds the tensor surface")
    if method == "iid_absolute_index":
        values = [
            absolute_normal_v1(seed, tensor_key, "iid_element", index)
            for index in range(start, start + count)
        ]
        return torch.tensor(values, dtype=torch.float32)

    rank = identity["structured_rank"]
    rows, columns = dimensions
    row_ids = sorted({index // columns for index in range(start, start + count)})
    column_ids = sorted({index % columns for index in range(start, start + count)})
    left = {
        row: [
            absolute_normal_v1(
                seed,
                tensor_key,
                f"structured_left_rank_{rank}",
                row * rank + component,
            )
            for component in range(rank)
        ]
        for row in row_ids
    }
    right = {
        column: [
            absolute_normal_v1(
                seed,
                tensor_key,
                f"structured_right_rank_{rank}",
                column * rank + component,
            )
            for component in range(rank)
        ]
        for column in column_ids
    }
    scale = identity["rank_scale"]
    values = []
    for index in range(start, start + count):
        row, column = divmod(index, columns)
        dot = 0.0
        for component in range(rank):
            product = _float32_v1(
                float(left[row][component]) * float(right[column][component])
            )
            dot = _float32_v1(dot + product)
        values.append(_float32_v1(dot * scale))
    return torch.tensor(values, dtype=torch.float32)


def shard_range_v1(total: int, world_size: int, rank: int) -> tuple[int, int]:
    total = _exact_int_v1(total, "shard total", minimum=1)
    world_size = _exact_int_v1(world_size, "world size", minimum=1)
    rank = _exact_int_v1(rank, "shard rank")
    if rank >= world_size:
        raise ValueError("v1 shard rank is outside world size")
    return (total * rank // world_size, total * (rank + 1) // world_size)


def chunk_ranges_v1(start: int, end: int, chunk_elements: int) -> list[tuple[int, int]]:
    start = _exact_int_v1(start, "range start")
    end = _exact_int_v1(end, "range end")
    chunk_elements = _exact_int_v1(chunk_elements, "chunk elements", minimum=1)
    if end <= start:
        raise ValueError("v1 chunk range must be nonempty")
    result = []
    cursor = start
    while cursor < end:
        next_cursor = min(end, cursor + chunk_elements)
        result.append((cursor, next_cursor))
        cursor = next_cursor
    return result


def _tensor_sha256_v1(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
        raise RuntimeError("v1 digest requires CPU FP32 values")
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def build_reconstruction_receipt_v1(
    method: str,
    seed: int,
    tensor_key: str,
    shape: Sequence[int],
    *,
    structured_rank: int | None = None,
    world_size: int = WORLD_SIZE_V1,
    chunk_elements: int = CHUNK_ELEMENTS_V1,
) -> dict[str, Any]:
    identity = noise_identity_v1(method, seed, tensor_key, shape, structured_rank)
    total = math.prod(identity["shape"])
    world_size = _exact_int_v1(world_size, "world size", minimum=1)
    chunks = []
    full_digest = hashlib.sha256()
    # Generate in canonical absolute order for the partition-independent digest.
    for rank in range(world_size):
        shard_start, shard_end = shard_range_v1(total, world_size, rank)
        if shard_end == shard_start:
            continue
        for start, end in chunk_ranges_v1(shard_start, shard_end, chunk_elements):
            values = noise_chunk_v1(
                method,
                seed,
                tensor_key,
                shape,
                start,
                end - start,
                structured_rank,
            )
            raw = values.contiguous().view(torch.uint8).numpy().tobytes()
            full_digest.update(raw)
            chunks.append({
                "rank": rank,
                "start": start,
                "end": end,
                "value_sha256": hashlib.sha256(raw).hexdigest(),
            })
    result = {
        "schema": "structured-es-exact-reconstruction-receipt-v1",
        "noise_identity": identity,
        "world_size": world_size,
        "chunk_elements": chunk_elements,
        "total_elements": total,
        "chunks": chunks,
        "canonical_full_stream_sha256": full_digest.hexdigest(),
        "dense_noise_materialized": False,
    }
    result["content_sha256"] = canonical_sha256_v1(result)
    return result


def validate_reconstruction_receipt_v1(receipt: Mapping[str, Any]) -> dict[str, Any]:
    _strict_fields_v1(
        receipt,
        {
            "schema", "noise_identity", "world_size", "chunk_elements",
            "total_elements", "chunks", "canonical_full_stream_sha256",
            "dense_noise_materialized", "content_sha256",
        },
        "reconstruction receipt",
    )
    compact = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if (
        receipt["schema"] != "structured-es-exact-reconstruction-receipt-v1"
        or receipt["dense_noise_materialized"] is not False
        or canonical_sha256_v1(compact) != receipt["content_sha256"]
    ):
        raise RuntimeError("v1 reconstruction receipt identity changed")
    identity = receipt["noise_identity"]
    expected_identity = noise_identity_v1(
        identity.get("method", ""),
        identity.get("seed", -1),
        identity.get("tensor_key", ""),
        identity.get("shape", []),
        identity.get("structured_rank"),
    )
    if identity != expected_identity:
        raise RuntimeError("v1 noise identity changed")
    total = math.prod(identity["shape"])
    world_size = _exact_int_v1(receipt["world_size"], "world size", minimum=1)
    chunk_elements = _exact_int_v1(
        receipt["chunk_elements"], "chunk elements", minimum=1
    )
    if receipt["total_elements"] != total:
        raise RuntimeError("v1 reconstruction total changed")
    chunks = receipt["chunks"]
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("v1 reconstruction chunks are absent")
    ordered = sorted(chunks, key=lambda item: (item.get("start", -1), item.get("end", -1)))
    cursor = 0
    full_digest = hashlib.sha256()
    for chunk in ordered:
        _strict_fields_v1(
            chunk, {"rank", "start", "end", "value_sha256"}, "chunk receipt"
        )
        rank = _exact_int_v1(chunk["rank"], "chunk rank")
        start = _exact_int_v1(chunk["start"], "chunk start")
        end = _exact_int_v1(chunk["end"], "chunk end")
        if start != cursor or end <= start or end > total:
            raise RuntimeError("v1 reconstruction has a gap, overlap, or duplicate")
        expected_shard = shard_range_v1(total, world_size, rank)
        if (
            start < expected_shard[0]
            or end > expected_shard[1]
            or end - start > chunk_elements
        ):
            raise RuntimeError("v1 chunk crossed its absolute shard")
        values = noise_chunk_v1(
            identity["method"],
            identity["seed"],
            identity["tensor_key"],
            identity["shape"],
            start,
            end - start,
            identity["structured_rank"],
        )
        raw = values.contiguous().view(torch.uint8).numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != chunk["value_sha256"]:
            raise RuntimeError("v1 chunk values are not absolute-index reconstruction")
        full_digest.update(raw)
        cursor = end
    if cursor != total or full_digest.hexdigest() != receipt[
        "canonical_full_stream_sha256"
    ]:
        raise RuntimeError("v1 canonical reconstructed stream changed")
    return {
        "status": "exact_absolute_index_reconstruction",
        "noise_identity_sha256": identity["identity_sha256"],
        "canonical_full_stream_sha256": full_digest.hexdigest(),
    }


def streamed_candidate_chunks_v1(
    master: torch.Tensor,
    method: str,
    seed: int,
    tensor_key: str,
    sigma: float,
    sign: int,
    *,
    structured_rank: int | None = None,
    chunk_elements: int = CHUNK_ELEMENTS_V1,
):
    if (
        not isinstance(master, torch.Tensor)
        or master.device.type != "cpu"
        or master.dtype != torch.float32
        or master.ndim < 1
        or not bool(torch.isfinite(master).all())
    ):
        raise RuntimeError("v1 streamed candidate requires a finite CPU FP32 matrix")
    sigma = _finite_float_v1(sigma, "sigma", positive=True)
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("v1 candidate sign must be exactly +/-1")
    flat = master.reshape(-1)
    for start, end in chunk_ranges_v1(0, flat.numel(), chunk_elements):
        noise = noise_chunk_v1(
            method,
            seed,
            tensor_key,
            list(master.shape),
            start,
            end - start,
            structured_rank,
        )
        candidate = flat[start:end].add(noise, alpha=sign * sigma).contiguous()
        yield {
            "start": start,
            "end": end,
            "candidate": candidate,
            "noise": noise,
            "dense_candidate_materialized": False,
            "dense_noise_materialized": False,
        }


def streamed_weighted_update_chunks_v1(
    shape: Sequence[int],
    method: str,
    seeds: Sequence[int],
    coefficients: Sequence[float],
    tensor_key: str,
    sigma: float,
    *,
    structured_rank: int | None = None,
    chunk_elements: int = CHUNK_ELEMENTS_V1,
):
    dimensions = _tensor_shape_v1(shape)
    if method == "structured_outer_product":
        _shape_v1(dimensions)
    seeds = [_exact_int_v1(seed, "direction seed") for seed in seeds]
    coefficients = [
        _finite_float_v1(value, "pair coefficient") for value in coefficients
    ]
    sigma = _finite_float_v1(sigma, "sigma", positive=True)
    if not seeds or len(seeds) != len(coefficients) or len(set(seeds)) != len(seeds):
        raise ValueError("v1 update seeds/coefficients must be unique and matched")
    population = len(seeds)
    total = math.prod(dimensions)
    for start, end in chunk_ranges_v1(0, total, chunk_elements):
        accumulator = torch.zeros(end - start, dtype=torch.float32)
        for seed, coefficient in zip(seeds, coefficients, strict=True):
            noise = noise_chunk_v1(
                method,
                seed,
                tensor_key,
                dimensions,
                start,
                end - start,
                structured_rank,
            )
            accumulator.add_(noise, alpha=coefficient)
        accumulator.mul_(1.0 / (2.0 * population * sigma))
        if not bool(torch.isfinite(accumulator).all()):
            raise FloatingPointError("v1 streamed structured update became nonfinite")
        yield {
            "start": start,
            "end": end,
            "update": accumulator.contiguous(),
            "dense_update_materialized": False,
            "dense_noise_materialized": False,
        }


def structured_moment_theory_v1(rank: int) -> dict[str, Any]:
    rank = _exact_int_v1(rank, "structured rank", minimum=1)
    if rank not in STRUCTURED_RANKS_V1:
        raise ValueError("v1 structured rank is not registered")
    # Each term U_iq*V_jq has variance one and fourth moment nine.  Summing k
    # independent terms and dividing by sqrt(k) yields 3 + 6/k.
    return {
        "rank": rank,
        "entry_mean": 0.0,
        "entry_variance": 1.0,
        "entry_fourth_moment": 3.0 + 6.0 / rank,
        "entry_excess_kurtosis": 6.0 / rank,
        "expected_frobenius_l2_squared_per_element": 1.0,
        "rank_scale": 1.0 / math.sqrt(rank),
        "distinct_entries_are_uncorrelated": True,
        "finite_difference_statement": (
            "isotropic first-order directional estimator; not an exact "
            "Gaussian-smoothing score estimator at finite sigma"
        ),
    }


def lora_streaming_accounting_v1(
    tensor_shapes: Sequence[Mapping[str, Any]],
    chunk_elements: int = CHUNK_ELEMENTS_V1,
) -> dict[str, Any]:
    chunk_elements = _exact_int_v1(chunk_elements, "chunk elements", minimum=1)
    rows = []
    for item in tensor_shapes:
        if set(item) != {"key", "shape"}:
            raise ValueError("v1 tensor shape inventory fields changed")
        row, column = _shape_v1(item["shape"])
        rows.append((item["key"], row, column))
    if (
        len(rows) != LORA_TENSORS_V1
        or sum(row * column for _, row, column in rows) != LORA_ELEMENTS_V1
    ):
        raise RuntimeError("v1 LoRA streaming surface changed")
    maximum_chunk = min(chunk_elements, max(row * column for _, row, column in rows))
    methods = {
        "iid_absolute_index": {
            "unique_random_values_per_direction": LORA_ELEMENTS_V1,
            "factor_values_per_direction": 0,
            "factor_bytes_per_direction": 0,
            "maximum_factor_cache_bytes_per_tensor": 0,
            "candidate_scratch_ceiling_bytes": 2 * maximum_chunk * 4,
            "weighted_update_scratch_ceiling_bytes": 3 * maximum_chunk * 4,
            "expected_entry_variance": 1.0,
            "dense_noise_elements_allocated": 0,
            "dense_candidate_elements_allocated": 0,
        }
    }
    total_dimension_sum = sum(row + column for _, row, column in rows)
    for rank in STRUCTURED_RANKS_V1:
        factor_values = total_dimension_sum * rank
        max_factor_values = max((row + column) * rank for _, row, column in rows)
        methods[f"structured_rank_{rank}"] = {
            "unique_random_values_per_direction": factor_values,
            "factor_values_per_direction": factor_values,
            "factor_bytes_per_direction": factor_values * 4,
            "factor_draw_ratio_vs_iid": factor_values / LORA_ELEMENTS_V1,
            "maximum_factor_cache_bytes_per_tensor": max_factor_values * 4,
            "candidate_scratch_ceiling_bytes": (
                max_factor_values * 4 + 2 * maximum_chunk * 4
            ),
            "weighted_update_scratch_ceiling_bytes": (
                max_factor_values * 4 + 3 * maximum_chunk * 4
            ),
            "expected_entry_variance": 1.0,
            "entry_fourth_moment": 3.0 + 6.0 / rank,
            "dense_noise_elements_allocated": 0,
            "dense_candidate_elements_allocated": 0,
        }
    return {
        "schema": "structured-es-lora-streaming-accounting-v1",
        "tensor_count": len(rows),
        "elements": LORA_ELEMENTS_V1,
        "fp32_master_bytes": LORA_FP32_BYTES_V1,
        "runtime_bf16_view_bytes": LORA_RUNTIME_BF16_BYTES_V1,
        "chunk_elements": chunk_elements,
        "maximum_chunk_bytes": maximum_chunk * 4,
        "whole_surface_noise_or_candidate_prohibited": True,
        "methods": methods,
    }


@dataclass
class RuntimeTransactionOracleV1:
    total_elements: int
    canonical_master_sha256: str
    canonical_runtime_sha256: str
    phase: str = "quiescent"
    poisoned: bool = False
    active_noise_identity_sha256: str | None = None
    candidate_ranges: list[tuple[int, int]] = field(default_factory=list)
    restore_ranges: list[tuple[int, int]] = field(default_factory=list)
    poison_reason: str | None = None

    def __post_init__(self):
        self.total_elements = _exact_int_v1(
            self.total_elements, "transaction elements", minimum=1
        )
        _hex_sha_v1(self.canonical_master_sha256, "canonical master")
        _hex_sha_v1(self.canonical_runtime_sha256, "canonical runtime")

    def _require_healthy(self):
        if self.poisoned:
            raise RuntimeError("v1 structured runtime is terminally poisoned")

    def _record(self, collection: list[tuple[int, int]], start: int, end: int):
        start = _exact_int_v1(start, "transaction range start")
        end = _exact_int_v1(end, "transaction range end")
        if end <= start or end > self.total_elements:
            self.poison("invalid_chunk_range")
            raise RuntimeError("v1 transaction chunk range is invalid")
        if any(not (end <= left or start >= right) for left, right in collection):
            self.poison("overlapping_chunk_write")
            raise RuntimeError("v1 transaction chunk write overlaps")
        collection.append((start, end))

    def begin_candidate(self, noise_identity_sha256: str):
        self._require_healthy()
        if self.phase != "quiescent":
            raise RuntimeError("v1 candidate began outside quiescent state")
        self.active_noise_identity_sha256 = _hex_sha_v1(
            noise_identity_sha256, "noise identity"
        )
        self.candidate_ranges = []
        self.phase = "materializing_candidate"

    def record_candidate_chunk(self, start: int, end: int):
        self._require_healthy()
        if self.phase != "materializing_candidate":
            raise RuntimeError("v1 candidate chunk arrived in the wrong phase")
        self._record(self.candidate_ranges, start, end)

    def finish_candidate(self):
        self._require_healthy()
        if self.phase != "materializing_candidate" or not _exact_coverage_v1(
            self.candidate_ranges, self.total_elements
        ):
            self.poison("incomplete_candidate_coverage")
            raise RuntimeError("v1 candidate coverage is incomplete")
        self.phase = "candidate_active"

    def begin_restore(self):
        self._require_healthy()
        if self.phase not in {"materializing_candidate", "candidate_active"}:
            raise RuntimeError("v1 restore began without an uncertain/active candidate")
        self.restore_ranges = []
        self.phase = "restoring_canonical"

    def record_restore_chunk(self, start: int, end: int):
        self._require_healthy()
        if self.phase != "restoring_canonical":
            raise RuntimeError("v1 restore chunk arrived in the wrong phase")
        self._record(self.restore_ranges, start, end)

    def finish_restore(self, observed_master_sha256: str, observed_runtime_sha256: str):
        self._require_healthy()
        if (
            self.phase != "restoring_canonical"
            or not _exact_coverage_v1(self.restore_ranges, self.total_elements)
            or observed_master_sha256 != self.canonical_master_sha256
            or observed_runtime_sha256 != self.canonical_runtime_sha256
        ):
            self.poison("restore_coverage_or_identity_failure")
            raise RuntimeError("v1 exact structured restore could not be proven")
        self.phase = "quiescent"
        self.active_noise_identity_sha256 = None
        self.candidate_ranges = []
        self.restore_ranges = []

    def poison(self, reason: str):
        if not isinstance(reason, str) or not reason:
            raise ValueError("v1 poison reason must be nonempty")
        self.poisoned = True
        self.phase = "terminal_poison"
        self.poison_reason = reason

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "structured-es-runtime-transaction-receipt-v1",
            "total_elements": self.total_elements,
            "canonical_master_sha256": self.canonical_master_sha256,
            "canonical_runtime_sha256": self.canonical_runtime_sha256,
            "phase": self.phase,
            "poisoned": self.poisoned,
            "poison_reason": self.poison_reason,
            "active_noise_identity_sha256": self.active_noise_identity_sha256,
            "candidate_ranges": [list(item) for item in sorted(self.candidate_ranges)],
            "restore_ranges": [list(item) for item in sorted(self.restore_ranges)],
        }


def _exact_coverage_v1(ranges: Sequence[tuple[int, int]], total: int) -> bool:
    cursor = 0
    for start, end in sorted(ranges):
        if start != cursor or end <= start:
            return False
        cursor = end
    return cursor == total


def comparison_arms_v1() -> list[dict[str, Any]]:
    arms = [{
        "arm_id": "lora_iid_absolute_index",
        "surface": "matched_lora_4528128",
        "method": "iid_absolute_index",
        "structured_rank": None,
        "causal_group": "matched_lora_space",
    }]
    for rank in STRUCTURED_RANKS_V1:
        arms.append({
            "arm_id": f"lora_structured_rank_{rank}",
            "surface": "matched_lora_4528128",
            "method": "structured_outer_product",
            "structured_rank": rank,
            "causal_group": "matched_lora_space",
        })
    arms.append({
        "arm_id": "dense_fullweight_iid_inplace_system_anchor",
        "surface": "qwen36_fullweight_35951822704",
        "method": "iid_absolute_index",
        "structured_rank": None,
        "causal_group": "unmatched_system_anchor_no_quality_causal_claim",
    })
    return arms


def validate_preregistration_v1(plan: Mapping[str, Any], *, launch: bool = False) -> dict:
    fields = {
        "schema", "status", "purpose", "authorization", "dependencies",
        "source_contracts", "rng_contract", "structured_scale_theory",
        "surfaces", "streaming_contract", "memory_bandwidth_contract",
        "transaction_contract", "compute_contract", "arms",
        "benchmark_gates", "failure_policy", "reporting",
        "implementation_bindings", "artifacts", "content_sha256_before_self_field",
    }
    _strict_fields_v1(plan, fields, "preregistration")
    compact = {key: value for key, value in plan.items()
               if key != "content_sha256_before_self_field"}
    if (
        plan["schema"] != SCHEMA_V1
        or canonical_sha256_v1(compact) != plan["content_sha256_before_self_field"]
    ):
        raise RuntimeError("v1 structured preregistration identity changed")
    authorization = plan["authorization"]
    _strict_fields_v1(
        authorization,
        {
            "cpu_correctness", "gpu_launch", "train_identity_use", "dev_or_ood",
            "protected_holdout", "live_run_read", "candidate_commit", "promotion",
        },
        "authorization",
    )
    if authorization != {
        "cpu_correctness": True,
        "gpu_launch": False,
        "train_identity_use": True,
        "dev_or_ood": False,
        "protected_holdout": False,
        "live_run_read": False,
        "candidate_commit": False,
        "promotion": False,
    }:
        raise RuntimeError("v1 structured authorization changed")
    sources = plan["source_contracts"]
    if (
        sources["fp32_optimizer_sigma"]["file_sha256"]
        != OPTIMIZER_CONTRACT_FILE_SHA256_V1
        or sources["fp32_optimizer_sigma"]["content_sha256"]
        != OPTIMIZER_CONTRACT_CONTENT_SHA256_V1
        or sources["v66d_preregistration"]["file_sha256"]
        != V66D_PREREG_FILE_SHA256_V1
        or sources["v66d_preregistration"]["content_sha256"]
        != V66D_PREREG_CONTENT_SHA256_V1
        or sources["v66d_accepted_telemetry"]["report_file_sha256"]
        != V66D_REPORT_FILE_SHA256_V1
        or sources["v66d_accepted_telemetry"]["report_content_sha256"]
        != V66D_REPORT_CONTENT_SHA256_V1
        or sources["v66d_accepted_telemetry"]["gpu_telemetry_file_sha256"]
        != V66D_TELEMETRY_FILE_SHA256_V1
        or sources["v66d_accepted_telemetry"]["actor_log_file_sha256"]
        != V66D_ACTOR_LOG_FILE_SHA256_V1
        or sources["v66d_accepted_telemetry"]["accepted"] is not True
    ):
        raise RuntimeError("v1 source contract or accepted telemetry changed")
    rng = plan["rng_contract"]
    if (
        rng.get("algorithm") != RNG_ALGORITHM_V1
        or rng.get("absolute_indexed") is not True
        or rng.get("chunk_shard_tensor_order_independent") is not True
        or rng.get("left_right_tensor_and_method_domains_distinct") is not True
        or rng.get("local_chunk_index_rng") != "prohibited"
    ):
        raise RuntimeError("v1 absolute-index RNG contract changed")
    theory = plan["structured_scale_theory"]
    if (
        theory.get("ranks") != list(STRUCTURED_RANKS_V1)
        or theory.get("rank_scale") != "1/sqrt(k)"
        or theory.get("entry_variance") != 1.0
        or theory.get("expected_frobenius_square") != "rows*columns"
        or theory.get("extra_row_or_column_normalization") != "prohibited"
        or theory.get("moments")
        != [structured_moment_theory_v1(rank) for rank in STRUCTURED_RANKS_V1]
    ):
        raise RuntimeError("v1 structured scale theory changed")
    surfaces = plan["surfaces"]
    if (
        surfaces["lora"]["tensor_count"] != LORA_TENSORS_V1
        or surfaces["lora"]["elements"] != LORA_ELEMENTS_V1
        or surfaces["lora"]["fp32_bytes"] != LORA_FP32_BYTES_V1
        or surfaces["lora"]["master_identity_sha256"]
        != LORA_MASTER_IDENTITY_SHA256_V1
        or surfaces["dense_fullweight"]["tensor_count"] != FULL_MODEL_TENSORS_V1
        or surfaces["dense_fullweight"]["elements"] != FULL_MODEL_ELEMENTS_V1
        or surfaces["dense_fullweight"]["bf16_bytes"] != FULL_MODEL_BF16_BYTES_V1
        or surfaces["dense_fullweight"]["fp32_master_bytes"]
        != FULL_MODEL_FP32_MASTER_BYTES_V1
        or surfaces["dense_fullweight"]["index_file_sha256"]
        != MODEL_INDEX_FILE_SHA256_V1
        or surfaces["dense_fullweight"]["config_file_sha256"]
        != MODEL_CONFIG_FILE_SHA256_V1
        or surfaces["dense_fullweight"]["metadata_only_no_tensor_payload_loaded"]
        is not True
        or surfaces["dense_fullweight"]["quality_causal_comparison_to_lora"]
        is not False
        or surfaces["dense_fullweight"]["systems_anchor_only"] is not True
    ):
        raise RuntimeError("v1 parameter surface changed")
    streaming = plan["streaming_contract"]
    if (
        streaming.get("chunk_elements") != CHUNK_ELEMENTS_V1
        or streaming.get("dense_full_surface_noise_materialization") != "prohibited"
        or streaming.get("dense_full_surface_candidate_materialization") != "prohibited"
        or streaming.get("fixed_ascending_rank_component_reduction") is not True
        or streaming.get("exact_absolute_coverage_no_gap_overlap_duplicate") is not True
    ):
        raise RuntimeError("v1 streaming contract changed")
    inventory = surfaces["lora"]["tensor_shapes"]
    if plan["memory_bandwidth_contract"]["lora"] != lora_streaming_accounting_v1(
        inventory
    ):
        raise RuntimeError("v1 structured scratch/byte accounting changed")
    if (
        plan["memory_bandwidth_contract"].get(
            "common_candidate_runtime_write_bytes_per_lora_signed_candidate"
        )
        != LORA_RUNTIME_BF16_BYTES_V1
        or plan["memory_bandwidth_contract"].get(
            "structured_does_not_reduce_runtime_install_bytes_without_fusion"
        )
        is not True
    ):
        raise RuntimeError("v1 common LoRA runtime write contract changed")
    dense_memory = plan["memory_bandwidth_contract"]["dense_fullweight_system_anchor"]
    if (
        dense_memory.get("whole_surface_noise_elements_allocated") != 0
        or dense_memory.get("whole_surface_candidate_elements_allocated") != 0
        or dense_memory.get("inplace_tensor_streaming_required") is not True
        or dense_memory.get("bf16_model_bytes_per_replica")
        != FULL_MODEL_BF16_BYTES_V1
        or dense_memory.get("fp32_master_bytes_per_replica")
        != FULL_MODEL_FP32_MASTER_BYTES_V1
        or dense_memory.get("four_replica_fp32_master_bytes")
        != FULL_MODEL_FP32_MASTER_BYTES_V1 * WORLD_SIZE_V1
        or dense_memory.get("maximum_single_tensor_fp32_noise_bytes")
        != FULL_MODEL_MAX_TENSOR_ELEMENTS_V1 * 4
        or dense_memory.get("weighted_update_scratch_ceiling_bytes")
        != FULL_MODEL_MAX_TENSOR_ELEMENTS_V1 * 8
        or dense_memory.get("candidate_parameter_write_bytes_per_signed_candidate")
        != FULL_MODEL_BF16_BYTES_V1
        or dense_memory.get("capacity_preflight_required_before_launch") is not True
    ):
        raise RuntimeError("v1 dense anchor scratch/byte accounting changed")
    if plan["arms"] != comparison_arms_v1():
        raise RuntimeError("v1 structured arm grid changed")
    compute = plan["compute_contract"]
    if (
        compute.get("systems_updates_per_arm_seed") != 1
        or compute.get("systems_replicate_seed") != 1701
        or compute.get("gpu_second_ceiling_per_systems_arm_seed") != 14_400.0
        or compute.get("directions_per_update") != DIRECTIONS_PER_UPDATE_V1
        or compute.get("signed_candidates_per_update")
        != SIGNED_CANDIDATES_PER_UPDATE_V1
        or compute.get("train_units_per_candidate") != TRAIN_UNITS_PER_CANDIDATE_V1
        or compute.get("rollouts_per_systems_arm_seed") != ROLLOUTS_PER_UPDATE_V1
        or compute.get("quality_sigma_schedule") != list(SIGMA_SCHEDULE_V1)
        or compute.get("quality_replicate_seeds") != list(REPLICATE_SEEDS_V1)
        or compute.get("quality_lora_updates_per_seed") != len(SIGMA_SCHEDULE_V1)
        or compute.get("quality_rollouts_per_lora_arm_seed")
        != len(SIGMA_SCHEDULE_V1) * ROLLOUTS_PER_UPDATE_V1
        or compute.get("fixed_optimizer") != "sgd"
        or compute.get("fixed_sigma_mode") != "global"
        or compute.get("update_budget_ratio") != UPDATE_BUDGET_RATIO_V1
        or compute.get("update_norm_relative_tolerance")
        != UPDATE_NORM_RELATIVE_TOLERANCE_V1
        or compute.get("same_train_panel_prompt_decode_judge_and_telemetry")
        is not True
        or compute.get("failed_work_reallocation") != "prohibited"
    ):
        raise RuntimeError("v1 structured equal-work contract changed")
    dependencies = plan["dependencies"]
    _strict_fields_v1(
        dependencies,
        {
            "v66d_accepted_gpu_attribution_complete",
            "fp32_optimizer_sigma_cpu_contract_complete",
            "production_streaming_worker_complete",
            "optimizer_phase_pcie_profile_complete",
            "dense_fullweight_capacity_preflight_complete", "runtime_blockers",
        },
        "dependencies",
    )
    if (
        dependencies["v66d_accepted_gpu_attribution_complete"] is not True
        or dependencies["fp32_optimizer_sigma_cpu_contract_complete"] is not True
    ):
        raise RuntimeError("v1 completed source dependency changed")
    ready = (
        dependencies.get("production_streaming_worker_complete") is True
        and dependencies.get("optimizer_phase_pcie_profile_complete") is True
        and dependencies.get("dense_fullweight_capacity_preflight_complete") is True
    )
    if ready or plan["status"] != "sealed_cpu_correctness_runtime_dependencies_pending":
        raise RuntimeError("v1 structured runtime dependency/status gate changed")
    if launch:
        raise RuntimeError("v1 structured GPU launch is not authorized by this artifact")
    return {
        "status": "sealed_cpu_correctness_runtime_dependencies_pending",
        "content_sha256": plan["content_sha256_before_self_field"],
        "arm_count": len(plan["arms"]),
    }


def validate_systems_receipt_v1(plan: Mapping[str, Any], receipt: Mapping[str, Any]) -> dict:
    validate_preregistration_v1(plan)
    fields = {
        "schema", "plan_content_sha256", "arm_id", "replicate_seed",
        "directions", "signed_candidates", "train_units_per_candidate", "rollouts",
        "surface_elements", "method", "structured_rank", "rng_algorithm",
        "dense_noise_elements_allocated", "dense_candidate_elements_allocated",
        "peak_scratch_bytes", "scratch_ceiling_bytes", "candidate_bytes_written",
        "reward_vector_sha256", "reward_mean", "reward_variance",
        "stream_update_vs_oracle_max_abs", "stream_update_vs_oracle_max_ulp",
        "target_update_l2", "observed_update_l2", "update_norm_relative_error",
        "canonical_master_before_sha256", "canonical_master_after_sha256",
        "canonical_runtime_before_sha256", "canonical_runtime_after_sha256",
        "rollback_complete", "poisoned", "useful_physical_gpus",
        "v66d_telemetry_report_content_sha256", "protected_or_eval_opened",
        "charged_gpu_seconds",
    }
    _strict_fields_v1(receipt, fields, "systems receipt")
    if (
        receipt["schema"] != "structured-es-systems-receipt-v1"
        or receipt["plan_content_sha256"] != plan["content_sha256_before_self_field"]
    ):
        raise RuntimeError("v1 systems receipt identity changed")
    arms = {item["arm_id"]: item for item in plan["arms"]}
    arm = arms.get(receipt["arm_id"])
    replicate_seed = _exact_int_v1(receipt["replicate_seed"], "replicate seed")
    for field_name, label, minimum in (
        ("directions", "directions", 1),
        ("signed_candidates", "signed candidates", 1),
        ("train_units_per_candidate", "train units per candidate", 1),
        ("rollouts", "rollouts", 1),
        ("surface_elements", "surface elements", 1),
        ("dense_noise_elements_allocated", "dense noise elements", 0),
        ("dense_candidate_elements_allocated", "dense candidate elements", 0),
        ("candidate_bytes_written", "candidate bytes written", 1),
    ):
        _exact_int_v1(receipt[field_name], label, minimum=minimum)
    if receipt["structured_rank"] is not None:
        _exact_int_v1(receipt["structured_rank"], "receipt structured rank", minimum=1)
    if (
        arm is None
        or replicate_seed != plan["compute_contract"]["systems_replicate_seed"]
    ):
        raise RuntimeError("v1 systems arm/seed is not registered")
    expected_elements = (
        LORA_ELEMENTS_V1
        if arm["surface"] == "matched_lora_4528128"
        else FULL_MODEL_ELEMENTS_V1
    )
    charged_gpu_seconds = _finite_float_v1(
        receipt["charged_gpu_seconds"], "charged GPU seconds", True
    )
    if (
        receipt["directions"] != DIRECTIONS_PER_UPDATE_V1
        or receipt["signed_candidates"] != SIGNED_CANDIDATES_PER_UPDATE_V1
        or receipt["train_units_per_candidate"] != TRAIN_UNITS_PER_CANDIDATE_V1
        or receipt["rollouts"] != ROLLOUTS_PER_UPDATE_V1
        or receipt["surface_elements"] != expected_elements
        or receipt["method"] != arm["method"]
        or receipt["structured_rank"] != arm["structured_rank"]
        or receipt["rng_algorithm"] != RNG_ALGORITHM_V1
        or receipt["dense_noise_elements_allocated"] != 0
        or receipt["dense_candidate_elements_allocated"] != 0
    ):
        raise RuntimeError("v1 systems work/surface/allocation contract changed")
    peak = _exact_int_v1(receipt["peak_scratch_bytes"], "peak scratch bytes", minimum=1)
    ceiling = _exact_int_v1(
        receipt["scratch_ceiling_bytes"], "scratch ceiling bytes", minimum=1
    )
    if peak > ceiling:
        raise RuntimeError("v1 structured scratch ceiling exceeded")
    if arm["surface"] == "matched_lora_4528128":
        method_key = (
            "iid_absolute_index" if arm["structured_rank"] is None
            else f"structured_rank_{arm['structured_rank']}"
        )
        expected_ceiling = plan["memory_bandwidth_contract"]["lora"]["methods"][
            method_key
        ]["weighted_update_scratch_ceiling_bytes"]
        if ceiling != expected_ceiling:
            raise RuntimeError("v1 LoRA scratch ceiling changed")
        expected_candidate_bytes = (
            SIGNED_CANDIDATES_PER_UPDATE_V1 * LORA_RUNTIME_BF16_BYTES_V1
        )
    else:
        expected_ceiling = plan["memory_bandwidth_contract"][
            "dense_fullweight_system_anchor"
        ]["weighted_update_scratch_ceiling_bytes"]
        expected_candidate_bytes = (
            SIGNED_CANDIDATES_PER_UPDATE_V1 * FULL_MODEL_BF16_BYTES_V1
        )
        if ceiling != expected_ceiling:
            raise RuntimeError("v1 dense anchor scratch ceiling changed")
    if receipt["candidate_bytes_written"] != expected_candidate_bytes:
        raise RuntimeError("v1 candidate parameter write bytes changed")
    _hex_sha_v1(receipt["reward_vector_sha256"], "reward vector")
    _finite_float_v1(receipt["reward_mean"], "reward mean")
    variance = _finite_float_v1(receipt["reward_variance"], "reward variance")
    if variance <= 0.0:
        raise RuntimeError("v1 systems reward variance is not positive")
    max_abs = _finite_float_v1(
        receipt["stream_update_vs_oracle_max_abs"], "update max absolute error"
    )
    max_ulp = _exact_int_v1(
        receipt["stream_update_vs_oracle_max_ulp"], "update max ULP"
    )
    if max_abs < 0.0 or max_ulp > 2:
        raise RuntimeError("v1 streamed update disagrees with correctness oracle")
    target = _finite_float_v1(receipt["target_update_l2"], "target update L2", True)
    observed = _finite_float_v1(
        receipt["observed_update_l2"], "observed update L2", True
    )
    measured_error = abs(observed - target) / target
    reported_error = _finite_float_v1(
        receipt["update_norm_relative_error"], "update norm relative error"
    )
    if (
        reported_error < 0.0
        or reported_error != measured_error
        or measured_error > UPDATE_NORM_RELATIVE_TOLERANCE_V1
    ):
        raise RuntimeError("v1 structured update norm budget changed")
    for label in (
        "canonical_master_before_sha256", "canonical_master_after_sha256",
        "canonical_runtime_before_sha256", "canonical_runtime_after_sha256",
    ):
        _hex_sha_v1(receipt[label], label)
    if (
        receipt["canonical_master_before_sha256"]
        != receipt["canonical_master_after_sha256"]
        or receipt["canonical_runtime_before_sha256"]
        != receipt["canonical_runtime_after_sha256"]
        or receipt["rollback_complete"] is not True
        or receipt["poisoned"] is not False
        or receipt["useful_physical_gpus"] != [0, 1, 2, 3]
        or receipt["v66d_telemetry_report_content_sha256"]
        != V66D_REPORT_CONTENT_SHA256_V1
        or receipt["protected_or_eval_opened"] is not False
        or charged_gpu_seconds <= 0.0
        or charged_gpu_seconds
        > plan["compute_contract"]["gpu_second_ceiling_per_systems_arm_seed"]
    ):
        raise RuntimeError("v1 restore, telemetry, or access receipt changed")
    return {
        "status": "valid_structured_systems_receipt",
        "arm_id": arm["arm_id"],
        "replicate_seed": receipt["replicate_seed"],
    }
