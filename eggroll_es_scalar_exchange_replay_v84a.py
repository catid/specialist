#!/usr/bin/env python3
"""CPU-only scalar-consensus/full-replay contract for LoRA ES (V84A).

The accepted V72 update forms one dense FP32 tensor per LoRA tensor, reduces
that tensor across four actors, and then scales it.  Algebraically the same
update can be formed if every actor first agrees on the complete finite,
ordered ``(direction_seed, coefficient)`` list and independently replays the
noise generator.  This module makes that alternative executable only with
synthetic CPU tensors.  It has no live communicator, CUDA, model, dataset, or
promotion authority.

The prospective design deliberately exchanges IEEE-754 binary64 coefficient
bits.  V72 consumes Python ``float`` coefficients, so narrowing the wire value
to FP32 would change the accepted arithmetic before noise accumulation.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import struct
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

import structured_es_oracle_v1 as oracle


SCHEMA_V84A = "eggroll-es-scalar-exchange-replay-v84a"
CONSENSUS_SCHEMA_V84A = "synthetic-four-rank-scalar-consensus-v84a"
TRANSACTION_SCHEMA_V84A = "synthetic-scalar-replay-transaction-v84a"
ACCOUNTING_SCHEMA_V84A = "qwen36-lora-scalar-replay-accounting-v84a"
SYNTHETIC_AUTHORITY_V84A = "synthetic_cpu_fake_four_rank_only_v84a"

WORLD_SIZE_V84A = 4
DIRECTIONS_V84A = 8
SIGNED_CANDIDATES_V84A = 16
DIRECTIONS_PER_RANK_V84A = 2
TENSOR_COUNT_V84A = 70
MODULE_COUNT_V84A = 35
TOTAL_ELEMENTS_V84A = 4_528_128
TOTAL_FP32_BYTES_V84A = 18_112_512
MAX_TENSOR_ELEMENTS_V84A = 262_144
NATIVE_COLLECTIVE_CALLS_V84A = 70
FP32_BYTES_V84A = 4
SEED_BYTES_V84A = 8
COEFFICIENT_BYTES_V84A = 8
PAIR_BYTES_V84A = SEED_BYTES_V84A + COEFFICIENT_BYTES_V84A
DIGEST_BYTES_V84A = 32
DEFAULT_CHUNK_ELEMENTS_V84A = 16_384
MAXIMUM_UPDATE_ULP_V84A = 2

EXPECTED_SOURCE_MANIFEST_SHA256_V84A = (
    "e12f7199343477db3927bda67bf5f364030a47216be8aa2b30fc3b71c261da2b"
)
EXPECTED_ORDERED_KEY_SHA256_V84A = (
    "ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280"
)


def canonical_sha256_v84a(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_v84a(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _exact_int_v84a(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"V84A {label} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"V84A {label} must be >= {minimum}")
    return result


def _finite_v84a(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"V84A {label} must be real")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V84A {label} must be finite" + ("/positive" if positive else ""))
    # Canonicalize the two IEEE zero encodings.  They are algebraically equal,
    # and allowing both would create distinct retry identities for one update.
    return 0.0 if result == 0.0 else result


def _sha_v84a(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"V84A {label} must be lowercase SHA-256")
    return value


def _pair_v84a(seed: Any, coefficient: Any) -> dict[str, Any]:
    seed = _exact_int_v84a(seed, "direction seed")
    if seed >= (1 << 64):
        raise ValueError("V84A direction seed exceeds uint64 wire domain")
    coefficient = _finite_v84a(coefficient, "coefficient")
    wire = struct.pack(">Qd", seed, coefficient)
    return {
        "seed": seed,
        "coefficient": coefficient,
        "wire_hex": wire.hex(),
        "wire_bytes": PAIR_BYTES_V84A,
    }


def _pairs_v84a(value: Sequence[Any], label: str) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"V84A {label} must be a sequence")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, Mapping):
            _require_v84a(
                set(item) == {"seed", "coefficient"},
                f"V84A {label} fields changed at {index}",
            )
            result.append(_pair_v84a(item["seed"], item["coefficient"]))
        else:
            _require_v84a(
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) == 2,
                f"V84A {label} pair changed at {index}",
            )
            result.append(_pair_v84a(item[0], item[1]))
    return result


def seed_identity_v84a(seeds: Sequence[int]) -> str:
    values = [_exact_int_v84a(seed, "expected seed") for seed in seeds]
    _require_v84a(
        len(values) == DIRECTIONS_V84A
        and values == sorted(values)
        and len(set(values)) == DIRECTIONS_V84A,
        "V84A expected seed inventory changed",
    )
    return canonical_sha256_v84a(values)


def pair_identity_v84a(pairs: Sequence[Any]) -> str:
    values = _pairs_v84a(pairs, "pair identity")
    values.sort(key=lambda row: row["seed"])
    _require_v84a(
        len(values) == DIRECTIONS_V84A
        and len({row["seed"] for row in values}) == DIRECTIONS_V84A,
        "V84A pair identity coverage changed",
    )
    return canonical_sha256_v84a(
        [{"seed": row["seed"], "wire_hex": row["wire_hex"]} for row in values]
    )


def collapse_antithetic_rewards_v84a(
    signed_rewards: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse ``r+ * eps + r- * (-eps)`` to ``(r+ - r-) * eps``."""
    if (
        isinstance(signed_rewards, (str, bytes))
        or not isinstance(signed_rewards, Sequence)
        or len(signed_rewards) != SIGNED_CANDIDATES_V84A
    ):
        raise ValueError("V84A requires exactly sixteen signed rewards")
    grouped: dict[int, dict[int, float]] = {}
    for index, row in enumerate(signed_rewards):
        _require_v84a(
            isinstance(row, Mapping) and set(row) == {"reward", "seed", "sign"},
            f"V84A signed reward fields changed at {index}",
        )
        seed = _exact_int_v84a(row["seed"], "signed reward seed")
        sign = row["sign"]
        if isinstance(sign, bool) or sign not in (-1, 1):
            raise ValueError("V84A signed reward sign must be exactly +/-1")
        reward = _finite_v84a(row["reward"], "signed reward")
        bucket = grouped.setdefault(seed, {})
        _require_v84a(sign not in bucket, "V84A duplicate antithetic sign")
        bucket[sign] = reward
    _require_v84a(
        len(grouped) == DIRECTIONS_V84A
        and all(set(bucket) == {-1, 1} for bucket in grouped.values()),
        "V84A antithetic pair coverage changed",
    )
    return [
        {"seed": seed, "coefficient": grouped[seed][1] - grouped[seed][-1]}
        for seed in sorted(grouped)
    ]


def build_local_shard_v84a(rank: int, pairs: Sequence[Any]) -> dict[str, Any]:
    rank = _exact_int_v84a(rank, "rank")
    if rank >= WORLD_SIZE_V84A:
        raise ValueError("V84A rank is outside the four-rank world")
    records = _pairs_v84a(pairs, "local shard")
    _require_v84a(
        len(records) == DIRECTIONS_PER_RANK_V84A,
        "V84A local shard must contain exactly two directions",
    )
    _require_v84a(
        len({row["seed"] for row in records}) == len(records),
        "V84A local shard contains duplicate seeds",
    )
    records.sort(key=lambda row: row["seed"])
    body = {
        "schema": "scalar-pair-local-shard-v84a",
        "rank": rank,
        "world_size": WORLD_SIZE_V84A,
        "pair_wire_format": "big_endian_uint64_seed_plus_ieee754_binary64_coefficient",
        "pairs": records,
        "wire_bytes": len(records) * PAIR_BYTES_V84A,
    }
    return {**body, "content_sha256": canonical_sha256_v84a(body)}


def build_update_plan_v84a(
    *,
    seeds: Sequence[int],
    pairs: Sequence[Any],
    method: str,
    sigma: float,
    step_size: float,
    structured_rank: int | None,
    population_acceptance_sha256: str,
    expected_master_sha256: str,
    plan_id: str,
    update_sequence: int,
) -> dict[str, Any]:
    expected_seed_sha256 = seed_identity_v84a(seeds)
    expected_pair_sha256 = pair_identity_v84a(pairs)
    sigma = _finite_v84a(sigma, "sigma", positive=True)
    step_size = _finite_v84a(step_size, "step size", positive=True)
    if method == "iid_absolute_index":
        if structured_rank is not None:
            raise ValueError("V84A IID plan must not declare a structured rank")
        rank = None
    elif method == "structured_outer_product":
        rank = _exact_int_v84a(structured_rank, "structured rank", 1)
        _require_v84a(rank in oracle.STRUCTURED_RANKS_V1, "V84A rank is not registered")
    else:
        raise ValueError("V84A perturbation method is not registered")
    if not isinstance(plan_id, str) or not plan_id:
        raise ValueError("V84A plan id is empty")
    body = {
        "schema": "scalar-replay-update-plan-v84a",
        "world_size": WORLD_SIZE_V84A,
        "directions": DIRECTIONS_V84A,
        "signed_candidates": SIGNED_CANDIDATES_V84A,
        "rng_algorithm": oracle.RNG_ALGORITHM_V1,
        "method": method,
        "structured_rank": rank,
        "sigma": sigma,
        "step_size": step_size,
        "normalization": "step_size/(2*directions*sigma)",
        "expected_seed_sha256": expected_seed_sha256,
        "expected_pair_sha256": expected_pair_sha256,
        "population_acceptance_sha256": _sha_v84a(
            population_acceptance_sha256, "population acceptance"
        ),
        "expected_master_sha256": _sha_v84a(expected_master_sha256, "master"),
        "plan_id": plan_id,
        "update_sequence": _exact_int_v84a(update_sequence, "update sequence", 1),
    }
    return {**body, "content_sha256": canonical_sha256_v84a(body)}


def seal_consensus_v84a(
    local_shards: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
    rank_digest_views: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate pair allgather and a second all-rank digest allgather."""
    if (
        isinstance(local_shards, (str, bytes))
        or not isinstance(local_shards, Sequence)
        or len(local_shards) != WORLD_SIZE_V84A
    ):
        raise ValueError("V84A consensus requires exactly four local shards")
    if not isinstance(plan, Mapping):
        raise TypeError("V84A update plan must be a mapping")
    plan_body = {key: copy.deepcopy(value) for key, value in plan.items() if key != "content_sha256"}
    _require_v84a(
        plan.get("content_sha256") == canonical_sha256_v84a(plan_body)
        and plan.get("schema") == "scalar-replay-update-plan-v84a"
        and plan.get("world_size") == WORLD_SIZE_V84A
        and plan.get("directions") == DIRECTIONS_V84A,
        "V84A update plan identity changed",
    )
    validated: list[dict[str, Any]] = []
    for raw in local_shards:
        if not isinstance(raw, Mapping):
            raise TypeError("V84A local shard changed")
        shard = build_local_shard_v84a(raw.get("rank"), [
            {"seed": row.get("seed"), "coefficient": row.get("coefficient")}
            for row in raw.get("pairs", [])
        ])
        _require_v84a(dict(raw) == shard, "V84A local shard identity changed")
        validated.append(shard)
    validated.sort(key=lambda row: row["rank"])
    _require_v84a(
        [row["rank"] for row in validated] == list(range(WORLD_SIZE_V84A)),
        "V84A consensus rank coverage changed",
    )
    gathered = [
        {**pair, "origin_rank": shard["rank"]}
        for shard in validated
        for pair in shard["pairs"]
    ]
    gathered.sort(key=lambda row: row["seed"])
    seeds = [row["seed"] for row in gathered]
    _require_v84a(
        len(gathered) == DIRECTIONS_V84A
        and len(set(seeds)) == DIRECTIONS_V84A,
        "V84A gathered direction coverage has a duplicate or missing seed",
    )
    _require_v84a(
        canonical_sha256_v84a(seeds) == plan["expected_seed_sha256"],
        "V84A gathered seed inventory is missing or unexpected",
    )
    canonical_pairs = [
        {"seed": row["seed"], "coefficient": row["coefficient"]}
        for row in gathered
    ]
    _require_v84a(
        pair_identity_v84a(canonical_pairs) == plan["expected_pair_sha256"],
        "V84A gathered coefficient identity changed",
    )
    proposal_body = {
        "schema": "scalar-consensus-proposal-v84a",
        "plan_sha256": plan["content_sha256"],
        "ordered_pairs": gathered,
        "pair_wire_bytes": DIRECTIONS_V84A * PAIR_BYTES_V84A,
        "rank_shard_sha256": [row["content_sha256"] for row in validated],
    }
    proposal_sha256 = canonical_sha256_v84a(proposal_body)
    if rank_digest_views is None:
        rank_digest_views = [
            {"rank": rank, "proposal_sha256": proposal_sha256}
            for rank in range(WORLD_SIZE_V84A)
        ]
    if (
        isinstance(rank_digest_views, (str, bytes))
        or not isinstance(rank_digest_views, Sequence)
        or len(rank_digest_views) != WORLD_SIZE_V84A
    ):
        raise ValueError("V84A digest consensus requires four rank views")
    views = []
    for row in rank_digest_views:
        _require_v84a(
            isinstance(row, Mapping)
            and set(row) == {"rank", "proposal_sha256"},
            "V84A digest view fields changed",
        )
        rank = _exact_int_v84a(row["rank"], "digest rank")
        views.append({"rank": rank, "proposal_sha256": _sha_v84a(
            row["proposal_sha256"], "proposal digest"
        )})
    views.sort(key=lambda row: row["rank"])
    _require_v84a(
        [row["rank"] for row in views] == list(range(WORLD_SIZE_V84A))
        and all(row["proposal_sha256"] == proposal_sha256 for row in views),
        "V84A all-rank digest consensus failed",
    )
    body = {
        "schema": CONSENSUS_SCHEMA_V84A,
        "plan": copy.deepcopy(dict(plan)),
        "proposal": proposal_body,
        "proposal_sha256": proposal_sha256,
        "rank_digest_views": views,
        "all_rank_consensus": True,
        "duplicates_missing_nonfinite_rejected": True,
        "canonical_order": "ascending_uint64_seed",
        "pair_allgather_wire_bytes": DIRECTIONS_V84A * PAIR_BYTES_V84A,
        "digest_allgather_wire_bytes": WORLD_SIZE_V84A * DIGEST_BYTES_V84A,
        "live_collective_executed": False,
    }
    return {**body, "content_sha256": canonical_sha256_v84a(body)}


def validate_consensus_v84a(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("V84A consensus must be a mapping")
    plan = value.get("plan")
    proposal = value.get("proposal")
    _require_v84a(
        isinstance(plan, Mapping) and isinstance(proposal, Mapping),
        "V84A consensus body is absent",
    )
    shards = []
    pairs_by_rank: dict[int, list[dict[str, Any]]] = {
        rank: [] for rank in range(WORLD_SIZE_V84A)
    }
    for row in proposal.get("ordered_pairs", []):
        if not isinstance(row, Mapping):
            raise RuntimeError("V84A consensus pair changed")
        pairs_by_rank[row.get("origin_rank")].append(
            {"seed": row.get("seed"), "coefficient": row.get("coefficient")}
        )
    for rank in range(WORLD_SIZE_V84A):
        shards.append(build_local_shard_v84a(rank, pairs_by_rank[rank]))
    expected = seal_consensus_v84a(shards, plan, value.get("rank_digest_views"))
    _require_v84a(dict(value) == expected, "V84A consensus identity changed")
    return copy.deepcopy(expected)


def _source_records_v84a(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("V84A source manifest must be a sequence")
    fields = {"dtype", "elements", "key", "module", "ordinal", "role", "shape"}
    result = []
    for ordinal, row in enumerate(records):
        _require_v84a(
            isinstance(row, Mapping) and set(row) == fields,
            f"V84A source record fields changed at {ordinal}",
        )
        shape = row["shape"]
        _require_v84a(
            row["ordinal"] == ordinal
            and row["dtype"] == "float32"
            and isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) and item > 0 for item in shape)
            and math.prod(shape) == row["elements"],
            f"V84A source record changed at {ordinal}",
        )
        result.append(copy.deepcopy(dict(row)))
    keys = [row["key"] for row in result]
    _require_v84a(
        len(result) == TENSOR_COUNT_V84A
        and keys == sorted(keys)
        and len(set(keys)) == TENSOR_COUNT_V84A
        and len({row["module"] for row in result}) == MODULE_COUNT_V84A
        and sum(row["elements"] for row in result) == TOTAL_ELEMENTS_V84A
        and max(row["elements"] for row in result) == MAX_TENSOR_ELEMENTS_V84A
        and canonical_sha256_v84a(result) == EXPECTED_SOURCE_MANIFEST_SHA256_V84A
        and canonical_sha256_v84a(keys) == EXPECTED_ORDERED_KEY_SHA256_V84A,
        "V84A sealed LoRA source surface changed",
    )
    return result


def byte_rng_accounting_v84a(
    records: Sequence[Mapping[str, Any]],
    *,
    chunk_elements: int = DEFAULT_CHUNK_ELEMENTS_V84A,
) -> dict[str, Any]:
    source = _source_records_v84a(records)
    chunk_elements = _exact_int_v84a(chunk_elements, "chunk elements", 1)
    maximum_chunk = min(chunk_elements, MAX_TENSOR_ELEMENTS_V84A)
    ring_numerator = 2 * (WORLD_SIZE_V84A - 1)
    ring_denominator = WORLD_SIZE_V84A
    native_ring = TOTAL_FP32_BYTES_V84A * ring_numerator // ring_denominator
    pair_wire = DIRECTIONS_V84A * PAIR_BYTES_V84A
    digest_wire = WORLD_SIZE_V84A * DIGEST_BYTES_V84A
    pair_allgather = pair_wire * (WORLD_SIZE_V84A - 1) // WORLD_SIZE_V84A
    digest_allgather = digest_wire * (WORLD_SIZE_V84A - 1) // WORLD_SIZE_V84A
    scalar_network = pair_allgather + digest_allgather

    sum_dimensions = sum(sum(row["shape"]) for row in source)
    max_dimensions = max(sum(row["shape"]) for row in source)
    methods: dict[str, Any] = {}
    for method, rank in [("iid_absolute_index", None)] + [
        ("structured_outer_product", rank) for rank in oracle.STRUCTURED_RANKS_V1
    ]:
        unique_per_direction = (
            TOTAL_ELEMENTS_V84A if rank is None else sum_dimensions * rank
        )
        factor_cache = 0 if rank is None else max_dimensions * rank * FP32_BYTES_V84A
        native_rng = unique_per_direction * DIRECTIONS_PER_RANK_V84A
        replay_rng = unique_per_direction * DIRECTIONS_V84A
        structured_multiply = 0 if rank is None else TOTAL_ELEMENTS_V84A * rank
        structured_add = 0 if rank is None else TOTAL_ELEMENTS_V84A * rank
        structured_scale = 0 if rank is None else TOTAL_ELEMENTS_V84A
        methods[f"{method}:rank={rank}"] = {
            "method": method,
            "structured_rank": rank,
            "unique_rng_normals_per_direction": unique_per_direction,
            "native_balanced_rng_normals_per_actor": native_rng,
            "replay_rng_normals_per_actor": replay_rng,
            "native_rng_normals_all_actors": native_rng * WORLD_SIZE_V84A,
            "replay_rng_normals_all_actors": replay_rng * WORLD_SIZE_V84A,
            "replay_rng_multiplier_per_actor_and_all_actors": (
                replay_rng / native_rng
            ),
            "structured_noise_multiplies_per_direction": structured_multiply,
            "structured_noise_accumulating_adds_per_direction": structured_add,
            "structured_noise_final_scales_per_direction": structured_scale,
            "maximum_factor_cache_bytes": factor_cache,
            # Exact CPU parity retains one rank-local accumulator, one
            # rank-reduced accumulator, one generated-noise chunk, and one
            # candidate chunk.  The two accumulator levels are required
            # because globally sorted FP32 summation can exceed V72's 2-ULP
            # final-update gate.
            "maximum_streamed_update_scratch_bytes": (
                factor_cache + maximum_chunk * 16
            ),
        }

    # Explicit tensor traffic from V72's source operations, excluding RNG
    # implementation internals and collective internals.  For each direction:
    # write noise + read noise + read accumulator + write accumulator = 16E.
    # Common work is accumulator initialization (4E), scaling R+W (8E), and
    # D2H source read (4E), also 16E.
    native_explicit_hbm = (
        16 * TOTAL_ELEMENTS_V84A
        * (1 + DIRECTIONS_PER_RANK_V84A)
    )
    # Exact-order replay reconstructs each rank-local accumulator and then
    # combines those four accumulators in ascending rank order.  Besides the
    # 16E traffic per direction, that adds 4E to initialize each local bank,
    # 12E to fold each local bank into the reduced bank, and the common 16E.
    replay_explicit_hbm = 16 * TOTAL_ELEMENTS_V84A * (
        DIRECTIONS_V84A + WORLD_SIZE_V84A + 1
    )
    fused_replay_accumulator_hbm = 8 * TOTAL_ELEMENTS_V84A * (
        DIRECTIONS_V84A + 2 * WORLD_SIZE_V84A + 2
    )
    fused_output_floor = 8 * TOTAL_ELEMENTS_V84A
    body = {
        "schema": ACCOUNTING_SCHEMA_V84A,
        "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256_V84A,
        "world_size": WORLD_SIZE_V84A,
        "directions": DIRECTIONS_V84A,
        "signed_candidates": SIGNED_CANDIDATES_V84A,
        "tensor_count": TENSOR_COUNT_V84A,
        "module_count": MODULE_COUNT_V84A,
        "total_elements": TOTAL_ELEMENTS_V84A,
        "total_fp32_bytes": TOTAL_FP32_BYTES_V84A,
        "network_projection": {
            "native_collective": "70_tensor_fp32_all_reduce",
            "native_collective_calls_per_actor": NATIVE_COLLECTIVE_CALLS_V84A,
            "native_nominal_ring_bus_bytes_per_actor": native_ring,
            "native_nominal_ring_bus_bytes_all_actors": native_ring * WORLD_SIZE_V84A,
            "scalar_pair_collective": "fixed_size_allgather",
            "scalar_digest_collective": "four_digest_allgather_consensus",
            "scalar_collective_calls_per_actor": 2,
            "pair_wire_bytes_global": pair_wire,
            "digest_wire_bytes_global": digest_wire,
            "pair_allgather_bus_bytes_per_actor": pair_allgather,
            "digest_allgather_bus_bytes_per_actor": digest_allgather,
            "scalar_total_bus_bytes_per_actor": scalar_network,
            "scalar_total_bus_bytes_all_actors": scalar_network * WORLD_SIZE_V84A,
            "nominal_bus_bytes_saved_per_actor": native_ring - scalar_network,
            "native_to_scalar_nominal_bus_ratio": native_ring / scalar_network,
            "ring_and_allgather_are_projections_not_measured_pynccl_behavior": True,
        },
        "rng_and_canonical_work": methods,
        "scratch_and_residency": {
            "pair_gather_buffer_bytes_per_actor": pair_wire,
            "digest_gather_buffer_bytes_per_actor": digest_wire,
            "conservative_simultaneously_retained_consensus_bytes": pair_wire + digest_wire,
            "pending_master_transaction_output_bytes": TOTAL_FP32_BYTES_V84A,
            "rollback_master_aliases_original": True,
            "persistent_replay_gpu_state_bytes": 0,
            "whole_surface_noise_elements_allocated": 0,
            "whole_surface_update_elements_allocated": 0,
            "streamed_chunk_elements": maximum_chunk,
        },
        "explicit_hbm_projection_excluding_rng_and_collective_internals": {
            "native_source_equivalent_bytes_per_actor": native_explicit_hbm,
            "replay_source_equivalent_bytes_per_actor": replay_explicit_hbm,
            "replay_incremental_bytes_per_actor": replay_explicit_hbm - native_explicit_hbm,
            "replay_source_equivalent_multiplier": replay_explicit_hbm / native_explicit_hbm,
            "fused_no_dense_noise_accumulator_bytes_per_actor": fused_replay_accumulator_hbm,
            "ideal_fused_final_update_write_plus_d2h_read_floor_bytes_per_actor": fused_output_floor,
            "native_source_equivalent_formula": "16*E*(1+directions/world_size)",
            "exact_order_replay_source_equivalent_formula": "16*E*(directions+world_size+1)",
            "fused_exact_order_accumulator_formula": "8*E*(directions+2*world_size+2)",
            "exact_order_reason": (
                "globally_sorted_FP32_accumulation_can_exceed_the_registered_2_ULP_gate;_"
                "retain_origin_rank_then_seed_arithmetic_order"
            ),
            "not_a_live_hbm_measurement": True,
        },
        "antithetic_algebra": {
            "signed_form": "r_plus*epsilon + r_minus*(-epsilon)",
            "collapsed_form": "(r_plus-r_minus)*epsilon",
            "coefficient": "r_plus-r_minus",
            "update": "step_size/(2*directions*sigma)*sum(coefficient_i*epsilon_i)",
            "exchanged_direction_pairs": DIRECTIONS_V84A,
            "signed_candidates_not_exchanged": SIGNED_CANDIDATES_V84A,
        },
        "authority": {
            "synthetic_cpu_only": True,
            "gpu_or_model_opened": False,
            "dataset_or_evaluation_opened": False,
            "live_scalar_arm_authorized": False,
            "quality_or_speed_claim": False,
            "promotion_authorized": False,
        },
    }
    return {**body, "content_sha256": canonical_sha256_v84a(body)}


def _tensor_sha256_v84a(tensor: torch.Tensor) -> str:
    raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def mapping_identity_v84a(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    if not isinstance(tensors, Mapping) or not tensors:
        raise TypeError("V84A tensor mapping must be nonempty")
    rows = []
    for key in sorted(tensors):
        tensor = tensors[key]
        _require_v84a(
            isinstance(key, str)
            and bool(key)
            and isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and tensor.ndim == 2
            and tensor.is_contiguous()
            and bool(torch.isfinite(tensor).all()),
            f"V84A tensor mapping changed at {key}",
        )
        rows.append({
            "key": key,
            "shape": list(tensor.shape),
            "elements": int(tensor.numel()),
            "dtype": "torch.float32",
            "sha256": _tensor_sha256_v84a(tensor),
        })
    body = {
        "tensor_count": len(rows),
        "elements": sum(row["elements"] for row in rows),
        "bytes": sum(row["elements"] * FP32_BYTES_V84A for row in rows),
        "tensors": rows,
    }
    return {**body, "sha256": canonical_sha256_v84a(body)}


def _noise_chunk_v84a(
    method: str,
    pair: Mapping[str, Any],
    key: str,
    shape: Sequence[int],
    start: int,
    count: int,
    structured_rank: int | None,
) -> torch.Tensor:
    return oracle.noise_chunk_v1(
        method,
        pair["seed"],
        key,
        shape,
        start,
        count,
        structured_rank,
    )


def native_dense_reference_v84a(
    master: Mapping[str, torch.Tensor],
    local_shards: Sequence[Mapping[str, Any]],
    plan: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Synthetic rank-ordered dense all-reduce reference."""
    identity = mapping_identity_v84a(master)
    del identity
    consensus = seal_consensus_v84a(local_shards, plan)
    scale = plan["step_size"] / (
        2.0 * DIRECTIONS_V84A * plan["sigma"]
    )
    result = {}
    ordered_shards = sorted(local_shards, key=lambda row: row["rank"])
    for key in sorted(master):
        rank_accumulators = []
        for shard in ordered_shards:
            accumulator = torch.zeros_like(master[key])
            for pair in shard["pairs"]:
                noise = _noise_chunk_v84a(
                    plan["method"], pair, key, list(master[key].shape),
                    0, master[key].numel(), plan["structured_rank"],
                ).view_as(master[key])
                accumulator.add_(noise, alpha=pair["coefficient"])
            rank_accumulators.append(accumulator)
        reduced = torch.zeros_like(master[key])
        for accumulator in rank_accumulators:
            reduced.add_(accumulator)
        reduced.mul_(scale)
        result[key] = master[key].add(reduced).contiguous()
    _require_v84a(consensus["all_rank_consensus"] is True, "V84A consensus failed")
    return result


class ScalarReplayTransactionV84A:
    """One-shot synthetic CPU replay with exact restore-or-poison boundaries."""

    def __init__(
        self,
        master: Mapping[str, torch.Tensor],
        consensus: Mapping[str, Any],
        rank: int,
        *,
        authority: str,
        chunk_elements: int = DEFAULT_CHUNK_ELEMENTS_V84A,
    ) -> None:
        if authority != SYNTHETIC_AUTHORITY_V84A:
            raise RuntimeError("V84A live authority is absent")
        self.consensus = validate_consensus_v84a(consensus)
        self.rank = _exact_int_v84a(rank, "transaction rank")
        if self.rank >= WORLD_SIZE_V84A:
            raise ValueError("V84A transaction rank is outside world")
        self.chunk_elements = _exact_int_v84a(chunk_elements, "chunk elements", 1)
        self.original_master = {key: tensor for key, tensor in master.items()}
        self.original_identity = mapping_identity_v84a(self.original_master)
        self.current_master = self.original_master
        self.pending_master: dict[str, torch.Tensor] | None = None
        self.phase = "ready"
        self.poison_reason: str | None = None
        transaction_body = {
            "consensus_sha256": self.consensus["content_sha256"],
            "original_master_sha256": self.original_identity["sha256"],
            "rank": self.rank,
            "chunk_elements": self.chunk_elements,
            "rng_algorithm": oracle.RNG_ALGORITHM_V1,
        }
        self.transaction_sha256 = canonical_sha256_v84a(transaction_body)

    def _original_exact_v84a(self, boundary: str) -> dict[str, Any]:
        identity = mapping_identity_v84a(self.original_master)
        _require_v84a(
            identity == self.original_identity,
            f"V84A original master changed at {boundary}",
        )
        return identity

    def execute_v84a(
        self,
        expected_transaction_sha256: str,
        *,
        chunk_hook: Callable[[str, int, int], None] | None = None,
    ) -> dict[str, Any]:
        if self.phase != "ready":
            raise RuntimeError("V84A stale replay/retry rejected")
        _require_v84a(
            _sha_v84a(expected_transaction_sha256, "transaction")
            == self.transaction_sha256,
            "V84A resume/retry transaction identity changed",
        )
        if chunk_hook is not None and not callable(chunk_hook):
            raise TypeError("V84A chunk hook must be callable")
        self._original_exact_v84a("execute_preflight")
        self.phase = "executing"
        plan = self.consensus["plan"]
        pair_rows = self.consensus["proposal"]["ordered_pairs"]
        pairs = [
            {"seed": row["seed"], "coefficient": row["coefficient"]}
            for row in pair_rows
        ]
        scale = plan["step_size"] / (
            2.0 * DIRECTIONS_V84A * plan["sigma"]
        )
        pending: dict[str, torch.Tensor] = {}
        ranges = []
        maximum_scratch = 0
        try:
            for key in sorted(self.original_master):
                master = self.original_master[key]
                oracle.noise_identity_v1(
                    plan["method"], pairs[0]["seed"], key,
                    list(master.shape), plan["structured_rank"],
                )
                candidate = torch.empty_like(master).view(-1)
                flat = master.view(-1)
                for start in range(0, master.numel(), self.chunk_elements):
                    end = min(master.numel(), start + self.chunk_elements)
                    if chunk_hook is not None:
                        chunk_hook(key, start, end)
                    # Reproduce the fake dense collective's rank-local
                    # accumulation followed by ascending-rank reduction.
                    # Merely summing globally sorted pairs is mathematically
                    # equal but is not reliably within two FP32 ULP.
                    accumulator = torch.zeros(end - start, dtype=torch.float32)
                    for origin_rank in range(WORLD_SIZE_V84A):
                        local = torch.zeros(end - start, dtype=torch.float32)
                        for row in pair_rows:
                            if row["origin_rank"] != origin_rank:
                                continue
                            pair = {
                                "seed": row["seed"],
                                "coefficient": row["coefficient"],
                            }
                            noise = _noise_chunk_v84a(
                                plan["method"], pair, key, list(master.shape),
                                start, end - start, plan["structured_rank"],
                            )
                            local.add_(noise, alpha=pair["coefficient"])
                        accumulator.add_(local)
                    accumulator.mul_(scale)
                    _require_v84a(
                        bool(torch.isfinite(accumulator).all()),
                        "V84A replay update became nonfinite",
                    )
                    candidate[start:end].copy_(
                        flat[start:end].add(accumulator)
                    )
                    factor = 0
                    if plan["structured_rank"] is not None:
                        factor = (
                            sum(master.shape)
                            * plan["structured_rank"]
                            * FP32_BYTES_V84A
                        )
                    maximum_scratch = max(
                        maximum_scratch,
                        factor + (end - start) * 16,
                    )
                    ranges.append({"key": key, "start": start, "end": end})
                pending[key] = candidate.view_as(master).contiguous()
            self._original_exact_v84a("execute_postflight")
            pending_identity = mapping_identity_v84a(pending)
        except BaseException as error:
            pending.clear()
            self.pending_master = None
            self.current_master = self.original_master
            self.phase = "terminal_poison"
            self.poison_reason = f"execute_failure:{type(error).__name__}"
            self._original_exact_v84a("execute_failure_rollback")
            raise
        self.pending_master = pending
        self.phase = "pending_not_committed"
        body = {
            "schema": TRANSACTION_SCHEMA_V84A,
            "phase": self.phase,
            "rank": self.rank,
            "world_size": WORLD_SIZE_V84A,
            "transaction_sha256": self.transaction_sha256,
            "consensus_sha256": self.consensus["content_sha256"],
            "canonical_pair_order": "ascending_uint64_seed",
            "arithmetic_order": "ascending_origin_rank_then_ascending_seed",
            "pending_master_identity": pending_identity,
            "original_master_identity": self.original_identity,
            "rng_algorithm": oracle.RNG_ALGORITHM_V1,
            "maximum_streamed_scratch_bytes": maximum_scratch,
            "range_count": len(ranges),
            "whole_surface_noise_elements_allocated": 0,
            "whole_surface_update_elements_allocated": 0,
            "dense_full_noise_allocated": False,
            "master_committed": False,
            "gpu_model_or_live_collective_executed": False,
        }
        return {**body, "content_sha256": canonical_sha256_v84a(body)}

    def commit_provisional_v84a(self, expected_candidate_sha256: str) -> dict[str, Any]:
        if self.phase != "pending_not_committed" or self.pending_master is None:
            raise RuntimeError("V84A pending replay is not committable")
        identity = mapping_identity_v84a(self.pending_master)
        _require_v84a(
            identity["sha256"] == _sha_v84a(expected_candidate_sha256, "candidate"),
            "V84A candidate identity changed before provisional commit",
        )
        self._original_exact_v84a("provisional_commit")
        self.current_master = self.pending_master
        self.phase = "committed_provisional"
        return {
            "schema": "synthetic-scalar-replay-provisional-commit-v84a",
            "committed": True,
            "candidate_identity": identity,
            "rollback_retained": True,
            "finalized": False,
        }

    def restore_v84a(self, reason: str) -> dict[str, Any]:
        if self.phase not in ("pending_not_committed", "committed_provisional"):
            raise RuntimeError("V84A stale restore rejected")
        if not isinstance(reason, str) or not reason:
            raise ValueError("V84A restore reason is required")
        identity = self._original_exact_v84a("restore")
        self.current_master = self.original_master
        self.pending_master = None
        self.phase = "restored_exact_original"
        return {
            "schema": "synthetic-scalar-replay-restore-v84a",
            "restored": True,
            "reason": reason,
            "original_identity": identity,
            "terminal_poisoned": False,
        }

    def finalize_v84a(self, expected_candidate_sha256: str) -> dict[str, Any]:
        if self.phase != "committed_provisional" or self.pending_master is None:
            raise RuntimeError("V84A provisional replay is not finalizable")
        identity = mapping_identity_v84a(self.pending_master)
        if identity["sha256"] != _sha_v84a(expected_candidate_sha256, "candidate"):
            self.phase = "terminal_poison"
            self.poison_reason = "final_candidate_identity_changed"
            raise RuntimeError("V84A final candidate identity changed; transaction poisoned")
        self.original_master = self.pending_master
        self.original_identity = identity
        self.current_master = self.pending_master
        self.pending_master = None
        self.phase = "finalized"
        return {
            "schema": "synthetic-scalar-replay-finalized-v84a",
            "finalized": True,
            "final_master_identity": identity,
            "rollback_released": True,
            "terminal_poisoned": False,
        }
