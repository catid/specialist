#!/usr/bin/env python3
"""CPU oracle and fail-closed contract helpers for compressed ES collectives.

This module deliberately imports neither torch nor CUDA.  It defines the
bit-level BF16/error-feedback semantics, transaction/resume rules, launch gate,
and live receipt surface that a future four-GPU implementation must match.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
import struct
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_V82 = "eggroll-es-collective-compression-oracle-v82"
STATE_SCHEMA_V82 = "eggroll-es-fp32-collective-residual-state-v82"
TRANSACTION_SCHEMA_V82 = "eggroll-es-collective-residual-transaction-v82"
GATE_SCHEMA_V82 = "qwen36-collective-residual-bottleneck-gate-v82"
LIVE_SCHEMA_V82 = "qwen36-collective-compression-live-evidence-v82"
WORLD_SIZE_V82 = 4
REGISTERED_SEEDS_V82 = (1701, 1702, 1703)
RUNTIME_PARAMETER_ELEMENTS_V82 = (
    (131_072, 25_165_824, 8_388_608, 524_288, 1_048_576, 2_097_152) * 3
    + (18_874_368, 8_388_608, 524_288, 1_048_576, 2_097_152)
)
TOTAL_ELEMENTS_V82 = sum(RUNTIME_PARAMETER_ELEMENTS_V82)
MAX_TENSOR_ELEMENTS_V82 = max(RUNTIME_PARAMETER_ELEMENTS_V82)
FP32_BYTES_PER_ELEMENT_V82 = 4
BF16_BYTES_PER_ELEMENT_V82 = 2
BF16_MIN_SUBNORMAL_V82 = math.ldexp(1.0, -133)


def canonical_sha256_v82(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_v82(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _strict_fields_v82(
    value: Mapping[str, Any], fields: set[str], label: str
) -> None:
    _require_v82(
        isinstance(value, Mapping) and set(value) == fields,
        f"v82 {label} fields changed",
    )


def _exact_int_v82(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"v82 {label} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"v82 {label} must be >= {minimum}")
    return result


def _finite_v82(value: Any, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"v82 {label} must be real")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"v82 {label} must be finite")
    return result


def _sha_v82(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"v82 {label} must be lowercase SHA-256")
    return value


def float32_v82(value: Any) -> float:
    value = _finite_v82(value, "FP32 input")
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except OverflowError as error:
        raise ValueError("v82 FP32 input overflowed") from error
    if not math.isfinite(result):
        raise ValueError("v82 FP32 input became nonfinite")
    return result


def float32_bits_v82(value: Any) -> int:
    return struct.unpack("<I", struct.pack("<f", float32_v82(value)))[0]


def float32_from_bits_v82(bits: Any) -> float:
    bits = _exact_int_v82(bits, "FP32 bits")
    if bits > 0xFFFFFFFF:
        raise ValueError("v82 FP32 bits exceed uint32")
    value = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(value):
        raise ValueError("v82 residual/checkpoint FP32 bits are nonfinite")
    return value


def _bits_hex_v82(value: Any) -> str:
    return f"{float32_bits_v82(value):08x}"


def _from_bits_hex_v82(value: Any) -> float:
    if (
        not isinstance(value, str)
        or len(value) != 8
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("v82 FP32 bit string changed")
    return float32_from_bits_v82(int(value, 16))


def bf16_roundtrip_v82(value: Any) -> dict[str, Any]:
    """Round one finite FP32 value to BF16 with round-to-nearest-even."""
    source = float32_v82(value)
    source_bits = float32_bits_v82(source)
    # For finite IEEE FP32 values, the standard bias-plus-tie-bit transform is
    # RNE when the low 16 bits are discarded.
    bf16_bits = ((source_bits + 0x7FFF + ((source_bits >> 16) & 1)) >> 16) & 0xFFFF
    decoded_bits = bf16_bits << 16
    decoded = struct.unpack("<f", struct.pack("<I", decoded_bits))[0]
    if not math.isfinite(decoded):
        raise ValueError("v82 BF16 rounding overflowed or became nonfinite")
    error = abs(float(source) - float(decoded))
    magnitude = abs(source)
    if magnitude == 0.0 or magnitude < math.ldexp(1.0, -126):
        spacing_bound = BF16_MIN_SUBNORMAL_V82
    else:
        spacing_bound = math.ldexp(1.0, math.floor(math.log2(magnitude)) - 7)
    _require_v82(
        math.isfinite(error) and error <= spacing_bound,
        "v82 BF16 quantization error exceeded one local spacing",
    )
    return {
        "source": source,
        "source_bits": f"{source_bits:08x}",
        "bf16_bits": f"{bf16_bits:04x}",
        "decoded": decoded,
        "decoded_bits": f"{decoded_bits:08x}",
        "absolute_error": error,
        "absolute_error_bound": spacing_bound,
    }


def prepare_bf16_element_v82(update: Any, residual: Any) -> dict[str, Any]:
    update_f32 = float32_v82(update)
    residual_f32 = float32_v82(residual)
    compensated = float32_v82(update_f32 + residual_f32)
    quantized = bf16_roundtrip_v82(compensated)
    next_residual = float32_v82(compensated - quantized["decoded"])
    reconstructed = float32_v82(quantized["decoded"] + next_residual)
    _require_v82(
        float32_bits_v82(reconstructed) == float32_bits_v82(compensated),
        "v82 local error-feedback conservation changed",
    )
    return {
        "update": update_f32,
        "compensated": compensated,
        "transmitted": quantized["decoded"],
        "next_residual": next_residual,
        "absolute_error": quantized["absolute_error"],
        "absolute_error_bound": quantized["absolute_error_bound"],
        "conservation_bit_exact": True,
    }


def prepare_bf16_shard_v82(
    updates: Sequence[Any], residuals: Sequence[Any]
) -> dict[str, Any]:
    if (
        isinstance(updates, (str, bytes))
        or isinstance(residuals, (str, bytes))
        or len(updates) != len(residuals)
        or not updates
    ):
        raise ValueError("v82 BF16 shard coverage changed")
    elements = [
        prepare_bf16_element_v82(update, residual)
        for update, residual in zip(updates, residuals, strict=True)
    ]
    result = {
        "schema": "eggroll-es-bf16-error-feedback-shard-v82",
        "elements": len(elements),
        "transmitted_bits": [_bits_hex_v82(item["transmitted"]) for item in elements],
        "next_residual_bits": [
            _bits_hex_v82(item["next_residual"]) for item in elements
        ],
        "compensated_bits": [
            _bits_hex_v82(item["compensated"]) for item in elements
        ],
        "maximum_absolute_error": max(item["absolute_error"] for item in elements),
        "maximum_absolute_error_bound": max(
            item["absolute_error_bound"] for item in elements
        ),
        "all_finite": True,
        "local_conservation_bit_exact": True,
    }
    result["content_sha256"] = canonical_sha256_v82(result)
    return result


def prepare_fp32_fallback_shard_v82(
    updates: Sequence[Any], residuals: Sequence[Any]
) -> dict[str, Any]:
    """Model the untouched production FP32 path.

    A fallback arm cannot inherit compressed-arm residual state: that would
    silently change the canonical FP32 input.  A failed compressed transaction
    must first roll back to its own checkpoint; the independent FP32 control
    then runs this exact identity path.
    """
    if (
        isinstance(updates, (str, bytes))
        or isinstance(residuals, (str, bytes))
        or len(updates) != len(residuals)
        or not updates
    ):
        raise ValueError("v82 FP32 fallback shard coverage changed")
    values = [float32_v82(value) for value in updates]
    residual_values = [float32_v82(value) for value in residuals]
    if any(float32_bits_v82(value) != 0 for value in residual_values):
        raise RuntimeError("v82 exact FP32 fallback cannot consume residual state")
    result = {
        "schema": "eggroll-es-exact-fp32-collective-shard-v82",
        "elements": len(values),
        "input_bits": [_bits_hex_v82(value) for value in values],
        "transmitted_bits": [_bits_hex_v82(value) for value in values],
        "next_residual_bits": ["00000000"] * len(values),
        "conversion_or_rescaling_used": False,
        "bit_exact_to_input": True,
    }
    result["content_sha256"] = canonical_sha256_v82(result)
    return result


def strided_seed_shards_v82(
    seeds: Sequence[Any], world_size: Any = WORLD_SIZE_V82
) -> tuple[dict[str, Any], ...]:
    world_size = _exact_int_v82(world_size, "world size", minimum=1)
    canonical = [_exact_int_v82(seed, "direction seed") for seed in seeds]
    if not canonical or len(set(canonical)) != len(canonical):
        raise ValueError("v82 direction seed coverage changed")
    result = []
    for rank in range(world_size):
        indices = list(range(rank, len(canonical), world_size))
        result.append({
            "rank": rank,
            "indices": indices,
            "seeds": [canonical[index] for index in indices],
        })
    flattened = sorted(
        index for shard in result for index in shard["indices"]
    )
    _require_v82(
        flattened == list(range(len(canonical))),
        "v82 seed shards overlap or have gaps",
    )
    return tuple(result)


def antithetic_coefficients_v82(
    pairs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(pairs, (str, bytes)) or not pairs:
        raise ValueError("v82 antithetic pair matrix is empty")
    seeds: list[int] = []
    coefficients: list[float] = []
    records = []
    for direction_index, pair in enumerate(pairs):
        _strict_fields_v82(
            pair, {"direction_seed", "reward_plus", "reward_minus"},
            "antithetic pair",
        )
        seed = _exact_int_v82(pair["direction_seed"], "direction seed")
        plus = _finite_v82(pair["reward_plus"], "plus reward")
        minus = _finite_v82(pair["reward_minus"], "minus reward")
        difference = plus - minus
        _require_v82(math.isfinite(difference), "v82 pair difference is nonfinite")
        seeds.append(seed)
        coefficients.append(difference)
        records.append({
            "direction_index": direction_index,
            "direction_seed": seed,
            "reward_plus": plus,
            "reward_minus": minus,
            "pair_difference": difference,
        })
    if len(set(seeds)) != len(seeds):
        raise ValueError("v82 direction seeds are not unique")
    return {
        "schema": "eggroll-es-antithetic-pair-coefficients-v82",
        "estimator": "learning_rate/(2*N*sigma)*sum((Rplus-Rminus)*epsilon)",
        "direction_seeds": seeds,
        "coefficients": coefficients,
        "pairs": records,
        "unpaired_reward_centering_used": False,
        "coefficient_sha256": canonical_sha256_v82({
            "seeds": seeds, "coefficients": coefficients
        }),
    }


def deterministic_rank_sum_v82(
    rank_vectors: Mapping[Any, Sequence[Any]],
) -> tuple[float, ...]:
    if not isinstance(rank_vectors, Mapping) or set(rank_vectors) != set(
        range(WORLD_SIZE_V82)
    ):
        raise ValueError("v82 deterministic rank-sum coverage changed")
    lengths = {len(rank_vectors[rank]) for rank in range(WORLD_SIZE_V82)}
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("v82 deterministic rank-sum shape changed")
    result = []
    for index in range(next(iter(lengths))):
        accumulator = float32_v82(0.0)
        for rank in range(WORLD_SIZE_V82):
            accumulator = float32_v82(
                accumulator + float32_v82(rank_vectors[rank][index])
            )
        result.append(accumulator)
    return tuple(result)


def _state_body_v82(
    version: int, shard_names: Sequence[str], residual_bits: Sequence[Sequence[str]]
) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA_V82,
        "version": version,
        "shard_order": list(shard_names),
        "residual_bits_by_shard": [list(values) for values in residual_bits],
        "dtype": "float32",
        "canonical_master_or_optimizer_stored_here": False,
    }


def validate_residual_state_v82(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict_fields_v82(
        value,
        {
            "schema", "version", "shard_order", "residual_bits_by_shard",
            "dtype", "canonical_master_or_optimizer_stored_here",
            "content_sha256",
        },
        "residual state",
    )
    _require_v82(value["schema"] == STATE_SCHEMA_V82, "v82 residual schema changed")
    version = _exact_int_v82(value["version"], "residual version")
    names = value["shard_order"]
    residuals = value["residual_bits_by_shard"]
    _require_v82(
        isinstance(names, list)
        and names
        and all(isinstance(name, str) and name for name in names)
        and len(set(names)) == len(names)
        and isinstance(residuals, list)
        and len(residuals) == len(names)
        and value["dtype"] == "float32"
        and value["canonical_master_or_optimizer_stored_here"] is False,
        "v82 residual state metadata changed",
    )
    for shard in residuals:
        _require_v82(isinstance(shard, list) and shard, "v82 empty residual shard")
        for item in shard:
            _from_bits_hex_v82(item)
    body = _state_body_v82(version, names, residuals)
    _require_v82(
        value["content_sha256"] == canonical_sha256_v82(body),
        "v82 residual state identity changed",
    )
    return copy.deepcopy(dict(value))


def new_residual_state_v82(
    shard_lengths: Mapping[str, Any], version: Any = 0
) -> dict[str, Any]:
    if not isinstance(shard_lengths, Mapping) or not shard_lengths:
        raise ValueError("v82 residual shard layout is empty")
    names = list(shard_lengths)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("v82 residual shard name changed")
    lengths = [
        _exact_int_v82(shard_lengths[name], f"{name} length", minimum=1)
        for name in names
    ]
    version = _exact_int_v82(version, "residual version")
    body = _state_body_v82(
        version, names, [["00000000"] * length for length in lengths]
    )
    return {**body, "content_sha256": canonical_sha256_v82(body)}


def prepare_residual_transaction_v82(
    state: Mapping[str, Any],
    updates_by_shard: Mapping[str, Sequence[Any]],
    transaction_id: str,
) -> dict[str, Any]:
    before = validate_residual_state_v82(state)
    if not isinstance(transaction_id, str) or not transaction_id:
        raise ValueError("v82 transaction id is empty")
    names = before["shard_order"]
    if not isinstance(updates_by_shard, Mapping) or list(updates_by_shard) != names:
        raise RuntimeError("v82 update shard order changed")
    receipts = []
    next_bits = []
    transmitted_bits = []
    for name, residual_bits in zip(
        names, before["residual_bits_by_shard"], strict=True
    ):
        updates = updates_by_shard[name]
        residuals = [_from_bits_hex_v82(value) for value in residual_bits]
        receipt = prepare_bf16_shard_v82(updates, residuals)
        receipts.append({"shard": name, "receipt": receipt})
        next_bits.append(receipt["next_residual_bits"])
        transmitted_bits.append(receipt["transmitted_bits"])
    candidate_body = _state_body_v82(
        before["version"] + 1, names, next_bits
    )
    candidate = {
        **candidate_body,
        "content_sha256": canonical_sha256_v82(candidate_body),
    }
    body = {
        "schema": TRANSACTION_SCHEMA_V82,
        "transaction_id": transaction_id,
        "before_state": before,
        "candidate_state": candidate,
        "transmitted_bits_by_shard": transmitted_bits,
        "shard_receipts": receipts,
        "phase": "prepared_not_published",
        "canonical_master_committed": False,
    }
    return {**body, "content_sha256": canonical_sha256_v82(body)}


def validate_residual_transaction_v82(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict_fields_v82(
        value,
        {
            "schema", "transaction_id", "before_state", "candidate_state",
            "transmitted_bits_by_shard", "shard_receipts", "phase",
            "canonical_master_committed", "content_sha256",
        },
        "residual transaction",
    )
    _require_v82(
        value["schema"] == TRANSACTION_SCHEMA_V82
        and isinstance(value["transaction_id"], str)
        and bool(value["transaction_id"])
        and value["phase"] == "prepared_not_published"
        and value["canonical_master_committed"] is False,
        "v82 residual transaction metadata changed",
    )
    before = validate_residual_state_v82(value["before_state"])
    candidate = validate_residual_state_v82(value["candidate_state"])
    _require_v82(
        candidate["version"] == before["version"] + 1
        and candidate["shard_order"] == before["shard_order"],
        "v82 residual transaction generation changed",
    )
    body = {key: copy.deepcopy(item) for key, item in value.items() if key != "content_sha256"}
    _require_v82(
        value["content_sha256"] == canonical_sha256_v82(body),
        "v82 residual transaction identity changed",
    )
    return copy.deepcopy(dict(value))


def commit_residual_transaction_v82(
    current: Mapping[str, Any], transaction: Mapping[str, Any]
) -> dict[str, Any]:
    state = validate_residual_state_v82(current)
    pending = validate_residual_transaction_v82(transaction)
    _require_v82(
        state["content_sha256"]
        == pending["before_state"]["content_sha256"],
        "v82 residual commit started from the wrong generation",
    )
    return copy.deepcopy(pending["candidate_state"])


def rollback_residual_transaction_v82(
    current: Mapping[str, Any], transaction: Mapping[str, Any]
) -> dict[str, Any]:
    state = validate_residual_state_v82(current)
    pending = validate_residual_transaction_v82(transaction)
    allowed = {
        pending["before_state"]["content_sha256"],
        pending["candidate_state"]["content_sha256"],
    }
    _require_v82(
        state["content_sha256"] in allowed,
        "v82 residual rollback observed an unknown generation",
    )
    return copy.deepcopy(pending["before_state"])


def serialize_residual_checkpoint_v82(state: Mapping[str, Any]) -> bytes:
    value = validate_residual_state_v82(state)
    return (
        json.dumps(
            value, allow_nan=False, ensure_ascii=True, separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def resume_residual_checkpoint_v82(payload: bytes) -> dict[str, Any]:
    if not isinstance(payload, bytes):
        raise TypeError("v82 residual checkpoint must be bytes")

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError("v82 residual checkpoint has duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("ascii"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError("v82 residual checkpoint is invalid") from error
    return validate_residual_state_v82(value)


def collective_byte_accounting_v82() -> dict[str, Any]:
    fp32_payload = TOTAL_ELEMENTS_V82 * FP32_BYTES_PER_ELEMENT_V82
    bf16_payload = TOTAL_ELEMENTS_V82 * BF16_BYTES_PER_ELEMENT_V82
    ring_factor_numerator = 2 * (WORLD_SIZE_V82 - 1)
    ring_factor_denominator = WORLD_SIZE_V82
    fp32_bus = fp32_payload * ring_factor_numerator // ring_factor_denominator
    bf16_bus = bf16_payload * ring_factor_numerator // ring_factor_denominator
    residual_bank = fp32_payload
    bf16_staging = MAX_TENSOR_ELEMENTS_V82 * BF16_BYTES_PER_ELEMENT_V82
    # Fused local prepare reads update+old residual and writes next residual+BF16.
    fused_prepare_hbm = fp32_payload * 3 + bf16_payload
    return {
        "schema": "qwen36-collective-compression-byte-accounting-v82",
        "world_size": WORLD_SIZE_V82,
        "native_tensor_count": len(RUNTIME_PARAMETER_ELEMENTS_V82),
        "total_elements_per_rank": TOTAL_ELEMENTS_V82,
        "maximum_tensor_elements": MAX_TENSOR_ELEMENTS_V82,
        "flat_shadow_bytes_per_rank": fp32_payload,
        "flat_shadow_forbidden": True,
        "fp32_control": {
            "collective_payload_bytes_per_rank": fp32_payload,
            "nominal_ring_bus_bytes_per_rank": fp32_bus,
            "residual_bytes_per_rank": 0,
            "incremental_staging_bytes_per_rank": 0,
        },
        "bf16_error_feedback": {
            "collective_payload_bytes_per_rank": bf16_payload,
            "nominal_ring_bus_bytes_per_rank": bf16_bus,
            "nominal_collective_payload_fraction_of_fp32": 0.5,
            "steady_residual_bytes_per_rank": residual_bank,
            "transaction_peak_residual_bytes_per_rank": 2 * residual_bank,
            "maximum_native_tensor_bf16_staging_bytes_per_rank": bf16_staging,
            "incremental_transaction_peak_bytes_per_rank": (
                2 * residual_bank + bf16_staging
            ),
            "fused_prepare_hbm_bytes_per_rank_per_update_lower_bound": (
                fused_prepare_hbm
            ),
            "naive_full_fp32_decode_staging_forbidden": True,
            "post_collective_bf16_is_converted_and_scaled_on_cpu_fp32": True,
        },
        "nominal_ring_bus_bytes_saved_per_rank": fp32_bus - bf16_bus,
        "nominal_ring_bus_bytes_saved_all_ranks": (
            (fp32_bus - bf16_bus) * WORLD_SIZE_V82
        ),
        "link_bytes_are_projection_not_measurement": True,
    }


def validate_bottleneck_gate_v82(receipt: Mapping[str, Any]) -> dict[str, Any]:
    _strict_fields_v82(
        receipt,
        {
            "schema", "dependency_beads", "post_optimization_profile",
            "decision", "content_sha256",
        },
        "bottleneck gate",
    )
    _require_v82(receipt["schema"] == GATE_SCHEMA_V82, "v82 gate schema changed")
    dependencies = receipt["dependency_beads"]
    expected_dependencies = {
        "specialist-0j5.14", "specialist-0j5.18", "specialist-0j5.19",
        "specialist-0j5.32",
    }
    _require_v82(
        isinstance(dependencies, Mapping)
        and set(dependencies) == expected_dependencies,
        "v82 dependency gate coverage changed",
    )
    for bead, evidence in dependencies.items():
        _strict_fields_v82(
            evidence, {"status", "acceptance_passed", "evidence_sha256"},
            f"dependency {bead}",
        )
        _require_v82(
            evidence["status"] == "closed"
            and evidence["acceptance_passed"] is True,
            f"v82 dependency {bead} is incomplete",
        )
        _sha_v82(evidence["evidence_sha256"], f"{bead} evidence")
    profile = receipt["post_optimization_profile"]
    _strict_fields_v82(
        profile,
        {
            "profile_sha256", "recipe_layout_sha256", "measured_after_beads",
            "ranked_residual_bottlenecks", "collective_link_bytes_measured",
            "collective_time_measured", "hbm_bytes_measured",
            "all_four_gpus_attributed", "protected_content_opened",
        },
        "post-optimization profile",
    )
    _sha_v82(profile["profile_sha256"], "profile")
    _sha_v82(profile["recipe_layout_sha256"], "recipe layout")
    _require_v82(
        profile["measured_after_beads"]
        == ["specialist-0j5.18", "specialist-0j5.19"]
        and profile["collective_link_bytes_measured"] is True
        and profile["collective_time_measured"] is True
        and profile["hbm_bytes_measured"] is True
        and profile["all_four_gpus_attributed"] is True
        and profile["protected_content_opened"] is False,
        "v82 post-optimization measurement gate failed",
    )
    ranked = profile["ranked_residual_bottlenecks"]
    _require_v82(
        isinstance(ranked, list) and len(ranked) >= 3,
        "v82 requires at least three ranked residual bottlenecks",
    )
    identifiers = []
    prior_seconds = math.inf
    for expected_rank, item in enumerate(ranked, start=1):
        _strict_fields_v82(
            item,
            {
                "rank", "id", "unoverlapped_wall_seconds", "link_bytes",
                "hbm_bytes", "measurement_sha256",
            },
            "ranked bottleneck",
        )
        seconds = _finite_v82(
            item["unoverlapped_wall_seconds"], "bottleneck seconds", positive=True
        )
        _require_v82(
            item["rank"] == expected_rank
            and isinstance(item["id"], str)
            and bool(item["id"])
            and seconds <= prior_seconds
            and _exact_int_v82(item["link_bytes"], "link bytes") >= 0
            and _exact_int_v82(item["hbm_bytes"], "HBM bytes") >= 0,
            "v82 residual bottleneck ordering changed",
        )
        _sha_v82(item["measurement_sha256"], "bottleneck measurement")
        identifiers.append(item["id"])
        prior_seconds = seconds
    _require_v82(
        len(set(identifiers)) == len(identifiers)
        and identifiers.count("update_collective") == 1,
        "v82 update collective rank is absent or duplicated",
    )
    collective_rank = identifiers.index("update_collective") + 1
    expected_decision = (
        "launch_fp32_vs_bf16_ablation"
        if collective_rank <= 3
        else "close_not_applicable_without_gpu_ablation"
    )
    _require_v82(
        receipt["decision"] == expected_decision,
        "v82 bottleneck decision changed",
    )
    body = {key: copy.deepcopy(value) for key, value in receipt.items() if key != "content_sha256"}
    _require_v82(
        receipt["content_sha256"] == canonical_sha256_v82(body),
        "v82 bottleneck receipt identity changed",
    )
    return copy.deepcopy(dict(receipt))


def live_launch_authorized_v82(receipt: Mapping[str, Any]) -> bool:
    value = validate_bottleneck_gate_v82(receipt)
    return value["decision"] == "launch_fp32_vs_bf16_ablation"


def validate_live_evidence_v82(
    evidence: Mapping[str, Any], prereg_content_sha256: str
) -> dict[str, Any]:
    _strict_fields_v82(
        evidence,
        {
            "schema", "preregistration_content_sha256", "gate_content_sha256",
            "training_seeds", "arms", "paired_systems_results",
            "quality_results", "residual_results", "integrity",
            "promotion_decision", "content_sha256",
        },
        "live evidence",
    )
    _require_v82(evidence["schema"] == LIVE_SCHEMA_V82, "v82 live schema changed")
    _require_v82(
        evidence["preregistration_content_sha256"]
        == _sha_v82(prereg_content_sha256, "preregistration")
        and evidence["training_seeds"] == list(REGISTERED_SEEDS_V82)
        and evidence["arms"] == ["fp32_control", "bf16_error_feedback"],
        "v82 live identity or arm order changed",
    )
    _sha_v82(evidence["gate_content_sha256"], "gate")
    systems = evidence["paired_systems_results"]
    _require_v82(
        isinstance(systems, list) and len(systems) == len(REGISTERED_SEEDS_V82),
        "v82 paired systems result count changed",
    )
    throughput_ratios = []
    for expected_seed, row in zip(REGISTERED_SEEDS_V82, systems, strict=True):
        _strict_fields_v82(
            row,
            {
                "seed", "counterbalanced_order", "physical_gpus",
                "all_gpus_useful_each_arm", "cleanup_idle_each_arm",
                "fp32_link_bytes", "bf16_link_bytes", "fp32_collective_seconds",
                "bf16_collective_seconds", "fp32_throughput_updates_per_second",
                "bf16_throughput_updates_per_second", "bf16_peak_staging_bytes",
                "bf16_peak_residual_bytes", "hbm_bytes_measured",
            },
            "paired systems row",
        )
        for field in (
            "fp32_link_bytes", "bf16_link_bytes", "bf16_peak_staging_bytes",
            "bf16_peak_residual_bytes",
        ):
            _exact_int_v82(row[field], field, minimum=1)
        for field in (
            "fp32_collective_seconds", "bf16_collective_seconds",
            "fp32_throughput_updates_per_second",
            "bf16_throughput_updates_per_second",
        ):
            _finite_v82(row[field], field, positive=True)
        _require_v82(
            row["seed"] == expected_seed
            and row["counterbalanced_order"] in ("AB", "BA")
            and row["physical_gpus"] == [0, 1, 2, 3]
            and row["all_gpus_useful_each_arm"] is True
            and row["cleanup_idle_each_arm"] is True
            and row["hbm_bytes_measured"] is True,
            "v82 live systems attribution changed",
        )
        accounting = collective_byte_accounting_v82()["bf16_error_feedback"]
        _require_v82(
            row["bf16_peak_staging_bytes"]
            == accounting["maximum_native_tensor_bf16_staging_bytes_per_rank"]
            and row["bf16_peak_residual_bytes"]
            == accounting["transaction_peak_residual_bytes_per_rank"],
            "v82 BF16 component peak accounting changed",
        )
        throughput_ratios.append(
            row["bf16_throughput_updates_per_second"]
            / row["fp32_throughput_updates_per_second"]
        )
    quality = evidence["quality_results"]
    _strict_fields_v82(
        quality,
        {
            "source_disjoint_contract_passed", "protected_holdout_opened",
            "three_seed_dev_paired_lcb", "positive_dev_seeds",
            "ood_qa_reward_lcb", "ood_qa_exact_delta", "ood_prose_logprob_lcb",
            "all_ood_noninferiority_conditions_passed",
        },
        "quality results",
    )
    dev_lcb = _finite_v82(quality["three_seed_dev_paired_lcb"], "dev LCB")
    qa_lcb = _finite_v82(quality["ood_qa_reward_lcb"], "OOD QA LCB")
    prose_lcb = _finite_v82(
        quality["ood_prose_logprob_lcb"], "OOD prose LCB"
    )
    _require_v82(
        quality["source_disjoint_contract_passed"] is True
        and quality["protected_holdout_opened"] is False
        and dev_lcb >= 0.0
        and _exact_int_v82(quality["positive_dev_seeds"], "positive seeds") >= 2
        and qa_lcb >= -0.02
        and _exact_int_v82(quality["ood_qa_exact_delta"], "OOD exact delta", -1)
        >= -1
        and prose_lcb >= -0.02
        and quality["all_ood_noninferiority_conditions_passed"] is True,
        "v82 validation or OOD gate failed",
    )
    residual = evidence["residual_results"]
    _strict_fields_v82(
        residual,
        {
            "all_finite", "local_conservation_bit_exact",
            "resume_replay_bit_exact", "rollback_bit_exact",
            "rank_local_residual_receipts_complete", "maximum_absolute_error",
            "maximum_registered_bound", "canonical_master_dtype",
            "optimizer_dtype", "communication_dtype",
        },
        "residual results",
    )
    maximum_error = _finite_v82(
        residual["maximum_absolute_error"], "maximum residual error"
    )
    maximum_bound = _finite_v82(
        residual["maximum_registered_bound"], "maximum residual bound"
    )
    _require_v82(
        residual["all_finite"] is True
        and residual["local_conservation_bit_exact"] is True
        and residual["resume_replay_bit_exact"] is True
        and residual["rollback_bit_exact"] is True
        and residual["rank_local_residual_receipts_complete"] is True
        and 0.0 <= maximum_error <= maximum_bound
        and residual["canonical_master_dtype"] == "float32"
        and residual["optimizer_dtype"] == "float32"
        and residual["communication_dtype"] == "bfloat16",
        "v82 residual live gate failed",
    )
    integrity = evidence["integrity"]
    _strict_fields_v82(
        integrity,
        {
            "native_23_tensor_order_exact", "flat_shadow_allocated",
            "fp8_collective_attempted", "fp32_fallback_bit_exact",
            "unknown_collective_outcome_restored_or_poisoned",
            "checkpoint_master_residual_atomic", "protocol_or_leak_counter_increased",
        },
        "live integrity",
    )
    _require_v82(
        integrity["native_23_tensor_order_exact"] is True
        and integrity["flat_shadow_allocated"] is False
        and integrity["fp8_collective_attempted"] is False
        and integrity["fp32_fallback_bit_exact"] is True
        and integrity["unknown_collective_outcome_restored_or_poisoned"] is True
        and integrity["checkpoint_master_residual_atomic"] is True
        and integrity["protocol_or_leak_counter_increased"] is False,
        "v82 live integrity gate failed",
    )
    decision = evidence["promotion_decision"]
    _require_v82(
        decision in ("promote_bf16_error_feedback", "retain_fp32_control"),
        "v82 promotion decision changed",
    )
    if decision == "promote_bf16_error_feedback":
        geometric_ratio = math.exp(
            math.fsum(math.log(value) for value in throughput_ratios)
            / len(throughput_ratios)
        )
        _require_v82(
            geometric_ratio > 1.0
            and sum(value > 1.0 for value in throughput_ratios) >= 2
            and sum(
                row["bf16_link_bytes"] < row["fp32_link_bytes"]
                for row in systems
            )
            >= 2,
            "v82 BF16 promotion lacks replicated systems improvement",
        )
    body = {key: copy.deepcopy(value) for key, value in evidence.items() if key != "content_sha256"}
    _require_v82(
        evidence["content_sha256"] == canonical_sha256_v82(body),
        "v82 live evidence identity changed",
    )
    return copy.deepcopy(dict(evidence))
