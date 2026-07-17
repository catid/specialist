#!/usr/bin/env python3
"""CPU-only LoRA-scope oracle for the V82B collective correction.

V82 accidentally modeled the 23-tensor, 142,999,552-element selected base
weight surface.  The production LoRA update instead reduces the canonical
70-tensor, 4,528,128-element FP32 master one tensor at a time.  This module
contains no torch/CUDA import and defines only shape, byte, and future
materiality-reconsideration contracts for that real update path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA_V82B = "eggroll-es-lora-collective-scope-oracle-v82b"
PROFILE_SCHEMA_V82B = "qwen36-lora-collective-materiality-profile-v82b"
WORLD_SIZE_V82B = 4
TENSOR_COUNT_V82B = 70
MODULE_COUNT_V82B = 35
TOTAL_ELEMENTS_V82B = 4_528_128
MAX_TENSOR_ELEMENTS_V82B = 262_144
FP32_BYTES_V82B = 4
BF16_BYTES_V82B = 2
EXPECTED_SURFACE_SHA256_V82B = (
    "6c4c219f92fba3d7d01e08f439b7b1f21a1d07bc9893cdd18f860994668e0fb8"
)
EXPECTED_ORDERED_KEY_SHA256_V82B = (
    "ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280"
)
PROFILE_SEEDS_V82B = (1821, 1822, 1823)


def canonical_sha256_v82b(value: Any) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _require_v82b(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _exact_int_v82b(value: Any, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"V82B {label} must be an exact integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"V82B {label} must be >= {minimum}")
    return result


def _finite_v82b(value: Any, label: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"V82B {label} must be real")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V82B {label} must be finite")
    return result


def validate_ordered_shape_manifest_v82b(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and canonicalize the exact ordered PEFT shape surface."""
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError("V82B ordered shape manifest must be a sequence")
    _require_v82b(
        len(records) == TENSOR_COUNT_V82B,
        "V82B LoRA tensor count changed",
    )
    canonical = []
    for ordinal, source in enumerate(records):
        _require_v82b(
            isinstance(source, Mapping)
            and set(source)
            == {"dtype", "elements", "key", "module", "role", "shape"},
            f"V82B shape record fields changed at {ordinal}",
        )
        key = source["key"]
        module = source["module"]
        role = source["role"]
        shape = source["shape"]
        elements = _exact_int_v82b(source["elements"], "tensor elements", 1)
        _require_v82b(
            isinstance(key, str)
            and key.startswith("base_model.model.model.layers.")
            and key.endswith(f".lora_{role}.weight")
            and isinstance(module, str)
            and key == f"base_model.model.{module}.lora_{role}.weight"
            and role in ("A", "B")
            and source["dtype"] == "torch.float32"
            and isinstance(shape, list)
            and len(shape) == 2
            and all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in shape
            )
            and math.prod(shape) == elements,
            f"V82B LoRA shape record changed at {ordinal}",
        )
        canonical.append(
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
    keys = [row["key"] for row in canonical]
    _require_v82b(
        keys == sorted(keys)
        and len(set(keys)) == TENSOR_COUNT_V82B
        and len({row["module"] for row in canonical}) == MODULE_COUNT_V82B
        and sum(row["role"] == "A" for row in canonical) == MODULE_COUNT_V82B
        and sum(row["role"] == "B" for row in canonical) == MODULE_COUNT_V82B
        and sum(row["elements"] for row in canonical) == TOTAL_ELEMENTS_V82B
        and max(row["elements"] for row in canonical) == MAX_TENSOR_ELEMENTS_V82B,
        "V82B ordered LoRA surface totals changed",
    )
    source_records = [
        {
            "key": row["key"],
            "module": row["module"],
            "role": row["role"],
            "shape": row["shape"],
            "dtype": "torch.float32",
            "elements": row["elements"],
        }
        for row in canonical
    ]
    _require_v82b(
        canonical_sha256_v82b(source_records) == EXPECTED_SURFACE_SHA256_V82B
        and canonical_sha256_v82b(keys) == EXPECTED_ORDERED_KEY_SHA256_V82B,
        "V82B sealed LoRA surface identity changed",
    )
    return canonical


def collective_byte_accounting_v82b(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest = validate_ordered_shape_manifest_v82b(records)
    elements = sum(row["elements"] for row in manifest)
    maximum = max(row["elements"] for row in manifest)
    fp32_payload = elements * FP32_BYTES_V82B
    bf16_payload = elements * BF16_BYTES_V82B
    ring_numerator = 2 * (WORLD_SIZE_V82B - 1)
    ring_denominator = WORLD_SIZE_V82B
    _require_v82b(
        fp32_payload * ring_numerator % ring_denominator == 0
        and bf16_payload * ring_numerator % ring_denominator == 0,
        "V82B nominal ring accounting stopped being integral",
    )
    fp32_ring = fp32_payload * ring_numerator // ring_denominator
    bf16_ring = bf16_payload * ring_numerator // ring_denominator
    residual_bank = fp32_payload
    bf16_staging = maximum * BF16_BYTES_V82B
    fp32_accumulator = maximum * FP32_BYTES_V82B
    fused_prepare_hbm = elements * (
        FP32_BYTES_V82B  # read update
        + FP32_BYTES_V82B  # read current residual
        + FP32_BYTES_V82B  # write pending residual
        + BF16_BYTES_V82B  # write transmitted staging
    )
    control_d2h_hbm_read = fp32_payload
    compressed_d2h_hbm_read = bf16_payload
    compressed_local_hbm_lower_bound = fused_prepare_hbm + compressed_d2h_hbm_read
    incremental_local_hbm_lower_bound = (
        compressed_local_hbm_lower_bound - control_d2h_hbm_read
    )
    distribution: dict[str, int] = {}
    for row in manifest:
        key = str(row["elements"])
        distribution[key] = distribution.get(key, 0) + 1
    value = {
        "schema": "qwen36-lora-collective-byte-accounting-v82b",
        "scope": {
            "world_size": WORLD_SIZE_V82B,
            "collective_calls_per_actor_per_update": TENSOR_COUNT_V82B,
            "tensor_count": TENSOR_COUNT_V82B,
            "module_count": MODULE_COUNT_V82B,
            "total_elements_per_actor": elements,
            "minimum_tensor_elements": min(row["elements"] for row in manifest),
            "maximum_tensor_elements": maximum,
            "median_tensor_elements": int(
                sorted(row["elements"] for row in manifest)[len(manifest) // 2]
            ),
            "element_count_distribution": distribution,
        },
        "fp32_control": {
            "payload_bytes_per_actor_per_update": fp32_payload,
            "nominal_ring_bus_bytes_per_actor_per_update": fp32_ring,
            "nominal_ring_bus_bytes_all_actors_per_update": (
                fp32_ring * WORLD_SIZE_V82B
            ),
            "maximum_gpu_accumulator_bytes": fp32_accumulator,
            "maximum_cpu_host_delta_staging_bytes": fp32_accumulator,
            "persistent_residual_bytes": 0,
            "d2h_source_hbm_read_bytes_lower_bound": control_d2h_hbm_read,
        },
        "bf16_error_feedback_hypothetical": {
            "payload_bytes_per_actor_per_update": bf16_payload,
            "nominal_ring_bus_bytes_per_actor_per_update": bf16_ring,
            "nominal_ring_bus_bytes_all_actors_per_update": (
                bf16_ring * WORLD_SIZE_V82B
            ),
            "steady_residual_bank_bytes_per_actor": residual_bank,
            "transaction_two_residual_banks_bytes_per_actor": 2 * residual_bank,
            "maximum_bf16_gpu_staging_bytes": bf16_staging,
            "incremental_transaction_peak_gpu_bytes_per_actor": (
                2 * residual_bank + bf16_staging
            ),
            "maximum_cpu_host_bf16_staging_bytes": bf16_staging,
            "fused_prepare_hbm_bytes_per_actor_per_update_lower_bound": (
                fused_prepare_hbm
            ),
            "d2h_source_hbm_read_bytes_lower_bound": compressed_d2h_hbm_read,
            "local_hbm_bytes_per_actor_per_update_lower_bound_excluding_nccl": (
                compressed_local_hbm_lower_bound
            ),
            "incremental_local_hbm_bytes_per_actor_per_update_lower_bound_"
            "versus_fp32_excluding_nccl": incremental_local_hbm_lower_bound,
        },
        "nominal_projection": {
            "ring_factor_numerator": ring_numerator,
            "ring_factor_denominator": ring_denominator,
            "ring_bus_bytes_saved_per_actor_per_update": fp32_ring - bf16_ring,
            "ring_bus_bytes_saved_all_actors_per_update": (
                (fp32_ring - bf16_ring) * WORLD_SIZE_V82B
            ),
            "payload_fraction_of_fp32": 0.5,
            "algorithm_is_projection_not_measured_pynccl_behavior": True,
            "nccl_internal_hbm_bytes_excluded": True,
        },
    }
    value["content_sha256"] = canonical_sha256_v82b(value)
    return value


def evaluate_materiality_profile_v82b(
    value: Mapping[str, Any], expected_scope_content_sha256: str
) -> dict[str, Any]:
    """Fail-closed future gate before implementing a compressed live arm."""
    expected_fields = {
        "schema",
        "scope_content_sha256",
        "worker_v72_file_sha256",
        "runs",
        "authority",
        "content_sha256",
    }
    _require_v82b(
        isinstance(value, Mapping) and set(value) == expected_fields,
        "V82B materiality profile fields changed",
    )
    body = {key: copy.deepcopy(item) for key, item in value.items() if key != "content_sha256"}
    _require_v82b(
        value["schema"] == PROFILE_SCHEMA_V82B
        and value["scope_content_sha256"] == expected_scope_content_sha256
        and value["worker_v72_file_sha256"]
        == "547d525edfd51412abb3a4980ddc4a55730ad0eb09987ec202ce2ce8f701a2c2"
        and value["content_sha256"] == canonical_sha256_v82b(body),
        "V82B materiality profile identity changed",
    )
    authority = value["authority"]
    _require_v82b(
        authority
        == {
            "dataset_or_training_examples_opened": False,
            "protected_dev_ood_or_holdout_opened": False,
            "adapter_update_committed": False,
            "compression_arm_executed": False,
            "promotion_authorized": False,
        },
        "V82B materiality profile exceeded authority",
    )
    runs = value["runs"]
    _require_v82b(
        isinstance(runs, list) and len(runs) == len(PROFILE_SEEDS_V82B),
        "V82B materiality replicate count changed",
    )
    fractions = []
    upper_bound_improvements = []
    top_three = 0
    for row, seed in zip(runs, PROFILE_SEEDS_V82B, strict=True):
        _require_v82b(
            isinstance(row, Mapping)
            and set(row)
            == {
                "seed",
                "world_size",
                "actors",
                "collective_calls_per_actor",
                "elements_per_actor",
                "update_execute_seconds",
                "unoverlapped_collective_seconds",
                "collective_bottleneck_rank",
                "link_bytes_measured",
                "hbm_bytes_measured",
                "all_four_gpus_attributed",
                "cleanup_idle",
            },
            "V82B materiality run fields changed",
        )
        update_seconds = _finite_v82b(
            row["update_execute_seconds"], "update seconds", positive=True
        )
        collective_seconds = _finite_v82b(
            row["unoverlapped_collective_seconds"],
            "collective seconds",
            positive=True,
        )
        rank = _exact_int_v82b(row["collective_bottleneck_rank"], "rank", 1)
        _require_v82b(
            row["seed"] == seed
            and row["world_size"] == WORLD_SIZE_V82B
            and row["actors"] == WORLD_SIZE_V82B
            and row["collective_calls_per_actor"] == TENSOR_COUNT_V82B
            and row["elements_per_actor"] == TOTAL_ELEMENTS_V82B
            and collective_seconds <= update_seconds
            and row["link_bytes_measured"] is True
            and row["hbm_bytes_measured"] is True
            and row["all_four_gpus_attributed"] is True
            and row["cleanup_idle"] is True,
            "V82B materiality run contract changed",
        )
        fraction = collective_seconds / update_seconds
        fractions.append(fraction)
        # Halving collective payload can save at most half of the measured
        # collective time before quantization/residual overhead is included.
        upper_bound_improvements.append(0.5 * fraction)
        top_three += rank <= 3
    median_fraction = sorted(fractions)[1]
    median_upper_bound = sorted(upper_bound_improvements)[1]
    material = (
        median_fraction >= 0.05
        and median_upper_bound >= 0.01
        and top_three >= 2
    )
    return {
        "profile_valid": True,
        "median_unoverlapped_collective_fraction_of_update": median_fraction,
        "median_perfect_half_payload_speedup_upper_bound_fraction": (
            median_upper_bound
        ),
        "top_three_replicates": top_three,
        "materiality_thresholds_passed": material,
        "compressed_live_arm_implementation_authorized": material,
        "training_or_promotion_authorized": False,
    }

