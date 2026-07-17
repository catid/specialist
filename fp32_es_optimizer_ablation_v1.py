#!/usr/bin/env python3
"""CPU-testable contract for FP32 mirrored-ES optimizer/sigma ablations.

The production worker owns distributed execution.  This module owns the
numerical contract which must be satisfied before a GPU launch: unit Gaussian
directions are scaled exactly once for candidate evaluation, inverted exactly
once in the mirrored estimator, optimizer masters and slots remain CPU FP32,
and every nonzero update is projected to the same relative FP32 L2 budget.

No dataset, model, Ray, vLLM, CUDA, or protected-evaluation import is made here.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from typing import Any

import torch


SCHEMA_V1 = "fp32-mirrored-es-optimizer-sigma-ablation-v1"
MASTER_TENSOR_COUNT_V1 = 70
LOGICAL_MODULE_COUNT_V1 = 35
MASTER_ELEMENTS_V1 = 4_528_128
MASTER_BYTES_V1 = 18_112_512
RUNTIME_ELEMENTS_V1 = 4_921_344
RUNTIME_BF16_BYTES_V1 = 9_842_688
MAX_MASTER_TENSOR_ELEMENTS_V1 = 262_144
FP32_BYTES_V1 = 4
UINT64_BYTES_V1 = 8
REPLICA_COUNT_V1 = 4
DIRECTIONS_PER_UPDATE_V1 = 8
SIGNED_CANDIDATES_PER_UPDATE_V1 = 16
TRAIN_UNITS_PER_CANDIDATE_V1 = 64
ROLLOUTS_PER_UPDATE_V1 = 1_024
UPDATES_PER_REPLICATE_V1 = 2
ROLLOUTS_PER_REPLICATE_V1 = 2_048
REPLICATE_SEEDS_V1 = (1701, 1702, 1703)
SIGMA_SCHEDULE_V1 = (0.0006, 0.0003)
RMS_FLOOR_V1 = 2.0 ** -24
UPDATE_BUDGET_RATIO_V1 = 0.0005
# Candidate subtraction is measured after FP32 rounding.  Fifty parts per
# million is tight enough to match arms while remaining attainable on the
# sealed 4.5M-element surface without iterative extra memory passes.
UPDATE_NORM_RELATIVE_TOLERANCE_V1 = 5.0e-5
GPU_SECOND_CEILING_PER_REPLICATE_V1 = 14_400.0
PARENT_CONTRACT_CONTENT_SHA256_V1 = (
    "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
)
TRAIN_DATASET_FILE_SHA256_V1 = (
    "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a"
)
TRAIN_PANEL_CONTENT_SHA256_V1 = (
    "cdfa9d10669171d5d814b55df1f674a89dfa557c5376b45c8d0073e5d1acaec7"
)
MASTER_TENSOR_INVENTORY_SHA256_V1 = (
    "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
)

OPTIMIZER_CONFIGS_V1 = {
    "sgd": {
        "optimizer": "sgd",
        "momentum": 0.0,
        "weight_decay": 0.0,
    },
    "momentum": {
        "optimizer": "momentum",
        "momentum": 0.9,
        "weight_decay": 0.0,
    },
    "adamw": {
        "optimizer": "adamw",
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1.0e-8,
        "weight_decay": 0.01,
        "bias_correction": "one_minus_beta_power_exact_committed_step",
    },
}

SCALE_MODES_V1 = ("global", "module_fp32_rms_shape_normalized")
COEFFICIENT_MODES_V1 = (
    "raw_pair_difference",
    "pair_sign",
    "signed_rank_absolute_pair_difference",
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
        raise ValueError(f"v1 {label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"v1 {label} must be {qualifier}")
    return result


def _strict_fields_v1(value: Mapping[str, Any], fields: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"v1 {label} fields changed")


def _tensor_bytes_sha256_v1(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _validate_tensor_map_v1(
    tensors: Mapping[str, torch.Tensor],
    label: str,
    *,
    expected_keys: set[str] | None = None,
    allow_empty: bool = False,
) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, Mapping) or (not tensors and not allow_empty):
        raise ValueError(f"v1 {label} must be a nonempty tensor mapping")
    keys = set(tensors)
    if expected_keys is not None and keys != expected_keys:
        raise RuntimeError(f"v1 {label} parameter surface changed")
    result = {}
    for key in sorted(tensors):
        tensor = tensors[key]
        if (
            not isinstance(key, str)
            or not key
            or not isinstance(tensor, torch.Tensor)
            or tensor.device.type != "cpu"
            or tensor.dtype != torch.float32
            or tensor.layout != torch.strided
        ):
            raise RuntimeError(f"v1 {label} must contain CPU FP32 strided tensors")
        value = tensor.detach().contiguous()
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"v1 {label} contains a nonfinite tensor")
        result[key] = value
    return result


def _require_same_shapes_v1(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    label: str,
) -> None:
    if any(candidate[key].shape != reference[key].shape for key in reference):
        raise RuntimeError(f"v1 {label} tensor shape changed")


def _logical_module_key_v1(key: str) -> tuple[str, str]:
    suffixes = {
        ".lora_A.weight": "A",
        ".lora_B.weight": "B",
    }
    for suffix, side in suffixes.items():
        if key.endswith(suffix):
            return key[: -len(suffix)], side
    raise RuntimeError(f"v1 non-LoRA tensor entered the master surface: {key}")


def logical_module_inventory_v1(
    master: Mapping[str, torch.Tensor],
    *,
    require_production_surface: bool = False,
) -> list[dict[str, Any]]:
    values = _validate_tensor_map_v1(master, "master")
    grouped: dict[str, dict[str, tuple[str, torch.Tensor]]] = {}
    for key, tensor in values.items():
        module, side = _logical_module_key_v1(key)
        if side in grouped.setdefault(module, {}):
            raise RuntimeError("v1 duplicated LoRA side")
        grouped[module][side] = (key, tensor)
    if any(set(sides) != {"A", "B"} for sides in grouped.values()):
        raise RuntimeError("v1 every logical module requires exactly one A/B pair")
    if require_production_surface and (
        len(values) != MASTER_TENSOR_COUNT_V1
        or len(grouped) != LOGICAL_MODULE_COUNT_V1
        or sum(t.numel() for t in values.values()) != MASTER_ELEMENTS_V1
    ):
        raise RuntimeError("v1 production LoRA parameter surface changed")

    rows = []
    for module in sorted(grouped):
        sides = grouped[module]
        elements = sum(int(item[1].numel()) for item in sides.values())
        square_sum = sum(
            float(torch.sum(item[1].double().square()).item())
            for item in sides.values()
        )
        rms = math.sqrt(square_sum / elements)
        rows.append({
            "module": module,
            "a_key": sides["A"][0],
            "b_key": sides["B"][0],
            "a_shape": list(sides["A"][1].shape),
            "b_shape": list(sides["B"][1].shape),
            "elements": elements,
            "fp32_square_sum": square_sum,
            "fp32_rms": rms,
        })
    return rows


def fp32_master_identity_v1(
    master: Mapping[str, torch.Tensor],
    *,
    require_production_surface: bool = False,
) -> dict[str, Any]:
    values = _validate_tensor_map_v1(master, "master")
    inventory = logical_module_inventory_v1(
        values, require_production_surface=require_production_surface
    )
    tensors = [{
        "key": key,
        "shape": list(tensor.shape),
        "dtype": "torch.float32",
        "elements": int(tensor.numel()),
        "sha256": _tensor_bytes_sha256_v1(tensor),
    } for key, tensor in values.items()]
    result = {
        "schema": "fp32-es-master-identity-v1",
        "tensor_count": len(tensors),
        "logical_module_count": len(inventory),
        "elements": sum(item["elements"] for item in tensors),
        "bytes": sum(item["elements"] * FP32_BYTES_V1 for item in tensors),
        "ordered_key_sha256": canonical_sha256_v1([item["key"] for item in tensors]),
        "tensor_inventory_sha256": canonical_sha256_v1(tensors),
        "module_shape_inventory_sha256": canonical_sha256_v1([{
            key: item[key]
            for key in ("module", "a_key", "b_key", "a_shape", "b_shape", "elements")
        } for item in inventory]),
    }
    result["identity_sha256"] = canonical_sha256_v1(result)
    return result


def module_sigma_table_v1(
    master: Mapping[str, torch.Tensor],
    base_sigma: float,
    mode: str,
    rms_floor: float = RMS_FLOOR_V1,
) -> dict[str, Any]:
    """Return shape-aware scales with equal total expected perturbation L2.

    The module arm uses ``raw_m=max(||theta_m||/sqrt(n_m), floor)`` and then
    multiplies every raw scale by ``sqrt(P/sum(n_m*raw_m**2))``.  Therefore
    ``sum(n_m*sigma_m**2) == P*base_sigma**2`` for both scale modes.  Using
    raw module L2 instead of RMS is deliberately forbidden because module size
    would be counted twice.
    """
    base_sigma = _finite_float_v1(base_sigma, "base sigma", positive=True)
    rms_floor = _finite_float_v1(rms_floor, "RMS floor", positive=True)
    if mode not in SCALE_MODES_V1:
        raise ValueError("v1 sigma scale mode is not registered")
    inventory = logical_module_inventory_v1(master)
    total_elements = sum(item["elements"] for item in inventory)
    if mode == "global":
        raw = {item["module"]: 1.0 for item in inventory}
    else:
        raw = {
            item["module"]: max(float(item["fp32_rms"]), rms_floor)
            for item in inventory
        }
    weighted_square = sum(
        item["elements"] * raw[item["module"]] ** 2 for item in inventory
    )
    normalization = math.sqrt(total_elements / weighted_square)
    records = []
    for item in inventory:
        scale = base_sigma * raw[item["module"]] * normalization
        records.append({
            "module": item["module"],
            "elements": item["elements"],
            "fp32_rms": item["fp32_rms"],
            "raw_scale_basis": raw[item["module"]],
            "sigma": scale,
            "expected_perturbation_square": item["elements"] * scale * scale,
        })
    expected_square = sum(
        item["expected_perturbation_square"] for item in records
    )
    target_square = total_elements * base_sigma * base_sigma
    if not math.isclose(expected_square, target_square, rel_tol=2e-12, abs_tol=0.0):
        raise RuntimeError("v1 sigma normalization changed perturbation budget")
    result = {
        "schema": "fp32-es-module-sigma-table-v1",
        "mode": mode,
        "base_sigma": base_sigma,
        "rms_floor": rms_floor,
        "scale_basis": (
            "constant_one"
            if mode == "global"
            else "max(fp32_sqrt(sum_squares/elements),rms_floor)"
        ),
        "shape_normalization": "sqrt(total_elements/sum(elements*raw_basis^2))",
        "normalization": normalization,
        "total_elements": total_elements,
        "expected_perturbation_l2_squared": expected_square,
        "target_global_perturbation_l2_squared": target_square,
        "records": records,
    }
    result["content_sha256"] = canonical_sha256_v1(result)
    return result


def _scale_by_tensor_key_v1(
    master: Mapping[str, torch.Tensor], sigma_table: Mapping[str, Any]
) -> dict[str, float]:
    expected = module_sigma_table_v1(
        master,
        sigma_table.get("base_sigma", float("nan")),
        sigma_table.get("mode", ""),
        sigma_table.get("rms_floor", float("nan")),
    )
    if dict(sigma_table) != expected:
        raise RuntimeError("v1 module sigma table identity changed")
    by_module = {item["module"]: item["sigma"] for item in expected["records"]}
    return {
        key: by_module[_logical_module_key_v1(key)[0]] for key in master
    }


def materialize_antithetic_candidate_v1(
    master: Mapping[str, torch.Tensor],
    unit_noise: Mapping[str, torch.Tensor],
    sigma_table: Mapping[str, Any],
    sign: int,
) -> dict[str, torch.Tensor]:
    values = _validate_tensor_map_v1(master, "master")
    noise = _validate_tensor_map_v1(
        unit_noise, "unit noise", expected_keys=set(values)
    )
    _require_same_shapes_v1(values, noise, "unit noise")
    if isinstance(sign, bool) or sign not in (-1, 1):
        raise ValueError("v1 antithetic sign must be exactly +/-1")
    scales = _scale_by_tensor_key_v1(values, sigma_table)
    return {
        key: values[key].add(noise[key], alpha=sign * scales[key]).contiguous()
        for key in values
    }


def shape_pair_coefficients_v1(
    pair_differences: Sequence[float], mode: str
) -> list[float]:
    if mode not in COEFFICIENT_MODES_V1:
        raise ValueError("v1 coefficient mode is not mirrored-pair safe")
    differences = [
        _finite_float_v1(value, "pair difference") for value in pair_differences
    ]
    if not differences:
        raise ValueError("v1 pair differences must not be empty")
    if mode == "raw_pair_difference":
        return differences
    if mode == "pair_sign":
        return [1.0 if value > 0.0 else -1.0 if value < 0.0 else 0.0
                for value in differences]

    # Average ranks for ties, followed by sign restoration.  Swapping + and -
    # negates a coefficient exactly; ranking signed candidates independently is
    # intentionally not an available mode.
    magnitudes = [abs(value) for value in differences]
    ranks = [0.0] * len(magnitudes)
    ordered = sorted(range(len(magnitudes)), key=lambda index: magnitudes[index])
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while (
            end < len(ordered)
            and magnitudes[ordered[end]] == magnitudes[ordered[cursor]]
        ):
            end += 1
        average_rank = 0.5 * ((cursor + 1) + end) / len(ordered)
        for index in ordered[cursor:end]:
            ranks[index] = average_rank
        cursor = end
    return [
        rank if value > 0.0 else -rank if value < 0.0 else 0.0
        for rank, value in zip(ranks, differences, strict=True)
    ]


def mirrored_gradient_v1(
    master: Mapping[str, torch.Tensor],
    unit_noises: Sequence[Mapping[str, torch.Tensor]],
    pair_differences: Sequence[float],
    sigma_table: Mapping[str, Any],
    coefficient_mode: str,
) -> dict[str, Any]:
    values = _validate_tensor_map_v1(master, "master")
    differences = [
        _finite_float_v1(value, "pair difference") for value in pair_differences
    ]
    if len(unit_noises) != len(differences) or len(differences) < 2:
        raise ValueError("v1 directions and pair differences must have equal size >=2")
    mean = math.fsum(differences) / len(differences)
    variance = math.fsum((value - mean) ** 2 for value in differences) / len(
        differences
    )
    if variance == 0.0:
        return {
            "status": "skip_zero_pair_difference_variance",
            "pair_difference_variance": 0.0,
            "gradient": {key: torch.zeros_like(tensor) for key, tensor in values.items()},
            "optimizer_state_may_advance": False,
            "checkpoint_may_change": False,
        }
    coefficients = shape_pair_coefficients_v1(differences, coefficient_mode)
    noises = [
        _validate_tensor_map_v1(noise, "unit noise", expected_keys=set(values))
        for noise in unit_noises
    ]
    for noise in noises:
        _require_same_shapes_v1(values, noise, "unit noise")
    scales = _scale_by_tensor_key_v1(values, sigma_table)
    gradient = {}
    population = len(noises)
    for key, tensor in values.items():
        accumulator = torch.zeros_like(tensor)
        for noise, coefficient in zip(noises, coefficients, strict=True):
            accumulator.add_(noise[key], alpha=coefficient)
        accumulator.mul_(1.0 / (2.0 * population * scales[key]))
        if not bool(torch.isfinite(accumulator).all()):
            raise FloatingPointError("v1 mirrored gradient became nonfinite")
        gradient[key] = accumulator.contiguous()
    return {
        "status": "finite_positive_pair_difference_variance",
        "pair_difference_variance": variance,
        "coefficient_mode": coefficient_mode,
        "coefficients": coefficients,
        "coefficient_sha256": canonical_sha256_v1(coefficients),
        "gradient": gradient,
        "candidate_scale_application_count": 1,
        "estimator_inverse_scale_application_count": 1,
        "noise_representation": "unscaled_unit_standard_normal",
        "optimizer_state_may_advance": True,
        "checkpoint_may_change": True,
    }


def initial_optimizer_state_v1(optimizer: str) -> dict[str, Any]:
    if optimizer not in OPTIMIZER_CONFIGS_V1:
        raise ValueError("v1 optimizer is not registered")
    return {
        "schema": "fp32-es-optimizer-state-v1",
        "optimizer": optimizer,
        "step": 0,
        "slot_dtype": "torch.float32",
        "slots": {},
    }


def validate_optimizer_state_v1(
    state: Mapping[str, Any],
    master: Mapping[str, torch.Tensor],
    optimizer: str,
) -> dict[str, Any]:
    _strict_fields_v1(
        state,
        {"schema", "optimizer", "step", "slot_dtype", "slots"},
        "optimizer state",
    )
    if (
        state["schema"] != "fp32-es-optimizer-state-v1"
        or state["optimizer"] != optimizer
        or state["slot_dtype"] != "torch.float32"
    ):
        raise RuntimeError("v1 optimizer state identity or dtype changed")
    step = _exact_int_v1(state["step"], "optimizer step")
    values = _validate_tensor_map_v1(master, "master")
    slots = state["slots"]
    if not isinstance(slots, Mapping):
        raise ValueError("v1 optimizer slots must be a mapping")
    expected_slot_names = {
        "sgd": set(),
        "momentum": set() if step == 0 else {"velocity"},
        "adamw": set() if step == 0 else {"first_moment", "second_moment"},
    }[optimizer]
    if set(slots) != expected_slot_names:
        raise RuntimeError("v1 optimizer slot surface changed")
    checked_slots = {}
    for slot_name, tensors in slots.items():
        checked = _validate_tensor_map_v1(
            tensors,
            f"optimizer slot {slot_name}",
            expected_keys=set(values),
        )
        _require_same_shapes_v1(values, checked, f"optimizer slot {slot_name}")
        if slot_name == "second_moment" and any(
            bool(torch.any(tensor < 0.0)) for tensor in checked.values()
        ):
            raise RuntimeError("v1 Adam second moment became negative")
        checked_slots[slot_name] = checked
    return {
        "schema": state["schema"],
        "optimizer": optimizer,
        "step": step,
        "slot_dtype": state["slot_dtype"],
        "slots": checked_slots,
    }


def optimizer_state_identity_v1(
    state: Mapping[str, Any],
    master: Mapping[str, torch.Tensor],
    optimizer: str,
) -> dict[str, Any]:
    checked = validate_optimizer_state_v1(state, master, optimizer)
    slots = []
    tensor_bytes = 0
    for slot_name in sorted(checked["slots"]):
        for key, tensor in checked["slots"][slot_name].items():
            tensor_bytes += int(tensor.numel()) * FP32_BYTES_V1
            slots.append({
                "slot": slot_name,
                "key": key,
                "shape": list(tensor.shape),
                "dtype": "torch.float32",
                "sha256": _tensor_bytes_sha256_v1(tensor),
            })
    result = {
        "schema": "fp32-es-optimizer-state-identity-v1",
        "optimizer": optimizer,
        "step": checked["step"],
        "slot_dtype": "torch.float32",
        "slot_tensor_bytes": tensor_bytes,
        "logical_step_counter_bytes": UINT64_BYTES_V1,
        "slot_inventory_sha256": canonical_sha256_v1(slots),
    }
    result["identity_sha256"] = canonical_sha256_v1(result)
    return result


def optimizer_direction_v1(
    master: Mapping[str, torch.Tensor],
    gradient: Mapping[str, torch.Tensor],
    state: Mapping[str, Any],
    optimizer: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    values = _validate_tensor_map_v1(master, "master")
    gradients = _validate_tensor_map_v1(
        gradient, "gradient", expected_keys=set(values)
    )
    _require_same_shapes_v1(values, gradients, "gradient")
    checked = validate_optimizer_state_v1(state, values, optimizer)
    config = OPTIMIZER_CONFIGS_V1[optimizer]
    new_step = checked["step"] + 1
    direction: dict[str, torch.Tensor] = {}
    new_slots: dict[str, dict[str, torch.Tensor]] = {}

    if optimizer == "sgd":
        direction = {key: tensor.clone() for key, tensor in gradients.items()}
    elif optimizer == "momentum":
        prior = checked["slots"].get("velocity", {
            key: torch.zeros_like(tensor) for key, tensor in values.items()
        })
        velocity = {}
        for key in values:
            value = prior[key].mul(config["momentum"]).add(gradients[key])
            velocity[key] = value.contiguous()
            direction[key] = value.clone()
        new_slots["velocity"] = velocity
    else:
        prior_first = checked["slots"].get("first_moment", {
            key: torch.zeros_like(tensor) for key, tensor in values.items()
        })
        prior_second = checked["slots"].get("second_moment", {
            key: torch.zeros_like(tensor) for key, tensor in values.items()
        })
        first, second = {}, {}
        beta1, beta2 = config["beta1"], config["beta2"]
        correction1 = 1.0 - beta1 ** new_step
        correction2 = 1.0 - beta2 ** new_step
        for key in values:
            m = prior_first[key].mul(beta1).add(
                gradients[key], alpha=1.0 - beta1
            )
            v = prior_second[key].mul(beta2).addcmul(
                gradients[key], gradients[key], value=1.0 - beta2
            )
            adaptive = (m / correction1) / (
                torch.sqrt(v / correction2) + config["epsilon"]
            )
            # Reward is maximized.  Decoupled weight decay therefore points in
            # the negative-master direction before the shared norm projection.
            adaptive = adaptive.add(values[key], alpha=-config["weight_decay"])
            first[key], second[key] = m.contiguous(), v.contiguous()
            direction[key] = adaptive.contiguous()
        new_slots["first_moment"] = first
        new_slots["second_moment"] = second

    if not all(bool(torch.isfinite(item).all()) for item in direction.values()):
        raise FloatingPointError("v1 optimizer direction became nonfinite")
    new_state = {
        "schema": "fp32-es-optimizer-state-v1",
        "optimizer": optimizer,
        "step": new_step,
        "slot_dtype": "torch.float32",
        "slots": new_slots,
    }
    validate_optimizer_state_v1(new_state, values, optimizer)
    receipt = {
        "optimizer": optimizer,
        "step_before": checked["step"],
        "step_after": new_step,
        "bias_correction_step": new_step if optimizer == "adamw" else None,
        "ascent": True,
        "state_dtype": "torch.float32",
    }
    return direction, new_state, receipt


def budgeted_optimizer_update_v1(
    master: Mapping[str, torch.Tensor],
    gradient: Mapping[str, torch.Tensor],
    state: Mapping[str, Any],
    optimizer: str,
    budget_ratio: float = UPDATE_BUDGET_RATIO_V1,
) -> dict[str, Any]:
    """Prepare, but do not commit, one transactional optimizer update.

    A zero direction returns the exact prior optimizer state and master.  The
    caller may commit the candidate state only after four-replica consensus;
    until then the supplied state remains the rollback authority.
    """
    values = _validate_tensor_map_v1(master, "master")
    prior = validate_optimizer_state_v1(state, values, optimizer)
    direction, candidate_state, optimizer_receipt = optimizer_direction_v1(
        values, gradient, prior, optimizer
    )
    budget = apply_update_norm_budget_v1(
        values, direction, budget_ratio=budget_ratio
    )
    if budget["status"] == "skip_zero_optimizer_direction":
        return {
            "status": budget["status"],
            "candidate": budget["candidate"],
            "candidate_optimizer_state": prior,
            "optimizer_receipt": {
                **optimizer_receipt,
                "step_after": prior["step"],
                "bias_correction_step": None,
                "state_advance_rolled_back": True,
            },
            "budget_receipt": {
                key: value for key, value in budget.items() if key != "candidate"
            },
        }
    return {
        "status": "prepared_finite_nonzero_budgeted_update",
        "candidate": budget["candidate"],
        "candidate_optimizer_state": candidate_state,
        "optimizer_receipt": {
            **optimizer_receipt,
            "state_advance_rolled_back": False,
        },
        "budget_receipt": {
            key: value for key, value in budget.items() if key != "candidate"
        },
    }


def apply_update_norm_budget_v1(
    master: Mapping[str, torch.Tensor],
    direction: Mapping[str, torch.Tensor],
    budget_ratio: float = UPDATE_BUDGET_RATIO_V1,
    master_rms_floor: float = RMS_FLOOR_V1,
) -> dict[str, Any]:
    values = _validate_tensor_map_v1(master, "master")
    directions = _validate_tensor_map_v1(
        direction, "optimizer direction", expected_keys=set(values)
    )
    _require_same_shapes_v1(values, directions, "optimizer direction")
    budget_ratio = _finite_float_v1(
        budget_ratio, "update budget ratio", positive=True
    )
    master_rms_floor = _finite_float_v1(
        master_rms_floor, "master RMS floor", positive=True
    )
    elements = sum(int(tensor.numel()) for tensor in values.values())
    master_square = math.fsum(
        float(torch.sum(tensor.double().square()).item())
        for tensor in values.values()
    )
    direction_square = math.fsum(
        float(torch.sum(tensor.double().square()).item())
        for tensor in directions.values()
    )
    if direction_square == 0.0:
        return {
            "status": "skip_zero_optimizer_direction",
            "candidate": {key: tensor.clone() for key, tensor in values.items()},
            "target_update_l2": 0.0,
            "observed_update_l2": 0.0,
            "optimizer_state_may_advance": False,
            "checkpoint_may_change": False,
        }
    master_l2 = math.sqrt(master_square)
    target = budget_ratio * max(master_l2, master_rms_floor * math.sqrt(elements))
    direction_l2 = math.sqrt(direction_square)
    multiplier = target / direction_l2
    candidate = {
        key: values[key].add(directions[key], alpha=multiplier).contiguous()
        for key in values
    }
    observed_square = math.fsum(
        float(torch.sum((candidate[key].double() - values[key].double()).square()).item())
        for key in values
    )
    observed = math.sqrt(observed_square)
    relative_error = abs(observed - target) / target
    if relative_error > UPDATE_NORM_RELATIVE_TOLERANCE_V1:
        raise RuntimeError("v1 FP32 update missed its norm budget")
    return {
        "status": "finite_nonzero_budgeted_update",
        "candidate": candidate,
        "master_l2": master_l2,
        "raw_direction_l2": direction_l2,
        "budget_ratio": budget_ratio,
        "normalization_multiplier": multiplier,
        "target_update_l2": target,
        "observed_update_l2": observed,
        "relative_error": relative_error,
        "optimizer_state_may_advance": True,
        "checkpoint_may_change": True,
    }


def checkpoint_identity_v1(
    master: Mapping[str, torch.Tensor],
    optimizer_state: Mapping[str, Any],
    optimizer: str,
    run_state: Mapping[str, Any],
) -> dict[str, Any]:
    master_identity = fp32_master_identity_v1(master)
    state_identity = optimizer_state_identity_v1(
        optimizer_state, master, optimizer
    )
    try:
        reopened = json.loads(json.dumps(
            dict(run_state), allow_nan=False, sort_keys=True, separators=(",", ":")
        ))
    except (TypeError, ValueError) as error:
        raise ValueError("v1 run state must be finite JSON") from error
    if reopened != dict(run_state):
        raise ValueError("v1 run state changes under JSON round trip")
    result = {
        "schema": "fp32-es-checkpoint-identity-v1",
        "master_identity_sha256": master_identity["identity_sha256"],
        "optimizer_state_identity_sha256": state_identity["identity_sha256"],
        "optimizer": optimizer,
        "optimizer_step": state_identity["step"],
        "run_state_sha256": canonical_sha256_v1(reopened),
    }
    result["checkpoint_sha256"] = canonical_sha256_v1(result)
    return result


def direction_seeds_v1(
    parent_contract_content_sha256: str,
    replicate_seed: int,
    update_index: int,
) -> list[int]:
    parent = _hex_sha_v1(parent_contract_content_sha256, "parent contract content")
    seed = _exact_int_v1(replicate_seed, "replicate seed", minimum=1)
    update = _exact_int_v1(update_index, "update index")
    result = []
    for direction_index in range(DIRECTIONS_PER_UPDATE_V1):
        payload = (
            f"{parent}\0fp32-es-optimizer-sigma-v1\0{seed}\0"
            f"{update}\0{direction_index}"
        ).encode("ascii")
        value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
        result.append(value & ((1 << 63) - 1))
    if len(set(result)) != len(result):
        raise RuntimeError("v1 derived a duplicated direction seed")
    return result


def arm_grid_v1() -> list[dict[str, Any]]:
    arms = []
    for optimizer in ("sgd", "momentum", "adamw"):
        for scale_mode in SCALE_MODES_V1:
            arms.append({
                "arm_id": f"{optimizer}__{scale_mode}__raw_pair_difference",
                "optimizer": optimizer,
                "optimizer_config": OPTIMIZER_CONFIGS_V1[optimizer],
                "scale_mode": scale_mode,
                "coefficient_mode": "raw_pair_difference",
            })
    for coefficient_mode in (
        "pair_sign",
        "signed_rank_absolute_pair_difference",
    ):
        arms.append({
            "arm_id": f"sgd__global__{coefficient_mode}",
            "optimizer": "sgd",
            "optimizer_config": OPTIMIZER_CONFIGS_V1["sgd"],
            "scale_mode": "global",
            "coefficient_mode": coefficient_mode,
        })
    return arms


def optimizer_memory_contract_v1(elements: int = MASTER_ELEMENTS_V1) -> dict[str, Any]:
    elements = _exact_int_v1(elements, "master elements", minimum=1)
    master_bytes = elements * FP32_BYTES_V1
    optimizer_slot_vectors = {"sgd": 0, "momentum": 1, "adamw": 2}
    # Algorithmic host-memory traffic is a lower bound.  It includes one common
    # master-RMS inventory pass, a rollback clone, and the minimum optimizer /
    # norm-projection passes described in the receipt.  Allocator and checkpoint
    # traffic must be measured and reported separately at runtime.
    traffic_multipliers = {"sgd": 7, "momentum": 10, "adamw": 15}
    optimizers = {}
    for optimizer in optimizer_slot_vectors:
        slot_bytes = optimizer_slot_vectors[optimizer] * master_bytes
        checkpoint_bytes = master_bytes + slot_bytes + UINT64_BYTES_V1
        optimizers[optimizer] = {
            "persistent_optimizer_host_tensor_bytes_per_replica": slot_bytes,
            "logical_step_counter_bytes_per_replica": UINT64_BYTES_V1,
            "persistent_optimizer_gpu_bytes_per_replica": 0,
            "checkpoint_tensor_and_step_bytes_per_replica": checkpoint_bytes,
            "four_replica_optimizer_host_tensor_bytes": slot_bytes * REPLICA_COUNT_V1,
            "four_replica_checkpoint_tensor_and_step_bytes": (
                checkpoint_bytes * REPLICA_COUNT_V1
            ),
            "algorithmic_minimum_host_memory_traffic_bytes_per_update_per_replica": (
                traffic_multipliers[optimizer] * master_bytes
            ),
        }
    population_transfer = (
        2 * SIGNED_CANDIDATES_PER_UPDATE_V1 * RUNTIME_BF16_BYTES_V1
    )
    return {
        "schema": "fp32-es-optimizer-memory-bandwidth-contract-v1",
        "master_elements": elements,
        "master_bytes_per_replica": master_bytes,
        "replicas": REPLICA_COUNT_V1,
        "runtime_bf16_view_elements_per_replica": RUNTIME_ELEMENTS_V1,
        "runtime_bf16_view_bytes_per_replica": RUNTIME_BF16_BYTES_V1,
        "persistent_optimizer_state_location": "host",
        "persistent_optimizer_state_dtype": "torch.float32",
        "optimizer_gpu_state_prohibited": True,
        "rollback_master_transient_host_bytes_per_replica": master_bytes,
        "candidate_master_transient_host_bytes_per_replica": master_bytes,
        "streamed_gpu_fp32_noise_plus_accumulator_scratch_ceiling_bytes_per_replica": (
            2 * MAX_MASTER_TENSOR_ELEMENTS_V1 * FP32_BYTES_V1
        ),
        "reduced_gradient_d2h_bytes_per_update_per_replica": master_bytes,
        "committed_runtime_h2d_bytes_per_update_per_replica": RUNTIME_BF16_BYTES_V1,
        "fixed_population_candidate_plus_restore_h2d_bytes_per_update_all_replicas": (
            population_transfer
        ),
        "sigma_table_host_bytes": LOGICAL_MODULE_COUNT_V1 * FP32_BYTES_V1,
        "common_master_rms_inventory_read_bytes_per_update_per_replica": master_bytes,
        "runtime_measurements_required": [
            "peak allocated and reserved VRAM by phase and physical GPU",
            "D2H/H2D bytes and elapsed time by phase",
            "host optimizer elapsed time and achieved memory bandwidth",
            "checkpoint bytes and elapsed time",
            "allocator/copy traffic above algorithmic minimum",
        ],
        "optimizers": optimizers,
    }


def validate_preregistration_v1(plan: Mapping[str, Any], *, launch: bool = False) -> dict:
    top_fields = {
        "schema", "status", "purpose", "authorization", "dependencies",
        "parent_contract", "inputs", "parameter_surface", "sigma_contract",
        "estimator_contract", "optimizer_contract", "update_budget_contract",
        "compute_contract", "grid", "memory_bandwidth_contract",
        "checkpoint_contract", "evaluation_gates", "failure_policy",
        "reporting", "implementation_bindings", "artifacts",
        "content_sha256_before_self_field",
    }
    _strict_fields_v1(plan, top_fields, "preregistration")
    compact = {key: value for key, value in plan.items()
               if key != "content_sha256_before_self_field"}
    if (
        plan["schema"] != SCHEMA_V1
        or canonical_sha256_v1(compact) != plan["content_sha256_before_self_field"]
    ):
        raise RuntimeError("v1 preregistration content identity changed")
    authorization = plan["authorization"]
    _strict_fields_v1(
        authorization,
        {
            "cpu_preview", "gpu_launch", "train_semantics", "dev_after_training",
            "ood_after_training", "protected_holdout", "live_run_access",
        },
        "authorization",
    )
    if (
        authorization["cpu_preview"] is not True
        or authorization["train_semantics"] is not True
        or authorization["dev_after_training"] is not True
        or authorization["ood_after_training"] is not True
        or authorization["protected_holdout"] is not False
        or authorization["live_run_access"] is not False
    ):
        raise RuntimeError("v1 data authorization changed")
    dependencies = plan["dependencies"]
    _strict_fields_v1(
        dependencies,
        {"mirrored_es", "memory_roofline", "all_runtime_dependencies_complete"},
        "dependencies",
    )
    mirrored_dependency = dependencies["mirrored_es"]
    _strict_fields_v1(
        mirrored_dependency,
        {
            "path", "file_sha256", "content_sha256", "cpu_contract_complete",
            "substantive_nonzero_pair_differences_and_exact_restore_observed",
            "accepted_all_four_gpu_activity_receipt_complete", "status",
        },
        "mirrored dependency",
    )
    memory_dependency = dependencies["memory_roofline"]
    _strict_fields_v1(
        memory_dependency,
        {
            "bead", "existing_profile_is_diagnostic",
            "optimizer_phase_transfer_and_bandwidth_receipt_complete", "status",
        },
        "memory dependency",
    )
    mirrored_ready = (
        mirrored_dependency["cpu_contract_complete"] is True
        and mirrored_dependency[
            "substantive_nonzero_pair_differences_and_exact_restore_observed"
        ] is True
        and mirrored_dependency[
            "accepted_all_four_gpu_activity_receipt_complete"
        ] is True
    )
    memory_ready = memory_dependency[
        "optimizer_phase_transfer_and_bandwidth_receipt_complete"
    ] is True
    expected_launch = mirrored_ready and memory_ready
    if dependencies["all_runtime_dependencies_complete"] is not expected_launch:
        raise RuntimeError("v1 aggregate dependency gate changed")
    if (
        (mirrored_ready and mirrored_dependency["status"] != "complete")
        or (
            not mirrored_ready
            and mirrored_dependency["status"]
            != "v66d_all_four_gpu_activity_receipt_pending"
        )
        or (memory_ready and memory_dependency["status"] != "complete")
        or (
            not memory_ready
            and memory_dependency["status"]
            != "optimizer_phase_empirical_profile_pending"
        )
    ):
        raise RuntimeError("v1 dependency status changed")
    if authorization["gpu_launch"] is not expected_launch:
        raise RuntimeError("v1 launch authorization disagrees with dependencies")
    if launch and not expected_launch:
        raise RuntimeError("v1 runtime dependencies remain incomplete")
    expected_status = (
        "launch_ready_after_runtime_dependencies"
        if expected_launch
        else "sealed_cpu_preview_runtime_dependencies_pending"
    )
    if plan["status"] != expected_status:
        raise RuntimeError("v1 status disagrees with runtime dependencies")
    parent = plan["parent_contract"]
    if (
        parent.get("content_sha256") != PARENT_CONTRACT_CONTENT_SHA256_V1
        or parent.get("source_disjointness_passed") is not True
        or parent.get("adaptation_train_file_sha256")
        != TRAIN_DATASET_FILE_SHA256_V1
        or parent.get("estimator_control_rollouts_per_seed")
        != ROLLOUTS_PER_REPLICATE_V1
        or parent.get("estimator_control_gpu_second_ceiling_per_seed")
        != GPU_SECOND_CEILING_PER_REPLICATE_V1
        or parent.get("protected_access_authorized") is not False
        or plan["inputs"]["train_dataset"].get("file_sha256")
        != TRAIN_DATASET_FILE_SHA256_V1
        or plan["inputs"]["train_panel"].get("content_sha256")
        != TRAIN_PANEL_CONTENT_SHA256_V1
    ):
        raise RuntimeError("v1 parent/input identity or data gate changed")

    surface = plan["parameter_surface"]
    surface_identity_fields = {
        "schema", "tensor_count", "logical_module_count", "elements", "bytes",
        "ordered_key_sha256", "tensor_inventory_sha256",
        "module_shape_inventory_sha256",
    }
    surface_identity = {
        key: surface[key] for key in surface_identity_fields
    }
    if (
        surface.get("tensor_count") != MASTER_TENSOR_COUNT_V1
        or surface.get("logical_module_count") != LOGICAL_MODULE_COUNT_V1
        or surface.get("elements") != MASTER_ELEMENTS_V1
        or surface.get("bytes") != MASTER_BYTES_V1
        or surface.get("dtype") != "torch.float32"
        or canonical_sha256_v1(surface_identity) != surface.get("identity_sha256")
        or surface.get("tensor_inventory_sha256")
        != MASTER_TENSOR_INVENTORY_SHA256_V1
        or plan["inputs"]["source_adapter"]["canonical_v41_master_sha256"]
        != MASTER_TENSOR_INVENTORY_SHA256_V1
    ):
        raise RuntimeError("v1 parameter surface changed")
    sigma = plan["sigma_contract"]
    _strict_fields_v1(
        sigma,
        {
            "modes", "schedule", "schedule_rule", "rms_floor", "module_basis",
            "shape_normalization", "candidate_scale_application_count",
            "estimator_inverse_scale_application_count",
            "equal_expected_perturbation_l2_across_modes",
            "max_module_expected_energy_share", "tables",
        },
        "sigma contract",
    )
    if (
        sigma.get("modes") != list(SCALE_MODES_V1)
        or tuple(sigma.get("schedule", [])) != SIGMA_SCHEDULE_V1
        or sigma.get("rms_floor") != RMS_FLOOR_V1
        or sigma.get("candidate_scale_application_count") != 1
        or sigma.get("estimator_inverse_scale_application_count") != 1
        or sigma.get("equal_expected_perturbation_l2_across_modes") is not True
        or sigma.get("module_basis")
        != "max(fp32_sqrt(sum_squares/elements),rms_floor)"
        or sigma.get("shape_normalization")
        != "sqrt(total_elements/sum(elements*raw_basis^2))"
    ):
        raise RuntimeError("v1 sigma contract changed")
    tables = sigma["tables"]
    if not isinstance(tables, list) or len(tables) != (
        len(SIGMA_SCHEDULE_V1) * len(SCALE_MODES_V1)
    ):
        raise RuntimeError("v1 sigma table coverage changed")
    module_rows = {
        item["module"]: item for item in surface.get("module_inventory", [])
    }
    if (
        len(module_rows) != LOGICAL_MODULE_COUNT_V1
        or sum(item.get("elements", -1) for item in module_rows.values())
        != MASTER_ELEMENTS_V1
    ):
        raise RuntimeError("v1 module inventory changed")
    observed_table_keys = set()
    observed_max_energy_share = 0.0
    for wrapper in tables:
        _strict_fields_v1(wrapper, {"update_index", "mode", "table"}, "sigma table wrapper")
        update_index = _exact_int_v1(wrapper["update_index"], "sigma update index")
        mode = wrapper["mode"]
        key = (update_index, mode)
        if (
            update_index >= len(SIGMA_SCHEDULE_V1)
            or mode not in SCALE_MODES_V1
            or key in observed_table_keys
        ):
            raise RuntimeError("v1 duplicated or unregistered sigma table")
        observed_table_keys.add(key)
        table = wrapper["table"]
        table_fields = {
            "schema", "mode", "base_sigma", "rms_floor", "scale_basis",
            "shape_normalization", "normalization", "total_elements",
            "expected_perturbation_l2_squared",
            "target_global_perturbation_l2_squared", "records", "content_sha256",
        }
        _strict_fields_v1(table, table_fields, "sigma table")
        table_compact = {item: table[item] for item in table_fields - {"content_sha256"}}
        if (
            table["schema"] != "fp32-es-module-sigma-table-v1"
            or table["mode"] != mode
            or table["base_sigma"] != SIGMA_SCHEDULE_V1[update_index]
            or table["rms_floor"] != RMS_FLOOR_V1
            or table["shape_normalization"]
            != "sqrt(total_elements/sum(elements*raw_basis^2))"
            or table["total_elements"] != MASTER_ELEMENTS_V1
            or canonical_sha256_v1(table_compact) != table["content_sha256"]
        ):
            raise RuntimeError("v1 sigma table identity changed")
        records = table["records"]
        if not isinstance(records, list) or len(records) != LOGICAL_MODULE_COUNT_V1:
            raise RuntimeError("v1 sigma table module count changed")
        raw_by_module = {}
        for record in records:
            _strict_fields_v1(
                record,
                {
                    "module", "elements", "fp32_rms", "raw_scale_basis",
                    "sigma", "expected_perturbation_square",
                },
                "sigma record",
            )
            source = module_rows.get(record["module"])
            if (
                source is None
                or record["elements"] != source["elements"]
                or record["fp32_rms"] != source["fp32_rms"]
            ):
                raise RuntimeError("v1 sigma record left the master module surface")
            raw = (
                1.0 if mode == "global"
                else max(float(source["fp32_rms"]), RMS_FLOOR_V1)
            )
            if record["raw_scale_basis"] != raw:
                raise RuntimeError("v1 module-size-dominated sigma basis detected")
            raw_by_module[record["module"]] = raw
        weighted = math.fsum(
            module_rows[module]["elements"] * raw * raw
            for module, raw in raw_by_module.items()
        )
        expected_normalization = math.sqrt(MASTER_ELEMENTS_V1 / weighted)
        if not math.isclose(
            table["normalization"], expected_normalization, rel_tol=2e-15, abs_tol=0.0
        ):
            raise RuntimeError("v1 sigma shape normalization changed")
        expected_total = 0.0
        for record in records:
            expected_sigma = (
                table["base_sigma"]
                * record["raw_scale_basis"]
                * expected_normalization
            )
            expected_square = record["elements"] * expected_sigma * expected_sigma
            if (
                not math.isclose(record["sigma"], expected_sigma, rel_tol=2e-15, abs_tol=0.0)
                or not math.isclose(
                    record["expected_perturbation_square"],
                    expected_square,
                    rel_tol=2e-15,
                    abs_tol=0.0,
                )
            ):
                raise RuntimeError("v1 sigma was not derived exactly once from RMS/shape")
            expected_total += expected_square
            if mode == "module_fp32_rms_shape_normalized":
                observed_max_energy_share = max(
                    observed_max_energy_share,
                    expected_square / table["expected_perturbation_l2_squared"],
                )
        target_total = (
            MASTER_ELEMENTS_V1 * table["base_sigma"] * table["base_sigma"]
        )
        if (
            not math.isclose(expected_total, target_total, rel_tol=2e-12, abs_tol=0.0)
            or not math.isclose(
                table["expected_perturbation_l2_squared"],
                expected_total,
                rel_tol=2e-15,
                abs_tol=0.0,
            )
            or table["target_global_perturbation_l2_squared"] != target_total
        ):
            raise RuntimeError("v1 unequal expected perturbation norm")
    expected_table_keys = {
        (update, mode)
        for update in range(len(SIGMA_SCHEDULE_V1))
        for mode in SCALE_MODES_V1
    }
    if observed_table_keys != expected_table_keys or not math.isclose(
        sigma["max_module_expected_energy_share"],
        observed_max_energy_share,
        rel_tol=2e-15,
        abs_tol=0.0,
    ):
        raise RuntimeError("v1 sigma table summary changed")
    estimator = plan["estimator_contract"]
    if (
        estimator.get("directions_per_update") != DIRECTIONS_PER_UPDATE_V1
        or estimator.get("signed_candidates_per_update")
        != SIGNED_CANDIDATES_PER_UPDATE_V1
        or estimator.get("pair_sign_swap_must_negate_coefficient") is not True
        or estimator.get("independent_signed_candidate_ranking") != "prohibited"
        or estimator.get("registered_coefficient_modes")
        != list(COEFFICIENT_MODES_V1)
    ):
        raise RuntimeError("v1 estimator contract changed")
    optimizer_contract = plan["optimizer_contract"]
    if (
        optimizer_contract.get("configs") != OPTIMIZER_CONFIGS_V1
        or optimizer_contract.get("master_dtype") != "torch.float32"
        or optimizer_contract.get("slot_dtype") != "torch.float32"
        or optimizer_contract.get("state_replication")
        != "exact_on_all_four_replicas"
    ):
        raise RuntimeError("v1 optimizer contract changed")
    budget = plan["update_budget_contract"]
    if (
        budget.get("ratio") != UPDATE_BUDGET_RATIO_V1
        or budget.get("relative_tolerance") != UPDATE_NORM_RELATIVE_TOLERANCE_V1
        or budget.get("formula")
        != "ratio*max(fp32_master_l2,rms_floor*sqrt(elements))"
        or budget.get("nonzero_update_hits_exact_budget") is not True
    ):
        raise RuntimeError("v1 update budget changed")
    compute = plan["compute_contract"]
    expected_seed_sets = {
        str(seed): [
            direction_seeds_v1(
                plan["parent_contract"]["content_sha256"], seed, update
            )
            for update in range(UPDATES_PER_REPLICATE_V1)
        ]
        for seed in REPLICATE_SEEDS_V1
    }
    if (
        compute.get("updates_per_replicate") != UPDATES_PER_REPLICATE_V1
        or compute.get("rollouts_per_update") != ROLLOUTS_PER_UPDATE_V1
        or compute.get("rollouts_per_replicate") != ROLLOUTS_PER_REPLICATE_V1
        or compute.get("replicate_seeds") != list(REPLICATE_SEEDS_V1)
        or compute.get("gpu_second_ceiling_per_replicate")
        != GPU_SECOND_CEILING_PER_REPLICATE_V1
        or compute.get("same_parameter_surface_all_arms") is not True
        or compute.get("failed_budget_reallocation") != "prohibited"
        or compute.get("direction_seeds_by_replicate_and_update")
        != expected_seed_sets
    ):
        raise RuntimeError("v1 equal-compute contract changed")
    expected_arms = arm_grid_v1()
    if plan["grid"] != expected_arms or len({x["arm_id"] for x in expected_arms}) != 8:
        raise RuntimeError("v1 arm grid changed")
    if plan["memory_bandwidth_contract"] != optimizer_memory_contract_v1():
        raise RuntimeError("v1 optimizer memory/bandwidth accounting changed")
    checkpoint = plan["checkpoint_contract"]
    initial = checkpoint.get("initial_checkpoint_sha256_by_arm_and_seed", {})
    if set(initial) != {arm["arm_id"] for arm in expected_arms}:
        raise RuntimeError("v1 initial checkpoint arm coverage changed")
    for arm in expected_arms:
        per_seed = initial[arm["arm_id"]]
        if set(per_seed) != {str(seed) for seed in REPLICATE_SEEDS_V1}:
            raise RuntimeError("v1 initial checkpoint seed coverage changed")
        for value in per_seed.values():
            _hex_sha_v1(value, "initial checkpoint")
    gates = plan["evaluation_gates"]
    if (
        gates.get("train_only_during_updates") is not True
        or gates.get("dev_and_ood_open_only_after_final_checkpoint_sealed") is not True
        or gates.get("protected_holdout_access") is not False
        or gates.get("ood_use") != "noninferiority_only_not_point_optimization"
    ):
        raise RuntimeError("v1 evaluation gates changed")
    failure = plan["failure_policy"]
    if (
        failure.get("nonfinite") != "abort_restore_exact_checkpoint_no_retry_seed"
        or failure.get("zero_pair_difference_variance")
        != "skip_without_optimizer_step_or_checkpoint_change"
        or failure.get("zero_optimizer_direction")
        != "skip_without_optimizer_step_or_checkpoint_change"
        or failure.get("replica_or_resume_mismatch")
        != "abort_restore_exact_checkpoint"
    ):
        raise RuntimeError("v1 failure policy changed")
    return {
        "status": (
            "launch_ready" if expected_launch
            else "sealed_cpu_preview_runtime_dependencies_pending"
        ),
        "content_sha256": plan["content_sha256_before_self_field"],
        "arm_count": len(expected_arms),
    }


def _hex_sha_v1(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"v1 {label} must be lowercase SHA-256")
    return value


def validate_run_receipt_v1(
    plan: Mapping[str, Any], receipt: Mapping[str, Any]
) -> dict[str, Any]:
    validate_preregistration_v1(plan, launch=True)
    fields = {
        "schema", "plan_content_sha256", "arm_id", "replicate_seed",
        "start_checkpoint_sha256", "updates", "final_checkpoint_sha256",
        "total_directions", "total_signed_candidates", "total_rollouts",
        "parameter_surface_identity_sha256", "update_budget_ratio",
        "charged_gpu_seconds", "useful_physical_gpus", "final_eval_requests",
        "train_only_during_updates", "dev_opened_after_training",
        "ood_opened_after_training", "protected_holdout_opened",
        "last_update_monotonic_ns", "training_sealed_monotonic_ns",
        "dev_opened_monotonic_ns", "ood_opened_monotonic_ns",
    }
    _strict_fields_v1(receipt, fields, "run receipt")
    if (
        receipt["schema"] != "fp32-es-optimizer-sigma-run-receipt-v1"
        or receipt["plan_content_sha256"] != plan["content_sha256_before_self_field"]
    ):
        raise RuntimeError("v1 run receipt plan identity changed")
    arms = {item["arm_id"]: item for item in plan["grid"]}
    arm = arms.get(receipt["arm_id"])
    if arm is None or receipt["replicate_seed"] not in REPLICATE_SEEDS_V1:
        raise RuntimeError("v1 receipt arm or replicate is not registered")
    if (
        receipt["total_directions"]
        != DIRECTIONS_PER_UPDATE_V1 * UPDATES_PER_REPLICATE_V1
        or receipt["total_signed_candidates"]
        != SIGNED_CANDIDATES_PER_UPDATE_V1 * UPDATES_PER_REPLICATE_V1
        or receipt["total_rollouts"] != ROLLOUTS_PER_REPLICATE_V1
        or receipt["update_budget_ratio"] != UPDATE_BUDGET_RATIO_V1
        or receipt["parameter_surface_identity_sha256"]
        != plan["parameter_surface"]["identity_sha256"]
    ):
        raise RuntimeError("v1 receipt native work, surface, or update budget changed")
    gpu_seconds = _finite_float_v1(
        receipt["charged_gpu_seconds"], "charged GPU seconds", positive=True
    )
    if gpu_seconds > GPU_SECOND_CEILING_PER_REPLICATE_V1:
        raise RuntimeError("v1 receipt exceeded the compute ceiling")
    if receipt["useful_physical_gpus"] != [0, 1, 2, 3]:
        raise RuntimeError("v1 receipt lacks useful activity on all GPUs")
    if (
        receipt["train_only_during_updates"] is not True
        or receipt["dev_opened_after_training"] is not True
        or receipt["ood_opened_after_training"] is not True
        or receipt["protected_holdout_opened"] is not False
    ):
        raise RuntimeError("v1 receipt crossed an evaluation gate")
    last_update_ns = _exact_int_v1(
        receipt["last_update_monotonic_ns"], "last update timestamp", minimum=1
    )
    sealed_ns = _exact_int_v1(
        receipt["training_sealed_monotonic_ns"], "training seal timestamp", minimum=1
    )
    dev_ns = _exact_int_v1(
        receipt["dev_opened_monotonic_ns"], "dev open timestamp", minimum=1
    )
    ood_ns = _exact_int_v1(
        receipt["ood_opened_monotonic_ns"], "OOD open timestamp", minimum=1
    )
    if not last_update_ns < sealed_ns < min(dev_ns, ood_ns):
        raise RuntimeError("v1 dev/OOD was opened before training was sealed")
    if receipt["final_eval_requests"] != plan["compute_contract"][
        "exact_final_eval_requests_per_replicate"
    ]:
        raise RuntimeError("v1 final evaluation compute changed")

    updates = receipt["updates"]
    if not isinstance(updates, list) or len(updates) != UPDATES_PER_REPLICATE_V1:
        raise RuntimeError("v1 receipt update count changed")
    previous_checkpoint = _hex_sha_v1(
        receipt["start_checkpoint_sha256"], "start checkpoint"
    )
    expected_start = plan["checkpoint_contract"][
        "initial_checkpoint_sha256_by_arm_and_seed"
    ][arm["arm_id"]][str(receipt["replicate_seed"])]
    if previous_checkpoint != expected_start:
        raise RuntimeError("v1 unregistered initial or resumed checkpoint")
    previous_optimizer_state = None
    for index, update in enumerate(updates):
        update_fields = {
            "update_index", "base_sigma", "directions", "signed_candidates",
            "rollouts", "scale_mode", "coefficient_mode", "optimizer",
            "candidate_scale_application_count",
            "estimator_inverse_scale_application_count", "noise_representation",
            "pair_difference_variance", "finite_gradient",
            "target_update_l2", "observed_update_l2", "update_norm_relative_error",
            "budget_ratio", "optimizer_state_dtype", "optimizer_step_before",
            "optimizer_step_after", "bias_correction_step",
            "pre_optimizer_state_sha256", "rollback_optimizer_state_sha256",
            "post_optimizer_state_sha256", "resume_checkpoint_sha256",
            "previous_checkpoint_sha256", "rollback_checkpoint_sha256",
            "candidate_checkpoint_sha256", "committed_checkpoint_sha256",
            "replica_checkpoint_sha256", "direction_seed_set_sha256",
            "train_panel_content_sha256", "useful_physical_gpus",
            "persistent_optimizer_host_tensor_bytes_per_replica",
            "persistent_optimizer_gpu_bytes_per_replica",
            "measured_host_memory_traffic_bytes_per_replica",
            "reduced_gradient_d2h_bytes_per_replica",
            "committed_runtime_h2d_bytes_per_replica",
            "population_candidate_restore_h2d_bytes_all_replicas",
            "checkpoint_logical_bytes_per_replica",
            "host_optimizer_elapsed_seconds",
            "achieved_host_bandwidth_bytes_per_second",
            "peak_phase_vram_bytes_by_gpu",
        }
        _strict_fields_v1(update, update_fields, f"update {index}")
        if (
            update["update_index"] != index
            or update["base_sigma"] != SIGMA_SCHEDULE_V1[index]
            or update["directions"] != DIRECTIONS_PER_UPDATE_V1
            or update["signed_candidates"] != SIGNED_CANDIDATES_PER_UPDATE_V1
            or update["rollouts"] != ROLLOUTS_PER_UPDATE_V1
            or update["scale_mode"] != arm["scale_mode"]
            or update["coefficient_mode"] != arm["coefficient_mode"]
            or update["optimizer"] != arm["optimizer"]
            or update["candidate_scale_application_count"] != 1
            or update["estimator_inverse_scale_application_count"] != 1
            or update["noise_representation"] != "unscaled_unit_standard_normal"
            or _finite_float_v1(
                update["pair_difference_variance"], "pair variance", positive=True
            ) <= 0.0
            or update["finite_gradient"] is not True
            or update["budget_ratio"] != UPDATE_BUDGET_RATIO_V1
            or update["optimizer_state_dtype"] != "torch.float32"
            or update["optimizer_step_before"] != index
            or update["optimizer_step_after"] != index + 1
            or update["bias_correction_step"]
            != (index + 1 if arm["optimizer"] == "adamw" else None)
            or update["useful_physical_gpus"] != [0, 1, 2, 3]
        ):
            raise RuntimeError(f"v1 update {index} algebra or compute changed")
        memory = plan["memory_bandwidth_contract"]
        optimizer_memory = memory["optimizers"][arm["optimizer"]]
        host_traffic = _exact_int_v1(
            update["measured_host_memory_traffic_bytes_per_replica"],
            "measured host memory traffic",
            minimum=1,
        )
        elapsed = _finite_float_v1(
            update["host_optimizer_elapsed_seconds"],
            "host optimizer elapsed seconds",
            positive=True,
        )
        bandwidth = _finite_float_v1(
            update["achieved_host_bandwidth_bytes_per_second"],
            "host bandwidth",
            positive=True,
        )
        peak_vram = update["peak_phase_vram_bytes_by_gpu"]
        if (
            update["persistent_optimizer_host_tensor_bytes_per_replica"]
            != optimizer_memory[
                "persistent_optimizer_host_tensor_bytes_per_replica"
            ]
            or update["persistent_optimizer_gpu_bytes_per_replica"] != 0
            or host_traffic
            < optimizer_memory[
                "algorithmic_minimum_host_memory_traffic_bytes_per_update_per_replica"
            ]
            or update["reduced_gradient_d2h_bytes_per_replica"]
            != memory["reduced_gradient_d2h_bytes_per_update_per_replica"]
            or update["committed_runtime_h2d_bytes_per_replica"]
            != memory["committed_runtime_h2d_bytes_per_update_per_replica"]
            or update["population_candidate_restore_h2d_bytes_all_replicas"]
            != memory[
                "fixed_population_candidate_plus_restore_h2d_bytes_per_update_all_replicas"
            ]
            or update["checkpoint_logical_bytes_per_replica"]
            != optimizer_memory["checkpoint_tensor_and_step_bytes_per_replica"]
            or not math.isclose(
                bandwidth,
                host_traffic / elapsed,
                rel_tol=1.0e-6,
                abs_tol=0.0,
            )
            or not isinstance(peak_vram, Mapping)
            or set(peak_vram) != {"0", "1", "2", "3"}
            or any(
                _exact_int_v1(value, "peak phase VRAM", minimum=1) <= 0
                for value in peak_vram.values()
            )
        ):
            raise RuntimeError("v1 optimizer memory or bandwidth receipt changed")
        target = _finite_float_v1(
            update["target_update_l2"], "target update L2", positive=True
        )
        observed = _finite_float_v1(
            update["observed_update_l2"], "observed update L2", positive=True
        )
        measured_error = abs(observed - target) / target
        reported_error = _finite_float_v1(
            update["update_norm_relative_error"], "update norm relative error"
        )
        if (
            reported_error < 0.0
            or not math.isclose(reported_error, measured_error, rel_tol=1e-9, abs_tol=1e-12)
            or reported_error > UPDATE_NORM_RELATIVE_TOLERANCE_V1
        ):
            raise RuntimeError("v1 unequal update norm budget")
        pre_state = _hex_sha_v1(
            update["pre_optimizer_state_sha256"], "pre optimizer state"
        )
        rollback_state = _hex_sha_v1(
            update["rollback_optimizer_state_sha256"], "rollback optimizer state"
        )
        post_state = _hex_sha_v1(
            update["post_optimizer_state_sha256"], "post optimizer state"
        )
        if rollback_state != pre_state or (
            previous_optimizer_state is not None and pre_state != previous_optimizer_state
        ):
            raise RuntimeError("v1 optimizer resume or rollback state mismatch")
        for key in (
            "resume_checkpoint_sha256", "previous_checkpoint_sha256",
            "rollback_checkpoint_sha256", "candidate_checkpoint_sha256",
            "committed_checkpoint_sha256", "direction_seed_set_sha256",
            "train_panel_content_sha256",
        ):
            _hex_sha_v1(update[key], key)
        expected_direction_sha = canonical_sha256_v1(
            plan["compute_contract"]["direction_seeds_by_replicate_and_update"]
            [str(receipt["replicate_seed"])][index]
        )
        if (
            update["resume_checkpoint_sha256"] != previous_checkpoint
            or update["previous_checkpoint_sha256"] != previous_checkpoint
            or update["rollback_checkpoint_sha256"] != previous_checkpoint
            or update["candidate_checkpoint_sha256"]
            != update["committed_checkpoint_sha256"]
            or update["replica_checkpoint_sha256"]
            != [update["committed_checkpoint_sha256"]] * REPLICA_COUNT_V1
            or update["direction_seed_set_sha256"] != expected_direction_sha
            or update["train_panel_content_sha256"]
            != plan["inputs"]["train_panel"]["content_sha256"]
        ):
            raise RuntimeError("v1 checkpoint resume, rollback, or replica identity mismatch")
        previous_checkpoint = update["committed_checkpoint_sha256"]
        previous_optimizer_state = post_state
    if _hex_sha_v1(receipt["final_checkpoint_sha256"], "final checkpoint") != previous_checkpoint:
        raise RuntimeError("v1 final checkpoint chain changed")
    return {
        "status": "valid_complete_run_receipt",
        "arm_id": arm["arm_id"],
        "replicate_seed": receipt["replicate_seed"],
        "final_checkpoint_sha256": previous_checkpoint,
    }


def validate_grid_receipts_v1(
    plan: Mapping[str, Any], receipts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected = {
        (arm["arm_id"], seed)
        for arm in plan["grid"]
        for seed in REPLICATE_SEEDS_V1
    }
    observed = {}
    for receipt in receipts:
        result = validate_run_receipt_v1(plan, receipt)
        key = (result["arm_id"], result["replicate_seed"])
        if key in observed:
            raise RuntimeError("v1 duplicated arm/replicate receipt")
        observed[key] = result
    if set(observed) != expected:
        raise RuntimeError("v1 grid receipt coverage is incomplete")
    return {
        "status": "valid_complete_nonadaptive_grid",
        "receipt_count": len(observed),
        "total_rollouts": len(observed) * ROLLOUTS_PER_REPLICATE_V1,
    }
