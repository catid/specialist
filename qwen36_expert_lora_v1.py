#!/usr/bin/env python3
"""Expert-aware LoRA attachment and fail-closed scope auditing for Qwen3.6.

The architecture contract, rather than suffix guesses, supplies every shared
expert module and routed expert parameter name.  This module deliberately has
no model-loading or dataset-loading side effects.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ARCHITECTURE_SCHEMA = "qwen36-low-regression-architecture-contract-v1"
DEFAULT_ARCHITECTURE_CONTRACT = Path(
    "training_protocol/qwen36_architecture_contract_v1.json"
)
ADAPTER_NAME = "default"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_architecture_contract(path: Path = DEFAULT_ARCHITECTURE_CONTRACT) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "architecture contract must be an object")
    claimed = value.get("content_sha256_before_self_field")
    body = copy.deepcopy(value)
    body.pop("content_sha256_before_self_field", None)
    _require(claimed == canonical_sha256(body), "architecture contract self hash changed")
    _require(value.get("schema") == ARCHITECTURE_SCHEMA, "architecture contract schema changed")
    acceptance = value.get("acceptance", {})
    for key in (
        "actual_module_and_parameter_names_verified",
        "checkpoint_geometry_verified",
        "fused_routed_expert_storage_verified",
        "literal_full_name_targets_only",
        "text_only_loader_and_prefix_conversion_verified",
        "vision_and_mtp_exclusion_decided",
    ):
        _require(acceptance.get(key) is True, f"architecture gate is not accepted: {key}")
    return value


@dataclass(frozen=True)
class ExpertLoraSpec:
    shared_targets: tuple[str, ...]
    routed_targets: tuple[str, ...]
    shared_rank: int
    shared_alpha: int
    routed_rank: int | None
    routed_alpha: int | None
    dropout: float = 0.0

    @property
    def all_targets(self) -> tuple[str, ...]:
        return self.shared_targets + self.routed_targets


def spec_from_contract(
    contract: dict[str, Any],
    *,
    routed_rank: int | None = None,
    shared_rank: int | None = None,
) -> ExpertLoraSpec:
    target = contract["architecture"]["adapter_target_contract"]
    _require(target.get("matching_mode") == "literal_full_name_membership_only", "nonliteral targeting")
    _require(target.get("regex_targeting_allowed") is False, "regex targeting unexpectedly allowed")
    _require(target.get("suffix_guessing_allowed") is False, "suffix guessing unexpectedly allowed")
    selected_shared_rank = target["shared_rank"] if shared_rank is None else shared_rank
    selected_routed_rank = target["routed_rank"] if routed_rank is None else routed_rank
    _require(selected_shared_rank in (8, 16), "shared expert rank must be 8 or 16")
    _require(selected_routed_rank in (2, 4), "routed expert rank must be 2 or 4")
    shared = tuple(target["shared_expert_target_modules"])
    routed = tuple(target["routed_expert_target_parameters"])
    _require(len(shared) == 120 and len(set(shared)) == 120, "shared target inventory changed")
    _require(len(routed) == 80 and len(set(routed)) == 80, "routed target inventory changed")
    _require(set(shared).isdisjoint(routed), "shared and routed targets overlap")
    return ExpertLoraSpec(
        shared_targets=shared,
        routed_targets=routed,
        shared_rank=selected_shared_rank,
        shared_alpha=selected_shared_rank,
        routed_rank=selected_routed_rank,
        routed_alpha=selected_routed_rank,
    )


def shared_only_spec_from_contract(
    contract: dict[str, Any], *, shared_rank: int | None = None
) -> ExpertLoraSpec:
    complete = spec_from_contract(contract, shared_rank=shared_rank)
    return ExpertLoraSpec(
        shared_targets=complete.shared_targets,
        routed_targets=(),
        shared_rank=complete.shared_rank,
        shared_alpha=complete.shared_alpha,
        routed_rank=None,
        routed_alpha=None,
    )


def validate_preattach_model(model: Any, spec: ExpertLoraSpec) -> dict[str, Any]:
    modules = dict(model.named_modules())
    parameters = dict(model.named_parameters())
    observed_shared = []
    observed_routed = []

    for name in spec.shared_targets:
        _require(name in modules, f"missing literal shared target module: {name}")
        module = modules[name]
        _require(type(module).__name__ == "Linear", f"shared target is not Linear: {name}")
        _require(hasattr(module, "weight") and module.weight.ndim == 2, f"invalid shared weight: {name}")
        suffix_matches = [candidate for candidate in modules if candidate == name or candidate.endswith("." + name)]
        _require(suffix_matches == [name], f"shared PEFT suffix match is ambiguous: {name}")
        observed_shared.append(
            {"name": name, "weight_shape": list(module.weight.shape), "elements": module.weight.numel()}
        )

    for name in spec.routed_targets:
        _require(name in parameters, f"missing literal routed target parameter: {name}")
        parameter = parameters[name]
        _require(parameter.ndim == 3, f"routed target is not fused 3D storage: {name}")
        _require(parameter.shape[0] == 256, f"routed target expert count changed: {name}")
        suffix_matches = [candidate for candidate in parameters if candidate == name or candidate.endswith("." + name)]
        _require(suffix_matches == [name], f"routed PEFT suffix match is ambiguous: {name}")
        observed_routed.append(
            {"name": name, "shape": list(parameter.shape), "elements": parameter.numel()}
        )

    return {
        "shared": observed_shared,
        "routed": observed_routed,
        "identity_sha256": canonical_sha256({"shared": observed_shared, "routed": observed_routed}),
    }


def make_lora_config(spec: ExpertLoraSpec):
    from peft import LoraConfig

    rank_pattern = {
        name: spec.routed_rank for name in spec.routed_targets
    }
    alpha_pattern = {
        name: spec.routed_alpha for name in spec.routed_targets
    }
    return LoraConfig(
        task_type="CAUSAL_LM",
        r=spec.shared_rank,
        lora_alpha=spec.shared_alpha,
        target_modules=list(spec.shared_targets),
        target_parameters=list(spec.routed_targets) or None,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
        lora_dropout=spec.dropout,
        bias="none",
        init_lora_weights=True,
        use_rslora=False,
        use_dora=False,
        modules_to_save=None,
    )


def _original_target_from_wrapper_path(wrapper_path: str, parameter_name: str | None) -> str:
    prefix = "base_model.model."
    _require(wrapper_path.startswith(prefix), f"unexpected PEFT wrapper prefix: {wrapper_path}")
    parts = [part for part in wrapper_path[len(prefix):].split(".") if part != "base_layer"]
    _require(parts and all(parts), f"invalid PEFT wrapper path: {wrapper_path}")
    if parameter_name is not None:
        parts.append(parameter_name)
    return ".".join(parts)


def audit_postattach_scope(model: Any, spec: ExpertLoraSpec, *, adapter_name: str = ADAPTER_NAME) -> dict[str, Any]:
    from peft.tuners.lora.layer import LoraLayer, ParamWrapper

    expected = set(spec.all_targets)
    wrapper_rows = []
    allowed_parameter_ids: set[int] = set()
    observed: set[str] = set()

    for wrapper_path, module in model.named_modules():
        if not isinstance(module, LoraLayer) or adapter_name not in module.lora_A:
            continue
        parameter_name = module.parameter_name if isinstance(module, ParamWrapper) else None
        target = _original_target_from_wrapper_path(wrapper_path, parameter_name)
        _require(target in expected, f"LoRA attached outside the exact contract: {target}")
        _require(target not in observed, f"duplicate LoRA wrapper for target: {target}")
        observed.add(target)
        expected_rank = spec.routed_rank if target in spec.routed_targets else spec.shared_rank
        expected_alpha = spec.routed_alpha if target in spec.routed_targets else spec.shared_alpha
        _require(module.r[adapter_name] == expected_rank, f"wrong LoRA rank for {target}")
        _require(module.lora_alpha[adapter_name] == expected_alpha, f"wrong LoRA alpha for {target}")
        dropout = module.lora_dropout[adapter_name]
        _require(
            type(dropout).__name__ == "Identity",
            f"nonzero or unknown LoRA dropout for {target}",
        )
        _require(module.lora_bias[adapter_name] is False, f"LoRA bias enabled for {target}")
        _require(module.use_rslora[adapter_name] is False, f"RSLoRA enabled for {target}")
        _require(module.use_dora[adapter_name] is False, f"DoRA enabled for {target}")
        _require(
            module.scaling[adapter_name] == expected_alpha / expected_rank,
            f"wrong classic LoRA scaling for {target}",
        )
        _require(not module.disable_adapters, f"LoRA adapters are disabled for {target}")
        _require(
            module.active_adapters == [adapter_name],
            f"unexpected active LoRA adapters for {target}",
        )
        _require(not module.merged, f"LoRA adapter is already merged for {target}")
        a_parameters = list(module.lora_A[adapter_name].parameters())
        b_parameters = list(module.lora_B[adapter_name].parameters())
        _require(len(a_parameters) == 1 and len(b_parameters) == 1, f"unexpected LoRA tensor layout: {target}")
        allowed_parameter_ids.update((id(a_parameters[0]), id(b_parameters[0])))
        wrapper_rows.append(
            {
                "target": target,
                "kind": "routed_parameter" if target in spec.routed_targets else "shared_module",
                "rank": module.r[adapter_name],
                "alpha": module.lora_alpha[adapter_name],
                "scaling": module.scaling[adapter_name],
                "dropout": 0.0,
                "bias": False,
                "use_rslora": False,
                "use_dora": False,
                "active_and_unmerged": True,
                "a_shape": list(a_parameters[0].shape),
                "b_shape": list(b_parameters[0].shape),
                "trainable_elements": a_parameters[0].numel() + b_parameters[0].numel(),
            }
        )

    _require(observed == expected, f"post-attach target set mismatch: missing={sorted(expected - observed)} extra={sorted(observed - expected)}")
    trainable_rows = [
        {"name": name, "shape": list(parameter.shape), "elements": parameter.numel(), "dtype": str(parameter.dtype)}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_ids = {
        id(parameter) for _, parameter in model.named_parameters() if parameter.requires_grad
    }
    _require(trainable_ids == allowed_parameter_ids, "a non-LoRA tensor is trainable or a LoRA tensor is frozen")
    _require(len(trainable_rows) == 2 * len(expected), "trainable LoRA tensor cardinality changed")
    wrapper_rows.sort(key=lambda row: row["target"])
    trainable_rows.sort(key=lambda row: row["name"])
    total = sum(row["elements"] for row in trainable_rows)
    return {
        "adapter_name": adapter_name,
        "target_count": len(expected),
        "shared_target_count": len(spec.shared_targets),
        "routed_target_count": len(spec.routed_targets),
        "trainable_tensor_count": len(trainable_rows),
        "trainable_elements": total,
        "wrappers": wrapper_rows,
        "trainable_parameters": trainable_rows,
        "identity_sha256": canonical_sha256({"wrappers": wrapper_rows, "trainable_parameters": trainable_rows}),
    }
