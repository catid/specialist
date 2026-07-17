#!/usr/bin/env python3
"""Fail-closed, metadata-only autograd contract for LoRA ES V83.

The contract deliberately does not walk the model object graph.  It inspects
the worker's explicit ES ownership registries, direct built-in containers
registered on the worker, and optional transition inputs/results.  Tensor
contents are never read and tensors are never copied or cloned.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Callable

import torch
from torch.utils._python_dispatch import TorchDispatchMode


SCHEMA_V83 = "eggroll-es-autograd-free-surface-v83"
TRANSITION_SCHEMA_V83 = "eggroll-es-autograd-free-transition-v83"
MAX_REGISTERED_CONTAINER_NODES_V83 = 100_000
MAX_REGISTERED_CONTAINER_DEPTH_V83 = 12


class AutogradContractViolationV83(RuntimeError):
    """A fail-closed violation of the V83 autograd ownership contract."""


def canonical_sha256_v83(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _container_items_v83(value: Any):
    if isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: repr(item)):
            yield f"[{key!r}]", value[key]
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield f"[{index}]", item
    elif isinstance(value, (set, frozenset)):
        for index, item in enumerate(sorted(value, key=repr)):
            yield f"[set:{index}]", item


def _is_builtin_container_v83(value: Any) -> bool:
    return isinstance(value, (Mapping, list, tuple, set, frozenset))


def _is_torch_optimizer_v83(value: Any) -> bool:
    return isinstance(value, torch.optim.Optimizer)


def _looks_like_optimizer_state_dict_v83(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and "state" in value
        and "param_groups" in value
        and isinstance(value.get("state"), Mapping)
        and isinstance(value.get("param_groups"), (list, tuple))
    )


def _registered_state_inventory_v83(worker: Any):
    """Inspect direct worker state and built-in containers, never model objects."""
    try:
        roots = vars(worker)
    except TypeError as error:
        raise AutogradContractViolationV83(
            "v83 worker state has no explicit attribute registry"
        ) from error

    tensors: list[tuple[str, torch.Tensor]] = []
    visited_containers: set[int] = set()
    stack = [(f"worker.{name}", value, 0) for name, value in roots.items()]
    nodes = 0
    optimizer_count = 0
    optimizer_state_dict_count = 0
    while stack:
        path, value, depth = stack.pop()
        nodes += 1
        if nodes > MAX_REGISTERED_CONTAINER_NODES_V83:
            raise AutogradContractViolationV83(
                "v83 registered worker container graph exceeded its sealed bound"
            )
        if _is_torch_optimizer_v83(value):
            optimizer_count += 1
            raise AutogradContractViolationV83(
                f"v83 hidden torch optimizer authority registered at {path}"
            )
        if _looks_like_optimizer_state_dict_v83(value):
            optimizer_state_dict_count += 1
            raise AutogradContractViolationV83(
                f"v83 hidden optimizer state-dict authority registered at {path}"
            )
        if isinstance(value, torch.Tensor):
            tensors.append((path, value))
            continue
        if not _is_builtin_container_v83(value):
            # In particular, do not recurse through the vLLM/model/manager
            # object graph.  Those are not worker-owned ES state registries.
            continue
        token = id(value)
        if token in visited_containers:
            continue
        visited_containers.add(token)
        if depth >= MAX_REGISTERED_CONTAINER_DEPTH_V83:
            raise AutogradContractViolationV83(
                f"v83 registered worker container nesting exceeded its bound at {path}"
            )
        for suffix, item in _container_items_v83(value):
            stack.append((f"{path}{suffix}", item, depth + 1))
    return {
        "tensors": tensors,
        "visited_container_count": len(visited_containers),
        "visited_node_count": nodes,
        "torch_optimizer_instance_count": optimizer_count,
        "optimizer_state_dict_registry_count": optimizer_state_dict_count,
    }


def _explicit_surfaces_v83(
    worker: Any,
    extra_surfaces: Mapping[str, Any] | None,
):
    surfaces: dict[str, Any] = {}
    for label, attribute in (
        ("master.canonical", "_v41_master"),
        ("runtime.lora_views", "_v71_runtime_views"),
        ("runtime.lora_parents", "_v71_runtime_parents"),
        ("runtime.base_tensors", "_v71_base_tensors"),
        ("runtime.pinned_publication_bank", "_v81a_bank"),
    ):
        value = getattr(worker, attribute, None)
        if value is not None:
            surfaces[label] = value

    pending = getattr(worker, "_v41_pending_update", None)
    if isinstance(pending, Mapping):
        for key in ("rollback_master", "candidate_master"):
            if pending.get(key) is not None:
                surfaces[f"update.pending.{key}"] = pending[key]
    committed = getattr(worker, "_v41_committed_rollback", None)
    if isinstance(committed, Mapping) and committed.get("rollback_master") is not None:
        surfaces["update.committed.rollback_master"] = committed[
            "rollback_master"
        ]
    if extra_surfaces is not None:
        if not isinstance(extra_surfaces, Mapping):
            raise AutogradContractViolationV83(
                "v83 extra transition surfaces are not a mapping"
            )
        for label, value in extra_surfaces.items():
            label = str(label)
            if not label:
                raise AutogradContractViolationV83(
                    "v83 extra transition surface label is empty"
                )
            surfaces[f"transition.{label}"] = value
    return surfaces


def _tensor_leaves_v83(value: Any, root: str):
    stack = [(root, value, 0)]
    visited: set[int] = set()
    nodes = 0
    while stack:
        path, item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_REGISTERED_CONTAINER_NODES_V83:
            raise AutogradContractViolationV83(
                "v83 explicit tensor surface exceeded its sealed node bound"
            )
        if isinstance(item, torch.Tensor):
            yield path, item
            continue
        if not _is_builtin_container_v83(item):
            continue
        token = id(item)
        if token in visited:
            continue
        visited.add(token)
        if depth >= MAX_REGISTERED_CONTAINER_DEPTH_V83:
            raise AutogradContractViolationV83(
                f"v83 explicit tensor surface nesting exceeded its bound at {path}"
            )
        for suffix, child in _container_items_v83(item):
            stack.append((f"{path}{suffix}", child, depth + 1))


def _tensor_record_v83(labels: list[str], tensor: torch.Tensor):
    if tensor.requires_grad:
        raise AutogradContractViolationV83(
            f"v83 tensor requires_grad=True at {labels[0]}"
        )
    if tensor.grad is not None:
        raise AutogradContractViolationV83(
            f"v83 tensor retained a non-None .grad at {labels[0]}"
        )
    if tensor.grad_fn is not None:
        raise AutogradContractViolationV83(
            f"v83 tensor retained an autograd graph at {labels[0]}"
        )
    if not tensor.is_leaf:
        raise AutogradContractViolationV83(
            f"v83 tensor is not an autograd leaf at {labels[0]}"
        )
    return {
        "labels": sorted(labels),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device_type": tensor.device.type,
        "elements": int(tensor.numel()),
        "bytes": int(tensor.numel() * tensor.element_size()),
        "requires_grad": False,
        "grad_is_none": True,
        "grad_fn_is_none": True,
        "is_leaf": True,
    }


def audit_autograd_surfaces_v83(
    worker: Any,
    phase: str,
    *,
    extra_surfaces: Mapping[str, Any] | None = None,
    require_grad_disabled: bool = False,
):
    """Return a content-free receipt or reject the worker state fail-closed."""
    phase = str(phase)
    if not phase:
        raise AutogradContractViolationV83("v83 audit phase is empty")
    grad_enabled = bool(torch.is_grad_enabled())
    if require_grad_disabled and grad_enabled:
        raise AutogradContractViolationV83(
            f"v83 autograd was enabled during guarded phase {phase}"
        )

    registered = _registered_state_inventory_v83(worker)
    references: list[tuple[str, torch.Tensor]] = list(registered["tensors"])
    surfaces = _explicit_surfaces_v83(worker, extra_surfaces)
    for label, value in surfaces.items():
        references.extend(_tensor_leaves_v83(value, label))

    aliases: dict[int, dict[str, Any]] = {}
    for label, tensor in references:
        token = id(tensor)
        if token not in aliases:
            aliases[token] = {"tensor": tensor, "labels": []}
        aliases[token]["labels"].append(label)
    records = [
        _tensor_record_v83(item["labels"], item["tensor"])
        for item in aliases.values()
    ]
    records.sort(key=lambda item: item["labels"])
    receipt = {
        "schema": SCHEMA_V83,
        "phase": phase,
        "passed": True,
        "torch_grad_enabled_at_audit": grad_enabled,
        "grad_disabled_required": bool(require_grad_disabled),
        "explicit_surface_names": sorted(surfaces),
        "surface_reference_count": len(references),
        "unique_tensor_count": len(records),
        "unique_tensor_elements": sum(item["elements"] for item in records),
        "unique_tensor_bytes": sum(item["bytes"] for item in records),
        "tensors": records,
        "all_requires_grad_false": True,
        "all_grad_none": True,
        "all_grad_fn_none": True,
        "all_autograd_leaves": True,
        "torch_optimizer_instance_count": registered[
            "torch_optimizer_instance_count"
        ],
        "optimizer_state_dict_registry_count": registered[
            "optimizer_state_dict_registry_count"
        ],
        "registered_container_count": registered["visited_container_count"],
        "registered_node_count": registered["visited_node_count"],
        "arbitrary_object_graph_traversed": False,
        "tensor_content_read_bytes": 0,
        "tensor_clone_bytes": 0,
        "full_model_clone_bytes": 0,
        "metadata_only": True,
    }
    receipt["receipt_sha256"] = canonical_sha256_v83(receipt)
    return receipt


class RejectGradEnabledTensorOpsV83(TorchDispatchMode):
    """Reject tensor operations if transition code re-enables autograd."""

    def __init__(self):
        super().__init__()
        self.violation_count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if torch.is_grad_enabled():
            self.violation_count += 1
            raise AutogradContractViolationV83(
                "v83 guarded transition attempted a tensor operation with "
                "autograd enabled"
            )
        return func(*args, **(kwargs or {}))


def run_autograd_free_transition_v83(
    worker: Any,
    phase: str,
    operation: Callable[[], Any],
    *,
    before_surfaces: Mapping[str, Any] | None = None,
    after_surfaces: Callable[[Any], Mapping[str, Any] | None] | None = None,
):
    """Execute one transition under no-grad plus an enabled-op tripwire."""
    if not callable(operation):
        raise TypeError("v83 transition operation is not callable")
    phase = str(phase)
    before = audit_autograd_surfaces_v83(
        worker,
        f"{phase}:before",
        extra_surfaces=before_surfaces,
        require_grad_disabled=False,
    )
    caller_grad_enabled = bool(torch.is_grad_enabled())
    tripwire = RejectGradEnabledTensorOpsV83()
    try:
        with torch.no_grad(), tripwire:
            if torch.is_grad_enabled():
                raise AutogradContractViolationV83(
                    "v83 no-grad guard failed at transition entry"
                )
            result = operation()
            if tripwire.violation_count:
                raise AutogradContractViolationV83(
                    "v83 guarded transition suppressed a grad-enabled tensor "
                    "operation violation"
                )
            if torch.is_grad_enabled():
                raise AutogradContractViolationV83(
                    "v83 transition leaked enabled autograd"
                )
            extra_after = (
                after_surfaces(result) if after_surfaces is not None else None
            )
            after = audit_autograd_surfaces_v83(
                worker,
                f"{phase}:after",
                extra_surfaces=extra_after,
                require_grad_disabled=True,
            )
    except BaseException:
        # A callback that leaks global grad state without performing a tensor
        # op is rejected here as well.  torch.no_grad restores caller state.
        if torch.is_grad_enabled() != caller_grad_enabled:
            torch.set_grad_enabled(caller_grad_enabled)
        raise
    receipt = {
        "schema": TRANSITION_SCHEMA_V83,
        "phase": phase,
        "passed": True,
        "caller_grad_enabled": caller_grad_enabled,
        "guard_mode": "torch.no_grad+reject_grad_enabled_tensor_ops",
        "inference_mode_used": False,
        "inference_mode_omission_reason": (
            "preserve accepted tensor version-counter invariants"
        ),
        "grad_enabled_tensor_op_violation_count": tripwire.violation_count,
        "before_receipt_sha256": before["receipt_sha256"],
        "after_receipt_sha256": after["receipt_sha256"],
        "enforcement_tensor_content_read_bytes": 0,
        "enforcement_tensor_clone_bytes": 0,
        "full_model_clone_bytes": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256_v83(receipt)
    return result, receipt


def validate_autograd_receipt_v83(receipt: Mapping[str, Any]):
    if not isinstance(receipt, Mapping) or receipt.get("schema") != SCHEMA_V83:
        raise AutogradContractViolationV83("v83 receipt schema changed")
    observed = dict(receipt)
    claimed = observed.pop("receipt_sha256", None)
    if claimed != canonical_sha256_v83(observed):
        raise AutogradContractViolationV83("v83 receipt identity changed")
    required_true = (
        "passed",
        "all_requires_grad_false",
        "all_grad_none",
        "all_grad_fn_none",
        "all_autograd_leaves",
        "metadata_only",
    )
    if any(receipt.get(key) is not True for key in required_true):
        raise AutogradContractViolationV83("v83 receipt lost a required proof")
    required_zero = (
        "torch_optimizer_instance_count",
        "optimizer_state_dict_registry_count",
        "tensor_content_read_bytes",
        "tensor_clone_bytes",
        "full_model_clone_bytes",
    )
    if any(receipt.get(key) != 0 for key in required_zero):
        raise AutogradContractViolationV83("v83 receipt gained hidden authority")
    if receipt.get("arbitrary_object_graph_traversed") is not False:
        raise AutogradContractViolationV83("v83 receipt traversed model state")
    return dict(receipt)


__all__ = [
    "AutogradContractViolationV83",
    "RejectGradEnabledTensorOpsV83",
    "SCHEMA_V83",
    "TRANSITION_SCHEMA_V83",
    "audit_autograd_surfaces_v83",
    "canonical_sha256_v83",
    "run_autograd_free_transition_v83",
    "validate_autograd_receipt_v83",
]
