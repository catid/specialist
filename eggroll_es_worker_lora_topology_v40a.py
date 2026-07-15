#!/usr/bin/env python3
"""Train-only vLLM LoRA buffer topology and mutation audit for V40A."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open

from eggroll_es_worker_v2 import ExactAuditWorkerExtension


PACKED_MODULES = {
    "q_proj": ("qkv_proj", (0,)),
    "k_proj": ("qkv_proj", (1,)),
    "v_proj": ("qkv_proj", (2,)),
    "gate_proj": ("gate_up_proj", (0,)),
    "up_proj": ("gate_up_proj", (1,)),
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
    "in_proj_z": ("in_proj_qkvz", (3,)),
    "in_proj_b": ("in_proj_ba", (0,)),
    "in_proj_a": ("in_proj_ba", (1,)),
}


def _canonical_sha256(value) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def _tensor_record(tensor: torch.Tensor) -> dict:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "elements": int(tensor.numel()),
        "sha256": _tensor_sha256(tensor),
        "device": str(tensor.device),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "storage_data_ptr": int(tensor.untyped_storage().data_ptr()),
        "storage_bytes": int(tensor.untyped_storage().nbytes()),
        "contiguous": bool(tensor.is_contiguous()),
    }


def _source_parts(key: str) -> tuple[str, str]:
    prefix = "base_model.model."
    if not key.startswith(prefix) or not key.endswith(".weight"):
        raise RuntimeError(f"v40a unexpected PEFT key: {key}")
    body = key[len(prefix):-len(".weight")]
    if body.endswith(".lora_A"):
        return body[:-len(".lora_A")], "A"
    if body.endswith(".lora_B"):
        return body[:-len(".lora_B")], "B"
    raise RuntimeError(f"v40a unexpected PEFT key: {key}")


def _runtime_target(logical: str) -> tuple[str, tuple[int, ...]]:
    prefix, _dot, leaf = logical.rpartition(".")
    replacement, slices = PACKED_MODULES.get(leaf, (leaf, (0,)))
    return f"{prefix}.{replacement}", tuple(slices)


def _suffix_match(modules: dict, target: str):
    matches = [(name, module) for name, module in modules.items()
               if name == target or name.endswith(f".{target}")]
    if len(matches) != 1:
        raise RuntimeError(
            f"v40a runtime module mapping is not unique for {target}: "
            f"{[name for name, _module in matches]}"
        )
    return matches[0]


def _manager(worker):
    facade = getattr(worker.model_runner, "lora_manager", None)
    manager = getattr(facade, "_adapter_manager", None)
    if manager is None:
        raise RuntimeError("v40a vLLM LoRA manager is unavailable")
    if list(manager.lora_index_to_id) != [1]:
        raise RuntimeError(
            f"v40a expected adapter 1 in sole slot: {manager.lora_index_to_id}"
        )
    return manager


def _base_identity(modules: dict, runtime_names: set[str]) -> dict:
    records = []
    seen = set()
    for name in sorted(runtime_names):
        module = modules[name]
        weight = getattr(getattr(module, "base_layer", None), "weight", None)
        if not isinstance(weight, torch.Tensor):
            raise RuntimeError(f"v40a wrapper has no base_layer.weight: {name}")
        identity = (int(weight.untyped_storage().data_ptr()), int(weight.storage_offset()))
        if identity in seen:
            continue
        seen.add(identity)
        records.append({"runtime_module": name, **_tensor_record(weight)})
    compact = [{key: value for key, value in row.items()
                if key not in {"storage_data_ptr"}}
               for row in records]
    return {
        "tensor_count": len(records),
        "elements": sum(row["elements"] for row in records),
        "bytes": sum(row["elements"] * torch.tensor([], dtype=getattr(
            torch, row["dtype"].split(".")[-1])).element_size() for row in records),
        "inventory_sha256": _canonical_sha256(compact),
        "tensors": records,
    }


class LoRATopologyWorkerExtensionV40A(ExactAuditWorkerExtension):
    """Expose exact resident LoRA slot inventory without reading any dataset."""

    def runtime_identity_v40a(self):
        return {
            "schema": "lora-topology-worker-identity-v40a",
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_current_device": (
                int(torch.cuda.current_device()) if torch.cuda.is_available() else None
            ),
        }

    def inventory_lora_topology_v40a(self, adapter_file, expected_file_sha256):
        adapter_file = Path(adapter_file).resolve()
        raw = adapter_file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != str(expected_file_sha256):
            raise RuntimeError("v40a adapter file identity changed")
        manager = _manager(self)
        modules = manager.modules
        source = []
        mapping = []
        runtime_names = set()
        with safe_open(adapter_file, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if len(keys) != 70:
                raise RuntimeError("v40a PEFT tensor inventory changed")
            for key in keys:
                tensor = handle.get_tensor(key)
                logical, side = _source_parts(key)
                target, slices = _runtime_target(logical)
                runtime_name, module = _suffix_match(modules, target)
                runtime_names.add(runtime_name)
                if len(module.lora_a_stacked) != len(module.lora_b_stacked):
                    raise RuntimeError("v40a A/B runtime slice count differs")
                if max(slices) >= len(module.lora_a_stacked):
                    raise RuntimeError(
                        f"v40a runtime slice mapping exceeds {runtime_name}"
                    )
                source_record = {
                    "peft_key": key, "logical_module": logical, "side": side,
                    "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                    "elements": int(tensor.numel()), "sha256": _tensor_sha256(tensor),
                }
                source.append(source_record)
                views = []
                for slice_index in slices:
                    stacked = (module.lora_a_stacked if side == "A"
                               else module.lora_b_stacked)[slice_index]
                    view = stacked[0, 0]
                    views.append({
                        "slice_index": int(slice_index),
                        "parent_shape": list(stacked.shape),
                        **_tensor_record(view),
                    })
                mapping.append({
                    **source_record, "runtime_module": runtime_name,
                    "slot": 0, "runtime_views": views,
                    "runtime_output_slices": list(module.output_slices),
                })

        # Verify source-to-runtime values. vLLM folds alpha/rank=2 into B.
        for item in mapping:
            with safe_open(adapter_file, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(item["peft_key"])
            module = modules[item["runtime_module"]]
            slices = [view["slice_index"] for view in item["runtime_views"]]
            if item["side"] == "A":
                expected = tensor.to(dtype=torch.bfloat16)
                for index in slices:
                    actual = module.lora_a_stacked[index][0, 0]
                    if actual.shape != expected.shape or not torch.equal(actual.cpu(), expected):
                        raise RuntimeError(f"v40a A mapping mismatch: {item['peft_key']}")
            else:
                expected = (tensor * 2.0).to(dtype=torch.bfloat16)
                if len(slices) == 1:
                    actual = module.lora_b_stacked[slices[0]][0, 0]
                    if actual.shape != expected.shape or not torch.equal(actual.cpu(), expected):
                        raise RuntimeError(f"v40a B mapping mismatch: {item['peft_key']}")
                else:
                    sizes = [int(module.output_slices[index]) for index in slices]
                    chunks = list(torch.split(expected, sizes, dim=0))
                    if len(chunks) != len(slices):
                        raise RuntimeError("v40a packed B split count changed")
                    for index, chunk in zip(slices, chunks, strict=True):
                        actual = module.lora_b_stacked[index][0, 0]
                        if actual.shape != chunk.shape or not torch.equal(actual.cpu(), chunk):
                            raise RuntimeError(f"v40a packed B mismatch: {item['peft_key']}")

        runtime_views = []
        for name in sorted(runtime_names):
            module = modules[name]
            for side, stacked_values in (("A", module.lora_a_stacked),
                                         ("B", module.lora_b_stacked)):
                for index, stacked in enumerate(stacked_values):
                    view = stacked[0, 0]
                    runtime_views.append({
                        "runtime_module": name, "side": side,
                        "slice_index": index, "slot": 0,
                        "parent_shape": list(stacked.shape),
                        "nonzero_elements": int(torch.count_nonzero(view).item()),
                        **_tensor_record(view),
                    })
        base = _base_identity(modules, runtime_names)
        result = {
            "schema": "vllm-lora-topology-inventory-v40a",
            "pid": os.getpid(), "adapter_id": 1, "slot": 0,
            "manager_type": type(manager).__name__,
            "packed_modules_mapping": manager.packed_modules_mapping,
            "peft_tensor_count": len(source),
            "peft_elements": sum(item["elements"] for item in source),
            "peft_dtype_counts": {"torch.float32": len(source)},
            "runtime_module_count": len(runtime_names),
            "runtime_view_count": len(runtime_views),
            "runtime_active_allocated_elements": sum(
                item["elements"] for item in runtime_views
            ),
            "runtime_active_nonzero_elements": sum(
                item["nonzero_elements"] for item in runtime_views
            ),
            "runtime_dtypes": sorted({item["dtype"] for item in runtime_views}),
            "source": source, "mapping": mapping,
            "runtime_views": runtime_views,
            "base_layer_weights": base,
        }
        compact = dict(result)
        result["content_sha256_before_self_field"] = _canonical_sha256(compact)
        self._v40a_inventory = result
        self._v40a_base_identity = base
        self._v40a_runtime_names = runtime_names
        return result

    def mutate_one_lora_element_v40a(self, delta=1.0):
        if not hasattr(self, "_v40a_inventory"):
            raise RuntimeError("v40a inventory must precede mutation")
        manager = _manager(self)
        target = "model.layers.23.self_attn.o_proj"
        name, module = _suffix_match(manager.modules, target)
        view = module.lora_b_stacked[0][0, 0]
        index = (0, 0)
        before = view[index].detach().clone()
        after = (before.float() + float(delta)).to(dtype=view.dtype)
        if torch.equal(before, after):
            raise RuntimeError("v40a mutation rounded to zero")
        self._v40a_mutation = {
            "runtime_module": name, "side": "B", "slice_index": 0,
            "slot": 0, "index": list(index), "before": before,
        }
        view[index] = after
        torch.cuda.synchronize()
        base = _base_identity(manager.modules, self._v40a_runtime_names)
        if base["inventory_sha256"] != self._v40a_base_identity["inventory_sha256"]:
            raise RuntimeError("v40a base weight changed during LoRA mutation")
        return {
            "schema": "vllm-lora-one-element-mutation-v40a",
            "pid": os.getpid(), "runtime_module": name, "side": "B",
            "slice_index": 0, "slot": 0, "index": list(index),
            "before": float(before.float().item()),
            "after": float(after.float().item()),
            "delta_requested": float(delta),
            "base_identity_unchanged": True,
        }

    def restore_one_lora_element_v40a(self):
        state = getattr(self, "_v40a_mutation", None)
        if state is None:
            raise RuntimeError("v40a mutation is not active")
        manager = _manager(self)
        module = manager.modules[state["runtime_module"]]
        view = module.lora_b_stacked[state["slice_index"]][state["slot"], 0]
        index = tuple(state["index"])
        view[index] = state["before"]
        torch.cuda.synchronize()
        base = _base_identity(manager.modules, self._v40a_runtime_names)
        if base["inventory_sha256"] != self._v40a_base_identity["inventory_sha256"]:
            raise RuntimeError("v40a base weight changed during LoRA restoration")
        self._v40a_mutation = None
        return {
            "schema": "vllm-lora-one-element-restoration-v40a",
            "pid": os.getpid(), "runtime_module": state["runtime_module"],
            "restored_exactly": bool(torch.equal(view[index], state["before"])),
            "base_identity_unchanged": True,
        }
