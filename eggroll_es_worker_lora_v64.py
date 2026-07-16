#!/usr/bin/env python3
"""V64 worker receipt proving exact LoRA bytes reached the live GPU slot."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

import eggroll_es_worker_lora_topology_v40a as topology
import eggroll_es_worker_lora_v41a as state_v41a
import eggroll_es_worker_lora_v52 as state_v52
from eggroll_es_worker_v3 import canonical_sha256_v3


EXPECTED_CANONICAL_KEY_SHA256_V64 = (
    "ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280"
)
EXPECTED_RUNTIME_MODULE_SHA256_V64 = (
    "f09f656d7890c8776170bcc65e9273fbafefad2651a9ab6bc2ef805dfae6eeca"
)
STAGED_KEY_PREFIX_V64 = "base_model.model.model.language_model.layers."
CANONICAL_KEY_PREFIX_V64 = "base_model.model.model.layers."


def _full_slot_expected_v64(
    actual: torch.Tensor,
    source: torch.Tensor | None,
) -> torch.Tensor:
    full = torch.zeros_like(actual)
    if source is None:
        return full
    if (
        not isinstance(source, torch.Tensor)
        or source.ndim != 2
        or source.shape[0] > full.shape[0]
        or source.shape[1] > full.shape[1]
    ):
        raise RuntimeError("v64 LoRA source/slot shape changed")
    full[: source.shape[0], : source.shape[1]].copy_(
        source.to(device=full.device, dtype=full.dtype),
        non_blocking=False,
    )
    return full


def _registered_slot_records_v64(manager, lora_model, slot: int):
    """Compare every post-pack registered weight with the entire GPU slot."""
    records = []
    matched = []
    for name in sorted(manager.modules):
        layer_weights = manager._get_lora_layer_weights(lora_model, name)
        if layer_weights is None:
            continue
        matched.append(name)
        module = manager.modules[name]
        if int(getattr(module, "tp_size", 1)) != 1:
            raise RuntimeError("v64 requires TP1 exact LoRA slot audit")
        a_sources = (
            [layer_weights.lora_a]
            if isinstance(layer_weights.lora_a, torch.Tensor)
            else list(layer_weights.lora_a)
        )
        b_sources = (
            [layer_weights.lora_b]
            if isinstance(layer_weights.lora_b, torch.Tensor)
            else list(layer_weights.lora_b)
        )
        slice_count = int(module.n_slices)
        if len(b_sources) != slice_count:
            expand = getattr(module, "expand_packed_lora", None)
            if not callable(expand):
                raise RuntimeError("v64 packed LoRA expansion unavailable")
            a_sources, b_sources = expand(a_sources, b_sources)
        if len(a_sources) != slice_count or len(b_sources) != slice_count:
            raise RuntimeError("v64 packed registered slice coverage changed")
        for side, sources, parents in (
            ("A", a_sources, module.lora_a_stacked),
            ("B", b_sources, module.lora_b_stacked),
        ):
            if len(parents) != slice_count:
                raise RuntimeError("v64 resident LoRA slice coverage changed")
            for slice_index, (source, parent) in enumerate(
                zip(sources, parents, strict=True)
            ):
                actual = parent[slot, 0]
                expected = _full_slot_expected_v64(actual, source)
                if not torch.equal(actual, expected):
                    raise RuntimeError(
                        f"v64 applied slot differs from registered LoRA: "
                        f"{name} {side}{slice_index}"
                    )
                records.append({
                    "runtime_module": name,
                    "side": side,
                    "slice_index": slice_index,
                    "shape": list(actual.shape),
                    "dtype": str(actual.dtype),
                    "elements": int(actual.numel()),
                    "sha256": state_v41a._tensor_sha256_v41a(actual),
                })
    return matched, records


class LoRAAdapterStateWorkerExtensionV64(
    state_v52.LoRAAdapterStateWorkerExtensionV52,
):
    """Retain V52 state methods and attest actual applied inference bytes."""

    def runtime_lora_capacity_v64(self):
        config = getattr(self, "lora_config", None)
        if (
            config is None
            or config.max_loras != 1
            or config.max_cpu_loras != 2
            or config.max_lora_rank != 32
        ):
            raise RuntimeError("v64 effective worker LoRA capacity changed")
        return {
            "schema": "v64-effective-worker-lora-capacity",
            "lora_enabled": True,
            "max_loras": int(config.max_loras),
            "max_cpu_loras": int(config.max_cpu_loras),
            "max_lora_rank": int(config.max_lora_rank),
        }

    def runtime_applied_lora_v64(
        self,
        expected_lora_int_id,
        staged_adapter_weights,
        expected_weights_file_sha256,
        expected_canonical_fp32_state_sha256,
        expected_runtime_bf16_values_sha256,
    ):
        """Fail unless exact staged FP32 bytes occupy every expected GPU view."""
        expected_lora_int_id = int(expected_lora_int_id)
        staged_adapter_weights = Path(staged_adapter_weights).resolve()
        expected_weights_file_sha256 = str(expected_weights_file_sha256)
        expected_canonical_fp32_state_sha256 = str(
            expected_canonical_fp32_state_sha256
        )
        expected_runtime_bf16_values_sha256 = str(
            expected_runtime_bf16_values_sha256
        )
        if expected_lora_int_id not in (1, 2):
            raise RuntimeError("v64 LoRA integer identity changed")

        config = getattr(self, "lora_config", None)
        runner = getattr(self, "model_runner", None)
        facade = getattr(runner, "lora_manager", None)
        manager = getattr(facade, "_adapter_manager", None)
        if (
            config is None
            or config.max_loras != 1
            or config.max_cpu_loras != 2
            or manager is None
            or type(facade).__name__ != "LRUCacheWorkerLoRAManager"
            or type(manager).__name__ != "LRUCacheLoRAModelManager"
        ):
            raise RuntimeError("v64 effective LoRA manager topology changed")

        active = [
            int(value)
            for value in getattr(manager, "lora_index_to_id", [])
            if value is not None
        ]
        active_cache = [
            int(value) for value in getattr(manager._active_adapters, "cache", {})
        ]
        loaded = sorted(int(value) for value in facade.list_adapters())
        registered_models = manager.list_adapters()
        if (
            active != [expected_lora_int_id]
            or active_cache != [expected_lora_int_id]
            or expected_lora_int_id not in loaded
            or not set(loaded).issubset({1, 2})
            or not 1 <= len(loaded) <= 2
            or expected_lora_int_id not in registered_models
        ):
            raise RuntimeError("v64 effective active LoRA identity changed")
        slot = manager.lora_index_to_id.index(expected_lora_int_id)
        if slot != 0:
            raise RuntimeError("v64 sole active LoRA slot changed")

        if (
            state_v41a.file_sha256_v41a(staged_adapter_weights)
            != expected_weights_file_sha256
        ):
            raise RuntimeError("v64 staged adapter weights changed")
        canonical = {}
        staged_keys = []
        with safe_open(
            staged_adapter_weights, framework="pt", device="cpu"
        ) as handle:
            for key in handle.keys():
                if not key.startswith(STAGED_KEY_PREFIX_V64):
                    raise RuntimeError("v64 adapter is not in staged vLLM namespace")
                canonical_key = CANONICAL_KEY_PREFIX_V64 + key[
                    len(STAGED_KEY_PREFIX_V64):
                ]
                tensor = handle.get_tensor(key)
                if tensor.dtype != torch.float32:
                    raise RuntimeError("v64 staged adapter master is not FP32")
                staged_keys.append(key)
                canonical[canonical_key] = tensor.contiguous()
        identity = state_v41a.adapter_identity_v41a(canonical)
        if (
            len(staged_keys) != 70
            or identity["sha256"] != expected_canonical_fp32_state_sha256
            or identity["ordered_key_sha256"]
            != EXPECTED_CANONICAL_KEY_SHA256_V64
            or identity["tensor_count"] != 70
            or identity["elements"] != 4_528_128
        ):
            raise RuntimeError("v64 staged canonical LoRA identity changed")

        lora_model = registered_models[expected_lora_int_id]
        assignments = state_v41a._runtime_assignments_v41a(manager, canonical)
        runtime_names = sorted({item["runtime_module"] for item in assignments})
        registered_names = sorted(lora_model.loras)
        matched_names = sorted(
            name for name in manager.modules
            if manager._get_lora_layer_weights(lora_model, name) is not None
        )
        if (
            len(assignments) != 82
            or sum(item["elements"] for item in identity["tensors"]) != 4_528_128
            or len(runtime_names) != 23
            or canonical_sha256_v3(runtime_names)
            != EXPECTED_RUNTIME_MODULE_SHA256_V64
            or registered_names != runtime_names
            or matched_names != runtime_names
        ):
            raise RuntimeError("v64 registered/live LoRA module coverage changed")

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        source_linked_records = []
        for item in assignments:
            source = canonical[item["peft_key"]]
            logical, side = topology._source_parts(item["peft_key"])
            _target, slices = topology._runtime_target(logical)
            module = manager.modules[item["runtime_module"]]
            expected = state_v41a._expected_runtime_value_v41a(
                source,
                side,
                slices,
                item["slice_index"],
                module.output_slices,
                2.0,
            )
            stacked = (
                module.lora_a_stacked if side == "A" else module.lora_b_stacked
            )[item["slice_index"]]
            actual = stacked[slot, 0]
            if (
                actual.shape != expected.shape
                or actual.dtype != torch.bfloat16
                or not torch.equal(actual.cpu(), expected.cpu())
            ):
                raise RuntimeError(
                    f"v64 staged FP32 bytes do not match live GPU slot: "
                    f"{item['peft_key']}"
                )
            source_linked_records.append({
                **item,
                "dtype": str(actual.dtype),
                "elements": int(actual.numel()),
                "sha256": state_v41a._tensor_sha256_v41a(actual),
            })
        runtime_values_sha256 = canonical_sha256_v3(source_linked_records)
        runtime_elements = sum(
            item["elements"] for item in source_linked_records
        )
        if (
            len(source_linked_records) != 82
            or runtime_elements != 4_921_344
            or runtime_values_sha256 != expected_runtime_bf16_values_sha256
        ):
            raise RuntimeError("v64 exact source-linked runtime values changed")

        matched_again, registered_records = _registered_slot_records_v64(
            manager, lora_model, slot
        )
        if (
            matched_again != runtime_names
            or len(registered_records) != 82
            or sum(item["elements"] for item in registered_records)
            != 4_921_344
            or {item["dtype"] for item in registered_records}
            != {"torch.bfloat16"}
        ):
            raise RuntimeError("v64 registered-object GPU slot audit changed")

        return {
            "schema": "v64-effective-applied-lora-receipt",
            "expected_lora_int_id": expected_lora_int_id,
            "active_lora_ids": active,
            "active_manager_cache_lora_ids": active_cache,
            "loaded_cpu_cache_lora_ids": loaded,
            "active_slot_index": slot,
            "facade_type": type(facade).__name__,
            "manager_type": type(manager).__name__,
            "staged_weights_file_sha256": expected_weights_file_sha256,
            "canonical_fp32_state_sha256": identity["sha256"],
            "canonical_ordered_key_sha256": identity["ordered_key_sha256"],
            "canonical_tensor_count": identity["tensor_count"],
            "canonical_elements": identity["elements"],
            "registered_lora_module_count": len(registered_names),
            "matched_live_lora_module_count": len(matched_names),
            "unmatched_registered_lora_module_count": 0,
            "runtime_module_manifest_sha256": canonical_sha256_v3(runtime_names),
            "source_linked_runtime_view_count": len(source_linked_records),
            "source_linked_runtime_elements": runtime_elements,
            "source_linked_runtime_dtype": "torch.bfloat16",
            "source_linked_runtime_values_sha256": runtime_values_sha256,
            "registered_slot_view_count": len(registered_records),
            "registered_slot_records_sha256": canonical_sha256_v3(
                registered_records
            ),
            "exact_staged_fp32_to_gpu_slot_equality": True,
            "exact_registered_postpack_to_gpu_slot_equality": True,
            "active_matches_expected": True,
            "max_loras": int(config.max_loras),
            "max_cpu_loras": int(config.max_cpu_loras),
        }
