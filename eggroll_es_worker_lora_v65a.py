#!/usr/bin/env python3
"""Exact-master slot-write extension for V65A and later V65 evaluation."""

from __future__ import annotations

import os
import time

import eggroll_es_worker_lora_v41a as state_v41a
from eggroll_es_worker_lora_v64 import LoRAAdapterStateWorkerExtensionV64
from eggroll_es_worker_lora_v41a import adapter_identity_v41a


class LoRAAdapterStateWorkerExtensionV65A(LoRAAdapterStateWorkerExtensionV64):
    """Re-materialize the pinned master without changing canonical state."""

    def runtime_identity_v40a(self):
        """Retain the exact V40A worker-identity compatibility endpoint."""
        return {
            "schema": "lora-topology-worker-identity-v40a",
            "pid": os.getpid(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_current_device": (
                int(state_v41a.torch.cuda.current_device())
                if state_v41a.torch.cuda.is_available() else None
            ),
        }

    def runtime_active_lora_v65a(
        self,
        expected_lora_int_id,
        staged_adapter_weights,
        expected_weights_file_sha256,
        expected_canonical_fp32_state_sha256,
        expected_runtime_bf16_values_sha256,
    ):
        expected_lora_int_id = int(expected_lora_int_id)
        config = getattr(self, "lora_config", None)
        runner = getattr(self, "model_runner", None)
        facade = getattr(runner, "lora_manager", None)
        manager = getattr(facade, "_adapter_manager", None)
        active = [
            int(value)
            for value in getattr(manager, "lora_index_to_id", [])
            if value is not None
        ]
        active_cache = [
            int(value)
            for value in getattr(
                getattr(manager, "_active_adapters", None), "cache", {}
            )
        ]
        loaded = sorted(int(value) for value in facade.list_adapters()) \
            if facade is not None else []
        if (
            config is None or manager is None
            or config.max_loras != 1 or config.max_cpu_loras != 2
            or config.max_lora_rank != 32
            or expected_lora_int_id != 1
            or active != [1] or active_cache != [1] or loaded != [1]
            or type(facade).__name__ != "LRUCacheWorkerLoRAManager"
            or type(manager).__name__ != "LRUCacheLoRAModelManager"
        ):
            raise RuntimeError("v65a effective active LoRA identity changed")
        applied = super().runtime_applied_lora_v64(
            expected_lora_int_id,
            staged_adapter_weights,
            expected_weights_file_sha256,
            expected_canonical_fp32_state_sha256,
            expected_runtime_bf16_values_sha256,
        )
        if (
            applied.get("active_lora_ids") != [1]
            or applied.get("active_manager_cache_lora_ids") != [1]
            or applied.get("loaded_cpu_cache_lora_ids") != [1]
            or applied.get("exact_staged_fp32_to_gpu_slot_equality") is not True
            or applied.get("exact_registered_postpack_to_gpu_slot_equality")
            is not True
        ):
            raise RuntimeError("v65a staged V434 active receipt changed")
        return {
            "schema": "v65a-effective-active-lora-receipt",
            "expected_lora_int_id": 1,
            "active_lora_ids": active,
            "active_manager_cache_lora_ids": active_cache,
            "loaded_cpu_cache_lora_ids": loaded,
            "facade_type": type(facade).__name__,
            "manager_type": type(manager).__name__,
            "active_slot_index": manager.lora_index_to_id.index(1),
            "max_loras": int(config.max_loras),
            "max_cpu_loras": int(config.max_cpu_loras),
            "max_lora_rank": int(config.max_lora_rank),
            "staged_v434_applied_receipt": applied,
            "extra_or_candidate_adapter_loaded": False,
        }

    def rematerialize_exact_master_v65a(
        self, period_kind, period_index,
        expected_master_sha256, expected_runtime_values_sha256,
    ):
        started_ns = time.monotonic_ns()
        period_kind = str(period_kind)
        period_index = int(period_index)
        if (
            period_kind not in ("unscored_warmup", "scored")
            or period_index not in range(4)
        ):
            raise ValueError("v65a invalid exact-master materialization period")
        self._require_population_transition_quiescent_v51()
        if getattr(self, "_v41_active_perturbation", None) is not None:
            raise RuntimeError("v65a exact-master write found an active perturbation")
        if getattr(self, "_v41_active_plan_id", None) is not None:
            raise RuntimeError("v65a exact-master write found an active plan")
        master_identity = adapter_identity_v41a(self._v41_master)
        if (
            master_identity != self._v41_current_identity
            or master_identity["sha256"] != str(expected_master_sha256)
        ):
            raise RuntimeError("v65a pinned FP32 master identity changed")
        materialization = self._materialize_v41a(
            self._v41_master, "v65a_exact_master_slot_write",
        )
        if materialization["runtime_values_sha256"] != str(
            expected_runtime_values_sha256
        ):
            raise RuntimeError("v65a exact master runtime values changed")
        base_identity = self._base_check_v41a("v65a_exact_master_slot_write")
        ended_ns = time.monotonic_ns()
        return {
            "schema": "exact-master-slot-write-v65a",
            "period_kind": period_kind,
            "period_index": period_index,
            "master_identity": master_identity,
            "materialization": materialization,
            "base_identity": base_identity,
            "transaction_state_quiescent": True,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }

    def read_only_exact_master_slot_v65a(self, period_kind, period_index, edge):
        """Hash the live LoRA slot without resetting or writing any weight."""
        started_ns = time.monotonic_ns()
        period_kind = str(period_kind)
        period_index = int(period_index)
        edge = str(edge)
        if (
            period_kind not in ("unscored_warmup", "scored")
            or period_index not in range(4)
            or edge not in ("before_generation", "after_generation")
        ):
            raise ValueError("v65a invalid read-only slot receipt request")
        self._require_population_transition_quiescent_v51()
        if (
            getattr(self, "_v41_active_perturbation", None) is not None
            or getattr(self, "_v41_active_plan_id", None) is not None
        ):
            raise RuntimeError("v65a read-only slot receipt found active state")
        master_identity = adapter_identity_v41a(self._v41_master)
        if master_identity != self._v41_current_identity:
            raise RuntimeError("v65a read-only canonical master changed")
        manager = self._manager_v41a()
        active = [
            int(value)
            for value in getattr(manager, "lora_index_to_id", [])
            if value is not None
        ]
        active_cache = [
            int(value)
            for value in getattr(
                getattr(manager, "_active_adapters", None), "cache", {}
            )
        ]
        if active != [1] or active_cache != [1]:
            raise RuntimeError("v65a read-only slot found another active LoRA")
        records = []
        with state_v41a.torch.no_grad():
            for item in self._v41_assignments:
                source = self._v41_master[item["peft_key"]]
                logical, side = state_v41a.topology._source_parts(
                    item["peft_key"]
                )
                _target, slices = state_v41a.topology._runtime_target(logical)
                module = manager.modules[item["runtime_module"]]
                expected = state_v41a._expected_runtime_value_v41a(
                    source, side, slices, item["slice_index"],
                    module.output_slices, self._v41_scale,
                )
                stacked = (
                    module.lora_a_stacked if side == "A"
                    else module.lora_b_stacked
                )[item["slice_index"]]
                view = stacked[state_v41a.ADAPTER_SLOT_V41A, 0]
                if (
                    view.shape != expected.shape
                    or view.dtype != state_v41a.torch.bfloat16
                    or not state_v41a.torch.equal(view.cpu(), expected.cpu())
                ):
                    raise RuntimeError(
                        "v65a read-only live LoRA slot differed from master"
                    )
                records.append({
                    **item,
                    "dtype": str(view.dtype),
                    "elements": int(view.numel()),
                    "sha256": state_v41a._tensor_sha256_v41a(view),
                })
        if state_v41a.torch.cuda.is_available():
            state_v41a.torch.cuda.synchronize()
        runtime_values_sha256 = state_v41a.canonical_sha256_v3(records)
        runtime_elements = sum(item["elements"] for item in records)
        if (
            len(records) != state_v41a.EXPECTED_RUNTIME_VIEWS_V41A
            or runtime_elements != state_v41a.EXPECTED_RUNTIME_ELEMENTS_V41A
            or runtime_values_sha256
            != self._v41_active_materialization["runtime_values_sha256"]
        ):
            raise RuntimeError("v65a read-only runtime inventory changed")
        base_identity = self._base_check_v41a("v65a_read_only_slot_receipt")
        ended_ns = time.monotonic_ns()
        return {
            "schema": "read-only-exact-master-slot-v65a",
            "period_kind": period_kind,
            "period_index": period_index,
            "edge": edge,
            "master_identity": master_identity,
            "runtime_view_count": len(records),
            "runtime_elements": runtime_elements,
            "runtime_dtype": "torch.bfloat16",
            "runtime_values_sha256": runtime_values_sha256,
            "active_lora_ids": active,
            "active_manager_cache_lora_ids": active_cache,
            "base_identity": base_identity,
            "transaction_state_quiescent": True,
            "slot_read_only_no_weight_write_or_reset": True,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }
