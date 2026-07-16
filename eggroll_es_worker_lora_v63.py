#!/usr/bin/env python3
"""V63 worker-side receipt for the effective vLLM LoRA capacity."""

from __future__ import annotations

import eggroll_es_worker_lora_v52 as state_v52


class LoRAAdapterStateWorkerExtensionV63(
    state_v52.LoRAAdapterStateWorkerExtensionV52,
):
    """Retain V52 state methods and attest the live worker LoRA config."""

    def runtime_lora_capacity_v63(self):
        config = getattr(self, "lora_config", None)
        if (
            config is None
            or config.max_loras != 1
            or config.max_cpu_loras != 2
            or config.max_lora_rank != 32
        ):
            raise RuntimeError("v63 effective worker LoRA capacity changed")
        return {
            "schema": "v63-effective-worker-lora-capacity",
            "lora_enabled": True,
            "max_loras": int(config.max_loras),
            "max_cpu_loras": int(config.max_cpu_loras),
            "max_lora_rank": int(config.max_lora_rank),
        }

    def runtime_active_lora_v63(self, expected_lora_int_id):
        expected_lora_int_id = int(expected_lora_int_id)
        config = getattr(self, "lora_config", None)
        runner = getattr(self, "model_runner", None)
        worker_manager = getattr(runner, "lora_manager", None)
        adapter_manager = getattr(worker_manager, "_adapter_manager", None)
        active = [
            int(value)
            for value in getattr(adapter_manager, "lora_index_to_id", [])
            if value is not None
        ]
        loaded = sorted(int(value) for value in worker_manager.list_adapters())
        if (
            config is None
            or config.max_loras != 1
            or config.max_cpu_loras != 2
            or expected_lora_int_id not in (1, 2)
            or active != [expected_lora_int_id]
            or expected_lora_int_id not in loaded
            or not set(loaded).issubset({1, 2})
            or not 1 <= len(loaded) <= 2
        ):
            raise RuntimeError("v63 effective active LoRA identity changed")
        return {
            "schema": "v63-effective-active-lora-receipt",
            "expected_lora_int_id": expected_lora_int_id,
            "active_lora_ids": active,
            "loaded_cpu_cache_lora_ids": loaded,
            "active_matches_expected": True,
            "max_loras": int(config.max_loras),
            "max_cpu_loras": int(config.max_cpu_loras),
        }
