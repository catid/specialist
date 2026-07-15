#!/usr/bin/env python3
"""Exact vLLM model-load-memory RPC for the exclusive V24A retry R1."""

from __future__ import annotations

import os

import eggroll_es_worker_v23a_retry_r1 as worker_r1


class HybridBackendMemoryWorkerExtensionV24ARetryR1(
    worker_r1.InsertionLocationAuditWorkerExtensionV23ARetryR1,
):
    """Expose only vLLM's integer model-load-consumed-memory counter."""

    def model_load_memory_v24a_r1(self, expected_arm):
        communicator = self._communicator_state_v3(4)
        runner = getattr(self, "model_runner", None)
        value = getattr(runner, "model_memory_usage", None)
        if type(value) is not int or value <= 0:
            raise RuntimeError("v24a-r1 model_memory_usage is not a positive exact int")
        return {
            "schema": "eggroll-es-v24a-model-load-memory-worker-r1",
            "rank": communicator["rank"],
            "world_size": communicator["world_size"],
            "arm": str(expected_arm),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "model_load_consumed_bytes": value,
            "source_object": "self.model_runner.model_memory_usage",
            "source_assignment": "self.model_memory_usage = m.consumed_memory",
            "measured_after_model_load_before_scoring": True,
        }
