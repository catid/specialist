#!/usr/bin/env python3
"""High-rep exact-master worker endpoints for V65B calibration."""

from __future__ import annotations

import os

from eggroll_es_worker_lora_v65a import LoRAAdapterStateWorkerExtensionV65A


class LoRAAdapterStateWorkerExtensionV65B(
    LoRAAdapterStateWorkerExtensionV65A,
):
    """Extend exact V65A slot receipts to the frozen V65B schedule."""

    @staticmethod
    def _period_v65b(period_kind, period_index) -> tuple[str, int]:
        if type(period_kind) is not str or type(period_index) is not int:
            raise ValueError("v65b period coordinate must use exact types")
        limit = 8 if period_kind == "unscored_warmup" else (
            72 if period_kind == "scored" else -1
        )
        if period_index not in range(limit):
            raise ValueError("v65b period coordinate changed")
        return period_kind, period_index

    @staticmethod
    def _intrinsic_worker_identity_v65b() -> dict:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        try:
            physical_gpu = int(visible)
        except (TypeError, ValueError) as error:
            raise RuntimeError("v65b worker GPU identity is not singular") from error
        pid = os.getpid()
        if (
            type(pid) is not int or pid <= 0
            or type(physical_gpu) is not int or physical_gpu not in range(4)
            or visible != str(physical_gpu)
        ):
            raise RuntimeError("v65b intrinsic worker identity changed")
        return {
            "worker_pid": pid,
            "worker_physical_gpu_id": physical_gpu,
            "worker_cuda_visible_devices": visible,
        }

    def rematerialize_exact_master_v65b(
        self, period_kind, period_index,
        expected_master_sha256, expected_runtime_values_sha256,
    ):
        if (
            type(expected_master_sha256) is not str
            or type(expected_runtime_values_sha256) is not str
        ):
            raise ValueError("v65b master hashes must be exact strings")
        period_kind, period_index = self._period_v65b(
            period_kind, period_index,
        )
        receipt = super().rematerialize_exact_master_v65a(
            period_kind,
            period_index % 4,
            expected_master_sha256,
            expected_runtime_values_sha256,
        )
        if receipt.get("schema") != "exact-master-slot-write-v65a":
            raise RuntimeError("v65b inherited slot write changed")
        return {
            **receipt,
            **self._intrinsic_worker_identity_v65b(),
            "schema": "exact-master-slot-write-v65b",
            "period_index": period_index,
        }

    def read_only_exact_master_slot_v65b(
        self, period_kind, period_index, edge,
    ):
        if type(edge) is not str or edge not in {
            "before_generation", "after_generation",
        }:
            raise ValueError("v65b read edge must be an exact sealed string")
        period_kind, period_index = self._period_v65b(
            period_kind, period_index,
        )
        receipt = super().read_only_exact_master_slot_v65a(
            period_kind, period_index % 4, edge,
        )
        if receipt.get("schema") != "read-only-exact-master-slot-v65a":
            raise RuntimeError("v65b inherited read-only slot receipt changed")
        return {
            **receipt,
            **self._intrinsic_worker_identity_v65b(),
            "schema": "read-only-exact-master-slot-v65b",
            "period_index": period_index,
        }
