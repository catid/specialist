#!/usr/bin/env python3
"""V73C systems-only receipt endpoint over the unchanged V72 worker."""

from __future__ import annotations

import v73c_path_open_guard

from eggroll_es_worker_lora_v72 import LoRAAdapterStateWorkerExtensionV72


if not v73c_path_open_guard.installed():
    raise RuntimeError("V73C worker imported without the pre-import path guard")


class LoRAAdapterStateWorkerExtensionV73C(LoRAAdapterStateWorkerExtensionV72):
    def systems_only_path_guard_receipt_v73c(self):
        return v73c_path_open_guard.receipt()
