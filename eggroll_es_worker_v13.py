#!/usr/bin/env python3
"""Train-panel diagnostic worker with every model-update RPC disabled."""

from __future__ import annotations

import eggroll_es_worker_v11c as worker_v11c


FROZEN_LAYER_PLANS_V13 = worker_v11c.FROZEN_LAYER_PLANS_V11C
validate_frozen_layer_plan_v13 = worker_v11c.validate_frozen_layer_plan_v11c


class TrainPanelDiagnosticWorkerExtensionV13(
    worker_v11c.ResidentSignAuditWorkerExtensionV11C,
):
    """Keep V11c perturb/restore RPCs while making updates unreachable."""

    @staticmethod
    def _forbid_update_v13():
        raise RuntimeError("v13 alpha-zero diagnostic forbids model updates")

    def prepare_sharded_seed_update_v4(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_update_v13()

    def execute_prepared_seed_update_v4(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_update_v13()

    def commit_prepared_seed_update_v4(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_update_v13()

    def update_weights_from_seeds(self, *args, **kwargs):
        del args, kwargs
        return self._forbid_update_v13()
