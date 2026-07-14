#!/usr/bin/env python3
"""V11 resident-sign worker with the exact frozen V10 allowlist."""

from __future__ import annotations

import eggroll_es_worker_v10 as worker_v10


FROZEN_LAYER_PLANS_V11 = worker_v10.FROZEN_LAYER_PLANS_V10
validate_frozen_layer_plan_v11 = worker_v10.validate_frozen_layer_plan_v10


class ResidentSignAuditWorkerExtensionV11(
    worker_v10.AntitheticCrossedAuditWorkerExtensionV10,
):
    """Use the unchanged selected-parameter perturb/restore RPC contract."""

