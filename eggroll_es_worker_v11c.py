#!/usr/bin/env python3
"""V11c API-surface retry worker with unchanged V11b RPC behavior."""

from __future__ import annotations

import eggroll_es_worker_v11b as worker_v11b


FROZEN_LAYER_PLANS_V11C = worker_v11b.FROZEN_LAYER_PLANS_V11B
validate_frozen_layer_plan_v11c = worker_v11b.validate_frozen_layer_plan_v11b


class ResidentSignAuditWorkerExtensionV11C(
    worker_v11b.ResidentSignAuditWorkerExtensionV11B,
):
    """Use unchanged V11b selected-layer perturb/restore/update RPCs."""

