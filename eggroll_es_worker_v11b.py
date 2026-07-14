#!/usr/bin/env python3
"""V11b retry worker; execution RPCs are byte-for-byte inherited from V11."""

from __future__ import annotations

import eggroll_es_worker_v11 as worker_v11


FROZEN_LAYER_PLANS_V11B = worker_v11.FROZEN_LAYER_PLANS_V11
validate_frozen_layer_plan_v11b = worker_v11.validate_frozen_layer_plan_v11


class ResidentSignAuditWorkerExtensionV11B(
    worker_v11.ResidentSignAuditWorkerExtensionV11,
):
    """Use unchanged V11 selected-layer perturb/restore/update RPCs."""

