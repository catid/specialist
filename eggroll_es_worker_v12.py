#!/usr/bin/env python3
"""V12 consensus worker using the repaired frozen V11c update contract."""

from __future__ import annotations

import eggroll_es_worker_v11c as worker_v11c


FROZEN_LAYER_PLANS_V12 = worker_v11c.FROZEN_LAYER_PLANS_V11C
validate_frozen_layer_plan_v12 = worker_v11c.validate_frozen_layer_plan_v11c


class ConsensusCandidateWorkerExtensionV12(
    worker_v11c.ResidentSignAuditWorkerExtensionV11C,
):
    """Retain V11c's repaired API surface and unchanged update RPCs."""
