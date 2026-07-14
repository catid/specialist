#!/usr/bin/env python3
"""V10 antithetic/crossed diagnostic worker with the exact v8 allowlist."""

from __future__ import annotations

import eggroll_es_worker_v8 as worker_v8


FROZEN_LAYER_PLANS_V10 = worker_v8.FROZEN_LAYER_PLANS_V8
validate_frozen_layer_plan_v10 = worker_v8.validate_frozen_layer_plan_v8


class AntitheticCrossedAuditWorkerExtensionV10(
    worker_v8.SplitSeedPopulation32AuditWorkerExtensionV8,
):
    """Use inherited selected-parameter noise, including its negate flag."""
