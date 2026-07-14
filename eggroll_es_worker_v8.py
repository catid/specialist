#!/usr/bin/env python3
"""Middle-late-only split-seed population-32 worker allowlist for v8."""

from __future__ import annotations

from types import FunctionType

import eggroll_es_worker_v7 as worker_v7


FROZEN_LAYER_PLANS_V8 = {
    key: value
    for key, value in worker_v7.FROZEN_LAYER_PLANS_V7.items()
    if value["plan"] == "middle_late"
}


def _clone_with_globals(function, replacements, name):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    clone = FunctionType(
        function.__code__, namespace, name, function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__doc__ = function.__doc__
    clone.__module__ = __name__
    clone.__qualname__ = name
    return clone


validate_frozen_layer_plan_v8 = _clone_with_globals(
    worker_v7.validate_frozen_layer_plan_v7,
    {"FROZEN_LAYER_PLANS_V4": FROZEN_LAYER_PLANS_V8},
    "validate_frozen_layer_plan_v8",
)

_install_layer_plan_v8 = _clone_with_globals(
    worker_v7.DirectionStabilityAuditWorkerExtensionV7.install_layer_plan_v4,
    {"validate_frozen_layer_plan_v4": validate_frozen_layer_plan_v8},
    "install_layer_plan_v4",
)


class SplitSeedPopulation32AuditWorkerExtensionV8(
    worker_v7.DirectionStabilityAuditWorkerExtensionV7,
):
    """Exact v7 worker restricted to the v8 middle-late diagnostic."""

    install_layer_plan_v4 = _install_layer_plan_v8
