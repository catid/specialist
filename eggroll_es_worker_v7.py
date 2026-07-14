#!/usr/bin/env python3
"""Two-arm direction-stability worker allowlist for EGGROLL-ES v7."""

from __future__ import annotations

from types import FunctionType

import eggroll_es_worker_v6 as worker_v6


FROZEN_LAYER_PLANS_V7 = {
    key: value
    for key, value in worker_v6.FROZEN_LAYER_PLANS_V6.items()
    if value["plan"] in {"front", "middle_late"}
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


validate_frozen_layer_plan_v7 = _clone_with_globals(
    worker_v6.validate_frozen_layer_plan_v6,
    {"FROZEN_LAYER_PLANS_V4": FROZEN_LAYER_PLANS_V7},
    "validate_frozen_layer_plan_v7",
)

_install_layer_plan_v7 = _clone_with_globals(
    worker_v6.FrozenEdgeSplitAuditWorkerExtensionV6.install_layer_plan_v4,
    {"validate_frozen_layer_plan_v4": validate_frozen_layer_plan_v7},
    "install_layer_plan_v4",
)


class DirectionStabilityAuditWorkerExtensionV7(
    worker_v6.FrozenEdgeSplitAuditWorkerExtensionV6,
):
    """Exact v6 update worker restricted to the two preregistered v7 arms."""

    install_layer_plan_v4 = _install_layer_plan_v7
