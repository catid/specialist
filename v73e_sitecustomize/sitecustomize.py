"""Install V73E's path guard before target and Ray worker imports."""

from __future__ import annotations

import os


if os.environ.get("SPECIALIST_V73E_SYSTEMS_ONLY_GUARD") == "1" and (
    os.environ.get("SPECIALIST_V73E_CONTROLLER_GUARD_PID") in {None, str(os.getpid())}
):
    import v73e_path_open_guard

    os.environ["SPECIALIST_V73E_CONTROLLER_GUARD_PID"] = str(os.getpid())
    v73e_path_open_guard.install("controller_sitecustomize")
