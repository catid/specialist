"""Install V73H's controller or actor guard before process runtime imports."""

from __future__ import annotations

import os


EXPECTED_GUARD_SHA256 = (
    "bf753cc721821e8f7e955a26d6a634766bb0909e89f93c22bbd71ce8642b55d7"
)
ACTOR_MECHANISM = "ray_actor_sitecustomize_pre_runtime_imports"


if os.environ.get("SPECIALIST_V73E_SYSTEMS_ONLY_GUARD") == "1":
    import v73e_path_open_guard

    actor_bootstrap = os.environ.get("SPECIALIST_V73E_ACTOR_BOOTSTRAP")
    controller_pid = os.environ.get("SPECIALIST_V73E_CONTROLLER_GUARD_PID")
    if actor_bootstrap == "1":
        if (
            os.environ.get("SPECIALIST_V73E_ACTOR_GUARD_SHA256")
            != EXPECTED_GUARD_SHA256
            or controller_pid in {None, str(os.getpid())}
        ):
            raise RuntimeError("V73H actor sitecustomize bootstrap changed")
        v73e_path_open_guard.install(ACTOR_MECHANISM)
    elif controller_pid in {None, str(os.getpid())}:
        os.environ["SPECIALIST_V73E_CONTROLLER_GUARD_PID"] = str(os.getpid())
        v73e_path_open_guard.install("controller_sitecustomize")
