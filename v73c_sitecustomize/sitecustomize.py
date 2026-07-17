"""Install V73C's path guard before target and Ray worker imports."""

from __future__ import annotations

import os


if os.environ.get("SPECIALIST_V73C_SYSTEMS_ONLY_GUARD") == "1":
    import v73c_path_open_guard

    v73c_path_open_guard.install()
