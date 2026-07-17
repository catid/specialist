#!/usr/bin/env python3
"""V73D actor bootstrap that installs the guard before any LoRA parent."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path


ACTOR_BOOTSTRAP_ENV = "SPECIALIST_V73D_ACTOR_BOOTSTRAP"
ACTOR_GUARD_SHA_ENV = "SPECIALIST_V73D_ACTOR_GUARD_SHA256"
EXPECTED_GUARD_SHA256 = (
    "750a445693425cb859bd4a632f14cad98799df7f969d833458282efa9e9c481c"
)
GUARD_PATH = (
    Path(__file__).resolve().parent
    / "v73d_sitecustomize/v73d_path_open_guard.py"
).resolve()
PARENT_MODULES = (
    "eggroll_es_worker_lora_v72",
    "eggroll_es_worker_lora_v71",
    "eggroll_es_worker_lora_v41a",
)
GUARDED_HISTORICAL_REFERENCE_MODULES = (
    "build_lora_es_mirrored_calibration_preregistration_v66",
    "build_eval_v3",
    "build_general_prose_anchor",
)
GUARDED_HISTORICAL_REFERENCE_MODULE_PATH_IDENTITY_SHA256 = frozenset({
    "b1df103b03468a730decd14a258faf872c259e7c119dc56633254e1123e9ed59",
    "4f6101ddbffc3ca10a9e0f83a46a05bce40a6cc8d9bbf132ebfa823f4347ea3c",
    "6860215d19c0b69020d7692e8962e2a98a7e84d4e05bbe4e9b1c6dcc08427879",
})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_exact_guard():
    _require(
        GUARD_PATH.is_file()
        and not GUARD_PATH.is_symlink()
        and _file_sha256(GUARD_PATH) == EXPECTED_GUARD_SHA256,
        "V73D actor guard source identity changed",
    )
    existing = sys.modules.get("v73d_path_open_guard")
    if existing is not None:
        _require(
            Path(existing.__file__).resolve() == GUARD_PATH,
            "V73D imported guard path changed",
        )
        return existing
    specification = importlib.util.spec_from_file_location(
        "v73d_path_open_guard", GUARD_PATH
    )
    _require(
        specification is not None and specification.loader is not None,
        "V73D actor guard loader is unavailable",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules["v73d_path_open_guard"] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("v73d_path_open_guard", None)
        raise
    return module


_PREIMPORTED_PARENTS = sorted(set(PARENT_MODULES).intersection(sys.modules))
_PREIMPORTED_HISTORICAL_REFERENCE_MODULES = sorted(
    set(GUARDED_HISTORICAL_REFERENCE_MODULES).intersection(sys.modules)
)
_guard = _load_exact_guard()
_guard_was_preinstalled = _guard.installed()

if _guard_was_preinstalled:
    _require(
        _guard.installation_mechanism() == "controller_sitecustomize"
        and os.environ.get("SPECIALIST_V73D_CONTROLLER_GUARD_PID")
        == str(os.getpid())
        and os.environ.get(ACTOR_BOOTSTRAP_ENV) != "1"
        and not _PREIMPORTED_PARENTS,
        "V73D controller worker validation guard state changed",
    )
    _PROCESS_ROLE = "controller_worker_contract_validation"
    _BOOTSTRAP_MECHANISM = "controller_sitecustomize"
else:
    _require(
        os.environ.get(ACTOR_BOOTSTRAP_ENV) == "1"
        and os.environ.get(ACTOR_GUARD_SHA_ENV) == EXPECTED_GUARD_SHA256
        and os.environ.get("SPECIALIST_V73D_CONTROLLER_GUARD_PID")
        != str(os.getpid())
        and not _PREIMPORTED_PARENTS
        and not _PREIMPORTED_HISTORICAL_REFERENCE_MODULES,
        "V73D actor bootstrap environment or import order changed",
    )
    _guard.install("ray_actor_worker_extension_pre_parent_import")
    _PROCESS_ROLE = "ray_actor_worker_extension"
    _BOOTSTRAP_MECHANISM = "ray_actor_worker_extension_pre_parent_import"

_PRE_PARENT_GUARD_RECEIPT = _guard.receipt()
_require(
    _PRE_PARENT_GUARD_RECEIPT["installed_before_runtime_imports"] is True
    and _PRE_PARENT_GUARD_RECEIPT["installation_mechanism"]
    == _BOOTSTRAP_MECHANISM
    and _PRE_PARENT_GUARD_RECEIPT["successful_protected_opens"] == 0
    and _PRE_PARENT_GUARD_RECEIPT["successful_protected_resolves"] == 0,
    "V73D pre-parent guard receipt changed",
)

# This is deliberately the first local runtime-parent import in this module.
from eggroll_es_worker_lora_v72 import LoRAAdapterStateWorkerExtensionV72


_require(
    all(name in sys.modules for name in PARENT_MODULES),
    "V73D LoRA parent closure import changed",
)


class LoRAAdapterStateWorkerExtensionV73D(LoRAAdapterStateWorkerExtensionV72):
    def systems_only_path_guard_receipt_v73d(self):
        guard = _guard.receipt()
        result = {
            "schema": "qwen36-v73d-worker-bootstrap-receipt-v1",
            "pid": os.getpid(),
            "process_role": _PROCESS_ROLE,
            "bootstrap_mechanism": _BOOTSTRAP_MECHANISM,
            "guard_was_preinstalled": _guard_was_preinstalled,
            "parent_modules_absent_before_guard_install": (
                not _PREIMPORTED_PARENTS
            ),
            "historical_reference_modules_absent_before_guard_install": (
                not _PREIMPORTED_HISTORICAL_REFERENCE_MODULES
            ),
            "historical_reference_module_identity_count": len(
                GUARDED_HISTORICAL_REFERENCE_MODULE_PATH_IDENTITY_SHA256
            ),
            "historical_reference_module_identity_set_sha256": (
                _guard.canonical_sha256(sorted(
                    GUARDED_HISTORICAL_REFERENCE_MODULE_PATH_IDENTITY_SHA256
                ))
            ),
            "parent_module_count_after_guard_install": sum(
                name in sys.modules for name in PARENT_MODULES
            ),
            "guard_source_sha256": EXPECTED_GUARD_SHA256,
            "actor_bootstrap_env_exact": (
                _PROCESS_ROLE != "ray_actor_worker_extension"
                or (
                    os.environ.get(ACTOR_BOOTSTRAP_ENV) == "1"
                    and os.environ.get(ACTOR_GUARD_SHA_ENV)
                    == EXPECTED_GUARD_SHA256
                )
            ),
            "pre_parent_guard_receipt_sha256": (
                _PRE_PARENT_GUARD_RECEIPT["receipt_sha256"]
            ),
            "pre_parent_guard_receipt": _PRE_PARENT_GUARD_RECEIPT,
            "guard_process_receipt": guard,
            "quality_hpo_or_promotion_authorized": False,
        }
        result["receipt_sha256"] = _guard.canonical_sha256(result)
        return result
