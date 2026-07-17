#!/usr/bin/env python3
"""V73G bootstrap repair for controller validation and guarded Ray actors.

The controller's sitecustomize guard covers its complete target import graph,
so controller-side class validation may legitimately observe parent worker
modules imported after that guard.  A new Ray actor still must install its own
guard before importing any parent or historical reference module.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path


ACTOR_BOOTSTRAP_ENV = "SPECIALIST_V73E_ACTOR_BOOTSTRAP"
ACTOR_GUARD_SHA_ENV = "SPECIALIST_V73E_ACTOR_GUARD_SHA256"
EXPECTED_GUARD_SHA256 = (
    "fdf4d7fa0d96d58dce85edfdad45fd21dc8ed32db33b497d5208b62ac22d2724"
)
GUARD_PATH = (
    Path(__file__).resolve().parent
    / "v73e_sitecustomize/v73e_path_open_guard.py"
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
        "V73E actor guard source identity changed",
    )
    existing = sys.modules.get("v73e_path_open_guard")
    if existing is not None:
        _require(
            Path(existing.__file__).resolve() == GUARD_PATH,
            "V73E imported guard path changed",
        )
        return existing
    specification = importlib.util.spec_from_file_location(
        "v73e_path_open_guard", GUARD_PATH
    )
    _require(
        specification is not None and specification.loader is not None,
        "V73E actor guard loader is unavailable",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules["v73e_path_open_guard"] = module
    try:
        specification.loader.exec_module(module)
    except BaseException:
        sys.modules.pop("v73e_path_open_guard", None)
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
        and os.environ.get("SPECIALIST_V73E_CONTROLLER_GUARD_PID")
        == str(os.getpid())
        and os.environ.get(ACTOR_BOOTSTRAP_ENV) != "1",
        "V73E controller worker validation guard state changed",
    )
    _PROCESS_ROLE = "controller_worker_contract_validation"
    _BOOTSTRAP_MECHANISM = "controller_sitecustomize"
else:
    _require(
        os.environ.get(ACTOR_BOOTSTRAP_ENV) == "1"
        and os.environ.get(ACTOR_GUARD_SHA_ENV) == EXPECTED_GUARD_SHA256
        and os.environ.get("SPECIALIST_V73E_CONTROLLER_GUARD_PID")
        != str(os.getpid())
        and not _PREIMPORTED_PARENTS
        and not _PREIMPORTED_HISTORICAL_REFERENCE_MODULES,
        "V73E actor bootstrap environment or import order changed",
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
    "V73E pre-parent guard receipt changed",
)

# This is deliberately the first local runtime-parent import in this module.
from eggroll_es_worker_lora_v72 import LoRAAdapterStateWorkerExtensionV72

import threading

import eggroll_es_worker_lora_v71 as state_v71
import eggroll_es_worker_lora_v72 as state_v72
import qwen36_v73g_exact_phase_profiler_contract as contract_v73e


_require(
    all(name in sys.modules for name in PARENT_MODULES),
    "V73E LoRA parent closure import changed",
)

_STAGED_LOAD_LOCK = threading.Lock()


def inverse_staged_master_v73e(path):
    """Own a canonical PEFT master by inverting only the sealed key prefix."""
    source_path = Path(path).resolve()
    _require(
        source_path == contract_v73e.STAGED_ADAPTER_WEIGHTS,
        "V73E staged inverse source path changed",
    )
    tensors = {}
    records = []
    target_keys = []
    with state_v72.safe_open(
        source_path, framework="pt", device="cpu"
    ) as handle:
        keys = sorted(handle.keys())
        for target_key in keys:
            _require(
                target_key.startswith(contract_v73e.STAGED_TARGET_PREFIX),
                "V73E staged key is outside the exact target namespace",
            )
            suffix = target_key[len(contract_v73e.STAGED_TARGET_PREFIX):]
            _require(
                bool(suffix)
                and not suffix.startswith("language_model.")
                and contract_v73e.CANONICAL_SOURCE_PREFIX not in suffix,
                "V73E staged key cannot be inverted exactly once",
            )
            source_key = contract_v73e.CANONICAL_SOURCE_PREFIX + suffix
            tensor = handle.get_tensor(target_key)
            _require(
                tensor.dtype == state_v72.torch.float32,
                "V73E staged tensor is not FP32",
            )
            owned = tensor.detach().clone().contiguous()
            tensor_sha256 = hashlib.sha256(
                owned.view(state_v72.torch.uint8).numpy().tobytes()
            ).hexdigest()
            _require(
                source_key not in tensors,
                "V73E inverse source key collided",
            )
            tensors[source_key] = owned
            target_keys.append(target_key)
            records.append({
                "source_key": source_key,
                "target_key": target_key,
                "shape": list(owned.shape),
                "dtype": str(owned.dtype),
                "elements": int(owned.numel()),
                "tensor_sha256": tensor_sha256,
                "target_tensor_sha256": tensor_sha256,
                "tensor_bytes_preserved_exactly": True,
            })
    identity = state_v71.adapter_identity_no_clone_v71(tensors)
    key_mapping_sha256 = _guard.canonical_sha256([
        {
            "source_key": row["source_key"],
            "target_key": row["target_key"],
        }
        for row in records
    ])
    _require(
        len(records) == contract_v73e.STAGED_TENSOR_COUNT
        and identity.get("tensor_count")
        == contract_v73e.STAGED_TENSOR_COUNT
        and identity.get("elements")
        == contract_v73e.STAGED_ELEMENT_COUNT
        and identity.get("bytes") == contract_v73e.STAGED_TENSOR_BYTES
        and identity.get("ordered_key_sha256")
        == contract_v73e.CANONICAL_SOURCE_KEY_INVENTORY_SHA256
        and identity.get("sha256") == contract_v73e.CANONICAL_MASTER_SHA256
        and _guard.canonical_sha256(target_keys)
        == contract_v73e.STAGED_TARGET_KEY_INVENTORY_SHA256
        and _guard.canonical_sha256(records)
        == contract_v73e.STAGED_TRANSFORM_IDENTITY_SHA256
        and key_mapping_sha256 == contract_v73e.INVERSE_KEY_MAPPING_SHA256,
        "V73E staged inverse transform identity changed",
    )
    return tensors


def inverse_transform_proof_v73e():
    return {
        "schema": "qwen36-v73e-staged-inverse-transform-proof-v1",
        "operation": "exact_prefix_inverse_only",
        "source_tensor_namespace": contract_v73e.STAGED_TARGET_PREFIX,
        "canonical_tensor_namespace": contract_v73e.CANONICAL_SOURCE_PREFIX,
        "tensor_count": contract_v73e.STAGED_TENSOR_COUNT,
        "element_count": contract_v73e.STAGED_ELEMENT_COUNT,
        "tensor_arithmetic_performed": False,
        "tensor_cast_performed": False,
        "tensor_bytes_preserved_exactly": True,
        "inverse_key_mapping_sha256": contract_v73e.INVERSE_KEY_MAPPING_SHA256,
        "staged_transform_identity_sha256": (
            contract_v73e.STAGED_TRANSFORM_IDENTITY_SHA256
        ),
        "canonical_master_sha256": contract_v73e.CANONICAL_MASTER_SHA256,
        "canonical_runtime_values_sha256": (
            contract_v73e.CANONICAL_RUNTIME_VALUES_SHA256
        ),
        "historical_protected_source_opened_resolved_statted_or_hashed": False,
    }


class LoRAAdapterStateWorkerExtensionV73E(LoRAAdapterStateWorkerExtensionV72):
    def install_adapter_state_v41a(
        self,
        adapter_weights_path,
        adapter_config_path,
        expected_weights_sha256,
        expected_config_sha256,
    ):
        _require(
            Path(adapter_weights_path).resolve()
            == contract_v73e.STAGED_ADAPTER_WEIGHTS
            and Path(adapter_config_path).resolve()
            == contract_v73e.STAGED_ADAPTER_CONFIG
            and str(expected_weights_sha256)
            == contract_v73e.STAGED_ADAPTER_WEIGHTS_SHA256
            and str(expected_config_sha256)
            == contract_v73e.STAGED_ADAPTER_CONFIG_SHA256,
            "V73E canonical install must use the exact staged adapter",
        )
        with _STAGED_LOAD_LOCK:
            original_loader = state_v72._load_owned_master_v72
            _require(
                original_loader.__module__ == "eggroll_es_worker_lora_v72"
                and original_loader.__name__ == "_load_owned_master_v72",
                "V73E V72 owned-master loader changed",
            )
            state_v72._load_owned_master_v72 = inverse_staged_master_v73e
            try:
                result = super().install_adapter_state_v41a(
                    adapter_weights_path,
                    adapter_config_path,
                    expected_weights_sha256,
                    expected_config_sha256,
                )
            finally:
                state_v72._load_owned_master_v72 = original_loader
        _require(
            result.get("canonical_identity", {}).get("sha256")
            == contract_v73e.CANONICAL_MASTER_SHA256
            and result.get("canonical_identity", {}).get(
                "ordered_key_sha256"
            ) == contract_v73e.CANONICAL_SOURCE_KEY_INVENTORY_SHA256
            and result.get("materialization", {}).get("runtime_values_sha256")
            == contract_v73e.CANONICAL_RUNTIME_VALUES_SHA256,
            "V73E staged inverse install identity changed",
        )
        self._v73e_staged_inverse_install = {
            "schema": "qwen36-v73e-staged-inverse-install-v1",
            "complete": True,
            "staged_weights_sha256": (
                contract_v73e.STAGED_ADAPTER_WEIGHTS_SHA256
            ),
            "staged_config_sha256": (
                contract_v73e.STAGED_ADAPTER_CONFIG_SHA256
            ),
            "canonical_master_sha256": (
                contract_v73e.CANONICAL_MASTER_SHA256
            ),
            "canonical_runtime_values_sha256": (
                contract_v73e.CANONICAL_RUNTIME_VALUES_SHA256
            ),
            "inverse_transform_proof": inverse_transform_proof_v73e(),
            "historical_protected_source_opened_resolved_statted_or_hashed": (
                False
            ),
        }
        result.update({
            "staged_weights_sha256": (
                contract_v73e.STAGED_ADAPTER_WEIGHTS_SHA256
            ),
            "staged_config_sha256": contract_v73e.STAGED_ADAPTER_CONFIG_SHA256,
            "inverse_transform_proof": inverse_transform_proof_v73e(),
            "historical_protected_source_opened_resolved_statted_or_hashed": (
                False
            ),
        })
        return result

    def systems_only_path_guard_receipt_v73e(self):
        guard = _guard.receipt()
        staged_install = getattr(self, "_v73e_staged_inverse_install", None)
        result = {
            "schema": "qwen36-v73e-worker-bootstrap-receipt-v1",
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
            "staged_inverse_install_complete": isinstance(
                staged_install, dict
            ) and staged_install.get("complete") is True,
            "staged_inverse_install": staged_install,
            "quality_hpo_or_promotion_authorized": False,
        }
        result["receipt_sha256"] = _guard.canonical_sha256(result)
        return result
