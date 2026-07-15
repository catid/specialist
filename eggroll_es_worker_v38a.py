#!/usr/bin/env python3
"""V38A resident-sign worker with nonzero updates and sealed snapshots."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import eggroll_es_worker_v11c as worker_v11c


FROZEN_LAYER_PLANS_V38A = worker_v11c.FROZEN_LAYER_PLANS_V11C
validate_frozen_layer_plan_v38a = worker_v11c.validate_frozen_layer_plan_v11c


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def readback_selected_snapshot_v38a(
    worker,
    path: Path,
    selected,
    expected_metadata: dict[str, str],
    expected_selected_identity: dict,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> dict:
    """Reopen a snapshot and prove its selected bytes match the live state."""
    path = Path(path).resolve()
    selected = list(selected)
    expected_names = [name for name, _parameter in selected]
    if not expected_names or len(set(expected_names)) != len(expected_names):
        raise RuntimeError("v38a snapshot selected inventory is invalid")
    with safe_open(path, framework="pt") as source:
        metadata = source.metadata() or {}
        keys = list(source.keys())
    if metadata != expected_metadata:
        raise RuntimeError("v38a snapshot metadata changed on readback")
    if keys != sorted(expected_names):
        raise RuntimeError("v38a snapshot selected-name inventory changed on readback")
    tensors = load_file(path, device="cpu")
    if set(tensors) != set(expected_names) or len(tensors) != len(expected_names):
        raise RuntimeError("v38a snapshot tensor inventory changed on readback")
    reopened = []
    for name, parameter in selected:
        tensor = tensors[name]
        if (
            tensor.device.type != "cpu"
            or tensor.dtype != parameter.dtype
            or tuple(tensor.shape) != tuple(parameter.shape)
            or not tensor.is_contiguous()
        ):
            raise RuntimeError("v38a snapshot tensor metadata changed on readback")
        reopened.append((name, tensor))
    reopened_identity = worker._partition_identity_v4(
        "selected", reopened, int(chunk_bytes),
    )
    if reopened_identity != expected_selected_identity:
        raise RuntimeError("v38a snapshot selected bytes differ from live final state")
    return {
        "schema": "eggroll-es-selected-snapshot-readback-v38a",
        "verified": True,
        "metadata": metadata,
        "file_sha256": file_sha256(path),
        "file_bytes": path.stat().st_size,
        "tensor_count": len(reopened),
        "tensor_elements": sum(tensor.numel() for _name, tensor in reopened),
        "selected_identity": reopened_identity,
    }


class EqualUnitUpdateWorkerExtensionV38A(
    worker_v11c.ResidentSignAuditWorkerExtensionV11C,
):
    """Retain audited V4 updates and add selected-state persistence only."""

    def runtime_identity_v38a(self):
        return {
            "schema": "eggroll-es-worker-runtime-identity-v38a",
            "pid": os.getpid(),
            "inter_engine_rank": int(self.inter_pg.rank),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_current_device": (
                int(torch.cuda.current_device()) if torch.cuda.is_available() else None
            ),
        }

    def save_selected_snapshot_v38a(
        self, path, expected_final_sha256, expected_alpha,
    ):
        """Persist rank zero's selected packed tensors after a committed update."""
        self._require_no_pending_update_v3()
        expected_alpha = float(expected_alpha)
        if (
            self._v3_reference_fresh is not False
            or float(self._v3_accepted_alpha) != expected_alpha
            or int(self._v3_update_sequence) != 1
        ):
            raise RuntimeError("v38a snapshot requested before one committed update")
        current = self._partitioned_weight_state_v4(
            require_unselected_origin=True,
        )
        if (
            current != self._v3_current_identity
            or current.get("sha256") != str(expected_final_sha256)
            or current["unselected"] != self._v4_unselected_origin_identity
        ):
            raise RuntimeError("v38a snapshot state identity changed")
        path = Path(path).resolve()
        rank = int(self.inter_pg.rank)
        result = {
            "schema": "eggroll-es-selected-snapshot-v38a",
            "rank": rank,
            "written": False,
            "path": str(path),
            "accepted_alpha": expected_alpha,
            "final_identity": current,
            **self._binding_fields_v4(),
        }
        if rank != 0:
            return result
        if path.exists():
            raise FileExistsError(f"v38a snapshot exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(f"v38a snapshot temporary exists: {temporary}")
        _, selected = self._validated_parameters_v4()
        tensors = {
            name: parameter.detach().to(device="cpu", copy=True).contiguous()
            for name, parameter in selected
        }
        metadata = {
            "schema": "eggroll-es-selected-runtime-snapshot-v38a",
            "final_identity_sha256": str(expected_final_sha256),
            "accepted_alpha": repr(expected_alpha),
            "layer_plan_sha256": self._v4_layer_plan_sha256,
        }
        linked = False
        try:
            save_file(tensors, temporary, metadata=metadata)
            os.link(temporary, path)
            linked = True
            readback = readback_selected_snapshot_v38a(
                self,
                path,
                selected,
                metadata,
                current["selected"],
            )
        except BaseException:
            if linked:
                path.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        result.update({
            "written": True,
            "readback_verified": True,
            "readback": readback,
            "file_sha256": readback["file_sha256"],
            "file_bytes": readback["file_bytes"],
            "tensor_count": readback["tensor_count"],
            "tensor_elements": readback["tensor_elements"],
            "reopened_selected_identity": readback["selected_identity"],
        })
        return result

    def load_selected_snapshot_v38a(
        self, path, expected_file_sha256, expected_final_identity,
    ):
        """Load a sealed selected snapshot into a fresh exact-reference engine."""
        self._require_no_pending_update_v3()
        if self._v3_reference_fresh is not True:
            raise RuntimeError("v38a snapshot load requires a fresh base reference")
        path = Path(path).resolve()
        if file_sha256(path) != str(expected_file_sha256):
            raise RuntimeError("v38a snapshot file identity changed")
        with safe_open(path, framework="pt") as source:
            metadata = source.metadata() or {}
            keys = list(source.keys())
        _, selected = self._validated_parameters_v4()
        if keys != sorted(name for name, _parameter in selected):
            raise RuntimeError("v38a snapshot selected-name inventory changed")
        tensors = load_file(path, device="cpu")
        with torch.no_grad():
            for name, parameter in selected:
                tensor = tensors[name]
                if tensor.dtype != parameter.dtype or tensor.shape != parameter.shape:
                    raise RuntimeError("v38a snapshot tensor metadata changed")
                parameter.data.copy_(tensor.to(parameter.device), non_blocking=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        current = self._partitioned_weight_state_v4(require_unselected_origin=True)
        if (
            current != expected_final_identity
            or metadata.get("final_identity_sha256") != current["sha256"]
            or current["unselected"] != self._v4_unselected_origin_identity
        ):
            raise RuntimeError("v38a loaded snapshot identity changed")
        self._v3_current_identity = dict(current)
        self._v3_reference_fresh = False
        self._v3_update_sequence = 1
        self._v3_accepted_alpha = float(metadata["accepted_alpha"])
        return {
            "schema": "eggroll-es-selected-snapshot-loaded-v38a",
            "rank": int(self.inter_pg.rank),
            "file_sha256": expected_file_sha256,
            "current_identity": current,
            "accepted_alpha": self._v3_accepted_alpha,
            **self._binding_fields_v4(),
        }
