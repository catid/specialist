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
        try:
            save_file(tensors, temporary, metadata={
                "schema": "eggroll-es-selected-runtime-snapshot-v38a",
                "final_identity_sha256": str(expected_final_sha256),
                "accepted_alpha": repr(expected_alpha),
                "layer_plan_sha256": self._v4_layer_plan_sha256,
            })
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        result.update({
            "written": True,
            "file_sha256": file_sha256(path),
            "file_bytes": path.stat().st_size,
            "tensor_count": len(tensors),
            "tensor_elements": sum(tensor.numel() for tensor in tensors.values()),
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
