#!/usr/bin/env python3
"""Resident canonical-FP32 LoRA ES state for Qwen3.5/vLLM V41A.

The trainable state is always the original 70-tensor PEFT namespace in
unscaled FP32.  vLLM's 82 packed BF16 slot views are a derived inference
cache: A is duplicated where packing requires it and B is multiplied by the
adapter scale (64 / 32 == 2) before packed splitting.  No BF16 value is ever
used as the optimizer master or as the source of a restore.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import eggroll_es_worker_lora_topology_v40a as topology
from eggroll_es_worker_v3 import (
    DistributedExactAuditWorkerExtensionV3,
    canonical_sha256_v3,
    coefficient_sha256_v3,
    seed_shard_v3,
    validate_seed_coefficients_v3,
)


EXPECTED_TENSOR_COUNT_V41A = 70
EXPECTED_MASTER_ELEMENTS_V41A = 4_528_128
EXPECTED_RUNTIME_MODULES_V41A = 23
EXPECTED_RUNTIME_VIEWS_V41A = 82
EXPECTED_RUNTIME_ELEMENTS_V41A = 4_921_344
EXPECTED_BASE_ELEMENTS_V41A = 142_999_552
EXPECTED_BASE_BYTES_V41A = 285_999_104
REQUIRED_WORLD_SIZE_V41A = 4
ADAPTER_ID_V41A = 1
ADAPTER_SLOT_V41A = 0


def file_sha256_v41a(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256_v41a(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _validate_master_v41a(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(tensors, dict) or len(tensors) != EXPECTED_TENSOR_COUNT_V41A:
        raise RuntimeError("v41a canonical adapter must contain exactly 70 tensors")
    result = {}
    elements = 0
    for key in sorted(tensors):
        tensor = tensors[key]
        if (
            not key.startswith("base_model.model.model.layers.")
            or not key.endswith((".lora_A.weight", ".lora_B.weight"))
            or not isinstance(tensor, torch.Tensor)
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or tensor.ndim != 2
        ):
            raise RuntimeError(f"v41a invalid canonical tensor: {key}")
        value = tensor.detach().clone().contiguous()
        result[key] = value
        elements += int(value.numel())
    if elements != EXPECTED_MASTER_ELEMENTS_V41A:
        raise RuntimeError("v41a canonical adapter element count changed")
    return result


def adapter_identity_v41a(tensors: dict[str, torch.Tensor]) -> dict:
    tensors = _validate_master_v41a(tensors)
    records = [{
        "key": key, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "elements": int(tensor.numel()), "sha256": _tensor_sha256_v41a(tensor),
    } for key, tensor in tensors.items()]
    return {
        "schema": "canonical-peft-fp32-state-v41a",
        "sha256": canonical_sha256_v3(records),
        "tensor_count": len(records),
        "elements": sum(item["elements"] for item in records),
        "bytes": sum(item["elements"] * 4 for item in records),
        "ordered_key_sha256": canonical_sha256_v3([item["key"] for item in records]),
        "tensors": records,
    }


def _clone_master_v41a(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: tensor.detach().clone().contiguous()
            for key, tensor in _validate_master_v41a(tensors).items()}


def _noise_seed_v41a(seed: int, key: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or int(seed) < 0:
        raise ValueError("v41a noise seed must be a nonnegative integer")
    raw = hashlib.sha256(f"{int(seed)}\0{key}".encode("utf-8")).digest()
    return int.from_bytes(raw[:8], "little") & ((1 << 63) - 1)


def noise_like_v41a(
    tensor: torch.Tensor, key: str, seed: int, device: torch.device | str,
) -> torch.Tensor:
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(_noise_seed_v41a(seed, key))
    return torch.randn(
        tensor.shape, dtype=torch.float32, device=device, generator=generator,
    )


def antithetic_candidate_v41a(
    master: dict[str, torch.Tensor], seed: int, sigma: float, sign: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    master = _validate_master_v41a(master)
    sigma = float(sigma)
    sign = int(sign)
    if not math.isfinite(sigma) or sigma <= 0.0 or sign not in (-1, 1):
        raise ValueError("v41a perturbation requires finite sigma>0 and sign +/-1")
    device = torch.device(device)
    result = {}
    for key, tensor in master.items():
        base = tensor.to(device=device, dtype=torch.float32)
        noise = noise_like_v41a(tensor, key, seed, device)
        result[key] = base.add(noise, alpha=sign * sigma)
    return result


def zero_zero_antithetic_degeneracy_v41a(
    a_noise: torch.Tensor, b_noise: torch.Tensor, sigma: float = 1.0,
) -> dict:
    """Prove the simultaneous A/B zero-origin central difference is zero."""
    if a_noise.ndim != 2 or b_noise.ndim != 2 or b_noise.shape[1] != a_noise.shape[0]:
        raise ValueError("v41a zero-zero degeneracy matrices are incompatible")
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma == 0.0:
        raise ValueError("v41a zero-zero degeneracy sigma must be finite/nonzero")
    plus = (sigma * b_noise.float()) @ (sigma * a_noise.float())
    minus = (-sigma * b_noise.float()) @ (-sigma * a_noise.float())
    central = 0.5 * (plus - minus)
    passed = torch.equal(plus, minus) and int(torch.count_nonzero(central)) == 0
    return {
        "schema": "lora-zero-zero-antithetic-degeneracy-v41a",
        "passed": passed,
        "plus_equals_minus": bool(torch.equal(plus, minus)),
        "central_nonzero_elements": int(torch.count_nonzero(central)),
        "explanation": "simultaneous sign reversal leaves B@A unchanged at A=B=0",
    }


def _source_inventory_v41a(path: Path) -> dict[str, torch.Tensor]:
    with safe_open(Path(path), framework="pt", device="cpu") as handle:
        tensors = {}
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.float32:
                raise RuntimeError(f"v41a source PEFT tensor is not FP32: {key}")
            tensors[key] = tensor.contiguous()
    return _validate_master_v41a(tensors)


def _runtime_assignments_v41a(manager, master: dict[str, torch.Tensor]) -> list[dict]:
    assignments = []
    for key, tensor in _validate_master_v41a(master).items():
        logical, side = topology._source_parts(key)
        target, slices = topology._runtime_target(logical)
        runtime_name, module = topology._suffix_match(manager.modules, target)
        for segment_index, slice_index in enumerate(slices):
            stacked = (module.lora_a_stacked if side == "A"
                       else module.lora_b_stacked)[slice_index]
            assignments.append({
                "peft_key": key, "logical_module": logical, "side": side,
                "runtime_module": runtime_name, "slot": ADAPTER_SLOT_V41A,
                "slice_index": int(slice_index), "segment_index": segment_index,
                "segment_count": len(slices), "source_shape": list(tensor.shape),
                "parent_shape": list(stacked.shape),
                "runtime_shape": list(stacked[ADAPTER_SLOT_V41A, 0].shape),
                "output_slices": list(module.output_slices),
            })
    signatures = [(item["runtime_module"], item["side"], item["slice_index"])
                  for item in assignments]
    if len(assignments) != EXPECTED_RUNTIME_VIEWS_V41A or len(set(signatures)) != len(signatures):
        raise RuntimeError("v41a canonical-to-runtime assignment coverage changed")
    return assignments


def _expected_runtime_value_v41a(
    tensor: torch.Tensor, side: str, slices: tuple[int, ...],
    slice_index: int, output_slices: tuple[int, ...] | list[int], scale: float,
) -> torch.Tensor:
    if side == "A":
        return tensor.to(dtype=torch.bfloat16)
    scaled = tensor.float().mul(float(scale)).to(dtype=torch.bfloat16)
    if len(slices) == 1:
        return scaled
    sizes = [int(output_slices[index]) for index in slices]
    chunks = torch.split(scaled, sizes, dim=0)
    return chunks[list(slices).index(int(slice_index))]


class LoRAAdapterStateWorkerExtensionV41A(DistributedExactAuditWorkerExtensionV3):
    """Canonical adapter-state ES operations over vLLM's resident LoRA slot."""

    def _manager_v41a(self):
        manager = topology._manager(self)
        if list(manager.lora_index_to_id) != [ADAPTER_ID_V41A]:
            raise RuntimeError("v41a requires adapter id 1 in sole slot 0")
        return manager

    def _require_installed_v41a(self):
        if not getattr(self, "_v41_installed", False):
            raise RuntimeError("v41a canonical adapter state is not installed")

    def _require_quiescent_v41a(self):
        self._require_installed_v41a()
        if getattr(self, "_v41_active_perturbation", None) is not None:
            raise RuntimeError("v41a adapter is currently perturbed")
        if getattr(self, "_v41_pending_update", None) is not None:
            raise RuntimeError("v41a adapter update is pending")
        if getattr(self, "_v41_committed_rollback", None) is not None:
            raise RuntimeError("v41a committed update awaits finalize or rollback")

    def _base_check_v41a(self, phase: str) -> dict:
        manager = self._manager_v41a()
        current = topology._base_identity(manager.modules, self._v41_runtime_names)
        origin = self._v41_base_identity
        if (
            current["tensor_count"] != EXPECTED_RUNTIME_MODULES_V41A
            or current["elements"] != EXPECTED_BASE_ELEMENTS_V41A
            or current["bytes"] != EXPECTED_BASE_BYTES_V41A
            or current["inventory_sha256"] != origin["inventory_sha256"]
        ):
            raise RuntimeError(f"v41a relevant base_layer.weight drifted at {phase}")
        return {
            "phase": str(phase), "unchanged": True,
            "tensor_count": current["tensor_count"], "elements": current["elements"],
            "bytes": current["bytes"], "inventory_sha256": current["inventory_sha256"],
        }

    def _materialize_v41a(self, tensors: dict[str, torch.Tensor], phase: str) -> dict:
        tensors = _validate_master_v41a(tensors)
        manager = self._manager_v41a()
        assignments = self._v41_assignments
        modules = manager.modules
        runtime_names = {item["runtime_module"] for item in assignments}
        with torch.no_grad():
            for name in sorted(runtime_names):
                modules[name].reset_lora(ADAPTER_SLOT_V41A)
            records = []
            storage = []
            for item in assignments:
                source = tensors[item["peft_key"]]
                logical, side = topology._source_parts(item["peft_key"])
                _target, slices = topology._runtime_target(logical)
                module = modules[item["runtime_module"]]
                expected = _expected_runtime_value_v41a(
                    source, side, slices, item["slice_index"],
                    module.output_slices, self._v41_scale,
                )
                stacked = (module.lora_a_stacked if side == "A"
                           else module.lora_b_stacked)[item["slice_index"]]
                view = stacked[ADAPTER_SLOT_V41A, 0]
                if view.shape != expected.shape or view.dtype != torch.bfloat16:
                    raise RuntimeError(f"v41a runtime view metadata changed: {item['peft_key']}")
                view.copy_(expected.to(device=view.device), non_blocking=False)
                if not torch.equal(view.cpu(), expected.cpu()):
                    raise RuntimeError(f"v41a runtime materialization mismatch: {item['peft_key']}")
                records.append({
                    **item, "dtype": str(view.dtype), "elements": int(view.numel()),
                    "sha256": _tensor_sha256_v41a(view),
                })
                storage.append({
                    "signature": [item["runtime_module"], side, item["slice_index"]],
                    "storage_data_ptr": int(stacked.untyped_storage().data_ptr()),
                    "view_storage_data_ptr": int(view.untyped_storage().data_ptr()),
                    "view_storage_offset": int(view.storage_offset()),
                    "view_aliases_parent": (
                        stacked.untyped_storage().data_ptr()
                        == view.untyped_storage().data_ptr()
                    ),
                })
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elements = sum(item["elements"] for item in records)
        pointers = [item["storage_data_ptr"] for item in storage]
        if (
            len(records) != EXPECTED_RUNTIME_VIEWS_V41A
            or elements != EXPECTED_RUNTIME_ELEMENTS_V41A
            or len(set(pointers)) != EXPECTED_RUNTIME_VIEWS_V41A
            or not all(item["view_aliases_parent"] for item in storage)
        ):
            raise RuntimeError("v41a runtime residency/storage coverage changed")
        compact = [{key: value for key, value in item.items()
                    if key not in {"storage_data_ptr", "view_storage_data_ptr"}}
                   for item in storage]
        certificate = {
            "schema": "canonical-to-vllm-lora-materialization-v41a",
            "phase": str(phase), "adapter_id": 1, "slot": 0,
            "source_tensor_count": len(tensors),
            "source_elements": sum(tensor.numel() for tensor in tensors.values()),
            "runtime_module_count": len(runtime_names),
            "runtime_view_count": len(records), "runtime_elements": elements,
            "runtime_dtype": "torch.bfloat16", "b_scale": self._v41_scale,
            "a_duplication_and_b_splitting_verified": True,
            "unique_parent_storage_count": len(set(pointers)),
            "runtime_views_share_no_parent_storage": len(set(pointers)) == len(pointers),
            "slot_views_alias_parent_buffers": all(
                item["view_aliases_parent"] for item in storage
            ),
            "storage_layout_sha256": canonical_sha256_v3(compact),
            "runtime_values_sha256": canonical_sha256_v3(records),
        }
        self._v41_active_materialization = certificate
        return certificate

    def _verify_master_materialized_v41a(self, phase: str) -> dict:
        expected = self._materialize_v41a(self._v41_master, phase)
        identity = adapter_identity_v41a(self._v41_master)
        if identity != self._v41_current_identity:
            raise RuntimeError("v41a canonical master identity changed")
        return {"master_identity": identity, "materialization": expected}

    def install_adapter_state_v41a(
        self, adapter_weights_path, adapter_config_path,
        expected_weights_sha256, expected_config_sha256,
    ):
        if getattr(self, "_v41_installed", False):
            raise RuntimeError("v41a adapter state installation is one-shot")
        weights_path = Path(adapter_weights_path).resolve()
        config_path = Path(adapter_config_path).resolve()
        if (
            file_sha256_v41a(weights_path) != str(expected_weights_sha256)
            or file_sha256_v41a(config_path) != str(expected_config_sha256)
        ):
            raise RuntimeError("v41a source adapter identity changed")
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if (
            config.get("r") != 32 or config.get("lora_alpha") != 64
            or config.get("bias") != "none"
        ):
            raise RuntimeError("v41a source adapter configuration changed")
        manager = self._manager_v41a()
        master = _source_inventory_v41a(weights_path)
        all_a_zero = all(
            int(torch.count_nonzero(tensor)) == 0
            for key, tensor in master.items() if ".lora_A." in key
        )
        all_b_zero = all(
            int(torch.count_nonzero(tensor)) == 0
            for key, tensor in master.items() if ".lora_B." in key
        )
        if all_a_zero and all_b_zero:
            raise RuntimeError(
                "v41a simultaneous all-zero A/B initialization is ES-degenerate"
            )
        identity = adapter_identity_v41a(master)
        assignments = _runtime_assignments_v41a(manager, master)
        runtime_names = {item["runtime_module"] for item in assignments}
        base = topology._base_identity(manager.modules, runtime_names)
        if (
            len(runtime_names) != EXPECTED_RUNTIME_MODULES_V41A
            or base["tensor_count"] != EXPECTED_RUNTIME_MODULES_V41A
            or base["elements"] != EXPECTED_BASE_ELEMENTS_V41A
            or base["bytes"] != EXPECTED_BASE_BYTES_V41A
        ):
            raise RuntimeError("v41a relevant base-layer inventory changed")
        self._v41_installed = True
        self._v41_master = _clone_master_v41a(master)
        self._v41_current_identity = identity
        self._v41_reference = _clone_master_v41a(master)
        self._v41_reference_identity = identity
        self._v41_reference_generation = 1
        self._v41_reference_fresh = True
        self._v41_config_bytes = config_bytes
        self._v41_config = config
        self._v41_source_weights_path = str(weights_path)
        self._v41_source_config_path = str(config_path)
        self._v41_source_weights_sha256 = str(expected_weights_sha256)
        self._v41_source_config_sha256 = str(expected_config_sha256)
        self._v41_scale = 2.0
        self._v41_assignments = assignments
        self._v41_runtime_names = runtime_names
        self._v41_base_identity = base
        self._v41_active_perturbation = None
        self._v41_pending_update = None
        self._v41_committed_rollback = None
        self._v41_update_sequence = 0
        self._v41_active_plan_id = None
        try:
            materialization = self._materialize_v41a(master, "install")
            base_check = self._base_check_v41a("install")
        except Exception:
            for name in [key for key in vars(self) if key.startswith("_v41")]:
                delattr(self, name)
            raise
        assignment_certificate = [{
            key: item[key] for key in (
                "peft_key", "side", "runtime_module", "slot", "slice_index",
                "segment_index", "segment_count", "source_shape", "runtime_shape",
            )
        } for item in assignments]
        base_origin = {
            "tensor_count": base["tensor_count"], "elements": base["elements"],
            "bytes": base["bytes"], "inventory_sha256": base["inventory_sha256"],
            "tensors": [{
                key: value for key, value in item.items()
                if key != "storage_data_ptr"
            } for item in base["tensors"]],
        }
        return {
            "schema": "canonical-lora-adapter-installed-v41a",
            "installed": True, "adapter_id": 1, "slot": 0,
            "source_weights_sha256": expected_weights_sha256,
            "source_config_sha256": expected_config_sha256,
            "canonical_identity": identity,
            "assignment_count": len(assignments),
            "assignment_sha256": canonical_sha256_v3(assignment_certificate),
            "assignments": assignment_certificate,
            "materialization": materialization, "base_identity": base_check,
            "base_origin_inventory": base_origin,
            "zero_zero_degeneracy_guard": {
                "all_a_zero": all_a_zero,
                "all_b_zero": all_b_zero,
                "simultaneous_all_zero_forbidden": True,
            },
        }

    def adapter_state_certificate_v41a(self):
        self._require_quiescent_v41a()
        verified = self._verify_master_materialized_v41a("state_certificate")
        base = self._base_check_v41a("state_certificate")
        return {
            "schema": "canonical-lora-state-certificate-v41a",
            "adapter_id": 1, "slot": 0,
            "current_identity": verified["master_identity"],
            "reference_identity": self._v41_reference_identity,
            "reference_generation": self._v41_reference_generation,
            "reference_fresh": self._v41_reference_fresh,
            "update_sequence": self._v41_update_sequence,
            "materialization": verified["materialization"], "base_identity": base,
        }

    def capture_adapter_reference_v41a(self):
        self._require_quiescent_v41a()
        self._verify_master_materialized_v41a("reference_capture")
        self._base_check_v41a("reference_capture")
        self._v41_reference = _clone_master_v41a(self._v41_master)
        self._v41_reference_identity = adapter_identity_v41a(self._v41_reference)
        self._v41_reference_generation += 1
        self._v41_reference_fresh = True
        return {
            "schema": "canonical-lora-reference-captured-v41a",
            "reference_generation": self._v41_reference_generation,
            "reference_identity": self._v41_reference_identity,
        }

    def materialize_antithetic_adapter_v41a(
        self, seed, sigma, sign, expected_master_sha256,
    ):
        self._require_quiescent_v41a()
        if not self._v41_reference_fresh or self._v41_reference_identity != self._v41_current_identity:
            raise RuntimeError("v41a perturbation requires a fresh exact reference")
        if self._v41_current_identity["sha256"] != str(expected_master_sha256):
            raise RuntimeError("v41a perturbation base identity changed")
        candidate_device = getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        candidate = antithetic_candidate_v41a(
            self._v41_master, int(seed), float(sigma), int(sign), candidate_device,
        )
        candidate_cpu = {key: tensor.cpu().contiguous() for key, tensor in candidate.items()}
        certificate = self._materialize_v41a(candidate_cpu, "antithetic_perturbation")
        candidate_identity = adapter_identity_v41a(candidate_cpu)
        self._base_check_v41a("antithetic_perturbation")
        self._v41_active_perturbation = {
            "seed": int(seed), "sigma": float(sigma), "sign": int(sign),
            "base_identity": self._v41_current_identity,
            "candidate_identity": candidate_identity,
        }
        return {
            "schema": "canonical-lora-antithetic-materialized-v41a",
            **self._v41_active_perturbation,
            "materialization": certificate,
            "master_unchanged": adapter_identity_v41a(self._v41_master)
            == self._v41_current_identity,
            "restore_policy": "copy_exact_canonical_master_never_algebraic_bf16",
        }

    def restore_adapter_master_v41a(self):
        self._require_installed_v41a()
        if self._v41_pending_update is not None:
            raise RuntimeError("v41a perturbation restore is forbidden during update")
        prior = self._v41_active_perturbation
        if prior is None:
            raise RuntimeError("v41a no active perturbation to restore")
        materialization = self._materialize_v41a(self._v41_master, "exact_restore")
        identity = adapter_identity_v41a(self._v41_master)
        if identity != self._v41_current_identity:
            raise RuntimeError("v41a canonical master changed during perturbation")
        self._base_check_v41a("exact_restore")
        self._v41_active_perturbation = None
        return {
            "schema": "canonical-lora-exact-restore-v41a", "restored": True,
            "restored_identity": identity, "prior_perturbation": prior,
            "materialization": materialization,
            "algebraic_bf16_restore_used": False,
        }

    def prepare_sharded_adapter_update_v41a(
        self, seeds, coefficients, coefficient_sha256, population_size,
        expected_world_size, alpha, plan_id, expected_master_sha256,
        reference_generation,
    ):
        self._require_quiescent_v41a()
        if int(expected_world_size) != REQUIRED_WORLD_SIZE_V41A:
            raise RuntimeError("v41a update requires exactly four ranks")
        communicator = self._communicator_state_v3(expected_world_size)
        seeds, coefficients = validate_seed_coefficients_v3(
            seeds, coefficients, population_size, expected_world_size,
        )
        if coefficient_sha256_v3(seeds, coefficients) != str(coefficient_sha256):
            raise RuntimeError("v41a coefficient identity changed")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha == 0.0:
            raise ValueError("v41a update alpha must be finite/nonzero")
        if (
            self._v41_current_identity["sha256"] != str(expected_master_sha256)
            or int(reference_generation) != self._v41_reference_generation
            or not self._v41_reference_fresh
            or self._v41_reference_identity != self._v41_current_identity
        ):
            raise RuntimeError("v41a update used a stale canonical reference")
        shard = seed_shard_v3(
            seeds, coefficients, communicator["rank"], communicator["world_size"],
        )
        manifest = {
            "schema": "canonical-lora-sharded-update-manifest-v41a",
            "seeds": seeds, "coefficients": coefficients,
            "coefficient_sha256": coefficient_sha256,
            "population_size": int(population_size),
            "world_size": communicator["world_size"], "alpha": alpha,
            "plan_id": str(plan_id), "expected_master_sha256": expected_master_sha256,
            "reference_generation": int(reference_generation),
            "update_sequence": self._v41_update_sequence + 1,
        }
        if not manifest["plan_id"]:
            raise ValueError("v41a update plan id is empty")
        manifest_sha = canonical_sha256_v3(manifest)
        self._v41_pending_update = {
            "phase": "prepared", "manifest": manifest,
            "manifest_sha256": manifest_sha, "shard": shard,
            "rollback_master": _clone_master_v41a(self._v41_master),
            "rollback_identity": self._v41_current_identity,
            "rollback_update_sequence": self._v41_update_sequence,
            "rollback_active_plan_id": self._v41_active_plan_id,
            "rollback_reference_fresh": self._v41_reference_fresh,
        }
        return {
            "schema": "canonical-lora-sharded-update-prepared-v41a",
            "prepared": True, "manifest_sha256": manifest_sha,
            "rank": communicator["rank"], "world_size": communicator["world_size"],
            "shard_indices": shard["indices"], "shard_seeds": shard["seeds"],
            "shard_pair_sha256": canonical_sha256_v3({
                "seeds": shard["seeds"], "coefficients": shard["coefficients"],
            }),
            "master_identity": self._v41_current_identity,
        }

    def _rollback_pending_v41a(self, phase: str) -> dict:
        pending = self._v41_pending_update
        if not isinstance(pending, dict):
            raise RuntimeError("v41a no pending update to roll back")
        rollback = _clone_master_v41a(pending["rollback_master"])
        rollback_identity = adapter_identity_v41a(rollback)
        if rollback_identity != pending["rollback_identity"]:
            raise RuntimeError("v41a rollback master identity changed")
        self._v41_master = rollback
        self._v41_current_identity = rollback_identity
        materialization = self._materialize_v41a(rollback, f"rollback_{phase}")
        base = self._base_check_v41a(f"rollback_{phase}")
        self._v41_pending_update = None
        return {
            "schema": "canonical-lora-update-rollback-v41a", "rolled_back": True,
            "phase": phase, "identity": rollback_identity,
            "materialization": materialization, "base_identity": base,
        }

    def execute_sharded_adapter_update_v41a(self, manifest_sha256):
        pending = self._v41_pending_update
        if not isinstance(pending, dict) or pending.get("phase") != "prepared":
            raise RuntimeError("v41a adapter update was not prepared")
        if pending["manifest_sha256"] != str(manifest_sha256):
            raise RuntimeError("v41a prepared manifest identity changed")
        manifest, shard = pending["manifest"], pending["shard"]
        communicator = self._communicator_state_v3(manifest["world_size"])
        device = torch.device(getattr(self, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        candidate = {}
        reduced_elements = 0
        try:
            for key, master in self._v41_master.items():
                accumulator = torch.zeros(master.shape, dtype=torch.float32, device=device)
                for seed, coefficient in zip(
                    shard["seeds"], shard["coefficients"], strict=True,
                ):
                    noise = noise_like_v41a(master, key, seed, device)
                    accumulator.add_(noise, alpha=float(coefficient))
                stream = torch.cuda.current_stream() if accumulator.is_cuda else None
                reduced = self.inter_pg.all_reduce(
                    accumulator, out_tensor=accumulator, stream=stream,
                )
                if (
                    reduced is None or reduced.dtype != torch.float32
                    or reduced.shape != master.shape
                ):
                    raise RuntimeError("v41a PyNccl reduction returned incompatible tensor")
                reduced.mul_(float(manifest["alpha"]) / manifest["population_size"])
                candidate[key] = master.add(reduced.cpu()).contiguous()
                reduced_elements += int(reduced.numel())
            candidate = _validate_master_v41a(candidate)
            candidate_identity = adapter_identity_v41a(candidate)
            materialization = self._materialize_v41a(candidate, "executed_candidate")
            base = self._base_check_v41a("post_update_execution")
        except Exception:
            self._rollback_pending_v41a("execute_failure")
            raise
        pending["phase"] = "executed"
        pending["candidate_master"] = candidate
        pending["candidate_identity"] = candidate_identity
        return {
            "schema": "canonical-lora-sharded-update-executed-v41a",
            "executed": True, "manifest_sha256": manifest_sha256,
            "rank": communicator["rank"], "world_size": communicator["world_size"],
            "collective_dtype": "torch.float32", "tensor_count": len(candidate),
            "reduced_elements": reduced_elements,
            "candidate_identity": candidate_identity,
            "materialization": materialization, "base_identity": base,
            "master_committed": False,
        }

    def commit_sharded_adapter_update_v41a(
        self, manifest_sha256, expected_final_sha256,
    ):
        pending = self._v41_pending_update
        if not isinstance(pending, dict) or pending.get("phase") != "executed":
            raise RuntimeError("v41a adapter update was not executed")
        if (
            pending["manifest_sha256"] != str(manifest_sha256)
            or pending["candidate_identity"]["sha256"] != str(expected_final_sha256)
        ):
            self._rollback_pending_v41a("commit_identity_failure")
            raise RuntimeError("v41a cross-rank final identity changed")
        old_master = _clone_master_v41a(self._v41_master)
        old_identity = self._v41_current_identity
        try:
            self._v41_master = _clone_master_v41a(pending["candidate_master"])
            self._v41_current_identity = adapter_identity_v41a(self._v41_master)
            materialization = self._materialize_v41a(self._v41_master, "commit")
            base = self._base_check_v41a("post_update_commit")
            if self._v41_current_identity["sha256"] != str(expected_final_sha256):
                raise RuntimeError("v41a committed master identity changed")
        except Exception:
            self._v41_master = old_master
            self._v41_current_identity = old_identity
            self._materialize_v41a(old_master, "commit_failure_rollback")
            self._base_check_v41a("commit_failure_rollback")
            self._v41_pending_update = None
            raise
        manifest = pending["manifest"]
        self._v41_update_sequence = int(manifest["update_sequence"])
        self._v41_active_plan_id = manifest["plan_id"]
        self._v41_reference_fresh = False
        self._v41_committed_rollback = {
            "manifest_sha256": str(manifest_sha256),
            "rollback_master": _clone_master_v41a(pending["rollback_master"]),
            "rollback_identity": pending["rollback_identity"],
            "rollback_update_sequence": pending["rollback_update_sequence"],
            "rollback_active_plan_id": pending["rollback_active_plan_id"],
            "rollback_reference_fresh": pending["rollback_reference_fresh"],
            "committed_identity": self._v41_current_identity,
        }
        self._v41_pending_update = None
        return {
            "schema": "canonical-lora-sharded-update-committed-v41a",
            "committed": True, "manifest_sha256": manifest_sha256,
            "rank": int(self.inter_pg.rank), "final_identity": self._v41_current_identity,
            "update_sequence": self._v41_update_sequence,
            "reference_fresh_for_population": False,
            "requires_cross_rank_finalize": True,
            "materialization": materialization, "base_identity": base,
        }

    def abort_sharded_adapter_update_v41a(self, manifest_sha256):
        pending = self._v41_pending_update
        if isinstance(pending, dict):
            if pending.get("manifest_sha256") != str(manifest_sha256):
                raise RuntimeError("v41a abort manifest does not match pending update")
            return self._rollback_pending_v41a("controller_abort")
        committed = self._v41_committed_rollback
        if (
            not isinstance(committed, dict)
            or committed.get("manifest_sha256") != str(manifest_sha256)
        ):
            raise RuntimeError("v41a abort manifest does not match committed update")
        rollback = _clone_master_v41a(committed["rollback_master"])
        identity = adapter_identity_v41a(rollback)
        if identity != committed["rollback_identity"]:
            raise RuntimeError("v41a committed rollback identity changed")
        self._v41_master = rollback
        self._v41_current_identity = identity
        self._v41_update_sequence = committed["rollback_update_sequence"]
        self._v41_active_plan_id = committed["rollback_active_plan_id"]
        self._v41_reference_fresh = committed["rollback_reference_fresh"]
        materialization = self._materialize_v41a(
            rollback, "partial_commit_controller_abort",
        )
        base = self._base_check_v41a("partial_commit_controller_abort")
        self._v41_committed_rollback = None
        return {
            "schema": "canonical-lora-partial-commit-rollback-v41a",
            "rolled_back": True, "manifest_sha256": manifest_sha256,
            "identity": identity, "update_sequence": self._v41_update_sequence,
            "reference_fresh": self._v41_reference_fresh,
            "materialization": materialization, "base_identity": base,
        }

    def finalize_sharded_adapter_update_v41a(
        self, manifest_sha256, expected_final_sha256,
    ):
        committed = self._v41_committed_rollback
        if (
            not isinstance(committed, dict)
            or committed.get("manifest_sha256") != str(manifest_sha256)
            or self._v41_current_identity["sha256"] != str(expected_final_sha256)
            or committed.get("committed_identity") != self._v41_current_identity
        ):
            raise RuntimeError("v41a committed update finalize identity changed")
        materialization = self._materialize_v41a(
            self._v41_master, "cross_rank_finalize",
        )
        base = self._base_check_v41a("cross_rank_finalize")
        self._v41_committed_rollback = None
        return {
            "schema": "canonical-lora-sharded-update-finalized-v41a",
            "finalized": True, "manifest_sha256": manifest_sha256,
            "rank": int(self.inter_pg.rank),
            "final_identity": self._v41_current_identity,
            "reference_fresh_for_population": False,
            "materialization": materialization, "base_identity": base,
        }

    def save_adapter_snapshot_v41a(
        self, output_directory, expected_master_sha256,
    ):
        self._require_quiescent_v41a()
        if self._v41_current_identity["sha256"] != str(expected_master_sha256):
            raise RuntimeError("v41a snapshot master identity changed")
        verified = self._verify_master_materialized_v41a("snapshot")
        self._base_check_v41a("snapshot")
        rank = int(self.inter_pg.rank)
        output = Path(output_directory).resolve()
        result = {
            "schema": "canonical-peft-fp32-snapshot-v41a", "rank": rank,
            "written": False, "directory": str(output),
            "master_identity": self._v41_current_identity,
        }
        if rank != 0:
            return result
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        weights = output / "adapter_model.safetensors"
        config = output / "adapter_config.json"
        temporary_weights = output / f".adapter_model.safetensors.tmp-{os.getpid()}"
        temporary_config = output / f".adapter_config.json.tmp-{os.getpid()}"
        try:
            save_file(
                _clone_master_v41a(self._v41_master), temporary_weights,
                metadata={
                    "format": "pt", "schema": "canonical-peft-fp32-v41a",
                    "master_sha256": self._v41_current_identity["sha256"],
                },
            )
            temporary_config.write_bytes(self._v41_config_bytes)
            os.link(temporary_weights, weights)
            os.link(temporary_config, config)
            readback_master = _source_inventory_v41a(weights)
            readback_identity = adapter_identity_v41a(readback_master)
            if (
                readback_identity != self._v41_current_identity
                or config.read_bytes() != self._v41_config_bytes
                or file_sha256_v41a(config) != self._v41_source_config_sha256
            ):
                raise RuntimeError("v41a PEFT snapshot readback changed")
        except Exception:
            weights.unlink(missing_ok=True); config.unlink(missing_ok=True)
            temporary_weights.unlink(missing_ok=True); temporary_config.unlink(missing_ok=True)
            try: output.rmdir()
            except OSError: pass
            raise
        finally:
            temporary_weights.unlink(missing_ok=True); temporary_config.unlink(missing_ok=True)
        result.update({
            "written": True, "weights_path": str(weights), "config_path": str(config),
            "weights_sha256": file_sha256_v41a(weights),
            "config_sha256": file_sha256_v41a(config),
            "readback_verified": True, "readback_identity": readback_identity,
            "original_canonical_key_namespace": True,
            "unscaled_fp32_master_persisted": True,
            "materialization_at_snapshot": verified["materialization"],
        })
        return result
