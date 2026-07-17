#!/usr/bin/env python3
"""Versioned one-copy canonical LoRA host state over the V71 audit lifecycle."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

import eggroll_es_audit_contract_v71 as audit_v71
import eggroll_es_host_state_contract_v72 as contract_v72
import eggroll_es_worker_lora_v41a as state_v41a
from eggroll_es_worker_lora_v71 import (
    LoRAAdapterStateWorkerExtensionV71,
    _validate_adapter_no_clone_v71,
    adapter_identity_no_clone_v71,
)


def _load_owned_master_v72(path):
    """Detach once from the source mmap and retain that copy as the master."""
    path = Path(path)
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in sorted(handle.keys()):
            source = handle.get_tensor(key)
            if source.dtype != torch.float32:
                raise RuntimeError(f"v72 source PEFT tensor is not FP32: {key}")
            tensors[key] = source.detach().clone().contiguous()
    return _validate_adapter_no_clone_v71(tensors)


def _runtime_assignments_no_clone_v72(manager, master):
    master = _validate_adapter_no_clone_v71(master)
    assignments = []
    for key in sorted(master):
        tensor = master[key]
        logical, side = state_v41a.topology._source_parts(key)
        target, slices = state_v41a.topology._runtime_target(logical)
        runtime_name, module = state_v41a.topology._suffix_match(
            manager.modules, target
        )
        for segment_index, slice_index in enumerate(slices):
            stacked = (
                module.lora_a_stacked
                if side == "A" else module.lora_b_stacked
            )[slice_index]
            assignments.append({
                "peft_key": key,
                "logical_module": logical,
                "side": side,
                "runtime_module": runtime_name,
                "slot": state_v41a.ADAPTER_SLOT_V41A,
                "slice_index": int(slice_index),
                "segment_index": segment_index,
                "segment_count": len(slices),
                "source_shape": list(tensor.shape),
                "parent_shape": list(stacked.shape),
                "runtime_shape": list(
                    stacked[state_v41a.ADAPTER_SLOT_V41A, 0].shape
                ),
                "output_slices": list(module.output_slices),
            })
    signatures = [
        (item["runtime_module"], item["side"], item["slice_index"])
        for item in assignments
    ]
    if (
        len(assignments) != state_v41a.EXPECTED_RUNTIME_VIEWS_V41A
        or len(set(signatures)) != len(signatures)
    ):
        raise RuntimeError("v72 canonical-to-runtime assignment coverage changed")
    return assignments


def _stream_snapshot_identity_v72(path):
    """Hash one safetensors tensor at a time without building a second bank."""
    records = []
    elements = 0
    max_tensor_bytes = 0
    with safe_open(Path(path), framework="pt", device="cpu") as handle:
        keys = sorted(handle.keys())
        if len(keys) != state_v41a.EXPECTED_TENSOR_COUNT_V41A:
            raise RuntimeError("v72 snapshot tensor count changed")
        for key in keys:
            tensor = handle.get_tensor(key)
            if (
                not key.startswith("base_model.model.model.layers.")
                or not key.endswith((".lora_A.weight", ".lora_B.weight"))
                or tensor.dtype != torch.float32
                or tensor.device.type != "cpu"
                or tensor.ndim != 2
                or not tensor.is_contiguous()
            ):
                raise RuntimeError(f"v72 invalid snapshot tensor: {key}")
            tensor_elements = int(tensor.numel())
            tensor_bytes = tensor_elements * int(tensor.element_size())
            elements += tensor_elements
            max_tensor_bytes = max(max_tensor_bytes, tensor_bytes)
            records.append({
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "elements": tensor_elements,
                "sha256": audit_v71.tensor_sha256_v71(tensor),
            })
    if elements != state_v41a.EXPECTED_MASTER_ELEMENTS_V41A:
        raise RuntimeError("v72 snapshot element count changed")
    if max_tensor_bytes != contract_v72.MAX_MASTER_TENSOR_BYTES_V72:
        raise RuntimeError("v72 snapshot maximum tensor size changed")
    identity = {
        "schema": "canonical-peft-fp32-state-v41a",
        "sha256": state_v41a.canonical_sha256_v3(records),
        "tensor_count": len(records),
        "elements": elements,
        "bytes": elements * 4,
        "ordered_key_sha256": state_v41a.canonical_sha256_v3([
            item["key"] for item in records
        ]),
        "tensors": records,
    }
    return identity, {
        "schema": "eggroll-es-streamed-snapshot-readback-v72",
        "tensor_count": len(records),
        "elements": elements,
        "bytes": elements * 4,
        "max_resident_readback_tensor_bytes": max_tensor_bytes,
        "full_state_clone_bytes": 0,
        "one_tensor_at_a_time": True,
    }


def _fsync_file_v72(path):
    descriptor = os.open(Path(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_v72(path):
    descriptor = os.open(Path(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LoRAAdapterStateWorkerExtensionV72(LoRAAdapterStateWorkerExtensionV71):
    """Own one host master and move generations instead of cloning banks."""

    def _initialize_v71_audits_v72(self, identity, base_identity):
        views, parents = self._current_runtime_maps_v71()
        base_tensors = self._current_base_map_v71()
        base_hashes = {
            item["runtime_module"]: item["sha256"]
            for item in base_identity["tensors"]
        }
        if set(base_hashes) != set(base_tensors):
            raise RuntimeError("v72 base bootstrap hash coverage changed")
        self._v71_master_cache = audit_v71.OwnedMasterIdentityCacheV71(
            self._v41_master, identity=identity
        )
        self._v71_runtime_views = views
        self._v71_runtime_parents = parents
        # These slot bytes are overwritten and exactly verified before the
        # worker can return from install.  Bootstrap only their metadata here.
        self._v71_runtime_registry = audit_v71.TensorInvariantRegistryV71(
            "runtime_lora_views",
            views,
            precomputed_sha256={key: "0" * 64 for key in views},
        )
        self._v71_base_tensors = base_tensors
        self._v71_base_registry = audit_v71.TensorInvariantRegistryV71(
            "immutable_base_weights",
            base_tensors,
            precomputed_sha256=base_hashes,
        )
        self._v71_traffic = {
            "h2d_bytes": 0,
            "lora_d2h_bytes": 0,
            "base_d2h_bytes": 0,
            "master_validation_host_copy_bytes": 0,
            "cheap_transition_checks": 0,
            "lora_d2h_calls": 0,
            "exact_base_audits": 0,
            "master_cache_hits": 0,
        }
        self._v71_candidate_audits = []
        self._v71_transition_audits = []
        self._v71_population_acceptance_sha256 = None
        self._v71_update_acceptance_sha256 = None
        self._v71_completed_boundaries = []
        self._v71_ready = True

    def install_adapter_state_v41a(
        self,
        adapter_weights_path,
        adapter_config_path,
        expected_weights_sha256,
        expected_config_sha256,
    ):
        if getattr(self, "_v41_installed", False):
            raise RuntimeError("v72 adapter state installation is one-shot")
        weights_path = Path(adapter_weights_path).resolve()
        config_path = Path(adapter_config_path).resolve()
        if (
            state_v41a.file_sha256_v41a(weights_path)
            != str(expected_weights_sha256)
            or state_v41a.file_sha256_v41a(config_path)
            != str(expected_config_sha256)
        ):
            raise RuntimeError("v72 source adapter identity changed")
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
        if (
            config.get("r") != 32
            or config.get("lora_alpha") != 64
            or config.get("bias") != "none"
        ):
            raise RuntimeError("v72 source adapter configuration changed")
        manager = self._manager_v41a()
        master = _load_owned_master_v72(weights_path)
        if state_v41a.file_sha256_v41a(weights_path) != str(
            expected_weights_sha256
        ):
            raise RuntimeError("v72 source adapter changed during owned load")
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
                "v72 simultaneous all-zero A/B initialization is ES-degenerate"
            )
        identity = adapter_identity_no_clone_v71(master)
        assignments = _runtime_assignments_no_clone_v72(manager, master)
        runtime_names = {item["runtime_module"] for item in assignments}
        base = state_v41a.topology._base_identity(
            manager.modules, runtime_names
        )
        if (
            len(runtime_names) != state_v41a.EXPECTED_RUNTIME_MODULES_V41A
            or base["tensor_count"]
            != state_v41a.EXPECTED_RUNTIME_MODULES_V41A
            or base["elements"] != state_v41a.EXPECTED_BASE_ELEMENTS_V41A
            or base["bytes"] != state_v41a.EXPECTED_BASE_BYTES_V41A
        ):
            raise RuntimeError("v72 relevant base-layer inventory changed")

        self._v41_installed = True
        self._v41_master = master
        self._v41_current_identity = identity
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
        self._v72_install_owned_copy_bytes = identity["bytes"]
        try:
            self._initialize_v71_audits_v72(identity, base)
            materialization = self._materialize_v41a(master, "v72_install")
            base_check = self._base_check_v41a("v72_install")
        except BaseException:
            for name in [
                key for key in vars(self)
                if key.startswith(("_v41", "_v71", "_v72"))
            ]:
                delattr(self, name)
            raise
        assignment_certificate = [{
            key: item[key] for key in (
                "peft_key", "side", "runtime_module", "slot",
                "slice_index", "segment_index", "segment_count",
                "source_shape", "runtime_shape",
            )
        } for item in assignments]
        base_origin = {
            "tensor_count": base["tensor_count"],
            "elements": base["elements"],
            "bytes": base["bytes"],
            "inventory_sha256": base["inventory_sha256"],
            "tensors": [{
                key: value for key, value in item.items()
                if key != "storage_data_ptr"
            } for item in base["tensors"]],
        }
        return {
            "schema": "canonical-lora-adapter-installed-v72",
            "installed": True,
            "adapter_id": 1,
            "slot": 0,
            "source_weights_sha256": str(expected_weights_sha256),
            "source_config_sha256": str(expected_config_sha256),
            "canonical_identity": identity,
            "assignment_count": len(assignments),
            "assignment_sha256": state_v41a.canonical_sha256_v3(
                assignment_certificate
            ),
            "assignments": assignment_certificate,
            "materialization": materialization,
            "base_identity": base_check,
            "base_origin_inventory": base_origin,
            "zero_zero_degeneracy_guard": {
                "all_a_zero": all_a_zero,
                "all_b_zero": all_b_zero,
                "simultaneous_all_zero_forbidden": True,
            },
            "resident_reference_tensor_bank": False,
            "reference_is_versioned_identity_only": True,
            "install_owned_copy_bytes": identity["bytes"],
            "install_full_state_clone_passes": 1,
            "v71_audit_contract": audit_v71.SCHEMA_V71,
            "owned_master_identity_cached": True,
            "v71_exact_audit_contract_retained": True,
        }

    def capture_adapter_reference_v41a(self):
        self._require_quiescent_v41a()
        verified = self._verify_master_materialized_v41a(
            "v72_reference_capture"
        )
        base = self._base_check_v41a("v72_reference_capture")
        exact = self._exact_master_audit_v71("v72_reference_capture")
        if hasattr(self, "_v41_reference"):
            raise RuntimeError("v72 forbidden resident reference bank appeared")
        self._v41_reference_identity = self._v41_current_identity
        self._v41_reference_generation += 1
        self._v41_reference_fresh = True
        return {
            "schema": "canonical-lora-reference-captured-v72",
            "reference_generation": self._v41_reference_generation,
            "reference_identity": self._v41_reference_identity,
            "reference_tensor_bank_allocated": False,
            "full_state_clone_bytes": 0,
            "master_exact_audit": exact,
            "materialization": verified["materialization"],
            "base_identity": base,
        }

    def prepare_sharded_adapter_update_v72(
        self,
        population_acceptance_sha256,
        seeds,
        coefficients,
        coefficient_sha256,
        population_size,
        expected_world_size,
        alpha,
        plan_id,
        expected_master_sha256,
        reference_generation,
    ):
        if (
            not self._v71_population_acceptance_sha256
            or str(population_acceptance_sha256)
            != self._v71_population_acceptance_sha256
        ):
            raise RuntimeError("v72 update preceded population reward acceptance")
        self._require_quiescent_v41a()
        boundary = self.exact_boundary_audit_v71("update_acceptance")
        if int(expected_world_size) != state_v41a.REQUIRED_WORLD_SIZE_V41A:
            raise RuntimeError("v72 update requires exactly four ranks")
        communicator = self._communicator_state_v3(expected_world_size)
        seeds, coefficients = state_v41a.validate_seed_coefficients_v3(
            seeds, coefficients, population_size, expected_world_size
        )
        if state_v41a.coefficient_sha256_v3(
            seeds, coefficients
        ) != str(coefficient_sha256):
            raise RuntimeError("v72 coefficient identity changed")
        alpha = float(alpha)
        if not math.isfinite(alpha) or alpha == 0.0:
            raise ValueError("v72 update alpha must be finite/nonzero")
        if (
            self._v41_current_identity["sha256"]
            != str(expected_master_sha256)
            or int(reference_generation) != self._v41_reference_generation
            or not self._v41_reference_fresh
            or self._v41_reference_identity != self._v41_current_identity
        ):
            raise RuntimeError("v72 update used a stale canonical reference")
        shard = state_v41a.seed_shard_v3(
            seeds,
            coefficients,
            communicator["rank"],
            communicator["world_size"],
        )
        manifest = {
            "schema": "canonical-lora-sharded-update-manifest-v41a",
            "seeds": seeds,
            "coefficients": coefficients,
            "coefficient_sha256": str(coefficient_sha256),
            "population_size": int(population_size),
            "world_size": communicator["world_size"],
            "alpha": alpha,
            "plan_id": str(plan_id),
            "expected_master_sha256": str(expected_master_sha256),
            "reference_generation": int(reference_generation),
            "update_sequence": self._v41_update_sequence + 1,
        }
        if not manifest["plan_id"]:
            raise ValueError("v72 update plan id is empty")
        manifest_sha = state_v41a.canonical_sha256_v3(manifest)
        rollback_lease = contract_v72.ImmutableStateLeaseV72(
            "prepared_rollback_master",
            self._v41_update_sequence,
            self._v41_master,
            self._v41_current_identity,
        )
        lease_receipt = rollback_lease.cheap_certificate("v72_prepare")
        self._v41_pending_update = {
            "phase": "prepared",
            "manifest": manifest,
            "manifest_sha256": manifest_sha,
            "shard": shard,
            "rollback_master": self._v41_master,
            "rollback_lease_v72": rollback_lease,
            "rollback_identity": self._v41_current_identity,
            "rollback_update_sequence": self._v41_update_sequence,
            "rollback_active_plan_id": self._v41_active_plan_id,
            "rollback_reference_fresh": self._v41_reference_fresh,
        }
        result = {
            "schema": "canonical-lora-sharded-update-prepared-v72",
            "prepared": True,
            "manifest_sha256": manifest_sha,
            "rank": communicator["rank"],
            "world_size": communicator["world_size"],
            "shard_indices": shard["indices"],
            "shard_seeds": shard["seeds"],
            "shard_pair_sha256": state_v41a.canonical_sha256_v3({
                "seeds": shard["seeds"],
                "coefficients": shard["coefficients"],
            }),
            "master_identity": self._v41_current_identity,
            "population_acceptance_sha256": str(
                population_acceptance_sha256
            ),
            "update_boundary_audit_sha256": boundary["audit_sha256"],
            "rollback_ownership": lease_receipt,
            "rollback_aliases_master": True,
            "rollback_clone_bytes": 0,
        }
        result["update_acceptance_sha256"] = (
            audit_v71.canonical_sha256_v71(result)
        )
        self._v71_update_acceptance_sha256 = result[
            "update_acceptance_sha256"
        ]
        return result

    def prepare_sharded_adapter_update_v71(self, *args, **kwargs):
        return self.prepare_sharded_adapter_update_v72(*args, **kwargs)

    def execute_sharded_adapter_update_v41a(self, manifest_sha256):
        pending = self._v41_pending_update
        if not isinstance(pending, dict) or pending.get("phase") != "prepared":
            raise RuntimeError("v72 adapter update was not prepared")
        if pending["manifest_sha256"] != str(manifest_sha256):
            raise RuntimeError("v72 prepared manifest identity changed")
        lease = pending.get("rollback_lease_v72")
        if not isinstance(lease, contract_v72.ImmutableStateLeaseV72):
            raise RuntimeError("v72 rollback ownership lease is absent")
        try:
            rollback_exact = lease.exact_certificate(
                "v72_update_execute_preflight"
            )
        except BaseException as error:
            self._poison_v66("v72_update_execute_master", error)
            raise
        manifest, shard = pending["manifest"], pending["shard"]
        communicator = self._communicator_state_v3(manifest["world_size"])
        device = torch.device(getattr(
            self,
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        ))
        candidate = {}
        reduced_elements = 0
        max_host_staging_bytes = 0
        try:
            for key, master in self._v41_master.items():
                accumulator = torch.zeros(
                    master.shape, dtype=torch.float32, device=device
                )
                for seed, coefficient in zip(
                    shard["seeds"], shard["coefficients"], strict=True
                ):
                    noise = state_v41a.noise_like_v41a(
                        master, key, seed, device
                    )
                    accumulator.add_(noise, alpha=float(coefficient))
                stream = (
                    torch.cuda.current_stream()
                    if accumulator.is_cuda else None
                )
                reduced = self.inter_pg.all_reduce(
                    accumulator, out_tensor=accumulator, stream=stream
                )
                if (
                    reduced is None
                    or reduced.dtype != torch.float32
                    or reduced.shape != master.shape
                ):
                    raise RuntimeError(
                        "v72 PyNccl reduction returned incompatible tensor"
                    )
                reduced.mul_(
                    float(manifest["alpha"]) / manifest["population_size"]
                )
                host_delta = reduced.detach().to(
                    device="cpu", dtype=torch.float32, non_blocking=False
                )
                max_host_staging_bytes = max(
                    max_host_staging_bytes,
                    int(host_delta.numel() * host_delta.element_size()),
                )
                candidate[key] = master.add(host_delta).contiguous()
                reduced_elements += int(reduced.numel())
                del host_delta, reduced, accumulator
            candidate = _validate_adapter_no_clone_v71(candidate)
            candidate_identity = adapter_identity_no_clone_v71(candidate)
            materialization = self._materialize_v41a(
                candidate, "v72_executed_candidate"
            )
            base = self._base_check_v41a("v72_post_update_execution")
        except BaseException:
            self._rollback_pending_v41a("v72_execute_failure")
            raise
        pending["phase"] = "executed"
        pending["candidate_master"] = candidate
        pending["candidate_identity"] = candidate_identity
        return {
            "schema": "canonical-lora-sharded-update-executed-v72",
            "executed": True,
            "manifest_sha256": str(manifest_sha256),
            "rank": communicator["rank"],
            "world_size": communicator["world_size"],
            "collective_dtype": "torch.float32",
            "tensor_count": len(candidate),
            "reduced_elements": reduced_elements,
            "candidate_identity": candidate_identity,
            "materialization": materialization,
            "base_identity": base,
            "master_committed": False,
            "rollback_lease_exact_preflight": rollback_exact,
            "candidate_owned_bytes": candidate_identity["bytes"],
            "full_state_validation_clone_bytes": 0,
            "max_host_update_staging_bytes": max_host_staging_bytes,
            "pinned_host_staging_used": False,
        }

    def commit_sharded_adapter_update_v71(
        self,
        update_acceptance_sha256,
        manifest_sha256,
        expected_final_sha256,
    ):
        if (
            not self._v71_update_acceptance_sha256
            or str(update_acceptance_sha256)
            != self._v71_update_acceptance_sha256
        ):
            raise RuntimeError("v72 commit preceded exact update acceptance")
        pending = self._v41_pending_update
        lease = (
            pending.get("rollback_lease_v72")
            if isinstance(pending, dict) else None
        )
        if not isinstance(lease, contract_v72.ImmutableStateLeaseV72):
            raise RuntimeError("v72 commit rollback ownership lease is absent")
        try:
            lease_exact = lease.exact_certificate("v72_commit_preflight")
        except BaseException as error:
            self._poison_v66("v72_commit_rollback_lease", error)
            raise
        try:
            receipt = super().commit_sharded_adapter_update_v71(
                update_acceptance_sha256,
                manifest_sha256,
                expected_final_sha256,
            )
        except BaseException as primary_error:
            pending = getattr(self, "_v41_pending_update", None)
            if (
                isinstance(pending, dict)
                and getattr(self, "_v66_terminal_poison", None) is None
            ):
                try:
                    self.abort_sharded_adapter_update_v71(
                        pending["manifest_sha256"],
                        "v72_commit_preflight_failure",
                    )
                except BaseException as restore_error:
                    raise RuntimeError(
                        "v72 commit failure rollback was unverified; actor poisoned"
                    ) from restore_error
            raise primary_error
        return {
            **receipt,
            "schema": "canonical-lora-sharded-update-committed-v72",
            "rollback_lease_exact_preflight": lease_exact,
            "host_state_residency": self.host_state_residency_v72(),
            "full_state_clone_bytes": 0,
        }

    def commit_sharded_adapter_update_v72(self, *args, **kwargs):
        return self.commit_sharded_adapter_update_v71(*args, **kwargs)

    def abort_sharded_adapter_update_v71(self, *args, **kwargs):
        receipt = super().abort_sharded_adapter_update_v71(*args, **kwargs)
        return {
            **receipt,
            "schema": "canonical-lora-update-rollback-v72",
            "host_state_residency": self.host_state_residency_v72(),
            "full_state_clone_bytes": 0,
        }

    def finalize_sharded_adapter_update_v71(self, *args, **kwargs):
        receipt = super().finalize_sharded_adapter_update_v71(*args, **kwargs)
        return {
            **receipt,
            "schema": "canonical-lora-sharded-update-finalized-v72",
            "host_state_residency": self.host_state_residency_v72(),
            "full_state_clone_bytes": 0,
        }

    def finalize_sharded_adapter_update_v72(self, *args, **kwargs):
        return self.finalize_sharded_adapter_update_v71(*args, **kwargs)

    def host_state_residency_v72(self):
        self._require_installed_v41a()
        if hasattr(self, "_v41_reference"):
            raise RuntimeError("v72 resident reference tensor bank appeared")
        roles = {"canonical_master": self._v41_master}
        pending = getattr(self, "_v41_pending_update", None)
        committed = getattr(self, "_v41_committed_rollback", None)
        if isinstance(pending, dict):
            roles["pending_rollback"] = pending["rollback_master"]
            if isinstance(pending.get("candidate_master"), dict):
                roles["executed_candidate"] = pending["candidate_master"]
        if isinstance(committed, dict):
            roles["committed_rollback"] = committed["rollback_master"]
        unique = {}
        role_to_bank = {}
        for role, tensors in roles.items():
            _validate_adapter_no_clone_v71(tensors)
            token = id(tensors)
            if token not in unique:
                unique[token] = sum(
                    int(tensor.numel() * tensor.element_size())
                    for tensor in tensors.values()
                )
            role_to_bank[role] = list(unique).index(token)
        if isinstance(pending, dict) and (
            pending["rollback_master"] is not self._v41_master
        ):
            raise RuntimeError("v72 prepared rollback stopped aliasing master")
        if isinstance(committed, dict) and (
            committed["rollback_master"] is self._v41_master
        ):
            raise RuntimeError("v72 committed rollback aliases new master")
        if isinstance(committed, dict):
            phase = "committed_rollback_retained"
        elif isinstance(pending, dict) and "executed_candidate" in roles:
            phase = "executed_candidate_retained"
        elif isinstance(pending, dict):
            phase = "prepared_rollback_alias"
        else:
            phase = "quiescent_one_master"
        result = {
            "schema": "eggroll-es-host-state-residency-v72",
            "phase": phase,
            "roles": sorted(roles),
            "role_to_unique_bank": role_to_bank,
            "unique_owned_bank_count": len(unique),
            "unique_owned_tensor_bytes": sum(unique.values()),
            "reference_tensor_bank_present": False,
            "full_state_clone_bytes_for_observation": 0,
        }
        result["receipt_sha256"] = contract_v72.canonical_sha256_v72(result)
        return result

    def save_adapter_snapshot_v41a(
        self, output_directory, expected_master_sha256
    ):
        """Publish one verified directory atomically from the immutable master."""
        self._require_quiescent_v41a()
        if (
            self._v41_current_identity["sha256"]
            != str(expected_master_sha256)
        ):
            raise RuntimeError("v72 snapshot master identity changed")
        boundary = self.exact_boundary_audit_v71(
            audit_v71.CHECKPOINT_BOUNDARY_V71
        )
        rank = int(self.inter_pg.rank)
        output = Path(output_directory).resolve()
        result = {
            "schema": "canonical-peft-fp32-snapshot-v72",
            "rank": rank,
            "written": False,
            "directory": str(output),
            "master_identity": self._v41_current_identity,
            "checkpoint_boundary_audit_sha256": boundary["audit_sha256"],
            "checkpoint_exact_content_identity": True,
            "full_state_clone_bytes": 0,
        }
        if rank != 0:
            return result
        if output.exists():
            raise FileExistsError(output)
        if not output.parent.is_dir():
            raise FileNotFoundError(output.parent)
        temporary = output.parent / (
            f".{output.name}.tmp-v72-{os.getpid()}-{time.monotonic_ns()}"
        )
        if temporary.exists():
            raise FileExistsError(temporary)
        temporary.mkdir(mode=0o700)
        weights = temporary / "adapter_model.safetensors"
        config = temporary / "adapter_config.json"
        published = False
        try:
            state_v41a.save_file(
                self._v41_master,
                weights,
                metadata={
                    "format": "pt",
                    "schema": "canonical-peft-fp32-v72",
                    "master_sha256": self._v41_current_identity["sha256"],
                },
            )
            config.write_bytes(self._v41_config_bytes)
            _fsync_file_v72(weights)
            _fsync_file_v72(config)
            _fsync_directory_v72(temporary)
            readback_identity, streaming = _stream_snapshot_identity_v72(
                weights
            )
            weights_sha = state_v41a.file_sha256_v41a(weights)
            config_sha = state_v41a.file_sha256_v41a(config)
            if (
                readback_identity != self._v41_current_identity
                or config.read_bytes() != self._v41_config_bytes
                or config_sha != self._v41_source_config_sha256
            ):
                raise RuntimeError("v72 PEFT snapshot readback changed")
            post_master = self._exact_master_audit_v71(
                "checkpoint_pre_publish"
            )
            os.rename(temporary, output)
            published = True
        except BaseException:
            if not published:
                weights.unlink(missing_ok=True)
                config.unlink(missing_ok=True)
                try:
                    temporary.rmdir()
                except OSError:
                    pass
            raise
        result.update({
            "written": True,
            "weights_path": str(output / weights.name),
            "config_path": str(output / config.name),
            "weights_sha256": weights_sha,
            "config_sha256": config_sha,
            "readback_verified": True,
            "readback_identity": readback_identity,
            "streaming_readback": streaming,
            "post_write_master_exact_audit": post_master,
            "original_canonical_key_namespace": True,
            "unscaled_fp32_master_persisted": True,
            "materialization_at_snapshot": self._v41_active_materialization,
            "atomic_directory_publication": True,
            "temporary_directory_visible_after_success": False,
            "host_state_residency": self.host_state_residency_v72(),
        })
        return result
