#!/usr/bin/env python3
"""Lower-traffic exact-audit worker for mirrored LoRA ES V71.

This additive worker keeps V66/V66d immutable.  It fuses live LoRA equality
and SHA into one flattened D2H readback, uses pointer/metadata/version guards
on every transition, caches the owned CPU FP32 master identity, and exposes
exact population/update/commit/final trust-boundary audits.
"""

from __future__ import annotations

import math
import time

import torch

import eggroll_es_audit_contract_v71 as contract
import eggroll_es_worker_lora_v41a as state_v41a
import eggroll_es_worker_lora_v66 as state_v66
from eggroll_es_worker_lora_v66d import LoRAAdapterStateWorkerExtensionV66D


def _runtime_key_v71(item):
    return "\0".join((
        item["runtime_module"], item["side"], str(item["slice_index"]),
    ))


def _validate_adapter_no_clone_v71(tensors):
    if (
        not isinstance(tensors, dict)
        or len(tensors) != state_v41a.EXPECTED_TENSOR_COUNT_V41A
    ):
        raise RuntimeError("v71 canonical adapter tensor count changed")
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
            or not tensor.is_contiguous()
        ):
            raise RuntimeError(f"v71 invalid canonical tensor: {key}")
        elements += int(tensor.numel())
    if elements != state_v41a.EXPECTED_MASTER_ELEMENTS_V41A:
        raise RuntimeError("v71 canonical adapter element count changed")
    return tensors


def adapter_identity_no_clone_v71(tensors):
    tensors = _validate_adapter_no_clone_v71(tensors)
    records = [{
        "key": key,
        "shape": list(tensors[key].shape),
        "dtype": str(tensors[key].dtype),
        "elements": int(tensors[key].numel()),
        "sha256": contract.tensor_sha256_v71(tensors[key]),
    } for key in sorted(tensors)]
    return {
        "schema": "canonical-peft-fp32-state-v41a",
        "sha256": state_v41a.canonical_sha256_v3(records),
        "tensor_count": len(records),
        "elements": sum(item["elements"] for item in records),
        "bytes": sum(item["elements"] * 4 for item in records),
        "ordered_key_sha256": state_v41a.canonical_sha256_v3([
            item["key"] for item in records
        ]),
        "tensors": records,
    }


def fused_lora_readback_v71(
    actual_by_key,
    expected_by_key=None,
    *,
    readback_fn=None,
):
    """Read all live views to host once, then perform equality and SHA there."""
    if not isinstance(actual_by_key, dict) or not actual_by_key:
        raise RuntimeError("v71 live LoRA view map is empty")
    keys = tuple(sorted(actual_by_key))
    actual = [actual_by_key[key] for key in keys]
    if any(
        not isinstance(tensor, torch.Tensor)
        or tensor.dtype != torch.bfloat16
        for tensor in actual
    ):
        raise RuntimeError("v71 live LoRA view dtype changed")
    devices = {str(tensor.device) for tensor in actual}
    if len(devices) != 1:
        raise RuntimeError("v71 live LoRA views span devices")
    flat_device = torch.cat([
        tensor.detach().reshape(-1) for tensor in actual
    ]).contiguous()
    if readback_fn is None:
        readback_fn = lambda tensor: tensor.detach().cpu().contiguous()
    flat_cpu = readback_fn(flat_device)
    if (
        not isinstance(flat_cpu, torch.Tensor)
        or flat_cpu.device.type != "cpu"
        or flat_cpu.dtype != torch.bfloat16
        or flat_cpu.ndim != 1
        or flat_cpu.numel() != flat_device.numel()
        or not flat_cpu.is_contiguous()
    ):
        raise RuntimeError("v71 fused LoRA readback changed")
    expected_equal = None
    if expected_by_key is not None:
        if set(expected_by_key) != set(keys):
            raise RuntimeError("v71 expected LoRA readback coverage changed")
    hashes = {}
    offset = 0
    for key, tensor in zip(keys, actual, strict=True):
        elements = int(tensor.numel())
        segment = flat_cpu[offset:offset + elements]
        if expected_by_key is not None:
            expected = expected_by_key[key]
            if (
                not isinstance(expected, torch.Tensor)
                or expected.device.type != "cpu"
                or expected.dtype != torch.bfloat16
                or expected.numel() != elements
                or not torch.equal(segment, expected.detach().reshape(-1))
            ):
                raise RuntimeError(
                    f"v71 fused live LoRA equality check failed: {key}"
                )
        hashes[key] = contract.tensor_sha256_v71(segment)
        offset += elements
    if offset != flat_cpu.numel():
        raise RuntimeError("v71 fused readback partition changed")
    if expected_by_key is not None:
        expected_equal = True
    return {
        "schema": "eggroll-es-single-d2h-lora-readback-v71",
        "single_d2h_readback": True,
        "d2h_calls": 1,
        "d2h_bytes": int(flat_cpu.numel() * flat_cpu.element_size()),
        "view_count": len(keys),
        "elements": int(flat_cpu.numel()),
        "dtype": str(flat_cpu.dtype),
        "device_before_readback": next(iter(devices)),
        "equality_checked": expected_by_key is not None,
        "exact_equal": expected_equal,
        "expected_value_concat_host_copy_bytes": 0,
        "sha256_by_key": hashes,
    }


def _owned_candidate_v71(master, seed, sigma, sign, device):
    sigma = float(sigma)
    sign = int(sign)
    if not math.isfinite(sigma) or sigma <= 0.0 or sign not in (-1, 1):
        raise ValueError("v71 perturbation requires sigma>0 and sign +/-1")
    result = {}
    for key, tensor in master.items():
        direction = state_v66.signed_noise_like_v66(
            tensor, key, int(seed), sign, device
        )
        result[key] = tensor.to(
            device=device, dtype=torch.float32
        ).add_(direction, alpha=sigma)
    return result


class LoRAAdapterStateWorkerExtensionV71(LoRAAdapterStateWorkerExtensionV66D):
    """V66d state safety with lower-traffic V71 verification semantics."""

    def _traffic_add_v71(self, key, value):
        if hasattr(self, "_v71_traffic"):
            self._v71_traffic[key] = self._v71_traffic.get(key, 0) + int(value)

    def _rebind_master_cache_v71(
        self,
        phase,
        expected_sha256=None,
        *,
        precomputed_identity=None,
    ):
        """Adopt a deliberate master replacement after one exact no-clone read."""
        identity = (
            adapter_identity_no_clone_v71(self._v41_master)
            if precomputed_identity is None
            else precomputed_identity
        )
        expected = (
            self._v41_current_identity["sha256"]
            if expected_sha256 is None else str(expected_sha256)
        )
        if identity != self._v41_current_identity or identity["sha256"] != expected:
            error = RuntimeError(
                f"v71 controlled master replacement changed at {phase}"
            )
            self._poison_v66(f"v71_master_rebind_{phase}", error)
            raise error
        self._v71_master_cache = contract.OwnedMasterIdentityCacheV71(
            self._v41_master,
            identity=identity,
        )
        return {
            "schema": "eggroll-es-controlled-master-rebind-v71",
            "phase": str(phase),
            "identity": identity,
            "exact_content_read_bytes": (
                identity["bytes"] if precomputed_identity is None else 0
            ),
            "precomputed_exact_identity_reused": (
                precomputed_identity is not None
            ),
            "validation_clone_bytes": 0,
        }

    def _current_runtime_maps_v71(self):
        manager = self._manager_v41a()
        views = {}
        parents = {}
        for item in self._v41_assignments:
            key = _runtime_key_v71(item)
            module = manager.modules[item["runtime_module"]]
            parent = (
                module.lora_a_stacked
                if item["side"] == "A"
                else module.lora_b_stacked
            )[item["slice_index"]]
            views[key] = parent[state_v41a.ADAPTER_SLOT_V41A, 0]
            parents[key] = parent
        return views, parents

    def _current_base_map_v71(self):
        manager = self._manager_v41a()
        result = {}
        seen = set()
        for name in sorted(self._v41_runtime_names):
            weight = getattr(
                getattr(manager.modules[name], "base_layer", None),
                "weight",
                None,
            )
            if not isinstance(weight, torch.Tensor):
                raise RuntimeError("v71 base tensor link changed")
            storage = (
                int(weight.untyped_storage().data_ptr()),
                int(weight.storage_offset()),
            )
            if storage in seen:
                continue
            seen.add(storage)
            result[name] = weight
        return result

    def _assert_runtime_links_v71(self):
        current_views, current_parents = self._current_runtime_maps_v71()
        if set(current_views) != set(self._v71_runtime_views):
            raise RuntimeError("v71 runtime view coverage drifted")
        for key in sorted(current_views):
            expected_view = self._v71_runtime_views[key]
            current_view = current_views[key]
            if (
                current_parents[key] is not self._v71_runtime_parents[key]
                or current_view.untyped_storage().data_ptr()
                != expected_view.untyped_storage().data_ptr()
                or current_view.storage_offset() != expected_view.storage_offset()
                or current_view.shape != expected_view.shape
                or current_view.stride() != expected_view.stride()
            ):
                raise RuntimeError(f"v71 runtime manager link drifted: {key}")

    def _assert_base_links_v71(self):
        current = self._current_base_map_v71()
        if (
            set(current) != set(self._v71_base_tensors)
            or any(
                current[key] is not self._v71_base_tensors[key]
                for key in current
            )
        ):
            raise RuntimeError("v71 base manager object link drifted")

    def install_adapter_state_v41a(self, *args, **kwargs):
        receipt = super().install_adapter_state_v41a(*args, **kwargs)
        master_cache = contract.OwnedMasterIdentityCacheV71(
            self._v41_master,
            identity=self._v41_current_identity,
        )
        if master_cache.sha256 != self._v41_current_identity["sha256"]:
            raise RuntimeError("v71 cached master identity differs from V41 identity")
        views, parents = self._current_runtime_maps_v71()
        base = self._current_base_map_v71()
        self._v71_master_cache = master_cache
        self._v71_runtime_views = views
        self._v71_runtime_parents = parents
        self._v71_runtime_registry = contract.TensorInvariantRegistryV71(
            "runtime_lora_views", views
        )
        self._v71_base_tensors = base
        self._v71_base_registry = contract.TensorInvariantRegistryV71(
            "immutable_base_weights", base
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
        return {
            **receipt,
            "schema": "canonical-lora-adapter-installed-v71",
            "v71_audit_contract": contract.SCHEMA_V71,
            "owned_master_identity_cached": True,
            "master_validation_clone_bytes_after_install": 0,
            "base_cheap_invariant": self._base_check_v41a("v71_install_bind"),
        }

    def _master_identity_v71(self, phase):
        certificate = self._v71_master_cache.cached_identity(
            self._v41_master, phase
        )
        if certificate["sha256"] != self._v41_current_identity["sha256"]:
            raise RuntimeError("v71 cached master differs from current identity")
        self._traffic_add_v71("master_cache_hits", 1)
        return {
            "identity": self._v41_current_identity,
            "cache": certificate,
        }

    def _base_check_v41a(self, phase):
        if not getattr(self, "_v71_ready", False):
            return super()._base_check_v41a(phase)
        self._assert_base_links_v71()
        cheap = self._v71_base_registry.cheap_certificate(
            self._v71_base_tensors, str(phase)
        )
        self._traffic_add_v71("cheap_transition_checks", 1)
        origin = self._v41_base_identity
        return {
            "phase": str(phase),
            "unchanged": True,
            "tensor_count": origin["tensor_count"],
            "elements": origin["elements"],
            "bytes": origin["bytes"],
            "inventory_sha256": origin["inventory_sha256"],
            "exact_content_audit_performed": False,
            "d2h_bytes": 0,
            "cheap_invariant": cheap,
        }

    def _materialize_v41a(self, tensors, phase):
        if not getattr(self, "_v71_ready", False):
            return super()._materialize_v41a(tensors, phase)
        tensors = _validate_adapter_no_clone_v71(tensors)
        if tensors is self._v41_master:
            self._master_identity_v71(f"materialize_{phase}")
        self._assert_runtime_links_v71()
        self._v71_runtime_registry.cheap_certificate(
            self._v71_runtime_views, f"before_{phase}"
        )
        manager = self._manager_v41a()
        runtime_names = {item["runtime_module"] for item in self._v41_assignments}
        expected = {}
        storage = []
        with torch.no_grad():
            for name in sorted(runtime_names):
                manager.modules[name].reset_lora(state_v41a.ADAPTER_SLOT_V41A)
            for item in self._v41_assignments:
                key = _runtime_key_v71(item)
                source = tensors[item["peft_key"]]
                logical, side = state_v41a.topology._source_parts(
                    item["peft_key"]
                )
                _target, slices = state_v41a.topology._runtime_target(logical)
                module = manager.modules[item["runtime_module"]]
                expected_value = state_v41a._expected_runtime_value_v41a(
                    source,
                    side,
                    slices,
                    item["slice_index"],
                    module.output_slices,
                    self._v41_scale,
                ).contiguous()
                view = self._v71_runtime_views[key]
                parent = self._v71_runtime_parents[key]
                if (
                    view.shape != expected_value.shape
                    or view.dtype != torch.bfloat16
                ):
                    raise RuntimeError(
                        f"v71 runtime view metadata changed: {item['peft_key']}"
                    )
                view.copy_(
                    expected_value.to(device=view.device),
                    non_blocking=False,
                )
                expected[key] = expected_value.to(device="cpu")
                storage.append({
                    "signature": [
                        item["runtime_module"], side, item["slice_index"],
                    ],
                    "storage_data_ptr": int(
                        parent.untyped_storage().data_ptr()
                    ),
                    "view_storage_data_ptr": int(
                        view.untyped_storage().data_ptr()
                    ),
                    "view_storage_offset": int(view.storage_offset()),
                    "view_aliases_parent": (
                        parent.untyped_storage().data_ptr()
                        == view.untyped_storage().data_ptr()
                    ),
                })
        readback = fused_lora_readback_v71(
            self._v71_runtime_views, expected
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        records = []
        for item in self._v41_assignments:
            key = _runtime_key_v71(item)
            view = self._v71_runtime_views[key]
            records.append({
                **item,
                "dtype": str(view.dtype),
                "elements": int(view.numel()),
                "sha256": readback["sha256_by_key"][key],
            })
        elements = sum(item["elements"] for item in records)
        pointers = [item["storage_data_ptr"] for item in storage]
        if (
            len(records) != state_v41a.EXPECTED_RUNTIME_VIEWS_V41A
            or elements != state_v41a.EXPECTED_RUNTIME_ELEMENTS_V41A
            or len(set(pointers)) != state_v41a.EXPECTED_RUNTIME_VIEWS_V41A
            or not all(item["view_aliases_parent"] for item in storage)
            or readback["d2h_bytes"]
            != state_v41a.EXPECTED_RUNTIME_ELEMENTS_V41A * 2
        ):
            raise RuntimeError("v71 runtime residency/readback coverage changed")
        self._v71_runtime_registry.rebind_controlled_write(
            self._v71_runtime_views,
            str(phase),
            expected_sha256=readback["sha256_by_key"],
        )
        compact = [{
            key: value for key, value in item.items()
            if key not in {"storage_data_ptr", "view_storage_data_ptr"}
        } for item in storage]
        certificate = {
            "schema": "canonical-to-vllm-lora-materialization-v71",
            "phase": str(phase),
            "adapter_id": 1,
            "slot": 0,
            "source_tensor_count": len(tensors),
            "source_elements": sum(tensor.numel() for tensor in tensors.values()),
            "runtime_module_count": len(runtime_names),
            "runtime_view_count": len(records),
            "runtime_elements": elements,
            "runtime_dtype": "torch.bfloat16",
            "b_scale": self._v41_scale,
            "a_duplication_and_b_splitting_verified": True,
            "unique_parent_storage_count": len(set(pointers)),
            "runtime_views_share_no_parent_storage": (
                len(set(pointers)) == len(pointers)
            ),
            "slot_views_alias_parent_buffers": all(
                item["view_aliases_parent"] for item in storage
            ),
            "storage_layout_sha256": state_v41a.canonical_sha256_v3(compact),
            "runtime_values_sha256": state_v41a.canonical_sha256_v3(records),
            "single_d2h_readback": readback,
            "validation_clone_bytes": 0,
        }
        self._v41_active_materialization = certificate
        self._traffic_add_v71("h2d_bytes", readback["d2h_bytes"])
        self._traffic_add_v71("lora_d2h_bytes", readback["d2h_bytes"])
        self._traffic_add_v71("lora_d2h_calls", 1)
        return certificate

    def _exact_base_audit_v71(self, boundary):
        try:
            self._assert_base_links_v71()
            exact = self._v71_base_registry.exact_certificate(
                self._v71_base_tensors, boundary
            )
        except BaseException as error:
            self._poison_v66(f"v71_base_{boundary}", error)
            raise
        self._traffic_add_v71("base_d2h_bytes", exact["d2h_bytes"])
        self._traffic_add_v71("exact_base_audits", 1)
        return exact

    def _exact_master_audit_v71(self, boundary):
        try:
            exact = self._v71_master_cache.exact_audit(
                self._v41_master, boundary
            )
            if exact["sha256"] != self._v41_current_identity["sha256"]:
                raise RuntimeError("v71 exact master identity changed")
            return exact
        except BaseException as error:
            self._poison_v66(f"v71_master_{boundary}", error)
            raise

    def _exact_lora_audit_v71(self, boundary):
        try:
            self._assert_runtime_links_v71()
            self._v71_runtime_registry.cheap_certificate(
                self._v71_runtime_views, boundary
            )
            readback = fused_lora_readback_v71(self._v71_runtime_views)
            records = []
            for item in self._v41_assignments:
                key = _runtime_key_v71(item)
                records.append({
                    **item,
                    "dtype": str(self._v71_runtime_views[key].dtype),
                    "elements": int(self._v71_runtime_views[key].numel()),
                    "sha256": readback["sha256_by_key"][key],
                })
            runtime_sha = state_v41a.canonical_sha256_v3(records)
            if runtime_sha != self._v41_active_materialization[
                "runtime_values_sha256"
            ]:
                raise RuntimeError("v71 exact live LoRA content drifted")
        except BaseException as error:
            self._fail_closed_runtime_v71(boundary, error)
        self._traffic_add_v71("lora_d2h_bytes", readback["d2h_bytes"])
        self._traffic_add_v71("lora_d2h_calls", 1)
        return {
            **readback,
            "boundary": str(boundary),
            "runtime_values_sha256": runtime_sha,
            "matches_active_materialization": True,
        }

    def _fail_closed_runtime_v71(self, phase, primary_error):
        try:
            restored = self._restore_exact_master_v66(
                f"v71_{phase}_audit_repair"
            )
            if (
                restored["master_identity"]["sha256"]
                != self._v41_current_identity["sha256"]
            ):
                raise RuntimeError("v71 repair restored another master")
        except BaseException as restore_error:
            if getattr(self, "_v66_terminal_poison", None) is None:
                self._poison_v66(f"v71_{phase}_audit_repair", restore_error)
            raise RuntimeError(
                "v71 runtime audit failed and exact restore was unverified; poisoned"
            ) from restore_error
        raise RuntimeError(
            "v71 runtime audit failed; exact master restored and reward rejected"
        ) from primary_error

    def _restore_exact_master_v66(self, phase):
        try:
            master = self._master_identity_v71(f"restore_{phase}")
            materialization = self._materialize_v41a(
                self._v41_master, f"v71_{phase}_exact_master"
            )
            base = self._base_check_v41a(f"v71_{phase}_exact_master")
        except BaseException as error:
            self._poison_v66(f"v71_{phase}", error)
            raise RuntimeError(
                "v71 could not prove exact master restoration; actor poisoned"
            ) from error
        self._v41_active_perturbation = None
        self._v66_candidate_transaction = None
        return {
            "master_identity": master["identity"],
            "materialization": materialization,
            "base_identity": base,
            "cached_master_identity": master["cache"],
        }

    def _verify_master_materialized_v41a(self, phase):
        """Verify runtime from the cached master without a validation clone."""
        master = self._master_identity_v71(f"verify_{phase}")
        materialization = self._materialize_v41a(
            self._v41_master, f"v71_{phase}"
        )
        return {
            "master_identity": master["identity"],
            "cached_master_identity": master["cache"],
            "materialization": materialization,
            "validation_clone_bytes": 0,
        }

    def post_generation_lora_audit_v71(self, candidate_id):
        candidate_id = str(candidate_id)
        if not candidate_id:
            raise ValueError("v71 candidate id is empty")
        if (
            getattr(self, "_v41_active_perturbation", None) is None
            and getattr(self, "_v66_candidate_transaction", None) is None
        ):
            raise RuntimeError("v71 post-generation audit has no active candidate")
        cheap_base = self._base_check_v41a(
            f"candidate_{candidate_id}_post_generation"
        )
        cheap_master = self._master_identity_v71(
            f"candidate_{candidate_id}_post_generation"
        )
        lora = self._exact_lora_audit_v71(
            f"candidate_{candidate_id}_post_generation"
        )
        receipt = {
            "schema": "eggroll-es-provisional-candidate-audit-v71",
            "candidate_id": candidate_id,
            "cheap_base_invariant": cheap_base,
            "cheap_master_invariant": cheap_master["cache"],
            "exact_lora": lora,
            "reward_status": "provisional_not_accepted",
        }
        receipt["audit_sha256"] = contract.canonical_sha256_v71(receipt)
        self._v71_candidate_audits.append(receipt["audit_sha256"])
        return receipt

    def _cheap_transition_audit_v71(self, phase):
        self._assert_runtime_links_v71()
        runtime = self._v71_runtime_registry.cheap_certificate(
            self._v71_runtime_views, str(phase)
        )
        base = self._base_check_v41a(str(phase))
        master = self._master_identity_v71(str(phase))
        receipt = {
            "schema": "eggroll-es-cheap-transition-audit-v71",
            "phase": str(phase),
            "runtime": runtime,
            "base": base,
            "master": master["cache"],
            "d2h_bytes": 0,
        }
        receipt["audit_sha256"] = contract.canonical_sha256_v71(receipt)
        return receipt

    def begin_actor_gpu_work_v66d(self, assignment):
        """Place the cheap pre-generation invariant before the CUDA event."""
        work_identity = contract.canonical_sha256_v71(assignment)
        transition = self._cheap_transition_audit_v71(
            f"pre_generation_{work_identity}"
        )
        receipt = super().begin_actor_gpu_work_v66d(assignment)
        self._v71_transition_audits.append(transition["audit_sha256"])
        return receipt

    def end_actor_gpu_work_v66d(self, assignment, output_cardinality):
        """Withhold the unchanged actor receipt until exact LoRA post-audit."""
        receipt = super().end_actor_gpu_work_v66d(
            assignment, output_cardinality
        )
        self.post_generation_lora_audit_v71(receipt["work_id"])
        return receipt

    def candidate_audit_matrix_v71(self):
        if (
            getattr(self, "_v41_active_perturbation", None) is not None
            or getattr(self, "_v66_candidate_transaction", None) is not None
            or getattr(self, "_v66d_active_gpu_work", None) is not None
        ):
            raise RuntimeError("v71 candidate audit matrix is not quiescent")
        return {
            "schema": "eggroll-es-candidate-audit-matrix-v71",
            "candidate_audit_sha256": list(self._v71_candidate_audits),
            "transition_audit_sha256": list(self._v71_transition_audits),
            "candidate_count": len(self._v71_candidate_audits),
            "all_rewards_provisional": True,
        }

    def exact_boundary_audit_v71(self, boundary):
        boundary = str(boundary)
        if boundary not in (
            *contract.EXACT_BOUNDARIES_V71,
            contract.CHECKPOINT_BOUNDARY_V71,
        ):
            raise ValueError("v71 exact audit boundary changed")
        base = self._exact_base_audit_v71(boundary)
        master = self._exact_master_audit_v71(boundary)
        lora = self._exact_lora_audit_v71(boundary)
        receipt = {
            "schema": "eggroll-es-exact-trust-boundary-v71",
            "boundary": boundary,
            "base": base,
            "master": master,
            "lora": lora,
            "passed": True,
            "reward_or_update_accepted": False,
        }
        receipt["audit_sha256"] = contract.canonical_sha256_v71(receipt)
        self._v71_completed_boundaries.append(boundary)
        return receipt

    def accept_population_rewards_v71(self, candidate_audit_sha256):
        observed = list(candidate_audit_sha256)
        if (
            not observed
            or observed != self._v71_candidate_audits
            or len(set(observed)) != len(observed)
        ):
            raise RuntimeError("v71 candidate audit matrix changed")
        boundary = self.exact_boundary_audit_v71(
            "population_reward_acceptance"
        )
        value = {
            "schema": "eggroll-es-population-reward-acceptance-v71",
            "candidate_audit_sha256": observed,
            "candidate_count": len(observed),
            "boundary_audit_sha256": boundary["audit_sha256"],
            "rewards_accepted": True,
            "update_authorized": False,
        }
        value["acceptance_sha256"] = contract.canonical_sha256_v71(value)
        self._v71_population_acceptance_sha256 = value["acceptance_sha256"]
        return value

    def prepare_sharded_adapter_update_v71(
        self, population_acceptance_sha256, *args, **kwargs
    ):
        if (
            not self._v71_population_acceptance_sha256
            or str(population_acceptance_sha256)
            != self._v71_population_acceptance_sha256
        ):
            raise RuntimeError("v71 update preceded population reward acceptance")
        boundary = self.exact_boundary_audit_v71("update_acceptance")
        receipt = super().prepare_sharded_adapter_update_v41a(*args, **kwargs)
        result = {
            **receipt,
            "schema": "canonical-lora-sharded-update-prepared-v71",
            "population_acceptance_sha256": str(
                population_acceptance_sha256
            ),
            "update_boundary_audit_sha256": boundary["audit_sha256"],
        }
        result["update_acceptance_sha256"] = contract.canonical_sha256_v71(
            result
        )
        self._v71_update_acceptance_sha256 = result[
            "update_acceptance_sha256"
        ]
        return result

    def prepare_sharded_adapter_update_v41a(self, *args, **kwargs):
        raise RuntimeError(
            "v71 update prepare requires population acceptance; use "
            "prepare_sharded_adapter_update_v71"
        )

    def _restore_update_master_v71(
        self, master, identity, phase, *, precomputed_identity=None
    ):
        """Restore an owned rollback master and prove the live slot exactly."""
        try:
            self._v41_master = master
            self._v41_current_identity = identity
            master_rebind = self._rebind_master_cache_v71(
                phase,
                identity["sha256"],
                precomputed_identity=precomputed_identity,
            )
            materialization = self._materialize_v41a(
                master, f"v71_{phase}_exact_master"
            )
            base = self._exact_base_audit_v71(phase)
        except BaseException as error:
            if getattr(self, "_v66_terminal_poison", None) is None:
                self._poison_v66(f"v71_{phase}", error)
            raise RuntimeError(
                "v71 update rollback could not be proved; actor poisoned"
            ) from error
        self._v41_active_perturbation = None
        self._v66_candidate_transaction = None
        return {
            "master_rebind": master_rebind,
            "materialization": materialization,
            "base_exact_audit": base,
        }

    def commit_sharded_adapter_update_v71(
        self,
        update_acceptance_sha256,
        manifest_sha256,
        expected_final_sha256,
    ):
        """Commit provisionally, then return only after the exact boundary."""
        self._require_not_poisoned_v66()
        if (
            not self._v71_update_acceptance_sha256
            or str(update_acceptance_sha256)
            != self._v71_update_acceptance_sha256
        ):
            raise RuntimeError("v71 commit preceded exact update acceptance")
        pending = self._v41_pending_update
        if not isinstance(pending, dict) or pending.get("phase") != "executed":
            raise RuntimeError("v71 adapter update was not executed")
        if pending.get("manifest_sha256") != str(manifest_sha256):
            self.abort_sharded_adapter_update_v71(
                pending.get("manifest_sha256"), "commit_manifest_mismatch"
            )
            raise RuntimeError("v71 commit manifest identity changed")
        candidate = pending.get("candidate_master")
        candidate_identity = adapter_identity_no_clone_v71(candidate)
        if (
            candidate_identity != pending.get("candidate_identity")
            or candidate_identity["sha256"] != str(expected_final_sha256)
        ):
            self.abort_sharded_adapter_update_v71(
                str(manifest_sha256), "commit_identity_mismatch"
            )
            raise RuntimeError("v71 cross-rank final identity changed")

        old_master = self._v41_master
        old_identity = self._v41_current_identity
        old_cache = self._v71_master_cache
        try:
            # Ownership moves from the private pending transaction to the
            # canonical master.  It is not cloned merely to validate it.
            self._v41_master = candidate
            self._v41_current_identity = candidate_identity
            master_rebind = self._rebind_master_cache_v71(
                "commit",
                str(expected_final_sha256),
                precomputed_identity=candidate_identity,
            )
            materialization = self._materialize_v41a(
                candidate, "v71_commit_provisional"
            )
            cheap_base = self._base_check_v41a("v71_commit_provisional")
            boundary = self.exact_boundary_audit_v71("commit")
        except BaseException as primary_error:
            self._v41_master = old_master
            self._v41_current_identity = old_identity
            self._v71_master_cache = old_cache
            self._v41_pending_update = None
            try:
                self._restore_update_master_v71(
                    old_master, old_identity, "commit_failure_rollback"
                )
            except BaseException as restore_error:
                raise RuntimeError(
                    "v71 commit failed and rollback was unverified; actor poisoned"
                ) from restore_error
            raise primary_error

        manifest = pending["manifest"]
        self._v41_update_sequence = int(manifest["update_sequence"])
        self._v41_active_plan_id = manifest["plan_id"]
        self._v41_reference_fresh = False
        self._v41_committed_rollback = {
            "manifest_sha256": str(manifest_sha256),
            "rollback_master": old_master,
            "rollback_identity": old_identity,
            "rollback_update_sequence": pending["rollback_update_sequence"],
            "rollback_active_plan_id": pending["rollback_active_plan_id"],
            "rollback_reference_fresh": pending["rollback_reference_fresh"],
            "committed_identity": candidate_identity,
        }
        self._v41_pending_update = None
        return {
            "schema": "canonical-lora-sharded-update-committed-v71",
            "committed": True,
            "manifest_sha256": str(manifest_sha256),
            "rank": int(self.inter_pg.rank),
            "final_identity": candidate_identity,
            "update_sequence": self._v41_update_sequence,
            "reference_fresh_for_population": False,
            "requires_cross_rank_finalize": True,
            "materialization": materialization,
            "cheap_base_invariant": cheap_base,
            "master_rebind": master_rebind,
            "commit_boundary_audit_sha256": boundary["audit_sha256"],
            "validation_clone_bytes": 0,
        }

    def commit_sharded_adapter_update_v41a(self, *args, **kwargs):
        raise RuntimeError(
            "v71 commit requires update acceptance; use "
            "commit_sharded_adapter_update_v71"
        )

    def abort_sharded_adapter_update_v71(
        self, manifest_sha256, reason="controller_abort"
    ):
        """Repair pending or partial commit state for an uncertain RPC wave."""
        self._require_not_poisoned_v66()
        reason = str(reason)
        if not reason:
            raise ValueError("v71 update abort reason is empty")
        pending = self._v41_pending_update
        committed = self._v41_committed_rollback
        if isinstance(pending, dict):
            if pending.get("manifest_sha256") != str(manifest_sha256):
                raise RuntimeError("v71 abort manifest differs from pending update")
            rollback_master = pending["rollback_master"]
            rollback_identity = pending["rollback_identity"]
            rollback_sequence = pending["rollback_update_sequence"]
            rollback_plan = pending["rollback_active_plan_id"]
            rollback_reference = pending["rollback_reference_fresh"]
            phase = f"pending_abort_{reason}"
        elif isinstance(committed, dict):
            if committed.get("manifest_sha256") != str(manifest_sha256):
                raise RuntimeError("v71 abort manifest differs from committed update")
            rollback_master = committed["rollback_master"]
            rollback_identity = committed["rollback_identity"]
            rollback_sequence = committed["rollback_update_sequence"]
            rollback_plan = committed["rollback_active_plan_id"]
            rollback_reference = committed["rollback_reference_fresh"]
            phase = f"partial_commit_abort_{reason}"
        else:
            raise RuntimeError("v71 abort found no pending or committed update")
        observed_rollback_identity = adapter_identity_no_clone_v71(
            rollback_master
        )
        if observed_rollback_identity != rollback_identity:
            error = RuntimeError("v71 rollback master exact identity changed")
            self._poison_v66(phase, error)
            raise error
        restored = self._restore_update_master_v71(
            rollback_master,
            rollback_identity,
            phase,
            precomputed_identity=observed_rollback_identity,
        )
        self._v41_update_sequence = rollback_sequence
        self._v41_active_plan_id = rollback_plan
        self._v41_reference_fresh = rollback_reference
        self._v41_pending_update = None
        self._v41_committed_rollback = None
        self._v71_update_acceptance_sha256 = None
        return {
            "schema": "canonical-lora-update-rollback-v71",
            "rolled_back": True,
            "reason": reason,
            "manifest_sha256": str(manifest_sha256),
            "identity": rollback_identity,
            "update_sequence": rollback_sequence,
            "reference_fresh": rollback_reference,
            "exact_restore": restored,
            "terminal_poisoned": False,
        }

    def abort_sharded_adapter_update_v41a(self, manifest_sha256):
        return self.abort_sharded_adapter_update_v71(
            manifest_sha256, "legacy_or_unknown_rpc"
        )

    def _rollback_pending_v41a(self, phase):
        pending = self._v41_pending_update
        if not isinstance(pending, dict):
            raise RuntimeError("v71 no pending update to roll back")
        return self.abort_sharded_adapter_update_v71(
            pending["manifest_sha256"], str(phase)
        )

    def finalize_sharded_adapter_update_v71(
        self, manifest_sha256, expected_final_sha256
    ):
        """Retain rollback ownership until the final exact audit has passed."""
        self._require_not_poisoned_v66()
        committed = self._v41_committed_rollback
        if (
            not isinstance(committed, dict)
            or committed.get("manifest_sha256") != str(manifest_sha256)
            or self._v41_current_identity["sha256"]
            != str(expected_final_sha256)
            or committed.get("committed_identity")
            != self._v41_current_identity
        ):
            raise RuntimeError("v71 committed update finalize identity changed")
        materialization = self._materialize_v41a(
            self._v41_master, "v71_cross_rank_finalize_provisional"
        )
        cheap_base = self._base_check_v41a(
            "v71_cross_rank_finalize_provisional"
        )
        boundary = self.exact_boundary_audit_v71("final")
        self._v41_committed_rollback = None
        self._v71_update_acceptance_sha256 = None
        return {
            "schema": "canonical-lora-sharded-update-finalized-v71",
            "finalized": True,
            "manifest_sha256": str(manifest_sha256),
            "rank": int(self.inter_pg.rank),
            "final_identity": self._v41_current_identity,
            "reference_fresh_for_population": False,
            "materialization": materialization,
            "cheap_base_invariant": cheap_base,
            "final_boundary_audit_sha256": boundary["audit_sha256"],
            "validation_clone_bytes": 0,
        }

    def finalize_sharded_adapter_update_v41a(
        self, manifest_sha256, expected_final_sha256
    ):
        return self.finalize_sharded_adapter_update_v71(
            manifest_sha256, expected_final_sha256
        )

    def save_adapter_snapshot_v41a(
        self, output_directory, expected_master_sha256
    ):
        """Bind the persisted readback to an exact pre-checkpoint boundary."""
        self._require_quiescent_v41a()
        if (
            self._v41_current_identity["sha256"]
            != str(expected_master_sha256)
        ):
            raise RuntimeError("v71 snapshot master identity changed")
        boundary = self.exact_boundary_audit_v71(
            contract.CHECKPOINT_BOUNDARY_V71
        )
        receipt = super().save_adapter_snapshot_v41a(
            output_directory, expected_master_sha256
        )
        return {
            **receipt,
            "schema": "canonical-peft-fp32-snapshot-v71",
            "checkpoint_boundary_audit_sha256": boundary["audit_sha256"],
            "checkpoint_exact_content_identity": True,
        }

    def materialize_mirrored_adapter_v71(
        self,
        direction_seed,
        sigma,
        sign,
        pair_id,
        evaluation_contract_sha256,
        expected_master_sha256,
    ):
        started_ns = time.monotonic_ns()
        self._require_not_poisoned_v66()
        self._require_quiescent_v41a()
        pair_id = state_v66._sha256_string_v66(pair_id, "pair id")
        contract_sha = state_v66._sha256_string_v66(
            evaluation_contract_sha256, "evaluation contract identity"
        )
        expected_master = state_v66._sha256_string_v66(
            expected_master_sha256, "expected master identity"
        )
        master = self._master_identity_v71("mirrored_candidate_v71")
        if master["identity"]["sha256"] != expected_master:
            raise RuntimeError("v71 mirrored candidate master identity changed")
        transaction = {
            "schema": "mirrored-es-candidate-transaction-v71",
            "phase": "before_runtime_write",
            "direction_seed": int(direction_seed),
            "sigma": float(sigma),
            "sign": int(sign),
            "pair_id": pair_id,
            "evaluation_contract_sha256": contract_sha,
            "expected_master_sha256": expected_master,
        }
        self._v66_candidate_transaction = transaction
        device = torch.device(getattr(
            self,
            "device",
            "cuda" if torch.cuda.is_available() else "cpu",
        ))
        try:
            candidate = _owned_candidate_v71(
                self._v41_master,
                int(direction_seed),
                float(sigma),
                int(sign),
                device,
            )
            candidate_cpu = {
                key: tensor.detach().cpu().contiguous()
                for key, tensor in candidate.items()
            }
            candidate_identity = adapter_identity_no_clone_v71(candidate_cpu)
            transaction["phase"] = "runtime_write_started"
            materialization = self._materialize_v41a(
                candidate_cpu, "v71_mirrored_candidate"
            )
            base = self._base_check_v41a("v71_mirrored_candidate")
            active = {
                **transaction,
                "phase": "candidate_active",
                "candidate_identity": candidate_identity,
            }
            self._v41_active_perturbation = active
            self._v66_candidate_transaction = active
        except BaseException:
            self._restore_exact_master_v66("v71_candidate_failure_repair")
            raise
        ended_ns = time.monotonic_ns()
        return {
            "schema": "mirrored-es-candidate-materialized-v71",
            **{
                key: transaction[key] for key in (
                    "direction_seed", "sigma", "sign", "pair_id",
                    "evaluation_contract_sha256",
                )
            },
            "master_identity": master["identity"],
            "cached_master_identity": master["cache"],
            "candidate_identity": candidate_identity,
            "materialization": materialization,
            "base_identity": base,
            "master_validation_clone_bytes": 0,
            "exact_restore_required": True,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }

    def materialize_mirrored_adapter_v66(self, *args, **kwargs):
        """Route existing mirrored controllers through the V71 state path."""
        return self.materialize_mirrored_adapter_v71(*args, **kwargs)

    def restore_mirrored_adapter_v71(
        self, expected_master_sha256, reason, expected_pair_id=None
    ):
        receipt = super().restore_mirrored_adapter_v66(
            expected_master_sha256, reason, expected_pair_id
        )
        return {
            **receipt,
            "schema": "mirrored-es-exact-master-restore-v71",
            "master_validation_clone_bytes": 0,
        }

    def restore_mirrored_adapter_v66(self, *args, **kwargs):
        """Keep unknown-completion repair on the V71 exact restore path."""
        return self.restore_mirrored_adapter_v71(*args, **kwargs)

    def abort_mirrored_update_if_present_v66(
        self,
        expected_master_sha256,
        reason,
        expected_manifest_sha256=None,
    ):
        """Abort an uncertain update without redundantly materializing twice."""
        started_ns = time.monotonic_ns()
        self._require_not_poisoned_v66()
        self._require_installed_v41a()
        expected_master = state_v66._sha256_string_v66(
            expected_master_sha256, "expected update-abort master identity"
        )
        reason = str(reason)
        if not reason:
            raise ValueError("v71 update-abort reason is empty")
        expected_manifest = None
        if expected_manifest_sha256 is not None:
            expected_manifest = state_v66._sha256_string_v66(
                expected_manifest_sha256,
                "expected update-abort manifest",
            )
        pending = getattr(self, "_v41_pending_update", None)
        committed = getattr(self, "_v41_committed_rollback", None)
        active_manifest = None
        if isinstance(pending, dict):
            active_manifest = pending.get("manifest_sha256")
        elif isinstance(committed, dict):
            active_manifest = committed.get("manifest_sha256")
        if (
            expected_manifest is not None
            and active_manifest is not None
            and active_manifest != expected_manifest
        ):
            error = RuntimeError("v71 update-abort found another manifest")
            self._poison_v66("v71_update_abort_manifest", error)
            raise error
        try:
            if active_manifest is None:
                restored = self._restore_exact_master_v66(
                    "v71_update_abort_no_transaction"
                )
                rollback = None
                restored_identity = restored["master_identity"]
                materialization = restored["materialization"]
                base = restored["base_identity"]
            else:
                rollback = self.abort_sharded_adapter_update_v71(
                    active_manifest, reason
                )
                restored_identity = rollback["identity"]
                materialization = rollback["exact_restore"]["materialization"]
                base = rollback["exact_restore"]["base_exact_audit"]
        except BaseException as error:
            if getattr(self, "_v66_terminal_poison", None) is None:
                self._poison_v66("v71_update_abort", error)
            raise RuntimeError(
                "v71 could not prove exact update abort; actor poisoned"
            ) from error
        if restored_identity["sha256"] != expected_master:
            error = RuntimeError("v71 update abort restored another master")
            self._poison_v66("v71_update_abort_identity", error)
            raise error
        ended_ns = time.monotonic_ns()
        return {
            "schema": "mirrored-es-idempotent-update-abort-v71",
            "aborted": active_manifest is not None,
            "active_manifest_sha256": active_manifest,
            "expected_manifest_sha256": expected_manifest,
            "reason": reason,
            "rollback": rollback,
            "restored_identity": restored_identity,
            "materialization": materialization,
            "base_identity": base,
            "terminal_poisoned": False,
            "idempotent_when_prepare_or_execute_completion_unknown": True,
            "redundant_second_materialization_avoided": True,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }

    def audit_traffic_receipt_v71(self, candidate_count=16):
        self._require_not_poisoned_v66()
        observed = dict(self._v71_traffic)
        observed["total_device_transfer_bytes"] = (
            observed["h2d_bytes"]
            + observed["lora_d2h_bytes"]
            + observed["base_d2h_bytes"]
        )
        return {
            "schema": "eggroll-es-worker-audit-traffic-receipt-v71",
            "observed": observed,
            "byte_accounted_model": contract.traffic_account_v71(
                int(candidate_count)
            ),
            "completed_boundaries": list(self._v71_completed_boundaries),
            "population_acceptance_sha256": (
                self._v71_population_acceptance_sha256
            ),
            "terminal_poisoned": False,
        }
