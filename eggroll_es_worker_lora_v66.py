#!/usr/bin/env python3
"""Transactional mirrored-ES endpoints over the current canonical LoRA stack.

V66 retains the V65B/V64/V52/V41A worker surface and adds a small production
surface for mirrored evaluation.  A candidate transaction is recorded before
the first runtime-slot write.  Candidate failures reconstruct the sole vLLM
slot from the immutable FP32 master; restoration failures poison the worker so
later inherited state operations also fail closed.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

import eggroll_es_worker_lora_v41a as state_v41a
from eggroll_es_mirrored_v66 import canonical_sha256_v66
from eggroll_es_worker_lora_v65b import LoRAAdapterStateWorkerExtensionV65B


NOISE_PROTOCOL_V66 = "v41a-sha256-keyed-torch-fp32-exact-sign-negation-v66"


def signed_noise_like_v66(
    tensor: torch.Tensor,
    key: str,
    seed: int,
    sign: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Return +epsilon or its exact elementwise negation from one seed rule."""
    if isinstance(sign, bool) or int(sign) not in (-1, 1):
        raise ValueError("v66 signed noise requires sign +/-1")
    direction = state_v41a.noise_like_v41a(tensor, key, seed, device)
    return direction if int(sign) == 1 else direction.neg()


def antithetic_candidate_v66(
    master: dict[str, torch.Tensor],
    seed: int,
    sigma: float,
    sign: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Materialize one sign without retaining a second candidate or noise copy."""
    master = state_v41a._validate_master_v41a(master)
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("v66 perturbation sigma must be finite and positive")
    if isinstance(sign, bool) or int(sign) not in (-1, 1):
        raise ValueError("v66 perturbation sign must be +/-1")
    device = torch.device(device)
    candidate = {}
    for key, tensor in master.items():
        signed_direction = signed_noise_like_v66(
            tensor, key, seed, int(sign), device
        )
        candidate[key] = tensor.to(
            device=device, dtype=torch.float32
        ).add_(signed_direction, alpha=sigma)
    return candidate


def _sha256_string_v66(value: Any, label: str) -> str:
    value = str(value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"v66 {label} must be a lowercase SHA-256")
    return value


class LoRAAdapterStateWorkerExtensionV66(LoRAAdapterStateWorkerExtensionV65B):
    """Add fail-closed mirrored candidates to the latest LoRA worker stack."""

    def _require_not_poisoned_v66(self) -> None:
        poison = getattr(self, "_v66_terminal_poison", None)
        if poison is not None:
            raise RuntimeError(
                "v66 worker is terminally poisoned after an unverified restore: "
                f"{poison.get('error_type')}: {poison.get('error')}"
            )

    def _require_installed_v41a(self):
        super()._require_installed_v41a()
        self._require_not_poisoned_v66()

    def _poison_v66(self, phase: str, error: BaseException) -> None:
        self._v66_terminal_poison = {
            "schema": "mirrored-es-terminal-poison-v66",
            "phase": str(phase),
            "error_type": type(error).__name__,
            "error": str(error),
            "requires_actor_recreation": True,
        }

    def active_lora_slot_certificate_v66(self, expected_lora_int_id=1):
        """Prove adapter 1 is active before any canonical-state write."""
        self._require_not_poisoned_v66()
        if isinstance(expected_lora_int_id, bool) or expected_lora_int_id != 1:
            raise ValueError("v66 expected active LoRA id must be exactly 1")
        config = getattr(self, "lora_config", None)
        runner = getattr(self, "model_runner", None)
        facade = getattr(runner, "lora_manager", None)
        manager = state_v41a.topology._manager(self)
        active = [
            int(value)
            for value in getattr(manager, "lora_index_to_id", [])
            if value is not None
        ]
        active_cache = [
            int(value)
            for value in getattr(
                getattr(manager, "_active_adapters", None), "cache", {}
            )
        ]
        loaded = sorted(int(value) for value in facade.list_adapters()) \
            if facade is not None else []
        if (
            config is None
            or config.max_loras != 1
            or config.max_cpu_loras != 1
            or config.max_lora_rank != 32
            or active != [1]
            or active_cache != [1]
            or loaded != [1]
            or manager.lora_index_to_id.index(1) != 0
        ):
            raise RuntimeError("v66 active sole-slot LoRA identity changed")
        return {
            "schema": "v66-active-lora-slot-certificate",
            "expected_lora_int_id": 1,
            "active_lora_ids": active,
            "active_manager_cache_lora_ids": active_cache,
            "loaded_cpu_cache_lora_ids": loaded,
            "active_slot_index": 0,
            "max_loras": int(config.max_loras),
            "max_cpu_loras": int(config.max_cpu_loras),
            "max_lora_rank": int(config.max_lora_rank),
            "canonical_state_write_performed": False,
        }

    def _restore_exact_master_v66(self, phase: str) -> dict:
        """Unconditionally reconstruct the live slot; poison on uncertainty."""
        master_identity = state_v41a.adapter_identity_v41a(self._v41_master)
        if master_identity != self._v41_current_identity:
            error = RuntimeError("v66 canonical FP32 master identity changed")
            self._poison_v66(phase, error)
            raise error
        try:
            materialization = self._materialize_v41a(
                self._v41_master, f"v66_{phase}_exact_master"
            )
            base_identity = self._base_check_v41a(
                f"v66_{phase}_exact_master"
            )
        except BaseException as error:
            self._poison_v66(phase, error)
            raise RuntimeError(
                "v66 could not prove exact master restoration; actor poisoned"
            ) from error
        self._v41_active_perturbation = None
        self._v66_candidate_transaction = None
        return {
            "master_identity": master_identity,
            "materialization": materialization,
            "base_identity": base_identity,
        }

    def materialize_mirrored_adapter_v66(
        self,
        direction_seed,
        sigma,
        sign,
        pair_id,
        evaluation_contract_sha256,
        expected_master_sha256,
    ):
        """Write one signed candidate while the canonical master stays immutable."""
        started_ns = time.monotonic_ns()
        self._require_not_poisoned_v66()
        self._require_quiescent_v41a()
        direction_seed = int(direction_seed)
        sigma = float(sigma)
        sign = int(sign)
        pair_id = _sha256_string_v66(pair_id, "pair id")
        contract_sha = _sha256_string_v66(
            evaluation_contract_sha256, "evaluation contract identity"
        )
        expected_master_sha = _sha256_string_v66(
            expected_master_sha256, "expected master identity"
        )
        if direction_seed < 0 or not math.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("v66 seed/sigma changed")
        if sign not in (-1, 1):
            raise ValueError("v66 sign must be +/-1")
        master_identity = state_v41a.adapter_identity_v41a(self._v41_master)
        if (
            master_identity != self._v41_current_identity
            or master_identity["sha256"] != expected_master_sha
        ):
            raise RuntimeError("v66 mirrored candidate master identity changed")
        transaction = {
            "schema": "mirrored-es-candidate-transaction-v66",
            "phase": "before_runtime_write",
            "direction_seed": direction_seed,
            "sigma": sigma,
            "sign": sign,
            "pair_id": pair_id,
            "evaluation_contract_sha256": contract_sha,
            "expected_master_sha256": expected_master_sha,
        }
        self._v66_candidate_transaction = transaction
        candidate_device = getattr(
            self,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        try:
            candidate = antithetic_candidate_v66(
                self._v41_master,
                direction_seed,
                sigma,
                sign,
                candidate_device,
            )
            candidate_cpu = {
                key: tensor.detach().cpu().contiguous()
                for key, tensor in candidate.items()
            }
            candidate_identity = state_v41a.adapter_identity_v41a(candidate_cpu)
            transaction["phase"] = "runtime_write_started"
            materialization = self._materialize_v41a(
                candidate_cpu, "v66_mirrored_candidate"
            )
            base_identity = self._base_check_v41a("v66_mirrored_candidate")
            active = {
                **transaction,
                "phase": "candidate_active",
                "candidate_identity": candidate_identity,
            }
            self._v41_active_perturbation = active
            self._v66_candidate_transaction = active
        except BaseException:
            # A failed write can leave an arbitrary prefix of the BF16 views
            # changed.  Repair before exposing the original error.  If repair
            # itself fails, _restore_exact_master_v66 poisons the actor.
            self._restore_exact_master_v66("candidate_failure_repair")
            raise
        ended_ns = time.monotonic_ns()
        return {
            "schema": "mirrored-es-candidate-materialized-v66",
            "direction_seed": direction_seed,
            "sigma": sigma,
            "sign": sign,
            "pair_id": pair_id,
            "evaluation_contract_sha256": contract_sha,
            "master_identity": master_identity,
            "candidate_identity": candidate_identity,
            "materialization": materialization,
            "base_identity": base_identity,
            "noise_protocol": NOISE_PROTOCOL_V66,
            "master_unchanged": state_v41a.adapter_identity_v41a(
                self._v41_master
            ) == master_identity,
            "exact_restore_required": True,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }

    def restore_mirrored_adapter_v66(
        self,
        expected_master_sha256,
        reason,
        expected_pair_id=None,
    ):
        """Idempotently restore even when candidate RPC completion is unknown."""
        started_ns = time.monotonic_ns()
        self._require_not_poisoned_v66()
        self._require_installed_v41a()
        expected_master_sha = _sha256_string_v66(
            expected_master_sha256, "expected restore master identity"
        )
        reason = str(reason)
        if not reason:
            raise ValueError("v66 restore reason must not be empty")
        prior = getattr(self, "_v66_candidate_transaction", None)
        pair_mismatch = False
        if expected_pair_id is not None:
            expected_pair_id = _sha256_string_v66(
                expected_pair_id, "expected restore pair id"
            )
            pair_mismatch = (
                isinstance(prior, dict)
                and prior.get("pair_id") != expected_pair_id
            )
        restored = self._restore_exact_master_v66("controller_restore")
        if restored["master_identity"]["sha256"] != expected_master_sha:
            error = RuntimeError("v66 restored a different canonical master")
            self._poison_v66("controller_restore_identity", error)
            raise error
        if pair_mismatch:
            raise RuntimeError(
                "v66 active pair differed, but the exact master was restored"
            )
        ended_ns = time.monotonic_ns()
        return {
            "schema": "mirrored-es-exact-master-restore-v66",
            "restored": True,
            "reason": reason,
            "prior_transaction": prior,
            "restored_identity": restored["master_identity"],
            "materialization": restored["materialization"],
            "base_identity": restored["base_identity"],
            "algebraic_bf16_restore_used": False,
            "idempotent_when_candidate_completion_unknown": True,
            "terminal_poisoned": False,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }

    def abort_mirrored_update_if_present_v66(
        self,
        expected_master_sha256,
        reason,
        expected_manifest_sha256=None,
    ):
        """Idempotently abort a possibly partial prepare/execute RPC wave."""
        started_ns = time.monotonic_ns()
        self._require_not_poisoned_v66()
        self._require_installed_v41a()
        expected_master_sha = _sha256_string_v66(
            expected_master_sha256, "expected update-abort master identity"
        )
        reason = str(reason)
        if not reason:
            raise ValueError("v66 update-abort reason must not be empty")
        expected_manifest = None
        if expected_manifest_sha256 is not None:
            expected_manifest = _sha256_string_v66(
                expected_manifest_sha256, "expected update-abort manifest"
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
            error = RuntimeError("v66 update-abort found another manifest")
            self._poison_v66("update_abort_manifest", error)
            raise error

        rollback = None
        try:
            if active_manifest is not None:
                rollback = self.abort_sharded_adapter_update_v41a(
                    active_manifest
                )
            restored = self._restore_exact_master_v66("update_abort")
        except BaseException as error:
            if getattr(self, "_v66_terminal_poison", None) is None:
                self._poison_v66("update_abort", error)
            raise RuntimeError(
                "v66 could not prove exact update abort; actor poisoned"
            ) from error
        if restored["master_identity"]["sha256"] != expected_master_sha:
            error = RuntimeError("v66 update abort restored another master")
            self._poison_v66("update_abort_identity", error)
            raise error
        ended_ns = time.monotonic_ns()
        return {
            "schema": "mirrored-es-idempotent-update-abort-v66",
            "aborted": active_manifest is not None,
            "active_manifest_sha256": active_manifest,
            "expected_manifest_sha256": expected_manifest,
            "reason": reason,
            "rollback": rollback,
            "restored_identity": restored["master_identity"],
            "materialization": restored["materialization"],
            "terminal_poisoned": False,
            "idempotent_when_prepare_or_execute_completion_unknown": True,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }

    def mirrored_adapter_state_certificate_v66(self):
        self._require_not_poisoned_v66()
        if getattr(self, "_v66_candidate_transaction", None) is not None:
            raise RuntimeError("v66 state certificate found an active candidate")
        certificate = self.adapter_state_certificate_v52()
        if certificate.get("transaction_state_quiescent") is not True:
            raise RuntimeError("v66 inherited transaction state changed")
        return {
            **certificate,
            "schema": "mirrored-es-canonical-lora-state-certificate-v66",
            "noise_protocol": NOISE_PROTOCOL_V66,
            "terminal_poisoned": False,
            "mirrored_surface_sha256": canonical_sha256_v66({
                "worker": type(self).__name__,
                "noise_protocol": NOISE_PROTOCOL_V66,
                "master_sha256": certificate["current_identity"]["sha256"],
            }),
        }
