#!/usr/bin/env python3
"""Pinned-master population transitions with exact V41A reconstruction.

V41A deliberately requires an exact restore between perturbations.  V51 keeps
the same immutable FP32 master and sole vLLM LoRA slot, but permits the slot to
move directly from one perturbation to the next.  Every state is independently
reconstructed from the pinned master; no candidate is derived from the prior
candidate and no BF16 algebraic rollback is used.
"""

from __future__ import annotations

import time

import torch

import eggroll_es_worker_lora_v41a as state_v41a
import eggroll_es_worker_lora_v43i as state_v43i


class LoRAAdapterStateWorkerExtensionV51(
    state_v43i.LoRAAdapterStateWorkerExtensionV43I,
):
    """Add fail-closed direct transitions between antithetic states."""

    def _require_population_transition_quiescent_v51(self) -> None:
        self._require_installed_v41a()
        if self._v41_pending_update is not None:
            raise RuntimeError("v51 transition found a pending adapter update")
        if self._v41_committed_rollback is not None:
            raise RuntimeError("v51 transition found a committed rollback")
        if getattr(self, "_v43i_accepted_rollback", None) is not None:
            raise RuntimeError("v51 transition found an accepted rollback")
        if (
            not self._v41_reference_fresh
            or self._v41_reference_identity != self._v41_current_identity
        ):
            raise RuntimeError("v51 pinned reference is not fresh and exact")

    def transition_antithetic_from_pinned_master_v51(
        self,
        state_index,
        seed,
        sigma,
        sign,
        expected_master_sha256,
        expected_previous_candidate_sha256,
        expected_candidate_sha256,
        expected_runtime_values_sha256,
    ):
        """Replace the runtime slot directly from the immutable FP32 master."""
        started_ns = time.monotonic_ns()
        state_index = int(state_index)
        seed = int(seed)
        sigma = float(sigma)
        sign = int(sign)
        expected_master_sha256 = str(expected_master_sha256)
        expected_previous_candidate_sha256 = str(
            expected_previous_candidate_sha256
        )
        expected_candidate_sha256 = str(expected_candidate_sha256)
        expected_runtime_values_sha256 = str(expected_runtime_values_sha256)
        if state_index < 0:
            raise ValueError("v51 state index must be nonnegative")
        self._require_population_transition_quiescent_v51()

        master_identity = state_v41a.adapter_identity_v41a(self._v41_master)
        if (
            master_identity != self._v41_current_identity
            or master_identity["sha256"] != expected_master_sha256
        ):
            raise RuntimeError("v51 pinned FP32 master identity changed")

        prior = self._v41_active_perturbation
        if state_index == 0:
            if prior is not None or (
                expected_previous_candidate_sha256 != expected_master_sha256
            ):
                raise RuntimeError("v51 first state did not start at the master")
        elif (
            not isinstance(prior, dict)
            or int(prior.get("state_index", -1)) != state_index - 1
            or prior.get("candidate_identity", {}).get("sha256")
            != expected_previous_candidate_sha256
        ):
            raise RuntimeError("v51 prior perturbation continuity changed")

        candidate_device = getattr(
            self,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        candidate = state_v41a.antithetic_candidate_v41a(
            self._v41_master, seed, sigma, sign, candidate_device,
        )
        candidate_cpu = {
            key: tensor.cpu().contiguous() for key, tensor in candidate.items()
        }
        candidate_identity = state_v41a.adapter_identity_v41a(candidate_cpu)
        if candidate_identity["sha256"] != expected_candidate_sha256:
            raise RuntimeError("v51 direct candidate reconstruction changed")
        materialization = self._materialize_v41a(
            candidate_cpu, "v51_direct_pinned_master_transition",
        )
        if (
            materialization["runtime_values_sha256"]
            != expected_runtime_values_sha256
        ):
            raise RuntimeError("v51 direct runtime materialization changed")
        base_identity = self._base_check_v41a(
            "v51_direct_pinned_master_transition"
        )
        self._v41_active_perturbation = {
            "state_index": state_index,
            "seed": seed,
            "sigma": sigma,
            "sign": sign,
            "base_identity": self._v41_current_identity,
            "candidate_identity": candidate_identity,
        }
        ended_ns = time.monotonic_ns()
        return {
            "schema": "direct-pinned-master-transition-v51",
            "state_index": state_index,
            "seed": seed,
            "sigma": sigma,
            "sign": sign,
            "previous_candidate_sha256": expected_previous_candidate_sha256,
            "candidate_identity": candidate_identity,
            "materialization": materialization,
            "master_identity": master_identity,
            "base_identity": base_identity,
            "direct_from_pinned_fp32_master": True,
            "cumulative_candidate_delta_used": False,
            "intermediate_master_restore_elided": state_index > 0,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }

    def restore_pinned_master_v51(
        self,
        expected_master_sha256,
        expected_runtime_values_sha256,
        reason,
    ):
        """Unconditionally reconstruct and verify the exact pinned master."""
        started_ns = time.monotonic_ns()
        expected_master_sha256 = str(expected_master_sha256)
        expected_runtime_values_sha256 = str(expected_runtime_values_sha256)
        reason = str(reason)
        self._require_population_transition_quiescent_v51()
        master_identity = state_v41a.adapter_identity_v41a(self._v41_master)
        if (
            master_identity != self._v41_current_identity
            or master_identity["sha256"] != expected_master_sha256
        ):
            raise RuntimeError("v51 final restore found another FP32 master")
        prior = self._v41_active_perturbation
        materialization = self._materialize_v41a(
            self._v41_master, "v51_exact_final_restore",
        )
        if (
            materialization["runtime_values_sha256"]
            != expected_runtime_values_sha256
        ):
            raise RuntimeError("v51 final master runtime readback changed")
        base_identity = self._base_check_v41a("v51_exact_final_restore")
        self._v41_active_perturbation = None
        ended_ns = time.monotonic_ns()
        return {
            "schema": "exact-pinned-master-restore-v51",
            "restored": True,
            "reason": reason,
            "prior_perturbation": prior,
            "restored_identity": master_identity,
            "materialization": materialization,
            "base_identity": base_identity,
            "algebraic_bf16_restore_used": False,
            "transaction_state_quiescent": True,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }
