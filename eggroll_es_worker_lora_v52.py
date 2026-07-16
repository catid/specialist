#!/usr/bin/env python3
"""V52 direct-master transitions with four-actor runtime identity sealing.

Candidate identities are a deterministic function of the sealed master, seed,
sigma, sign, and V41A implementation.  They are intentionally sealed by GPU
runtime consensus because torch CPU and CUDA RNG streams are not byte-equal.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch

import eggroll_es_worker_lora_v41a as state_v41a
import eggroll_es_worker_lora_v51 as state_v51


class LoRAAdapterStateWorkerExtensionV52(
    state_v51.LoRAAdapterStateWorkerExtensionV51,
):
    """Retain V51 scheduling while returning derived state identities."""

    def adapter_state_certificate_v52(self):
        """Certify the exact adapter state and every transaction namespace."""
        certificate = super().adapter_state_certificate_v41a()
        accepted = getattr(self, "_v43i_accepted_rollback", None)
        active_plan_id = getattr(self, "_v41_active_plan_id", None)
        if accepted is not None:
            raise RuntimeError("v52 certificate found an accepted rollback")
        if active_plan_id is not None:
            raise RuntimeError("v52 certificate found an active update plan")
        return {
            **certificate,
            "schema": "canonical-lora-state-certificate-v52",
            "transaction_state_quiescent": True,
            "active_perturbation": None,
            "pending_update": None,
            "committed_rollback": None,
            "accepted_rollback": None,
            "active_plan_id": None,
        }

    def transition_derived_antithetic_from_pinned_master_v52(
        self,
        state_index,
        seed,
        sigma,
        sign,
        expected_master_sha256,
        expected_previous_candidate_sha256,
    ):
        started_ns = time.monotonic_ns()
        state_index = int(state_index)
        seed = int(seed)
        sigma = float(sigma)
        sign = int(sign)
        expected_master_sha256 = str(expected_master_sha256)
        expected_previous_candidate_sha256 = str(
            expected_previous_candidate_sha256
        )
        if state_index < 0:
            raise ValueError("v52 state index must be nonnegative")
        self._require_population_transition_quiescent_v51()
        master_identity = state_v41a.adapter_identity_v41a(self._v41_master)
        if (
            master_identity != self._v41_current_identity
            or master_identity["sha256"] != expected_master_sha256
        ):
            raise RuntimeError("v52 pinned FP32 master identity changed")
        prior = self._v41_active_perturbation
        if state_index == 0:
            if prior is not None or (
                expected_previous_candidate_sha256 != expected_master_sha256
            ):
                raise RuntimeError("v52 first state did not start at master")
        elif (
            not isinstance(prior, dict)
            or int(prior.get("state_index", -1)) != state_index - 1
            or prior.get("candidate_identity", {}).get("sha256")
            != expected_previous_candidate_sha256
        ):
            raise RuntimeError("v52 prior perturbation continuity changed")

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
        materialization = self._materialize_v41a(
            candidate_cpu, "v52_direct_pinned_master_transition",
        )
        base_identity = self._base_check_v41a(
            "v52_direct_pinned_master_transition"
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
            "schema": "derived-direct-pinned-master-transition-v52",
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
            "identity_sealed_by_four_actor_runtime_consensus": True,
            "timing": {
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
            },
        }

    def save_pending_candidate_snapshot_v52(
        self,
        output_directory,
        expected_manifest_sha256,
        expected_candidate_sha256,
        expected_runtime_values_sha256,
    ):
        """Persist an executed-but-uncommitted candidate without moving master.

        V52 compares two population sizes from the same exact master.  A passing
        arm is therefore snapshotted while its transaction is still pending,
        then exactly aborted before the other arm is evaluated.  Only rank zero
        writes; every rank proves the same pending FP32 and BF16 identities.
        """
        self._require_installed_v41a()
        expected_manifest_sha256 = str(expected_manifest_sha256)
        expected_candidate_sha256 = str(expected_candidate_sha256)
        expected_runtime_values_sha256 = str(
            expected_runtime_values_sha256
        )
        pending = self._v41_pending_update
        if (
            self._v41_active_perturbation is not None
            or not isinstance(pending, dict)
            or pending.get("phase") != "executed"
            or pending.get("manifest_sha256") != expected_manifest_sha256
            or pending.get("candidate_identity", {}).get("sha256")
            != expected_candidate_sha256
            or self._v41_current_identity != pending.get("rollback_identity")
            or self._v41_committed_rollback is not None
            or getattr(self, "_v43i_accepted_rollback", None) is not None
        ):
            raise RuntimeError("v52 pending snapshot transaction changed")
        candidate = state_v41a._clone_master_v41a(
            pending["candidate_master"]
        )
        candidate_identity = state_v41a.adapter_identity_v41a(candidate)
        if candidate_identity != pending["candidate_identity"]:
            raise RuntimeError("v52 pending snapshot FP32 identity changed")
        materialization = self._materialize_v41a(
            candidate, "v52_pending_candidate_snapshot",
        )
        if (
            materialization["runtime_values_sha256"]
            != expected_runtime_values_sha256
        ):
            raise RuntimeError("v52 pending snapshot BF16 identity changed")
        base_identity = self._base_check_v41a(
            "v52_pending_candidate_snapshot"
        )
        rank = int(self.inter_pg.rank)
        output = Path(output_directory).resolve()
        result = {
            "schema": "uncommitted-canonical-peft-fp32-snapshot-v52",
            "rank": rank,
            "written": False,
            "directory": str(output),
            "manifest_sha256": expected_manifest_sha256,
            "candidate_identity": candidate_identity,
            "materialization": materialization,
            "base_identity": base_identity,
            "master_committed": False,
            "exact_abort_required_after_snapshot": True,
        }
        if rank != 0:
            return result
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        weights = output / "adapter_model.safetensors"
        config = output / "adapter_config.json"
        temporary_weights = output / (
            f".adapter_model.safetensors.tmp-{os.getpid()}"
        )
        temporary_config = output / (
            f".adapter_config.json.tmp-{os.getpid()}"
        )
        try:
            state_v41a.save_file(
                candidate,
                temporary_weights,
                metadata={
                    "format": "pt",
                    "schema": "uncommitted-canonical-peft-fp32-v52",
                    "candidate_sha256": candidate_identity["sha256"],
                    "manifest_sha256": expected_manifest_sha256,
                },
            )
            temporary_config.write_bytes(self._v41_config_bytes)
            os.link(temporary_weights, weights)
            os.link(temporary_config, config)
            readback = state_v41a._source_inventory_v41a(weights)
            readback_identity = state_v41a.adapter_identity_v41a(readback)
            if (
                readback_identity != candidate_identity
                or config.read_bytes() != self._v41_config_bytes
                or state_v41a.file_sha256_v41a(config)
                != self._v41_source_config_sha256
            ):
                raise RuntimeError("v52 pending snapshot readback changed")
        except Exception:
            weights.unlink(missing_ok=True)
            config.unlink(missing_ok=True)
            temporary_weights.unlink(missing_ok=True)
            temporary_config.unlink(missing_ok=True)
            try:
                output.rmdir()
            except OSError:
                pass
            raise
        finally:
            temporary_weights.unlink(missing_ok=True)
            temporary_config.unlink(missing_ok=True)
        result.update({
            "written": True,
            "weights_path": str(weights),
            "config_path": str(config),
            "weights_sha256": state_v41a.file_sha256_v41a(weights),
            "config_sha256": state_v41a.file_sha256_v41a(config),
            "readback_verified": True,
            "readback_identity": readback_identity,
            "original_canonical_key_namespace": True,
            "unscaled_fp32_candidate_persisted": True,
        })
        return result
