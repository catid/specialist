#!/usr/bin/env python3
"""Minimal fail-closed transaction guard layered on the verified V41A worker."""

from __future__ import annotations

import eggroll_es_worker_lora_v41a as base


class LoRAAdapterStateWorkerExtensionV43I(
    base.LoRAAdapterStateWorkerExtensionV41A,
):
    """Add an idempotent exact-abort/readback endpoint for controller failures."""

    def abort_or_readback_sharded_adapter_update_v43i(
        self,
        manifest_sha256,
        expected_master_sha256,
        expected_runtime_values_sha256,
    ):
        manifest_sha256 = str(manifest_sha256)
        expected_master_sha256 = str(expected_master_sha256)
        expected_runtime_values_sha256 = str(expected_runtime_values_sha256)
        pending = self._v41_pending_update
        committed = self._v41_committed_rollback
        accepted = getattr(self, "_v43i_accepted_rollback", None)
        if isinstance(pending, dict):
            if pending.get("manifest_sha256") != manifest_sha256:
                raise RuntimeError("v43i pending abort manifest changed")
            action = self.abort_sharded_adapter_update_v41a(manifest_sha256)
            disposition = "pending_update_rolled_back"
        elif isinstance(committed, dict):
            if committed.get("manifest_sha256") != manifest_sha256:
                raise RuntimeError("v43i committed abort manifest changed")
            action = self.abort_sharded_adapter_update_v41a(manifest_sha256)
            disposition = "partial_commit_rolled_back"
        elif isinstance(accepted, dict):
            if accepted.get("manifest_sha256") != manifest_sha256:
                raise RuntimeError("v43i accepted abort manifest changed")
            self._v41_committed_rollback = accepted
            self._v43i_accepted_rollback = None
            action = self.abort_sharded_adapter_update_v41a(manifest_sha256)
            disposition = "accepted_but_unsealed_update_rolled_back"
        else:
            if self._v41_current_identity["sha256"] != expected_master_sha256:
                raise RuntimeError("v43i quiescent abort readback found another master")
            action = {
                "schema": "canonical-lora-update-already-quiescent-v43i",
                "rolled_back": False,
                "identity": self._v41_current_identity,
            }
            disposition = "already_quiescent_at_expected_master"
        if (
            self._v41_pending_update is not None
            or self._v41_committed_rollback is not None
            or getattr(self, "_v43i_accepted_rollback", None) is not None
        ):
            raise RuntimeError("v43i abort left transactional state resident")
        verified = self._verify_master_materialized_v41a("v43i_abort_readback")
        base_identity = self._base_check_v41a("v43i_abort_readback")
        if (
            verified["master_identity"]["sha256"] != expected_master_sha256
            or verified["materialization"]["runtime_values_sha256"]
            != expected_runtime_values_sha256
        ):
            raise RuntimeError("v43i abort exact master/runtime readback changed")
        return {
            "schema": "canonical-lora-exact-abort-readback-v43i",
            "aborted_or_verified": True,
            "manifest_sha256": manifest_sha256,
            "disposition": disposition,
            "action": action,
            "master_identity": verified["master_identity"],
            "materialization": verified["materialization"],
            "base_identity": base_identity,
            "transaction_state_quiescent": True,
        }

    def validate_committed_adapter_update_v43i(
        self, manifest_sha256, expected_final_sha256,
    ):
        committed = self._v41_committed_rollback
        if (
            not isinstance(committed, dict)
            or committed.get("manifest_sha256") != str(manifest_sha256)
            or self._v41_current_identity["sha256"] != str(expected_final_sha256)
            or committed.get("committed_identity") != self._v41_current_identity
        ):
            raise RuntimeError("v43i committed candidate validation changed")
        verified = self._verify_master_materialized_v41a("v43i_preaccept_validate")
        base_identity = self._base_check_v41a("v43i_preaccept_validate")
        return {
            "schema": "canonical-lora-committed-validation-v43i",
            "validated": True,
            "manifest_sha256": str(manifest_sha256),
            "final_identity": self._v41_current_identity,
            "materialization": verified["materialization"],
            "base_identity": base_identity,
            "rollback_retained": True,
        }

    def accept_committed_adapter_update_v43i(
        self, manifest_sha256, expected_final_sha256,
    ):
        """Idempotently cross the accept point after all-rank validation."""
        manifest_sha256 = str(manifest_sha256)
        expected_final_sha256 = str(expected_final_sha256)
        committed = self._v41_committed_rollback
        if isinstance(committed, dict):
            if (
                committed.get("manifest_sha256") != manifest_sha256
                or self._v41_current_identity["sha256"] != expected_final_sha256
                or committed.get("committed_identity") != self._v41_current_identity
            ):
                raise RuntimeError("v43i accepted candidate identity changed")
            disposition = "accepted_with_rollback_retained"
            self._v43i_accepted_rollback = committed
            self._v41_committed_rollback = None
        elif (
            isinstance(getattr(self, "_v43i_accepted_rollback", None), dict)
            and self._v43i_accepted_rollback.get("manifest_sha256")
            == manifest_sha256
            and self._v41_current_identity["sha256"] == expected_final_sha256
        ):
            disposition = "already_accepted_with_rollback_retained"
        elif (
            self._v41_pending_update is None
            and self._v41_current_identity["sha256"] == expected_final_sha256
        ):
            disposition = "already_accepted"
        else:
            raise RuntimeError("v43i accept found an incompatible transaction state")
        verified = self._verify_master_materialized_v41a("v43i_accept_readback")
        base_identity = self._base_check_v41a("v43i_accept_readback")
        return {
            "schema": "canonical-lora-committed-accepted-v43i",
            "accepted": True,
            "manifest_sha256": manifest_sha256,
            "disposition": disposition,
            "final_identity": self._v41_current_identity,
            "materialization": verified["materialization"],
            "base_identity": base_identity,
            "transaction_state_quiescent": True,
            "rollback_retained_until_snapshot_sealed": (
                getattr(self, "_v43i_accepted_rollback", None) is not None
            ),
        }

    def release_accepted_adapter_rollback_v43i(
        self, manifest_sha256, expected_final_sha256,
    ):
        """Idempotently release rollback only after the snapshot is sealed."""
        manifest_sha256 = str(manifest_sha256)
        expected_final_sha256 = str(expected_final_sha256)
        accepted = getattr(self, "_v43i_accepted_rollback", None)
        if isinstance(accepted, dict):
            if (
                accepted.get("manifest_sha256") != manifest_sha256
                or self._v41_current_identity["sha256"] != expected_final_sha256
                or accepted.get("committed_identity") != self._v41_current_identity
            ):
                raise RuntimeError("v43i rollback release identity changed")
            self._v43i_accepted_rollback = None
            disposition = "released_after_snapshot"
        elif (
            self._v41_pending_update is None
            and self._v41_committed_rollback is None
            and self._v41_current_identity["sha256"] == expected_final_sha256
        ):
            disposition = "already_released"
        else:
            raise RuntimeError("v43i rollback release found incompatible state")
        verified = self._verify_master_materialized_v41a("v43i_release_readback")
        base_identity = self._base_check_v41a("v43i_release_readback")
        return {
            "schema": "canonical-lora-accepted-rollback-released-v43i",
            "released": True,
            "manifest_sha256": manifest_sha256,
            "disposition": disposition,
            "final_identity": self._v41_current_identity,
            "materialization": verified["materialization"],
            "base_identity": base_identity,
            "transaction_state_quiescent": True,
        }
