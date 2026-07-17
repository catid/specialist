#!/usr/bin/env python3
"""Additive autograd-free enforcement for the accepted V81A LoRA ES worker.

V71, V72, and V81A remain immutable.  This subclass wraps state transitions
under the V83 no-grad tripwire and exposes a metadata-only live receipt.  It
does not create a model, optimizer, tensor bank, CUDA object, or clone.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from eggroll_es_autograd_free_contract_v83 import (
    AutogradContractViolationV83,
    audit_autograd_surfaces_v83,
    canonical_sha256_v83,
    run_autograd_free_transition_v83,
)
from eggroll_es_worker_lora_pinned_transport_v81a import (
    LoRAAdapterStateWorkerExtensionV81A,
)


WORKER_SCHEMA_V83 = "eggroll-es-lora-autograd-free-worker-v83"


class LoRAAdapterStateWorkerExtensionV83(
    LoRAAdapterStateWorkerExtensionV81A
):
    """V81A transport with fail-closed, autograd-free ES transitions."""

    def _poison_autograd_v83(self, phase: str, error: BaseException):
        poison = {
            "schema": "eggroll-es-autograd-terminal-poison-v83",
            "phase": str(phase),
            "error_type": type(error).__name__,
            "terminal": True,
        }
        poison["receipt_sha256"] = canonical_sha256_v83(poison)
        self._v83_autograd_terminal_poison = poison
        poison_v66 = getattr(self, "_poison_v66", None)
        if callable(poison_v66) and getattr(
            self, "_v66_terminal_poison", None
        ) is None:
            try:
                poison_v66(f"v83_autograd_{phase}", error)
            except BaseException:
                # The additive poison remains authoritative even if an
                # incompletely initialized synthetic/failed parent cannot
                # record its older poison format.
                pass

    def _require_not_poisoned_v83(self):
        if getattr(self, "_v83_autograd_terminal_poison", None) is not None:
            raise AutogradContractViolationV83(
                "v83 autograd contract is terminally poisoned"
            )

    def _run_transition_v83(
        self,
        phase: str,
        operation,
        *,
        before_surfaces: Mapping[str, Any] | None = None,
        after_surfaces=None,
    ):
        self._require_not_poisoned_v83()
        depth = int(getattr(self, "_v83_transition_depth", 0))
        if depth:
            if torch.is_grad_enabled():
                error = AutogradContractViolationV83(
                    "v83 nested transition escaped the no-grad guard"
                )
                self._poison_autograd_v83(phase, error)
                raise error
            return operation()
        self._v83_transition_depth = 1
        try:
            result, receipt = run_autograd_free_transition_v83(
                self,
                phase,
                operation,
                before_surfaces=before_surfaces,
                after_surfaces=after_surfaces,
            )
        except AutogradContractViolationV83 as error:
            self._poison_autograd_v83(phase, error)
            raise
        finally:
            self._v83_transition_depth = 0
        receipts = getattr(self, "_v83_transition_receipt_sha256", None)
        if receipts is None:
            receipts = []
            self._v83_transition_receipt_sha256 = receipts
        receipts.append(receipt["receipt_sha256"])
        return result

    @staticmethod
    def _result_surfaces_v83(result):
        return {"result": result}

    def install_adapter_state_v41a(self, *args, **kwargs):
        parent = super().install_adapter_state_v41a
        return self._run_transition_v83(
            "install_adapter_state",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def capture_adapter_reference_v41a(self):
        parent = super().capture_adapter_reference_v41a
        return self._run_transition_v83(
            "capture_adapter_reference",
            parent,
            after_surfaces=self._result_surfaces_v83,
        )

    def _materialize_v41a(self, tensors, phase):
        parent = super()._materialize_v41a
        return self._run_transition_v83(
            f"materialize:{phase}",
            lambda: parent(tensors, phase),
            before_surfaces={"materialization_input": tensors},
            after_surfaces=lambda result: {
                "materialization_input": tensors,
                "result": result,
            },
        )

    def materialize_mirrored_adapter_v71(self, *args, **kwargs):
        parent = super().materialize_mirrored_adapter_v71
        return self._run_transition_v83(
            "materialize_mirrored_candidate",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def restore_mirrored_adapter_v71(self, *args, **kwargs):
        parent = super().restore_mirrored_adapter_v71
        return self._run_transition_v83(
            "restore_mirrored_adapter",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def _restore_exact_master_v66(self, phase):
        parent = super()._restore_exact_master_v66
        return self._run_transition_v83(
            f"restore_exact_master:{phase}",
            lambda: parent(phase),
            after_surfaces=self._result_surfaces_v83,
        )

    def _restore_update_master_v71(
        self, master, identity, phase, *, precomputed_identity=None
    ):
        parent = super()._restore_update_master_v71
        return self._run_transition_v83(
            f"restore_update_master:{phase}",
            lambda: parent(
                master,
                identity,
                phase,
                precomputed_identity=precomputed_identity,
            ),
            before_surfaces={"rollback_master": master},
            after_surfaces=lambda result: {
                "rollback_master": master,
                "result": result,
            },
        )

    def accept_population_rewards_v71(self, *args, **kwargs):
        parent = super().accept_population_rewards_v71
        return self._run_transition_v83(
            "accept_population_rewards",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def prepare_sharded_adapter_update_v72(self, *args, **kwargs):
        parent = super().prepare_sharded_adapter_update_v72
        return self._run_transition_v83(
            "prepare_sharded_adapter_update",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def execute_sharded_adapter_update_v41a(self, *args, **kwargs):
        parent = super().execute_sharded_adapter_update_v41a
        return self._run_transition_v83(
            "execute_sharded_adapter_update",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def commit_sharded_adapter_update_v71(self, *args, **kwargs):
        parent = super().commit_sharded_adapter_update_v71
        return self._run_transition_v83(
            "commit_sharded_adapter_update",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def abort_sharded_adapter_update_v71(self, *args, **kwargs):
        parent = super().abort_sharded_adapter_update_v71
        return self._run_transition_v83(
            "abort_sharded_adapter_update",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def finalize_sharded_adapter_update_v71(self, *args, **kwargs):
        parent = super().finalize_sharded_adapter_update_v71
        return self._run_transition_v83(
            "finalize_sharded_adapter_update",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def abort_mirrored_update_if_present_v66(self, *args, **kwargs):
        parent = super().abort_mirrored_update_if_present_v66
        return self._run_transition_v83(
            "abort_mirrored_update_if_present",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def begin_actor_gpu_work_v66d(self, *args, **kwargs):
        parent = super().begin_actor_gpu_work_v66d
        return self._run_transition_v83(
            "begin_actor_generation",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def end_actor_gpu_work_v66d(self, *args, **kwargs):
        parent = super().end_actor_gpu_work_v66d
        return self._run_transition_v83(
            "end_actor_generation",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def save_adapter_snapshot_v41a(self, *args, **kwargs):
        parent = super().save_adapter_snapshot_v41a
        return self._run_transition_v83(
            "save_adapter_snapshot",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def final_transport_receipt_v81a(self, *args, **kwargs):
        parent = super().final_transport_receipt_v81a
        return self._run_transition_v83(
            "final_transport_release",
            lambda: parent(*args, **kwargs),
            after_surfaces=self._result_surfaces_v83,
        )

    def autograd_free_status_receipt_v83(self, phase="live_status"):
        self._require_not_poisoned_v83()
        with torch.no_grad():
            surface = audit_autograd_surfaces_v83(
                self,
                str(phase),
                require_grad_disabled=True,
            )
        receipt = {
            "schema": WORKER_SCHEMA_V83,
            "phase": str(phase),
            "passed": True,
            "surface_receipt_sha256": surface["receipt_sha256"],
            "transition_receipt_sha256": list(getattr(
                self, "_v83_transition_receipt_sha256", []
            )),
            "transition_receipt_count": len(getattr(
                self, "_v83_transition_receipt_sha256", []
            )),
            "terminal_poisoned": False,
            "torch_optimizer_authority_present": False,
            "tensor_content_read_bytes": 0,
            "tensor_clone_bytes": 0,
            "full_model_clone_bytes": 0,
            "single_worker_surface_receipt": True,
            "live_four_gpu_receipt": False,
            "requires_controller_four_worker_aggregate": True,
        }
        receipt["receipt_sha256"] = canonical_sha256_v83(receipt)
        return receipt


__all__ = [
    "LoRAAdapterStateWorkerExtensionV83",
    "WORKER_SCHEMA_V83",
]
