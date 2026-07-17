#!/usr/bin/env python3
"""Synthetic CPU-only tests for the additive V83 autograd contract."""

from __future__ import annotations

import ast
from types import SimpleNamespace

import pytest
import torch

import build_qwen36_lora_es_autograd_free_preregistration_v83 as build_v83
from eggroll_es_autograd_free_contract_v83 import (
    AutogradContractViolationV83,
    audit_autograd_surfaces_v83,
    run_autograd_free_transition_v83,
    validate_autograd_receipt_v83,
)
from eggroll_es_worker_lora_autograd_free_v83 import (
    LoRAAdapterStateWorkerExtensionV83,
)
from eggroll_es_worker_lora_pinned_transport_v81a import (
    LoRAAdapterStateWorkerExtensionV81A,
)


def _clean_worker_v83():
    master = torch.zeros(2, 3, dtype=torch.float32)
    runtime = torch.zeros(2, 3, dtype=torch.bfloat16)
    return SimpleNamespace(
        _v41_master={"adapter": master},
        _v71_runtime_views={"view": runtime},
        _v71_runtime_parents={"view": runtime},
        _v71_base_tensors={"base": torch.zeros(1, dtype=torch.int8)},
        _v41_pending_update=None,
        _v41_committed_rollback=None,
    )


def test_clean_surface_receipt_is_metadata_only_and_valid():
    receipt = audit_autograd_surfaces_v83(_clean_worker_v83(), "synthetic")
    assert validate_autograd_receipt_v83(receipt) == receipt
    assert receipt["unique_tensor_count"] == 3
    assert receipt["surface_reference_count"] > receipt["unique_tensor_count"]
    assert receipt["tensor_content_read_bytes"] == 0
    assert receipt["tensor_clone_bytes"] == 0
    assert receipt["full_model_clone_bytes"] == 0
    assert receipt["arbitrary_object_graph_traversed"] is False


def test_requires_grad_tensor_is_rejected():
    worker = _clean_worker_v83()
    worker._v41_master["adapter"] = torch.ones(
        2, 3, dtype=torch.float32, requires_grad=True
    )
    with pytest.raises(AutogradContractViolationV83, match="requires_grad=True"):
        audit_autograd_surfaces_v83(worker, "requires-grad-fault")


def test_non_none_grad_is_rejected_even_after_requires_grad_disabled():
    worker = _clean_worker_v83()
    tensor = torch.ones(2, 3, dtype=torch.float32, requires_grad=True)
    tensor.grad = torch.ones_like(tensor)
    tensor.requires_grad_(False)
    worker._v41_master["adapter"] = tensor
    with pytest.raises(AutogradContractViolationV83, match="non-None .grad"):
        audit_autograd_surfaces_v83(worker, "retained-grad-fault")


def test_hidden_registered_tensor_is_also_checked():
    worker = _clean_worker_v83()
    worker._opaque = {"nested": [torch.ones(1, requires_grad=True)]}
    with pytest.raises(AutogradContractViolationV83, match="requires_grad=True"):
        audit_autograd_surfaces_v83(worker, "hidden-tensor-fault")


def test_grad_enabled_tensor_operation_inside_transition_is_rejected():
    worker = _clean_worker_v83()

    def bad_transition():
        with torch.enable_grad():
            return torch.add(worker._v41_master["adapter"], 1.0)

    with pytest.raises(
        AutogradContractViolationV83,
        match="tensor operation with autograd enabled",
    ):
        run_autograd_free_transition_v83(
            worker, "grad-enabled-fault", bad_transition
        )


def test_swallowed_grad_enabled_operation_violation_is_still_rejected():
    worker = _clean_worker_v83()

    def bad_transition():
        try:
            with torch.enable_grad():
                torch.add(worker._v41_master["adapter"], 1.0)
        except AutogradContractViolationV83:
            pass
        return None

    with pytest.raises(AutogradContractViolationV83, match="suppressed"):
        run_autograd_free_transition_v83(
            worker, "swallowed-grad-fault", bad_transition
        )


def test_grad_mode_leak_without_tensor_op_is_rejected_and_restored():
    worker = _clean_worker_v83()
    caller_mode = torch.is_grad_enabled()

    def bad_transition():
        torch.set_grad_enabled(True)
        return None

    with pytest.raises(AutogradContractViolationV83, match="leaked enabled"):
        run_autograd_free_transition_v83(
            worker, "grad-mode-leak", bad_transition
        )
    assert torch.is_grad_enabled() is caller_mode


def test_clean_transition_forces_no_grad_and_returns_receipt():
    worker = _clean_worker_v83()
    observed = []

    def transition():
        observed.append(torch.is_grad_enabled())
        worker._v41_master["adapter"].add_(2.0)
        return {"ok": True}

    result, receipt = run_autograd_free_transition_v83(
        worker,
        "clean-transition",
        transition,
        after_surfaces=lambda value: {"result": value},
    )
    assert result == {"ok": True}
    assert observed == [False]
    assert receipt["passed"] is True
    assert receipt["guard_mode"] == (
        "torch.no_grad+reject_grad_enabled_tensor_ops"
    )


class _SyntheticHiddenOptimizerV83(torch.optim.Optimizer):
    pass


def test_hidden_torch_optimizer_registry_is_rejected():
    worker = _clean_worker_v83()
    hidden = object.__new__(_SyntheticHiddenOptimizerV83)
    worker._opaque_registry = {"innocent_name": hidden}
    with pytest.raises(AutogradContractViolationV83, match="torch optimizer"):
        audit_autograd_surfaces_v83(worker, "hidden-optimizer-fault")


def test_hidden_optimizer_state_dict_registry_is_rejected():
    worker = _clean_worker_v83()
    worker._opaque_registry = {"state": {}, "param_groups": []}
    with pytest.raises(
        AutogradContractViolationV83, match="optimizer state-dict"
    ):
        audit_autograd_surfaces_v83(worker, "hidden-state-fault")


def test_additive_subclass_poison_is_terminal_after_surface_fault():
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV83)
    worker._v41_master = {"adapter": torch.ones(1, requires_grad=True)}
    with pytest.raises(AutogradContractViolationV83):
        worker._run_transition_v83("poison-fault", lambda: None)
    assert worker._v83_autograd_terminal_poison["terminal"] is True
    with pytest.raises(AutogradContractViolationV83, match="terminally poisoned"):
        worker._run_transition_v83("after-poison", lambda: None)


def test_single_worker_status_cannot_claim_four_gpu_acceptance():
    clean = _clean_worker_v83()
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV83)
    vars(worker).update(vars(clean))
    receipt = worker.autograd_free_status_receipt_v83("synthetic-status")
    assert receipt["single_worker_surface_receipt"] is True
    assert receipt["live_four_gpu_receipt"] is False
    assert receipt["requires_controller_four_worker_aggregate"] is True
    assert receipt["transition_receipt_count"] == 0


def test_v83_is_additive_v81a_subclass_without_parent_edits():
    assert issubclass(
        LoRAAdapterStateWorkerExtensionV83,
        LoRAAdapterStateWorkerExtensionV81A,
    )
    evidence = build_v83.build_source_evidence_v83()
    assert evidence["accepted_trainers_immutable"] is True
    assert evidence["transition_coverage"]["missing_transitions"] == []
    assert evidence["transition_coverage"]["unguarded_transitions"] == []


def test_enforcement_source_contains_no_tensor_clone_call():
    for filename in (
        "eggroll_es_autograd_free_contract_v83.py",
        "eggroll_es_worker_lora_autograd_free_v83.py",
    ):
        tree = ast.parse(
            (build_v83.ROOT / filename).read_text(encoding="utf-8"),
            filename=filename,
        )
        clone_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "clone"
        ]
        assert clone_calls == []


def test_preregistration_withholds_live_and_promotion_claims():
    prereg = build_v83.build_preregistration_v83()
    gate = prereg["live_acceptance_gate"]
    assert gate["dependency_satisfied_in_this_milestone"] is False
    assert gate["live_receipt_present"] is False
    assert gate["bead_must_remain_in_progress"] is True
    assert "peak_vram_reduction" in prereg["claims_withheld"]
    assert "training_or_hpo_promotion" in prereg["claims_withheld"]


def test_checked_artifacts_match_deterministic_builders():
    prereg, evidence = build_v83.check_artifacts_v83()
    assert prereg == build_v83.build_preregistration_v83()
    assert evidence == build_v83.build_source_evidence_v83()
