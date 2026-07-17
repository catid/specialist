from types import SimpleNamespace

import pytest
import torch

import eggroll_es_audit_contract_v71 as audit
import eggroll_es_worker_lora_v41a as state_v41a
from eggroll_es_worker_lora_v66d import LoRAAdapterStateWorkerExtensionV66D
from eggroll_es_worker_lora_v71 import (
    LoRAAdapterStateWorkerExtensionV71,
    _runtime_key_v71,
    adapter_identity_no_clone_v71,
    fused_lora_readback_v71,
)


def _tiny_master(monkeypatch):
    tensors = {
        "base_model.model.model.layers.0.a.lora_A.weight": torch.arange(
            6, dtype=torch.float32
        ).reshape(2, 3),
        "base_model.model.model.layers.0.a.lora_B.weight": torch.arange(
            8, dtype=torch.float32
        ).reshape(4, 2),
    }
    monkeypatch.setattr(state_v41a, "EXPECTED_TENSOR_COUNT_V41A", 2)
    monkeypatch.setattr(state_v41a, "EXPECTED_MASTER_ELEMENTS_V41A", 14)
    return tensors


def test_fused_lora_equality_and_sha_use_one_readback_without_expected_concat():
    actual = {
        "a": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        "b": torch.arange(8, dtype=torch.bfloat16).reshape(4, 2),
    }
    expected = {key: tensor.clone() for key, tensor in actual.items()}
    calls = []

    def readback(tensor):
        calls.append((tensor.numel(), tensor.dtype, tensor.device.type))
        return tensor.detach().cpu().contiguous()

    receipt = fused_lora_readback_v71(
        actual, expected, readback_fn=readback
    )

    assert calls == [(14, torch.bfloat16, "cpu")]
    assert receipt["d2h_calls"] == 1
    assert receipt["d2h_bytes"] == 28
    assert receipt["exact_equal"] is True
    assert receipt["expected_value_concat_host_copy_bytes"] == 0
    assert set(receipt["sha256_by_key"]) == {"a", "b"}


def test_one_element_lora_mutation_is_rejected_by_the_same_fused_readback():
    actual = {"a": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)}
    expected = {"a": actual["a"].clone()}
    actual["a"].data[0, 0].add_(1)
    calls = []

    with pytest.raises(RuntimeError, match="equality check failed: a"):
        fused_lora_readback_v71(
            actual,
            expected,
            readback_fn=lambda tensor: calls.append(1) or tensor.cpu(),
        )
    assert calls == [1]


def test_no_clone_adapter_identity_and_master_cache_reject_object_drift(monkeypatch):
    master = _tiny_master(monkeypatch)
    identity = adapter_identity_no_clone_v71(master)
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    worker._v41_master = master
    worker._v41_current_identity = identity
    worker._v66_terminal_poison = None
    worker._v71_traffic = {}
    worker._poison_v66 = lambda phase, error: setattr(
        worker, "_v66_terminal_poison", {"phase": phase, "error": str(error)}
    )

    receipt = worker._rebind_master_cache_v71("test", identity["sha256"])
    assert receipt["validation_clone_bytes"] == 0
    assert worker._master_identity_v71("candidate_0")["cache"][
        "validation_clone_bytes"
    ] == 0

    worker._v41_master = {key: tensor.clone() for key, tensor in master.items()}
    with pytest.raises(RuntimeError, match="object/key mapping drifted"):
        worker._master_identity_v71("object_replacement")


def test_one_element_master_data_mutation_survives_cheap_check_but_fails_exact(monkeypatch):
    master = _tiny_master(monkeypatch)
    identity = adapter_identity_no_clone_v71(master)
    cache = audit.OwnedMasterIdentityCacheV71(master, identity=identity)
    target = next(iter(master.values()))
    version = target._version
    target.data.reshape(-1)[0].add_(1)

    assert target._version == version
    assert cache.cached_identity(master, "transition")["sha256"] == identity["sha256"]
    with pytest.raises(RuntimeError, match="exact content drifted"):
        cache.exact_audit(master, "commit")


def _runtime_audit_worker(view):
    item = {
        "runtime_module": "layer.q_proj",
        "side": "A",
        "slice_index": 0,
        "peft_key": "base_model.model.model.layers.0.q_proj.lora_A.weight",
    }
    key = _runtime_key_v71(item)
    views = {key: view}
    readback = fused_lora_readback_v71(views)
    records = [{
        **item,
        "dtype": str(view.dtype),
        "elements": int(view.numel()),
        "sha256": readback["sha256_by_key"][key],
    }]
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    worker._v71_runtime_views = views
    worker._v71_runtime_registry = audit.TensorInvariantRegistryV71(
        "runtime_lora_views", views
    )
    worker._v41_assignments = [item]
    worker._v41_active_materialization = {
        "runtime_values_sha256": state_v41a.canonical_sha256_v3(records)
    }
    worker._v71_traffic = {}
    worker._assert_runtime_links_v71 = lambda: None
    worker._v41_current_identity = {"sha256": "a" * 64}
    worker._v66_terminal_poison = None
    return worker


def test_adversarial_lora_content_mutation_exact_restores_and_rejects_reward():
    worker = _runtime_audit_worker(torch.arange(8, dtype=torch.bfloat16))
    restored = []
    worker._restore_exact_master_v66 = lambda phase: (
        restored.append(phase)
        or {"master_identity": {"sha256": "a" * 64}}
    )
    worker._poison_v66 = lambda phase, error: pytest.fail(
        f"unexpected poison: {phase}: {error}"
    )
    target = next(iter(worker._v71_runtime_views.values()))
    target.data[0].add_(1)

    with pytest.raises(RuntimeError, match="exact master restored and reward rejected"):
        worker._exact_lora_audit_v71("candidate_post_generation")
    assert restored == ["v71_candidate_post_generation_audit_repair"]


def test_unknown_partial_rpc_restores_or_terminally_poisons():
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    worker._v41_current_identity = {"sha256": "b" * 64}
    worker._v66_terminal_poison = None
    poisoned = []
    worker._poison_v66 = lambda phase, error: (
        poisoned.append((phase, str(error))),
        setattr(worker, "_v66_terminal_poison", {"phase": phase}),
    )
    worker._restore_exact_master_v66 = lambda phase: {
        "master_identity": {"sha256": "b" * 64}
    }

    with pytest.raises(RuntimeError, match="exact master restored and reward rejected"):
        worker._fail_closed_runtime_v71(
            "unknown_candidate_rpc", RuntimeError("partial write")
        )
    assert poisoned == []

    worker._v66_terminal_poison = None

    def failed_restore(_phase):
        raise RuntimeError("synthetic restore failure")

    worker._restore_exact_master_v66 = failed_restore
    with pytest.raises(RuntimeError, match="unverified; poisoned"):
        worker._fail_closed_runtime_v71(
            "unknown_candidate_rpc", RuntimeError("partial write")
        )
    assert worker._v66_terminal_poison is not None


def test_legacy_update_prepare_and_commit_cannot_bypass_acceptance_token():
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    with pytest.raises(RuntimeError, match="population acceptance"):
        worker.prepare_sharded_adapter_update_v41a()
    with pytest.raises(RuntimeError, match="update acceptance"):
        worker.commit_sharded_adapter_update_v41a()


def _tiny_transaction_worker(monkeypatch):
    old_master = _tiny_master(monkeypatch)
    candidate = {
        key: tensor.add(0.25).contiguous() for key, tensor in old_master.items()
    }
    old_identity = adapter_identity_no_clone_v71(old_master)
    candidate_identity = adapter_identity_no_clone_v71(candidate)
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    worker._v41_master = old_master
    worker._v41_current_identity = old_identity
    worker._v71_master_cache = audit.OwnedMasterIdentityCacheV71(
        old_master, identity=old_identity
    )
    worker._v71_update_acceptance_sha256 = "update-token"
    worker._v66_terminal_poison = None
    worker._v71_traffic = {}
    worker._v41_pending_update = {
        "phase": "executed",
        "manifest_sha256": "manifest",
        "manifest": {
            "update_sequence": 1,
            "plan_id": "plan",
        },
        "candidate_master": candidate,
        "candidate_identity": candidate_identity,
        "rollback_update_sequence": 0,
        "rollback_active_plan_id": None,
        "rollback_reference_fresh": True,
    }
    worker._v41_committed_rollback = None
    worker._v41_update_sequence = 0
    worker._v41_active_plan_id = None
    worker._v41_reference_fresh = True
    worker._require_not_poisoned_v66 = lambda: None
    worker._poison_v66 = lambda phase, error: setattr(
        worker, "_v66_terminal_poison", {"phase": phase, "error": str(error)}
    )
    worker._materialize_v41a = lambda tensors, phase: {
        "phase": phase,
        "sha256": adapter_identity_no_clone_v71(tensors)["sha256"],
    }
    worker._base_check_v41a = lambda phase: {"phase": phase, "unchanged": True}
    worker.inter_pg = SimpleNamespace(rank=0)
    return worker, old_master, old_identity, candidate, candidate_identity


def test_commit_and_final_keep_rollback_until_each_exact_boundary_passes(monkeypatch):
    worker, old_master, old_identity, candidate, candidate_identity = (
        _tiny_transaction_worker(monkeypatch)
    )
    boundaries = []
    worker.exact_boundary_audit_v71 = lambda boundary: (
        boundaries.append(boundary)
        or {"audit_sha256": f"{boundary}-audit"}
    )

    committed = worker.commit_sharded_adapter_update_v71(
        "update-token", "manifest", candidate_identity["sha256"]
    )

    assert boundaries == ["commit"]
    assert committed["committed"] is True
    assert committed["validation_clone_bytes"] == 0
    assert worker._v41_master is candidate
    assert worker._v41_committed_rollback["rollback_master"] is old_master
    assert worker._v41_committed_rollback["rollback_identity"] == old_identity

    finalized = worker.finalize_sharded_adapter_update_v71(
        "manifest", candidate_identity["sha256"]
    )
    assert boundaries == ["commit", "final"]
    assert finalized["finalized"] is True
    assert worker._v41_committed_rollback is None


def test_failed_final_boundary_retains_cross_rank_rollback(monkeypatch):
    worker, _old_master, _old_identity, _candidate, candidate_identity = (
        _tiny_transaction_worker(monkeypatch)
    )
    worker.exact_boundary_audit_v71 = lambda boundary: {
        "audit_sha256": f"{boundary}-audit"
    }
    worker.commit_sharded_adapter_update_v71(
        "update-token", "manifest", candidate_identity["sha256"]
    )
    rollback = worker._v41_committed_rollback

    def fail_final(boundary):
        assert boundary == "final"
        raise RuntimeError("synthetic final exact-audit failure")

    worker.exact_boundary_audit_v71 = fail_final
    with pytest.raises(RuntimeError, match="final exact-audit failure"):
        worker.finalize_sharded_adapter_update_v71(
            "manifest", candidate_identity["sha256"]
        )
    assert worker._v41_committed_rollback is rollback


def test_checkpoint_is_an_additional_exact_content_boundary():
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    calls = []
    worker._exact_base_audit_v71 = lambda boundary: (
        calls.append(("base", boundary)) or {"passed": True}
    )
    worker._exact_master_audit_v71 = lambda boundary: (
        calls.append(("master", boundary)) or {"passed": True}
    )
    worker._exact_lora_audit_v71 = lambda boundary: (
        calls.append(("lora", boundary)) or {"passed": True}
    )
    worker._v71_completed_boundaries = []

    receipt = worker.exact_boundary_audit_v71("checkpoint")

    assert receipt["passed"] is True
    assert calls == [
        ("base", "checkpoint"),
        ("master", "checkpoint"),
        ("lora", "checkpoint"),
    ]
    assert worker._v71_completed_boundaries == ["checkpoint"]


def test_unknown_update_abort_avoids_a_redundant_second_materialization():
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    master_sha = "a" * 64
    manifest_sha = "b" * 64
    identity = {"sha256": master_sha}
    worker._v41_pending_update = {"manifest_sha256": manifest_sha}
    worker._v41_committed_rollback = None
    worker._v66_terminal_poison = None
    worker._require_not_poisoned_v66 = lambda: None
    worker._require_installed_v41a = lambda: None
    worker._poison_v66 = lambda phase, error: pytest.fail(
        f"unexpected poison: {phase}: {error}"
    )
    calls = []
    worker.abort_sharded_adapter_update_v71 = lambda manifest, reason: (
        calls.append((manifest, reason))
        or {
            "identity": identity,
            "exact_restore": {
                "materialization": {"exact": True},
                "base_exact_audit": {"passed": True},
            },
        }
    )
    worker._restore_exact_master_v66 = lambda phase: pytest.fail(
        f"redundant restore called: {phase}"
    )

    receipt = worker.abort_mirrored_update_if_present_v66(
        master_sha, "unknown_execute_rpc", manifest_sha
    )

    assert calls == [(manifest_sha, "unknown_execute_rpc")]
    assert receipt["aborted"] is True
    assert receipt["redundant_second_materialization_avoided"] is True
    assert receipt["restored_identity"] == identity


def test_gpu_work_hooks_gate_unchanged_actor_receipt_with_v71_audits(monkeypatch):
    assignment = {
        "wave_index": 0,
        "engine_rank": 0,
        "direction_seed": 7,
        "sign": 1,
        "pair_id": "a" * 64,
        "evaluation_contract_sha256": "b" * 64,
    }
    actor_receipt = {
        "schema": "unchanged-v66d-receipt",
        "work_id": "c" * 64,
    }
    monkeypatch.setattr(
        LoRAAdapterStateWorkerExtensionV66D,
        "begin_actor_gpu_work_v66d",
        lambda self, observed: actor_receipt,
    )
    monkeypatch.setattr(
        LoRAAdapterStateWorkerExtensionV66D,
        "end_actor_gpu_work_v66d",
        lambda self, observed, cardinality: actor_receipt,
    )
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    transitions = []
    post = []
    worker._v71_transition_audits = []
    worker._cheap_transition_audit_v71 = lambda phase: (
        transitions.append(phase) or {"audit_sha256": "d" * 64}
    )
    worker.post_generation_lora_audit_v71 = lambda candidate_id: (
        post.append(candidate_id) or {"audit_sha256": "e" * 64}
    )

    begin = worker.begin_actor_gpu_work_v66d(assignment)
    end = worker.end_actor_gpu_work_v66d(assignment, {"samples": 1})

    assert begin is actor_receipt
    assert end is actor_receipt
    assert transitions[0].startswith("pre_generation_")
    assert worker._v71_transition_audits == ["d" * 64]
    assert post == ["c" * 64]


def test_candidate_audit_matrix_is_exposed_only_after_exact_restore():
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    worker._v71_candidate_audits = ["a" * 64, "b" * 64]
    worker._v71_transition_audits = ["c" * 64, "d" * 64]
    worker._v41_active_perturbation = {"phase": "candidate_active"}
    worker._v66_candidate_transaction = {"phase": "candidate_active"}
    worker._v66d_active_gpu_work = None
    with pytest.raises(RuntimeError, match="not quiescent"):
        worker.candidate_audit_matrix_v71()

    worker._v41_active_perturbation = None
    worker._v66_candidate_transaction = None
    receipt = worker.candidate_audit_matrix_v71()
    assert receipt["candidate_count"] == 2
    assert receipt["all_rewards_provisional"] is True
    assert receipt["candidate_audit_sha256"] == ["a" * 64, "b" * 64]


def test_exact_base_corruption_terminally_poisons_before_acceptance():
    base = {"base": torch.arange(8, dtype=torch.bfloat16)}
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    worker._v71_base_tensors = base
    worker._v71_base_registry = audit.TensorInvariantRegistryV71("base", base)
    worker._v71_traffic = {}
    worker._v66_terminal_poison = None
    worker._assert_base_links_v71 = lambda: None
    worker._poison_v66 = lambda phase, error: setattr(
        worker, "_v66_terminal_poison", {"phase": phase, "error": str(error)}
    )
    base["base"].data[0].add_(1)

    with pytest.raises(RuntimeError, match="exact content drifted"):
        worker._exact_base_audit_v71("population_reward_acceptance")
    assert worker._v66_terminal_poison["phase"] \
        == "v71_base_population_reward_acceptance"


def test_exact_master_corruption_terminally_poisons_before_update():
    master = {"master": torch.arange(8, dtype=torch.float32)}
    cache = audit.OwnedMasterIdentityCacheV71(master)
    worker = object.__new__(LoRAAdapterStateWorkerExtensionV71)
    worker._v41_master = master
    worker._v41_current_identity = {"sha256": cache.sha256}
    worker._v71_master_cache = cache
    worker._v66_terminal_poison = None
    worker._poison_v66 = lambda phase, error: setattr(
        worker, "_v66_terminal_poison", {"phase": phase, "error": str(error)}
    )
    master["master"].data[0].add_(1)

    with pytest.raises(RuntimeError, match="exact content drifted"):
        worker._exact_master_audit_v71("update_acceptance")
    assert worker._v66_terminal_poison["phase"] \
        == "v71_master_update_acceptance"
