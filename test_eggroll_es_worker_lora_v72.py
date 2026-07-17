from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

import eggroll_es_host_state_contract_v72 as contract
import eggroll_es_worker_lora_v41a as state_v41a
import eggroll_es_worker_lora_v72 as worker_v72
from eggroll_es_worker_lora_v71 import adapter_identity_no_clone_v71
from eggroll_es_worker_v3 import coefficient_sha256_v3
from test_eggroll_es_worker_lora_v41a import (
    CONFIG,
    WEIGHTS,
    FakePG,
    fake_manager,
    full_reductions,
    source_master,
)


def make_worker_v72(monkeypatch, rank=0):
    monkeypatch.setattr(state_v41a, "EXPECTED_BASE_ELEMENTS_V41A", 23)
    monkeypatch.setattr(state_v41a, "EXPECTED_BASE_BYTES_V41A", 46)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    manager = fake_manager(source_master())
    value = object.__new__(worker_v72.LoRAAdapterStateWorkerExtensionV72)
    value.device = torch.device("cpu")
    value.model_runner = SimpleNamespace(
        lora_manager=SimpleNamespace(_adapter_manager=manager)
    )
    value.inter_pg = FakePG(rank=rank)
    installed = value.install_adapter_state_v41a(
        WEIGHTS,
        CONFIG,
        state_v41a.file_sha256_v41a(WEIGHTS),
        state_v41a.file_sha256_v41a(CONFIG),
    )
    return value, manager, installed


def _prepare(value, seeds, coefficients):
    value._v71_population_acceptance_sha256 = "population-token"
    value.exact_boundary_audit_v71 = lambda boundary: {
        "boundary": boundary,
        "audit_sha256": f"{boundary}-audit",
    }
    return value.prepare_sharded_adapter_update_v72(
        "population-token",
        seeds,
        coefficients,
        coefficient_sha256_v3(seeds, coefficients),
        4,
        4,
        0.01,
        "v72-plan",
        value._v41_current_identity["sha256"],
        value._v41_reference_generation,
    )


def test_install_and_reference_capture_keep_one_owned_host_bank(monkeypatch):
    value, _manager, installed = make_worker_v72(monkeypatch)

    assert installed["install_full_state_clone_passes"] == 1
    assert installed["resident_reference_tensor_bank"] is False
    assert not hasattr(value, "_v41_reference")
    residency = value.host_state_residency_v72()
    assert residency["phase"] == "quiescent_one_master"
    assert residency["unique_owned_bank_count"] == 1
    assert residency["unique_owned_tensor_bytes"] == contract.MASTER_BYTES_V72

    captured = value.capture_adapter_reference_v41a()
    assert captured["reference_tensor_bank_allocated"] is False
    assert captured["full_state_clone_bytes"] == 0
    assert not hasattr(value, "_v41_reference")
    assert value.host_state_residency_v72()["unique_owned_bank_count"] == 1


def test_prepare_aliases_master_execute_owns_only_one_candidate_and_moves_it(
    monkeypatch,
):
    value, _manager, _installed = make_worker_v72(monkeypatch)
    seeds = [11, 22, 33, 44]
    coefficients = [1.0, -0.5, 0.25, -0.125]
    origin_master = value._v41_master
    origin_identity = value._v41_current_identity
    reductions = full_reductions(origin_master, seeds, coefficients)
    value.inter_pg = FakePG(
        rank=0,
        reduced=reductions,
    )

    prepared = _prepare(value, seeds, coefficients)
    pending = value._v41_pending_update
    assert pending["rollback_master"] is origin_master
    assert pending["rollback_lease_v72"].tensors is origin_master
    assert prepared["rollback_clone_bytes"] == 0
    assert value.host_state_residency_v72()["unique_owned_bank_count"] == 1

    executed = value.execute_sharded_adapter_update_v41a(
        prepared["manifest_sha256"]
    )
    candidate = value._v41_pending_update["candidate_master"]
    expected_candidate = {
        key: master.add(reduced.clone().mul_(0.01 / 4)).contiguous()
        for (key, master), reduced in zip(
            origin_master.items(), reductions, strict=True
        )
    }
    assert candidate is not origin_master
    assert executed["candidate_identity"] \
        == adapter_identity_no_clone_v71(expected_candidate)
    assert executed["full_state_validation_clone_bytes"] == 0
    assert value.host_state_residency_v72()["unique_owned_bank_count"] == 2

    committed = value.commit_sharded_adapter_update_v72(
        prepared["update_acceptance_sha256"],
        prepared["manifest_sha256"],
        executed["candidate_identity"]["sha256"],
    )
    assert value._v41_master is candidate
    assert value._v41_committed_rollback["rollback_master"] is origin_master
    assert committed["full_state_clone_bytes"] == 0
    assert committed["host_state_residency"]["unique_owned_bank_count"] == 2

    finalized = value.finalize_sharded_adapter_update_v72(
        prepared["manifest_sha256"],
        executed["candidate_identity"]["sha256"],
    )
    assert finalized["host_state_residency"]["unique_owned_bank_count"] == 1
    assert value._v41_master is candidate
    assert value._v41_current_identity != origin_identity


def test_mutated_rollback_lease_poisons_before_update_execution(monkeypatch):
    value, _manager, _installed = make_worker_v72(monkeypatch)
    seeds = [1, 2, 3, 4]
    coefficients = [1.0, 0.5, -0.25, 0.125]
    prepared = _prepare(value, seeds, coefficients)
    target = next(iter(value._v41_master.values()))
    version = target._version
    target.data.reshape(-1)[0].add_(1)

    assert target._version == version
    with pytest.raises(RuntimeError, match="exact content drifted"):
        value.execute_sharded_adapter_update_v41a(prepared["manifest_sha256"])
    assert value._v66_terminal_poison["phase"] == "v72_update_execute_master"
    assert "candidate_master" not in value._v41_pending_update


def test_partial_candidate_generation_is_rejected_and_old_master_restored(
    monkeypatch,
):
    value, _manager, _installed = make_worker_v72(monkeypatch)
    seeds = [5, 6, 7, 8]
    coefficients = [0.5, -0.25, 0.125, -0.0625]
    prepared = _prepare(value, seeds, coefficients)
    old_master = value._v41_master
    candidate = {
        key: tensor.add(0.1).contiguous()
        for key, tensor in old_master.items()
    }
    candidate_identity = adapter_identity_no_clone_v71(candidate)
    value._v41_pending_update.update({
        "phase": "executed",
        "candidate_master": candidate,
        "candidate_identity": candidate_identity,
    })
    next(iter(candidate.values())).data.reshape(-1)[0].add_(1)

    with pytest.raises(RuntimeError, match="cross-rank final identity changed"):
        value.commit_sharded_adapter_update_v72(
            prepared["update_acceptance_sha256"],
            prepared["manifest_sha256"],
            candidate_identity["sha256"],
        )
    assert value._v41_master is old_master
    assert value._v41_pending_update is None
    assert value._v41_committed_rollback is None
    assert getattr(value, "_v66_terminal_poison", None) is None


def test_incomplete_candidate_bank_is_never_adopted(monkeypatch):
    value, _manager, _installed = make_worker_v72(monkeypatch)
    seeds = [9, 10, 11, 12]
    coefficients = [0.5, -0.25, 0.125, -0.0625]
    prepared = _prepare(value, seeds, coefficients)
    old_master = value._v41_master
    candidate = {
        key: tensor.add(0.1).contiguous()
        for key, tensor in old_master.items()
    }
    candidate.pop(next(iter(candidate)))
    value._v41_pending_update.update({
        "phase": "executed",
        "candidate_master": candidate,
        "candidate_identity": {"sha256": "c" * 64},
    })

    with pytest.raises(RuntimeError, match="tensor count changed"):
        value.commit_sharded_adapter_update_v72(
            prepared["update_acceptance_sha256"],
            prepared["manifest_sha256"],
            "c" * 64,
        )
    assert value._v41_master is old_master
    assert value._v41_pending_update is None
    assert value._v41_committed_rollback is None
    assert getattr(value, "_v66_terminal_poison", None) is None


def test_atomic_streamed_snapshot_has_no_full_state_clone(monkeypatch, tmp_path):
    value, _manager, _installed = make_worker_v72(monkeypatch, rank=0)
    output = tmp_path / "snapshot"
    snapshot = value.save_adapter_snapshot_v41a(
        output, value._v41_current_identity["sha256"]
    )

    assert snapshot["written"] is True
    assert snapshot["atomic_directory_publication"] is True
    assert snapshot["full_state_clone_bytes"] == 0
    assert snapshot["streaming_readback"]["one_tensor_at_a_time"] is True
    assert snapshot["streaming_readback"]["full_state_clone_bytes"] == 0
    assert snapshot["streaming_readback"][
        "max_resident_readback_tensor_bytes"
    ] == contract.MAX_MASTER_TENSOR_BYTES_V72
    assert snapshot["readback_identity"] == value._v41_current_identity
    assert not list(tmp_path.glob(".snapshot.tmp-v72-*"))
    with safe_open(
        output / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        assert len(handle.keys()) == 70


def test_partial_snapshot_write_never_creates_final_directory(monkeypatch, tmp_path):
    value, _manager, _installed = make_worker_v72(monkeypatch, rank=0)

    def partial_write(_tensors, path, metadata):
        del metadata
        Path(path).write_bytes(b"partial")
        raise RuntimeError("synthetic partial checkpoint write")

    monkeypatch.setattr(state_v41a, "save_file", partial_write)
    output = tmp_path / "snapshot"
    with pytest.raises(RuntimeError, match="partial checkpoint write"):
        value.save_adapter_snapshot_v41a(
            output, value._v41_current_identity["sha256"]
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".snapshot.tmp-v72-*"))


def test_master_mutation_during_snapshot_cannot_publish(monkeypatch, tmp_path):
    value, _manager, _installed = make_worker_v72(monkeypatch, rank=0)
    real_save = state_v41a.save_file

    def mutate_after_write(tensors, path, metadata):
        real_save(tensors, path, metadata=metadata)
        next(iter(tensors.values())).data.reshape(-1)[0].add_(1)

    monkeypatch.setattr(state_v41a, "save_file", mutate_after_write)
    output = tmp_path / "snapshot"
    with pytest.raises(RuntimeError, match="exact content drifted"):
        value.save_adapter_snapshot_v41a(
            output, value._v41_current_identity["sha256"]
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".snapshot.tmp-v72-*"))
    assert value._v66_terminal_poison["phase"] \
        == "v71_master_checkpoint_pre_publish"
