from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

import eggroll_es_worker_lora_topology_v40a as topology
import eggroll_es_worker_lora_v41a as worker
from eggroll_es_worker_v3 import coefficient_sha256_v3


ADAPTER = (
    worker.Path(__file__).resolve().parent
    / "experiments/sft_controls/v37a_equal_unit_fold3_v412/"
    "middle_late_r32_seed17/final"
)
WEIGHTS = ADAPTER / "adapter_model.safetensors"
CONFIG = ADAPTER / "adapter_config.json"


class FakeWrapper:
    def __init__(self, a_shapes, b_shapes):
        self.lora_a_stacked = tuple(
            torch.zeros((1, 1, *shape), dtype=torch.bfloat16)
            for shape in a_shapes
        )
        self.lora_b_stacked = tuple(
            torch.zeros((1, 1, *shape), dtype=torch.bfloat16)
            for shape in b_shapes
        )
        self.output_slices = tuple(shape[0] for shape in b_shapes)
        self.base_layer = SimpleNamespace(weight=torch.zeros(1, dtype=torch.bfloat16))

    def reset_lora(self, index):
        for tensor in self.lora_a_stacked + self.lora_b_stacked:
            tensor[index].zero_()


class FakePG:
    def __init__(self, rank=0, reduced=None, fail_at=None):
        self.rank = rank
        self.world_size = 4
        self.available = True
        self.disabled = False
        self.reduced = list(reduced or [])
        self.fail_at = fail_at
        self.calls = 0

    def all_reduce(self, tensor, out_tensor, stream=None):
        del tensor, stream
        if self.fail_at is not None and self.calls == self.fail_at:
            raise RuntimeError("synthetic collective failure")
        value = self.reduced[self.calls]
        self.calls += 1
        out_tensor.copy_(value)
        return out_tensor


def source_master():
    with safe_open(WEIGHTS, framework="pt", device="cpu") as handle:
        return {key: handle.get_tensor(key).float().contiguous()
                for key in handle.keys()}


def fake_manager(master):
    groups = {}
    for key, tensor in master.items():
        logical, side = topology._source_parts(key)
        target, slices = topology._runtime_target(logical)
        group = groups.setdefault(target, {"a": {}, "b": {}})
        if side == "A":
            for index in slices:
                group["a"][index] = tuple(tensor.shape)
        elif len(slices) == 1:
            group["b"][slices[0]] = tuple(tensor.shape)
        else:
            # Qwen3.5 GDN's packed in_proj_qkv is q:k:v = 2:1:1.
            rows = tensor.shape[0]
            sizes = (rows // 2, rows // 4, rows // 4)
            assert sum(sizes) == rows and len(sizes) == len(slices)
            for index, size in zip(slices, sizes, strict=True):
                group["b"][index] = (size, tensor.shape[1])
    modules = {}
    for target, group in groups.items():
        count = max(set(group["a"]) | set(group["b"])) + 1
        assert set(group["a"]) == set(range(count))
        assert set(group["b"]) == set(range(count))
        modules[f"language_model.{target}"] = FakeWrapper(
            [group["a"][index] for index in range(count)],
            [group["b"][index] for index in range(count)],
        )
    assert len(modules) == 23
    return SimpleNamespace(
        lora_index_to_id=[1], modules=modules,
        packed_modules_mapping={
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
            "in_proj_ba": ["in_proj_b", "in_proj_a"],
        },
    )


def make_worker(monkeypatch, rank=0):
    monkeypatch.setattr(worker, "EXPECTED_BASE_ELEMENTS_V41A", 23)
    monkeypatch.setattr(worker, "EXPECTED_BASE_BYTES_V41A", 46)
    master = source_master()
    manager = fake_manager(master)
    value = object.__new__(worker.LoRAAdapterStateWorkerExtensionV41A)
    value.device = torch.device("cpu")
    value.model_runner = SimpleNamespace(
        lora_manager=SimpleNamespace(_adapter_manager=manager)
    )
    value.inter_pg = FakePG(rank=rank)
    installed = value.install_adapter_state_v41a(
        WEIGHTS, CONFIG, worker.file_sha256_v41a(WEIGHTS),
        worker.file_sha256_v41a(CONFIG),
    )
    return value, manager, installed


def test_install_materializes_exact_70_to_82_and_b_scale(monkeypatch):
    value, manager, installed = make_worker(monkeypatch)
    assert installed["canonical_identity"]["tensor_count"] == 70
    assert installed["canonical_identity"]["elements"] == 4_528_128
    assert installed["assignment_count"] == 82
    assert installed["materialization"]["runtime_module_count"] == 23
    assert installed["materialization"]["runtime_view_count"] == 82
    assert installed["materialization"]["runtime_elements"] == 4_921_344
    assert installed["materialization"]["b_scale"] == 2.0
    assert installed["materialization"]["unique_parent_storage_count"] == 82
    assert installed["materialization"]["runtime_views_share_no_parent_storage"]
    assert installed["materialization"]["slot_views_alias_parent_buffers"]
    assert installed["base_identity"] == {
        "phase": "install", "unchanged": True, "tensor_count": 23,
        "elements": 23, "bytes": 46,
        "inventory_sha256": installed["base_identity"]["inventory_sha256"],
    }

    b_key = "base_model.model.model.layers.23.self_attn.o_proj.lora_B.weight"
    logical, _side = topology._source_parts(b_key)
    target, _slices = topology._runtime_target(logical)
    name, module = topology._suffix_match(manager.modules, target)
    assert name == "language_model.model.layers.23.self_attn.o_proj"
    expected_b = (value._v41_master[b_key] * 2.0).to(torch.bfloat16)
    assert torch.equal(module.lora_b_stacked[0][0, 0], expected_b)

    a_key = "base_model.model.model.layers.20.linear_attn.in_proj_qkv.lora_A.weight"
    logical, _side = topology._source_parts(a_key)
    target, slices = topology._runtime_target(logical)
    _name, module = topology._suffix_match(manager.modules, target)
    expected_a = value._v41_master[a_key].to(torch.bfloat16)
    assert slices == (0, 1, 2)
    assert all(torch.equal(module.lora_a_stacked[index][0, 0], expected_a)
               for index in slices)
    assert len({module.lora_a_stacked[index].untyped_storage().data_ptr()
                for index in slices}) == 3


def test_antithetic_is_deterministic_master_preserving_and_exactly_restored(monkeypatch):
    value, _manager, _installed = make_worker(monkeypatch)
    origin = worker.adapter_identity_v41a(value._v41_master)
    plus = value.materialize_antithetic_adapter_v41a(
        1234, 0.0003, 1, origin["sha256"],
    )
    first_runtime = plus["materialization"]["runtime_values_sha256"]
    assert plus["master_unchanged"] is True
    assert worker.adapter_identity_v41a(value._v41_master) == origin
    restored = value.restore_adapter_master_v41a()
    assert restored["restored_identity"] == origin
    assert restored["algebraic_bf16_restore_used"] is False
    assert restored["materialization"]["runtime_values_sha256"] != first_runtime
    repeated = value.materialize_antithetic_adapter_v41a(
        1234, 0.0003, 1, origin["sha256"],
    )
    assert repeated["candidate_identity"] == plus["candidate_identity"]
    assert repeated["materialization"]["runtime_values_sha256"] == first_runtime
    value.restore_adapter_master_v41a()


def full_reductions(master, seeds, coefficients):
    values = []
    for key, tensor in master.items():
        accumulator = torch.zeros_like(tensor)
        for seed, coefficient in zip(seeds, coefficients, strict=True):
            accumulator.add_(
                worker.noise_like_v41a(tensor, key, seed, "cpu"),
                alpha=float(coefficient),
            )
        values.append(accumulator)
    return values


def test_four_rank_sharded_fp32_update_commit_and_abort(monkeypatch):
    value, _manager, _installed = make_worker(monkeypatch)
    seeds = [11, 22, 33, 44]
    coefficients = [1.0, -0.5, 0.25, -0.125]
    coefficient_sha = coefficient_sha256_v3(seeds, coefficients)
    origin = worker.adapter_identity_v41a(value._v41_master)
    reductions = full_reductions(value._v41_master, seeds, coefficients)
    value.inter_pg = FakePG(rank=0, reduced=reductions)
    prepared = value.prepare_sharded_adapter_update_v41a(
        seeds, coefficients, coefficient_sha, 4, 4, 0.01, "plan-a",
        origin["sha256"], value._v41_reference_generation,
    )
    assert prepared["shard_indices"] == [0]
    assert prepared["shard_seeds"] == [11]
    executed = value.execute_sharded_adapter_update_v41a(
        prepared["manifest_sha256"]
    )
    assert executed["collective_dtype"] == "torch.float32"
    assert executed["reduced_elements"] == 4_528_128
    assert executed["master_committed"] is False
    assert worker.adapter_identity_v41a(value._v41_master) == origin
    assert executed["candidate_identity"] != origin
    committed = value.commit_sharded_adapter_update_v41a(
        prepared["manifest_sha256"], executed["candidate_identity"]["sha256"],
    )
    assert committed["committed"] is True
    assert committed["reference_fresh_for_population"] is False
    assert committed["requires_cross_rank_finalize"] is True
    assert committed["final_identity"] == executed["candidate_identity"]
    assert committed["base_identity"]["unchanged"] is True
    finalized = value.finalize_sharded_adapter_update_v41a(
        prepared["manifest_sha256"], executed["candidate_identity"]["sha256"],
    )
    assert finalized["finalized"] is True

    value.capture_adapter_reference_v41a()
    committed_origin = worker.adapter_identity_v41a(value._v41_master)
    value.inter_pg = FakePG(rank=0, reduced=full_reductions(
        value._v41_master, seeds, coefficients,
    ))
    prepared = value.prepare_sharded_adapter_update_v41a(
        seeds, coefficients, coefficient_sha, 4, 4, 0.01, "plan-b",
        committed_origin["sha256"], value._v41_reference_generation,
    )
    value.execute_sharded_adapter_update_v41a(prepared["manifest_sha256"])
    aborted = value.abort_sharded_adapter_update_v41a(prepared["manifest_sha256"])
    assert aborted["rolled_back"] is True
    assert aborted["identity"] == committed_origin
    assert worker.adapter_identity_v41a(value._v41_master) == committed_origin


def test_partial_cross_rank_commit_can_still_rollback(monkeypatch):
    value, _manager, _installed = make_worker(monkeypatch)
    seeds = [5, 6, 7, 8]
    coefficients = [0.5, -0.25, 0.125, -0.0625]
    origin = worker.adapter_identity_v41a(value._v41_master)
    value.inter_pg = FakePG(
        rank=0, reduced=full_reductions(value._v41_master, seeds, coefficients),
    )
    prepared = value.prepare_sharded_adapter_update_v41a(
        seeds, coefficients, coefficient_sha256_v3(seeds, coefficients),
        4, 4, 0.01, "partial", origin["sha256"],
        value._v41_reference_generation,
    )
    executed = value.execute_sharded_adapter_update_v41a(prepared["manifest_sha256"])
    value.commit_sharded_adapter_update_v41a(
        prepared["manifest_sha256"], executed["candidate_identity"]["sha256"],
    )
    rollback = value.abort_sharded_adapter_update_v41a(
        prepared["manifest_sha256"]
    )
    assert rollback["rolled_back"] is True
    assert rollback["identity"] == origin
    assert rollback["reference_fresh"] is True
    assert worker.adapter_identity_v41a(value._v41_master) == origin


def test_collective_failure_rolls_back_exactly(monkeypatch):
    value, _manager, _installed = make_worker(monkeypatch)
    seeds = [1, 2, 3, 4]
    coefficients = [1.0, 1.0, 1.0, 1.0]
    origin = worker.adapter_identity_v41a(value._v41_master)
    reductions = full_reductions(value._v41_master, seeds, coefficients)
    value.inter_pg = FakePG(rank=0, reduced=reductions, fail_at=2)
    prepared = value.prepare_sharded_adapter_update_v41a(
        seeds, coefficients, coefficient_sha256_v3(seeds, coefficients),
        4, 4, 0.01, "failure", origin["sha256"],
        value._v41_reference_generation,
    )
    with pytest.raises(RuntimeError, match="synthetic collective failure"):
        value.execute_sharded_adapter_update_v41a(prepared["manifest_sha256"])
    assert value._v41_pending_update is None
    assert worker.adapter_identity_v41a(value._v41_master) == origin
    certificate = value.adapter_state_certificate_v41a()
    assert certificate["current_identity"] == origin
    assert certificate["base_identity"]["unchanged"] is True


def test_rank_zero_snapshot_is_standard_unscaled_fp32_peft_and_readbacks(
    monkeypatch, tmp_path,
):
    value, _manager, _installed = make_worker(monkeypatch, rank=0)
    identity = worker.adapter_identity_v41a(value._v41_master)
    snapshot = value.save_adapter_snapshot_v41a(
        tmp_path / "snapshot", identity["sha256"],
    )
    assert snapshot["written"] is True
    assert snapshot["readback_verified"] is True
    assert snapshot["readback_identity"] == identity
    assert snapshot["original_canonical_key_namespace"] is True
    assert snapshot["unscaled_fp32_master_persisted"] is True
    assert worker.file_sha256_v41a(snapshot["config_path"]) == worker.file_sha256_v41a(CONFIG)
    with safe_open(snapshot["weights_path"], framework="pt", device="cpu") as handle:
        assert len(handle.keys()) == 70
        assert all(key.startswith("base_model.model.model.layers.") for key in handle.keys())
        assert all(handle.get_tensor(key).dtype == torch.float32 for key in handle.keys())


def test_zero_zero_lora_antithetic_direction_is_degenerate():
    generator = torch.Generator().manual_seed(7)
    a_noise = torch.randn((3, 5), generator=generator)
    b_noise = torch.randn((4, 3), generator=generator)
    result = worker.zero_zero_antithetic_degeneracy_v41a(
        a_noise, b_noise, sigma=0.3,
    )
    assert result["passed"] is True
    assert result["plus_equals_minus"] is True
    assert result["central_nonzero_elements"] == 0
