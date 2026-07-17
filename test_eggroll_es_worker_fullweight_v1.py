import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

import eggroll_es_worker_fullweight_v1 as subject
from train_eggroll_es_specialist import (
    validate_full_weight_install_receipts,
)


class TinyModel(torch.nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.full(
            (4,), value, dtype=torch.bfloat16,
        ))
        self.omega = torch.nn.Parameter(torch.full(
            (4,), value + 0.25, dtype=torch.bfloat16,
        ))


class AliasedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
        self.primary = torch.nn.Module()
        self.alternate = torch.nn.Module()
        self.primary.register_parameter("weight", parameter)
        self.alternate.register_parameter("weight", parameter)


class FakeCommunicator:
    rank = 0
    world_size = 1
    available = True
    disabled = False

    def __init__(self):
        self.broadcasts = 0

    def broadcast(self, tensor, src, stream):
        assert src == 0
        assert stream is None
        self.broadcasts += 1
        return tensor

    def all_reduce(self, tensor, out_tensor, stream):
        assert out_tensor is tensor
        assert stream is None
        return tensor


class ThreadedCollectiveBus:
    def __init__(self, world_size):
        self.world_size = world_size
        self.condition = threading.Condition()
        self.entries = {}

    def run(self, operation, index, rank, tensor, src=None):
        with self.condition:
            entry = self.entries.setdefault(index, {
                "operation": operation,
                "src": src,
                "inputs": {},
                "result": None,
                "readers": 0,
            })
            assert entry["operation"] == operation
            assert entry["src"] == src
            entry["inputs"][rank] = tensor.detach().clone()
            if len(entry["inputs"]) == self.world_size:
                if operation == "broadcast":
                    entry["result"] = entry["inputs"][src].clone()
                else:
                    result = entry["inputs"][0].clone()
                    for other_rank in range(1, self.world_size):
                        result.add_(entry["inputs"][other_rank])
                    entry["result"] = result
                self.condition.notify_all()
            assert self.condition.wait_for(
                lambda: entry["result"] is not None, timeout=5.0,
            )
            result = entry["result"].clone()
            entry["readers"] += 1
            if entry["readers"] == self.world_size:
                del self.entries[index]
            return result


class ThreadedCommunicator:
    available = True
    disabled = False

    def __init__(self, bus, rank):
        self.bus = bus
        self.rank = rank
        self.world_size = bus.world_size
        self.index = 0

    def broadcast(self, tensor, src, stream):
        assert stream is None
        index = self.index
        self.index += 1
        return self.bus.run("broadcast", index, self.rank, tensor, src=src)

    def all_reduce(self, tensor, out_tensor, stream):
        assert out_tensor is tensor
        assert stream is None
        index = self.index
        self.index += 1
        return self.bus.run("all_reduce", index, self.rank, tensor)


def make_worker(value=1.0):
    worker = object.__new__(subject.CanonicalFullWeightWorkerExtensionV1)
    worker.model_runner = SimpleNamespace(model=TinyModel(value))
    worker.inter_pg = FakeCommunicator()
    worker.world_size = 1
    receipt = worker.install_full_weight_master_v1()
    return worker, receipt


def make_threaded_workers(count=4, value=1.0):
    bus = ThreadedCollectiveBus(count)
    workers = []
    receipts = []
    for rank in range(count):
        worker = object.__new__(subject.CanonicalFullWeightWorkerExtensionV1)
        worker.model_runner = SimpleNamespace(model=TinyModel(value))
        worker.inter_pg = ThreadedCommunicator(bus, rank)
        worker.world_size = 1
        receipts.append(worker.install_full_weight_master_v1())
        workers.append(worker)
    return workers, receipts


def runtime_values(worker):
    return {
        name: parameter.detach().clone()
        for name, parameter in worker.model_runner.model.named_parameters()
    }


def master_values(worker):
    return {
        name: tensor.detach().clone()
        for name, tensor in worker._fw_v1_master.items()
    }


def test_install_receipts_require_tp1_replica_consensus():
    worker, receipt = make_worker()
    receipt = copy.deepcopy(receipt)
    receipt["communicator"] = {
        "rank": 0, "world_size": 2, "tp_world_size": 1,
    }
    other = copy.deepcopy(receipt)
    other["communicator"]["rank"] = 1
    assert validate_full_weight_install_receipts(
        [[receipt], [other]], 2,
    ) == [receipt, other]
    other["master_identity"]["sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="master_identity"):
        validate_full_weight_install_receipts([[receipt], [other]], 2)
    with pytest.raises(RuntimeError, match="exactly one TP worker"):
        validate_full_weight_install_receipts([[receipt, receipt]], 1)
    del worker


def test_uneven_seed_shards_cover_population_once_without_overlap():
    seeds = list(range(30))
    coefficients = [float(index) for index in seeds]
    shards = [
        subject.seed_shard_v1(seeds, coefficients, rank, 4)
        for rank in range(4)
    ]
    assert [len(shard["indices"]) for shard in shards] == [8, 8, 7, 7]
    indices = [index for shard in shards for index in shard["indices"]]
    assert sorted(indices) == list(range(30))
    assert len(set(indices)) == 30
    for shard in shards:
        assert shard["seeds"] == [seeds[index] for index in shard["indices"]]
        assert shard["coefficients"] == [
            coefficients[index] for index in shard["indices"]
        ]


def test_four_rank_sharded_update_uses_disjoint_seeds_and_reaches_consensus():
    workers, receipts = make_threaded_workers()
    validate_full_weight_install_receipts(
        [[receipt] for receipt in receipts], 4,
    )
    seeds = [11, 12, 13, 14, 15]
    coefficients = [1.0, -0.5, 0.25, 0.75, -1.25]
    prepared = workers[0].update_weights_from_seeds(
        seeds, coefficients, 0.01, len(seeds),
    )
    assert prepared["schema"] == "eggroll-es-full-weight-update-prepared-v1"
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda worker: worker.broadcast_all_weights(0), workers,
        ))
    all_indices = [
        index for result in results for index in result["shard_indices"]
    ]
    assert sorted(all_indices) == list(range(len(seeds)))
    assert len(set(all_indices)) == len(seeds)
    assert all(result["replica_consensus"]["passed"] for result in results)
    reference_master = master_values(workers[0])
    reference_runtime = runtime_values(workers[0])
    for worker in workers[1:]:
        assert all(
            torch.equal(worker._fw_v1_master[name], value)
            for name, value in reference_master.items()
        )
        assert all(
            torch.equal(runtime_values(worker)[name], value)
            for name, value in reference_runtime.items()
        )


def test_one_seed_stream_advances_across_parameters_and_replays_exactly():
    worker, _receipt = make_worker()
    before = runtime_values(worker)
    master_before = master_values(worker)
    generator = subject._new_generator_v1(torch.device("cpu"), 123)
    expected_noise = {
        name: subject.draw_noise_v1(parameter, generator)
        for name, parameter in sorted(
            worker.model_runner.model.named_parameters(),
        )
    }
    assert not torch.equal(expected_noise["alpha"], expected_noise["omega"])

    first = worker.perturb_self_weights(123, 0.125, False)
    first_candidate = runtime_values(worker)
    for name, parameter in worker.model_runner.model.named_parameters():
        expected = master_before[name].add(
            expected_noise[name], alpha=0.125,
        ).to(dtype=parameter.dtype)
        assert torch.equal(first_candidate[name], expected)
    assert worker.restore_self_weights(123, 0.125)["restored"] is True
    assert runtime_values(worker).keys() == before.keys()
    assert all(
        torch.equal(runtime_values(worker)[name], value)
        for name, value in before.items()
    )

    second = worker.perturb_self_weights(123, 0.125, False)
    assert second["derived_candidate_sha256"] == first["derived_candidate_sha256"]
    replay = runtime_values(worker)
    assert all(torch.equal(replay[name], first_candidate[name]) for name in replay)
    worker.restore_self_weights(123, 0.125)


@pytest.mark.parametrize("negate", [False, True])
def test_both_signs_restore_bit_identically_without_algebraic_subtraction(negate):
    worker, receipt = make_worker(value=1.0)
    before = runtime_values(worker)
    master_sha = receipt["master_identity"]["sha256"]
    candidate = worker.perturb_self_weights(7, 0.03125, negate)
    assert any(
        not torch.equal(value, before[name])
        for name, value in runtime_values(worker).items()
    )
    restored = worker.restore_or_readback_full_weight_v1(
        candidate["transaction_id"], master_sha,
    )
    assert restored["algebraic_native_restore_used"] is False
    after = runtime_values(worker)
    assert all(torch.equal(after[name], before[name]) for name in before)


def test_partial_candidate_write_is_fully_repaired_and_restore_is_idempotent():
    worker, receipt = make_worker()
    before = runtime_values(worker)
    original = subject.draw_noise_v1
    calls = 0

    def fail_second(parameter, generator):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second-tensor failure")
        return original(parameter, generator)

    transaction_id = "partial-candidate"
    with patch.object(subject, "draw_noise_v1", side_effect=fail_second):
        with pytest.raises(RuntimeError, match="second-tensor"):
            worker.materialize_full_weight_candidate_v1(
                transaction_id, 0, 99, 0.125, 1,
                receipt["master_identity"]["sha256"],
            )
    assert worker._fw_v1_active_candidate["phase"] == "partial_materialization"
    restored = worker.restore_or_readback_full_weight_v1(
        transaction_id, receipt["master_identity"]["sha256"],
    )
    assert restored["restored"] is True
    assert all(
        torch.equal(runtime_values(worker)[name], value)
        for name, value in before.items()
    )
    retry = worker.restore_or_readback_full_weight_v1(
        transaction_id, receipt["master_identity"]["sha256"],
    )
    assert retry["idempotent_readback"] is True


def test_fp32_master_accumulates_sub_bf16_updates_until_runtime_changes():
    worker, _receipt = make_worker(value=1.0)
    initial_runtime = runtime_values(worker)
    initial_master = master_values(worker)

    with patch.object(
        subject, "draw_noise_v1",
        side_effect=lambda parameter, generator: torch.ones_like(
            parameter, dtype=torch.float32,
        ),
    ):
        prepared = worker.update_weights_from_seeds([1], [1.0], 0.001, 1)
        assert prepared["prepared"] is True
        assert torch.equal(worker._fw_v1_master["alpha"], initial_master["alpha"])
        first = worker.broadcast_all_weights(0)
        assert first["master_changed_elements"] == 8
        assert first["master_update_l2"] > 0.0
        assert first["master_update_max_abs"] > 0.0
        assert torch.equal(worker.model_runner.model.alpha, initial_runtime["alpha"])
        assert not torch.equal(worker._fw_v1_master["alpha"], initial_master["alpha"])

        for _ in range(3):
            worker.update_weights_from_seeds([1], [1.0], 0.001, 1)
            worker.broadcast_all_weights(0)

    assert torch.all(worker.model_runner.model.alpha == torch.tensor(
        1.0078125, dtype=torch.bfloat16,
    ))
    assert worker._fw_v1_generation == 4
    assert worker._fw_v1_update_sequence == 4


def test_partial_update_is_terminal_and_blocks_observation_or_publication():
    worker, _receipt = make_worker()
    original = subject.draw_noise_v1
    calls = 0

    def fail_second(parameter, generator):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected update interruption")
        return original(parameter, generator)

    with patch.object(subject, "draw_noise_v1", side_effect=fail_second):
        worker.update_weights_from_seeds([3], [1.0], 0.01, 1)
        with pytest.raises(RuntimeError, match="update interruption"):
            worker.broadcast_all_weights(0)
    assert worker._fw_v1_poisoned is True
    with pytest.raises(RuntimeError, match="poisoned"):
        worker.full_weight_state_certificate_v1()
    with TemporaryDirectory() as directory, pytest.raises(
        RuntimeError, match="poisoned",
    ):
        worker.save_self_weights_to_disk(Path(directory) / "forbidden.pt")


def test_post_update_identity_failure_is_terminally_poisoned():
    worker, _receipt = make_worker()
    worker.update_weights_from_seeds([3], [1.0], 0.01, 1)
    with patch.object(
        subject, "_state_identity_v1",
        side_effect=RuntimeError("injected post-update hash failure"),
    ):
        with pytest.raises(RuntimeError, match="post-update hash"):
            worker.broadcast_all_weights(0)
    assert worker._fw_v1_poisoned is True
    assert worker._fw_v1_update_phase == "terminal_partial_sharded_update"
    with pytest.raises(RuntimeError, match="poisoned"):
        worker.full_weight_state_certificate_v1()


def test_state_certificate_detects_drift_without_repairing_it():
    worker, _receipt = make_worker()
    with torch.no_grad():
        worker.model_runner.model.alpha.add_(torch.tensor(
            0.125, dtype=torch.bfloat16,
        ))
    drifted = worker.model_runner.model.alpha.detach().clone()
    with pytest.raises(RuntimeError, match="differs from canonical"):
        worker.full_weight_state_certificate_v1()
    assert torch.equal(worker.model_runner.model.alpha, drifted)
    assert worker._fw_v1_poisoned is True


def test_state_certificate_detects_sub_bf16_master_drift_without_repair():
    worker, _receipt = make_worker()
    runtime_before = runtime_values(worker)
    worker._fw_v1_master["alpha"].add_(0.0001)
    drifted_master = worker._fw_v1_master["alpha"].detach().clone()
    with pytest.raises(RuntimeError, match="FP32 master identity changed"):
        worker.full_weight_state_certificate_v1()
    assert torch.equal(worker._fw_v1_master["alpha"], drifted_master)
    assert worker._fw_v1_poisoned is True
    assert all(
        torch.equal(runtime_values(worker)[name], value)
        for name, value in runtime_before.items()
    )
    with pytest.raises(RuntimeError, match="poisoned"):
        worker.update_weights_from_seeds([1], [1.0], 0.01, 1)
    with pytest.raises(RuntimeError, match="poisoned"):
        worker.perturb_self_weights(1, 0.01, False)
    with pytest.raises(RuntimeError, match="poisoned"):
        worker.full_weight_state_certificate_v1()
    with TemporaryDirectory() as directory, pytest.raises(
        RuntimeError, match="poisoned",
    ):
        worker.save_self_weights_to_disk(Path(directory) / "forbidden.pt")


def test_checkpoint_save_detects_master_drift_and_blocks_reuse():
    worker, _receipt = make_worker()
    worker._fw_v1_master["alpha"].add_(0.0001)
    with TemporaryDirectory() as directory, pytest.raises(
        RuntimeError, match="cannot save a changed canonical FP32 master",
    ):
        worker.save_self_weights_to_disk(Path(directory) / "corrupted.pt")
    assert worker._fw_v1_poisoned is True
    with pytest.raises(RuntimeError, match="poisoned"):
        worker.update_weights_from_seeds([1], [1.0], 0.01, 1)


def test_fp32_checkpoint_round_trip_preserves_sub_bf16_residual():
    worker, _receipt = make_worker(value=1.0)
    with patch.object(
        subject, "draw_noise_v1",
        side_effect=lambda parameter, generator: torch.ones_like(
            parameter, dtype=torch.float32,
        ),
    ):
        worker.update_weights_from_seeds([1], [1.0], 0.001, 1)
        worker.broadcast_all_weights(0)
    saved_master = master_values(worker)
    saved_runtime = runtime_values(worker)
    assert torch.equal(saved_runtime["alpha"], torch.ones(
        4, dtype=torch.bfloat16,
    ))
    assert torch.all(saved_master["alpha"] > 1.0)

    with TemporaryDirectory() as directory:
        path = Path(directory) / "canonical.pt"
        save_receipt = worker.save_self_weights_to_disk(path)
        assert save_receipt["resumable_fp32_master"] is True

        restored = object.__new__(subject.CanonicalFullWeightWorkerExtensionV1)
        restored.model_runner = SimpleNamespace(model=TinyModel(value=0.0))
        restored.inter_pg = FakeCommunicator()
        restored.world_size = 1
        load_receipt = restored.load_weights_from_disk(path)
        assert load_receipt["canonical_v1"] is True
        restored.install_full_weight_master_v1()

    assert all(
        torch.equal(restored._fw_v1_master[name], value)
        for name, value in saved_master.items()
    )
    assert all(
        torch.equal(runtime_values(restored)[name], value)
        for name, value in saved_runtime.items()
    )


def test_canonical_checkpoint_rejects_state_or_schedule_tampering():
    worker, _receipt = make_worker()
    with TemporaryDirectory() as directory:
        directory = Path(directory)
        original = directory / "canonical.pt"
        worker.save_self_weights_to_disk(original)
        payload = torch.load(original, map_location="cpu", weights_only=True)

        payload["state_dict"]["alpha"].add_(0.0001)
        state_tampered = directory / "state-tampered.pt"
        torch.save(payload, state_tampered)
        target = object.__new__(
            subject.CanonicalFullWeightWorkerExtensionV1
        )
        target.model_runner = SimpleNamespace(model=TinyModel(value=0.0))
        target.inter_pg = FakeCommunicator()
        target.world_size = 1
        with pytest.raises(RuntimeError, match="master identity changed"):
            target.load_weights_from_disk(state_tampered)

        payload = torch.load(original, map_location="cpu", weights_only=True)
        payload["noise_schedule"] = "tampered-schedule"
        schedule_tampered = directory / "schedule-tampered.pt"
        torch.save(payload, schedule_tampered)
        target = object.__new__(
            subject.CanonicalFullWeightWorkerExtensionV1
        )
        target.model_runner = SimpleNamespace(model=TinyModel(value=0.0))
        target.inter_pg = FakeCommunicator()
        target.world_size = 1
        with pytest.raises(RuntimeError, match="noise schedule changed"):
            target.load_weights_from_disk(schedule_tampered)


def test_tp_greater_than_one_is_rejected_before_state_capture():
    worker = object.__new__(subject.CanonicalFullWeightWorkerExtensionV1)
    worker.model_runner = SimpleNamespace(model=TinyModel())
    worker.inter_pg = FakeCommunicator()
    worker.world_size = 2
    with pytest.raises(RuntimeError, match="requires TP=1"):
        worker.install_full_weight_master_v1()


def test_runtime_aliases_are_sealed_without_perturbing_storage_twice():
    worker = object.__new__(subject.CanonicalFullWeightWorkerExtensionV1)
    worker.model_runner = SimpleNamespace(model=AliasedModel())
    worker.inter_pg = FakeCommunicator()
    worker.world_size = 1
    receipt = worker.install_full_weight_master_v1()
    assert receipt["master_identity"]["parameter_count"] == 1
    assert worker._fw_v1_manifest == [{
        "name": "primary.weight",
        "aliases": ["alternate.weight"],
        "shape": [4],
        "dtype": "torch.bfloat16",
        "elements": 4,
        "runtime_bytes": 8,
        "offset": 0,
    }]
    before = worker.model_runner.model.primary.weight.detach().clone()
    worker.perturb_self_weights(5, 0.125, False)
    assert not torch.equal(worker.model_runner.model.primary.weight, before)
    worker.restore_self_weights(5, 0.125)
    assert torch.equal(worker.model_runner.model.primary.weight, before)
    assert (
        worker.model_runner.model.primary.weight
        is worker.model_runner.model.alternate.weight
    )
