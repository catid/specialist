from __future__ import annotations

import hashlib

import pytest

import eggroll_es_gpu_telemetry_v66 as telemetry
import eggroll_es_worker_lora_v66d as subject


class FakeEvent:
    def __init__(self, elapsed_ms=4.25):
        self.elapsed_ms = elapsed_ms
        self.recorded = False
        self.synchronized = False

    def record(self):
        self.recorded = True

    def synchronize(self):
        self.synchronized = True

    def elapsed_time(self, _end):
        return self.elapsed_ms


class FakeCuda:
    def __init__(self, elapsed_ms=4.25):
        self.elapsed_ms = elapsed_ms
        self.synchronizations = []
        self.events = []

    @staticmethod
    def is_available():
        return True

    def synchronize(self, device):
        self.synchronizations.append(str(device))

    def Event(self, *, enable_timing):
        assert enable_timing is True
        event = FakeEvent(self.elapsed_ms)
        self.events.append(event)
        return event


def assignment():
    return {
        "wave_index": 2,
        "engine_rank": 1,
        "direction_seed": 1703,
        "sign": -1,
        "pair_id": hashlib.sha256(b"pair").hexdigest(),
        "evaluation_contract_sha256": hashlib.sha256(b"contract").hexdigest(),
    }


def worker(monkeypatch, *, active=True, elapsed_ms=4.25):
    value = object.__new__(subject.LoRAAdapterStateWorkerExtensionV66D)
    value.device = "cuda:0"
    value._v66_terminal_poison = None
    value._v66d_active_gpu_work = None
    item = assignment()
    value._v66_candidate_transaction = ({
        "phase": "candidate_active",
        "direction_seed": item["direction_seed"],
        "sign": item["sign"],
        "pair_id": item["pair_id"],
        "evaluation_contract_sha256": item[
            "evaluation_contract_sha256"
        ],
    } if active else None)
    monkeypatch.setattr(
        value,
        "_intrinsic_worker_identity_v65b",
        lambda: {
            "worker_pid": 4321,
            "worker_physical_gpu_id": 3,
            "worker_cuda_visible_devices": "3",
        },
    )
    fake_cuda = FakeCuda(elapsed_ms)
    monkeypatch.setattr(subject.torch, "cuda", fake_cuda)
    return value, fake_cuda


def cardinality():
    return {
        "request_outputs": 64,
        "samples": 64,
        "generated_tokens": 64,
        "prompt_tokens": 4096,
    }


def test_cuda_events_span_active_candidate_and_seal_actor_bound_receipt(
    monkeypatch,
):
    value, fake_cuda = worker(monkeypatch)
    begin = value.begin_actor_gpu_work_v66d(assignment())
    assert begin["worker_pid"] == 4321
    assert begin["physical_gpu_id"] == 3
    assert begin["cuda_event_start_recorded"] is True
    receipt = value.end_actor_gpu_work_v66d(assignment(), cardinality())
    assert receipt["cuda_event"]["elapsed_ms"] == 4.25
    assert receipt["output_cardinality"] == cardinality()
    assert receipt == telemetry.seal_actor_work_receipt_v66d(receipt)
    assert value._v66d_active_gpu_work is None
    assert len(fake_cuda.synchronizations) == 2
    assert len(fake_cuda.events) == 2
    assert fake_cuda.events[0].recorded is True
    assert fake_cuda.events[1].recorded is True
    assert fake_cuda.events[1].synchronized is True


def test_begin_rejects_missing_or_wrong_active_candidate(monkeypatch):
    value, _ = worker(monkeypatch, active=False)
    with pytest.raises(RuntimeError, match="active signed candidate"):
        value.begin_actor_gpu_work_v66d(assignment())


def test_begin_rejects_overlapping_event(monkeypatch):
    value, _ = worker(monkeypatch)
    value.begin_actor_gpu_work_v66d(assignment())
    with pytest.raises(RuntimeError, match="already has an active"):
        value.begin_actor_gpu_work_v66d(assignment())


def test_end_rejects_missing_event_and_nonpositive_cardinality(monkeypatch):
    value, _ = worker(monkeypatch)
    with pytest.raises(RuntimeError, match="missing or changed"):
        value.end_actor_gpu_work_v66d(assignment(), cardinality())
    value.begin_actor_gpu_work_v66d(assignment())
    bad = cardinality()
    bad["generated_tokens"] = 0
    with pytest.raises(RuntimeError, match="not positive"):
        value.end_actor_gpu_work_v66d(assignment(), bad)


def test_end_rejects_zero_cuda_elapsed_time(monkeypatch):
    value, _ = worker(monkeypatch, elapsed_ms=0.0)
    value.begin_actor_gpu_work_v66d(assignment())
    with pytest.raises(RuntimeError, match="did not prove"):
        value.end_actor_gpu_work_v66d(assignment(), cardinality())
    assert value._v66d_active_gpu_work is None
