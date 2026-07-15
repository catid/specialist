from types import SimpleNamespace

import pytest

import eggroll_es_worker_v24a_memory_retry_r1 as worker_r1


def _worker(value):
    worker = object.__new__(worker_r1.HybridBackendMemoryWorkerExtensionV24ARetryR1)
    worker.model_runner = SimpleNamespace(model_memory_usage=value)
    worker._communicator_state_v3 = lambda expected: {
        "rank": 2, "world_size": expected,
    }
    return worker


def test_v24a_r1_worker_returns_exact_model_load_int(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    value = 35_991_420_928
    report = _worker(value).model_load_memory_v24a_r1("bf16_b")
    assert report == {
        "schema": "eggroll-es-v24a-model-load-memory-worker-r1",
        "rank": 2,
        "world_size": 4,
        "arm": "bf16_b",
        "cuda_visible_devices": "2",
        "model_load_consumed_bytes": value,
        "source_object": "self.model_runner.model_memory_usage",
        "source_assignment": "self.model_memory_usage = m.consumed_memory",
        "measured_after_model_load_before_scoring": True,
    }


@pytest.mark.parametrize("value", [None, True, 1.0, "1", 0, -1])
def test_v24a_r1_worker_rejects_nonpositive_or_nonexact_int(value):
    with pytest.raises(RuntimeError, match="positive exact int"):
        _worker(value).model_load_memory_v24a_r1("bf16_b")
