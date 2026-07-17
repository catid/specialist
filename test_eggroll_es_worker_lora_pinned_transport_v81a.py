from __future__ import annotations

import ast
import contextlib
import gc
import inspect
import weakref
from types import SimpleNamespace

import pytest
import torch

import eggroll_es_worker_lora_v41a as state_v41a
import eggroll_es_worker_lora_pinned_transport_v81a as worker_v81a
from eggroll_es_adapter_transport_precision_v81 import (
    EXPECTED_RUNTIME_BYTES_V81,
    EXPECTED_RUNTIME_ELEMENTS_V81,
    EXPECTED_RUNTIME_VIEWS_V81,
)
from test_eggroll_es_worker_lora_v41a import (
    CONFIG,
    WEIGHTS,
    FakePG,
    fake_manager,
    source_master,
)


class FakeEventV81A:
    def __init__(self):
        self.recorded_stream = None
        self.synchronize_calls = 0
        self.complete = False

    def record(self, stream):
        self.recorded_stream = stream

    def synchronize(self):
        if self.recorded_stream is None:
            raise RuntimeError("synthetic event was never recorded")
        self.synchronize_calls += 1
        self.complete = True

    def query(self):
        return self.complete


class FakeConsumerStreamV81A:
    def __init__(self):
        self.waited = []

    def wait_event(self, event):
        if event.recorded_stream is None:
            raise RuntimeError("synthetic consumer saw an unrecorded event")
        self.waited.append(event)


def make_worker_v81a(monkeypatch):
    monkeypatch.setattr(state_v41a, "EXPECTED_BASE_ELEMENTS_V41A", 23)
    monkeypatch.setattr(state_v41a, "EXPECTED_BASE_BYTES_V41A", 46)
    manager = fake_manager(source_master())
    value = object.__new__(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A
    )
    value.device = torch.device("cpu")
    value.model_runner = SimpleNamespace(
        lora_manager=SimpleNamespace(_adapter_manager=manager)
    )
    value.inter_pg = FakePG(rank=0)
    value._test_consumer_v81a = FakeConsumerStreamV81A()

    # Test the complete worker lifecycle without initializing CUDA.  The
    # production implementations of these hooks are source-attested below.
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_allocate_pinned_bank_v81a",
        lambda self: torch.empty(
            EXPECTED_RUNTIME_ELEMENTS_V81, dtype=torch.bfloat16
        ),
    )
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_bank_is_pinned_v81a",
        lambda self, bank: isinstance(bank, torch.Tensor),
    )
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_validate_cuda_views_v81a",
        lambda self, views: torch.device("cpu"),
    )
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_new_copy_stream_v81a",
        lambda self, device: object(),
    )
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_new_copy_event_v81a",
        lambda self: FakeEventV81A(),
    )
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_copy_stream_context_v81a",
        lambda self, stream: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_consumer_stream_v81a",
        lambda self, device: self._test_consumer_v81a,
    )
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_device_synchronize_v81a",
        lambda self, device: None,
    )
    installed = value.install_adapter_state_v41a(
        WEIGHTS,
        CONFIG,
        state_v41a.file_sha256_v41a(WEIGHTS),
        state_v41a.file_sha256_v41a(CONFIG),
    )
    return value, manager, installed


def test_v81a_is_additive_v72_subclass_and_has_no_cuda_import_side_effect():
    assert issubclass(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        __import__("eggroll_es_worker_lora_v72").LoRAAdapterStateWorkerExtensionV72,
    )
    source = inspect.getsource(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A
    )
    tree = ast.parse(source)
    constructors = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"Stream", "Event", "empty"}
    ]
    assert constructors
    # All allocations/event creation are method-local; importing the module
    # never constructs a CUDA stream, event, or tensor.
    assert not any(isinstance(node, (ast.Assign, ast.AnnAssign)) for node in ast.walk(tree)
                   if any(call is node for call in constructors))


def test_install_reuses_one_exact_bank_and_fences_before_exact_readback(
    monkeypatch,
):
    value, _manager, installed = make_worker_v81a(monkeypatch)
    materialization = installed["materialization"]
    transport = materialization["transport"]
    assert installed["schema"] == "canonical-lora-adapter-installed-v81a"
    assert installed["canonical_noise_update_checkpoint_authority"] == (
        "cpu_float32"
    )
    assert installed["sole_resident_adapter_slot_retained"] is True
    assert transport == {
        "schema": "eggroll-es-pinned-bf16-runtime-publication-v81a",
        "generation": 1,
        "host_bank_count": 1,
        "host_bank_bytes": EXPECTED_RUNTIME_BYTES_V81,
        "host_bank_is_pinned": True,
        "direct_h2d_copy_count": EXPECTED_RUNTIME_VIEWS_V81,
        "direct_h2d_bytes": EXPECTED_RUNTIME_BYTES_V81,
        "temporary_device_publication_staging_bytes": 0,
        "device_to_device_copy_bytes": 0,
        "consumer_stream_waited_event": True,
        "event_synchronized_before_exact_audit": True,
        "event_token": "v81a-event-1",
    }
    assert value._v81a_bank.numel() == EXPECTED_RUNTIME_ELEMENTS_V81
    assert value._v81a_bank.element_size() == 2
    assert len(value._v81a_offsets) == EXPECTED_RUNTIME_VIEWS_V81
    assert len(value._test_consumer_v81a.waited) == 1
    event = value._v81a_latest_event
    assert event.recorded_stream is value._v81a_copy_stream
    assert event.synchronize_calls == 1
    assert event.query() is True
    assert installed["pinned_transport"]["generation_may_proceed"] is True


def test_same_bank_storage_survives_master_reverify_candidate_and_restore(
    monkeypatch,
):
    value, _manager, _installed = make_worker_v81a(monkeypatch)
    pointer = value._v81a_bank_storage_data_ptr
    captured = value.capture_adapter_reference_v41a()
    assert captured["materialization"]["transport"]["generation"] == 2
    assert value._v81a_bank_storage_data_ptr == pointer

    candidate_tensors = {
        key: tensor.add(0.0003).contiguous()
        for key, tensor in value._v41_master.items()
    }
    value._v66_candidate_transaction = {
        "phase": "runtime_write_started",
        "pair_id": "a" * 64,
    }
    candidate = value._materialize_v41a(
        candidate_tensors, "v71_mirrored_candidate"
    )
    assert candidate["transport"]["generation"] == 3
    assert value._v81a_bank_storage_data_ptr == pointer
    value._v66_candidate_transaction["phase"] = "candidate_active"
    restored = value._restore_exact_master_v66("v81a_test_restore")
    assert restored["materialization"]["transport"]["generation"] == 4
    assert value._v81a_bank_storage_data_ptr == pointer
    assert value._same_source_objects_v81a(
        value._v81a_publication, value._v41_master
    )


def test_arbitrary_cross_candidate_reuse_fails_before_bank_overwrite(monkeypatch):
    value, _manager, _installed = make_worker_v81a(monkeypatch)
    before = value._v81a_bank.clone()
    candidate = {
        key: tensor.add(0.01).contiguous()
        for key, tensor in value._v41_master.items()
    }
    with pytest.raises(RuntimeError, match="cross-candidate"):
        value._materialize_v41a(candidate, "unowned_candidate")
    assert torch.equal(value._v81a_bank, before)
    assert value._v81a_generation == 1


def test_publication_weakrefs_do_not_retain_a_hidden_fp32_candidate_bank(
    monkeypatch,
):
    value, _manager, _installed = make_worker_v81a(monkeypatch)
    candidate = {
        key: tensor.add(0.0002).contiguous()
        for key, tensor in value._v41_master.items()
    }
    witness = weakref.ref(next(iter(candidate.values())))
    value._v66_candidate_transaction = {
        "phase": "runtime_write_started",
        "pair_id": "c" * 64,
    }
    value._materialize_v41a(candidate, "v71_mirrored_candidate")
    value._v66_candidate_transaction["phase"] = "candidate_active"
    del candidate
    gc.collect()
    assert witness() is None
    assert all(
        reference() is None
        for reference in value._v81a_publication["source_tensor_weakrefs"].values()
    )
    value._restore_exact_master_v66("weakref_candidate_restore")
    assert value.host_state_residency_v72()["unique_owned_bank_count"] == 1


def test_pageable_bank_is_rejected_before_any_publication(monkeypatch):
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_bank_is_pinned_v81a",
        lambda self, bank: False,
    )
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_validate_cuda_views_v81a",
        lambda self, views: torch.device("cpu"),
    )
    monkeypatch.setattr(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A,
        "_allocate_pinned_bank_v81a",
        lambda self: torch.empty(
            EXPECTED_RUNTIME_ELEMENTS_V81, dtype=torch.bfloat16
        ),
    )
    monkeypatch.setattr(state_v41a, "EXPECTED_BASE_ELEMENTS_V41A", 23)
    monkeypatch.setattr(state_v41a, "EXPECTED_BASE_BYTES_V41A", 46)
    manager = fake_manager(source_master())
    value = object.__new__(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A
    )
    value.device = torch.device("cpu")
    value.model_runner = SimpleNamespace(
        lora_manager=SimpleNamespace(_adapter_manager=manager)
    )
    value.inter_pg = FakePG(rank=0)
    with pytest.raises(RuntimeError, match="pageable"):
        value.install_adapter_state_v41a(
            WEIGHTS,
            CONFIG,
            state_v41a.file_sha256_v41a(WEIGHTS),
            state_v41a.file_sha256_v41a(CONFIG),
        )
    assert not any(name.startswith("_v81a") for name in vars(value))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value._v81a_publication.__setitem__("copy_count", 81),
         "partial or unfenced"),
        (lambda value: value._v81a_publication.__setitem__("h2d_bytes", 2),
         "partial or unfenced"),
        (lambda value: value._v81a_publication.__setitem__(
            "consumer_waited", False), "partial or unfenced"),
        (lambda value: setattr(value._v81a_latest_event, "complete", False),
         "pending"),
        (lambda value: setattr(value, "_v81a_latest_event", FakeEventV81A()),
         "partial or unfenced"),
    ],
)
def test_partial_unfenced_and_stale_publications_fail_closed(
    monkeypatch, mutation, match,
):
    value, _manager, _installed = make_worker_v81a(monkeypatch)
    mutation(value)
    with pytest.raises(RuntimeError, match=match):
        value._assert_publication_ready_v81a("synthetic_boundary")


def test_source_has_only_pinned_host_to_existing_view_h2d_copy():
    source = inspect.getsource(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A._materialize_v41a
    )
    assert "view.copy_(expected[key], non_blocking=True)" in source
    assert "expected_value.to(device=" not in source
    assert ".to(device=view.device" not in source
    assert "temporary_device_publication_staging_bytes\": 0" in source
    assert "device_to_device_copy_bytes\": 0" in source
    assert "consumer.wait_event(event)" in source
    assert "event.synchronize()" in source
    assert source.index("consumer.wait_event(event)") < source.index(
        "fused_lora_readback_v71("
    )
    assert source.index("event.synchronize()") < source.index(
        "fused_lora_readback_v71("
    )


def test_production_allocator_requires_exact_pinned_bf16_cpu_bank():
    source = inspect.getsource(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A
        ._allocate_pinned_bank_v81a
    )
    assert "pin_memory=True" in source
    assert "device=\"cpu\"" in source
    assert "EXPECTED_RUNTIME_ELEMENTS_V81" in source
    validator = inspect.getsource(
        worker_v81a.LoRAAdapterStateWorkerExtensionV81A
        ._initialize_transport_v81a
    )
    assert "_bank_is_pinned_v81a(bank)" in validator
    assert "EXPECTED_RUNTIME_BYTES_V81" in validator


def test_final_cleanup_requires_exact_master_and_releases_bank(monkeypatch):
    value, _manager, _installed = make_worker_v81a(monkeypatch)
    with pytest.raises(RuntimeError, match="exact master"):
        value.final_transport_receipt_v81a("f" * 64)
    receipt = value.final_transport_receipt_v81a(
        value._v41_current_identity["sha256"]
    )
    assert receipt["final_idle"] is True
    assert receipt["bank_released"] is True
    assert receipt["copy_stream_released"] is True
    assert value._v81a_bank is None
    assert value._v81a_publication is None
    assert value._v81a_ready is False
