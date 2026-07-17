#!/usr/bin/env python3
"""Additive pinned-BF16 publication worker for the accepted V71/V72 path.

The canonical LoRA state and all ES arithmetic remain CPU FP32.  This subclass
changes only V71's final projection transport: it fills one stable pinned BF16
host bank and copies its 82 disjoint segments directly into the already
resident vLLM slot on a dedicated CUDA stream.  A generation-scoped event is
waited by the consumer stream and synchronized before V71's exact readback.

V71 and V72 are intentionally imported, not modified.  No CUDA object is
created at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import weakref
from typing import Any, Mapping

import torch

import eggroll_es_worker_lora_v41a as state_v41a
from eggroll_es_adapter_transport_precision_v81 import (
    EXPECTED_RUNTIME_BYTES_V81,
    EXPECTED_RUNTIME_ELEMENTS_V81,
    EXPECTED_RUNTIME_VIEWS_V81,
)
from eggroll_es_worker_lora_v71 import (
    _runtime_key_v71,
    _validate_adapter_no_clone_v71,
    fused_lora_readback_v71,
)
from eggroll_es_worker_lora_v72 import LoRAAdapterStateWorkerExtensionV72


SCHEMA_V81A = "eggroll-es-pinned-bf16-runtime-publication-v81a"
WORKER_SCHEMA_V81A = "canonical-lora-adapter-installed-v81a"
RUNTIME_DTYPE_V81A = torch.bfloat16


def canonical_sha256_v81a(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _proc_pin_status_v81a(path: str = "/proc/self/status") -> dict[str, Any]:
    """Return non-authoritative process pin counters for live receipts."""
    fields: dict[str, int | None] = {"VmLck": None, "VmPin": None}
    try:
        with open(path, "r", encoding="ascii") as handle:
            for line in handle:
                label = line.split(":", 1)[0]
                if label not in fields:
                    continue
                parts = line.split()
                if len(parts) != 3 or parts[2] != "kB":
                    raise RuntimeError("v81a /proc pin counter format changed")
                fields[label] = int(parts[1]) * 1024
    except FileNotFoundError:
        pass
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    return {
        "pid": os.getpid(),
        "vm_lck_bytes": fields["VmLck"],
        "vm_pin_bytes": fields["VmPin"],
        "memlock_soft_bytes": int(soft) if soft != resource.RLIM_INFINITY else None,
        "memlock_hard_bytes": int(hard) if hard != resource.RLIM_INFINITY else None,
        "memlock_soft_unlimited": soft == resource.RLIM_INFINITY,
        "memlock_hard_unlimited": hard == resource.RLIM_INFINITY,
        "actual_tensor_is_pinned_remains_authoritative": True,
    }


class LoRAAdapterStateWorkerExtensionV81A(LoRAAdapterStateWorkerExtensionV72):
    """V72 ownership plus one event-fenced, direct-H2D pinned host bank."""

    # The small hooks below keep CPU-only contract tests from initializing CUDA.
    # Production uses their exact default implementations.
    def _allocate_pinned_bank_v81a(self):
        return torch.empty(
            EXPECTED_RUNTIME_ELEMENTS_V81,
            dtype=RUNTIME_DTYPE_V81A,
            device="cpu",
            pin_memory=True,
        )

    def _bank_is_pinned_v81a(self, bank):
        return bool(bank.is_pinned())

    def _validate_cuda_views_v81a(self, views: Mapping[str, torch.Tensor]):
        devices = {view.device for view in views.values()}
        if (
            not torch.cuda.is_available()
            or len(devices) != 1
            or next(iter(devices)).type != "cuda"
        ):
            raise RuntimeError("v81a runtime views require one CUDA device")
        return next(iter(devices))

    def _new_copy_stream_v81a(self, device):
        return torch.cuda.Stream(device=device)

    def _new_copy_event_v81a(self):
        return torch.cuda.Event(
            enable_timing=False,
            blocking=False,
            interprocess=False,
        )

    def _copy_stream_context_v81a(self, stream):
        return torch.cuda.stream(stream)

    def _consumer_stream_v81a(self, device):
        return torch.cuda.current_stream(device=device)

    def _device_synchronize_v81a(self, device):
        torch.cuda.synchronize(device)

    def _initialize_transport_v81a(self):
        if getattr(self, "_v81a_ready", False):
            return
        views = self._v71_runtime_views
        if (
            not isinstance(views, dict)
            or len(views) != EXPECTED_RUNTIME_VIEWS_V81
            or sum(int(view.numel()) for view in views.values())
            != EXPECTED_RUNTIME_ELEMENTS_V81
            or any(
                not isinstance(view, torch.Tensor)
                or view.dtype != RUNTIME_DTYPE_V81A
                for view in views.values()
            )
        ):
            raise RuntimeError("v81a exact 82-view BF16 runtime geometry changed")
        device = self._validate_cuda_views_v81a(views)
        bank = self._allocate_pinned_bank_v81a()
        if (
            not isinstance(bank, torch.Tensor)
            or bank.device.type != "cpu"
            or bank.dtype != RUNTIME_DTYPE_V81A
            or bank.ndim != 1
            or bank.numel() != EXPECTED_RUNTIME_ELEMENTS_V81
            or not bank.is_contiguous()
            or not self._bank_is_pinned_v81a(bank)
        ):
            raise RuntimeError("v81a exact host bank is absent or pageable")
        soft, _hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft != resource.RLIM_INFINITY and int(soft) < EXPECTED_RUNTIME_BYTES_V81:
            raise RuntimeError("v81a process memlock limit is below one bank")

        offsets: dict[str, tuple[int, int]] = {}
        offset = 0
        for item in self._v41_assignments:
            key = _runtime_key_v71(item)
            elements = int(views[key].numel())
            if key in offsets or elements <= 0:
                raise RuntimeError("v81a runtime segment coverage changed")
            offsets[key] = (offset, elements)
            offset += elements
        if offset != EXPECTED_RUNTIME_ELEMENTS_V81 or set(offsets) != set(views):
            raise RuntimeError("v81a pinned-bank partition is partial")

        self._v81a_bank = bank
        self._v81a_bank_storage_data_ptr = int(
            bank.untyped_storage().data_ptr()
        )
        self._v81a_offsets = offsets
        self._v81a_device = device
        self._v81a_copy_stream = self._new_copy_stream_v81a(device)
        self._v81a_generation = 0
        self._v81a_latest_event = None
        self._v81a_publication = None
        self._v81a_receipts = []
        self._v81a_ready = True

    def _publication_receipt_v81a(self, action: str, **fields: Any):
        value = {
            "schema": SCHEMA_V81A,
            "action": str(action),
            "generation": int(getattr(self, "_v81a_generation", 0)),
            **fields,
        }
        value["content_sha256_before_self_field"] = canonical_sha256_v81a(value)
        if hasattr(self, "_v81a_receipts"):
            self._v81a_receipts.append(
                value["content_sha256_before_self_field"]
            )
        return value

    def _event_is_complete_v81a(self, event) -> bool:
        if event is None or event is not getattr(self, "_v81a_latest_event", None):
            raise RuntimeError("v81a publication event is stale or foreign")
        try:
            return bool(event.query())
        except BaseException as error:
            raise RuntimeError("v81a publication event query failed") from error

    def _legal_replacement_v81a(self, tensors, phase: str) -> bool:
        publication = self._v81a_publication
        if publication is None:
            return True
        if self._same_source_objects_v81a(publication, tensors):
            return True
        if getattr(self, "_v66d_active_gpu_work", None) is not None:
            return False

        transaction = getattr(self, "_v66_candidate_transaction", None)
        pending = getattr(self, "_v41_pending_update", None)
        committed = getattr(self, "_v41_committed_rollback", None)
        # Master -> candidate is legal only at the two accepted candidate
        # creation boundaries.  Candidate -> master is legal only through the
        # inherited exact restore/abort state, never by an arbitrary caller.
        candidate_start = self._same_source_objects_v81a(
            publication, self._v41_master
        ) and ((
            isinstance(transaction, dict)
            and transaction.get("phase") == "runtime_write_started"
        ) or (
            isinstance(pending, dict) and pending.get("phase") == "prepared"
        ))
        exact_master_replacement = tensors is self._v41_master and (
            isinstance(transaction, dict)
            or isinstance(pending, dict)
            or isinstance(committed, dict)
            or "exact_master" in phase
            or "rollback" in phase
            or "abort" in phase
        )
        return bool(candidate_start or exact_master_replacement)

    @staticmethod
    def _same_source_objects_v81a(publication, tensors) -> bool:
        references = publication.get("source_tensor_weakrefs")
        return (
            isinstance(references, dict)
            and isinstance(tensors, dict)
            and set(references) == set(tensors)
            and all(references[key]() is tensors[key] for key in references)
        )

    def _retire_for_replacement_v81a(self, tensors, phase: str):
        publication = self._v81a_publication
        if publication is None:
            return
        if not self._legal_replacement_v81a(tensors, phase):
            raise RuntimeError("v81a cross-candidate pinned-bank reuse rejected")
        if getattr(self, "_v66d_active_gpu_work", None) is not None:
            raise RuntimeError("v81a bank reuse overlaps active generation")

        status = publication["status"]
        if status == "published":
            if not self._event_is_complete_v81a(publication["event"]):
                raise RuntimeError("v81a completed publication event became unfenced")
        else:
            # A partial copy has unknown device state.  Only a legal inherited
            # exact-master repair may settle the stream before rewriting.
            if tensors is not self._v41_master or not (
                "exact_master" in phase
                or "repair" in phase
                or "rollback" in phase
                or "abort" in phase
            ):
                raise RuntimeError("v81a uncertain copy requires exact master repair")
            try:
                self._device_synchronize_v81a(self._v81a_device)
            except BaseException as error:
                self._poison_v66("v81a_uncertain_copy_synchronize", error)
                raise RuntimeError(
                    "v81a uncertain copy could not synchronize; actor poisoned"
                ) from error
        self._publication_receipt_v81a(
            "retire_for_replacement",
            retired_generation=publication["generation"],
            retired_status=status,
            replacement_phase=str(phase),
            bank_reuse_authorized=True,
        )
        self._v81a_publication = None

    def _assert_publication_ready_v81a(self, boundary: str):
        self._require_not_poisoned_v66()
        publication = getattr(self, "_v81a_publication", None)
        if (
            not getattr(self, "_v81a_ready", False)
            or not isinstance(publication, dict)
            or publication.get("status") != "published"
            or publication.get("generation") != self._v81a_generation
            or publication.get("event") is not self._v81a_latest_event
            or publication.get("copy_count") != EXPECTED_RUNTIME_VIEWS_V81
            or publication.get("h2d_bytes") != EXPECTED_RUNTIME_BYTES_V81
            or publication.get("consumer_waited") is not True
            or publication.get("event_synchronized") is not True
            or publication.get("exact_audited") is not True
            or publication.get("bank_storage_data_ptr")
            != self._v81a_bank_storage_data_ptr
            or not self._bank_is_pinned_v81a(self._v81a_bank)
        ):
            raise RuntimeError(
                f"v81a publication is partial or unfenced at {boundary}"
            )
        if not self._event_is_complete_v81a(publication["event"]):
            raise RuntimeError(f"v81a completion event is pending at {boundary}")
        return self._publication_receipt_v81a(
            "ready_boundary",
            boundary=str(boundary),
            event_token=publication["event_token"],
            runtime_values_sha256=publication["runtime_values_sha256"],
            generation_may_proceed=True,
        )

    def install_adapter_state_v41a(self, *args, **kwargs):
        try:
            receipt = super().install_adapter_state_v41a(*args, **kwargs)
            ready = self._assert_publication_ready_v81a("install_return")
        except BaseException:
            self._discard_transport_v81a(best_effort=True)
            raise
        return {
            **receipt,
            "schema": WORKER_SCHEMA_V81A,
            "pinned_transport": ready,
            "host_pin_status": _proc_pin_status_v81a(),
            "canonical_noise_update_checkpoint_authority": "cpu_float32",
            "sole_resident_adapter_slot_retained": True,
        }

    def _materialize_v41a(self, tensors, phase):
        if not getattr(self, "_v71_ready", False):
            return super()._materialize_v41a(tensors, phase)
        tensors = _validate_adapter_no_clone_v71(tensors)
        if tensors is self._v41_master:
            self._master_identity_v71(f"materialize_{phase}")
        self._assert_runtime_links_v71()
        self._v71_runtime_registry.cheap_certificate(
            self._v71_runtime_views, f"before_{phase}"
        )
        self._initialize_transport_v81a()
        self._retire_for_replacement_v81a(tensors, str(phase))

        manager = self._manager_v41a()
        runtime_names = {
            item["runtime_module"] for item in self._v41_assignments
        }
        expected: dict[str, torch.Tensor] = {}
        storage = []
        self._v81a_generation += 1
        generation = self._v81a_generation
        publication = {
            "status": "staging",
            "generation": generation,
            # Weak references prove same-object reuse without retaining a
            # hidden FP32 candidate bank or weakening V72's one/two/one
            # ownership ledger.
            "source_tensor_weakrefs": {
                key: weakref.ref(tensor) for key, tensor in tensors.items()
            },
            "phase": str(phase),
            "event": None,
            "event_token": f"v81a-event-{generation}",
            "copy_count": 0,
            "h2d_bytes": 0,
            "consumer_waited": False,
            "event_synchronized": False,
            "exact_audited": False,
            "bank_storage_data_ptr": self._v81a_bank_storage_data_ptr,
        }
        self._v81a_publication = publication
        try:
            with torch.no_grad():
                for item in self._v41_assignments:
                    key = _runtime_key_v71(item)
                    source = tensors[item["peft_key"]]
                    logical, side = state_v41a.topology._source_parts(
                        item["peft_key"]
                    )
                    _target, slices = state_v41a.topology._runtime_target(logical)
                    module = manager.modules[item["runtime_module"]]
                    expected_value = state_v41a._expected_runtime_value_v41a(
                        source,
                        side,
                        slices,
                        item["slice_index"],
                        module.output_slices,
                        self._v41_scale,
                    ).contiguous()
                    offset, elements = self._v81a_offsets[key]
                    segment = self._v81a_bank.narrow(0, offset, elements).view(
                        expected_value.shape
                    )
                    view = self._v71_runtime_views[key]
                    parent = self._v71_runtime_parents[key]
                    if (
                        view.shape != expected_value.shape
                        or view.dtype != RUNTIME_DTYPE_V81A
                        or segment.dtype != RUNTIME_DTYPE_V81A
                        or not self._bank_is_pinned_v81a(segment)
                    ):
                        raise RuntimeError(
                            f"v81a runtime view metadata changed: {item['peft_key']}"
                        )
                    segment.copy_(expected_value, non_blocking=False)
                    expected[key] = segment
                    storage.append({
                        "signature": [
                            item["runtime_module"], side, item["slice_index"],
                        ],
                        "storage_data_ptr": int(
                            parent.untyped_storage().data_ptr()
                        ),
                        "view_storage_data_ptr": int(
                            view.untyped_storage().data_ptr()
                        ),
                        "view_storage_offset": int(view.storage_offset()),
                        "view_aliases_parent": (
                            parent.untyped_storage().data_ptr()
                            == view.untyped_storage().data_ptr()
                        ),
                        "bank_offset_elements": offset,
                        "bank_elements": elements,
                    })
                if (
                    len(expected) != EXPECTED_RUNTIME_VIEWS_V81
                    or sum(item.numel() for item in expected.values())
                    != EXPECTED_RUNTIME_ELEMENTS_V81
                    or not self._bank_is_pinned_v81a(self._v81a_bank)
                ):
                    raise RuntimeError("v81a staged pinned-bank coverage is partial")

                event = self._new_copy_event_v81a()
                publication["event"] = event
                self._v81a_latest_event = event
                publication["status"] = "copies_inflight"
                with self._copy_stream_context_v81a(self._v81a_copy_stream):
                    for name in sorted(runtime_names):
                        manager.modules[name].reset_lora(
                            state_v41a.ADAPTER_SLOT_V41A
                        )
                    for item in self._v41_assignments:
                        key = _runtime_key_v71(item)
                        view = self._v71_runtime_views[key]
                        # This is the sole H2D call site.  Its source is a view
                        # of the pinned CPU bank and its destination is the
                        # existing resident runtime slot; there is no device
                        # staging tensor and therefore no D2D copy.
                        view.copy_(expected[key], non_blocking=True)
                        publication["copy_count"] += 1
                        publication["h2d_bytes"] += int(
                            expected[key].numel() * expected[key].element_size()
                        )
                    event.record(self._v81a_copy_stream)
                if (
                    publication["copy_count"] != EXPECTED_RUNTIME_VIEWS_V81
                    or publication["h2d_bytes"] != EXPECTED_RUNTIME_BYTES_V81
                ):
                    raise RuntimeError("v81a direct H2D publication is partial")
                consumer = self._consumer_stream_v81a(self._v81a_device)
                consumer.wait_event(event)
                publication["consumer_waited"] = True
                event.synchronize()
                publication["event_synchronized"] = True
                if not self._event_is_complete_v81a(event):
                    raise RuntimeError("v81a completion event did not fence copies")
                publication["status"] = "copies_complete"

            readback = fused_lora_readback_v71(
                self._v71_runtime_views, expected
            )
            self._device_synchronize_v81a(self._v81a_device)
            records = []
            for item in self._v41_assignments:
                key = _runtime_key_v71(item)
                view = self._v71_runtime_views[key]
                records.append({
                    **item,
                    "dtype": str(view.dtype),
                    "elements": int(view.numel()),
                    "sha256": readback["sha256_by_key"][key],
                })
            elements = sum(item["elements"] for item in records)
            pointers = [item["storage_data_ptr"] for item in storage]
            if (
                len(records) != EXPECTED_RUNTIME_VIEWS_V81
                or elements != EXPECTED_RUNTIME_ELEMENTS_V81
                or len(set(pointers)) != EXPECTED_RUNTIME_VIEWS_V81
                or not all(item["view_aliases_parent"] for item in storage)
                or readback["d2h_bytes"] != EXPECTED_RUNTIME_BYTES_V81
                or readback["d2h_calls"] != 1
            ):
                raise RuntimeError("v81a residency or exact readback changed")
            self._v71_runtime_registry.rebind_controlled_write(
                self._v71_runtime_views,
                str(phase),
                expected_sha256=readback["sha256_by_key"],
            )
            compact = [{
                key: value
                for key, value in item.items()
                if key not in {"storage_data_ptr", "view_storage_data_ptr"}
            } for item in storage]
            certificate = {
                "schema": "canonical-to-vllm-lora-materialization-v81a",
                "phase": str(phase),
                "adapter_id": 1,
                "slot": 0,
                "source_tensor_count": len(tensors),
                "source_elements": sum(
                    tensor.numel() for tensor in tensors.values()
                ),
                "runtime_module_count": len(runtime_names),
                "runtime_view_count": len(records),
                "runtime_elements": elements,
                "runtime_dtype": "torch.bfloat16",
                "b_scale": self._v41_scale,
                "a_duplication_and_b_splitting_verified": True,
                "unique_parent_storage_count": len(set(pointers)),
                "runtime_views_share_no_parent_storage": (
                    len(set(pointers)) == len(pointers)
                ),
                "slot_views_alias_parent_buffers": all(
                    item["view_aliases_parent"] for item in storage
                ),
                "storage_layout_sha256": state_v41a.canonical_sha256_v3(
                    compact
                ),
                "runtime_values_sha256": state_v41a.canonical_sha256_v3(
                    records
                ),
                "single_d2h_readback": readback,
                "validation_clone_bytes": 0,
                "transport": {
                    "schema": SCHEMA_V81A,
                    "generation": generation,
                    "host_bank_count": 1,
                    "host_bank_bytes": EXPECTED_RUNTIME_BYTES_V81,
                    "host_bank_is_pinned": True,
                    "direct_h2d_copy_count": publication["copy_count"],
                    "direct_h2d_bytes": publication["h2d_bytes"],
                    "temporary_device_publication_staging_bytes": 0,
                    "device_to_device_copy_bytes": 0,
                    "consumer_stream_waited_event": True,
                    "event_synchronized_before_exact_audit": True,
                    "event_token": publication["event_token"],
                },
            }
            publication.update({
                "status": "published",
                "exact_audited": True,
                "runtime_values_sha256": certificate[
                    "runtime_values_sha256"
                ],
            })
            self._v41_active_materialization = certificate
            self._traffic_add_v71("h2d_bytes", readback["d2h_bytes"])
            self._traffic_add_v71("lora_d2h_bytes", readback["d2h_bytes"])
            self._traffic_add_v71("lora_d2h_calls", 1)
            self._assert_publication_ready_v81a(
                f"materialize_return_{phase}"
            )
            return certificate
        except BaseException:
            publication["status"] = "uncertain"
            raise

    def _exact_lora_audit_v71(self, boundary):
        try:
            self._assert_publication_ready_v81a(
                f"before_exact_lora_audit_{boundary}"
            )
        except BaseException as error:
            self._fail_closed_runtime_v71(
                f"v81a_unfenced_{boundary}", error
            )
        return super()._exact_lora_audit_v71(boundary)

    def _cheap_transition_audit_v71(self, phase):
        self._assert_publication_ready_v81a(
            f"before_transition_audit_{phase}"
        )
        return super()._cheap_transition_audit_v71(phase)

    def begin_actor_gpu_work_v66d(self, assignment):
        try:
            self._assert_publication_ready_v81a("before_generation")
        except BaseException as error:
            self._fail_closed_runtime_v71("v81a_before_generation", error)
        return super().begin_actor_gpu_work_v66d(assignment)

    def final_transport_receipt_v81a(self, expected_master_sha256):
        self._require_not_poisoned_v66()
        self._require_quiescent_v41a()
        if (
            self._v41_current_identity["sha256"]
            != str(expected_master_sha256)
            or not self._same_source_objects_v81a(
                self._v81a_publication, self._v41_master
            )
        ):
            raise RuntimeError("v81a final cleanup is not on the exact master")
        ready = self._assert_publication_ready_v81a("final_cleanup")
        generation = self._v81a_generation
        bank_pointer = self._v81a_bank_storage_data_ptr
        chain = canonical_sha256_v81a(self._v81a_receipts)
        pin_status_before_release = _proc_pin_status_v81a()
        self._device_synchronize_v81a(self._v81a_device)
        self._v81a_publication = None
        self._v81a_latest_event = None
        self._v81a_copy_stream = None
        self._v81a_bank = None
        self._v81a_ready = False
        return self._publication_receipt_v81a(
            "final_cleanup",
            completed_generations=generation,
            bank_storage_data_ptr=bank_pointer,
            bank_released=True,
            copy_stream_released=True,
            final_master_sha256=str(expected_master_sha256),
            ready_boundary_sha256=ready[
                "content_sha256_before_self_field"
            ],
            prior_receipt_chain_sha256=chain,
            host_pin_status_before_release=pin_status_before_release,
            host_pin_status_after_release=_proc_pin_status_v81a(),
            final_idle=True,
        )

    def transport_status_receipt_v81a(self):
        """Attest live allocation/event state without weakening V71 audits."""
        ready = self._assert_publication_ready_v81a("status_receipt")
        return self._publication_receipt_v81a(
            "status",
            runtime_view_count=len(self._v71_runtime_views),
            runtime_elements=sum(
                int(view.numel()) for view in self._v71_runtime_views.values()
            ),
            host_bank_count=1,
            host_bank_bytes=int(
                self._v81a_bank.numel() * self._v81a_bank.element_size()
            ),
            host_bank_is_pinned=self._bank_is_pinned_v81a(self._v81a_bank),
            bank_storage_data_ptr=self._v81a_bank_storage_data_ptr,
            event_token=self._v81a_publication["event_token"],
            event_complete=self._event_is_complete_v81a(
                self._v81a_publication["event"]
            ),
            ready_boundary_sha256=ready[
                "content_sha256_before_self_field"
            ],
            host_pin_status=_proc_pin_status_v81a(),
        )

    def _discard_transport_v81a(self, *, best_effort: bool):
        if not any(name.startswith("_v81a") for name in vars(self)):
            return
        if getattr(self, "_v81a_ready", False):
            try:
                self._device_synchronize_v81a(self._v81a_device)
            except BaseException:
                if not best_effort:
                    raise
        for name in [key for key in vars(self) if key.startswith("_v81a")]:
            delattr(self, name)


__all__ = [
    "LoRAAdapterStateWorkerExtensionV81A",
    "SCHEMA_V81A",
    "WORKER_SCHEMA_V81A",
    "_proc_pin_status_v81a",
]
