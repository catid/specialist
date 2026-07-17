#!/usr/bin/env python3
"""CPU contract for lower-precision LoRA execution transport V81.

Canonical adapter, perturbation, update, optimizer, and checkpoint authority
remain FP32 on CPU.  The only transport payload is the installed vLLM LoRA
execution dtype.  For the sealed Qwen3.6 topology that dtype is BF16.

This module deliberately does not allocate pinned memory or initialize CUDA.
It attests the installed source surface, produces an exact byte ledger, and
models the stream/event publication protocol that a later CUDA integration
must obey.
"""

from __future__ import annotations

import ast
import hashlib
import json
import numbers
from dataclasses import dataclass, field
from typing import Any, Mapping


SCHEMA_V81 = "eggroll-es-adapter-transport-precision-v81"
PLAN_SCHEMA_V81 = "qwen36-lora-transport-plan-v81"
FENCE_SCHEMA_V81 = "stream-safe-lora-publication-v81"

EXPECTED_SOURCE_TENSORS_V81 = 70
EXPECTED_SOURCE_ELEMENTS_V81 = 4_528_128
EXPECTED_SOURCE_BYTES_V81 = 18_112_512
EXPECTED_RUNTIME_VIEWS_V81 = 82
EXPECTED_RUNTIME_ELEMENTS_V81 = 4_921_344
EXPECTED_RUNTIME_BYTES_V81 = 9_842_688
EXPECTED_RUNTIME_DTYPE_V81 = "bfloat16"
EXPECTED_PROJECTION_SHA256_V81 = (
    "7ad7c2ec6f55d38915744a6287e1d0bd56b4393f319053c62f3f4c9e36c9dcf5"
)
SUPPORTED_VLLM_DTYPES_V81 = ("auto", "float16", "bfloat16")


def canonical_sha256_v81(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def text_sha256_v81(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _exact_int_v81(value: Any, label: str, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or int(value) < minimum
    ):
        raise ValueError(f"v81 {label} must be an integer >= {minimum}")
    return int(value)


def _nonempty_v81(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"v81 {label} must be a nonempty string")
    return value


def _sha256_v81(value: Any, label: str) -> str:
    value = _nonempty_v81(value, label)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"v81 {label} must be a lowercase SHA-256")
    return value


def _normalize_dtype_v81(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("v81 LoRA dtype must be a string")
    value = value.removeprefix("torch.").lower()
    aliases = {"fp16": "float16", "bf16": "bfloat16"}
    return aliases.get(value, value)


def _literal_strings_v81(annotation: ast.expr) -> tuple[str, ...]:
    if not isinstance(annotation, ast.Subscript):
        return ()
    name = annotation.value
    if not (
        isinstance(name, ast.Name) and name.id == "Literal"
        or isinstance(name, ast.Attribute) and name.attr == "Literal"
    ):
        return ()
    values = annotation.slice.elts if isinstance(annotation.slice, ast.Tuple) else [annotation.slice]
    result = []
    for value in values:
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            return ()
        result.append(value.value)
    return tuple(result)


def attest_vllm_lora_sources_v81(
    config_source: str,
    base_linear_source: str,
    fused_moe_source: str,
    model_manager_source: str,
) -> dict[str, Any]:
    """Attest dtype allocation, direct async copy, and CPU pinning support."""
    tree = ast.parse(config_source)
    dtype_literals = ()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "LoRADType"
        ):
            dtype_literals = _literal_strings_v81(node.value)
            break
    if dtype_literals != SUPPORTED_VLLM_DTYPES_V81:
        raise RuntimeError(
            f"v81 installed vLLM LoRA dtype surface changed: {dtype_literals}"
        )
    if (
        "self.lora_dtype = model_config.dtype" not in config_source
        or "self.lora_dtype = getattr(torch, self.lora_dtype)" not in config_source
    ):
        raise RuntimeError("v81 vLLM LoRA dtype resolution changed")

    base_allocations = base_linear_source.count("dtype=lora_config.lora_dtype")
    fused_allocations = fused_moe_source.count("dtype=lora_config.lora_dtype")
    base_async_copies = base_linear_source.count("non_blocking=True")
    fused_async_copies = fused_moe_source.count("non_blocking=True")
    if (
        base_allocations != 2
        or fused_allocations != 6
        or base_async_copies < 2
        or fused_async_copies < 10
    ):
        raise RuntimeError("v81 vLLM LoRA allocation/copy surface changed")
    if (
        "pin_memory = str(lora_device) == \"cpu\" and PIN_MEMORY"
        not in model_manager_source
        or model_manager_source.count(".pin_memory()") < 4
    ):
        raise RuntimeError("v81 vLLM packed-host LoRA pinning surface changed")
    return {
        "schema": "installed-vllm-lora-transport-capability-v81",
        "supported_lora_dtypes": list(dtype_literals),
        "byte_lower_than_bfloat16_supported": False,
        "float16_supported_but_byte_neutral": True,
        "fp8_lora_execution_supported": False,
        "auto_resolves_to_model_dtype": True,
        "dense_slot_allocations_use_lora_dtype": base_allocations,
        "fused_moe_slot_allocations_use_lora_dtype": fused_allocations,
        "dense_direct_nonblocking_copy_sites_at_least": base_async_copies,
        "fused_moe_direct_nonblocking_copy_sites_at_least": fused_async_copies,
        "packed_cpu_lora_pinning_present": True,
        "config_source_sha256": text_sha256_v81(config_source),
        "base_linear_source_sha256": text_sha256_v81(base_linear_source),
        "fused_moe_source_sha256": text_sha256_v81(fused_moe_source),
        "model_manager_source_sha256": text_sha256_v81(model_manager_source),
    }


def resolve_execution_dtype_v81(
    requested_dtype: str,
    model_dtype: str,
    supported_dtypes: tuple[str, ...] | list[str] = SUPPORTED_VLLM_DTYPES_V81,
) -> str:
    requested = _normalize_dtype_v81(requested_dtype)
    model = _normalize_dtype_v81(model_dtype)
    supported = tuple(_normalize_dtype_v81(value) for value in supported_dtypes)
    resolved = model if requested == "auto" else requested
    if requested not in supported or resolved not in supported or resolved == "auto":
        raise RuntimeError(f"v81 unsupported vLLM LoRA execution dtype: {requested}")
    if resolved not in {"float16", "bfloat16"}:
        raise RuntimeError(f"v81 byte-lower LoRA execution dtype is unsupported: {resolved}")
    return resolved


def validate_projection_manifest_v81(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise TypeError("v81 projection manifest must be a mapping")
    compact = dict(manifest)
    claimed = compact.pop("content_sha256_before_self_field", None)
    if (
        manifest.get("schema") != "qwen36-lora-runtime-projection-v72"
        or claimed != EXPECTED_PROJECTION_SHA256_V81
        or canonical_sha256_v81(compact) != claimed
        or manifest.get("source_tensor_count") != EXPECTED_SOURCE_TENSORS_V81
        or manifest.get("source_elements") != EXPECTED_SOURCE_ELEMENTS_V81
        or manifest.get("runtime_view_count") != EXPECTED_RUNTIME_VIEWS_V81
        or manifest.get("runtime_elements") != EXPECTED_RUNTIME_ELEMENTS_V81
        or manifest.get("runtime_dtype") != "torch.bfloat16"
        or manifest.get("runtime_bytes") != EXPECTED_RUNTIME_BYTES_V81
        or manifest.get("b_scale") != 2.0
        or manifest.get("packed_a_full_duplication") is not True
        or manifest.get("packed_b_disjoint_split") is not True
    ):
        raise RuntimeError("v81 production LoRA projection manifest changed")
    projections = manifest.get("projections")
    if not isinstance(projections, list) or len(projections) != EXPECTED_RUNTIME_VIEWS_V81:
        raise RuntimeError("v81 production runtime projection coverage changed")
    keys = []
    view_bytes = []
    for item in projections:
        if not isinstance(item, Mapping):
            raise RuntimeError("v81 runtime projection row changed")
        keys.append(_nonempty_v81(item.get("runtime_key"), "runtime key"))
        shape = item.get("runtime_shape")
        if not isinstance(shape, list) or not shape:
            raise RuntimeError("v81 runtime projection shape changed")
        elements = 1
        for dimension in shape:
            elements *= _exact_int_v81(dimension, "runtime dimension", 1)
        view_bytes.append(elements * 2)
    if len(set(keys)) != len(keys) or sum(view_bytes) != EXPECTED_RUNTIME_BYTES_V81:
        raise RuntimeError("v81 runtime projection storage coverage changed")
    return {
        "projection_sha256": claimed,
        "runtime_view_count": len(keys),
        "runtime_bytes": sum(view_bytes),
        "maximum_runtime_view_bytes": max(view_bytes),
        "runtime_keys_sha256": canonical_sha256_v81(sorted(keys)),
    }


def build_transport_plan_v81(
    manifest: Mapping[str, Any],
    capability: Mapping[str, Any],
    *,
    requested_dtype: str = "auto",
    model_dtype: str = "bfloat16",
) -> dict[str, Any]:
    projection = validate_projection_manifest_v81(manifest)
    supported = capability.get("supported_lora_dtypes")
    if (
        not isinstance(supported, list)
        or capability.get("byte_lower_than_bfloat16_supported") is not False
        or capability.get("fp8_lora_execution_supported") is not False
    ):
        raise RuntimeError("v81 installed capability receipt changed")
    dtype = resolve_execution_dtype_v81(requested_dtype, model_dtype, supported)
    if dtype != EXPECTED_RUNTIME_DTYPE_V81:
        raise RuntimeError(
            "v81 Qwen production transport must retain the attested BF16 execution view"
        )

    runtime_bytes = projection["runtime_bytes"]
    maximum_view_bytes = projection["maximum_runtime_view_bytes"]
    current_hbm = runtime_bytes * 3
    direct_hbm = runtime_bytes
    compact = {
        "schema": PLAN_SCHEMA_V81,
        "canonical_authority": {
            "dtype": "float32",
            "location": "cpu",
            "tensor_count": EXPECTED_SOURCE_TENSORS_V81,
            "elements": EXPECTED_SOURCE_ELEMENTS_V81,
            "bytes": EXPECTED_SOURCE_BYTES_V81,
            "roles": [
                "canonical_master",
                "perturbation_and_update_arithmetic",
                "optimizer_state",
                "checkpoint_authority",
            ],
        },
        "execution_view": {
            "requested_dtype": _normalize_dtype_v81(requested_dtype),
            "model_compute_dtype": _normalize_dtype_v81(model_dtype),
            "resolved_dtype": dtype,
            "view_count": projection["runtime_view_count"],
            "elements": EXPECTED_RUNTIME_ELEMENTS_V81,
            "persistent_device_bytes": runtime_bytes,
            "projection_sha256": projection["projection_sha256"],
            "runtime_keys_sha256": projection["runtime_keys_sha256"],
        },
        "control_v71": {
            "pageable_bf16_fragment_bytes": runtime_bytes,
            "h2d_bytes": runtime_bytes,
            "h2d_copy_count": EXPECTED_RUNTIME_VIEWS_V81,
            "ephemeral_device_staging_write_bytes": runtime_bytes,
            "device_to_device_copy_payload_bytes": runtime_bytes,
            "materialization_hbm_read_write_lower_bound_bytes": current_hbm,
            "maximum_ephemeral_device_tensor_bytes": maximum_view_bytes,
            "synchronous_copy_requested": True,
        },
        "challenger_pinned_direct": {
            "reused_host_pinned_bank_count": 1,
            "host_pinned_bank_bytes": runtime_bytes,
            "host_bank_storage_count": 1,
            "h2d_bytes": runtime_bytes,
            "h2d_copy_count": EXPECTED_RUNTIME_VIEWS_V81,
            "direct_copy_into_existing_runtime_views": True,
            "device_staging_bytes": 0,
            "device_to_device_copy_payload_bytes": 0,
            "materialization_hbm_read_write_lower_bound_bytes": direct_hbm,
            "maximum_ephemeral_device_tensor_bytes": 0,
            "non_blocking_requires_pinned_host_source": True,
            "one_completion_event_before_exact_audit_and_publication": True,
            "double_buffering_authorized": False,
        },
        "exact_delta": {
            "h2d_bytes_saved_per_transition": 0,
            "h2d_copy_calls_saved_per_transition": 0,
            "device_to_device_payload_bytes_saved_per_transition": runtime_bytes,
            "hbm_read_write_lower_bound_bytes_saved_per_transition": runtime_bytes * 2,
            "hbm_materialization_lower_bound_fraction_saved": 2 / 3,
            "logical_transient_device_bytes_saved_at_peak": maximum_view_bytes,
            "host_staging_bytes_saved_at_peak": 0,
            "per_16_candidate_hbm_bytes_saved": runtime_bytes * 2 * 16,
        },
        "rejected_precision_alternatives": {
            "float16": "supported but byte-neutral and semantically different",
            "float8": "not accepted by installed vLLM LoRADType",
            "int8_or_int4": "not accepted by installed vLLM LoRADType",
            "fp32_runtime": "doubles runtime-view and H2D bytes",
        },
        "non_overlap": {
            "fp32_host_ownership": "specialist-0j5.19/V72",
            "structured_noise_and_update_generation": "specialist-0j5.18/V72",
            "exact_runtime_and_base_audits": "specialist-0j5.21/V71",
            "sole_resident_slot": "specialist-0j5.16",
            "phase_telemetry": "specialist-0j5.14",
        },
    }
    compact["content_sha256_before_self_field"] = canonical_sha256_v81(compact)
    return compact


@dataclass
class StreamSafePublicationFenceV81:
    """Fail-closed CPU state machine for one reused pinned execution bank."""

    plan: Mapping[str, Any]
    phase: str = "idle"
    generation: int = 0
    bank_storage_token: str | None = None
    bank_version: int = 0
    values_sha256: str | None = None
    event_token: str | None = None
    stream_token: str | None = None
    candidate_id: str | None = None
    poisoned: bool = False
    receipts: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.plan.get("schema") != PLAN_SCHEMA_V81:
            raise RuntimeError("v81 publication fence received another transport plan")
        compact = dict(self.plan)
        claimed = compact.pop("content_sha256_before_self_field", None)
        if claimed != canonical_sha256_v81(compact):
            raise RuntimeError("v81 transport plan self-hash changed")

    def _require_live(self) -> None:
        if self.poisoned:
            raise RuntimeError("v81 adapter transport is terminally poisoned")

    def _receipt(self, action: str, **fields: Any) -> dict[str, Any]:
        receipt = {
            "schema": FENCE_SCHEMA_V81,
            "action": action,
            "phase": self.phase,
            "generation": self.generation,
            "bank_version": self.bank_version,
            **fields,
        }
        receipt["content_sha256_before_self_field"] = canonical_sha256_v81(receipt)
        self.receipts.append(receipt["content_sha256_before_self_field"])
        return receipt

    def stage(
        self,
        *,
        generation: int,
        candidate_id: str,
        bank_storage_token: str,
        values_sha256: str,
    ) -> dict[str, Any]:
        self._require_live()
        if self.phase != "idle":
            raise RuntimeError("v81 pinned bank cannot be reused before retirement")
        generation = _exact_int_v81(generation, "generation", 1)
        if generation != self.generation + 1:
            raise RuntimeError("v81 transport generation is not monotonic")
        token = _nonempty_v81(bank_storage_token, "bank storage token")
        if self.bank_storage_token is not None and token != self.bank_storage_token:
            raise RuntimeError("v81 reused pinned bank storage changed")
        self.bank_storage_token = token
        self.generation = generation
        self.bank_version += 1
        self.candidate_id = _nonempty_v81(candidate_id, "candidate id")
        self.values_sha256 = _sha256_v81(values_sha256, "projected values SHA-256")
        self.event_token = None
        self.stream_token = None
        self.phase = "staged"
        return self._receipt(
            "stage",
            candidate_id=self.candidate_id,
            bank_storage_token=self.bank_storage_token,
            values_sha256=self.values_sha256,
            host_pinned_bank_bytes=self.plan["challenger_pinned_direct"][
                "host_pinned_bank_bytes"
            ],
        )

    def issue_async_copies(
        self,
        *,
        pinned: bool,
        non_blocking: bool,
        direct_to_runtime_views: bool,
        copy_count: int,
        h2d_bytes: int,
        device_staging_bytes: int,
        stream_token: str,
        event_token: str,
    ) -> dict[str, Any]:
        self._require_live()
        if self.phase != "staged":
            raise RuntimeError("v81 async copies were issued outside staged state")
        expected = self.plan["challenger_pinned_direct"]
        if pinned is not True or non_blocking is not True:
            raise RuntimeError("v81 nonblocking H2D requires a pinned host bank")
        if direct_to_runtime_views is not True:
            raise RuntimeError("v81 challenger forbids an intermediate device bank")
        if (
            _exact_int_v81(copy_count, "copy count", 1) != expected["h2d_copy_count"]
            or _exact_int_v81(h2d_bytes, "H2D bytes", 1) != expected["h2d_bytes"]
            or _exact_int_v81(device_staging_bytes, "device staging bytes") != 0
        ):
            raise RuntimeError("v81 partial or byte-mismatched async copy set")
        self.stream_token = _nonempty_v81(stream_token, "stream token")
        self.event_token = _nonempty_v81(event_token, "event token")
        self.phase = "copies_inflight"
        return self._receipt(
            "issue_async_copies",
            pinned=True,
            non_blocking=True,
            direct_to_runtime_views=True,
            copy_count=copy_count,
            h2d_bytes=h2d_bytes,
            device_staging_bytes=0,
            stream_token=self.stream_token,
            event_token=self.event_token,
        )

    def observe_completion(self, *, event_token: str, complete: bool) -> dict[str, Any]:
        self._require_live()
        if self.phase != "copies_inflight":
            raise RuntimeError("v81 completion observed outside inflight state")
        if _nonempty_v81(event_token, "event token") != self.event_token:
            raise RuntimeError("v81 stale or foreign completion event")
        if complete is not True:
            return self._receipt("completion_pending", event_token=self.event_token)
        self.phase = "copies_complete"
        return self._receipt("copies_complete", event_token=self.event_token)

    def exact_runtime_audit(
        self,
        *,
        event_token: str,
        runtime_values_sha256: str,
        d2h_bytes: int,
        d2h_calls: int,
    ) -> dict[str, Any]:
        self._require_live()
        if self.phase != "copies_complete":
            raise RuntimeError("v81 runtime audit preceded copy completion")
        if _nonempty_v81(event_token, "event token") != self.event_token:
            raise RuntimeError("v81 runtime audit used a stale event")
        runtime_sha = _sha256_v81(runtime_values_sha256, "runtime values SHA-256")
        if runtime_sha != self.values_sha256:
            raise RuntimeError("v81 exact runtime execution view differs from staging")
        if (
            _exact_int_v81(d2h_bytes, "D2H bytes", 1)
            != EXPECTED_RUNTIME_BYTES_V81
            or _exact_int_v81(d2h_calls, "D2H calls", 1) != 1
        ):
            raise RuntimeError("v81 exact V71 readback coverage changed")
        self.phase = "exact_audited"
        return self._receipt(
            "exact_runtime_audit",
            event_token=self.event_token,
            runtime_values_sha256=runtime_sha,
            d2h_bytes=d2h_bytes,
            d2h_calls=d2h_calls,
        )

    def publish(self) -> dict[str, Any]:
        self._require_live()
        if self.phase != "exact_audited":
            raise RuntimeError("v81 adapter cannot publish before exact audit")
        self.phase = "published"
        return self._receipt(
            "publish",
            candidate_id=self.candidate_id,
            event_token=self.event_token,
            generation_may_begin=True,
        )

    def retire(self, *, candidate_id: str) -> dict[str, Any]:
        self._require_live()
        if self.phase != "published" or candidate_id != self.candidate_id:
            raise RuntimeError("v81 adapter cannot retire another/unpublished candidate")
        retired = self.candidate_id
        self.phase = "idle"
        self.values_sha256 = None
        self.event_token = None
        self.stream_token = None
        self.candidate_id = None
        return self._receipt(
            "retire",
            retired_candidate_id=retired,
            bank_reuse_authorized=True,
        )

    def recover_uncertain(
        self,
        *,
        exact_restore_succeeded: bool,
        restored_master_sha256: str | None,
        expected_master_sha256: str,
    ) -> dict[str, Any]:
        self._require_live()
        expected = _sha256_v81(expected_master_sha256, "expected master SHA-256")
        restored = None
        if restored_master_sha256 is not None:
            restored = _sha256_v81(restored_master_sha256, "restored master SHA-256")
        if exact_restore_succeeded is True and restored == expected:
            previous_phase = self.phase
            self.phase = "idle"
            self.values_sha256 = None
            self.event_token = None
            self.stream_token = None
            self.candidate_id = None
            return self._receipt(
                "uncertain_copy_exact_restore",
                previous_phase=previous_phase,
                restored_master_sha256=restored,
                reward_or_update_accepted=False,
                bank_reuse_authorized=True,
            )
        self.poisoned = True
        self.phase = "poisoned"
        return self._receipt(
            "uncertain_copy_poison",
            restored_master_sha256=restored,
            expected_master_sha256=expected,
            reward_or_update_accepted=False,
            bank_reuse_authorized=False,
        )

    def final_receipt(self) -> dict[str, Any]:
        if self.phase != "idle" or self.poisoned:
            raise RuntimeError("v81 final transport receipt requires clean idle state")
        return self._receipt(
            "final",
            completed_generations=self.generation,
            receipt_chain_sha256=canonical_sha256_v81(self.receipts),
            clean_idle=True,
        )
