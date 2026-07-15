#!/usr/bin/env python3
"""Memory-endpoint repair mechanics for the exclusive V24A retry R1."""

from __future__ import annotations

import copy

import train_eggroll_es_hybrid_backend_v24a as base


ARMS = base.ARMS
PAIRS = base.PAIRS
MODEL_LOAD_REDUCTION_THRESHOLD_R1 = 0.40
RUNTIME_INTEGRITY_KEYS_R1 = set(base.RUNTIME_INTEGRITY_KEYS) | {
    "model_load_memory_boundary_verified",
}


canonical_sha256 = base.canonical_sha256


def _seal(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_model_load_memory_preflight_r1(model_load_consumed_bytes):
    """Require exact duplicate backend loads and both reductions before scoring."""
    if (
        tuple(model_load_consumed_bytes) != ARMS
        or any(type(value) is not int or value <= 0 for value in model_load_consumed_bytes.values())
    ):
        raise RuntimeError("v24a-r1 model-load memory coverage changed")
    if model_load_consumed_bytes["bf16_a"] != model_load_consumed_bytes["bf16_b"]:
        raise RuntimeError("v24a-r1 duplicate BF16 model-load values differ")
    if model_load_consumed_bytes["hybrid_a"] != model_load_consumed_bytes["hybrid_b"]:
        raise RuntimeError("v24a-r1 duplicate hybrid model-load values differ")
    reductions = {
        "pair_a": 1.0 - (
            model_load_consumed_bytes["hybrid_a"]
            / model_load_consumed_bytes["bf16_a"]
        ),
        "pair_b": 1.0 - (
            model_load_consumed_bytes["hybrid_b"]
            / model_load_consumed_bytes["bf16_b"]
        ),
    }
    if any(value < MODEL_LOAD_REDUCTION_THRESHOLD_R1 for value in reductions.values()):
        raise RuntimeError("v24a-r1 model-load memory reduction is below 0.40")
    return _seal({
        "schema": "eggroll-es-v24a-model-load-memory-preflight-r1",
        "endpoint": "int(self.model_runner.model_memory_usage)",
        "source_assignment": "self.model_memory_usage = m.consumed_memory",
        "model_load_consumed_bytes": copy.deepcopy(model_load_consumed_bytes),
        "duplicate_bf16_values_exact": True,
        "duplicate_hybrid_values_exact": True,
        "reduction_by_pair": reductions,
        "threshold": MODEL_LOAD_REDUCTION_THRESHOLD_R1,
        "both_pairs_meet_threshold": True,
        "completed_before_reference_a_b_c_or_first_perturbation": True,
        "nvml_excluded_from_gate": True,
    })


def build_compact_summary_v24a_memory_retry_r1(
    unit_scores,
    reference_scores,
    wave_seconds,
    model_load_consumed_bytes,
    nvml_resident_bytes_diagnostic,
    panel_bundle,
    runtime_integrity,
):
    """Run unchanged V24A inference, substituting only the memory gate endpoint."""
    preflight = validate_model_load_memory_preflight_r1(model_load_consumed_bytes)
    if (
        tuple(nvml_resident_bytes_diagnostic) != ARMS
        or any(
            type(value) is not int or value <= 0
            for value in nvml_resident_bytes_diagnostic.values()
        )
        or tuple(runtime_integrity) != ARMS
        or any(
            set(runtime_integrity[arm]) != RUNTIME_INTEGRITY_KEYS_R1
            or not all(runtime_integrity[arm].values())
            for arm in ARMS
        )
    ):
        raise RuntimeError("v24a-r1 NVML diagnostic or runtime integrity changed")
    base_integrity = {
        arm: {
            key: runtime_integrity[arm][key]
            for key in base.RUNTIME_INTEGRITY_KEYS
        }
        for arm in ARMS
    }
    result = base.build_compact_summary_v24a(
        unit_scores,
        reference_scores,
        wave_seconds,
        model_load_consumed_bytes,
        panel_bundle,
        base_integrity,
    )
    pair_specs = base.prereg.build_preregistration_v24a()["pairing"]["pairs"]
    for pair in PAIRS:
        bf16_arm = pair_specs[pair]["bf16"]
        hybrid_arm = pair_specs[pair]["hybrid"]
        memory = result["pairs"][pair]["memory"]
        if (
            memory["bf16_resident_bytes"] != model_load_consumed_bytes[bf16_arm]
            or memory["hybrid_resident_bytes"] != model_load_consumed_bytes[hybrid_arm]
            or memory["reduction"] != preflight["reduction_by_pair"][pair]
            or memory["threshold"] != MODEL_LOAD_REDUCTION_THRESHOLD_R1
            or result["pairs"][pair]["memory_pass"] is not True
        ):
            raise RuntimeError("v24a-r1 base estimator memory substitution changed")
        result["pairs"][pair]["memory"] = {
            "endpoint": "vllm_model_load_consumed_bytes",
            "bf16_model_load_consumed_bytes": model_load_consumed_bytes[bf16_arm],
            "hybrid_model_load_consumed_bytes": model_load_consumed_bytes[hybrid_arm],
            "model_load_reduction": memory["reduction"],
            "threshold": MODEL_LOAD_REDUCTION_THRESHOLD_R1,
            "gate_input": True,
        }
        result["pairs"][pair]["nvml_resident_memory_diagnostic"] = {
            "bf16_nvml_resident_bytes": nvml_resident_bytes_diagnostic[bf16_arm],
            "hybrid_nvml_resident_bytes": nvml_resident_bytes_diagnostic[hybrid_arm],
            "gate_input": False,
            "excluded_from_quality_speed_memory_and_global_pass": True,
        }
    result.update({
        "schema": "eggroll-es-hybrid-backend-compact-estimator-v24a-memory-retry-r1",
        "memory_endpoint_preflight": preflight,
        "memory_gate_uses_only_vllm_model_load_consumed_bytes": True,
        "nvml_resident_bytes_retained_only_as_diagnostic": True,
        "inherited_v24a_quality_speed_bootstrap_and_decision_logic_unchanged": True,
    })
    return _seal(result)
