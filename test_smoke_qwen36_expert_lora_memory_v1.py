from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import smoke_qwen36_expert_lora_memory_v1 as smoke


BINDINGS = {
    "causal_conv1d_fn": None,
    "causal_conv1d_update": "synthetic.torch_causal_conv1d_update",
    "chunk_gated_delta_rule": "synthetic.chunk_gated_delta_rule",
    "recurrent_gated_delta_rule": "synthetic.torch_recurrent_gated_delta_rule",
}


def _synthetic_contract_value() -> dict:
    results = []
    for index in range(4):
        gpu = {"index": index, "uuid": f"GPU-SYNTHETIC-{index}"}
        probe_gpu = {"physical_index": index, "worker_visible_index": 0}
        benchmarks = {}
        for sequence_length in (128, 2048):
            reference = {
                "tokens_per_second": 100.0,
                "milliseconds_per_iteration": 10.0,
                "peak_allocated_bytes": 1000,
                "baseline_allocated_bytes": 500,
                "peak_reserved_bytes": 1200,
                "baseline_reserved_bytes": 600,
            }
            hybrid = {
                "tokens_per_second": 120.0,
                "milliseconds_per_iteration": 8.0,
                "peak_allocated_bytes": 900,
                "baseline_allocated_bytes": 500,
                "peak_reserved_bytes": 1100,
                "baseline_reserved_bytes": 600,
            }
            benchmarks[str(sequence_length)] = {
                "hybrid_training": hybrid,
                "torch_reference": reference,
                "hybrid_throughput_speedup": 1.2,
                "hybrid_peak_allocated_bytes_ratio": 0.9,
            }
        probes = {
            name: {"status": "ok", "gpu": dict(probe_gpu)}
            for name in (
                "causal_fallback",
                "gated_delta",
                "qwen35_moe_hybrid_module",
            )
        }
        probes["qwen35_moe_hybrid_module"]["benchmarks"] = benchmarks
        results.append({
            "gpu": gpu,
            "probes": probes,
        })
    return {
        "schema": "synthetic-fast-contract-v1",
        "content_sha256_before_self_field": "a" * 64,
        "selected_fast_or_fallback": {
            "selected": "hybrid_training",
            "selected_bindings": BINDINGS,
            "hybrid_training_path_runtime_validated_on_all_four_gpus": True,
            "hybrid_path_validation_failures": [],
            "material_improvement_gate": {"passed_on_all_four_gpus": True},
            "training_launch_authorized": False,
        },
        "gpu_execution": {
            "gpu_baseline_checks": {
                "all_match_architecture_contract": True,
                "per_gpu": [{"all_match": True} for _ in range(4)],
            },
            "per_gpu_results": results,
        },
    }


def _module(tmp_path: Path, value: dict):
    output = tmp_path / "fast_contract.json"
    output.write_text("synthetic sealed contract\n", encoding="utf-8")
    return SimpleNamespace(
        SCHEMA="synthetic-fast-contract-v1",
        HYBRID_TRAINING_BINDINGS=BINDINGS,
        OUTPUT=output,
        ROOT=tmp_path,
        build=lambda check: value if check else pytest.fail("check=True required"),
        file_sha256=lambda path: hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_fast_policy_validation_accepts_exact_four_gpu_synthetic_receipt(tmp_path):
    value = _synthetic_contract_value()
    receipt = smoke.validate_fast_kernel_training_policy(_module(tmp_path, value))
    assert receipt["selected"] == "hybrid_training"
    assert receipt["selected_bindings"] == BINDINGS
    assert receipt["all_four_physical_gpus_revalidated"] is True


def test_fast_policy_validation_rejects_hardware_baseline_drift(tmp_path):
    value = _synthetic_contract_value()
    value["gpu_execution"]["gpu_baseline_checks"]["per_gpu"][2][
        "all_match"
    ] = False
    with pytest.raises(RuntimeError, match="GPU baseline"):
        smoke.validate_fast_kernel_training_policy(_module(tmp_path, value))


def test_fast_policy_validation_rejects_probe_gpu_misbinding(tmp_path):
    value = _synthetic_contract_value()
    value["gpu_execution"]["per_gpu_results"][3]["probes"]["gated_delta"][
        "gpu"
    ]["physical_index"] = 0
    with pytest.raises(RuntimeError, match="did not run on physical GPU 3"):
        smoke.validate_fast_kernel_training_policy(_module(tmp_path, value))


def test_fast_policy_validation_rejects_unselected_or_failed_hybrid(tmp_path):
    value = _synthetic_contract_value()
    value["selected_fast_or_fallback"]["selected"] = "torch_fallback"
    with pytest.raises(RuntimeError, match="does not authorize"):
        smoke.validate_fast_kernel_training_policy(_module(tmp_path, value))


def test_fast_policy_validation_rejects_negative_derived_benchmark_ratios(tmp_path):
    value = _synthetic_contract_value()
    benchmark = value["gpu_execution"]["per_gpu_results"][0]["probes"][
        "qwen35_moe_hybrid_module"
    ]["benchmarks"]["2048"]
    benchmark["hybrid_throughput_speedup"] = -1.0
    benchmark["hybrid_peak_allocated_bytes_ratio"] = -1.0
    with pytest.raises(RuntimeError, match="invalid fast-kernel seq-2048 derived ratios"):
        smoke.validate_fast_kernel_training_policy(_module(tmp_path, value))


def test_fast_policy_validation_rejects_forged_positive_derived_ratio(tmp_path):
    value = _synthetic_contract_value()
    benchmark = value["gpu_execution"]["per_gpu_results"][1]["probes"][
        "qwen35_moe_hybrid_module"
    ]["benchmarks"]["2048"]
    benchmark["hybrid_throughput_speedup"] = 9.0
    with pytest.raises(RuntimeError, match="derived ratios are inconsistent"):
        smoke.validate_fast_kernel_training_policy(_module(tmp_path, value))
