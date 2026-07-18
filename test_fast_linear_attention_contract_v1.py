from __future__ import annotations

import json
from pathlib import Path

import torch

import build_fast_linear_attention_contract_v1 as contract


def _checked_contract() -> dict:
    value = json.loads(contract.OUTPUT.read_text(encoding="utf-8"))
    unsigned = dict(value)
    declared = unsigned.pop("content_sha256_before_self_field")
    assert contract.canonical_sha256(unsigned) == declared
    return value


def test_contract_is_synthetic_only_and_does_not_authorize_training():
    value = _checked_contract()
    assert value["schema"] == contract.SCHEMA
    assert value["authority"] == {
        "synthetic_tensors_only": True,
        "model_or_adapter_weights_loaded": False,
        "datasets_or_training_rows_opened": False,
        "protected_holdout_ood_terminal_incident_or_manual_review_opened": False,
        "optimizer_created": False,
        "training_launched": False,
        "gpu_mutation_beyond_ephemeral_synthetic_allocations": False,
    }


def test_core_distributions_and_recorded_sources_match_architecture_baseline():
    environment = _checked_contract()["environment_integrity"]
    assert environment["all_core_distributions_preserved"] is True
    assert environment["all_fast_distribution_versions_exact"] is True
    assert set(environment["core_baseline_checks"]) == set(
        contract.CORE_DISTRIBUTIONS
    )
    assert all(
        check["version_matches"]
        and check["direct_url_matches"]
        and check["vcs_commit_matches"]
        for check in environment["core_baseline_checks"].values()
    )
    assert all(
        check["matches"]
        for check in environment["baseline_source_checks"].values()
    )
    assert all(
        check["matches"]
        for check in environment["fast_distribution_version_checks"].values()
    )
    assert environment["torch_runtime_checks"]["git_version_matches"] is True
    for name, expected in contract.FAST_DISTRIBUTIONS.items():
        assert environment["distributions"][name]["version"] == expected
    for receipt in environment["source_receipts"].values():
        path = Path(receipt["path"])
        if not path.is_absolute():
            path = contract.ROOT / path
        assert contract.file_sha256(path) == receipt["sha256"]


def test_transformers_qwen35_moe_import_fast_path_is_proven_but_not_overclaimed():
    availability = _checked_contract()["transformers_qwen35_moe_fast_path"]
    assert availability["transformers_is_causal_conv1d_available"] is True
    assert availability["transformers_is_flash_linear_attention_available"] is True
    assert availability["qwen35_moe_module_is_fast_path_available"] is True
    assert availability["all_required_bindings_imported"] is True
    assert availability["availability_is_import_only_not_runtime_validation"] is True
    assert all(item["available"] for item in availability["bindings"].values())


def test_all_four_gpus_have_parity_throughput_and_peak_vram_results():
    execution = _checked_contract()["gpu_execution"]
    assert execution["selection_mode"] == (
        "parallel_one_isolated_worker_per_physical_gpu_with_sequential_probes"
    )
    results = execution["per_gpu_results"]
    assert [item["gpu"]["index"] for item in results] == [0, 1, 2, 3]
    assert len({item["gpu"]["uuid"] for item in results}) == 4
    for item in results:
        probes = item["probes"]
        causal_fast = probes["causal_fast"]
        assert causal_fast["status"] == "worker_process_error"
        assert causal_fast["returncode"] < 0
        assert causal_fast["termination_signal_name"] == "SIGSEGV"

        causal_fallback = probes["causal_fallback"]
        assert causal_fallback["status"] == "ok"
        _assert_benchmark(causal_fallback["benchmark"])

        gated = probes["gated_delta"]
        assert gated["status"] == "ok"
        chunk = gated["chunk_training_kernel_parity"]
        assert chunk["all_passed"] is True
        assert chunk["forward"]["passed"] is True
        assert set(chunk["gradients"]) == {"query", "key", "value", "g", "beta"}
        assert all(metric["passed"] for metric in chunk["gradients"].values())
        recurrent = gated["fused_recurrent_kernel_parity"]
        assert recurrent["forward_all_passed"] is True
        assert recurrent["backward"]["status"] == "fast_backward_unsupported"
        assert recurrent["backward"]["error"]["type"] == "NotImplementedError"
        _assert_benchmark(gated["benchmarks"]["fast"])
        _assert_benchmark(gated["benchmarks"]["torch_fallback"])

        hybrid = probes["qwen35_moe_hybrid_module"]
        assert hybrid["status"] == "ok"
        assert hybrid["all_sequence_parity_passed"] is True
        assert hybrid["module_contract"]["realized_geometry"] == (
            contract.QWEN35_MOE_GATED_DELTA_GEOMETRY
        )
        assert hybrid["module_contract"]["hybrid"]["bindings"] == (
            contract.HYBRID_TRAINING_BINDINGS
        )
        assert hybrid["module_contract"]["hybrid"][
            "all_matched_modules_configured"
        ] is True
        for sequence_length in (128, 2048):
            parity = hybrid["parity"][str(sequence_length)]
            assert parity["all_passed"] is True
            assert parity["forward"]["passed"] is True
            assert parity["input_gradient"]["passed"] is True
            assert parity["all_parameter_gradients_present"] is True
            assert all(
                metric["passed"]
                for metric in parity["parameter_gradients"].values()
            )
            benchmark = hybrid["benchmarks"][str(sequence_length)]
            _assert_benchmark(benchmark["hybrid_training"])
            _assert_benchmark(benchmark["torch_reference"])
            assert benchmark["hybrid_throughput_speedup"] > 1.0
        assert hybrid["benchmarks"]["2048"][
            "hybrid_throughput_speedup"
        ] >= 1.10
        assert hybrid["benchmarks"]["2048"][
            "hybrid_peak_allocated_bytes_ratio"
        ] <= 0.95


def _assert_benchmark(benchmark: dict) -> None:
    assert benchmark["mode"] == "forward_plus_backward"
    assert benchmark["tokens_per_second"] > 0
    assert benchmark["milliseconds_per_iteration"] > 0
    assert benchmark["peak_allocated_bytes"] >= benchmark[
        "baseline_allocated_bytes"
    ]
    assert benchmark["peak_reserved_bytes"] >= benchmark["baseline_reserved_bytes"]


def test_hybrid_training_decision_is_explicit_and_reproducible():
    value = _checked_contract()
    decision = value["selected_fast_or_fallback"]
    assert decision["selected"] == "hybrid_training"
    assert decision["selected_training_path"] == (
        "torch_causal_conv1d_plus_fla_chunk_gated_delta_plus_torch_recurrent"
    )
    assert decision["selected_bindings"] == contract.HYBRID_TRAINING_BINDINGS
    assert decision["fast_path_import_available"] is True
    assert decision["fast_path_runtime_validated_on_all_four_gpus"] is False
    assert decision[
        "hybrid_training_path_runtime_validated_on_all_four_gpus"
    ] is True
    assert decision["fallback_is_installed_transformers_torch_implementation"] is False
    assert decision["hybrid_path_validation_failures"] == []
    assert decision["fast_components_selected"] == ["chunk_gated_delta_rule"]
    assert decision["material_improvement_gate"][
        "passed_on_all_four_gpus"
    ] is True
    assert any("causal_conv1d_fast" in reason for reason in decision["reasons"])
    assert decision["training_launch_authorized"] is False
    expected = contract._selection_decision(
        value["environment_integrity"],
        value["transformers_qwen35_moe_fast_path"],
        value["gpu_execution"]["per_gpu_results"],
    )
    assert decision == expected


def test_hybrid_policy_exactly_declares_every_training_binding():
    policy = _checked_contract()["hybrid_training_policy"]
    assert policy["scope"] == "every_Qwen3_5MoeGatedDeltaNet_below_model_root"
    assert policy["bindings"] == contract.HYBRID_TRAINING_BINDINGS
    assert policy["normalization_binding_changed"] is False
    assert policy["geometry"] == contract.QWEN35_MOE_GATED_DELTA_GEOMETRY
    assert policy["sequence_lengths_validated"] == [128, 2048]
    assert policy["training_only_no_cache_parity_scope"] is True
    assert policy["single_token_decode_not_selected_for_fast_recurrent"] is True


def test_declared_bfloat16_tolerances_and_exact_synthetic_distributions():
    value = _checked_contract()
    assert value["declared_bfloat16_tolerances"] == contract.BF16_TOLERANCES
    assert value["synthetic_workloads"] == contract.SYNTHETIC_WORKLOADS
    assert all(
        spec["dtype"] == "torch.bfloat16"
        for spec in value["synthetic_workloads"].values()
    )
    observed = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    reference = torch.tensor([1.0, 2.015625], dtype=torch.bfloat16)
    metrics = contract._comparison_metrics(
        torch,
        observed,
        reference,
        {"atol": 0.03, "rtol": 0.03},
    )
    assert metrics["passed"] is True


def test_checked_contract_matches_current_packages_sources_and_gpu_inventory():
    assert contract.build(check=True) == _checked_contract()
