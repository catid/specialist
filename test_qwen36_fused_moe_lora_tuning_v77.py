#!/usr/bin/env python3

import copy
import json
import math
import subprocess
import sys

import pytest

import build_qwen36_fused_moe_lora_tuning_preregistration_v77 as builder
import qwen36_fused_moe_lora_tuning_v77 as subject


def frozen():
    return json.loads(builder.OUTPUT.read_text(encoding="utf-8"))


def resign(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = subject.canonical_sha256_v77(value)
    return value


def config_bundle(precision="fp8_serialized"):
    plan = frozen()
    return subject.build_config_bundle_manifest_v77(
        precision,
        subject.build_default_config_documents_v77(),
        directory_id=f"v77-{precision.replace('_', '-')}-0123456789abcdef",
        source_bundle_sha256=plan["installed_source"]["bundle_sha256"],
        selection_receipt_sha256="a" * 64,
    )


def arm(tokens_per_second, token_hash, peak_vram):
    return {
        "aggregate_generated_tokens_per_second": tokens_per_second,
        "aggregate_token_hash_sha256": token_hash,
        "per_gpu": [
            {
                "physical_gpu": gpu,
                "gpu_utilization_percent": 55.0 + gpu,
                "memory_activity_percent": 30.0 + gpu,
                "power_watts": 300.0 + gpu,
                "peak_vram_mib": peak_vram + gpu,
                "generated_tokens_per_second": tokens_per_second / 4,
            }
            for gpu in range(4)
        ],
    }


def valid_receipt(precision="fp8_serialized"):
    plan = frozen()
    ratios = [1.03, 1.04, 1.025]
    pairs = []
    for index, ratio in enumerate(ratios):
        token_hash = f"{index + 1:064x}"
        default_tps = 100.0 + index
        pairs.append(
            {
                "replicate": index,
                "order": "default_then_tuned" if index % 2 == 0 else "tuned_then_default",
                "physical_gpus": [0, 1, 2, 3],
                "default": arm(default_tps, token_hash, 1000.0),
                "tuned": arm(default_tps * ratio, token_hash, 1005.0),
            }
        )
    value = {
        "schema": subject.RECEIPT_SCHEMA_V77,
        "precision": precision,
        "preregistration_sha256": plan["content_sha256_before_self_field"],
        "environment": {
            "VLLM_USE_DEEP_GEMM": "0",
            "deepgemm_accuracy_fallback_warning_count": 0,
            "quant_config_use_deep_gemm": False,
            "deepgemm_disable_is_routed_backend_evidence": False,
            "moe_backend_argument": "triton",
            "flashinfer_autotune_enabled": False,
            "site_package_modified": False,
            "child_process_backend_attestation_passed": True,
            "installed_source_bundle_sha256": plan["installed_source"]["bundle_sha256"],
            "fp8_routed_runtime_attestation": (
                {
                    "backend_class": "Fp8MoeBackend",
                    "backend_name": "TRITON",
                    "routed_expert_owner_class": "RoutedExperts",
                    "routed_expert_owner_count": 40,
                    "quant_method_class": "Fp8MoEMethod",
                    "runtime_quant_wrapper_class": "FusedMoEModularMethod",
                    "experts_implementation_class": "TritonExperts",
                    "lora_quant_reference_count": 80,
                }
                if precision == "fp8_serialized"
                else None
            ),
        },
        "baseline": {
            "kind": "fresh_empty_default",
            "folder_was_fresh_and_empty": True,
            "base_moe_table_present": False,
            "rejected_v29_present": False,
        },
        "tuned_bundle": config_bundle(precision),
        "cache": {
            "manifest_before_measurement_sha256": "b" * 64,
            "manifest_after_measurement_sha256": "b" * 64,
            "warmup_compiled_kernel_names": list(subject.RELEVANT_JIT_KERNELS_V77),
            "inference_time_jit_messages": [],
            "missing_config_messages": [],
            "fallback_messages": [],
        },
        "replicates": pairs,
        "throughput_gate": {
            "paired_tuned_over_default_ratios": ratios,
            "median_ratio": 1.03,
            "paired_bootstrap_95pct_lower_bound": 1.001,
            "bootstrap_draw_plan_sha256": "c" * 64,
            "pass": True,
        },
        "token_identity_gate": {
            "pair_count": 3,
            "exact_pair_count": 3,
            "all_pairs_exact": True,
        },
        "semantic_validation_gate": {
            "selection_frozen_before_access": True,
            "source_disjoint": True,
            "paired_point_delta": 0.0,
            "paired_95pct_lower_bound": -0.001,
            "pass": True,
        },
        "protected_ood_gate": {
            "selection_frozen_before_access": True,
            "one_shot": True,
            "aggregate_95pct_lower_bound": -0.004,
            "worst_stratum_point_delta": -0.009,
            "new_safety_failures": 0,
            "pass": True,
        },
        "resource_gate": {
            "maximum_peak_vram_ratio": 1008.0 / 1003.0,
            "pass": True,
        },
        "authority": {
            "measurement_only": True,
            "training_or_model_update_performed": False,
            "dataset_mutation_performed": False,
            "checkpoint_or_config_promoted": False,
        },
    }
    return resign(value)


def test_v77_frozen_preregistration_rebuilds_and_checks():
    value = frozen()
    assert builder.build_preregistration_v77() == value
    assert subject.validate_preregistration_v77(value) == value
    completed = subprocess.run(
        [sys.executable, str(builder.__file__), "--check"],
        cwd=builder.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout)["gpu_launch"] is False


def test_v77_builder_does_not_import_torch_vllm_or_initialize_cuda():
    code = """
import json, sys
import build_qwen36_fused_moe_lora_tuning_preregistration_v77 as b
b.build_preregistration_v77()
print(json.dumps({"torch": "torch" in sys.modules, "vllm": "vllm" in sys.modules}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=builder.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(completed.stdout) == {"torch": False, "vllm": False}


def test_v77_actual_v73_through_v76_evidence_is_bound_and_fail_closed():
    evidence = frozen()["observed_evidence"]
    assert set(evidence["run_inventories"]) == {
        "v73_wave1", "v73_wave2", "v74", "v75", "v76"
    }
    assert evidence["run_inventories"]["v76"]["file_count"] == 9
    assert all(
        row["file_count"] == 10
        for label, row in evidence["run_inventories"].items()
        if label != "v76"
    )
    assert evidence["v73"]["bf16_actor_count"] == 4
    assert evidence["v73"]["fp8_actor_count"] == 4
    assert evidence["v73"]["fp8_over_bf16_runtime_ratio"] > 1.09
    assert evidence["v74"]["deepgemm_e8m0_enabled_actor_count"] == 4
    assert evidence["v75"]["explicit_deepgemm_disable_actor_count"] == 4
    assert evidence["v75"]["deepgemm_warning_actor_count"] == 4
    assert evidence["v75"]["deepgemm_e8m0_enabled_actor_count"] == 0
    assert evidence["v75"]["fail_closed_environment_pass"] is False
    assert evidence["v75"]["missing_lora_config_messages"] == 24
    assert set(evidence["v75"]["inference_jit_actor_counts"].values()) == {4}
    assert evidence["v76"]["environment_preflight_pass"] is True
    assert evidence["v76"]["deepgemm_disable"]["deepgemm_warning_actor_count"] == 0
    assert evidence["v76"]["deepgemm_disable"]["deepgemm_e8m0_enabled_actor_count"] == 0
    assert evidence["v76"]["routed_backend"] == {
        "moe_backend_argument": "triton",
        "backend_class": "Fp8MoeBackend",
        "backend_name": "TRITON",
        "routed_expert_owner_class": "RoutedExperts",
        "routed_expert_owner_count_per_actor": 40,
        "quant_method_class": "Fp8MoEMethod",
        "runtime_quant_wrapper_class": "FusedMoEModularMethod",
        "experts_implementation_class": "TritonExperts",
        "lora_quant_reference_count_per_actor": 80,
        "backend_log_actor_count": 4,
    }
    assert evidence["v76"]["stale_receipt_claim"] == {
        "field": "explicit_kernel_environment.explicit_cutlass_path_requested_without_runtime_fallback",
        "true_actor_count": 4,
        "trusted_for_backend_identity": False,
        "contradicted_by_live_runtime_audit": True,
    }
    blocker = frozen()["blockers"][0]
    assert blocker["bead"] == "specialist-0j5.22"
    assert blocker["kind"] == "cpu_only_authority_scope"
    assert blocker["fail_closed"] is True
    assert blocker["resolved"] is False
    resolution = frozen()["environment_resolution"]
    assert resolution["resolved_for_bound_v76_baseline"] is True
    assert resolution["legacy_cutlass_request_field_is_not_backend_evidence"] is True
    assert resolution["independent_routed_backend_gate"]["live_backend_name"] == "TRITON"


def test_v77_v29_negative_table_is_exact_and_forbidden():
    rejected = frozen()["rejected_prior_table"]
    assert rejected["identity"] == subject.V29_SELECTED_TABLE_V77
    assert rejected["all_five_latency_endpoints_failed"] is True
    assert rejected["global_geometric_mean_tuned_over_default_speedup"] < 1.0
    assert rejected["reuse_forbidden"] is True


def test_v77_six_exact_filenames_and_precision_isolation():
    names = subject.required_filenames_v77()
    assert list(names) == list(subject.OP_TYPES_V77)
    assert len(set(names.values())) == 6
    assert names["expand"].endswith("_EXPAND_TRUE.json")
    assert names["shrink"].endswith("_SHRINK.json")
    assert names["fused_moe_lora_w13_shrink"].endswith(
        "_FUSED_MOE_LORA_W13_SHRINK.json"
    )
    isolation = frozen()["tuning_plan"]["precision_isolation"]
    assert isolation["separate_config_directories_required"] is True
    assert isolation["cross_precision_causal_claims_forbidden"] is True


def test_v77_operation_inventory_covers_fused_and_actual_dense_shapes():
    inventory = subject.operation_inventory_v77()
    assert list(inventory) == list(subject.OP_TYPES_V77)
    assert len(inventory["shrink"]["module_cases"]) == 8
    assert len(inventory["expand"]["module_cases"]) == 8
    dense_inputs = {row["input"] for row in inventory["shrink"]["module_cases"]}
    dense_outputs = {row["max_output"] for row in inventory["expand"]["module_cases"]}
    assert dense_inputs == {512, 2048, 4096}
    assert dense_outputs == {32, 256, 512, 2048, 8192}
    assert inventory["fused_moe_lora_w13_shrink"]["lookup_cases"][0] == {
        "id": "s2_h2048_r32_i512",
        "num_slices": 2,
        "hidden_size": 2048,
        "rank": 32,
        "moe_intermediate_size": 512,
    }
    assert inventory["fused_moe_lora_w2_expand"]["lookup_cases"][0]["num_slices"] == 1


def test_v77_loader_hierarchy_reproduces_installed_fused_shrink_quirk():
    dense = subject.loader_lookup_key_v77(
        "shrink", max_loras=1, num_slices=2, m=68,
        hidden_size=2048, rank=32,
    )
    fused_shrink = subject.loader_lookup_key_v77(
        "fused_moe_lora_w13_shrink", max_loras=1, num_slices=2, m=68,
        hidden_size=2048, rank=32, moe_intermediate_size=512,
    )
    fused_expand = subject.loader_lookup_key_v77(
        "fused_moe_lora_w13_expand", max_loras=1, num_slices=2, m=68,
        hidden_size=2048, rank=32, moe_intermediate_size=512,
    )
    assert dense == ("1", "2", "68", "2048", "32")
    assert fused_shrink == ("1", "2", "68", "32", "2048", "512")
    assert fused_expand == fused_shrink


@pytest.mark.parametrize("op_type", subject.OP_TYPES_V77)
def test_v77_default_document_covers_every_exact_shape_and_M(op_type):
    document = subject.build_config_document_v77(
        op_type, subject.default_selections_v77(op_type)
    )
    for case in subject.operation_inventory_v77()[op_type]["lookup_cases"]:
        for m in subject.M_GRID_V77:
            selected = subject.lookup_config_document_v77(
                document,
                op_type,
                max_loras=1,
                num_slices=case["num_slices"],
                m=m,
                hidden_size=case["hidden_size"],
                rank=32,
                moe_intermediate_size=case["moe_intermediate_size"],
            )
            subject.validate_kernel_config_v77(op_type, selected)


def test_v77_nearest_M_matches_loader_but_num_slices_is_exact():
    document = subject.build_config_document_v77(
        "shrink", subject.default_selections_v77("shrink")
    )
    nearest = subject.lookup_config_document_v77(
        document, "shrink", max_loras=1, num_slices=2, m=67,
        hidden_size=2048, rank=32,
    )
    exact = subject.lookup_config_document_v77(
        document, "shrink", max_loras=1, num_slices=2, m=68,
        hidden_size=2048, rank=32,
    )
    assert nearest == exact
    with pytest.raises(RuntimeError, match="num_slices"):
        subject.lookup_config_document_v77(
            document, "shrink", max_loras=1, num_slices=4, m=68,
            hidden_size=2048, rank=32,
        )


def test_v77_config_validation_rejects_unsafe_fields_values_and_gaps():
    config = subject.default_kernel_config_v77("fused_moe_lora_w13_shrink", 68)
    changed = copy.deepcopy(config)
    changed["split_k"] = 8
    with pytest.raises(RuntimeError, match="unsafe"):
        subject.validate_kernel_config_v77("fused_moe_lora_w13_shrink", changed)
    changed = copy.deepcopy(config)
    changed["arbitrary_code"] = 1
    with pytest.raises(RuntimeError, match="keys"):
        subject.validate_kernel_config_v77("fused_moe_lora_w13_shrink", changed)
    selections = subject.default_selections_v77("fused_moe_lora_w13_shrink")
    with pytest.raises(RuntimeError, match="cover"):
        subject.build_config_document_v77(
            "fused_moe_lora_w13_shrink", selections[:-1]
        )


@pytest.mark.parametrize("precision", subject.PRECISIONS_V77)
def test_v77_bundle_roundtrip_is_exact_and_content_addressed(precision):
    value = config_bundle(precision)
    assert subject.validate_config_bundle_manifest_v77(value) == value
    assert len(value["files"]) == 6
    assert value["precision"] == precision
    assert value["rejected_v29_selected_table_present"] is False


def test_v77_bundle_rejects_traversal_and_every_V29_identity():
    documents = subject.build_default_config_documents_v77()
    plan = frozen()
    with pytest.raises(RuntimeError, match="unsafe config directory"):
        subject.build_config_bundle_manifest_v77(
            "fp8_serialized", documents,
            directory_id="../v29",
            source_bundle_sha256=plan["installed_source"]["bundle_sha256"],
            selection_receipt_sha256="a" * 64,
        )
    for forbidden in (
        subject.V29_SELECTED_TABLE_V77["file_sha256"],
        subject.V29_SELECTED_TABLE_V77["content_sha256"],
        subject.V29_SELECTED_TABLE_V77["loaded_config_sha256"],
    ):
        value = config_bundle()
        value["selection_receipt_sha256"] = forbidden
        resign(value)
        with pytest.raises(RuntimeError, match="V29"):
            subject.validate_config_bundle_manifest_v77(value)


def test_v77_preregistration_semantic_tamper_fails_even_if_resigned():
    value = frozen()
    value["tuning_plan"]["base_moe_isolation"]["V29_selected_table_reuse_forbidden"] = False
    resign(value)
    with pytest.raises(RuntimeError, match="contract changed"):
        subject.validate_preregistration_v77(value)
    value = frozen()
    value["authority"]["gpu_launch"] = True
    resign(value)
    with pytest.raises(RuntimeError, match="authority"):
        subject.validate_preregistration_v77(value)


@pytest.mark.parametrize("precision", subject.PRECISIONS_V77)
def test_v77_valid_synthetic_measurement_covers_all_paired_gates(precision):
    receipt = valid_receipt(precision)
    assert subject.validate_measurement_receipt_v77(frozen(), receipt) == receipt
    with pytest.raises(RuntimeError, match="promotion blocked"):
        subject.validate_promotion_v77(frozen(), receipt)


def test_v77_measurement_fail_closed_environment_cache_and_pairing_mutations():
    mutations = [
        lambda value: value["environment"].__setitem__(
            "deepgemm_accuracy_fallback_warning_count", 1
        ),
        lambda value: value["environment"].__setitem__(
            "quant_config_use_deep_gemm", True
        ),
        lambda value: value["environment"].__setitem__(
            "deepgemm_disable_is_routed_backend_evidence", True
        ),
        lambda value: value["environment"]["fp8_routed_runtime_attestation"].__setitem__(
            "backend_name", "VLLM_CUTLASS"
        ),
        lambda value: value["baseline"].__setitem__("folder_was_fresh_and_empty", False),
        lambda value: value["cache"]["inference_time_jit_messages"].append("fused_moe_kernel"),
        lambda value: value["cache"].__setitem__(
            "manifest_after_measurement_sha256", "d" * 64
        ),
        lambda value: value["replicates"][1].__setitem__("order", "default_then_tuned"),
    ]
    for mutate in mutations:
        value = valid_receipt()
        mutate(value)
        resign(value)
        with pytest.raises(RuntimeError):
            subject.validate_measurement_receipt_v77(frozen(), value)


def test_v77_measurement_rejects_performance_semantic_ood_and_resource_failures():
    mutations = [
        lambda value: value["throughput_gate"].__setitem__("median_ratio", 1.0),
        lambda value: value["throughput_gate"].__setitem__(
            "paired_bootstrap_95pct_lower_bound", 0.99
        ),
        lambda value: value["token_identity_gate"].__setitem__("exact_pair_count", 2),
        lambda value: value["semantic_validation_gate"].__setitem__(
            "paired_95pct_lower_bound", -0.003
        ),
        lambda value: value["protected_ood_gate"].__setitem__(
            "new_safety_failures", 1
        ),
        lambda value: value["resource_gate"].__setitem__(
            "maximum_peak_vram_ratio", 1.02
        ),
        lambda value: value["replicates"][0]["tuned"]["per_gpu"].pop(),
    ]
    for mutate in mutations:
        value = valid_receipt()
        mutate(value)
        resign(value)
        with pytest.raises(RuntimeError):
            subject.validate_measurement_receipt_v77(frozen(), value)


def test_v77_every_metric_is_finite_and_positive():
    value = valid_receipt()
    value["replicates"][0]["tuned"]["per_gpu"][0]["memory_activity_percent"] = math.nan
    resign(value)
    with pytest.raises(RuntimeError, match="telemetry"):
        subject.validate_measurement_receipt_v77(frozen(), value)
