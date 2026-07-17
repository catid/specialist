#!/usr/bin/env python3

import copy
import json
import statistics
import subprocess
import sys

import pytest

import build_qwen36_text_only_checkpoint_preregistration_v78 as builder
import qwen36_text_only_checkpoint_v78 as subject


def frozen():
    return json.loads(builder.OUTPUT.read_text(encoding="utf-8"))


def resign(value):
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = subject.canonical_sha256_v78(value)
    return value


def candidate_manifest(precision="fp8_serialized"):
    preregistration = frozen()
    checkpoint = preregistration["checkpoints"][precision]
    language = checkpoint["categories"]["language"]
    physical_bytes = language["logical_bytes"] + 4096
    value = {
        "schema": subject.ARTIFACT_SCHEMA_V78,
        "precision": precision,
        "preregistration_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "format": "standard_huggingface_safetensors",
        "load_format": "safetensors",
        "custom_loader_or_plugin": False,
        "source_checkpoint_modified": False,
        "source_and_candidate_roots_differ": True,
        "symlink_or_external_hardlink_count": 0,
        "source_config_sha256": checkpoint["config_sha256"],
        "candidate_config_sha256": checkpoint["config_sha256"],
        "candidate_index_sha256": "a" * 64,
        "tokenizer_file_sha256": checkpoint["tokenizer_file_sha256"],
        "retained_tensor_count": language["tensor_count"],
        "retained_logical_bytes": language["logical_bytes"],
        "retained_metadata_manifest_sha256": language["manifest_sha256"],
        "retained_key_names_sha256": language["key_names_sha256"],
        "candidate_index_key_names_sha256": language["key_names_sha256"],
        "candidate_index_key_count": language["tensor_count"],
        "candidate_visual_key_count": 0,
        "candidate_mtp_key_count": 0,
        "candidate_other_key_count": 0,
        "omitted_tensor_count": checkpoint["omitted_tensor_count"],
        "omitted_logical_bytes": checkpoint["omitted_logical_bytes"],
        "omitted_manifest_sha256": checkpoint["omitted_manifest_sha256"],
        "omitted_key_names_sha256": checkpoint["omitted_key_names_sha256"],
        "payload_identity": {
            "algorithm": "sha256_per_tensor_then_canonical_manifest",
            "source_retained_payload_manifest_sha256": "b" * 64,
            "candidate_retained_payload_manifest_sha256": "b" * 64,
            "all_retained_payloads_exact": True,
        },
        "candidate_weight_files": [
            {
                "file": "model-00001-of-00001.safetensors",
                "bytes": physical_bytes,
                "sha256": "c" * 64,
            }
        ],
        "candidate_physical_weight_bytes": physical_bytes,
        "artifact_creation_authority": {
            "standard_derivative_only": True,
            "source_checkpoint_modified": False,
            "GPU_used": False,
            "model_or_adapter_updated": False,
            "dataset_or_protected_content_opened": False,
            "promotion_authorized": False,
        },
    }
    return resign(value)


def live_inventory(precision="fp8_serialized"):
    checkpoint = frozen()["checkpoints"][precision]
    return {
        "outer_model_class": "Qwen3_5MoeForConditionalGeneration",
        "language_model_class": "Qwen3_5MoeForCausalLM",
        "named_parameter_count": 813,
        "named_parameter_bytes": 35_712_084_096,
        "named_parameter_manifest_sha256": "d" * 64,
        "visual_named_parameters": [],
        "mtp_named_parameters": [],
        "stage_missing_modules": [
            {"name": "visual", "stage_name": "vision_tower"}
        ],
        "vision_wrapped_parameter_count": 333,
        "vision_wrapped_parameters_all_meta": True,
        "speculative_module_names": [],
        "loaded_checkpoint_language_manifest_sha256": checkpoint["categories"][
            "language"
        ]["manifest_sha256"],
        "LoRA_target_manifest_sha256": "e" * 64,
    }


def actor_receipt(
    arm,
    gpu,
    precision="fp8_serialized",
    replicate=0,
    load_seconds=10.0,
    tokens_per_second=100.0,
    peak_vram_mib=40_000.0,
):
    preregistration = frozen()
    manifest = candidate_manifest(precision)
    identity_seed = replicate * 10 + gpu + 1
    identity = {key: f"{identity_seed:064x}" for key in (
        "sealed_prompt_token_ids_sha256",
        "selected_logits_sha256",
        "greedy_token_ids_sha256",
        "candidate_switch_sha256",
        "reference_restore_sha256",
    )}
    value = {
        "schema": subject.ACTOR_SCHEMA_V78,
        "precision": precision,
        "arm": arm,
        "physical_gpu": gpu,
        "preregistration_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "candidate_manifest_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "engine_contract": subject.engine_contract_v78(),
        "live_inventory": live_inventory(precision),
        "tokenizer_file_sha256": preregistration["checkpoints"][precision][
            "tokenizer_file_sha256"
        ],
        "metrics": {
            "weight_load_seconds": load_seconds,
            "peak_vram_mib": peak_vram_mib,
            "post_load_vram_mib": peak_vram_mib - 100,
            "generated_tokens_per_second": tokens_per_second,
            "gpu_utilization_percent": 75.0,
            "memory_activity_percent": 55.0,
            "power_watts": 300.0,
        },
        "identity": identity,
        "request_scope": {
            "text_only": True,
            "multimodal_request_count": 0,
            "multimodal_embedding_request_count": 0,
            "speculative_request_count": 0,
            "dataset_rows_opened": 0,
            "protected_rows_opened": 0,
        },
        "cleanup": {
            "engine_shutdown_completed": True,
            "torch_process_group_destroyed": True,
            "actor_process_exited": True,
        },
    }
    return resign(value)


def paired_receipt(precision="fp8_serialized"):
    preregistration = frozen()
    manifest = candidate_manifest(precision)
    replicates = []
    for replicate in range(3):
        full = []
        candidate = []
        for gpu in subject.PHYSICAL_GPUS_V78:
            full.append(
                actor_receipt(
                    "full_checkpoint",
                    gpu,
                    precision,
                    replicate,
                    load_seconds=10.0 + replicate / 10 + gpu / 100,
                    tokens_per_second=100.0 + replicate + gpu / 10,
                    peak_vram_mib=40_000.0 + gpu,
                )
            )
            candidate.append(
                actor_receipt(
                    "text_only_derivative",
                    gpu,
                    precision,
                    replicate,
                    load_seconds=(10.0 + replicate / 10 + gpu / 100) * 0.95,
                    tokens_per_second=100.0 + replicate + gpu / 10,
                    peak_vram_mib=40_000.0 + gpu,
                )
            )
        replicates.append(
            {
                "replicate": replicate,
                "order": (
                    "full_then_derivative"
                    if replicate % 2 == 0
                    else "derivative_then_full"
                ),
                "physical_gpus": list(subject.PHYSICAL_GPUS_V78),
                "full_checkpoint": full,
                "text_only_derivative": candidate,
            }
        )
    load_ratios = []
    throughput_ratios = []
    peak_deltas = []
    for pair in replicates:
        for gpu in subject.PHYSICAL_GPUS_V78:
            full = pair["full_checkpoint"][gpu]
            candidate = pair["text_only_derivative"][gpu]
            load_ratios.append(
                candidate["metrics"]["weight_load_seconds"]
                / full["metrics"]["weight_load_seconds"]
            )
            throughput_ratios.append(
                candidate["metrics"]["generated_tokens_per_second"]
                / full["metrics"]["generated_tokens_per_second"]
            )
            peak_deltas.append(
                candidate["metrics"]["peak_vram_mib"]
                - full["metrics"]["peak_vram_mib"]
            )
    value = {
        "schema": subject.PAIRED_SCHEMA_V78,
        "precision": precision,
        "preregistration_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "candidate_manifest_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "replicates": replicates,
        "gates": {
            "load_time": {
                "paired_ratios": load_ratios,
                "median_ratio": statistics.median(load_ratios),
                "paired_bootstrap_95pct_upper_bound": 0.98,
                "bootstrap_draw_plan_sha256": "f" * 64,
                "pass": True,
            },
            "throughput": {
                "paired_ratios": throughput_ratios,
                "median_ratio": statistics.median(throughput_ratios),
                "paired_bootstrap_95pct_lower_bound": 0.99,
                "bootstrap_draw_plan_sha256": "1" * 64,
                "pass": True,
            },
            "residency": {
                "candidate_minus_full_peak_vram_mib": peak_deltas,
                "maximum_delta_mib": max(peak_deltas),
                "named_parameter_identity_exact": True,
                "pass": True,
            },
            "semantic_validation": {
                "selection_frozen_before_access": True,
                "source_disjoint": True,
                "paired_point_delta": 0.0,
                "paired_95pct_lower_bound": -0.001,
                "pass": True,
            },
            "protected_ood": {
                "selection_frozen_before_access": True,
                "one_shot": True,
                "aggregate_95pct_lower_bound": -0.004,
                "worst_stratum_point_delta": -0.009,
                "new_safety_failures": 0,
                "pass": True,
            },
        },
        "cleanup": {
            "all_four_processes_exited": True,
            "all_four_gpus_finally_idle": True,
            "foreign_compute_processes": [],
        },
        "authority": {
            "measurement_only": True,
            "source_checkpoint_modified": False,
            "model_or_adapter_updated": False,
            "dataset_mutated": False,
            "candidate_promoted": False,
        },
    }
    return resign(value)


def test_v78_frozen_preregistration_rebuilds_and_checks():
    value = frozen()
    assert builder.build_preregistration_v78() == value
    assert subject.validate_preregistration_v78(value) == value
    completed = subprocess.run(
        [sys.executable, str(builder.__file__), "--check"],
        cwd=builder.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result["gpu_launch"] is False
    assert result["candidate_artifact_created"] is False


def test_v78_builder_does_not_import_torch_vllm_or_initialize_cuda():
    code = """
import json, sys
import build_qwen36_text_only_checkpoint_preregistration_v78 as b
b.build_preregistration_v78()
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


@pytest.mark.parametrize("precision", subject.PRECISIONS_V78)
def test_v78_every_omitted_key_and_byte_is_bound(precision):
    checkpoint = frozen()["checkpoints"][precision]
    rows = checkpoint["omitted_tensors"]
    assert len(rows) == checkpoint["omitted_tensor_count"]
    assert sum(row["logical_bytes"] for row in rows) == checkpoint[
        "omitted_logical_bytes"
    ]
    assert subject.canonical_sha256_v78(rows) == checkpoint[
        "omitted_manifest_sha256"
    ]
    assert subject.canonical_sha256_v78([row["name"] for row in rows]) == checkpoint[
        "omitted_key_names_sha256"
    ]
    assert all(
        row["name"].startswith("model.visual.")
        if row["category"] == "visual"
        else row["name"].startswith("mtp.")
        for row in rows
    )


def test_v78_checkpoint_bytes_are_not_whole_file_or_live_residency_claims():
    checkpoints = frozen()["checkpoints"]
    assert checkpoints["bf16"]["omitted_logical_bytes"] == 2_582_424_032
    assert checkpoints["fp8_serialized"]["omitted_logical_bytes"] == 1_746_810_976
    assert checkpoints["bf16"]["omitted_only_files"] == []
    assert checkpoints["fp8_serialized"]["omitted_only_files"] == [
        "mtp.safetensors"
    ]
    assert checkpoints["fp8_serialized"]["mixed_language_and_omitted_files"] == [
        "outside.safetensors"
    ]
    assert all(
        item["whole_file_ignore_can_remove_all_omitted_tensors"] is False
        for item in checkpoints.values()
    )


def test_v78_four_live_actors_are_language_only():
    live = frozen()["live_fp8_evidence"]
    assert live["actor_count"] == 4
    assert live["component_names"] == ["language"]
    assert live["total_parameter_count_per_actor"] == 813
    assert live["total_logical_bytes_per_actor"] == 35_712_084_096
    assert live["visual_named_parameter_count_per_actor"] == 0
    assert live["mtp_named_parameter_count_per_actor"] == 0
    assert live["zero_multimodal_limits_log_actor_count"] == 4
    assert live["speculative_config_none_log_actor_count"] == 4
    assert subject.canonical_sha256_v78(live["run_files"]) == live[
        "run_bundle_sha256"
    ]


def test_v78_source_and_decision_reject_derivative_creation():
    value = frozen()
    findings = value["installed_source"]["source_findings"]
    assert "meta" in findings["vision_construction"]
    assert "StageMissingLayer" in findings["stage_missing_loading"]
    assert "mtp.*" in findings["mtp_target_loading"]
    assert "materializes" in findings["checkpoint_iterator_order"]
    assert value["decision"] == {
        "candidate_artifact_recommended": False,
        "candidate_artifact_created": False,
        "loader_filter_recommended": False,
        "steady_state_VRAM_opportunity": False,
        "steady_state_memory_bandwidth_opportunity": False,
        "checkpoint_storage_or_startup_IO_opportunity_only": True,
        "reason": (
            "checkpoint bytes are present, but installed vLLM and four live "
            "actors already omit visual/MTP persistent parameter residency"
        ),
    }
    assert value["supported_artifact_plan"]["candidate_creation_authorized"] is False
    assert value["authority"]["candidate_artifact_creation"] is False


def test_v78_engine_contract_disables_mm_and_speculation_without_graph_confound():
    engine = frozen()["engine_contract"]
    assert engine["limit_mm_per_prompt"] == {"image": 0, "video": 0}
    assert engine["speculative_config"] is None
    assert engine["num_speculative_tokens"] == 0
    assert engine["language_model_only"] is False
    assert frozen()["supported_artifact_plan"]["separate_future_ablation_not_bundled"][
        "language_model_only_true"
    ] is True


def test_v78_preregistration_hash_and_resigned_semantic_tampering_fail():
    value = frozen()
    value["decision"]["candidate_artifact_recommended"] = True
    with pytest.raises(RuntimeError, match="self hash"):
        subject.validate_preregistration_v78(value)
    mutations = [
        lambda item: item["decision"].__setitem__(
            "candidate_artifact_recommended", True
        ),
        lambda item: item["authority"].__setitem__("gpu_launch", True),
        lambda item: item["live_fp8_evidence"].__setitem__(
            "visual_named_parameter_count_per_actor", 1
        ),
        lambda item: item["checkpoints"]["fp8_serialized"]["omitted_tensors"][0].__setitem__(
            "logical_bytes", 1
        ),
        lambda item: item["installed_source"]["files"][0].__setitem__(
            "sha256", "0" * 64
        ),
    ]
    for mutate in mutations:
        item = frozen()
        mutate(item)
        resign(item)
        with pytest.raises(RuntimeError):
            subject.validate_preregistration_v78(item)


@pytest.mark.parametrize("precision", subject.PRECISIONS_V78)
def test_v78_supported_hypothetical_standard_artifact_manifest(precision):
    manifest = candidate_manifest(precision)
    assert (
        subject.validate_candidate_artifact_manifest_v78(
            frozen(), manifest
        )
        == manifest
    )


def test_v78_candidate_manifest_rejects_unsupported_or_nonexact_paths():
    mutations = [
        lambda item: item.__setitem__("custom_loader_or_plugin", True),
        lambda item: item.__setitem__("candidate_config_sha256", "0" * 64),
        lambda item: item.__setitem__("candidate_visual_key_count", 1),
        lambda item: item.__setitem__("retained_key_names_sha256", "0" * 64),
        lambda item: item.__setitem__("omitted_logical_bytes", 1),
        lambda item: item["payload_identity"].__setitem__(
            "all_retained_payloads_exact", False
        ),
        lambda item: item.__setitem__(
            "candidate_physical_weight_bytes", 1
        ),
    ]
    for mutate in mutations:
        item = candidate_manifest()
        mutate(item)
        if item["candidate_physical_weight_bytes"] == 1:
            item["candidate_weight_files"][0]["bytes"] = 1
        resign(item)
        with pytest.raises(RuntimeError):
            subject.validate_candidate_artifact_manifest_v78(frozen(), item)


def test_v78_valid_hypothetical_actor_receipt_and_fail_closed_scope():
    manifest = candidate_manifest()
    receipt = actor_receipt("full_checkpoint", 0)
    assert subject.validate_actor_receipt_v78(frozen(), manifest, receipt) == receipt
    mutations = [
        lambda item: item["request_scope"].__setitem__("multimodal_request_count", 1),
        lambda item: item["request_scope"].__setitem__("speculative_request_count", 1),
        lambda item: item["live_inventory"]["visual_named_parameters"].append("visual.x"),
        lambda item: item["live_inventory"].__setitem__(
            "vision_wrapped_parameters_all_meta", False
        ),
        lambda item: item["cleanup"].__setitem__("actor_process_exited", False),
    ]
    for mutate in mutations:
        item = actor_receipt("full_checkpoint", 0)
        mutate(item)
        resign(item)
        with pytest.raises(RuntimeError):
            subject.validate_actor_receipt_v78(frozen(), manifest, item)


@pytest.mark.parametrize("precision", subject.PRECISIONS_V78)
def test_v78_valid_hypothetical_paired_receipt_is_measurable_not_promotable(
    precision,
):
    manifest = candidate_manifest(precision)
    receipt = paired_receipt(precision)
    assert (
        subject.validate_paired_runtime_receipt_v78(frozen(), manifest, receipt)
        == receipt
    )
    with pytest.raises(RuntimeError, match="promotion blocked"):
        subject.validate_promotion_v78(frozen(), manifest, receipt)


def test_v78_paired_receipt_rejects_performance_identity_and_ood_failures():
    manifest = candidate_manifest()
    mutations = [
        lambda item: item["gates"]["load_time"].__setitem__("median_ratio", 1.0),
        lambda item: item["gates"]["throughput"].__setitem__(
            "paired_bootstrap_95pct_lower_bound", 0.97
        ),
        lambda item: item["gates"]["residency"].__setitem__("maximum_delta_mib", 1),
        lambda item: item["gates"]["semantic_validation"].__setitem__(
            "source_disjoint", False
        ),
        lambda item: item["gates"]["protected_ood"].__setitem__(
            "new_safety_failures", 1
        ),
        lambda item: item["replicates"][1].__setitem__(
            "order", "full_then_derivative"
        ),
        lambda item: item["replicates"][0]["text_only_derivative"][0][
            "identity"
        ].__setitem__("greedy_token_ids_sha256", "9" * 64),
    ]
    for mutate in mutations:
        item = paired_receipt()
        mutate(item)
        # Nested actor mutation requires re-signing that actor as well.
        if (
            item["replicates"][0]["text_only_derivative"][0]["identity"][
                "greedy_token_ids_sha256"
            ]
            == "9" * 64
        ):
            resign(item["replicates"][0]["text_only_derivative"][0])
        resign(item)
        with pytest.raises(RuntimeError):
            subject.validate_paired_runtime_receipt_v78(frozen(), manifest, item)
