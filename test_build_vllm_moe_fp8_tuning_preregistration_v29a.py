#!/usr/bin/env python3

import copy
import json

import pytest

import build_vllm_moe_fp8_tuning_preregistration_v29a as prereg


def test_v29a_preregistration_is_deterministic_and_self_sealed():
    frozen = json.loads(prereg.OUTPUT_PATH_V29A.read_text(encoding="utf-8"))
    built = prereg.build_preregistration_v29a()
    assert built == frozen
    assert prereg.validate_preregistration_v29a(built) == built


def test_v29a_exact_fp8_model_and_all_file_shard_seals():
    value = prereg.build_preregistration_v29a()
    model = value["model_contract"]
    assert model["path"].endswith("Qwen3.6-35B-A3B-FP8")
    assert model["config_sha256"] == prereg.MODEL_CONFIG_SHA256_V29A
    assert model["index_sha256"] == prereg.MODEL_INDEX_SHA256_V29A
    assert model["weight_shards"] == {
        "file_count": 42,
        "total_bytes": 37_463_662_160,
        "manifest_sha256": "25ae972a0ac80b7875b5e041172d5ad572b522619040f4786a9facdf0e36e5dd",
    }
    assert model["all_files"]["file_count"] == 56
    assert model["all_files"]["fingerprint_sha256"] == (
        "b3307a2ce16d029a5ca0b7fb2828070c1fe07c232d396f877ea6d9a3cc9d22c9"
    )
    quant = model["quantization"]
    assert quant["dtype_cli"] == "fp8_w8a8"
    assert quant["block_shape_auto_detected"] == [128, 128]
    assert quant["block_shape_must_equal"] == [128, 128]


def test_v29a_official_filename_is_exact_and_rejects_python_list_space():
    value = prereg.build_preregistration_v29a()
    tuning = value["tuning_contract"]
    assert tuning["expected_official_output_filename"] == (
        "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
        "Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json"
    )
    assert "[128, 128]" not in tuning["expected_official_output_filename"]
    source = prereg.FUSED_MOE_PATH_V29A.read_text(encoding="utf-8")
    assert ').replace(" ", "")' in source
    assert tuning["filename_with_python_list_space_is_rejected"] is True


def test_v29a_all_four_gpus_are_concurrent_and_each_owns_one_exhaustive_batch():
    value = prereg.build_preregistration_v29a()
    tuning = value["tuning_contract"]
    assignments = tuning["batch_assignment_by_physical_gpu"]
    assert [assignments[str(gpu)]["batch_size"] for gpu in range(4)] == [
        256, 512, 1024, 2048,
    ]
    assert all(
        assignments[str(gpu)]["search_space_configuration_count"] == 1920
        for gpu in range(4)
    )
    assert tuning["total_configurations_across_four_concurrent_workers"] == 7680
    hardware = value["hardware_contract"]
    assert hardware["all_four_workers_tune_concurrently"] is True
    assert hardware["placement_group_bundle_per_worker"] == {"CPU": 1, "GPU": 1}
    assert hardware[
        "simultaneous_all_four_actor_pid_and_positive_utilization_observation_required"
    ] is True
    search = prereg.search_space_v29a()
    assert len(search) == 1920
    assert prereg.canonical_sha256(search) == prereg.SEARCH_SPACE_SHA256_V29A


def test_v29a_bf16_table_compiler_exceptions_and_all_broad_authority_are_closed():
    value = prereg.build_preregistration_v29a()
    assert value["basis"]["bf16_v27c_table_reuse_for_fp8_forbidden"] is True
    compiler = value["compiler_error_policy"]
    assert compiler["official_OutOfResources_skip_unchanged"] is True
    assert compiler["additional_CompilationError_skip_authorized"] is False
    assert compiler["additional_RuntimeError_skip_authorized"] is False
    authority = value["authority"]
    assert authority["fp8_tuning_selection_launch_authorized"] is True
    for key in (
        "selected_table_direct_adoption_authorized", "training_authorized",
        "model_update_authorized", "checkpoint_write_authorized",
        "evaluation_authorized", "dataset_promotion_authorized",
        "validation_heldout_ood_or_benchmark_open_authorized",
    ):
        assert authority[key] is False


def test_v29a_contract_mutations_fail_closed():
    value = prereg.build_preregistration_v29a()
    for mutate in (
        lambda item: item["basis"].__setitem__(
            "bf16_v27c_table_reuse_for_fp8_forbidden", False
        ),
        lambda item: item["model_contract"]["quantization"].__setitem__(
            "block_shape_must_equal", [0, 0]
        ),
        lambda item: item["compiler_error_policy"].__setitem__(
            "additional_RuntimeError_skip_authorized", True
        ),
    ):
        changed = copy.deepcopy(value)
        mutate(changed)
        changed["content_sha256_before_self_field"] = prereg.canonical_sha256({
            key: item for key, item in changed.items()
            if key != "content_sha256_before_self_field"
        })
        with pytest.raises(RuntimeError, match="contract changed"):
            prereg.validate_preregistration_v29a(changed)
