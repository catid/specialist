#!/usr/bin/env python3

import copy
import json

import pytest

import build_vllm_moe_fp8_tuning_preregistration_v29a as prereg_v29a
import build_vllm_moe_fp8_tuning_preregistration_v29b as prereg


def test_v29b_preregistration_is_deterministic_and_self_sealed():
    frozen = json.loads(prereg.OUTPUT_PATH_V29B.read_text(encoding="utf-8"))
    built = prereg.build_preregistration_v29b()
    assert built == frozen
    assert prereg.validate_preregistration_v29b(built) == built


def test_v29b_binds_committed_v29a_failure_and_only_builtin_pid_correction():
    value = prereg.build_preregistration_v29b()
    retry = value["retry_of"]
    correction = value["sole_infrastructure_correction"]
    assert retry["v29a_failure_evidence_commit"] == (
        "d8877ffbb3434e1f11e6368999e3e0a9c859d0bc"
    )
    assert retry["v29a_failure_evidence_file_sha256"] == (
        "bb03815f9ae0e6704b77880f1d6b395c5a6784dbc0d6dc7ab42881b75b288753"
    )
    assert retry["v29a_failure_evidence_content_sha256"] == (
        "643c3a0eebb4953fe40e92ef7bd48f0d95d15511b44315308043ffb2388ad6d3"
    )
    assert correction["ray_include_dashboard"] is False
    assert correction["ray_state_api_or_get_actor_used"] is False
    assert correction["official_actor_pid_rpc"] == (
        "worker.__ray_call__.remote(lambda self: os.getpid())"
    )
    assert correction["pinned_ray_actor_file_sha256"] == (
        "c7af32157f768ed104dd80311b1ae67a275183bb32836b19c987561e7cc0562a"
    )
    assert correction[
        "pid_concurrency_utilization_exception_and_closed_data_gates_weakened"
    ] is False


def test_v29b_official_search_model_hardware_compiler_and_authority_match_v29a():
    original = prereg_v29a.build_preregistration_v29a()
    retry = prereg.build_preregistration_v29b()
    for key in (
        "basis", "official_source_contract", "model_contract",
        "hardware_contract", "runtime_environment_contract",
        "compiler_error_policy", "persistence_contract", "authority",
    ):
        assert retry[key] == original[key]
    original_tuning = copy.deepcopy(original["tuning_contract"])
    retry_tuning = copy.deepcopy(retry["tuning_contract"])
    original_tuning.pop("fresh_exclusive_output_directory")
    retry_tuning.pop("fresh_exclusive_output_directory")
    assert retry_tuning == original_tuning
    assert retry["tuning_contract"]["fresh_exclusive_output_directory"].endswith(
        "exhaustive_v29b"
    )
    assert retry["tuning_contract"]["fresh_exclusive_output_directory"] != (
        original["tuning_contract"]["fresh_exclusive_output_directory"]
    )
    assert prereg.ATTEMPT_PATH_V29B != prereg_v29a.ATTEMPT_PATH_V29A
    assert prereg.REPORT_PATH_V29B != prereg_v29a.REPORT_PATH_V29A
    assert prereg.OUTPUT_DIRECTORY_V29B != prereg_v29a.OUTPUT_DIRECTORY_V29A


def test_v29b_exact_fp8_model_and_all_file_shard_seals():
    value = prereg.build_preregistration_v29b()
    model = value["model_contract"]
    assert model["path"].endswith("Qwen3.6-35B-A3B-FP8")
    assert model["config_sha256"] == prereg.MODEL_CONFIG_SHA256_V29B
    assert model["index_sha256"] == prereg.MODEL_INDEX_SHA256_V29B
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


def test_v29b_official_filename_is_exact_and_rejects_python_list_space():
    value = prereg.build_preregistration_v29b()
    tuning = value["tuning_contract"]
    assert tuning["expected_official_output_filename"] == (
        "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
        "Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json"
    )
    assert "[128, 128]" not in tuning["expected_official_output_filename"]
    source = prereg.FUSED_MOE_PATH_V29B.read_text(encoding="utf-8")
    assert ').replace(" ", "")' in source
    assert tuning["filename_with_python_list_space_is_rejected"] is True


def test_v29b_all_four_gpus_are_concurrent_and_each_owns_one_exhaustive_batch():
    value = prereg.build_preregistration_v29b()
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
    search = prereg.search_space_v29b()
    assert len(search) == 1920
    assert prereg.canonical_sha256(search) == prereg.SEARCH_SPACE_SHA256_V29B


def test_v29b_bf16_table_compiler_exceptions_and_all_broad_authority_are_closed():
    value = prereg.build_preregistration_v29b()
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


def test_v29b_contract_mutations_fail_closed():
    value = prereg.build_preregistration_v29b()
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
        lambda item: item["sole_infrastructure_correction"].__setitem__(
            "ray_state_api_or_get_actor_used", True
        ),
        lambda item: item["retry_of"].__setitem__(
            "v29a_failure_evidence_content_sha256", "0" * 64
        ),
        lambda item: item["sole_infrastructure_correction"].__setitem__(
            "official_actor_pid_rpc", "different"
        ),
        lambda item: item["hardware_contract"].__setitem__(
            "simultaneous_all_four_actor_pid_and_positive_utilization_observation_required",
            False,
        ),
    ):
        changed = copy.deepcopy(value)
        mutate(changed)
        changed["content_sha256_before_self_field"] = prereg.canonical_sha256({
            key: item for key, item in changed.items()
            if key != "content_sha256_before_self_field"
        })
        with pytest.raises(RuntimeError, match="contract changed"):
            prereg.validate_preregistration_v29b(changed)
