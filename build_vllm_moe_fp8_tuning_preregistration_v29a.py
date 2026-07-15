#!/usr/bin/env python3
"""Preregister the exact four-GPU FP8 vLLM 0.25 MoE tuner selection."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V29A = ROOT / (
    "experiments/eggroll_es_hpo/S6_V29A_FP8_MOE_TUNING_PREREGISTRATION.json"
)
MODEL_PATH_V29A = Path("/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8")
OFFICIAL_TUNER_PATH_V29A = Path("/tmp/benchmark_moe_v025_official.py")
OFFICIAL_TUNER_URL_V29A = (
    "https://raw.githubusercontent.com/vllm-project/vllm/v0.25.0/"
    "benchmarks/kernels/benchmark_moe.py"
)
VLLM_ROOT_V29A = ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/vllm"
FUSED_MOE_PATH_V29A = (
    VLLM_ROOT_V29A / "model_executor/layers/fused_moe/fused_moe.py"
)
FUSED_MOE_CONFIG_PATH_V29A = (
    VLLM_ROOT_V29A / "model_executor/layers/fused_moe/config.py"
)
VLLM_ENVS_PATH_V29A = VLLM_ROOT_V29A / "envs.py"
V28B_EVIDENCE_PATH_V29A = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V28B_V27C_TUNED_RUNTIME_AB_POSITIVE_EVIDENCE.json"
)
OUTPUT_DIRECTORY_V29A = ROOT / (
    "experiments/vllm_moe_tuning/"
    "v025_rtx_pro_6000_fp8_w8a8_block128_tp1_exhaustive_v29a"
)
RUN_DIRECTORY_V29A = ROOT / "experiments/eggroll_es_hpo/runs"
EXPERIMENT_NAME_V29A = "s6_v29a_fp8_w8a8_block128_moe_tuning_selection"
ATTEMPT_PATH_V29A = RUN_DIRECTORY_V29A / f".{EXPERIMENT_NAME_V29A}.launch_attempt.json"
REPORT_PATH_V29A = RUN_DIRECTORY_V29A / EXPERIMENT_NAME_V29A / (
    "fp8_moe_tuning_selection_report_v29a.json"
)

OFFICIAL_TUNER_SHA256_V29A = (
    "230151de56d177ac22920ff9dada010f4361933b34b54e18d5981367723a8bcf"
)
FUSED_MOE_SHA256_V29A = (
    "72811a4e543cc6f415f184cb951b61522643cddc4d6456f61a2f8c1a53b2cf79"
)
FUSED_MOE_CONFIG_SHA256_V29A = (
    "4090ec9e367fee4344c9d06f12e083ff37df427784e6736847e092222c9e1415"
)
VLLM_ENVS_SHA256_V29A = (
    "15ab853b73b26da5dc2808699138dff7217b72b9f661da274cc0c9f6c262f631"
)
MODEL_CONFIG_SHA256_V29A = (
    "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
)
MODEL_INDEX_SHA256_V29A = (
    "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6"
)
MODEL_METADATA_MANIFEST_SHA256_V29A = (
    "76ace9582b6502d99577391aad378b83b99872fb92d91303843bbe5a7e3bdb4e"
)
MODEL_WEIGHT_SHARDS_V29A = {
    "file_count": 42,
    "total_bytes": 37_463_662_160,
    "manifest_sha256": "25ae972a0ac80b7875b5e041172d5ad572b522619040f4786a9facdf0e36e5dd",
}
MODEL_ALL_FILES_FINGERPRINT_SHA256_V29A = (
    "b3307a2ce16d029a5ca0b7fb2828070c1fe07c232d396f877ea6d9a3cc9d22c9"
)
MODEL_ALL_FILES_SIZE_MANIFEST_V29A = {
    "file_count": 56,
    "total_bytes": 37_493_015_668,
    "size_manifest_sha256": (
        "46b80dca12b6cedfa9444cf7e7d13b03175a9e75a6abd775d235b9ad658737b0"
    ),
}
V28B_EVIDENCE_COMMIT_V29A = "ba87095d71e4b0f94a44cb9dfaaf2e34e43f6283"
V28B_EVIDENCE_FILE_SHA256_V29A = (
    "034b34166324359687398dd2825a0a602e444360144fc645c51e5e399e972041"
)
V28B_EVIDENCE_CONTENT_SHA256_V29A = (
    "d2601ec9636fd1df100018bc96d74adbdbc9fd2d4b1e0415cb20df683ae0326f"
)
PHYSICAL_GPU_IDS_V29A = (0, 1, 2, 3)
BATCH_SIZES_V29A = (256, 512, 1024, 2048)
BATCH_BY_GPU_V29A = dict(zip(PHYSICAL_GPU_IDS_V29A, BATCH_SIZES_V29A))
GPU_IDENTITIES_V29A = {
    0: {
        "uuid": "GPU-4c394fc5-b18f-6622-ca94-f7fbd7112927",
        "pci_bus_id": "00000000:01:00.0",
        "total_bytes": 102_641_958_912,
    },
    1: {
        "uuid": "GPU-f10c2baf-536b-1d40-cd4b-25b202ae0ded",
        "pci_bus_id": "00000000:21:00.0",
        "total_bytes": 102_641_958_912,
    },
    2: {
        "uuid": "GPU-04cde663-7c53-2f18-3ec4-1699820e2640",
        "pci_bus_id": "00000000:C1:00.0",
        "total_bytes": 102_641_958_912,
    },
    3: {
        "uuid": "GPU-972bf85d-1b32-2d1b-20f6-babc4c804999",
        "pci_bus_id": "00000000:F1:00.0",
        "total_bytes": 102_641_958_912,
    },
}
BLOCK_SHAPE_V29A = (128, 128)
EXPECTED_OUTPUT_FILENAME_V29A = (
    "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
    "Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json"
)
SEARCH_SPACE_SHA256_V29A = (
    "e9a5db2e566fab0e43bc47f7e92682debc830617199f15f9e0275e7eaed81c98"
)
SELECTION_SEED_V29A = 20260929


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"V29A JSON object required: {path}")
    return value


def search_space_v29a():
    ranges = {
        "BLOCK_SIZE_M": [16, 32, 64, 128, 256],
        "BLOCK_SIZE_N": [32, 64, 128, 256],
        "BLOCK_SIZE_K": [64, 128, 256],
        "GROUP_SIZE_M": [1, 16, 32, 64],
        "num_warps": [4, 8],
        "num_stages": [2, 3, 4, 5],
    }
    keys = tuple(ranges)
    configs = [
        dict(zip(keys, values))
        for values in itertools.product(*(ranges[key] for key in keys))
    ]
    block_n, block_k = BLOCK_SHAPE_V29A
    return [
        config for config in configs
        if (
            config["BLOCK_SIZE_N"] % block_n == 0
            or block_n % config["BLOCK_SIZE_N"] == 0
        ) and (
            config["BLOCK_SIZE_K"] % block_k == 0
            or block_k % config["BLOCK_SIZE_K"] == 0
        )
    ]


def validate_static_inputs_v29a():
    expected_files = {
        OFFICIAL_TUNER_PATH_V29A: OFFICIAL_TUNER_SHA256_V29A,
        FUSED_MOE_PATH_V29A: FUSED_MOE_SHA256_V29A,
        FUSED_MOE_CONFIG_PATH_V29A: FUSED_MOE_CONFIG_SHA256_V29A,
        VLLM_ENVS_PATH_V29A: VLLM_ENVS_SHA256_V29A,
        MODEL_PATH_V29A / "config.json": MODEL_CONFIG_SHA256_V29A,
        MODEL_PATH_V29A / "model.safetensors.index.json": MODEL_INDEX_SHA256_V29A,
        V28B_EVIDENCE_PATH_V29A: V28B_EVIDENCE_FILE_SHA256_V29A,
    }
    for path, expected in expected_files.items():
        _require(
            path.is_file() and not path.is_symlink() and file_sha256(path) == expected,
            f"V29A static input identity changed: {path}",
        )
    config = _load_json(MODEL_PATH_V29A / "config.json")
    text = config.get("text_config", {})
    quant = config.get("quantization_config", {})
    _require(
        config.get("architectures") == ["Qwen3_5MoeForConditionalGeneration"]
        and text.get("hidden_size") == 2048
        and text.get("moe_intermediate_size") == 512
        and text.get("num_experts") == 256
        and text.get("num_experts_per_tok") == 8
        and text.get("dtype") == "bfloat16"
        and quant.get("quant_method") == "fp8"
        and quant.get("fmt") == "e4m3"
        and quant.get("activation_scheme") == "dynamic"
        and quant.get("weight_block_size") == list(BLOCK_SHAPE_V29A),
        "V29A FP8 model geometry or quantization changed",
    )
    index = _load_json(MODEL_PATH_V29A / "model.safetensors.index.json")
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    _require(
        len(shard_names) == MODEL_WEIGHT_SHARDS_V29A["file_count"]
        and all((MODEL_PATH_V29A / name).is_file() for name in shard_names),
        "V29A FP8 model shard surface changed",
    )
    size_records = [
        {"file": path.name, "bytes": path.stat().st_size}
        for path in sorted(MODEL_PATH_V29A.iterdir()) if path.is_file()
    ]
    _require(
        len(size_records) == MODEL_ALL_FILES_SIZE_MANIFEST_V29A["file_count"]
        and sum(item["bytes"] for item in size_records)
        == MODEL_ALL_FILES_SIZE_MANIFEST_V29A["total_bytes"]
        and canonical_sha256(size_records)
        == MODEL_ALL_FILES_SIZE_MANIFEST_V29A["size_manifest_sha256"],
        "V29A FP8 model all-file size surface changed",
    )
    evidence = _load_json(V28B_EVIDENCE_PATH_V29A)
    _require(
        evidence.get("content_sha256_before_self_field")
        == V28B_EVIDENCE_CONTENT_SHA256_V29A
        and canonical_sha256(_without_self(evidence))
        == V28B_EVIDENCE_CONTENT_SHA256_V29A
        and evidence.get("decision", {}).get("bf16_v27c_table_reuse_for_fp8_authorized")
        is False,
        "V29A V28B evidence binding changed",
    )
    search = search_space_v29a()
    _require(
        len(search) == 1920 and canonical_sha256(search) == SEARCH_SPACE_SHA256_V29A,
        "V29A official exhaustive search space changed",
    )
    return config, index, evidence


def build_preregistration_v29a():
    validate_static_inputs_v29a()
    assignments = {
        str(gpu_id): {
            "physical_gpu_id": gpu_id,
            "batch_size": BATCH_BY_GPU_V29A[gpu_id],
            "search_space_configuration_count": 1920,
            "search_space_sha256": SEARCH_SPACE_SHA256_V29A,
        }
        for gpu_id in PHYSICAL_GPU_IDS_V29A
    }
    value = {
        "schema": "vllm-moe-fp8-tuning-preregistration-v29a",
        "status": "preregistered_before_fp8_tuning_output_tuner_not_launched",
        "strict_selection_only": True,
        "basis": {
            "v28b_positive_evidence_commit": V28B_EVIDENCE_COMMIT_V29A,
            "v28b_positive_evidence_file_sha256": V28B_EVIDENCE_FILE_SHA256_V29A,
            "v28b_positive_evidence_content_sha256": (
                V28B_EVIDENCE_CONTENT_SHA256_V29A
            ),
            "bf16_v27c_table_validated_only_for_bf16_runtime": True,
            "bf16_v27c_table_reuse_for_fp8_forbidden": True,
        },
        "official_source_contract": {
            "vllm_version": "0.25.0",
            "url": OFFICIAL_TUNER_URL_V29A,
            "local_path": str(OFFICIAL_TUNER_PATH_V29A),
            "file_sha256": OFFICIAL_TUNER_SHA256_V29A,
            "installed_fused_moe_path": str(FUSED_MOE_PATH_V29A),
            "installed_fused_moe_sha256": FUSED_MOE_SHA256_V29A,
            "installed_fused_moe_config_path": str(FUSED_MOE_CONFIG_PATH_V29A),
            "installed_fused_moe_config_sha256": FUSED_MOE_CONFIG_SHA256_V29A,
            "installed_vllm_envs_path": str(VLLM_ENVS_PATH_V29A),
            "installed_vllm_envs_sha256": VLLM_ENVS_SHA256_V29A,
            "runner_calls_official_BenchmarkWorker_tune_and_save_configs": True,
        },
        "model_contract": {
            "path": str(MODEL_PATH_V29A),
            "config_sha256": MODEL_CONFIG_SHA256_V29A,
            "index_sha256": MODEL_INDEX_SHA256_V29A,
            "tensor_metadata_manifest_sha256": MODEL_METADATA_MANIFEST_SHA256_V29A,
            "weight_shards": MODEL_WEIGHT_SHARDS_V29A,
            "all_files": {
                **MODEL_ALL_FILES_SIZE_MANIFEST_V29A,
                "fingerprint_sha256": MODEL_ALL_FILES_FINGERPRINT_SHA256_V29A,
                "real_launch_rehashes_every_file_before_gpu_claim": True,
            },
            "geometry": {
                "experts": 256,
                "experts_per_token": 8,
                "hidden_size": 2048,
                "moe_intermediate_size": 512,
                "official_shard_intermediate_size": 1024,
                "official_config_N": 512,
            },
            "quantization": {
                "dtype_cli": "fp8_w8a8",
                "resolved_activation_dtype": "bfloat16",
                "quant_method": "fp8",
                "format": "e4m3",
                "activation_scheme": "dynamic",
                "block_shape_source": "config.quantization_config.weight_block_size",
                "block_shape_auto_detected": [128, 128],
                "block_shape_must_equal": [128, 128],
            },
        },
        "hardware_contract": {
            "physical_gpu_ids": list(PHYSICAL_GPU_IDS_V29A),
            "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            "driver_version": "610.43.02",
            "identities": {str(key): value for key, value in GPU_IDENTITIES_V29A.items()},
            "all_four_must_be_idle_immediately_before_attempt_claim": True,
            "placement_groups_mapped_by_string_int_canonicalized_probe": True,
            "placement_group_creation_order_never_defines_physical_gpu": True,
            "placement_group_bundle_per_worker": {"CPU": 1, "GPU": 1},
            "one_non_detached_one_gpu_official_worker_per_physical_gpu": True,
            "all_four_workers_tune_concurrently": True,
            "simultaneous_all_four_actor_pid_and_positive_utilization_observation_required": True,
            "final_bounded_cleanup_requires_all_four_idle": True,
        },
        "runtime_environment_contract": {
            "sys_prefix": str(ROOT / "es-at-scale/.venv"),
            "sys_executable": str(ROOT / "es-at-scale/.venv/bin/python"),
            "cuda_visible_devices": "0,1,2,3",
            "versions": {
                "ray": "2.56.0",
                "torch": "2.11.0+cu130",
                "vllm": "0.25.0",
                "triton": "3.6.0",
            },
            "external_moe_backend_tuning_and_batch_invariance_overrides_empty": True,
            "vllm_moe_tune_cache_clear_interval": 50,
        },
        "tuning_contract": {
            "selection_seed": SELECTION_SEED_V29A,
            "tensor_parallel_size": 1,
            "enable_expert_parallel": False,
            "dtype_cli": "fp8_w8a8",
            "use_deep_gemm": False,
            "tune": True,
            "official_num_iters_per_configuration": 20,
            "batch_sizes": list(BATCH_SIZES_V29A),
            "batch_assignment_by_physical_gpu": assignments,
            "exhaustive_search_space_configurations_per_batch": 1920,
            "total_configurations_across_four_concurrent_workers": 7680,
            "search_space_sha256": SEARCH_SPACE_SHA256_V29A,
            "block_shape_argument_to_every_official_worker": [128, 128],
            "fresh_exclusive_output_directory": str(OUTPUT_DIRECTORY_V29A),
            "expected_output_file_count": 1,
            "expected_official_output_filename": EXPECTED_OUTPUT_FILENAME_V29A,
            "official_filename_removes_spaces_from_python_block_shape_list": True,
            "filename_with_python_list_space_is_rejected": True,
            "output_must_contain_exactly_four_batch_keys": True,
        },
        "compiler_error_policy": {
            "official_OutOfResources_skip_unchanged": True,
            "additional_CompilationError_skip_authorized": False,
            "additional_RuntimeError_skip_authorized": False,
            "any_non_OutOfResources_exception_fails_run": True,
            "failure_requires_separate_evidence_and_preregistration_before_retry": True,
            "known_bf16_compiler_failures_do_not_pre_authorize_fp8_exception_filters": True,
        },
        "persistence_contract": {
            "selected_official_table_is_the_only_tuning_payload": True,
            "attempt_and_report_are_compact_hashes_identities_counts_and_configs_only": True,
            "raw_progress_timing_vectors_compiler_logs_or_search_results_persisted": False,
            "contains_train_or_evaluation_rows": False,
        },
        "authority": {
            "fp8_tuning_selection_launch_authorized": True,
            "selected_table_direct_adoption_authorized": False,
            "training_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
            "validation_heldout_ood_or_benchmark_open_authorized": False,
            "successful_selection_authorizes_only_separate_fp8_table_evaluation_preregistration": True,
        },
        "contains_dataset_rows_questions_answers_or_document_content": False,
        "contains_validation_heldout_ood_or_benchmark_content": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_preregistration_v29a(value):
    if (
        not isinstance(value, dict)
        or value.get("schema") != "vllm-moe-fp8-tuning-preregistration-v29a"
        or value.get("status")
        != "preregistered_before_fp8_tuning_output_tuner_not_launched"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(_without_self(value))
        or value.get("basis", {}).get("bf16_v27c_table_reuse_for_fp8_forbidden")
        is not True
        or value.get("model_contract", {}).get("quantization", {}).get(
            "block_shape_must_equal"
        ) != [128, 128]
        or value.get("tuning_contract", {}).get("expected_official_output_filename")
        != EXPECTED_OUTPUT_FILENAME_V29A
        or value.get("tuning_contract", {}).get(
            "total_configurations_across_four_concurrent_workers"
        ) != 7680
        or value.get("compiler_error_policy", {}).get(
            "additional_CompilationError_skip_authorized"
        ) is not False
        or value.get("compiler_error_policy", {}).get(
            "additional_RuntimeError_skip_authorized"
        ) is not False
        or value.get("authority", {}).get("selected_table_direct_adoption_authorized")
        is not False
        or value.get("authority", {}).get("evaluation_authorized") is not False
        or value.get("contains_dataset_rows_questions_answers_or_document_content")
        is not False
        or value.get("contains_validation_heldout_ood_or_benchmark_content") is not False
    ):
        raise RuntimeError("V29A FP8 tuner preregistration contract changed")
    assignments = value["tuning_contract"]["batch_assignment_by_physical_gpu"]
    _require(tuple(assignments) == ("0", "1", "2", "3"), "V29A GPU map changed")
    _require(
        tuple(assignments[str(gpu)]["batch_size"] for gpu in PHYSICAL_GPU_IDS_V29A)
        == BATCH_SIZES_V29A,
        "V29A batch map changed",
    )
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = validate_preregistration_v29a(build_preregistration_v29a())
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return value


if __name__ == "__main__":
    main()
