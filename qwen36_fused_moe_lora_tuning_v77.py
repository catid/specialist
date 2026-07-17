#!/usr/bin/env python3
"""Fail-closed contract helpers for Qwen3.6 fused-MoE LoRA tuning.

This module is deliberately CPU-only: it imports neither torch nor vLLM.  It
models the vLLM 0.25 LoRA config lookup exactly, constructs the six config
documents consumed by that loader, and validates future four-GPU paired
measurement receipts.  It does not grant GPU, dataset, training, evaluation,
or checkpoint authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


SCHEMA_V77 = "qwen36-fused-moe-lora-kernel-tuning-preregistration-v77"
BUNDLE_SCHEMA_V77 = "qwen36-lora-kernel-config-bundle-v77"
RECEIPT_SCHEMA_V77 = "qwen36-lora-kernel-paired-measurement-v77"
GPU_NAME_V77 = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
SANITIZED_GPU_NAME_V77 = (
    "NVIDIA_RTX_PRO_6000_Blackwell_Max_Q_Workstation_Edition"
)
PRECISIONS_V77 = ("bf16", "fp8_serialized")
OP_TYPES_V77 = (
    "shrink",
    "expand",
    "fused_moe_lora_w13_shrink",
    "fused_moe_lora_w13_expand",
    "fused_moe_lora_w2_shrink",
    "fused_moe_lora_w2_expand",
)
M_GRID_V77 = (1, 2, 4, 8, 16, 32, 64, 68, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
RELEVANT_JIT_KERNELS_V77 = (
    "_fused_moe_lora_one_shot_kernel",
    "_lora_expand_kernel",
    "_lora_shrink_kernel",
    "fused_moe_kernel",
)

V29_SELECTED_TABLE_V77 = {
    "relative_path": (
        "experiments/vllm_moe_tuning/"
        "v025_rtx_pro_6000_fp8_w8a8_block128_tp1_exhaustive_v29b/"
        "E=256,N=512,device_name=NVIDIA_RTX_PRO_6000_Blackwell_"
        "Max-Q_Workstation_Edition,dtype=fp8_w8a8,block_shape=[128,128].json"
    ),
    "file_sha256": "1a4ed0f44c6d7cc788baecd073107b4634db4c769f0820d10174f61117b25618",
    "content_sha256": "d4a49735ccfd094d6e5a3ee763eca99ed355a51fd11a7c835bfecf9fafeaa50d",
    "loaded_config_sha256": "ebf00590ac51e66e52f5e99b933d1be72703fbbcc809cc2d585eca8d6b0c0a5d",
}


def canonical_sha256_v77(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256_v77(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _finite_positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def sanitize_gpu_name_v77(name: str) -> str:
    _require(isinstance(name, str) and name, "GPU name is required")
    value = name.replace(" ", "_").replace("-", "_")
    _require(
        re.fullmatch(r"[A-Za-z0-9_]+", value) is not None,
        "GPU name would produce an unsafe config filename",
    )
    return value


def config_filename_v77(op_type: str, gpu_name: str = GPU_NAME_V77) -> str:
    _require(op_type in OP_TYPES_V77, f"unsupported LoRA op type: {op_type}")
    stem = sanitize_gpu_name_v77(gpu_name)
    suffix = "EXPAND_TRUE" if op_type == "expand" else op_type.upper()
    result = f"{stem}_{suffix}.json"
    _require(Path(result).name == result, "unsafe LoRA config filename")
    return result


def required_filenames_v77() -> dict[str, str]:
    return {op_type: config_filename_v77(op_type) for op_type in OP_TYPES_V77}


def _dense_module_cases_v77() -> list[dict[str, Any]]:
    # These cases are derived from the exact rank-32 adapter header and the
    # qwen3_5.py packed_modules_mapping.  max_output is what lora_expand passes
    # as MAX_N, not the sum of packed slice widths.
    return [
        {"id": "linear_in_proj_qkvz", "input": 2048, "max_output": 8192, "num_slices": 2},
        {"id": "linear_in_proj_ba", "input": 2048, "max_output": 32, "num_slices": 2},
        {"id": "linear_out_proj", "input": 4096, "max_output": 2048, "num_slices": 1},
        {"id": "attention_qkv", "input": 2048, "max_output": 8192, "num_slices": 3},
        {"id": "attention_o", "input": 4096, "max_output": 2048, "num_slices": 1},
        {"id": "shared_expert_gate_up", "input": 2048, "max_output": 512, "num_slices": 2},
        {"id": "shared_expert_down", "input": 512, "max_output": 2048, "num_slices": 1},
        {"id": "router_gate", "input": 2048, "max_output": 256, "num_slices": 1},
    ]


def _dedupe_lookup_cases_v77(cases: list[dict[str, Any]], op_type: str) -> list[dict[str, Any]]:
    seen: dict[tuple[int, int, int, int | None], dict[str, Any]] = {}
    for module in cases:
        hidden = module["input"] if op_type == "shrink" else module["max_output"]
        key = (module["num_slices"], hidden, 32, None)
        record = {
            "id": (
                f"s{module['num_slices']}_h{hidden}_r32"
            ),
            "num_slices": module["num_slices"],
            "hidden_size": hidden,
            "rank": 32,
            "moe_intermediate_size": None,
        }
        seen.setdefault(key, record)
    return sorted(
        seen.values(), key=lambda item: (item["num_slices"], item["hidden_size"])
    )


def operation_inventory_v77() -> dict[str, dict[str, Any]]:
    modules = _dense_module_cases_v77()
    inventory: dict[str, dict[str, Any]] = {
        "shrink": {
            "family": "dense_lora",
            "add_inputs": None,
            "module_cases": copy.deepcopy(modules),
            "lookup_cases": _dedupe_lookup_cases_v77(modules, "shrink"),
        },
        "expand": {
            "family": "dense_lora",
            "add_inputs": True,
            "module_cases": copy.deepcopy(modules),
            "lookup_cases": _dedupe_lookup_cases_v77(modules, "expand"),
        },
    }
    for projection, slices in (("w13", 2), ("w2", 1)):
        for direction in ("shrink", "expand"):
            op_type = f"fused_moe_lora_{projection}_{direction}"
            inventory[op_type] = {
                "family": "fused_moe_lora",
                "add_inputs": None,
                "module_cases": [
                    {
                        "id": f"routed_expert_{projection}",
                        "hidden_size": 2048,
                        "moe_intermediate_size": 512,
                        "num_experts": 256,
                        "top_k": 8,
                        "num_slices": slices,
                    }
                ],
                "lookup_cases": [
                    {
                        "id": f"s{slices}_h2048_r32_i512",
                        "num_slices": slices,
                        "hidden_size": 2048,
                        "rank": 32,
                        "moe_intermediate_size": 512,
                    }
                ],
            }
    return {op_type: inventory[op_type] for op_type in OP_TYPES_V77}


def loader_lookup_key_v77(
    op_type: str,
    *,
    max_loras: int,
    num_slices: int,
    m: int,
    hidden_size: int,
    rank: int,
    moe_intermediate_size: int | None = None,
) -> tuple[str, ...]:
    """Return the exact vLLM 0.25 nested JSON lookup key.

    The apparent fused-shrink inversion is intentional.  Installed vLLM uses
    ``(hidden, rank)`` only when ``op_type == 'shrink'`` exactly; all fused op
    names, including fused shrink, use ``(rank, hidden)``.
    """

    _require(op_type in OP_TYPES_V77, "unsupported lookup operation")
    for value, label in (
        (max_loras, "max_loras"),
        (num_slices, "num_slices"),
        (m, "M"),
        (hidden_size, "hidden_size"),
        (rank, "rank"),
    ):
        _require(_positive_int(value), f"invalid {label}")
    if op_type == "shrink":
        k, n = hidden_size, rank
    else:
        k, n = rank, hidden_size
    keys = [str(max_loras), str(num_slices), str(m), str(k), str(n)]
    if moe_intermediate_size is not None:
        _require(_positive_int(moe_intermediate_size), "invalid MoE intermediate size")
        keys.append(str(moe_intermediate_size))
    return tuple(keys)


def kernel_search_space_v77(op_type: str) -> dict[str, list[Any]]:
    _require(op_type in OP_TYPES_V77, "unsupported search-space operation")
    shrink = op_type.endswith("shrink")
    fused = op_type.startswith("fused_moe_lora_")
    result: dict[str, list[Any]] = {
        "block_m": [16, 32, 64, 128],
        "block_n": [16, 32] if shrink else [16, 32, 64, 128],
        "block_k": [16, 32, 64, 128, 256] if shrink else [16, 32],
        "num_warps": [2, 4, 8],
        "num_stages": [2, 3, 4, 5],
    }
    if shrink or fused:
        result["group_size_m"] = [1, 4, 8]
    if shrink:
        result["split_k"] = [1] if fused else [1, 8, 64]
    elif fused:
        result["split_k"] = [1]
    if not fused:
        result["num_ctas"] = [1]
        result["max_nreg"] = [None]
    return result


def _required_config_keys_v77(op_type: str) -> set[str]:
    fused = op_type.startswith("fused_moe_lora_")
    if fused:
        return {
            "block_m", "block_n", "block_k", "num_warps", "num_stages",
            "group_size_m", "split_k",
        }
    if op_type == "shrink":
        return {
            "block_m", "block_n", "block_k", "num_warps", "num_stages",
            "group_size_m", "split_k", "num_ctas", "max_nreg",
        }
    return {
        "block_m", "block_n", "block_k", "num_warps", "num_stages",
        "num_ctas", "max_nreg",
    }


def validate_kernel_config_v77(op_type: str, config: dict[str, Any]) -> dict[str, Any]:
    _require(op_type in OP_TYPES_V77, "unsupported kernel config operation")
    _require(isinstance(config, dict), "kernel config must be an object")
    _require(set(config) == _required_config_keys_v77(op_type), "kernel config keys changed")
    space = kernel_search_space_v77(op_type)
    for key, choices in space.items():
        _require(config.get(key) in choices, f"unsafe {op_type} kernel value: {key}")
        if config.get(key) is not None:
            _require(not isinstance(config[key], bool), f"boolean kernel value: {key}")
    _require(config["block_m"] * config["block_n"] <= 16384, "unsafe tile area")
    if op_type.endswith("expand"):
        _require(32 % config["block_k"] == 0, "expand BLOCK_K must divide rank 32")
    if op_type.startswith("fused_moe_lora_"):
        _require(config["split_k"] == 1, "fused LoRA split-K is forbidden")
    return config


def default_kernel_config_v77(
    op_type: str, m: int, *, num_slices: int = 2
) -> dict[str, Any]:
    _require(_positive_int(m), "default config M must be positive")
    _require(_positive_int(num_slices), "default config num_slices must be positive")
    if op_type == "shrink":
        return {
            "block_m": 32, "block_n": 16,
            "block_k": 256 if m < 128 else 32,
            "split_k": 64 if m < 128 else 8,
            "num_warps": 4, "num_ctas": 1, "group_size_m": 8,
            "num_stages": 2, "max_nreg": None,
        }
    if op_type in ("fused_moe_lora_w13_shrink", "fused_moe_lora_w2_shrink"):
        return {
            "block_m": 64, "block_n": 32, "block_k": 32,
            "num_warps": 4, "num_stages": 3, "group_size_m": 8,
            "split_k": 1,
        }
    if op_type in ("fused_moe_lora_w13_expand", "fused_moe_lora_w2_expand"):
        return {
            "block_m": 64, "block_n": 64, "block_k": 32,
            "num_warps": 4, "num_stages": 3, "group_size_m": 8,
            "split_k": 1,
        }
    _require(op_type == "expand", "unsupported default config operation")
    return {
        "block_m": 64, "block_n": 64 if num_slices > 1 else 128,
        "block_k": 32,
        "num_warps": 4, "num_ctas": 1, "num_stages": 2,
        "max_nreg": None,
    }


def _lookup_case_map_v77(op_type: str) -> dict[str, dict[str, Any]]:
    return {
        item["id"]: item
        for item in operation_inventory_v77()[op_type]["lookup_cases"]
    }


def build_config_document_v77(op_type: str, selections: list[dict[str, Any]]) -> dict:
    """Build one loader-compatible nested table from exact-M selections."""

    _require(op_type in OP_TYPES_V77, "unsupported config document operation")
    _require(isinstance(selections, list), "selections must be a list")
    cases = _lookup_case_map_v77(op_type)
    expected = {(case_id, m) for case_id in cases for m in M_GRID_V77}
    actual: set[tuple[str, int]] = set()
    document: dict[str, Any] = {}
    for row in selections:
        _require(isinstance(row, dict) and set(row) == {"case_id", "m", "config"}, "selection row changed")
        case_id, m = row["case_id"], row["m"]
        _require(case_id in cases and m in M_GRID_V77, "selection is outside sealed lookup grid")
        _require((case_id, m) not in actual, "duplicate exact LoRA lookup selection")
        actual.add((case_id, m))
        config = validate_kernel_config_v77(op_type, row["config"])
        case = cases[case_id]
        keys = loader_lookup_key_v77(
            op_type,
            max_loras=1,
            num_slices=case["num_slices"],
            m=m,
            hidden_size=case["hidden_size"],
            rank=case["rank"],
            moe_intermediate_size=case["moe_intermediate_size"],
        )
        cursor = document
        for key in keys[:-1]:
            child = cursor.setdefault(key, {})
            _require(isinstance(child, dict), "config lookup collision")
            cursor = child
        _require(keys[-1] not in cursor, "config leaf collision")
        cursor[keys[-1]] = copy.deepcopy(config)
    _require(actual == expected, "config document does not cover every sealed shape/M")
    return document


def lookup_config_document_v77(
    document: dict,
    op_type: str,
    *,
    max_loras: int,
    num_slices: int,
    m: int,
    hidden_size: int,
    rank: int,
    moe_intermediate_size: int | None = None,
) -> dict[str, Any]:
    """Reproduce vLLM's nearest-key lookup, including exact slice indexing."""

    keys = loader_lookup_key_v77(
        op_type,
        max_loras=max_loras,
        num_slices=num_slices,
        m=m,
        hidden_size=hidden_size,
        rank=rank,
        moe_intermediate_size=moe_intermediate_size,
    )
    cursor: Any = document
    for index, key in enumerate(keys):
        _require(isinstance(cursor, dict) and cursor, "empty config lookup level")
        if index == 1:  # num_slices is exact in installed vLLM 0.25.
            _require(key in cursor, "missing exact num_slices table")
            selected = key
        else:
            selected = key if key in cursor else min(
                cursor, key=lambda candidate: abs(int(candidate) - int(key))
            )
        cursor = cursor[selected]
    _require(isinstance(cursor, dict), "kernel config leaf is not an object")
    return validate_kernel_config_v77(op_type, cursor)


def default_selections_v77(op_type: str) -> list[dict[str, Any]]:
    return [
        {
            "case_id": case["id"],
            "m": m,
            "config": default_kernel_config_v77(
                op_type, m, num_slices=case["num_slices"]
            ),
        }
        for case in operation_inventory_v77()[op_type]["lookup_cases"]
        for m in M_GRID_V77
    ]


def build_default_config_documents_v77() -> dict[str, dict]:
    return {
        config_filename_v77(op_type): build_config_document_v77(
            op_type, default_selections_v77(op_type)
        )
        for op_type in OP_TYPES_V77
    }


def build_config_bundle_manifest_v77(
    precision: str,
    documents: dict[str, dict],
    *,
    directory_id: str,
    source_bundle_sha256: str,
    selection_receipt_sha256: str,
) -> dict[str, Any]:
    _require(precision in PRECISIONS_V77, "unsupported precision bundle")
    _require(
        isinstance(directory_id, str)
        and re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", directory_id) is not None
        and "/" not in directory_id
        and ".." not in directory_id,
        "unsafe config directory id",
    )
    _require(_sha256(source_bundle_sha256), "invalid source bundle hash")
    _require(_sha256(selection_receipt_sha256), "invalid selection receipt hash")
    expected = required_filenames_v77()
    _require(set(documents) == set(expected.values()), "six exact LoRA config files are required")
    files = []
    for op_type, filename in expected.items():
        document = documents[filename]
        cases = operation_inventory_v77()[op_type]["lookup_cases"]
        for case in cases:
            for m in M_GRID_V77:
                lookup_config_document_v77(
                    document,
                    op_type,
                    max_loras=1,
                    num_slices=case["num_slices"],
                    m=m,
                    hidden_size=case["hidden_size"],
                    rank=32,
                    moe_intermediate_size=case["moe_intermediate_size"],
                )
        files.append(
            {
                "op_type": op_type,
                "filename": filename,
                "content_sha256": canonical_sha256_v77(document),
                "document": copy.deepcopy(document),
            }
        )
    result = {
        "schema": BUNDLE_SCHEMA_V77,
        "precision": precision,
        "directory_id": directory_id,
        "gpu_name": GPU_NAME_V77,
        "vllm_version": "0.25.0",
        "rank": 32,
        "max_loras": 1,
        "source_bundle_sha256": source_bundle_sha256,
        "selection_receipt_sha256": selection_receipt_sha256,
        "files": files,
        "rejected_v29_selected_table_present": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v77(result)
    return result


def validate_config_bundle_manifest_v77(value: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, dict), "config bundle must be an object")
    original = copy.deepcopy(value)
    claimed = original.pop("content_sha256_before_self_field", None)
    _require(_sha256(claimed) and canonical_sha256_v77(original) == claimed, "config bundle self hash changed")
    _require(
        original.get("schema") == BUNDLE_SCHEMA_V77
        and original.get("precision") in PRECISIONS_V77
        and original.get("gpu_name") == GPU_NAME_V77
        and original.get("vllm_version") == "0.25.0"
        and original.get("rank") == 32
        and original.get("max_loras") == 1
        and original.get("rejected_v29_selected_table_present") is False,
        "config bundle identity changed",
    )
    directory_id = original.get("directory_id")
    _require(
        isinstance(directory_id, str)
        and re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", directory_id) is not None
        and "/" not in directory_id
        and ".." not in directory_id,
        "unsafe config directory id",
    )
    _require(
        _sha256(original.get("source_bundle_sha256"))
        and _sha256(original.get("selection_receipt_sha256")),
        "config bundle provenance is incomplete",
    )
    serialized = json.dumps(original, sort_keys=True)
    for forbidden in (
        V29_SELECTED_TABLE_V77["relative_path"],
        V29_SELECTED_TABLE_V77["file_sha256"],
        V29_SELECTED_TABLE_V77["content_sha256"],
        V29_SELECTED_TABLE_V77["loaded_config_sha256"],
    ):
        _require(forbidden not in serialized, "rejected V29 table leaked into config bundle")
    files = original.get("files")
    _require(isinstance(files, list) and len(files) == 6, "config bundle must contain six files")
    by_op = {row.get("op_type"): row for row in files if isinstance(row, dict)}
    _require(set(by_op) == set(OP_TYPES_V77), "config bundle operations changed")
    documents: dict[str, dict] = {}
    for op_type in OP_TYPES_V77:
        row = by_op[op_type]
        filename = config_filename_v77(op_type)
        document = row.get("document")
        _require(
            set(row) == {"op_type", "filename", "content_sha256", "document"}
            and row.get("filename") == filename
            and row.get("content_sha256") == canonical_sha256_v77(document),
            "config file identity changed",
        )
        documents[filename] = document
    rebuilt = build_config_bundle_manifest_v77(
        original["precision"],
        documents,
        directory_id=directory_id,
        source_bundle_sha256=original["source_bundle_sha256"],
        selection_receipt_sha256=original["selection_receipt_sha256"],
    )
    _require(rebuilt == value, "config bundle is not canonical")
    return value


def tuning_plan_v77() -> dict[str, Any]:
    return {
        "precision_isolation": {
            "arms": list(PRECISIONS_V77),
            "separate_config_directories_required": True,
            "reason": "vLLM LoRA config filenames do not encode dtype or quantization",
            "cross_precision_causal_claims_forbidden": True,
        },
        "stages": [
            {
                "index": 0,
                "name": "environment_attestation",
                "requirements": [
                    "fresh_child_process_per_arm",
                    "explicit_VLLM_USE_DEEP_GEMM_0",
                    "zero_DeepGemm_accuracy_fallback_warnings",
                    "FP8_quant_config_use_deep_gemm_false_before_engine_start",
                    "DeepGemm_disable_is_not_used_as_routed_backend_evidence",
                    "moe_backend_argument_triton",
                    "FP8_routed_backend_TRITON_attested",
                    "FP8_40_RoutedExperts_owners_attested",
                    "FP8_80_LoRA_quant_references_attested",
                    "FP8_TritonExperts_and_FusedMoEModularMethod_attested",
                    "FlashInfer_autotune_disabled",
                ],
            },
            {
                "index": 1,
                "name": "empty_default_baseline",
                "requirements": [
                    "new_empty_VLLM_TUNED_CONFIG_FOLDER_per_process",
                    "no_base_MoE_or_LoRA_table_reuse",
                    "rejected_V29_table_path_and_hashes_absent",
                    "default_behavior_measured_before_any_search",
                ],
            },
            {
                "index": 2,
                "name": "isolated_microbenchmark_search",
                "requirements": [
                    "all_six_operations",
                    "all_sealed_shape_cases",
                    "all_sealed_M_grid_points",
                    "default_seed_included",
                    "correctness_before_latency",
                    "median_of_at_least_100_steady_iterations",
                    "fresh_cache_identity_recorded",
                ],
            },
            {
                "index": 3,
                "name": "sealed_integrated_pairing",
                "requirements": [
                    "base_MoE_table_remains_empty_default",
                    "three_or_more_counterbalanced_four_GPU_pairs_per_precision",
                    "same_physical_GPUs_and_workload_within_pair",
                    "cache_read_only_after_sealed_warmup",
                    "zero_inference_time_JIT_or_missing_config_or_fallback_messages",
                ],
            },
            {
                "index": 4,
                "name": "semantic_and_OOD_confirmation",
                "requirements": [
                    "selection_frozen_before_semantic_access",
                    "paired_source_disjoint_validation",
                    "one_shot_protected_OOD_after_all_train_dev_selection",
                    "no_config_retuning_after_protected_result",
                ],
            },
        ],
        "M_grid": list(M_GRID_V77),
        "candidate_spaces": {
            op_type: kernel_search_space_v77(op_type) for op_type in OP_TYPES_V77
        },
        "selection_rule": {
            "per_exact_lookup_leaf": "minimum median kernel time among correctness-passing candidates",
            "tie_break_order": [
                "lower_p95_kernel_time",
                "lower_workspace_bytes",
                "lower_register_count",
                "lexicographically_smaller_canonical_config",
            ],
            "nearest_M_runtime_fallback_allowed_only_within_sealed_grid": True,
            "manual_post_result_table_edits_forbidden": True,
        },
        "base_moe_isolation": {
            "base_moe_tuning_in_scope": False,
            "base_moe_table_must_be_absent_in_both_paired_arms": True,
            "base_fused_moe_kernel_must_be_covered_by_warmup": True,
            "V29_selected_table_reuse_forbidden": True,
        },
    }


def promotion_gates_v77() -> dict[str, Any]:
    return {
        "replication": {
            "minimum_four_gpu_paired_replicates_per_precision": 3,
            "physical_gpus": [0, 1, 2, 3],
            "counterbalanced_orders": ["default_then_tuned", "tuned_then_default"],
        },
        "throughput": {
            "metric": "aggregate_generated_tokens_per_second",
            "minimum_median_tuned_over_default_ratio": 1.02,
            "minimum_paired_bootstrap_95pct_lower_bound": 1.0,
        },
        "token_identity": {
            "report_every_pair": True,
            "exact_all_pairs_preferred": True,
            "nonexact_requires_semantic_noninferiority": True,
        },
        "semantic_validation": {
            "always_evaluate_after_selection_freeze": True,
            "minimum_paired_point_delta": -0.001,
            "minimum_paired_95pct_lower_bound": -0.002,
        },
        "protected_ood": {
            "always_evaluate_once_after_selection_freeze": True,
            "minimum_aggregate_95pct_lower_bound": -0.005,
            "minimum_worst_stratum_point_delta": -0.01,
            "maximum_new_safety_failures": 0,
        },
        "resource": {
            "maximum_peak_vram_tuned_over_default_ratio": 1.01,
            "required_metrics": [
                "gpu_utilization_percent",
                "memory_activity_percent",
                "power_watts",
                "peak_vram_mib",
                "generated_tokens_per_second",
            ],
        },
        "runtime": {
            "all_inference_time_JIT_messages": [],
            "missing_config_messages": [],
            "fallback_messages": [],
            "DeepGemm_accuracy_fallback_warning_count": 0,
            "required_warmup_kernel_names": list(RELEVANT_JIT_KERNELS_V77),
            "cache_manifest_must_not_change_during_measurement": True,
        },
    }


def validate_preregistration_v77(value: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, dict), "preregistration must be an object")
    original = copy.deepcopy(value)
    claimed = original.pop("content_sha256_before_self_field", None)
    _require(_sha256(claimed) and canonical_sha256_v77(original) == claimed, "preregistration self hash changed")
    _require(
        original.get("schema") == SCHEMA_V77
        and original.get("bead") == "specialist-0j5.22"
        and original.get("status") == "cpu_preregistration_complete_launch_blocked"
        and original.get("operation_inventory") == operation_inventory_v77()
        and original.get("tuning_plan") == tuning_plan_v77()
        and original.get("promotion_gates") == promotion_gates_v77(),
        "preregistration contract changed",
    )
    authority = original.get("authority")
    _require(
        authority == {
            "gpu_launch": False,
            "protected_or_ood_access": False,
            "dataset_access": False,
            "model_update_or_training": False,
            "checkpoint_or_config_promotion": False,
            "site_package_modification": False,
        },
        "preregistration authority widened",
    )
    blockers = original.get("blockers")
    _require(
        isinstance(blockers, list)
        and any(
            row.get("bead") == "specialist-0j5.22"
            and row.get("kind") == "cpu_only_authority_scope"
            and row.get("fail_closed") is True
            and row.get("resolved") is False
            for row in blockers
            if isinstance(row, dict)
        ),
        "CPU-only authority blocker is not fail-closed",
    )
    resolution = original.get("environment_resolution", {})
    _require(
        resolution.get("bead") == "specialist-nen.28"
        and resolution.get("resolved_for_bound_v76_baseline") is True
        and resolution.get("legacy_cutlass_request_field_is_not_backend_evidence") is True
        and resolution.get("deepgemm_disable_gate")
        == {
            "VLLM_USE_DEEP_GEMM": "0",
            "quant_config_use_deep_gemm_before_post_init": False,
            "zero_warning_actors": 4,
            "zero_e8m0_enabled_actors": 4,
        }
        and resolution.get("independent_routed_backend_gate")
        == {
            "moe_backend_argument": "triton",
            "live_backend_class": "Fp8MoeBackend",
            "live_backend_name": "TRITON",
            "experts_implementation_class": "TritonExperts",
            "runtime_quant_wrapper_class": "FusedMoEModularMethod",
            "routed_expert_owner_count_per_actor": 40,
            "lora_quant_reference_count_per_actor": 80,
            "attested_actor_count": 4,
        },
        "bound V76 environment resolution changed",
    )
    rejected = original.get("rejected_prior_table")
    _require(
        rejected.get("identity") == V29_SELECTED_TABLE_V77
        and rejected.get("reuse_forbidden") is True
        and rejected.get("all_five_latency_endpoints_failed") is True,
        "rejected V29 evidence changed",
    )
    _require(
        original.get("model_and_adapter", {}).get("rank") == 32
        and original.get("model_and_adapter", {}).get("max_loras") == 1
        and original.get("model_and_adapter", {}).get("hidden_size") == 2048
        and original.get("model_and_adapter", {}).get("moe_intermediate_size") == 512
        and original.get("model_and_adapter", {}).get("num_experts") == 256
        and original.get("model_and_adapter", {}).get("top_k") == 8,
        "model geometry changed",
    )
    observed = original.get("observed_evidence", {})
    _require(
        observed.get("v75", {}).get("explicit_deepgemm_disable_actor_count") == 4
        and observed.get("v75", {}).get("deepgemm_warning_actor_count") == 4
        and observed.get("v75", {}).get("fail_closed_environment_pass") is False
        and observed.get("v75", {}).get("missing_lora_config_messages") == 24,
        "V75 fail-closed evidence changed",
    )
    _require(
        observed.get("v76", {}).get("environment_preflight_pass") is True
        and observed.get("v76", {}).get(
            "deepgemm_disable_and_routed_backend_are_independent"
        )
        is True
        and observed.get("v76", {}).get("routed_backend")
        == {
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
        and observed.get("v76", {}).get("stale_receipt_claim", {}).get(
            "trusted_for_backend_identity"
        )
        is False,
        "V76 live TRITON baseline changed",
    )
    _require(
        _sha256(original.get("installed_source", {}).get("bundle_sha256"))
        and original.get("installed_source", {}).get("vllm_version") == "0.25.0",
        "installed source binding changed",
    )
    return value


def _validate_gpu_arm_v77(arm: dict[str, Any], expected_gpus: list[int]) -> None:
    _require(isinstance(arm, dict), "paired arm must be an object")
    _require(
        set(arm) == {
            "aggregate_generated_tokens_per_second",
            "aggregate_token_hash_sha256",
            "per_gpu",
        },
        "paired arm fields changed",
    )
    _require(_finite_positive(arm["aggregate_generated_tokens_per_second"]), "invalid aggregate throughput")
    _require(_sha256(arm["aggregate_token_hash_sha256"]), "invalid token hash bundle")
    rows = arm["per_gpu"]
    _require(isinstance(rows, list) and len(rows) == 4, "four per-GPU rows required")
    _require(sorted(row.get("physical_gpu") for row in rows) == expected_gpus, "physical GPU attribution changed")
    required = {
        "physical_gpu", "gpu_utilization_percent", "memory_activity_percent",
        "power_watts", "peak_vram_mib", "generated_tokens_per_second",
    }
    for row in rows:
        _require(isinstance(row, dict) and set(row) == required, "per-GPU telemetry fields changed")
        for key in required - {"physical_gpu"}:
            _require(_finite_positive(row[key]), f"invalid per-GPU telemetry: {key}")
        _require(row["gpu_utilization_percent"] <= 100 and row["memory_activity_percent"] <= 100, "invalid activity percentage")


def validate_measurement_receipt_v77(
    preregistration: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    validate_preregistration_v77(preregistration)
    _require(isinstance(receipt, dict), "measurement receipt must be an object")
    original = copy.deepcopy(receipt)
    claimed = original.pop("content_sha256_before_self_field", None)
    _require(_sha256(claimed) and canonical_sha256_v77(original) == claimed, "measurement receipt self hash changed")
    _require(
        original.get("schema") == RECEIPT_SCHEMA_V77
        and original.get("precision") in PRECISIONS_V77
        and original.get("preregistration_sha256")
        == preregistration["content_sha256_before_self_field"],
        "measurement receipt identity changed",
    )
    environment = original.get("environment", {})
    _require(
        environment.get("VLLM_USE_DEEP_GEMM") == "0"
        and environment.get("deepgemm_accuracy_fallback_warning_count") == 0
        and environment.get("quant_config_use_deep_gemm") is False
        and environment.get("deepgemm_disable_is_routed_backend_evidence") is False
        and environment.get("moe_backend_argument") == "triton"
        and environment.get("flashinfer_autotune_enabled") is False
        and environment.get("site_package_modified") is False
        and environment.get("child_process_backend_attestation_passed") is True,
        "clean fail-closed kernel environment was not attested",
    )
    routed = environment.get("fp8_routed_runtime_attestation")
    if original["precision"] == "fp8_serialized":
        _require(
            routed
            == {
                "backend_class": "Fp8MoeBackend",
                "backend_name": "TRITON",
                "routed_expert_owner_class": "RoutedExperts",
                "routed_expert_owner_count": 40,
                "quant_method_class": "Fp8MoEMethod",
                "runtime_quant_wrapper_class": "FusedMoEModularMethod",
                "experts_implementation_class": "TritonExperts",
                "lora_quant_reference_count": 80,
            },
            "FP8 routed runtime is not the sealed live TRITON baseline",
        )
    else:
        _require(
            routed is None,
            "BF16 receipt must not claim an FP8 routed-runtime attestation",
        )
    _require(
        environment.get("installed_source_bundle_sha256")
        == preregistration["installed_source"]["bundle_sha256"],
        "measurement source bundle changed",
    )
    baseline = original.get("baseline", {})
    _require(
        baseline.get("kind") == "fresh_empty_default"
        and baseline.get("folder_was_fresh_and_empty") is True
        and baseline.get("base_moe_table_present") is False
        and baseline.get("rejected_v29_present") is False,
        "measurement did not start from explicit empty/default",
    )
    validate_config_bundle_manifest_v77(original.get("tuned_bundle"))
    _require(original["tuned_bundle"]["precision"] == original["precision"], "precision-mismatched tuned table")
    cache = original.get("cache", {})
    _require(
        _sha256(cache.get("manifest_before_measurement_sha256"))
        and cache.get("manifest_before_measurement_sha256")
        == cache.get("manifest_after_measurement_sha256")
        and sorted(cache.get("warmup_compiled_kernel_names", []))
        == sorted(RELEVANT_JIT_KERNELS_V77)
        and cache.get("inference_time_jit_messages") == []
        and cache.get("missing_config_messages") == []
        and cache.get("fallback_messages") == [],
        "sealed warmup/JIT/cache gate failed",
    )
    replicates = original.get("replicates")
    _require(isinstance(replicates, list) and len(replicates) >= 3, "at least three paired replicates required")
    expected_gpus = [0, 1, 2, 3]
    ratios: list[float] = []
    vram_ratios: list[float] = []
    exact_pairs = 0
    orders = set()
    for index, pair in enumerate(replicates):
        _require(
            isinstance(pair, dict)
            and set(pair) == {"replicate", "order", "physical_gpus", "default", "tuned"}
            and pair["replicate"] == index
            and pair["order"] in ("default_then_tuned", "tuned_then_default")
            and pair["physical_gpus"] == expected_gpus,
            "paired replicate identity changed",
        )
        orders.add(pair["order"])
        _validate_gpu_arm_v77(pair["default"], expected_gpus)
        _validate_gpu_arm_v77(pair["tuned"], expected_gpus)
        ratios.append(
            pair["tuned"]["aggregate_generated_tokens_per_second"]
            / pair["default"]["aggregate_generated_tokens_per_second"]
        )
        default_peak = max(row["peak_vram_mib"] for row in pair["default"]["per_gpu"])
        tuned_peak = max(row["peak_vram_mib"] for row in pair["tuned"]["per_gpu"])
        vram_ratios.append(tuned_peak / default_peak)
        if pair["default"]["aggregate_token_hash_sha256"] == pair["tuned"]["aggregate_token_hash_sha256"]:
            exact_pairs += 1
    _require(len(orders) == 2, "paired arm order was not counterbalanced")
    throughput = original.get("throughput_gate", {})
    _require(
        len(throughput.get("paired_tuned_over_default_ratios", [])) == len(ratios)
        and all(
            math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
            for a, b in zip(throughput["paired_tuned_over_default_ratios"], ratios)
        )
        and math.isclose(throughput.get("median_ratio", -1), statistics.median(ratios), rel_tol=1e-12, abs_tol=1e-12)
        and throughput["median_ratio"] >= 1.02
        and _finite_positive(throughput.get("paired_bootstrap_95pct_lower_bound"))
        and throughput["paired_bootstrap_95pct_lower_bound"] >= 1.0
        and _sha256(throughput.get("bootstrap_draw_plan_sha256"))
        and throughput.get("pass") is True,
        "throughput promotion gate failed",
    )
    token_gate = original.get("token_identity_gate", {})
    _require(
        token_gate.get("pair_count") == len(replicates)
        and token_gate.get("exact_pair_count") == exact_pairs
        and token_gate.get("all_pairs_exact") is (exact_pairs == len(replicates)),
        "token identity accounting changed",
    )
    semantic = original.get("semantic_validation_gate", {})
    _require(
        semantic.get("selection_frozen_before_access") is True
        and semantic.get("source_disjoint") is True
        and semantic.get("paired_point_delta") >= -0.001
        and semantic.get("paired_95pct_lower_bound") >= -0.002
        and semantic.get("pass") is True,
        "semantic non-inferiority gate failed",
    )
    ood = original.get("protected_ood_gate", {})
    _require(
        ood.get("selection_frozen_before_access") is True
        and ood.get("one_shot") is True
        and ood.get("aggregate_95pct_lower_bound") >= -0.005
        and ood.get("worst_stratum_point_delta") >= -0.01
        and ood.get("new_safety_failures") == 0
        and ood.get("pass") is True,
        "protected OOD gate failed",
    )
    resource = original.get("resource_gate", {})
    _require(
        math.isclose(resource.get("maximum_peak_vram_ratio", -1), max(vram_ratios), rel_tol=1e-12, abs_tol=1e-12)
        and resource["maximum_peak_vram_ratio"] <= 1.01
        and resource.get("pass") is True,
        "VRAM gate failed",
    )
    _require(
        original.get("authority", {}) == {
            "measurement_only": True,
            "training_or_model_update_performed": False,
            "dataset_mutation_performed": False,
            "checkpoint_or_config_promoted": False,
        },
        "measurement receipt authority changed",
    )
    return receipt


def validate_promotion_v77(
    preregistration: dict[str, Any], receipt: dict[str, Any]
) -> dict[str, Any]:
    validate_measurement_receipt_v77(preregistration, receipt)
    # This CPU-only preregistration intentionally cannot promote anything.
    _require(
        preregistration.get("status") == "launch_authorized_after_environment_fix"
        and preregistration.get("authority", {}).get("checkpoint_or_config_promotion") is True,
        "promotion blocked by CPU-only preregistration authority",
    )
    return receipt
