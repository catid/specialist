#!/usr/bin/env python3
"""Data-free, CPU-only preflight for Qwen3.6 frozen-base precision arms.

This builder deliberately does not import torch or vLLM and opens safetensor
headers through the NumPy surface on CPU only.  It proves that a launch *could* use the exact BF16 and
serialized FP8 artifacts below; it does not claim that a CUDA kernel, LoRA
switch, throughput, validation, or OOD gate has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_quantized_base_ablation_v67.json"
)
VLLM_ROOT = (
    ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/vllm"
)
BF16_MODEL = ROOT / "models/Qwen3.6-35B-A3B"
FP8_MODEL = ROOT / "models/Qwen3.6-35B-A3B-FP8"
FP8_FULL_MODEL_NEGATIVE_EVIDENCE = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V29I_V29H_FP8_FULL_MODEL_RUNTIME_AB_NEGATIVE_EVIDENCE.json"
)
FP8_FULL_MODEL_NEGATIVE_EVIDENCE_FILE_SHA256 = (
    "c40f01f23b4a3a47cc53ed282045d3c2d34bbe594ecf43a9717e5e21b4a32a65"
)
FP8_FULL_MODEL_NEGATIVE_EVIDENCE_CONTENT_SHA256 = (
    "24c269796194fe997f8e641ff25e7702b775e45449022917f5a498db354d204c"
)

SOURCE_SHA256 = {
    "config/model.py": (
        "41cdd96fd1c9f141a96938fa2d08032cfcb89a4925d7d10b64ef937884791908"
    ),
    "model_executor/models/qwen3_5.py": (
        "5f47ae4f4a08d0a78dd681d58b290f3298744c73a82f1349f3e2853469ef73e6"
    ),
    "model_executor/layers/quantization/fp8.py": (
        "26ede801aceeadea1e263b854424ed4fbd0bf5c081bf5beceaf4b877fb1d4fc0"
    ),
    "model_executor/layers/quantization/auto_awq.py": (
        "d4a8fc645e0e4a49a79e0869ce58754d7bd6f9c89103336cfb35b72442d7b534"
    ),
    "model_executor/layers/quantization/auto_gptq.py": (
        "2e50123fa82358bc3431f8c5afb01747e5c2b797a4c9120826cb36ce2181ce21"
    ),
    "model_executor/layers/quantization/bitsandbytes.py": (
        "9817c93811bde5967ee9eed1efa551b8fc9e857e59fbd1b033d16c82ca42f112"
    ),
    (
        "model_executor/layers/quantization/compressed_tensors/"
        "compressed_tensors.py"
    ): "3e7992232412b7d761f599cd14794bdc3a5757128a05fa81e7b2b2ed8578b4f7",
    (
        "model_executor/layers/quantization/compressed_tensors/"
        "compressed_tensors_moe/compressed_tensors_moe.py"
    ): "36d9f20271434f85f5d2adc8339655511db968d10685e8ab5c78c1f049923837",
    "model_executor/layers/fused_moe/modular_kernel.py": (
        "0722a23b0526c141206d17aa6472a610aca77d24f80fa72194c44d320a133687"
    ),
    "lora/model_manager.py": (
        "13201a06e17cccffb30c90bf3d268dfbf901567623b03454003afb5a922ae45a"
    ),
}

CHECKPOINT_EXPECTATIONS = {
    "bf16": {
        "path": BF16_MODEL,
        "config_sha256": (
            "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
        ),
        "index_sha256": (
            "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
        ),
        "index_key_count": 1045,
        "shard_count": 26,
        "shard_bytes": 71_903_776_776,
        "shard_size_manifest_sha256": (
            "b878392051d6a40017ea4d096c786470aee2dab5e0440b33bb1b7b92142c60c9"
        ),
        "tensor_metadata_manifest_sha256": (
            "7e5634b3166be1f66cbba4830a34fcabb8e2d1c0ea87c2423943dd41c8e1d95a"
        ),
        "dtype_tensor_counts": {"BF16": 1045},
        "logical_tensor_bytes": 71_903_645_408,
    },
    "fp8_serialized": {
        "path": FP8_MODEL,
        "config_sha256": (
            "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
        ),
        "index_sha256": (
            "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6"
        ),
        "index_key_count": 64_196,
        "shard_count": 42,
        "shard_bytes": 37_463_662_160,
        "shard_size_manifest_sha256": (
            "3bc687cefc89541a4768655cd110ffecf1d6a1ca94b5072903037dab119dcdaf"
        ),
        "tensor_metadata_manifest_sha256": (
            "d66cb081f921a46d81f5b39e4dd2c5d97d673de2e74f43f1d50cc42eca4cfcc7"
        ),
        "dtype_tensor_counts": {"BF16": 32_451, "F8_E4M3": 31_745},
        "logical_tensor_bytes": 37_454_789_472,
        "fp8_expert_weight_count": 31_488,
        "fp8_scale_count": 31_745,
        "bf16_router_count": 41,
    },
}

DTYPE_BITS = {
    "BF16": 16,
    "F16": 16,
    "F32": 32,
    "F8_E4M3": 8,
    "F8_E5M2": 8,
    "I8": 8,
    "U8": 8,
    "I32": 32,
    "U32": 32,
}


def canonical_sha256_v67(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256_v67(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_no_duplicates(path: Path) -> dict:
    def hook(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _shape_numel(shape: list[int]) -> int:
    value = 1
    for dimension in shape:
        _require(
            isinstance(dimension, int) and dimension >= 0,
            "invalid safetensor dimension",
        )
        value *= dimension
    return value


def _safe_shard_names(weight_map: dict) -> list[str]:
    _require(weight_map, "empty checkpoint weight map")
    names = set()
    for key, name in weight_map.items():
        _require(isinstance(key, str) and key, "invalid checkpoint tensor name")
        _require(
            isinstance(name, str)
            and Path(name).name == name
            and name.endswith(".safetensors"),
            "unsafe checkpoint shard reference",
        )
        names.add(name)
    return sorted(names)


def inspect_checkpoint_v67(label: str, expected: dict | None = None) -> dict:
    if expected is None:
        expected = CHECKPOINT_EXPECTATIONS[label]
    root = Path(expected["path"])
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    _require(root.is_dir() and not root.is_symlink(), f"missing model: {root}")
    _require(
        config_path.is_file()
        and not config_path.is_symlink()
        and index_path.is_file()
        and not index_path.is_symlink(),
        f"missing checkpoint metadata: {root}",
    )
    _require(
        file_sha256_v67(config_path) == expected["config_sha256"]
        and file_sha256_v67(index_path) == expected["index_sha256"],
        f"{label} config or index identity changed",
    )
    config = _json_no_duplicates(config_path)
    index = _json_no_duplicates(index_path)
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict), f"{label} weight map missing")
    shard_names = _safe_shard_names(weight_map)
    _require(
        len(weight_map) == expected["index_key_count"]
        and len(shard_names) == expected["shard_count"],
        f"{label} checkpoint cardinality changed",
    )
    size_records = []
    for name in shard_names:
        shard = root / name
        _require(
            shard.is_file() and not shard.is_symlink(),
            f"{label} checkpoint shard missing or symlinked: {name}",
        )
        size_records.append({"file": name, "bytes": shard.stat().st_size})
    _require(
        sum(row["bytes"] for row in size_records) == expected["shard_bytes"]
        and canonical_sha256_v67(size_records)
        == expected["shard_size_manifest_sha256"],
        f"{label} checkpoint shard size surface changed",
    )

    dtype_counts: Counter[str] = Counter()
    logical_bytes = 0
    metadata_digest = hashlib.sha256()
    seen = set()
    fp8_expert_weights = 0
    fp8_scales = 0
    bf16_routers = 0
    for name in shard_names:
        with safe_open(root / name, framework="numpy", device="cpu") as source:
            for key in sorted(source.keys()):
                _require(key not in seen, f"duplicate checkpoint tensor: {key}")
                _require(
                    weight_map.get(key) == name,
                    f"{label} index points tensor to a different shard: {key}",
                )
                seen.add(key)
                view = source.get_slice(key)
                dtype = str(view.get_dtype())
                shape = list(view.get_shape())
                _require(dtype in DTYPE_BITS, f"unsealed tensor dtype: {dtype}")
                dtype_counts[dtype] += 1
                logical_bytes += _shape_numel(shape) * DTYPE_BITS[dtype] // 8
                record = {
                    "dtype": dtype,
                    "file": name,
                    "key": key,
                    "shape": shape,
                }
                encoded = json.dumps(
                    record, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                metadata_digest.update(len(encoded).to_bytes(8, "big"))
                metadata_digest.update(encoded)
                if (
                    dtype == "F8_E4M3"
                    and ".mlp.experts." in key
                    and key.endswith(
                        (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
                    )
                ):
                    fp8_expert_weights += 1
                if dtype == "BF16" and key.endswith(".weight_scale_inv"):
                    fp8_scales += 1
                if (
                    dtype == "BF16"
                    and ".mlp.gate." in key
                    and key.endswith(".weight")
                ):
                    bf16_routers += 1
    _require(seen == set(weight_map), f"{label} index/tensor key surface changed")
    _require(
        dict(sorted(dtype_counts.items())) == expected["dtype_tensor_counts"]
        and logical_bytes == expected["logical_tensor_bytes"]
        and metadata_digest.hexdigest()
        == expected["tensor_metadata_manifest_sha256"],
        f"{label} safetensor metadata or dtype surface changed",
    )

    text = config.get("text_config", {})
    geometry_ok = (
        config.get("architectures") == ["Qwen3_5MoeForConditionalGeneration"]
        and config.get("model_type") == "qwen3_5_moe"
        and text.get("model_type") == "qwen3_5_moe_text"
        and text.get("dtype") == "bfloat16"
        and text.get("hidden_size") == 2048
        and text.get("num_hidden_layers") == 40
        and text.get("num_experts") == 256
        and text.get("num_experts_per_tok") == 8
        and text.get("full_attention_interval") == 4
    )
    _require(geometry_ok, f"{label} Qwen geometry changed")
    quant = config.get("quantization_config")
    if label == "bf16":
        _require(quant is None, "BF16 checkpoint unexpectedly declares quantization")
    else:
        _require(
            isinstance(quant, dict)
            and quant.get("quant_method") == "fp8"
            and quant.get("fmt") == "e4m3"
            and quant.get("activation_scheme") == "dynamic"
            and quant.get("weight_block_size") == [128, 128]
            and len(quant.get("modules_to_not_convert", [])) == 648
            and fp8_expert_weights == expected["fp8_expert_weight_count"]
            and fp8_scales == expected["fp8_scale_count"]
            and bf16_routers == expected["bf16_router_count"],
            "serialized FP8 format, expert, scale, or router surface changed",
        )
    return {
        "path": str(root),
        "config_sha256": expected["config_sha256"],
        "index_sha256": expected["index_sha256"],
        "index_key_count": len(weight_map),
        "shard_count": len(shard_names),
        "shard_bytes": sum(row["bytes"] for row in size_records),
        "shard_size_manifest_sha256": canonical_sha256_v67(size_records),
        "tensor_metadata_manifest_sha256": metadata_digest.hexdigest(),
        "dtype_tensor_counts": dict(sorted(dtype_counts.items())),
        "logical_tensor_bytes": logical_bytes,
        "fp8_expert_weight_count": fp8_expert_weights,
        "fp8_scale_count": fp8_scales,
        "bf16_router_count": bf16_routers,
    }


def inspect_installed_stack_v67() -> dict:
    versions = {}
    for package in (
        "vllm",
        "torch",
        "transformers",
        "compressed-tensors",
        "bitsandbytes",
        "autoawq",
        "auto-gptq",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    _require(
        versions == {
            "auto-gptq": None,
            "autoawq": None,
            "bitsandbytes": None,
            "compressed-tensors": "0.17.0",
            "torch": "2.11.0",
            "transformers": "5.13.1",
            "vllm": "0.25.0",
        },
        "pinned quantized-base package surface changed",
    )
    source_receipts = {}
    for relative, expected in SOURCE_SHA256.items():
        path = VLLM_ROOT / relative
        _require(
            path.is_file() and file_sha256_v67(path) == expected,
            f"pinned vLLM quantization source changed: {relative}",
        )
        source_receipts[relative] = expected

    qwen_source = (VLLM_ROOT / "model_executor/models/qwen3_5.py").read_text()
    fp8_source = (
        VLLM_ROOT / "model_executor/layers/quantization/fp8.py"
    ).read_text()
    bnb_source = (
        VLLM_ROOT / "model_executor/layers/quantization/bitsandbytes.py"
    ).read_text()
    _require(
        "SupportsLoRA" in qwen_source
        and "is_3d_moe_weight: bool = True" in qwen_source
        and "return Fp8MoEMethod(self, layer)" in fp8_source
        and "return BitsAndBytesMoEMethod(self, layer.moe_config)" in bnb_source
        and 'min_version = "0.49.2" if current_platform.is_rocm() else "0.48.1"'
        in bnb_source,
        "installed Qwen/LoRA/MoE loader support markers changed",
    )
    return {"package_versions": versions, "source_sha256": source_receipts}


def inspect_prior_fp8_runtime_evidence_v67() -> dict:
    _require(
        FP8_FULL_MODEL_NEGATIVE_EVIDENCE.is_file()
        and file_sha256_v67(FP8_FULL_MODEL_NEGATIVE_EVIDENCE)
        == FP8_FULL_MODEL_NEGATIVE_EVIDENCE_FILE_SHA256,
        "prior full-model FP8 negative evidence file changed",
    )
    evidence = _json_no_duplicates(FP8_FULL_MODEL_NEGATIVE_EVIDENCE)
    compact = {
        key: item
        for key, item in evidence.items()
        if key != "content_sha256_before_self_field"
    }
    performance = evidence.get("aggregate_performance", {})
    global_latency = performance.get("global_full_model_latency", {})
    _require(
        evidence.get("content_sha256_before_self_field")
        == FP8_FULL_MODEL_NEGATIVE_EVIDENCE_CONTENT_SHA256
        and canonical_sha256_v67(compact)
        == FP8_FULL_MODEL_NEGATIVE_EVIDENCE_CONTENT_SHA256
        and evidence.get("decision", {}).get(
            "retain_empty_default_serialized_fp8_runtime"
        )
        is True
        and performance.get("all_five_latency_endpoints_failed") is True
        and global_latency.get("geometric_mean_speedup")
        == 0.9813847810295329
        and global_latency.get("pass") is False,
        "prior full-model FP8 negative evidence content changed",
    )
    return {
        "path": str(FP8_FULL_MODEL_NEGATIVE_EVIDENCE),
        "file_sha256": FP8_FULL_MODEL_NEGATIVE_EVIDENCE_FILE_SHA256,
        "content_sha256": FP8_FULL_MODEL_NEGATIVE_EVIDENCE_CONTENT_SHA256,
        "all_five_latency_endpoints_failed": True,
        "global_geometric_mean_tuned_over_default_speedup": (
            global_latency["geometric_mean_speedup"]
        ),
        "required_starting_fp8_moe_tuning_table": "empty_default",
    }


def discover_local_int4_checkpoints_v67(model_root: Path | None = None) -> list[dict]:
    model_root = ROOT / "models" if model_root is None else Path(model_root)
    found = []
    if not model_root.is_dir():
        return found
    for config_path in sorted(model_root.glob("*/config.json")):
        config = _json_no_duplicates(config_path)
        quant = config.get("quantization_config")
        if not isinstance(quant, dict):
            continue
        method = str(quant.get("quant_method", "")).lower()
        serialized_int4 = (
            method in {"awq", "gptq"}
            and int(quant.get("bits", quant.get("w_bit", 0)) or 0) == 4
        ) or (
            method == "compressed-tensors"
            and "4" in json.dumps(quant, sort_keys=True)
        ) or (
            method == "bitsandbytes" and quant.get("load_in_4bit") is True
        )
        if serialized_int4:
            found.append(
                {
                    "path": str(config_path.parent),
                    "quant_method": method,
                    "config_sha256": file_sha256_v67(config_path),
                }
            )
    return found


def engine_contract_v67(model: Path, quantization: str | None) -> dict:
    return {
        "model": str(model),
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "quantization": quantization,
        "kv_cache_dtype": "auto",
        "gpu_memory_utilization": 0.82,
        "max_model_len": 2048,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "async_scheduling": False,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "mm_processor_cache_gb": 0,
        "skip_mm_profiling": True,
        "moe_backend": "triton",
        "enable_lora": True,
        "max_lora_rank": 32,
        "max_loras": 1,
        "max_cpu_loras": 2,
        "lora_dtype": "auto_from_bfloat16_base",
    }


def build_preregistration_v67() -> dict:
    stack = inspect_installed_stack_v67()
    prior_fp8 = inspect_prior_fp8_runtime_evidence_v67()
    bf16 = inspect_checkpoint_v67("bf16")
    fp8 = inspect_checkpoint_v67("fp8_serialized")
    int4 = discover_local_int4_checkpoints_v67()
    _require(int4 == [], "an unsealed local INT4 checkpoint appeared")
    saved = bf16["logical_tensor_bytes"] - fp8["logical_tensor_bytes"]
    _require(saved == 34_448_855_936, "frozen-base storage delta changed")
    value = {
        "schema": "qwen36-quantized-base-ablation-preregistration-v67",
        "status": "cpu_preflight_passed_gpu_launch_waits_for_specialist_0j5_14",
        "data_free": True,
        "cuda_import_or_launch_performed_by_builder": False,
        "dependency": {
            "bead": "specialist-0j5.14",
            "required_before_gpu_launch": True,
            "reason": "reuse the sealed phase/VRAM/bandwidth measurement surface",
        },
        "installed_stack": stack,
        "prior_fp8_runtime_evidence": prior_fp8,
        "model_contract": {
            "architecture": "Qwen3_5MoeForConditionalGeneration",
            "requested_family_name": "Qwen3.6-35B-A3B",
            "layers": 40,
            "routed_experts_per_layer": 256,
            "experts_per_token": 8,
            "full_attention_layers": 10,
            "hybrid_linear_attention_layers": 30,
            "lora_static_support": {
                "model_inherits_SupportsLoRA": True,
                "moe_declares_is_3d_moe_weight": True,
                "static_support_does_not_replace_live_adapter_switch_preflight": True,
            },
        },
        "arms": {
            "bf16": {
                "gpu_preflight_safe_after_dependency": True,
                "scored_evaluation_safe_before_live_preflight": False,
                "checkpoint": bf16,
                "engine_kwargs": engine_contract_v67(BF16_MODEL, None),
                "required_resolved_quantization": None,
                "required_serialized_quantization": False,
                "forbidden_cli": ["--quantization fp8", "--quantization bitsandbytes"],
            },
            "fp8_serialized": {
                "gpu_preflight_safe_after_dependency": True,
                "scored_evaluation_safe_before_live_preflight": False,
                "checkpoint": fp8,
                "engine_kwargs": engine_contract_v67(FP8_MODEL, "fp8"),
                "required_resolved_quantization": "fp8",
                "required_serialized_quantization": True,
                "required_block_shape": [128, 128],
                "bf16_moe_tuning_table_reuse_forbidden": True,
                "rejected_v29_selected_fp8_table_reuse_forbidden": True,
                "required_starting_moe_tuning_table": "empty_default",
                "online_requantization_of_bf16_checkpoint_for_this_arm_forbidden": True,
            },
            "int4": {
                "gpu_preflight_safe_after_dependency": False,
                "scored_evaluation_safe_before_live_preflight": False,
                "local_serialized_checkpoint_candidates": int4,
                "launch_authorized": False,
                "blocking_reasons": [
                    "no local sealed Qwen3.6 INT4 checkpoint",
                    "bitsandbytes is not installed (vLLM requires >=0.48.1 on CUDA)",
                    "LoRA plus 3D MoE correctness is not proven for an INT4 kernel",
                    "AWQ/GPTQ/CompressedTensors backend selection and fallback are unobserved",
                ],
                "potential_paths_requiring_new_preregistration": {
                    "compressed_tensors_w4a16": {
                        "dependency_installed": "compressed-tensors==0.17.0",
                        "vllm_quantization": "compressed-tensors",
                        "vllm_has_routed_expert_methods": True,
                        "preferred_first_serialized_int4_path": True,
                    },
                    "awq_w4a16": {
                        "vllm_quantization": "auto_awq",
                        "vllm_has_routed_expert_methods": True,
                        "fallback_risk": "AutoAWQ MoE Marlin may select Moe WNA16",
                    },
                    "gptq_w4a16": {
                        "vllm_quantization": "auto_gptq",
                        "vllm_has_routed_expert_methods": True,
                        "fallback_risk": "GPTQ MoE Marlin may select Moe WNA16",
                    },
                    "bitsandbytes_online_nf4_or_fp4": {
                        "vllm_quantization": "bitsandbytes",
                        "dependency_installed": False,
                        "vllm_has_routed_expert_method": True,
                        "bandwidth_risk": (
                            "installed method dequantizes routed expert weights before "
                            "the generic fused_experts call on the hot path"
                        ),
                    },
                },
            },
        },
        "alternative_supported_but_excluded_paths": {
            "online_fp8_from_bf16": {
                "vllm_cli": {"model": str(BF16_MODEL), "quantization": "fp8"},
                "supported_by_installed_source": True,
                "excluded_reason": (
                    "would conflate online lossy conversion/startup work with the "
                    "sealed serialized-FP8 checkpoint comparison"
                ),
            },
            "mxfp4_or_nvfp4": {
                "excluded_reason": "4-bit floating point is not the requested INT4 arm"
            },
        },
        "memory_expectations": {
            "bf16_logical_weight_bytes": bf16["logical_tensor_bytes"],
            "fp8_logical_weight_and_scale_bytes": fp8["logical_tensor_bytes"],
            "logical_bytes_saved_by_fp8": saved,
            "fraction_of_bf16_logical_bytes_saved": saved
            / bf16["logical_tensor_bytes"],
            "nominal_full_attention_bf16_kv_bytes_per_token": 20_480,
            "kv_cache_dtype_is_identical_auto_bf16_in_both_arms": True,
            "weight_format_does_not_directly_reduce_kv_bytes_per_token": True,
            "fp8_should_increase_available_kv_blocks_at_fixed_gpu_fraction": True,
            "hybrid_linear_attention_state_and_allocator_padding_must_be_measured": True,
            "logical_checkpoint_bytes_are_not_a_claim_about_cuda_allocated_bytes": True,
        },
        "paired_gpu_preflight": {
            "physical_gpus": [0, 1, 2, 3],
            "wave_1": {"bf16": [0, 2], "fp8_serialized": [1, 3]},
            "wave_2": {"bf16": [1, 3], "fp8_serialized": [0, 2]},
            "all_four_gpus_must_have_live_actor_and_positive_activity_witness": True,
            "same_prompts_generation_params_and_adapter_state": True,
            "separate_format_specific_moe_tuning_tables": True,
            "measure": [
                "post_load_and_peak_vram_bytes",
                "available_kv_cache_memory_bytes",
                "gpu_and_cpu_kv_block_counts",
                "prefill_and_decode_tokens_per_second",
                "candidate_switch_and_restore_latency",
                "dram_read_write_bytes_or_best_available_bandwidth_proxy",
                "sm_and_memory_controller_utilization",
            ],
        },
        "required_live_receipt": {
            "load_and_generate_passed": True,
            "resolved_quantization_exactly_matches_arm": True,
            "serialized_format_exactly_matches_arm": True,
            "fallback_messages_must_equal": [],
            "unexpected_unquantized_module_prefixes_must_equal": [],
            "fp8_routed_expert_method_count_must_equal": 40,
            "adapter_load_switch_candidate_restore_passed": True,
            "candidate_changes_output": True,
            "restore_reproduces_exact_deterministic_output": True,
            "canonical_fp32_master_identity_unchanged": True,
            "frozen_base_identity_unchanged": True,
            "all_four_actor_receipts_reach_consensus": True,
        },
        "fallback_policy": {
            "forbidden_case_insensitive_log_fragments": [
                "falling back",
                "fallback to",
                "not supported by",
                "please install bitsandbytes",
            ],
            "warning_only_acceptance_for_unknown_or_different_format": False,
            "runtime_config_autodetection_without_exact_attestation": False,
            "unquantized_routed_expert_fallback_in_quantized_arm": False,
            "score_or_validation_on_failed_preflight": False,
        },
        "authority": {
            "cpu_preparation_complete": True,
            "bf16_gpu_preflight_after_0j5_14": True,
            "fp8_serialized_gpu_preflight_after_0j5_14": True,
            "int4_gpu_launch": False,
            "scored_validation_before_per_arm_live_preflight": False,
            "ood_or_holdout_opened_by_builder": False,
            "training_or_model_update_performed_by_builder": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v67(value)
    return value


def validate_preregistration_v67(value: dict) -> dict:
    expected = build_preregistration_v67()
    _require(value == expected, "V67 preregistration contract changed")
    _require(
        value.get("content_sha256_before_self_field")
        == canonical_sha256_v67(
            {
                key: item
                for key, item in value.items()
                if key != "content_sha256_before_self_field"
            }
        ),
        "V67 preregistration self hash changed",
    )
    return value


def _sha(value: Any, label: str) -> str:
    value = str(value)
    _require(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"invalid {label} SHA-256",
    )
    return value


def forbidden_fallback_messages_v67(
    preregistration: dict, log_text: str
) -> list[str]:
    """Extract forbidden loader/kernel fallback lines case-insensitively."""
    fragments = preregistration["fallback_policy"][
        "forbidden_case_insensitive_log_fragments"
    ]
    found = []
    for line in str(log_text).splitlines():
        normalized = line.casefold()
        if any(fragment.casefold() in normalized for fragment in fragments):
            stripped = line.strip()
            if stripped and stripped not in found:
                found.append(stripped)
    return found


def validate_live_receipt_v67(preregistration: dict, receipt: dict) -> dict:
    """Fail closed on a later GPU preflight receipt.

    The runtime probe is intentionally separate: this validator gives it a
    small, explicit evidence surface and is fully unit-testable on CPU.
    """
    validate_preregistration_v67(preregistration)
    _require(isinstance(receipt, dict), "V67 live receipt object required")
    arm = receipt.get("arm")
    _require(arm in {"bf16", "fp8_serialized"}, "V67 unsafe live arm")
    contract = preregistration["arms"][arm]
    _require(
        receipt.get("schema") == "qwen36-quantized-base-live-preflight-v67"
        and receipt.get("actor_count") == 4
        and receipt.get("physical_gpus") == [0, 1, 2, 3]
        and receipt.get("load_and_generate_passed") is True
        and receipt.get("resolved_dtype") == "bfloat16"
        and receipt.get("resolved_quantization")
        == contract["required_resolved_quantization"]
        and receipt.get("serialized_quantization")
        is contract["required_serialized_quantization"]
        and receipt.get("fallback_messages") == []
        and receipt.get("unexpected_unquantized_module_prefixes") == [],
        "V67 load, format, or fallback receipt failed",
    )
    methods = receipt.get("routed_expert_method_counts")
    if arm == "fp8_serialized":
        _require(
            methods == {"Fp8MoEMethod": 40}
            and receipt.get("fp8_block_shape") == [128, 128],
            "V67 FP8 routed-expert kernel receipt failed",
        )
    else:
        _require(
            methods == {"UnquantizedFusedMoEMethod": 40},
            "V67 BF16 routed-expert method receipt failed",
        )
    state = receipt.get("adapter_state", {})
    _require(
        state.get("load_switch_candidate_restore_passed") is True
        and state.get("candidate_changed_output") is True
        and state.get("restore_exact_output") is True
        and _sha(state.get("master_sha256_before"), "master before")
        == _sha(state.get("master_sha256_after"), "master after")
        and _sha(state.get("base_sha256_before"), "base before")
        == _sha(state.get("base_sha256_after"), "base after"),
        "V67 LoRA candidate continuity receipt failed",
    )
    measurements = receipt.get("measurements", {})
    required_measurements = (
        "post_load_vram_bytes",
        "peak_vram_bytes",
        "available_kv_cache_memory_bytes",
        "gpu_kv_block_count",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "candidate_switch_seconds",
        "restore_seconds",
    )
    _require(
        all(
            isinstance(measurements.get(key), (int, float))
            and not isinstance(measurements.get(key), bool)
            and math.isfinite(float(measurements[key]))
            and float(measurements[key]) >= 0.0
            for key in required_measurements
        ),
        "V67 memory or throughput measurements missing",
    )
    _require(
        receipt.get("all_four_actor_receipts_consensus") is True
        and receipt.get("all_four_gpus_positive_activity_witness") is True,
        "V67 four-GPU consensus/activity receipt failed",
    )
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, default=OUTPUT)
    value.add_argument("--check", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    value = build_preregistration_v67()
    if args.check:
        frozen = _json_no_duplicates(args.output)
        validate_preregistration_v67(frozen)
        print(json.dumps({
            "schema": frozen["schema"],
            "status": frozen["status"],
            "content_sha256": frozen["content_sha256_before_self_field"],
        }, sort_keys=True))
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
