#!/usr/bin/env python3
"""Build the CPU-only Qwen3.6 fused-MoE-LoRA tuning preregistration.

The builder reads only explicitly authorized V73/V74/V75/V76 runtime artifacts,
model/adapter metadata, installed-source text, and compact V29 negative
evidence.  It imports neither torch nor vLLM and cannot initialize CUDA.
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import qwen36_fused_moe_lora_tuning_v77 as contract


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fused_moe_lora_kernel_tuning_v77.json"
)
VLLM_ROOT = ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/vllm"
SITE_PACKAGES = VLLM_ROOT.parent
MODEL_CONFIG = ROOT / "models/Qwen3.6-35B-A3B/config.json"
REFERENCE_ADAPTER = ROOT / (
    "experiments/eggroll_es_hpo/staged_adapters/"
    "v434_equal_sft_qwen35_vllm_namespace_v49d"
)
CANDIDATE_ADAPTER = ROOT / (
    "experiments/eggroll_es_hpo/runs/v59_lora_es_fragile_priority/"
    "selected_candidate_v59"
)
NEGATIVE_EVIDENCE = ROOT / (
    "experiments/eggroll_es_hpo/"
    "S6_V29I_V29H_FP8_FULL_MODEL_RUNTIME_AB_NEGATIVE_EVIDENCE.json"
)

RUN_ROOTS = {
    "v73_wave1": ROOT / "experiments/eggroll_es_hpo/runs/v73_quantized_base_paired_wave1",
    "v73_wave2": ROOT / "experiments/eggroll_es_hpo/runs/v73_quantized_base_paired_wave2",
    "v74": ROOT / "experiments/eggroll_es_hpo/runs/v74_fp8_rightsized_050_r1",
    "v75": ROOT / "experiments/eggroll_es_hpo/runs/v75_fp8_clean_050_r1",
    "v76": ROOT / "experiments/eggroll_es_hpo/runs/v76_fp8_attested_050_r5",
}

RUN_BUNDLE_SHA256 = {
    "v73_wave1": "a549328f469a4078037291f79ca1feade23f5961993327ef2820001e838efd15",
    "v73_wave2": "aa9ef6b9ec63a587039224ed7fc70860c323dd4909a04870b9173ae69f34e4c8",
    "v74": "41f637a0fd96a33a1237e80a4182dab75275019127ce0a925f8b732e836c100a",
    "v75": "742d81d0747733de69f7053ccb89918edf374e090cc09bcba4bb374e454cd36c",
    "v76": "5124652dc91af81de6e55c66d5eaa6b8a6b355a85f50369052d960eb5c028d87",
}

RUN_FILE_COUNTS = {
    "v73_wave1": 10,
    "v73_wave2": 10,
    "v74": 10,
    "v75": 10,
    "v76": 9,
}

SOURCE_SHA256 = {
    "config/vllm.py": "caf6db4dbbafb3e2194022d779e4635e78ea1f51bfc5d299997640cd2806cb05",
    "envs.py": "15ab853b73b26da5dc2808699138dff7217b72b9f661da274cc0c9f6c262f631",
    "lora/ops/triton_ops/utils.py": "ba7110f5f52b6b5172aede8cb15da4fac93358a0d0c7c4004743ed5a1c979343",
    "lora/punica_wrapper/punica_gpu.py": "d0c8ce69191d733d479e50399bfc86d6b89a55915dda4dd0cf32b8921e32bf9d",
    "lora/ops/triton_ops/fused_moe_lora_op.py": "c14320405b2bada038fe3cfc924010245007146793d6b8af4f206a07acf4faa2",
    "lora/ops/triton_ops/fused_moe_lora_fp8_op.py": "68f47e94ac7d17b15eef2df817ca514a265c198d452538bdc81555f9cdb012d6",
    "lora/layers/fused_moe.py": "62dcf28f2af0906b420c75bba171dcf4e5a392e02dc9860c32fad88f46b95b7e",
    "model_executor/layers/fused_moe/fused_moe.py": "72811a4e543cc6f415f184cb951b61522643cddc4d6456f61a2f8c1a53b2cf79",
    "model_executor/models/qwen3_5.py": "5f47ae4f4a08d0a78dd681d58b290f3298744c73a82f1349f3e2853469ef73e6",
}

ADAPTER_EXPECTATIONS = {
    "reference": {
        "root": REFERENCE_ADAPTER,
        "config_sha256": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
        "weights_sha256": "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a",
    },
    "candidate": {
        "root": CANDIDATE_ADAPTER,
        "config_sha256": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
        "weights_sha256": "c2665b60928b16120a2b98fdf137fafd250644852c86a02d797689f02105c6c8",
    },
}

LOG_PATTERNS = {
    "deepgemm_warning": "Auto-disabled DeepGemm",
    "deepgemm_e8m0_enabled": "DeepGEMM E8M0 enabled on current platform",
    "missing_lora_config": "No LoRA kernel configs found",
    "default_lora_config": "Using default LoRA kernel configs",
    "jit_lora_shrink": "Triton kernel JIT compilation during inference: _lora_shrink_kernel",
    "jit_lora_expand": "Triton kernel JIT compilation during inference: _lora_expand_kernel",
    "jit_fused_moe_lora": "Triton kernel JIT compilation during inference: _fused_moe_lora_one_shot_kernel",
    "jit_base_fused_moe": "Triton kernel JIT compilation during inference: fused_moe_kernel",
    "flashinfer_autotune_skipped": "Skipping FlashInfer autotune because it is disabled",
    "routed_triton_backend": "Using TRITON Fp8 MoE backend",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _validate_receipt_self_hash(value: dict[str, Any], path: Path) -> None:
    claimed = value.get("content_sha256_before_self_field")
    body = copy.deepcopy(value)
    body.pop("content_sha256_before_self_field", None)
    _require(
        claimed == contract.canonical_sha256_v77(body),
        f"receipt self hash changed: {path}",
    )


def _run_inventory(label: str, root: Path) -> dict[str, Any]:
    _require(root.is_dir() and not root.is_symlink(), f"missing run directory: {root}")
    rows = []
    for path in sorted(root.iterdir()):
        if path.is_file():
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": contract.file_sha256_v77(path),
                }
            )
    _require(len(rows) == RUN_FILE_COUNTS[label], f"{label} run file count changed")
    bundle = contract.canonical_sha256_v77(rows)
    _require(bundle == RUN_BUNDLE_SHA256[label], f"{label} run inventory changed")
    return {"file_count": len(rows), "bundle_sha256": bundle, "files": rows}


def _read_run(label: str, root: Path) -> dict[str, Any]:
    inventory = _run_inventory(label, root)
    log_paths = sorted(root.glob("*.log"))
    receipt_paths = sorted(
        root.glob("gpu_*.json") if label == "v76" else root.glob("*receipt.json")
    )
    _require(len(log_paths) == 5 and len(receipt_paths) == 4, f"{label} actor surface changed")
    # nvidia_smi_samples.log is telemetry rather than actor stdout.
    actor_logs = [path for path in log_paths if path.name != "nvidia_smi_samples.log"]
    _require(len(actor_logs) == 4, f"{label} actor-log count changed")
    texts = [path.read_text(encoding="utf-8", errors="strict") for path in actor_logs]
    receipts = []
    for path in receipt_paths:
        value = _json(path)
        _validate_receipt_self_hash(value, path)
        _require(
            value.get("source_dataset_rows_opened") == 0
            and value.get("protected_ood_shadow_or_terminal_opened") is False
            and value.get("adapter_update_or_hpo_performed") is False,
            f"{label} receipt exceeded data-free diagnostic authority",
        )
        receipts.append(value)
    pattern_counts = {
        key: sum(text.count(pattern) for text in texts)
        for key, pattern in LOG_PATTERNS.items()
    }
    actor_pattern_counts = {
        key: sum(pattern in text for text in texts)
        for key, pattern in LOG_PATTERNS.items()
    }
    return {
        "inventory": inventory,
        "pattern_counts": pattern_counts,
        "actor_pattern_counts": actor_pattern_counts,
        "receipts": receipts,
    }


def _installed_version(project: str) -> str:
    candidates = sorted(SITE_PACKAGES.glob(f"{project}-*.dist-info/METADATA"))
    _require(len(candidates) == 1, f"installed metadata changed for {project}")
    for line in candidates[0].read_text(encoding="utf-8").splitlines():
        if line.startswith("Version: "):
            return line.split(": ", 1)[1]
    raise RuntimeError(f"missing installed version for {project}")


def inspect_installed_source_v77() -> dict[str, Any]:
    rows = []
    for relative, expected in SOURCE_SHA256.items():
        path = VLLM_ROOT / relative
        _require(path.is_file() and not path.is_symlink(), f"missing installed source: {relative}")
        actual = contract.file_sha256_v77(path)
        _require(actual == expected, f"installed vLLM source changed: {relative}")
        rows.append({"relative_path": relative, "sha256": actual, "bytes": path.stat().st_size})
    utils_text = (VLLM_ROOT / "lora/ops/triton_ops/utils.py").read_text(encoding="utf-8")
    vllm_text = (VLLM_ROOT / "config/vllm.py").read_text(encoding="utf-8")
    _require(
        'k, n = (hidden_size, rank) if op_type == "shrink" else (rank, hidden_size)' in utils_text,
        "installed LoRA lookup semantics changed",
    )
    _require(
        "and self.quant_config.use_deep_gemm is None" in vllm_text
        and "should_auto_disable_deep_gemm(model_type)" in vllm_text
        and "envs.VLLM_USE_DEEP_GEMM" not in vllm_text,
        "installed DeepGemm warning behavior changed; rebuild preregistration",
    )
    versions = {
        project: _installed_version(project)
        for project in ("vllm", "torch", "triton", "safetensors")
    }
    _require(
        versions == {
            "vllm": "0.25.0",
            "torch": "2.11.0",
            "triton": "3.6.0",
            "safetensors": "0.8.0",
        },
        "installed runtime versions changed",
    )
    return {
        "vllm_root": str(VLLM_ROOT.relative_to(ROOT)),
        "vllm_version": versions["vllm"],
        "versions": versions,
        "files": rows,
        "bundle_sha256": contract.canonical_sha256_v77(rows),
        "loader_behavior": {
            "dense_shrink_K_N": ["hidden_size", "rank"],
            "all_other_operations_K_N": ["rank", "hidden_size"],
            "num_slices_lookup_is_exact": True,
            "M_K_N_and_intermediate_use_nearest_integer_key": True,
        },
        "deepgemm_warning_behavior": {
            "use_deep_gemm_None_checked_before_warning": True,
            "architecture_auto_disable_checked": True,
            "explicit_environment_disable_consulted_by_this_branch": False,
        },
    }


def _safetensor_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        header_size = struct.unpack("<Q", source.read(8))[0]
        _require(0 < header_size < 16 << 20, "unsafe safetensor header length")
        value = json.loads(source.read(header_size))
    _require(isinstance(value, dict), "invalid safetensor header")
    return value


def inspect_adapter_v77(label: str) -> dict[str, Any]:
    expected = ADAPTER_EXPECTATIONS[label]
    root = expected["root"]
    config_path = root / "adapter_config.json"
    weights_path = root / "adapter_model.safetensors"
    _require(
        root.is_dir()
        and not root.is_symlink()
        and config_path.is_file()
        and not config_path.is_symlink()
        and weights_path.is_file()
        and not weights_path.is_symlink(),
        f"missing exact {label} adapter",
    )
    config_hash = contract.file_sha256_v77(config_path)
    weights_hash = contract.file_sha256_v77(weights_path)
    _require(
        config_hash == expected["config_sha256"]
        and weights_hash == expected["weights_sha256"],
        f"{label} adapter identity changed",
    )
    config = _json(config_path)
    _require(
        config.get("r") == 32
        and config.get("layers_to_transform") == [20, 21, 22, 23]
        and config.get("layers_pattern") == "layers",
        f"{label} adapter geometry changed",
    )
    header = _safetensor_header(weights_path)
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    _require(len(tensors) == 70, f"{label} adapter tensor count changed")
    _require({value.get("dtype") for value in tensors.values()} == {"F32"}, f"{label} adapter dtype changed")
    shape_counts = Counter(tuple(value.get("shape", [])) for value in tensors.values())
    expected_counts = {
        (2048, 32): 8,
        (256, 32): 4,
        (32, 2048): 27,
        (32, 32): 6,
        (32, 4096): 4,
        (32, 512): 4,
        (4096, 32): 3,
        (512, 32): 10,
        (8192, 32): 4,
    }
    _require(dict(shape_counts) == expected_counts, f"{label} adapter shapes changed")
    return {
        "relative_path": str(root.relative_to(ROOT)),
        "config_sha256": config_hash,
        "weights_sha256": weights_hash,
        "rank": 32,
        "layers": [20, 21, 22, 23],
        "tensor_count": len(tensors),
        "dtype_counts": {"F32": 70},
        "shape_counts": {
            "x".join(str(item) for item in shape): count
            for shape, count in sorted(shape_counts.items())
        },
        "header_metadata_sha256": contract.canonical_sha256_v77(
            [
                {"name": key, "dtype": value["dtype"], "shape": value["shape"]}
                for key, value in sorted(tensors.items())
            ]
        ),
    }


def inspect_model_geometry_v77() -> dict[str, Any]:
    _require(
        contract.file_sha256_v77(MODEL_CONFIG)
        == "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        "Qwen3.6 model config changed",
    )
    config = _json(MODEL_CONFIG)
    text = config.get("text_config", {})
    expected = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "num_hidden_layers": 40,
        "full_attention_interval": 4,
    }
    _require(
        {key: text.get(key) for key in expected} == expected,
        "Qwen3.6 geometry changed",
    )
    return {
        "model_config_relative_path": str(MODEL_CONFIG.relative_to(ROOT)),
        "model_config_sha256": contract.file_sha256_v77(MODEL_CONFIG),
        **expected,
    }


def inspect_negative_evidence_v77() -> dict[str, Any]:
    _require(
        contract.file_sha256_v77(NEGATIVE_EVIDENCE)
        == "c40f01f23b4a3a47cc53ed282045d3c2d34bbe594ecf43a9717e5e21b4a32a65",
        "V29 compact negative evidence changed",
    )
    value = _json(NEGATIVE_EVIDENCE)
    selected = value["transitively_frozen_contract_identities"]["selected_table"]
    _require(
        {key: selected[key] for key in contract.V29_SELECTED_TABLE_V77}
        == contract.V29_SELECTED_TABLE_V77,
        "V29 selected table identity changed",
    )
    performance = value["aggregate_performance"]
    _require(
        performance.get("all_five_latency_endpoints_failed") is True
        and performance.get("all_five_vram_endpoints_passed") is True
        and performance["global_full_model_latency"]["geometric_mean_speedup"]
        == 0.9813847810295329,
        "V29 negative result changed",
    )
    return {
        "evidence_relative_path": str(NEGATIVE_EVIDENCE.relative_to(ROOT)),
        "evidence_file_sha256": contract.file_sha256_v77(NEGATIVE_EVIDENCE),
        "evidence_content_sha256": value["content_sha256_before_self_field"],
        "identity": contract.V29_SELECTED_TABLE_V77,
        "all_five_latency_endpoints_failed": True,
        "all_five_vram_endpoints_passed": True,
        "global_geometric_mean_tuned_over_default_speedup": 0.9813847810295329,
        "reuse_forbidden": True,
    }


def _validate_v76_actor_v77(value: dict[str, Any]) -> None:
    _require(
        value.get("schema") == "v76-qwen36-fp8-routed-runtime-attestation",
        "V76 actor schema changed",
    )
    workaround = value.get("deepgemm_ordering_workaround", {})
    _require(
        workaround
        == {
            "VLLM_USE_DEEP_GEMM": "0",
            "quant_config_use_deep_gemm_before_post_init": False,
            "schema": "v76-explicit-deepgemm-disable-ordering-workaround",
            "upstream_source_modified": False,
        },
        "V76 DeepGemm disable attestation changed",
    )
    explicit = value.get("explicit_kernel_environment", {})
    _require(
        explicit.get("VLLM_USE_DEEP_GEMM") == "0"
        and explicit.get("enable_flashinfer_autotune") is False
        and explicit.get("tuned_config_folder_was_fresh_and_empty") is True,
        "V76 explicit kernel environment changed",
    )
    audit = value.get("routed_fp8_runtime_attestation", {})
    _require(
        audit.get("schema") == "v76-fp8-routed-model-runtime-audit"
        and audit.get("fp8_moe_method_count") == 40
        and audit.get("fp8_moe_backend_class_counts") == {"Fp8MoeBackend": 40}
        and audit.get("fp8_moe_names_sha256")
        == "2b645a7e4c4488c548549ecd7326411fb7347bf569c78cd8640f9824b2178b55"
        and audit.get("fp8_quant_reference_count") == 80
        and audit.get("fp8_quant_references_sha256")
        == "aaa98d3eac3e160f1ca905a366175ed97d31f27da873fb641d5b1920fdf7eb95"
        and audit.get("moe_runner_module_count") == 40
        and audit.get("routed_like_module_count") == 40
        and audit.get("routed_like_without_fp8_method") == [],
        "V76 routed FP8 audit summary changed",
    )
    records = audit.get("fp8_moe_records")
    _require(isinstance(records, list) and len(records) == 40, "V76 routed owner records changed")
    for record in records:
        _require(
            record.get("module_class") == "RoutedExperts"
            and record.get("quant_method_class") == "Fp8MoEMethod"
            and record.get("runtime_quant_wrapper_class") == "FusedMoEModularMethod"
            and record.get("experts_implementation_class") == "TritonExperts"
            and record.get("fp8_backend_class") == "Fp8MoeBackend"
            and record.get("fp8_backend_name") == "TRITON"
            and record.get("fp8_backend_value") == "TRITON"
            and record.get("block_quant") is True
            and record.get("weight_block_size") == [128, 128]
            and record.get("w13_dtype") == "torch.float8_e4m3fn"
            and record.get("w2_dtype") == "torch.float8_e4m3fn",
            "V76 routed FP8 owner implementation changed",
        )
    runners = audit.get("moe_runner_modules")
    _require(isinstance(runners, list) and len(runners) == 40, "V76 MoE runner records changed")
    _require(
        all(
            row.get("module_class") == "FusedMoE3DWithLoRA"
            and row.get("quant_method_class") == "FusedMoEModularMethod"
            for row in runners
        ),
        "V76 MoE runner implementation changed",
    )


def inspect_observed_evidence_v77() -> dict[str, Any]:
    runs = {label: _read_run(label, root) for label, root in RUN_ROOTS.items()}
    v73_receipts = runs["v73_wave1"]["receipts"] + runs["v73_wave2"]["receipts"]
    by_precision: dict[str, list[float]] = {"bf16": [], "fp8_serialized": []}
    for value in v73_receipts:
        precision = value.get("precision_arm")
        _require(precision in by_precision, "V73 precision arm changed")
        by_precision[precision].append(value["wall_runtime_seconds_excluding_model_load_and_cleanup"])
    _require(all(len(values) == 4 for values in by_precision.values()), "V73 precision pairing changed")
    medians = {key: statistics.median(values) for key, values in by_precision.items()}
    v74_times = [
        value["wall_runtime_seconds_excluding_model_load_and_cleanup"]
        for value in runs["v74"]["receipts"]
    ]
    v75_times = [
        value["wall_runtime_seconds_excluding_model_load_and_cleanup"]
        for value in runs["v75"]["receipts"]
    ]
    v76_times = [
        value["wall_runtime_seconds_excluding_model_load_and_cleanup"]
        for value in runs["v76"]["receipts"]
    ]
    for value in runs["v75"]["receipts"]:
        explicit = value.get("explicit_kernel_environment", {})
        _require(
            explicit.get("VLLM_USE_DEEP_GEMM") == "0"
            and explicit.get("enable_flashinfer_autotune") is False
            and explicit.get("tuned_config_folder_was_fresh_and_empty") is True,
            "V75 explicit kernel environment changed",
        )
        _require(
            value.get("runtime", {}).get("starting_moe_tuning_table")
            == "fresh_empty_default",
            "V75 did not start from explicit empty/default",
        )
    _require(
        runs["v75"]["pattern_counts"] == {
            "deepgemm_warning": 4,
            "deepgemm_e8m0_enabled": 0,
            "missing_lora_config": 24,
            "default_lora_config": 4,
            "jit_lora_shrink": 4,
            "jit_lora_expand": 4,
            "jit_fused_moe_lora": 4,
            "jit_base_fused_moe": 4,
            "flashinfer_autotune_skipped": 4,
            "routed_triton_backend": 4,
        },
        "V75 log evidence changed",
    )
    _require(
        runs["v74"]["pattern_counts"]["deepgemm_e8m0_enabled"] == 4,
        "V74 unsafe DeepGemm E8M0 evidence changed",
    )
    for value in runs["v76"]["receipts"]:
        _validate_v76_actor_v77(value)
    _require(
        runs["v76"]["actor_pattern_counts"]["deepgemm_warning"] == 0
        and runs["v76"]["actor_pattern_counts"]["deepgemm_e8m0_enabled"] == 0
        and runs["v76"]["actor_pattern_counts"]["routed_triton_backend"] == 4,
        "V76 clean TRITON log evidence changed",
    )
    return {
        "run_inventories": {
            label: value["inventory"] for label, value in runs.items()
        },
        "v73": {
            "actor_count": 8,
            "bf16_actor_count": 4,
            "fp8_actor_count": 4,
            "bf16_median_seconds": medians["bf16"],
            "fp8_median_seconds": medians["fp8_serialized"],
            "fp8_over_bf16_runtime_ratio": medians["fp8_serialized"] / medians["bf16"],
            "missing_lora_config_messages": (
                runs["v73_wave1"]["pattern_counts"]["missing_lora_config"]
                + runs["v73_wave2"]["pattern_counts"]["missing_lora_config"]
            ),
        },
        "v74": {
            "actor_count": 4,
            "fp8_median_seconds": statistics.median(v74_times),
            "deepgemm_warning_actor_count": runs["v74"]["actor_pattern_counts"]["deepgemm_warning"],
            "deepgemm_e8m0_enabled_actor_count": runs["v74"]["actor_pattern_counts"]["deepgemm_e8m0_enabled"],
            "unsafe_environment": True,
        },
        "v75": {
            "actor_count": 4,
            "fp8_median_seconds": statistics.median(v75_times),
            "explicit_deepgemm_disable_actor_count": 4,
            "deepgemm_warning_actor_count": runs["v75"]["actor_pattern_counts"]["deepgemm_warning"],
            "deepgemm_e8m0_enabled_actor_count": runs["v75"]["actor_pattern_counts"]["deepgemm_e8m0_enabled"],
            "flashinfer_autotune_skipped_actor_count": runs["v75"]["actor_pattern_counts"]["flashinfer_autotune_skipped"],
            "missing_lora_config_messages": runs["v75"]["pattern_counts"]["missing_lora_config"],
            "default_lora_config_actor_count": runs["v75"]["actor_pattern_counts"]["default_lora_config"],
            "inference_jit_actor_counts": {
                key: runs["v75"]["actor_pattern_counts"][key]
                for key in (
                    "jit_lora_shrink", "jit_lora_expand",
                    "jit_fused_moe_lora", "jit_base_fused_moe",
                )
            },
            "fail_closed_environment_pass": False,
        },
        "v76": {
            "actor_count": 4,
            "fp8_median_seconds": statistics.median(v76_times),
            "immutable_receipt_and_log_bundle_is_authoritative": True,
            "deepgemm_disable": {
                "VLLM_USE_DEEP_GEMM": "0",
                "quant_config_use_deep_gemm_before_post_init": False,
                "upstream_source_modified": False,
                "deepgemm_warning_actor_count": 0,
                "deepgemm_e8m0_enabled_actor_count": 0,
            },
            "routed_backend": {
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
            },
            "stale_receipt_claim": {
                "field": "explicit_kernel_environment.explicit_cutlass_path_requested_without_runtime_fallback",
                "true_actor_count": 4,
                "trusted_for_backend_identity": False,
                "contradicted_by_live_runtime_audit": True,
            },
            "deepgemm_disable_and_routed_backend_are_independent": True,
            "environment_preflight_pass": True,
        },
    }


def build_preregistration_v77() -> dict[str, Any]:
    source = inspect_installed_source_v77()
    model = inspect_model_geometry_v77()
    reference = inspect_adapter_v77("reference")
    candidate = inspect_adapter_v77("candidate")
    _require(
        reference["rank"] == candidate["rank"] == 32
        and reference["shape_counts"] == candidate["shape_counts"],
        "paired adapter geometry differs",
    )
    value: dict[str, Any] = {
        "schema": contract.SCHEMA_V77,
        "bead": "specialist-0j5.22",
        "status": "cpu_preregistration_complete_launch_blocked",
        "authority": {
            "gpu_launch": False,
            "protected_or_ood_access": False,
            "dataset_access": False,
            "model_update_or_training": False,
            "checkpoint_or_config_promotion": False,
            "site_package_modification": False,
        },
        "blockers": [
            {
                "bead": "specialist-0j5.22",
                "kind": "cpu_only_authority_scope",
                "issue": (
                    "this preregistration records evidence and validates future "
                    "receipts but grants no GPU launch or promotion authority"
                ),
                "fail_closed": True,
                "resolved": False,
                "unblock_requires": [
                    "explicit launch authority outside this CPU-only artifact",
                    "fresh measurement receipts satisfying every sealed gate",
                    "separate promotion authority after semantic and OOD review",
                ],
            }
        ],
        "environment_resolution": {
            "bead": "specialist-nen.28",
            "prior_issue": (
                "V75 emitted the DeepGemm accuracy warning despite the environment "
                "variable because the installed warning branch does not consult it"
            ),
            "resolved_for_bound_v76_baseline": True,
            "resolution": (
                "V76 applies a process-local pre-post-init quant-config disable, "
                "without source edits, before engine construction"
            ),
            "deepgemm_disable_gate": {
                "VLLM_USE_DEEP_GEMM": "0",
                "quant_config_use_deep_gemm_before_post_init": False,
                "zero_warning_actors": 4,
                "zero_e8m0_enabled_actors": 4,
            },
            "independent_routed_backend_gate": {
                "moe_backend_argument": "triton",
                "live_backend_class": "Fp8MoeBackend",
                "live_backend_name": "TRITON",
                "experts_implementation_class": "TritonExperts",
                "runtime_quant_wrapper_class": "FusedMoEModularMethod",
                "routed_expert_owner_count_per_actor": 40,
                "lora_quant_reference_count_per_actor": 80,
                "attested_actor_count": 4,
            },
            "legacy_cutlass_request_field_is_not_backend_evidence": True,
            "warning_suppression_or_site_package_edit_is_not_evidence": True,
        },
        "installed_source": source,
        "model_and_adapter": {
            **model,
            "rank": 32,
            "max_loras": 1,
            "max_cpu_loras": 2,
            "top_k": model["num_experts_per_tok"],
            "max_num_seqs": 68,
            "max_num_batched_tokens": 16384,
            "reference_adapter": reference,
            "candidate_adapter": candidate,
        },
        "observed_evidence": inspect_observed_evidence_v77(),
        "rejected_prior_table": inspect_negative_evidence_v77(),
        "operation_inventory": contract.operation_inventory_v77(),
        "required_config_filenames": contract.required_filenames_v77(),
        "tuning_plan": contract.tuning_plan_v77(),
        "promotion_gates": contract.promotion_gates_v77(),
        "side_effects": {
            "torch_or_vllm_imported": False,
            "CUDA_initialized": False,
            "GPU_accessed": False,
            "site_packages_modified": False,
            "dataset_or_protected_content_opened": False,
            "config_table_written_or_promoted": False,
        },
    }
    value["content_sha256_before_self_field"] = contract.canonical_sha256_v77(value)
    return contract.validate_preregistration_v77(value)


def write_preregistration_v77(path: Path = OUTPUT) -> dict[str, Any]:
    value = build_preregistration_v77()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(payload, encoding="ascii")
    temporary.replace(path)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = build_preregistration_v77()
    if args.check:
        _require(args.output.is_file(), f"missing preregistration: {args.output}")
        frozen = _json(args.output)
        contract.validate_preregistration_v77(frozen)
        _require(frozen == value, "frozen preregistration is stale")
    else:
        write_preregistration_v77(args.output)
    print(
        json.dumps(
            {
                "schema": contract.SCHEMA_V77,
                "output": str(args.output),
                "status": value["status"],
                "content_sha256": value["content_sha256_before_self_field"],
                "gpu_launch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
