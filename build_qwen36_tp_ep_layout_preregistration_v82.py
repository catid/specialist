#!/usr/bin/env python3
"""Build the CPU-only Qwen3.6 TP/EP layout comparison contract (V82).

The builder deliberately imports neither torch nor vLLM.  It verifies the
installed vLLM sources, the frozen model config, and sealed V66D/V79 ancestry
as ordinary files.  GPU execution remains a later, explicitly gated step.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_tp_ep_layout_comparison_v82.json"
)
REPORT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_tp_ep_layout_v82_cpu_evidence_20260717.md"
)

V66D = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_mirrored_calibration_v66d.json"
)
V79 = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fp8_kv_capacity_matched_v79.json"
)
MODEL_CONFIG = ROOT / "models/Qwen3.6-35B-A3B-FP8/config.json"
SITE_PACKAGES = ROOT / "es-at-scale/.venv/lib/python3.12/site-packages"

SCHEMA = "v82-qwen36-tp-ep-layout-comparison-preregistration"
V66D_FILE_SHA256 = (
    "3269f7138d74266538cc3b0f31e1a904808f8f3751dde5a7a9456e93b13314b0"
)
V66D_CONTENT_SHA256 = (
    "2f8e23b643507b594b05719966da1b9bcc64a2b7f412021066df4c6418144531"
)
V79_FILE_SHA256 = (
    "0e195c05fd72e36656ee6536d6656d932aac0028fbbc7983f688df9dc7b18753"
)
V79_CONTENT_SHA256 = (
    "6c73ac0f6bf4019cdf297546e4315dc99b68d9549a24c03f4eaa9c8ebb589023"
)
MODEL_CONFIG_SHA256 = (
    "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
)

VLLM_FILES = {
    "vllm-0.25.0.dist-info/METADATA": (
        "8a317380a72e2a4e58f80188aacf0dca83c5ac2e1995063964b7eac27b7bab24"
    ),
    "vllm/config/parallel.py": (
        "2bb0c2d54a0c3211c5ea1ddeed731d0e2a5d95fb58972f92845268959f0c818e"
    ),
    "vllm/model_executor/models/registry.py": (
        "97d48b835001ff7d8e25a5ffb6641f68c021a010bbcfca1cead4ddab53ace7a6"
    ),
    "vllm/model_executor/models/qwen3_5.py": (
        "5f47ae4f4a08d0a78dd681d58b290f3298744c73a82f1349f3e2853469ef73e6"
    ),
    "vllm/model_executor/models/qwen3_next.py": (
        "b0e0b96d08329e7ad91756eb67e8c4a9ad19ac0a38cbebbc3fbf13716ec424cd"
    ),
    "vllm/model_executor/layers/fused_moe/config.py": (
        "4090ec9e367fee4344c9d06f12e083ff37df427784e6736847e092222c9e1415"
    ),
    "vllm/distributed/device_communicators/base_device_communicator.py": (
        "19945f4d567280654e54099c23e61918a046c37515c4edc95ada720804aa5202"
    ),
    "vllm/model_executor/model_loader/default_loader.py": (
        "5d120c07b8eb4d08ce1d4e9759b832a07086dcc78d0df4cefe9beb5c29b7de4e"
    ),
}

REQUIRED_SOURCE_FRAGMENTS = {
    "vllm-0.25.0.dist-info/METADATA": (
        "Name: vllm",
        "Version: 0.25.0",
    ),
    "vllm/config/parallel.py": (
        "class ParallelConfig:",
        "enable_expert_parallel: bool = False",
        'all2all_backend: All2AllBackend = "allgather_reducescatter"',
        'if self.all2all_backend in ["pplx", "naive"]:',
        "effect on 3D fused-expert checkpoints",
    ),
    "vllm/model_executor/models/registry.py": (
        '"Qwen3_5MoeForConditionalGeneration"',
        '"qwen3_5"',
    ),
    "vllm/model_executor/models/qwen3_5.py": (
        "class Qwen3_5MoeForConditionalGeneration(",
        "is_3d_moe_weight: bool = True",
        "tp_size = parallel_config.tensor_parallel_size",
    ),
    "vllm/model_executor/models/qwen3_next.py": (
        "class Qwen3NextSparseMoeBlock(nn.Module):",
        "self.ep_group = get_ep_group().device_group",
        "assert tp_size % self.total_num_kv_heads == 0",
        "SupportsLoRA",
    ),
    "vllm/model_executor/layers/fused_moe/config.py": (
        "class FusedMoEParallelConfig:",
        "vllm_parallel_config.enable_expert_parallel",
        "ep_size = tp_size",
        "return self.use_ep and (self.dp_size > 1 or self.is_sequence_parallel)",
        "allgather_reducescatter",
    ),
    "vllm/distributed/device_communicators/base_device_communicator.py": (
        "initialize the all2all manager for DP or sequence-parallel EP",
        "config.parallel_config.data_parallel_size > 1",
        "self.use_all2all = self.is_ep_communicator and use_ep",
    ),
    "vllm/model_executor/model_loader/default_loader.py": (
        "def _init_ep_weight_filter",
        "parallel_config.enable_ep_weight_filter",
        "compute_local_expert_ids",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def file_sha256_v82(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256_v82(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _sealed_json(
    path: Path,
    *,
    file_sha256: str,
    content_sha256: str,
    schema: str,
) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"missing sealed input: {path}")
    _require(file_sha256_v82(path) == file_sha256, f"sealed file changed: {path}")
    value = json.loads(path.read_text(encoding="ascii"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(claimed == content_sha256, f"sealed content identity changed: {path}")
    _require(canonical_sha256_v82(body) == claimed, f"invalid self hash: {path}")
    _require(value.get("schema") == schema, f"sealed schema changed: {path}")
    return value


def _load_ancestry_v82() -> tuple[dict[str, Any], dict[str, Any]]:
    v66d = _sealed_json(
        V66D,
        file_sha256=V66D_FILE_SHA256,
        content_sha256=V66D_CONTENT_SHA256,
        schema="lora-es-mirrored-crn-qwen36-calibration-preregistration-v66d",
    )
    v79 = _sealed_json(
        V79,
        file_sha256=V79_FILE_SHA256,
        content_sha256=V79_CONTENT_SHA256,
        schema="v79-qwen36-fp8-kv-capacity-matched-preregistration",
    )

    recipe = v66d.get("fixed_recipe", {})
    decode = recipe.get("decode", {})
    adapter = v66d.get("adapter", {})
    _require(
        recipe.get("engines") == 4
        and recipe.get("tensor_parallel_size") == 1
        and recipe.get("signed_population_size") == 16
        and recipe.get("direction_count") == 8
        and len(recipe.get("direction_seeds", [])) == 8
        and recipe.get("train_rows_per_candidate") == 64
        and recipe.get("sigma") == 0.0006,
        "V66D population contract changed",
    )
    _require(
        decode
        == {
            "detokenize": False,
            "max_tokens": 1,
            "n": 1,
            "prompt_logprobs": 1,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "V66D decode contract changed",
    )
    _require(
        adapter.get("canonical_fp32_master_sha256")
        == "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
        and adapter.get("runtime_bf16_values_sha256")
        == "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
        and v66d.get("train_only_input", {}).get(
            "protected_dev_ood_or_holdout_paths"
        )
        == [],
        "V66D adapter/data authority changed",
    )

    runtime = v79.get("selected_runtime", {})
    v76 = v79.get("sealed_evidence", {}).get("v76_bf16_kv", {})
    residence = v79.get("sealed_evidence", {}).get(
        "current_parameter_residency", {}
    )
    hardware = v79.get("sealed_evidence", {}).get("hardware", {})
    _require(
        v79.get("status") == "cpu_preregistered_live_launch_not_performed"
        and v79.get("authority", {}).get("scored_or_training_authority") is False,
        "V79 authority changed",
    )
    _require(
        runtime.get("quantization") == "fp8"
        and runtime.get("kv_cache_dtype") == "fp8_per_token_head"
        and runtime.get("mamba_ssm_cache_dtype") == "float32"
        and runtime.get("routed_moe_backend") == "TRITON"
        and runtime.get("gpu_memory_utilization") == 0.485
        and runtime.get("max_num_seqs") == 68
        and runtime.get("max_model_len") == 2048
        and runtime.get("max_loras") == 1,
        "V79 selected runtime changed",
    )
    _require(
        v76.get("actor_count") == 4
        and v76.get("telemetry", {}).get("all_four_gpus_useful") is True
        and v76.get("capacity_tokens_per_actor") == [157696] * 4
        and residence.get("total_logical_bytes") == 35_712_084_096
        and residence.get("components", {}).keys() == {"language"}
        and hardware.get("physical_gpu_ids") == [0, 1, 2, 3]
        and hardware.get("memory_total_mib_per_gpu") == 97_887,
        "V76/V79 hardware or residency evidence changed",
    )
    return v66d, v79


def _source_inventory_v82() -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    texts: dict[str, str] = {}
    _require(SITE_PACKAGES.is_dir(), "installed vLLM site-packages missing")
    for relative, expected in VLLM_FILES.items():
        path = SITE_PACKAGES / relative
        _require(path.is_file() and not path.is_symlink(), f"missing vLLM file: {path}")
        observed = file_sha256_v82(path)
        _require(observed == expected, f"installed vLLM source changed: {relative}")
        text = path.read_text(encoding="utf-8", errors="strict")
        for fragment in REQUIRED_SOURCE_FRAGMENTS[relative]:
            _require(
                fragment in text,
                f"vLLM support fragment missing: {relative}: {fragment}",
            )
        texts[relative] = text
        rows.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": observed,
            }
        )
    return rows, texts


def _module_present_v82(name: str) -> bool:
    return (SITE_PACKAGES / name).is_dir() or (SITE_PACKAGES / f"{name}.py").is_file()


def _optional_backend_inventory_v82() -> dict[str, Any]:
    return {
        "allgather_reducescatter": {
            "cpu_support_evidence": "built_in_vllm_default",
            "admitted_for_live_attestation": True,
            "runtime_resolution_still_required": True,
        },
        "deepep": {
            "module_present": _module_present_v82("deep_ep")
            or _module_present_v82("deepep"),
            "admitted_for_live_attestation": False,
            "reason": "DeepEP module is absent; no fallback is permitted",
        },
        "nixl_ep": {
            "module_present": _module_present_v82("nixl"),
            "admitted_for_live_attestation": False,
            "reason": "NIXL module is absent; no fallback is permitted",
        },
        "pplx_or_naive": {
            "admitted_for_live_attestation": False,
            "reason": (
                "vLLM 0.25 removes these choices and rewrites them to "
                "allgather_reducescatter; silent rewrite is forbidden"
            ),
        },
        "flashinfer_nvlink": {
            "module_present": _module_present_v82("flashinfer"),
            "admitted_for_live_attestation": False,
            "reason": (
                "package presence does not prove topology, FP8-Qwen, LoRA, or "
                "runtime-kernel support"
            ),
        },
    }


def _model_support_v82() -> dict[str, Any]:
    _require(MODEL_CONFIG.is_file(), "Qwen3.6 FP8 config missing")
    _require(
        file_sha256_v82(MODEL_CONFIG) == MODEL_CONFIG_SHA256,
        "Qwen3.6 FP8 config changed",
    )
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    required = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "mamba_ssm_dtype": "float32",
    }
    _require(
        config.get("architectures") == ["Qwen3_5MoeForConditionalGeneration"]
        and all(text.get(key) == value for key, value in required.items()),
        "Qwen3.6 architecture/config support surface changed",
    )
    tp = 4
    _require(text["num_attention_heads"] % tp == 0, "TP4 attention heads unsupported")
    _require(text["hidden_size"] % tp == 0, "TP4 hidden size unsupported")
    _require(text["linear_num_key_heads"] % tp == 0, "TP4 linear K heads unsupported")
    _require(text["linear_num_value_heads"] % tp == 0, "TP4 linear V heads unsupported")
    _require(text["num_experts"] % tp == 0, "EP4 expert ownership unsupported")
    _require(
        tp % text["num_key_value_heads"] == 0,
        "TP4 grouped-query KV replication rule unsupported",
    )
    kv_heads_per_tp1_rank = text["num_key_value_heads"]
    kv_heads_per_tp4_rank = max(1, text["num_key_value_heads"] // tp)
    return {
        "path": str(MODEL_CONFIG.relative_to(ROOT)),
        "sha256": MODEL_CONFIG_SHA256,
        "architecture": config["architectures"][0],
        "model_type": text["model_type"],
        "hidden_size": text["hidden_size"],
        "layers": text["num_hidden_layers"],
        "attention_heads": text["num_attention_heads"],
        "kv_heads": text["num_key_value_heads"],
        "linear_key_heads": text["linear_num_key_heads"],
        "linear_value_heads": text["linear_num_value_heads"],
        "experts": text["num_experts"],
        "experts_per_token": text["num_experts_per_tok"],
        "mamba_ssm_dtype": text["mamba_ssm_dtype"],
        "tp4_static_divisibility_passed": True,
        "ep4_local_experts_per_rank": text["num_experts"] // tp,
        "attention_kv_heads_per_tp1_rank": kv_heads_per_tp1_rank,
        "attention_kv_heads_per_tp4_rank": kv_heads_per_tp4_rank,
        "attention_kv_elements_per_token_ratio_tp4_rank_to_tp1_rank": (
            kv_heads_per_tp4_rank / kv_heads_per_tp1_rank
        ),
        "important_kv_caveat": (
            "TP4 halves, rather than quarters, full-attention KV elements per "
            "rank because two KV heads are replicated across four TP ranks"
        ),
        "mamba_state_shape_requires_live_rank_attestation": True,
        "checkpoint_declares_3d_moe_weights": True,
        "ep_weight_filter_memory_or_io_savings_assumed": False,
    }


def _common_runtime_v82(v79: dict[str, Any]) -> dict[str, Any]:
    selected = copy.deepcopy(v79["selected_runtime"])
    return {
        key: selected[key]
        for key in (
            "async_scheduling",
            "calculate_kv_scales",
            "enable_flashinfer_autotune",
            "enforce_eager",
            "gpu_memory_utilization",
            "kv_cache_dtype",
            "kv_cache_dtype_skip_layers",
            "mamba_cache_dtype",
            "mamba_ssm_cache_dtype",
            "max_cpu_loras",
            "max_loras",
            "max_model_len",
            "max_num_seqs",
            "prefix_caching",
            "quantization",
            "routed_moe_backend",
            "scheduling_policy",
        )
    }


def _admitted_arms_v82(common_runtime: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "arm_id": "replicated_4xtp1_control",
            "role": "control",
            "admitted": True,
            "engine_count": 4,
            "worker_rank_count": 4,
            "candidate_concurrency": 4,
            "signed_candidate_waves": 4,
            "parallel": {
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "enable_expert_parallel": False,
                "all2all_backend": "allgather_reducescatter",
                "all2all_active": False,
            },
            "runtime": copy.deepcopy(common_runtime),
            "expected_moe_rank_contract": {
                "use_ep": False,
                "tp_size": 1,
                "ep_size": 1,
                "logical_experts_visible": 256,
            },
            "candidate_isolation": (
                "one independently materialized sole LoRA slot per TP1 actor"
            ),
        },
        {
            "arm_id": "single_tp4_tensor_sharded",
            "role": "challenger",
            "admitted": True,
            "engine_count": 1,
            "worker_rank_count": 4,
            "candidate_concurrency": 1,
            "signed_candidate_waves": 16,
            "parallel": {
                "tensor_parallel_size": 4,
                "data_parallel_size": 1,
                "enable_expert_parallel": False,
                "all2all_backend": "allgather_reducescatter",
                "all2all_active": False,
            },
            "runtime": copy.deepcopy(common_runtime),
            "expected_moe_rank_contract": {
                "use_ep": False,
                "tp_size": 4,
                "ep_size": 1,
                "logical_experts_visible": 256,
                "expert_tensors_must_be_tp_sharded": True,
            },
            "candidate_isolation": (
                "one canonical candidate is sharded identically over all four "
                "ranks and evaluated serially"
            ),
        },
        {
            "arm_id": "single_tp4_ep4_tp_collective",
            "role": "challenger",
            "admitted": True,
            "engine_count": 1,
            "worker_rank_count": 4,
            "candidate_concurrency": 1,
            "signed_candidate_waves": 16,
            "parallel": {
                "tensor_parallel_size": 4,
                "data_parallel_size": 1,
                "enable_expert_parallel": True,
                "all2all_backend": "allgather_reducescatter",
                "all2all_active": False,
                "enable_ep_weight_filter": False,
                "enable_eplb": False,
                "expert_placement_strategy": "linear",
            },
            "runtime": copy.deepcopy(common_runtime),
            "expected_moe_rank_contract": {
                "use_ep": True,
                "tp_size_inside_fused_moe": 1,
                "ep_size": 4,
                "local_experts_per_rank": 64,
                "redundant_experts": 0,
                "expert_ranges": [[0, 64], [64, 128], [128, 192], [192, 256]],
                "use_all2all_kernels": False,
                "reason_all2all_is_inactive": (
                    "vLLM activates fused-MoE all2all only for DP greater than "
                    "one or sequence-parallel EP; this arm is TP-only EP"
                ),
            },
            "candidate_isolation": (
                "one canonical candidate is installed on every rank before "
                "rank-local expert outputs are combined"
            ),
        },
        {
            "arm_id": "wide_dp4_tp1_ep4_agrs",
            "role": "challenger",
            "admitted": True,
            "engine_count": 1,
            "worker_rank_count": 4,
            "candidate_concurrency": 1,
            "signed_candidate_waves": 16,
            "parallel": {
                "tensor_parallel_size": 1,
                "data_parallel_size": 4,
                "data_parallel_size_local": 4,
                "enable_expert_parallel": True,
                "all2all_backend": "allgather_reducescatter",
                "all2all_active": True,
                "enable_ep_weight_filter": False,
                "enable_eplb": False,
                "expert_placement_strategy": "linear",
            },
            "runtime": copy.deepcopy(common_runtime),
            "expected_moe_rank_contract": {
                "use_ep": True,
                "tp_size_inside_fused_moe": 1,
                "dp_size": 4,
                "ep_size": 4,
                "local_experts_per_rank": 64,
                "redundant_experts": 0,
                "expert_ranges": [[0, 64], [64, 128], [128, 192], [192, 256]],
                "use_all2all_kernels": True,
                "use_ag_rs_all2all_kernels": True,
                "dense_and_kv_state_remain_tp1_replicated": True,
            },
            "candidate_isolation": (
                "one globally identical canonical candidate is installed on "
                "all DP/EP ranks; the 64-prompt panel is deterministically "
                "partitioned across ranks and recombined before scoring"
            ),
        },
    ]


def _rejected_arms_v82(optional: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "arm_id": "dp4_tp1_wide_ep_four_distinct_one_slot",
            "admitted": False,
            "rejection_stage": "cpu_recipe_compatibility",
            "reason": (
                "the sealed recipe concurrently materializes four distinct "
                "candidates in one LoRA slot per actor; wide EP routes tokens "
                "to ranks holding different candidate values, so candidate "
                "identity is not preserved"
            ),
            "future_reconsideration": (
                "requires a separately preregistered globally identical "
                "multi-LoRA slot implementation with per-token adapter identity"
            ),
        },
        {
            "arm_id": "wide_dp4_tp1_ep4_deepep",
            "admitted": False,
            "rejection_stage": "installed_dependency",
            "reason": optional["deepep"]["reason"],
        },
        {
            "arm_id": "wide_dp4_tp1_ep4_nixl",
            "admitted": False,
            "rejection_stage": "installed_dependency",
            "reason": optional["nixl_ep"]["reason"],
        },
        {
            "arm_id": "wide_dp4_tp1_ep4_pplx_or_naive",
            "admitted": False,
            "rejection_stage": "removed_backend",
            "reason": optional["pplx_or_naive"]["reason"],
        },
        {
            "arm_id": "wide_dp4_tp1_ep4_flashinfer_nvlink",
            "admitted": False,
            "rejection_stage": "runtime_topology_and_kernel_support_unproven",
            "reason": optional["flashinfer_nvlink"]["reason"],
        },
        {
            "arm_id": "ep_weight_filter_as_memory_arm",
            "admitted": False,
            "rejection_stage": "checkpoint_format",
            "reason": (
                "Qwen3.5-MoE declares 3D fused MoE weights and vLLM documents "
                "the pre-read EP filter as ineffective for 3D fused-expert "
                "checkpoints; no VRAM or I/O saving may be assumed"
            ),
        },
    ]


def _workload_contract_v82(v66d: dict[str, Any]) -> dict[str, Any]:
    recipe = v66d["fixed_recipe"]
    adapter = v66d["adapter"]
    panel = v66d["train_only_input"]["panel"]
    return {
        "candidate_state": {
            "canonical_fp32_master_sha256": adapter[
                "canonical_fp32_master_sha256"
            ],
            "canonical_runtime_bf16_values_sha256": adapter[
                "runtime_bf16_values_sha256"
            ],
            "direction_seeds_exact": recipe["direction_seeds"],
            "direction_count": 8,
            "signs_per_direction": [-1, 1],
            "signed_population_size": 16,
            "sigma": recipe["sigma"],
            "candidate_order_and_physical_rank_rotation_must_equal_v66d": True,
            "candidate_hash_must_be_computed_before_rank_sharding": True,
        },
        "prompt_panel": {
            "path": panel["path"],
            "file_sha256": panel["file_sha256"],
            "content_sha256": panel["content_sha256"],
            "rows_per_candidate": panel["selected_rows"],
            "conflict_units_per_candidate": panel["selected_conflict_units"],
            "tokenized_prompt_ids_sha256_must_be_sealed_once_and_shared": True,
            "raw_text_must_not_be_persisted_in_runtime_receipts": True,
        },
        "decode": copy.deepcopy(recipe["decode"]),
        "reward": recipe["reward"],
        "request_outputs_total": 16 * panel["selected_rows"],
        "generated_tokens_upper_bound": 16
        * panel["selected_rows"]
        * recipe["decode"]["max_tokens"],
        "teacher_forced_answer_token_count": (
            "seal once from the shared tokenized panel and require exact equality "
            "for every arm"
        ),
        "candidate_commit_or_promotion": False,
        "protected_dev_ood_or_holdout_opened_during_live_system_ablation": False,
    }


def _runtime_attestation_v82() -> dict[str, Any]:
    return {
        "fail_closed_before_first_generation": {
            "physical_gpu_ids_exact": [0, 1, 2, 3],
            "unique_worker_pid_and_cuda_device_per_rank": True,
            "parallel_config_fields_equal_selected_arm": True,
            "world_tp_dp_ep_rank_maps_complete_and_unique": True,
            "resolved_all2all_backend_exact_no_rewrite": True,
            "vllm_version_exact": "0.25.0",
            "model_config_sha256_exact": MODEL_CONFIG_SHA256,
            "quantization_exact": "fp8",
            "attention_backend_exact": "TRITON_ATTN",
            "routed_moe_backend_exact": "TRITON",
            "forbid_any_fallback_warning_or_backend_substitution": True,
        },
        "model_and_rank_ownership": {
            "named_parameter_name_dtype_shape_numel_bytes_per_rank": True,
            "global_parameter_names_and_logical_bytes_reconstructed_once": True,
            "visual_or_mtp_parameter_count_exact": 0,
            "routed_layer_count_exact": 40,
            "routed_layer_owner_records_per_rank_exact": 40,
            "fused_moe_parallel_config_per_layer_and_rank": True,
            "expert_map_and_local_expert_range_per_layer_and_rank": True,
            "w13_w2_dtype_shape_numel_storage_pointer_alias_audit": True,
            "attention_kv_heads_and_cache_block_shapes_per_rank": True,
            "mamba_conv_and_ssm_state_dtype_shape_bytes_per_rank": True,
            "kv_cache_capacity_tokens_and_full_2048_contexts_per_rank": True,
            "available_kv_gib_per_rank": True,
            "ep_weight_filter_savings_must_not_be_claimed": True,
        },
        "lora_identity_and_restore": {
            "adapter_slot_ids_and_target_module_names_per_rank": True,
            "canonical_candidate_sha256_equal_across_all_rank_shards": True,
            "rank_shard_offset_shape_dtype_numel_sha256_inventory": True,
            "candidate_runtime_must_differ_from_master": True,
            "restore_after_every_signed_candidate": True,
            "canonical_fp32_master_restore_sha256_exact": True,
            "gathered_runtime_bf16_values_restore_sha256_exact": True,
            "all_rank_local_restore_certificates_required_before_next_candidate": True,
            "poison_and_abort_on_partial_collective_or_hash_mismatch": True,
        },
        "useful_gpu_work": {
            "all_four_physical_gpus_resident_and_foreign_pid_free": True,
            "actor_cuda_event_elapsed_positive_per_candidate_per_rank": True,
            "output_and_answer_token_cardinality_positive_per_candidate": True,
            "phase_barrier_after_four_rank_receipts": True,
            "nvml_positive_gpu_or_memory_utilization_per_rank_per_wave": True,
            "resident_but_idle_rank_fails": True,
        },
        "collective_trace": {
            "nsight_systems_available_path": "/usr/local/cuda/bin/nsys",
            "profile_trace_exact": "cuda,nvtx,nccl,osrt",
            "nccl_trace_exact": "api,ce-coll,group,gpu",
            "rank_collective_operation_count_size_bytes_and_duration_required": True,
            "tp4_non_ep_requires_observed_tp_collectives": True,
            "tp4_ep_requires_observed_tp_expert_output_collectives": True,
            "tp4_ep_requires_zero_agrs_dispatch_and_combine_collectives": True,
            "wide_dp4_ep_requires_observed_agrs_dispatch_and_combine_collectives": True,
            "control_requires_zero_model_parallel_nccl_collective_bytes": True,
            "profiler_replicate_is_separate_from_timing_replicates": True,
        },
    }


def _measurement_contract_v82() -> dict[str, Any]:
    return {
        "replication": {
            "fresh_process_replicates_per_admitted_arm": 3,
            "arm_order": [
                [
                    "replicated_4xtp1_control",
                    "single_tp4_tensor_sharded",
                    "single_tp4_ep4_tp_collective",
                    "wide_dp4_tp1_ep4_agrs",
                ],
                [
                    "wide_dp4_tp1_ep4_agrs",
                    "replicated_4xtp1_control",
                    "single_tp4_tensor_sharded",
                    "single_tp4_ep4_tp_collective",
                ],
                [
                    "single_tp4_ep4_tp_collective",
                    "single_tp4_tensor_sharded",
                    "wide_dp4_tp1_ep4_agrs",
                    "replicated_4xtp1_control",
                ],
            ],
            "same_candidate_order_inside_every_arm": True,
            "warmup_candidate_not_in_timing_or_quality_statistics": True,
            "timing_excludes_model_load_but_reports_load_separately": True,
        },
        "vram_and_cache": {
            "per_rank_peak_and_phase_peak_memory_used_mib": True,
            "aggregate_peak_memory_used_mib": True,
            "named_parameter_logical_and_storage_bytes_per_rank": True,
            "adapter_and_runtime_scratch_bytes_per_rank": True,
            "attention_kv_and_mamba_state_bytes_per_rank": True,
            "kv_capacity_tokens_contexts_and_available_gib_per_rank": True,
            "physical_headroom_mib_per_rank": True,
        },
        "bandwidth_and_communication": {
            "nvml_gpu_memory_utilization_power_and_pcie_rx_tx_at_4hz": True,
            "pcie_rx_tx_integrals_are_left_rectangle_estimates": True,
            "memory_utilization_is_activity_proxy_not_hbm_bytes_per_second": True,
            "nsys_gpu_metrics_dram_activity_proxy_required": True,
            "nsys_nccl_message_bytes_and_duration_per_rank_required": True,
            "ncu_finalists_only_fixed_candidate": "direction_0_sign_positive",
            "ncu_metrics": [
                "dram__bytes_read.sum",
                "dram__bytes_write.sum",
                "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "gpu__time_duration.sum",
            ],
            "ncu_profile_not_used_for_timing": True,
        },
        "throughput": {
            "end_to_end_population_wall_seconds": True,
            "aggregate_and_per_rank_generated_tokens_per_second": True,
            "teacher_forced_answer_tokens_per_second": True,
            "signed_candidates_or_rollouts_per_second": True,
            "active_gpu_seconds_sum": True,
            "rollouts_per_gpu_second": True,
            "per_candidate_p50_p95_max_latency_seconds": True,
            "materialize_restore_and_collective_time_seconds": True,
        },
        "correctness": {
            "request_output_and_answer_token_counts_exact": True,
            "all_candidate_and_restore_hashes_exact": True,
            "all_rewards_finite": True,
            "max_absolute_candidate_mean_logprob_delta_vs_control": 0.001,
            "pair_difference_vector_pearson_min": 0.99,
            "candidate_update_vector_cosine_min": 0.999,
            "no_new_safety_failure": True,
        },
    }


def _selection_and_quality_v82(v79: dict[str, Any]) -> dict[str, Any]:
    semantic = copy.deepcopy(v79["live_acceptance"]["semantic"])
    ood = copy.deepcopy(v79["live_acceptance"]["protected_ood"])
    return {
        "early_reject": {
            "unsupported_or_resolved_backend_mismatch": "reject_before_generation",
            "oom_or_fallback": "reject_arm_without_retrying_a_different_backend",
            "idle_or_foreign_gpu_rank": "reject_replicate",
            "candidate_or_restore_identity_mismatch": "poison_run_and_reject_arm",
            "correctness_threshold_failure": "reject_arm",
        },
        "system_selection": {
            "compare_point_estimates_and_paired_95pct_bootstrap_intervals": True,
            "rollouts_per_gpu_second_point_ratio_vs_control_min": 1.0,
            "rollouts_per_gpu_second_paired_95pct_lower_ratio_min": 0.98,
            "per_gpu_peak_vram_reduction_fraction_min_for_memory_claim": 0.10,
            "collective_bytes_and_hbm_metrics_reported_even_if_arm_rejected": True,
            "any_arm_with_point_throughput_below_control_is_slower_and_rejected": True,
            "winner_must_pass_every_identity_and_useful_gpu_gate": True,
        },
        "source_disjoint_semantic": semantic,
        "protected_ood_one_shot_after_layout_freeze": ood,
        "promotion": {
            "default": False,
            "requires_system_semantic_and_ood_gates": True,
            "no_retuning_after_protected_ood_open": True,
            "production_layout_decision_remains_provisional": True,
        },
    }


def build_preregistration_v82() -> dict[str, Any]:
    v66d, v79 = _load_ancestry_v82()
    source_rows, _ = _source_inventory_v82()
    optional = _optional_backend_inventory_v82()
    model = _model_support_v82()
    common_runtime = _common_runtime_v82(v79)
    admitted = _admitted_arms_v82(common_runtime)
    rejected = _rejected_arms_v82(optional)
    _require(
        [arm["candidate_concurrency"] for arm in admitted] == [4, 1, 1, 1],
        "arm concurrency changed",
    )
    _require(
        sum(arm["admitted"] for arm in admitted) == 4,
        "admitted arm count changed",
    )
    _require(all(not arm["admitted"] for arm in rejected), "rejected arm admitted")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "bead": "specialist-0j5.25",
        "status": "cpu_preregistered_live_launch_not_authorized",
        "purpose": (
            "Compare four concurrent TP1 LoRA-ES actors with TP4 tensor "
            "sharding, TP-only EP4, and DP4/TP1 wide EP4 execution, preserving "
            "the exact candidate population and measuring whether VRAM savings "
            "offset collective traffic and lost candidate parallelism."
        ),
        "authority": {
            "cpu_file_inspection_only": True,
            "gpu_api_or_model_launch_performed_by_builder": False,
            "dataset_or_prompt_rows_opened_by_builder": False,
            "model_or_adapter_update_performed": False,
            "checkpoint_or_layout_promotion_performed": False,
            "live_launch_authorized": False,
            "live_launch_waits_for_beads": ["specialist-0j5.14", "specialist-0j5.15"],
            "site_packages_or_es_at_scale_modified": False,
        },
        "sealed_ancestry": {
            "v66d_candidate_workload": {
                "path": str(V66D.relative_to(ROOT)),
                "file_sha256": V66D_FILE_SHA256,
                "content_sha256": V66D_CONTENT_SHA256,
            },
            "v79_fp8_runtime_and_v76_live_evidence": {
                "path": str(V79.relative_to(ROOT)),
                "file_sha256": V79_FILE_SHA256,
                "content_sha256": V79_CONTENT_SHA256,
                "v79_is_preregistered_not_promoted": True,
                "v76_four_tp1_median_seconds": v79["sealed_evidence"][
                    "v76_bf16_kv"
                ]["median_runtime_seconds"],
                "v76_peak_memory_used_mib_per_gpu": v79["sealed_evidence"][
                    "v76_bf16_kv"
                ]["telemetry"]["peak_memory_used_mib"],
                "v76_logical_parameter_bytes_per_actor": v79["sealed_evidence"][
                    "current_parameter_residency"
                ]["total_logical_bytes"],
            },
        },
        "installed_support_evidence": {
            "vllm_version": "0.25.0",
            "source_files": source_rows,
            "source_bundle_sha256": canonical_sha256_v82(source_rows),
            "model": model,
            "optional_backends": optional,
            "source_inspection_is_not_live_runtime_attestation": True,
        },
        "common_workload": _workload_contract_v82(v66d),
        "admitted_arms": admitted,
        "explicitly_rejected_arms": rejected,
        "runtime_attestation": _runtime_attestation_v82(),
        "measurement": _measurement_contract_v82(),
        "selection_and_quality": _selection_and_quality_v82(v79),
        "implementation_order": [
            "close accepted phase-profile and frozen-base precision dependencies",
            "implement rank-aware LoRA candidate materialize/restore certificates",
            "run CPU contract tests and dry-run argument validation",
            "run three clean system replicates for each admitted arm",
            "run separate Nsight Systems communication/profile replicate",
            "reject unsupported, incorrect, idle-rank, fallback, or slower arms",
            "run NCU memory-bandwidth diagnostic only for surviving finalists",
            "freeze the system winner and run source-disjoint semantic gate",
            "run protected OOD once with no subsequent tuning",
        ],
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v82(result)
    return result


def render_json_v82(value: dict[str, Any]) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"


def render_report_v82(value: dict[str, Any]) -> str:
    model = value["installed_support_evidence"]["model"]
    rejected = value["explicitly_rejected_arms"]
    lines = [
        "# Qwen3.6 TP/EP layout V82 CPU evidence",
        "",
        "V82 admits exactly four live arms after the prerequisite Beads close: "
        "four replicated TP1 actors, one tensor-sharded TP4 engine, TP-only "
        "EP4, and DP4/TP1 wide EP4 with vLLM's built-in "
        "allgather/reducescatter backend. The builder did not import vLLM or "
        "torch, launch a model, open dataset rows, or use a GPU.",
        "",
        "## Support findings",
        "",
        "The installed vLLM is source-bound to 0.25.0. Its Qwen3.5-MoE "
        "implementation exposes LoRA, TP-aware attention/Mamba state, and "
        "FusedMoE expert parallelism. The checkpoint has 256 experts, so EP4 "
        "has an exact static partition of 64 experts per rank.",
        "",
        f"A non-obvious limitation is KV scaling: Qwen has {model['kv_heads']} "
        "full-attention KV heads. TP4 therefore retains "
        f"{model['attention_kv_heads_per_tp4_rank']} KV head per rank, versus "
        f"{model['attention_kv_heads_per_tp1_rank']} at TP1. Full-attention KV "
        "elements per rank are expected to fall by only 2x, not 4x. The live "
        "receipt must separately attest hybrid attention and Mamba-state bytes.",
        "",
        "vLLM's all-to-all distinction is also important. TP-only EP4 does not "
        "activate its all-to-all kernels; it uses TP collectives around "
        "rank-local experts. DP4/TP1 wide EP does activate "
        "allgather/reducescatter. V82 therefore measures both instead of "
        "mislabeling TP-only EP traffic as all-to-all.",
        "",
        "The admitted wide-EP arm installs one globally identical candidate on "
        "all four ranks and partitions the 64 prompts. The tempting variant "
        "that keeps four different one-slot candidates concurrently is rejected: "
        "expert routing would mix candidate states unless a separately sealed "
        "global multi-LoRA implementation carries adapter identity end to end.",
        "",
        "## Rejected configurations",
        "",
    ]
    for item in rejected:
        lines.append(f"- `{item['arm_id']}`: {item['reason']}")
    lines.extend(
        [
            "",
            "## Live comparison contract",
            "",
            "All arms consume the same 16 signed V66D candidates, 64 prompts per "
            "candidate, decode parameters, tokenized prompt hash, reward, and "
            "canonical LoRA master. TP1 evaluates four candidates concurrently; "
            "each sharded arm evaluates them serially while all four ranks work. "
            "This makes the loss of candidate parallelism part of the measured "
            "end-to-end cost rather than hiding it in a per-call microbenchmark.",
            "",
            "Three clean timing replicates are required per arm. A separate "
            "Nsight Systems replicate records NCCL operation sizes/durations and "
            "GPU DRAM activity; NCU HBM byte metrics run only on finalists and "
            "never supply timing. Per-rank VRAM, parameter ownership, attention "
            "KV, Mamba state, PCIe, NCCL bytes, tokens/s, rollouts/s, and "
            "rollouts/GPU-second are all mandatory.",
            "",
            "A sharded arm is rejected on any backend rewrite/fallback, incorrect "
            "rank ownership, idle GPU, candidate/restore hash mismatch, numerical "
            "correctness failure, or point throughput below the replicated TP1 "
            "control. A surviving winner still needs the sealed source-disjoint "
            "semantic gate and one-shot protected OOD gate before promotion.",
            "",
            f"Contract content SHA-256: `{value['content_sha256_before_self_field']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_preregistration_v82()
    rendered_json = render_json_v82(value)
    rendered_report = render_report_v82(value)
    if args.check:
        _require(OUTPUT.is_file(), "V82 preregistration missing")
        _require(REPORT.is_file(), "V82 CPU evidence report missing")
        _require(
            OUTPUT.read_text(encoding="ascii") == rendered_json,
            "V82 preregistration stale",
        )
        _require(
            REPORT.read_text(encoding="utf-8") == rendered_report,
            "V82 report stale",
        )
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(rendered_json, encoding="ascii")
        REPORT.write_text(rendered_report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "report": str(REPORT),
                "content_sha256": value["content_sha256_before_self_field"],
                "admitted_arms": [arm["arm_id"] for arm in value["admitted_arms"]],
                "live_launch_authorized": value["authority"]["live_launch_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
