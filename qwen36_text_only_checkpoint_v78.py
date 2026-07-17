#!/usr/bin/env python3
"""Fail-closed contracts for Qwen3.6 text-only checkpoint omission.

This module has no torch/vLLM dependency and performs no CUDA work.  It
validates future standard-HuggingFace derivative artifacts and four-GPU live
residency/output receipts against a CPU-built checkpoint-byte inventory.
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


PREREG_SCHEMA_V78 = "qwen36-text-only-checkpoint-preregistration-v78"
ARTIFACT_SCHEMA_V78 = "qwen36-text-only-hf-safetensors-artifact-v78"
ACTOR_SCHEMA_V78 = "qwen36-text-only-live-residency-actor-v78"
PAIRED_SCHEMA_V78 = "qwen36-text-only-checkpoint-paired-runtime-v78"
PRECISIONS_V78 = ("bf16", "fp8_serialized")
ARMS_V78 = ("full_checkpoint", "text_only_derivative")
PHYSICAL_GPUS_V78 = [0, 1, 2, 3]

TOKENIZER_FILES_V78 = (
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def canonical_sha256_v78(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def file_sha256_v78(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _nonnegative(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0
    )


def engine_contract_v78() -> dict[str, Any]:
    """The artifact A/B keeps these settings identical on both arms.

    `language_model_only=True` is supported and useful, but installed Qwen3Next
    enables an additional fused text-only kernel under that flag.  It is kept
    out of this artifact-only A/B so any output/performance change is not
    confounded with a graph change.
    """

    return {
        "load_format": "safetensors",
        "safetensors_load_strategy": "lazy",
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "language_model_only": False,
        "enable_mm_embeds": False,
        "mm_processor_cache_gb": 0,
        "skip_mm_profiling": True,
        "speculative_config": None,
        "num_speculative_tokens": 0,
        "tensor_parallel_size": 1,
        "enable_lora": True,
        "max_lora_rank": 32,
        "max_loras": 1,
        "max_cpu_loras": 2,
        "max_num_seqs": 68,
        "enforce_eager": True,
    }


def supported_artifact_plan_v78() -> dict[str, Any]:
    return {
        "decision": "documented_only_do_not_build_under_current_evidence",
        "candidate_creation_authorized": False,
        "reason": (
            "four live FP8 actors already have language-only named-parameter "
            "residency, so a derivative cannot remove visual or MTP parameters "
            "from steady-state residency"
        ),
        "candidate_format": "standard_huggingface_safetensors",
        "candidate_architecture_config": "byte_identical_to_source",
        "candidate_tokenizer_and_generation_files": "byte_identical_to_source",
        "retained_tensor_rule": ["model.language_model.*", "lm_head.*"],
        "omitted_tensor_rule": ["model.visual.*", "mtp.*"],
        "other_tensor_prefix_allowed": False,
        "index_regenerated_for_retained_keys_only": True,
        "payload_copy_must_be_bit_exact": True,
        "source_checkpoint_modified": False,
        "source_and_candidate_roots_must_differ": True,
        "candidate_symlinks_or_external_hardlinks_allowed": False,
        "vllm_load_format": "safetensors",
        "custom_loader_or_plugin": False,
        "model_loader_extra_filter": None,
        "checkpoint_ignore_pattern_filter": None,
        "architecture_rewrite": False,
        "site_package_patch": False,
        "supported_runtime_omission": {
            "vision": (
                "zero image/video limits activate _mark_tower_model and "
                "StageMissingLayer"
            ),
            "mtp": (
                "target Qwen3_5 load_weights skips mtp.*; no speculative "
                "config means Qwen3_5MoeMTP is not instantiated"
            ),
        },
        "separate_future_ablation_not_bundled": {
            "language_model_only_true": True,
            "reason": (
                "installed Qwen3Next may enable a fused QK-norm/RoPE/gate "
                "kernel, so it is not an artifact-only change"
            ),
        },
        "reopen_only_if": [
            "a fresh live audit finds visual or MTP CUDA parameters",
            "startup checkpoint I/O is separately measured as a material bottleneck",
            "new artifact authority explicitly accepts derivative maintenance cost",
        ],
    }


def runtime_gates_v78() -> dict[str, Any]:
    return {
        "minimum_counterbalanced_four_gpu_replicates_per_precision": 3,
        "physical_gpus": list(PHYSICAL_GPUS_V78),
        "load_time": {
            "maximum_candidate_over_full_median_ratio": 0.98,
            "maximum_paired_bootstrap_95pct_upper_bound": 1.0,
        },
        "throughput": {
            "minimum_candidate_over_full_median_ratio": 0.99,
            "minimum_paired_bootstrap_95pct_lower_bound": 0.98,
        },
        "live_residency": {
            "named_parameter_manifest_exact_between_arms": True,
            "named_parameter_bytes_exact_between_arms": True,
            "visual_named_parameters": [],
            "mtp_named_parameters": [],
            "maximum_candidate_minus_full_peak_vram_mib": 0,
            "note": (
                "No VRAM saving is predicted because the supported full-artifact "
                "path already omits persistent visual/MTP parameters"
            ),
        },
        "identity": {
            "tokenizer_files_exact": True,
            "LoRA_target_manifest_exact": True,
            "language_payload_manifest_exact": True,
            "selected_logits_sha256_exact": True,
            "greedy_token_ids_sha256_exact": True,
            "candidate_switch_and_restore_exact": True,
        },
        "semantic_validation": {
            "selection_frozen_before_access": True,
            "source_disjoint": True,
            "minimum_paired_point_delta": -0.001,
            "minimum_paired_95pct_lower_bound": -0.002,
        },
        "protected_ood": {
            "one_shot_after_selection_freeze": True,
            "minimum_aggregate_95pct_lower_bound": -0.005,
            "minimum_worst_stratum_point_delta": -0.01,
            "maximum_new_safety_failures": 0,
        },
        "cleanup": {
            "all_four_processes_exited": True,
            "all_four_gpus_finally_idle": True,
            "foreign_compute_processes": [],
        },
    }


def validate_preregistration_v78(value: dict[str, Any]) -> dict[str, Any]:
    _require(isinstance(value, dict), "preregistration must be an object")
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(_sha256(claimed) and canonical_sha256_v78(body) == claimed, "preregistration self hash changed")
    _require(
        body.get("schema") == PREREG_SCHEMA_V78
        and body.get("bead") == "specialist-0j5.23"
        and body.get("status")
        == "inventory_and_live_fp8_audit_complete_no_artifact_recommended"
        and body.get("engine_contract") == engine_contract_v78()
        and body.get("supported_artifact_plan") == supported_artifact_plan_v78()
        and body.get("runtime_gates") == runtime_gates_v78(),
        "text-only preregistration contract changed",
    )
    _require(
        body.get("authority") == {
            "gpu_launch": False,
            "protected_or_ood_access": False,
            "candidate_artifact_creation": False,
            "source_checkpoint_modification": False,
            "site_package_or_es_at_scale_modification": False,
            "training_or_model_update": False,
            "promotion": False,
        },
        "text-only authority widened",
    )
    checkpoints = body.get("checkpoints")
    _require(isinstance(checkpoints, dict) and set(checkpoints) == set(PRECISIONS_V78), "checkpoint inventory arms changed")
    expected = {
        "bf16": {
            "tensor_count": 1045,
            "config_sha256": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
            "index_sha256": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
            "weight_file_count": 26,
            "physical_weight_bytes": 71_903_776_776,
            "logical_tensor_bytes": 71_903_645_408,
            "shard_size_manifest_sha256": "b878392051d6a40017ea4d096c786470aee2dab5e0440b33bb1b7b92142c60c9",
            "tensor_metadata_manifest_sha256": "d3dbdde871f50ceffa43ecc5e203b0d251bd09438837707d6f2f178b3acb4d37",
            "file_manifest_sha256": "9377411657cf12ebaf41c94b39d9217d5baa09bb77555ca8ca3a9893a81eb3ae",
            "language": (693, 69_321_221_376),
            "visual": (333, 893_142_496),
            "mtp": (19, 1_689_281_536),
            "omitted": (352, 2_582_424_032),
            "category_manifests": {
                "language": "066ced8ee35a27bbed875d4f73a1257802600349990a6c4ca1bb4e643cd8dcab",
                "visual": "22e59a49560759a7d89a53caefb2b3e872500f95404809becec9dbc1417d5ef7",
                "mtp": "5fef680476646e7e5741671f9374cea9c873a8a5e9d4d13ece980e91c6b19f1c",
            },
            "category_key_names": {
                "language": "3009dd3c57dc1caf938c765567fee074d64eb5fe677110f67ccec0af4316395a",
                "visual": "74db978fa718039692971f2ece846f2cc7a62a960047b5cebe2d758cc82ac2e9",
                "mtp": "76e2ce7a2d481fdadb6f5d969e890b632fbc35570a52c4677d9605dc4bc81097",
            },
            "omitted_manifest_sha256": "670c5a2d0ecdee733b17f44a23b690a34e8a4754a0785a04e5f3e845729cefe9",
            "omitted_key_names_sha256": "b7f3e717e4223a0b1079041f080fc9e47ee44dfe816548ca418638745471463a",
            "mixed_files": [
                "model-00001-of-00026.safetensors",
                "model-00002-of-00026.safetensors",
                "model-00025-of-00026.safetensors",
                "model-00026-of-00026.safetensors",
            ],
            "omitted_only_files": [],
        },
        "fp8_serialized": {
            "tensor_count": 64_196,
            "config_sha256": "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845",
            "index_sha256": "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6",
            "weight_file_count": 42,
            "physical_weight_bytes": 37_463_662_160,
            "logical_tensor_bytes": 37_454_789_472,
            "shard_size_manifest_sha256": "3bc687cefc89541a4768655cd110ffecf1d6a1ca94b5072903037dab119dcdaf",
            "tensor_metadata_manifest_sha256": "11d40e408ad3def3cb28fc698d47a32c2db76e484e3f1cf61ffbd4e89e97de50",
            "file_manifest_sha256": "46a29c0d5659539b0ac01941c7ce50421ad43cef7f42faf65996953c997456d6",
            "language": (62_303, 35_707_978_496),
            "visual": (333, 893_142_496),
            "mtp": (1_560, 853_668_480),
            "omitted": (1_893, 1_746_810_976),
            "category_manifests": {
                "language": "8a32168922a925113f7974f3194c7c774cd9658d95225698e1cf9dbe8115f2d1",
                "visual": "14b6e05ee3ae973bd3639f2f9cf1ec142177536bbd7bce5f6211bce6856c3aa6",
                "mtp": "d78589ff9b8b10d35d6289e2254e5890239dfeeba17f375289cdc596cae46118",
            },
            "category_key_names": {
                "language": "e3420af325101fa5c5694c7fb9155ef9e66db6da6296f4f5bc163fcc7daf51ee",
                "visual": "74db978fa718039692971f2ece846f2cc7a62a960047b5cebe2d758cc82ac2e9",
                "mtp": "ef21cc4d7ab3f640ac33aaaf0db8ba47f0a35dbbda908d55fc51416d5db77f9d",
            },
            "omitted_manifest_sha256": "79220a53c9ab8acb66dab66da3f08d4b7e20a94ea20ae46ec161ad225d898c73",
            "omitted_key_names_sha256": "d12905f3e99c7d09046578946c38b3c939e151e34c699052e72d5a6ac44d098f",
            "mixed_files": ["outside.safetensors"],
            "omitted_only_files": ["mtp.safetensors"],
        },
    }
    for precision, surface in expected.items():
        item = checkpoints[precision]
        _require(
            item.get("tensor_count") == surface["tensor_count"]
            and item.get("config_sha256") == surface["config_sha256"]
            and item.get("index_sha256") == surface["index_sha256"]
            and item.get("weight_file_count") == surface["weight_file_count"]
            and item.get("physical_weight_bytes") == surface["physical_weight_bytes"]
            and item.get("logical_tensor_bytes") == surface["logical_tensor_bytes"]
            and item.get("shard_size_manifest_sha256")
            == surface["shard_size_manifest_sha256"]
            and item.get("tensor_metadata_manifest_sha256")
            == surface["tensor_metadata_manifest_sha256"]
            and item.get("file_manifest_sha256") == surface["file_manifest_sha256"]
            and item.get("mixed_language_and_omitted_files") == surface["mixed_files"]
            and item.get("omitted_only_files") == surface["omitted_only_files"]
            and item.get("whole_file_ignore_can_remove_all_omitted_tensors") is False,
            f"{precision} checkpoint identity changed",
        )
        for category in ("language", "visual", "mtp"):
            count, byte_count = surface[category]
            _require(
                item.get("categories", {}).get(category, {}).get("tensor_count") == count
                and item.get("categories", {}).get(category, {}).get("logical_bytes") == byte_count
                and item.get("categories", {}).get(category, {}).get("manifest_sha256")
                == surface["category_manifests"][category]
                and item.get("categories", {}).get(category, {}).get("key_names_sha256")
                == surface["category_key_names"][category],
                f"{precision} {category} inventory changed",
            )
        omitted_count, omitted_bytes = surface["omitted"]
        _require(
            item.get("omitted_tensor_count") == omitted_count
            and item.get("omitted_logical_bytes") == omitted_bytes
            and len(item.get("omitted_tensors", [])) == omitted_count
            and item.get("omitted_manifest_sha256")
            == surface["omitted_manifest_sha256"]
            and item.get("omitted_key_names_sha256")
            == surface["omitted_key_names_sha256"]
            and item.get("other_tensor_count") == 0,
            f"{precision} omission surface changed",
        )
        _require(
            canonical_sha256_v78(item["omitted_tensors"])
            == item["omitted_manifest_sha256"],
            f"{precision} omitted manifest changed",
        )
        omitted_names = [row.get("name") for row in item["omitted_tensors"]]
        _require(
            len(set(omitted_names)) == omitted_count
            and all(
                isinstance(row, dict)
                and set(row)
                == {"name", "file", "dtype", "shape", "logical_bytes", "category"}
                and row["category"] in ("visual", "mtp")
                and (
                    row["name"].startswith("model.visual.")
                    if row["category"] == "visual"
                    else row["name"].startswith("mtp.")
                )
                and isinstance(row["logical_bytes"], int)
                and row["logical_bytes"] > 0
                for row in item["omitted_tensors"]
            )
            and canonical_sha256_v78(omitted_names)
            == surface["omitted_key_names_sha256"],
            f"{precision} omitted key/byte binding changed",
        )
    tokenizer_hashes = {
        "chat_template.jinja": "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259",
        "generation_config.json": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e",
        "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b",
        "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    }
    _require(
        all(
            checkpoints[precision].get("tokenizer_file_sha256")
            == tokenizer_hashes
            for precision in PRECISIONS_V78
        ),
        "tokenizer byte identity changed",
    )
    source_hashes = {
        "model_executor/models/qwen3_5.py": "5f47ae4f4a08d0a78dd681d58b290f3298744c73a82f1349f3e2853469ef73e6",
        "model_executor/models/qwen3_5_mtp.py": "9b36e5bfcee4faf8d04319e069032c3c4a01c4aaf49f86108eca788038c0c7fd",
        "model_executor/models/interfaces.py": "52a4de9e636afe58aaa8f8fa06cff94a2126aec59cae132212944e9d1e0323a5",
        "model_executor/models/utils.py": "921e1f9d1e78cc65bb68b53b4c2648444936c09b9b63e375b352e848cdecf3e8",
        "model_executor/model_loader/default_loader.py": "5d120c07b8eb4d08ce1d4e9759b832a07086dcc78d0df4cefe9beb5c29b7de4e",
        "model_executor/model_loader/weight_utils.py": "ad98e4040aa78fe1803ad47d8e2caf0a67445eb5aeffa00b0bc5525ec7eff198",
        "model_executor/model_loader/ep_weight_filter.py": "09df680b306b9882c9b67779e9bba6450f9d4602e48a876b8d2ee08f94543339",
        "config/multimodal.py": "4df62382d49521afe0196cf18078f9d807939ea088c5442279235585ea3ce612",
        "config/speculative.py": "3f1abd1ca3042fba239e7bf98b08f645f3e950c16ab510fbc99a49c5c507721f",
        "engine/arg_utils.py": "3b3ffa6b403d34188c6d2fe7a2dc36debcee7402a17fc6a6145e885130f3dacd",
    }
    source = body.get("installed_source", {})
    source_rows = source.get("files")
    _require(
        source.get("vllm_version") == "0.25.0"
        and isinstance(source_rows, list)
        and len(source_rows) == len(source_hashes)
        and {
            row.get("relative_path"): row.get("sha256")
            for row in source_rows
            if isinstance(row, dict)
        }
        == source_hashes
        and canonical_sha256_v78(source_rows) == source.get("bundle_sha256")
        and source.get("source_findings", {}).get("vision_construction")
        and source.get("source_findings", {}).get("mtp_target_loading")
        and source.get("source_findings", {}).get("checkpoint_iterator_order"),
        "installed source residency/load-order binding changed",
    )
    conclusion = body.get("cpu_source_conclusion", {})
    _require(
        conclusion.get("vision_checkpoint_present") is True
        and conclusion.get("vision_persistent_live_residency_expected") is False
        and conclusion.get("mtp_checkpoint_present") is True
        and conclusion.get("mtp_persistent_live_residency_expected") is False
        and conclusion.get("live_fp8_measurement_completed") is True
        and conclusion.get("predicted_incremental_VRAM_saving_bytes") == 0,
        "checkpoint/live-residency distinction changed",
    )
    live = body.get("live_fp8_evidence", {})
    _require(
        live.get("run_bundle_sha256")
        == "142fea7a45b62ec87d1d60c35f8819e017b79ac3a4004aa1fdb3e4882d775795"
        and live.get("actor_count") == 4
        and live.get("component_names") == ["language"]
        and live.get("total_parameter_count_per_actor") == 813
        and live.get("total_logical_bytes_per_actor") == 35_712_084_096
        and live.get("parameter_names_sha256")
        == "a850f55c3f02ef904041d48b29f13af2d29834da200f92dcc9728760cb185b90"
        and live.get("visual_named_parameter_count_per_actor") == 0
        and live.get("mtp_named_parameter_count_per_actor") == 0
        and live.get("all_actor_receipt_self_hashes_valid") is True
        and live.get("zero_multimodal_limits_log_actor_count") == 4
        and live.get("speculative_config_none_log_actor_count") == 4,
        "four-actor FP8 live-residency evidence changed",
    )
    _require(
        live.get("run_file_count") == 9
        and isinstance(live.get("run_files"), list)
        and len(live["run_files"]) == 9
        and canonical_sha256_v78(live["run_files"])
        == live["run_bundle_sha256"]
        and isinstance(live.get("actor_receipts"), list)
        and len(live["actor_receipts"]) == 4
        and len(
            {
                row.get("receipt_content_sha256")
                for row in live["actor_receipts"]
                if isinstance(row, dict)
            }
        )
        == 4
        and live.get("dtype_counts_per_actor")
        == {
            "torch.bfloat16": 303,
            "torch.float32": 270,
            "torch.float8_e4m3fn": 240,
        }
        and live.get("device_counts_per_actor") == {"cuda:0": 813}
        and live.get("dataset_or_protected_rows_opened") == 0
        and live.get("training_or_adapter_update_performed") is False,
        "V76 R6 run/receipt evidence surface changed",
    )
    decision = body.get("decision", {})
    _require(
        decision
        == {
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
        },
        "no-artifact decision changed",
    )
    return value


def _checkpoint_for_precision_v78(
    preregistration: dict[str, Any], precision: str
) -> dict[str, Any]:
    validate_preregistration_v78(preregistration)
    _require(precision in PRECISIONS_V78, "unsupported precision")
    return preregistration["checkpoints"][precision]


def validate_candidate_artifact_manifest_v78(
    preregistration: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    _require(isinstance(manifest, dict), "candidate artifact manifest must be an object")
    body = copy.deepcopy(manifest)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(_sha256(claimed) and canonical_sha256_v78(body) == claimed, "candidate artifact self hash changed")
    precision = body.get("precision")
    checkpoint = _checkpoint_for_precision_v78(preregistration, precision)
    _require(
        body.get("schema") == ARTIFACT_SCHEMA_V78
        and body.get("preregistration_sha256")
        == preregistration["content_sha256_before_self_field"]
        and body.get("format") == "standard_huggingface_safetensors"
        and body.get("load_format") == "safetensors"
        and body.get("custom_loader_or_plugin") is False
        and body.get("source_checkpoint_modified") is False
        and body.get("source_and_candidate_roots_differ") is True
        and body.get("symlink_or_external_hardlink_count") == 0,
        "unsupported candidate artifact or loader path",
    )
    _require(
        body.get("source_config_sha256") == checkpoint["config_sha256"]
        and body.get("candidate_config_sha256") == checkpoint["config_sha256"]
        and body.get("tokenizer_file_sha256") == checkpoint["tokenizer_file_sha256"]
        and _sha256(body.get("candidate_index_sha256")),
        "candidate config/tokenizer identity changed",
    )
    language = checkpoint["categories"]["language"]
    _require(
        body.get("retained_tensor_count") == language["tensor_count"]
        and body.get("retained_logical_bytes") == language["logical_bytes"]
        and body.get("retained_metadata_manifest_sha256")
        == language["manifest_sha256"]
        and body.get("retained_key_names_sha256")
        == language["key_names_sha256"]
        and body.get("candidate_index_key_names_sha256")
        == language["key_names_sha256"]
        and body.get("candidate_index_key_count") == language["tensor_count"]
        and body.get("candidate_visual_key_count") == 0
        and body.get("candidate_mtp_key_count") == 0
        and body.get("candidate_other_key_count") == 0,
        "candidate retained-key inventory changed",
    )
    _require(
        body.get("omitted_tensor_count") == checkpoint["omitted_tensor_count"]
        and body.get("omitted_logical_bytes") == checkpoint["omitted_logical_bytes"]
        and body.get("omitted_manifest_sha256")
        == checkpoint["omitted_manifest_sha256"]
        and body.get("omitted_key_names_sha256")
        == checkpoint["omitted_key_names_sha256"],
        "candidate omitted-key inventory changed",
    )
    payload = body.get("payload_identity", {})
    _require(
        payload.get("algorithm") == "sha256_per_tensor_then_canonical_manifest"
        and _sha256(payload.get("source_retained_payload_manifest_sha256"))
        and payload.get("candidate_retained_payload_manifest_sha256")
        == payload.get("source_retained_payload_manifest_sha256")
        and payload.get("all_retained_payloads_exact") is True,
        "retained payload identity was not proven",
    )
    files = body.get("candidate_weight_files")
    _require(isinstance(files, list) and files, "candidate weight-file manifest missing")
    names = []
    physical_bytes = 0
    for row in files:
        _require(
            isinstance(row, dict)
            and set(row) == {"file", "bytes", "sha256"}
            and isinstance(row["file"], str)
            and Path(row["file"]).name == row["file"]
            and row["file"].endswith(".safetensors")
            and _positive(row["bytes"])
            and _sha256(row["sha256"]),
            "unsafe candidate weight-file record",
        )
        names.append(row["file"])
        physical_bytes += row["bytes"]
    _require(len(set(names)) == len(names), "duplicate candidate weight file")
    _require(
        body.get("candidate_physical_weight_bytes") == physical_bytes
        and physical_bytes >= language["logical_bytes"]
        and physical_bytes < checkpoint["physical_weight_bytes"],
        "candidate physical checkpoint size did not improve",
    )
    _require(
        body.get("artifact_creation_authority", {}) == {
            "standard_derivative_only": True,
            "source_checkpoint_modified": False,
            "GPU_used": False,
            "model_or_adapter_updated": False,
            "dataset_or_protected_content_opened": False,
            "promotion_authorized": False,
        },
        "candidate artifact authority changed",
    )
    return manifest


def _validate_live_inventory_v78(
    checkpoint: dict[str, Any], live: dict[str, Any]
) -> None:
    _require(isinstance(live, dict), "live inventory must be an object")
    expected_keys = {
        "outer_model_class",
        "language_model_class",
        "named_parameter_count",
        "named_parameter_bytes",
        "named_parameter_manifest_sha256",
        "visual_named_parameters",
        "mtp_named_parameters",
        "stage_missing_modules",
        "vision_wrapped_parameter_count",
        "vision_wrapped_parameters_all_meta",
        "speculative_module_names",
        "loaded_checkpoint_language_manifest_sha256",
        "LoRA_target_manifest_sha256",
    }
    _require(set(live) == expected_keys, "live inventory fields changed")
    _require(
        live["outer_model_class"] == "Qwen3_5MoeForConditionalGeneration"
        and live["language_model_class"] == "Qwen3_5MoeForCausalLM"
        and isinstance(live["named_parameter_count"], int)
        and live["named_parameter_count"] > 0
        and _positive(live["named_parameter_bytes"])
        and _sha256(live["named_parameter_manifest_sha256"])
        and live["visual_named_parameters"] == []
        and live["mtp_named_parameters"] == []
        and live["speculative_module_names"] == []
        and live["vision_wrapped_parameter_count"] == 333
        and live["vision_wrapped_parameters_all_meta"] is True
        and live["loaded_checkpoint_language_manifest_sha256"]
        == checkpoint["categories"]["language"]["manifest_sha256"]
        and _sha256(live["LoRA_target_manifest_sha256"]),
        "live parameter residency audit failed",
    )
    _require(
        live["stage_missing_modules"] == [
            {"name": "visual", "stage_name": "vision_tower"}
        ],
        "vision tower is not the supported StageMissingLayer",
    )


def validate_actor_receipt_v78(
    preregistration: dict[str, Any],
    candidate_manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    validate_candidate_artifact_manifest_v78(preregistration, candidate_manifest)
    _require(isinstance(receipt, dict), "actor receipt must be an object")
    body = copy.deepcopy(receipt)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(_sha256(claimed) and canonical_sha256_v78(body) == claimed, "actor receipt self hash changed")
    precision = body.get("precision")
    checkpoint = _checkpoint_for_precision_v78(preregistration, precision)
    _require(candidate_manifest["precision"] == precision, "candidate/actor precision mismatch")
    _require(
        body.get("schema") == ACTOR_SCHEMA_V78
        and body.get("arm") in ARMS_V78
        and body.get("physical_gpu") in PHYSICAL_GPUS_V78
        and body.get("preregistration_sha256")
        == preregistration["content_sha256_before_self_field"]
        and body.get("candidate_manifest_sha256")
        == candidate_manifest["content_sha256_before_self_field"]
        and body.get("engine_contract") == engine_contract_v78(),
        "actor identity or engine contract changed",
    )
    _validate_live_inventory_v78(checkpoint, body.get("live_inventory"))
    tokenizer = body.get("tokenizer_file_sha256")
    _require(tokenizer == checkpoint["tokenizer_file_sha256"], "live tokenizer identity changed")
    metrics = body.get("metrics", {})
    _require(
        set(metrics) == {
            "weight_load_seconds",
            "peak_vram_mib",
            "post_load_vram_mib",
            "generated_tokens_per_second",
            "gpu_utilization_percent",
            "memory_activity_percent",
            "power_watts",
        }
        and all(_positive(value) for value in metrics.values())
        and metrics["gpu_utilization_percent"] <= 100
        and metrics["memory_activity_percent"] <= 100,
        "actor performance/telemetry receipt changed",
    )
    identity = body.get("identity", {})
    _require(
        set(identity) == {
            "sealed_prompt_token_ids_sha256",
            "selected_logits_sha256",
            "greedy_token_ids_sha256",
            "candidate_switch_sha256",
            "reference_restore_sha256",
        }
        and all(_sha256(value) for value in identity.values()),
        "actor output-identity hashes missing",
    )
    _require(
        body.get("request_scope") == {
            "text_only": True,
            "multimodal_request_count": 0,
            "multimodal_embedding_request_count": 0,
            "speculative_request_count": 0,
            "dataset_rows_opened": 0,
            "protected_rows_opened": 0,
        }
        and body.get("cleanup") == {
            "engine_shutdown_completed": True,
            "torch_process_group_destroyed": True,
            "actor_process_exited": True,
        },
        "actor scope or cleanup changed",
    )
    return receipt


def validate_paired_runtime_receipt_v78(
    preregistration: dict[str, Any],
    candidate_manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    validate_candidate_artifact_manifest_v78(preregistration, candidate_manifest)
    _require(isinstance(receipt, dict), "paired runtime receipt must be an object")
    body = copy.deepcopy(receipt)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(_sha256(claimed) and canonical_sha256_v78(body) == claimed, "paired receipt self hash changed")
    precision = body.get("precision")
    checkpoint = _checkpoint_for_precision_v78(preregistration, precision)
    _require(
        body.get("schema") == PAIRED_SCHEMA_V78
        and candidate_manifest["precision"] == precision
        and body.get("preregistration_sha256")
        == preregistration["content_sha256_before_self_field"]
        and body.get("candidate_manifest_sha256")
        == candidate_manifest["content_sha256_before_self_field"],
        "paired runtime identity changed",
    )
    replicates = body.get("replicates")
    _require(isinstance(replicates, list) and len(replicates) >= 3, "three paired replicates required")
    load_ratios = []
    throughput_ratios = []
    peak_deltas = []
    orders = set()
    actor_receipt_hashes = []
    for index, pair in enumerate(replicates):
        _require(
            isinstance(pair, dict)
            and set(pair) == {
                "replicate", "order", "physical_gpus", "full_checkpoint",
                "text_only_derivative",
            }
            and pair["replicate"] == index
            and pair["order"] in (
                "full_then_derivative", "derivative_then_full"
            )
            and pair["physical_gpus"] == PHYSICAL_GPUS_V78,
            "paired replicate attribution changed",
        )
        orders.add(pair["order"])
        arms = {}
        for arm in ARMS_V78:
            actors = pair[arm]
            _require(isinstance(actors, list) and len(actors) == 4, "four actor receipts required per arm")
            by_gpu = {}
            for actor in actors:
                validate_actor_receipt_v78(preregistration, candidate_manifest, actor)
                _require(actor["arm"] == arm, "actor placed under wrong arm")
                by_gpu[actor["physical_gpu"]] = actor
                actor_receipt_hashes.append(actor["content_sha256_before_self_field"])
            _require(sorted(by_gpu) == PHYSICAL_GPUS_V78, "actor GPU cardinality changed")
            arms[arm] = by_gpu
        for gpu in PHYSICAL_GPUS_V78:
            full = arms["full_checkpoint"][gpu]
            candidate = arms["text_only_derivative"][gpu]
            full_live = full["live_inventory"]
            candidate_live = candidate["live_inventory"]
            _require(
                full_live["named_parameter_manifest_sha256"]
                == candidate_live["named_parameter_manifest_sha256"]
                and full_live["named_parameter_count"]
                == candidate_live["named_parameter_count"]
                and full_live["named_parameter_bytes"]
                == candidate_live["named_parameter_bytes"],
                "live named-parameter residency differs between artifacts",
            )
            _require(
                full["tokenizer_file_sha256"] == candidate["tokenizer_file_sha256"]
                and full_live["LoRA_target_manifest_sha256"]
                == candidate_live["LoRA_target_manifest_sha256"]
                and full["identity"] == candidate["identity"],
                "tokenizer, LoRA, logits, or token output identity changed",
            )
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
    _require(len(orders) == 2, "artifact arms were not counterbalanced")
    _require(len(set(actor_receipt_hashes)) == len(actor_receipt_hashes), "actor receipt reused")
    gates = body.get("gates", {})
    load = gates.get("load_time", {})
    throughput = gates.get("throughput", {})
    residency = gates.get("residency", {})
    _require(
        load.get("paired_ratios") == load_ratios
        and math.isclose(load.get("median_ratio", -1), statistics.median(load_ratios), rel_tol=1e-12, abs_tol=1e-12)
        and load["median_ratio"] <= 0.98
        and _nonnegative(load.get("paired_bootstrap_95pct_upper_bound"))
        and load["paired_bootstrap_95pct_upper_bound"] <= 1.0
        and _sha256(load.get("bootstrap_draw_plan_sha256"))
        and load.get("pass") is True,
        "load-time improvement gate failed",
    )
    _require(
        throughput.get("paired_ratios") == throughput_ratios
        and math.isclose(throughput.get("median_ratio", -1), statistics.median(throughput_ratios), rel_tol=1e-12, abs_tol=1e-12)
        and throughput["median_ratio"] >= 0.99
        and throughput.get("paired_bootstrap_95pct_lower_bound") >= 0.98
        and _sha256(throughput.get("bootstrap_draw_plan_sha256"))
        and throughput.get("pass") is True,
        "throughput non-inferiority gate failed",
    )
    _require(
        residency.get("candidate_minus_full_peak_vram_mib") == peak_deltas
        and residency.get("maximum_delta_mib") == max(peak_deltas)
        and residency["maximum_delta_mib"] <= 0
        and residency.get("named_parameter_identity_exact") is True
        and residency.get("pass") is True,
        "live-residency equality/non-regression gate failed",
    )
    semantic = gates.get("semantic_validation", {})
    _require(
        semantic.get("selection_frozen_before_access") is True
        and semantic.get("source_disjoint") is True
        and semantic.get("paired_point_delta") >= -0.001
        and semantic.get("paired_95pct_lower_bound") >= -0.002
        and semantic.get("pass") is True,
        "semantic validation gate failed",
    )
    ood = gates.get("protected_ood", {})
    _require(
        ood.get("selection_frozen_before_access") is True
        and ood.get("one_shot") is True
        and ood.get("aggregate_95pct_lower_bound") >= -0.005
        and ood.get("worst_stratum_point_delta") >= -0.01
        and ood.get("new_safety_failures") == 0
        and ood.get("pass") is True,
        "protected OOD gate failed",
    )
    _require(
        body.get("cleanup") == {
            "all_four_processes_exited": True,
            "all_four_gpus_finally_idle": True,
            "foreign_compute_processes": [],
        }
        and body.get("authority") == {
            "measurement_only": True,
            "source_checkpoint_modified": False,
            "model_or_adapter_updated": False,
            "dataset_mutated": False,
            "candidate_promoted": False,
        },
        "paired cleanup or authority changed",
    )
    _require(
        checkpoint["omitted_logical_bytes"] > 0,
        "paired receipt is not bound to an omission opportunity",
    )
    return receipt


def validate_promotion_v78(
    preregistration: dict[str, Any],
    candidate_manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, Any]:
    validate_paired_runtime_receipt_v78(
        preregistration, candidate_manifest, receipt
    )
    _require(
        preregistration.get("status") == "live_audit_complete_promotion_authorized"
        and preregistration.get("authority", {}).get("promotion") is True,
        "promotion blocked by CPU-only preregistration",
    )
    return receipt
