#!/usr/bin/env python3
"""Build the CPU-only Qwen3.6 text-only residency preregistration.

The builder parses safetensor headers directly, reads installed-source text,
and validates already-produced data-free V76 actor receipts.  It imports
neither torch nor vLLM, creates no derivative artifact, and cannot initialize
CUDA.
"""

from __future__ import annotations

import argparse
import copy
import json
import struct
from collections import Counter
from pathlib import Path
from typing import Any

import qwen36_text_only_checkpoint_v78 as contract


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_text_only_checkpoint_v78.json"
)
VLLM_ROOT = ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/vllm"
BF16_ROOT = ROOT / "models/Qwen3.6-35B-A3B"
FP8_ROOT = ROOT / "models/Qwen3.6-35B-A3B-FP8"
V76_R6_ROOT = ROOT / (
    "experiments/eggroll_es_hpo/runs/v76_fp8_attested_050_r6_residency"
)
V76_R6_BUNDLE_SHA256 = (
    "142fea7a45b62ec87d1d60c35f8819e017b79ac3a4004aa1fdb3e4882d775795"
)

TOKENIZER_SHA256 = {
    "chat_template.jinja": (
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    ),
    "generation_config.json": (
        "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
    ),
    "merges.txt": (
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": (
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ),
    "vocab.json": (
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
    ),
}

CHECKPOINT_EXPECTATIONS = {
    "bf16": {
        "root": BF16_ROOT,
        "config_sha256": (
            "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
        ),
        "index_sha256": (
            "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
        ),
        "tensor_count": 1_045,
        "weight_file_count": 26,
        "physical_weight_bytes": 71_903_776_776,
        "logical_tensor_bytes": 71_903_645_408,
        "shard_size_manifest_sha256": (
            "b878392051d6a40017ea4d096c786470aee2dab5e0440b33bb1b7b92142c60c9"
        ),
        "tensor_metadata_manifest_sha256": (
            "d3dbdde871f50ceffa43ecc5e203b0d251bd09438837707d6f2f178b3acb4d37"
        ),
        "file_manifest_sha256": (
            "9377411657cf12ebaf41c94b39d9217d5baa09bb77555ca8ca3a9893a81eb3ae"
        ),
        "category_manifest_sha256": {
            "language": (
                "066ced8ee35a27bbed875d4f73a1257802600349990a6c4ca1bb4e643cd8dcab"
            ),
            "visual": (
                "22e59a49560759a7d89a53caefb2b3e872500f95404809becec9dbc1417d5ef7"
            ),
            "mtp": (
                "5fef680476646e7e5741671f9374cea9c873a8a5e9d4d13ece980e91c6b19f1c"
            ),
        },
        "category_key_names_sha256": {
            "language": (
                "3009dd3c57dc1caf938c765567fee074d64eb5fe677110f67ccec0af4316395a"
            ),
            "visual": (
                "74db978fa718039692971f2ece846f2cc7a62a960047b5cebe2d758cc82ac2e9"
            ),
            "mtp": (
                "76e2ce7a2d481fdadb6f5d969e890b632fbc35570a52c4677d9605dc4bc81097"
            ),
        },
        "omitted_manifest_sha256": (
            "670c5a2d0ecdee733b17f44a23b690a34e8a4754a0785a04e5f3e845729cefe9"
        ),
        "omitted_key_names_sha256": (
            "b7f3e717e4223a0b1079041f080fc9e47ee44dfe816548ca418638745471463a"
        ),
    },
    "fp8_serialized": {
        "root": FP8_ROOT,
        "config_sha256": (
            "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845"
        ),
        "index_sha256": (
            "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6"
        ),
        "tensor_count": 64_196,
        "weight_file_count": 42,
        "physical_weight_bytes": 37_463_662_160,
        "logical_tensor_bytes": 37_454_789_472,
        "shard_size_manifest_sha256": (
            "3bc687cefc89541a4768655cd110ffecf1d6a1ca94b5072903037dab119dcdaf"
        ),
        "tensor_metadata_manifest_sha256": (
            "11d40e408ad3def3cb28fc698d47a32c2db76e484e3f1cf61ffbd4e89e97de50"
        ),
        "file_manifest_sha256": (
            "46a29c0d5659539b0ac01941c7ce50421ad43cef7f42faf65996953c997456d6"
        ),
        "category_manifest_sha256": {
            "language": (
                "8a32168922a925113f7974f3194c7c774cd9658d95225698e1cf9dbe8115f2d1"
            ),
            "visual": (
                "14b6e05ee3ae973bd3639f2f9cf1ec142177536bbd7bce5f6211bce6856c3aa6"
            ),
            "mtp": (
                "d78589ff9b8b10d35d6289e2254e5890239dfeeba17f375289cdc596cae46118"
            ),
        },
        "category_key_names_sha256": {
            "language": (
                "e3420af325101fa5c5694c7fb9155ef9e66db6da6296f4f5bc163fcc7daf51ee"
            ),
            "visual": (
                "74db978fa718039692971f2ece846f2cc7a62a960047b5cebe2d758cc82ac2e9"
            ),
            "mtp": (
                "ef21cc4d7ab3f640ac33aaaf0db8ba47f0a35dbbda908d55fc51416d5db77f9d"
            ),
        },
        "omitted_manifest_sha256": (
            "79220a53c9ab8acb66dab66da3f08d4b7e20a94ea20ae46ec161ad225d898c73"
        ),
        "omitted_key_names_sha256": (
            "d12905f3e99c7d09046578946c38b3c939e151e34c699052e72d5a6ac44d098f"
        ),
    },
}

SOURCE_SHA256 = {
    "model_executor/models/qwen3_5.py": (
        "5f47ae4f4a08d0a78dd681d58b290f3298744c73a82f1349f3e2853469ef73e6"
    ),
    "model_executor/models/qwen3_5_mtp.py": (
        "9b36e5bfcee4faf8d04319e069032c3c4a01c4aaf49f86108eca788038c0c7fd"
    ),
    "model_executor/models/interfaces.py": (
        "52a4de9e636afe58aaa8f8fa06cff94a2126aec59cae132212944e9d1e0323a5"
    ),
    "model_executor/models/utils.py": (
        "921e1f9d1e78cc65bb68b53b4c2648444936c09b9b63e375b352e848cdecf3e8"
    ),
    "model_executor/model_loader/default_loader.py": (
        "5d120c07b8eb4d08ce1d4e9759b832a07086dcc78d0df4cefe9beb5c29b7de4e"
    ),
    "model_executor/model_loader/weight_utils.py": (
        "ad98e4040aa78fe1803ad47d8e2caf0a67445eb5aeffa00b0bc5525ec7eff198"
    ),
    "model_executor/model_loader/ep_weight_filter.py": (
        "09df680b306b9882c9b67779e9bba6450f9d4602e48a876b8d2ee08f94543339"
    ),
    "config/multimodal.py": (
        "4df62382d49521afe0196cf18078f9d807939ea088c5442279235585ea3ce612"
    ),
    "config/speculative.py": (
        "3f1abd1ca3042fba239e7bf98b08f645f3e950c16ab510fbc99a49c5c507721f"
    ),
    "engine/arg_utils.py": (
        "3b3ffa6b403d34188c6d2fe7a2dc36debcee7402a17fc6a6145e885130f3dacd"
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_no_duplicates(path: Path) -> dict[str, Any]:
    def hook(pairs):
        value = {}
        for key, item in pairs:
            _require(key not in value, f"duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _tensor_category(name: str) -> str:
    if name.startswith(("model.language_model.", "lm_head.")):
        return "language"
    if name.startswith("model.visual."):
        return "visual"
    if name.startswith("mtp."):
        return "mtp"
    return "other"


def _read_safetensor_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as source:
        raw = source.read(8)
        _require(len(raw) == 8, f"truncated safetensor: {path}")
        header_bytes = struct.unpack("<Q", raw)[0]
        _require(0 < header_bytes < 64 << 20, f"unsafe safetensor header: {path}")
        payload = source.read(header_bytes)
        _require(len(payload) == header_bytes, f"truncated safetensor header: {path}")
    value = json.loads(payload)
    _require(isinstance(value, dict), f"invalid safetensor header: {path}")
    return header_bytes, value


def inspect_checkpoint_v78(precision: str) -> dict[str, Any]:
    expected = CHECKPOINT_EXPECTATIONS[precision]
    root = Path(expected["root"])
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    _require(root.is_dir() and not root.is_symlink(), f"missing checkpoint: {root}")
    _require(
        config_path.is_file()
        and not config_path.is_symlink()
        and index_path.is_file()
        and not index_path.is_symlink(),
        f"unsafe checkpoint metadata path: {root}",
    )
    _require(
        contract.file_sha256_v78(config_path) == expected["config_sha256"]
        and contract.file_sha256_v78(index_path) == expected["index_sha256"],
        f"{precision} config/index identity changed",
    )
    config = _json_no_duplicates(config_path)
    index = _json_no_duplicates(index_path)
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict) and weight_map, "checkpoint weight map missing")
    filenames = sorted(set(weight_map.values()))
    _require(
        len(weight_map) == expected["tensor_count"]
        and len(filenames) == expected["weight_file_count"],
        f"{precision} checkpoint cardinality changed",
    )
    for key, filename in weight_map.items():
        _require(isinstance(key, str) and key, "invalid tensor name")
        _require(
            isinstance(filename, str)
            and Path(filename).name == filename
            and filename.endswith(".safetensors"),
            "unsafe weight-map filename",
        )

    tensor_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    seen = set()
    for filename in filenames:
        path = root / filename
        _require(path.is_file() and not path.is_symlink(), f"unsafe shard: {path}")
        header_size, header = _read_safetensor_header(path)
        local_rows = []
        maximum_end = 0
        for name in sorted(key for key in header if key != "__metadata__"):
            _require(name not in seen, f"duplicate tensor: {name}")
            _require(weight_map.get(name) == filename, f"index/header mismatch: {name}")
            metadata = header[name]
            _require(isinstance(metadata, dict), f"invalid tensor metadata: {name}")
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            _require(
                isinstance(dtype, str)
                and isinstance(shape, list)
                and all(isinstance(item, int) and item >= 0 for item in shape)
                and isinstance(offsets, list)
                and len(offsets) == 2
                and all(isinstance(item, int) and item >= 0 for item in offsets)
                and offsets[1] >= offsets[0],
                f"invalid tensor metadata: {name}",
            )
            maximum_end = max(maximum_end, offsets[1])
            row = {
                "name": name,
                "file": filename,
                "dtype": dtype,
                "shape": shape,
                "logical_bytes": offsets[1] - offsets[0],
                "category": _tensor_category(name),
            }
            local_rows.append(row)
            tensor_rows.append(row)
            seen.add(name)
        _require(
            8 + header_size + maximum_end <= path.stat().st_size,
            f"tensor data exceeds shard: {filename}",
        )
        categories = {}
        for category in ("language", "visual", "mtp", "other"):
            selected = [row for row in local_rows if row["category"] == category]
            categories[category] = {
                "tensor_count": len(selected),
                "logical_bytes": sum(row["logical_bytes"] for row in selected),
            }
        file_rows.append(
            {
                "file": filename,
                "physical_bytes": path.stat().st_size,
                "header_bytes": 8 + header_size,
                "tensor_count": len(local_rows),
                "logical_tensor_bytes": sum(
                    row["logical_bytes"] for row in local_rows
                ),
                "categories": categories,
            }
        )
    _require(seen == set(weight_map), f"{precision} index/header key set changed")
    tensor_rows.sort(key=lambda row: row["name"])
    size_rows = [
        {"file": row["file"], "bytes": row["physical_bytes"]}
        for row in file_rows
    ]
    categories = {}
    for category in ("language", "visual", "mtp"):
        selected = [row for row in tensor_rows if row["category"] == category]
        categories[category] = {
            "tensor_count": len(selected),
            "logical_bytes": sum(row["logical_bytes"] for row in selected),
            "manifest_sha256": contract.canonical_sha256_v78(selected),
            "key_names_sha256": contract.canonical_sha256_v78(
                [row["name"] for row in selected]
            ),
            "dtype_tensor_counts": dict(
                sorted(Counter(row["dtype"] for row in selected).items())
            ),
        }
    other = [row for row in tensor_rows if row["category"] == "other"]
    omitted = [
        row for row in tensor_rows if row["category"] in ("visual", "mtp")
    ]
    mixed = [
        row["file"]
        for row in file_rows
        if row["categories"]["language"]["tensor_count"]
        and (
            row["categories"]["visual"]["tensor_count"]
            or row["categories"]["mtp"]["tensor_count"]
        )
    ]
    omitted_only = [
        row["file"]
        for row in file_rows
        if not row["categories"]["language"]["tensor_count"]
        and (
            row["categories"]["visual"]["tensor_count"]
            or row["categories"]["mtp"]["tensor_count"]
        )
    ]
    _require(
        sum(row["physical_bytes"] for row in file_rows)
        == expected["physical_weight_bytes"]
        and sum(row["logical_bytes"] for row in tensor_rows)
        == expected["logical_tensor_bytes"]
        and contract.canonical_sha256_v78(size_rows)
        == expected["shard_size_manifest_sha256"]
        and contract.canonical_sha256_v78(tensor_rows)
        == expected["tensor_metadata_manifest_sha256"]
        and contract.canonical_sha256_v78(file_rows)
        == expected["file_manifest_sha256"]
        and all(
            categories[name]["manifest_sha256"]
            == expected["category_manifest_sha256"][name]
            for name in ("language", "visual", "mtp")
        )
        and all(
            categories[name]["key_names_sha256"]
            == expected["category_key_names_sha256"][name]
            for name in ("language", "visual", "mtp")
        )
        and contract.canonical_sha256_v78(omitted)
        == expected["omitted_manifest_sha256"]
        and contract.canonical_sha256_v78([row["name"] for row in omitted])
        == expected["omitted_key_names_sha256"],
        f"{precision} checkpoint byte inventory changed",
    )
    _require(not other, f"{precision} has an unclassified tensor prefix")
    tokenizer = {}
    for filename, expected_hash in TOKENIZER_SHA256.items():
        path = root / filename
        _require(
            path.is_file()
            and not path.is_symlink()
            and contract.file_sha256_v78(path) == expected_hash,
            f"{precision} tokenizer file changed: {filename}",
        )
        tokenizer[filename] = expected_hash
    text = config.get("text_config", {})
    _require(
        config.get("architectures") == ["Qwen3_5MoeForConditionalGeneration"]
        and config.get("model_type") == "qwen3_5_moe"
        and text.get("model_type") == "qwen3_5_moe_text"
        and text.get("hidden_size") == 2048
        and text.get("num_hidden_layers") == 40
        and text.get("num_experts") == 256
        and text.get("num_experts_per_tok") == 8,
        f"{precision} Qwen geometry changed",
    )
    return {
        "source_root": str(root.relative_to(ROOT)),
        "config_sha256": expected["config_sha256"],
        "index_sha256": expected["index_sha256"],
        "tensor_count": len(tensor_rows),
        "weight_file_count": len(file_rows),
        "physical_weight_bytes": sum(row["physical_bytes"] for row in file_rows),
        "logical_tensor_bytes": sum(row["logical_bytes"] for row in tensor_rows),
        "shard_size_manifest_sha256": contract.canonical_sha256_v78(size_rows),
        "tensor_metadata_manifest_sha256": contract.canonical_sha256_v78(
            tensor_rows
        ),
        "file_manifest_sha256": contract.canonical_sha256_v78(file_rows),
        "weight_files": file_rows,
        "categories": categories,
        "other_tensor_count": 0,
        "omitted_tensor_count": len(omitted),
        "omitted_logical_bytes": sum(row["logical_bytes"] for row in omitted),
        "omitted_manifest_sha256": contract.canonical_sha256_v78(omitted),
        "omitted_key_names_sha256": contract.canonical_sha256_v78(
            [row["name"] for row in omitted]
        ),
        "omitted_tensors": omitted,
        "mixed_language_and_omitted_files": mixed,
        "omitted_only_files": omitted_only,
        "whole_file_ignore_can_remove_all_omitted_tensors": not mixed,
        "tokenizer_file_sha256": tokenizer,
    }


def inspect_installed_source_v78() -> dict[str, Any]:
    rows = []
    texts = {}
    for relative, expected_hash in SOURCE_SHA256.items():
        path = VLLM_ROOT / relative
        _require(path.is_file() and not path.is_symlink(), f"missing source: {relative}")
        _require(
            contract.file_sha256_v78(path) == expected_hash,
            f"installed source changed: {relative}",
        )
        texts[relative] = path.read_text(encoding="utf-8")
        rows.append(
            {"relative_path": relative, "bytes": path.stat().st_size, "sha256": expected_hash}
        )
    qwen = texts["model_executor/models/qwen3_5.py"]
    interfaces = texts["model_executor/models/interfaces.py"]
    utils = texts["model_executor/models/utils.py"]
    loader = texts["model_executor/model_loader/default_loader.py"]
    weights = texts["model_executor/model_loader/weight_utils.py"]
    ep_filter = texts["model_executor/model_loader/ep_weight_filter.py"]
    multimodal = texts["config/multimodal.py"]
    speculative = texts["config/speculative.py"]
    arg_utils = texts["engine/arg_utils.py"]
    mtp = texts["model_executor/models/qwen3_5_mtp.py"]
    _require(
        qwen.count('with self._mark_tower_model(vllm_config, {"image", "video"}):')
        >= 2
        and qwen.count('skip_prefixes=["mtp."]') >= 2
        and 'all(mm_config.get_limit_per_prompt(m) == 0 for m in modalities)'
        in interfaces
        and 'lambda mod: StageMissingLayer(stage_name, mod)' in interfaces
        and 'with register_module_module_registration_hook(hook), torch.device("meta"):'
        in utils
        and 'self.__dict__["module"] = module' in utils
        and 'if isinstance(module, (StageMissingLayer, PPMissingLayer)):' in utils
        and "loaded_weights = model.load_weights(self.get_all_weights(model_config, model))"
        in loader
        and "param = f.get_tensor(name)" in weights
        and "if local_expert_ids is None:\n        return False" in ep_filter
        and "if self.language_model_only:\n            return 0" in multimodal
        and '"architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"]'
        in speculative
        and "if self.speculative_config is None:\n            return None" in arg_utils
        and "class Qwen3_5MoeMTP" in mtp,
        "installed omission/load-order source markers changed",
    )
    return {
        "vllm_version": "0.25.0",
        "files": rows,
        "bundle_sha256": contract.canonical_sha256_v78(rows),
        "source_findings": {
            "vision_construction": (
                "zero image/video limits construct the tower under meta and replace "
                "the registered child with StageMissingLayer"
            ),
            "stage_missing_registration": (
                "the wrapped original module is stored through __dict__, so it is "
                "not a registered persistent child"
            ),
            "stage_missing_loading": "AutoWeightsLoader skips StageMissingLayer",
            "mtp_target_loading": "Qwen3.5 target load_weights skips mtp.*",
            "mtp_instantiation": (
                "Qwen3_5MoeMTP is selected only through a non-null speculative config"
            ),
            "checkpoint_iterator_order": (
                "the default safetensors iterator materializes a tensor before the "
                "model-level StageMissing/mtp skip consumes it"
            ),
            "tp1_expert_filter": (
                "with no local expert-id filter, the pre-read filter returns false"
            ),
        },
    }


def _validate_receipt_self_hash(value: dict[str, Any], path: Path) -> None:
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        claimed == contract.canonical_sha256_v78(body),
        f"V76 receipt self hash changed: {path}",
    )


def inspect_live_fp8_v78() -> dict[str, Any]:
    _require(
        V76_R6_ROOT.is_dir() and not V76_R6_ROOT.is_symlink(),
        "missing V76 R6 residency run",
    )
    files = []
    for path in sorted(V76_R6_ROOT.iterdir()):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": contract.file_sha256_v78(path),
                }
            )
    _require(
        len(files) == 9
        and contract.canonical_sha256_v78(files) == V76_R6_BUNDLE_SHA256,
        "V76 R6 run inventory changed",
    )
    receipt_paths = sorted(V76_R6_ROOT.glob("gpu_*.json"))
    log_paths = sorted(V76_R6_ROOT.glob("gpu_*.log"))
    _require(len(receipt_paths) == len(log_paths) == 4, "V76 R6 actor surface changed")
    receipt_rows = []
    expected_residency = {
        "components": {
            "language": {
                "device_counts": {"cuda:0": 813},
                "dtype_counts": {
                    "torch.bfloat16": 303,
                    "torch.float32": 270,
                    "torch.float8_e4m3fn": 240,
                },
                "logical_bytes": 35_712_084_096,
                "parameter_count": 813,
                "parameter_names_sha256": (
                    "a850f55c3f02ef904041d48b29f13af2d29834da200f92dcc9728760cb185b90"
                ),
            }
        },
        "named_parameters_remove_duplicate_default": True,
        "schema": "v76-live-named-parameter-residency",
        "total_logical_bytes": 35_712_084_096,
        "total_parameter_count": 813,
    }
    for index, path in enumerate(receipt_paths):
        value = _json_no_duplicates(path)
        _validate_receipt_self_hash(value, path)
        _require(
            value.get("schema") == "v76-qwen36-fp8-routed-runtime-attestation"
            and value.get("actor_label") == f"gpu-{index}"
            and value.get("precision_arm") == "fp8_serialized"
            and value.get("source_dataset_rows_opened") == 0
            and value.get("protected_ood_shadow_or_terminal_opened") is False
            and value.get("adapter_update_or_hpo_performed") is False
            and value.get("routed_fp8_runtime_attestation", {}).get(
                "parameter_residency"
            )
            == expected_residency,
            "V76 R6 residency receipt changed",
        )
        receipt_rows.append(
            {
                "actor_label": value["actor_label"],
                "receipt_relative_path": str(path.relative_to(ROOT)),
                "receipt_file_sha256": contract.file_sha256_v78(path),
                "receipt_content_sha256": value[
                    "content_sha256_before_self_field"
                ],
            }
        )
    texts = [path.read_text(encoding="utf-8") for path in log_paths]
    zero_limit_pattern = "'limit_mm_per_prompt': {'image': 0, 'video': 0}"
    spec_none_pattern = "speculative_config=None"
    _require(
        sum(zero_limit_pattern in text for text in texts) == 4
        and sum(spec_none_pattern in text for text in texts) == 4,
        "V76 R6 text-only engine log evidence changed",
    )
    language = expected_residency["components"]["language"]
    return {
        "run_relative_path": str(V76_R6_ROOT.relative_to(ROOT)),
        "run_bundle_sha256": V76_R6_BUNDLE_SHA256,
        "run_file_count": len(files),
        "run_files": files,
        "actor_count": 4,
        "actor_receipts": receipt_rows,
        "all_actor_receipt_self_hashes_valid": True,
        "component_names": ["language"],
        "total_parameter_count_per_actor": language["parameter_count"],
        "total_logical_bytes_per_actor": language["logical_bytes"],
        "dtype_counts_per_actor": language["dtype_counts"],
        "device_counts_per_actor": language["device_counts"],
        "parameter_names_sha256": language["parameter_names_sha256"],
        "visual_named_parameter_count_per_actor": 0,
        "mtp_named_parameter_count_per_actor": 0,
        "zero_multimodal_limits_log_actor_count": 4,
        "speculative_config_none_log_actor_count": 4,
        "dataset_or_protected_rows_opened": 0,
        "training_or_adapter_update_performed": False,
    }


def build_preregistration_v78() -> dict[str, Any]:
    checkpoints = {
        precision: inspect_checkpoint_v78(precision)
        for precision in contract.PRECISIONS_V78
    }
    live = inspect_live_fp8_v78()
    value: dict[str, Any] = {
        "schema": contract.PREREG_SCHEMA_V78,
        "bead": "specialist-0j5.23",
        "status": "inventory_and_live_fp8_audit_complete_no_artifact_recommended",
        "authority": {
            "gpu_launch": False,
            "protected_or_ood_access": False,
            "candidate_artifact_creation": False,
            "source_checkpoint_modification": False,
            "site_package_or_es_at_scale_modification": False,
            "training_or_model_update": False,
            "promotion": False,
        },
        "decision": {
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
        "engine_contract": contract.engine_contract_v78(),
        "supported_artifact_plan": contract.supported_artifact_plan_v78(),
        "runtime_gates": contract.runtime_gates_v78(),
        "checkpoints": checkpoints,
        "installed_source": inspect_installed_source_v78(),
        "live_fp8_evidence": live,
        "cpu_source_conclusion": {
            "checkpoint_bytes_and_live_residency_are_distinct": True,
            "vision_checkpoint_present": True,
            "vision_persistent_live_residency_expected": False,
            "mtp_checkpoint_present": True,
            "mtp_persistent_live_residency_expected": False,
            "live_fp8_measurement_completed": True,
            "live_bf16_measurement_completed": False,
            "predicted_incremental_VRAM_saving_bytes": 0,
            "observed_fp8_visual_named_parameter_count_per_actor": 0,
            "observed_fp8_mtp_named_parameter_count_per_actor": 0,
            "full_checkpoint_iterator_still_reads_omitted_payloads": True,
            "possible_benefit_class": "checkpoint_storage_and_startup_IO_only",
        },
        "side_effects": {
            "torch_or_vllm_imported": False,
            "CUDA_initialized_or_accessed_by_builder": False,
            "candidate_artifact_created": False,
            "source_checkpoint_modified": False,
            "site_packages_or_es_at_scale_modified": False,
            "dataset_or_protected_content_opened": False,
        },
    }
    value["content_sha256_before_self_field"] = contract.canonical_sha256_v78(value)
    return contract.validate_preregistration_v78(value)


def write_preregistration_v78(path: Path = OUTPUT) -> dict[str, Any]:
    value = build_preregistration_v78()
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
    value = build_preregistration_v78()
    if args.check:
        _require(args.output.is_file(), f"missing preregistration: {args.output}")
        frozen = _json_no_duplicates(args.output)
        contract.validate_preregistration_v78(frozen)
        _require(frozen == value, "frozen preregistration is stale")
    else:
        write_preregistration_v78(args.output)
    print(
        json.dumps(
            {
                "schema": contract.PREREG_SCHEMA_V78,
                "output": str(args.output),
                "status": value["status"],
                "content_sha256": value["content_sha256_before_self_field"],
                "candidate_artifact_created": False,
                "gpu_launch": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
