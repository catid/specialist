#!/usr/bin/env python3
"""Build Qwen3.6 with an exact BF16 backbone and only routed experts in FP8."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


SCHEMA = "qwen36-fp8-routed-experts-bf16-backbone-v26"
ROOT = Path(__file__).resolve().parent
DEFAULT_BF16 = Path("/home/catid/specialist/models/Qwen3.6-35B-A3B")
DEFAULT_FP8 = Path("/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8")
DEFAULT_OUTPUT = Path(
    "/home/catid/specialist/models/"
    "Qwen3.6-35B-A3B-FP8-routed-experts-BF16-backbone-v26"
)
PROVENANCE_NAME = "hybrid_routed_experts_bf16_backbone_manifest_v26.json"
BACKBONE_SHARD_TEMPLATE = "bf16-backbone-v26-{index:05d}-of-{total:05d}.safetensors"
SCALE_SUFFIX = "_scale_inv"
EXPECTED_REAL_SOURCE_DIGESTS = {
    "bf16": {
        "config_sha256": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
        "index_sha256": "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
        "metadata_manifest_sha256": "b4cb915c111c930dc48fb389a60cbd640167050126def570c434423bebf0d1c9",
        "weight_shards": {
            "file_count": 26,
            "total_bytes": 71_903_776_776,
            "manifest_sha256": "79777c87360592a91223e2eb93bce9069732dbdc82538ff67af08278ba18ce69",
        },
    },
    "fp8": {
        "config_sha256": "570ef7ea45a7e1d3de2b1d3c70c4ac3562d0e768acdc195778cb4f4d95025845",
        "index_sha256": "6f176f344e41d35b17af12904e33401da5ebff3b49fccb8bfa0185bc2d50f9d6",
        "metadata_manifest_sha256": "76ace9582b6502d99577391aad378b83b99872fb92d91303843bbe5a7e3bdb4e",
        "weight_shards": {
            "file_count": 42,
            "total_bytes": 37_463_662_160,
            "manifest_sha256": "25ae972a0ac80b7875b5e041172d5ad572b522619040f4786a9facdf0e36e5dd",
        },
    },
}
BF16_ROUTED_PATTERN = re.compile(
    r"^(?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts\."
    r"(?:gate_up_proj|down_proj)$"
)
FP8_ROUTED_WEIGHT_PATTERN = re.compile(
    r"^(?:model\.language_model|mtp)\.layers\.\d+\.mlp\.experts\.\d+\."
    r"(?:gate_proj|up_proj|down_proj)\.weight$"
)
DTYPE_BYTES = {"BF16": 2, "F8_E4M3": 1}


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _write_json(path: Path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _is_routed(name: str) -> bool:
    return ".mlp.experts." in name


def _is_scale(name: str) -> bool:
    return name.endswith(".weight" + SCALE_SUFFIX)


def _inventory(model: Path, index):
    weight_map = index.get("weight_map")
    _require(
        isinstance(weight_map, dict)
        and weight_map
        and all(isinstance(name, str) and isinstance(shard, str)
                for name, shard in weight_map.items()),
        "checkpoint weight map changed",
    )
    expected_by_shard = defaultdict(set)
    for name, shard in weight_map.items():
        expected_by_shard[shard].add(name)
    records = {}
    physical = set()
    for shard, expected in sorted(expected_by_shard.items()):
        path = model / shard
        _require(path.is_file() and not path.is_symlink(), "weight shard missing or symlinked")
        with safe_open(path, framework="pt", device="cpu") as source:
            names = set(source.keys())
            _require(names == expected, f"physical/index key mismatch: {shard}")
            _require(not physical.intersection(names), "tensor appears in multiple shards")
            physical.update(names)
            for name in sorted(names):
                value = source.get_slice(name)
                dtype = value.get_dtype()
                shape = tuple(int(item) for item in value.get_shape())
                _require(dtype in DTYPE_BYTES, f"unsupported tensor dtype: {dtype}")
                elements = math.prod(shape)
                records[name] = {
                    "shape": shape,
                    "dtype": dtype,
                    "source_shard": shard,
                    "elements": elements,
                    "bytes": elements * DTYPE_BYTES[dtype],
                }
    _require(physical == set(weight_map), "checkpoint physical key coverage changed")
    return {"weight_map": weight_map, "records": records}


def _group_summary(records, names):
    names = sorted(names)
    metadata = [
        {
            "name": name,
            "shape": list(records[name]["shape"]),
            "dtype": records[name]["dtype"],
            "source_shard": records[name]["source_shard"],
        }
        for name in names
    ]
    return {
        "key_count": len(names),
        "key_sha256": canonical_sha256(names),
        "metadata_manifest_sha256": canonical_sha256(metadata),
        "element_count": sum(records[name]["elements"] for name in names),
        "byte_count": sum(records[name]["bytes"] for name in names),
        "dtype_counts": dict(sorted(Counter(
            records[name]["dtype"] for name in names
        ).items())),
    }


def _weight_shard_file_manifest(model: Path, shard_names):
    records = []
    for name in sorted(set(shard_names)):
        path = model / name
        records.append({
            "file": name,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        })
    return {
        "file_count": len(records),
        "total_bytes": sum(item["bytes"] for item in records),
        "manifest_sha256": canonical_sha256(records),
    }


def _validate_architecture(bf16_config, fp8_config):
    _require(
        bf16_config.get("architectures") == fp8_config.get("architectures")
        and bf16_config.get("model_type") == fp8_config.get("model_type")
        and bf16_config.get("text_config")
        and bf16_config.get("text_config") == fp8_config.get("text_config"),
        "BF16 and FP8 architectures differ",
    )
    quant = fp8_config.get("quantization_config", {})
    _require(
        quant.get("quant_method") == "fp8"
        and quant.get("fmt") == "e4m3"
        and quant.get("activation_scheme") == "dynamic"
        and isinstance(quant.get("weight_block_size"), list)
        and len(quant["weight_block_size"]) == 2
        and all(isinstance(item, int) and item > 0 for item in quant["weight_block_size"]),
        "source is not the expected blockwise dynamic E4M3 FP8 checkpoint",
    )


def _expected_fp8_expert_surface(bf16_routed, bf16_records, num_experts):
    weights = set()
    expected_shapes = {}
    for name in sorted(bf16_routed):
        match = BF16_ROUTED_PATTERN.fullmatch(name)
        _require(match is not None, "BF16 routed expert key grammar changed")
        prefix, projection = name.rsplit(".", 1)
        expert_prefix = prefix + ".{expert}."
        shape = bf16_records[name]["shape"]
        _require(
            len(shape) == 3 and shape[0] == num_experts,
            "BF16 packed routed expert shape changed",
        )
        if projection == "gate_up_proj":
            _require(shape[1] % 2 == 0, "BF16 gate/up packed dimension changed")
            projections = ("gate_proj", "up_proj")
            expected = (shape[1] // 2, shape[2])
        else:
            projections = ("down_proj",)
            expected = (shape[1], shape[2])
        for expert in range(num_experts):
            for item in projections:
                target = expert_prefix.format(expert=expert) + item + ".weight"
                weights.add(target)
                expected_shapes[target] = expected
    return weights, expected_shapes


def _source_contract(
    bf16: Path,
    fp8: Path,
    *,
    include_weight_file_digests: bool,
):
    bf16 = Path(bf16).resolve()
    fp8 = Path(fp8).resolve()
    _require(
        bf16.is_dir() and fp8.is_dir()
        and not bf16.is_symlink() and not fp8.is_symlink(),
        "source model directory is missing or symlinked",
    )
    bf16_config_path = bf16 / "config.json"
    fp8_config_path = fp8 / "config.json"
    bf16_index_path = bf16 / "model.safetensors.index.json"
    fp8_index_path = fp8 / "model.safetensors.index.json"
    bf16_config = _load_json(bf16_config_path)
    fp8_config = _load_json(fp8_config_path)
    bf16_index = _load_json(bf16_index_path)
    fp8_index = _load_json(fp8_index_path)
    _validate_architecture(bf16_config, fp8_config)
    bf16_inventory = _inventory(bf16, bf16_index)
    fp8_inventory = _inventory(fp8, fp8_index)
    bf16_records = bf16_inventory["records"]
    fp8_records = fp8_inventory["records"]

    bf16_routed = {name for name in bf16_records if _is_routed(name)}
    bf16_backbone = set(bf16_records) - bf16_routed
    fp8_routed = {name for name in fp8_records if _is_routed(name)}
    fp8_routed_weights = {name for name in fp8_routed if not _is_scale(name)}
    fp8_routed_scales = fp8_routed - fp8_routed_weights
    fp8_backbone_scales = {
        name for name in fp8_records if not _is_routed(name) and _is_scale(name)
    }
    fp8_backbone = set(fp8_records) - fp8_routed - fp8_backbone_scales
    _require(bf16_backbone == fp8_backbone, "BF16/FP8 backbone key surface changed")
    _require(
        all(BF16_ROUTED_PATTERN.fullmatch(name) for name in bf16_routed),
        "BF16 routed expert representation changed",
    )
    _require(
        all(FP8_ROUTED_WEIGHT_PATTERN.fullmatch(name) for name in fp8_routed_weights),
        "FP8 routed expert weight representation changed",
    )
    _require(
        fp8_routed_scales == {
            name + SCALE_SUFFIX for name in fp8_routed_weights
        },
        "routed expert scale surface changed",
    )
    _require(
        fp8_backbone_scales == {
            name + SCALE_SUFFIX
            for name in fp8_backbone
            if fp8_records[name]["dtype"] == "F8_E4M3"
        },
        "non-routed FP8 scale surface changed",
    )
    _require(
        all(bf16_records[name]["dtype"] == "BF16" for name in bf16_records),
        "BF16 source contains a non-BF16 tensor",
    )
    for name in sorted(bf16_backbone):
        _require(
            bf16_records[name]["shape"] == fp8_records[name]["shape"],
            f"backbone tensor shape changed: {name}",
        )
        _require(
            fp8_records[name]["dtype"] in {"BF16", "F8_E4M3"},
            "FP8 source backbone dtype changed",
        )

    num_experts = bf16_config["text_config"].get("num_experts")
    _require(isinstance(num_experts, int) and num_experts > 0, "expert count changed")
    expected_weights, expected_shapes = _expected_fp8_expert_surface(
        bf16_routed, bf16_records, num_experts
    )
    _require(
        fp8_routed_weights == expected_weights,
        "expanded FP8 routed expert surface changed",
    )
    block_rows, block_columns = fp8_config["quantization_config"][
        "weight_block_size"
    ]
    for name in sorted(fp8_routed_weights):
        record = fp8_records[name]
        _require(
            record["dtype"] == "F8_E4M3"
            and record["shape"] == expected_shapes[name],
            "FP8 routed expert weight dtype or shape changed",
        )
        scale = fp8_records[name + SCALE_SUFFIX]
        expected_scale_shape = (
            math.ceil(record["shape"][0] / block_rows),
            math.ceil(record["shape"][1] / block_columns),
        )
        _require(
            scale["dtype"] == "BF16" and scale["shape"] == expected_scale_shape,
            "FP8 routed expert scale dtype or shape changed",
        )

    groups = {
        "bf16_backbone": _group_summary(bf16_records, bf16_backbone),
        "bf16_packed_routed_experts": _group_summary(bf16_records, bf16_routed),
        "fp8_routed_weights": _group_summary(fp8_records, fp8_routed_weights),
        "fp8_routed_scales": _group_summary(fp8_records, fp8_routed_scales),
        "fp8_removed_backbone_scales": _group_summary(
            fp8_records, fp8_backbone_scales
        ),
    }
    target_keys = bf16_backbone | fp8_routed
    summary = {
        **groups,
        "target_key_count": len(target_keys),
        "target_key_sha256": canonical_sha256(sorted(target_keys)),
        "target_element_count": (
            groups["bf16_backbone"]["element_count"]
            + groups["fp8_routed_weights"]["element_count"]
            + groups["fp8_routed_scales"]["element_count"]
        ),
        "target_byte_count": (
            groups["bf16_backbone"]["byte_count"]
            + groups["fp8_routed_weights"]["byte_count"]
            + groups["fp8_routed_scales"]["byte_count"]
        ),
    }
    source_identity = {
        "bf16": {
            "path": str(bf16),
            "config_sha256": file_sha256(bf16_config_path),
            "index_sha256": file_sha256(bf16_index_path),
            "metadata_manifest_sha256": canonical_sha256([
                {
                    "name": name,
                    "shape": list(bf16_records[name]["shape"]),
                    "dtype": bf16_records[name]["dtype"],
                    "source_shard": bf16_records[name]["source_shard"],
                    "class": "routed_tensor" if name in bf16_routed else "backbone_tensor",
                }
                for name in sorted(bf16_records)
            ]),
        },
        "fp8": {
            "path": str(fp8),
            "config_sha256": file_sha256(fp8_config_path),
            "index_sha256": file_sha256(fp8_index_path),
            "metadata_manifest_sha256": canonical_sha256([
                {
                    "name": name,
                    "shape": list(fp8_records[name]["shape"]),
                    "dtype": fp8_records[name]["dtype"],
                    "source_shard": fp8_records[name]["source_shard"],
                    "class": (
                        "routed_scale" if name in fp8_routed_scales
                        else "routed_tensor" if name in fp8_routed_weights
                        else "backbone_scale" if name in fp8_backbone_scales
                        else "backbone_tensor"
                    ),
                }
                for name in sorted(fp8_records)
            ]),
        },
    }
    if include_weight_file_digests:
        source_identity["bf16"]["weight_shards"] = _weight_shard_file_manifest(
            bf16, bf16_inventory["weight_map"].values()
        )
        source_identity["fp8"]["weight_shards"] = _weight_shard_file_manifest(
            fp8, fp8_inventory["weight_map"].values()
        )
    if bf16 == DEFAULT_BF16.resolve() and fp8 == DEFAULT_FP8.resolve():
        for label in ("bf16", "fp8"):
            expected = {
                key: value
                for key, value in EXPECTED_REAL_SOURCE_DIGESTS[label].items()
                if include_weight_file_digests or key != "weight_shards"
            }
            _require(
                all(source_identity[label].get(key) == value
                    for key, value in expected.items()),
                f"real {label} source identity changed",
            )
        _require(
            summary["target_key_count"] == 63_939
            and summary["target_key_sha256"]
            == "63945fcf2bd6745aa9f492d1ad488d122294a775568f1af9cbe68998cb8b986b",
            "real V26 target key contract changed",
        )
    return {
        "summary": summary,
        "source_identity": source_identity,
        "bf16_config": bf16_config,
        "fp8_config": fp8_config,
        "bf16_inventory": bf16_inventory,
        "fp8_inventory": fp8_inventory,
        "bf16_backbone": bf16_backbone,
        "bf16_routed": bf16_routed,
        "fp8_routed": fp8_routed,
        "fp8_routed_weights": fp8_routed_weights,
        "fp8_routed_scales": fp8_routed_scales,
        "fp8_backbone_scales": fp8_backbone_scales,
        "target_keys": target_keys,
    }


def inspect_source_contract_v26(bf16: Path, fp8: Path):
    contract = _source_contract(
        bf16, fp8, include_weight_file_digests=False
    )
    return {
        "schema": "qwen36-v26-source-representation-contract",
        "summary": contract["summary"],
        "source_identity": contract["source_identity"],
        "contains_dataset_or_evaluation_content": False,
    }


def _backbone_runtime_exclusions(names):
    modules = {
        name.removesuffix(".weight") if name.endswith(".weight") else name
        for name in names
    }
    for name in tuple(modules):
        if name.endswith(".linear_attn.in_proj_qkv") or name.endswith(
            ".linear_attn.in_proj_z"
        ):
            modules.add(name.rsplit(".", 1)[0] + ".in_proj_qkvz")
        if name.endswith(".linear_attn.in_proj_a") or name.endswith(
            ".linear_attn.in_proj_b"
        ):
            modules.add(name.rsplit(".", 1)[0] + ".in_proj_ba")
        if any(name.endswith(f".self_attn.{item}_proj") for item in ("q", "k", "v")):
            modules.add(name.rsplit(".", 1)[0] + ".qkv_proj")
        if name.endswith(".mlp.shared_expert.gate_proj") or name.endswith(
            ".mlp.shared_expert.up_proj"
        ):
            modules.add(name.rsplit(".", 1)[0] + ".gate_up_proj")
    _require(not any(_is_routed(name) for name in modules), "routed expert was excluded from FP8")
    return sorted(modules)


def _link_or_copy(source: Path, destination: Path):
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_subset_shard(source_path: Path, output_path: Path, names):
    names = sorted(names)
    _require(names, f"empty output shard requested: {output_path.name}")
    tensors = {}
    with safe_open(source_path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {"format": "pt"}
        for name in names:
            tensors[name] = source.get_tensor(name).contiguous()
    save_file(tensors, output_path, metadata=metadata)


def _physical_keys(path: Path):
    with safe_open(path, framework="pt", device="cpu") as source:
        return set(source.keys())


def build_hybrid_checkpoint_v26(bf16: Path, fp8: Path, output: Path):
    bf16, fp8, output = (Path(item).resolve() for item in (bf16, fp8, output))
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    contract = _source_contract(
        bf16, fp8, include_weight_file_digests=True
    )
    bf16_map = contract["bf16_inventory"]["weight_map"]
    fp8_map = contract["fp8_inventory"]["weight_map"]
    bf16_backbone = contract["bf16_backbone"]
    fp8_routed = contract["fp8_routed"]
    summary = contract["summary"]

    backbone_by_source = defaultdict(set)
    for name in bf16_backbone:
        backbone_by_source[bf16_map[name]].add(name)
    backbone_sources = sorted(backbone_by_source)
    backbone_output_names = {
        shard: BACKBONE_SHARD_TEMPLATE.format(
            index=index + 1, total=len(backbone_sources)
        )
        for index, shard in enumerate(backbone_sources)
    }
    expert_by_source = defaultdict(set)
    for name in fp8_routed:
        expert_by_source[fp8_map[name]].add(name)
    _require(
        not set(backbone_output_names.values()).intersection(expert_by_source),
        "V26 output shard filename collision",
    )

    runtime_exclusions = _backbone_runtime_exclusions(bf16_backbone)
    source_exclusions = contract["fp8_config"]["quantization_config"].get(
        "modules_to_not_convert", []
    )
    _require(
        isinstance(source_exclusions, list)
        and all(isinstance(name, str) for name in source_exclusions)
        and not any(_is_routed(name) for name in source_exclusions),
        "source FP8 exclusion surface changed",
    )
    all_exclusions = sorted(set(source_exclusions) | set(runtime_exclusions))
    exclusion_summary = {
        "module_count": len(all_exclusions),
        "module_sha256": canonical_sha256(all_exclusions),
        "all_backbone_runtime_modules_excluded": True,
        "no_routed_expert_module_excluded": True,
    }

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    temporary.mkdir(parents=True)
    transfer_modes = Counter()
    try:
        fp8_weight_shards = set(fp8_map.values())
        for source in sorted(fp8.iterdir()):
            if source.name == ".cache" and source.is_dir() and not source.is_symlink():
                continue
            _require(
                source.is_file() and not source.is_symlink(),
                "FP8 source contains a non-file or symlink entry",
            )
            if source.name in fp8_weight_shards | {
                "config.json", "model.safetensors.index.json", PROVENANCE_NAME,
            }:
                continue
            transfer_modes[_link_or_copy(source, temporary / source.name)] += 1

        for source_shard in backbone_sources:
            _write_subset_shard(
                bf16 / source_shard,
                temporary / backbone_output_names[source_shard],
                backbone_by_source[source_shard],
            )
        for source_shard, names in sorted(expert_by_source.items()):
            _write_subset_shard(
                fp8 / source_shard, temporary / source_shard, names
            )

        weight_map = {}
        for name in sorted(bf16_backbone):
            weight_map[name] = backbone_output_names[bf16_map[name]]
        for name in sorted(fp8_routed):
            weight_map[name] = fp8_map[name]
        _require(
            set(weight_map) == contract["target_keys"],
            "V26 target index key set changed",
        )
        hybrid_index = {
            "metadata": {"total_size": summary["target_byte_count"]},
            "weight_map": weight_map,
        }
        _write_json(temporary / "model.safetensors.index.json", hybrid_index)

        hybrid_config = json.loads(json.dumps(contract["fp8_config"]))
        hybrid_config["quantization_config"]["modules_to_not_convert"] = all_exclusions
        hybrid_config["hybrid_routed_experts_fp8_bf16_backbone_v26"] = {
            "schema": SCHEMA,
            "target_key_count": summary["target_key_count"],
            "target_key_sha256": summary["target_key_sha256"],
            "target_element_count": summary["target_element_count"],
            "target_byte_count": summary["target_byte_count"],
            "bf16_backbone_tensor_count": summary["bf16_backbone"]["key_count"],
            "fp8_routed_weight_count": summary["fp8_routed_weights"]["key_count"],
            "fp8_routed_scale_count": summary["fp8_routed_scales"]["key_count"],
            "removed_non_routed_scale_count": summary[
                "fp8_removed_backbone_scales"
            ]["key_count"],
            "runtime_exclusion_sha256": exclusion_summary["module_sha256"],
        }
        _write_json(temporary / "config.json", hybrid_config)

        expected_by_output = defaultdict(set)
        for name, shard in weight_map.items():
            expected_by_output[shard].add(name)
        for shard, expected in sorted(expected_by_output.items()):
            _require(
                _physical_keys(temporary / shard) == expected,
                f"V26 physical/index mismatch: {shard}",
            )
        output_shards = _weight_shard_file_manifest(
            temporary, expected_by_output
        )
        provenance = {
            "schema": SCHEMA,
            "source_identity": contract["source_identity"],
            "tensor_contract": {
                key: summary[key]
                for key in (
                    "bf16_backbone", "bf16_packed_routed_experts",
                    "fp8_routed_weights", "fp8_routed_scales",
                    "fp8_removed_backbone_scales",
                )
            },
            "target_key_count": summary["target_key_count"],
            "target_key_sha256": summary["target_key_sha256"],
            "target_element_count": summary["target_element_count"],
            "target_byte_count": summary["target_byte_count"],
            "removed_non_routed_scale_count": summary[
                "fp8_removed_backbone_scales"
            ]["key_count"],
            "backbone_output_shard_count": len(backbone_sources),
            "routed_expert_output_shard_count": len(expert_by_source),
            "output_weight_shards": output_shards,
            "hybrid_config_sha256": file_sha256(temporary / "config.json"),
            "hybrid_index_sha256": file_sha256(
                temporary / "model.safetensors.index.json"
            ),
            "runtime_exclusions": exclusion_summary,
            "unaffected_file_transfer_modes": dict(sorted(transfer_modes.items())),
            "bf16_backbone_source": "exact_bf16_tensors",
            "routed_expert_source": "exact_fp8_weights_and_scales",
            "non_routed_fp8_scales_retained": False,
            "contains_dataset_or_evaluation_content": False,
        }
        provenance["content_sha256_before_self_field"] = canonical_sha256(provenance)
        _write_json(temporary / PROVENANCE_NAME, provenance)
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, default=DEFAULT_BF16)
    parser.add_argument("--fp8", type=Path, default=DEFAULT_FP8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = build_hybrid_checkpoint_v26(args.bf16, args.fp8, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
