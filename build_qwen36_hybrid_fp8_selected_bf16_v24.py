#!/usr/bin/env python3
"""Build an FP8 Qwen checkpoint with one exact BF16 ES target partition.

The frozen complement remains byte-identical to the source FP8 checkpoint.
Only the dense units selected by an ES layer plan are replaced with tensors
from the BF16 checkpoint.  Quantization scales for those tensors are removed,
and both checkpoint and fused runtime module names are excluded from FP8.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import es_layer_plan


SCHEMA = "qwen36-hybrid-fp8-selected-bf16-v24"
ROOT = Path(__file__).resolve().parent
DEFAULT_BF16 = Path("/home/catid/specialist/models/Qwen3.6-35B-A3B")
DEFAULT_FP8 = Path("/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8")
DEFAULT_OUTPUT = Path(
    "/home/catid/specialist/models/Qwen3.6-35B-A3B-FP8-middle-late-BF16-v24"
)
OVERLAY_NAME = "selected-middle-late-bf16-v24.safetensors"
PROVENANCE_NAME = "hybrid_selected_bf16_manifest.json"


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _tensor(model: Path, weight_map, name: str):
    shard = weight_map.get(name)
    _require(isinstance(shard, str), f"missing indexed tensor: {name}")
    with safe_open(model / shard, framework="pt", device="cpu") as source:
        return source.get_tensor(name)


def _scale_name(weight_name: str) -> str:
    _require(weight_name.endswith(".weight"), "selected unit is not a weight")
    return weight_name + "_scale_inv"


def _runtime_exclusions(selected_units):
    modules = {name.removesuffix(".weight") for name in selected_units}
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
    return sorted(modules)


def _link_or_copy(source: Path, destination: Path):
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _physical_keys(path: Path):
    with safe_open(path, framework="pt", device="cpu") as source:
        return set(source.keys())


def _write_stripped_shard(source_path: Path, output_path: Path, removed):
    tensors = {}
    with safe_open(source_path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {"format": "pt"}
        for name in source.keys():
            if name not in removed:
                tensors[name] = source.get_tensor(name)
    _require(tensors, f"stripping emptied shard: {source_path.name}")
    save_file(tensors, output_path, metadata=metadata)


def _validate_architecture(bf16_config, fp8_config):
    left = bf16_config.get("text_config", {})
    right = fp8_config.get("text_config", {})
    _require(left and left == right, "BF16 and FP8 text configurations differ")
    quant = fp8_config.get("quantization_config", {})
    _require(
        quant.get("quant_method") == "fp8"
        and quant.get("fmt") == "e4m3"
        and quant.get("activation_scheme") == "dynamic",
        "source is not the expected dynamic E4M3 FP8 checkpoint",
    )


def build_hybrid_checkpoint(
    bf16: Path,
    fp8: Path,
    output: Path,
    *,
    layers=(20, 21, 22, 23),
):
    bf16, fp8, output = (Path(item).resolve() for item in (bf16, fp8, output))
    if output.exists():
        raise FileExistsError(f"destination already exists: {output}")
    _require(bf16.is_dir() and fp8.is_dir(), "source model directory is missing")

    bf16_config_path = bf16 / "config.json"
    fp8_config_path = fp8 / "config.json"
    bf16_index_path = bf16 / "model.safetensors.index.json"
    fp8_index_path = fp8 / "model.safetensors.index.json"
    bf16_config = _load_json(bf16_config_path)
    fp8_config = _load_json(fp8_config_path)
    bf16_index = _load_json(bf16_index_path)
    fp8_index = _load_json(fp8_index_path)
    _validate_architecture(bf16_config, fp8_config)

    plan = es_layer_plan.plan_manifest(
        bf16, "front", ["dense"], custom_layers=list(layers)
    )
    selected_units = list(plan["units"])
    _require(len(selected_units) == 35, "hybrid plan must select one 35-unit motif")
    bf16_map = bf16_index.get("weight_map", {})
    fp8_map = fp8_index.get("weight_map", {})
    _require(
        all(name in bf16_map and name in fp8_map for name in selected_units),
        "selected unit is absent from a source checkpoint",
    )

    selected_scales = sorted(
        scale for name in selected_units if (scale := _scale_name(name)) in fp8_map
    )
    _require(selected_scales, "no selected FP8 scales were found")
    removed = set(selected_units) | set(selected_scales)
    affected = sorted({fp8_map[name] for name in removed})
    _require(
        OVERLAY_NAME not in set(fp8_map.values())
        and PROVENANCE_NAME not in {item.name for item in fp8.iterdir()},
        "hybrid output filename collides with FP8 source",
    )

    overlay = {}
    for name in selected_units:
        source_tensor = _tensor(bf16, bf16_map, name)
        quantized_tensor = _tensor(fp8, fp8_map, name)
        _require(
            source_tensor.shape == quantized_tensor.shape,
            f"selected BF16/FP8 tensor shape changed: {name}",
        )
        overlay[name] = source_tensor.contiguous()
    _require(
        all(tensor.dtype == torch.bfloat16 for tensor in overlay.values()),
        "selected source partition is not entirely BF16",
    )
    selected_elements = sum(tensor.numel() for tensor in overlay.values())
    selected_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in overlay.values()
    )

    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    temporary.mkdir(parents=True)
    transfer_modes = defaultdict(int)
    try:
        for source in sorted(fp8.iterdir()):
            if (
                source.name == ".cache"
                and source.is_dir()
                and not source.is_symlink()
            ):
                continue
            _require(source.is_file() and not source.is_symlink(), "FP8 source has non-file entry")
            if source.name in {
                "config.json", "model.safetensors.index.json", PROVENANCE_NAME,
            } or source.name in affected:
                continue
            mode = _link_or_copy(source, temporary / source.name)
            transfer_modes[mode] += 1

        removed_by_shard = defaultdict(set)
        for name in removed:
            removed_by_shard[fp8_map[name]].add(name)
        for shard in affected:
            _write_stripped_shard(
                fp8 / shard, temporary / shard, removed_by_shard[shard]
            )
        save_file(overlay, temporary / OVERLAY_NAME, metadata={"format": "pt"})

        hybrid_index = json.loads(json.dumps(fp8_index))
        weight_map = hybrid_index["weight_map"]
        for name in selected_scales:
            del weight_map[name]
        for name in selected_units:
            weight_map[name] = OVERLAY_NAME
        logical_removed_bytes = 0
        for name in removed:
            value = _tensor(fp8, fp8_map, name)
            logical_removed_bytes += value.numel() * value.element_size()
        metadata = hybrid_index.setdefault("metadata", {})
        if "total_size" in metadata:
            metadata["total_size"] = (
                int(metadata["total_size"]) - logical_removed_bytes + selected_bytes
            )
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(hybrid_index, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        hybrid_config = json.loads(json.dumps(fp8_config))
        quant = hybrid_config["quantization_config"]
        exclusions = set(quant.get("modules_to_not_convert", []))
        runtime_exclusions = _runtime_exclusions(selected_units)
        exclusions.update(runtime_exclusions)
        quant["modules_to_not_convert"] = sorted(exclusions)
        hybrid_config["hybrid_selected_bf16"] = {
            "schema": SCHEMA,
            "layer_plan_sha256": plan["plan_sha256"],
            "layers": list(layers),
            "selected_unit_count": len(selected_units),
            "selected_element_count": selected_elements,
            "selected_byte_count": selected_bytes,
        }
        (temporary / "config.json").write_text(
            json.dumps(hybrid_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        expected_by_shard = defaultdict(set)
        for name, shard in weight_map.items():
            expected_by_shard[shard].add(name)
        for shard, expected in expected_by_shard.items():
            _require(
                _physical_keys(temporary / shard) == expected,
                f"hybrid physical/index mismatch: {shard}",
            )
        _require(
            _physical_keys(temporary / OVERLAY_NAME) == set(selected_units),
            "hybrid BF16 overlay key set changed",
        )

        provenance = {
            "schema": SCHEMA,
            "bf16_source": {
                "path": str(bf16),
                "config_sha256": file_sha256(bf16_config_path),
                "index_sha256": file_sha256(bf16_index_path),
            },
            "fp8_source": {
                "path": str(fp8),
                "config_sha256": file_sha256(fp8_config_path),
                "index_sha256": file_sha256(fp8_index_path),
            },
            "layer_plan": plan,
            "selected_unit_count": len(selected_units),
            "selected_scale_count_removed": len(selected_scales),
            "selected_element_count": selected_elements,
            "selected_byte_count": selected_bytes,
            "affected_fp8_shards": affected,
            "rewritten_fp8_shard_count": len(affected),
            "overlay_file": OVERLAY_NAME,
            "overlay_file_sha256": file_sha256(temporary / OVERLAY_NAME),
            "runtime_modules_excluded_from_fp8": runtime_exclusions,
            "unaffected_file_transfer_modes": dict(sorted(transfer_modes.items())),
            "frozen_complement_source": "byte_identical_fp8_files_or_stripped_fp8_shards",
            "selected_partition_source": "exact_bf16_overlay",
        }
        provenance["content_sha256_before_self_field"] = canonical_sha256(provenance)
        (temporary / PROVENANCE_NAME).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
    parser.add_argument("--layers", default="20,21,22,23")
    args = parser.parse_args(argv)
    layers = tuple(int(item) for item in args.layers.split(","))
    result = build_hybrid_checkpoint(
        args.bf16, args.fp8, args.output, layers=layers
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
