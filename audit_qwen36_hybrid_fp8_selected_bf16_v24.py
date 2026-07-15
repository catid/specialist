#!/usr/bin/env python3
"""Fail-closed post-build audit for the Qwen3.6 hybrid FP8/BF16 checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

import build_qwen36_hybrid_fp8_selected_bf16_v24 as builder


ROOT = Path(__file__).resolve().parent
DEFAULT_BF16 = ROOT / "models/Qwen3.6-35B-A3B"
DEFAULT_FP8 = ROOT / "models/Qwen3.6-35B-A3B-FP8"
DEFAULT_HYBRID = ROOT / "models/Qwen3.6-35B-A3B-FP8-middle-late-BF16-v24"
AUDIT_SCHEMA = "qwen36-hybrid-fp8-selected-bf16-audit-v24"


def _load_json(path: Path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _keys(path: Path):
    with safe_open(path, framework="pt", device="cpu") as source:
        return set(source.keys())


def _compare_shard_tensors(left: Path, right: Path, names):
    names = sorted(names)
    with safe_open(left, framework="pt", device="cpu") as left_source, safe_open(
        right, framework="pt", device="cpu"
    ) as right_source:
        for name in names:
            left_tensor = left_source.get_tensor(name)
            right_tensor = right_source.get_tensor(name)
            _require(
                left_tensor.dtype == right_tensor.dtype,
                f"retained tensor dtype changed: {name}",
            )
            _require(
                tuple(left_tensor.shape) == tuple(right_tensor.shape),
                f"retained tensor shape changed: {name}",
            )
            _require(torch.equal(left_tensor, right_tensor), f"retained tensor changed: {name}")


def _compare_overlay(bf16: Path, hybrid: Path, bf16_map, selected, overlay_name):
    by_shard = defaultdict(list)
    for name in selected:
        by_shard[bf16_map[name]].append(name)
    overlay_path = hybrid / overlay_name
    with safe_open(overlay_path, framework="pt", device="cpu") as overlay:
        for shard, names in sorted(by_shard.items()):
            with safe_open(bf16 / shard, framework="pt", device="cpu") as source:
                for name in sorted(names):
                    expected = source.get_tensor(name)
                    observed = overlay.get_tensor(name)
                    _require(expected.dtype == torch.bfloat16, "selected BF16 source dtype changed")
                    _require(observed.dtype == torch.bfloat16, "hybrid overlay dtype changed")
                    _require(
                        tuple(expected.shape) == tuple(observed.shape),
                        f"overlay tensor shape changed: {name}",
                    )
                    _require(torch.equal(expected, observed), f"overlay tensor changed: {name}")


def audit_hybrid_checkpoint(
    bf16: Path,
    fp8: Path,
    hybrid: Path,
    *,
    compare_retained_tensors: bool = True,
):
    bf16, fp8, hybrid = (Path(item).resolve() for item in (bf16, fp8, hybrid))
    _require(all(path.is_dir() for path in (bf16, fp8, hybrid)), "model directory missing")
    for directory in (bf16, fp8, hybrid):
        _require(not directory.is_symlink(), f"model directory may not be a symlink: {directory}")

    provenance_path = hybrid / builder.PROVENANCE_NAME
    provenance = _load_json(provenance_path)
    _require(provenance.get("schema") == builder.SCHEMA, "hybrid provenance schema changed")
    without_self = {
        key: value
        for key, value in provenance.items()
        if key != "content_sha256_before_self_field"
    }
    _require(
        provenance.get("content_sha256_before_self_field")
        == builder.canonical_sha256(without_self),
        "hybrid provenance content identity changed",
    )

    bf16_config = bf16 / "config.json"
    bf16_index_path = bf16 / "model.safetensors.index.json"
    fp8_config = fp8 / "config.json"
    fp8_index_path = fp8 / "model.safetensors.index.json"
    hybrid_config_path = hybrid / "config.json"
    hybrid_index_path = hybrid / "model.safetensors.index.json"
    _require(
        provenance["bf16_source"] == {
            "path": str(bf16),
            "config_sha256": builder.file_sha256(bf16_config),
            "index_sha256": builder.file_sha256(bf16_index_path),
        },
        "BF16 source binding changed",
    )
    _require(
        provenance["fp8_source"] == {
            "path": str(fp8),
            "config_sha256": builder.file_sha256(fp8_config),
            "index_sha256": builder.file_sha256(fp8_index_path),
        },
        "FP8 source binding changed",
    )

    selected = list(provenance["layer_plan"]["units"])
    selected_set = set(selected)
    _require(len(selected) == len(selected_set) == 35, "selected unit surface changed")
    _require(provenance["selected_unit_count"] == len(selected), "selected count changed")
    overlay_name = provenance["overlay_file"]
    overlay_path = hybrid / overlay_name
    _require(
        builder.file_sha256(overlay_path) == provenance["overlay_file_sha256"],
        "overlay file identity changed",
    )
    _require(_keys(overlay_path) == selected_set, "overlay physical keys changed")

    bf16_index = _load_json(bf16_index_path)
    fp8_index = _load_json(fp8_index_path)
    hybrid_index = _load_json(hybrid_index_path)
    bf16_map = bf16_index["weight_map"]
    fp8_map = fp8_index["weight_map"]
    hybrid_map = hybrid_index["weight_map"]
    scales = {
        name + "_scale_inv"
        for name in selected
        if name + "_scale_inv" in fp8_map
    }
    removed = selected_set | scales
    _require(
        len(scales) == provenance["selected_scale_count_removed"] and len(scales) > 0,
        "scale removal changed",
    )
    _require(set(hybrid_map) == (set(fp8_map) - scales), "hybrid index key set changed")
    for name in selected:
        _require(name in bf16_map and name in fp8_map, f"selected source key missing: {name}")
        _require(hybrid_map[name] == overlay_name, f"selected index target changed: {name}")
    for name in set(fp8_map) - removed:
        _require(hybrid_map[name] == fp8_map[name], f"frozen index target changed: {name}")

    affected = sorted({fp8_map[name] for name in removed})
    _require(affected == provenance["affected_fp8_shards"], "affected shard set changed")
    _require(
        len(affected) == provenance["rewritten_fp8_shard_count"] and len(affected) > 0,
        "rewritten shard count changed",
    )
    retained_tensor_count = 0
    for shard in affected:
        source_names = {name for name, mapped in fp8_map.items() if mapped == shard}
        expected_names = source_names - removed
        _require(_keys(hybrid / shard) == expected_names, f"rewritten shard key set changed: {shard}")
        retained_tensor_count += len(expected_names)
        if compare_retained_tensors:
            _compare_shard_tensors(fp8 / shard, hybrid / shard, expected_names)

    for shard in sorted(set(fp8_map.values()) - set(affected)):
        _require((hybrid / shard).is_file(), f"frozen shard missing: {shard}")
        _require(
            os.path.samefile(fp8 / shard, hybrid / shard),
            f"frozen shard is not the exact hardlink: {shard}",
        )

    excluded = {
        "config.json",
        "model.safetensors.index.json",
        builder.PROVENANCE_NAME,
        *affected,
    }
    transferable = [
        path for path in sorted(fp8.iterdir())
        if path.name != ".cache" and path.name not in excluded
    ]
    _require(all(path.is_file() and not path.is_symlink() for path in transferable), "FP8 source transfer surface changed")
    _require(
        provenance["unaffected_file_transfer_modes"] == {"hardlink": len(transferable)},
        "unaffected transfer count changed",
    )
    _require(
        all(os.path.samefile(path, hybrid / path.name) for path in transferable),
        "unaffected file is not an exact hardlink",
    )

    _compare_overlay(bf16, hybrid, bf16_map, selected, overlay_name)
    selected_elements = 0
    selected_bytes = 0
    with safe_open(overlay_path, framework="pt", device="cpu") as overlay:
        for name in selected:
            tensor = overlay.get_tensor(name)
            selected_elements += tensor.numel()
            selected_bytes += tensor.numel() * tensor.element_size()
    _require(selected_elements == provenance["selected_element_count"], "selected elements changed")
    _require(selected_bytes == provenance["selected_byte_count"], "selected bytes changed")

    config = _load_json(hybrid_config_path)
    advertised = config.get("hybrid_selected_bf16", {})
    _require(advertised.get("selected_unit_count") == len(selected), "hybrid config count changed")
    _require(advertised.get("selected_element_count") == selected_elements, "hybrid config elements changed")
    _require(advertised.get("selected_byte_count") == selected_bytes, "hybrid config bytes changed")
    exclusions = set(config["quantization_config"]["modules_to_not_convert"])
    _require(
        set(provenance["runtime_modules_excluded_from_fp8"]).issubset(exclusions),
        "runtime FP8 exclusions changed",
    )

    result = {
        "schema": AUDIT_SCHEMA,
        "provenance_file_sha256": builder.file_sha256(provenance_path),
        "provenance_content_sha256": provenance["content_sha256_before_self_field"],
        "hybrid_config_sha256": builder.file_sha256(hybrid_config_path),
        "hybrid_index_sha256": builder.file_sha256(hybrid_index_path),
        "overlay_file_sha256": provenance["overlay_file_sha256"],
        "selected_unit_count": len(selected),
        "selected_scale_count_removed": len(scales),
        "selected_element_count": selected_elements,
        "selected_byte_count": selected_bytes,
        "rewritten_fp8_shard_count": len(affected),
        "retained_tensor_count_in_rewritten_shards": retained_tensor_count,
        "unaffected_hardlink_count": len(transferable),
        "all_selected_tensors_exact_bf16": True,
        "all_retained_rewritten_shard_tensors_exact_fp8": bool(compare_retained_tensors),
        "all_unaffected_files_exact_hardlinks": True,
        "contains_dataset_or_evaluation_content": False,
    }
    result["content_sha256_before_self_field"] = builder.canonical_sha256(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, default=DEFAULT_BF16)
    parser.add_argument("--fp8", type=Path, default=DEFAULT_FP8)
    parser.add_argument("--hybrid", type=Path, default=DEFAULT_HYBRID)
    parser.add_argument("--skip-retained-tensor-compare", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_hybrid_checkpoint(
        args.bf16,
        args.fp8,
        args.hybrid,
        compare_retained_tensors=not args.skip_retained_tensor_compare,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return result


if __name__ == "__main__":
    main()
