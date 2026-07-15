#!/usr/bin/env python3
"""Aggregate-only exact post-build audit for the V26 Qwen3.6 hybrid."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

import build_qwen36_fp8_routed_experts_bf16_backbone_v26 as builder


AUDIT_SCHEMA = "qwen36-fp8-routed-experts-bf16-backbone-audit-v26"
DEFAULT_BF16 = builder.DEFAULT_BF16
DEFAULT_FP8 = builder.DEFAULT_FP8
DEFAULT_HYBRID = builder.DEFAULT_OUTPUT


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _without_self(value):
    return {
        key: item
        for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _compare_exact_tensors(source_model, hybrid, source_map, hybrid_map, names, label):
    pairs = defaultdict(list)
    for name in names:
        pairs[(source_map[name], hybrid_map[name])].append(name)
    compared = 0
    for (source_shard, output_shard), shard_names in sorted(pairs.items()):
        with safe_open(
            source_model / source_shard, framework="pt", device="cpu"
        ) as source, safe_open(
            hybrid / output_shard, framework="pt", device="cpu"
        ) as output:
            for name in sorted(shard_names):
                expected = source.get_tensor(name)
                observed = output.get_tensor(name)
                _require(expected.dtype == observed.dtype, f"{label} tensor dtype changed")
                _require(
                    tuple(expected.shape) == tuple(observed.shape),
                    f"{label} tensor shape changed",
                )
                _require(torch.equal(expected, observed), f"{label} tensor content changed")
                compared += 1
    return compared


def _transferable_auxiliary_files(fp8, fp8_weight_shards):
    result = []
    for path in sorted(fp8.iterdir()):
        if path.name == ".cache" and path.is_dir() and not path.is_symlink():
            continue
        _require(
            path.is_file() and not path.is_symlink(),
            "FP8 source auxiliary surface changed",
        )
        if path.name in fp8_weight_shards | {
            "config.json", "model.safetensors.index.json", builder.PROVENANCE_NAME,
        }:
            continue
        result.append(path)
    return result


def audit_hybrid_checkpoint_v26(bf16: Path, fp8: Path, hybrid: Path):
    bf16, fp8, hybrid = (Path(item).resolve() for item in (bf16, fp8, hybrid))
    _require(
        all(path.is_dir() and not path.is_symlink() for path in (bf16, fp8, hybrid)),
        "model directory is missing or symlinked",
    )
    provenance_path = hybrid / builder.PROVENANCE_NAME
    provenance = _load_json(provenance_path)
    _require(provenance.get("schema") == builder.SCHEMA, "V26 provenance schema changed")
    _require(
        provenance.get("content_sha256_before_self_field")
        == builder.canonical_sha256(_without_self(provenance)),
        "V26 provenance content identity changed",
    )

    contract = builder._source_contract(
        bf16, fp8, include_weight_file_digests=True
    )
    summary = contract["summary"]
    _require(
        provenance.get("source_identity") == contract["source_identity"],
        "source shard file identity changed",
    )
    expected_tensor_contract = {
        key: summary[key]
        for key in (
            "bf16_backbone", "bf16_packed_routed_experts",
            "fp8_routed_weights", "fp8_routed_scales",
            "fp8_removed_backbone_scales",
        )
    }
    _require(
        provenance.get("tensor_contract") == expected_tensor_contract,
        "source tensor representation contract changed",
    )

    config_path = hybrid / "config.json"
    index_path = hybrid / "model.safetensors.index.json"
    _require(
        builder.file_sha256(config_path) == provenance.get("hybrid_config_sha256"),
        "hybrid config identity changed",
    )
    _require(
        builder.file_sha256(index_path) == provenance.get("hybrid_index_sha256"),
        "hybrid index identity changed",
    )
    hybrid_config = _load_json(config_path)
    hybrid_index = _load_json(index_path)
    hybrid_inventory = builder._inventory(hybrid, hybrid_index)
    hybrid_map = hybrid_inventory["weight_map"]
    _require(
        set(hybrid_map) == contract["target_keys"]
        and len(hybrid_map) == summary["target_key_count"]
        and builder.canonical_sha256(sorted(hybrid_map)) == summary["target_key_sha256"],
        "hybrid target key set changed",
    )
    _require(
        hybrid_index.get("metadata", {}).get("total_size")
        == summary["target_byte_count"],
        "hybrid index total size changed",
    )

    bf16_map = contract["bf16_inventory"]["weight_map"]
    fp8_map = contract["fp8_inventory"]["weight_map"]
    backbone_by_source = defaultdict(set)
    for name in contract["bf16_backbone"]:
        backbone_by_source[bf16_map[name]].add(name)
    backbone_sources = sorted(backbone_by_source)
    backbone_output_names = {
        shard: builder.BACKBONE_SHARD_TEMPLATE.format(
            index=index + 1, total=len(backbone_sources)
        )
        for index, shard in enumerate(backbone_sources)
    }
    for name in contract["bf16_backbone"]:
        _require(
            hybrid_map[name] == backbone_output_names[bf16_map[name]],
            "BF16 backbone index mapping changed",
        )
    for name in contract["fp8_routed"]:
        _require(
            hybrid_map[name] == fp8_map[name],
            "FP8 routed expert index mapping changed",
        )

    output_shards = builder._weight_shard_file_manifest(
        hybrid, hybrid_map.values()
    )
    _require(
        output_shards == provenance.get("output_weight_shards"),
        "output shard file identity changed",
    )

    source_exclusions = contract["fp8_config"]["quantization_config"].get(
        "modules_to_not_convert", []
    )
    runtime_exclusions = builder._backbone_runtime_exclusions(
        contract["bf16_backbone"]
    )
    expected_exclusions = sorted(set(source_exclusions) | set(runtime_exclusions))
    observed_exclusions = hybrid_config.get("quantization_config", {}).get(
        "modules_to_not_convert"
    )
    _require(
        observed_exclusions == expected_exclusions
        and not any(builder._is_routed(name) for name in observed_exclusions),
        "non-routed quantization exclusion surface changed",
    )
    exclusion_summary = {
        "module_count": len(expected_exclusions),
        "module_sha256": builder.canonical_sha256(expected_exclusions),
        "all_backbone_runtime_modules_excluded": True,
        "no_routed_expert_module_excluded": True,
    }
    _require(
        provenance.get("runtime_exclusions") == exclusion_summary,
        "runtime exclusion commitment changed",
    )
    advertised = hybrid_config.get(
        "hybrid_routed_experts_fp8_bf16_backbone_v26", {}
    )
    expected_advertised = {
        "schema": builder.SCHEMA,
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
    _require(advertised == expected_advertised, "hybrid config declaration changed")

    backbone_compared = _compare_exact_tensors(
        bf16, hybrid, bf16_map, hybrid_map, contract["bf16_backbone"], "BF16 backbone"
    )
    routed_compared = _compare_exact_tensors(
        fp8, hybrid, fp8_map, hybrid_map, contract["fp8_routed"], "FP8 routed expert"
    )
    _require(
        backbone_compared == summary["bf16_backbone"]["key_count"]
        and routed_compared == (
            summary["fp8_routed_weights"]["key_count"]
            + summary["fp8_routed_scales"]["key_count"]
        ),
        "exact tensor comparison coverage changed",
    )

    fp8_weight_shards = set(fp8_map.values())
    transferable = _transferable_auxiliary_files(fp8, fp8_weight_shards)
    mode_counts = provenance.get("unaffected_file_transfer_modes")
    _require(
        isinstance(mode_counts, dict)
        and sum(mode_counts.values()) == len(transferable),
        "auxiliary transfer accounting changed",
    )
    exact_hardlinks = 0
    exact_copies = 0
    for source in transferable:
        destination = hybrid / source.name
        _require(
            destination.is_file() and not destination.is_symlink(),
            "hybrid auxiliary file missing or symlinked",
        )
        if os.path.samefile(source, destination):
            exact_hardlinks += 1
        else:
            _require(
                builder.file_sha256(source) == builder.file_sha256(destination),
                "copied auxiliary file content changed",
            )
            exact_copies += 1
    observed_modes = {
        key: value for key, value in {
            "copy": exact_copies, "hardlink": exact_hardlinks,
        }.items() if value
    }
    _require(observed_modes == mode_counts, "auxiliary transfer mode changed")
    expected_files = {
        "config.json", "model.safetensors.index.json", builder.PROVENANCE_NAME,
        *set(hybrid_map.values()), *{path.name for path in transferable},
    }
    _require(
        {path.name for path in hybrid.iterdir()} == expected_files
        and all(path.is_file() and not path.is_symlink() for path in hybrid.iterdir()),
        "hybrid output file surface changed",
    )

    result = {
        "schema": AUDIT_SCHEMA,
        "provenance_file_sha256": builder.file_sha256(provenance_path),
        "provenance_content_sha256": provenance[
            "content_sha256_before_self_field"
        ],
        "hybrid_config_sha256": provenance["hybrid_config_sha256"],
        "hybrid_index_sha256": provenance["hybrid_index_sha256"],
        "source_weight_shard_manifest_sha256": builder.canonical_sha256({
            label: contract["source_identity"][label]["weight_shards"]
            for label in ("bf16", "fp8")
        }),
        "output_weight_shard_manifest_sha256": output_shards["manifest_sha256"],
        "target_key_count": summary["target_key_count"],
        "target_key_sha256": summary["target_key_sha256"],
        "target_element_count": summary["target_element_count"],
        "target_byte_count": summary["target_byte_count"],
        "bf16_backbone_tensor_count": backbone_compared,
        "fp8_routed_weight_count": summary["fp8_routed_weights"]["key_count"],
        "fp8_routed_scale_count": summary["fp8_routed_scales"]["key_count"],
        "removed_non_routed_scale_count": summary[
            "fp8_removed_backbone_scales"
        ]["key_count"],
        "output_weight_shard_count": output_shards["file_count"],
        "auxiliary_hardlink_count": exact_hardlinks,
        "auxiliary_copy_count": exact_copies,
        "all_backbone_tensors_exact_bf16": True,
        "all_routed_expert_tensors_and_scales_exact_fp8": True,
        "all_output_shard_file_digests_exact": True,
        "all_source_shard_file_digests_exact": True,
        "all_auxiliary_files_exact_hardlinks": exact_copies == 0,
        "non_routed_fp8_scales_present": False,
        "contains_dataset_or_evaluation_content": False,
    }
    result["content_sha256_before_self_field"] = builder.canonical_sha256(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16", type=Path, default=DEFAULT_BF16)
    parser.add_argument("--fp8", type=Path, default=DEFAULT_FP8)
    parser.add_argument("--hybrid", type=Path, default=DEFAULT_HYBRID)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_hybrid_checkpoint_v26(args.bf16, args.fp8, args.hybrid)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return result


if __name__ == "__main__":
    main()
