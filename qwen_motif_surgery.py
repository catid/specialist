#!/usr/bin/env python3
"""Insert a copied four-layer Qwen3.6 motif into a sharded HF checkpoint.

Only complete [linear, linear, linear, full-attention] motifs are inserted, so
the model's modulo-four execution pattern remains valid. Inserted residual
branches are damped by scaling attention output projections and routed/shared
MLP down projections. The operation is not exactly function preserving, but a
small epsilon makes it a controlled pre-screen before continued training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCHEMA = "qwen36-motif-insertion-v1"
LAYER_RE = re.compile(r"^(model\.language_model\.layers\.)(\d+)(\..+)$")
PLANS = {
    "front": {"insert_after": 3, "source_start": 0},
    "middle": {"insert_after": 19, "source_start": 16},
    "back": {"insert_after": 39, "source_start": 36},
}
SCALED_SUFFIXES = (
    ".linear_attn.out_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.experts.down_proj",
    ".mlp.shared_expert.down_proj.weight",
)


def expected_scaled_suffixes(layer_type: str) -> frozenset[str]:
    """Return every residual-output tensor that must be damped in one layer."""
    if layer_type == "linear_attention":
        attention_suffix = ".linear_attn.out_proj.weight"
    elif layer_type == "full_attention":
        attention_suffix = ".self_attn.o_proj.weight"
    else:
        raise ValueError(f"unsupported inserted layer type: {layer_type!r}")
    return frozenset({
        attention_suffix,
        ".mlp.experts.down_proj",
        ".mlp.shared_expert.down_proj.weight",
    })


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_map(num_layers: int, plan: str, width: int = 4):
    spec = PLANS[plan]
    insert_after = spec["insert_after"]
    source_start = spec["source_start"]
    if insert_after >= num_layers or source_start + width > num_layers:
        raise ValueError("insertion plan is outside the checkpoint layer range")
    original = list(range(num_layers))
    mapping = (original[:insert_after + 1] +
               list(range(source_start, source_start + width)) +
               original[insert_after + 1:])
    inserted = set(range(insert_after + 1, insert_after + 1 + width))
    return mapping, inserted


def remapped_names(name: str, destinations_by_source):
    match = LAYER_RE.match(name)
    if not match:
        return [(name, False)]
    source = int(match.group(2))
    return [(f"{match.group(1)}{destination}{match.group(3)}", destination)
            for destination in destinations_by_source[source]]


def validate_config(config, mapping):
    text = config.get("text_config", {})
    count = int(text.get("num_hidden_layers", 0))
    types = text.get("layer_types", [])
    if count != 40 or len(types) != count:
        raise ValueError("expected a 40-layer Qwen3.6 text configuration")
    expected = ["full_attention" if (index + 1) % 4 == 0
                else "linear_attention" for index in range(count)]
    if types != expected:
        raise ValueError("layer_types do not match the expected four-layer motif")
    expanded = [types[source] for source in mapping]
    expected_expanded = ["full_attention" if (index + 1) % 4 == 0
                         else "linear_attention" for index in range(len(mapping))]
    if expanded != expected_expanded:
        raise ValueError("insertion would break the modulo-four layer pattern")


def build_checkpoint(src: Path, dst: Path, plan: str, epsilon: float):
    if dst.exists():
        raise FileExistsError(f"destination already exists: {dst}")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be in [0, 1]")
    config_path = src / "config.json"
    index_path = src / "model.safetensors.index.json"
    config = json.loads(config_path.read_text())
    index = json.loads(index_path.read_text())
    num_layers = int(config["text_config"]["num_hidden_layers"])
    mapping, inserted_destinations = source_map(num_layers, plan)
    validate_config(config, mapping)

    destinations_by_source = defaultdict(list)
    for destination, source in enumerate(mapping):
        destinations_by_source[source].append(destination)

    new_weight_map = {}
    for name, shard in index["weight_map"].items():
        for remapped, _destination in remapped_names(name, destinations_by_source):
            if remapped in new_weight_map:
                raise ValueError(f"remapping generated duplicate key {remapped}")
            new_weight_map[remapped] = shard

    temporary = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"temporary destination already exists: {temporary}")
    temporary.mkdir(parents=True)
    duplicated_bytes = 0
    inserted_tensors = 0
    scaled_suffixes_by_destination = defaultdict(set)
    try:
        shards = sorted(set(index["weight_map"].values()))
        for shard_index, shard in enumerate(shards, 1):
            output_tensors = {}
            with safe_open(src / shard, framework="pt") as source:
                metadata = source.metadata() or {"format": "pt"}
                for name in source.keys():
                    tensor = source.get_tensor(name)
                    remaps = remapped_names(name, destinations_by_source)
                    for remapped, destination in remaps:
                        output = tensor.clone()
                        inserted = destination in inserted_destinations
                        if inserted and remapped.endswith(SCALED_SUFFIXES):
                            if not output.is_floating_point():
                                raise TypeError(f"cannot scale non-floating tensor {name}")
                            output.mul_(epsilon)
                            matched = [
                                suffix for suffix in SCALED_SUFFIXES
                                if remapped.endswith(suffix)
                            ]
                            if len(matched) != 1:
                                raise ValueError(
                                    f"ambiguous residual-output tensor {remapped}")
                            scaled_suffixes_by_destination[destination].add(
                                matched[0])
                        output_tensors[remapped] = output.contiguous()
                        if inserted:
                            inserted_tensors += 1
                            duplicated_bytes += output.numel() * output.element_size()
            save_file(output_tensors, temporary / shard, metadata=metadata)
            print(f"shard {shard_index}/{len(shards)} written", flush=True)

        expected_scaled_by_destination = {}
        for destination in sorted(inserted_destinations):
            layer_type = config["text_config"]["layer_types"][
                mapping[destination]
            ]
            expected_scaled = expected_scaled_suffixes(layer_type)
            observed_scaled = frozenset(
                scaled_suffixes_by_destination[destination])
            if observed_scaled != expected_scaled:
                raise ValueError(
                    "inserted layer residual-output damping coverage changed; "
                    f"destination={destination}, "
                    f"missing={sorted(expected_scaled - observed_scaled)}, "
                    f"extra={sorted(observed_scaled - expected_scaled)}")
            expected_scaled_by_destination[str(destination)] = sorted(
                expected_scaled)

        for path in src.iterdir():
            if path.name.startswith("model-") and path.suffix == ".safetensors":
                continue
            if path.name in {"config.json", "model.safetensors.index.json"}:
                continue
            if path.is_file():
                shutil.copy2(path, temporary / path.name)

        expanded_config = json.loads(json.dumps(config))
        expanded_config["text_config"]["num_hidden_layers"] = len(mapping)
        expanded_config["text_config"]["layer_types"] = [
            config["text_config"]["layer_types"][source] for source in mapping]
        (temporary / "config.json").write_text(
            json.dumps(expanded_config, indent=2, ensure_ascii=False) + "\n")

        expanded_index = json.loads(json.dumps(index))
        expanded_index["weight_map"] = new_weight_map
        expanded_index.setdefault("metadata", {})["total_size"] = (
            int(index.get("metadata", {}).get("total_size", 0)) + duplicated_bytes)
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(expanded_index, indent=2, sort_keys=True) + "\n")

        provenance = {
            "schema": SCHEMA,
            "source": str(src.resolve()),
            "source_config_sha256": sha256(config_path),
            "source_index_sha256": sha256(index_path),
            "plan": plan,
            "epsilon": epsilon,
            "num_hidden_layers": len(mapping),
            "destination_to_source_layer": mapping,
            "inserted_destination_layers": sorted(inserted_destinations),
            "scaled_suffixes": list(SCALED_SUFFIXES),
            "scaled_suffixes_by_inserted_destination": (
                expected_scaled_by_destination),
            "scaled_inserted_tensors": sum(
                len(value) for value in expected_scaled_by_destination.values()),
            "inserted_tensors": inserted_tensors,
            "inserted_tensor_bytes": duplicated_bytes,
        }
        (temporary / "layer_source_map.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n")

        # Cheap structural verification: every indexed key must be physically
        # present in the output shard to which it is assigned.
        keys_by_shard = defaultdict(set)
        for name, shard in new_weight_map.items():
            keys_by_shard[shard].add(name)
        for shard, expected_keys in keys_by_shard.items():
            with safe_open(temporary / shard, framework="pt") as output:
                actual = set(output.keys())
            if actual != expected_keys:
                missing = sorted(expected_keys - actual)[:3]
                extra = sorted(actual - expected_keys)[:3]
                raise ValueError(
                    f"output shard {shard} failed key verification; "
                    f"missing={missing}, extra={extra}")
        temporary.rename(dst)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--plan", choices=sorted(PLANS), required=True)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        config = json.loads((args.src / "config.json").read_text())
        mapping, inserted = source_map(
            int(config["text_config"]["num_hidden_layers"]), args.plan)
        validate_config(config, mapping)
        print(json.dumps({"plan": args.plan, "epsilon": args.epsilon,
                          "destination_to_source_layer": mapping,
                          "inserted_destination_layers": sorted(inserted)}, indent=2))
        return
    result = build_checkpoint(args.src, args.dst, args.plan, args.epsilon)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
