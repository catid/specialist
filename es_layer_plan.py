#!/usr/bin/env python3
"""Generate validated Qwen3.6-35B-A3B layer-location ES target manifests.

The model repeats a four-layer motif: three linear-attention layers followed
by one full-attention layer. Plans therefore select whole motifs so location
comparisons do not confound front/back position with attention type.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen3.6-35B-A3B"

PLANS = {name: name for name in (
    "front", "back", "front_back", "middle_matched",
    "front_back_wide", "all", "inserted", "inserted_edges")}

LINEAR_ATTENTION = (
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.out_proj.weight",
)
FULL_ATTENTION = tuple(f"self_attn.{name}_proj.weight"
                       for name in ("q", "k", "v", "o"))
ROUTED = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
ROUTER = ("mlp.gate.weight",)
SHARED = tuple(f"mlp.shared_expert.{name}.weight"
               for name in ("gate_proj", "up_proj", "down_proj"))
GROUPS = {"attention", "shared", "router", "routed", "dense", "all"}


def layer_suffixes(layer_type: str):
    attention = FULL_ATTENTION if layer_type == "full_attention" else LINEAR_ATTENTION
    return {
        "attention": attention,
        "shared": SHARED,
        "router": ROUTER,
        "routed": ROUTED,
        "dense": (*attention, *SHARED, *ROUTER),
        "all": (*attention, *SHARED, *ROUTER, *ROUTED),
    }


def selected_unit_names(layers, layer_types, groups):
    names = []
    expanded_groups = set(groups)
    if "all" in expanded_groups:
        expanded_groups = {"all"}
    elif "dense" in expanded_groups:
        expanded_groups.remove("dense")
        expanded_groups.update(("attention", "shared", "router"))
    for layer in layers:
        suffix_map = layer_suffixes(layer_types[layer])
        suffixes = set()
        for group in expanded_groups:
            suffixes.update(suffix_map[group])
        names.extend(f"model.language_model.layers.{layer}.{suffix}"
                     for suffix in sorted(suffixes))
    return sorted(names)


def target_regex(unit_names):
    if not unit_names:
        raise ValueError("target plan contains no units")
    return "^(?:" + "|".join(re.escape(name) for name in unit_names) + ")$"


def load_config(model_path: Path):
    config_path = model_path / "config.json"
    config = json.loads(config_path.read_text())
    text = config["text_config"]
    count = int(text["num_hidden_layers"])
    types = text.get("layer_types") or [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(count)
    ]
    if len(types) != count:
        raise ValueError("config layer_types length does not match num_hidden_layers")
    if any((kind == "full_attention") != ((index + 1) % 4 == 0)
           for index, kind in enumerate(types)):
        raise ValueError("model does not have the expected [linear,linear,linear,full] motif")
    return config_path, count, types


def resolve_plan(plan: str, count: int):
    if count % 4:
        raise ValueError("whole-motif plans require a layer count divisible by four")
    front = tuple(range(4))
    back = tuple(range(count - 4, count))
    middle_start = ((count // 2 - 4) // 4) * 4
    plans = {
        "front": front,
        "back": back,
        "front_back": (*front, *back),
        "middle_matched": tuple(range(middle_start, middle_start + 8)),
        "front_back_wide": (*range(8), *range(count - 8, count)),
        "all": tuple(range(count)),
    }
    return plans[plan]


def plan_manifest(model_path: Path, plan: str, groups, custom_layers=None):
    config_path, count, layer_types = load_config(model_path)
    if custom_layers is not None:
        layers = tuple(custom_layers)
    elif plan in {"inserted", "inserted_edges"}:
        provenance_path = model_path / "layer_source_map.json"
        if not provenance_path.exists():
            raise ValueError(f"{plan} requires layer_source_map.json")
        inserted = tuple(json.loads(provenance_path.read_text())[
            "inserted_destination_layers"])
        layers = inserted
        if plan == "inserted_edges":
            layers = tuple(dict.fromkeys((*range(4), *inserted,
                                          *range(count - 4, count))))
    else:
        layers = resolve_plan(plan, count)
    if not layers or min(layers) < 0 or max(layers) >= count:
        raise ValueError(f"layers must be within [0, {count - 1}]")
    if len(set(layers)) != len(layers):
        raise ValueError("layer plan contains duplicates")
    unknown = set(groups) - GROUPS
    if unknown:
        raise ValueError(f"unknown unit groups: {sorted(unknown)}")
    units = selected_unit_names(layers, layer_types, groups)
    description = {
        "schema": "qwen36-es-layer-plan-v1",
        "model_config": str(config_path.resolve()),
        "model_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "plan": plan if custom_layers is None else "custom",
        "layers": list(layers),
        "layer_types": {str(layer): layer_types[layer] for layer in layers},
        "groups": sorted(groups),
        "num_units": len(units),
        "units": units,
        "include_regex": target_regex(units),
    }
    canonical = json.dumps(description, sort_keys=True, separators=(",", ":"))
    description["plan_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return description


def parse_layers(value):
    layers = []
    for part in value.split(","):
        if "-" in part:
            start, stop = (int(number) for number in part.split("-", 1))
            layers.extend(range(start, stop + 1))
        else:
            layers.append(int(part))
    return layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--plan", choices=sorted(PLANS), default="front_back")
    parser.add_argument("--layers", type=parse_layers,
                        help="custom comma/range list, e.g. 0-3,36-39")
    parser.add_argument("--groups", nargs="+", choices=sorted(GROUPS),
                        default=["dense"])
    parser.add_argument("--format", choices=("json", "regex"), default="json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = plan_manifest(args.model, args.plan, args.groups, args.layers)
    rendered = (manifest["include_regex"] if args.format == "regex"
                else json.dumps(manifest, indent=2, sort_keys=True))
    if args.output:
        args.output.write_text(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
