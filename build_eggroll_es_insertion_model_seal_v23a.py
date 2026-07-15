#!/usr/bin/env python3
"""Seal untouched Qwen3.6 epsilon insertion checkpoints for V23A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V23A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V23A_INSERTION_MODEL_SEAL.json"
).resolve()
BASE_MODEL_V23A = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
MODEL_SPECS_V23A = {
    "base_middle_late": {
        "path": BASE_MODEL_V23A,
        "plan": "base", "epsilon": 0.0,
        "source_layers": [20, 21, 22, 23],
        "target_layers": [20, 21, 22, 23],
        "config_sha256": (
            "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
        ),
        "index_sha256": (
            "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
        ),
    },
    "insert_front_e005": {
        "path": (ROOT / "models/Qwen3.6-35B-A3B-depth-front-e005").resolve(),
        "plan": "front", "epsilon": 0.05,
        "source_layers": [0, 1, 2, 3], "target_layers": [4, 5, 6, 7],
        "config_sha256": (
            "70710d6076ffc2e3e993de84826f3f6b85dc77eb4d3100a38556c4197bfce72e"
        ),
        "index_sha256": (
            "a803261527e0cde26468a6cbe4f07220db2e0b49eb05502e1cc949541a78a7a6"
        ),
        "source_map_sha256": (
            "9daef62282fdb886e2870004289defbeab76dd7edac02ac4ed35d075434cb8fa"
        ),
    },
    "insert_middle_e005": {
        "path": (ROOT / "models/Qwen3.6-35B-A3B-depth-middle-e005").resolve(),
        "plan": "middle", "epsilon": 0.05,
        "source_layers": [16, 17, 18, 19], "target_layers": [20, 21, 22, 23],
        "config_sha256": (
            "70710d6076ffc2e3e993de84826f3f6b85dc77eb4d3100a38556c4197bfce72e"
        ),
        "index_sha256": (
            "914468ff77c4a7312bebbc7bb400108ac3d163b90637d76bd3303c568c0d2523"
        ),
        "source_map_sha256": (
            "cd49f160be7ec1e2a11df5f63a76b8fd8896ba8b576f83258c8cf3d5f5496b81"
        ),
    },
    "insert_back_e005": {
        "path": (ROOT / "models/Qwen3.6-35B-A3B-depth-back-e005").resolve(),
        "plan": "back", "epsilon": 0.05,
        "source_layers": [36, 37, 38, 39], "target_layers": [40, 41, 42, 43],
        "config_sha256": (
            "70710d6076ffc2e3e993de84826f3f6b85dc77eb4d3100a38556c4197bfce72e"
        ),
        "index_sha256": (
            "4b108609384dc5a497590e8c0c1233c4cfe20f3034df15458c7fc010551b5068"
        ),
        "source_map_sha256": (
            "b0291aa943c341bb01dc843a9e2f4b154eb695eef996449f71db74e66a5a749e"
        ),
    },
}
ARM_ORDER_V23A = tuple(MODEL_SPECS_V23A)
INELIGIBLE_PATTERNS_V23A = (
    "gen20", "gen30", "fp32_master", "fp32-master", ".jsonl",
    "eval_qa", "checkpoint", "es_location",
)
LAYER_RE_V23A = re.compile(r"^(model\.language_model\.layers\.)(\d+)(\..+)$")
SCALED_SUFFIXES_V23A = (
    ".linear_attn.out_proj.weight",
    ".self_attn.o_proj.weight",
    ".mlp.experts.down_proj",
    ".mlp.shared_expert.down_proj.weight",
)
STANDARD_NONMODEL_FILES_V23A = {
    ".gitattributes", "LICENSE", "README.md", "chat_template.jinja", "configuration.json",
    "generation_config.json", "merges.txt", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json",
    "vocab.json",
}


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


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _expected_layer_types(count):
    return [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(count)
    ]


def _source_mapping(plan):
    original = list(range(40))
    if plan == "front":
        return original[:4] + list(range(4)) + original[4:]
    if plan == "middle":
        return original[:20] + list(range(16, 20)) + original[20:]
    if plan == "back":
        return original + list(range(36, 40))
    raise ValueError("v23a insertion plan changed")


def _expected_scaled_suffixes(layer_type):
    attention = (
        ".linear_attn.out_proj.weight"
        if layer_type == "linear_attention"
        else ".self_attn.o_proj.weight"
    )
    return {
        attention, ".mlp.experts.down_proj",
        ".mlp.shared_expert.down_proj.weight",
    }


def _file_fingerprints(directory):
    files = sorted(path for path in directory.iterdir() if path.is_file())
    _require(files and not any(path.is_symlink() for path in files),
             "v23a model directory is empty or contains a symlink")
    return {
        path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
        for path in files
    }


def _load_model_metadata(spec):
    directory = spec["path"]
    _require(
        directory.is_dir()
        and not any(
            pattern in str(directory).lower() for pattern in INELIGIBLE_PATTERNS_V23A
        ),
        "v23a model path is ineligible",
    )
    config_path = directory / "config.json"
    index_path = directory / "model.safetensors.index.json"
    _require(
        file_sha256(config_path) == spec["config_sha256"]
        and file_sha256(index_path) == spec["index_sha256"],
        "v23a config or index fingerprint changed",
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    expected_count = 40 if spec["plan"] == "base" else 44
    _require(
        text.get("num_hidden_layers") == expected_count
        and text.get("layer_types") == _expected_layer_types(expected_count)
        and text.get("hidden_size") == 2048
        and text.get("num_attention_heads") == 16
        and text.get("num_key_value_heads") == 2
        and text.get("num_experts") == 256
        and text.get("num_experts_per_tok") == 8
        and text.get("moe_intermediate_size") == 512
        and text.get("shared_expert_intermediate_size") == 512,
        "v23a exact model architecture changed",
    )
    shards = sorted(set(index.get("weight_map", {}).values()))
    expected_shards = [f"model-{index:05d}-of-00026.safetensors" for index in range(1, 27)]
    _require(
        shards == expected_shards
        and all((directory / name).is_file() for name in shards),
        "v23a exact shard set changed",
    )
    for shard in shards:
        with safe_open(directory / shard, framework="pt") as source:
            physical = set(source.keys())
        indexed = {
            name for name, assigned in index["weight_map"].items()
            if assigned == shard
        }
        _require(physical == indexed, "v23a physical shard/index keys changed")
    return config, index


def _validate_candidate_index(base_index, candidate_index, mapping):
    destinations = defaultdict(list)
    for destination, source in enumerate(mapping):
        destinations[source].append(destination)
    expected = {}
    for name, shard in base_index["weight_map"].items():
        match = LAYER_RE_V23A.match(name)
        if not match:
            expected[name] = shard
            continue
        source = int(match.group(2))
        for destination in destinations[source]:
            remapped = f"{match.group(1)}{destination}{match.group(3)}"
            _require(remapped not in expected, "v23a remap generated a duplicate")
            expected[remapped] = shard
    _require(
        candidate_index.get("weight_map") == expected,
        "v23a candidate index is not exact untouched surgery output",
    )


def _load_tensor(directory, index, name):
    shard = index["weight_map"].get(name)
    _require(isinstance(shard, str), "v23a required tensor is unindexed")
    with safe_open(directory / shard, framework="pt") as source:
        return source.get_tensor(name)


def _tensor_copy_and_damping_audit(spec, base_index, candidate_index, config):
    source_layers = spec["source_layers"]
    target_layers = spec["target_layers"]
    records = []
    copied_count = 0
    damped_count = 0
    for source_layer, target_layer in zip(source_layers, target_layers, strict=True):
        prefix = f"model.language_model.layers.{source_layer}."
        source_names = sorted(
            name for name in base_index["weight_map"] if name.startswith(prefix)
        )
        _require(source_names, "v23a source motif tensor set is empty")
        observed_scaled = set()
        for source_name in source_names:
            suffix = source_name[len(prefix) - 1:]
            target_name = f"model.language_model.layers.{target_layer}{suffix}"
            source_tensor = _load_tensor(BASE_MODEL_V23A, base_index, source_name)
            target_tensor = _load_tensor(spec["path"], candidate_index, target_name)
            _require(
                source_tensor.shape == target_tensor.shape
                and source_tensor.dtype == target_tensor.dtype,
                "v23a inserted tensor shape or dtype changed",
            )
            matched = [item for item in SCALED_SUFFIXES_V23A if target_name.endswith(item)]
            if matched:
                _require(len(matched) == 1, "v23a damping suffix became ambiguous")
                _require(
                    torch.equal(target_tensor, source_tensor * spec["epsilon"]),
                    "v23a inserted residual output is not exactly epsilon damped",
                )
                observed_scaled.add(matched[0])
                damped_count += 1
            else:
                _require(
                    torch.equal(target_tensor, source_tensor),
                    "v23a untouched inserted tensor differs from source",
                )
                copied_count += 1
        expected_scaled = _expected_scaled_suffixes(
            config["text_config"]["layer_types"][target_layer]
        )
        _require(
            observed_scaled == expected_scaled,
            "v23a attention routed/shared damping coverage changed",
        )
        records.append({
            "source_layer": source_layer, "inserted_layer": target_layer,
            "source_tensor_count": len(source_names),
            "exact_scaled_suffixes": sorted(observed_scaled),
        })
    _require(
        copied_count == 57 and damped_count == 12,
        "v23a exact inserted tensor audit counts changed",
    )
    return {
        "inserted_tensor_count": copied_count + damped_count,
        "exact_undamped_copy_count": copied_count,
        "exact_epsilon_damped_output_count": damped_count,
        "attention_routed_and_shared_outputs_complete": True,
        "per_layer": records,
        "audit_sha256": canonical_sha256(records),
    }


def build_model_seal_v23a():
    loaded = {
        arm: _load_model_metadata(spec)
        for arm, spec in MODEL_SPECS_V23A.items()
    }
    base_config, base_index = loaded["base_middle_late"]
    base_files = _file_fingerprints(BASE_MODEL_V23A)
    expected_base_names = (
        STANDARD_NONMODEL_FILES_V23A
        | {"config.json", "model.safetensors.index.json"}
        | {f"model-{index:05d}-of-00026.safetensors" for index in range(1, 27)}
    )
    _require(set(base_files) == expected_base_names, "v23a base model file set changed")
    arms = {}
    for arm, spec in MODEL_SPECS_V23A.items():
        config, index = loaded[arm]
        files = base_files if arm == "base_middle_late" else _file_fingerprints(spec["path"])
        if arm == "base_middle_late":
            tensor_audit = {
                "inserted_tensor_count": 0,
                "exact_undamped_copy_count": 0,
                "exact_epsilon_damped_output_count": 0,
                "attention_routed_and_shared_outputs_complete": True,
                "per_layer": [], "audit_sha256": canonical_sha256([]),
            }
            provenance = None
        else:
            _require(
                set(files) == expected_base_names | {"layer_source_map.json"}
                and all(
                    files[name]["sha256"] == base_files[name]["sha256"]
                    for name in STANDARD_NONMODEL_FILES_V23A
                )
                and not any(
                    pattern in name.lower()
                    for name in files for pattern in INELIGIBLE_PATTERNS_V23A
                ),
                "v23a inserted directory contains non-surgery or ineligible files",
            )
            source_map_path = spec["path"] / "layer_source_map.json"
            _require(
                file_sha256(source_map_path) == spec["source_map_sha256"],
                "v23a insertion provenance bytes changed",
            )
            provenance = json.loads(source_map_path.read_text(encoding="utf-8"))
            mapping = _source_mapping(spec["plan"])
            _require(
                provenance == {
                    "schema": "qwen36-motif-insertion-v1",
                    "source": str(BASE_MODEL_V23A),
                    "source_config_sha256": MODEL_SPECS_V23A[
                        "base_middle_late"
                    ]["config_sha256"],
                    "source_index_sha256": MODEL_SPECS_V23A[
                        "base_middle_late"
                    ]["index_sha256"],
                    "plan": spec["plan"], "epsilon": 0.05,
                    "num_hidden_layers": 44,
                    "destination_to_source_layer": mapping,
                    "inserted_destination_layers": spec["target_layers"],
                    "scaled_suffixes": list(SCALED_SUFFIXES_V23A),
                    "inserted_tensors": 69,
                    "inserted_tensor_bytes": 6_728_697_984,
                },
                "v23a exact insertion provenance changed",
            )
            _validate_candidate_index(base_index, index, mapping)
            tensor_audit = _tensor_copy_and_damping_audit(
                spec, base_index, index, config
            )
        shards = {
            name: files[name]
            for name in sorted(files) if name.endswith(".safetensors")
        }
        arms[arm] = {
            "path": str(spec["path"]), "plan": spec["plan"],
            "epsilon": spec["epsilon"], "source_layers": spec["source_layers"],
            "target_layers": spec["target_layers"],
            "num_hidden_layers": config["text_config"]["num_hidden_layers"],
            "config_sha256": spec["config_sha256"],
            "index_sha256": spec["index_sha256"],
            "source_map_sha256": spec.get("source_map_sha256"),
            "shard_count": len(shards), "shards": shards,
            "all_files_fingerprint_sha256": canonical_sha256(files),
            "weight_map_sha256": canonical_sha256(index["weight_map"]),
            "tensor_audit": tensor_audit,
            "untouched_surgery_output_eligible": True,
            "old_training_or_probe_artifact_used": False,
            "gen20_or_gen30_master_used": False,
        }
    value = {
        "schema": "eggroll-es-insertion-model-seal-v23a",
        "arm_order": list(ARM_ORDER_V23A),
        "source_model": {
            "path": str(BASE_MODEL_V23A),
            "config_sha256": MODEL_SPECS_V23A["base_middle_late"]["config_sha256"],
            "index_sha256": MODEL_SPECS_V23A["base_middle_late"]["index_sha256"],
        },
        "arms": arms,
        "capacity_match": {
            "source_unit_count_per_motif": 35,
            "runtime_selected_parameter_count_per_motif": 23,
            "selected_element_count_per_motif": 142_999_552,
            "selected_byte_count_per_motif_bfloat16": 285_999_104,
            "all_four_motifs_shape_matched": True,
        },
        "eligibility": {
            "only_untouched_base_and_surgery_output_directories": True,
            "old_insert_training_journals_used": False,
            "old_eval_probe_or_report_used": False,
            "gen20_or_gen30_master_used": False,
            "ineligible_patterns": list(INELIGIBLE_PATTERNS_V23A),
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_model_seal_v23a(value)


def validate_model_seal_v23a(value):
    _require(
        isinstance(value, dict)
        and set(value) == {
            "schema", "arm_order", "source_model", "arms", "capacity_match",
            "eligibility", "content_sha256_before_self_field",
        }
        and value.get("schema") == "eggroll-es-insertion-model-seal-v23a"
        and value.get("arm_order") == list(ARM_ORDER_V23A)
        and value.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(value))
        and set(value.get("arms", {})) == set(ARM_ORDER_V23A)
        and all(value["arms"][arm].get("shard_count") == 26 for arm in ARM_ORDER_V23A)
        and all(
            value["arms"][arm].get("untouched_surgery_output_eligible") is True
            and value["arms"][arm].get("old_training_or_probe_artifact_used") is False
            and value["arms"][arm].get("gen20_or_gen30_master_used") is False
            for arm in ARM_ORDER_V23A
        )
        and all(
            value["arms"][arm]["tensor_audit"].get(
                "attention_routed_and_shared_outputs_complete"
            ) is True
            for arm in ARM_ORDER_V23A
        )
        and value.get("capacity_match") == {
            "source_unit_count_per_motif": 35,
            "runtime_selected_parameter_count_per_motif": 23,
            "selected_element_count_per_motif": 142_999_552,
            "selected_byte_count_per_motif_bfloat16": 285_999_104,
            "all_four_motifs_shape_matched": True,
        }
        and all(
            value.get("eligibility", {}).get(key) is False for key in (
                "old_insert_training_journals_used", "old_eval_probe_or_report_used",
                "gen20_or_gen30_master_used",
            )
        ),
        "v23a insertion model seal changed",
    )
    return value


def _exclusive_write(path, value):
    path = Path(path).resolve()
    if path != OUTPUT_PATH_V23A:
        raise ValueError("v23a model seal output path changed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v23a model seal already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V23A))
    args = parser.parse_args(argv)
    value = build_model_seal_v23a()
    _exclusive_write(Path(args.output), value)
    result = {
        "schema": "eggroll-es-insertion-model-seal-build-v23a",
        "path": str(OUTPUT_PATH_V23A),
        "file_sha256": file_sha256(OUTPUT_PATH_V23A),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
