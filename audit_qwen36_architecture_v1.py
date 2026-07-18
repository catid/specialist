#!/usr/bin/env python3
"""Build the data-free Qwen3.6 architecture and runtime contract.

This audit reads only local checkpoint configuration, safetensor headers,
installed package source, Git metadata, and hardware metadata.  Model objects
are constructed exclusively on the PyTorch ``meta`` device with CUDA hidden;
weights, optimizers, datasets, and training code are never loaded.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from importlib import metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
OUTPUT = ROOT / "training_protocol/qwen36_architecture_contract_v1.json"
SCHEMA = "qwen36-low-regression-architecture-contract-v1"

# The audit process must be unable to initialize or allocate on CUDA.  NVML's
# nvidia-smi subprocess remains able to inventory the physical host.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

EXPECTED_GPU = {
    "count": 4,
    "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    "memory_total_mib": 97_887,
    "compute_capability": "12.0",
}

PACKAGE_PINS = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "unsloth",
    "unsloth_zoo",
    "triton",
    "flash-attn",
    "fla-core",
    "flash-linear-attention",
    "causal-conv1d",
    "safetensors",
    "huggingface_hub",
    "datasets",
    "tokenizers",
    "numpy",
    "ninja",
)

SAFETENSOR_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _command(*args: str) -> str:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _relative_source(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _module_source(module_or_object: Any) -> dict[str, Any]:
    path = Path(inspect.getsourcefile(module_or_object) or "").resolve()
    _require(path.is_file(), f"missing installed source for {module_or_object!r}")
    return {
        "path": _relative_source(path),
        "sha256": _file_sha256(path),
    }


def _package_pin(name: str) -> dict[str, Any]:
    try:
        distribution = metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return {
            "installed": False,
            "version": None,
            "vcs_commit": None,
            "direct_url": None,
        }

    direct_url = None
    try:
        raw = distribution.read_text("direct_url.json")
        if raw:
            direct_url = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        direct_url = {"unreadable": True}
    vcs_commit = None
    if isinstance(direct_url, dict):
        vcs_commit = (direct_url.get("vcs_info") or {}).get("commit_id")
    return {
        "installed": True,
        "canonical_name": distribution.metadata.get("Name"),
        "version": distribution.version,
        "vcs_commit": vcs_commit,
        "direct_url": direct_url,
    }


def _hardware_contract() -> dict[str, Any]:
    query = (
        "nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version "
        "--format=csv,noheader,nounits"
    )
    raw = _command(
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    )
    rows = []
    for parsed in csv.reader(raw.splitlines(), skipinitialspace=True):
        _require(len(parsed) == 5, f"unexpected nvidia-smi row: {parsed!r}")
        rows.append(
            {
                "index": int(parsed[0]),
                "name": parsed[1],
                "memory_total_mib": int(parsed[2]),
                "compute_capability": parsed[3],
                "driver_version": parsed[4],
            }
        )
    _require(len(rows) == EXPECTED_GPU["count"], "physical GPU count changed")
    _require([item["index"] for item in rows] == [0, 1, 2, 3], "GPU indexes changed")
    for item in rows:
        _require(item["name"] == EXPECTED_GPU["name"], "GPU model changed")
        _require(
            item["memory_total_mib"] == EXPECTED_GPU["memory_total_mib"],
            "GPU memory changed",
        )
        _require(
            item["compute_capability"] == EXPECTED_GPU["compute_capability"],
            "GPU compute capability changed",
        )

    version_lines = _command("nvidia-smi", "--version").splitlines()
    versions = {}
    for line in version_lines:
        if ":" in line:
            key, value = line.split(":", 1)
            versions[key.strip().lower().replace(" ", "_")] = value.strip()
    nvcc = _command("nvcc", "--version")
    release = re.search(r"release ([0-9.]+), V([0-9.]+)", nvcc)
    _require(release is not None, "cannot parse nvcc version")
    return {
        "inventory_command": query,
        "expected": EXPECTED_GPU,
        "gpus": rows,
        "nvidia_runtime": versions,
        "nvcc": {
            "release": release.group(1),
            "version": release.group(2),
            "raw_sha256": hashlib.sha256(nvcc.encode("utf-8")).hexdigest(),
        },
    }


def _submodule_pins() -> list[dict[str, Any]]:
    raw = _command("git", "submodule", "status", "--recursive")
    result = []
    for line in raw.splitlines():
        fields = line.split(maxsplit=2)
        _require(len(fields) >= 2, f"invalid submodule status: {line!r}")
        token, path = fields[:2]
        status = token[0] if token[0] in "+-U" else " "
        commit = token[1:] if status != " " else token
        _require(re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "invalid submodule commit")
        result.append(
            {
                "path": path,
                "commit": commit,
                "status": status,
                "description": fields[2] if len(fields) == 3 else None,
            }
        )
    return result


def _read_safetensor_headers(
    index: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], int]:
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict) and weight_map, "checkpoint weight map missing")
    _require(
        all(isinstance(name, str) and name for name in weight_map)
        and all(isinstance(name, str) and name for name in weight_map.values()),
        "checkpoint weight map names must be nonempty strings",
    )
    metadata = index.get("metadata")
    _require(isinstance(metadata, dict), "checkpoint index metadata missing")
    raw_total_size = metadata.get("total_size")
    _require(
        isinstance(raw_total_size, (int, float))
        and not isinstance(raw_total_size, bool)
        and math.isfinite(float(raw_total_size))
        and float(raw_total_size).is_integer()
        and 0 <= raw_total_size <= 2**53,
        "checkpoint total_size is invalid",
    )
    declared_total_size = int(raw_total_size)
    filenames = sorted(set(weight_map.values()))
    tensors: dict[str, dict[str, Any]] = {}
    physical_bytes = 0
    for filename in filenames:
        _require(
            isinstance(filename, str)
            and Path(filename).name == filename
            and filename.endswith(".safetensors"),
            f"unsafe checkpoint shard name: {filename!r}",
        )
        path = MODEL_ROOT / filename
        _require(path.is_file(), f"checkpoint shard missing: {path}")
        physical_bytes += path.stat().st_size
        with path.open("rb") as source:
            raw_length = source.read(8)
            _require(len(raw_length) == 8, f"truncated safetensor: {path}")
            header_length = struct.unpack("<Q", raw_length)[0]
            _require(0 < header_length < (64 << 20), f"unsafe header length: {path}")
            payload = source.read(header_length)
            _require(len(payload) == header_length, f"truncated header: {path}")
        data_length = path.stat().st_size - 8 - header_length
        _require(data_length >= 0, f"negative safetensor data length: {path}")

        def reject_header_duplicates(pairs):
            result = {}
            for key, value in pairs:
                _require(
                    key not in result,
                    f"duplicate safetensor header key in {path}: {key}",
                )
                result[key] = value
            return result

        try:
            header = json.loads(
                payload,
                object_pairs_hook=reject_header_duplicates,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RuntimeError(f"invalid safetensor header JSON: {path}") from error
        _require(isinstance(header, dict), f"invalid safetensor header: {path}")
        intervals = []
        for name, item in header.items():
            if name == "__metadata__":
                _require(isinstance(item, dict), f"invalid safetensor metadata: {path}")
                continue
            _require(name not in tensors, f"duplicate checkpoint tensor: {name}")
            _require(weight_map.get(name) == filename, f"index/header mismatch: {name}")
            _require(isinstance(item, dict), f"invalid tensor metadata: {name}")
            shape = item.get("shape")
            offsets = item.get("data_offsets")
            dtype = item.get("dtype")
            _require(
                isinstance(shape, list)
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    and value >= 0
                    for value in shape
                )
                and isinstance(offsets, list)
                and len(offsets) == 2
                and all(
                    isinstance(value, int)
                    and not isinstance(value, bool)
                    for value in offsets
                )
                and 0 <= offsets[0] <= offsets[1] <= data_length
                and isinstance(dtype, str)
                and dtype in SAFETENSOR_DTYPE_BYTES,
                f"invalid tensor metadata: {name}",
            )
            logical_bytes = offsets[1] - offsets[0]
            expected_bytes = math.prod(shape) * SAFETENSOR_DTYPE_BYTES[dtype]
            _require(
                logical_bytes == expected_bytes,
                f"tensor byte span does not match shape and dtype: {name}",
            )
            intervals.append((offsets[0], offsets[1], name))
            tensors[name] = {
                "shape": shape,
                "dtype": dtype,
                "logical_bytes": logical_bytes,
                "file": filename,
            }
        intervals.sort()
        _require(
            not intervals or intervals[0][0] == 0,
            f"safetensor data region has an unclaimed prefix: {path}",
        )
        for left, right in zip(intervals, intervals[1:]):
            _require(
                left[1] == right[0],
                f"noncontiguous safetensor data spans: {left[2]} and {right[2]}",
            )
        _require(
            not intervals or intervals[-1][1] == data_length,
            f"safetensor data region has unclaimed trailing bytes: {path}",
        )
    _require(set(tensors) == set(weight_map), "checkpoint index/header key set changed")
    _require(
        sum(item["logical_bytes"] for item in tensors.values())
        == declared_total_size,
        "checkpoint index total_size does not match tensor byte spans",
    )
    return tensors, physical_bytes


def _checkpoint_category(name: str) -> str:
    if name.startswith(("model.language_model.", "lm_head.")):
        return "language"
    if name.startswith("model.visual."):
        return "visual"
    if name.startswith("mtp."):
        return "mtp"
    return "other"


def _conversion_row(transform: Any) -> dict[str, Any]:
    return {
        "class": f"{type(transform).__module__}.{type(transform).__name__}",
        "source_patterns": list(transform.source_patterns),
        "target_patterns": list(transform.target_patterns),
        "scope_prefix": transform.scope_prefix,
        "base_model_prefix": transform.base_model_prefix,
    }


def _apply_key_conversions(key: str, transforms: list[Any]) -> str:
    result = key
    for transform in transforms:
        result, _ = transform.rename_source_key(result)
    return result


def _model_and_checkpoint_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    # Imports happen only after CUDA has been hidden at module import time.
    import torch
    import transformers
    import transformers.conversion_mapping as conversion_mapping
    import transformers.models.auto.auto_factory as auto_factory
    import transformers.models.qwen3_5_moe.configuration_qwen3_5_moe as configuration
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as modeling
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.conversion_mapping import get_model_conversion_mapping

    config_path = MODEL_ROOT / "config.json"
    index_path = MODEL_ROOT / "model.safetensors.index.json"
    _require(config_path.is_file() and index_path.is_file(), "checkpoint metadata missing")
    raw_config = _json_object(config_path)
    index = _json_object(index_path)
    tensors, physical_bytes = _read_safetensor_headers(index)

    config = AutoConfig.from_pretrained(MODEL_ROOT, local_files_only=True)
    _require(type(config).__name__ == "Qwen3_5MoeConfig", "composite config class changed")
    _require(
        type(config.text_config).__name__ == "Qwen3_5MoeTextConfig",
        "text config class changed",
    )

    with torch.device("meta"):
        full_model = modeling.Qwen3_5MoeForConditionalGeneration(config)
        causal_model = AutoModelForCausalLM.from_config(config)

    _require(
        type(causal_model).__name__ == "Qwen3_5MoeForCausalLM",
        "AutoModelForCausalLM dispatch changed",
    )
    _require(
        type(causal_model.config).__name__ == "Qwen3_5MoeTextConfig",
        "AutoModelForCausalLM did not extract text_config",
    )
    causal_modules = dict(causal_model.named_modules())
    causal_parameters = dict(causal_model.named_parameters())
    full_modules = dict(full_model.named_modules())
    full_parameters = dict(full_model.named_parameters())
    _require(not any(name.startswith("model.visual") for name in causal_modules), "vision instantiated in causal model")
    _require(
        not any(name.startswith("model.language_model") for name in causal_modules),
        "composite language namespace leaked into causal model",
    )

    transforms = get_model_conversion_mapping(causal_model)
    converted_language = {
        _apply_key_conversions(name, transforms): item
        for name, item in tensors.items()
        if _checkpoint_category(name) == "language"
    }
    _require(
        set(converted_language) == set(causal_parameters),
        "built-in text prefix conversion does not cover the exact language key set",
    )
    for name, parameter in causal_parameters.items():
        _require(
            converted_language[name]["shape"] == list(parameter.shape),
            f"text runtime/checkpoint shape mismatch: {name}",
        )

    non_mtp = {name: item for name, item in tensors.items() if _checkpoint_category(name) != "mtp"}
    _require(set(non_mtp) == set(full_parameters), "conditional model/checkpoint key set changed")
    for name, parameter in full_parameters.items():
        _require(
            non_mtp[name]["shape"] == list(parameter.shape),
            f"conditional runtime/checkpoint shape mismatch: {name}",
        )

    text_config = raw_config["text_config"]
    layer_types = text_config["layer_types"]
    _require(len(layer_types) == text_config["num_hidden_layers"] == 40, "layer geometry changed")
    _require(Counter(layer_types) == {"linear_attention": 30, "full_attention": 10}, "mixer layout changed")

    layers = []
    shared_module_targets = []
    routed_parameter_targets = []
    frozen_router_parameters = []
    frozen_shared_gate_parameters = []
    for layer_index, layer_type in enumerate(layer_types):
        checkpoint_base = f"model.language_model.layers.{layer_index}"
        runtime_base = f"model.layers.{layer_index}"
        checkpoint_mlp = f"{checkpoint_base}.mlp"
        runtime_mlp = f"{runtime_base}.mlp"
        module_specs = [
            ("moe_block", checkpoint_mlp, runtime_mlp),
            ("router", f"{checkpoint_mlp}.gate", f"{runtime_mlp}.gate"),
            ("routed_experts", f"{checkpoint_mlp}.experts", f"{runtime_mlp}.experts"),
            ("shared_expert", f"{checkpoint_mlp}.shared_expert", f"{runtime_mlp}.shared_expert"),
            (
                "shared_gate_projection",
                f"{checkpoint_mlp}.shared_expert.gate_proj",
                f"{runtime_mlp}.shared_expert.gate_proj",
            ),
            (
                "shared_up_projection",
                f"{checkpoint_mlp}.shared_expert.up_proj",
                f"{runtime_mlp}.shared_expert.up_proj",
            ),
            (
                "shared_down_projection",
                f"{checkpoint_mlp}.shared_expert.down_proj",
                f"{runtime_mlp}.shared_expert.down_proj",
            ),
            (
                "shared_expert_gate",
                f"{checkpoint_mlp}.shared_expert_gate",
                f"{runtime_mlp}.shared_expert_gate",
            ),
        ]
        modules = []
        for role, checkpoint_name, runtime_name in module_specs:
            _require(checkpoint_name in full_modules, f"missing checkpoint module: {checkpoint_name}")
            _require(runtime_name in causal_modules, f"missing runtime module: {runtime_name}")
            checkpoint_class = (
                f"{type(full_modules[checkpoint_name]).__module__}."
                f"{type(full_modules[checkpoint_name]).__name__}"
            )
            runtime_class = (
                f"{type(causal_modules[runtime_name]).__module__}."
                f"{type(causal_modules[runtime_name]).__name__}"
            )
            _require(checkpoint_class == runtime_class, f"module class mismatch: {runtime_name}")
            modules.append(
                {
                    "role": role,
                    "checkpoint_name": checkpoint_name,
                    "runtime_name": runtime_name,
                    "class": runtime_class,
                }
            )

        parameter_specs = [
            ("router", "gate.weight", False),
            ("routed_gate_up", "experts.gate_up_proj", True),
            ("routed_down", "experts.down_proj", True),
            ("shared_gate", "shared_expert.gate_proj.weight", True),
            ("shared_up", "shared_expert.up_proj.weight", True),
            ("shared_down", "shared_expert.down_proj.weight", True),
            ("shared_expert_gate", "shared_expert_gate.weight", False),
        ]
        parameters = []
        for role, suffix, train_with_lora in parameter_specs:
            checkpoint_name = f"{checkpoint_mlp}.{suffix}"
            runtime_name = f"{runtime_mlp}.{suffix}"
            _require(checkpoint_name in tensors, f"missing checkpoint parameter: {checkpoint_name}")
            _require(runtime_name in causal_parameters, f"missing runtime parameter: {runtime_name}")
            item = tensors[checkpoint_name]
            parameters.append(
                {
                    "role": role,
                    "checkpoint_name": checkpoint_name,
                    "runtime_name": runtime_name,
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                    "file": item["file"],
                    "train_with_lora": train_with_lora,
                }
            )

        shared = [
            f"{runtime_mlp}.shared_expert.gate_proj",
            f"{runtime_mlp}.shared_expert.up_proj",
            f"{runtime_mlp}.shared_expert.down_proj",
        ]
        routed = [
            f"{runtime_mlp}.experts.gate_up_proj",
            f"{runtime_mlp}.experts.down_proj",
        ]
        shared_module_targets.extend(shared)
        routed_parameter_targets.extend(routed)
        frozen_router_parameters.append(f"{runtime_mlp}.gate.weight")
        frozen_shared_gate_parameters.append(f"{runtime_mlp}.shared_expert_gate.weight")

        mixer_name = f"{runtime_base}." + (
            "linear_attn" if layer_type == "linear_attention" else "self_attn"
        )
        _require(mixer_name in causal_modules, f"missing token mixer: {mixer_name}")
        layers.append(
            {
                "index": layer_index,
                "type": layer_type,
                "token_mixer_runtime_name": mixer_name,
                "token_mixer_class": (
                    f"{type(causal_modules[mixer_name]).__module__}."
                    f"{type(causal_modules[mixer_name]).__name__}"
                ),
                "modules": modules,
                "parameters": parameters,
            }
        )

    direct_expert_parameters = sorted(
        name for name, _ in causal_modules["model.layers.0.mlp.experts"].named_parameters(recurse=False)
    )
    _require(
        direct_expert_parameters == ["down_proj", "gate_up_proj"],
        "routed experts are no longer direct fused parameters",
    )

    target_modules = set(shared_module_targets)
    target_parameters = set(routed_parameter_targets)
    _require(len(target_modules) == 120 and len(target_parameters) == 80, "target cardinality changed")
    _require(target_modules <= set(causal_modules), "shared target module missing")
    _require(target_parameters <= set(causal_parameters), "routed target parameter missing")
    for name in target_modules | target_parameters:
        _require(name.startswith("model.layers."), f"non-literal target path: {name}")
        _require(not any(char in name for char in "*[]()?$^|\\"), f"pattern target forbidden: {name}")

    categories = {}
    for category in ("language", "visual", "mtp", "other"):
        selected = {name: item for name, item in tensors.items() if _checkpoint_category(name) == category}
        categories[category] = {
            "tensor_count": len(selected),
            "logical_bytes": sum(item["logical_bytes"] for item in selected.values()),
            "key_names_sha256": _canonical_sha256(sorted(selected)),
        }

    checkpoint = {
        "path": str(MODEL_ROOT),
        "config_sha256": _file_sha256(config_path),
        "index_sha256": _file_sha256(index_path),
        "weight_file_count": len(set(index["weight_map"].values())),
        "tensor_count": len(tensors),
        "logical_tensor_bytes": int(index["metadata"]["total_size"]),
        "physical_weight_bytes": physical_bytes,
        "categories": categories,
        "declared_model_type": raw_config["model_type"],
        "declared_architectures": raw_config["architectures"],
        "declared_transformers_version": raw_config.get("transformers_version"),
        "geometry": {
            "hidden_size": text_config["hidden_size"],
            "vocab_size": text_config["vocab_size"],
            "num_hidden_layers": text_config["num_hidden_layers"],
            "num_experts": text_config["num_experts"],
            "num_experts_per_token": text_config["num_experts_per_tok"],
            "routed_expert_intermediate_size": text_config["moe_intermediate_size"],
            "shared_expert_intermediate_size": text_config["shared_expert_intermediate_size"],
            "linear_attention_layers": layer_types.count("linear_attention"),
            "full_attention_layers": layer_types.count("full_attention"),
            "layer_types": layer_types,
            "mtp_num_hidden_layers": text_config["mtp_num_hidden_layers"],
            "vision_depth": raw_config["vision_config"]["depth"],
            "vision_hidden_size": raw_config["vision_config"]["hidden_size"],
            "vision_intermediate_size": raw_config["vision_config"]["intermediate_size"],
        },
    }

    architecture = {
        "installed_classes": {
            "composite_config": f"{type(config).__module__}.{type(config).__name__}",
            "text_config": f"{type(config.text_config).__module__}.{type(config.text_config).__name__}",
            "checkpoint_architecture": (
                f"{type(full_model).__module__}.{type(full_model).__name__}"
            ),
            "training_architecture": (
                f"{type(causal_model).__module__}.{type(causal_model).__name__}"
            ),
        },
        "auto_causal_lm_dispatch": {
            "input_is_composite_config": True,
            "auto_factory_extracts_text_config": True,
            "result_has_visual_module": False,
            "result_has_composite_language_model_namespace": False,
            "result_parameter_count": len(causal_parameters),
            "checkpoint_language_tensor_count": categories["language"]["tensor_count"],
            "mapped_language_keys_equal_runtime_parameters": True,
            "built_in_conversion_mapping": [_conversion_row(item) for item in transforms],
            "effective_prefix_mapping": {
                "checkpoint": "model.language_model.<suffix>",
                "runtime": "model.<suffix>",
                "lm_head": "unchanged",
            },
            "ignored_checkpoint_prefixes": {
                "model.visual.*": categories["visual"]["tensor_count"],
                "mtp.*": categories["mtp"]["tensor_count"],
            },
            "installed_ignore_on_load_unexpected": list(
                modeling.Qwen3_5MoeForCausalLM._keys_to_ignore_on_load_unexpected
            ),
        },
        "training_modality_decision": {
            "decision": "text_only_exclude_vision_and_mtp",
            "loader": "transformers.AutoModelForCausalLM.from_pretrained",
            "config_input": str(MODEL_ROOT),
            "local_files_only": True,
            "trust_remote_code": False,
            "vision_instantiated": False,
            "vision_parameters_in_optimizer": False,
            "mtp_instantiated": False,
            "require_fail_closed_source_and_mapping_recheck": True,
            "reason": (
                "The training corpus is text-only. Installed Transformers 5.12.1 "
                "dispatches the composite checkpoint to Qwen3_5MoeForCausalLM, "
                "extracts text_config, applies its registered language_model-to-model "
                "prefix conversion, and does not instantiate the vision or MTP modules."
            ),
        },
        "fused_routed_expert_storage": {
            "module_class": (
                f"{type(causal_modules['model.layers.0.mlp.experts']).__module__}."
                f"{type(causal_modules['model.layers.0.mlp.experts']).__name__}"
            ),
            "direct_parameter_names": direct_expert_parameters,
            "has_projection_submodules": False,
            "gate_up_shape": tensors["model.language_model.layers.0.mlp.experts.gate_up_proj"]["shape"],
            "down_shape": tensors["model.language_model.layers.0.mlp.experts.down_proj"]["shape"],
            "checkpoint_names_have_weight_suffix": False,
        },
        "layers": layers,
        "adapter_target_contract": {
            "matching_mode": "literal_full_name_membership_only",
            "suffix_guessing_allowed": False,
            "regex_targeting_allowed": False,
            "shared_expert_target_modules": sorted(shared_module_targets),
            "routed_expert_target_parameters": sorted(routed_parameter_targets),
            "shared_rank": 16,
            "shared_alpha": 16,
            "routed_rank": 4,
            "routed_alpha": 4,
            "require_pre_attach_exact_set_equality": True,
            "require_post_attach_observed_exact_set_equality": True,
        },
        "explicitly_frozen": {
            "router_parameters": sorted(frozen_router_parameters),
            "shared_expert_gate_parameters": sorted(frozen_shared_gate_parameters),
            "attention_and_linear_mixing": True,
            "embeddings": ["model.embed_tokens.weight"],
            "language_model_head": ["lm_head.weight"],
            "normalization_parameters": True,
            "vision": "excluded_not_instantiated",
            "mtp": "excluded_not_instantiated",
        },
    }

    source_objects = {
        "transformers_qwen_configuration": configuration,
        "transformers_qwen_modeling": modeling,
        "transformers_auto_factory": auto_factory,
        "transformers_conversion_mapping": conversion_mapping,
    }
    implementation_sources = {
        name: _module_source(value) for name, value in source_objects.items()
    }
    implementation_sources["transformers_package"] = _module_source(transformers)
    architecture["implementation_sources"] = implementation_sources
    architecture["class_source_lines"] = {
        name: inspect.getsourcelines(getattr(modeling, name))[1]
        for name in (
            "Qwen3_5MoeMLP",
            "Qwen3_5MoeExperts",
            "Qwen3_5MoeTopKRouter",
            "Qwen3_5MoeSparseMoeBlock",
            "Qwen3_5MoeTextModel",
            "Qwen3_5MoeForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
        )
    }
    return checkpoint, architecture


def _software_contract() -> dict[str, Any]:
    import torch
    import peft.tuners.lora.config as peft_lora_config
    import peft.tuners.lora.layer as peft_lora_layer
    import peft.tuners.lora.model as peft_lora_model
    import peft.tuners.tuners_utils as peft_tuners_utils

    package_pins = {name: _package_pin(name) for name in PACKAGE_PINS}
    _require(package_pins["transformers"]["version"] == "5.12.1", "Transformers pin changed")
    _require(package_pins["peft"]["version"] == "0.19.1", "PEFT pin changed")
    _require(package_pins["accelerate"]["version"] == "1.14.0", "Accelerate pin changed")
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": _relative_source(Path(sys.executable)),
        },
        "host": {
            "system": platform.system(),
            "kernel_release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": package_pins,
        "torch_runtime": {
            "version": torch.__version__,
            "git_version": torch.version.git_version,
            "compiled_cuda": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "audit_cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "audit_cuda_available": torch.cuda.is_available(),
            "audit_cuda_device_count": torch.cuda.device_count(),
        },
        "peft_implementation_sources": {
            "lora_config": _module_source(peft_lora_config),
            "lora_layer": _module_source(peft_lora_layer),
            "lora_model": _module_source(peft_lora_model),
            "tuner_utils": _module_source(peft_tuners_utils),
        },
        "optional_training_packages": {
            "trl": "absent_not_required_for_initial_sft",
            "unsloth": "absent_not_used_by_authoritative_initial_sft_path",
            "fast_linear_attention": (
                "installed_runtime_validation_delegated_to_"
                "fast_linear_attention_contract_v1"
                if package_pins["flash-linear-attention"]["installed"]
                and package_pins["fla-core"]["installed"]
                else "absent_transformers_uses_torch_fallback"
            ),
            "causal_conv1d": (
                "installed_runtime_validation_delegated_to_"
                "fast_linear_attention_contract_v1"
                if package_pins["causal-conv1d"]["installed"]
                else "absent_transformers_uses_torch_fallback"
            ),
            "flash_attention": "absent_no_flash_attention_kernel_available",
        },
    }


def build_contract(repository_commit: str | None = None) -> dict[str, Any]:
    repository_commit = repository_commit or _command("git", "rev-parse", "HEAD")
    _require(re.fullmatch(r"[0-9a-f]{40}", repository_commit) is not None, "invalid Git commit")
    checkpoint, architecture = _model_and_checkpoint_contract()
    body = {
        "schema": SCHEMA,
        "bead": "specialist-j59.2",
        "repository": {
            "path": str(ROOT),
            "remote": _command("git", "remote", "get-url", "origin"),
            "audit_base_commit": repository_commit,
            "submodules": _submodule_pins(),
        },
        "authority": {
            "training": False,
            "optimizer_creation": False,
            "weight_loading": False,
            "weight_or_adapter_update": False,
            "dataset_access": False,
            "protected_evaluation_access": False,
            "gpu_cuda_visibility_in_audit": False,
        },
        "hardware": _hardware_contract(),
        "software": _software_contract(),
        "checkpoint": checkpoint,
        "architecture": architecture,
        "acceptance": {
            "exact_four_gpu_host_verified": True,
            "software_and_runtime_state_pinned": True,
            "checkpoint_geometry_verified": True,
            "actual_module_and_parameter_names_verified": True,
            "fused_routed_expert_storage_verified": True,
            "text_only_loader_and_prefix_conversion_verified": True,
            "vision_and_mtp_exclusion_decided": True,
            "literal_full_name_targets_only": True,
            "training_launched": False,
        },
    }
    result = copy.deepcopy(body)
    result["content_sha256_before_self_field"] = _canonical_sha256(body)
    validate_contract(result)
    return result


def validate_contract(value: dict[str, Any]) -> None:
    _require(isinstance(value, dict), "contract must be an object")
    body = copy.deepcopy(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and _canonical_sha256(body) == claimed,
        "contract content hash changed",
    )
    _require(body.get("schema") == SCHEMA and body.get("bead") == "specialist-j59.2", "contract identity changed")
    _require(body.get("authority") == {
        "training": False,
        "optimizer_creation": False,
        "weight_loading": False,
        "weight_or_adapter_update": False,
        "dataset_access": False,
        "protected_evaluation_access": False,
        "gpu_cuda_visibility_in_audit": False,
    }, "audit authority widened")
    acceptance = body.get("acceptance")
    _require(isinstance(acceptance, dict), "acceptance evidence missing")
    _require(
        acceptance.get("training_launched") is False
        and all(value is True for key, value in acceptance.items() if key != "training_launched"),
        "acceptance evidence incomplete",
    )
    targets = body["architecture"]["adapter_target_contract"]
    _require(
        targets["matching_mode"] == "literal_full_name_membership_only"
        and targets["suffix_guessing_allowed"] is False
        and targets["regex_targeting_allowed"] is False
        and len(targets["shared_expert_target_modules"]) == 120
        and len(targets["routed_expert_target_parameters"]) == 80,
        "adapter target contract changed",
    )
    _require(
        len(body["architecture"]["explicitly_frozen"]["router_parameters"]) == 40
        and len(body["architecture"]["explicitly_frozen"]["shared_expert_gate_parameters"]) == 40,
        "router freeze manifest changed",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify the committed artifact")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.check:
        _require(output.is_file(), f"contract missing: {output}")
        existing = _json_object(output)
        validate_contract(existing)
        pinned_commit = existing["repository"]["audit_base_commit"]
        _command("git", "cat-file", "-e", f"{pinned_commit}^{{commit}}")
        actual = build_contract(repository_commit=pinned_commit)
        _require(existing == actual, f"architecture contract drift: {output}")
        print(f"verified {output}")
        return

    value = build_contract()
    _atomic_write(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
