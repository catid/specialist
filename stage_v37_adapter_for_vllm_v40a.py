#!/usr/bin/env python3
"""Stage V37A under Qwen3.5 vLLM's outer language-model key namespace."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

import run_sft_train_only_control_v36a as hashing


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT / "experiments/sft_controls/v37a_equal_unit_fold3_v412/"
    "middle_late_r32_seed17/final"
).resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v37a_qwen35_vllm_namespace_v40a"
).resolve()
SOURCE_PREFIX = "base_model.model.model.layers."
STAGED_PREFIX = "base_model.model.model.language_model.layers."


def file_sha256(path: Path) -> str:
    return hashing.file_sha256(path)


def canonical_sha256(value) -> str:
    return hashing.canonical_sha256(value)


def tensor_sha256(tensor) -> str:
    raw = tensor.contiguous().view(__import__("torch").uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.link(temporary, path)
    temporary.unlink()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    source_weights = SOURCE / "adapter_model.safetensors"
    tensors = {}
    mapping = []
    with safe_open(source_weights, framework="pt", device="cpu") as handle:
        if len(handle.keys()) != 70:
            raise RuntimeError("v40a source V37A tensor count changed")
        for source_key in handle.keys():
            if not source_key.startswith(SOURCE_PREFIX):
                raise RuntimeError(f"v40a unexpected source key: {source_key}")
            staged_key = STAGED_PREFIX + source_key[len(SOURCE_PREFIX):]
            tensor = handle.get_tensor(source_key).contiguous()
            tensors[staged_key] = tensor
            mapping.append({
                "source_key": source_key, "staged_key": staged_key,
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "elements": int(tensor.numel()), "tensor_sha256": tensor_sha256(tensor),
            })
    if len(tensors) != 70 or sum(t.numel() for t in tensors.values()) != 4_528_128:
        raise RuntimeError("v40a staged adapter inventory changed")
    staged_weights = OUTPUT / "adapter_model.safetensors"
    temporary = OUTPUT / f".adapter_model.safetensors.tmp-{os.getpid()}"
    save_file(tensors, temporary, metadata={
        "schema": "v37a-qwen35-vllm-namespace-stage-v40a",
        "source_weights_sha256": file_sha256(source_weights),
        "key_mapping_sha256": canonical_sha256(mapping),
        "tensor_values": "byte_exact_source_values_only_keys_renamed",
    })
    os.link(temporary, staged_weights)
    temporary.unlink()
    source_config = SOURCE / "adapter_config.json"
    staged_config = OUTPUT / "adapter_config.json"
    shutil.copyfile(source_config, staged_config)
    manifest = {
        "schema": "v37a-qwen35-vllm-namespace-stage-manifest-v40a",
        "source": {
            "directory": str(SOURCE),
            "weights_sha256": file_sha256(source_weights),
            "config_sha256": file_sha256(source_config),
        },
        "staged": {
            "directory": str(OUTPUT),
            "weights_sha256": file_sha256(staged_weights),
            "config_sha256": file_sha256(staged_config),
        },
        "transformation": {
            "source_prefix": SOURCE_PREFIX, "staged_prefix": STAGED_PREFIX,
            "key_only_rename": True, "tensor_values_byte_exact": True,
            "tensor_count": 70, "tensor_elements": 4_528_128,
            "key_mapping_sha256": canonical_sha256(mapping),
            "mapping": mapping,
        },
        "dataset_or_evaluation_accessed": False,
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    atomic_json(OUTPUT / "stage_manifest_v40a.json", manifest)
    print(json.dumps({
        "directory": str(OUTPUT),
        "weights_sha256": file_sha256(staged_weights),
        "manifest_sha256": file_sha256(OUTPUT / "stage_manifest_v40a.json"),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
