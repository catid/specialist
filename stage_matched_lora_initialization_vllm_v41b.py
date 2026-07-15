#!/usr/bin/env python3
"""Stage the sealed V41A LoRA initialization in vLLM's key namespace.

This CPU-only utility performs exactly one semantic operation: it rewrites
``base_model.model.model.layers.*`` tensor keys to
``base_model.model.model.language_model.layers.*``.  Tensor storage bytes and
the adapter configuration bytes are preserved exactly.  The sealed V41A
source is verified before use, the staged artifact is reopened after writing,
and a self-hashed manifest records every source/target tensor identity.

No model, tokenizer, dataset, evaluation path, or GPU is opened here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY_V41B = (
    ROOT / "experiments/eggroll_es_hpo/initial_adapters/"
    "matched_lora_initialization_v41a_seed20260715041"
).resolve()
DEFAULT_OUTPUT_V41B = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "matched_lora_initialization_v41b"
).resolve()

SOURCE_PREFIX_V41B = "base_model.model.model.layers."
TARGET_PREFIX_V41B = "base_model.model.model.language_model.layers."
EXPECTED_SOURCE_WEIGHTS_SHA256_V41B = (
    "29fe0beead8a491cf06e9f562a1838d9c44e94a74e6a4024549e87f10557111f"
)
EXPECTED_SOURCE_CONFIG_SHA256_V41B = (
    "ede582c12e82fb50eb97ac934ff08eb553a79d2c2d999235abcd8b29795b1d52"
)
EXPECTED_SOURCE_MANIFEST_FILE_SHA256_V41B = (
    "a2fa79e6ac06f75743d3fee8f5c0b1aabe6bb83b52b05910ed6460438e2640a2"
)
EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256_V41B = (
    "5f885b415302c4e748e19f4d535f1e57ff87f785370f653b345cbdfafda3224b"
)
EXPECTED_SOURCE_TENSOR_IDENTITY_SHA256_V41B = (
    "2dcb9ab45ec26c7041b9782a30fe3c82b987b605b6b0bd95ab5b905b1371ae2e"
)
EXPECTED_TENSOR_COUNT_V41B = 70
EXPECTED_ELEMENTS_V41B = 4_528_128
EXPECTED_TENSOR_BYTES_V41B = EXPECTED_ELEMENTS_V41B * 4


def canonical_sha256_v41b(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256_v41b(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256_v41b(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise RuntimeError("v41b tensor hashing requires contiguous CPU storage")
    return hashlib.sha256(
        tensor.view(torch.uint8).numpy().tobytes(order="C")
    ).hexdigest()


def transform_key_v41b(source_key: str) -> str:
    if source_key.startswith(TARGET_PREFIX_V41B):
        raise ValueError(f"v41b source key cannot be transformed exactly once: {source_key}")
    if not source_key.startswith(SOURCE_PREFIX_V41B):
        raise ValueError(f"v41b source key is outside the sealed namespace: {source_key}")
    suffix = source_key[len(SOURCE_PREFIX_V41B):]
    if not suffix or suffix.startswith("language_model."):
        raise ValueError(f"v41b source key cannot be transformed exactly once: {source_key}")
    return TARGET_PREFIX_V41B + suffix


def _read_self_hashed_manifest_v41b(path: Path) -> tuple[dict, str]:
    manifest = json.loads(Path(path).read_text())
    claimed = manifest.pop("content_sha256_before_self_field", None)
    actual = canonical_sha256_v41b(manifest)
    if claimed != actual:
        raise RuntimeError("v41b source manifest self-hash does not verify")
    manifest["content_sha256_before_self_field"] = claimed
    return manifest, actual


def audit_sealed_source_v41b() -> dict:
    """Verify and load the immutable V41A CPU master."""
    directory = SOURCE_DIRECTORY_V41B
    weights = directory / "adapter_model.safetensors"
    config = directory / "adapter_config.json"
    manifest_path = directory / "initialization_manifest_v41a.json"
    actual_file_hashes = {
        "weights": file_sha256_v41b(weights),
        "config": file_sha256_v41b(config),
        "manifest": file_sha256_v41b(manifest_path),
    }
    expected_file_hashes = {
        "weights": EXPECTED_SOURCE_WEIGHTS_SHA256_V41B,
        "config": EXPECTED_SOURCE_CONFIG_SHA256_V41B,
        "manifest": EXPECTED_SOURCE_MANIFEST_FILE_SHA256_V41B,
    }
    if actual_file_hashes != expected_file_hashes:
        raise RuntimeError("v41b sealed V41A source file identity changed")

    manifest, manifest_content_hash = _read_self_hashed_manifest_v41b(manifest_path)
    if (
        manifest_content_hash != EXPECTED_SOURCE_MANIFEST_CONTENT_SHA256_V41B
        or manifest.get("tensor_identity", {}).get("sha256")
        != EXPECTED_SOURCE_TENSOR_IDENTITY_SHA256_V41B
        or manifest.get("artifact", {}).get("weights_file_sha256")
        != EXPECTED_SOURCE_WEIGHTS_SHA256_V41B
        or manifest.get("artifact", {}).get("adapter_config_file_sha256")
        != EXPECTED_SOURCE_CONFIG_SHA256_V41B
    ):
        raise RuntimeError("v41b sealed V41A source manifest identity changed")

    with safe_open(weights, framework="pt", device="cpu") as handle:
        source_keys = list(handle.keys())
        metadata = handle.metadata() or {}
    if metadata != {"format": "pt"}:
        raise RuntimeError("v41b sealed V41A safetensors metadata changed")
    tensors = load_file(weights, device="cpu")
    if (
        source_keys != sorted(source_keys)
        or source_keys != sorted(tensors)
        or len(source_keys) != EXPECTED_TENSOR_COUNT_V41B
        or any(not key.startswith(SOURCE_PREFIX_V41B) for key in source_keys)
        or len({transform_key_v41b(key) for key in source_keys}) != len(source_keys)
    ):
        raise RuntimeError("v41b sealed V41A tensor key surface changed")

    manifest_records = {
        item["key"]: item for item in manifest.get("tensor_records", [])
    }
    records = []
    for key in source_keys:
        tensor = tensors[key]
        tensor_hash = tensor_sha256_v41b(tensor)
        prior = manifest_records.get(key)
        if (
            tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or not tensor.is_contiguous()
            or prior is None
            or prior.get("tensor_sha256") != tensor_hash
            or prior.get("shape") != list(tensor.shape)
            or prior.get("elements") != tensor.numel()
        ):
            raise RuntimeError(f"v41b sealed V41A tensor identity changed: {key}")
        records.append({
            "source_key": key,
            "target_key": transform_key_v41b(key),
            "shape": list(tensor.shape),
            "dtype": "torch.float32",
            "elements": tensor.numel(),
            "bytes": tensor.numel() * tensor.element_size(),
            "source_tensor_sha256": tensor_hash,
        })
    if (
        sum(item["elements"] for item in records) != EXPECTED_ELEMENTS_V41B
        or sum(item["bytes"] for item in records) != EXPECTED_TENSOR_BYTES_V41B
    ):
        raise RuntimeError("v41b sealed V41A tensor element count changed")
    return {
        "directory": directory,
        "weights": weights,
        "config": config,
        "manifest_path": manifest_path,
        "config_bytes": config.read_bytes(),
        "tensors": tensors,
        "records": records,
        "file_hashes": actual_file_hashes,
        "manifest_content_sha256": manifest_content_hash,
        "source_tensor_identity_sha256": manifest["tensor_identity"]["sha256"],
    }


def _transformed_records_v41b(source: dict) -> tuple[dict[str, torch.Tensor], list[dict]]:
    target_tensors: dict[str, torch.Tensor] = {}
    records = []
    for source_record in source["records"]:
        source_key = source_record["source_key"]
        target_key = source_record["target_key"]
        tensor = source["tensors"][source_key]
        target_tensors[target_key] = tensor
        record = {
            **source_record,
            "target_tensor_sha256": tensor_sha256_v41b(tensor),
            "tensor_bytes_preserved_exactly": True,
        }
        if record["target_tensor_sha256"] != record["source_tensor_sha256"]:
            raise RuntimeError(f"v41b in-memory tensor bytes changed: {source_key}")
        record["record_sha256"] = canonical_sha256_v41b(record)
        records.append(record)
    if len(target_tensors) != EXPECTED_TENSOR_COUNT_V41B:
        raise RuntimeError("v41b transformed tensor key collision")
    return target_tensors, records


def transformed_identity_v41b(records: list[dict]) -> dict:
    value_sequence = [item["target_tensor_sha256"] for item in records]
    return {
        "schema": "vllm-key-only-staged-tensor-identity-v41b",
        "sha256": canonical_sha256_v41b(records),
        "tensor_count": len(records),
        "elements": sum(item["elements"] for item in records),
        "tensor_bytes": sum(item["bytes"] for item in records),
        "source_key_inventory_sha256": canonical_sha256_v41b(
            [item["source_key"] for item in records]
        ),
        "target_key_inventory_sha256": canonical_sha256_v41b(
            [item["target_key"] for item in records]
        ),
        "ordered_value_sequence_sha256": canonical_sha256_v41b(value_sequence),
        "all_tensor_bytes_preserved_exactly": all(
            item["source_tensor_sha256"] == item["target_tensor_sha256"]
            for item in records
        ),
    }


def reopen_and_verify_staged_v41b(
    weights: Path,
    source: dict,
    expected_records: list[dict],
) -> dict:
    expected_target_keys = [item["target_key"] for item in expected_records]
    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}
    if keys != sorted(expected_target_keys) or metadata != {"format": "pt"}:
        raise RuntimeError("v41b staged safetensors header changed on reopen")
    reopened = load_file(weights, device="cpu")
    expected_by_target = {item["target_key"]: item for item in expected_records}
    reopened_records = []
    for source_record in source["records"]:
        source_key = source_record["source_key"]
        target_key = source_record["target_key"]
        source_tensor = source["tensors"][source_key]
        target_tensor = reopened[target_key]
        target_hash = tensor_sha256_v41b(target_tensor)
        record = {
            **source_record,
            "target_tensor_sha256": target_hash,
            "tensor_bytes_preserved_exactly": (
                target_hash == source_record["source_tensor_sha256"]
            ),
        }
        record["record_sha256"] = canonical_sha256_v41b(record)
        if (
            target_tensor.dtype != torch.float32
            or not target_tensor.is_contiguous()
            or target_tensor.shape != source_tensor.shape
            or not torch.equal(target_tensor, source_tensor)
            or record != expected_by_target[target_key]
        ):
            raise RuntimeError(f"v41b staged tensor bytes changed: {target_key}")
        reopened_records.append(record)
    identity = transformed_identity_v41b(reopened_records)
    if not identity["all_tensor_bytes_preserved_exactly"]:
        raise RuntimeError("v41b staged value sequence changed on reopen")
    return {
        "schema": "vllm-key-only-stage-readback-v41b",
        "verified": True,
        "weights_file_sha256": file_sha256_v41b(weights),
        "weights_file_bytes": weights.stat().st_size,
        "metadata": metadata,
        "transformed_identity": identity,
    }


def _atomic_bytes_exclusive_v41b(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_bytes(raw)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_exclusive_v41b(path: Path, value: dict) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes_exclusive_v41b(path, raw)


def stage_artifact_v41b(output_directory: Path = DEFAULT_OUTPUT_V41B) -> dict:
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = audit_sealed_source_v41b()
    target_tensors, records = _transformed_records_v41b(source)
    identity = transformed_identity_v41b(records)
    if not identity["all_tensor_bytes_preserved_exactly"]:
        raise RuntimeError("v41b key-only transform changed tensor values")

    output.mkdir(parents=True)
    weights = output / "adapter_model.safetensors"
    config = output / "adapter_config.json"
    manifest_path = output / "stage_manifest_v41b.json"
    temporary_weights = output / f".adapter_model.safetensors.tmp-{os.getpid()}"
    linked_weights = False
    try:
        save_file(target_tensors, temporary_weights, metadata={"format": "pt"})
        os.link(temporary_weights, weights)
        linked_weights = True
        _atomic_bytes_exclusive_v41b(config, source["config_bytes"])
        if (
            config.read_bytes() != source["config_bytes"]
            or file_sha256_v41b(config) != EXPECTED_SOURCE_CONFIG_SHA256_V41B
        ):
            raise RuntimeError("v41b adapter config bytes changed while staging")
        readback = reopen_and_verify_staged_v41b(weights, source, records)
        if readback["transformed_identity"] != identity:
            raise RuntimeError("v41b staged readback identity changed")

        manifest = {
            "schema": "vllm-key-only-stage-manifest-v41b",
            "status": "complete_cpu_only_key_transform",
            "source": {
                "repository_relative_directory": str(
                    SOURCE_DIRECTORY_V41B.relative_to(ROOT)
                ),
                "weights_file_sha256": source["file_hashes"]["weights"],
                "adapter_config_file_sha256": source["file_hashes"]["config"],
                "manifest_file_sha256": source["file_hashes"]["manifest"],
                "manifest_content_sha256": source["manifest_content_sha256"],
                "tensor_identity_sha256": source["source_tensor_identity_sha256"],
            },
            "implementation": {
                "repository_relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
                "file_sha256": file_sha256_v41b(Path(__file__).resolve()),
                "torch_version": torch.__version__,
            },
            "transform": {
                "operation": "key_prefix_replacement_only",
                "source_prefix": SOURCE_PREFIX_V41B,
                "target_prefix": TARGET_PREFIX_V41B,
                "tensor_arithmetic_performed": False,
                "tensor_cast_performed": False,
                "tensor_bytes_preserved_exactly": True,
                "adapter_config_copied_byte_exact": True,
            },
            "artifact": {
                "weights_file": weights.name,
                "weights_file_sha256": readback["weights_file_sha256"],
                "weights_file_bytes": readback["weights_file_bytes"],
                "adapter_config_file": config.name,
                "adapter_config_file_sha256": file_sha256_v41b(config),
                "target_namespace": TARGET_PREFIX_V41B + "*",
            },
            "transformed_identity": identity,
            "tensor_mapping_records": records,
            "readback": readback,
            "dataset_or_training_examples_accessed": False,
            "shadow_ood_holdout_or_heldout_accessed": False,
            "gpu_accessed": False,
            "evaluation_performed": False,
        }
        manifest["content_sha256_before_self_field"] = canonical_sha256_v41b(manifest)
        _atomic_json_exclusive_v41b(manifest_path, manifest)
        reopened, content_hash = _read_self_hashed_manifest_v41b(manifest_path)
        if reopened != manifest or content_hash != manifest["content_sha256_before_self_field"]:
            raise RuntimeError("v41b stage manifest changed on reopen")
        return manifest
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        config.unlink(missing_ok=True)
        if linked_weights:
            weights.unlink(missing_ok=True)
        try:
            output.rmdir()
        except OSError:
            pass
        raise
    finally:
        temporary_weights.unlink(missing_ok=True)


def parser_v41b() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_V41B))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_v41b().parse_args(argv)
    manifest = stage_artifact_v41b(Path(args.output))
    output = Path(args.output).resolve()
    print(json.dumps({
        "directory": str(output),
        "weights_sha256": manifest["artifact"]["weights_file_sha256"],
        "transformed_identity_sha256": manifest["transformed_identity"]["sha256"],
        "ordered_value_sequence_sha256": manifest["transformed_identity"][
            "ordered_value_sequence_sha256"
        ],
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "manifest_file_sha256": file_sha256_v41b(
            output / "stage_manifest_v41b.json"
        ),
        "tensor_count": manifest["transformed_identity"]["tensor_count"],
        "all_tensor_bytes_preserved_exactly": manifest["transformed_identity"][
            "all_tensor_bytes_preserved_exactly"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
