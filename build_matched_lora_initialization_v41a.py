#!/usr/bin/env python3
"""Build the sealed, optimizer-neutral LoRA initialization for V41A.

The artifact is deliberately expressed in the original V37A PEFT namespace:
70 unscaled FP32 tensors covering the exact rank-32 layer-20--23 surface.  Each
LoRA A matrix receives an independently derived, deterministic standard PEFT
Kaiming-uniform initialization; every B matrix is exact positive zero.  The
nonzero-A/zero-B state has no initial model effect but avoids the zero/zero
bilinear antithetic degeneracy.

This builder is train-only and CPU-only.  It has no dataset or evaluation
arguments and refuses to overwrite an existing artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


ROOT = Path(__file__).resolve().parent
SOURCE_CONFIG_V41A = (
    ROOT / "experiments/sft_controls/v37a_equal_unit_fold3_v412/"
    "middle_late_r32_seed17/final/adapter_config.json"
).resolve()
DEFAULT_OUTPUT_V41A = (
    ROOT / "experiments/eggroll_es_hpo/initial_adapters/"
    "matched_lora_initialization_v41a_seed20260715041"
).resolve()

SEALED_INITIALIZATION_SEED_V41A = 20_260_715_041
SOURCE_CONFIG_SHA256_V41A = (
    "ede582c12e82fb50eb97ac934ff08eb553a79d2c2d999235abcd8b29795b1d52"
)
EXPECTED_TENSOR_COUNT_V41A = 70
EXPECTED_MODULE_COUNT_V41A = 35
EXPECTED_ELEMENTS_V41A = 4_528_128
EXPECTED_A_ELEMENTS_V41A = 2_359_296
EXPECTED_B_ELEMENTS_V41A = 2_168_832
EXPECTED_RANK_V41A = 32
EXPECTED_ALPHA_V41A = 64
KEY_PREFIX_V41A = "base_model.model.model.layers."
WEIGHTS_METADATA_SCHEMA_V41A = "matched-lora-initialization-weights-v41a"


def canonical_sha256_v41a(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256_v41a(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256_v41a(tensor: torch.Tensor) -> str:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise RuntimeError("v41a tensor hashing requires contiguous CPU storage")
    raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def module_shapes_v41a() -> dict[str, tuple[tuple[int, int], tuple[int, int]]]:
    """Return the exact 35-module V37A rank-32 LoRA surface."""
    common_mlp = {
        "mlp.gate": ((32, 2048), (256, 32)),
        "mlp.shared_expert.down_proj": ((32, 512), (2048, 32)),
        "mlp.shared_expert.gate_proj": ((32, 2048), (512, 32)),
        "mlp.shared_expert.up_proj": ((32, 2048), (512, 32)),
    }
    linear_attention = {
        "linear_attn.in_proj_a": ((32, 2048), (32, 32)),
        "linear_attn.in_proj_b": ((32, 2048), (32, 32)),
        "linear_attn.in_proj_qkv": ((32, 2048), (8192, 32)),
        "linear_attn.in_proj_z": ((32, 2048), (4096, 32)),
        "linear_attn.out_proj": ((32, 4096), (2048, 32)),
    }
    full_attention = {
        "self_attn.k_proj": ((32, 2048), (512, 32)),
        "self_attn.o_proj": ((32, 4096), (2048, 32)),
        "self_attn.q_proj": ((32, 2048), (8192, 32)),
        "self_attn.v_proj": ((32, 2048), (512, 32)),
    }
    result: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
    for layer in (20, 21, 22):
        for suffix, shapes in {**linear_attention, **common_mlp}.items():
            result[f"model.layers.{layer}.{suffix}"] = shapes
    for suffix, shapes in {**full_attention, **common_mlp}.items():
        result[f"model.layers.23.{suffix}"] = shapes
    if len(result) != EXPECTED_MODULE_COUNT_V41A:
        raise RuntimeError("v41a module surface changed")
    return dict(sorted(result.items()))


def tensor_specs_v41a() -> list[dict]:
    records = []
    for module, (a_shape, b_shape) in module_shapes_v41a().items():
        for role, shape in (("A", a_shape), ("B", b_shape)):
            key = f"base_model.model.{module}.lora_{role}.weight"
            records.append({
                "key": key,
                "module": module,
                "role": role,
                "shape": list(shape),
                "dtype": "torch.float32",
                "elements": math.prod(shape),
            })
    records.sort(key=lambda item: item["key"])
    a_elements = sum(item["elements"] for item in records if item["role"] == "A")
    b_elements = sum(item["elements"] for item in records if item["role"] == "B")
    if (
        len(records) != EXPECTED_TENSOR_COUNT_V41A
        or a_elements != EXPECTED_A_ELEMENTS_V41A
        or b_elements != EXPECTED_B_ELEMENTS_V41A
        or a_elements + b_elements != EXPECTED_ELEMENTS_V41A
        or len({item["key"] for item in records}) != len(records)
    ):
        raise RuntimeError("v41a tensor surface count changed")
    return records


def surface_identity_v41a() -> dict:
    records = tensor_specs_v41a()
    return {
        "schema": "matched-lora-peft-surface-v41a",
        "sha256": canonical_sha256_v41a(records),
        "module_count": EXPECTED_MODULE_COUNT_V41A,
        "tensor_count": len(records),
        "elements": sum(item["elements"] for item in records),
        "a_elements": sum(item["elements"] for item in records if item["role"] == "A"),
        "b_elements": sum(item["elements"] for item in records if item["role"] == "B"),
        "records": records,
    }


def validate_source_config_v41a(path: Path = SOURCE_CONFIG_V41A) -> tuple[bytes, dict]:
    path = Path(path).resolve()
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_CONFIG_SHA256_V41A:
        raise RuntimeError("v41a frozen V37A adapter config bytes changed")
    config = json.loads(raw)
    expected_targets = {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
        "down_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
        "out_proj",
    }
    expected_parameters = {
        f"model.layers.{layer}.mlp.gate.weight" for layer in (20, 21, 22, 23)
    }
    if (
        config.get("r") != EXPECTED_RANK_V41A
        or config.get("lora_alpha") != EXPECTED_ALPHA_V41A
        or config.get("bias") != "none"
        or config.get("lora_dropout") != 0.0
        or config.get("init_lora_weights") is not True
        or config.get("use_rslora") is not False
        or config.get("layers_to_transform") != [20, 21, 22, 23]
        or set(config.get("target_modules", [])) != expected_targets
        or set(config.get("target_parameters", [])) != expected_parameters
    ):
        raise RuntimeError("v41a frozen V37A adapter semantics changed")
    return raw, config


def initialization_seed_for_key_v41a(key: str) -> int:
    if key not in {item["key"] for item in tensor_specs_v41a()}:
        raise ValueError(f"v41a seed requested for unknown tensor: {key}")
    payload = (
        f"matched-lora-initialization-v41a\0"
        f"{SEALED_INITIALIZATION_SEED_V41A}\0{key}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & (
        (1 << 63) - 1
    )


def _tensor_record_v41a(spec: dict, tensor: torch.Tensor) -> dict:
    nonzero = int(torch.count_nonzero(tensor).item())
    role = spec["role"]
    record = {
        **spec,
        "tensor_sha256": tensor_sha256_v41a(tensor),
        "nonzero_elements": nonzero,
        "all_zero": nonzero == 0,
        "initialization": (
            {
                "method": "torch.nn.init.kaiming_uniform_",
                "a": repr(math.sqrt(5.0)),
                "mode": "fan_in",
                "nonlinearity": "leaky_relu",
                "expected_bound": 1.0 / math.sqrt(spec["shape"][1]),
                "per_tensor_seed": initialization_seed_for_key_v41a(spec["key"]),
                "seed_derivation": "sha256(schema_nul_sealed_seed_nul_peft_key)_u63_le",
            }
            if role == "A"
            else {
                "method": "torch.zeros",
                "positive_zero_bytes": True,
                "per_tensor_seed": None,
            }
        ),
    }
    record["record_sha256"] = canonical_sha256_v41a(record)
    return record


def validate_zero_zero_degeneracy_guard_v41a(
    tensors: dict[str, torch.Tensor],
) -> dict:
    """Reject any bilinear LoRA pair initialized at A=B=0."""
    expected = {item["key"]: item for item in tensor_specs_v41a()}
    if set(tensors) != set(expected):
        raise RuntimeError("v41a degeneracy guard received a different tensor surface")
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for key in sorted(tensors):
        tensor = tensors[key]
        spec = expected[key]
        if (
            tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or not tensor.is_contiguous()
            or list(tensor.shape) != spec["shape"]
        ):
            raise RuntimeError(f"v41a tensor metadata changed: {key}")
        pair = pairs.setdefault(spec["module"], {})
        pair[spec["role"]] = tensor
    records = []
    for module in sorted(pairs):
        pair = pairs[module]
        if set(pair) != {"A", "B"}:
            raise RuntimeError(f"v41a incomplete LoRA pair: {module}")
        a_tensor, b_tensor = pair["A"], pair["B"]
        if b_tensor.shape[1] != a_tensor.shape[0]:
            raise RuntimeError(f"v41a incompatible bilinear pair: {module}")
        a_all_zero = int(torch.count_nonzero(a_tensor).item()) == 0
        b_all_zero = int(torch.count_nonzero(b_tensor).item()) == 0
        if a_all_zero and b_all_zero:
            raise RuntimeError(
                f"v41a forbidden zero/zero antithetic LoRA initialization: {module}"
            )
        if a_all_zero or not b_all_zero:
            raise RuntimeError(
                f"v41a requires standard nonzero-A/exact-zero-B initialization: {module}"
            )
        records.append({
            "module": module,
            "a_all_zero": False,
            "b_all_zero": True,
            "simultaneous_zero_zero": False,
        })
    if len(records) != EXPECTED_MODULE_COUNT_V41A:
        raise RuntimeError("v41a degeneracy guard pair count changed")
    return {
        "schema": "matched-lora-zero-zero-degeneracy-guard-v41a",
        "passed": True,
        "module_pairs": len(records),
        "nonzero_a_pairs": len(records),
        "exact_zero_b_pairs": len(records),
        "zero_zero_pairs": 0,
        "reason": (
            "At A=B=0, (+sigma*B_noise)@(+sigma*A_noise) equals "
            "(-sigma*B_noise)@(-sigma*A_noise), so the antithetic central "
            "difference is identically zero."
        ),
        "records_sha256": canonical_sha256_v41a(records),
    }


def zero_zero_degeneracy_witness_v41a() -> dict:
    a_noise = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32)
    b_noise = torch.tensor([[5.0, -6.0], [7.0, 8.0]], dtype=torch.float32)
    sigma = 0.125
    plus = (sigma * b_noise) @ (sigma * a_noise)
    minus = (-sigma * b_noise) @ (-sigma * a_noise)
    central = 0.5 * (plus - minus)
    passed = torch.equal(plus, minus) and int(torch.count_nonzero(central)) == 0
    if not passed:
        raise RuntimeError("v41a zero/zero antithetic degeneracy witness changed")
    return {
        "schema": "matched-lora-zero-zero-degeneracy-witness-v41a",
        "passed": True,
        "plus_equals_minus": True,
        "central_nonzero_elements": 0,
        "sigma": sigma,
    }


def build_tensors_v41a() -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    tensors: dict[str, torch.Tensor] = {}
    records = []
    for spec in tensor_specs_v41a():
        shape = tuple(spec["shape"])
        if spec["role"] == "A":
            tensor = torch.empty(shape, dtype=torch.float32, device="cpu")
            generator = torch.Generator(device="cpu")
            generator.manual_seed(initialization_seed_for_key_v41a(spec["key"]))
            torch.nn.init.kaiming_uniform_(
                tensor, a=math.sqrt(5.0), generator=generator,
            )
        else:
            tensor = torch.zeros(shape, dtype=torch.float32, device="cpu")
        tensor = tensor.contiguous()
        tensors[spec["key"]] = tensor
        records.append(_tensor_record_v41a(spec, tensor))
    guard = validate_zero_zero_degeneracy_guard_v41a(tensors)
    if (
        len(tensors) != EXPECTED_TENSOR_COUNT_V41A
        or sum(tensor.numel() for tensor in tensors.values())
        != EXPECTED_ELEMENTS_V41A
    ):
        raise RuntimeError("v41a generated tensor inventory changed")
    return tensors, records, guard


def tensor_identity_v41a(records: list[dict]) -> dict:
    return {
        "schema": "matched-lora-tensor-identity-v41a",
        "sha256": canonical_sha256_v41a(records),
        "tensor_count": len(records),
        "elements": sum(item["elements"] for item in records),
        "bytes": sum(item["elements"] * 4 for item in records),
        "ordered_key_sha256": canonical_sha256_v41a(
            [item["key"] for item in records]
        ),
        "ordered_record_sha256": canonical_sha256_v41a(
            [item["record_sha256"] for item in records]
        ),
    }


def reopen_and_verify_v41a(
    path: Path,
    expected_tensors: dict[str, torch.Tensor],
    expected_records: list[dict],
    expected_metadata: dict[str, str],
) -> dict:
    path = Path(path).resolve()
    expected_keys = sorted(expected_tensors)
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}
    if keys != expected_keys or metadata != expected_metadata:
        raise RuntimeError("v41a safetensors header changed on reopen")
    reopened = load_file(path, device="cpu")
    if set(reopened) != set(expected_tensors):
        raise RuntimeError("v41a safetensors key inventory changed on reopen")
    reopened_records = []
    expected_by_key = {item["key"]: item for item in expected_records}
    for spec in tensor_specs_v41a():
        key = spec["key"]
        actual = reopened[key]
        expected = expected_tensors[key]
        if (
            actual.dtype != torch.float32
            or actual.device.type != "cpu"
            or not actual.is_contiguous()
            or actual.shape != expected.shape
            or not torch.equal(actual, expected)
        ):
            raise RuntimeError(f"v41a tensor bytes changed on reopen: {key}")
        record = _tensor_record_v41a(spec, actual)
        if record != expected_by_key[key]:
            raise RuntimeError(f"v41a tensor identity changed on reopen: {key}")
        reopened_records.append(record)
    guard = validate_zero_zero_degeneracy_guard_v41a(reopened)
    return {
        "schema": "matched-lora-safetensors-readback-v41a",
        "verified": True,
        "file_sha256": file_sha256_v41a(path),
        "file_bytes": path.stat().st_size,
        "metadata": metadata,
        "tensor_identity": tensor_identity_v41a(reopened_records),
        "degeneracy_guard": guard,
    }


def _atomic_bytes_exclusive_v41a(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_bytes(raw)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_exclusive_v41a(path: Path, value: dict) -> None:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes_exclusive_v41a(path, raw)


def build_artifact_v41a(output_directory: Path = DEFAULT_OUTPUT_V41A) -> dict:
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(output)
    config_bytes, config = validate_source_config_v41a()
    tensors, records, guard = build_tensors_v41a()
    surface = surface_identity_v41a()
    tensor_identity = tensor_identity_v41a(records)
    # safetensors serializes a multi-entry metadata mapping through a Rust
    # HashMap whose key order is intentionally randomized.  Keep the standard
    # PEFT metadata as one item so two executions produce the same complete
    # file bytes; all richer identities live in the self-hashed manifest.
    metadata = {"format": "pt"}
    output.mkdir(parents=True)
    weights = output / "adapter_model.safetensors"
    adapter_config = output / "adapter_config.json"
    manifest_path = output / "initialization_manifest_v41a.json"
    temporary_weights = output / f".adapter_model.safetensors.tmp-{os.getpid()}"
    linked_weights = False
    try:
        save_file(tensors, temporary_weights, metadata=metadata)
        os.link(temporary_weights, weights)
        linked_weights = True
        _atomic_bytes_exclusive_v41a(adapter_config, config_bytes)
        readback = reopen_and_verify_v41a(
            weights, tensors, records, metadata,
        )
        if (
            adapter_config.read_bytes() != config_bytes
            or file_sha256_v41a(adapter_config) != SOURCE_CONFIG_SHA256_V41A
            or readback["tensor_identity"] != tensor_identity
            or readback["degeneracy_guard"] != guard
        ):
            raise RuntimeError("v41a artifact readback consensus changed")
        manifest = {
            "schema": "matched-lora-initialization-manifest-v41a",
            "status": "complete_cpu_only_train_initialization",
            "sealed_initialization_seed": SEALED_INITIALIZATION_SEED_V41A,
            "seed_override_supported": False,
            "source_adapter_config": {
                "repository_relative_path": str(SOURCE_CONFIG_V41A.relative_to(ROOT)),
                "file_sha256": SOURCE_CONFIG_SHA256_V41A,
                "bytes_preserved_exactly": True,
                "semantic_summary": {
                    "rank": config["r"],
                    "alpha": config["lora_alpha"],
                    "scale": config["lora_alpha"] / config["r"],
                    "layers_to_transform": config["layers_to_transform"],
                    "bias": config["bias"],
                    "lora_dropout": config["lora_dropout"],
                    "init_lora_weights": config["init_lora_weights"],
                },
            },
            "implementation": {
                "repository_relative_path": str(Path(__file__).resolve().relative_to(ROOT)),
                "file_sha256": file_sha256_v41a(Path(__file__).resolve()),
                "torch_version": torch.__version__,
            },
            "artifact": {
                "weights_file": weights.name,
                "weights_file_sha256": readback["file_sha256"],
                "weights_file_bytes": readback["file_bytes"],
                "adapter_config_file": adapter_config.name,
                "adapter_config_file_sha256": file_sha256_v41a(adapter_config),
                "original_v37a_peft_key_namespace": True,
                "unscaled_fp32_master": True,
            },
            "initialization_contract": {
                "a": (
                    "independent deterministic torch.nn.init.kaiming_uniform_ "
                    "with a=sqrt(5), standard PEFT default"
                ),
                "b": "exact positive FP32 zero",
                "initial_effective_delta": "(alpha/r) * B @ A == 0",
                "per_tensor_seed_derivation": (
                    "sha256(schema_nul_sealed_seed_nul_peft_key)_u63_le"
                ),
            },
            "surface": surface,
            "tensor_identity": tensor_identity,
            "tensor_records": records,
            "zero_zero_antithetic_degeneracy_guard": guard,
            "zero_zero_antithetic_degeneracy_witness": (
                zero_zero_degeneracy_witness_v41a()
            ),
            "readback": readback,
            "dataset_or_training_examples_accessed": False,
            "shadow_ood_holdout_or_heldout_accessed": False,
            "gpu_accessed": False,
            "evaluation_performed": False,
        }
        manifest["content_sha256_before_self_field"] = canonical_sha256_v41a(manifest)
        _atomic_json_exclusive_v41a(manifest_path, manifest)
        reopened_manifest = json.loads(manifest_path.read_text())
        content = reopened_manifest.pop("content_sha256_before_self_field", None)
        if content != canonical_sha256_v41a(reopened_manifest):
            raise RuntimeError("v41a manifest self-hash changed on reopen")
        return manifest
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        adapter_config.unlink(missing_ok=True)
        if linked_weights:
            weights.unlink(missing_ok=True)
        try:
            output.rmdir()
        except OSError:
            pass
        raise
    finally:
        temporary_weights.unlink(missing_ok=True)


def parser_v41a() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_V41A))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_v41a().parse_args(argv)
    manifest = build_artifact_v41a(Path(args.output))
    output = Path(args.output).resolve()
    print(json.dumps({
        "directory": str(output),
        "weights_sha256": manifest["artifact"]["weights_file_sha256"],
        "tensor_identity_sha256": manifest["tensor_identity"]["sha256"],
        "manifest_file_sha256": file_sha256_v41a(
            output / "initialization_manifest_v41a.json"
        ),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "zero_zero_pairs": manifest[
            "zero_zero_antithetic_degeneracy_guard"
        ]["zero_zero_pairs"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
