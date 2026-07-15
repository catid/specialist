#!/usr/bin/env python3
"""Four-GPU equal-unit LoRA SFT from the sealed V41A initialization.

V42A is the matched-initialization SFT control for the canonical LoRA ES arm.
It deliberately accepts no evaluation input.  The only adapter it will load is
the content-addressed, unscaled FP32 V41A master, and it verifies every loaded
tensor before constructing the Trainer/optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

import torch
from datasets import Dataset
from peft import PeftModel
from peft.utils import get_peft_model_state_dict
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

import sft_lora as base
import sft_lora_equal_unit_v37a as objective
from qa_quality import qa_pair_from_record


ROOT = Path(__file__).resolve().parent
INITIAL_ADAPTER_V42A = (
    ROOT / "experiments/eggroll_es_hpo/initial_adapters/"
    "matched_lora_initialization_v41a_seed20260715041"
).resolve()
INITIAL_WEIGHTS_SHA256_V42A = (
    "29fe0beead8a491cf06e9f562a1838d9c44e94a74e6a4024549e87f10557111f"
)
INITIAL_CONFIG_SHA256_V42A = (
    "ede582c12e82fb50eb97ac934ff08eb553a79d2c2d999235abcd8b29795b1d52"
)
INITIAL_MANIFEST_SHA256_V42A = (
    "a2fa79e6ac06f75743d3fee8f5c0b1aabe6bb83b52b05910ed6460438e2640a2"
)
INITIAL_MANIFEST_CONTENT_SHA256_V42A = (
    "5f885b415302c4e748e19f4d535f1e57ff87f785370f653b345cbdfafda3224b"
)
INITIAL_TENSOR_IDENTITY_SHA256_V42A = (
    "2dcb9ab45ec26c7041b9782a30fe3c82b987b605b6b0bd95ab5b905b1371ae2e"
)
INITIAL_SURFACE_SHA256_V42A = (
    "6c4c219f92fba3d7d01e08f439b7b1f21a1d07bc9893cdd18f860994668e0fb8"
)
INITIAL_SEED_V42A = 20_260_715_041
EXPECTED_RANK_V42A = 32
EXPECTED_ALPHA_V42A = 64
EXPECTED_LAYERS_V42A = [20, 21, 22, 23]
EXPECTED_TENSORS_V42A = 70
EXPECTED_MODULES_V42A = 35
EXPECTED_ELEMENTS_V42A = 4_528_128
EXPECTED_A_ELEMENTS_V42A = 2_359_296
EXPECTED_B_ELEMENTS_V42A = 2_168_832


EqualUnitCollator = objective.EqualUnitCollator
EqualUnitTrainer = objective.EqualUnitTrainer
assign_equal_unit_weights = objective.assign_equal_unit_weights
load_records = objective.load_records


def canonical_sha256_v42a(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256_v42a(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256_v42a(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _require_exact_initial_adapter_path_v42a(path: Path) -> Path:
    resolved = Path(path).resolve()
    if resolved != INITIAL_ADAPTER_V42A:
        raise ValueError("V42A only permits the sealed matched initialization")
    return resolved


def validate_initialization_artifact_v42a(path: Path) -> dict:
    """Verify the complete sealed V41A source without opening any dataset."""
    directory = _require_exact_initial_adapter_path_v42a(path)
    weights_path = directory / "adapter_model.safetensors"
    config_path = directory / "adapter_config.json"
    manifest_path = directory / "initialization_manifest_v41a.json"
    observed_files = {
        "weights": file_sha256_v42a(weights_path),
        "config": file_sha256_v42a(config_path),
        "manifest": file_sha256_v42a(manifest_path),
    }
    expected_files = {
        "weights": INITIAL_WEIGHTS_SHA256_V42A,
        "config": INITIAL_CONFIG_SHA256_V42A,
        "manifest": INITIAL_MANIFEST_SHA256_V42A,
    }
    if observed_files != expected_files:
        raise RuntimeError("V42A matched initialization file identity changed")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    content_hash = manifest.get("content_sha256_before_self_field")
    content = {
        key: value for key, value in manifest.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        content_hash != INITIAL_MANIFEST_CONTENT_SHA256_V42A
        or canonical_sha256_v42a(content) != content_hash
        or manifest.get("schema") != "matched-lora-initialization-manifest-v41a"
        or manifest.get("status") != "complete_cpu_only_train_initialization"
        or manifest.get("sealed_initialization_seed") != INITIAL_SEED_V42A
        or manifest.get("seed_override_supported") is not False
    ):
        raise RuntimeError("V42A matched initialization manifest changed")

    artifact = manifest.get("artifact", {})
    source_config = manifest.get("source_adapter_config", {})
    semantics = source_config.get("semantic_summary", {})
    surface = manifest.get("surface", {})
    identity = manifest.get("tensor_identity", {})
    guard = manifest.get("zero_zero_antithetic_degeneracy_guard", {})
    if (
        artifact.get("weights_file_sha256") != INITIAL_WEIGHTS_SHA256_V42A
        or artifact.get("adapter_config_file_sha256")
        != INITIAL_CONFIG_SHA256_V42A
        or artifact.get("unscaled_fp32_master") is not True
        or artifact.get("original_v37a_peft_key_namespace") is not True
        or source_config.get("file_sha256") != INITIAL_CONFIG_SHA256_V42A
        or semantics.get("rank") != EXPECTED_RANK_V42A
        or semantics.get("alpha") != EXPECTED_ALPHA_V42A
        or semantics.get("scale") != 2.0
        or semantics.get("layers_to_transform") != EXPECTED_LAYERS_V42A
        or semantics.get("bias") != "none"
        or semantics.get("lora_dropout") != 0.0
        or surface.get("sha256") != INITIAL_SURFACE_SHA256_V42A
        or surface.get("module_count") != EXPECTED_MODULES_V42A
        or surface.get("tensor_count") != EXPECTED_TENSORS_V42A
        or surface.get("elements") != EXPECTED_ELEMENTS_V42A
        or surface.get("a_elements") != EXPECTED_A_ELEMENTS_V42A
        or surface.get("b_elements") != EXPECTED_B_ELEMENTS_V42A
        or identity.get("sha256") != INITIAL_TENSOR_IDENTITY_SHA256_V42A
        or identity.get("tensor_count") != EXPECTED_TENSORS_V42A
        or identity.get("elements") != EXPECTED_ELEMENTS_V42A
        or guard.get("passed") is not True
        or guard.get("zero_zero_pairs") != 0
        or guard.get("nonzero_a_pairs") != EXPECTED_MODULES_V42A
        or guard.get("exact_zero_b_pairs") != EXPECTED_MODULES_V42A
    ):
        raise RuntimeError("V42A matched initialization semantic binding changed")

    tensor_records = manifest.get("tensor_records", [])
    if (
        len(tensor_records) != EXPECTED_TENSORS_V42A
        or canonical_sha256_v42a(tensor_records) != identity["sha256"]
    ):
        raise RuntimeError("V42A matched initialization tensor records changed")
    expected_by_key = {record["key"]: record for record in tensor_records}
    if len(expected_by_key) != EXPECTED_TENSORS_V42A:
        raise RuntimeError("V42A matched initialization has duplicate tensor keys")
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        metadata = handle.metadata() or {}
        if keys != sorted(expected_by_key) or metadata != {"format": "pt"}:
            raise RuntimeError("V42A matched initialization tensor header changed")
        for key in keys:
            tensor = handle.get_tensor(key)
            record = expected_by_key[key]
            if (
                tensor.dtype != torch.float32
                or list(tensor.shape) != record.get("shape")
                or tensor.numel() != record.get("elements")
                or tensor_sha256_v42a(tensor) != record.get("tensor_sha256")
                or int(torch.count_nonzero(tensor).item())
                != record.get("nonzero_elements")
            ):
                raise RuntimeError(
                    f"V42A matched initialization tensor changed: {key}"
                )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("r") != EXPECTED_RANK_V42A
        or config.get("lora_alpha") != EXPECTED_ALPHA_V42A
        or config.get("layers_to_transform") != EXPECTED_LAYERS_V42A
        or config.get("bias") != "none"
        or config.get("lora_dropout") != 0.0
    ):
        raise RuntimeError("V42A matched initialization PEFT config changed")

    return {
        "schema": "specialist-matched-lora-source-audit-v42a",
        "verified": True,
        "directory": str(directory),
        "sealed_initialization_seed": INITIAL_SEED_V42A,
        "weights_file_sha256": INITIAL_WEIGHTS_SHA256_V42A,
        "weights_file_bytes": weights_path.stat().st_size,
        "adapter_config_file_sha256": INITIAL_CONFIG_SHA256_V42A,
        "manifest_file_sha256": INITIAL_MANIFEST_SHA256_V42A,
        "manifest_content_sha256": INITIAL_MANIFEST_CONTENT_SHA256_V42A,
        "tensor_identity_sha256": INITIAL_TENSOR_IDENTITY_SHA256_V42A,
        "surface_identity_sha256": INITIAL_SURFACE_SHA256_V42A,
        "tensor_count": EXPECTED_TENSORS_V42A,
        "module_count": EXPECTED_MODULES_V42A,
        "elements": EXPECTED_ELEMENTS_V42A,
        "rank": EXPECTED_RANK_V42A,
        "alpha": EXPECTED_ALPHA_V42A,
        "scale": 2.0,
        "layers": EXPECTED_LAYERS_V42A,
        "unscaled_fp32_master": True,
        "original_v37a_peft_key_namespace": True,
        "zero_zero_pairs": 0,
    }


def validate_loaded_adapter_state_v42a(
    loaded_state: Mapping[str, torch.Tensor],
    source_directory: Path,
    source_audit: dict,
) -> dict:
    """Prove that PEFT's trainable state exactly equals the sealed FP32 source."""
    if source_audit != validate_initialization_artifact_v42a(source_directory):
        raise RuntimeError("V42A source audit changed before model load validation")
    source = load_file(
        str(Path(source_directory) / "adapter_model.safetensors"), device="cpu"
    )
    if set(loaded_state) != set(source):
        raise RuntimeError("V42A loaded adapter key inventory differs from source")
    records = []
    for key in sorted(source):
        expected = source[key]
        actual = loaded_state[key].detach().to(device="cpu")
        if (
            actual.dtype != torch.float32
            or actual.shape != expected.shape
            or not actual.is_contiguous()
            or not torch.equal(actual, expected)
        ):
            raise RuntimeError(f"V42A loaded adapter differs from source: {key}")
        records.append({
            "key": key,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "elements": actual.numel(),
            "tensor_sha256": tensor_sha256_v42a(actual),
        })
    return {
        "schema": "specialist-matched-lora-loaded-state-audit-v42a",
        "verified_before_optimizer_construction": True,
        "matches_source_tensor_bytes": True,
        "source_weights_file_sha256": source_audit["weights_file_sha256"],
        "source_tensor_identity_sha256": source_audit[
            "tensor_identity_sha256"
        ],
        "loaded_tensor_count": len(records),
        "loaded_elements": sum(item["elements"] for item in records),
        "loaded_dtype": "torch.float32",
        "loaded_records_sha256": canonical_sha256_v42a(records),
    }


def validate_recipe_arguments_v42a(arguments: argparse.Namespace) -> list[int]:
    target_layers = base.parse_target_layers(arguments.target_layers)
    if (
        arguments.rank != EXPECTED_RANK_V42A
        or target_layers != EXPECTED_LAYERS_V42A
        or arguments.expected_trainable_elements != EXPECTED_ELEMENTS_V42A
        or arguments.expected_trainable_tensors != EXPECTED_TENSORS_V42A
        or arguments.expected_world_size != 4
    ):
        raise ValueError("V42A matched SFT recipe surface changed")
    if (
        arguments.initial_adapter_weights_sha256
        != INITIAL_WEIGHTS_SHA256_V42A
        or arguments.initial_adapter_config_sha256
        != INITIAL_CONFIG_SHA256_V42A
        or arguments.initial_adapter_manifest_sha256
        != INITIAL_MANIFEST_SHA256_V42A
        or arguments.initial_adapter_manifest_content_sha256
        != INITIAL_MANIFEST_CONTENT_SHA256_V42A
        or arguments.initial_adapter_tensor_identity_sha256
        != INITIAL_TENSOR_IDENTITY_SHA256_V42A
    ):
        raise ValueError("V42A matched SFT initialization argument changed")
    _require_exact_initial_adapter_path_v42a(Path(arguments.initial_adapter))
    return target_layers


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", required=True)
    result.add_argument("--data-sha256", required=True)
    result.add_argument("--data-rows", required=True, type=int)
    result.add_argument("--expected-conflict-units", required=True, type=int)
    result.add_argument("--expected-weight-identity-sha256", required=True)
    result.add_argument("--initial-adapter", required=True)
    result.add_argument("--initial-adapter-weights-sha256", required=True)
    result.add_argument("--initial-adapter-config-sha256", required=True)
    result.add_argument("--initial-adapter-manifest-sha256", required=True)
    result.add_argument(
        "--initial-adapter-manifest-content-sha256", required=True
    )
    result.add_argument(
        "--initial-adapter-tensor-identity-sha256", required=True
    )
    result.add_argument("--out", required=True)
    result.add_argument("--epochs", type=float, default=3.0)
    result.add_argument("--rank", type=int, default=32)
    result.add_argument("--per-device-batch-size", type=int, default=7)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--max-length", type=int, default=1024)
    result.add_argument("--save-steps", type=int, default=16)
    result.add_argument("--target-layers", default="20,21,22,23")
    result.add_argument(
        "--expected-trainable-elements", type=int, default=EXPECTED_ELEMENTS_V42A
    )
    result.add_argument(
        "--expected-trainable-tensors", type=int, default=EXPECTED_TENSORS_V42A
    )
    result.add_argument("--expected-world-size", type=int, default=4)
    result.add_argument("--attn-implementation", default="sdpa")
    return result


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    target_layers = validate_recipe_arguments_v42a(arguments)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != arguments.expected_world_size:
        raise ValueError(
            f"expected world size {arguments.expected_world_size}, got {world_size}"
        )
    source_audit = validate_initialization_artifact_v42a(
        Path(arguments.initial_adapter)
    )
    if base.file_sha256(arguments.data) != arguments.data_sha256:
        raise ValueError("content-addressed equal-unit SFT data changed")
    records = load_records(arguments.data)
    if len(records) != arguments.data_rows:
        raise ValueError("equal-unit SFT data row count changed")
    weights, weighting_audit = assign_equal_unit_weights(records)
    if (
        weighting_audit["conflict_units"] != arguments.expected_conflict_units
        or weighting_audit["identity_sha256"]
        != arguments.expected_weight_identity_sha256
    ):
        raise ValueError("equal-unit weighting identity changed")

    tokenizer = AutoTokenizer.from_pretrained(base.BASE)
    encoded = []
    for record, weight in zip(records, weights):
        item = base.encode_pair(
            tokenizer,
            qa_pair_from_record(record),
            arguments.max_length,
            "es_exact",
        )
        item["example_weight"] = weight
        encoded.append(item)
    encoding_audit = {
        "prompt_mode": "es_exact",
        "eos_appended": False,
        "train_prompt_tokens": sum(
            item.pop("prompt_token_count") for item in encoded
        ),
        "train_answer_tokens": sum(
            item.pop("answer_token_count") for item in encoded
        ),
        "train_rows": len(encoded),
    }
    print(json.dumps({"encoding_audit": encoding_audit}, sort_keys=True), flush=True)
    print(
        json.dumps({"weighting_audit": weighting_audit}, sort_keys=True),
        flush=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base.BASE,
        dtype=torch.bfloat16,
        attn_implementation=arguments.attn_implementation,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    model = PeftModel.from_pretrained(
        base_model,
        arguments.initial_adapter,
        is_trainable=True,
        autocast_adapter_dtype=True,
    )
    loaded_state = get_peft_model_state_dict(
        model, adapter_name="default", save_embedding_layers=False
    )
    loaded_audit = validate_loaded_adapter_state_v42a(
        loaded_state, Path(arguments.initial_adapter), source_audit
    )
    initialization_audit = {
        "schema": "specialist-matched-lora-initialization-runtime-v42a",
        "source": source_audit,
        "loaded": loaded_audit,
    }
    print(
        json.dumps(
            {"source_initialization_audit": initialization_audit},
            sort_keys=True,
        ),
        flush=True,
    )

    trainable = [
        {"name": name, "elements": parameter.numel()}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    inventory = {
        "target_layers": target_layers,
        "tensor_count": len(trainable),
        "elements": sum(item["elements"] for item in trainable),
        "identity_sha256": canonical_sha256_v42a(trainable),
    }
    if (
        inventory["elements"] != arguments.expected_trainable_elements
        or inventory["tensor_count"] != arguments.expected_trainable_tensors
    ):
        raise ValueError("equal-unit SFT trainable inventory changed")
    print(json.dumps({"trainable_inventory": inventory}, sort_keys=True), flush=True)

    training_args = TrainingArguments(
        output_dir=arguments.out,
        num_train_epochs=arguments.epochs,
        per_device_train_batch_size=arguments.per_device_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=arguments.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=8,
        save_strategy="steps",
        save_steps=arguments.save_steps,
        save_total_limit=3,
        bf16=True,
        report_to=[],
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        seed=arguments.seed,
    )
    trainer = EqualUnitTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(encoded),
        data_collator=EqualUnitCollator(tokenizer),
    )
    trainer.train()
    if int(os.environ.get("RANK", "0")) == 0:
        final = arguments.out + "/final"
        model.save_pretrained(final)
        print("saved", final, flush=True)


if __name__ == "__main__":
    main()
