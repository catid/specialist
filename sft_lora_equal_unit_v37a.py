#!/usr/bin/env python3
"""Four-GPU sparse LoRA SFT with a frozen equal-conflict-unit objective.

This is the V37A quality-control training arm.  It deliberately has no
evaluation-data argument: optimization sees only one content-addressed fold
train file.  Rows in the same conservative conflict unit share a total weight
of one, preventing large source/near-duplicate clusters from dominating the
objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

import build_train_shadow_folds_v37a as shadow
import sft_lora as base
from qa_quality import qa_pair_from_record


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def assign_equal_unit_weights(records: list[dict]) -> tuple[list[float], dict]:
    """Return row weights whose ordinary row mean is the equal-unit mean."""
    units = shadow.build_conflict_units(records)
    unit_count = len(units)
    row_count = len(records)
    if not units or not records:
        raise ValueError("equal-unit SFT requires nonempty records")
    weights: list[float | None] = [None] * row_count
    identities = []
    for unit in units:
        unit_rows = len(unit["indices"])
        for index in unit["indices"]:
            if weights[index] is not None:
                raise RuntimeError("row appeared in more than one conflict unit")
            weights[index] = row_count / (unit_count * unit_rows)
            identities.append({
                "fact_id": records[index]["fact_id"],
                "unit_identity_sha256": unit["identity_sha256"],
                "unit_rows": unit_rows,
                "weight_numerator": row_count,
                "weight_denominator": unit_count * unit_rows,
            })
    if any(item is None for item in weights):
        raise RuntimeError("conflict-unit weighting omitted a row")
    exact_unit_mass = {
        unit["identity_sha256"]: sum(
            weights[index] / row_count for index in unit["indices"]
        ) for unit in units
    }
    target_mass = 1.0 / unit_count
    if any(
        abs(value - target_mass) > 1e-15 for value in exact_unit_mass.values()
    ):
        raise RuntimeError("equal-unit coefficient construction changed")
    identities.sort(key=lambda item: item["fact_id"])
    concrete = [float(item) for item in weights]
    audit = {
        "schema": "specialist-equal-conflict-unit-weighting-v37a",
        "rows": row_count,
        "conflict_units": unit_count,
        "ordinary_row_mean_weight": sum(concrete) / row_count,
        "minimum_row_weight": min(concrete),
        "maximum_row_weight": max(concrete),
        "unit_objective_mass": target_mass,
        "identity_sha256": canonical_sha256(identities),
    }
    return concrete, audit


class EqualUnitCollator:
    def __init__(self, tokenizer):
        self.base = DataCollatorForSeq2Seq(
            tokenizer,
            padding=True,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        )

    def __call__(self, features: list[dict]) -> dict:
        features = [dict(item) for item in features]
        weights = torch.tensor(
            [item.pop("example_weight") for item in features],
            dtype=torch.float32,
        )
        batch = self.base(features)
        batch["example_weight"] = weights
        return batch


class EqualUnitTrainer(Trainer):
    """Answer-token mean per row, then frozen equal-unit row weights."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None,
    ):
        del num_items_in_batch
        labels = inputs.pop("labels")
        weights = inputs.pop("example_weight").float()
        outputs = model(**inputs)
        shifted_logits = outputs.logits[..., :-1, :].contiguous().float()
        shifted_labels = labels[..., 1:].contiguous()
        flat = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shifted_labels)
        mask = shifted_labels.ne(-100)
        counts = mask.sum(dim=-1)
        if torch.any(counts <= 0):
            raise RuntimeError("equal-unit SFT batch has an empty answer")
        per_example = (flat * mask).sum(dim=-1) / counts
        if weights.shape != per_example.shape or torch.any(weights <= 0):
            raise RuntimeError("equal-unit SFT batch has invalid row weights")
        loss = (per_example * weights).mean()
        return (loss, outputs) if return_outputs else loss


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--data", required=True)
    result.add_argument("--data-sha256", required=True)
    result.add_argument("--data-rows", required=True, type=int)
    result.add_argument("--expected-conflict-units", required=True, type=int)
    result.add_argument("--expected-weight-identity-sha256", required=True)
    result.add_argument("--out", required=True)
    result.add_argument("--epochs", type=float, default=3.0)
    result.add_argument("--rank", type=int, default=32)
    result.add_argument("--per-device-batch-size", type=int, default=7)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--max-length", type=int, default=1024)
    result.add_argument("--save-steps", type=int, default=16)
    result.add_argument("--target-layers", default="20,21,22,23")
    result.add_argument("--expected-trainable-elements", type=int, default=4_528_128)
    result.add_argument("--expected-trainable-tensors", type=int, default=70)
    result.add_argument("--expected-world-size", type=int, default=4)
    result.add_argument("--attn-implementation", default="sdpa")
    return result


def load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            record = json.loads(line)
            pair = qa_pair_from_record(record)
            if pair is None:
                raise ValueError(f"unsupported QA record at line {line_number}")
            records.append(record)
    return records


def main(argv: list[str] | None = None) -> None:
    arguments = parser().parse_args(argv)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != arguments.expected_world_size:
        raise ValueError(
            f"expected world size {arguments.expected_world_size}, got {world_size}"
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
            tokenizer, qa_pair_from_record(record), arguments.max_length, "es_exact",
        )
        item["example_weight"] = weight
        encoded.append(item)
    encoding_audit = {
        "prompt_mode": "es_exact",
        "eos_appended": False,
        "train_prompt_tokens": sum(item.pop("prompt_token_count") for item in encoded),
        "train_answer_tokens": sum(item.pop("answer_token_count") for item in encoded),
        "train_rows": len(encoded),
    }
    print(json.dumps({"encoding_audit": encoding_audit}, sort_keys=True), flush=True)
    print(json.dumps({"weighting_audit": weighting_audit}, sort_keys=True), flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        base.BASE,
        dtype=torch.bfloat16,
        attn_implementation=arguments.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    target_layers = base.parse_target_layers(arguments.target_layers)
    if target_layers is None:
        raise ValueError("V37A requires an explicit layer list")
    lora = LoraConfig(
        r=arguments.rank,
        lora_alpha=2 * arguments.rank,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
            "down_proj", "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
            "out_proj",
        ],
        target_parameters=[
            f"model.layers.{layer}.mlp.gate.weight" for layer in target_layers
        ],
        layers_to_transform=target_layers,
        layers_pattern="layers",
    )
    model = get_peft_model(model, lora)
    trainable = [
        {"name": name, "elements": parameter.numel()}
        for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    inventory = {
        "target_layers": target_layers,
        "tensor_count": len(trainable),
        "elements": sum(item["elements"] for item in trainable),
        "identity_sha256": canonical_sha256(trainable),
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
