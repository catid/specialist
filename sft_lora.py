#!/usr/bin/env python
"""LoRA SFT control for the ES comparison with assistant-only supervision.

The default encoding reproduces EGGROLL-ES's exact prompt+raw-answer token
sequence, masks every prompt token, and does not append or score EOS.  The
default loss averages answer-token NLL within each example before averaging
examples, matching the ES dense-gold reward's reduction.  Run under torchrun
for multi-GPU DDP.
"""
import hashlib
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)

from qa_quality import qa_pair_from_record
from train_eggroll_es_specialist import specialist_template

BASE = "/home/catid/specialist/models/Qwen3.6-35B-A3B"
MAXLEN = 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token_ids(value):
    """Normalize tokenizer list/BatchEncoding returns without guessing."""
    if isinstance(value, dict) or hasattr(value, "keys"):
        if "input_ids" not in value:
            raise ValueError("tokenizer mapping omitted input_ids")
        value = value["input_ids"]
    if isinstance(value, torch.Tensor):
        value = value.tolist()
    if (
        not isinstance(value, (list, tuple))
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError("tokenizer returned invalid token IDs")
    return list(value)


def encode_pair(tokenizer, pair, max_length, prompt_mode="es_exact"):
    question, answer = pair
    if prompt_mode == "es_exact":
        prompt = specialist_template(question)
        prompt_ids = token_ids(tokenizer.encode(prompt, add_special_tokens=False))
        input_ids = token_ids(
            tokenizer.encode(prompt + answer, add_special_tokens=False)
        )
    elif prompt_mode == "chat_template":
        user = [{
            "role": "user",
            "content": (
                "Answer this question about rope bondage briefly and factually "
                f"(one sentence):\n\n{question}"
            ),
        }]
        messages = user + [{"role": "assistant", "content": answer}]
        prompt_ids = token_ids(tokenizer.apply_chat_template(
            user, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
        ))
        input_ids = token_ids(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            enable_thinking=False,
        ))
    else:
        raise ValueError(f"unsupported prompt mode: {prompt_mode}")
    if not prompt_ids or input_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("SFT prompt/answer tokenizer boundary mismatch")
    answer_ids = input_ids[len(prompt_ids):]
    if not answer_ids:
        raise ValueError("SFT answer has no aligned tokens")
    if len(input_ids) > max_length:
        raise ValueError(
            f"SFT example has {len(input_ids)} tokens above cap {max_length}"
        )
    labels = [-100] * len(prompt_ids) + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "prompt_token_count": len(prompt_ids),
        "answer_token_count": len(answer_ids),
    }


class ExampleMeanTrainer(Trainer):
    """Match ES: mean answer-token NLL per example, then mean examples."""

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        del num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shifted_logits = logits[..., :-1, :].contiguous().float()
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
            raise RuntimeError("SFT batch contains an example with no answer tokens")
        per_example = (flat * mask).sum(dim=-1) / counts
        loss = per_example.mean()
        return (loss, outputs) if return_outputs else loss


def parse_target_layers(value):
    if value is None or value.strip().casefold() in {"", "all"}:
        return None
    layers = sorted({int(item) for item in value.split(",")})
    if not layers or any(layer < 0 or layer >= 40 for layer in layers):
        raise ValueError("target layers must be comma-separated values in [0, 39]")
    return layers

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["/home/catid/specialist/data/train_qa_curated_v1.jsonl"])
    ap.add_argument("--eval-data", nargs="+", default=[])
    ap.add_argument("--out", default="/home/catid/specialist/models/lora-rope-sft")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.0)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-length", type=int, default=MAXLEN)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--data-sha256", default=None)
    ap.add_argument("--data-rows", type=int, default=None)
    ap.add_argument("--expected-world-size", type=int, default=None)
    ap.add_argument(
        "--prompt-mode", choices=("es_exact", "chat_template"),
        default="es_exact",
    )
    ap.add_argument(
        "--loss-mode", choices=("example_mean", "token_mean"),
        default="example_mean",
    )
    ap.add_argument(
        "--target-layers", default="all",
        help="'all' or comma-separated transformer layer indices",
    )
    ap.add_argument("--expected-trainable-elements", type=int, default=None)
    ap.add_argument("--expected-trainable-tensors", type=int, default=None)
    ap.add_argument("--resume", default=None,
                    help="Trainer checkpoint path, or 'latest' for automatic resume")
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--attn-implementation", default="sdpa",
                    choices=("sdpa", "eager", "flash_attention_2"))
    a = ap.parse_args()
    OUT = a.out
    if a.expected_world_size is not None:
        observed_world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if observed_world_size != a.expected_world_size:
            raise ValueError(
                f"expected world size {a.expected_world_size}, got {observed_world_size}"
            )
    if a.data_sha256 is not None:
        if len(a.data) != 1 or file_sha256(a.data[0]) != a.data_sha256:
            raise ValueError("content-addressed SFT data binding changed")
    tok = AutoTokenizer.from_pretrained(BASE)

    def load_records(paths):
        records = []
        for path in paths:
            for line_number, line in enumerate(open(path), 1):
                item = json.loads(line)
                try:
                    pair = qa_pair_from_record(item)
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
                if pair is None:
                    raise ValueError(
                        f"{path}:{line_number}: unsupported QA serialization")
                records.append(pair)
        return records

    train_pairs = load_records(a.data)
    if a.data_rows is not None and len(train_pairs) != a.data_rows:
        raise ValueError("content-addressed SFT data row count changed")
    train_encoded = [
        encode_pair(tok, pair, a.max_length, a.prompt_mode)
        for pair in train_pairs
    ]
    eval_encoded = [
        encode_pair(tok, pair, a.max_length, a.prompt_mode)
        for pair in load_records(a.eval_data)
    ]
    prompt_tokens = sum(item["prompt_token_count"] for item in train_encoded)
    answer_tokens = sum(item["answer_token_count"] for item in train_encoded)
    for item in (*train_encoded, *eval_encoded):
        item.pop("prompt_token_count")
        item.pop("answer_token_count")
    print(f"training examples: {len(train_encoded)}; eval: {len(eval_encoded)}",
          flush=True)
    print(json.dumps({
        "encoding_audit": {
            "prompt_mode": a.prompt_mode,
            "eos_appended": a.prompt_mode != "es_exact",
            "train_prompt_tokens": prompt_tokens,
            "train_answer_tokens": answer_tokens,
            "train_rows": len(train_encoded),
        }
    }, sort_keys=True), flush=True)
    ds = Dataset.from_list(train_encoded)
    eval_ds = Dataset.from_list(eval_encoded) if eval_encoded else None

    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation=a.attn_implementation)
    model.config.use_cache = False
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    target_layers = parse_target_layers(a.target_layers)
    if a.lora_dropout != 0.0:
        raise ValueError(
            "router target_parameters require zero LoRA dropout in this PEFT version"
        )
    router_layers = target_layers if target_layers is not None else list(range(40))
    lora_kwargs = dict(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=a.lora_dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z",
                        "out_proj"],
        target_parameters=[
            f"model.layers.{layer}.mlp.gate.weight" for layer in router_layers
        ],
    )
    if target_layers is not None:
        lora_kwargs.update(
            layers_to_transform=target_layers,
            layers_pattern="layers",
        )
    lora = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    trainable = [
        {"name": name, "elements": parameter.numel()}
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_elements = sum(item["elements"] for item in trainable)
    if (
        a.expected_trainable_elements is not None
        and trainable_elements != a.expected_trainable_elements
    ):
        raise ValueError("SFT trainable element inventory changed")
    if (
        a.expected_trainable_tensors is not None
        and len(trainable) != a.expected_trainable_tensors
    ):
        raise ValueError("SFT trainable tensor inventory changed")
    print(json.dumps({
        "trainable_inventory": {
            "target_layers": target_layers,
            "tensor_count": len(trainable),
            "elements": trainable_elements,
            "identity_sha256": hashlib.sha256(json.dumps(
                trainable, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest(),
        }
    }, sort_keys=True), flush=True)

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.per_device_batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.learning_rate,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="steps", save_steps=a.save_steps,
        save_total_limit=3, bf16=True,
        report_to=[], ddp_find_unused_parameters=False,
        gradient_checkpointing=a.gradient_checkpointing,
        dataloader_num_workers=2, dataloader_pin_memory=True,
        group_by_length=True, seed=a.seed,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=a.save_steps if eval_ds is not None else None,
    )
    trainer_class = ExampleMeanTrainer if a.loss_mode == "example_mean" else Trainer
    trainer = trainer_class(
        model=model, args=args, train_dataset=ds, eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tok, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8))
    resume = True if a.resume == "latest" else a.resume
    trainer.train(resume_from_checkpoint=resume)
    if int(os.environ.get("RANK", 0)) == 0:
        model.save_pretrained(OUT + "/final")
        print("saved", OUT + "/final")

if __name__ == "__main__":
    main()
