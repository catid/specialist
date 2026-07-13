#!/usr/bin/env python
"""LoRA SFT control for the ES comparison with assistant-only supervision.

QA records are rendered through the checkpoint's real chat template. Prompt
tokens are masked from the loss, EOS is emitted exactly once, and checkpoints
can be resumed safely. Run under torchrun for multi-GPU DDP.
"""
import json, os
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)

from qa_quality import qa_pair_from_record

BASE = "/home/catid/specialist/models/Qwen3.6-35B-A3B"
MAXLEN = 1024

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["/home/catid/specialist/data/train_qa_verified_leakfree_v2.jsonl"])
    ap.add_argument("--eval-data", nargs="+", default=[])
    ap.add_argument("--out", default="/home/catid/specialist/models/lora-rope-sft")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=MAXLEN)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--resume", default=None,
                    help="Trainer checkpoint path, or 'latest' for automatic resume")
    ap.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--attn-implementation", default="sdpa",
                    choices=("sdpa", "eager", "flash_attention_2"))
    a = ap.parse_args()
    OUT = a.out
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

    def encode(pair):
        question, answer = pair
        user = [{"role": "user", "content":
                 "Answer this question about rope bondage briefly and factually "
                 f"(one sentence):\n\n{question}"}]
        messages = user + [{"role": "assistant", "content": answer}]
        prompt_ids = tok.apply_chat_template(
            user, tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        input_ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            enable_thinking=False)
        if len(input_ids) > a.max_length:
            # QA samples are short; an overlong record is safer to drop than
            # to truncate away either the question or supervised answer.
            return None
        labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids),
                "labels": labels}

    train_encoded = [encoded for pair in load_records(a.data)
                     if (encoded := encode(pair)) is not None]
    eval_encoded = [encoded for pair in load_records(a.eval_data)
                    if (encoded := encode(pair)) is not None]
    print(f"training examples: {len(train_encoded)}; eval: {len(eval_encoded)}",
          flush=True)
    ds = Dataset.from_list(train_encoded)
    eval_ds = Dataset.from_list(eval_encoded) if eval_encoded else None

    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, attn_implementation=a.attn_implementation)
    model.config.use_cache = False
    if a.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                        "out_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=a.epochs, per_device_train_batch_size=4,
        gradient_accumulation_steps=a.grad_accum, learning_rate=1e-4,
        lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=10, save_strategy="steps", save_steps=a.save_steps,
        save_total_limit=3, bf16=True,
        report_to=[], ddp_find_unused_parameters=False,
        gradient_checkpointing=a.gradient_checkpointing,
        dataloader_num_workers=2, dataloader_pin_memory=True,
        group_by_length=True, seed=17,
        eval_strategy="steps" if eval_ds is not None else "no",
        eval_steps=a.save_steps if eval_ds is not None else None,
    )
    trainer = Trainer(
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
