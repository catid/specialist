#!/usr/bin/env python3
"""Synthetic BF16 Qwen3.6 expert-LoRA forward/backward/optimizer smoke.

This script opens no dataset or evaluation source.  It is intended to run in a
process exposing exactly one physical GPU through ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time


MODEL_ROOT = "/home/catid/specialist/models/Qwen3.6-35B-A3B"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def gib(value: int) -> float:
    return value / (1 << 30)


def memory(torch) -> dict[str, float]:
    return {
        "allocated_gib": gib(torch.cuda.memory_allocated()),
        "reserved_gib": gib(torch.cuda.memory_reserved()),
        "peak_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": gib(torch.cuda.max_memory_reserved()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--routed-rank", type=int, choices=(2, 4), default=4)
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--seed", type=int, default=19021)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    import build_fast_linear_attention_contract_v1 as fast_contract
    import qwen36_expert_lora_v1 as expert_lora

    _require(torch.cuda.is_available(), "CUDA is required")
    _require(torch.cuda.device_count() == 1, "worker must see exactly one GPU")
    _require(16 <= arguments.sequence_length <= 2048, "invalid synthetic sequence length")
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.cuda.set_device(0)
    device_properties = torch.cuda.get_device_properties(0)
    _require(device_properties.major == 12, "expected Blackwell compute capability 12.0")

    started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    hybrid = fast_contract.apply_qwen35_moe_training_hybrid(model)
    _require(hybrid["matched_module_count"] == 30, "hybrid did not cover 30 linear-attention layers")

    architecture = expert_lora.load_architecture_contract()
    spec = (
        expert_lora.shared_only_spec_from_contract(architecture)
        if arguments.shared_only
        else expert_lora.spec_from_contract(
            architecture, routed_rank=arguments.routed_rank
        )
    )
    preattach = expert_lora.validate_preattach_model(model, spec)
    model = get_peft_model(model, expert_lora.make_lora_config(spec))
    scope = expert_lora.audit_postattach_scope(model, spec)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    after_setup = memory(torch)

    generator = torch.Generator(device="cpu").manual_seed(arguments.seed + 1)
    input_ids = torch.randint(
        low=100,
        high=model.config.vocab_size - 100,
        size=(1, arguments.sequence_length),
        generator=generator,
        dtype=torch.long,
    ).cuda()
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    torch.cuda.reset_peak_memory_stats()
    step_started = time.perf_counter()
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss
    _require(torch.isfinite(loss).item(), "synthetic loss is not finite")
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
    _require(torch.isfinite(gradient_norm).item(), "synthetic gradient norm is not finite")
    after_backward = memory(torch)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    step_seconds = time.perf_counter() - step_started
    after_step = memory(torch)

    result = {
        "schema": "qwen36-expert-lora-synthetic-memory-smoke-v1",
        "authority": {
            "synthetic_token_ids_only": True,
            "dataset_or_evaluation_source_opened": False,
            "checkpoint_weights_loaded": True,
            "optimizer_created": True,
            "synthetic_adapter_update_performed": True,
        },
        "gpu": {
            "name": device_properties.name,
            "total_memory_gib": gib(device_properties.total_memory),
            "compute_capability": f"{device_properties.major}.{device_properties.minor}",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "configuration": {
            "sequence_length": arguments.sequence_length,
            "routed_rank": None if arguments.shared_only else arguments.routed_rank,
            "shared_rank": spec.shared_rank,
            "shared_only": arguments.shared_only,
            "dtype": "torch.bfloat16",
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "use_cache": False,
            "hybrid_module_count": hybrid["matched_module_count"],
        },
        "scope": {
            "preattach_identity_sha256": preattach["identity_sha256"],
            "postattach_identity_sha256": scope["identity_sha256"],
            "target_count": scope["target_count"],
            "trainable_tensor_count": scope["trainable_tensor_count"],
            "trainable_elements": scope["trainable_elements"],
        },
        "measurement": {
            "loss": float(loss.detach().cpu()),
            "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            "model_and_adapter_setup_seconds": step_started - started,
            "step_seconds": step_seconds,
            "tokens_per_second": arguments.sequence_length / step_seconds,
            "after_setup": after_setup,
            "after_backward": after_backward,
            "after_step": after_step,
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output is not None:
        output = arguments.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{output.name}.", dir=output.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, output)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
