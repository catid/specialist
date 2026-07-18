#!/usr/bin/env python3
"""Synthetic BF16 Qwen3.6 expert-LoRA forward/backward/optimizer smoke.

This script opens no dataset or evaluation source.  It is intended to run in a
process exposing exactly one physical GPU through ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any


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


def validate_fast_kernel_training_policy(fast_contract: Any) -> dict[str, Any]:
    """Revalidate the sealed hybrid decision before mutating model bindings."""
    value = fast_contract.build(check=True)
    _require(
        value.get("schema") == fast_contract.SCHEMA,
        "fast linear-attention contract schema changed",
    )
    decision = value.get("selected_fast_or_fallback")
    _require(isinstance(decision, dict), "fast-kernel selection decision is missing")
    _require(
        decision.get("selected") == "hybrid_training"
        and decision.get("selected_bindings")
        == fast_contract.HYBRID_TRAINING_BINDINGS
        and decision.get("hybrid_training_path_runtime_validated_on_all_four_gpus")
        is True
        and decision.get("hybrid_path_validation_failures") == []
        and decision.get("material_improvement_gate", {}).get(
            "passed_on_all_four_gpus"
        )
        is True
        and decision.get("training_launch_authorized") is False,
        "sealed fast-kernel contract does not authorize the validated hybrid policy",
    )
    execution = value.get("gpu_execution")
    _require(isinstance(execution, dict), "fast-kernel GPU execution receipt is missing")
    baseline = execution.get("gpu_baseline_checks")
    _require(
        isinstance(baseline, dict)
        and baseline.get("all_match_architecture_contract") is True
        and len(baseline.get("per_gpu", [])) == 4
        and all(item.get("all_match") is True for item in baseline["per_gpu"]),
        "fast-kernel GPU baseline does not match the architecture contract",
    )
    results = execution.get("per_gpu_results")
    _require(isinstance(results, list), "fast-kernel per-GPU results are missing")
    _require(
        [item.get("gpu", {}).get("index") for item in results] == [0, 1, 2, 3],
        "fast-kernel physical GPU indices changed",
    )
    uuids = [item["gpu"].get("uuid") for item in results]
    _require(
        all(isinstance(value, str) and value for value in uuids)
        and len(set(uuids)) == 4,
        "fast-kernel physical GPU identities are not distinct",
    )
    for item in results:
        physical_index = item["gpu"]["index"]
        probes = item.get("probes")
        _require(isinstance(probes, dict), "fast-kernel probe set is missing")
        for probe_name in (
            "causal_fallback",
            "gated_delta",
            "qwen35_moe_hybrid_module",
        ):
            probe = probes.get(probe_name)
            _require(
                isinstance(probe, dict)
                and probe.get("status") == "ok"
                and probe.get("gpu", {}).get("physical_index") == physical_index
                and probe.get("gpu", {}).get("worker_visible_index") == 0,
                f"fast-kernel {probe_name} did not run on physical GPU {physical_index}",
            )
        hybrid = probes["qwen35_moe_hybrid_module"]
        for sequence_length in (128, 2048):
            benchmark = hybrid.get("benchmarks", {}).get(str(sequence_length))
            _require(
                isinstance(benchmark, dict),
                f"fast-kernel seq-{sequence_length} benchmark is missing",
            )
            modes = {}
            for mode in ("hybrid_training", "torch_reference"):
                measured = benchmark.get(mode)
                _require(
                    isinstance(measured, dict),
                    f"fast-kernel seq-{sequence_length} {mode} benchmark is missing",
                )
                checked = {}
                for field in (
                    "tokens_per_second",
                    "milliseconds_per_iteration",
                    "peak_allocated_bytes",
                    "baseline_allocated_bytes",
                    "peak_reserved_bytes",
                    "baseline_reserved_bytes",
                ):
                    observed = measured.get(field)
                    _require(
                        isinstance(observed, (int, float))
                        and not isinstance(observed, bool)
                        and math.isfinite(float(observed))
                        and observed >= 0,
                        f"invalid fast-kernel benchmark field: {mode}.{field}",
                    )
                    checked[field] = float(observed)
                _require(
                    checked["tokens_per_second"] > 0
                    and checked["milliseconds_per_iteration"] > 0
                    and checked["peak_allocated_bytes"] > 0
                    and checked["peak_reserved_bytes"] > 0
                    and checked["peak_allocated_bytes"]
                    >= checked["baseline_allocated_bytes"]
                    and checked["peak_reserved_bytes"]
                    >= checked["baseline_reserved_bytes"],
                    f"inconsistent fast-kernel seq-{sequence_length} {mode} benchmark",
                )
                modes[mode] = checked
            speedup = benchmark.get("hybrid_throughput_speedup")
            memory_ratio = benchmark.get("hybrid_peak_allocated_bytes_ratio")
            _require(
                isinstance(speedup, (int, float))
                and not isinstance(speedup, bool)
                and math.isfinite(float(speedup))
                and speedup > 0
                and isinstance(memory_ratio, (int, float))
                and not isinstance(memory_ratio, bool)
                and math.isfinite(float(memory_ratio))
                and memory_ratio > 0,
                f"invalid fast-kernel seq-{sequence_length} derived ratios",
            )
            expected_speedup = (
                modes["hybrid_training"]["tokens_per_second"]
                / modes["torch_reference"]["tokens_per_second"]
            )
            expected_memory_ratio = (
                modes["hybrid_training"]["peak_allocated_bytes"]
                / modes["torch_reference"]["peak_allocated_bytes"]
            )
            _require(
                math.isclose(float(speedup), expected_speedup, rel_tol=1e-12)
                and math.isclose(
                    float(memory_ratio), expected_memory_ratio, rel_tol=1e-12
                ),
                f"fast-kernel seq-{sequence_length} derived ratios are inconsistent",
            )
            if sequence_length == 2048:
                _require(
                    speedup >= 1.10 or memory_ratio <= 0.95,
                    "fast-kernel seq-2048 material-improvement gate failed",
                )
    return {
        "path": fast_contract.OUTPUT.relative_to(fast_contract.ROOT).as_posix(),
        "file_sha256": fast_contract.file_sha256(fast_contract.OUTPUT),
        "content_sha256": value["content_sha256_before_self_field"],
        "selected": decision["selected"],
        "selected_bindings": decision["selected_bindings"],
        "all_four_physical_gpus_revalidated": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--routed-rank", type=int, choices=(2, 4), default=4)
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--seed", type=int, default=19021)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--minimum-headroom-gib", type=float, default=3.0)
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
    _require(3 <= arguments.steps <= 10, "memory smoke must run three to ten steps")
    _require(
        math.isfinite(arguments.minimum_headroom_gib)
        and arguments.minimum_headroom_gib >= 3.0,
        "minimum reliable headroom must be at least three GiB",
    )
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.cuda.set_device(0)
    device_properties = torch.cuda.get_device_properties(0)
    _require(device_properties.major == 12, "expected Blackwell compute capability 12.0")
    fast_policy_receipt = validate_fast_kernel_training_policy(fast_contract)

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
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    _require(model.config.use_cache is False, "KV cache was not disabled")
    _require(
        model.config.output_router_logits is False
        and model.config.router_aux_loss_coef == 0.0,
        "router auxiliary output or loss was not disabled",
    )
    _require(
        model.is_gradient_checkpointing is True,
        "gradient checkpointing was not enabled",
    )
    _require(
        not any(
            name == "visual" or name.startswith(("visual.", "model.visual."))
            for name, _ in model.named_modules()
        ),
        "text-only memory smoke unexpectedly instantiated vision modules",
    )
    hybrid = fast_contract.apply_qwen35_moe_training_hybrid(model)
    _require(hybrid["matched_module_count"] == 30, "hybrid did not cover 30 linear-attention layers")
    _require(
        hybrid["bindings"] == fast_contract.HYBRID_TRAINING_BINDINGS,
        "applied hybrid bindings differ from the sealed policy",
    )

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
    _require(trainable, "adapter has no trainable parameters")
    _require(
        all(parameter.dtype == torch.float32 for parameter in trainable),
        "every trainable LoRA tensor must be FP32 before AdamW",
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    after_setup = memory(torch)
    setup_seconds = time.perf_counter() - started

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
    measurements = []
    for step_index in range(arguments.steps):
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
        loss_value = float(loss.detach().cpu())
        gradient_norm_value = float(gradient_norm.detach().cpu())
        del outputs, loss
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_seconds = time.perf_counter() - step_started
        after_step = memory(torch)
        measurements.append({
            "step": step_index + 1,
            "loss": loss_value,
            "gradient_norm_before_clip": gradient_norm_value,
            "step_seconds": step_seconds,
            "tokens_per_second": arguments.sequence_length / step_seconds,
            "after_backward": after_backward,
            "after_step": after_step,
        })

    total_memory_gib = gib(device_properties.total_memory)
    peak_allocated_gib = max(
        row[phase]["peak_allocated_gib"]
        for row in measurements
        for phase in ("after_backward", "after_step")
    )
    peak_reserved_gib = max(
        row[phase]["peak_reserved_gib"]
        for row in measurements
        for phase in ("after_backward", "after_step")
    )
    allocated_headroom_gib = total_memory_gib - peak_allocated_gib
    reserved_headroom_gib = total_memory_gib - peak_reserved_gib
    _require(
        min(allocated_headroom_gib, reserved_headroom_gib)
        >= arguments.minimum_headroom_gib,
        "BF16 LoRA memory smoke lacks the required reliable VRAM headroom",
    )
    adapter_parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in trainable
    )
    optimizer_tensor_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    )

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
            "steps": arguments.steps,
            "minimum_headroom_gib": arguments.minimum_headroom_gib,
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
        "runtime_contracts": {
            "fast_linear_attention": fast_policy_receipt,
            "vision_excluded_not_instantiated": True,
            "gradient_checkpointing_observed_enabled": True,
            "kv_cache_observed_disabled": True,
            "router_auxiliary_loss_disabled": True,
        },
        "measurement": {
            "model_and_adapter_setup_seconds": setup_seconds,
            "after_setup": after_setup,
            "steps": measurements,
            "adapter_parameter_bytes": adapter_parameter_bytes,
            "optimizer_tensor_bytes": optimizer_tensor_bytes,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "allocated_headroom_gib": allocated_headroom_gib,
            "reserved_headroom_gib": reserved_headroom_gib,
            "reliable_headroom_gate_passed": True,
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
