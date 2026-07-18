#!/usr/bin/env python3
"""One-GPU BF16 expert-aware LoRA SFT for the sealed mixed snapshot.

The program is intentionally independent of the abandoned EGGROLL-ES path.
It consumes the deterministic one-epoch schedule produced by
``build_mixed_training_snapshot_v1.py`` and supports both assistant-only chat
loss and all-token Markdown causal loss in the same run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import qwen36_expert_lora_v1 as expert_lora
import qwen36_mixed_training_runtime_v1 as mixed


ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "models/Qwen3.6-35B-A3B"
FAST_CONTRACT = ROOT / "training_protocol/fast_linear_attention_contract_v1.json"
RUN_SCHEMA = "qwen36-low-regression-expert-lora-sft-run-v1"
STATE_SCHEMA = "qwen36-low-regression-sft-resume-state-v1"
CHECKPOINT_SCHEMA = "qwen36-low-regression-sft-checkpoint-v1"
METRICS_SCHEMA = "qwen36-low-regression-sft-training-metric-v1"
ROUTING_SCHEMA = "qwen36-low-regression-sft-routing-metric-v1"
ROUTER_LAYER_NAMES = tuple(
    f"model.layers.{layer}.mlp.gate" for layer in range(40)
)
ROUTED_EXPERT_COUNT = 256
ROUTED_TOP_K = 8


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, payload)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _require(path.is_file() and not path.is_symlink(), f"JSONL artifact is invalid: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            _require(line.endswith("\n") and line.strip(), f"{path}:{line_number}: invalid line")
            row = json.loads(line)
            _require(isinstance(row, dict), f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def _truncate_step_log(path: Path, *, maximum_optimizer_step: int, schema: str) -> None:
    rows = _read_jsonl(path)
    kept = []
    previous = 0
    for row in rows:
        _require(row.get("schema") == schema, f"{path}: log schema changed")
        step = row.get("optimizer_step")
        _require(isinstance(step, int) and step > previous, f"{path}: optimizer steps are not strictly increasing")
        previous = step
        if step <= maximum_optimizer_step:
            kept.append(row)
    payload = b"".join(
        (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        for row in kept
    )
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _routing_summary(
    path: Path, *, interval: int, completed_optimizer_steps: int
) -> dict[str, Any]:
    _require(interval >= 0, "routing interval is invalid")
    _require(completed_optimizer_steps > 0, "routing summary needs a completed run")
    rows = _read_jsonl(path)
    expected_steps = (
        []
        if interval == 0
        else list(range(1, completed_optimizer_steps + 1, interval))
    )
    observed_steps: list[int] = []
    previous_cursor = 0
    layers: dict[str, dict[str, Any]] = {}
    for row in rows:
        _require(row.get("schema") == ROUTING_SCHEMA, "routing log schema changed")
        step = row.get("optimizer_step")
        cursor = row.get("next_cursor")
        _require(isinstance(step, int), "routing optimizer step is invalid")
        _require(
            isinstance(cursor, int) and cursor > previous_cursor,
            "routing cursor coverage is not strictly increasing",
        )
        previous_cursor = cursor
        observed_steps.append(step)
        row_layers = row.get("layers")
        _require(
            row.get("layer_count") == len(ROUTER_LAYER_NAMES)
            and isinstance(row_layers, dict)
            and set(row_layers) == set(ROUTER_LAYER_NAMES),
            "routing log does not contain the exact layer inventory",
        )
        row_token_counts = set()
        for name in ROUTER_LAYER_NAMES:
            item = row_layers[name]
            _require(isinstance(item, dict), "routing layer metric is invalid")
            counts = item.get("expert_counts")
            _require(
                isinstance(counts, list)
                and len(counts) == ROUTED_EXPERT_COUNT
                and all(isinstance(value, int) and value >= 0 for value in counts),
                "routing expert-count vector changed",
            )
            tokens = item.get("tokens")
            assignments = sum(counts)
            _require(isinstance(tokens, int) and tokens > 0, "routing token count is invalid")
            _require(
                assignments == tokens * ROUTED_TOP_K
                and item.get("total_assignments") == assignments,
                "routing assignment accounting changed",
            )
            _require(
                item.get("active_experts") == sum(value > 0 for value in counts),
                "routing active-expert accounting changed",
            )
            expected_maximum = max(counts) / assignments
            maximum = item.get("maximum_expert_fraction")
            _require(
                isinstance(maximum, (int, float))
                and math.isfinite(maximum)
                and math.isclose(maximum, expected_maximum, rel_tol=1e-12, abs_tol=1e-12),
                "routing maximum-expert fraction changed",
            )
            router_entropy = item.get("router_entropy")
            selected_entropy = item.get("selected_topk_entropy")
            _require(
                isinstance(router_entropy, (int, float))
                and math.isfinite(router_entropy)
                and -1e-6 <= router_entropy <= math.log(ROUTED_EXPERT_COUNT) + 1e-6,
                "full routing entropy is invalid",
            )
            _require(
                isinstance(selected_entropy, (int, float))
                and math.isfinite(selected_entropy)
                and -1e-6 <= selected_entropy <= math.log(ROUTED_TOP_K) + 1e-6,
                "selected routing entropy is invalid",
            )
            row_token_counts.add(tokens)
            target = layers.setdefault(
                name,
                {
                    "tokens": 0,
                    "expert_counts": [0] * ROUTED_EXPERT_COUNT,
                    "router_entropy_token_sum": 0.0,
                    "selected_topk_entropy_token_sum": 0.0,
                },
            )
            target["tokens"] += tokens
            target["expert_counts"] = [
                left + right
                for left, right in zip(target["expert_counts"], counts, strict=True)
            ]
            target["router_entropy_token_sum"] += router_entropy * tokens
            target["selected_topk_entropy_token_sum"] += selected_entropy * tokens
        _require(
            len(row_token_counts) == 1,
            "routing layers disagree on sampled token coverage",
        )
    _require(
        observed_steps == expected_steps,
        "routing optimizer-step coverage differs from the sampling contract",
    )
    aggregate = {}
    for name, item in sorted(layers.items()):
        assignments = sum(item["expert_counts"])
        aggregate[name] = {
            "tokens": item["tokens"],
            "total_assignments": assignments,
            "expert_counts": item["expert_counts"],
            "router_entropy": item["router_entropy_token_sum"] / max(1, item["tokens"]),
            "selected_topk_entropy": (
                item["selected_topk_entropy_token_sum"] / max(1, item["tokens"])
            ),
            "active_experts": sum(value > 0 for value in item["expert_counts"]),
            "maximum_expert_fraction": (
                max(item["expert_counts"]) / assignments if assignments else 0.0
            ),
        }
    _require(
        not rows or len(aggregate) == len(ROUTER_LAYER_NAMES),
        "aggregate routing summary omitted a layer",
    )
    return _self_addressed({
        "schema": "qwen36-low-regression-sft-routing-summary-v1",
        "status": "complete" if rows else "disabled",
        "sampling_interval_optimizer_steps": interval,
        "completed_optimizer_steps": completed_optimizer_steps,
        "observed_optimizer_steps": observed_steps,
        "expected_sample_count": len(expected_steps),
        "step_coverage_complete": True,
        "checkpoint_recomputation_excluded": True,
        "routing_log": {
            "path": path.name,
            "rows": len(rows),
            "sha256": file_sha256(path) if path.exists() else hashlib.sha256(b"").hexdigest(),
        },
        "layers": aggregate,
        "layer_count": len(aggregate),
    })


def _step_log_receipt(
    path: Path,
    *,
    schema: str,
    completed_optimizer_steps: int,
    final_cursor: int,
) -> dict[str, Any]:
    rows = _read_jsonl(path)
    _require(
        len(rows) == completed_optimizer_steps,
        "training metric row count differs from completed optimizer steps",
    )
    expected_steps = list(range(1, completed_optimizer_steps + 1))
    observed_steps = [row.get("optimizer_step") for row in rows]
    _require(observed_steps == expected_steps, "training metric step coverage differs")
    cursors = []
    for row in rows:
        _require(row.get("schema") == schema, "training metric schema changed")
        cursor = row.get("next_cursor")
        _require(isinstance(cursor, int), "training metric cursor is invalid")
        cursors.append(cursor)
    _require(
        cursors == sorted(set(cursors)) and cursors[-1] == final_cursor,
        "training metric cursor coverage differs",
    )
    return {
        "path": path.name,
        "rows": len(rows),
        "sha256": file_sha256(path),
        "first_optimizer_step": observed_steps[0],
        "last_optimizer_step": observed_steps[-1],
        "final_cursor": cursors[-1],
    }


def _self_addressed(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def _load_self_addressed(path: Path) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"contract is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"contract is not an object: {path}")
    claimed = value.get("content_sha256_before_self_field")
    unsigned = copy.deepcopy(value)
    unsigned.pop("content_sha256_before_self_field", None)
    _require(claimed == canonical_sha256(unsigned), f"contract self address changed: {path}")
    return value


class CosineToFloorSchedule:
    """Five-percent linear warmup followed by cosine decay to a nonzero floor."""

    def __init__(
        self,
        optimizer: Any,
        *,
        peak_lr: float,
        total_steps: int,
        warmup_ratio: float = 0.05,
        floor_ratio: float = 0.10,
    ) -> None:
        _require(peak_lr > 0.0 and math.isfinite(peak_lr), "peak LR is invalid")
        _require(total_steps > 0, "scheduler needs at least one optimizer step")
        _require(0.0 < warmup_ratio < 1.0, "warmup ratio is invalid")
        _require(0.0 < floor_ratio <= 1.0, "LR floor ratio is invalid")
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.total_steps = int(total_steps)
        self.warmup_ratio = float(warmup_ratio)
        self.floor_ratio = float(floor_ratio)
        self.warmup_steps = max(1, math.ceil(total_steps * warmup_ratio))
        self.step_count = 0

    def lr_for_step(self, step: int) -> float:
        _require(0 <= step < self.total_steps, "scheduler step is outside run")
        if step < self.warmup_steps:
            factor = (step + 1) / self.warmup_steps
        else:
            decay_steps = self.total_steps - self.warmup_steps
            progress = 1.0 if decay_steps <= 1 else (step - self.warmup_steps) / (decay_steps - 1)
            factor = self.floor_ratio + (1.0 - self.floor_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        return self.peak_lr * factor

    def prepare_step(self) -> float:
        lr = self.lr_for_step(self.step_count)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def step(self) -> None:
        _require(self.step_count < self.total_steps, "scheduler advanced beyond run")
        self.step_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "cosine-to-floor-schedule-v1",
            "peak_lr": self.peak_lr,
            "total_steps": self.total_steps,
            "warmup_ratio": self.warmup_ratio,
            "floor_ratio": self.floor_ratio,
            "warmup_steps": self.warmup_steps,
            "step_count": self.step_count,
        }

    def load_state_dict(self, value: dict[str, Any]) -> None:
        expected = self.state_dict()
        for key in (
            "schema",
            "peak_lr",
            "total_steps",
            "warmup_ratio",
            "floor_ratio",
            "warmup_steps",
        ):
            _require(value.get(key) == expected[key], f"scheduler resume differs: {key}")
        step = value.get("step_count")
        _require(isinstance(step, int) and 0 <= step <= self.total_steps, "invalid scheduler resume step")
        self.step_count = step


def token_cross_entropy_sum(logits: Any, labels: Any) -> tuple[Any, int]:
    import torch
    import torch.nn.functional as functional

    shifted_logits = logits[..., :-1, :].contiguous().float()
    shifted_labels = labels[..., 1:].contiguous()
    supervised = int(shifted_labels.ne(-100).sum().item())
    _require(supervised > 0, "sequence has no shifted supervised token")
    loss_sum = functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    _require(torch.isfinite(loss_sum).item(), "token loss is not finite")
    return loss_sum, supervised


class RoutingRecorder:
    """Collect exact top-k use and entropy during the forward pass only."""

    def __init__(self, model: Any) -> None:
        self.enabled = False
        self.rows: dict[str, dict[str, Any]] = {}
        self.handles = []
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        modules = dict(base.named_modules())
        for name in ROUTER_LAYER_NAMES:
            _require(name in modules, f"router module missing for routing metrics: {name}")
            self.handles.append(modules[name].register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            if not self.enabled:
                return
            import torch

            _require(isinstance(output, tuple) and len(output) >= 3, f"router output changed: {name}")
            router_logits, weights, selected = output[0], output[1], output[2]
            _require(weights.shape == selected.shape and selected.ndim == 2, f"router shape changed: {name}")
            _require(
                router_logits.ndim == 2
                and router_logits.shape[0] == selected.shape[0]
                and router_logits.shape[1] == 256,
                f"full router-logit shape changed: {name}",
            )
            with torch.no_grad():
                counts = torch.bincount(
                    selected.detach().reshape(-1), minlength=ROUTED_EXPERT_COUNT
                )
                normalized = weights.detach().float()
                normalized = normalized / normalized.sum(
                    dim=-1, keepdim=True
                ).clamp_min(1e-12)
                selected_entropy_sum = (
                    -(normalized * normalized.clamp_min(1e-12).log())
                    .sum(dim=-1)
                    .sum()
                )
                full_probabilities = router_logits.detach().float().softmax(dim=-1)
                full_entropy_sum = (
                    -(full_probabilities * full_probabilities.clamp_min(1e-12).log())
                    .sum(dim=-1)
                    .sum()
                )
            row = self.rows.setdefault(
                name,
                {
                    "expert_counts": torch.zeros(
                        ROUTED_EXPERT_COUNT, dtype=torch.int64, device=counts.device
                    ),
                    "full_entropy_sum": torch.zeros(
                        (), dtype=torch.float32, device=counts.device
                    ),
                    "selected_entropy_sum": torch.zeros(
                        (), dtype=torch.float32, device=counts.device
                    ),
                    "tokens": 0,
                },
            )
            row["expert_counts"] += counts
            row["full_entropy_sum"] += full_entropy_sum.detach()
            row["selected_entropy_sum"] += selected_entropy_sum.detach()
            row["tokens"] += int(selected.shape[0])

        return record

    def start(self) -> None:
        self.rows = {}
        self.enabled = True

    def pause(self) -> None:
        """Exclude gradient-checkpoint recomputation forwards from metrics."""
        self.enabled = False

    def resume(self) -> None:
        self.enabled = True

    def stop(self) -> dict[str, Any]:
        self.enabled = False
        import torch

        layers = {}
        ordered = sorted(self.rows.items())
        if ordered:
            # Avoid forty device synchronizations in forward hooks.  Transfer
            # all counters/scalars in three compact batches after the sampled
            # forward has finished.
            counts_batch = torch.stack(
                [row["expert_counts"] for _, row in ordered]
            ).cpu()
            full_entropy_batch = torch.stack(
                [row["full_entropy_sum"] for _, row in ordered]
            ).cpu()
            selected_entropy_batch = torch.stack(
                [row["selected_entropy_sum"] for _, row in ordered]
            ).cpu()
        for index, (name, row) in enumerate(ordered):
            counts = counts_batch[index]
            total_assignments = int(counts.sum().item())
            layers[name] = {
                "tokens": row["tokens"],
                "total_assignments": total_assignments,
                "expert_counts": counts.tolist(),
                "router_entropy": float(full_entropy_batch[index])
                / max(1, row["tokens"]),
                "selected_topk_entropy": (
                    float(selected_entropy_batch[index])
                    / max(1, row["tokens"])
                ),
                "active_experts": int(counts.gt(0).sum().item()),
                "maximum_expert_fraction": (
                    float(counts.max().item()) / total_assignments if total_assignments else 0.0
                ),
            }
        self.rows = {}
        _require(
            set(layers) == set(ROUTER_LAYER_NAMES),
            "routing pass did not observe the exact router layer inventory",
        )
        return {"layers": layers, "layer_count": len(layers)}

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


def _selected_end_cursor(authority: mixed.MixedTrainingAuthority, token_limit: int | None) -> int:
    if token_limit is None:
        return len(authority.schedule)
    _require(token_limit > 0, "pilot token limit must be positive")
    for row in authority.schedule:
        if row["cumulative_budget_tokens"] >= token_limit:
            return row["cursor"] + 1
    raise RuntimeError("pilot token limit exceeds snapshot budget")


def _group_ranges(end_cursor: int, accumulation: int) -> list[tuple[int, int]]:
    _require(end_cursor > 0 and accumulation in (1, 2), "invalid optimizer grouping")
    return [
        (start, min(start + accumulation, end_cursor))
        for start in range(0, end_cursor, accumulation)
    ]


def _run_config(arguments: argparse.Namespace, authority: mixed.MixedTrainingAuthority, end_cursor: int) -> dict[str, Any]:
    selected_budget_tokens = authority.schedule[end_cursor - 1][
        "cumulative_budget_tokens"
    ]
    value = {
        "schema": RUN_SCHEMA,
        "run_id": arguments.run_id,
        "model": {
            "path": MODEL_ROOT.relative_to(ROOT).as_posix(),
            "config_sha256": file_sha256(MODEL_ROOT / "config.json"),
            "index_sha256": file_sha256(
                MODEL_ROOT / "model.safetensors.index.json"
            ),
            "dtype": "bfloat16",
            "text_only_auto_causal_lm": True,
            "vision_excluded": True,
        },
        "snapshot": {
            "top_manifest_path": authority.top_manifest_path.relative_to(ROOT).as_posix(),
            "top_manifest_content_sha256": authority.top_manifest["content_sha256_before_self_field"],
            "variant": authority.variant,
            "variant_manifest_content_sha256": authority.variant_manifest["content_sha256_before_self_field"],
            "sequence_set_identity_sha256": authority.sequence_set_identity_sha256,
            "final_cursor_commitment_sha256": authority.final_cursor_commitment_sha256,
            "scheduled_cursor_count": len(authority.schedule),
            "selected_end_cursor": end_cursor,
            "pilot_budget_token_limit": arguments.max_budget_tokens,
            "selected_budget_tokens": selected_budget_tokens,
        },
        "adapter": {
            "shared_only": arguments.shared_only,
            "routed_rank": None if arguments.shared_only else arguments.routed_rank,
            "shared_rank": arguments.shared_rank,
            "alpha_equals_rank": True,
            "dropout": 0.0,
            "bias": "none",
            "router_frozen": True,
            "attention_and_mixing_frozen": True,
        },
        "optimization": {
            "epochs": 1.0,
            "per_device_sequence_batch": 1,
            "gradient_accumulation": arguments.gradient_accumulation,
            "peak_learning_rate": arguments.learning_rate,
            "warmup_ratio": 0.05,
            "schedule": "cosine_to_10_percent_peak",
            "optimizer": "AdamW",
            "betas": [0.9, 0.999],
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "cache": False,
            "loss_reduction": "supervised_token_mean_across_optimizer_update",
            "checkpoint_interval_budget_tokens": arguments.checkpoint_tokens,
            "seed": arguments.seed,
        },
        "runtime": {
            "world_size_required": 1,
            "one_visible_gpu_required": True,
            "attention_implementation": "sdpa",
            "hybrid_linear_attention_contract": FAST_CONTRACT.relative_to(ROOT).as_posix(),
            "routing_metric_interval_optimizer_steps": arguments.routing_every,
            "implementation_receipts": {
                path.relative_to(ROOT).as_posix(): file_sha256(path)
                for path in (
                    Path(__file__).resolve(),
                    Path(expert_lora.__file__).resolve(),
                    Path(mixed.__file__).resolve(),
                    (ROOT / expert_lora.DEFAULT_ARCHITECTURE_CONTRACT).resolve(),
                    FAST_CONTRACT.resolve(),
                )
            },
        },
    }
    return _self_addressed(value)


def _write_static_artifacts(
    output: Path,
    run_config: dict[str, Any],
    authority: mixed.MixedTrainingAuthority,
    model: Any,
    lora_config: Any,
    scope: dict[str, Any],
    gpu: dict[str, Any],
) -> None:
    _atomic_json(output / "run_config.json", run_config)
    _atomic_json(output / "model_config.json", model.config.to_dict())
    adapter = lora_config.to_dict()
    for key in ("target_modules", "target_parameters"):
        if adapter.get(key) is not None:
            adapter[key] = sorted(adapter[key])
    _atomic_json(output / "adapter_config.json", adapter)
    _atomic_json(output / "dataset_manifest.json", {
        "top": authority.top_manifest,
        "variant": authority.variant_manifest,
    })
    _atomic_json(output / "dataset_hashes.json", {
        "top_manifest_file_sha256": file_sha256(authority.top_manifest_path),
        "top_manifest_content_sha256": authority.top_manifest["content_sha256_before_self_field"],
        "variant_manifest_content_sha256": authority.variant_manifest["content_sha256_before_self_field"],
        "sequence_set_identity_sha256": authority.sequence_set_identity_sha256,
        "final_cursor_commitment_sha256": authority.final_cursor_commitment_sha256,
    })
    trainable_lines = [
        json.dumps(row, sort_keys=True)
        for row in scope["trainable_parameters"]
    ]
    _atomic_bytes(
        output / "trainable_parameters.txt",
        ("\n".join(trainable_lines) + "\n").encode("utf-8"),
    )
    # Distribution metadata alone loses editable/direct-URL origins and VCS
    # commits.  Preserve the actual pip replay specification requested by the
    # protocol using the exact interpreter running this training process.
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    _require(pip_freeze.endswith("\n"), "pip freeze output is incomplete")
    _atomic_bytes(output / "pip_freeze.txt", pip_freeze.encode("utf-8"))
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu,
    }
    _atomic_bytes(
        output / "environment.txt",
        (json.dumps(environment, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        ),
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    _atomic_json(output / "git_commits.json", {"specialist": commit, "dirty_paths": status})
    _atomic_json(output / "evaluation_results.json", {
        "status": "not_run_by_training_process",
        "reason": "development and final evaluation are separate sealed stages",
    })
    (output / "checkpoints").mkdir(exist_ok=True)
    (output / "plots").mkdir(exist_ok=True)
    (output / "samples").mkdir(exist_ok=True)


def _memory(torch: Any) -> dict[str, float]:
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def _checkpoint_state(
    *,
    authority: mixed.MixedTrainingAuthority,
    next_cursor: int,
    optimizer_step: int,
    cumulative: dict[str, int],
    run_config_sha256: str,
) -> dict[str, Any]:
    commitment = (
        mixed.ZERO_COMMITMENT
        if next_cursor == 0
        else authority.schedule[next_cursor - 1]["cursor_commitment_sha256"]
    )
    return _self_addressed({
        "schema": STATE_SCHEMA,
        "variant": authority.variant,
        "sequence_set_identity_sha256": authority.sequence_set_identity_sha256,
        "cursor": next_cursor,
        "cursor_commitment_sha256": commitment,
        "optimizer_step": optimizer_step,
        "cumulative": cumulative,
        "run_config_content_sha256": run_config_sha256,
    })


def _save_checkpoint(
    output: Path,
    *,
    name: str,
    model: Any,
    optimizer: Any,
    scheduler: CosineToFloorSchedule,
    state: dict[str, Any],
    torch: Any,
) -> Path:
    import numpy

    target = output / "checkpoints" / name
    _require(not target.exists(), f"checkpoint already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=target.parent))
    try:
        model.save_pretrained(temporary / "adapter", safe_serialization=True)
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(scheduler.state_dict(), temporary / "scheduler.pt")
        torch.save({
            "python_random_state": random.getstate(),
            "numpy_random_state": numpy.random.get_state(),
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        }, temporary / "rng_state.pt")
        _atomic_json(temporary / "state.json", state)
        receipt = _self_addressed({
            "schema": CHECKPOINT_SCHEMA,
            "state_content_sha256": state["content_sha256_before_self_field"],
            "files": {
                path.relative_to(temporary).as_posix(): file_sha256(path)
                for path in sorted(temporary.rglob("*"))
                if path.is_file()
            },
        })
        _atomic_json(temporary / "checkpoint_receipt.json", receipt)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return target


def _find_latest_checkpoint(output: Path) -> Path:
    candidates = sorted(
        path for path in (output / "checkpoints").glob("checkpoint-*")
        if path.is_dir() and (path / "state.json").is_file()
    )
    _require(candidates, "no resumable checkpoint exists")
    states = [(_load_self_addressed(path / "state.json"), path) for path in candidates]
    return max(states, key=lambda item: item[0]["optimizer_step"])[1]


def _validate_checkpoint_files(path: Path) -> dict[str, Any]:
    receipt = _load_self_addressed(path / "checkpoint_receipt.json")
    _require(receipt.get("schema") == CHECKPOINT_SCHEMA, "checkpoint receipt schema changed")
    files = receipt.get("files")
    _require(isinstance(files, dict) and files, "checkpoint file receipts are missing")
    observed = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name != "checkpoint_receipt.json"
    }
    _require(observed == set(files), "checkpoint file inventory changed")
    for relative, expected in files.items():
        _require(isinstance(relative, str) and relative, "invalid checkpoint receipt path")
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected), "invalid checkpoint receipt hash")
        declared = path / relative
        _require(not declared.is_symlink(), "checkpoint contains a symlink file")
        item = declared.resolve()
        _require(item.is_relative_to(path), "checkpoint receipt path escapes checkpoint")
        _require(item.is_file(), "checkpoint contains a missing file")
        _require(file_sha256(item) == expected, f"checkpoint file changed: {relative}")
    state = _load_self_addressed(path / "state.json")
    _require(
        receipt.get("state_content_sha256")
        == state["content_sha256_before_self_field"],
        "checkpoint state receipt changed",
    )
    return state


def _expected_cumulative(
    authority: mixed.MixedTrainingAuthority, next_cursor: int
) -> dict[str, Any]:
    sequences = [
        authority.sequence_for_cursor(cursor) for cursor in range(next_cursor)
    ]
    result: dict[str, Any] = {
        "input_tokens": sum(row["input_token_count"] for row in sequences),
        "budget_tokens": sum(row["budget_token_count"] for row in sequences),
        "shifted_supervised_tokens": sum(
            row["shifted_supervised_token_count"] for row in sequences
        ),
        "stream_budget_tokens": {
            stream: sum(
                row["budget_token_count"]
                for row in sequences
                if row["stream"] == stream
            )
            for stream in sorted(mixed.ALLOWED_STREAMS)
        },
        "stream_input_tokens": {
            stream: sum(
                row["input_token_count"]
                for row in sequences
                if row["stream"] == stream
            )
            for stream in sorted(mixed.ALLOWED_STREAMS)
        },
    }
    return result


def _set_adapter_state_exact(model: Any, weights: dict[str, Any]) -> None:
    """Restore one PEFT adapter and prove that every adapter tensor was loaded.

    PEFT's ``set_peft_model_state_dict`` delegates to ``load_state_dict`` on
    the complete wrapped model.  Its ``missing_keys`` therefore includes the
    frozen base model during a normal adapter-only restore.  Rejecting all
    missing keys makes every valid resume fail.  We instead reject missing
    LoRA state and unexpected input, then compare PEFT's normalized adapter
    state with the checkpoint tensor-for-tensor.
    """
    from peft.utils.save_and_load import (
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )

    _require(isinstance(weights, dict) and weights, "resume adapter state is empty")
    _require(
        all(isinstance(name, str) and name for name in weights),
        "resume adapter contains an invalid key",
    )
    result = set_peft_model_state_dict(model, weights, adapter_name="default")
    _require(not result.unexpected_keys, "resume adapter has unexpected keys")
    missing_adapter = [name for name in result.missing_keys if ".lora_" in name]
    _require(not missing_adapter, "resume adapter is missing LoRA keys")

    observed = get_peft_model_state_dict(model, adapter_name="default")
    _require(set(observed) == set(weights), "resume adapter key inventory differs")
    for name, expected in weights.items():
        actual = observed[name]
        _require(
            actual.shape == expected.shape and actual.dtype == expected.dtype,
            f"resume adapter tensor metadata differs: {name}",
        )
        _require(
            actual.detach().cpu().equal(expected.detach().cpu()),
            f"resume adapter tensor differs after load: {name}",
        )


def _restore_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    scheduler: CosineToFloorSchedule,
    authority: mixed.MixedTrainingAuthority,
    output: Path,
    run_config_sha256: str,
    torch: Any,
) -> dict[str, Any]:
    import numpy
    from peft.utils.save_and_load import load_peft_weights

    path = path.resolve()
    _require(path.is_dir() and not path.is_symlink(), "resume checkpoint is invalid")
    _require(
        path.is_relative_to((output / "checkpoints").resolve()),
        "resume checkpoint is outside this run",
    )
    state = _validate_checkpoint_files(path)
    _require(state.get("schema") == STATE_SCHEMA, "resume state schema changed")
    mixed.validate_resume_identity(authority, state)
    _require(
        state.get("cumulative")
        == _expected_cumulative(authority, state["cursor"]),
        "resume cumulative token accounting differs",
    )
    _require(state.get("run_config_content_sha256") == run_config_sha256, "resume run config differs")
    weights = load_peft_weights(str(path / "adapter"), device="cuda")
    _set_adapter_state_exact(model, weights)
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True))
    rng = torch.load(path / "rng_state.pt", map_location="cpu", weights_only=False)
    random.setstate(rng["python_random_state"])
    numpy.random.set_state(rng["numpy_random_state"])
    torch.set_rng_state(rng["torch_cpu_rng_state"])
    torch.cuda.set_rng_state_all(rng["torch_cuda_rng_state_all"])
    _require(state["optimizer_step"] == scheduler.step_count, "resume optimizer/scheduler step differs")
    return state


def _prepare_model(arguments: argparse.Namespace):
    import numpy
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM

    import build_fast_linear_attention_contract_v1 as fast

    _require(torch.cuda.is_available(), "CUDA is required")
    _require(torch.cuda.device_count() == 1, "training worker must see exactly one GPU")
    _require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "four-way DDP is not authorized")
    properties = torch.cuda.get_device_properties(0)
    _require(properties.major == 12 and properties.total_memory >= 90 * 2**30, "expected one 96 GB Blackwell GPU")
    # The LoRA A matrices are randomly initialized.  Seed before model loading
    # and, critically, before PEFT attachment so independent runs and resumes
    # have a reproducible initialization identity.
    random.seed(arguments.seed)
    numpy.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    architecture = expert_lora.load_architecture_contract()
    expected_packages = architecture["software"]["packages"]
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "numpy",
        "safetensors",
        "tokenizers",
        "huggingface_hub",
    ):
        observed = importlib.metadata.version(name)
        _require(observed == expected_packages[name]["version"], f"pinned {name} version changed")
    _require(
        torch.__version__ == architecture["software"]["torch_runtime"]["version"],
        "pinned torch CUDA runtime build changed",
    )
    fast_value = _load_self_addressed(FAST_CONTRACT)
    decision = fast_value.get("selected_fast_or_fallback", {})
    _require(decision.get("selected") == "hybrid_training", "hybrid kernel contract is not selected")
    _require(decision.get("hybrid_training_path_runtime_validated_on_all_four_gpus") is True, "hybrid path lacks four-GPU validation")
    for receipt in fast_value["environment_integrity"]["source_receipts"].values():
        source = Path(receipt["path"])
        if not source.is_absolute():
            source = ROOT / source
        _require(source.is_file() and not source.is_symlink(), "hybrid runtime source disappeared")
        _require(file_sha256(source) == receipt["sha256"], f"hybrid runtime source changed: {source}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    _require(type(model).__name__ == "Qwen3_5MoeForCausalLM", "text-only model dispatch changed")
    _require(not any("visual" in name or "vision" in name for name, _ in model.named_modules()), "vision module was instantiated")
    model.config.use_cache = False
    model.config.output_router_logits = False
    model.config.router_aux_loss_coef = 0.0
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    hybrid = fast.apply_qwen35_moe_training_hybrid(model)
    _require(hybrid["matched_module_count"] == 30, "hybrid kernel did not cover 30 layers")
    spec = (
        expert_lora.shared_only_spec_from_contract(architecture, shared_rank=arguments.shared_rank)
        if arguments.shared_only
        else expert_lora.spec_from_contract(
            architecture,
            routed_rank=arguments.routed_rank,
            shared_rank=arguments.shared_rank,
        )
    )
    preattach = expert_lora.validate_preattach_model(model, spec)
    lora_config = expert_lora.make_lora_config(spec)
    model = get_peft_model(model, lora_config)
    scope = expert_lora.audit_postattach_scope(model, spec)
    model.train()
    gpu = {
        "name": properties.name,
        "total_memory_gib": properties.total_memory / 2**30,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    return torch, model, lora_config, scope, preattach, hybrid, gpu


def train(arguments: argparse.Namespace) -> dict[str, Any]:
    authority = mixed.load_training_authority(arguments.snapshot_manifest, variant=arguments.variant)
    observed_top = authority.top_manifest["content_sha256_before_self_field"]
    _require(observed_top == arguments.snapshot_content_sha256, "snapshot CLI content binding changed")
    end_cursor = _selected_end_cursor(authority, arguments.max_budget_tokens)
    groups = _group_ranges(end_cursor, arguments.gradient_accumulation)
    output = arguments.output.resolve()
    _require(output.is_relative_to(ROOT), "run output must remain inside repository")
    if arguments.resume is None:
        _require(not output.exists(), "run output already exists; use --resume")
    else:
        _require(output.is_dir() and not output.is_symlink(), "resume run directory is missing")

    torch, model, lora_config, scope, preattach, hybrid, gpu = _prepare_model(arguments)
    if arguments.resume is None:
        # Do not leave a misleading, non-resumable run directory when model
        # loading or exact adapter-scope validation fails.
        output.mkdir(parents=True)
    run_config = _run_config(arguments, authority, end_cursor)
    run_config_sha = run_config["content_sha256_before_self_field"]
    if arguments.resume is None:
        _write_static_artifacts(output, run_config, authority, model, lora_config, scope, gpu)
    else:
        existing = _load_self_addressed(output / "run_config.json")
        _require(existing == run_config, "resume run_config.json changed")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=arguments.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    scheduler = CosineToFloorSchedule(
        optimizer,
        peak_lr=arguments.learning_rate,
        total_steps=len(groups),
    )
    state = {
        "next_cursor": 0,
        "optimizer_step": 0,
        "cumulative": {
            "input_tokens": 0,
            "budget_tokens": 0,
            "shifted_supervised_tokens": 0,
            "stream_budget_tokens": {
                stream: 0 for stream in sorted(mixed.ALLOWED_STREAMS)
            },
            "stream_input_tokens": {
                stream: 0 for stream in sorted(mixed.ALLOWED_STREAMS)
            },
        },
    }
    resumed_checkpoint: Path | None = None
    if arguments.resume is not None:
        latest_checkpoint = _find_latest_checkpoint(output).resolve()
        resume_path = latest_checkpoint if arguments.resume == "latest" else Path(arguments.resume).resolve()
        _require(
            resume_path == latest_checkpoint,
            "rollback resume is not authorized; use the latest sealed checkpoint",
        )
        restored = _restore_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            authority=authority,
            output=output,
            run_config_sha256=run_config_sha,
            torch=torch,
        )
        state = {
            "next_cursor": restored["cursor"],
            "optimizer_step": restored["optimizer_step"],
            "cumulative": restored["cumulative"],
        }
        resumed_checkpoint = resume_path

        # A process may fail after appending a metric but before sealing the
        # corresponding checkpoint.  Roll logs back to the sealed cursor so a
        # resumed run never duplicates or preserves uncheckpointed steps.
        _truncate_step_log(
            output / "training_metrics.jsonl",
            maximum_optimizer_step=state["optimizer_step"],
            schema=METRICS_SCHEMA,
        )
        _truncate_step_log(
            output / "routing_metrics.jsonl",
            maximum_optimizer_step=state["optimizer_step"],
            schema=ROUTING_SCHEMA,
        )

    _require(state["next_cursor"] <= end_cursor, "resume cursor exceeds selected run")
    _require(state["next_cursor"] % arguments.gradient_accumulation == 0 or state["next_cursor"] == end_cursor, "resume cursor splits an optimizer group")
    start_group = math.ceil(state["next_cursor"] / arguments.gradient_accumulation)
    _require(start_group == state["optimizer_step"], "resume cursor and optimizer step differ")
    routing = RoutingRecorder(model)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    next_checkpoint = (
        (state["cumulative"]["budget_tokens"] // arguments.checkpoint_tokens + 1)
        * arguments.checkpoint_tokens
    )
    metrics_path = output / "training_metrics.jsonl"
    routing_path = output / "routing_metrics.jsonl"

    try:
        for optimizer_step, (group_start, group_stop) in enumerate(groups[start_group:], start=start_group):
            _require(group_start == state["next_cursor"], "schedule cursor skipped or repeated")
            rows = [authority.sequence_for_cursor(cursor) for cursor in range(group_start, group_stop)]
            group_supervised = sum(row["shifted_supervised_token_count"] for row in rows)
            _require(group_supervised > 0, "optimizer group has no supervised tokens")
            optimizer.zero_grad(set_to_none=True)
            collect_routing = arguments.routing_every > 0 and optimizer_step % arguments.routing_every == 0
            if collect_routing:
                routing.start()
            step_started = time.perf_counter()
            group_loss_sum = 0.0
            observed_supervised = 0
            group_input = 0
            group_budget = 0
            group_stream_budget = {
                stream: 0 for stream in sorted(mixed.ALLOWED_STREAMS)
            }
            group_stream_input = {
                stream: 0 for stream in sorted(mixed.ALLOWED_STREAMS)
            }
            for row in rows:
                input_ids = torch.tensor(row["input_ids"], dtype=torch.long, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(row["attention_mask"], dtype=torch.long, device="cuda").unsqueeze(0)
                labels = torch.tensor(row["labels"], dtype=torch.long, device="cuda").unsqueeze(0)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
                loss_sum, supervised = token_cross_entropy_sum(outputs.logits, labels)
                _require(supervised == row["shifted_supervised_token_count"], "runtime target count differs from snapshot")
                if collect_routing:
                    routing.pause()
                (loss_sum / group_supervised).backward()
                if collect_routing:
                    routing.resume()
                group_loss_sum += float(loss_sum.detach().cpu())
                observed_supervised += supervised
                group_input += row["input_token_count"]
                group_budget += row["budget_token_count"]
                group_stream_budget[row["stream"]] += row["budget_token_count"]
                group_stream_input[row["stream"]] += row["input_token_count"]
                del outputs, loss_sum, input_ids, attention_mask, labels
            route_value = routing.stop() if collect_routing else None
            _require(observed_supervised == group_supervised, "optimizer group target count changed")
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            _require(torch.isfinite(gradient_norm).item(), "gradient norm is not finite")
            lr = scheduler.prepare_step()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - step_started
            state["next_cursor"] = group_stop
            state["optimizer_step"] = optimizer_step + 1
            state["cumulative"] = {
                "input_tokens": state["cumulative"]["input_tokens"] + group_input,
                "budget_tokens": state["cumulative"]["budget_tokens"] + group_budget,
                "shifted_supervised_tokens": state["cumulative"]["shifted_supervised_tokens"] + group_supervised,
                "stream_budget_tokens": {
                    stream: state["cumulative"]["stream_budget_tokens"][stream]
                    + group_stream_budget[stream]
                    for stream in sorted(mixed.ALLOWED_STREAMS)
                },
                "stream_input_tokens": {
                    stream: state["cumulative"]["stream_input_tokens"][stream]
                    + group_stream_input[stream]
                    for stream in sorted(mixed.ALLOWED_STREAMS)
                },
            }
            metric = {
                "schema": METRICS_SCHEMA,
                "optimizer_step": optimizer_step + 1,
                "next_cursor": group_stop,
                "cursor_commitment_sha256": authority.schedule[group_stop - 1]["cursor_commitment_sha256"],
                "loss": group_loss_sum / group_supervised,
                "learning_rate": lr,
                "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
                "step_seconds": seconds,
                "input_tokens": group_input,
                "budget_tokens": group_budget,
                "shifted_supervised_tokens": group_supervised,
                "stream_budget_tokens": group_stream_budget,
                "stream_input_tokens": group_stream_input,
                "input_tokens_per_second": group_input / seconds,
                "cumulative": dict(state["cumulative"]),
                "memory": _memory(torch),
            }
            _append_jsonl(metrics_path, metric)
            if route_value is not None:
                _append_jsonl(routing_path, {
                    "schema": ROUTING_SCHEMA,
                    "optimizer_step": optimizer_step + 1,
                    "next_cursor": group_stop,
                    **route_value,
                })
            if state["cumulative"]["budget_tokens"] >= next_checkpoint:
                checkpoint_state = _checkpoint_state(
                    authority=authority,
                    next_cursor=state["next_cursor"],
                    optimizer_step=state["optimizer_step"],
                    cumulative=state["cumulative"],
                    run_config_sha256=run_config_sha,
                )
                _save_checkpoint(
                    output,
                    name=(
                        f"checkpoint-tokens-{state['cumulative']['budget_tokens']:010d}"
                        f"-cursor-{state['next_cursor']:06d}"
                    ),
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state=checkpoint_state,
                    torch=torch,
                )
                while next_checkpoint <= state["cumulative"]["budget_tokens"]:
                    next_checkpoint += arguments.checkpoint_tokens
    finally:
        routing.close()

    final_state = _checkpoint_state(
        authority=authority,
        next_cursor=state["next_cursor"],
        optimizer_step=state["optimizer_step"],
        cumulative=state["cumulative"],
        run_config_sha256=run_config_sha,
    )
    if resumed_checkpoint is not None and start_group == len(groups):
        final_checkpoint = resumed_checkpoint
    else:
        final_checkpoint = _save_checkpoint(
            output,
            name=f"checkpoint-final-cursor-{state['next_cursor']:06d}",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=final_state,
            torch=torch,
        )
    sealed_final_state = _validate_checkpoint_files(final_checkpoint)
    _require(
        sealed_final_state == final_state,
        "final checkpoint state differs from the completed run",
    )
    final_checkpoint_receipt = _load_self_addressed(
        final_checkpoint / "checkpoint_receipt.json"
    )
    training_metrics_receipt = _step_log_receipt(
        metrics_path,
        schema=METRICS_SCHEMA,
        completed_optimizer_steps=state["optimizer_step"],
        final_cursor=state["next_cursor"],
    )
    routing_summary = _routing_summary(
        routing_path,
        interval=arguments.routing_every,
        completed_optimizer_steps=state["optimizer_step"],
    )
    _atomic_json(output / "routing_metrics.json", routing_summary)
    torch.cuda.synchronize()
    memory = _self_addressed({
        "schema": "qwen36-low-regression-sft-memory-profile-v1",
        "gpu": gpu,
        "peak": _memory(torch),
        "elapsed_seconds": time.perf_counter() - started,
        "final_checkpoint": final_checkpoint.relative_to(output).as_posix(),
        "final_checkpoint_receipt": {
            "file_sha256": file_sha256(
                final_checkpoint / "checkpoint_receipt.json"
            ),
            "content_sha256": final_checkpoint_receipt[
                "content_sha256_before_self_field"
            ],
            "state_content_sha256": final_state[
                "content_sha256_before_self_field"
            ],
        },
        "operational_headroom_gib": gpu["total_memory_gib"] - _memory(torch)["peak_reserved_gib"],
    })
    _atomic_json(output / "memory_profile.json", memory)
    result = _self_addressed({
        "schema": "qwen36-low-regression-sft-run-result-v1",
        "run_id": arguments.run_id,
        "status": "complete_selected_schedule",
        "run_config_content_sha256": run_config_sha,
        "next_cursor": state["next_cursor"],
        "selected_end_cursor": end_cursor,
        "optimizer_steps": state["optimizer_step"],
        "cumulative": state["cumulative"],
        "final_checkpoint": final_checkpoint.relative_to(output).as_posix(),
        "scope_identity_sha256": scope["identity_sha256"],
        "preattach_identity_sha256": preattach["identity_sha256"],
        "hybrid_module_count": hybrid["matched_module_count"],
        "training_metrics": training_metrics_receipt,
        "routing_metrics": {
            "path": "routing_metrics.json",
            "file_sha256": file_sha256(output / "routing_metrics.json"),
            "content_sha256": routing_summary[
                "content_sha256_before_self_field"
            ],
            "routing_log": routing_summary["routing_log"],
        },
        "memory_profile": {
            "path": "memory_profile.json",
            "file_sha256": file_sha256(output / "memory_profile.json"),
            "content_sha256": memory["content_sha256_before_self_field"],
        },
    })
    _atomic_json(output / "run_result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--snapshot-manifest", type=Path, required=True)
    result.add_argument("--snapshot-content-sha256", required=True)
    result.add_argument("--variant", choices=sorted(mixed.ALLOWED_VARIANTS), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--routed-rank", type=int, choices=(2, 4), default=4)
    result.add_argument("--shared-rank", type=int, choices=(8, 16), default=16)
    result.add_argument("--shared-only", action="store_true")
    result.add_argument("--learning-rate", type=float, choices=(5e-5, 1e-4, 2e-4), default=1e-4)
    result.add_argument("--gradient-accumulation", type=int, choices=(1, 2), default=1)
    result.add_argument("--checkpoint-tokens", type=int, default=50_000)
    result.add_argument("--max-budget-tokens", type=int)
    result.add_argument("--routing-every", type=int, default=10)
    result.add_argument("--seed", type=int, default=17)
    result.add_argument("--resume", help="checkpoint path or 'latest'")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    _require(re.fullmatch(r"[0-9a-f]{64}", arguments.snapshot_content_sha256) is not None, "invalid snapshot content hash")
    _require(arguments.checkpoint_tokens > 0, "checkpoint token interval must be positive")
    _require(arguments.routing_every >= 0, "routing interval must be nonnegative")
    value = train(arguments)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
