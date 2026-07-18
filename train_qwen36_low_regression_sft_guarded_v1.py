#!/usr/bin/env python3
"""Guarded entry point for the single-GPU low-regression mixed SFT engine.

The underlying training engine already owns tensor creation, expert-aware
PEFT attachment, optimization, checkpointing, and section-19 artifacts.  This
entry point adds the two launch-boundary checks which must happen outside that
engine:

* validate the sealed mixed-training authority before model or checkpoint
  content is opened; and
* recompute the current hybrid-kernel decision before the model bindings are
  mutated.

It also records an independently content-addressed schedule-window receipt and
post-validates the engine's LoRA-only trainable manifest and explicit
non-evaluation result.  No evaluation or protected source loader is imported
or called here.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Callable

import qwen36_mixed_training_runtime_v1 as mixed
import train_qwen36_low_regression_sft_v1 as engine


ROOT = Path(__file__).resolve().parent
SCHEMA = "qwen36-low-regression-sft-guarded-launch-v1"
RESULT_SCHEMA = "qwen36-low-regression-sft-guarded-result-v1"
SCHEDULE_WINDOW_SCHEMA = "qwen36-low-regression-sft-schedule-window-v1"
TRAINABLE_NAME = re.compile(r"(?:^|\.)lora_[AB]\.default\.weight$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


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


def self_address(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    _require(
        "content_sha256_before_self_field" not in result,
        "self address was supplied before sealing",
    )
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_self_address(value: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label}: object required")
    claimed = value.get("content_sha256_before_self_field")
    unsigned = copy.deepcopy(value)
    unsigned.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str)
        and HEX64.fullmatch(claimed) is not None
        and canonical_sha256(unsigned) == claimed,
        f"{label}: stale or forged self address",
    )
    return value


def build_schedule_window(
    authority: mixed.MixedTrainingAuthority,
    *,
    maximum_budget_tokens: int | None,
    gradient_accumulation: int,
) -> dict[str, Any]:
    """Seal the exact deterministic schedule prefix selected for this run."""

    end_cursor = engine._selected_end_cursor(authority, maximum_budget_tokens)
    groups = engine._group_ranges(end_cursor, gradient_accumulation)
    _require(end_cursor > 0, "selected schedule is empty")
    _require(groups and groups[0][0] == 0, "optimizer groups do not start at zero")

    previous = mixed.ZERO_COMMITMENT
    selected_rows: list[dict[str, Any]] = []
    for expected_cursor, row in enumerate(authority.schedule[:end_cursor]):
        _require(
            isinstance(row, dict) and row.get("cursor") == expected_cursor,
            "selected schedule cursor is not contiguous",
        )
        _require(
            row.get("previous_cursor_commitment_sha256") == previous,
            "selected schedule commitment chain changed",
        )
        commitment = row.get("cursor_commitment_sha256")
        _require(
            isinstance(commitment, str) and HEX64.fullmatch(commitment) is not None,
            "selected schedule commitment is invalid",
        )
        selected_rows.append(
            {
                "cursor": expected_cursor,
                "sequence_id": row.get("sequence_id"),
                "stream": row.get("stream"),
                "budget_token_count": row.get("budget_token_count"),
                "cumulative_budget_tokens": row.get("cumulative_budget_tokens"),
                "cursor_commitment_sha256": commitment,
            }
        )
        previous = commitment

    group_rows = []
    expected_start = 0
    for optimizer_step, (start, stop) in enumerate(groups, 1):
        _require(
            start == expected_start and start < stop <= end_cursor,
            "optimizer grouping skips or repeats a cursor",
        )
        group_rows.append(
            {
                "optimizer_step": optimizer_step,
                "cursor_start": start,
                "cursor_stop": stop,
                "previous_cursor_commitment_sha256": (
                    mixed.ZERO_COMMITMENT
                    if start == 0
                    else authority.schedule[start - 1]["cursor_commitment_sha256"]
                ),
                "cursor_commitment_sha256": authority.schedule[stop - 1][
                    "cursor_commitment_sha256"
                ],
                "sequence_ids": [
                    authority.schedule[cursor]["sequence_id"]
                    for cursor in range(start, stop)
                ],
            }
        )
        expected_start = stop
    _require(expected_start == end_cursor, "optimizer grouping omits a cursor")

    selected_budget = authority.schedule[end_cursor - 1][
        "cumulative_budget_tokens"
    ]
    if maximum_budget_tokens is not None:
        _require(
            selected_budget >= maximum_budget_tokens,
            "token-limited schedule ended below the requested token boundary",
        )
        if end_cursor > 1:
            _require(
                authority.schedule[end_cursor - 2]["cumulative_budget_tokens"]
                < maximum_budget_tokens,
                "token-limited schedule did not choose the minimal prefix",
            )

    return self_address(
        {
            "schema": SCHEDULE_WINDOW_SCHEMA,
            "variant": authority.variant,
            "sequence_set_identity_sha256": authority.sequence_set_identity_sha256,
            "full_schedule_cursor_count": len(authority.schedule),
            "selected_end_cursor": end_cursor,
            "selected_budget_tokens": selected_budget,
            "maximum_budget_tokens": maximum_budget_tokens,
            "gradient_accumulation": gradient_accumulation,
            "optimizer_steps": len(groups),
            "initial_cursor_commitment_sha256": mixed.ZERO_COMMITMENT,
            "selected_final_cursor_commitment_sha256": previous,
            "full_final_cursor_commitment_sha256": (
                authority.final_cursor_commitment_sha256
            ),
            "selected_schedule_rows_sha256": canonical_sha256(selected_rows),
            "optimizer_groups": group_rows,
        }
    )


def validate_resume_window(
    authority: mixed.MixedTrainingAuthority,
    state: dict[str, Any],
    schedule_window: dict[str, Any],
) -> dict[str, Any]:
    """Bind one sealed checkpoint cursor to an optimizer-group boundary."""

    validate_self_address(schedule_window, label="schedule window")
    cursor = mixed.validate_resume_identity(authority, state)
    end_cursor = schedule_window["selected_end_cursor"]
    _require(cursor <= end_cursor, "resume cursor exceeds selected schedule")
    boundaries = {0: 0}
    for group in schedule_window["optimizer_groups"]:
        boundaries[group["cursor_stop"]] = group["optimizer_step"]
    _require(cursor in boundaries, "resume cursor splits an optimizer group")
    optimizer_step = state.get("optimizer_step")
    _require(
        optimizer_step == boundaries[cursor],
        "resume optimizer step differs from schedule cursor",
    )
    return {
        "cursor": cursor,
        "cursor_commitment_sha256": state["cursor_commitment_sha256"],
        "optimizer_step": optimizer_step,
        "schedule_window_content_sha256": schedule_window[
            "content_sha256_before_self_field"
        ],
    }


def validate_current_hybrid_policy() -> dict[str, Any]:
    """Recompute the sealed hybrid decision against the current host."""

    # Deliberately imported after mixed authority validation by ``preflight``.
    # This helper performs no model or dataset loading and allocates no GPU
    # tensors; it checks the persisted synthetic evidence and current host.
    import build_fast_linear_attention_contract_v1 as fast_contract
    from smoke_qwen36_expert_lora_memory_v1 import (
        validate_fast_kernel_training_policy,
    )

    return validate_fast_kernel_training_policy(fast_contract)


def _latest_resume_state(
    arguments: argparse.Namespace,
    authority: mixed.MixedTrainingAuthority,
    schedule_window: dict[str, Any],
) -> dict[str, Any]:
    if arguments.resume is None:
        return {
            "cursor": 0,
            "cursor_commitment_sha256": mixed.ZERO_COMMITMENT,
            "optimizer_step": 0,
            "schedule_window_content_sha256": schedule_window[
                "content_sha256_before_self_field"
            ],
        }
    output = arguments.output.resolve()
    latest = engine._find_latest_checkpoint(output).resolve()
    selected = latest if arguments.resume == "latest" else Path(arguments.resume).resolve()
    _require(selected == latest, "rollback resume is not authorized")
    state = engine._validate_checkpoint_files(selected)
    return validate_resume_window(authority, state, schedule_window)


def preflight(
    arguments: argparse.Namespace,
    *,
    authority_loader: Callable[..., mixed.MixedTrainingAuthority] | None = None,
    hybrid_validator: Callable[[], dict[str, Any]] | None = None,
) -> tuple[mixed.MixedTrainingAuthority, dict[str, Any]]:
    """Validate authority first, then all non-model launch prerequisites."""

    loader = mixed.load_training_authority if authority_loader is None else authority_loader
    validate_hybrid = (
        validate_current_hybrid_policy
        if hybrid_validator is None
        else hybrid_validator
    )

    # This is intentionally the first callback.  A failed authority must stop
    # before hybrid inspection, checkpoint reads, model loading, or training.
    authority = loader(arguments.snapshot_manifest, variant=arguments.variant)
    observed_top = authority.top_manifest.get("content_sha256_before_self_field")
    _require(
        observed_top == arguments.snapshot_content_sha256,
        "snapshot CLI content binding changed",
    )
    schedule_window = build_schedule_window(
        authority,
        maximum_budget_tokens=arguments.max_budget_tokens,
        gradient_accumulation=arguments.gradient_accumulation,
    )
    hybrid = validate_hybrid()
    _require(isinstance(hybrid, dict), "hybrid policy receipt is absent")
    _require(
        hybrid.get("selected") == "hybrid_training"
        and hybrid.get("all_four_physical_gpus_revalidated") is True,
        "current host does not satisfy the sealed hybrid policy",
    )
    resume = _latest_resume_state(arguments, authority, schedule_window)

    plan = ROOT / "plan.md"
    protocol_files = (
        plan,
        Path(__file__).resolve(),
        Path(engine.__file__).resolve(),
        Path(mixed.__file__).resolve(),
        (ROOT / "qwen36_expert_lora_v1.py").resolve(),
        (ROOT / "qwen36_section19_artifact_contract_v1.py").resolve(),
        (ROOT / "smoke_qwen36_expert_lora_memory_v1.py").resolve(),
        (ROOT / "build_fast_linear_attention_contract_v1.py").resolve(),
    )
    _require(all(path.is_file() and not path.is_symlink() for path in protocol_files),
             "protocol implementation receipt path changed")
    receipt = self_address(
        {
            "schema": SCHEMA,
            "status": "launch_prerequisites_passed",
            "run_id": arguments.run_id,
            "authority": {
                "top_manifest_content_sha256": observed_top,
                "variant": authority.variant,
                "sequence_set_identity_sha256": (
                    authority.sequence_set_identity_sha256
                ),
                "final_cursor_commitment_sha256": (
                    authority.final_cursor_commitment_sha256
                ),
            },
            "schedule_window": schedule_window,
            "resume": resume,
            "hybrid_kernel_policy": copy.deepcopy(hybrid),
            "configuration": {
                "routed_rank": None if arguments.shared_only else arguments.routed_rank,
                "shared_rank": arguments.shared_rank,
                "shared_only": arguments.shared_only,
                "learning_rate": arguments.learning_rate,
                "gradient_accumulation": arguments.gradient_accumulation,
                "checkpoint_tokens": arguments.checkpoint_tokens,
                "maximum_budget_tokens": arguments.max_budget_tokens,
                "routing_every": arguments.routing_every,
                "seed": arguments.seed,
                "bf16": True,
                "gradient_checkpointing": True,
                "cache_disabled": True,
                "vision_excluded": True,
                "optimizer": "AdamW",
                "gradient_clip_norm": 1.0,
                "warmup_ratio": 0.05,
                "cosine_floor_ratio": 0.10,
            },
            "protocol_receipts": {
                path.relative_to(ROOT).as_posix(): file_sha256(path)
                for path in protocol_files
            },
            "authority_bound_before_hybrid_or_model_use": True,
            "protected_evaluation_source_opened": False,
            "evaluation_loader_imported": False,
        }
    )
    return authority, receipt


def validate_trainable_manifest(
    path: Path,
    *,
    shared_only: bool,
) -> dict[str, Any]:
    """Prove the persisted optimizer scope contains only LoRA A/B tensors."""

    _require(path.is_file() and not path.is_symlink(), "trainable manifest is missing")
    raw = path.read_bytes()
    _require(raw and raw.endswith(b"\n"), "trainable manifest is incomplete")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        value = json.loads(line)
        _require(isinstance(value, dict), f"trainable row {line_number} is invalid")
        _require(
            set(value) == {"name", "shape", "elements", "dtype"},
            f"trainable row {line_number} fields changed",
        )
        name = value.get("name")
        _require(
            isinstance(name, str) and TRAINABLE_NAME.search(name) is not None,
            f"non-LoRA tensor entered optimizer scope: {name!r}",
        )
        _require(
            isinstance(value.get("shape"), list)
            and value["shape"]
            and isinstance(value.get("elements"), int)
            and not isinstance(value["elements"], bool)
            and value["elements"] > 0,
            f"trainable row {line_number} tensor metadata changed",
        )
        _require(
            value.get("dtype") == "torch.float32",
            f"trainable row {line_number} is not FP32 before AdamW",
        )
        rows.append(value)
    expected = 240 if shared_only else 400
    _require(len(rows) == expected, "trainable LoRA tensor cardinality changed")
    names = [row["name"] for row in rows]
    _require(len(names) == len(set(names)), "trainable LoRA tensor name duplicated")
    return {
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "trainable_tensors": len(rows),
        "trainable_elements": sum(row["elements"] for row in rows),
        "only_lora_a_and_b": True,
    }


def validate_engine_artifacts(
    output: Path,
    *,
    arguments: argparse.Namespace,
    engine_result: dict[str, Any],
    launch_receipt: dict[str, Any],
) -> dict[str, Any]:
    """Validate single-run receipts without opening any evaluation source."""

    _require(output.is_dir() and not output.is_symlink(), "engine output is missing")
    run_config = validate_self_address(
        json.loads((output / "run_config.json").read_text(encoding="utf-8")),
        label="run config",
    )
    _require(run_config.get("run_id") == arguments.run_id, "run ID changed")
    _require(
        run_config.get("model", {}).get("dtype") == "bfloat16"
        and run_config.get("model", {}).get("vision_excluded") is True,
        "BF16 text-only model contract changed",
    )
    optimization = run_config.get("optimization", {})
    _require(
        optimization.get("gradient_checkpointing") is True
        and optimization.get("cache") is False
        and optimization.get("optimizer") == "AdamW"
        and optimization.get("warmup_ratio") == 0.05
        and optimization.get("schedule") == "cosine_to_10_percent_peak"
        and optimization.get("gradient_clip_norm") == 1.0,
        "optimizer/runtime protocol changed",
    )
    validate_self_address(launch_receipt, label="guarded launch receipt")
    _require(
        run_config.get("runtime", {}).get(
            "hybrid_linear_attention_policy_receipt"
        )
        == launch_receipt.get("hybrid_kernel_policy"),
        "hybrid policy changed between guarded preflight and model preparation",
    )
    trainable = validate_trainable_manifest(
        output / "trainable_parameters.txt", shared_only=arguments.shared_only
    )
    evaluation = json.loads(
        (output / "evaluation_results.json").read_text(encoding="utf-8")
    )
    _require(
        evaluation
        == {
            "status": "not_run_by_training_process",
            "reason": "development and final evaluation are separate sealed stages",
        },
        "training process evaluation boundary changed",
    )
    dataset = validate_self_address(
        json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8")),
        label="dataset manifest",
    )
    _require(
        dataset.get("protected_evaluation_content_opened") is False,
        "dataset manifest reports protected evaluation access",
    )
    artifacts = validate_self_address(
        json.loads((output / "run_artifact_receipts.json").read_text(encoding="utf-8")),
        label="run artifact contract",
    )
    _require(
        artifacts.get("gates", {}).get("protected_evaluation_content_opened")
        is False,
        "run artifact contract reports protected evaluation access",
    )
    validate_self_address(engine_result, label="engine result")
    return {
        "run_config_content_sha256": run_config["content_sha256_before_self_field"],
        "dataset_manifest_content_sha256": dataset[
            "content_sha256_before_self_field"
        ],
        "run_artifact_contract_content_sha256": artifacts[
            "content_sha256_before_self_field"
        ],
        "engine_result_content_sha256": engine_result[
            "content_sha256_before_self_field"
        ],
        "trainable_parameters": trainable,
        "protected_evaluation_source_opened": False,
        "evaluation_run_by_training_process": False,
    }


def execute(
    arguments: argparse.Namespace,
    *,
    authority_loader: Callable[..., mixed.MixedTrainingAuthority] | None = None,
    hybrid_validator: Callable[[], dict[str, Any]] | None = None,
    training_engine: Callable[[argparse.Namespace], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the guarded preflight, engine, and post-run artifact audit."""

    _, launch_receipt = preflight(
        arguments,
        authority_loader=authority_loader,
        hybrid_validator=hybrid_validator,
    )
    run_engine = engine.train if training_engine is None else training_engine
    engine_result = run_engine(arguments)
    post = validate_engine_artifacts(
        arguments.output.resolve(),
        arguments=arguments,
        engine_result=engine_result,
        launch_receipt=launch_receipt,
    )
    # Supplemental receipts are intentionally written only after the engine's
    # own section-19 contract is complete.  They do not replace or weaken it.
    engine._atomic_json(arguments.output / "guarded_launch_receipt.json", launch_receipt)
    result = self_address(
        {
            "schema": RESULT_SCHEMA,
            "status": "complete_selected_schedule",
            "run_id": arguments.run_id,
            "guarded_launch_receipt": {
                "path": "guarded_launch_receipt.json",
                "file_sha256": file_sha256(
                    arguments.output / "guarded_launch_receipt.json"
                ),
                "content_sha256": launch_receipt[
                    "content_sha256_before_self_field"
                ],
            },
            "post_run_validation": post,
        }
    )
    engine._atomic_json(arguments.output / "guarded_run_result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    return engine.parser()


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    _require(
        HEX64.fullmatch(arguments.snapshot_content_sha256) is not None,
        "invalid snapshot content hash",
    )
    _require(arguments.checkpoint_tokens > 0, "checkpoint token interval must be positive")
    _require(arguments.routing_every >= 0, "routing interval must be nonnegative")
    result = execute(arguments)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
