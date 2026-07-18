from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import qwen36_mixed_training_runtime_v1 as mixed
import train_qwen36_low_regression_sft_guarded_v1 as guarded
import train_qwen36_low_regression_sft_v1 as engine


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _authority() -> SimpleNamespace:
    rows = (
        {
            "cursor": 0,
            "sequence_id": "sequence-0",
            "stream": "domain_qa",
            "budget_token_count": 60,
            "cumulative_budget_tokens": 60,
            "previous_cursor_commitment_sha256": mixed.ZERO_COMMITMENT,
            "cursor_commitment_sha256": HEX_A,
        },
        {
            "cursor": 1,
            "sequence_id": "sequence-1",
            "stream": "replay",
            "budget_token_count": 65,
            "cumulative_budget_tokens": 125,
            "previous_cursor_commitment_sha256": HEX_A,
            "cursor_commitment_sha256": HEX_B,
        },
        {
            "cursor": 2,
            "sequence_id": "sequence-2",
            "stream": "raw_markdown",
            "budget_token_count": 75,
            "cumulative_budget_tokens": 200,
            "previous_cursor_commitment_sha256": HEX_B,
            "cursor_commitment_sha256": HEX_C,
        },
    )
    return SimpleNamespace(
        variant="protocol_core_100k",
        top_manifest={"content_sha256_before_self_field": HEX_A},
        sequence_set_identity_sha256=HEX_B,
        final_cursor_commitment_sha256=HEX_C,
        schedule=rows,
    )


def _arguments(tmp_path: Path, **overrides: object) -> SimpleNamespace:
    values = {
        "snapshot_manifest": tmp_path / "synthetic-manifest.json",
        "snapshot_content_sha256": HEX_A,
        "variant": "protocol_core_100k",
        "output": tmp_path / "synthetic-run",
        "run_id": "synthetic-only",
        "routed_rank": 4,
        "shared_rank": 16,
        "shared_only": False,
        "learning_rate": 1e-4,
        "gradient_accumulation": 2,
        "checkpoint_tokens": 50_000,
        "max_budget_tokens": 100,
        "routing_every": 10,
        "seed": 17,
        "resume": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_bad_authority_stops_before_hybrid_checkpoint_or_model_engine(tmp_path: Path):
    calls: list[str] = []

    def bad_loader(*_args: object, **_kwargs: object):
        calls.append("authority")
        raise mixed.SnapshotContractError("synthetic bad authority")

    def hybrid():
        calls.append("hybrid")
        raise AssertionError("hybrid must not run")

    def training(_arguments: object):
        calls.append("model_engine")
        raise AssertionError("model engine must not run")

    with pytest.raises(mixed.SnapshotContractError, match="bad authority"):
        guarded.execute(
            _arguments(tmp_path),
            authority_loader=bad_loader,
            hybrid_validator=hybrid,
            training_engine=training,
        )
    assert calls == ["authority"]
    assert not (tmp_path / "synthetic-run").exists()


def test_snapshot_digest_mismatch_stops_before_hybrid_or_model_engine(tmp_path: Path):
    calls: list[str] = []

    def loader(*_args: object, **_kwargs: object):
        calls.append("authority")
        return _authority()

    def hybrid():
        calls.append("hybrid")
        raise AssertionError("hybrid must not run")

    def training(_arguments: object):
        calls.append("model_engine")
        raise AssertionError("model engine must not run")

    arguments = _arguments(tmp_path, snapshot_content_sha256=HEX_C)
    with pytest.raises(RuntimeError, match="snapshot CLI content binding"):
        guarded.execute(
            arguments,
            authority_loader=loader,
            hybrid_validator=hybrid,
            training_engine=training,
        )
    assert calls == ["authority"]
    assert not arguments.output.exists()


def test_engine_itself_binds_snapshot_digest_before_prepare_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls: list[str] = []

    def loader(*_args: object, **_kwargs: object):
        calls.append("authority")
        return _authority()

    def prepare(_arguments: object):
        calls.append("prepare_model")
        raise AssertionError("model preparation must not run")

    monkeypatch.setattr(engine.mixed, "load_training_authority", loader)
    monkeypatch.setattr(engine, "_prepare_model", prepare)
    arguments = _arguments(tmp_path, snapshot_content_sha256=HEX_C)
    with pytest.raises(RuntimeError, match="snapshot CLI content binding"):
        engine.train(arguments)
    assert calls == ["authority"]
    assert not arguments.output.exists()


def test_schedule_window_selects_minimal_exact_prefix_and_groups_deterministically():
    authority = _authority()
    first = guarded.build_schedule_window(
        authority, maximum_budget_tokens=100, gradient_accumulation=2
    )
    second = guarded.build_schedule_window(
        authority, maximum_budget_tokens=100, gradient_accumulation=2
    )
    assert first == second
    assert first["selected_end_cursor"] == 2
    assert first["selected_budget_tokens"] == 125
    assert first["optimizer_steps"] == 1
    assert first["selected_final_cursor_commitment_sha256"] == HEX_B
    assert first["optimizer_groups"] == [
        {
            "optimizer_step": 1,
            "cursor_start": 0,
            "cursor_stop": 2,
            "previous_cursor_commitment_sha256": mixed.ZERO_COMMITMENT,
            "cursor_commitment_sha256": HEX_B,
            "sequence_ids": ["sequence-0", "sequence-1"],
        }
    ]

    full = guarded.build_schedule_window(
        authority, maximum_budget_tokens=None, gradient_accumulation=2
    )
    assert [(row["cursor_start"], row["cursor_stop"]) for row in full["optimizer_groups"]] == [
        (0, 2),
        (2, 3),
    ]


def test_schedule_window_rejects_commitment_chain_drift():
    authority = _authority()
    rows = list(copy.deepcopy(authority.schedule))
    rows[1]["previous_cursor_commitment_sha256"] = HEX_C
    authority.schedule = tuple(rows)
    with pytest.raises(RuntimeError, match="commitment chain"):
        guarded.build_schedule_window(
            authority, maximum_budget_tokens=None, gradient_accumulation=1
        )


def test_resume_identity_requires_exact_commitment_group_boundary_and_step():
    authority = _authority()
    window = guarded.build_schedule_window(
        authority, maximum_budget_tokens=None, gradient_accumulation=2
    )
    valid = {
        "variant": authority.variant,
        "sequence_set_identity_sha256": authority.sequence_set_identity_sha256,
        "cursor": 2,
        "cursor_commitment_sha256": HEX_B,
        "optimizer_step": 1,
    }
    receipt = guarded.validate_resume_window(authority, valid, window)
    assert receipt["cursor"] == 2
    assert receipt["optimizer_step"] == 1

    wrong_commitment = dict(valid, cursor_commitment_sha256=HEX_C)
    with pytest.raises(mixed.SnapshotContractError, match="commitment"):
        guarded.validate_resume_window(authority, wrong_commitment, window)

    split_group = dict(valid, cursor=1, cursor_commitment_sha256=HEX_A)
    with pytest.raises(RuntimeError, match="splits an optimizer group"):
        guarded.validate_resume_window(authority, split_group, window)

    wrong_step = dict(valid, optimizer_step=2)
    with pytest.raises(RuntimeError, match="optimizer step"):
        guarded.validate_resume_window(authority, wrong_step, window)


def test_scheduler_exact_five_percent_warmup_peak_and_ten_percent_floor():
    optimizer = SimpleNamespace(param_groups=[{"lr": 0.0}])
    schedule = engine.CosineToFloorSchedule(
        optimizer, peak_lr=2e-4, total_steps=100
    )
    assert schedule.warmup_steps == 5
    assert schedule.lr_for_step(0) == pytest.approx(4e-5)
    assert schedule.lr_for_step(4) == pytest.approx(2e-4)
    assert schedule.lr_for_step(99) == pytest.approx(2e-5)


class _SyntheticParameter:
    def __init__(self, dtype: torch.dtype, *, trainable: bool = True) -> None:
        self.dtype = dtype
        self.requires_grad = trainable


class _SyntheticNamedParameters:
    def __init__(self, rows: list[tuple[str, _SyntheticParameter]]) -> None:
        self.rows = rows

    def named_parameters(self):
        return iter(self.rows)


def test_fp32_lora_scope_is_proven_before_adamw():
    valid = _SyntheticNamedParameters(
        [
            ("layer.lora_A.default.weight", _SyntheticParameter(torch.float32)),
            ("layer.lora_B.default.weight", _SyntheticParameter(torch.float32)),
            ("layer.base_layer.weight", _SyntheticParameter(torch.bfloat16, trainable=False)),
        ]
    )
    engine._validate_fp32_lora_optimizer_scope(
        valid, float32_dtype=torch.float32
    )

    wrong_dtype = _SyntheticNamedParameters(
        [("layer.lora_A.default.weight", _SyntheticParameter(torch.bfloat16))]
    )
    with pytest.raises(RuntimeError, match="FP32 before AdamW"):
        engine._validate_fp32_lora_optimizer_scope(
            wrong_dtype, float32_dtype=torch.float32
        )

    non_lora = _SyntheticNamedParameters(
        [("layer.gate.weight", _SyntheticParameter(torch.float32))]
    )
    with pytest.raises(RuntimeError, match="non-LoRA"):
        engine._validate_fp32_lora_optimizer_scope(
            non_lora, float32_dtype=torch.float32
        )


def test_mutated_runtime_policy_requires_exact_flags_and_hybrid_bindings():
    expected = {"chunk_gated_delta_rule": "synthetic.fast"}
    model = SimpleNamespace(
        config=SimpleNamespace(
            use_cache=False,
            output_router_logits=False,
            router_aux_loss_coef=0.0,
        ),
        is_gradient_checkpointing=True,
    )
    engine._validate_mutated_runtime_policy(
        model,
        hybrid={"matched_module_count": 30, "bindings": expected},
        expected_hybrid_bindings=expected,
    )
    with pytest.raises(RuntimeError, match="bindings"):
        engine._validate_mutated_runtime_policy(
            model,
            hybrid={"matched_module_count": 30, "bindings": {}},
            expected_hybrid_bindings=expected,
        )
    model.config.use_cache = True
    with pytest.raises(RuntimeError, match="cache"):
        engine._validate_mutated_runtime_policy(
            model,
            hybrid={"matched_module_count": 30, "bindings": expected},
            expected_hybrid_bindings=expected,
        )


def test_pinned_router_tuple_matches_routing_recorder_without_gpu():
    import qwen36_expert_lora_v1 as expert_lora
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeTopKRouter,
    )

    architecture = expert_lora.load_architecture_contract()
    source = architecture["architecture"]["implementation_sources"][
        "transformers_qwen_modeling"
    ]
    assert guarded.file_sha256(guarded.ROOT / source["path"]) == source["sha256"]
    router = Qwen3_5MoeTopKRouter(
        SimpleNamespace(num_experts_per_tok=8, num_experts=256, hidden_size=16)
    )
    output = router(torch.zeros(3, 16))
    assert isinstance(output, tuple) and len(output) == 3
    logits, weights, selected = output
    assert logits.shape == (3, 256)
    assert weights.shape == selected.shape == (3, 8)

    # Exercise the exact hook body used by the trainer, without constructing a
    # model or touching CUDA.
    recorder = object.__new__(engine.RoutingRecorder)
    recorder.enabled = True
    recorder.rows = {}
    recorder._hook("model.layers.0.mlp.gate")(router, (), output)
    row = recorder.rows["model.layers.0.mlp.gate"]
    assert row["tokens"] == 3
    assert int(row["expert_counts"].sum()) == 24


def _trainable_row(name: str) -> dict:
    return {"name": name, "shape": [2, 3], "elements": 6, "dtype": "torch.float32"}


def test_trainable_manifest_accepts_only_exact_lora_tensors(tmp_path: Path):
    path = tmp_path / "trainable_parameters.txt"
    rows = [
        _trainable_row(
            f"base_model.model.model.layers.{index}.mlp.shared_expert."
            f"gate_proj.lora_{side}.default.weight"
        )
        for index in range(120)
        for side in ("A", "B")
    ]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = guarded.validate_trainable_manifest(path, shared_only=True)
    assert receipt["trainable_tensors"] == 240
    assert receipt["only_lora_a_and_b"] is True

    rows[-1]["dtype"] = "torch.bfloat16"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not FP32"):
        guarded.validate_trainable_manifest(path, shared_only=True)

    rows[-1]["dtype"] = "torch.float32"
    rows[-1]["name"] = "base_model.model.model.layers.0.mlp.gate.weight"
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="non-LoRA tensor"):
        guarded.validate_trainable_manifest(path, shared_only=True)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_post_run_audit_proves_training_did_not_run_or_open_evaluation(tmp_path: Path):
    output = tmp_path / "synthetic-run"
    output.mkdir()
    arguments = _arguments(tmp_path, output=output, shared_only=True)
    hybrid_policy = {
        "selected": "hybrid_training",
        "all_four_physical_gpus_revalidated": True,
    }
    launch_receipt = guarded.self_address(
        {
            "schema": guarded.SCHEMA,
            "hybrid_kernel_policy": hybrid_policy,
        }
    )
    run_config = guarded.self_address(
        {
            "schema": engine.RUN_SCHEMA,
            "run_id": arguments.run_id,
            "model": {"dtype": "bfloat16", "vision_excluded": True},
            "optimization": {
                "gradient_checkpointing": True,
                "cache": False,
                "optimizer": "AdamW",
                "warmup_ratio": 0.05,
                "schedule": "cosine_to_10_percent_peak",
                "gradient_clip_norm": 1.0,
            },
            "runtime": {
                "hybrid_linear_attention_policy_receipt": hybrid_policy,
            },
        }
    )
    dataset = guarded.self_address(
        {"schema": "synthetic-dataset-v1", "protected_evaluation_content_opened": False}
    )
    artifacts = guarded.self_address(
        {
            "schema": "synthetic-artifacts-v1",
            "gates": {"protected_evaluation_content_opened": False},
        }
    )
    engine_result = guarded.self_address(
        {"schema": "synthetic-engine-result-v1", "status": "complete"}
    )
    _write_json(output / "run_config.json", run_config)
    _write_json(output / "dataset_manifest.json", dataset)
    _write_json(output / "run_artifact_receipts.json", artifacts)
    _write_json(
        output / "evaluation_results.json",
        {
            "status": "not_run_by_training_process",
            "reason": "development and final evaluation are separate sealed stages",
        },
    )
    rows = [
        _trainable_row(
            f"base_model.model.model.layers.{index}.mlp.shared_expert."
            f"gate_proj.lora_{side}.default.weight"
        )
        for index in range(120)
        for side in ("A", "B")
    ]
    (output / "trainable_parameters.txt").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    receipt = guarded.validate_engine_artifacts(
        output,
        arguments=arguments,
        engine_result=engine_result,
        launch_receipt=launch_receipt,
    )
    assert receipt["protected_evaluation_source_opened"] is False
    assert receipt["evaluation_run_by_training_process"] is False

    _write_json(
        output / "evaluation_results.json",
        {"status": "complete", "per_item_metrics": ["forbidden"]},
    )
    with pytest.raises(RuntimeError, match="evaluation boundary"):
        guarded.validate_engine_artifacts(
            output,
            arguments=arguments,
            engine_result=engine_result,
            launch_receipt=launch_receipt,
        )

    restored_evaluation = {
        "status": "not_run_by_training_process",
        "reason": "development and final evaluation are separate sealed stages",
    }
    _write_json(output / "evaluation_results.json", restored_evaluation)
    stale_launch = copy.deepcopy(launch_receipt)
    stale_launch["hybrid_kernel_policy"]["selected"] = "torch_reference"
    unsigned = copy.deepcopy(stale_launch)
    unsigned.pop("content_sha256_before_self_field")
    stale_launch["content_sha256_before_self_field"] = guarded.canonical_sha256(
        unsigned
    )
    with pytest.raises(RuntimeError, match="hybrid policy changed"):
        guarded.validate_engine_artifacts(
            output,
            arguments=arguments,
            engine_result=engine_result,
            launch_receipt=stale_launch,
        )
