from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from peft.utils.save_and_load import get_peft_model_state_dict
from torch import nn

import train_qwen36_low_regression_sft_v1 as trainer


class FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 123.0}]


class TinyAdapterModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.linear(hidden)


def _tiny_adapter_model():
    return get_peft_model(
        TinyAdapterModel(),
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["linear"],
            lora_dropout=0.0,
            bias="none",
        ),
    )


def test_adapter_restore_accepts_frozen_base_missing_keys_and_is_exact():
    model = _tiny_adapter_model()
    expected = {
        name: tensor.detach().clone()
        for name, tensor in get_peft_model_state_dict(model).items()
    }
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(1.0)

    trainer._set_adapter_state_exact(model, expected)
    observed = get_peft_model_state_dict(model)
    assert observed.keys() == expected.keys()
    assert all(torch.equal(observed[name], expected[name]) for name in expected)


def test_adapter_restore_rejects_incomplete_adapter_state():
    model = _tiny_adapter_model()
    expected = {
        name: tensor.detach().clone()
        for name, tensor in get_peft_model_state_dict(model).items()
    }
    expected.pop(next(iter(expected)))
    with pytest.raises(RuntimeError, match="missing LoRA keys|key inventory"):
        trainer._set_adapter_state_exact(model, expected)


def test_cosine_schedule_has_exact_warmup_peak_floor_and_resume_contract():
    optimizer = FakeOptimizer()
    schedule = trainer.CosineToFloorSchedule(
        optimizer, peak_lr=1e-4, total_steps=100
    )
    assert schedule.warmup_steps == 5
    assert schedule.lr_for_step(0) == pytest.approx(2e-5)
    assert schedule.lr_for_step(4) == pytest.approx(1e-4)
    assert schedule.lr_for_step(99) == pytest.approx(1e-5)
    assert schedule.prepare_step() == pytest.approx(2e-5)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)
    schedule.step()

    resumed = trainer.CosineToFloorSchedule(
        FakeOptimizer(), peak_lr=1e-4, total_steps=100
    )
    resumed.load_state_dict(schedule.state_dict())
    assert resumed.step_count == 1
    mutated = copy.deepcopy(schedule.state_dict())
    mutated["floor_ratio"] = 0.0
    with pytest.raises(RuntimeError, match="scheduler resume differs"):
        resumed.load_state_dict(mutated)


def test_token_cross_entropy_is_shifted_sum_and_rejects_empty_targets():
    logits = torch.tensor(
        [[[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 0, 1]])
    observed, count = trainer.token_cross_entropy_sum(logits, labels)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 3),
        labels[:, 1:].reshape(-1),
        reduction="sum",
    )
    assert count == 2
    assert torch.allclose(observed, expected)
    observed.backward()
    assert logits.grad is not None

    with pytest.raises(RuntimeError, match="no shifted supervised token"):
        trainer.token_cross_entropy_sum(logits.detach(), torch.full_like(labels, -100))


def test_token_limited_cursor_and_optimizer_groups_are_deterministic():
    authority = SimpleNamespace(
        schedule=(
            {"cursor": 0, "cumulative_budget_tokens": 60},
            {"cursor": 1, "cumulative_budget_tokens": 125},
            {"cursor": 2, "cumulative_budget_tokens": 200},
        )
    )
    assert trainer._selected_end_cursor(authority, None) == 3
    assert trainer._selected_end_cursor(authority, 100) == 2
    assert trainer._group_ranges(5, 2) == [(0, 2), (2, 4), (4, 5)]
    with pytest.raises(RuntimeError, match="exceeds snapshot budget"):
        trainer._selected_end_cursor(authority, 201)


class FakeGate(nn.Module):
    def forward(self, hidden: torch.Tensor):
        logits = torch.zeros(hidden.shape[0], 256)
        logits[:, 0] = 2.0
        probabilities = logits.softmax(dim=-1)
        weights, selected = probabilities.topk(2, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        return logits, weights, selected


class FakeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate = FakeGate()


class FakeLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = FakeMlp()


class FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(FakeLayer() for _ in range(40))


class FakeBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()


def test_routing_recorder_uses_full_entropy_and_pause_excludes_recompute():
    base = FakeBase()
    recorder = trainer.RoutingRecorder(base)
    hidden = torch.zeros(3, 4)
    recorder.start()
    for layer in base.model.layers:
        layer.mlp.gate(hidden)
    recorder.pause()
    # Simulate gradient-checkpoint recomputation; these calls must not double
    # token counts or utilization.
    for layer in base.model.layers:
        layer.mlp.gate(hidden)
    result = recorder.stop()
    recorder.close()
    assert result["layer_count"] == 40
    for row in result["layers"].values():
        assert row["tokens"] == 3
        assert row["total_assignments"] == 6
        assert row["router_entropy"] > row["selected_topk_entropy"]
        assert row["expert_counts"][0] == 3


def _synthetic_routing_metric(
    optimizer_step: int, *, tokens: int, next_cursor: int
) -> dict:
    counts = [tokens if expert < trainer.ROUTED_TOP_K else 0 for expert in range(256)]
    layer = {
        "tokens": tokens,
        "total_assignments": tokens * trainer.ROUTED_TOP_K,
        "expert_counts": counts,
        "router_entropy": 5.0,
        "selected_topk_entropy": 2.0,
        "active_experts": trainer.ROUTED_TOP_K,
        "maximum_expert_fraction": 1.0 / trainer.ROUTED_TOP_K,
    }
    return {
        "schema": trainer.ROUTING_SCHEMA,
        "optimizer_step": optimizer_step,
        "next_cursor": next_cursor,
        "layer_count": len(trainer.ROUTER_LAYER_NAMES),
        "layers": {
            name: copy.deepcopy(layer) for name in trainer.ROUTER_LAYER_NAMES
        },
    }


def test_routing_summary_covers_disabled_and_multiple_intervals_exactly(tmp_path: Path):
    disabled_path = tmp_path / "disabled.jsonl"
    disabled = trainer._routing_summary(
        disabled_path, interval=0, completed_optimizer_steps=3
    )
    assert disabled["status"] == "disabled"
    assert disabled["observed_optimizer_steps"] == []
    assert disabled["routing_log"]["sha256"] == trainer.hashlib.sha256(b"").hexdigest()

    path = tmp_path / "routing.jsonl"
    trainer._append_jsonl(
        path, _synthetic_routing_metric(1, tokens=2, next_cursor=1)
    )
    trainer._append_jsonl(
        path, _synthetic_routing_metric(3, tokens=1, next_cursor=3)
    )
    summary = trainer._routing_summary(
        path, interval=2, completed_optimizer_steps=3
    )
    assert summary["status"] == "complete"
    assert summary["observed_optimizer_steps"] == [1, 3]
    assert summary["routing_log"]["rows"] == 2
    assert summary["routing_log"]["sha256"] == trainer.file_sha256(path)
    assert summary["layer_count"] == 40
    for layer in summary["layers"].values():
        assert layer["tokens"] == 3
        assert layer["total_assignments"] == 24
        assert layer["expert_counts"][:8] == [3] * 8
        assert layer["router_entropy"] == pytest.approx(5.0)


def test_routing_summary_rejects_incomplete_layer_and_step_coverage(tmp_path: Path):
    incomplete = _synthetic_routing_metric(1, tokens=1, next_cursor=1)
    incomplete["layers"].pop(trainer.ROUTER_LAYER_NAMES[-1])
    path = tmp_path / "incomplete.jsonl"
    trainer._append_jsonl(path, incomplete)
    with pytest.raises(RuntimeError, match="exact layer inventory"):
        trainer._routing_summary(path, interval=1, completed_optimizer_steps=1)

    wrong_step_path = tmp_path / "wrong-step.jsonl"
    trainer._append_jsonl(
        wrong_step_path, _synthetic_routing_metric(2, tokens=1, next_cursor=2)
    )
    with pytest.raises(RuntimeError, match="step coverage"):
        trainer._routing_summary(
            wrong_step_path, interval=2, completed_optimizer_steps=2
        )


def test_training_metric_receipt_requires_exact_step_and_cursor_coverage(tmp_path: Path):
    path = tmp_path / "training.jsonl"
    trainer._append_jsonl(
        path,
        {"schema": trainer.METRICS_SCHEMA, "optimizer_step": 1, "next_cursor": 2},
    )
    trainer._append_jsonl(
        path,
        {"schema": trainer.METRICS_SCHEMA, "optimizer_step": 2, "next_cursor": 4},
    )
    receipt = trainer._step_log_receipt(
        path,
        schema=trainer.METRICS_SCHEMA,
        completed_optimizer_steps=2,
        final_cursor=4,
    )
    assert receipt["rows"] == 2
    assert receipt["sha256"] == trainer.file_sha256(path)

    with pytest.raises(RuntimeError, match="row count"):
        trainer._step_log_receipt(
            path,
            schema=trainer.METRICS_SCHEMA,
            completed_optimizer_steps=3,
            final_cursor=4,
        )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_checkpoint_receipt_detects_inventory_hash_and_state_tampering(tmp_path: Path):
    checkpoint = tmp_path / "checkpoint-one"
    checkpoint.mkdir()
    state = trainer._self_addressed(
        {"schema": trainer.STATE_SCHEMA, "optimizer_step": 1}
    )
    _write_json(checkpoint / "state.json", state)
    (checkpoint / "dummy.bin").write_bytes(b"safe synthetic fixture")
    receipt = trainer._self_addressed(
        {
            "schema": trainer.CHECKPOINT_SCHEMA,
            "state_content_sha256": state["content_sha256_before_self_field"],
            "files": {
                "dummy.bin": trainer.file_sha256(checkpoint / "dummy.bin"),
                "state.json": trainer.file_sha256(checkpoint / "state.json"),
            },
        }
    )
    _write_json(checkpoint / "checkpoint_receipt.json", receipt)
    assert trainer._validate_checkpoint_files(checkpoint) == state

    (checkpoint / "dummy.bin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checkpoint file changed"):
        trainer._validate_checkpoint_files(checkpoint)


def test_main_validates_snapshot_hash_without_script_only_import_state():
    with pytest.raises(RuntimeError, match="invalid snapshot content hash"):
        trainer.main(
            [
                "--snapshot-manifest", "synthetic.json",
                "--snapshot-content-sha256", "bad",
                "--variant", "protocol_core_100k",
                "--output", "synthetic-output",
                "--run-id", "synthetic",
            ]
        )
