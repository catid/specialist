from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from peft import get_peft_model
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig

import qwen36_expert_lora_v1 as expert_lora


class TinyConfig(PretrainedConfig):
    model_type = "specialist_qwen36_expert_lora_test"


class TinyExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(256, 4, 6))
        self.down_proj = nn.Parameter(torch.randn(256, 6, 3))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states


class TinySharedExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 3, bias=False)
        self.up_proj = nn.Linear(4, 3, bias=False)
        self.down_proj = nn.Linear(3, 4, bias=False)


class TinyMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = TinyExperts()
        self.shared_expert = TinySharedExpert()


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = TinyMlp()


class TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TinyLayer() for _ in range(40))


class TinyCausalLM(PreTrainedModel):
    config_class = TinyConfig

    def __init__(self, config: TinyConfig) -> None:
        super().__init__(config)
        self.model = TinyBackbone()

    def forward(self, input_ids: torch.Tensor | None = None, **_: object):
        assert input_ids is not None
        return self.model.layers[0].mlp.experts(input_ids)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs: object):
        return {"input_ids": input_ids, **kwargs}


def _contract() -> dict:
    return expert_lora.load_architecture_contract()


def test_contract_drives_exact_full_target_names_and_ranks():
    spec = expert_lora.spec_from_contract(_contract(), routed_rank=2)
    assert len(spec.shared_targets) == 120
    assert len(spec.routed_targets) == 80
    assert spec.shared_rank == spec.shared_alpha == 16
    assert spec.routed_rank == spec.routed_alpha == 2
    assert all(name.startswith("model.layers.") for name in spec.all_targets)

    model = TinyCausalLM(TinyConfig())
    preattach = expert_lora.validate_preattach_model(model, spec)
    assert len(preattach["shared"]) == 120
    assert len(preattach["routed"]) == 80

    config = expert_lora.make_lora_config(spec)
    assert set(config.target_modules) == set(spec.shared_targets)
    assert set(config.target_parameters) == set(spec.routed_targets)
    assert set(config.rank_pattern) == set(spec.routed_targets)
    assert set(config.rank_pattern.values()) == {2}


def test_installed_peft_attaches_both_fused_parameters_and_audits_exact_scope():
    spec = expert_lora.spec_from_contract(_contract())
    base = TinyCausalLM(TinyConfig())
    expert_lora.validate_preattach_model(base, spec)
    adapted = get_peft_model(base, expert_lora.make_lora_config(spec))
    audit = expert_lora.audit_postattach_scope(adapted, spec)

    assert audit["target_count"] == 200
    assert audit["shared_target_count"] == 120
    assert audit["routed_target_count"] == 80
    assert audit["trainable_tensor_count"] == 400
    assert {row["rank"] for row in audit["wrappers"] if row["kind"] == "shared_module"} == {16}
    assert {row["rank"] for row in audit["wrappers"] if row["kind"] == "routed_parameter"} == {4}
    routed_names = {row["target"] for row in audit["wrappers"] if row["kind"] == "routed_parameter"}
    assert "model.layers.0.mlp.experts.gate_up_proj" in routed_names
    assert "model.layers.0.mlp.experts.down_proj" in routed_names


def test_shared_only_ablation_and_scope_violation_fail_closed():
    spec = expert_lora.shared_only_spec_from_contract(_contract())
    base = TinyCausalLM(TinyConfig())
    adapted = get_peft_model(base, expert_lora.make_lora_config(spec))
    audit = expert_lora.audit_postattach_scope(adapted, spec)
    assert audit["target_count"] == 120
    assert audit["routed_target_count"] == 0
    assert audit["trainable_tensor_count"] == 240

    adapted.base_model.model.model.layers[0].mlp.experts.gate_up_proj.requires_grad_(True)
    with pytest.raises(RuntimeError, match="non-LoRA tensor"):
        expert_lora.audit_postattach_scope(adapted, spec)


def test_mutated_architecture_contract_is_rejected(tmp_path: Path):
    value = copy.deepcopy(_contract())
    value["architecture"]["adapter_target_contract"]["shared_rank"] = 8
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="self hash changed"):
        expert_lora.load_architecture_contract(path)
