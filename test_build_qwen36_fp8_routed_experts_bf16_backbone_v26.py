#!/usr/bin/env python3
"""CPU-only representation tests for the V26 routed-expert hybrid."""

import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import build_qwen36_fp8_routed_experts_bf16_backbone_v26 as hybrid


def _config(*, fp8=False):
    value = {
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "model_type": "qwen3_5_moe",
        "text_config": {
            "num_hidden_layers": 1,
            "hidden_size": 3,
            "num_experts": 2,
            "moe_intermediate_size": 2,
            "layer_types": ["full_attention"],
        },
    }
    if fp8:
        value["quantization_config"] = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [2, 2],
            "modules_to_not_convert": [
                "model.language_model.layers.0.input_layernorm",
            ],
        }
    return value


def _write_checkpoint(directory: Path, tensors, config):
    directory.mkdir(parents=True)
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, directory / shard, metadata={"format": "pt"})
    total = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": total},
            "weight_map": {name: shard for name in tensors},
        }),
        encoding="utf-8",
    )
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")


def _fixture(tmp_path, *, bad_backbone_shape=False, missing_expert_scale=False):
    bf16 = tmp_path / "bf16"
    fp8 = tmp_path / "fp8"
    output = tmp_path / "hybrid"
    prefix = "model.language_model.layers.0"
    bf16_tensors = {
        f"{prefix}.input_layernorm.weight": torch.arange(
            3, dtype=torch.bfloat16
        ),
        f"{prefix}.self_attn.q_proj.weight": torch.arange(
            9, dtype=torch.bfloat16
        ).reshape(3, 3),
        f"{prefix}.mlp.experts.gate_up_proj": torch.arange(
            24, dtype=torch.bfloat16
        ).reshape(2, 4, 3),
        f"{prefix}.mlp.experts.down_proj": torch.arange(
            12, dtype=torch.bfloat16
        ).reshape(2, 3, 2),
    }
    fp8_tensors = {
        f"{prefix}.input_layernorm.weight": torch.ones(3, dtype=torch.bfloat16),
        f"{prefix}.self_attn.q_proj.weight": torch.ones(
            (2, 3) if bad_backbone_shape else (3, 3),
            dtype=torch.float8_e4m3fn,
        ),
        f"{prefix}.self_attn.q_proj.weight_scale_inv": torch.ones(
            (2, 2), dtype=torch.bfloat16
        ),
    }
    for expert in range(2):
        for projection, shape, scale_shape in (
            ("gate_proj", (2, 3), (1, 2)),
            ("up_proj", (2, 3), (1, 2)),
            ("down_proj", (3, 2), (2, 1)),
        ):
            name = f"{prefix}.mlp.experts.{expert}.{projection}.weight"
            fp8_tensors[name] = torch.full(
                shape, expert + 1, dtype=torch.float8_e4m3fn
            )
            if not (missing_expert_scale and expert == 1 and projection == "up_proj"):
                fp8_tensors[name + "_scale_inv"] = torch.full(
                    scale_shape, expert + 1, dtype=torch.bfloat16
                )
    _write_checkpoint(bf16, bf16_tensors, _config())
    _write_checkpoint(fp8, fp8_tensors, _config(fp8=True))
    return bf16, fp8, output, bf16_tensors, fp8_tensors


def _all_physical_keys(directory):
    result = set()
    for path in directory.glob("*.safetensors"):
        with safe_open(path, framework="pt", device="cpu") as source:
            result.update(source.keys())
    return result


def _indexed_tensor(directory, name):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    with safe_open(
        directory / index["weight_map"][name], framework="pt", device="cpu"
    ) as source:
        return source.get_tensor(name)


def test_v26_representation_is_bf16_backbone_and_only_fp8_routed_experts(tmp_path):
    bf16, fp8, output, bf16_tensors, fp8_tensors = _fixture(tmp_path)
    result = hybrid.build_hybrid_checkpoint_v26(bf16, fp8, output)

    assert result["schema"] == hybrid.SCHEMA
    assert result["tensor_contract"]["bf16_backbone"]["key_count"] == 2
    assert result["tensor_contract"]["fp8_routed_weights"]["key_count"] == 6
    assert result["tensor_contract"]["fp8_routed_scales"]["key_count"] == 6
    assert result["removed_non_routed_scale_count"] == 1
    assert result["target_key_count"] == 14
    assert result["contains_dataset_or_evaluation_content"] is False

    keys = _all_physical_keys(output)
    assert keys == (
        {name for name in bf16_tensors if ".mlp.experts." not in name}
        | {name for name in fp8_tensors if ".mlp.experts." in name}
    )
    assert not any(
        ".mlp.experts." not in name and name.endswith("_scale_inv")
        for name in keys
    )
    for name, expected in bf16_tensors.items():
        if ".mlp.experts." not in name:
            observed = _indexed_tensor(output, name)
            assert observed.dtype == torch.bfloat16
            assert torch.equal(observed, expected)
    for name, expected in fp8_tensors.items():
        if ".mlp.experts." in name:
            observed = _indexed_tensor(output, name)
            assert observed.dtype == expected.dtype
            assert torch.equal(observed, expected)

    config = json.loads((output / "config.json").read_text())
    exclusions = set(config["quantization_config"]["modules_to_not_convert"])
    assert "model.language_model.layers.0.self_attn.q_proj" in exclusions
    assert "model.language_model.layers.0.self_attn.qkv_proj" in exclusions
    assert not any(".mlp.experts." in name for name in exclusions)
    advertised = config["hybrid_routed_experts_fp8_bf16_backbone_v26"]
    assert advertised["target_key_count"] == 14
    assert os.stat(fp8 / "tokenizer.json").st_ino == os.stat(
        output / "tokenizer.json"
    ).st_ino


def test_v26_build_is_deterministic_exclusive_and_fail_closed(tmp_path):
    bf16, fp8, first, _bf16_tensors, _fp8_tensors = _fixture(tmp_path / "good")
    second = tmp_path / "second"
    first_result = hybrid.build_hybrid_checkpoint_v26(bf16, fp8, first)
    second_result = hybrid.build_hybrid_checkpoint_v26(bf16, fp8, second)
    assert first_result == second_result
    assert hybrid.file_sha256(first / "config.json") == hybrid.file_sha256(
        second / "config.json"
    )
    assert hybrid.file_sha256(first / "model.safetensors.index.json") == (
        hybrid.file_sha256(second / "model.safetensors.index.json")
    )
    with pytest.raises(FileExistsError):
        hybrid.build_hybrid_checkpoint_v26(bf16, fp8, first)

    bad_bf16, bad_fp8, bad_output, *_ = _fixture(
        tmp_path / "bad-shape", bad_backbone_shape=True
    )
    with pytest.raises(RuntimeError, match="backbone tensor shape changed"):
        hybrid.build_hybrid_checkpoint_v26(bad_bf16, bad_fp8, bad_output)
    assert not bad_output.exists()

    scale_bf16, scale_fp8, scale_output, *_ = _fixture(
        tmp_path / "bad-scale", missing_expert_scale=True
    )
    with pytest.raises(RuntimeError, match="routed expert scale surface changed"):
        hybrid.build_hybrid_checkpoint_v26(scale_bf16, scale_fp8, scale_output)
    assert not scale_output.exists()


def test_v26_real_source_representation_contract_is_exact_and_aggregate_only():
    if not hybrid.DEFAULT_BF16.is_dir() or not hybrid.DEFAULT_FP8.is_dir():
        pytest.skip("real Qwen checkpoints are unavailable")
    contract = hybrid.inspect_source_contract_v26(
        hybrid.DEFAULT_BF16, hybrid.DEFAULT_FP8
    )["summary"]
    assert contract["bf16_backbone"]["key_count"] == 963
    assert contract["bf16_packed_routed_experts"]["key_count"] == 82
    assert contract["fp8_routed_weights"]["key_count"] == 31_488
    assert contract["fp8_routed_scales"]["key_count"] == 31_488
    assert contract["fp8_removed_backbone_scales"]["key_count"] == 257
    assert contract["target_key_count"] == 63_939
    assert contract["target_element_count"] == 35_953_837_936
    assert contract["target_byte_count"] == 38_890_114_784
    assert contract["target_key_sha256"] == (
        "63945fcf2bd6745aa9f492d1ad488d122294a775568f1af9cbe68998cb8b986b"
    )
    assert contract["bf16_backbone"]["dtype_counts"] == {"BF16": 963}
    assert contract["fp8_routed_weights"]["dtype_counts"] == {
        "F8_E4M3": 31_488
    }
    assert contract["fp8_routed_scales"]["dtype_counts"] == {"BF16": 31_488}
    serialized = json.dumps(contract, sort_keys=True)
    for forbidden in ("tensor_names", "tensor_values", "rows", "prompts"):
        assert forbidden not in serialized
