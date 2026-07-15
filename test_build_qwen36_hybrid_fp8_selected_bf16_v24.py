import json
import os
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import build_qwen36_hybrid_fp8_selected_bf16_v24 as hybrid
import es_layer_plan


def _config(*, fp8=False):
    types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(40)
    ]
    value = {
        "text_config": {
            "num_hidden_layers": 40,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "layer_types": types,
        }
    }
    if fp8:
        value["quantization_config"] = {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "modules_to_not_convert": ["existing.module"],
        }
    return value


def _write_checkpoint(directory: Path, tensors, config):
    directory.mkdir()
    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, directory / shard, metadata={"format": "pt"})
    total = sum(item.numel() * item.element_size() for item in tensors.values())
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total},
        "weight_map": {name: shard for name in tensors},
    }))
    (directory / "config.json").write_text(json.dumps(config))
    (directory / "tokenizer.json").write_text("{}")


def _fixture(tmp_path, *, mismatch=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bf16, fp8, output = tmp_path / "bf16", tmp_path / "fp8", tmp_path / "out"
    bf16.mkdir()
    (bf16 / "config.json").write_text(json.dumps(_config()))
    plan = es_layer_plan.plan_manifest(
        bf16, "front", ["dense"], custom_layers=[20, 21, 22, 23]
    )
    (bf16 / "config.json").unlink()
    bf16.rmdir()
    units = plan["units"]
    bf16_tensors = {
        name: torch.full((2, 2), index / 10, dtype=torch.bfloat16)
        for index, name in enumerate(units, 1)
    }
    fp8_tensors = {}
    for name in units:
        shape = (3, 2) if mismatch and name == units[0] else (2, 2)
        fp8_tensors[name] = torch.ones(shape, dtype=torch.uint8)
        fp8_tensors[name + "_scale_inv"] = torch.ones((1,), dtype=torch.float32)
    fp8_tensors["model.language_model.layers.0.input_layernorm.weight"] = (
        torch.ones((2,), dtype=torch.bfloat16)
    )
    _write_checkpoint(bf16, bf16_tensors, _config())
    _write_checkpoint(fp8, fp8_tensors, _config(fp8=True))
    return bf16, fp8, output, units, bf16_tensors


def _keys(path):
    with safe_open(path, framework="pt", device="cpu") as source:
        return set(source.keys())


def test_hybrid_replaces_exact_selected_partition_and_strips_scales(tmp_path):
    bf16, fp8, output, units, originals = _fixture(tmp_path)
    result = hybrid.build_hybrid_checkpoint(bf16, fp8, output)

    assert result["schema"] == hybrid.SCHEMA
    assert result["selected_unit_count"] == 35
    assert result["selected_scale_count_removed"] == 35
    assert result["rewritten_fp8_shard_count"] == 1
    assert result["selected_element_count"] == 140
    assert result["selected_byte_count"] == 280
    assert result["content_sha256_before_self_field"] == hybrid.canonical_sha256({
        key: value for key, value in result.items()
        if key != "content_sha256_before_self_field"
    })

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert all(index["weight_map"][name] == hybrid.OVERLAY_NAME for name in units)
    assert not any(name + "_scale_inv" in index["weight_map"] for name in units)
    assert _keys(output / hybrid.OVERLAY_NAME) == set(units)
    assert _keys(output / "model-00001-of-00001.safetensors") == {
        "model.language_model.layers.0.input_layernorm.weight"
    }
    with safe_open(output / hybrid.OVERLAY_NAME, framework="pt") as source:
        for name in units:
            assert torch.equal(source.get_tensor(name), originals[name])

    config = json.loads((output / "config.json").read_text())
    exclusions = set(config["quantization_config"]["modules_to_not_convert"])
    assert "existing.module" in exclusions
    assert "model.language_model.layers.20.linear_attn.in_proj_qkvz" in exclusions
    assert "model.language_model.layers.20.linear_attn.in_proj_ba" in exclusions
    assert "model.language_model.layers.23.self_attn.qkv_proj" in exclusions
    assert "model.language_model.layers.20.mlp.shared_expert.gate_up_proj" in exclusions
    assert config["hybrid_selected_bf16"]["selected_byte_count"] == 280
    assert os.stat(fp8 / "tokenizer.json").st_ino == os.stat(
        output / "tokenizer.json"
    ).st_ino


def test_hybrid_is_exclusive_and_rejects_shape_mismatch(tmp_path):
    bf16, fp8, output, _units, _originals = _fixture(tmp_path / "good")
    hybrid.build_hybrid_checkpoint(bf16, fp8, output)
    with pytest.raises(FileExistsError):
        hybrid.build_hybrid_checkpoint(bf16, fp8, output)

    bad_bf16, bad_fp8, bad_output, *_ = _fixture(tmp_path / "bad", mismatch=True)
    with pytest.raises(RuntimeError, match="tensor shape changed"):
        hybrid.build_hybrid_checkpoint(bad_bf16, bad_fp8, bad_output)
    assert not bad_output.exists()


def test_real_sources_have_exact_expected_hybrid_surface():
    bf16, fp8 = hybrid.DEFAULT_BF16, hybrid.DEFAULT_FP8
    if not bf16.is_dir() or not fp8.is_dir():
        pytest.skip("real Qwen checkpoints are unavailable")
    plan = es_layer_plan.plan_manifest(
        bf16, "front", ["dense"], custom_layers=[20, 21, 22, 23]
    )
    bf16_map = json.loads(
        (bf16 / "model.safetensors.index.json").read_text()
    )["weight_map"]
    fp8_map = json.loads(
        (fp8 / "model.safetensors.index.json").read_text()
    )["weight_map"]
    scales = [name + "_scale_inv" for name in plan["units"]
              if name + "_scale_inv" in fp8_map]
    affected = {fp8_map[name] for name in plan["units"] + scales}
    assert len(plan["units"]) == 35
    assert all(name in bf16_map and name in fp8_map for name in plan["units"])
    assert len(scales) == 25
    assert affected == {
        "layers-20.safetensors", "layers-21.safetensors",
        "layers-22.safetensors", "layers-23.safetensors",
    }
