#!/usr/bin/env python3
"""CPU-only exact-source audit tests for the V26 hybrid checkpoint."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

import audit_qwen36_fp8_routed_experts_bf16_backbone_v26 as audit
import build_qwen36_fp8_routed_experts_bf16_backbone_v26 as builder
from test_build_qwen36_fp8_routed_experts_bf16_backbone_v26 import _fixture


def _built_fixture(tmp_path):
    bf16, fp8, output, _bf16_tensors, _fp8_tensors = _fixture(tmp_path)
    builder.build_hybrid_checkpoint_v26(bf16, fp8, output)
    return bf16, fp8, output


def test_v26_audit_proves_exact_sources_representation_and_artifact_digests(tmp_path):
    bf16, fp8, output = _built_fixture(tmp_path)
    result = audit.audit_hybrid_checkpoint_v26(bf16, fp8, output)
    assert result["schema"] == audit.AUDIT_SCHEMA
    assert result["target_key_count"] == 14
    assert result["bf16_backbone_tensor_count"] == 2
    assert result["fp8_routed_weight_count"] == 6
    assert result["fp8_routed_scale_count"] == 6
    assert result["removed_non_routed_scale_count"] == 1
    assert result["all_backbone_tensors_exact_bf16"] is True
    assert result["all_routed_expert_tensors_and_scales_exact_fp8"] is True
    assert result["all_output_shard_file_digests_exact"] is True
    assert result["all_source_shard_file_digests_exact"] is True
    assert result["all_auxiliary_files_exact_hardlinks"] is True
    assert result["contains_dataset_or_evaluation_content"] is False
    assert result["content_sha256_before_self_field"] == builder.canonical_sha256({
        key: value
        for key, value in result.items()
        if key != "content_sha256_before_self_field"
    })
    serialized = json.dumps(result, sort_keys=True)
    for forbidden in ("tensor_names", "tensor_values", "rows", "prompts"):
        assert forbidden not in serialized


def test_v26_audit_rejects_bf16_backbone_tamper(tmp_path):
    bf16, fp8, output = _built_fixture(tmp_path)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    backbone_name = next(
        name for name in index["weight_map"] if ".mlp.experts." not in name
    )
    shard = output / index["weight_map"][backbone_name]
    tensors = load_file(shard)
    tensors[backbone_name] = tensors[backbone_name].clone()
    tensors[backbone_name].view(-1)[0] += torch.tensor(1, dtype=torch.bfloat16)
    save_file(tensors, shard, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="output shard file identity changed"):
        audit.audit_hybrid_checkpoint_v26(bf16, fp8, output)


def test_v26_audit_rejects_fp8_expert_tamper(tmp_path):
    bf16, fp8, output = _built_fixture(tmp_path)
    index = json.loads((output / "model.safetensors.index.json").read_text())
    expert_name = next(
        name
        for name in index["weight_map"]
        if ".mlp.experts." in name and not name.endswith("_scale_inv")
    )
    shard = output / index["weight_map"][expert_name]
    tensors = load_file(shard)
    tensors[expert_name] = torch.zeros_like(tensors[expert_name])
    save_file(tensors, shard, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="output shard file identity changed"):
        audit.audit_hybrid_checkpoint_v26(bf16, fp8, output)


def test_v26_audit_rejects_non_routed_quantization_declaration(tmp_path):
    bf16, fp8, output = _built_fixture(tmp_path)
    config_path = output / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"]["modules_to_not_convert"].remove(
        "model.language_model.layers.0.self_attn.q_proj"
    )
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hybrid config identity changed"):
        audit.audit_hybrid_checkpoint_v26(bf16, fp8, output)
