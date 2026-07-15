#!/usr/bin/env python3
"""CPU-only contract tests for the V24 hybrid checkpoint post-build audit."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

import audit_qwen36_hybrid_fp8_selected_bf16_v24 as audit
import build_qwen36_hybrid_fp8_selected_bf16_v24 as builder
from test_build_qwen36_hybrid_fp8_selected_bf16_v24 import _fixture


def _built_fixture(tmp_path):
    bf16, fp8, output, units, _originals = _fixture(tmp_path)
    builder.build_hybrid_checkpoint(bf16, fp8, output)
    return bf16, fp8, output, units


def test_audit_proves_exact_overlay_retained_tensors_and_hardlinks(tmp_path):
    bf16, fp8, output, _units = _built_fixture(tmp_path)
    result = audit.audit_hybrid_checkpoint(bf16, fp8, output)
    assert result["schema"] == audit.AUDIT_SCHEMA
    assert result["selected_unit_count"] == 35
    assert result["selected_scale_count_removed"] == 35
    assert result["selected_element_count"] == 140
    assert result["selected_byte_count"] == 280
    assert result["rewritten_fp8_shard_count"] == 1
    assert result["retained_tensor_count_in_rewritten_shards"] == 1
    assert result["unaffected_hardlink_count"] == 1
    assert result["all_selected_tensors_exact_bf16"] is True
    assert result["all_retained_rewritten_shard_tensors_exact_fp8"] is True
    assert result["all_unaffected_files_exact_hardlinks"] is True
    assert result["contains_dataset_or_evaluation_content"] is False
    assert result["content_sha256_before_self_field"] == builder.canonical_sha256({
        key: value for key, value in result.items()
        if key != "content_sha256_before_self_field"
    })


def test_audit_rejects_overlay_tamper(tmp_path):
    bf16, fp8, output, units = _built_fixture(tmp_path)
    overlay = output / builder.OVERLAY_NAME
    tensors = load_file(overlay)
    tensors[units[0]] = tensors[units[0]].clone()
    tensors[units[0]].view(-1)[0] += torch.tensor(1, dtype=torch.bfloat16)
    save_file(tensors, overlay, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="overlay file identity changed"):
        audit.audit_hybrid_checkpoint(bf16, fp8, output)


def test_audit_rejects_retained_tensor_tamper_even_with_consistent_keys(tmp_path):
    bf16, fp8, output, _units = _built_fixture(tmp_path)
    provenance = json.loads((output / builder.PROVENANCE_NAME).read_text())
    shard = output / provenance["affected_fp8_shards"][0]
    tensors = load_file(shard)
    retained = next(iter(tensors))
    tensors[retained] = tensors[retained].clone()
    tensors[retained].view(-1)[0] += torch.tensor(1, dtype=tensors[retained].dtype)
    save_file(tensors, shard, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="retained tensor changed"):
        audit.audit_hybrid_checkpoint(bf16, fp8, output)


def test_audit_rejects_non_hardlinked_frozen_file(tmp_path):
    bf16, fp8, output, _units = _built_fixture(tmp_path)
    tokenizer = output / "tokenizer.json"
    contents = tokenizer.read_bytes()
    tokenizer.unlink()
    tokenizer.write_bytes(contents)
    with pytest.raises(RuntimeError, match="exact hardlink"):
        audit.audit_hybrid_checkpoint(bf16, fp8, output)
