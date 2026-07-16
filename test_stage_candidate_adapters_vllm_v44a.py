#!/usr/bin/env python3

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import stage_candidate_adapters_vllm_v44a as subject


def test_transform_is_exactly_one_way_v44a():
    source = "base_model.model.model.layers.23.self_attn.q_proj.lora_A.weight"
    target = (
        "base_model.model.model.language_model.layers.23."
        "self_attn.q_proj.lora_A.weight"
    )
    assert subject.transform_key_v44a(source) == target
    with pytest.raises(ValueError, match="outside the canonical namespace"):
        subject.transform_key_v44a("model.layers.23.self_attn.q_proj.weight")
    with pytest.raises(ValueError, match="already transformed"):
        subject.transform_key_v44a(target)


@pytest.mark.parametrize("arm", list(subject.CANDIDATE_SPECS_V44A))
def test_source_audit_is_exact_v44a(arm):
    source = subject.audit_source_v44a(arm)
    assert len(source["records"]) == 70
    assert sum(item["elements"] for item in source["records"]) == 4_528_128
    assert all(item["dtype"] == "torch.float32" for item in source["records"])
    assert all(
        item["target_key"] == subject.transform_key_v44a(item["source_key"])
        for item in source["records"]
    )
    assert source["seal"]["selection_data_opened"] is False


@pytest.mark.parametrize("arm", list(subject.CANDIDATE_SPECS_V44A))
def test_stage_is_byte_exact_and_immutable_v44a(tmp_path, arm):
    output = tmp_path / arm
    manifest = subject.stage_one_v44a(arm, output)
    source = subject.audit_source_v44a(arm)
    with safe_open(
        output / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        keys = list(handle.keys())
        assert handle.metadata() == {"format": "pt"}
    assert len(keys) == 70
    assert all(key.startswith(subject.TARGET_PREFIX_V44A) for key in keys)
    staged = load_file(output / "adapter_model.safetensors", device="cpu")
    for record in manifest["tensor_mapping_records"]:
        assert torch.equal(
            staged[record["target_key"]],
            source["tensors"][record["source_key"]],
        )
        assert record["tensor_bytes_preserved_exactly"] is True
    assert (output / "adapter_config.json").read_bytes() == source["config_bytes"]
    raw = json.loads((output / "stage_manifest_v44a.json").read_text())
    content = raw.pop("content_sha256_before_self_field")
    assert content == subject.canonical_sha256_v44a(raw)
    assert raw["dataset_or_evaluation_accessed"] is False
    assert raw["shadow_ood_holdout_or_heldout_accessed"] is False
    assert raw["gpu_accessed"] is False
    with pytest.raises(FileExistsError):
        subject.stage_one_v44a(arm, output)
