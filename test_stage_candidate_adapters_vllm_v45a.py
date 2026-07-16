#!/usr/bin/env python3

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

import stage_candidate_adapters_vllm_v44a as prior
import stage_candidate_adapters_vllm_v45a as subject


@pytest.mark.parametrize("arm", list(subject.SOURCE_SPECS_V45A))
def test_train_only_source_and_matched_recipe_are_sealed_v45a(arm):
    value = subject.audit_source_v45a(arm)
    assert len(value["records"]) == 70
    assert sum(row["elements"] for row in value["records"]) == 4_528_128
    assert value["seal"]["state_complete"] is True
    assert value["seal"]["selection_data_opened"] is False
    report = json.loads(subject.SOURCE_SPECS_V45A[arm]["report"].read_text())
    assert report["validation_ood_or_holdout_opened"] is False
    assert report["recipe"]["learning_rate"] == subject.SOURCE_SPECS_V45A[
        arm
    ]["learning_rate"]


@pytest.mark.parametrize("arm", list(subject.SOURCE_SPECS_V45A))
def test_stage_preserves_every_fp32_tensor_in_language_model_namespace_v45a(
    tmp_path, arm
):
    output = tmp_path / arm
    manifest = subject.stage_one_v45a(arm, output)
    source = subject.audit_source_v45a(arm)
    with safe_open(
        output / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        assert len(list(handle.keys())) == 70
        assert all(
            key.startswith(prior.TARGET_PREFIX_V44A) for key in handle.keys()
        )
    staged = load_file(output / "adapter_model.safetensors", device="cpu")
    for row in manifest["tensor_mapping_records"]:
        assert staged[row["target_key"]].dtype == torch.float32
        assert torch.equal(
            staged[row["target_key"]], source["tensors"][row["source_key"]]
        )
    assert manifest["transformed_identity"][
        "all_tensor_bytes_preserved_exactly"
    ] is True
    assert manifest["dataset_or_evaluation_accessed"] is False
    assert manifest["shadow_ood_holdout_or_heldout_accessed"] is False
