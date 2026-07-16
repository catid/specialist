#!/usr/bin/env python3

import torch
from safetensors.torch import load_file

import stage_candidate_adapters_vllm_v44a as canonical
import stage_v49b_adapter_vllm_v49c as subject


def test_v49b_source_is_sealed_complete_train_only_v49c():
    seal = subject.source_seal_v49c()
    result = subject.audit_source_v49c()
    assert seal["state_complete"] is True
    assert seal["shadow_ood_holdout_or_heldout_opened"] is False
    assert seal["only_per_row_weights_changed_from_v47c"] is True
    assert seal["exact_v49a_weight_identity"] == subject.EXPECTED_WEIGHT_IDENTITY
    assert seal["all_four_training_gpus_attributed_positive"] is True
    assert len(result["records"]) == 70
    assert sum(row["elements"] for row in result["records"]) == 4_528_128


def test_v49b_stage_is_byte_exact_and_cpu_only_v49c(tmp_path):
    output = tmp_path / "v49b"
    manifest = subject.stage_v49c(output)
    source = subject.audit_source_v49c()
    staged = load_file(output / "adapter_model.safetensors", device="cpu")
    assert len(staged) == 70
    assert all(key.startswith(canonical.TARGET_PREFIX_V44A) for key in staged)
    for row in manifest["tensor_mapping_records"]:
        assert torch.equal(staged[row["target_key"]], source["tensors"][row["source_key"]])
    assert manifest["transformed_identity"]["all_tensor_bytes_preserved_exactly"]
    assert manifest["dataset_or_evaluation_accessed"] is False
    assert manifest["shadow_ood_holdout_or_heldout_accessed"] is False
    assert manifest["gpu_accessed"] is False
