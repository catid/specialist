#!/usr/bin/env python3

import torch
from safetensors.torch import load_file

import stage_candidate_adapters_vllm_v44a as canonical
import stage_v47c_adapter_vllm_v47d as subject


def test_v47c_source_is_sealed_lineage_stable_train_only_v47d():
    seal = subject.source_seal_v47d()
    result = subject.audit_source_v47d()
    assert seal["shadow_ood_holdout_or_heldout_opened"] is False
    assert seal["lineage_stable_fold_assignment"] is True
    assert seal["all_four_training_gpus_attributed_positive"] is True
    assert len(result["records"]) == 70
    assert sum(row["elements"] for row in result["records"]) == 4_528_128


def test_v47c_stage_is_byte_exact_and_cpu_only_v47d(tmp_path):
    output = tmp_path / "v47c"
    manifest = subject.stage_v47d(output)
    source = subject.audit_source_v47d()
    staged = load_file(output / "adapter_model.safetensors", device="cpu")
    assert len(staged) == 70
    assert all(key.startswith(canonical.TARGET_PREFIX_V44A) for key in staged)
    for row in manifest["tensor_mapping_records"]:
        assert torch.equal(
            staged[row["target_key"]], source["tensors"][row["source_key"]]
        )
    assert manifest["transformed_identity"]["all_tensor_bytes_preserved_exactly"]
    assert manifest["dataset_or_evaluation_accessed"] is False
    assert manifest["shadow_ood_holdout_or_heldout_accessed"] is False
    assert manifest["gpu_accessed"] is False
