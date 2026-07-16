#!/usr/bin/env python3

import torch
from safetensors.torch import load_file

import stage_candidate_adapters_vllm_v44a as canonical
import stage_v42k_adapter_vllm_v45e as subject


def test_v42k_source_is_sealed_matched_train_only_v45e():
    result = subject.audit_source_v45e()
    assert len(result["records"]) == 70
    assert sum(row["elements"] for row in result["records"]) == 4_528_128


def test_v42k_stage_is_byte_exact_v45e(tmp_path):
    output = tmp_path / "v42k"
    manifest = subject.stage_v45e(output)
    source = subject.audit_source_v45e()
    staged = load_file(output / "adapter_model.safetensors", device="cpu")
    assert len(staged) == 70
    assert all(key.startswith(canonical.TARGET_PREFIX_V44A) for key in staged)
    for row in manifest["tensor_mapping_records"]:
        assert torch.equal(
            staged[row["target_key"]], source["tensors"][row["source_key"]]
        )
    assert manifest["transformed_identity"][
        "all_tensor_bytes_preserved_exactly"
    ] is True
    assert manifest["shadow_ood_holdout_or_heldout_accessed"] is False
