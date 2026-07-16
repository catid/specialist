#!/usr/bin/env python3

import torch
from safetensors.torch import load_file

import stage_candidate_adapters_vllm_v44a as prior
import stage_v42h_adapter_vllm_v45c as subject


def test_v42h_report_and_canonical_source_are_sealed_v45c():
    seal = subject.source_seal_v45c()
    source = subject.audit_source_v45c()
    assert seal["completed_steps"] == 48
    assert seal["learning_rate"] == 6e-5
    assert seal["selection_data_opened"] is False
    assert seal["heldout_or_holdout_opened"] is False
    assert len(source["records"]) == 70
    assert sum(row["elements"] for row in source["records"]) == 4_528_128


def test_v42h_stage_preserves_all_tensor_bytes_v45c(tmp_path):
    output = tmp_path / "v42h"
    manifest = subject.stage_v45c(output)
    source = subject.audit_source_v45c()
    staged = load_file(output / "adapter_model.safetensors", device="cpu")
    assert len(staged) == 70
    assert all(key.startswith(prior.TARGET_PREFIX_V44A) for key in staged)
    for row in manifest["tensor_mapping_records"]:
        assert staged[row["target_key"]].dtype == torch.float32
        assert torch.equal(
            staged[row["target_key"]], source["tensors"][row["source_key"]]
        )
    assert manifest["transformed_identity"][
        "all_tensor_bytes_preserved_exactly"
    ] is True
    assert manifest["shadow_ood_holdout_or_heldout_accessed"] is False
