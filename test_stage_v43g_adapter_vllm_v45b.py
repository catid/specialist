#!/usr/bin/env python3

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import stage_candidate_adapters_vllm_v44a as prior
import stage_v43g_adapter_vllm_v45b as subject


def test_v43g_success_report_snapshot_and_identity_are_bound_v45b():
    seal = subject.source_seal_v45b()
    source = subject.audit_source_v45b()
    assert seal["state_complete"] is True
    assert seal["selection_data_opened"] is False
    assert seal["heldout_or_holdout_opened"] is False
    assert seal["snapshot_actor_count"] == 4
    assert seal["snapshot_master_sha256"] == subject.EXPECTED["master_sha256"]
    assert source["source_metadata"]["master_sha256"] == subject.EXPECTED[
        "master_sha256"
    ]
    assert len(source["records"]) == 70
    assert sum(row["elements"] for row in source["records"]) == 4_528_128


def test_v43g_stage_is_byte_exact_language_model_namespace_v45b(tmp_path):
    output = tmp_path / "v43g"
    manifest = subject.stage_v45b(output)
    source = subject.audit_source_v45b()
    with safe_open(
        output / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        assert handle.metadata() == {"format": "pt"}
        assert len(list(handle.keys())) == 70
        assert all(key.startswith(prior.TARGET_PREFIX_V44A) for key in handle.keys())
    staged = load_file(output / "adapter_model.safetensors", device="cpu")
    for row in manifest["tensor_mapping_records"]:
        assert staged[row["target_key"]].dtype == torch.float32
        assert torch.equal(
            staged[row["target_key"]], source["tensors"][row["source_key"]]
        )
    assert manifest["transformed_identity"][
        "all_tensor_bytes_preserved_exactly"
    ] is True
    assert manifest["shadow_ood_holdout_or_heldout_accessed"] is False
