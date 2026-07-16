#!/usr/bin/env python3

import torch
from safetensors import safe_open
from safetensors.torch import load_file

import stage_candidate_adapters_vllm_v44a as prior
import stage_v43l_adapter_vllm_v46e as subject


def test_v43l_result_gate_consensus_snapshot_and_gpu_seals_are_bound_v46e():
    seal = subject.source_seal_v46e()
    source = subject.audit_source_v46e()
    assert seal["state_complete"] is True
    assert seal["accepted_target_norm_ratio"] == 0.03125
    assert seal["gate_check_count"] == 6
    assert seal["all_four_gpus_attributed_positive"] is True
    assert seal["selection_data_opened"] is False
    assert seal["heldout_or_holdout_opened"] is False
    assert seal["snapshot_actor_count"] == 4
    assert seal["snapshot_master_sha256"] == subject.EXPECTED["master_sha256"]
    assert source["source_metadata"]["master_sha256"] == subject.EXPECTED[
        "master_sha256"
    ]
    assert len(source["records"]) == 70
    assert sum(row["elements"] for row in source["records"]) == 4_528_128


def test_v43l_stage_is_byte_exact_language_model_namespace_v46e(tmp_path):
    output = tmp_path / "v43l"
    manifest = subject.stage_v46e(output)
    source = subject.audit_source_v46e()
    with safe_open(
        output / "adapter_model.safetensors", framework="pt", device="cpu"
    ) as handle:
        assert handle.metadata() == {"format": "pt"}
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
    assert manifest["shadow_ood_holdout_or_heldout_accessed"] is False
    assert manifest["gpu_accessed"] is False
