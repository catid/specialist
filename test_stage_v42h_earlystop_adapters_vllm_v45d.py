#!/usr/bin/env python3

import torch
from safetensors.torch import load_file

import stage_candidate_adapters_vllm_v44a as canonical
import stage_v42h_earlystop_adapters_vllm_v45d as subject


def test_v42h_earlystop_sources_bind_step_epoch_and_report_v45d():
    for arm, step, epoch in (
        ("sft_v42h_step16", 16, 1.0),
        ("sft_v42h_step32", 32, 2.0),
    ):
        result = subject.audit_source_v45d(arm)
        state = result["trainer_state_binding_v45d"]
        assert state["global_step"] == step
        assert state["epoch"] == epoch
        assert state["max_steps"] == 48
        assert len(result["records"]) == 70
        assert sum(row["elements"] for row in result["records"]) == 4_528_128


def test_v42h_earlystop_stage_is_byte_exact_v45d(tmp_path):
    for arm in subject.SOURCE_SPECS_V45D:
        output = tmp_path / arm
        manifest = subject.stage_one_v45d(arm, output)
        source = subject.audit_source_v45d(arm)
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
