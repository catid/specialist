from __future__ import annotations

import json

from safetensors import safe_open

import build_v55b_candidate_topology_preflight_v56p as builder
import run_v55b_candidate_topology_preflight_v56p as runtime


def test_v56p_stage_and_candidate_are_exactly_bound():
    binding = runtime.stage_binding_v56p()
    assert binding["candidate"] == runtime.EXPECTED["candidate"]
    assert binding["tensor_count"] == 70
    assert binding["elements"] == 4_528_128
    assert binding["all_tensor_bytes_preserved_exactly"] is True
    assert binding["all_nine_train_endpoint_gates_passed"] is True
    with safe_open(
        runtime.stage.SOURCE / "adapter_model.safetensors",
        framework="pt",
        device="cpu",
    ) as handle:
        assert len(list(handle.keys())) == 70


def test_v56p_preregistration_is_synthetic_only_and_self_hashed():
    value = builder.build_v56p()
    assert value["schema"] == (
        "v55b-candidate-topology-preflight-preregistration-v56p"
    )
    assert value["dataset_or_evaluation_access_authorized"] is False
    assert value["synthetic_prompt_only"] is True
    assert value["quality_claim_authorized"] is False
    assert value["terminal_holdout_access_authorized"] is False
    assert value["runtime"]["physical_gpu_ids"] == [0, 1, 2, 3]
    assert value["runtime"]["engine_count"] == 4
    content = value.pop("content_sha256_before_self_field")
    assert content == runtime.canonical_sha256(value)
    paths = json.dumps(value["implementation_bindings"]).lower()
    assert not any(
        term in paths
        for term in ("heldout", "holdout", "shadow", "ood", "train.jsonl")
    )


def test_v56p_parent_patch_is_scoped_and_points_at_fixed_snapshot():
    parent = runtime.parent
    old = parent.ADAPTER_FILE
    with runtime._patched_parent_v56p():
        assert parent.ADAPTER_FILE == (
            runtime.stage.SOURCE / "adapter_model.safetensors"
        )
        assert parent.STAGED_ADAPTER == runtime.stage.OUTPUT
        assert parent.RUN_DIR == runtime.RUN_DIR
        assert parent._lora_request is runtime.lora_request_v56p
    assert parent.ADAPTER_FILE == old
