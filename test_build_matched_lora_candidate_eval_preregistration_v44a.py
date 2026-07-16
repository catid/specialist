#!/usr/bin/env python3

import build_matched_lora_candidate_eval_preregistration_v44a as builder
import run_matched_lora_candidate_eval_v44a as runtime


def test_build_binds_six_arms_without_protected_semantic_access_v44a(monkeypatch):
    protected = {item["path"] for item in runtime.PROTECTED_INPUTS_V44A.values()}
    original = runtime.file_sha256

    def guarded(path):
        assert str(path) not in protected
        return original(path)

    monkeypatch.setattr(runtime, "file_sha256", guarded)
    value = builder.build()
    assert value["arms"] == list(runtime.ARMS)
    assert value["candidate_arms"] == list(runtime.CANDIDATE_ARMS)
    assert value["single_access_inputs"] == runtime.PROTECTED_INPUTS_V44A
    assert value["raw_shadow_or_ood_content_opened_before_preregistration"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["runtime"]["obsolete_full_weight_layer_plan_or_snapshot_path_authorized"] is False
    assert "layer_plan" not in value["implementation_bindings"]
    assert value["content_sha256_before_self_field"] == runtime.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
