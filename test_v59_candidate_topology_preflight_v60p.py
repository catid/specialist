#!/usr/bin/env python3

from __future__ import annotations

import json

import pytest
from safetensors import safe_open

import build_v59_candidate_topology_preflight_v60p as builder
import run_v59_candidate_topology_preflight_v60p as runtime


def test_v60p_stage_and_candidate_are_exactly_bound():
    binding = runtime.stage_binding_v60p()
    assert binding["candidate"] == runtime.EXPECTED["candidate"]
    assert binding["tensor_count"] == 70
    assert binding["elements"] == 4_528_128
    assert binding["all_tensor_bytes_preserved_exactly"] is True
    assert binding["all_nine_train_endpoint_gates_passed"] is True
    with safe_open(
        runtime.stage.SOURCE / "adapter_model.safetensors",
        framework="pt", device="cpu",
    ) as handle:
        assert len(list(handle.keys())) == 70


def test_v60p_preregistration_is_synthetic_only_and_self_hashed():
    value = builder.build_v60p()
    assert value["schema"] == (
        "v59-candidate-topology-preflight-preregistration-v60p"
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


def test_v60p_parent_patch_is_scoped_and_uses_the_fixed_v59_snapshot():
    parent = runtime.parent
    old = parent.ADAPTER_FILE
    with runtime.patched_parent_v60p():
        assert parent.ADAPTER_FILE == (
            runtime.stage.SOURCE / "adapter_model.safetensors"
        )
        assert parent.STAGED_ADAPTER == runtime.stage.OUTPUT
        assert parent.RUN_DIR == runtime.RUN_DIR
        assert parent._lora_request is runtime.lora_request_v60p
    assert parent.ADAPTER_FILE == old


def test_v60p_resolver_installs_callable_surface():
    class Trainer:
        pass

    calls = []
    trainer = runtime.attach_resolver_v60p(
        Trainer(), lambda handles: calls.append(handles) or ["resolved"]
    )
    assert trainer._resolve(["handle"]) == ["resolved"]
    assert calls == [["handle"]]


def test_v60p_dry_run_opens_no_gpu_or_data(tmp_path, monkeypatch, capsys):
    value = builder.build_v60p()
    path = tmp_path / "prereg.json"
    runtime.parent.atomic_json(path, value)
    monkeypatch.setattr(
        runtime.parent, "gpu_preflight",
        lambda: (_ for _ in ()).throw(AssertionError("dry run reached GPU")),
    )
    argv = [
        "--preregistration", str(path),
        "--preregistration-sha256", runtime.file_sha256(path),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]
    assert runtime.main(argv) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dataset_or_evaluation_accessed"] is False


def test_v60p_parser_requires_preregistration_hashes():
    with pytest.raises(SystemExit):
        runtime.parent.parser().parse_args([])
