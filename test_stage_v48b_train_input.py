#!/usr/bin/env python3

import stage_v48b_train_input as subject
import run_lora_es_multi_anchor_v43i as v43i


def test_v48b_staged_train_path_is_byte_exact_and_runtime_safe(tmp_path, monkeypatch):
    output = tmp_path / "train_only.jsonl"
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(subject, "OUTPUT", output)
    monkeypatch.setattr(subject, "MANIFEST", manifest)
    value = subject.stage_v48b()
    assert output.read_bytes() == v43i.DATASET.read_bytes()
    assert v43i.v40a.file_sha256(output) == v43i.DATASET_SHA256
    assert value["byte_exact"] is True
    assert value["semantic_transform_performed"] is False
    assert value["nontrain_or_protected_input_opened"] is False
    assert all(token not in str(output).casefold() for token in (
        "shadow", "ood", "holdout", "heldout", "benchmark",
    ))
