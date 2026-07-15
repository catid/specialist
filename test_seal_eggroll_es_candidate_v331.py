#!/usr/bin/env python3
"""Offline tests for the exact commit-derived v331 train candidate seal."""

import copy
import json

import pytest

import seal_eggroll_es_candidate_v331 as seal_v331


def test_v331_candidate_and_manifest_have_exact_commit_bound_identity():
    seal_v331.validate_candidate_snapshot_v331()
    manifest = json.loads(seal_v331.MANIFEST_PATH_V331.read_text())
    assert seal_v331.validate_manifest_v331(manifest) == manifest
    assert manifest["candidate"] == {
        "path": str(seal_v331.OUTPUT_PATH_V331),
        "rows": 527,
        "file_sha256": seal_v331.V331_SHA256,
    }
    assert manifest["provenance"]["source_commit"] == (
        "9d31e3407dd96f80011bdb2202e8aa9689b1d193"
    )


def test_v331_source_inventory_comes_only_from_exact_git_commit():
    assert seal_v331.verify_source_commit_v331() == (
        seal_v331.SOURCE_ARTIFACT_SHA256
    )
    manifest = json.loads(seal_v331.MANIFEST_PATH_V331.read_text())
    assert manifest["provenance"]["ongoing_working_tree_curation_used"] is False
    assert manifest["firewall"][
        "v21a_builder_opened_heldout_validation_ood_eval_or_benchmark_content"
    ] is False


def test_v331_manifest_rejects_authority_and_source_tampering(monkeypatch):
    manifest = json.loads(seal_v331.MANIFEST_PATH_V331.read_text())
    tampered = copy.deepcopy(manifest)
    tampered["firewall"]["runtime_launch_authorized"] = True
    tampered["content_sha256_before_self_field"] = seal_v331.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="manifest changed"):
        seal_v331.validate_manifest_v331(tampered)

    original = seal_v331._git_blob
    monkeypatch.setattr(
        seal_v331,
        "_git_blob",
        lambda relative: original(relative) + b"tamper",
    )
    with pytest.raises(RuntimeError, match="source-commit artifact identity"):
        seal_v331.verify_source_commit_v331()
