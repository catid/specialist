#!/usr/bin/env python3
"""Offline tests for the exact commit-derived v341 train candidate seal."""

import copy
import json

import pytest

import seal_eggroll_es_candidate_v341 as seal_v341


def test_v341_candidate_and_manifest_have_exact_commit_bound_identity():
    seal_v341.validate_candidate_snapshot_v341()
    manifest = json.loads(seal_v341.MANIFEST_PATH_V341.read_text())
    assert seal_v341.validate_manifest_v341(manifest) == manifest
    assert manifest["candidate"] == {
        "path": str(seal_v341.OUTPUT_PATH_V341),
        "rows": 528,
        "file_sha256": seal_v341.V341_SHA256,
    }
    assert manifest["provenance"]["source_commit"] == (
        "162e39408f4af0feee694dd4c128e9bb10dac057"
    )


def test_v341_source_inventory_and_firewall_are_exact_commit_bound():
    assert seal_v341.verify_source_commit_v341() == (
        seal_v341.SOURCE_ARTIFACT_SHA256
    )
    manifest = json.loads(seal_v341.MANIFEST_PATH_V341.read_text())
    assert manifest["provenance"]["ongoing_working_tree_curation_used"] is False
    assert manifest["provenance"]["collision_input_file_sha256"] == (
        seal_v341.COLLISION_INPUT_SHA256
    )
    assert manifest["firewall"][
        "heldout_validation_ood_eval_or_benchmark_content_opened"
    ] is False
    assert manifest["firewall"]["replay_stdout_exposed"] is False
    assert manifest["firewall"]["replay_stderr_exposed"] is False


def test_v341_exact_commit_projection_replay_is_byte_identical(tmp_path):
    replay_path = tmp_path / "replay.jsonl"
    result = seal_v341.replay_projection_v341(replay_path)
    assert result == {
        "rows": 528,
        "file_sha256": seal_v341.V341_SHA256,
        "stdout_exposed": False,
        "stderr_exposed": False,
        "jsonl_rows_parsed_by_seal": False,
    }
    assert replay_path.read_bytes() == seal_v341.OUTPUT_PATH_V341.read_bytes()


def test_v341_manifest_rejects_authority_and_source_tampering(monkeypatch):
    manifest = json.loads(seal_v341.MANIFEST_PATH_V341.read_text())
    tampered = copy.deepcopy(manifest)
    tampered["firewall"]["runtime_launch_authorized"] = True
    tampered["content_sha256_before_self_field"] = seal_v341.canonical_sha256({
        key: item for key, item in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="manifest changed"):
        seal_v341.validate_manifest_v341(tampered)

    original = seal_v341._git_blob
    monkeypatch.setattr(
        seal_v341,
        "_git_blob",
        lambda relative: original(relative) + b"tamper",
    )
    with pytest.raises(RuntimeError, match="source-commit artifact identity"):
        seal_v341.verify_source_commit_v341()
