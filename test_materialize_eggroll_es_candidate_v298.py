#!/usr/bin/env python3
"""Offline tests for the eval-free immutable v298 materialization."""

import hashlib
import json

import materialize_eggroll_es_candidate_v298 as materialize_v298


def test_v298_materialized_candidate_has_exact_sealed_identity():
    path = materialize_v298.OUTPUT_PATH_V298
    assert materialize_v298.file_sha256(path) == materialize_v298.V298_SHA256
    assert sum(1 for line in path.open("rb") if line.strip()) == 519
    manifest = json.loads(materialize_v298.MANIFEST_PATH_V298.read_text())
    assert materialize_v298.validate_manifest_v298(manifest) == manifest
    assert manifest["candidate"] == {
        "path": str(path),
        "rows": 519,
        "file_sha256": materialize_v298.V298_SHA256,
    }


def test_v298_train_only_replay_is_byte_identical_at_every_checkpoint():
    candidate, checkpoints = materialize_v298.build_candidate_bytes_v298()
    assert hashlib.sha256(candidate).hexdigest() == materialize_v298.V298_SHA256
    assert len(checkpoints) == 15
    assert [item["version"] for item in checkpoints] == list(range(284, 299))
    assert [
        (item["rows"], item["sha256"]) for item in checkpoints
    ] == [materialize_v298.CHECKPOINTS[version] for version in range(284, 299)]


def test_v298_manifest_seals_reported_components_and_eval_firewall():
    manifest = json.loads(materialize_v298.MANIFEST_PATH_V298.read_text())
    provenance = manifest["provenance"]
    assert provenance["seal_commit"] == materialize_v298.V298_SEAL_COMMIT
    assert provenance["sealed_reported_component_file_sha256"] == dict(
        sorted(materialize_v298.SEAL_REPORTED_COMPONENTS.items())
    )
    assert provenance["checkpoint_chain"][-1] == {
        "version": 298,
        "rows": 519,
        "sha256": materialize_v298.V298_SHA256,
    }
    assert manifest["firewall"]["evaluation_paths_passed_to_merge"] == []
    assert manifest["firewall"]["evaluation_facts"] == "empty_tuple"
    assert manifest["firewall"][
        "heldout_validation_ood_eval_or_benchmark_content_opened"
    ] is False
    assert manifest["firewall"]["runtime_launch_authorized"] is False
    assert manifest["firewall"]["model_update_authorized"] is False
