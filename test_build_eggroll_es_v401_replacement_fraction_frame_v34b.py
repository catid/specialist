import copy
import hashlib
import json
from pathlib import Path

import pytest

import build_eggroll_es_v401_replacement_fraction_frame_v34b as frame


def test_bound_v401_and_production_committed_identities():
    rows = frame.load_bound_rows()
    assert [len(side) for side in rows] == [531, 784]
    assert frame.CANDIDATE_FREEZE_COMMIT == "59dfe718a914be8b37e05ff9daa822ab467d18a4"
    assert frame.CANDIDATE_SHA256 == hashlib.sha256(frame.CANDIDATE_PATH.read_bytes()).hexdigest()
    assert frame.PRODUCTION_SHA256 == hashlib.sha256(frame.PRODUCTION_PATH.read_bytes()).hexdigest()


def test_runtime_frame_is_exact_hardened_joint_design():
    value = frame.build_runtime_manifest_v34b()
    assert frame.validate_runtime_manifest_v34b(value) is value
    assert value["content_sha256_before_self_field"] == frame.EXPECTED_RUNTIME_FRAME_CONTENT_SHA256
    assert value["joint_frame"]["paired_stratum_counts"] == {
        "safety_consent": 62,
        "technique": 54,
        "equipment_material": 13,
        "resources_general": 76,
    }
    items = [item for panel in value["panels"] for item in panel["items"]]
    assert len(items) == len({item["unit_id"] for item in items}) == 195
    assert sum(item["pairing_anchor"] == "shared_document" for item in items) == 193


def test_persistent_frame_is_content_free_and_exact():
    value = frame.build_manifest()
    assert frame.validate_manifest(value) is value
    assert value["content_sha256_before_self_field"] == frame.EXPECTED_AGGREGATE_CONTENT_SHA256
    encoded = json.dumps(value).lower()
    for forbidden in ('"question"', '"answer"', '"row_index"', '"unit_id"', '"document_sha256"'):
        assert forbidden not in encoded
    assert value["separation"][
        "contains_per_unit_ids_hashes_indices_documents_strata_weights_or_anchors"
    ] is False


def test_mutations_fail_closed():
    runtime = frame.build_runtime_manifest_v34b()
    changed = copy.deepcopy(runtime)
    changed["joint_frame"]["paired_unit_count"] -= 1
    changed["content_sha256_before_self_field"] = frame.canonical_sha256(frame._without_self(changed))
    with pytest.raises(RuntimeError):
        frame.validate_runtime_manifest_v34b(changed)
    aggregate = frame.build_manifest()
    changed = copy.deepcopy(aggregate)
    changed["status"] = "changed"
    changed["content_sha256_before_self_field"] = frame.canonical_sha256(frame._without_self(changed))
    with pytest.raises(RuntimeError):
        frame.validate_manifest(changed)


def test_exclusive_write_refuses_existing_and_noncanonical(tmp_path):
    value = frame.build_manifest()
    with pytest.raises(ValueError):
        frame.exclusive_write(tmp_path / "wrong.json", value)
    if frame.OUTPUT_PATH.exists():
        with pytest.raises(ValueError):
            frame.exclusive_write(frame.OUTPUT_PATH, value)
