#!/usr/bin/env python3

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def test_v364_frozen_candidate_manifest_and_bytes_are_exact():
    manifest = json.loads(MANIFEST.read_text())
    candidate = ROOT / manifest["candidate"]["path"]
    report_path = ROOT / manifest["projection_report"]["path"]
    report = json.loads(report_path.read_text())
    assert manifest["content_sha256_before_self_field"] == canonical_sha256({
        key: value for key, value in manifest.items()
        if key != "content_sha256_before_self_field"
    })
    assert file_sha256(candidate) == manifest["candidate"]["file_sha256"]
    assert sum(1 for _ in candidate.open("rb")) == manifest["candidate"]["rows"]
    assert file_sha256(report_path) == manifest["projection_report"]["file_sha256"]
    assert report["schema"] == manifest["projection_report"]["schema"]
    assert report["output_sha256"] == manifest["candidate"]["file_sha256"]
    assert report["counts"]["output"] == manifest["candidate"]["rows"]


def test_v364_source_artifacts_and_curator_report_are_bound():
    manifest = json.loads(MANIFEST.read_text())
    artifacts = manifest["provenance"]["source_artifact_file_sha256"]
    assert {
        path: file_sha256(ROOT / path) for path in sorted(artifacts)
    } == artifacts
    assert canonical_sha256(artifacts) == manifest["provenance"][
        "source_artifact_inventory_sha256"
    ]
    curator = manifest["provenance"]["curator_report"]
    report = json.loads((ROOT / curator["path"]).read_text())
    assert file_sha256(ROOT / curator["path"]) == curator["file_sha256"]
    projection = report["isolated_build_projection"]
    assert projection["output_rows"] == curator["candidate_rows"]
    assert projection["output_sha256"] == curator["candidate_sha256"]
    assert projection["repeat_dataset_byte_identical"] is True
    assert projection["repeat_projection_report_byte_identical"] is True


def test_v364_freeze_has_no_runtime_or_nontrain_authority():
    manifest = json.loads(MANIFEST.read_text())
    firewall = manifest["firewall"]
    assert firewall["scope"] == "train_only"
    assert firewall["candidate_row_content_in_manifest"] is False
    assert firewall["runtime_launch_authorized"] is False
    assert firewall["evaluation_authorized"] is False
    assert firewall["model_update_authorized"] is False
    assert firewall["dataset_promotion_authorized"] is False
