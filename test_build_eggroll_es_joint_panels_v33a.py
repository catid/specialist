#!/usr/bin/env python3

import json

import pytest

import build_eggroll_es_joint_panels_v33a as frame_v33a


def test_v33a_aggregate_frame_is_exact_deterministic_and_detail_free():
    frozen = json.loads(frame_v33a.OUTPUT_PATH.read_text(encoding="utf-8"))
    built = frame_v33a.build_manifest()
    assert built == frozen
    assert frame_v33a.file_sha256(frame_v33a.OUTPUT_PATH) == (
        "8c8ab38e03949e01982701b58e0f273c23305e67b114f7c607e7e4d83d10666a"
    )
    assert frozen["content_sha256_before_self_field"] == (
        "1690210499f6ac2739b6d63c02c8b3101dc36027438042c9534d2ad9de5c9d68"
    )
    frame_v33a.validate_manifest(frozen)
    serialized = json.dumps(frozen, sort_keys=True)
    for forbidden in (
        '"items"', '"unit_id"', '"row_index"', '"row_sha256"',
        '"document_sha256"', '"pairing_anchor"', '"question"', '"answer"',
    ):
        assert forbidden not in serialized


def test_v33a_transient_frame_has_five_disjoint_39_unit_panels_and_reserve():
    runtime = frame_v33a.build_runtime_manifest_v33a()
    joint = runtime["joint_frame"]
    assert joint["joint_component_count"] == 326
    assert joint["paired_unit_count"] == 205
    assert joint["paired_stratum_counts"] == frame_v33a.EXPECTED_PAIRED_STRATA
    assert joint["selected_paired_unit_count"] == 195
    assert joint["reserve_paired_unit_count"] == 10
    assert joint["paired_assignment_root_sha256"] == (
        frame_v33a.EXPECTED_PAIRED_ASSIGNMENT_ROOT_SHA256
    )
    assert joint["selected_unit_id_root_sha256"] == (
        frame_v33a.EXPECTED_SELECTED_UNIT_ROOT_SHA256
    )
    items = [item for panel in runtime["panels"] for item in panel["items"]]
    assert len(items) == len({item["unit_id"] for item in items}) == 195
    assert all(len(panel["items"]) == 39 for panel in runtime["panels"])
    assert runtime["panel_contract"]["panel_seed"] == 20261006
    assert runtime["panel_contract"]["representative_seed"] == 20261007


def test_v33a_transient_cross_side_anchors_and_weights_are_exact():
    runtime = frame_v33a.build_runtime_manifest_v33a()
    items = [item for panel in runtime["panels"] for item in panel["items"]]
    fallback = [
        item for item in items
        if item["pairing_anchor"] == "joint_component_cross_side_link"
    ]
    shared = [item for item in items if item["pairing_anchor"] == "shared_document"]
    assert len(fallback) == 2
    assert len(shared) == 193
    assert all(item["shared_document_sha256"] is None for item in fallback)
    assert all(
        item["inclusion_probability_per_panel"]
        * item["horvitz_thompson_unit_weight"] == pytest.approx(1.0)
        for item in items
    )


def test_v33a_binds_exact_v364_and_production_commits():
    candidate, production = frame_v33a.load_bound_rows()
    assert len(candidate) == 531
    assert len(production) == 784
    aggregate = frame_v33a.build_manifest()
    assert aggregate["inputs"]["candidate"]["freeze_commit"] == (
        "de0d5518f5cffe2ee71d8fc6884e506f3c1f3272"
    )
    assert aggregate["inputs"]["production"]["source_commit"] == (
        "a21de35748054c3ae8737a767606234952f9561e"
    )


def test_v33a_rejects_changed_candidate_and_output_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_v33a, "CANDIDATE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="frozen data"):
        frame_v33a.load_bound_rows()
    monkeypatch.undo()
    output = tmp_path / "paired.json"
    monkeypatch.setattr(frame_v33a, "OUTPUT_PATH", output.resolve())
    frame_v33a.exclusive_write(output, {"bound": True})
    with pytest.raises(ValueError, match="already exists"):
        frame_v33a.exclusive_write(output, {"bound": True})
