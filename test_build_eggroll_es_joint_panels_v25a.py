#!/usr/bin/env python3

import json

import pytest

import build_eggroll_es_joint_panels_v25a as frame_v25a


def test_v25a_paired_frame_is_exact_and_deterministic():
    frozen = json.loads(frame_v25a.OUTPUT_PATH.read_text())
    built = frame_v25a.build_manifest()
    assert built == frozen
    assert frame_v25a.file_sha256(frame_v25a.OUTPUT_PATH) == (
        "5b7e8f5d24b00e9e6d7a3490e46a134fe95c3beda5bffa9e06b3dc02bcc7f79e"
    )
    assert frozen["content_sha256_before_self_field"] == (
        "90b0ecf8d483959076ed87dfd46703e3894ab97cd59f9a1d6fff9bdc79e73b0b"
    )
    frame_v25a.validate_manifest(frozen)


def test_v25a_has_five_disjoint_39_unit_panels_and_ten_reserves():
    manifest = frame_v25a.build_manifest()
    joint = manifest["joint_frame"]
    assert joint["joint_component_count"] == 326
    assert joint["paired_unit_count"] == 205
    assert joint["paired_stratum_counts"] == {
        "safety_consent": 62,
        "technique": 54,
        "equipment_material": 13,
        "resources_general": 76,
    }
    assert joint["selected_paired_unit_count"] == 195
    assert joint["reserve_paired_unit_count"] == 10
    items = [item for panel in manifest["panels"] for item in panel["items"]]
    assert len(items) == len({item["unit_id"] for item in items}) == 195
    assert all(len(panel["items"]) == 39 for panel in manifest["panels"])
    assert all(panel["stratum_counts"] == frame_v25a.STRATUM_QUOTAS for panel in manifest["panels"])


def test_v25a_fixes_cross_side_components_without_shared_document_ids():
    items = [
        item
        for panel in frame_v25a.build_manifest()["panels"]
        for item in panel["items"]
    ]
    fallback = [
        item for item in items
        if item["pairing_anchor"] == "joint_component_cross_side_link"
    ]
    shared = [item for item in items if item["pairing_anchor"] == "shared_document"]
    assert len(fallback) == 2
    assert len(shared) == 193
    assert all(item["shared_document_sha256"] is None for item in fallback)
    assert all(item["common_document_count"] == 0 for item in fallback)
    assert all(
        item["sides"]["candidate"]["classified_stratum"] == item["stratum"]
        for item in fallback
    )
    assert all(
        side["document_sha256"] == item["shared_document_sha256"]
        for item in shared for side in item["sides"].values()
    )


def test_v25a_frame_persists_identities_and_weights_not_row_content():
    manifest = frame_v25a.build_manifest()
    assert manifest["separation"]["contains_row_prompt_or_answer_content"] is False
    assert manifest["separation"]["contains_evaluation_content"] is False
    items = [item for panel in manifest["panels"] for item in panel["items"]]
    allowed_side_keys = {
        "row_index", "row_sha256", "document_sha256", "classified_stratum",
        "preferred_dominant_stratum", "reuse",
    }
    assert all(
        set(side) == allowed_side_keys
        for item in items for side in item["sides"].values()
    )
    assert all(
        item["inclusion_probability_per_panel"]
        * item["horvitz_thompson_unit_weight"] == pytest.approx(1.0)
        for item in items
    )


def test_v25a_rejects_changed_candidate_and_output_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(frame_v25a, "CANDIDATE_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="frozen data"):
        frame_v25a.load_bound_rows()
    monkeypatch.undo()
    output = tmp_path / "paired.json"
    monkeypatch.setattr(frame_v25a, "OUTPUT_PATH", output.resolve())
    frame_v25a.exclusive_write(output, {"bound": True})
    with pytest.raises(ValueError, match="already exists"):
        frame_v25a.exclusive_write(output, {"bound": True})
