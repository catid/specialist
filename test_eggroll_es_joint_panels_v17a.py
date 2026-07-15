#!/usr/bin/env python3
"""Offline aggregate-only tests for the paired V17A joint frame."""

import copy
import json

import pytest

import build_eggroll_es_joint_panels_v17a as builder


def test_v17a_joint_frame_exact_capacity_and_geometry():
    manifest = builder.build_manifest_v17a()
    assert manifest["joint_frame"] == {
        **{
            key: manifest["joint_frame"][key]
            for key in (
                "paired_assignment_root_sha256", "selected_unit_id_root_sha256",
                "cross_side_grouping", "stratum_assignment",
            )
        },
        "joint_component_count": 276,
        "candidate_only_unit_count": 4,
        "production_only_unit_count": 77,
        "paired_unit_count": 195,
        "paired_stratum_counts": {
            "safety_consent": 70, "technique": 41,
            "equipment_material": 13, "resources_general": 71,
        },
        "selected_paired_unit_count": 190,
        "reserve_paired_unit_count": 5,
    }
    assert [panel["stratum_counts"] for panel in manifest["panels"]] == [
        {"safety_consent": 14, "technique": 8,
         "equipment_material": 2, "resources_general": 14}
    ] * 5
    units = [
        item["unit_id"] for panel in manifest["panels"]
        for item in panel["items"]
    ]
    assert len(units) == len(set(units)) == 190


def test_v17a_manifest_is_deterministic_and_matches_committed_output_if_present():
    first = builder.build_manifest_v17a()
    second = builder.build_manifest_v17a()
    assert first == second
    if builder.OUTPUT_PATH_V17A.exists():
        assert json.loads(builder.OUTPUT_PATH_V17A.read_text()) == first


def test_v17a_paired_rotation_and_ht_contract_contains_no_raw_content():
    manifest = builder.build_manifest_v17a()
    serialized = json.dumps(manifest, sort_keys=True)
    assert '"question"' not in serialized
    assert '"answer"' not in serialized
    assert '"text"' not in serialized
    for panel in manifest["panels"]:
        for item in panel["items"]:
            assert set(item["sides"]) == {"candidate", "production"}
            for side in item["sides"].values():
                assert side["rotation"] == (
                    "direction_index_mod_side_row_count; "
                    "plus_and_minus_share_row"
                )
                assert side["row_count"] > 0
            assert item["inclusion_probability_per_panel"] * (
                item["horvitz_thompson_unit_weight"]
            ) == pytest.approx(1.0)


def test_v17a_validator_rejects_rehashable_overlap_and_capacity_tampering():
    manifest = builder.build_manifest_v17a()
    changed = copy.deepcopy(manifest)
    changed["panels"][1]["items"][0] = copy.deepcopy(
        changed["panels"][0]["items"][0]
    )
    changed["content_sha256_before_self_field"] = builder.canonical_sha256({
        key: value for key, value in changed.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="identity or weighting"):
        builder.validate_manifest_v17a(changed)
    changed = copy.deepcopy(manifest)
    changed["joint_frame"]["paired_stratum_counts"]["equipment_material"] = 9
    changed["content_sha256_before_self_field"] = builder.canonical_sha256({
        key: value for key, value in changed.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="manifest contract"):
        builder.validate_manifest_v17a(changed)
