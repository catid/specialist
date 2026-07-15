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


def test_v17a_fixed_same_document_crn_and_ht_contract_contains_no_raw_content():
    manifest = builder.build_manifest_v17a()
    serialized = json.dumps(manifest, sort_keys=True)
    assert '"question"' not in serialized
    assert '"answer"' not in serialized
    assert '"text"' not in serialized
    for panel in manifest["panels"]:
        for item in panel["items"]:
            assert set(item["sides"]) == {"candidate", "production"}
            assert len(item["shared_document_sha256"]) == 64
            assert item["shared_document_count"] >= 1
            candidate = item["sides"]["candidate"]
            assert candidate["classified_stratum"] == item["stratum"]
            assert candidate["preferred_dominant_stratum"] is True
            for side in item["sides"].values():
                assert side["reuse"] == (
                    "fixed_for_every_direction_and_both_signs"
                )
                assert isinstance(side["row_index"], int)
                assert len(side["row_sha256"]) == 64
                assert side["document_sha256"] == item["shared_document_sha256"]
            assert item["inclusion_probability_per_panel"] * (
                item["horvitz_thompson_unit_weight"]
            ) == pytest.approx(1.0)
    assert manifest["panel_contract"][
        "fixed_side_representative_every_direction_and_sign"
    ] is True
    assert "rotation" not in serialized


def test_v17a_every_direction_and_sign_reuses_exact_ordered_side_batches():
    manifest = builder.build_manifest_v17a()
    frozen = {
        panel["name"]: panel["ordered_side_row_identity_sha256"]
        for panel in manifest["panels"]
    }
    for _direction_index in range(32):
        for _sign in ("plus", "minus"):
            observed = {
                panel["name"]: {
                    side: builder.canonical_sha256([
                        item["sides"][side]["row_sha256"]
                        for item in panel["items"]
                    ])
                    for side in ("candidate", "production")
                }
                for panel in manifest["panels"]
            }
            assert observed == frozen


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
    with pytest.raises(RuntimeError, match="manifest contract|identity or weighting"):
        builder.validate_manifest_v17a(changed)
    changed = copy.deepcopy(manifest)
    changed["joint_frame"]["paired_stratum_counts"]["equipment_material"] = 9
    changed["content_sha256_before_self_field"] = builder.canonical_sha256({
        key: value for key, value in changed.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="manifest contract"):
        builder.validate_manifest_v17a(changed)
