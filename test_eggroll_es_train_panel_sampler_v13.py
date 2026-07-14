#!/usr/bin/env python3

import copy
import json
from collections import Counter

import pytest

import build_eggroll_es_train_panels_v13 as builder
import eggroll_es_train_panel_sampler_v13 as sampler


MANIFEST_PATH = builder.DEFAULT_OUTPUT


@pytest.fixture(scope="module")
def frozen():
    rows, source_sha = sampler.load_frozen_train()
    manifest = json.loads(MANIFEST_PATH.read_text())
    return rows, source_sha, manifest


def _row(question, answer="answer", document="d", fact="f", **extra):
    return {
        "question": question,
        "answer": answer,
        "document_sha256": document,
        "fact_id": fact,
        **extra,
    }


def test_explicit_strata_prioritize_safety_and_equipment():
    assert sampler.classify_stratum(
        _row("What consent check-in should be used during this tie?")
    ) == "safety_consent"
    assert sampler.classify_stratum(
        _row("What diameter jute is sold for this kit?")
    ) == "equipment_material"
    assert sampler.classify_stratum(
        _row("How is a single column knot dressed?")
    ) == "technique"
    assert sampler.classify_stratum(
        _row("Where can someone find this community resource?")
    ) == "resources_general"


def test_shared_document_and_semantics_collapse_into_conflict_units():
    rows = [
        _row("How is this knot tied?", "with two loops", "doc-a", "f-a"),
        _row("What is the history?", "long", "doc-a", "f-b"),
        _row("How is this knot tied?", "with two loops", "doc-b", "f-c"),
        _row("Where is an event listed?", "online", "doc-c", "f-d"),
    ]
    units, semantic_ids = sampler.build_conflict_units(rows)
    assert semantic_ids[0] == semantic_ids[2]
    assert len(units) == 2
    connected = max(units, key=lambda unit: unit["row_count"])
    assert connected["row_count"] == 3
    assert connected["document_sha256s"] == ["doc-a", "doc-b"]


def test_frozen_manifest_is_deterministic_and_content_addressed(frozen):
    rows, source_sha, manifest = frozen
    rebuilt = sampler.build_manifest(rows, sampler.DEFAULT_SOURCE, source_sha)
    assert rebuilt == manifest
    assert sampler.validate_manifest(manifest, rows)
    assert manifest["source"]["jsonl_sha256"] == sampler.SOURCE_SHA256
    assert manifest["source"]["arrow_sha256"] == sampler.SOURCE_ARROW_SHA256
    assert manifest["sampling_frame"] == {
        **manifest["sampling_frame"],
        "raw_rows": 794,
        "source_documents": 310,
        "semantic_clusters": 794,
        "conflict_units": 310,
        "unit_stratum_counts": {
            "safety_consent": 48,
            "technique": 94,
            "equipment_material": 39,
            "resources_general": 129,
        },
    }


def test_panels_are_stratified_and_globally_conflict_free(frozen):
    _, _, manifest = frozen
    all_units, all_documents, all_semantics, all_rows = [], [], [], []
    assert [panel["role"] for panel in manifest["panels"]] == [
        "optimization", "optimization", "optimization",
        "train_only_screen", "train_only_screen",
    ]
    for panel in manifest["panels"]:
        assert panel["rows"] == 56
        assert Counter(item["stratum"] for item in panel["items"]) == Counter(
            sampler.STRATUM_QUOTAS
        )
        all_units.extend(item["conflict_unit_sha256"] for item in panel["items"])
        all_documents.extend(item["document_sha256"] for item in panel["items"])
        all_semantics.extend(item["semantic_cluster_sha256"] for item in panel["items"])
        all_rows.extend(item["row_sha256"] for item in panel["items"])
    assert len(all_units) == len(set(all_units)) == 280
    assert len(all_documents) == len(set(all_documents)) == 280
    assert len(all_semantics) == len(set(all_semantics)) == 280
    assert len(all_rows) == len(set(all_rows)) == 280


def test_importance_weights_match_exact_stratum_inclusion(frozen):
    _, _, manifest = frozen
    populations = manifest["sampling_frame"]["unit_stratum_counts"]
    for panel in manifest["panels"]:
        for item in panel["items"]:
            quota = sampler.STRATUM_QUOTAS[item["stratum"]]
            expected_probability = quota / populations[item["stratum"]]
            assert item["unit_selection_probability_for_this_panel"] == pytest.approx(
                expected_probability, abs=0.0, rel=1e-15
            )
            assert item["horvitz_thompson_unit_weight"] == pytest.approx(
                1.0 / expected_probability, abs=0.0, rel=1e-15
            )


def test_common_random_numbers_are_identical_for_signs_and_directions(frozen):
    _, _, manifest = frozen
    schedule = sampler.common_random_number_schedule(
        manifest, "optimization_1", ["d0", "d1", "d2"],
    )
    assert len(schedule) == 6
    assert {item["sign"] for item in schedule} == {"plus", "minus"}
    assert len({item["ordered_row_identity_sha256"] for item in schedule}) == 1
    assert {item["panel"] for item in schedule} == {"optimization_1"}


def test_screen_materialization_addresses_only_frozen_train(frozen):
    _, _, manifest = frozen
    materialized = sampler.materialize_panel(manifest, "train_screen_0")
    assert len(materialized["questions"]) == len(materialized["answers"]) == 56
    assert len(set(materialized["fact_ids"])) == 56
    assert materialized["ordered_row_identity_sha256"] == next(
        panel["ordered_row_identity_sha256"]
        for panel in manifest["panels"]
        if panel["name"] == "train_screen_0"
    )


def test_hard_examples_fail_closed_until_independent_train_artifact(frozen):
    _, _, manifest = frozen
    policy = manifest["hard_example_mixture"]
    assert policy["enabled"] is False
    assert policy["configured_fraction"] == 0.0
    assert policy["maximum_fraction"] == 0.25
    assert "train-only" in policy["reason"]
    tampered = copy.deepcopy(manifest)
    tampered["hard_example_mixture"]["enabled"] = True
    tampered.pop("content_sha256_before_self_field")
    tampered["content_sha256_before_self_field"] = sampler.canonical_sha256(tampered)
    with pytest.raises(ValueError, match="must remain disabled"):
        sampler.validate_manifest(tampered)


def test_tampering_and_nontrain_sources_are_rejected(frozen, tmp_path):
    _, _, manifest = frozen
    tampered = copy.deepcopy(manifest)
    tampered["panels"][0]["items"][0]["fact_id"] = "changed"
    with pytest.raises(ValueError, match="content hash"):
        sampler.validate_manifest(tampered)
    forbidden = tmp_path / "heldout.jsonl"
    forbidden.write_text("{}\n")
    with pytest.raises(ValueError, match="train-only"):
        sampler.load_frozen_train(forbidden)


def test_builder_verifies_byte_identical_existing_manifest():
    rebuilt = builder.main(["--verify-existing"])
    assert rebuilt["content_sha256_before_self_field"] == (
        "46cc98b694c98c1ee1c5456b855fb3b1db4534b3df2dcda69fc690a2d8a61bf5"
    )
