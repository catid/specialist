#!/usr/bin/env python3

import build_v48b_train_membership as subject
import run_lora_es_multi_anchor_v43i as v43i


def test_v48b_membership_is_exact_content_free_v43i_projection():
    value = subject.build_membership_v48b()
    assert value["rows"] == 448
    assert value["conflict_units"] == 208
    assert len(value["items"]) == 448
    assert [item["row_index"] for item in value["items"]] == list(range(448))
    assert len({item["row_sha256"] for item in value["items"]}) == 448
    assert len({item["unit_identity_sha256"] for item in value["items"]}) == 208
    assert value["source"]["train_bundle_content_sha256"] == (
        v43i.TRAIN_BUNDLE_SHA256
    )
    assert value["question_answer_evidence_or_text_persisted"] is False
    assert value["nontrain_semantics_opened"] is False
    assert value["runtime_requires_original_split_commitment"] is False


def test_v48b_membership_self_hash_and_multiplicities():
    value = subject.build_membership_v48b()
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    assert value["content_sha256_before_self_field"] == (
        v43i.v40a.canonical_sha256(compact)
    )
    counts = {}
    declared = {}
    for item in value["items"]:
        unit = item["unit_identity_sha256"]
        counts[unit] = counts.get(unit, 0) + 1
        declared.setdefault(unit, item["row_count"])
        assert declared[unit] == item["row_count"]
    assert counts == declared
