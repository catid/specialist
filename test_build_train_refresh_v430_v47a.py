#!/usr/bin/env python3

import json
from pathlib import Path

import build_train_refresh_v430_v47a as subject


def test_holdout_blind_replay_and_fold_contract(monkeypatch):
    def forbidden_eval_facts(*args, **kwargs):
        raise AssertionError("V47A tried to load protected evaluation facts")

    monkeypatch.setattr(subject.curated, "eval_facts", forbidden_eval_facts)
    value, payloads = subject.construct()
    assert value["projection"]["sha256"] == subject.PROJECTION_SHA256
    assert value["projection"]["repeat_replay_byte_identical"] is True
    assert payloads[subject.PROJECTION].count(b"\n") == 531
    assert value["fold"]["train"]["rows"] == 459
    assert value["fold"]["train"]["conflict_units"] == 208
    assert value["fold"]["shadow_dev"]["rows"] == 72
    assert value["fold"]["shadow_dev"]["conflict_units"] == 51
    assert not any(value["fold"]["train_dev_edge_identity_intersections"].values())
    assert value["access_firewall"] == {
        "eval_ood_holdout_or_benchmark_opened": False,
        "eval_facts_supplied_to_replay": 0,
        "shadow_dev_opened_after_split_construction": False,
        "shadow_dev_may_be_opened_during_v47a_training": False,
        "v430_replay_used_only_preaccepted_train_curation": True,
    }


def test_manifest_is_self_hashed_and_has_no_v46d_binding():
    value, _ = subject.construct()
    content = value.pop("content_sha256_before_self_field")
    assert content == subject.engine.canonical_sha256(value)
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    assert "v46d" not in source
    serialized = json.dumps(value).lower()
    assert "holdout_eval" not in serialized
