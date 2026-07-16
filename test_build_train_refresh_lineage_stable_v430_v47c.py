#!/usr/bin/env python3

import json
from pathlib import Path

import build_train_refresh_lineage_stable_v430_v47c as subject


def test_lineage_stable_refresh_preserves_original_fold_membership(monkeypatch):
    def forbidden_eval_facts(*args, **kwargs):
        raise AssertionError("V47C attempted protected evaluation access")

    monkeypatch.setattr(subject.replay.curated, "eval_facts", forbidden_eval_facts)
    value, payloads = subject.construct()
    proof = value["lineage_stability_proof"]
    assert proof["components_before"] == proof["components_after"] == 259
    assert proof["root_membership_sets_identical"] is True
    assert proof["unit_row_multiplicities_identical"] is True
    assert proof["fold_assignment_changes"] == 0
    assert proof["fold_3_root_memberships_retained"] == 51
    assert payloads[subject.TRAIN].count(b"\n") == 448
    assert payloads[subject.SHADOW].count(b"\n") == 83


def test_lineage_stable_fold_is_disjoint_and_has_no_protected_binding():
    value, _ = subject.construct()
    fold = value["fold"]
    assert fold["train"]["conflict_units"] == 208
    assert fold["shadow_dev"]["conflict_units"] == 51
    assert not any(fold["train_dev_edge_identity_intersections"].values())
    assert value["access_firewall"]["eval_ood_holdout_or_benchmark_opened"] is False
    source = Path(subject.__file__).read_text(encoding="utf-8").lower()
    assert "v47b" not in source
    assert "v46d" not in source
    serialized = json.dumps(value).lower()
    assert "holdout_eval" not in serialized
