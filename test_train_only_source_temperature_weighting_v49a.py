#!/usr/bin/env python3

import math
from pathlib import Path

import build_train_only_source_temperature_design_v49a as builder
import v49a_train_only_source_temperature_weighting as subject


def test_v434_projection_preserves_exact_frozen_train_membership():
    value = subject.analyze()
    identity = value["membership_and_weight_identities"]
    assert identity["root_membership_exactly_frozen_v412_fold3_train"] is True
    assert identity["train_rows"] == 448
    assert identity["train_conflict_units"] == 208
    assert identity["train_jsonl_sha256"] == subject.V434_TRAIN_SHA256
    assert value["access_firewall"]["non_train_semantics_opened_received_or_inferred"] is False


def test_capped_source_temperature_reduces_dominance_without_explosion():
    value = subject.analyze()
    diagnostics = value["diagnostics"]
    bounds = diagnostics["weight_bounds"]
    ess = diagnostics["effective_sample_size"]
    concentration = diagnostics["source_concentration"]
    assert bounds["min_applied_multiplier"] >= 2 / 3 - 1e-15
    assert bounds["max_applied_multiplier"] <= 1.5 + 1e-15
    assert bounds["alternative_max_to_min_ratio"] <= (
        bounds["current_max_to_min_ratio"] + 1e-12
    )
    assert ess["source_improvement_fraction"] >= 0.20
    assert ess["row_retention_fraction"] >= 0.90
    assert ess["conflict_unit_retention_fraction"] >= 0.90
    assert concentration["current_top_two_sources"] == ["rope365", "kinbakutoday"]
    assert concentration["alternative_top_two_sources"] == ["rope365", "kinbakutoday"]
    assert concentration["top_two_mass_delta"] <= -0.08
    assert all(abs(row["mass_delta"]) <= 1e-15 for row in value["per_category_mass"])


def test_design_is_deterministic_self_hashed_and_has_no_launch_authority():
    first = builder.build()
    second = builder.build()
    assert first == second
    content = first.pop("content_sha256_before_self_field")
    assert content == builder.engine.canonical_sha256(first)
    assert first["gpu_launch_authorized"] is False
    assert first["training_launch_authorized"] is False
    assert first["evaluation_launch_authorized"] is False
    assert first["recommendation"]["merits_one_later_preregistered_hpo_arm"] is True
    source = "\n".join(
        Path(path).read_text(encoding="utf-8").lower()
        for path in (subject.__file__, builder.__file__)
    )
    for forbidden in ("ood_qa_v3", "ood_prose_v3", "eval_qa", "holdout_eval"):
        assert forbidden not in source
