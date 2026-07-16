#!/usr/bin/env python3
"""Regression tests for the compact V61A census evidence finalizer."""

from __future__ import annotations

import json

import finalize_lora_es_baseline_census_evidence_v61a as subject


def test_v61a_completed_evidence_is_content_free_and_fail_closed():
    value = subject.build_evidence_v61a()
    assert value["status"] == (
        "complete_content_free_characterization_stable_exact_support_fail_closed"
    )
    assert value["coverage"] == {
        "rows": 448,
        "conflict_units": 208,
        "actors": 4,
        "numeric_actor_metrics": 1792,
    }
    assert value["scientific_gate"]["later_v61_hpo_authorized"] is False
    assert value["scientific_gate"]["threshold_or_quota_relaxed_after_outcomes"] is False
    assert value["raw_question_answer_or_generation_text_persisted"] is False
    assert value["protected_semantics_opened"] is False
    assert value["ood_shadow_or_terminal_holdout_opened"] is False


def test_v61a_observed_strata_and_numeric_instability_are_exact():
    outcome = subject.build_evidence_v61a()["outcome_characterization"]
    assert {key: item["total"] for key, item in outcome["unit_stratum_counts"].items()} == {
        "actor_unstable": 140,
        "difficult": 30,
        "stable_exact": 3,
        "stable_partial": 35,
    }
    assert outcome["row_stratum_counts"] == {
        "actor_unstable": 249,
        "difficult": 92,
        "stable_exact": 3,
        "stable_partial": 104,
    }
    assert outcome["f1_range_rows_strictly_above"] == {
        "1e-12": 249,
        "0.01": 225,
        "0.05": 124,
        "0.1": 55,
        "0.25": 4,
    }
    assert outcome["all_four_exact_rows"] == 3
    assert outcome["any_actor_exact_rows"] == 4


def test_v61a_gpu_attribution_and_cleanup_are_exact():
    value = subject.build_evidence_v61a()
    telemetry = value["telemetry"]
    assert telemetry["all_four_attributed_positive"] is True
    assert telemetry["foreign_compute_pid_rows"] == 0
    assert set(telemetry["by_gpu"]) == {"0", "1", "2", "3"}
    assert all(item["positive_samples"] > 0 for item in telemetry["by_gpu"].values())
    assert value["cleanup"] == {
        "engine_kill_count": 4,
        "placement_group_remove_count": 4,
        "all_four_gcs_states_removed": True,
        "sealed_final_gpu_idle": True,
    }


def test_v61a_manifest_is_self_hashed_and_contains_no_chat_text():
    value = subject.build_evidence_v61a()
    content = value.pop("content_sha256_before_self_field")
    assert content == subject.strata.canonical_sha256_v61a(value)
    serialized = json.dumps(value, ensure_ascii=False)
    assert not any(marker in serialized for marker in subject.RAW_MARKERS)
