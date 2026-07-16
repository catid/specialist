#!/usr/bin/env python3
"""Regression tests for the compact V61B repeat-census finalizer."""

from __future__ import annotations

import json

import finalize_lora_es_baseline_repeat_evidence_v61b as subject


def test_v61b_evidence_is_complete_content_free_characterization():
    value = subject.build_evidence_v61b()
    assert value["status"] == "complete_content_free_same_seed_repeat_characterization"
    assert value["coverage"] == {
        "rows": 448,
        "conflict_units": 208,
        "actors": 4,
        "sequential_passes": 2,
        "numeric_actor_pass_metrics": 3584,
    }
    assert value["raw_question_answer_or_generation_text_persisted"] is False
    assert value["selection_update_or_promotion_performed"] is False
    assert value["protected_semantics_opened"] is False
    assert value["v61a_row_level_evidence_opened"] is False


def test_v61b_repeat_and_cross_actor_counts_are_exact():
    repeat = subject.build_evidence_v61b()["repeat_characterization"]
    assert repeat["within_actor_all_row_comparisons"][
        "f1_absolute_delta_gt_counts"
    ] == subject.EXPECTED_WITHIN_COUNTS
    assert repeat["within_actor_exact_label_disagreement_rows"] == 0
    assert repeat["within_actor_nonzero_label_disagreement_rows"] == 5
    assert [
        item["f1_absolute_delta_gt_counts"]
        for item in repeat["cross_actor_same_seed_by_pass"]
    ] == subject.EXPECTED_CROSS_COUNTS
    assert repeat["causal_variance_source_claimed"] is False


def test_v61b_all_four_gpus_and_cleanup_are_sealed():
    value = subject.build_evidence_v61b()
    assert value["telemetry"]["all_four_attributed_positive"] is True
    assert value["telemetry"]["foreign_compute_pid_rows"] == 0
    assert set(value["telemetry"]["by_gpu"]) == {"0", "1", "2", "3"}
    assert all(
        item["positive_samples"] > 0
        for item in value["telemetry"]["by_gpu"].values()
    )
    assert value["cleanup"]["all_four_gcs_states_removed"] is True
    assert value["cleanup"]["sealed_final_gpu_idle"] is True


def test_v61b_manifest_self_hash_and_no_chat_text():
    value = subject.build_evidence_v61b()
    content = value.pop("content_sha256_before_self_field")
    assert content == subject.analysis_v61b.canonical_sha256_v61b(value)
    serialized = json.dumps(value, ensure_ascii=False)
    assert not any(marker in serialized for marker in subject.RAW_MARKERS)
