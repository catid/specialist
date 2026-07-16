#!/usr/bin/env python3

import build_v46d_terminal_holdout_receipt as subject


def test_terminal_receipt_binds_consumed_failure_without_reopening_v46d():
    value = subject.build()
    outcome = value["audited_outcome"]
    policy = value["terminal_policy"]
    assert value["status"] == "terminal_holdout_consumed_gate_failed_no_retry"
    assert outcome["single_holdout_access_consumed"] is True
    assert outcome["holdout_semantic_access_count"] == 1
    assert outcome["fixed_gate_passed"] is False
    assert outcome["only_failed_point_gate"] == (
        "mean_reward_point_non_degradation"
    )
    assert outcome["mean_reward_delta"] == -0.0036501348892653135
    assert outcome["all_four_gpus_resident_and_positive"] is True
    assert outcome["raw_questions_answers_or_generations_persisted"] is False
    assert policy["retry_or_reopen_holdout_permitted"] is False
    assert policy["use_result_for_candidate_selection_permitted"] is False
    assert policy["use_result_for_hyperparameter_tuning_permitted"] is False
    assert policy["use_result_for_dataset_selection_or_resampling_permitted"] is False
    assert value[
        "protected_or_holdout_semantics_inspected_while_building_receipt"
    ] is False
    assert value["content_sha256_before_self_field"] == subject.core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def test_terminal_receipt_does_not_hash_or_open_holdout_v46d(monkeypatch):
    holdout = subject.ROOT / "data/eval_qa_v3.jsonl"
    original_hash = subject.core.file_sha256
    original_read_bytes = subject.Path.read_bytes

    def guarded_hash(path):
        assert subject.Path(path).resolve() != holdout.resolve()
        return original_hash(path)

    def guarded_read_bytes(path):
        assert subject.Path(path).resolve() != holdout.resolve()
        return original_read_bytes(path)

    monkeypatch.setattr(subject.core, "file_sha256", guarded_hash)
    monkeypatch.setattr(subject.Path, "read_bytes", guarded_read_bytes)
    assert subject.build()["holdout_reopened_or_rehashed_while_building_receipt"] is False
