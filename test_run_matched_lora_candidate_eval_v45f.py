#!/usr/bin/env python3

import json

import pytest

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45f as subject


def test_failure_evidence_is_bound_without_holdout_v45f():
    value = subject.failure_evidence_v45f()
    assert value["failed_rule"] == (
        "cross-GPU candidate replica bit-exact shadow equality"
    )
    assert value["heldout_or_holdout_opened"] is False
    assert value["raw_semantics_inspected_for_revision"] is False


def test_replica_ranges_and_mean_rank_v45f():
    metrics = {}
    for logical_index, (logical, replicas) in enumerate(
        subject.prior.LOGICAL_REPLICAS_V45E.items()
    ):
        for replica_index, replica in enumerate(replicas):
            metrics[replica] = {
                "generated_equal_unit_mean_reward": logical_index + replica_index / 10,
                "generated_row_mean_reward": logical_index + replica_index / 10,
                "generated_exact_count": logical_index + replica_index,
                "generated_nonzero_count": 10 + logical_index + replica_index,
                "teacher_forced_equal_unit_mean_answer_logprob": -3 + logical_index,
            }
    ranges = {
        logical: {
            "shadow": subject.replicated_summary_v45f(
                metrics, replicas, (
                    "generated_equal_unit_mean_reward", "generated_exact_count",
                    "generated_nonzero_count",
                    "teacher_forced_equal_unit_mean_answer_logprob",
                )
            )
        }
        for logical, replicas in subject.prior.LOGICAL_REPLICAS_V45E.items()
    }
    assert ranges["sft_v42i"]["shadow"][
        "generated_equal_unit_mean_reward"
    ]["range"] == pytest.approx(0.1)
    assert subject.logical_mean_selection_key_v45f(
        ranges, "sft_v42k"
    ) > subject.logical_mean_selection_key_v45f(ranges, "sft_v42j")


def test_six_base_equivalence_remains_exact_v45f():
    metrics = {arm: {"same": True} for arm in subject.prior.BASE_ARMS_V45E}
    assert subject.assert_base_equivalence_v45f(metrics, "synthetic")[
        "all_six_base_outputs_exact"
    ] is True
    metrics["base_f"] = {"same": False}
    with pytest.raises(RuntimeError, match="six-base equivalence"):
        subject.assert_base_equivalence_v45f(metrics, "synthetic")


def test_builder_loader_zero_protected_semantics_v45f(tmp_path, monkeypatch):
    import build_matched_lora_candidate_eval_preregistration_v45f as builder
    protected = {item["path"] for item in core.PROTECTED_INPUTS_V44A.values()}
    original = core.file_sha256

    def guarded(path):
        assert str(path) not in protected
        return original(path)

    monkeypatch.setattr(core, "file_sha256", guarded)
    value = builder.build()
    assert value["protected_semantics_inspected_during_v45f_revision"] is False
    assert value["heldout_or_holdout_access_authorized"] is False
    assert value["conservative_replica_consensus_v45f"][
        "logical_candidate_eligible_iff_both_replicas_pass"
    ] is True
    assert value["raw_persistence_policy_v45f"][
        "raw_questions_answers_or_generations_persisted"
    ] is False
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    args = type("Args", (), {
        "preregistration": str(path),
        "preregistration_sha256": original(path),
        "preregistration_content_sha256": value[
            "content_sha256_before_self_field"
        ],
    })()
    assert subject.load_preregistration_v45f(args)["logical_candidates"] == [
        "sft_v42i", "sft_v42j", "sft_v42k"
    ]
