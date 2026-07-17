#!/usr/bin/env python3

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import recipe_evaluation_contract_v1 as subject


def _selection(contract: dict) -> dict:
    value = {
        "schema": "specialist-recipe-selection-receipt-v1",
        "status": "recipe_selected_frozen_hpo_closed",
        "contract_content_sha256": contract[
            "content_sha256_before_self_field"
        ],
        "hpo_closed": True,
        "protected_access_count_before_selection": 0,
        "selected_recipe_id": "frozen-recipe-a",
        "selected_checkpoint_sha256": "a" * 64,
    }
    value["content_sha256_before_self_field"] = subject.canonical_sha256(value)
    return value


def _attempt(
    arm: str,
    *,
    seconds: float,
    optimization_rollouts: int,
    evaluation_rollouts: int = 83,
    generated_tokens: int = 1000,
    teacher_tokens: int = 2000,
    sft_tokens: int = 0,
) -> dict:
    return {
        "arm": arm,
        "gpu_residency_intervals": [
            {"physical_gpu_id": gpu, "start_s": 0.0, "end_s": seconds}
            for gpu in range(4)
        ],
        "optimization_generated_rollouts": optimization_rollouts,
        "evaluation_generated_rollouts": evaluation_rollouts,
        "generated_tokens": generated_tokens,
        "teacher_forced_tokens": teacher_tokens,
        "sft_nonpadding_tokens": sft_tokens,
    }


def _synthetic_row(question: str, answer: str, identity: str) -> dict:
    return {
        "question": question,
        "answer": answer,
        "document_sha256": identity * 64,
        "url": f"https://example-{identity}.invalid/article",
    }


def test_build_is_deterministic_content_addressed_and_fully_disjoint():
    first = subject.build_contract()
    second = subject.build_contract()
    assert first == second
    subject.validate_contract(first)
    assert first["disjointness"]["passed"] is True
    assert all(
        value["colliding_row_pairs"] == 0
        for value in first["disjointness"]["audit"]["pairs"].values()
    )
    assert first["content_sha256_before_self_field"] == subject.canonical_sha256({
        key: value for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })


def test_current_protected_subset_removes_contaminated_legacy_rows():
    value = subject.build_contract()
    protected = value["roles"]["protected_holdout"]
    exclusion = protected["excluded_for_current_train_or_dev_collision"]
    assert protected["legacy_heldout_candidate_rows"] == 18
    assert protected["rows"] == 12
    assert protected["documents"] == 8
    assert exclusion == {
        "candidate_rows": 18,
        "selected_rows": 12,
        "excluded_rows": 6,
        "excluded_row_reason_counts": {
            "document_sha256": 4,
            "normalized_url": 6,
            "raw_lineage": 0,
            "near_duplicate": 0,
        },
    }
    assert len(protected["selected_opaque_item_identities"]) == 12
    assert all(
        len(identity) == 64
        for identity in protected["selected_opaque_item_identities"]
    )


def test_contract_persists_no_selected_protected_text_or_url():
    value = subject.build_contract()
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    selected = set(
        value["roles"]["protected_holdout"][
            "selected_opaque_item_identities"
        ]
    )
    protected_rows = [
        row for row in subject._read_jsonl(subject.DOMAIN_EVAL)
        if row.get("split") == "heldout"
        and subject._item_identity(row["item_id"]) in selected
    ]
    assert len(protected_rows) == 12
    for row in protected_rows:
        for field in subject.PROTECTED_TEXT_FIELDS:
            text = row.get(field)
            if isinstance(text, str) and text:
                assert text not in serialized
    minimized = value["content_minimization"]
    assert all(minimized[key] is False for key in (
        "protected_question_persisted",
        "protected_answer_persisted",
        "protected_excerpt_persisted",
        "protected_url_persisted",
        "protected_per_item_metric_persisted",
    ))


def test_near_duplicate_rule_catches_reformatted_and_close_copies():
    role_rows = {
        "train": [subject._record(_synthetic_row(
            "How should the rope be checked before every use?",
            "Inspect its entire length for damage and retire unsafe rope.",
            "a",
        ), "train")],
        "dev": [subject._record(_synthetic_row(
            "What unrelated material property is being tested?", "Color.", "b"
        ), "dev")],
        "protected_holdout": [subject._record(_synthetic_row(
            "HOW should rope be checked before every use!",
            "Inspect the entire length for damage; retire unsafe rope.",
            "c",
        ), "protected_holdout")],
        "ood_qa": [subject._record(_synthetic_row(
            "Which planet is closest to the Sun?", "Mercury.", "d"
        ), "ood_qa")],
        "ood_prose": [subject._record({
            "title": "A separate passage",
            "text": "This prose has no lexical relationship to any QA item.",
            "url": "https://example-e.invalid/article",
        }, "ood_prose")],
    }
    audit = subject.audit_role_records(role_rows)
    collision = audit["pairs"]["train__protected_holdout"]
    assert audit["passed"] is False
    assert collision["colliding_row_pairs"] == 1
    assert collision["by_identity_domain"]["near_duplicate"] == 1
    assert collision["by_identity_domain"]["document_sha256"] == 0
    assert collision["by_identity_domain"]["normalized_url"] == 0


def test_only_registered_train_bytes_can_enter_adaptation():
    value = subject.build_contract()
    subject.assert_adaptation_inputs([subject.TRAIN], value)
    with pytest.raises(RuntimeError, match="protected holdout"):
        subject.assert_adaptation_inputs([subject.DOMAIN_EVAL], value)
    with pytest.raises(RuntimeError, match="absent from the audited train"):
        subject.assert_adaptation_inputs([subject.DEV], value)


def test_compute_ledger_counts_failed_observed_work_and_rejects_overlap():
    attempts = [
        _attempt("a", seconds=10, optimization_rollouts=100),
        _attempt("a", seconds=2, optimization_rollouts=5),
        _attempt("b", seconds=12, optimization_rollouts=105),
    ]
    totals = subject.aggregate_compute_ledger(attempts)
    assert totals["a"]["charged_gpu_seconds"] == 48.0
    assert totals["a"]["optimization_generated_rollouts"] == 105
    assert totals["a"]["evaluation_generated_rollouts"] == 166
    assert totals["b"]["charged_gpu_seconds"] == 48.0
    bad = _attempt("x", seconds=2, optimization_rollouts=1)
    bad["gpu_residency_intervals"].append({
        "physical_gpu_id": 0, "start_s": 1.0, "end_s": 3.0,
    })
    with pytest.raises(ValueError, match="overlapping"):
        subject.charge_compute_attempt(bad)


def test_estimator_and_compute_matched_budget_modes_fail_closed():
    contract = subject.build_contract()
    estimator = subject.aggregate_compute_ledger([
        _attempt("a", seconds=1000, optimization_rollouts=2048),
        _attempt("b", seconds=1100, optimization_rollouts=2048),
    ])
    assert subject.validate_compute_match(
        estimator, mode="estimator_control", contract=contract
    )["passed"]
    estimator["b"]["optimization_generated_rollouts"] -= 1
    with pytest.raises(RuntimeError, match="rollout budgets differ"):
        subject.validate_compute_match(
            estimator, mode="estimator_control", contract=contract
        )

    matched = subject.aggregate_compute_ledger([
        _attempt("a", seconds=1000, optimization_rollouts=1000),
        _attempt("b", seconds=1015, optimization_rollouts=1200),
    ])
    assert subject.validate_compute_match(
        matched, mode="compute_matched_quality", contract=contract
    )["passed"]
    unmatched = subject.aggregate_compute_ledger([
        _attempt("a", seconds=1000, optimization_rollouts=1000),
        _attempt("b", seconds=1050, optimization_rollouts=1200),
    ])
    with pytest.raises(RuntimeError, match="GPU-second matched"):
        subject.validate_compute_match(
            unmatched, mode="compute_matched_quality", contract=contract
        )


def test_protected_access_is_claimed_before_read_and_consumed_on_first_use():
    contract = subject.build_contract()
    selection = _selection(contract)
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "protected-access.json"
        state = subject.claim_protected_access_once(
            state_path, contract, selection
        )
        assert state["status"] == "claimed_before_source_read"
        assert state["protected_source_opened"] is False
        with pytest.raises(FileExistsError):
            subject.claim_protected_access_once(state_path, contract, selection)

        def loader(path: Path) -> list[dict]:
            before = json.loads(state_path.read_text())
            assert before["status"] == "claimed_before_source_read"
            assert before["access_count"] == 1
            return subject._read_jsonl(path)

        rows = subject.load_claimed_protected_rows(
            state_path, contract, loader=loader
        )
        assert len(rows) == 12
        consumed = json.loads(state_path.read_text())
        assert consumed["status"] == "consumed_no_retry"
        assert consumed["protected_source_opened"] is True
        with pytest.raises(RuntimeError, match="already consumed"):
            subject.load_claimed_protected_rows(state_path, contract)


def test_terminal_receipt_rejects_raw_or_per_item_content():
    subject.validate_terminal_aggregate_receipt({
        "schema": "terminal-aggregate-v1",
        "aggregate_only": True,
        "metrics": {"mean_reward": 0.5, "exact_count": 7},
        "intervals": {"reward_lcb": -0.01, "reward_ucb": 0.04},
    })
    for forbidden in (
        {"aggregate_only": True, "per_item": [0.5]},
        {"aggregate_only": True, "answer": "protected sentinel"},
        {"aggregate_only": True, "raw_completion": "protected sentinel"},
    ):
        with pytest.raises(RuntimeError, match="forbidden field"):
            subject.validate_terminal_aggregate_receipt(forbidden)


def test_persisted_preregistration_matches_rebuild_when_present():
    if subject.CONTRACT.exists():
        assert json.loads(subject.CONTRACT.read_text()) == subject.build_contract()

