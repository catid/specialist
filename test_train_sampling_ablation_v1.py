#!/usr/bin/env python3

from __future__ import annotations

import json

import pytest

import recipe_evaluation_contract_v1 as evaluation_contract
import train_sampling_ablation_v1 as subject


def _qa_row(
    question: str,
    answer: str,
    *,
    kind: str = "qa_manual",
    source: str = "synthetic",
) -> dict:
    return {
        "question": question,
        "answer": answer,
        "text": f"Question: {question}\nAnswer: {answer}",
        "kind": kind,
        "source": source,
    }


def _inputs():
    _contract, rows, census, _evidence = subject._validate_inputs()
    frame = subject.build_frame(rows)
    ledger = subject.build_initial_weakness_ledger(census, frame)
    return rows, frame, ledger


def test_manifest_is_deterministic_and_content_addressed():
    first = subject.build_manifest()
    second = subject.build_manifest()
    assert first == second
    assert first["content_sha256_before_self_field"] == subject.canonical_sha256({
        key: value for key, value in first.items()
        if key != "content_sha256_before_self_field"
    })
    assert first["access_receipt"] == {
        "train_semantics_opened": True,
        "dev_semantics_opened": False,
        "ood_semantics_opened": False,
        "protected_semantics_opened": False,
        "model_or_gpu_accessed": False,
        "model_outcomes_used": "frozen V61A train-only census only",
    }


def test_variants_have_exact_compute_matched_cardinality_and_caps():
    value = subject.build_manifest()
    assert [item["name"] for item in value["variants"]] == [
        "uniform_capped", "category_stratified", "prioritized_capped"
    ]
    for panel in value["variants"]:
        assert panel["rows"] == 64
        assert len(panel["items"]) == 64
        assert len({item["unit_identity_sha256"] for item in panel["items"]}) == 64
        assert max(panel["source_counts"].values()) <= 15
        assert max(panel["category_counts"].values()) <= 20
        assert max(panel["source_counts"].values()) / panel["rows"] <= 0.234375
    assert value["variants"][1]["category_counts"] == subject.CATEGORY_QUOTAS
    assert value["variants"][2]["category_counts"] == subject.CATEGORY_QUOTAS
    assert value["variants"][2]["component_counts"] == {
        "category_uniform": 32,
        "weakness_replay": 32,
    }
    compute = value["compute_match"]
    assert compute["optimization_generated_rollouts_per_variant"] == 2048
    assert compute["optimization_generated_rollouts_per_variant"] == (
        compute["panel_rows_per_variant_per_generation"]
        * compute["mirrored_population_per_generation"]
        * compute["screen_generations"]
    )
    assert compute["contract_rollout_target"] == 2048


def test_compute_ledger_accepts_equal_sampling_arms_and_rejects_one_less_rollout():
    value = subject.build_manifest()
    contract = json.loads(subject.CONTRACT.read_text())
    totals = {
        panel["name"]: {
            "charged_gpu_seconds": 10000.0,
            "optimization_generated_rollouts": 2048,
            "evaluation_generated_rollouts": 83,
            "generated_tokens": 100000,
            "teacher_forced_tokens": 50000,
            "sft_nonpadding_tokens": 0,
        }
        for panel in value["variants"]
    }
    assert evaluation_contract.validate_compute_match(
        totals, mode="estimator_control", contract=contract
    )["passed"]
    totals["prioritized_capped"]["optimization_generated_rollouts"] -= 1
    with pytest.raises(RuntimeError, match="rollout budgets differ"):
        evaluation_contract.validate_compute_match(
            totals, mode="estimator_control", contract=contract
        )


def test_url_trivia_is_excluded_but_useful_resource_answers_are_retained():
    bad = _qa_row(
        "Which canonical Rope-topia URL was listed for the bowline tutorial?",
        "https://rope-topia.com/bowline",
        kind="qa_resource_index",
    )
    useful = _qa_row(
        "Where can a beginner find a structured suspension course?",
        "Crash Restraint offers suspension fundamentals at https://crash-restraint.com/.",
        kind="qa_resource_direct",
    )
    malformed = dict(useful)
    malformed["answer"] = "</think>\njute<|im_end|>"
    malformed["text"] = (
        f"Question: {malformed['question']}\nAnswer: {malformed['answer']}"
    )
    assert subject.exclusion_reason(bad) == "canonical_url_trivia"
    assert subject.exclusion_reason(useful) is None
    assert subject.classify_category(useful) == "resources"
    assert subject.exclusion_reason(malformed) == "defective_qa"


def test_frozen_frame_excludes_all_15_url_index_rows_from_every_variant():
    rows, frame, _ledger = _inputs()
    assert frame["eligible_rows"] == 433
    assert frame["exclusions"] == {"canonical_url_trivia": 15}
    assert len(frame["excluded_row_sha256s"]) == 15
    excluded = set(frame["excluded_row_sha256s"])
    manifest = subject.build_manifest()
    for panel in manifest["variants"]:
        assert not excluded & {item["row_sha256"] for item in panel["items"]}
    assert sum(subject.exclusion_reason(row) is not None for row in rows) == 15


@pytest.mark.parametrize(("question", "answer", "expected"), [
    (
        "How is a Somerville bowline tied around a single column?",
        "Form the loop, pass the bight, and finish the column tie.",
        "techniques",
    ),
    (
        "How should an indoor hardpoint be evaluated for suspension?",
        "Inspect the anchor and its load-bearing structure.",
        "rigging",
    ),
    (
        "What signs can indicate nerve compression?",
        "Numbness, tingling, weakness, or unusual pain.",
        "safety",
    ),
    (
        "Who influenced the development of this historical style?",
        "Its lineage includes several earlier practitioners.",
        "lineage",
    ),
    (
        "How should jute rope be stored after cleaning?",
        "Coil it only after it is completely dry.",
        "equipment",
    ),
    (
        "How can an uneven friction be corrected?",
        "Loosen and adjust it before adding tension.",
        "troubleshooting",
    ),
    (
        "Where can someone find an online rope course?",
        "Use a structured education resource.",
        "resources",
    ),
])
def test_seven_category_classifier(question, answer, expected):
    assert subject.classify_category(_qa_row(question, answer)) == expected


def test_initial_priority_uses_all_and_only_train_census_units_with_cap():
    _rows, frame, ledger = _inputs()
    subject.validate_weakness_ledger(ledger, frame)
    assert ledger["as_of_completed_generation"] == -1
    assert len(ledger["entries"]) == 208
    assert {
        item["unit_identity_sha256"] for item in ledger["entries"]
    } == frame["unit_ids"]
    multipliers = [
        float.fromhex(item["priority_multiplier_hex"])
        for item in ledger["entries"]
    ]
    assert min(multipliers) >= 1.0
    assert max(multipliers) <= 2.0
    assert ledger["source"]["nontrain_surfaces_opened"] is False
    assert ledger["source"]["model_updates_or_candidate_selection_used"] is False


def test_lagged_train_observation_updates_next_generation_only():
    _rows, frame, ledger = _inputs()
    easiest = min(
        ledger["entries"], key=lambda item: float.fromhex(
            item["ema_weakness_hex"]
        )
    )
    unit_id = easiest["unit_identity_sha256"]
    prior = float.fromhex(easiest["ema_weakness_hex"])
    updated = subject.update_weakness_ledger(
        ledger,
        frame,
        [{
            "unit_identity_sha256": unit_id,
            "mean_reward": 0.0,
            "exact_rate": 0.0,
            "nonzero_rate": 0.0,
            "observations": 16,
        }],
        completed_generation=0,
    )
    current = next(
        item for item in updated["entries"]
        if item["unit_identity_sha256"] == unit_id
    )
    assert float.fromhex(current["ema_weakness_hex"]) > prior
    assert float.fromhex(current["priority_multiplier_hex"]) <= 2.0
    assert updated["as_of_completed_generation"] == 0
    assert subject.build_manifest(
        generation=1, weakness_ledger=updated
    )["generation"] == 1
    with pytest.raises(RuntimeError, match="must predate"):
        subject.build_manifest(generation=0, weakness_ledger=updated)


def test_weakness_update_rejects_eval_fields_and_unknown_units():
    _rows, frame, ledger = _inputs()
    unit_id = ledger["entries"][0]["unit_identity_sha256"]
    valid = {
        "unit_identity_sha256": unit_id,
        "mean_reward": 0.5,
        "exact_rate": 0.0,
        "nonzero_rate": 1.0,
        "observations": 4,
    }
    with pytest.raises(RuntimeError, match="schema is not train-only"):
        subject.update_weakness_ledger(
            ledger, frame, [{**valid, "dev_reward": 1.0}],
            completed_generation=0,
        )
    with pytest.raises(RuntimeError, match="nontrain unit"):
        subject.update_weakness_ledger(
            ledger, frame,
            [{**valid, "unit_identity_sha256": "f" * 64}],
            completed_generation=0,
        )


def test_manifest_persists_no_question_answer_evidence_or_url_content():
    rows, _frame, _ledger = _inputs()
    value = subject.build_manifest()
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True)
    selected = {
        item["row_sha256"]
        for panel in value["variants"] for item in panel["items"]
    }
    for row in rows:
        if subject.conflict_v37a.row_sha256(row) not in selected:
            continue
        for field in ("question", "answer", "evidence", "url"):
            text = row.get(field)
            if isinstance(text, str) and len(text) >= 20:
                assert text not in raw
    minimized = value["content_minimization"]
    assert all(minimized[key] is False for key in (
        "question_persisted", "answer_persisted", "evidence_persisted",
        "url_persisted", "row_content_persisted",
    ))


def test_each_sealed_variant_materializes_exact_registered_train_rows():
    value = subject.build_manifest()
    subject.validate_manifest(value)
    for panel in value["variants"]:
        rows = subject.materialize_variant_rows(value, panel["name"])
        assert len(rows) == 64
        assert [subject.conflict_v37a.row_sha256(row) for row in rows] == [
            item["row_sha256"] for item in panel["items"]
        ]
        assert all(subject.exclusion_reason(row) is None for row in rows)


def test_persisted_manifest_matches_rebuild_when_present():
    if subject.OUTPUT.exists():
        assert json.loads(subject.OUTPUT.read_text()) == subject.build_manifest()
