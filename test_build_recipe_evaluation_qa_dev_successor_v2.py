"""Synthetic-only security and quality tests for the QA DEV successor."""

from __future__ import annotations

import copy

import pytest

import build_recipe_evaluation_qa_dev_successor_v2 as subject


def _sha(number: int) -> str:
    return f"{number:064x}"


def _sources() -> list[dict]:
    return [
        {
            "repository_relative_path": f"synthetic/source_{index}/CORPUS.md",
            "opaque_item_identity": _sha(100 + index),
            "source_path_identity_sha256": _sha(200 + index),
            "file_sha256": _sha(300 + index),
            "permission": "read_for_manual_multi_item_qa_dev_curation_only",
        }
        for index in range(4)
    ]


def _qa_rows() -> list[dict]:
    sources = _sources()
    rows = []
    for index in range(20):
        source = sources[index // 5]
        rows.append(
            {
                "schema": subject.QA_SCHEMA,
                "id": f"synthetic_topic_{index:03d}",
                "split": "dev",
                "role": "dev_only_never_model_adaptation",
                "question": (
                    f"How does synthetic mechanism {index} affect a bounded load?"
                ),
                "answer": (
                    f"Synthetic mechanism {index} changes the bounded response "
                    "through a distinct, directly observed process."
                ),
                "source_document": {
                    key: source[key]
                    for key in subject.SOURCE_KEYS
                },
                "grounding": {
                    "evidence": f"Synthetic direct evidence {index}.",
                    "locator": f"Synthetic section {index}",
                    "support": "direct",
                },
                "reviewer": {
                    "reviewer_id": "synthetic_manual_reviewer",
                    "reviewer_type": "manual_agent",
                    "reviewed_at_utc": "2026-07-17T00:00:00+00:00",
                },
                "curation": {
                    "method": "manual_exact_source_read",
                    "decision": "include",
                    "grounding_check": "passed",
                    "scope_check": "passed",
                    "safety_check": "passed",
                    "url_or_source_trivia": False,
                },
            }
        )
    return rows


def _curation_rows(qa_rows: list[dict]) -> list[dict]:
    return [
        {
            "schema": subject.CURATION_SCHEMA,
            "qa_id": row["id"],
            "source_document_identity": {
                key: row["source_document"][key]
                for key in (
                    "opaque_item_identity",
                    "source_path_identity_sha256",
                    "file_sha256",
                )
            },
            "grounding": {
                "evidence_locator": f"Synthetic locator {index}",
                "alignment": "direct",
                "answer_faithfulness": "passed",
            },
            "reviewer": dict(row["reviewer"]),
            "curation": {
                "decision": "include",
                "question_type": "mechanistic_application",
                "usefulness": "passed",
                "clarity": "passed",
                "safety": "passed",
                "source_or_url_trivia": False,
                "weird_list_recall": False,
                "unsupported_advice": False,
            },
        }
        for index, row in enumerate(qa_rows)
    ]


def _synthetic_contract() -> dict:
    terminal_sources = [
        {
            "added_commit": "a" * 40,
            "corpus_file_sha256": _sha(500 + index),
            "metadata_bundle_sha256": _sha(600 + index),
            "opaque_item_identity": _sha(700 + index),
            "source_path_identity_sha256": _sha(800 + index),
            "url_identity_set_sha256": _sha(900 + index),
        }
        for index in range(subject.EXPECTED_TERMINAL_SOURCE_COUNT)
    ]
    value = {
        "schema": subject.SCHEMA,
        "status": subject.STATUS,
        "predecessor": {
            "file_sha256": subject.EXPECTED_FILE_SHA256[subject.V2_CONTRACT],
            "content_sha256_before_self_field": (
                subject.EXPECTED_V2_CONTENT_SHA256
            ),
        },
        "bound_inputs": {
            "allowlist": {
                "file_sha256": subject.EXPECTED_FILE_SHA256[subject.ALLOWLIST]
            },
            "qa_dev": {
                "file_sha256": subject.EXPECTED_FILE_SHA256[subject.QA_DEV]
            },
            "curation": {
                "file_sha256": subject.EXPECTED_FILE_SHA256[subject.CURATION]
            },
            "report": {
                "file_sha256": subject.EXPECTED_FILE_SHA256[subject.REPORT]
            },
        },
        "qa_dev": {
            "role": "dev_only_never_model_adaptation",
            "source_identity_set_sha256": (
                subject.EXPECTED_DEV_SOURCE_IDENTITY_SET_SHA256
            ),
            "source_bindings": [{} for _ in range(subject.EXPECTED_SOURCE_COUNT)],
            "quality_audit": {
                "qa_row_count": subject.EXPECTED_QA_ROWS,
                "curation_row_count": subject.EXPECTED_CURATION_ROWS,
            },
        },
        "source_disjointness": {
            "passed": True,
            "intersection_counts": {
                "opaque_item_identity": 0,
                "source_path_identity_sha256": 0,
                "corpus_file_sha256": 0,
            },
        },
        "terminal_boundary": {
            "source_count": subject.EXPECTED_TERMINAL_SOURCE_COUNT,
            "selected_identity_set_sha256": (
                subject.EXPECTED_TERMINAL_IDENTITY_SET_SHA256
            ),
            "selected_sources_opaque_only": terminal_sources,
            "semantic_source_fields_persisted": False,
            "terminal_source_opened_or_resolved_by_builder": False,
            "terminal_claim_state_probed_by_builder": False,
            "terminal_claim_created_by_successor": False,
            "terminal_access_authorized": False,
        },
        "authority": {
            "qa_dev_scoring_authorized": True,
            "general_quality_dev_scoring_authorized": True,
            "model_adaptation_or_training_authorized": False,
            "gradient_or_optimizer_update_authorized": False,
            "qa_rows_may_enter_training_sampler": False,
            "qa_hpo_or_recipe_promotion_authorized": False,
            "checkpoint_selection_authorized": False,
            "ood_evaluation_authorized": False,
            "terminal_evaluation_authorized": False,
            "final_benchmark_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = subject.canonical_sha256(value)
    return value


def test_synthetic_useful_grounded_unique_bundle_passes():
    rows = _qa_rows()
    qa_audit = subject.validate_qa_rows(rows, _sources())
    curation_audit = subject.validate_curation_rows(
        _curation_rows(rows), rows
    )
    assert qa_audit["qa_row_count"] == 20
    assert qa_audit["per_source_counts"] == [5, 5, 5, 5]
    assert qa_audit["url_or_source_trivia_rows"] == 0
    assert curation_audit["qa_and_curation_id_sets_match"] is True


def test_contract_tamper_fails_even_when_self_hash_is_recomputed():
    value = _synthetic_contract()
    subject.validate_contract(value)
    value["authority"]["model_adaptation_or_training_authorized"] = True
    value["content_sha256_before_self_field"] = subject.canonical_sha256(
        subject._without_self_hash(value)
    )
    with pytest.raises(RuntimeError, match="invalid recipe evaluation"):
        subject.validate_contract(value)


@pytest.mark.parametrize(
    "domain,terminal_key",
    (
        ("opaque_item_identity", "opaque_item_identity"),
        ("source_path_identity_sha256", "source_path_identity_sha256"),
        ("file_sha256", "corpus_file_sha256"),
    ),
)
def test_source_overlap_via_each_opaque_identity_domain_fails(
    domain, terminal_key
):
    sources = _sources()
    terminal = {
        "opaque_item_identity": _sha(9991),
        "source_path_identity_sha256": _sha(9992),
        "corpus_file_sha256": _sha(9993),
    }
    terminal[terminal_key] = sources[0][domain]
    with pytest.raises(RuntimeError, match="overlaps terminal"):
        subject.audit_source_disjointness(sources, [terminal])


@pytest.mark.parametrize(
    "bad_question",
    (
        "Which canonical URL was listed for this synthetic method?",
        "What page number contains the synthetic method?",
        "What is the website address for the synthetic method?",
    ),
)
def test_url_and_source_trivia_questions_fail(bad_question):
    rows = _qa_rows()
    rows[0]["question"] = bad_question
    with pytest.raises(RuntimeError, match="URL or source trivia"):
        subject.validate_qa_rows(rows, _sources())


def test_missing_curation_record_fails_closed():
    rows = _qa_rows()
    curation = _curation_rows(rows)
    curation.pop()
    with pytest.raises(RuntimeError, match="exactly 20"):
        subject.validate_curation_rows(curation, rows)


@pytest.mark.parametrize("semantic_key", sorted(subject.TERMINAL_SEMANTIC_KEYS))
def test_terminal_semantic_leakage_fails(semantic_key):
    with pytest.raises(RuntimeError, match="semantic leakage"):
        subject._assert_no_terminal_semantics(
            {"opaque_item_identity": _sha(1), semantic_key: "synthetic secret"}
        )


def test_chat_control_semantic_leakage_fails():
    rows = _qa_rows()
    rows[0]["answer"] += " </think>"
    with pytest.raises(RuntimeError, match="chat-control semantic leakage"):
        subject.validate_qa_rows(rows, _sources())


def test_protected_path_is_denied_before_open(monkeypatch):
    monkeypatch.setattr(
        subject.os,
        "open",
        lambda *_args, **_kwargs: pytest.fail("denied path reached os.open"),
    )
    with pytest.raises(RuntimeError, match="denied before resolution"):
        subject.read_allowed_semantic_bytes(
            subject.ROOT / "data/eval_qa_v3.jsonl"
        )


def test_only_dev_scoring_purposes_are_authorized():
    for purpose in sorted(subject.ALLOWED_SCORING_PURPOSES):
        subject.assert_authorized_use(purpose)
    for purpose in sorted(subject.PROHIBITED_PURPOSES):
        with pytest.raises(RuntimeError, match="not authorized"):
            subject.assert_authorized_use(purpose)
    with pytest.raises(RuntimeError, match="unknown purpose"):
        subject.assert_authorized_use("synthetic_unregistered_use")


def test_duplicate_question_fails_quality_gate():
    rows = _qa_rows()
    rows[1]["question"] = rows[0]["question"]
    with pytest.raises(RuntimeError, match="questions are not unique"):
        subject.validate_qa_rows(rows, _sources())


def test_contract_terminal_semantics_fail_before_other_validation():
    value = copy.deepcopy(_synthetic_contract())
    value["terminal_boundary"]["answer"] = "synthetic hidden answer"
    value["content_sha256_before_self_field"] = subject.canonical_sha256(
        subject._without_self_hash(value)
    )
    with pytest.raises(RuntimeError, match="semantic leakage"):
        subject.validate_contract(value)
