from __future__ import annotations

import copy

import pytest

import build_high_information_generated_domain_authority_v1 as authority


def targets(*, source_a: int = 5, source_b: int = 5) -> dict:
    total = source_a + source_b
    return {
        "assistant_tokens": total,
        "by_source": {"source-a": source_a, "source-b": source_b},
        "by_category": {"category-a": source_a, "category-b": source_b},
        "by_task_family": {"closed_book_application": total},
        "by_task_subtype": {"direct_explanation": total},
        "by_generation_mode": {"positive": total},
    }


def candidate(
    identity: str,
    *,
    tokens: int,
    source: str,
    category: str,
    request_id: str | None = None,
    consensus: bool = True,
    manual_required: bool = False,
    exact: str | None = None,
    cluster: str | None = None,
) -> dict:
    digest = authority.corpus.canonical_sha256(identity)
    exact = exact or digest
    cluster = cluster or f"near-cluster-v1:{digest}"
    return {
        "selection_candidate_id": identity,
        "candidate_example_id": f"candidate-{identity}",
        "request_id": request_id or f"request-{identity}",
        "source_context_id": f"context-{identity}",
        "source_group_id": f"group-{source}",
        "resource_id": source,
        "artifact_id": f"artifact-{source}",
        "task_family": "closed_book_application",
        "task_subtype": "direct_explanation",
        "generation_mode": "positive",
        "category": category,
        "question": f"How does synthetic mechanism {identity} work?",
        "answer": f"Synthetic mechanism {identity} works by bounded evidence.",
        "assistant_qwen36_token_count": tokens,
        "evidence_quote_sha256s": ["e" * 64],
        "semantic_result": {
            "judge_consensus_passed": consensus,
            "manual_review_required": manual_required,
            "manual_review_reasons": (
                ["judge_gate_disagreement:source_entailment"]
                if manual_required else []
            ),
        },
        "semantic_record_sha256": "f" * 64,
        "semantic_pass_output_sha256s": {
            authority.primary_judge.PASS_NAMES[0]: "1" * 64,
            authority.primary_judge.PASS_NAMES[1]: "2" * 64,
        },
        "nli_result": {
            "verdict": "pass",
            "probabilities": {
                "entailment": 0.9,
                "neutral": 0.08,
                "contradiction": 0.02,
            },
        },
        "nli_result_sha256": "3" * 64,
        "structural_review_sha256": "4" * 64,
        "semantic_shard_receipt": {"output_sha256": "5" * 64},
        "dedupe": {
            "exact_key_sha256": exact,
            "near_duplicate_cluster_id": cluster,
        },
        "selection_eligibility_status": (
            "passed_two_pass_consensus" if consensus and not manual_required
            else "manual_review_unresolved"
        ),
        "selection_eligible": consensus and not manual_required,
        "manual_review_receipt": (
            {"status": "not_required"}
            if consensus and not manual_required else {"status": "unresolved"}
        ),
    }


def test_exact_key_uses_nfkc_casefold_and_whitespace():
    left = authority.exact_key("  FULLWIDTH Ａ test ", "Answer\nwith   spacing")
    right = authority.exact_key("fullwidth a TEST", "answer with spacing")
    assert left == right


def test_near_duplicate_clusters_are_source_aware_and_allow_only_distinct_views():
    rows = []
    for identity, source, subtype, answer, evidence in (
        ("selection-a", "source-a", "direct_explanation", "The bounded mechanism uses friction and load.", "1" * 64),
        ("selection-b", "source-a", "direct_explanation", "The bounded mechanism uses friction and load safely.", "1" * 64),
        ("selection-c", "source-b", "direct_explanation", "The bounded mechanism uses friction and load safely.", "1" * 64),
        ("selection-d", "source-a", "application_scenario", "The bounded mechanism uses friction and load in a scenario.", "2" * 64),
    ):
        row = candidate(
            identity,
            tokens=5,
            source=source,
            category="category-a" if source == "source-a" else "category-b",
        )
        row["question"] = "How does the bounded mechanism use friction and load?"
        row["answer"] = answer
        row["task_subtype"] = subtype
        row["evidence_quote_sha256s"] = [evidence]
        rows.append(row)
    clustered = {row["selection_candidate_id"]: row for row in authority.attach_deduplication(rows)}
    assert clustered["selection-a"]["dedupe"]["near_duplicate_cluster_id"] == (
        clustered["selection-b"]["dedupe"]["near_duplicate_cluster_id"]
    )
    assert clustered["selection-c"]["dedupe"]["near_duplicate_base_cluster_sha256"] != (
        clustered["selection-a"]["dedupe"]["near_duplicate_base_cluster_sha256"]
    )
    assert clustered["selection-d"]["dedupe"]["near_duplicate_cluster_id"] != (
        clustered["selection-a"]["dedupe"]["near_duplicate_cluster_id"]
    )


def test_atomic_solver_fills_exact_axes_and_prefers_consensus_duplicate():
    low = candidate(
        "selection-low",
        tokens=5,
        source="source-a",
        category="category-a",
        consensus=False,
        exact="a" * 64,
        cluster="near-cluster-v1:" + "a" * 64,
    )
    low["selection_eligibility_status"] = "passed_after_manual_resolution"
    low["selection_eligible"] = True
    high = candidate(
        "selection-high",
        tokens=5,
        source="source-a",
        category="category-a",
        exact="a" * 64,
        cluster="near-cluster-v1:" + "a" * 64,
    )
    other = candidate(
        "selection-other",
        tokens=5,
        source="source-b",
        category="category-b",
    )
    result = authority.solve_atomic_selection([low, high, other], targets())
    assert result["exact_solution"] is True
    assert result["accounting"]["assistant_tokens"] == 10
    assert result["deficits"]["assistant_tokens"] == 0
    assert "selection-high" in result["selected_candidate_ids"]
    assert "selection-low" not in result["selected_candidate_ids"]
    assert result["padding_used"] is False
    assert result["borrowing_used"] is False


def test_solver_reports_true_source_deficit_and_does_not_borrow_surplus():
    pool = [
        candidate("selection-a5", tokens=5, source="source-a", category="category-a"),
        candidate("selection-a4", tokens=4, source="source-a", category="category-a"),
        candidate("selection-b3", tokens=3, source="source-b", category="category-b"),
    ]
    result = authority.solve_atomic_selection(pool, targets())
    assert result["exact_solution"] is False
    assert result["accounting"]["by_source"] == {"source-a": 5, "source-b": 3}
    assert result["deficits"]["assistant_tokens"] == 2
    assert result["deficits"]["by_source"] == {"source-a": 0, "source-b": 2}
    assert result["deficits"]["by_category"] == {"category-a": 0, "category-b": 2}
    assert result["borrowing_used"] is False


def test_solver_never_splits_atomic_rows_to_hide_a_deficit():
    pool = [
        candidate("selection-a5", tokens=5, source="source-a", category="category-a"),
        candidate("selection-b4", tokens=4, source="source-b", category="category-b"),
        candidate("selection-b2", tokens=2, source="source-b", category="category-b"),
    ]
    result = authority.solve_atomic_selection(pool, targets())
    assert result["exact_solution"] is False
    assert result["accounting"]["assistant_tokens"] == 9
    assert result["deficits"]["by_source"]["source-b"] == 1
    assert result["truncation_used"] is False


def test_nonconsensus_is_ineligible_until_exact_manual_resolution():
    source = candidate(
        "selection-review",
        tokens=5,
        source="source-a",
        category="category-a",
        consensus=False,
        manual_required=True,
    )
    queue = authority.make_manual_queue([source])
    assert len(queue) == 1
    resolutions, receipt = authority.validate_manual_resolutions([], queue)
    unresolved = authority.apply_semantic_eligibility([source], resolutions)[0]
    assert unresolved["selection_eligible"] is False
    assert unresolved["selection_eligibility_status"] == "manual_review_unresolved"
    assert receipt["unresolved"] == 1

    resolution = {
        "schema": authority.MANUAL_RESOLUTION_SCHEMA,
        "selection_candidate_id": source["selection_candidate_id"],
        "candidate_example_id": source["candidate_example_id"],
        "semantic_record_sha256": source["semantic_record_sha256"],
        "queue_row_sha256": queue[0]["content_sha256_before_self_field"],
        "status": "resolved_pass",
        "reviewer_identity_sha256": "9" * 64,
        "protocol_sha256": authority.MANUAL_REVIEW_PROTOCOL_SHA256,
    }
    resolution["content_sha256_before_self_field"] = authority._self_address(resolution)
    resolutions, receipt = authority.validate_manual_resolutions([resolution], queue)
    resolved = authority.apply_semantic_eligibility([source], resolutions)[0]
    assert resolved["selection_eligible"] is True
    assert resolved["selection_eligibility_status"] == "passed_after_manual_resolution"
    assert resolved["manual_review_receipt"]["status"] == "resolved_pass"
    assert receipt["unresolved"] == 0


def test_manual_resolution_cannot_override_an_independent_nli_failure():
    source = candidate(
        "selection-nli-fail",
        tokens=5,
        source="source-a",
        category="category-a",
        consensus=False,
        manual_required=True,
    )
    source["nli_result"]["verdict"] = "fail"
    queue = authority.make_manual_queue([source])
    resolution = {
        "schema": authority.MANUAL_RESOLUTION_SCHEMA,
        "selection_candidate_id": source["selection_candidate_id"],
        "candidate_example_id": source["candidate_example_id"],
        "semantic_record_sha256": source["semantic_record_sha256"],
        "queue_row_sha256": queue[0]["content_sha256_before_self_field"],
        "status": "resolved_pass",
        "reviewer_identity_sha256": "9" * 64,
        "protocol_sha256": authority.MANUAL_REVIEW_PROTOCOL_SHA256,
    }
    resolution["content_sha256_before_self_field"] = authority._self_address(resolution)
    resolutions, _ = authority.validate_manual_resolutions([resolution], queue)
    evaluated = authority.apply_semantic_eligibility([source], resolutions)[0]
    assert evaluated["selection_eligible"] is False
    assert evaluated["selection_eligibility_status"] == "rejected_independent_nli"


def test_training_row_matches_fixed_mixed_snapshot_interface():
    source = candidate(
        "selection-row",
        tokens=5,
        source="source-a",
        category="category-a",
    )
    source["lane"] = "fill"
    source["rights_basis"] = {"status": "explicit_open_license"}
    source["safety_transfer_flags"] = ["synthetic_safety_flag"]
    source["context"] = {
        "text": "Synthetic bounded context.",
        "artifact_id": "artifact-source-a",
        "lineage": {"source_document_identity_sha256": "8" * 64},
    }
    source["request"] = {
        "task_family": "closed_book_application",
    }
    row = authority.make_training_row(source, "7" * 64)
    assert authority.GENERATED_REQUIRED_ROW_KEYS.issubset(row)
    assert row["schema"] == authority.ROW_SCHEMA
    assert row["eligible_for_training"] is True
    assert row["hard_negative"] is False
    assert row["messages"][-1] == {"role": "assistant", "content": source["answer"]}
    assert set(row["verification_receipts"]) == {
        "structural",
        "nli",
        "semantic_judge_pass_1",
        "semantic_judge_pass_2",
        "selection",
    }
    assert all(value["status"] == "passed" for value in row["verification_receipts"].values())
    assert row["rights_authorization"]["source_rights_status_preserved"] is True
    assert row["lineage"]["source_document_identity_sha256"] == "8" * 64


def test_blocked_scaffold_reads_no_semantic_rows_and_authorizes_nothing(monkeypatch):
    states = []
    for lane in ("primary", "fill"):
        for shard in range(4):
            states.append({
                "lane": lane,
                "gpu_shard": shard,
                "artifacts": {},
                "sealed_set_present": False,
            })
    monkeypatch.setattr(authority, "semantic_input_states", lambda: states)
    scaffold = authority.build_scaffold()
    assert scaffold["status"] == "blocked_pending_primary_and_fill_semantic_judges"
    assert scaffold["selection_may_run"] is False
    assert scaffold["selection_materialized"] is False
    assert scaffold["eligible_for_training"] is False
    assert scaffold["training_launch_authorized"] is False
    assert scaffold["selection"]["padding_allowed"] is False
    assert scaffold["selection"]["cross_source_borrowing_allowed"] is False
    assert scaffold["final_interface"]["row_schema"] == authority.ROW_SCHEMA
    assert scaffold["content_sha256_before_self_field"] == authority._self_address(scaffold)


def test_deficit_report_never_claims_or_emits_training_rows():
    pool = [
        candidate("selection-a5", tokens=5, source="source-a", category="category-a"),
        candidate("selection-b3", tokens=3, source="source-b", category="category-b"),
    ]
    selection = authority.solve_atomic_selection(pool, targets())
    report = authority.build_deficit_report(
        selection=selection,
        pool=pool,
        queue=[],
        input_receipt={"content_sha256": "6" * 64},
    )
    assert report["status"] == "exact_atomic_selection_unsolved_no_training_rows_emitted"
    assert report["selection"]["deficits"]["assistant_tokens"] == 2
    assert report["train_dataset_emitted"] is False
    assert report["eligible_for_training"] is False
    assert report["training_launch_authorized"] is False
    assert report["content_sha256_before_self_field"] == authority._self_address(report)


def test_mutating_build_refuses_stale_or_partial_canonical_authority_before_reads(
    monkeypatch, tmp_path
):
    train = tmp_path / "train.jsonl"
    report = tmp_path / "report.json"
    manifest = tmp_path / "manifest.json"
    train.write_text("stale canonical rows\n", encoding="utf-8")
    monkeypatch.setattr(authority, "TRAIN_OUTPUT", train)
    monkeypatch.setattr(authority, "REPORT_OUTPUT", report)
    monkeypatch.setattr(authority, "MANIFEST_OUTPUT", manifest)
    monkeypatch.setattr(
        authority,
        "build_scaffold",
        lambda: pytest.fail("semantic input state must not be read"),
    )
    with pytest.raises(
        RuntimeError, match="sealed generated-domain authority already exists"
    ):
        authority.build_global_authority(write=True)
