#!/usr/bin/env python3
"""Make three learning, confidentiality, and partner-first answers standalone."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V439 = DATA / "manual_reviews/context_merit_audit_v439"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V439), str(V290)]

import build_context_merit_audit_v439 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v440.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v440.jsonl"
REPORT = OUT_DIR / "report_context_merit_v440.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "8fd31994bf89dfa01dd3f1baae3b93ffa273cba9c325417ff11275eac5e0f8ad"
EXPECTED_OUTPUT_SHA256 = "8988b14443dd3ab51be98be24b1aec6bc82ec4233241f3422cb0887c842af078"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 259,
    "equipment_material": 23,
    "resources_general": 84,
    "safety_consent": 81,
    "technique": 71,
}
EXPECTED_CAPACITY_AFTER = dict(EXPECTED_CAPACITY_BEFORE)

RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
conservative_capacity = previous.conservative_capacity
portable = previous.portable
PRIOR = (
    V439 / "context_merit_audit_v439.jsonl",
    V439 / "pending_curation_context_merit_v439.jsonl",
    V439 / "report_context_merit_v439.json",
)

SPECS = (
    {
        "fact_id": "fact-4d3b9b8593faad48d001",
        "active_index": 84,
        "expected_question": "How does Rope365 recommend combining in-person and online rope learning?",
        "expected_answer": "Use in-person learning for tactile subtleties when available, use books and videos to explore and get started, and evaluate every resource critically because rope knowledge continues to evolve.",
        "expected_evidence_sha256": "0c771de90bbcf13d32528727a9fea0fea01c34b4a2184e45a8b793d8bddc5ef8",
        "question": "How does Rope365 recommend combining in-person and online rope learning?",
        "answer": "Rope365 recommends learning tactile subtleties in person when possible, using books and videos to explore or begin without a local community, and evaluating every resource critically as rope knowledge evolves.",
        "source_document": "data/raw/rope_resources_v1/rope365__f9753746348ca9cc2f11.json",
        "reason_code": "complete_rope365_mixed_learning_guidance",
        "reason": "The replacement supplies a standalone source subject and retains the evidence's distinct roles for tactile in-person learning, remote exploration, critical evaluation, and evolving safety knowledge.",
    },
    {
        "fact_id": "fact-12330efcb361031926b5",
        "active_index": 183,
        "expected_question": "What confidentiality rule do Tethered Together’s collected 2025 rules give for recognizing another attendee?",
        "expected_answer": "Do not share their presence or information without explicit permission; outing an attendee is treated as a serious offense.",
        "expected_evidence_sha256": "8a284af909d82ed4fd9f1b945cd5a0ccd7c319eacd5f3c47687cb83274c32957",
        "question": "What confidentiality rule do Tethered Together’s collected 2025 rules give for recognizing another attendee?",
        "answer": "Tethered Together’s 2025 rules say not to disclose another attendee’s presence or information without explicit permission and warn that outing someone is a serious offense that can lead to expulsion from current and future events.",
        "source_document": "data/raw/rope_resources_v1/tethered_together__cfcc0aab47acc802dd94.json",
        "reason_code": "complete_attendee_confidentiality_rule_and_consequence",
        "reason": "The replacement attributes the rule, resolves the pronoun, and includes the evidence's stated current-and-future-event expulsion consequence without broadening the policy.",
    },
    {
        "fact_id": "fact-25186f8cf3a77a31ca65",
        "active_index": 287,
        "expected_question": "What partner-first principle does the beginner article recommend before studying rope technique?",
        "expected_answer": "Before you study the rope, study your partner. Learn how to read them and how to create experiences for them. Learn to communicate with your body, your hands, your body position, your timing, and your intention.",
        "expected_evidence_sha256": "e73b67b4208e42f8d76486fa7669ccc31b20e97a81ba1b93dcaa31df3ec0bf03",
        "question": "What partner-first principle does the beginner article recommend before studying rope technique?",
        "answer": "The article says to study the partner before the rope: learn to read their responses, create experiences for them, and communicate through touch, body position, timing, and intention.",
        "source_document": "data/raw/kinbakutoday_63fcb1570feac169.json",
        "reason_code": "paraphrase_partner_first_principle_standalone",
        "reason": "The replacement converts an unattributed imperative quotation into a concise standalone paraphrase while retaining partner reading, experience creation, and embodied communication.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v439 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v440-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v439.jsonl"
    build_baseline(base, inputs / "v439.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v440-observe-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        out = d / "out.jsonl"
        report = d / "out.report.json"
        datasets: list[bytes] = []
        reports: list[bytes] = []
        for _ in (1, 2):
            build_projection(out, report)
            datasets.append(out.read_bytes())
            reports.append(report.read_bytes())
        rows = read_jsonl(out)
        return {
            "rows": len(rows),
            "sha": hashlib.sha256(datasets[0]).hexdigest(),
            "eval": json.loads(reports[0])["eval_fact_count"],
            "dataset_equal": datasets[0] == datasets[1],
            "report_equal": reports[0] == reports[1],
            "before": conservative_capacity(before),
            "after": conservative_capacity(rows),
        }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v440-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v439.jsonl"
        build_baseline(base, d / "v439.report.json")
        before = read_jsonl(base)

    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}
    audits: list[dict] = []
    curations: list[dict] = []
    for audit_index, spec in enumerate(SPECS, 1):
        index, active = by_fact[spec["fact_id"]]
        if index != spec["active_index"]:
            raise ValueError(f"candidate index drift {spec['fact_id']}")
        if (active["question"], active["answer"]) != (
            spec["expected_question"],
            spec["expected_answer"],
        ):
            raise ValueError(f"candidate semantic drift {spec['fact_id']}")
        evidence = active.get("evidence")
        if not evidence or text_sha256(evidence) != spec["expected_evidence_sha256"]:
            raise ValueError(f"candidate evidence drift {spec['fact_id']}")
        source_document = ROOT / spec["source_document"]
        source = json.loads(source_document.read_text())
        if source.get("url") != active["url"]:
            raise ValueError(f"source URL drift {spec['fact_id']}")
        if evidence not in source["text"]:
            raise ValueError(f"source evidence drift {spec['fact_id']}")

        curations.append(
            {
                "action": "edit",
                "answer": spec["answer"],
                "document_sha256": active["document_sha256"],
                "evidence": evidence,
                "evidence_url": active["url"],
                "expected_answer": active["answer"],
                "expected_question": active["question"],
                "fact_id": active["fact_id"],
                "paraphrase_rationale": spec["reason"],
                "question": spec["question"],
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v440",
                "source_lineage": active["source_lineage"],
                "support_type": "manual_paraphrase",
            }
        )
        audits.append(
            {
                "active_answer": active["answer"],
                "active_index": index,
                "active_question": active["question"],
                "audit_index": audit_index,
                "decision": "edit",
                "document_sha256": active["document_sha256"],
                "edited_answer": spec["answer"],
                "edited_question": spec["question"],
                "fact_id": active["fact_id"],
                "paraphrase_rationale": spec["reason"],
                "projection_lineage": {
                    "active_index": index,
                    "baseline_rows": BASELINE_ROWS,
                    "baseline_sha256": BASELINE_SHA256,
                },
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "review_pass": "bounded_standalone_attribution_and_context_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v440",
                "schema": "context-merit-audit-v440",
                "source": active["source"],
                "source_document": spec["source_document"],
                "source_document_file_sha256": file_sha256(source_document),
                "source_snapshot_chars_manually_reviewed": len(source["text"]),
                "source_support": "manual_full_snapshot_and_dataset_context_review",
                "support_evidence": evidence,
                "support_evidence_sha256": text_sha256(evidence),
                "url": active["url"],
            }
        )

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection {observation}")
    if (observation["rows"], observation["eval"]) != (531, 612):
        raise ValueError(f"projection count drift {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift {observation}")
    if observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"output drift {observation['sha']}")

    REPORT.write_text(
        json.dumps(
            {
                "audit": {
                    "bounded_manual_selection_rows": 3,
                    "by_decision": {"drop": 0, "edit": 3, "keep": 0},
                    "path": portable(AUDIT),
                    "rows": 3,
                    "sha256": file_sha256(AUDIT),
                },
                "conservative_capacity": {
                    "after": observation["after"],
                    "before": observation["before"],
                    "delta": {
                        key: observation["after"][key] - observation["before"][key]
                        for key in observation["before"]
                    },
                    "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
                },
                "prior_checkpoint": {
                    "candidate": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256},
                    "artifacts": [
                        {"path": portable(path), "sha256": file_sha256(path)} for path in PRIOR
                    ],
                },
                "isolated_build_projection": {
                    "automated_projection_runs": 2,
                    "new_additions_applied": 0,
                    "output_rows": observation["rows"],
                    "output_sha256": observation["sha"],
                    "repeat_dataset_byte_identical": observation["dataset_equal"],
                    "repeat_projection_report_byte_identical": observation["report_equal"],
                    "sealed_eval_fact_count_reported_by_tooling": observation["eval"],
                },
                "new_pending_curation": {
                    "decisions": 3,
                    "path": portable(CURATION),
                    "sha256": file_sha256(CURATION),
                },
                "schema": "context-merit-audit-report-v440",
                "sealed_evaluation_policy": {
                    "automated_collision_tool_reads_sealed_content": True,
                    "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
                    "manual_worker_opened_eval_or_heldout_content": False,
                    "manual_worker_received_eval_or_heldout_content": False,
                    "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
                    "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
                },
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
