#!/usr/bin/env python3
"""Complete three source-grounded safety and consent answers in training data."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V435 = DATA / "manual_reviews/context_merit_audit_v435"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V435), str(V290)]

import build_context_merit_audit_v435 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v436.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v436.jsonl"
REPORT = OUT_DIR / "report_context_merit_v436.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "25bc23bf80ed9cd027ead487e41ab8c9364b0b7db9c6de236f54b5128d6b87c8"
EXPECTED_OUTPUT_SHA256 = "9f1b6c4cf21bef4ecbefb33cf58063dcec1ce44f52efa9cfb501a3aa2e9eb7f8"
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
    V435 / "context_merit_audit_v435.jsonl",
    V435 / "pending_curation_context_merit_v435.jsonl",
    V435 / "report_context_merit_v435.json",
)

SPECS = (
    {
        "fact_id": "fact-b166dee2d7c7dd21994e",
        "active_index": 193,
        "expected_question": "What corrective steps does WykD’s article on self-awareness and responsibility in rope-bondage injuries recommend when a rigger causes repeated injuries that others are not causing?",
        "expected_answer": "They should reduce how much they rig, examine the common themes in the injuries, and work hard to prevent them from recurring.",
        "expected_evidence_sha256": "022c6cae251e9940532d1bf852061ced7c8db0c5abb9f8bbdfc40a7113bc4730",
        "question": "What corrective steps does WykD’s article on self-awareness and responsibility in rope-bondage injuries recommend when a rigger causes repeated injuries that others are not causing?",
        "answer": "WykD says a rigger causing repeated injuries that others are not causing should reduce how much they rig, examine common themes in the injuries, and work hard to prevent recurrence.",
        "source_document": "data/raw/wykd_19d6a26116e26c70.json",
        "reason_code": "resolve_repeated_injury_corrective_actor",
        "reason": "The replacement resolves the ambiguous actor and attributes the corrective guidance while preserving reduced tying, common-theme review, and recurrence prevention.",
    },
    {
        "fact_id": "fact-7ad4bbd614a6ba3102ec",
        "active_index": 327,
        "expected_question": "What should partners discuss before exploring exposed rope poses, and why?",
        "expected_answer": "Discuss the poses in advance because they can make a person feel especially vulnerable; whether sexual activity is included remains a separate choice.",
        "expected_evidence_sha256": "6827220c0862179ed41a4ec84faa0eda837663d8c005bb98cc0f992fe9898d8f",
        "question": "What should partners discuss before exploring exposed rope poses, and why?",
        "answer": "Rope365 recommends that partners discuss exposed poses in advance because the poses can make someone feel especially vulnerable; whether sexual activity is included remains a separate choice.",
        "source_document": "data/raw/rope365_1a9f4f2fef392a9c.json",
        "reason_code": "complete_exposed_pose_discussion_subject",
        "reason": "The replacement names the partners as actors and attributes the guidance while preserving vulnerability and the separate choice about sexual activity.",
    },
    {
        "fact_id": "fact-a24eab095e1701d6cb32",
        "active_index": 504,
        "expected_question": "Why does Rope365 suggest that new partners discuss their experience levels and past experiences?",
        "expected_answer": "It lets new partners adapt their conversation and avoid pushing each other too quickly.",
        "expected_evidence_sha256": "5a1c4ed1167ec11309cbc9144a8d7c164e7d53ec470d7c4e638b7df83943c9eb",
        "question": "Why does Rope365 suggest that new partners discuss their experience levels and past experiences?",
        "answer": "Rope365 says this discussion helps new partners adapt their conversation and avoid pushing each other too quickly.",
        "source_document": "data/raw/rope365_c73bc6fb66977a2d.json",
        "reason_code": "resolve_new_partner_discussion_subject",
        "reason": "The replacement resolves the ambiguous pronoun and attributes the rationale while preserving adaptive dialogue and slower pacing.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v435 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v436-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v435.jsonl"
    build_baseline(base, inputs / "v435.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v436-observe-", dir=OUT_DIR) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v436-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v435.jsonl"
        build_baseline(base, d / "v435.report.json")
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
                "reviewer": "codex-context-merit-audit-v436",
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
                "review_pass": "bounded_safety_consent_actionability_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v436",
                "schema": "context-merit-audit-v436",
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
                "schema": "context-merit-audit-report-v436",
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
