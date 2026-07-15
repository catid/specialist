#!/usr/bin/env python3
"""Complete three context, consent, and support train-only answers."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V409 = DATA / "manual_reviews/context_merit_audit_v409"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V409), str(V290)]

import build_context_merit_audit_v409 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v410.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v410.jsonl"
REPORT = OUT_DIR / "report_context_merit_v410.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "7bc66895c8709997f5baf655442993e6bb764264935517ab40d586253bd2be69"
EXPECTED_OUTPUT_SHA256 = "c8a12fea954754ffa70c7b2f660fc4aed04d08022acc14a0bdaa75fdfad16378"
EXPECTED_CAPACITY_BEFORE = {"conflict_units": 259, "equipment_material": 23, "resources_general": 84, "safety_consent": 81, "technique": 71}
EXPECTED_CAPACITY_AFTER = {"conflict_units": 259, "equipment_material": 23, "resources_general": 84, "safety_consent": 81, "technique": 71}
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
conservative_capacity = previous.conservative_capacity
portable = previous.portable
PRIOR = (
    V409 / "context_merit_audit_v409.jsonl",
    V409 / "pending_curation_context_merit_v409.jsonl",
    V409 / "report_context_merit_v409.json",
)

SPECS = (
    {
        "fact_id": "fact-064c09ce8203c1ec350e",
        "active_index": 191,
        "expected_question": "What context question does Rope365 suggest asking when analyzing an old bondage image?",
        "expected_answer": "What was the context of the picture (ex: photoshoot, play, performance)?",
        "question": "What context question does Rope365 suggest asking when analyzing an old bondage image?",
        "answer": "Rope365 suggests asking what the picture's context was, such as whether it was a photoshoot, play, or performance.",
        "reason_code": "complete_image_context_question",
        "reason": "The replacement turns the source question into a complete attributed answer while preserving its three example contexts.",
    },
    {
        "fact_id": "fact-b622f79d2908922fd73a",
        "active_index": 209,
        "expected_question": "What does Anatomie Studio recommend if someone doubts a partner's verbal or physical enthusiasm?",
        "expected_answer": "Check in, and do not proceed if doubt remains.",
        "question": "What does Anatomie Studio recommend if someone doubts a partner's verbal or physical enthusiasm?",
        "answer": "Anatomie Studio recommends checking in and not proceeding if doubt remains.",
        "reason_code": "complete_doubt_check_in_guidance",
        "reason": "The replacement turns the two source-supported imperatives into a complete attributed answer without weakening either condition.",
    },
    {
        "fact_id": "fact-1a70c3206523e0ef70ac",
        "active_index": 286,
        "expected_question": "What open question does Rope365 recommend for supporting a partner after a scene is paused or stopped?",
        "expected_answer": "Ask the partner, “What do you need right now?”",
        "question": "What open question does Rope365 recommend for supporting a partner after a scene is paused or stopped?",
        "answer": "Rope365 recommends asking the partner, “What do you need right now?”",
        "reason_code": "complete_post_stop_support_question",
        "reason": "The replacement turns the source-supported imperative into a complete attributed answer while retaining the exact open question.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v409 drift")


def build_projection(out: Path, report: Path) -> None:
    d = out.parent / f".{out.name}.v410-input"
    d.mkdir(parents=True, exist_ok=True)
    base = d / "v409.jsonl"
    build_baseline(base, d / "v409.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v410-observe-", dir=OUT_DIR) as t:
        d = Path(t)
        out = d / "out.jsonl"
        report = d / "out.report.json"
        datasets = []
        reports = []
        for _ in (1, 2):
            build_projection(out, report)
            datasets.append(out.read_bytes())
            reports.append(report.read_bytes())
        rows = read_jsonl(out)
        return {
            "rows": len(rows),
            "sha": hashlib.sha256(datasets[0]).hexdigest(),
            "eval": json.loads(reports[0])["eval_fact_count"],
            "de": datasets[0] == datasets[1],
            "re": reports[0] == reports[1],
            "before": conservative_capacity(before),
            "after": conservative_capacity(rows),
        }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v410-base-", dir=OUT_DIR) as t:
        d = Path(t)
        base = d / "v409.jsonl"
        build_baseline(base, d / "v409.report.json")
        before = read_jsonl(base)
    by_fact = {r["fact_id"]: (i, r) for i, r in enumerate(before, 1)}
    audits = []
    curations = []
    for audit_index, spec in enumerate(SPECS, 1):
        index, active = by_fact[spec["fact_id"]]
        if index != spec["active_index"] or (active["question"], active["answer"]) != (spec["expected_question"], spec["expected_answer"]):
            raise ValueError(f"candidate drift {spec['fact_id']}")
        evidence = active.get("evidence")
        if not evidence:
            raise ValueError("missing evidence")
        curations.append({
            "action": "edit", "answer": spec["answer"], "document_sha256": active["document_sha256"],
            "evidence": evidence, "evidence_url": active["url"], "expected_answer": active["answer"],
            "expected_question": active["question"], "fact_id": active["fact_id"],
            "paraphrase_rationale": spec["reason"], "question": spec["question"], "reason": spec["reason"],
            "reason_code": spec["reason_code"], "reviewed_at": "2026-07-15",
            "reviewer": "codex-context-merit-audit-v410", "source_lineage": active["source_lineage"],
            "support_type": "manual_paraphrase",
        })
        audits.append({
            "active_answer": active["answer"], "active_index": index, "active_question": active["question"],
            "audit_index": audit_index, "decision": "edit", "document_sha256": active["document_sha256"],
            "edited_answer": spec["answer"], "edited_question": spec["question"], "fact_id": active["fact_id"],
            "paraphrase_rationale": spec["reason"],
            "projection_lineage": {"active_index": index, "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256},
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_pass": "context_consent_support_answer_completeness_train_only_cleanup",
            "reviewed_at": "2026-07-15", "reviewer": "codex-context-merit-audit-v410",
            "schema": "context-merit-audit-v410", "source": active["source"],
            "source_support": "manual_source_and_dataset_context_review", "support_evidence": evidence,
            "support_evidence_sha256": text_sha256(evidence), "url": active["url"],
        })
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observed = observe(before)
    if not observed["de"] or not observed["re"] or (observed["rows"], observed["eval"]) != (531, 612) or observed["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"projection drift {observed}")
    if EXPECTED_CAPACITY_AFTER != "PENDING" and observed["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"capacity drift {observed}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["sha"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("output drift")
    REPORT.write_text(json.dumps({
        "audit": {"by_decision": {"drop": 0, "edit": 3, "keep": 0}, "path": portable(AUDIT), "rows": 3, "sha256": file_sha256(AUDIT)},
        "conservative_capacity": {"after": observed["after"], "before": observed["before"], "delta": {k: observed["after"][k] - observed["before"][k] for k in observed["before"]}, "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"},
        "prior_checkpoint": {"candidate": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256}, "artifacts": [{"path": portable(p), "sha256": file_sha256(p)} for p in PRIOR]},
        "isolated_build_projection": {"automated_projection_runs": 2, "new_additions_applied": 0, "output_rows": observed["rows"], "output_sha256": observed["sha"], "repeat_dataset_byte_identical": observed["de"], "repeat_projection_report_byte_identical": observed["re"], "sealed_eval_fact_count_reported_by_tooling": observed["eval"]},
        "new_pending_curation": {"decisions": 3, "path": portable(CURATION), "sha256": file_sha256(CURATION)},
        "schema": "context-merit-audit-report-v410",
        "sealed_evaluation_policy": {"automated_collision_tool_reads_sealed_content": True, "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only", "manual_worker_opened_eval_or_heldout_content": False, "manual_worker_received_eval_or_heldout_content": False},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
