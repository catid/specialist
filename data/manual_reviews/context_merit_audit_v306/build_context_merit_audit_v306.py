#!/usr/bin/env python3
"""Integrate distinct Rope365 learning and safety facts."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V305 = DATA / "manual_reviews/context_merit_audit_v305"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
ADD = DATA / "manual_reviews/rope365_learning_safety_additions_v17"
sys.path[:0] = [str(ROOT), str(V305), str(V290), str(ADD)]

import build_context_merit_audit_v305 as previous
import build_context_merit_audit_v290 as core
import build_rope365_learning_safety_additions_v17 as additions_builder

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v306.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v306.jsonl"
REPORT = OUT_DIR / "report_context_merit_v306.json"
ADDITIONS = additions_builder.OUTPUT
EXPECTED_ADDITIONS_SHA256 = "64f8573ae90d544bced28945af616f5299c5113ac597c1dcf42d0f886957226d"
EXPECTED_OUTPUT_SHA256 = "ca43bcbd324a267afaef20aa69f1d2b8859b633335f6242133ac4e0233deccb4"
BASELINE_ROWS = 539
BASELINE_SHA256 = "80eba8b89487052c10e046328211282e159ae334661b6776efba571d4e2824bc"
EXPECTED_CAPACITY = {
    "before": {
        "conflict_units": 246,
        "equipment_material": 22,
        "resources_general": 81,
        "safety_consent": 81,
        "technique": 62,
    },
    "after": {
        "conflict_units": 249,
        "equipment_material": 22,
        "resources_general": 82,
        "safety_consent": 82,
        "technique": 63,
    },
}
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
conservative_capacity = previous.conservative_capacity
portable = previous.portable
PRIOR = (
    V305 / "context_merit_audit_v305.jsonl",
    V305 / "pending_curation_context_merit_v305.jsonl",
    V305 / "report_context_merit_v305.json",
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v305 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v306-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v305.jsonl"
    build_baseline(base, inputs / "v305.report.json")
    core.build_projection_with_inputs(out, report, (), (base, ADDITIONS))


def observe() -> dict:
    with tempfile.TemporaryDirectory(prefix=".v306-observe-", dir=OUT_DIR) as temp:
        d = Path(temp)
        base = d / "base.jsonl"
        build_baseline(base, d / "base.report.json")
        before = read_jsonl(base)
        out = d / "out.jsonl"
        report = d / "out.report.json"
        datasets, reports = [], []
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
    additions_builder.main()
    if EXPECTED_ADDITIONS_SHA256 != "PENDING" and file_sha256(ADDITIONS) != EXPECTED_ADDITIONS_SHA256:
        raise ValueError("addition drift")
    audits = []
    for index, row in enumerate(read_jsonl(ADDITIONS), 1):
        path = ROOT / row["source_lineage"]["raw_document"]
        document = json.loads(path.read_text())
        if (row["url"], row["document_sha256"]) != (document["url"], document["document_sha256"]):
            raise ValueError("lineage")
        if not all(part in document["text"] for part in row["evidence"].splitlines()):
            raise ValueError("evidence")
        audits.append(
            {
                "audit_index": index,
                "decision": "add",
                "document_sha256": row["document_sha256"],
                "fact_id": row["fact_id"],
                "proposed_answer": row["answer"],
                "proposed_question": row["question"],
                "reason": row["paraphrase_rationale"],
                "reason_code": f"add_distinct_{row['topic']}_fact",
                "review_pass": "distinct_document_rope365_learning_safety_additions",
                "reviewed_at": "2026-07-15",
                "reviewer": "codex-context-merit-audit-v306",
                "schema": "context-merit-audit-v306",
                "source": row["source"],
                "source_document": portable(path),
                "source_document_file_sha256": file_sha256(path),
                "source_support": "manual_paraphrase",
                "support_evidence": row["evidence"],
                "support_evidence_sha256": text_sha256(row["evidence"]),
                "url": row["url"],
            }
        )
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, [])
    observed = observe()
    if not observed["dataset_equal"] or not observed["report_equal"]:
        raise ValueError("non-deterministic projection")
    if (observed["rows"], observed["eval"]) != (542, 612):
        raise ValueError(f"projection drift {observed}")
    if observed["before"] != EXPECTED_CAPACITY["before"] or observed["after"] != EXPECTED_CAPACITY["after"]:
        raise ValueError(f"capacity drift {observed}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["sha"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("output drift")
    delta = {k: observed["after"][k] - observed["before"][k] for k in observed["before"]}
    additions_report = json.loads(additions_builder.REPORT.read_text())
    REPORT.write_text(
        json.dumps(
            {
                "addition_artifact": {
                    "path": portable(ADDITIONS),
                    "rows": 3,
                    "sha256": file_sha256(ADDITIONS),
                },
                "audit": {
                    "by_decision": {"add": 3, "drop": 0, "edit": 0, "keep": 0},
                    "path": portable(AUDIT),
                    "rows": 3,
                    "sha256": file_sha256(AUDIT),
                },
                "conservative_capacity": {
                    "after": observed["after"],
                    "before": observed["before"],
                    "delta": delta,
                    "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
                },
                "excluded_source": additions_report["excluded_source"],
                "prior_checkpoint": {
                    "candidate": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256},
                    "artifacts": [{"path": portable(p), "sha256": file_sha256(p)} for p in PRIOR],
                },
                "isolated_build_projection": {
                    "automated_projection_runs": 2,
                    "new_additions_applied": 3,
                    "output_rows": observed["rows"],
                    "output_sha256": observed["sha"],
                    "repeat_dataset_byte_identical": observed["dataset_equal"],
                    "repeat_projection_report_byte_identical": observed["report_equal"],
                    "sealed_eval_fact_count_reported_by_tooling": observed["eval"],
                },
                "new_pending_curation": {
                    "decisions": 0,
                    "path": portable(CURATION),
                    "sha256": file_sha256(CURATION),
                },
                "schema": "context-merit-audit-report-v306",
                "sealed_evaluation_policy": {
                    "automated_collision_tool_reads_sealed_content": True,
                    "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
                    "manual_worker_opened_eval_or_heldout_content": False,
                    "manual_worker_received_eval_or_heldout_content": False,
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
