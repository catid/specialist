#!/usr/bin/env python3
"""Integrate distinct hogtie mobility-safety facts."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V306 = DATA / "manual_reviews/context_merit_audit_v306"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
ADD = DATA / "manual_reviews/hogtie_mobility_safety_additions_v18"
sys.path[:0] = [str(ROOT), str(V306), str(V290), str(ADD)]

import build_context_merit_audit_v306 as previous
import build_context_merit_audit_v290 as core
import build_hogtie_mobility_safety_additions_v18 as additions_builder

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v307.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v307.jsonl"
REPORT = OUT_DIR / "report_context_merit_v307.json"
ADDITIONS = additions_builder.OUTPUT
EXPECTED_ADDITIONS_SHA256 = "5470244d47d37746d8587d10371bcbee5b834d87df78a741fb0fd103a2b830bd"
EXPECTED_OUTPUT_SHA256 = "deff6ff8a1902d6b83d6d81351d7596811e863008ecbebc42914fdcd9ed64df3"
BASELINE_ROWS = 542
BASELINE_SHA256 = "ca43bcbd324a267afaef20aa69f1d2b8859b633335f6242133ac4e0233deccb4"
EXPECTED_CAPACITY = {
    "before": {"conflict_units": 249, "equipment_material": 22, "resources_general": 82, "safety_consent": 82, "technique": 63},
    "after": {"conflict_units": 252, "equipment_material": 22, "resources_general": 82, "safety_consent": 84, "technique": 64},
}
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
file_sha256, text_sha256 = previous.file_sha256, previous.text_sha256
read_jsonl, write_jsonl = previous.read_jsonl, previous.write_jsonl
conservative_capacity, portable = previous.conservative_capacity, previous.portable
PRIOR = (
    V306 / "context_merit_audit_v306.jsonl",
    V306 / "pending_curation_context_merit_v306.jsonl",
    V306 / "report_context_merit_v306.json",
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v306 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v307-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v306.jsonl"
    build_baseline(base, inputs / "v306.report.json")
    core.build_projection_with_inputs(out, report, (), (base, ADDITIONS))


def observe() -> dict:
    with tempfile.TemporaryDirectory(prefix=".v307-observe-", dir=OUT_DIR) as temp:
        d = Path(temp)
        base = d / "base.jsonl"
        build_baseline(base, d / "base.report.json")
        before = read_jsonl(base)
        out, report = d / "out.jsonl", d / "out.report.json"
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
                "review_pass": "distinct_document_hogtie_mobility_safety_additions",
                "reviewed_at": "2026-07-15",
                "reviewer": "codex-context-merit-audit-v307",
                "schema": "context-merit-audit-v307",
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
    if (observed["rows"], observed["eval"]) != (545, 612):
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
                "addition_artifact": {"path": portable(ADDITIONS), "rows": 3, "sha256": file_sha256(ADDITIONS)},
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
                "new_pending_curation": {"decisions": 0, "path": portable(CURATION), "sha256": file_sha256(CURATION)},
                "schema": "context-merit-audit-report-v307",
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
