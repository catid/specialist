#!/usr/bin/env python3
"""Integrate three distinct-document technique additions after v290."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V290_DIR = DATA / "manual_reviews/context_merit_audit_v290"
ADDITION_DIR = DATA / "manual_reviews/technique_additions_v2"
sys.path[:0] = [str(ROOT), str(V290_DIR), str(ADDITION_DIR)]
import build_context_merit_audit_v290 as previous
import build_technique_additions_v2 as additions_builder

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v291.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v291.jsonl"
REPORT = OUT_DIR / "report_context_merit_v291.json"
ADDITIONS = additions_builder.OUTPUT
EXPECTED_ADDITIONS_SHA256 = "3125b51e446ee47b708d155b7fbb4fa0640476b422a29d897429e2e0525917d7"
EXPECTED_OUTPUT_SHA256 = "ed516dffd88a6300945ead3b83062ca667d1b18977a6c96a8bcb6724880830fa"
BASELINE_ROWS = 495
BASELINE_SHA256 = "1afd8517320e5465ad3f52d915bc9391b19ca56a32a2db3ffcd713f88442acf1"
EXPECTED_CAPACITY = {
    "before": {"conflict_units": 202, "equipment_material": 17, "resources_general": 71, "safety_consent": 72, "technique": 42},
    "after": {"conflict_units": 205, "equipment_material": 17, "resources_general": 71, "safety_consent": 72, "technique": 45},
}
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
conservative_capacity = previous.conservative_capacity
portable = previous.portable


def build_baseline(output: Path, report: Path) -> None:
    previous.build_projection(output, report)
    if (len(read_jsonl(output)), file_sha256(output)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v290 baseline drift")


def build_projection(output: Path, report: Path) -> None:
    replay_dir = output.parent / f".{output.name}.v291-input"
    replay_dir.mkdir(parents=True, exist_ok=True)
    baseline = replay_dir / "v290.jsonl"
    build_baseline(baseline, replay_dir / "v290.report.json")
    previous.build_projection_with_inputs(output, report, (), (baseline, ADDITIONS))


def prior_decision_artifacts() -> tuple[Path, ...]:
    return tuple(
        DATA / "manual_reviews" / f"context_merit_audit_v{version}" / name
        for version in range(1, 291)
        for name in (
            f"context_merit_audit_v{version}.jsonl",
            f"pending_curation_context_merit_v{version}.jsonl",
            f"report_context_merit_v{version}.json",
        )
    )


def observe() -> dict:
    with tempfile.TemporaryDirectory(prefix=".v291-observation-", dir=OUT_DIR) as temp:
        directory = Path(temp)
        baseline = directory / "baseline.jsonl"
        build_baseline(baseline, directory / "baseline.report.json")
        baseline_rows = read_jsonl(baseline)
        output = directory / "projection.jsonl"
        report = directory / "projection.report.json"
        datasets, reports = [], []
        for _ in (1, 2):
            build_projection(output, report)
            datasets.append(output.read_bytes())
            reports.append(report.read_bytes())
        rows = read_jsonl(output)
        return {
            "baseline_capacity": conservative_capacity(baseline_rows),
            "dataset_equal": datasets[0] == datasets[1],
            "dataset_sha256": hashlib.sha256(datasets[0]).hexdigest(),
            "eval_fact_count": json.loads(reports[0])["eval_fact_count"],
            "output_capacity": conservative_capacity(rows),
            "report_equal": reports[0] == reports[1],
            "rows": len(rows),
        }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    additions_builder.main()
    if file_sha256(ADDITIONS) != EXPECTED_ADDITIONS_SHA256:
        raise ValueError("addition artifact drift")
    additions = read_jsonl(ADDITIONS)
    audits = []
    for index, row in enumerate(additions, 1):
        source_path = ROOT / row["source_lineage"]["raw_document"]
        document = json.loads(source_path.read_text())
        if (row["url"], row["document_sha256"]) != (document["url"], document["document_sha256"]):
            raise ValueError("addition source lineage drift")
        if not all(line in document["text"] for line in row["evidence"].splitlines()):
            raise ValueError("addition evidence drift")
        audits.append({
            "audit_index": index, "decision": "add", "document_sha256": row["document_sha256"],
            "fact_id": row["fact_id"], "proposed_answer": row["answer"], "proposed_question": row["question"],
            "reason": row["paraphrase_rationale"], "reason_code": f"add_distinct_{row['topic']}_fact",
            "review_pass": "distinct_document_technique_additions", "reviewed_at": "2026-07-15",
            "reviewer": "codex-context-merit-audit-v291", "schema": "context-merit-audit-v291",
            "source": row["source"], "source_document": portable(source_path),
            "source_document_file_sha256": file_sha256(source_path), "source_support": "manual_paraphrase",
            "support_evidence": row["evidence"], "support_evidence_sha256": text_sha256(row["evidence"]),
            "url": row["url"],
        })
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, [])
    observed = observe()
    if not observed["dataset_equal"] or not observed["report_equal"]:
        raise ValueError("v291 projection is nondeterministic")
    if (observed["rows"], observed["eval_fact_count"]) != (498, 612):
        raise ValueError("v291 row/eval aggregate drift")
    if observed["baseline_capacity"] != EXPECTED_CAPACITY["before"]:
        raise ValueError("v291 baseline capacity drift")
    if observed["output_capacity"] != EXPECTED_CAPACITY["after"]:
        raise ValueError(f"v291 output capacity drift: {observed['output_capacity']}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v291 output hash drift")
    report = {
        "addition_artifact": {"path": portable(ADDITIONS), "rows": 3, "sha256": file_sha256(ADDITIONS)},
        "audit": {"by_decision": {"add": 3, "drop": 0, "edit": 0, "keep": 0}, "path": portable(AUDIT), "rows": 3, "sha256": file_sha256(AUDIT)},
        "conservative_capacity": {
            "after": observed["output_capacity"], "before": observed["baseline_capacity"],
            "delta": {key: observed["output_capacity"][key] - observed["baseline_capacity"][key] for key in observed["baseline_capacity"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "frozen_prior_decision_artifacts": [{"path": portable(path), "sha256": file_sha256(path)} for path in prior_decision_artifacts()],
        "isolated_build_projection": {
            "automated_projection_runs": 2, "new_additions_applied": 3, "output_rows": observed["rows"],
            "output_sha256": observed["dataset_sha256"], "repeat_dataset_byte_identical": observed["dataset_equal"],
            "repeat_projection_report_byte_identical": observed["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observed["eval_fact_count"],
        },
        "new_pending_curation": {"decisions": 0, "path": portable(CURATION), "sha256": file_sha256(CURATION)},
        "projected_baseline": {"description": "complete train-only candidate through v290", "rows": BASELINE_ROWS, "sha256": BASELINE_SHA256},
        "schema": "context-merit-audit-report-v291",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
