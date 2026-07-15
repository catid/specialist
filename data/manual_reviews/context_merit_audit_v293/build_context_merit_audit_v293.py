#!/usr/bin/env python3
"""Integrate the fourth distinct-document technique/equipment tranche."""
from __future__ import annotations
import hashlib, json, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]; DATA = ROOT / "data"
V292_DIR = DATA / "manual_reviews/context_merit_audit_v292"; ADDITION_DIR = DATA / "manual_reviews/technique_equipment_additions_v4"
sys.path[:0] = [str(ROOT), str(V292_DIR), str(ADDITION_DIR)]
import build_context_merit_audit_v292 as previous
import build_technique_equipment_additions_v4 as additions_builder

OUT_DIR = Path(__file__).resolve().parent; AUDIT = OUT_DIR / "context_merit_audit_v293.jsonl"; CURATION = OUT_DIR / "pending_curation_context_merit_v293.jsonl"; REPORT = OUT_DIR / "report_context_merit_v293.json"
ADDITIONS = additions_builder.OUTPUT; EXPECTED_ADDITIONS_SHA256 = "b929fde8bf96fe7dafb9dbc02f0827c0b20310c30a00683b398bc070481d7e9a"
EXPECTED_OUTPUT_SHA256 = "1913fbab4e3b0638683b44f02682cd9316548750046f13c70bd8abfe7327d960"; BASELINE_ROWS = 501; BASELINE_SHA256 = "3a1b3ca8e9ada0be0a8758fd9b9cc4ce01d17164746d587dcb028a0e916a7a17"
EXPECTED_CAPACITY = {"before": {"conflict_units": 208, "equipment_material": 17, "resources_general": 74, "safety_consent": 72, "technique": 45}, "after": {"conflict_units": 211, "equipment_material": 18, "resources_general": 74, "safety_consent": 72, "technique": 47}}
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST; file_sha256 = previous.file_sha256; text_sha256 = previous.text_sha256; read_jsonl = previous.read_jsonl; write_jsonl = previous.write_jsonl; conservative_capacity = previous.conservative_capacity; portable = previous.portable


def build_baseline(output, report):
    previous.build_projection(output, report)
    if (len(read_jsonl(output)), file_sha256(output)) != (BASELINE_ROWS, BASELINE_SHA256): raise ValueError("v292 baseline drift")


def build_projection(output, report):
    d = Path(output).parent / f".{Path(output).name}.v293-input"; d.mkdir(parents=True, exist_ok=True); baseline = d / "v292.jsonl"
    build_baseline(baseline, d / "v292.report.json"); previous.previous.previous.build_projection_with_inputs(output, report, (), (baseline, ADDITIONS))


def prior_decision_artifacts():
    return tuple(DATA / "manual_reviews" / f"context_merit_audit_v{v}" / name for v in range(1, 293) for name in (f"context_merit_audit_v{v}.jsonl", f"pending_curation_context_merit_v{v}.jsonl", f"report_context_merit_v{v}.json"))


def observe():
    with tempfile.TemporaryDirectory(prefix=".v293-observation-", dir=OUT_DIR) as temp:
        d = Path(temp); baseline = d / "baseline.jsonl"; build_baseline(baseline, d / "baseline.report.json"); base_rows = read_jsonl(baseline)
        output, report = d / "projection.jsonl", d / "projection.report.json"; datasets, reports = [], []
        for _ in (1, 2): build_projection(output, report); datasets.append(output.read_bytes()); reports.append(report.read_bytes())
        rows = read_jsonl(output)
        return {"baseline_capacity": conservative_capacity(base_rows), "dataset_equal": datasets[0] == datasets[1], "dataset_sha256": hashlib.sha256(datasets[0]).hexdigest(), "eval_fact_count": json.loads(reports[0])["eval_fact_count"], "output_capacity": conservative_capacity(rows), "report_equal": reports[0] == reports[1], "rows": len(rows)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True); additions_builder.main()
    if file_sha256(ADDITIONS) != EXPECTED_ADDITIONS_SHA256: raise ValueError("addition artifact drift")
    audits = []
    for i, row in enumerate(read_jsonl(ADDITIONS), 1):
        source_path = ROOT / row["source_lineage"]["raw_document"]; document = json.loads(source_path.read_text())
        if (row["url"], row["document_sha256"]) != (document["url"], document["document_sha256"]): raise ValueError("source lineage drift")
        if not all(line in document["text"] for line in row["evidence"].splitlines()): raise ValueError("evidence drift")
        audits.append({"audit_index": i, "decision": "add", "document_sha256": row["document_sha256"], "fact_id": row["fact_id"], "proposed_answer": row["answer"], "proposed_question": row["question"], "reason": row["paraphrase_rationale"], "reason_code": f"add_distinct_{row['topic']}_fact", "review_pass": "distinct_document_technique_equipment_additions", "reviewed_at": "2026-07-15", "reviewer": "codex-context-merit-audit-v293", "schema": "context-merit-audit-v293", "source": row["source"], "source_document": portable(source_path), "source_document_file_sha256": file_sha256(source_path), "source_support": "manual_paraphrase", "support_evidence": row["evidence"], "support_evidence_sha256": text_sha256(row["evidence"]), "url": row["url"]})
    write_jsonl(AUDIT, audits); write_jsonl(CURATION, []); observed = observe()
    if not observed["dataset_equal"] or not observed["report_equal"]: raise ValueError("v293 projection nondeterministic")
    if (observed["rows"], observed["eval_fact_count"]) != (504, 612): raise ValueError("row/eval aggregate drift")
    if observed["baseline_capacity"] != EXPECTED_CAPACITY["before"]: raise ValueError("baseline capacity drift")
    if observed["output_capacity"] != EXPECTED_CAPACITY["after"]: raise ValueError(f"output capacity drift: {observed['output_capacity']}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256: raise ValueError("output hash drift")
    report = {"addition_artifact": {"path": portable(ADDITIONS), "rows": 3, "sha256": file_sha256(ADDITIONS)}, "audit": {"by_decision": {"add": 3, "drop": 0, "edit": 0, "keep": 0}, "path": portable(AUDIT), "rows": 3, "sha256": file_sha256(AUDIT)}, "conservative_capacity": {"after": observed["output_capacity"], "before": observed["baseline_capacity"], "delta": {key: observed["output_capacity"][key] - observed["baseline_capacity"][key] for key in observed["baseline_capacity"]}, "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster"}, "frozen_prior_decision_artifacts": [{"path": portable(path), "sha256": file_sha256(path)} for path in prior_decision_artifacts()], "isolated_build_projection": {"automated_projection_runs": 2, "new_additions_applied": 3, "output_rows": observed["rows"], "output_sha256": observed["dataset_sha256"], "repeat_dataset_byte_identical": observed["dataset_equal"], "repeat_projection_report_byte_identical": observed["report_equal"], "sealed_eval_fact_count_reported_by_tooling": observed["eval_fact_count"]}, "new_pending_curation": {"decisions": 0, "path": portable(CURATION), "sha256": file_sha256(CURATION)}, "projected_baseline": {"description": "complete train-only candidate through v292", "rows": BASELINE_ROWS, "sha256": BASELINE_SHA256}, "schema": "context-merit-audit-report-v293", "sealed_evaluation_policy": {"automated_collision_tool_reads_sealed_content": True, "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only", "manual_worker_opened_eval_or_heldout_content": False, "manual_worker_received_eval_or_heldout_content": False}}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__": main()
