#!/usr/bin/env python3
"""Integrate the twelfth distinct-document equipment/safety tranche."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V300_DIR = DATA / "manual_reviews/context_merit_audit_v300"
V290_DIR = DATA / "manual_reviews/context_merit_audit_v290"
ADDITION_DIR = DATA / "manual_reviews/equipment_safety_additions_v12"
sys.path[:0] = [str(ROOT), str(V300_DIR), str(V290_DIR), str(ADDITION_DIR)]
import build_context_merit_audit_v300 as previous
import build_context_merit_audit_v290 as core
import build_equipment_safety_additions_v12 as additions_builder

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v301.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v301.jsonl"
REPORT = OUT_DIR / "report_context_merit_v301.json"
ADDITIONS = additions_builder.OUTPUT
EXPECTED_ADDITIONS_SHA256 = "7dddff797d1e3c3bc8480b34b75b319d021190b22f7c4c8037ffb421d78f351e"
EXPECTED_OUTPUT_SHA256 = "cbba5461f1f60bd1770ef6a776be9747d20fd58745ff20263bc4983f31dddbf7"
BASELINE_ROWS = 525
BASELINE_SHA256 = "b0315512acb0af95ff5fd0f0af835b21fcda293ff87c3cf9329a0ca1e44493a0"
EXPECTED_CAPACITY = {
    "before": {"conflict_units": 232, "equipment_material": 21, "resources_general": 79, "safety_consent": 77, "technique": 55},
    "after": {"conflict_units": 235, "equipment_material": 22, "resources_general": 79, "safety_consent": 79, "technique": 55},
}
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
conservative_capacity = previous.conservative_capacity
portable = previous.portable
PRIOR_ARTIFACTS = (
    V300_DIR / "context_merit_audit_v300.jsonl",
    V300_DIR / "pending_curation_context_merit_v300.jsonl",
    V300_DIR / "report_context_merit_v300.json",
)


def build_baseline(output: Path, report: Path) -> None:
    previous.build_projection(output, report)
    if (len(read_jsonl(output)), file_sha256(output)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v300 baseline drift")


def build_projection(output: Path, report: Path) -> None:
    directory = output.parent / f".{output.name}.v301-input"
    directory.mkdir(parents=True, exist_ok=True)
    baseline = directory / "v300.jsonl"
    build_baseline(baseline, directory / "v300.report.json")
    core.build_projection_with_inputs(output, report, (), (baseline, ADDITIONS))


def observe() -> dict:
    with tempfile.TemporaryDirectory(prefix=".v301-observation-", dir=OUT_DIR) as temp:
        directory = Path(temp)
        baseline = directory / "baseline.jsonl"
        build_baseline(baseline, directory / "baseline.report.json")
        baseline_rows = read_jsonl(baseline)
        output, report = directory / "projection.jsonl", directory / "projection.report.json"
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
    audits = []
    for index, row in enumerate(read_jsonl(ADDITIONS), 1):
        source_path = ROOT / row["source_lineage"]["raw_document"]
        document = json.loads(source_path.read_text())
        document_sha = document.get("document_sha256") or text_sha256(document["text"])
        if (row["url"], row["document_sha256"]) != (document["url"], document_sha):
            raise ValueError("source lineage drift")
        if not all(line in document["text"] for line in row["evidence"].splitlines()):
            raise ValueError("evidence drift")
        audits.append({
            "audit_index": index,
            "decision": "add",
            "document_sha256": row["document_sha256"],
            "fact_id": row["fact_id"],
            "proposed_answer": row["answer"],
            "proposed_question": row["question"],
            "reason": row["paraphrase_rationale"],
            "reason_code": f"add_distinct_{row['topic']}_fact",
            "review_pass": "distinct_document_equipment_safety_additions",
            "reviewed_at": "2026-07-15",
            "reviewer": "codex-context-merit-audit-v301",
            "schema": "context-merit-audit-v301",
            "source": row["source"],
            "source_document": portable(source_path),
            "source_document_file_sha256": file_sha256(source_path),
            "source_support": "manual_paraphrase",
            "support_evidence": row["evidence"],
            "support_evidence_sha256": text_sha256(row["evidence"]),
            "url": row["url"],
        })
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, [])
    observed = observe()
    if not observed["dataset_equal"] or not observed["report_equal"]:
        raise ValueError("v301 projection is nondeterministic")
    if (observed["rows"], observed["eval_fact_count"]) != (528, 612):
        raise ValueError("v301 row/eval aggregate drift")
    if observed["baseline_capacity"] != EXPECTED_CAPACITY["before"] or observed["output_capacity"] != EXPECTED_CAPACITY["after"]:
        raise ValueError(f"v301 capacity drift: {observed['baseline_capacity']} -> {observed['output_capacity']}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v301 output hash drift")
    report = {
        "addition_artifact": {"path": portable(ADDITIONS), "rows": 3, "sha256": file_sha256(ADDITIONS)},
        "audit": {"by_decision": {"add": 3, "drop": 0, "edit": 0, "keep": 0}, "path": portable(AUDIT), "rows": 3, "sha256": file_sha256(AUDIT)},
        "conservative_capacity": {
            "after": observed["output_capacity"],
            "before": observed["baseline_capacity"],
            "delta": {key: observed["output_capacity"][key] - observed["baseline_capacity"][key] for key in observed["baseline_capacity"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "prior_checkpoint": {"candidate": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256}, "artifacts": [{"path": portable(path), "sha256": file_sha256(path)} for path in PRIOR_ARTIFACTS]},
        "isolated_build_projection": {
            "automated_projection_runs": 2,
            "new_additions_applied": 3,
            "output_rows": observed["rows"],
            "output_sha256": observed["dataset_sha256"],
            "repeat_dataset_byte_identical": observed["dataset_equal"],
            "repeat_projection_report_byte_identical": observed["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observed["eval_fact_count"],
        },
        "new_pending_curation": {"decisions": 0, "path": portable(CURATION), "sha256": file_sha256(CURATION)},
        "schema": "context-merit-audit-report-v301",
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
