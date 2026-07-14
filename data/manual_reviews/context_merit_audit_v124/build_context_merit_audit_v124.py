#!/usr/bin/env python3
"""Remove a promotional Esinem tutorial-title lookup."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V123_DIR = DATA / "manual_reviews/context_merit_audit_v123"
sys.path[:0] = [str(ROOT), str(V123_DIR)]
import build_context_merit_audit_v123 as previous

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v124.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v124.jsonl"
REPORT = OUT_DIR / "report_context_merit_v124.json"
REVIEWER = "codex-context-merit-audit-v124"
REVIEWED_AT = "2026-07-14"

RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
CORE = previous.CORE
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
CONTEXT_CURATIONS = previous.OUTPUT_CONTEXT_CURATIONS
PRIOR_PROJECTION_CURATIONS = previous.OUTPUT_PROJECTION_CURATIONS
OUTPUT_CONTEXT_CURATIONS = (*CONTEXT_CURATIONS, CURATION)
OUTPUT_PROJECTION_CURATIONS = (*PRIOR_PROJECTION_CURATIONS, CURATION)
SOURCE = DATA / "raw/esinem_7b67baa6649b24c5.json"

SPECS = (
    {
        "fact_id": "fact-632fafd34ec06074a6c2",
        "active_index": 453,
        "marker": "Our latest shibari tutorial for ShibariClasses.com is far more than a simple ‘paint by numbers’ futo momo, not only does it explore numerous forms of this versatile tie but it also teaches a wealth of tying techniques applicable to many situations.",
        "decision": "drop",
        "reason_code": "promotional_tutorial_title_lookup",
        "reason": "This asks only which tie appears in a promotional article title and preserves none of the article's technique, fit-adaptation, or suspension-scope context.",
    },
)
EXPECTED_SELECTION = tuple(s["fact_id"] for s in SPECS)
PROJECTED_ACTIVE_INDICES = {s["fact_id"]: s["active_index"] for s in SPECS}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v123",
    "direct_rows_without_prior_curation": 92,
    "esinem_promotional_tutorial_rows_selected": 1,
    "rows": 520,
    "sha256": "dfd5f6ee141ff1ac0daff204693a65bdfd44a656a7f4b2d2c0e19c1206a9e62e",
}
EXPECTED_OUTPUT_SHA256 = "d3cd46e07eca8e4f6570448645324b02a34b45c03aac896292496ed4210f4bf7"


def build_projection(output: Path, report: Path, curations) -> None:
    previous.build_projection(output, report, curations)


def prior_decision_artifacts():
    out = []
    for version in range(1, 124):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        out.extend(
            (
                directory / f"context_merit_audit_v{version}.jsonl",
                directory / f"pending_curation_context_merit_v{version}.jsonl",
                directory / f"report_context_merit_v{version}.json",
            )
        )
    return tuple(out)


def evidence(document, marker):
    matches = [line for line in document["text"].splitlines() if marker in line]
    if len(matches) != 1:
        raise ValueError(f"evidence drift: {marker}")
    return matches[0]


def observation():
    with tempfile.TemporaryDirectory(prefix=".v124-observation-", dir=OUT_DIR) as temp:
        directory = Path(temp)
        dataset = directory / "projection.jsonl"
        report = directory / "projection.report.json"
        dataset_bytes = []
        report_bytes = []
        for _ in (1, 2):
            build_projection(dataset, report, OUTPUT_PROJECTION_CURATIONS)
            dataset_bytes.append(dataset.read_bytes())
            report_bytes.append(report.read_bytes())
        parsed = json.loads(report_bytes[0])
        normalized = dict(parsed)
        normalized["output"] = "<projection-output>"
        normalized_bytes = (json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode()
        return {
            "dataset_equal": dataset_bytes[0] == dataset_bytes[1],
            "dataset_sha256": hashlib.sha256(dataset_bytes[0]).hexdigest(),
            "report_equal": report_bytes[0] == report_bytes[1],
            "report_normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
            "rows": dataset_bytes[0].count(b"\n"),
            "eval_fact_count": parsed["eval_fact_count"],
        }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v123-projection-", dir=OUT_DIR) as temp:
        directory = Path(temp)
        baseline = directory / "v123.jsonl"
        baseline_report = directory / "v123.report.json"
        build_projection(baseline, baseline_report, PRIOR_PROJECTION_CURATIONS)
        rows = read_jsonl(baseline)
        if len(rows) != 520 or file_sha256(baseline) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v123 projection drift")
        by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(rows, 1)}
        if {fact_id: by_fact[fact_id][0] for fact_id in EXPECTED_SELECTION} != PROJECTED_ACTIVE_INDICES:
            raise ValueError("v124 candidate drift")

    document = json.loads(SOURCE.read_text())
    audits = []
    curations = []
    for audit_index, spec in enumerate(SPECS, 1):
        active = by_fact[spec["fact_id"]][1]
        support = evidence(document, spec["marker"])
        if active["source"] != document["source"] or active["url"] != document["url"]:
            raise ValueError(f"{spec['fact_id']}: Esinem tutorial source lineage drift")
        audit = {
            "active_answer": active["answer"],
            "active_index": spec["active_index"],
            "active_question": active["question"],
            "audit_index": audit_index,
            "decision": spec["decision"],
            "document_sha256": active["document_sha256"],
            "fact_id": spec["fact_id"],
            "projection_lineage": {
                "active_index": spec["active_index"],
                "baseline_rows": 520,
                "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"],
            "reason_code": spec["reason_code"],
            "review_pass": "esinem_promotional_tutorial_reaudit",
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "risk_features": CORE.risk_features(active),
            "schema": "context-merit-audit-v124",
            "source": document["source"],
            "source_document": str(SOURCE.relative_to(ROOT)),
            "source_document_file_sha256": file_sha256(SOURCE),
            "source_support": "manual_paraphrase" if spec["decision"] == "edit" else "normalized_extractive",
            "support_evidence": support,
            "support_evidence_sha256": text_sha256(support),
            "url": document["url"],
        }
        if spec["decision"] == "edit":
            audit.update(
                edited_answer=spec["answer"],
                edited_question=spec["question"],
                paraphrase_rationale=spec["reason"],
            )
            curations.append(
                {
                    "action": "edit",
                    "answer": spec["answer"],
                    "document_sha256": active["document_sha256"],
                    "evidence": support,
                    "evidence_url": document["url"],
                    "expected_answer": active["answer"],
                    "expected_question": active["question"],
                    "fact_id": spec["fact_id"],
                    "paraphrase_rationale": spec["reason"],
                    "question": spec["question"],
                    "reason": spec["reason"],
                    "reason_code": spec["reason_code"],
                    "reviewed_at": REVIEWED_AT,
                    "reviewer": REVIEWER,
                    "source_lineage": active["source_lineage"],
                    "support_type": "manual_paraphrase",
                }
            )
        elif spec["decision"] == "drop":
            curations.append(
                {
                    "action": "drop",
                    "document_sha256": active["document_sha256"],
                    "evidence": support,
                    "evidence_url": document["url"],
                    "expected_answer": active["answer"],
                    "expected_question": active["question"],
                    "fact_id": spec["fact_id"],
                    "reason": spec["reason"],
                    "reason_code": spec["reason_code"],
                    "reviewed_at": REVIEWED_AT,
                    "reviewer": REVIEWER,
                    "source_lineage": active["source_lineage"],
                }
            )
        audits.append(audit)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observed = observation()
    if not observed["dataset_equal"] or not observed["report_equal"] or observed["rows"] != 519 or observed["eval_fact_count"] != 612:
        raise ValueError("v124 deterministic projection drift")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v124 output hash drift")
    report = {
        "active_baseline": {
            "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784, "sha256": file_sha256(ACTIVE_DATASET)},
            "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)), "sha256": file_sha256(ACTIVE_REPORT)},
            "curation": [{"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
        },
        "audit": {
            "by_decision": {"drop": 1, "edit": 0, "keep": 0},
            "by_reason": {spec["reason_code"]: 1 for spec in SPECS},
            "path": str(AUDIT.relative_to(ROOT)),
            "rows": 1,
            "sha256": file_sha256(AUDIT),
        },
        "frozen_prior_decision_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in prior_decision_artifacts()
        ],
        "isolated_build_projection": {
            "active_after_context_merit_v123": 482,
            "active_after_this_tranche": 481,
            "automated_projection_runs": 2,
            "build_script": "build_curated_qa.py",
            "determinism_comparison_scope": "identical inputs, curation chain, and output/report paths",
            "new_drops_applied": 1,
            "new_edits_applied": 0,
            "output_rows": observed["rows"],
            "output_sha256": observed["dataset_sha256"],
            "prior_pending_addition_fact_ids_preserved": 35,
            "projection_report_normalized_sha256": observed["report_normalized_sha256"],
            "repeat_dataset_byte_identical": observed["dataset_equal"],
            "repeat_projection_report_byte_identical": observed["report_equal"],
            "projection_strategy": "delegate to the v123 second-stage projection and append this tranche's decisions",
            "reviewed_keep_fact_ids_preserved": 0,
            "sealed_eval_fact_count_reported_by_tooling": observed["eval_fact_count"],
            "unexpected_fact_ids": 0,
        },
        "new_pending_curation": {
            "by_action": {"drop": 1},
            "decisions": 1,
            "edit_support_types": {"extractive": 0, "manual_paraphrase": 0},
            "path": str(CURATION.relative_to(ROOT)),
            "sha256": file_sha256(CURATION),
        },
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "schema": "context-merit-audit-report-v124",
        "sealed_evaluation_policy": {
            "automated_collision_tool": "build_curated_qa.py",
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-id collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "selection": {
            "active_rows": 520,
            "projected_baseline": PROJECTED_SELECTION_BASELINE,
            "ranking": {
                "candidate_rule": "the not-yet-context-audited title lookup from the fully read Esinem futo-momo promotional article",
                "score": "manual source support, promotional-content risk, transferable value, and training merit",
                "tie_break": "active projection order",
            },
            "rows_selected": 1,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
