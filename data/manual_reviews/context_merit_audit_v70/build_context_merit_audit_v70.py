#!/usr/bin/env python3
"""Replace v69's still-awkward wet-jute question with the natural reviewed wording."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V69_DIR = DATA / "manual_reviews/context_merit_audit_v69"
sys.path.insert(0, str(V69_DIR))
import build_context_merit_audit_v69 as previous

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v70.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v70.jsonl"
REPORT = OUT_DIR / "report_context_merit_v70.json"
REVIEWER = "codex-context-merit-audit-v70"
REVIEWED_AT = "2026-07-14"

RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl

V69_CURATION = V69_DIR / "pending_curation_context_merit_v69.jsonl"
SOURCE = DATA / "raw/anatomiestudio_144932682af9c846.json"
ORIGINAL_FACT_ID = "fact-da15b630db4ec0ed79cf"
AWKWARD_QUESTION = "After wet jute rope dries, how may its tightened twist make the rope feel, according to Anatomie Studio?"
CORRECTED_QUESTION = "How can jute rope feel after getting wet and drying out?"
ANSWER = "spongy and springy"
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v69",
    "direct_rows_without_prior_curation": 150,
    "rows": 537,
    "sha256": "53ab9887a265e8d490849875dd98bf92ab33a88ce6f6c5e54ffb571c1d86d3ad",
}
EXPECTED_OUTPUT_SHA256 = "1c0aca8c60cafda413040401ae0320518715378b711d73add82075eb98e6f797"

PRIOR_PROJECTION_CURATIONS = previous.OUTPUT_PROJECTION_CURATIONS
if PRIOR_PROJECTION_CURATIONS[-1] != previous.CURATION:
    raise ValueError("v69 output curation chain drift")
OUTPUT_CONTEXT_CURATIONS = (*previous.OUTPUT_CONTEXT_CURATIONS, CURATION)
OUTPUT_PROJECTION_CURATIONS = (*PRIOR_PROJECTION_CURATIONS[:-1], CURATION)


def build_projection(output: Path, report: Path, curations):
    previous.build_projection(output, report, curations)


def prior_decision_artifacts():
    artifacts = []
    for version in range(1, 70):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        artifacts.extend(
            (
                directory / f"context_merit_audit_v{version}.jsonl",
                directory / f"pending_curation_context_merit_v{version}.jsonl",
                directory / f"report_context_merit_v{version}.json",
            )
        )
    return tuple(artifacts)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def corrected_curation_rows():
    rows = read_jsonl(V69_CURATION)
    by_fact_id = {row["fact_id"]: row for row in rows}
    expected = {
        ORIGINAL_FACT_ID,
        "fact-069a861dbb2bea9e47ca",
        "fact-f7e802bf0b2759290dc6",
    }
    if set(by_fact_id) != expected:
        raise ValueError("v69 curation membership drift")
    if by_fact_id[ORIGINAL_FACT_ID]["question"] != AWKWARD_QUESTION:
        raise ValueError("v69 wet-jute wording drift")
    corrected = []
    for row in rows:
        item = dict(row)
        if item["fact_id"] == ORIGINAL_FACT_ID:
            if item.get("expected_question") != CORRECTED_QUESTION:
                raise ValueError("reviewed expected wet-jute question drift")
            item.update(
                paraphrase_rationale="Uses the source's exact answer but asks the care question in direct, everyday language without retaining the article-specific attribution or twist-mechanism wording.",
                question=CORRECTED_QUESTION,
                reason="The short, natural question already recorded as the reviewed expectation asks the useful care fact directly without an awkward attribution clause.",
                reason_code="replace_awkward_wet_jute_question_with_natural_wording",
                reviewed_at=REVIEWED_AT,
                reviewer=REVIEWER,
                support_type="manual_paraphrase",
            )
        corrected.append(item)
    return corrected


def source_evidence():
    document = json.loads(SOURCE.read_text())
    marker = "making the ropes feel spongy and springy when they dry out"
    matches = [line for line in document["text"].splitlines() if marker in line]
    if len(matches) != 1:
        raise ValueError("wet-jute evidence marker drift")
    return document, matches[0]


def projection_observation():
    with tempfile.TemporaryDirectory(prefix=".v70-observation-", dir=OUT_DIR) as temporary:
        directory = Path(temporary)
        dataset = directory / "projection.jsonl"
        projection_report = directory / "projection.report.json"
        datasets = []
        reports = []
        for _ in (1, 2):
            build_projection(dataset, projection_report, OUTPUT_PROJECTION_CURATIONS)
            datasets.append(dataset.read_bytes())
            reports.append(projection_report.read_bytes())
        parsed_report = json.loads(reports[0])
        normalized_report = dict(parsed_report)
        normalized_report["output"] = "<projection-output>"
        normalized_bytes = (json.dumps(normalized_report, indent=2, sort_keys=True) + "\n").encode()
        return {
            "dataset_bytes_identical": datasets[0] == datasets[1],
            "dataset_sha256": sha_bytes(datasets[0]),
            "output_rows": datasets[0].count(b"\n"),
            "projection_report_bytes_identical": reports[0] == reports[1],
            "projection_report_normalized_sha256": sha_bytes(normalized_bytes),
            "sealed_eval_fact_count_reported_by_tooling": parsed_report["eval_fact_count"],
        }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v69-baseline-", dir=OUT_DIR) as temporary:
        baseline = Path(temporary) / "v69.jsonl"
        baseline_report = Path(temporary) / "v69.report.json"
        build_projection(baseline, baseline_report, PRIOR_PROJECTION_CURATIONS)
        rows = read_jsonl(baseline)
        if len(rows) != 537 or file_sha256(baseline) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v69 projection drift")
        matches = [
            (index, row)
            for index, row in enumerate(rows, 1)
            if row.get("curation", {}).get("original_fact_id") == ORIGINAL_FACT_ID
        ]
        if len(matches) != 1:
            raise ValueError("v69 wet-jute derived row membership drift")
        active_index, active_row = matches[0]
        if (active_row["question"], active_row["answer"]) != (AWKWARD_QUESTION, ANSWER):
            raise ValueError("v69 wet-jute derived row content drift")

    write_jsonl(CURATION, corrected_curation_rows())
    document, evidence = source_evidence()
    audit = {
        "active_answer": ANSWER,
        "active_index": active_index,
        "active_question": AWKWARD_QUESTION,
        "audit_index": 1,
        "decision": "edit",
        "document_sha256": "b316fd1a708dcb7688d44826496dd58b8c06d20b9c01d8daa73e99e7704abc53",
        "edited_answer": ANSWER,
        "edited_question": CORRECTED_QUESTION,
        "fact_id": ORIGINAL_FACT_ID,
        "projection_lineage": {
            "active_index": active_index,
            "baseline_rows": 537,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "derived_fact_id": active_row["fact_id"],
            "supersedes_context_merit_version": 69,
        },
        "reason": "The v69 wording remained awkward; this version uses the concise, natural question already recorded as the reviewed expectation.",
        "reason_code": "replace_awkward_wet_jute_question_with_natural_wording",
        "review_pass": "independent_natural_wording_repair",
        "reviewed_at": REVIEWED_AT,
        "reviewer": REVIEWER,
        "schema": "context-merit-audit-v70",
        "source": document["source"],
        "source_document": str(SOURCE.relative_to(ROOT)),
        "source_document_file_sha256": file_sha256(SOURCE),
        "source_support": "manual_paraphrase",
        "support_evidence": evidence,
        "support_evidence_sha256": text_sha256(evidence),
        "url": document["url"],
    }
    write_jsonl(AUDIT, [audit])

    observed = projection_observation()
    if not observed["dataset_bytes_identical"] or not observed["projection_report_bytes_identical"]:
        raise ValueError("projection determinism drift")
    if observed["output_rows"] != 537 or observed["sealed_eval_fact_count_reported_by_tooling"] != 612:
        raise ValueError("projection rows or aggregate eval count drift")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v70 output hash drift")

    report = {
        "active_baseline": {
            "dataset": {
                "path": str(ACTIVE_DATASET.relative_to(ROOT)),
                "rows": 784,
                "sha256": file_sha256(ACTIVE_DATASET),
            },
            "report": {
                "path": str(ACTIVE_REPORT.relative_to(ROOT)),
                "sha256": file_sha256(ACTIVE_REPORT),
            },
            "curation": [
                {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
                for path in ACTIVE_CURATIONS
            ],
        },
        "audit": {
            "by_decision": {"edit": 1},
            "by_reason": {"replace_awkward_wet_jute_question_with_natural_wording": 1},
            "path": str(AUDIT.relative_to(ROOT)),
            "rows": 1,
            "sha256": file_sha256(AUDIT),
        },
        "frozen_prior_decision_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            for path in prior_decision_artifacts()
        ],
        "isolated_build_projection": {
            "active_after_context_merit_v69": 499,
            "active_after_this_tranche": 499,
            "automated_projection_runs": 2,
            "build_script": "build_curated_qa.py",
            "determinism_comparison_scope": "identical inputs, curation chain, and output/report paths",
            "new_drops_applied": 0,
            "new_edits_applied": 1,
            "output_rows": observed["output_rows"],
            "output_sha256": observed["dataset_sha256"],
            "prior_pending_addition_fact_ids_preserved": 36,
            "projection_report_normalized_sha256": observed["projection_report_normalized_sha256"],
            "repeat_dataset_byte_identical": observed["dataset_bytes_identical"],
            "repeat_projection_report_byte_identical": observed["projection_report_bytes_identical"],
            "reviewed_keep_fact_ids_preserved": 0,
            "sealed_eval_fact_count_reported_by_tooling": observed["sealed_eval_fact_count_reported_by_tooling"],
            "unexpected_fact_ids": 0,
        },
        "new_pending_curation": {
            "by_action": {"edit": 3},
            "carried_v69_decisions": 2,
            "decisions": 3,
            "edit_support_types": {"extractive": 2, "manual_paraphrase": 1},
            "path": str(CURATION.relative_to(ROOT)),
            "sha256": file_sha256(CURATION),
            "superseded_path": str(V69_CURATION.relative_to(ROOT)),
            "superseded_path_excluded_from_v70_output_projection": True,
        },
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "schema": "context-merit-audit-report-v70",
        "sealed_evaluation_policy": {
            "automated_collision_tool": "build_curated_qa.py",
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-id collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "supersession": {
            "corrected_original_fact_id": ORIGINAL_FACT_ID,
            "reason": "v69 bundled three decisions, so v70 replaces that file in the isolated curation chain, improves one question, and carries the other two decisions semantically unchanged.",
            "supersedes_context_merit_version": 69,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
