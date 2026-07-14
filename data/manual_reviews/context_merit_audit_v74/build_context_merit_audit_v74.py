#!/usr/bin/env python3
"""Finish a second bounded Rope365 safety-document Q&A quality tranche."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V73_DIR = DATA / "manual_reviews/context_merit_audit_v73"
sys.path[:0] = [str(ROOT), str(V73_DIR)]
import build_context_merit_audit_v73 as previous

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v74.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v74.jsonl"
REPORT = OUT_DIR / "report_context_merit_v74.json"
REVIEWER = "codex-context-merit-audit-v74"
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
SOURCE = previous.SOURCE

SPECS = (
    {
        "fact_id": "fact-e7a338d404977ce267a6",
        "active_index": 84,
        "marker": "Tie kneeling or sitting",
        "decision": "edit",
        "question": "What lower-risk body positions does Rope365 recommend for tying?",
        "answer": "kneeling or sitting",
        "reason_code": "naturalize_lower_fall_risk_position_guidance",
        "reason": "The revised Q&A asks directly for the recommended lower-risk positions and removes an imperative fragment from the answer.",
    },
    {
        "fact_id": "fact-a45715221b107ff37347",
        "active_index": 230,
        "marker": "Stay with your bound partner at all times",
        "decision": "edit",
        "question": "According to Rope365, what supervision is required while someone is bound?",
        "answer": "Stay with the bound partner at all times.",
        "reason_code": "frame_constant_presence_as_supervision_requirement",
        "reason": "The revised question states the safety concept as a supervision requirement and the answer is a complete standalone instruction.",
    },
    {
        "fact_id": "fact-582bc37560761a271e19",
        "active_index": 314,
        "marker": "Make sure your partner is aware of the risks involved in the type of play you will share together",
        "decision": "keep",
        "reason_code": "retain_clear_preplay_risk_disclosure_guidance",
        "reason": "The existing Q&A clearly teaches that a partner should receive risk information specific to the planned play before it begins.",
    },
    {
        "fact_id": "fact-55f26dc889d7aaa6c515",
        "active_index": 357,
        "marker": "Participating in a first aid class is a good way to get training for emergency incident response",
        "decision": "edit",
        "question": "What training does Rope365 recommend for emergency incident response?",
        "answer": "a first-aid class",
        "reason_code": "repair_first_aid_training_answer_form",
        "reason": "The revised answer is a concise noun phrase that fits the question instead of a dangling gerund construction.",
    },
)
EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    spec["fact_id"]: spec["active_index"] for spec in SPECS
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v73",
    "direct_rows_without_prior_curation": 141,
    "rope365_safety_rows_selected": 4,
    "rows": 536,
    "sha256": "97c1f00b7c7bf9dcf7fbb6f5e71f0dadc2baaac2b04fafa8cc1af852c98cafe4",
}
EXPECTED_OUTPUT_SHA256 = "a390c09ac2a26013e17a33713786c418451031c0cb5c0d8a66a5496e3b5ef151"


def build_projection(output: Path, report: Path, curations):
    previous.build_projection(output, report, curations)


def prior_decision_artifacts():
    artifacts = []
    for version in range(1, 74):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        artifacts.extend(
            (
                directory / f"context_merit_audit_v{version}.jsonl",
                directory / f"pending_curation_context_merit_v{version}.jsonl",
                directory / f"report_context_merit_v{version}.json",
            )
        )
    return tuple(artifacts)


def source_evidence(document, marker: str) -> str:
    matches = [line for line in document["text"].splitlines() if marker in line]
    if len(matches) != 1:
        raise ValueError(f"evidence marker drift: {marker}")
    return matches[0]


def deterministic_projection():
    with tempfile.TemporaryDirectory(prefix=".v74-observation-", dir=OUT_DIR) as temporary:
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
            "dataset_equal": datasets[0] == datasets[1],
            "dataset_sha256": hashlib.sha256(datasets[0]).hexdigest(),
            "report_equal": reports[0] == reports[1],
            "report_normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
            "rows": datasets[0].count(b"\n"),
            "eval_fact_count": parsed_report["eval_fact_count"],
        }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v73-projection-", dir=OUT_DIR) as temporary:
        baseline = Path(temporary) / "v73.jsonl"
        baseline_report = Path(temporary) / "v73.report.json"
        build_projection(baseline, baseline_report, PRIOR_PROJECTION_CURATIONS)
        rows = read_jsonl(baseline)
        if (
            len(rows) != 536
            or file_sha256(baseline) != PROJECTED_SELECTION_BASELINE["sha256"]
        ):
            raise ValueError("v73 projection drift")
        by_fact_id = {
            row["fact_id"]: (index, row) for index, row in enumerate(rows, 1)
        }
        observed_indices = {
            fact_id: by_fact_id[fact_id][0] for fact_id in EXPECTED_SELECTION
        }
        if observed_indices != PROJECTED_ACTIVE_INDICES:
            raise ValueError("v74 candidate drift")

    document = json.loads(SOURCE.read_text())
    if document.get("document_sha256") != "f83771d810b2b197ec3a0fb58660ac529b996951413e58fa87ff48e37f52a6d7":
        raise ValueError("Rope365 safety document drift")
    audits = []
    curations = []
    for audit_index, spec in enumerate(SPECS, 1):
        active = by_fact_id[spec["fact_id"]][1]
        evidence = source_evidence(document, spec["marker"])
        if active["document_sha256"] != document["document_sha256"]:
            raise ValueError(f"{spec['fact_id']}: document lineage drift")
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
                "baseline_rows": 536,
                "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"],
            "reason_code": spec["reason_code"],
            "review_pass": "rope365_safety_supervision_and_preparation_reaudit",
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "risk_features": CORE.risk_features(active),
            "schema": "context-merit-audit-v74",
            "source": document["source"],
            "source_document": str(SOURCE.relative_to(ROOT)),
            "source_document_file_sha256": file_sha256(SOURCE),
            "source_support": "normalized_extractive" if spec["decision"] == "keep" else "manual_paraphrase",
            "support_evidence": evidence,
            "support_evidence_sha256": text_sha256(evidence),
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
                    "evidence": evidence,
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
        audits.append(audit)
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)

    observed = deterministic_projection()
    if (
        not observed["dataset_equal"]
        or not observed["report_equal"]
        or observed["rows"] != 536
        or observed["eval_fact_count"] != 612
    ):
        raise ValueError("v74 deterministic projection drift")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v74 output hash drift")

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
            "by_decision": {"edit": 3, "keep": 1},
            "by_reason": {spec["reason_code"]: 1 for spec in SPECS},
            "path": str(AUDIT.relative_to(ROOT)),
            "rows": 4,
            "sha256": file_sha256(AUDIT),
        },
        "frozen_prior_decision_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            for path in prior_decision_artifacts()
        ],
        "isolated_build_projection": {
            "active_after_context_merit_v73": 498,
            "active_after_this_tranche": 498,
            "automated_projection_runs": 2,
            "build_script": "build_curated_qa.py",
            "determinism_comparison_scope": "identical inputs, curation chain, and output/report paths",
            "new_drops_applied": 0,
            "new_edits_applied": 3,
            "output_rows": observed["rows"],
            "output_sha256": observed["dataset_sha256"],
            "prior_pending_addition_fact_ids_preserved": 36,
            "projection_report_normalized_sha256": observed["report_normalized_sha256"],
            "repeat_dataset_byte_identical": observed["dataset_equal"],
            "repeat_projection_report_byte_identical": observed["report_equal"],
            "reviewed_keep_fact_ids_preserved": 1,
            "sealed_eval_fact_count_reported_by_tooling": observed["eval_fact_count"],
            "unexpected_fact_ids": 0,
        },
        "new_pending_curation": {
            "by_action": {"edit": 3},
            "decisions": 3,
            "edit_support_types": {"extractive": 0, "manual_paraphrase": 3},
            "path": str(CURATION.relative_to(ROOT)),
            "sha256": file_sha256(CURATION),
        },
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "schema": "context-merit-audit-report-v74",
        "sealed_evaluation_policy": {
            "automated_collision_tool": "build_curated_qa.py",
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-id collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "selection": {
            "active_rows": 536,
            "projected_baseline": PROJECTED_SELECTION_BASELINE,
            "ranking": {
                "candidate_rule": "four additional direct Q&A from the same manually reviewed Rope365 safety document",
                "score": "manual standalone grammar and actionable safety value review",
                "tie_break": "active projection order",
            },
            "rows_selected": 4,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
