#!/usr/bin/env python3
"""Improve four Rope365 safety Q&A from one manually reviewed source document."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V72_DIR = DATA / "manual_reviews/context_merit_audit_v72"
sys.path[:0] = [str(ROOT), str(V72_DIR)]
import build_context_merit_audit_v72 as previous

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v73.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v73.jsonl"
REPORT = OUT_DIR / "report_context_merit_v73.json"
REVIEWER = "codex-context-merit-audit-v73"
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
SOURCE = DATA / "raw/rope_resources_v1/rope365__7b5d548036392d65fec7.json"

SPECS = (
    {
        "fact_id": "fact-bda97c904a9412765ad2",
        "active_index": 229,
        "marker": "Make sure you can quickly untie anything that interferes with breathing",
        "question": "According to Rope365, what should a rigger be able to do quickly when a tie interferes with breathing?",
        "answer": "untie anything causing the interference",
        "reason_code": "repair_breathing_emergency_action_grammar",
        "reason": "The revised Q&A states the emergency action directly and removes the original question’s circular ‘what ... untie’ construction.",
    },
    {
        "fact_id": "fact-4634be1b1daf9ac325da",
        "active_index": 306,
        "marker": "The best way to be prepared for an incident is to visualize what you are about to do and figure out what could go wrong",
        "question": "What planning exercise does Rope365 recommend before tying to prepare for a possible incident?",
        "answer": "Visualize the planned tie and identify what could go wrong.",
        "reason_code": "naturalize_preincident_visualization_guidance",
        "reason": "The revised Q&A turns the source’s long wording into a clear, actionable pre-scene planning instruction.",
    },
    {
        "fact_id": "fact-79ea40e1492359a99725",
        "active_index": 370,
        "marker": "Both full and partial suspensions can increase the potential for nerve damage because the person’s weight adds to the pressure of the rope on the body",
        "question": "Why can full and partial suspensions increase the risk of nerve damage?",
        "answer": "The person’s weight adds to the rope’s pressure on the body.",
        "reason_code": "replace_damage_label_recall_with_suspension_mechanism",
        "reason": "The revised Q&A teaches why suspension can increase nerve risk instead of asking the learner to name the already-stated damage type.",
    },
    {
        "fact_id": "fact-42e3526e742dc48ac095",
        "active_index": 448,
        "marker": "Tie loosely and/or away from joints (wrist, elbows, knees, ankles)",
        "question": "Which joints does Rope365 say to keep rope away from to reduce nerve-compression risk?",
        "answer": "the wrists, elbows, knees, and ankles",
        "reason_code": "repair_nerve_risk_joint_list_grammar",
        "reason": "The revised wording removes the original ‘places to tie away from’ phrasing and presents the source’s joint list grammatically.",
    },
)
EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    spec["fact_id"]: spec["active_index"] for spec in SPECS
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v72",
    "direct_rows_without_prior_curation": 145,
    "rope365_safety_rows_selected": 4,
    "rows": 536,
    "sha256": "e54ecadd3dc2dafed0df4784e557d75980fbb4e341110446136df325a4f8a90d",
}
EXPECTED_OUTPUT_SHA256 = "97c1f00b7c7bf9dcf7fbb6f5e71f0dadc2baaac2b04fafa8cc1af852c98cafe4"


def build_projection(output: Path, report: Path, curations):
    previous.build_projection(output, report, curations)


def prior_decision_artifacts():
    artifacts = []
    for version in range(1, 73):
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
    with tempfile.TemporaryDirectory(prefix=".v73-observation-", dir=OUT_DIR) as temporary:
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
    with tempfile.TemporaryDirectory(prefix=".v72-projection-", dir=OUT_DIR) as temporary:
        baseline = Path(temporary) / "v72.jsonl"
        baseline_report = Path(temporary) / "v72.report.json"
        build_projection(baseline, baseline_report, PRIOR_PROJECTION_CURATIONS)
        rows = read_jsonl(baseline)
        if (
            len(rows) != 536
            or file_sha256(baseline) != PROJECTED_SELECTION_BASELINE["sha256"]
        ):
            raise ValueError("v72 projection drift")
        by_fact_id = {
            row["fact_id"]: (index, row) for index, row in enumerate(rows, 1)
        }
        observed_indices = {
            fact_id: by_fact_id[fact_id][0] for fact_id in EXPECTED_SELECTION
        }
        if observed_indices != PROJECTED_ACTIVE_INDICES:
            raise ValueError("v73 candidate drift")

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
        rationale = spec["reason"]
        audits.append(
            {
                "active_answer": active["answer"],
                "active_index": spec["active_index"],
                "active_question": active["question"],
                "audit_index": audit_index,
                "decision": "edit",
                "document_sha256": active["document_sha256"],
                "edited_answer": spec["answer"],
                "edited_question": spec["question"],
                "fact_id": spec["fact_id"],
                "paraphrase_rationale": rationale,
                "projection_lineage": {
                    "active_index": spec["active_index"],
                    "baseline_rows": 536,
                    "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
                    "prior_context_merit_review": True,
                },
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "review_pass": "rope365_safety_action_and_mechanism_reaudit",
                "reviewed_at": REVIEWED_AT,
                "reviewer": REVIEWER,
                "risk_features": CORE.risk_features(active),
                "schema": "context-merit-audit-v73",
                "source": document["source"],
                "source_document": str(SOURCE.relative_to(ROOT)),
                "source_document_file_sha256": file_sha256(SOURCE),
                "source_support": "manual_paraphrase",
                "support_evidence": evidence,
                "support_evidence_sha256": text_sha256(evidence),
                "url": document["url"],
            }
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
                "paraphrase_rationale": rationale,
                "question": spec["question"],
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "reviewed_at": REVIEWED_AT,
                "reviewer": REVIEWER,
                "source_lineage": active["source_lineage"],
                "support_type": "manual_paraphrase",
            }
        )
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)

    observed = deterministic_projection()
    if (
        not observed["dataset_equal"]
        or not observed["report_equal"]
        or observed["rows"] != 536
        or observed["eval_fact_count"] != 612
    ):
        raise ValueError("v73 deterministic projection drift")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v73 output hash drift")

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
            "by_decision": {"edit": 4},
            "by_reason": {
                spec["reason_code"]: 1 for spec in SPECS
            },
            "path": str(AUDIT.relative_to(ROOT)),
            "rows": 4,
            "sha256": file_sha256(AUDIT),
        },
        "frozen_prior_decision_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            for path in prior_decision_artifacts()
        ],
        "isolated_build_projection": {
            "active_after_context_merit_v72": 498,
            "active_after_this_tranche": 498,
            "automated_projection_runs": 2,
            "build_script": "build_curated_qa.py",
            "determinism_comparison_scope": "identical inputs, curation chain, and output/report paths",
            "new_drops_applied": 0,
            "new_edits_applied": 4,
            "output_rows": observed["rows"],
            "output_sha256": observed["dataset_sha256"],
            "prior_pending_addition_fact_ids_preserved": 36,
            "projection_report_normalized_sha256": observed["report_normalized_sha256"],
            "repeat_dataset_byte_identical": observed["dataset_equal"],
            "repeat_projection_report_byte_identical": observed["report_equal"],
            "reviewed_keep_fact_ids_preserved": 0,
            "sealed_eval_fact_count_reported_by_tooling": observed["eval_fact_count"],
            "unexpected_fact_ids": 0,
        },
        "new_pending_curation": {
            "by_action": {"edit": 4},
            "decisions": 4,
            "edit_support_types": {"extractive": 0, "manual_paraphrase": 4},
            "path": str(CURATION.relative_to(ROOT)),
            "sha256": file_sha256(CURATION),
        },
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "schema": "context-merit-audit-report-v73",
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
                "candidate_rule": "four direct Q&A from the same manually reviewed Rope365 safety document with circular grammar, label-recall framing, or missed causal explanation",
                "score": "manual standalone clarity and actionable safety value review",
                "tie_break": "active projection order",
            },
            "rows_selected": 4,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
