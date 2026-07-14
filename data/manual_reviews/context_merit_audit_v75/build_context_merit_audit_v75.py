#!/usr/bin/env python3
"""Audit four partner-vetting and feedback rows from Rope365's communication page."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V74_DIR = DATA / "manual_reviews/context_merit_audit_v74"
sys.path[:0] = [str(ROOT), str(V74_DIR)]
import build_context_merit_audit_v74 as previous

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v75.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v75.jsonl"
REPORT = OUT_DIR / "report_context_merit_v75.json"
REVIEWER = "codex-context-merit-audit-v75"
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
SOURCE = DATA / "raw/rope_resources_v1/rope365__b602b6493b5eb6f55206.json"

SPECS = (
    {
        "fact_id": "fact-f2f2561a20d4f7b3ef11",
        "active_index": 223,
        "marker": "have a safety call (someone who will check back on you after a certain period of time)",
        "decision": "keep",
        "reason_code": "retain_clear_safety_call_definition",
        "reason": "The existing Q&A gives a concise, source-supported definition of a safety call for a first meeting with a new rope partner.",
    },
    {
        "fact_id": "fact-d64b1477fee41ef24905",
        "active_index": 256,
        "marker": "contact someone a few days after a session to open the channel for feedback",
        "decision": "edit",
        "question": "What follow-up does Rope365 recommend when a partner cannot give feedback immediately after a session?",
        "answer": "Contact the partner a few days later to reopen the feedback channel and check whether they need help.",
        "reason_code": "resolve_ambiguous_feedback_followup_referent",
        "reason": "The revised answer resolves the source’s ambiguous ‘contact someone’ phrasing using the paragraph’s partner referent and retains both feedback and support purposes.",
    },
    {
        "fact_id": "fact-fa78f985ebbbb2370a5f",
        "active_index": 370,
        "marker": "make the first contact with a new partner in a public space (or with trusted people around)",
        "decision": "edit",
        "question": "Where does Rope365 suggest first meeting a new rope partner?",
        "answer": "in a public space or with trusted people around",
        "reason_code": "restore_trusted_people_alternative_to_first_meeting_guidance",
        "reason": "The revised answer restores the source’s omitted alternative of having trusted people present while keeping the guidance concise.",
    },
    {
        "fact_id": "fact-a4501a08619f5fc7886c",
        "active_index": 516,
        "marker": "This will allow you to adapt the dialogue and make sure you don’t push each other too fast",
        "decision": "edit",
        "question": "Why does Rope365 suggest that new partners discuss their experience levels and past experiences?",
        "answer": "so they can adapt the dialogue and avoid pushing each other too fast",
        "reason_code": "repair_new_partner_experience_discussion_answer_grammar",
        "reason": "The revised answer supplies the missing subject and expresses the source’s pacing purpose as a grammatical response.",
    },
)
EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {spec["fact_id"]: spec["active_index"] for spec in SPECS}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v74",
    "direct_rows_without_prior_curation": 138,
    "rope365_communication_rows_selected": 4,
    "rows": 536,
    "sha256": "a390c09ac2a26013e17a33713786c418451031c0cb5c0d8a66a5496e3b5ef151",
}
EXPECTED_OUTPUT_SHA256 = "555a70967b3de5462174995928569a0795df5b4335ab7584934998ea35e1a619"


def build_projection(output: Path, report: Path, curations):
    previous.build_projection(output, report, curations)


def prior_decision_artifacts():
    artifacts = []
    for version in range(1, 75):
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
    with tempfile.TemporaryDirectory(prefix=".v75-observation-", dir=OUT_DIR) as temporary:
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
    with tempfile.TemporaryDirectory(prefix=".v74-projection-", dir=OUT_DIR) as temporary:
        baseline = Path(temporary) / "v74.jsonl"
        baseline_report = Path(temporary) / "v74.report.json"
        build_projection(baseline, baseline_report, PRIOR_PROJECTION_CURATIONS)
        rows = read_jsonl(baseline)
        if len(rows) != 536 or file_sha256(baseline) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v74 projection drift")
        by_fact_id = {row["fact_id"]: (index, row) for index, row in enumerate(rows, 1)}
        if {fact_id: by_fact_id[fact_id][0] for fact_id in EXPECTED_SELECTION} != PROJECTED_ACTIVE_INDICES:
            raise ValueError("v75 candidate drift")

    document = json.loads(SOURCE.read_text())
    if document.get("document_sha256") != "ba41f96db0578f593930a21a579f6a30f3658b100da8390fea2edbdf5b4abb3d":
        raise ValueError("Rope365 communication document drift")
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
            "review_pass": "rope365_partner_vetting_and_feedback_reaudit",
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "risk_features": CORE.risk_features(active),
            "schema": "context-merit-audit-v75",
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
    if not observed["dataset_equal"] or not observed["report_equal"] or observed["rows"] != 536 or observed["eval_fact_count"] != 612:
        raise ValueError("v75 deterministic projection drift")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v75 output hash drift")

    report = {
        "active_baseline": {
            "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784, "sha256": file_sha256(ACTIVE_DATASET)},
            "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)), "sha256": file_sha256(ACTIVE_REPORT)},
            "curation": [{"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
        },
        "audit": {
            "by_decision": {"edit": 3, "keep": 1},
            "by_reason": {spec["reason_code"]: 1 for spec in SPECS},
            "path": str(AUDIT.relative_to(ROOT)),
            "rows": 4,
            "sha256": file_sha256(AUDIT),
        },
        "frozen_prior_decision_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in prior_decision_artifacts()
        ],
        "isolated_build_projection": {
            "active_after_context_merit_v74": 498,
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
        "schema": "context-merit-audit-report-v75",
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
                "candidate_rule": "four direct partner-vetting, first-meeting, pacing, and feedback Q&A from one manually read Rope365 communication page",
                "score": "manual completeness, referent clarity, and practical partner-safety value review",
                "tie_break": "active projection order",
            },
            "rows_selected": 4,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
