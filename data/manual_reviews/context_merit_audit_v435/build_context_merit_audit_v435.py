#!/usr/bin/env python3
"""Complete a bounded set of three high-risk train-only answers."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V434 = DATA / "manual_reviews/context_merit_audit_v434"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V434), str(V290)]

import build_context_merit_audit_v434 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v435.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v435.jsonl"
REPORT = OUT_DIR / "report_context_merit_v435.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "f86f0618b0ac87ffd58b863763fd8d6609179c13dce2b945ddf0b96d75f3c099"
EXPECTED_OUTPUT_SHA256 = "25bc23bf80ed9cd027ead487e41ab8c9364b0b7db9c6de236f54b5128d6b87c8"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 259,
    "equipment_material": 23,
    "resources_general": 84,
    "safety_consent": 81,
    "technique": 71,
}
EXPECTED_CAPACITY_AFTER = dict(EXPECTED_CAPACITY_BEFORE)

RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
conservative_capacity = previous.conservative_capacity
portable = previous.portable
PRIOR = (
    V434 / "context_merit_audit_v434.jsonl",
    V434 / "pending_curation_context_merit_v434.jsonl",
    V434 / "report_context_merit_v434.json",
)

SPECS = (
    {
        "fact_id": "fact-c673f82c799599d4c5d5",
        "active_index": 4,
        "expected_question": "According to Esinem, what factors should be considered together when assessing a box tie rather than evaluating the tie in isolation?",
        "expected_answer": "Consider the person being tied and their individual differences, the bondage planned, and whether the tie fulfills its intended function.",
        "expected_evidence_sha256": "7f085589d276c60ed346d2de018cfd638c7f215402e6a758c3fe8b26805f0fbd",
        "question": "According to Esinem, what factors should be considered together when assessing a box tie rather than evaluating the tie in isolation?",
        "answer": "Esinem recommends assessing the person being tied and their individual differences, the bondage planned, and whether the box tie fulfills its intended function.",
        "source_document": "data/raw/esinem_3bbb8f77f7d062a7.json",
        "reason_code": "complete_holistic_box_tie_assessment_attribution",
        "reason": "The replacement turns the source-supported imperative into a standalone attributed answer while preserving the person, planned-bondage, and intended-function factors.",
    },
    {
        "fact_id": "fact-f1eb752737fc72675037",
        "active_index": 335,
        "expected_question": "What status pressures does Tessin Doyama say can make it harder for a tied person to stop a scene?",
        "expected_answer": "They may fear that an admired or famous partner will not tie them again, or that stopping means others will not see them as a “real” submissive.",
        "expected_evidence_sha256": "ea5f390c5f59dfdae990387fbe5f220bb07e2709bf588db8cb030bd228a88c3d",
        "question": "What status pressures does Tessin Doyama say can make it harder for a tied person to stop a scene?",
        "answer": "Tessin Doyama says a tied person may fear that an admired or famous partner will not tie them again, or that stopping will make others see them as not a “real” submissive.",
        "source_document": "data/raw/kinbakutoday_43095aa305e35a41.json",
        "reason_code": "resolve_status_pressure_subject_and_attribution",
        "reason": "The replacement resolves the ambiguous pronoun and attributes the explanation while preserving both source-supported pressures against stopping.",
    },
    {
        "fact_id": "fact-c7f7ab99d9a0ee4a7061",
        "active_index": 379,
        "expected_question": "When does WykD’s newcomer article say a dominant has the right to tell a submissive what to do?",
        "expected_answer": "It says a dominant has that right only when the individual has actually consented.",
        "expected_evidence_sha256": "6b9145178f08f067fc096149d52a0ef31a7c5ec70b1a3a75749aefeb693d3d6f",
        "question": "When does WykD’s newcomer article say a dominant has the right to tell a submissive what to do?",
        "answer": "WykD says a dominant has that right only when the individual has actually consented.",
        "source_document": "data/raw/wykd_944e4e6d621a97c9.json",
        "reason_code": "resolve_consent_authority_source_subject",
        "reason": "The replacement resolves the ambiguous source pronoun and explicitly attributes the consent condition without broadening the claim.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v434 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v435-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v434.jsonl"
    build_baseline(base, inputs / "v434.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v435-observe-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        out = d / "out.jsonl"
        report = d / "out.report.json"
        datasets: list[bytes] = []
        reports: list[bytes] = []
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
    with tempfile.TemporaryDirectory(prefix=".v435-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v434.jsonl"
        build_baseline(base, d / "v434.report.json")
        before = read_jsonl(base)

    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}
    audits: list[dict] = []
    curations: list[dict] = []
    for audit_index, spec in enumerate(SPECS, 1):
        index, active = by_fact[spec["fact_id"]]
        if index != spec["active_index"]:
            raise ValueError(f"candidate index drift {spec['fact_id']}")
        if (active["question"], active["answer"]) != (
            spec["expected_question"],
            spec["expected_answer"],
        ):
            raise ValueError(f"candidate semantic drift {spec['fact_id']}")
        evidence = active.get("evidence")
        if not evidence or text_sha256(evidence) != spec["expected_evidence_sha256"]:
            raise ValueError(f"candidate evidence drift {spec['fact_id']}")
        source_document = ROOT / spec["source_document"]
        source = json.loads(source_document.read_text())
        if source.get("url") != active["url"]:
            raise ValueError(f"source URL drift {spec['fact_id']}")
        if evidence not in source["text"]:
            raise ValueError(f"source evidence drift {spec['fact_id']}")

        curations.append(
            {
                "action": "edit",
                "answer": spec["answer"],
                "document_sha256": active["document_sha256"],
                "evidence": evidence,
                "evidence_url": active["url"],
                "expected_answer": active["answer"],
                "expected_question": active["question"],
                "fact_id": active["fact_id"],
                "paraphrase_rationale": spec["reason"],
                "question": spec["question"],
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v435",
                "source_lineage": active["source_lineage"],
                "support_type": "manual_paraphrase",
            }
        )
        audits.append(
            {
                "active_answer": active["answer"],
                "active_index": index,
                "active_question": active["question"],
                "audit_index": audit_index,
                "decision": "edit",
                "document_sha256": active["document_sha256"],
                "edited_answer": spec["answer"],
                "edited_question": spec["question"],
                "fact_id": active["fact_id"],
                "paraphrase_rationale": spec["reason"],
                "projection_lineage": {
                    "active_index": index,
                    "baseline_rows": BASELINE_ROWS,
                    "baseline_sha256": BASELINE_SHA256,
                },
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "review_pass": "bounded_safety_consent_fit_context_merit_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v435",
                "schema": "context-merit-audit-v435",
                "source": active["source"],
                "source_document": spec["source_document"],
                "source_document_file_sha256": file_sha256(source_document),
                "source_snapshot_chars_manually_reviewed": len(source["text"]),
                "source_support": "manual_full_snapshot_and_dataset_context_review",
                "support_evidence": evidence,
                "support_evidence_sha256": text_sha256(evidence),
                "url": active["url"],
            }
        )

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection {observation}")
    if (observation["rows"], observation["eval"]) != (531, 612):
        raise ValueError(f"projection count drift {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift {observation}")
    if observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"output drift {observation['sha']}")

    REPORT.write_text(
        json.dumps(
            {
                "audit": {
                    "bounded_manual_selection_rows": 3,
                    "by_decision": {"drop": 0, "edit": 3, "keep": 0},
                    "path": portable(AUDIT),
                    "rows": 3,
                    "sha256": file_sha256(AUDIT),
                },
                "conservative_capacity": {
                    "after": observation["after"],
                    "before": observation["before"],
                    "delta": {
                        key: observation["after"][key] - observation["before"][key]
                        for key in observation["before"]
                    },
                    "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
                },
                "prior_checkpoint": {
                    "candidate": {"rows": BASELINE_ROWS, "sha256": BASELINE_SHA256},
                    "artifacts": [
                        {"path": portable(path), "sha256": file_sha256(path)} for path in PRIOR
                    ],
                },
                "isolated_build_projection": {
                    "automated_projection_runs": 2,
                    "new_additions_applied": 0,
                    "output_rows": observation["rows"],
                    "output_sha256": observation["sha"],
                    "repeat_dataset_byte_identical": observation["dataset_equal"],
                    "repeat_projection_report_byte_identical": observation["report_equal"],
                    "sealed_eval_fact_count_reported_by_tooling": observation["eval"],
                },
                "new_pending_curation": {
                    "decisions": 3,
                    "path": portable(CURATION),
                    "sha256": file_sha256(CURATION),
                },
                "schema": "context-merit-audit-report-v435",
                "sealed_evaluation_policy": {
                    "automated_collision_tool_reads_sealed_content": True,
                    "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
                    "manual_worker_opened_eval_or_heldout_content": False,
                    "manual_worker_received_eval_or_heldout_content": False,
                    "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
                    "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
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
