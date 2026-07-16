#!/usr/bin/env python3
"""Complete three source-grounded safety answers in the train-only projection."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V433 = DATA / "manual_reviews/context_merit_audit_v433"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V433), str(V290)]

import build_context_merit_audit_v433 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v434.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v434.jsonl"
REPORT = OUT_DIR / "report_context_merit_v434.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "5d21081e044c3f5a2110212dae19e22dfdc08e4b59d483a134871cbeaa9c268c"
EXPECTED_OUTPUT_SHA256 = "f86f0618b0ac87ffd58b863763fd8d6609179c13dce2b945ddf0b96d75f3c099"
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
    V433 / "context_merit_audit_v433.jsonl",
    V433 / "pending_curation_context_merit_v433.jsonl",
    V433 / "report_context_merit_v433.json",
)

INSTRUCTOR_EVIDENCE = (
    "There are no standards for rope education, and a tremendous amount of misinformation and confusion gets passed around. "
    "Attending classes at reputable community venues does not necessarily mean the information is correct. Instructors can be "
    "hired to teach at top-tier venues or websites even if their content is misleading or omits crucial safety information. "
    "Absorb what you can with a grain of salt. Ask whether what you are told makes sense and whether it agrees with what you have "
    "learned from other sources. In person, ask lots of \"why\" questions; if an instructor cannot answer, they may be repeating "
    "material without understanding it."
)

SPECS = (
    {
        "fact_id": "fact-6d1fd5a6040e7583d9ab",
        "active_index": 57,
        "expected_question": "How does Crash Restraint suggest testing a rope instructor's understanding?",
        "expected_answer": "Ask many \"why\" questions, check whether the answers make sense and agree with other sources, and be wary if the instructor cannot explain them.",
        "expected_evidence_sha256": "fd7f4b9e6051c18d727ecf4aac8ff0904141cdfff9dbce1fc826dbd0591ac935",
        "question": "How does Crash Restraint suggest testing a rope instructor's understanding?",
        "answer": "Crash Restraint recommends asking many \"why\" questions, checking whether the answers make sense and agree with other sources, and being wary when an instructor cannot explain the material.",
        "evidence": INSTRUCTOR_EVIDENCE,
        "source_document": "data/raw/crash_restraint_getting_started_20260714.json",
        "reason_code": "complete_instructor_due_diligence_attribution",
        "reason": "The replacement turns the source-supported imperative into a standalone attributed answer while retaining the why-question, cross-source, and explanation checks; the duplicated evidence paragraph is normalized to its single exact raw occurrence.",
    },
    {
        "fact_id": "fact-b402e893845d19ea97df",
        "active_index": 95,
        "expected_question": "How does Rope365 suggest adapting a two-rope box tie's lower wrap to the person being tied?",
        "expected_answer": "Keep it off the elbow and base of the deltoid; place it just below the top wrap or leave a comfortable gap, use a chest wrap if the lower arms cannot tolerate rope, and keep its tension slightly looser than the top wrap.",
        "expected_evidence_sha256": "d506a259c8254022f05e84b9f22bbcc9ddf6c67f00ad3b62bc03b8519108bb06",
        "question": "How does Rope365 suggest adapting a two-rope box tie's lower wrap to the person being tied?",
        "answer": "Rope365 recommends keeping the lower wrap away from the elbow and from pressure at the base of the deltoid, placing it just below the top wrap or leaving a comfortable gap, substituting a chest wrap when the lower arms cannot tolerate rope, and using slightly less tension than the top wrap.",
        "source_document": "data/raw/rope365_1616ffce57d993f3.json",
        "reason_code": "complete_lower_box_tie_wrap_attribution",
        "reason": "The replacement resolves the imperative pronoun, attributes the guidance, and preserves the source-supported placement, accommodation, and relative-tension options.",
    },
    {
        "fact_id": "fact-99adc3e1171124c3c376",
        "active_index": 321,
        "expected_question": "What self-audit does WykD recommend when similar injuries recur across multiple partners and sessions?",
        "expected_answer": "Look for the common factor across the incidents, including the tying practices of the person present in each case.",
        "expected_evidence_sha256": "67cd3f732f0458c9a71d61fe893df19f13bc3748c6f7bbc3683a730f96690273",
        "question": "What self-audit does WykD recommend when similar injuries recur across multiple partners and sessions?",
        "answer": "WykD recommends looking for the common factor across the incidents, including the tying practices of the person present in each case.",
        "source_document": "data/raw/wykd_19d6a26116e26c70.json",
        "reason_code": "complete_repeated_injury_self_audit_attribution",
        "reason": "The replacement turns the source-supported imperative into a standalone attributed answer while retaining the cross-incident common-factor self-audit.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v433 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v434-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v433.jsonl"
    build_baseline(base, inputs / "v433.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v434-observe-", dir=OUT_DIR) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v434-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v433.jsonl"
        build_baseline(base, d / "v433.report.json")
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
        if text_sha256(active["evidence"]) != spec["expected_evidence_sha256"]:
            raise ValueError(f"candidate evidence drift {spec['fact_id']}")
        evidence = spec.get("evidence", active["evidence"])
        source_document = ROOT / spec["source_document"]
        source = json.loads(source_document.read_text())
        if source.get("url") != active["url"]:
            raise ValueError(f"source URL drift {spec['fact_id']}")
        if not evidence or evidence not in source["text"]:
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
                "reviewer": "codex-context-merit-audit-v434",
                "source_lineage": active["source_lineage"],
                "support_type": "manual_paraphrase",
            }
        )
        audits.append(
            {
                "active_answer": active["answer"],
                "active_evidence_sha256": text_sha256(active["evidence"]),
                "active_index": index,
                "active_question": active["question"],
                "audit_index": audit_index,
                "decision": "edit",
                "document_sha256": active["document_sha256"],
                "edited_answer": spec["answer"],
                "edited_evidence_sha256": text_sha256(evidence),
                "edited_question": spec["question"],
                "evidence_normalized": evidence != active["evidence"],
                "fact_id": active["fact_id"],
                "paraphrase_rationale": spec["reason"],
                "projection_lineage": {
                    "active_index": index,
                    "baseline_rows": BASELINE_ROWS,
                    "baseline_sha256": BASELINE_SHA256,
                },
                "reason": spec["reason"],
                "reason_code": spec["reason_code"],
                "review_pass": "safety_due_diligence_and_fit_attribution_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v434",
                "schema": "context-merit-audit-v434",
                "source": active["source"],
                "source_document": spec["source_document"],
                "source_document_file_sha256": file_sha256(source_document),
                "source_support": "manual_source_lineage_and_dataset_context_review",
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
                    "by_decision": {"drop": 0, "edit": 3, "keep": 0},
                    "evidence_normalizations": sum(row["evidence_normalized"] for row in audits),
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
                "schema": "context-merit-audit-report-v434",
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
