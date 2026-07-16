#!/usr/bin/env python3
"""Complete three safety-critical train-only answers with source attribution."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V432 = DATA / "manual_reviews/context_merit_audit_v432"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V432), str(V290)]

import build_context_merit_audit_v432 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v433.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v433.jsonl"
REPORT = OUT_DIR / "report_context_merit_v433.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "6bb26f0a0ef094168bed42826eadd1dc39dbd891d1ff62798032f16250c8becd"
EXPECTED_OUTPUT_SHA256 = "5d21081e044c3f5a2110212dae19e22dfdc08e4b59d483a134871cbeaa9c268c"
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
    V432 / "context_merit_audit_v432.jsonl",
    V432 / "pending_curation_context_merit_v432.jsonl",
    V432 / "report_context_merit_v432.json",
)

SPECS = (
    {
        "fact_id": "fact-5c72b1443ef0f31cb34f",
        "active_index": 56,
        "expected_question": "How does Crash Restraint recommend screening a potential rope partner?",
        "expected_answer": "Check them through the Rope Bottoms' Share Group and ask other local bottoms; neither is foolproof, but both can reduce risk.",
        "question": "How does Crash Restraint recommend screening a potential rope partner?",
        "answer": "Crash Restraint recommends checking a potential partner through the Rope Bottoms' Share Group and asking other local bottoms; neither method is foolproof, but both can reduce risk.",
        "source_document": "data/raw/crash_restraint_getting_started_20260714.json",
        "reason_code": "complete_partner_screening_attribution",
        "reason": "The replacement turns the source-supported imperative into a standalone attributed answer while retaining both screening channels and the no-silver-bullet caveat.",
    },
    {
        "fact_id": "fact-a156169f551a48932a1b",
        "active_index": 315,
        "expected_question": "What safer pivot points does Rope365 recommend when rolling someone in an asymmetric hogtie?",
        "expected_answer": "Use a knee or elbow as the pivot rather than putting substantial body weight on an ankle or wrist.",
        "question": "What safer pivot points does Rope365 recommend when rolling someone in an asymmetric hogtie?",
        "answer": "Rope365 recommends using a knee or elbow as the pivot rather than putting substantial body weight on an ankle or wrist.",
        "source_document": "data/raw/rope_resources_v1/rope365__377d038c4dbf9423e3b6.json",
        "reason_code": "complete_asymmetric_hogtie_pivot_attribution",
        "reason": "The replacement turns the source-supported imperative into a standalone attributed answer while preserving the safer pivot points and the ankle-or-wrist loading warning.",
    },
    {
        "fact_id": "fact-e43d81750c67def8527a",
        "active_index": 491,
        "expected_question": "Why does Rope365 advise against using elastic rope directly on the skin?",
        "expected_answer": "It may create a tourniquet effect and can behave unpredictably.",
        "question": "Why does Rope365 advise against using elastic rope directly on the skin?",
        "answer": "Rope365 warns that elastic rope used directly on the skin may create a tourniquet effect and behave unpredictably.",
        "source_document": "data/raw/rope365_0b4f44c8fabfc202.json",
        "reason_code": "resolve_elastic_rope_skin_risk_subject",
        "reason": "The replacement resolves the ambiguous pronoun and attributes the warning while retaining the source-supported tourniquet and unpredictable-behavior risks.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v432 drift")


def build_projection(out: Path, report: Path) -> None:
    d = out.parent / f".{out.name}.v433-input"
    d.mkdir(parents=True, exist_ok=True)
    base = d / "v432.jsonl"
    build_baseline(base, d / "v432.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v433-observe-", dir=OUT_DIR) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v433-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v432.jsonl"
        build_baseline(base, d / "v432.report.json")
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
        source_document = ROOT / spec["source_document"]
        source = json.loads(source_document.read_text())
        source_text = source["text"]
        if source.get("url") != active["url"]:
            raise ValueError(f"source URL drift {spec['fact_id']}")
        if not evidence or evidence not in source_text:
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
                "reviewer": "codex-context-merit-audit-v433",
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
                "review_pass": "safety_critical_standalone_attribution_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v433",
                "schema": "context-merit-audit-v433",
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
                        {"path": portable(path), "sha256": file_sha256(path)}
                        for path in PRIOR
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
                "schema": "context-merit-audit-report-v433",
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
