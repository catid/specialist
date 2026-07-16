#!/usr/bin/env python3
"""Complete three high-value safety and technique answers in training data."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V436 = DATA / "manual_reviews/context_merit_audit_v436"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V436), str(V290)]

import build_context_merit_audit_v436 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v437.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v437.jsonl"
REPORT = OUT_DIR / "report_context_merit_v437.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "9f1b6c4cf21bef4ecbefb33cf58063dcec1ce44f52efa9cfb501a3aa2e9eb7f8"
EXPECTED_OUTPUT_SHA256 = "9183e7422b64678bd5a618b02a5d3ca648bd605d40064f999da2321d6a094385"
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
    V436 / "context_merit_audit_v436.jsonl",
    V436 / "pending_curation_context_merit_v436.jsonl",
    V436 / "report_context_merit_v436.json",
)

SPECS = (
    {
        "fact_id": "fact-bcdb3e9358e31e3a58e4",
        "active_index": 159,
        "expected_question": "Under what condition does Rope365 say a half-hitch inline cuff remains stable?",
        "expected_answer": "It will be stable as long as tension is maintained in both the origin and exit rope.",
        "expected_evidence_sha256": "b501dd40365db7a7fefaeedf71968956365cf2451955555df9f69532bfe4d347",
        "question": "Under what condition does Rope365 say a half-hitch inline cuff remains stable?",
        "answer": "Rope365 says a half-hitch inline cuff remains stable only while tension is maintained in both the origin and exit rope.",
        "source_document": "data/raw/rope365_5c01d29a044cabf7.json",
        "reason_code": "resolve_half_hitch_inline_cuff_subject",
        "reason": "The replacement resolves the ambiguous pronoun and attributes the condition while preserving the required tension in both rope directions.",
    },
    {
        "fact_id": "fact-9ac5736f0b520782b2c9",
        "active_index": 237,
        "expected_question": "What does Tessin Doyama urge a person being tied to do when something feels wrong?",
        "expected_answer": "They should have the courage to stop the tying.",
        "expected_evidence_sha256": "89963081e990443e88e7eb04d0d4e5c73cdb89580ddab95c592dded1fbc8fd4f",
        "question": "What does Tessin Doyama urge a person being tied to do when something feels wrong?",
        "answer": "Tessin Doyama urges a person being tied to have the courage to stop when something feels wrong.",
        "source_document": "data/raw/kinbakutoday_43095aa305e35a41.json",
        "reason_code": "resolve_stop_when_wrong_actor",
        "reason": "The replacement resolves the ambiguous pronoun and attributes the safety guidance without changing the stop-when-something-feels-wrong claim.",
    },
    {
        "fact_id": "fact-467d6b69f8c76b6efb08",
        "active_index": 357,
        "expected_question": "What two limitations does Rope365 identify for slip knots?",
        "expected_answer": "They tighten and are not stable.",
        "expected_evidence_sha256": "ae042301b8e30bcbf5b1f2a09f1b689bcccda53e6bbd9b3d48db92aa43ea48f5",
        "question": "What two limitations does Rope365 identify for slip knots?",
        "answer": "Rope365 says slip knots tighten and are not stable.",
        "source_document": "data/raw/rope365_25f1b23eb40be00e.json",
        "reason_code": "resolve_slip_knot_limitations_subject",
        "reason": "The replacement resolves the ambiguous pronoun and attributes both source-supported limitations without broadening the claim.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v436 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v437-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v436.jsonl"
    build_baseline(base, inputs / "v436.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v437-observe-", dir=OUT_DIR) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v437-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v436.jsonl"
        build_baseline(base, d / "v436.report.json")
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
                "reviewer": "codex-context-merit-audit-v437",
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
                "review_pass": "bounded_safety_technique_standalone_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v437",
                "schema": "context-merit-audit-v437",
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
                "schema": "context-merit-audit-report-v437",
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
