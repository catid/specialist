#!/usr/bin/env python3
"""Make three style and technique answers independently interpretable."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V441 = DATA / "manual_reviews/context_merit_audit_v441"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V441), str(V290)]

import build_context_merit_audit_v441 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v442.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v442.jsonl"
REPORT = OUT_DIR / "report_context_merit_v442.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "c5a53b330bf4c338aeab2d78e074a24b2d353d81465ec0265e443300fadbb410"
EXPECTED_OUTPUT_SHA256 = "048688bd57829c70101fd0003116f4cef0ed37df3cd4b6d443f4e3b0ebd74555"
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
    V441 / "context_merit_audit_v441.jsonl",
    V441 / "pending_curation_context_merit_v441.jsonl",
    V441 / "report_context_merit_v441.json",
)

SPECS = (
    {
        "fact_id": "fact-b26eca627e566932f296",
        "active_index": 54,
        "expected_question": "How does BightBound describe its rope style, and why does it generally avoid Japanese terminology?",
        "expected_answer": "It teaches a modern fusion heavily inspired by Japanese Kinbaku, often called Shibari in the West, but says Japanese terms are frequently misused and can approach cultural appropriation.",
        "expected_evidence_sha256": "7dd86c898f1e33bcb56cc46f867d7edcde32a0c34508fc983cdf479f861ba14e",
        "question": "How does BightBound describe its rope style, and why does it generally avoid Japanese terminology?",
        "answer": "BightBound describes its style as a modern fusion heavily inspired by Japanese Kinbaku, often called Shibari in the West; it generally avoids Japanese terms because it says they are often misused and can approach cultural appropriation.",
        "source_document": "data/raw/bightbound_faq_20260714.json",
        "reason_code": "resolve_bightbound_style_and_terminology_subject",
        "reason": "The replacement resolves the ambiguous pronoun, attributes both parts of the answer, and retains the style lineage and stated cultural-appropriation rationale.",
    },
    {
        "fact_id": "fact-e22f074b6357dee3e385",
        "active_index": 204,
        "expected_question": "What does a cinch add to a two-column rope structure?",
        "expected_answer": "It catches rope through the gap between the columns to prevent rotation or sliding and make the structure more solid.",
        "expected_evidence_sha256": "23c32e730fe90624ab805b93660534e5181b66bc91c8d46f71d92b954193ff8f",
        "question": "What does a cinch add to a two-column rope structure?",
        "answer": "A cinch passes rope through the gap between two columns and catches rope on the other side, preventing rotation or sliding and making the structure more solid.",
        "source_document": "data/raw/rope_resources_v1/rope365__e156eea7b8dfda8b39a9.json",
        "reason_code": "resolve_cinch_action_subject",
        "reason": "The replacement turns the pronoun-led answer into a standalone definition while preserving the routing action and the evidence-supported rotation, sliding, and solidity effects.",
    },
    {
        "fact_id": "fact-a722d3aaab3ebe2a16f2",
        "active_index": 258,
        "expected_question": "What happens if a quick-release tail continues into the rest of a tie and the quick release is then undone?",
        "expected_answer": "It adds some slack but does not completely untie the continuing tie.",
        "expected_evidence_sha256": "d98028d06bce22df07457c5eab6356d043153bc4088eaf8005085078d8a50421",
        "question": "What happens if a quick-release tail continues into the rest of a tie and the quick release is then undone?",
        "answer": "Rope365 says undoing a quick release after its tail has continued into the rest of a tie adds slack but does not completely untie the continuing tie.",
        "source_document": "data/raw/rope365_c89abf7c3a5c30e1.json",
        "reason_code": "resolve_continuing_quick_release_effect_subject",
        "reason": "The replacement resolves the ambiguous pronoun and attributes the effect while preserving the source's exact distinction between added slack and complete untying.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v441 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v442-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v441.jsonl"
    build_baseline(base, inputs / "v441.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v442-observe-", dir=OUT_DIR) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v442-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v441.jsonl"
        build_baseline(base, d / "v441.report.json")
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
                "reviewer": "codex-context-merit-audit-v442",
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
                "review_pass": "bounded_style_technique_standalone_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v442",
                "schema": "context-merit-audit-v442",
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
                "schema": "context-merit-audit-report-v442",
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
