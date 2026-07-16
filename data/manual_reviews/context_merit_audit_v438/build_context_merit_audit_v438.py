#!/usr/bin/env python3
"""Complete three actionable technique answers in training data."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V437 = DATA / "manual_reviews/context_merit_audit_v437"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V437), str(V290)]

import build_context_merit_audit_v437 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v438.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v438.jsonl"
REPORT = OUT_DIR / "report_context_merit_v438.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "9183e7422b64678bd5a618b02a5d3ca648bd605d40064f999da2321d6a094385"
EXPECTED_OUTPUT_SHA256 = "4fdfc367a79f238d057116c0995bcc90ba6dd76e4a56ad25b6aadd5beb61d7d4"
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
    V437 / "context_merit_audit_v437.jsonl",
    V437 / "pending_curation_context_merit_v437.jsonl",
    V437 / "report_context_merit_v437.json",
)

SPECS = (
    {
        "fact_id": "fact-9925156a1d2b84d5afe3",
        "active_index": 38,
        "expected_question": "How can bulk be reduced when threading a cinch through a narrow gap?",
        "expected_answer": "Keep the rope in a straight line with enough tension to prevent bunching, and do not place a rope extension in the cinch.",
        "expected_evidence_sha256": "dae9ed1fecff6b9e69a3aa7a5a6c46ea09fbb3280a293a5b039a5da9a47b033b",
        "question": "How can bulk be reduced when threading a cinch through a narrow gap?",
        "answer": "Rope365 recommends keeping the rope straight and sufficiently tensioned to prevent bunching, and avoiding rope extensions within the cinch.",
        "source_document": "data/raw/rope_resources_v1/rope365__4b71bf74a2741be37598.json",
        "reason_code": "complete_narrow_cinch_bulk_reduction_attribution",
        "reason": "The replacement attributes the source-supported procedure while preserving straight rope, sufficient tension, and avoiding an extension in the cinch.",
    },
    {
        "fact_id": "fact-0b3a1e1a96dfaad86289",
        "active_index": 77,
        "expected_question": "How does Rope365 compensate for the unlocked entry of a slipped-half-hitch quick release and then lock its exit?",
        "expected_answer": "Start at a change of direction or with friction at the hitch's origin because the entry is not locked, then place a second half hitch around the first hitch's loop to lock the exit.",
        "expected_evidence_sha256": "5e2e82a0d1f4ae01ff1edec5835cf9ebe48561607d18af123a2d95391fa20a01",
        "question": "How does Rope365 compensate for the unlocked entry of a slipped-half-hitch quick release and then lock its exit?",
        "answer": "Rope365 says to start the slipped half hitch at a change of direction or with friction at its origin because the entry is not locked, then place a second half hitch around the first hitch’s loop to lock the exit.",
        "source_document": "data/raw/rope365_c89abf7c3a5c30e1.json",
        "reason_code": "complete_slipped_half_hitch_locking_attribution",
        "reason": "The replacement attributes the source-supported two-stage procedure while preserving the friction-or-direction entry and second-half-hitch exit lock.",
    },
    {
        "fact_id": "fact-6630ca5813b6bbc49e40",
        "active_index": 89,
        "expected_question": "How does Rope365’s self-evaluation checklist suggest checking a chest harness’s solidity?",
        "expected_answer": "Pull on the harness in different directions and observe how solidly it holds.",
        "expected_evidence_sha256": "5f0f057db83860ff34d99c23f7f4567cb59f1a03444b139027d2c4cd2bff1011",
        "question": "How does Rope365’s self-evaluation checklist suggest checking a chest harness’s solidity?",
        "answer": "Rope365 recommends pulling the harness in different directions and observing how solidly it holds.",
        "source_document": "data/raw/rope365_bac79ac0456a12c1.json",
        "reason_code": "complete_chest_harness_solidity_test_attribution",
        "reason": "The replacement turns the imperative into a standalone attributed answer while preserving the multidirectional pull test and solidity observation.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v437 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v438-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v437.jsonl"
    build_baseline(base, inputs / "v437.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v438-observe-", dir=OUT_DIR) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v438-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v437.jsonl"
        build_baseline(base, d / "v437.report.json")
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
                "reviewer": "codex-context-merit-audit-v438",
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
                "review_pass": "bounded_actionable_technique_standalone_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v438",
                "schema": "context-merit-audit-v438",
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
                "schema": "context-merit-audit-report-v438",
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
