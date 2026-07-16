#!/usr/bin/env python3
"""Complete three equipment, material, and fit answers in training data."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V438 = DATA / "manual_reviews/context_merit_audit_v438"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V438), str(V290)]

import build_context_merit_audit_v438 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v439.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v439.jsonl"
REPORT = OUT_DIR / "report_context_merit_v439.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "4fdfc367a79f238d057116c0995bcc90ba6dd76e4a56ad25b6aadd5beb61d7d4"
EXPECTED_OUTPUT_SHA256 = "8fd31994bf89dfa01dd3f1baae3b93ffa273cba9c325417ff11275eac5e0f8ad"
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
    V438 / "context_merit_audit_v438.jsonl",
    V438 / "pending_curation_context_merit_v438.jsonl",
    V438 / "report_context_merit_v438.json",
)

SPECS = (
    {
        "fact_id": "fact-cbd4904dd0d50fd3d759",
        "active_index": 42,
        "expected_question": "How can shoes change a foot tie according to Rope365?",
        "expected_answer": "They change how the rope feels and how the ankles behave, add some foot protection, and can provide heel attachment points.",
        "expected_evidence_sha256": "8bcb98b99820eb18d523a71ace064167842a091634aab80674b1d3fdf4775ee3",
        "question": "How can shoes change a foot tie according to Rope365?",
        "answer": "Rope365 says shoes change how the rope feels and how the ankles behave, add some foot protection, and can make heels useful attachment points.",
        "source_document": "data/raw/rope_resources_v1/rope365__79dd5614e734dbf898fa.json",
        "reason_code": "resolve_shoe_effects_on_foot_tie_subject",
        "reason": "The replacement resolves the ambiguous pronoun and attributes the explanation while preserving sensation, ankle behavior, protection, and heel attachment points.",
    },
    {
        "fact_id": "fact-7c8143d2ac8a0e3f642b",
        "active_index": 350,
        "expected_question": "What tub-wash procedure does deGiotto specify for its own jute rope?",
        "expected_answer": "Use cool water and one-half teaspoon of free-and-clear detergent, stir in the unbundled rope, soak it for 20 minutes, rinse with cool water, then follow deGiotto's special jute drying and treatment directions.",
        "expected_evidence_sha256": "98d21b2f4fe4ccab73f6e5eacd11b92306b44344078163312599f0a1d840edf6",
        "question": "What tub-wash procedure does deGiotto specify for its own jute rope?",
        "answer": "deGiotto specifies filling a tub or bucket with cool water, adding one-half teaspoon of free-and-clear detergent, stirring in unbundled rope, soaking it for 20 minutes, rinsing with cool water, then following its special jute drying and treatment directions.",
        "source_document": "data/raw/degiotto_care_maintenance_20260714.json",
        "reason_code": "complete_degiotto_jute_wash_attribution",
        "reason": "The replacement turns the imperative into a standalone attributed procedure while preserving the water, detergent, unbundling, soak, rinse, and special aftercare details.",
    },
    {
        "fact_id": "fact-f6b7cc342d8eab96ec81",
        "active_index": 500,
        "expected_question": "Why does Rope365 recommend the overhand hank for coiling rope?",
        "expected_answer": "It is easy and fast, and it avoids adding unnecessary twists and stress during storage.",
        "expected_evidence_sha256": "0d613543442d7a3110754a23571b0959b7c0b7638039be30554cc4a42c7a8229",
        "question": "Why does Rope365 recommend the overhand hank for coiling rope?",
        "answer": "Rope365 recommends the overhand hank because it is easy and fast and avoids adding unnecessary twists and stress during storage.",
        "source_document": "data/raw/rope365_d9c48a4547717047.json",
        "reason_code": "resolve_overhand_hank_rationale_subject",
        "reason": "The replacement resolves the ambiguous pronoun and attributes the rationale while preserving speed, ease, and reduced storage stress and twisting.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v438 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v439-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v438.jsonl"
    build_baseline(base, inputs / "v438.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v439-observe-", dir=OUT_DIR) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v439-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v438.jsonl"
        build_baseline(base, d / "v438.report.json")
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
                "reviewer": "codex-context-merit-audit-v439",
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
                "review_pass": "bounded_equipment_material_fit_standalone_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v439",
                "schema": "context-merit-audit-v439",
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
                "schema": "context-merit-audit-report-v439",
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
