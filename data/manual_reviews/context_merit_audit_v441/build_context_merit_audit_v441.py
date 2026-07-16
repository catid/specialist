#!/usr/bin/env python3
"""Make three underrepresented-source answers standalone and specific."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V440 = DATA / "manual_reviews/context_merit_audit_v440"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V440), str(V290)]

import build_context_merit_audit_v440 as previous
import build_context_merit_audit_v290 as core

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v441.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v441.jsonl"
REPORT = OUT_DIR / "report_context_merit_v441.json"
BASELINE_ROWS = 531
BASELINE_SHA256 = "8988b14443dd3ab51be98be24b1aec6bc82ec4233241f3422cb0887c842af078"
EXPECTED_OUTPUT_SHA256 = "c5a53b330bf4c338aeab2d78e074a24b2d353d81465ec0265e443300fadbb410"
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
    V440 / "context_merit_audit_v440.jsonl",
    V440 / "pending_curation_context_merit_v440.jsonl",
    V440 / "report_context_merit_v440.json",
)

SPECS = (
    {
        "fact_id": "fact-faf3903dfe22d445fd15",
        "active_index": 3,
        "expected_question": "According to Esinem’s tutorial announcement, what is Aisatsu shibari designed to gauge?",
        "expected_answer": "It gauges a rope partner’s overall readiness for an emotional exchange and helps open a dialogue about the session.",
        "expected_evidence_sha256": "ed419fc381291d5954d70b007ca4518132fa903cb938299167bb580e4e10cf57",
        "question": "According to Esinem’s tutorial announcement, what is Aisatsu shibari designed to gauge?",
        "answer": "Esinem says Aisatsu shibari is designed to gauge a rope partner’s overall readiness for an emotional exchange and gently open a dialogue to align the session with their desires.",
        "source_document": "data/raw/esinem_63054bcce13821a0.json",
        "reason_code": "complete_aisatsu_readiness_and_dialogue_subject",
        "reason": "The replacement resolves the ambiguous pronoun, attributes the claim, and retains both the emotional-readiness assessment and the evidence's desire-aligned dialogue purpose.",
    },
    {
        "fact_id": "fact-196eb40865b3f9984eba",
        "active_index": 41,
        "expected_question": "How can Shibari Study members access tutorials for practice away from a computer?",
        "expected_answer": "They can use the mobile app and either stream tutorials or download them to a device for on-the-go practice.",
        "expected_evidence_sha256": "bb88e44366f3c5eb0460a5f031376ee7e2a1250dc4324a50cbaae2194c664ee3",
        "question": "How can Shibari Study members access tutorials for practice away from a computer?",
        "answer": "Shibari Study members can use its mobile app to stream tutorials or download them to a device for on-the-go practice.",
        "source_document": "data/raw/rope_resources_v1/shibari_study__7b7df424f9d0383080e9.json",
        "reason_code": "resolve_mobile_tutorial_access_subject",
        "reason": "The replacement resolves the ambiguous pronoun with the membership subject while preserving the app, streaming, downloading, device, and on-the-go access details.",
    },
    {
        "fact_id": "fact-369b257dd923aefc33ad",
        "active_index": 58,
        "expected_question": "How does deGiotto describe the length, material, end finish, and delivered condition of its 30-foot Jute Shibari Rope?",
        "expected_answer": "It is 30 feet of 6mm jute rope finished with overhand knots, and deGiotto says it arrives conditioned, buttered, and ready to use.",
        "expected_evidence_sha256": "66f406d3b47e89b7c85f5e470975256f474dde1e49b221f231599b829e112a82",
        "question": "How does deGiotto describe the length, material, end finish, and delivered condition of its 30-foot Jute Shibari Rope?",
        "answer": "deGiotto describes the product as 30 feet of premium 6 mm jute bondage rope finished with overhand knots and delivered conditioned, buttered, and ready to use.",
        "source_document": "data/raw/degiotto_jute_30_product_20260714.json",
        "reason_code": "complete_degiotto_jute_product_subject",
        "reason": "The replacement resolves the ambiguous pronoun, attributes the product description, and retains length, diameter, material, end finish, and delivered condition while restoring the evidence's premium qualifier.",
    },
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v440 drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v441-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v440.jsonl"
    build_baseline(base, inputs / "v440.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v441-observe-", dir=OUT_DIR) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v441-base-", dir=OUT_DIR) as tmp:
        d = Path(tmp)
        base = d / "v440.jsonl"
        build_baseline(base, d / "v440.report.json")
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
                "reviewer": "codex-context-merit-audit-v441",
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
                "review_pass": "bounded_underrepresented_source_standalone_train_only_cleanup",
                "reviewed_at": "2026-07-16",
                "reviewer": "codex-context-merit-audit-v441",
                "schema": "context-merit-audit-v441",
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
                "schema": "context-merit-audit-report-v441",
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
