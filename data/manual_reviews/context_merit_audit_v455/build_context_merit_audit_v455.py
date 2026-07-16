#!/usr/bin/env python3
"""Curate six existing knot-mechanics Q&As.

The only candidate base is sealed v454. Two generic or redundant recall rows
are quarantined, while four rows with durable mechanics, terminology, or body-
risk value are retained. No corpus or protected evaluation artifact is read
or changed.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
V454 = DATA / "manual_reviews/context_merit_audit_v454"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V454), str(V290)]

import build_context_merit_audit_v454 as previous
import build_context_merit_audit_v290 as core


AUDIT = HERE / "context_merit_audit_v455.jsonl"
CURATION = HERE / "pending_curation_context_merit_v455.jsonl"
REPORT = HERE / "report_context_merit_v455.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
SOURCE_DOCUMENT = "data/raw/rope_resources_v1/rope365__a53247111d1623ea270f.json"
SOURCE_FILE_SHA256 = "a3c817bcdffe4b4f387c4b8e38bce79fc6f12ef5770a9890d6fdde2ec4cbe94e"
SOURCE_TEXT_SHA256 = "b0b19a7d953de1efe052774f459ef7d121086b3c9fba90dc1381b7c56641161d"
SOURCE_CHARS = 3439
SOURCE_URL = "https://rope365.com/more-knots/"
BASELINE_ROWS = 499
BASELINE_SHA256 = "68246e06a02ba275e33574a0e0818a6209f605f13d73f300e40d6ec6ce11ff4d"
EXPECTED_OUTPUT_SHA256 = "3c12dc3811b8c6b79a7bde5795c27a21ebb2e22073eff036da7b7405279202f6"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 258,
    "equipment_material": 21,
    "resources_general": 77,
    "safety_consent": 85,
    "technique": 75,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 258,
    "equipment_material": 21,
    "resources_general": 77,
    "safety_consent": 86,
    "technique": 74,
}
OWNER_SOURCE = previous.OWNER_SOURCE
OWNER_FACT_IDS = previous.OWNER_FACT_IDS
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v455"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-f3a9f1f3b2b5bfcceb2f": {
        "active_index": 161,
        "question": "What capability does Rope365 use to distinguish an exploding knot from an ordinary quick release?",
        "answer": "An exploding knot releases without leaving a tangle; designs such as the lapp knot can let the rope detach even when tying continued with the working end.",
        "decision": "keep",
        "knot_scope": "functional distinction plus one source-attributed detachment example",
        "reason_code": "keep_exploding_knot_detachment_behavior",
        "reason": "The row teaches the no-tangle functional distinction and the page's concrete lapp-knot behavior after the working end has continued into a tie.",
        "review_class": "exploding_knot_behavior_keep",
    },
    "fact-e02a4b470633be872834": {
        "active_index": 212,
        "question": "What does Rope365 mean by bends, and what tradeoff does it note among them?",
        "answer": "Rope365 calls knots used to extend rope “bends”; some are easier to tie and others sturdier, with the sheet bend noted as popular in rope bondage.",
        "decision": "keep",
        "knot_scope": "source-attributed extension-knot terminology and selection tradeoff",
        "reason_code": "keep_bend_definition_and_tradeoff",
        "reason": "The row combines a useful knot-family definition with ease-versus-sturdiness selection context and clearly attributes the example to Rope365.",
        "review_class": "bend_definition_tradeoff_keep",
    },
    "fact-f85dd797b667780b5fca": {
        "active_index": 213,
        "question": "What does Rope365 recommend doing to learn which knots may be useful in different tying contexts?",
        "answer": "Rope365 recommends learning and practising a few knots, then considering which tying contexts make each knot useful.",
        "decision": "drop",
        "knot_scope": "generic weekly curriculum goal without knot mechanics",
        "reason_code": "quarantine_generic_knot_practice_curriculum_goal",
        "reason": "The row paraphrases the page's generic weekly goal and teaches no knot behavior or selection criterion. The surviving bends row supplies a concrete ease-versus-sturdiness comparison, and the v454 learning-progression row already preserves context-based technique selection.",
        "review_class": "generic_curriculum_goal_drop",
        "substantive_survivors": ["fact-e02a4b470633be872834", "fact-638a33841c6b1bdddd55"],
    },
    "fact-873875ebb1134d37a6ab": {
        "active_index": 242,
        "question": "What functional tradeoffs does Rope365 give for basic slip knots?",
        "answer": "Slip knots are fast and can serve for quick capture or an adjustable loop, but they tighten and are not stable, so those limitations must be considered when choosing where to use them.",
        "decision": "keep",
        "knot_scope": "uses paired with tightening and instability limitations",
        "reason_code": "keep_slip_knot_uses_and_limitations",
        "reason": "The answer pairs every named use with the page's tightening and instability limitations, preventing a context-free recommendation of the construction.",
        "review_class": "slip_knot_tradeoff_keep",
    },
    "fact-3e29b3c630c8583a4755": {
        "active_index": 352,
        "question": "What use and body risk does Rope365 identify for the handcuff knot?",
        "answer": "A handcuff knot can capture two limbs at once, but its slipped loops may tighten when the knot is placed directly on the body.",
        "decision": "keep",
        "knot_scope": "two-limb capture function paired with direct-body tightening risk",
        "reason_code": "keep_handcuff_knot_use_and_body_risk",
        "reason": "The row states the useful function and the construction's body-contact tightening risk together, with no unsupported claim of structural safety.",
        "review_class": "handcuff_knot_risk_keep",
    },
    "fact-6a6f49157591a8d29ea9": {
        "active_index": 397,
        "question": "Which knot does Rope365 suggest for quickly marking the middle of a rope that may be used again later?",
        "answer": "A quick slip knot can mark the rope’s middle for later reuse.",
        "decision": "drop",
        "knot_scope": "isolated knot-name recall for a minor temporary-marking convenience",
        "reason_code": "quarantine_quick_slip_knot_middle_marking_recall",
        "reason": "The row asks for a knot name tied to a minor reuse convenience and duplicates the same evidence already represented by the more substantive slip-knot uses-and-limitations row.",
        "review_class": "minor_marking_recall_drop",
        "substantive_survivors": ["fact-873875ebb1134d37a6ab"],
    },
}


def requested_urls() -> set[str]:
    resources = json.loads(RESOURCE_MANIFEST.read_text())["resources"]
    return {
        url for resource in resources
        for url in (resource["canonical_url"], resource.get("recommendation_url")) if url
    }


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v454 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v455-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v454.jsonl"
    build_baseline(base, inputs / "v454.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def validate_source(row: dict, evidence: str) -> None:
    source_path = ROOT / SOURCE_DOCUMENT
    if file_sha256(source_path) != SOURCE_FILE_SHA256:
        raise ValueError(f"source file drift: {row['fact_id']}")
    document = json.loads(source_path.read_text())
    if document["url"] != SOURCE_URL or row["url"] != SOURCE_URL:
        raise ValueError(f"source URL drift: {row['fact_id']}")
    if len(document["text"]) != SOURCE_CHARS or text_sha256(document["text"]) != SOURCE_TEXT_SHA256:
        raise ValueError(f"source text drift: {row['fact_id']}")
    if row["document_sha256"] != SOURCE_TEXT_SHA256 or evidence not in document["text"]:
        raise ValueError(f"evidence absent from full source snapshot: {row['fact_id']}")


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v455-observe-", dir=HERE) as tmp:
        directory = Path(tmp)
        out = directory / "out.jsonl"
        projection_report = directory / "out.report.json"
        datasets, reports = [], []
        for _ in (1, 2):
            build_projection(out, projection_report)
            datasets.append(out.read_bytes())
            reports.append(projection_report.read_bytes())
        rows = read_jsonl(out)
        return {
            "after": conservative_capacity(rows),
            "before": conservative_capacity(before),
            "dataset_equal": datasets[0] == datasets[1],
            "eval": json.loads(reports[0])["eval_fact_count"],
            "report_equal": reports[0] == reports[1],
            "rows": len(rows),
            "sha256": hashlib.sha256(datasets[0]).hexdigest(),
        }


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v455-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v454.jsonl"
        build_baseline(base, directory / "v454.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}

    audits, curations = [], []
    for audit_index, (fact_id, spec) in enumerate(SPECS.items(), 1):
        active_index, row = by_fact[fact_id]
        if active_index != spec["active_index"]:
            raise ValueError(f"active index drift: {fact_id}")
        if (row["question"], row["answer"]) != (spec["question"], spec["answer"]):
            raise ValueError(f"active Q&A drift: {fact_id}")
        evidence = row["evidence"]
        validate_source(row, evidence)

        decision = spec["decision"]
        if decision == "drop":
            curations.append({
                "action": "drop", "expected_answer": row["answer"],
                "expected_question": row["question"], "fact_id": fact_id,
                "reason": spec["reason"], "reason_code": spec["reason_code"],
                "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
            })
        elif decision != "keep":
            raise ValueError(f"unsupported decision: {decision}")

        audit = {
            "active_answer": row["answer"], "active_index": active_index,
            "active_question": row["question"], "audit_index": audit_index,
            "decision": decision, "document_sha256": row["document_sha256"],
            "fact_id": fact_id, "knot_scope": spec["knot_scope"],
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_class": spec["review_class"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v455",
            "source": row["source"], "source_document": SOURCE_DOCUMENT,
            "source_document_chars": SOURCE_CHARS,
            "source_document_file_sha256": SOURCE_FILE_SHA256,
            "source_document_text_sha256": SOURCE_TEXT_SHA256,
            "source_support": "full_snapshot_review",
            "support_evidence": evidence, "support_evidence_sha256": text_sha256(evidence),
            "url": row["url"],
        }
        if spec.get("substantive_survivors"):
            audit["substantive_survivors"] = spec["substantive_survivors"]
        audits.append(audit)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection: {observation}")
    if (observation["rows"], observation["eval"]) != (497, 612):
        raise ValueError(f"projection count drift: {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift: {observation}")
    if EXPECTED_CAPACITY_AFTER is not None and observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift: {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"projection hash drift: {observation}")

    with tempfile.TemporaryDirectory(prefix=".v455-resource-check-", dir=HERE) as tmp:
        projected = Path(tmp) / "projected.jsonl"
        build_projection(projected, Path(tmp) / "projected.report.json")
        projected_rows = read_jsonl(projected)
        blob = projected.read_text()
    urls = requested_urls()
    if len(urls) != 24 or any(url not in blob for url in urls):
        raise ValueError("requested resource coverage drift")
    if {row["fact_id"] for row in projected_rows if row.get("source") == OWNER_SOURCE} != set(OWNER_FACT_IDS):
        raise ValueError("owner resource fact drift")

    curation_counts = Counter(row["action"] for row in curations)
    REPORT.write_text(json.dumps({
        "audit": {
            "by_decision": dict(Counter(row["decision"] for row in audits)),
            "by_knot_scope": dict(Counter(row["knot_scope"] for row in audits)),
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "path": portable(AUDIT), "rows": len(audits), "sha256": file_sha256(AUDIT),
        },
        "conservative_capacity": {
            "after": observation["after"], "before": observation["before"],
            "delta": {key: observation["after"][key] - observation["before"][key] for key in observation["before"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "curation_outcome": {
            "generic_or_redundant_rows_quarantined": 2,
            "mechanics_terminology_or_risk_rows_kept": 4,
        },
        "isolated_build_projection": {
            "automated_projection_runs": 2, "new_additions_applied": 0,
            "output_rows": observation["rows"], "output_sha256": observation["sha256"],
            "repeat_dataset_byte_identical": observation["dataset_equal"],
            "repeat_projection_report_byte_identical": observation["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observation["eval"],
        },
        "new_pending_curation": {
            "by_action": dict(curation_counts), "decisions": len(curations),
            "path": portable(CURATION), "sha256": file_sha256(CURATION),
        },
        "requested_resource_coverage": {
            "manifest_urls": len(urls), "manifest_urls_present": sum(url in blob for url in urls),
            "owner_resource_facts": len(OWNER_FACT_IDS),
        },
        "source_boundary": {
            "new_or_pending_corpus_inputs": 0,
            "protected_eval_heldout_holdout_ood_shadow_probe_rows_opened": 0,
            "training_or_eval_artifacts_modified": 0,
        },
        "source_snapshot_inventory": {
            "documents": 1, "paths": [SOURCE_DOCUMENT], "reviewed_rows": len(SPECS),
            "total_unique_characters": SOURCE_CHARS,
        },
        "schema": "context-merit-audit-report-v455",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; no corpus output was ingested by v455.",
            "derived_qa": "Distinct first-class training layer; v455 changes only existing sealed Q&A using attached evidence and one pinned source snapshot.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to cleaned source documents through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
