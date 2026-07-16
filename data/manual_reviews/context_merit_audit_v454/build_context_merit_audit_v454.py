#!/usr/bin/env python3
"""Curate five existing single-column technique Q&As.

The only candidate base is sealed v453. Three rows receive source-attributed
construction or tradeoff repairs, and two already bounded technique rows are
retained. No corpus or protected evaluation artifact is read or changed.
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
V453 = DATA / "manual_reviews/context_merit_audit_v453"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V453), str(V290)]

import build_context_merit_audit_v453 as previous
import build_context_merit_audit_v290 as core
from qa_quality import stable_fact_id


AUDIT = HERE / "context_merit_audit_v454.jsonl"
CURATION = HERE / "pending_curation_context_merit_v454.jsonl"
REPORT = HERE / "report_context_merit_v454.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
SOURCE_DOCUMENT = "data/raw/rope_resources_v1/rope365__772d43d614d87876db29.json"
SOURCE_FILE_SHA256 = "9711683cee312ca051f141de8a0aa4d39f9a66daff92ef4febc70f8057e3cb40"
SOURCE_TEXT_SHA256 = "71c8e316396b9026b3e45da9d2142a29bd4abe6cb654ea451d260f535ce5354a"
SOURCE_CHARS = 5674
SOURCE_URL = "https://rope365.com/more-single-column-ties/"
BASELINE_ROWS = 499
BASELINE_SHA256 = "1245a8c31f5c984f8af9aa0f5af87927b3014de6c27ac60a997d165833446491"
EXPECTED_OUTPUT_SHA256 = "68246e06a02ba275e33574a0e0818a6209f605f13d73f300e40d6ec6ce11ff4d"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 258,
    "equipment_material": 21,
    "resources_general": 77,
    "safety_consent": 84,
    "technique": 76,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 258,
    "equipment_material": 21,
    "resources_general": 77,
    "safety_consent": 85,
    "technique": 75,
}
OWNER_SOURCE = previous.OWNER_SOURCE
OWNER_FACT_IDS = previous.OWNER_FACT_IDS
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v454"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


SPECS = {
    "fact-638a33841c6b1bdddd55": {
        "active_index": 260,
        "question": "What learning approach does Rope365 recommend before comparing many single-column ties?",
        "answer": "Master one single-column tie first; later, compare methods and learn which contexts suit each one.",
        "decision": "keep",
        "technique_scope": "progressive learning and context-based knot selection",
        "reason_code": "keep_single_column_learning_progression",
        "reason": "The row gives a practical progression from mastering one tie to comparing techniques by context, without asserting that one construction is universally best.",
        "review_class": "learning_progression_keep",
    },
    "fact-f9e47fcb67e8d5f788cc": {
        "active_index": 333,
        "question": "What tradeoffs does Rope365 identify for quick-release single-column ties?",
        "answer": "They are less stable, cannot be locked, and leave the bight unavailable as an attachment point even though they can be released swiftly.",
        "decision": "edit",
        "edited_question": "For the quick-release single-column variants Rope365 describes, what emergency benefit and structural compromises does the page identify?",
        "edited_answer": "They can be undone swiftly in an emergency, but the page says they are less stable, cannot be locked, and leave the bight unavailable as an attachment point.",
        "technique_scope": "page-specific pulled-bight or daisy-chain quick-release variants",
        "reason_code": "scope_quick_release_tradeoffs_to_described_variants",
        "reason": "The replacement keeps the emergency benefit and all three compromises while limiting them to the page's described variants rather than every possible quick-release construction.",
        "review_class": "quick_release_scope_edit",
    },
    "fact-f835f5544918b44f7ce6": {
        "active_index": 334,
        "question": "What tradeoffs follow from Rope365's lark’s-head single-column tie having no bight?",
        "answer": "It has a cleaner look, but cannot be untied from or attached through a bight and takes longer to tie because it needs two full rope pull-throughs.",
        "decision": "keep",
        "technique_scope": "handling and attachment tradeoffs of one specified no-bight construction",
        "reason_code": "keep_larks_head_no_bight_tradeoff",
        "reason": "The question is already limited to Rope365's specified construction and the answer gives appearance, untying, attachment, and tying-time tradeoffs without repeating the page's absolute stability claim.",
        "review_class": "no_bight_tradeoff_keep",
    },
    "fact-aaf15ae2047b2bec3678": {
        "active_index": 361,
        "question": "When can Rope365’s reversed French bowline still capsize?",
        "answer": "It can still capsize under heavy loads or when used with slippery rope.",
        "decision": "edit",
        "edited_question": "How does Rope365 characterize the reversed French bowline’s advantages and failure limits?",
        "edited_answer": "The page describes it as fast, more solid than a square or granny knot, slightly trickier to tie, and easy to untie under tension, but warns that it can still capsize under heavy loads or with slippery rope.",
        "technique_scope": "source-attributed performance tradeoff and capsize conditions",
        "reason_code": "restore_reversed_french_bowline_tradeoff_context",
        "reason": "The replacement retains both capsize conditions and restores the construction's comparative benefits and handling cost, making the warning useful for selection rather than isolated failure-mode recall.",
        "review_class": "bowline_tradeoff_edit",
    },
    "fact-b75ced13031c2bd49793": {
        "active_index": 363,
        "question": "When does Rope365 say sliding cuffs should be avoided?",
        "answer": "Rope365 says sliding cuffs should be avoided when structural stability is required.",
        "decision": "edit",
        "edited_question": "Why does Rope365 say its sliding-cuff examples should not be used where structural stability is required?",
        "edited_answer": "The cuffs are pre-tied for quick entry and release and rely on rope friction to stay in place, so the page limits them to uses where structural stability is not required.",
        "technique_scope": "construction rationale for a non-structural-use limitation",
        "reason_code": "add_sliding_cuff_friction_and_speed_context",
        "reason": "The replacement explains the pre-tied, friction-dependent construction and speed tradeoff behind the source's limitation instead of teaching a context-free prohibition.",
        "review_class": "sliding_cuff_construction_edit",
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
        raise ValueError("v453 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v454-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v453.jsonl"
    build_baseline(base, inputs / "v453.report.json")
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
    with tempfile.TemporaryDirectory(prefix=".v454-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v454-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v453.jsonl"
        build_baseline(base, directory / "v453.report.json")
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
        if decision == "edit":
            curations.append({
                "action": "edit", "answer": spec["edited_answer"],
                "document_sha256": row["document_sha256"], "evidence": evidence,
                "evidence_url": row["url"], "expected_answer": row["answer"],
                "expected_question": row["question"], "fact_id": fact_id,
                "paraphrase_rationale": spec["reason"], "question": spec["edited_question"],
                "reason": spec["reason"], "reason_code": spec["reason_code"],
                "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
                "source_lineage": {"source_document": SOURCE_DOCUMENT},
                "support_type": "manual_paraphrase",
            })
        elif decision != "keep":
            raise ValueError(f"unsupported decision: {decision}")

        audit = {
            "active_answer": row["answer"], "active_index": active_index,
            "active_question": row["question"], "audit_index": audit_index,
            "decision": decision, "document_sha256": row["document_sha256"],
            "fact_id": fact_id, "technique_scope": spec["technique_scope"],
            "projection_lineage": {
                "baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256,
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_class": spec["review_class"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v454",
            "source": row["source"], "source_document": SOURCE_DOCUMENT,
            "source_document_chars": SOURCE_CHARS,
            "source_document_file_sha256": SOURCE_FILE_SHA256,
            "source_document_text_sha256": SOURCE_TEXT_SHA256,
            "source_support": "manual_paraphrase" if decision == "edit" else "full_snapshot_review",
            "support_evidence": evidence, "support_evidence_sha256": text_sha256(evidence),
            "url": row["url"],
        }
        if decision == "edit":
            audit.update({
                "edited_answer": spec["edited_answer"],
                "edited_fact_id": stable_fact_id(spec["edited_question"], spec["edited_answer"]),
                "edited_question": spec["edited_question"],
            })
        audits.append(audit)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection: {observation}")
    if (observation["rows"], observation["eval"]) != (499, 612):
        raise ValueError(f"projection count drift: {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift: {observation}")
    if EXPECTED_CAPACITY_AFTER is not None and observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift: {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"projection hash drift: {observation}")

    with tempfile.TemporaryDirectory(prefix=".v454-resource-check-", dir=HERE) as tmp:
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
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "by_technique_scope": dict(Counter(row["technique_scope"] for row in audits)),
            "path": portable(AUDIT), "rows": len(audits), "sha256": file_sha256(AUDIT),
        },
        "conservative_capacity": {
            "after": observation["after"], "before": observation["before"],
            "delta": {key: observation["after"][key] - observation["before"][key] for key in observation["before"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "curation_outcome": {
            "construction_or_tradeoff_repairs": 3,
            "bounded_technique_rows_kept": 2,
            "rows_quarantined": 0,
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
        "schema": "context-merit-audit-report-v454",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "training_layer_contract": {
            "cleaned_markdown_site_corpora": "Distinct first-class future training artifacts; no corpus output was ingested by v454.",
            "derived_qa": "Distinct first-class training layer; v454 changes only existing sealed Q&A using attached evidence and one pinned source snapshot.",
            "deduplication_policy": "Do not collapse a cleaned Markdown document and its derived Q&A merely because they express the same source knowledge.",
            "provenance_requirement": "Link future derived Q&A to cleaned source documents through stable source URL and document or snapshot hashes.",
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
