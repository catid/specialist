#!/usr/bin/env python3
"""Quarantine remaining site-metadata recall and one awkward pending-source row.

The pass scans the complete sealed v443 candidate.  Owner-requested resource
recommendations are retained even when they include locations or URLs.  Five
non-owner navigation/index rows and one awkward BightBound row are dropped;
no replacement fact is authored from a site whose dense corpus is pending.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
HERE = Path(__file__).resolve().parent
V443 = DATA / "manual_reviews/context_merit_audit_v443"
V290 = DATA / "manual_reviews/context_merit_audit_v290"
sys.path[:0] = [str(ROOT), str(V443), str(V290)]

import build_context_merit_audit_v443 as previous
import build_context_merit_audit_v290 as core


AUDIT = HERE / "context_merit_audit_v444.jsonl"
CURATION = HERE / "pending_curation_context_merit_v444.jsonl"
REPORT = HERE / "report_context_merit_v444.json"
RESOURCE_MANIFEST = ROOT / "sources/rope_resources_v1.json"
SITE_CORPUS_QUEUE = ROOT / "sources/site_corpus_queue_v1.json"
BASELINE_ROWS = 524
BASELINE_SHA256 = "9f8e28292f65b4c8f0928f3fbde353a431686753bb3af42435b2a1d43c2b4d0d"
EXPECTED_OUTPUT_SHA256 = "41c8df268b82bfc33a4f9f74adfbdd5519aa892464569937afeb39cbb0a54565"
EXPECTED_CAPACITY_BEFORE = {
    "conflict_units": 266,
    "equipment_material": 23,
    "resources_general": 85,
    "safety_consent": 86,
    "technique": 72,
}
EXPECTED_CAPACITY_AFTER = {
    "conflict_units": 261,
    "equipment_material": 22,
    "resources_general": 81,
    "safety_consent": 86,
    "technique": 72,
}
OWNER_SOURCE = "owner_curated_rope_resource_directory"
REVIEWED_AT = "2026-07-16"
REVIEWER = "codex-context-merit-audit-v444"

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
portable = previous.portable
conservative_capacity = previous.conservative_capacity


OWNER_FACT_IDS = (
    "fact-966eb937bef6831c819a", "fact-433bdbf1247c4ed36895",
    "fact-28f8d3225998dc803502", "fact-5da2e7fcfd2ad0aac48e",
    "fact-250efee9a0d65d87955f", "fact-63e179cc596d19256952",
    "fact-5dcda16beacf5b48a099", "fact-be942a7b2387eb374e0d",
    "fact-7b9730efad5e1f58d941", "fact-697f713f2af3f44c8639",
    "fact-c582f3b8edbba190cec5", "fact-3d55fcb3448d57ad58c1",
    "fact-24d10e1ec7e24079ee67", "fact-749ad10e8c33dcc729eb",
    "fact-b04f011ad190ffabcb22", "fact-feb8f5ee5bdd7a59339f",
    "fact-7997bd2f70052a765e40", "fact-95c6c6763c0722f57258",
    "fact-b18e8884104d75e6bf42", "fact-2d054b924311ba9b82a2",
    "fact-9643a3209f4eed0c63ac", "fact-64d5128c1a734c3d20a1",
    "fact-08545681242a0cab6bc8", "fact-423f9123655368882738",
)


DROP_SPECS = {
    "fact-160eb6a8334e5ec8a9cf": {
        "active_index": 212,
        "question": "What does BightBound prioritize in its teaching, and who might prefer a different resource?",
        "answer": "It combines fundamental technique—including why it works, not only what to do—with play, interaction, connection, and shared emotional experience; it says people seeking purely technical or artistic instruction may prefer elsewhere.",
        "reason_code": "quarantine_awkward_pending_dense_corpus_answer",
        "reason": "The answer is pronoun-led and ends with the awkward phrase 'prefer elsewhere.' BightBound's dense corpus is still pending, so this pass quarantines the row instead of authoring a replacement.",
        "review_class": "awkward_tranche_drop",
    },
    "fact-3e3ce74ba929fd66c5d8": {
        "active_index": 284,
        "question": "What official A-frame equipment documents does X-POLE's support page provide?",
        "answer": "Setup guidance, load-test certificates, the version 2.0 manual, footprint and build dimensions, sling-use guidance, and an aerial-hoop instruction sheet.",
        "reason_code": "support_page_document_index_recall",
        "reason": "The row memorizes the contents of a support-page document index rather than an equipment fact; X-POLE's dense corpus is pending, so it is quarantined without replacement.",
        "review_class": "metadata_drop",
    },
    "fact-b44fa6bd57d935682b96": {
        "active_index": 344,
        "question": "What three roles does Rope365 describe for its website?",
        "answer": "A learning program, a practice guide for improvement, and a compendium of ideas for keeping study fresh.",
        "reason_code": "website_self_description_recall",
        "reason": "The row asks for an about-page description of the website rather than durable rope knowledge; Rope365's dense corpus is in progress, so it is quarantined without replacement.",
        "review_class": "metadata_drop",
    },
    "fact-59c279323ffca52a2d80": {
        "active_index": 402,
        "question": "Where does Rope365 say its free activity videos can be found?",
        "answer": "Rope365 says its free activity videos can be found on YouTube, Vimeo, and within its activity pages.",
        "reason_code": "video_location_navigation_recall",
        "reason": "The row memorizes video locations and site navigation rather than the content of a rope lesson; Rope365's dense corpus is in progress, so it is quarantined without replacement.",
        "review_class": "metadata_drop",
    },
    "fact-578bcde973608d033299": {
        "active_index": 413,
        "question": "Which five historical rope artists does Rope365’s captured reference page list?",
        "answer": "The page lists Itō Seiu, Akechi Denki, Chimuo Nureki, Yukimura Haruki, and Osada Eikechi.",
        "reason_code": "captured_reference_page_list_recall",
        "reason": "The row reproduces a coming-soon reference-page name list without teaching a historical relationship or claim; Rope365's dense corpus is in progress, so it is quarantined without replacement.",
        "review_class": "metadata_drop",
    },
    "fact-a657b3497c216fe6099f": {
        "active_index": 420,
        "question": "Which learning or information resources does Rope365's directory describe for Japanese-rope news, free beginner classes, and interviews?",
        "answer": "It lists Kinbaku Today for Japanese-rope news, Rope Study for free introductions to rope bondage and bottoming, and Tokyo Bound for interviews and other Japanese-bondage information.",
        "reason_code": "nonowner_resource_directory_index_recall",
        "reason": "The row asks the model to reproduce a third-party website directory rather than an owner-requested resource recommendation; Rope365's dense corpus is in progress, so it is quarantined without replacement.",
        "review_class": "metadata_drop",
    },
}


AWKWARD_KEEP_SPECS = {
    "fact-b7799df10cc9d701bf7b": 196,
    "fact-7c524319515c76564591": 410,
}


SCAN_RULES = {
    "explicit_url_canonical_or_sitemap": re.compile(r"\b(?:url|canonical|sitemap)\b", re.I),
    "owner_or_page_location": re.compile(r"\bwhere\s+(?:can\s+someone|is\s+the\s+owner-recommended)\b", re.I),
    "page_title_recall": re.compile(r"\b(?:page|article|post)\s+title\b|\bwhat\s+is\s+the\s+title\b", re.I),
    "publication_date_index_recall": re.compile(r"\bpublication\s+date\b|\bwhen\s+(?:was\s+)?(?:the\s+)?(?:page|post|article)\s+(?:posted|published)\b", re.I),
    "site_structure_or_index": re.compile(
        r"(?:\bwebsite\b.*\broles\b|\broles\b.*\bwebsite\b)|"
        r"(?:\bsupport\s+page\b.*\bdocuments\b|\bdocuments\b.*\bsupport\s+page\b)|"
        r"\bfree\s+activity\s+videos\b.*\bfound\b|\bcaptured\s+reference\s+page\b.*\blist\b|"
        r"\bdirectory\b.*\b(?:describe|list)\b",
        re.I,
    ),
}


REGRESSION_PATTERNS = (
    re.compile(r"\bwhich\s+(?:canonical|current)\b.*\burl\b", re.I),
    re.compile(r"\bwhat\s+url\b|\bsitemap\b", re.I),
    re.compile(r"\b(?:page|article|post)\s+title\b|\bwhat\s+is\s+the\s+title\b", re.I),
    re.compile(r"\bpublication\s+date\b|\bwhen\s+(?:was\s+)?(?:the\s+)?(?:page|post|article)\s+(?:posted|published)\b", re.I),
    SCAN_RULES["site_structure_or_index"],
)


def build_baseline(out: Path, report: Path) -> None:
    previous.build_projection(out, report)
    if (len(read_jsonl(out)), file_sha256(out)) != (BASELINE_ROWS, BASELINE_SHA256):
        raise ValueError("v443 baseline drift")


def build_projection(out: Path, report: Path) -> None:
    inputs = out.parent / f".{out.name}.v444-input"
    inputs.mkdir(parents=True, exist_ok=True)
    base = inputs / "v443.jsonl"
    build_baseline(base, inputs / "v443.report.json")
    core.build_projection_with_inputs(out, report, (CURATION,), (base,))


def whole_candidate_scan(rows: list[dict]) -> dict:
    by_rule = {
        name: [row["fact_id"] for row in rows if pattern.search(row["question"])]
        for name, pattern in SCAN_RULES.items()
    }
    owner = [row["fact_id"] for row in rows if row.get("source") == OWNER_SOURCE]
    nonowner_metadata = sorted({
        fact_id
        for fact_ids in by_rule.values()
        for fact_id in fact_ids
        if fact_id not in owner
    })
    return {
        "by_rule": by_rule,
        "legitimate_owner_resource_recommendations": owner,
        "nonowner_metadata_candidates": nonowner_metadata,
        "rows_scanned": len(rows),
    }


def observe(before: list[dict]) -> dict:
    with tempfile.TemporaryDirectory(prefix=".v444-observe-", dir=HERE) as tmp:
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
    with tempfile.TemporaryDirectory(prefix=".v444-base-", dir=HERE) as tmp:
        directory = Path(tmp)
        base = directory / "v443.jsonl"
        build_baseline(base, directory / "v443.report.json")
        before = read_jsonl(base)
    by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(before, 1)}
    scan = whole_candidate_scan(before)
    if tuple(scan["legitimate_owner_resource_recommendations"]) != OWNER_FACT_IDS:
        raise ValueError("owner resource recommendation inventory drift")
    expected_nonowner = sorted(
        fact_id for fact_id, spec in DROP_SPECS.items() if spec["review_class"] == "metadata_drop"
    )
    if scan["nonowner_metadata_candidates"] != expected_nonowner:
        raise ValueError(f"nonowner metadata inventory drift: {scan['nonowner_metadata_candidates']}")

    queue = {row["resource_id"]: row["status"] for row in json.loads(SITE_CORPUS_QUEUE.read_text())["queue"]}
    audits, curations = [], []
    audit_index = 0
    for fact_id in OWNER_FACT_IDS:
        active_index, row = by_fact[fact_id]
        audit_index += 1
        audits.append({
            "active_answer": row["answer"], "active_index": active_index,
            "active_question": row["question"], "audit_index": audit_index,
            "decision": "keep", "document_sha256": row["document_sha256"],
            "fact_id": fact_id,
            "projection_lineage": {"baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256},
            "reason": "This row answers the owner's explicit request to retain useful rope-resource recommendations and includes contextual selection information rather than arbitrary site metadata.",
            "reason_code": "keep_legitimate_owner_requested_resource_recommendation",
            "review_class": "whole_candidate_metadata_scan",
            "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
            "schema": "context-merit-audit-v444", "source": row["source"], "url": row["url"],
        })

    for fact_id, spec in sorted(DROP_SPECS.items(), key=lambda item: item[1]["active_index"]):
        active_index, row = by_fact[fact_id]
        if active_index != spec["active_index"] or (row["question"], row["answer"]) != (spec["question"], spec["answer"]):
            raise ValueError(f"drop candidate drift: {fact_id}")
        curations.append({
            "action": "drop", "expected_answer": row["answer"],
            "expected_question": row["question"], "fact_id": fact_id,
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
        })
        audit_index += 1
        source_resource_id = {
            "rope365": "rope365", "xpole_a_frame": "xpole_a_frame", "bight_bound": "bight_bound",
        }.get(row["source"])
        audits.append({
            "active_answer": row["answer"], "active_index": active_index,
            "active_question": row["question"], "audit_index": audit_index,
            "decision": "drop", "document_sha256": row["document_sha256"],
            "evidence": row.get("evidence"), "evidence_sha256": text_sha256(row["evidence"]),
            "fact_id": fact_id,
            "projection_lineage": {"baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256},
            "reason": spec["reason"], "reason_code": spec["reason_code"],
            "review_class": spec["review_class"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v444",
            "site_corpus_status_at_review": queue[source_resource_id],
            "source": row["source"], "source_lineage": row["source_lineage"], "url": row["url"],
        })

    for fact_id, expected_index in AWKWARD_KEEP_SPECS.items():
        active_index, row = by_fact[fact_id]
        if active_index != expected_index:
            raise ValueError(f"awkward keep index drift: {fact_id}")
        audit_index += 1
        audits.append({
            "active_answer": row["answer"], "active_index": active_index,
            "active_question": row["question"], "audit_index": audit_index,
            "decision": "keep", "document_sha256": row["document_sha256"],
            "evidence": row["evidence"], "evidence_sha256": text_sha256(row["evidence"]),
            "fact_id": fact_id,
            "projection_lineage": {"baseline_rows": BASELINE_ROWS, "baseline_sha256": BASELINE_SHA256},
            "reason": "Although the answer begins with a pronoun, the named equipment in the question supplies an unambiguous antecedent and the answer gives durable, actionable specifications supported by the captured evidence; no rewrite is warranted before the site's dense corpus is complete.",
            "reason_code": "keep_unambiguous_actionable_equipment_fact",
            "review_class": "awkward_tranche_keep", "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "schema": "context-merit-audit-v444",
            "source": row["source"], "source_lineage": row["source_lineage"], "url": row["url"],
        })

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observation = observe(before)
    if not observation["dataset_equal"] or not observation["report_equal"]:
        raise ValueError(f"nondeterministic projection: {observation}")
    if (observation["rows"], observation["eval"]) != (518, 612):
        raise ValueError(f"projection count drift: {observation}")
    if observation["before"] != EXPECTED_CAPACITY_BEFORE:
        raise ValueError(f"baseline capacity drift: {observation}")
    if EXPECTED_CAPACITY_AFTER is not None and observation["after"] != EXPECTED_CAPACITY_AFTER:
        raise ValueError(f"candidate capacity drift: {observation}")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observation["sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError(f"projection hash drift: {observation}")

    REPORT.write_text(json.dumps({
        "audit": {
            "by_decision": dict(Counter(row["decision"] for row in audits)),
            "by_review_class": dict(Counter(row["review_class"] for row in audits)),
            "path": portable(AUDIT), "rows": len(audits), "sha256": file_sha256(AUDIT),
        },
        "awkward_tranche": {"drops": 1, "keeps": 2, "rows": 3},
        "conservative_capacity": {
            "after": observation["after"], "before": observation["before"],
            "delta": {key: observation["after"][key] - observation["before"][key] for key in observation["before"]},
            "grouping": "shared document SHA, normalized URL, raw lineage family, or pinned v13 lexical-semantic cluster",
        },
        "isolated_build_projection": {
            "automated_projection_runs": 2, "new_additions_applied": 0,
            "output_rows": observation["rows"], "output_sha256": observation["sha256"],
            "repeat_dataset_byte_identical": observation["dataset_equal"],
            "repeat_projection_report_byte_identical": observation["report_equal"],
            "sealed_eval_fact_count_reported_by_tooling": observation["eval"],
        },
        "new_pending_curation": {
            "by_action": {"drop": 6}, "decisions": 6,
            "path": portable(CURATION), "sha256": file_sha256(CURATION),
        },
        "schema": "context-merit-audit-report-v444",
        "sealed_evaluation_policy": {
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-ID collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_heldout_ood_shadow_or_benchmark_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "v52_isolation": {"active_v52_inputs_modified": False, "candidate_status": "isolated_pending_not_promoted"},
        "whole_candidate_metadata_scan": {
            "by_rule_counts": {name: len(ids) for name, ids in scan["by_rule"].items()},
            "flagged_unique_rows": len(set(sum(scan["by_rule"].values(), []))),
            "legitimate_owner_resource_recommendations_kept": 24,
            "nonowner_metadata_rows_dropped": 5,
            "rows_scanned": scan["rows_scanned"],
        },
    }, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
