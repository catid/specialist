#!/usr/bin/env python3
"""Serialize the manual quality-merit re-audit of overlap tranches 1-2."""

from __future__ import annotations

import collections
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data"
PASS_DIR = DATA / "manual_reviews" / "kinbakutoday_third_pass"
OUT_DIR = Path(__file__).resolve().parent
REVIEWER = "codex-kinbakutoday-overlap-quality-merit"
REVIEWED_AT = "2026-07-14"

INPUTS = (
    PASS_DIR / "pending_curation_tranche_01_overlap_a_v1.jsonl",
    PASS_DIR / "pending_replacements_tranche_01_overlap_a_v1.jsonl",
    PASS_DIR / "pending_curation_tranche_02_overlap_b_v1.jsonl",
    PASS_DIR / "pending_replacements_tranche_02_overlap_b_v1.jsonl",
)

# These are manual row-level decisions after rereading all 30 stored source
# documents. Historical evaluation overlap is deliberately not a reason here.
DROP = {
    "fact-7429c1cb3e50a4a74443": (
        "contextless_or_low_value",
        "The isolated model-and-photographer example adds little reusable rope knowledge, and its long answer depends on the surrounding essay.",
    ),
    "fact-46ad8b5088637acbae36": (
        "out_of_domain_or_personal_trivia",
        "The purpose of plastic covers on adult magazines is retail-history trivia rather than reusable rope-bondage knowledge.",
    ),
    "fact-17eac38008e3a9aadf44": (
        "contextless_or_low_value",
        "The generic teach-to-learn aphorism supplies no rope-specific factual content.",
    ),
    "fact-bd009e091c895502828e": (
        "contextless_or_low_value",
        "The single-word translation of sabi is generic language trivia and is less useful than the retained contextual coverage of wabi-sabi in rope aesthetics.",
    ),
    "fact-0a3e9c95515484013183": (
        "out_of_domain_or_personal_trivia",
        "A named performer's instrument is biographical trivia without enough explanation of its role in the rope performance.",
    ),
    "fact-7d02a71cba1bca8bfb57": (
        "out_of_domain_or_personal_trivia",
        "The calendar year in which the Showa era ended is general historical trivia, not a rope-specific fact.",
    ),
    "fact-e3e3336b3698447b97b9": (
        "volatile_or_promotional",
        "The yoga-outfit claim depends on changing social-platform restrictions and market presentation.",
    ),
    "fact-7316b98040367d88ecda": (
        "out_of_domain_or_personal_trivia",
        "The unidentified workshop anecdote rests on one author's personal judgment and yields no durable practice guidance.",
    ),
    "fact-44fa9db34813d1c7d1ca": (
        "semantic_duplicate",
        "Retained QAs from the dedicated hazukashii article define the term and its role more clearly and with better context.",
    ),
    "fact-6c642360467617e05b13": (
        "contextless_or_low_value",
        "The fill-in fragment '100 individuals, 100 ...' is not a self-contained question about the underlying idea of individual styles.",
    ),
    "fact-6139993d177b7259c12e": (
        "semantic_duplicate",
        "A retained QA already identifies Naka Akira as Nureki's deshi while adding the useful context of style development and collaboration.",
    ),
    "fact-6b48affc2b72d4152978": (
        "contextless_or_low_value",
        "The one-word 'felt' answer is an isolated aphorism rather than a durable, explanatory fact.",
    ),
    "fact-6e6484443fe7da7d1791": (
        "contextless_or_low_value",
        "The toolbox comparison is a generic metaphor and does not preserve the source's substantive teaching argument.",
    ),
    "fact-85eca8980cb67601148b": (
        "volatile_or_promotional",
        "The composition of a named commercial Koumanawa rope is product-specific and may change.",
    ),
    "fact-65192847c9c12555eb73": (
        "volatile_or_promotional",
        "Kannōnawa is the interviewee's self-defined performance label rather than a stable, broadly established rope taxonomy.",
    ),
    "fact-f0f0e649064f9349b68d": (
        "contextless_or_low_value",
        "The typewriter is only the object in a generic analogy and contributes no rope-specific knowledge.",
    ),
    "fact-afe480be364f840718e0": (
        "semantic_duplicate",
        "Retained QAs explain kokoro through heart, connection, communication, and practice more clearly than this bare term lookup.",
    ),
    "fact-f22c45622f158b2bb9a5": (
        "semantic_duplicate",
        "Retained QAs already define semenawa directly as rope punishment or torment and add style context.",
    ),
    "fact-3ff91dcaa7dedad25913": (
        "contextless_or_low_value",
        "The paintbrush is only an isolated analogy and does not capture the source's substantive point about kokoro.",
    ),
    "fact-b8b34fc95d5cdc32cafd": (
        "source_error_or_misread",
        "The question generalizes one interviewee's rough 'something like thirty' estimate into a universal count for Japanese rope bondage.",
    ),
    "fact-58f7b7f9868d561b2fd1": (
        "contextless_or_low_value",
        "The phrase is tied only to an undefined 'previous period' and is not a stable or self-contained factual claim.",
    ),
    "fact-c1e38aa956a30b45efdb": (
        "contextless_or_low_value",
        "The broad answer 'emotionally and/or sexually' is an obvious low-information categorization rather than a useful standalone fact.",
    ),
}

MANUAL_PARAPHRASE_SUPPORT = {
    "fact-450cc6ca8581da4c2681":
        "One is obviously the physical aspect such as potential nerve injuries or dropping someone. The other is more on the mental level.",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False,
                                         sort_keys=True) + "\n")


def portable(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def main() -> None:
    raw_documents = {}
    for path in sorted((DATA / "raw").glob("kinbakutoday_*.json")):
        document = json.loads(path.read_text())
        raw_documents.setdefault(document["url"], []).append((path, document))

    input_rows = []
    for input_path in INPUTS:
        tranche = 1 if "tranche_01" in input_path.name else 2
        for row in read_jsonl(input_path):
            row = dict(row)
            row["source_overlap_tranche"] = tranche
            row["source_overlap_ledger"] = portable(input_path)
            input_rows.append(row)

    if len(input_rows) != 107:
        raise ValueError(f"expected 107 input rows, found {len(input_rows)}")
    original_ids = [row["fact_id"] for row in input_rows]
    if len(set(original_ids)) != 107:
        raise ValueError("input overlap fact_ids are not unique")
    if not set(DROP).issubset(original_ids):
        raise ValueError("manual drop map contains an unknown fact_id")

    active_general = {
        row["fact_id"]: row
        for row in read_jsonl(DATA / "train_qa_curated_v1.curation.jsonl")
    }
    active_kinbaku = {
        row["fact_id"]: row
        for row in read_jsonl(DATA / "train_qa_kinbakutoday.curation.jsonl")
    }
    active_decisions = set(active_general) | set(active_kinbaku)

    audits = []
    append_safe = []
    replacements = []
    source_support_counts = collections.Counter()
    decisions_by_tranche = collections.Counter()
    reason_counts = collections.Counter()
    prior_edits_retained = []

    for row in input_rows:
        original_fact_id = row["fact_id"]
        audited_fact_id = row.get("active_fact_id") or original_fact_id
        question = row.get("active_question") or row["expected_question"]
        answer = row.get("active_answer") or row["expected_answer"]
        prior_edit = row.get("replaces_decision") is not None

        matching_documents = [
            (path, document) for path, document
            in raw_documents.get(row["evidence_url"], [])
            if text_sha256(document["text"]) == row["document_sha256"]
        ]
        if len(matching_documents) != 1:
            raise ValueError(
                f"{original_fact_id}: expected exactly one hash-matched raw document")
        raw_path, document = matching_documents[0]

        if normalized(answer) in normalized(document["text"]):
            source_support = "normalized_extractive"
            support_evidence = answer
        elif original_fact_id in MANUAL_PARAPHRASE_SUPPORT:
            source_support = "manual_paraphrase"
            support_evidence = MANUAL_PARAPHRASE_SUPPORT[original_fact_id]
            if normalized(support_evidence) not in normalized(document["text"]):
                raise ValueError(
                    f"{original_fact_id}: manual support excerpt is absent")
        else:
            raise ValueError(f"{original_fact_id}: answer not supported")

        if original_fact_id in DROP:
            decision = "drop"
            reason_code, reason = DROP[original_fact_id]
            reason_counts[reason_code] += 1
        else:
            decision = "keep"
            reason_code = "source_grounded_and_useful"
            reason = (
                "The active QA is supported by the stored source and preserves "
                "durable rope-safety, terminology, technique, history, culture, "
                "communication, or resource value."
            )
            if prior_edit:
                prior_edits_retained.append(original_fact_id)

        source_support_counts[source_support] += 1
        decisions_by_tranche[(row["source_overlap_tranche"], decision)] += 1
        audit = {
            "active_was_prior_edit": prior_edit,
            "audited_answer": answer,
            "audited_fact_id": audited_fact_id,
            "audited_question": question,
            "decision": decision,
            "document_sha256": row["document_sha256"],
            "historical_overlap_reason_used": False,
            "original_fact_id": original_fact_id,
            "raw_document": portable(raw_path),
            "reason": reason,
            "reason_code": reason_code,
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "schema": "kinbakutoday-quality-merit-audit-v1",
            "source_lineage": row["source_lineage"],
            "source_overlap_ledger": row["source_overlap_ledger"],
            "source_overlap_tranche": row["source_overlap_tranche"],
            "source_support": source_support,
            "support_evidence": support_evidence,
            "url": row["evidence_url"],
        }
        audits.append(audit)

        if decision == "drop":
            curation = {
                "action": "drop",
                "document_sha256": row["document_sha256"],
                "evidence_url": row["evidence_url"],
                "expected_answer": answer,
                "expected_question": question,
                "fact_id": original_fact_id,
                "historical_overlap_reason_used": False,
                "reason": reason,
                "reason_code": reason_code,
                "reviewed_at": REVIEWED_AT,
                "reviewer": REVIEWER,
                "source_lineage": row["source_lineage"],
            }
            if original_fact_id in active_decisions:
                replacements.append(curation)
            else:
                append_safe.append(curation)

    audits.sort(key=lambda row: (
        row["source_overlap_tranche"], row["url"],
        row["original_fact_id"]))
    append_safe.sort(key=lambda row: (row["evidence_url"], row["fact_id"]))
    replacements.sort(key=lambda row: (row["evidence_url"], row["fact_id"]))

    if len(audits) != 107 or len(DROP) != 22:
        raise ValueError("manual decision coverage changed unexpectedly")
    if sum(row["decision"] == "keep" for row in audits) != 85:
        raise ValueError("expected 85 explicit keep decisions")
    if len(append_safe) != 22 or replacements:
        raise ValueError(
            "expected 22 append-safe drops and no replacement drops")
    if sorted(prior_edits_retained) != [
        "fact-8dbbd54ad40d9410125d",
        "fact-aff84cf200ba95716418",
    ]:
        raise ValueError("the two prior edits were not both retained")

    audit_path = OUT_DIR / "quality_merit_audit_v1.jsonl"
    append_path = OUT_DIR / "pending_curation_quality_merit_v1.jsonl"
    replacement_path = OUT_DIR / "pending_replacements_quality_merit_v1.jsonl"
    write_jsonl(audit_path, audits)
    write_jsonl(append_path, append_safe)
    write_jsonl(replacement_path, replacements)

    report = {
        "active_artifacts_observed": {
            portable(DATA / "train_qa_curated_v1.curation.jsonl"): {
                "rows": len(active_general),
                "sha256": file_sha256(
                    DATA / "train_qa_curated_v1.curation.jsonl"),
            },
            portable(DATA / "train_qa_curated_v1.jsonl"): {
                "rows": sum(1 for line in (
                    DATA / "train_qa_curated_v1.jsonl").open() if line.strip()),
                "sha256": file_sha256(DATA / "train_qa_curated_v1.jsonl"),
            },
            portable(DATA / "train_qa_curated_v1.report.json"): {
                "sha256": file_sha256(
                    DATA / "train_qa_curated_v1.report.json"),
            },
            portable(DATA / "train_qa_kinbakutoday.curation.jsonl"): {
                "rows": len(active_kinbaku),
                "sha256": file_sha256(
                    DATA / "train_qa_kinbakutoday.curation.jsonl"),
            },
        },
        "active_files_mutated": False,
        "artifacts": {
            "explicit_audit": {
                "path": portable(audit_path),
                "rows": len(audits),
                "sha256": file_sha256(audit_path),
            },
            "pending_append_safe": {
                "path": portable(append_path),
                "rows": len(append_safe),
                "sha256": file_sha256(append_path),
            },
            "pending_replacements": {
                "path": portable(replacement_path),
                "rows": len(replacements),
                "sha256": file_sha256(replacement_path),
            },
        },
        "historical_overlap_reason_used_for_quality_decisions": False,
        "inputs": [
            {"path": portable(path), "rows": len(read_jsonl(path)),
             "sha256": file_sha256(path)}
            for path in INPUTS
        ],
        "manual_outcome": {
            "drop": len(DROP),
            "keep": len(audits) - len(DROP),
            "prior_edit_keep": len(prior_edits_retained),
            "reason_counts": dict(sorted(reason_counts.items())),
            "reviewed_rows": len(audits),
        },
        "prior_edits_retained": sorted(prior_edits_retained),
        "promotion_scope": (
            "segregated_quality_merit_candidate; no active promotion was "
            "performed by this audit"
        ),
        "reviewed_at": REVIEWED_AT,
        "reviewer": REVIEWER,
        "schema": "manual-source-audit-report-v1",
        "source_verification": {
            "manual_paraphrase_rows": [
                row["original_fact_id"] for row in audits
                if row["source_support"] == "manual_paraphrase"
            ],
            "raw_document_hash_matches": len(audits),
            "rows_manually_read": len(audits),
            "source_documents_checked": len({row["url"] for row in audits}),
            "support_counts": dict(sorted(source_support_counts.items())),
        },
        "tranches": {
            str(tranche): {
                "drop": decisions_by_tranche[(tranche, "drop")],
                "keep": decisions_by_tranche[(tranche, "keep")],
                "reviewed": sum(
                    decisions_by_tranche[(tranche, decision)]
                    for decision in ("drop", "keep")),
            }
            for tranche in (1, 2)
        },
        "validated_projection": {
            "curated_output": {
                "curation_decisions": 2570,
                "excluded_rows": 2524,
                "kinbakutoday_rows": 277,
                "output_rows": 762,
                "output_sha256": "54880739de80ba751ce0e7b476d6bfc02e29bed6851db558a2ac5e1043536e74",
                "report_sha256": "8b2c0ce78d5d3785b505861ab69da33d3c831a91905694fb9bbf0b11dd8ccf27",
            },
            "method": (
                "promote_curation_ledgers.py appended the pending merit "
                "ledger to an isolated copy of the active general ledger; "
                "build_curated_qa.py then produced byte-identical outputs "
                "twice. Embedded provenance paths were normalized to their "
                "final active paths for the projected production hashes."
            ),
            "promoted_general_curation": {
                "appended_rows": 22,
                "drop_decisions": 1771,
                "edit_decisions": 48,
                "replacement_rows": 0,
                "rows": 1819,
                "sha256": "e1b73252e966fa343b611982900209c7900f6373508cfed2c5782309ac1e65c6",
            },
            "repeat_build_byte_identical": True,
            "status": "passed",
        },
    }
    report_path = OUT_DIR / "report_quality_merit_v1.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
