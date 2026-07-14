#!/usr/bin/env python3
"""Audit context-merit tranche v6 without rewriting prior tranches."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
V5_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v5"
sys.path[:0] = [str(ROOT), str(V5_DIR)]

import build_context_merit_audit_v5 as previous
from qa_quality import qa_pair_from_record


common = previous.common
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v6.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v6.jsonl"
REPORT = OUT_DIR / "report_context_merit_v6.json"
REVIEWER = "codex-context-merit-audit-v6"
REVIEWED_AT = "2026-07-14"
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST

ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
CONTEXT_CURATIONS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"pending_curation_context_merit_v{version}.jsonl"
    for version in (1, 2, 3, 4, 5)
)
PRIOR_CONTEXT_MERIT_DIRS = frozenset(
    path.parent.name for path in CONTEXT_CURATIONS
)

file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
portable = previous.portable
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
risk_features = previous.risk_features
source_evidence = previous.source_evidence


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-dc424e61c6c9c3551d03",
        "source_path": raw("kinbakutoday_43095aa305e35a41.json"),
        "marker": "ask if the ukete is really ok",
        "decision": "keep",
        "reason_code": "participant_role_term_context_complete",
        "reason": "Ukete is a reusable term for the person being tied, and the question defines the role directly.",
    },
    {
        "fact_id": "fact-f08accf0dce3b37452e8",
        "source_path": raw("wikipedia_2151448295a2af9b.json"),
        "marker": "Usenet post from 1991",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": "The first recorded year of a broad BDSM initialism is tangential etymology trivia, not rope guidance.",
    },
    {
        "fact_id": "fact-3d3373b7b595e1b6324b",
        "source_path": raw("kinbakutoday_34dec041941868d9.json"),
        "marker": "posturing called “Omoi-ire”",
        "decision": "keep",
        "reason_code": "rope_performance_concept_context_complete",
        "reason": "Omoi-ire is explicitly connected to nonverbal expression in both Kabuki and rope scenes.",
    },
    {
        "fact_id": "fact-87d6b257169d6a7e409c",
        "source_path": raw("kinbakutoday_5783480db327a3ba.json"),
        "marker": "rare moments in early postwar bondage culture where a model’s own voice appears in print",
        "decision": "edit",
        "question": "Why does Kinbaku Today consider Kawabata Tanako’s May 1953 Kitan Club page historically significant?",
        "answer": "it provides one of the rare moments in early postwar bondage culture where a model’s own voice appears in print",
        "reason_code": "replace_publication_date_with_historical_significance",
        "reason": "The edit replaces a bare year with why the page matters: it preserves a model's own historical voice.",
    },
    {
        "fact_id": "fact-904e5f318a018e7878af",
        "source_path": raw("rope365_62f7e527bb35b47d.json"),
        "marker": "averaging ~60% of what a comparable nylon kit weighs",
        "decision": "keep",
        "reason_code": "source_attributed_material_comparison",
        "reason": "The approximate weight comparison is clearly scoped to a comparable kit and can inform material choice.",
    },
    {
        "fact_id": "fact-a667748ec149351ab907",
        "source_path": raw("rope365_0b4f44c8fabfc202.json"),
        "marker": "Hempex floats in water and doesn’t shrink when wet",
        "decision": "keep",
        "reason_code": "rope_material_property_context_complete",
        "reason": "The QA records two concrete, source-attributed Hempex properties without generalizing them to safety.",
    },
    {
        "fact_id": "fact-b784b4278644a2f88656",
        "source_path": raw("rope365_6f46d5169ca32ec7.json"),
        "marker": "co-founder of Bakuyukai",
        "decision": "keep",
        "reason_code": "rope_organization_history_context_complete",
        "reason": "Bakuyukai is a stable historical institution tied directly to Akechi Denki's rope lineage.",
    },
    {
        "fact_id": "fact-da08e38f2a3bd86a6c62",
        "source_path": raw("rope365_d7f0891ee630f057.json"),
        "marker": "futomomo shibari 太腿縛り",
        "decision": "keep",
        "reason_code": "technique_term_context_complete",
        "reason": "The Japanese writing is explicitly paired with futomomo shibari and a thigh-tie gloss.",
    },
    {
        "fact_id": "fact-ddaca44426279e13b0a7",
        "source_path": raw("kinbakutoday_d5d373e4a55ff204.json"),
        "marker": "pronounced “gyaggu.”",
        "decision": "drop",
        "reason_code": "contextless_or_low_value",
        "reason": "A katakana pronunciation lookup adds little beyond the retained native Japanese gag term sarugutsuwa.",
    },
    {
        "fact_id": "fact-0168b34eab462dc6be58",
        "source_path": raw("rope365_25ef6b53056c7cea.json"),
        "marker": "Rope365 curriculum is built",
        "decision": "drop",
        "reason_code": "volatile_or_promotional",
        "reason": "Naming the site from its own promotional curriculum description is a circular low-information lookup.",
    },
    {
        "fact_id": "fact-1c5a8e2126b5014aecaa",
        "source_path": raw("esinem_d00f706084758774.json"),
        "marker": "Jute wore faster than a very tight linen hemp",
        "decision": "edit",
        "question": "What wear result did Esinem report when comparing jute with a very tight, high-yarn-count linen hemp?",
        "answer": "Jute wore faster than a very tight linen hemp with a high yarn count (40 per ply) by a factor of several times",
        "reason_code": "restore_test_specific_material_context",
        "reason": "The edit retains the tested rope's construction so a single comparison is not presented as a universal rule.",
    },
    {
        "fact_id": "fact-474453892a0ce51d0169",
        "source_path": raw("rope365_1db41038891a4e1f.json"),
        "marker": "(hishi in Japanese)",
        "decision": "drop",
        "reason_code": "semantic_duplicate",
        "reason": "A retained edited QA already identifies hishi as a diamond shape without course-day wording.",
    },
    {
        "fact_id": "fact-733d870abe140a50c2e0",
        "source_path": raw("kinbakutoday_aec9a390fd230290.json"),
        "marker": "called the ‘Naka-Ryu’ style",
        "decision": "keep",
        "reason_code": "rope_style_lineage_context_complete",
        "reason": "Naka-Ryu is the explicitly named offshoot associated with Akira Naka and is useful style-lineage context.",
    },
    {
        "fact_id": "fact-83f5e74aec39cbbced3a",
        "source_path": raw("kinbakutoday_938724eb415ca5c0.json"),
        "marker": "rope is made of Ramie",
        "decision": "keep",
        "reason_code": "historical_rope_material_context_complete",
        "reason": "The material is tied to a specific documented 1930 image rather than generalized to modern rope practice.",
    },
    {
        "fact_id": "fact-ccc3a31cf551a8ac8f5d",
        "source_path": raw("esinem_f2dfde25be14a7a8.json"),
        "marker": "Taisho era 1912-1926",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": "The date range of a Japanese era is broad history trivia and the QA omits its relevance to kinbaku terminology.",
    },
    {
        "fact_id": "fact-cf250d3c67a7283327d4",
        "source_path": raw("wikipedia_2151448295a2af9b.json"),
        "marker": "are called switches",
        "decision": "keep",
        "reason_code": "bdsm_role_term_context_complete",
        "reason": "Switch is a common role term, and the question precisely explains the role change it denotes.",
    },
    {
        "fact_id": "fact-59be55d66c8b1e6fba0c",
        "source_path": raw("rope365_d9f9d21430ff4bba.json"),
        "marker": "GSG企画",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": "An untranslated theater-group name from a career chronology is personal biography trivia.",
    },
    {
        "fact_id": "fact-59e4bb42809729ee43fd",
        "source_path": raw("wikipedia_ea35c24ae8ca2151.json"),
        "marker": "clinical use in his work",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": "The clinical coinage year for broad sadomasochism terminology is tangential to rope-specific assistance.",
    },
    {
        "fact_id": "fact-5ba72485e95be3a6972f",
        "source_path": raw("kinbakutoday_34dec041941868d9.json"),
        "marker": "feeling of Urami",
        "decision": "keep",
        "reason_code": "rope_art_emotional_concept_context_complete",
        "reason": "Urami is explained as grudge or resentment and connected by the source to emotional expression in rope art.",
    },
    {
        "fact_id": "fact-6aff3284b50d59e50f50",
        "source_path": raw("kinbakutoday_b776fccd348e2538.json"),
        "marker": "In 1989, CineMagic released",
        "decision": "keep",
        "reason_code": "rope_media_history_context_complete",
        "reason": "The date anchors the first entry in Nureki's named landmark eight-video rope series.",
    },
    {
        "fact_id": "fact-8a1b266bc8d3e9cf09ad",
        "source_path": raw("kinbakutoday_79178d1cf5f1d949.json"),
        "marker": "one of the two primary texts that Akechi Denki used",
        "decision": "edit",
        "question": "Why is Watatani Kiyoshi’s hojojutsu guide significant to students of kinbaku?",
        "answer": "it is one of the two primary texts that Akechi Denki used to study the art of hojojutsu",
        "reason_code": "replace_publication_date_with_lineage_significance",
        "reason": "The edit replaces an isolated publication year with the guide's documented importance to Akechi's study.",
    },
    {
        "fact_id": "fact-8f107ba229ab1943faba",
        "source_path": raw("kinbakutoday_89994b45562f9ad8.json"),
        "marker": "please don’t think that tokonawa is safer",
        "decision": "edit",
        "question": "What should someone not assume about tokonawa, according to Ouji?",
        "answer": "that tokonawa is safer",
        "reason_code": "replace_term_lookup_with_safety_warning",
        "reason": "The edit prioritizes the source's explicit warning over memorizing a bed-tying label without risk context.",
    },
    {
        "fact_id": "fact-a9a2266b2cc6177274c6",
        "source_path": raw("esinem_4a64b6d0e45a7c34.json"),
        "marker": "clockwise loop around your thumb",
        "decision": "drop",
        "reason_code": "contextless_or_low_value",
        "reason": "A handed, method-specific thumb-loop direction is a fragile micro-step that the source says needs video explanation.",
    },
    {
        "fact_id": "fact-abe07db12e2f94fbbb7e",
        "source_path": raw("kinbakutoday_2ded51b08225bd4b.json"),
        "marker": "considered shibusa",
        "decision": "keep",
        "reason_code": "rope_aesthetic_term_context_complete",
        "reason": "Shibusa is directly applied to the simple elegance of a high-hands takate kote, giving the term rope context.",
    },
    {
        "fact_id": "fact-b349d72365353eaf5227",
        "source_path": raw("rope365_8ae9e3d93b31601b.json"),
        "marker": "Switch – Someone who can play both roles",
        "decision": "drop",
        "reason_code": "semantic_duplicate",
        "reason": "This duplicates the retained, more precisely worded definition of switches in the same tranche.",
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v5": 733,
    "active_after_this_tranche": 724,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 9,
    "new_edits_applied": 4,
    "output_rows": 762,
    "output_sha256":
        "1b68d8e25db0bbab2e243a371af68ad781ce3e5de64f97a866d5e0e0f67e0ec6",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 12,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def reviewed_fact_ids() -> set[str]:
    reviewed = set()
    manual_root = DATA / "manual_reviews"
    for path in sorted(manual_root.rglob("*.jsonl")):
        if OUT_DIR in path.parents:
            continue
        review_dir = path.relative_to(manual_root).parts[0]
        if (re.fullmatch(r"context_merit_audit_v\d+", review_dir) and
                review_dir not in PRIOR_CONTEXT_MERIT_DIRS):
            continue
        for row in read_jsonl(path):
            for field in common.ID_FIELDS:
                value = row.get(field)
                if isinstance(value, str) and value.startswith("fact-"):
                    reviewed.add(value)
            reviewed.update(value for value in row.get("candidate_fact_ids", [])
                            if isinstance(value, str) and value.startswith("fact-"))
    return reviewed


def ranked_unreviewed(rows: list[dict]) -> tuple[list[dict], int, int]:
    reviewed = reviewed_fact_ids()
    ranked = []
    provenance_count = 0
    for index, row in enumerate(rows, 1):
        provenance = any(field in row for field in common.ACTIVE_REVIEW_FIELDS)
        if row["fact_id"] in reviewed or provenance:
            provenance_count += provenance
            continue
        features = risk_features(row)
        ranked.append((-features["risk_score"], features["question_tokens"],
                       features["answer_tokens"], row["fact_id"], index, row,
                       features))
    ranked.sort(key=lambda item: item[:4])
    return ([{"active_index": item[4], "row": item[5], "features": item[6]}
             for item in ranked], len(reviewed), provenance_count)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active_rows = read_jsonl(ACTIVE_DATASET)
    ranked, excluded, provenance = ranked_unreviewed(active_rows)
    selected = ranked[:25]
    selected_ids = tuple(item["row"]["fact_id"] for item in selected)
    if selected_ids != EXPECTED_SELECTION:
        raise ValueError(f"selection drift: {selected_ids!r}")
    audits, curations = [], []
    for audit_index, (spec, selected_item) in enumerate(zip(SPECS, selected), 1):
        row = selected_item["row"]
        question, answer = qa_pair_from_record(row)
        evidence, support_type = source_evidence(spec, row)
        audit = {
            "active_answer": answer, "active_index": selected_item["active_index"],
            "active_question": question, "audit_index": audit_index,
            "decision": spec["decision"], "document_sha256": row["document_sha256"],
            "fact_id": row["fact_id"], "reason": spec["reason"],
            "reason_code": spec["reason_code"], "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER, "risk_features": selected_item["features"],
            "schema": "context-merit-audit-v6", "source": row["source"],
            "source_document": portable(spec["source_path"]),
            "source_document_file_sha256": file_sha256(spec["source_path"]),
            "source_support": support_type, "support_evidence": evidence,
            "support_evidence_sha256": text_sha256(evidence), "url": row["url"],
        }
        if spec["decision"] == "edit":
            audit.update(edited_question=spec["question"], edited_answer=spec["answer"])
        audits.append(audit)
        if spec["decision"] in {"drop", "edit"}:
            curation = {
                "action": spec["decision"], "document_sha256": row["document_sha256"],
                "evidence_url": row["url"], "expected_answer": answer,
                "expected_question": question, "fact_id": row["fact_id"],
                "reason": spec["reason"], "reason_code": spec["reason_code"],
                "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
                "source_lineage": row["source_lineage"],
            }
            if spec["decision"] == "edit":
                curation.update(answer=spec["answer"], evidence=evidence,
                                question=spec["question"], support_type="extractive")
            curations.append(curation)
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    decisions = collections.Counter(row["decision"] for row in audits)
    report = {
        "schema": "context-merit-audit-report-v6", "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT, "status": "segregated_pending_not_promoted",
        "selection": {
            "active_rows": len(active_rows), "eligible_unreviewed_rows": len(ranked),
            "excluded_ledger_fact_ids": excluded,
            "excluded_active_review_provenance": provenance,
            "rows_selected": 25, "fact_ids_in_rank_order": list(selected_ids),
            "ranking": {
                "score": "short_question_points + 3*pronoun_count + bare_answer_points + named_person_trivia_points",
                "tie_break": "risk_score descending, question tokens ascending, answer tokens ascending, fact_id ascending"}},
        "audit": {"path": portable(AUDIT), "sha256": file_sha256(AUDIT),
                  "rows": 25, "by_decision": dict(sorted(decisions.items())),
                  "by_reason": dict(sorted(collections.Counter(
                      row["reason_code"] for row in audits).items()))},
        "new_pending_curation": {
            "path": portable(CURATION), "sha256": file_sha256(CURATION),
            "decisions": len(curations), "by_action": dict(sorted(
                collections.Counter(row["action"] for row in curations).items()))},
        "prior_pending": {
            "additions": [{"path": portable(path), "rows": len(read_jsonl(path)),
                           "sha256": file_sha256(path)}
                          for path in PRIOR_PENDING_ADDITIONS],
            "curations": [{"path": portable(path), "rows": len(read_jsonl(path)),
                           "sha256": file_sha256(path)} for path in
                          (QUALITY_MERIT_CURATION, TASUKI_CURATION, *CONTEXT_CURATIONS)]},
        "active_baseline": {
            "dataset": {"path": portable(ACTIVE_DATASET), "rows": len(active_rows),
                        "sha256": file_sha256(ACTIVE_DATASET)},
            "report": {"path": portable(ACTIVE_REPORT), "sha256": file_sha256(ACTIVE_REPORT)},
            "curation": [{"path": portable(path), "sha256": file_sha256(path)}
                         for path in ACTIVE_CURATIONS]},
        "sealed_evaluation_policy": {
            "manual_review_opened_eval_or_heldout_content": False,
            "generator_opens_eval_or_heldout_content": False,
            "collision_check": "sealed evaluation paths are handled only by build_curated_qa.py during isolated projection"},
        "isolated_build_projection": ISOLATED_PROJECTION,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
