#!/usr/bin/env python3
"""Audit context-merit tranche v9 without rewriting prior tranches."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V8_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v8"
sys.path[:0] = [str(ROOT), str(V8_DIR)]
import build_context_merit_audit_v8 as previous
from qa_quality import qa_pair_from_record

common = previous.common
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v9.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v9.jsonl"
REPORT = OUT_DIR / "report_context_merit_v9.json"
REVIEWER = "codex-context-merit-audit-v9"
REVIEWED_AT = "2026-07-14"
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
CONTEXT_CURATIONS = tuple(DATA / "manual_reviews" / f"context_merit_audit_v{v}" /
                          f"pending_curation_context_merit_v{v}.jsonl" for v in range(1, 9))
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
    {"fact_id": "fact-a8f9b828a68537fb8bd8", "source_path": raw("esinem_a94ba59553f0cc78.json"),
     "marker": "average of 10 wraps around a cylinder", "decision": "keep",
     "reason_code": "rope_measurement_context_complete",
     "reason": "Averaging ten wraps is a concrete, repeatable way the source estimates representative rope diameter."},
    {"fact_id": "fact-e18190210c4bd73d6008", "source_path": raw("wikipedia_a97e19abfe49afa9.json"),
     "marker": "Masaki-ryū, Nawa Yumio", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "An isolated lineage-head name lookup omits the books, preservation work, and historical content that could make it useful."},
    {"fact_id": "fact-fa3c992a1748e50fc62c", "source_path": raw("kinbakutoday_e2c96f38ced1cb5d.json"),
     "marker": "adviser, artist, author, bakushi and critic", "decision": "edit",
     "question": "What roles did Minomura Kou serve for SM Collector?",
     "answer": "adviser, artist, author, bakushi and critic",
     "reason_code": "replace_name_lookup_with_contribution",
     "reason": "The edit makes Minomura's unusually broad contribution the answer rather than another bare person lookup."},
    {"fact_id": "fact-fcf827073a98f75fd9a5", "source_path": raw("rope365_6a23940f6bc37fc1.json"),
     "marker": "granny knot (aka tatemusubi 縦結び)", "decision": "keep",
     "reason_code": "knot_alias_context_complete",
     "reason": "The Japanese term and writing are directly paired with the granny knot."},
    {"fact_id": "fact-186a05e6675e4b4d7355", "source_path": raw("rope365_682937f92222bf87.json"),
     "marker": "arborists and stage riggers", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "Listing two modern rope-using occupations is broad rope trivia with little relevance to bondage assistance."},
    {"fact_id": "fact-099b416b74b5f9d328cb", "source_path": raw("rope365_62f7e527bb35b47d.json"),
     "marker": "it weighs about 60% as much", "decision": "keep",
     "reason_code": "source_attributed_material_comparison",
     "reason": "The approximate weight comparison is explicitly limited to single-braid MFP versus single-braid nylon."},
    {"fact_id": "fact-224434e4450cb9a39363", "source_path": raw("esinem_4cdf0843c58d3796.json"),
     "marker": "Hashira shibari, vertical beam suspension", "decision": "keep",
     "reason_code": "suspension_term_context_complete",
     "reason": "The QA defines the named vertical-beam suspension category without supplying unsupported execution steps."},
    {"fact_id": "fact-4fad27c6ab983ca6849d", "source_path": raw("kinbakutoday_5a6e15f57e52a345.json"),
     "marker": "Matsui Kenji naming ties", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "The isolated attribution is an interviewer's anecdotal example and gives no named ties or reusable terminology."},
    {"fact_id": "fact-854abc7d9f7b294521d3", "source_path": raw("rope365_d1dab5f5fa3d5cbc.json"),
     "marker": "classic torture technique from the Edo period", "decision": "keep",
     "reason_code": "historical_restraint_context_complete",
     "reason": "The period is explicitly tied to the shrimp tie's historical torture context rather than modern safety advice."},
    {"fact_id": "fact-8b91fe6ef801b39b61f6", "source_path": raw("rope365_e5a66b9fd87de937.json"),
     "marker": "bent leg tie is also called futomomo shibari", "decision": "drop", "reason_code": "semantic_duplicate",
     "reason": "A retained QA already gives futomomo shibari with its Japanese writing and thigh-tie gloss."},
    {"fact_id": "fact-b0325d750b296af187c6", "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
     "marker": "McLelland makes a similar caution about women’s reader letters in the postwar “perverse press", "decision": "keep",
     "reason_code": "postwar_media_term_context_complete",
     "reason": "Perverse press is a cited scholarly term for the postwar publications discussed, with the reader-letter context supplied."},
    {"fact_id": "fact-b94b186f58ec4c0bf40a", "source_path": raw("rope365_5c01d29a044cabf7.json"),
     "marker": "Inline cuffs are also called hojo cuff", "decision": "drop", "reason_code": "semantic_duplicate",
     "reason": "The retained hojo-cuff QA already connects inline cuffs to Hojōjutsu more completely."},
    {"fact_id": "fact-bcc7840caa09e7ca9afb", "source_path": raw("rope365_bac79ac0456a12c1.json"),
     "marker": "horizontal structure can be the base", "decision": "drop", "reason_code": "contextless_or_low_value",
     "reason": "The base-shape lookup is a course-specific construction fragment and omits the source's neck and comfort warning."},
    {"fact_id": "fact-c7f97d425dddd7aecbb5", "source_path": raw("kinbakutoday_4f26f20c5f1dc7ba.json"),
     "marker": "Roman Porno was a genre", "decision": "keep",
     "reason_code": "rope_media_history_context_complete",
     "reason": "Roman Porno is a stable genre term in the history of mainstream Japanese erotic and bondage film."},
    {"fact_id": "fact-c81b3c9e157104beadc0", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "preventing the elbows from moving out", "decision": "drop", "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A restriction-increasing box-tie micro-step is presented without enough nerve, circulation, consent, or release context."},
    {"fact_id": "fact-e7d2fc133a984af988a7", "source_path": raw("kinbakutoday_92c9fc29a66300a0.json"),
     "marker": "listed as Hitoshi Sharaku", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "The film-credit identity is explicitly unresolved in the source and is low-value person trivia."},
    {"fact_id": "fact-f11b8cd79977beb93ed4", "source_path": raw("rope365_0a23b686a4f493e1.json"),
     "marker": "ABOK 2490", "decision": "keep", "reason_code": "knot_reference_context_complete",
     "reason": "The exact Ashley Book of Knots reference is durable and useful for locating the half-hitch definition."},
    {"fact_id": "fact-f6e16a9975395e3c7ec0", "source_path": raw("kinbakutoday_3fe23acd4ef143eb.json"),
     "marker": "chief editor of Uramado Magazine", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "The magazine-name lookup is a personal-career detail and omits Nureki's relevant writing or editorial contribution."},
    {"fact_id": "fact-fb0119c5862636f2e7d7", "source_path": raw("rope365_0b4f44c8fabfc202.json"),
     "marker": "tourniquet effect", "decision": "keep", "reason_code": "material_safety_context_complete",
     "reason": "The tourniquet effect is the source's direct warning against using elastic rope on skin."},
    {"fact_id": "fact-fcc8539f59ea2b810265", "source_path": raw("wikipedia_12053c31533307a9.json"),
     "marker": "literal translation of the French name", "decision": "drop", "reason_code": "semantic_duplicate",
     "reason": "A retained QA already gives lark's head as the cow hitch's alternate name without translation trivia."},
    {"fact_id": "fact-fdbdb2cba818b0315af6", "source_path": raw("kinbakutoday_381214111022a952.json"),
     "marker": "ø4mm jute Aranawa", "decision": "drop", "reason_code": "contextless_or_low_value",
     "reason": "A rope diameter from one narrative scene is a fragile equipment detail, not a recommendation or style requirement."},
    {"fact_id": "fact-7062996ff78795383f5a", "source_path": raw("rope365_da67938bae2edac4.json"),
     "marker": "Edo period (1603-1867) is considered the golden age", "decision": "keep",
     "reason_code": "restraint_history_context_complete",
     "reason": "The Edo period supplies stable historical context for the development of Hojōjutsu."},
    {"fact_id": "fact-1718144091206df00ebb", "source_path": raw("wikipedia_85577b2bf05b0cce.json"),
     "marker": "safe, sane and consensual", "decision": "edit",
     "question": "What does the BDSM risk framework SSC stand for?",
     "answer": "safe, sane and consensual",
     "reason_code": "replace_adoption_decade_with_framework_definition",
     "reason": "The edit replaces a bare adoption decade with the framework's reusable expansion."},
    {"fact_id": "fact-18ad3b9fe7fd13d65cec", "source_path": raw("rope365_753bb782335285c6.json"),
     "marker": "Life drawing – Volunteer to tie", "decision": "keep",
     "reason_code": "rope_community_activity_context_complete",
     "reason": "Life drawing is directly defined as volunteering to tie while being captured by drawing participants."},
    {"fact_id": "fact-28d15f24f196d7ea55c6", "source_path": raw("rope365_f250f228cc370052.json"),
     "marker": "Somerville bowline, one of the many variations", "decision": "keep",
     "reason_code": "single_column_knot_context_complete",
     "reason": "The Somerville bowline is accurately situated as a single-column-tie variation."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v8": 710,
    "active_after_this_tranche": 699,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 11,
    "new_edits_applied": 2,
    "output_rows": 737,
    "output_sha256": "e5767343cba5907abd3abb907f227c382abc65277a3b845afcbd9447d969a0a3",
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
    reviewed, ranked, provenance_count = reviewed_fact_ids(), [], 0
    for index, row in enumerate(rows, 1):
        provenance = any(field in row for field in common.ACTIVE_REVIEW_FIELDS)
        if row["fact_id"] in reviewed or provenance:
            provenance_count += provenance
            continue
        features = risk_features(row)
        ranked.append((-features["risk_score"], features["question_tokens"],
                       features["answer_tokens"], row["fact_id"], index, row, features))
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
    for audit_index, (spec, item) in enumerate(zip(SPECS, selected), 1):
        row = item["row"]
        question, answer = qa_pair_from_record(row)
        evidence, support_type = source_evidence(spec, row)
        audit = {"active_answer": answer, "active_index": item["active_index"],
                 "active_question": question, "audit_index": audit_index,
                 "decision": spec["decision"], "document_sha256": row["document_sha256"],
                 "fact_id": row["fact_id"], "reason": spec["reason"],
                 "reason_code": spec["reason_code"], "reviewed_at": REVIEWED_AT,
                 "reviewer": REVIEWER, "risk_features": item["features"],
                 "schema": "context-merit-audit-v9", "source": row["source"],
                 "source_document": portable(spec["source_path"]),
                 "source_document_file_sha256": file_sha256(spec["source_path"]),
                 "source_support": support_type, "support_evidence": evidence,
                 "support_evidence_sha256": text_sha256(evidence), "url": row["url"]}
        if spec["decision"] == "edit":
            audit.update(edited_question=spec["question"], edited_answer=spec["answer"])
        audits.append(audit)
        if spec["decision"] in {"drop", "edit"}:
            curation = {"action": spec["decision"], "document_sha256": row["document_sha256"],
                        "evidence_url": row["url"], "expected_answer": answer,
                        "expected_question": question, "fact_id": row["fact_id"],
                        "reason": spec["reason"], "reason_code": spec["reason_code"],
                        "reviewed_at": REVIEWED_AT, "reviewer": REVIEWER,
                        "source_lineage": row["source_lineage"]}
            if spec["decision"] == "edit":
                curation.update(answer=spec["answer"], evidence=evidence,
                                question=spec["question"], support_type="extractive")
            curations.append(curation)
    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    decisions = collections.Counter(row["decision"] for row in audits)
    report = {"schema": "context-merit-audit-report-v9", "reviewer": REVIEWER,
              "reviewed_at": REVIEWED_AT, "status": "segregated_pending_not_promoted",
              "selection": {"active_rows": len(active_rows), "eligible_unreviewed_rows": len(ranked),
                            "excluded_ledger_fact_ids": excluded,
                            "excluded_active_review_provenance": provenance,
                            "rows_selected": 25, "fact_ids_in_rank_order": list(selected_ids),
                            "ranking": {"score": "short_question_points + 3*pronoun_count + bare_answer_points + named_person_trivia_points",
                                        "tie_break": "risk_score descending, question tokens ascending, answer tokens ascending, fact_id ascending"}},
              "audit": {"path": portable(AUDIT), "sha256": file_sha256(AUDIT), "rows": 25,
                        "by_decision": dict(sorted(decisions.items())),
                        "by_reason": dict(sorted(collections.Counter(row["reason_code"] for row in audits).items()))},
              "new_pending_curation": {"path": portable(CURATION), "sha256": file_sha256(CURATION),
                                       "decisions": len(curations), "by_action": dict(sorted(collections.Counter(row["action"] for row in curations).items()))},
              "prior_pending": {"additions": [{"path": portable(path), "rows": len(read_jsonl(path)), "sha256": file_sha256(path)} for path in PRIOR_PENDING_ADDITIONS],
                                "curations": [{"path": portable(path), "rows": len(read_jsonl(path)), "sha256": file_sha256(path)} for path in (QUALITY_MERIT_CURATION, TASUKI_CURATION, *CONTEXT_CURATIONS)]},
              "active_baseline": {"dataset": {"path": portable(ACTIVE_DATASET), "rows": len(active_rows), "sha256": file_sha256(ACTIVE_DATASET)},
                                  "report": {"path": portable(ACTIVE_REPORT), "sha256": file_sha256(ACTIVE_REPORT)},
                                  "curation": [{"path": portable(path), "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS]},
              "sealed_evaluation_policy": {"manual_review_opened_eval_or_heldout_content": False,
                                           "generator_opens_eval_or_heldout_content": False,
                                           "collision_check": "sealed evaluation paths are handled only by build_curated_qa.py during isolated projection"},
              "isolated_build_projection": ISOLATED_PROJECTION}
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
