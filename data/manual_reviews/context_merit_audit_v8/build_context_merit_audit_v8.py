#!/usr/bin/env python3
"""Audit context-merit tranche v8 without rewriting prior tranches."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V7_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v7"
sys.path[:0] = [str(ROOT), str(V7_DIR)]

import build_context_merit_audit_v7 as previous
from qa_quality import qa_pair_from_record

common = previous.common
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v8.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v8.jsonl"
REPORT = OUT_DIR / "report_context_merit_v8.json"
REVIEWER = "codex-context-merit-audit-v8"
REVIEWED_AT = "2026-07-14"
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
CONTEXT_CURATIONS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"pending_curation_context_merit_v{version}.jsonl"
    for version in range(1, 8))
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
    {"fact_id": "fact-41006d3854057b0f888e", "source_path": raw("rope365_1ee277117c1ce420.json"),
     "marker": "bind two columns at a 90-degree angle", "decision": "keep",
     "reason_code": "lashing_structure_context_complete",
     "reason": "The angle is the defining structural use of a square lashing and the question names the lashing directly."},
    {"fact_id": "fact-4a1a5f61155506c25498", "source_path": raw("rope365_bc4120687a97c8c3.json"),
     "marker": "called awajimusubi 淡路結び", "decision": "keep",
     "reason_code": "knot_alias_context_complete",
     "reason": "The Japanese name and characters are directly paired with the double coin knot."},
    {"fact_id": "fact-7ae68d3ea57c54953080", "source_path": raw("esinem_a337d2fe39281aa2.json"),
     "marker": "natural jute being the usual medium", "decision": "drop",
     "reason_code": "overgeneralized_or_time_sensitive",
     "reason": "A single teacher's broad claim about the usual Japanese medium is unqualified and can vary by person, place, and period."},
    {"fact_id": "fact-8b29fcbc98f4a055598b", "source_path": raw("kinbakutoday_34dec041941868d9.json"),
     "marker": "Kita Reiko", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "A second artist-name lookup adds little beyond the retained definition and rope-art context of urami."},
    {"fact_id": "fact-2566ed1ef3852d69cf1e", "source_path": raw("rope365_682937f92222bf87.json"),
     "marker": "Gene Bilbrew", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "The isolated illustrator attribution is visual-culture trivia and does not preserve a rope-specific contribution."},
    {"fact_id": "fact-5e4a5bf55948ce3fca27", "source_path": raw("esinem_888997ffcb0c181d.json"),
     "marker": "public kinbaku shows began with Eikichi Osada", "decision": "keep",
     "reason_code": "rope_performance_history_context_complete",
     "reason": "The question appropriately frames the answer as a source attribution about the history of public kinbaku shows."},
    {"fact_id": "fact-0aa7a33b3f304ab39406", "source_path": raw("kinbakutoday_98c19e5a4c2038e0.json"),
     "marker": "published in 1973 through Kitan Club", "decision": "drop", "reason_code": "semantic_duplicate",
     "reason": "The retained edited QA about this artifact already captures its historical significance rather than publisher trivia."},
    {"fact_id": "fact-0c5693d6e4d98729f4b3", "source_path": raw("kinbakutoday_4df72c39939f2a24.json"),
     "marker": "hebizeme or “snake torment”", "decision": "keep",
     "reason_code": "historical_torture_term_context_complete",
     "reason": "Snake torment is the exact source gloss for hebizeme and the question states its historical torture context."},
    {"fact_id": "fact-1d018345158308bd6605", "source_path": raw("kinbakutoday_aeee5ea2ccc3874a.json"),
     "marker": "In the early 1950s, kinbaku and kinbaku photography was something new", "decision": "keep",
     "reason_code": "rope_media_history_context_complete",
     "reason": "The decade provides useful context for the emergence and visual character of postwar kinbaku photography."},
    {"fact_id": "fact-2bb654a348c7338992b9", "source_path": raw("kinbakutoday_2ded51b08225bd4b.json"),
     "marker": "Yukimura ryu (with one rope)", "decision": "keep",
     "reason_code": "rope_style_context_complete",
     "reason": "The one-rope gote is explicitly situated within Yukimura-ryū and supports a useful style comparison."},
    {"fact_id": "fact-2fcc3143425b4d3b39fd", "source_path": raw("rope365_5c6e72d053d47d66.json"),
     "marker": "Is the TK dangerous by WykD Dave", "decision": "edit",
     "question": "Which TK safety resource does Rope365 list, and who wrote it?",
     "answer": "Is the TK dangerous by WykD Dave",
     "reason_code": "replace_author_lookup_with_safety_resource",
     "reason": "The edit preserves both the safety-resource title and attribution rather than testing the author's name alone."},
    {"fact_id": "fact-357124ef36227161605e", "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "informed consent of both the partners is essential", "decision": "edit",
     "question": "What does the source say is essential when BDSM roles involve unequal power?",
     "answer": "informed consent of both the partners",
     "reason_code": "replace_reductive_role_claim_with_consent",
     "reason": "The edit replaces a reductive control definition with the source's explicit consent requirement."},
    {"fact_id": "fact-3e99a6bd2b4c67934b60", "source_path": raw("rope365_049578f567a2879d.json"),
     "marker": "hishi TK for short", "decision": "drop", "reason_code": "semantic_duplicate",
     "reason": "The abbreviation duplicates retained hishi and takate-kote terminology without adding safety or structure."},
    {"fact_id": "fact-66a988eebc31a6ade1e4", "source_path": raw("rope365_682937f92222bf87.json"),
     "marker": "Album de photographies de Charles-François Jeandel by Musée d’Orsay", "decision": "keep",
     "reason_code": "historical_archive_resource_context_complete",
     "reason": "The museum attribution locates an early erotic-rope photography archive and is durable resource metadata."},
    {"fact_id": "fact-69734d1ddfd89a3214fd", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "EURIX guidelines for negotiation and establishing consent by Felix Ruckert", "decision": "keep",
     "reason_code": "consent_resource_attribution_context_complete",
     "reason": "The answer identifies the author of a specifically named negotiation-and-consent resource."},
    {"fact_id": "fact-6e8831744f619c0ee923", "source_path": raw("rope365_bc4120687a97c8c3.json"),
     "marker": "known as Josephine knot", "decision": "keep",
     "reason_code": "knot_alias_context_complete",
     "reason": "Josephine knot is a direct, practical macramé alias for the double coin knot."},
    {"fact_id": "fact-b1c31ac5d2b6bdbd3cf0", "source_path": raw("kinbakutoday_cbbff84b319d0813.json"),
     "marker": "positions hojojutsu in the erotic contexts of gay male desire and self-bondage", "decision": "edit",
     "question": "How does Kinbaku Today say Takeshi Shuichi’s 1953 essay positions hojojutsu?",
     "answer": "in the erotic contexts of gay male desire and self-bondage",
     "reason_code": "replace_publication_lookup_with_historical_context",
     "reason": "The edit retains the essay's substantive erotic and self-bondage context rather than its magazine name."},
    {"fact_id": "fact-b8f1368ec2ff4d9b828a", "source_path": raw("kinbakutoday_69840e818a0cf979.json"),
     "marker": "guerrilla bondage (aka rope bombing)", "decision": "drop", "reason_code": "volatile_or_promotional",
     "reason": "The alias appears only in a promotional book announcement and has no durable definition or practice context."},
    {"fact_id": "fact-c60deb441c3129759143", "source_path": raw("kinbakutoday_941bd97ddf6d72fa.json"),
     "marker": "Toshio Saeki, one of the fathers of modern EroGuro art", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "An honorific artist-name lookup is broad art-history trivia and omits the work's concrete relationship to kinbaku."},
    {"fact_id": "fact-d00f6772e10f963f9e17", "source_path": raw("esinem_f358e1984a8e04dc.json"),
     "marker": "takate kote, the box-tie", "decision": "keep",
     "reason_code": "technique_alias_context_complete",
     "reason": "Takate kote is directly paired with the widely used box-tie description."},
    {"fact_id": "fact-d7e6a05a62554a6dbe9d", "source_path": raw("rope365_049578f567a2879d.json"),
     "marker": "avoid rope that presses in the armpit", "decision": "keep",
     "reason_code": "upper_body_safety_context_complete",
     "reason": "Avoiding armpit pressure is a clear placement warning within explicit nerve-monitoring guidance."},
    {"fact_id": "fact-7ff4c1e29a09c5f1d863", "source_path": raw("rope365_da67938bae2edac4.json"),
     "marker": "hayanawa 早縄 (fast rope) and honnawa 本縄 (main rope)", "decision": "edit",
     "question": "What two categories are hojōjutsu techniques usually divided into?",
     "answer": "hayanawa 早縄 (fast rope) and honnawa 本縄 (main rope)",
     "reason_code": "restore_category_glosses",
     "reason": "The edit preserves both named categories with their Japanese writing and English glosses."},
    {"fact_id": "fact-0ebc180747663c175f13", "source_path": raw("rope365_4196a8f7e326a5cc.json"),
     "marker": "Kuzushi, wabi-sabi", "decision": "keep",
     "reason_code": "japanese_aesthetic_concepts_context_complete",
     "reason": "The pair identifies traditional aesthetic concepts used to discuss imperfection in rope composition."},
    {"fact_id": "fact-4e683b3c8dedcefec3a3", "source_path": raw("rope365_1ee277117c1ce420.json"),
     "marker": "shear lashing allows it to have more range of movement", "decision": "keep",
     "reason_code": "lashing_structure_context_complete",
     "reason": "The answer gives the specific lashing whose twisting structure permits changing angles."},
    {"fact_id": "fact-95e20e94e8a2d9868673", "source_path": raw("rope365_d41bd680406d8e96.json"),
     "marker": "pulling the feet in opposite directions", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A foot-and-toe shaping microinstruction lacks joint, circulation, nerve, consent, and release guidance."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v7": 718,
    "active_after_this_tranche": 710,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 8,
    "new_edits_applied": 4,
    "output_rows": 748,
    "output_sha256": "a5469d9e83eaebc14da3a8e04bf7634b19c8d9a4707333effd1bf5f774eb0bbc",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 13,
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
                 "schema": "context-merit-audit-v8", "source": row["source"],
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
    report = {"schema": "context-merit-audit-report-v8", "reviewer": REVIEWER,
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
