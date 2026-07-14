#!/usr/bin/env python3
"""Audit context-merit tranche v7 without rewriting prior tranches."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V6_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v6"
sys.path[:0] = [str(ROOT), str(V6_DIR)]

import build_context_merit_audit_v6 as previous
from qa_quality import qa_pair_from_record

common = previous.common
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v7.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v7.jsonl"
REPORT = OUT_DIR / "report_context_merit_v7.json"
REVIEWER = "codex-context-merit-audit-v7"
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
    for version in (1, 2, 3, 4, 5, 6))
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
    {"fact_id": "fact-09b7cf8b4887281e153d", "source_path": raw("kinbakutoday_c799ecf7b51f866b.json"),
     "marker": "often termed, in Japanese, a “ryuu.”", "decision": "keep",
     "reason_code": "rope_school_term_context_complete",
     "reason": "Ryuu is explicitly defined as a unified rope school or approach, making the term reusable."},
    {"fact_id": "fact-0fd37dbddac15fb0325d", "source_path": raw("kinbakutoday_cbbff84b319d0813.json"),
     "marker": "hojojutsu, the older Japanese arts of capture and restraint", "decision": "keep",
     "reason_code": "restraint_history_context_complete",
     "reason": "The question accurately names hojojutsu's older capture-and-restraint context without claiming a simple kinbaku lineage."},
    {"fact_id": "fact-34b0c5871898aba9c5c3", "source_path": raw("kinbakutoday_cbbff84b319d0813.json"),
     "marker": "prevention of escape their primary object", "decision": "edit",
     "question": "What primary purpose of hojojutsu did Tsujimura contrast with kinbaku’s pursuit of aesthetic beauty?",
     "answer": "the prevention of escape", "reason_code": "replace_year_with_aesthetic_distinction",
     "reason": "The edit replaces an isolated year with the substantive functional distinction in Tsujimura's argument."},
    {"fact_id": "fact-8866b2077eec8f5c877c", "source_path": raw("esinem_3d33420019ff74ab.json"),
     "marker": "e.g. neobari", "decision": "drop", "reason_code": "volatile_or_promotional",
     "reason": "A passing style label in promotional copy is undefined and gives no durable technique or lineage context."},
    {"fact_id": "fact-97d5378c5c21c4383afd", "source_path": raw("kinbakutoday_edba1220873364c8.json"),
     "marker": "illustrations of 26 ties, with commentary", "decision": "edit",
     "question": "What does the 1930 Secret View of Hojojutsu volume provide beyond its two-color diagrams?",
     "answer": "illustrations of 26 ties, with commentary, from one particular school of hojojutsu",
     "reason_code": "replace_diagram_count_with_document_scope",
     "reason": "The edit replaces a diagram-count quiz with the volume's useful instructional and school scope."},
    {"fact_id": "fact-c49ef46be45ea5bf079e", "source_path": raw("kinbakutoday_dc711609dfc7a35f.json"),
     "marker": "create the nawajiri, the point of connection", "decision": "keep",
     "reason_code": "technique_term_context_complete",
     "reason": "Nawajiri is directly defined as a point of connection, control, and communication in a one-rope gote."},
    {"fact_id": "fact-ebe594640d31a72fc120", "source_path": raw("kinbakutoday_c82ab35fef320690.json"),
     "marker": "one of the earliest representations of erotic bondage in Japan", "decision": "edit",
     "question": "Why does Kinbaku Today call the 1925 publication of Sawara Kise’s snow photographs historically important?",
     "answer": "it is one of the earliest representations of erotic bondage in Japan",
     "reason_code": "replace_year_with_historical_significance",
     "reason": "The edit preserves why the images matter instead of testing their year alone."},
    {"fact_id": "fact-42b01632d4f86565fc0a", "source_path": raw("rope365_f43c9fde09431a5f.json"),
     "marker": "moving around, breathing, lifting your arms, curling your back", "decision": "edit",
     "question": "How does Rope365 suggest checking the stability of a diagonal chest harness?",
     "answer": "moving around, breathing, lifting your arms, curling your back",
     "reason_code": "replace_body_label_with_stability_check",
     "reason": "The edit replaces a body-shape label with an actionable, partner-discussed harness stability check."},
    {"fact_id": "fact-569503ae1926b6987900", "source_path": raw("rope365_a3b2e9e479b0c70f.json"),
     "marker": "medium sized (8-10m)", "decision": "keep",
     "reason_code": "source_attributed_equipment_guidance",
     "reason": "The length is explicitly scoped to a typical Western-inspired kit rather than asserted as universal."},
    {"fact_id": "fact-6141492fade8325a49be", "source_path": raw("rope365_0b4f44c8fabfc202.json"),
     "marker": "Coconut is sometimes used in the brewing process", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A brewing-store sourcing fact sits beside an unsupported skin-safety claim and could encourage unsuitable rope use."},
    {"fact_id": "fact-6215416a2438227520a3", "source_path": raw("kinbakutoday_381214111022a952.json"),
     "marker": "cognitive use of Kotobazeme", "decision": "keep",
     "reason_code": "style_communication_term_context_complete",
     "reason": "Kotobazeme is a named verbal technique in Yukimura-ryū and the question describes its communicative use."},
    {"fact_id": "fact-aa51767b03dd040b7b37", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "known as spotter", "decision": "keep", "reason_code": "self_tie_safety_term_context_complete",
     "reason": "Spotter is practical self-tie safety vocabulary, with supervision context supplied directly."},
    {"fact_id": "fact-e5ce7819dad034e9fc2c", "source_path": raw("rope365_682937f92222bf87.json"),
     "marker": "European fetish illustrator Carlõ", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "An isolated illustrator-name lookup is visual-culture trivia and omits any concrete relationship to rope practice."},
    {"fact_id": "fact-58af822afb984d0315d9", "source_path": raw("kinbakutoday_011f67c75b8f999f.json"),
     "marker": "Humility is necessary to question oneself", "decision": "keep",
     "reason_code": "reflective_practice_principle_context_complete",
     "reason": "The answer captures the source's concise principle for continuing self-questioning in kinbaku."},
    {"fact_id": "fact-5e1831451feb28ca6497", "source_path": raw("wikipedia_19338018629394d7.json"),
     "marker": "working end of the rope as a rabbit", "decision": "keep", "reason_code": "knot_mnemonic_context_complete",
     "reason": "The rabbit is a common bowline mnemonic and the question fully identifies what the image represents."},
    {"fact_id": "fact-671eb475f3a96a9a26e5", "source_path": raw("esinem_f070e94e0fc41757.json"),
     "marker": "Pizza Parlour in 2010", "decision": "drop", "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "The year and venue of a teacher's discovery are personal anecdotes rather than Kazami-style knowledge."},
    {"fact_id": "fact-8a4114e4bb909424294b", "source_path": raw("kinbakutoday_c799ecf7b51f866b.json"),
     "marker": "The psychological background of the impulse to do Seme", "decision": "edit",
     "question": "What topic did Tsujimura Takashi’s 1953 Kitan Club work address?",
     "answer": "the psychological background of the impulse to do Seme",
     "reason_code": "replace_year_with_article_subject",
     "reason": "The edit retains the work's substantive historical topic instead of another bare 1953 lookup."},
    {"fact_id": "fact-b277f525a0012a481fd4", "source_path": raw("esinem_0e05c5a35e129510.json"),
     "marker": "masters like Yukimura can achieve", "decision": "keep",
     "reason_code": "rope_handling_principle_context_complete",
     "reason": "The example supports a useful principle: skilled handling can achieve results with little rope and simple ties."},
    {"fact_id": "fact-5856f8f0293c373a066f", "source_path": raw("rope365_5f637b24ae808142.json"),
     "marker": "important part of breathing", "decision": "edit",
     "question": "What breathing consideration does Rope365 give when adding a waistline?",
     "answer": "it is also an important part of breathing, so adding a waistline can be a part of play and intensity",
     "reason_code": "replace_anchor_claim_with_breathing_warning",
     "reason": "The edit preserves the source's full breathing-and-intensity context instead of a bare body-part anchor claim."},
    {"fact_id": "fact-11facb540676c51e7ebb", "source_path": raw("rope365_8ae9e3d93b31601b.json"),
     "marker": "Predicament – A situation", "decision": "keep", "reason_code": "bondage_structure_term_context_complete",
     "reason": "Predicament is accurately defined through constrained agency among negative options."},
    {"fact_id": "fact-ca8bc45c9aec3d993ec3", "source_path": raw("rope365_d1dab5f5fa3d5cbc.json"),
     "marker": "tie the rest of the arms to the torso", "decision": "drop", "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A restriction-increasing micro-step provides no circulation, nerve, consent, or release guidance."},
    {"fact_id": "fact-17b5a49e3a54220c3c36", "source_path": raw("kinbakutoday_c5e568667b495473.json"),
     "marker": "正座 seiza", "decision": "keep", "reason_code": "position_term_context_complete",
     "reason": "Seiza is a useful position term and the question fully describes the posture."},
    {"fact_id": "fact-b7828e0890bc728bce44", "source_path": raw("kinbakutoday_89994b45562f9ad8.json"),
     "marker": "hojō to capture criminals", "decision": "drop", "reason_code": "semantic_duplicate",
     "reason": "The retained hojojutsu definition already provides the older capture-and-restraint function more clearly."},
    {"fact_id": "fact-18128e1fc133eaf19e89", "source_path": raw("esinem_77368ccdc66acad0.json"),
     "marker": "Risk Aware Consensual Kink", "decision": "keep", "reason_code": "risk_framework_term_context_complete",
     "reason": "RACK is core risk-and-consent vocabulary and the acronym is expanded exactly."},
    {"fact_id": "fact-048e419a15cfc8118217", "source_path": raw("rope365_5a4dc6c711aaa005.json"),
     "marker": "quite common in twisted ropes", "decision": "keep", "reason_code": "rope_care_context_complete",
     "reason": "The answer adds a practical material context to the retained definition of high-stranding."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v6": 724,
    "active_after_this_tranche": 718,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 6,
    "new_edits_applied": 6,
    "output_rows": 756,
    "output_sha256": "7c51f2ccc81196dab0e0d5eb441df77cd52dc2180519572ccf2cdfc2de481622",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 13,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def reviewed_fact_ids() -> set[str]:
    reviewed = set()
    for path in sorted((DATA / "manual_reviews").rglob("*.jsonl")):
        if OUT_DIR in path.parents:
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
                 "schema": "context-merit-audit-v7", "source": row["source"],
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
    report = {"schema": "context-merit-audit-report-v7", "reviewer": REVIEWER,
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
