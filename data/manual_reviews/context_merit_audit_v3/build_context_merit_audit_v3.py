#!/usr/bin/env python3
"""Audit context-merit tranche v3: the next 25 unreviewed active QAs."""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
V2_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v2"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(V2_DIR))

import build_context_merit_audit_v2 as common
from qa_quality import normalize_text, qa_pair_from_record


DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v3.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v3.jsonl"
REPORT = OUT_DIR / "report_context_merit_v3.json"
REVIEWER = "codex-context-merit-audit-v3"
REVIEWED_AT = "2026-07-14"

ACTIVE_DATASET = common.ACTIVE_DATASET
ACTIVE_REPORT = common.ACTIVE_REPORT
ACTIVE_CURATIONS = common.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = common.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = common.QUALITY_MERIT_CURATION
TASUKI_CURATION = common.TASUKI_CURATION
CONTEXT_MERIT_V1_CURATION = common.CONTEXT_MERIT_V1_CURATION
CONTEXT_MERIT_V2_CURATION = (
    DATA / "manual_reviews" / "context_merit_audit_v2" /
    "pending_curation_context_merit_v2.jsonl"
)

file_sha256 = common.file_sha256
text_sha256 = common.text_sha256
portable = common.portable
read_jsonl = common.read_jsonl
write_jsonl = common.write_jsonl
risk_features = common.risk_features


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-ba0bcde9e13c55a82239",
        "source_path": raw("esinem_77368ccdc66acad0.json"),
        "evidence_marker": "Rope suspension falls under the term “edge play”",
        "decision": "keep",
        "reason_code": "suspension_risk_term_context_complete",
        "reason": (
            "The source directly identifies suspension as edge play and explains "
            "that real injury is possible; the concise safety term is useful."
        ),
    },
    {
        "fact_id": "fact-001d2e72ae8f7a500c26",
        "source_path": raw("rope365_5c6e72d053d47d66.json"),
        "evidence_marker":
            "Mynx’s interpretation of the box tie is influenced by her teacher Naka Akira",
        "decision": "keep",
        "reason_code": "technique_design_lineage_context_complete",
        "reason": (
            "The teacher attribution is tied directly to a box-tie interpretation "
            "and its technical and aesthetic design lineage."
        ),
    },
    {
        "fact_id": "fact-2125ce90d1d75e5ca31d",
        "source_path": raw("rope365_30eaedb9cb1e6d5b.json"),
        "evidence_marker": "This approach is popular in the Japanese style",
        "decision": "keep",
        "reason_code": "rope_length_practice_context_complete",
        "reason": (
            "The answer places a clearly named one-length rope-kit approach in "
            "the style context where the source says it is commonly used."
        ),
    },
    {
        "fact_id": "fact-899f8d3d1cd306390e00",
        "source_path": raw("rope365_627927cb0eb89fdc.json"),
        "evidence_marker": "on Youtube, Vimeo and also within the activity pages",
        "decision": "drop",
        "reason_code": "volatile_or_promotional",
        "reason": (
            "A platform-name lookup from a promotional video index is volatile "
            "and less useful than retaining the canonical Rope365 resource URL."
        ),
    },
    {
        "fact_id": "fact-27c1ea4d38aa44273817",
        "source_path": raw("kinbakutoday_e2c96f38ced1cb5d.json"),
        "evidence_marker":
            "classic magazines, published by Sun, was called SM Collector",
        "decision": "keep",
        "reason_code": "rope_media_history_context_complete",
        "reason": (
            "The magazine name, publisher, and run identify a significant "
            "second-wave Japanese SM and kinbaku publication."
        ),
    },
    {
        "fact_id": "fact-39f074e28d0a4cd40ea3",
        "source_path": raw("kinbakutoday_4330ca3210d81512.json"),
        "evidence_marker": "photo series Utsukushiki Imashime",
        "decision": "keep",
        "reason_code": "rope_photography_history_context_complete",
        "reason": (
            "The title identifies a documented early staged-bondage photography "
            "series and names its model in the question."
        ),
    },
    {
        "fact_id": "fact-4a5cebf4e405af18a2dd",
        "source_path": raw("rope365_f43c9fde09431a5f.json"),
        "evidence_marker": "Teardrop Harness",
        "evidence_lines": 2,
        "decision": "keep",
        "reason_code": "harness_design_term_context_complete",
        "reason": (
            "Teardrop Harness is the source's exact name for a functional weaving-"
            "based chest-harness design."
        ),
    },
    {
        "fact_id": "fact-d04078c171abafc3eb69",
        "source_path": raw("kinbakutoday_6f1f3d70caa0e6dc.json"),
        "evidence_marker":
            "‘Harder or softer?’ for when I need to balance my sadism against her masochism",
        "decision": "keep",
        "reason_code": "intensity_checkin_phrase_context_complete",
        "reason": (
            "The brief phrase is a concrete intensity check-in and the question "
            "states exactly when the source uses it."
        ),
    },
    {
        "fact_id": "fact-712c0acf3d4097d353a8",
        "source_path": raw("rope365_86043726f080f903.json"),
        "evidence_marker": "The diaphragm has importance for how you breathe",
        "decision": "edit",
        "question": (
            "Which breathing muscle can rope near the lower rib cage restrict by "
            "limiting its movement?"
        ),
        "answer": "the diaphragm",
        "reason_code": "clarify_anatomical_safety_context",
        "reason": (
            "The edit replaces awkward grammar with the source's concrete lower-"
            "rib-cage rope context while preserving its exact anatomical answer."
        ),
    },
    {
        "fact_id": "fact-ada66dade9582072e166",
        "source_path": raw("kinbakutoday_78b58fbf1524edc7.json"),
        "evidence_marker": "publication of the book California Club",
        "decision": "keep",
        "reason_code": "useful_named_historical_resource",
        "reason": (
            "The named book documents a specific Japanese-and-American rope-work "
            "collaboration and is useful as a historical resource."
        ),
    },
    {
        "fact_id": "fact-d5e941391d0ebc01a800",
        "source_path": raw("kinbakutoday_fd0d7ba4b6589765.json"),
        "evidence_marker": "primary rope artist of the time was Nureki Chimuo",
        "decision": "keep",
        "reason_code": "style_history_context_complete",
        "reason": (
            "The artist identification is connected to a dated suspension and "
            "semenawa style, making it substantive rope-history context."
        ),
    },
    {
        "fact_id": "fact-e39c053ace018b31331b",
        "source_path": raw("kinbakutoday_7113a15b5e5e3aa3.json"),
        "evidence_marker": "it’s called Hashira Shibari",
        "decision": "keep",
        "reason_code": "tie_term_context_complete",
        "reason": (
            "Hashira Shibari is the exact term for tying to a vertical post, with "
            "the defining structure already given in the question."
        ),
    },
    {
        "fact_id": "fact-fc7900e3dd81ee4f6f6c",
        "source_path": raw("kinbakutoday_f6ccdaa49bed3fa5.json"),
        "evidence_marker":
            "described his work with Sugiura sensei as constantly pushing the envelope",
        "decision": "edit",
        "question":
            "How did Naka sensei describe his work with Sugiura sensei?",
        "answer": (
            "constantly pushing the envelope, always looking for the limits of "
            "kinbaku in order to capture the extremes"
        ),
        "reason_code": "replace_person_lookup_with_substantive_description",
        "reason": (
            "The original asks only for a collaborator's name; the edit retains "
            "the source's meaningful description of their work."
        ),
    },
    {
        "fact_id": "fact-a23aee2e599cfbccafd5",
        "source_path": raw("rope365_2ea01101bf29d77c.json"),
        "evidence_marker": "Most twisted ropes are made from 3 strands",
        "decision": "keep",
        "reason_code": "rope_construction_context_complete",
        "reason": (
            "The strand count is an exact construction fact paired with the "
            "source's stability rationale."
        ),
    },
    {
        "fact_id": "fact-388ba93105c948d44738",
        "source_path": raw("kinbakutoday_f370696af0359092.json"),
        "evidence_marker": "“Tsuji Banzuke”(辻番付, 辻番附) were wood printed theater posters",
        "decision": "keep",
        "reason_code": "visual_rope_history_term_context_complete",
        "reason": (
            "The poster term supports the source's history of theatrical torture "
            "and shibari scenes that influenced Itō Seiu's research."
        ),
    },
    {
        "fact_id": "fact-df658316cbcd50830590",
        "source_path": raw("rope365_d7cb8892cca8b93a.json"),
        "evidence_marker": "OsakaDan who teaches Kazami Ranki style",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The row extracts an incidental teacher-style name from the author's "
            "personal learning anecdote without preserving the tasuki lesson."
        ),
    },
    {
        "fact_id": "fact-680b56c1349919593a2c",
        "source_path": raw("kinbakutoday_521c171716c46594.json"),
        "evidence_marker": "Kasumi Hourai san’s wabi-sabi style",
        "decision": "edit",
        "question": (
            "What aesthetic label does interviewer Sin associate with Kasumi "
            "Hourai’s style?"
        ),
        "answer": "wabi-sabi style",
        "reason_code": "restore_subjective_attribution",
        "reason": (
            "The original treats the interviewer's aesthetic label as canonical; "
            "the edit makes that attribution explicit."
        ),
    },
    {
        "fact_id": "fact-ec9b9dbbca11f8e3134c",
        "source_path": raw("kinbakutoday_92c9fc29a66300a0.json"),
        "evidence_marker": "“Fairy in a Cage,” written by the legendary Oniroku Dan",
        "decision": "keep",
        "reason_code": "rope_film_history_context_complete",
        "reason": (
            "The writer-film relationship is stable Japanese SM-film history for "
            "a work whose bondage advisor is central to the source."
        ),
    },
    {
        "fact_id": "fact-36c4b6643088dc0a087f",
        "source_path": raw("esinem_f2dfde25be14a7a8.json"),
        "evidence_marker": "Meirokusha a learning society founded in 1874",
        "decision": "drop",
        "reason_code": "out_of_domain_or_personal_trivia",
        "reason": (
            "The founding year of a general learning society is incidental to a "
            "quotation about early non-erotic use of the word kinbaku."
        ),
    },
    {
        "fact_id": "fact-4cc86a9034f52e41c611",
        "source_path": raw("kinbakutoday_454370c8a4f42708.json"),
        "evidence_marker": "the magazine lasted until 1975",
        "decision": "keep",
        "reason_code": "rope_media_history_context_complete",
        "reason": (
            "Kitan Club's end year is durable context for an important Japanese "
            "SM and kinbaku publication."
        ),
    },
    {
        "fact_id": "fact-6495f735547cdca55497",
        "source_path": raw("rope365_d09bf786f1bfe97f.json"),
        "evidence_marker": "kimono tie is inspired by the Japanese tasuki",
        "decision": "keep",
        "reason_code": "tie_inspiration_context_complete",
        "reason": (
            "Tasuki is the exact Japanese sleeve-tying technique identified as "
            "the design inspiration for the kimono tie."
        ),
    },
    {
        "fact_id": "fact-88343fa63a33ac34ce71",
        "source_path": raw("kinbakutoday_1c6818645cd43d8f.json"),
        "evidence_marker": "books such as Phantom Mirror from 1976",
        "decision": "keep",
        "reason_code": "useful_named_historical_resource",
        "reason": (
            "The date identifies a named, harder-to-find collection of Ozuma's "
            "bondage-focused artwork."
        ),
    },
    {
        "fact_id": "fact-8d2da18f7c39ecb52757",
        "source_path": raw("kinbakutoday_eb9778bb44eeecfc.json"),
        "evidence_marker": "SM Play: You can play S&M (published 1972)",
        "decision": "keep",
        "reason_code": "tutorial_history_context_complete",
        "reason": (
            "The publication year anchors what the source identifies as Japan's "
            "first kinbaku how-to or tutorial book."
        ),
    },
    {
        "fact_id": "fact-c1df997a17d15ae6a8b0",
        "source_path": raw("kinbakutoday_f57559bbb4c8b826.json"),
        "evidence_marker": "Uramado, which Nureki would later go on to edit",
        "decision": "keep",
        "reason_code": "rope_media_history_context_complete",
        "reason": (
            "The editor-publication relationship is meaningful context for "
            "Nureki's role in the development of modern kinbaku."
        ),
    },
    {
        "fact_id": "fact-288b385a8956a6de1618",
        "source_path": raw("kinbakutoday_521c171716c46594.json"),
        "evidence_marker": "She is often called the mother of Kinbaku",
        "decision": "edit",
        "question": (
            "What did interviewer Sin credit Yoi Yoshida with doing for bakushi "
            "previously unknown outside Japan?"
        ),
        "answer": "bringing their skills to wider attention",
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "This concise grammatical answer faithfully restates the immediately "
            "preceding source clause without retaining its promotional superlative."
        ),
        "reason_code": "replace_honorific_lookup_with_contribution",
        "reason": (
            "The original tests a broad honorific; the edit preserves the concrete "
            "contribution the interviewer cites before using it."
        ),
    },
)

EXPECTED_SELECTION = tuple(specification["fact_id"] for specification in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v2": 751,
    "active_after_this_tranche": 748,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 3,
    "new_edits_applied": 4,
    "output_rows": 786,
    "output_sha256":
        "80457faf72c8bfe9c9ccfbbf7dd6991f1a746539c38059f2b7c555a383d4a696",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 18,
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
            for value in row.get("candidate_fact_ids", []):
                if isinstance(value, str) and value.startswith("fact-"):
                    reviewed.add(value)
    return reviewed


def ranked_unreviewed(active_rows: list[dict]) -> tuple[list[dict], int, int]:
    excluded_ids = reviewed_fact_ids()
    ranked = []
    provenance_excluded = 0
    for active_index, row in enumerate(active_rows, 1):
        has_provenance = any(
            field in row for field in common.ACTIVE_REVIEW_FIELDS)
        if row["fact_id"] in excluded_ids or has_provenance:
            provenance_excluded += has_provenance
            continue
        features = risk_features(row)
        ranked.append((
            -features["risk_score"], features["question_tokens"],
            features["answer_tokens"], row["fact_id"], active_index,
            row, features,
        ))
    ranked.sort(key=lambda item: item[:4])
    return ([
        {"active_index": item[4], "row": item[5], "features": item[6]}
        for item in ranked
    ], len(excluded_ids), provenance_excluded)


def source_evidence(specification: dict, active_row: dict) -> tuple[str, str]:
    document = json.loads(specification["source_path"].read_text())
    if document["url"] != active_row["url"]:
        raise ValueError(f'{active_row["fact_id"]}: source URL drift')
    if text_sha256(document["text"]) != active_row["document_sha256"]:
        raise ValueError(f'{active_row["fact_id"]}: source text hash drift')
    matching = [line for line in document["text"].splitlines()
                if specification["evidence_marker"] in line]
    if len(matching) != 1:
        raise ValueError(
            f'{active_row["fact_id"]}: expected one evidence paragraph')
    lines = document["text"].splitlines()
    start = lines.index(matching[0])
    evidence = "\n".join(lines[
        start:start + specification.get("evidence_lines", 1)])
    support_type = specification.get("support_type", "extractive")
    answer = specification.get("answer", active_row["answer"])
    if (support_type == "extractive" and
            normalize_text(answer) not in normalize_text(evidence)):
        raise ValueError(f'{active_row["fact_id"]}: answer absent from evidence')
    if (support_type == "manual_paraphrase" and
            not specification.get("paraphrase_rationale")):
        raise ValueError(f'{active_row["fact_id"]}: missing paraphrase rationale')
    return evidence, support_type


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    active_rows = read_jsonl(ACTIVE_DATASET)
    ranked, excluded_count, provenance_excluded = ranked_unreviewed(active_rows)
    selected = ranked[:25]
    selected_ids = tuple(item["row"]["fact_id"] for item in selected)
    if selected_ids != EXPECTED_SELECTION:
        raise ValueError(
            "deterministic selection drift:\nexpected " +
            repr(EXPECTED_SELECTION) + "\nfound " + repr(selected_ids))

    audits = []
    curations = []
    for audit_index, (specification, ranked_item) in enumerate(
            zip(SPECS, selected), 1):
        row = ranked_item["row"]
        fact_id = specification["fact_id"]
        if row["fact_id"] != fact_id:
            raise ValueError(f"{fact_id}: selected row drift")
        question, answer = qa_pair_from_record(row)
        evidence, support_type = source_evidence(specification, row)
        decision = specification["decision"]
        audit = {
            "active_answer": answer,
            "active_index": ranked_item["active_index"],
            "active_question": question,
            "audit_index": audit_index,
            "decision": decision,
            "document_sha256": row["document_sha256"],
            "fact_id": fact_id,
            "reason": specification["reason"],
            "reason_code": specification["reason_code"],
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "risk_features": ranked_item["features"],
            "schema": "context-merit-audit-v3",
            "source": row["source"],
            "source_document": portable(specification["source_path"]),
            "source_document_file_sha256": file_sha256(
                specification["source_path"]),
            "source_support": support_type,
            "support_evidence": evidence,
            "support_evidence_sha256": text_sha256(evidence),
            "url": row["url"],
        }
        if decision == "edit":
            audit.update({
                "edited_answer": specification["answer"],
                "edited_question": specification["question"],
            })
            if support_type == "manual_paraphrase":
                audit["paraphrase_rationale"] = specification[
                    "paraphrase_rationale"]
        audits.append(audit)

        if decision in {"drop", "edit"}:
            curation = {
                "action": decision,
                "document_sha256": row["document_sha256"],
                "evidence_url": row["url"],
                "expected_answer": answer,
                "expected_question": question,
                "fact_id": fact_id,
                "reason": specification["reason"],
                "reason_code": specification["reason_code"],
                "reviewed_at": REVIEWED_AT,
                "reviewer": REVIEWER,
                "source_lineage": row["source_lineage"],
            }
            if decision == "edit":
                curation.update({
                    "answer": specification["answer"],
                    "evidence": evidence,
                    "question": specification["question"],
                    "support_type": support_type,
                })
                if support_type == "manual_paraphrase":
                    curation["paraphrase_rationale"] = specification[
                        "paraphrase_rationale"]
            curations.append(curation)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    decisions = collections.Counter(row["decision"] for row in audits)
    reasons = collections.Counter(row["reason_code"] for row in audits)
    prior_curations = (
        QUALITY_MERIT_CURATION, TASUKI_CURATION,
        CONTEXT_MERIT_V1_CURATION, CONTEXT_MERIT_V2_CURATION,
    )
    report = {
        "schema": "context-merit-audit-report-v3",
        "reviewer": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "status": "segregated_pending_not_promoted",
        "selection": {
            "active_rows": len(active_rows),
            "eligible_unreviewed_rows": len(ranked),
            "excluded_ledger_fact_ids": excluded_count,
            "excluded_active_review_provenance": provenance_excluded,
            "rows_selected": 25,
            "fact_ids_in_rank_order": list(selected_ids),
            "ranking": {
                "score": (
                    "short_question_points + 3*pronoun_count + "
                    "bare_answer_points + named_person_trivia_points"
                ),
                "tie_break": (
                    "risk_score descending, question tokens ascending, answer "
                    "tokens ascending, fact_id ascending"
                ),
            },
        },
        "audit": {
            "path": portable(AUDIT), "sha256": file_sha256(AUDIT),
            "rows": 25, "by_decision": dict(sorted(decisions.items())),
            "by_reason": dict(sorted(reasons.items())),
        },
        "new_pending_curation": {
            "path": portable(CURATION), "sha256": file_sha256(CURATION),
            "decisions": len(curations),
            "by_action": dict(sorted(collections.Counter(
                row["action"] for row in curations).items())),
            "edit_support_types": dict(sorted(collections.Counter(
                row.get("support_type", "extractive")
                for row in curations if row["action"] == "edit").items())),
        },
        "prior_pending": {
            "additions": [
                {"path": portable(path), "rows": len(read_jsonl(path)),
                 "sha256": file_sha256(path)}
                for path in PRIOR_PENDING_ADDITIONS
            ],
            "curations": [
                {"path": portable(path), "rows": len(read_jsonl(path)),
                 "sha256": file_sha256(path)} for path in prior_curations
            ],
        },
        "active_baseline": {
            "dataset": {"path": portable(ACTIVE_DATASET),
                        "rows": len(active_rows),
                        "sha256": file_sha256(ACTIVE_DATASET)},
            "report": {"path": portable(ACTIVE_REPORT),
                       "sha256": file_sha256(ACTIVE_REPORT)},
            "curation": [
                {"path": portable(path), "sha256": file_sha256(path)}
                for path in ACTIVE_CURATIONS
            ],
        },
        "sealed_evaluation_policy": {
            "manual_review_opened_eval_or_heldout_content": False,
            "generator_opens_eval_or_heldout_content": False,
            "collision_check": (
                "sealed evaluation paths are handled only by existing "
                "build_curated_qa.py during isolated projection"
            ),
        },
        "isolated_build_projection": ISOLATED_PROJECTION,
        "validation": {
            "combined_tests_expected": 17,
            "test_paths": [
                "data/manual_reviews/context_merit_audit_v3/"
                "test_context_merit_audit_v3.py",
                "test_build_curated_qa.py",
            ],
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
