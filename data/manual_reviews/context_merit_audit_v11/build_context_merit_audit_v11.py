#!/usr/bin/env python3
"""Audit context-merit tranche v11 without rewriting prior tranches."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V10_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v10"
sys.path[:0] = [str(ROOT), str(V10_DIR)]
import build_context_merit_audit_v10 as previous
from qa_quality import qa_pair_from_record

common = previous.common
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v11.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v11.jsonl"
REPORT = OUT_DIR / "report_context_merit_v11.json"
REVIEWER = "codex-context-merit-audit-v11"
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
    for version in range(1, 11)
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


def resource(name: str) -> Path:
    return DATA / "raw" / "rope_resources_v1" / name


SPECS = (
    {"fact_id": "fact-ad61009350df05ecd6a1", "source_path": resource("rope365__3c086f7e4e6983e8b57f.json"),
     "marker": "Pull-on the tie in different directions to measure its solidity", "decision": "edit",
     "question": "How does Rope365’s self-evaluation checklist suggest checking a chest harness’s solidity?",
     "answer": "pull on the tie in different directions", "reason_code": "replace_pronoun_lookup_with_evaluation_method",
     "reason": "The edit replaces a pronoun-dependent property lookup with the source's concrete self-evaluation method."},
    {"fact_id": "fact-cbc2e6c8ee57d6ed2de1", "source_path": raw("kinbakutoday_5783480db327a3ba.json"),
     "marker": "photo collection Utsukushiki Shibari", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "A retained edited QA from this page already preserves Kawabata Tanako's more meaningful historical voice and significance."},
    {"fact_id": "fact-dbe28e502714693bae78", "source_path": raw("kinbakutoday_e92a74807fe69d15.json"),
     "marker": "something that is so rare, precious and valued that it is very rarely shown or shared with others", "decision": "edit",
     "question": "What does mongai fushutsu mean in the description of Ozuma’s work?",
     "answer": "something that is so rare, precious and valued that it is very rarely shown or shared with others",
     "reason_code": "replace_term_lookup_with_complete_definition",
     "reason": "The edit makes the source's complete definition retrievable instead of testing only the Japanese label."},
    {"fact_id": "fact-f0fef2ee956cc106d74a", "source_path": resource("rope365__22f7d5d8e78a36e5cbaa.json"),
     "marker": "Tanuki shibari (Racoon tie)", "decision": "keep",
     "reason_code": "pose_alias_context_complete",
     "reason": "The source directly pairs Tanuki shibari with the Tuck Position and Racoon-tie alias."},
    {"fact_id": "fact-f1b2eecdde34a2b86f2d", "source_path": raw("esinem_d20c9dbd4e144989.json"),
     "marker": "average of 10 wraps of rope around a cylinder with a vernier caliper", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "A retained QA already teaches the ten-wrap averaging method; this tool-name lookup adds little distinct value."},
    {"fact_id": "fact-feadf2d476a11a4a1140", "source_path": resource("rope365__077aecb334166abba18c.json"),
     "marker": "Nylon/MFP are very slippery which allows tying complex knots faster than a high friction rope", "decision": "keep",
     "reason_code": "source_attributed_material_characteristic",
     "reason": "The material identification is explicitly limited to Rope365's comparison of knot-tying friction and speed."},
    {"fact_id": "fact-1206e62b7a2b89262497", "source_path": resource("rope365__1f02fa6488558b05bf1f.json"),
     "marker": "using your hands to wiggle the rope or a vibrator", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "The object lookup from one sensory exploration prompt is too narrow to improve general rope-bondage assistance."},
    {"fact_id": "fact-2b8aa0be6da3f5ebfeb1", "source_path": raw("kinbakutoday_c799ecf7b51f866b.json"),
     "marker": "muga” (無我), which refers to a state of selflessness or self-effacement", "decision": "edit",
     "question": "What does muga (無我) mean in Kinbaku Today’s discussion of rope anarchy?",
     "answer": "a state of selflessness or self-effacement", "reason_code": "replace_term_lookup_with_complete_definition",
     "reason": "The edit makes the source-defined meaning the answer and preserves the warning not to conflate it with rope anarchy."},
    {"fact_id": "fact-35f7fb97761789cef478", "source_path": resource("rope365__772d43d614d87876db29.json"),
     "marker": "we can’t untie from the bight or use it later on as an attachment point", "decision": "edit",
     "question": "What tradeoff does Rope365 note when its lark’s-head single-column tie has no bight?",
     "answer": "we can’t untie from the bight or use it later on as an attachment point",
     "reason_code": "replace_knot_lookup_with_operational_tradeoff",
     "reason": "The edit preserves the no-bight tradeoff instead of asking the model to identify a knot from incidental construction clues."},
    {"fact_id": "fact-58a68a30cde8f76a3afa", "source_path": resource("rope365__22f7d5d8e78a36e5cbaa.json"),
     "marker": "both wrists and both ankles all tie together in this classic kidnapping pose", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A four-limb restriction prompt supplies no circulation, nerve, consent, monitoring, or release guidance."},
    {"fact_id": "fact-84ff8ac41974d117f106", "source_path": resource("rope365__1f02fa6488558b05bf1f.json"),
     "marker": "pass a finger between the skin and the rope and slide", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "The skin-and-rope sliding micro-step lacks friction-injury precautions and is grammatically incomplete as generated."},
    {"fact_id": "fact-95a99acebc4142e85ed9", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "a second rope between 7-8 meters will be sufficient", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A body-dependent box-tie rope length is decontextualized from fit, anatomy, construction, and risk management."},
    {"fact_id": "fact-2002c92260ac3a3d80a0", "source_path": resource("rope365__f9753746348ca9cc2f11.json"),
     "marker": "Learning in person is the best way to discover rope", "decision": "keep",
     "reason_code": "learning_resource_guidance_context_complete",
     "reason": "The recommendation is qualified by tactile learning needs, online access, critical thinking, and evolving safety knowledge."},
    {"fact_id": "fact-0baf582cb52116a482ed", "source_path": raw("kinbakutoday_aeee5ea2ccc3874a.json"),
     "marker": "two compelling images from 1954, Fuzoku Soshi, vol 1", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "The volume-title lookup comes from a personal reflection and adds no durable historical claim about the images."},
    {"fact_id": "fact-ef984720dbea72deed6d", "source_path": raw("kinbakutoday_cbbff84b319d0813.json"),
     "marker": "modern kinbaku inherits at least part of its technical vocabulary from hojojutsu", "decision": "edit",
     "question": "What hojojutsu–kinbaku position did Master K’s The Beauty of Kinbaku popularize in English?",
     "answer": "modern kinbaku inherits at least part of its technical vocabulary from hojojutsu",
     "reason_code": "replace_book_lookup_with_historical_position",
     "reason": "The edit preserves the attributed historical position and avoids reducing a contested genealogy to a title quiz."},
    {"fact_id": "fact-199223208dc335e8f1b5", "source_path": raw("kinbakutoday_c799ecf7b51f866b.json"),
     "marker": "understanding the foundations and the principles were the essence of Yukimura Ryuu", "decision": "edit",
     "question": "What did Yukimura consider the essence of Yukimura Ryuu?",
     "answer": "understanding the foundations and the principles", "reason_code": "replace_school_lookup_with_principle",
     "reason": "The edit makes Yukimura's stated principle the answer instead of awkwardly asking for the school already implied by the question."},
    {"fact_id": "fact-b34aaf4b8874aa5cfeff", "source_path": raw("kinbakutoday_c49ee02a4ad5a3d4.json"),
     "marker": "attitude with which it is frequently taught", "decision": "edit",
     "question": "Besides how often it is tied, what does Kinbaku Today say contributes to takate kote’s higher incident rate?",
     "answer": "the attitude with which it is frequently taught", "reason_code": "replace_tie_lookup_with_incident_context",
     "reason": "The edit retains the source's safety-relevant teaching critique rather than another bare tie-name answer."},
    {"fact_id": "fact-bf9c4d0b59850a2dc1f0", "source_path": raw("kinbakutoday_54fd61832eaafa33.json"),
     "marker": "using FOUR strings rather than two strings to make a coil", "decision": "drop",
     "reason_code": "overgeneralized_from_personal_anecdote",
     "reason": "The generated row turns one author's personal decorative-coil speed trick into a general recommendation."},
    {"fact_id": "fact-a81264b699fcb00f8a2e", "source_path": resource("rope365__ac5f45d316f177fc0d38.json"),
     "marker": "Also called Egyptian or sarcophagus tie", "decision": "keep",
     "reason_code": "pose_alias_context_complete",
     "reason": "The source directly pairs the descriptive aliases with the crossing-arms-on-chest position."},
    {"fact_id": "fact-2483854a1ce91c077032", "source_path": resource("rope365__5d7a851080ae72c9620d.json"),
     "marker": "mayfly pattern is created when we tie inline cuffs in a zig-zag pattern", "decision": "keep",
     "reason_code": "pattern_term_context_complete",
     "reason": "The named pattern is defined by its zig-zag cuff symmetry rather than presented as an unsupported load claim."},
    {"fact_id": "fact-9e8a20aa3ad29af076ab", "source_path": resource("rope365__3c086f7e4e6983e8b57f.json"),
     "marker": "moves the rope away from the clavicles and makes an angle that creates more natural pressure on the body", "decision": "edit",
     "question": "Why does Rope365 say the Open V chest harness can be more comfortable?",
     "answer": "it moves the rope away from the clavicles and makes an angle that creates more natural pressure on the body",
     "reason_code": "replace_anatomy_lookup_with_qualified_comfort_rationale",
     "reason": "The edit preserves the complete source-attributed comfort rationale instead of testing one anatomical noun."},
    {"fact_id": "fact-a236300d1e531bd987b5", "source_path": resource("rope365__2db0b0e55d35bf0aa676.json"),
     "marker": "The overhand hank coiling method is easy, fast and doesn’t add unnecessary twists and stress on the rope for storage", "decision": "keep",
     "reason_code": "rope_storage_method_context_complete",
     "reason": "The named coiling method is directly tied to a durable rope-storage benefit and complements the retained loose-knot precaution."},
    {"fact_id": "fact-aa0901ec4009914c47aa", "source_path": resource("rope365__ac5f45d316f177fc0d38.json"),
     "marker": "Aka straightjacket tie, front box tie", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "The colloquial, misspelled alias adds little beyond the position already described and omits practical safety context."},
    {"fact_id": "fact-c08b6ebe47d0100aa85c", "source_path": resource("rope365__7c520e466d199159381f.json"),
     "marker": "inspired by the work of Yukimura Haruki", "decision": "drop",
     "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "A bare inspiration attribution is less useful than the retained safety precaution from the same tie discussion."},
    {"fact_id": "fact-ce4a67c55c94743ad476", "source_path": raw("kinbakutoday_f57559bbb4c8b826.json"),
     "marker": "aesthetics of the stage show dominate the Akechi lineage, the Minomura approach focuses on the aesthetic of the woman", "decision": "edit",
     "question": "How does Kinbaku Today contrast the aesthetics of the Akechi and Minomura lineages?",
     "answer": "the aesthetics of the stage show dominate the Akechi lineage, the Minomura approach focuses on the aesthetic of the woman",
     "reason_code": "replace_style_lookup_with_aesthetic_comparison",
     "reason": "The edit preserves the source's substantive lineage comparison rather than asking which style matches a single technical clue."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v10": 690,
    "active_after_this_tranche": 680,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 10,
    "new_edits_applied": 9,
    "output_rows": 718,
    "output_sha256": "948b14244c493d360c8090d425cc20d3c584c157cfd8e467338008b9d16c4abf",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 6,
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
                 "schema": "context-merit-audit-v11", "source": row["source"],
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
    report = {"schema": "context-merit-audit-report-v11", "reviewer": REVIEWER,
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
