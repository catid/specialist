#!/usr/bin/env python3
"""Audit context-merit tranche v10 without rewriting prior tranches."""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V9_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v9"
sys.path[:0] = [str(ROOT), str(V9_DIR)]
import build_context_merit_audit_v9 as previous
from qa_quality import qa_pair_from_record

common = previous.common
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v10.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v10.jsonl"
REPORT = OUT_DIR / "report_context_merit_v10.json"
REVIEWER = "codex-context-merit-audit-v10"
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
    for version in range(1, 10)
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
    {"fact_id": "fact-30ee6fab83448a85a5dd", "source_path": raw("kinbakutoday_f8d47d490455d57a.json"),
     "marker": "kata ashi (single leg suspension)", "decision": "edit",
     "question": "What does kata ashi mean in Yukimura’s suspension context?",
     "answer": "single leg suspension", "reason_code": "replace_name_lookup_with_term_gloss",
     "reason": "The source directly glosses kata ashi; the edit makes that reusable meaning the answer."},
    {"fact_id": "fact-41730b05fae2c6d5c530", "source_path": raw("kinbakutoday_1f811a9abc9520b4.json"),
     "marker": "psychological kinbaku’ – an extension of rope in words", "decision": "edit",
     "question": "How does Aotsuki describe Kotobazeme?",
     "answer": "psychological kinbaku – an extension of rope in words",
     "reason_code": "replace_partial_answer_with_complete_definition",
     "reason": "The edit preserves the source's complete description instead of reducing it to a two-word label."},
    {"fact_id": "fact-43799bd9c38c17f05b26", "source_path": raw("rope365_7fe5be31cb8dc67e.json"),
     "marker": "two ropes of the same size, or a smaller bight to a larger rope’s running ends", "decision": "edit",
     "question": "When does Rope365 say a sheet bend is appropriate for joining two ropes?",
     "answer": "two ropes of the same size, or a smaller bight to a larger rope’s running ends",
     "reason_code": "replace_author_preference_with_joining_condition",
     "reason": "The edit asks for the source's stated joining condition, which is more useful than a bare knot preference."},
    {"fact_id": "fact-68ee2829604b440ada03", "source_path": raw("rope365_bdb2a15da368be35.json"),
     "marker": "For more solidity, try using more complex frictions like the crossing hitches", "decision": "drop",
     "reason_code": "unsafe_or_medically_unsupported",
     "reason": "A restriction-increasing frog-tie micro-step lacks circulation, nerve, consent, and release context."},
    {"fact_id": "fact-e671d0e62ba89b932261", "source_path": raw("esinem_0b1850ee9a40c337.json"),
     "marker": "sensitive region where three major nerves are located", "decision": "edit",
     "question": "What does Esinem say is located in the sensitive upper-cinch region?",
     "answer": "three major nerves", "reason_code": "replace_function_lookup_with_anatomy_warning",
     "reason": "The edit retains the source's anatomical warning instead of an isolated tie-construction label."},
    {"fact_id": "fact-3ed42b45f003efc3820b", "source_path": raw("esinem_bebe8839b92231b9.json"),
     "marker": "secure and comforting", "decision": "drop",
     "reason_code": "overgeneralized_psychological_claim",
     "reason": "The answer generalizes a subjective metaphor into an expected psychological effect of rope."},
    {"fact_id": "fact-a2f7e13d445717ecdd37", "source_path": raw("kinbakutoday_454370c8a4f42708.json"),
     "marker": "training session (講習会)", "decision": "keep",
     "reason_code": "historical_training_term_context_complete",
     "reason": "The translated Japanese term is explicitly tied to the historical training sessions discussed."},
    {"fact_id": "fact-133008d0ea45d5ccf254", "source_path": raw("rope365_729efcd749abbd37.json"),
     "marker": "cross at 90 degrees on a flat plane", "decision": "keep",
     "reason_code": "knot_geometry_context_complete",
     "reason": "The answer states the source's concrete geometric condition for minimizing directional difference."},
    {"fact_id": "fact-1c162569dda4ddcdcb87", "source_path": raw("rope365_f4ec955e36db6b06.json"),
     "marker": "Look at Baladi dancers", "decision": "drop",
     "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "A dancer-style lookup from a balance analogy has little reusable value for rope-bondage assistance."},
    {"fact_id": "fact-2edd73476d647026bde4", "source_path": raw("kinbakutoday_91bd5cf512bd9ecf.json"),
     "marker": "focused exclusively on tying men", "decision": "keep",
     "reason_code": "rope_media_history_context_complete",
     "reason": "The demographic focus is a source-grounded distinction in the history of Japanese instructional rope media."},
    {"fact_id": "fact-3194a975ddec21ab1705", "source_path": raw("rope365_27a1135f100eebcf.json"),
     "marker": "super middle is the new middle", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "The coined course shorthand is not broadly established and is unclear without the surrounding folding exercise."},
    {"fact_id": "fact-37f5e669dedb1fe5da30", "source_path": raw("rope365_441f9cc87ead6159.json"),
     "marker": "cross-legged pose aka agura shibari", "decision": "keep",
     "reason_code": "pose_term_context_complete",
     "reason": "The source directly pairs agura shibari with the cross-legged pose."},
    {"fact_id": "fact-38de97660f344b03efb8", "source_path": raw("rope365_8012cecdbbfc3fc6.json"),
     "marker": "not to trap the fingers against the forearms", "decision": "keep",
     "reason_code": "escapability_safety_context_complete",
     "reason": "The finger-placement warning is a concrete, source-grounded precaution for preserving escapability."},
    {"fact_id": "fact-44b8628b723246a6a2a5", "source_path": raw("kinbakutoday_209cfdfa24ad7561.json"),
     "marker": "ninety percent of Yoji Muku’s kinbaku pictures", "decision": "keep",
     "reason_code": "source_attributed_art_provenance",
     "reason": "The percentage is explicitly attributed to Nureki's account of the artist's photographic references."},
    {"fact_id": "fact-6676997d84fcf7ca74ac", "source_path": raw("rope365_da67938bae2edac4.json"),
     "marker": "coil that leash in a chain sinnet", "decision": "keep",
     "reason_code": "historical_restraint_knot_context_complete",
     "reason": "The chain sinnet is identified in a bounded historical transport-restraint context."},
    {"fact_id": "fact-727116892d2372a8da55", "source_path": raw("kinbakutoday_8057255610ff3d29.json"),
     "marker": "Hojōjutsu-type shibari", "decision": "keep",
     "reason_code": "historical_image_context_complete",
     "reason": "The answer identifies the restraint tradition shown in the named historical picture scroll."},
    {"fact_id": "fact-9814c2656b67d9b09cf4", "source_path": raw("esinem_3bbb8f77f7d062a7.json"),
     "marker": "designed the much-acclaimed Somerville Bowline", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "A retained QA already identifies the Somerville bowline as a single-column-tie variation."},
    {"fact_id": "fact-b600d100395168a1fce7", "source_path": raw("kinbakutoday_5783480db327a3ba.json"),
     "marker": "modeled for the magazine Kitan Club", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "A retained edited QA already captures Kawabata Tanako's more useful historical voice and perspective."},
    {"fact_id": "fact-c94dc475d83f73b18d19", "source_path": raw("kinbakutoday_f57559bbb4c8b826.json"),
     "marker": "working with Minomura Kou", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "A retained edited QA already describes Minomura Kou's substantive roles instead of this person-name lookup."},
    {"fact_id": "fact-d5a388ed0511d14804de", "source_path": raw("kinbakutoday_edba1220873364c8.json"),
     "marker": "during the Tokugawa Shogunate", "decision": "keep",
     "reason_code": "restraint_history_context_complete",
     "reason": "The period is directly tied to the shift from military restraint toward law-enforcement use."},
    {"fact_id": "fact-eb8c105adce5e1d530fc", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "triple-negative", "decision": "keep",
     "reason_code": "communication_safety_context_complete",
     "reason": "Avoiding triple negatives is a concrete communication precaution for nervous or distressed partners."},
    {"fact_id": "fact-5f1c7e21663cfc5c41d5", "source_path": raw("kinbakutoday_381214111022a952.json"),
     "marker": "samurai movie dramas", "decision": "keep",
     "reason_code": "historical_media_influence_context_complete",
     "reason": "The source explicitly names the media through which Yukimura first encountered kinbaku imagery."},
    {"fact_id": "fact-22cca55cd781ffc08765", "source_path": raw("kinbakutoday_490a87ec78c3d64b.json"),
     "marker": "real name Suma Toshiyuki", "decision": "drop",
     "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "An artist's real-name lookup is personal trivia and omits the work or historical contribution."},
    {"fact_id": "fact-2c85b2bf452ff523496b", "source_path": raw("rope365_0b4f44c8fabfc202.json"),
     "marker": "Holding friction – The capacity", "decision": "keep",
     "reason_code": "rope_property_definition_context_complete",
     "reason": "Holding friction is directly defined as the rope's capacity to hold while wrapping on itself."},
    {"fact_id": "fact-36538fb9cc5c96581f9e", "source_path": raw("kinbakutoday_dc3acb0dba2a7693.json"),
     "marker": "classical Kinbaku style of the Showa period", "decision": "drop",
     "reason_code": "overgeneralized_from_personal_anecdote",
     "reason": "The answer turns one interviewee's present suspension style into a general period association."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
ISOLATED_PROJECTION = {
    "active_after_context_merit_v9": 699,
    "active_after_this_tranche": 690,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 9,
    "new_edits_applied": 4,
    "output_rows": 728,
    "output_sha256": "2793bed4b995edb05ba315259f820999982d8f59e3dc1b38ee7803733f869acf",
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
                 "schema": "context-merit-audit-v10", "source": row["source"],
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
    report = {"schema": "context-merit-audit-report-v10", "reviewer": REVIEWER,
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
