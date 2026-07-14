#!/usr/bin/env python3
"""Deterministically re-audit the next weak-context surviving keeps in v27."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V26_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v26"
sys.path[:0] = [str(ROOT), str(V26_DIR)]
import build_context_merit_audit_v26 as previous

BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v27.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v27.jsonl"
REPORT = OUT_DIR / "report_context_merit_v27.json"
REVIEWER = "codex-context-merit-audit-v27"
REVIEWED_AT = "2026-07-14"
RESOURCE_MANIFEST = ROOT / "sources" / "rope_resources_v1.json"
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
CONTEXT_CURATIONS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"pending_curation_context_merit_v{version}.jsonl"
    for version in range(1, 27)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 27)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


def resource(name: str) -> Path:
    return DATA / "raw" / "rope_resources_v1" / name


SPECS = (
    {"fact_id": "fact-f0fef2ee956cc106d74a", "source_path": resource("rope365__22f7d5d8e78a36e5cbaa.json"),
     "marker": "Make sure to discuss these poses before exploring them as they will make people feel quite vulnerable", "decision": "edit",
     "question": "Why does Rope365 say exposed poses should be discussed before exploring them?",
     "answer": "Make sure to discuss these poses before exploring them as they will make people feel quite vulnerable.",
     "reason_code": "replace_pose_alias_with_vulnerability_discussion",
     "reason": "The edit replaces an exposed-position alias with the source's explicit instruction to discuss vulnerable poses before attempting them."},
    {"fact_id": "fact-feadf2d476a11a4a1140", "source_path": resource("rope365__077aecb334166abba18c.json"),
     "marker": "Nylon/MFP are very slippery which allows tying complex knots faster than a high friction rope", "decision": "keep",
     "reason_code": "source_attributed_material_characteristic",
     "reason": "The question preserves Rope365's bounded comparison of nylon and MFP friction rather than making a universal material-safety claim."},
    {"fact_id": "fact-2002c92260ac3a3d80a0", "source_path": resource("rope365__f9753746348ca9cc2f11.json"),
     "marker": "Rope is a rather recent practice in history and we are constantly learning to make it safer", "decision": "edit",
     "question": "Why does Rope365 recommend critical thinking when using rope resources?",
     "answer": "Rope is a rather recent practice in history and we are constantly learning to make it safer.",
     "reason_code": "replace_best_format_claim_with_critical_thinking_reason",
     "reason": "The edit replaces an absolute learning-format preference with the page's reason for critically evaluating evolving safety information."},
    {"fact_id": "fact-a81264b699fcb00f8a2e", "source_path": resource("rope365__ac5f45d316f177fc0d38.json"),
     "marker": "these ties are generally less restrictive and more sustainable", "decision": "edit",
     "question": "Why does Rope365 begin its body-position study with hands in front?",
     "answer": "these ties are generally less restrictive and more sustainable",
     "reason_code": "replace_pose_alias_with_accessibility_rationale",
     "reason": "The edit removes another pose alias and retains the source's practical reason for beginning with hands-in-front positions."},
    {"fact_id": "fact-2483854a1ce91c077032", "source_path": resource("rope365__5d7a851080ae72c9620d.json"),
     "marker": "It will be stable as long as tension is maintained in both the origin and exit rope", "decision": "edit",
     "question": "Under what condition does Rope365 say a half-hitch inline cuff remains stable?",
     "answer": "It will be stable as long as tension is maintained in both the origin and exit rope.",
     "reason_code": "replace_pattern_lookup_with_half_hitch_limit",
     "reason": "The edit replaces a pattern-name lookup with the source's explicit limitation that the half-hitch cuff depends on maintained tension at both ends."},
    {"fact_id": "fact-a236300d1e531bd987b5", "source_path": resource("rope365__2db0b0e55d35bf0aa676.json"),
     "marker": "The overhand hank coiling method is easy, fast and doesn’t add unnecessary twists and stress on the rope for storage", "decision": "keep",
     "reason_code": "rope_storage_method_context_complete",
     "reason": "The method and its storage benefit are fully specified, practical, and independent of high-risk body-tying instruction."},
    {"fact_id": "fact-e2fe113f6dcb3c25fa4f", "source_path": raw("kinbakutoday_86c13124375e690a.json"),
     "marker": "Perhaps the most famous of these authors was Oniroku Dan", "decision": "drop",
     "reason_code": "promotional_call_person_trivia",
     "reason": "The famous-author lookup is a brief lead-in to a dated fiction-submission call and duplicates richer literary-history coverage elsewhere."},
    {"fact_id": "fact-79ea40e1492359a99725", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "person’s weight adds to the pressure of the rope on the body", "decision": "keep",
     "reason_code": "suspension_nerve_risk_context_complete",
     "reason": "The question clearly connects added bodyweight pressure in partial and full suspension with the source's nerve-injury warning."},
    {"fact_id": "fact-e03d73de7713c52bfcde", "source_path": raw("kinbakutoday_1f811a9abc9520b4.json"),
     "marker": "Always think of the communication first and foremost", "decision": "edit",
     "question": "What does Nagare Aotsuki say beginners should prioritize beyond tying technique?",
     "answer": "Always think of the communication first and foremost.",
     "reason_code": "replace_person_influence_with_communication_priority",
     "reason": "The edit replaces a partial lineage lookup with the interviewee's direct and broadly useful advice to prioritize communication."},
    {"fact_id": "fact-61369f572c05bbc6974c", "source_path": raw("kinbakutoday_5110ea5e4aec6400.json"),
     "marker": "create your own original style after mastering one of these styles and understanding the other tying styles", "decision": "edit",
     "question": "When does Kazami say learners can create an original style?",
     "answer": "after mastering one of these styles and understanding the other tying styles",
     "reason_code": "replace_organization_roles_with_learning_progression",
     "reason": "The edit replaces organization-role trivia with the interviewee's substantive learning progression from mastery and comparison to original work."},
    {"fact_id": "fact-7ebce2d94e6dd71d7414", "source_path": raw("kinbakutoday_7fb6f1e0e0d186b2.json"),
     "marker": "makaseru, which means to entrust something to someone else", "decision": "drop",
     "reason_code": "consent_conflation_context_risk",
     "reason": "Although the translation is correct, the surrounding essay repeatedly frames trust as replacing or superseding consent, making the isolated term unsafe and misleading in this dataset."},
    {"fact_id": "fact-19ef1ddd27dbd7f4202f", "source_path": raw("rope365_62f7e527bb35b47d.json"),
     "marker": "cotton/coconut/bamboo should not be oiled or waxed", "decision": "drop",
     "reason_code": "unfinished_personal_material_care_claim",
     "reason": "The source explicitly calls its maintenance section unfinished and personal, without enough construction- or product-specific support for categorical care advice."},
    {"fact_id": "fact-f039341f5dca43990ad6", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["tetruss"], "evidence": "Tetruss: https://tetruss.com/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that the Tetruss portable-frame URL remain available for resource recommendations."},
    {"fact_id": "fact-517f5ab1f5e32057e9b0", "source_path": raw("rope365_3d015bb41296c9fa.json"),
     "marker": "cinches, they are flat with no bulk between the arms and the torso", "decision": "keep",
     "reason_code": "box_tie_self_evaluation_context_complete",
     "reason": "The conditional checklist item is a concrete fit check and does not claim that cinches are universally required."},
    {"fact_id": "fact-c44bed0a3db7c0c73dbc", "source_path": raw("kinbakutoday_1775dd4176b24104.json"),
     "marker": "Competence is important for keeping your partner safe. Knowledge is important, even essential, for teaching others how to become competent", "decision": "edit",
     "question": "How does the author distinguish competence from knowledge in rope teaching?",
     "answer": "Competence is important for keeping your partner safe. Knowledge is important, even essential, for teaching others how to become competent.",
     "reason_code": "replace_self_blame_with_teaching_distinction",
     "reason": "The edit replaces personal self-blame with the essay's developed distinction between performing safely and understanding enough to teach others."},
    {"fact_id": "fact-0a9bde892f63c70189ff", "source_path": raw("esinem_77368ccdc66acad0.json"),
     "marker": "Bondage & Discipline", "manual_evidence": "BD: Bondage & Discipline\nDs: Domination & submission\nSM: Sadism & Masochism", "decision": "keep",
     "reason_code": "bdsm_term_definition_context_complete",
     "reason": "The three expansions correctly unpack the paired meanings represented by the BDSM acronym."},
    {"fact_id": "fact-568e19d1c85f56b99ce1", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "solid stick like a marlinspike", "decision": "keep",
     "reason_code": "emergency_kit_context_complete",
     "reason": "The tool is narrowly recommended for loosening hard knots alongside safer cutting tools and incident planning."},
    {"fact_id": "fact-dc125508d925fcfc272b", "source_path": raw("rope365_682937f92222bf87.json"),
     "marker": "Violence, unsafe practices, and lack of consent", "decision": "keep",
     "reason_code": "bondage_history_caution_context_complete",
     "reason": "The answer concisely preserves the source's warning against romanticizing bondage history."},
    {"fact_id": "fact-4471cc69e2bb67b1059c", "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
     "marker": "how to distinguish seme from mere violence, vulgar appetite, or crude spectacle", "decision": "keep",
     "reason_code": "postwar_media_ethics_context_complete",
     "reason": "The question preserves the article's substantive media-history problem rather than an isolated person, date, or title lookup."},
    {"fact_id": "fact-35a0f1a70111fab6fbff", "source_path": raw("kinbakutoday_c5e568667b495473.json"),
     "marker": "ashi o kuzushite ii yo", "decision": "keep",
     "reason_code": "partner_comfort_phrase_context_complete",
     "reason": "The question explains the phrase's function as permission to leave an uncomfortable formal kneeling position."},
    {"fact_id": "fact-222cfb0d001c49fdaf22", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["crash_restraint"], "evidence": "Crash Restraint: https://crash-restraint.com/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that Crash Restraint remain available as the introductory suspension resource."},
    {"fact_id": "fact-9aad2d24020dfc169e35", "source_path": raw("wikipedia_914d249c3d7d542c.json"),
     "marker": "father of modern kinbaku", "decision": "drop",
     "reason_code": "thin_honorific_history_lookup",
     "reason": "The honorific is a thin encyclopedia label, while the dataset already retains more concrete and better sourced accounts of Itō Seiu's work and influence."},
    {"fact_id": "fact-e14d15f8a97e0a51371e", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["subspace_designs"], "evidence": "Subspace Designs: https://www.subspacedesigns.shop/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that the Subspace Designs rigging-plate URL remain available for resource recommendations."},
    {"fact_id": "fact-47a176a9c9f1245c596f", "source_path": raw("rope365_773cf4d4be0e2895.json"),
     "marker": "do not press into the armpit or inside of the arms", "decision": "keep",
     "reason_code": "cinch_placement_safety_context_complete",
     "reason": "The conditional checklist item gives a concrete pressure-placement precaution for the described structure."},
    {"fact_id": "fact-ca3a4fcac4583ae3a5ca", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["my_nawashi", "de_giotto_rope"],
     "evidence": "My Nawashi: https://www.etsy.com/shop/MyNawashi; De Giotto Rope: https://degiottorope.com/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory",
     "reason": "Both natural-fiber rope suppliers were explicitly supplied by the project owner for durable resource lookup."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        230, 510, 222, 537, 308, 490, 562, 415, 506, 554,
        195, 511, 457, 26, 412, 121, 368, 28, 322, 313,
        446, 394, 458, 126, 530,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (11 if index < 6 else 12 if index < 14 else 13)
    for index, fact_id in enumerate(EXPECTED_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v26",
    "eligible_prior_keeps_before_reaudit_exclusion": 255,
    "secondary_eligible_prior_keeps": 105,
    "rows": 604,
    "sha256": "2febd23fb80cdd241b9f8244a9bdc840b4107a2a76aa3c800f8dfd66a9a71e75",
    "v21_v26_reviewed_fact_ids_excluded": 150,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v26": 566,
    "active_after_this_tranche": 562,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 4,
    "new_edits_applied": 7,
    "output_rows": 600,
    "output_sha256": "a1a2574d502aeeee4fe792373ff7858a65405479d85b2ae5827f710868b6b54a",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 14,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 27):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        paths.extend((
            directory / f"context_merit_audit_v{version}.jsonl",
            directory / f"pending_curation_context_merit_v{version}.jsonl",
            directory / f"report_context_merit_v{version}.json",
        ))
    return tuple(paths)


def secondary_ranked(rows: list[dict]) -> list[dict]:
    by_id = {row["fact_id"]: (index, row)
             for index, row in enumerate(rows, 1)}
    occurrences: dict[str, int] = {}
    prior_keeps: dict[str, int] = {}
    for version, path in enumerate(CONTEXT_AUDITS[:20], 1):
        for audit in read_jsonl(path):
            fact_id = audit["fact_id"]
            occurrences[fact_id] = occurrences.get(fact_id, 0) + 1
            if audit["decision"] == "keep":
                prior_keeps[fact_id] = version
    rereviewed = {
        row["fact_id"] for path in CONTEXT_AUDITS[20:]
        for row in read_jsonl(path)
    }
    candidates = []
    for fact_id, version in prior_keeps.items():
        if (occurrences[fact_id] != 1 or fact_id not in by_id or
                fact_id in rereviewed):
            continue
        index, row = by_id[fact_id]
        features = CORE.risk_features(row)
        candidates.append((
            -features["risk_score"], features["question_tokens"],
            features["answer_tokens"], fact_id, index, row, features, version,
        ))
    candidates.sort(key=lambda item: item[:4])
    ranked = [{
        "active_index": item[4], "row": item[5], "features": item[6],
        "prior_version": item[7],
    } for item in candidates]
    if len(ranked) != 105:
        raise ValueError(f"v27 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:25]) != EXPECTED_SELECTION:
        raise ValueError("v27 secondary selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return secondary_ranked(rows), 0, 0


@contextlib.contextmanager
def patched_base():
    replacements = {
        "OUT_DIR": OUT_DIR, "AUDIT": AUDIT, "CURATION": CURATION,
        "REPORT": REPORT, "REVIEWER": REVIEWER, "REVIEWED_AT": REVIEWED_AT,
        "CONTEXT_CURATIONS": CONTEXT_CURATIONS, "SPECS": SPECS,
        "EXPECTED_SELECTION": EXPECTED_SELECTION,
        "ISOLATED_PROJECTION": ISOLATED_PROJECTION,
    }
    originals = {name: getattr(BASE, name) for name in replacements}
    original_ranking = CORE.ranked_unreviewed
    try:
        for name, value in replacements.items():
            setattr(BASE, name, value)
        CORE.ranked_unreviewed = selected_ranked
        yield
    finally:
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def ranked_unreviewed(rows: list[dict]) -> tuple[list[dict], int, int]:
    with patched_base():
        return BASE.ranked_unreviewed(rows)


def main() -> None:
    with patched_base():
        BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        fact_id = row["fact_id"]
        version = SECONDARY_PRIOR_VERSIONS[fact_id]
        prior_path = (DATA / "manual_reviews" /
                      f"context_merit_audit_v{version}" /
                      f"context_merit_audit_v{version}.jsonl")
        row["schema"] = "context-merit-audit-v27"
        row["active_index"] = PROJECTED_ACTIVE_INDICES[fact_id]
        row["review_pass"] = "secondary_prior_keep_reaudit"
        row["prior_review"] = {
            "decision": "keep",
            "path": str(prior_path.relative_to(ROOT)),
            "sha256": file_sha256(prior_path),
            "version": version,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v27"
    report["selection"].update({
        "active_rows": 604,
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 105,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v26 projection, and the fact was not reviewed "
                "again in v21 through v26"
            ),
            "score": (
                "short_question_points + 3*pronoun_count + "
                "bare_answer_points + named_person_trivia_points"
            ),
            "tie_break": (
                "risk_score descending, question tokens ascending, answer "
                "tokens ascending, fact_id ascending"
            ),
        },
    })
    report["audit"]["sha256"] = file_sha256(AUDIT)
    report["new_pending_curation"]["sha256"] = file_sha256(CURATION)
    report["new_pending_curation"]["edit_support_types"] = {
        "extractive": 7, "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
