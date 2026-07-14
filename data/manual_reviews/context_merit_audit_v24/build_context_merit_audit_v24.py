#!/usr/bin/env python3
"""Deterministically re-audit the next weak-context surviving keeps in v24."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V23_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v23"
sys.path[:0] = [str(ROOT), str(V23_DIR)]
import build_context_merit_audit_v23 as previous

BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v24.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v24.jsonl"
REPORT = OUT_DIR / "report_context_merit_v24.json"
REVIEWER = "codex-context-merit-audit-v24"
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
    for version in range(1, 24)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 24)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-d219f277448544afa318", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["my_nawashi"],
     "evidence": "My Nawashi: https://www.etsy.com/shop/MyNawashi",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that this natural-fiber rope-shop URL remain available for recommendations."},
    {"fact_id": "fact-38c6234e2a72dd14287f", "source_path": raw("esinem_b249eddc1f5e1864.json"),
     "marker": "High-stranding is the term used to describe one of the 3 strands going out of balance", "decision": "keep",
     "reason_code": "rope_condition_term_context_complete",
     "reason": "The question and evidence define a useful rope-inspection term without turning the repair discussion into unsupported safety assurance."},
    {"fact_id": "fact-5e7ec77ea97c66a99295", "source_path": raw("kinbakutoday_3ee6ab45e0f9b1ae.json"),
     "marker": "Technique was always second to heart", "decision": "edit",
     "question": "How does the author summarize the priority his Tokyo teacher set between technique and heart?",
     "answer": "Technique was always second to heart.",
     "reason_code": "replace_inverted_lookup_with_attributed_priority",
     "reason": "The edit turns an awkward inverted lookup into a clearly attributed summary of the author's lesson about motivation and relational intent."},
    {"fact_id": "fact-e5950fbfe9d881d40bb3", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["shibari_study"],
     "evidence": "Shibari Study: https://shibaristudy.com/",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that this intermediate-suspension course catalog remain available for recommendations."},
    {"fact_id": "fact-396293a77723206abb84", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["knothead_nylon"],
     "evidence": "Knot Head Nylon: https://knotheadnylon.net/",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that this synthetic-rope vendor URL remain available for recommendations."},
    {"fact_id": "fact-5c02852cca5c4979d32e", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["strugglers_knot_somerville_bowline"],
     "evidence": "Struggler's Knot and Somerville Bowline: https://www.youtube.com/watch?v=OsIcEtCoKHo",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that this lesson URL remain available for rope-resource recommendations."},
    {"fact_id": "fact-12234c1f51530e1e0526", "source_path": raw("esinem_5d1d44089d8bedf1.json"),
     "marker": "signature L-friction", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_person_tie_lookup",
     "reason": "A signature-friction name from a promotional tutorial post omits anatomy, fit, loading, monitoring, and emergency-release context for the high-risk ties it advertises."},
    {"fact_id": "fact-1e25e0594b6e1382ab22", "source_path": raw("rope365_62f7e527bb35b47d.json"),
     "marker": "short”” ropes (10-15ft)", "decision": "drop",
     "reason_code": "source_personal_recommendation_without_use_context",
     "reason": "The bare length range is one author's starting-kit preference and gives no intended tie, body, handling, loading, or safety context."},
    {"fact_id": "fact-24adb04721d158df31dc", "source_path": raw("kinbakutoday_5a6e15f57e52a345.json"),
     "marker": "For Ito, Seme (torture) was the equivalent term for sadomasochism", "decision": "edit",
     "question": "According to Ugo, how did Ito Seiu use the term seme around 1900?",
     "answer": "the equivalent term for sadomasochism",
     "reason_code": "add_speaker_and_historical_qualification",
     "reason": "The edit preserves Ugo's historical terminology claim while making its speaker, period, and non-universal scope explicit."},
    {"fact_id": "fact-366cdcf2556613c8f486", "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "trance-like mental state is also called subspace", "decision": "keep",
     "reason_code": "bdsm_state_term_context_complete",
     "reason": "The question limits the answer to the commonly used name for the submissive's described trance-like state."},
    {"fact_id": "fact-498f0c0b7d716b63aca4", "source_path": raw("kinbakutoday_dc3acb0dba2a7693.json"),
     "marker": "Semenawa (rope punishment/torture)", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "The bare translation duplicates stronger retained seme terminology while coming from an interview that also advances unsupported therapeutic claims."},
    {"fact_id": "fact-4d843edd13c9109507f6", "source_path": raw("kinbakutoday_df9c70212a927199.json"),
     "marker": "hazukashii centers on self-awareness, vulnerability, and relational intimacy", "decision": "edit",
     "question": "How does the source distinguish hazukashii from harsher forms of humiliation?",
     "answer": "hazukashii centers on self-awareness, vulnerability, and relational intimacy",
     "reason_code": "replace_dictionary_gloss_with_source_distinction",
     "reason": "The edit replaces a shallow translation lookup with the article's central, source-qualified emotional distinction."},
    {"fact_id": "fact-db4e6b687bd1fba0dde5", "source_path": raw("rope365_25f1b23eb40be00e.json"),
     "marker": "family of knots used to extend rope is called bends", "decision": "keep",
     "reason_code": "foundational_knot_family_context_complete",
     "reason": "The answer is a reusable, low-risk vocabulary fact identifying the knot family used to join or extend rope."},
    {"fact_id": "fact-dc424e61c6c9c3551d03", "source_path": raw("kinbakutoday_43095aa305e35a41.json"),
     "marker": "anyone being tied should also have the courage to stop when they feel something is wrong", "decision": "edit",
     "question": "What does Tessin Doyama urge a person being tied to do when something feels wrong?",
     "answer": "have the courage to stop",
     "reason_code": "replace_role_label_with_partner_safety_action",
     "reason": "The edit replaces a bare role-term lookup with the safety article's explicit, actionable stop guidance for the person being tied."},
    {"fact_id": "fact-3d3373b7b595e1b6324b", "source_path": raw("kinbakutoday_34dec041941868d9.json"),
     "marker": "in the posturing called “Omoi-ire”", "decision": "keep",
     "reason_code": "kabuki_expression_term_context_complete",
     "reason": "The question identifies the Kabuki context and directly defines the nonverbal expression term used by the source."},
    {"fact_id": "fact-904e5f318a018e7878af", "source_path": raw("rope365_62f7e527bb35b47d.json"),
     "marker": "averaging ~60% of what a comparable nylon kit weighs", "decision": "drop",
     "reason_code": "source_personal_material_comparison",
     "reason": "The percentage is an uncited personal comparison from a broad primer and is not stable across rope diameter, construction, maker, or kit composition."},
    {"fact_id": "fact-a667748ec149351ab907", "source_path": raw("rope365_0b4f44c8fabfc202.json"),
     "marker": "Hempex floats in water and doesn’t shrink when wet", "decision": "keep",
     "reason_code": "rope_material_water_property_context_complete",
     "reason": "The narrowly worded question preserves two directly stated material properties without generalizing them into a safety recommendation."},
    {"fact_id": "fact-b784b4278644a2f88656", "source_path": raw("rope365_6f46d5169ca32ec7.json"),
     "marker": "co-founder of Bakuyukai, the first Japanese rope dojo", "decision": "keep",
     "reason_code": "source_qualified_rope_history_context_complete",
     "reason": "The question supplies the co-founder and institution type, making the historical name meaningful rather than isolated person trivia."},
    {"fact_id": "fact-da08e38f2a3bd86a6c62", "source_path": raw("rope365_d7f0891ee630f057.json"),
     "marker": "futomomo shibari 太腿縛り", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_person_tie_lookup",
     "reason": "The character-string lookup is attached to a bent-limb tie family but contributes no mobility, joint, nerve, circulation, monitoring, or release guidance."},
    {"fact_id": "fact-733d870abe140a50c2e0", "source_path": raw("kinbakutoday_aec9a390fd230290.json"),
     "marker": "unique offshoot, called the ‘Naka-Ryu’ style", "decision": "drop",
     "reason_code": "volatile_or_promotional_person_style_trivia",
     "reason": "The style-name lookup comes from a dated workshop advertisement and adds little durable knowledge beyond promotional biography."},
    {"fact_id": "fact-83f5e74aec39cbbced3a", "source_path": raw("kinbakutoday_938724eb415ca5c0.json"),
     "marker": "confirms our view that Edo torture rope is unrelated to hojojutsu", "decision": "edit",
     "question": "What does the servant holding rope in the 1930 illustration suggest to the post's author?",
     "answer": "Edo torture rope is unrelated to hojojutsu",
     "reason_code": "replace_image_material_trivia_with_attributed_inference",
     "reason": "The edit replaces incidental material recall with the article's explicitly attributed interpretation of the servant's lack of formal hojojutsu access."},
    {"fact_id": "fact-cf250d3c67a7283327d4", "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "are called switches", "decision": "keep",
     "reason_code": "bdsm_role_term_context_complete",
     "reason": "The question defines the role-changing behavior before asking for the commonly used BDSM term."},
    {"fact_id": "fact-5ba72485e95be3a6972f", "source_path": raw("kinbakutoday_34dec041941868d9.json"),
     "marker": "the soul is bound to Earth by unfulfilled desires", "decision": "edit",
     "question": "What deeper Shinto/Buddhist idea does the source say can underlie urami?",
     "answer": "the soul is bound to Earth by unfulfilled desires",
     "reason_code": "replace_one_word_gloss_with_cultural_context",
     "reason": "The edit replaces a one-word grudge gloss with the source's deeper, explicitly attributed cultural account of the term."},
    {"fact_id": "fact-6aff3284b50d59e50f50", "source_path": raw("kinbakutoday_b776fccd348e2538.json"),
     "marker": "first of eight videos in his landmark “World of Rope” series", "decision": "edit",
     "question": "What was significant about the World of Rope video CineMagic released in 1989?",
     "answer": "Nureki Chimuo’s first of eight videos in his landmark “World of Rope” series",
     "reason_code": "replace_bare_year_with_series_significance",
     "reason": "The edit keeps the date as context while asking what made the release significant within Nureki's video series."},
    {"fact_id": "fact-abe07db12e2f94fbbb7e", "source_path": raw("kinbakutoday_2ded51b08225bd4b.json"),
     "marker": "It gives a strong feeling, one which encompasses simplicity without being simple",
     "manual_evidence": "It is an example, for me, of what might be considered shibusa (渋さ). It gives a strong feeling, one which encompasses simplicity without being simple. It is modest and natural, not showy or ostentatious. It has an element of imperfection.",
     "decision": "edit",
     "question": "Which qualities does the author associate with shibusa?",
     "answer": "simplicity without being simple, modest naturalness rather than showiness, and an element of imperfection",
     "reason_code": "replace_risky_tie_reference_with_aesthetic_definition",
     "reason": "The edit removes the high-hands tie from the prompt and gives a fuller, source-grounded account of the aesthetic concept itself."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        478, 395, 153, 467, 477, 484, 254, 261, 397, 282,
        236, 290, 247, 32, 291, 320, 551, 450, 240, 257,
        305, 406, 238, 98, 289,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (5 if index < 13 else 6)
    for index, fact_id in enumerate(EXPECTED_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v23",
    "eligible_prior_keeps_before_reaudit_exclusion": 255,
    "secondary_eligible_prior_keeps": 180,
    "rows": 625,
    "sha256": "efe176ecfd680754b4e87852bb8f22d9f99d52d2158f2aafd11ec081e7d2d45a",
    "v21_v23_reviewed_fact_ids_excluded": 75,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v23": 587,
    "active_after_this_tranche": 581,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 6,
    "new_edits_applied": 8,
    "output_rows": 619,
    "output_sha256": "267fa20ea6c41521147d83e1086a19a48d04478ddc52f9b442cb1c8cd231c72d",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 11,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 24):
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
    if len(ranked) != 180:
        raise ValueError(f"v24 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:25]) != EXPECTED_SELECTION:
        raise ValueError("v24 secondary selection drift")
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
        row["schema"] = "context-merit-audit-v24"
        row["active_index"] = PROJECTED_ACTIVE_INDICES[fact_id]
        row["review_pass"] = "secondary_prior_keep_reaudit"
        row["prior_review"] = {
            "decision": "keep",
            "path": str(prior_path.relative_to(ROOT)),
            "sha256": file_sha256(prior_path),
            "version": version,
        }
    write_jsonl(AUDIT, audits)

    curations = read_jsonl(CURATION)
    paraphrase_fact_id = "fact-abe07db12e2f94fbbb7e"
    for row in curations:
        if row["fact_id"] == paraphrase_fact_id:
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = (
                "Combines three adjacent source sentences into one concise "
                "definition while preserving simplicity, natural modesty, "
                "and imperfection as the stated qualities."
            )
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v24"
    report["selection"].update({
        "active_rows": 625,
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 180,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v23 projection, and the fact was not reviewed "
                "again in v21, v22, or v23"
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
        "extractive": 7, "manual_paraphrase": 1,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
