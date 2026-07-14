#!/usr/bin/env python3
"""Deterministically re-audit the next weak-context surviving keeps in v26."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V25_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v25"
sys.path[:0] = [str(ROOT), str(V25_DIR)]
import build_context_merit_audit_v25 as previous

BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v26.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v26.jsonl"
REPORT = OUT_DIR / "report_context_merit_v26.json"
REVIEWER = "codex-context-merit-audit-v26"
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
    for version in range(1, 26)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 26)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-4e683b3c8dedcefec3a3", "source_path": raw("rope365_1ee277117c1ce420.json"),
     "marker": "Shear Lashing", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_person_tie_lookup",
     "reason": "The term comes from an exercise that transfers structural lashing to moving limbs without adequate anatomy, circulation, monitoring, or release guidance."},
    {"fact_id": "fact-a8f9b828a68537fb8bd8", "source_path": raw("esinem_a94ba59553f0cc78.json"),
     "marker": "so many factors influence diameter", "decision": "edit",
     "question": "Why does Esinem call rope-diameter measurement more art than science?",
     "answer": "so many factors influence diameter",
     "reason_code": "replace_arbitrary_wrap_count_with_measurement_limit",
     "reason": "The edit replaces an arbitrary ten-wrap measurement detail with the source's durable warning that rope diameter depends on many variables."},
    {"fact_id": "fact-fcf827073a98f75fd9a5", "source_path": raw("rope365_6a23940f6bc37fc1.json"),
     "marker": "granny knot (aka tatemusubi 縦結び)", "decision": "drop",
     "reason_code": "duplicate_knot_alias_without_added_merit",
     "reason": "This alias lookup duplicates the square-versus-granny security limitation already retained from the same source and adds no safety context."},
    {"fact_id": "fact-099b416b74b5f9d328cb", "source_path": raw("rope365_62f7e527bb35b47d.json"),
     "marker": "it weighs about 60% as much", "decision": "drop",
     "reason_code": "source_personal_material_comparison_without_use_context",
     "reason": "The percentage is a personal material comparison without enough construction, use, conditioning, or measurement context to generalize."},
    {"fact_id": "fact-224434e4450cb9a39363", "source_path": raw("esinem_4cdf0843c58d3796.json"),
     "marker": "To be safe and creative, you need to understand what you are doing and why", "decision": "edit",
     "question": "Why does Esinem say suspension should not be done by rote?",
     "answer": "To be safe and creative, you need to understand what you are doing and why.",
     "reason_code": "replace_suspension_name_with_understanding_principle",
     "reason": "The edit removes a high-risk suspension-name lookup and retains the source's explicit principle that safe, creative work requires understanding rather than rote copying."},
    {"fact_id": "fact-854abc7d9f7b294521d3", "source_path": raw("rope365_d1dab5f5fa3d5cbc.json"),
     "marker": "Bending forward makes breathing difficult", "decision": "edit",
     "question": "What breathing and release warning does Rope365 give for the shrimp tie?",
     "answer": "Bending forward makes breathing difficult and you have to be prepared to untie quickly.",
     "reason_code": "replace_historical_pose_name_with_breathing_warning",
     "reason": "The edit replaces historical pose-name trivia with the source's direct warning about breathing difficulty and rapid release."},
    {"fact_id": "fact-b0325d750b296af187c6", "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
     "marker": "the magazine wanted such a readerly position to exist", "decision": "edit",
     "question": "What does the source say the uncertainty around S-ko’s reader letter reveals?",
     "answer": "the magazine wanted such a readerly position to exist",
     "reason_code": "replace_periodical_label_with_source_critique",
     "reason": "The edit replaces a bare periodical label with the article's substantive media-literacy conclusion about editorially constructed readership."},
    {"fact_id": "fact-c7f97d425dddd7aecbb5", "source_path": raw("kinbakutoday_4f26f20c5f1dc7ba.json"),
     "marker": "Roman Porno was a genre created by Nikkatsu studios", "decision": "keep",
     "reason_code": "media_history_genre_definition_context_complete",
     "reason": "The question asks for a clearly source-defined mainstream-film genre and preserves useful context for Japanese erotic-media history."},
    {"fact_id": "fact-f11b8cd79977beb93ed4", "source_path": raw("rope365_0a23b686a4f493e1.json"),
     "marker": "it requires maintained tension in order to keep its grip", "decision": "edit",
     "question": "What limitation does Rope365 give for using a half hitch as a friction?",
     "answer": "it requires maintained tension in order to keep its grip",
     "reason_code": "replace_reference_number_with_friction_limit",
     "reason": "The edit replaces an incidental Ashley Book number with the source's operational limitation that the half hitch loses grip without maintained tension."},
    {"fact_id": "fact-fb0119c5862636f2e7d7", "source_path": raw("rope365_0b4f44c8fabfc202.json"),
     "marker": "elastic ropes directly on the skin", "decision": "keep",
     "reason_code": "elastic_rope_tourniquet_risk_context_complete",
     "reason": "The question directly identifies the tourniquet risk of elastic rope on skin and preserves a concise material-safety warning."},
    {"fact_id": "fact-7062996ff78795383f5a", "source_path": raw("rope365_da67938bae2edac4.json"),
     "marker": "not only are these techniques risky, they actually put the rope in dangerous places on purpose", "decision": "edit",
     "question": "What does Rope365 warn distinguishes historical hojojutsu techniques from modern bondage?",
     "answer": "not only are these techniques risky, they actually put the rope in dangerous places on purpose",
     "reason_code": "replace_period_trivia_with_historical_technique_warning",
     "reason": "The edit replaces a golden-age date lookup with the source's explicit warning against treating historical restraint techniques as ordinary modern bondage."},
    {"fact_id": "fact-18ad3b9fe7fd13d65cec", "source_path": raw("rope365_753bb782335285c6.json"),
     "marker": "One needs a deep understanding of something to be able to explain it with words", "decision": "edit",
     "question": "Why can teaching or explaining rope deepen the teacher’s understanding?",
     "answer": "One needs a deep understanding of something to be able to explain it with words.",
     "reason_code": "replace_activity_label_with_learning_principle",
     "reason": "The edit replaces a life-drawing activity label with the source's reusable principle that explanation tests and deepens understanding."},
    {"fact_id": "fact-28d15f24f196d7ea55c6", "source_path": raw("rope365_f250f228cc370052.json"),
     "marker": "The Somerville bowline is a very stable knot",
     "manual_evidence": "The Somerville bowline is a very stable knot that will work with any types of rope. It can be tied using only the bight, which allows for speed as we don’t need to pull the whole rope through. It also leaves the bight accessible, making it possible to untie from the starting point without untying the rest, which is great in case of emergency. You can easily decide to use more or less wraps to distribute the pressure on the body.\nThe biggest con of this knot is that it is a bit more difficult to learn, take your time to visualize the different steps and you will master it in no time. If you are struggling, don’t hesitate to explore other topics and come back to it when you feel ready. Another downside is that with a lot of pressure, the knot can compact and become difficult to untie, especially when the tail is under tension. In case of emergency, you can always use your safety cutting device if you are having difficulty untying the knot.",
     "decision": "edit",
     "question": "What emergency advantage and load-related drawback does Rope365 give for the Somerville bowline?",
     "answer": "Its accessible bight can let you untie from the starting point, but heavy pressure can compact the knot and make it difficult to untie.",
     "reason_code": "replace_knot_name_lookup_with_release_tradeoff",
     "reason": "The edit replaces a name lookup with the source's useful tradeoff between an accessible emergency release path and compaction under heavy pressure."},
    {"fact_id": "fact-a2f7e13d445717ecdd37", "source_path": raw("kinbakutoday_454370c8a4f42708.json"),
     "marker": "training session (講習会)", "decision": "drop",
     "reason_code": "narrow_interview_terminology_without_added_merit",
     "reason": "Whether one historical group called early meetings a training session or dojo is narrow interview terminology and adds little durable safety, practice, or historical understanding."},
    {"fact_id": "fact-133008d0ea45d5ccf254", "source_path": raw("rope365_729efcd749abbd37.json"),
     "marker": "the forces inside the rope will make the rope shift in different ways", "decision": "edit",
     "question": "Why can crossing-hitch direction matter even when differences seem minimal at 90 degrees on a flat plane?",
     "answer": "the forces inside the rope will make the rope shift in different ways",
     "reason_code": "replace_flat_plane_fact_with_three_dimensional_effect",
     "reason": "The edit replaces a flat-plane angle observation with the source's explanation that three-dimensional loading changes how rope shifts."},
    {"fact_id": "fact-2edd73476d647026bde4", "source_path": raw("kinbakutoday_91bd5cf512bd9ecf.json"),
     "marker": "focused exclusively on tying men", "decision": "drop",
     "reason_code": "commercial_explicit_tutorial_trivia",
     "reason": "The claim is promotional trivia for an explicit, high-risk instructional video and offers no anatomy, consent, monitoring, or release value."},
    {"fact_id": "fact-37f5e669dedb1fe5da30", "source_path": raw("rope365_441f9cc87ead6159.json"),
     "marker": "cross-legged pose aka agura shibari", "decision": "drop",
     "reason_code": "context_incomplete_pose_practice_lookup",
     "reason": "The pose-name lookup comes from a practice prompt without enough fit, anatomy, loading, monitoring, or release context to stand alone."},
    {"fact_id": "fact-38de97660f344b03efb8", "source_path": raw("rope365_8012cecdbbfc3fc6.json"),
     "marker": "not to trap the fingers against the forearms", "decision": "keep",
     "reason_code": "finger_placement_safety_check_context_complete",
     "reason": "The question preserves a direct finger-placement check that reduces trapping risk in the described arm position."},
    {"fact_id": "fact-44b8628b723246a6a2a5", "source_path": raw("kinbakutoday_209cfdfa24ad7561.json"),
     "marker": "ninety percent of Yoji Muku’s kinbaku pictures", "decision": "drop",
     "reason_code": "duplicate_personal_photo_influence_trivia",
     "reason": "The personal percentage estimate duplicates the stronger source-attributed photography influence already retained from this article."},
    {"fact_id": "fact-6676997d84fcf7ca74ac", "source_path": raw("rope365_da67938bae2edac4.json"),
     "marker": "coil that leash in a chain sinnet", "decision": "drop",
     "reason_code": "unsafe_prisoner_transport_technique_lookup",
     "reason": "The term is embedded in a prisoner-transport exercise that intentionally allows acceleration before restraint and lacks adequate modern safety context."},
    {"fact_id": "fact-727116892d2372a8da55", "source_path": raw("kinbakutoday_8057255610ff3d29.json"),
     "marker": "Hojōjutsu-type shibari", "decision": "drop",
     "reason_code": "low_value_image_classification",
     "reason": "A one-word classification of an isolated image is weakly contextualized and adds no technique, safety, consent, or developed historical explanation."},
    {"fact_id": "fact-d5a388ed0511d14804de", "source_path": raw("kinbakutoday_edba1220873364c8.json"),
     "marker": "the focus on tying shifted from military application to law enforcement", "decision": "edit",
     "question": "What change in hojojutsu does the source associate with the Tokugawa Shogunate?",
     "answer": "the focus on tying shifted from military application to law enforcement",
     "reason_code": "replace_regime_lookup_with_historical_change",
     "reason": "The edit replaces a regime-name lookup with the source's actual historical claim about the shift from military restraint to law enforcement."},
    {"fact_id": "fact-eb8c105adce5e1d530fc", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "Favour simple words and simple sentence structures", "decision": "edit",
     "question": "How does Rope365 recommend phrasing communication when someone may be nervous, distressed, or under intense stimuli?",
     "answer": "Favour simple words and simple sentence structures.",
     "reason_code": "replace_triple_negative_lookup_with_clear_instruction",
     "reason": "The edit replaces an awkward triple-negative lookup with the source's positive, broadly useful communication instruction."},
    {"fact_id": "fact-5f1c7e21663cfc5c41d5", "source_path": raw("kinbakutoday_381214111022a952.json"),
     "marker": "samurai movie dramas", "decision": "drop",
     "reason_code": "incidental_personal_media_trivia",
     "reason": "A named participant's favorite film genre is incidental personal trivia and adds no durable safety, consent, practice, or historical insight."},
    {"fact_id": "fact-2c85b2bf452ff523496b", "source_path": raw("rope365_0b4f44c8fabfc202.json"),
     "marker": "Holding friction – The capacity of rope to hold in place", "decision": "keep",
     "reason_code": "rope_property_definition_context_complete",
     "reason": "The question directly defines a useful material property that affects knot and friction choices."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        519, 94, 230, 314, 257, 46, 390, 213, 227, 371,
        44, 426, 365, 387, 121, 262, 249, 132, 7, 368,
        433, 43, 347, 430, 269,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (8 if index == 0 else 9 if index < 13 else 10)
    for index, fact_id in enumerate(EXPECTED_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v25",
    "eligible_prior_keeps_before_reaudit_exclusion": 255,
    "secondary_eligible_prior_keeps": 130,
    "rows": 614,
    "sha256": "fc704352e4bd193c10b81117c54459bb96cb111274536d6d58e6ab3fe4220d8a",
    "v21_v25_reviewed_fact_ids_excluded": 125,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v25": 576,
    "active_after_this_tranche": 566,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 10,
    "new_edits_applied": 11,
    "output_rows": 604,
    "output_sha256": "2febd23fb80cdd241b9f8244a9bdc840b4107a2a76aa3c800f8dfd66a9a71e75",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 4,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 26):
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
    if len(ranked) != 130:
        raise ValueError(f"v26 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:25]) != EXPECTED_SELECTION:
        raise ValueError("v26 secondary selection drift")
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
        row["schema"] = "context-merit-audit-v26"
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
    paraphrase_fact_id = "fact-28d15f24f196d7ea55c6"
    for row in curations:
        if row["fact_id"] == paraphrase_fact_id:
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = (
                "Combines two adjacent source paragraphs into one tradeoff: "
                "the accessible bight supports release from the start, while "
                "heavy pressure can compact the knot and impede untying."
            )
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v26"
    report["selection"].update({
        "active_rows": 614,
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 130,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v25 projection, and the fact was not reviewed "
                "again in v21 through v25"
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
        "extractive": 10, "manual_paraphrase": 1,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
