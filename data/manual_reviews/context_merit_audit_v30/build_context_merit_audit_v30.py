#!/usr/bin/env python3
"""Deterministically re-audit the next weak-context surviving keeps in v30."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V29_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v29"
sys.path[:0] = [str(ROOT), str(V29_DIR)]
import build_context_merit_audit_v29 as previous

BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v30.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v30.jsonl"
REPORT = OUT_DIR / "report_context_merit_v30.json"
REVIEWER = "codex-context-merit-audit-v30"
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
    for version in range(1, 30)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 30)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-99d3f29be1c8ebe8a59b",
     "source_path": raw("rope365_30eaedb9cb1e6d5b.json"),
     "marker": "arm span of the person tying as a measurement guideline",
     "decision": "edit",
     "question": "What body-based guideline does Rope365 describe for choosing a preferred rope length?",
     "answer": "the arm span of the person tying",
     "reason_code": "clarify_preferred_rope_length_guideline",
     "reason": "The edit replaces the awkward 'favorite length' wording while preserving arm span as one common personal guideline, not a universal requirement."},
    {"fact_id": "fact-2a7d7893e51e8bb68a2c",
     "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "The purpose of this kind of agreement is primarily",
     "manual_evidence": "The purpose of this kind of agreement is primarily to encourage discussion and negotiation in advance and then to document that understanding for the benefit of all parties. Such documents have not been recognized as being legally binding, nor are they intended to be.",
     "decision": "edit",
     "question": "What purpose do written BDSM agreements serve, and are they legally binding?",
     "answer": "The purpose of this kind of agreement is primarily to encourage discussion and negotiation in advance and then to document that understanding for the benefit of all parties. Such documents have not been recognized as being legally binding, nor are they intended to be.",
     "reason_code": "add_nonbinding_caveat_to_agreement_purpose",
     "reason": "The edit preserves the negotiation purpose while adding the source's essential caveat that these documents are neither intended nor recognized as legally binding."},
    {"fact_id": "fact-35aa8d5c46826a467bd3",
     "source_path": raw("esinem_4a64b6d0e45a7c34.json"),
     "marker": "Applying a good technique in handling the ropes prevent",
     "manual_evidence": "Sabaku 捌\nDexterity in avoiding friction movement. Flying rope. “Meters of rope rub against the skin”. Applying a good technique in handling the ropes prevent the skin of the person tied experiencing unnecessary friction and fatigue, unanticipated pinching or getting hit with the knots on the ends of the rope. Every good rigger ensures that none of these things happen without having planned or agreed to previously. (Source: Notes on Osada-ryu)",
     "decision": "edit",
     "question": "What injuries or discomfort does the source say good rope handling should prevent?",
     "answer": "Good rope handling should prevent unnecessary skin friction and fatigue, unanticipated pinching, and accidental impacts from knots at the rope ends.",
     "reason_code": "replace_term_gloss_with_handling_precautions",
     "reason": "The edit replaces an awkward term gloss with the source's concrete rope-handling precautions.",
     "paraphrase_rationale": "Repairs the source's grammar while preserving its complete list of unwanted skin friction, fatigue, pinching, and end-knot impacts."},
    {"fact_id": "fact-b2de0d8786d7488a796b",
     "source_path": raw("rope365_f43c9fde09431a5f.json"),
     "marker": "alternating going over and under when lines cross perpendicularly",
     "decision": "edit",
     "question": "How does Rope365 define weaving when lines cross perpendicularly?",
     "answer": "alternating going over and under",
     "reason_code": "clarify_weaving_definition_question",
     "reason": "The edit replaces a generated-sounding action lookup with a direct question about the foundational over-under definition."},
    {"fact_id": "fact-5b492ea062f8b5f976cb",
     "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "For each situation, visualize the way out",
     "decision": "edit",
     "question": "What does Rope365 recommend planning before a difficult situation, and what should take priority if something goes wrong?",
     "answer": "For each situation, visualize the way out in case something goes wrong. Remember: Save the relationship, not the scene.",
     "reason_code": "replace_slogan_lookup_with_exit_planning",
     "reason": "The edit gives the safety-planning action that precedes the source's memorable relationship-first priority."},
    {"fact_id": "fact-6ba5f96154209f20abae",
     "source_path": raw("kinbakutoday_fd0d7ba4b6589765.json"),
     "marker": "Aibunawa (愛撫縄）or Aibu no Nawa",
     "decision": "drop",
     "reason_code": "duplicate_style_alias_from_promotional_gallery",
     "reason": "The style alias comes from an adult-video gallery, while a more substantive explanation of why Yukimura developed aibunawa already survives from the same source."},
    {"fact_id": "fact-a09f4a17066c23e7d4e8",
     "source_path": raw("rope365_9a5c5810310fa0f0.json"),
     "marker": "avoid rope on the front of the neck and keep rope loose near joints",
     "decision": "edit",
     "question": "What two placement precautions does Rope365 give for one-rope improvisation?",
     "answer": "Take your time, avoid rope on the front of the neck and keep rope loose near joints.",
     "reason_code": "complete_neck_and_joint_placement_precautions",
     "reason": "The edit restores the source's paired precautions for both the front of the neck and areas near joints."},
    {"fact_id": "fact-d4877ffdbfde2bfbd238",
     "source_path": raw("kinbakutoday_fe60466ee5e6689e.json"),
     "marker": "above definition is oversimplification",
     "manual_evidence": "Wabi-sabi, like many other Japanese concepts, is very elusive to describe. Even for somebody who study it for many years. It is related to aesthetics, tea ceremony, and Zen philosophy. It can be generally characterized by simplicity, rusticity, imperfection and transience of life. But above definition is oversimplification. That concept is hard to comprehend by Westerners because we lack context to the concept. Rather then explanation, we need exposure.",
     "decision": "edit",
     "question": "What caveat does the article give about defining wabi-sabi with four general characteristics?",
     "answer": "The article says that definition is an oversimplification because wabi-sabi depends on cultural context and is better approached through exposure than a short explanation.",
     "reason_code": "replace_trait_list_with_cultural_context_caveat",
     "reason": "The edit replaces decontextualized trait recall with the article's explicit warning against reducing wabi-sabi to a short definition.",
     "paraphrase_rationale": "Repairs the source's English while retaining its claims that the four traits are an oversimplification and that cultural exposure supplies missing context."},
    {"fact_id": "fact-6c669507afcf171c5d53",
     "source_path": raw("rope365_1db41038891a4e1f.json"),
     "marker": "shinju 真珠 (pearl) and munenawa 胸縄",
     "decision": "drop",
     "reason_code": "conflated_chest_harness_terms",
     "reason": "The source presents shinju and munenawa as generic equivalents for chest harnesses without explaining their differing usage, and embeds them in an unsafe breathing-restriction exercise."},
    {"fact_id": "fact-7821b554008b1a2889dd",
     "source_path": raw("kinbakutoday_20d1161d0b76162f.json"),
     "marker": "the basic is an agreement and a confidential relationship",
     "manual_evidence": "The reason and the purpose might differ among people, but the basic is an agreement and a confidential relationship with a rope bottom, and not to just gratify one’s own desire. When emotions such as ‘I want a lover’ or ‘I want to play sex’ are clear on someone’s face, he is not attractive. Kinbaku, whip, wax play, Etc. might be awful things for common people, but it is our tendency to expect them for each other depending on a confidential relationship. Precisely because it is considered abnormal, we need to play more cautiously. Only then we can experience another pleasure.",
     "decision": "edit",
     "question": "What does Nawanojyoh say should ground kinbaku practice rather than self-gratification?",
     "answer": "Agreement and a trusted relationship with the rope bottom should ground the practice, rather than simply gratifying one’s own desire.",
     "reason_code": "clarify_source_attributed_agreement_and_trust",
     "reason": "The edit removes the translation's unnatural phrase 'confidential relationship' while preserving the attributed agreement, trust, and anti-self-gratification principle.",
     "paraphrase_rationale": "Uses 'trusted relationship' for the translated phrase 'confidential relationship' and retains the source's explicit contrast with gratifying one's own desire."},
    {"fact_id": "fact-bce15b367fb8de140d70",
     "source_path": raw("kinbakutoday_df9c70212a927199.json"),
     "marker": "Scenes can become more harsh than was intended",
     "decision": "edit",
     "question": "What risks does the article identify when hazukashii is translated simply as humiliation?",
     "answer": "Scenes can become more harsh than was intended and models can feel emotionally misread.",
     "reason_code": "replace_reaction_list_with_mistranslation_risk",
     "reason": "The edit replaces a mismatched list of physical and emotional reactions with the article's practical warning about harsher scenes and emotional misreading."},
    {"fact_id": "fact-9abcb867551f5f6b8c0b",
     "source_path": raw("rope365_6a23940f6bc37fc1.json"),
     "marker": "the knot will be less stable, impossible to lock",
     "decision": "edit",
     "question": "What three limitations does Rope365 identify for quick-release single-column ties?",
     "answer": "less stable, impossible to lock, and the bight cannot be used as an attachment point",
     "reason_code": "clarify_quick_release_tradeoffs_question",
     "reason": "The edit removes a day-number lookup and directly asks for the three operational compromises accompanying emergency release."},
    {"fact_id": "fact-0df8d417f95717af4ff9",
     "source_path": raw("rope365_a44c30b18f15504d.json"),
     "marker": "connecting line(s) don’t press on exposed nerves",
     "decision": "drop",
     "reason_code": "duplicate_precaution_in_high_risk_hogtie_context",
     "reason": "The generic exposed-nerve checklist item is embedded in a high-risk hogtie exercise and duplicates the surviving, more concrete hand-movement monitoring check from the same source."},
    {"fact_id": "fact-0de09627431b33c05134",
     "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "What do you need right now?",
     "decision": "edit",
     "question": "What open question does Rope365 recommend for supporting a partner after a scene is paused or stopped?",
     "answer": "What do you need right now?",
     "reason_code": "clarify_aftercare_support_prompt_context",
     "reason": "The edit adds the missing paused-or-stopped-scene context to the source's concrete support question."},
    {"fact_id": "fact-2676e5d4761356b170d3",
     "source_path": raw("kinbakutoday_c799ecf7b51f866b.json"),
     "marker": "ryuu” refers to a source from which things flow",
     "decision": "drop",
     "reason_code": "duplicate_martial_arts_etymology",
     "reason": "The narrow martial-arts etymology duplicates two clearer surviving definitions from this article and the Yukimura-ryū source."},
    {"fact_id": "fact-45b6fd8394bc104b35d2",
     "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "dropping a ball or ringing a bell",
     "decision": "edit",
     "question": "What nonverbal safety signals does the source give as alternatives when speech is restricted?",
     "answer": "dropping a ball or ringing a bell",
     "reason_code": "clarify_nonverbal_safety_signal_question",
     "reason": "The edit fixes a singular-question mismatch and clearly frames both examples as pre-agreed nonverbal alternatives."},
    {"fact_id": "fact-56153390adce08719f5b",
     "source_path": raw("rope365_0b4f44c8fabfc202.json"),
     "marker": "most handmade rope won’t as there is too much variation",
     "manual_evidence": "The strength of the rope is usually measured by the amount of loads that can be put on it before it breaks. This criteria has less impact if you aren’t doing suspension or climbing. Even weaker ropes like jute are used in suspension but the line system needs to have more redundancy to mitigate the risks. Industrial ropes such as nylon and cotton are likely to have official tensile strength on the package but most handmade rope won’t as there is too much variation in the making process to make a precise assessment.",
     "decision": "edit",
     "question": "Why may handmade rope lack an official tensile-strength rating?",
     "answer": "Handmade rope can vary too much in its construction for a precise tensile-strength assessment.",
     "reason_code": "replace_label_lookup_with_rating_limitation",
     "reason": "The edit replaces product-category recall with the source's important limitation on assigning precise strength ratings to variable handmade rope.",
     "paraphrase_rationale": "Repairs the source's grammar and directly states its reason that construction variability prevents a precise assessment."},
    {"fact_id": "fact-79c048429e93f627201e",
     "source_path": raw("kinbakutoday_d7aad07b39a5b1e1.json"),
     "marker": "Bakushi: Scenes from Nikkatsu’s Roman Porno SM Dramas",
     "decision": "drop",
     "reason_code": "promotional_biography_title_lookup",
     "reason": "The biography-title lookup comes from a movie-promotion article centered on extreme staged suspensions and adds little durable historical understanding."},
    {"fact_id": "fact-db5df8a1f479dc24a573",
     "source_path": raw("kinbakutoday_df9c70212a927199.json"),
     "marker": "humiliation play within Western BDSM is not a single, uniform experience",
     "decision": "edit",
     "question": "What caveat does the article give about Western BDSM humiliation play?",
     "answer": "Humiliation play within Western BDSM is not a single, uniform experience. In practice it can range from harsh forms of degradation and power exchange to much lighter dynamics involving playful embarrassment, teasing, or exhibitionism.",
     "reason_code": "replace_reductive_humiliation_claim_with_range",
     "reason": "The edit removes a reductive attack-on-self formulation and restores the article's explicit range from harsh degradation to lighter teasing or exhibitionism."},
    {"fact_id": "fact-4844ac7f2672a5f62e82",
     "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["deep_dive_single_columns", "hip_harness_playlist",
                      "strugglers_knot_somerville_bowline"],
     "evidence": "Deep Dive into Single Columns: https://www.youtube.com/watch?v=vnDvjAaQU8g; Hip Harness Videos: https://www.youtube.com/playlist?list=PLkrdRffh_Gg2S9QccbRyiLE5x4SIacgoM; Struggler's Knot and Somerville Bowline: https://www.youtube.com/watch?v=OsIcEtCoKHo",
     "decision": "keep",
     "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that all three exact tutorial links remain memorized for relevant rope-resource questions."},
    {"fact_id": "fact-ee8e2cd6104d741c4967",
     "source_path": raw("esinem_87458162bcec570b.json"),
     "marker": "culinary ingredients can be substituted",
     "decision": "drop",
     "reason_code": "promotional_analogy_in_high_risk_tutorial",
     "reason": "The culinary analogy comes from a promotion for choker ties and floor suspensions and adds less value than the source's surviving construction-category fact."},
    {"fact_id": "fact-339cfc45e7f31c668db5",
     "source_path": raw("kinbakutoday_0d87b1ac1a49f2d4.json"),
     "marker": "magazines, reader letters, captions, advertisements, censorship battles",
     "decision": "edit",
     "question": "What source types does the article say a fuller history of kinbaku should examine beyond masters and lineages?",
     "answer": "magazines, reader letters, captions, advertisements, censorship battles",
     "reason_code": "clarify_historical_source_criticism_question",
     "reason": "The edit removes a person-dependent prompt while retaining concrete primary-source categories for evaluating kinbaku history."},
    {"fact_id": "fact-59ed68f7f1f2816cebbe",
     "source_path": raw("rope365_8e6f4abea3bf4f6d.json"),
     "marker": "Progressively adding turns will make frictions more and more solid",
     "manual_evidence": "Once the rope is safely anchored, the rest of the rope can be used to continue tying. The action of rope wrapping around rope creates friction. It can be used to add structure to a rope pattern without any knot. Frictions are great because they are fast to tie and don’t compact into something hard to untie but they require tension to stay in place.\nProgressively adding turns will make frictions more and more solid. It is the compromise between speed and solidity, you can choose what’s best depending on the context and intentions. Some common frictions have names, usually because of their aesthetic or efficiency but there is no problem creating your own by making more or less turns.\nHere we explore the basic frictions and decompose the X frictions. There are infinite ways to make frictions and we will explore more styles on Day 204.\nFull Turn\nWrapping a rope around another rope, making a complete loop, creates a first hint of structure. This is also known as 360 friction, loop around or wrap around.",
     "decision": "edit",
     "question": "What tradeoff does Rope365 identify when adding turns to a friction?",
     "answer": "Additional turns make a friction more solid, trading speed for solidity according to the context and intended use.",
     "reason_code": "replace_friction_aliases_with_functional_tradeoff",
     "reason": "The edit replaces a list of aliases with the source's transferable speed-versus-solidity tradeoff.",
     "paraphrase_rationale": "Combines the source's adjacent statements into one grammatical sentence without changing the effect of added turns or the contextual tradeoff."},
    {"fact_id": "fact-957aca0799be5e77cb7e",
     "source_path": raw("kinbakutoday_1775dd4176b24104.json"),
     "marker": "you are learning to be competent, not knowledgeable",
     "decision": "drop",
     "reason_code": "duplicate_teaching_competence_distinction",
     "reason": "A clearer surviving fact from the same source already explains both sides of the competence-versus-knowledge distinction for teaching."},
    {"fact_id": "fact-82c84e1f83e6a246e2cf",
     "source_path": raw("rope365_5a4dc6c711aaa005.json"),
     "marker": "recommended to avoid this kind of rope when strength is important",
     "decision": "edit",
     "question": "What does Rope365 recommend when high stranding appears inside a strand and strength matters?",
     "answer": "It is recommended to avoid this kind of rope when strength is important as this is generally the sign that the tension inside the rope might weaken it.",
     "reason_code": "replace_risky_repair_with_strength_warning",
     "reason": "The edit removes an underspecified load-bearing-rope repair and retains the source's safer warning to avoid internally high-stranded rope when strength matters."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        288, 250, 207, 344, 253, 233, 451, 119, 123, 397,
        352, 121, 27, 320, 33, 347, 529, 261, 208, 505,
        117, 360, 234, 155, 230,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (18 if index < 10 else 19 if index < 20 else 20)
    for index, fact_id in enumerate(EXPECTED_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v29",
    "eligible_prior_keeps_before_reaudit_exclusion": 255,
    "secondary_eligible_prior_keeps": 30,
    "rows": 587,
    "sha256": "e05be81d4c4e2cc9038c9225cbb4372b2bbc627cb4ba219c4dc93c14ad87ba13",
    "v21_v29_reviewed_fact_ids_excluded": 225,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v29": 549,
    "active_after_this_tranche": 542,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 7,
    "new_edits_applied": 17,
    "output_rows": 580,
    "output_sha256": "3b28bdbdc78503cf1beb497630581d83751982a00d206145003eb92def3099dd",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 1,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 30):
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
    if len(ranked) != 30:
        raise ValueError(f"v30 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:25]) != EXPECTED_SELECTION:
        raise ValueError("v30 secondary selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return secondary_ranked(rows), 0, 0


@contextlib.contextmanager
def patched_base():
    replacements = {
        "OUT_DIR": OUT_DIR, "AUDIT": AUDIT, "CURATION": CURATION,
        "REPORT": REPORT, "REVIEWER": REVIEWER,
        "REVIEWED_AT": REVIEWED_AT,
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
        row["schema"] = "context-merit-audit-v30"
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
    manual_ids = {
        spec["fact_id"]: spec["paraphrase_rationale"] for spec in SPECS
        if "paraphrase_rationale" in spec
    }
    for row in curations:
        if row["fact_id"] in manual_ids:
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = manual_ids[row["fact_id"]]
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v30"
    report["selection"].update({
        "active_rows": 587,
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 30,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v29 projection, and the fact was not reviewed "
                "again in v21 through v29"
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
        "extractive": 12, "manual_paraphrase": 5,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
