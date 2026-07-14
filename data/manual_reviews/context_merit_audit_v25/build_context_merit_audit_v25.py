#!/usr/bin/env python3
"""Deterministically re-audit the next weak-context surviving keeps in v25."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V24_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v24"
sys.path[:0] = [str(ROOT), str(V24_DIR)]
import build_context_merit_audit_v24 as previous

BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v25.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v25.jsonl"
REPORT = OUT_DIR / "report_context_merit_v25.json"
REVIEWER = "codex-context-merit-audit-v25"
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
    for version in range(1, 25)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 25)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-09b7cf8b4887281e153d", "source_path": raw("kinbakutoday_c799ecf7b51f866b.json"),
     "marker": "The kokoro is the “why” and the katachi is the “how” of tying", "decision": "edit",
     "question": "What two elements does the author say a ryuu combines?",
     "answer": "The kokoro is the “why” and the katachi is the “how” of tying.",
     "reason_code": "replace_school_label_with_source_framework",
     "reason": "The edit replaces a one-word school label with the source's substantive distinction between a style's motivating heart and its technique."},
    {"fact_id": "fact-0fd37dbddac15fb0325d", "source_path": raw("kinbakutoday_cbbff84b319d0813.json"),
     "marker": "one influence among many: not a stable lineage, but a touchstone", "decision": "edit",
     "question": "How does the source ultimately characterize hojojutsu’s relationship to modern kinbaku?",
     "answer": "one influence among many: not a stable lineage, but a touchstone",
     "reason_code": "replace_term_lookup_with_historical_nuance",
     "reason": "The edit replaces a bare term lookup with the article's carefully qualified conclusion about influence without direct continuity."},
    {"fact_id": "fact-c49ef46be45ea5bf079e", "source_path": raw("kinbakutoday_dc711609dfc7a35f.json"),
     "marker": "create the nawajiri, the point of connection, control and communication", "decision": "keep",
     "reason_code": "source_defined_connection_term_context_complete",
     "reason": "The question embeds the source's three-part definition, making the Japanese term meaningful without providing a standalone high-risk tie recipe."},
    {"fact_id": "fact-569503ae1926b6987900", "source_path": raw("rope365_a3b2e9e479b0c70f.json"),
     "marker": "Asking to try other people’s rope and asking them about their experience with it", "decision": "edit",
     "question": "What hands-on approach does Rope365 suggest for comparing rope qualities before buying?",
     "answer": "Asking to try other people’s rope and asking them about their experience with it",
     "reason_code": "replace_kit_length_preference_with_try_before_buy_advice",
     "reason": "The edit replaces a context-dependent kit-length claim with durable, low-risk shopping advice grounded in direct comparison and user experience."},
    {"fact_id": "fact-6215416a2438227520a3", "source_path": raw("kinbakutoday_381214111022a952.json"),
     "marker": "basic knowledge and technique in order to protect the rope bottoms’ safety", "decision": "edit",
     "question": "Why did Yuki say basic knowledge and technique are necessary in kinbaku?",
     "answer": "to protect the rope bottoms’ safety",
     "reason_code": "replace_humiliation_technique_lookup_with_safety_purpose",
     "reason": "The edit removes a humiliation-technique label from an unsafe instructional context and retains the participant's explicit reason for learning fundamentals."},
    {"fact_id": "fact-aa51767b03dd040b7b37", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "known as spotter", "decision": "keep",
     "reason_code": "self_tie_supervision_term_context_complete",
     "reason": "The question directly defines the safety role of checking on or supervising a self-tying person."},
    {"fact_id": "fact-58af822afb984d0315d9", "source_path": raw("kinbakutoday_011f67c75b8f999f.json"),
     "marker": "Humility is necessary to question oneself all the time", "decision": "keep",
     "reason_code": "source_attributed_reflective_principle",
     "reason": "The question explicitly attributes the reflective principle to the author and preserves its full kinbaku context."},
    {"fact_id": "fact-5e1831451feb28ca6497", "source_path": raw("wikipedia_19338018629394d7.json"),
     "marker": "tendency to work loose when not under load", "decision": "edit",
     "question": "Which deficiencies can make a common bowline unsuitable for safety-critical use?",
     "answer": "to work loose when not under load (or under cyclic loading), to slip when pulled sideways, and the bight portion of the knot to capsize in certain circumstances",
     "reason_code": "replace_mnemonic_trivia_with_knot_failure_modes",
     "reason": "The edit replaces rabbit-mnemonic recall with the source's concrete limitations relevant to deciding when a more secure knot is needed."},
    {"fact_id": "fact-b277f525a0012a481fd4", "source_path": raw("esinem_0e05c5a35e129510.json"),
     "marker": "masters like Yukimura can achieve all they want with so little rope", "decision": "drop",
     "reason_code": "volatile_or_promotional_person_trivia",
     "reason": "The named-person example appears in a promotional tutorial announcement and adds little durable knowledge beyond subjective mastery praise."},
    {"fact_id": "fact-11facb540676c51e7ebb", "source_path": raw("rope365_8ae9e3d93b31601b.json"),
     "marker": "Predicament – A situation in which the different options will yield a negative outcome", "decision": "keep",
     "reason_code": "bondage_term_definition_context_complete",
     "reason": "The question fully defines agency among painful or difficult options before asking for the established vocabulary term."},
    {"fact_id": "fact-17b5a49e3a54220c3c36", "source_path": raw("kinbakutoday_c5e568667b495473.json"),
     "marker": "正座 seiza", "decision": "keep",
     "reason_code": "sitting_position_term_context_complete",
     "reason": "The question supplies the complete neutral body position and asks only for its Japanese name."},
    {"fact_id": "fact-18128e1fc133eaf19e89", "source_path": raw("esinem_77368ccdc66acad0.json"),
     "marker": "That’s Risk Aware Consensual Kink", "decision": "keep",
     "reason_code": "consent_framework_acronym_context_complete",
     "reason": "The concise expansion defines a common risk-and-consent framework used in BDSM discussion."},
    {"fact_id": "fact-048e419a15cfc8118217", "source_path": raw("rope365_5a4dc6c711aaa005.json"),
     "marker": "The cleaning process will progressively weaken the rope", "decision": "edit",
     "question": "Why does Rope365 recommend washing rope only when needed?",
     "answer": "The cleaning process will progressively weaken the rope.",
     "reason_code": "replace_high_stranding_duplicate_with_cleaning_tradeoff",
     "reason": "The edit removes a duplicate high-stranding fact and retains the source's practical reason for balancing hygiene against rope longevity."},
    {"fact_id": "fact-41006d3854057b0f888e", "source_path": raw("rope365_1ee277117c1ce420.json"),
     "marker": "square lashing is a great technique to bind two columns at a 90-degree angle", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_person_tie_lookup",
     "reason": "The angle lookup comes from an exercise that transfers structural lashing directly to limbs without fit, anatomy, circulation, monitoring, or release guidance."},
    {"fact_id": "fact-4a1a5f61155506c25498", "source_path": raw("rope365_bc4120687a97c8c3.json"),
     "marker": "neither of these knots are fully secured", "decision": "edit",
     "question": "Why should neither a square knot nor a granny knot be used when solidity is critical?",
     "answer": "neither of these knots are fully secured and they should not be used when solidity is critical. They can both capsize when jerking the ends.",
     "reason_code": "replace_decorative_alias_with_knot_security_limit",
     "reason": "The edit replaces Japanese-name recall with the source's explicit warning about two familiar knots' security limitations."},
    {"fact_id": "fact-5e4a5bf55948ce3fca27", "source_path": raw("esinem_888997ffcb0c181d.json"),
     "marker": "concept of public kinbaku shows began with Eikichi Osada", "decision": "drop",
     "reason_code": "source_qualified_but_incidental_person_trivia",
     "reason": "The attribution is presented only as the author's recollection inside a dated media-correction post and is not developed or independently supported there."},
    {"fact_id": "fact-0c5693d6e4d98729f4b3", "source_path": raw("kinbakutoday_4df72c39939f2a24.json"),
     "marker": "hebizeme or “snake torment”", "decision": "drop",
     "reason_code": "contextless_or_low_value_torture_term",
     "reason": "The translation labels a violent fantasy depicted in two artworks but contributes little practical, safety, consent, or broader historical understanding."},
    {"fact_id": "fact-1d018345158308bd6605", "source_path": raw("kinbakutoday_aeee5ea2ccc3874a.json"),
     "marker": "power of the image comes from the story that is being told", "decision": "edit",
     "question": "What does the author say gives early-1950s kinbaku photographs their power?",
     "answer": "the story that is being told",
     "reason_code": "replace_bare_decade_with_attributed_aesthetic_insight",
     "reason": "The edit keeps the period as context while asking for the article's central observation about narrative rather than date recall."},
    {"fact_id": "fact-2bb654a348c7338992b9", "source_path": raw("kinbakutoday_2ded51b08225bd4b.json"),
     "marker": "Yukimura ryu (with one rope)", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_person_tie_lookup",
     "reason": "The style-and-rope-count lookup is attached to a high-hands gote essay that explicitly declines to resolve suspension safety and later dismisses that concern."},
    {"fact_id": "fact-66a988eebc31a6ade1e4", "source_path": raw("rope365_682937f92222bf87.json"),
     "marker": "What was the context of the picture", "decision": "edit",
     "question": "What context question does Rope365 suggest asking when analyzing an old bondage image?",
     "answer": "What was the context of the picture (ex: photoshoot, play, performance)?",
     "reason_code": "replace_museum_credit_with_source_analysis_prompt",
     "reason": "The edit replaces an image-credit lookup with a reusable critical question for interpreting historical bondage imagery."},
    {"fact_id": "fact-69734d1ddfd89a3214fd", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "Consent can be withdrawn at any point during play", "decision": "edit",
     "question": "Can consent be withdrawn after tying has begun?",
     "answer": "Consent can be withdrawn at any point during play.",
     "reason_code": "replace_resource_author_lookup_with_consent_rule",
     "reason": "The edit replaces incidental resource authorship with the article's explicit and broadly useful consent rule."},
    {"fact_id": "fact-6e8831744f619c0ee923", "source_path": raw("rope365_bc4120687a97c8c3.json"),
     "marker": "used to represent good relationships", "decision": "edit",
     "question": "What does the awajimusubi, or double coin knot, represent in Japan?",
     "answer": "good relationships",
     "reason_code": "replace_macrame_alias_with_cultural_meaning",
     "reason": "The edit replaces another alias lookup with the source's concise cultural meaning for the decorative knot."},
    {"fact_id": "fact-d00f6772e10f963f9e17", "source_path": raw("esinem_f358e1984a8e04dc.json"),
     "marker": "understanding of each element’s purpose and how they can be interchanged", "decision": "edit",
     "question": "What does Esinem say learners should understand instead of merely replicating shibari patterns?",
     "answer": "each element’s purpose and how they can be interchanged",
     "reason_code": "replace_tie_alias_with_component_learning_principle",
     "reason": "The edit replaces box-tie terminology with the article's more useful principle of understanding component function and substitution."},
    {"fact_id": "fact-d7e6a05a62554a6dbe9d", "source_path": raw("rope365_049578f567a2879d.json"),
     "marker": "Monitor nerves by scratching the hands", "manual_evidence": "Monitor nerves by scratching the hands and validating movements in fingers and wrists. Adjust the placement on the arms to avoid sensitive spots and avoid rope that presses in the armpit.",
     "decision": "edit",
     "question": "What nerve-monitoring and placement checks does Rope365 give for an open-diamond box tie?",
     "answer": "Check hand sensation and finger/wrist movement, adjust arm placement away from sensitive spots, and prevent armpit pressure.",
     "reason_code": "replace_vague_location_lookup_with_complete_monitoring_checks",
     "reason": "The edit restores the source's motor and sensory checks and couples the armpit warning with broader placement guidance."},
    {"fact_id": "fact-0ebc180747663c175f13", "source_path": raw("rope365_4196a8f7e326a5cc.json"),
     "marker": "Beauty doesn’t need to be perfect", "decision": "edit",
     "question": "What idea about beauty does Rope365 connect with kuzushi and wabi-sabi?",
     "answer": "Beauty doesn’t need to be perfect; it can emerge from the details of a unique process.",
     "reason_code": "replace_concept_list_with_aesthetic_principle",
     "reason": "The edit replaces a two-term list with the complete aesthetic principle those concepts illustrate in the source."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        285, 250, 251, 280, 385, 270, 330, 122, 525, 244,
        240, 189, 567, 120, 237, 580, 248, 102, 546, 528,
        583, 225, 373, 361, 413,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (7 if index < 13 else 8)
    for index, fact_id in enumerate(EXPECTED_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v24",
    "eligible_prior_keeps_before_reaudit_exclusion": 255,
    "secondary_eligible_prior_keeps": 155,
    "rows": 619,
    "sha256": "267fa20ea6c41521147d83e1086a19a48d04478ddc52f9b442cb1c8cd231c72d",
    "v21_v24_reviewed_fact_ids_excluded": 100,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v24": 581,
    "active_after_this_tranche": 576,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 5,
    "new_edits_applied": 14,
    "output_rows": 614,
    "output_sha256": "fc704352e4bd193c10b81117c54459bb96cb111274536d6d58e6ab3fe4220d8a",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 6,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 25):
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
    if len(ranked) != 155:
        raise ValueError(f"v25 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:25]) != EXPECTED_SELECTION:
        raise ValueError("v25 secondary selection drift")
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
        row["schema"] = "context-merit-audit-v25"
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
    paraphrase_fact_id = "fact-d7e6a05a62554a6dbe9d"
    for row in curations:
        if row["fact_id"] == paraphrase_fact_id:
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = (
                "Condenses two adjacent source sentences into one checklist "
                "while retaining hand sensation, finger and wrist movement, "
                "sensitive arm placement, and armpit-pressure checks."
            )
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v25"
    report["selection"].update({
        "active_rows": 619,
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 155,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v24 projection, and the fact was not reviewed "
                "again in v21 through v24"
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
        "extractive": 13, "manual_paraphrase": 1,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
