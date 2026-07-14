#!/usr/bin/env python3
"""Deterministically re-audit the next weak-context surviving keeps in v28."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V27_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v27"
sys.path[:0] = [str(ROOT), str(V27_DIR)]
import build_context_merit_audit_v27 as previous

BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v28.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v28.jsonl"
REPORT = OUT_DIR / "report_context_merit_v28.json"
REVIEWER = "codex-context-merit-audit-v28"
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
    for version in range(1, 28)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 28)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-63283786303c4df91008", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["rw_rope", "chromaknotz", "knothead_nylon"],
     "evidence": "RW Rope (synthetic upline rope): https://www.rwrope.com/; ChromaKnotz (synthetic bondage rope): https://chromaknotz.square.site/; Knot Head Nylon (synthetic bondage rope): https://knotheadnylon.net/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that all three synthetic-rope supplier URLs remain available for resource lookup."},
    {"fact_id": "fact-2ed5cfecf9fd87c86405", "source_path": raw("wikipedia_a19eb7f4fbb670b4.json"),
     "marker": "self-rigging or self-suspension", "decision": "drop",
     "reason_code": "conflated_self_tying_terms",
     "reason": "The answer treats self-rigging and self-suspension as interchangeable names for all self-tying, erasing the important distinction between floor work and suspension."},
    {"fact_id": "fact-74b67e379ff5d1c80099", "source_path": raw("rope365_c00bc803927bda26.json"),
     "marker": "Hitches are a family of knots used to bind a rope to an object, including another rope", "decision": "edit",
     "question": "What is a hitch?",
     "answer": "a family of knots used to bind a rope to an object, including another rope",
     "reason_code": "replace_high_risk_tie_alias_with_hitch_definition",
     "reason": "The edit replaces a high-risk adjustable shrimp-tie alias with the source's safe, general definition of a hitch."},
    {"fact_id": "fact-fb98f437712003cfcb5e", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["ropecraft"], "evidence": "ROPECRAFT: https://ropecraft.net/", "decision": "keep",
     "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that ROPECRAFT remain available as a United States rope-convention resource."},
    {"fact_id": "fact-5b41f161d5e544be5df1", "source_path": raw("wikipedia_a97e19abfe49afa9.json"),
     "marker": "more than 25 traditional ties", "decision": "drop",
     "reason_code": "unsafe_reference_count_trivia",
     "reason": "A count of techniques in a historical prisoner-restraint manual is incidental trivia and points toward unsafe instructional material without modern risk context."},
    {"fact_id": "fact-89d30b473e2d3d1b2ff2", "source_path": raw("kinbakutoday_00507964d2e38f67.json"),
     "marker": "bakushi to first produce photo books in Japan featuring kinbaku", "decision": "edit",
     "question": "What publishing milestone does the source attribute to Minomura Kou?",
     "answer": "the bakushi to first produce photo books in Japan featuring kinbaku",
     "reason_code": "replace_book_nickname_with_publishing_contribution",
     "reason": "The edit replaces a collector nickname with the article's substantive claim about Minomura's contribution to Japanese kinbaku publishing."},
    {"fact_id": "fact-5dd070f289170de1736a", "source_path": raw("kinbakutoday_aeb5f266584c30d1.json"),
     "marker": "The Indecent Rope Kamasutra", "decision": "drop",
     "reason_code": "promotional_explicit_title_lookup",
     "reason": "The translated title appears in a crowdfunding promotion for easily reproduced explicit positions and adds little durable safety, consent, or historical understanding."},
    {"fact_id": "fact-2434e4e894af5c3b05f7", "source_path": raw("rope365_5a4dc6c711aaa005.json"),
     "marker": "Discussing rope maintenance with your partner is part of informed consent", "decision": "edit",
     "question": "Why should rope maintenance and hygiene practices be discussed with a partner?",
     "answer": "Discussing rope maintenance with your partner is part of informed consent.",
     "reason_code": "replace_lay_trivia_with_hygiene_consent",
     "reason": "The edit replaces a narrow repair comparison with the source's broader and more important connection between hygiene practices and informed consent."},
    {"fact_id": "fact-2a1bf6a6cce932c51390", "source_path": raw("rope365_5fdb5e78c2471772.json"),
     "marker": "Twisted or braided", "decision": "drop",
     "reason_code": "low_value_observation_exercise",
     "reason": "Naming two construction types from a take-apart exercise is low-value recall and duplicates better material and construction coverage in the curated dataset."},
    {"fact_id": "fact-575ef3a6bf4535b20142", "source_path": raw("esinem_64c4212a11867d61.json"),
     "marker": "it is the one that will create the initial impression and set the mood", "decision": "edit",
     "question": "Why does Esinem describe the first tie of a rope session as especially important?",
     "answer": "it is the one that will create the initial impression and set the mood",
     "reason_code": "replace_gote_alias_with_first_tie_rationale",
     "reason": "The edit replaces another gote alias with the source's relational explanation of why the opening tie matters."},
    {"fact_id": "fact-62163ba3520628ad592e", "source_path": raw("rope365_d37244d61d012d8b.json"),
     "marker": "when wrapping the leg straight, folding it afterward may cause compression on the limb", "decision": "edit",
     "question": "What compression warning does Rope365 give when wrapping a leg while it is straight?",
     "answer": "when wrapping the leg straight, folding it afterward may cause compression on the limb",
     "reason_code": "replace_pose_alias_with_limb_compression_warning",
     "reason": "The edit removes a lower-body pose alias and retains the source's warning that changing joint position after wrapping can create compression."},
    {"fact_id": "fact-7a2364ae08d837da2d48", "source_path": raw("rope365_c706558d746d11de.json"),
     "marker": "Ichinawa and Ipponnawa", "decision": "drop",
     "reason_code": "promotional_activity_alias",
     "reason": "The two aliases are listed in a dated curriculum announcement without a definition, safety framing, or enough context to stand alone."},
    {"fact_id": "fact-c0cc30ba0b937b0f11f1", "source_path": raw("kinbakutoday_cbbff84b319d0813.json"),
     "marker": "prevention of escape their primary object", "decision": "keep",
     "reason_code": "hojojutsu_function_context_complete",
     "reason": "The named author's statement identifies the historical function of capture rope and is clearly framed as history rather than modern bondage instruction."},
    {"fact_id": "fact-994ebfad714f14f3be5d", "source_path": raw("kinbakutoday_86c13124375e690a.json"),
     "marker": "Flower and Snake", "decision": "drop",
     "reason_code": "duplicate_promotional_media_title",
     "reason": "The media-title lookup duplicates Oniroku Dan coverage and comes from the same dated fiction-submission call rejected in v27."},
    {"fact_id": "fact-c5b7b45a142ef29c7a32", "source_path": raw("rope365_7b0aa0a7481d6e35.json"),
     "marker": "zenpou takate shibari 前方高手縛り", "decision": "drop",
     "reason_code": "duplicate_high_risk_pose_alias",
     "reason": "The translated pose alias adds no safety context, and v27 already retained the more useful accessibility rationale from this source."},
    {"fact_id": "fact-fa78f985ebbbb2370a5f", "source_path": raw("rope365_c73bc6fb66977a2d.json"),
     "marker": "first contact with a new partner in a public space", "decision": "keep",
     "reason_code": "new_partner_safety_context_complete",
     "reason": "The public first-contact recommendation is directly supported alongside reference checks and a safety-call precaution."},
    {"fact_id": "fact-474bee747d1af1e3b77c", "source_path": raw("kinbakutoday_d33c9aadf7e593e7.json"),
     "marker": "Somatics for Rope Bottoms", "decision": "drop",
     "reason_code": "promotional_book_title_lookup",
     "reason": "The title lookup comes from a short promotional book announcement whose broad somatic claims are not developed or independently supported."},
    {"fact_id": "fact-f05b17a22ab0bd03af7b", "source_path": raw("rope365_8ae9e3d93b31601b.json"),
     "marker": "Which one was chosen is often a subjective choice and is a reflection of the period of time these texts were written. Language is fluid and continues to evolve", "decision": "edit",
     "question": "Why should Rope365’s vocabulary labels not be treated as fixed?",
     "answer": "Which one was chosen is often a subjective choice and is a reflection of the period of time these texts were written. Language is fluid and continues to evolve.",
     "reason_code": "replace_tie_term_with_vocabulary_caveat",
     "reason": "The edit replaces a high-hands tie term with the glossary's important warning that naming is subjective, historical, and evolving."},
    {"fact_id": "fact-16926b4cd64215195ce8", "source_path": raw("kinbakutoday_86086b3e64ae77fa.json"),
     "marker": "you aren’t obligated to tell Kristoff anything",
     "manual_evidence": "What’s wrong with this? This is a tricky one. First, you aren’t obligated to tell Kristoff anything, especially if it causes you any kind of mental or emotional pain or fear (of retribution, of physical harm, etc.). But by sharing the info with other people and nottelling Kristoff, you run the risks that a) only half the story will get spread around, creating a biased and possibly inaccurate picture, and b) Kristoff will never realize what he did wrong and will keep doing it to others.\nThe better way: Again, this is tricky, and getting injured or violated brings up other issues that may override being considerate of the partner involved. Personally, however, I think it’s a good idea to muster up your courage and discuss it with Kristoff, unless it might cause you harm, as noted above. And I think it’s perfectly OK to discuss your experience with a trusted friend, but I urge you to examine your reasons for doing so and to do so in a thoughtful, honorable way. Telling close friend Elsa in confidence so she can help you process or deal with what happened is one thing. Telling a dozen people that you’re not going to a party because Kristoff will be there and you’re avoiding him because he injured you is entirely another. Talking with a trained counselor can be a good idea too.",
     "decision": "edit",
     "question": "Does the source say a harmed person is obligated to confront the rope partner?",
     "answer": "No. A harmed person is not obligated to confront the partner, especially if doing so could cause mental or emotional pain, fear, or other harm.",
     "reason_code": "replace_support_role_lookup_with_no_confrontation_obligation",
     "reason": "The edit preserves the source's survivor-centered boundary that confrontation is not owed when it could cause fear, distress, retaliation, or physical harm."},
    {"fact_id": "fact-3b647333a1ace1f99666", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "ulnar nerve is often exposed at the elbow", "decision": "keep",
     "reason_code": "nerve_anatomy_context_complete",
     "reason": "The source directly connects the exposed ulnar nerve at the elbow with the familiar funny-bone location in a placement discussion."},
    {"fact_id": "fact-531fe868840964d4d4af", "source_path": raw("rope365_5c6e72d053d47d66.json"),
     "marker": "rope pressure should be on the forearms", "decision": "keep",
     "reason_code": "wrist_pressure_safety_context_complete",
     "reason": "The question preserves the source's explicit precaution to avoid direct pressure on the wrists and hands in the described structure."},
    {"fact_id": "fact-ec22c5ec841d990b5c62", "source_path": raw("kinbakutoday_5354aadb617a8182.json"),
     "marker": "Shunga erotic paintings", "decision": "drop",
     "reason_code": "anecdotal_historical_inference",
     "reason": "The artwork lookup supports an interviewee's broad inference about a long history of rope play, but the source provides no specific work, date, or historical analysis."},
    {"fact_id": "fact-42e3526e742dc48ac095", "source_path": raw("rope365_8a4c32897b80339b.json"),
     "marker": "wrist, elbows, knees, ankles", "decision": "keep",
     "reason_code": "nerve_risk_mitigation_context_complete",
     "reason": "The source explicitly identifies these joints in its advice to tie loosely or away from joints to mitigate nerve compression."},
    {"fact_id": "fact-2f3e2526a82142759634", "source_path": raw("kinbakutoday_381214111022a952.json"),
     "marker": "rope bottom that will lose the feeling of connection first", "decision": "keep",
     "reason_code": "source_attributed_partner_attunement",
     "reason": "The answer preserves Yukimura's explicitly attributed observation that a partner may sense disengagement before the person tying recognizes it."},
    {"fact_id": "fact-6ad3ef0f91eb2f4932f0", "source_path": raw("rope365_5a4dc6c711aaa005.json"),
     "marker": "Synthetic and cotton ropes are easy to wash", "decision": "keep",
     "reason_code": "rope_hygiene_context_complete",
     "reason": "The source directly distinguishes easy-to-wash synthetic and cotton ropes from natural fibers weakened by wet cleaning."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        527, 265, 419, 447, 91, 232, 243, 107, 521, 229,
        117, 230, 15, 495, 240, 418, 268, 239, 417, 510,
        465, 413, 502, 35, 543,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (13 if index < 3 else 14 if index < 12 else
              15 if index < 22 else 16)
    for index, fact_id in enumerate(EXPECTED_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v27",
    "eligible_prior_keeps_before_reaudit_exclusion": 255,
    "secondary_eligible_prior_keeps": 80,
    "rows": 600,
    "sha256": "a1a2574d502aeeee4fe792373ff7858a65405479d85b2ae5827f710868b6b54a",
    "v21_v27_reviewed_fact_ids_excluded": 175,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v27": 562,
    "active_after_this_tranche": 553,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 9,
    "new_edits_applied": 7,
    "output_rows": 591,
    "output_sha256": "790549d4a1a9f65c7538ea50e6eb6f329b5bd6ae429cd4cac12cf38bee8e2b6e",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 9,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 28):
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
    if len(ranked) != 80:
        raise ValueError(f"v28 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:25]) != EXPECTED_SELECTION:
        raise ValueError("v28 secondary selection drift")
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
        row["schema"] = "context-merit-audit-v28"
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
    paraphrase_fact_id = "fact-16926b4cd64215195ce8"
    for row in curations:
        if row["fact_id"] == paraphrase_fact_id:
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = (
                "Removes fictional names while preserving the source's "
                "explicit no-confrontation obligation and its mental, "
                "emotional, retaliatory, and physical-harm conditions."
            )
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v28"
    report["selection"].update({
        "active_rows": 600,
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 80,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v27 projection, and the fact was not reviewed "
                "again in v21 through v27"
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
        "extractive": 6, "manual_paraphrase": 1,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
