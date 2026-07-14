#!/usr/bin/env python3
"""Deterministically re-audit the next weak-context surviving keeps in v23."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V22_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v22"
sys.path[:0] = [str(ROOT), str(V22_DIR)]
import build_context_merit_audit_v22 as previous

BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v23.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v23.jsonl"
REPORT = OUT_DIR / "report_context_merit_v23.json"
REVIEWER = "codex-context-merit-audit-v23"
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
    for version in range(1, 23)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 23)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-a23aee2e599cfbccafd5", "source_path": raw("rope365_2ea01101bf29d77c.json"),
     "marker": "Most twisted ropes are made from 3 strands", "decision": "keep",
     "reason_code": "twisted_rope_construction_context_complete",
     "reason": "The question gives the construction type and the source's stability rationale for a reusable basic rope-material fact."},
    {"fact_id": "fact-388ba93105c948d44738", "source_path": raw("kinbakutoday_f370696af0359092.json"),
     "marker": "were wood printed theater posters", "decision": "keep",
     "reason_code": "theater_poster_term_context_complete",
     "reason": "The question defines the posters by medium and narrative function, preserving a useful term in Ito Seiu's theater and visual-culture context."},
    {"fact_id": "fact-ec9b9dbbca11f8e3134c", "source_path": raw("kinbakutoday_92c9fc29a66300a0.json"),
     "marker": "no one, not even those involved with many of Nikkatsu’s productions seem to know who that person was", "decision": "edit",
     "question": "Why does the source call Hitoshi Sharaku, the credited bondage adviser for Fairy in a Cage, a mystery bakushi?",
     "answer": "no one, not even those involved with many of Nikkatsu’s productions seem to know who that person was",
     "reason_code": "replace_writer_lookup_with_credit_mystery",
     "reason": "The edit replaces incidental screenwriter recall with the article's central archival uncertainty about the bondage adviser's identity."},
    {"fact_id": "fact-4cc86a9034f52e41c611", "source_path": raw("kinbakutoday_454370c8a4f42708.json"),
     "marker": "They paid manuscript fees, so enthusiasts would submit their writings", "decision": "edit",
     "question": "Why did Saikatsu think Kitan Club could continue until 1975 despite periodic publication suspensions?",
     "answer": "They paid manuscript fees, so enthusiasts would submit their writings",
     "reason_code": "replace_bare_end_year_with_reader_contribution_model",
     "reason": "The edit retains the date as context but makes the magazine's enthusiast-contribution model the answer."},
    {"fact_id": "fact-6495f735547cdca55497", "source_path": raw("rope365_d09bf786f1bfe97f.json"),
     "marker": "kimono tie is inspired by the Japanese tasuki", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_person_tie_lookup",
     "reason": "A one-word inspiration lookup attached to a box-tie variant contributes no anatomy, fit, monitoring, loading, or emergency-release guidance."},
    {"fact_id": "fact-88343fa63a33ac34ce71", "source_path": raw("kinbakutoday_1c6818645cd43d8f.json"),
     "marker": "focused much more on bondage and SM play", "decision": "edit",
     "question": "What lesser-known side of Ozuma’s art does the source say Phantom Mirror reveals?",
     "answer": "work focused much more on bondage and SM play",
     "reason_code": "replace_bare_publication_year_with_artistic_focus",
     "reason": "The edit uses the book and year as context while asking what the source says the less familiar work contributes to Ozuma's artistic record."},
    {"fact_id": "fact-8d2da18f7c39ecb52757", "source_path": raw("kinbakutoday_eb9778bb44eeecfc.json"),
     "marker": "SM Play: You can play S&M (published 1972)", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "The bare year is redundant with the already retained, more informative question identifying SM Play as an early Japanese kinbaku tutorial book."},
    {"fact_id": "fact-c1df997a17d15ae6a8b0", "source_path": raw("kinbakutoday_f57559bbb4c8b826.json"),
     "marker": "include Uramado, which Nureki would later go on to edit",
     "manual_evidence": "After a brief association with Kitan Club in Osaka, Minomura moved to Tokyo and began a series of publications that would include Uramado, which Nureki would later go on to edit.",
     "decision": "edit",
     "question": "How does the source connect Minomura Kou and Nureki Chimuo through Uramado?",
     "answer": "Minomura began Uramado, which Nureki later edited.",
     "reason_code": "replace_publication_lookup_with_editorial_connection",
     "reason": "The edit turns a title lookup into a concise account of the publication and editorial link between the two historical figures."},
    {"fact_id": "fact-afb269c7fdc5968e1793", "source_path": raw("esinem_62d7dca7b38dbd4e.json"),
     "marker": "closed in January 2009", "decision": "drop",
     "reason_code": "contextless_or_time_bound_media_trivia",
     "reason": "A month-and-year closure lookup from a link-replacement blog post is less useful than the dataset's substantive print-history material."},
    {"fact_id": "fact-d596019f1c21d094ee06", "source_path": raw("kinbakutoday_e7f3e175c6e3bfd7.json"),
     "marker": "This, Yukimura explains, is aibunawa", "decision": "keep",
     "reason_code": "aibunawa_translation_context_complete",
     "reason": "The directly supported translation defines a recurring historical and stylistic term without teaching the source's risky application."},
    {"fact_id": "fact-d70379ce60d247a310e7", "source_path": raw("rope365_8ae9e3d93b31601b.json"),
     "marker": "Lark’s head", "decision": "keep",
     "reason_code": "foundational_hitch_alias_context_complete",
     "reason": "The alias is part of a compact foundational vocabulary entry that also defines the hitch's two-half-hitch structure."},
    {"fact_id": "fact-0aa98abb926464bc987f", "source_path": raw("rope365_6f46d5169ca32ec7.json"),
     "marker": "caressing style, also known as aibunawa", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "This reverse alias question duplicates the retained direct definition of aibunawa as caressing rope."},
    {"fact_id": "fact-3f30298063c10c1a2702", "source_path": raw("rope365_f2d9e825760af158.json"),
     "marker": "commonly used in Japanese martial art Hojōjutsu", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_person_tie_lookup",
     "reason": "The martial-art name is attached to a high-risk cuffed box-tie family but omits the source's nerve locations, fit, tension, monitoring, and release cautions."},
    {"fact_id": "fact-54b5e92e0ada0d0f492f", "source_path": raw("rope365_095aa0f0eea4c62c.json"),
     "marker": "cow hitch is technically two half hitches", "decision": "keep",
     "reason_code": "cow_hitch_structure_context_complete",
     "reason": "The answer is a basic knot-structure fact, and the question explicitly limits the claim to the cow hitch's topology."},
    {"fact_id": "fact-7fade041650a54ac24f3", "source_path": raw("kinbakutoday_d5d373e4a55ff204.json"),
     "marker": "native Japanese word is sarugutsuwa", "decision": "drop",
     "reason_code": "unsafe_or_airway_context_incomplete",
     "reason": "The terminology lookup is drawn from an interview that normalizes obstructive and nose-covering gags without a reliable airway, communication, or emergency-removal protocol."},
    {"fact_id": "fact-fee7de51d520ffac2394", "source_path": raw("rope365_15518f0912cce205.json"),
     "marker": "based on a design by Tifereth", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_person_lookup",
     "reason": "A designer-name lookup for a high-risk elbow position adds no shoulder-mobility, nerve, circulation, monitoring, or release context."},
    {"fact_id": "fact-fd929790183178649bcb", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["atx_empty_space"],
     "evidence": "The Empty Space: https://www.atxempty.space/space-schedule",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that this Austin event-calendar resource remain available for recommendations."},
    {"fact_id": "fact-36ce0e031716dd3cebff", "source_path": raw("kinbakutoday_5783480db327a3ba.json"),
     "marker": "Attitude toward “torment” (責め)", "decision": "drop",
     "reason_code": "semantic_duplicate",
     "reason": "The isolated kanji lookup is redundant with the retained seme question that includes the term's broader punishment, torment, pressure, accusation, and cruelty range."},
    {"fact_id": "fact-4cad0a397023824f5b94", "source_path": raw("esinem_0b1850ee9a40c337.json"),
     "marker": "it’s a danger zone, and it’s best to avoid upper cinches", "decision": "edit",
     "question": "What conclusion did the upper-limb surgeon quoted by Esinem reach about upper cinches?",
     "answer": "it’s a danger zone, and it’s best to avoid upper cinches",
     "reason_code": "replace_lineage_lookup_with_expert_safety_conclusion",
     "reason": "The edit replaces an incidental style attribution with the source's explicit expert warning about a region containing three major nerves."},
    {"fact_id": "fact-58d0899e951e3758bdbb", "source_path": raw("kinbakutoday_490a87ec78c3d64b.json"),
     "marker": "details their offenses for all the world to see", "decision": "edit",
     "question": "In the source’s 1878 illustration of monks publicly shamed as punishment, what did the nearby sign display?",
     "answer": "details their offenses for all the world to see",
     "reason_code": "replace_bare_book_year_with_sign_function",
     "reason": "The edit retains the dated historical illustration as context while asking how a public sign functioned in the punishment scene."},
    {"fact_id": "fact-7e567cc256ae27660f3f", "source_path": raw("wikipedia_a97e19abfe49afa9.json"),
     "marker": "was typically hemp in material", "decision": "keep",
     "reason_code": "source_qualified_honnawa_material",
     "reason": "The question preserves the source's 'typically' qualification and identifies the historical main-rope category rather than generalizing to modern bondage rope."},
    {"fact_id": "fact-fd5c96b72210af42fd05", "source_path": raw("rope365_b781bc1188743976.json"),
     "marker": "use hitches to connect the ropes together", "decision": "keep",
     "reason_code": "low_risk_rope_craft_technique_context_complete",
     "reason": "The suggestion concerns non-body rope crafting and gives a simple technique for joining lines in a spatial art exercise."},
    {"fact_id": "fact-3733f5fd6b66feb1b5ed", "source_path": raw("kinbakutoday_209cfdfa24ad7561.json"),
     "marker": "I saw the bondage photographs for ninety percent of Yoji Muku’s kinbaku pictures", "decision": "edit",
     "question": "How did Nureki describe the relationship between bondage photographs and Yoji Muku’s drawings?",
     "answer": "I saw the bondage photographs for ninety percent of Yoji Muku’s kinbaku pictures, and the art looks just like the photos.",
     "reason_code": "replace_artist_lookup_with_source_quoted_influence",
     "reason": "The edit replaces two-name recall with Nureki's concrete, source-quoted account of how strongly photographs informed Muku's drawings."},
    {"fact_id": "fact-9b9ba4f465c2a44a290f", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["tethered_together"],
     "evidence": "Tethered Together: https://tetheredtogether.net/",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that this national rope-convention URL remain memorized for recommendations."},
    {"fact_id": "fact-dff1d2131b0db9449d06", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["de_giotto_rope"],
     "evidence": "De Giotto Rope: https://degiottorope.com/",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that this natural-fiber rope-shop URL remain memorized for recommendations."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        247, 459, 606, 112, 291, 107, 98, 553, 103, 235,
        222, 228, 290, 86, 263, 37, 474, 244, 595, 99,
        312, 399, 586, 479, 482,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (3 if index < 8 else 4 if index < 23 else 5)
    for index, fact_id in enumerate(EXPECTED_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v22",
    "eligible_prior_keeps_before_reaudit_exclusion": 255,
    "secondary_eligible_prior_keeps": 205,
    "rows": 633,
    "sha256": "a990d08a9175a8b7976348c09c66e57fc502dd1b3ec59f18115d2db256e2da69",
    "v21_v22_reviewed_fact_ids_excluded": 50,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v22": 595,
    "active_after_this_tranche": 587,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 8,
    "new_edits_applied": 7,
    "output_rows": 625,
    "output_sha256": "efe176ecfd680754b4e87852bb8f22d9f99d52d2158f2aafd11ec081e7d2d45a",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 10,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 23):
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
    if len(ranked) != 205:
        raise ValueError(f"v23 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:25]) != EXPECTED_SELECTION:
        raise ValueError("v23 secondary selection drift")
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
        row["schema"] = "context-merit-audit-v23"
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
    paraphrase_fact_id = "fact-c1df997a17d15ae6a8b0"
    for row in curations:
        if row["fact_id"] == paraphrase_fact_id:
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = (
                "Condenses the source's single sentence into a grammatical "
                "subject-action statement while preserving its publication "
                "and editorial relationship."
            )
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v23"
    report["selection"].update({
        "active_rows": 633,
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 205,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v22 projection, and the fact was not reviewed "
                "again in v21 or v22"
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
