#!/usr/bin/env python3
"""Deterministically re-audit the next weak-context surviving keeps in v22."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V21_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v21"
sys.path[:0] = [str(ROOT), str(V21_DIR)]
import build_context_merit_audit_v21 as previous

# v19 is the stable wrapper around the v11 materializing main().  Bypassing
# v20/v21's custom selection wrappers keeps v22's selection isolated.
BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v22.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v22.jsonl"
REPORT = OUT_DIR / "report_context_merit_v22.json"
REVIEWER = "codex-context-merit-audit-v22"
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
    for version in range(1, 22)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 22)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-305e711b9258cb4cd9a2", "source_path": raw("kinbakutoday_9a72524da47a3539.json"),
     "marker": "our Semenawa favours the experimentation of suffering, not pain", "decision": "drop",
     "reason_code": "unsafe_or_consent_context_incomplete",
     "reason": "The label is extracted from a personal essay that encourages enduring escalating discomfort and delaying release, so the isolated style-name lookup omits essential stop and injury safeguards."},
    {"fact_id": "fact-7d40fbdc3504caa3efcf", "source_path": raw("esinem_dce6c59fa90ecae0.json"),
     "marker": "‘rigger’ as a word for the active party", "decision": "keep",
     "reason_code": "rigger_role_and_ambiguity_context_complete",
     "reason": "The question identifies both the active rope role and the term's potential ambiguity with theatrical rigging."},
    {"fact_id": "fact-bc11688b1543ec8b2d9f", "source_path": raw("kinbakutoday_f57559bbb4c8b826.json"),
     "marker": "find the deep meaning of kinbaku, to find kinbaku-bi", "decision": "keep",
     "reason_code": "kinbaku_bi_contextual_aesthetic_concept",
     "reason": "The source directly names kinbaku-bi as the deeper contextual meaning demonstrated through Nureki's aesthetic work rather than a knot or pattern."},
    {"fact_id": "fact-1a17204bc89767b3d93e", "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
     "marker": "I leave seme untranslated because no single English term", "decision": "keep",
     "reason_code": "seme_translation_range_context_complete",
     "reason": "The question carries the source's full translation range, making the retained Japanese term a meaningful language and media-history concept."},
    {"fact_id": "fact-ace591d75c7226db653a", "source_path": raw("kinbakutoday_4f9dec06e4af751a.json"),
     "marker": "onnen (怨念) does not simply name anger or resentment", "decision": "keep",
     "reason_code": "onnen_definition_context_complete",
     "reason": "The question preserves the source's precise distinction between ordinary resentment and a durable unresolved grievance in its ghost-story analysis."},
    {"fact_id": "fact-978096701201424beddd", "source_path": raw("kinbakutoday_a358fd398f91040a.json"),
     "marker": "Kinbaku means “tight binding”", "decision": "keep",
     "reason_code": "core_kinbaku_translation",
     "reason": "The literal translation is a concise, reusable definition of a core domain term."},
    {"fact_id": "fact-cffa44c03400fc16f70f", "source_path": raw("esinem_f2dfde25be14a7a8.json"),
     "marker": "profoundly influenced by the work of ukiyoe artist Tsukioka Yoshitoshi", "decision": "keep",
     "reason_code": "source_attributed_kinbaku_art_history",
     "reason": "The named influence is explicitly source-attributed and helps connect Itoh Seiu's work to an earlier visual-art lineage."},
    {"fact_id": "fact-12a26a2774ec26f80047", "source_path": raw("kinbakutoday_2265a4f9ae40d83a.json"),
     "marker": "would later start Uramado in 1955", "decision": "edit",
     "question": "What publication did Suma Toshiyuki start in 1955?",
     "answer": "Uramado",
     "reason_code": "replace_bare_year_with_publication_history",
     "reason": "The edit makes the historically relevant publication the answer while retaining its date and founder in the question."},
    {"fact_id": "fact-3a4869e89e813235976a", "source_path": raw("wikipedia_2151448295a2af9b.json"),
     "marker": "femdom (short for female dominance)", "decision": "edit",
     "question": "What does “femdom” mean in BDSM terminology?",
     "answer": "female dominance",
     "reason_code": "replace_reverse_abbreviation_lookup_with_definition",
     "reason": "The edit asks for the term's meaning directly instead of asking for a short form that is already implicit in the answer."},
    {"fact_id": "fact-501aa559677afe168fad", "source_path": raw("esinem_5c862cbce9ff02bd.json"),
     "marker": "tenugui, which are the cotton Japanese washcloths", "decision": "drop",
     "reason_code": "volatile_or_promotional_product_trivia",
     "reason": "The material lookup comes from a dated shop-stock announcement and adds little to rope selection, care, consent, or safety guidance."},
    {"fact_id": "fact-5ab3eb30b97381b1fbe4", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["house_of_bound_tutorials"],
     "evidence": "Lief Bound Tutorials: https://www.houseofbound.com/tutorials",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that this exact purchase link remain available for rope-resource recommendations."},
    {"fact_id": "fact-3c85f97bbd348e7d84e0", "source_path": raw("kinbakutoday_4f9dec06e4af751a.json"),
     "marker": "Fujimi Iku was, in fact, one of the many additional names that Nureki Chimuo used", "decision": "keep",
     "reason_code": "pseudonymous_authorship_history_context_complete",
     "reason": "The identification resolves authorship in a substantive discussion of pseudonyms, censorship, and archival provenance."},
    {"fact_id": "fact-621763f7f46d14be96dc", "source_path": raw("rope365_7fe5be31cb8dc67e.json"),
     "marker": "draws it’s strength from a bowline structure", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_technical_claim",
     "reason": "An isolated structural claim about a load-bearing rope join omits inspection, loading, material, tail, backup, and failure conditions and should not serve as standalone technique guidance."},
    {"fact_id": "fact-8d6fa405856a471d80eb", "source_path": raw("rope365_682937f92222bf87.json"),
     "marker": "Starting with Osada Eikichi in the 1960s onward", "decision": "edit",
     "question": "Whom does Rope365 credit with developing rope performance as its own art form from the 1960s onward?",
     "answer": "Osada Eikichi",
     "reason_code": "qualify_performance_history_attribution",
     "reason": "The edit preserves the history fact while clearly attributing the claim to Rope365 instead of presenting a broad first-person assertion as uncontested."},
    {"fact_id": "fact-7c645dabc1427a6e1e14", "source_path": raw("kinbakutoday_edba1220873364c8.json"),
     "marker": "Initially a battlefield technique", "decision": "drop",
     "reason_code": "source_reliability_or_overgeneralized_history",
     "reason": "The uncited gallery introduction compresses varied hojojutsu histories into a single battlefield-origin claim and is too broad to retain as settled fact."},
    {"fact_id": "fact-ba0bcde9e13c55a82239", "source_path": raw("esinem_77368ccdc66acad0.json"),
     "marker": "Rope suspension falls under the term “edge play” because it is inherently dangerous", "decision": "keep",
     "reason_code": "suspension_risk_term_context_complete",
     "reason": "The question retains the source's explicit danger rationale, and the surrounding evidence names injury mechanisms and shared responsibility."},
    {"fact_id": "fact-001d2e72ae8f7a500c26", "source_path": raw("rope365_5c6e72d053d47d66.json"),
     "marker": "Mynx’s interpretation of the box tie is influenced by her teacher", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete_person_lookup",
     "reason": "A teacher-name lookup attached to a high-risk box-tie pattern contributes no anatomy, monitoring, loading, adaptation, or emergency-release guidance."},
    {"fact_id": "fact-2125ce90d1d75e5ca31d", "source_path": raw("rope365_30eaedb9cb1e6d5b.json"),
     "marker": "several of the same length, usually between 7-9 meters or 30’", "decision": "edit",
     "question": "What rope-kit approach does Rope365 describe as popular in Japanese-style and connective bondage?",
     "answer": "several of the same length, usually between 7-9 meters or 30’",
     "reason_code": "replace_style_lookup_with_practical_kit_approach",
     "reason": "The edit makes the practical same-length kit convention the answer and explicitly attributes a preference that varies by style and practitioner."},
    {"fact_id": "fact-27c1ea4d38aa44273817", "source_path": raw("kinbakutoday_e2c96f38ced1cb5d.json"),
     "marker": "due in large part to the influence of Minomura Kou", "decision": "edit",
     "question": "Why does the source call the 1972–1985 SM Collector one of the most important second-wave magazines?",
     "answer": "due in large part to the influence of Minomura Kou",
     "reason_code": "replace_title_lookup_with_historical_significance",
     "reason": "The edit keeps the title and dates as context while asking for the source's substantive reason for the magazine's historical importance."},
    {"fact_id": "fact-39f074e28d0a4cd40ea3", "source_path": raw("kinbakutoday_4330ca3210d81512.json"),
     "marker": "photo series Utsukushiki Imashime", "decision": "drop",
     "reason_code": "contextless_or_low_value_media_title",
     "reason": "A title lookup for one model's staged-photo series is minor archival trivia and teaches neither the article's safety lesson nor broader media history."},
    {"fact_id": "fact-4a5cebf4e405af18a2dd", "source_path": raw("rope365_f43c9fde09431a5f.json"),
     "marker": "Teardrop Harness", "decision": "keep",
     "reason_code": "woven_harness_design_term_context_complete",
     "reason": "The source directly defines the named design by its use of weaving for an aesthetic and functional chest-harness structure."},
    {"fact_id": "fact-d04078c171abafc3eb69", "source_path": raw("kinbakutoday_6f1f3d70caa0e6dc.json"),
     "marker": "‘Harder or softer?’", "decision": "drop",
     "reason_code": "contextless_or_personal_anecdote",
     "reason": "A writer's personal erotic catchphrase is not a general consent protocol and is weaker than explicit negotiation, check-in, and stop guidance elsewhere in the dataset."},
    {"fact_id": "fact-ada66dade9582072e166", "source_path": raw("kinbakutoday_78b58fbf1524edc7.json"),
     "marker": "East meets West bondage styles", "decision": "edit",
     "question": "How did Ken Marcus describe the concept behind the California Club collaboration between Takeshi Nagaike and Ernest Greene?",
     "answer": "East meets West bondage styles",
     "reason_code": "replace_book_title_lookup_with_exchange_concept",
     "reason": "The edit replaces a bare title quiz with the collaboration's source-quoted concept and keeps both participants in context."},
    {"fact_id": "fact-d5e941391d0ebc01a800", "source_path": raw("kinbakutoday_fd0d7ba4b6589765.json"),
     "marker": "a way to distinguish himself from Nureki and provide an alternative approach to rope that would give him a signature", "decision": "edit",
     "question": "Why does the source say Yukimura developed aibunawa after initially using a harder seme style?",
     "answer": "to distinguish himself from Nureki and provide an alternative approach to rope that would give him a signature",
     "reason_code": "replace_artist_lookup_with_style_development_reason",
     "reason": "The edit preserves the historically useful explanation for aibunawa's development instead of testing one artist's name."},
    {"fact_id": "fact-e39c053ace018b31331b", "source_path": raw("kinbakutoday_7113a15b5e5e3aa3.json"),
     "marker": "it’s called Hashira Shibari", "decision": "keep",
     "reason_code": "hashira_shibari_term_context_complete",
     "reason": "The question defines the vertical-post setting and asks for its directly supported Japanese rope term without teaching a construction or load-bearing step."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        259, 416, 392, 414, 381, 198, 591, 104, 271, 314,
        495, 558, 399, 614, 463, 390, 611, 401, 254, 464,
        258, 140, 250, 563, 274,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (2 if index < 15 else 3)
    for index, fact_id in enumerate(EXPECTED_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v21",
    "eligible_prior_keeps_before_v21_exclusion": 255,
    "secondary_eligible_prior_keeps": 230,
    "rows": 640,
    "sha256": "f5cff736fba9cd45707b86f0462a9d8e4dfad5740407e92dd6daadec6813b453",
    "v21_reviewed_fact_ids_excluded": 25,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v21": 602,
    "active_after_this_tranche": 595,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 7,
    "new_edits_applied": 7,
    "output_rows": 633,
    "output_sha256": "a990d08a9175a8b7976348c09c66e57fc502dd1b3ec59f18115d2db256e2da69",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 11,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    """Return the byte-pinned audit, curation, and report for every prior tranche."""
    paths = []
    for version in range(1, 22):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        paths.extend((
            directory / f"context_merit_audit_v{version}.jsonl",
            directory / f"pending_curation_context_merit_v{version}.jsonl",
            directory / f"report_context_merit_v{version}.json",
        ))
    return tuple(paths)


def secondary_ranked(rows: list[dict]) -> list[dict]:
    """Rank unre-reviewed v1-v20 keeps surviving the cumulative v21 projection."""
    by_id = {row["fact_id"]: (index, row)
             for index, row in enumerate(rows, 1)}
    occurrences: dict[str, int] = {}
    prior_keeps: dict[str, int] = {}
    # v1-v20 contain the original review population.  v21 is consulted only
    # as the explicit exclusion ledger, so its five first-pass keeps cannot
    # accidentally enter this secondary re-review pool.
    for version, path in enumerate(CONTEXT_AUDITS[:-1], 1):
        for audit in read_jsonl(path):
            fact_id = audit["fact_id"]
            occurrences[fact_id] = occurrences.get(fact_id, 0) + 1
            if audit["decision"] == "keep":
                prior_keeps[fact_id] = version
    v21_reviewed = {row["fact_id"] for row in read_jsonl(CONTEXT_AUDITS[-1])}
    candidates = []
    for fact_id, version in prior_keeps.items():
        if (occurrences[fact_id] != 1 or fact_id not in by_id or
                fact_id in v21_reviewed):
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
    if len(ranked) != 230:
        raise ValueError(f"v22 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:25]) != EXPECTED_SELECTION:
        raise ValueError("v22 secondary selection drift")
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
        row["schema"] = "context-merit-audit-v22"
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
    report["schema"] = "context-merit-audit-report-v22"
    report["selection"].update({
        "active_rows": 640,
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 230,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v21 projection, and the fact was not reviewed "
                "again in v21"
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
