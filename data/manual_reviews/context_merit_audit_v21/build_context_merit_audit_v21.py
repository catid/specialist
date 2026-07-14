#!/usr/bin/env python3
"""Audit final first-pass rows plus deterministic keep re-audit tranche v21."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V20_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v20"
sys.path[:0] = [str(ROOT), str(V20_DIR)]
import build_context_merit_audit_v20 as previous

BASE = previous.previous

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v21.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v21.jsonl"
REPORT = OUT_DIR / "report_context_merit_v21.json"
REVIEWER = "codex-context-merit-audit-v21"
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
    for version in range(1, 21)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 21)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl

# v11 owns the materializing main() and ranking function used through wrappers.
CORE = (previous.previous.previous.previous.previous.previous.previous
        .previous.previous.previous)
ORIGINAL_CORE_RANKING = CORE.ranked_unreviewed


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-8eabb652976366dbf7df", "source_path": raw("esinem_a337d2fe39281aa2.json"),
     "marker": "Speed, tension, timing and the energy the rigger projects", "decision": "edit",
     "question": "Which four elements does Esinem describe as the ‘how’ of good rope?",
     "answer": "Speed, tension, timing and the energy the rigger projects",
     "reason_code": "clarify_four_rope_handling_elements",
     "reason": "The edit removes an indirect analogy and directly asks for the four handling and interaction elements the source names."},
    {"fact_id": "fact-4c2485ee552804986ce9", "source_path": raw("kinbakutoday_78b58fbf1524edc7.json"),
     "marker": "Rope of the Rising Sun and Bondage of the Rising Sun", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "Two adult-video titles are minor media trivia compared with the source's substantive account of early U.S.–Japanese rope exchange."},
    {"fact_id": "fact-a54c90c5b9fb003b1d0b", "source_path": raw("rope365_9a5c5810310fa0f0.json"),
     "marker": "paradox of patterns and presence", "decision": "keep",
     "reason_code": "patterns_presence_learning_concept_context_complete",
     "reason": "The question defines a useful learning tension between recalling patterns and remaining attentive to one's partner."},
    {"fact_id": "fact-930219c67b5a7befbc47", "source_path": raw("esinem_502c90cb43cc9dec.json"),
     "marker": "Midori’s ‘Seductive Art of Japanese Bondage’", "decision": "drop",
     "reason_code": "contextless_or_time_bound_resource",
     "reason": "A personal recollection of the closest English guide available about twenty years earlier is dated and weaker than the owner-curated current resource directory."},
    {"fact_id": "fact-fc75e65fe01974dba7d2", "source_path": raw("rope365_42818bd00e645bda.json"),
     "marker": "Waki Zarashi 脇晒し (exposed armpit) or Usagi ウサギ (rabbit)", "decision": "keep",
     "reason_code": "bunny_ears_terms_context_complete",
     "reason": "The question includes the defining arm position and asks for the two source-attributed Japanese names rather than teaching the tie as a standalone procedure."},
    {"fact_id": "fact-1149b6fa977f50a6331c", "source_path": raw("kinbakutoday_b776fccd348e2538.json"),
     "marker": "further developed by Nureki’s deshi, Naka Akira",
     "manual_evidence": "This style has been further developed by Nureki’s deshi, Naka Akira, who has also worked closely with Sugiura Norio, in the evolution of his semenawa style.",
     "decision": "edit",
     "question": "How does the source connect Naka Akira to Nureki and Sugiura Norio?",
     "answer": "It describes Naka as Nureki’s deshi who further developed the style and worked closely with Sugiura Norio.",
     "reason_code": "replace_person_lookup_with_lineage_contribution",
     "reason": "The edit turns a bare person lookup into a concise, source-grounded account of apprenticeship, development, and collaboration."},
    {"fact_id": "fact-f29a8c4521af38a94e7b", "source_path": raw("kinbakutoday_fbda0ee2dffbc811.json"),
     "marker": "influence of Minomura Kou", "decision": "drop",
     "reason_code": "contextless_or_personal_anecdote",
     "reason": "A memorial's brief influence-name lookup gives no technique, evidence, or explanation of the claimed influence."},
    {"fact_id": "fact-d51f978bc64b51b4c65e", "source_path": raw("rope365_682937f92222bf87.json"),
     "marker": "The History & Myths of Japanese Bondage by Midori", "decision": "keep",
     "reason_code": "useful_named_history_resource",
     "reason": "The author lookup identifies a specifically named history resource in Rope365's reviewed reading list."},
    {"fact_id": "fact-0a49ad482749ece6643a", "source_path": raw("esinem_05356ae92c0e84da.json"),
     "marker": "antecedent, hojojutsu", "decision": "drop",
     "reason_code": "source_error_or_overstated_origin_claim",
     "reason": "The claimed hojojutsu ancestry is asserted without evidence and conflicts with stronger retained sources that explicitly caution against this simple origin story."},
    {"fact_id": "fact-e41fdaf9f974e31e9179", "source_path": raw("kinbakutoday_43095aa305e35a41.json"),
     "marker": "ask if the ukete is really ok or not", "decision": "edit",
     "question": "What should the shibarite do when they feel something may be wrong?",
     "answer": "ask if the ukete is really ok or not",
     "reason_code": "replace_role_lookup_with_safety_action",
     "reason": "The edit replaces a role-name quiz with the source's actionable instruction to check directly with the person being tied."},
    {"fact_id": "fact-789b90e4c19cf97de3a7", "source_path": raw("kinbakutoday_454370c8a4f42708.json"),
     "marker": "model Junko Aoki from around 1964", "decision": "drop",
     "reason_code": "out_of_domain_or_personal_trivia",
     "reason": "A performer's model-name lookup is minor personal trivia within a much larger oral-history interview."},
    {"fact_id": "fact-45ad14e9dd7ee2526c24", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "How To Tie A Boxtie (TK) by Bondage Tuition", "decision": "drop",
     "reason_code": "contextless_or_low_value",
     "reason": "An author-name lookup from a long uncurated video-reference list is less reliable and useful than the owner-requested instructional resources."},
    {"fact_id": "fact-e96b48731c29761bc94d", "source_path": raw("wikipedia_093ebd176b6adaaf.json"),
     "marker": "According to David Stein, the man who coined", "decision": "keep",
     "reason_code": "consent_framework_history_context_complete",
     "reason": "The question fully identifies the SSC phrase and organization, preserving a useful source-attributed fact about consent-framework history."},
    {"fact_id": "fact-7f44ac0d2f1de5926b66", "source_path": raw("rope365_1616ffce57d993f3.json"),
     "marker": "their teacher Akechi Denki", "decision": "drop",
     "reason_code": "unsafe_or_context_incomplete",
     "reason": "A designer-lineage name attached to a high-risk box-tie pattern adds little without the source's anatomy, monitoring, tension, and release constraints."},
    {"fact_id": "fact-ae00d3e5d00d8fe80feb", "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
     "marker": "a ventriloquized interiority", "decision": "keep",
     "reason_code": "critical_media_history_concept_context_complete",
     "reason": "The question contains the complete definition of a distinctive analytical concept used to examine gendered authorship in early SM magazines."},
    {"fact_id": "fact-61797044f6aa991a119a", "source_path": raw("kinbakutoday_3376e34b04fe0fb1.json"),
     "marker": "Founded in 1983, the production company played an vital role",
     "manual_evidence": "Cinemagic grew to one of the largest and most important rope and SM studios in the 1980s. Founded in 1983, the production company played an vital role in the growing SM and rope bondage market in Japan.",
     "decision": "edit",
     "question": "What role did CineMagic, founded in 1983, play in Japanese SM and rope-bondage media?",
     "answer": "It played a vital role in the growing SM and rope-bondage market in Japan.",
     "reason_code": "replace_bare_year_with_media_significance",
     "reason": "The edit preserves the year as context while making CineMagic's historical media role the answer and correcting the source's article."},
    {"fact_id": "fact-7be09ff3273023906b25", "source_path": raw("kinbakutoday_eb9778bb44eeecfc.json"),
     "marker": "SM Play: You can play S&M (published 1972)", "decision": "edit",
     "question": "What 1972 publication does the source describe as Japan’s first kinbaku how-to book?",
     "answer": "SM Play: You can play S&M",
     "reason_code": "replace_magazine_lookup_with_first_tutorial_book",
     "reason": "The edit replaces an incidental magazine credit with the source's more important identification of an early kinbaku tutorial book."},
    {"fact_id": "fact-c06bbd52eab400cf8bf0", "source_path": raw("kinbakutoday_f57559bbb4c8b826.json"),
     "marker": "Kinbiken, is a contraction of Kinbakubi kenkyūkai", "decision": "keep",
     "reason_code": "historical_study_group_term_context_complete",
     "reason": "The expansion and contraction identify Nureki's named Beautiful Bondage Study Group within a substantive historical discussion."},
    {"fact_id": "fact-77971e5d6be6742d6b54", "source_path": raw("kinbakutoday_938724eb415ca5c0.json"),
     "marker": "Nihon keibatsu fūzoku zukan 日本刑罰風俗図史 in 1948", "decision": "edit",
     "question": "What 1948 volume by Itoh Seiyu does the source discuss in relation to Edo torture-rope imagery?",
     "answer": "Nihon keibatsu fūzoku zukan 日本刑罰風俗図史",
     "reason_code": "replace_bare_year_with_historical_volume",
     "reason": "The edit makes the identifiable historical volume the answer while retaining its year and analytical context in the question."},
    {"fact_id": "fact-109571ca967c263f9444", "source_path": RESOURCE_MANIFEST,
     "resource_ids": ["austin_rope_slingers"],
     "evidence": "Austin Rope Slingers: https://www.austinropeslingers.com/",
     "decision": "keep", "reason_code": "owner_requested_resource_directory",
     "reason": "The project owner explicitly requested that this exact Austin rope-community resource remain memorized for recommendations."},
    {"fact_id": "fact-019df482b79d13db9a0f", "source_path": raw("esinem_d00f706084758774.json"),
     "marker": "Adding lubrication in the form of wax reduced friction and extended life", "decision": "edit",
     "question": "What did Esinem’s limited natural-fiber friction tests report after wax was added?",
     "answer": "reduced friction and extended life",
     "reason_code": "qualify_non_scientific_wax_test_result",
     "reason": "The edit states the reported effect while explicitly retaining the source's warning that the small tests were not scientifically conclusive."},
    {"fact_id": "fact-426bc3d2c6101548619f", "source_path": raw("kinbakutoday_d4dcb268cb41c5e4.json"),
     "marker": "What I call 悦虐, “ecstatic cruelty,” means this", "decision": "keep",
     "reason_code": "historical_source_term_context_complete",
     "reason": "The question provides the English gloss and narrator attribution for a distinctive term analyzed in its early-magazine context."},
    {"fact_id": "fact-65c43946e82ed5b6e3a2", "source_path": raw("esinem_9a5aab43708932b3.json"),
     "marker": "‘ingredients’ in shibari, i.e. the components that are repeated in ties", "decision": "keep",
     "reason_code": "teaching_framework_term_context_complete",
     "reason": "The question carries the complete source definition of ingredients as recurring tie components in a transferable construction framework."},
    {"fact_id": "fact-9d2d0301b00637f79be6", "source_path": raw("kinbakutoday_5a6e15f57e52a345.json"),
     "marker": "started his periodical SM theatre show in Tokyo in 1976", "decision": "edit",
     "question": "What did Tamai Keiyuu establish in Tokyo in 1976?",
     "answer": "his periodical SM theatre show",
     "reason_code": "replace_bare_year_with_performance_history_event",
     "reason": "The edit retains the year but makes the historically significant public theatre development the answer."},
    {"fact_id": "fact-eb4cd9947c4f82a880bf", "source_path": raw("esinem_2b762f05bc1bf364.json"),
     "marker": "training in Aikido from a very young age", "decision": "drop",
     "reason_code": "volatile_or_promotional_person_trivia",
     "reason": "The martial-art name is biographical promotion in a dated commercial tour post whose broader technical claims are not independently supported."},
)

PRIMARY_SELECTION = tuple(spec["fact_id"] for spec in SPECS[:5])
SECONDARY_SELECTION = tuple(spec["fact_id"] for spec in SPECS[5:])
EXPECTED_SELECTION = PRIMARY_SELECTION + SECONDARY_SELECTION
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION, (
        208, 130, 31, 465, 236,
        613, 623, 611, 249, 418, 622, 612, 598, 617, 336,
        97, 570, 232, 95, 486, 319, 298, 391, 106, 546,
    ))
}
SECONDARY_PRIOR_VERSIONS = {
    fact_id: (1 if index < 17 else 2)
    for index, fact_id in enumerate(SECONDARY_SELECTION)
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v20",
    "primary_eligible_unreviewed_rows": 5,
    "secondary_eligible_prior_keeps": 250,
    "excluded_active_review_provenance": 308,
    "excluded_ledger_fact_ids": 1545,
    "rows": 648,
    "sha256": "46cf30a0e49a9daaf874462e20448d52f43f73b551b611e5734a52b09065349a",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v20": 610,
    "active_after_this_tranche": 602,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 8,
    "new_edits_applied": 8,
    "output_rows": 640,
    "output_sha256": "f5cff736fba9cd45707b86f0462a9d8e4dfad5740407e92dd6daadec6813b453",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 9,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def secondary_ranked(rows: list[dict]) -> list[dict]:
    by_id = {row["fact_id"]: (index, row)
             for index, row in enumerate(rows, 1)}
    occurrences: dict[str, int] = {}
    prior_keeps: dict[str, int] = {}
    for version, path in enumerate(CONTEXT_AUDITS, 1):
        for audit in read_jsonl(path):
            fact_id = audit["fact_id"]
            occurrences[fact_id] = occurrences.get(fact_id, 0) + 1
            if audit["decision"] == "keep":
                prior_keeps[fact_id] = version
    candidates = []
    for fact_id, version in prior_keeps.items():
        if occurrences[fact_id] != 1 or fact_id not in by_id:
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
    if len(ranked) != 250:
        raise ValueError(f"v21 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:20]) != SECONDARY_SELECTION:
        raise ValueError("v21 secondary selection drift")
    return ranked


def mixed_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    primary_ranked, excluded, provenance = ORIGINAL_CORE_RANKING(rows)
    if tuple(item["row"]["fact_id"] for item in primary_ranked[:5]) != PRIMARY_SELECTION:
        raise ValueError("v21 primary selection drift")
    return primary_ranked[:5] + secondary_ranked(rows)[:20], excluded, provenance


@contextlib.contextmanager
def patched_previous():
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
        CORE.ranked_unreviewed = mixed_ranked
        yield
    finally:
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def ranked_unreviewed(rows: list[dict]) -> tuple[list[dict], int, int]:
    with patched_previous():
        return BASE.ranked_unreviewed(rows)


def main() -> None:
    with patched_previous():
        BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        fact_id = row["fact_id"]
        row["schema"] = "context-merit-audit-v21"
        row["active_index"] = PROJECTED_ACTIVE_INDICES[fact_id]
        if fact_id in SECONDARY_SELECTION:
            version = SECONDARY_PRIOR_VERSIONS[fact_id]
            prior_path = (DATA / "manual_reviews" /
                          f"context_merit_audit_v{version}" /
                          f"context_merit_audit_v{version}.jsonl")
            row["review_pass"] = "secondary_prior_keep_reaudit"
            row["prior_review"] = {
                "decision": "keep",
                "path": str(prior_path.relative_to(ROOT)),
                "sha256": file_sha256(prior_path),
                "version": version,
            }
        else:
            row["review_pass"] = "primary_first_pass"
    write_jsonl(AUDIT, audits)

    curations = read_jsonl(CURATION)
    rationales = {
        "fact-1149b6fa977f50a6331c": (
            "Condenses the source's passive sentence into a grammatical account "
            "of the same apprenticeship, development, and collaboration."
        ),
        "fact-61797044f6aa991a119a": (
            "Corrects ‘an vital’ to ‘a vital’ and turns the source's historical "
            "statement into a complete sentence without changing its claim."
        ),
    }
    for row in curations:
        if row["fact_id"] in rationales:
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = rationales[row["fact_id"]]
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v21"
    report["selection"].update({
        "active_rows": 648,
        "eligible_unreviewed_rows": 5,
        "excluded_active_review_provenance": 308,
        "excluded_ledger_fact_ids": 1545,
        "primary_fact_ids": list(PRIMARY_SELECTION),
        "secondary_fact_ids": list(SECONDARY_SELECTION),
        "secondary_eligible_prior_keeps": 250,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "prior context-merit decision is keep, fact remains in the "
                "v20 projection, and fact has no earlier re-review"
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
        "extractive": 6, "manual_paraphrase": 2,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
