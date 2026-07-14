#!/usr/bin/env python3
"""Deterministically re-audit the final weak-context prior keeps in v31."""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V30_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v30"
sys.path[:0] = [str(ROOT), str(V30_DIR)]
import build_context_merit_audit_v30 as previous

BASE = previous.BASE
CORE = previous.CORE

DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v31.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v31.jsonl"
REPORT = OUT_DIR / "report_context_merit_v31.json"
REVIEWER = "codex-context-merit-audit-v31"
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
    for version in range(1, 31)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 31)
)
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {"fact_id": "fact-aadae478a8a110b37947",
     "source_path": raw("rope365_0c46b0f988c7a825.json"),
     "marker": "alternatives that allow you to open “the wings” more",
     "decision": "edit",
     "question": "What does Rope365 suggest if a folded-arm position causes tingling in the little and ring fingers?",
     "answer": "experiment together with finding alternatives that allow you to open “the wings” more and compress your elbows less",
     "reason_code": "replace_anatomy_list_with_symptom_response",
     "reason": "The edit replaces a vulnerable-body-part list with the source's concrete response to a possible ulnar-nerve symptom."},
    {"fact_id": "fact-f6819a1cedb44d6048a2",
     "source_path": raw("rope365_a3b2e9e479b0c70f.json"),
     "marker": "Tie them in a daisy chain or put them in a pillowcase",
     "decision": "edit",
     "question": "How does Rope365 suggest preventing washable cotton or synthetic rope from tangling in a washing machine?",
     "answer": "Tie them in a daisy chain or put them in a pillowcase",
     "reason_code": "clarify_machine_wash_tangle_prevention",
     "reason": "The edit replaces the vague 'what process' wording with the exact washable materials, machine context, and practical goal."},
    {"fact_id": "fact-44480871b875f72e8c29",
     "source_path": raw("rope365_9a5c5810310fa0f0.json"),
     "marker": "letting the rope lead me",
     "decision": "drop",
     "reason_code": "underspecified_random_improvisation_phrase",
     "reason": "The phrase lookup promotes random frictions and unplanned direction without carrying forward the source's negotiation and placement precautions, which already survive separately."},
    {"fact_id": "fact-0717fb4cb44551293095",
     "source_path": raw("kinbakutoday_0d87b1ac1a49f2d4.json"),
     "marker": "indispensable because his writings preserve otherwise inaccessible knowledge",
     "decision": "edit",
     "question": "Why does the article describe Nureki’s writings as both indispensable and dangerous historical sources?",
     "answer": "indispensable because his writings preserve otherwise inaccessible knowledge about postwar SM magazine culture, but dangerous because his accounts are shaped by literary craft, memory, loyalty, inference, and retrospective storytelling",
     "reason_code": "replace_book_title_with_source_criticism",
     "reason": "The edit replaces a title lookup with the article's substantive explanation of both the value and limits of memoir as historical evidence."},
    {"fact_id": "fact-d0ab81fad8b39983bc97",
     "source_path": raw("wikipedia_5368de8bae5a78d8.json"),
     "marker": "about equal to the diameter of the rope",
     "decision": "edit",
     "question": "What width does The Ashley Book of Knots recommend for a whipping on a rope end?",
     "answer": "about equal to the diameter of the rope",
     "reason_code": "clarify_whipping_proportion_question",
     "reason": "The edit keeps the source-attributed finishing guideline while replacing an awkward relative-width question with direct wording."},
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(EXPECTED_SELECTION,
                                             (507, 307, 296, 253, 19))
}
SECONDARY_PRIOR_VERSIONS = {fact_id: 20 for fact_id in EXPECTED_SELECTION}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v30",
    "eligible_prior_keeps_before_reaudit_exclusion": 255,
    "secondary_eligible_prior_keeps": 5,
    "rows": 580,
    "sha256": "3b28bdbdc78503cf1beb497630581d83751982a00d206145003eb92def3099dd",
    "v21_v30_reviewed_fact_ids_excluded": 250,
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v30": 542,
    "active_after_this_tranche": 541,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 4,
    "output_rows": 579,
    "output_sha256": "0707cb2678ff2e7ba6f3eac1355eac2236d4217dcf163a7bf8802be3ff09106d",
    "prior_pending_addition_fact_ids_preserved": 38,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 0,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 31):
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
    if len(ranked) != 5:
        raise ValueError(f"v31 secondary candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked) != EXPECTED_SELECTION:
        raise ValueError("v31 secondary selection drift")
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
        prior_path = (DATA / "manual_reviews" / "context_merit_audit_v20" /
                      "context_merit_audit_v20.jsonl")
        row["schema"] = "context-merit-audit-v31"
        row["active_index"] = PROJECTED_ACTIVE_INDICES[fact_id]
        row["review_pass"] = "secondary_prior_keep_reaudit"
        row["prior_review"] = {
            "decision": "keep",
            "path": str(prior_path.relative_to(ROOT)),
            "sha256": file_sha256(prior_path),
            "version": 20,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v31"
    report["selection"].update({
        "active_rows": 580,
        "rows_selected": len(EXPECTED_SELECTION),
        "eligible_unreviewed_rows": 0,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0,
        "primary_fact_ids": [],
        "secondary_fact_ids": list(EXPECTED_SELECTION),
        "secondary_eligible_prior_keeps": 5,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "secondary_ranking": {
            "candidate_rule": (
                "a sole v1-v20 context-merit decision is keep, the fact "
                "survives the v30 projection, and the fact was not reviewed "
                "again in v21 through v30"
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
    report["audit"]["rows"] = len(audits)
    report["audit"]["sha256"] = file_sha256(AUDIT)
    report["new_pending_curation"]["sha256"] = file_sha256(CURATION)
    report["new_pending_curation"]["edit_support_types"] = {
        "extractive": 4, "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
