#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v50 projection rows in v51."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V50_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v50"
sys.path[:0] = [str(ROOT), str(V50_DIR)]
import build_context_merit_audit_v50 as previous

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v51.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v51.jsonl"
REPORT = OUT_DIR / "report_context_merit_v51.json"
REVIEWER = "codex-context-merit-audit-v51"
REVIEWED_AT = "2026-07-14"
RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl

CONTEXT_CURATIONS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"pending_curation_context_merit_v{version}.jsonl"
    for version in range(1, 51)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 51)
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-5b934d2edcb41402cdf8",
        "source_path": raw("rope365_441f9cc87ead6159.json"),
        "marker": "Learn the double column tie and the cinched friction",
        "decision": "keep",
        "reason_code": "foundational_cinch_curriculum_scope",
        "reason": (
            "The answer identifies the two foundational techniques in the named lesson "
            "and helps users navigate the Rope365 curriculum."
        ),
    },
    {
        "fact_id": "fact-7c33f617d6092f800235",
        "source_path": raw("kinbakutoday_73b16e835ab63cc2.json"),
        "marker": "It is not something given at one’s expense",
        "decision": "keep",
        "reason_code": "explicitly_attributed_kindness_sacrifice_distinction",
        "reason": (
            "The question clearly attributes a coherent distinction that supports the "
            "article's retained relational ethic."
        ),
    },
    {
        "fact_id": "fact-2528d6e7aae9f43eeb43",
        "source_path": raw("anatomiestudio_9749de0eb1ff4ef3.json"),
        "marker": "the less well you know someone, the more you should be communicating",
        "decision": "keep",
        "reason_code": "increase_communication_with_unfamiliar_partners",
        "reason": (
            "The answer gives direct, practical consent guidance for newer or less "
            "familiar rope partnerships."
        ),
    },
    {
        "fact_id": "fact-582bc37560761a271e19",
        "source_path": raw("rope_resources_v1/rope365__7b5d548036392d65fec7.json"),
        "marker": "aware of the risks involved in the type of play you will share together",
        "decision": "keep",
        "reason_code": "play_specific_risk_disclosure",
        "reason": (
            "Making a partner aware of the specific play's risks is essential informed-"
            "consent guidance."
        ),
    },
    {
        "fact_id": "fact-94b42b931282e211349e",
        "source_path": raw("rope_resources_v1/rope365__32dac30c0ebad4f0ad21.json"),
        "marker": "make sure it has good aeration to prevent drying and moulding and oil it before storage",
        "decision": "edit",
        "question": "How should natural rope be prepared for long-term storage according to Rope365?",
        "answer": "give it good aeration, and oil it before storage",
        "support_type": "manual_paraphrase",
        "paraphrase_support_fragments": (
            "make sure it has good aeration",
            "oil it before storage",
        ),
        "paraphrase_rationale": (
            "The answer retains both preparation actions while omitting the source's "
            "confusing claim that aeration itself prevents drying."
        ),
        "reason_code": "retain_storage_actions_remove_confusing_causal_clause",
        "reason": (
            "The edit preserves the actionable storage guidance without teaching an "
            "internally confusing moisture mechanism."
        ),
    },
    {
        "fact_id": "fact-6b6b6fcbc60678bd5d07",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": "<loc>https://rope-topia.com/newcomers-information/out-into-the-kink-community/</loc>",
        "decision": "keep",
        "allow_document_sha_mismatch": True,
        "reason_code": "owner_requested_kink_community_resource_url",
        "reason": (
            "The canonical sitemap URL preserves owner-requested newcomer resource "
            "metadata without inferring inaccessible article content."
        ),
    },
    {
        "fact_id": "fact-888004ffb3d5f1de2543",
        "source_path": raw("rope365_25f1b23eb40be00e.json"),
        "marker": "knots to tie at the end of the rope to keep it from fraying",
        "decision": "keep",
        "reason_code": "stopper_knot_fray_prevention",
        "reason": (
            "The answer states the practical maintenance purpose of end knots in a "
            "clear, reusable way."
        ),
    },
    {
        "fact_id": "fact-c6715c55bd2f15780891",
        "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
        "marker": "away from joints and nerves",
        "decision": "keep",
        "reason_code": "cuff_location_safety_check",
        "reason": (
            "Keeping cuffs away from joints and nerves is concise, practical placement "
            "guidance from the lesson's self-evaluation checklist."
        ),
    },
    {
        "fact_id": "fact-f03cdb915412c2d39d28",
        "source_path": raw("kinbakutoday_241155b848764148.json"),
        "marker": "communities developed almost exclusively as communities of learning and practice",
        "decision": "edit",
        "question": "According to the Rope Community & Culture essay, how did Western kinbaku communities initially develop?",
        "answer": "almost exclusively as communities of learning and practice",
        "reason_code": "attribute_western_community_history_thesis",
        "reason": (
            "The edit frames the broad East-West comparison as the essay author's thesis "
            "rather than an unqualified universal fact."
        ),
    },
    {
        "fact_id": "fact-2b0dac401eae9ee43a0e",
        "source_path": DATA / "rope_resource_manual_v1.jsonl",
        "marker": "It WILL tighten up unless you combine it with a half hitch",
        "decision": "keep",
        "reason_code": "vendor_larks_head_tightening_control",
        "reason": (
            "The attributed answer gives the tutorial's specific method for stopping a "
            "reverse-tension lark's head from tightening."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (487, 85, 99, 306, 100, 428, 521, 416, 109, 65))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v50",
    "direct_rows_without_prior_curation": 250,
    "eligible_unreviewed_direct_rows": 56,
    "prior_context_reviewed_direct_rows_excluded": 194,
    "rows": 546,
    "sha256": "1d8517c1a7681f07e9381cb0ed88f3455324563d94184d5131b9969c68f79cc5",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v50": 508,
    "active_after_this_tranche": 508,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 0,
    "new_edits_applied": 2,
    "output_rows": 546,
    "output_sha256": "48fbff4e40596e9ce544196db7dde7301acfb592ab63a1112997f4a5132ff56a",
    "prior_pending_addition_fact_ids_preserved": 37,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 8,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
    "validated_runs": 2,
}

PRODUCTION_INPUTS = previous.PRODUCTION_INPUTS
SEALED_EVAL_PATHS = previous.SEALED_EVAL_PATHS
PRIOR_PROJECTION_CURATIONS = (
    *ACTIVE_CURATIONS, QUALITY_MERIT_CURATION, TASUKI_CURATION,
    *CONTEXT_CURATIONS,
)


def prior_decision_artifacts() -> tuple[Path, ...]:
    paths = []
    for version in range(1, 51):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        paths.extend((directory / f"context_merit_audit_v{version}.jsonl",
                      directory / f"pending_curation_context_merit_v{version}.jsonl",
                      directory / f"report_context_merit_v{version}.json"))
    return tuple(paths)


def build_projection(output: Path, report: Path,
                     curations: tuple[Path, ...]) -> None:
    previous.build_projection(output, report, curations)


def prior_reviewed_fact_ids() -> set[str]:
    return {row["fact_id"] for path in CONTEXT_AUDITS for row in read_jsonl(path)}


def ranked_unreviewed_direct(rows: list[dict]) -> list[dict]:
    reviewed = prior_reviewed_fact_ids()
    candidates = []
    for index, row in enumerate(rows, 1):
        if row.get("curation") or row["fact_id"] in reviewed:
            continue
        features = CORE.risk_features(row)
        candidates.append((-features["risk_score"], features["question_tokens"],
                           features["answer_tokens"], row["fact_id"], index,
                           row, features))
    candidates.sort(key=lambda item: item[:4])
    ranked = [{"active_index": item[4], "row": item[5], "features": item[6]}
              for item in candidates]
    if len(ranked) != 56:
        raise ValueError(f"v51 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v51 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


@contextlib.contextmanager
def patched_base(projected_dataset: Path):
    replacements = {
        "OUT_DIR": OUT_DIR, "AUDIT": AUDIT, "CURATION": CURATION,
        "REPORT": REPORT, "REVIEWER": REVIEWER, "REVIEWED_AT": REVIEWED_AT,
        "CONTEXT_CURATIONS": CONTEXT_CURATIONS, "SPECS": SPECS,
        "EXPECTED_SELECTION": EXPECTED_SELECTION,
        "ISOLATED_PROJECTION": ISOLATED_PROJECTION,
    }
    originals = {name: getattr(BASE, name) for name in replacements}
    original_ranking = CORE.ranked_unreviewed
    original_active = CORE.ACTIVE_DATASET
    original_evidence = EVIDENCE_PATCH_MODULE.source_evidence
    try:
        for name, value in replacements.items():
            setattr(BASE, name, value)
        CORE.ranked_unreviewed = selected_ranked
        CORE.ACTIVE_DATASET = projected_dataset
        EVIDENCE_PATCH_MODULE.source_evidence = previous.previous.previous.source_evidence
        yield
    finally:
        EVIDENCE_PATCH_MODULE.source_evidence = original_evidence
        CORE.ACTIVE_DATASET = original_active
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v50-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v50.jsonl"
        projected_report = Path(temp) / "projection-v50.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 546:
            raise ValueError("v50 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v50 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v51"
        row["review_pass"] = "first_context_merit_review_of_v50_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 546,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
        spec = next(spec for spec in SPECS if spec["fact_id"] == row["fact_id"])
        if spec.get("support_type") == "manual_paraphrase":
            row["paraphrase_rationale"] = spec["paraphrase_rationale"]
    write_jsonl(AUDIT, audits)

    curations = read_jsonl(CURATION)
    for row in curations:
        spec = next(spec for spec in SPECS if spec["fact_id"] == row["fact_id"])
        if spec.get("support_type") == "manual_paraphrase":
            row["support_type"] = "manual_paraphrase"
            row["paraphrase_rationale"] = spec["paraphrase_rationale"]
    write_jsonl(CURATION, curations)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v51"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [{"path": str(path.relative_to(ROOT)),
                      "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
    }
    report["selection"].update({
        "active_rows": 546, "eligible_unreviewed_rows": 56,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0, "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v50 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v50"
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
        "extractive": 1, "manual_paraphrase": 1,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
