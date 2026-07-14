#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v46 projection rows in v47."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V46_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v46"
sys.path[:0] = [str(ROOT), str(V46_DIR)]
import build_context_merit_audit_v46 as previous

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v47.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v47.jsonl"
REPORT = OUT_DIR / "report_context_merit_v47.json"
REVIEWER = "codex-context-merit-audit-v47"
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
    for version in range(1, 47)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 47)
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-1de93de4783e8a1c41db",
        "source_path": raw("kinbakutoday_432c8adfc1abe686.json"),
        "marker": "If your fingers begin to feel tingly tell your rigger immediately",
        "decision": "keep",
        "reason_code": "immediate_tingling_communication",
        "reason": (
            "The source gives direct, time-sensitive safety guidance for a possible "
            "circulation or nerve symptom."
        ),
    },
    {
        "fact_id": "fact-55436d2b7b98db1e98d4",
        "source_path": raw("kinbakutoday_c364f23ce34ae761.json"),
        "marker": "narrow categorization as ‘sexual’ or ‘not sexual’",
        "decision": "drop",
        "reason_code": "redundant_binary_sexuality_label",
        "reason": (
            "v46 already replaces this same page's binary label with the more useful "
            "distinction among intimacy, arousal, and sexual acts."
        ),
    },
    {
        "fact_id": "fact-97f65c79582226307d96",
        "source_path": raw("kinbakutoday_73b16e835ab63cc2.json"),
        "marker": "It is a giving, rather than a taking",
        "decision": "keep",
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The stored answer removes the source's article from 'a giving' while "
            "preserving its exact giving-versus-taking contrast."
        ),
        "paraphrase_support_fragments": ("a giving, rather than a taking",),
        "reason_code": "attributed_partner_centered_rope_ethic",
        "reason": (
            "The explicitly attributed answer captures the article's partner-centered "
            "ethic of kindness without presenting an opinion as universal fact."
        ),
    },
    {
        "fact_id": "fact-00fe778028241519d8af",
        "source_path": raw("wykd_19d6a26116e26c70.json"),
        "marker": "that common factor is likely to be the person tying",
        "decision": "keep",
        "reason_code": "repeated_injury_accountability_signal",
        "reason": (
            "The answer provides a clear accountability signal when injuries recur "
            "across different partners and sessions."
        ),
    },
    {
        "fact_id": "fact-f4dc829c030824dc6d0a",
        "source_path": ROOT / "sources" / "manual_facts" / "resource_group_b.jsonl",
        "marker": "Hang a weight of about one pound in the center of the rope",
        "decision": "keep",
        "reason_code": "manufacturer_specific_jute_drying_weight",
        "reason": (
            "The question is explicitly limited to deGiotto jute and preserves a "
            "specific step in that manufacturer's complete drying procedure."
        ),
    },
    {
        "fact_id": "fact-3bc8480b6200731486c6",
        "source_path": raw("rope365_d9c48a4547717047.json"),
        "marker": "you can keep the knot loose",
        "decision": "keep",
        "reason_code": "gentle_long_term_natural_rope_storage",
        "reason": (
            "The loose-knot option is practical storage guidance for fragile natural "
            "rope that may kink under prolonged tight pressure."
        ),
    },
    {
        "fact_id": "fact-83f8ab37a6e9fbd8f0be",
        "source_path": raw("kinbakutoday_1dccbc876bee51b6.json"),
        "marker": "traditional hitotsumusubi overhand knots",
        "decision": "drop",
        "reason_code": "incidental_advanced_rope_end_terminology",
        "reason": (
            "The Japanese knot-name lookup is an incidental aside inside an advanced "
            "historical profile and lacks broadly useful construction or safety context."
        ),
    },
    {
        "fact_id": "fact-392013e296aa55be8770",
        "source_path": raw("rope365_f250f228cc370052.json"),
        "marker": "Column” refers to anything you can tie around",
        "decision": "keep",
        "reason_code": "foundational_single_column_definition",
        "reason": (
            "The definition explains a basic rope term and immediately grounds it with "
            "body and object examples."
        ),
    },
    {
        "fact_id": "fact-4cc9e4229c016704eeee",
        "source_path": raw("rope365_c73bc6fb66977a2d.json"),
        "marker": "It is best to discuss this beforehand, when emotions aren’t high",
        "decision": "keep",
        "reason_code": "useful_aftercare_timing",
        "reason": (
            "Discussing aftercare before heightened emotions is direct, reusable "
            "communication guidance."
        ),
    },
    {
        "fact_id": "fact-597fa7fd78d5fdcb4f10",
        "source_path": raw("kinbakutoday_432c8adfc1abe686.json"),
        "marker": "communicate and give feedback before, during and after bondage",
        "decision": "keep",
        "reason_code": "continuous_rope_bottom_feedback",
        "reason": (
            "The answer concisely preserves the guide's expectation that feedback "
            "continues through preparation, tying, and reflection."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (331, 280, 25, 400, 399, 50, 335, 104, 395, 401))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v46",
    "direct_rows_without_prior_curation": 257,
    "eligible_unreviewed_direct_rows": 96,
    "prior_context_reviewed_direct_rows_excluded": 161,
    "rows": 551,
    "sha256": "898af65291f91e80f9f2d8ee548eaecb8f065861cf5ba95f43f452891814a3b6",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v46": 513,
    "active_after_this_tranche": 511,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 2,
    "new_edits_applied": 0,
    "output_rows": 549,
    "output_sha256": "d2998dcacbe7dbc02ed8ccf6dbbab953fdc431eed3c9845d3051fa9badf4cb23",
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
    for version in range(1, 47):
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
    if len(ranked) != 96:
        raise ValueError(f"v47 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v47 selection drift")
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
        EVIDENCE_PATCH_MODULE.source_evidence = previous.source_evidence
        yield
    finally:
        EVIDENCE_PATCH_MODULE.source_evidence = original_evidence
        CORE.ACTIVE_DATASET = original_active
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v46-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v46.jsonl"
        projected_report = Path(temp) / "projection-v46.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 551:
            raise ValueError("v46 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v46 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v47"
        row["review_pass"] = "first_context_merit_review_of_v46_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 551,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
        spec = next(spec for spec in SPECS if spec["fact_id"] == row["fact_id"])
        if spec.get("support_type") == "manual_paraphrase":
            row["paraphrase_rationale"] = spec["paraphrase_rationale"]
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v47"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [{"path": str(path.relative_to(ROOT)),
                      "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
    }
    report["selection"].update({
        "active_rows": 551, "eligible_unreviewed_rows": 96,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0, "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v46 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v46"
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
        "extractive": 0, "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
