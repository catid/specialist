#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v49 projection rows in v50."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V49_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v49"
sys.path[:0] = [str(ROOT), str(V49_DIR)]
import build_context_merit_audit_v49 as previous

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v50.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v50.jsonl"
REPORT = OUT_DIR / "report_context_merit_v50.json"
REVIEWER = "codex-context-merit-audit-v50"
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
    for version in range(1, 50)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 50)
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-87951dd8d03450c6f602",
        "source_path": raw("rope365_15518f0912cce205.json"),
        "marker": "Warm up the arms, especially the shoulders",
        "decision": "drop",
        "reason_code": "redundant_elbow_warmup_fragment",
        "reason": (
            "The cumulative projection already contains a reviewed preparation answer "
            "that includes this warmup plus the more important sustainable-position check."
        ),
    },
    {
        "fact_id": "fact-9262d7a403e6f990b346",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": "<loc>https://rope-topia.com/portfolio-items/wicked-fast-bowline/</loc>",
        "decision": "keep",
        "allow_document_sha_mismatch": True,
        "reason_code": "owner_requested_wicked_fast_bowline_url",
        "reason": (
            "The sitemap-backed canonical tutorial URL preserves owner-requested "
            "resource metadata without inventing content from the gated page."
        ),
    },
    {
        "fact_id": "fact-c093a3dd0f041615ac43",
        "source_path": raw("kinbakutoday_463d41ffc7b6725a.json"),
        "marker": "four key principles for Kinbaku.  Control, Composure,  Intent, and Passion",
        "decision": "edit",
        "question": "Which four principles did Milla Reika say she formulated for her own kinbaku practice?",
        "answer": "control, composure, intent, and passion",
        "reason_code": "clarify_attribution_and_personal_scope",
        "reason": (
            "The edit identifies the speaker and makes clear that these were her own "
            "formulated practice principles, not universal rules from an unnamed blog."
        ),
    },
    {
        "fact_id": "fact-f271fc3eb63a3caa0e08",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": "<loc>https://rope-topia.com/portfolio-items/wet-treating-rope/</loc>",
        "decision": "keep",
        "allow_document_sha_mismatch": True,
        "reason_code": "owner_requested_wet_treating_rope_url",
        "reason": (
            "The canonical sitemap URL is useful owner-requested rope-care resource "
            "metadata and does not infer inaccessible article content."
        ),
    },
    {
        "fact_id": "fact-4f233aa2c03b91505209",
        "source_path": raw("wykd_a74fec63b0114fff.json"),
        "marker": "not necessarily the only correct usage",
        "decision": "keep",
        "reason_code": "attributed_terminology_nonexclusivity_caveat",
        "reason": (
            "The explicitly attributed caveat prevents the author's preferred term from "
            "being misrepresented as the sole correct terminology."
        ),
    },
    {
        "fact_id": "fact-520cc8f7fae5a8f2904d",
        "source_path": raw("anatomiestudio_9749de0eb1ff4ef3.json"),
        "marker": "No response equals no more scene",
        "decision": "keep",
        "reason_code": "positive_action_consent_stop_condition",
        "reason": (
            "The answer captures the decisive safety rule in the source's positive-action "
            "consent example: absent confirmation ends the scene."
        ),
    },
    {
        "fact_id": "fact-59847c1427eaee28d7de",
        "source_path": raw("anatomiestudio_144932682af9c846.json"),
        "marker": "intensive wash and spin cycle we wouldn’t recommend",
        "decision": "keep",
        "reason_code": "jute_machine_washing_warning",
        "reason": (
            "The answer clearly identifies the washing-machine treatment the studio "
            "advises against for jute rope."
        ),
    },
    {
        "fact_id": "fact-961251799c9b231ff6c1",
        "source_path": raw("rope_resources_v1/rope365__b602b6493b5eb6f55206.json"),
        "marker": "get clear communication and confirm consent",
        "decision": "keep",
        "reason_code": "power_dynamic_consent_confirmation",
        "reason": (
            "The answer provides direct consent guidance for relationships where D/s, "
            "teaching, or financial power can distort communication."
        ),
    },
    {
        "fact_id": "fact-e372ba5fbcb0dbc9d51c",
        "source_path": raw("anatomiestudio_27ecdd4d7c9a5560.json"),
        "marker": "touch people or their equipment (rope included)",
        "decision": "keep",
        "reason_code": "observer_scene_space_etiquette",
        "reason": (
            "Not touching participants or their equipment is concrete, useful scene-space "
            "etiquette for observers."
        ),
    },
    {
        "fact_id": "fact-2cb7ea98d3938f5ce55e",
        "source_path": raw("rope_resources_v1/rope365__b602b6493b5eb6f55206.json"),
        "marker": "their words, their breathing, their whole body",
        "decision": "keep",
        "reason_code": "multichannel_partner_monitoring",
        "reason": (
            "Monitoring words, breathing, and whole-body behavior is concise, practical "
            "guidance for communication during tying."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (127, 443, 125, 442, 42, 107, 444, 195, 278, 352))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v49",
    "direct_rows_without_prior_curation": 252,
    "eligible_unreviewed_direct_rows": 66,
    "prior_context_reviewed_direct_rows_excluded": 186,
    "rows": 547,
    "sha256": "e92dc20eec64faf1c49d2660520ec972261411a62eb4053de86cf4d67f31da2c",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v49": 509,
    "active_after_this_tranche": 508,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 1,
    "output_rows": 546,
    "output_sha256": "1d8517c1a7681f07e9381cb0ed88f3455324563d94184d5131b9969c68f79cc5",
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
    for version in range(1, 50):
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
    if len(ranked) != 66:
        raise ValueError(f"v50 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v50 selection drift")
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
        EVIDENCE_PATCH_MODULE.source_evidence = previous.previous.source_evidence
        yield
    finally:
        EVIDENCE_PATCH_MODULE.source_evidence = original_evidence
        CORE.ACTIVE_DATASET = original_active
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v49-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v49.jsonl"
        projected_report = Path(temp) / "projection-v49.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 547:
            raise ValueError("v49 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v49 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v50"
        row["review_pass"] = "first_context_merit_review_of_v49_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 547,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v50"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [{"path": str(path.relative_to(ROOT)),
                      "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
    }
    report["selection"].update({
        "active_rows": 547, "eligible_unreviewed_rows": 66,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0, "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v49 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v49"
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
        "extractive": 1, "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
