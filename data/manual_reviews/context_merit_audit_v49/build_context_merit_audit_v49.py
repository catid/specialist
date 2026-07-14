#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v48 projection rows in v49."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V48_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v48"
sys.path[:0] = [str(ROOT), str(V48_DIR)]
import build_context_merit_audit_v48 as previous

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v49.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v49.jsonl"
REPORT = OUT_DIR / "report_context_merit_v49.json"
REVIEWER = "codex-context-merit-audit-v49"
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
    for version in range(1, 49)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 49)
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-7d3c573dece9b1f178e2",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": "<loc>https://rope-topia.com/portfolio-items/joining-rope/</loc>",
        "decision": "keep",
        "allow_document_sha_mismatch": True,
        "reason_code": "owner_requested_joining_rope_resource_url",
        "reason": (
            "The sitemap-backed canonical tutorial URL is useful owner-requested "
            "resource metadata and makes no claim about the gated article body."
        ),
    },
    {
        "fact_id": "fact-9f46087b1fbe700e61f3",
        "source_path": raw("rope365_5fdb5e78c2471772.json"),
        "marker": "then retwist it and massage it to get it back into place",
        "decision": "keep",
        "reason_code": "practical_high_stranding_maintenance",
        "reason": (
            "The answer gives a concise corrective action for an intentionally "
            "created rope-maintenance problem."
        ),
    },
    {
        "fact_id": "fact-6df55b87852373a02db3",
        "source_path": ROOT / "sources" / "manual_facts" / "resource_group_c.jsonl",
        "marker": "The focus of the group is in-person peer to peer rope education",
        "decision": "keep",
        "reason_code": "owner_requested_austin_group_education_scope",
        "reason": (
            "The group's own description tells users what educational role to expect "
            "from this owner-requested local resource."
        ),
    },
    {
        "fact_id": "fact-da12e05feb5af6c4533b",
        "source_path": raw("kinbakutoday_5cea0bc70445dc65.json"),
        "marker": "seductive art of Japanese Bondage",
        "decision": "edit",
        "question": "Which book by Midori is mentioned as an early English reference?",
        "answer": "The Seductive Art of Japanese Bondage",
        "reason_code": "normalize_book_title_capitalization",
        "reason": (
            "Title capitalization is normalized while preserving the exact book "
            "named by the interviewee as an early English reference."
        ),
    },
    {
        "fact_id": "fact-947f495396a7edf09414",
        "source_path": raw("anatomiestudio_144932682af9c846.json"),
        "marker": "mark the ends with a certain colour",
        "decision": "keep",
        "reason_code": "visual_designated_rope_hygiene_marker",
        "reason": (
            "A visible end marker is practical hygiene guidance for distinguishing a "
            "partner- or role-specific rope from the rest of a kit."
        ),
    },
    {
        "fact_id": "fact-495c6aa066f884214063",
        "source_path": raw("rope_resources_v1/rope365__7b5d548036392d65fec7.json"),
        "marker": "Before trying a new technique, take the time to research the specific risks",
        "decision": "keep",
        "reason_code": "technique_specific_risk_research",
        "reason": (
            "Researching technique-specific risks before attempting a tie is direct, "
            "generalizable risk-mitigation advice."
        ),
    },
    {
        "fact_id": "fact-9702dff6060a2544685b",
        "source_path": raw("rope365_5fdb5e78c2471772.json"),
        "marker": "Compare the weight of the rope, burn speed, friction, smell",
        "decision": "drop",
        "reason_code": "underspecified_hardware_store_burn_exercise",
        "reason": (
            "The isolated list omits test procedures and fire/fume precautions, so it "
            "is not safe or sufficiently actionable equipment guidance by itself."
        ),
    },
    {
        "fact_id": "fact-2fcf9a51af80c7c418d6",
        "source_path": raw("anatomiestudio_9749de0eb1ff4ef3.json"),
        "marker": "combined with more precise language relating to body parts, sensations and urgency",
        "decision": "keep",
        "reason_code": "rope_safeword_precision",
        "reason": (
            "Body location, sensation, and urgency turn a general safeword into the "
            "specific information needed to respond safely during rope."
        ),
    },
    {
        "fact_id": "fact-a3fb0bf625fecebfa5e0",
        "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
        "marker": "a half hitch on the loop created by the first half hitch to lock the exit into place",
        "decision": "keep",
        "reason_code": "quick_release_locking_mechanism",
        "reason": (
            "The answer identifies the specific second hitch that locks the exit of "
            "the described quick-release structure."
        ),
    },
    {
        "fact_id": "fact-fcea3c04b3e8979d1e24",
        "source_path": raw("wykd_a74fec63b0114fff.json"),
        "marker": "a technique in its own right that specifically only ever uses one rope",
        "decision": "keep",
        "reason_code": "explicitly_attributed_ichinawa_technique_distinction",
        "reason": (
            "Unlike an isolated translation lookup, the attributed answer explains "
            "the author's substantive distinction between a named technique and a "
            "one-rope practice constraint."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (438, 66, 218, 426, 47, 186, 459, 264, 273, 95))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v48",
    "direct_rows_without_prior_curation": 254,
    "eligible_unreviewed_direct_rows": 76,
    "prior_context_reviewed_direct_rows_excluded": 178,
    "rows": 548,
    "sha256": "1c585357d18896cdd0264ce0502dfb600a2c2bf76f6c726e3b74b46d4902a8f7",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v48": 510,
    "active_after_this_tranche": 509,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 1,
    "output_rows": 547,
    "output_sha256": "e92dc20eec64faf1c49d2660520ec972261411a62eb4053de86cf4d67f31da2c",
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
    for version in range(1, 49):
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
    if len(ranked) != 76:
        raise ValueError(f"v49 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v49 selection drift")
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
    with tempfile.TemporaryDirectory(prefix=".v48-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v48.jsonl"
        projected_report = Path(temp) / "projection-v48.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 548:
            raise ValueError("v48 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v48 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v49"
        row["review_pass"] = "first_context_merit_review_of_v48_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 548,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v49"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [{"path": str(path.relative_to(ROOT)),
                      "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
    }
    report["selection"].update({
        "active_rows": 548, "eligible_unreviewed_rows": 76,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0, "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v48 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v48"
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
