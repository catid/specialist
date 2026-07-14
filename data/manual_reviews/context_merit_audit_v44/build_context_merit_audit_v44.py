#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v43 projection rows in v44."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V43_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v43"
sys.path[:0] = [str(ROOT), str(V43_DIR)]
import build_context_merit_audit_v43 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v44.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v44.jsonl"
REPORT = OUT_DIR / "report_context_merit_v44.json"
REVIEWER = "codex-context-merit-audit-v44"
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
    for version in range(1, 44)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 44)
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-e1bf63a96bbb1c23765d",
        "source_path": raw("rope365_c89abf7c3a5c30e1.json"),
        "marker": "A quick-release is a way to create a temporary lock",
        "decision": "drop",
        "reason_code": "redundant_quick_release_definition",
        "reason": (
            "The bare definition omits accidental-release and solidity "
            "tradeoffs already preserved from this page in v34."
        ),
    },
    {
        "fact_id": "fact-bac953eeaadaa1b7763c",
        "source_path": raw("anatomiestudio_27ecdd4d7c9a5560.json"),
        "marker": "Both parties must be informed of the activity’s risks",
        "decision": "edit",
        "question": (
            "Before consenting to a rope or kink activity, what must both "
            "parties be informed about?"
        ),
        "answer": "the activity’s risks",
        "reason_code": "add_rope_kink_context_to_informed_consent",
        "reason": (
            "The edit retains the source's concise informed-consent rule while "
            "making the rope-or-kink setting explicit in the standalone question."
        ),
    },
    {
        "fact_id": "fact-73b8e3f019d949a0f738",
        "source_path": raw("rope365_25f1b23eb40be00e.json"),
        "marker": "The triple crown knot is the basic version of the flower knot",
        "decision": "drop",
        "reason_code": "incidental_flower_knot_name_lookup",
        "reason": (
            "The proper-name lookup is low-value decorative-knot trivia; the "
            "same page's learning goal is already preserved."
        ),
    },
    {
        "fact_id": "fact-95b9675d4375c43d22cb",
        "source_path": raw("rope365_8e6f4abea3bf4f6d.json"),
        "marker": "The action of rope wrapping around rope creates friction",
        "decision": "keep",
        "reason_code": "foundational_friction_definition",
        "reason": (
            "The source directly defines the physical action underlying rope "
            "frictions, a reusable foundational concept."
        ),
    },
    {
        "fact_id": "fact-c0369e7b1688f40ad963",
        "source_path": raw("rope365_25f1b23eb40be00e.json"),
        "marker": "Slip knots are fast, they tighten",
        "decision": "edit",
        "question": "What two limitations does Rope365 identify for slip knots?",
        "answer": "They tighten and are not stable.",
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The answer combines the source's two explicit limitations into a "
            "grammatical sentence without retaining its unrelated benefits."
        ),
        "paraphrase_support_fragments": ("they tighten", "they are not stable"),
        "reason_code": "complete_slip_knot_limitations",
        "reason": (
            "The edit adds the source's tightening behavior to the instability "
            "warning, making the safety limitation materially more complete."
        ),
    },
    {
        "fact_id": "fact-7755f8eb9d6e2e97d23d",
        "source_path": raw("rope365_5fdb5e78c2471772.json"),
        "marker": "Keep an untreated rope as your comparison tool",
        "decision": "keep",
        "reason_code": "useful_conditioning_control_comparison",
        "reason": (
            "An untreated comparison is a concise, transferable control for "
            "observing how conditioning changes otherwise similar rope."
        ),
    },
    {
        "fact_id": "fact-da15b630db4ec0ed79cf",
        "source_path": raw("anatomiestudio_144932682af9c846.json"),
        "marker": "making the ropes feel spongy and springy when they dry out",
        "decision": "keep",
        "reason_code": "useful_jute_wetting_effect",
        "reason": (
            "The source directly describes a recognizable texture change after "
            "jute swells and its twist tightens from wetting."
        ),
    },
    {
        "fact_id": "fact-7598fe1816e06f222c5a",
        "source_path": raw("kinbakutoday_4f417369f7269c51.json"),
        "marker": "like they used to do in the 70s and 80s with Scandinavian magazines",
        "decision": "drop",
        "reason_code": "hypothetical_interviewer_premise",
        "reason": (
            "The decades appear only inside an interviewer's hypothetical "
            "proposal and are not established or discussed as a sourced fact."
        ),
    },
    {
        "fact_id": "fact-a5d23c2e0d91786879f7",
        "source_path": raw("kinbakutoday_5cea0bc70445dc65.json"),
        "marker": "even the way of teaching, always one to one",
        "decision": "drop",
        "reason_code": "promotional_style_teaching_claim",
        "reason": (
            "The one-to-one assertion is a devotee's promotional claim about "
            "one named style, not general educational guidance."
        ),
    },
    {
        "fact_id": "fact-dea15f7c3a866a3328d7",
        "source_path": raw("rope365_5fdb5e78c2471772.json"),
        "marker": "whipping (sailmaker whipping, french whipping",
        "decision": "drop",
        "reason_code": "incidental_whipping_method_names",
        "reason": (
            "Two example names from a broader rope-end experiment add less "
            "value than the already retained comparative-learning guidance."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (234, 282, 466, 120, 349, 332, 46, 109, 81, 501))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v43",
    "direct_rows_without_prior_curation": 273,
    "eligible_unreviewed_direct_rows": 126,
    "prior_context_reviewed_direct_rows_excluded": 147,
    "rows": 560,
    "sha256": "520ee9b6b37f5729ba98c6232680094b9727d471fe71a9d35481ca7b498c18dd",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v43": 522,
    "active_after_this_tranche": 517,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 5,
    "new_edits_applied": 2,
    "output_rows": 555,
    "output_sha256": "489933ac1e1ae563e4894fffbba0627b428d3384843f4d76a3a86fa9285c4d13",
    "prior_pending_addition_fact_ids_preserved": 37,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 3,
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
    for version in range(1, 44):
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
    if len(ranked) != 126:
        raise ValueError(f"v44 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v44 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    document = json.loads(spec["source_path"].read_text())
    if document["url"] != active["url"]:
        raise ValueError(f'{active["fact_id"]}: source URL drift')
    active_evidence = active.get("evidence", "")
    if active["document_sha256"] not in {
            text_sha256(document["text"]),
            text_sha256(active_evidence) if active_evidence else ""}:
        raise ValueError(f'{active["fact_id"]}: source hash drift')
    if active_evidence and normalize_text(active_evidence) not in normalize_text(
            document["text"]):
        raise ValueError(f'{active["fact_id"]}: stored evidence drift')
    matches = [line for line in document["text"].splitlines()
               if spec["marker"] in line]
    if len(matches) != 1:
        raise ValueError(f'{active["fact_id"]}: evidence marker drift')
    evidence = matches[0]
    answer = spec.get("answer", active["answer"])
    support_type = spec.get("support_type", "normalized_extractive")
    if support_type == "manual_paraphrase":
        for fragment in spec["paraphrase_support_fragments"]:
            if normalize_text(fragment) not in normalize_text(evidence):
                raise ValueError(f'{active["fact_id"]}: unsupported paraphrase fragment')
    elif normalize_text(answer) not in normalize_text(evidence):
        raise ValueError(f'{active["fact_id"]}: unsupported answer')
    return evidence, support_type


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
        EVIDENCE_PATCH_MODULE.source_evidence = source_evidence
        yield
    finally:
        EVIDENCE_PATCH_MODULE.source_evidence = original_evidence
        CORE.ACTIVE_DATASET = original_active
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v43-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v43.jsonl"
        projected_report = Path(temp) / "projection-v43.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 560:
            raise ValueError("v43 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v43 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v44"
        row["review_pass"] = "first_context_merit_review_of_v43_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 560,
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
    report["schema"] = "context-merit-audit-report-v44"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [{"path": str(path.relative_to(ROOT)),
                      "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
    }
    report["selection"].update({
        "active_rows": 560, "eligible_unreviewed_rows": 126,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0, "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v43 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v43"
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
