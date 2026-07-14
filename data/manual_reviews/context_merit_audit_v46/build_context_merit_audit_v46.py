#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v45 projection rows in v46."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V45_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v45"
sys.path[:0] = [str(ROOT), str(V45_DIR)]
import build_context_merit_audit_v45 as previous
from qa_quality import normalize_text

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v46.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v46.jsonl"
REPORT = OUT_DIR / "report_context_merit_v46.json"
REVIEWER = "codex-context-merit-audit-v46"
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
    for version in range(1, 46)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 46)
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-365b99f4c4b01456b018",
        "source_path": raw("kinbakutoday_82071039cb003b58.json"),
        "marker": "Kinbaku has aspects of art",
        "decision": "keep",
        "reason_code": "attributed_theatrical_art_history",
        "reason": (
            "The answer preserves the source's attributed cultural-history claim "
            "linking Kinbaku's artistic dimension to Kabuki and Bunraku."
        ),
    },
    {
        "fact_id": "fact-712d310e90256e614708",
        "source_path": raw("anatomiestudio_27ecdd4d7c9a5560.json"),
        "marker": "incredibly important to discuss safety, boundaries and care",
        "decision": "keep",
        "reason_code": "essential_consensual_rope_discussion_topics",
        "reason": (
            "Safety, boundaries, and care are concise, foundational topics for "
            "consensual rope exploration."
        ),
    },
    {
        "fact_id": "fact-be26b9bcd62cfc651e85",
        "source_path": ROOT / "sources" / "manual_facts" / "resource_group_b.jsonl",
        "marker": "8 FT Triangular Footprint",
        "decision": "keep",
        "reason_code": "owner_requested_frame_footprint_specification",
        "reason": (
            "The manufacturer-attributed footprint is directly useful for deciding "
            "whether the owner-requested portable frame fits a space."
        ),
    },
    {
        "fact_id": "fact-45702017f4bcd78252db",
        "source_path": ROOT / "sources" / "manual_facts" / "resource_group_b.jsonl",
        "marker": "finest spun polyester yarns available",
        "decision": "keep",
        "reason_code": "owner_requested_vendor_rope_material",
        "reason": (
            "The vendor-attributed fiber content is useful when comparing material "
            "properties for the owner-requested synthetic-rope resource."
        ),
    },
    {
        "fact_id": "fact-949b2e98d14015289b42",
        "source_path": DATA / "rope_resource_manual_v1.jsonl",
        "marker": "Not every bowline is stable with nylon",
        "decision": "keep",
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The source says to test the bowline before depending on it; the stored "
            "answer expresses that timing as 'test it first'."
        ),
        "paraphrase_support_fragments": (
            "test it before you depend on it",
        ),
        "reason_code": "protected_nylon_bowline_stability_test",
        "reason": (
            "Testing a chosen bowline in nylon before depending on it is concrete, "
            "material-specific stability guidance."
        ),
    },
    {
        "fact_id": "fact-c3d3df1a50b4d695c35f",
        "source_path": raw("kinbakutoday_4f417369f7269c51.json"),
        "marker": "From September 1979 until January 2009 S & M Sniper",
        "evidence_end_marker": (
            "resident Editor–in–Chief between May 1980 and June 1982"
        ),
        "decision": "drop",
        "reason_code": "incidental_editor_publication_lookup",
        "reason": (
            "The publication-name lookup is incidental biographical trivia and adds "
            "less value than the interview's substantive publishing and cultural history."
        ),
    },
    {
        "fact_id": "fact-c1e07cd004048c483c86",
        "source_path": raw("anatomiestudio_451ac66001188a42.json"),
        "marker": "- Let them talk first",
        "decision": "keep",
        "reason_code": "protected_power_aware_partner_agency",
        "reason": (
            "Letting the less-powerful partner speak first is a clear, actionable way "
            "to reduce influence during planning and support their agency."
        ),
    },
    {
        "fact_id": "fact-d63b471f2cb9a7bf95f9",
        "source_path": raw("rope365_15518f0912cce205.json"),
        "marker": "Some of these positions may be difficult to sustain",
        "decision": "edit",
        "question": (
            "Before exploring restrictive elbow positions, what preparation does "
            "Rope365 recommend?"
        ),
        "answer": (
            "Warm up the arms, especially the shoulders, and identify positions "
            "that can be sustained comfortably."
        ),
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The answer combines the source's sustainability comparison and warm-up "
            "instruction into one practical preparation step."
        ),
        "paraphrase_support_fragments": (
            "positions may be difficult to sustain while others can be very comfortable",
            "Warm up the arms, especially the shoulders",
        ),
        "reason_code": "replace_upper_body_lookup_with_elbow_preparation",
        "reason": (
            "The edit replaces a vague anatomy answer with useful preparation and "
            "position-selection guidance from the same paragraph."
        ),
    },
    {
        "fact_id": "fact-e4f16fcd18549d4a1e80",
        "source_path": raw("rope365_25f1b23eb40be00e.json"),
        "marker": "Since it’s a type of slip knot",
        "decision": "keep",
        "reason_code": "useful_handcuff_knot_tightening_warning",
        "reason": (
            "The material directly warns that this slip-knot structure can tighten "
            "when placed on the body."
        ),
    },
    {
        "fact_id": "fact-05a050c66a8ee25a8fee",
        "source_path": raw("kinbakutoday_c364f23ce34ae761.json"),
        "marker": "sexuality and sexual acts are often still categorized",
        "evidence_end_marker": "different facets that shouldn’t be equated",
        "decision": "edit",
        "question": (
            "Why does the author say classifying rope as simply 'sexual' or 'not "
            "sexual' is too shallow?"
        ),
        "answer": (
            "Because sexual intimacy, arousal, and sexual acts are distinct facets "
            "that should not be assumed to imply one another."
        ),
        "support_type": "manual_paraphrase",
        "paraphrase_rationale": (
            "The answer condenses the source's binary-category criticism and explicit "
            "examples of facets that people wrongly equate."
        ),
        "paraphrase_support_fragments": (
            "sexuality and sexual acts are often still categorized following an all–or–nothing principle",
            "equate processes like (sexual) intimacy, sexual arousal and engaging in a sexual act",
            "different facets that shouldn’t be equated",
        ),
        "reason_code": "replace_abstract_principle_lookup_with_sexuality_distinction",
        "reason": (
            "The edit makes the useful consent implication explicit: arousal, intimacy, "
            "and sexual acts are different and cannot be inferred from one label."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (491, 358, 20, 10, 295, 474, 48, 286, 310, 301))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v45",
    "direct_rows_without_prior_curation": 260,
    "eligible_unreviewed_direct_rows": 106,
    "prior_context_reviewed_direct_rows_excluded": 154,
    "rows": 552,
    "sha256": "4e7b4cd68cefc09eedd78c81f54ae2c5d3c64930ab4adbb6b09dab52525bf2c2",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v45": 514,
    "active_after_this_tranche": 513,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 2,
    "output_rows": 551,
    "output_sha256": "898af65291f91e80f9f2d8ee548eaecb8f065861cf5ba95f43f452891814a3b6",
    "prior_pending_addition_fact_ids_preserved": 37,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 7,
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
    for version in range(1, 46):
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
    if len(ranked) != 106:
        raise ValueError(f"v46 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v46 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    if not spec.get("evidence_end_marker"):
        return previous.source_evidence(spec, active)
    source_text = previous.source_text_for(spec, active)
    active_evidence = active.get("evidence", "")
    acceptable_hashes = {text_sha256(source_text)}
    if active_evidence:
        acceptable_hashes.add(text_sha256(active_evidence))
    if active["document_sha256"] not in acceptable_hashes:
        raise ValueError(f'{active["fact_id"]}: source hash drift')
    lines = source_text.splitlines()
    starts = [index for index, line in enumerate(lines) if spec["marker"] in line]
    ends = [index for index, line in enumerate(lines)
            if spec["evidence_end_marker"] in line]
    if len(starts) != 1 or len(ends) != 1 or ends[0] < starts[0]:
        raise ValueError(f'{active["fact_id"]}: evidence span drift')
    evidence = "\n".join(lines[starts[0]:ends[0] + 1])
    support_type = spec.get("support_type", "normalized_extractive")
    if support_type == "manual_paraphrase":
        for fragment in spec["paraphrase_support_fragments"]:
            if normalize_text(fragment) not in normalize_text(evidence):
                raise ValueError(
                    f'{active["fact_id"]}: unsupported paraphrase fragment')
    elif normalize_text(spec.get("answer", active["answer"])) not in normalize_text(
            evidence):
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
    with tempfile.TemporaryDirectory(prefix=".v45-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v45.jsonl"
        projected_report = Path(temp) / "projection-v45.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 552:
            raise ValueError("v45 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v45 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v46"
        row["review_pass"] = "first_context_merit_review_of_v45_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 552,
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
    report["schema"] = "context-merit-audit-report-v46"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [{"path": str(path.relative_to(ROOT)),
                      "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
    }
    report["selection"].update({
        "active_rows": 552, "eligible_unreviewed_rows": 106,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0, "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v45 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v45"
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
        "extractive": 0, "manual_paraphrase": 2,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
