#!/usr/bin/env python3
"""Audit the highest-risk directly curatable v47 projection rows in v48."""

from __future__ import annotations

import contextlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V47_DIR = ROOT / "data" / "manual_reviews" / "context_merit_audit_v47"
sys.path[:0] = [str(ROOT), str(V47_DIR)]
import build_context_merit_audit_v47 as previous

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
DATA = ROOT / "data"
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v48.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v48.jsonl"
REPORT = OUT_DIR / "report_context_merit_v48.json"
REVIEWER = "codex-context-merit-audit-v48"
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
    for version in range(1, 48)
)
CONTEXT_AUDITS = tuple(
    DATA / "manual_reviews" / f"context_merit_audit_v{version}" /
    f"context_merit_audit_v{version}.jsonl"
    for version in range(1, 48)
)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-481da1996df16fbdcd09",
        "source_path": ROOT / "sources" / "manual_facts" / "resource_group_b.jsonl",
        "marker": "designed to be a portable aerial rig for Hammocks, Silks, Hoops",
        "decision": "keep",
        "reason_code": "manufacturer_stated_a_frame_uses",
        "reason": (
            "The manufacturer-stated disciplines clarify the intended scope of an "
            "owner-requested frame resource instead of implying unsupported uses."
        ),
    },
    {
        "fact_id": "fact-854eee7f7afdc00ab504",
        "source_path": raw("kinbakutoday_56b268785cbff3bb.json"),
        "marker": "Titled Nawa no Jiba",
        "decision": "drop",
        "reason_code": "incidental_exhibition_title_lookup",
        "reason": (
            "The retrospective's proper-name lookup is low-value event trivia; the "
            "source's career and photographic-history context is more substantive."
        ),
    },
    {
        "fact_id": "fact-9e1d84a06b0b7efe1dc3",
        "source_path": raw("rope365_f250f228cc370052.json"),
        "marker": "A single column tie is a cuff that will not tighten",
        "decision": "keep",
        "reason_code": "foundational_non_tightening_column_tie_definition",
        "reason": (
            "The answer defines a foundational structure by its essential behavior "
            "under tail tension."
        ),
    },
    {
        "fact_id": "fact-55f26dc889d7aaa6c515",
        "source_path": raw("rope_resources_v1/rope365__7b5d548036392d65fec7.json"),
        "marker": "Participating in a first aid class",
        "decision": "keep",
        "reason_code": "first_aid_emergency_preparation",
        "reason": (
            "Formal first-aid training is practical preparation for responding to "
            "common emergencies rather than improvising during an incident."
        ),
    },
    {
        "fact_id": "fact-9fb9d97a8ac45f31a092",
        "source_path": raw("anatomiestudio_144932682af9c846.json"),
        "marker": "Where ropes can and can’t go on the body",
        "decision": "keep",
        "reason_code": "pre_scene_rope_placement_agreement",
        "reason": (
            "Agreeing where rope may touch is concrete informed-consent guidance, "
            "especially when hygiene and intimate areas are relevant."
        ),
    },
    {
        "fact_id": "fact-166c469beb2e54bb5e96",
        "source_path": DATA / "rope_resource_manual_v1.jsonl",
        "marker": "either twisted (most commonly with 3, but sometimes 4, strands)",
        "support_type": "manual_paraphrase",
        "paraphrase_support_fragments": (
            "either twisted",
            "or a solid braid with no core",
        ),
        "paraphrase_rationale": (
            "The answer removes the parenthetical strand-count aside while preserving "
            "the source's two recommended coreless construction categories."
        ),
        "decision": "keep",
        "reason_code": "protected_coreless_bondage_rope_construction",
        "reason": (
            "The answer preserves a construction-level equipment recommendation and "
            "explicitly excludes a core from the braided option."
        ),
    },
    {
        "fact_id": "fact-ee87d53a840fc422951b",
        "source_path": raw("wykd_944e4e6d621a97c9.json"),
        "marker": "Not to be touched, slapped, spanked, groped",
        "decision": "keep",
        "reason_code": "newcomer_event_contact_consent",
        "reason": (
            "The source clearly states that attending a kink event does not imply "
            "consent to physical or sexualized contact."
        ),
    },
    {
        "fact_id": "fact-38ec1728983f826643a2",
        "source_path": raw("kinbakutoday_73b16e835ab63cc2.json"),
        "marker": "kindness is nothing more than the quality of seeing another",
        "decision": "keep",
        "reason_code": "explicitly_attributed_kindness_definition",
        "reason": (
            "The question explicitly attributes a coherent relational definition and "
            "supports the same article's partner-centered rope ethic retained in v47."
        ),
    },
    {
        "fact_id": "fact-52aaf228bd4fb0739e29",
        "source_path": raw("rope_resources_v1/rope365__7b5d548036392d65fec7.json"),
        "marker": "Clean up your play space to prevent tripping hazards",
        "decision": "keep",
        "reason_code": "practical_play_space_trip_prevention",
        "reason": (
            "Removing mats, rope bags, people, and clutter from the path is direct "
            "fall-prevention guidance."
        ),
    },
    {
        "fact_id": "fact-5c67a11fec80968c6494",
        "source_path": DATA / "rope_topia_manual_v1.jsonl",
        "marker": "<loc>https://rope-topia.com/portfolio-items/strugglers-knot/</loc>",
        "decision": "keep",
        "allow_document_sha_mismatch": True,
        "reason_code": "owner_requested_strugglers_knot_resource_url",
        "reason": (
            "The sitemap-backed canonical tutorial URL is retained as owner-requested "
            "resource metadata without inferring unavailable article content."
        ),
    },
)

EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = {
    fact_id: index for fact_id, index in zip(
        EXPECTED_SELECTION, (124, 258, 11, 361, 322, 447, 320, 84, 73, 442))
}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated cumulative training projection through context-merit v47",
    "direct_rows_without_prior_curation": 255,
    "eligible_unreviewed_direct_rows": 86,
    "prior_context_reviewed_direct_rows_excluded": 169,
    "rows": 549,
    "sha256": "d2998dcacbe7dbc02ed8ccf6dbbab953fdc431eed3c9845d3051fa9badf4cb23",
}
ISOLATED_PROJECTION = {
    "active_after_context_merit_v47": 511,
    "active_after_this_tranche": 510,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 0,
    "output_rows": 548,
    "output_sha256": "1c585357d18896cdd0264ce0502dfb600a2c2bf76f6c726e3b74b46d4902a8f7",
    "prior_pending_addition_fact_ids_preserved": 37,
    "repeat_dataset_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 9,
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
    for version in range(1, 48):
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
    if len(ranked) != 86:
        raise ValueError(f"v48 candidate drift: {len(ranked)}")
    if tuple(item["row"]["fact_id"] for item in ranked[:10]) != EXPECTED_SELECTION:
        raise ValueError("v48 selection drift")
    return ranked


def selected_ranked(rows: list[dict]) -> tuple[list[dict], int, int]:
    return ranked_unreviewed_direct(rows)[:10], 0, 0


def source_evidence(spec: dict, active: dict) -> tuple[str, str]:
    """Use v46's span-aware validator, adding discontinuous paraphrase support."""
    if not spec.get("paraphrase_support_fragments"):
        return previous.previous.source_evidence(spec, active)
    source_text = previous.previous.previous.source_text_for(spec, active)
    active_evidence = active.get("evidence", "")
    acceptable_hashes = {text_sha256(source_text)}
    if active_evidence:
        acceptable_hashes.add(text_sha256(active_evidence))
    if active["document_sha256"] not in acceptable_hashes:
        raise ValueError(f'{active["fact_id"]}: source hash drift')
    matches = [line for line in source_text.splitlines()
               if spec["marker"] in line]
    if len(matches) != 1:
        raise ValueError(f'{active["fact_id"]}: evidence marker drift')
    evidence = matches[0]
    normalized_evidence = previous.previous.previous.normalize_text(evidence)
    for fragment in spec["paraphrase_support_fragments"]:
        if previous.previous.previous.normalize_text(fragment) not in normalized_evidence:
            raise ValueError(f'{active["fact_id"]}: unsupported paraphrase fragment')
    return evidence, "manual_paraphrase"


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
    with tempfile.TemporaryDirectory(prefix=".v47-projection-", dir=OUT_DIR) as temp:
        projected_dataset = Path(temp) / "projection-v47.jsonl"
        projected_report = Path(temp) / "projection-v47.report.json"
        build_projection(projected_dataset, projected_report,
                         PRIOR_PROJECTION_CURATIONS)
        if len(read_jsonl(projected_dataset)) != 549:
            raise ValueError("v47 projection row-count drift")
        if file_sha256(projected_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v47 projection hash drift")
        with patched_base(projected_dataset):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row["schema"] = "context-merit-audit-v48"
        row["review_pass"] = "first_context_merit_review_of_v47_projection_row"
        row["projection_lineage"] = {
            "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
            "baseline_rows": 549,
            "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
            "prior_context_merit_review": False,
        }
        spec = next(spec for spec in SPECS if spec["fact_id"] == row["fact_id"])
        if spec.get("support_type") == "manual_paraphrase":
            row["paraphrase_rationale"] = spec["paraphrase_rationale"]
    write_jsonl(AUDIT, audits)

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v48"
    report["active_baseline"] = {
        "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784,
                    "sha256": file_sha256(ACTIVE_DATASET)},
        "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)),
                   "sha256": file_sha256(ACTIVE_REPORT)},
        "curation": [{"path": str(path.relative_to(ROOT)),
                      "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
    }
    report["selection"].update({
        "active_rows": 549, "eligible_unreviewed_rows": 86,
        "excluded_active_review_provenance": 0,
        "excluded_ledger_fact_ids": 0, "rows_selected": 10,
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "ranking": {
            "candidate_rule": (
                "the row survives the v47 projection, has no prior curation "
                "metadata, and its fact_id has no context-merit decision in "
                "v1 through v47"
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
