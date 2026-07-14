#!/usr/bin/env python3
"""Audit two remaining Wikipedia-derived history rows for source quality and merit."""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V70_DIR = DATA / "manual_reviews/context_merit_audit_v70"
V68_DIR = DATA / "manual_reviews/context_merit_audit_v68"
sys.path[:0] = [str(ROOT), str(V70_DIR), str(V68_DIR)]
import build_context_merit_audit_v70 as previous
import build_context_merit_audit_v68 as framework

BASE = framework.BASE
CORE = framework.CORE
EVIDENCE_PATCH_MODULE = framework.EVIDENCE_PATCH_MODULE
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v71.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v71.jsonl"
REPORT = OUT_DIR / "report_context_merit_v71.json"
REVIEWER = "codex-context-merit-audit-v71"
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

CONTEXT_CURATIONS = previous.OUTPUT_CONTEXT_CURATIONS
PRIOR_PROJECTION_CURATIONS = previous.OUTPUT_PROJECTION_CURATIONS
OUTPUT_CONTEXT_CURATIONS = (*CONTEXT_CURATIONS, CURATION)
OUTPUT_PROJECTION_CURATIONS = (*PRIOR_PROJECTION_CURATIONS, CURATION)


def raw(name: str) -> Path:
    return DATA / "raw" / name


SPECS = (
    {
        "fact_id": "fact-7e567cc256ae27660f3f",
        "source_path": raw("wikipedia_a97e19abfe49afa9.json"),
        "marker": 'one or occasionally two "main ropes" or honnawa which, like the hayanawa, could be any one of many different lengths, but was typically hemp in material',
        "decision": "edit",
        "question": "According to Wikipedia’s Hojōjutsu article, what material was the main rope, or honnawa, typically made from?",
        "answer": "hemp",
        "reason_code": "scope_honnawa_material_to_wikipedia_source",
        "reason": "The historical material detail is relevant, but the revised wording attributes the claim to its secondary source instead of presenting it without provenance.",
    },
    {
        "fact_id": "fact-e96b48731c29761bc94d",
        "source_path": raw("wikipedia_093ebd176b6adaaf.json"),
        "marker": 'According to David Stein, the man who coined "Safe, Sane, and Consensual S/M"',
        "decision": "drop",
        "reason_code": "secondary_source_consent_slogan_attribution_trivia",
        "reason": "The dataset already retains practical consent guidance and the RACK expansion; this secondary-source name attribution adds trivia rather than actionable rope knowledge.",
    },
)
EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = dict(zip(EXPECTED_SELECTION, (288, 483)))
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v70",
    "direct_rows_without_prior_curation": 150,
    "rows": 537,
    "sha256": "1c0aca8c60cafda413040401ae0320518715378b711d73add82075eb98e6f797",
    "wikipedia_history_rows_selected": 2,
}
EXPECTED_OUTPUT_SHA256 = "65644c7ce87ff5d7cbf9d1c35f12299bc06459daabba4b779693ff1602bdb63d"
ISOLATED_PROJECTION = {
    "active_after_context_merit_v70": 499,
    "active_after_this_tranche": 498,
    "automated_projection_runs": 2,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 1,
    "new_edits_applied": 1,
    "output_rows": 536,
    "output_sha256": EXPECTED_OUTPUT_SHA256,
    "prior_pending_addition_fact_ids_preserved": 36,
    "repeat_dataset_byte_identical": True,
    "repeat_projection_report_byte_identical": True,
    "reviewed_keep_fact_ids_preserved": 0,
    "sealed_eval_fact_count_reported_by_tooling": 612,
    "unexpected_fact_ids": 0,
}


def prior_decision_artifacts():
    artifacts = []
    for version in range(1, 71):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        artifacts.extend(
            (
                directory / f"context_merit_audit_v{version}.jsonl",
                directory / f"pending_curation_context_merit_v{version}.jsonl",
                directory / f"report_context_merit_v{version}.json",
            )
        )
    return tuple(artifacts)


def build_projection(output: Path, report: Path, curations):
    previous.build_projection(output, report, curations)


def selected(rows):
    by_fact_id = {row["fact_id"]: (index, row) for index, row in enumerate(rows, 1)}
    result = [
        {
            "active_index": by_fact_id[fact_id][0],
            "row": by_fact_id[fact_id][1],
            "features": CORE.risk_features(by_fact_id[fact_id][1]),
        }
        for fact_id in EXPECTED_SELECTION
    ]
    observed_indices = {
        item["row"]["fact_id"]: item["active_index"] for item in result
    }
    if observed_indices != PROJECTED_ACTIVE_INDICES:
        raise ValueError("v71 candidate drift")
    return result


def selected_ranked(rows):
    return selected(rows), 0, 0


def evidence_validator():
    module = framework
    while not hasattr(module, "source_evidence"):
        module = module.previous
    return module.source_evidence


@contextlib.contextmanager
def patched_base(dataset: Path):
    replacements = {
        "OUT_DIR": OUT_DIR,
        "AUDIT": AUDIT,
        "CURATION": CURATION,
        "REPORT": REPORT,
        "REVIEWER": REVIEWER,
        "REVIEWED_AT": REVIEWED_AT,
        "CONTEXT_CURATIONS": CONTEXT_CURATIONS,
        "SPECS": SPECS,
        "EXPECTED_SELECTION": EXPECTED_SELECTION,
        "ISOLATED_PROJECTION": ISOLATED_PROJECTION,
    }
    originals = {name: getattr(BASE, name) for name in replacements}
    original_ranking = CORE.ranked_unreviewed
    original_dataset = CORE.ACTIVE_DATASET
    original_evidence = EVIDENCE_PATCH_MODULE.source_evidence
    try:
        for name, value in replacements.items():
            setattr(BASE, name, value)
        CORE.ranked_unreviewed = selected_ranked
        CORE.ACTIVE_DATASET = dataset
        EVIDENCE_PATCH_MODULE.source_evidence = evidence_validator()
        yield
    finally:
        EVIDENCE_PATCH_MODULE.source_evidence = original_evidence
        CORE.ACTIVE_DATASET = original_dataset
        CORE.ranked_unreviewed = original_ranking
        for name, value in originals.items():
            setattr(BASE, name, value)


def deterministic_projection():
    with tempfile.TemporaryDirectory(prefix=".v71-observation-", dir=OUT_DIR) as temporary:
        directory = Path(temporary)
        dataset = directory / "projection.jsonl"
        projection_report = directory / "projection.report.json"
        datasets = []
        reports = []
        for _ in (1, 2):
            build_projection(dataset, projection_report, OUTPUT_PROJECTION_CURATIONS)
            datasets.append(dataset.read_bytes())
            reports.append(projection_report.read_bytes())
        parsed_report = json.loads(reports[0])
        normalized_report = dict(parsed_report)
        normalized_report["output"] = "<projection-output>"
        normalized_bytes = (json.dumps(normalized_report, indent=2, sort_keys=True) + "\n").encode()
        return {
            "dataset_equal": datasets[0] == datasets[1],
            "dataset_sha256": hashlib.sha256(datasets[0]).hexdigest(),
            "report_equal": reports[0] == reports[1],
            "report_normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
            "rows": datasets[0].count(b"\n"),
            "eval_fact_count": parsed_report["eval_fact_count"],
        }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v70-projection-", dir=OUT_DIR) as temporary:
        baseline = Path(temporary) / "v70.jsonl"
        baseline_report = Path(temporary) / "v70.report.json"
        build_projection(baseline, baseline_report, PRIOR_PROJECTION_CURATIONS)
        if (
            len(read_jsonl(baseline)) != 537
            or file_sha256(baseline) != PROJECTED_SELECTION_BASELINE["sha256"]
        ):
            raise ValueError("v70 projection drift")
        with patched_base(baseline):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row.update(
            schema="context-merit-audit-v71",
            review_pass="wikipedia_history_source_quality_reaudit",
            projection_lineage={
                "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
                "baseline_rows": 537,
                "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
                "prior_context_merit_review": True,
            },
        )
    write_jsonl(AUDIT, audits)

    observed = deterministic_projection()
    if (
        not observed["dataset_equal"]
        or not observed["report_equal"]
        or observed["rows"] != 536
        or observed["eval_fact_count"] != 612
    ):
        raise ValueError("v71 deterministic projection drift")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v71 output hash drift")

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v71"
    report["active_baseline"] = {
        "dataset": {
            "path": str(ACTIVE_DATASET.relative_to(ROOT)),
            "rows": 784,
            "sha256": file_sha256(ACTIVE_DATASET),
        },
        "report": {
            "path": str(ACTIVE_REPORT.relative_to(ROOT)),
            "sha256": file_sha256(ACTIVE_REPORT),
        },
        "curation": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
            for path in ACTIVE_CURATIONS
        ],
    }
    report["selection"].update(
        active_rows=537,
        eligible_unreviewed_rows=0,
        excluded_active_review_provenance=0,
        excluded_ledger_fact_ids=0,
        rows_selected=2,
        projected_baseline=PROJECTED_SELECTION_BASELINE,
        ranking={
            "candidate_rule": "remaining concise Wikipedia-derived history rows selected for bounded full-source relevance and provenance review",
            "score": "manual source quality, rope relevance, and actionability review",
            "tie_break": "active projection order",
        },
    )
    report["audit"]["rows"] = len(audits)
    report["audit"]["sha256"] = file_sha256(AUDIT)
    report["new_pending_curation"]["sha256"] = file_sha256(CURATION)
    report["new_pending_curation"]["edit_support_types"] = {
        "extractive": 1,
        "manual_paraphrase": 0,
    }
    report["frozen_prior_decision_artifacts"] = [
        {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)}
        for path in prior_decision_artifacts()
    ]
    report["isolated_build_projection"].update(
        output_sha256=observed["dataset_sha256"],
        output_rows=observed["rows"],
        repeat_dataset_byte_identical=observed["dataset_equal"],
        repeat_projection_report_byte_identical=observed["report_equal"],
        projection_report_normalized_sha256=observed["report_normalized_sha256"],
        sealed_eval_fact_count_reported_by_tooling=observed["eval_fact_count"],
        determinism_comparison_scope="identical inputs, curation chain, and output/report paths",
    )
    report["sealed_evaluation_policy"] = {
        "automated_collision_tool": "build_curated_qa.py",
        "automated_collision_tool_reads_sealed_content": True,
        "automated_read_scope": "fact-id collision exclusion and aggregate eval_fact_count reporting only",
        "manual_worker_opened_eval_or_heldout_content": False,
        "manual_worker_received_eval_or_heldout_content": False,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
