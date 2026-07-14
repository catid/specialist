#!/usr/bin/env python3
"""Improve three source-bounded Kinbaku Today culture and teaching Q&A rows."""
from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V71_DIR = DATA / "manual_reviews/context_merit_audit_v71"
sys.path[:0] = [str(ROOT), str(V71_DIR)]
import build_context_merit_audit_v71 as previous

BASE = previous.BASE
CORE = previous.CORE
EVIDENCE_PATCH_MODULE = previous.EVIDENCE_PATCH_MODULE
OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v72.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v72.jsonl"
REPORT = OUT_DIR / "report_context_merit_v72.json"
REVIEWER = "codex-context-merit-audit-v72"
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
        "fact_id": "fact-08a27108ab9b8483340c",
        "source_path": raw("kinbakutoday_b6ca615acc15e5f5.json"),
        "marker": "he did no retouching at all to the photos for this exhibition",
        "decision": "edit",
        "question": "Why did photographer Yasui Kissyou leave the Kinbaku Shashuu portraits unretouched?",
        "answer": "He wanted to show kinbaku’s distinctive tempestuousness without modification.",
        "reason_code": "scope_unretouched_portraits_to_named_exhibition",
        "reason": "The revised question names the exhibition and the answer is a grammatical normalization of the photographer’s stated artistic reason.",
        "support_type": "manual_paraphrase",
        "paraphrase_support_fragments": (
            "he did no retouching at all",
            "he wanted to express the distinctive tempestuousness of Kinbaku without modification",
        ),
    },
    {
        "fact_id": "fact-872e45e0ff772b90fa8e",
        "source_path": raw("kinbakutoday_b2454d5b6578b8c6.json"),
        "marker": "constant questioning and not doing anything because he was told",
        "decision": "edit",
        "question": "According to Scott, what habit kept Osada Steve’s rope and teaching focus changing over time?",
        "answer": "He continually questioned whether each choice was best for ease of tying, connection, aesthetics, and other considerations.",
        "reason_code": "replace_vague_considerations_list_with_teaching_habit",
        "reason": "The revised Q&A captures the interview’s transferable lesson—continual critical questioning—instead of asking for an incomplete list ending in ‘etc.’",
        "support_type": "manual_paraphrase",
        "paraphrase_support_fragments": (
            "constant questioning",
            "best thing to do based on a range of considerations",
            "ease of tying, connection, aesthetics",
            "his rope and focus is always in flux",
        ),
    },
    {
        "fact_id": "fact-ace591d75c7226db653a",
        "source_path": raw("kinbakutoday_4f9dec06e4af751a.json"),
        "marker": "onnen (怨念) does not simply name anger or resentment",
        "decision": "edit",
        "question": "In Kinbaku Today’s “The Ghosts of Kinbaku History,” what does onnen mean?",
        "answer": "resentment that has congealed into a durable, almost haunting force when injury cannot be resolved through ordinary justice",
        "reason_code": "replace_rare_term_recall_with_source_scoped_definition",
        "reason": "The revised Q&A teaches the article’s definition directly instead of requiring the learner to retrieve a rare Japanese term from a long definition.",
        "support_type": "manual_paraphrase",
        "paraphrase_support_fragments": (
            "resentment that has congealed into a durable, almost haunting force",
            "onnen is what remains when injury cannot be resolved through ordinary justice",
        ),
    },
)
EXPECTED_SELECTION = tuple(spec["fact_id"] for spec in SPECS)
PROJECTED_ACTIVE_INDICES = dict(zip(EXPECTED_SELECTION, (312, 335, 336)))
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v71",
    "direct_rows_without_prior_curation": 148,
    "kinbakutoday_culture_and_teaching_rows_selected": 3,
    "rows": 536,
    "sha256": "65644c7ce87ff5d7cbf9d1c35f12299bc06459daabba4b779693ff1602bdb63d",
}
EXPECTED_OUTPUT_SHA256 = "e54ecadd3dc2dafed0df4784e557d75980fbb4e341110446136df325a4f8a90d"
ISOLATED_PROJECTION = {
    "active_after_context_merit_v71": 498,
    "active_after_this_tranche": 498,
    "automated_projection_runs": 2,
    "build_script": "build_curated_qa.py",
    "new_drops_applied": 0,
    "new_edits_applied": 3,
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
    for version in range(1, 72):
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
    if {
        item["row"]["fact_id"]: item["active_index"] for item in result
    } != PROJECTED_ACTIVE_INDICES:
        raise ValueError("v72 candidate drift")
    return result


def selected_ranked(rows):
    return selected(rows), 0, 0


def evidence_validator():
    module = previous.framework
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
    with tempfile.TemporaryDirectory(prefix=".v72-observation-", dir=OUT_DIR) as temporary:
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
    with tempfile.TemporaryDirectory(prefix=".v71-projection-", dir=OUT_DIR) as temporary:
        baseline = Path(temporary) / "v71.jsonl"
        baseline_report = Path(temporary) / "v71.report.json"
        build_projection(baseline, baseline_report, PRIOR_PROJECTION_CURATIONS)
        if (
            len(read_jsonl(baseline)) != 536
            or file_sha256(baseline) != PROJECTED_SELECTION_BASELINE["sha256"]
        ):
            raise ValueError("v71 projection drift")
        with patched_base(baseline):
            BASE.main()

    audits = read_jsonl(AUDIT)
    for row in audits:
        row.update(
            schema="context-merit-audit-v72",
            review_pass="kinbakutoday_culture_and_teaching_quality_reaudit",
            projection_lineage={
                "active_index": PROJECTED_ACTIVE_INDICES[row["fact_id"]],
                "baseline_rows": 536,
                "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
                "prior_context_merit_review": True,
            },
        )
        spec = next(spec for spec in SPECS if spec["fact_id"] == row["fact_id"])
        if spec.get("support_type") == "manual_paraphrase":
            row["paraphrase_rationale"] = spec["reason"]
    write_jsonl(AUDIT, audits)

    curations = read_jsonl(CURATION)
    for row in curations:
        spec = next(spec for spec in SPECS if spec["fact_id"] == row["fact_id"])
        if spec.get("support_type") == "manual_paraphrase":
            row.update(
                support_type="manual_paraphrase",
                paraphrase_rationale=spec["reason"],
            )
    write_jsonl(CURATION, curations)

    observed = deterministic_projection()
    if (
        not observed["dataset_equal"]
        or not observed["report_equal"]
        or observed["rows"] != 536
        or observed["eval_fact_count"] != 612
    ):
        raise ValueError("v72 deterministic projection drift")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v72 output hash drift")

    report = json.loads(REPORT.read_text())
    report["schema"] = "context-merit-audit-report-v72"
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
        active_rows=536,
        eligible_unreviewed_rows=0,
        excluded_active_review_provenance=0,
        excluded_ledger_fact_ids=0,
        rows_selected=3,
        projected_baseline=PROJECTED_SELECTION_BASELINE,
        ranking={
            "candidate_rule": "three remaining direct Kinbaku Today rows with vague list answers, missing article scope, or low-value term-recall framing",
            "score": "manual full-source standalone clarity, explanatory value, and attribution review",
            "tie_break": "active projection order",
        },
    )
    report["audit"]["rows"] = len(audits)
    report["audit"]["sha256"] = file_sha256(AUDIT)
    report["new_pending_curation"]["sha256"] = file_sha256(CURATION)
    report["new_pending_curation"]["edit_support_types"] = {
        "extractive": 0,
        "manual_paraphrase": 3,
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
