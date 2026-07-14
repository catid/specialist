#!/usr/bin/env python3
"""Repair second-stage projection hermeticity without changing dataset content."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V128_DIR = DATA / "manual_reviews/context_merit_audit_v128"
V117_DIR = DATA / "manual_reviews/context_merit_audit_v117"
sys.path[:0] = [str(ROOT), str(V128_DIR), str(V117_DIR)]
import build_context_merit_audit_v128 as previous
import build_context_merit_audit_v117 as stage1

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v129.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v129.jsonl"
REPORT = OUT_DIR / "report_context_merit_v129.json"
REVIEWER = "codex-context-merit-audit-v129"
REVIEWED_AT = "2026-07-14"

RESOURCE_MANIFEST = previous.RESOURCE_MANIFEST
ACTIVE_DATASET = previous.ACTIVE_DATASET
ACTIVE_REPORT = previous.ACTIVE_REPORT
ACTIVE_CURATIONS = previous.ACTIVE_CURATIONS
PRIOR_PENDING_ADDITIONS = previous.PRIOR_PENDING_ADDITIONS
QUALITY_MERIT_CURATION = previous.QUALITY_MERIT_CURATION
TASUKI_CURATION = previous.TASUKI_CURATION
CORE = previous.CORE
file_sha256 = previous.file_sha256
text_sha256 = previous.text_sha256
read_jsonl = previous.read_jsonl
write_jsonl = previous.write_jsonl
CONTEXT_CURATIONS = previous.OUTPUT_CONTEXT_CURATIONS
PRIOR_PROJECTION_CURATIONS = previous.OUTPUT_PROJECTION_CURATIONS
OUTPUT_CONTEXT_CURATIONS = (*CONTEXT_CURATIONS, CURATION)
OUTPUT_PROJECTION_CURATIONS = (*PRIOR_PROJECTION_CURATIONS, CURATION)
SQUASHED_PROJECTION_CURATIONS = stage1.OUTPUT_PROJECTION_CURATIONS
SEALED_EVAL_PATHS = (
    DATA / "eval_qa.jsonl",
    DATA / "eval_qa_v2.jsonl",
    DATA / "eval_qa_v3.jsonl",
    DATA / "ood_qa.jsonl",
    DATA / "ood_qa_v3.jsonl",
)
BRIDGE_INPUT_LABEL = "<isolated-v117-projection>"

# This tranche is an infrastructure repair. No content row is selected or changed.
SPECS = ()
EXPECTED_SELECTION = ()
PROJECTED_ACTIVE_INDICES = {}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v128",
    "direct_rows_without_prior_curation": 92,
    "hermetic_bridge_regression_rows_selected": 0,
    "rows": 515,
    "sha256": "0d1293f93bde58ea0df221217036ab6215198851ed88d05f5879f0b2c46b7f58",
}
EXPECTED_OUTPUT_SHA256 = PROJECTED_SELECTION_BASELINE["sha256"]


def _canonicalize_bridge_path(value, volatile_path: str):
    """Replace the invocation-local path everywhere, including dictionary keys."""
    if isinstance(value, str):
        return value.replace(volatile_path, BRIDGE_INPUT_LABEL)
    if isinstance(value, list):
        return [_canonicalize_bridge_path(item, volatile_path) for item in value]
    if isinstance(value, dict):
        return {
            _canonicalize_bridge_path(key, volatile_path): _canonicalize_bridge_path(item, volatile_path)
            for key, item in value.items()
        }
    return value


def build_projection(output: Path, report: Path, curations) -> None:
    """Build through an invocation-unique v117 bridge and emit a stable report."""
    curations = tuple(curations)
    prefix_length = len(SQUASHED_PROJECTION_CURATIONS)
    if curations[:prefix_length] != SQUASHED_PROJECTION_CURATIONS:
        raise ValueError("v129 projection curation-prefix drift")
    later_curations = curations[prefix_length:]

    with tempfile.TemporaryDirectory(prefix=".v129-bridge-", dir=OUT_DIR) as temp:
        bridge_dir = Path(temp)
        upstream_dataset = bridge_dir / "v117.jsonl"
        upstream_report = bridge_dir / "v117.report.json"
        stage1.build_projection(
            upstream_dataset,
            upstream_report,
            SQUASHED_PROJECTION_CURATIONS,
        )
        volatile_path = str(upstream_dataset.relative_to(ROOT))
        command = [
            sys.executable,
            "build_curated_qa.py",
            "--inputs",
            volatile_path,
            "--eval",
            *(str(path.relative_to(ROOT)) for path in SEALED_EVAL_PATHS),
            "--curation",
            *(str(path.relative_to(ROOT)) for path in later_curations),
            "--output",
            str(output),
            "--report",
            str(report),
        ]
        subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        parsed = json.loads(report.read_text())
        if parsed.get("inputs") != [volatile_path] or list(parsed.get("input_sha256", {})) != [volatile_path]:
            raise ValueError("v129 bridge provenance drift")
        parsed = _canonicalize_bridge_path(parsed, volatile_path)
        report.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n")


def prior_decision_artifacts():
    out = []
    for version in range(1, 129):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        out.extend(
            (
                directory / f"context_merit_audit_v{version}.jsonl",
                directory / f"pending_curation_context_merit_v{version}.jsonl",
                directory / f"report_context_merit_v{version}.json",
            )
        )
    return tuple(out)


def observation():
    with tempfile.TemporaryDirectory(prefix=".v129-observation-", dir=OUT_DIR) as temp:
        directory = Path(temp)
        dataset = directory / "projection.jsonl"
        report = directory / "projection.report.json"
        dataset_bytes = []
        report_bytes = []
        for _ in (1, 2):
            build_projection(dataset, report, OUTPUT_PROJECTION_CURATIONS)
            dataset_bytes.append(dataset.read_bytes())
            report_bytes.append(report.read_bytes())
        parsed = json.loads(report_bytes[0])
        normalized = dict(parsed)
        normalized["output"] = "<projection-output>"
        normalized_bytes = (json.dumps(normalized, indent=2, sort_keys=True) + "\n").encode()
        return {
            "dataset_equal": dataset_bytes[0] == dataset_bytes[1],
            "dataset_sha256": hashlib.sha256(dataset_bytes[0]).hexdigest(),
            "report_equal": report_bytes[0] == report_bytes[1],
            "report_normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
            "rows": dataset_bytes[0].count(b"\n"),
            "eval_fact_count": parsed["eval_fact_count"],
        }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".v128-projection-", dir=OUT_DIR) as temp:
        directory = Path(temp)
        baseline = directory / "v128.jsonl"
        baseline_report = directory / "v128.report.json"
        build_projection(baseline, baseline_report, PRIOR_PROJECTION_CURATIONS)
        rows = read_jsonl(baseline)
        if len(rows) != 515 or file_sha256(baseline) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v128 projection drift")

    write_jsonl(AUDIT, [])
    write_jsonl(CURATION, [])
    observed = observation()
    if not observed["dataset_equal"] or not observed["report_equal"] or observed["rows"] != 515 or observed["eval_fact_count"] != 612:
        raise ValueError("v129 deterministic projection drift")
    if observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v129 output hash drift")

    report = {
        "active_baseline": {
            "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784, "sha256": file_sha256(ACTIVE_DATASET)},
            "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)), "sha256": file_sha256(ACTIVE_REPORT)},
            "curation": [{"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
        },
        "audit": {
            "by_decision": {"drop": 0, "edit": 0, "keep": 0},
            "by_reason": {},
            "path": str(AUDIT.relative_to(ROOT)),
            "rows": 0,
            "sha256": file_sha256(AUDIT),
        },
        "frozen_prior_decision_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in prior_decision_artifacts()
        ],
        "isolated_build_projection": {
            "active_after_context_merit_v128": 477,
            "active_after_this_tranche": 477,
            "automated_projection_runs": 2,
            "build_script": "build_curated_qa.py",
            "determinism_comparison_scope": "identical inputs, curation chain, and output/report paths",
            "hermetic_bridge_label": BRIDGE_INPUT_LABEL,
            "new_drops_applied": 0,
            "new_edits_applied": 0,
            "output_rows": observed["rows"],
            "output_sha256": observed["dataset_sha256"],
            "prior_pending_addition_fact_ids_preserved": 35,
            "projection_report_normalized_sha256": observed["report_normalized_sha256"],
            "projection_strategy": "invocation-unique v117 bridge input with deterministic report-path canonicalization",
            "repeat_dataset_byte_identical": observed["dataset_equal"],
            "repeat_projection_report_byte_identical": observed["report_equal"],
            "reviewed_keep_fact_ids_preserved": 0,
            "sealed_eval_fact_count_reported_by_tooling": observed["eval_fact_count"],
            "unexpected_fact_ids": 0,
        },
        "new_pending_curation": {
            "by_action": {},
            "decisions": 0,
            "edit_support_types": {"extractive": 0, "manual_paraphrase": 0},
            "path": str(CURATION.relative_to(ROOT)),
            "sha256": file_sha256(CURATION),
        },
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "schema": "context-merit-audit-report-v129",
        "sealed_evaluation_policy": {
            "automated_collision_tool": "build_curated_qa.py",
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-id collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "selection": {
            "active_rows": 515,
            "projected_baseline": PROJECTED_SELECTION_BASELINE,
            "ranking": {
                "candidate_rule": "hermetic second-stage bridge regression; no content rows selected",
                "score": "not applicable",
                "tie_break": "not applicable",
            },
            "rows_selected": 0,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
