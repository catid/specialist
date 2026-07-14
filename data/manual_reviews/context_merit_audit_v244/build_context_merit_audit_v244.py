#!/usr/bin/env python3
"""Remove two duplicate Rope365 handcuff-knot risk answers."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.setrecursionlimit(5000)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
V243_DIR = DATA / "manual_reviews/context_merit_audit_v243"
sys.path[:0] = [str(ROOT), str(V243_DIR)]
import build_context_merit_audit_v243 as previous

OUT_DIR = Path(__file__).resolve().parent
AUDIT = OUT_DIR / "context_merit_audit_v244.jsonl"
CURATION = OUT_DIR / "pending_curation_context_merit_v244.jsonl"
REPORT = OUT_DIR / "report_context_merit_v244.json"
REVIEWER = "codex-context-merit-audit-v244"
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
SQUASHED_PROJECTION_CURATIONS = previous.OUTPUT_PROJECTION_CURATIONS
SEALED_EVAL_PATHS = (
    DATA / "eval_qa.jsonl",
    DATA / "eval_qa_v2.jsonl",
    DATA / "eval_qa_v3.jsonl",
    DATA / "ood_qa.jsonl",
    DATA / "ood_qa_v3.jsonl",
)
BRIDGE_INPUT_LABEL = "<isolated-v243-projection>"
SOURCE = DATA / "raw/rope365_25f1b23eb40be00e.json"

SPECS = (
    {
        "fact_id": "fact-43a668fb705fa1cc2a6f",
        "active_index": 297,
        "markers": (
            "The handcuff knot is a popular type of knots to capture two limbs at once. Since it’s a type of slip knot, it comes with the risk that it may tighten when put directly on the body.",
        ),
        "decision": "drop",
        "reason_code": "drop_duplicate_handcuff_knot_risk_question",
        "reason": "This narrow risk-only row repeats the same source sentence already covered by a stronger active question that preserves both the handcuff knot's use and its tightening risk.",
    },
    {
        "fact_id": "fact-b3c1565b466e252d20f1",
        "active_index": 480,
        "markers": (
            "The handcuff knot is a popular type of knots to capture two limbs at once. Since it’s a type of slip knot, it comes with the risk that it may tighten when put directly on the body.",
        ),
        "decision": "drop",
        "reason_code": "drop_second_duplicate_handcuff_knot_risk_question",
        "reason": "This second risk-only paraphrase repeats the same source sentence and tightening warning already represented by the more informative use-and-risk Q&A.",
    },
)
EXPECTED_SELECTION = tuple(s["fact_id"] for s in SPECS)
PROJECTED_ACTIVE_INDICES = {s["fact_id"]: s["active_index"] for s in SPECS}
PROJECTED_SELECTION_BASELINE = {
    "description": "isolated corrected training projection through context-merit v243",
    "direct_rows_without_prior_curation": 1,
    "duplicate_handcuff_knot_rows_selected": 2,
    "rows": 500,
    "sha256": "0512c1aef1e40e66d31926a4f5d4639489c41fa3ee93ec31daa65d43572b2681",
}
EXPECTED_OUTPUT_SHA256 = "d002101b402cc3727cbd01c62e6d700917051732c011c53e32d8a22051729994"


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
    """Build through a byte-pinned v243 bridge and emit a stable report."""
    curations = tuple(curations)
    prefix_length = len(SQUASHED_PROJECTION_CURATIONS)
    if curations[:prefix_length] != SQUASHED_PROJECTION_CURATIONS:
        raise ValueError("v244 projection curation-prefix drift")
    later_curations = curations[prefix_length:]

    with tempfile.TemporaryDirectory(prefix=".v244-bridge-", dir=OUT_DIR) as temp:
        bridge_dir = Path(temp)
        upstream_dataset = bridge_dir / "v243.jsonl"
        upstream_report = bridge_dir / "v243.report.json"
        previous.build_projection(
            upstream_dataset,
            upstream_report,
            SQUASHED_PROJECTION_CURATIONS,
        )
        if file_sha256(upstream_dataset) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v244 bridge input drift")
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
            raise ValueError("v244 bridge provenance drift")
        parsed = _canonicalize_bridge_path(parsed, volatile_path)
        report.write_text(json.dumps(parsed, indent=2, sort_keys=True) + "\n")


def prior_decision_artifacts():
    out = []
    for version in range(1, 244):
        directory = DATA / "manual_reviews" / f"context_merit_audit_v{version}"
        out.extend(
            (
                directory / f"context_merit_audit_v{version}.jsonl",
                directory / f"pending_curation_context_merit_v{version}.jsonl",
                directory / f"report_context_merit_v{version}.json",
            )
        )
    return tuple(out)


def evidence(document, markers):
    support = []
    for marker in markers:
        matches = [line for line in document["text"].splitlines() if marker in line]
        if len(matches) != 1:
            raise ValueError(f"evidence drift: {marker}")
        support.append(matches[0])
    return "\n".join(support)


def observation():
    with tempfile.TemporaryDirectory(prefix=".v244-observation-", dir=OUT_DIR) as temp:
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
    with tempfile.TemporaryDirectory(prefix=".v243-projection-", dir=OUT_DIR) as temp:
        directory = Path(temp)
        baseline = directory / "v243.jsonl"
        baseline_report = directory / "v243.report.json"
        build_projection(baseline, baseline_report, PRIOR_PROJECTION_CURATIONS)
        rows = read_jsonl(baseline)
        if len(rows) != 500 or file_sha256(baseline) != PROJECTED_SELECTION_BASELINE["sha256"]:
            raise ValueError("v243 projection drift")
        by_fact = {row["fact_id"]: (index, row) for index, row in enumerate(rows, 1)}
        if {fact_id: by_fact[fact_id][0] for fact_id in EXPECTED_SELECTION} != PROJECTED_ACTIVE_INDICES:
            raise ValueError("v244 candidate drift")

    document = json.loads(SOURCE.read_text())
    audits = []
    curations = []
    for audit_index, spec in enumerate(SPECS, 1):
        active = by_fact[spec["fact_id"]][1]
        support = evidence(document, spec["markers"])
        if active["source"] != document["source"] or active["url"] != document["url"]:
            raise ValueError(f"{spec['fact_id']}: source lineage drift")
        audit = {
            "active_answer": active["answer"],
            "active_index": spec["active_index"],
            "active_question": active["question"],
            "audit_index": audit_index,
            "decision": spec["decision"],
            "document_sha256": active["document_sha256"],
            "fact_id": spec["fact_id"],
            "projection_lineage": {
                "active_index": spec["active_index"],
                "baseline_rows": 500,
                "baseline_sha256": PROJECTED_SELECTION_BASELINE["sha256"],
                "prior_context_merit_review": True,
            },
            "reason": spec["reason"],
            "reason_code": spec["reason_code"],
            "review_pass": "duplicate_handcuff_knot_risk_reaudit",
            "reviewed_at": REVIEWED_AT,
            "reviewer": REVIEWER,
            "risk_features": CORE.risk_features(active),
            "schema": "context-merit-audit-v244",
            "source": document["source"],
            "source_document": str(SOURCE.relative_to(ROOT)),
            "source_document_file_sha256": file_sha256(SOURCE),
            "source_support": "manual_paraphrase" if spec["decision"] == "edit" else "normalized_extractive",
            "support_evidence": support,
            "support_evidence_sha256": text_sha256(support),
            "url": active["url"],
        }
        if spec["decision"] == "edit":
            audit.update(
                edited_answer=spec["answer"],
                edited_question=spec["question"],
                paraphrase_rationale=spec["reason"],
            )
            curations.append(
                {
                    "action": "edit",
                    "answer": spec["answer"],
                    "document_sha256": active["document_sha256"],
                    "evidence": support,
                    "evidence_url": active["url"],
                    "expected_answer": active["answer"],
                    "expected_question": active["question"],
                    "fact_id": spec["fact_id"],
                    "paraphrase_rationale": spec["reason"],
                    "question": spec["question"],
                    "reason": spec["reason"],
                    "reason_code": spec["reason_code"],
                    "reviewed_at": REVIEWED_AT,
                    "reviewer": REVIEWER,
                    "source_lineage": active["source_lineage"],
                    "support_type": "manual_paraphrase",
                }
            )
        elif spec["decision"] == "drop":
            curations.append(
                {
                    "action": "drop",
                    "document_sha256": active["document_sha256"],
                    "evidence": support,
                    "evidence_url": active["url"],
                    "expected_answer": active["answer"],
                    "expected_question": active["question"],
                    "fact_id": spec["fact_id"],
                    "reason": spec["reason"],
                    "reason_code": spec["reason_code"],
                    "reviewed_at": REVIEWED_AT,
                    "reviewer": REVIEWER,
                    "source_lineage": active["source_lineage"],
                }
            )
        audits.append(audit)

    write_jsonl(AUDIT, audits)
    write_jsonl(CURATION, curations)
    observed = observation()
    if not observed["dataset_equal"] or not observed["report_equal"] or observed["rows"] != 498 or observed["eval_fact_count"] != 612:
        raise ValueError("v244 deterministic projection drift")
    if EXPECTED_OUTPUT_SHA256 != "PENDING" and observed["dataset_sha256"] != EXPECTED_OUTPUT_SHA256:
        raise ValueError("v244 output hash drift")
    report = {
        "active_baseline": {
            "dataset": {"path": str(ACTIVE_DATASET.relative_to(ROOT)), "rows": 784, "sha256": file_sha256(ACTIVE_DATASET)},
            "report": {"path": str(ACTIVE_REPORT.relative_to(ROOT)), "sha256": file_sha256(ACTIVE_REPORT)},
            "curation": [{"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in ACTIVE_CURATIONS],
        },
        "audit": {
            "by_decision": {"drop": 2, "edit": 0, "keep": 0},
            "by_reason": {spec["reason_code"]: 1 for spec in SPECS},
            "path": str(AUDIT.relative_to(ROOT)),
            "rows": 2,
            "sha256": file_sha256(AUDIT),
        },
        "frozen_prior_decision_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": file_sha256(path)} for path in prior_decision_artifacts()
        ],
        "isolated_build_projection": {
            "active_after_context_merit_v243": 462,
            "active_after_this_tranche": 460,
            "automated_projection_runs": 2,
            "build_script": "build_curated_qa.py",
            "determinism_comparison_scope": "identical inputs, curation chain, and output/report paths",
            "new_drops_applied": 2,
            "new_edits_applied": 0,
            "output_rows": observed["rows"],
            "output_sha256": observed["dataset_sha256"],
            "prior_pending_addition_fact_ids_preserved": 35,
            "projection_report_normalized_sha256": observed["report_normalized_sha256"],
            "repeat_dataset_byte_identical": observed["dataset_equal"],
            "repeat_projection_report_byte_identical": observed["report_equal"],
            "reviewed_keep_fact_ids_preserved": 0,
            "sealed_eval_fact_count_reported_by_tooling": observed["eval_fact_count"],
            "unexpected_fact_ids": 0,
        },
        "new_pending_curation": {
            "by_action": {"drop": 2},
            "decisions": 2,
            "edit_support_types": {"extractive": 0, "manual_paraphrase": 0},
            "path": str(CURATION.relative_to(ROOT)),
            "sha256": file_sha256(CURATION),
        },
        "projected_baseline": PROJECTED_SELECTION_BASELINE,
        "schema": "context-merit-audit-report-v244",
        "sealed_evaluation_policy": {
            "automated_collision_tool": "build_curated_qa.py",
            "automated_collision_tool_reads_sealed_content": True,
            "automated_read_scope": "fact-id collision exclusion and aggregate eval_fact_count reporting only",
            "manual_worker_opened_eval_or_heldout_content": False,
            "manual_worker_received_eval_or_heldout_content": False,
        },
        "selection": {
            "active_rows": 500,
            "projected_baseline": PROJECTED_SELECTION_BASELINE,
            "ranking": {
                "candidate_rule": "two narrow handcuff-knot risk questions whose warning is already covered by a stronger use-and-risk row",
                "score": "answer identity, source-sentence identity, semantic question overlap, and redundancy reduction",
                "tie_break": "active projection order",
            },
            "rows_selected": 2,
        },
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
