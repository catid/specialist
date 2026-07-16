#!/usr/bin/env python3
"""Seal V48B's deterministic 64-unit subset from numeric base evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lora_es_generation_boundary_sampling_v48a as boundary
import run_lora_es_base_generation_evidence_v48b as evidence_runtime
import run_lora_es_multi_anchor_v43i as v43i


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v48b_train_generation_boundary_subset.json"
).resolve()


def _read_sealed(path: Path, file_sha: str, content_sha: str) -> dict:
    if v43i.v40a.file_sha256(path) != file_sha:
        raise RuntimeError(f"v48b subset input file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or v43i.v40a.canonical_sha256(compact) != content_sha
    ):
        raise RuntimeError(f"v48b subset input content changed: {path}")
    return value


def seal_subset_v48b(
    evidence_path: Path, evidence_file_sha: str, evidence_content_sha: str,
    report_path: Path, report_file_sha: str, report_content_sha: str,
) -> dict:
    evidence = _read_sealed(
        evidence_path, evidence_file_sha, evidence_content_sha
    )
    report = _read_sealed(report_path, report_file_sha, report_content_sha)
    membership = _read_sealed(
        evidence_runtime.MEMBERSHIP,
        evidence_runtime.EXPECTED_MEMBERSHIP_SHA256,
        evidence_runtime.EXPECTED_MEMBERSHIP_CONTENT_SHA256,
    )
    gpu = report.get("gpu_activity", {})
    if (
        evidence.get("schema")
        != "train-only-four-actor-base-generation-evidence-v48a"
        or evidence.get("revision") != "v48b"
        or evidence.get("status") != "complete_before_population"
        or evidence.get("row_count") != 448
        or evidence.get("actor_count") != 4
        or evidence.get("membership_file_sha256")
        != evidence_runtime.EXPECTED_MEMBERSHIP_SHA256
        or evidence.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or evidence.get("selection_or_population_opened") is not False
        or evidence.get("protected_semantics_opened") is not False
        or evidence.get("shadow_ood_holdout_or_benchmark_opened") is not False
        or report.get("schema")
        != "matched-lora-es-base-generation-evidence-report-v48b"
        or report.get("status") != "complete_train_only_evidence_sealed"
        or report.get("evidence", {}).get("file_sha256") != evidence_file_sha
        or report.get("evidence", {}).get("content_sha256")
        != evidence_content_sha
        or gpu.get("all_four_attributed_positive") is not True
        or len(gpu.get("by_gpu", {})) != 4
        or report.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or report.get("protected_semantics_opened") is not False
        or report.get("shadow_ood_holdout_or_benchmark_opened") is not False
    ):
        raise RuntimeError("v48b base evidence/report seal changed")
    evidence_by_row = {item["row_sha256"]: item for item in evidence["rows"]}
    canonical_rows = [item["row_sha256"] for item in membership["items"]]
    if set(evidence_by_row) != set(canonical_rows) or len(evidence_by_row) != 448:
        raise RuntimeError("v48b evidence/membership row coverage changed")
    for member in membership["items"]:
        row = evidence_by_row[member["row_sha256"]]
        if (
            row.get("unit_identity_sha256") != member["unit_identity_sha256"]
            or row.get("row_count") != member["row_count"]
        ):
            raise RuntimeError("v48b evidence conflict-unit identity changed")
    bundle = {
        "row_sha256": canonical_rows,
        "unit_membership_v48a": [{
            "row_sha256": item["row_sha256"],
            "unit_identity_sha256": item["unit_identity_sha256"],
            "row_count": item["row_count"],
        } for item in membership["items"]],
        "train_bundle_content_sha256": v43i.TRAIN_BUNDLE_SHA256,
    }
    subset = boundary.build_fragile_subset_v48a(bundle, evidence)
    result = {
        "schema": "sealed-train-generation-boundary-subset-v48b",
        "status": "complete_before_population_launch",
        "source": {
            "evidence_path": str(Path(evidence_path).resolve()),
            "evidence_file_sha256": evidence_file_sha,
            "evidence_content_sha256": evidence_content_sha,
            "evidence_report_path": str(Path(report_path).resolve()),
            "evidence_report_file_sha256": report_file_sha,
            "evidence_report_content_sha256": report_content_sha,
            "membership_file_sha256": (
                evidence_runtime.EXPECTED_MEMBERSHIP_SHA256
            ),
            "membership_content_sha256": (
                evidence_runtime.EXPECTED_MEMBERSHIP_CONTENT_SHA256
            ),
        },
        "subset": subset,
        "selected_rows": subset["selected_rows"],
        "selected_conflict_units": subset["selected_conflict_units"],
        "request_order_sha256": subset["request_order_sha256"],
        "common_random_generation_params": dict(
            boundary.GENERATION_PARAMS_V48A
        ),
        "question_answer_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
        "gpu_or_model_accessed": False,
        "current_fixed_holdout_cycle_eligible": False,
    }
    result["content_sha256_before_self_field"] = v43i.v40a.canonical_sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--evidence-content-sha256", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--report-sha256", required=True)
    parser.add_argument("--report-content-sha256", required=True)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = seal_subset_v48b(
        Path(args.evidence).resolve(), args.evidence_sha256,
        args.evidence_content_sha256, Path(args.report).resolve(),
        args.report_sha256, args.report_content_sha256,
    )
    v43i.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": v43i.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "selected_conflict_units": value["selected_conflict_units"],
        "request_order_sha256": value["request_order_sha256"],
        "gpu_or_model_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
