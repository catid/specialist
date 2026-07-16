#!/usr/bin/env python3
"""Print a transient blinded slice of V35A candidates for manual review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import run_eggroll_es_lagged_replay_calibration_v35a as runtime


REPORT_FILE_SHA256 = (
    "7c2c9e330b43d320bc08b4b030ae35b0533707da4eca1314a8c4938d2a423d99"
)
REPORT_CONTENT_SHA256 = (
    "c102fb72a5ebbd6dc800633fc1ddf303919d917aba389a70ed9d0db4e52f0c8c"
)
CRITERIA = (
    "eligible only if the question is clear and useful, the answer is "
    "factually supported by its training source, the answer directly answers "
    "the question, there is no template/protocol/control-token leakage, and "
    "no unsafe instruction or missing material safety context"
)


def load_report():
    if runtime.file_sha256(runtime.REPORT_PATH) != REPORT_FILE_SHA256:
        raise RuntimeError("V35A manual review report file changed")
    report = json.loads(runtime.REPORT_PATH.read_text(encoding="utf-8"))
    if (
        report.get("content_sha256_before_self_field")
        != REPORT_CONTENT_SHA256
        or runtime.canonical_sha256(runtime._without_self(report))
        != REPORT_CONTENT_SHA256
        or report.get("status")
        != "completed_calibration_pending_blinded_manual_review"
        or report.get("provisional_review_pool_count") != 87
        or report.get("manual_review_complete") is not False
    ):
        raise RuntimeError("V35A manual review report contract changed")
    runtime.assert_content_free(report)
    return report


def load_blinded_rows():
    report = load_report()
    preregistration = runtime.load_preregistration()
    manifest = runtime.load_manifest(preregistration)
    metadata = runtime.optimization_metadata(manifest)
    by_sha = {item["row_sha256"]: item for item in metadata}
    pool = report["provisional_review_pool"]
    pool_sha = {item["row_sha256"] for item in pool}
    if len(pool_sha) != len(pool) != 87 or not pool_sha.issubset(by_sha):
        raise RuntimeError("V35A provisional review identities changed")
    row_indices = {by_sha[row_sha]["row_index"]: row_sha for row_sha in pool_sha}
    source_path = Path(manifest["source"]["path"]).resolve()
    if runtime.file_sha256(source_path) != runtime.SOURCE_FILE_SHA256:
        raise RuntimeError("V35A blinded review source changed")
    selected = {}
    line_count = 0
    with source_path.open("rb") as source:
        for row_index, raw_line in enumerate(source):
            line_count += 1
            if row_index not in row_indices:
                continue
            row = json.loads(raw_line.decode("utf-8"))
            row_sha = row_indices[row_index]
            if runtime.sampler_v13.row_sha256(row) != row_sha:
                raise RuntimeError("V35A blinded review row identity changed")
            selected[row_sha] = row
    if line_count != runtime.SOURCE_ROWS or set(selected) != pool_sha:
        raise RuntimeError("V35A blinded review source coverage changed")
    order = runtime.deterministic_manual_review_order(pool)
    if runtime.canonical_sha256(order) != report[
        "deterministic_blinded_review_order_sha256"
    ]:
        raise RuntimeError("V35A blinded review order changed")
    result = []
    for review_position, row_sha in enumerate(order):
        row = selected[row_sha]
        result.append({
            "review_position": review_position,
            "row_sha256": row_sha,
            "question": row["question"],
            "answer": row["answer"],
            "source": row.get("source"),
            "url": row.get("url"),
            "source_text": row.get("text"),
            "eligibility_criteria": CRITERIA,
        })
    return result


def _parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if not 0 <= args.start < args.stop <= 87:
        raise ValueError("V35A blinded review slice must be within [0, 87)")
    rows = load_blinded_rows()[args.start:args.stop]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return rows


if __name__ == "__main__":
    main()
