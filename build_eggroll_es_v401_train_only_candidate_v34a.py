#!/usr/bin/env python3
"""Freeze curator snapshot v401 by replaying only train-side edit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import build_curated_qa


ROOT = Path(__file__).resolve().parent
SOURCE_CANDIDATE = ROOT / (
    "experiments/eggroll_es_hpo/dataset_candidates/v389/"
    "train_qa_context_merit_v389.jsonl"
)
SOURCE_MANIFEST = SOURCE_CANDIDATE.parent / "manifest.json"
SOURCE_CANDIDATE_SHA256 = (
    "4b6da77e7e1ae3d1145b3f2d29c7774b6aad2b4cb520fcea9a48af93d4322388"
)
SOURCE_MANIFEST_FILE_SHA256 = (
    "a7d096e7345b54c986ca210a6153cb22bde0e6ef05b94b530ffb9cf260139065"
)
SOURCE_MANIFEST_CONTENT_SHA256 = (
    "7eb3b179b79ee499c2a1ca7676b9b2a7a4122577c42786432250761afd3bed8f"
)
SOURCE_FREEZE_COMMIT = "c54cbf4cdea670f4044a6dc5fb035eb20face83c"
SOURCE_ROWS = 531

CURATOR_COMMIT = "d7abea3540bd1cb43d725ec94772385821d2cee4"
CURATOR_REPORT = ROOT / (
    "data/manual_reviews/context_merit_audit_v401/"
    "report_context_merit_v401.json"
)
CURATOR_REPORT_SHA256 = (
    "e5d10d2b570092de95f68967461bac2512d1742f1b1053a186d20c01238b5478"
)
OUTPUT_CANDIDATE = ROOT / (
    "experiments/eggroll_es_hpo/dataset_candidates/v401/"
    "train_qa_context_merit_v401.jsonl"
)
OUTPUT_MANIFEST = OUTPUT_CANDIDATE.parent / "manifest.json"
OUTPUT_CANDIDATE_SHA256 = (
    "8e29826dd389171c69f5eb6f43781f900345974c3d4d11274268e86c6145693b"
)
OUTPUT_ROWS = 531

CURATION_SHA256_BY_VERSION = {
    390: "672d8679a4506959e92b4ef28806861699d2e7f8752a6935a70052976a8f115a",
    391: "93c3cac92ee1e82e84b8d46631dd62e830319d3e214983cc8342318330a42ea0",
    392: "a13b3a7d59ee7f1d2214292c75ec92740c94ebf824a5aea4e9a583026dd54999",
    393: "773f92299b1553339a63e16fb246edb09354c775afa528821af2f077b50a7057",
    394: "c21bad881f58109ebd628501206a56b04b3d210af9209ef316ca9e8a59b135d0",
    395: "1c8d01bc8600396611ffeadf46375609f3029f215dbc2ce5378ac510e3f9b9b5",
    396: "a9c4e77324d2ffed39d6f4358b51c96b0816dbe28a7b9710d398db37c0d00e80",
    397: "f06a4026035c0b2cc27b57a4b0a50a388486a70afd224d8ba1b20dd54399e683",
    398: "220042942e30874b986d676dd190ee814685374f746d79e1c223129bdab94a54",
    399: "43fd4e0091775937a7e8698d5e26c9f1809ff0eda755a6f8e9edeae458b2f498",
    400: "11dc6db870e0a7c022c7833bc6374ece4c6f5502fac0f58673e138707dbf410f",
    401: "d8f490894cc2f77711196a7c4832dd8d862edea120b8f7cf0d45df3905351a20",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def curation_path(version: int) -> Path:
    return ROOT / (
        f"data/manual_reviews/context_merit_audit_v{version}/"
        f"pending_curation_context_merit_v{version}.jsonl"
    )


def row_count(path: Path) -> int:
    with Path(path).open("rb") as source:
        return sum(1 for line in source if line.strip())


def validate_inputs() -> list[dict]:
    if (
        file_sha256(SOURCE_CANDIDATE) != SOURCE_CANDIDATE_SHA256
        or row_count(SOURCE_CANDIDATE) != SOURCE_ROWS
        or file_sha256(SOURCE_MANIFEST) != SOURCE_MANIFEST_FILE_SHA256
        or file_sha256(CURATOR_REPORT) != CURATOR_REPORT_SHA256
    ):
        raise RuntimeError("V401 frozen train-only source identity changed")
    source_manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    if (
        source_manifest.get("content_sha256_before_self_field")
        != SOURCE_MANIFEST_CONTENT_SHA256
        or canonical_sha256({
            key: item for key, item in source_manifest.items()
            if key != "content_sha256_before_self_field"
        }) != SOURCE_MANIFEST_CONTENT_SHA256
    ):
        raise RuntimeError("V401 source manifest seal changed")
    report = json.loads(CURATOR_REPORT.read_text(encoding="utf-8"))
    projection = report.get("isolated_build_projection", {})
    policy = report.get("sealed_evaluation_policy", {})
    if (
        report.get("schema") != "context-merit-audit-report-v401"
        or projection.get("output_rows") != OUTPUT_ROWS
        or projection.get("output_sha256") != OUTPUT_CANDIDATE_SHA256
        or projection.get("repeat_dataset_byte_identical") is not True
        or policy.get("manual_worker_opened_eval_or_heldout_content") is not False
        or policy.get("manual_worker_received_eval_or_heldout_content") is not False
    ):
        raise RuntimeError("V401 aggregate curator report changed")
    records = []
    for version, expected in CURATION_SHA256_BY_VERSION.items():
        path = curation_path(version)
        if file_sha256(path) != expected:
            raise RuntimeError(f"V401 curation identity changed at v{version}")
        decisions = build_curated_qa.load_curation([path])[0]
        if len(decisions) != 3 or any(
            item.get("action") != "edit" for item in decisions.values()
        ):
            raise RuntimeError(f"V401 train-only edit contract changed at v{version}")
        records.append({
            "version": version,
            "relative_path": str(path.relative_to(ROOT)),
            "file_sha256": expected,
            "edit_count": 3,
        })
    return records


def build_candidate(output: Path, manifest_path: Path) -> dict:
    """Replay v390-v401 with an explicitly empty collision-fact set."""
    output = Path(output).resolve()
    manifest_path = Path(manifest_path).resolve()
    records = validate_inputs()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v401-train-only-replay-", dir=output.parent,
    ) as temporary:
        directory = Path(temporary)
        current = SOURCE_CANDIDATE
        for record in records:
            version = record["version"]
            next_output = directory / f"v{version}.jsonl"
            report_path = directory / f"v{version}.report.json"
            summary = build_curated_qa.merge(
                [current], next_output, report_path, frozenset(),
                [curation_path(version)],
            )
            if (
                summary.get("counts", {}).get("output") != OUTPUT_ROWS
                or summary.get("counts", {}).get("excluded") != 0
                or summary.get("curation", {}).get("by_action") != {"edit": 3}
            ):
                raise RuntimeError(f"V401 train-only replay changed at v{version}")
            current = next_output
        if row_count(current) != OUTPUT_ROWS or file_sha256(current) != OUTPUT_CANDIDATE_SHA256:
            raise RuntimeError("V401 exact candidate output identity changed")
        temporary_output = output.with_suffix(output.suffix + ".tmp")
        temporary_output.write_bytes(current.read_bytes())
        os.replace(temporary_output, output)
    manifest = {
        "schema": "eggroll-es-v401-train-only-candidate-manifest-v34a",
        "curator_snapshot": {
            "version": 401,
            "commit": CURATOR_COMMIT,
            "report_relative_path": str(CURATOR_REPORT.relative_to(ROOT)),
            "report_file_sha256": CURATOR_REPORT_SHA256,
        },
        "source_candidate": {
            "version": 389,
            "freeze_commit": SOURCE_FREEZE_COMMIT,
            "relative_path": str(SOURCE_CANDIDATE.relative_to(ROOT)),
            "file_sha256": SOURCE_CANDIDATE_SHA256,
            "manifest_file_sha256": SOURCE_MANIFEST_FILE_SHA256,
            "manifest_content_sha256": SOURCE_MANIFEST_CONTENT_SHA256,
            "rows": SOURCE_ROWS,
        },
        "train_only_replay": {
            "versions": [390, 401],
            "curation_artifacts": records,
            "total_edits": 36,
            "total_drops_or_additions": 0,
            "collision_fact_set_count": 0,
            "validation_heldout_ood_or_benchmark_file_opened": False,
        },
        "candidate": {
            "relative_path": str(OUTPUT_CANDIDATE.relative_to(ROOT)),
            "file_sha256": file_sha256(output),
            "rows": row_count(output),
        },
        "contains_row_content": False,
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_CANDIDATE)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_MANIFEST)
    args = parser.parse_args(argv)
    value = build_candidate(args.output, args.manifest)
    print(json.dumps({
        "candidate_file_sha256": file_sha256(args.output),
        "candidate_rows": row_count(args.output),
        "manifest_file_sha256": file_sha256(args.manifest),
        "manifest_content_sha256": value["content_sha256_before_self_field"],
        "validation_heldout_ood_or_benchmark_file_opened": False,
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
