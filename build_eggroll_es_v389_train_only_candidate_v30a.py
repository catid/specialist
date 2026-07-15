#!/usr/bin/env python3
"""Materialize curator snapshot v389 using train-only edit replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import build_curated_qa


ROOT = Path(__file__).resolve().parent
SOURCE_CANDIDATE_V364 = ROOT / (
    "experiments/eggroll_es_hpo/dataset_candidates/v364/"
    "train_qa_context_merit_v364.jsonl"
)
SOURCE_CANDIDATE_SHA256_V364 = (
    "874b77dd8ef988bb24d4b13999ddebc2068b053eab208def9bb1e23e7138c36a"
)
SOURCE_CANDIDATE_ROWS_V364 = 531
CURATOR_COMMIT_V389 = "2e8a6b7d02fbc77a2442f6790fe0f80f1bebc02e"
CURATOR_REPORT_V389 = ROOT / (
    "data/manual_reviews/context_merit_audit_v389/"
    "report_context_merit_v389.json"
)
CURATOR_REPORT_SHA256_V389 = (
    "809cbe853b24129c9d957bc20219a61c0c7ba14044c49e48f5541c71a946349f"
)
OUTPUT_CANDIDATE_V389 = ROOT / (
    "experiments/eggroll_es_hpo/dataset_candidates/v389/"
    "train_qa_context_merit_v389.jsonl"
)
OUTPUT_MANIFEST_V389 = OUTPUT_CANDIDATE_V389.parent / "manifest.json"
OUTPUT_CANDIDATE_SHA256_V389 = (
    "4b6da77e7e1ae3d1145b3f2d29c7774b6aad2b4cb520fcea9a48af93d4322388"
)
OUTPUT_CANDIDATE_ROWS_V389 = 531
CURATION_SHA256_BY_VERSION_V389 = {
    365: "21ad276ff41f2229291000fd9590e5f5bf53f38fe77994515c7a69444e2f5515",
    366: "e0036dd5595258580a8e524f905f5da0311fa60ae8bdbcd7f5a695d627a58598",
    367: "044d59619cd0b2de7fa170c8131462d357b760c6ca31ae88b6facd753b4d49ab",
    368: "70e2b31679e0096f90b20dfea5a3c40abfda4cd50ecfe4e85d1258144ea6a6b4",
    369: "f2d2d27702cd3937cd0f04180cb4a77eb3d747bc5773ce6a928e4e8a73762c52",
    370: "5482251022c60e3ccf5dece8aa51cb63753d9583fd6cc2c36e110583e198cf9c",
    371: "95b357977696ebfd81c80cb27da4e9774d26508338f6af7a36f8a6d25322d45c",
    372: "5a3636f80ce5c0487fc9c703cc73bc7185197999a78d49c58d8805cdb22e6768",
    373: "a3d8d4293c68f3df55401775d3476d8e309b2ad6d3a78be742b0eadaa23d738d",
    374: "db0e5d98008ac38a2df2b0767a7efd82fab97834e23dc486479b360865aca968",
    375: "c270e083d332b2ed5f5c067d45e07391f35c0276f17629cb199e4c68f7e6d248",
    376: "b9dfdfcab79fd37bc44ebd0b395c9d8f8eb535968d962522c65036b3b4cee07f",
    377: "16eca11c3e1f4fb7142b0006d396f2d0ca73a7be400a431e3d25f6d8cd4f2cfa",
    378: "327947a0d4471464457ce0a68d54e8fca4fb33053b227c4bdfb2242ef60b1c41",
    379: "138fc4d9a3ccc378d98b31cb4e173b697d460b5720184de885a97c67a8e9ade5",
    380: "88fa80ad33faf81628dfeac9f5ad24ea9d0f9cb96ca9e32e4e56276f5906b037",
    381: "26a9c3d7b9c52b1da9c527000554eb748add2424593d05f77a8b1891c6d8428d",
    382: "8f50f84437eb8ed31d0dee2dbc43cda9008f0047d8536709b32df94f66d48dc2",
    383: "d98c1f021f506df993ef561b0d1ebd2c373fa6929dcc4c75a016d6be10482cb1",
    384: "79bdb7a8260054b634006f79c148b9cfba43654d7b915eec10a3a6db7dc5d37f",
    385: "a4517d82059bf6ca35bca9da3715eefd8670fd001c7bcec663998132d8ef312f",
    386: "9cd792f367bfeb399cba21977f1ae0a7b6ab3191be7a9bffd7b6bd82c6014c47",
    387: "8ccadf822ef5a91943842c8286289f536cd9fe2dbcca335fe638e45576ab93ba",
    388: "b14918cce65f908600f03f30625564c95f2b77e9de3b03de0f1c19da8c80c8e9",
    389: "d09b7b0e19b12bd793a1e3bcece23114a12a53439330b0c3294c1c817e9b5888",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def curation_path_v389(version: int) -> Path:
    return ROOT / (
        f"data/manual_reviews/context_merit_audit_v{version}/"
        f"pending_curation_context_merit_v{version}.jsonl"
    )


def _row_count(path: Path) -> int:
    with Path(path).open("rb") as source:
        return sum(1 for line in source if line.strip())


def _validate_inputs_v389() -> list[dict]:
    if (
        file_sha256(SOURCE_CANDIDATE_V364) != SOURCE_CANDIDATE_SHA256_V364
        or _row_count(SOURCE_CANDIDATE_V364) != SOURCE_CANDIDATE_ROWS_V364
        or file_sha256(CURATOR_REPORT_V389) != CURATOR_REPORT_SHA256_V389
    ):
        raise RuntimeError("V389 frozen train-only source identity changed")
    report = json.loads(CURATOR_REPORT_V389.read_text(encoding="utf-8"))
    projection = report.get("isolated_build_projection", {})
    policy = report.get("sealed_evaluation_policy", {})
    if (
        report.get("schema") != "context-merit-audit-report-v389"
        or projection.get("output_rows") != OUTPUT_CANDIDATE_ROWS_V389
        or projection.get("output_sha256") != OUTPUT_CANDIDATE_SHA256_V389
        or projection.get("repeat_dataset_byte_identical") is not True
        or policy.get("manual_worker_opened_eval_or_heldout_content") is not False
        or policy.get("manual_worker_received_eval_or_heldout_content") is not False
    ):
        raise RuntimeError("V389 aggregate curator report changed")
    records = []
    for version, expected in CURATION_SHA256_BY_VERSION_V389.items():
        path = curation_path_v389(version)
        if file_sha256(path) != expected:
            raise RuntimeError(f"V389 curation identity changed at v{version}")
        decisions = build_curated_qa.load_curation([path])[0]
        if len(decisions) != 3 or any(
            item.get("action") != "edit" for item in decisions.values()
        ):
            raise RuntimeError(f"V389 train-only edit contract changed at v{version}")
        records.append({
            "version": version,
            "relative_path": str(path.relative_to(ROOT)),
            "file_sha256": expected,
            "edit_count": 3,
        })
    return records


def build_candidate_v389(output: Path, manifest_path: Path):
    """Replay train-only edits with an explicitly empty collision-fact set."""
    output = Path(output).resolve()
    manifest_path = Path(manifest_path).resolve()
    records = _validate_inputs_v389()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".v389-train-only-replay-", dir=output.parent,
    ) as temporary:
        directory = Path(temporary)
        current = SOURCE_CANDIDATE_V364
        for record in records:
            version = record["version"]
            next_output = directory / f"v{version}.jsonl"
            report_path = directory / f"v{version}.report.json"
            summary = build_curated_qa.merge(
                [current],
                next_output,
                report_path,
                frozenset(),
                [curation_path_v389(version)],
            )
            if (
                summary.get("counts", {}).get("output")
                != OUTPUT_CANDIDATE_ROWS_V389
                or summary.get("counts", {}).get("excluded") != 0
                or summary.get("curation", {}).get("by_action") != {"edit": 3}
            ):
                raise RuntimeError(f"V389 train-only replay changed at v{version}")
            current = next_output
        if (
            _row_count(current) != OUTPUT_CANDIDATE_ROWS_V389
            or file_sha256(current) != OUTPUT_CANDIDATE_SHA256_V389
        ):
            raise RuntimeError("V389 exact candidate output identity changed")
        temporary_output = output.with_suffix(output.suffix + ".tmp")
        temporary_output.write_bytes(current.read_bytes())
        os.replace(temporary_output, output)
    manifest = {
        "schema": "eggroll-es-v389-train-only-candidate-manifest-v30a",
        "curator_snapshot": {
            "version": 389,
            "commit": CURATOR_COMMIT_V389,
            "report_relative_path": str(CURATOR_REPORT_V389.relative_to(ROOT)),
            "report_file_sha256": CURATOR_REPORT_SHA256_V389,
        },
        "source_candidate": {
            "version": 364,
            "relative_path": str(SOURCE_CANDIDATE_V364.relative_to(ROOT)),
            "file_sha256": SOURCE_CANDIDATE_SHA256_V364,
            "rows": SOURCE_CANDIDATE_ROWS_V364,
        },
        "train_only_replay": {
            "versions": [365, 389],
            "curation_artifacts": records,
            "total_edits": 75,
            "total_drops_or_additions": 0,
            "collision_fact_set_count": 0,
            "validation_heldout_ood_or_benchmark_file_opened": False,
        },
        "candidate": {
            "relative_path": str(OUTPUT_CANDIDATE_V389.relative_to(ROOT)),
            "file_sha256": file_sha256(output),
            "rows": _row_count(output),
        },
        "contains_row_content": False,
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_CANDIDATE_V389)
    parser.add_argument("--manifest", type=Path, default=OUTPUT_MANIFEST_V389)
    args = parser.parse_args(argv)
    value = build_candidate_v389(args.output, args.manifest)
    print(json.dumps({
        "candidate_file_sha256": file_sha256(args.output),
        "candidate_rows": _row_count(args.output),
        "manifest_file_sha256": file_sha256(args.manifest),
        "manifest_content_sha256": value["content_sha256_before_self_field"],
        "validation_heldout_ood_or_benchmark_file_opened": False,
    }, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
