#!/usr/bin/env python3
"""Freeze five train-derived conflict-unit-disjoint shadow folds.

The source is the content-addressed v412 training projection.  No external
evaluation, OOD, holdout, judge, or benchmark artifact is read.  Connected
components conservatively join rows that share a document, normalized URL,
raw lineage identity, or the frozen V13 lexical-semantic cluster.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import eggroll_es_train_panel_sampler_v13 as panel_rules
import run_sft_train_only_control_v36a as runtime


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT / "experiments/sft_controls/v36a_v412/train_v412.jsonl"
).resolve()
OUTPUT_DIR = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412"
).resolve()
MANIFEST = OUTPUT_DIR / "manifest_v37a.json"
SOURCE_SHA256 = "44a5482e49073d23a35bdc4c574d6c52c9e8d7f946559dfa722dda16eec5882b"
SOURCE_ROWS = 531
FOLD_COUNT = 5
MASTER_SEED = "specialist-v37a-train-shadow-conflict-folds-20260715"
EXPECTED_CONFLICT_UNITS = 259
EXPECTED_MANIFEST_CONTENT_SHA256 = (
    "3fcc2820e8dffe6a21198d0520365aace049735ac84bda179ea44bc8ad0881eb"
)
PRE_COMMITMENT_MANIFEST_FILE_SHA256 = (
    "4cf8d271b928cb4540ccdd96d0a5c0cc4971b31651a43e526f8df16249811332"
)


def normalize_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if not parsed.scheme or not parsed.netloc:
        return str(value).strip()
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in {"si", "fbclid", "gclid"}
    ]
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((
        parsed.scheme.lower(), parsed.netloc.lower(), path,
        urlencode(sorted(query)), "",
    ))


def row_urls(row: dict) -> set[str]:
    scalar = (
        "url", "evidence_url", "canonical_url", "supplied_url",
        "original_ropetopia_url", "title_evidence_url", "url_evidence_url",
        "canonical_resource_url",
    )
    arrays = ("urls", "canonical_urls", "supplied_urls")
    values = {row[key] for key in scalar if row.get(key)}
    values.update(
        url for key in arrays for url in row.get(key, ()) if url
    )
    return {normalize_url(value) for value in values}


def row_lineage_identities(row: dict) -> set[str]:
    lineage = row.get("source_lineage") or {}
    return {
        f"{key}:{json.dumps(lineage[key], sort_keys=True)}"
        for key in ("raw", "raw_document", "raw_successor_document")
        if lineage.get(key)
    }


class DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def row_sha256(row: dict) -> str:
    return hashlib.sha256(json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def build_conflict_units(rows: list[dict]) -> list[dict]:
    semantic_ids = panel_rules.build_semantic_clusters(rows)
    disjoint = DisjointSet(len(rows))
    first = {}

    def connect(identifier: str, index: int) -> None:
        if identifier in first:
            disjoint.union(index, first[identifier])
        else:
            first[identifier] = index

    for index, (row, semantic_id) in enumerate(zip(rows, semantic_ids)):
        connect(f"document:{row['document_sha256']}", index)
        for url in row_urls(row):
            connect(f"url:{url}", index)
        for lineage_identity in row_lineage_identities(row):
            connect(f"lineage:{lineage_identity}", index)
        connect(f"semantic:{semantic_id}", index)

    components = defaultdict(list)
    for index in range(len(rows)):
        components[disjoint.find(index)].append(index)
    units = []
    for indices in components.values():
        indices = sorted(indices)
        stratum_counts = Counter(
            panel_rules.classify_stratum(rows[index]) for index in indices
        )
        dominant = max(
            panel_rules.STRATA,
            key=lambda name: (
                stratum_counts[name], panel_rules._TIE_PRIORITY[name]
            ),
        )
        identity = runtime.canonical_sha256({
            "row_sha256": sorted(row_sha256(rows[index]) for index in indices),
        })
        units.append({
            "identity_sha256": identity,
            "indices": indices,
            "rows": len(indices),
            "stratum": dominant,
            "row_stratum_counts": {
                name: stratum_counts[name] for name in panel_rules.STRATA
            },
        })
    units.sort(key=lambda item: item["identity_sha256"])
    return units


def assign_folds(units: list[dict]) -> list[list[dict]]:
    folds = [[] for _ in range(FOLD_COUNT)]
    for stratum in panel_rules.STRATA:
        candidates = [unit for unit in units if unit["stratum"] == stratum]
        candidates.sort(key=lambda unit: hashlib.sha256(
            f"{MASTER_SEED}\0{stratum}\0{unit['identity_sha256']}".encode()
        ).digest())
        for index, unit in enumerate(candidates):
            folds[index % FOLD_COUNT].append(unit)
    for fold in folds:
        fold.sort(key=lambda item: item["identity_sha256"])
    return folds


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def identity_set(rows: list[dict], domain: str, semantic_ids: dict[str, str]) -> set[str]:
    if domain == "document_sha256":
        return {row["document_sha256"] for row in rows}
    if domain == "normalized_url":
        return {value for row in rows for value in row_urls(row)}
    if domain == "raw_lineage":
        return {
            value for row in rows for value in row_lineage_identities(row)
        }
    if domain == "semantic_cluster":
        return {semantic_ids[row_sha256(row)] for row in rows}
    raise ValueError(f"unsupported edge domain: {domain}")


def _atomic_exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def construct() -> tuple[dict, dict[Path, bytes]]:
    if runtime.file_sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("v37a frozen v412 source changed")
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    if len(rows) != SOURCE_ROWS:
        raise RuntimeError("v37a frozen v412 row count changed")
    units = build_conflict_units(rows)
    if len(units) != EXPECTED_CONFLICT_UNITS:
        raise RuntimeError("v37a conservative conflict-unit count changed")
    folds = assign_folds(units)
    fold_by_unit = {
        unit["identity_sha256"]: fold_index
        for fold_index, fold in enumerate(folds) for unit in fold
    }
    if sorted(
        unit["identity_sha256"] for fold in folds for unit in fold
    ) != sorted(unit["identity_sha256"] for unit in units):
        raise RuntimeError("v37a fold assignment is not a partition")
    semantic_values = panel_rules.build_semantic_clusters(rows)
    semantic_ids = {
        row_sha256(row): semantic_id
        for row, semantic_id in zip(rows, semantic_values)
    }
    payloads: dict[Path, bytes] = {}
    fold_records = []
    all_dev_rows = Counter()
    for fold_index, dev_units in enumerate(folds):
        dev_unit_ids = {unit["identity_sha256"] for unit in dev_units}
        dev_indices = {
            index for unit in dev_units for index in unit["indices"]
        }
        train_units = [
            unit for unit in units if unit["identity_sha256"] not in dev_unit_ids
        ]
        train_indices = set(range(len(rows))) - dev_indices
        if not dev_indices or not train_indices or dev_indices & train_indices:
            raise RuntimeError("v37a train/dev row partition changed")
        if dev_unit_ids & {
            unit["identity_sha256"] for unit in train_units
        }:
            raise RuntimeError("v37a conflict unit crossed a fold boundary")
        for index in dev_indices:
            all_dev_rows[index] += 1
        train_rows = [rows[index] for index in sorted(train_indices)]
        dev_rows = [rows[index] for index in sorted(dev_indices)]
        train_path = OUTPUT_DIR / f"fold_{fold_index}_train.jsonl"
        dev_path = OUTPUT_DIR / f"fold_{fold_index}_shadow_dev.jsonl"
        train_payload = jsonl_bytes(train_rows)
        dev_payload = jsonl_bytes(dev_rows)
        payloads[train_path] = train_payload
        payloads[dev_path] = dev_payload
        edge_intersections = {
            domain: len(
                identity_set(train_rows, domain, semantic_ids)
                & identity_set(dev_rows, domain, semantic_ids)
            )
            for domain in (
                "document_sha256", "normalized_url", "raw_lineage",
                "semantic_cluster",
            )
        }
        if any(edge_intersections.values()):
            raise RuntimeError("v37a component edge crossed a fold boundary")
        fold_records.append({
            "fold": fold_index,
            "train": {
                "path": str(train_path),
                "rows": len(train_rows),
                "sha256": bytes_sha256(train_payload),
                "conflict_units": len(train_units),
                "ordered_row_identity_sha256": runtime.canonical_sha256([
                    row_sha256(row) for row in train_rows
                ]),
                "ordered_unit_identity_sha256": runtime.canonical_sha256([
                    unit["identity_sha256"] for unit in train_units
                ]),
            },
            "shadow_dev": {
                "path": str(dev_path),
                "rows": len(dev_rows),
                "sha256": bytes_sha256(dev_payload),
                "conflict_units": len(dev_units),
                "ordered_row_identity_sha256": runtime.canonical_sha256([
                    row_sha256(row) for row in dev_rows
                ]),
                "ordered_unit_identity_sha256": runtime.canonical_sha256([
                    unit["identity_sha256"] for unit in dev_units
                ]),
            },
            "shadow_dev_unit_strata": {
                name: sum(unit["stratum"] == name for unit in dev_units)
                for name in panel_rules.STRATA
            },
            "shadow_dev_row_strata": {
                name: sum(
                    panel_rules.classify_stratum(row) == name for row in dev_rows
                ) for name in panel_rules.STRATA
            },
            "train_dev_conflict_unit_intersection": 0,
            "train_dev_edge_identity_intersections": edge_intersections,
        })
    if all_dev_rows != Counter({index: 1 for index in range(len(rows))}):
        raise RuntimeError("v37a each row must appear in exactly one shadow dev fold")
    result = {
        "schema": "specialist-train-shadow-conflict-folds-v37a",
        "status": "frozen_train_derived_not_external_evaluation",
        "source": {
            "path": str(SOURCE),
            "rows": SOURCE_ROWS,
            "sha256": SOURCE_SHA256,
        },
        "policy": {
            "fold_count": FOLD_COUNT,
            "master_seed": MASTER_SEED,
            "conflict_units": len(units),
            "component_edges": [
                "shared document_sha256", "shared normalized URL",
                "shared raw/raw_document/raw_successor_document lineage",
                "shared frozen V13 lexical-semantic cluster",
            ],
            "assignment": (
                "within each dominant stratum, keyed permutation then round-robin "
                "assignment across five folds"
            ),
            "external_evaluation_ood_holdout_or_benchmark_opened": False,
        },
        "conflict_unit_strata": {
            name: sum(unit["stratum"] == name for unit in units)
            for name in panel_rules.STRATA
        },
        "content_free_unit_commitments": [
            {
                "unit_identity_sha256": unit["identity_sha256"],
                "row_count": unit["rows"],
                "dominant_stratum": unit["stratum"],
                "fold": fold_by_unit[unit["identity_sha256"]],
                "row_sha256": sorted(
                    row_sha256(rows[index]) for index in unit["indices"]
                ),
            }
            for unit in units
        ],
        "folds": fold_records,
        "coverage": {
            "each_source_row_shadow_dev_count": 1,
            "each_source_row_train_count": FOLD_COUNT - 1,
            "all_fold_train_rows": sum(item["train"]["rows"] for item in fold_records),
            "all_fold_shadow_dev_rows": sum(
                item["shadow_dev"]["rows"] for item in fold_records
            ),
        },
        "selection_firewall": {
            "allowed": [
                "fold-train optimization", "frozen shadow-dev aggregate metrics"
            ],
            "forbidden": [
                "external validation", "OOD", "holdout", "judge feedback",
                "benchmark feedback", "moving a conflict unit between folds",
                "training or recipe selection on folds 0, 1, 2, or 4 before the "
                "terminal fold-3 score",
            ],
            "confirmatory_fold": 3,
            "confirmatory_fold_selection_rule": (
                "lowest fold index with train_rows divisible by 28, at least 50 "
                "shadow-dev conflict units, and zero edge intersections"
            ),
            "shadow_dev_access": (
                "exactly once after both SFT and ES recipes and states are sealed; "
                "never an HPO, dataset, or future-recipe selection surface"
            ),
        },
    }
    result["content_sha256_before_self_field"] = runtime.canonical_sha256(result)
    if (
        EXPECTED_MANIFEST_CONTENT_SHA256 != "PENDING"
        and result["content_sha256_before_self_field"]
        != EXPECTED_MANIFEST_CONTENT_SHA256
    ):
        raise RuntimeError("v37a frozen shadow-fold manifest changed")
    manifest_payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    payloads[MANIFEST] = manifest_payload
    return result, payloads


def build() -> dict:
    """Pure construction used by tests; never writes an artifact."""
    return construct()[0]


def verify_existing(payloads: dict[Path, bytes]) -> None:
    for path, expected in payloads.items():
        if not path.is_file() or path.read_bytes() != expected:
            raise RuntimeError(f"v37a frozen artifact verification failed: {path}")


def write_new(payloads: dict[Path, bytes]) -> None:
    existing = [str(path) for path in payloads if path.exists()]
    if existing:
        raise RuntimeError(f"v37a refuses to overwrite artifacts: {existing}")
    for path, payload in payloads.items():
        _atomic_exclusive_write(path, payload)


def migrate_pre_commitment_manifest(payloads: dict[Path, bytes]) -> None:
    """One-way migration after verifying every immutable split byte."""
    verify_existing({path: value for path, value in payloads.items() if path != MANIFEST})
    if runtime.file_sha256(MANIFEST) != PRE_COMMITMENT_MANIFEST_FILE_SHA256:
        raise RuntimeError("v37a pre-commitment manifest identity changed")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{MANIFEST.name}.migrate-", dir=MANIFEST.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payloads[MANIFEST])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, MANIFEST)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=(
            "verify-existing", "write-new", "migrate-pre-commitment-manifest",
        ),
        default="verify-existing",
    )
    arguments = parser.parse_args()
    result, payloads = construct()
    if arguments.mode == "write-new":
        write_new(payloads)
    elif arguments.mode == "migrate-pre-commitment-manifest":
        migrate_pre_commitment_manifest(payloads)
    else:
        verify_existing(payloads)
    print(MANIFEST)
    print(result["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
