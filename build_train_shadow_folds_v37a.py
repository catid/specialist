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
import re
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
    "417ba17e0cbaad30ac8d60fa1ce58acdbbffbff1052695ae3e2005bd43504b5a"
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
        lineage = row.get("source_lineage") or {}
        for key in ("raw", "raw_document", "raw_successor_document"):
            if lineage.get(key):
                connect(
                    f"lineage:{key}:{json.dumps(lineage[key], sort_keys=True)}",
                    index,
                )
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def build() -> dict:
    if runtime.file_sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("v37a frozen v412 source changed")
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    if len(rows) != SOURCE_ROWS:
        raise RuntimeError("v37a frozen v412 row count changed")
    units = build_conflict_units(rows)
    if len(units) != EXPECTED_CONFLICT_UNITS:
        raise RuntimeError("v37a conservative conflict-unit count changed")
    folds = assign_folds(units)
    if sorted(
        unit["identity_sha256"] for fold in folds for unit in fold
    ) != sorted(unit["identity_sha256"] for unit in units):
        raise RuntimeError("v37a fold assignment is not a partition")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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
        write_jsonl(train_path, train_rows)
        write_jsonl(dev_path, dev_rows)
        fold_records.append({
            "fold": fold_index,
            "train": {
                "path": str(train_path),
                "rows": len(train_rows),
                "sha256": runtime.file_sha256(train_path),
                "conflict_units": len(train_units),
            },
            "shadow_dev": {
                "path": str(dev_path),
                "rows": len(dev_rows),
                "sha256": runtime.file_sha256(dev_path),
                "conflict_units": len(dev_units),
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
            "train_dev_document_sha256_intersection": len(
                {row["document_sha256"] for row in train_rows}
                & {row["document_sha256"] for row in dev_rows}
            ),
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
            ],
        },
    }
    result["content_sha256_before_self_field"] = runtime.canonical_sha256(result)
    if (
        EXPECTED_MANIFEST_CONTENT_SHA256 != "PENDING"
        and result["content_sha256_before_self_field"]
        != EXPECTED_MANIFEST_CONTENT_SHA256
    ):
        raise RuntimeError("v37a frozen shadow-fold manifest changed")
    return result


def main():
    result = build()
    MANIFEST.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(MANIFEST)
    print(result["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
