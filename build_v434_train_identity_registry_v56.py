#!/usr/bin/env python3
"""Build the content-minimized V434 train identity registry for V56.

The registry lets the OOD-only runtime prove document, URL, raw-lineage, and
frozen lexical-semantic disjointness without reopening any training row.  It
persists no question, answer, evidence, or document text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_eval_v3 as eval_v3
import build_train_shadow_folds_v37a as folds
import eggroll_es_train_panel_sampler_v13 as semantic
import run_sft_train_only_control_v36a as hashing


ROOT = Path(__file__).resolve().parent
TRAIN = (
    ROOT / "experiments/sft_controls/v49d_v434_sampling_midpoint_lr5p5e5/"
    "train_v434_fold3_v49d.jsonl"
).resolve()
TRAIN_FILE_SHA256 = (
    "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a"
)
TRAIN_ROWS = 448
MEMBERSHIP = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v52_v434_train_row_conflict_unit_membership.json"
).resolve()
MEMBERSHIP_FILE_SHA256 = (
    "e9b073369966e21912a0bda86da501ab0975646df2a7d80bf5675c3dfec8c121"
)
MEMBERSHIP_CONTENT_SHA256 = (
    "a8870fdce8fbf631b3d3472fd03690f6987590ee6e8758dc8fdcb4556dcc9096"
)
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v56_v434_train_disjoint_identity_registry.json"
).resolve()


def canonical_lineage_identity_v56(value: str) -> str:
    return hashing.canonical_sha256({
        "schema": "raw-lineage-edge-identity-v56",
        "value": value,
    })


def normalized_urls_v56(row: dict) -> list[str]:
    return sorted({
        eval_v3.normalize_source_url(value)
        for _field, value in eval_v3.source_urls(row)
    })


def build_registry_v56() -> dict:
    if hashing.file_sha256(TRAIN) != TRAIN_FILE_SHA256:
        raise RuntimeError("V56 V434 train bytes changed")
    if hashing.file_sha256(MEMBERSHIP) != MEMBERSHIP_FILE_SHA256:
        raise RuntimeError("V56 V52 membership bytes changed")
    membership = json.loads(MEMBERSHIP.read_text(encoding="utf-8"))
    compact_membership = {
        key: value for key, value in membership.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        membership.get("content_sha256_before_self_field")
        != MEMBERSHIP_CONTENT_SHA256
        or hashing.canonical_sha256(compact_membership)
        != MEMBERSHIP_CONTENT_SHA256
        or membership.get("rows") != TRAIN_ROWS
        or membership.get("source", {}).get("train_dataset_file_sha256")
        != TRAIN_FILE_SHA256
        or membership.get("question_answer_evidence_or_text_persisted")
        is not False
        or membership.get("nontrain_semantics_opened") is not False
    ):
        raise RuntimeError("V56 V52 train membership contract changed")

    rows = [
        json.loads(line) for line in TRAIN.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != TRAIN_ROWS:
        raise RuntimeError("V56 V434 train row count changed")
    required = {"question", "answer", "fact_id", "document_sha256"}
    if any(
        not isinstance(row, dict)
        or not required.issubset(row)
        or any(not isinstance(row[key], str) or not row[key] for key in required)
        for row in rows
    ):
        raise RuntimeError("V56 V434 train schema changed")

    semantic_ids = semantic.build_semantic_clusters(rows)
    items = []
    for row, semantic_id in zip(rows, semantic_ids, strict=True):
        row_identity = semantic.row_sha256(row)
        lineages = sorted(
            canonical_lineage_identity_v56(value)
            for value in folds.row_lineage_identities(row)
        )
        items.append({
            "row_sha256": row_identity,
            "document_sha256": row["document_sha256"],
            "normalized_urls": normalized_urls_v56(row),
            "raw_lineage_identity_sha256s": lineages,
            "semantic_cluster_sha256": semantic_id,
            "semantic_question_tokens": sorted(
                semantic._content_tokens(row["question"])
            ),
            "semantic_answer_tokens": sorted(
                semantic._content_tokens(row["answer"])
            ),
        })
    items.sort(key=lambda item: item["row_sha256"])
    if len({item["row_sha256"] for item in items}) != TRAIN_ROWS:
        raise RuntimeError("V56 V434 train row identities repeated")

    value = {
        "schema": "v434-train-disjoint-identity-registry-v56",
        "status": "complete_train_derived_identity_only_before_ood_preregistration",
        "source": {
            "path": str(TRAIN),
            "file_sha256": TRAIN_FILE_SHA256,
            "rows": TRAIN_ROWS,
            "root_membership_exactly_frozen_v412_fold3_train": True,
            "membership_path": str(MEMBERSHIP),
            "membership_file_sha256": MEMBERSHIP_FILE_SHA256,
            "membership_content_sha256": MEMBERSHIP_CONTENT_SHA256,
        },
        "identity_domains": [
            "document_sha256",
            "normalized_url",
            "raw_lineage",
            "semantic_cluster",
        ],
        "semantic_rule": {
            "implementation": str(Path(semantic.__file__).resolve()),
            "implementation_file_sha256": hashing.file_sha256(
                Path(semantic.__file__).resolve()
            ),
            "rule": "frozen V13 lexical-semantic match and train cluster identity",
            "question_similarity_direct_threshold": 0.82,
            "question_similarity_joint_threshold": 0.66,
            "answer_similarity_joint_threshold": 0.86,
        },
        "normalization_rule": {
            "implementation": str(Path(eval_v3.__file__).resolve()),
            "implementation_file_sha256": hashing.file_sha256(
                Path(eval_v3.__file__).resolve()
            ),
            "function": "normalize_source_url",
        },
        "items": items,
        "aggregate": {
            "rows": len(items),
            "documents": len({item["document_sha256"] for item in items}),
            "normalized_urls": len({
                value for item in items for value in item["normalized_urls"]
            }),
            "raw_lineages": len({
                value for item in items
                for value in item["raw_lineage_identity_sha256s"]
            }),
            "semantic_clusters": len({
                item["semantic_cluster_sha256"] for item in items
            }),
            "ordered_row_identity_sha256": hashing.canonical_sha256([
                item["row_sha256"] for item in items
            ]),
            "ordered_identity_record_sha256": hashing.canonical_sha256(items),
        },
        "content_minimization": {
            "question_persisted": False,
            "answer_persisted": False,
            "evidence_persisted": False,
            "document_text_persisted": False,
            "only_hashes_normalized_provenance_and_lexical_feature_sets": True,
        },
        "access_receipt": {
            "train_semantics_opened_by_registry_builder": True,
            "ood_shadow_holdout_or_benchmark_semantics_opened": False,
            "gpu_accessed": False,
            "model_outcomes_used": False,
        },
    }
    value["content_sha256_before_self_field"] = hashing.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_registry_v56()
    hashing.atomic_write_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": hashing.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "rows": value["aggregate"]["rows"],
        "question_answer_evidence_or_document_text_persisted": False,
        "nontrain_semantics_opened": False,
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
