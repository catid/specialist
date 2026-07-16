#!/usr/bin/env python3
"""Build the holdout-blind v430 projection and frozen-policy fold 3.

The original manual-review projection tool performs evaluation collision
checks.  This refresh must not reopen those files, so it replays only the
already accepted train-side edits from the frozen v412 projection.  Every
intermediate output is checked against its precommitted train projection hash.
The fold construction then reuses the exact v37a conflict graph, keyed
assignment seed, and confirmatory fold index.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import build_curated_qa as curated
import build_train_shadow_folds_v37a as frozen
import run_sft_train_only_control_v36a as engine


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT / "experiments/sft_controls/v36a_v412/train_v412.jsonl"
).resolve()
ARTIFACT_DIR = (
    ROOT / "experiments/sft_controls/v47a_train_refresh_v430"
).resolve()
PROJECTION = ARTIFACT_DIR / "train_v430.jsonl"
TRAIN = ARTIFACT_DIR / "fold_3_train.jsonl"
SHADOW = ARTIFACT_DIR / "fold_3_shadow_dev.jsonl"
MANIFEST = ARTIFACT_DIR / "manifest_v47a.json"

SOURCE_ROWS = 531
SOURCE_SHA256 = "44a5482e49073d23a35bdc4c574d6c52c9e8d7f946559dfa722dda16eec5882b"
PROJECTION_ROWS = 531
PROJECTION_SHA256 = "a230b0f0c32a53e8b40140bd5881cfe7e7102f565dd2b3989e38b1fe9a8804c2"
EXPECTED_CONFLICT_UNITS = 259
CONFIRMATORY_FOLD = 3

REPLAY = (
    (413, "4bc7b1f4aafa4154ed63fea50b24ff21d78f317f65757452d4e29e1442358502"),
    (414, "6058812940dee22251462094c20a85862b847a57827ebf88e3800b05d9a9d216"),
    (415, "50b9ae2525b7d1fb8ec71fd6c42da75db5504440d9cecfa9fa2ed829cb3d9a15"),
    (416, "5a05c154f6f4a30c6e7cdb501b309216bca0e38e860c1775e62b06940176b811"),
    (417, "ba2ca0ac7d352d2c9c739c48458dc9785223b7c18eada866ab01d05b521b8664"),
    (418, "ce5d5480f1605431a70de46e706c70f16a1bedbc3e87c62d11b31153afd86250"),
    (419, "46122e53ad9588e0821b2ff24e482d7f89abfa037215a77eb798bba654d11661"),
    (420, "cb27fa80768af54941bac4d354d6c09a5d1c683df530e39c36689401383b13ee"),
    (421, "36a7bd4e519dd290f1026bfe4fb754a41ba4ff6f69ae50d32a17b361f5a5e7a5"),
    (422, "41e4c23472c50d69208a7d569d8229bf2c65e74f5d2587ee6a65301147402812"),
    (423, "b394c37c915d9d42c3d3c20a1fdc17d9a0b7090c8930c3f51f5d775f725bf64a"),
    (425, "448ea81d0ba806bee202f1c32fe2e939ff01c19de390dc231739624ada3b231a"),
    (426, "f26628b352335a6594254e535f60fb32bb3ea82dc4791e5317a216b58a613dcf"),
    (427, "33f7d17b14acbe775c4c153ac0639047b7e5c6b105e618dbc34aadf0e77edb91"),
    (428, "ce7f57eee2cf24b374ca25dfadd85761f1be817c0308050fb40c62ebc820af47"),
    (429, "93ac8de403dbdc1c129198b1535127158e2706af545bde16c460a0dd03d03042"),
    (430, PROJECTION_SHA256),
)

EXPECTED = {
    "train_rows": 459,
    "train_sha256": "2cea57038b4c78acb76de4e9efeb4d405900482fa2e4b71d270909c15e7f8e42",
    "train_conflict_units": 208,
    "shadow_rows": 72,
    "shadow_sha256": "bf67aff37ca3f18fa2e91e62935b804c4e38db30e91ed294b9b2a0fcf21cc5ae",
    "shadow_conflict_units": 51,
    "manifest_content_sha256": "326522033a8d085fe815f5a9b9b2793c687944462fc18c9a28340794f70a70f3",
}


def curation_path(version: int) -> Path:
    return (
        ROOT / "data/manual_reviews" / f"context_merit_audit_v{version}"
        / f"pending_curation_context_merit_v{version}.jsonl"
    ).resolve()


def replay_projection_once() -> tuple[bytes, list[dict]]:
    """Replay accepted train edits with an intentionally empty eval fact set."""
    if engine.file_sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("V47A frozen v412 train source changed")
    if sum(bool(line) for line in SOURCE.read_text().splitlines()) != SOURCE_ROWS:
        raise RuntimeError("V47A frozen v412 train row count changed")
    records = []
    with tempfile.TemporaryDirectory(
        prefix=".v47a-v430-replay-", dir=ARTIFACT_DIR.parent,
    ) as temporary_name:
        directory = Path(temporary_name)
        current = SOURCE
        for version, expected_sha256 in REPLAY:
            decision = curation_path(version)
            output = directory / f"train_v{version}.jsonl"
            report = directory / f"train_v{version}.report.json"
            # Empty facts are deliberate: collision filtering already happened
            # before these decisions were accepted and must not be repeated now.
            curated.merge([current], output, report, [], [decision])
            observed_sha256 = engine.file_sha256(output)
            if observed_sha256 != expected_sha256:
                raise RuntimeError(f"V47A holdout-blind replay drift at v{version}")
            records.append({
                "version": version,
                "curation_path": str(decision),
                "curation_sha256": engine.file_sha256(decision),
                "output_sha256": observed_sha256,
            })
            current = output
        payload = current.read_bytes()
    if (
        frozen.bytes_sha256(payload) != PROJECTION_SHA256
        or payload.count(b"\n") != PROJECTION_ROWS
    ):
        raise RuntimeError("V47A final v430 projection identity changed")
    return payload, records


def split_projection(payload: bytes) -> tuple[dict, bytes, bytes, list[dict]]:
    rows = [json.loads(line) for line in payload.splitlines() if line]
    units = frozen.build_conflict_units(rows)
    if len(units) != EXPECTED_CONFLICT_UNITS:
        raise RuntimeError("V47A refreshed conflict-unit count changed")
    folds = frozen.assign_folds(units)
    dev_units = folds[CONFIRMATORY_FOLD]
    dev_indices = {index for unit in dev_units for index in unit["indices"]}
    train_units = [unit for unit in units if unit not in dev_units]
    train_rows = [row for index, row in enumerate(rows) if index not in dev_indices]
    shadow_rows = [row for index, row in enumerate(rows) if index in dev_indices]
    train_payload = frozen.jsonl_bytes(train_rows)
    shadow_payload = frozen.jsonl_bytes(shadow_rows)
    semantic_values = frozen.panel_rules.build_semantic_clusters(rows)
    semantic_ids = {
        frozen.row_sha256(row): semantic_id
        for row, semantic_id in zip(rows, semantic_values)
    }
    intersections = {
        domain: len(
            frozen.identity_set(train_rows, domain, semantic_ids)
            & frozen.identity_set(shadow_rows, domain, semantic_ids)
        )
        for domain in (
            "document_sha256", "normalized_url", "raw_lineage",
            "semantic_cluster",
        )
    }
    observed = {
        "train_rows": len(train_rows),
        "train_sha256": frozen.bytes_sha256(train_payload),
        "train_conflict_units": len(train_units),
        "shadow_rows": len(shadow_rows),
        "shadow_sha256": frozen.bytes_sha256(shadow_payload),
        "shadow_conflict_units": len(dev_units),
    }
    for key, expected in EXPECTED.items():
        if key != "manifest_content_sha256" and observed.get(key) != expected:
            raise RuntimeError(f"V47A refreshed fold aggregate changed: {key}")
    if any(intersections.values()):
        raise RuntimeError("V47A train/shadow conflict edge crossed fold 3")
    commitments = [{
        "unit_identity_sha256": unit["identity_sha256"],
        "row_count": unit["rows"],
        "dominant_stratum": unit["stratum"],
        "fold": fold_index,
    } for fold_index, fold in enumerate(folds) for unit in fold]
    commitments.sort(key=lambda item: item["unit_identity_sha256"])
    fold = {
        "fold": CONFIRMATORY_FOLD,
        "train": {
            "path": str(TRAIN),
            "rows": len(train_rows),
            "sha256": observed["train_sha256"],
            "conflict_units": len(train_units),
            "unique_documents": len({row["document_sha256"] for row in train_rows}),
        },
        "shadow_dev": {
            "path": str(SHADOW),
            "rows": len(shadow_rows),
            "sha256": observed["shadow_sha256"],
            "conflict_units": len(dev_units),
            "opened_during_v47a_preregistration_or_training": False,
        },
        "train_dev_conflict_unit_intersection": 0,
        "train_dev_edge_identity_intersections": intersections,
    }
    return fold, train_payload, shadow_payload, commitments


def construct() -> tuple[dict, dict[Path, bytes]]:
    first, replay = replay_projection_once()
    second, second_replay = replay_projection_once()
    if first != second or replay != second_replay:
        raise RuntimeError("V47A v430 replay is nondeterministic")
    fold, train_payload, shadow_payload, commitments = split_projection(first)
    result = {
        "schema": "specialist-holdout-blind-train-refresh-fold-v47a",
        "status": "sealed_v430_projection_and_fold3_training_unlaunched",
        "source": {
            "path": str(SOURCE), "rows": SOURCE_ROWS, "sha256": SOURCE_SHA256,
        },
        "projection": {
            "path": str(PROJECTION),
            "rows": PROJECTION_ROWS,
            "sha256": PROJECTION_SHA256,
            "repeat_replay_byte_identical": True,
            "replay": replay,
        },
        "policy": {
            "conflict_units": EXPECTED_CONFLICT_UNITS,
            "component_edges": [
                "shared document_sha256", "shared normalized URL",
                "shared raw/raw_document/raw_successor_document lineage",
                "shared frozen V13 lexical-semantic cluster",
            ],
            "fold_count": frozen.FOLD_COUNT,
            "master_seed": frozen.MASTER_SEED,
            "assignment": (
                "exact frozen v37a within-stratum keyed permutation and "
                "round-robin assignment"
            ),
            "confirmatory_fold": CONFIRMATORY_FOLD,
        },
        "fold": fold,
        "content_free_unit_commitments": commitments,
        "access_firewall": {
            "eval_ood_holdout_or_benchmark_opened": False,
            "eval_facts_supplied_to_replay": 0,
            "shadow_dev_opened_after_split_construction": False,
            "shadow_dev_may_be_opened_during_v47a_training": False,
            "v430_replay_used_only_preaccepted_train_curation": True,
        },
        "selection_firewall": {
            "allowed": ["refreshed fold-3 train optimization"],
            "forbidden": [
                "shadow-dev training access", "OOD access", "holdout access",
                "benchmark access", "evaluation-driven recipe changes",
            ],
        },
    }
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    expected_manifest = EXPECTED["manifest_content_sha256"]
    if expected_manifest != "PENDING" and result[
        "content_sha256_before_self_field"
    ] != expected_manifest:
        raise RuntimeError("V47A train-refresh manifest changed")
    manifest_payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return result, {
        PROJECTION: first,
        TRAIN: train_payload,
        SHADOW: shadow_payload,
        MANIFEST: manifest_payload,
    }


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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("describe", "write-new", "verify-existing"),
        default="verify-existing",
    )
    args = parser.parse_args()
    result, payloads = construct()
    if args.mode == "write-new":
        existing = [str(path) for path in payloads if path.exists()]
        if existing:
            raise RuntimeError(f"V47A refuses to overwrite artifacts: {existing}")
        for path, payload in payloads.items():
            _atomic_exclusive_write(path, payload)
    elif args.mode == "verify-existing":
        for path, payload in payloads.items():
            if not path.is_file() or path.read_bytes() != payload:
                raise RuntimeError(f"V47A artifact verification failed: {path}")
    print(json.dumps({
        "manifest": str(MANIFEST),
        "manifest_content_sha256": result["content_sha256_before_self_field"],
        "projection_sha256": PROJECTION_SHA256,
        "train": result["fold"]["train"],
        "shadow_aggregate": {
            key: value for key, value in result["fold"]["shadow_dev"].items()
            if key != "path"
        },
        "mode": args.mode,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
