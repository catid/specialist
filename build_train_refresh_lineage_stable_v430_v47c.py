#!/usr/bin/env python3
"""Build the holdout-blind, lineage-stable v430 fold-3 control V47C.

Accepted v413-v430 train-side edits are replayed byte exactly, but edited
content hashes are not allowed to reshuffle the split.  Each refreshed conflict
component is matched to its original v412 component by root fact membership and
inherits that component's already-frozen v37a fold assignment.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import build_train_refresh_v430_v47a as replay
import build_train_shadow_folds_v37a as frozen
import run_sft_train_only_control_v36a as engine
from qa_quality import stable_fact_id


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = (
    ROOT / "experiments/sft_controls/v47c_lineage_stable_train_refresh_v430"
).resolve()
PROJECTION = ARTIFACT_DIR / "train_v430.jsonl"
TRAIN = ARTIFACT_DIR / "fold_3_train.jsonl"
SHADOW = ARTIFACT_DIR / "fold_3_shadow_dev.jsonl"
MANIFEST = ARTIFACT_DIR / "manifest_v47c.json"
CONFIRMATORY_FOLD = 3

EXPECTED = {
    "projection_rows": 531,
    "projection_sha256": replay.PROJECTION_SHA256,
    "conflict_units": 259,
    "train_rows": 448,
    "train_conflict_units": 208,
    "train_sha256": "2ae86c7364e0a997df9471251b8dece539421f1b9716b3f80e5bd7e240fa7b16",
    "shadow_rows": 83,
    "shadow_conflict_units": 51,
    "shadow_sha256": "8481e2ad14c8660e54b4c1bd06281918a8396f6c6dc48c372435ce09b43962ef",
    "manifest_content_sha256": "7a988edf5d55e4be5270b30147d6f3b99cd980380c2dbcc5d3c5e18ef86aed35",
}


def read_jsonl_bytes(payload: bytes) -> list[dict]:
    return [json.loads(line) for line in payload.splitlines() if line]


def root_lineage_map(old_rows: list[dict]) -> tuple[dict[str, str], int]:
    roots = {row["fact_id"]: row["fact_id"] for row in old_rows}
    decisions = 0
    for version, _ in replay.REPLAY:
        path = replay.curation_path(version)
        for raw in path.read_text(encoding="utf-8").splitlines():
            if not raw:
                continue
            decision = json.loads(raw)
            if decision["action"] != "edit":
                raise RuntimeError("V47C replay unexpectedly adds or drops a row")
            root = roots.pop(decision["fact_id"])
            successor = stable_fact_id(decision["question"], decision["answer"])
            if successor in roots:
                raise RuntimeError("V47C edited fact lineage collided")
            roots[successor] = root
            decisions += 1
    return roots, decisions


def membership(unit: dict, rows: list[dict], roots: dict[str, str] | None = None):
    return frozenset(
        (roots or {}).get(rows[index]["fact_id"], rows[index]["fact_id"])
        for index in unit["indices"]
    )


def split_lineage_stable(payload: bytes) -> tuple[dict, bytes, bytes, list[dict]]:
    old_rows = read_jsonl_bytes(replay.SOURCE.read_bytes())
    new_rows = read_jsonl_bytes(payload)
    roots, decisions = root_lineage_map(old_rows)
    if set(roots) != {row["fact_id"] for row in new_rows} or decisions != 51:
        raise RuntimeError("V47C root lineage does not exactly cover v430")

    old_units = frozen.build_conflict_units(old_rows)
    old_folds = frozen.assign_folds(old_units)
    old_by_membership = {}
    old_fold_by_membership = {}
    for fold_index, fold in enumerate(old_folds):
        for unit in fold:
            members = membership(unit, old_rows)
            if members in old_by_membership:
                raise RuntimeError("V47C original component membership collided")
            old_by_membership[members] = unit
            old_fold_by_membership[members] = fold_index

    new_units = frozen.build_conflict_units(new_rows)
    new_by_membership = {
        membership(unit, new_rows, roots): unit for unit in new_units
    }
    if (
        len(old_by_membership) != EXPECTED["conflict_units"]
        or len(new_by_membership) != EXPECTED["conflict_units"]
        or set(old_by_membership) != set(new_by_membership)
    ):
        raise RuntimeError("V47C refreshed component memberships changed")
    multiplicity_changes = {
        members for members in old_by_membership
        if old_by_membership[members]["rows"]
        != new_by_membership[members]["rows"]
    }
    if multiplicity_changes:
        raise RuntimeError("V47C refreshed component multiplicity changed")

    shadow_memberships = {
        members for members, fold in old_fold_by_membership.items()
        if fold == CONFIRMATORY_FOLD
    }
    shadow_indices = {
        index
        for members in shadow_memberships
        for index in new_by_membership[members]["indices"]
    }
    train_rows = [
        row for index, row in enumerate(new_rows) if index not in shadow_indices
    ]
    shadow_rows = [
        row for index, row in enumerate(new_rows) if index in shadow_indices
    ]
    train_payload = frozen.jsonl_bytes(train_rows)
    shadow_payload = frozen.jsonl_bytes(shadow_rows)

    semantic_values = frozen.panel_rules.build_semantic_clusters(new_rows)
    semantic_ids = {
        frozen.row_sha256(row): semantic_id
        for row, semantic_id in zip(new_rows, semantic_values)
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
        "train_conflict_units": len(new_units) - len(shadow_memberships),
        "train_sha256": frozen.bytes_sha256(train_payload),
        "shadow_rows": len(shadow_rows),
        "shadow_conflict_units": len(shadow_memberships),
        "shadow_sha256": frozen.bytes_sha256(shadow_payload),
    }
    for key in (
        "train_rows", "train_conflict_units", "train_sha256",
        "shadow_rows", "shadow_conflict_units", "shadow_sha256",
    ):
        expected = EXPECTED[key]
        if expected != "PENDING" and observed[key] != expected:
            raise RuntimeError(f"V47C lineage-stable aggregate changed: {key}")
    if any(intersections.values()):
        raise RuntimeError("V47C train/shadow conflict edge crossed fold 3")

    commitments = []
    for members in sorted(old_by_membership, key=lambda item: sorted(item)):
        old_unit = old_by_membership[members]
        new_unit = new_by_membership[members]
        commitments.append({
            "root_membership_sha256": engine.canonical_sha256(sorted(members)),
            "original_unit_identity_sha256": old_unit["identity_sha256"],
            "refreshed_unit_identity_sha256": new_unit["identity_sha256"],
            "row_count": old_unit["rows"],
            "fold": old_fold_by_membership[members],
        })
    proof = {
        "accepted_edit_decisions": decisions,
        "added_rows": 0,
        "dropped_rows": 0,
        "components_before": len(old_by_membership),
        "components_after": len(new_by_membership),
        "root_membership_sets_identical": True,
        "unit_row_multiplicities_identical": True,
        "units_with_row_multiplicity_change": 0,
        "original_fold_assignment_inherited_by_root_membership": True,
        "fold_assignment_changes": 0,
        "fold_3_root_memberships_retained": len(shadow_memberships),
    }
    fold = {
        "fold": CONFIRMATORY_FOLD,
        "train": {
            "path": str(TRAIN), "rows": observed["train_rows"],
            "sha256": observed["train_sha256"],
            "conflict_units": observed["train_conflict_units"],
            "unique_documents": len({r["document_sha256"] for r in train_rows}),
        },
        "shadow_dev": {
            "path": str(SHADOW), "rows": observed["shadow_rows"],
            "sha256": observed["shadow_sha256"],
            "conflict_units": observed["shadow_conflict_units"],
            "opened_after_split_construction": False,
        },
        "train_dev_conflict_unit_intersection": 0,
        "train_dev_edge_identity_intersections": intersections,
    }
    return {"proof": proof, "fold": fold}, train_payload, shadow_payload, commitments


def construct() -> tuple[dict, dict[Path, bytes]]:
    first, first_replay = replay.replay_projection_once()
    second, second_replay = replay.replay_projection_once()
    if first != second or first_replay != second_replay:
        raise RuntimeError("V47C train-only replay is nondeterministic")
    if (
        len(read_jsonl_bytes(first)) != EXPECTED["projection_rows"]
        or frozen.bytes_sha256(first) != EXPECTED["projection_sha256"]
    ):
        raise RuntimeError("V47C v430 projection identity changed")
    split, train_payload, shadow_payload, commitments = split_lineage_stable(first)
    result = {
        "schema": "specialist-lineage-stable-train-refresh-fold-v47c",
        "status": "sealed_v430_projection_lineage_stable_fold3_unlaunched",
        "source": {
            "path": str(replay.SOURCE), "rows": replay.SOURCE_ROWS,
            "sha256": replay.SOURCE_SHA256,
        },
        "projection": {
            "path": str(PROJECTION), "rows": EXPECTED["projection_rows"],
            "sha256": EXPECTED["projection_sha256"],
            "repeat_replay_byte_identical": True, "replay": first_replay,
        },
        "split_policy": {
            "source_policy": "frozen v37a v412 conflict graph and fold assignment",
            "preservation_key": "root fact membership across accepted edit lineage",
            "edited_content_rehashed_for_fold_assignment": False,
            "fold_permutation_rerun_on_v430": False,
            "confirmatory_fold": CONFIRMATORY_FOLD,
        },
        "lineage_stability_proof": split["proof"],
        "fold": split["fold"],
        "content_free_unit_commitments": commitments,
        "step_schedule": {
            "world_size": 4, "per_device_batch_size": 7,
            "effective_global_batch_size": 28,
            "dataloader_drop_last": True,
            "optimizer_steps_per_epoch": 16,
            "epochs_argument": 3.0,
            "explicit_max_steps": 48,
            "expected_optimizer_steps": 48,
            "examples_emitted_per_epoch": 448,
            "total_examples_emitted": 1344,
            "row_equivalent_passes": 3.0,
            "complete_row_passes": 3.0,
        },
        "access_firewall": {
            "train_artifacts_opened": True,
            "preaccepted_train_curation_only": True,
            "shadow_dev_opened_after_split_construction": False,
            "eval_ood_holdout_or_benchmark_opened": False,
            "external_metrics_used": False,
        },
        "selection_firewall": {
            "training_input": "lineage-stable refreshed fold-3 train only",
            "shadow_ood_holdout_feedback_authorized": False,
            "post_training_evaluation_authorized": False,
        },
    }
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    expected_content = EXPECTED["manifest_content_sha256"]
    if (
        expected_content != "PENDING"
        and result["content_sha256_before_self_field"] != expected_content
    ):
        raise RuntimeError("V47C manifest content identity changed")
    manifest_payload = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    return result, {
        PROJECTION: first, TRAIN: train_payload, SHADOW: shadow_payload,
        MANIFEST: manifest_payload,
    }


def main() -> None:
    result, payloads = construct()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    existing = [path for path in payloads if path.exists()]
    if existing:
        raise RuntimeError(f"V47C refuses to overwrite artifacts: {existing}")
    for path, payload in payloads.items():
        replay._atomic_exclusive_write(path, payload)
    print(json.dumps({
        "manifest": str(MANIFEST),
        "manifest_file_sha256": engine.file_sha256(MANIFEST),
        "manifest_content_sha256": result["content_sha256_before_self_field"],
        "train_sha256": result["fold"]["train"]["sha256"],
        "shadow_sha256": result["fold"]["shadow_dev"]["sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
