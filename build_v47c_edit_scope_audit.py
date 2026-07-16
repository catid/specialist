#!/usr/bin/env python3
"""Build a content-free audit of which frozen V47C folds contain edits.

V47C correctly preserves the v412 conflict-unit folds, but the cumulative
v413--v430 projection was produced before those folds were consulted.  This
audit distinguishes edits that actually entered the training partition from
edits that belong to the immutable fold-3 evaluation partition.  It persists
only hashes and aggregate counts, never questions, answers, or evidence.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import build_train_refresh_lineage_stable_v430_v47c as refresh
import build_train_shadow_folds_v37a as frozen
import run_sft_train_only_control_v36a as engine


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/sft_controls/v47c_lineage_stable_train_refresh_v430/"
    "edit_scope_audit_v47c.json"
).resolve()
ORIGINAL_SHADOW = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/"
    "fold_3_shadow_dev.jsonl"
).resolve()
ORIGINAL_SHADOW_SHA256 = (
    "6d5b72f7506a752fd5275425739ec785e25f0ff486f5c03b68e91c8e99d7ebeb"
)


def build() -> dict:
    old_rows = refresh.read_jsonl_bytes(refresh.replay.SOURCE.read_bytes())
    new_rows = refresh.read_jsonl_bytes(refresh.PROJECTION.read_bytes())
    roots, replay_decisions = refresh.root_lineage_map(old_rows)
    if replay_decisions != 51:
        raise RuntimeError("V47C replay decision count changed")

    old_units = frozen.build_conflict_units(old_rows)
    old_folds = frozen.assign_folds(old_units)
    root_to_fold: dict[str, int] = {}
    root_to_unit: dict[str, tuple[int, str]] = {}
    for fold_index, units in enumerate(old_folds):
        for unit in units:
            membership = refresh.membership(unit, old_rows)
            for root in membership:
                if root in root_to_fold:
                    raise RuntimeError("V47C root appears in multiple frozen units")
                root_to_fold[root] = fold_index
                root_to_unit[root] = (fold_index, unit["identity_sha256"])

    changed = []
    for row in new_rows:
        root = roots[row["fact_id"]]
        if row["fact_id"] != root:
            changed.append((root, row["fact_id"], root_to_fold[root]))
    if len(changed) != replay_decisions:
        raise RuntimeError("V47C final edited successors no longer match replay")

    rows_by_fold = Counter(fold for _, _, fold in changed)
    units_by_fold = Counter(
        fold for fold, _ in {root_to_unit[root] for root, _, _ in changed}
    )
    if rows_by_fold != Counter({0: 16, 1: 13, 2: 4, 3: 9, 4: 9}):
        raise RuntimeError("V47C edited-row fold scope changed")
    if units_by_fold != Counter({0: 10, 1: 11, 2: 4, 3: 3, 4: 9}):
        raise RuntimeError("V47C edited-unit fold scope changed")
    if engine.file_sha256(ORIGINAL_SHADOW) != ORIGINAL_SHADOW_SHA256:
        raise RuntimeError("V47C original frozen shadow identity changed")
    if engine.file_sha256(refresh.SHADOW) != refresh.EXPECTED["shadow_sha256"]:
        raise RuntimeError("V47C refreshed shadow proof identity changed")

    result = {
        "schema": "specialist-v47c-edit-scope-audit-v1",
        "status": "complete_content_free_pre_evaluation_audit",
        "inputs": {
            "v412_projection": {
                "path": str(refresh.replay.SOURCE),
                "rows": len(old_rows),
                "sha256": refresh.replay.SOURCE_SHA256,
            },
            "v430_projection": {
                "path": str(refresh.PROJECTION),
                "rows": len(new_rows),
                "sha256": refresh.EXPECTED["projection_sha256"],
            },
            "original_frozen_fold3_shadow": {
                "path": str(ORIGINAL_SHADOW),
                "rows": 83,
                "conflict_units": 51,
                "sha256": ORIGINAL_SHADOW_SHA256,
            },
            "refreshed_fold3_shadow_lineage_proof_only": {
                "path": str(refresh.SHADOW),
                "rows": refresh.EXPECTED["shadow_rows"],
                "conflict_units": refresh.EXPECTED["shadow_conflict_units"],
                "sha256": refresh.EXPECTED["shadow_sha256"],
            },
        },
        "replay_scope": {
            "accepted_edit_decisions": replay_decisions,
            "final_edited_successor_rows": len(changed),
            "edited_rows_by_frozen_fold": {
                str(fold): rows_by_fold[fold] for fold in range(5)
            },
            "edited_units_by_frozen_fold": {
                str(fold): units_by_fold[fold] for fold in range(5)
            },
            "edited_rows_entering_fold3_train": sum(
                count for fold, count in rows_by_fold.items() if fold != 3
            ),
            "edited_units_entering_fold3_train": sum(
                count for fold, count in units_by_fold.items() if fold != 3
            ),
            "edited_rows_excluded_in_frozen_fold3_shadow": rows_by_fold[3],
            "edited_units_excluded_in_frozen_fold3_shadow": units_by_fold[3],
            "fold_assignment_changes": 0,
        },
        "evaluation_policy": {
            "primary_shadow": "original immutable v412 fold-3 shadow",
            "primary_shadow_targets_refreshed_after_training": False,
            "refreshed_v430_shadow_authorized_as_selection_input": False,
            "reason": (
                "changing nine validation answers after the original split "
                "would confound the effect of the 42 edited training rows"
            ),
        },
        "semantic_access": {
            "manual_question_answer_or_evidence_inspection": False,
            "content_free_hash_and_membership_processing_only": True,
            "question_answer_or_evidence_persisted": False,
            "ood_holdout_or_benchmark_opened": False,
        },
    }
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    return result


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    engine.atomic_write_json(OUTPUT, build())
    print(json.dumps({
        "path": str(OUTPUT),
        "file_sha256": engine.file_sha256(OUTPUT),
        "content_sha256": json.loads(OUTPUT.read_text())[
            "content_sha256_before_self_field"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
