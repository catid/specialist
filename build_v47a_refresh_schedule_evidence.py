#!/usr/bin/env python3
"""Seal train-only refresh-delta and 48-step schedule evidence for V47A."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, RandomSampler

import build_train_refresh_v430_v47a as refresh
import build_train_shadow_folds_v37a as frozen
import run_sft_train_only_control_v36a as engine
from qa_quality import stable_fact_id


OUTPUT = refresh.ARTIFACT_DIR / "refresh_delta_and_step_evidence_v47a.json"
EXPECTED_CONTENT_SHA256 = "1cb5da88101e8742642ff0238ea7b8bbf04c2c5a1a0617ebc60b5c70010f11e4"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def lineage_map(old_rows: list[dict]) -> tuple[dict[str, str], set[str], int]:
    roots = {row["fact_id"]: row["fact_id"] for row in old_rows}
    edited_roots = set()
    decisions = 0
    for version, _ in refresh.REPLAY:
        for decision in read_jsonl(refresh.curation_path(version)):
            if decision["action"] != "edit":
                raise RuntimeError("V47A v413-v430 replay unexpectedly adds/drops")
            old_fact_id = decision["fact_id"]
            root = roots.pop(old_fact_id)
            new_fact_id = stable_fact_id(
                decision["question"], decision["answer"]
            )
            if new_fact_id in roots:
                raise RuntimeError("V47A refresh fact lineage collided")
            roots[new_fact_id] = root
            edited_roots.add(root)
            decisions += 1
    return roots, edited_roots, decisions


def unit_map(rows: list[dict], roots: dict[str, str] | None = None) -> dict:
    units = frozen.build_conflict_units(rows)
    folds = frozen.assign_folds(units)
    result = {}
    for fold_index, fold in enumerate(folds):
        for unit in fold:
            membership = frozenset(
                (roots or {}).get(rows[index]["fact_id"], rows[index]["fact_id"])
                for index in unit["indices"]
            )
            result[membership] = {"unit": unit, "fold": fold_index}
    return result


def refresh_delta() -> dict:
    old_rows = read_jsonl(refresh.SOURCE)
    new_rows = read_jsonl(refresh.PROJECTION)
    roots, edited_roots, decisions = lineage_map(old_rows)
    if set(roots) != {row["fact_id"] for row in new_rows}:
        raise RuntimeError("V47A refresh lineage does not cover v430")
    old = unit_map(old_rows)
    new = unit_map(new_rows, roots)
    if set(old) != set(new):
        raise RuntimeError("V47A refresh changed conflict-component membership")
    memberships = set(old)
    identity_changed = {
        members for members in memberships
        if old[members]["unit"]["identity_sha256"]
        != new[members]["unit"]["identity_sha256"]
    }
    edited_units = {members for members in memberships if members & edited_roots}
    unexplained = identity_changed - edited_units
    missed = edited_units - identity_changed
    multiplicity_changed = {
        members for members in memberships
        if old[members]["unit"]["rows"] != new[members]["unit"]["rows"]
    }
    old_fold = {members for members in memberships if old[members]["fold"] == 3}
    new_fold = {members for members in memberships if new[members]["fold"] == 3}
    result = {
        "source_rows": len(old_rows),
        "refreshed_rows": len(new_rows),
        "accepted_edit_decisions": decisions,
        "added_rows": 0,
        "dropped_rows": 0,
        "conflict_units_before": len(old),
        "conflict_units_after": len(new),
        "component_membership_sets_identical": True,
        "unit_row_multiplicities_identical": not multiplicity_changed,
        "units_with_row_multiplicity_change": len(multiplicity_changed),
        "dominant_stratum_changes": sum(
            old[members]["unit"]["stratum"]
            != new[members]["unit"]["stratum"]
            for members in memberships
        ),
        "edited_units": len(edited_units),
        "expected_edit_driven_unit_identity_changes": len(identity_changed),
        "unexplained_unit_identity_changes": len(unexplained),
        "edited_units_without_identity_change": len(missed),
        "unedited_units_with_exact_identity_preserved": sum(
            members not in edited_units
            and old[members]["unit"]["identity_sha256"]
            == new[members]["unit"]["identity_sha256"]
            for members in memberships
        ),
        "fold_assignment_changes": sum(
            old[members]["fold"] != new[members]["fold"]
            for members in memberships
        ),
        "fold_3": {
            "units_before": len(old_fold),
            "units_after": len(new_fold),
            "units_retained": len(old_fold & new_fold),
            "units_moved_out": len(old_fold - new_fold),
            "units_moved_in": len(new_fold - old_fold),
            "rows_before": sum(old[m]["unit"]["rows"] for m in old_fold),
            "rows_after": sum(new[m]["unit"]["rows"] for m in new_fold),
            "rows_moved_out": sum(
                old[m]["unit"]["rows"] for m in old_fold - new_fold
            ),
            "rows_moved_in": sum(
                new[m]["unit"]["rows"] for m in new_fold - old_fold
            ),
            "row_multiplicity_histogram_before": {
                str(size): sum(
                    old[m]["unit"]["rows"] == size for m in old_fold
                )
                for size in sorted({old[m]["unit"]["rows"] for m in old_fold})
            },
            "row_multiplicity_histogram_after": {
                str(size): sum(
                    new[m]["unit"]["rows"] == size for m in new_fold
                )
                for size in sorted({new[m]["unit"]["rows"] for m in new_fold})
            },
            "moved_out_row_multiplicity_histogram": {
                str(size): sum(
                    old[m]["unit"]["rows"] == size for m in old_fold - new_fold
                )
                for size in sorted({
                    old[m]["unit"]["rows"] for m in old_fold - new_fold
                })
            },
            "moved_in_row_multiplicity_histogram": {
                str(size): sum(
                    new[m]["unit"]["rows"] == size for m in new_fold - old_fold
                )
                for size in sorted({
                    new[m]["unit"]["rows"] for m in new_fold - old_fold
                })
            },
        },
        "interpretation": (
            "The graph partition and strata are unchanged. Accepted edits alter "
            "content hashes for exactly their containing units; the frozen keyed "
            "permutation consequently reorders units. Fold 3 still contains 51 "
            "components, but the 29 components entering it contain 47 rows while "
            "the 29 leaving contain 58. That fixed-component, variable-multiplicity "
            "composition explains the 83 -> 72 row change with no membership or "
            "per-unit multiplicity drift and no protected-data feedback."
        ),
    }
    expected = {
        "source_rows": 531, "refreshed_rows": 531,
        "accepted_edit_decisions": 51, "conflict_units_before": 259,
        "conflict_units_after": 259, "dominant_stratum_changes": 0,
        "edited_units": 37, "expected_edit_driven_unit_identity_changes": 37,
        "unexplained_unit_identity_changes": 0,
        "units_with_row_multiplicity_change": 0,
        "edited_units_without_identity_change": 0,
        "unedited_units_with_exact_identity_preserved": 222,
        "fold_assignment_changes": 147,
    }
    if any(result[key] != value for key, value in expected.items()):
        raise RuntimeError("V47A refresh-delta evidence changed")
    expected_fold_3 = {
        "units_before": 51, "units_after": 51, "units_retained": 22,
        "units_moved_out": 29, "units_moved_in": 29,
        "rows_before": 83, "rows_after": 72,
        "rows_moved_out": 58, "rows_moved_in": 47,
        "row_multiplicity_histogram_before": {
            "1": 39, "2": 6, "3": 3, "4": 1, "6": 1, "13": 1,
        },
        "row_multiplicity_histogram_after": {
            "1": 42, "2": 4, "3": 4, "10": 1,
        },
        "moved_out_row_multiplicity_histogram": {
            "1": 19, "2": 5, "3": 2, "4": 1, "6": 1, "13": 1,
        },
        "moved_in_row_multiplicity_histogram": {
            "1": 22, "2": 3, "3": 3, "10": 1,
        },
    }
    if result["fold_3"] != expected_fold_3:
        raise RuntimeError("V47A fold-3 delta evidence changed")
    return result


def schedule() -> dict:
    rows = refresh.EXPECTED["train_rows"]
    batch_size = 7
    world_size = 4
    epochs = 3
    base = BatchSampler(RandomSampler(range(rows)), batch_size, drop_last=True)
    observed = []
    for rank in range(world_size):
        shard = BatchSamplerShard(
            base, num_processes=world_size, process_index=rank,
            split_batches=False, even_batches=True,
        )
        observed.append({
            "rank": rank, "len": len(shard),
            "iterated_batches": sum(1 for _ in shard),
        })
    result = {
        "implementation": "installed Accelerate BatchSamplerShard no-split path",
        "train_rows": rows,
        "world_size": world_size,
        "per_device_batch_size": batch_size,
        "effective_global_batch_size": world_size * batch_size,
        "dataloader_drop_last": True,
        "unsharded_complete_per_device_batches": len(base),
        "source_rows_dropped_by_unsharded_incomplete_batch": rows - len(base) * batch_size,
        "rank_batch_counts": observed,
        "optimizer_steps_per_dataloader_epoch": observed[0]["len"],
        "epochs_argument": epochs,
        "expected_completed_dataloader_epochs": epochs,
        "expected_optimizer_steps_without_cap": epochs * observed[0]["len"],
        "expected_optimizer_steps": 48,
        "examples_consumed_per_dataloader_epoch": (
            observed[0]["len"] * world_size * batch_size
        ),
        "source_rows_not_emitted_per_dataloader_epoch": (
            rows - observed[0]["len"] * world_size * batch_size
        ),
        "explicit_max_steps_cap": 48,
        "max_steps_is_terminal_authority": True,
        "total_examples_emitted": 48 * world_size * batch_size,
        "row_equivalent_passes": (48 * world_size * batch_size) / rows,
        "complete_three_all_row_passes_claimed": False,
        "batch_sampler_shard_len_source_sha256": engine.canonical_sha256(
            inspect.getsource(BatchSamplerShard.__len__)
        ),
        "batch_sampler_shard_iterator_source_sha256": engine.canonical_sha256(
            inspect.getsource(BatchSamplerShard._iter_with_no_split)
        ),
        "interpretation": (
            "The explicit max_steps=48 cap is the terminal authority and exactly "
            "matches V42I's realized compute. With the inherited drop-last path, "
            "the installed Trainer has 16 steps per dataloader epoch, so 48 steps "
            "also lands at epoch 3.0. Each pass emits 448 of 459 rows, however, so "
            "this is 2.928 row-equivalent passes and not a claim of three complete "
            "all-row passes. The often-assumed 48/17 = 2.82 figure does not apply "
            "to this exact V42I drop-last recipe."
        ),
    }
    if (
        len(base) != 65
        or observed != [
            {"rank": rank, "len": 16, "iterated_batches": 16}
            for rank in range(4)
        ]
        or result["expected_optimizer_steps"] != 48
        or result["explicit_max_steps_cap"] != 48
        or result["examples_consumed_per_dataloader_epoch"] != 448
        or result["source_rows_not_emitted_per_dataloader_epoch"] != 11
    ):
        raise RuntimeError("V47A installed Trainer batch schedule changed")
    return result


def build() -> dict:
    result = {
        "schema": "specialist-v47a-refresh-delta-and-step-evidence",
        "status": "sealed_train_only_evidence_training_unlaunched",
        "inputs": {
            "v412": {"path": str(refresh.SOURCE), "sha256": refresh.SOURCE_SHA256},
            "v430": {"path": str(refresh.PROJECTION), "sha256": refresh.PROJECTION_SHA256},
            "fold_manifest": {
                "path": str(refresh.MANIFEST),
                "sha256": engine.file_sha256(refresh.MANIFEST),
                "content_sha256": refresh.EXPECTED["manifest_content_sha256"],
            },
        },
        "refresh_delta": refresh_delta(),
        "step_schedule": schedule(),
        "access_firewall": {
            "train_artifacts_opened": True,
            "shadow_dev_opened": False,
            "eval_ood_holdout_or_benchmark_opened": False,
            "external_metrics_used": False,
        },
    }
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    if (
        EXPECTED_CONTENT_SHA256 != "PENDING"
        and result["content_sha256_before_self_field"] != EXPECTED_CONTENT_SHA256
    ):
        raise RuntimeError("V47A refresh/schedule evidence identity changed")
    return result


def main() -> None:
    result = build()
    if OUTPUT.exists():
        raise RuntimeError("V47A refuses to overwrite refresh/schedule evidence")
    engine.atomic_write_json(OUTPUT, result)
    print(json.dumps({
        "path": str(OUTPUT),
        "file_sha256": engine.file_sha256(OUTPUT),
        "content_sha256": result["content_sha256_before_self_field"],
        "optimizer_steps": result["step_schedule"]["expected_optimizer_steps"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
