#!/usr/bin/env python3
"""CPU-only contract tests for the V23A heterogeneous worker."""

import json
from pathlib import Path

import pytest

import eggroll_es_worker_v23a as worker_v23a


ROOT = Path(__file__).resolve().parent


def test_v23a_worker_accepts_only_the_four_exact_frozen_plans():
    paths = sorted((ROOT / "experiments/layer_plans").glob("v23a_*_dense.json"))
    assert len(paths) == 4
    for path in paths:
        raw = path.read_bytes()
        value = json.loads(raw)
        frozen = worker_v23a.FROZEN_LAYER_PLANS_V23A[value["plan_sha256"]]
        plan, observed = worker_v23a.validate_frozen_layer_plan_v23a(
            raw, frozen["file_sha256"], value["plan_sha256"]
        )
        assert plan == value and observed == frozen
        assert frozen["source_unit_count"] == 35
        assert frozen["runtime_selected_parameter_count"] == 23
        assert frozen["selected_element_count"] == 142_999_552


def test_v23a_worker_rejects_unknown_or_tampered_plan():
    path = ROOT / "experiments/layer_plans/v23a_insert_front_e005_dense.json"
    value = json.loads(path.read_text())
    frozen = worker_v23a.FROZEN_LAYER_PLANS_V23A[value["plan_sha256"]]
    with pytest.raises(ValueError, match="not frozen"):
        worker_v23a.validate_frozen_layer_plan_v23a(
            path.read_bytes(), frozen["file_sha256"], "0" * 64
        )
    raw = path.read_bytes() + b"\n"
    with pytest.raises(ValueError, match="identity changed"):
        worker_v23a.validate_frozen_layer_plan_v23a(
            raw, frozen["file_sha256"], value["plan_sha256"]
        )


def test_v23a_worker_update_and_checkpoint_surfaces_remain_closed():
    worker = object.__new__(worker_v23a.InsertionLocationAuditWorkerExtensionV23A)
    for method in (
        "prepare_sharded_seed_update_v4", "execute_prepared_seed_update_v4",
        "commit_prepared_seed_update_v4", "update_weights_from_seeds",
        "save_self_initial_weights", "save_self_weights_to_disk",
        "load_weights_from_disk", "broadcast_all_weights",
        "abort_distributed_update_v4", "restore_self_weights",
    ):
        with pytest.raises(RuntimeError):
            if method == "broadcast_all_weights":
                getattr(worker, method)(0)
            else:
                getattr(worker, method)()
