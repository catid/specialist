#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import build_sft_v440_equal_preregistration_v53a as prereg
import build_train_refresh_v440_v53a as refresh
import run_sft_v440_equal_matched_init_v53a as launcher


def test_v440_replay_is_train_only_and_preserves_frozen_membership(monkeypatch):
    def forbidden_eval(*args, **kwargs):
        raise AssertionError("V53A attempted protected evaluation access")

    monkeypatch.setattr(refresh.curated, "eval_facts", forbidden_eval)
    value, payloads = refresh.construct()
    assert payloads[refresh.PROJECTION].count(b"\n") == 531
    assert payloads[refresh.TRAIN].count(b"\n") == 448
    assert value["projection"]["sha256"] == refresh.EXPECTED["projection_sha256"]
    assert value["fold_3_train"]["sha256"] == refresh.EXPECTED["train_sha256"]
    assert value["fold_3_train"]["root_membership_sha256"] == (
        refresh.EXPECTED["root_membership_sha256"]
    )
    assert value["fold_3_train"][
        "membership_exactly_frozen_v412_fold3_train"
    ] is True
    assert value["lineage_stability"]["fold_assignment_changes"] == 0
    assert value["access_firewall"]["shadow_artifact_opened"] is False
    assert value["access_firewall"][
        "eval_ood_holdout_or_benchmark_opened"
    ] is False


def test_v440_equal_preregistration_matches_v49d_equal_recipe():
    value = prereg.build()
    recipe = value["recipe"]
    assert value["status"] == "sealed_unlaunched_train_only"
    assert value["training_launch_authorized"] is True
    assert value["evaluation_launch_authorized"] is False
    assert value["contains_external_validation_ood_or_holdout_content"] is False
    assert value["dataset"]["sha256"] == refresh.EXPECTED["train_sha256"]
    assert recipe["world_size"] == 4
    assert recipe["physical_gpu_ids"] == [0, 1, 2, 3]
    assert recipe["effective_global_batch_size"] == 28
    assert recipe["explicit_max_steps_cap"] == 48
    assert recipe["expected_optimizer_steps"] == 48
    assert recipe["learning_rate"] == 5.5e-5
    assert recipe["target_layers"] == [20, 21, 22, 23]
    assert recipe["rank"] == 32
    assert recipe["lora_alpha"] == 64
    assert recipe["expected_encoding_audit"] == launcher.EXPECTED_ENCODING_AUDIT
    assert recipe["expected_weighting_audit"] == launcher.EXPECTED_WEIGHTING_AUDIT
    assert value["access_firewall"]["shadow_artifact_opened"] is False
    assert value["access_firewall"]["eval_ood_holdout_opened"] is False


def test_v53a_source_has_no_protected_runtime_path():
    sources = "\n".join(
        Path(path).read_text(encoding="utf-8").casefold()
        for path in (refresh.__file__, launcher.__file__, prereg.__file__)
    )
    assert "holdout_eval" not in sources
    assert "fold_3_shadow_dev.jsonl" not in sources
    value = prereg.build()
    assert "evaluation_launch_authorized" in json.dumps(value)
    assert value["evaluation_launch_authorized"] is False
