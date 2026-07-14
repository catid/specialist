#!/usr/bin/env python3

import copy

import pytest

import run_eggroll_es_train_panels_v13 as driver_v13
import run_eggroll_es_train_panels_v13b as driver_v13b


def cli(extra=None):
    plan_sha = driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10
    spec = driver_v13.anchor_v11.FROZEN_STABILITY_PLANS_V11[plan_sha]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", plan_sha,
        "--expected-model-config-sha256",
        driver_v13.anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v13-dry-run",
        *(extra or []),
    ]


def test_v13b_binds_the_exact_failed_real_attempt():
    binding = driver_v13b.bind_failed_v13_attempt_v13b()
    assert binding["file_sha256"] == driver_v13b.FAILED_ATTEMPT_FILE_SHA256_V13B
    assert binding["content_sha256"] == (
        driver_v13b.FAILED_ATTEMPT_CONTENT_SHA256_V13B
    )
    assert binding["model_update_applied"] is False
    assert binding["report_written"] is False


def test_v13b_rejects_a_changed_failed_attempt_identity(monkeypatch):
    monkeypatch.setattr(
        driver_v13b, "FAILED_ATTEMPT_FILE_SHA256_V13B", "0" * 64,
    )
    with pytest.raises(RuntimeError, match="file identity changed"):
        driver_v13b.bind_failed_v13_attempt_v13b()


def test_v13b_dry_recipe_changes_only_retry_and_implementation_metadata():
    payload = driver_v13b.main(cli())
    recipe = payload["recipe"]
    assert payload["gpu_launched"] is False
    assert recipe["schema"] == "eggroll-es-five-panel-recipe-v13b"
    assert recipe["experiment_name"] == driver_v13b.EXPERIMENT_NAME_V13B
    assert recipe["alpha"] == 0.0
    assert recipe["model_update_allowed"] is False
    assert recipe["v13_failure_binding"]["content_sha256"] == (
        driver_v13b.FAILED_ATTEMPT_CONTENT_SHA256_V13B
    )
    assert recipe["runtime_expectation_retry_v13b"]["recipe_or_data_changed"] is False


def test_v13b_restores_v13_module_globals_after_dry_run():
    original_name = driver_v13.EXPERIMENT_NAME_V13
    original_paths = copy.deepcopy(driver_v13.IMPLEMENTATION_PATHS_V13)
    driver_v13b.main(cli())
    assert driver_v13.EXPERIMENT_NAME_V13 == original_name
    assert driver_v13.IMPLEMENTATION_PATHS_V13 == original_paths


def test_v13b_keeps_the_nontrain_surface_firewall():
    with pytest.raises(ValueError, match="rejects non-train surface"):
        driver_v13b.main(cli(["--heldout", "forbidden.jsonl"]))
