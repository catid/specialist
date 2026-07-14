import types

import pytest

import eggroll_es_worker_v11c as worker_v11c
import run_eggroll_es_anchor_equivalence_v11 as driver_v11
import run_eggroll_es_anchor_equivalence_v11c as driver_v11c
import run_eggroll_es_anchor_line_search as driver_v1
import train_eggroll_es_specialist_anchor_v11b as anchor_v11b
import train_eggroll_es_specialist_anchor_v11c as anchor_v11c


PLAN_SHA = driver_v11.MIDDLE_LATE_PLAN_SHA256_V11


def load_bundle():
    spec = anchor_v11b.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return anchor_v11c.load_frozen_layer_plan_v11c(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=(
            anchor_v11b.anchor_v11.MODEL_CONFIG_SHA256_V11
        ),
    )


def cli(batch=128, target="0", basis=20260714, dry=True):
    spec = anchor_v11b.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    values = [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256",
        anchor_v11b.anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v11c-stage", "equivalence_api_retry",
        "--v11c-v10-report", str(driver_v11.V10_REPORT_PATH_V11),
        "--v11c-failed-v11-journal",
        str(driver_v11c.driver_v11b.FAILED_V11_JOURNAL_PATH_V11B),
        "--v11c-v11b-failure-evidence",
        str(driver_v11c.V11B_FAILURE_PATH_V11C),
        "--v11c-perturbation-basis-seed", str(basis),
        "--population-size", "32", "--batch-size", str(batch),
        "--mini-batch-size", "64", "--seed", "43",
        "--target-alphas", target,
        "--experiment-name", driver_v11c.EXPERIMENT_NAME_V11C,
    ]
    if dry:
        values.append("--v11c-dry-run")
    return values


def test_v11c_exports_all_driver_v1_anchor_symbols_and_v11b_missed_two():
    assert driver_v11c.ANCHOR_RUNTIME_API_V11C == (
        "__file__", "coefficient_sha256", "load_anchor_prose", "load_trainer",
    )
    assert hasattr(anchor_v11b, "__file__")
    assert hasattr(anchor_v11b, "load_trainer")
    assert not hasattr(anchor_v11b, "load_anchor_prose")
    assert not hasattr(anchor_v11b, "coefficient_sha256")
    for name in driver_v11c.ANCHOR_RUNTIME_API_V11C:
        assert hasattr(anchor_v11c, name)


def test_v11c_launch_shaped_pre_engine_api_preflight_and_missing_symbol_gate():
    audit = driver_v11c.audit_anchor_runtime_api_v11c(
        anchor_v11c, load_bundle(),
    )
    assert audit["passed"] is True
    assert audit["engine_creation_attempted"] is False
    assert audit["anchor_rows"] == 128
    incomplete = types.SimpleNamespace(
        __file__=anchor_v11c.__file__,
        load_trainer=anchor_v11c.load_trainer,
        coefficient_sha256=anchor_v11c.coefficient_sha256,
    )
    with pytest.raises(RuntimeError, match="missing load_anchor_prose"):
        driver_v11c.audit_anchor_runtime_api_v11c(incomplete, load_bundle())


def test_v11c_non_dry_driver_path_reaches_complete_facade_before_engine(monkeypatch):
    reached = {}

    def stop_before_inherited_engine_creation():
        reached["anchor_module"] = driver_v1.anchor
        reached["all_symbols"] = all(
            hasattr(driver_v1.anchor, name)
            for name in driver_v11c.ANCHOR_RUNTIME_API_V11C
        )
        raise RuntimeError("test stop before engine creation")

    monkeypatch.setattr(driver_v1, "main", stop_before_inherited_engine_creation)
    with pytest.raises(RuntimeError, match="test stop before engine creation"):
        driver_v11c.main(cli(dry=False))
    assert reached == {"anchor_module": anchor_v11c, "all_symbols": True}


def test_v11c_strict_dry_run_and_failure_evidence(capsys):
    failure = driver_v11c._v11b_failure_evidence_v11c(
        driver_v11c.V11B_FAILURE_PATH_V11C
    )
    assert failure["missing_symbol"] == "load_anchor_prose"
    assert failure["gpu_allocation_started"] is False
    result = driver_v11c.main(cli())
    assert result["schema"] == (
        "eggroll-es-resident-sign-complete-api-dry-run-v11c"
    )
    assert result["anchor_runtime_api_preflight"]["passed"] is True
    assert result["actual_perturb_restore_cycle_count"] == 64
    assert result["gpu_ids"] == [0, 1, 2, 3]
    assert "complete-api-dry-run-v11c" in capsys.readouterr().out


def test_v11c_cli_and_mro_are_frozen():
    for args, message in (
        (cli(batch=64), "combined-batch128"),
        (cli(target="0,0.1"), "exactly alpha zero"),
        (cli(basis=43), "basis seed changed"),
    ):
        bundle, remaining = anchor_v11c.parse_frozen_layer_plan_cli_v11c(args)
        with pytest.raises(ValueError, match=message):
            driver_v11c.validate_frozen_execution_cli_v11c(
                remaining, bundle,
            )
    trainer = anchor_v11c.load_trainer(load_bundle())
    assert trainer.estimate_step_coefficients.__module__ == anchor_v11b.__name__
    assert list(worker_v11c.FROZEN_LAYER_PLANS_V11C) == [PLAN_SHA]
