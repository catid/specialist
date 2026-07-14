#!/usr/bin/env python3

import copy
import inspect
import json
from pathlib import Path

import pytest

import eggroll_es_back_plan_confirmation_preregistration_v15b as prereg
import run_eggroll_es_back_plan_confirmation_v15b as driver
import train_eggroll_es_specialist_anchor_v15b as anchor


def _cli(extra=None):
    return ["--v15b-dry-run", *(extra or [])]


def _integrity():
    value = {
        "alpha_zero_no_applications": True,
        "model_update_applied_false": True,
        "exact_reference_checks_passed": True,
        "pre_post_base_probe_equal": True,
        "population_boundary_passed": True,
        "population_boundary_hash_valid": True,
        "hardware_contract_passed": True,
        "dense_direction_sign_hash_coverage_passed": True,
        "fresh_basis_bound": True,
    }
    value["content_sha256_before_self_field"] = prereg.canonical_sha256(value)
    return value


def _summary(arm):
    frozen = prereg.build_preregistration_v15b()
    stability = copy.deepcopy(
        frozen["promotion_gate"][
            "historical_v13_baseline"
            if arm == "middle_late" else "v15a_back_reference"
        ]
    )
    value = {
        "schema": "eggroll-es-paired-confirmation-arm-summary-v15b",
        "arm": arm,
        "plan_sha256": frozen["paired_architecture"]["arms"][arm][
            "plan_sha256"
        ],
        "diagnostic_content_sha256": prereg.canonical_sha256(
            ["diagnostic", arm]
        ),
        "stability": stability,
        "robust_aggregate": {
            "coefficient_sha256": prereg.canonical_sha256(
                ["coefficient", arm]
            ),
            "l2_norm": 4.0 if arm == "middle_late" else 4.5,
            "nonzero_coordinate_count": 32,
        },
        "all_panel_spreads_nonzero": True,
        "integrity_audits": _integrity(),
        "dense_direction_sign_hash_manifest_sha256": prereg.canonical_sha256(
            ["dense", arm]
        ),
        "persisted_response_vectors": False,
        "persisted_row_content": False,
    }
    value["content_sha256_before_self_field"] = prereg.canonical_sha256(value)
    return value


def _configuration(arm):
    frozen = prereg.build_preregistration_v15b()
    value = {
        "schema": "eggroll-es-paired-confirmation-configuration-v15b",
        "arm": arm,
        "plan_sha256": frozen["paired_architecture"]["arms"][arm][
            "plan_sha256"
        ],
        "panel_bundle_content_sha256": frozen["estimator"][
            "panel_bundle_content_sha256"
        ],
        "layer_plan_install_content_sha256": prereg.canonical_sha256(
            ["install", arm]
        ),
        "reference_identity_content_sha256": prereg.canonical_sha256(
            ["reference", arm]
        ),
        "full_configuration_content_sha256": prereg.canonical_sha256(
            ["configuration", arm]
        ),
        "persisted_configuration_payload": False,
    }
    value["content_sha256_before_self_field"] = prereg.canonical_sha256(value)
    return value


def test_v15b_reuses_exact_diagnostic_bytecode_with_only_fresh_basis():
    assert anchor.analyze_panel_responses_v15b.__code__ is (
        anchor.anchor_v13.analyze_panel_responses_v13.__code__
    )
    assert anchor.validate_diagnostic_v15b.__code__ is (
        anchor.anchor_v13.validate_diagnostic_v13.__code__
    )
    assert anchor._estimate_train_panels_v15b.__code__ is (
        anchor.anchor_v13.TrainPanelDiagnosticMixinV13
        .estimate_train_panels_v13.__code__
    )
    assert anchor.PERTURBATION_SEEDS_V15B == prereg.PERTURBATION_SEEDS_V15B
    assert anchor.PERTURBATION_BASIS_SHA256_V15B == (
        "97e9c5687677bd02365f77671141031ba2739018ed07ccd1bbb3eaabbc0a94f8"
    )
    worker = object.__new__(anchor.PairedConfirmationWorkerExtensionV15B)
    for name in (
        "prepare_sharded_seed_update_v4", "execute_prepared_seed_update_v4",
        "commit_prepared_seed_update_v4", "update_weights_from_seeds",
    ):
        with pytest.raises(RuntimeError, match="forbids model updates"):
            getattr(worker, name)(None)
    source = inspect.getsource(driver) + inspect.getsource(anchor)
    assert "eval_dataloader_dict={}" in source
    assert "alpha=0.0" in source


def test_v15b_dry_run_is_hash_only_deterministic_and_launches_no_gpu(monkeypatch):
    monkeypatch.setattr(
        driver,
        "_validate_real_train_inputs_v15b",
        lambda *_args: pytest.fail("dry run opened train inputs"),
    )
    monkeypatch.setattr(
        driver,
        "_make_trainer_v15b",
        lambda *_args: pytest.fail("dry run constructed trainer"),
    )
    first = driver.main(_cli())
    second = driver.main(_cli())
    assert first == second
    audit = driver.audit_dry_run_payload_v15b(first)
    recipe = first["recipe"]
    assert first["gpu_launched"] is False
    assert recipe["alpha"] == 0.0
    assert recipe["model_update_allowed"] is False
    assert recipe["perturbation_basis_sha256"] == (
        prereg.PERTURBATION_BASIS_SHA256_V15B
    )
    assert recipe["paired_architecture"]["arm_order"] == [
        "middle_late", "back",
    ]
    assert recipe["paired_architecture"]["control_can_be_promoted"] is False
    assert recipe["hardware"] == {
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "complete_four_direction_waves_required": True,
        "population_waves_per_arm": 8,
        "signed_waves_per_arm": 16,
    }
    assert audit["recipe_content_sha256"] == recipe[
        "content_sha256_before_self_field"
    ]


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("population_size", 28),
        ("alpha", 0.1),
        ("n_vllm_engines", 3),
        ("use_gpus", "0,1,2"),
        ("experiment_name", "changed"),
    ],
)
def test_v15b_runtime_contract_fails_closed(attribute, value):
    implementation = driver.implementation_identity_v15b()
    args = driver._parser_v15b().parse_args(["--v15b-dry-run"])
    setattr(args, attribute, value)
    with pytest.raises(ValueError, match="paired four-GPU recipe"):
        driver.validate_runtime_v15b(args, driver._load_plans_v15b(), implementation)


def test_v15b_rejects_nontrain_argv_and_backend_overrides(monkeypatch):
    with pytest.raises(ValueError, match="non-train surface"):
        driver.main(["--heldout", "forbidden.jsonl"])
    monkeypatch.setenv("VLLM_TUNED_CONFIG_FOLDER", "/tmp/confound")
    with pytest.raises(ValueError, match="backend environment overrides unset"):
        driver._moe_environment_binding_v15b()


def test_v15b_real_launch_requires_exact_bundle_and_recipe_hashes():
    implementation = driver.implementation_identity_v15b()
    plans = driver._load_plans_v15b()
    args = driver._parser_v15b().parse_args([])
    with pytest.raises(ValueError, match="implementation bundle hash"):
        driver.validate_runtime_v15b(args, plans, implementation)
    args.expected_implementation_bundle_sha256 = "0" * 64
    with pytest.raises(ValueError, match="implementation bundle hash changed"):
        driver.validate_runtime_v15b(args, plans, implementation)
    payload = driver.main(_cli())
    with pytest.raises(ValueError, match="recipe content hash changed"):
        driver.main(_cli(["--expected-recipe-content-sha256", "0" * 64]))
    assert driver.main(_cli([
        "--expected-recipe-content-sha256",
        payload["recipe"]["content_sha256_before_self_field"],
    ]))["recipe"] == payload["recipe"]


def test_v15b_execute_arm_always_closes_trainer(monkeypatch):
    calls = []

    class Trainer:
        def configure_train_panels_v15b(self, *_args, **_kwargs):
            calls.append("configure")
            return {"synthetic": True}

        def estimate_train_panels_v15b(self, seeds):
            calls.append("estimate")
            assert seeds == prereg.PERTURBATION_SEEDS_V15B
            raise RuntimeError("synthetic failure")

    monkeypatch.setattr(driver, "_make_trainer_v15b", lambda *_args: Trainer())
    monkeypatch.setattr(
        driver.base, "close_trainer", lambda _trainer: calls.append("close")
    )
    args = driver._parser_v15b().parse_args(["--v15b-dry-run"])
    with pytest.raises(RuntimeError, match="synthetic failure"):
        driver._execute_arm_v15b(args, "middle_late", {}, {})
    assert calls == ["configure", "estimate", "close"]


def test_v15b_success_is_o_excl_compact_bound_and_replayable(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "FROZEN_OUTPUT_DIRECTORY_V15B", tmp_path)
    monkeypatch.setattr(
        driver,
        "_source_provenance_v15b",
        lambda implementation: {
            "schema": "synthetic-source-v15b",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
        },
    )
    monkeypatch.setattr(
        driver,
        "_execute_arm_v15b",
        lambda _args, name, _bundle, _panels: (
            _configuration(name), _summary(name),
        ),
    )
    args = driver._parser_v15b().parse_args(["--v15b-dry-run"])
    args.output_directory = str(tmp_path)
    plans = driver._load_plans_v15b()
    implementation = {"files": {}, "bundle_sha256": prereg.canonical_sha256({})}
    report = driver.run_exact_v15b(
        args, plans, {"hash_only": True}, implementation,
        {"schema": "synthetic-recipe-v15b"},
    )
    assert report["persisted_response_vectors_or_row_content"] is False
    assert report["sealed_or_nontrain_surface_opened"] is False
    assert report["model_update_applied"] is False
    assert report["promotion_gate"][
        "eligible_for_separate_back_plan_train_update_preregistration"
    ]
    assert all(
        not item["persisted_configuration_payload"]
        for item in report["configurations"].values()
    )
    persisted = json.loads((
        tmp_path / prereg.EXPERIMENT_NAME_V15B / driver.REPORT_NAME_V15B
    ).read_text())
    replayed = prereg.evaluate_candidate_v15b(persisted["candidate_summary"])
    assert replayed == persisted["promotion_gate"]
    attempt = json.loads(driver._attempt_path_v15b().read_text())
    assert attempt["status"] == "complete"
    assert attempt["phase"] == "after_both_arm_cleanups_and_report"
    with pytest.raises(ValueError, match="fresh exclusive"):
        driver.run_exact_v15b(
            args, plans, {}, implementation, {"schema": "synthetic"},
        )


def test_v15b_second_arm_failure_is_durable_and_no_report(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "FROZEN_OUTPUT_DIRECTORY_V15B", tmp_path)
    monkeypatch.setattr(
        driver,
        "_source_provenance_v15b",
        lambda implementation: {
            "schema": "synthetic-source-v15b",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
        },
    )

    def execute(_args, name, _bundle, _panels):
        if name == "back":
            raise RuntimeError("synthetic second-arm failure")
        return _configuration(name), _summary(name)

    monkeypatch.setattr(driver, "_execute_arm_v15b", execute)
    args = driver._parser_v15b().parse_args(["--v15b-dry-run"])
    args.output_directory = str(tmp_path)
    implementation = {"files": {}, "bundle_sha256": prereg.canonical_sha256({})}
    with pytest.raises(RuntimeError, match="synthetic second-arm failure"):
        driver.run_exact_v15b(
            args, driver._load_plans_v15b(), {}, implementation,
            {"schema": "synthetic-recipe-v15b"},
        )
    attempt = json.loads(driver._attempt_path_v15b().read_text())
    assert attempt["status"] == "failed"
    assert attempt["active_arm"] == "back"
    assert tuple(attempt["completed_arm_summary_bindings"]) == ("middle_late",)
    assert attempt["report_exists_after_attempt"] is False


def test_v15b_report_write_failure_is_durable(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "FROZEN_OUTPUT_DIRECTORY_V15B", tmp_path)
    monkeypatch.setattr(
        driver,
        "_source_provenance_v15b",
        lambda implementation: {
            "schema": "synthetic-source-v15b",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
        },
    )
    monkeypatch.setattr(
        driver,
        "_execute_arm_v15b",
        lambda _args, name, _bundle, _panels: (
            _configuration(name), _summary(name),
        ),
    )
    original = driver.driver_v15a.driver_v13._exclusive_write_json
    calls = 0

    def fail_report(path, value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic report write failure")
        return original(path, value)

    monkeypatch.setattr(
        driver.driver_v15a.driver_v13, "_exclusive_write_json", fail_report
    )
    args = driver._parser_v15b().parse_args(["--v15b-dry-run"])
    args.output_directory = str(tmp_path)
    implementation = {"files": {}, "bundle_sha256": prereg.canonical_sha256({})}
    with pytest.raises(OSError, match="synthetic report write failure"):
        driver.run_exact_v15b(
            args, driver._load_plans_v15b(), {}, implementation,
            {"schema": "synthetic-recipe-v15b"},
        )
    attempt = json.loads(driver._attempt_path_v15b().read_text())
    assert attempt["status"] == "failed"
    assert attempt["phase"] == "writing_final_report"
    assert attempt["report_exists_after_attempt"] is False
