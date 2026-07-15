#!/usr/bin/env python3
"""Offline fail-closed tests for the V17A paired runtime wrapper."""

import copy
import json

import numpy as np
import pytest

import eggroll_es_worker_v13 as worker_v13
import run_eggroll_es_paired_data_compat_v17a as driver_v17a


PLAN_SHA = driver_v17a.driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10


def load_layer_bundle():
    spec = driver_v17a.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return driver_v17a.anchor_v13.load_frozen_layer_plan_v13(
        spec["path"],
        expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=driver_v17a.anchor_v11.MODEL_CONFIG_SHA256_V11,
    )


def cli(extra=None):
    spec = driver_v17a.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256",
        driver_v17a.anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v17a-dry-run",
        *(extra or []),
    ]


def synthetic_candidate(delta=0.01, lcb=0.001):
    production = {
        name: 0.4 + 0.01 * index
        for index, name in enumerate(
            driver_v17a.prereg_v17a.ENDPOINT_CONTRACT_V17A
        )
    }
    candidate = {name: value + delta for name, value in production.items()}
    value = {
        "schema": "eggroll-es-paired-data-compat-summary-v17a",
        "experiment_name": driver_v17a.EXPERIMENT_NAME_V17A,
        "alpha": 0.0,
        "sigma": 0.0003,
        "model_update_applied": False,
        "evaluation_surfaces_opened": False,
        "frame_content_sha256": driver_v17a.prereg_v17a.FRAME_CONTENT_SHA256_V17A,
        "perturbation_basis_sha256": (
            driver_v17a.anchor_v13.PERTURBATION_BASIS_SHA256_V13
        ),
        "runtime_integrity": {
            "all_four_engines_every_signed_wave": True,
            "fixed_side_batch_identity_every_direction_and_sign": True,
            "same_resident_perturbation_both_versions": True,
            "alternating_version_order_complete": True,
            "exact_reference_restoration_passed": True,
            "pre_post_base_probes_equal_both_versions": True,
            "population_boundary_audit_passed": True,
            "tokenizer_and_prompt_logprob_contract_passed": True,
            "all_integrity_audits_passed": True,
        },
        "versions": {
            "production": {
                "all_panel_spreads_nonzero": True,
                "endpoint_values": production,
                "compact_estimator_sha256": "1" * 64,
            },
            "candidate_v283": {
                "all_panel_spreads_nonzero": True,
                "endpoint_values": candidate,
                "compact_estimator_sha256": "2" * 64,
            },
        },
        "paired_bootstrap": {
            "seed": driver_v17a.prereg_v17a.BOOTSTRAP_SEED_V17A,
            "repetitions": driver_v17a.prereg_v17a.BOOTSTRAP_REPETITIONS_V17A,
            "one_sided_quantile": 0.05 / 12,
            "endpoints": {
                name: {
                    "candidate_minus_production": delta,
                    "familywise_lcb": lcb,
                    "noninferiority_margin": 0.0,
                }
                for name in driver_v17a.prereg_v17a.ENDPOINT_CONTRACT_V17A
            },
        },
        "cross_dataset_direction_similarity_diagnostic": {
            "used_for_gate": False,
            "content_sha256": "3" * 64,
        },
        "persisted_response_vectors_or_row_content": False,
    }
    value["content_sha256_before_self_field"] = driver_v17a.canonical_sha256(
        value
    )
    return value


def synthetic_configuration():
    return {
        "schema": "eggroll-es-paired-runtime-configuration-v17a",
        "layer_plan_install_sha256": "4" * 64,
        "reference_identity_sha256": "5" * 64,
        "panel_bundle_content_sha256": (
            driver_v17a.mechanics_v17a.PAIRED_PANEL_BUNDLE_CONTENT_SHA256_V17A
        ),
        "worker_extension": (
            "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
        ),
        "model_update_allowed": False,
        "checkpoint_write_allowed": False,
        "evaluation_surfaces_opened": False,
    }


def synthetic_runtime_audit():
    value = {
        "schema": "eggroll-es-paired-runtime-compact-audit-v17a",
        "fixed_request_identity_sha256": "6" * 64,
        "pre_post_probe_identity_sha256": "7" * 64,
        "signed_wave_schedule_sha256": "8" * 64,
        "restore_checks_sha256": "9" * 64,
        "dense_result_commitments_sha256": "a" * 64,
        "population_boundary_audit_sha256": "b" * 64,
        "signed_wave_count": 16,
        "per_unit_scores_persisted": False,
        "bootstrap_replicates_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v17a.canonical_sha256(
        value
    )
    return value


def runtime_inputs():
    layer = load_layer_bundle()
    implementation = driver_v17a.implementation_identity_v17a()
    preregistration, panels = driver_v17a._load_bound_inputs_v17a(layer)
    recipe = driver_v17a.recipe_v17a(
        layer, preregistration, panels, implementation,
    )
    return layer, panels, implementation, recipe


def test_v17a_runtime_installs_v13_worker_and_closes_historical_surfaces():
    trainer_class = driver_v17a.load_runtime_trainer_v17a(load_layer_bundle())
    assert trainer_class.launch_engines.__globals__["WORKER_EXTENSION"] == (
        "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
    )
    controller = object.__new__(trainer_class)
    for name in (
        "configure_train_panels_v13", "estimate_train_panels_v13",
        "estimate_step_coefficients", "apply_seed_coefficients", "train_step",
        "evaluate_handle", "evaluate_population_on_batch", "eval_step", "fit",
    ):
        with pytest.raises(RuntimeError, match="closes every update checkpoint"):
            getattr(controller, name)()

    worker = object.__new__(worker_v13.TrainPanelDiagnosticWorkerExtensionV13)
    for name in (
        "prepare_sharded_seed_update_v4", "execute_prepared_seed_update_v4",
        "commit_prepared_seed_update_v4", "update_weights_from_seeds",
    ):
        with pytest.raises(RuntimeError, match="forbids model updates"):
            getattr(worker, name)()
    with pytest.raises(RuntimeError, match="forbid|layer-restricted"):
        worker.save_self_weights_to_disk("/tmp/forbidden")


class FakeSignedWaveController(
    driver_v17a.PairedDataCompatRuntimeMixinV17A,
):
    def __init__(self, fail_arm=None):
        self.events = []
        self.fail_arm = fail_arm

    def _perturb_signed_wave_v17a(self, engine_seeds, negate):
        self.events.append(("perturb", list(engine_seeds), bool(negate)))

    def _score_resident_arm_v17a(
        self, arm, _prepared, _schedule_item, _unit_scores, dense_commitments,
    ):
        self.events.append(("score", arm))
        if arm == self.fail_arm:
            raise RuntimeError("synthetic resident arm failure")
        dense_commitments.append(arm)
        return arm

    def _restore_and_verify_signed_wave_v17a(self):
        self.events.append(("restore_verify",))
        return "c" * 64


def test_v17a_runtime_exact_resident_call_order_and_restore_on_failure():
    schedule = driver_v17a.mechanics_v17a.resident_signed_wave_schedule_v17a()
    unit_scores = np.empty((2, 5, 2, 32, 38))
    controller = FakeSignedWaveController()
    restore_hash = controller._run_signed_wave_v17a(
        schedule[1], {}, unit_scores, [],
    )
    assert restore_hash == "c" * 64
    assert controller.events == [
        ("perturb", schedule[1]["engine_seeds"], True),
        ("score", "candidate_v283"),
        ("score", "production"),
        ("restore_verify",),
    ]

    failed = FakeSignedWaveController(fail_arm="candidate_v283")
    with pytest.raises(RuntimeError, match="synthetic resident arm failure"):
        failed._run_signed_wave_v17a(schedule[0], {}, unit_scores, [])
    assert failed.events == [
        ("perturb", schedule[0]["engine_seeds"], False),
        ("score", "production"),
        ("score", "candidate_v283"),
        ("restore_verify",),
    ]


def test_v17a_dry_run_is_deterministic_and_constructs_no_trainer(
    monkeypatch, capsys,
):
    monkeypatch.setattr(
        driver_v17a, "_make_trainer_v17a",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run constructed a trainer")
        ),
    )
    first = driver_v17a.main(cli())
    first_output = capsys.readouterr().out
    second = driver_v17a.main(cli())
    second_output = capsys.readouterr().out
    assert first == second
    assert first["schema"] == "eggroll-es-paired-data-compat-dry-run-v17a"
    assert first["gpu_launched"] is False
    assert first["implementation_bundle_sha256"] == first["implementation"][
        "bundle_sha256"
    ]
    assert first["recipe_sha256"] == first["recipe"][
        "content_sha256_before_self_field"
    ]
    assert first["recipe"]["model_update_allowed"] is False
    assert first["recipe"]["checkpoint_write_allowed"] is False
    assert first["recipe"]["evaluation_surfaces_opened"] is False
    assert first["implementation_bundle_sha256"] in first_output
    assert first["recipe_sha256"] in first_output
    assert first_output == second_output


def test_v17a_runtime_rejects_source_recipe_and_nontrain_surface_changes(
    monkeypatch,
):
    layer, _panels, implementation, recipe = runtime_inputs()
    assert implementation["trainer_phase_bundle_sha256"] == (
        driver_v17a.canonical_sha256({
            key: implementation["files"][key]
            for key in driver_v17a.TRAINER_PHASE_PATHS_V17A
        })
    )
    args = driver_v17a._parser_v17a().parse_args(["--v17a-dry-run"])
    args.expected_recipe_sha256 = "0" * 64
    with pytest.raises(ValueError, match="recipe hash changed"):
        driver_v17a.validate_runtime_v17a(args, layer, implementation, recipe)

    monkeypatch.setitem(
        driver_v17a.TRAINER_PHASE_HASHES_V17A,
        "trainer_mechanics_v17a", "0" * 64,
    )
    with pytest.raises(RuntimeError, match="trainer phase changed"):
        driver_v17a.implementation_identity_v17a()

    with pytest.raises(ValueError, match="forbidden runtime surface"):
        driver_v17a.main(cli(["--checkpoint", "/tmp/x"]))
    with pytest.raises(SystemExit):
        driver_v17a.main(cli(["--alpha", "0.1"]))


def test_v17a_compact_report_rejects_hidden_scores_or_row_content():
    _layer, _panels, implementation, recipe = runtime_inputs()
    summary = synthetic_candidate()
    report = {
        "schema": "eggroll-es-paired-data-compat-report-v17a",
        "recipe": recipe,
        "configuration": synthetic_configuration(),
        "runtime_audit": synthetic_runtime_audit(),
        "summary": summary,
        "gate": driver_v17a.prereg_v17a.evaluate_candidate_v17a(summary),
        "implementation": implementation,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
    }
    report["content_sha256_before_self_field"] = driver_v17a.canonical_sha256(
        report
    )
    assert driver_v17a.validate_compact_report_v17a(
        report,
        expected_recipe=recipe,
        expected_implementation=implementation,
    ) == report
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in (
        '"questions"', '"answers"', '"prompt_token_ids"', '"unit_scores"',
        '"responses"', '"coefficients"', '"bootstrap_replicates"',
    ):
        assert forbidden not in serialized

    hidden = copy.deepcopy(report)
    hidden["runtime_audit"]["unit_scores"] = [0.1]
    hidden["runtime_audit"]["content_sha256_before_self_field"] = (
        driver_v17a.canonical_sha256({
            key: value for key, value in hidden["runtime_audit"].items()
            if key != "content_sha256_before_self_field"
        })
    )
    hidden["content_sha256_before_self_field"] = driver_v17a.canonical_sha256({
        key: value for key, value in hidden.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="compact report contract"):
        driver_v17a.validate_compact_report_v17a(
            hidden,
            expected_recipe=recipe,
            expected_implementation=implementation,
        )


def test_v17a_fake_real_path_writes_only_compact_o_excl_report(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(driver_v17a, "FROZEN_OUTPUT_DIRECTORY_V17A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()
    summary = synthetic_candidate()
    gate = driver_v17a.prereg_v17a.evaluate_candidate_v17a(summary)

    class FakeTrainer:
        def configure_paired_data_compat_v17a(self, panel_bundle, **_kwargs):
            assert panel_bundle["content_sha256_before_self_field"] == (
                driver_v17a.mechanics_v17a.PAIRED_PANEL_BUNDLE_CONTENT_SHA256_V17A
            )
            return synthetic_configuration()

        def estimate_paired_data_compat_v17a(self, seeds):
            assert seeds == driver_v17a.anchor_v13.PERTURBATION_SEEDS_V13
            return summary, gate, synthetic_runtime_audit()

    closed = []
    monkeypatch.setattr(
        driver_v17a, "_source_provenance_v17a",
        lambda current: {
            "schema": "synthetic-committed-source-v17a",
            "implementation_bundle_sha256": current["bundle_sha256"],
            "content_sha256_before_self_field": "d" * 64,
        },
    )
    monkeypatch.setattr(driver_v17a, "_make_trainer_v17a", lambda _bundle: FakeTrainer())
    monkeypatch.setattr(driver_v17a.base, "close_trainer", lambda trainer: closed.append(trainer))
    report = driver_v17a.run_exact_v17a(
        layer, panels, implementation, recipe,
    )
    assert len(closed) == 1
    assert driver_v17a.validate_compact_report_v17a(
        report,
        expected_recipe=recipe,
        expected_implementation=implementation,
    ) == report
    report_path = (
        tmp_path / driver_v17a.EXPERIMENT_NAME_V17A / driver_v17a.REPORT_NAME_V17A
    )
    assert json.loads(report_path.read_text()) == report
    assert driver_v17a._attempt_path_v17a().exists()
    with pytest.raises(ValueError, match="fresh exclusive"):
        driver_v17a.run_exact_v17a(layer, panels, implementation, recipe)


def test_v17a_failure_attempt_hashes_error_and_persists_no_message(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(driver_v17a, "FROZEN_OUTPUT_DIRECTORY_V17A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()

    class FailingTrainer:
        def configure_paired_data_compat_v17a(self, *_args, **_kwargs):
            raise RuntimeError("synthetic sensitive failure payload")

    monkeypatch.setattr(
        driver_v17a, "_source_provenance_v17a",
        lambda current: {
            "schema": "synthetic-committed-source-v17a",
            "implementation_bundle_sha256": current["bundle_sha256"],
            "content_sha256_before_self_field": "e" * 64,
        },
    )
    monkeypatch.setattr(
        driver_v17a, "_make_trainer_v17a", lambda _bundle: FailingTrainer(),
    )
    monkeypatch.setattr(driver_v17a.base, "close_trainer", lambda _trainer: None)
    with pytest.raises(RuntimeError, match="synthetic sensitive failure payload"):
        driver_v17a.run_exact_v17a(layer, panels, implementation, recipe)
    attempt = json.loads(driver_v17a._attempt_path_v17a().read_text())
    assert attempt["status"] == "failed"
    assert attempt["failure_type"] == "RuntimeError"
    assert len(attempt["failure_sha256"]) == 64
    assert "failure_message" not in attempt
    assert "traceback" not in attempt
    assert "sensitive failure payload" not in json.dumps(attempt)
