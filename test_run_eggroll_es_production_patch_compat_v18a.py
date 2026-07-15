#!/usr/bin/env python3
"""Offline fail-closed tests for the V18A production-patch runtime."""

import copy
import json

import numpy as np
import pytest

import run_eggroll_es_production_patch_compat_v18a as driver_v18a


PLAN_SHA = driver_v18a.driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10


def load_layer_bundle():
    spec = driver_v18a.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return driver_v18a.anchor_v13.load_frozen_layer_plan_v13(
        spec["path"],
        expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=driver_v18a.anchor_v11.MODEL_CONFIG_SHA256_V11,
    )


def cli(extra=None):
    spec = driver_v18a.anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256",
        driver_v18a.anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v18a-dry-run",
        *(extra or []),
    ]


def runtime_inputs():
    layer = load_layer_bundle()
    implementation = driver_v18a.implementation_identity_v18a()
    preregistration, panels = driver_v18a._load_bound_inputs_v18a(layer)
    recipe = driver_v18a.recipe_v18a(
        layer,
        preregistration,
        panels,
        implementation,
        driver_v18a._declared_moe_backend_v18a(),
    )
    return layer, panels, implementation, recipe


def synthetic_summary():
    endpoint_names = driver_v18a.prereg_v18a.ENDPOINT_NAMES_V18A
    arms = {
        arm: {
            "all_panel_spreads_nonzero": True,
            "endpoint_values": {
                name: 0.5 + 0.001 * index
                for index, name in enumerate(endpoint_names)
            },
            "compact_estimator_sha256": str(index + 1) * 64,
        }
        for index, arm in enumerate(driver_v18a.mechanics_v18a.ARMS_V18A)
    }
    value = {
        "schema": "eggroll-es-production-patch-compat-summary-v18a",
        "experiment_name": driver_v18a.EXPERIMENT_NAME_V18A,
        "alpha": 0.0,
        "sigma": 0.0003,
        "model_update_applied": False,
        "evaluation_surfaces_opened": False,
        "frame_content_sha256": (
            driver_v18a.prereg_v18a.FLOW_CERTIFICATE_CONTENT_SHA256_V18A
        ),
        "perturbation_basis_sha256": (
            driver_v18a.anchor_v13.PERTURBATION_BASIS_SHA256_V13
        ),
        "runtime_integrity": {"all_integrity_audits_passed": True},
        "arms": arms,
        "paired_bootstrap": {
            "seed": driver_v18a.prereg_v18a.BOOTSTRAP_SEED_V18A,
            "repetitions": 50_000,
            "one_sided_quantile": 0.05 / 36,
            "comparisons": {
                arm: {
                    name: {
                        "patch_minus_production": 0.01,
                        "familywise_lcb": 0.001,
                        "noninferiority_margin": 0.0,
                    }
                    for name in endpoint_names
                }
                for arm in driver_v18a.mechanics_v18a.ARMS_V18A[1:]
            },
        },
        "persisted_response_vectors_or_row_content": False,
    }
    value["content_sha256_before_self_field"] = driver_v18a.canonical_sha256(
        value
    )
    return value


def synthetic_configuration():
    return {
        "schema": "eggroll-es-production-patch-runtime-configuration-v18a",
        "layer_plan_install_sha256": "4" * 64,
        "reference_identity_sha256": "5" * 64,
        "panel_bundle_content_sha256": (
            driver_v18a.mechanics_v18a.PANEL_BUNDLE_CONTENT_SHA256_V18A
        ),
        "worker_extension": (
            "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
        ),
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "model_update_allowed": False,
        "checkpoint_write_allowed": False,
        "evaluation_surfaces_opened": False,
    }


def synthetic_runtime_audit():
    value = {
        "schema": "eggroll-es-production-patch-runtime-compact-audit-v18a",
        "fixed_request_identity_sha256": "6" * 64,
        "pre_post_probe_identity_sha256": "7" * 64,
        "signed_wave_schedule_sha256": "8" * 64,
        "restore_checks_sha256": "9" * 64,
        "dense_result_commitments_sha256": "a" * 64,
        "population_boundary_audit_sha256": "b" * 64,
        "signed_wave_count": 16,
        "requests_per_engine_per_signed_wave": 1070,
        "per_unit_scores_persisted": False,
        "bootstrap_replicates_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v18a.canonical_sha256(
        value
    )
    return value


def test_v18a_dry_run_binds_commits_recipe_and_request_totals(
    monkeypatch, capsys,
):
    monkeypatch.setattr(
        driver_v18a,
        "_make_trainer_v18a",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run constructed a trainer")
        ),
    )
    first = driver_v18a.main(cli())
    first_output = capsys.readouterr().out
    second = driver_v18a.main(cli())
    second_output = capsys.readouterr().out
    assert first == second
    assert first_output == second_output
    assert first["gpu_launched"] is False
    assert first["implementation"]["trainer_phase_bundle_sha256"] == (
        first["trainer_phase_bundle_sha256"]
    )
    assert first["recipe_sha256"] == first["recipe"][
        "content_sha256_before_self_field"
    ]
    assert first["recipe"]["scoring"][
        "requests_per_engine_per_arm_per_signed_wave"
    ] == {
        "production_only": 260,
        "patch_one_third": 265,
        "patch_two_thirds": 270,
        "patch_full": 275,
    }
    assert first["recipe"]["scoring"][
        "requests_per_engine_per_signed_wave_all_arms"
    ] == 1070
    assert first["recipe"]["hardware"] == {
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "all_engines_every_signed_wave": True,
    }
    assert first["recipe"]["perturbation"]["layers"] == [20, 21, 22, 23]
    assert first["recipe"]["model_update_allowed"] is False
    assert first["recipe"]["checkpoint_write_allowed"] is False
    assert first["recipe"]["evaluation_surfaces_opened"] is False
    assert driver_v18a.BASE_FRAME_COMMIT_V18A.startswith("7055a62")
    assert driver_v18a.TOKEN_CORRECTION_COMMIT_V18A.startswith("a3480e5")
    assert driver_v18a.RUNTIME_CORRECTION_COMMIT_V18A.startswith("3b77622")
    assert driver_v18a.TRAINER_COMMIT_V18A.startswith("5f511f4")


@pytest.mark.parametrize("name", driver_v18a.MOE_OVERRIDE_ENVIRONMENT_V18A)
def test_v18a_rejects_all_five_moe_overrides_before_assembly(
    name, monkeypatch,
):
    monkeypatch.setenv(name, "synthetic-confound")
    monkeypatch.setattr(
        driver_v18a,
        "implementation_identity_v18a",
        lambda: (_ for _ in ()).throw(
            AssertionError("override reached runtime assembly")
        ),
    )
    with pytest.raises(ValueError, match="every MoE backend override unset"):
        driver_v18a.main(cli())


def test_v18a_rejects_wrong_hashes_and_nontrain_surfaces():
    layer, _panels, implementation, recipe = runtime_inputs()
    args = driver_v18a._parser_v18a().parse_args(["--v18a-dry-run"])
    args.expected_recipe_sha256 = "0" * 64
    with pytest.raises(ValueError, match="recipe hash changed"):
        driver_v18a.validate_runtime_v18a(
            args, layer, implementation, recipe
        )
    args.expected_recipe_sha256 = recipe["content_sha256_before_self_field"]
    args.expected_implementation_bundle_sha256 = "0" * 64
    with pytest.raises(ValueError, match="implementation bundle hash changed"):
        driver_v18a.validate_runtime_v18a(
            args, layer, implementation, recipe
        )
    with pytest.raises(ValueError, match="forbidden runtime surface"):
        driver_v18a.main(cli(["--checkpoint", "/tmp/x"]))
    with pytest.raises(SystemExit):
        driver_v18a.main(cli(["--alpha", "0.1"]))


class FakeSignedWaveController(
    driver_v18a.ProductionPatchCompatRuntimeMixinV18A
):
    def __init__(self, fail_arm=None):
        self.events = []
        self.fail_arm = fail_arm

    def _perturb_signed_wave_v18a(self, engine_seeds, negate):
        self.events.append(("perturb", list(engine_seeds), bool(negate)))

    def _score_resident_arm_v18a(
        self, arm, _prepared, _schedule_item, _unit_scores, dense_commitments,
    ):
        self.events.append(("score", arm))
        if arm == self.fail_arm:
            raise RuntimeError("synthetic resident arm failure")
        dense_commitments.append(arm)
        return arm

    def _restore_and_verify_signed_wave_v18a(self):
        self.events.append(("restore_verify",))
        return "c" * 64


def test_v18a_runtime_scores_latin_arm_order_and_restores_once_on_failure():
    schedule = driver_v18a.mechanics_v18a.resident_signed_wave_schedule_v18a()
    unit_scores = {
        arm: np.empty((5, 2, 32, 52 + index))
        for index, arm in enumerate(driver_v18a.mechanics_v18a.ARMS_V18A)
    }
    controller = FakeSignedWaveController()
    restore_hash = controller._run_signed_wave_v18a(
        schedule[3], {}, unit_scores, [],
    )
    assert restore_hash == "c" * 64
    assert controller.events == [
        ("perturb", schedule[3]["engine_seeds"], True),
        *(("score", arm) for arm in schedule[3]["resident_arm_order"]),
        ("restore_verify",),
    ]
    failing_arm = schedule[0]["resident_arm_order"][2]
    failed = FakeSignedWaveController(fail_arm=failing_arm)
    with pytest.raises(RuntimeError, match="synthetic resident arm failure"):
        failed._run_signed_wave_v18a(schedule[0], {}, unit_scores, [])
    assert failed.events[-1] == ("restore_verify",)
    assert sum(event[0] == "perturb" for event in failed.events) == 1
    assert sum(event[0] == "restore_verify" for event in failed.events) == 1


def test_v18a_compact_report_rejects_hidden_content():
    _layer, _panels, implementation, recipe = runtime_inputs()
    summary = synthetic_summary()
    report = {
        "schema": "eggroll-es-production-patch-compat-report-v18a",
        "recipe": recipe,
        "configuration": synthetic_configuration(),
        "runtime_audit": synthetic_runtime_audit(),
        "summary": summary,
        "gate": driver_v18a.mechanics_v18a.evaluate_patch_gate_v18a(summary),
        "implementation": implementation,
        "model_update_applied": False,
        "checkpoint_written": False,
        "evaluation_surfaces_opened": False,
    }
    report["content_sha256_before_self_field"] = driver_v18a.canonical_sha256(
        report
    )
    assert driver_v18a.validate_compact_report_v18a(
        report,
        expected_recipe=recipe,
        expected_implementation=implementation,
    ) == report
    serialized = json.dumps(report, sort_keys=True)
    for forbidden in (
        '"questions"', '"answers"', '"prompt_token_ids"', '"unit_scores"',
        '"responses"', '"coefficients"', '"bootstrap_draws"', '"row_sha256"',
    ):
        assert forbidden not in serialized

    hidden = copy.deepcopy(report)
    hidden["runtime_audit"]["unit_scores"] = [0.1]
    hidden["runtime_audit"]["content_sha256_before_self_field"] = (
        driver_v18a.canonical_sha256({
            key: value
            for key, value in hidden["runtime_audit"].items()
            if key != "content_sha256_before_self_field"
        })
    )
    hidden["content_sha256_before_self_field"] = driver_v18a.canonical_sha256({
        key: value
        for key, value in hidden.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="compact report contract"):
        driver_v18a.validate_compact_report_v18a(
            hidden,
            expected_recipe=recipe,
            expected_implementation=implementation,
        )


def test_v18a_fake_real_path_uses_unique_o_excl_paths_and_cleanup(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(driver_v18a, "FROZEN_OUTPUT_DIRECTORY_V18A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()
    summary = synthetic_summary()
    gate = driver_v18a.mechanics_v18a.evaluate_patch_gate_v18a(summary)

    class FakeTrainer:
        def configure_production_patch_compat_v18a(self, panel_bundle, **_kwargs):
            assert panel_bundle["content_sha256_before_self_field"] == (
                driver_v18a.mechanics_v18a.PANEL_BUNDLE_CONTENT_SHA256_V18A
            )
            return synthetic_configuration()

        def estimate_production_patch_compat_v18a(self, seeds):
            assert seeds == driver_v18a.anchor_v13.PERTURBATION_SEEDS_V13
            return summary, gate, synthetic_runtime_audit()

    launch_ids = iter(("success-one", "success-two"))
    monkeypatch.setattr(driver_v18a, "_new_launch_id_v18a", lambda: next(launch_ids))
    monkeypatch.setattr(
        driver_v18a,
        "_source_provenance_v18a",
        lambda current: {
            "schema": "synthetic-committed-source-v18a",
            "implementation_bundle_sha256": current["bundle_sha256"],
            "content_sha256_before_self_field": "d" * 64,
        },
    )
    monkeypatch.setattr(driver_v18a, "_make_trainer_v18a", lambda _value: FakeTrainer())
    closed = []
    monkeypatch.setattr(
        driver_v18a.base, "close_trainer", lambda trainer: closed.append(trainer)
    )
    first = driver_v18a.run_exact_v18a(
        layer, panels, implementation, recipe
    )
    second = driver_v18a.run_exact_v18a(
        layer, panels, implementation, recipe
    )
    assert first == second
    assert len(closed) == 2
    root = tmp_path / driver_v18a.EXPERIMENT_NAME_V18A
    assert len(list((root / "attempts").glob("*.json"))) == 2
    reports = list((root / "runs").glob(f"*/{driver_v18a.REPORT_NAME_V18A}"))
    assert len(reports) == 2
    assert all(json.loads(path.read_text()) == first for path in reports)


def test_v18a_failure_attempt_hashes_error_and_closes_trainer(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(driver_v18a, "FROZEN_OUTPUT_DIRECTORY_V18A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()

    class FailingTrainer:
        def configure_production_patch_compat_v18a(self, *_args, **_kwargs):
            raise RuntimeError("synthetic sensitive failure payload")

    monkeypatch.setattr(driver_v18a, "_new_launch_id_v18a", lambda: "failure")
    monkeypatch.setattr(
        driver_v18a,
        "_source_provenance_v18a",
        lambda current: {
            "schema": "synthetic-committed-source-v18a",
            "implementation_bundle_sha256": current["bundle_sha256"],
            "content_sha256_before_self_field": "e" * 64,
        },
    )
    trainer = FailingTrainer()
    monkeypatch.setattr(driver_v18a, "_make_trainer_v18a", lambda _value: trainer)
    closed = []
    monkeypatch.setattr(
        driver_v18a.base, "close_trainer", lambda value: closed.append(value)
    )
    with pytest.raises(RuntimeError, match="synthetic sensitive failure payload"):
        driver_v18a.run_exact_v18a(layer, panels, implementation, recipe)
    assert closed == [trainer]
    attempt_path = next(
        (tmp_path / driver_v18a.EXPERIMENT_NAME_V18A / "attempts").glob("*.json")
    )
    attempt = json.loads(attempt_path.read_text())
    assert attempt["status"] == "failed"
    assert attempt["failure_type"] == "RuntimeError"
    assert len(attempt["failure_sha256"]) == 64
    assert "failure_message" not in attempt
    assert "traceback" not in attempt
    assert "sensitive failure payload" not in json.dumps(attempt)
