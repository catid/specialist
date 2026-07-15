#!/usr/bin/env python3
"""CPU-only fail-closed tests for authoritative raw V22A attribution."""

import copy
import json

import numpy as np
import pytest

import run_eggroll_es_raw_attribution_v22a as driver_v22a


def load_layer_bundle():
    return driver_v22a.anchor_v13.load_frozen_layer_plan_v13(
        driver_v22a.prereg_v22a.LAYER_PLAN_PATH_V22A,
        expected_file_sha256=driver_v22a.prereg_v22a.LAYER_PLAN_FILE_SHA256_V22A,
        expected_plan_sha256=driver_v22a.prereg_v22a.LAYER_PLAN_SHA256_V22A,
        expected_model_config_sha256=driver_v22a.prereg_v22a.MODEL_CONFIG_SHA256_V22A,
    )


def cli(extra=None):
    return [
        "--layer-plan-json", str(driver_v22a.prereg_v22a.LAYER_PLAN_PATH_V22A),
        "--expected-layer-plan-file-sha256",
        driver_v22a.prereg_v22a.LAYER_PLAN_FILE_SHA256_V22A,
        "--expected-layer-plan-sha256", driver_v22a.prereg_v22a.LAYER_PLAN_SHA256_V22A,
        "--expected-model-config-sha256",
        driver_v22a.prereg_v22a.MODEL_CONFIG_SHA256_V22A,
        "--v22a-raw-dry-run", *(extra or []),
    ]


def runtime_inputs():
    layer = load_layer_bundle()
    implementation = driver_v22a.implementation_identity_v22a()
    preregistration, panels = driver_v22a._load_bound_inputs_v22a(layer)
    recipe = driver_v22a.recipe_v22a(
        layer, preregistration, panels, implementation,
        driver_v22a._declared_moe_backend_v22a(),
    )
    return layer, panels, implementation, recipe


def synthetic_compact():
    names = driver_v22a.mechanics_v22a.ENDPOINT_NAMES_V22A
    arms = {}
    for index, arm in enumerate(driver_v22a.mechanics_v22a.ARMS_V22A):
        arms[arm] = {
            "all_panel_spreads_nonzero": True,
            "endpoint_values": {
                name: 1.0 + 0.25 * index for name in names
            },
            "compact_estimator_sha256": f"{index + 1:064x}",
        }
    return {
        "arms": arms,
        "paired_bootstrap": {
            "seed": driver_v22a.prereg_v22a.BOOTSTRAP_SEED_V22A,
            "repetitions": 50_000,
            "one_sided_quantile": 0.05 / 12,
            "quantile_method": "linear",
            "draw_plan_content_sha256": driver_v22a.mechanics_v22a.DRAW_CONTENT_SHA256_V22A,
            "paired_same_draws_both_arms": True,
            "same_ht_coefficients_and_denominator_both_arms": True,
            "candidate_only_resampling_present": False,
            "whole_panel_block_resampling_used": False,
            "comparison": {
                "name": "v341_matched_replacement_vs_production",
                "treatment": "v341_matched_replacement",
                "control": "production_control",
                "endpoints": {
                    name: {
                        "treatment_minus_control": 0.25,
                        "familywise_lcb": 0.1,
                        "noninferiority_margin": 0.0,
                    } for name in names
                },
            },
        },
    }


def synthetic_summary():
    compact = synthetic_compact()
    value = {
        "schema": "eggroll-es-raw-v341-matched-replacement-summary-v22a",
        "experiment_name": driver_v22a.EXPERIMENT_NAME_V22A,
        "alpha": 0.0, "sigma": 0.0003,
        "runtime_integrity": {
            "all_four_tp1_engines_every_signed_wave": True,
            "all_ten_panels_every_direction_sign_and_arm": True,
            "all_thirty_two_signed_waves_complete": True,
            "counterbalanced_arm_order_complete": True,
            "same_resident_perturbation_both_arms": True,
            "exact_reference_restored_once_per_signed_wave": True,
            "pre_post_raw_reference_probes_equal": True,
            "population_boundary_audit_passed": True,
            "unselected_origin_audit_passed": True,
            "union_planner_called": False,
            "all_integrity_audits_passed": True,
        },
        "arms": compact["arms"], "paired_bootstrap": compact["paired_bootstrap"],
        "union_planner_called": False,
        "model_update_applied": False, "checkpoint_written": False,
        "evaluation_surfaces_opened": False, "dataset_promotion_applied": False,
        "persisted_response_vectors_or_row_content": False,
        "bootstrap_draws_persisted": False, "unit_scores_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v22a.canonical_sha256(value)
    return value


def synthetic_configuration():
    return {
        "schema": "eggroll-es-authoritative-raw-runtime-configuration-v22a",
        "layer_plan_install_sha256": "1" * 64,
        "reference_identity_sha256": "2" * 64,
        "unselected_origin_sha256": "3" * 64,
        "panel_bundle_content_sha256": (
            driver_v22a.mechanics_v22a.PANEL_BUNDLE_CONTENT_SHA256_V22A
        ),
        "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
        "union_planner_called": False, "train_only_raw_runtime_opened": True,
        "model_update_allowed": False, "checkpoint_write_allowed": False,
        "evaluation_surfaces_opened": False, "dataset_promotion_allowed": False,
    }


def synthetic_runtime_audit():
    value = {
        "schema": "eggroll-es-raw-v341-matched-replacement-runtime-audit-v22a",
        "fixed_request_identity_sha256": "4" * 64,
        "token_boundary_audit_sha256": "5" * 64,
        "pre_post_probe_identity_sha256": "6" * 64,
        "signed_wave_schedule_sha256": "7" * 64,
        "restore_checks_sha256": "8" * 64,
        "dense_result_commitments_sha256": "9" * 64,
        "population_boundary_audit_sha256": "a" * 64,
        "unselected_origin_sha256": "b" * 64,
        "unselected_origin_audit_sha256": "c" * 64,
        "signed_wave_count": 32, "panel_count": 10,
        "requests_per_engine_per_signed_wave": 480,
        "requests_per_engine_all_signed_waves": 15_360,
        "requests_all_engines_all_signed_waves": 61_440,
        "dense_result_commitment_count": 2_560,
        "union_planner_called": False, "per_unit_scores_persisted": False,
        "bootstrap_replicates_persisted": False,
        "bootstrap_draws_persisted": False, "row_content_persisted": False,
    }
    value["content_sha256_before_self_field"] = driver_v22a.canonical_sha256(value)
    return value


def make_report(implementation, recipe):
    summary = synthetic_summary()
    value = {
        "schema": "eggroll-es-authoritative-raw-v341-matched-replacement-report-v22a",
        "recipe": recipe, "configuration": synthetic_configuration(),
        "runtime_audit": synthetic_runtime_audit(), "summary": summary,
        "gate": driver_v22a.mechanics_v22a.evaluate_compatibility_gate_v22a(summary),
        "implementation": implementation, "union_planner_called": False,
        "model_update_applied": False, "checkpoint_written": False,
        "evaluation_surfaces_opened": False, "dataset_promotion_applied": False,
    }
    value["content_sha256_before_self_field"] = driver_v22a.canonical_sha256(value)
    return value


def reseal(report):
    for key in ("runtime_audit", "summary"):
        report[key]["content_sha256_before_self_field"] = (
            driver_v22a.canonical_sha256(driver_v22a._without_self(report[key]))
        )
    report["content_sha256_before_self_field"] = driver_v22a.canonical_sha256(
        driver_v22a._without_self(report)
    )


def test_v22a_dry_run_is_deterministic_and_opens_no_gpu(monkeypatch, capsys):
    monkeypatch.setattr(
        driver_v22a, "_make_trainer_v22a",
        lambda *_args: (_ for _ in ()).throw(AssertionError("trainer constructed")),
    )
    first = driver_v22a.main(cli())
    first_output = capsys.readouterr().out
    second = driver_v22a.main(cli())
    second_output = capsys.readouterr().out
    assert first == second and first_output == second_output
    assert first["gpu_launched"] is False
    assert first["union_planner_called"] is False
    recipe = first["recipe"]
    assert recipe["model"].endswith("Qwen3.6-35B-A3B")
    assert recipe["layers"] == [20, 21, 22, 23]
    assert recipe["population_size"] == 64
    assert recipe["authoritative_raw_scoring"] == {
        "requests_per_engine_per_signed_wave": 480,
        "requests_by_arm": {
            "production_control": 240, "v341_matched_replacement": 240,
        },
        "requests_per_engine_all_signed_waves": 15_360,
        "requests_all_engines_all_signed_waves": 61_440,
        "dense_result_commitment_count": 2_560,
        "union_planner_called": False,
    }
    assert recipe["hardware"]["gpu_ids"] == [0, 1, 2, 3]


@pytest.mark.parametrize("name", driver_v22a.MOE_OVERRIDE_ENVIRONMENT_V22A)
def test_v22a_rejects_every_moe_override(name, monkeypatch):
    monkeypatch.setenv(name, "synthetic-confound")
    with pytest.raises(ValueError, match="every MoE backend override unset"):
        driver_v22a.main(cli())


def test_v22a_rejects_hash_tampering_forbidden_and_missing_real_hashes():
    layer, _panels, implementation, recipe = runtime_inputs()
    args = driver_v22a._parser_v22a().parse_args(["--v22a-raw-dry-run"])
    args.expected_recipe_sha256 = "0" * 64
    with pytest.raises(ValueError, match="recipe hash changed"):
        driver_v22a.validate_runtime_v22a(args, layer, implementation, recipe)
    args.expected_recipe_sha256 = recipe["content_sha256_before_self_field"]
    args.expected_implementation_bundle_sha256 = "0" * 64
    with pytest.raises(ValueError, match="implementation bundle hash changed"):
        driver_v22a.validate_runtime_v22a(args, layer, implementation, recipe)
    real = driver_v22a._parser_v22a().parse_args([])
    with pytest.raises(ValueError, match="requires exact implementation and recipe"):
        driver_v22a.validate_runtime_v22a(real, layer, implementation, recipe)
    for forbidden in ("--checkpoint", "--validation-json", "--union-plan"):
        with pytest.raises(ValueError, match="forbidden runtime surface"):
            driver_v22a.main(cli([forbidden, "/tmp/x"]))


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("hardware", "engine_count"), 3),
        (("authoritative_raw_scoring", "requests_per_engine_per_signed_wave"), 481),
        (("authority", "model_update_allowed"), True),
    ),
)
def test_v22a_rejects_resealed_recipe_structure_tampering(path, value):
    layer, _panels, implementation, recipe = runtime_inputs()
    changed = copy.deepcopy(recipe)
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    changed["content_sha256_before_self_field"] = driver_v22a.canonical_sha256(
        driver_v22a._without_self(changed)
    )
    args = driver_v22a._parser_v22a().parse_args(["--v22a-raw-dry-run"])
    with pytest.raises(ValueError, match="frozen recipe changed"):
        driver_v22a.validate_runtime_v22a(args, layer, implementation, changed)


def test_v22a_schedule_is_exact_32_wave_counterbalanced_basis():
    schedule = driver_v22a.resident_signed_wave_schedule_v22a()
    assert len(schedule) == 32
    assert [item["resident_signed_wave_index"] for item in schedule] == list(range(32))
    assert sum(len(item["engine_direction_seeds"]) for item in schedule) == 128
    for sign in ("plus", "minus"):
        signed = [item for item in schedule if item["sign"] == sign]
        for arm in driver_v22a.mechanics_v22a.ARMS_V22A:
            assert sorted(item["resident_arm_order"].index(arm) for item in signed) == (
                [0] * 8 + [1] * 8
            )
    with pytest.raises(RuntimeError, match="exact fresh 64-direction basis changed"):
        driver_v22a.resident_signed_wave_schedule_v22a(list(range(64)))


class FakeWaveController(driver_v22a.RawProductionV341MatchedAttributionRuntimeMixinV22A):
    def __init__(self, fail_arm=None):
        self.events = []
        self.fail_arm = fail_arm

    def _perturb_signed_wave_v19a(self, seeds, negate):
        self.events.append(("perturb", tuple(seeds), negate))

    def _score_resident_arm_v22a(self, arm, *_args):
        self.events.append(("score", arm))
        if arm == self.fail_arm:
            raise RuntimeError("synthetic raw arm failure")
        return arm

    def _restore_and_verify_signed_wave_v19a(self):
        self.events.append(("restore",))
        return "d" * 64


def test_v22a_wave_scores_both_arms_and_restores_once_on_success_or_failure():
    item = driver_v22a.resident_signed_wave_schedule_v22a()[0]
    controller = FakeWaveController()
    assert controller._run_signed_wave_v22a(item, {}, {}, []) == "d" * 64
    assert [event[1] for event in controller.events if event[0] == "score"] == (
        item["resident_arm_order"]
    )
    assert controller.events[-1] == ("restore",)
    failing = FakeWaveController(fail_arm=item["resident_arm_order"][1])
    with pytest.raises(RuntimeError, match="synthetic raw arm failure"):
        failing._run_signed_wave_v22a(item, {}, {}, [])
    assert failing.events[-1] == ("restore",)
    assert sum(event[0] == "restore" for event in failing.events) == 1


class FakeFullController(driver_v22a.RawProductionV341MatchedAttributionRuntimeMixinV22A):
    def __init__(self):
        self._v22a_panel_bundle = {"synthetic": True}
        self._v22a_token_boundary_audit_sha256 = "e" * 64
        self._v4_layer_plan_install = {
            "plan_sha256": "f" * 64, "unselected_origin_sha256": "1" * 64,
        }
        self.wave_indices = []

    def _prepared_fixed_batches_v22a(self):
        return {"synthetic": True}, {"fixed": True}

    def _raw_reference_probe_v22a(self, _prepared):
        return "2" * 64

    def _run_signed_wave_v22a(self, item, _prepared, unit_scores, commitments):
        self.wave_indices.append(item["resident_signed_wave_index"])
        for values in unit_scores.values():
            values.fill(float(item["resident_signed_wave_index"] + 1))
        commitments.extend(
            f"{item['resident_signed_wave_index'] * 80 + index:064x}"
            for index in range(80)
        )
        return f"{item['resident_signed_wave_index']:064x}"

    def _population_boundary_audit_v4(self, _iteration):
        install = self._v4_layer_plan_install
        value = {
            "engine_count": 4, "runtime_mapping": copy.deepcopy(install),
            "unselected_origin_sha256": install["unselected_origin_sha256"],
            "worker_reports": [
                {"rank": rank, "passed": True, **install} for rank in range(4)
            ],
            "passed": True,
        }
        value["audit_sha256"] = driver_v22a.canonical_sha256(value)
        return value


def test_v22a_full_controller_fills_exact_geometry_and_2560_commitments(monkeypatch):
    observed = {}

    def compact_builder(unit_scores, panel_bundle):
        assert panel_bundle == {"synthetic": True}
        observed.update({arm: values.shape for arm, values in unit_scores.items()})
        return synthetic_compact()

    monkeypatch.setattr(
        driver_v22a.mechanics_v22a,
        "build_compact_estimator_summary_v22a", compact_builder,
    )
    controller = FakeFullController()
    summary, gate, audit = controller.estimate_raw_attribution_v22a(
        driver_v22a.prereg_v22a.PERTURBATION_SEEDS_V22A
    )
    assert controller.wave_indices == list(range(32))
    assert observed == {
        arm: (10, 2, 64, driver_v22a.frame_v22a.ARM_REQUESTS_PER_PANEL_V22A[arm])
        for arm in driver_v22a.mechanics_v22a.ARMS_V22A
    }
    assert gate["observed_pass_count"] == gate["bootstrap_pass_count"] == 12
    assert summary["runtime_integrity"]["all_integrity_audits_passed"] is True
    assert audit["signed_wave_count"] == 32
    assert audit["requests_per_engine_per_signed_wave"] == 480
    assert audit["requests_per_engine_all_signed_waves"] == 15_360
    assert audit["requests_all_engines_all_signed_waves"] == 61_440
    assert audit["dense_result_commitment_count"] == 2_560
    assert audit["union_planner_called"] is False


def test_v22a_report_rejects_hidden_content_integrity_and_delta_tampering():
    _layer, _panels, implementation, recipe = runtime_inputs()
    report = make_report(implementation, recipe)
    assert driver_v22a.validate_compact_report_v22a(
        report, expected_recipe=recipe, expected_implementation=implementation
    ) == report
    serialized = json.dumps(report, sort_keys=True)
    for key in ("questions", "answers", "prompt_token_ids", "unit_scores"):
        assert f'"{key}"' not in serialized
    for mutate in ("hidden", "summary_extra", "config_extra", "integrity", "delta"):
        changed = copy.deepcopy(report)
        if mutate == "hidden":
            changed["runtime_audit"]["prompt_token_ids"] = [1]
        elif mutate == "summary_extra":
            changed["summary"]["harmless_aggregate"] = 1
        elif mutate == "config_extra":
            changed["configuration"]["harmless_aggregate"] = 1
        elif mutate == "integrity":
            changed["summary"]["runtime_integrity"]["extra"] = True
        else:
            endpoint = next(iter(
                changed["summary"]["paired_bootstrap"]["comparison"]["endpoints"].values()
            ))
            endpoint["treatment_minus_control"] = 0.2
        reseal(changed)
        with pytest.raises(RuntimeError):
            driver_v22a.validate_compact_report_v22a(
                changed, expected_recipe=recipe, expected_implementation=implementation
            )


def test_v22a_fake_real_path_is_durable_and_cleanup_is_mandatory(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_v22a, "FROZEN_OUTPUT_DIRECTORY_V22A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()
    summary = synthetic_summary()
    gate = driver_v22a.mechanics_v22a.evaluate_compatibility_gate_v22a(summary)

    class FakeTrainer:
        def configure_raw_attribution_v22a(self, panel_bundle, **_kwargs):
            assert panel_bundle["content_sha256_before_self_field"] == (
                driver_v22a.mechanics_v22a.PANEL_BUNDLE_CONTENT_SHA256_V22A
            )
            return synthetic_configuration()

        def estimate_raw_attribution_v22a(self, seeds):
            assert seeds == driver_v22a.prereg_v22a.PERTURBATION_SEEDS_V22A
            return summary, gate, synthetic_runtime_audit()

    monkeypatch.setattr(
        driver_v22a, "_source_provenance_v22a",
        lambda current: {"schema": "synthetic-source",
                         "implementation_bundle_sha256": current["bundle_sha256"],
                         "content_sha256_before_self_field": "d" * 64},
    )
    trainer = FakeTrainer()
    monkeypatch.setattr(driver_v22a, "_make_trainer_v22a", lambda _value: trainer)
    closed = []
    monkeypatch.setattr(driver_v22a.base, "close_trainer", closed.append)
    report = driver_v22a.run_exact_v22a(layer, panels, implementation, recipe)
    assert closed == [trainer]
    root = tmp_path / driver_v22a.EXPERIMENT_NAME_V22A
    attempt = json.loads(next((root / "attempts").glob("*.json")).read_text())
    report_path = next((root / "runs").glob(f"*/{driver_v22a.REPORT_NAME_V22A}"))
    assert attempt["status"] == "complete"
    assert json.loads(report_path.read_text()) == report


def test_v22a_failure_hashes_error_cleans_up_and_writes_no_report(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_v22a, "FROZEN_OUTPUT_DIRECTORY_V22A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()

    class FailingTrainer:
        def configure_raw_attribution_v22a(self, *_args, **_kwargs):
            raise RuntimeError("synthetic sensitive raw failure")

    trainer = FailingTrainer()
    monkeypatch.setattr(
        driver_v22a, "_source_provenance_v22a",
        lambda current: {"schema": "synthetic-source",
                         "implementation_bundle_sha256": current["bundle_sha256"],
                         "content_sha256_before_self_field": "e" * 64},
    )
    monkeypatch.setattr(driver_v22a, "_make_trainer_v22a", lambda _value: trainer)
    closed = []
    monkeypatch.setattr(driver_v22a.base, "close_trainer", closed.append)
    with pytest.raises(RuntimeError, match="synthetic sensitive raw failure"):
        driver_v22a.run_exact_v22a(layer, panels, implementation, recipe)
    assert closed == [trainer]
    root = tmp_path / driver_v22a.EXPERIMENT_NAME_V22A
    attempt = json.loads(next((root / "attempts").glob("*.json")).read_text())
    assert attempt["status"] == "failed" and len(attempt["failure_sha256"]) == 64
    assert "sensitive raw failure" not in json.dumps(attempt)
    assert not list((root / "runs").glob(f"*/{driver_v22a.REPORT_NAME_V22A}"))


def test_v22a_cleanup_failure_is_fail_closed_and_writes_no_report(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_v22a, "FROZEN_OUTPUT_DIRECTORY_V22A", tmp_path)
    layer, panels, implementation, recipe = runtime_inputs()
    summary = synthetic_summary()
    gate = driver_v22a.mechanics_v22a.evaluate_compatibility_gate_v22a(summary)

    class FakeTrainer:
        def configure_raw_attribution_v22a(self, *_args, **_kwargs):
            return synthetic_configuration()

        def estimate_raw_attribution_v22a(self, _seeds):
            return summary, gate, synthetic_runtime_audit()

    monkeypatch.setattr(
        driver_v22a, "_source_provenance_v22a",
        lambda current: {"schema": "synthetic-source",
                         "implementation_bundle_sha256": current["bundle_sha256"],
                         "content_sha256_before_self_field": "f" * 64},
    )
    monkeypatch.setattr(driver_v22a, "_make_trainer_v22a", lambda _value: FakeTrainer())
    monkeypatch.setattr(
        driver_v22a.base, "close_trainer",
        lambda _trainer: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        driver_v22a.run_exact_v22a(layer, panels, implementation, recipe)
    root = tmp_path / driver_v22a.EXPERIMENT_NAME_V22A
    attempt = json.loads(next((root / "attempts").glob("*.json")).read_text())
    assert attempt["status"] == "failed"
    assert attempt["phase"] == "inside_raw_runtime_or_cleanup"
    assert not list((root / "runs").glob(f"*/{driver_v22a.REPORT_NAME_V22A}"))


def test_v22a_union_mutation_checkpoint_and_evaluation_surfaces_are_closed():
    trainer = object.__new__(driver_v22a.RawProductionV341MatchedAttributionRuntimeMixinV22A)
    for method in (
        "configure_union_equivalence_v20a", "run_union_equivalence_v20a",
        "build_union_request_plan_v20a", "_generate_union_v20a",
        "configure_disjoint_tier_attribution_v19a",
        "configure_raw_attribution_v21a", "estimate_raw_attribution_v21a",
        "estimate_disjoint_tier_attribution_v19a", "train_step",
        "apply_seed_coefficients", "fit", "eval_step", "evaluate_handle",
        "evaluate_population_on_batch",
    ):
        with pytest.raises(RuntimeError, match="closes union update checkpoint"):
            getattr(trainer, method)()
