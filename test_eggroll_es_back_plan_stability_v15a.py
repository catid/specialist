#!/usr/bin/env python3

import copy
import inspect
import json
import subprocess
from pathlib import Path

import pytest

import eggroll_es_back_plan_preregistration_v15a as prereg_v15a
import run_eggroll_es_back_plan_stability_v15a as driver_v15a
import train_eggroll_es_specialist_anchor_v15a as anchor_v15a


def plans():
    return prereg_v15a.validate_layer_plans_v15a()


def cli(extra=None):
    return ["--v15a-dry-run", *(extra or [])]


def synthetic_responses(scale=1.0):
    result = {}
    base_vector = [float(index) - 15.5 for index in range(32)]
    for panel_index, name in enumerate(anchor_v15a.anchor_v13.PANEL_NAMES_V13):
        central = [
            scale * (1.0 + 0.05 * panel_index) * value
            for value in base_vector
        ]
        weighted = {
            "plus": [10.0 + value for value in central],
            "minus": [10.0 - value for value in central],
        }
        unweighted = {
            "plus": [20.0 + 0.5 * value for value in central],
            "minus": [20.0 - 0.5 * value for value in central],
        }
        result[name] = {
            "weighted_sign_scores": weighted,
            "unweighted_sign_scores": unweighted,
            "stratum_sign_scores": {
                stratum: {
                    sign: list(unweighted[sign])
                    for sign in anchor_v15a.SIGNS_V15A
                }
                for stratum in anchor_v15a.anchor_v13.panel_sampler.STRATA
            },
            "weighted_stratum_contributions": {
                stratum: {
                    sign: [
                        value
                        * anchor_v15a.anchor_v13.STRATUM_POPULATION_V13[stratum]
                        / 310.0
                        for value in weighted[sign]
                    ]
                    for sign in anchor_v15a.SIGNS_V15A
                }
                for stratum in anchor_v15a.anchor_v13.panel_sampler.STRATA
            },
            "dense_result_sha256": {
                sign: [
                    anchor_v15a.canonical_sha256(
                        [scale, panel_index, sign, index]
                    )
                    for index in range(32)
                ]
                for sign in anchor_v15a.SIGNS_V15A
            },
        }
    return result


def synthetic_diagnostic(scale=1.0):
    responses = synthetic_responses(scale)
    analysis = anchor_v15a.analyze_panel_responses_v15a(responses)
    panel_bundle = anchor_v15a.anchor_v13.load_panel_bundle_v13()
    probe = {
        "schema": "synthetic-v15a-probe",
        "panel_dense_result_sha256": {
            name: anchor_v15a.canonical_sha256(["base", name])
            for name in anchor_v15a.anchor_v13.PANEL_NAMES_V13
        },
        "combined_request_count": 280,
    }
    boundary = {"passed": True, "iteration": 0}
    boundary["audit_sha256"] = anchor_v15a.canonical_sha256(boundary)
    value = {
        "schema": "eggroll-es-five-panel-resident-sign-diagnostic-v13",
        "iteration": 0,
        "alpha": 0.0,
        "model_update_applied": False,
        "applications": [],
        "perturbation_basis": {
            "basis_seed": anchor_v15a.PERTURBATION_BASIS_SEED_V15A,
            "basis_sha256": anchor_v15a.PERTURBATION_BASIS_SHA256_V15A,
            "seeds": list(anchor_v15a.PERTURBATION_SEEDS_V15A),
            "seed_sha256": anchor_v15a.canonical_sha256(
                anchor_v15a.PERTURBATION_SEEDS_V15A
            ),
            "sign_order": list(anchor_v15a.SIGNS_V15A),
        },
        "panel_bundle_content_sha256": (
            anchor_v15a.anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
        ),
        "panel_contract": {
            name: {
                "role": anchor_v15a.anchor_v13.panel_sampler.PANEL_ROLES[name],
                "rows": 56,
                "ordered_row_identity_sha256": (
                    anchor_v15a.anchor_v13.PANEL_ORDERED_ROW_SHA256_V13[name]
                ),
                "templated_prompt_answer_sha256": (
                    anchor_v15a.anchor_v13.PANEL_TEMPLATED_QA_SHA256_V13[name]
                ),
            }
            for name in anchor_v15a.anchor_v13.PANEL_NAMES_V13
        },
        "common_random_numbers": {
            "generation_seed": 43,
            "temperature": 0.0,
            "same_order_every_direction_and_sign": True,
            "combined_panel_order": list(
                anchor_v15a.anchor_v13.PANEL_NAMES_V13
            ),
        },
        "responses": responses,
        "analysis": analysis,
        "identity_audit": {
            "pre_probe": probe,
            "post_probe": copy.deepcopy(probe),
            "exact_reference_checks": [
                {"passed": True, "rank": rank} for rank in range(4)
            ],
            "passed": True,
        },
        "population_boundary_audit_v4": boundary,
        "hardware_coverage": {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "population_waves": 8,
            "signed_waves": 16,
            "partial_waves": 0,
            "all_engines_generate_every_signed_wave": True,
        },
        "interpretation": "diagnostic_only_no_promotion_decision",
    }
    value["content_sha256_before_self_field"] = anchor_v15a.canonical_sha256(
        value
    )
    anchor_v15a.validate_diagnostic_v15a(value)
    return value


def walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_keys(item)


def test_v15a_composes_exact_v6_installer_with_v13_no_update_boundary():
    assert anchor_v15a._install_layer_plan_v15a.__code__ is (
        anchor_v15a.worker_v6.FrozenEdgeSplitAuditWorkerExtensionV6
        .install_layer_plan_v4.__code__
    )
    assert anchor_v15a._estimate_train_panels_v15a.__code__ is (
        anchor_v15a.anchor_v13.TrainPanelDiagnosticMixinV13
        .estimate_train_panels_v13.__code__
    )
    worker = object.__new__(
        anchor_v15a.PairedArchitectureDiagnosticWorkerExtensionV15A
    )
    for method in (
        "prepare_sharded_seed_update_v4", "execute_prepared_seed_update_v4",
        "commit_prepared_seed_update_v4", "update_weights_from_seeds",
    ):
        with pytest.raises(RuntimeError, match="forbids model updates"):
            getattr(worker, method)()
    with pytest.raises(RuntimeError, match="forbids model updates"):
        anchor_v15a.PairedArchitectureArmMixinV15A.apply_seed_coefficients(
            object(), {}, 0.0,
        )
    source = inspect.getsource(driver_v15a)
    assert "update_weights_from_seeds" not in source
    assert "apply_seed_coefficients(" not in source


def test_v15a_both_exact_v6_plans_have_identical_capacity_and_fresh_basis():
    bundles = plans()
    assert tuple(bundles) == ("middle_late", "back")
    assert [
        anchor_v15a.validate_frozen_layer_plan_bundle_v15a(bundle)["arm"]
        for bundle in bundles.values()
    ] == ["middle_late", "back"]
    assert [
        anchor_v15a.validate_frozen_layer_plan_bundle_v15a(bundle)["layers"]
        for bundle in bundles.values()
    ] == [[20, 21, 22, 23], [36, 37, 38, 39]]
    assert all(
        anchor_v15a.anchor_v6.FROZEN_RUNTIME_EXPECTATIONS_V6[
            bundle["plan_sha256"]
        ] == {
            "source_unit_count": 35,
            "runtime_selected_parameter_count": 23,
            "selected_element_count": 142_999_552,
            "selected_byte_count": 285_999_104,
        }
        for bundle in bundles.values()
    )
    assert anchor_v15a.PERTURBATION_BASIS_SHA256_V15A == (
        "6c358060c5f9a0a7b00e953bd230b18f915950f0233f38321e0e048a67ea05e7"
    )
    assert anchor_v15a.PERTURBATION_BASIS_SHA256_V15A != (
        prereg_v15a.PREVIOUS_PERTURBATION_BASIS_SHA256_V15A
    )
    assert len(set(anchor_v15a.PERTURBATION_SEEDS_V15A)) == 32


def test_v15a_trainer_loads_both_arms_with_v13_worker_and_triton_recipe():
    for name, bundle in plans().items():
        trainer = anchor_v15a.load_trainer(bundle)
        assert trainer.launch_engines.__globals__["WORKER_EXTENSION"] == (
            anchor_v15a.WORKER_EXTENSION
        )
        assert '"moe_backend": "triton"' in inspect.getsource(
            trainer.launch_engines
        )
        assert trainer.configure_train_panels_v15a.__module__ == (
            anchor_v15a.__name__
        )
        assert trainer.estimate_train_panels_v15a.__module__ == (
            anchor_v15a.__name__
        )
        metadata = anchor_v15a.validate_frozen_layer_plan_bundle_v15a(bundle)
        assert metadata["arm"] == name


def test_v15a_exact_four_engine_fresh_basis_wave_contract_without_gpu(monkeypatch):
    calls = {"generate": [], "rpc": [], "restores": 0}
    seed_index = {
        seed: index for index, seed in enumerate(
            anchor_v15a.PERTURBATION_SEEDS_V15A
        )
    }

    class Remote:
        def __init__(self, callback):
            self.callback = callback

        def remote(self, *args, **kwargs):
            return self.callback(*args, **kwargs)

    class Engine:
        def __init__(self, index):
            self.index = index
            self.state = None
            self.collective_rpc = Remote(self.rpc)
            self.generate = Remote(self.run)

        def rpc(self, name, args):
            calls["rpc"].append((self.index, name, args))
            if name == "perturb_self_weights":
                seed, sigma, negate = args
                assert sigma == 0.0003
                self.state = (int(seed), bool(negate))
                return {"passed": True}
            if name == "verify_self_exact_reference":
                return {"passed": True}
            raise AssertionError(name)

        def run(self, prompts, _params, *, use_tqdm):
            assert len(prompts) == 280 and use_tqdm is False
            calls["generate"].append((self.index, self.state))
            marker = {"seed": self.state[0], "negate": self.state[1]}
            return [marker] * 280

    bundle = anchor_v15a.anchor_v13.load_panel_bundle_v13()
    prepared = {}
    cursor = 0
    for name in anchor_v15a.anchor_v13.PANEL_NAMES_V13:
        prepared[name] = {
            "panel": bundle["panels"][name],
            "dense_items": [{}] * 56,
            "slice": (cursor, cursor + 56),
            "templated_prompt_answer_sha256": (
                anchor_v15a.anchor_v13.PANEL_TEMPLATED_QA_SHA256_V13[name]
            ),
        }
        cursor += 56
    controller = object.__new__(anchor_v15a.PairedArchitectureArmMixinV15A)
    controller.engines = [Engine(index) for index in range(4)]
    controller.sigma = 0.0003
    controller._v13_panel_bundle = bundle
    controller._resolve = lambda values: values
    controller._dense_sampling_params_v4 = lambda _seed: {"synthetic": True}
    controller._prepared_panels_v13 = lambda: (prepared, [{}] * 280)
    controller._base_probe_v13 = lambda _prepared, prompts: {
        "schema": "synthetic-probe", "request_count": len(prompts),
    }
    controller._restore_all_engines_exact = lambda: calls.__setitem__(
        "restores", calls["restores"] + 1,
    )
    boundary = {"passed": True, "iteration": 0}
    boundary["audit_sha256"] = anchor_v15a.canonical_sha256(boundary)
    controller._population_boundary_audit_v4 = lambda _iteration: boundary

    def score(panel, _dense, outputs):
        marker = outputs[0]
        panel_index = list(anchor_v15a.anchor_v13.PANEL_NAMES_V13).index(
            panel["name"]
        )
        coordinate = float(seed_index[marker["seed"]]) - 15.5
        central = coordinate * (1.0 + 0.05 * panel_index)
        if marker["negate"]:
            central = -central
        value = 10.0 + central
        return {
            "weighted_mean": value,
            "unweighted_mean": value,
            "stratum_unweighted_means": {
                stratum: value
                for stratum in anchor_v15a.anchor_v13.panel_sampler.STRATA
            },
            "weighted_stratum_contributions": {
                stratum: value
                * anchor_v15a.anchor_v13.STRATUM_POPULATION_V13[stratum]
                / 310.0
                for stratum in anchor_v15a.anchor_v13.panel_sampler.STRATA
            },
            "scored_answer_tokens": 56,
            "dense_result_sha256": anchor_v15a.canonical_sha256(
                [panel["name"], marker]
            ),
        }

    monkeypatch.setitem(
        anchor_v15a._estimate_train_panels_v15a.__globals__,
        "_score_panel_outputs_v13",
        score,
    )
    artifact = controller.estimate_train_panels_v15a(
        anchor_v15a.PERTURBATION_SEEDS_V15A
    )
    assert calls["restores"] == 16
    assert len(calls["generate"]) == 64
    assert {name for _engine, name, _args in calls["rpc"]} == {
        "perturb_self_weights", "verify_self_exact_reference",
    }
    assert artifact["perturbation_basis"]["basis_sha256"] == (
        anchor_v15a.PERTURBATION_BASIS_SHA256_V15A
    )
    assert artifact["hardware_coverage"]["signed_waves"] == 16


def test_v15a_compact_summary_binds_dense_hashes_without_persisting_vectors():
    summary = anchor_v15a.compact_arm_summary_v15a(
        "middle_late", synthetic_diagnostic(),
    )
    assert summary["persisted_response_vectors"] is False
    assert summary["persisted_row_content"] is False
    assert len(summary["dense_direction_sign_hash_manifest_sha256"]) == 64
    assert summary["integrity_audits"][
        "dense_direction_sign_hash_coverage_passed"
    ] is True
    assert not {"responses", "coefficients", "questions", "answers"}.intersection(
        walk_keys(summary)
    )


def test_v15a_parser_firewall_runs_before_parser(monkeypatch):
    parser = driver_v15a._parser_v15a()
    for action in parser._actions:
        surface = [*action.option_strings]
        if isinstance(action.default, (str, Path)):
            surface.append(str(action.default))
        assert not any(
            forbidden in value.lower()
            for value in surface
            for forbidden in driver_v15a.FORBIDDEN_SURFACE_TOKENS_V15A
        )
    monkeypatch.setattr(
        driver_v15a, "_parser_v15a", lambda: pytest.fail("parser reached")
    )
    with pytest.raises(ValueError, match="rejects non-train surface"):
        driver_v15a.main(["--heldout", "forbidden.jsonl"])


@pytest.mark.parametrize(
    "name",
    [
        *driver_v15a.MOE_ENVIRONMENT_OVERRIDES_V15A,
        "VLLM_EXPERIMENTAL_MOE_BACKEND",
    ],
)
def test_v15a_rejects_tuned_and_every_backend_override(name, monkeypatch):
    monkeypatch.setenv(name, "/tmp/confound")
    with pytest.raises(ValueError, match="backend environment overrides unset"):
        driver_v15a._moe_environment_binding_v15a()


def test_v15a_dry_run_has_exact_recipe_payload_and_no_gpu(monkeypatch, capsys):
    monkeypatch.setattr(
        driver_v15a, "_make_trainer_v15a",
        lambda *_args, **_kwargs: pytest.fail("dry run constructed trainer"),
    )
    first = driver_v15a.main(cli())
    second = driver_v15a.main(cli())
    assert first == second
    audit = driver_v15a.audit_dry_run_payload_v15a(first)
    recipe = first["recipe"]
    assert first["gpu_launched"] is False
    assert recipe["paired_architecture"]["arm_order"] == [
        "middle_late", "back",
    ]
    assert recipe["paired_architecture"]["same_fresh_basis_both_arms"] is True
    assert recipe["perturbation_basis"]["seeds"] == (
        anchor_v15a.PERTURBATION_SEEDS_V15A
    )
    assert recipe["alpha"] == 0.0 and recipe["model_update_allowed"] is False
    assert recipe["hardware"] == {
        "engine_count": 4,
        "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3],
        "complete_four_direction_waves_required": True,
        "population_waves_per_arm": 8,
        "signed_waves_per_arm": 16,
    }
    assert recipe["moe_kernel_environment"][
        "v026_tuned_config_commit_82cdf8e_used"
    ] is False
    assert audit["recipe_content_sha256"] == recipe[
        "content_sha256_before_self_field"
    ]
    assert "paired-architecture-launch-dry-run-v15a" in capsys.readouterr().out


def test_v15a_runtime_requires_exact_bundle_recipe_and_four_gpu_contract():
    implementation = driver_v15a.implementation_identity_v15a()
    args = driver_v15a._parser_v15a().parse_args(["--v15a-dry-run"])
    for attribute, value in (
        ("n_vllm_engines", 3), ("population_size", 28),
        ("alpha", 0.1), ("experiment_name", "changed"),
    ):
        changed = copy.copy(args)
        setattr(changed, attribute, value)
        with pytest.raises(ValueError, match="paired four-GPU recipe"):
            driver_v15a.validate_runtime_v15a(
                changed, plans(), implementation,
            )
    args.expected_implementation_bundle_sha256 = "0" * 64
    with pytest.raises(ValueError, match="implementation bundle"):
        driver_v15a.validate_runtime_v15a(args, plans(), implementation)
    payload = driver_v15a.main(cli())
    recipe_hash = payload["recipe"]["content_sha256_before_self_field"]
    assert driver_v15a.main(cli([
        "--expected-recipe-content-sha256", recipe_hash,
    ]))["recipe"] == payload["recipe"]
    with pytest.raises(ValueError, match="recipe content hash"):
        driver_v15a.main(cli([
            "--expected-recipe-content-sha256", "0" * 64,
        ]))


def test_v15a_source_provenance_rejects_uncommitted_runtime(monkeypatch):
    implementation = {
        "files": {
            "synthetic": {
                "path": str(Path(driver_v15a.__file__).resolve()),
                "file_sha256": "0" * 64,
            },
        },
        "bundle_sha256": "1" * 64,
    }

    def fake(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "head\n"
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(driver_v15a.subprocess, "check_output", fake)
    with pytest.raises(RuntimeError, match="requires committed implementation"):
        driver_v15a._source_provenance_v15a(implementation)


def test_v15a_second_arm_failure_closes_both_trainers_and_is_durable(
    tmp_path, monkeypatch,
):
    calls = []

    class Trainer:
        def __init__(self, arm):
            self.arm = arm

        def configure_train_panels_v15a(self, _panels, *, frozen_layer_plan):
            calls.append((self.arm, "configure"))
            assert frozen_layer_plan["plan_sha256"] == (
                prereg_v15a.LAYER_PLANS_V15A[self.arm]["plan_sha256"]
            )
            return {
                "layer_plan_install": {"arm": self.arm},
                "reference_identity": {"same": True},
                "panel_bundle_content_sha256": (
                    anchor_v15a.anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
                ),
            }

        def estimate_train_panels_v15a(self, seeds):
            calls.append((self.arm, "estimate"))
            assert seeds == anchor_v15a.PERTURBATION_SEEDS_V15A
            if self.arm == "back":
                raise RuntimeError("synthetic second-arm failure")
            return synthetic_diagnostic(1.0)

    arm_by_plan = {
        spec["plan_sha256"]: name
        for name, spec in prereg_v15a.LAYER_PLANS_V15A.items()
    }
    monkeypatch.setattr(driver_v15a, "FROZEN_OUTPUT_DIRECTORY_V15A", tmp_path)
    monkeypatch.setattr(
        driver_v15a, "_source_provenance_v15a",
        lambda implementation: {
            "schema": "synthetic-source",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
        },
    )
    monkeypatch.setattr(
        driver_v15a,
        "_make_trainer_v15a",
        lambda _args, bundle: Trainer(arm_by_plan[bundle["plan_sha256"]]),
    )
    monkeypatch.setattr(
        driver_v15a.base,
        "close_trainer",
        lambda trainer: calls.append((trainer.arm, "close")),
    )
    args = driver_v15a._parser_v15a().parse_args([])
    args.output_directory = str(tmp_path)
    implementation = driver_v15a.implementation_identity_v15a()
    with pytest.raises(RuntimeError, match="synthetic second-arm failure"):
        driver_v15a.run_exact_v15a(
            args, plans(), anchor_v15a.anchor_v13.load_panel_bundle_v13(),
            implementation, {"schema": "synthetic-recipe"},
        )
    assert calls == [
        ("middle_late", "configure"), ("middle_late", "estimate"),
        ("middle_late", "close"), ("back", "configure"),
        ("back", "estimate"), ("back", "close"),
    ]
    attempt = json.loads(driver_v15a._attempt_path_v15a().read_text())
    assert attempt["status"] == "failed"
    assert attempt["active_arm"] == "back"
    assert attempt["failure"]["message"] == "synthetic second-arm failure"
    assert tuple(attempt["completed_arm_summary_bindings"]) == ("middle_late",)
    assert attempt["report_exists_after_attempt"] is False
    with pytest.raises(ValueError, match="fresh exclusive"):
        driver_v15a.run_exact_v15a(
            args, plans(), anchor_v15a.anchor_v13.load_panel_bundle_v13(),
            implementation, {"schema": "synthetic-recipe"},
        )


def test_v15a_success_report_is_compact_bound_and_contains_no_raw_vectors(
    tmp_path, monkeypatch,
):
    class Trainer:
        def __init__(self, arm):
            self.arm = arm

        def configure_train_panels_v15a(self, _panels, *, frozen_layer_plan):
            return {
                "layer_plan_install": {
                    "arm": self.arm,
                    "plan_sha256": frozen_layer_plan["plan_sha256"],
                },
                "reference_identity": {"same": True},
                "panel_bundle_content_sha256": (
                    anchor_v15a.anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
                ),
            }

        def estimate_train_panels_v15a(self, _seeds):
            return synthetic_diagnostic(1.0 if self.arm == "middle_late" else 1.1)

    arm_by_plan = {
        spec["plan_sha256"]: name
        for name, spec in prereg_v15a.LAYER_PLANS_V15A.items()
    }
    monkeypatch.setattr(driver_v15a, "FROZEN_OUTPUT_DIRECTORY_V15A", tmp_path)
    monkeypatch.setattr(
        driver_v15a, "_source_provenance_v15a",
        lambda implementation: {
            "schema": "synthetic-source",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
        },
    )
    monkeypatch.setattr(
        driver_v15a,
        "_make_trainer_v15a",
        lambda _args, bundle: Trainer(arm_by_plan[bundle["plan_sha256"]]),
    )
    monkeypatch.setattr(driver_v15a.base, "close_trainer", lambda _trainer: None)
    args = driver_v15a._parser_v15a().parse_args([])
    args.output_directory = str(tmp_path)
    implementation = driver_v15a.implementation_identity_v15a()
    report = driver_v15a.run_exact_v15a(
        args, plans(), anchor_v15a.anchor_v13.load_panel_bundle_v13(),
        implementation, {"schema": "synthetic-recipe"},
    )
    assert report["persisted_response_vectors_or_row_content"] is False
    assert not {"responses", "coefficients", "questions", "answers"}.intersection(
        walk_keys(report)
    )
    assert all(
        configuration["persisted_configuration_payload"] is False
        for configuration in report["configurations"].values()
    )
    attempt = json.loads(driver_v15a._attempt_path_v15a().read_text())
    report_path = tmp_path / driver_v15a.EXPERIMENT_NAME_V15A / (
        driver_v15a.REPORT_NAME_V15A
    )
    assert attempt["status"] == "complete"
    assert attempt["report_binding"] == {
        "path": str(report_path),
        "file_sha256": driver_v15a._file_sha256(report_path),
        "content_sha256": report["content_sha256_before_self_field"],
    }
    persisted = json.loads(report_path.read_text())
    assert persisted == report
