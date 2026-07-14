#!/usr/bin/env python3

import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

import eggroll_es_worker_v11c as worker_v11c
import eggroll_es_worker_v13 as worker_v13
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_train_panels_v13 as driver_v13
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v11b as anchor_v11b
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


PLAN_SHA = driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10


def load_bundle():
    spec = anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return anchor_v13.load_frozen_layer_plan_v13(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=anchor_v11.MODEL_CONFIG_SHA256_V11,
    )


def cli(extra=None):
    spec = anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256", anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v13-dry-run",
        *(extra or []),
    ]


def _responses(offsets=None):
    offsets = offsets or {}
    result = {}
    base_vector = [float(index - 15) for index in range(32)]
    for panel_index, name in enumerate(anchor_v13.PANEL_NAMES_V13):
        shift = offsets.get(name, 0.0)
        central = [
            (1.0 + 0.05 * panel_index) * value + shift
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
                    for sign in anchor_v13.SIGNS_V13
                }
                for stratum in anchor_v13.panel_sampler.STRATA
            },
            "weighted_stratum_contributions": {
                stratum: {
                    sign: [
                        value * anchor_v13.STRATUM_POPULATION_V13[stratum] / 310.0
                        for value in weighted[sign]
                    ]
                    for sign in anchor_v13.SIGNS_V13
                }
                for stratum in anchor_v13.panel_sampler.STRATA
            },
            "dense_result_sha256": {
                "plus": [
                    driver_v1.canonical_sha256(["p", panel_index, index])
                    for index in range(32)
                ],
                "minus": [
                    driver_v1.canonical_sha256(["m", panel_index, index])
                    for index in range(32)
                ],
            },
        }
    return result


def test_v13_materialization_binds_exact_five_train_panels():
    bundle = anchor_v13.load_panel_bundle_v13()
    anchor_v13.validate_panel_bundle_v13(bundle)
    assert tuple(bundle["panels"]) == anchor_v13.PANEL_NAMES_V13
    assert [bundle["panels"][name]["role"] for name in bundle["panels"]] == [
        "optimization", "optimization", "optimization",
        "train_only_screen", "train_only_screen",
    ]
    assert all(
        len(bundle["panels"][name]["questions"]) == 56
        for name in bundle["panels"]
    )
    assert bundle["manifest"]["file_sha256"] == (
        anchor_v13.PANEL_MANIFEST_FILE_SHA256_V13
    )
    assert bundle["source"]["file_sha256"] == (
        anchor_v13.TRAIN_SOURCE_FILE_SHA256_V13
    )
    tampered = copy.deepcopy(bundle)
    tampered["panels"]["optimization_0"]["questions"][0] += " changed"
    tampered["content_sha256_before_self_field"] = driver_v1.canonical_sha256({
        key: value for key, value in tampered.items()
        if key != "content_sha256_before_self_field"
    })
    # Re-hashing cannot repair a mismatch to the ordered committed row identity.
    with pytest.raises(RuntimeError, match="panel bundle changed|contract changed"):
        anchor_v13.validate_panel_bundle_v13(tampered)


def test_v13_preregistered_aggregate_excludes_both_screens():
    original = anchor_v13.analyze_panel_responses_v13(_responses())
    changed = anchor_v13.analyze_panel_responses_v13(_responses({
        "train_screen_0": 1000.0,
        "train_screen_1": -1000.0,
    }))
    aggregate = original["robust_optimization_aggregate"]
    assert aggregate == changed["robust_optimization_aggregate"]
    assert aggregate["input_panels"] == list(anchor_v13.OPTIMIZATION_PANELS_V13)
    assert aggregate["excluded_panels"] == list(anchor_v13.TRAIN_SCREENS_V13)
    assert set(original["optimization_pairwise"]) == {
        "optimization_0__optimization_1",
        "optimization_0__optimization_2",
        "optimization_1__optimization_2",
    }
    assert set(original["train_screen_transfer"]) == set(
        anchor_v13.TRAIN_SCREENS_V13
    )


def test_v13_worker_and_controller_make_model_updates_unreachable():
    assert issubclass(
        worker_v13.TrainPanelDiagnosticWorkerExtensionV13,
        worker_v11c.ResidentSignAuditWorkerExtensionV11C,
    )
    worker = object.__new__(worker_v13.TrainPanelDiagnosticWorkerExtensionV13)
    for method in (
        "prepare_sharded_seed_update_v4", "execute_prepared_seed_update_v4",
        "commit_prepared_seed_update_v4", "update_weights_from_seeds",
    ):
        with pytest.raises(RuntimeError, match="forbids model updates"):
            getattr(worker, method)()
    with pytest.raises(RuntimeError, match="forbids model updates"):
        anchor_v13.TrainPanelDiagnosticMixinV13.apply_seed_coefficients(
            object(), {}, 0.0,
        )


def test_v13_trainer_extends_repaired_v11c_manifest_path():
    trainer = anchor_v13.load_trainer(load_bundle())
    assert anchor_v11b.DualManifestResidentSignContractMixinV11B in trainer.__mro__
    assert trainer.estimate_train_panels_v13.__module__ == anchor_v13.__name__
    assert trainer.estimate_step_coefficients.__module__ == anchor_v13.__name__
    assert trainer.apply_seed_coefficients.__module__ == anchor_v13.__name__
    assert trainer.launch_engines.__globals__["WORKER_EXTENSION"] == (
        anchor_v13.WORKER_EXTENSION
    )


def test_v13_templated_panel_identity_is_checked_before_tokenization(monkeypatch):
    controller = object.__new__(anchor_v13.TrainPanelDiagnosticMixinV13)
    controller._v13_panel_bundle = anchor_v13.load_panel_bundle_v13()
    monkeypatch.setattr(
        anchor_v13.base, "specialist_template", lambda question: "changed " + question,
    )
    with pytest.raises(RuntimeError, match="before generation"):
        controller._prepared_panels_v13()


def test_v13_parser_has_no_nontrain_option_or_default_and_rejects_tokens():
    parser = driver_v13._parser_v13()
    for action in parser._actions:
        surface = [*action.option_strings]
        if isinstance(action.default, (str, Path)):
            surface.append(str(action.default))
        assert not any(
            forbidden in value.lower()
            for value in surface
            for forbidden in driver_v13.FORBIDDEN_SURFACE_TOKENS_V13
        )
    with pytest.raises(ValueError, match="rejects non-train surface"):
        driver_v13.main(cli(["--ood-prose-jsonl", "/tmp/x"]))
    with pytest.raises(ValueError, match="rejects non-train surface"):
        driver_v13.main(cli(["--panel-manifest", "/tmp/heldout.json"]))


def test_v13_dry_run_is_launch_shaped_without_constructing_trainer(
    monkeypatch, capsys,
):
    monkeypatch.setattr(
        driver_v13, "_make_trainer_v13",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dry run constructed a trainer")
        ),
    )
    payload = driver_v13.main(cli())
    assert payload["schema"] == "eggroll-es-five-panel-launch-dry-run-v13"
    assert payload["gpu_launched"] is False
    assert payload["recipe"]["alpha"] == 0.0
    assert payload["recipe"]["model_update_allowed"] is False
    assert payload["recipe"]["optimization_panels"] == [
        "optimization_0", "optimization_1", "optimization_2",
    ]
    assert payload["recipe"]["train_screens"] == [
        "train_screen_0", "train_screen_1",
    ]
    assert payload["recipe"]["hardware"] == {
        "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
        "complete_wave_required": True,
    }
    assert "five-panel-launch-dry-run-v13" in capsys.readouterr().out


def test_v13_runtime_fails_closed_on_gpu_and_implementation_changes():
    bundle = load_bundle()
    implementation = driver_v13.implementation_identity_v13()
    args = driver_v13._parser_v13().parse_args(["--v13-dry-run"])
    args.n_vllm_engines = 3
    with pytest.raises(ValueError, match="four-GPU recipe"):
        driver_v13.validate_runtime_v13(args, bundle, implementation)
    args.n_vllm_engines = 4
    args.expected_implementation_bundle_sha256 = "0" * 64
    with pytest.raises(ValueError, match="implementation bundle"):
        driver_v13.validate_runtime_v13(args, bundle, implementation)

    alternate_bundle = copy.deepcopy(bundle)
    alternate_bundle["path"] = str(Path(bundle["path"]).with_name("copied.json"))
    args.expected_implementation_bundle_sha256 = None
    with pytest.raises(ValueError, match="identity|four-GPU recipe"):
        driver_v13.validate_runtime_v13(args, alternate_bundle, implementation)


def test_v13_exclusive_artifact_creation_is_race_safe(tmp_path):
    output = tmp_path / "attempt.json"
    barrier = threading.Barrier(2)

    def write(index):
        barrier.wait()
        try:
            driver_v13._exclusive_write_json(output, {"writer": index})
            return "written"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, (0, 1)))
    assert sorted(outcomes) == ["rejected", "written"]
    value = json.loads(output.read_text())
    assert value["content_sha256_before_self_field"] == driver_v1.canonical_sha256({
        "writer": value["writer"],
    })


def test_v13_failed_launch_attempt_is_durable_and_hermetic(tmp_path, monkeypatch):
    monkeypatch.setattr(driver_v13, "FROZEN_OUTPUT_DIRECTORY_V13", tmp_path)
    monkeypatch.setattr(
        driver_v13, "_source_provenance_v13",
        lambda implementation: {
            "schema": "synthetic-v13-source",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
        },
    )
    monkeypatch.setattr(
        driver_v13, "_make_trainer_v13",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic pre-engine failure")
        ),
    )
    bundle = load_bundle()
    implementation = driver_v13.implementation_identity_v13()
    args = driver_v13._parser_v13().parse_args([])
    args.output_directory = str(tmp_path)
    arrow = {"arrow_sha256": anchor_v13.TRAIN_ARROW_FILE_SHA256_V13}
    panels = anchor_v13.load_panel_bundle_v13()
    recipe = {
        "schema": "synthetic-v13-recipe",
        "content_sha256_before_self_field": "a" * 64,
    }
    with pytest.raises(RuntimeError, match="synthetic pre-engine failure"):
        driver_v13.run_exact_v13(
            args, bundle, arrow, panels, implementation, recipe,
        )
    attempt_path = driver_v13._attempt_path_v13()
    attempt = json.loads(attempt_path.read_text())
    assert attempt["status"] == "failed"
    assert attempt["failure"]["type"] == "RuntimeError"
    assert "synthetic pre-engine failure" in attempt["failure"]["traceback"]
    assert attempt["model_update_applied"] is False
    assert attempt["sealed_or_nontrain_surface_opened"] is False
    assert attempt["content_sha256_before_self_field"] == driver_v1.canonical_sha256({
        key: value for key, value in attempt.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(ValueError, match="fresh exclusive"):
        driver_v13.run_exact_v13(
            args, bundle, arrow, panels, implementation, recipe,
        )


def test_v13_cleanup_failure_is_recorded_before_any_report(tmp_path, monkeypatch):
    class SyntheticTrainer:
        def configure_train_panels_v13(self, panels, *, frozen_layer_plan):
            del panels, frozen_layer_plan
            return {"configured": True}

        def estimate_train_panels_v13(self, seeds):
            assert seeds == anchor_v13.PERTURBATION_SEEDS_V13
            return {"synthetic": True}

    monkeypatch.setattr(driver_v13, "FROZEN_OUTPUT_DIRECTORY_V13", tmp_path)
    monkeypatch.setattr(
        driver_v13, "_source_provenance_v13",
        lambda implementation: {
            "schema": "synthetic-v13-source",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
        },
    )
    monkeypatch.setattr(
        driver_v13, "_make_trainer_v13", lambda *_args, **_kwargs: SyntheticTrainer(),
    )
    monkeypatch.setattr(
        driver_v13.anchor_v13, "validate_diagnostic_v13", lambda value: value,
    )
    monkeypatch.setattr(
        driver_v13.base, "close_trainer",
        lambda _trainer: (_ for _ in ()).throw(RuntimeError("cleanup failed")),
    )
    bundle = load_bundle()
    implementation = driver_v13.implementation_identity_v13()
    args = driver_v13._parser_v13().parse_args([])
    args.output_directory = str(tmp_path)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        driver_v13.run_exact_v13(
            args, bundle,
            {"arrow_sha256": anchor_v13.TRAIN_ARROW_FILE_SHA256_V13},
            anchor_v13.load_panel_bundle_v13(), implementation,
            {"schema": "synthetic-v13-recipe"},
        )
    attempt = json.loads(driver_v13._attempt_path_v13().read_text())
    assert attempt["status"] == "failed"
    assert attempt["failure"]["type"] == "RuntimeError"
    assert attempt["failure"]["message"] == "cleanup failed"
    assert attempt["report_exists_after_attempt"] is False
    assert not (tmp_path / driver_v13.EXPERIMENT_NAME_V13 / driver_v13.REPORT_NAME_V13).exists()
