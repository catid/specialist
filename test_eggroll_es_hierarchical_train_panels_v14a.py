#!/usr/bin/env python3

import copy
import json
import math
import subprocess
from pathlib import Path

import pytest

import eggroll_es_hierarchical_preregistration_v14a as prereg_v14a
import run_eggroll_es_hierarchical_train_panels_v14a as driver_v14a
import train_eggroll_es_specialist_anchor_v11b as anchor_v11b
import train_eggroll_es_specialist_anchor_v14a as anchor_v14a


PLAN_SHA = driver_v14a.driver_v13.driver_v10.MIDDLE_LATE_PLAN_SHA256_V10


def load_bundle():
    spec = driver_v14a.driver_v13.anchor_v11.FROZEN_STABILITY_PLANS_V11[
        PLAN_SHA
    ]
    return anchor_v14a.load_frozen_layer_plan_v14a(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=(
            driver_v14a.driver_v13.anchor_v11.MODEL_CONFIG_SHA256_V11
        ),
    )


def cli(extra=None):
    spec = driver_v14a.driver_v13.anchor_v11.FROZEN_STABILITY_PLANS_V11[
        PLAN_SHA
    ]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256",
        driver_v14a.driver_v13.anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v14a-dry-run",
        *(extra or []),
    ]


def synthetic_responses():
    base_vector = [float(index) - 15.5 for index in range(32)]

    def signed(scale):
        central = [scale * value for value in base_vector]
        return {
            "plus": [10.0 + value for value in central],
            "minus": [10.0 - value for value in central],
        }

    return {
        "full_frame_sign_scores": signed(1.0),
        "matched56_sign_scores": {
            name: signed(1.0 + 0.05 * index)
            for index, name in enumerate(prereg_v14a.PANEL_NAMES_V14A)
        },
        "complement_sign_scores": {
            name: signed(1.4 + 0.05 * index)
            for index, name in enumerate(prereg_v14a.PANEL_NAMES_V14A[3:])
        },
        "dense_result_sha256": {
            sign: [
                anchor_v14a.canonical_sha256([sign, index])
                for index in range(32)
            ] for sign in anchor_v14a.SIGNS_V14A
        },
    }


def test_v14a_materialization_binds_one_full_frame_and_five_matched_panels():
    bundle = anchor_v14a.load_panel_bundle_v14a()
    anchor_v14a.validate_panel_bundle_v14a(bundle)
    full = bundle["full_frame"]
    assert len(full["questions"]) == len(full["answers"]) == 310
    assert len(full["document_sha256s"]) == len(set(full["document_sha256s"])) == 310
    assert tuple(bundle["matched56"]) == prereg_v14a.PANEL_NAMES_V14A
    assert all(
        len(panel["positions"]) == len(set(panel["positions"])) == 56
        for panel in bundle["matched56"].values()
    )
    assert all(
        math.isclose(math.fsum(panel["weights"]), 310.0, abs_tol=1e-12)
        for panel in bundle["matched56"].values()
    )
    assert bundle["source"]["documents"] == 310
    assert bundle["source"]["frame_sha256"] == prereg_v14a.FRAME_SHA256_V14A


def test_v14a_score_reduces_dense_rewards_immediately_to_exact_aggregates(
    monkeypatch,
):
    bundle = anchor_v14a.load_panel_bundle_v14a()
    rewards = [float(index) / 100.0 for index in range(310)]
    dense = {
        "schema": "synthetic-dense-result",
        "examples": [
            {"mean_answer_token_logprob": reward} for reward in rewards
        ],
    }
    monkeypatch.setattr(
        anchor_v14a.anchor_v4, "score_gold_answer_outputs_v4",
        lambda _items, _outputs: copy.deepcopy(dense),
    )
    reduced = anchor_v14a._score_full_outputs([], [], bundle)
    assert set(reduced) == {
        "full_frame", "matched56", "complements", "dense_result_sha256",
    }
    assert reduced["full_frame"] == pytest.approx(math.fsum(rewards) / 310.0)
    for name, panel in bundle["matched56"].items():
        expected = math.fsum(
            weight * rewards[position]
            for position, weight in zip(panel["positions"], panel["weights"])
        ) / 310.0
        assert reduced["matched56"][name] == pytest.approx(expected)
    for name in prereg_v14a.PANEL_NAMES_V14A[3:]:
        screen = set(bundle["matched56"][name]["positions"])
        expected = math.fsum(
            rewards[position] for position in range(310)
            if position not in screen
        ) / 254.0
        assert reduced["complements"][name] == pytest.approx(expected)
    assert all(not isinstance(value, list) for value in reduced.values())


def test_v14a_analysis_standardizes_all_eight_vectors_and_uses_full_frame():
    bundle = anchor_v14a.load_panel_bundle_v14a()
    responses = synthetic_responses()
    analysis = anchor_v14a.analyze_responses_v14a(responses, bundle)
    assert set(analysis["standardization"]) == {
        "full_frame", *prereg_v14a.PANEL_NAMES_V14A,
        "complement_train_screen_0", "complement_train_screen_1",
    }
    assert all(
        item["zero_spread"] is False
        for item in analysis["standardization"].values()
    )
    full, _summary = anchor_v14a._standardize(
        [float(index) - 15.5 for index in range(32)]
    )
    aggregate = analysis["candidate_summary"]["robust_aggregate"]
    assert aggregate["coefficient_sha256"] == anchor_v14a.coefficient_sha256(
        anchor_v14a.PERTURBATION_SEEDS_V14A, full,
    )
    assert aggregate["nonzero_coordinate_count"] == 32
    assert analysis["candidate_summary"]["all_panel_spreads_nonzero"] is True
    assert analysis["promotion_gate"][
        "eligible_for_train_only_sampler_adoption"
    ] is True
    assert analysis["promotion_gate"]["eligible_for_model_update"] is False


def test_v14a_response_contract_rejects_per_document_or_nonfinite_payloads():
    valid = synthetic_responses()
    anchor_v14a.validate_responses_v14a(valid)
    extra = copy.deepcopy(valid)
    extra["per_document_rewards"] = [[0.0] * 310]
    with pytest.raises(RuntimeError, match="response shape"):
        anchor_v14a.validate_responses_v14a(extra)
    bad = copy.deepcopy(valid)
    bad["full_frame_sign_scores"]["plus"][0] = float("nan")
    with pytest.raises(RuntimeError, match="response vectors"):
        anchor_v14a.validate_responses_v14a(bad)


def test_v14a_trainer_inherits_update_disabled_v13_worker_and_refuses_update():
    trainer = anchor_v14a.load_trainer(load_bundle())
    assert anchor_v11b.DualManifestResidentSignContractMixinV11B in trainer.__mro__
    assert trainer.launch_engines.__globals__["WORKER_EXTENSION"] == (
        anchor_v14a.WORKER_EXTENSION
    )
    assert anchor_v14a.WORKER_EXTENSION == (
        "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
    )
    assert trainer.estimate_full_frame_v14a.__module__ == anchor_v14a.__name__
    with pytest.raises(RuntimeError, match="forbids model updates"):
        anchor_v14a.FullFrameDiagnosticMixinV14A.apply_seed_coefficients(
            object(), {}, 0.1,
        )


def test_v14a_exact_four_engine_signed_wave_contract_without_gpu(monkeypatch):
    calls = {"generate": [], "rpc": [], "restores": 0}
    seed_index = {
        seed: index for index, seed in enumerate(
            anchor_v14a.PERTURBATION_SEEDS_V14A
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
            self.generate = Remote(self.run_generate)

        def rpc(self, name, args):
            calls["rpc"].append((self.index, name, args))
            if name == "perturb_self_weights":
                seed, sigma, negate = args
                assert sigma == 0.0003
                self.state = (int(seed), bool(negate))
                return {"passed": True}
            if name == "verify_self_exact_reference":
                assert args == ()
                return {"passed": True}
            raise AssertionError(f"unexpected RPC {name}")

        def run_generate(self, prompts, _params, *, use_tqdm):
            assert len(prompts) == 310
            assert use_tqdm is False
            calls["generate"].append((self.index, self.state, len(prompts)))
            marker = {"seed": self.state[0], "negate": self.state[1]}
            return [marker] * 310

    controller = object.__new__(anchor_v14a.FullFrameDiagnosticMixinV14A)
    controller.engines = [Engine(index) for index in range(4)]
    controller.sigma = 0.0003
    controller._v14a_panel_bundle = anchor_v14a.load_panel_bundle_v14a()
    controller._resolve = lambda values: values
    controller._dense_sampling_params_v4 = lambda _seed: {"synthetic": True}
    controller._prepared_full_frame_v14a = lambda: (
        [{}] * 310, [{}] * 310, anchor_v14a.FULL_TEMPLATED_QA_SHA256_V14A,
    )
    controller._base_probe_v14a = lambda _dense, prompts: {
        "schema": "synthetic-base-probe", "request_count": len(prompts),
    }

    def restore():
        calls["restores"] += 1

    controller._restore_all_engines_exact = restore
    boundary = {"passed": True, "iteration": 0}
    boundary["audit_sha256"] = anchor_v14a.canonical_sha256(boundary)
    controller._population_boundary_audit_v4 = lambda iteration: (
        boundary if iteration == 0 else pytest.fail("wrong boundary")
    )

    def score(_dense, outputs, _bundle):
        marker = outputs[0]
        coordinate = float(seed_index[marker["seed"]]) - 15.5
        central = -coordinate if marker["negate"] else coordinate
        return {
            "full_frame": 10.0 + central,
            "matched56": {
                name: 10.0 + central * (1.0 + 0.05 * index)
                for index, name in enumerate(prereg_v14a.PANEL_NAMES_V14A)
            },
            "complements": {
                name: 10.0 + central * (1.4 + 0.05 * index)
                for index, name in enumerate(prereg_v14a.PANEL_NAMES_V14A[3:])
            },
            "dense_result_sha256": anchor_v14a.canonical_sha256(marker),
        }

    monkeypatch.setattr(anchor_v14a, "_score_full_outputs", score)
    artifact = controller.estimate_full_frame_v14a(
        anchor_v14a.PERTURBATION_SEEDS_V14A
    )
    assert calls["restores"] == 16
    assert len(calls["generate"]) == 64
    assert all(count == 310 for _engine, _state, count in calls["generate"])
    assert all(
        sum(engine == engine_index for engine, _state, _count in calls["generate"])
        == 16
        for engine_index in range(4)
    )
    rpc_names = [name for _engine, name, _args in calls["rpc"]]
    assert set(rpc_names) == {
        "perturb_self_weights", "verify_self_exact_reference",
    }
    assert rpc_names.count("perturb_self_weights") == 64
    assert artifact["hardware_coverage"]["population_waves"] == 8
    assert artifact["generation_contract"] == {
        "prompts_per_engine_per_sign": 310,
        "generation_calls_per_engine_per_sign": 1,
        "matched_and_crossfit_responses_derived_without_generation": True,
    }
    assert "per_document_rewards" not in json.dumps(artifact)


def test_v14a_parser_surface_is_train_only_and_firewall_runs_before_parser(
    monkeypatch,
):
    parser = driver_v14a._parser_v14a()
    for action in parser._actions:
        surface = [*action.option_strings]
        if isinstance(action.default, (str, Path)):
            surface.append(str(action.default))
        assert not any(
            forbidden in value.lower()
            for value in surface
            for forbidden in driver_v14a.FORBIDDEN_SURFACE_TOKENS_V14A
        )
    monkeypatch.setattr(
        driver_v14a, "_parser_v14a",
        lambda: pytest.fail("parser reached before firewall"),
    )
    with pytest.raises(ValueError, match="rejects non-train surface"):
        driver_v14a.main(["--heldout", "forbidden.jsonl"])
    with pytest.raises(ValueError, match="rejects non-train surface"):
        driver_v14a.main(["--benchmark-results", "forbidden.json"])


def test_v14a_dry_run_binds_recipe_without_constructing_trainer(
    monkeypatch, capsys,
):
    monkeypatch.setattr(
        driver_v14a, "_make_trainer_v14a",
        lambda *_args, **_kwargs: pytest.fail("dry run constructed trainer"),
    )
    payload = driver_v14a.main(cli())
    recipe = payload["recipe"]
    assert payload["gpu_launched"] is False
    assert recipe["alpha"] == 0.0
    assert recipe["model_update_allowed"] is False
    assert recipe["generation"] == {
        "prompts_per_engine_per_sign": 310,
        "generation_calls_per_engine_per_sign": 1,
        "matched_and_crossfit_responses_derived_without_generation": True,
    }
    assert recipe["hardware"] == {
        "engine_count": 4, "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3], "complete_wave_required": True,
    }
    assert recipe["preregistration"]["file_sha256"] == (
        "d27052ee26d9ba5dd4383491b3d093d0a2f9469ddb4a073909a2b6590e0cba3e"
    )
    assert "full-frame-launch-dry-run-v14a" in capsys.readouterr().out


def test_v14a_source_bundle_includes_prereg_evidence_sampler_and_v13_lineage():
    implementation = driver_v14a.implementation_identity_v14a()
    files = implementation["files"]
    assert set(driver_v14a.driver_v13.IMPLEMENTATION_PATHS_V13) <= set(files)
    assert {
        "v14_sampler", "v14_protocol", "v14a_prereg_module",
        "v14a_prereg_tests", "v14a_protocol", "v14a_preregistration",
        "v13b_evidence_builder", "v13b_compact_evidence",
        "v12_evidence_builder", "v12_negative_evidence",
        "trainer_v14a", "driver_v14a", "tests_v14a",
    } <= set(files)


def test_v14a_real_source_provenance_rejects_uncommitted_or_dirty_source(
    monkeypatch,
):
    implementation = {
        "files": {"synthetic": {
            "path": str(Path(driver_v14a.__file__).resolve()),
            "file_sha256": "0" * 64,
        }},
        "bundle_sha256": "1" * 64,
    }

    def uncommitted(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            assert kwargs.get("text") is True
            return "synthetic-head\n"
        raise subprocess.CalledProcessError(128, command)

    monkeypatch.setattr(driver_v14a.subprocess, "check_output", uncommitted)
    with pytest.raises(RuntimeError, match="requires committed implementation"):
        driver_v14a._source_provenance_v14a(implementation)

    def dirty(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "synthetic-head\n"
        return b"different committed bytes"

    monkeypatch.setattr(driver_v14a.subprocess, "check_output", dirty)
    with pytest.raises(RuntimeError, match="differs from committed HEAD"):
        driver_v14a._source_provenance_v14a(implementation)


def test_v14a_runtime_fails_closed_on_gpu_batch_alpha_and_bundle_changes():
    bundle = load_bundle()
    implementation = driver_v14a.implementation_identity_v14a()
    args = driver_v14a._parser_v14a().parse_args(["--v14a-dry-run"])
    for attribute, value in (
        ("n_vllm_engines", 3), ("batch_size", 56), ("alpha", 0.0003),
    ):
        changed = copy.copy(args)
        setattr(changed, attribute, value)
        with pytest.raises(ValueError, match="four-GPU recipe"):
            driver_v14a.validate_runtime_v14a(changed, bundle, implementation)
    args.expected_implementation_bundle_sha256 = "0" * 64
    with pytest.raises(ValueError, match="implementation bundle"):
        driver_v14a.validate_runtime_v14a(args, bundle, implementation)


def test_v14a_failed_attempt_is_exclusive_durable_and_closes_trainer(
    tmp_path, monkeypatch,
):
    calls = []

    class SyntheticTrainer:
        def configure_full_frame_v14a(self, panels, *, frozen_layer_plan):
            del panels, frozen_layer_plan
            calls.append("configure")
            return {"configured": True}

        def estimate_full_frame_v14a(self, seeds):
            assert seeds == anchor_v14a.PERTURBATION_SEEDS_V14A
            calls.append("estimate")
            raise RuntimeError("synthetic signed-wave failure")

        def apply_seed_coefficients(self, *_args, **_kwargs):
            pytest.fail("update path was invoked")

    monkeypatch.setattr(driver_v14a, "FROZEN_OUTPUT_DIRECTORY_V14A", tmp_path)
    monkeypatch.setattr(
        driver_v14a, "_source_provenance_v14a",
        lambda implementation: {
            "schema": "synthetic-v14a-source",
            "implementation_bundle_sha256": implementation["bundle_sha256"],
        },
    )
    monkeypatch.setattr(
        driver_v14a, "_make_trainer_v14a",
        lambda *_args, **_kwargs: SyntheticTrainer(),
    )
    monkeypatch.setattr(
        driver_v14a.base, "close_trainer",
        lambda trainer: calls.append(("close", type(trainer).__name__)),
    )
    implementation = driver_v14a.implementation_identity_v14a()
    args = driver_v14a._parser_v14a().parse_args([])
    args.output_directory = str(tmp_path)
    recipe = {"schema": "synthetic-v14a-recipe"}
    with pytest.raises(RuntimeError, match="synthetic signed-wave failure"):
        driver_v14a.run_exact_v14a(
            args, load_bundle(), anchor_v14a.load_panel_bundle_v14a(),
            implementation, recipe,
        )
    attempt = json.loads(driver_v14a._attempt_path_v14a().read_text())
    assert attempt["status"] == "failed"
    assert attempt["failure"]["type"] == "RuntimeError"
    assert attempt["failure"]["message"] == "synthetic signed-wave failure"
    assert attempt["report_exists_after_attempt"] is False
    assert attempt["model_update_applied"] is False
    assert attempt["sealed_or_nontrain_surface_opened"] is False
    assert calls == ["configure", "estimate", ("close", "SyntheticTrainer")]
    with pytest.raises(ValueError, match="fresh exclusive"):
        driver_v14a.run_exact_v14a(
            args, load_bundle(), anchor_v14a.load_panel_bundle_v14a(),
            implementation, recipe,
        )
