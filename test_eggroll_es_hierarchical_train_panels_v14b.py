#!/usr/bin/env python3

import copy
import inspect
import json
import math
import subprocess
from pathlib import Path

import pytest

import eggroll_es_hierarchical_preregistration_v14b as prereg_v14b
import run_eggroll_es_hierarchical_train_panels_v14b as driver_v14b
import train_eggroll_es_specialist_anchor_v11b as anchor_v11b
import train_eggroll_es_specialist_anchor_v14b as anchor_v14b


PLAN_SHA = prereg_v14b.LAYER_PLAN_SHA256_V14B


def load_bundle():
    spec = driver_v14b.driver_v14a.driver_v13.anchor_v11.FROZEN_STABILITY_PLANS_V11[
        PLAN_SHA
    ]
    return anchor_v14b.load_frozen_layer_plan_v14b(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=prereg_v14b.MODEL_CONFIG_SHA256_V14B,
    )


def cli(extra=None):
    spec = driver_v14b.driver_v14a.driver_v13.anchor_v11.FROZEN_STABILITY_PLANS_V11[
        PLAN_SHA
    ]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256", prereg_v14b.MODEL_CONFIG_SHA256_V14B,
        "--v14b-dry-run", *(extra or []),
    ]


def synthetic_responses():
    base = [float(index) - 15.5 for index in range(32)]

    def signed(scale):
        central = [scale * value for value in base]
        return {
            "plus": [10.0 + value for value in central],
            "minus": [10.0 - value for value in central],
        }

    return {
        "full_frame_sign_scores": signed(1.0),
        "matched56_sign_scores": {
            name: signed(1.0 + 0.05 * index)
            for index, name in enumerate(prereg_v14b.PANEL_NAMES_V14B)
        },
        "complement_sign_scores": {
            name: signed(1.4 + 0.05 * index)
            for index, name in enumerate(prereg_v14b.PANEL_NAMES_V14B[3:])
        },
        "dense_result_sha256": {
            sign: [anchor_v14b.canonical_sha256([sign, index]) for index in range(32)]
            for sign in anchor_v14b.SIGNS_V14B
        },
    }


def synthetic_integrity(*, boundary_passed=True):
    identity = {
        "pre_probe": {"same": True}, "post_probe": {"same": True},
        "exact_reference_checks": [{"passed": True} for _ in range(4)],
        "passed": True,
    }
    boundary = {"passed": boundary_passed, "iteration": 0}
    boundary["audit_sha256"] = anchor_v14b.canonical_sha256(boundary)
    hardware = {
        "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
        "population_waves": 8, "signed_waves": 16,
        "partial_waves": 0, "all_engines_generate_every_signed_wave": True,
    }
    return anchor_v14b.build_integrity_audits_v14b(
        alpha=0.0, applications=[], model_update_applied=False,
        identity_audit=identity, population_boundary_audit=boundary,
        hardware_coverage=hardware,
    )


def test_v14b_runtime_bundle_binds_481_prompts_to_310_document_groups():
    bundle = anchor_v14b.load_panel_bundle_v14b()
    anchor_v14b.validate_panel_bundle_v14b(bundle)
    full = bundle["full_frame"]
    assert len(full["questions"]) == len(full["answers"]) == 481
    assert len(full["document_groups"]) == 310
    assert sorted(
        position
        for group in full["document_groups"]
        for position in group["prompt_positions"]
    ) == list(range(481))
    assert sorted(len(group["prompt_positions"]) for group in full["document_groups"]).count(1) == 139
    assert sorted(len(group["prompt_positions"]) for group in full["document_groups"]).count(2) == 171
    assert tuple(bundle["matched56"]) == prereg_v14b.PANEL_NAMES_V14B
    assert tuple(bundle["complements"]) == prereg_v14b.PANEL_NAMES_V14B[3:]


def test_v14b_reducer_averages_within_document_before_all_aggregates(monkeypatch):
    bundle = anchor_v14b.load_panel_bundle_v14b()
    rewards = [float(index) / 100.0 for index in range(481)]
    dense = {
        "schema": "synthetic-dense",
        "examples": [{"mean_answer_token_logprob": value} for value in rewards],
    }
    monkeypatch.setattr(
        anchor_v14b.anchor_v4, "score_gold_answer_outputs_v4",
        lambda _items, _outputs: copy.deepcopy(dense),
    )
    reduced = anchor_v14b._score_full_outputs_v14b([], [], bundle)
    document_values = [
        math.fsum(rewards[position] for position in group["prompt_positions"])
        / len(group["prompt_positions"])
        for group in bundle["full_frame"]["document_groups"]
    ]
    assert reduced["full_frame"] == pytest.approx(math.fsum(document_values) / 310.0)
    for name, panel in bundle["matched56"].items():
        expected = math.fsum(
            weight * document_values[position]
            for position, weight in zip(panel["document_positions"], panel["weights"])
        ) / 310.0
        assert reduced["matched56"][name] == pytest.approx(expected)
    for name, complement in bundle["complements"].items():
        expected = math.fsum(
            document_values[position] for position in complement["document_positions"]
        ) / 254.0
        assert reduced["complements"][name] == pytest.approx(expected)
    assert set(reduced) == {
        "full_frame", "matched56", "complements", "dense_result_sha256",
    }
    assert all(not isinstance(value, list) for value in reduced.values())


def test_v14b_analysis_uses_full_frame_and_all_eight_verified_spreads():
    analysis = anchor_v14b.analyze_responses_v14b(
        synthetic_responses(), synthetic_integrity(),
    )
    assert len(analysis["standardization"]) == 8
    assert all(not value["zero_spread"] for value in analysis["standardization"].values())
    candidate = analysis["candidate_summary"]
    full, _summary = anchor_v14b.anchor_v14a._standardize(
        [float(index) - 15.5 for index in range(32)]
    )
    assert candidate["robust_aggregate"]["coefficient_sha256"] == (
        anchor_v14b.coefficient_sha256(anchor_v14b.PERTURBATION_SEEDS_V14B, full)
    )
    assert candidate["all_integrity_audits_passed"] is True
    assert analysis["promotion_gate"][
        "eligible_for_train_only_estimator_confirmation"
    ] is True
    assert analysis["promotion_gate"]["eligible_for_model_update"] is False


def test_v14b_valid_responses_cannot_claim_or_promote_failed_integrity():
    failed = synthetic_integrity(boundary_passed=False)
    assert failed["population_boundary_passed"] is False
    with pytest.raises(RuntimeError, match="integrity audits"):
        anchor_v14b.analyze_responses_v14b(synthetic_responses(), failed)


def test_v14b_response_contract_rejects_per_document_or_nonfinite_payloads():
    valid = synthetic_responses()
    anchor_v14b.validate_responses_v14b(valid)
    extra = copy.deepcopy(valid)
    extra["per_document_rewards"] = [[0.0] * 310]
    with pytest.raises(RuntimeError, match="response shape"):
        anchor_v14b.validate_responses_v14b(extra)
    bad = copy.deepcopy(valid)
    bad["full_frame_sign_scores"]["plus"][0] = float("nan")
    with pytest.raises(RuntimeError, match="response vectors"):
        anchor_v14b.validate_responses_v14b(bad)


def test_v14b_trainer_inherits_update_disabled_v13_worker_and_refuses_update():
    trainer = anchor_v14b.load_trainer(load_bundle())
    assert anchor_v11b.DualManifestResidentSignContractMixinV11B in trainer.__mro__
    assert trainer.launch_engines.__globals__["WORKER_EXTENSION"] == (
        "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
    )
    assert '"moe_backend": "triton"' in inspect.getsource(trainer.launch_engines)
    with pytest.raises(RuntimeError, match="forbids model updates"):
        anchor_v14b.PairedDistinctRowDiagnosticMixinV14B.apply_seed_coefficients(
            object(), {}, 0.1,
        )


def test_v14b_exact_four_engine_signed_wave_contract_without_gpu(monkeypatch):
    calls = {"generate": [], "rpc": [], "restores": 0}
    seed_index = {seed: index for index, seed in enumerate(anchor_v14b.PERTURBATION_SEEDS_V14B)}

    class Remote:
        def __init__(self, callback): self.callback = callback
        def remote(self, *args, **kwargs): return self.callback(*args, **kwargs)

    class Engine:
        def __init__(self, index):
            self.index = index; self.state = None
            self.collective_rpc = Remote(self.rpc); self.generate = Remote(self.run)
        def rpc(self, name, args):
            calls["rpc"].append((self.index, name, args))
            if name == "perturb_self_weights":
                seed, sigma, negate = args; assert sigma == 0.0003
                self.state = (int(seed), bool(negate)); return {"passed": True}
            if name == "verify_self_exact_reference": return {"passed": True}
            raise AssertionError(name)
        def run(self, prompts, _params, *, use_tqdm):
            assert len(prompts) == 481 and use_tqdm is False
            calls["generate"].append((self.index, self.state, len(prompts)))
            marker = {"seed": self.state[0], "negate": self.state[1]}
            return [marker] * 481

    controller = object.__new__(anchor_v14b.PairedDistinctRowDiagnosticMixinV14B)
    controller.engines = [Engine(index) for index in range(4)]
    controller.sigma = 0.0003
    controller._v14b_panel_bundle = anchor_v14b.load_panel_bundle_v14b()
    controller._resolve = lambda values: values
    controller._dense_sampling_params_v4 = lambda _seed: {"synthetic": True}
    controller._prepared_full_frame_v14b = lambda: (
        [{}] * 481, [{}] * 481, anchor_v14b.FULL_TEMPLATED_QA_SHA256_V14B,
    )
    controller._base_probe_v14b = lambda _dense, prompts: {
        "schema": "synthetic-probe", "request_count": len(prompts),
    }
    controller._restore_all_engines_exact = lambda: calls.__setitem__(
        "restores", calls["restores"] + 1,
    )
    boundary = {"passed": True, "iteration": 0}
    boundary["audit_sha256"] = anchor_v14b.canonical_sha256(boundary)
    controller._population_boundary_audit_v4 = lambda iteration: boundary

    def score(_dense, outputs, _bundle):
        marker = outputs[0]
        coordinate = float(seed_index[marker["seed"]]) - 15.5
        central = -coordinate if marker["negate"] else coordinate
        return {
            "full_frame": 10.0 + central,
            "matched56": {
                name: 10.0 + central * (1.0 + 0.05 * index)
                for index, name in enumerate(prereg_v14b.PANEL_NAMES_V14B)
            },
            "complements": {
                name: 10.0 + central * (1.4 + 0.05 * index)
                for index, name in enumerate(prereg_v14b.PANEL_NAMES_V14B[3:])
            },
            "dense_result_sha256": anchor_v14b.canonical_sha256(marker),
        }

    monkeypatch.setattr(anchor_v14b, "_score_full_outputs_v14b", score)
    artifact = controller.estimate_full_frame_v14b(anchor_v14b.PERTURBATION_SEEDS_V14B)
    assert calls["restores"] == 16
    assert len(calls["generate"]) == 64
    assert all(count == 481 for _engine, _state, count in calls["generate"])
    assert {name for _engine, name, _args in calls["rpc"]} == {
        "perturb_self_weights", "verify_self_exact_reference",
    }
    assert artifact["integrity_audits"]["hardware_contract_passed"] is True
    assert "per_document_rewards" not in json.dumps(artifact)


def test_v14b_parser_firewall_runs_before_parser(monkeypatch):
    parser = driver_v14b._parser_v14b()
    for action in parser._actions:
        surface = [*action.option_strings]
        if isinstance(action.default, (str, Path)): surface.append(str(action.default))
        assert not any(
            forbidden in value.lower()
            for value in surface for forbidden in driver_v14b.FORBIDDEN_SURFACE_TOKENS_V14B
        )
    monkeypatch.setattr(driver_v14b, "_parser_v14b", lambda: pytest.fail("parser reached"))
    with pytest.raises(ValueError, match="rejects non-train surface"):
        driver_v14b.main(["--heldout", "forbidden.jsonl"])


def test_v14b_dry_run_binds_recipe_without_constructing_trainer(monkeypatch, capsys):
    monkeypatch.setattr(
        driver_v14b, "_make_trainer_v14b",
        lambda *_args, **_kwargs: pytest.fail("dry run constructed trainer"),
    )
    payload = driver_v14b.main(cli())
    recipe = payload["recipe"]
    assert payload["gpu_launched"] is False
    assert recipe["alpha"] == 0.0 and recipe["model_update_allowed"] is False
    assert recipe["full_frame"]["prompts"] == 481
    assert recipe["generation"]["document_means_before_equal_document_aggregation"]
    assert recipe["hardware"] == {
        "engine_count": 4, "tp_per_engine": 1,
        "gpu_ids": [0, 1, 2, 3], "complete_wave_required": True,
    }
    assert recipe["moe_kernel_environment"] == {
        "moe_backend": "triton", "vllm_tuned_config_folder": None,
        "backend_selector_environment_overrides": {
            name: None for name in driver_v14b.MOE_ENVIRONMENT_OVERRIDES_V14B[1:]
        },
        "explicit_backend_bound_by_committed_inherited_engine_recipe": True,
    }
    assert "k2-launch-dry-run-v14b" in capsys.readouterr().out


def test_v14b_source_bundle_includes_prereg_sampler_evidence_and_lineage():
    files = driver_v14b.implementation_identity_v14b()["files"]
    assert set(driver_v14b.driver_v14a.IMPLEMENTATION_PATHS_V14A) <= set(files)
    assert set(driver_v14b.COMMITTED_DEPENDENCY_HASHES_V14B) <= set(files)
    assert {"trainer_v14b", "driver_v14b", "tests_v14b"} <= set(files)


def test_v14b_source_provenance_rejects_uncommitted_source(monkeypatch):
    implementation = {
        "files": {"synthetic": {
            "path": str(Path(driver_v14b.__file__).resolve()), "file_sha256": "0" * 64,
        }}, "bundle_sha256": "1" * 64,
    }
    def fake(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]: return "head\n"
        raise subprocess.CalledProcessError(128, command)
    monkeypatch.setattr(driver_v14b.subprocess, "check_output", fake)
    with pytest.raises(RuntimeError, match="requires committed implementation"):
        driver_v14b._source_provenance_v14b(implementation)


def test_v14b_runtime_fails_closed_on_gpu_batch_alpha_and_bundle_changes():
    bundle = load_bundle()
    implementation = driver_v14b.implementation_identity_v14b()
    args = driver_v14b._parser_v14b().parse_args(["--v14b-dry-run"])
    for attribute, value in (("n_vllm_engines", 3), ("batch_size", 310), ("alpha", 0.1)):
        changed = copy.copy(args); setattr(changed, attribute, value)
        with pytest.raises(ValueError, match="four-GPU recipe"):
            driver_v14b.validate_runtime_v14b(changed, bundle, implementation)
    args.expected_implementation_bundle_sha256 = "0" * 64
    with pytest.raises(ValueError, match="implementation bundle"):
        driver_v14b.validate_runtime_v14b(args, bundle, implementation)


@pytest.mark.parametrize("name", driver_v14b.MOE_ENVIRONMENT_OVERRIDES_V14B)
def test_v14b_runtime_rejects_tuned_or_backend_override_environment(
    name, monkeypatch,
):
    monkeypatch.setenv(name, "/tmp/confound" if name == "VLLM_TUNED_CONFIG_FOLDER" else "1")
    with pytest.raises(ValueError, match="environment overrides unset"):
        driver_v14b.validate_runtime_v14b(
            driver_v14b._parser_v14b().parse_args(["--v14b-dry-run"]),
            load_bundle(), driver_v14b.implementation_identity_v14b(),
        )


def test_v14b_failed_attempt_is_durable_exclusive_and_closes_trainer(tmp_path, monkeypatch):
    calls = []
    class Trainer:
        def configure_full_frame_v14b(self, panels, *, frozen_layer_plan):
            calls.append("configure"); return {"configured": True}
        def estimate_full_frame_v14b(self, seeds):
            calls.append("estimate"); raise RuntimeError("synthetic k2 failure")
        def apply_seed_coefficients(self, *_args, **_kwargs): pytest.fail("update invoked")
    monkeypatch.setattr(driver_v14b, "FROZEN_OUTPUT_DIRECTORY_V14B", tmp_path)
    monkeypatch.setattr(driver_v14b, "_source_provenance_v14b", lambda implementation: {
        "schema": "synthetic-source", "implementation_bundle_sha256": implementation["bundle_sha256"],
    })
    monkeypatch.setattr(driver_v14b, "_make_trainer_v14b", lambda *_args: Trainer())
    monkeypatch.setattr(driver_v14b.base, "close_trainer", lambda trainer: calls.append("close"))
    implementation = driver_v14b.implementation_identity_v14b()
    args = driver_v14b._parser_v14b().parse_args([]); args.output_directory = str(tmp_path)
    with pytest.raises(RuntimeError, match="synthetic k2 failure"):
        driver_v14b.run_exact_v14b(
            args, load_bundle(), anchor_v14b.load_panel_bundle_v14b(),
            implementation, {"schema": "synthetic-recipe"},
        )
    attempt = json.loads(driver_v14b._attempt_path_v14b().read_text())
    assert attempt["status"] == "failed"
    assert attempt["failure"]["message"] == "synthetic k2 failure"
    assert attempt["report_exists_after_attempt"] is False
    assert attempt["model_update_applied"] is False
    assert calls == ["configure", "estimate", "close"]
    with pytest.raises(ValueError, match="fresh exclusive"):
        driver_v14b.run_exact_v14b(
            args, load_bundle(), anchor_v14b.load_panel_bundle_v14b(),
            implementation, {"schema": "synthetic-recipe"},
        )
