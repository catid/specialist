import contextlib
import copy
import json
from pathlib import Path

import pytest

import report_eggroll_es_direction_stability_v8 as reporter_v8
import run_eggroll_es_anchor_stability_v8 as stability_v8
import train_eggroll_es_specialist_anchor_v8 as anchor_v8


ROOT = Path(__file__).resolve().parent
PLAN_SHA = stability_v8.MIDDLE_LATE_PLAN_SHA256_V8


def load_bundle():
    spec = anchor_v8.FROZEN_STABILITY_PLANS_V8[PLAN_SHA]
    return anchor_v8.load_frozen_layer_plan_v8(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=anchor_v8.MODEL_CONFIG_SHA256_V8,
    )


def cli(seed=43, population=32, target="0", basis=20260714):
    spec = anchor_v8.FROZEN_STABILITY_PLANS_V8[PLAN_SHA]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256", anchor_v8.MODEL_CONFIG_SHA256_V8,
        "--v8-stage", "stability",
        "--v8-v7-family-report", str(stability_v8.V7_REPORT_PATH_V8),
        "--v8-perturbation-basis-seed", str(basis),
        "--population-size", str(population), "--batch-size", "64",
        "--seed", str(seed), "--target-alphas", target,
        "--experiment-name", (
            "snapshot794_layer_v8_middle_late_stability_"
            f"data{seed}_basis20260714"
        ),
        "--v8-dry-run",
    ]


def test_v8_plan_and_perturbation_basis_are_exact():
    assert list(anchor_v8.FROZEN_STABILITY_PLANS_V8) == [PLAN_SHA]
    assert anchor_v8.validate_frozen_layer_plan_bundle_v8(
        load_bundle()
    ) == {
        "schema": "eggroll-es-split-seed-pop32-plan-v8",
        "plan": "middle_late", "layers": [20, 21, 22, 23],
        "source_unit_count": 35, "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
        "selected_byte_count": 285_999_104,
    }
    assert len(stability_v8.PERTURBATION_SEEDS_V8) == 32
    assert len(set(stability_v8.PERTURBATION_SEEDS_V8)) == 32
    assert stability_v8.PERTURBATION_BASIS_SHA256_V8 == (
        "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11"
    )


def test_v8_cli_dry_run_and_fail_closed_recipe(capsys):
    result = stability_v8.main(cli())
    assert result["schema"] == "eggroll-es-split-seed-pop32-dry-run-v8"
    assert result["data_bootstrap_seed"] == 43
    assert result["population_size"] == 32
    assert result["targets"] == [0.0]
    assert result["perturbation_basis_sha256"] == (
        stability_v8.PERTURBATION_BASIS_SHA256_V8
    )
    assert result["recipe_sha256"] == stability_v8.EXPECTED_RECIPE_SHA256_V8[43]
    assert "split-seed-pop32-dry-run-v8" in capsys.readouterr().out
    for args, message in (
        (cli(population=16), "runtime recipe changed"),
        (cli(basis=43), "basis seed changed"),
        (cli(target="0,0.000001"), "exactly alpha zero"),
    ):
        bundle, remaining = anchor_v8.parse_frozen_layer_plan_cli_v8(args)
        with pytest.raises(ValueError, match=message):
            stability_v8.validate_frozen_execution_cli_v8(remaining, bundle)


def test_v8_execute_replaces_only_inherited_population_seeds(monkeypatch):
    stability_v8._ACTIVE_EXECUTION_V8 = {"data_bootstrap_seed": 43}
    inherited = stability_v8.np.random.default_rng(seed=43).integers(
        0, 2**30, size=32, dtype=stability_v8.np.int64,
    ).tolist()
    captured = {}

    class StopAfterCapture(Exception):
        pass

    def capture(*args, **kwargs):
        captured["seeds"] = kwargs["seeds"]
        raise StopAfterCapture

    monkeypatch.setattr(stability_v8.driver_v4, "execute_line_search", capture)
    monkeypatch.setattr(
        stability_v8.driver_v6, "scoped_legacy_audit_v6",
        contextlib.nullcontext,
    )
    with pytest.raises(StopAfterCapture):
        stability_v8.execute_line_search(object(), seeds=inherited)
    assert captured["seeds"] == stability_v8.PERTURBATION_SEEDS_V8
    with pytest.raises(RuntimeError, match="inherited driver population"):
        stability_v8.execute_line_search(object(), seeds=[1] * 32)


def test_v8_report_contract_is_same_basis_and_benchmark_free(monkeypatch):
    fixtures = {}
    for seed in (43, 44):
        path = f"/middle-late-data{seed}.json"
        fixtures[path] = {
            "arm": "middle_late", "data_bootstrap_seed": seed,
            "perturbation_basis_sha256": (
                stability_v8.PERTURBATION_BASIS_SHA256_V8
            ),
            "journal": path, "journal_file_sha256": f"file-{seed}",
            "content_sha256": f"content-{seed}",
            "coefficient_sha256": f"coeff-{seed}",
            "robust_plan_sha256": f"robust-{seed}",
            "geometry": {
                "raw_domain_anchor_cosine": 0.1,
                "projected_anchor_cosine": 0.8,
                "projection_lambda": 1.0, "update_norm_ratio": 1.0,
                "projected_vs_raw_domain_cosine": 0.7,
                "domain_mean": -1.0, "domain_std": 0.1,
                "anchor_mean": -0.1, "anchor_std": 0.1,
            },
            "coefficients": [float(index + seed) for index in range(32)],
        }
    monkeypatch.setattr(
        reporter_v8, "_load_run", lambda path: copy.deepcopy(fixtures[path]),
    )
    report = reporter_v8.build_report(list(fixtures))
    assert report["passed"] is True
    assert report["preregistered_threshold"] == 0.5
    assert report["coverage"]["population_size"] == 32
    serialized = repr(report).casefold()
    assert "mean_reward" not in serialized
    assert "ood_prose" not in serialized
    with pytest.raises(ValueError, match="data seeds43/44"):
        reporter_v8.build_report(list(fixtures)[:1])


def test_v8_completed_outer_contract_rejects_basis_tamper(monkeypatch):
    source = ROOT / (
        "experiments/eggroll_es_hpo/runs/"
        "snapshot794_layer_v7_middle_late_stability_seed43/"
        "alpha_line_search.json"
    )
    journal = json.loads(source.read_text())
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v8"
    journal["seeds"] = list(stability_v8.PERTURBATION_SEEDS_V8)
    for key in (
        "direction_stability_family_v7", "stage_v7",
        "target_alpha_zero_only_v7", "benchmark_selection_forbidden_v7",
        "cross_seed_coefficient_cosine_threshold_v7",
        "requires_clean_v6_family_v7",
    ):
        journal["policy"].pop(key)
    journal["policy"].update({
        "split_seed_stability_family_v8": "middle_late_pop32_same_basis",
        "stage_v8": "stability", "target_alpha_zero_only_v8": True,
        "benchmark_selection_forbidden_v8": True,
        "same_perturbation_basis_required_v8": True,
        "cross_data_seed_coefficient_cosine_threshold_v8": 0.5,
        "requires_complete_v7_family_v8": True,
    })
    evidence = {"schema": "test-v7-evidence"}
    monkeypatch.setattr(
        stability_v8, "_v7_family_evidence_v8", lambda path: evidence,
    )
    snapshot = journal["snapshot"]
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v8"
    snapshot.pop("direction_stability_v7")
    implementation = {
        key: anchor_v8.file_sha256(path)
        for key, path in stability_v8.V8_IMPLEMENTATION_PATHS.items()
    }
    snapshot["implementation"] = implementation
    recipe = stability_v8.frozen_recipe_v8(43, evidence)
    monkeypatch.setitem(
        stability_v8.EXPECTED_RECIPE_SHA256_V8, 43,
        stability_v8.driver_v1.canonical_sha256(recipe),
    )
    snapshot["recipe"] = {
        key: recipe[key] for key in (
            "model_name", "checkpoint", "sigma", "population_size",
            "batch_size", "mini_batch_size", "max_tokens", "seed",
            "min_anchor_cosine", "anchor_items_per_step", "target_alphas",
        )
    }
    spec = anchor_v8.FROZEN_STABILITY_PLANS_V8[PLAN_SHA]
    snapshot["split_seed_stability_v8"] = {
        "schema": "eggroll-es-split-seed-pop32-snapshot-v8",
        "family": "middle_late_same_basis_cross_data_seed",
        "stage": "stability", "arm": "middle_late",
        "layers": [20, 21, 22, 23],
        "data_bootstrap_seed_pair": [43, 44], "data_bootstrap_seed": 43,
        "perturbation_basis_seed": 20260714,
        "perturbation_seed_count": 32,
        "perturbation_seeds": list(stability_v8.PERTURBATION_SEEDS_V8),
        "perturbation_basis_sha256": stability_v8.PERTURBATION_BASIS_SHA256_V8,
        "target_alphas": [0.0], "benchmark_treatment_applied": False,
        "selection_surface": "same_basis_coefficients_only",
        "coefficient_cosine_threshold": 0.5, "recipe": recipe,
        "plan_sha256": PLAN_SHA, "plan_file_sha256": spec["file_sha256"],
        "source_unit_count": 35, "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
        "selected_byte_count": 285_999_104,
        "implementation_bundle_sha256": (
            stability_v8.driver_v1.canonical_sha256(implementation)
        ),
    }
    journal["trainer_configuration"]["population_size"] = 32
    journal.pop("content_sha256_before_self_field")
    journal["content_sha256_before_self_field"] = (
        stability_v8.driver_v1.canonical_sha256(journal)
    )
    monkeypatch.setattr(
        stability_v8, "_validate_inherited_zero_target_v8",
        lambda value: {
            "data_bootstrap_seed": 43, "state_count": 1,
            "coefficient_sha256": value["coefficient_plan"][
                "coefficient_sha256"
            ],
            "robust_plan_sha256": value["coefficient_plan"][
                "robust_plan_binding_v5"
            ]["robust_plan_sha256"],
            "distributed_update_v4": {},
        },
    )
    assert stability_v8.validate_completed_journal_v8(journal)[
        "data_bootstrap_seed"
    ] == 43
    tampered = copy.deepcopy(journal)
    tampered["seeds"][0] += 1
    tampered["content_sha256_before_self_field"] = (
        stability_v8.driver_v1.canonical_sha256({
            key: value for key, value in tampered.items()
            if key != "content_sha256_before_self_field"
        })
    )
    with pytest.raises(RuntimeError, match="changed basis/target"):
        stability_v8.validate_completed_journal_v8(tampered)
