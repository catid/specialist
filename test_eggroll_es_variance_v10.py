import copy
import ast
import inspect
import json
from pathlib import Path

import pytest

import eggroll_es_worker_v10 as worker_v10
import report_eggroll_es_variance_v10 as reporter_v10
import run_eggroll_es_anchor_variance_v10 as variance_v10
import train_eggroll_es_specialist_anchor_v10 as anchor_v10


PLAN_SHA = variance_v10.MIDDLE_LATE_PLAN_SHA256_V10


def load_bundle():
    spec = anchor_v10.FROZEN_STABILITY_PLANS_V10[PLAN_SHA]
    return anchor_v10.load_frozen_layer_plan_v10(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=anchor_v10.MODEL_CONFIG_SHA256_V10,
    )


def cli(batch=128, target="0", basis=20260714):
    spec = anchor_v10.FROZEN_STABILITY_PLANS_V10[PLAN_SHA]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256", anchor_v10.MODEL_CONFIG_SHA256_V10,
        "--v10-stage", "variance", "--v10-v9-report",
        str(variance_v10.V9_REPORT_PATH_V10),
        "--v10-perturbation-basis-seed", str(basis),
        "--population-size", "32", "--batch-size", str(batch),
        "--mini-batch-size", "64", "--seed", "43",
        "--target-alphas", target,
        "--experiment-name", variance_v10.EXPERIMENT_NAME_V10,
        "--v10-dry-run",
    ]


def test_v10_worker_basis_and_crossed_manifests_are_frozen():
    assert list(worker_v10.FROZEN_LAYER_PLANS_V10) == [PLAN_SHA]
    assert len(anchor_v10.PERTURBATION_SEEDS_V10) == 32
    assert len(set(anchor_v10.PERTURBATION_SEEDS_V10)) == 32
    assert anchor_v10.DOMAIN_MANIFESTS_V10 == {
        "D43": {
            "seed": 43, "rows": 64,
            "sha256": "b864cfcc4ebcd987d8091f1067f631366c128d63d09fb7160a09561d10063a0f",
        },
        "D44": {
            "seed": 44, "rows": 64,
            "sha256": "3574ff126f727a262957f34ab83fbefce6754ae9e4be790f810f42656e692bc2",
        },
    }
    signature = inspect.signature(
        worker_v10.AntitheticCrossedAuditWorkerExtensionV10
        .perturb_self_weights
    )
    assert "negate" in signature.parameters


def test_v10_crossed_loader_reproduces_both_manifests():
    train = variance_v10.driver_v1.load_from_disk(
        str(variance_v10.FROZEN_TRAIN_DATASET_V10)
    )["train"]
    loader = variance_v10._crossed_train_loader_v10(train, 128, 43)
    questions, answers = next(iter(loader))
    assert len(questions) == len(answers) == 128
    assert variance_v10.driver_v1.canonical_sha256({
        "questions": questions, "answers": answers,
    }) == anchor_v10.COMBINED_DOMAIN_MANIFEST_SHA256_V10


def test_v10_dry_run_and_fail_closed_cli(monkeypatch, capsys):
    evidence = {"schema": "test-v9", "binding_sha256": "v9-pass"}
    monkeypatch.setattr(
        variance_v10, "_v9_determinism_evidence_v10",
        lambda path: evidence,
    )
    recipe = variance_v10.frozen_recipe_v10(evidence)
    monkeypatch.setattr(
        variance_v10, "EXPECTED_RECIPE_SHA256_V10",
        variance_v10.driver_v1.canonical_sha256(recipe),
    )
    result = variance_v10.main(cli())
    assert result["schema"] == "eggroll-es-antithetic-crossed-dry-run-v10"
    assert result["base_direction_count"] == 32
    assert result["unique_signed_direction_count"] == 64
    assert result["actual_perturb_restore_cycle_count"] == 128
    assert result["domain_signed_score_count"] == 128
    assert result["anchor_signed_response_count"] == 128
    assert result["anchor_generation_seeds"] == [43, 44]
    assert result["targets"] == [0.0]
    assert "antithetic-crossed-dry-run-v10" in capsys.readouterr().out
    for args, message in (
        (cli(batch=64), "combined-batch128"),
        (cli(target="0,0.1"), "exactly alpha zero"),
        (cli(basis=43), "basis seed changed"),
    ):
        bundle, remaining = anchor_v10.parse_frozen_layer_plan_cli_v10(args)
        with pytest.raises(ValueError, match=message):
            variance_v10.validate_frozen_execution_cli_v10(remaining, bundle)


def _minimal_robust_result(seeds, offset):
    return {
        "robust_scores": [
            {"seed": seed, "score": float(index + offset)}
            for index, seed in enumerate(seeds)
        ],
        "content_sha256_before_self_field": f"result-{offset}",
    }


def test_v10_antithetic_cross_numeric_replay_and_tamper():
    seeds = list(anchor_v10.PERTURBATION_SEEDS_V10)
    plan = {
        "seeds": seeds,
        "document_lcb_anchor_v5": _minimal_robust_result(seeds, 1),
    }
    domain = {}
    for label, scale in (("D43", 1.0), ("D44", 0.8)):
        domain[label] = {
            "plus": [scale * (index + 1) for index in range(32)],
            "minus": [-scale * (index + 1) for index in range(32)],
        }
    captures = {
        "domain_sign_scores": domain,
        "anchor_reference_identities": {
            "A43": {"identity": "43"}, "A44": {"identity": "44"},
        },
        "anchor_results": {
            "A43_minus": _minimal_robust_result(seeds, -1),
            "A44_plus": _minimal_robust_result(seeds, 2),
            "A44_minus": _minimal_robust_result(seeds, -2),
        },
    }
    plan["antithetic_cross_v10"] = anchor_v10._build_cross_artifact_v10(
        plan, captures,
    )
    audit = anchor_v10.validate_antithetic_cross_v10(plan)
    assert set(audit["cell_coefficient_sha256"]) == {
        "D43xA43", "D43xA44", "D44xA43", "D44xA44",
    }
    assert plan["antithetic_cross_v10"]["central_domain_scores"]["D43"][0] == 1.0
    tampered = copy.deepcopy(plan)
    tampered["antithetic_cross_v10"]["domain_sign_scores"]["D43"]["plus"][0] += 1
    with pytest.raises(RuntimeError, match="identity changed"):
        anchor_v10.validate_antithetic_cross_v10(tampered)

    incomplete = copy.deepcopy(plan)
    del incomplete["antithetic_cross_v10"]["domain_sign_scores"]["D44"]["minus"]
    body = incomplete["antithetic_cross_v10"]
    body["content_sha256_before_self_field"] = anchor_v10.canonical_sha256({
        key: value for key, value in body.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="sign map changed"):
        anchor_v10.validate_antithetic_cross_v10(incomplete)

    leaked = copy.deepcopy(plan)
    leaked_cell = leaked["antithetic_cross_v10"]["cells"]["D43xA44"]
    leaked_cell["anchor_generation_seed"] = 43
    body = leaked["antithetic_cross_v10"]
    body["content_sha256_before_self_field"] = anchor_v10.canonical_sha256({
        key: value for key, value in body.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="cell/reference binding changed"):
        anchor_v10.validate_antithetic_cross_v10(leaked)


def test_v10_full_wave_and_restore_finally_are_launch_critical_contracts():
    seeds = anchor_v10.PERTURBATION_SEEDS_V10[:4]
    assert anchor_v10.validate_full_engine_wave_v10(seeds, 4) == seeds
    with pytest.raises(ValueError, match="partial.*idle a GPU"):
        anchor_v10.validate_full_engine_wave_v10(seeds[:3], 4)
    with pytest.raises(ValueError, match="exactly four engines"):
        anchor_v10.validate_full_engine_wave_v10(seeds, 3)
    with pytest.raises(ValueError, match="duplicate seeds"):
        anchor_v10.validate_full_engine_wave_v10([seeds[0]] * 4, 4)

    source = inspect.getsource(
        anchor_v10.AntitheticCrossedContractMixinV10
        ._evaluate_population_with_anchor
    )
    tree = ast.parse(inspect.cleandoc(source))
    finally_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    assert len(finally_nodes) == 1
    final_calls = [
        node.func.attr for node in ast.walk(finally_nodes[0].finalbody[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert final_calls == ["_restore_all_engines_exact"]
    assert "for sign, negate in ((\"plus\", False), (\"minus\", True))" in source
    assert "validate_full_engine_wave_v10(" in source


def test_v10_report_thresholds_are_predeclared(monkeypatch, tmp_path):
    coefficients = [float(index - 16) for index in range(32)]
    cross = {
        "cells": {
            name: {"coefficients": list(coefficients)}
            for name in ("D43xA43", "D43xA44", "D44xA43", "D44xA44")
        },
        "central_domain_scores": {
            "D43": list(coefficients), "D44": list(coefficients),
        },
        "central_anchor_scores": {
            "A43": list(coefficients), "A44": list(coefficients),
        },
    }
    path = tmp_path / "journal.json"
    path.write_text(json.dumps({
        "coefficient_plan": {"antithetic_cross_v10": cross},
    }))
    monkeypatch.setattr(
        variance_v10, "validate_completed_journal_v10",
        lambda journal: {
            "content_sha256": "journal-content",
            "cross": {
                "content_sha256": "cross-content",
                "cell_coefficient_sha256": {name: name for name in cross["cells"]},
            },
        },
    )
    report = reporter_v10.build_report(path)
    assert report["passed"] is True
    assert report["minimum_pairwise_coefficient_cosine"] == pytest.approx(1.0)
    assert report["median_pairwise_coefficient_cosine"] == pytest.approx(1.0)
    assert report["thresholds"] == {
        "minimum_pairwise_coefficient_cosine": 0.5,
        "median_pairwise_coefficient_cosine": 0.7,
    }
    assert "mean_reward" not in repr(report).casefold()
