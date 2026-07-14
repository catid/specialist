import ast
import copy
import inspect
import json

import pytest

import eggroll_es_worker_v11 as worker_v11
import run_eggroll_es_anchor_equivalence_v11 as equivalence_v11
import train_eggroll_es_specialist_anchor_v11 as anchor_v11


PLAN_SHA = equivalence_v11.MIDDLE_LATE_PLAN_SHA256_V11


def load_bundle():
    spec = anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return anchor_v11.load_frozen_layer_plan_v11(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=anchor_v11.MODEL_CONFIG_SHA256_V11,
    )


def cli(batch=128, target="0", basis=20260714):
    spec = anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return [
        "--layer-plan-json", str(spec["path"]),
        "--expected-layer-plan-file-sha256", spec["file_sha256"],
        "--expected-layer-plan-sha256", PLAN_SHA,
        "--expected-model-config-sha256", anchor_v11.MODEL_CONFIG_SHA256_V11,
        "--v11-stage", "equivalence", "--v11-v10-report",
        str(equivalence_v11.V10_REPORT_PATH_V11),
        "--v11-perturbation-basis-seed", str(basis),
        "--population-size", "32", "--batch-size", str(batch),
        "--mini-batch-size", "64", "--seed", "43",
        "--target-alphas", target,
        "--experiment-name", equivalence_v11.EXPERIMENT_NAME_V11,
        "--v11-dry-run",
    ]


def _minimal_robust_result(seeds, offset):
    return {
        "robust_scores": [
            {"seed": seed, "score": float(index + offset)}
            for index, seed in enumerate(seeds)
        ],
        "content_sha256_before_self_field": f"result-{offset}",
    }


def make_plans():
    seeds = list(anchor_v11.PERTURBATION_SEEDS_V11)
    v10_plan = {
        "seeds": seeds,
        "coefficients": [float(index) for index in range(32)],
        "coefficient_sha256": "legacy-coefficients",
        "domain_scores": [float(index + 1) for index in range(32)],
        "anchor_scores": [float(index - 1) for index in range(32)],
        "document_lcb_anchor_v5": _minimal_robust_result(seeds, 1),
        "robust_plan_binding_v5": {"binding": "v5"},
    }
    captures = {
        "domain_sign_scores": {
            label: {
                "plus": [scale * (index + 1) for index in range(32)],
                "minus": [-scale * (index + 1) for index in range(32)],
            }
            for label, scale in (("D43", 1.0), ("D44", 0.8))
        },
        "anchor_reference_identities": {
            "A43": {"identity": "43"}, "A44": {"identity": "44"},
        },
        "anchor_results": {
            "A43_minus": _minimal_robust_result(seeds, -1),
            "A44_plus": _minimal_robust_result(seeds, 2),
            "A44_minus": _minimal_robust_result(seeds, -2),
        },
    }
    v10_plan["antithetic_cross_v10"] = (
        anchor_v11.anchor_v10._build_cross_artifact_v10(v10_plan, captures)
    )
    v11_plan = copy.deepcopy(v10_plan)
    v11_plan.pop("antithetic_cross_v10")
    v11_plan["resident_sign_cross_v11"] = (
        anchor_v11._build_resident_artifact_v11(v11_plan, captures)
    )
    return v10_plan, v11_plan


def test_v11_frozen_worker_mro_and_counts():
    assert list(worker_v11.FROZEN_LAYER_PLANS_V11) == [PLAN_SHA]
    trainer = anchor_v11.load_trainer(load_bundle())
    assert trainer._evaluate_population_with_anchor.__module__ == anchor_v11.__name__
    assert trainer.estimate_step_coefficients.__module__ == anchor_v11.__name__
    assert anchor_v11.ACTUAL_PERTURB_RESTORE_CYCLE_COUNT_V11 == 64
    assert anchor_v11.ALL_ENGINE_SIGN_RESIDENCY_COUNT_V11 == 16
    assert anchor_v11.DOMAIN_SIGNED_SCORE_COUNT_V11 == 128
    assert anchor_v11.ANCHOR_SIGNED_RESPONSE_COUNT_V11 == 128


def test_v11_resident_artifact_numeric_replay_and_exact_v10_gate():
    v10_plan, v11_plan = make_plans()
    audit = anchor_v11.validate_resident_cross_v11(v11_plan)
    assert len(audit["cell_coefficient_sha256"]) == 4
    binding = anchor_v11.compare_exact_v10_v11(v10_plan, v11_plan)
    assert binding["all_exact"] is True
    tampered = copy.deepcopy(v11_plan)
    cross = tampered["resident_sign_cross_v11"]
    cross["domain_sign_scores"]["D44"]["minus"][0] += 1
    cross["content_sha256_before_self_field"] = anchor_v11.canonical_sha256({
        key: value for key, value in cross.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(RuntimeError, match="numeric replay changed"):
        anchor_v11.validate_resident_cross_v11(tampered)
    with pytest.raises(RuntimeError, match="differs from v10"):
        anchor_v11.compare_exact_v10_v11(v10_plan, tampered)


def test_v11_restore_order_and_no_dispatch_cache_contract_are_static_gates():
    source = inspect.getsource(
        anchor_v11.ResidentSignContractMixinV11
        ._evaluate_population_with_anchor
    )
    tree = ast.parse(inspect.cleandoc(source))
    tries = [node for node in ast.walk(tree) if isinstance(node, ast.Try)]
    assert len(tries) == 1
    finally_calls = [
        node.func.attr for node in ast.walk(tries[0].finalbody[0])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]
    assert finally_calls == ["_restore_all_engines_exact"]
    assert source.index('batches["D43"]') < source.index('batches["A43"]')
    assert source.index('batches["A43"]') < source.index('batches["A44"]')
    assert source.index('batches["A44"]') < source.index('batches["D44"]')
    assert source.index("finally:") < source.index("_score_dense_v11(")
    cache_branch = source[source.index("if call_index == 1:"):source.index(
        "if len(anchor_items) != 128:"
    )]
    assert "collective_rpc" not in cache_branch
    assert ".generate" not in cache_branch
    assert "_v11_d44_cache_consumed = True" in cache_branch


def test_v11_dry_run_and_fail_closed_cli(monkeypatch, capsys):
    evidence = {"schema": "test-v10", "binding_sha256": "v10-pass"}
    monkeypatch.setattr(
        equivalence_v11, "_v10_equivalence_evidence_v11",
        lambda path: evidence,
    )
    monkeypatch.setattr(
        equivalence_v11.driver_v10, "_v9_determinism_evidence_v10",
        lambda path: {"schema": "test-v9", "binding_sha256": "v9-pass"},
    )
    recipe = equivalence_v11.frozen_recipe_v11(evidence)
    monkeypatch.setattr(
        equivalence_v11, "EXPECTED_RECIPE_SHA256_V11",
        equivalence_v11.driver_v1.canonical_sha256(recipe),
    )
    result = equivalence_v11.main(cli())
    assert result["schema"] == "eggroll-es-resident-sign-dry-run-v11"
    assert result["actual_perturb_restore_cycle_count"] == 64
    assert result["all_engine_sign_residency_count"] == 16
    assert result["domain_signed_score_count"] == 128
    assert result["anchor_signed_response_count"] == 128
    assert result["resident_generation_order"] == ["D43", "A43", "A44", "D44"]
    assert "resident-sign-dry-run-v11" in capsys.readouterr().out
    for args, message in (
        (cli(batch=64), "combined-batch128"),
        (cli(target="0,0.1"), "exactly alpha zero"),
        (cli(basis=43), "basis seed changed"),
    ):
        bundle, remaining = anchor_v11.parse_frozen_layer_plan_cli_v11(args)
        with pytest.raises(ValueError, match=message):
            equivalence_v11.validate_frozen_execution_cli_v11(
                remaining, bundle,
            )


def test_v11_v10_evidence_aggregates_are_frozen():
    report = json.loads(equivalence_v11.V10_REPORT_PATH_V11.read_text())
    assert report["passed"] is True
    assert report["content_sha256_before_self_field"] == (
        equivalence_v11.V10_REPORT_CONTENT_SHA256_V11
    )
    assert report["journal_file_sha256"] == (
        equivalence_v11.V10_JOURNAL_FILE_SHA256_V11
    )
    assert report["journal_content_sha256"] == (
        equivalence_v11.V10_JOURNAL_CONTENT_SHA256_V11
    )
    assert report["cross_artifact_content_sha256"] == (
        anchor_v11.V10_EQUIVALENCE_TARGET_V11[
            "cross_artifact_content_sha256"
        ]
    )

