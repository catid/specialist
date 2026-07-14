import copy
import json

import pytest

import report_eggroll_es_replay_v9b as reporter_v9b
import run_eggroll_es_anchor_replay_v9b as replay_v9b
import train_eggroll_es_specialist_anchor_v8 as anchor_v8


PLAN_SHA = replay_v9b.driver_v8.MIDDLE_LATE_PLAN_SHA256_V8


def fake_evidence():
    return {
        "schema": "eggroll-es-v8-data44-reference-binding-v9b",
        "failed_v8_family_binding_sha256": "failed-v8",
        "reference": {"data_seed": 44, "journal": "/original44.json"},
    }


def load_bundle():
    spec = anchor_v8.FROZEN_STABILITY_PLANS_V8[PLAN_SHA]
    return anchor_v8.load_frozen_layer_plan_v8(
        spec["path"], expected_file_sha256=spec["file_sha256"],
        expected_plan_sha256=PLAN_SHA,
        expected_model_config_sha256=anchor_v8.MODEL_CONFIG_SHA256_V8,
    )


def cli(seed=44, population=32, target="0", name=None):
    return [
        "--v9b-stage", "deterministic_replay",
        "--v9b-v8-failed-report",
        str(replay_v9b.replay_v9.V8_FAILED_REPORT_PATH_V9),
        "--v9b-perturbation-basis-seed", "20260714",
        "--population-size", str(population), "--batch-size", "64",
        "--seed", str(seed), "--target-alphas", target,
        "--experiment-name", name or replay_v9b.EXPERIMENT_NAME_V9B,
    ]


def test_v9b_exact_seed44_reference_binding(monkeypatch):
    family = {
        "binding_sha256": "failed-v8", "formal_v8_result": {"passed": False},
        "runs": [{
            "data_seed": 44,
            "journal": str(
                replay_v9b.replay_v9.ORIGINAL_V8_JOURNALS_V9[44]["path"]
            ),
            **{
                key: replay_v9b.replay_v9.ORIGINAL_V8_JOURNALS_V9[44][key]
                for key in (
                    "journal_file_sha256", "content_sha256",
                    "coefficient_sha256", "robust_plan_sha256",
                )
                if key in replay_v9b.replay_v9.ORIGINAL_V8_JOURNALS_V9[44]
            },
        }],
    }
    expected = replay_v9b.replay_v9.ORIGINAL_V8_JOURNALS_V9[44]
    family["runs"][0]["journal_file_sha256"] = expected["file_sha256"]
    monkeypatch.setattr(
        replay_v9b.replay_v9, "_v8_failed_evidence_v9",
        lambda path: family,
    )
    evidence = replay_v9b._failed_v8_seed44_evidence_v9b("ignored")
    assert evidence["reference"]["data_seed"] == 44
    assert evidence["reference"]["coefficient_sha256"] == (
        expected["coefficient_sha256"]
    )


def test_v9b_cli_is_seed44_alpha0_and_distinct_name(monkeypatch):
    evidence = fake_evidence()
    monkeypatch.setattr(
        replay_v9b, "_failed_v8_seed44_evidence_v9b",
        lambda path: evidence,
    )
    monkeypatch.setattr(
        replay_v9b.driver_v8, "validate_frozen_execution_cli_v8",
        lambda argv, bundle: ({"data_bootstrap_seed": 44}, []),
    )
    recipe = replay_v9b.wrapper_recipe_v9b(evidence)
    monkeypatch.setattr(
        replay_v9b, "EXPECTED_WRAPPER_RECIPE_SHA256_V9B",
        replay_v9b.driver_v1.canonical_sha256(recipe),
    )
    execution, remaining = replay_v9b.validate_frozen_execution_cli_v9b(
        cli(), load_bundle(),
    )
    assert execution["data_seed"] == 44
    assert execution["wrapper_recipe"] == recipe
    assert replay_v9b.EXPERIMENT_NAME_V9B in remaining
    for args, message in (
        (cli(seed=43), "seed must be 44"),
        (cli(population=16), "population must be 32"),
        (cli(target="0,0.1"), "exactly alpha zero"),
        (cli(name="overwrite"), "experiment name changed"),
    ):
        with pytest.raises(ValueError, match=message):
            replay_v9b.validate_frozen_execution_cli_v9b(args, load_bundle())


def test_v9b_completed_envelope_and_tamper(monkeypatch):
    journal = json.loads(
        replay_v9b.replay_v9.ORIGINAL_V8_JOURNALS_V9[44]["path"].read_text()
    )
    evidence = fake_evidence()
    recipe = replay_v9b.wrapper_recipe_v9b(evidence)
    monkeypatch.setattr(
        replay_v9b, "_failed_v8_seed44_evidence_v9b",
        lambda path: evidence,
    )
    monkeypatch.setattr(
        replay_v9b, "EXPECTED_WRAPPER_RECIPE_SHA256_V9B",
        replay_v9b.driver_v1.canonical_sha256(recipe),
    )
    monkeypatch.setattr(
        replay_v9b.driver_v8, "validate_completed_journal_v8",
        lambda value: {
            "data_bootstrap_seed": 44, "coefficient_sha256": "coefficient",
        },
    )
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v9b"
    journal["snapshot"]["schema"] = (
        "eggroll-es-anchor-line-search-snapshot-v9b"
    )
    implementation = {
        key: anchor_v8.file_sha256(path)
        for key, path in replay_v9b.V9B_IMPLEMENTATION_PATHS.items()
    }
    journal["snapshot"]["implementation"] = implementation
    journal["snapshot"]["deterministic_replay_v9b"] = {
        "schema": "eggroll-es-v8-data44-replay-snapshot-v9b",
        "family": "exact_v8_data44_control",
        "stage": "deterministic_replay", "arm": "middle_late",
        "data_seed": 44,
        "actual_experiment_name": replay_v9b.EXPERIMENT_NAME_V9B,
        "target_alphas": [0.0], "benchmark_treatment_applied": False,
        "selection_surface": "coefficient_and_raw_score_identity_only",
        "replay_cosine_threshold": 0.99,
        "reference_evidence": evidence, "wrapper_recipe": recipe,
    }
    journal["policy"].update(replay_v9b.V9B_POLICY)
    journal.pop("content_sha256_before_self_field")
    journal["content_sha256_before_self_field"] = (
        replay_v9b.driver_v1.canonical_sha256(journal)
    )
    assert replay_v9b.validate_completed_journal_v9b(journal)[
        "data_seed"
    ] == 44
    tampered = copy.deepcopy(journal)
    tampered["seeds"][0] += 1
    tampered["content_sha256_before_self_field"] = (
        replay_v9b.driver_v1.canonical_sha256({
            key: value for key, value in tampered.items()
            if key != "content_sha256_before_self_field"
        })
    )
    with pytest.raises(RuntimeError, match="changed basis/target"):
        replay_v9b.validate_completed_journal_v9b(tampered)


def test_v9b_report_requires_exact_identity(monkeypatch):
    coefficients = [float(index - 16) for index in range(32)]
    domain = [float(index * index + 1) for index in range(32)]
    anchor = [float(64 - index) for index in range(32)]
    fixtures = {}
    for role in ("original", "replay"):
        fixtures[role] = {
            "role": role, "data_seed": 44, "journal": f"/{role}.json",
            "journal_file_sha256": f"file-{role}",
            "content_sha256": f"content-{role}",
            "coefficient_sha256": "same", "robust_plan_sha256": f"r-{role}",
            "domain_scores_sha256": reporter_v9b.reporter_v9.vector_sha256(domain),
            "anchor_scores_sha256": reporter_v9b.reporter_v9.vector_sha256(anchor),
            "perturbation_basis_sha256": (
                replay_v9b.driver_v8.PERTURBATION_BASIS_SHA256_V8
            ),
            "coefficients": coefficients, "domain_scores": domain,
            "anchor_scores": anchor,
        }
    monkeypatch.setattr(
        reporter_v9b, "_load",
        lambda path, role: copy.deepcopy(fixtures[role]),
    )
    report = reporter_v9b.build_report("original", "replay")
    assert report["passed"] is True
    assert report["cosines"] == {
        "coefficient": pytest.approx(1.0),
        "standardized_domain_score": pytest.approx(1.0),
        "standardized_anchor_score": pytest.approx(1.0),
    }
    fixtures["replay"]["coefficient_sha256"] = "different"
    assert reporter_v9b.build_report("original", "replay")["passed"] is False
