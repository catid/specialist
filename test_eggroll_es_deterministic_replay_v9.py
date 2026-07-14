import contextlib
import copy
import json
from pathlib import Path

import pytest

import report_eggroll_es_deterministic_replay_v9 as reporter_v9
import run_eggroll_es_anchor_replay_v9 as replay_v9
import train_eggroll_es_specialist_anchor_v8 as anchor_v8


ROOT = Path(__file__).resolve().parent
PLAN_SHA = replay_v9.MIDDLE_LATE_PLAN_SHA256_V9


def fake_evidence():
    return {
        "schema": "eggroll-es-failed-v8-evidence-binding-v9",
        "binding_sha256": "test-v8-failure-binding",
    }


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
        "--v9-stage", "deterministic_replay",
        "--v9-v8-failed-report", str(replay_v9.V8_FAILED_REPORT_PATH_V9),
        "--v9-perturbation-basis-seed", str(basis),
        "--population-size", str(population), "--batch-size", "64",
        "--seed", str(seed), "--target-alphas", target,
        "--experiment-name", (
            "snapshot794_layer_v9_middle_late_exact_replay_"
            "data43_basis20260714_retry1"
        ),
        "--v9-dry-run",
    ]


def test_v9_exact_predecessor_evidence_and_basis():
    evidence = replay_v9._v8_failed_evidence_v9(
        replay_v9.V8_FAILED_REPORT_PATH_V9
    )
    assert evidence["formal_v8_result"] == {
        "same_basis_coefficient_cosine": 0.4276943787514416,
        "threshold": 0.5, "passed": False,
    }
    assert evidence["binding_sha256"] == (
        replay_v9.EXPECTED_V8_FAILED_EVIDENCE_BINDING_SHA256_V9
    )
    assert replay_v9.PERTURBATION_SEEDS_V9 == (
        replay_v9.driver_v8.PERTURBATION_SEEDS_V8
    )
    assert replay_v9.validate_effective_anchor_api()


def test_v9_dry_run_and_fail_closed_cli(monkeypatch, capsys):
    evidence = fake_evidence()
    monkeypatch.setattr(replay_v9, "_v8_failed_evidence_v9", lambda path: evidence)
    recipe = replay_v9.frozen_recipe_v9(evidence)
    monkeypatch.setattr(
        replay_v9, "EXPECTED_RECIPE_SHA256_V9",
        replay_v9.driver_v1.canonical_sha256(recipe),
    )
    result = replay_v9.main(cli())
    assert result["schema"] == "eggroll-es-deterministic-replay-dry-run-v9"
    assert result["data_seed"] == 43
    assert result["population_size"] == 32
    assert result["targets"] == [0.0]
    assert result["replay_cosine_threshold"] == 0.99
    assert result["v8_failed_evidence_binding_sha256"] == (
        "test-v8-failure-binding"
    )
    assert "deterministic-replay-dry-run-v9" in capsys.readouterr().out
    for args, message in (
        (cli(seed=44), "seed must be 43"),
        (cli(population=16), "runtime recipe changed"),
        (cli(basis=43), "basis seed changed"),
        (cli(target="0,0.000001"), "exactly alpha zero"),
    ):
        bundle, remaining = anchor_v8.parse_frozen_layer_plan_cli_v8(args)
        with pytest.raises(ValueError, match=message):
            replay_v9.validate_frozen_execution_cli_v9(remaining, bundle)


def test_v9_execute_replaces_only_inherited_population_seeds(monkeypatch):
    replay_v9._ACTIVE_EXECUTION_V9 = {"data_seed": 43}
    inherited = replay_v9.np.random.default_rng(seed=43).integers(
        0, 2**30, size=32, dtype=replay_v9.np.int64,
    ).tolist()
    captured = {}

    class StopAfterCapture(Exception):
        pass

    def capture(*args, **kwargs):
        captured["seeds"] = kwargs["seeds"]
        raise StopAfterCapture

    monkeypatch.setattr(replay_v9.driver_v4, "execute_line_search", capture)
    monkeypatch.setattr(
        replay_v9.driver_v6, "scoped_legacy_audit_v6",
        contextlib.nullcontext,
    )
    with pytest.raises(StopAfterCapture):
        replay_v9.execute_line_search(object(), seeds=inherited)
    assert captured["seeds"] == replay_v9.PERTURBATION_SEEDS_V9
    with pytest.raises(RuntimeError, match="population schedule changed"):
        replay_v9.execute_line_search(object(), seeds=[1] * 32)


def test_v9_report_requires_exact_identity_and_cosine(monkeypatch):
    coefficients = [float(index - 16) for index in range(32)]
    domain = [float(index * index + 1) for index in range(32)]
    anchor = [float(64 - index) for index in range(32)]

    def fixture(role):
        suffix = "original" if role == "original" else "replay"
        return {
            "role": role, "data_seed": 43,
            "journal": f"/{suffix}.json",
            "journal_file_sha256": f"file-{suffix}",
            "content_sha256": f"content-{suffix}",
            "coefficient_sha256": "same-coefficient",
            "robust_plan_sha256": f"robust-{suffix}",
            "domain_scores_sha256": reporter_v9.vector_sha256(domain),
            "anchor_scores_sha256": reporter_v9.vector_sha256(anchor),
            "perturbation_basis_sha256": (
                replay_v9.PERTURBATION_BASIS_SHA256_V9
            ),
            "coefficients": list(coefficients),
            "domain_scores": list(domain), "anchor_scores": list(anchor),
        }

    fixtures = {role: fixture(role) for role in ("original", "replay")}
    monkeypatch.setattr(
        replay_v9, "_v8_failed_evidence_v9",
        lambda path: {"binding_sha256": "failed-v8"},
    )
    monkeypatch.setattr(
        reporter_v9, "_load_vectors",
        lambda path, role: copy.deepcopy(fixtures[role]),
    )
    report = reporter_v9.build_report("original", "replay")
    assert report["passed"] is True
    assert report["all_exact_identities_match"] is True
    assert report["cosines"] == {
        "coefficient": pytest.approx(1.0),
        "standardized_domain_score": pytest.approx(1.0),
        "standardized_anchor_score": pytest.approx(1.0),
    }
    assert "mean_reward" not in repr(report).casefold()
    fixtures["replay"]["coefficient_sha256"] = "different"
    fixtures["replay"]["coefficients"] = list(reversed(coefficients))
    failed = reporter_v9.build_report("original", "replay")
    assert failed["passed"] is False
    assert failed["exact_identities"]["coefficient_sha256"] is False
    assert failed["cosines"]["coefficient"] < 0.99


def test_v9_completed_outer_contract_rejects_basis_tamper(monkeypatch):
    source = replay_v9.ORIGINAL_V8_JOURNALS_V9[43]["path"]
    journal = json.loads(source.read_text())
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v9"
    snapshot = journal["snapshot"]
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v9"
    snapshot.pop("split_seed_stability_v8")
    for key in (
        "split_seed_stability_family_v8", "stage_v8",
        "target_alpha_zero_only_v8", "benchmark_selection_forbidden_v8",
        "same_perturbation_basis_required_v8",
        "cross_data_seed_coefficient_cosine_threshold_v8",
        "requires_complete_v7_family_v8",
    ):
        journal["policy"].pop(key)
    journal["policy"].update(replay_v9.V9_POLICY)
    evidence = fake_evidence()
    monkeypatch.setattr(replay_v9, "_v8_failed_evidence_v9", lambda path: evidence)
    recipe = replay_v9.frozen_recipe_v9(evidence)
    monkeypatch.setattr(
        replay_v9, "EXPECTED_RECIPE_SHA256_V9",
        replay_v9.driver_v1.canonical_sha256(recipe),
    )
    implementation = {
        key: anchor_v8.file_sha256(path)
        for key, path in replay_v9.V9_IMPLEMENTATION_PATHS.items()
    }
    snapshot["implementation"] = implementation
    snapshot["deterministic_replay_v9"] = {
        "schema": "eggroll-es-deterministic-replay-snapshot-v9",
        "family": "exact_v8_data43_replay", "stage": "deterministic_replay",
        "arm": "middle_late", "layers": [20, 21, 22, 23], "data_seed": 43,
        "reference_role": "original_v8_data43", "replay_role": "v9_exact_data43",
        "perturbation_basis_seed": 20260714, "perturbation_seed_count": 32,
        "perturbation_seeds": list(replay_v9.PERTURBATION_SEEDS_V9),
        "perturbation_basis_sha256": replay_v9.PERTURBATION_BASIS_SHA256_V9,
        "target_alphas": [0.0], "benchmark_treatment_applied": False,
        "selection_surface": "coefficient_and_raw_score_identity_only",
        "replay_cosine_threshold": 0.99,
        "required_exact_identities": [
            "coefficient_sha256", "domain_scores_sha256", "anchor_scores_sha256",
        ],
        "v8_failed_evidence": evidence, "recipe": recipe,
        "plan_sha256": PLAN_SHA,
        "plan_file_sha256": anchor_v8.FROZEN_STABILITY_PLANS_V8[PLAN_SHA][
            "file_sha256"
        ],
        "source_unit_count": 35, "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
        "selected_byte_count": 285_999_104,
        "implementation_bundle_sha256": (
            replay_v9.driver_v1.canonical_sha256(implementation)
        ),
    }
    snapshot["recipe"] = {
        key: recipe[key] for key in (
            "model_name", "checkpoint", "sigma", "population_size",
            "batch_size", "mini_batch_size", "max_tokens", "seed",
            "min_anchor_cosine", "anchor_items_per_step", "target_alphas",
        )
    }
    monkeypatch.setattr(
        replay_v9, "_validate_inherited_zero_target_v9",
        lambda value: {"data_bootstrap_seed": 43},
    )
    def forbidden_outer_scope():
        raise AssertionError("v9 must let the v8 validator own the v6 scope")
    monkeypatch.setattr(
        replay_v9.driver_v6, "scoped_legacy_audit_v6",
        forbidden_outer_scope,
    )
    journal.pop("content_sha256_before_self_field")
    journal["content_sha256_before_self_field"] = (
        replay_v9.driver_v1.canonical_sha256(journal)
    )
    assert replay_v9.validate_completed_journal_v9(journal)["data_seed"] == 43
    tampered = copy.deepcopy(journal)
    tampered["seeds"][0] += 1
    tampered["content_sha256_before_self_field"] = (
        replay_v9.driver_v1.canonical_sha256({
            key: value for key, value in tampered.items()
            if key != "content_sha256_before_self_field"
        })
    )
    with pytest.raises(RuntimeError, match="changed basis/target"):
        replay_v9.validate_completed_journal_v9(tampered)
