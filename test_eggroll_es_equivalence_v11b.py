import inspect
import json

import pytest
from datasets import load_from_disk

import eggroll_es_worker_v11b as worker_v11b
import run_eggroll_es_anchor_equivalence_v11 as driver_v11
import run_eggroll_es_anchor_equivalence_v11b as driver_v11b
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v11b as anchor_v11b


PLAN_SHA = driver_v11.MIDDLE_LATE_PLAN_SHA256_V11


def load_bundle():
    spec = anchor_v11.FROZEN_STABILITY_PLANS_V11[PLAN_SHA]
    return anchor_v11b.load_frozen_layer_plan_v11b(
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
        "--v11b-stage", "equivalence_retry",
        "--v11b-v10-report", str(driver_v11.V10_REPORT_PATH_V11),
        "--v11b-failed-v11-journal",
        str(driver_v11b.FAILED_V11_JOURNAL_PATH_V11B),
        "--v11b-perturbation-basis-seed", str(basis),
        "--population-size", "32", "--batch-size", str(batch),
        "--mini-batch-size", "64", "--seed", "43",
        "--target-alphas", target,
        "--experiment-name", driver_v11b.EXPERIMENT_NAME_V11B,
        "--v11b-dry-run",
    ]


def raw_and_templated_batches():
    dataset = load_from_disk(str(driver_v11.FROZEN_TRAIN_DATASET_V11))["train"]
    questions, answers = next(iter(driver_v11b._raw_crossed_train_loader_v11b(
        dataset, 128, 43,
    )))
    prompts = [base.specialist_template(question) for question in questions]
    return questions, answers, {
        "D43": (prompts[:64], answers[:64]),
        "D44": (prompts[64:], answers[64:]),
    }


def test_v11b_raw_and_templated_manifests_are_distinct_and_both_exact():
    questions, answers, templated = raw_and_templated_batches()
    assert questions[0] != templated["D43"][0][0]
    anchor_v11b.validate_templated_domain_batches_v11b(templated)
    assert anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B == anchor_v11.DOMAIN_MANIFESTS_V11
    assert {
        label: spec["sha256"]
        for label, spec in anchor_v11b.TEMPLATED_DOMAIN_MANIFESTS_V11B.items()
    } == {
        "D43": "54f53464e479fa9dd0c80263f0e424a3d225681c1d8f15554b171f6d5b40c637",
        "D44": "44cc0ba38c7b2c685a2c44699be9f6dd6313c1391765e13c046812f06e280c23",
    }
    wrong = {
        "D43": (questions[:64], answers[:64]),
        "D44": (questions[64:], answers[64:]),
    }
    with pytest.raises(RuntimeError, match="templated D43"):
        anchor_v11b.validate_templated_domain_batches_v11b(wrong)


def test_v11b_regression_rejects_templated_values_on_raw_loader_surface(monkeypatch):
    dataset = load_from_disk(str(driver_v11.FROZEN_TRAIN_DATASET_V11))["train"]
    changed = json.loads(json.dumps(anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B))
    changed["D43"]["sha256"] = anchor_v11b.TEMPLATED_DOMAIN_MANIFESTS_V11B[
        "D43"
    ]["sha256"]
    monkeypatch.setattr(anchor_v11b, "RAW_DOMAIN_MANIFESTS_V11B", changed)
    with pytest.raises(RuntimeError, match="raw D43"):
        driver_v11b._raw_crossed_train_loader_v11b(dataset, 128, 43)


def test_v11b_failed_v11_and_v10_pass_are_both_bound():
    failed = driver_v11b._failed_v11_evidence_v11b(
        driver_v11b.FAILED_V11_JOURNAL_PATH_V11B
    )
    failed_journal = json.loads(
        driver_v11b.FAILED_V11_JOURNAL_PATH_V11B.read_text()
    )
    v10 = failed_journal["snapshot"]["resident_sign_equivalence_v11"][
        "v10_equivalence_evidence"
    ]
    assert failed["failure_message"] == "v11 captured D43 manifest changed"
    assert failed["no_coefficient_plan_estimated"] is True
    assert v10["passed"] is True


def test_v11b_strict_dry_run_and_fail_closed_cli(capsys):
    result = driver_v11b.main(cli())
    assert result["schema"] == (
        "eggroll-es-resident-sign-dual-manifest-dry-run-v11b"
    )
    assert result["actual_perturb_restore_cycle_count"] == 64
    assert result["all_engine_sign_residency_count"] == 16
    assert result["gpu_ids"] == [0, 1, 2, 3]
    assert "dual-manifest-dry-run-v11b" in capsys.readouterr().out
    for args, message in (
        (cli(batch=64), "combined-batch128"),
        (cli(target="0,0.1"), "exactly alpha zero"),
        (cli(basis=43), "basis seed changed"),
    ):
        bundle, remaining = anchor_v11b.parse_frozen_layer_plan_cli_v11b(args)
        with pytest.raises(ValueError, match=message):
            driver_v11b.validate_frozen_execution_cli_v11b(
                remaining, bundle,
            )


def test_v11b_worker_mro_and_original_v11_sources_remain_dependencies():
    trainer = anchor_v11b.load_trainer(load_bundle())
    assert trainer.estimate_step_coefficients.__module__ == anchor_v11b.__name__
    assert trainer._evaluate_population_with_anchor.__module__ == anchor_v11.__name__
    assert list(worker_v11b.FROZEN_LAYER_PLANS_V11B) == [PLAN_SHA]
    source = inspect.getsource(
        anchor_v11b.DualManifestResidentSignContractMixinV11B
        .estimate_step_coefficients
    )
    assert "validate_templated_domain_batches_v11b" in source
    capture_validation = source[:source.index("self._v11_population_call_index")]
    assert "RAW_DOMAIN_MANIFESTS_V11B" not in capture_validation
