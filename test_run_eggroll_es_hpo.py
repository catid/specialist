import copy
import json
import sys
from types import SimpleNamespace

import pytest

import run_eggroll_es_hpo as hpo


def make_args(tmp_path, trials=None, force=False):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}\n")
    (model / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    dataset_root = tmp_path / "dataset"
    train_dataset = dataset_root / "train"
    eval_dataset = dataset_root / "eval"
    (train_dataset / "train").mkdir(parents=True, exist_ok=True)
    (eval_dataset / "train").mkdir(parents=True, exist_ok=True)
    (train_dataset / "train/data.arrow").write_bytes(b"training")
    (eval_dataset / "train/data.arrow").write_bytes(b"validation")
    (dataset_root / "manifest.json").write_text(
        json.dumps({"schema": "test"}) + "\n"
    )
    return SimpleNamespace(
        python=tmp_path / "python",
        output=tmp_path / "output",
        model_name=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        steps=3,
        population_size=8,
        batch_size=64,
        max_tokens=32,
        seed=42,
        selection_split="train",
        guard_splits="",
        max_guard_degradation=0.0,
        max_guard_exact_loss=0,
        ood_prose_jsonl=None,
        max_ood_prose_degradation=0.0,
        n_vllm_engines=4,
        n_gpu_per_vllm_engine=1,
        use_gpus="0,1,2,3",
        reward_function_timeout=10,
        trials=trials,
        force=force,
    )


def make_summary(args, sigma, alpha, steps, score=0.5):
    return {
        "schema": "eggroll-es-exact-run-v1",
        **hpo.expected_summary(args, sigma, alpha, steps),
        "evaluations": {"final": {"train": score}},
    }


def test_cli_defaults_to_strict_ood_prose_policy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_eggroll_es_hpo.py"])

    assert hpo.parse_args().max_ood_prose_degradation == 0.0


def write_cache(args, name, sigma, alpha, steps, dataset, model, summary=None):
    run_dir = args.output / "runs" / name
    run_dir.mkdir(parents=True)
    command = hpo.command_for(args, name, sigma, alpha, steps)
    request = hpo.run_request(command, dataset, model)
    (run_dir / "command.json").write_text(json.dumps(command))
    (run_dir / "run_request.json").write_text(json.dumps(request))
    if summary is None:
        summary = make_summary(args, sigma, alpha, steps)
    (run_dir / "run_summary.json").write_text(json.dumps(summary))
    return run_dir


def test_select_best_can_retain_baseline():
    results = [{"name": "treatment", "validation_score": 0.4}]

    baseline, best_treatment, selected = hpo.select_best(
        0.5, "baseline.json", results
    )

    assert baseline["name"] == "baseline"
    assert best_treatment["name"] == "treatment"
    assert selected["name"] == "baseline"


def test_select_best_can_choose_treatment():
    results = [{"name": "treatment", "validation_score": 0.6}]

    _, _, selected = hpo.select_best(0.5, "baseline.json", results)

    assert selected["name"] == "treatment"


def test_guarded_selection_rejects_ood_regression():
    baseline = {
        "evaluations": {"final": {"train": 0.5, "ood": 0.4}}
    }
    results = [{
        "name": "treatment",
        "validation_score": 0.6,
        "evaluation_scores": {"train": 0.6, "ood": 0.39},
    }]

    _, best, best_guarded, selected = hpo.select_best_guarded(
        baseline, results, "train", ["ood"], 0.0
    )

    assert best["name"] == "treatment"
    assert best["guard_passed"] is False
    assert best_guarded is None
    assert selected["name"] == "baseline"


def test_guard_tolerance_can_admit_small_ood_loss():
    baseline = {
        "evaluations": {"final": {"train": 0.5, "ood": 0.4}}
    }
    results = [{
        "name": "treatment",
        "validation_score": 0.6,
        "evaluation_scores": {"train": 0.6, "ood": 0.395},
    }]

    _, _, best_guarded, selected = hpo.select_best_guarded(
        baseline, results, "train", ["ood"], 0.01
    )

    assert best_guarded["name"] == "treatment"
    assert selected["name"] == "treatment"


def test_guarded_selection_rejects_failed_ood_prose_gate():
    baseline = {
        "evaluations": {"final": {"train": 0.5, "ood": 0.4}}
    }
    results = [{
        "name": "treatment",
        "validation_score": 0.6,
        "evaluation_scores": {"train": 0.6, "ood": 0.4},
        "ood_prose_guard_passed": False,
    }]

    _, _, best_guarded, selected = hpo.select_best_guarded(
        baseline, results, "train", ["ood"], 0.0
    )

    assert best_guarded is None
    assert selected["name"] == "baseline"


def test_guarded_selection_rejects_exact_answer_loss():
    baseline = {
        "evaluations": {"final": {"train": 0.5, "ood": 0.4}},
        "evaluation_details": {"ood": {"exact": 5}},
    }
    results = [{
        "name": "treatment",
        "validation_score": 0.6,
        "evaluation_scores": {"train": 0.6, "ood": 0.4},
        "evaluation_details": {"ood": {"exact": 4}},
    }]

    _, _, best_guarded, selected = hpo.select_best_guarded(
        baseline, results, "train", ["ood"], 0.0, 0,
    )

    assert best_guarded is None
    assert selected["name"] == "baseline"

    _, _, best_guarded, selected = hpo.select_best_guarded(
        baseline, results, "train", ["ood"], 0.0, 1,
    )
    assert best_guarded["name"] == "treatment"
    assert selected["name"] == "treatment"


def test_command_preserves_virtualenv_python_symlink(tmp_path):
    args = make_args(tmp_path)
    target = tmp_path / "system-python"
    target.write_bytes(b"")
    args.python.symlink_to(target)

    command = hpo.command_for(args, "baseline", 0.0, 0.0, 0)

    assert command[0] == str(args.python.absolute())
    assert command[0] != str(args.python.resolve())


def test_command_and_snapshot_pin_opt_in_ood_prose(tmp_path):
    args = make_args(tmp_path)
    args.ood_prose_jsonl = tmp_path / "ood.jsonl"
    args.ood_prose_jsonl.write_text('{"text":"frozen"}\n')
    args.max_ood_prose_degradation = 0.02

    command = hpo.command_for(args, "candidate", 0.001, 0.00025, 3)
    snapshot = hpo.dataset_snapshot(args)

    assert command[-4:] == [
        "--ood-prose-jsonl", str(args.ood_prose_jsonl.resolve()),
        "--max-ood-prose-degradation", "0.02",
    ]
    assert snapshot["ood_prose_jsonl"]["sha256"] == hpo.file_sha256(
        args.ood_prose_jsonl
    )


def test_summary_requires_matching_ood_prose_gate(tmp_path):
    args = make_args(tmp_path)
    args.ood_prose_jsonl = tmp_path / "ood.jsonl"
    args.ood_prose_jsonl.write_text('{"text":"frozen"}\n')
    dataset = hpo.dataset_snapshot(args)
    summary = make_summary(args, 0.0, 0.0, 0)

    with pytest.raises(RuntimeError, match="ood_prose"):
        hpo.validate_summary(
            summary, hpo.expected_summary(args, 0.0, 0.0, 0),
            tmp_path / "summary.json", ["train"],
            dataset["ood_prose_jsonl"],
            args.max_ood_prose_degradation,
        )

    summary["ood_prose"] = {
        "dataset": {"sha256": dataset["ood_prose_jsonl"]["sha256"]},
        "gate": {
            "delta": 0.0,
            "max_degradation": 0.0,
            "paired_document_bootstrap_95_ci": [0.0, 0.0],
            "passed": True,
        },
    }
    hpo.validate_summary(
        summary, hpo.expected_summary(args, 0.0, 0.0, 0),
        tmp_path / "summary.json", ["train"],
        dataset["ood_prose_jsonl"],
        args.max_ood_prose_degradation,
    )


def test_summary_rejects_true_prose_gate_from_looser_policy(tmp_path):
    args = make_args(tmp_path)
    args.ood_prose_jsonl = tmp_path / "ood.jsonl"
    args.ood_prose_jsonl.write_text('{"text":"frozen"}\n')
    dataset = hpo.dataset_snapshot(args)
    summary = make_summary(args, 0.001, 0.00025, 3)
    summary["ood_prose"] = {
        "dataset": {"sha256": dataset["ood_prose_jsonl"]["sha256"]},
        "gate": {
            "delta": -0.004,
            "max_degradation": 0.02,
            "paired_document_bootstrap_95_ci": [-0.007, -0.001],
            "passed": True,
        },
    }

    with pytest.raises(RuntimeError, match="requested threshold"):
        hpo.validate_summary(
            summary, hpo.expected_summary(args, 0.001, 0.00025, 3),
            tmp_path / "summary.json", ["train"],
            dataset["ood_prose_jsonl"], 0.0,
        )


def test_guarded_selection_recomputes_strict_prose_policy():
    baseline = {
        "evaluations": {"final": {"train": 0.5, "ood": 0.4}}
    }
    results = [{
        "name": "treatment",
        "validation_score": 0.6,
        "evaluation_scores": {"train": 0.6, "ood": 0.4},
        "ood_prose_guard_passed": True,
        "ood_prose_gate": {
            "delta": -0.004,
            "max_degradation": 0.02,
            "paired_document_bootstrap_95_ci": [-0.007, -0.001],
            "passed": True,
        },
    }]

    _, _, best_guarded, selected = hpo.select_best_guarded(
        baseline, results, "train", ["ood"], 0.0, 0, 0.0,
    )

    assert best_guarded is None
    assert selected["name"] == "baseline"
    assert results[0]["ood_prose_gate_inspection"]["valid"] is False


def test_evaluation_details_pin_exact_nonzero_and_raw_hash(tmp_path):
    run_dir = tmp_path / "run"
    output = run_dir / "eval-output"
    output.mkdir(parents=True)
    path = output / "model_eval_taskood_iteration4.json"
    path.write_text(json.dumps([
        {"reward": 1.0, "format": "exact"},
        {"reward": 0.25, "format": "partial"},
        {"reward": 0.0, "format": "incorrect"},
    ]))

    details = hpo.evaluation_details(run_dir, {"ood": 1.25 / 3}, 3)

    assert details["ood"]["exact"] == 1
    assert details["ood"]["nonzero"] == 2
    assert details["ood"]["sha256"] == hpo.file_sha256(path)

    with pytest.raises(RuntimeError, match="disagrees"):
        hpo.evaluation_details(run_dir, {"ood": 0.0}, 3)


def test_cached_run_requires_exact_command_and_dataset_hash(
        tmp_path, monkeypatch):
    args = make_args(tmp_path)
    dataset = hpo.dataset_snapshot(args)
    model = hpo.model_snapshot(args.model_name)
    write_cache(args, "baseline", 0.0, 0.0, 0, dataset, model)
    monkeypatch.setattr(
        hpo.subprocess, "run",
        lambda *args, **kwargs: pytest.fail("cache should be reused"),
    )

    summary = hpo.run_one(
        args, "baseline", 0.0, 0.0, 0, dataset, model
    )
    assert summary["seed"] == 42

    changed_dataset = copy.deepcopy(dataset)
    changed_dataset["train_arrow"]["sha256"] = "changed"
    with pytest.raises(RuntimeError, match="does not match"):
        hpo.run_one(
            args, "baseline", 0.0, 0.0, 0, changed_dataset, model
        )

    command_path = args.output / "runs/baseline/command.json"
    recorded_command = json.loads(command_path.read_text())
    recorded_command.extend(["--max-tokens", "999"])
    command_path.write_text(json.dumps(recorded_command))
    with pytest.raises(RuntimeError, match="does not match"):
        hpo.run_one(args, "baseline", 0.0, 0.0, 0, dataset, model)


def test_cached_summary_must_match_seed_and_other_run_args(tmp_path):
    args = make_args(tmp_path)
    dataset = hpo.dataset_snapshot(args)
    model = hpo.model_snapshot(args.model_name)
    summary = make_summary(args, 0.0, 0.0, 0)
    summary["seed"] = 0
    write_cache(
        args, "baseline", 0.0, 0.0, 0, dataset, model, summary=summary
    )

    with pytest.raises(RuntimeError, match="requested run"):
        hpo.run_one(args, "baseline", 0.0, 0.0, 0, dataset, model)


def test_force_removes_stale_summary_and_reruns(tmp_path, monkeypatch):
    args = make_args(tmp_path)
    dataset = hpo.dataset_snapshot(args)
    model = hpo.model_snapshot(args.model_name)
    run_dir = write_cache(
        args, "baseline", 0.0, 0.0, 0, dataset, model
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)
        assert not (run_dir / "run_summary.json").exists()
        (run_dir / "run_summary.json").write_text(
            json.dumps(make_summary(args, 0.0, 0.0, 0, score=0.75))
        )

    monkeypatch.setattr(hpo.subprocess, "run", fake_run)
    summary = hpo.run_one(
        args, "baseline", 0.0, 0.0, 0, dataset, model, force=True
    )

    assert len(calls) == 1
    assert summary["evaluations"]["final"]["train"] == 0.75


def test_main_force_reaches_baseline_and_every_requested_trial(
        tmp_path, monkeypatch, capsys):
    trial_names = [trial["name"] for trial in hpo.CANDIDATES[:2]]
    args = make_args(tmp_path, trials=",".join(trial_names), force=True)
    calls = []

    def fake_run_one(args, name, sigma, alpha, steps, frozen_dataset,
                     frozen_model, force=False, trainer_sha256=None):
        calls.append((name, force))
        return make_summary(args, sigma, alpha, steps)

    monkeypatch.setattr(hpo, "parse_args", lambda: args)
    monkeypatch.setattr(hpo, "run_one", fake_run_one)
    monkeypatch.setattr(
        hpo, "evaluation_details",
        lambda run_dir, scores, steps: {
            split: {"exact": 1, "mean_reward": score}
            for split, score in scores.items()
        },
    )
    hpo.main()
    capsys.readouterr()

    assert calls == [
        ("baseline", True),
        (trial_names[0], True),
        (trial_names[1], True),
    ]
    journal = json.loads((args.output / "hpo_results.json").read_text())
    assert journal["baseline_selected"] is True
    assert journal["dataset"]["train_arrow_sha256"] == (
        hpo.file_sha256(args.train_dataset / "train/data.arrow")
    )


def test_dataset_snapshot_detects_arrow_mutation(tmp_path):
    args = make_args(tmp_path)
    before = hpo.dataset_snapshot(args)
    (args.train_dataset / "train/data.arrow").write_bytes(b"new training")

    with pytest.raises(RuntimeError, match="frozen dataset snapshot changed"):
        hpo.assert_snapshot(
            before, hpo.dataset_snapshot(args), "dataset", "after trial"
        )
