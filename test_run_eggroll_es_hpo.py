import copy
import json
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


def test_command_preserves_virtualenv_python_symlink(tmp_path):
    args = make_args(tmp_path)
    target = tmp_path / "system-python"
    target.write_bytes(b"")
    args.python.symlink_to(target)

    command = hpo.command_for(args, "baseline", 0.0, 0.0, 0)

    assert command[0] == str(args.python.absolute())
    assert command[0] != str(args.python.resolve())


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
