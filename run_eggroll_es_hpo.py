#!/usr/bin/env python3
"""Run reproducible short ES hyperparameter search on a frozen snapshot."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_PYTHON = ROOT / "es-at-scale/.venv/bin/python"
DEFAULT_OUTPUT = ROOT / "experiments/eggroll_es_hpo"
DEFAULT_MODEL = ROOT / "models/Qwen3.6-35B-A3B"
DEFAULT_DATASET_ROOT = ROOT / "data/eggroll_es_specialist"
DEFAULT_TRAIN_DATASET = DEFAULT_DATASET_ROOT / "train"
DEFAULT_EVAL_DATASET = DEFAULT_DATASET_ROOT / "eval"
TRAINER = ROOT / "train_eggroll_es_specialist.py"
CANDIDATES = [
    {"name": "sigma3e-4_alpha1.5e-4", "sigma": 0.0003, "alpha": 0.00015},
    {"name": "sigma5e-4_alpha2.5e-4", "sigma": 0.0005, "alpha": 0.00025},
    {"name": "sigma1e-3_alpha2.5e-4", "sigma": 0.001, "alpha": 0.00025},
    {"name": "sigma1e-3_alpha5e-4", "sigma": 0.001, "alpha": 0.0005},
    {"name": "sigma1e-3_alpha1e-3", "sigma": 0.001, "alpha": 0.001},
    {"name": "sigma2e-3_alpha1e-3", "sigma": 0.002, "alpha": 0.001},
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--train-dataset", type=Path, default=DEFAULT_TRAIN_DATASET
    )
    parser.add_argument(
        "--eval-dataset", type=Path, default=DEFAULT_EVAL_DATASET
    )
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-vllm-engines", type=int, default=4)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--use-gpus", default="0,1,2,3")
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument(
        "--trials", help="Optional comma-separated candidate names"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def normalized_path(path):
    return Path(path).expanduser().resolve()


def executable_path(path):
    """Return an absolute executable path without dereferencing venv links."""
    return Path(path).expanduser().absolute()


def command_for(args, name, sigma, alpha, steps):
    return [
        str(executable_path(args.python)), str(TRAINER),
        "--model-name", str(normalized_path(args.model_name)),
        "--train-dataset", str(normalized_path(args.train_dataset)),
        "--eval-dataset", str(normalized_path(args.eval_dataset)),
        "--exact-train-steps", str(steps),
        "--skip-baseline-eval",
        "--eval-splits", "train",
        "--sigma", str(sigma),
        "--alpha", str(alpha),
        "--population-size", str(args.population_size),
        "--batch-size", str(args.batch_size),
        "--mini-batch-size", str(args.batch_size),
        "--max-tokens", str(args.max_tokens),
        "--seed", str(args.seed),
        "--n-vllm-engines", str(args.n_vllm_engines),
        "--n-gpu-per-vllm-engine", str(args.n_gpu_per_vllm_engine),
        "--use-gpus", args.use_gpus,
        "--reward-function-timeout", str(args.reward_function_timeout),
        "--logging", "none",
        "--output-directory", str(normalized_path(args.output) / "runs"),
        "--experiment-name", name,
    ]


def single_arrow(dataset_path, split):
    arrow_files = sorted(
        (normalized_path(dataset_path) / split).glob("*.arrow")
    )
    if len(arrow_files) != 1:
        raise RuntimeError(
            f"expected exactly one Arrow file for {split!r} under "
            f"{normalized_path(dataset_path)}, found {len(arrow_files)}"
        )
    return arrow_files[0]


def file_snapshot(path):
    path = normalized_path(path)
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def dataset_snapshot(args):
    """Hash every Arrow file that can affect HPO train-split selection."""
    train_dataset = normalized_path(args.train_dataset)
    eval_dataset = normalized_path(args.eval_dataset)
    common_root = train_dataset.parent
    manifest = common_root / "manifest.json"
    snapshot = {
        "train_dataset": str(train_dataset),
        "eval_dataset": str(eval_dataset),
        "train_arrow": file_snapshot(single_arrow(train_dataset, "train")),
        "selection_eval_arrow": file_snapshot(
            single_arrow(eval_dataset, "train")
        ),
    }
    if manifest.is_file() and eval_dataset.parent == common_root:
        snapshot["manifest"] = file_snapshot(manifest)
    return snapshot


def model_snapshot(model_path):
    """Record a cheap identity guard for the large local model snapshot.

    Hashing every 35B-model weight for each HPO invocation is prohibitive.  The
    index/config hashes plus every shard's path, size, and modification time
    detect normal replacement or mutation while keeping startup inexpensive.
    """
    model_path = normalized_path(model_path)
    if not model_path.is_dir():
        raise RuntimeError(f"model directory does not exist: {model_path}")
    metadata = {}
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
    ):
        path = model_path / name
        if path.is_file():
            metadata[name] = file_snapshot(path)
    shards = [
        {
            "path": str(path.relative_to(model_path)),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in sorted(model_path.glob("*.safetensors"))
    ]
    if not shards:
        raise RuntimeError(f"model has no safetensor shards: {model_path}")
    return {
        "path": str(model_path),
        "metadata": metadata,
        "weight_shards": shards,
    }


def assert_snapshot(expected, actual, label, stage):
    if actual != expected:
        raise RuntimeError(
            f"frozen {label} snapshot changed {stage}. Refusing to mix "
            "incomparable HPO results"
        )


def expected_summary(args, sigma, alpha, steps):
    return {
        "model": str(normalized_path(args.model_name)),
        "steps": steps,
        "sigma": sigma,
        "alpha": alpha,
        "population_size": args.population_size,
        "batch_size": args.batch_size,
        "mini_batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }


def validate_summary(summary, expected, summary_path):
    if not isinstance(summary, dict):
        raise RuntimeError(f"invalid cached run summary: {summary_path}")
    mismatches = {
        key: {"expected": value, "recorded": summary.get(key)}
        for key, value in expected.items()
        if summary.get(key) != value
    }
    try:
        float(summary["evaluations"]["final"]["train"])
    except (KeyError, TypeError, ValueError):
        mismatches["evaluations.final.train"] = {
            "expected": "numeric validation score",
            "recorded": summary.get("evaluations"),
        }
    if mismatches:
        raise RuntimeError(
            f"cached run summary does not match requested run at "
            f"{summary_path}: {json.dumps(mismatches, sort_keys=True)}"
        )


def run_request(command, frozen_dataset, frozen_model, trainer_sha256=None):
    if trainer_sha256 is None:
        trainer_sha256 = file_sha256(TRAINER)
    return {
        "schema": "eggroll-es-hpo-run-request-v1",
        "command": command,
        "dataset": frozen_dataset,
        "model": frozen_model,
        "trainer_sha256": trainer_sha256,
    }


def load_cached_run(run_dir, command, request, expected):
    summary_path = run_dir / "run_summary.json"
    command_path = run_dir / "command.json"
    request_path = run_dir / "run_request.json"
    missing = [
        str(path) for path in (command_path, request_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "cached summary lacks frozen-run provenance; rerun with --force: "
            + ", ".join(missing)
        )
    recorded_command = json.loads(command_path.read_text())
    recorded_request = json.loads(request_path.read_text())
    if recorded_command != command or recorded_request != request:
        raise RuntimeError(
            f"cached command or snapshot does not match requested run at "
            f"{run_dir}. Rerun with --force"
        )
    summary = json.loads(summary_path.read_text())
    validate_summary(summary, expected, summary_path)
    return summary


def run_one(args, name, sigma, alpha, steps, frozen_dataset, frozen_model,
            force=False, trainer_sha256=None):
    run_dir = normalized_path(args.output) / "runs" / name
    summary_path = run_dir / "run_summary.json"
    command = command_for(args, name, sigma, alpha, steps)
    request = run_request(
        command, frozen_dataset, frozen_model, trainer_sha256
    )
    expected = expected_summary(args, sigma, alpha, steps)
    if summary_path.exists() and not force:
        return load_cached_run(run_dir, command, request, expected)

    run_dir.mkdir(parents=True, exist_ok=True)
    # A failed forced rerun must never leave a stale summary that looks fresh.
    summary_path.unlink(missing_ok=True)
    (run_dir / "command.json").write_text(
        json.dumps(command, indent=2) + "\n"
    )
    (run_dir / "run_request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n"
    )
    with (run_dir / "stdout.log").open("w") as log:
        subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            check=True,
        )
    if not summary_path.is_file():
        raise RuntimeError(f"training did not write summary: {summary_path}")
    summary = json.loads(summary_path.read_text())
    validate_summary(summary, expected, summary_path)
    return summary


def select_best(baseline_score, baseline_summary, results):
    """Select across the no-op baseline and trained candidates."""
    baseline = {
        "name": "baseline",
        "sigma": 0.0,
        "alpha": 0.0,
        "steps": 0,
        "validation_score": baseline_score,
        "run_summary": str(baseline_summary),
    }
    best_treatment = max(
        results, key=lambda item: item["validation_score"]
    )
    selected = max(
        [baseline, *results], key=lambda item: item["validation_score"]
    )
    return baseline, best_treatment, selected


def selected_candidates(trials):
    if not trials:
        return CANDIDATES
    requested = {
        name.strip() for name in trials.split(",") if name.strip()
    }
    selected = [
        trial for trial in CANDIDATES if trial["name"] in requested
    ]
    missing = requested - {trial["name"] for trial in selected}
    if missing:
        raise ValueError(f"unknown HPO trials: {sorted(missing)}")
    if not selected:
        raise ValueError("at least one HPO trial must be requested")
    return selected


def main():
    args = parse_args()
    args.output = normalized_path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    selected = selected_candidates(args.trials)

    # Capture once before baseline, then compare every run boundary against
    # this exact dataset/model identity.  This prevents a curator update from
    # silently mixing two snapshots within one A/B journal.
    frozen_dataset = dataset_snapshot(args)
    frozen_model = model_snapshot(args.model_name)
    frozen_trainer_sha256 = file_sha256(TRAINER)
    manifest_path = (
        normalized_path(args.train_dataset).parent / "manifest.json"
    )
    dataset_manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.is_file() else None
    )
    assert_snapshot(
        frozen_dataset, dataset_snapshot(args), "dataset", "before baseline"
    )
    assert_snapshot(
        frozen_trainer_sha256, file_sha256(TRAINER), "trainer source",
        "before baseline",
    )
    baseline = run_one(
        args, "baseline", 0.0, 0.0, 0, frozen_dataset, frozen_model,
        force=args.force, trainer_sha256=frozen_trainer_sha256,
    )
    assert_snapshot(
        frozen_dataset, dataset_snapshot(args), "dataset", "after baseline"
    )
    assert_snapshot(
        frozen_model, model_snapshot(args.model_name), "model",
        "after baseline",
    )
    assert_snapshot(
        frozen_trainer_sha256, file_sha256(TRAINER), "trainer source",
        "after baseline",
    )
    results = []
    for trial in selected:
        assert_snapshot(
            frozen_dataset, dataset_snapshot(args), "dataset",
            f"before trial {trial['name']}",
        )
        assert_snapshot(
            frozen_model, model_snapshot(args.model_name), "model",
            f"before trial {trial['name']}",
        )
        assert_snapshot(
            frozen_trainer_sha256, file_sha256(TRAINER), "trainer source",
            f"before trial {trial['name']}",
        )
        summary = run_one(
            args, steps=args.steps, frozen_dataset=frozen_dataset,
            frozen_model=frozen_model, force=args.force,
            trainer_sha256=frozen_trainer_sha256, **trial,
        )
        assert_snapshot(
            frozen_dataset, dataset_snapshot(args), "dataset",
            f"after trial {trial['name']}",
        )
        assert_snapshot(
            frozen_model, model_snapshot(args.model_name), "model",
            f"after trial {trial['name']}",
        )
        assert_snapshot(
            frozen_trainer_sha256, file_sha256(TRAINER), "trainer source",
            f"after trial {trial['name']}",
        )
        results.append({
            **trial,
            "steps": args.steps,
            "validation_score": summary["evaluations"]["final"]["train"],
            "run_summary": str(
                args.output / "runs" / trial["name"] / "run_summary.json"
            ),
        })
        baseline_score = baseline["evaluations"]["final"]["train"]
        baseline_result, best_treatment, selected_result = select_best(
            baseline_score,
            args.output / "runs" / "baseline" / "run_summary.json",
            results,
        )
        journal = {
            "schema": "eggroll-es-hpo-v1",
            "selection_split": "train",
            "final_holdout_used_for_selection": False,
            "baseline_validation_score": baseline_score,
            "baseline": baseline_result,
            "population_size": args.population_size,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "dataset": {
                "manifest": dataset_manifest,
                "snapshot": frozen_dataset,
                # Retain the original flat key for journal consumers.
                "train_arrow_sha256": frozen_dataset["train_arrow"][
                    "sha256"
                ],
            },
            "model": frozen_model,
            "trainer_sha256": frozen_trainer_sha256,
            "results": results,
            "best_treatment": best_treatment,
            "best": selected_result,
            "baseline_selected": selected_result["name"] == "baseline",
        }
        (args.output / "hpo_results.json").write_text(
            json.dumps(journal, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(results[-1], sort_keys=True), flush=True)

    print(json.dumps(journal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
