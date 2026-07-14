#!/usr/bin/env python3
"""Resident-engine monotonic alpha search for prose-anchored EGGROLL-ES.

One population and anchor estimate is reused at every target alpha.  Alpha zero
is evaluated first, then strictly positive increments are applied without
rollback.  The driver is deliberately evaluation-only with respect to model
selection: it records strict guards for every state but never chooses a winner.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from torch.utils.data import DataLoader

import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor as anchor


ROOT = Path(__file__).resolve().parent
ALLOWED_EVAL_SPLITS = ("validation", "ood_qa")
JOURNAL_NAME = "alpha_line_search.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def atomic_write_json(path, value):
    """Replace a journal only after its complete bytes reach disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    raw = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True,
    ).encode("utf-8") + b"\n"
    with temporary.open("wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def parse_target_alphas(value):
    """Require an explicit zero followed by strictly increasing targets."""
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("target alphas are empty")
    try:
        targets = [float(piece) for piece in pieces]
    except ValueError as error:
        raise ValueError("target alphas must be numeric") from error
    if not all(math.isfinite(target) for target in targets):
        raise ValueError("target alphas must be finite")
    if targets[0] != 0.0:
        raise ValueError("target alphas must begin with exactly zero")
    if any(right <= left for left, right in zip(targets, targets[1:])):
        raise ValueError("target alphas must be strictly increasing")
    return targets


def validate_eval_splits(value):
    splits = tuple(
        piece.strip() for piece in value.split(",") if piece.strip()
    )
    if splits != ALLOWED_EVAL_SPLITS:
        raise ValueError(
            "line search eval splits must be exactly validation,ood_qa"
        )
    return splits


def load_allowlisted_eval_datasets(root, splits, loader=load_from_disk):
    """Load only allowed split directories, never the sealed heldout Arrow."""
    if tuple(splits) != ALLOWED_EVAL_SPLITS:
        raise ValueError("unsafe evaluation split request")
    root = Path(root)
    metadata_path = root / "dataset_dict.json"
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError("evaluation dataset metadata is invalid")
    available = metadata.get("splits")
    if not isinstance(available, list) or not all(
        isinstance(split, str) for split in available
    ):
        raise ValueError("evaluation dataset split metadata is invalid")
    missing = [split for split in splits if split not in available]
    if missing:
        raise ValueError(f"missing allowlisted evaluation splits: {missing}")
    return {
        split: loader(str(root / split))
        for split in splits
    }


def summarize_eval_file(path):
    path = Path(path)
    raw = path.read_bytes()
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"evaluation output is empty or invalid: {path}")
    rewards = []
    exact = 0
    nonzero = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"evaluation output row is invalid: {path}")
        reward = float(row.get("reward"))
        if not math.isfinite(reward):
            raise ValueError(f"evaluation reward is not finite: {path}")
        rewards.append(reward)
        exact += row.get("format") == "exact"
        nonzero += reward > 0.0
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows),
        "mean_reward": math.fsum(rewards) / len(rewards),
        "exact": exact,
        "nonzero": nonzero,
    }


def strict_qa_gate(baseline, candidate):
    deltas = {
        "mean_reward": (
            candidate["mean_reward"] - baseline["mean_reward"]
        ),
        "exact": candidate["exact"] - baseline["exact"],
        "nonzero": candidate["nonzero"] - baseline["nonzero"],
    }
    return {
        "schema": "eggroll-es-strict-qa-nondegradation-v1",
        "max_mean_reward_degradation": 0.0,
        "max_exact_loss": 0,
        "max_nonzero_loss": 0,
        "deltas": deltas,
        "passed": (
            deltas["mean_reward"] >= 0.0
            and deltas["exact"] >= 0
            and deltas["nonzero"] >= 0
        ),
    }


def summarize_prose_evaluation(evaluation):
    path = Path(evaluation["results_path"])
    return {
        "results_path": str(path),
        "results_sha256": file_sha256(path),
        "item_count": int(evaluation["item_count"]),
        "scored_token_count": int(evaluation["scored_token_count"]),
        "mean_token_logprob": float(evaluation["mean_token_logprob"]),
    }


def dataset_split_identity(dataset):
    files = []
    for item in sorted(
        dataset.cache_files, key=lambda value: value["filename"],
    ):
        path = Path(item["filename"])
        files.append({
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        })
    if not files:
        raise ValueError("dataset split has no pinned Arrow cache file")
    return {"rows": len(dataset), "arrow_files": files}


def build_snapshot(
    train_dataset,
    eval_datasets,
    anchor_dataset,
    batch_questions,
    batch_answers,
):
    if set(eval_datasets) != set(ALLOWED_EVAL_SPLITS):
        raise ValueError("snapshot received an unsafe evaluation split set")
    return {
        "schema": "eggroll-es-anchor-line-search-snapshot-v1",
        "train": dataset_split_identity(train_dataset),
        "evaluations": {
            split: dataset_split_identity(eval_datasets[split])
            for split in ALLOWED_EVAL_SPLITS
        },
        "anchor": {
            "path": anchor_dataset["path"],
            "sha256": anchor_dataset["sha256"],
            "rows": len(anchor_dataset["rows"]),
            "report": anchor_dataset["report"],
        },
        "fixed_train_batch": {
            "rows": len(batch_questions),
            "sha256": canonical_sha256({
                "questions": list(batch_questions),
                "answers": list(batch_answers),
            }),
        },
        "implementation": {
            "driver": file_sha256(Path(__file__).resolve()),
            "anchor_trainer": file_sha256(Path(anchor.__file__).resolve()),
            "base_trainer": file_sha256(Path(base.__file__).resolve()),
            "projection": file_sha256(ROOT / "eggroll_es_anchor.py"),
            "upstream_trainer": file_sha256(
                ROOT / "es-at-scale/es_at_scale/trainer/es_trainer.py"
            ),
            "upstream_worker": file_sha256(
                ROOT / "es-at-scale/es_at_scale/utils/worker_extension.py"
            ),
        },
    }


def _eval_paths(trainer, iteration, splits):
    return {
        split: (
            Path(trainer.logging_dir) / "eval-output"
            / f"model_eval_task{split}_iteration{iteration + 1}.json"
        )
        for split in splits
    }


def _verify_plan(plan, expected_sha):
    actual = anchor.coefficient_sha256(
        plan.get("seeds", []), plan.get("coefficients", []),
    )
    if (
        actual != expected_sha
        or plan.get("coefficient_sha256") != expected_sha
    ):
        raise ValueError("fixed coefficient plan changed during line search")


def execute_line_search(
    trainer,
    *,
    targets,
    seeds,
    input_prompts,
    target_answers,
    snapshot,
    journal_path,
    eval_splits=ALLOWED_EVAL_SPLITS,
    prose_scorer=None,
    prose_comparator=None,
):
    """Evaluate a monotonic path and journal every completed state."""
    journal_path = Path(journal_path)
    if journal_path.exists() or journal_path.with_name(
        journal_path.name + ".tmp"
    ).exists():
        raise ValueError(
            "line-search journal already exists; resume is forbidden"
        )
    if tuple(eval_splits) != ALLOWED_EVAL_SPLITS:
        raise ValueError("unsafe line-search evaluation splits")
    if targets[0] != 0.0 or any(
        right <= left for left, right in zip(targets, targets[1:])
    ):
        raise ValueError("line-search targets must start at zero and increase")
    if set(trainer.eval_dataloader_dict) != set(ALLOWED_EVAL_SPLITS):
        raise ValueError("trainer contains a non-allowlisted evaluation split")
    if (prose_scorer is None) != (prose_comparator is None):
        raise ValueError(
            "prose scorer and comparator must be enabled together"
        )

    journal = {
        "schema": "eggroll-es-anchor-alpha-line-search-v1",
        "status": "running",
        "policy": {
            "alpha_order": "zero_then_strictly_increasing",
            "branching": False,
            "resume": False,
            "rollback": False,
            "selection_during_execution": False,
            "ood_qa_max_degradation": 0.0,
            "ood_prose_max_degradation": 0.0,
        },
        "targets": list(targets),
        "seeds": list(seeds),
        "trainer_configuration": {
            "model_name": getattr(trainer, "model_name", None),
            "sigma": getattr(trainer, "sigma", None),
            "population_size": getattr(trainer, "population_size", None),
            "batch_size": getattr(trainer, "batch_size", None),
            "mini_batch_size": getattr(trainer, "mini_batch_size", None),
            "max_tokens": getattr(trainer, "max_tokens", None),
            "global_seed": getattr(trainer, "global_seed", None),
            "min_anchor_cosine": getattr(
                trainer, "min_anchor_cosine", None,
            ),
            "anchor_items_per_step": getattr(
                trainer, "anchor_items_per_step", None,
            ),
        },
        "snapshot": snapshot,
        "coefficient_plan": None,
        "in_progress": {
            "state_index": 0,
            "target_alpha": 0.0,
            "phase": "before_initial_evaluation",
        },
        "states": [],
    }
    atomic_write_json(journal_path, journal)
    baseline_qa = None
    baseline_prose = None
    plan = None
    try:
        # Evaluate the true initial weights before population perturb/restore.
        trainer.eval_step(iteration=0)
        qa = {
            split: summarize_eval_file(path)
            for split, path in _eval_paths(trainer, 0, eval_splits).items()
        }
        baseline_qa = qa
        journal["in_progress"] = {
            "state_index": 0,
            "target_alpha": 0.0,
            "phase": "initial_qa_evaluated",
            "qa": qa,
        }
        atomic_write_json(journal_path, journal)
        prose_evaluation = (
            prose_scorer(trainer, "alpha_0000")
            if prose_scorer is not None else None
        )
        baseline_prose = prose_evaluation
        baseline_prose_gate = (
            prose_comparator(prose_evaluation, prose_evaluation)
            if prose_evaluation is not None else None
        )
        baseline_qa_gate = strict_qa_gate(qa["ood_qa"], qa["ood_qa"])
        state = {
            "state_index": 0,
            "target_alpha": 0.0,
            "alpha_increment": 0.0,
            "eval_iteration": 0,
            "coefficient_sha256": None,
            "qa": qa,
            "ood_qa_gate": baseline_qa_gate,
            "ood_prose": (
                summarize_prose_evaluation(prose_evaluation)
                if prose_evaluation is not None else None
            ),
            "ood_prose_gate": baseline_prose_gate,
            "strict_guards_passed": (
                baseline_qa_gate["passed"]
                and (
                    baseline_prose_gate is None
                    or baseline_prose_gate["passed"]
                )
            ),
        }
        journal["states"].append(state)
        journal["in_progress"] = {
            "state_index": 0,
            "target_alpha": 0.0,
            "phase": "estimating_fixed_coefficient_plan",
        }
        atomic_write_json(journal_path, journal)

        plan = trainer.estimate_step_coefficients(
            0, seeds, input_prompts, target_answers,
        )
        coefficient_sha = plan["coefficient_sha256"]
        _verify_plan(plan, coefficient_sha)
        journal["coefficient_plan"] = {
            "coefficient_sha256": coefficient_sha,
            "journal_path": plan.get("journal_path"),
            "projection": plan.get("projection"),
            "seed_count": len(plan["seeds"]),
        }
        journal["states"][0]["coefficient_sha256"] = coefficient_sha
        journal["in_progress"] = None
        atomic_write_json(journal_path, journal)

        previous_alpha = 0.0
        for state_index, target_alpha in enumerate(targets[1:], 1):
            increment = target_alpha - previous_alpha
            journal["in_progress"] = {
                "state_index": state_index,
                "target_alpha": target_alpha,
                "alpha_increment": increment,
                "phase": "before_weight_increment",
            }
            atomic_write_json(journal_path, journal)
            _verify_plan(plan, coefficient_sha)
            trainer.apply_seed_coefficients(plan, target_alpha)
            _verify_plan(plan, coefficient_sha)
            journal["in_progress"]["phase"] = "weights_incremented"
            atomic_write_json(journal_path, journal)
            trainer.eval_step(iteration=state_index)
            qa = {
                split: summarize_eval_file(path)
                for split, path in _eval_paths(
                    trainer, state_index, eval_splits,
                ).items()
            }
            qa_gate = strict_qa_gate(
                baseline_qa["ood_qa"], qa["ood_qa"],
            )
            journal["in_progress"].update({
                "phase": "qa_evaluated",
                "qa": qa,
                "ood_qa_gate": qa_gate,
            })
            atomic_write_json(journal_path, journal)
            prose_evaluation = (
                prose_scorer(
                    trainer, "alpha_" + format(state_index, "04d")
                )
                if prose_scorer is not None else None
            )
            prose_gate = (
                prose_comparator(baseline_prose, prose_evaluation)
                if prose_evaluation is not None else None
            )
            state = {
                "state_index": state_index,
                "target_alpha": target_alpha,
                "alpha_increment": increment,
                "eval_iteration": state_index,
                "coefficient_sha256": coefficient_sha,
                "qa": qa,
                "ood_qa_gate": qa_gate,
                "ood_prose": (
                    summarize_prose_evaluation(prose_evaluation)
                    if prose_evaluation is not None else None
                ),
                "ood_prose_gate": prose_gate,
                "strict_guards_passed": (
                    qa_gate["passed"]
                    and (prose_gate is None or prose_gate["passed"])
                ),
            }
            journal["states"].append(state)
            journal["in_progress"] = None
            atomic_write_json(journal_path, journal)
            previous_alpha = target_alpha

        journal["status"] = "complete"
        journal["in_progress"] = None
        journal["content_sha256_before_self_field"] = canonical_sha256(
            journal
        )
        atomic_write_json(journal_path, journal)
        return journal
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "completed_state_count": len(journal["states"]),
            "coefficient_plan_estimated": plan is not None,
        }
        atomic_write_json(journal_path, journal)
        raise


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", default=str(ROOT / "models/Qwen3.6-35B-A3B")
    )
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--sigma", type=float, default=0.0003)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-vllm-engines", type=int, default=4)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--use-gpus", default="0,1,2,3")
    parser.add_argument("--eval-splits", default="validation,ood_qa")
    parser.add_argument(
        "--target-alphas", default="0,0.000025,0.00005,0.0001,0.00015",
    )
    parser.add_argument(
        "--anchor-prose-jsonl",
        default=str(ROOT / "data/general_prose_anchor_v1.jsonl"),
    )
    parser.add_argument(
        "--anchor-prose-report",
        default=str(ROOT / "data/general_prose_anchor_v1.report.json"),
    )
    parser.add_argument("--anchor-items-per-step", type=int, default=2)
    parser.add_argument("--anchor-max-input-tokens", type=int, default=512)
    parser.add_argument("--min-anchor-cosine", type=float, default=0.1)
    parser.add_argument(
        "--ood-prose-jsonl", default=str(ROOT / "data/ood_prose_v3.jsonl"),
    )
    parser.add_argument("--ood-prose-max-input-tokens", type=int, default=1024)
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument(
        "--output-directory",
        default=str(ROOT / "experiments/eggroll_es_hpo/runs"),
    )
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--logging", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    return parser.parse_args()


def main():
    args = parse_args()
    targets = parse_target_alphas(args.target_alphas)
    eval_splits = validate_eval_splits(args.eval_splits)
    if args.n_vllm_engines * args.n_gpu_per_vllm_engine != len(
        args.use_gpus.split(",")
    ):
        raise ValueError("engine allocation must consume every selected GPU")
    if not math.isfinite(args.sigma) or args.sigma <= 0.0:
        raise ValueError("sigma must be finite and positive")
    if args.population_size < 2:
        raise ValueError("population size must be at least two")
    if args.batch_size <= 0 or args.mini_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    if args.anchor_items_per_step <= 0:
        raise ValueError("anchor items per step must be positive")
    if args.anchor_max_input_tokens < 2:
        raise ValueError("anchor max input tokens must be at least two")
    if args.ood_prose_max_input_tokens < 2:
        raise ValueError("OOD prose max input tokens must be at least two")
    if not 0.0 <= args.min_anchor_cosine < 1.0:
        raise ValueError("minimum anchor cosine must be in [0, 1)")
    run_dir = Path(args.output_directory) / args.experiment_name
    journal_path = run_dir / JOURNAL_NAME
    if run_dir.exists():
        raise ValueError(
            "line-search output already exists; resume is forbidden"
        )

    base.set_seed(args.seed)
    train_dict = load_from_disk(args.train_dataset)
    if list(train_dict) != ["train"]:
        raise ValueError("training artifact must contain exactly train")
    train_dataset = train_dict["train"]
    train_loader = base.build_train_loader(
        train_dataset, args.batch_size, args.seed,
    )
    batch_questions, batch_answers = next(iter(train_loader))
    input_prompts = [
        base.specialist_template(item) for item in batch_questions
    ]

    eval_datasets = load_allowlisted_eval_datasets(
        args.eval_dataset, eval_splits,
    )
    eval_loaders = {
        split: DataLoader(
            eval_datasets[split], batch_size=args.mini_batch_size,
            collate_fn=base.specialist_collate, shuffle=False,
        )
        for split in eval_splits
    }
    anchor_dataset = anchor.load_anchor_prose(
        args.anchor_prose_jsonl, args.anchor_prose_report,
    )
    ood_prose = base.load_ood_prose(args.ood_prose_jsonl)
    snapshot = build_snapshot(
        train_dataset, eval_datasets, anchor_dataset,
        batch_questions, batch_answers,
    )
    snapshot["recipe"] = {
        "model_name": str(Path(args.model_name).resolve()),
        "checkpoint": (
            {
                "path": str(Path(args.checkpoint).resolve()),
                "sha256": file_sha256(args.checkpoint),
            }
            if args.checkpoint else None
        ),
        "sigma": args.sigma,
        "population_size": args.population_size,
        "batch_size": args.batch_size,
        "mini_batch_size": args.mini_batch_size,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "min_anchor_cosine": args.min_anchor_cosine,
        "anchor_items_per_step": args.anchor_items_per_step,
        "target_alphas": targets,
    }
    seeds = np.random.default_rng(
        seed=base.exact_step_seed(args.seed, 0)
    ).integers(
        0, 2**30, size=args.population_size, dtype=np.int64,
    ).tolist()

    trainer = None
    try:
        trainer_class = anchor.load_trainer()
        trainer = trainer_class(
            model_name=args.model_name,
            checkpoint=args.checkpoint,
            sigma=args.sigma,
            alpha=targets[-1],
            population_size=args.population_size,
            reward_shaping="z-scores",
            num_iterations=1,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            mini_batch_size=args.mini_batch_size,
            reward_function=base.specialist_reward,
            template_function=base.specialist_template,
            train_dataloader=train_loader,
            eval_dataloader_dict=eval_loaders,
            eval_freq=1,
            n_vllm_engines=args.n_vllm_engines,
            n_gpu_per_vllm_engine=args.n_gpu_per_vllm_engine,
            logging=args.logging,
            global_seed=args.seed,
            use_gpus=args.use_gpus,
            experiment_name=args.experiment_name,
            wandb_project=args.wandb_project,
            save_best_models=False,
            reward_function_timeout=args.reward_function_timeout,
            output_directory=args.output_directory,
        )
        trainer.configure_anchor(
            anchor_dataset,
            items_per_step=args.anchor_items_per_step,
            max_input_tokens=args.anchor_max_input_tokens,
            min_anchor_cosine=args.min_anchor_cosine,
        )

        def prose_scorer(current_trainer, label):
            return base.score_ood_prose(
                current_trainer, ood_prose, label,
                args.ood_prose_max_input_tokens,
            )

        def prose_comparator(baseline, candidate):
            return base.compare_ood_prose(
                baseline, candidate, max_degradation=0.0,
            )

        execute_line_search(
            trainer,
            targets=targets,
            seeds=seeds,
            input_prompts=input_prompts,
            target_answers=batch_answers,
            snapshot=snapshot,
            journal_path=journal_path,
            eval_splits=eval_splits,
            prose_scorer=prose_scorer,
            prose_comparator=prose_comparator,
        )
    finally:
        if trainer is not None:
            base.close_trainer(trainer)


if __name__ == "__main__":
    main()
