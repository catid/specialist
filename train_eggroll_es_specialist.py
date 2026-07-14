#!/usr/bin/env python3
"""Run upstream ES-at-Scale full-parameter training on specialist QA."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from es_train_acc import answer_score


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "es-at-scale"
COMPAT = ROOT / "eggroll_es_compat"


def specialist_collate(batch):
    return ([item["question"] for item in batch],
            [item["answer"] for item in batch])


def build_train_loader(dataset, batch_size, seed):
    """Build a shuffled loader whose order is isolated from global RNG use."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=specialist_collate,
        shuffle=True, generator=generator,
    )


def specialist_template(question):
    """Qwen prompt with thinking disabled, matching its bundled template."""
    return (
        "<|im_start|>system\n"
        "Answer the question briefly and factually. Return only the answer."
        "<|im_end|>\n<|im_start|>user\n"
        f"{question}<|im_end|>\n<|im_start|>assistant\n"
        "<think>\n\n</think>\n\n"
    )


def extract_answer(response):
    response = re.sub(r"<think>.*?(?:</think>|$)", " ", response,
                      flags=re.DOTALL)
    response = response.replace("<|im_end|>", " ").strip()
    return next((line.strip() for line in response.splitlines()
                 if line.strip()), "")


def specialist_reward(response, target):
    prediction = extract_answer(response)
    score = answer_score(prediction, target)
    if score == 1.0:
        label = "exact"
    elif score > 0.0:
        label = "partial"
    else:
        label = "incorrect"
    return label, score


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def exact_step_seed(global_seed, iteration):
    """Derive the per-step seed without treating the valid seed 0 as absent."""
    base_seed = 42 if global_seed is None else global_seed
    return base_seed + iteration


def load_ood_prose(path):
    """Load and validate a frozen OOD prose JSONL artifact."""
    path = Path(path)
    raw = path.read_bytes()
    rows = []
    seen = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid OOD prose JSON on line {line_number}: {error}"
            ) from error
        item_id = row.get("item_id")
        text = row.get("text")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(
                f"OOD prose line {line_number} has no non-empty item_id"
            )
        if item_id in seen:
            raise ValueError(f"duplicate OOD prose item_id: {item_id}")
        if row.get("split") != "ood_prose":
            raise ValueError(
                f"OOD prose item {item_id} has split {row.get('split')!r}"
            )
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"OOD prose item {item_id} has no text")
        seen.add(item_id)
        rows.append(row)
    if not rows:
        raise ValueError("OOD prose artifact is empty")
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": rows,
    }


def prepare_ood_prose_items(rows, tokenizer, max_input_tokens):
    """Tokenize prose exactly once and attach stable alignment hashes."""
    if max_input_tokens < 2:
        raise ValueError("OOD prose max input tokens must be at least 2")
    prepared = []
    for row in rows:
        token_ids = list(tokenizer.encode(
            row["text"], add_special_tokens=False,
        ))
        if len(token_ids) < 2:
            raise ValueError(
                f"OOD prose item {row['item_id']} has fewer than two tokens"
            )
        if len(token_ids) > max_input_tokens:
            raise ValueError(
                f"OOD prose item {row['item_id']} has {len(token_ids)} "
                f"tokens, above the explicit cap {max_input_tokens}"
            )
        token_bytes = json.dumps(
            token_ids, separators=(",", ":"), ensure_ascii=True,
        ).encode("ascii")
        prepared.append({
            "item_id": row["item_id"],
            "source": row.get("source"),
            "title": row.get("title"),
            "url": row.get("url"),
            "normalized_source_url": row.get("normalized_source_url"),
            "text_sha256": hashlib.sha256(
                row["text"].encode("utf-8")
            ).hexdigest(),
            "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
            "prompt_token_ids": token_ids,
        })
    return prepared


def dispatch_ood_prose(engines, items, sampling_params, resolve):
    """Score round-robin shards concurrently, restoring source-file order."""
    if not engines:
        raise ValueError("OOD prose scoring requires at least one engine")
    partitions = [[] for _ in engines]
    for position, item in enumerate(items):
        partitions[position % len(engines)].append((position, item))

    handles = []
    assignments = []
    for engine, partition in zip(engines, partitions):
        if not partition:
            continue
        prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for _, item in partition
        ]
        handles.append(engine.generate.remote(
            prompts, sampling_params, use_tqdm=False,
        ))
        assignments.append([position for position, _ in partition])

    batches = resolve(handles)
    if len(batches) != len(assignments):
        raise ValueError("OOD prose engine result batch count changed")
    ordered = [None] * len(items)
    for positions, batch in zip(assignments, batches):
        if len(batch) != len(positions):
            raise ValueError("OOD prose engine changed its request count")
        for position, output in zip(positions, batch):
            ordered[position] = output
    if any(output is None for output in ordered):
        raise ValueError("OOD prose engine omitted a request")
    return ordered


def prompt_token_logprobs(output, expected_token_ids):
    """Extract the selected token logprob at each teacher-forced position."""
    returned_ids = list(output.prompt_token_ids or [])
    if returned_ids != expected_token_ids:
        raise ValueError("vLLM returned different OOD prose prompt token IDs")
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None:
        raise ValueError("vLLM did not return requested prompt logprobs")
    if len(prompt_logprobs) != len(expected_token_ids):
        raise ValueError("vLLM prompt logprob/token lengths differ")

    # A decoder-only LM cannot score the first token without left context.
    selected_logprobs = []
    for position, token_id in enumerate(expected_token_ids[1:], 1):
        candidates = prompt_logprobs[position]
        if candidates is None or token_id not in candidates:
            raise ValueError(
                f"vLLM omitted selected token {token_id} at position "
                f"{position}"
            )
        selected = candidates[token_id]
        value = (
            selected.logprob
            if hasattr(selected, "logprob")
            else selected["logprob"]
        )
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("vLLM returned a non-finite prompt logprob")
        selected_logprobs.append(value)
    return selected_logprobs


def summarize_ood_prose(items, outputs):
    """Build aligned per-item and token-weighted corpus metrics."""
    if len(items) != len(outputs):
        raise ValueError("OOD prose item/output counts differ")
    results = []
    for item, output in zip(items, outputs):
        values = prompt_token_logprobs(
            output, item["prompt_token_ids"],
        )
        item_sum = math.fsum(values)
        result = {
            key: item[key]
            for key in (
                "item_id", "source", "title", "url",
                "normalized_source_url", "text_sha256",
                "token_ids_sha256",
            )
        }
        result.update({
            "prompt_token_count": len(item["prompt_token_ids"]),
            "scored_token_count": len(values),
            "sum_token_logprob": item_sum,
            "mean_token_logprob": item_sum / len(values),
        })
        results.append(result)

    scored_token_count = sum(
        row["scored_token_count"] for row in results
    )
    sum_token_logprob = math.fsum(
        row["sum_token_logprob"] for row in results
    )
    return {
        "item_count": len(results),
        "scored_token_count": scored_token_count,
        "sum_token_logprob": sum_token_logprob,
        "mean_token_logprob": sum_token_logprob / scored_token_count,
        "items": results,
    }


def compare_ood_prose(baseline, final, max_degradation):
    """Require aligned items and apply a higher-is-better logprob gate."""
    if not math.isfinite(max_degradation) or max_degradation < 0.0:
        raise ValueError("maximum OOD prose degradation must be non-negative")
    alignment_fields = (
        "item_id", "text_sha256", "token_ids_sha256",
        "prompt_token_count", "scored_token_count",
    )
    baseline_alignment = [
        tuple(row[field] for field in alignment_fields)
        for row in baseline["items"]
    ]
    final_alignment = [
        tuple(row[field] for field in alignment_fields)
        for row in final["items"]
    ]
    if baseline_alignment != final_alignment:
        raise ValueError("baseline/final OOD prose items are not aligned")
    baseline_mean = float(baseline["mean_token_logprob"])
    final_mean = float(final["mean_token_logprob"])
    delta = final_mean - baseline_mean
    return {
        "metric": "mean_token_logprob",
        "higher_is_better": True,
        "baseline": baseline_mean,
        "final": final_mean,
        "delta": delta,
        "max_degradation": max_degradation,
        "passed": delta >= -max_degradation,
    }


def score_ood_prose(trainer, dataset, label, max_input_tokens):
    """Score frozen prose on all live vLLM engines and save item results."""
    import ray
    from vllm import SamplingParams

    items = prepare_ood_prose_items(
        dataset["rows"], trainer.tokenizer, max_input_tokens,
    )
    sampling_params = SamplingParams(
        n=1,
        seed=42 if trainer.global_seed is None else trainer.global_seed,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
    )
    outputs = dispatch_ood_prose(
        trainer.engines, items, sampling_params, ray.get,
    )
    evaluation = summarize_ood_prose(items, outputs)
    result_path = (
        Path(trainer.logging_dir) / "eval-output"
        / f"ood_prose_{label}.json"
    )
    result_path.write_text(
        json.dumps(evaluation["items"], indent=2, sort_keys=True) + "\n"
    )
    evaluation["results_path"] = str(result_path)
    return evaluation


def load_trainer():
    """Load upstream with the minimum Qwen3.6/vLLM compatibility layer."""
    pythonpath = [str(COMPAT), str(UPSTREAM)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    sys.path.insert(0, str(UPSTREAM))
    import vllm.utils
    from vllm.utils.network_utils import get_ip, get_open_port

    # ES-at-Scale pins vLLM 0.11, where these were exported from vllm.utils.
    # Qwen3.6 needs vLLM 0.25; aliasing the relocated helpers is the only
    # trainer compatibility shim.
    vllm.utils.get_ip = get_ip
    vllm.utils.get_open_port = get_open_port
    import ray
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from es_at_scale.trainer.es_trainer import (
        ESNcclLLM,
        EvolutionStrategiesTrainer,
    )

    class Qwen36EvolutionStrategiesTrainer(EvolutionStrategiesTrainer):
        def launch_engines(self, num_engines=4, n_gpu_per_vllm_engine=1,
                           model_name="Qwen/Qwen2.5-Math-1.5B",
                           precision="bfloat16"):
            pgs = [
                placement_group(
                    [{"GPU": 1, "CPU": 0}] * n_gpu_per_vllm_engine,
                    strategy="PACK", lifetime="detached",
                )
                for _ in range(num_engines)
            ]
            ray.get([pg.ready() for pg in pgs])
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
                for pg in pgs
            ]
            engine_args = {
                "model": model_name,
                "tensor_parallel_size": n_gpu_per_vllm_engine,
                "worker_extension_cls": (
                    "es_at_scale.utils.worker_extension.WorkerExtension"
                ),
                "dtype": precision,
                "enable_prefix_caching": False,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.82,
                "max_model_len": 2048,
                "limit_mm_per_prompt": {"image": 0, "video": 0},
                "mm_processor_cache_gb": 0,
                "skip_mm_profiling": True,
                # FlashInfer's auto-selected unquantized MoE backend JIT-builds
                # a large CUTLASS extension on first use. Triton is already
                # supported by vLLM for this model and avoids that CPU-only
                # startup compile, so the four accelerators begin work sooner.
                "moe_backend": "triton",
            }
            if n_gpu_per_vllm_engine > 1:
                engine_args["distributed_executor_backend"] = "ray"
            actor_gpus = 1 if n_gpu_per_vllm_engine == 1 else 0
            engines = [
                ray.remote(
                    num_cpus=0, num_gpus=actor_gpus,
                    scheduling_strategy=strategy,
                )(ESNcclLLM).remote(**engine_args)
                for strategy in strategies
            ]
            return engines, pgs

    return Qwen36EvolutionStrategiesTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Faithful ES-at-Scale specialist fine-tuning adapter"
    )
    parser.add_argument(
        "--model-name", default=str(ROOT / "models/Qwen3.6-35B-A3B")
    )
    parser.add_argument(
        "--train-dataset",
        default=str(ROOT / "data/eggroll_es_specialist/train"),
    )
    parser.add_argument(
        "--eval-dataset", default=str(ROOT / "data/eggroll_es_specialist/eval")
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--population-size", type=int, default=30)
    parser.add_argument("--n-iterations", type=int, default=500)
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--mini-batch-size", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-vllm-engines", type=int, default=4)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--use-gpus", default="0,1,2,3")
    parser.add_argument(
        "--eval-splits", default="all",
        help="Comma-separated evaluation splits, or 'all'",
    )
    parser.add_argument(
        "--exact-train-steps", type=int,
        help="Run exactly this many upstream ES updates (HPO-friendly mode)",
    )
    parser.add_argument(
        "--skip-baseline-eval", action="store_true",
        help=(
            "In exact-step mode, skip only the generated-QA baseline; an "
            "enabled OOD prose gate still scores both weight states"
        ),
    )
    parser.add_argument(
        "--save-final-checkpoint", action="store_true",
        help="In exact-step mode, serialize the final model weights",
    )
    parser.add_argument(
        "--ood-prose-jsonl",
        help=(
            "Opt-in frozen prose JSONL for baseline/final mean-token-logprob "
            "gating in exact-step mode"
        ),
    )
    parser.add_argument(
        "--ood-prose-max-input-tokens", type=int, default=1024,
        help=(
            "Reject, rather than truncate, OOD prose above this token count"
        ),
    )
    parser.add_argument(
        "--max-ood-prose-degradation", type=float, default=0.0,
        help=(
            "Allowed decrease in final mean token logprob versus baseline"
        ),
    )
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument(
        "--output-directory", default=str(ROOT / "experiments/eggroll_es")
    )
    parser.add_argument(
        "--experiment-name", default="qwen36-specialist-upstream"
    )
    parser.add_argument("--logging", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    parser.add_argument("--save-best-models", action="store_true")
    return parser.parse_args()


def eval_metrics(logging_dir, iteration, split_names):
    metrics = {}
    for name in split_names:
        path = (
            Path(logging_dir) / "eval-output"
            / f"model_eval_task{name}_iteration{iteration + 1}.json"
        )
        rows = json.loads(path.read_text())
        rewards = [float(row["reward"]) for row in rows]
        metrics[name] = sum(rewards) / len(rewards) if rewards else 0.0
    return metrics


def close_trainer(trainer):
    trainer.cleanup()
    trainer.mp_pool.close()
    trainer.mp_pool.join()
    if trainer.logging == "wandb":
        try:
            trainer.wandb.finish()
        except Exception:
            pass


def run_exact_steps(trainer, steps, skip_baseline_eval=False,
                    save_final_checkpoint=False, ood_prose=None,
                    ood_prose_max_input_tokens=1024,
                    max_ood_prose_degradation=0.0):
    """Run an exact number of otherwise unchanged upstream ES updates."""
    if steps < 0:
        raise ValueError("exact train steps must be non-negative")
    evaluated = []
    ood_prose_evaluations = {}
    try:
        if ood_prose is not None:
            ood_prose_evaluations["baseline"] = score_ood_prose(
                trainer, ood_prose, "baseline",
                ood_prose_max_input_tokens,
            )
        if not skip_baseline_eval:
            trainer.eval_step(iteration=0)
            evaluated.append(("baseline", 0))

        train_iterator = iter(trainer.train_dataloader)
        for iteration in range(steps):
            try:
                input_text, target_text = next(train_iterator)
            except StopIteration:
                train_iterator = iter(trainer.train_dataloader)
                input_text, target_text = next(train_iterator)
            prompts = [trainer.template(item) for item in input_text]
            loop_rng = np.random.default_rng(
                seed=exact_step_seed(trainer.global_seed, iteration)
            )
            seeds = loop_rng.integers(
                0, 2**30, size=trainer.population_size, dtype=np.int64
            ).tolist()
            started = time.time()
            print(f"\n\n=== Exact ES step {iteration + 1}/{steps} ===")
            trainer.train_step(
                iteration=iteration, seeds=seeds,
                input_text=prompts, target_text=target_text,
            )
            elapsed_seconds = round(time.time() - started, 3)
            print(
                f"=== Exact ES step {iteration + 1}/{steps} finished in "
                f"{elapsed_seconds}s ==="
            )

        if steps > 0 or skip_baseline_eval:
            trainer.eval_step(iteration=steps)
            evaluated.append(("final", steps))

        if ood_prose is not None:
            ood_prose_evaluations["final"] = score_ood_prose(
                trainer, ood_prose, "final",
                ood_prose_max_input_tokens,
            )

        checkpoint = None
        if save_final_checkpoint:
            import ray

            checkpoint_dir = (
                Path(trainer.logging_dir)
                / f"checkpoint-es_exact_steps_{steps}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = checkpoint_dir / "pytorch_model.pth"
            ray.get(
                trainer.engines[0].collective_rpc.remote(
                    "save_self_weights_to_disk", args=(str(checkpoint),)
                )
            )

        summary = {
            "schema": "eggroll-es-exact-run-v1",
            "model": trainer.model_name,
            "steps": steps,
            "sigma": trainer.sigma,
            "alpha": trainer.alpha,
            "population_size": trainer.population_size,
            "batch_size": trainer.batch_size,
            "mini_batch_size": trainer.mini_batch_size,
            "max_tokens": trainer.max_tokens,
            "seed": trainer.global_seed,
            "checkpoint": str(checkpoint) if checkpoint else None,
            "evaluations": {
                label: eval_metrics(
                    trainer.logging_dir, iteration,
                    trainer.eval_dataloader_dict.keys(),
                )
                for label, iteration in evaluated
            },
        }
        if ood_prose is not None:
            summary["ood_prose"] = {
                "schema": "eggroll-es-ood-prose-logprob-v1",
                "dataset": {
                    "path": ood_prose["path"],
                    "sha256": ood_prose["sha256"],
                    "item_count": len(ood_prose["rows"]),
                },
                "scoring": {
                    "tokenizer": trainer.model_name,
                    "add_special_tokens": False,
                    "first_token_policy": "excluded_no_left_context",
                    "aggregation": "token_weighted_corpus_mean",
                    "max_input_tokens": ood_prose_max_input_tokens,
                    "prompt_logprobs": 1,
                    "temperature": 0.0,
                    "generation_tokens": 1,
                    "seed": (
                        42 if trainer.global_seed is None
                        else trainer.global_seed
                    ),
                },
                "evaluations": ood_prose_evaluations,
                "gate": compare_ood_prose(
                    ood_prose_evaluations["baseline"],
                    ood_prose_evaluations["final"],
                    max_ood_prose_degradation,
                ),
            }
        summary_path = Path(trainer.logging_dir) / "run_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    finally:
        close_trainer(trainer)


def main():
    args = parse_args()
    if args.alpha == -1.0:
        args.alpha = args.sigma / 2.0
    if args.n_vllm_engines * args.n_gpu_per_vllm_engine != len(
            args.use_gpus.split(",")):
        raise ValueError(
            "engine GPU allocation must consume every selected GPU"
        )
    if args.ood_prose_jsonl and args.exact_train_steps is None:
        raise ValueError(
            "--ood-prose-jsonl currently requires --exact-train-steps"
        )
    if args.ood_prose_max_input_tokens < 2:
        raise ValueError("--ood-prose-max-input-tokens must be at least 2")
    if (
        not math.isfinite(args.max_ood_prose_degradation)
        or args.max_ood_prose_degradation < 0.0
    ):
        raise ValueError(
            "--max-ood-prose-degradation must be non-negative"
        )
    ood_prose = (
        load_ood_prose(args.ood_prose_jsonl)
        if args.ood_prose_jsonl else None
    )
    set_seed(args.seed)

    train_dict = load_from_disk(args.train_dataset)
    if list(train_dict) != ["train"]:
        raise ValueError(
            "training artifact must contain exactly the train split"
        )
    # Keep the shuffled batch order independent of Ray/vLLM startup.  Those
    # libraries touch the process-global Torch RNG, so relying on it here made
    # nominally identical HPO trials see different examples.
    train_loader = build_train_loader(
        train_dict["train"], batch_size=args.batch_size, seed=args.seed,
    )
    eval_dict = load_from_disk(args.eval_dataset)
    if args.eval_splits != "all":
        requested_splits = [
            split.strip() for split in args.eval_splits.split(",")
            if split.strip()
        ]
        missing = sorted(set(requested_splits) - set(eval_dict))
        if missing:
            raise ValueError(f"unknown evaluation splits: {missing}")
        eval_dict = {
            split: eval_dict[split] for split in requested_splits
        }
    eval_loaders = {
        name: DataLoader(dataset, batch_size=args.mini_batch_size,
                         collate_fn=specialist_collate, shuffle=False)
        for name, dataset in eval_dict.items()
    }

    trainer_class = load_trainer()
    trainer = trainer_class(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        sigma=args.sigma,
        alpha=args.alpha,
        population_size=args.population_size,
        reward_shaping="z-scores",
        num_iterations=args.n_iterations,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        reward_function=specialist_reward,
        template_function=specialist_template,
        train_dataloader=train_loader,
        eval_dataloader_dict=eval_loaders,
        eval_freq=args.eval_freq,
        n_vllm_engines=args.n_vllm_engines,
        n_gpu_per_vllm_engine=args.n_gpu_per_vllm_engine,
        logging=args.logging,
        global_seed=args.seed,
        use_gpus=args.use_gpus,
        experiment_name=args.experiment_name,
        wandb_project=args.wandb_project,
        save_best_models=args.save_best_models,
        reward_function_timeout=args.reward_function_timeout,
        output_directory=args.output_directory,
    )
    if args.exact_train_steps is None:
        trainer.fit()
    else:
        run_exact_steps(
            trainer, args.exact_train_steps,
            skip_baseline_eval=args.skip_baseline_eval,
            save_final_checkpoint=args.save_final_checkpoint,
            ood_prose=ood_prose,
            ood_prose_max_input_tokens=(
                args.ood_prose_max_input_tokens
            ),
            max_ood_prose_degradation=(
                args.max_ood_prose_degradation
            ),
        )


if __name__ == "__main__":
    main()
