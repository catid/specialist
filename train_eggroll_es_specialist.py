#!/usr/bin/env python3
"""Run upstream ES-at-Scale full-parameter training on specialist QA."""

import argparse
import os
import random
import re
import sys
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


def specialist_template(question):
    """Qwen chat prompt with thinking disabled, matching its bundled template."""
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
        "--train-dataset", default=str(ROOT / "data/eggroll_es_specialist/train")
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
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument(
        "--output-directory", default=str(ROOT / "experiments/eggroll_es")
    )
    parser.add_argument("--experiment-name", default="qwen36-specialist-upstream")
    parser.add_argument("--logging", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    parser.add_argument("--save-best-models", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.alpha == -1.0:
        args.alpha = args.sigma / 2.0
    if args.n_vllm_engines * args.n_gpu_per_vllm_engine != len(
            args.use_gpus.split(",")):
        raise ValueError("engine GPU allocation must consume every selected GPU")
    set_seed(args.seed)

    train_dict = load_from_disk(args.train_dataset)
    if list(train_dict) != ["train"]:
        raise ValueError("training artifact must contain exactly the train split")
    train_loader = DataLoader(
        train_dict["train"], batch_size=args.batch_size,
        collate_fn=specialist_collate, shuffle=True,
    )
    eval_dict = load_from_disk(args.eval_dataset)
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
    trainer.fit()


if __name__ == "__main__":
    main()
