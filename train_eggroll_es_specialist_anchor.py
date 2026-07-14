#!/usr/bin/env python3
"""Anchored EGGROLL-ES specialist training experiment family.

This adapter imports, but never edits, the faithful base trainer.  It adds a
train-only teacher-forced prose objective evaluated under the exact same
population perturbations as domain reward.  Domain coefficients are projected
in the shared seed-noise basis and then passed directly to upstream's existing
weight update without a second normalization.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

import train_eggroll_es_specialist as base
from build_general_prose_anchor import text_sha256 as normalized_text_sha256
from eggroll_es_anchor import project_anchor_safe_coefficients


ROOT = Path(__file__).resolve().parent
DEFAULT_ANCHOR = ROOT / "data/general_prose_anchor_v1.jsonl"
DEFAULT_ANCHOR_REPORT = ROOT / "data/general_prose_anchor_v1.report.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_anchor_prose(path, report_path=None):
    """Load a train-only anchor and pin its exact build report."""
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = []
    item_ids = set()
    document_ids = set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid anchor JSON on line {line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"anchor line {line_number} is not an object")
        item_id = row.get("item_id")
        document_id = row.get("document_id")
        text = row.get("text")
        if row.get("split") != "anchor_prose":
            raise ValueError(f"anchor item {item_id!r} has the wrong split")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"anchor line {line_number} has no item_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"anchor item {item_id} has no document_id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"anchor item {item_id} has no text")
        if item_id in item_ids:
            raise ValueError(f"duplicate anchor item_id: {item_id}")
        if document_id in document_ids:
            raise ValueError(f"duplicate anchor document_id: {document_id}")
        recorded_text_hash = row.get("text_sha256")
        if not isinstance(recorded_text_hash, str) or not recorded_text_hash:
            raise ValueError(f"anchor item {item_id} has no text_sha256")
        if recorded_text_hash != normalized_text_sha256(text):
            raise ValueError(f"anchor item {item_id} text hash does not match")
        item_ids.add(item_id)
        document_ids.add(document_id)
        rows.append(row)
    if not rows:
        raise ValueError("anchor prose artifact is empty")

    report = None
    if report_path is not None:
        report_path = Path(report_path)
        report_raw = report_path.read_bytes()
        report = json.loads(report_raw)
        if not isinstance(report, dict):
            raise ValueError("anchor report is not an object")
        if report.get("schema") != "general-prose-anchor-build-v1":
            raise ValueError("anchor report has the wrong schema")
        if report.get("output_sha256") != digest:
            raise ValueError("anchor report does not pin the anchor bytes")
        if report.get("output_rows") != len(rows):
            raise ValueError("anchor report row count does not match")
        report = {
            "path": str(report_path.resolve()),
            "sha256": hashlib.sha256(report_raw).hexdigest(),
            "schema": report["schema"],
            "protected_artifact_count": len(
                report.get("protected_artifacts", [])
            ),
        }
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "rows": rows,
        "report": report,
    }


def prepare_anchor_items(rows, tokenizer, max_input_tokens):
    """Tokenize all anchor documents once without truncation."""
    if max_input_tokens < 2:
        raise ValueError("anchor max input tokens must be at least two")
    prepared = []
    for row in rows:
        token_ids = list(tokenizer.encode(
            row["text"], add_special_tokens=False,
        ))
        if len(token_ids) < 2:
            raise ValueError(f"anchor item {row['item_id']} is too short")
        if len(token_ids) > max_input_tokens:
            raise ValueError(
                f"anchor item {row['item_id']} has {len(token_ids)} tokens, "
                f"above the explicit cap {max_input_tokens}"
            )
        token_bytes = json.dumps(
            token_ids, ensure_ascii=True, separators=(",", ":"),
        ).encode("ascii")
        prepared.append({
            "document_id": row["document_id"],
            "item_id": row["item_id"],
            "prompt_token_ids": token_ids,
            "text_sha256": row["text_sha256"],
            "token_ids_sha256": hashlib.sha256(token_bytes).hexdigest(),
        })
    return prepared


def select_anchor_items(items, iteration, count, global_seed):
    """Select a deterministic rotating microbatch without global RNG use."""
    if count <= 0:
        raise ValueError("anchor items per step must be positive")
    if count > len(items):
        raise ValueError("anchor items per step exceeds anchor dataset size")
    seed = 42 if global_seed is None else int(global_seed)
    ordered = sorted(items, key=lambda item: hashlib.sha256(
        f"{seed}\0{item['item_id']}".encode("utf-8")
    ).digest())
    start = (int(iteration) * count) % len(ordered)
    return [
        ordered[(start + offset) % len(ordered)]
        for offset in range(count)
    ]


def score_anchor_outputs(items, outputs):
    """Return a token-weighted selected-token mean log probability."""
    if len(items) != len(outputs):
        raise ValueError("anchor item/output counts differ")
    values = []
    for item, output in zip(items, outputs):
        values.extend(base.prompt_token_logprobs(
            output, item["prompt_token_ids"],
        ))
    if not values:
        raise ValueError("anchor scorer received no predicted tokens")
    score = math.fsum(values) / len(values)
    if not math.isfinite(score):
        raise ValueError("anchor score is not finite")
    return score


def coefficient_sha256(seeds, coefficients):
    payload = json.dumps(
        {"seeds": list(seeds), "coefficients": list(coefficients)},
        ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def dispatch_eval_batch(engines, prompts, sampling_params, resolve):
    """Shard a deterministic eval batch and restore its original order."""
    if not engines:
        raise ValueError("evaluation requires at least one engine")
    partitions = [[] for _ in engines]
    for position, prompt in enumerate(prompts):
        partitions[position % len(engines)].append((position, prompt))
    handles = []
    assignments = []
    for engine, partition in zip(engines, partitions):
        if not partition:
            continue
        handles.append(engine.generate.remote(
            [prompt for _, prompt in partition], sampling_params,
            use_tqdm=False,
        ))
        assignments.append([position for position, _ in partition])
    batches = resolve(handles)
    if len(batches) != len(assignments):
        raise ValueError("evaluation engine result batch count changed")
    ordered = [None] * len(prompts)
    for positions, batch in zip(assignments, batches):
        if len(positions) != len(batch):
            raise ValueError("evaluation engine changed request count")
        for position, output in zip(positions, batch):
            ordered[position] = output
    if any(output is None for output in ordered):
        raise ValueError("evaluation engine omitted a request")
    return ordered


class AnchoredStepMixin:
    """Add same-perturbation prose scoring and reusable ES coefficients."""

    def configure_anchor(
        self,
        dataset,
        *,
        items_per_step,
        max_input_tokens,
        min_anchor_cosine,
    ):
        if not 0.0 <= min_anchor_cosine < 1.0:
            raise ValueError("minimum anchor cosine must be in [0, 1)")
        if items_per_step <= 0:
            raise ValueError("anchor items per step must be positive")
        self.anchor_dataset = dataset
        self.anchor_items = prepare_anchor_items(
            dataset["rows"], self.tokenizer, max_input_tokens,
        )
        if items_per_step > len(self.anchor_items):
            raise ValueError("anchor items per step exceeds dataset size")
        self.anchor_items_per_step = int(items_per_step)
        self.anchor_max_input_tokens = int(max_input_tokens)
        self.min_anchor_cosine = float(min_anchor_cosine)
        self.anchor_step_plans = []
        self._latest_anchor_plan = None

    def _resolve(self, handles):
        import ray
        return ray.get(handles)

    def _sampling_params(self, **kwargs):
        from vllm import SamplingParams
        return SamplingParams(**kwargs)

    def _evaluate_population_with_anchor(
        self,
        seeds,
        input_batch,
        target_batch,
        domain_sampling_params,
        anchor_items,
        iteration,
    ):
        """Score both objectives before restoring each identical noise seed."""
        seeds_perf = {}
        anchor_scores = {}
        results = []
        anchor_sampling = self._sampling_params(
            n=1,
            seed=(42 if self.global_seed is None else self.global_seed)
            + iteration,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1,
            prompt_logprobs=1,
            detokenize=False,
        )
        anchor_prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in anchor_items
        ]

        for start in range(0, len(seeds), self.n_vllm_engines):
            engine_batch = seeds[start:start + self.n_vllm_engines]
            perturb = [
                self.engines[engine_index].collective_rpc.remote(
                    "perturb_self_weights",
                    args=(int(seed), self.sigma, False),
                )
                for engine_index, seed in enumerate(engine_batch)
            ]
            self._resolve(perturb)
            try:
                handles = []
                for engine_index, _ in enumerate(engine_batch):
                    if anchor_items:
                        prompts = list(input_batch) + anchor_prompts
                        params = (
                            [domain_sampling_params] * len(input_batch)
                            + [anchor_sampling] * len(anchor_items)
                        )
                    else:
                        prompts = input_batch
                        params = domain_sampling_params
                    handles.append(
                        self.engines[engine_index].generate.remote(
                            prompts, params, use_tqdm=False,
                        )
                    )
                outputs_per_engine = self._resolve(handles)
            finally:
                restore = [
                    self.engines[engine_index].collective_rpc.remote(
                        "restore_self_weights", args=(int(seed), self.sigma),
                    )
                    for engine_index, seed in enumerate(engine_batch)
                ]
                self._resolve(restore)

            for engine_index, seed in enumerate(engine_batch):
                outputs = outputs_per_engine[engine_index]
                expected_count = len(input_batch) + len(anchor_items)
                if len(outputs) != expected_count:
                    raise ValueError(
                        "population engine changed combined request count"
                    )
                domain_outputs = outputs[:len(input_batch)]
                metrics = self._postprocess_outputs(
                    domain_outputs, target_batch,
                )
                seeds_perf[int(seed)] = metrics
                results.append({
                    "seed": int(seed),
                    "avg_reward": metrics["avg_reward"],
                })
                if anchor_items:
                    anchor_scores[int(seed)] = score_anchor_outputs(
                        anchor_items, outputs[len(input_batch):],
                    )
        return seeds_perf, anchor_scores, results

    def _persist_anchor_plan(self, plan):
        path = (
            Path(self.logging_dir) / "anchor-plan-iteration-"
            f"{plan['iteration'] + 1}.json"
        )
        plan["journal_path"] = str(path)
        path.write_text(json.dumps(
            plan, ensure_ascii=False, indent=2, sort_keys=True,
        ) + "\n")

    def estimate_step_coefficients(
        self, iteration, seeds, input_text, target_text,
    ):
        """Evaluate one fixed population and return an unapplied seed plan."""
        seeds = [int(seed) for seed in seeds]
        if len(seeds) != self.population_size or len(set(seeds)) != len(seeds):
            raise ValueError("population seeds must be unique and complete")
        domain_sampling = self._sampling_params(
            n=self.n_samples,
            seed=(42 if self.global_seed is None else self.global_seed)
            + iteration,
            temperature=self.train_temperature,
            top_p=self.train_top_p,
            max_tokens=self.max_tokens,
        )
        selected_anchor = select_anchor_items(
            self.anchor_items, iteration, self.anchor_items_per_step,
            self.global_seed,
        )
        aggregate = {
            seed: {"reward_sum": 0.0, "count": 0} for seed in seeds
        }
        anchor_scores = {}
        mini_batch_count = 0
        for mini_batch_count, (input_batch, target_batch) in enumerate(
            self._iter_minibatches(
                input_text, target_text, self.mini_batch_size,
            ),
            1,
        ):
            if not input_batch:
                continue
            anchors = selected_anchor if mini_batch_count == 1 else []
            member_metrics, batch_anchor_scores, _ = (
                self._evaluate_population_with_anchor(
                    seeds, input_batch, target_batch, domain_sampling,
                    anchors, iteration,
                )
            )
            for seed, metrics in member_metrics.items():
                count = len(input_batch)
                aggregate[seed]["reward_sum"] += metrics["avg_reward"] * count
                aggregate[seed]["count"] += count
            if batch_anchor_scores:
                if anchor_scores:
                    raise ValueError("anchor population was scored twice")
                anchor_scores = batch_anchor_scores
        if mini_batch_count == 0:
            raise ValueError("training step received an empty batch")
        if set(anchor_scores) != set(seeds):
            raise ValueError(
                "anchor scores do not align with population seeds"
            )

        domain_scores = []
        ordered_anchor_scores = []
        for seed in seeds:
            count = aggregate[seed]["count"]
            if count <= 0:
                raise ValueError(f"domain seed {seed} received no examples")
            domain_scores.append(aggregate[seed]["reward_sum"] / count)
            ordered_anchor_scores.append(anchor_scores[seed])
        projection = project_anchor_safe_coefficients(
            domain_scores,
            ordered_anchor_scores,
            min_anchor_cosine=self.min_anchor_cosine,
        )
        coefficients = projection["coefficients"]
        plan = {
            "schema": "eggroll-es-anchored-seed-plan-v1",
            "iteration": int(iteration),
            "seeds": seeds,
            "coefficients": coefficients,
            "coefficient_sha256": coefficient_sha256(seeds, coefficients),
            "domain_scores": domain_scores,
            "anchor_scores": ordered_anchor_scores,
            "anchor_items": [{
                key: item[key]
                for key in (
                    "document_id", "item_id", "text_sha256",
                    "token_ids_sha256",
                )
            } for item in selected_anchor],
            "projection": projection["diagnostics"],
            "applied_alpha": 0.0,
            "applications": [],
        }
        self.anchor_step_plans.append(plan)
        self._latest_anchor_plan = plan
        self._persist_anchor_plan(plan)
        print(json.dumps({
            "anchor_seed_plan": plan["coefficient_sha256"],
            "iteration": iteration,
            "projection": plan["projection"],
        }, sort_keys=True))
        return plan

    def apply_seed_coefficients(self, plan, target_alpha):
        """Apply a monotonic target alpha along one reusable fixed seed plan.

        Calling this successively with increasing target alphas applies only
        the increment since the previous target.  This is the integration seam
        for a resident-engine one-step alpha line search: population and anchor
        rollouts are evaluated once, while validation/OOD can be scored after
        each monotonic increment.  No rollback is attempted in BF16.
        """
        if plan is not self._latest_anchor_plan:
            raise ValueError("only the latest unapplied seed plan may be used")
        if coefficient_sha256(
            plan["seeds"], plan["coefficients"],
        ) != plan["coefficient_sha256"]:
            raise ValueError("seed plan coefficients changed after estimation")
        target_alpha = float(target_alpha)
        previous = float(plan["applied_alpha"])
        if not math.isfinite(target_alpha) or target_alpha < previous:
            raise ValueError("target alpha must be finite and monotonic")
        increment = target_alpha - previous
        if increment == 0.0:
            return plan
        self._resolve(
            self.engines[0].collective_rpc.remote(
                "update_weights_from_seeds",
                args=(
                    plan["seeds"], plan["coefficients"], increment,
                    self.population_size,
                ),
            )
        )
        self._resolve([
            engine.collective_rpc.remote("broadcast_all_weights", args=(0,))
            for engine in self.engines
        ])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        plan["applied_alpha"] = target_alpha
        plan["applications"].append({
            "alpha_increment": increment,
            "target_alpha": target_alpha,
        })
        self._persist_anchor_plan(plan)
        return plan

    def train_step(self, iteration, seeds, input_text, target_text):
        plan = self.estimate_step_coefficients(
            iteration, seeds, input_text, target_text,
        )
        self.apply_seed_coefficients(plan, self.alpha)
        return plan

    def eval_step(self, iteration):
        """Run upstream-equivalent deterministic eval on every live engine."""
        to_log = {"eval-iteration": iteration}
        mean_eval_results = []
        sampling_params = self._sampling_params(
            n=1,
            seed=(self.global_seed or 42) + iteration,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_tokens,
        )
        for name, eval_loader in self.eval_dataloader_dict.items():
            sum_reward = 0.0
            count = 0
            save_results = []
            for input_text, target_text in eval_loader:
                prompts = [self.template(item) for item in input_text]
                outputs = dispatch_eval_batch(
                    self.engines, prompts, sampling_params, self._resolve,
                )
                metrics = self._postprocess_outputs(
                    outputs, target_text, eval=True,
                )
                item_count = len(metrics["rewards"])
                sum_reward += metrics["avg_reward"] * item_count
                count += item_count
                save_results.extend(metrics["results"])
            dataset_result = sum_reward / count if count else 0.0
            mean_eval_results.append(dataset_result)
            print(f"{name} -- eval pass@1: {dataset_result} --")
            to_log.update({
                "global_step": iteration,
                f"eval/{name}/pass@1/mean": dataset_result,
            })
            output_path = (
                Path(self.logging_dir) / "eval-output"
                / f"model_eval_task{name}_iteration{iteration + 1}.json"
            )
            output_path.write_text(json.dumps(
                save_results, ensure_ascii=False, indent=4,
            ))

        mean_result = (
            math.fsum(mean_eval_results) / len(mean_eval_results)
            if mean_eval_results else 0.0
        )
        to_log["eval/avgpass@1/mean"] = mean_result
        if self.logging == "wandb":
            self.wandb.log(to_log, commit=True)
        if self.save_best_models and mean_result > self.best_avg:
            self.best_avg = mean_result
            model_path = (
                Path(self.logging_dir) / "checkpoints"
                / f"{self.experiment_name}-mean{mean_result}"
            )
            model_path.mkdir(parents=True, exist_ok=True)
            self._resolve(
                self.engines[0].collective_rpc.remote(
                    "save_self_weights_to_disk",
                    args=(str(model_path / "pytorch_model.pth"),),
                )
            )


def load_trainer():
    parent = base.load_trainer()

    class AnchoredEvolutionStrategiesTrainer(AnchoredStepMixin, parent):
        pass

    return AnchoredEvolutionStrategiesTrainer


def run_exact_steps(trainer, *args, **kwargs):
    summary = base.run_exact_steps(trainer, *args, **kwargs)
    summary["schema"] = "eggroll-es-anchored-exact-run-v1"
    summary["anchor"] = {
        "dataset": {
            "path": trainer.anchor_dataset["path"],
            "sha256": trainer.anchor_dataset["sha256"],
            "rows": len(trainer.anchor_dataset["rows"]),
            "report": trainer.anchor_dataset["report"],
        },
        "scoring": {
            "add_special_tokens": False,
            "aggregation": "token_weighted_selected_token_mean_logprob",
            "items_per_step": trainer.anchor_items_per_step,
            "max_input_tokens": trainer.anchor_max_input_tokens,
            "same_perturbation_as_domain": True,
        },
        "projection": {
            "min_anchor_cosine": trainer.min_anchor_cosine,
            "restandardize_after_projection": False,
            "update_norm_cap": "unconstrained_domain_coefficient_norm",
        },
        "step_plans": trainer.anchor_step_plans,
    }
    summary["implementation"] = {
        "anchor_trainer": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "base_trainer": {
            "path": str(Path(base.__file__).resolve()),
            "sha256": file_sha256(Path(base.__file__).resolve()),
        },
        "coefficient_projection": {
            "path": str((ROOT / "eggroll_es_anchor.py").resolve()),
            "sha256": file_sha256(ROOT / "eggroll_es_anchor.py"),
        },
        "upstream_trainer": {
            "path": str((
                ROOT / "es-at-scale/es_at_scale/trainer/es_trainer.py"
            ).resolve()),
            "sha256": file_sha256(
                ROOT / "es-at-scale/es_at_scale/trainer/es_trainer.py"
            ),
        },
        "upstream_worker_extension": {
            "path": str((
                ROOT / "es-at-scale/es_at_scale/utils/worker_extension.py"
            ).resolve()),
            "sha256": file_sha256(
                ROOT / "es-at-scale/es_at_scale/utils/worker_extension.py"
            ),
        },
    }
    path = Path(trainer.logging_dir) / "run_summary.json"
    path.write_text(json.dumps(
        summary, ensure_ascii=False, indent=2, sort_keys=True,
    ) + "\n")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train-only prose-anchored EGGROLL-ES specialist family"
    )
    parser.add_argument(
        "--model-name", default=str(ROOT / "models/Qwen3.6-35B-A3B")
    )
    parser.add_argument(
        "--train-dataset",
        default=str(ROOT / "data/eggroll_es_specialist/train"),
    )
    parser.add_argument(
        "--eval-dataset",
        default=str(ROOT / "data/eggroll_es_specialist/eval"),
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--sigma", type=float, default=0.001)
    parser.add_argument("--alpha", type=float, default=-1.0)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mini-batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-vllm-engines", type=int, default=4)
    parser.add_argument("--n-gpu-per-vllm-engine", type=int, default=1)
    parser.add_argument("--use-gpus", default="0,1,2,3")
    parser.add_argument("--eval-splits", default="validation,ood_qa")
    parser.add_argument("--exact-train-steps", type=int, required=True)
    parser.add_argument("--skip-baseline-eval", action="store_true")
    parser.add_argument("--save-final-checkpoint", action="store_true")
    parser.add_argument("--ood-prose-jsonl")
    parser.add_argument(
        "--ood-prose-max-input-tokens", type=int, default=1024,
    )
    parser.add_argument(
        "--max-ood-prose-degradation", type=float, default=0.0,
    )
    parser.add_argument(
        "--anchor-prose-jsonl", default=str(DEFAULT_ANCHOR),
    )
    parser.add_argument(
        "--anchor-prose-report", default=str(DEFAULT_ANCHOR_REPORT),
    )
    parser.add_argument("--anchor-items-per-step", type=int, default=2)
    parser.add_argument("--anchor-max-input-tokens", type=int, default=512)
    parser.add_argument("--min-anchor-cosine", type=float, default=0.1)
    parser.add_argument("--reward-function-timeout", type=int, default=10)
    parser.add_argument(
        "--output-directory", default=str(ROOT / "experiments/eggroll_es")
    )
    parser.add_argument(
        "--experiment-name", default="qwen36-specialist-anchor-v1"
    )
    parser.add_argument("--logging", choices=["none", "wandb"], default="none")
    parser.add_argument("--wandb-project", default="specialist-eggroll-es")
    parser.add_argument("--save-best-models", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.alpha == -1.0:
        args.alpha = args.sigma / 2.0
    if args.n_vllm_engines * args.n_gpu_per_vllm_engine != len(
        args.use_gpus.split(",")
    ):
        raise ValueError(
            "engine GPU allocation must consume every selected GPU"
        )
    if args.anchor_items_per_step <= 0:
        raise ValueError("anchor items per step must be positive")
    if args.anchor_max_input_tokens < 2:
        raise ValueError("anchor max input tokens must be at least two")
    if not 0.0 <= args.min_anchor_cosine < 1.0:
        raise ValueError("minimum anchor cosine must be in [0, 1)")
    if args.ood_prose_max_input_tokens < 2:
        raise ValueError("OOD prose max input tokens must be at least two")
    if (
        not math.isfinite(args.max_ood_prose_degradation)
        or args.max_ood_prose_degradation < 0.0
    ):
        raise ValueError("maximum OOD prose degradation must be non-negative")
    anchor = load_anchor_prose(
        args.anchor_prose_jsonl, args.anchor_prose_report,
    )
    ood_prose = (
        base.load_ood_prose(args.ood_prose_jsonl)
        if args.ood_prose_jsonl else None
    )
    base.set_seed(args.seed)

    train_dict = load_from_disk(args.train_dataset)
    if list(train_dict) != ["train"]:
        raise ValueError("training artifact must contain exactly train")
    train_loader = base.build_train_loader(
        train_dict["train"], args.batch_size, args.seed,
    )
    eval_dict = load_from_disk(args.eval_dataset)
    requested = [
        split.strip() for split in args.eval_splits.split(",")
        if split.strip()
    ]
    missing = sorted(set(requested) - set(eval_dict))
    if missing:
        raise ValueError(f"unknown evaluation splits: {missing}")
    eval_loaders = {
        name: DataLoader(
            eval_dict[name], batch_size=args.mini_batch_size,
            collate_fn=base.specialist_collate, shuffle=False,
        )
        for name in requested
    }

    trainer_class = load_trainer()
    trainer = trainer_class(
        model_name=args.model_name,
        checkpoint=args.checkpoint,
        sigma=args.sigma,
        alpha=args.alpha,
        population_size=args.population_size,
        reward_shaping="z-scores",
        num_iterations=args.exact_train_steps,
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
        save_best_models=args.save_best_models,
        reward_function_timeout=args.reward_function_timeout,
        output_directory=args.output_directory,
    )
    trainer.configure_anchor(
        anchor,
        items_per_step=args.anchor_items_per_step,
        max_input_tokens=args.anchor_max_input_tokens,
        min_anchor_cosine=args.min_anchor_cosine,
    )
    run_exact_steps(
        trainer, args.exact_train_steps,
        skip_baseline_eval=args.skip_baseline_eval,
        save_final_checkpoint=args.save_final_checkpoint,
        ood_prose=ood_prose,
        ood_prose_max_input_tokens=args.ood_prose_max_input_tokens,
        max_ood_prose_degradation=args.max_ood_prose_degradation,
    )


if __name__ == "__main__":
    main()
