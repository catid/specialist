#!/usr/bin/env python3
"""Run upstream ES-at-Scale full-parameter training on specialist QA."""

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

from es_train_acc import answer_score


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "es-at-scale"
COMPAT = ROOT / "eggroll_es_compat"
OOD_PROSE_BOOTSTRAP_SAMPLES = 20000
OOD_PROSE_BOOTSTRAP_SEED = 20260714
OOD_PROSE_DEFAULT_MAX_DEGRADATION = 0.02
REWARD_POOL_SHUTDOWN_TIMEOUT_SECONDS = 10.0
ACTOR_RPC_TIMEOUT_SECONDS = 300.0


class OODProseGateFailure(RuntimeError):
    """The frozen OOD non-inferiority gate rejected checkpoint promotion."""


class _TruthyZero(int):
    def __new__(cls):
        return super().__new__(cls, 0)

    def __bool__(self):
        return True


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


def unique_population_seeds(rng, population_size, upper_bound=2**30):
    """Draw deterministic, order-preserving ES seeds without replacement."""
    if (
        isinstance(population_size, bool)
        or not isinstance(population_size, int)
        or population_size <= 0
        or population_size > upper_bound
    ):
        raise ValueError("population size must fit the unique seed domain")
    seeds = []
    seen = set()
    while len(seeds) < population_size:
        needed = population_size - len(seeds)
        draws = rng.integers(
            0, upper_bound, size=needed, dtype=np.int64,
        ).tolist()
        for seed in draws:
            seed = int(seed)
            if seed not in seen:
                seen.add(seed)
                seeds.append(seed)
    return seeds


def validate_population_seeds(seeds, population_size):
    if (
        not isinstance(seeds, list)
        or len(seeds) != population_size
        or any(type(seed) is not int or not 0 <= seed < 2**30 for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("ES population seeds must be exact, unique integers")
    return seeds


def atomic_json_write(path, value):
    """Publish finite JSON through one atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_trainer_healthy(trainer, operation):
    """Refuse observations or publication from an uncertain weight state."""
    if getattr(trainer, "_specialist_state_poisoned", False):
        raise RuntimeError(
            f"cannot {operation}: ES trainer state is poisoned"
        )


def _ray_get_with_timeout(
    ray_module, getter, handles, *, timeout, context,
):
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0.0
    ):
        raise ValueError("Ray timeout must be finite and positive")
    try:
        return getter(handles, timeout=float(timeout))
    except BaseException as error:
        timeout_type = getattr(
            getattr(ray_module, "exceptions", None), "GetTimeoutError", None,
        )
        if timeout_type is not None and isinstance(error, timeout_type):
            raise TimeoutError(f"timed out while {context}") from error
        raise


def bounded_ray_get(
    ray_module, handles, context, timeout=ACTOR_RPC_TIMEOUT_SECONDS,
):
    """Resolve Ray work under one fixed fail-stop deadline."""
    return _ray_get_with_timeout(
        ray_module, ray_module.get, handles,
        timeout=timeout, context=context,
    )


@contextmanager
def fixed_ray_get_timeout(
    ray_module, *, timeout=ACTOR_RPC_TIMEOUT_SECONDS, context,
):
    """Bound inherited Ray gets without patching the upstream submodule."""
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or float(timeout) <= 0.0
    ):
        raise ValueError("Ray timeout must be finite and positive")
    fixed_timeout = float(timeout)
    original_get = ray_module.get

    def bounded(handles, *, timeout=None):
        if timeout is None:
            effective = fixed_timeout
        else:
            if (
                isinstance(timeout, bool)
                or not isinstance(timeout, (int, float))
                or not math.isfinite(float(timeout))
                or float(timeout) <= 0.0
            ):
                raise ValueError("Ray timeout must be finite and positive")
            effective = min(float(timeout), fixed_timeout)
        return _ray_get_with_timeout(
            ray_module, original_get, handles,
            timeout=effective, context=context,
        )

    ray_module.get = bounded
    try:
        yield
    finally:
        ray_module.get = original_get


def save_checkpoint_atomic(trainer, checkpoint_dir):
    """Save through a private sibling directory, then publish atomically."""
    assert_trainer_healthy(trainer, "save a checkpoint")
    import ray

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
    temporary_dir = Path(tempfile.mkdtemp(
        prefix=f".{checkpoint_dir.name}.tmp-", dir=checkpoint_dir.parent,
    ))
    temporary_checkpoint = temporary_dir / "pytorch_model.pth"
    try:
        bounded_ray_get(
            ray,
            trainer.engines[0].collective_rpc.remote(
                "save_self_weights_to_disk", args=(str(temporary_checkpoint),),
            ),
            "saving checkpoint",
        )
        if not temporary_checkpoint.is_file():
            raise RuntimeError("checkpoint RPC did not publish its file")
        checkpoint_sha256 = file_sha256(temporary_checkpoint)
        os.rename(temporary_dir, checkpoint_dir)
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    checkpoint = checkpoint_dir / "pytorch_model.pth"
    return checkpoint, checkpoint_sha256


def safe_postprocess_outputs(trainer, generated_text, target_text, eval=False):
    """Score complete aligned batches with deterministic timeout artifacts."""
    from multiprocessing import TimeoutError as PoolTimeoutError

    if len(generated_text) != len(target_text):
        raise ValueError("generated and target batch sizes differ")
    if type(trainer.n_samples) is not int or trainer.n_samples <= 0:
        raise ValueError("n_samples must be a positive exact integer")
    rewards_per_prompt = []
    lengths_per_prompt = []
    raw_rewards_per_prompt = []
    raw_lengths_per_prompt = []
    saved = []
    for generation, target in zip(generated_text, target_text, strict=True):
        if len(generation.outputs) != trainer.n_samples:
            raise RuntimeError("generation rollout cardinality changed")
        rollout_rewards = []
        rollout_lengths = []
        for rollout_index, output in enumerate(generation.outputs):
            response_text = output.text
            token_ids = output.token_ids
            generation_length = len(token_ids)
            decoded_response = None
            if eval:
                decoded_response = trainer.tokenizer.decode(
                    token_ids, skip_special_tokens=True,
                )
            label, reward = "timeout", 0.0
            result = trainer.mp_pool.apply_async(
                trainer.task, (response_text, target),
            )
            try:
                label, reward = result.get(
                    timeout=trainer.reward_function_timeout,
                )
            except PoolTimeoutError:
                pass
            reward = float(reward)
            if not math.isfinite(reward):
                raise ValueError("reward function returned a non-finite value")
            rollout_rewards.append(reward)
            rollout_lengths.append(int(generation_length))
            if eval:
                saved.append({
                    "prompt": generation.prompt,
                    "answer": target,
                    "rollout_idx": int(rollout_index),
                    "decoded_response": decoded_response,
                    "model_output": response_text,
                    "reward": reward,
                    "format": str(label),
                    "response_length": int(generation_length),
                })
        raw_rewards_per_prompt.append(rollout_rewards)
        raw_lengths_per_prompt.append(rollout_lengths)
        rewards_per_prompt.append(
            float(np.mean(rollout_rewards)) if rollout_rewards else 0.0
        )
        lengths_per_prompt.append(
            float(np.mean(rollout_lengths)) if rollout_lengths else 0.0
        )
    return {
        "rewards": rewards_per_prompt,
        "avg_reward": (
            float(np.mean(rewards_per_prompt)) if rewards_per_prompt else 0.0
        ),
        "gen_lengths": lengths_per_prompt,
        "avg_gen_lengths": (
            float(np.mean(lengths_per_prompt)) if lengths_per_prompt else 0.0
        ),
        "results": saved,
        "raw_rewards_per_prompt": raw_rewards_per_prompt,
        "raw_lens_per_prompt": raw_lengths_per_prompt,
        "rollout_reduce": trainer.rollout_reduce,
        "n_samples": int(trainer.n_samples),
    }


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
        source_url = row.get("normalized_source_url")
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
        if not isinstance(source_url, str) or not source_url.strip():
            raise ValueError(
                f"OOD prose item {item_id} has no normalized_source_url"
            )
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


def dispatch_qa_eval(engines, prompts, sampling_params, resolve):
    """Generate deterministic round-robin QA shards on every live engine."""
    if not engines:
        raise ValueError("QA evaluation requires at least one engine")
    if not prompts:
        return []
    partitions = [[] for _ in engines]
    for position, prompt in enumerate(prompts):
        partitions[position % len(engines)].append((position, prompt))

    handles = []
    assignments = []
    for engine, partition in zip(engines, partitions, strict=True):
        if not partition:
            continue
        handles.append(engine.generate.remote(
            [prompt for _, prompt in partition],
            sampling_params, use_tqdm=False,
        ))
        assignments.append([position for position, _ in partition])

    batches = resolve(handles)
    if len(batches) != len(assignments):
        raise RuntimeError("QA eval engine result batch count changed")
    ordered = [None] * len(prompts)
    for positions, batch in zip(assignments, batches, strict=True):
        if len(batch) != len(positions):
            raise RuntimeError("QA eval engine changed its request count")
        for position, output in zip(positions, batch, strict=True):
            ordered[position] = output
    if any(output is None for output in ordered):
        raise RuntimeError("QA eval engine omitted a request")
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


def linear_percentile(values, probability):
    """Linearly interpolate a percentile, matching paired QA reports."""
    if not values:
        raise ValueError("cannot take a percentile of no values")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    if ordered[lower] == ordered[upper]:
        return ordered[lower]
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compare_ood_prose(
    baseline,
    final,
    max_degradation=OOD_PROSE_DEFAULT_MAX_DEGRADATION,
    bootstrap_samples=OOD_PROSE_BOOTSTRAP_SAMPLES,
    bootstrap_seed=OOD_PROSE_BOOTSTRAP_SEED,
):
    """Apply a paired source-document bootstrap non-inferiority gate."""
    if not math.isfinite(max_degradation) or max_degradation < 0.0:
        raise ValueError("maximum OOD prose degradation must be non-negative")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples <= 0
    ):
        raise ValueError("OOD prose bootstrap samples must be positive")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int):
        raise ValueError("OOD prose bootstrap seed must be an integer")
    alignment_fields = (
        "item_id", "normalized_source_url", "text_sha256",
        "token_ids_sha256",
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

    documents = {}
    for baseline_item, final_item in zip(
        baseline["items"], final["items"],
    ):
        document_id = baseline_item["normalized_source_url"]
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("OOD prose item has no source-document identity")
        scored_tokens = baseline_item["scored_token_count"]
        if (
            isinstance(scored_tokens, bool)
            or not isinstance(scored_tokens, int)
            or scored_tokens <= 0
        ):
            raise ValueError("OOD prose scored token count must be positive")
        baseline_sum = float(baseline_item["sum_token_logprob"])
        final_sum = float(final_item["sum_token_logprob"])
        if not math.isfinite(baseline_sum) or not math.isfinite(final_sum):
            raise ValueError("OOD prose item has a non-finite logprob sum")
        document = documents.setdefault(document_id, {
            "baseline_sums": [],
            "final_sums": [],
            "scored_token_count": 0,
        })
        document["baseline_sums"].append(baseline_sum)
        document["final_sums"].append(final_sum)
        document["scored_token_count"] += scored_tokens

    document_rows = [
        {
            "baseline_sum": math.fsum(document["baseline_sums"]),
            "final_sum": math.fsum(document["final_sums"]),
            "scored_token_count": document["scored_token_count"],
        }
        for document in documents.values()
    ]
    total_tokens = sum(
        document["scored_token_count"] for document in document_rows
    )
    baseline_mean = (
        math.fsum(document["baseline_sum"] for document in document_rows)
        / total_tokens
    )
    final_mean = (
        math.fsum(document["final_sum"] for document in document_rows)
        / total_tokens
    )
    delta = final_mean - baseline_mean

    rng = random.Random(bootstrap_seed)
    document_count = len(document_rows)
    bootstrap_deltas = []
    for _ in range(bootstrap_samples):
        sample = [
            document_rows[rng.randrange(document_count)]
            for _ in range(document_count)
        ]
        sample_tokens = sum(
            document["scored_token_count"] for document in sample
        )
        sample_baseline = (
            math.fsum(document["baseline_sum"] for document in sample)
            / sample_tokens
        )
        sample_final = (
            math.fsum(document["final_sum"] for document in sample)
            / sample_tokens
        )
        bootstrap_deltas.append(sample_final - sample_baseline)
    confidence_interval = [
        linear_percentile(bootstrap_deltas, 0.025),
        linear_percentile(bootstrap_deltas, 0.975),
    ]
    return {
        "metric": "mean_token_logprob",
        "higher_is_better": True,
        "baseline": baseline_mean,
        "final": final_mean,
        "delta": delta,
        "max_degradation": max_degradation,
        "paired_document_bootstrap_95_ci": confidence_interval,
        "bootstrap": {
            "unit": "normalized_source_url",
            "document_count": document_count,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "percentiles": [0.025, 0.975],
        },
        "passed": confidence_interval[0] >= -max_degradation,
    }


def score_ood_prose(trainer, dataset, label, max_input_tokens):
    """Score frozen prose on all live vLLM engines and save item results."""
    assert_trainer_healthy(trainer, "evaluate OOD prose")
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
        trainer.engines, items, sampling_params,
        lambda handles: bounded_ray_get(
            ray, handles, "OOD prose generation",
        ),
    )
    evaluation = summarize_ood_prose(items, outputs)
    result_path = (
        Path(trainer.logging_dir) / "eval-output"
        / f"ood_prose_{label}.json"
    )
    atomic_json_write(result_path, evaluation["items"])
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
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
    from es_at_scale.trainer.es_trainer import (
        ESNcclLLM,
        EvolutionStrategiesTrainer,
    )

    class Qwen36EvolutionStrategiesTrainer(EvolutionStrategiesTrainer):
        def _postprocess_outputs(self, generated_text, target_text, eval=False):
            return safe_postprocess_outputs(
                self, generated_text, target_text, eval=eval,
            )

        def _with_seed_zero_preserved(self, function, *args, **kwargs):
            original = self.global_seed
            if original == 0:
                self.global_seed = _TruthyZero()
            try:
                return function(*args, **kwargs)
            finally:
                self.global_seed = original

        def train_step(self, iteration, seeds, input_text, target_text):
            validate_population_seeds(seeds, self.population_size)
            if getattr(self, "_specialist_state_poisoned", False):
                raise RuntimeError("ES trainer state is poisoned after failed wave")
            try:
                with fixed_ray_get_timeout(
                    ray, context="ES train-step actor work",
                ):
                    return self._with_seed_zero_preserved(
                        super().train_step,
                        iteration, seeds, input_text, target_text,
                    )
            except BaseException:
                # Update/broadcast RPCs occur after evaluation.  If any
                # inherited wait fails, actor state is no longer provable.
                self._specialist_state_poisoned = True
                raise

        def eval_step(self, iteration):
            from vllm import SamplingParams

            assert_trainer_healthy(self, "evaluate QA")
            to_log = {"eval-iteration": iteration}
            mean_eval_results = []
            for name, eval_loader in self.eval_dataloader_dict.items():
                sum_reward = 0.0
                count = 0
                saved = []
                for input_text, target_text in eval_loader:
                    prompts = [self.template(item) for item in input_text]
                    sampling_params = SamplingParams(
                        n=1,
                        seed=exact_step_seed(self.global_seed, iteration),
                        temperature=0.0,
                        top_p=1.0,
                        max_tokens=self.max_tokens,
                    )
                    outputs = dispatch_qa_eval(
                        self.engines, prompts, sampling_params,
                        lambda handles: bounded_ray_get(
                            ray, handles, "QA evaluation generation",
                        ),
                    )
                    metrics = self._postprocess_outputs(
                        outputs, target_text, eval=True,
                    )
                    prompt_count = len(metrics["rewards"])
                    sum_reward += metrics["avg_reward"] * prompt_count
                    count += prompt_count
                    saved.extend(metrics["results"])
                dataset_score = sum_reward / count if count else 0.0
                mean_eval_results.append(dataset_score)
                print(f"{name} -- eval pass@1: {dataset_score} --")
                to_log.update({
                    "global_step": iteration,
                    f"eval/{name}/pass@1/mean": dataset_score,
                })
                path = (
                    Path(self.logging_dir) / "eval-output"
                    / f"model_eval_task{name}_iteration{iteration + 1}.json"
                )
                atomic_json_write(path, saved)
            to_log["eval/avgpass@1/mean"] = float(
                np.mean(mean_eval_results)
            )
            if self.logging == "wandb":
                self.wandb.log(to_log, commit=True)
            if self.save_best_models:
                maybe_save_best_checkpoint_atomic(self, iteration)

        def fit(self):
            return run_standard_fit_exact(self)

        def _handle_exit(self, sig, frame):
            del frame
            raise SystemExit(128 + int(sig))

        def cleanup(self):
            """Tolerate construction that stopped before attributes existed."""
            for engine in list(getattr(self, "engines", [])):
                try:
                    ray.kill(engine)
                except BaseException:
                    pass
            for group in list(getattr(self, "pgs", [])):
                try:
                    remove_placement_group(group)
                except BaseException:
                    pass

        def evaluate_population_on_batch(
            self, seeds, input_batch, target_batch, sampling_params,
        ):
            validate_population_seeds(seeds, self.population_size)
            if getattr(self, "_specialist_state_poisoned", False):
                raise RuntimeError("ES trainer state is poisoned after failed wave")
            seeds_perf_batch = {}
            results_this_generation = []
            for start in range(0, len(seeds), self.n_vllm_engines):
                engine_batch = seeds[start:start + self.n_vllm_engines]
                engines = self.engines[:len(engine_batch)]
                perturbations = []
                try:
                    for engine, seed in zip(
                        engines, engine_batch, strict=True,
                    ):
                        perturbations.append(
                            engine.collective_rpc.remote(
                                "perturb_self_weights",
                                args=(int(seed), self.sigma, False),
                            )
                        )
                    bounded_ray_get(
                        ray, perturbations, "perturbing ES population wave",
                    )
                except BaseException as error:
                    # Which actors committed is unknowable after a partial RPC
                    # failure.  Never subtract speculatively or train again.
                    self._specialist_state_poisoned = True
                    raise RuntimeError(
                        "ES perturbation wave failed with uncertain state"
                    ) from error

                handles = []
                generation_error = None
                outputs = None
                try:
                    for engine in engines:
                        handles.append(self.evaluate_handle(
                            engine, input_batch,
                            sampling_params=sampling_params,
                        ))
                    outputs = bounded_ray_get(
                        ray, handles, "generating ES population wave",
                    )
                except BaseException as error:
                    generation_error = error
                    for handle in handles:
                        try:
                            ray.cancel(handle, force=True)
                        except BaseException:
                            pass

                try:
                    bounded_ray_get(
                        ray,
                        [
                            engine.collective_rpc.remote(
                                "restore_self_weights",
                                args=(int(seed), self.sigma),
                            )
                            for engine, seed in zip(
                                engines, engine_batch, strict=True,
                            )
                        ],
                        "restoring ES population wave",
                    )
                except BaseException as restore_error:
                    self._specialist_state_poisoned = True
                    raise RuntimeError(
                        "ES wave restoration failed; trainer state is poisoned"
                    ) from restore_error
                if generation_error is not None:
                    self._specialist_state_poisoned = True
                    raise RuntimeError(
                        "ES generation failed after perturbation; wave aborted"
                    ) from generation_error

                for seed, output in zip(engine_batch, outputs, strict=True):
                    metrics = self._postprocess_outputs(output, target_batch)
                    seeds_perf_batch[int(seed)] = metrics
                    results_this_generation.append({
                        "seed": int(seed),
                        "avg_reward": metrics["avg_reward"],
                    })
            return seeds_perf_batch, results_this_generation

        def launch_engines(self, num_engines=4, n_gpu_per_vllm_engine=1,
                           model_name="Qwen/Qwen2.5-Math-1.5B",
                           precision="bfloat16"):
            # Attach each resource before the next allocation can fail so the
            # partial-construction cleanup path can always target it.
            self.pgs = []
            for _ in range(num_engines):
                self.pgs.append(placement_group(
                    [{"GPU": 1, "CPU": 0}] * n_gpu_per_vllm_engine,
                    strategy="PACK",
                ))
            bounded_ray_get(
                ray, [pg.ready() for pg in self.pgs],
                "waiting for GPU placement groups",
            )
            strategies = [
                PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=0,
                )
                for pg in self.pgs
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
            self.engines = []
            for strategy in strategies:
                self.engines.append(
                    ray.remote(
                        num_cpus=0, num_gpus=actor_gpus,
                        scheduling_strategy=strategy,
                    )(ESNcclLLM).remote(**engine_args)
                )
            return self.engines, self.pgs

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
        "--max-ood-prose-degradation", type=float,
        default=OOD_PROSE_DEFAULT_MAX_DEGRADATION,
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


def maybe_save_best_checkpoint_atomic(trainer, iteration):
    """Atomically publish an opt-in best checkpoint after complete eval."""
    metrics = eval_metrics(
        trainer.logging_dir, iteration, trainer.eval_dataloader_dict.keys(),
    )
    if not metrics:
        raise ValueError("best-model selection requires evaluation splits")
    mean_score = float(np.mean(list(metrics.values())))
    if not math.isfinite(mean_score):
        raise ValueError("best-model score is non-finite")
    if mean_score <= float(trainer.best_avg):
        return None
    checkpoint_dir = (
        Path(trainer.logging_dir) / "checkpoints"
        / f"{trainer.experiment_name}-mean{mean_score!r}"
    )
    checkpoint, _checkpoint_sha256 = save_checkpoint_atomic(
        trainer, checkpoint_dir,
    )
    trainer.best_avg = mean_score
    return checkpoint


def _call_with_timeout(function, timeout, description):
    outcome = []

    def invoke():
        try:
            outcome.append((True, function()))
        except BaseException as error:
            outcome.append((False, error))

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"timed out while {description}")
    succeeded, value = outcome[0]
    if not succeeded:
        raise value
    return value


def close_trainer(trainer, timeout=REWARD_POOL_SHUTDOWN_TIMEOUT_SECONDS):
    """Idempotently bound reward workers, Ray actors, and placement groups."""
    if getattr(trainer, "_specialist_closed", False):
        return
    errors = []
    pool = getattr(trainer, "mp_pool", None)
    if pool is not None:
        pool_errors = []
        processes = list(getattr(pool, "_pool", []))
        try:
            action = getattr(pool, "terminate", None) or getattr(pool, "close")
            _call_with_timeout(action, timeout * 0.4, "stopping reward pool")
        except BaseException as error:
            pool_errors.append(error)
        deadline = time.monotonic() + timeout * 0.6
        for process in processes:
            try:
                remaining = max(0.0, deadline - time.monotonic())
                process.join(timeout=remaining)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=max(0.0, deadline - time.monotonic()))
                if process.is_alive():
                    raise TimeoutError("reward worker remained alive")
            except BaseException as error:
                pool_errors.append(error)
        errors.extend(pool_errors)
        if not pool_errors:
            trainer.mp_pool = None
    try:
        _call_with_timeout(trainer.cleanup, timeout, "closing Ray actors")
    except BaseException as error:
        errors.append(error)
    try:
        import ray
    except ImportError:
        ray = None
    if ray is not None:
        try:
            _call_with_timeout(ray.shutdown, timeout, "shutting down Ray")
        except BaseException as error:
            errors.append(error)
    if getattr(trainer, "logging", "none") == "wandb":
        try:
            getattr(trainer, "wandb", None).finish()
        except Exception:
            pass
    if errors:
        raise RuntimeError("trainer cleanup did not complete") from errors[0]
    trainer._specialist_closed = True


def close_trainer_preserving_primary(trainer):
    """Close resources without replacing an exception already in flight."""
    primary_error = sys.exception()
    try:
        close_trainer(trainer)
    except BaseException as cleanup_error:
        if primary_error is None:
            raise
        primary_error.add_note(
            "trainer cleanup also failed: "
            f"{type(cleanup_error).__name__}"
        )


def run_standard_fit_exact(trainer):
    """Faithful default fit with exactly num_iterations updates."""
    if (
        isinstance(trainer.num_iterations, bool)
        or not isinstance(trainer.num_iterations, int)
        or trainer.num_iterations < 0
    ):
        raise ValueError("num_iterations must be a non-negative integer")
    trainer.eval_step(iteration=0)
    if trainer.num_iterations == 0:
        print("-- Evaluation completed! --")
        return None
    train_iterator = iter(trainer.train_dataloader)
    for iteration in range(trainer.num_iterations):
        try:
            input_text, target_text = next(train_iterator)
        except StopIteration:
            train_iterator = iter(trainer.train_dataloader)
            try:
                input_text, target_text = next(train_iterator)
            except StopIteration as error:
                raise ValueError("training dataloader is empty") from error
        prompts = [trainer.template(item) for item in input_text]
        rng = np.random.default_rng(
            seed=exact_step_seed(trainer.global_seed, iteration)
        )
        seeds = unique_population_seeds(rng, trainer.population_size)
        trainer.train_step(
            iteration=iteration, seeds=seeds,
            input_text=prompts, target_text=target_text,
        )
        completed_steps = iteration + 1
        if (
            completed_steps % trainer.eval_freq == 0
            and completed_steps < trainer.num_iterations
        ):
            trainer.eval_step(iteration=completed_steps)
    # The checkpoint must have evaluation evidence for its exact post-update
    # state, even when the periodic cadence did not land on the last update.
    trainer.eval_step(iteration=trainer.num_iterations)
    checkpoint_dir = (
        Path(trainer.logging_dir)
        / f"checkpoint-es_fine_tuned_iteration_{trainer.num_iterations}"
    )
    checkpoint, _checkpoint_sha256 = save_checkpoint_atomic(
        trainer, checkpoint_dir,
    )
    print(f"Final model weights saved to {checkpoint_dir}.")
    return checkpoint


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
        if ood_prose is not None and getattr(
            trainer, "save_best_models", False,
        ):
            raise ValueError(
                "save_best_models is incompatible with an unpassed OOD gate"
            )
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
            seeds = unique_population_seeds(
                loop_rng, trainer.population_size,
            )
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

        summary = {
            "schema": "eggroll-es-exact-run-v1",
            "status": "complete",
            "model": trainer.model_name,
            "steps": steps,
            "sigma": trainer.sigma,
            "alpha": trainer.alpha,
            "population_size": trainer.population_size,
            "batch_size": trainer.batch_size,
            "mini_batch_size": trainer.mini_batch_size,
            "max_tokens": trainer.max_tokens,
            "seed": trainer.global_seed,
            "actor_rpc_timeout_seconds": ACTOR_RPC_TIMEOUT_SECONDS,
            "checkpoint": None,
            "evaluations": {
                label: eval_metrics(
                    trainer.logging_dir, iteration,
                    trainer.eval_dataloader_dict.keys(),
                )
                for label, iteration in evaluated
            },
        }
        if ood_prose is not None:
            ood_gate = compare_ood_prose(
                ood_prose_evaluations["baseline"],
                ood_prose_evaluations["final"],
                max_ood_prose_degradation,
            )
            summary["ood_prose"] = {
                "schema": "eggroll-es-ood-prose-logprob-v2",
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
                "gate": ood_gate,
            }
        summary_path = Path(trainer.logging_dir) / "run_summary.json"
        if ood_prose is not None and not ood_gate["passed"]:
            summary["status"] = "failed_ood_prose_gate"
            atomic_json_write(summary_path, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
            raise OODProseGateFailure(
                "OOD prose non-inferiority gate rejected checkpoint"
            )

        checkpoint = None
        if save_final_checkpoint:
            checkpoint_dir = (
                Path(trainer.logging_dir)
                / f"checkpoint-es_exact_steps_{steps}"
            )
            checkpoint, checkpoint_sha256 = save_checkpoint_atomic(
                trainer, checkpoint_dir,
            )
            summary["checkpoint"] = str(checkpoint)
            summary["checkpoint_sha256"] = checkpoint_sha256
        atomic_json_write(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return summary
    finally:
        close_trainer_preserving_primary(trainer)


def main():
    args = parse_args()
    if args.alpha == -1.0:
        args.alpha = args.sigma / 2.0
    gpu_fields = [field.strip() for field in args.use_gpus.split(",")]
    if not gpu_fields or any(not field.isdecimal() for field in gpu_fields):
        raise ValueError("--use-gpus must contain unique numeric GPU IDs")
    gpu_ids = [int(field) for field in gpu_fields]
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--use-gpus must contain unique numeric GPU IDs")
    args.use_gpus = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    if args.n_vllm_engines * args.n_gpu_per_vllm_engine != len(gpu_fields):
        raise ValueError(
            "engine GPU allocation must consume every selected GPU"
        )
    positive = {
        "population-size": args.population_size,
        "batch-size": args.batch_size,
        "mini-batch-size": args.mini_batch_size,
        "max-tokens": args.max_tokens,
        "eval-freq": args.eval_freq,
        "n-vllm-engines": args.n_vllm_engines,
        "n-gpu-per-vllm-engine": args.n_gpu_per_vllm_engine,
        "reward-function-timeout": args.reward_function_timeout,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"positive arguments required: {', '.join(invalid)}")
    if args.population_size > 2**30:
        raise ValueError("population size exceeds the unique seed domain")
    if args.n_iterations < 0 or (
        args.exact_train_steps is not None and args.exact_train_steps < 0
    ):
        raise ValueError("training iteration counts must be non-negative")
    if (
        not math.isfinite(args.sigma) or args.sigma <= 0.0
        or not math.isfinite(args.alpha) or args.alpha < 0.0
    ):
        raise ValueError("sigma must be positive and alpha non-negative")
    if args.ood_prose_jsonl and args.exact_train_steps is None:
        raise ValueError(
            "--ood-prose-jsonl currently requires --exact-train-steps"
        )
    if args.ood_prose_jsonl and args.save_best_models:
        raise ValueError(
            "--save-best-models cannot precede the frozen OOD gate"
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
    # Must precede the first torch.cuda query in set_seed().
    os.environ["CUDA_VISIBLE_DEVICES"] = args.use_gpus
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
        if not requested_splits:
            raise ValueError("--eval-splits must select at least one split")
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

    if not eval_dict:
        raise ValueError("evaluation artifact must contain at least one split")

    trainer_class = load_trainer()
    trainer_kwargs = dict(
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
    # Retain a reference before the inherited constructor creates the reward
    # pool or Ray actors.  If initialization fails or receives a signal, the
    # same bounded cleanup path can still terminate partial resources.
    trainer = trainer_class.__new__(trainer_class)
    try:
        import ray

        with fixed_ray_get_timeout(
            ray, context="trainer construction actor work",
        ):
            trainer_class.__init__(trainer, **trainer_kwargs)
    except BaseException as construction_error:
        try:
            close_trainer(trainer)
        except BaseException as cleanup_error:
            construction_error.add_note(
                "partial trainer cleanup also failed: "
                f"{type(cleanup_error).__name__}"
            )
        raise
    if args.exact_train_steps is None:
        try:
            trainer.fit()
        finally:
            close_trainer_preserving_primary(trainer)
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
