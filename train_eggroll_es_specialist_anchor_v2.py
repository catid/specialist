#!/usr/bin/env python3
"""Fail-closed anchored EGGROLL-ES recipe with exact BF16 restoration.

This is a new implementation family.  It imports but does not modify the
frozen v1 anchor trainer or the upstream ES-at-Scale checkout.  Every engine
keeps a CPU copy of its latest committed weights; population perturbations are
restored with ``copy_`` in a ``finally`` block.  Domain generation is resolved
before a distinct anchor generation call, so anchor count cannot change the
domain request batch.
"""

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import train_eggroll_es_specialist_anchor as anchor_v1


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = "eggroll_es_worker_v2.ExactAuditWorkerExtension"
_V1_LOAD_TRAINER = anchor_v1.load_trainer
_V1_RUN_EXACT_STEPS = anchor_v1.run_exact_steps


# The resident v1 driver deliberately accepts an anchor module as its adapter
# seam.  Keep the two pure data/identity helpers explicit here rather than
# relying on accidental transitive imports.
def load_anchor_prose(*args, **kwargs):
    return anchor_v1.load_anchor_prose(*args, **kwargs)


def coefficient_sha256(seeds, coefficients):
    return anchor_v1.coefficient_sha256(seeds, coefficients)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value):
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _simple_value(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity probe contains a non-finite float")
        return value
    return str(value)


def domain_output_sha256(outputs):
    """Hash generated tokens and text, not merely their scalar reward."""
    requests = []
    for request in outputs:
        completions = getattr(request, "outputs", None)
        if completions is None:
            # Small mock seam used by CPU regression tests.
            requests.append({
                "reward": _simple_value(getattr(request, "reward", None)),
            })
            continue
        rows = []
        for completion in completions:
            token_ids = getattr(completion, "token_ids", [])
            rows.append({
                "token_ids": [int(token_id) for token_id in token_ids],
                "text": str(getattr(completion, "text", "")),
                "finish_reason": _simple_value(
                    getattr(completion, "finish_reason", None),
                ),
                "stop_reason": _simple_value(
                    getattr(completion, "stop_reason", None),
                ),
                "cumulative_logprob": _simple_value(
                    getattr(completion, "cumulative_logprob", None),
                ),
            })
        requests.append({
            "prompt": _simple_value(getattr(request, "prompt", None)),
            "prompt_token_ids": [
                int(token_id)
                for token_id in (getattr(request, "prompt_token_ids", []) or [])
            ],
            "outputs": rows,
        })
    return canonical_sha256(requests)


def anchor_output_sha256(items, outputs):
    if len(items) != len(outputs):
        raise ValueError("anchor identity probe changed request count")
    values = []
    for item, output in zip(items, outputs):
        values.append({
            "item_id": item["item_id"],
            "selected_token_logprobs": [
                float(value) for value in anchor_v1.base.prompt_token_logprobs(
                    output, item["prompt_token_ids"],
                )
            ],
        })
    return canonical_sha256(values)


def _atomic_write_json(path, value):
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


def _all_collective_results(value, predicate):
    """Accept vLLM's per-TP-worker result lists without weakening checks."""
    if isinstance(value, (list, tuple)):
        return bool(value) and all(
            _all_collective_results(item, predicate) for item in value
        )
    return predicate(value)


class ExactRestoredAnchoredStepMixin:
    """Correct v1 batching/restoration and gate all nonzero updates."""

    def configure_anchor(self, *args, **kwargs):
        result = super().configure_anchor(*args, **kwargs)
        if not self.engines:
            raise ValueError("anchored v2 requires live engines")
        if self.population_size % len(self.engines) != 0:
            raise ValueError(
                "anchored v2 population size must be a multiple of engine "
                "count so every GPU generates in every population wave"
            )
        states = self._resolve([
            engine.collective_rpc.remote(
                "save_self_exact_reference", args=(),
            )
            for engine in self.engines
        ])
        self._exact_reference_states = list(states)
        identities = [canonical_sha256(state) for state in states]
        if len(set(identities)) != 1:
            raise RuntimeError(
                "engines captured different initial exact weight references"
            )
        self._pending_identity_audit = None
        return result

    def _restore_all_engines_exact(self):
        restored = self._resolve([
            engine.collective_rpc.remote(
                "restore_self_weights_exact", args=(),
            )
            for engine in self.engines
        ])
        if not _all_collective_results(restored, lambda value: value is True):
            raise RuntimeError("an engine did not confirm exact restoration")

    def _evaluate_population_with_anchor(
        self,
        seeds,
        input_batch,
        target_batch,
        domain_sampling_params,
        anchor_items,
        iteration,
    ):
        """Generate domain then anchor requests under each exact perturbation."""
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

        for start in range(0, len(seeds), len(self.engines)):
            engine_batch = seeds[start:start + len(self.engines)]
            if len(engine_batch) != len(self.engines):
                raise ValueError(
                    "partial population wave would leave a GPU unused"
                )
            domain_batches = None
            anchor_batches = None
            try:
                self._resolve([
                    self.engines[index].collective_rpc.remote(
                        "perturb_self_weights",
                        args=(int(seed), self.sigma, False),
                    )
                    for index, seed in enumerate(engine_batch)
                ])
                # These are deliberately separate calls.  In particular, the
                # domain prompt list and SamplingParams object are independent
                # of anchor_items and its length.
                domain_batches = self._resolve([
                    self.engines[index].generate.remote(
                        list(input_batch), domain_sampling_params,
                        use_tqdm=False,
                    )
                    for index, _ in enumerate(engine_batch)
                ])
                if anchor_items:
                    anchor_batches = self._resolve([
                        self.engines[index].generate.remote(
                            list(anchor_prompts), anchor_sampling,
                            use_tqdm=False,
                        )
                        for index, _ in enumerate(engine_batch)
                    ])
            finally:
                self._restore_all_engines_exact()

            if len(domain_batches) != len(engine_batch):
                raise ValueError("domain population engine count changed")
            if anchor_items and len(anchor_batches) != len(engine_batch):
                raise ValueError("anchor population engine count changed")
            for index, seed in enumerate(engine_batch):
                domain_outputs = domain_batches[index]
                if len(domain_outputs) != len(input_batch):
                    raise ValueError(
                        "population engine changed domain request count"
                    )
                metrics = self._postprocess_outputs(
                    domain_outputs, target_batch,
                )
                seeds_perf[int(seed)] = metrics
                results.append({
                    "seed": int(seed),
                    "avg_reward": metrics["avg_reward"],
                })
                if anchor_items:
                    outputs = anchor_batches[index]
                    if len(outputs) != len(anchor_items):
                        raise ValueError(
                            "population engine changed anchor request count"
                        )
                    anchor_scores[int(seed)] = anchor_v1.score_anchor_outputs(
                        anchor_items, outputs,
                    )
        return seeds_perf, anchor_scores, results

    def _identity_probe(
        self, input_batch, domain_sampling, anchor_items, iteration,
    ):
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
        domain_outputs = self._resolve(
            self.engines[0].generate.remote(
                list(input_batch), domain_sampling, use_tqdm=False,
            )
        )
        anchor_outputs = self._resolve(
            self.engines[0].generate.remote(
                [{"prompt_token_ids": item["prompt_token_ids"]}
                 for item in anchor_items],
                anchor_sampling,
                use_tqdm=False,
            )
        )
        return {
            "schema": "eggroll-es-train-only-identity-probe-v2",
            "domain_output_sha256": domain_output_sha256(domain_outputs),
            "anchor_output_sha256": anchor_output_sha256(
                anchor_items, anchor_outputs,
            ),
            "domain_requests": len(input_batch),
            "anchor_requests": len(anchor_items),
        }

    def _persist_identity_audit(self, iteration, audit):
        path = (
            Path(self.logging_dir)
            / f"alpha-zero-identity-audit-iteration-{iteration + 1}.json"
        )
        audit["journal_path"] = str(path)
        _atomic_write_json(path, audit)

    def _persist_anchor_plan(self, plan):
        if getattr(self, "_pending_identity_audit", None) is not None:
            plan["identity_audit"] = self._pending_identity_audit
        return super()._persist_anchor_plan(plan)

    def estimate_step_coefficients(
        self, iteration, seeds, input_text, target_text,
    ):
        """Estimate once, then require exact alpha-zero identity before return."""
        batches = self._iter_minibatches(
            input_text, target_text, self.mini_batch_size,
        )
        try:
            probe_inputs, _ = next(batches)
        except StopIteration as error:
            raise ValueError("identity audit received an empty train batch") from error
        if not probe_inputs:
            raise ValueError("identity audit received an empty train batch")
        selected_anchor = anchor_v1.select_anchor_items(
            self.anchor_items, iteration, self.anchor_items_per_step,
            self.global_seed,
        )
        domain_sampling = self._sampling_params(
            n=self.n_samples,
            seed=(42 if self.global_seed is None else self.global_seed)
            + iteration,
            temperature=self.train_temperature,
            top_p=self.train_top_p,
            max_tokens=self.max_tokens,
        )
        audit = {
            "schema": "eggroll-es-alpha-zero-identity-audit-v2",
            "iteration": int(iteration),
            "status": "running",
            "training_signal": "train_batch_and_train_only_anchor_only",
            "reference_states": self._exact_reference_states,
            "pre_probe": None,
            "post_probe": None,
            "post_reference_checks": None,
            "passed": False,
        }
        self._pending_identity_audit = audit
        self._persist_identity_audit(iteration, audit)
        plan = None
        try:
            audit["pre_probe"] = self._identity_probe(
                probe_inputs, domain_sampling, selected_anchor, iteration,
            )
            self._persist_identity_audit(iteration, audit)
            plan = super().estimate_step_coefficients(
                iteration, seeds, input_text, target_text,
            )
            checks = self._resolve([
                engine.collective_rpc.remote(
                    "verify_self_exact_reference", args=(),
                )
                for engine in self.engines
            ])
            audit["post_reference_checks"] = checks
            if not _all_collective_results(
                checks,
                lambda check: (
                    isinstance(check, dict) and check.get("passed") is True
                ),
            ):
                raise RuntimeError("post-population exact weight check failed")
            audit["post_probe"] = self._identity_probe(
                probe_inputs, domain_sampling, selected_anchor, iteration,
            )
            if audit["post_probe"] != audit["pre_probe"]:
                raise RuntimeError(
                    "post-population alpha-zero evaluation drifted"
                )
            audit["status"] = "passed"
            audit["passed"] = True
            self._persist_identity_audit(iteration, audit)
            plan["identity_audit"] = audit
            self._persist_anchor_plan(plan)
            return plan
        except Exception as error:
            audit["status"] = "failed"
            audit["failure"] = {
                "type": type(error).__name__,
                "message": str(error),
                "update_applied": False,
            }
            self._persist_identity_audit(iteration, audit)
            if plan is not None:
                plan["identity_audit"] = audit
                self._persist_anchor_plan(plan)
            raise

    def apply_seed_coefficients(self, plan, target_alpha):
        audit = plan.get("identity_audit")
        if not isinstance(audit, dict) or audit.get("passed") is not True:
            raise RuntimeError(
                "anchored v2 refuses an update without a passed alpha-zero "
                "identity audit"
            )
        previous_alpha = float(plan["applied_alpha"])
        result = super().apply_seed_coefficients(plan, target_alpha)
        if float(target_alpha) > previous_alpha:
            states = self._resolve([
                engine.collective_rpc.remote(
                    "save_self_exact_reference", args=(),
                )
                for engine in self.engines
            ])
            identities = [canonical_sha256(state) for state in states]
            if len(set(identities)) != 1:
                raise RuntimeError(
                    "engines captured different post-update references"
                )
            self._exact_reference_states = list(states)
            plan["applications"][-1]["exact_reference_states"] = states
            self._persist_anchor_plan(plan)
        return result


def load_trainer():
    """Load the v1 compatibility trainer with the v2 worker extension."""
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parent = _V1_LOAD_TRAINER()

    class ExactRestoredAnchoredEvolutionStrategiesTrainer(
        ExactRestoredAnchoredStepMixin, parent,
    ):
        def launch_engines(
            self, num_engines=4, n_gpu_per_vllm_engine=1,
            model_name="Qwen/Qwen2.5-Math-1.5B", precision="bfloat16",
        ):
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy,
            )
            from es_at_scale.trainer.es_trainer import ESNcclLLM

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
                "worker_extension_cls": WORKER_EXTENSION,
                "dtype": precision,
                "enable_prefix_caching": False,
                "enforce_eager": True,
                "gpu_memory_utilization": 0.82,
                "max_model_len": 2048,
                "limit_mm_per_prompt": {"image": 0, "video": 0},
                "mm_processor_cache_gb": 0,
                "skip_mm_profiling": True,
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

    return ExactRestoredAnchoredEvolutionStrategiesTrainer


def run_exact_steps(trainer, *args, **kwargs):
    summary = _V1_RUN_EXACT_STEPS(trainer, *args, **kwargs)
    summary["schema"] = "eggroll-es-anchored-exact-run-v2"
    summary["anchor"]["restoration"] = {
        "method": "per_engine_cpu_committed_reference_copy",
        "subtractive_restore_forbidden": True,
        "domain_anchor_generate_calls": "separate",
        "alpha_zero_identity_audit_required": True,
        "reference_states": trainer._exact_reference_states,
    }
    summary["anchor"]["identity_audits"] = [
        plan["identity_audit"] for plan in trainer.anchor_step_plans
    ]
    summary["implementation"]["corrected_anchor_trainer"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": file_sha256(Path(__file__).resolve()),
    }
    worker_path = ROOT / "eggroll_es_worker_v2.py"
    summary["implementation"]["exact_worker_extension"] = {
        "path": str(worker_path.resolve()),
        "sha256": file_sha256(worker_path),
    }
    path = Path(trainer.logging_dir) / "run_summary.json"
    _atomic_write_json(path, summary)
    return summary


def main():
    """Reuse the frozen v1 CLI surface while substituting only v2 seams."""
    old_load = anchor_v1.load_trainer
    old_run = anchor_v1.run_exact_steps
    anchor_v1.load_trainer = load_trainer
    anchor_v1.run_exact_steps = run_exact_steps
    try:
        anchor_v1.main()
    finally:
        anchor_v1.load_trainer = old_load
        anchor_v1.run_exact_steps = old_run


if __name__ == "__main__":
    main()
