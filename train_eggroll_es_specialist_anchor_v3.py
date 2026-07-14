#!/usr/bin/env python3
"""Four-engine sharded update recipe for exact-restored anchored EGGROLL-ES.

This new family layers a two-phase distributed update protocol on v2's exact
population restoration and separate domain/anchor generation.  It never edits
the frozen v1/v2 adapters or upstream checkout.
"""

import math
import os
import sys
from pathlib import Path

import train_eggroll_es_specialist_anchor_v2 as anchor_v2
from eggroll_es_worker_v3 import (
    REQUIRED_ENGINE_COUNT,
    canonical_sha256_v3,
    coefficient_sha256_v3,
    seed_shard_v3,
    update_manifest_v3,
    validate_seed_coefficients_v3,
)


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = (
    "eggroll_es_worker_v3.DistributedExactAuditWorkerExtensionV3"
)


def load_anchor_prose(*args, **kwargs):
    return anchor_v2.load_anchor_prose(*args, **kwargs)


def coefficient_sha256(seeds, coefficients):
    return anchor_v2.coefficient_sha256(seeds, coefficients)


def file_sha256(path):
    return anchor_v2.file_sha256(path)


def canonical_sha256(value):
    return anchor_v2.canonical_sha256(value)


def _unwrap_tp1_results_v3(results, expected_count, operation):
    if not isinstance(results, (list, tuple)) or len(results) != expected_count:
        raise RuntimeError(
            f"{operation} did not return one result per v3 engine"
        )
    unwrapped = []
    for engine_index, result in enumerate(results):
        if not isinstance(result, (list, tuple)) or len(result) != 1:
            raise RuntimeError(
                f"{operation} engine {engine_index} did not return exactly "
                "one TP=1 worker result"
            )
        unwrapped.append(result[0])
    return unwrapped


def validate_prepared_shards_v3(
    reports, seeds, coefficients, expected_manifest_sha256,
    reference_generation, expected_base_sha256, expected_update_sequence,
):
    """Bind every prepared rank to the controller's exact shard manifest."""
    engine_count = len(reports)
    if engine_count != REQUIRED_ENGINE_COUNT:
        raise RuntimeError("prepared update did not cover four engines")
    ranks = []
    all_indices = []
    expected_per_rank = len(seeds) // engine_count
    for report in reports:
        if not isinstance(report, dict) or report.get("prepared") is not True:
            raise RuntimeError("an engine did not prepare its update shard")
        if report.get("manifest_sha256") != expected_manifest_sha256:
            raise RuntimeError("prepared update manifests differ")
        if report.get("world_size") != engine_count:
            raise RuntimeError("prepared communicator world size changed")
        if report.get("reference_generation") != reference_generation:
            raise RuntimeError("prepared reference generations differ")
        if report.get("base_sha256") != expected_base_sha256:
            raise RuntimeError("prepared base hashes differ")
        if report.get("update_sequence") != expected_update_sequence:
            raise RuntimeError("prepared update sequences differ")
        preflight = report.get("allocation_preflight")
        if (
            not isinstance(preflight, dict)
            or preflight.get("schema")
            != "eggroll-es-local-allocation-preflight-v3"
            or preflight.get("passed") is not True
            or preflight.get("collectives_created") is not False
            or preflight.get("rng_consumed") is not False
            or preflight.get("weights_changed") is not False
            or preflight.get("scratch_freed_before_collectives") is not True
            or preflight.get("accumulator_dtype") != "torch.float32"
            or not isinstance(preflight.get("parameter_count"), int)
            or preflight["parameter_count"] <= 0
            or not isinstance(preflight.get("largest_parameter_name"), str)
            or not preflight["largest_parameter_name"]
            or not isinstance(preflight.get("largest_parameter_shape"), list)
            or not isinstance(preflight.get("parameter_dtype"), str)
            or not isinstance(
                preflight.get("simulated_peak_temporary_bytes"), int,
            )
            or preflight["simulated_peak_temporary_bytes"] <= 0
        ):
            raise RuntimeError("prepared allocation preflight is unsafe")
        rank = report.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int):
            raise RuntimeError("prepared communicator rank is not an integer")
        ranks.append(rank)
        indices = report.get("shard_indices")
        if not isinstance(indices, list) or len(indices) != expected_per_rank:
            raise RuntimeError("prepared seed shards are not evenly balanced")
        if rank < 0 or rank >= engine_count:
            raise RuntimeError("prepared communicator rank is out of range")
        expected_shard = seed_shard_v3(
            seeds, coefficients, rank, engine_count,
        )
        if indices != expected_shard["indices"]:
            raise RuntimeError("prepared seed shard is not the expected stride")
        if report.get("shard_seeds") != expected_shard["seeds"]:
            raise RuntimeError("prepared seed shard does not match its indices")
        expected_pair_sha = canonical_sha256_v3({
            "seeds": expected_shard["seeds"],
            "coefficients": expected_shard["coefficients"],
        })
        if report.get("shard_pair_sha256") != expected_pair_sha:
            raise RuntimeError("prepared seed/coefficient shard identity differs")
        all_indices.extend(indices)
    if sorted(ranks) != list(range(engine_count)):
        raise RuntimeError("prepared communicator ranks are incomplete")
    if sorted(all_indices) != list(range(len(seeds))):
        raise RuntimeError("prepared seed shards overlap or omit a seed")
    if len(set(all_indices)) != len(all_indices):
        raise RuntimeError("prepared seed shards overlap")
    return True


def validate_executed_updates_v3(reports, expected_manifest_sha256):
    if len(reports) != REQUIRED_ENGINE_COUNT:
        raise RuntimeError("executed update did not cover four engines")
    identities = []
    ranks = []
    parameter_counts = []
    element_counts = []
    for report in reports:
        if not isinstance(report, dict) or report.get("executed") is not True:
            raise RuntimeError("an engine did not execute its collective update")
        if report.get("manifest_sha256") != expected_manifest_sha256:
            raise RuntimeError("executed update manifests differ")
        if report.get("world_size") != REQUIRED_ENGINE_COUNT:
            raise RuntimeError("executed communicator world size changed")
        if report.get("collective_dtype") != "torch.float32":
            raise RuntimeError("distributed update collective was not FP32")
        identity = report.get("final_identity")
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("sha256"), str)
            or not identity["sha256"]
        ):
            raise RuntimeError("distributed update omitted its final identity")
        identities.append(identity)
        ranks.append(report.get("rank"))
        parameter_counts.append(report.get("parameter_count"))
        element_counts.append(report.get("reduced_element_count"))
    if sorted(ranks) != list(range(REQUIRED_ENGINE_COUNT)):
        raise RuntimeError("executed communicator ranks are incomplete")
    if len({canonical_sha256_v3(item) for item in identities}) != 1:
        raise RuntimeError("final distributed model hashes differ")
    if (
        len(set(parameter_counts)) != 1
        or not isinstance(parameter_counts[0], int)
        or parameter_counts[0] <= 0
    ):
        raise RuntimeError("distributed parameter collective counts differ")
    if (
        len(set(element_counts)) != 1
        or not isinstance(element_counts[0], int)
        or element_counts[0] <= 0
    ):
        raise RuntimeError("distributed element collective counts differ")
    return identities[0]


class DistributedAnchoredStepMixinV3:
    """Add balanced all-reduce updates to the exact-restored v2 mixin."""

    def _rpc_all_engines_v3(self, method, args):
        handles = [
            engine.collective_rpc.remote(method, args=args)
            for engine in self.engines
        ]
        results = self._resolve(handles)
        return _unwrap_tp1_results_v3(results, len(self.engines), method)

    def _persist_anchor_plan(self, plan):
        """Persist v3 plan/audit provenance through the v2 atomic writer."""
        if getattr(self, "_pending_identity_audit", None) is not None:
            plan["identity_audit"] = self._pending_identity_audit
        path = (
            Path(self.logging_dir) / "anchor-plan-iteration-"
            f"{plan['iteration'] + 1}.json"
        )
        plan["journal_path"] = str(path)
        anchor_v2._atomic_write_json(path, plan)

    @staticmethod
    def _validate_worker_states_v3(states, require_fresh):
        if len(states) != REQUIRED_ENGINE_COUNT:
            raise RuntimeError("v3 worker state did not cover four engines")
        ranks = []
        generations = []
        references = []
        currents = []
        for state in states:
            if not isinstance(state, dict) or state.get("pending") is not False:
                raise RuntimeError("v3 worker has invalid or pending state")
            communicator = state.get("communicator")
            if (
                not isinstance(communicator, dict)
                or communicator.get("world_size") != REQUIRED_ENGINE_COUNT
                or communicator.get("tp_world_size") != 1
                or communicator.get("available") is not True
                or communicator.get("disabled") is not False
            ):
                raise RuntimeError("v3 communicator state is unsafe")
            ranks.append(communicator.get("rank"))
            generations.append(state.get("reference_generation"))
            reference_identity = state.get("reference_identity")
            current_identity = state.get("current_identity")
            if (
                not isinstance(reference_identity, dict)
                or not isinstance(reference_identity.get("sha256"), str)
                or not reference_identity["sha256"]
                or not isinstance(current_identity, dict)
                or not isinstance(current_identity.get("sha256"), str)
                or not current_identity["sha256"]
            ):
                raise RuntimeError("v3 worker omitted a model identity")
            references.append(reference_identity)
            currents.append(current_identity)
            if bool(state.get("reference_fresh_for_population")) != require_fresh:
                raise RuntimeError("v3 population reference freshness differs")
        if sorted(ranks) != list(range(REQUIRED_ENGINE_COUNT)):
            raise RuntimeError("v3 communicator ranks are incomplete")
        if len(set(generations)) != 1 or not isinstance(generations[0], int):
            raise RuntimeError("v3 reference generations differ")
        if len({canonical_sha256_v3(item) for item in references}) != 1:
            raise RuntimeError("v3 exact reference identities differ")
        if len({canonical_sha256_v3(item) for item in currents}) != 1:
            raise RuntimeError("v3 current model identities differ")
        return {
            "reference_generation": generations[0],
            "reference_identity": references[0],
            "current_identity": currents[0],
        }

    def _set_coordinator_reference_v3(self, summary, fresh):
        self._v3_reference_generation = summary["reference_generation"]
        self._v3_reference_identity = dict(summary["reference_identity"])
        self._v3_current_identity = dict(summary["current_identity"])
        self._v3_reference_fresh = bool(fresh)
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._v3_active_plan_id = None

    def configure_anchor(self, *args, **kwargs):
        result = super().configure_anchor(*args, **kwargs)
        if len(self.engines) != REQUIRED_ENGINE_COUNT:
            raise ValueError("anchored v3 requires exactly four engines")
        if int(self.n_vllm_engines) != REQUIRED_ENGINE_COUNT:
            raise ValueError("anchored v3 engine configuration changed")
        if int(self.n_gpu_per_vllm_engine) != 1:
            raise ValueError("anchored v3 initially requires TP=1")
        if self.population_size % REQUIRED_ENGINE_COUNT != 0:
            raise ValueError(
                "anchored v3 population must be divisible by four engines"
            )
        states = self._rpc_all_engines_v3(
            "inspect_distributed_update_state_v3",
            (REQUIRED_ENGINE_COUNT,),
        )
        summary = self._validate_worker_states_v3(states, require_fresh=True)
        self._set_coordinator_reference_v3(summary, fresh=True)
        return result

    def _refresh_population_references_v3(self):
        """Capture once before a new population, never between alpha probes."""
        states = self._rpc_all_engines_v3(
            "save_self_exact_reference", (),
        )
        # save_self_exact_reference returns a smaller state object; inspect the
        # complete communicator/current state before trusting it.
        if len({canonical_sha256_v3(item) for item in states}) != 1:
            raise RuntimeError("engines captured different refreshed references")
        inspected = self._rpc_all_engines_v3(
            "inspect_distributed_update_state_v3",
            (REQUIRED_ENGINE_COUNT,),
        )
        summary = self._validate_worker_states_v3(
            inspected, require_fresh=True,
        )
        self._set_coordinator_reference_v3(summary, fresh=True)
        self._exact_reference_states = [list([state]) for state in states]

    def _identity_probe(
        self, input_batch, domain_sampling, anchor_items, iteration,
    ):
        """Use all engines for domain prompts; report actual anchor coverage."""
        if len(self.engines) != REQUIRED_ENGINE_COUNT:
            raise RuntimeError("v3 identity probe requires exactly four engines")
        domain_prompts = list(input_batch)
        anchor_prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in anchor_items
        ]
        domain_coverage = min(len(domain_prompts), len(self.engines))
        anchor_coverage = min(len(anchor_prompts), len(self.engines))
        if domain_coverage != REQUIRED_ENGINE_COUNT:
            raise RuntimeError(
                "v3 identity probe domain batch does not cover all four engines"
            )
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
        domain_outputs = anchor_v2.anchor_v1.dispatch_eval_batch(
            self.engines,
            domain_prompts,
            domain_sampling,
            self._resolve,
        )
        anchor_outputs = anchor_v2.anchor_v1.dispatch_eval_batch(
            self.engines,
            anchor_prompts,
            anchor_sampling,
            self._resolve,
        )
        return {
            "schema": "eggroll-es-train-only-identity-probe-v3",
            "domain_output_sha256": anchor_v2.domain_output_sha256(
                domain_outputs,
            ),
            "anchor_output_sha256": anchor_v2.anchor_output_sha256(
                anchor_items, anchor_outputs,
            ),
            "domain_requests": len(domain_prompts),
            "anchor_requests": len(anchor_prompts),
            "dispatch": "strided_engine_shards_separate_calls",
            "engine_coverage": {
                "configured_engines": len(self.engines),
                "domain_nonempty_engines": domain_coverage,
                "anchor_nonempty_engines": anchor_coverage,
                "domain_uses_all_engines": (
                    domain_coverage == len(self.engines)
                ),
                "anchor_uses_all_engines": (
                    anchor_coverage == len(self.engines)
                ),
            },
        }

    def estimate_step_coefficients(
        self, iteration, seeds, input_text, target_text,
    ):
        if not getattr(self, "_v3_reference_fresh", False):
            self._refresh_population_references_v3()
        plan = super().estimate_step_coefficients(
            iteration, seeds, input_text, target_text,
        )
        plan_id = canonical_sha256_v3({
            "schema": "eggroll-es-distributed-plan-id-v3",
            "iteration": int(iteration),
            "coefficient_sha256": plan["coefficient_sha256"],
            "reference_generation": self._v3_reference_generation,
            "reference_sha256": self._v3_reference_identity["sha256"],
        })
        plan["distributed_update_v3"] = {
            "schema": "eggroll-es-distributed-seed-plan-v3",
            "plan_id": plan_id,
            "engine_count": REQUIRED_ENGINE_COUNT,
            "tp_per_engine": 1,
            "reference_generation": self._v3_reference_generation,
            "reference_identity": dict(self._v3_reference_identity),
            "seed_sharding": "strided_by_inter_engine_rank",
            "collective_dtype": "torch.float32",
            "reference_recapture_policy": "once_before_next_population_only",
        }
        self._v3_active_plan_id = plan_id
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._persist_anchor_plan(plan)
        return plan

    def _abort_update_v3(self, plan, failure):
        metadata = plan["distributed_update_v3"]
        aborts = self._rpc_all_engines_v3(
            "abort_distributed_update_v3",
            (metadata["plan_id"], metadata["reference_generation"]),
        )
        if len(aborts) != REQUIRED_ENGINE_COUNT or any(
            not isinstance(item, dict) or item.get("aborted") is not True
            for item in aborts
        ):
            raise RuntimeError("v3 exact-reference abort was not unanimous")
        identities = [item.get("restored_identity") for item in aborts]
        if len({canonical_sha256_v3(item) for item in identities}) != 1:
            raise RuntimeError("v3 abort restored different engine weights")
        expected_reference = metadata["reference_identity"]
        if identities[0] != expected_reference:
            raise RuntimeError("v3 abort did not restore the retained reference")
        plan["distributed_update_v3"]["last_failure"] = {
            "type": type(failure).__name__,
            "message": str(failure),
            "aborted_to_reference": True,
            "restored_identity": identities[0],
        }
        plan["applied_alpha"] = 0.0
        self._v3_current_identity = dict(identities[0])
        self._v3_reference_fresh = True
        self._v3_update_sequence = 0
        self._v3_accepted_alpha = 0.0
        self._v3_active_plan_id = None
        self._persist_anchor_plan(plan)

    def apply_seed_coefficients(self, plan, target_alpha):
        """Apply one monotonic increment using all four model replicas."""
        audit = plan.get("identity_audit")
        if not isinstance(audit, dict) or audit.get("passed") is not True:
            raise RuntimeError(
                "anchored v3 refuses an update without a passed alpha-zero "
                "identity audit"
            )
        if plan is not self._latest_anchor_plan:
            raise ValueError("only the latest v3 seed plan may be used")
        seeds, coefficients = validate_seed_coefficients_v3(
            plan.get("seeds", []),
            plan.get("coefficients", []),
            self.population_size,
            REQUIRED_ENGINE_COUNT,
        )
        coefficient_identity = coefficient_sha256_v3(seeds, coefficients)
        if (
            coefficient_identity != plan.get("coefficient_sha256")
            or coefficient_identity
            != coefficient_sha256(plan.get("seeds", []), plan.get("coefficients", []))
        ):
            raise ValueError("v3 seed plan coefficient integrity failed")
        metadata = plan.get("distributed_update_v3")
        if (
            not isinstance(metadata, dict)
            or metadata.get("plan_id") != self._v3_active_plan_id
            or metadata.get("reference_generation")
            != self._v3_reference_generation
        ):
            raise RuntimeError("v3 distributed seed plan is stale")
        target_alpha = float(target_alpha)
        previous_alpha = float(plan["applied_alpha"])
        if not math.isfinite(target_alpha) or target_alpha < previous_alpha:
            raise ValueError("target alpha must be finite and monotonic")
        if target_alpha == previous_alpha:
            return plan
        if previous_alpha != self._v3_accepted_alpha:
            raise RuntimeError("coordinator alpha state is stale")
        if self._v3_update_sequence == 0 and not self._v3_reference_fresh:
            raise RuntimeError("first v3 update has a stale population reference")
        if self._v3_update_sequence > 0 and self._v3_reference_fresh:
            raise RuntimeError("continued v3 update has an invalid fresh reference")
        update_sequence = self._v3_update_sequence + 1
        expected_base_sha = self._v3_current_identity.get("sha256")
        if not expected_base_sha:
            raise RuntimeError("coordinator has no current model identity")

        prepared = None
        expected_manifest = update_manifest_v3(
            coefficient_sha256=coefficient_identity,
            population_size=self.population_size,
            world_size=REQUIRED_ENGINE_COUNT,
            reference_generation=self._v3_reference_generation,
            plan_id=metadata["plan_id"],
            update_sequence=update_sequence,
            previous_alpha=previous_alpha,
            target_alpha=target_alpha,
            expected_base_sha256=expected_base_sha,
        )
        manifest_sha = canonical_sha256_v3(expected_manifest)
        try:
            prepared = self._rpc_all_engines_v3(
                "prepare_sharded_seed_update_v3",
                (
                    seeds,
                    coefficients,
                    coefficient_identity,
                    self.population_size,
                    REQUIRED_ENGINE_COUNT,
                    self._v3_reference_generation,
                    metadata["plan_id"],
                    update_sequence,
                    previous_alpha,
                    target_alpha,
                    expected_base_sha,
                ),
            )
            if any(
                not isinstance(item, dict)
                or item.get("manifest_sha256") != manifest_sha
                for item in prepared
            ):
                raise RuntimeError(
                    "prepared manifest differs from controller expectation"
                )
            validate_prepared_shards_v3(
                prepared,
                seeds,
                coefficients,
                manifest_sha,
                self._v3_reference_generation,
                expected_base_sha,
                update_sequence,
            )
            executed = self._rpc_all_engines_v3(
                "execute_prepared_seed_update_v3", (manifest_sha,),
            )
            final_identity = validate_executed_updates_v3(
                executed, manifest_sha,
            )
            committed = self._rpc_all_engines_v3(
                "commit_prepared_seed_update_v3",
                (manifest_sha, final_identity["sha256"]),
            )
            if (
                len(committed) != REQUIRED_ENGINE_COUNT
                or any(
                    not isinstance(item, dict)
                    or item.get("committed") is not True
                    or item.get("manifest_sha256") != manifest_sha
                    or item.get("final_sha256") != final_identity["sha256"]
                    or item.get("reference_fresh_for_population") is not False
                    or item.get("update_sequence") != update_sequence
                    for item in committed
                )
                or sorted(item["rank"] for item in committed)
                != list(range(REQUIRED_ENGINE_COUNT))
            ):
                raise RuntimeError("distributed update commit was not unanimous")
            post_commit_states = self._rpc_all_engines_v3(
                "inspect_distributed_update_state_v3",
                (REQUIRED_ENGINE_COUNT,),
            )
            post_commit = self._validate_worker_states_v3(
                post_commit_states, require_fresh=False,
            )
            if (
                post_commit["reference_generation"]
                != self._v3_reference_generation
                or post_commit["current_identity"] != final_identity
                or any(
                    state.get("update_session") != metadata["plan_id"]
                    or state.get("update_sequence") != update_sequence
                    or state.get("accepted_alpha") != target_alpha
                    for state in post_commit_states
                )
            ):
                raise RuntimeError("post-commit distributed state differs")
        except Exception as error:
            # Prepared RPCs are side-effect free with respect to weights, but
            # aborting them as well makes a partially completed Ray result safe.
            try:
                self._abort_update_v3(plan, error)
            except Exception as abort_error:
                raise RuntimeError(
                    f"distributed update failed ({error}); exact abort also "
                    f"failed ({abort_error})"
                ) from error
            raise

        self._v3_current_identity = dict(final_identity)
        self._v3_reference_fresh = False
        self._v3_update_sequence = update_sequence
        self._v3_accepted_alpha = target_alpha
        plan["applied_alpha"] = target_alpha
        application = {
            "schema": "eggroll-es-distributed-alpha-application-v3",
            "target_alpha": target_alpha,
            "alpha_increment": target_alpha - previous_alpha,
            "update_sequence": update_sequence,
            "manifest_sha256": manifest_sha,
            "manifest": expected_manifest,
            "coefficient_sha256": coefficient_identity,
            "prepared_shards": prepared,
            "executed_collectives": executed,
            "commits": committed,
            "post_commit_states": post_commit_states,
            "final_identity": final_identity,
            "reference_recaptured": False,
            "reference_fresh_for_population": False,
            "bf16_alpha_semantics": "path_dependent_monotonic_increment",
            "direct_alpha_confirmation_required": True,
        }
        plan["applications"].append(application)
        self._persist_anchor_plan(plan)
        return plan


def load_trainer():
    """Load v2 exact restoration with the v3 worker/update coordinator."""
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    parent = anchor_v2.load_trainer()

    class DistributedExactRestoredAnchoredTrainerV3(
        DistributedAnchoredStepMixinV3, parent,
    ):
        def launch_engines(
            self,
            num_engines=4,
            n_gpu_per_vllm_engine=1,
            model_name="Qwen/Qwen2.5-Math-1.5B",
            precision="bfloat16",
        ):
            if num_engines != REQUIRED_ENGINE_COUNT:
                raise ValueError("anchored v3 requires exactly four engines")
            if n_gpu_per_vllm_engine != 1:
                raise ValueError("anchored v3 initially requires TP=1")
            import ray
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy,
            )
            from es_at_scale.trainer.es_trainer import ESNcclLLM

            pgs = [
                placement_group(
                    [{"GPU": 1, "CPU": 0}],
                    strategy="PACK",
                    lifetime="detached",
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
                "tensor_parallel_size": 1,
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
            engines = [
                ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=strategy,
                )(ESNcclLLM).remote(**engine_args)
                for strategy in strategies
            ]
            return engines, pgs

    return DistributedExactRestoredAnchoredTrainerV3


def run_exact_steps(trainer, *args, **kwargs):
    summary = anchor_v2.run_exact_steps(trainer, *args, **kwargs)
    summary["schema"] = "eggroll-es-anchored-exact-run-v3"
    summary["anchor"]["distributed_update_v3"] = {
        "engine_count": REQUIRED_ENGINE_COUNT,
        "tp_per_engine": 1,
        "seed_sharding": "strided_by_inter_engine_rank",
        "collective_dtype": "torch.float32",
        "two_phase_commit": True,
        "final_hash_consensus_required": True,
        "reference_recapture_policy": "once_before_next_population_only",
        "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
        "direct_alpha_confirmation_required": True,
    }
    summary["implementation"]["distributed_anchor_trainer_v3"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": file_sha256(Path(__file__).resolve()),
    }
    worker_path = ROOT / "eggroll_es_worker_v3.py"
    summary["implementation"]["distributed_worker_extension_v3"] = {
        "path": str(worker_path.resolve()),
        "sha256": file_sha256(worker_path),
    }
    path = Path(trainer.logging_dir) / "run_summary.json"
    anchor_v2._atomic_write_json(path, summary)
    return summary


def main():
    """Reuse the frozen v1 CLI while substituting the v3 implementation."""
    anchor_v1 = anchor_v2.anchor_v1
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
