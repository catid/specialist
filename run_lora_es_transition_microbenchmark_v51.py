#!/usr/bin/env python3
"""Launch-sealed V51 train-only population transition microbenchmark."""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import lora_es_direct_master_pipeline_v51 as pipeline
import lora_es_transition_microbenchmark_v51 as planning
import run_lora_es_generation_boundary_v48b as v48b


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v51_direct_pinned_master_transition_microbenchmark"
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT
).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "transition_microbenchmark_report_v51.json").resolve()
FAILURE = (RUN_DIR / "failure_v51.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v51.jsonl").resolve()
TIMING_ARTIFACT = (RUN_DIR / "per_state_timing_v51.json").resolve()
POPULATION_ARTIFACT = (RUN_DIR / "population_v51.json").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_direct_pinned_master_transition_v51.json"
).resolve()
PARENT_PREREGISTRATION = v48b.PREREGISTRATION
PARENT_PREREGISTRATION_FILE_SHA256 = (
    "34e19fe84ff061b98a8627f07daab59f5cbb8c718668fc479454114fef67c3d0"
)
PARENT_PREREGISTRATION_CONTENT_SHA256 = (
    "4d5e17a07551377f0ef39c3dfa306fda68b9b669eeafd3c78ebbfff894072d1e"
)
WORKER_EXTENSION_V51 = (
    "eggroll_es_worker_lora_v51.LoRAAdapterStateWorkerExtensionV51"
)
SCORER_ACTORS_V51 = 4

v43i = v48b.v43i
v40a = v43i.v40a
numeric = v43i.numeric


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _load_parent_v51() -> dict:
    if (
        planning.file_sha256_v51(PARENT_PREREGISTRATION)
        != PARENT_PREREGISTRATION_FILE_SHA256
    ):
        raise RuntimeError("v51 frozen V48B parent file changed")
    parent = json.loads(PARENT_PREREGISTRATION.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in parent.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        parent.get("content_sha256_before_self_field")
        != PARENT_PREREGISTRATION_CONTENT_SHA256
        or planning.canonical_sha256_v51(compact)
        != PARENT_PREREGISTRATION_CONTENT_SHA256
        or parent.get("protected_semantics_opened") is not False
        or parent.get("shadow_ood_holdout_or_benchmark_opened") is not False
    ):
        raise RuntimeError("v51 frozen V48B parent content changed")
    return parent


def implementation_bindings_v51() -> dict[str, str]:
    paths = {
        "runtime_v51": Path(__file__).resolve(),
        "planning_v51": Path(planning.__file__).resolve(),
        "pipeline_v51": Path(pipeline.__file__).resolve(),
        "worker_v51": ROOT / "eggroll_es_worker_lora_v51.py",
        "builder_v51": (
            ROOT / "build_lora_es_transition_microbenchmark_preregistration_v51.py"
        ),
        "tests_v51": ROOT / "test_lora_es_transition_microbenchmark_v51.py",
        "runtime_v48b": Path(v48b.__file__).resolve(),
        "parent_preregistration_v48b": PARENT_PREREGISTRATION,
    }
    return {
        label: planning.file_sha256_v51(path)
        for label, path in paths.items()
    }


def load_preregistration_v51(args) -> tuple[dict, dict]:
    path = Path(args.preregistration).resolve()
    if planning.file_sha256_v51(path) != args.preregistration_sha256:
        raise RuntimeError("v51 preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    parent = _load_parent_v51()
    design = planning.build_design_v51()
    expected_recipe = dict(parent["recipe"])
    expected_recipe["worker_extension"] = WORKER_EXTENSION_V51
    artifacts = {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "report": str(REPORT),
        "failure": str(FAILURE),
        "gpu_log": str(GPU_LOG),
        "timing": str(TIMING_ARTIFACT),
        "population": str(POPULATION_ARTIFACT),
        "snapshot": None,
    }
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or planning.canonical_sha256_v51(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "matched-lora-es-direct-pinned-master-transition-preregistration-v51"
        or value.get("status") != "preregistered_before_train_only_launch"
        or value.get("gpu_launch_authorized") is not True
        or value.get("microbenchmark_only") is not True
        or value.get("optimizer_update_authorized") is not False
        or value.get("quality_selection_or_promotion_authorized") is not False
        or value.get("protected_semantic_access_authorized") is not False
        or value.get("shadow_ood_holdout_or_benchmark_authorized") is not False
        or value.get("recipe") != expected_recipe
        or value.get("design") != design
        or value.get("artifacts") != artifacts
        or value.get("implementation_bindings")
        != implementation_bindings_v51()
        or value.get("required_gates", {}).get(
            "all_16x4x5_timing_receipts"
        ) is not True
        or value.get("required_gates", {}).get(
            "all_16_candidate_and_runtime_identities_exact"
        ) is not True
        or value.get("required_gates", {}).get(
            "final_all_four_exact_master_runtime_restore"
        ) is not True
        or value.get("required_gates", {}).get(
            "strict_cleanup_and_gpu_idle"
        ) is not True
    ):
        raise RuntimeError("v51 preregistration contract changed")
    recipe = value["recipe"]
    subset = v48b.load_subset_v48b(
        Path(recipe["subset"]), recipe["subset_file_sha256"],
        recipe["subset_content_sha256"],
    )
    if subset["request_order_sha256"] != recipe["request_order_sha256"]:
        raise RuntimeError("v51 frozen V48B request order changed")
    v48b._SEALED_SUBSET = subset
    return value, design


@contextmanager
def patched_runtime_v51():
    values = {
        "EXPERIMENT": EXPERIMENT,
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "REPORT": REPORT,
        "GPU_LOG": GPU_LOG,
        "WORKER_EXTENSION": WORKER_EXTENSION_V51,
    }
    saved = {key: getattr(v43i, key) for key in values}
    for key, item in values.items():
        setattr(v43i, key, item)
    try:
        yield
    finally:
        for key, item in saved.items():
            setattr(v43i, key, item)


def _controller_timing_v51(started_ns: int, ended_ns: int) -> dict:
    return {
        "clock": "controller_monotonic_ns",
        "started_ns": started_ns,
        "ended_ns": ended_ns,
        "elapsed_ns": ended_ns - started_ns,
    }


class PopulationScoringActorV51:
    """GPU-free scorer with per-actor monotonic and process timing."""

    def __init__(
        self, actor_rank, bundle, dense_items, plan, anchors,
        prepared_fragile, sealed_subset,
    ):
        import ray

        gpu_ids = list(ray.get_gpu_ids())
        if gpu_ids or os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            raise RuntimeError("v51 CPU scoring actor received GPU visibility")
        self.actor_rank = int(actor_rank)
        if self.actor_rank not in range(SCORER_ACTORS_V51):
            raise RuntimeError("v51 invalid scoring actor rank")
        self.bundle = bundle
        self.dense_items = dense_items
        self.plan = plan
        self.anchors = anchors
        self.prepared_fragile = prepared_fragile
        self.sealed_subset = sealed_subset
        self.gpu_ids = gpu_ids

    def runtime_identity_v51(self) -> dict:
        return {
            "schema": "timed-generation-boundary-cpu-scorer-v51",
            "actor_rank": self.actor_rank,
            "gpu_ids": self.gpu_ids,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def score_v51(self, state: dict, batch: list) -> dict:
        expected = dict(state)
        expected["actor_rank"] = self.actor_rank
        if state != expected or len(batch) != 608:
            raise RuntimeError("v51 CPU scorer input coverage changed")
        started_ns = time.monotonic_ns()
        process_started_ns = time.process_time_ns()
        saved_fragile = v48b._PREPARED_FRAGILE
        saved_subset = v48b._SEALED_SUBSET
        v48b._PREPARED_FRAGILE = self.prepared_fragile
        v48b._SEALED_SUBSET = self.sealed_subset
        try:
            scored = v48b.score_fused_outputs_v48b(
                self.plan, batch, self.anchors, v43i.anchor_v4,
                domain_scorer=lambda outputs: v43i.score_batch_detailed_v43i(
                    self.bundle, self.dense_items, outputs,
                ),
            )
        finally:
            v48b._PREPARED_FRAGILE = saved_fragile
            v48b._SEALED_SUBSET = saved_subset
        process_ended_ns = time.process_time_ns()
        ended_ns = time.monotonic_ns()
        return {
            "schema": "timed-generation-boundary-actor-score-v51",
            "state": expected,
            "gpu_ids": self.gpu_ids,
            "score": {
                "actor_rank": self.actor_rank,
                **v48b.compact_signed_score_v48b(scored),
            },
            "timing": {
                "actor_rank": self.actor_rank,
                "state_index": state["state_index"],
                "clock": "worker_monotonic_ns",
                "started_ns": started_ns,
                "ended_ns": ended_ns,
                "elapsed_ns": ended_ns - started_ns,
                "process_elapsed_ns": process_ended_ns - process_started_ns,
            },
        }


class RayPopulationOperationsV51:
    def __init__(
        self, trainer, prereg, bundle, dense_items, plan, params, anchors,
        master_sha: str, master_runtime_sha: str,
    ) -> None:
        import ray

        if v48b._PREPARED_FRAGILE is None or v48b._SEALED_SUBSET is None:
            raise RuntimeError("v51 immutable scoring inputs are incomplete")
        self.ray = ray
        self.trainer = trainer
        self.prereg = prereg
        self.plan = plan
        self.params = params
        self.master_sha = master_sha
        self.master_runtime_sha = master_runtime_sha
        self.expected_states = prereg["design"]["selected_transition"][
            "expected_states"
        ]
        actor_type = ray.remote(
            num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0,
        )(PopulationScoringActorV51)
        self.scorers = [
            actor_type.options(runtime_env={
                "env_vars": {"CUDA_VISIBLE_DEVICES": ""},
            }).remote(
                rank, bundle, dense_items, plan, anchors,
                list(v48b._PREPARED_FRAGILE), dict(v48b._SEALED_SUBSET),
            )
            for rank in range(SCORER_ACTORS_V51)
        ]
        identities = ray.get([
            scorer.runtime_identity_v51.remote() for scorer in self.scorers
        ])
        if identities != [{
            "schema": "timed-generation-boundary-cpu-scorer-v51",
            "actor_rank": rank,
            "gpu_ids": [],
            "cuda_visible_devices": "",
        } for rank in range(SCORER_ACTORS_V51)]:
            self.close()
            raise RuntimeError("v51 CPU scorer identities changed")

    @staticmethod
    def _state_tag(state: pipeline.SignedStateV51) -> dict:
        return {
            "state_index": state.state_index,
            "direction": state.direction,
            "label": state.label,
            "sign": state.sign,
        }

    def transition(self, state, previous) -> dict:
        expected_previous = (
            self.master_sha if previous is None
            else previous.candidate_identity_sha256
        )
        started_ns = time.monotonic_ns()
        values = self.trainer._resolve([
            self.trainer.engines[rank].collective_rpc.remote(
                "transition_antithetic_from_pinned_master_v51",
                args=(
                    state.state_index, state.seed, v43i.SIGMA, state.sign,
                    self.master_sha, expected_previous,
                    state.candidate_identity_sha256,
                    state.runtime_values_sha256,
                ),
            )
            for rank in range(SCORER_ACTORS_V51)
        ])
        ended_ns = time.monotonic_ns()
        if len(values) != SCORER_ACTORS_V51 or any(
            len(item) != 1 for item in values
        ):
            raise RuntimeError("v51 transition actor coverage changed")
        actors = []
        for rank, value in enumerate(values):
            receipt = dict(value[0])
            receipt["actor_rank"] = rank
            receipt["elapsed_ns"] = receipt["timing"]["elapsed_ns"]
            if (
                receipt.get("state_index") != state.state_index
                or receipt.get("master_identity", {}).get("sha256")
                != self.master_sha
                or receipt.get("candidate_identity", {}).get("sha256")
                != state.candidate_identity_sha256
                or receipt.get("materialization", {}).get(
                    "runtime_values_sha256"
                ) != state.runtime_values_sha256
                or receipt.get("direct_from_pinned_fp32_master") is not True
            ):
                raise RuntimeError("v51 direct transition identity changed")
            actors.append(receipt)
        return {
            "schema": "four-actor-direct-transition-timing-v51",
            "state_index": state.state_index,
            "actors": actors,
            "controller": _controller_timing_v51(started_ns, ended_ns),
        }

    def launch_generation(self, state):
        started_ns = time.monotonic_ns()
        refs = [
            self.trainer.engines[rank].generate.remote(
                self.plan["requests"], self.params, use_tqdm=False,
                lora_request=v43i._lora_request(),
            )
            for rank in range(SCORER_ACTORS_V51)
        ]
        return {"refs": refs, "started_ns": started_ns}

    def wait_generation(self, state, handles) -> dict:
        refs = handles.get("refs", [])
        if len(refs) != SCORER_ACTORS_V51:
            raise RuntimeError("v51 generation handle coverage changed")
        by_ref = {ref: rank for rank, ref in enumerate(refs)}
        pending = list(refs)
        ready_ns = {}
        while pending:
            ready, pending = self.ray.wait(
                pending, num_returns=1, fetch_local=False,
            )
            ready_ns[by_ref[ready[0]]] = time.monotonic_ns()
        ended_ns = max(ready_ns.values())
        actors = [{
            "schema": "controller-observed-generation-timing-v51",
            "state_index": state.state_index,
            "actor_rank": rank,
            "clock": "controller_monotonic_ns",
            "started_ns": handles["started_ns"],
            "ended_ns": ready_ns[rank],
            "elapsed_ns": ready_ns[rank] - handles["started_ns"],
        } for rank in range(SCORER_ACTORS_V51)]
        return {
            "schema": "four-actor-generation-timing-v51",
            "state_index": state.state_index,
            "actors": actors,
            "controller": _controller_timing_v51(
                handles["started_ns"], ended_ns,
            ),
        }

    def submit_scoring(self, state, generation_handles):
        refs = generation_handles["refs"]
        return [
            self.scorers[rank].score_v51.remote(
                {**self._state_tag(state), "actor_rank": rank}, refs[rank],
            )
            for rank in range(SCORER_ACTORS_V51)
        ]

    def resolve_scoring(self, state, scoring_handles):
        if len(scoring_handles) != SCORER_ACTORS_V51:
            raise RuntimeError("v51 scoring handle coverage changed")
        started_ns = time.monotonic_ns()
        by_ref = {ref: rank for rank, ref in enumerate(scoring_handles)}
        pending = list(scoring_handles)
        ready_ns = {}
        while pending:
            ready, pending = self.ray.wait(
                pending, num_returns=1, fetch_local=False,
            )
            ready_ns[by_ref[ready[0]]] = time.monotonic_ns()
        rows = self.ray.get(scoring_handles)
        actors = []
        scores = []
        for rank, row in enumerate(rows):
            if (
                row.get("schema")
                != "timed-generation-boundary-actor-score-v51"
                or row.get("state") != {
                    **self._state_tag(state), "actor_rank": rank,
                }
                or row.get("gpu_ids") != []
                or row.get("score", {}).get("actor_rank") != rank
                or row.get("timing", {}).get("actor_rank") != rank
            ):
                raise RuntimeError("v51 scoring receipt changed")
            timing = dict(row["timing"])
            timing["schema"] = "worker-score-timing-v51"
            actors.append(timing)
            scores.append(row["score"])
        score_timing = {
            "schema": "four-actor-score-timing-v51",
            "state_index": state.state_index,
            "actors": actors,
        }
        drain_actors = [{
            "schema": "controller-observed-drain-timing-v51",
            "state_index": state.state_index,
            "actor_rank": rank,
            "clock": "controller_monotonic_ns",
            "started_ns": started_ns,
            "ended_ns": ready_ns[rank],
            "elapsed_ns": ready_ns[rank] - started_ns,
        } for rank in range(SCORER_ACTORS_V51)]
        drain_timing = {
            "schema": "four-actor-drain-timing-v51",
            "state_index": state.state_index,
            "actors": drain_actors,
            "controller": _controller_timing_v51(
                started_ns, max(ready_ns.values()),
            ),
        }
        return scores, score_timing, drain_timing

    def final_restore(self, reason: str) -> dict:
        started_ns = time.monotonic_ns()
        values = v40a._rpc_all(
            self.trainer, "restore_pinned_master_v51",
            (self.master_sha, self.master_runtime_sha, reason),
        )
        ended_ns = time.monotonic_ns()
        actors = []
        for rank, value in enumerate(values):
            receipt = dict(value)
            receipt["state_index"] = 15
            receipt["actor_rank"] = rank
            receipt["elapsed_ns"] = receipt["timing"]["elapsed_ns"]
            if (
                receipt.get("restored") is not True
                or receipt.get("restored_identity", {}).get("sha256")
                != self.master_sha
                or receipt.get("materialization", {}).get(
                    "runtime_values_sha256"
                ) != self.master_runtime_sha
                or receipt.get("transaction_state_quiescent") is not True
            ):
                raise RuntimeError("v51 final exact restore changed")
            actors.append(receipt)
        if len(actors) != SCORER_ACTORS_V51:
            raise RuntimeError("v51 final restore actor coverage changed")
        return {
            "schema": "four-actor-exact-final-restore-v51",
            "mode": "actual_exact_final_restore",
            "reason": reason,
            "actors": actors,
            "controller": _controller_timing_v51(started_ns, ended_ns),
        }

    def cancel_scoring(self, _state, scoring_handles) -> None:
        for handle in scoring_handles:
            self.ray.cancel(handle, force=False)

    def close(self) -> None:
        for scorer in getattr(self, "scorers", []):
            self.ray.kill(scorer, no_restart=True)

    def operations(self) -> pipeline.PipelineOperationsV51:
        return pipeline.PipelineOperationsV51(
            transition=self.transition,
            launch_generation=self.launch_generation,
            wait_generation=self.wait_generation,
            submit_scoring=self.submit_scoring,
            resolve_scoring=self.resolve_scoring,
            final_restore=self.final_restore,
            cancel_scoring=self.cancel_scoring,
        )


def replicated_population_v51(
    trainer, prereg, bundle, dense_items, requests, anchors,
    master_sha: str, master_runtime_sha: str,
) -> tuple[dict, dict]:
    plan = v48b.fused_requests_v48b(requests, anchors)
    params = v48b.sampling_params_for_plan_v48b(plan)
    if len(plan["requests"]) != 608 or len(params) != 608:
        raise RuntimeError("v51 fused population request coverage changed")
    states = pipeline.signed_states_v51(
        prereg["design"]["selected_transition"]["expected_states"]
    )
    runtime = RayPopulationOperationsV51(
        trainer, prereg, bundle, dense_items, plan, params, anchors,
        master_sha, master_runtime_sha,
    )
    try:
        completed, final_restore = pipeline.run_direct_master_pipeline_v51(
            states, runtime.operations(),
        )
    finally:
        runtime.close()
    coverage = pipeline.validate_complete_timing_v51(
        completed, final_restore,
    )

    scores = {
        label: [[None] * 4 for _ in range(8)]
        for label in ("plus", "minus")
    }
    perturbations = {
        label: [[None] * 4 for _ in range(8)]
        for label in ("plus", "minus")
    }
    receipts = []
    state_timings = []
    for item in completed:
        state = item.state
        if len(item.actor_scores) != 4:
            raise RuntimeError("v51 completed actor score coverage changed")
        for rank in range(4):
            scores[state.label][state.direction][rank] = item.actor_scores[rank]
            perturbations[state.label][state.direction][rank] = (
                item.transition["actors"][rank]
            )
            receipts.append({
                "direction": state.direction,
                "sign": state.label,
                "actor_rank": rank,
                "subset_content_sha256": v48b._SEALED_SUBSET["subset"][
                    "content_sha256_before_self_field"
                ],
                "request_order_sha256": v48b._SEALED_SUBSET[
                    "request_order_sha256"
                ],
                "generation_params": dict(
                    v48b.boundary.GENERATION_PARAMS_V48A
                ),
            })
        state_timings.append({
            "state": {
                key: getattr(state, key) for key in (
                    "state_index", "direction", "label", "sign", "seed",
                    "candidate_identity_sha256", "runtime_values_sha256",
                )
            },
            "materialize": item.transition,
            "generate": item.generation,
            "score": item.score_timing,
            "restore": item.restore_timing,
            "drain": item.drain_timing,
        })

    common = v48b.boundary.assert_common_random_plan_v48a(
        receipts, v48b._SEALED_SUBSET["subset"],
    )
    paths = {
        "domain": ("domain", "aggregate", "equal_unit_mean"),
        "prose_lm": ("prose_lm", "mean_token_logprob"),
        "qa_answer_logprob": (
            "qa_answer_logprob", "mean_example_logprob",
        ),
        "fragile_generation_f1": (
            "fragile_generation", "equal_conflict_unit_mean_f1",
        ),
    }
    sign_scores = {
        objective: {
            label: [[float(v43i._nested_value(
                scores[label][direction][rank], path,
            )) for rank in range(4)] for direction in range(8)]
            for label in ("plus", "minus")
        }
        for objective, path in paths.items()
    }
    direct = v48b.boundary.direct_generation_objective_v48a(
        sign_scores["domain"], sign_scores["fragile_generation_f1"],
        sign_scores["prose_lm"], sign_scores["qa_answer_logprob"],
    )
    population = {
        "schema": "direct-pinned-master-population-v51",
        "assignments": [
            numeric.complete_actor_assignments_v43g(direction)
            for direction in range(8)
        ],
        "signed_scores": scores,
        "objective_sign_scores": sign_scores,
        "objective_fitness": direct["objective_fitness"],
        "central_replicates": v43i._central_replicates(
            sign_scores["domain"]
        ),
        "coefficients": direct["projection"]["coefficients"],
        "projection": direct["projection"],
        "common_random_plan": common,
        "fused_requests_per_actor_state": 608,
        "perturbation_certificates": perturbations,
        "intermediate_master_restores_eliminated": 15,
        "actual_final_restore_actor_receipts": 4,
        "all_expected_state_identities_passed": True,
        "final_exact_restore_passed": True,
    }
    timing = {
        "schema": "direct-pinned-master-per-state-timing-v51",
        "status": "complete_before_any_optimizer_update",
        "coverage": coverage,
        "states": state_timings,
        "final_restore": final_restore,
        "historical_train_only_baseline": prereg["design"][
            "historical_train_only_baseline"
        ],
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    return population, timing


def _persist_v51(path: Path, value: dict) -> dict:
    return v43i._persist_phase(path, value)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg, design = load_preregistration_v51(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "microbenchmark_only": True,
            "optimizer_update_authorized": False,
            "direct_transition_states": 16,
            "intermediate_restores_eliminated": 15,
            "expected_state_inventory_sha256": (
                planning.EXPECTED_STATE_INVENTORY_SHA256_V51
            ),
            "historical_train_only_artifacts_loaded": True,
            "train_semantics_loaded": False,
            "model_or_gpu_loaded": False,
            "protected_semantic_access_count": 0,
            "shadow_ood_holdout_or_benchmark_opened": False,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v51 requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "direct-pinned-master-transition-attempt-v51",
        "status": "launching",
        "phase": "before_train_semantics_model_or_gpu_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "protected_semantics_opened": False,
    })
    v40a.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    master = master_runtime_sha = None
    emergency_restore = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    try:
        with patched_runtime_v51():
            bundle = v48b.load_train_bundle_v48b()
            if bundle["content_sha256_before_self_field"] != (
                v48b.EXPECTED_TRAIN_BUNDLE_CONTENT_SHA256_V48B
            ):
                raise RuntimeError("v51 frozen train bundle changed")
            bundle = v48b.augment_unit_membership_v48b(bundle)
            anchor_bundle = v43i.fused.load_anchor_bundle_v43i(
                v43i.PROSE_ANCHOR, v43i.PROSE_REPORT,
                v43i.QA_ANCHOR, v43i.QA_REPORT,
            )
            v40a.base.set_seed(v43i.GLOBAL_SEED)
            trainer, saved = v43i._make_trainer(prereg)
            actor_ids = trainer._resolve([
                engine.runtime_identity_v40a.remote()
                for engine in trainer.engines
            ])
            pid_map = v43i.prior._actor_pid_map(actor_ids)
            monitor = threading.Thread(
                target=v40a.monitor_gpus,
                args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
            )
            monitor.start()
            dense_items, requests, panel_anchors, _full = v48b.prepare_v48b(
                trainer, bundle, anchor_bundle,
            )
            phase.value = "activate_matched_initialization"
            preinstall = v43i._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            phase.value = "install_canonical_master"
            installs = v40a._rpc_all(
                trainer, "install_adapter_state_v41a", (
                    str(v43i.SOURCE_WEIGHTS), str(v43i.SOURCE_CONFIG),
                    v40a.file_sha256(v43i.SOURCE_WEIGHTS),
                    v40a.file_sha256(v43i.SOURCE_CONFIG),
                ),
            )
            certificates = v40a._rpc_all(
                trainer, "adapter_state_certificate_v41a",
            )
            masters = [item["current_identity"] for item in certificates]
            runtimes = {
                item["materialization"]["runtime_values_sha256"]
                for item in certificates
            }
            if (
                any(item["sha256"] != planning.EXPECTED_MASTER_SHA256_V51
                    for item in masters)
                or runtimes != {planning.EXPECTED_MASTER_RUNTIME_SHA256_V51}
            ):
                raise RuntimeError("v51 installed master identity changed")
            master = masters[0]
            master_runtime_sha = next(iter(runtimes))
            references = v40a._rpc_all(
                trainer, "capture_adapter_reference_v41a",
            )
            postinstall = v43i._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            if preinstall["consensus"] != postinstall["consensus"]:
                raise RuntimeError("v51 canonical install changed base score")

            phase.value = "direct_pinned_master_population_v51"
            population, timing = replicated_population_v51(
                trainer, prereg, bundle, dense_items, requests, panel_anchors,
                master["sha256"], master_runtime_sha,
            )
            timing_artifact = _persist_v51(TIMING_ARTIFACT, timing)
            population_artifact = _persist_v51(POPULATION_ARTIFACT, {
                "schema": "direct-pinned-master-population-artifact-v51",
                "status": "complete_without_optimizer_update",
                "population": population,
                "timing_content_sha256": timing_artifact[
                    "content_sha256_before_self_field"
                ],
                "protected_semantics_opened": False,
                "shadow_ood_holdout_or_benchmark_opened": False,
            })
            phase.value = "verify_final_exact_master"
            post_population = v43i._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            final_certificates = v40a._rpc_all(
                trainer, "adapter_state_certificate_v41a",
            )
            if (
                post_population["consensus"] != postinstall["consensus"]
                or any(
                    item["current_identity"] != master
                    or item["materialization"]["runtime_values_sha256"]
                    != master_runtime_sha
                    for item in final_certificates
                )
            ):
                raise RuntimeError("v51 final master reconstruction changed")

            stop.set()
            monitor.join(timeout=10)
            if monitor.is_alive() or not failures.empty():
                raise RuntimeError("v51 GPU monitor failed") from (
                    failures.get() if not failures.empty() else None
                )
            gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
            cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
            trainer = None
            import ray
            ray.shutdown()
            idle = v40a.cleanup_v38a.wait_for_gpu_idle()

        report = v40a.self_hashed({
            "schema": "direct-pinned-master-transition-microbenchmark-report-v51",
            "status": "complete_no_optimizer_update",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": (
                args.preregistration_content_sha256
            ),
            "design_content_sha256": design["content_sha256"],
            "installations": installs,
            "initial_master_identity": master,
            "initial_references": references,
            "score_audits": {
                "preinstall": preinstall,
                "postinstall": postinstall,
                "post_population": post_population,
                "final_exact_master_score_preserved": True,
            },
            "final_state_certificates": final_certificates,
            "population": {
                "path": str(POPULATION_ARTIFACT),
                "file_sha256": v40a.file_sha256(POPULATION_ARTIFACT),
                "content_sha256": population_artifact[
                    "content_sha256_before_self_field"
                ],
            },
            "timing": {
                "path": str(TIMING_ARTIFACT),
                "file_sha256": v40a.file_sha256(TIMING_ARTIFACT),
                "content_sha256": timing_artifact[
                    "content_sha256_before_self_field"
                ],
                "coverage": timing["coverage"],
            },
            "optimizer_update_performed": False,
            "snapshot": None,
            "gpu_activity": gpu,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "gpu_log": {
                "path": str(GPU_LOG),
                "file_sha256": v40a.file_sha256(GPU_LOG),
            },
            "protected_semantics_opened": False,
            "shadow_ood_holdout_or_benchmark_opened": False,
            "quality_or_promotion_conclusion_authorized": False,
        })
        v40a.atomic_json(REPORT, report)
        print(json.dumps({
            "report": str(REPORT),
            "report_sha256": v40a.file_sha256(REPORT),
            "report_content_sha256": report[
                "content_sha256_before_self_field"
            ],
            "timing": str(TIMING_ARTIFACT),
            "intermediate_restores_eliminated": 15,
            "optimizer_update_performed": False,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        restore_failure = cleanup_failure = None
        if trainer is not None and master is not None and master_runtime_sha:
            try:
                emergency_restore = v40a._rpc_all(
                    trainer, "restore_pinned_master_v51", (
                        master["sha256"], master_runtime_sha,
                        f"controller_exception:{type(error).__name__}",
                    ),
                )
            except BaseException as restore_error:
                restore_failure = {
                    "type": type(restore_error).__name__,
                    "message": str(restore_error),
                    "traceback": traceback.format_exc(),
                }
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        if trainer is not None:
            try:
                cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
                trainer = None
                import ray
                ray.shutdown()
                idle = v40a.cleanup_v38a.wait_for_gpu_idle()
            except BaseException as cleanup_error:
                cleanup_failure = {
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                    "traceback": traceback.format_exc(),
                }
        v40a.atomic_json(FAILURE, v40a.self_hashed({
            "schema": "direct-pinned-master-transition-failure-v51",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "emergency_exact_restore": emergency_restore,
            "emergency_exact_restore_failure": restore_failure,
            "strict_cleanup_failure": cleanup_failure,
            "protected_semantics_opened": False,
            "shadow_ood_holdout_or_benchmark_opened": False,
        }))
        raise
    finally:
        if trainer is not None:
            try:
                v40a.base.close_trainer(trainer)
            except Exception:
                pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved


if __name__ == "__main__":
    raise SystemExit(main())
