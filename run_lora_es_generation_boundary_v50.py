#!/usr/bin/env python3
"""V48B objective with one-state double-buffered CPU scoring.

Only population scoring is rescheduled.  V48B request construction, scoring,
objective construction, exact restoration checks, post-population base check,
reliability gate, candidate gate, transaction, and exact-abort code are reused.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path

import lora_es_bounded_scoring_pipeline_v50 as pipeline
import run_lora_es_generation_boundary_v48b as v48b


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v50_matched_lora_es_generation_boundary_pop8_cpu_pipeline"
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT
).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "matched_lora_es_report_v50.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v50.jsonl").resolve()
SNAPSHOT = (RUN_DIR / "adapter_step1_v50").resolve()
CALIBRATION_ARTIFACT = (RUN_DIR / "numeric_calibration_v50.json").resolve()
ANCHOR_CALIBRATION_ARTIFACT = (
    RUN_DIR / "anchor_calibration_v50.json"
).resolve()
RELIABILITY_ARTIFACT = (
    RUN_DIR / "population_reliability_v50.json"
).resolve()
POST_UPDATE_ARTIFACT = (
    RUN_DIR / "post_update_consensus_v50.json"
).resolve()
CANDIDATE_GATE_ARTIFACT = (RUN_DIR / "candidate_gate_v50.json").resolve()
ABORT_ARTIFACT = (RUN_DIR / "exact_abort_v50.json").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_generation_boundary_pop8_cpu_pipeline_v50.json"
).resolve()
PARENT_PREREGISTRATION = v48b.PREREGISTRATION
PARENT_PREREGISTRATION_FILE_SHA256 = (
    "34e19fe84ff061b98a8627f07daab59f5cbb8c718668fc479454114fef67c3d0"
)
PARENT_PREREGISTRATION_CONTENT_SHA256 = (
    "4d5e17a07551377f0ef39c3dfa306fda68b9b669eeafd3c78ebbfff894072d1e"
)
SCORER_ACTORS = 4
MAX_OUTSTANDING_SCORING_STATES = 1


def implementation_bindings_v50() -> dict[str, str]:
    paths = {
        "runtime_v50": Path(__file__).resolve(),
        "pipeline_v50": ROOT / "lora_es_bounded_scoring_pipeline_v50.py",
        "builder_v50": (
            ROOT / "build_lora_es_generation_boundary_preregistration_v50.py"
        ),
        "tests_v50": ROOT / "test_lora_es_bounded_scoring_pipeline_v50.py",
        "runtime_v48b": Path(v48b.__file__).resolve(),
        "parent_preregistration_v48b": PARENT_PREREGISTRATION,
    }
    return {
        label: v48b.v43i.v40a.file_sha256(path)
        for label, path in paths.items()
    }


def _load_parent_v50() -> dict:
    if (
        v48b.v43i.v40a.file_sha256(PARENT_PREREGISTRATION)
        != PARENT_PREREGISTRATION_FILE_SHA256
    ):
        raise RuntimeError("v50 frozen V48B parent file changed")
    parent = json.loads(PARENT_PREREGISTRATION.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in parent.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        parent.get("content_sha256_before_self_field")
        != PARENT_PREREGISTRATION_CONTENT_SHA256
        or v48b.v43i.v40a.canonical_sha256(compact)
        != PARENT_PREREGISTRATION_CONTENT_SHA256
        or parent.get("protected_semantics_opened") is not False
        or parent.get("shadow_ood_holdout_or_benchmark_opened") is not False
        or parent.get("implementation_bindings")
        != v48b.implementation_bindings_v48b(
            parent["recipe"]["subset_file_sha256"]
        )
    ):
        raise RuntimeError("v50 frozen V48B parent content changed")
    return parent


def load_preregistration_v50(args) -> dict:
    path = Path(args.preregistration).resolve()
    if v48b.v43i.v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v50 preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    parent = _load_parent_v50()
    schedule = value.get("population_scoring_schedule", {})
    expected_parent = {
        "path": str(PARENT_PREREGISTRATION),
        "file_sha256": PARENT_PREREGISTRATION_FILE_SHA256,
        "content_sha256": PARENT_PREREGISTRATION_CONTENT_SHA256,
        "objective_and_gate_contract_inherited_exactly": True,
    }
    expected_artifacts = {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "report": str(REPORT),
        "gpu_log": str(GPU_LOG),
        "snapshot": str(SNAPSHOT),
        "population": str(RELIABILITY_ARTIFACT),
        "candidate_gate": str(CANDIDATE_GATE_ARTIFACT),
        "exact_abort": str(ABORT_ARTIFACT),
    }
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or v48b.v43i.v40a.canonical_sha256(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "matched-lora-es-generation-boundary-preregistration-v50"
        or value.get("status")
        != "preregistered_before_train_only_launch"
        or value.get("gpu_launch_authorized") is not True
        or value.get("protected_semantic_access_authorized") is not False
        or value.get("shadow_ood_holdout_or_benchmark_authorized") is not False
        or value.get("quality_selection_or_promotion_authorized") is not False
        or value.get("recipe") != parent.get("recipe")
        or value.get("generation_boundary_objective")
        != parent.get("generation_boundary_objective")
        or value.get("uncommitted_candidate_gate")
        != parent.get("uncommitted_candidate_gate")
        or value.get("access_contract") != parent.get("access_contract")
        or value.get("runtime") != parent.get("runtime")
        or value.get("parents", {}).get("v48b_preregistration")
        != expected_parent
        or value.get("artifacts") != expected_artifacts
        or value.get("implementation_bindings")
        != implementation_bindings_v50()
        or schedule.get("cpu_scorer_actors") != SCORER_ACTORS
        or schedule.get("cpus_per_scorer_actor") != 1
        or schedule.get("gpus_per_scorer_actor") != 0
        or schedule.get("max_outstanding_scoring_states")
        != MAX_OUTSTANDING_SCORING_STATES
        or schedule.get("state_order")
        != "direction ascending; plus then minus; actor rank ascending"
        or schedule.get("restore_verified_before_next_materialization")
        is not True
        or schedule.get("completion_order_affects_placement") is not False
        or schedule.get("objective_or_gate_changed") is not False
        or schedule.get("generation_outputs_passed_by_object_reference")
        is not True
        or schedule.get("scorer_inputs_initialized_once_and_immutable")
        is not True
        or value.get("required_gates", {}).get(
            "one_scoring_state_queue_bound"
        ) is not True
        or value.get("required_gates", {}).get(
            "all_state_restores_verified_before_successor_materialization"
        ) is not True
        or value.get("required_gates", {}).get(
            "deterministic_actor_rank_placement"
        ) is not True
        or value.get("required_gates", {}).get(
            "cpu_scorers_have_zero_gpu_visibility"
        ) is not True
        or value.get("required_gates", {}).get(
            "V48B_population_numeric_equivalence"
        ) is not True
        or value.get("protected_semantics_opened") is not False
        or value.get("shadow_ood_holdout_or_benchmark_opened") is not False
        or value.get("current_fixed_holdout_cycle_eligible") is not False
    ):
        raise RuntimeError("v50 preregistration contract changed")
    recipe = value["recipe"]
    subset = v48b.load_subset_v48b(
        Path(recipe["subset"]), recipe["subset_file_sha256"],
        recipe["subset_content_sha256"],
    )
    if subset["request_order_sha256"] != recipe["request_order_sha256"]:
        raise RuntimeError("v50 frozen V48B request order changed")
    v48b._SEALED_SUBSET = subset
    return value


def _state_tag_v50(
    state: pipeline.SignedStateV50, actor_rank: int,
) -> dict:
    return {
        "direction": state.direction,
        "label": state.label,
        "sign": state.sign,
        "actor_rank": actor_rank,
    }


def order_actor_scores_v50(
    state: pipeline.SignedStateV50, rows: list[dict],
) -> list[dict]:
    """Validate receipts and place scores by actor rank, not finish order."""
    if len(rows) != SCORER_ACTORS:
        raise RuntimeError("v50 incomplete CPU scorer coverage")
    by_rank = {}
    for row in rows:
        tag = row.get("state", {})
        rank = tag.get("actor_rank")
        if (
            not isinstance(rank, int) or rank not in range(SCORER_ACTORS)
            or tag != _state_tag_v50(state, rank)
            or row.get("schema") != "generation-boundary-actor-score-v50"
            or row.get("gpu_ids") != []
            or row.get("score", {}).get("actor_rank") != rank
            or rank in by_rank
        ):
            raise RuntimeError("v50 CPU scorer receipt changed")
        by_rank[rank] = row["score"]
    if set(by_rank) != set(range(SCORER_ACTORS)):
        raise RuntimeError("v50 actor score matrix is incomplete")
    return [by_rank[rank] for rank in range(SCORER_ACTORS)]


class PopulationScoringActorV50:
    """A GPU-free, persistent scorer holding immutable train-only inputs."""

    def __init__(
        self, actor_rank, bundle, dense_items, plan, anchors,
        prepared_fragile, sealed_subset,
    ):
        import ray

        gpu_ids = list(ray.get_gpu_ids())
        if gpu_ids or os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            raise RuntimeError("v50 CPU scoring actor received GPU visibility")
        actor_rank = int(actor_rank)
        if actor_rank not in range(SCORER_ACTORS):
            raise RuntimeError("v50 invalid scoring actor rank")
        self.actor_rank = actor_rank
        self.bundle = bundle
        self.dense_items = dense_items
        self.plan = plan
        self.anchors = anchors
        self.prepared_fragile = prepared_fragile
        self.sealed_subset = sealed_subset
        self.gpu_ids = gpu_ids

    def runtime_identity_v50(self) -> dict:
        return {
            "schema": "generation-boundary-cpu-scorer-identity-v50",
            "actor_rank": self.actor_rank,
            "gpu_ids": self.gpu_ids,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }

    def score_v50(self, state: dict, batch: list) -> dict:
        expected = dict(state)
        expected["actor_rank"] = self.actor_rank
        if state != expected or len(batch) != 608:
            raise RuntimeError("v50 scoring actor input coverage changed")
        saved_fragile = v48b._PREPARED_FRAGILE
        saved_subset = v48b._SEALED_SUBSET
        v48b._PREPARED_FRAGILE = self.prepared_fragile
        v48b._SEALED_SUBSET = self.sealed_subset
        try:
            scored = v48b.score_fused_outputs_v48b(
                self.plan, batch, self.anchors, v48b.v43i.anchor_v4,
                domain_scorer=lambda outputs: (
                    v48b.v43i.score_batch_detailed_v43i(
                        self.bundle, self.dense_items, outputs,
                    )
                ),
            )
        finally:
            v48b._PREPARED_FRAGILE = saved_fragile
            v48b._SEALED_SUBSET = saved_subset
        return {
            "schema": "generation-boundary-actor-score-v50",
            "state": expected,
            "gpu_ids": self.gpu_ids,
            "score": {
                "actor_rank": self.actor_rank,
                **v48b.compact_signed_score_v48b(scored),
            },
        }


class _RayPopulationOperationsV50:
    def __init__(
        self, trainer, bundle, dense_items, plan, params, anchors,
        master_sha: str,
    ) -> None:
        import ray

        if v48b._PREPARED_FRAGILE is None or v48b._SEALED_SUBSET is None:
            raise RuntimeError("v50 immutable scoring inputs are incomplete")
        self.ray = ray
        self.trainer = trainer
        self.plan = plan
        self.params = params
        self.master_sha = master_sha
        actor_type = ray.remote(
            num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0,
        )(PopulationScoringActorV50)
        self.scorers = [
            actor_type.options(runtime_env={
                "env_vars": {"CUDA_VISIBLE_DEVICES": ""},
            }).remote(
                rank, bundle, dense_items, plan, anchors,
                list(v48b._PREPARED_FRAGILE), dict(v48b._SEALED_SUBSET),
            )
            for rank in range(SCORER_ACTORS)
        ]
        try:
            identities = ray.get([
                scorer.runtime_identity_v50.remote()
                for scorer in self.scorers
            ])
            if identities != [{
                "schema": "generation-boundary-cpu-scorer-identity-v50",
                "actor_rank": rank,
                "gpu_ids": [],
                "cuda_visible_devices": "",
            } for rank in range(SCORER_ACTORS)]:
                raise RuntimeError("v50 CPU scorer identity coverage changed")
        except BaseException:
            for scorer in self.scorers:
                ray.kill(scorer, no_restart=True)
            raise

    def materialize(self, state: pipeline.SignedStateV50) -> list[dict]:
        values = self.trainer._resolve([
            self.trainer.engines[actor].collective_rpc.remote(
                "materialize_antithetic_adapter_v41a",
                args=(
                    v48b.v43i.SEEDS[state.direction], v48b.v43i.SIGMA,
                    state.sign, self.master_sha,
                ),
            )
            for actor in range(SCORER_ACTORS)
        ])
        if len(values) != SCORER_ACTORS or any(len(value) != 1 for value in values):
            raise RuntimeError("v50 perturbation actor coverage changed")
        return [value[0] for value in values]

    def launch_generation(self, _state: pipeline.SignedStateV50):
        return [
            self.trainer.engines[actor].generate.remote(
                self.plan["requests"], self.params, use_tqdm=False,
                lora_request=v48b.v43i._lora_request(),
            )
            for actor in range(SCORER_ACTORS)
        ]

    def wait_generation(self, _state, handles) -> None:
        if len(handles) != SCORER_ACTORS:
            raise RuntimeError("v50 generation handle coverage changed")
        ready, remaining = self.ray.wait(
            handles, num_returns=SCORER_ACTORS, fetch_local=False,
        )
        if len(ready) != SCORER_ACTORS or remaining:
            raise RuntimeError("v50 generation did not reach a terminal state")

    def restore(self, _state: pipeline.SignedStateV50) -> list[dict]:
        restored = v48b.v43i.v40a._rpc_all(
            self.trainer, "restore_adapter_master_v41a",
        )
        if (
            len(restored) != SCORER_ACTORS
            or any(
                item.get("restored_identity", {}).get("sha256")
                != self.master_sha
                for item in restored
            )
        ):
            raise RuntimeError("v50 signed exact restore changed master")
        return restored

    def submit_scoring(self, state, generation_handles):
        return [
            self.scorers[rank].score_v50.remote(
                _state_tag_v50(state, rank), generation_handles[rank],
            )
            for rank in range(SCORER_ACTORS)
        ]

    def resolve_scoring(self, state, scoring_handles):
        return order_actor_scores_v50(
            state, self.ray.get(scoring_handles),
        )

    def cancel_scoring(self, _state, scoring_handles) -> None:
        for handle in scoring_handles:
            self.ray.cancel(handle, force=False)

    def close(self) -> None:
        for scorer in self.scorers:
            self.ray.kill(scorer, no_restart=True)

    def operations(self) -> pipeline.PipelineOperationsV50:
        return pipeline.PipelineOperationsV50(
            materialize=self.materialize,
            launch_generation=self.launch_generation,
            wait_generation=self.wait_generation,
            restore=self.restore,
            submit_scoring=self.submit_scoring,
            resolve_scoring=self.resolve_scoring,
            cancel_scoring=self.cancel_scoring,
        )


def replicated_population_v50(
    trainer, bundle, dense_items, requests, anchors: dict, master_sha: str,
) -> dict:
    """Drop-in replacement for V48B's population scheduler only."""
    numeric = v48b.v43i.numeric
    plan = v48b.fused_requests_v48b(requests, anchors)
    params = v48b.sampling_params_for_plan_v48b(plan)
    if (
        len(plan["requests"]) != 608 or len(params) != 608
        or v48b.v43i.POPULATION_SIZE != 8
        or numeric.SIGNED_REPLICATES_V43G != SCORER_ACTORS
    ):
        raise RuntimeError("v50 frozen V48B population contract changed")

    states = pipeline.signed_states_v50(v48b.v43i.POPULATION_SIZE)
    assignments = [
        numeric.complete_actor_assignments_v43g(direction)
        for direction in range(v48b.v43i.POPULATION_SIZE)
    ]
    runtime = _RayPopulationOperationsV50(
        trainer, bundle, dense_items, plan, params, anchors, master_sha,
    )
    try:
        completed = pipeline.run_one_state_double_buffer_v50(
            states, runtime.operations(),
        )
    finally:
        runtime.close()
    if [item.state for item in completed] != list(states):
        raise RuntimeError("v50 signed state completion order changed")

    scores = {
        label: [[None] * SCORER_ACTORS
                for _ in range(v48b.v43i.POPULATION_SIZE)]
        for label in ("plus", "minus")
    }
    perturbations = {
        label: [[None] * SCORER_ACTORS
                for _ in range(v48b.v43i.POPULATION_SIZE)]
        for label in ("plus", "minus")
    }
    restorations = []
    receipts = []
    for item in completed:
        state = item.state
        if (
            len(item.actor_scores) != SCORER_ACTORS
            or len(item.materialization) != SCORER_ACTORS
            or len(item.restoration) != SCORER_ACTORS
        ):
            raise RuntimeError("v50 completed state coverage changed")
        restorations.extend(item.restoration)
        for actor_rank in range(SCORER_ACTORS):
            scores[state.label][state.direction][actor_rank] = (
                item.actor_scores[actor_rank]
            )
            perturbations[state.label][state.direction][actor_rank] = (
                item.materialization[actor_rank]
            )
            receipts.append({
                "direction": state.direction,
                "sign": state.label,
                "actor_rank": actor_rank,
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
    if any(item is None for label in scores.values()
           for direction in label for item in direction):
        raise RuntimeError("v50 signed score matrix incomplete")
    if len(restorations) != 64 or any(
        item["restored_identity"]["sha256"] != master_sha
        for item in restorations
    ):
        raise RuntimeError("v50 exact restoration certificate matrix changed")
    for label in ("plus", "minus"):
        for direction in range(v48b.v43i.POPULATION_SIZE):
            certificates = perturbations[label][direction]
            if (
                len({v48b.v43i.v40a.canonical_sha256(
                    item["candidate_identity"]
                ) for item in certificates}) != 1
                or len({item["materialization"]["runtime_values_sha256"]
                        for item in certificates}) != 1
            ):
                raise RuntimeError("v50 replicated signed state differs")

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
    sign_scores = {}
    for objective, path in paths.items():
        sign_scores[objective] = {
            label: [[float(v48b.v43i._nested_value(
                scores[label][direction][replicate], path,
            )) for replicate in range(SCORER_ACTORS)]
                    for direction in range(v48b.v43i.POPULATION_SIZE)]
            for label in ("plus", "minus")
        }
    direct = v48b.boundary.direct_generation_objective_v48a(
        sign_scores["domain"], sign_scores["fragile_generation_f1"],
        sign_scores["prose_lm"], sign_scores["qa_answer_logprob"],
    )
    return {
        "schema": "fused-generation-boundary-population-v48b",
        "assignments": assignments,
        "signed_scores": scores,
        "objective_sign_scores": sign_scores,
        "objective_fitness": direct["objective_fitness"],
        "central_replicates": v48b.v43i._central_replicates(
            sign_scores["domain"]
        ),
        "coefficients": direct["projection"]["coefficients"],
        "unconstrained_domain_coefficients": direct["objective_fitness"][
            "domain"
        ]["coefficients"],
        "projection": direct["projection"],
        "direct_generation_boundary_objective": direct,
        "common_random_plan": common,
        "fused_requests_per_actor_state": 608,
        "perturbation_certificates": perturbations,
        "restoration_certificate_count": len(restorations),
        "all_exact_restores_passed": True,
    }


@contextmanager
def patched_v48b_v50():
    replacements = {
        "EXPERIMENT": EXPERIMENT,
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "REPORT": REPORT,
        "GPU_LOG": GPU_LOG,
        "SNAPSHOT": SNAPSHOT,
        "CALIBRATION_ARTIFACT": CALIBRATION_ARTIFACT,
        "ANCHOR_CALIBRATION_ARTIFACT": ANCHOR_CALIBRATION_ARTIFACT,
        "RELIABILITY_ARTIFACT": RELIABILITY_ARTIFACT,
        "POST_UPDATE_ARTIFACT": POST_UPDATE_ARTIFACT,
        "CANDIDATE_GATE_ARTIFACT": CANDIDATE_GATE_ARTIFACT,
        "ABORT_ARTIFACT": ABORT_ARTIFACT,
        "PREREGISTRATION": PREREGISTRATION,
        "load_preregistration_v48b": load_preregistration_v50,
        "replicated_population_v48b": replicated_population_v50,
    }
    saved = {name: getattr(v48b, name) for name in replacements}
    for name, value in replacements.items():
        setattr(v48b, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(v48b, name, value)


def _rewrite_report_v50() -> None:
    if not REPORT.exists():
        return
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report.pop("content_sha256_before_self_field", None)
    report["schema"] = "matched-lora-es-generation-boundary-report-v50"
    report["population_scoring_schedule"] = {
        "cpu_scorer_actors": SCORER_ACTORS,
        "gpus_per_scorer_actor": 0,
        "max_outstanding_scoring_states": MAX_OUTSTANDING_SCORING_STATES,
        "restore_verified_before_next_materialization": True,
        "completion_order_affects_placement": False,
        "objective_or_gate_changed": False,
    }
    report["content_sha256_before_self_field"] = (
        v48b.v43i.v40a.canonical_sha256(report)
    )
    temporary = REPORT.with_name(f".{REPORT.name}.rewrite")
    if temporary.exists():
        raise FileExistsError(temporary)
    v48b.v43i.v40a.atomic_json(temporary, report)
    temporary.replace(REPORT)


def main(argv: list[str] | None = None) -> int:
    args = v48b.v43i.parser().parse_args(argv)
    if args.dry_run:
        prereg = load_preregistration_v50(args)
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "cpu_scorer_actors": SCORER_ACTORS,
            "max_outstanding_scoring_states": (
                MAX_OUTSTANDING_SCORING_STATES
            ),
            "train_semantics_loaded": False,
            "model_or_gpu_loaded": False,
            "protected_semantic_access_count": 0,
            "shadow_ood_holdout_or_benchmark_opened": False,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    with patched_v48b_v50():
        code = v48b.main(argv)
    if code == 0:
        _rewrite_report_v50()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
