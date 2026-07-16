#!/usr/bin/env python3
"""Run the V65 robust-sampling LoRA-ES population on four exclusive GPUs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import eggroll_es_worker_robust_sampling_v65 as scoring65
import lora_es_nested_population_v52 as design52
import lora_es_robust_sampling_population_v65 as design65
import run_lora_es_nested_population_v52 as runtime52
import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime64


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_robust_sampling_population_v65.json"
).resolve()
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v65_lora_es_robust_sampling_population"
).resolve()
ATTEMPT = (
    RUN_DIR.parent / ".v65_lora_es_robust_sampling_population.attempt.json"
).resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v65.jsonl").resolve()
EVIDENCE = (RUN_DIR / "robust_sampling_evidence_v65.json").resolve()
ANALYSIS = (RUN_DIR / "robust_sampling_analysis_v65.json").resolve()
REPORT = (RUN_DIR / "robust_sampling_report_v65.json").resolve()
FAILURE = (RUN_DIR / "failure_v65.json").resolve()

RANKING_PREFIX_BYTES_V65 = 136_848
RANKING_PREFIX_SHA256_V65 = (
    "8259894003268a2fafed6a9a66ce3e604d5eb76cdf19a1c1c759e5ffc5916c70"
)
GENERATION_SEED_V65 = 2_026_071_601


def artifacts_v65() -> dict:
    return {
        "run_directory": str(RUN_DIR),
        "attempt": str(ATTEMPT),
        "gpu_log": str(GPU_LOG),
        "evidence": str(EVIDENCE),
        "analysis": str(ANALYSIS),
        "report": str(REPORT),
        "failure": str(FAILURE),
    }


def _write_self_hashed_v65(path: Path, value: dict) -> dict:
    result = dict(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = (
        design65.canonical_sha256_v65(result)
    )
    payload = (json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return result


def _implementation_binding_exact_v65(binding) -> bool:
    if isinstance(binding, str):
        return False
    if not isinstance(binding, dict) or set(binding) != {"path", "file_sha256"}:
        return False
    path = Path(binding["path"]).resolve()
    return path.is_file() and design65.file_sha256_v65(path) == binding["file_sha256"]


def load_preregistration_v65(args) -> dict:
    path = Path(args.preregistration).resolve()
    if design65.file_sha256_v65(path) != args.preregistration_sha256:
        raise RuntimeError("v65 preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    authorization = value.get("authorization", {})
    access = value.get("access_contract", {})
    recipe = value.get("fixed_measurement_recipe", {})
    if (
        value.get("schema")
        != "v65-lora-es-robust-sampling-population-preregistration"
        or value.get("status")
        != "sealed_before_v65_train_semantics_model_ray_or_gpu_access"
        or value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or design65.self_content_sha256_v65(value)
        != args.preregistration_content_sha256
        or value.get("artifacts") != artifacts_v65()
        or authorization.get("gpu_launch") is not True
        or authorization.get("population_generation_measurement") is not True
        or any(authorization.get(key) is not False for key in (
            "projection", "optimizer_update", "candidate_snapshot",
            "train_holdback", "exact_sentinel", "ood_shadow",
            "protected_semantics", "terminal_holdout", "promotion",
        ))
        or access.get("decode_exactly_first_64_v61c_ranking_rows") is not True
        or access.get("decode_v61c_row_64_or_later") is not False
        or access.get("ranking_prefix_bytes") != RANKING_PREFIX_BYTES_V65
        or access.get("ranking_prefix_sha256") != RANKING_PREFIX_SHA256_V65
        or recipe.get("population_size") != design65.POPULATION_SIZE_V65
        or recipe.get("scheduled_state_occurrences") != design65.STATE_COUNT_V65
        or recipe.get("unique_exact_v53_states") != 32
        or recipe.get("actors") != design65.ACTORS_V65
        or recipe.get("passes_per_signed_state")
        != design65.PASSES_PER_SIGNED_STATE_V65
        or recipe.get("sigma") != design65.SIGMA_V65
        or recipe.get("seeds") != list(design65.SEEDS_V65)
        or recipe.get("state_schedule") != design65.state_derivations_v65()
        or recipe.get("generation_seed") != GENERATION_SEED_V65
        or value.get("population_analysis_contract", {}).get(
            "paired_replicas_per_unit_preserved_and_averaged"
        ) != 8
        or value.get("population_analysis_contract", {}).get(
            "resampled_axis"
        ) != "conflict_unit_only"
        or not value.get("implementation_bindings")
        or not all(_implementation_binding_exact_v65(binding)
                   for binding in value["implementation_bindings"].values())
    ):
        raise RuntimeError("v65 preregistration contract changed")
    return value


def _row_sha256_v65(row: dict) -> str:
    return hashlib.sha256(json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def _load_hash_only_panel_v65(preregistration: dict) -> tuple[dict, dict]:
    preview = design65.read_exact_self_hashed_v65(
        design65.PREVIEW_V61, design65.PREVIEW_V61_FILE_SHA256,
        design65.PREVIEW_V61_CONTENT_SHA256,
    )
    expected = design65.build_ranking_panel_v65(preview)
    bound = preregistration.get("ranking_panel", {})
    path = Path(bound.get("path", "")).resolve()
    if (
        not path.is_file()
        or design65.file_sha256_v65(path) != bound.get("file_sha256")
    ):
        raise RuntimeError("v65 ranking panel file changed")
    observed = json.loads(path.read_text(encoding="utf-8"))
    if (
        observed != expected
        or observed.get("content_sha256_before_self_field")
        != bound.get("content_sha256")
        or design65.self_content_sha256_v65(observed)
        != bound.get("content_sha256")
    ):
        raise RuntimeError("v65 ranking panel content changed")
    panel61c = design65.read_exact_self_hashed_v65(
        design65.V61C_PANEL, design65.V61C_PANEL_FILE_SHA256,
        design65.V61C_PANEL_CONTENT_SHA256,
    )
    first64 = panel61c.get("items", [])[:64]
    if (
        len(panel61c.get("items", [])) != 68
        or panel61c.get("ranking_units") != 64
        or panel61c.get("exact_sentinel_units") != 4
        or any(item.get("role") != "ranking" for item in first64)
        or [item["row_sha256"] for item in first64]
        != [item["row_sha256"] for item in observed["items"]]
        or [item["unit_identity_sha256"] for item in first64]
        != [item["unit_identity_sha256"] for item in observed["items"]]
    ):
        raise RuntimeError("v65 V61C ranking projection changed")
    return observed, panel61c


def _read_ranking_prefix_rows_v65(panel: dict) -> tuple[list[dict], dict]:
    """Read only the authorized byte prefix; never request byte 136848+."""
    descriptor = os.open(design65.V61C_ROWS, os.O_RDONLY | os.O_CLOEXEC)
    try:
        payload = os.pread(descriptor, RANKING_PREFIX_BYTES_V65, 0)
    finally:
        os.close(descriptor)
    if (
        len(payload) != RANKING_PREFIX_BYTES_V65
        or hashlib.sha256(payload).hexdigest() != RANKING_PREFIX_SHA256_V65
        or not payload.endswith(b"\n")
    ):
        raise RuntimeError("v65 authorized ranking byte prefix changed")
    raw_lines = payload.splitlines()
    if len(raw_lines) != 64:
        raise RuntimeError("v65 ranking prefix line count changed")
    rows = []
    for index, (raw, item) in enumerate(zip(
        raw_lines, panel["items"], strict=True,
    )):
        row = json.loads(raw.decode("utf-8"))
        if (
            item.get("request_index") != index
            or _row_sha256_v65(row) != item["row_sha256"]
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
            or not isinstance(row.get("answer"), str)
            or not row["answer"].strip()
        ):
            raise RuntimeError("v65 decoded ranking-row identity changed")
        rows.append(row)
    receipt = {
        "schema": "v65-exact-authorized-ranking-prefix-receipt",
        "path": str(design65.V61C_ROWS),
        "source_full_file_sha256_bound_but_not_recomputed": (
            design65.V61C_ROWS_FILE_SHA256
        ),
        "authorized_prefix_bytes": RANKING_PREFIX_BYTES_V65,
        "authorized_prefix_sha256": RANKING_PREFIX_SHA256_V65,
        "decoded_ranking_rows": 64,
        "requested_byte_offset_at_or_after_prefix": False,
        "remaining_exact_sentinel_rows_decoded": 0,
        "question_answer_or_text_persisted": False,
    }
    return rows, receipt


def prepare_ranking_requests_v65(
    trainer, prior, panel: dict,
) -> tuple[list[dict], list, list[str], dict]:
    rows, input_receipt = _read_ranking_prefix_rows_v65(panel)
    requests = []
    answers = []
    for row in rows:
        prompt = prior.v40a.base.specialist_template(row["question"])
        token_ids = prior.fused._encode(
            trainer.tokenizer, prompt, prior.fused.MAX_QA_TOKENS_V43I,
            "V65 ranking generation prompt",
        )
        requests.append({"prompt_token_ids": token_ids})
        answers.append(row["answer"])
    from vllm import SamplingParams
    generation = SamplingParams(
        n=1, seed=GENERATION_SEED_V65, temperature=0.0, top_p=1.0,
        max_tokens=prior.fused.MAX_GENERATION_TOKENS_V43I,
        prompt_logprobs=None, detokenize=True,
    )
    params = [generation for _ in requests]
    input_receipt.update({
        "request_count": len(requests),
        "request_prompt_token_ids_sha256": design65.canonical_sha256_v65([
            request["prompt_token_ids"] for request in requests
        ]),
        "generation_seed": GENERATION_SEED_V65,
        "generation_temperature": 0.0,
        "generation_max_tokens": prior.fused.MAX_GENERATION_TOKENS_V43I,
    })
    if len(requests) != 64 or len(params) != 64 or len(answers) != 64:
        raise RuntimeError("v65 prepared request coverage changed")
    return requests, params, answers, input_receipt


class RayRobustPopulationOperationsV65(runtime52.RayPopulationOperationsV52):
    """V52 exact LoRA transitions with a ranking-only numeric CPU scorer."""

    def __init__(
        self, trainer, requests, params, panel_items, answers,
        expected_states, master_sha: str, master_runtime_sha: str,
        phase, *, prior, v40a,
    ) -> None:
        import ray

        self.ray = ray
        self.trainer = trainer
        self.plan = {"requests": requests}
        self.params = params
        self.master_sha = master_sha
        self.master_runtime_sha = master_runtime_sha
        self.expected_states = {
            (row["direction"], row["label"]): dict(row)
            for row in expected_states
        }
        if len(self.expected_states) != 32:
            raise RuntimeError("v65 exact expected state coverage changed")
        self.prior = prior
        self.v40a = v40a
        self.phase = phase
        actor_type = ray.remote(
            num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0,
        )(scoring65.RobustGenerationScoringActorV65)
        self.scorers = [
            actor_type.options(runtime_env={
                "env_vars": {"CUDA_VISIBLE_DEVICES": ""},
            }).remote(rank, panel_items, answers)
            for rank in range(design65.ACTORS_V65)
        ]
        identities = ray.get([
            actor.runtime_identity_v65.remote() for actor in self.scorers
        ])
        expected = [{
            "schema": "v65-robust-generation-cpu-scorer",
            "actor_rank": rank,
            "gpu_ids": [],
            "cuda_visible_devices": "",
        } for rank in range(design65.ACTORS_V65)]
        if identities != expected:
            self.close()
            raise RuntimeError("v65 CPU scorer identities changed")
        self.last_state_index = -1

    @staticmethod
    def _score_tag_v65(state: dict, actor_rank: int) -> dict:
        return {
            "state_index": state["state_index"],
            "direction": state["direction"],
            "label": state["label"],
            "sign": state["sign"],
            "pass_index": state["pass_index"],
            "actor_rank": actor_rank,
        }

    def transition(self, derivation: dict, previous: dict | None) -> dict:
        expected_previous = (
            self.master_sha if previous is None
            else previous["candidate_identity_sha256"]
        )
        exact = self.expected_states[(
            derivation["direction"], derivation["label"],
        )]
        started_ns = time.monotonic_ns()
        values = self.trainer._resolve([
            self.trainer.engines[rank].collective_rpc.remote(
                "transition_antithetic_from_pinned_master_v51",
                args=(
                    derivation["state_index"], derivation["seed"],
                    derivation["sigma"], derivation["sign"],
                    self.master_sha, expected_previous,
                    exact["candidate_identity_sha256"],
                    exact["runtime_values_sha256"],
                ),
            )
            for rank in range(design65.ACTORS_V65)
        ])
        ended_ns = time.monotonic_ns()
        if len(values) != design65.ACTORS_V65 or any(
            len(item) != 1 for item in values
        ):
            raise RuntimeError("v65 transition actor coverage changed")
        actors = []
        for rank, value in enumerate(values):
            receipt = dict(value[0])
            receipt["actor_rank"] = rank
            receipt["elapsed_ns"] = receipt["timing"]["elapsed_ns"]
            actors.append(receipt)
        consensus = runtime52.seal_runtime_state_consensus_v52(
            derivation, actors,
        )
        if (
            consensus["candidate_identity_sha256"]
            != exact["candidate_identity_sha256"]
            or consensus["runtime_values_sha256"]
            != exact["runtime_values_sha256"]
        ):
            raise RuntimeError("v65 transition differed from exact V53 state")
        self.last_state_index = derivation["state_index"]
        return {
            "schema": "four-actor-exact-v53-transition-v65",
            "state_index": derivation["state_index"],
            "actors": actors,
            "consensus": consensus,
            "exact_v53_identity_matched_before_scoring": True,
            "controller": runtime52._controller_timing_v52(
                started_ns, ended_ns,
            ),
        }

    def launch_generation(self, state: dict) -> dict:
        self.phase.value = (
            f"population_state_{state['state_index']}_generation_all_actors"
        )
        return super().launch_generation(state)

    def submit_scoring(self, state: dict, generation_handles: dict):
        refs = generation_handles["refs"]
        return [
            self.scorers[rank].score_v65.remote(
                self._score_tag_v65(state, rank), refs[rank],
            )
            for rank in range(design65.ACTORS_V65)
        ]

    def resolve_scoring(self, state: dict, scoring_handles):
        if len(scoring_handles) != design65.ACTORS_V65:
            raise RuntimeError("v65 scoring handle coverage changed")
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
                row.get("schema") != "v65-robust-generation-actor-score"
                or row.get("state") != self._score_tag_v65(state, rank)
                or row.get("gpu_ids") != []
                or len(row.get("unit_metrics", [])) != 64
                or row.get("question_answer_or_generation_text_persisted")
                is not False
            ):
                raise RuntimeError("v65 numeric scorer receipt changed")
            timing = dict(row["timing"])
            timing["schema"] = "worker-score-timing-v65"
            actors.append(timing)
            scores.append(row)
        drain_actors = [{
            "schema": "controller-observed-drain-timing-v65",
            "state_index": state["state_index"],
            "actor_rank": rank,
            "clock": "controller_monotonic_ns",
            "started_ns": started_ns,
            "ended_ns": ready_ns[rank],
            "elapsed_ns": ready_ns[rank] - started_ns,
        } for rank in range(design65.ACTORS_V65)]
        return scores, {
            "schema": "four-actor-score-timing-v65",
            "state_index": state["state_index"],
            "actors": actors,
        }, {
            "schema": "four-actor-drain-timing-v65",
            "state_index": state["state_index"],
            "actors": drain_actors,
            "controller": runtime52._controller_timing_v52(
                started_ns, max(ready_ns.values()),
            ),
        }


def run_direct_master_pipeline_v65(
    operations: RayRobustPopulationOperationsV65,
) -> tuple[list[dict], dict]:
    """Retain V52 overlap while executing V65's 64 counterbalanced states."""
    derivations = design65.state_derivations_v65()
    pending = None
    completed = []
    previous = None
    transition_attempted = False
    try:
        for derivation in derivations:
            transition_attempted = True
            transition = operations.transition(derivation, previous)
            state = runtime52._runtime_state_v52(derivation, transition)
            generation_handles = operations.launch_generation(state)
            generation = operations.wait_generation(state, generation_handles)
            if pending is not None:
                completed.append(runtime52._drain_pending_v52(
                    pending,
                    runtime52._elided_restore_v52(
                        pending.state, state, transition,
                    ),
                    operations,
                ))
                pending = None
            scoring_handles = operations.submit_scoring(
                state, generation_handles,
            )
            pending = runtime52._PendingStateV52(
                state, transition, generation, scoring_handles,
            )
            previous = state
        final_restore = operations.final_restore("v65_population_complete")
        transition_attempted = False
        if pending is not None:
            completed.append(runtime52._drain_pending_v52(
                pending, final_restore, operations,
            ))
            pending = None
        if [row["state"]["state_index"] for row in completed] != list(
            range(design65.STATE_COUNT_V65)
        ):
            raise RuntimeError("v65 state completion order changed")
        return completed, final_restore
    except BaseException as error:
        if pending is not None:
            try:
                operations.cancel_scoring(
                    pending.state, pending.scoring_handles,
                )
            except BaseException as cancel_error:
                error.add_note(
                    "v65 scoring cancellation also failed: "
                    f"{type(cancel_error).__name__}: {cancel_error}"
                )
        if transition_attempted:
            try:
                operations.final_restore(
                    f"v65_exception:{type(error).__name__}",
                )
            except BaseException as restore_error:
                raise RuntimeError(
                    "v65 exact final pinned-master restore failed"
                ) from restore_error
        raise


def compile_population_v65(
    completed: list[dict], panel: dict, expected_states: list[dict],
) -> tuple[dict, dict]:
    matrices = {
        label: [np.full(
            (64, 4, 2, 3), np.nan, dtype=np.float64,
        ) for _ in range(16)]
        for label in ("plus", "minus")
    }
    expected_map = {
        (row["direction"], row["label"]): row for row in expected_states
    }
    state_receipts = []
    occurrence_candidates = []
    occurrence_runtimes = []
    canonical_candidates = []
    canonical_runtimes = []
    panel_rows = [row["row_sha256"] for row in panel["items"]]
    panel_units = [row["unit_identity_sha256"] for row in panel["items"]]
    for item in completed:
        state = item["state"]
        direction = state["direction"]
        label = state["label"]
        pass_index = state["pass_index"]
        exact = expected_map[(direction, label)]
        if (
            state["candidate_identity_sha256"]
            != exact["candidate_identity_sha256"]
            or state["runtime_values_sha256"]
            != exact["runtime_values_sha256"]
            or len(item["actor_scores"]) != 4
        ):
            raise RuntimeError("v65 compiled state identity changed")
        occurrence_candidates.append(state["candidate_identity_sha256"])
        occurrence_runtimes.append(state["runtime_values_sha256"])
        if pass_index == 0:
            canonical_candidates.append(state["candidate_identity_sha256"])
            canonical_runtimes.append(state["runtime_values_sha256"])
        actor_score_hashes = []
        for actor_rank, score in enumerate(item["actor_scores"]):
            metrics = score["unit_metrics"]
            if (
                [row["row_sha256"] for row in metrics] != panel_rows
                or [row["unit_identity_sha256"] for row in metrics]
                != panel_units
                or [row["request_index"] for row in metrics] != list(range(64))
            ):
                raise RuntimeError("v65 scorer panel order changed")
            matrices[label][direction][:, actor_rank, pass_index, :] = [
                [row["f1"], row["exact"], row["nonzero"]]
                for row in metrics
            ]
            actor_score_hashes.append(design65.canonical_sha256_v65(score))
        state_receipts.append({
            "state": state,
            "transition": item["transition"],
            "generation": item["generation"],
            "score_timing": item["score_timing"],
            "restore_timing": item["restore_timing"],
            "drain_timing": item["drain_timing"],
            "actor_numeric_score_sha256": actor_score_hashes,
            "raw_question_answer_or_generation_text_persisted": False,
        })
    signed = {
        label: [array.tolist() for array in matrices[label]]
        for label in ("plus", "minus")
    }
    if (
        any(np.isnan(array).any() for values in matrices.values() for array in values)
        or len(canonical_candidates) != 32
        or design65.canonical_sha256_v65(canonical_candidates)
        != design65.V53_CANDIDATE_IDENTITY_INVENTORY_SHA256
        or design65.canonical_sha256_v65(canonical_runtimes)
        != design65.V53_RUNTIME_IDENTITY_INVENTORY_SHA256
        or len(set(occurrence_candidates)) != 32
        or len(set(occurrence_runtimes)) != 32
    ):
        raise RuntimeError("v65 completed population coverage changed")
    evidence = {
        "schema": "v65-robust-sampling-population-evidence",
        "status": "complete_numeric_population_no_update",
        "panel_file_content_sha256": panel["content_sha256_before_self_field"],
        "metric_order": list(design65.METRIC_ORDER_V65),
        "signed_metrics": signed,
        "signed_metrics_sha256": design65.canonical_sha256_v65(signed),
        "state_receipts": state_receipts,
        "state_receipts_sha256": design65.canonical_sha256_v65(state_receipts),
        "scheduled_state_occurrences": 64,
        "unique_exact_v53_states": 32,
        "canonical_candidate_identity_inventory_sha256": (
            design65.canonical_sha256_v65(canonical_candidates)
        ),
        "canonical_runtime_identity_inventory_sha256": (
            design65.canonical_sha256_v65(canonical_runtimes)
        ),
        "occurrence_candidate_identity_inventory_sha256": (
            design65.canonical_sha256_v65(occurrence_candidates)
        ),
        "occurrence_runtime_identity_inventory_sha256": (
            design65.canonical_sha256_v65(occurrence_runtimes)
        ),
        "all_occurrences_exactly_matched_sealed_v53_states_before_scoring": True,
        "raw_question_answer_or_generation_text_persisted": False,
        "train_holdback_or_exact_sentinel_opened": False,
        "protected_semantics_opened": False,
        "ood_shadow_or_terminal_holdout_opened": False,
        "projection_update_snapshot_or_promotion_performed": False,
    }
    analysis = design65.analyze_signed_metrics_v65(signed)
    return evidence, analysis


def gpu_generation_phase_summary_v65(
    path: Path, expected_pids: dict[int, int],
) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line]
    phases = [
        f"population_state_{index}_generation_all_actors"
        for index in range(design65.STATE_COUNT_V65)
    ]
    by_phase = {}
    foreign = 0
    for phase in phases:
        by_gpu = {}
        for gpu in range(4):
            selected = [row for row in rows
                        if row["phase"] == phase and row["gpu"] == gpu]
            resident = [row for row in selected
                        if expected_pids[gpu] in row["compute_pids"]]
            foreign += sum(len(row["foreign_compute_pids"]) for row in selected)
            if not resident or not any(row["utilization_percent"] > 0 for row in resident):
                raise RuntimeError(
                    f"v65 GPU {gpu} lacked positive activity in {phase}"
                )
            by_gpu[str(gpu)] = {
                "samples": len(selected),
                "resident_samples": len(resident),
                "positive_resident_samples": sum(
                    row["utilization_percent"] > 0 for row in resident
                ),
                "peak_utilization_percent": max(
                    row["utilization_percent"] for row in resident
                ),
                "peak_memory_used_mib": max(
                    row["memory_used_mib"] for row in resident
                ),
            }
        by_phase[phase] = by_gpu
    if foreign != 0:
        raise RuntimeError("v65 observed a foreign GPU compute process")
    return {
        "schema": "v65-per-generation-phase-four-gpu-activity",
        "generation_phases": len(phases),
        "all_64_generation_phases_positive_on_all_four_gpus": True,
        "foreign_compute_process_observations": 0,
        "by_phase": by_phase,
    }


def execute_v65(preregistration: dict, args) -> int:
    import run_lora_es_generation_boundary_v48b as v48b
    import run_lora_es_transition_microbenchmark_v51 as v51

    prior = v48b.v43i
    v40a = prior.v40a
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v65 requires fresh artifact paths")
    expectation = runtime64.base_model_artifact_expectation_v64()
    pre_model_receipt = runtime64.verify_base_model_artifacts_v64(expectation)
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "v65-robust-sampling-population-attempt",
        "status": "launching_measurement_only",
        "phase": "before_train_semantics_model_ray_or_gpu_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "base_model_artifact_receipt": pre_model_receipt,
        "preflight": preflight,
        "train_semantics_opened": False,
        "protected_semantics_opened": False,
        "train_holdback_or_exact_sentinel_opened": False,
        "optimizer_update_projection_or_promotion": False,
    })
    v40a.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = operations = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    cleanup = idle = None
    try:
        panel, _panel61c = _load_hash_only_panel_v65(preregistration)
        v53_arm = design65.read_exact_self_hashed_v65(
            design65.V53_SIGMA_ARM, design65.V53_SIGMA_ARM_FILE_SHA256,
            design65.V53_SIGMA_ARM_CONTENT_SHA256,
        )
        expected_states = design65.expected_v53_state_identities_v65(v53_arm)
        with v48b.patched_v43i_v48b(), runtime52.patched_runtime_v52(prior):
            adapter_contract = runtime52.verify_adapter_contract_v52()
            v40a.base.set_seed(GENERATION_SEED_V65)
            trainer, saved = prior._make_trainer(preregistration)
            actor_ids = trainer._resolve([
                engine.runtime_identity_v40a.remote()
                for engine in trainer.engines
            ])
            worker_ids = v40a._rpc_all(trainer, "runtime_identity_v40a")
            pid_map = v40a.validate_identities(actor_ids, worker_ids)
            monitor = threading.Thread(
                target=v40a.monitor_gpus,
                args=(stop, phase, pid_map, GPU_LOG, failures),
                daemon=True,
            )
            monitor.start()
            requests, params, answers, input_receipt = (
                prepare_ranking_requests_v65(trainer, prior, panel)
            )
            phase.value = "install_exact_v434_master_all_actors"
            installations = v40a._rpc_all(
                trainer, "install_adapter_state_v41a", (
                    str(prior.SOURCE_WEIGHTS), str(prior.SOURCE_CONFIG),
                    v40a.file_sha256(prior.SOURCE_WEIGHTS),
                    v40a.file_sha256(prior.SOURCE_CONFIG),
                ),
            )
            certificates = v40a._rpc_all(
                trainer, "adapter_state_certificate_v41a",
            )
            if (
                {row["current_identity"]["sha256"] for row in certificates}
                != {design52.MASTER_SHA256_V52}
                or {row["materialization"]["runtime_values_sha256"]
                    for row in certificates}
                != {design52.MASTER_RUNTIME_SHA256_V52}
            ):
                raise RuntimeError("v65 exact V434 master install changed")
            references = v40a._rpc_all(
                trainer, "capture_adapter_reference_v41a",
            )
            reference_generation = references[0]["reference_generation"]
            if {row["reference_generation"] for row in references} != {
                reference_generation
            }:
                raise RuntimeError("v65 reference generations differ")
            operations = RayRobustPopulationOperationsV65(
                trainer, requests, params, panel["items"], answers,
                expected_states, design52.MASTER_SHA256_V52,
                design52.MASTER_RUNTIME_SHA256_V52, phase,
                prior=prior, v40a=v40a,
            )
            completed, final_restore = run_direct_master_pipeline_v65(operations)
            operations.close()
            operations = None
            final_certificates = v40a._rpc_all(
                trainer, "adapter_state_certificate_v52",
            )
            final_master_gate = runtime52.validate_exact_master_state_certificates_v52(
                final_certificates, certificates[0]["current_identity"],
                design52.MASTER_RUNTIME_SHA256_V52,
                phase="v65_final_restored", reference_generation=reference_generation,
                update_sequence=0, controller_transaction_quiescent=True,
            )
            evidence_value, analysis_value = compile_population_v65(
                completed, panel, expected_states,
            )
            evidence_value.update({
                "preregistration_content_sha256": args.preregistration_content_sha256,
                "authorized_input_receipt": input_receipt,
                "adapter_contract": adapter_contract,
                "installations": installations,
                "initial_certificates": certificates,
                "references": references,
                "final_restore": final_restore,
                "final_master_gate": final_master_gate,
            })
            evidence_artifact = _write_self_hashed_v65(EVIDENCE, evidence_value)
            analysis_value.update({
                "source_evidence_content_sha256": evidence_artifact[
                    "content_sha256_before_self_field"
                ],
            })
            analysis_artifact = _write_self_hashed_v65(ANALYSIS, analysis_value)
            stop.set()
            monitor.join(timeout=10)
            if monitor.is_alive() or not failures.empty():
                raise RuntimeError("v65 GPU monitor failed") from (
                    failures.get() if not failures.empty() else None
                )
            gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
            gpu_phases = gpu_generation_phase_summary_v65(GPU_LOG, pid_map)
            cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
            trainer = None
            import ray
            ray.shutdown()
            idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        post_model_receipt = runtime64.verify_base_model_artifacts_v64(expectation)
        if post_model_receipt != pre_model_receipt:
            raise RuntimeError("v65 base-model bytes changed during run")
        report = _write_self_hashed_v65(REPORT, {
            "schema": "v65-robust-sampling-population-report",
            "status": "complete_measurement_only_no_update",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "attempt": {
                "path": str(ATTEMPT),
                "file_sha256": design65.file_sha256_v65(ATTEMPT),
                "content_sha256": attempt["content_sha256_before_self_field"],
            },
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": design65.file_sha256_v65(EVIDENCE),
                "content_sha256": evidence_artifact[
                    "content_sha256_before_self_field"
                ],
            },
            "analysis": {
                "path": str(ANALYSIS),
                "file_sha256": design65.file_sha256_v65(ANALYSIS),
                "content_sha256": analysis_artifact[
                    "content_sha256_before_self_field"
                ],
                "discriminability_gate": analysis_artifact[
                    "discriminability_gate"
                ],
                "coefficients_actionable_for_later_preregistered_projection": (
                    analysis_artifact[
                        "coefficients_actionable_for_later_preregistered_projection"
                    ]
                ),
            },
            "base_model_prelaunch_artifact_receipt": pre_model_receipt,
            "base_model_postrun_artifact_receipt": post_model_receipt,
            "gpu_activity": gpu,
            "gpu_generation_phases": gpu_phases,
            "gpu_log_file_sha256": design65.file_sha256_v65(GPU_LOG),
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "raw_question_answer_or_generation_text_persisted": False,
            "train_holdback_or_exact_sentinel_opened": False,
            "protected_semantics_opened": False,
            "ood_shadow_or_terminal_holdout_opened": False,
            "projection_update_snapshot_or_promotion_performed": False,
        })
        print(json.dumps({
            "report_file_sha256": design65.file_sha256_v65(REPORT),
            "report_content_sha256": report["content_sha256_before_self_field"],
            "discriminability_gate_passed": report["analysis"][
                "discriminability_gate"
            ]["passed"],
            "coefficients_actionable_for_later_preregistered_projection": (
                report["analysis"][
                    "coefficients_actionable_for_later_preregistered_projection"
                ]
            ),
            "all_64_generation_phases_positive_on_all_four_gpus": True,
            "projection_update_or_promotion_performed": False,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        if operations is not None:
            try:
                operations.close()
            except Exception:
                pass
        cleanup_failure = None
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
                }
        _write_self_hashed_v65(FAILURE, {
            "schema": "v65-robust-sampling-population-failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "cleanup": cleanup,
            "cleanup_failure": cleanup_failure,
            "final_gpu_idle": idle,
            "raw_question_answer_or_generation_text_persisted": False,
            "train_holdback_or_exact_sentinel_opened": False,
            "protected_semantics_opened": False,
            "ood_shadow_or_terminal_holdout_opened": False,
            "projection_update_snapshot_or_promotion_performed": False,
        })
        raise
    finally:
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved


def parser_v65() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv=None) -> int:
    args = parser_v65().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("v65 requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v65(args)
    if args.dry_run:
        print(json.dumps({
            "schema": preregistration["schema"],
            "scheduled_state_occurrences": design65.STATE_COUNT_V65,
            "unique_exact_v53_states": 32,
            "generation_completions": design65.GENERATION_COMPLETIONS_V65,
            "four_gpus": True,
            "train_semantics_model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "train_holdback_or_exact_sentinel_opened": False,
            "protected_semantics_opened": False,
            "projection_update_or_promotion_authorized": False,
        }, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != design52.REQUIRED_PYTHON_V52:
        raise RuntimeError("v65 requires the sealed es-at-scale interpreter")
    return execute_v65(preregistration, args)


if __name__ == "__main__":
    raise SystemExit(main())
