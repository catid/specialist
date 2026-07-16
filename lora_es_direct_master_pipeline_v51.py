#!/usr/bin/env python3
"""Pure scheduling and timing contracts for V51 population transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


ACTORS_V51 = 4
PHASES_V51 = ("materialize", "generate", "score", "restore", "drain")


@dataclass(frozen=True)
class SignedStateV51:
    state_index: int
    direction: int
    label: str
    sign: int
    seed: int
    candidate_identity_sha256: str
    runtime_values_sha256: str

    def __post_init__(self) -> None:
        if self.state_index < 0 or self.direction < 0 or self.seed < 0:
            raise ValueError("v51 state indices and seeds must be nonnegative")
        if (self.label, self.sign) not in (("plus", 1), ("minus", -1)):
            raise ValueError("v51 signed state must be plus/+1 or minus/-1")
        for value in (
            self.candidate_identity_sha256, self.runtime_values_sha256,
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError("v51 state identities must be SHA-256 strings")


@dataclass(frozen=True)
class PendingStateV51:
    state: SignedStateV51
    transition: Any
    generation: Any
    scoring_handles: Any


@dataclass(frozen=True)
class CompletedStateV51:
    state: SignedStateV51
    transition: Any
    generation: Any
    actor_scores: Any
    score_timing: Any
    restore_timing: Any
    drain_timing: Any


@dataclass
class PipelineOperationsV51:
    transition: Callable[[SignedStateV51, SignedStateV51 | None], Any]
    launch_generation: Callable[[SignedStateV51], Any]
    wait_generation: Callable[[SignedStateV51, Any], Any]
    submit_scoring: Callable[[SignedStateV51, Any], Any]
    resolve_scoring: Callable[[SignedStateV51, Any], tuple[Any, Any, Any]]
    final_restore: Callable[[str], Any]
    cancel_scoring: Callable[[SignedStateV51, Any], None]


class ExactFinalRestoreFailureV51(RuntimeError):
    """The pinned master could not be proven restored after population work."""

    def __init__(self, work_error: BaseException | None, restore_error: BaseException):
        detail = (
            f" after {type(work_error).__name__}: {work_error}"
            if work_error is not None else ""
        )
        super().__init__(
            "v51 exact final pinned-master restore failed"
            f"{detail}: {restore_error}"
        )
        self.work_error = work_error
        self.restore_error = restore_error


def signed_states_v51(inventory: list[dict]) -> tuple[SignedStateV51, ...]:
    result = tuple(SignedStateV51(**item) for item in inventory)
    for index, state in enumerate(result):
        if (
            state.state_index != index
            or state.direction != index // 2
            or (state.label, state.sign)
            != (("plus", 1) if index % 2 == 0 else ("minus", -1))
        ):
            raise ValueError("v51 signed state presentation order changed")
    return result


def _elided_restore_v51(
    previous: SignedStateV51, current: SignedStateV51, transition: dict,
) -> dict:
    actors = []
    transition_actors = transition.get("actors", [])
    if len(transition_actors) != ACTORS_V51:
        raise RuntimeError("v51 transition actor coverage changed")
    for rank, receipt in enumerate(transition_actors):
        if (
            receipt.get("actor_rank") != rank
            or receipt.get("previous_candidate_sha256")
            != previous.candidate_identity_sha256
            or receipt.get("candidate_identity", {}).get("sha256")
            != current.candidate_identity_sha256
            or receipt.get("intermediate_master_restore_elided") is not True
        ):
            raise RuntimeError("v51 elided-restore proof changed")
        actors.append({
            "schema": "elided-intermediate-restore-timing-v51",
            "state_index": previous.state_index,
            "actor_rank": rank,
            "mode": "elided_by_next_direct_pinned_master_transition",
            "elapsed_ns": 0,
            "next_state_index": current.state_index,
            "previous_candidate_sha256": previous.candidate_identity_sha256,
            "next_candidate_sha256": current.candidate_identity_sha256,
            "exact_pinned_master_reconstruction_proved": True,
        })
    return {
        "schema": "elided-intermediate-restore-v51",
        "mode": "elided_by_next_direct_pinned_master_transition",
        "actors": actors,
    }


def _drain_v51(
    pending: PendingStateV51,
    restore_timing: dict,
    operations: PipelineOperationsV51,
) -> CompletedStateV51:
    actor_scores, score_timing, drain_timing = operations.resolve_scoring(
        pending.state, pending.scoring_handles,
    )
    return CompletedStateV51(
        state=pending.state,
        transition=pending.transition,
        generation=pending.generation,
        actor_scores=actor_scores,
        score_timing=score_timing,
        restore_timing=restore_timing,
        drain_timing=drain_timing,
    )


def run_direct_master_pipeline_v51(
    states: Iterable[SignedStateV51],
    operations: PipelineOperationsV51,
) -> tuple[list[CompletedStateV51], dict]:
    """Overlap CPU scoring while chaining runtime states from one FP32 master.

    The next state directly overwrites the sole runtime slot after the current
    generation is complete.  A final exact pinned-master restore is mandatory
    on success and is attempted after every transition/generation failure.
    """
    states = tuple(states)
    if not states:
        raise ValueError("v51 pipeline requires at least one state")
    pending: PendingStateV51 | None = None
    completed: list[CompletedStateV51] = []
    previous: SignedStateV51 | None = None
    transition_attempted = False
    final_restore = None
    try:
        for state in states:
            if not isinstance(state, SignedStateV51):
                raise TypeError("v51 pipeline received an invalid state")
            transition_attempted = True
            transition = operations.transition(state, previous)
            generation_handles = operations.launch_generation(state)
            generation = operations.wait_generation(state, generation_handles)

            if pending is not None:
                restore_timing = _elided_restore_v51(
                    pending.state, state, transition,
                )
                completed.append(_drain_v51(
                    pending, restore_timing, operations,
                ))
                pending = None

            scoring_handles = operations.submit_scoring(
                state, generation_handles,
            )
            pending = PendingStateV51(
                state, transition, generation, scoring_handles,
            )
            previous = state

        final_restore = operations.final_restore("population_complete")
        transition_attempted = False
        if pending is not None:
            completed.append(_drain_v51(
                pending, final_restore, operations,
            ))
            pending = None
        if [item.state for item in completed] != list(states):
            raise RuntimeError("v51 state completion order changed")
        return completed, final_restore
    except BaseException as error:
        if pending is not None:
            try:
                operations.cancel_scoring(
                    pending.state, pending.scoring_handles,
                )
            except BaseException as cancel_error:
                error.add_note(
                    "v51 scoring cancellation also failed: "
                    f"{type(cancel_error).__name__}: {cancel_error}"
                )
        if transition_attempted:
            try:
                operations.final_restore(
                    f"exception:{type(error).__name__}",
                )
            except BaseException as restore_error:
                raise ExactFinalRestoreFailureV51(
                    error, restore_error,
                ) from restore_error
        raise


def validate_complete_timing_v51(
    completed: list[CompletedStateV51], final_restore: dict,
) -> dict:
    """Require five timing phases and four actor receipts for every state."""
    if len(completed) != 16:
        raise RuntimeError("v51 timing requires all 16 signed states")
    phase_rows = {phase: [] for phase in PHASES_V51}
    for expected_index, item in enumerate(completed):
        if item.state.state_index != expected_index:
            raise RuntimeError("v51 timing state order changed")
        values = {
            "materialize": item.transition,
            "generate": item.generation,
            "score": item.score_timing,
            "restore": item.restore_timing,
            "drain": item.drain_timing,
        }
        for phase, value in values.items():
            actors = value.get("actors", [])
            if (
                len(actors) != ACTORS_V51
                or [row.get("actor_rank") for row in actors]
                != list(range(ACTORS_V51))
                or any(
                    not isinstance(row.get("elapsed_ns"), int)
                    or row["elapsed_ns"] < 0
                    for row in actors
                )
            ):
                raise RuntimeError(f"v51 {phase} timing coverage changed")
            phase_rows[phase].extend(actors)
    actual_restores = [
        row for row in phase_rows["restore"]
        if row.get("mode") != "elided_by_next_direct_pinned_master_transition"
    ]
    if (
        len(actual_restores) != ACTORS_V51
        or final_restore != completed[-1].restore_timing
        or any(row.get("restored") is not True for row in actual_restores)
    ):
        raise RuntimeError("v51 final restore timing coverage changed")
    result = {
        "schema": "complete-five-phase-actor-timing-v51",
        "states": len(completed),
        "actors_per_state": ACTORS_V51,
        "phase_actor_receipts": {
            phase: len(rows) for phase, rows in phase_rows.items()
        },
        "intermediate_restore_actor_receipts_elided": (
            len(phase_rows["restore"]) - len(actual_restores)
        ),
        "actual_final_restore_actor_receipts": len(actual_restores),
        "all_five_phases_complete": True,
    }
    return result
