#!/usr/bin/env python3
"""Pure scheduling core for a one-state LoRA-ES scoring pipeline.

The controller may generate state N+1 while CPU workers score state N, but it
never has more than one state submitted for scoring.  Every materialization is
followed by a restore attempt before another state can be materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class SignedStateV50:
    direction: int
    label: str
    sign: int

    def __post_init__(self) -> None:
        if self.direction < 0:
            raise ValueError("v50 direction must be nonnegative")
        if (self.label, self.sign) not in (("plus", 1), ("minus", -1)):
            raise ValueError("v50 signed state must be plus/+1 or minus/-1")


@dataclass(frozen=True)
class CompletedStateV50:
    state: SignedStateV50
    materialization: Any
    restoration: Any
    actor_scores: Any


@dataclass
class PipelineOperationsV50:
    materialize: Callable[[SignedStateV50], Any]
    launch_generation: Callable[[SignedStateV50], Any]
    wait_generation: Callable[[SignedStateV50, Any], None]
    restore: Callable[[SignedStateV50], Any]
    submit_scoring: Callable[[SignedStateV50, Any], Any]
    resolve_scoring: Callable[[SignedStateV50, Any], Any]
    cancel_scoring: Callable[[SignedStateV50, Any], None]


@dataclass(frozen=True)
class _PendingStateV50:
    state: SignedStateV50
    materialization: Any
    restoration: Any
    scoring_handles: Any


class ExactRestoreFailureV50(RuntimeError):
    """A state could not be proven restored after work on that state began."""

    def __init__(
        self,
        state: SignedStateV50,
        work_error: BaseException | None,
        restore_error: BaseException,
    ) -> None:
        detail = (
            f" after {type(work_error).__name__}: {work_error}"
            if work_error is not None else ""
        )
        super().__init__(
            f"v50 exact restore failed for direction={state.direction} "
            f"sign={state.label}{detail}: {restore_error}"
        )
        self.state = state
        self.work_error = work_error
        self.restore_error = restore_error


def signed_states_v50(population_size: int) -> tuple[SignedStateV50, ...]:
    """Return the frozen V48 presentation order: plus then minus per seed."""
    if int(population_size) != population_size or population_size <= 0:
        raise ValueError("v50 population size must be a positive integer")
    return tuple(
        SignedStateV50(direction, label, sign)
        for direction in range(int(population_size))
        for label, sign in (("plus", 1), ("minus", -1))
    )


def _drain_v50(
    pending: _PendingStateV50,
    operations: PipelineOperationsV50,
) -> CompletedStateV50:
    scores = operations.resolve_scoring(
        pending.state, pending.scoring_handles,
    )
    return CompletedStateV50(
        state=pending.state,
        materialization=pending.materialization,
        restoration=pending.restoration,
        actor_scores=scores,
    )


def run_one_state_double_buffer_v50(
    states: Iterable[SignedStateV50],
    operations: PipelineOperationsV50,
) -> list[CompletedStateV50]:
    """Run a bounded scoring pipeline without changing state presentation.

    A prior state's CPU scoring overlaps only the current state's generation.
    The prior score is drained after the current state is exactly restored and
    before the current state is submitted, so at most one scoring state is ever
    outstanding.  Results are appended in input order, never completion order.
    """
    pending: _PendingStateV50 | None = None
    completed: list[CompletedStateV50] = []
    try:
        for state in states:
            if not isinstance(state, SignedStateV50):
                raise TypeError("v50 pipeline received an invalid state")
            materialization = None
            generation_handles = None
            work_error: BaseException | None = None
            try:
                materialization = operations.materialize(state)
                generation_handles = operations.launch_generation(state)
                operations.wait_generation(state, generation_handles)
            except BaseException as error:
                work_error = error

            try:
                restoration = operations.restore(state)
            except BaseException as restore_error:
                raise ExactRestoreFailureV50(
                    state, work_error, restore_error,
                ) from restore_error
            if work_error is not None:
                raise work_error

            # Generation above overlaps the only permitted outstanding score.
            # Drain it before submitting this state, preserving a queue depth of
            # one state rather than merely one task per actor.
            if pending is not None:
                completed.append(_drain_v50(pending, operations))
                pending = None

            scoring_handles = operations.submit_scoring(
                state, generation_handles,
            )
            pending = _PendingStateV50(
                state=state,
                materialization=materialization,
                restoration=restoration,
                scoring_handles=scoring_handles,
            )

        if pending is not None:
            completed.append(_drain_v50(pending, operations))
            pending = None
        return completed
    except BaseException as error:
        if pending is not None:
            try:
                operations.cancel_scoring(
                    pending.state, pending.scoring_handles,
                )
            except BaseException as cancel_error:
                error.add_note(
                    "v50 scoring cancellation also failed: "
                    f"{type(cancel_error).__name__}: {cancel_error}"
                )
        raise
