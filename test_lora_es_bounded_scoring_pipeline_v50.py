#!/usr/bin/env python3

import hashlib
import json
import sys
import types

import pytest

import build_lora_es_generation_boundary_preregistration_v50 as builder
import lora_es_bounded_scoring_pipeline_v50 as pipeline
import run_lora_es_generation_boundary_v50 as subject


def _key(state):
    return state.direction, state.label


def _numeric_score(state, actor_rank):
    # Deliberately use ordinary left-to-right addition so the test catches a
    # change in the within-actor summation order, not just mathematical equality.
    values = [
        1.0e16,
        float(100 * state.direction + 10 * state.sign + actor_rank),
        -1.0e16,
        actor_rank / 7.0,
    ]
    total = 0.0
    for value in values:
        total += value
    return {"actor_rank": actor_rank, "total": total, "values": values}


class _MockOperations:
    def __init__(self, *, fail=None):
        self.events = []
        self.outstanding = 0
        self.maximum_outstanding = 0
        self.fail = fail

    def _maybe_fail(self, phase, state):
        if self.fail == (phase, _key(state)):
            raise ValueError(f"mock {phase} failure {_key(state)}")

    def materialize(self, state):
        self.events.append(("materialize", _key(state)))
        self._maybe_fail("materialize", state)
        return [f"materialized:{_key(state)}:{rank}" for rank in range(4)]

    def launch_generation(self, state):
        self.events.append(("generate", _key(state)))
        self._maybe_fail("generate", state)
        return [
            {"state": _key(state), "actor_rank": rank}
            for rank in range(4)
        ]

    def wait_generation(self, state, handles):
        self.events.append(("wait", _key(state)))
        assert len(handles) == 4
        self._maybe_fail("wait", state)

    def restore(self, state):
        self.events.append(("restore", _key(state)))
        self._maybe_fail("restore", state)
        return [f"restored:{_key(state)}:{rank}" for rank in range(4)]

    def submit_scoring(self, state, generation):
        self.events.append(("submit", _key(state)))
        self._maybe_fail("submit", state)
        assert self.outstanding == 0
        self.outstanding += 1
        self.maximum_outstanding = max(
            self.maximum_outstanding, self.outstanding,
        )
        return generation

    def resolve_scoring(self, state, handles):
        self.events.append(("resolve", _key(state)))
        self._maybe_fail("resolve", state)
        assert self.outstanding == 1
        assert [item["actor_rank"] for item in handles] == list(range(4))
        self.outstanding -= 1
        return [_numeric_score(state, rank) for rank in range(4)]

    def cancel_scoring(self, state, _handles):
        self.events.append(("cancel", _key(state)))
        if self.outstanding:
            self.outstanding -= 1

    def operations(self):
        return pipeline.PipelineOperationsV50(
            materialize=self.materialize,
            launch_generation=self.launch_generation,
            wait_generation=self.wait_generation,
            restore=self.restore,
            submit_scoring=self.submit_scoring,
            resolve_scoring=self.resolve_scoring,
            cancel_scoring=self.cancel_scoring,
        )


def _synchronous(states, mock):
    result = []
    for state in states:
        materialization = mock.materialize(state)
        generation = mock.launch_generation(state)
        mock.wait_generation(state, generation)
        restoration = mock.restore(state)
        handles = mock.submit_scoring(state, generation)
        scores = mock.resolve_scoring(state, handles)
        result.append(pipeline.CompletedStateV50(
            state, materialization, restoration, scores,
        ))
    return result


def _result_sha(value):
    rows = [{
        "state": {
            "direction": item.state.direction,
            "label": item.state.label,
            "sign": item.state.sign,
        },
        "materialization": item.materialization,
        "restoration": item.restoration,
        "actor_scores": item.actor_scores,
    } for item in value]
    raw = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def test_v50_pipeline_is_bit_exact_to_synchronous_actor_scoring():
    states = pipeline.signed_states_v50(3)
    synchronous = _synchronous(states, _MockOperations())
    pipelined_mock = _MockOperations()
    pipelined = pipeline.run_one_state_double_buffer_v50(
        states, pipelined_mock.operations(),
    )
    assert pipelined == synchronous
    assert _result_sha(pipelined) == _result_sha(synchronous)
    assert [item.state for item in pipelined] == list(states)


def test_v50_overlaps_only_one_prior_score_and_restores_before_successor():
    states = pipeline.signed_states_v50(2)
    mock = _MockOperations()
    pipeline.run_one_state_double_buffer_v50(states, mock.operations())
    assert mock.maximum_outstanding == 1
    assert mock.outstanding == 0
    events = mock.events
    first, second = _key(states[0]), _key(states[1])
    assert events.index(("submit", first)) < events.index(("generate", second))
    assert events.index(("generate", second)) < events.index(("resolve", first))
    for left, right in zip(states, states[1:]):
        assert events.index(("restore", _key(left))) < events.index(
            ("materialize", _key(right))
        )
    assert events.index(("restore", second)) < events.index(
        ("submit", second)
    )


@pytest.mark.parametrize("phase", ["materialize", "generate", "wait"])
def test_v50_work_exception_restores_then_propagates_and_cancels(phase):
    states = pipeline.signed_states_v50(2)
    failing = states[1]
    mock = _MockOperations(fail=(phase, _key(failing)))
    with pytest.raises(ValueError, match=f"mock {phase} failure"):
        pipeline.run_one_state_double_buffer_v50(states, mock.operations())
    assert ("restore", _key(failing)) in mock.events
    assert ("cancel", _key(states[0])) in mock.events
    assert mock.outstanding == 0
    assert not any(
        event == "materialize" and key != _key(states[0])
        and key != _key(failing)
        for event, key in mock.events
    )


def test_v50_restore_failure_is_fail_closed_and_preserves_work_error():
    state = pipeline.signed_states_v50(1)[0]
    mock = _MockOperations(fail=("restore", _key(state)))
    with pytest.raises(pipeline.ExactRestoreFailureV50) as captured:
        pipeline.run_one_state_double_buffer_v50([state], mock.operations())
    assert captured.value.state == state
    assert captured.value.work_error is None
    assert isinstance(captured.value.restore_error, ValueError)
    assert not any(event == "submit" for event, _key_value in mock.events)


def test_v50_scoring_exception_propagates_after_current_exact_restore():
    states = pipeline.signed_states_v50(2)
    mock = _MockOperations(fail=("resolve", _key(states[0])))
    with pytest.raises(ValueError, match="mock resolve failure"):
        pipeline.run_one_state_double_buffer_v50(states, mock.operations())
    assert mock.events.index(("restore", _key(states[1]))) < mock.events.index(
        ("resolve", _key(states[0]))
    )
    assert ("submit", _key(states[1])) not in mock.events
    assert ("cancel", _key(states[0])) in mock.events
    assert mock.outstanding == 0


def test_v50_actor_scores_are_placed_by_rank_not_completion_order():
    state = pipeline.signed_states_v50(1)[0]
    rows = [{
        "schema": "generation-boundary-actor-score-v50",
        "state": subject._state_tag_v50(state, rank),
        "gpu_ids": [],
        "score": {"actor_rank": rank, "value": rank + 0.25},
    } for rank in reversed(range(4))]
    ordered = subject.order_actor_scores_v50(state, rows)
    assert [row["actor_rank"] for row in ordered] == list(range(4))
    assert [row["value"] for row in ordered] == [0.25, 1.25, 2.25, 3.25]


def test_v50_cpu_scorer_rejects_any_ray_gpu_assignment(monkeypatch):
    fake_ray = types.SimpleNamespace(get_gpu_ids=lambda: [0])
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(RuntimeError, match="received GPU visibility"):
        subject.PopulationScoringActorV50(
            0, {}, [], {}, {}, [], {},
        )


def test_v50_patch_changes_only_population_and_fresh_artifact_paths():
    saved = {
        "population": subject.v48b.replicated_population_v48b,
        "load": subject.v48b.load_preregistration_v48b,
        "run_dir": subject.v48b.RUN_DIR,
        "candidate_gate": subject.v48b.candidate_gate_v48b,
    }
    with subject.patched_v48b_v50():
        assert subject.v48b.replicated_population_v48b is (
            subject.replicated_population_v50
        )
        assert subject.v48b.load_preregistration_v48b is (
            subject.load_preregistration_v50
        )
        assert subject.v48b.RUN_DIR == subject.RUN_DIR
        assert subject.v48b.candidate_gate_v48b is saved["candidate_gate"]
    assert subject.v48b.replicated_population_v48b is saved["population"]
    assert subject.v48b.load_preregistration_v48b is saved["load"]
    assert subject.v48b.RUN_DIR == saved["run_dir"]


def test_v50_prereg_inherits_v48_objective_and_dry_run_is_gpu_free(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v50()
    parent = subject._load_parent_v50()
    assert value["recipe"] == parent["recipe"]
    assert value["generation_boundary_objective"] == (
        parent["generation_boundary_objective"]
    )
    assert value["uncommitted_candidate_gate"] == (
        parent["uncommitted_candidate_gate"]
    )
    schedule = value["population_scoring_schedule"]
    assert schedule["max_outstanding_scoring_states"] == 1
    assert schedule["gpus_per_scorer_actor"] == 0
    assert schedule["objective_or_gate_changed"] is False
    forbidden = ("shadow", "ood", "holdout", "heldout")
    assert not any(
        token in path.lower()
        for path in value["access_contract"]["only_runtime_train_paths_may_open"]
        for token in forbidden
    )

    path = tmp_path / "prereg_v50.json"
    subject.v48b.v43i.v40a.atomic_json(path, value)
    file_sha = subject.v48b.v43i.v40a.file_sha256(path)
    monkeypatch.setattr(
        subject.v48b, "main",
        lambda _argv: (_ for _ in ()).throw(
            AssertionError("dry-run entered the GPU runtime")
        ),
    )
    monkeypatch.setattr(
        subject, "_RayPopulationOperationsV50",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("dry-run created Ray scoring actors")
        ),
    )
    assert subject.main([
        "--preregistration", str(path),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["train_semantics_loaded"] is False
    assert output["model_or_gpu_loaded"] is False
    assert output["protected_semantic_access_count"] == 0
    assert output["filesystem_writes"] is False
