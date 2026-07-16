#!/usr/bin/env python3

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import build_lora_es_transition_microbenchmark_preregistration_v51 as builder
import eggroll_es_worker_lora_v51 as worker
import lora_es_direct_master_pipeline_v51 as pipeline
import lora_es_transition_microbenchmark_v51 as planning
import run_lora_es_transition_microbenchmark_v51 as runtime


def _inventory(count=16):
    return [{
        "state_index": index,
        "direction": index // 2,
        "label": "plus" if index % 2 == 0 else "minus",
        "sign": 1 if index % 2 == 0 else -1,
        "seed": 1000 + index // 2,
        "candidate_identity_sha256": f"{index + 1:064x}",
        "runtime_values_sha256": f"{index + 101:064x}",
    } for index in range(count)]


class _MockOperations:
    def __init__(self, fail=None, restore_fail=False):
        self.events = []
        self.pending = 0
        self.maximum_pending = 0
        self.fail = fail
        self.restore_fail = restore_fail
        self.restore_calls = 0

    @staticmethod
    def _actors(schema, state, elapsed=1):
        return [{
            "schema": schema,
            "state_index": state.state_index,
            "actor_rank": rank,
            "elapsed_ns": elapsed,
        } for rank in range(4)]

    def transition(self, state, previous):
        self.events.append(("transition", state.state_index))
        if self.fail == ("transition", state.state_index):
            raise ValueError("mock transition failure")
        prior = (
            "master" if previous is None
            else previous.candidate_identity_sha256
        )
        actors = self._actors("transition", state)
        for row in actors:
            row.update({
                "previous_candidate_sha256": prior,
                "candidate_identity": {
                    "sha256": state.candidate_identity_sha256,
                },
                "intermediate_master_restore_elided": previous is not None,
            })
        return {"schema": "transition", "actors": actors}

    def launch_generation(self, state):
        self.events.append(("launch", state.state_index))
        return [state.state_index] * 4

    def wait_generation(self, state, handles):
        self.events.append(("generate", state.state_index))
        if self.fail == ("generate", state.state_index):
            raise ValueError("mock generation failure")
        assert handles == [state.state_index] * 4
        return {"schema": "generate", "actors": self._actors("generate", state)}

    def submit_scoring(self, state, handles):
        self.events.append(("submit", state.state_index))
        assert self.pending == 0
        self.pending += 1
        self.maximum_pending = max(self.maximum_pending, self.pending)
        return handles

    def resolve_scoring(self, state, handles):
        self.events.append(("drain", state.state_index))
        if self.fail == ("drain", state.state_index):
            raise ValueError("mock drain failure")
        assert self.pending == 1
        self.pending -= 1
        scores = [{"actor_rank": rank} for rank in range(4)]
        score = {"schema": "score", "actors": self._actors("score", state)}
        drain = {"schema": "drain", "actors": self._actors("drain", state)}
        return scores, score, drain

    def final_restore(self, reason):
        self.events.append(("restore", reason))
        self.restore_calls += 1
        if self.restore_fail:
            raise ValueError("mock restore failure")
        state = SimpleNamespace(state_index=15)
        actors = self._actors("restore", state)
        for row in actors:
            row["restored"] = True
        return {
            "schema": "restore", "mode": "actual_exact_final_restore",
            "actors": actors,
        }

    def cancel_scoring(self, state, _handles):
        self.events.append(("cancel", state.state_index))
        self.pending = 0

    def operations(self):
        return pipeline.PipelineOperationsV51(
            transition=self.transition,
            launch_generation=self.launch_generation,
            wait_generation=self.wait_generation,
            submit_scoring=self.submit_scoring,
            resolve_scoring=self.resolve_scoring,
            final_restore=self.final_restore,
            cancel_scoring=self.cancel_scoring,
        )


def test_v51_pipeline_eliminates_15_restores_and_times_all_actor_phases():
    states = pipeline.signed_states_v51(_inventory())
    mock = _MockOperations()
    completed, restored = pipeline.run_direct_master_pipeline_v51(
        states, mock.operations(),
    )
    coverage = pipeline.validate_complete_timing_v51(completed, restored)
    assert mock.maximum_pending == 1
    assert mock.pending == 0
    assert mock.restore_calls == 1
    assert coverage["phase_actor_receipts"] == {
        phase: 64 for phase in pipeline.PHASES_V51
    }
    assert coverage["intermediate_restore_actor_receipts_elided"] == 60
    assert coverage["actual_final_restore_actor_receipts"] == 4
    for left, right in zip(states, states[1:]):
        assert mock.events.index(("submit", left.state_index)) < (
            mock.events.index(("transition", right.state_index))
        )
        assert mock.events.index(("transition", right.state_index)) < (
            mock.events.index(("drain", left.state_index))
        )


@pytest.mark.parametrize("phase,index", [
    ("transition", 1), ("generate", 1), ("drain", 0),
])
def test_v51_pipeline_failures_attempt_exact_final_restore(phase, index):
    mock = _MockOperations(fail=(phase, index))
    with pytest.raises(ValueError, match="mock"):
        pipeline.run_direct_master_pipeline_v51(
            pipeline.signed_states_v51(_inventory(4)), mock.operations(),
        )
    assert mock.restore_calls == 1
    assert mock.pending == 0


def test_v51_pipeline_restore_failure_preserves_original_error():
    mock = _MockOperations(fail=("generate", 1), restore_fail=True)
    with pytest.raises(pipeline.ExactFinalRestoreFailureV51) as captured:
        pipeline.run_direct_master_pipeline_v51(
            pipeline.signed_states_v51(_inventory(4)), mock.operations(),
        )
    assert isinstance(captured.value.work_error, ValueError)
    assert isinstance(captured.value.restore_error, ValueError)


def _fake_worker(monkeypatch):
    value = worker.LoRAAdapterStateWorkerExtensionV51.__new__(
        worker.LoRAAdapterStateWorkerExtensionV51
    )
    master_sha = "a" * 64
    runtime_master = "b" * 64
    value.device = torch.device("cpu")
    value._v41_master = {"kind": torch.tensor(0)}
    value._v41_current_identity = {"sha256": master_sha}
    value._v41_reference_identity = value._v41_current_identity
    value._v41_reference_fresh = True
    value._v41_active_perturbation = None
    value._v41_pending_update = None
    value._v41_committed_rollback = None
    value._v43i_accepted_rollback = None
    value._require_installed_v41a = lambda: None
    value._base_check_v41a = lambda phase: {
        "phase": phase, "unchanged": True,
    }

    def identity(tensors):
        kind = int(tensors["kind"].item())
        return {"sha256": master_sha if kind == 0 else f"{kind:064x}"}

    def candidate(master, seed, _sigma, sign, _device):
        assert master is value._v41_master
        return {"kind": torch.tensor(seed + (1 if sign > 0 else 100))}

    def materialize(tensors, phase):
        kind = int(tensors["kind"].item())
        identity = master_sha if kind == 0 else f"{kind:064x}"
        return {
            "phase": phase,
            "runtime_values_sha256": (
                runtime_master if kind == 0 else identity[::-1]
            ),
        }

    monkeypatch.setattr(worker.state_v41a, "adapter_identity_v41a", identity)
    monkeypatch.setattr(worker.state_v41a, "antithetic_candidate_v41a", candidate)
    value._materialize_v41a = materialize
    return value, master_sha, runtime_master


def test_v51_worker_transitions_twice_from_pinned_master_without_restore(monkeypatch):
    value, master_sha, runtime_master = _fake_worker(monkeypatch)
    first = f"{8:064x}"
    second = f"{107:064x}"
    one = value.transition_antithetic_from_pinned_master_v51(
        0, 7, 0.0006, 1, master_sha, master_sha, first, first[::-1],
    )
    two = value.transition_antithetic_from_pinned_master_v51(
        1, 7, 0.0006, -1, master_sha, first, second, second[::-1],
    )
    assert one["intermediate_master_restore_elided"] is False
    assert two["intermediate_master_restore_elided"] is True
    assert two["direct_from_pinned_fp32_master"] is True
    assert two["cumulative_candidate_delta_used"] is False
    restored = value.restore_pinned_master_v51(
        master_sha, runtime_master, "test",
    )
    assert restored["restored"] is True
    assert value._v41_active_perturbation is None


def test_v51_worker_rejects_broken_prior_candidate_continuity(monkeypatch):
    value, master_sha, _runtime_master = _fake_worker(monkeypatch)
    first = f"{8:064x}"
    value.transition_antithetic_from_pinned_master_v51(
        0, 7, 0.0006, 1, master_sha, master_sha, first, first[::-1],
    )
    with pytest.raises(RuntimeError, match="continuity"):
        value.transition_antithetic_from_pinned_master_v51(
            1, 7, 0.0006, -1, master_sha, "f" * 64,
            f"{107:064x}", f"{107:064x}"[::-1],
        )


def test_v51_runtime_submits_rank_bound_state_tags_to_cpu_scorers():
    class RemoteMethod:
        def remote(self, *args):
            return args

    value = runtime.RayPopulationOperationsV51.__new__(
        runtime.RayPopulationOperationsV51
    )
    value.scorers = [
        SimpleNamespace(score_v51=RemoteMethod()) for _ in range(4)
    ]
    state = pipeline.signed_states_v51(_inventory(2))[0]
    submitted = value.submit_scoring(
        state, {"refs": [f"ref-{rank}" for rank in range(4)]},
    )
    assert [item[0]["actor_rank"] for item in submitted] == list(range(4))
    assert [item[1] for item in submitted] == [
        f"ref-{rank}" for rank in range(4)
    ]


def test_v51_design_seals_exact_states_and_historical_bottleneck():
    value = planning.build_design_v51()
    selected = value["selected_transition"]
    assert selected["intermediate_master_restores_eliminated"] == 15
    assert selected["runtime_slots"] == 1
    assert selected["extra_gpu_memory_required"] is False
    assert planning.canonical_sha256_v51(
        selected["expected_states"]
    ) == planning.EXPECTED_STATE_INVENTORY_SHA256_V51
    baseline = value["historical_train_only_baseline"]
    assert baseline["v50"]["synchronized_idle_gap_count"] == 17
    assert baseline["v50"]["synchronized_idle_gap_median_seconds"] > 7.5
    assert value["protected_semantics_opened"] is False


def test_v51_preregistration_and_dry_run_are_zero_write_and_protected_free(
    tmp_path, monkeypatch, capsys,
):
    value = builder.build_v51()
    path = tmp_path / "prereg_v51.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    file_sha = planning.file_sha256_v51(path)
    monkeypatch.setattr(
        runtime, "RayPopulationOperationsV51",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("v51 dry-run entered GPU runtime")
        ),
    )
    before = set(tmp_path.iterdir())
    assert runtime.main([
        "--preregistration", str(path),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    after = set(tmp_path.iterdir())
    assert before == after
    output = json.loads(capsys.readouterr().out)
    assert output["train_semantics_loaded"] is False
    assert output["model_or_gpu_loaded"] is False
    assert output["protected_semantic_access_count"] == 0
    assert output["filesystem_writes"] is False


def test_v51_runtime_is_microbenchmark_only_and_strictly_cleans_failures():
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    assert "strict_close_trainer_v38a" in source
    assert "wait_for_gpu_idle" in source
    assert "optimizer_update_performed\": False" in source
    assert "_prepare_execute_update" not in source
    assert "V49B" not in source and "v437" not in source
