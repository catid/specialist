from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_lora_es_v71_v72_same_live_preregistration_v73b as builder
import eggroll_es_audit_contract_v71 as audit_v71
import run_lora_es_mirrored_calibration_v66 as v66
import run_lora_es_v71_v72_live_calibration_v73 as v73
import run_lora_es_v71_v72_same_live_calibration_v73b as runtime


def _control():
    return v73.load_accepted_control_values_v73()


def _candidate_with_reward_delta(delta=1e-9):
    candidate = copy.deepcopy(_control()["population"])
    candidate["signed_rewards"][0]["reward"] += delta
    candidate["signed_reward_sha256"] = v66.mirrored.canonical_sha256_v66(
        candidate["signed_rewards"]
    )
    return candidate


def test_builder_is_deterministic_and_binds_failed_v73_without_threshold():
    first = builder.build_preregistration_v73b()
    second = builder.build_preregistration_v73b()
    assert first == second
    compact = {
        key: item for key, item in first.items()
        if key != "content_sha256_before_self_field"
    }
    assert first["content_sha256_before_self_field"] == (
        v66.mirrored.canonical_sha256_v66(compact)
    )
    observation = first["failed_v73_observation"]
    assert observation["observed_population_difference_fields"] == [
        "signed_rewards", "signed_reward_sha256",
    ]
    assert observation[
        "numeric_live_reward_values_recoverable_from_failed_artifacts"
    ] is False
    surface = first["fixed_recipe"]["integration_v73b"][
        "reward_acceptance_surface"
    ]
    assert surface["no_post_hoc_numeric_tolerance"] is True
    assert surface["historical_reward_floats"] == (
        "diagnostic_only_not_an_acceptance_gate"
    )
    for binding in first["implementation_bindings"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert v66.file_sha256_v66(path) == binding["file_sha256"]


def test_independent_compiler_is_whole_mapping_exact_for_same_live_rewards():
    population = _control()["population"]
    learning_rate = builder.build_preregistration_v73b()["fixed_recipe"][
        "learning_rate"
    ]
    canonical = v66.mirrored.pair_difference_update_v66(
        population["plan"], population["signed_rewards"], learning_rate
    )
    independent = runtime.independent_pair_difference_update_v73b(
        population["plan"], population["signed_rewards"], learning_rate
    )
    assert independent == canonical
    permuted = list(reversed(population["signed_rewards"]))
    assert runtime.independent_pair_difference_update_v73b(
        population["plan"], permuted, learning_rate
    ) == canonical


@pytest.mark.parametrize("mutation, message", [
    ("duplicate", "coverage or value"),
    ("nonfinite", "coverage or value"),
    ("metadata", "direction_seed changed"),
    ("missing", "matrix is incomplete"),
])
def test_independent_compiler_rejects_bad_live_reward_matrix(mutation, message):
    population = _control()["population"]
    rewards = copy.deepcopy(population["signed_rewards"])
    if mutation == "duplicate":
        rewards[1] = copy.deepcopy(rewards[0])
    elif mutation == "nonfinite":
        rewards[0]["reward"] = math.inf
    elif mutation == "metadata":
        rewards[0]["direction_seed"] += 1
    else:
        rewards.pop()
    with pytest.raises((ValueError, RuntimeError), match=message):
        runtime.independent_pair_difference_update_v73b(
            population["plan"], rewards, 0.01
        )


def test_population_accepts_historical_float_drift_but_not_metadata(monkeypatch):
    control = _control()["population"]
    candidate = _candidate_with_reward_delta()
    context = SimpleNamespace(
        preregistration=builder.build_preregistration_v73b()
    )
    monkeypatch.setattr(v73, "_ACTIVE_CONTEXT_V73", context)
    result = runtime.population_equivalence_v73b(candidate, control)
    assert result["historical_reward_bit_exact"] is False
    assert result["historical_reward_delta"]["maximum_absolute"] > 0.0
    assert result["historical_reward_delta"][
        "acceptance_threshold_applied"
    ] is False
    assert result["reward_assignment_metadata_exact_to_accepted_v66d"] is True

    changed = copy.deepcopy(candidate)
    changed["signed_rewards"][0]["pair_id"] = "f" * 64
    changed["signed_reward_sha256"] = v66.mirrored.canonical_sha256_v66(
        changed["signed_rewards"]
    )
    with pytest.raises(RuntimeError, match="assignment metadata"):
        runtime.population_equivalence_v73b(changed, control)


def _candidate_matrices():
    return [{
        "schema": "eggroll-es-candidate-audit-matrix-v71",
        "candidate_audit_sha256": [
            f"{rank * 4 + item:064x}" for item in range(4)
        ],
        "transition_audit_sha256": [
            f"{100 + rank * 4 + item:064x}" for item in range(4)
        ],
        "candidate_count": 4,
        "all_rewards_provisional": True,
    } for rank in range(4)]


def _acceptance(candidate_hashes, rank):
    value = {
        "schema": "eggroll-es-population-reward-acceptance-v71",
        "candidate_audit_sha256": list(candidate_hashes),
        "candidate_count": 4,
        "boundary_audit_sha256": f"{200 + rank:064x}",
        "rewards_accepted": True,
        "update_authorized": False,
    }
    value["acceptance_sha256"] = audit_v71.canonical_sha256_v71(value)
    return value


def test_same_live_compilers_run_only_after_rank_local_acceptance(monkeypatch):
    calls = []

    class Phase:
        value = "restored"

    population = _control()["population"]
    trainer = object()
    context = SimpleNamespace(
        trainer=trainer,
        population_equivalence={"passed": True},
        population_acceptance=None,
        candidate_audit_consensus=None,
        phase=Phase(),
        preregistration=builder.build_preregistration_v73b(),
        record_operation=lambda name, *_args, **_kwargs: calls.append(name),
        capture_host_boundary=lambda name: calls.append(name),
    )
    matrices = _candidate_matrices()
    consensus = v73.candidate_audit_consensus_v73(matrices)
    acceptances = [
        _acceptance(item["candidate_audit_sha256"], rank)
        for rank, item in enumerate(consensus["by_rank"])
    ]

    def rpc_all(observed_trainer, method, args=()):
        assert observed_trainer is trainer
        assert method == "candidate_audit_matrix_v71"
        calls.append("candidate_rpc")
        return matrices

    def rpc_ranked(observed_trainer, method, args_by_rank):
        assert observed_trainer is trainer
        assert method == "accept_population_rewards_v71"
        calls.append("acceptance_rpc")
        return acceptances

    def canonical(plan, rewards, learning_rate):
        assert context.population_acceptance is not None
        calls.append("canonical_compiler")
        return v66.mirrored.pair_difference_update_v66(
            plan, rewards, learning_rate
        )

    monkeypatch.setattr(v73, "_ACTIVE_CONTEXT_V73", context)
    monkeypatch.setattr(v73, "_rpc_all_v73", rpc_all)
    monkeypatch.setattr(v73, "_rpc_ranked_v73", rpc_ranked)
    result = runtime._population_accept_then_update_v73b(
        canonical,
        population["plan"],
        population["signed_rewards"],
        context.preregistration["fixed_recipe"]["learning_rate"],
    )
    assert result == context.same_live_update
    assert context.same_live_compiler_equivalence[
        "whole_result_mapping_exact"
    ] is True
    assert calls.index("acceptance_rpc") < calls.index("canonical_compiler")


def test_four_actor_executor_consensus_is_required_and_rebinds_private_control(
    monkeypatch,
):
    context = SimpleNamespace(
        same_live_update={"coefficient_sha256": "c" * 64},
        control={"update": {
            "candidate_master_sha256": "old",
            "candidate_runtime_values_sha256": "old-runtime",
        }},
    )
    values = [{
        "candidate_identity": {"sha256": "a" * 64},
        "materialization": {"runtime_values_sha256": "b" * 64},
    } for _ in range(4)]
    monkeypatch.setattr(v73, "_ACTIVE_CONTEXT_V73", context)
    monkeypatch.setattr(runtime, "_BASE_RPC_ALL_V73", lambda *_args: values)
    assert runtime._rpc_all_v73b(
        object(), "execute_sharded_adapter_update_v41a"
    ) == values
    assert context.control["update"]["candidate_master_sha256"] == "a" * 64
    assert context.live_update_execution[
        "all_actor_identities_exact"
    ] is True

    bad = copy.deepcopy(values)
    bad[-1]["candidate_identity"]["sha256"] = "d" * 64
    monkeypatch.setattr(runtime, "_BASE_RPC_ALL_V73", lambda *_args: bad)
    with pytest.raises(RuntimeError, match="identity consensus"):
        runtime._rpc_all_v73b(
            object(), "execute_sharded_adapter_update_v41a"
        )


def test_update_gate_uses_same_live_compiler_and_live_actor_consensus(
    monkeypatch,
):
    control = _control()
    population = control["population"]
    historical = control["update"]
    update = v66.mirrored.pair_difference_update_v66(
        population["plan"],
        population["signed_rewards"],
        builder.build_preregistration_v73b()["fixed_recipe"]["learning_rate"],
    )
    candidate = copy.deepcopy(historical)
    execution = {
        "schema": "eggroll-es-four-actor-live-update-consensus-v73b",
        "actor_count": 4,
        "candidate_master_sha256": candidate["candidate_master_sha256"],
        "candidate_runtime_values_sha256": candidate[
            "candidate_runtime_values_sha256"
        ],
        "candidate_differs_from_master": True,
        "runtime_differs_from_master": True,
        "all_actor_identities_exact": True,
    }
    execution["consensus_sha256"] = v66.mirrored.canonical_sha256_v66(
        execution
    )
    compiler = {
        "schema": "eggroll-es-same-live-compiler-equivalence-v73b",
        "live_signed_reward_sha256": population["signed_reward_sha256"],
        "whole_result_mapping_exact": True,
        "coefficient_sha256": update["coefficient_sha256"],
    }
    compiler["equivalence_sha256"] = v66.mirrored.canonical_sha256_v66(
        compiler
    )
    context = SimpleNamespace(
        same_live_update=update,
        same_live_compiler_equivalence=compiler,
        live_update_execution=execution,
        update_invariants={"manifest_sha256": candidate["manifest_sha256"]},
        historical_update_control=historical,
    )
    monkeypatch.setattr(v73, "_ACTIVE_CONTEXT_V73", context)
    result = runtime.update_equivalence_v73b(candidate, historical)
    assert result["canonical_and_independent_compilers_exact"] is True
    assert result["four_actor_candidate_and_runtime_identity_exact"] is True
    changed = copy.deepcopy(candidate)
    changed["coefficient_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="same-live compiler"):
        runtime.update_equivalence_v73b(changed, historical)


def test_patched_surface_restores_v73_globals_without_launch():
    preregistration = builder.build_preregistration_v73b()
    control = _control()
    saved = {
        "run": v73.RUN_DIR,
        "population_gate": v73.population_equivalence_v73,
        "rpc": v73._rpc_all_v73,
    }
    with runtime.patched_live_v73b(
        preregistration, copy.deepcopy(control)
    ) as context:
        assert v73.RUN_DIR == runtime.RUN_DIR
        assert v73.population_equivalence_v73 is (
            runtime.population_equivalence_v73b
        )
        assert context.historical_update_control == control["update"]
    assert v73.RUN_DIR is saved["run"]
    assert v73.population_equivalence_v73 is saved["population_gate"]
    assert v73._rpc_all_v73 is saved["rpc"]


def test_dry_run_resolves_without_writes_or_gpu(tmp_path, capsys):
    preregistration = tmp_path / "v73b.json"
    value = builder.build_preregistration_v73b()
    preregistration.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    before = {
        path: Path(path).exists() for path in runtime.artifacts_v73b().values()
    }
    assert runtime.main([
        "--preregistration", str(preregistration),
        "--preregistration-sha256", v66.file_sha256_v66(preregistration),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ]) == 0
    assert {
        path: Path(path).exists() for path in runtime.artifacts_v73b().values()
    } == before
    output = json.loads(capsys.readouterr().out)
    assert output["same_live_reward_dual_compiler_equivalence"] == (
        "whole_mapping_exact"
    )
    assert output["historical_reward_floats_are_diagnostic_only"] is True
    assert output["train_semantics_model_ray_or_gpu_loaded"] is False
