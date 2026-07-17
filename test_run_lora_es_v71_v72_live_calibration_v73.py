from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_lora_es_v71_v72_live_calibration_preregistration_v73 as builder
import eggroll_es_audit_contract_v71 as audit_v71
import eggroll_es_host_state_contract_v72 as host_v72
import run_lora_es_mirrored_calibration_v66 as v66
import run_lora_es_v71_v72_live_calibration_v73 as runtime


def write_preregistration(path: Path):
    value = builder.build_preregistration_v73()
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return value


def test_builder_is_deterministic_and_binds_accepted_v66d_and_sources():
    first = builder.build_preregistration_v73()
    second = builder.build_preregistration_v73()
    assert first == second
    compact = {
        key: item for key, item in first.items()
        if key != "content_sha256_before_self_field"
    }
    assert first["content_sha256_before_self_field"] == (
        v66.mirrored.canonical_sha256_v66(compact)
    )
    anchor = first["accepted_v66d_control"]["semantic_anchor"]
    assert anchor["signed_candidate_count"] == 16
    assert anchor["candidate_count_per_actor"] == 4
    assert anchor["candidate_master_sha256"] == (
        "f7f016faf1a067c39efb08718fb7dceef04ee814b0ee72d3809b83813905606f"
    )
    assert first["fixed_recipe"]["integration_v73"][
        "population_acceptance_tokens_are_rank_local"
    ] is True
    for binding in first["implementation_bindings"].values():
        path = Path(binding["path"])
        assert path.is_file()
        assert v66.file_sha256_v66(path) == binding["file_sha256"]


def test_dry_run_resolves_v72_surface_without_writes_or_gpu(tmp_path, capsys):
    preregistration = tmp_path / "v73.json"
    value = write_preregistration(preregistration)
    before = {path: Path(path).exists() for path in runtime.artifacts_v73().values()}
    result = runtime.main([
        "--preregistration", str(preregistration),
        "--preregistration-sha256", v66.file_sha256_v66(preregistration),
        "--preregistration-content-sha256",
        value["content_sha256_before_self_field"],
        "--dry-run",
    ])
    assert result == 0
    assert {path: Path(path).exists() for path in runtime.artifacts_v73().values()} == before
    output = json.loads(capsys.readouterr().out)
    assert output["worker_contract"]["resolved_class"] == (
        runtime.WORKER_EXTENSION_V73
    )
    assert output["rank_local_population_and_update_tokens"] is True
    assert output["host_rss_numa_fault_and_phase_telemetry"] is True
    assert output["train_semantics_model_ray_or_gpu_loaded"] is False
    assert output["filesystem_writes"] is False
    assert output[
        "checkpoint_snapshot_commit_or_promotion_authorized"
    ] is False


def _fake_proc(tmp_path, pid=123):
    root = tmp_path / str(pid)
    root.mkdir()
    (root / "status").write_text(
        "\n".join([
            "Name:\tworker",
            "VmRSS:\t100 kB",
            "VmHWM:\t120 kB",
            "RssAnon:\t60 kB",
            "RssFile:\t35 kB",
            "RssShmem:\t5 kB",
            "VmLck:\t4 kB",
            "VmPin:\t2 kB",
            "Threads:\t8",
        ]) + "\n",
        encoding="ascii",
    )
    fields = ["0"] * 22
    fields[0] = "R"
    fields[7] = "11"
    fields[9] = "2"
    (root / "stat").write_text(
        "123 (worker with spaces) " + " ".join(fields) + "\n",
        encoding="ascii",
    )
    (root / "numa_maps").write_text(
        "1000 default file=/x mapped=2 N0=2 N1=1\n"
        "2000 default anon=3 dirty=3 N1=3\n",
        encoding="ascii",
    )
    return pid


def test_process_snapshot_records_rss_lock_faults_and_numa(tmp_path):
    pid = _fake_proc(tmp_path)
    value = runtime.process_snapshot_v73(
        pid, proc_root=tmp_path, include_numa=True
    )
    assert value["status_bytes"]["VmRSS"] == 100 * 1024
    assert value["status_bytes"]["VmLck"] == 4 * 1024
    assert value["status_bytes"]["VmPin"] == 2 * 1024
    assert value["minor_faults"] == 11
    assert value["major_faults"] == 2
    assert value["numa"]["node_pages"] == {"0": 2, "1": 4}


def _host_row(index, rank, boundary, minor, major):
    binding = {
        "actor_rank": rank,
        "worker_pid": 1000 + rank,
        "physical_gpu_id": rank,
    }
    return runtime._seal_sample_row_v73({
        "schema": "eggroll-es-actor-host-process-sample-v73",
        "sample_index": index,
        "monotonic_ns": 100 + index,
        "phase": boundary,
        "phase_generation": index,
        "boundary": boundary,
        "binding": binding,
        "process": {
            "pid": 1000 + rank,
            "status_bytes": {
                "VmRSS": 1000 + index,
                "VmHWM": 1200 + index,
                "RssAnon": 500,
                "RssFile": 400,
                "RssShmem": 100,
                "VmLck": 0,
                "VmPin": 0,
                "Threads": 8,
            },
            "minor_faults": minor,
            "major_faults": major,
            "numa_included": True,
            "numa": {
                "map_lines": 2,
                "page_size_bytes": 4096,
                "node_pages": {str(rank % 2): 20 + index},
                "node_bytes": {str(rank % 2): (20 + index) * 4096},
                "counters": {"anon": 10},
            },
        },
    })


def test_host_summary_binds_all_actors_and_rejects_fault_regression():
    bindings = [{
        "actor_rank": rank,
        "worker_pid": 1000 + rank,
        "physical_gpu_id": rank,
    } for rank in range(4)]
    rows = []
    for rank in range(4):
        rows.append(_host_row(len(rows), rank, "monitor_start", 10, 1))
    for rank in range(4):
        rows.append(_host_row(len(rows), rank, "pre_cleanup", 20, 2))
    summary = runtime.summarize_host_rows_v73(rows, bindings)
    assert summary["actor_count"] == 4
    assert summary["actors"]["0"]["minor_faults_delta"] == 10
    assert summary["actors"]["3"]["major_faults_delta"] == 1
    bad = copy.deepcopy(rows)
    bad[-1]["process"]["major_faults"] = 0
    bad[-1] = runtime._seal_sample_row_v73({
        key: value for key, value in bad[-1].items() if key != "row_sha256"
    })
    with pytest.raises(RuntimeError, match="fault counters moved backwards"):
        runtime.summarize_host_rows_v73(bad, bindings)


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


def test_rank_local_population_acceptance_cannot_be_broadcast_or_early():
    consensus = runtime.candidate_audit_consensus_v73(_candidate_matrices())
    receipts = [
        _acceptance(item["candidate_audit_sha256"], rank)
        for rank, item in enumerate(consensus["by_rank"])
    ]
    accepted = runtime.population_acceptance_consensus_v73(
        receipts, consensus
    )
    assert len(set(accepted["tokens"])) == 4
    broadcast = [copy.deepcopy(receipts[0]) for _ in range(4)]
    with pytest.raises(RuntimeError, match="population acceptance receipt"):
        runtime.population_acceptance_consensus_v73(broadcast, consensus)
    duplicated = _candidate_matrices()
    duplicated[3]["candidate_audit_sha256"][0] = duplicated[0][
        "candidate_audit_sha256"
    ][0]
    with pytest.raises(RuntimeError, match="aggregate candidate audit"):
        runtime.candidate_audit_consensus_v73(duplicated)


def test_pair_difference_math_runs_only_after_rank_local_acceptance(monkeypatch):
    calls = []

    class Phase:
        value = "restored"

    trainer = object()
    control_population = {
        "plan": {"plan": 1},
        "signed_rewards": [{"reward": 2.0}],
    }
    context = SimpleNamespace(
        trainer=trainer,
        population_equivalence={"passed": True},
        population_acceptance=None,
        candidate_audit_consensus=None,
        phase=Phase(),
        control={"population": control_population},
        preregistration={"fixed_recipe": {"learning_rate": 0.1}},
        record_operation=lambda name, *_args, **_kwargs: calls.append(name),
        capture_host_boundary=lambda name: calls.append(name),
    )
    matrices = _candidate_matrices()
    consensus = runtime.candidate_audit_consensus_v73(matrices)
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
        assert [list(args[0]) for args in args_by_rank] == [
            item["candidate_audit_sha256"] for item in consensus["by_rank"]
        ]
        calls.append("acceptance_rpc")
        return acceptances

    def update_math(plan, rewards, learning_rate):
        assert context.population_acceptance is not None
        calls.append("update_math")
        return {"plan": plan, "rewards": rewards, "lr": learning_rate}

    monkeypatch.setattr(runtime, "_ACTIVE_CONTEXT_V73", context)
    monkeypatch.setattr(runtime, "_rpc_all_v73", rpc_all)
    monkeypatch.setattr(runtime, "_rpc_ranked_v73", rpc_ranked)
    value = runtime._population_accept_then_update_v73(
        update_math,
        control_population["plan"],
        control_population["signed_rewards"],
        0.1,
    )
    assert value["lr"] == 0.1
    assert calls.index("acceptance_rpc") < calls.index("update_math")
    assert context.population_acceptance["rank_local_not_broadcast"] is True


def test_partial_prepare_consensus_always_enters_exact_abort(monkeypatch):
    calls = []

    class Phase:
        value = "population_accepted"

    trainer = object()
    context = SimpleNamespace(
        trainer=trainer,
        population_acceptance={"tokens": [f"{rank + 1:064x}" for rank in range(4)]},
        phase=Phase(),
        control={"update": {}},
        record_operation=lambda name, *_args, **_kwargs: calls.append(name),
        capture_host_boundary=lambda name: calls.append(name),
        residency={},
        update_invariants=None,
    )
    prepared = [{
        "manifest_sha256": "a" * 64,
        "rank": rank,
        "population_acceptance_sha256": "wrong",
        "rollback_aliases_master": True,
        "rollback_clone_bytes": 0,
    } for rank in range(4)]

    def rpc_ranked(*_args, **_kwargs):
        calls.append("prepare")
        return prepared

    def rpc_all(_trainer, method, args=()):
        calls.append(method)
        assert method == "abort_mirrored_update_if_present_v66"
        return [{
            "restored_identity": {"sha256": "m" * 64},
            "terminal_poisoned": False,
        } for _ in range(4)]

    monkeypatch.setattr(runtime, "_ACTIVE_CONTEXT_V73", context)
    monkeypatch.setattr(runtime, "_rpc_ranked_v73", rpc_ranked)
    monkeypatch.setattr(runtime, "_rpc_all_v73", rpc_all)
    update = {
        "coefficients": [1.0],
        "direction_seeds": [7],
        "coefficient_sha256": "c" * 64,
        "worker_population_size": 1,
        "worker_alpha": 0.1,
        "effective_noise_scale": 0.1,
    }
    with pytest.raises(RuntimeError, match="prepare consensus"):
        runtime._execute_and_abort_nonzero_update_v73(
            trainer, update, "m" * 64, 2, context.phase
        )
    assert calls == ["prepare", "abort_mirrored_update_if_present_v66",
                     "exact_update_abort"]

    calls.clear()

    def bad_abort(_trainer, method, args=()):
        calls.append(method)
        return [{
            "restored_identity": {"sha256": "wrong"},
            "terminal_poisoned": False,
        } for _ in range(4)]

    monkeypatch.setattr(runtime, "_rpc_all_v73", bad_abort)
    with pytest.raises(RuntimeError, match="abort master consensus"):
        runtime._execute_and_abort_nonzero_update_v73(
            trainer, update, "m" * 64, 2, context.phase
        )


def _residency(phase, banks):
    value = {
        "schema": "eggroll-es-host-state-residency-v72",
        "phase": phase,
        "roles": ["canonical_master"],
        "role_to_unique_bank": {"canonical_master": 0},
        "unique_owned_bank_count": banks,
        "unique_owned_tensor_bytes": banks * host_v72.MASTER_BYTES_V72,
        "reference_tensor_bank_present": False,
        "full_state_clone_bytes_for_observation": 0,
    }
    value["receipt_sha256"] = host_v72.canonical_sha256_v72(value)
    return value


def test_residency_requires_one_two_one_generation_ownership():
    one = [_residency("quiescent_one_master", 1) for _ in range(4)]
    two = [_residency("executed_candidate_retained", 2) for _ in range(4)]
    assert runtime.validate_residency_receipts_v73(
        one, "quiescent_one_master", 1
    )["aggregate_actor_tensor_bytes"] == 4 * host_v72.MASTER_BYTES_V72
    assert runtime.validate_residency_receipts_v73(
        two, "executed_candidate_retained", 2
    )["aggregate_actor_tensor_bytes"] == 8 * host_v72.MASTER_BYTES_V72
    bad = copy.deepcopy(two)
    bad[1]["unique_owned_bank_count"] = 3
    with pytest.raises(RuntimeError, match="residency receipt"):
        runtime.validate_residency_receipts_v73(
            bad, "executed_candidate_retained", 2
        )


def _traffic_receipt(rank=0):
    observed = {
        "h2d_bytes": 14 * audit_v71.RUNTIME_LORA_BYTES_V71,
        "lora_d2h_bytes": 20 * audit_v71.RUNTIME_LORA_BYTES_V71,
        "base_d2h_bytes": 3 * audit_v71.BASE_BYTES_V71,
        "master_validation_host_copy_bytes": 0,
        "cheap_transition_checks": 1,
        "lora_d2h_calls": 20,
        "exact_base_audits": 3,
        "master_cache_hits": 1,
    }
    observed["total_device_transfer_bytes"] = (
        observed["h2d_bytes"]
        + observed["lora_d2h_bytes"]
        + observed["base_d2h_bytes"]
    )
    return {
        "schema": "eggroll-es-worker-audit-traffic-receipt-v71",
        "observed": observed,
        "byte_accounted_model": {},
        "completed_boundaries": [
            "population_reward_acceptance", "update_acceptance",
        ],
        "population_acceptance_sha256": f"{300 + rank:064x}",
        "terminal_poisoned": False,
    }


def test_audit_traffic_is_exact_and_includes_abort_base_proof():
    receipts = [_traffic_receipt(rank) for rank in range(4)]
    value = runtime.validate_audit_traffic_v73(receipts)
    assert value["per_actor_expected_and_observed"]["exact_base_audits"] == 3
    assert value["aggregate_expected_and_observed"][
        "master_validation_host_copy_bytes"
    ] == 0
    outside = value[
        "known_code_path_device_transfer_outside_worker_counter_per_actor"
    ]
    assert outside["candidate_fp32_master_h2d_bytes"] == (
        4 * audit_v71.MASTER_BYTES_V71
    )
    assert value["known_code_path_device_transfer_total"][
        "per_actor_bytes"
    ] == (
        value["per_actor_expected_and_observed"][
            "total_device_transfer_bytes"
        ] + outside["total_bytes"]
    )
    comparison = value["accepted_v66d_measured_audit_d2h_comparison"]
    assert comparison["saved_bytes"] > 0
    assert comparison["saved_fraction"] > 0.7
    bad = copy.deepcopy(receipts)
    bad[2]["observed"]["lora_d2h_bytes"] += 2
    bad[2]["observed"]["total_device_transfer_bytes"] += 2
    with pytest.raises(RuntimeError, match="audit traffic receipt"):
        runtime.validate_audit_traffic_v73(bad)


def test_population_and_update_equivalence_fail_on_one_value_but_record_manifest_delta():
    control = runtime.load_accepted_control_values_v73()
    population = runtime.population_equivalence_v73(
        control["population"], control["population"]
    )
    assert population["reward_bit_exact"] is True
    changed_population = copy.deepcopy(control["population"])
    changed_population["signed_rewards"][0]["reward"] += 1e-12
    with pytest.raises(RuntimeError, match="signed_rewards"):
        runtime.population_equivalence_v73(
            changed_population, control["population"]
        )

    changed_manifest = copy.deepcopy(control["update"])
    changed_manifest["manifest_sha256"] = "f" * 64
    update = runtime.update_equivalence_v73(
        changed_manifest, control["update"]
    )
    assert update["manifest_sha256_exact_required"] is False
    assert update["prepared_shards_exact"] is True
    changed_candidate = copy.deepcopy(changed_manifest)
    changed_candidate["candidate_master_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="candidate_master_sha256"):
        runtime.update_equivalence_v73(changed_candidate, control["update"])

    actor = runtime.actor_work_equivalence_v73(
        control["actor_rows"], control["actor_rows"]
    )
    assert actor["work_assignment_and_cardinality_exact"] is True
    changed_actor = copy.deepcopy(control["actor_rows"])
    changed_actor[0]["output_cardinality"]["generated_tokens"] += 1
    with pytest.raises(RuntimeError, match="assignments/cardinality"):
        runtime.actor_work_equivalence_v73(
            changed_actor, control["actor_rows"]
        )


def test_v71_population_compactor_derives_only_sealed_legacy_invariants():
    control = runtime.load_accepted_control_values_v73()["population"]
    execution = {
        "plan_sha256": control["plan_sha256"],
        "evaluation_contract_sha256": control[
            "evaluation_contract_sha256"
        ],
        "signed_rewards": control["signed_rewards"],
        "all_submitted_work_drained": True,
        "all_four_actors_restored_after_every_wave": True,
        "materialization_receipts": [],
        "restore_receipts": [],
    }
    for item in control["materializations"]:
        execution["materialization_receipts"].append({
            "direction_seed": item["direction_seed"],
            "sigma": item["sigma"],
            "sign": item["sign"],
            "pair_id": item["pair_id"],
            "evaluation_contract_sha256": item[
                "evaluation_contract_sha256"
            ],
            "master_identity": {"sha256": item["master_sha256"]},
            "candidate_identity": {"sha256": item["candidate_sha256"]},
            "materialization": {
                "runtime_values_sha256": item["runtime_values_sha256"]
            },
            "master_validation_clone_bytes": 0,
            "exact_restore_required": True,
            "timing": {"elapsed_ns": 1},
        })
    for item in control["restorations"]:
        execution["restore_receipts"].append({
            "reason": item["reason"],
            "restored_identity": {
                "sha256": item["restored_master_sha256"]
            },
            "materialization": {
                "runtime_values_sha256": item["runtime_values_sha256"]
            },
            "algebraic_bf16_restore_used": False,
            "terminal_poisoned": False,
            "timing": {"elapsed_ns": 1},
        })
    compact = runtime.compact_population_v73(execution)
    assert compact["materializations"] == control["materializations"]
    assert compact["restorations"] == control["restorations"]
    assert compact["signed_rewards"] == control["signed_rewards"]


def test_patched_surface_restores_every_legacy_global_without_launch():
    preregistration = builder.build_preregistration_v73()
    control = runtime.load_accepted_control_values_v73()
    saved = {
        "run": runtime.v66d.RUN_DIR,
        "worker": runtime.v66d.WORKER_EXTENSION_V66D,
        "context": runtime.v66d.LiveContextV66D,
    }
    with runtime.patched_live_v73(preregistration, control) as context:
        assert isinstance(context, runtime.IntegrationContextV73)
        assert runtime.v66d.RUN_DIR == runtime.RUN_DIR
        assert runtime.v66d.WORKER_EXTENSION_V66D == (
            runtime.WORKER_EXTENSION_V73
        )
    assert runtime.v66d.RUN_DIR is saved["run"]
    assert runtime.v66d.WORKER_EXTENSION_V66D == saved["worker"]
    assert runtime.v66d.LiveContextV66D is saved["context"]
    assert runtime._ACTIVE_CONTEXT_V73 is None
