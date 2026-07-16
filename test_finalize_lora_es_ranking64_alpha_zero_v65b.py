#!/usr/bin/env python3
"""CPU-only adversarial tests for the independent V65B finalizer."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import build_lora_es_ranking64_alpha_zero_preregistration_v65b as builder
import finalize_lora_es_ranking64_alpha_zero_v65b as subject
import lora_es_ranking64_alpha_zero_calibration_v65b as analysis
import run_lora_es_ranking64_alpha_zero_calibration_v65b as runtime


ROOT = Path(__file__).resolve().parent
R1_EVIDENCE = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v65a_r1_ranking64_alpha_zero_calibration/"
    "ranking64_alpha_zero_evidence_v65a_r1.json"
)


def _self_hashed(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = (
        analysis.v65a.canonical_sha256_v65a(result)
    )
    return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _gpu_rows(reason: int = 4) -> list[dict]:
    phases = [
        f"unscored_warmup_{index}_generation_all_actors" for index in range(8)
    ] + [
        f"scored_period_{index}_generation_all_actors" for index in range(72)
    ]
    started = datetime(2026, 7, 16, tzinfo=timezone.utc)
    rows = []
    for phase_index, phase in enumerate(phases):
        sampled = (started + timedelta(seconds=phase_index)).isoformat()
        for gpu in range(4):
            forbidden = reason & runtime.FORBIDDEN_CLOCK_REASON_MASK_V65B
            rows.append({
                "sampled_at_utc": sampled,
                "phase": phase,
                "generation_phase": True,
                "gpu": gpu,
                "expected_pid": 100 + gpu,
                "compute_pids": [100 + gpu],
                "foreign_compute_pids": [],
                "utilization_percent": 90,
                "memory_used_mib": 1000 + gpu,
                "temperature_c": 80 + gpu,
                "power_draw_mw": 250000 + gpu,
                "clock_event_reasons_bitmask": reason,
                "clock_event_reasons_hex": f"0x{reason:016x}",
                "allowed_diagnostic_reasons_bitmask": (
                    reason & runtime.ALLOWED_DIAGNOSTIC_CLOCK_REASON_MASK_V65B
                ),
                "forbidden_hardware_or_thermal_reasons_bitmask": forbidden,
                "hardware_or_thermal_slowdown_active": bool(forbidden),
            })
    return rows


def _write_gpu_log(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _gpu_report(path: Path, rows: list[dict]) -> dict:
    pid_map = {gpu: 100 + gpu for gpu in range(4)}
    phases = runtime.gpu_period_phase_summary_v65b(path, pid_map)
    by_gpu = {}
    for gpu in range(4):
        selected = [row for row in rows if row["gpu"] == gpu]
        by_gpu[str(gpu)] = {
            "expected_pid": 100 + gpu,
            "samples": len(selected),
            "resident_samples": len(selected),
            "positive_samples": len(selected),
            "mean_resident_utilization_percent": 90.0,
            "peak_utilization_percent": 90,
            "peak_memory_used_mib": 1000 + gpu,
        }
    return {
        "started_at_utc": "2026-07-15T23:59:59+00:00",
        "completed_at_utc": "2026-07-16T00:01:30+00:00",
        "wall_runtime_seconds": 91.0,
        "gpu_period_phases_and_hardware_health": phases,
        "gpu_activity": {"all_four_attributed_positive": True, "by_gpu": by_gpu},
    }


def _r1_evidence() -> dict:
    return json.loads(R1_EVIDENCE.read_text(encoding="utf-8"))


def _expanded_period_edge_evidence() -> dict:
    source = _r1_evidence()
    installed = copy.deepcopy(source["installed_master_state"])
    writes = []
    reads = []
    source_writes = source["exact_master_slot_write_receipts"]
    source_reads = source["read_only_live_slot_receipts"]
    identities = source["actor_runtime_identities"]
    for kind, count in (("unscored_warmup", 8), ("scored", 72)):
        kind_writes = [row for row in source_writes
                       if row["period_kind"] == kind]
        kind_reads = [row for row in source_reads
                      if row["period_kind"] == kind]
        for index in range(count):
            ordinal = len(writes)
            write = copy.deepcopy(kind_writes[index % 4])
            write["period_index"] = index
            for rank, actor in enumerate(write["actors"]):
                actor["schema"] = "exact-master-slot-write-v65b"
                actor["period_index"] = index
                actor["controller_actor_rank"] = rank
                actor["controller_expected_pid"] = identities[rank]["pid"]
                actor["controller_physical_gpu_id"] = identities[rank][
                    "physical_gpu_id"
                ]
                actor["worker_pid"] = identities[rank]["pid"]
                actor["worker_physical_gpu_id"] = identities[rank][
                    "physical_gpu_id"
                ]
                actor["worker_cuda_visible_devices"] = str(
                    identities[rank]["physical_gpu_id"]
                )
                started = (ordinal * 3) * 1_000 + rank * 10
                actor["timing"] = {
                    "clock": "worker_monotonic_ns", "started_ns": started,
                    "ended_ns": started + 5, "elapsed_ns": 5,
                }
            write["actor_receipts_sha256"] = (
                analysis.v65a.canonical_sha256_v65a(write["actors"])
            )
            writes.append(write)
            for edge in ("before_generation", "after_generation"):
                edge_offset = 1 if edge == "before_generation" else 2
                candidates = [row for row in kind_reads
                              if row["edge"] == edge]
                read = copy.deepcopy(candidates[index % 4])
                read["period_index"] = index
                read["aggregate"]["schema"] = (
                    "v65b-read-only-four-actor-master-slot-consensus"
                )
                for rank, actor in enumerate(read["actors"]):
                    actor["schema"] = "read-only-exact-master-slot-v65b"
                    actor["period_index"] = index
                    actor["controller_actor_rank"] = rank
                    actor["controller_expected_pid"] = identities[rank]["pid"]
                    actor["controller_physical_gpu_id"] = identities[rank][
                        "physical_gpu_id"
                    ]
                    actor["worker_pid"] = identities[rank]["pid"]
                    actor["worker_physical_gpu_id"] = identities[rank][
                        "physical_gpu_id"
                    ]
                    actor["worker_cuda_visible_devices"] = str(
                        identities[rank]["physical_gpu_id"]
                    )
                    started = (
                        (ordinal * 3 + edge_offset) * 1_000 + rank * 10
                    )
                    actor["timing"] = {
                        "clock": "worker_monotonic_ns",
                        "started_ns": started, "ended_ns": started + 5,
                        "elapsed_ns": 5,
                    }
                read["actor_receipts_sha256"] = (
                    analysis.v65a.canonical_sha256_v65a(read["actors"])
                )
                reads.append(read)
    return {
        "installed_master_state": installed,
        "actor_runtime_identities": copy.deepcopy(identities),
        "exact_master_slot_write_receipts": writes,
        "exact_master_slot_write_receipts_sha256": (
            analysis.v65a.canonical_sha256_v65a(writes)
        ),
        "read_only_live_slot_receipts": reads,
        "read_only_live_slot_receipts_sha256": (
            analysis.v65a.canonical_sha256_v65a(reads)
        ),
    }


def _complete_state_evidence() -> dict:
    evidence = _expanded_period_edge_evidence()
    reads = evidence["read_only_live_slot_receipts"]
    ordinal = 0
    for key, kind, count in (
        ("warmup_state_receipts", "unscored_warmup", 8),
        ("scored_state_receipts", "scored", 72),
    ):
        values = []
        for index in range(count):
            values.append({
                "period_kind": kind,
                "period_index": index,
                "before": copy.deepcopy(reads[2 * ordinal]["aggregate"]),
                "after": copy.deepcopy(reads[2 * ordinal + 1]["aggregate"]),
                "identical_v434_state": True,
            })
            ordinal += 1
        evidence[key] = values
        evidence[f"{key}_sha256"] = analysis.v65a.canonical_sha256_v65a(
            values
        )
    evidence["final_restored_master_state"] = copy.deepcopy(
        evidence["installed_master_state"]
    )
    return evidence


def test_sha256_and_duplicate_json_are_fail_closed(tmp_path):
    with pytest.raises(RuntimeError, match="SHA-256"):
        subject._require_sha256_v65b("A" * 64, "test")
    path = tmp_path / "duplicate.json"
    path.write_text('{"x":1,"x":2}\n', encoding="utf-8")
    source = subject.SelfHashedSourceV65B(
        path=path,
        file_sha256=subject.file_sha256_v65b(path),
        content_sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        subject._read_self_hashed_v65b(source)


def test_self_hash_and_forbidden_text_are_verified(tmp_path):
    value = _self_hashed({"schema": "test", "numeric": 1})
    path = tmp_path / "value.json"
    _write_json(path, value)
    source = subject.SelfHashedSourceV65B(
        path, subject.file_sha256_v65b(path),
        value["content_sha256_before_self_field"],
    )
    assert subject._read_self_hashed_v65b(source) == value
    tampered = copy.deepcopy(value)
    tampered["numeric"] = 2
    _write_json(path, tampered)
    source = subject.SelfHashedSourceV65B(
        path, subject.file_sha256_v65b(path),
        value["content_sha256_before_self_field"],
    )
    with pytest.raises(RuntimeError, match="content changed"):
        subject._read_self_hashed_v65b(source)
    with pytest.raises(RuntimeError, match="forbidden text-bearing"):
        subject._verify_no_text_keys_v65b("evidence", {"question": "secret"})


def test_self_hashed_source_uses_one_byte_snapshot(tmp_path, monkeypatch):
    first = _self_hashed({"schema": "first", "numeric": 1})
    second = _self_hashed({"schema": "second", "numeric": 2})
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_json(first_path, first)
    _write_json(second_path, second)
    first_hash = subject.file_sha256_v65b(first_path)
    second_path.replace(first_path)
    monkeypatch.setattr(subject, "file_sha256_v65b", lambda _path: first_hash)
    source = subject.SelfHashedSourceV65B(
        first_path, first_hash, second["content_sha256_before_self_field"],
    )
    with pytest.raises(RuntimeError, match="input file changed"):
        subject._read_self_hashed_v65b(source)


def test_panel_metric_identities_are_exact_not_merely_unique(tmp_path):
    panel, _sources = builder.base.sealed_source_bindings_v65a()
    path = tmp_path / "panel.json"
    _write_json(path, panel)
    prereg = {"ranking_panel": {
        "path": str(path),
        "file_sha256": subject.file_sha256_v65b(path),
        "content_sha256": panel["content_sha256_before_self_field"],
    }}
    metrics = [{
        "request_index": index,
        "row_sha256": item["row_sha256"],
        "unit_identity_sha256": item["unit_identity_sha256"],
    } for index, item in enumerate(panel["items"])]
    scored = [[copy.deepcopy(metrics)]]
    assert subject._verify_panel_metric_identities_v65b(
        scored, prereg,
    )["all_64_metric_identities_match_sealed_panel"] is True
    scored[0][0][0]["row_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="differ from sealed panel"):
        subject._verify_panel_metric_identities_v65b(scored, prereg)


def test_all_80_writes_and_160_read_only_edges_are_reverified():
    evidence = _expanded_period_edge_evidence()
    value = subject._verify_period_edges_v65b(evidence)
    assert value["exact_master_slot_writes"] == 80
    assert value["read_only_pre_and_post_generation_edges"] == 160
    assert value["after_generation_edges_wrote_or_reset_slot"] is False

    missing = copy.deepcopy(evidence)
    missing["read_only_live_slot_receipts"].pop()
    missing["read_only_live_slot_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            missing["read_only_live_slot_receipts"]
        )
    )
    with pytest.raises(RuntimeError, match="edge coverage"):
        subject._verify_period_edges_v65b(missing)

    reordered = copy.deepcopy(evidence)
    reordered["exact_master_slot_write_receipts"][0]["period_index"] = 1
    reordered["exact_master_slot_write_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            reordered["exact_master_slot_write_receipts"]
        )
    )
    with pytest.raises(RuntimeError, match="ordered exact-master"):
        subject._verify_period_edges_v65b(reordered)


def test_runtime_evidence_and_independent_finalizer_schemas_integrate(tmp_path):
    source = _r1_evidence()
    edges = _expanded_period_edge_evidence()
    panel, sources = builder.base.sealed_source_bindings_v65a()
    panel_path = tmp_path / "v65b-panel.json"
    panel_path.write_bytes(builder.json_payload_v65b(panel))
    prereg = builder.build_preregistration_v65b(
        panel, sources, builder.implementation_bindings_v65b(),
        ranking_panel_output=panel_path,
    )
    installed = edges["installed_master_state"]

    def state_receipts(kind, count, start_ordinal):
        return [{
            "period_kind": kind,
            "period_index": index,
            "before": copy.deepcopy(
                edges["read_only_live_slot_receipts"][
                    2 * (start_ordinal + index)
                ]["aggregate"]
            ),
            "after": copy.deepcopy(
                edges["read_only_live_slot_receipts"][
                    2 * (start_ordinal + index) + 1
                ]["aggregate"]
            ),
            "identical_v434_state": True,
        } for index in range(count)]

    input_receipt = copy.deepcopy(source["authorized_input_receipt"])
    input_receipt["lora_adapter_request_name"] = (
        "v434_ranking64_alpha_zero_v65b"
    )
    scored = [copy.deepcopy(source["scored_periods"][0]) for _ in range(72)]
    evidence = runtime.build_evidence_v65b(
        panel=panel, input_receipt=input_receipt,
        actor_identities=source["actor_runtime_identities"],
        worker_identities=source["worker_runtime_identities"],
        active_lora_receipts=source["active_lora_receipts"],
        installations=source["initial_installations"],
        installed_master=installed,
        warmup_state_receipts=state_receipts("unscored_warmup", 8, 0),
        scored_state_receipts=state_receipts("scored", 72, 8),
        slot_write_receipts=edges["exact_master_slot_write_receipts"],
        read_only_slot_receipts=edges["read_only_live_slot_receipts"],
        final_master_state=copy.deepcopy(installed),
        scored_periods=scored,
        actor_wait_timeout_receipt=copy.deepcopy(
            runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B
        ),
    )
    evidence = _self_hashed({
        **evidence,
        "adapter_artifact_contract_prelaunch": source[
            "adapter_artifact_contract_prelaunch"
        ],
        "adapter_artifact_contract_postcleanup": source[
            "adapter_artifact_contract_postcleanup"
        ],
        "adapter_artifact_contract_unchanged": True,
    })
    verification, verified_scored, pid_map = subject._verify_evidence_v65b(
        evidence, prereg,
    )
    assert verification["fixed_schedule_and_numeric_coverage"] is True
    assert len(verified_scored) == 72
    assert set(pid_map) == {0, 1, 2, 3}


def test_stored_analysis_must_equal_exact_rebuild(monkeypatch):
    checks = {key: True for key in subject.EXPECTED_GATE_KEYS_V65B}
    rebuilt = {
        "schema": "v65b-ranking64-high-rep-alpha-zero-analysis",
        "status": "complete_numeric_only_high_rep_calibration",
        "primary_cluster_bootstrap": {
            "six_epoch_intervals_are_sealed_non_gating_diagnostics": True,
            "superblock_influence": {"sealed_non_gating_diagnostic": True},
        },
        "actor_influence": {},
        "required_alpha_zero_gate": {
            "checks": checks,
            "passed": True,
            "thresholds_relaxed_or_reinterpreted_after_r1": False,
            "success_directly_authorizes_v65_population": False,
        },
        "r1_used_for_prospective_sample_size_planning_only": True,
        "r1_threshold_relaxation_or_outcome_reinterpretation": False,
        "v65_population_launch_authorized": False,
    }
    monkeypatch.setattr(
        subject.analysis, "analyze_scored_periods_v65b",
        lambda _scored: copy.deepcopy(rebuilt),
    )
    stored = subject._rebuild_analysis_v65b([], "a" * 64)
    observed = subject._verify_analysis_v65b(stored, [], "a" * 64)
    assert observed["gate_observation"]["passed"] is True
    tampered = copy.deepcopy(stored)
    tampered["required_alpha_zero_gate"]["passed"] = False
    with pytest.raises(RuntimeError, match="differs"):
        subject._verify_analysis_v65b(tampered, [], "a" * 64)


def test_gpu_finalizer_accepts_power_cap_and_recomputes_all_80_phases(tmp_path):
    path = tmp_path / "gpu.jsonl"
    rows = _gpu_rows(4)
    _write_gpu_log(path, rows)
    report = _gpu_report(path, rows)
    result = subject._gpu_summary_v65b(
        path, subject.file_sha256_v65b(path),
        {gpu: 100 + gpu for gpu in range(4)}, report,
    )
    assert result["all_eighty_generation_phases_positive_on_all_four_gpus"]
    assert result["thermal_or_hardware_slowdown_observations_in_generation"] == 0
    assert result["foreign_compute_process_observations"] == 0


@pytest.mark.parametrize("reason", [8, 32, 64, 128])
def test_gpu_finalizer_rejects_each_hardware_or_thermal_reason(tmp_path, reason):
    healthy_path = tmp_path / "healthy.jsonl"
    healthy = _gpu_rows(4)
    _write_gpu_log(healthy_path, healthy)
    report = _gpu_report(healthy_path, healthy)
    rows = _gpu_rows(4)
    rows[0]["clock_event_reasons_bitmask"] = reason
    rows[0]["clock_event_reasons_hex"] = f"0x{reason:016x}"
    rows[0]["allowed_diagnostic_reasons_bitmask"] = 0
    rows[0]["forbidden_hardware_or_thermal_reasons_bitmask"] = reason
    rows[0]["hardware_or_thermal_slowdown_active"] = True
    path = tmp_path / f"gpu-{reason}.jsonl"
    _write_gpu_log(path, rows)
    with pytest.raises(RuntimeError, match="attribution or health"):
        subject._gpu_summary_v65b(
            path, subject.file_sha256_v65b(path),
            {gpu: 100 + gpu for gpu in range(4)}, report,
        )


@pytest.mark.parametrize("reason", [8 | 4, 32 | 4, 64 | 4, 128 | 4])
def test_gpu_finalizer_rejects_forbidden_plus_allowed_reasons(tmp_path, reason):
    healthy_path = tmp_path / "healthy.jsonl"
    healthy = _gpu_rows(4)
    _write_gpu_log(healthy_path, healthy)
    report = _gpu_report(healthy_path, healthy)
    path = tmp_path / f"mixed-{reason}.jsonl"
    _write_gpu_log(path, _gpu_rows(reason))
    with pytest.raises(RuntimeError, match="attribution or health"):
        subject._gpu_summary_v65b(
            path, subject.file_sha256_v65b(path),
            {gpu: 100 + gpu for gpu in range(4)}, report,
        )


def test_gpu_finalizer_rejects_foreign_pid_unknown_reason_and_missing_field(tmp_path):
    healthy_path = tmp_path / "healthy.jsonl"
    healthy = _gpu_rows(4)
    _write_gpu_log(healthy_path, healthy)
    report = _gpu_report(healthy_path, healthy)
    for name, mutate in (
        ("foreign", lambda row: (
            row["compute_pids"].append(999),
            row["foreign_compute_pids"].append(999),
        )),
        ("unknown", lambda row: (
            row.__setitem__("clock_event_reasons_bitmask", 512),
            row.__setitem__("clock_event_reasons_hex", "0x0000000000000200"),
            row.__setitem__("allowed_diagnostic_reasons_bitmask", 0),
        )),
        ("float-expected-pid", lambda row: row.__setitem__(
            "expected_pid", float(row["expected_pid"]),
        )),
        ("missing", lambda row: row.pop("temperature_c")),
    ):
        rows = _gpu_rows(4)
        mutate(rows[0])
        path = tmp_path / f"{name}.jsonl"
        _write_gpu_log(path, rows)
        with pytest.raises(RuntimeError):
            subject._gpu_summary_v65b(
                path, subject.file_sha256_v65b(path),
                {gpu: 100 + gpu for gpu in range(4)}, report,
            )


def test_timeout_contract_is_fixed_and_success_requires_final_state_receipts():
    contract = runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B
    assert contract["seconds"] == 300.0
    assert contract["scope"] == [
        "whole_trainer_construction_watchdog",
        "placement_group_ready_waits",
        "constructor_collective_rpc_waits",
        "actor_identity_waits", "collective_rpc_waits",
        "four_actor_generation_waits",
        "reward_pool_terminate_and_join",
        "gpu_preflight_and_nvidia_smi_queries",
        "strict_cleanup_and_four_gpu_idle_proof",
        "final_unconditional_ray_shutdown",
        "terminal_artifact_lock_waits",
    ]
    assert contract["timeout_retry_drop_reorder_or_early_stop"] is False
    assert contract[
        "timeout_enters_strict_engine_cleanup_and_four_gpu_idle_proof"
    ] is True
    assert contract[
        "successful_run_requires_all_after_generation_and_final_exact_state_receipts"
    ] is True


def test_source_paths_failure_absence_and_output_path_are_exact(
    tmp_path, monkeypatch,
):
    artifacts = {
        "run_directory": str(tmp_path / "run"),
        "attempt": str(tmp_path / "attempt.json"),
        "gpu_log": str(tmp_path / "gpu.jsonl"),
        "evidence": str(tmp_path / "evidence.json"),
        "analysis": str(tmp_path / "analysis.json"),
        "report": str(tmp_path / "report.json"),
        "failure": str(tmp_path / "failure.json"),
        "finalized": str(tmp_path / "finalized.json"),
    }
    prereg_path = tmp_path / "prereg.json"
    monkeypatch.setattr(subject.builder, "artifacts_v65b", lambda: artifacts)
    monkeypatch.setattr(subject.builder, "PREREGISTRATION_OUTPUT", prereg_path)

    def source(name):
        return subject.SelfHashedSourceV65B(
            Path(artifacts[name]), "a" * 64, "b" * 64,
        )

    sources = subject.FinalizerSourcesV65B(
        preregistration=subject.SelfHashedSourceV65B(
            prereg_path, "a" * 64, "b" * 64,
        ),
        attempt=source("attempt"), evidence=source("evidence"),
        analysis=source("analysis"), report=source("report"),
        gpu_log_path=Path(artifacts["gpu_log"]),
        gpu_log_file_sha256="c" * 64,
    )
    assert subject._verify_artifact_paths_v65b(
        {"artifacts": artifacts}, sources,
    )["preregistered_failure_artifact_absent"] is True

    mismatched = copy.copy(sources)
    object.__setattr__(
        mismatched, "evidence",
        subject.SelfHashedSourceV65B(
            tmp_path / "other.json", "a" * 64, "b" * 64,
        ),
    )
    with pytest.raises(RuntimeError, match="source arguments"):
        subject._verify_artifact_paths_v65b(
            {"artifacts": artifacts}, mismatched,
        )

    Path(artifacts["failure"]).write_text("failure\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="success-artifact exclusivity"):
        subject._verify_artifact_paths_v65b(
            {"artifacts": artifacts}, sources,
        )

    monkeypatch.setattr(subject, "OUTPUT", tmp_path / "exact.json")
    assert subject._require_finalized_output_path_v65b(
        tmp_path / "exact.json"
    ) == (tmp_path / "exact.json").resolve()
    with pytest.raises(RuntimeError, match="differs from preregistration"):
        subject._require_finalized_output_path_v65b(tmp_path / "other.json")


def test_master_and_state_receipts_require_exact_schemas_and_read_linkage():
    evidence = _complete_state_evidence()
    assert subject._verify_state_receipts_v65b(evidence)[
        "unchanged_period_state_receipts"
    ] == 80

    minimal = copy.deepcopy(evidence)
    minimal["warmup_state_receipts"][0]["before"] = copy.deepcopy(
        minimal["installed_master_state"]
    )
    minimal["warmup_state_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            minimal["warmup_state_receipts"]
        )
    )
    with pytest.raises(RuntimeError, match="read-only state aggregate"):
        subject._verify_state_receipts_v65b(minimal)

    forged = copy.deepcopy(evidence)
    forged_hash = "0" * 64
    forged["installed_master_state"]["four_actor_certificate_sha256"] = (
        forged_hash
    )
    forged["final_restored_master_state"][
        "four_actor_certificate_sha256"
    ] = forged_hash
    with pytest.raises(RuntimeError, match="certificate hash"):
        subject._verify_state_receipts_v65b(forged)


def test_period_edges_reject_duplicate_actors_negative_time_and_bad_order():
    evidence = _expanded_period_edge_evidence()
    duplicate = copy.deepcopy(evidence)
    original_actors = duplicate["exact_master_slot_write_receipts"][0][
        "actors"
    ]
    copied_actors = []
    for rank in range(4):
        copied = copy.deepcopy(original_actors[0])
        for key in runtime.CONTROLLER_ACTOR_BINDING_KEYS_V65B:
            copied[key] = original_actors[rank][key]
        copied["timing"] = copy.deepcopy(original_actors[rank]["timing"])
        copied_actors.append(copied)
    duplicate["exact_master_slot_write_receipts"][0]["actors"] = copied_actors
    duplicate["exact_master_slot_write_receipts"][0][
        "actor_receipts_sha256"
    ] = analysis.v65a.canonical_sha256_v65a(
        duplicate["exact_master_slot_write_receipts"][0]["actors"]
    )
    duplicate["exact_master_slot_write_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            duplicate["exact_master_slot_write_receipts"]
        )
    )
    with pytest.raises(RuntimeError, match="ordered exact-master"):
        subject._verify_period_edges_v65b(duplicate)

    negative = copy.deepcopy(evidence)
    timing = negative["read_only_live_slot_receipts"][0]["actors"][0][
        "timing"
    ]
    timing.update({"started_ns": -2, "ended_ns": -1, "elapsed_ns": 1})
    negative["read_only_live_slot_receipts"][0]["actor_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            negative["read_only_live_slot_receipts"][0]["actors"]
        )
    )
    negative["read_only_live_slot_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            negative["read_only_live_slot_receipts"]
        )
    )
    with pytest.raises(RuntimeError, match="ordered read-only"):
        subject._verify_period_edges_v65b(negative)

    reordered = copy.deepcopy(evidence)
    write_ended = reordered["exact_master_slot_write_receipts"][0][
        "actors"
    ][0]["timing"]["ended_ns"]
    timing = reordered["read_only_live_slot_receipts"][0]["actors"][0][
        "timing"
    ]
    timing.update({
        "started_ns": write_ended - 1,
        "ended_ns": write_ended,
        "elapsed_ns": 1,
    })
    reordered["read_only_live_slot_receipts"][0][
        "actor_receipts_sha256"
    ] = analysis.v65a.canonical_sha256_v65a(
        reordered["read_only_live_slot_receipts"][0]["actors"]
    )
    reordered["read_only_live_slot_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            reordered["read_only_live_slot_receipts"]
        )
    )
    with pytest.raises(RuntimeError, match="within-period receipt timing"):
        subject._verify_period_edges_v65b(reordered)

    replayed = copy.deepcopy(evidence)
    for collection, offsets in (
        ("exact_master_slot_write_receipts", (0, 1)),
        ("read_only_live_slot_receipts", (0, 2)),
    ):
        rows = replayed[collection]
        source_index, target_index = offsets
        for rank in range(4):
            rows[target_index]["actors"][rank]["timing"] = copy.deepcopy(
                rows[source_index]["actors"][rank]["timing"]
            )
        rows[target_index]["actor_receipts_sha256"] = (
            analysis.v65a.canonical_sha256_v65a(rows[target_index]["actors"])
        )
        replayed[f"{collection}_sha256"] = (
            analysis.v65a.canonical_sha256_v65a(rows)
        )
    with pytest.raises(RuntimeError, match="global period-edge receipt timing"):
        subject._verify_period_edges_v65b(replayed)


def test_period_edges_reject_bool_index_and_lora_id_aliases():
    evidence = _expanded_period_edge_evidence()
    bool_index = copy.deepcopy(evidence)
    bool_index["exact_master_slot_write_receipts"][0]["period_index"] = False
    bool_index["exact_master_slot_write_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            bool_index["exact_master_slot_write_receipts"]
        )
    )
    with pytest.raises(RuntimeError, match="ordered exact-master"):
        subject._verify_period_edges_v65b(bool_index)

    bool_lora = copy.deepcopy(evidence)
    actor = bool_lora["read_only_live_slot_receipts"][0]["actors"][0]
    actor["active_lora_ids"] = [True]
    actor["active_manager_cache_lora_ids"] = [True]
    bool_lora["read_only_live_slot_receipts"][0]["actor_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            bool_lora["read_only_live_slot_receipts"][0]["actors"]
        )
    )
    bool_lora["read_only_live_slot_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            bool_lora["read_only_live_slot_receipts"]
        )
    )
    with pytest.raises(RuntimeError, match="ordered read-only"):
        subject._verify_period_edges_v65b(bool_lora)

    float_count = copy.deepcopy(evidence)
    float_count["read_only_live_slot_receipts"][0]["aggregate"][
        "runtime_view_count_per_actor"
    ] = 82.0
    float_count["read_only_live_slot_receipts_sha256"] = (
        analysis.v65a.canonical_sha256_v65a(
            float_count["read_only_live_slot_receipts"]
        )
    )
    with pytest.raises(RuntimeError, match="aggregate changed"):
        subject._verify_period_edges_v65b(float_count)


def test_input_and_pool_receipts_reject_bool_integer_aliases():
    receipt = copy.deepcopy(_r1_evidence()["authorized_input_receipt"])
    receipt["lora_adapter_request_name"] = "v434_ranking64_alpha_zero_v65b"
    assert subject._verify_input_receipt_v65b(receipt)["decoded_ranking_rows"] == 64
    receipt["lora_adapter_request_id"] = True
    receipt["remaining_exact_sentinel_rows_decoded"] = False
    with pytest.raises(RuntimeError, match="authorized prefix"):
        subject._verify_input_receipt_v65b(receipt)

    for name, value in (
        ("generation_seed", True),
        ("generation_max_tokens", 4.0),
    ):
        receipt = copy.deepcopy(_r1_evidence()["authorized_input_receipt"])
        receipt["lora_adapter_request_name"] = "v434_ranking64_alpha_zero_v65b"
        receipt[name] = value
        with pytest.raises(RuntimeError, match="authorized prefix"):
            subject._verify_input_receipt_v65b(receipt)

    receipt = copy.deepcopy(_r1_evidence()["authorized_input_receipt"])
    receipt["lora_adapter_request_name"] = "v434_ranking64_alpha_zero_v65b"
    receipt["request_prompt_token_ids_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="authorized prefix"):
        subject._verify_input_receipt_v65b(receipt)

    pool = {
        "schema": "v65b-bounded-reward-pool-shutdown",
        "timeout_seconds": runtime.FIXED_REWARD_POOL_SHUTDOWN_TIMEOUT_SECONDS_V65B,
        "pool_found": 1,
        "worker_process_count": 8,
        "terminate_completed_before_deadline": 1,
        "join_completed_before_deadline": 1,
        "forced_worker_kill_count": False,
        "all_workers_stopped": 1,
    }
    with pytest.raises(RuntimeError, match="reward-pool cleanup"):
        subject._verify_bounded_reward_pool_cleanup_v65b(pool)


@pytest.mark.parametrize("reason", [0, 1, 2, 4, 16, 256, 279])
def test_gpu_finalizer_accepts_every_sealed_diagnostic_reason(tmp_path, reason):
    path = tmp_path / f"allowed-{reason}.jsonl"
    rows = _gpu_rows(reason)
    _write_gpu_log(path, rows)
    report = _gpu_report(path, rows)
    assert subject._gpu_summary_v65b(
        path, subject.file_sha256_v65b(path),
        {gpu: 100 + gpu for gpu in range(4)}, report,
    )["all_eighty_generation_phases_have_a_simultaneous_positive_four_gpu_cycle"]


def test_gpu_finalizer_rejects_bool_masks_and_reappearing_phase(tmp_path):
    healthy_path = tmp_path / "healthy.jsonl"
    healthy = _gpu_rows(0)
    _write_gpu_log(healthy_path, healthy)
    report = _gpu_report(healthy_path, healthy)

    bool_mask = copy.deepcopy(healthy)
    bool_mask[0]["allowed_diagnostic_reasons_bitmask"] = False
    path = tmp_path / "bool-mask.jsonl"
    _write_gpu_log(path, bool_mask)
    with pytest.raises(RuntimeError, match="attribution or health"):
        subject._gpu_summary_v65b(
            path, subject.file_sha256_v65b(path),
            {gpu: 100 + gpu for gpu in range(4)}, report,
        )

    aliased_summary = copy.deepcopy(report)
    first_phase = "unscored_warmup_0_generation_all_actors"
    aliased_summary["gpu_period_phases_and_hardware_health"]["by_phase"][
        first_phase
    ]["0"]["samples"] = True
    with pytest.raises(RuntimeError, match="reported GPU phase"):
        subject._gpu_summary_v65b(
            healthy_path, subject.file_sha256_v65b(healthy_path),
            {gpu: 100 + gpu for gpu in range(4)}, aliased_summary,
        )

    reappearing = copy.deepcopy(healthy)
    last = datetime.fromisoformat(reappearing[-1]["sampled_at_utc"])
    for row in copy.deepcopy(healthy[:4]):
        row["sampled_at_utc"] = (last + timedelta(seconds=1)).isoformat()
        reappearing.append(row)
    path = tmp_path / "reappearing.jsonl"
    _write_gpu_log(path, reappearing)
    with pytest.raises(RuntimeError, match="phase order"):
        subject._gpu_summary_v65b(
            path, subject.file_sha256_v65b(path),
            {gpu: 100 + gpu for gpu in range(4)}, report,
        )


def test_gpu_finalizer_requires_simultaneous_cycle_and_monitor_cadence(tmp_path):
    healthy_path = tmp_path / "healthy.jsonl"
    healthy = _gpu_rows(4)
    _write_gpu_log(healthy_path, healthy)
    report = _gpu_report(healthy_path, healthy)

    first_cycle = healthy[:4]
    started = datetime.fromisoformat(first_cycle[0]["sampled_at_utc"])
    staggered = []
    for cycle_index in range(4):
        for row in copy.deepcopy(first_cycle):
            row["sampled_at_utc"] = (
                started + timedelta(seconds=0.1 * cycle_index)
            ).isoformat()
            if row["gpu"] == cycle_index:
                row["utilization_percent"] = 0
            staggered.append(row)
    staggered.extend(copy.deepcopy(healthy[4:]))
    path = tmp_path / "staggered.jsonl"
    _write_gpu_log(path, staggered)
    with pytest.raises(RuntimeError, match="simultaneous positive"):
        subject._gpu_summary_v65b(
            path, subject.file_sha256_v65b(path),
            {gpu: 100 + gpu for gpu in range(4)}, report,
        )

    sparse = copy.deepcopy(healthy)
    origin = datetime(2026, 7, 16, tzinfo=timezone.utc)
    for cycle_index in range(80):
        sampled = (origin + timedelta(seconds=3 * cycle_index)).isoformat()
        for row in sparse[4 * cycle_index:4 * cycle_index + 4]:
            row["sampled_at_utc"] = sampled
    path = tmp_path / "sparse.jsonl"
    _write_gpu_log(path, sparse)
    sparse_report = copy.deepcopy(report)
    sparse_report.update({
        "completed_at_utc": "2026-07-16T00:04:00+00:00",
        "wall_runtime_seconds": 241.0,
    })
    with pytest.raises(RuntimeError, match="sampling cadence"):
        subject._gpu_summary_v65b(
            path, subject.file_sha256_v65b(path),
            {gpu: 100 + gpu for gpu in range(4)}, sparse_report,
        )

    boundary = copy.deepcopy(healthy)
    for cycle_index in range(80):
        sampled = (origin + timedelta(seconds=2 * cycle_index)).isoformat()
        for row in boundary[4 * cycle_index:4 * cycle_index + 4]:
            row["sampled_at_utc"] = sampled
    path = tmp_path / "exact-boundary.jsonl"
    _write_gpu_log(path, boundary)
    boundary_report = _gpu_report(path, boundary)
    boundary_report.update({
        "completed_at_utc": "2026-07-16T00:02:40+00:00",
        "wall_runtime_seconds": 161.0,
    })
    summary = subject._gpu_summary_v65b(
        path, subject.file_sha256_v65b(path),
        {gpu: 100 + gpu for gpu in range(4)}, boundary_report,
    )
    assert summary["maximum_consecutive_sample_cycle_gap_seconds"] == 2.0
    assert summary["sample_cycle_gap_within_sealed_limit"] is True


def test_gpu_log_and_wall_time_forgery_are_rejected(tmp_path):
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text('{"gpu":0,"gpu":1}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicate GPU JSON key"):
        subject._parse_gpu_rows_v65b(duplicate)

    path = tmp_path / "healthy.jsonl"
    rows = _gpu_rows(4)
    _write_gpu_log(path, rows)
    report = _gpu_report(path, rows)
    report["wall_runtime_seconds"] = 1.0
    with pytest.raises(RuntimeError, match="shorter than GPU sample span"):
        subject._gpu_summary_v65b(
            path, subject.file_sha256_v65b(path),
            {gpu: 100 + gpu for gpu in range(4)}, report,
        )
    with pytest.raises(RuntimeError, match="wall clocks disagree"):
        subject._verify_report_clock_v65b({
            "started_at_utc": "2026-07-16T00:00:00+00:00",
            "completed_at_utc": "2026-07-16T00:01:40+00:00",
            "wall_runtime_seconds": 1.0,
        })
