#!/usr/bin/env python3
"""No-CUDA-compute contract, numeric, finalizer, and tamper tests for V63."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import build_lora_es_v59_vs_v434_robust_confirmation_preregistration_v63 as builder
import finalize_lora_es_v59_vs_v434_robust_confirmation_v63 as finalizer
import lora_es_v59_vs_v434_robust_confirmation_v63 as analysis
import run_lora_es_v59_vs_v434_robust_confirmation_v63 as runtime


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _file_row(path: Path) -> dict:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "file_sha256": runtime.file_sha256_v63(path),
    }


def _toy_base_model(tmp_path: Path) -> tuple[Path, dict]:
    model = tmp_path / "toy-model"
    model.mkdir()
    shard_names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    (model / shard_names[0]).write_bytes(b"first-shard")
    (model / shard_names[1]).write_bytes(b"second-shard")
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"tensor.a": shard_names[0], "tensor.b": shard_names[1]}
    }), encoding="utf-8")
    weight_shards = [_file_row(model / name) for name in shard_names]
    non_weight_files = [
        _file_row(path) for path in sorted(model.iterdir())
        if not path.name.endswith(".safetensors")
    ]
    all_files = {
        row["name"]: {
            "bytes": row["bytes"], "sha256": row["file_sha256"]
        }
        for row in [*weight_shards, *non_weight_files]
    }
    expectation = {
        "schema": "v63-exact-base-model-artifact-expectation",
        "base_model_path": str(model),
        "committed_model_seal": {
            "path": "toy-seal", "commit": "toy",
            "file_sha256": _sha("toy-seal-file"),
            "content_sha256": _sha("toy-seal-content"),
        },
        "top_level_file_count": 4,
        "allowed_excluded_top_level_directories": [],
        "weight_shard_count": 2,
        "weight_shards": weight_shards,
        "weight_shard_manifest_sha256": analysis.canonical_sha256_v63(
            weight_shards
        ),
        "non_weight_file_count": 2,
        "non_weight_files": non_weight_files,
        "non_weight_manifest_sha256": analysis.canonical_sha256_v63(
            non_weight_files
        ),
        "runtime_metadata_file_sha256": {
            row["name"]: row["file_sha256"] for row in non_weight_files
        },
        "all_top_level_files_fingerprint_sha256": (
            analysis.canonical_sha256_v63(all_files)
        ),
    }
    return model, expectation


def _metric(f1: float, exact: int = 0) -> dict:
    return {
        "f1": float(f1),
        "exact": int(exact),
        "nonzero": int(float(f1) > 0.0),
    }


def _receipt(period_kind: str, period_index: int) -> dict:
    identities = analysis.expected_adapter_identities_v63()
    assignments = runtime._assignments_v63(period_kind, period_index)
    return {
        "period_kind": period_kind,
        "period_index": period_index,
        "before": copy.deepcopy(identities),
        "after": copy.deepcopy(identities),
        "actor_request_assignments": assignments,
        "active_adapter_receipts": [{
            "actor_rank": actor_rank,
            "schema": "v63-effective-active-lora-receipt",
            "expected_lora_int_id": assignment["lora_int_id"],
            "active_lora_ids": [assignment["lora_int_id"]],
            "loaded_cpu_cache_lora_ids": [assignment["lora_int_id"]],
            "active_matches_expected": True,
            "max_loras": 1,
            "max_cpu_loras": 2,
        } for actor_rank, assignment in enumerate(assignments)],
        "both_adapter_files_exact_and_unchanged": True,
    }


def make_evidence(
    delta: float | list[float] = 0.01,
    *,
    stable_reference_counts: tuple[int, int, int] = (48, 48, 48),
    stable_candidate_counts: tuple[int, int, int] = (48, 48, 48),
    stable_reference_nonzero_counts: tuple[int, int, int] = (48, 48, 48),
    stable_candidate_nonzero_counts: tuple[int, int, int] = (48, 48, 48),
    stress_reference_count: int = 0,
    stress_candidate_count: int = 0,
) -> dict:
    actor_delta = [float(delta)] * 4 if isinstance(delta, (int, float)) else delta
    rows = []
    for index in range(analysis.RANKING_UNITS_V63):
        rows.append({
            "row_sha256": _sha(f"row-{index}"),
            "unit_identity_sha256": _sha(f"unit-{index}"),
            "role": "ranking",
        })
    for sentinel_index, unit_sha in enumerate(analysis.SENTINEL_ORDER_V63):
        rows.append({
            "row_sha256": _sha(f"sentinel-row-{sentinel_index}"),
            "unit_identity_sha256": unit_sha,
            "role": "exact_sentinel",
        })

    label_seen = {
        (sentinel_index, label): 0
        for sentinel_index in range(analysis.EXACT_SENTINEL_UNITS_V63)
        for label in ("reference", "candidate")
    }
    scored = []
    for period_index in range(analysis.SCORED_PERIODS_V63):
        actor_batches = []
        for actor_rank in range(analysis.ACTORS_V63):
            label = analysis.LABEL_PLAN_V63[str(actor_rank)][period_index]
            batch = []
            for row_index in range(analysis.RANKING_UNITS_V63):
                base = 0.4
                value = base + (actor_delta[actor_rank]
                                if label == "candidate" else 0.0)
                batch.append(_metric(value))
            for sentinel_index in range(analysis.EXACT_SENTINEL_UNITS_V63):
                seen = label_seen[(sentinel_index, label)]
                label_seen[(sentinel_index, label)] += 1
                sentinel_sha = analysis.SENTINEL_ORDER_V63[sentinel_index]
                if sentinel_sha in analysis.STABLE_EXACT_UNIT_SHA256_V63:
                    stable_index = analysis.STABLE_EXACT_UNIT_SHA256_V63.index(
                        sentinel_sha
                    )
                    count = (
                        stable_reference_counts[stable_index]
                        if label == "reference"
                        else stable_candidate_counts[stable_index]
                    )
                    nonzero_count = (
                        stable_reference_nonzero_counts[stable_index]
                        if label == "reference"
                        else stable_candidate_nonzero_counts[stable_index]
                    )
                else:
                    count = (
                        stress_reference_count
                        if label == "reference" else stress_candidate_count
                    )
                    nonzero_count = count
                exact = int(seen < count)
                nonzero = int(seen < nonzero_count)
                if exact > nonzero:
                    raise ValueError("exact sentinel count cannot exceed nonzero count")
                batch.append(_metric(1.0 if exact else (0.5 if nonzero else 0.0), exact))
            actor_batches.append(batch)
        scored.append(actor_batches)
    assert all(value == 48 for value in label_seen.values())
    warmup_receipts = [
        _receipt("unscored_warmup", index)
        for index in range(analysis.WARMUP_PERIODS_V63)
    ]
    scored_receipts = [
        _receipt("scored", index)
        for index in range(analysis.SCORED_PERIODS_V63)
    ]
    return runtime.build_evidence_v63(
        rows, scored, warmup_receipts, scored_receipts
    )


@pytest.fixture(scope="session")
def passing_evidence() -> dict:
    return make_evidence(0.01)


@pytest.fixture(scope="session")
def passing_analysis(passing_evidence) -> dict:
    return analysis.build_analysis_v63(passing_evidence)


def _write_self_hashed(path: Path, value: dict) -> finalizer.SelfHashedSourceV63:
    value = copy.deepcopy(value)
    value.pop("content_sha256_before_self_field", None)
    content = analysis.canonical_sha256_v63(value)
    value["content_sha256_before_self_field"] = content
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return finalizer.SelfHashedSourceV63(
        path, runtime.file_sha256_v63(path), content
    )


def _actor_identities() -> list[dict]:
    return [{
        "schema": "pre-hpo-alpha-zero-actor-identity-v62a",
        "physical_gpu_id": gpu,
        "pid": 10_000 + gpu,
        "cuda_visible_devices": str(gpu),
        "cuda_current_device": 0,
        "VLLM_BATCH_INVARIANT": False,
        "async_scheduling": False,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
        "scheduler_class": "Scheduler",
        "enforce_eager": True,
        "tuned_folder": str(runtime.design_v52.RUNTIME_V52["tuned_folder"]),
        "tuned_table_content_sha256": (
            runtime.design_v52.RUNTIME_V52["tuned_table_content_sha256"]
        ),
        "submitted_request_batch_size": 68,
        "generation_only": True,
        "global_batch_invariance_claimed": False,
        "effective_lora_capacity": {
            "schema": "v63-effective-worker-lora-capacity",
            "lora_enabled": True,
            "max_loras": 1,
            "max_cpu_loras": 2,
            "max_lora_rank": 32,
        },
    } for gpu in range(4)]


def _write_gpu_log(path: Path, actors: list[dict]) -> tuple[str, dict]:
    phases = [
        *(
            f"unscored_warmup_{index}_generation_all_actors"
            for index in range(analysis.WARMUP_PERIODS_V63)
        ),
        *(
            f"scored_period_{index}_generation_all_actors"
            for index in range(analysis.SCORED_PERIODS_V63)
        ),
    ]
    pids = {item["physical_gpu_id"]: item["pid"] for item in actors}
    rows = []
    for phase_index, phase in enumerate(phases):
        sampled = f"2026-07-16T15:00:{phase_index:02d}+00:00"
        for gpu in range(4):
            rows.append({
                "sampled_at_utc": sampled,
                "phase": phase,
                "gpu": gpu,
                "expected_pid": pids[gpu],
                "compute_pids": [pids[gpu]],
                "foreign_compute_pids": [],
                "utilization_percent": 100,
                "memory_used_mib": 80_000 + gpu,
            })
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    samples = len(phases)
    summary = {
        "all_four_attributed_positive": True,
        "by_gpu": {
            str(gpu): {
                "expected_pid": pids[gpu],
                "samples": samples,
                "resident_samples": samples,
                "positive_samples": samples,
                "mean_resident_utilization_percent": 100.0,
                "peak_utilization_percent": 100,
                "peak_memory_used_mib": 80_000 + gpu,
            }
            for gpu in range(4)
        },
    }
    return runtime.file_sha256_v63(path), summary


def make_finalizer_sources(
    tmp_path: Path,
    evidence: dict,
) -> finalizer.FinalizerSourcesV63:
    prereg = builder.build_v63()
    panel_source = finalizer.SelfHashedSourceV63(
        runtime.runtime_v61c.STAGED_PANEL,
        runtime.runtime_v61c.STAGED_PANEL_FILE_SHA256,
        runtime.runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
    )
    panel = json.loads(panel_source.path.read_text(encoding="utf-8"))
    evidence = copy.deepcopy(evidence)
    for evidence_row, panel_item in zip(evidence["rows"], panel["items"]):
        for key in ("request_index", "row_sha256", "unit_identity_sha256", "role"):
            evidence_row[key] = panel_item[key]
    evidence["numeric_actor_period_manifest_sha256"] = (
        analysis.canonical_sha256_v63(evidence["rows"])
    )
    evidence.pop("content_sha256_before_self_field", None)
    evidence["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v63(evidence)
    )
    analysis.validate_evidence_v63(evidence)
    stored_analysis = analysis.build_analysis_v63(evidence)
    prereg_source = _write_self_hashed(tmp_path / "prereg.json", prereg)
    base_model_receipt = runtime.expected_base_model_artifact_receipt_v63(
        prereg["base_model_artifact_expectation"]
    )
    attempt = {
        "schema": "v63-v59-vs-v434-confirmation-attempt",
        "status": "launching_specific_train_only_confirmation",
        "phase": (
            "after_base_model_byte_audit_eligibility_adapter_and_gpu_preflight_"
            "before_staged_train_semantics_model_load_or_gpu_compute"
        ),
        "started_at_utc": "2026-07-16T15:00:00+00:00",
        "preregistration_file_sha256": prereg_source.file_sha256,
        "preregistration_content_sha256": prereg_source.content_sha256,
        "v62b_finalized": runtime.verify_v62b_eligibility_v63(),
        "adapter_artifacts": analysis.expected_adapter_identities_v63(),
        "base_model_artifact_receipt": base_model_receipt,
        "runtime_determinism_controls": analysis.RUNTIME_CONTROLS_V63,
        "fixed_unscored_warmup_periods": analysis.WARMUP_PERIODS_V63,
        "fixed_scored_periods": analysis.SCORED_PERIODS_V63,
        "fixed_scored_replicas_per_conflict_unit": analysis.REPLICAS_PER_UNIT_V63,
        "preflight": {
            "compute_process_query_empty": True,
            "memory_used_mib": {str(gpu): 4 for gpu in range(4)},
        },
        "gpu_inventory_preflight_performed": True,
        "model_loaded_or_gpu_compute_started": False,
        "update_hpo_master_checkpoint_or_protected_access": False,
    }
    attempt_source = _write_self_hashed(tmp_path / "attempt.json", attempt)
    evidence_source = _write_self_hashed(tmp_path / "evidence.json", evidence)
    analysis_source = _write_self_hashed(
        tmp_path / "analysis.json", stored_analysis
    )
    gate = stored_analysis["required_confirmation_gate"]
    actors = _actor_identities()
    gpu_log_path = tmp_path / "gpu.jsonl"
    gpu_log_sha256, gpu_summary = _write_gpu_log(gpu_log_path, actors)
    cleanup_rows = [{
        "placement_group_id": f"pg-{index}",
        "strategy": "PACK",
        "state": "CREATED",
        "bundles": [{"GPU": 1.0}],
        "bundles_to_node_id": {"0": "node-0"},
    } for index in range(4)]
    report = {
        "schema": "v63-v59-vs-v434-train-only-confirmation-report",
        "status": (
            "complete_gate_passed_without_promotion_authority"
            if gate["passed"] else "complete_gate_failed_closed"
        ),
        "started_at_utc": attempt["started_at_utc"],
        "completed_at_utc": "2026-07-16T15:10:00+00:00",
        "wall_runtime_seconds": 600.0,
        "preregistration_file_sha256": prereg_source.file_sha256,
        "preregistration_content_sha256": prereg_source.content_sha256,
        "attempt": {
            "path": str(attempt_source.path),
            "file_sha256": attempt_source.file_sha256,
            "content_sha256": attempt_source.content_sha256,
        },
        "v62b_finalized": runtime.verify_v62b_eligibility_v63(),
        "adapter_artifact_identities": analysis.expected_adapter_identities_v63(),
        "base_model_prelaunch_artifact_receipt": base_model_receipt,
        "base_model_postrun_artifact_receipt": base_model_receipt,
        "two_standard_lora_requests": {
            "reference_id": 1,
            "candidate_id": 2,
            "max_loras": 1,
            "max_cpu_loras": 2,
            "sequential_period_switching": True,
        },
        "panel_file_sha256": prereg["fixed_confirmation_recipe"][
            "staged_panel_file_sha256"
        ],
        "panel_content_sha256": prereg["fixed_confirmation_recipe"][
            "staged_panel_content_sha256"
        ],
        "panel_document_block_audit": panel["document_block_audit"],
        "actor_identities": actors,
        "warmup_state_receipts_sha256": evidence[
            "numeric_warmup_state_receipts_sha256"
        ],
        "scored_state_receipts_sha256": evidence[
            "numeric_scored_state_receipts_sha256"
        ],
        "evidence": {
            "path": str(evidence_source.path),
            "file_sha256": evidence_source.file_sha256,
            "content_sha256": evidence_source.content_sha256,
            "rows": 68,
            "actors": 4,
            "scored_periods": 24,
            "pairs_per_actor": 12,
            "replicas_per_conflict_unit": 48,
            "all_scored_periods_included_without_early_stop": True,
        },
        "analysis": {
            "path": str(analysis_source.path),
            "file_sha256": analysis_source.file_sha256,
            "content_sha256": analysis_source.content_sha256,
            "required_confirmation_gate": gate,
            "exact_sentinel_diagnostics": stored_analysis[
                "exact_sentinel_diagnostics"
            ],
        },
        "gpu_activity": gpu_summary,
        "cleanup": {
            "schema": "eggroll-es-placement-group-cleanup-v38a",
            "driver_scoped_non_detached_by_construction": True,
            "engine_kill_count": 4,
            "placement_group_remove_count": 4,
            "before": cleanup_rows,
            "after": [
                {**row, "state": "REMOVED"} for row in cleanup_rows
            ],
            "all_four_gcs_states_removed": True,
        },
        "final_gpu_idle": {"all_four_compute_process_lists_empty": True},
        "gpu_log_file_sha256": gpu_log_sha256,
        "generation_only": True,
        "teacher_forced_requests": 0,
        "adaptive_retry_drop_reorder_or_early_stop_performed": False,
        "median_consensus_or_best_of_selection_performed": False,
        "adapter_update_hpo_master_checkpoint_or_promotion_performed": False,
        "holdback_ood_shadow_or_protected_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
        "result_authorizes_update_hpo_promotion_or_protected_access": False,
    }
    report_source = _write_self_hashed(tmp_path / "report.json", report)
    return finalizer.FinalizerSourcesV63(
        preregistration=prereg_source,
        attempt=attempt_source,
        panel=panel_source,
        evidence=evidence_source,
        analysis=analysis_source,
        report=report_source,
        gpu_log_path=gpu_log_path,
        gpu_log_file_sha256=gpu_log_sha256,
    )


def _with_rehashed_report(
    sources: finalizer.FinalizerSourcesV63,
    report: dict,
) -> finalizer.FinalizerSourcesV63:
    return replace(
        sources,
        report=_write_self_hashed(sources.report.path, report),
    )


def test_schedule_is_exact_v62b_eager_design():
    assert analysis.WARMUP_PERIODS_V63 == 4
    assert analysis.SCORED_PERIODS_V63 == 24
    assert analysis.PAIRS_PER_ACTOR_V63 == 12
    assert analysis.REPLICAS_PER_UNIT_V63 == 48
    assert analysis.RUNTIME_CONTROLS_V63 == {
        "VLLM_BATCH_INVARIANT": False,
        "async_scheduling": False,
        "max_num_seqs": 68,
        "scheduling_policy": "fcfs",
        "enforce_eager": True,
        "max_loras": 1,
        "max_cpu_loras": 2,
    }


def test_exact_six_six_order_balance_per_actor():
    generation = np.zeros((64, 4, 24, 3), dtype=np.float64)
    _, receipts = analysis.paired_deltas_v63(generation)
    for actor in range(4):
        actor_receipts = [item for item in receipts if item["actor_rank"] == actor]
        assert sum(item["candidate_after_reference"] for item in actor_receipts) == 6
        assert sum(not item["candidate_after_reference"] for item in actor_receipts) == 6


def test_passing_primary_and_stable_exact_gate(passing_analysis):
    gate = passing_analysis["required_confirmation_gate"]
    assert gate["passed"] is True
    assert gate["robust_fitness"] == pytest.approx(0.01)
    assert all(gate["checks"].values())


def test_point_threshold_is_exactly_twice_frozen_null_width():
    assert analysis.MINIMUM_POINT_IMPROVEMENT_V63 == (
        2 * analysis.FROZEN_NULL_WIDTH_V63
    )


def test_point_threshold_is_inclusive():
    primary = {
        "point": analysis.MINIMUM_POINT_IMPROVEMENT_V63,
        "lcb": 0.01,
    }
    actor = {"maximum_absolute_leave_one_actor_out_shift": 0.0}
    exact = analysis.exact_diagnostics_v63(
        np.ones((4, 4, 24, 3), dtype=np.float64),
        [{"unit_identity_sha256": value} for value in analysis.SENTINEL_ORDER_V63],
    )
    assert analysis.robust_gate_v63(primary, actor, exact)["checks"][
        "point_improvement_at_least_twice_frozen_null_width_inclusive"
    ] is True


def test_robust_fitness_is_strictly_positive():
    exact = analysis.exact_diagnostics_v63(
        np.ones((4, 4, 24, 3), dtype=np.float64),
        [{"unit_identity_sha256": value} for value in analysis.SENTINEL_ORDER_V63],
    )
    gate = analysis.robust_gate_v63(
        {"point": 0.02, "lcb": 0.01},
        {"maximum_absolute_leave_one_actor_out_shift": 0.01},
        exact,
    )
    assert gate["robust_fitness"] == 0.0
    assert gate["checks"][
        "robust_fitness_lcb_minus_max_actor_loo_shift_strictly_positive"
    ] is False


def test_stable_per_unit_noninferiority_is_required():
    evidence = make_evidence(
        0.01,
        stable_reference_counts=(48, 48, 48),
        stable_candidate_counts=(48, 47, 48),
    )
    gate = analysis.build_analysis_v63(evidence)["required_confirmation_gate"]
    assert gate["passed"] is False
    assert gate["checks"][
        "stable_exact_every_unit_candidate_noninferior_inclusive"
    ] is False


def test_stable_nonzero_aggregate_noninferiority_is_required():
    evidence = make_evidence(
        0.01,
        stable_reference_counts=(0, 0, 0),
        stable_candidate_counts=(0, 0, 0),
        stable_reference_nonzero_counts=(48, 48, 48),
        stable_candidate_nonzero_counts=(48, 47, 48),
    )
    gate = analysis.build_analysis_v63(evidence)["required_confirmation_gate"]
    assert gate["checks"][
        "stable_nonzero_aggregate_candidate_noninferior_inclusive"
    ] is False
    assert gate["passed"] is False


def test_stress_sentinel_is_diagnostic_only():
    evidence = make_evidence(
        0.01, stress_reference_count=48, stress_candidate_count=0
    )
    built = analysis.build_analysis_v63(evidence)
    assert built["required_confirmation_gate"]["passed"] is True
    assert built["exact_sentinel_diagnostics"][
        "actor_unstable_stress_unit_used_in_required_gate"
    ] is False


def test_evidence_label_tamper_fails(passing_evidence):
    changed = copy.deepcopy(passing_evidence)
    changed["rows"][0]["scored_periods"][0]["actors"][0]["label"] = "candidate"
    changed["numeric_actor_period_manifest_sha256"] = analysis.canonical_sha256_v63(
        changed["rows"]
    )
    changed.pop("content_sha256_before_self_field")
    changed["content_sha256_before_self_field"] = analysis.canonical_sha256_v63(changed)
    with pytest.raises(ValueError, match="counterbalance"):
        analysis.validate_evidence_v63(changed)


def test_evidence_adapter_receipt_tamper_fails(passing_evidence):
    changed = copy.deepcopy(passing_evidence)
    changed["scored_state_receipts"][0]["before"]["candidate"][
        "canonical_fp32_state_sha256"
    ] = "0" * 64
    changed["numeric_scored_state_receipts_sha256"] = analysis.canonical_sha256_v63(
        changed["scored_state_receipts"]
    )
    changed.pop("content_sha256_before_self_field")
    changed["content_sha256_before_self_field"] = analysis.canonical_sha256_v63(changed)
    with pytest.raises(ValueError, match="adapter identity"):
        analysis.validate_evidence_v63(changed)


def test_evidence_self_hash_tamper_fails(passing_evidence):
    changed = copy.deepcopy(passing_evidence)
    changed["row_count"] = 67
    with pytest.raises(ValueError, match="contract"):
        analysis.validate_evidence_v63(changed)


def test_v62b_eligibility_hashes_and_non_authority_are_exact():
    value = runtime.verify_v62b_eligibility_v63()
    assert value["file_sha256"] == (
        "92b7e847ef42b06735d29d2a3f345a8c1cc8233c8408395de8d7016e9838ae72"
    )
    assert value["content_sha256"] == (
        "f05506bfcca63bf2723b10518708897ab871b1073f654ed8a221608aaacd2149"
    )
    assert value["launch_or_update_authority"] is False


def test_v59_and_v434_adapter_identities_are_exact():
    assert runtime.verify_adapter_artifacts_v63() == (
        analysis.expected_adapter_identities_v63()
    )


def test_static_two_request_support_binds_ids_one_and_two():
    support = runtime.installed_two_adapter_support_v63()
    assert support["status"] == "static_api_supported_live_switch_not_preclaimed"
    assert [item["lora_int_id"] for item in support["request_projection"]] == [1, 2]
    assert support["live_gpu_switch_probe_performed"] is False


def test_v63_worker_receipt_attests_effective_lora_capacity():
    import eggroll_es_worker_lora_v63 as worker_v63

    worker = object.__new__(worker_v63.LoRAAdapterStateWorkerExtensionV63)
    worker.lora_config = SimpleNamespace(
        max_loras=1, max_cpu_loras=2, max_lora_rank=32
    )
    assert worker.runtime_lora_capacity_v63() == {
        "schema": "v63-effective-worker-lora-capacity",
        "lora_enabled": True,
        "max_loras": 1,
        "max_cpu_loras": 2,
        "max_lora_rank": 32,
    }
    adapter_manager = SimpleNamespace(lora_index_to_id=[2])
    lora_manager = SimpleNamespace(
        _adapter_manager=adapter_manager,
        list_adapters=lambda: {1, 2},
    )
    worker.model_runner = SimpleNamespace(lora_manager=lora_manager)
    assert worker.runtime_active_lora_v63(2) == {
        "schema": "v63-effective-active-lora-receipt",
        "expected_lora_int_id": 2,
        "active_lora_ids": [2],
        "loaded_cpu_cache_lora_ids": [1, 2],
        "active_matches_expected": True,
        "max_loras": 1,
        "max_cpu_loras": 2,
    }
    with pytest.raises(RuntimeError, match="effective active LoRA identity"):
        worker.runtime_active_lora_v63(1)
    worker.lora_config.max_cpu_loras = 1
    with pytest.raises(RuntimeError, match="effective worker LoRA capacity"):
        worker.runtime_lora_capacity_v63()


def test_full_worker_execution_and_lora_manager_chain_is_hash_bound():
    import eggroll_es_worker_lora_v63 as worker_v63

    bindings = runtime.implementation_bindings_v63()["code_file_sha256"]
    assert set(runtime.WORKER_EXECUTION_PATHS_V63).issubset(bindings)
    for name, path in runtime.WORKER_EXECUTION_PATHS_V63.items():
        assert path.is_file(), name
        assert bindings[name] == runtime.file_sha256_v63(path)
    assert runtime.fixed_recipe_v63()["worker_execution_binding_keys"] == sorted(
        runtime.WORKER_EXECUTION_PATHS_V63
    )
    assert [item.__module__ for item in worker_v63.LoRAAdapterStateWorkerExtensionV63.__mro__] == [
        "eggroll_es_worker_lora_v63",
        "eggroll_es_worker_lora_v52",
        "eggroll_es_worker_lora_v51",
        "eggroll_es_worker_lora_v43i",
        "eggroll_es_worker_lora_v41a",
        "eggroll_es_worker_v3",
        "eggroll_es_worker_v2",
        "es_at_scale.utils.worker_extension",
        "builtins",
    ]
    closure = runtime.live_local_import_closure_v63()
    assert {
        "run_lora_es_generation_boundary_v48b.py",
        "run_lora_es_multi_anchor_v43i.py",
        "run_lora_es_equal_unit_v43a.py",
        "run_lora_topology_probe_v40a.py",
        "run_eggroll_es_equal_unit_v38a.py",
        "train_eggroll_es_specialist.py",
        "lora_es_fused_anchor_runtime_v43i.py",
        "lora_es_pre_hpo_alpha_zero_calibration_v62a.py",
        "build_lora_es_paired_null_inputs_v61c.py",
        "qa_quality.py",
        "lora_es_paired_null_calibration_v61c.py",
        "lora_es_baseline_census_strata_v61a.py",
        "eggroll_es_multi_anchor_v43h.py",
        "es-at-scale/es_at_scale/trainer/es_trainer.py",
        "es-at-scale/es_at_scale/utils/worker_extension.py",
    }.issubset(closure)
    assert runtime.fixed_recipe_v63()[
        "live_local_import_closure_manifest_sha256"
    ] == analysis.canonical_sha256_v63(closure)


def test_engine_kwargs_are_eager_bi_false_and_cpu_cache_two():
    holder = type("V40", (), {"MODEL": runtime.BASE_MODEL})()
    kwargs = runtime.engine_kwargs_v63(holder)
    assert kwargs["enforce_eager"] is True
    assert kwargs["async_scheduling"] is False
    assert kwargs["max_num_seqs"] == 68
    assert kwargs["max_loras"] == 1
    assert kwargs["max_cpu_loras"] == 2


def test_builder_initializes_no_cuda_compute_and_claims_no_support_authority():
    value = builder.build_v63()
    assert value["builder_or_dry_run_performed_cuda_compute_launch"] is False
    assert value["eligibility_or_static_support_alone_authorizes_launch"] is False
    assert value["access_contract"][
        "builder_or_dry_run_reads_staged_rows_or_panel"
    ] is False
    assert value[
        "update_hpo_candidate_promotion_or_protected_access_authorized"
    ] is False


def test_base_model_expectation_binds_all_shards_and_all_metadata():
    expectation = runtime.base_model_artifact_expectation_v63()
    assert expectation["top_level_file_count"] == 40
    assert expectation["weight_shard_count"] == 26
    assert expectation["non_weight_file_count"] == 14
    assert expectation["weight_shard_manifest_sha256"] == (
        runtime.BASE_MODEL_SHARDS_CONTENT_SHA256_V63
    )
    assert expectation["non_weight_manifest_sha256"] == (
        runtime.BASE_MODEL_NON_WEIGHT_MANIFEST_SHA256_V63
    )
    receipt = runtime.expected_base_model_artifact_receipt_v63(expectation)
    assert runtime.validate_base_model_artifact_receipt_v63(
        receipt, expectation
    ) == receipt


def test_base_model_receipt_rejects_rehashed_shard_or_metadata_drift():
    expectation = runtime.base_model_artifact_expectation_v63()
    for section in ("weight_shards", "non_weight_files"):
        receipt = runtime.expected_base_model_artifact_receipt_v63(expectation)
        receipt[section][0]["file_sha256"] = "0" * 64
        with pytest.raises(RuntimeError, match="base-model artifact receipt"):
            runtime.validate_base_model_artifact_receipt_v63(
                receipt, expectation
            )


@pytest.mark.parametrize("target", ("shard", "metadata"))
def test_base_model_live_verifier_rejects_file_byte_drift(
    tmp_path, monkeypatch, target
):
    model, expectation = _toy_base_model(tmp_path)
    monkeypatch.setattr(runtime, "BASE_MODEL", model)
    monkeypatch.setattr(
        runtime, "base_model_artifact_expectation_v63", lambda: expectation
    )
    path = (
        model / "model-00001-of-00002.safetensors"
        if target == "shard" else model / "config.json"
    )
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    with pytest.raises(RuntimeError, match="base-model artifact receipt"):
        runtime.verify_base_model_artifacts_v63(expectation)


def test_base_model_live_verifier_rejects_extra_entry(tmp_path, monkeypatch):
    model, expectation = _toy_base_model(tmp_path)
    monkeypatch.setattr(runtime, "BASE_MODEL", model)
    monkeypatch.setattr(
        runtime, "base_model_artifact_expectation_v63", lambda: expectation
    )
    (model / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    with pytest.raises(RuntimeError, match="top-level inventory"):
        runtime.verify_base_model_artifacts_v63(expectation)


def test_base_model_live_verifier_rejects_symlink(tmp_path, monkeypatch):
    model, expectation = _toy_base_model(tmp_path)
    monkeypatch.setattr(runtime, "BASE_MODEL", model)
    monkeypatch.setattr(
        runtime, "base_model_artifact_expectation_v63", lambda: expectation
    )
    target = tmp_path / "outside-config.json"
    target.write_text("{}\n", encoding="utf-8")
    (model / "config.json").unlink()
    (model / "config.json").symlink_to(target)
    with pytest.raises(RuntimeError, match="top-level symlink"):
        runtime.verify_base_model_artifacts_v63(expectation)


def test_base_model_live_verifier_rejects_index_traversal(
    tmp_path, monkeypatch
):
    model, expectation = _toy_base_model(tmp_path)
    monkeypatch.setattr(runtime, "BASE_MODEL", model)
    monkeypatch.setattr(
        runtime, "base_model_artifact_expectation_v63", lambda: expectation
    )
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"tensor.a": "../foreign.safetensors"}
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="index shard references"):
        runtime.verify_base_model_artifacts_v63(expectation)


def test_base_model_live_verifier_rejects_duplicate_index_key(
    tmp_path, monkeypatch
):
    model, expectation = _toy_base_model(tmp_path)
    monkeypatch.setattr(runtime, "BASE_MODEL", model)
    monkeypatch.setattr(
        runtime, "base_model_artifact_expectation_v63", lambda: expectation
    )
    shard = "model-00001-of-00002.safetensors"
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"tensor.a":"' + shard
        + '","tensor.a":"' + shard + '"}}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        runtime.verify_base_model_artifacts_v63(expectation)


def test_builder_and_dry_run_never_open_or_enumerate_model_directory(
    tmp_path, monkeypatch, capsys
):
    opened = []
    original_open = Path.open
    original_iterdir = Path.iterdir

    def guarded_open(path, *args, **kwargs):
        resolved = path.resolve()
        if runtime.BASE_MODEL in resolved.parents or resolved == runtime.BASE_MODEL:
            raise AssertionError(f"dry path opened base model bytes: {resolved}")
        opened.append(resolved)
        return original_open(path, *args, **kwargs)

    def guarded_iterdir(path):
        resolved = path.resolve()
        if resolved == runtime.BASE_MODEL:
            raise AssertionError("dry path enumerated base model directory")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    prereg = builder.build_v63()
    source = _write_self_hashed(tmp_path / "prereg.json", prereg)
    assert runtime.main([
        "--preregistration", str(source.path),
        "--preregistration-sha256", source.file_sha256,
        "--preregistration-content-sha256", source.content_sha256,
        "--dry-run",
    ]) == 0
    assert runtime.BASE_MODEL_SEAL_V63 in opened
    payload = json.loads(capsys.readouterr().out)
    assert payload["base_model_directory_bytes_read_or_hashed"] is False


def test_dry_run_creates_no_run_artifact_and_opens_no_semantic_rows_or_model(
    tmp_path, capsys
):
    prereg = builder.build_v63()
    prereg_source = _write_self_hashed(tmp_path / "prereg.json", prereg)
    before = set(tmp_path.iterdir())
    assert runtime.main([
        "--preregistration", str(prereg_source.path),
        "--preregistration-sha256", prereg_source.file_sha256,
        "--preregistration-content-sha256", prereg_source.content_sha256,
        "--dry-run",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["staged_train_rows_or_panel_opened"] == 0
    assert payload["base_model_loaded_or_cuda_compute_initialized"] is False
    assert payload["run_artifact_writes"] is False
    assert payload["process_wide_zero_filesystem_writes_claimed"] is False
    assert set(tmp_path.iterdir()) == before


def test_live_path_requires_explicit_execute(tmp_path):
    prereg_source = _write_self_hashed(
        tmp_path / "prereg.json", builder.build_v63()
    )
    with pytest.raises(RuntimeError, match="requires --execute"):
        runtime.main([
            "--preregistration", str(prereg_source.path),
            "--preregistration-sha256", prereg_source.file_sha256,
            "--preregistration-content-sha256", prereg_source.content_sha256,
        ])


def test_finalizer_production_path_requires_all_postrun_hash_arguments():
    hashes = ["0" * 64 for _ in range(11)]
    hashes[2] = "not-a-sha256"
    with pytest.raises(RuntimeError, match="production outcome hash is not sealed"):
        finalizer.production_sources_v63(*hashes)


def test_finalizer_remains_byte_identical_to_preregistered_implementation():
    prereg = builder.build_v63()
    assert prereg["implementation_bindings"]["code_file_sha256"][
        "finalizer_v63"
    ] == runtime.file_sha256_v63(Path(finalizer.__file__).resolve())
    assert "PREREGISTERED_FINALIZER_FILE_SHA256" not in Path(
        finalizer.__file__
    ).read_text(encoding="utf-8")


def test_finalizer_accepts_passing_outcome_without_authority(
    tmp_path, passing_evidence
):
    result = finalizer.finalize_v63(
        make_finalizer_sources(tmp_path, passing_evidence)
    )
    assert result["status"] == "complete_gate_passed_without_authority"
    assert result["observed_numeric_outcome_without_authorization"][
        "required_confirmation_gate"
    ]["passed"] is True
    assert result["frozen_non_authorization"][
        "adapter_update_hpo_candidate_promotion_authorized"
    ] is False


def test_finalizer_accepts_failed_outcome_without_relaxation(tmp_path):
    evidence = make_evidence(0.001)
    result = finalizer.finalize_v63(make_finalizer_sources(tmp_path, evidence))
    assert result["status"] == "complete_gate_failed_closed"
    assert result["observed_numeric_outcome_without_authorization"][
        "required_confirmation_gate"
    ]["passed"] is False
    assert result["frozen_non_authorization"][
        "failed_gate_reinterpreted_or_relaxed"
    ] is False


def test_finalizer_rejects_file_tamper(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    sources.evidence.path.write_text(
        sources.evidence.path.read_text() + "\n"
    )
    with pytest.raises(RuntimeError, match="input file changed"):
        finalizer.finalize_v63(sources)


def test_finalizer_rejects_rehashed_analysis_drift(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    changed = json.loads(sources.analysis.path.read_text())
    changed["primary_generated_f1"]["point"] += 0.1
    changed_source = _write_self_hashed(sources.analysis.path, changed)
    report = json.loads(sources.report.path.read_text())
    report["analysis"]["file_sha256"] = changed_source.file_sha256
    report["analysis"]["content_sha256"] = changed_source.content_sha256
    report_source = _write_self_hashed(sources.report.path, report)
    changed_sources = finalizer.FinalizerSourcesV63(
        preregistration=sources.preregistration,
        attempt=sources.attempt,
        panel=sources.panel,
        evidence=sources.evidence,
        analysis=changed_source,
        report=report_source,
        gpu_log_path=sources.gpu_log_path,
        gpu_log_file_sha256=sources.gpu_log_file_sha256,
    )
    with pytest.raises(RuntimeError, match="differs from exact numeric rebuild"):
        finalizer.finalize_v63(changed_sources)


def test_finalizer_rejects_rehashed_authority_tamper(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["result_authorizes_update_hpo_promotion_or_protected_access"] = True
    report_source = _write_self_hashed(sources.report.path, report)
    changed_sources = finalizer.FinalizerSourcesV63(
        preregistration=sources.preregistration,
        attempt=sources.attempt,
        panel=sources.panel,
        evidence=sources.evidence,
        analysis=sources.analysis,
        report=report_source,
        gpu_log_path=sources.gpu_log_path,
        gpu_log_file_sha256=sources.gpu_log_file_sha256,
    )
    with pytest.raises(RuntimeError, match="report or integrity"):
        finalizer.finalize_v63(changed_sources)


def test_finalizer_rejects_rehashed_text_leakage(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["prompt"] = "forbidden"
    report_source = _write_self_hashed(sources.report.path, report)
    changed_sources = finalizer.FinalizerSourcesV63(
        preregistration=sources.preregistration,
        attempt=sources.attempt,
        panel=sources.panel,
        evidence=sources.evidence,
        analysis=sources.analysis,
        report=report_source,
        gpu_log_path=sources.gpu_log_path,
        gpu_log_file_sha256=sources.gpu_log_file_sha256,
    )
    with pytest.raises(RuntimeError, match="forbidden text-bearing key"):
        finalizer.finalize_v63(changed_sources)


def test_finalizer_rejects_rehashed_prereg_gate_tamper(tmp_path, passing_evidence):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    prereg = json.loads(sources.preregistration.path.read_text())
    prereg["required_confirmation_gates"][
        "stable_nonzero_aggregate_candidate_at_least_reference_inclusive"
    ] = False
    prereg_source = _write_self_hashed(sources.preregistration.path, prereg)
    changed_sources = finalizer.FinalizerSourcesV63(
        preregistration=prereg_source,
        attempt=sources.attempt,
        panel=sources.panel,
        evidence=sources.evidence,
        analysis=sources.analysis,
        report=sources.report,
        gpu_log_path=sources.gpu_log_path,
        gpu_log_file_sha256=sources.gpu_log_file_sha256,
    )
    with pytest.raises(RuntimeError, match="preregistration changed"):
        finalizer.finalize_v63(changed_sources)


def test_finalizer_rejects_rehashed_nonexclusive_attempt_preflight(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    attempt = json.loads(sources.attempt.path.read_text())
    attempt["preflight"]["compute_process_query_empty"] = False
    attempt_source = _write_self_hashed(sources.attempt.path, attempt)
    report = json.loads(sources.report.path.read_text())
    report["attempt"]["file_sha256"] = attempt_source.file_sha256
    report["attempt"]["content_sha256"] = attempt_source.content_sha256
    changed = replace(sources, attempt=attempt_source)
    changed = _with_rehashed_report(changed, report)
    with pytest.raises(RuntimeError, match="exclusive-idle launch attempt"):
        finalizer.finalize_v63(changed)


def test_finalizer_rejects_rehashed_attempt_base_model_receipt(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    attempt = json.loads(sources.attempt.path.read_text())
    attempt["base_model_artifact_receipt"]["weight_shards"][0][
        "file_sha256"
    ] = "0" * 64
    attempt_source = _write_self_hashed(sources.attempt.path, attempt)
    report = json.loads(sources.report.path.read_text())
    report["attempt"]["file_sha256"] = attempt_source.file_sha256
    report["attempt"]["content_sha256"] = attempt_source.content_sha256
    changed = replace(sources, attempt=attempt_source)
    changed = _with_rehashed_report(changed, report)
    with pytest.raises(RuntimeError, match="base-model artifact receipt"):
        finalizer.finalize_v63(changed)


def test_finalizer_rejects_rehashed_postrun_base_model_drift(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["base_model_postrun_artifact_receipt"][
        "all_26_weight_shard_bytes_and_sha256_verified"
    ] = False
    with pytest.raises(RuntimeError, match="base-model artifact receipt"):
        finalizer.finalize_v63(_with_rehashed_report(sources, report))


@pytest.mark.parametrize(
    ("field", "value"),
    (("max_num_seqs", 1), ("enforce_eager", False)),
)
def test_finalizer_rejects_rehashed_actor_runtime_drift(
    tmp_path, passing_evidence, field, value
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["actor_identities"][0][field] = value
    with pytest.raises(RuntimeError, match="actor runtime identity"):
        finalizer.finalize_v63(_with_rehashed_report(sources, report))


def test_finalizer_rejects_rehashed_effective_lora_capacity_drift(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["actor_identities"][2]["effective_lora_capacity"][
        "max_cpu_loras"
    ] = 1
    with pytest.raises(RuntimeError, match="actor runtime identity"):
        finalizer.finalize_v63(_with_rehashed_report(sources, report))


def test_evidence_rejects_requested_vs_effective_active_lora_mismatch(
    passing_evidence,
):
    changed = copy.deepcopy(passing_evidence)
    receipt = changed["scored_state_receipts"][0][
        "active_adapter_receipts"
    ][0]
    receipt["active_lora_ids"] = [2 if receipt["expected_lora_int_id"] == 1 else 1]
    changed["numeric_scored_state_receipts_sha256"] = (
        analysis.canonical_sha256_v63(changed["scored_state_receipts"])
    )
    changed.pop("content_sha256_before_self_field", None)
    changed["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v63(changed)
    )
    with pytest.raises(ValueError, match="effective active adapter"):
        analysis.validate_evidence_v63(changed)


def test_finalizer_rejects_rehashed_lora_request_id_or_capacity_claim(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["two_standard_lora_requests"]["candidate_id"] = 1
    report["two_standard_lora_requests"]["max_cpu_loras"] = 1
    with pytest.raises(RuntimeError, match="report or integrity"):
        finalizer.finalize_v63(_with_rehashed_report(sources, report))


def test_finalizer_rejects_rehashed_foreign_gpu_process_log(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    rows = sources.gpu_log_path.read_text(encoding="utf-8").splitlines()
    first = json.loads(rows[0])
    first["compute_pids"].append(999_999)
    first["foreign_compute_pids"] = [999_999]
    rows[0] = json.dumps(first, sort_keys=True)
    sources.gpu_log_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    gpu_sha = runtime.file_sha256_v63(sources.gpu_log_path)
    report = json.loads(sources.report.path.read_text())
    report["gpu_log_file_sha256"] = gpu_sha
    changed = replace(sources, gpu_log_file_sha256=gpu_sha)
    with pytest.raises(RuntimeError, match="foreign, inactive, or mismatched"):
        finalizer.finalize_v63(_with_rehashed_report(changed, report))


def test_finalizer_rejects_gpu_without_positive_sample_in_generation_phase(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    changed_rows = []
    target = "scored_period_7_generation_all_actors"
    for line in sources.gpu_log_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["gpu"] == 0 and row["phase"] == target:
            row["utilization_percent"] = 0
        changed_rows.append(json.dumps(row, sort_keys=True))
    sources.gpu_log_path.write_text(
        "\n".join(changed_rows) + "\n", encoding="utf-8"
    )
    gpu_sha = runtime.file_sha256_v63(sources.gpu_log_path)
    report = json.loads(sources.report.path.read_text())
    report["gpu_log_file_sha256"] = gpu_sha
    changed = replace(sources, gpu_log_file_sha256=gpu_sha)
    with pytest.raises(RuntimeError, match="foreign, inactive, or mismatched"):
        finalizer.finalize_v63(_with_rehashed_report(changed, report))


def test_finalizer_rejects_rehashed_gpu_summary_not_matching_numeric_log(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["gpu_activity"]["all_four_attributed_positive"] = False
    with pytest.raises(RuntimeError, match="summary differs from numeric log"):
        finalizer.finalize_v63(_with_rehashed_report(sources, report))


@pytest.mark.parametrize(
    ("mutation", "value"),
    (("engine_kill_count", 0), ("placement_group_remove_count", 0)),
)
def test_finalizer_rejects_rehashed_cleanup_drift(
    tmp_path, passing_evidence, mutation, value
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["cleanup"][mutation] = value
    with pytest.raises(RuntimeError, match="cleanup changed"):
        finalizer.finalize_v63(_with_rehashed_report(sources, report))


def test_finalizer_rejects_rehashed_nonidle_final_gpu_state(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    report["final_gpu_idle"]["all_four_compute_process_lists_empty"] = False
    with pytest.raises(RuntimeError, match="cleanup changed"):
        finalizer.finalize_v63(_with_rehashed_report(sources, report))


@pytest.mark.parametrize(
    "mutation",
    (
        "preregistration_link",
        "attempt_link",
        "panel_hash",
        "warmup_receipt",
        "evidence_coverage",
        "sentinel_link",
    ),
)
def test_finalizer_rejects_rehashed_report_chain_or_schedule_drift(
    tmp_path, passing_evidence, mutation
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    report = json.loads(sources.report.path.read_text())
    if mutation == "preregistration_link":
        report["preregistration_file_sha256"] = "0" * 64
    elif mutation == "attempt_link":
        report["attempt"]["content_sha256"] = "0" * 64
    elif mutation == "panel_hash":
        report["panel_content_sha256"] = "0" * 64
    elif mutation == "warmup_receipt":
        report["warmup_state_receipts_sha256"] = "0" * 64
    elif mutation == "evidence_coverage":
        report["evidence"]["rows"] = 67
    elif mutation == "sentinel_link":
        report["analysis"]["exact_sentinel_diagnostics"] = {}
    with pytest.raises(RuntimeError, match="report or integrity"):
        finalizer.finalize_v63(_with_rehashed_report(sources, report))


def test_finalizer_rejects_panel_document_audit_or_evidence_projection_drift(
    tmp_path, passing_evidence
):
    sources = make_finalizer_sources(tmp_path, passing_evidence)
    panel = json.loads(sources.panel.path.read_text(encoding="utf-8"))
    evidence = json.loads(sources.evidence.path.read_text(encoding="utf-8"))
    report = json.loads(sources.report.path.read_text(encoding="utf-8"))
    report["panel_document_block_audit"] = {}
    with pytest.raises(RuntimeError, match="numeric panel projection"):
        finalizer._verify_panel_projection_v63(panel, evidence, report)
    report["panel_document_block_audit"] = panel["document_block_audit"]
    evidence["rows"][0]["row_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="numeric panel projection"):
        finalizer._verify_panel_projection_v63(panel, evidence, report)


def test_analysis_never_authorizes_or_selects(passing_analysis):
    assert passing_analysis[
        "update_hpo_candidate_promotion_or_protected_access_authorized"
    ] is False
    assert passing_analysis[
        "median_consensus_or_best_of_selection_performed"
    ] is False
    assert passing_analysis["required_confirmation_gate"][
        "authorizes_update_hpo_promotion_or_protected_access"
    ] is False
