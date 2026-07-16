#!/usr/bin/env python3
"""Focused CPU-only tests for the independent V65A finalizer."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from dataclasses import replace
from pathlib import Path

import pytest

import build_lora_es_ranking64_alpha_zero_preregistration_v65a as builder
import finalize_lora_es_ranking64_alpha_zero_v65a as subject
import lora_es_nested_population_v52 as design52
import lora_es_ranking64_alpha_zero_calibration_v65a as analysis
import run_lora_es_ranking64_alpha_zero_calibration_v65a as runtime
import run_lora_es_nested_population_v52 as runtime52
import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime64


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _self_hashed(value: dict) -> dict:
    result = copy.deepcopy(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v65a(result)
    )
    return result


def _write_source(
    path: Path, value: dict,
) -> subject.SelfHashedSourceV65A:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return subject.SelfHashedSourceV65A(
        path=path.resolve(),
        file_sha256=subject.file_sha256_v65a(path),
        content_sha256=value["content_sha256_before_self_field"],
    )


def _master_identity() -> dict:
    return copy.deepcopy(_real_installations()[0]["canonical_identity"])


@lru_cache(maxsize=1)
def _real_installations() -> list[dict]:
    path = subject.ROOT / (
        "experiments/eggroll_es_hpo/runs/"
        "v61a_v434_train_only_baseline_census/baseline_census_report_v61a.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    installations = value.get("installations")
    if not isinstance(installations, list) or len(installations) != 4:
        raise RuntimeError("test V61A installation fixture changed")
    return installations


def _installed_master() -> dict:
    identity = _master_identity()
    return {
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "canonical_master_identity_sha256": (
            analysis.canonical_sha256_v65a(identity)
        ),
        "four_actor_certificate_sha256": _sha("four-actor-certificate"),
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
    }


def _live_consensus() -> dict:
    return {
        "schema": "v65a-read-only-four-actor-master-slot-consensus",
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "canonical_master_identity_sha256": analysis.canonical_sha256_v65a(
            _master_identity()
        ),
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
        "runtime_view_count_per_actor": 82,
        "runtime_elements_per_actor": 4_921_344,
        "runtime_dtype": "torch.bfloat16",
        "base_inventory_sha256": subject.BASE_INVENTORY_SHA256_V65A,
        "four_actor_exact_read_only_consensus": True,
    }


def _state_receipts(kind: str) -> list[dict]:
    master = _live_consensus()
    return [{
        "period_kind": kind,
        "period_index": index,
        "before": copy.deepcopy(master),
        "after": copy.deepcopy(master),
        "identical_v434_state": True,
    } for index in range(4)]


def _slot_writes() -> list[dict]:
    master = _installed_master()
    result = []
    for kind in ("unscored_warmup", "scored"):
        for index in range(4):
            actors = []
            for actor in range(4):
                started = 1_000_000 + len(result) * 100 + actor * 10
                actors.append({
                    "schema": "exact-master-slot-write-v65a",
                    "period_kind": kind,
                    "period_index": index,
                    "master_identity": _master_identity(),
                    "materialization": {
                        **copy.deepcopy(
                            _real_installations()[actor]["materialization"]
                        ),
                        "phase": "v65a_exact_master_slot_write",
                    },
                    "base_identity": {
                        **copy.deepcopy(
                            _real_installations()[actor]["base_identity"]
                        ),
                        "phase": "v65a_exact_master_slot_write",
                    },
                    "transaction_state_quiescent": True,
                    "timing": {
                        "clock": "worker_monotonic_ns",
                        "started_ns": started,
                        "ended_ns": started + 7,
                        "elapsed_ns": 7,
                    },
                })
            result.append({
                "period_kind": kind,
                "period_index": index,
                "pre_write_master": copy.deepcopy(master),
                "post_write_master": copy.deepcopy(master),
                "actors": actors,
                "actor_receipts_sha256": analysis.canonical_sha256_v65a(actors),
            })
    return result


def _read_only_edges() -> list[dict]:
    aggregate = _live_consensus()
    result = []
    for kind in ("unscored_warmup", "scored"):
        for index in range(4):
            for edge in ("before_generation", "after_generation"):
                actors = []
                for actor in range(4):
                    started = 2_000_000 + len(result) * 100 + actor * 10
                    actors.append({
                        "schema": "read-only-exact-master-slot-v65a",
                        "period_kind": kind,
                        "period_index": index,
                        "edge": edge,
                        "master_identity": _master_identity(),
                        "runtime_view_count": 82,
                        "runtime_elements": 4_921_344,
                        "runtime_dtype": "torch.bfloat16",
                        "runtime_values_sha256": (
                            design52.MASTER_RUNTIME_SHA256_V52
                        ),
                        "active_lora_ids": [1],
                        "active_manager_cache_lora_ids": [1],
                        "base_identity": {
                            **copy.deepcopy(
                                _real_installations()[actor]["base_identity"]
                            ),
                            "phase": "v65a_read_only_slot_receipt",
                        },
                        "transaction_state_quiescent": True,
                        "slot_read_only_no_weight_write_or_reset": True,
                        "timing": {
                            "clock": "worker_monotonic_ns",
                            "started_ns": started,
                            "ended_ns": started + 5,
                            "elapsed_ns": 5,
                        },
                    })
                result.append({
                    "period_kind": kind,
                    "period_index": index,
                    "edge": edge,
                    "aggregate": copy.deepcopy(aggregate),
                    "actors": actors,
                    "actor_receipts_sha256": (
                        analysis.canonical_sha256_v65a(actors)
                    ),
                })
    return result


def _applied_lora_receipt() -> dict:
    return {
        "schema": "v64-effective-applied-lora-receipt",
        "expected_lora_int_id": 1,
        "active_lora_ids": [1],
        "active_manager_cache_lora_ids": [1],
        "loaded_cpu_cache_lora_ids": [1],
        "active_slot_index": 0,
        "facade_type": "LRUCacheWorkerLoRAManager",
        "manager_type": "LRUCacheLoRAModelManager",
        "staged_weights_file_sha256": design52.STAGED_WEIGHTS_SHA256_V52,
        "canonical_fp32_state_sha256": design52.MASTER_SHA256_V52,
        "canonical_ordered_key_sha256": design52.MASTER_ORDERED_KEY_SHA256_V52,
        "canonical_tensor_count": 70,
        "canonical_elements": 4_528_128,
        "registered_lora_module_count": 23,
        "matched_live_lora_module_count": 23,
        "unmatched_registered_lora_module_count": 0,
        "runtime_module_manifest_sha256": (
            subject.RUNTIME_MODULE_MANIFEST_SHA256_V65A
        ),
        "source_linked_runtime_view_count": 82,
        "source_linked_runtime_elements": 4_921_344,
        "source_linked_runtime_dtype": "torch.bfloat16",
        "source_linked_runtime_values_sha256": (
            design52.MASTER_RUNTIME_SHA256_V52
        ),
        "registered_slot_view_count": 82,
        "registered_slot_records_sha256": (
            subject.REGISTERED_SLOT_RECORDS_SHA256_V65A
        ),
        "exact_staged_fp32_to_gpu_slot_equality": True,
        "exact_registered_postpack_to_gpu_slot_equality": True,
        "active_matches_expected": True,
        "max_loras": 1,
        "max_cpu_loras": 2,
    }


def _active_lora_receipts() -> list[dict]:
    return [{
        "schema": "v65a-effective-active-lora-receipt",
        "expected_lora_int_id": 1,
        "active_lora_ids": [1],
        "active_manager_cache_lora_ids": [1],
        "loaded_cpu_cache_lora_ids": [1],
        "facade_type": "LRUCacheWorkerLoRAManager",
        "manager_type": "LRUCacheLoRAModelManager",
        "active_slot_index": 0,
        "max_loras": 1,
        "max_cpu_loras": 2,
        "max_lora_rank": 32,
        "staged_v434_applied_receipt": _applied_lora_receipt(),
        "extra_or_candidate_adapter_loaded": False,
    } for _actor in range(4)]


def _installations() -> list[dict]:
    return copy.deepcopy(_real_installations())


def _actors(prereg: dict) -> tuple[list[dict], list[dict]]:
    tuned = prereg["runtime"]["tuned_table_content_sha256"]
    actors = []
    workers = []
    for gpu in range(4):
        pid = 10_001 + gpu
        actors.append({
            "schema": "ranking64-alpha-zero-actor-identity-v65a",
            "pid": pid,
            "physical_gpu_id": gpu,
            "cuda_visible_devices": str(gpu),
            "cuda_current_device": 0,
            "runtime_determinism_controls": dict(
                analysis.ENGINE_CONTROLS_V65A
            ),
            "scheduler_class": "Scheduler",
            "tuned_folder": str(design52.RUNTIME_V52["tuned_folder"]),
            "tuned_table_content_sha256": tuned,
            "submitted_request_batch_size": 64,
            "generation_only": True,
            "global_batch_invariance_claimed": False,
        })
        workers.append({
            "schema": "lora-topology-worker-identity-v40a",
            "pid": pid,
            "cuda_visible_devices": str(gpu),
            "cuda_current_device": 0,
        })
    return actors, workers


def _scored(panel: dict) -> list:
    metric_rows = [{
        "request_index": item["request_index"],
        "row_sha256": item["row_sha256"],
        "unit_identity_sha256": item["unit_identity_sha256"],
        "f1": 0.5,
        "exact": 0,
        "nonzero": 1,
    } for item in panel["items"]]
    return [[copy.deepcopy(metric_rows) for _actor in range(4)]
            for _period in range(4)]


def _gpu_log(path: Path, pid_map: dict[int, int]) -> dict:
    rows = []
    for phase_index, phase in enumerate(subject.EXPECTED_PHASES_V65A):
        for gpu in range(4):
            rows.append({
                "sampled_at_utc": f"2026-07-16T00:00:{phase_index:02d}+00:00",
                "phase": phase,
                "gpu": gpu,
                "expected_pid": pid_map[gpu],
                "compute_pids": [pid_map[gpu]],
                "foreign_compute_pids": [],
                "utilization_percent": 50 + gpu,
                "memory_used_mib": 80_000 + gpu,
            })
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return {
        "rows": rows,
        "file_sha256": subject.file_sha256_v65a(path),
    }


def _gpu_summaries(rows: list[dict], pid_map: dict[int, int]) -> tuple[dict, dict]:
    by_gpu = {}
    for gpu in range(4):
        selected = [row for row in rows if row["gpu"] == gpu]
        by_gpu[str(gpu)] = {
            "expected_pid": pid_map[gpu],
            "samples": len(selected),
            "resident_samples": len(selected),
            "positive_samples": len(selected),
            "mean_resident_utilization_percent": float(50 + gpu),
            "peak_utilization_percent": 50 + gpu,
            "peak_memory_used_mib": 80_000 + gpu,
        }
    phases = {}
    for phase in subject.EXPECTED_PHASES_V65A:
        phases[phase] = {
            str(gpu): {
                "samples": 1,
                "resident_samples": 1,
                "positive_resident_samples": 1,
                "peak_utilization_percent": 50 + gpu,
                "peak_memory_used_mib": 80_000 + gpu,
            } for gpu in range(4)
        }
    return (
        {"all_four_attributed_positive": True, "by_gpu": by_gpu},
        {
            "schema": "v65a-per-period-four-gpu-activity",
            "generation_phases": 8,
            "all_eight_generation_phases_positive_on_all_four_gpus": True,
            "foreign_compute_process_observations": 0,
            "by_phase": phases,
        },
    )


def _cleanup() -> dict:
    node_id = f"{1:056x}"
    before = [{
        "placement_group_id": f"{index + 1:036x}",
        "strategy": "PACK",
        "state": "CREATED",
        "bundles": {"0": {"GPU": 1.0}},
        "bundles_to_node_id": {"0": node_id},
    } for index in range(4)]
    after = [{**item, "state": "REMOVED"} for item in before]
    return {
        "schema": "eggroll-es-placement-group-cleanup-v38a",
        "driver_scoped_non_detached_by_construction": True,
        "engine_kill_count": 4,
        "placement_group_remove_count": 4,
        "before": before,
        "after": after,
        "all_four_gcs_states_removed": True,
    }


def _fixture(tmp_path: Path) -> subject.FinalizerSourcesV65A:
    paths = {
        "preregistration": tmp_path / "prereg.json",
        "attempt": tmp_path / "attempt.json",
        "evidence": tmp_path / "evidence.json",
        "analysis": tmp_path / "analysis.json",
        "report": tmp_path / "report.json",
        "gpu_log": tmp_path / "gpu.jsonl",
    }
    panel, prereg = builder.build_v65a(
        ranking_panel_output=tmp_path / "panel.json",
    )
    (tmp_path / "panel.json").write_bytes(builder.json_payload_v65a(panel))
    prereg = copy.deepcopy(prereg)
    prereg["artifacts"].update({
        "run_directory": str(tmp_path),
        "attempt": str(paths["attempt"]),
        "evidence": str(paths["evidence"]),
        "analysis": str(paths["analysis"]),
        "report": str(paths["report"]),
        "gpu_log": str(paths["gpu_log"]),
        "failure": str(tmp_path / "failure_v65a.json"),
    })
    prereg = _self_hashed(prereg)
    prereg_source = _write_source(paths["preregistration"], prereg)

    expectation = runtime64.base_model_artifact_expectation_v64()
    base_receipt = runtime64.expected_base_model_artifact_receipt_v64(
        expectation
    )
    attempt = _self_hashed({
        "schema": "v65a-ranking64-alpha-zero-attempt",
        "status": "launching_exact64_calibration_only",
        "phase": "before_authorized_train_semantics_model_ray_or_gpu_load",
        "started_at_utc": "2026-07-16T00:00:00+00:00",
        "preregistration_file_sha256": prereg_source.file_sha256,
        "preregistration_content_sha256": prereg_source.content_sha256,
        "runtime_determinism_controls": dict(analysis.ENGINE_CONTROLS_V65A),
        "base_model_artifact_receipt": base_receipt,
        "preflight": {
            "compute_process_query_empty": True,
            "memory_used_mib": {gpu: 4 for gpu in range(4)},
        },
        "fixed_unscored_warmup_periods": 4,
        "fixed_scored_periods": 4,
        "fixed_paired_replicas_per_unit": 8,
        "model_loaded_or_gpu_compute_started": False,
        "candidate_hpo_update_projection_or_protected_access": False,
    })
    attempt_source = _write_source(paths["attempt"], attempt)

    actors, workers = _actors(prereg)
    active = _active_lora_receipts()
    writes = _slot_writes()
    reads = _read_only_edges()
    scored = _scored(panel)
    input_receipt = {
        "schema": "v65-exact-authorized-ranking-prefix-receipt",
        "path": str(subject.population65.V61C_ROWS),
        "source_full_file_sha256_bound_but_not_recomputed": (
            subject.population65.V61C_ROWS_FILE_SHA256
        ),
        "authorized_prefix_bytes": analysis.RANKING_PREFIX_BYTES_V65A,
        "authorized_prefix_sha256": analysis.RANKING_PREFIX_SHA256_V65A,
        "decoded_ranking_rows": 64,
        "requested_byte_offset_at_or_after_prefix": False,
        "remaining_exact_sentinel_rows_decoded": 0,
        "question_answer_or_text_persisted": False,
        "request_count": 64,
        "request_prompt_token_ids_sha256": _sha("prompt-token-ids"),
        "generation_seed": analysis.COMMON_GENERATION_SEED_V65A,
        "generation_temperature": 0.0,
        "generation_max_tokens": (
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V65A["max_tokens"]
        ),
        "submitted_request_batch_size": 64,
        "runtime_determinism_controls": dict(analysis.ENGINE_CONTROLS_V65A),
        "lora_adapter_request_name": "v434_ranking64_alpha_zero_v65a",
        "lora_adapter_request_id": 1,
        "lora_adapter_request_path": str(design52.STAGED_V52),
    }
    evidence = runtime.build_evidence_v65a(
        panel=panel,
        input_receipt=input_receipt,
        actor_identities=actors,
        worker_identities=workers,
        active_lora_receipts=active,
        installations=_installations(),
        installed_master=_installed_master(),
        warmup_state_receipts=_state_receipts("unscored_warmup"),
        scored_state_receipts=_state_receipts("scored"),
        slot_write_receipts=writes,
        read_only_slot_receipts=reads,
        final_master_state=_installed_master(),
        scored_periods=scored,
    )
    adapter_contract = runtime52.verify_adapter_contract_v52()
    evidence = _self_hashed({
        **evidence,
        "adapter_artifact_contract_prelaunch": adapter_contract,
        "adapter_artifact_contract_postcleanup": copy.deepcopy(adapter_contract),
        "adapter_artifact_contract_unchanged": True,
    })
    evidence_source = _write_source(paths["evidence"], evidence)

    stored_analysis = analysis.analyze_scored_periods_v65a(scored)
    stored_analysis = _self_hashed({
        **stored_analysis,
        "source_evidence_content_sha256": evidence_source.content_sha256,
    })
    analysis_source = _write_source(paths["analysis"], stored_analysis)

    pid_map = {item["physical_gpu_id"]: item["pid"] for item in actors}
    gpu = _gpu_log(paths["gpu_log"], pid_map)
    overall, phases = _gpu_summaries(gpu["rows"], pid_map)
    report = _self_hashed({
        "schema": "v65a-ranking64-alpha-zero-calibration-report",
        "status": "complete_gate_passed_population_still_unauthorized",
        "started_at_utc": attempt["started_at_utc"],
        "completed_at_utc": "2026-07-16T00:10:00+00:00",
        "wall_runtime_seconds": 600.0,
        "preregistration_file_sha256": prereg_source.file_sha256,
        "preregistration_content_sha256": prereg_source.content_sha256,
        "attempt": {
            "path": str(paths["attempt"]),
            "file_sha256": attempt_source.file_sha256,
            "content_sha256": attempt_source.content_sha256,
        },
        "evidence": {
            "path": str(paths["evidence"]),
            "file_sha256": evidence_source.file_sha256,
            "content_sha256": evidence_source.content_sha256,
        },
        "analysis": {
            "path": str(paths["analysis"]),
            "file_sha256": analysis_source.file_sha256,
            "content_sha256": analysis_source.content_sha256,
            "required_alpha_zero_gate": stored_analysis[
                "required_alpha_zero_gate"
            ],
        },
        "runtime_determinism_controls": dict(analysis.ENGINE_CONTROLS_V65A),
        "actor_runtime_identities": actors,
        "base_model_prelaunch_artifact_receipt": base_receipt,
        "base_model_postrun_artifact_receipt": base_receipt,
        "adapter_artifact_contract_prelaunch": adapter_contract,
        "adapter_artifact_contract_postcleanup": copy.deepcopy(adapter_contract),
        "adapter_artifact_contract_unchanged": True,
        "authorized_prefix_postrun_receipt": {
            "path": str(subject.population65.V61C_ROWS),
            "file_size_bytes_metadata_only": (
                analysis.RANKING_SOURCE_FILE_SIZE_BYTES_V65A
            ),
            "authorized_prefix_bytes": analysis.RANKING_PREFIX_BYTES_V65A,
            "authorized_prefix_sha256": analysis.RANKING_PREFIX_SHA256_V65A,
            "decoded_postrun": False,
            "requested_byte_offset_at_or_after_prefix": False,
            "full_file_read_or_hash_performed": False,
        },
        "gpu_activity": overall,
        "gpu_period_phases": phases,
        "gpu_log_file_sha256": gpu["file_sha256"],
        "cleanup": _cleanup(),
        "final_gpu_idle": {"all_four_compute_process_lists_empty": True},
        "warmup_generation_completions_discarded": 1024,
        "scored_generation_completions": 1024,
        "total_generation_completions": 2048,
        "raw_question_answer_prompt_or_generation_text_persisted": False,
        "candidate_hpo_update_projection_or_promotion_performed": False,
        "holdback_sentinel_reserve_ood_terminal_or_protected_opened": False,
        "v65_population_launch_authorized": False,
    })
    report_source = _write_source(paths["report"], report)
    return subject.FinalizerSourcesV65A(
        preregistration=prereg_source,
        attempt=attempt_source,
        evidence=evidence_source,
        analysis=analysis_source,
        report=report_source,
        gpu_log_path=paths["gpu_log"],
        gpu_log_file_sha256=gpu["file_sha256"],
    )


def _replace_json_source(
    sources: subject.FinalizerSourcesV65A, name: str, value: dict,
) -> subject.FinalizerSourcesV65A:
    source = getattr(sources, name)
    replacement = _write_source(source.path, _self_hashed(value))
    return replace(sources, **{name: replacement})


def _replace_gpu_rows(
    sources: subject.FinalizerSourcesV65A, rows: list[dict],
) -> subject.FinalizerSourcesV65A:
    sources.gpu_log_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    gpu_sha = subject.file_sha256_v65a(sources.gpu_log_path)
    report = json.loads(sources.report.path.read_text(encoding="utf-8"))
    report["gpu_log_file_sha256"] = gpu_sha
    report_source = _write_source(sources.report.path, _self_hashed(report))
    return replace(
        sources, report=report_source, gpu_log_file_sha256=gpu_sha,
    )


def test_v65a_finalizer_rebuilds_complete_numeric_transcript(tmp_path):
    result = subject.build_finalized_v65a(_fixture(tmp_path))
    assert result["schema"] == "v65a-ranking64-alpha-zero-independent-finalizer"
    assert result["content_sha256_before_self_field"] == (
        analysis.self_content_sha256_v65a(result)
    )
    schedule = result["verification"]["numeric_hash_only_evidence"][
        "fixed_complete_schedule"
    ]
    assert schedule["exact_five_key_state_receipts"] == 8
    assert schedule["exact_master_rematerialization"][
        "period_slot_write_receipts"
    ] == 8
    assert schedule["read_only_live_slot_edges"]["period_edge_receipts"] == 16
    assert schedule["read_only_live_slot_edges"]["actor_read_only_receipts"] == 64
    assert result["observed_numeric_outcome_without_authorization"][
        "required_alpha_zero_gate"
    ]["passed"] is True
    assert result["v65_population_launch_authorized"] is False
    assert result["frozen_non_authorization"][
        "adapter_update_candidate_projection_or_promotion_authorized"
    ] is False


def test_v65a_finalizer_rejects_deep_active_lora_tampering(tmp_path):
    sources = _fixture(tmp_path)
    evidence = json.loads(sources.evidence.path.read_text(encoding="utf-8"))
    evidence["active_lora_receipts"][2]["staged_v434_applied_receipt"][
        "exact_registered_postpack_to_gpu_slot_equality"
    ] = False
    sources = _replace_json_source(sources, "evidence", evidence)
    with pytest.raises(RuntimeError, match="active LoRA"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_cli_bound_file_hash_drift(tmp_path):
    sources = _fixture(tmp_path)
    with sources.evidence.path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="input file changed"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_read_only_edge_that_wrote_or_reset(tmp_path):
    sources = _fixture(tmp_path)
    evidence = json.loads(sources.evidence.path.read_text(encoding="utf-8"))
    evidence["read_only_live_slot_receipts"][9]["actors"][1][
        "slot_read_only_no_weight_write_or_reset"
    ] = False
    evidence["read_only_live_slot_receipts"][9]["actor_receipts_sha256"] = (
        analysis.canonical_sha256_v65a(
            evidence["read_only_live_slot_receipts"][9]["actors"]
        )
    )
    evidence["read_only_live_slot_receipts_sha256"] = (
        analysis.canonical_sha256_v65a(evidence["read_only_live_slot_receipts"])
    )
    sources = _replace_json_source(sources, "evidence", evidence)
    with pytest.raises(RuntimeError, match="read-only actor"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_reordered_exact_master_write(tmp_path):
    sources = _fixture(tmp_path)
    evidence = json.loads(sources.evidence.path.read_text(encoding="utf-8"))
    writes = evidence["exact_master_slot_write_receipts"]
    writes[0], writes[1] = writes[1], writes[0]
    evidence["exact_master_slot_write_receipts_sha256"] = (
        analysis.canonical_sha256_v65a(writes)
    )
    sources = _replace_json_source(sources, "evidence", evidence)
    with pytest.raises(RuntimeError, match="slot-write receipt"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_postcleanup_adapter_contract_drift(tmp_path):
    sources = _fixture(tmp_path)
    evidence = json.loads(sources.evidence.path.read_text(encoding="utf-8"))
    evidence["adapter_artifact_contract_postcleanup"]["file_sha256"][
        "staged_weights"
    ] = _sha("changed-after-cleanup")
    sources = _replace_json_source(sources, "evidence", evidence)
    with pytest.raises(RuntimeError, match="source/staged adapter"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_stored_analysis_not_exactly_rebuilt(tmp_path):
    sources = _fixture(tmp_path)
    stored = json.loads(sources.analysis.path.read_text(encoding="utf-8"))
    stored["actor_influence"]["full_four_actor_f1_point"] = 0.125
    sources = _replace_json_source(sources, "analysis", stored)
    with pytest.raises(RuntimeError, match="independent rebuild"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_foreign_gpu_pid(tmp_path):
    sources = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in sources.gpu_log_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[7]["compute_pids"].append(99_999)
    rows[7]["foreign_compute_pids"] = [99_999]
    sources = _replace_gpu_rows(sources, rows)
    with pytest.raises(RuntimeError, match="process attribution"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_gpu_timestamp_outside_report_interval(
    tmp_path,
):
    sources = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in sources.gpu_log_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["sampled_at_utc"] = "2026-07-15T23:59:59+00:00"
    sources = _replace_gpu_rows(sources, rows)
    with pytest.raises(RuntimeError, match="timestamp outside report interval"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_decreasing_gpu_timestamps(tmp_path):
    sources = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in sources.gpu_log_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[8]["sampled_at_utc"] = "2026-07-16T00:00:00.500000+00:00"
    sources = _replace_gpu_rows(sources, rows)
    with pytest.raises(RuntimeError, match="timestamps are not nondecreasing"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_forbidden_semantic_key(tmp_path):
    sources = _fixture(tmp_path)
    evidence = json.loads(sources.evidence.path.read_text(encoding="utf-8"))
    evidence["question"] = "must never be persisted"
    sources = _replace_json_source(sources, "evidence", evidence)
    with pytest.raises(RuntimeError, match="forbidden text-bearing key"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_text_hidden_under_unchecked_nested_key(tmp_path):
    sources = _fixture(tmp_path)
    evidence = json.loads(sources.evidence.path.read_text(encoding="utf-8"))
    evidence["initial_installations"][0]["base_identity"]["payload"] = (
        "RAW SECRET QUESTION AND ANSWER"
    )
    sources = _replace_json_source(sources, "evidence", evidence)
    with pytest.raises(RuntimeError, match="base-layer identity"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_reduced_implementation_closure(tmp_path):
    sources = _fixture(tmp_path)
    prereg = json.loads(
        sources.preregistration.path.read_text(encoding="utf-8")
    )
    prereg["implementation_bindings"] = {
        "entry__runtime_v65a": prereg["implementation_bindings"][
            "entry__runtime_v65a"
        ]
    }
    prereg["implementation_closure_manifest_sha256"] = (
        analysis.canonical_sha256_v65a({
            "entry__runtime_v65a": prereg["implementation_bindings"][
                "entry__runtime_v65a"
            ]["file_sha256"]
        })
    )
    sources = _replace_json_source(sources, "preregistration", prereg)
    with pytest.raises(RuntimeError, match="preregistration"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_forged_hash_only_panel(tmp_path):
    sources = _fixture(tmp_path)
    prereg = json.loads(
        sources.preregistration.path.read_text(encoding="utf-8")
    )
    panel_path = Path(prereg["ranking_panel"]["path"])
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel["items"][0]["row_sha256"] = _sha("forged-row")
    panel["request_order_sha256"] = analysis.canonical_sha256_v65a(
        [item["row_sha256"] for item in panel["items"]]
    )
    panel = _self_hashed(panel)
    panel_path.write_bytes(builder.json_payload_v65a(panel))
    prereg["ranking_panel"].update({
        "file_sha256": builder.payload_sha256_v65a(panel),
        "content_sha256": panel["content_sha256_before_self_field"],
        "request_order_sha256": panel["request_order_sha256"],
    })
    sources = _replace_json_source(sources, "preregistration", prereg)
    with pytest.raises(RuntimeError, match="preregistration|ranking panel"):
        subject.build_finalized_v65a(sources)


def test_v65a_finalizer_rejects_changed_registered_slot_hash(tmp_path):
    sources = _fixture(tmp_path)
    evidence = json.loads(sources.evidence.path.read_text(encoding="utf-8"))
    evidence["active_lora_receipts"][0]["staged_v434_applied_receipt"][
        "registered_slot_records_sha256"
    ] = _sha("different-registered-slot")
    evidence["active_lora_receipts_sha256"] = analysis.canonical_sha256_v65a(
        evidence["active_lora_receipts"]
    )
    sources = _replace_json_source(sources, "evidence", evidence)
    with pytest.raises(RuntimeError, match="active LoRA"):
        subject.build_finalized_v65a(sources)
