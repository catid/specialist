#!/usr/bin/env python3
"""Independently finalize the V65B high-rep alpha-zero calibration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import build_lora_es_ranking64_alpha_zero_preregistration_v65b as builder
import finalize_lora_es_ranking64_alpha_zero_v65a as basefinal
import lora_es_nested_population_v52 as design52
import lora_es_ranking64_alpha_zero_calibration_v65b as analysis
import lora_es_robust_sampling_population_v65 as population65
import run_lora_es_nested_population_v52 as runtime52
import run_lora_es_ranking64_alpha_zero_calibration_v65b as runtime
import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime64


ROOT = Path(__file__).resolve().parent
OUTPUT = Path(builder.artifacts_v65b()["finalized"])
TESTS = ROOT / "test_finalize_lora_es_ranking64_alpha_zero_v65b.py"


@dataclass(frozen=True)
class SelfHashedSourceV65B:
    path: Path
    file_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class FinalizerSourcesV65B:
    preregistration: SelfHashedSourceV65B
    attempt: SelfHashedSourceV65B
    evidence: SelfHashedSourceV65B
    analysis: SelfHashedSourceV65B
    report: SelfHashedSourceV65B
    gpu_log_path: Path
    gpu_log_file_sha256: str


def file_sha256_v65b(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256_v65b(value: object, name: str) -> str:
    if (
        not isinstance(value, str) or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v65b invalid or unsealed SHA-256: {name}")
    return value


def _require_aware_timestamp_v65b(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"v65b invalid timestamp: {name}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError(f"v65b invalid timestamp: {name}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"v65b timezone-naive timestamp: {name}")
    return value


def _json_without_duplicate_keys_v65b(path: Path) -> object:
    return _json_bytes_without_duplicate_keys_v65b(
        Path(path).read_bytes(), Path(path),
    )


def _json_bytes_without_duplicate_keys_v65b(
    payload: bytes, source_name: object,
) -> object:
    def reject(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(
                    f"v65b duplicate JSON key in {source_name}: {key}"
                )
            value[key] = item
        return value

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"v65b non-UTF-8 JSON: {source_name}") from error
    return json.loads(text, object_pairs_hook=reject)


def _read_self_hashed_v65b(source: SelfHashedSourceV65B) -> dict:
    _require_sha256_v65b(source.file_sha256, "source file")
    _require_sha256_v65b(source.content_sha256, "source content")
    payload = Path(source.path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != source.file_sha256:
        raise RuntimeError(f"v65b finalizer input file changed: {source.path}")
    value = _json_bytes_without_duplicate_keys_v65b(payload, source.path)
    if (
        not isinstance(value, dict)
        or value.get("content_sha256_before_self_field")
        != source.content_sha256
        or population65.self_content_sha256_v65(value) != source.content_sha256
    ):
        raise RuntimeError(f"v65b finalizer input content changed: {source.path}")
    return value


def _verify_no_text_keys_v65b(name: str, value: object) -> dict:
    # Exact key matching intentionally permits metadata such as generation_seed
    # while rejecting any persisted question, answer, prompt, prediction, or text.
    found = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in analysis.v65a.FORBIDDEN_TEXT_KEYS_V65A:
                    found.append(f"{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, name)
    if found:
        raise RuntimeError(f"v65b forbidden text-bearing key: {found[0]}")
    return {"source": name, "forbidden_text_key_count": 0}


def _verify_source_argument_paths_v65b(
    sources: FinalizerSourcesV65B, artifacts: dict,
) -> None:
    bound_sources = {
        "attempt": sources.attempt.path,
        "evidence": sources.evidence.path,
        "analysis": sources.analysis.path,
        "report": sources.report.path,
        "gpu_log": sources.gpu_log_path,
    }
    if (
        any(
            path.resolve() != Path(artifacts.get(name, "")).resolve()
            for name, path in bound_sources.items()
        )
        or sources.preregistration.path.resolve()
        != builder.PREREGISTRATION_OUTPUT.resolve()
    ):
        raise RuntimeError("v65b source arguments differ from fixed artifact paths")


def _verify_artifact_paths_v65b(
    prereg: dict, sources: FinalizerSourcesV65B,
) -> dict:
    artifacts = prereg.get("artifacts", {}) \
        if isinstance(prereg, dict) else {}
    if not runtime._canonical_equal_v65b(artifacts, builder.artifacts_v65b()):
        raise RuntimeError("v65b preregistered artifact paths changed")
    _verify_source_argument_paths_v65b(sources, artifacts)
    if (
        Path(artifacts.get("failure", "")).exists()
        or Path(artifacts.get("finalized", "")).exists()
    ):
        raise RuntimeError(
            "v65b source/output paths or success-artifact exclusivity changed"
        )
    return {
        "all_five_sources_equal_preregistered_artifact_paths": True,
        "preregistered_failure_artifact_absent": True,
        "preregistered_finalized_output_absent_before_exclusive_write": True,
    }


def _verify_preregistration_v65b(
    prereg: dict, sources: FinalizerSourcesV65B,
) -> dict:
    _panel, expected = builder.build_v65b()
    if (
        not runtime._canonical_equal_v65b(prereg, expected)
        or sources.preregistration.path.resolve()
        != builder.PREREGISTRATION_OUTPUT.resolve()
        or prereg.get("content_sha256_before_self_field")
        != sources.preregistration.content_sha256
        or not runtime._canonical_equal_v65b(
            prereg.get("artifacts"), builder.artifacts_v65b(),
        )
        or prereg.get("success_directly_authorizes_v65_population") is not False
        or prereg.get("authorization", {}).get("v65_population") is not False
    ):
        raise RuntimeError("v65b static preregistration or closure changed")
    artifact_paths = _verify_artifact_paths_v65b(prereg, sources)
    runtime_contract = runtime.validate_preregistered_runtime_contract_v65b(
        prereg,
    )
    planning = prereg.get("r1_prospective_planning", {})
    if (
        not runtime._canonical_equal_v65b(
            planning, builder.r1_planning_binding_v65b(),
        )
        or planning.get("use", {}).get("prospective_sample_size_planning_only")
        is not True
        or planning.get("use", {}).get("threshold_relaxation") is not False
        or planning.get("use", {}).get("failed_outcome_reinterpretation")
        is not False
        or planning.get("use", {}).get("v65_population_authority") is not False
    ):
        raise RuntimeError("v65b R1 prospective planning binding changed")
    return {
        "exact_builder_reconstruction": True,
        "implementation_closure_exact": True,
        "r1_numeric_evidence_and_planning_formula_exact": True,
        "r1_used_only_for_prospective_planning": True,
        "artifact_paths_and_success_exclusivity": artifact_paths,
        "sealed_runtime_constants": runtime_contract,
        "thresholds_relaxed_or_reinterpreted": False,
        "v65_population_authorized": False,
    }


def _verify_attempt_v65b(
    attempt: dict, sources: FinalizerSourcesV65B,
) -> dict:
    expected_keys = {
        "schema", "status", "phase", "started_at_utc",
        "preregistration_file_sha256", "preregistration_content_sha256",
        "runtime_determinism_controls", "base_model_artifact_receipt",
        "preflight", "fixed_unscored_warmup_periods",
        "fixed_scored_periods", "fixed_adjacent_blocks",
        "fixed_paired_replicas_per_unit",
        "fresh_process_warmup_state_from_r1_transferred",
        "gpu_hardware_health_monitor_required",
        "forbidden_clock_event_reason_mask",
        "fixed_actor_wait_timeout",
        "model_loaded_or_gpu_compute_started",
        "candidate_hpo_update_projection_or_protected_access",
        "v65_population_launch_authorized", "content_sha256_before_self_field",
    }
    preflight = attempt.get("preflight", {})
    memory = preflight.get("memory_used_mib", {}) if isinstance(preflight, dict) else {}
    _require_aware_timestamp_v65b(attempt.get("started_at_utc"), "attempt start")
    if (
        set(attempt) != expected_keys
        or attempt.get("schema") != "v65b-ranking64-high-rep-alpha-zero-attempt"
        or attempt.get("status")
        != "launching_fixed_high_rep_exact64_calibration_only"
        or attempt.get("phase")
        != "before_authorized_semantics_model_ray_or_gpu_load"
        or attempt.get("preregistration_file_sha256")
        != sources.preregistration.file_sha256
        or attempt.get("preregistration_content_sha256")
        != sources.preregistration.content_sha256
        or not runtime._canonical_equal_v65b(
            attempt.get("runtime_determinism_controls"),
            analysis.ENGINE_CONTROLS_V65B,
        )
        or not runtime._exact_int_v65b(
            attempt.get("fixed_unscored_warmup_periods"), 8,
        )
        or not runtime._exact_int_v65b(
            attempt.get("fixed_scored_periods"), 72,
        )
        or not runtime._exact_int_v65b(
            attempt.get("fixed_adjacent_blocks"), 36,
        )
        or not runtime._exact_int_v65b(
            attempt.get("fixed_paired_replicas_per_unit"), 144,
        )
        or attempt.get("fresh_process_warmup_state_from_r1_transferred") is not False
        or attempt.get("gpu_hardware_health_monitor_required") is not True
        or attempt.get("forbidden_clock_event_reason_mask")
        != runtime.FORBIDDEN_CLOCK_REASON_MASK_V65B
        or not runtime._canonical_equal_v65b(
            attempt.get("fixed_actor_wait_timeout"),
            runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B,
        )
        or attempt.get("model_loaded_or_gpu_compute_started") is not False
        or attempt.get("candidate_hpo_update_projection_or_protected_access")
        is not False
        or attempt.get("v65_population_launch_authorized") is not False
        or set(preflight) != {"compute_process_query_empty", "memory_used_mib"}
        or preflight.get("compute_process_query_empty") is not True
        or set(memory) != {"0", "1", "2", "3"}
        or any(type(value) is not int or not 0 <= value <= 2048
               for value in memory.values())
    ):
        raise RuntimeError("v65b launch attempt or idle preflight changed")
    expectation = runtime64.base_model_artifact_expectation_v64()
    receipt = runtime64.validate_base_model_artifact_receipt_v64(
        attempt.get("base_model_artifact_receipt"), expectation,
    )
    return {
        "started_at_utc": attempt["started_at_utc"],
        "exact_idle_four_gpu_preflight": True,
        "fresh_process_warmup_state_transferred": False,
        "base_model_receipt_sha256": population65.canonical_sha256_v65(receipt),
    }


def _verify_input_receipt_v65b(receipt: object) -> dict:
    expected_keys = {
        "schema", "path", "source_full_file_sha256_bound_but_not_recomputed",
        "authorized_prefix_bytes", "authorized_prefix_sha256",
        "decoded_ranking_rows", "requested_byte_offset_at_or_after_prefix",
        "remaining_exact_sentinel_rows_decoded",
        "question_answer_or_text_persisted", "request_count",
        "request_prompt_token_ids_sha256", "generation_seed",
        "generation_temperature", "generation_max_tokens",
        "submitted_request_batch_size", "runtime_determinism_controls",
        "lora_adapter_request_name", "lora_adapter_request_id",
        "lora_adapter_request_path",
    }
    if (
        not isinstance(receipt, dict) or set(receipt) != expected_keys
        or receipt.get("schema") != "v65-exact-authorized-ranking-prefix-receipt"
        or Path(receipt.get("path", "")).resolve()
        != population65.V61C_ROWS.resolve()
        or receipt.get("source_full_file_sha256_bound_but_not_recomputed")
        != population65.V61C_ROWS_FILE_SHA256
        or receipt.get("authorized_prefix_bytes")
        != analysis.v65a.RANKING_PREFIX_BYTES_V65A
        or receipt.get("authorized_prefix_sha256")
        != analysis.v65a.RANKING_PREFIX_SHA256_V65A
        or not runtime._exact_int_v65b(receipt.get("decoded_ranking_rows"), 64)
        or receipt.get("requested_byte_offset_at_or_after_prefix") is not False
        or not runtime._exact_int_v65b(
            receipt.get("remaining_exact_sentinel_rows_decoded"), 0,
        )
        or receipt.get("question_answer_or_text_persisted") is not False
        or not runtime._exact_int_v65b(receipt.get("request_count"), 64)
        or _require_sha256_v65b(
            receipt.get("request_prompt_token_ids_sha256"), "prompt-token identity",
        ) != builder.R1_REQUEST_PROMPT_TOKEN_IDS_SHA256
        or not runtime._exact_int_v65b(
            receipt.get("generation_seed"),
            analysis.COMMON_GENERATION_SEED_V65B,
        )
        or type(receipt.get("generation_temperature")) is not float
        or receipt.get("generation_temperature") != 0.0
        or not runtime._exact_int_v65b(
            receipt.get("generation_max_tokens"),
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V65B["max_tokens"],
        )
        or not runtime._exact_int_v65b(
            receipt.get("submitted_request_batch_size"), 64,
        )
        or not runtime._canonical_equal_v65b(
            receipt.get("runtime_determinism_controls"),
            analysis.ENGINE_CONTROLS_V65B,
        )
        or receipt.get("lora_adapter_request_name")
        != "v434_ranking64_alpha_zero_v65b"
        or not runtime._exact_int_v65b(
            receipt.get("lora_adapter_request_id"), 1,
        )
        or Path(receipt.get("lora_adapter_request_path", "")).resolve()
        != design52.STAGED_V52.resolve()
    ):
        raise RuntimeError("v65b exact authorized prefix input receipt changed")
    return {
        "decoded_ranking_rows": 64,
        "row_64_or_later_requested": False,
        "raw_text_persisted": False,
    }


def _verify_panel_metric_identities_v65b(
    scored: list, prereg: dict,
) -> dict:
    bound = prereg.get("ranking_panel", {})
    path = Path(bound.get("path", "")).resolve()
    payload = path.read_bytes() if path.is_file() else b""
    if (
        not path.is_file()
        or hashlib.sha256(payload).hexdigest() != bound.get("file_sha256")
    ):
        raise RuntimeError("v65b ranking panel file changed")
    panel = _json_bytes_without_duplicate_keys_v65b(payload, path)
    items = panel.get("items", []) if isinstance(panel, dict) else []
    expected = [
        (index, item.get("row_sha256"), item.get("unit_identity_sha256"))
        for index, item in enumerate(items) if isinstance(item, dict)
    ]
    observed = [
        (metric.get("request_index"), metric.get("row_sha256"),
         metric.get("unit_identity_sha256"))
        for metric in scored[0][0]
    ] if isinstance(scored, list) and scored else []
    if (
        not isinstance(panel, dict)
        or panel.get("content_sha256_before_self_field")
        != bound.get("content_sha256")
        or population65.self_content_sha256_v65(panel)
        != bound.get("content_sha256")
        or len(items) != 64 or len(expected) != 64
        or not runtime._canonical_equal_v65b(observed, expected)
        or len({value[1] for value in expected}) != 64
        or len({value[2] for value in expected}) != 64
    ):
        raise RuntimeError("v65b metric identities differ from sealed panel")
    return {"all_64_metric_identities_match_sealed_panel": True}


def _verify_master_certificate_v65b(receipt: object) -> dict:
    if (
        not isinstance(receipt, dict)
        or set(receipt) != runtime.MASTER_CERTIFICATE_KEYS_V65B
        or receipt.get("canonical_fp32_master_sha256")
        != design52.MASTER_SHA256_V52
        or receipt.get("canonical_master_identity_sha256")
        != analysis.v65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
        or receipt.get("bf16_runtime_values_sha256")
        != design52.MASTER_RUNTIME_SHA256_V52
    ):
        raise RuntimeError("v65b exact master certificate changed")
    _require_sha256_v65b(
        receipt.get("four_actor_certificate_sha256"),
        "four-actor master certificate",
    )
    if (
        receipt.get("four_actor_certificate_sha256")
        != runtime.FOUR_ACTOR_CERTIFICATE_SHA256_V65B
    ):
        raise RuntimeError("v65b four-actor certificate hash changed")
    return receipt


def _verify_read_aggregate_v65b(aggregate: object) -> dict:
    if (
        not isinstance(aggregate, dict)
        or set(aggregate) != runtime.READ_AGGREGATE_KEYS_V65B
        or aggregate.get("schema")
        != "v65b-read-only-four-actor-master-slot-consensus"
        or aggregate.get("canonical_fp32_master_sha256")
        != design52.MASTER_SHA256_V52
        or aggregate.get("canonical_master_identity_sha256")
        != analysis.v65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
        or aggregate.get("bf16_runtime_values_sha256")
        != design52.MASTER_RUNTIME_SHA256_V52
        or not runtime._exact_int_v65b(
            aggregate.get("runtime_view_count_per_actor"), 82,
        )
        or not runtime._exact_int_v65b(
            aggregate.get("runtime_elements_per_actor"), 4_921_344,
        )
        or aggregate.get("runtime_dtype") != "torch.bfloat16"
        or aggregate.get("base_inventory_sha256")
        != analysis.v65a.BASE_INVENTORY_SHA256_V65A
        or aggregate.get("four_actor_exact_read_only_consensus") is not True
    ):
        raise RuntimeError("v65b exact read-only state aggregate changed")
    return aggregate


def _verify_worker_timing_v65b(value: object) -> dict:
    if not runtime._valid_worker_timing_v65b(value):
        raise RuntimeError("v65b worker timing changed")
    return value


def _actor_binding_matches_v65b(
    actor: dict, identities: list[dict], rank: int,
) -> bool:
    if not isinstance(identities, list) or len(identities) != 4:
        return False
    identity = identities[rank]
    return (
        isinstance(identity, dict)
        and runtime._exact_int_v65b(identity.get("physical_gpu_id"), rank)
        and type(identity.get("pid")) is int
        and identity.get("pid") > 0
        and runtime._exact_int_v65b(
            actor.get("controller_actor_rank"), rank,
        )
        and type(actor.get("controller_expected_pid")) is int
        and actor.get("controller_expected_pid") == identity.get("pid")
        and runtime._exact_int_v65b(
            actor.get("controller_physical_gpu_id"),
            identity.get("physical_gpu_id"),
        )
        and type(actor.get("worker_pid")) is int
        and actor.get("worker_pid") == identity.get("pid")
        and runtime._exact_int_v65b(
            actor.get("worker_physical_gpu_id"),
            identity.get("physical_gpu_id"),
        )
        and actor.get("worker_cuda_visible_devices")
        == str(identity.get("physical_gpu_id"))
    )


def _verify_state_receipts_v65b(evidence: dict) -> dict:
    installed = evidence.get("installed_master_state")
    final = evidence.get("final_restored_master_state")
    if _verify_master_certificate_v65b(final) != _verify_master_certificate_v65b(
        installed
    ):
        raise RuntimeError("v65b final master state changed")
    reads = evidence.get("read_only_live_slot_receipts")
    if not isinstance(reads, list) or len(reads) != 160:
        raise RuntimeError("v65b state/read linkage coverage changed")
    ordinal = 0
    for key, kind, count in (
        ("warmup_state_receipts", "unscored_warmup", 8),
        ("scored_state_receipts", "scored", 72),
    ):
        values = evidence.get(key)
        if (
            not isinstance(values, list) or len(values) != count
            or evidence.get(f"{key}_sha256")
            != population65.canonical_sha256_v65(values)
        ):
            raise RuntimeError("v65b state receipt coverage changed")
        for index, receipt in enumerate(values):
            if (
                not isinstance(receipt, dict)
                or set(receipt) != {
                    "period_kind", "period_index", "before", "after",
                    "identical_v434_state",
                }
                or receipt.get("period_kind") != kind
                or not runtime._exact_int_v65b(
                    receipt.get("period_index"), index,
                )
                or _verify_read_aggregate_v65b(receipt.get("before"))
                != reads[2 * ordinal].get("aggregate")
                or _verify_read_aggregate_v65b(receipt.get("after"))
                != reads[2 * ordinal + 1].get("aggregate")
                or receipt.get("before") != receipt.get("after")
                or receipt.get("identical_v434_state") is not True
            ):
                raise RuntimeError("v65b state receipt changed")
            ordinal += 1
    return {"unchanged_period_state_receipts": 80}


def _verify_period_edges_v65b(evidence: dict) -> dict:
    writes = evidence.get("exact_master_slot_write_receipts")
    reads = evidence.get("read_only_live_slot_receipts")
    installed = _verify_master_certificate_v65b(
        evidence.get("installed_master_state")
    )
    installed_core = runtime.base65a._master_core_v65a(installed)
    actor_identities = evidence.get("actor_runtime_identities")
    coordinates = [
        (kind, index)
        for kind, count in (("unscored_warmup", 8), ("scored", 72))
        for index in range(count)
    ]
    write_keys = {
        "period_kind", "period_index", "pre_write_master",
        "post_write_master", "actors", "actor_receipts_sha256",
    }
    write_actor_keys = {
        "schema", "period_kind", "period_index", "master_identity",
        "materialization", "base_identity", "transaction_state_quiescent",
        "timing", *runtime.CONTROLLER_ACTOR_BINDING_KEYS_V65B,
        *runtime.INTRINSIC_WORKER_IDENTITY_KEYS_V65B,
    }
    if (
        not isinstance(writes, list) or len(writes) != 80
        or evidence.get("exact_master_slot_write_receipts_sha256")
        != population65.canonical_sha256_v65(writes)
    ):
        raise RuntimeError("v65b exact-master write coverage changed")
    for receipt, (kind, index) in zip(writes, coordinates, strict=True):
        actors = receipt.get("actors") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or set(receipt) != write_keys
            or receipt.get("period_kind") != kind
            or not runtime._exact_int_v65b(
                receipt.get("period_index"), index,
            )
            or _verify_master_certificate_v65b(
                receipt.get("pre_write_master")
            ) != installed
            or _verify_master_certificate_v65b(
                receipt.get("post_write_master")
            ) != installed
            or not isinstance(actors, list) or len(actors) != 4
            or receipt.get("actor_receipts_sha256")
            != population65.canonical_sha256_v65(actors)
            or any(
                not isinstance(actor, dict)
                or set(actor) != write_actor_keys
                or actor.get("schema") != "exact-master-slot-write-v65b"
                or actor.get("period_kind") != kind
                or not runtime._exact_int_v65b(
                    actor.get("period_index"), index,
                )
                or actor.get("transaction_state_quiescent") is not True
                or not _actor_binding_matches_v65b(
                    actor, actor_identities, rank,
                )
                or population65.canonical_sha256_v65(
                    actor.get("master_identity")
                ) != installed_core["canonical_master_identity_sha256"]
                or not runtime._valid_worker_timing_v65b(actor.get("timing"))
                for rank, actor in enumerate(actors)
            )
            or len({
                tuple(actor["timing"][key] for key in (
                    "started_ns", "ended_ns", "elapsed_ns",
                )) for actor in actors
            }) != 4
        ):
            raise RuntimeError("v65b ordered exact-master write changed")
        for actor in actors:
            basefinal._verify_master_identity_v65a(actor["master_identity"])
            basefinal._verify_materialization_v65a(
                actor["materialization"], "v65a_exact_master_slot_write",
            )
            basefinal._verify_base_check_v65a(
                actor["base_identity"], "v65a_exact_master_slot_write",
            )
    edges = [
        (kind, index, edge) for kind, index in coordinates
        for edge in ("before_generation", "after_generation")
    ]
    if (
        not isinstance(reads, list) or len(reads) != 160
        or evidence.get("read_only_live_slot_receipts_sha256")
        != population65.canonical_sha256_v65(reads)
    ):
        raise RuntimeError("v65b read-only edge coverage changed")
    read_keys = {
        "period_kind", "period_index", "edge", "aggregate", "actors",
        "actor_receipts_sha256",
    }
    aggregate_keys = set(runtime.READ_AGGREGATE_KEYS_V65B)
    read_actor_keys = {
        "schema", "period_kind", "period_index", "edge", "master_identity",
        "runtime_view_count", "runtime_elements", "runtime_dtype",
        "runtime_values_sha256", "active_lora_ids",
        "active_manager_cache_lora_ids", "base_identity",
        "transaction_state_quiescent", "slot_read_only_no_weight_write_or_reset",
        "timing", *runtime.CONTROLLER_ACTOR_BINDING_KEYS_V65B,
        *runtime.INTRINSIC_WORKER_IDENTITY_KEYS_V65B,
    }
    consensus = None
    for receipt, (kind, index, edge) in zip(reads, edges, strict=True):
        actors = receipt.get("actors") if isinstance(receipt, dict) else None
        aggregate = receipt.get("aggregate") if isinstance(receipt, dict) else None
        if (
            not isinstance(receipt, dict)
            or set(receipt) != read_keys
            or receipt.get("period_kind") != kind
            or not runtime._exact_int_v65b(
                receipt.get("period_index"), index,
            )
            or receipt.get("edge") != edge
            or not isinstance(aggregate, dict)
            or set(aggregate) != aggregate_keys
            or _verify_read_aggregate_v65b(aggregate) != aggregate
            or not isinstance(actors, list) or len(actors) != 4
            or receipt.get("actor_receipts_sha256")
            != population65.canonical_sha256_v65(actors)
            or any(
                not isinstance(actor, dict)
                or set(actor) != read_actor_keys
                or actor.get("schema") != "read-only-exact-master-slot-v65b"
                or actor.get("period_kind") != kind
                or not runtime._exact_int_v65b(
                    actor.get("period_index"), index,
                )
                or actor.get("edge") != edge
                or not _actor_binding_matches_v65b(
                    actor, actor_identities, rank,
                )
                or population65.canonical_sha256_v65(
                    actor.get("master_identity")
                ) != installed_core["canonical_master_identity_sha256"]
                or not runtime._exact_int_v65b(
                    actor.get("runtime_view_count"), 82,
                )
                or not runtime._exact_int_v65b(
                    actor.get("runtime_elements"), 4_921_344,
                )
                or actor.get("runtime_dtype") != "torch.bfloat16"
                or actor.get("runtime_values_sha256")
                != design52.MASTER_RUNTIME_SHA256_V52
                or not runtime._exact_active_lora_one_v65b(
                    actor.get("active_lora_ids")
                )
                or not runtime._exact_active_lora_one_v65b(
                    actor.get("active_manager_cache_lora_ids")
                )
                or actor.get("base_identity", {}).get("inventory_sha256")
                != aggregate.get("base_inventory_sha256")
                or actor.get("transaction_state_quiescent") is not True
                or actor.get("slot_read_only_no_weight_write_or_reset") is not True
                or not runtime._valid_worker_timing_v65b(actor.get("timing"))
                for rank, actor in enumerate(actors)
            )
            or len({
                tuple(actor["timing"][key] for key in (
                    "started_ns", "ended_ns", "elapsed_ns",
                )) for actor in actors
            }) != 4
        ):
            raise RuntimeError("v65b ordered read-only edge changed")
        if consensus is None:
            consensus = aggregate
        elif aggregate != consensus:
            raise RuntimeError("v65b read-only consensus changed between edges")
        for actor in actors:
            basefinal._verify_master_identity_v65a(actor["master_identity"])
            basefinal._verify_base_check_v65a(
                actor["base_identity"], "v65a_read_only_slot_receipt",
            )
    for ordinal, write in enumerate(writes):
        before = reads[2 * ordinal]
        after = reads[2 * ordinal + 1]
        for rank in range(4):
            write_timing = write["actors"][rank]["timing"]
            before_timing = before["actors"][rank]["timing"]
            after_timing = after["actors"][rank]["timing"]
            if (
                write_timing["ended_ns"] > before_timing["started_ns"]
                or before_timing["ended_ns"] > after_timing["started_ns"]
            ):
                raise RuntimeError("v65b within-period receipt timing changed")
    previous_ended = [None] * 4
    seen_timings = [set() for _ in range(4)]
    for ordinal, write in enumerate(writes):
        operations = (write, reads[2 * ordinal], reads[2 * ordinal + 1])
        for operation in operations:
            for rank, actor in enumerate(operation["actors"]):
                timing = actor["timing"]
                identity = (
                    timing["started_ns"], timing["ended_ns"],
                    timing["elapsed_ns"],
                )
                if (
                    identity in seen_timings[rank]
                    or previous_ended[rank] is not None
                    and timing["started_ns"] <= previous_ended[rank]
                ):
                    raise RuntimeError(
                        "v65b global period-edge receipt timing changed"
                    )
                seen_timings[rank].add(identity)
                previous_ended[rank] = timing["ended_ns"]
    return {
        "exact_master_slot_writes": 80,
        "read_only_pre_and_post_generation_edges": 160,
        "after_generation_edges_wrote_or_reset_slot": False,
    }


def _verify_evidence_v65b(
    evidence: dict, prereg: dict,
) -> tuple[dict, list, dict[int, int]]:
    expected_keys = {
        "schema", "status", "panel_content_sha256",
        "authorized_input_receipt", "canonical_fp32_master_sha256",
        "bf16_runtime_values_sha256", "row_count", "actor_count",
        "unscored_warmup_period_count", "scored_period_count",
        "adjacent_block_count", "four_period_superblock_count",
        "paired_replicas_per_unit", "label_plan", "schedule",
        "runtime_determinism_controls", "actor_runtime_identities",
        "fixed_actor_wait_timeout", "actor_wait_timeout_observed",
        "worker_runtime_identities", "active_lora_receipts",
        "active_lora_receipts_sha256", "initial_installations",
        "installed_master_state", "final_restored_master_state",
        "warmup_state_receipts", "warmup_state_receipts_sha256",
        "scored_state_receipts", "scored_state_receipts_sha256",
        "exact_master_slot_write_receipts",
        "exact_master_slot_write_receipts_sha256",
        "read_only_live_slot_receipts",
        "read_only_live_slot_receipts_sha256", "scored_periods",
        "numeric_scored_periods_sha256",
        "fresh_process_warmup_state_from_r1_transferred",
        "warmup_generation_completions_discarded",
        "scored_generation_completions", "total_generation_completions",
        "generation_only", "warmup_raw_outputs_persisted",
        "warmup_generation_metrics_computed_or_persisted",
        "adaptive_retry_drop_reorder_or_early_stop_performed", "alpha",
        "timeout_retry_drop_reorder_or_early_stop_performed",
        "sigma_or_direction",
        "adapter_update_candidate_hpo_or_projection_performed",
        "holdback_sentinel_reserve_ood_terminal_or_protected_opened",
        "raw_question_answer_prompt_or_generation_text_persisted",
        "v65_population_launch_authorized",
        "adapter_artifact_contract_prelaunch",
        "adapter_artifact_contract_postcleanup",
        "adapter_artifact_contract_unchanged",
        "content_sha256_before_self_field",
    }
    scored = evidence.get("scored_periods")
    if (
        set(evidence) != expected_keys
        or evidence.get("schema")
        != "v65b-ranking64-high-rep-generation-evidence"
        or evidence.get("status")
        != "complete_fixed_fresh_warmup_and_72_period_characterization"
        or evidence.get("panel_content_sha256")
        != prereg.get("ranking_panel", {}).get("content_sha256")
        or evidence.get("canonical_fp32_master_sha256")
        != design52.MASTER_SHA256_V52
        or evidence.get("bf16_runtime_values_sha256")
        != design52.MASTER_RUNTIME_SHA256_V52
        or not runtime._exact_int_v65b(evidence.get("row_count"), 64)
        or not runtime._exact_int_v65b(evidence.get("actor_count"), 4)
        or not runtime._exact_int_v65b(
            evidence.get("unscored_warmup_period_count"), 8,
        )
        or not runtime._exact_int_v65b(
            evidence.get("scored_period_count"), 72,
        )
        or not runtime._exact_int_v65b(
            evidence.get("adjacent_block_count"), 36,
        )
        or not runtime._exact_int_v65b(
            evidence.get("four_period_superblock_count"), 18,
        )
        or not runtime._exact_int_v65b(
            evidence.get("paired_replicas_per_unit"), 144,
        )
        or not runtime._canonical_equal_v65b(
            evidence.get("label_plan"), analysis.LABEL_PLAN_V65B,
        )
        or not runtime._canonical_equal_v65b(
            evidence.get("schedule"), analysis.validate_schedule_v65b(),
        )
        or not runtime._canonical_equal_v65b(
            evidence.get("runtime_determinism_controls"),
            analysis.ENGINE_CONTROLS_V65B,
        )
        or not runtime._canonical_equal_v65b(
            evidence.get("fixed_actor_wait_timeout"),
            runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B,
        )
        or evidence.get("actor_wait_timeout_observed") is not False
        or evidence.get("numeric_scored_periods_sha256")
        != population65.canonical_sha256_v65(scored)
        or evidence.get("fresh_process_warmup_state_from_r1_transferred")
        is not False
        or not runtime._exact_int_v65b(
            evidence.get("warmup_generation_completions_discarded"), 2_048,
        )
        or not runtime._exact_int_v65b(
            evidence.get("scored_generation_completions"), 18_432,
        )
        or not runtime._exact_int_v65b(
            evidence.get("total_generation_completions"), 20_480,
        )
        or evidence.get("generation_only") is not True
        or evidence.get("warmup_raw_outputs_persisted") is not False
        or evidence.get("warmup_generation_metrics_computed_or_persisted")
        is not False
        or evidence.get("adaptive_retry_drop_reorder_or_early_stop_performed")
        is not False
        or evidence.get("timeout_retry_drop_reorder_or_early_stop_performed")
        is not False
        or type(evidence.get("alpha")) is not float
        or evidence.get("alpha") != 0.0
        or evidence.get("sigma_or_direction") is not None
        or evidence.get("adapter_update_candidate_hpo_or_projection_performed")
        is not False
        or evidence.get(
            "holdback_sentinel_reserve_ood_terminal_or_protected_opened"
        ) is not False
        or evidence.get(
            "raw_question_answer_prompt_or_generation_text_persisted"
        ) is not False
        or evidence.get("v65_population_launch_authorized") is not False
    ):
        raise RuntimeError("v65b evidence schema, counts, or zero-access changed")
    analysis.validate_scored_periods_v65b(scored)
    panel_identities = _verify_panel_metric_identities_v65b(scored, prereg)
    prefix = _verify_input_receipt_v65b(evidence["authorized_input_receipt"])
    actors, pid_map = basefinal._verify_actor_and_lora_receipts_v65a(
        evidence, prereg,
    )
    adapter = basefinal._verify_adapter_artifact_contract_v65a(evidence)
    master = runtime.base65a._master_core_v65a(
        _verify_master_certificate_v65b(
            evidence.get("installed_master_state")
        )
    )
    installations = basefinal._verify_initial_installations_v65a(
        evidence, master,
    )
    edges = _verify_period_edges_v65b(evidence)
    states = _verify_state_receipts_v65b(evidence)
    return ({
        "authorized_prefix": prefix,
        "sealed_panel_metric_identities": panel_identities,
        "live_actor_and_active_lora": actors,
        "adapter_artifacts": adapter,
        "initial_installations": installations,
        "unchanged_state": states,
        "ordered_slot_edges": edges,
        "fixed_schedule_and_numeric_coverage": True,
        "zero_update_candidate_and_protected_access": True,
    }, scored, pid_map)


EXPECTED_GATE_KEYS_V65B = frozenset({
    "generated_f1_primary_interval_contains_zero",
    "joint_composite_interval_contains_zero",
    "stability_improvement_interval_contains_zero",
    "generated_f1_primary_ci_halfwidth_within_fixed_limit",
    "actor_leave_one_out_shift_within_fixed_limit",
    "both_temporal_pass_f1_intervals_contain_zero",
    "both_temporal_pass_joint_intervals_contain_zero",
    "both_temporal_pass_f1_halfwidths_within_fixed_limit",
    "orientation_effect_interval_contains_zero",
    "early_minus_late_joint_interval_contains_zero",
    "both_run_half_f1_joint_and_stability_intervals_contain_zero",
    "both_run_half_f1_halfwidths_within_fixed_limit",
})


def _rebuild_analysis_v65b(scored: list, evidence_content_sha256: str) -> dict:
    rebuilt = analysis.analyze_scored_periods_v65b(scored)
    rebuilt["source_evidence_content_sha256"] = evidence_content_sha256
    rebuilt["content_sha256_before_self_field"] = (
        population65.canonical_sha256_v65(rebuilt)
    )
    return rebuilt


def _verify_analysis_v65b(
    stored: dict, scored: list, evidence_content_sha256: str,
) -> dict:
    rebuilt = _rebuild_analysis_v65b(scored, evidence_content_sha256)
    if not runtime._canonical_equal_v65b(stored, rebuilt):
        raise RuntimeError("v65b stored analysis differs from numeric rebuild")
    gate = rebuilt.get("required_alpha_zero_gate", {})
    checks = gate.get("checks", {})
    primary = rebuilt.get("primary_cluster_bootstrap", {})
    if (
        rebuilt.get("schema")
        != "v65b-ranking64-high-rep-alpha-zero-analysis"
        or rebuilt.get("status")
        != "complete_numeric_only_high_rep_calibration"
        or set(checks) != EXPECTED_GATE_KEYS_V65B
        or any(type(value) is not bool for value in checks.values())
        or gate.get("passed") is not all(checks.values())
        or gate.get("thresholds_relaxed_or_reinterpreted_after_r1") is not False
        or gate.get("success_directly_authorizes_v65_population") is not False
        or primary.get("six_epoch_intervals_are_sealed_non_gating_diagnostics")
        is not True
        or primary.get("superblock_influence", {}).get(
            "sealed_non_gating_diagnostic"
        ) is not True
        or rebuilt.get("r1_used_for_prospective_sample_size_planning_only")
        is not True
        or rebuilt.get("r1_threshold_relaxation_or_outcome_reinterpretation")
        is not False
        or rebuilt.get("v65_population_launch_authorized") is not False
    ):
        raise RuntimeError("v65b rebuilt gate or diagnostics changed")
    failed = sorted(key for key, value in checks.items() if not value)
    return {
        "rebuilt": rebuilt,
        "gate_observation": {
            "checks": copy.deepcopy(checks),
            "passed": gate["passed"],
            "passed_gate_count": len(checks) - len(failed),
            "failed_gate_count": len(failed),
            "failed_gates": failed,
            "v65_population_launch_authorized": False,
        },
    }


def _allowed_gpu_phases_v65b() -> set[str]:
    phases = {
        "setup", "activate_v434_lora_slot_all_actors",
        "install_exact_v434_master_all_actors",
        "final_exact_master_restoration_certificate",
    }
    for kind, count, prefix in (
        ("warmup", 8, "unscored_warmup"),
        ("scored", 72, "scored_period"),
    ):
        del kind
        for index in range(count):
            phases.update({
                f"{prefix}_{index}_exact_master_slot_write",
                f"{prefix}_{index}_generation_all_actors",
                f"{prefix}_{index}_post_generation_integrity",
            })
            if prefix == "scored_period":
                phases.update({
                    f"{prefix}_{index}_numeric_reduction",
                    f"{prefix}_{index}_complete",
                })
    return phases


def _parse_gpu_rows_v65b(path: Path) -> list[dict]:
    return _parse_gpu_payload_v65b(Path(path).read_bytes(), path)


def _parse_gpu_payload_v65b(payload: bytes, source_name: object) -> list[dict]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RuntimeError(f"v65b non-UTF-8 GPU log: {source_name}") from error
    rows = []
    for line_number, line in enumerate(
        text.splitlines(), start=1,
    ):
        if not line:
            continue

        def reject(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise RuntimeError(
                        f"v65b duplicate GPU JSON key at row {line_number}: {key}"
                    )
                value[key] = item
            return value

        try:
            rows.append(json.loads(
                line, object_pairs_hook=reject,
                parse_constant=lambda value: (_ for _ in ()).throw(
                    RuntimeError(
                        f"v65b non-finite GPU JSON constant at row "
                        f"{line_number}: {value}"
                    )
                ),
            ))
        except json.JSONDecodeError as error:
            raise RuntimeError(f"v65b malformed GPU row {line_number}") from error
    if not rows:
        raise RuntimeError("v65b GPU log is empty")
    return rows


def _verify_gpu_row_v65b(
    row: object, pid_map: dict[int, int], allowed_phases: set[str],
) -> dict:
    if not isinstance(row, dict) or set(row) != runtime.GPU_SAMPLE_KEYS_V65B:
        raise RuntimeError("v65b GPU row schema changed")
    gpu = row.get("gpu")
    compute = row.get("compute_pids")
    reasons = row.get("clock_event_reasons_bitmask")
    forbidden = row.get("forbidden_hardware_or_thermal_reasons_bitmask")
    allowed = row.get("allowed_diagnostic_reasons_bitmask")
    known_mask = (
        runtime.ALLOWED_DIAGNOSTIC_CLOCK_REASON_MASK_V65B
        | runtime.FORBIDDEN_CLOCK_REASON_MASK_V65B
    )
    if (
        type(gpu) is not int or gpu not in range(4)
        or type(row.get("expected_pid")) is not int
        or row.get("expected_pid") != pid_map[gpu]
        or not isinstance(compute, list) or compute != sorted(set(compute))
        or any(type(pid) is not int or pid <= 0 for pid in compute)
        or any(pid != pid_map[gpu] for pid in compute)
        or row.get("foreign_compute_pids") != []
        or type(row.get("utilization_percent")) is not int
        or not 0 <= row["utilization_percent"] <= 100
        or any(
            type(row.get(key)) is not int or row[key] < 0
            for key in ("memory_used_mib", "temperature_c", "power_draw_mw")
        )
        or type(reasons) is not int or reasons < 0
        or type(allowed) is not int or allowed < 0
        or type(forbidden) is not int or forbidden < 0
        or reasons & ~known_mask
        or row.get("clock_event_reasons_hex") != f"0x{reasons:016x}"
        or allowed != reasons & runtime.ALLOWED_DIAGNOSTIC_CLOCK_REASON_MASK_V65B
        or forbidden != reasons & runtime.FORBIDDEN_CLOCK_REASON_MASK_V65B
        or row.get("hardware_or_thermal_slowdown_active") is not bool(forbidden)
        or row.get("phase") not in allowed_phases
        or row.get("generation_phase") is not runtime.is_generation_phase_v65b(
            row.get("phase")
        )
        or row.get("generation_phase") is True and forbidden != 0
    ):
        raise RuntimeError("v65b GPU row attribution or health changed")
    _require_aware_timestamp_v65b(row.get("sampled_at_utc"), "GPU sample")
    return row


def _gpu_summary_v65b(
    path: Path, expected_sha256: str, pid_map: dict[int, int], report: dict,
) -> dict:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError("v65b GPU log file changed")
    rows = _parse_gpu_payload_v65b(payload, path)
    _verify_no_text_keys_v65b("gpu_log", rows)
    allowed_phases = _allowed_gpu_phases_v65b()
    rows = [
        _verify_gpu_row_v65b(row, pid_map, allowed_phases) for row in rows
    ]
    started = datetime.fromisoformat(_require_aware_timestamp_v65b(
        report.get("started_at_utc"), "report start",
    ))
    completed = datetime.fromisoformat(_require_aware_timestamp_v65b(
        report.get("completed_at_utc"), "report completion",
    ))
    times = [datetime.fromisoformat(row["sampled_at_utc"]) for row in rows]
    if (
        completed < started
        or any(value < started or value > completed for value in times)
        or any(current < previous
               for previous, current in zip(times, times[1:]))
    ):
        raise RuntimeError("v65b GPU timestamp sequence changed")
    cycles = runtime._gpu_sample_cycles_v65b(rows)
    wall = report.get("wall_runtime_seconds")
    cycle_span = (
        datetime.fromisoformat(cycles[-1]["sampled_at_utc"])
        - datetime.fromisoformat(cycles[0]["sampled_at_utc"])
    ).total_seconds()
    if (
        type(wall) is not float or not math.isfinite(wall) or wall <= 0.0
        or wall + runtime.REPORT_WALL_CLOCK_TOLERANCE_SECONDS_V65B
        < cycle_span
    ):
        raise RuntimeError("v65b wall runtime is shorter than GPU sample span")

    phases = [
        f"unscored_warmup_{index}_generation_all_actors" for index in range(8)
    ] + [
        f"scored_period_{index}_generation_all_actors" for index in range(72)
    ]
    phase_ranges = []
    by_phase = {}
    for phase in phases:
        phase_offsets = [
            index for index, cycle in enumerate(cycles)
            if cycle["phase"] == phase
        ]
        if not phase_offsets:
            raise RuntimeError(f"v65b missing generation phase: {phase}")
        phase_ranges.append((min(phase_offsets), max(phase_offsets)))
        simultaneous = [
            cycle for cycle in cycles if cycle["phase"] == phase and all(
                pid_map[row["gpu"]] in row["compute_pids"]
                and row["utilization_percent"] > 0
                for row in cycle["rows"]
            )
        ]
        if not simultaneous:
            raise RuntimeError(
                f"v65b no simultaneous positive four-GPU cycle in {phase}"
            )
        by_gpu = {}
        for gpu in range(4):
            selected = [
                row for row in rows
                if row["phase"] == phase and row["gpu"] == gpu
            ]
            resident = [
                row for row in selected if pid_map[gpu] in row["compute_pids"]
            ]
            if (
                not resident
                or not any(row["utilization_percent"] > 0 for row in resident)
                or any(row["forbidden_hardware_or_thermal_reasons_bitmask"]
                       for row in selected)
            ):
                raise RuntimeError(
                    f"v65b GPU {gpu} attribution/health failed in {phase}"
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
                "peak_temperature_c": max(
                    row["temperature_c"] for row in resident
                ),
                "peak_power_draw_mw": max(
                    row["power_draw_mw"] for row in resident
                ),
                "allowed_diagnostic_clock_reason_observations": sum(
                    bool(row["allowed_diagnostic_reasons_bitmask"])
                    for row in resident
                ),
                "forbidden_hardware_or_thermal_reason_observations": 0,
            }
        by_phase[phase] = by_gpu
    if any(
        left[1] >= right[0]
        for left, right in zip(phase_ranges, phase_ranges[1:])
    ):
        raise RuntimeError("v65b warmup/scored GPU phase order changed")
    cycle_times = [
        datetime.fromisoformat(cycle["sampled_at_utc"]) for cycle in cycles
    ]
    maximum_gap = max(
        (current - previous).total_seconds()
        for previous, current in zip(cycle_times, cycle_times[1:])
    ) if len(cycle_times) > 1 else 0.0
    gap_within_sealed_limit = (
        maximum_gap <= runtime.MAXIMUM_GPU_SAMPLE_CYCLE_GAP_SECONDS_V65B
    )
    if not gap_within_sealed_limit:
        raise RuntimeError("v65b GPU sample gap exceeded the sealed limit")
    rebuilt_phases = {
        "schema": "v65b-per-period-four-gpu-activity-and-health",
        "generation_phases": 80,
        "all_eighty_generation_phases_positive_on_all_four_gpus": True,
        "all_eighty_generation_phases_have_a_simultaneous_positive_four_gpu_cycle": True,
        "complete_four_gpu_sample_cycles": len(cycles),
        "maximum_consecutive_sample_cycle_gap_seconds": maximum_gap,
        "sample_cycle_gap_within_sealed_limit": gap_within_sealed_limit,
        "foreign_compute_process_observations": 0,
        "thermal_or_hardware_slowdown_observations_in_generation": 0,
        "allowed_idle_app_clock_or_software_power_cap_reasons_are_nonfailing": True,
        "by_phase": by_phase,
    }
    if not runtime._canonical_equal_v65b(
        report.get("gpu_period_phases_and_hardware_health"), rebuilt_phases,
    ):
        raise RuntimeError("v65b reported GPU phase/health summary changed")

    by_gpu = {}
    for gpu in range(4):
        selected = [row for row in rows if row["gpu"] == gpu]
        resident = [
            row for row in selected if pid_map[gpu] in row["compute_pids"]
        ]
        if not resident or not any(row["utilization_percent"] > 0 for row in resident):
            raise RuntimeError(f"v65b GPU {gpu} had no positive resident sample")
        by_gpu[str(gpu)] = {
            "expected_pid": pid_map[gpu],
            "samples": len(selected),
            "resident_samples": len(resident),
            "positive_samples": sum(
                row["utilization_percent"] > 0 for row in resident
            ),
            "mean_resident_utilization_percent": math.fsum(
                row["utilization_percent"] for row in resident
            ) / len(resident),
            "peak_utilization_percent": max(
                row["utilization_percent"] for row in resident
            ),
            "peak_memory_used_mib": max(
                row["memory_used_mib"] for row in resident
            ),
        }
    rebuilt_overall = {"all_four_attributed_positive": True, "by_gpu": by_gpu}
    if not runtime._canonical_equal_v65b(
        report.get("gpu_activity"), rebuilt_overall,
    ):
        raise RuntimeError("v65b reported overall GPU summary changed")
    return {
        "gpu_log_rows": len(rows),
        "all_four_gpus_attributed_positive": True,
        "all_eighty_generation_phases_positive_on_all_four_gpus": True,
        "all_eighty_generation_phases_have_a_simultaneous_positive_four_gpu_cycle": True,
        "complete_four_gpu_sample_cycles": len(cycles),
        "maximum_consecutive_sample_cycle_gap_seconds": maximum_gap,
        "sample_cycle_gap_within_sealed_limit": gap_within_sealed_limit,
        "eight_warmups_preceded_seventy_two_scored_periods": True,
        "foreign_compute_process_observations": 0,
        "thermal_or_hardware_slowdown_observations_in_generation": 0,
        "by_gpu": by_gpu,
        "by_phase": by_phase,
    }


def _verify_postrun_prefix_v65b(receipt: object) -> dict:
    return basefinal._verify_postrun_prefix_v65a(receipt)


def _verify_bounded_reward_pool_cleanup_v65b(receipt: object) -> dict:
    expected = {
        "schema": "v65b-bounded-reward-pool-shutdown",
        "timeout_seconds": (
            runtime.FIXED_REWARD_POOL_SHUTDOWN_TIMEOUT_SECONDS_V65B
        ),
        "pool_found": True,
        "worker_process_count": 8,
        "terminate_completed_before_deadline": True,
        "join_completed_before_deadline": True,
        "forced_worker_kill_count": 0,
        "all_workers_stopped": True,
    }
    if not runtime._canonical_equal_v65b(receipt, expected):
        raise RuntimeError("v65b bounded reward-pool cleanup changed")
    return expected


def _verify_report_clock_v65b(report: dict) -> dict:
    started_text = _require_aware_timestamp_v65b(
        report.get("started_at_utc"), "report start",
    )
    completed_text = _require_aware_timestamp_v65b(
        report.get("completed_at_utc"), "report completion",
    )
    utc_runtime_seconds = (
        datetime.fromisoformat(completed_text)
        - datetime.fromisoformat(started_text)
    ).total_seconds()
    wall = report.get("wall_runtime_seconds")
    if (
        type(wall) is not float or not math.isfinite(wall) or wall <= 0.0
        or utc_runtime_seconds < 0.0
        or abs(wall - utc_runtime_seconds)
        > runtime.REPORT_WALL_CLOCK_TOLERANCE_SECONDS_V65B
    ):
        raise RuntimeError("v65b UTC and monotonic wall clocks disagree")
    return {
        "started_at_utc": started_text,
        "completed_at_utc": completed_text,
        "utc_runtime_seconds": utc_runtime_seconds,
        "wall_runtime_seconds": wall,
        "absolute_clock_difference_seconds": abs(wall - utc_runtime_seconds),
        "within_sealed_tolerance": True,
    }


def _verify_report_v65b(
    report: dict, attempt: dict, evidence: dict, analysis_check: dict,
    prereg: dict, sources: FinalizerSourcesV65B, pid_map: dict[int, int],
) -> dict:
    expected_keys = {
        "schema", "status", "started_at_utc", "completed_at_utc",
        "wall_runtime_seconds", "preregistration_file_sha256",
        "preregistration_content_sha256", "attempt", "evidence", "analysis",
        "runtime_determinism_controls", "actor_runtime_identities",
        "fixed_actor_wait_timeout", "actor_wait_timeout_observed",
        "base_model_prelaunch_artifact_receipt",
        "base_model_postrun_artifact_receipt",
        "adapter_artifact_contract_prelaunch",
        "adapter_artifact_contract_postcleanup",
        "adapter_artifact_contract_unchanged",
        "authorized_prefix_postrun_receipt", "gpu_activity",
        "gpu_period_phases_and_hardware_health", "gpu_log_file_sha256",
        "bounded_reward_pool_cleanup", "cleanup", "final_gpu_idle",
        "warmup_generation_completions_discarded",
        "scored_generation_completions", "total_generation_completions",
        "raw_question_answer_prompt_or_generation_text_persisted",
        "candidate_hpo_update_projection_or_promotion_performed",
        "holdback_sentinel_reserve_ood_terminal_or_protected_opened",
        "fresh_process_warmup_state_from_r1_transferred",
        "thermal_or_hardware_slowdown_observed_during_generation",
        "v65_population_launch_authorized", "content_sha256_before_self_field",
    }
    gate = analysis_check["rebuilt"]["required_alpha_zero_gate"]
    expected_status = (
        "complete_gate_passed_population_still_unauthorized"
        if gate["passed"] else "complete_gate_failed_closed"
    )
    clock = _verify_report_clock_v65b(report)
    started_text = clock["started_at_utc"]
    completed_text = clock["completed_at_utc"]
    refs = {
        "attempt": sources.attempt,
        "evidence": sources.evidence,
        "analysis": sources.analysis,
    }
    if (
        set(report) != expected_keys
        or report.get("schema") != "v65b-ranking64-high-rep-alpha-zero-report"
        or report.get("status") != expected_status
        or report.get("started_at_utc") != attempt.get("started_at_utc")
        or report.get("preregistration_file_sha256")
        != sources.preregistration.file_sha256
        or report.get("preregistration_content_sha256")
        != sources.preregistration.content_sha256
        or any(
            not isinstance(report.get(name), dict)
            or set(report[name]) != (
                {"path", "file_sha256", "content_sha256",
                 "required_alpha_zero_gate"}
                if name == "analysis"
                else {"path", "file_sha256", "content_sha256"}
            )
            or Path(report[name].get("path", "")).resolve()
            != source.path.resolve()
            or report[name].get("file_sha256") != source.file_sha256
            or report[name].get("content_sha256") != source.content_sha256
            for name, source in refs.items()
        )
        or not runtime._canonical_equal_v65b(
            report.get("analysis", {}).get("required_alpha_zero_gate"), gate,
        )
        or not runtime._canonical_equal_v65b(
            report.get("runtime_determinism_controls"),
            analysis.ENGINE_CONTROLS_V65B,
        )
        or not runtime._canonical_equal_v65b(
            report.get("fixed_actor_wait_timeout"),
            runtime.ACTOR_WAIT_TIMEOUT_CONTRACT_V65B,
        )
        or report.get("actor_wait_timeout_observed") is not False
        or not runtime._canonical_equal_v65b(
            report.get("actor_runtime_identities"),
            evidence.get("actor_runtime_identities"),
        )
        or not runtime._canonical_equal_v65b(
            report.get("adapter_artifact_contract_prelaunch"),
            evidence.get("adapter_artifact_contract_prelaunch"),
        )
        or not runtime._canonical_equal_v65b(
            report.get("adapter_artifact_contract_postcleanup"),
            evidence.get("adapter_artifact_contract_postcleanup"),
        )
        or report.get("adapter_artifact_contract_unchanged") is not True
        or report.get("gpu_log_file_sha256") != sources.gpu_log_file_sha256
        or not runtime._exact_int_v65b(
            report.get("warmup_generation_completions_discarded"), 2_048,
        )
        or not runtime._exact_int_v65b(
            report.get("scored_generation_completions"), 18_432,
        )
        or not runtime._exact_int_v65b(
            report.get("total_generation_completions"), 20_480,
        )
        or report.get("raw_question_answer_prompt_or_generation_text_persisted")
        is not False
        or report.get("candidate_hpo_update_projection_or_promotion_performed")
        is not False
        or report.get(
            "holdback_sentinel_reserve_ood_terminal_or_protected_opened"
        ) is not False
        or report.get("fresh_process_warmup_state_from_r1_transferred") is not False
        or report.get("thermal_or_hardware_slowdown_observed_during_generation")
        is not False
        or report.get("v65_population_launch_authorized") is not False
    ):
        raise RuntimeError("v65b report or source hash chain changed")
    expectation = runtime64.base_model_artifact_expectation_v64()
    pre = runtime64.validate_base_model_artifact_receipt_v64(
        report.get("base_model_prelaunch_artifact_receipt"), expectation,
    )
    post = runtime64.validate_base_model_artifact_receipt_v64(
        report.get("base_model_postrun_artifact_receipt"), expectation,
    )
    if pre != post or pre != attempt.get("base_model_artifact_receipt"):
        raise RuntimeError("v65b base-model receipts changed")
    prefix = _verify_postrun_prefix_v65b(
        report.get("authorized_prefix_postrun_receipt")
    )
    gpu = _gpu_summary_v65b(
        sources.gpu_log_path, sources.gpu_log_file_sha256, pid_map, report,
    )
    pool_cleanup = _verify_bounded_reward_pool_cleanup_v65b(
        report.get("bounded_reward_pool_cleanup")
    )
    cleanup = basefinal._verify_cleanup_v65a(report)
    return {
        "reported_status_matches_rebuilt_gate": True,
        "UTC_and_monotonic_wall_clocks_reconciled": clock,
        "attempt_evidence_analysis_and_gpu_hash_chain_exact": True,
        "base_model_prelaunch_and_postrun_receipts_exact": True,
        "postrun_authorized_prefix_receipt": prefix,
        "gpu_activity_and_hardware_health_recomputed": gpu,
        "bounded_reward_pool_cleanup": pool_cleanup,
        "cleanup_and_final_idle": cleanup,
        "v65_population_launch_authorized": False,
    }


def _performance_v65b(report: dict) -> dict:
    wall = float(report["wall_runtime_seconds"])
    return {
        "wall_runtime_seconds": wall,
        "unscored_warmup_generation_completions": 2_048,
        "scored_generation_completions": 18_432,
        "total_generation_completions": 20_480,
        "total_generation_completions_per_second": 20_480 / wall,
        "scored_generation_completions_per_second": 18_432 / wall,
    }


def build_finalized_v65b(sources: FinalizerSourcesV65B) -> dict:
    _verify_source_argument_paths_v65b(sources, builder.artifacts_v65b())
    prereg = _read_self_hashed_v65b(sources.preregistration)
    attempt = _read_self_hashed_v65b(sources.attempt)
    evidence = _read_self_hashed_v65b(sources.evidence)
    stored_analysis = _read_self_hashed_v65b(sources.analysis)
    report = _read_self_hashed_v65b(sources.report)
    leakage = {
        name: _verify_no_text_keys_v65b(name, value)
        for name, value in (
            ("preregistration", prereg), ("attempt", attempt),
            ("evidence", evidence), ("analysis", stored_analysis),
            ("report", report),
        )
    }
    static = _verify_preregistration_v65b(prereg, sources)
    attempt_check = _verify_attempt_v65b(attempt, sources)
    evidence_check, scored, pid_map = _verify_evidence_v65b(evidence, prereg)
    analysis_check = _verify_analysis_v65b(
        stored_analysis, scored, sources.evidence.content_sha256,
    )
    report_check = _verify_report_v65b(
        report, attempt, evidence, analysis_check, prereg, sources, pid_map,
    )
    value = {
        "schema": "v65b-ranking64-high-rep-independent-finalizer",
        "status": "complete_numeric_only_observation_v65_still_unauthorized",
        "source_hashes": {
            name: {
                "file_sha256": source.file_sha256,
                "content_sha256": source.content_sha256,
            }
            for name, source in (
                ("preregistration", sources.preregistration),
                ("attempt", sources.attempt), ("evidence", sources.evidence),
                ("analysis", sources.analysis), ("report", sources.report),
            )
        },
        "gpu_log_file_sha256": sources.gpu_log_file_sha256,
        "verification": {
            "all_six_file_hashes_and_five_self_hashes_verified": True,
            "static_preregistration_r1_planning_and_implementation_chain": static,
            "launch_attempt_and_preflight": attempt_check,
            "numeric_hash_only_evidence": evidence_check,
            "stored_analysis_exactly_equals_numeric_rebuild": True,
            "no_forbidden_text_keys": leakage,
            "report_gpu_health_cleanup_and_idle": report_check,
        },
        "observed_numeric_outcome_without_authorization": {
            "primary_cluster_bootstrap": copy.deepcopy(
                analysis_check["rebuilt"]["primary_cluster_bootstrap"]
            ),
            "actor_influence": copy.deepcopy(
                analysis_check["rebuilt"]["actor_influence"]
            ),
            "required_alpha_zero_gate": analysis_check["gate_observation"],
            "performance": _performance_v65b(report),
        },
        "frozen_non_authorization": {
            "finalizer_accepts_and_records_either_gate_outcome": True,
            "thresholds_changed_after_r1_or_v65b_outcome": False,
            "failed_gate_reinterpreted_or_relaxed": False,
            "v65_population_launch_authorized": False,
            "adapter_update_candidate_projection_or_promotion_authorized": False,
            "holdback_sentinel_reserve_ood_terminal_or_protected_access_authorized": False,
            "model_ray_or_gpu_launch_authorized": False,
        },
        "implementation_bindings": {
            "finalizer_file_sha256": file_sha256_v65b(Path(__file__).resolve()),
            "tests_file_sha256": file_sha256_v65b(TESTS),
        },
        "raw_question_answer_prompt_prediction_or_generation_text_persisted": False,
        "row_64_or_later_opened": False,
        "protected_semantics_opened": False,
        "v65_population_launch_authorized": False,
    }
    value["content_sha256_before_self_field"] = (
        population65.canonical_sha256_v65(value)
    )
    return value


def _exclusive_write_v65b(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent,
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _source_from_args_v65b(args, name: str) -> SelfHashedSourceV65B:
    return SelfHashedSourceV65B(
        path=Path(getattr(args, name)).resolve(),
        file_sha256=_require_sha256_v65b(
            getattr(args, f"{name}_sha256"), f"{name} file",
        ),
        content_sha256=_require_sha256_v65b(
            getattr(args, f"{name}_content_sha256"), f"{name} content",
        ),
    )


def parser_v65b() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for name in ("preregistration", "attempt", "evidence", "analysis", "report"):
        option = name.replace("_", "-")
        parser.add_argument(f"--{option}", required=True)
        parser.add_argument(f"--{option}-sha256", required=True)
        parser.add_argument(f"--{option}-content-sha256", required=True)
    parser.add_argument("--gpu-log", required=True)
    parser.add_argument("--gpu-log-sha256", required=True)
    parser.add_argument("--output", default=str(OUTPUT))
    return parser


def _require_finalized_output_path_v65b(path: Path) -> Path:
    output = Path(path).resolve()
    if output != OUTPUT.resolve():
        raise RuntimeError("v65b finalizer output differs from preregistration")
    return output


def main(argv=None) -> int:
    args = parser_v65b().parse_args(argv)
    sources = FinalizerSourcesV65B(
        preregistration=_source_from_args_v65b(args, "preregistration"),
        attempt=_source_from_args_v65b(args, "attempt"),
        evidence=_source_from_args_v65b(args, "evidence"),
        analysis=_source_from_args_v65b(args, "analysis"),
        report=_source_from_args_v65b(args, "report"),
        gpu_log_path=Path(args.gpu_log).resolve(),
        gpu_log_file_sha256=_require_sha256_v65b(
            args.gpu_log_sha256, "GPU log file",
        ),
    )
    output = _require_finalized_output_path_v65b(Path(args.output))
    if output.exists():
        raise FileExistsError(output)
    value = build_finalized_v65b(sources)
    failure_path = Path(builder.artifacts_v65b()["failure"])
    with runtime.terminal_artifact_lock_v65b(output.parent):
        if failure_path.exists() or output.exists():
            raise RuntimeError("v65b success/failure artifact exclusivity changed")
        _exclusive_write_v65b(
            output,
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v65b(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "required_alpha_zero_gate_passed": value[
            "observed_numeric_outcome_without_authorization"
        ]["required_alpha_zero_gate"]["passed"],
        "thermal_or_hardware_slowdown_observations_in_generation": 0,
        "v65_population_launch_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
