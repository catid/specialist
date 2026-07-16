#!/usr/bin/env python3
"""Independently finalize the numeric/hash-only V65A calibration.

The finalizer accepts only caller-supplied exact file/content hashes.  It
does not read dataset semantics, load a model, use Ray, use a GPU, or grant
authority to launch V65.  Its job is to fail closed unless the complete
V65A transcript can be reconstructed from the sealed numeric artifacts.
"""

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

import build_lora_es_ranking64_alpha_zero_preregistration_v65a as builder
import lora_es_nested_population_v52 as design52
import lora_es_ranking64_alpha_zero_calibration_v65a as analysis
import lora_es_robust_sampling_population_v65 as population65
import run_lora_es_ranking64_alpha_zero_calibration_v65a as runtime
import run_lora_es_nested_population_v52 as runtime52
import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime64


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    runtime.RUN_DIR / "ranking64_alpha_zero_finalized_v65a_r1.json"
).resolve()
TESTS = (
    ROOT / "test_finalize_lora_es_ranking64_alpha_zero_v65a.py"
).resolve()

SHA256_LENGTH_V65A = 64
RUNTIME_MODULE_MANIFEST_SHA256_V65A = (
    "f09f656d7890c8776170bcc65e9273fbafefad2651a9ab6bc2ef805dfae6eeca"
)
MASTER_IDENTITY_OBJECT_SHA256_V65A = (
    "a73b7ca35dee943e4e0c427a7e6f35648affb803ac11b55958dbf95019aab155"
)
ASSIGNMENT_SHA256_V65A = (
    "bac008805d7fc7c6279c47255d8d1563b0be978cb21109e8c013114f143e09df"
)
MATERIALIZATION_STORAGE_LAYOUT_SHA256_V65A = (
    "30a6adf9b47290e5954efa126bd7f51d0fb7fe9b3aa038188d1423627d97c8e5"
)
BASE_INVENTORY_SHA256_V65A = (
    "141fe85d7ac7512f18f7fb81e53677642d66a6d06ca5dee838e1f439646b8773"
)
REGISTERED_SLOT_RECORDS_SHA256_V65A = (
    "c7a5ce898287b80765330f1d5c7616f1baf4c9eaab971778b0ec817edb0ce8d8"
)
PREDECESSOR_RANKING_PANEL_V65A = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v65a_ranking64_alpha_zero_panel.json"
).resolve()
PREDECESSOR_RANKING_PANEL_FILE_SHA256_V65A = (
    "916aaa6d30d059207a4da02ce67368b3707bb46ac17e2fe1636a4f4e4aa48094"
)
PREDECESSOR_RANKING_PANEL_CONTENT_SHA256_V65A = (
    "267337eb9711b8592178afa98482d188edd4db4e715f74eb63c0a64fb90330c8"
)
PREDECESSOR_PREREGISTRATION_V65A = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "ranking64_alpha_zero_calibration_v65a.json"
).resolve()
PREDECESSOR_PREREGISTRATION_FILE_SHA256_V65A = (
    "473148e96b3c0153fa32abe2d2790bc089ef42cb854f6bf4808946598a892b99"
)
PREDECESSOR_PREREGISTRATION_CONTENT_SHA256_V65A = (
    "5de8762a06e90611f8f439cf6606c598b8f26b0a809e3db206df2a8f2e5496f3"
)
PREDECESSOR_ATTEMPT_V65A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".v65a_ranking64_alpha_zero_calibration.attempt.json"
).resolve()
PREDECESSOR_ATTEMPT_FILE_SHA256_V65A = (
    "d64f88a3924d34557be38bc2f43b0a714017fd6eefc6d79baaf7cb3af1510f2e"
)
PREDECESSOR_ATTEMPT_CONTENT_SHA256_V65A = (
    "b44bbd11525ea3f4f9be8378c97955f8c4b9edb0eecf44b4b4e7a5945298f2b6"
)
PREDECESSOR_FAILURE_V65A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v65a_ranking64_alpha_zero_calibration/failure_v65a.json"
).resolve()
PREDECESSOR_FAILURE_FILE_SHA256_V65A = (
    "bb642eba1a5e4ceaa53e14a67f68eb39544de613e7c6b04089e34c56a9640011"
)
PREDECESSOR_FAILURE_CONTENT_SHA256_V65A = (
    "99a00ab1dfd9c3ce1811206cfacd17643fcdf2062fa692012fff3ed47aedfdf7"
)
PREDECESSOR_FAILURE_MESSAGE_SHA256_V65A = (
    "50af24ded76854ac98724e20e8fbffcec146aae3228734b092f35151a2204e62"
)
EXPECTED_PHASES_V65A = [
    *(f"unscored_warmup_{index}_generation_all_actors" for index in range(4)),
    *(f"scored_period_{index}_generation_all_actors" for index in range(4)),
]
ALLOWED_GPU_PHASES_V65A = frozenset({
    "setup",
    "activate_v434_lora_slot_all_actors",
    "install_exact_v434_master_all_actors",
    "final_exact_master_restoration_certificate",
    *(f"unscored_warmup_{index}_exact_master_slot_write" for index in range(4)),
    *(f"unscored_warmup_{index}_generation_all_actors" for index in range(4)),
    *(f"unscored_warmup_{index}_post_generation_integrity" for index in range(4)),
    *(f"scored_period_{index}_exact_master_slot_write" for index in range(4)),
    *(f"scored_period_{index}_generation_all_actors" for index in range(4)),
    *(f"scored_period_{index}_post_generation_integrity" for index in range(4)),
    *(f"scored_period_{index}_numeric_reduction" for index in range(4)),
    *(f"scored_period_{index}_complete" for index in range(4)),
})


@dataclass(frozen=True)
class SelfHashedSourceV65A:
    path: Path
    file_sha256: str
    content_sha256: str


@dataclass(frozen=True)
class FinalizerSourcesV65A:
    preregistration: SelfHashedSourceV65A
    attempt: SelfHashedSourceV65A
    evidence: SelfHashedSourceV65A
    analysis: SelfHashedSourceV65A
    report: SelfHashedSourceV65A
    gpu_log_path: Path
    gpu_log_file_sha256: str


def file_sha256_v65a(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256_v65a(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != SHA256_LENGTH_V65A
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"v65a invalid or unsealed SHA-256: {name}")
    return value


def _require_aware_timestamp_v65a(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"v65a invalid timestamp: {name}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RuntimeError(f"v65a invalid timestamp: {name}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError(f"v65a timezone-naive timestamp: {name}")
    return value


def _exact_int_v65a(value: object, expected: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == expected


def _exact_float_v65a(value: object, expected: float) -> bool:
    return isinstance(value, float) and math.isfinite(value) and value == expected


def _exact_int_list_v65a(value: object, expected: list[int]) -> bool:
    return (
        isinstance(value, list) and len(value) == len(expected)
        and all(_exact_int_v65a(item, target)
                for item, target in zip(value, expected, strict=True))
    )


def _verify_master_identity_v65a(value: object) -> dict:
    keys = {
        "schema", "sha256", "tensor_count", "elements", "bytes",
        "ordered_key_sha256", "tensors",
    }
    tensor_keys = {"key", "shape", "dtype", "elements", "sha256"}
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError("v65a canonical master identity schema changed")
    tensors = value.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != 70:
        raise RuntimeError("v65a canonical master tensor coverage changed")
    ordered_keys = []
    elements = 0
    for tensor in tensors:
        shape = tensor.get("shape") if isinstance(tensor, dict) else None
        count = tensor.get("elements") if isinstance(tensor, dict) else None
        key = tensor.get("key") if isinstance(tensor, dict) else None
        if (
            not isinstance(tensor, dict) or set(tensor) != tensor_keys
            or not isinstance(key, str)
            or re.fullmatch(
                r"base_model\.model\.[a-zA-Z0-9_.]+\.lora_[AB]\.weight",
                key,
            ) is None
            or not isinstance(shape, list) or not shape
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0
                   for item in shape)
            or isinstance(count, bool) or not isinstance(count, int)
            or count != math.prod(shape)
            or tensor.get("dtype") != "torch.float32"
        ):
            raise RuntimeError("v65a canonical master tensor receipt changed")
        _require_sha256_v65a(tensor.get("sha256"), "master tensor")
        ordered_keys.append(key)
        elements += count
    if (
        len(set(ordered_keys)) != 70
        or ordered_keys != sorted(ordered_keys)
        or value.get("schema") != "canonical-peft-fp32-state-v41a"
        or value.get("sha256") != design52.MASTER_SHA256_V52
        or value.get("sha256") != analysis.canonical_sha256_v65a(tensors)
        or not _exact_int_v65a(value.get("tensor_count"), 70)
        or not _exact_int_v65a(value.get("elements"), 4_528_128)
        or elements != 4_528_128
        or not _exact_int_v65a(value.get("bytes"), 18_112_512)
        or value.get("ordered_key_sha256")
        != design52.MASTER_ORDERED_KEY_SHA256_V52
        or value.get("ordered_key_sha256")
        != analysis.canonical_sha256_v65a(ordered_keys)
        or analysis.canonical_sha256_v65a(value)
        != MASTER_IDENTITY_OBJECT_SHA256_V65A
    ):
        raise RuntimeError("v65a canonical master identity changed")
    return value


def _verify_materialization_v65a(value: object, phase: str) -> dict:
    keys = {
        "schema", "phase", "adapter_id", "slot", "source_tensor_count",
        "source_elements", "runtime_module_count", "runtime_view_count",
        "runtime_elements", "runtime_dtype", "b_scale",
        "a_duplication_and_b_splitting_verified", "unique_parent_storage_count",
        "runtime_views_share_no_parent_storage",
        "slot_views_alias_parent_buffers", "storage_layout_sha256",
        "runtime_values_sha256",
    }
    if (
        not isinstance(value, dict) or set(value) != keys
        or value.get("schema") != "canonical-to-vllm-lora-materialization-v41a"
        or value.get("phase") != phase
        or not _exact_int_v65a(value.get("adapter_id"), 1)
        or not _exact_int_v65a(value.get("slot"), 0)
        or not _exact_int_v65a(value.get("source_tensor_count"), 70)
        or not _exact_int_v65a(value.get("source_elements"), 4_528_128)
        or not _exact_int_v65a(value.get("runtime_module_count"), 23)
        or not _exact_int_v65a(value.get("runtime_view_count"), 82)
        or not _exact_int_v65a(value.get("runtime_elements"), 4_921_344)
        or value.get("runtime_dtype") != "torch.bfloat16"
        or not _exact_float_v65a(value.get("b_scale"), 2.0)
        or value.get("a_duplication_and_b_splitting_verified") is not True
        or not _exact_int_v65a(value.get("unique_parent_storage_count"), 82)
        or value.get("runtime_views_share_no_parent_storage") is not True
        or value.get("slot_views_alias_parent_buffers") is not True
        or value.get("storage_layout_sha256")
        != MATERIALIZATION_STORAGE_LAYOUT_SHA256_V65A
        or value.get("runtime_values_sha256")
        != design52.MASTER_RUNTIME_SHA256_V52
    ):
        raise RuntimeError("v65a exact master materialization changed")
    return value


def _verify_base_check_v65a(value: object, phase: str) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) != {
            "phase", "unchanged", "tensor_count", "elements", "bytes",
            "inventory_sha256",
        }
        or value.get("phase") != phase or value.get("unchanged") is not True
        or not _exact_int_v65a(value.get("tensor_count"), 23)
        or not _exact_int_v65a(value.get("elements"), 142_999_552)
        or not _exact_int_v65a(value.get("bytes"), 285_999_104)
        or value.get("inventory_sha256") != BASE_INVENTORY_SHA256_V65A
    ):
        raise RuntimeError("v65a live base-layer identity changed")
    return value


def _verify_base_origin_v65a(value: object) -> dict:
    record_keys = {
        "runtime_module", "shape", "dtype", "elements", "sha256", "device",
        "stride", "storage_offset", "storage_bytes", "contiguous",
    }
    if (
        not isinstance(value, dict)
        or set(value) != {
            "tensor_count", "elements", "bytes", "inventory_sha256", "tensors",
        }
        or not _exact_int_v65a(value.get("tensor_count"), 23)
        or not _exact_int_v65a(value.get("elements"), 142_999_552)
        or not _exact_int_v65a(value.get("bytes"), 285_999_104)
        or value.get("inventory_sha256") != BASE_INVENTORY_SHA256_V65A
        or not isinstance(value.get("tensors"), list)
        or len(value["tensors"]) != 23
        or any(not isinstance(row, dict) or set(row) != record_keys
               for row in value["tensors"])
        or analysis.canonical_sha256_v65a(value["tensors"])
        != BASE_INVENTORY_SHA256_V65A
    ):
        raise RuntimeError("v65a base-origin inventory changed")
    return value


def _json_without_duplicate_keys_v65a(path: Path) -> object:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"v65a duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicates,
    )


def _read_self_hashed_v65a(source: SelfHashedSourceV65A) -> dict:
    if file_sha256_v65a(source.path) != source.file_sha256:
        raise RuntimeError(f"v65a finalizer input file changed: {source.path}")
    value = _json_without_duplicate_keys_v65a(source.path)
    if (
        not isinstance(value, dict)
        or value.get("content_sha256_before_self_field")
        != source.content_sha256
        or analysis.self_content_sha256_v65a(value) != source.content_sha256
    ):
        raise RuntimeError(f"v65a finalizer input content changed: {source.path}")
    return value


def _verify_predecessor_failed_attempt_v65a(prereg: dict) -> dict:
    """Independently bind the failed pre-generation launch and cleanup."""
    sources = {
        "predecessor_preregistration": SelfHashedSourceV65A(
            PREDECESSOR_PREREGISTRATION_V65A,
            PREDECESSOR_PREREGISTRATION_FILE_SHA256_V65A,
            PREDECESSOR_PREREGISTRATION_CONTENT_SHA256_V65A,
        ),
        "predecessor_ranking_panel": SelfHashedSourceV65A(
            PREDECESSOR_RANKING_PANEL_V65A,
            PREDECESSOR_RANKING_PANEL_FILE_SHA256_V65A,
            PREDECESSOR_RANKING_PANEL_CONTENT_SHA256_V65A,
        ),
        "predecessor_attempt": SelfHashedSourceV65A(
            PREDECESSOR_ATTEMPT_V65A,
            PREDECESSOR_ATTEMPT_FILE_SHA256_V65A,
            PREDECESSOR_ATTEMPT_CONTENT_SHA256_V65A,
        ),
        "predecessor_failure": SelfHashedSourceV65A(
            PREDECESSOR_FAILURE_V65A,
            PREDECESSOR_FAILURE_FILE_SHA256_V65A,
            PREDECESSOR_FAILURE_CONTENT_SHA256_V65A,
        ),
    }
    observed = {
        name: _read_self_hashed_v65a(source)
        for name, source in sources.items()
    }
    old_prereg = observed["predecessor_preregistration"]
    old_panel = observed["predecessor_ranking_panel"]
    old_attempt = observed["predecessor_attempt"]
    old_failure = observed["predecessor_failure"]
    old_artifacts = old_prereg.get("artifacts", {})
    cleanup = old_failure.get("cleanup", {})
    before = cleanup.get("before", [])
    after = cleanup.get("after", [])
    outcome_names = ("evidence", "analysis", "report", "gpu_log")
    if (
        old_prereg.get("schema")
        != "v65a-ranking64-alpha-zero-calibration-preregistration"
        or old_prereg.get("ranking_panel") != {
            "path": str(PREDECESSOR_RANKING_PANEL_V65A),
            "file_sha256": PREDECESSOR_RANKING_PANEL_FILE_SHA256_V65A,
            "content_sha256": PREDECESSOR_RANKING_PANEL_CONTENT_SHA256_V65A,
            "units": 64,
            "request_order_sha256": old_panel.get("request_order_sha256"),
            "unit_order_sha256": old_panel.get("unit_order_sha256"),
            "hash_only": True,
        }
        or old_panel.get("schema") != "v65-robust-sampling-ranking-panel"
        or not _exact_int_v65a(old_panel.get("ranking_units"), 64)
        or old_panel.get("question_answer_or_generation_text_persisted")
        is not False
        or old_panel.get("protected_semantics_opened") is not False
        or old_artifacts.get("attempt") != str(PREDECESSOR_ATTEMPT_V65A)
        or old_artifacts.get("failure") != str(PREDECESSOR_FAILURE_V65A)
        or any(Path(old_artifacts.get(name, "")).exists()
               for name in outcome_names)
        or old_attempt.get("schema")
        != "v65a-ranking64-alpha-zero-attempt"
        or old_attempt.get("status") != "launching_exact64_calibration_only"
        or old_attempt.get("phase")
        != "before_authorized_train_semantics_model_ray_or_gpu_load"
        or old_attempt.get("preregistration_file_sha256")
        != PREDECESSOR_PREREGISTRATION_FILE_SHA256_V65A
        or old_attempt.get("preregistration_content_sha256")
        != PREDECESSOR_PREREGISTRATION_CONTENT_SHA256_V65A
        or old_attempt.get("model_loaded_or_gpu_compute_started") is not False
        or old_attempt.get(
            "candidate_hpo_update_projection_or_protected_access"
        ) is not False
        or old_failure.get("schema")
        != "v65a-ranking64-alpha-zero-calibration-failure"
        or old_failure.get("type") != "RayTaskError(NotImplementedError)"
        or old_failure.get("message_sha256")
        != PREDECESSOR_FAILURE_MESSAGE_SHA256_V65A
        or old_failure.get("raw_error_message_or_traceback_persisted")
        is not False
        or old_failure.get(
            "raw_question_answer_prompt_or_generation_text_persisted"
        ) is not False
        or old_failure.get(
            "candidate_hpo_update_projection_or_promotion_performed"
        ) is not False
        or old_failure.get(
            "holdback_sentinel_reserve_ood_terminal_or_protected_opened"
        ) is not False
        or old_failure.get(
            "adaptive_retry_drop_reorder_or_early_stop_performed"
        ) is not False
        or not isinstance(before, list) or len(before) != 4
        or not isinstance(after, list) or len(after) != 4
        or any(not isinstance(item, dict) or item.get("state") != "CREATED"
               for item in before)
        or any(not isinstance(item, dict) or item.get("state") != "REMOVED"
               for item in after)
        or [item.get("placement_group_id") for item in before]
        != [item.get("placement_group_id") for item in after]
        or not _exact_int_v65a(cleanup.get("engine_kill_count"), 4)
        or not _exact_int_v65a(
            cleanup.get("placement_group_remove_count"), 4,
        )
        or cleanup.get("all_four_gcs_states_removed") is not True
        or old_failure.get("cleanup_errors") != []
        or old_failure.get(
            "ray_shutdown_attempted_even_without_complete_trainer"
        ) is not True
        or old_failure.get("final_gpu_idle")
        != {"all_four_compute_process_lists_empty": True}
    ):
        raise RuntimeError("v65a predecessor failure transcript changed")

    def binding(source: SelfHashedSourceV65A) -> dict:
        return {
            "path": str(source.path),
            "file_sha256": source.file_sha256,
            "content_sha256": source.content_sha256,
        }

    expected = {
        "schema": "v65a-r1-predecessor-failed-attempt-binding",
        **{name: binding(source) for name, source in sources.items()},
        "diagnosed_failure_boundary": {
            "type": "RayTaskError(NotImplementedError)",
            "message_sha256": PREDECESSOR_FAILURE_MESSAGE_SHA256_V65A,
            "model_actors_constructed_and_model_or_gpu_runtime_accessed": True,
            "attempt_model_loaded_flag_is_prelaunch_snapshot_only": True,
            "actor_runtime_identity_validation_completed": True,
            "failing_collective_rpc": "runtime_identity_v40a",
            "worker_method_absent_from_v65a_inheritance_chain": True,
            "gpu_monitor_started": False,
            "authorized_semantic_prefix_decoded": False,
            "generation_started": False,
            "scoring_or_numeric_outcome_observed": False,
        },
        "cleanup_receipt": {
            "engine_kill_count": 4,
            "placement_group_remove_count": 4,
            "all_four_gcs_states_removed": True,
            "all_before_states_created": True,
            "all_after_states_removed": True,
            "cleanup_errors_empty": True,
            "ray_shutdown_attempted": True,
            "final_four_gpu_idle": True,
        },
        "retry_contract": {
            "fresh_ranking_panel_path": str(
                ROOT / "experiments/eggroll_es_hpo/datasets/"
                "v65a_r1_ranking64_alpha_zero_panel.json"
            ),
            "fresh_preregistration_path": str(
                ROOT / "experiments/eggroll_es_hpo/preregistrations/"
                "ranking64_alpha_zero_calibration_v65a_r1.json"
            ),
            "fresh_attempt_path": str(
                ROOT / "experiments/eggroll_es_hpo/runs/"
                ".v65a_r1_ranking64_alpha_zero_calibration.attempt.json"
            ),
            "fresh_run_directory": str(
                ROOT / "experiments/eggroll_es_hpo/runs/"
                "v65a_r1_ranking64_alpha_zero_calibration"
            ),
            "failed_artifact_path_reused": False,
            "schedule_or_numeric_contract_changed": False,
            "candidate_hpo_update_projection_or_promotion_authorized": False,
            "v65_population_authorized": False,
        },
    }
    predecessor = prereg.get("predecessor_failed_attempt")
    if (
        predecessor != expected
        or analysis.canonical_sha256_v65a(predecessor)
        != analysis.canonical_sha256_v65a(expected)
    ):
        raise RuntimeError("v65a predecessor failed-attempt binding changed")
    return {
        "predecessor_artifacts_exact": True,
        "failure_before_semantic_decode_or_generation": True,
        "strict_cleanup_and_final_idle_exact": True,
        "fresh_retry_paths_only": True,
    }


def _verify_no_text_keys_v65a(name: str, value: object) -> dict:
    found = []

    def visit(item: object, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).lower() in analysis.FORBIDDEN_TEXT_KEYS_V65A:
                    found.append(f"{path}.{key}")
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")

    visit(value, name)
    if found:
        raise RuntimeError(f"v65a forbidden text-bearing key: {found[0]}")
    return {
        "source": name,
        "forbidden_text_key_count": 0,
        "forbidden_exact_keys": sorted(analysis.FORBIDDEN_TEXT_KEYS_V65A),
    }


def _verify_preregistration_v65a(
    prereg: dict, sources: FinalizerSourcesV65A,
) -> dict:
    predecessor_receipt = _verify_predecessor_failed_attempt_v65a(prereg)
    authorization = prereg.get("authorization", {})
    access = prereg.get("access_contract", {})
    recipe = prereg.get("fixed_calibration_recipe", {})
    numeric = prereg.get("numeric_analysis_contract", {})
    transfer = numeric.get("future_v65_null_bound_transfer", {})
    artifacts = prereg.get("artifacts", {})
    zero_authority = (
        "projection", "optimizer_update", "adapter_update", "candidate",
        "candidate_snapshot", "hpo_population", "train_holdback",
        "exact_sentinel", "unused_reserve", "ood_shadow",
        "protected_semantics", "terminal_holdout", "promotion",
    )
    expected_artifacts = {
        "attempt": sources.attempt.path,
        "evidence": sources.evidence.path,
        "analysis": sources.analysis.path,
        "report": sources.report.path,
        "gpu_log": sources.gpu_log_path,
    }
    bound_artifacts_exact = all(
        Path(artifacts.get(name, "")).resolve() == path.resolve()
        for name, path in expected_artifacts.items()
    )
    implementation = prereg.get("implementation_bindings", {})
    implementation_exact = bool(implementation) and all(
        isinstance(binding, dict)
        and set(binding) == {"path", "file_sha256"}
        and Path(binding["path"]).resolve().is_file()
        and file_sha256_v65a(Path(binding["path"]).resolve())
        == binding["file_sha256"]
        for binding in implementation.values()
    )
    expected_bootstrap_sha = hashlib.sha256(
        analysis.frozen_bootstrap_indices_v65a().astype(
            "<i8", copy=False,
        ).tobytes(order="C")
    ).hexdigest()
    source_evidence = prereg.get("source_evidence", {})
    sealed_panel, sealed_sources = builder.sealed_source_bindings_v65a()
    panel_binding = prereg.get("ranking_panel", {})
    panel_path = Path(panel_binding.get("path", "")).resolve()
    if not panel_path.is_file():
        raise RuntimeError("v65a sealed hash-only ranking panel is missing")
    observed_panel = _json_without_duplicate_keys_v65a(panel_path)
    sealed_panel_exact = (
        observed_panel == sealed_panel
        and analysis.canonical_sha256_v65a(observed_panel)
        == analysis.canonical_sha256_v65a(sealed_panel)
        and file_sha256_v65a(panel_path) == builder.payload_sha256_v65a(
            sealed_panel
        )
        and panel_binding == {
            "path": str(panel_path),
            "file_sha256": builder.payload_sha256_v65a(sealed_panel),
            "content_sha256": sealed_panel["content_sha256_before_self_field"],
            "units": 64,
            "request_order_sha256": sealed_panel["request_order_sha256"],
            "unit_order_sha256": sealed_panel["unit_order_sha256"],
            "hash_only": True,
        }
        and analysis.self_content_sha256_v65a(observed_panel)
        == sealed_panel["content_sha256_before_self_field"]
    )
    expected_implementation = builder.implementation_bindings_v65a()
    expected_artifacts_full = {
        "run_directory": str(sources.evidence.path.parent.resolve()),
        "attempt": str(sources.attempt.path.resolve()),
        "gpu_log": str(sources.gpu_log_path.resolve()),
        "evidence": str(sources.evidence.path.resolve()),
        "analysis": str(sources.analysis.path.resolve()),
        "report": str(sources.report.path.resolve()),
        "failure": str(
            (sources.evidence.path.parent / runtime.FAILURE.name).resolve()
        ),
    }
    expected_prereg = builder.build_preregistration_v65a(
        sealed_panel,
        sealed_sources,
        expected_implementation,
        ranking_panel_output=panel_path,
    )
    expected_prereg["artifacts"] = expected_artifacts_full
    expected_prereg["content_sha256_before_self_field"] = (
        analysis.self_content_sha256_v65a(expected_prereg)
    )
    integrity = prereg.get("required_integrity_gates", {})
    expected_integrity_keys = {
        "exact_idle_four_gpu_preflight",
        "exact_base_model_and_v434_byte_identities",
        "exact_first64_prefix_and_panel_order",
        "four_unscored_master_warmups_before_scored_block",
        "all_warmup_outputs_discarded_without_scoring_or_persistence",
        "all_four_engine_and_cache_receipts_exact",
        "all_four_scored_periods_complete_without_schedule_change",
        "all_periods_have_read_only_exact_slot_hash_before_and_after",
        "all_eight_paired_replicas_preserved_before_unit_bootstrap",
        "all_four_gpus_attributed_positive_each_generation_phase",
        "strict_four_engine_cleanup_and_final_idle",
        "adapter_source_and_stage_contract_reverified_unchanged_postcleanup",
        "numeric_hash_only_evidence",
        "row64plus_update_hpo_holdback_sentinel_ood_protected_terminal_access_zero",
    }
    if (
        prereg.get("schema")
        != "v65a-ranking64-alpha-zero-calibration-preregistration"
        or prereg.get("status")
        != "sealed_before_v65a_train_semantics_model_ray_or_gpu_access"
        or prereg.get("specific_v65a_exact64_alpha_zero_gpu_launch_authorized")
        is not True
        or prereg.get("prior_batch68_calibration_alone_authorizes_exact64_launch")
        is not False
        or authorization.get("gpu_launch") is not True
        or authorization.get("alpha_zero_calibration") is not True
        or authorization.get("physical_gpu_ids") != [0, 1, 2, 3]
        or authorization.get("actors") != 4
        or authorization.get("tensor_parallel_size_per_actor") != 1
        or any(authorization.get(key) is not False for key in zero_authority)
        or access.get("decode_exactly_first_64_v61c_ranking_rows") is not True
        or access.get("decode_v61c_row_64_or_later") is not False
        or access.get("ranking_prefix_bytes")
        != analysis.RANKING_PREFIX_BYTES_V65A
        or access.get("ranking_prefix_sha256")
        != analysis.RANKING_PREFIX_SHA256_V65A
        or access.get("source_file_size_metadata_bytes")
        != analysis.RANKING_SOURCE_FILE_SIZE_BYTES_V65A
        or access.get("live_authorized_prefix_pread_count") != 2
        or access.get("postrun_prefix_integrity_pread_decodes_semantics")
        is not False
        or access.get("full_jsonl_hash_verification_or_full_file_read_live")
        is not False
        or access.get(
            "full_train_membership_holdback_sentinel_ood_protected_or_"
            "terminal_may_open"
        ) is not False
        or access.get("raw_question_answer_prompt_or_generation_text_may_be_persisted")
        is not False
        or recipe.get("base_model") != builder.common65.base_model_binding_v65()
        or recipe.get("v434_adapter") != builder.common65.v434_binding_v65()
        or recipe.get("lora_request") != {
            "name": "v434_ranking64_alpha_zero_v65a",
            "integer_id": 1,
            "path": str(design52.STAGED_V52),
        }
        or recipe.get("reference_and_candidate_are_identical_v434_aliases")
        is not True
        or recipe.get("alpha") != 0.0
        or recipe.get("sigma_or_direction") is not None
        or recipe.get("rows_per_actor_call") != 64
        or recipe.get("ranking_units") != 64
        or recipe.get("exact_sentinel_units") != 0
        or recipe.get("physical_gpu_ids") != [0, 1, 2, 3]
        or recipe.get("actors") != 4
        or recipe.get("tensor_parallel_size_per_actor") != 1
        or recipe.get("unscored_warmup_periods") != 4
        or recipe.get("warmup_generation_completions_discarded") != 1024
        or recipe.get("warmup_outputs_scored_or_persisted") is not False
        or recipe.get("warmup_generation_metrics_computed_or_persisted")
        is not False
        or recipe.get("scored_periods") != 4
        or recipe.get("scored_label_plan") != analysis.LABEL_PLAN_V65A
        or recipe.get("pair_periods") != [list(pair) for pair in analysis.PAIR_PERIODS_V65A]
        or recipe.get("pairs_per_actor") != 2
        or recipe.get("paired_replicas_per_unit") != 8
        or recipe.get("scored_generation_completions") != 1024
        or recipe.get("total_generation_completions") != 2048
        or recipe.get("runtime_determinism_controls")
        != analysis.ENGINE_CONTROLS_V65A
        or recipe.get("adaptive_retry_drop_reorder_or_early_stop") is not False
        or recipe.get("adapter_update_candidate_hpo_or_promotion_performed")
        is not False
        or recipe.get("exact_master_rematerialization", {}).get(
            "period_slot_write_receipts_required"
        ) != 8
        or recipe.get("exact_master_rematerialization", {}).get(
            "read_only_live_slot_receipts_required"
        ) != 16
        or recipe.get("exact_master_rematerialization", {}).get(
            "read_only_edges_per_period"
        ) != ["before_generation", "after_generation"]
        or recipe.get("exact_master_rematerialization", {}).get(
            "after_generation_receipt_may_write_or_reset_slot"
        ) is not False
        or recipe.get("exact_master_rematerialization", {}).get(
            "canonical_fp32_master_sha256"
        ) != design52.MASTER_SHA256_V52
        or recipe.get("exact_master_rematerialization", {}).get(
            "bf16_runtime_values_sha256"
        ) != design52.MASTER_RUNTIME_SHA256_V52
        or recipe.get("exact_master_rematerialization", {}).get(
            "candidate_or_perturbation_materialized"
        ) is not False
        or numeric.get("within_unit_actor_pair_replicas_preserved_and_averaged")
        != 8
        or numeric.get("resampled_axis") != "conflict_unit_only"
        or numeric.get("single_replica_per_resampled_unit_sampling") is not False
        or numeric.get("bootstrap_replicates")
        != analysis.BOOTSTRAP_REPLICATES_V65A
        or numeric.get("bootstrap_seed") != analysis.BOOTSTRAP_SEED_V65A
        or numeric.get("bootstrap_index_matrix_sha256") != expected_bootstrap_sha
        or numeric.get("one_sided_alpha") != analysis.ONE_SIDED_ALPHA_V65A
        or numeric.get("joint_composite_weights")
        != analysis.COMPOSITE_WEIGHTS_V65A
        or transfer.get("outcome_independent_field_mapping")
        != analysis.FUTURE_V65_NULL_BOUND_TRANSFER_V65A
        or transfer.get("required_spread_gates") != {
            "pooled_joint_composite": "spread_strictly_greater_than_2*B_C",
            "each_pass_joint_composite": (
                "spread_strictly_greater_than_2*B_C_pass"
            ),
            "generated_f1_when_used": "spread_strictly_greater_than_2*B_F",
            "stability_when_used": "spread_strictly_greater_than_2*B_S",
            "stability_coefficient_when_gate_not_met": 0.0,
            "stability_gate_not_met_causes_population_failure": False,
        }
        or transfer.get("mapping_or_gates_may_change_after_observing_v65a")
        is not False
        or transfer.get(
            "rebind_or_launch_requires_required_alpha_zero_gate_passed"
        ) is not True
        or transfer.get(
            "failed_required_alpha_zero_gate_forbids_bound_rebinding_"
            "and_v65_launch"
        ) is not True
        or numeric.get("exact_sentinel_logic_present") is not False
        or numeric.get("success_authorizes_population_update_or_promotion")
        is not False
        or not sealed_panel_exact
        or source_evidence != sealed_sources
        or implementation != expected_implementation
        or artifacts != expected_artifacts_full
        or prereg != expected_prereg
        or analysis.canonical_sha256_v65a(prereg)
        != analysis.canonical_sha256_v65a(expected_prereg)
        or prereg.get("runtime") != design52.RUNTIME_V52
        or prereg.get("required_python") != str(design52.REQUIRED_PYTHON_V52)
        or prereg.get("implementation_closure_manifest_sha256")
        != analysis.canonical_sha256_v65a({
            key: binding["file_sha256"]
            for key, binding in sorted(implementation.items())
        })
        or set(integrity) != expected_integrity_keys
        or any(integrity.get(key) is not True for key in expected_integrity_keys)
        or not bound_artifacts_exact
        or not implementation_exact
        or prereg.get("raw_question_answer_prompt_or_generation_text_may_be_persisted")
        is not False
        or prereg.get("row_64_or_later_opened") is not False
        or prereg.get("adapter_update_candidate_hpo_or_promotion_performed")
        is not False
        or prereg.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v65a preregistration or sealed code contract changed")
    return {
        "specific_v65a_calibration_launch_was_preregistered": True,
        **predecessor_receipt,
        "implementation_bindings_exact": True,
        "implementation_binding_set_equals_current_builder_closure": True,
        "artifact_paths_exact": True,
        "entire_preregistration_equals_independent_rebuild": True,
        "sealed_hash_only_panel_file_content_and_order_exact": True,
        "sealed_source_evidence_independently_rebuilt_exact": True,
        "authorized_ranking_rows": 64,
        "row_64_or_later_authorized": False,
        "update_candidate_hpo_or_protected_access_authorized": False,
    }


def _base_model_expectation_and_receipt_v65a(receipt: object) -> tuple[dict, dict]:
    expectation = runtime64.base_model_artifact_expectation_v64()
    validated = runtime64.validate_base_model_artifact_receipt_v64(
        receipt, expectation,
    )
    return expectation, validated


def _verify_attempt_v65a(
    attempt: dict, prereg: dict, sources: FinalizerSourcesV65A,
) -> dict:
    expected_keys = {
        "schema", "status", "phase", "started_at_utc",
        "preregistration_file_sha256", "preregistration_content_sha256",
        "runtime_determinism_controls", "base_model_artifact_receipt",
        "preflight", "fixed_unscored_warmup_periods",
        "fixed_scored_periods", "fixed_paired_replicas_per_unit",
        "model_loaded_or_gpu_compute_started",
        "candidate_hpo_update_projection_or_protected_access",
        "content_sha256_before_self_field",
    }
    preflight = attempt.get("preflight", {})
    memory = preflight.get("memory_used_mib", {})
    _require_aware_timestamp_v65a(attempt.get("started_at_utc"), "attempt start")
    if (
        set(attempt) != expected_keys
        or attempt.get("schema") != "v65a-ranking64-alpha-zero-attempt"
        or attempt.get("status") != "launching_exact64_calibration_only"
        or attempt.get("phase")
        != "before_authorized_train_semantics_model_ray_or_gpu_load"
        or set(preflight) != {"compute_process_query_empty", "memory_used_mib"}
        or attempt.get("preregistration_file_sha256")
        != sources.preregistration.file_sha256
        or attempt.get("preregistration_content_sha256")
        != sources.preregistration.content_sha256
        or attempt.get("runtime_determinism_controls")
        != analysis.ENGINE_CONTROLS_V65A
        or attempt.get("fixed_unscored_warmup_periods") != 4
        or attempt.get("fixed_scored_periods") != 4
        or attempt.get("fixed_paired_replicas_per_unit") != 8
        or attempt.get("model_loaded_or_gpu_compute_started") is not False
        or attempt.get("candidate_hpo_update_projection_or_protected_access")
        is not False
        or preflight.get("compute_process_query_empty") is not True
        or set(memory) != {"0", "1", "2", "3"}
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            or value < 0 or value > 2048
            for value in memory.values()
        )
    ):
        raise RuntimeError("v65a launch attempt or idle preflight changed")
    expectation, receipt = _base_model_expectation_and_receipt_v65a(
        attempt.get("base_model_artifact_receipt")
    )
    return {
        "started_at_utc": attempt["started_at_utc"],
        "exact_idle_four_gpu_preflight": True,
        "model_or_gpu_not_loaded_when_attempt_was_persisted": True,
        "base_model_expectation_sha256": analysis.canonical_sha256_v65a(
            expectation
        ),
        "base_model_receipt_sha256": analysis.canonical_sha256_v65a(receipt),
    }


def _verify_input_receipt_v65a(receipt: object) -> dict:
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
        not isinstance(receipt, dict)
        or set(receipt) != expected_keys
        or receipt.get("schema")
        != "v65-exact-authorized-ranking-prefix-receipt"
        or Path(receipt.get("path", "")).resolve()
        != population65.V61C_ROWS.resolve()
        or receipt.get("source_full_file_sha256_bound_but_not_recomputed")
        != population65.V61C_ROWS_FILE_SHA256
        or not _exact_int_v65a(
            receipt.get("authorized_prefix_bytes"),
            analysis.RANKING_PREFIX_BYTES_V65A,
        )
        or receipt.get("authorized_prefix_sha256")
        != analysis.RANKING_PREFIX_SHA256_V65A
        or not _exact_int_v65a(receipt.get("decoded_ranking_rows"), 64)
        or receipt.get("requested_byte_offset_at_or_after_prefix") is not False
        or not _exact_int_v65a(
            receipt.get("remaining_exact_sentinel_rows_decoded"), 0,
        )
        or receipt.get("question_answer_or_text_persisted") is not False
        or not _exact_int_v65a(receipt.get("request_count"), 64)
        or not _require_sha256_v65a(
            receipt.get("request_prompt_token_ids_sha256"),
            "request prompt-token identity",
        )
        or not _exact_int_v65a(
            receipt.get("generation_seed"), analysis.COMMON_GENERATION_SEED_V65A,
        )
        or not _exact_float_v65a(receipt.get("generation_temperature"), 0.0)
        or not _exact_int_v65a(
            receipt.get("generation_max_tokens"),
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V65A["max_tokens"],
        )
        or not _exact_int_v65a(receipt.get("submitted_request_batch_size"), 64)
        or receipt.get("runtime_determinism_controls")
        != analysis.ENGINE_CONTROLS_V65A
        or receipt.get("lora_adapter_request_name")
        != "v434_ranking64_alpha_zero_v65a"
        or not _exact_int_v65a(receipt.get("lora_adapter_request_id"), 1)
        or Path(receipt.get("lora_adapter_request_path", "")).resolve()
        != design52.STAGED_V52.resolve()
    ):
        raise RuntimeError("v65a exact authorized prefix input receipt changed")
    return {
        "authorized_prefix_bytes": analysis.RANKING_PREFIX_BYTES_V65A,
        "authorized_prefix_sha256": analysis.RANKING_PREFIX_SHA256_V65A,
        "decoded_ranking_rows": 64,
        "row_64_or_later_requested": False,
        "full_file_hash_recomputed": False,
    }


def _verify_actor_and_lora_receipts_v65a(
    evidence: dict, prereg: dict,
) -> tuple[dict, dict[int, int]]:
    actors = evidence.get("actor_runtime_identities", [])
    workers = evidence.get("worker_runtime_identities", [])
    active = evidence.get("active_lora_receipts", [])
    actor_keys = {
        "schema", "pid", "physical_gpu_id", "cuda_visible_devices",
        "cuda_current_device", "runtime_determinism_controls",
        "scheduler_class", "tuned_folder", "tuned_table_content_sha256",
        "submitted_request_batch_size", "generation_only",
        "global_batch_invariance_claimed",
    }
    worker_keys = {
        "schema", "pid", "cuda_visible_devices", "cuda_current_device",
    }
    by_gpu: dict[int, int] = {}
    pids = set()
    expected_tuned = prereg.get("runtime", {}).get(
        "tuned_table_content_sha256"
    )
    _require_sha256_v65a(expected_tuned, "tuned table content")
    for item in actors:
        gpu = item.get("physical_gpu_id") if isinstance(item, dict) else None
        pid = item.get("pid") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or set(item) != actor_keys
            or item.get("schema")
            != "ranking64-alpha-zero-actor-identity-v65a"
            or isinstance(gpu, bool) or not isinstance(gpu, int)
            or gpu not in range(4)
            or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
            or gpu in by_gpu or pid in pids
            or item.get("cuda_visible_devices") != str(gpu)
            or not _exact_int_v65a(item.get("cuda_current_device"), 0)
            or item.get("runtime_determinism_controls")
            != analysis.ENGINE_CONTROLS_V65A
            or item.get("scheduler_class") != "Scheduler"
            or item.get("tuned_folder") != str(design52.RUNTIME_V52["tuned_folder"])
            or item.get("tuned_table_content_sha256") != expected_tuned
            or not _exact_int_v65a(item.get("submitted_request_batch_size"), 64)
            or item.get("generation_only") is not True
            or item.get("global_batch_invariance_claimed") is not False
        ):
            raise RuntimeError("v65a live actor engine receipt changed")
        by_gpu[gpu] = pid
        pids.add(pid)
    worker_by_pid = {}
    for item in workers:
        pid = item.get("pid") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or set(item) != worker_keys
            or item.get("schema") != "lora-topology-worker-identity-v40a"
            or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
            or pid in worker_by_pid
            or not _exact_int_v65a(item.get("cuda_current_device"), 0)
        ):
            raise RuntimeError("v65a live worker identity receipt changed")
        worker_by_pid[pid] = item
    if (
        len(actors) != 4 or len(workers) != 4
        or set(by_gpu) != {0, 1, 2, 3}
        or set(worker_by_pid) != pids
        or any(
            worker_by_pid[pid].get("cuda_visible_devices") != str(gpu)
            for gpu, pid in by_gpu.items()
        )
    ):
        raise RuntimeError("v65a actor/worker GPU mapping changed")

    active_keys = {
        "schema", "expected_lora_int_id", "active_lora_ids",
        "active_manager_cache_lora_ids", "loaded_cpu_cache_lora_ids",
        "facade_type", "manager_type", "active_slot_index", "max_loras",
        "max_cpu_loras", "max_lora_rank", "staged_v434_applied_receipt",
        "extra_or_candidate_adapter_loaded",
    }
    applied_keys = {
        "schema", "expected_lora_int_id", "active_lora_ids",
        "active_manager_cache_lora_ids", "loaded_cpu_cache_lora_ids",
        "active_slot_index", "facade_type", "manager_type",
        "staged_weights_file_sha256", "canonical_fp32_state_sha256",
        "canonical_ordered_key_sha256", "canonical_tensor_count",
        "canonical_elements", "registered_lora_module_count",
        "matched_live_lora_module_count",
        "unmatched_registered_lora_module_count",
        "runtime_module_manifest_sha256", "source_linked_runtime_view_count",
        "source_linked_runtime_elements", "source_linked_runtime_dtype",
        "source_linked_runtime_values_sha256", "registered_slot_view_count",
        "registered_slot_records_sha256",
        "exact_staged_fp32_to_gpu_slot_equality",
        "exact_registered_postpack_to_gpu_slot_equality",
        "active_matches_expected", "max_loras", "max_cpu_loras",
    }
    if (
        not isinstance(active, list) or len(active) != 4
        or any(
            not isinstance(item, dict)
            or set(item) != active_keys
            or item.get("schema") != "v65a-effective-active-lora-receipt"
            or not _exact_int_v65a(item.get("expected_lora_int_id"), 1)
            or not _exact_int_list_v65a(item.get("active_lora_ids"), [1])
            or not _exact_int_list_v65a(
                item.get("active_manager_cache_lora_ids"), [1]
            )
            or not _exact_int_list_v65a(
                item.get("loaded_cpu_cache_lora_ids"), [1]
            )
            or item.get("facade_type") != "LRUCacheWorkerLoRAManager"
            or item.get("manager_type") != "LRUCacheLoRAModelManager"
            or not _exact_int_v65a(item.get("active_slot_index"), 0)
            or not _exact_int_v65a(item.get("max_loras"), 1)
            or not _exact_int_v65a(item.get("max_cpu_loras"), 2)
            or not _exact_int_v65a(item.get("max_lora_rank"), 32)
            or not isinstance(item.get("staged_v434_applied_receipt"), dict)
            or set(item["staged_v434_applied_receipt"]) != applied_keys
            or item["staged_v434_applied_receipt"].get("schema")
            != "v64-effective-applied-lora-receipt"
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get("expected_lora_int_id"),
                1,
            )
            or not _exact_int_list_v65a(
                item["staged_v434_applied_receipt"].get("active_lora_ids"), [1]
            )
            or not _exact_int_list_v65a(
                item["staged_v434_applied_receipt"].get(
                    "active_manager_cache_lora_ids"
                ), [1]
            )
            or not _exact_int_list_v65a(
                item["staged_v434_applied_receipt"].get(
                    "loaded_cpu_cache_lora_ids"
                ), [1]
            )
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get("active_slot_index"), 0
            )
            or item["staged_v434_applied_receipt"].get("facade_type")
            != "LRUCacheWorkerLoRAManager"
            or item["staged_v434_applied_receipt"].get("manager_type")
            != "LRUCacheLoRAModelManager"
            or item["staged_v434_applied_receipt"].get(
                "staged_weights_file_sha256"
            ) != design52.STAGED_WEIGHTS_SHA256_V52
            or item["staged_v434_applied_receipt"].get(
                "canonical_fp32_state_sha256"
            ) != design52.MASTER_SHA256_V52
            or item["staged_v434_applied_receipt"].get(
                "canonical_ordered_key_sha256"
            ) != design52.MASTER_ORDERED_KEY_SHA256_V52
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get("canonical_tensor_count"),
                70,
            )
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get("canonical_elements"),
                4_528_128,
            )
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get(
                    "registered_lora_module_count"
                ), 23,
            )
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get(
                    "matched_live_lora_module_count"
                ), 23,
            )
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get(
                    "unmatched_registered_lora_module_count"
                ), 0,
            )
            or item["staged_v434_applied_receipt"].get(
                "runtime_module_manifest_sha256"
            ) != RUNTIME_MODULE_MANIFEST_SHA256_V65A
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get(
                    "source_linked_runtime_view_count"
                ), 82,
            )
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get(
                    "source_linked_runtime_elements"
                ), 4_921_344,
            )
            or item["staged_v434_applied_receipt"].get(
                "source_linked_runtime_dtype"
            ) != "torch.bfloat16"
            or item["staged_v434_applied_receipt"].get(
                "source_linked_runtime_values_sha256"
            ) != design52.MASTER_RUNTIME_SHA256_V52
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get(
                    "registered_slot_view_count"
                ), 82,
            )
            or item["staged_v434_applied_receipt"].get(
                "registered_slot_records_sha256"
            ) != REGISTERED_SLOT_RECORDS_SHA256_V65A
            or item["staged_v434_applied_receipt"].get(
                "exact_staged_fp32_to_gpu_slot_equality"
            ) is not True
            or item["staged_v434_applied_receipt"].get(
                "exact_registered_postpack_to_gpu_slot_equality"
            ) is not True
            or item["staged_v434_applied_receipt"].get("active_matches_expected")
            is not True
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get("max_loras"), 1,
            )
            or not _exact_int_v65a(
                item["staged_v434_applied_receipt"].get("max_cpu_loras"), 2,
            )
            or item.get("extra_or_candidate_adapter_loaded") is not False
            for item in active
        )
        or evidence.get("active_lora_receipts_sha256")
        != analysis.canonical_sha256_v65a(active)
    ):
        raise RuntimeError("v65a active LoRA cache/slot receipts changed")
    return ({
        "actors": 4,
        "unique_processes": 4,
        "physical_gpu_ids": [0, 1, 2, 3],
        "all_live_engine_controls_exact": True,
        "active_lora_ids_all_actors": [1],
        "manager_and_cpu_cache_lora_ids_all_actors": [1],
        "extra_or_candidate_adapter_loaded": False,
    }, by_gpu)


def _verify_adapter_artifact_contract_v65a(evidence: dict) -> dict:
    prelaunch = evidence.get("adapter_artifact_contract_prelaunch")
    postcleanup = evidence.get("adapter_artifact_contract_postcleanup")
    # This independently re-hashes the source/staged adapter artifacts now;
    # it does not trust the run's pre-launch receipt or open model/dataset data.
    expected = runtime52.verify_adapter_contract_v52()
    if (
        prelaunch != expected
        or postcleanup != expected
        or prelaunch != postcleanup
        or analysis.canonical_sha256_v65a(prelaunch)
        != analysis.canonical_sha256_v65a(expected)
        or analysis.canonical_sha256_v65a(postcleanup)
        != analysis.canonical_sha256_v65a(expected)
        or evidence.get("adapter_artifact_contract_unchanged") is not True
    ):
        raise RuntimeError("v65a source/staged adapter artifact contract changed")
    return {
        "source_and_staged_file_hashes_exact": True,
        "prelaunch_postcleanup_and_independently_recomputed_contract_equal": True,
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
    }


def _verify_initial_installations_v65a(evidence: dict, master: dict) -> dict:
    installations = evidence.get("initial_installations", [])
    expected_keys = {
        "schema", "installed", "adapter_id", "slot",
        "source_weights_sha256", "source_config_sha256",
        "canonical_identity", "assignment_count", "assignment_sha256",
        "assignments", "materialization", "base_identity",
        "base_origin_inventory", "zero_zero_degeneracy_guard",
    }
    if not isinstance(installations, list) or len(installations) != 4:
        raise RuntimeError("v65a initial V434 installation coverage changed")
    for item in installations:
        identity = item.get("canonical_identity", {}) if isinstance(item, dict) else {}
        materialization = item.get("materialization", {}) if isinstance(item, dict) else {}
        assignments = item.get("assignments", []) if isinstance(item, dict) else []
        guard = item.get("zero_zero_degeneracy_guard", {}) if isinstance(item, dict) else {}
        _verify_master_identity_v65a(identity)
        _verify_materialization_v65a(materialization, "install")
        _verify_base_check_v65a(
            item.get("base_identity") if isinstance(item, dict) else None,
            "install",
        )
        _verify_base_origin_v65a(
            item.get("base_origin_inventory") if isinstance(item, dict) else None
        )
        if (
            not isinstance(item, dict)
            or set(item) != expected_keys
            or item.get("schema") != "canonical-lora-adapter-installed-v41a"
            or item.get("installed") is not True
            or not _exact_int_v65a(item.get("adapter_id"), 1)
            or not _exact_int_v65a(item.get("slot"), 0)
            or item.get("source_weights_sha256")
            != design52.SOURCE_WEIGHTS_SHA256_V52
            or item.get("source_config_sha256")
            != design52.SOURCE_CONFIG_SHA256_V52
            or analysis.canonical_sha256_v65a(identity)
            != MASTER_IDENTITY_OBJECT_SHA256_V65A
            or analysis.canonical_sha256_v65a(identity)
            != master["canonical_master_identity_sha256"]
            or not _exact_int_v65a(item.get("assignment_count"), 82)
            or not isinstance(assignments, list) or len(assignments) != 82
            or item.get("assignment_sha256")
            != ASSIGNMENT_SHA256_V65A
            or analysis.canonical_sha256_v65a(assignments)
            != ASSIGNMENT_SHA256_V65A
            or any(
                not isinstance(assignment, dict)
                or set(assignment) != {
                    "peft_key", "side", "runtime_module", "slot",
                    "slice_index", "segment_index", "segment_count",
                    "source_shape", "runtime_shape",
                }
                for assignment in assignments
            )
            or guard != {
                "all_a_zero": False,
                "all_b_zero": False,
                "simultaneous_all_zero_forbidden": True,
            }
        ):
            raise RuntimeError("v65a initial exact V434 installation changed")
    return {
        "actor_installations": 4,
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
        "simultaneous_zero_zero_forbidden": True,
    }


def _verify_slot_writes_v65a(evidence: dict, master: dict) -> dict:
    receipts = evidence.get("exact_master_slot_write_receipts", [])
    plan = [
        *(('unscored_warmup', index) for index in range(4)),
        *(('scored', index) for index in range(4)),
    ]
    top_keys = {
        "period_kind", "period_index", "pre_write_master",
        "post_write_master", "actors", "actor_receipts_sha256",
    }
    actor_keys = {
        "schema", "period_kind", "period_index", "master_identity",
        "materialization", "base_identity", "transaction_state_quiescent",
        "timing",
    }
    if not isinstance(receipts, list) or len(receipts) != 8:
        raise RuntimeError("v65a exact-master slot-write coverage changed")
    for receipt, (kind, index) in zip(receipts, plan, strict=True):
        actors = receipt.get("actors", []) if isinstance(receipt, dict) else []
        if (
            not isinstance(receipt, dict)
            or set(receipt) != top_keys
            or receipt.get("period_kind") != kind
            or not _exact_int_v65a(receipt.get("period_index"), index)
            or receipt.get("pre_write_master") != master
            or receipt.get("post_write_master") != master
            or not isinstance(actors, list) or len(actors) != 4
            or receipt.get("actor_receipts_sha256")
            != analysis.canonical_sha256_v65a(actors)
        ):
            raise RuntimeError("v65a exact-master slot-write receipt changed")
        for actor in actors:
            identity = actor.get("master_identity", {}) if isinstance(actor, dict) else {}
            materialization = actor.get("materialization", {}) if isinstance(actor, dict) else {}
            timing = actor.get("timing", {}) if isinstance(actor, dict) else {}
            started = timing.get("started_ns")
            ended = timing.get("ended_ns")
            elapsed = timing.get("elapsed_ns")
            _verify_master_identity_v65a(identity)
            _verify_materialization_v65a(
                materialization, "v65a_exact_master_slot_write",
            )
            _verify_base_check_v65a(
                actor.get("base_identity") if isinstance(actor, dict) else None,
                "v65a_exact_master_slot_write",
            )
            if (
                not isinstance(actor, dict)
                or set(actor) != actor_keys
                or actor.get("schema") != "exact-master-slot-write-v65a"
                or actor.get("period_kind") != kind
                or not _exact_int_v65a(actor.get("period_index"), index)
                or analysis.canonical_sha256_v65a(identity)
                != master["canonical_master_identity_sha256"]
                or actor.get("transaction_state_quiescent") is not True
                or set(timing) != {"clock", "started_ns", "ended_ns", "elapsed_ns"}
                or timing.get("clock") != "worker_monotonic_ns"
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (started, ended, elapsed)
                )
                or started < 0 or ended < started or elapsed != ended - started
            ):
                raise RuntimeError("v65a actor exact-master slot-write changed")
    if evidence.get("exact_master_slot_write_receipts_sha256") != (
        analysis.canonical_sha256_v65a(receipts)
    ):
        raise RuntimeError("v65a exact-master slot-write inventory hash changed")
    return {
        "period_slot_write_receipts": 8,
        "actor_slot_write_receipts": 32,
        "all_writes_source_unchanged_pinned_fp32_master": True,
        "all_transactions_quiescent": True,
    }


def _verify_read_only_slot_edges_v65a(
    evidence: dict, installed_master: dict,
) -> tuple[dict, dict]:
    receipts = evidence.get("read_only_live_slot_receipts", [])
    plan = []
    for kind in ("unscored_warmup", "scored"):
        for index in range(4):
            plan.extend((
                (kind, index, "before_generation"),
                (kind, index, "after_generation"),
            ))
    receipt_keys = {
        "period_kind", "period_index", "edge", "aggregate", "actors",
        "actor_receipts_sha256",
    }
    aggregate_keys = {
        "schema", "canonical_fp32_master_sha256",
        "canonical_master_identity_sha256", "bf16_runtime_values_sha256",
        "runtime_view_count_per_actor", "runtime_elements_per_actor",
        "runtime_dtype", "base_inventory_sha256",
        "four_actor_exact_read_only_consensus",
    }
    actor_keys = {
        "schema", "period_kind", "period_index", "edge", "master_identity",
        "runtime_view_count", "runtime_elements", "runtime_dtype",
        "runtime_values_sha256", "active_lora_ids",
        "active_manager_cache_lora_ids", "base_identity",
        "transaction_state_quiescent", "slot_read_only_no_weight_write_or_reset",
        "timing",
    }
    if not isinstance(receipts, list) or len(receipts) != 16:
        raise RuntimeError("v65a read-only live-slot edge coverage changed")
    consensus = None
    for receipt, (kind, index, edge) in zip(receipts, plan, strict=True):
        aggregate = receipt.get("aggregate", {}) if isinstance(receipt, dict) else {}
        actors = receipt.get("actors", []) if isinstance(receipt, dict) else []
        if (
            not isinstance(receipt, dict)
            or set(receipt) != receipt_keys
            or receipt.get("period_kind") != kind
            or not _exact_int_v65a(receipt.get("period_index"), index)
            or receipt.get("edge") != edge
            or not isinstance(aggregate, dict)
            or set(aggregate) != aggregate_keys
            or aggregate.get("schema")
            != "v65a-read-only-four-actor-master-slot-consensus"
            or aggregate.get("canonical_fp32_master_sha256")
            != design52.MASTER_SHA256_V52
            or aggregate.get("canonical_master_identity_sha256")
            != installed_master["canonical_master_identity_sha256"]
            or aggregate.get("bf16_runtime_values_sha256")
            != design52.MASTER_RUNTIME_SHA256_V52
            or aggregate.get("runtime_view_count_per_actor") != 82
            or aggregate.get("runtime_elements_per_actor") != 4_921_344
            or aggregate.get("runtime_dtype") != "torch.bfloat16"
            or not _require_sha256_v65a(
                aggregate.get("base_inventory_sha256"), "base inventory",
            )
            or aggregate.get("four_actor_exact_read_only_consensus") is not True
            or not isinstance(actors, list) or len(actors) != 4
            or receipt.get("actor_receipts_sha256")
            != analysis.canonical_sha256_v65a(actors)
        ):
            raise RuntimeError("v65a read-only four-actor slot consensus changed")
        if consensus is None:
            consensus = aggregate
        elif aggregate != consensus:
            raise RuntimeError("v65a live-slot consensus changed between edges")
        base_inventories = set()
        master_identities = set()
        for actor in actors:
            identity = actor.get("master_identity", {}) if isinstance(actor, dict) else {}
            base = actor.get("base_identity", {}) if isinstance(actor, dict) else {}
            timing = actor.get("timing", {}) if isinstance(actor, dict) else {}
            started = timing.get("started_ns")
            ended = timing.get("ended_ns")
            elapsed = timing.get("elapsed_ns")
            _verify_master_identity_v65a(identity)
            _verify_base_check_v65a(
                base, "v65a_read_only_slot_receipt",
            )
            if (
                not isinstance(actor, dict)
                or set(actor) != actor_keys
                or actor.get("schema") != "read-only-exact-master-slot-v65a"
                or actor.get("period_kind") != kind
                or not _exact_int_v65a(actor.get("period_index"), index)
                or actor.get("edge") != edge
                or analysis.canonical_sha256_v65a(identity)
                != aggregate["canonical_master_identity_sha256"]
                or actor.get("runtime_view_count") != 82
                or actor.get("runtime_elements") != 4_921_344
                or actor.get("runtime_dtype") != "torch.bfloat16"
                or actor.get("runtime_values_sha256")
                != design52.MASTER_RUNTIME_SHA256_V52
                or not _exact_int_list_v65a(actor.get("active_lora_ids"), [1])
                or not _exact_int_list_v65a(
                    actor.get("active_manager_cache_lora_ids"), [1]
                )
                or not isinstance(base, dict)
                or base.get("inventory_sha256")
                != aggregate["base_inventory_sha256"]
                or actor.get("transaction_state_quiescent") is not True
                or actor.get("slot_read_only_no_weight_write_or_reset") is not True
                or set(timing)
                != {"clock", "started_ns", "ended_ns", "elapsed_ns"}
                or timing.get("clock") != "worker_monotonic_ns"
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in (started, ended, elapsed)
                )
                or started < 0 or ended < started or elapsed != ended - started
            ):
                raise RuntimeError("v65a read-only actor live-slot receipt changed")
            base_inventories.add(base["inventory_sha256"])
            master_identities.add(analysis.canonical_sha256_v65a(identity))
        if len(base_inventories) != 1 or len(master_identities) != 1:
            raise RuntimeError("v65a read-only actor consensus changed")
    if evidence.get("read_only_live_slot_receipts_sha256") != (
        analysis.canonical_sha256_v65a(receipts)
    ):
        raise RuntimeError("v65a read-only live-slot inventory hash changed")
    assert consensus is not None
    return ({
        "period_edge_receipts": 16,
        "actor_read_only_receipts": 64,
        "before_and_after_generation_edges_per_period": 2,
        "all_receipts_exact_master_and_quiescent": True,
        "all_after_generation_receipts_read_only": True,
        "weight_write_or_slot_reset_during_edge_receipt": False,
    }, consensus)


def _verify_state_and_schedule_v65a(
    evidence: dict, prereg: dict,
) -> tuple[dict, list]:
    warmup = evidence.get("warmup_state_receipts", [])
    scored_receipts = evidence.get("scored_state_receipts", [])
    master = evidence.get("installed_master_state", {})
    final_master = evidence.get("final_restored_master_state", {})
    expected_master_keys = {
        "canonical_fp32_master_sha256", "canonical_master_identity_sha256",
        "four_actor_certificate_sha256", "bf16_runtime_values_sha256",
    }
    if (
        not isinstance(master, dict)
        or set(master) != expected_master_keys
        or master.get("canonical_fp32_master_sha256")
        != design52.MASTER_SHA256_V52
        or master.get("bf16_runtime_values_sha256")
        != design52.MASTER_RUNTIME_SHA256_V52
        or final_master != master
        or any(
            not _require_sha256_v65a(master.get(key), f"master {key}")
            for key in expected_master_keys
        )
    ):
        raise RuntimeError("v65a installed/final exact-master certificate changed")
    read_only, live_consensus = _verify_read_only_slot_edges_v65a(
        evidence, master,
    )
    analysis.validate_state_receipts_v65a(
        warmup, scored_receipts, live_consensus,
    )
    five_keys = {
        "period_kind", "period_index", "before", "after",
        "identical_v434_state",
    }
    if (
        any(set(item) != five_keys for item in [*warmup, *scored_receipts])
        or any(
            not _exact_int_v65a(item.get("period_index"), index)
            for rows in (warmup, scored_receipts)
            for index, item in enumerate(rows)
        )
        or any(
            item.get("before") != live_consensus
            or item.get("after") != live_consensus
            for item in [*warmup, *scored_receipts]
        )
        or evidence.get("warmup_state_receipts_sha256")
        != analysis.canonical_sha256_v65a(warmup)
        or evidence.get("scored_state_receipts_sha256")
        != analysis.canonical_sha256_v65a(scored_receipts)
    ):
        raise RuntimeError("v65a exact five-key state receipt changed")
    installation = _verify_initial_installations_v65a(evidence, master)
    slot_writes = _verify_slot_writes_v65a(evidence, master)

    scored_periods = evidence.get("scored_periods")
    numeric = analysis.validate_scored_periods_v65a(scored_periods)
    if (
        numeric.shape != (64, 4, 4, 3)
        or evidence.get("numeric_scored_periods_sha256")
        != analysis.canonical_sha256_v65a(scored_periods)
    ):
        raise RuntimeError("v65a exact numeric scored-period inventory changed")
    identities = [
        (item["row_sha256"], item["unit_identity_sha256"])
        for item in scored_periods[0][0]
    ]
    request_order = analysis.canonical_sha256_v65a(
        [row for row, _unit in identities]
    )
    unit_order = analysis.canonical_sha256_v65a(
        [unit for _row, unit in identities]
    )
    panel = prereg.get("ranking_panel", {})
    if (
        len(identities) != 64
        or len({row for row, _unit in identities}) != 64
        or len({unit for _row, unit in identities}) != 64
        or request_order != panel.get("request_order_sha256")
        or unit_order != panel.get("unit_order_sha256")
        or evidence.get("panel_content_sha256") != panel.get("content_sha256")
        or evidence.get("row_count") != 64
        or evidence.get("actor_count") != 4
        or evidence.get("unscored_warmup_period_count") != 4
        or evidence.get("scored_period_count") != 4
        or evidence.get("paired_replicas_per_unit") != 8
        or evidence.get("label_plan") != analysis.LABEL_PLAN_V65A
        or evidence.get("pair_periods")
        != [list(pair) for pair in analysis.PAIR_PERIODS_V65A]
        or evidence.get("warmup_generation_completions_discarded") != 1024
        or evidence.get("scored_generation_completions") != 1024
        or evidence.get("total_generation_completions") != 2048
        or evidence.get("generation_only") is not True
        or evidence.get("warmup_raw_outputs_persisted") is not False
        or evidence.get("warmup_generation_metrics_computed_or_persisted")
        is not False
        or evidence.get("adaptive_retry_drop_reorder_or_early_stop_performed")
        is not False
        or not _exact_float_v65a(evidence.get("alpha"), 0.0)
        or evidence.get("sigma_or_direction") is not None
    ):
        raise RuntimeError("v65a fixed warmup/scored schedule or panel changed")
    return ({
        "ranking_rows": 64,
        "unique_row_receipts": 64,
        "unique_conflict_unit_receipts": 64,
        "unscored_warmup_periods": 4,
        "scored_periods": 4,
        "paired_replicas_per_unit": 8,
        "exact_five_key_state_receipts": 8,
        "warmup_generation_completions_discarded": 1024,
        "scored_generation_completions": 1024,
        "total_generation_completions": 2048,
        "request_order_sha256": request_order,
        "unit_order_sha256": unit_order,
        "initial_exact_v434_installation": installation,
        "exact_master_rematerialization": slot_writes,
        "read_only_live_slot_edges": read_only,
        "final_mutating_master_certificate_after_all_periods": True,
        "final_restored_master_state_equals_initial_installed_state": True,
    }, scored_periods)


def _verify_evidence_v65a(
    evidence: dict, prereg: dict,
) -> tuple[dict, list, dict[int, int]]:
    expected_keys = {
        "schema", "status", "panel_content_sha256",
        "authorized_input_receipt", "canonical_fp32_master_sha256",
        "bf16_runtime_values_sha256", "row_count", "actor_count",
        "unscored_warmup_period_count", "scored_period_count",
        "paired_replicas_per_unit", "label_plan", "pair_periods",
        "runtime_determinism_controls", "actor_runtime_identities",
        "worker_runtime_identities", "active_lora_receipts",
        "active_lora_receipts_sha256", "initial_installations",
        "adapter_artifact_contract_prelaunch",
        "adapter_artifact_contract_postcleanup",
        "adapter_artifact_contract_unchanged", "installed_master_state",
        "final_restored_master_state", "warmup_state_receipts",
        "warmup_state_receipts_sha256", "scored_state_receipts",
        "scored_state_receipts_sha256", "exact_master_slot_write_receipts",
        "exact_master_slot_write_receipts_sha256",
        "read_only_live_slot_receipts",
        "read_only_live_slot_receipts_sha256", "scored_periods",
        "numeric_scored_periods_sha256",
        "warmup_generation_completions_discarded",
        "scored_generation_completions", "total_generation_completions",
        "generation_only", "warmup_raw_outputs_persisted",
        "warmup_generation_metrics_computed_or_persisted",
        "adaptive_retry_drop_reorder_or_early_stop_performed", "alpha",
        "sigma_or_direction",
        "adapter_update_candidate_hpo_or_projection_performed",
        "holdback_sentinel_reserve_ood_terminal_or_protected_opened",
        "raw_question_answer_prompt_or_generation_text_persisted",
        "content_sha256_before_self_field",
    }
    if (
        set(evidence) != expected_keys
        or evidence.get("schema")
        != "v65a-ranking64-alpha-zero-generation-evidence"
        or evidence.get("status")
        != "complete_fixed_warmup_and_scored_exact64_characterization"
        or evidence.get("canonical_fp32_master_sha256")
        != design52.MASTER_SHA256_V52
        or evidence.get("bf16_runtime_values_sha256")
        != design52.MASTER_RUNTIME_SHA256_V52
        or evidence.get("runtime_determinism_controls")
        != analysis.ENGINE_CONTROLS_V65A
        or evidence.get("adapter_update_candidate_hpo_or_projection_performed")
        is not False
        or evidence.get(
            "holdback_sentinel_reserve_ood_terminal_or_protected_opened"
        ) is not False
        or evidence.get(
            "raw_question_answer_prompt_or_generation_text_persisted"
        ) is not False
    ):
        raise RuntimeError("v65a evidence schema or zero-access contract changed")
    prefix = _verify_input_receipt_v65a(evidence.get("authorized_input_receipt"))
    adapter = _verify_adapter_artifact_contract_v65a(evidence)
    actors, pid_map = _verify_actor_and_lora_receipts_v65a(evidence, prereg)
    schedule, scored = _verify_state_and_schedule_v65a(evidence, prereg)
    return ({
        "exact_authorized_prefix_input_receipt": prefix,
        "source_staged_and_live_adapter_contract": adapter,
        "live_engine_and_active_lora_receipts": actors,
        "fixed_complete_schedule": schedule,
        "zero_update_candidate_and_protected_access": True,
    }, scored, pid_map)


def _rebuild_analysis_v65a(
    scored_periods: list, evidence_content_sha256: str,
) -> dict:
    rebuilt = analysis.analyze_scored_periods_v65a(scored_periods)
    rebuilt.pop("content_sha256_before_self_field", None)
    rebuilt["source_evidence_content_sha256"] = evidence_content_sha256
    rebuilt["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v65a(rebuilt)
    )
    return rebuilt


def _verify_analysis_v65a(
    stored: dict, scored_periods: list, evidence_content_sha256: str,
) -> dict:
    rebuilt = _rebuild_analysis_v65a(scored_periods, evidence_content_sha256)
    if stored != rebuilt:
        raise RuntimeError("v65a stored analysis differs from independent rebuild")
    gate = rebuilt.get("required_alpha_zero_gate", {})
    checks = gate.get("checks", {})
    expected_checks = {
        "generated_f1_primary_interval_contains_zero",
        "joint_composite_interval_contains_zero",
        "stability_improvement_interval_contains_zero",
        "generated_f1_primary_ci_halfwidth_within_v62b_limit",
        "actor_leave_one_out_shift_within_v62b_limit",
    }
    if (
        rebuilt.get("schema") != "v65a-ranking64-alpha-zero-analysis"
        or rebuilt.get("status") != "complete_numeric_only_calibration"
        or set(checks) != expected_checks
        or any(not isinstance(value, bool) for value in checks.values())
        or gate.get("passed") is not all(checks.values())
        or gate.get("exact_or_sentinel_gate_applied") is not False
        or rebuilt.get("adapter_update_candidate_hpo_or_promotion_performed")
        is not False
        or rebuilt.get("row_64_or_later_opened") is not False
        or rebuilt.get("raw_question_answer_prompt_or_generation_text_persisted")
        is not False
        or rebuilt.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v65a rebuilt gate or non-access semantics changed")
    failed = sorted(key for key, passed in checks.items() if not passed)
    return {
        "rebuilt": rebuilt,
        "gate_observation": {
            "checks": copy.deepcopy(checks),
            "passed": gate["passed"],
            "passed_gate_count": len(checks) - len(failed),
            "failed_gate_count": len(failed),
            "failed_gates": failed,
            "gate_outcome_does_not_authorize_v65": True,
        },
    }


def _verify_postrun_prefix_v65a(receipt: object) -> dict:
    expected_keys = {
        "path", "file_size_bytes_metadata_only", "authorized_prefix_bytes",
        "authorized_prefix_sha256", "decoded_postrun",
        "requested_byte_offset_at_or_after_prefix",
        "full_file_read_or_hash_performed",
    }
    if (
        not isinstance(receipt, dict)
        or set(receipt) != expected_keys
        or Path(receipt.get("path", "")).resolve()
        != population65.V61C_ROWS.resolve()
        or isinstance(receipt.get("file_size_bytes_metadata_only"), bool)
        or not isinstance(receipt.get("file_size_bytes_metadata_only"), int)
        or receipt.get("file_size_bytes_metadata_only")
        != analysis.RANKING_SOURCE_FILE_SIZE_BYTES_V65A
        or receipt.get("authorized_prefix_bytes")
        != analysis.RANKING_PREFIX_BYTES_V65A
        or receipt.get("authorized_prefix_sha256")
        != analysis.RANKING_PREFIX_SHA256_V65A
        or receipt.get("decoded_postrun") is not False
        or receipt.get("requested_byte_offset_at_or_after_prefix") is not False
        or receipt.get("full_file_read_or_hash_performed") is not False
    ):
        raise RuntimeError("v65a hash-only postrun prefix receipt changed")
    return {
        "authorized_prefix_bytes": analysis.RANKING_PREFIX_BYTES_V65A,
        "authorized_prefix_sha256": analysis.RANKING_PREFIX_SHA256_V65A,
        "source_file_size_metadata_bytes": (
            analysis.RANKING_SOURCE_FILE_SIZE_BYTES_V65A
        ),
        "decoded_postrun": False,
        "row_64_or_later_requested": False,
        "full_file_read_or_hash_performed": False,
    }


def _gpu_summary_v65a(
    path: Path, expected_sha256: str, pid_map: dict[int, int], report: dict,
) -> dict:
    if file_sha256_v65a(path) != expected_sha256:
        raise RuntimeError("v65a GPU log file changed")
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if not line:
            continue
        try:
            def reject_duplicates(pairs):
                value = {}
                for key, item in pairs:
                    if key in value:
                        raise RuntimeError(
                            f"v65a duplicate GPU JSON key at row "
                            f"{line_number}: {key}"
                        )
                    value[key] = item
                return value

            value = json.loads(line, object_pairs_hook=reject_duplicates)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"v65a malformed GPU row {line_number}"
            ) from error
        rows.append(value)
    _verify_no_text_keys_v65a("gpu_log", rows)
    expected_keys = {
        "compute_pids", "expected_pid", "foreign_compute_pids", "gpu",
        "memory_used_mib", "phase", "sampled_at_utc",
        "utilization_percent",
    }
    if not rows:
        raise RuntimeError("v65a GPU log is empty")
    report_started = datetime.fromisoformat(
        _require_aware_timestamp_v65a(
            report.get("started_at_utc"), "report start",
        )
    )
    report_completed = datetime.fromisoformat(
        _require_aware_timestamp_v65a(
            report.get("completed_at_utc"), "report completion",
        )
    )
    if report_completed < report_started:
        raise RuntimeError("v65a report completion precedes report start")
    sample_times = []
    for row in rows:
        gpu = row.get("gpu") if isinstance(row, dict) else None
        expected_pid = pid_map.get(gpu)
        compute = row.get("compute_pids") if isinstance(row, dict) else None
        sampled_at = datetime.fromisoformat(
            _require_aware_timestamp_v65a(
                row.get("sampled_at_utc") if isinstance(row, dict) else None,
                "GPU sample",
            )
        )
        sample_times.append(sampled_at)
        if sampled_at < report_started or sampled_at > report_completed:
            raise RuntimeError(
                "v65a GPU sample timestamp outside report interval"
            )
        if (
            not isinstance(row, dict)
            or set(row) != expected_keys
            or isinstance(gpu, bool) or not isinstance(gpu, int)
            or gpu not in range(4)
            or not _exact_int_v65a(row.get("expected_pid"), expected_pid)
            or not isinstance(compute, list)
            or compute != sorted(set(compute))
            or any(
                isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
                for pid in compute
            )
            or any(pid != expected_pid for pid in compute)
            or row.get("foreign_compute_pids") != []
            or isinstance(row.get("utilization_percent"), bool)
            or not isinstance(row.get("utilization_percent"), int)
            or not 0 <= row.get("utilization_percent") <= 100
            or isinstance(row.get("memory_used_mib"), bool)
            or not isinstance(row.get("memory_used_mib"), int)
            or row.get("memory_used_mib") < 0
            or row.get("phase") not in ALLOWED_GPU_PHASES_V65A
        ):
            raise RuntimeError("v65a GPU row schema or process attribution changed")
    if any(
        current < previous
        for previous, current in zip(sample_times, sample_times[1:])
    ):
        raise RuntimeError("v65a GPU sample timestamps are not nondecreasing")

    first_phase_offsets = []
    phase_summary = {}
    for phase in EXPECTED_PHASES_V65A:
        offsets = [index for index, row in enumerate(rows) if row["phase"] == phase]
        if not offsets:
            raise RuntimeError(f"v65a missing GPU generation phase: {phase}")
        first_phase_offsets.append(min(offsets))
        by_gpu = {}
        for gpu in range(4):
            expected_pid = pid_map[gpu]
            selected = [
                row for row in rows
                if row["phase"] == phase and row["gpu"] == gpu
            ]
            resident = [
                row for row in selected if expected_pid in row["compute_pids"]
            ]
            if (
                not selected or not resident
                or not any(row["utilization_percent"] > 0 for row in resident)
            ):
                raise RuntimeError(
                    f"v65a GPU {gpu} lacked positive attributed activity in {phase}"
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
            }
        phase_summary[phase] = by_gpu
    if first_phase_offsets != sorted(first_phase_offsets) or len(set(first_phase_offsets)) != 8:
        raise RuntimeError("v65a warmup/scored GPU phase order changed")
    rebuilt_phases = {
        "schema": "v65a-per-period-four-gpu-activity",
        "generation_phases": 8,
        "all_eight_generation_phases_positive_on_all_four_gpus": True,
        "foreign_compute_process_observations": 0,
        "by_phase": phase_summary,
    }
    if report.get("gpu_period_phases") != rebuilt_phases:
        raise RuntimeError("v65a reported phase GPU summary differs from log")

    by_gpu = {}
    for gpu in range(4):
        expected_pid = pid_map[gpu]
        selected = [row for row in rows if row["gpu"] == gpu]
        resident = [
            row for row in selected if expected_pid in row["compute_pids"]
        ]
        if not resident or not any(row["utilization_percent"] > 0 for row in resident):
            raise RuntimeError(f"v65a GPU {gpu} had no positive resident sample")
        by_gpu[str(gpu)] = {
            "expected_pid": expected_pid,
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
    if report.get("gpu_activity") != rebuilt_overall:
        raise RuntimeError("v65a reported overall GPU summary differs from log")
    return {
        "gpu_log_rows": len(rows),
        "all_four_gpus_attributed_positive": True,
        "all_eight_generation_phases_positive_on_all_four_gpus": True,
        "four_warmups_preceded_four_scored_periods": True,
        "foreign_compute_process_observations": 0,
        "by_gpu": by_gpu,
        "by_phase": phase_summary,
    }


def _verify_cleanup_v65a(report: dict) -> dict:
    cleanup = report.get("cleanup", {})
    idle = report.get("final_gpu_idle")
    expected_keys = {
        "schema", "driver_scoped_non_detached_by_construction",
        "engine_kill_count", "placement_group_remove_count", "before",
        "after", "all_four_gcs_states_removed",
    }
    before = cleanup.get("before", []) if isinstance(cleanup, dict) else []
    after = cleanup.get("after", []) if isinstance(cleanup, dict) else []
    placement_keys = {
        "placement_group_id", "strategy", "state", "bundles",
        "bundles_to_node_id",
    }
    before_ids = [item.get("placement_group_id") for item in before]
    after_ids = [item.get("placement_group_id") for item in after]
    expected_bundles = {"0": {"GPU": 1.0}}
    expected_bundles_sha = analysis.canonical_sha256_v65a(expected_bundles)
    all_placement_rows = [*before, *after]
    if (
        not isinstance(cleanup, dict)
        or set(cleanup) != expected_keys
        or cleanup.get("schema") != "eggroll-es-placement-group-cleanup-v38a"
        or cleanup.get("driver_scoped_non_detached_by_construction") is not True
        or not _exact_int_v65a(cleanup.get("engine_kill_count"), 4)
        or not _exact_int_v65a(cleanup.get("placement_group_remove_count"), 4)
        or cleanup.get("all_four_gcs_states_removed") is not True
        or len(before) != 4 or len(after) != 4
        or any(set(item) != placement_keys for item in all_placement_rows)
        or len(set(before_ids)) != 4 or before_ids != after_ids
        or any(
            not isinstance(item.get("placement_group_id"), str)
            or re.fullmatch(r"[0-9a-f]{36}", item["placement_group_id"]) is None
            or item.get("bundles") != expected_bundles
            or analysis.canonical_sha256_v65a(item.get("bundles"))
            != expected_bundles_sha
            or not isinstance(item.get("bundles_to_node_id"), dict)
            or set(item["bundles_to_node_id"]) != {"0"}
            or not isinstance(item["bundles_to_node_id"].get("0"), str)
            or re.fullmatch(
                r"[0-9a-f]{56}", item["bundles_to_node_id"]["0"]
            ) is None
            for item in all_placement_rows
        )
        or len({
            item["bundles_to_node_id"]["0"] for item in all_placement_rows
        }) != 1
        or any(
            item.get("state") != "CREATED" or item.get("strategy") != "PACK"
            for item in before
        )
        or any(
            item.get("state") != "REMOVED" or item.get("strategy") != "PACK"
            for item in after
        )
        or any(
            {key: value for key, value in left.items() if key != "state"}
            != {key: value for key, value in right.items() if key != "state"}
            for left, right in zip(before, after, strict=True)
        )
        or idle != {"all_four_compute_process_lists_empty": True}
    ):
        raise RuntimeError("v65a strict four-engine cleanup or final idle changed")
    return {
        "engines_killed": 4,
        "placement_groups_removed": 4,
        "all_four_gcs_states_removed": True,
        "all_four_final_gpu_compute_process_lists_empty": True,
    }


def _verify_report_v65a(
    report: dict, attempt: dict, evidence: dict, stored_analysis: dict,
    analysis_check: dict, prereg: dict, sources: FinalizerSourcesV65A,
    pid_map: dict[int, int],
) -> dict:
    expected_keys = {
        "schema", "status", "started_at_utc", "completed_at_utc",
        "wall_runtime_seconds", "preregistration_file_sha256",
        "preregistration_content_sha256", "attempt", "evidence", "analysis",
        "runtime_determinism_controls", "actor_runtime_identities",
        "base_model_prelaunch_artifact_receipt",
        "base_model_postrun_artifact_receipt",
        "adapter_artifact_contract_prelaunch",
        "adapter_artifact_contract_postcleanup",
        "adapter_artifact_contract_unchanged",
        "authorized_prefix_postrun_receipt", "gpu_activity",
        "gpu_period_phases", "gpu_log_file_sha256", "cleanup",
        "final_gpu_idle", "warmup_generation_completions_discarded",
        "scored_generation_completions", "total_generation_completions",
        "raw_question_answer_prompt_or_generation_text_persisted",
        "candidate_hpo_update_projection_or_promotion_performed",
        "holdback_sentinel_reserve_ood_terminal_or_protected_opened",
        "v65_population_launch_authorized", "content_sha256_before_self_field",
    }
    gate = analysis_check["rebuilt"]["required_alpha_zero_gate"]
    expected_status = (
        "complete_gate_passed_population_still_unauthorized"
        if gate["passed"] else "complete_gate_failed_closed"
    )
    attempt_ref = report.get("attempt", {})
    evidence_ref = report.get("evidence", {})
    analysis_ref = report.get("analysis", {})
    wall = report.get("wall_runtime_seconds")
    started_text = _require_aware_timestamp_v65a(
        report.get("started_at_utc"), "report start",
    )
    completed_text = _require_aware_timestamp_v65a(
        report.get("completed_at_utc"), "report completion",
    )
    started_time = datetime.fromisoformat(started_text)
    completed_time = datetime.fromisoformat(completed_text)
    if (
        set(report) != expected_keys
        or report.get("schema")
        != "v65a-ranking64-alpha-zero-calibration-report"
        or report.get("status") != expected_status
        or report.get("started_at_utc") != attempt.get("started_at_utc")
        or completed_time < started_time
        or not isinstance(wall, float)
        or not math.isfinite(float(wall)) or float(wall) <= 0.0
        or report.get("preregistration_file_sha256")
        != sources.preregistration.file_sha256
        or report.get("preregistration_content_sha256")
        != sources.preregistration.content_sha256
        or Path(attempt_ref.get("path", "")).resolve() != sources.attempt.path.resolve()
        or set(attempt_ref) != {"path", "file_sha256", "content_sha256"}
        or attempt_ref.get("file_sha256") != sources.attempt.file_sha256
        or attempt_ref.get("content_sha256") != sources.attempt.content_sha256
        or Path(evidence_ref.get("path", "")).resolve()
        != sources.evidence.path.resolve()
        or set(evidence_ref) != {"path", "file_sha256", "content_sha256"}
        or evidence_ref.get("file_sha256") != sources.evidence.file_sha256
        or evidence_ref.get("content_sha256") != sources.evidence.content_sha256
        or Path(analysis_ref.get("path", "")).resolve()
        != sources.analysis.path.resolve()
        or set(analysis_ref) != {
            "path", "file_sha256", "content_sha256",
            "required_alpha_zero_gate",
        }
        or analysis_ref.get("file_sha256") != sources.analysis.file_sha256
        or analysis_ref.get("content_sha256") != sources.analysis.content_sha256
        or analysis_ref.get("required_alpha_zero_gate") != gate
        or report.get("runtime_determinism_controls")
        != analysis.ENGINE_CONTROLS_V65A
        or report.get("actor_runtime_identities")
        != evidence.get("actor_runtime_identities")
        or analysis.canonical_sha256_v65a(report.get("actor_runtime_identities"))
        != analysis.canonical_sha256_v65a(
            evidence.get("actor_runtime_identities")
        )
        or report.get("adapter_artifact_contract_prelaunch")
        != evidence.get("adapter_artifact_contract_prelaunch")
        or report.get("adapter_artifact_contract_postcleanup")
        != evidence.get("adapter_artifact_contract_postcleanup")
        or report.get("adapter_artifact_contract_prelaunch")
        != report.get("adapter_artifact_contract_postcleanup")
        or report.get("adapter_artifact_contract_unchanged") is not True
        or report.get("gpu_log_file_sha256") != sources.gpu_log_file_sha256
        or not _exact_int_v65a(
            report.get("warmup_generation_completions_discarded"), 1024,
        )
        or not _exact_int_v65a(
            report.get("scored_generation_completions"), 1024,
        )
        or not _exact_int_v65a(
            report.get("total_generation_completions"), 2048,
        )
        or report.get("raw_question_answer_prompt_or_generation_text_persisted")
        is not False
        or report.get("candidate_hpo_update_projection_or_promotion_performed")
        is not False
        or report.get(
            "holdback_sentinel_reserve_ood_terminal_or_protected_opened"
        ) is not False
        or report.get("v65_population_launch_authorized") is not False
    ):
        raise RuntimeError("v65a sealed report or source hash chain changed")

    expectation, pre = _base_model_expectation_and_receipt_v65a(
        report.get("base_model_prelaunch_artifact_receipt")
    )
    _expectation2, post = _base_model_expectation_and_receipt_v65a(
        report.get("base_model_postrun_artifact_receipt")
    )
    if (
        pre != post
        or pre != attempt.get("base_model_artifact_receipt")
        or analysis.canonical_sha256_v65a(expectation)
        != analysis.canonical_sha256_v65a(runtime64.base_model_artifact_expectation_v64())
    ):
        raise RuntimeError("v65a base-model pre/post byte receipts changed")
    post_prefix = _verify_postrun_prefix_v65a(
        report.get("authorized_prefix_postrun_receipt")
    )
    gpu = _gpu_summary_v65a(
        sources.gpu_log_path, sources.gpu_log_file_sha256, pid_map, report,
    )
    cleanup = _verify_cleanup_v65a(report)
    return {
        "reported_status_matches_observed_gate": True,
        "attempt_evidence_analysis_and_gpu_hash_chain_exact": True,
        "base_model_prelaunch_and_postrun_receipts_exact_and_identical": True,
        "exact_authorized_prefix_pread_receipts": 2,
        "postrun_authorized_prefix_receipt": post_prefix,
        "gpu_activity_recomputed_from_numeric_log": gpu,
        "cleanup_and_final_idle": cleanup,
        "v65_population_launch_authorized": False,
    }


def _performance_v65a(report: dict) -> dict:
    wall = float(report["wall_runtime_seconds"])
    return {
        "wall_runtime_seconds": wall,
        "unscored_warmup_generation_completions": 1024,
        "scored_generation_completions": 1024,
        "total_generation_completions": 2048,
        "total_generation_completions_per_second": 2048 / wall,
        "scored_generation_completions_per_second": 1024 / wall,
    }


def build_finalized_v65a(sources: FinalizerSourcesV65A) -> dict:
    prereg = _read_self_hashed_v65a(sources.preregistration)
    attempt = _read_self_hashed_v65a(sources.attempt)
    evidence = _read_self_hashed_v65a(sources.evidence)
    stored_analysis = _read_self_hashed_v65a(sources.analysis)
    report = _read_self_hashed_v65a(sources.report)
    leakage = {
        name: _verify_no_text_keys_v65a(name, value)
        for name, value in (
            ("preregistration", prereg), ("attempt", attempt),
            ("evidence", evidence), ("analysis", stored_analysis),
            ("report", report),
        )
    }
    static = _verify_preregistration_v65a(prereg, sources)
    attempt_check = _verify_attempt_v65a(attempt, prereg, sources)
    evidence_check, scored, pid_map = _verify_evidence_v65a(evidence, prereg)
    analysis_check = _verify_analysis_v65a(
        stored_analysis, scored, sources.evidence.content_sha256,
    )
    report_check = _verify_report_v65a(
        report, attempt, evidence, stored_analysis, analysis_check, prereg,
        sources, pid_map,
    )
    gate = analysis_check["gate_observation"]
    value = {
        "schema": "v65a-ranking64-alpha-zero-independent-finalizer",
        "status": "complete_numeric_only_observation_v65_still_unauthorized",
        "source_hashes": {
            name: {
                "file_sha256": source.file_sha256,
                "content_sha256": source.content_sha256,
            }
            for name, source in (
                ("preregistration", sources.preregistration),
                ("attempt", sources.attempt),
                ("evidence", sources.evidence),
                ("analysis", sources.analysis),
                ("report", sources.report),
            )
        },
        "gpu_log_file_sha256": sources.gpu_log_file_sha256,
        "verification": {
            "all_six_file_hashes_and_five_self_hashes_verified": True,
            "static_preregistration_and_implementation_chain": static,
            "launch_attempt_and_preflight": attempt_check,
            "numeric_hash_only_evidence": evidence_check,
            "stored_analysis_exactly_equals_independent_numeric_rebuild": True,
            "no_forbidden_text_keys": leakage,
            "report_hash_chain_base_bytes_gpu_cleanup_and_idle": report_check,
        },
        "observed_numeric_outcome_without_authorization": {
            "primary_cluster_bootstrap": copy.deepcopy(
                analysis_check["rebuilt"]["primary_cluster_bootstrap"]
            ),
            "actor_influence": copy.deepcopy(
                analysis_check["rebuilt"]["actor_influence"]
            ),
            "required_alpha_zero_gate": gate,
            "performance": _performance_v65a(report),
        },
        "frozen_non_authorization": {
            "outcome_assumed_before_read": False,
            "finalizer_accepts_and_records_either_gate_outcome": True,
            "thresholds_changed_after_outcome": False,
            "failed_gate_reinterpreted_or_relaxed": False,
            "v65_population_launch_authorized": False,
            "adapter_update_candidate_projection_or_promotion_authorized": False,
            "holdback_sentinel_reserve_ood_terminal_or_protected_access_authorized": False,
            "model_ray_or_gpu_launch_authorized": False,
        },
        "implementation_bindings": {
            "finalizer_file_sha256": file_sha256_v65a(Path(__file__).resolve()),
            "tests_file_sha256": file_sha256_v65a(TESTS),
        },
        "raw_question_answer_prompt_prediction_or_generation_text_persisted": False,
        "row_64_or_later_opened": False,
        "protected_semantics_opened": False,
        "v65_population_launch_authorized": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v65a(value)
    )
    return value


def _exclusive_write_v65a(path: Path, payload: bytes) -> None:
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


def _source_from_args_v65a(args, name: str) -> SelfHashedSourceV65A:
    return SelfHashedSourceV65A(
        path=Path(getattr(args, name)).resolve(),
        file_sha256=_require_sha256_v65a(
            getattr(args, f"{name}_sha256"), f"{name} file",
        ),
        content_sha256=_require_sha256_v65a(
            getattr(args, f"{name}_content_sha256"), f"{name} content",
        ),
    )


def parser_v65a() -> argparse.ArgumentParser:
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


def main(argv=None) -> int:
    args = parser_v65a().parse_args(argv)
    sources = FinalizerSourcesV65A(
        preregistration=_source_from_args_v65a(args, "preregistration"),
        attempt=_source_from_args_v65a(args, "attempt"),
        evidence=_source_from_args_v65a(args, "evidence"),
        analysis=_source_from_args_v65a(args, "analysis"),
        report=_source_from_args_v65a(args, "report"),
        gpu_log_path=Path(args.gpu_log).resolve(),
        gpu_log_file_sha256=_require_sha256_v65a(
            args.gpu_log_sha256, "GPU log file",
        ),
    )
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_finalized_v65a(sources)
    _exclusive_write_v65a(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v65a(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "required_alpha_zero_gate_passed": value[
            "observed_numeric_outcome_without_authorization"
        ]["required_alpha_zero_gate"]["passed"],
        "v65_population_launch_authorized": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
