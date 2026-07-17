#!/usr/bin/env python3
"""Fresh-path V73E target for content-free exact-phase profiling.

V73E retains the inherited ES state/audit mechanics but rejects historical
semantic, reward, population, update, and equivalence authority.  It uses a
new deterministic token-ID workload, staged-only adapter bootstrap, controller
NVTX ranges, and self-hashed receipts.  The process is not a launcher: live
execution requires an exact expanded-command attestation from its parent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import qwen36_v73e_exact_phase_profiler_contract as builder
import run_lora_es_mirrored_calibration_v66 as v66
import build_lora_es_v71_v72_live_calibration_preregistration_v73 as v73_builder


def _sealed_historical_v73_contract_for_systems_runtime():
    path = builder.V73_PREREGISTRATION
    if builder.file_sha256(path) != builder.V73_PREREGISTRATION_FILE_SHA256:
        raise RuntimeError("V73E historical V73 systems contract changed")
    value = json.loads(path.read_text(encoding="ascii"))
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    if (
        claimed != builder.V73_PREREGISTRATION_CONTENT_SHA256
        or builder.canonical_sha256(body) != claimed
    ):
        raise RuntimeError("V73E historical V73 contract self hash changed")
    return value


# V73's legacy module computes endpoint constants by rebuilding V66 ancestry,
# which would open the quarantined historical contract.  Supply the already
# sealed V73 bytes for that import-time constant only; no quality authority is
# inherited or rebuilt.
_ORIGINAL_V73_PREREGISTRATION_BUILDER = v73_builder.build_preregistration_v73
v73_builder.build_preregistration_v73 = (
    _sealed_historical_v73_contract_for_systems_runtime
)
try:
    import run_lora_es_v71_v72_live_calibration_v73 as v73
    import run_lora_es_v71_v72_same_live_calibration_v73b as v73b
finally:
    v73_builder.build_preregistration_v73 = (
        _ORIGINAL_V73_PREREGISTRATION_BUILDER
    )


ROOT = Path(__file__).resolve().parent
COMMAND_ATTESTATION_ENV = "SPECIALIST_V73E_EXPANDED_COMMAND_SHA256"
_PHASE_INSTANCE = None
_GUARD_EVIDENCE = None
_GUARD_FAILURE = None
_RAY_BOOTSTRAP_EVIDENCE = None
_STAGED_ADAPTER_EVIDENCE = None
_CONTENT_FREE_INPUT_EVIDENCE = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="ascii"))
    _require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _validate_self_hash(value: Mapping[str, Any], label: str) -> str:
    body = dict(value)
    claimed = body.pop("content_sha256_before_self_field", None)
    _require(
        isinstance(claimed, str) and claimed == builder.canonical_sha256(body),
        f"V73E self hash changed: {label}",
    )
    return claimed


def _validate_inverse_transform_proof_v73e(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(value)
    _require(
        row.get("schema")
        == "qwen36-v73e-staged-inverse-transform-proof-v1"
        and row.get("operation") == "exact_prefix_inverse_only"
        and row.get("source_tensor_namespace")
        == builder.STAGED_TARGET_PREFIX
        and row.get("canonical_tensor_namespace")
        == builder.CANONICAL_SOURCE_PREFIX
        and row.get("tensor_count") == builder.STAGED_TENSOR_COUNT
        and row.get("element_count") == builder.STAGED_ELEMENT_COUNT
        and row.get("tensor_arithmetic_performed") is False
        and row.get("tensor_cast_performed") is False
        and row.get("tensor_bytes_preserved_exactly") is True
        and row.get("inverse_key_mapping_sha256")
        == builder.INVERSE_KEY_MAPPING_SHA256
        and row.get("staged_transform_identity_sha256")
        == builder.STAGED_TRANSFORM_IDENTITY_SHA256
        and row.get("canonical_master_sha256")
        == builder.CANONICAL_MASTER_SHA256
        and row.get("canonical_runtime_values_sha256")
        == builder.CANONICAL_RUNTIME_VALUES_SHA256
        and row.get(
            "historical_protected_source_opened_resolved_statted_or_hashed"
        ) is False,
        "V73E exact inverse-transform proof changed",
    )
    return row


def verify_staged_adapter_contract_v73e() -> dict[str, Any]:
    """Verify only the sealed, unprotected staged adapter and exact inverse."""
    files = (
        builder.STAGED_ADAPTER_WEIGHTS,
        builder.STAGED_ADAPTER_CONFIG,
        builder.STAGED_ADAPTER_MANIFEST,
        builder.STAGED_TRANSFORM_IMPLEMENTATION,
    )
    _require(
        all(path.is_file() and not path.is_symlink() for path in files),
        "V73E staged adapter inputs must be regular non-symlink files",
    )
    observed = {
        "weights": builder.file_sha256(builder.STAGED_ADAPTER_WEIGHTS),
        "config": builder.file_sha256(builder.STAGED_ADAPTER_CONFIG),
        "manifest": builder.file_sha256(builder.STAGED_ADAPTER_MANIFEST),
        "implementation": builder.file_sha256(
            builder.STAGED_TRANSFORM_IMPLEMENTATION
        ),
    }
    expected = {
        "weights": builder.STAGED_ADAPTER_WEIGHTS_SHA256,
        "config": builder.STAGED_ADAPTER_CONFIG_SHA256,
        "manifest": builder.STAGED_ADAPTER_MANIFEST_FILE_SHA256,
        "implementation": builder.STAGED_TRANSFORM_IMPLEMENTATION_SHA256,
    }
    _require(observed == expected, "V73E staged adapter file identity changed")
    manifest = _load_json(builder.STAGED_ADAPTER_MANIFEST)
    content_sha256 = _validate_self_hash(manifest, "staged adapter manifest")
    artifact = manifest.get("artifact", {})
    implementation = manifest.get("implementation", {})
    transform = manifest.get("transform", {})
    transformed = manifest.get("transformed_identity", {})
    source = manifest.get("source", {})
    source_seal = source.get("seal", {})
    records = manifest.get("tensor_mapping_records", [])
    _require(
        content_sha256 == builder.STAGED_ADAPTER_MANIFEST_CONTENT_SHA256
        and manifest.get("schema")
        == "candidate-lora-vllm-stage-manifest-v44a"
        and manifest.get("status") == "complete_cpu_only_key_transform"
        and manifest.get("arm") == "v434_equal"
        and manifest.get("dataset_or_evaluation_accessed") is False
        and manifest.get("gpu_accessed") is False
        and manifest.get("shadow_ood_holdout_or_heldout_accessed") is False
        and artifact.get("directory")
        == str(builder.STAGED_ADAPTER_DIRECTORY)
        and artifact.get("weights_file_sha256")
        == builder.STAGED_ADAPTER_WEIGHTS_SHA256
        and artifact.get("adapter_config_file_sha256")
        == builder.STAGED_ADAPTER_CONFIG_SHA256
        and artifact.get("target_namespace")
        == builder.STAGED_TARGET_PREFIX + "*"
        and implementation.get("path")
        == str(builder.STAGED_TRANSFORM_IMPLEMENTATION)
        and implementation.get("file_sha256")
        == builder.STAGED_TRANSFORM_IMPLEMENTATION_SHA256
        and transform == {
            "adapter_config_copied_byte_exact": True,
            "operation": "key_prefix_replacement_only",
            "source_prefix": builder.CANONICAL_SOURCE_PREFIX,
            "target_prefix": builder.STAGED_TARGET_PREFIX,
            "tensor_arithmetic_performed": False,
            "tensor_bytes_preserved_exactly": True,
            "tensor_cast_performed": False,
        }
        and source.get("weights_file_sha256")
        == builder.HISTORICAL_PROTECTED_SOURCE_WEIGHTS_SHA256
        and source.get("adapter_config_file_sha256")
        == builder.STAGED_ADAPTER_CONFIG_SHA256
        and source_seal.get("source_weights_sha256")
        == builder.HISTORICAL_PROTECTED_SOURCE_WEIGHTS_SHA256
        and source_seal.get("source_config_sha256")
        == builder.STAGED_ADAPTER_CONFIG_SHA256
        and source_seal.get("shadow_ood_holdout_or_heldout_opened") is False
        and isinstance(records, list)
        and len(records) == builder.STAGED_TENSOR_COUNT,
        "V73E sealed staged manifest contract changed",
    )
    source_keys = []
    target_keys = []
    value_hashes = []
    element_count = 0
    for row in records:
        source_key = row.get("source_key")
        target_key = row.get("target_key")
        shape = row.get("shape")
        elements = row.get("elements")
        _require(
            isinstance(source_key, str)
            and isinstance(target_key, str)
            and source_key.startswith(builder.CANONICAL_SOURCE_PREFIX)
            and target_key.startswith(builder.STAGED_TARGET_PREFIX)
            and target_key
            == builder.STAGED_TARGET_PREFIX
            + source_key[len(builder.CANONICAL_SOURCE_PREFIX):]
            and isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(item, int) and item > 0 for item in shape)
            and isinstance(elements, int)
            and elements == shape[0] * shape[1]
            and row.get("dtype") == "torch.float32"
            and isinstance(row.get("tensor_sha256"), str)
            and row.get("target_tensor_sha256")
            == row.get("tensor_sha256")
            and row.get("tensor_bytes_preserved_exactly") is True,
            "V73E staged tensor mapping changed",
        )
        source_keys.append(source_key)
        target_keys.append(target_key)
        value_hashes.append(row["tensor_sha256"])
        element_count += elements
    _require(
        source_keys == sorted(source_keys)
        and target_keys == sorted(target_keys)
        and len(set(source_keys)) == builder.STAGED_TENSOR_COUNT
        and len(set(target_keys)) == builder.STAGED_TENSOR_COUNT
        and element_count == builder.STAGED_ELEMENT_COUNT
        and builder.canonical_sha256(records)
        == builder.STAGED_TRANSFORM_IDENTITY_SHA256
        and builder.canonical_sha256(source_keys)
        == builder.CANONICAL_SOURCE_KEY_INVENTORY_SHA256
        and builder.canonical_sha256(target_keys)
        == builder.STAGED_TARGET_KEY_INVENTORY_SHA256
        and builder.canonical_sha256(value_hashes)
        == builder.STAGED_ORDERED_VALUE_SEQUENCE_SHA256
        and transformed == {
            "all_tensor_bytes_preserved_exactly": True,
            "elements": builder.STAGED_ELEMENT_COUNT,
            "ordered_value_sequence_sha256": (
                builder.STAGED_ORDERED_VALUE_SEQUENCE_SHA256
            ),
            "schema": "vllm-language-model-lora-stage-identity-v44a",
            "sha256": builder.STAGED_TRANSFORM_IDENTITY_SHA256,
            "source_key_inventory_sha256": (
                builder.CANONICAL_SOURCE_KEY_INVENTORY_SHA256
            ),
            "target_key_inventory_sha256": (
                builder.STAGED_TARGET_KEY_INVENTORY_SHA256
            ),
            "tensor_count": builder.STAGED_TENSOR_COUNT,
        },
        "V73E staged transform identity changed",
    )

    import torch
    from safetensors import safe_open

    canonical_records = []
    with safe_open(
        builder.STAGED_ADAPTER_WEIGHTS, framework="pt", device="cpu"
    ) as handle:
        _require(
            sorted(handle.keys()) == target_keys
            and handle.metadata() == {"format": "pt"},
            "V73E staged safetensors inventory or metadata changed",
        )
        for manifest_row in records:
            tensor = handle.get_tensor(manifest_row["target_key"])
            tensor_sha256 = hashlib.sha256(
                tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            _require(
                tensor.dtype == torch.float32
                and tensor.device.type == "cpu"
                and tensor.ndim == 2
                and tensor.is_contiguous()
                and list(tensor.shape) == manifest_row["shape"]
                and int(tensor.numel()) == manifest_row["elements"]
                and tensor_sha256 == manifest_row["tensor_sha256"],
                "V73E staged tensor bytes differ from sealed mapping",
            )
            canonical_records.append({
                "key": manifest_row["source_key"],
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "elements": int(tensor.numel()),
                "sha256": tensor_sha256,
            })
    identity = {
        "schema": "canonical-peft-fp32-state-v41a",
        "sha256": builder.canonical_sha256(canonical_records),
        "tensor_count": len(canonical_records),
        "elements": sum(row["elements"] for row in canonical_records),
        "bytes": sum(row["elements"] * 4 for row in canonical_records),
        "ordered_key_sha256": builder.canonical_sha256([
            row["key"] for row in canonical_records
        ]),
    }
    _require(
        identity.get("sha256") == builder.CANONICAL_MASTER_SHA256
        and identity.get("ordered_key_sha256")
        == builder.CANONICAL_SOURCE_KEY_INVENTORY_SHA256
        and identity.get("tensor_count") == builder.STAGED_TENSOR_COUNT
        and identity.get("elements") == builder.STAGED_ELEMENT_COUNT
        and identity.get("bytes") == builder.STAGED_TENSOR_BYTES,
        "V73E inverse canonical master identity changed",
    )
    result = {
        "schema": "qwen36-v73e-staged-only-adapter-contract-v1",
        "status": "sealed_staged_only_exact_inverse_verified",
        "staged_files": {
            "weights": {
                "path": str(builder.STAGED_ADAPTER_WEIGHTS),
                "file_sha256": observed["weights"],
            },
            "config": {
                "path": str(builder.STAGED_ADAPTER_CONFIG),
                "file_sha256": observed["config"],
            },
            "manifest": {
                "path": str(builder.STAGED_ADAPTER_MANIFEST),
                "file_sha256": observed["manifest"],
                "content_sha256": content_sha256,
            },
            "transform_implementation": {
                "path": str(builder.STAGED_TRANSFORM_IMPLEMENTATION),
                "file_sha256": observed["implementation"],
            },
        },
        "inverse_transform_proof": _validate_inverse_transform_proof_v73e({
            "schema": "qwen36-v73e-staged-inverse-transform-proof-v1",
            "operation": "exact_prefix_inverse_only",
            "source_tensor_namespace": builder.STAGED_TARGET_PREFIX,
            "canonical_tensor_namespace": builder.CANONICAL_SOURCE_PREFIX,
            "tensor_count": builder.STAGED_TENSOR_COUNT,
            "element_count": builder.STAGED_ELEMENT_COUNT,
            "tensor_arithmetic_performed": False,
            "tensor_cast_performed": False,
            "tensor_bytes_preserved_exactly": True,
            "inverse_key_mapping_sha256": builder.INVERSE_KEY_MAPPING_SHA256,
            "staged_transform_identity_sha256": (
                builder.STAGED_TRANSFORM_IDENTITY_SHA256
            ),
            "canonical_master_sha256": builder.CANONICAL_MASTER_SHA256,
            "canonical_runtime_values_sha256": (
                builder.CANONICAL_RUNTIME_VALUES_SHA256
            ),
            "historical_protected_source_opened_resolved_statted_or_hashed": (
                False
            ),
        }),
        "canonical_master_identity": {
            key: identity[key]
            for key in (
                "sha256", "tensor_count", "elements", "bytes",
                "ordered_key_sha256",
            )
        },
        "canonical_runtime_values_sha256": (
            builder.CANONICAL_RUNTIME_VALUES_SHA256
        ),
        "historical_source_seal_weights_sha256_from_manifest_only": (
            builder.HISTORICAL_PROTECTED_SOURCE_WEIGHTS_SHA256
        ),
        "historical_protected_source_path_persisted": False,
        "historical_protected_source_opened_resolved_statted_or_hashed": False,
        "protected_semantics_opened": False,
        "quality_hpo_or_promotion_authorized": False,
    }
    result["receipt_sha256"] = builder.canonical_sha256(result)
    return result


@contextmanager
def staged_activation_alias_v73e(prior):
    """Alias V66's three source fields without inspecting old path values."""
    names = ("SOURCE", "SOURCE_WEIGHTS", "SOURCE_CONFIG")
    saved = {name: getattr(prior, name) for name in names}
    prior.SOURCE = builder.STAGED_ADAPTER_DIRECTORY
    prior.SOURCE_WEIGHTS = builder.STAGED_ADAPTER_WEIGHTS
    prior.SOURCE_CONFIG = builder.STAGED_ADAPTER_CONFIG
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(prior, name, value)


def content_free_generation_panel_v73e() -> dict[str, Any]:
    """Compatibility shell for ancestry that requires a sealed subset value."""
    panel = builder.content_free_token_panel_v73e()
    _require(
        panel.get("content_sha256_before_self_field")
        == builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256,
        "V73E content-free token panel identity changed",
    )
    items = [{
        "request_index": row["request_index"],
        "row_sha256": row["prompt_token_ids_sha256"],
        "unit_identity_sha256": builder.canonical_sha256({
            "request_index": row["request_index"],
            "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
        }),
    } for row in panel["records"]]
    subset = {
        "schema": "qwen36-v73e-content-free-subset-compatibility-v1",
        "items": items,
        "content_sha256_before_self_field": builder.canonical_sha256(items),
        "semantic_rows_present": False,
    }
    return {
        "schema": "qwen36-v73e-content-free-generation-panel-v1",
        "subset": subset,
        "request_order_sha256": builder.canonical_sha256([
            row["row_sha256"] for row in items
        ]),
        "common_random_generation_params": {
            "n": 1,
            "seed": v66.EVALUATION_SEED_V66,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 1,
            "prompt_logprobs": 1,
            "detokenize": False,
        },
        "content_free_token_panel_content_sha256": (
            builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        ),
        "historical_same_live_semantic_authority_inherited": False,
    }


def expected_content_free_payload_and_plan_v73e() -> dict[str, Any]:
    """Rebuild the sealed numeric payload and exact mirrored assignment plan."""
    panel = builder.content_free_token_panel_v73e()
    requests = [
        {"prompt_token_ids": list(row["prompt_token_ids"])}
        for row in panel["records"]
    ]
    decode = {
        "n": 1,
        "seed": v66.EVALUATION_SEED_V66,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "prompt_logprobs": 1,
        "detokenize": False,
    }
    judge = {
        "schema": "qwen36-v73e-content-free-answer-span-logprob-v1",
        "reward_config_sha256": (
            builder.CONTENT_FREE_SUFFIX_REWARD_CONFIG_SHA256
        ),
        "reward_config": builder.content_free_suffix_reward_config_v73e(),
        "content_free_token_panel_content_sha256": (
            builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        ),
        "aggregation": "uniform_mean_of_64_answer_span_mean_logprobs",
        "text_question_answer_or_semantic_label_present": False,
        "historical_reward_or_update_authority_inherited": False,
    }
    payload = v66.mirrored.common_evaluation_payload_v66(
        requests, decode, judge, v66.EVALUATION_SEED_V66
    )
    plan = v66.mirrored.mirrored_population_plan_v66(
        builder.CONTENT_FREE_DIRECTION_SEEDS,
        builder.CONTENT_FREE_SIGMA,
        payload,
    )
    _require(
        panel.get("content_sha256_before_self_field")
        == builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        and builder.canonical_sha256(requests)
        == builder.CONTENT_FREE_REQUEST_BLOCK_SHA256
        and payload.get("contract", {}).get("evaluation_contract_sha256")
        == builder.CONTENT_FREE_EVALUATION_CONTRACT_SHA256
        and plan.get("plan_sha256") == builder.CONTENT_FREE_PLAN_SHA256,
        "V73E sealed numeric payload or mirrored plan changed",
    )
    return {
        "requests": requests,
        "decode": decode,
        "judge": judge,
        "payload": payload,
        "plan": plan,
    }


def prepare_content_free_inputs_v73e(trainer, prior, runtime52):
    """Prepare fixed token IDs directly, with no dataset or text dependency."""
    del runtime52
    global _CONTENT_FREE_INPUT_EVIDENCE
    _require(
        _CONTENT_FREE_INPUT_EVIDENCE is None,
        "V73E content-free input preparation was invoked more than once",
    )
    panel = builder.content_free_token_panel_v73e()
    _require(
        panel.get("content_sha256_before_self_field")
        == builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        and panel.get("request_count") == builder.CONTENT_FREE_TOKEN_PANEL_ROWS
        and len(panel.get("records", []))
        == builder.CONTENT_FREE_TOKEN_PANEL_ROWS,
        "V73E content-free input panel changed",
    )
    dense_items = []
    requests = []
    tokenizer = trainer.tokenizer
    try:
        tokenizer_vocab_size = len(tokenizer)
    except (TypeError, AttributeError) as error:
        raise RuntimeError("V73E tokenizer vocabulary size is unavailable") from error
    special_ids = sorted(set(getattr(tokenizer, "all_special_ids", []) or []))
    _require(
        isinstance(tokenizer_vocab_size, int)
        and tokenizer_vocab_size > 0
        and all(
            isinstance(item, int) and not isinstance(item, bool)
            for item in special_ids
        ),
        "V73E tokenizer metadata changed",
    )
    for index, row in enumerate(panel["records"]):
        token_ids = list(row["prompt_token_ids"])
        start = row["answer_token_start"]
        count = row["answer_token_count"]
        _require(
            row.get("request_index") == index
            and len(token_ids) == row.get("total_token_count")
            and row.get("unscored_prefix_token_count") == start
            and start > 0
            and start + count == len(token_ids)
            and count in {25, 26}
            and min(token_ids) >= builder.CONTENT_FREE_TOKEN_ID_MINIMUM
            and max(token_ids) < (
                builder.CONTENT_FREE_TOKEN_ID_MINIMUM
                + builder.CONTENT_FREE_TOKEN_ID_SPAN
            )
            and max(token_ids) < tokenizer_vocab_size
            and set(token_ids).isdisjoint(special_ids)
            and builder.canonical_sha256(token_ids)
            == row.get("prompt_token_ids_sha256")
            and row.get("text_question_answer_or_semantic_label_present")
            is False,
            "V73E content-free request shape changed",
        )
        prompt_ids = token_ids[:start]
        answer_ids = token_ids[start:]
        dense_items.append({
            "example_index": index,
            "unscored_prefix_token_ids_sha256": builder.canonical_sha256(
                prompt_ids
            ),
            "scored_suffix_token_ids_sha256": builder.canonical_sha256(
                answer_ids
            ),
            "prompt_token_count": start,
            "answer_token_start": start,
            "answer_token_count": count,
            "prompt_token_ids": token_ids,
            "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
            "eos_appended": False,
            "content_free_systems_request": True,
        })
        requests.append({"prompt_token_ids": token_ids})
    answer_token_count = sum(
        item["answer_token_count"] for item in dense_items
    )
    prefix_token_count = sum(
        item["prompt_token_count"] for item in dense_items
    )
    total_token_count = sum(
        len(item["prompt_token_ids"]) for item in dense_items
    )
    _require(
        answer_token_count
        == builder.CONTENT_FREE_ANSWER_TOKENS_PER_CANDIDATE
        and prefix_token_count
        == builder.CONTENT_FREE_UNSCORED_PREFIX_TOKENS_PER_CANDIDATE
        and total_token_count
        == builder.CONTENT_FREE_TOTAL_TOKENS_PER_CANDIDATE,
        "V73E content-free token totals changed",
    )
    length_manifest = [{
        "request_index": row["request_index"],
        "unscored_prefix_token_count": row[
            "unscored_prefix_token_count"
        ],
        "answer_token_start": row["answer_token_start"],
        "answer_token_count": row["answer_token_count"],
        "total_token_count": row["total_token_count"],
    } for row in panel["records"]]
    _require(
        builder.canonical_sha256(requests)
        == builder.CONTENT_FREE_REQUEST_BLOCK_SHA256
        and builder.canonical_sha256([
            row["prompt_token_ids"] for row in panel["records"]
        ]) == builder.CONTENT_FREE_RAW_TOKEN_INVENTORY_SHA256
        and builder.canonical_sha256(length_manifest)
        == builder.CONTENT_FREE_LENGTH_MANIFEST_SHA256,
        "V73E content-free token or length inventory changed",
    )
    decode = {
        "n": 1,
        "seed": v66.EVALUATION_SEED_V66,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 1,
        "prompt_logprobs": 1,
        "detokenize": False,
    }
    judge = {
        "schema": "qwen36-v73e-content-free-answer-span-logprob-v1",
        "reward_config_sha256": (
            builder.CONTENT_FREE_SUFFIX_REWARD_CONFIG_SHA256
        ),
        "reward_config": builder.content_free_suffix_reward_config_v73e(),
        "content_free_token_panel_content_sha256": (
            builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        ),
        "aggregation": "uniform_mean_of_64_answer_span_mean_logprobs",
        "text_question_answer_or_semantic_label_present": False,
        "historical_reward_or_update_authority_inherited": False,
    }
    payload = v66.mirrored.common_evaluation_payload_v66(
        requests, decode, judge, v66.EVALUATION_SEED_V66
    )
    prompt_sha256 = builder.canonical_sha256(requests)
    _require(
        payload["contract"]["prompt_block_sha256"] == prompt_sha256
        and payload["contract"]["evaluation_contract_sha256"]
        == builder.CONTENT_FREE_EVALUATION_CONTRACT_SHA256,
        "V73E content-free prompt block identity changed",
    )
    input_receipt = {
        "schema": "qwen36-v73e-content-free-systems-input-receipt-v1",
        "content_free_token_panel_content_sha256": (
            builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        ),
        "request_count": builder.CONTENT_FREE_TOKEN_PANEL_ROWS,
        "unscored_prefix_tokens_per_candidate": prefix_token_count,
        "total_tokens_per_candidate": total_token_count,
        "answer_tokens_per_candidate": answer_token_count,
        "minimum_total_tokens_per_request": min(
            row["total_token_count"] for row in panel["records"]
        ),
        "maximum_total_tokens_per_request": max(
            row["total_token_count"] for row in panel["records"]
        ),
        "per_request_length_records_sha256": (
            builder.CONTENT_FREE_LENGTH_MANIFEST_SHA256
        ),
        "prompt_token_ids_sha256": prompt_sha256,
        "evaluation_contract_sha256": payload["contract"][
            "evaluation_contract_sha256"
        ],
        "tokenizer_numeric_compatibility": {
            "vocab_size": tokenizer_vocab_size,
            "special_id_set_sha256": builder.canonical_sha256(special_ids),
            "synthetic_id_special_intersection_count": 0,
            "all_synthetic_ids_strictly_below_vocab_size": True,
            "adaptive_token_replacement_or_rejection_sampling": False,
        },
        "safe_historical_aggregate_answer_token_count_reproduced": True,
        "safe_historical_aggregate_full_input_token_count_reproduced": True,
        "historical_per_request_lengths_or_token_ids_reused": False,
        "qa_dev_or_other_semantic_dataset_used": False,
        "raw_text_questions_answers_or_outputs_persisted": False,
        "protected_semantics_opened": False,
        "historical_same_live_semantic_authority_inherited": False,
    }
    input_receipt["receipt_sha256"] = builder.canonical_sha256(input_receipt)
    _CONTENT_FREE_INPUT_EVIDENCE = copy.deepcopy(input_receipt)
    return {
        "panel": content_free_generation_panel_v73e(),
        "dense_items": dense_items,
        "payload": payload,
        "answer_token_count_per_candidate": answer_token_count,
        "input_receipt": input_receipt,
    }


def score_content_free_suffix_outputs_v73e(items, outputs):
    """Numeric-only suffix logprob reducer with no gold/text semantics."""
    items = list(items)
    outputs = list(outputs)
    if len(items) != len(outputs) or not items:
        raise ValueError("V73E suffix item/output counts differ or are empty")
    examples = []
    for item, output in zip(items, outputs, strict=True):
        expected_ids = list(item["prompt_token_ids"])
        returned_ids = list(getattr(output, "prompt_token_ids", None) or [])
        if returned_ids != expected_ids:
            raise ValueError("V73E runtime changed content-free token IDs")
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if prompt_logprobs is None or len(prompt_logprobs) != len(expected_ids):
            raise ValueError("V73E runtime omitted content-free prompt logprobs")
        start = item["answer_token_start"]
        stop = start + item["answer_token_count"]
        if start <= 0 or stop != len(expected_ids):
            raise ValueError("V73E scored suffix boundary changed")
        values = []
        for position in range(start, stop):
            token_id = expected_ids[position]
            candidates = prompt_logprobs[position]
            if candidates is None or token_id not in candidates:
                raise ValueError("V73E runtime omitted a scored suffix token")
            selected = candidates[token_id]
            value = float(
                selected.logprob
                if hasattr(selected, "logprob") else selected["logprob"]
            )
            if not math.isfinite(value):
                raise ValueError("V73E scored suffix logprob is non-finite")
            values.append(value)
        token_sum = math.fsum(values)
        examples.append({
            "example_index": item["example_index"],
            "unscored_prefix_token_ids_sha256": item[
                "unscored_prefix_token_ids_sha256"
            ],
            "scored_suffix_token_ids_sha256": item[
                "scored_suffix_token_ids_sha256"
            ],
            "prompt_token_ids_sha256": item["prompt_token_ids_sha256"],
            "answer_token_count": len(values),
            "sum_scored_suffix_token_logprob": token_sum,
            "mean_answer_token_logprob": token_sum / len(values),
            "eos_scored": False,
        })
    means = [row["mean_answer_token_logprob"] for row in examples]
    result = {
        "schema": "qwen36-v73e-content-free-suffix-logprob-result-v1",
        "reward_config_sha256": (
            builder.CONTENT_FREE_SUFFIX_REWARD_CONFIG_SHA256
        ),
        "example_count": len(examples),
        "answer_token_count": sum(
            row["answer_token_count"] for row in examples
        ),
        "mean_example_mean_logprob": math.fsum(means) / len(means),
        "question_answer_gold_or_semantic_label_present": False,
        "examples": examples,
    }
    return result


def population_self_consistency_v73e(
    candidate: Mapping[str, Any], _historical_control: Mapping[str, Any]
) -> dict[str, Any]:
    """Accept only internally complete live systems evidence, never history."""
    context = v73._ACTIVE_CONTEXT_V73
    rewards = candidate.get("signed_rewards", [])
    plan = candidate.get("plan", {})
    expected_plan = expected_content_free_payload_and_plan_v73e()["plan"]
    expected_assignments = [
        item for wave in expected_plan["waves"] for item in wave
    ]
    sealed_plan = dict(plan)
    plan_sha256 = sealed_plan.pop("plan_sha256", None)
    input_receipt = candidate.get("input_receipt", {})
    materializations = candidate.get("materializations", [])
    restorations = candidate.get("restorations", [])
    assignment_fields = (
        "direction_seed",
        "sigma",
        "sign",
        "pair_id",
        "evaluation_contract_sha256",
    )
    expected_materialization_coordinates = sorted(
        [
            {field: row[field] for field in assignment_fields}
            for row in expected_assignments
        ],
        key=lambda row: (row["pair_id"], row["sign"]),
    )
    observed_materialization_coordinates = sorted(
        [
            {field: row.get(field) for field in assignment_fields}
            for row in materializations
        ],
        key=lambda row: (str(row["pair_id"]), int(row["sign"] or 0)),
    )
    _require(
        context is not None
        and isinstance(_CONTENT_FREE_INPUT_EVIDENCE, dict)
        and input_receipt == _CONTENT_FREE_INPUT_EVIDENCE
        and plan_sha256 == builder.canonical_sha256(sealed_plan)
        and plan_sha256 == builder.CONTENT_FREE_PLAN_SHA256
        and plan == expected_plan
        and candidate.get("plan_sha256") == plan_sha256
        and candidate.get("evaluation_contract_sha256")
        == input_receipt.get("evaluation_contract_sha256")
        and candidate.get("install_master_consensus_sha256")
        == builder.CANONICAL_MASTER_SHA256
        and candidate.get("installation_count") == 4
        and isinstance(rewards, list)
        and len(rewards) == v73.SIGNED_CANDIDATES_V73
        and all(
            isinstance(row.get("reward"), (float, int))
            and not isinstance(row.get("reward"), bool)
            and math.isfinite(float(row["reward"]))
            for row in rewards
        )
        and candidate.get("signed_reward_sha256")
        == builder.canonical_sha256(rewards)
        and len(materializations) == v73.SIGNED_CANDIDATES_V73
        and len(restorations) == v73.SIGNED_CANDIDATES_V73
        and observed_materialization_coordinates
        == expected_materialization_coordinates
        and len({row.get("candidate_sha256") for row in materializations})
        == v73.SIGNED_CANDIDATES_V73
        and all(
            row.get("master_sha256") == builder.CANONICAL_MASTER_SHA256
            and row.get("evaluation_contract_sha256")
            == input_receipt["evaluation_contract_sha256"]
            for row in materializations
        )
        and all(
            row.get("restored_master_sha256")
            == builder.CANONICAL_MASTER_SHA256
            and row.get("runtime_values_sha256")
            == builder.CANONICAL_RUNTIME_VALUES_SHA256
            and row.get("terminal_poisoned") is False
            for row in restorations
        )
        and candidate.get("all_submitted_work_drained") is True
        and candidate.get("all_four_actors_restored_after_every_wave") is True
        and candidate.get("raw_questions_answers_or_outputs_persisted") is False
        and candidate.get("protected_dev_ood_or_holdout_opened") is False,
        "V73E content-free population self-consistency changed",
    )
    independent_update = v73b.independent_pair_difference_update_v73b(
        plan,
        rewards,
        context.preregistration["fixed_recipe"]["learning_rate"],
    )
    _require(
        independent_update.get("schema")
        == "mirrored-es-pair-difference-update-v66"
        and independent_update.get("plan_sha256")
        == builder.CONTENT_FREE_PLAN_SHA256
        and independent_update.get("direction_seeds")
        == list(builder.CONTENT_FREE_DIRECTION_SEEDS)
        and independent_update.get("worker_population_size") == 8,
        "V73E independently compiled population update changed",
    )
    result = {
        "schema": "qwen36-v73e-content-free-population-consistency-v1",
        "plan_sha256": plan_sha256,
        "evaluation_contract_sha256": input_receipt[
            "evaluation_contract_sha256"
        ],
        "signed_reward_sha256": candidate["signed_reward_sha256"],
        "candidate_count": len(rewards),
        "candidate_identity_count": len(materializations),
        "assignment_matrix_sha256": builder.canonical_sha256(
            expected_materialization_coordinates
        ),
        "independent_update_sha256": builder.canonical_sha256(
            independent_update
        ),
        "all_candidates_unique_and_finite": True,
        "all_restores_exact_to_canonical_master_and_runtime": True,
        "historical_reward_population_or_semantic_authority_inherited": False,
        "panel_evaluation_plan_candidate_and_restore_rerun_exact": True,
        "floating_reward_cross_rerun_equality_required": False,
        "floating_reward_cross_rerun_comparison": "diagnostic_only",
    }
    result["equivalence_sha256"] = builder.canonical_sha256(result)
    return result


def update_self_consistency_v73e(
    candidate: Mapping[str, Any], _historical_control: Mapping[str, Any]
) -> dict[str, Any]:
    context = v73._ACTIVE_CONTEXT_V73
    _require(
        context is not None
        and isinstance(getattr(context, "same_live_update", None), dict)
        and isinstance(
            getattr(context, "same_live_compiler_equivalence", None), dict
        )
        and isinstance(getattr(context, "live_update_execution", None), dict)
        and isinstance(context.update_invariants, dict),
        "V73E content-free update evidence is incomplete",
    )
    expected = context.same_live_update
    compiler = context.same_live_compiler_equivalence
    expected_fields = {
        "coefficient_l2": math.sqrt(math.fsum(
            value * value for value in expected["coefficients"]
        )),
        "nonzero_pair_differences": sum(
            value != 0.0 for value in expected["coefficients"]
        ),
        "direction_count": len(expected["coefficients"]),
        "worker_alpha": expected["worker_alpha"],
        "worker_population_size": expected["worker_population_size"],
        "effective_noise_scale": expected["effective_noise_scale"],
        "coefficient_sha256": expected["coefficient_sha256"],
    }
    execution = context.live_update_execution
    invariants = context.update_invariants
    shards = candidate.get("prepared_rank_shards", [])
    expected_shards = [
        {
            "rank": rank,
            "shard_indices": list(
                range(rank, len(builder.CONTENT_FREE_DIRECTION_SEEDS), 4)
            ),
            "shard_seeds": list(builder.CONTENT_FREE_DIRECTION_SEEDS[rank::4]),
        }
        for rank in range(4)
    ]
    compiler_body = dict(compiler)
    compiler_sha256 = compiler_body.pop("equivalence_sha256", None)
    execution_body = dict(execution)
    execution_sha256 = execution_body.pop("consensus_sha256", None)
    _require(
        all(candidate.get(key) == value for key, value in expected_fields.items())
        and expected_fields["nonzero_pair_differences"] > 0
        and expected.get("plan_sha256") == builder.CONTENT_FREE_PLAN_SHA256
        and expected.get("direction_seeds")
        == list(builder.CONTENT_FREE_DIRECTION_SEEDS)
        and candidate.get("plan_sha256") == builder.CONTENT_FREE_PLAN_SHA256
        and compiler.get("schema")
        == "eggroll-es-same-live-compiler-equivalence-v73b"
        and compiler.get("same_live_reward_object_used_for_both") is True
        and compiler.get("whole_result_mapping_exact") is True
        and compiler.get("coefficient_sha256")
        == expected.get("coefficient_sha256")
        and compiler.get("direction_count") == 8
        and compiler_sha256 == builder.canonical_sha256(compiler_body)
        and context.population_equivalence.get("signed_reward_sha256")
        == compiler.get("live_signed_reward_sha256")
        and execution.get("schema")
        == "eggroll-es-four-actor-live-update-consensus-v73b"
        and execution.get("actor_count") == 4
        and execution.get("all_actor_identities_exact") is True
        and execution_sha256 == builder.canonical_sha256(execution_body)
        and candidate.get("candidate_master_sha256")
        == execution.get("candidate_master_sha256")
        and candidate.get("candidate_runtime_values_sha256")
        == execution.get("candidate_runtime_values_sha256")
        and candidate.get("manifest_sha256") == invariants.get("manifest_sha256")
        and invariants.get("candidate_master_sha256")
        == candidate.get("candidate_master_sha256")
        and invariants.get("candidate_runtime_values_sha256")
        == candidate.get("candidate_runtime_values_sha256")
        and invariants.get("rank_local_tokens_not_broadcast") is True
        and len(invariants.get("population_acceptance_sha256_by_rank", [])) == 4
        and len(set(invariants["population_acceptance_sha256_by_rank"])) == 4
        and len(invariants.get("update_acceptance_sha256_by_rank", [])) == 4
        and len(set(invariants["update_acceptance_sha256_by_rank"])) == 4
        and isinstance(shards, list)
        and shards == expected_shards
        and candidate.get("candidate_differs_from_master") is True
        and candidate.get("candidate_runtime_differs_from_master") is True
        and candidate.get("master_committed") is False
        and candidate.get("all_four_abort_receipts_exact") is True
        and candidate.get("checkpoint_snapshot_or_promotion_performed") is False
        and candidate.get("protected_dev_ood_or_holdout_opened") is False,
        "V73E content-free update self-consistency changed",
    )
    result = {
        "schema": "qwen36-v73e-content-free-update-consistency-v1",
        "plan_sha256": builder.CONTENT_FREE_PLAN_SHA256,
        "signed_reward_sha256": compiler["live_signed_reward_sha256"],
        "coefficient_sha256": expected["coefficient_sha256"],
        "compiler_equivalence_sha256": compiler_sha256,
        "manifest_sha256": candidate["manifest_sha256"],
        "canonical_and_independent_compilers_exact": True,
        "four_actor_candidate_master_sha256": execution[
            "candidate_master_sha256"
        ],
        "four_actor_candidate_runtime_values_sha256": execution[
            "candidate_runtime_values_sha256"
        ],
        "all_four_abort_receipts_exact": True,
        "historical_reward_update_or_semantic_authority_inherited": False,
        "within_run_four_actor_candidate_and_abort_exact": True,
        "reward_derived_coefficient_manifest_and_candidate_cross_rerun": (
            "content_addressed_diagnostic_only"
        ),
    }
    result["equivalence_sha256"] = builder.canonical_sha256(result)
    return result


def actor_work_self_consistency_v73e(
    candidate_rows, _historical_rows
) -> dict[str, Any]:
    rows = list(candidate_rows)
    plan = expected_content_free_payload_and_plan_v73e()["plan"]
    assignments = [item for wave in plan["waves"] for item in wave]
    coordinate_fields = (
        "wave_index",
        "engine_rank",
        "direction_seed",
        "sign",
        "pair_id",
        "evaluation_contract_sha256",
    )
    expected_by_work_id = {}
    for assignment in assignments:
        coordinate = {
            field: assignment[field] for field in coordinate_fields
        }
        work_id = builder.canonical_sha256(coordinate)
        expected_by_work_id[work_id] = coordinate
    observed_by_work_id = {
        row.get("work_id"): {
            field: row.get(field) for field in coordinate_fields
        }
        for row in rows
    }
    _require(
        isinstance(_CONTENT_FREE_INPUT_EVIDENCE, dict)
        and len(rows) == v73.SIGNED_CANDIDATES_V73
        and len({row.get("work_id") for row in rows})
        == v73.SIGNED_CANDIDATES_V73
        and observed_by_work_id == expected_by_work_id
        and {(row.get("wave_index"), row.get("engine_rank")) for row in rows}
        == {(wave, rank) for wave in range(4) for rank in range(4)}
        and all(
            row.get("schema") == "eggroll-es-actor-cuda-work-receipt-v66d"
            and row.get("evaluation_contract_sha256")
            == _CONTENT_FREE_INPUT_EVIDENCE["evaluation_contract_sha256"]
            and row.get("output_cardinality", {}).get("request_outputs")
            == builder.CONTENT_FREE_TOKEN_PANEL_ROWS
            and row.get("output_cardinality", {}).get("samples")
            == builder.CONTENT_FREE_TOKEN_PANEL_ROWS
            and row.get("output_cardinality", {}).get("generated_tokens")
            == builder.CONTENT_FREE_GENERATED_TOKENS_PER_CANDIDATE
            and row.get("output_cardinality", {}).get("prompt_tokens")
            == builder.CONTENT_FREE_TOTAL_TOKENS_PER_CANDIDATE
            and row.get("receipt_sha256")
            == builder.canonical_sha256({
                key: value
                for key, value in row.items()
                if key != "receipt_sha256"
            })
            for row in rows
        ),
        "V73E content-free actor work self-consistency changed",
    )
    compact = [{
        key: row.get(key) for key in (
            "wave_index", "engine_rank", "direction_seed", "sign",
            "pair_id", "evaluation_contract_sha256", "work_id",
            "output_cardinality",
        )
    } for row in sorted(rows, key=lambda item: item["work_id"])]
    result = {
        "schema": "qwen36-v73e-content-free-actor-work-consistency-v1",
        "receipt_count": len(rows),
        "work_assignment_and_cardinality_complete": True,
        "semantic_rows_present": False,
        "historical_actor_work_authority_inherited": False,
        "sealed_plan_sha256": builder.CONTENT_FREE_PLAN_SHA256,
        "actor_work_sha256": builder.canonical_sha256(compact),
    }
    result["equivalence_sha256"] = builder.canonical_sha256(result)
    return result


def content_free_workload_v73e(
    preregistration: Mapping[str, Any],
) -> dict[str, Any]:
    """Retain only ES mechanics; replace every historical authority surface."""
    recipe = {
        "schema": "qwen36-v73e-content-free-es-mechanics-v1",
        "direction_seeds": list(builder.CONTENT_FREE_DIRECTION_SEEDS),
        "direction_count": 8,
        "signed_population_size": 16,
        "sigma": builder.CONTENT_FREE_SIGMA,
        "learning_rate": builder.CONTENT_FREE_LEARNING_RATE,
        "engine_count": 4,
        "tensor_parallel_size": 1,
        "precision": "bfloat16",
        "candidate_action": "exact_abort_no_commit",
        "input_panel": {
        "schema": "qwen36-v73e-content-free-token-id-panel-v1",
        "content_sha256": builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256,
        "request_count": builder.CONTENT_FREE_TOKEN_PANEL_ROWS,
        "unscored_prefix_tokens_per_candidate": (
            builder.CONTENT_FREE_UNSCORED_PREFIX_TOKENS_PER_CANDIDATE
        ),
        "total_tokens_per_candidate": (
            builder.CONTENT_FREE_TOTAL_TOKENS_PER_CANDIDATE
        ),
        "answer_tokens_per_candidate": (
            builder.CONTENT_FREE_ANSWER_TOKENS_PER_CANDIDATE
        ),
        "minimum_total_tokens_per_request": 74,
        "maximum_total_tokens_per_request": 76,
        "per_request_length_manifest_sha256": (
            builder.CONTENT_FREE_LENGTH_MANIFEST_SHA256
        ),
        "historical_semantic_authority": False,
        },
        "reward": builder.content_free_suffix_reward_config_v73e(),
        "historical_reward_population_update_or_semantic_authority": False,
    }
    rejected = {
        "schema": "qwen36-v73e-historical-authority-rejected-v1",
        "semantic_anchor": {"actor_work_id_sha256": "0" * 64},
        "population_reward_or_identity_authority": False,
        "update_or_candidate_identity_authority": False,
        "actor_work_identity_authority": False,
    }
    result = {
        "schema": "qwen36-v73e-content-free-systems-workload-v1",
        "status": "prospective_no_commit_systems_measurement_only",
        "fixed_recipe": recipe,
        "runtime": {
            "schema": "qwen36-v73e-existing-tp1-runtime-mechanics-v1",
            "tuned_table_content_sha256": (
                builder.CONTENT_FREE_TP1_TUNED_TABLE_CONTENT_SHA256
            ),
            "worker_extension": builder.WORKER_EXTENSION,
            "four_tp1_engines": True,
            "exclusive_physical_gpus": [0, 1, 2, 3],
            "historical_semantic_authority": False,
        },
        "accepted_v66d_control": rejected,
        "failed_v73_observation": {
            "authority": False,
            "retained_as_mechanics_ancestry": False,
        },
        "content_free_token_panel_content_sha256": (
            builder.CONTENT_FREE_TOKEN_PANEL_CONTENT_SHA256
        ),
        "v73e_preregistration_content_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "historical_same_live_semantic_or_equivalence_authority": False,
    }
    result["content_sha256_before_self_field"] = builder.canonical_sha256(
        result
    )
    return result


def content_free_control_v73e() -> dict[str, Any]:
    """Shape-only object for inherited hooks whose comparisons are replaced."""
    return {
        "schema": "qwen36-v73e-no-historical-control-authority-v1",
        "population": {},
        "update": {
            "candidate_master_sha256": "0" * 64,
            "candidate_runtime_values_sha256": "0" * 64,
        },
        "actor_rows": [],
        "historical_reward_population_update_or_semantic_authority": False,
    }


def load_preregistration_v73e(args) -> dict[str, Any]:
    path = Path(args.preregistration).resolve()
    _require(
        builder.file_sha256(path) == args.preregistration_sha256,
        "V73E preregistration file identity changed",
    )
    value = _load_json(path)
    _validate_self_hash(value, "preregistration")
    builder.validate_generated_preregistration_v73e(value)
    _require(
        value.get("content_sha256_before_self_field")
        == args.preregistration_content_sha256
        and value.get("status")
        == "sealed_cpu_only_before_v73e_model_ray_gpu_or_protected_access",
        "V73E preregistration content changed",
    )
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve()
    result = dict(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = builder.canonical_sha256(result)
    payload = (
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-v73e-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return result


def _prospective_json_reference_v73e(
    path: Path, value: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Self-hash JSON in memory so its file can be published last."""
    result = dict(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = builder.canonical_sha256(
        result
    )
    payload = (
        json.dumps(
            result,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    reference = {
        "path": str(Path(path).resolve()),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": result["content_sha256_before_self_field"],
    }
    return result, reference


def _rewrite_json(path: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    value = _load_json(path)
    value.update(dict(updates))
    return _atomic_json(path, value)


def silence_target_streams_v73e() -> None:
    """Keep Nsight's mandatory ProcessStreams payload empty before workload."""
    sys.stdout.flush()
    sys.stderr.flush()
    descriptor = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(descriptor, 1)
        os.dup2(descriptor, 2)
    finally:
        os.close(descriptor)


def _path_guard_v73e():
    module = importlib.import_module("v73e_path_open_guard")
    _require(
        module.installed()
        and module.installation_mechanism() == "controller_sitecustomize"
        and os.environ.get("SPECIALIST_V73E_SYSTEMS_ONLY_GUARD") == "1"
        and os.environ.get("SPECIALIST_V73E_CONTROLLER_GUARD_PID")
        == str(os.getpid())
        and Path(module.__file__).resolve() == builder.GUARD,
        "V73E systems-only path guard was not installed before imports",
    )
    return module


def _validate_guard_process_receipt_v73e(
    value: Mapping[str, Any], expected_mechanism: str
) -> dict[str, Any]:
    row = dict(value)
    claimed = row.pop("receipt_sha256", None)
    _require(
        claimed == builder.canonical_sha256(row)
        and row.get("schema")
        == "qwen36-v73e-systems-only-path-guard-process-v1"
        and row.get("installed_before_runtime_imports") is True
        and row.get("installation_mechanism") == expected_mechanism
        and row.get("installation_pid") == row.get("pid")
        and row.get("boundary_registry_file_sha256")
        == builder.BOUNDARY_REGISTRY_FILE_SHA256
        and row.get("boundary_registry_content_sha256")
        == builder.BOUNDARY_REGISTRY_CONTENT_SHA256
        and row.get("exact_path_identity_count") == 3
        and row.get("prefix_identity_count") == 2
        and row.get("successful_protected_opens") == 0
        and row.get("successful_protected_resolves") == 0
        and row.get("successful_protected_metadata") == 0
        and row.get("successful_protected_enumerations") == 0
        and row.get("protected_path_values_persisted") is False
        and row.get("quality_hpo_or_promotion_authorized") is False,
        "V73E systems-only process guard receipt changed",
    )
    return dict(value)


def _validate_worker_bootstrap_receipt_v73e(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    row = dict(value)
    claimed = row.pop("receipt_sha256", None)
    pre_parent_guard = _validate_guard_process_receipt_v73e(
        row.get("pre_parent_guard_receipt", {}),
        "ray_actor_worker_extension_pre_parent_import",
    )
    guard = _validate_guard_process_receipt_v73e(
        row.get("guard_process_receipt", {}),
        "ray_actor_worker_extension_pre_parent_import",
    )
    staged_install = row.get("staged_inverse_install", {})
    staged_proof = _validate_inverse_transform_proof_v73e(
        staged_install.get("inverse_transform_proof", {})
    )
    _require(
        claimed == builder.canonical_sha256(row)
        and row.get("schema")
        == "qwen36-v73e-worker-bootstrap-receipt-v1"
        and row.get("pid") == guard.get("pid")
        and row.get("process_role") == "ray_actor_worker_extension"
        and row.get("bootstrap_mechanism")
        == "ray_actor_worker_extension_pre_parent_import"
        and row.get("guard_was_preinstalled") is False
        and row.get("parent_modules_absent_before_guard_install") is True
        and row.get(
            "historical_reference_modules_absent_before_guard_install"
        )
        is True
        and row.get("historical_reference_module_identity_count") == 3
        and isinstance(
            row.get("historical_reference_module_identity_set_sha256"), str
        )
        and row["historical_reference_module_identity_set_sha256"]
        == builder.HISTORICAL_REFERENCE_MODULE_IDENTITY_SET_SHA256
        and row.get("parent_module_count_after_guard_install") == 3
        and row.get("guard_source_sha256")
        == builder.file_sha256(builder.GUARD)
        and row.get("actor_bootstrap_env_exact") is True
        and row.get("pre_parent_guard_receipt_sha256")
        == pre_parent_guard.get("receipt_sha256")
        and pre_parent_guard.get("pid") == guard.get("pid")
        and row.get("staged_inverse_install_complete") is True
        and staged_install.get("schema")
        == "qwen36-v73e-staged-inverse-install-v1"
        and staged_install.get("complete") is True
        and staged_install.get("staged_weights_sha256")
        == builder.STAGED_ADAPTER_WEIGHTS_SHA256
        and staged_install.get("staged_config_sha256")
        == builder.STAGED_ADAPTER_CONFIG_SHA256
        and staged_install.get("canonical_master_sha256")
        == builder.CANONICAL_MASTER_SHA256
        and staged_install.get("canonical_runtime_values_sha256")
        == builder.CANONICAL_RUNTIME_VALUES_SHA256
        and staged_proof.get("canonical_master_sha256")
        == builder.CANONICAL_MASTER_SHA256
        and staged_install.get(
            "historical_protected_source_opened_resolved_statted_or_hashed"
        ) is False
        and row.get("quality_hpo_or_promotion_authorized") is False,
        "V73E Ray actor worker bootstrap receipt changed",
    )
    return dict(value)


def _guard_evidence_v73e(worker_rows) -> dict[str, Any]:
    _require(
        isinstance(_RAY_BOOTSTRAP_EVIDENCE, dict),
        "V73E Ray runtime-env bootstrap evidence is absent",
    )
    controller = _validate_guard_process_receipt_v73e(
        _path_guard_v73e().receipt(), "controller_sitecustomize"
    )
    workers = [
        _validate_worker_bootstrap_receipt_v73e(row) for row in worker_rows
    ]
    worker_guards = [row["guard_process_receipt"] for row in workers]
    _require(
        len(workers) == 4
        and len({row["pid"] for row in workers}) == 4
        and controller["pid"] not in {row["pid"] for row in workers}
        and len({row["exact_path_identity_set_sha256"] for row in worker_guards})
        == 1
        and worker_guards[0]["exact_path_identity_set_sha256"]
        == controller["exact_path_identity_set_sha256"]
        and len({row["prefix_identity_set_sha256"] for row in worker_guards})
        == 1
        and worker_guards[0]["prefix_identity_set_sha256"]
        == controller["prefix_identity_set_sha256"],
        "V73E systems-only worker guard coverage changed",
    )
    processes = [controller, *worker_guards]
    return {
        "schema": "qwen36-v73e-systems-only-path-guard-receipt-v1",
        "status": (
            "zero_successful_protected_open_resolve_metadata_or_"
            "enumeration_systems_only"
        ),
        "controller": controller,
        "workers": workers,
        "controller_bootstrap_mechanism": "controller_sitecustomize",
        "actor_bootstrap_mechanism": (
            "ray_actor_worker_extension_pre_parent_import"
        ),
        "mechanisms_are_distinct": True,
        "ray_job_runtime_env_bootstrap": copy.deepcopy(
            _RAY_BOOTSTRAP_EVIDENCE
        ),
        "process_count": len(processes),
        "successful_protected_opens": sum(
            row["successful_protected_opens"] for row in processes
        ),
        "successful_protected_resolves": sum(
            row["successful_protected_resolves"] for row in processes
        ),
        "successful_protected_metadata": sum(
            row["successful_protected_metadata"] for row in processes
        ),
        "successful_protected_enumerations": sum(
            row["successful_protected_enumerations"] for row in processes
        ),
        "denied_protected_open_attempts": sum(
            row["protected_open_attempts_denied"] for row in processes
        ),
        "denied_protected_resolve_attempts": sum(
            row["protected_resolve_attempts_denied"] for row in processes
        ),
        "protected_path_values_persisted": False,
        "quality_hpo_or_promotion_performed": False,
        "lineage_rehabilitation_performed": False,
    }


class _PhaseRangeLedgerV73E:
    def __init__(self, domain):
        self.domain = domain
        self.rows: list[dict[str, Any]] = []
        self.active: dict[str, Any] | None = None
        self.controller_pid = os.getpid()

    def open(self, phase: str, epoch: int) -> None:
        _require(self.active is None, "V73E NVTX phase range nested unexpectedly")
        _require(
            isinstance(phase, str) and phase in builder.PHASES,
            f"V73E unsupported phase: {phase}",
        )
        started_ns = time.monotonic_ns()
        self.domain.push_range(message=phase, color="blue")
        self.active = {
            "phase": phase,
            "epoch": int(epoch),
            "controller_pid": self.controller_pid,
            "started_monotonic_ns": started_ns,
        }

    def close(self) -> None:
        if self.active is None:
            return
        ended_ns = time.monotonic_ns()
        self.domain.pop_range()
        row = dict(self.active)
        row["ended_monotonic_ns"] = ended_ns
        row["elapsed_ns"] = ended_ns - row["started_monotonic_ns"]
        _require(row["elapsed_ns"] > 0, "V73E NVTX phase range was empty")
        self.rows.append(row)
        self.active = None


def phase_class_v73e(base_phase_class, domain):
    """Build the additive phase class; injectable for CPU-only tests."""

    class ExactPhaseHandshakeV73E(base_phase_class):
        def __init__(self, *args, **kwargs):
            global _PHASE_INSTANCE
            super().__init__(*args, **kwargs)
            _require(_PHASE_INSTANCE is None, "V73E phase instance duplicated")
            self.v73e_ledger = _PhaseRangeLedgerV73E(domain)
            _PHASE_INSTANCE = self
            phase, epoch = self.snapshot()
            self.v73e_ledger.open(phase, epoch)

        @property
        def value(self):
            return base_phase_class.value.fget(self)

        @value.setter
        def value(self, phase):
            current, _ = self.snapshot()
            if phase == current:
                return
            self.v73e_ledger.close()
            base_phase_class.value.fset(self, phase)
            observed, epoch = self.snapshot()
            _require(observed == phase, "V73E phase transition did not publish")
            self.v73e_ledger.open(observed, epoch)

        def close_v73e_range(self):
            self.v73e_ledger.close()

    ExactPhaseHandshakeV73E.__name__ = "ExactPhaseHandshakeV73E"
    return ExactPhaseHandshakeV73E


def _v73b_artifact_mapping(paths: Mapping[str, str]) -> dict[str, str]:
    return {
        "attempt": paths["application_attempt"],
        "run_directory": paths["run_directory"],
        "gpu_log": paths["gpu_log"],
        "actor_cuda_work_log": paths["actor_cuda_work_log"],
        "host_process_samples": paths["host_process_samples"],
        "host_process_summary": paths["host_process_summary"],
        "population": paths["population"],
        "update": paths["update"],
        "audit_traffic": paths["audit_traffic"],
        "equivalence": paths["equivalence"],
        "report": paths["report"],
        "failure": paths["failure"],
    }


def ray_actor_bootstrap_runtime_env_v73e(
    runtime_env: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a copied Ray job runtime-env with the actor guard injected."""
    merged = copy.deepcopy(runtime_env or {})
    _require(
        isinstance(merged, dict),
        "V73E Ray runtime_env must be a mapping",
    )
    env_vars = copy.deepcopy(merged.pop("env_vars", {}) or {})
    _require(
        isinstance(env_vars, dict)
        and all(isinstance(key, str) and isinstance(value, str)
                for key, value in env_vars.items())
        and builder.ACTOR_BOOTSTRAP_ENV not in env_vars
        and builder.ACTOR_GUARD_SHA_ENV not in env_vars,
        "V73E Ray actor bootstrap env collided",
    )
    guard_sha256 = builder.file_sha256(builder.GUARD)
    env_vars.update({
        builder.ACTOR_BOOTSTRAP_ENV: "1",
        builder.ACTOR_GUARD_SHA_ENV: guard_sha256,
    })
    merged["env_vars"] = env_vars
    evidence = {
        "schema": "qwen36-v73e-ray-job-runtime-env-bootstrap-v1",
        "injected_before_ray_init": True,
        "job_env_merges_with_actor_specific_runtime_env": True,
        "actor_bootstrap_env_name": builder.ACTOR_BOOTSTRAP_ENV,
        "actor_guard_sha_env_name": builder.ACTOR_GUARD_SHA_ENV,
        "actor_guard_file_sha256": guard_sha256,
        "environment_values_persisted_beyond_fixed_flags_and_hash": False,
    }
    return merged, evidence


@contextmanager
def patched_live_v73e(
    preregistration: Mapping[str, Any],
    workload: Mapping[str, Any],
    control: Mapping[str, Any],
    arm: str,
    *,
    nvtx_domain=None,
):
    """Patch only the new process and restore every inherited global."""
    import ray
    import run_lora_topology_probe_v40a as v40a
    import run_lora_es_nested_population_v52 as runtime52
    import run_lora_es_generation_boundary_v48b as v48b

    global _PHASE_INSTANCE, _GUARD_EVIDENCE, _GUARD_FAILURE
    global _RAY_BOOTSTRAP_EVIDENCE, _STAGED_ADAPTER_EVIDENCE
    _require(_PHASE_INSTANCE is None, "V73E live patch is not reentrant")
    if nvtx_domain is None:
        import nvtx

        nvtx_domain = nvtx.Domain(builder.PHASE_DOMAIN)
    paths = preregistration["arms"][arm]["artifacts"]
    mapped = _v73b_artifact_mapping(paths)
    replacements = {
        "PREREGISTRATION": builder.OUTPUT,
        "RUN_DIR": Path(mapped["run_directory"]),
        "_ARTIFACTS": mapped,
        "ATTEMPT": Path(mapped["attempt"]),
        "GPU_LOG": Path(mapped["gpu_log"]),
        "GPU_WORK_LOG": Path(mapped["actor_cuda_work_log"]),
        "HOST_SAMPLES": Path(mapped["host_process_samples"]),
        "HOST_SUMMARY": Path(mapped["host_process_summary"]),
        "POPULATION": Path(mapped["population"]),
        "UPDATE": Path(mapped["update"]),
        "AUDIT_TRAFFIC": Path(mapped["audit_traffic"]),
        "EQUIVALENCE": Path(mapped["equivalence"]),
        "REPORT": Path(mapped["report"]),
        "FAILURE": Path(mapped["failure"]),
    }
    saved_v73b = {name: getattr(v73b, name) for name in replacements}
    base_phase = v73.CompatiblePhaseHandshakeV73
    base_abort = v73._execute_and_abort_nonzero_update_v73
    base_worker_extension = v73.WORKER_EXTENSION_V73
    base_cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a
    base_ray_init = ray.init
    base_verify_adapter = runtime52.verify_adapter_contract_v52
    base_activate_install = v66._activate_install_and_certify_v66
    base_load_generation_panel = runtime52.load_generation_panel_v52
    base_prepare_inputs = v66._prepare_train_only_inputs_v66
    base_population_equivalence = v73b.population_equivalence_v73b
    base_update_equivalence = v73b.update_equivalence_v73b
    base_actor_work_equivalence = v73.actor_work_equivalence_v73
    base_suffix_scorer = v48b.v43i.anchor_v4.score_gold_answer_outputs_v4
    phase_class = phase_class_v73e(base_phase, nvtx_domain)

    def verify_staged_adapter_once_v73e():
        global _STAGED_ADAPTER_EVIDENCE
        _require(
            _STAGED_ADAPTER_EVIDENCE is None,
            "V73E staged adapter verification was invoked more than once",
        )
        _STAGED_ADAPTER_EVIDENCE = verify_staged_adapter_contract_v73e()
        return copy.deepcopy(_STAGED_ADAPTER_EVIDENCE)

    def activate_staged_adapter_v73e(trainer, prior, runtime_v40a, phase):
        with staged_activation_alias_v73e(prior):
            return base_activate_install(
                trainer, prior, runtime_v40a, phase
            )

    def ray_init_v73e(*args, **kwargs):
        global _RAY_BOOTSTRAP_EVIDENCE
        _require(
            _RAY_BOOTSTRAP_EVIDENCE is None,
            "V73E Ray bootstrap injection was invoked more than once",
        )
        runtime_env, evidence = ray_actor_bootstrap_runtime_env_v73e(
            kwargs.pop("runtime_env", {}) or {}
        )
        kwargs["runtime_env"] = runtime_env
        _RAY_BOOTSTRAP_EVIDENCE = evidence
        return base_ray_init(*args, **kwargs)

    def abort_then_final_audit_v73e(*args, **kwargs):
        value = base_abort(*args, **kwargs)
        context = v73._ACTIVE_CONTEXT_V73
        _require(context is not None and context.phase is not None,
                 "V73E abort phase context disappeared")
        context.phase.value = "post_abort_final_audit_all_actors"
        return value

    def cleanup_phase_v73e(trainer):
        global _GUARD_EVIDENCE, _GUARD_FAILURE
        context = v73._ACTIVE_CONTEXT_V73
        if context is not None and context.phase is not None:
            context.phase.value = "cleanup_all_actors"
        try:
            worker_rows = v73._rpc_all_v73(
                trainer, "systems_only_path_guard_receipt_v73e"
            )
            _GUARD_EVIDENCE = _guard_evidence_v73e(worker_rows)
        except BaseException as error:
            _GUARD_FAILURE = error
        return base_cleanup(trainer)

    for name, value in replacements.items():
        setattr(v73b, name, value)
    v73.CompatiblePhaseHandshakeV73 = phase_class
    v73.WORKER_EXTENSION_V73 = builder.WORKER_EXTENSION
    v73._execute_and_abort_nonzero_update_v73 = abort_then_final_audit_v73e
    v40a.cleanup_v38a.strict_close_trainer_v38a = cleanup_phase_v73e
    ray.init = ray_init_v73e
    runtime52.verify_adapter_contract_v52 = verify_staged_adapter_once_v73e
    runtime52.load_generation_panel_v52 = content_free_generation_panel_v73e
    v66._prepare_train_only_inputs_v66 = prepare_content_free_inputs_v73e
    v66._activate_install_and_certify_v66 = activate_staged_adapter_v73e
    v73b.population_equivalence_v73b = population_self_consistency_v73e
    v73b.update_equivalence_v73b = update_self_consistency_v73e
    v73.actor_work_equivalence_v73 = actor_work_self_consistency_v73e
    v48b.v43i.anchor_v4.score_gold_answer_outputs_v4 = (
        score_content_free_suffix_outputs_v73e
    )
    try:
        with v73b.patched_live_v73b(workload, control) as context:
            yield context
    finally:
        if _PHASE_INSTANCE is not None:
            _PHASE_INSTANCE.close_v73e_range()
        v40a.cleanup_v38a.strict_close_trainer_v38a = base_cleanup
        ray.init = base_ray_init
        runtime52.verify_adapter_contract_v52 = base_verify_adapter
        runtime52.load_generation_panel_v52 = base_load_generation_panel
        v66._prepare_train_only_inputs_v66 = base_prepare_inputs
        v66._activate_install_and_certify_v66 = base_activate_install
        v73b.population_equivalence_v73b = base_population_equivalence
        v73b.update_equivalence_v73b = base_update_equivalence
        v73.actor_work_equivalence_v73 = base_actor_work_equivalence
        v48b.v43i.anchor_v4.score_gold_answer_outputs_v4 = base_suffix_scorer
        v73._execute_and_abort_nonzero_update_v73 = base_abort
        v73.CompatiblePhaseHandshakeV73 = base_phase
        v73.WORKER_EXTENSION_V73 = base_worker_extension
        for name, value in saved_v73b.items():
            setattr(v73b, name, value)


def phase_receipt_v73e(arm: str, *, complete: bool) -> dict[str, Any]:
    instance = _PHASE_INSTANCE
    rows = [] if instance is None else list(instance.v73e_ledger.rows)
    observed = [row["phase"] for row in rows]
    if complete:
        _require(
            observed == list(builder.PHASES),
            f"V73E phase sequence changed: {observed}",
        )
        _require(
            all(
                right["started_monotonic_ns"] >= left["ended_monotonic_ns"]
                for left, right in zip(rows, rows[1:])
            ),
            "V73E phase ranges overlapped",
        )
    result = {
        "schema": "eggroll-es-exact-phase-range-receipt-v73e",
        "arm": arm,
        "complete": bool(complete),
        "nvtx_domain": builder.PHASE_DOMAIN,
        "expected_phase_order": list(builder.PHASES),
        "observed_phase_order": observed,
        "phase_count": len(rows),
        "one_controller_pid": len({row["controller_pid"] for row in rows}) <= 1,
        "rows": rows,
        "contains_prompts_questions_answers_or_outputs": False,
    }
    return result


def _reference(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve()),
        "file_sha256": builder.file_sha256(path),
        "content_sha256": value["content_sha256_before_self_field"],
    }


def gpu_time_accounting_v73e(
    legacy_compute: Mapping[str, Any] | None,
) -> dict[str, Any]:
    legacy = dict(legacy_compute or {})
    return {
        "schema": "qwen36-v73e-gpu-time-accounting-split-v1",
        "reserved_wall_gpu_seconds": legacy.get("charged_gpu_seconds"),
        "reserved_wall_source": "legacy_compute_ledger_charged_gpu_seconds",
        "legacy_estimated_model_allocation_or_residency_seconds_per_gpu": (
            legacy.get("model_allocation_or_residency_seconds_per_gpu")
        ),
        "legacy_residency_estimate_classification": (
            "diagnostic_not_directly_measured_not_accepted"
        ),
        "model_resident_gpu_seconds_measured": None,
        "useful_gpu_seconds_measured": None,
        "promotion_charged_gpu_seconds": 0,
        "reserved_wall_is_not_model_residency_or_useful_work": True,
        "profiled_event_time_is_not_reclassified_as_unprofiled_useful_time": (
            True
        ),
    }


def finalize_success_v73e(
    context,
    preregistration: Mapping[str, Any],
    arm: str,
    command_sha256: str,
) -> dict[str, Any]:
    paths = preregistration["arms"][arm]["artifacts"]
    _require(
        _GUARD_FAILURE is None
        and isinstance(_GUARD_EVIDENCE, dict)
        and isinstance(_RAY_BOOTSTRAP_EVIDENCE, dict)
        and isinstance(_STAGED_ADAPTER_EVIDENCE, dict)
        and isinstance(_CONTENT_FREE_INPUT_EVIDENCE, dict),
        "V73E systems-only path guard evidence is incomplete",
    )
    guard_path = Path(paths["path_guard_receipt"])
    guard = _atomic_json(guard_path, _GUARD_EVIDENCE)
    phase_path = Path(paths["phase_receipt"])
    phase, phase_reference = _prospective_json_reference_v73e(
        phase_path, phase_receipt_v73e(arm, complete=True)
    )
    v73b.finalize_success_artifacts_v73b(context)

    population_path = Path(paths["population"])
    population = _load_json(population_path)
    population_consistency = population.pop(
        "same_live_population_equivalence", None
    )
    _require(
        population_consistency == context.population_equivalence
        and population_consistency.get("schema")
        == "qwen36-v73e-content-free-population-consistency-v1",
        "V73E population consistency receipt changed",
    )
    population.update({
        "schema": "v73e-v71-v72-qwen36-population-evidence",
        "profile_arm": arm,
        "content_free_population_consistency": population_consistency,
        "content_free_input": copy.deepcopy(_CONTENT_FREE_INPUT_EVIDENCE),
        "historical_semantic_reward_population_or_equivalence_authority": False,
    })
    population = _atomic_json(population_path, population)
    update_path = Path(paths["update"])
    update = _load_json(update_path)
    update_consistency = update.pop("same_live_update_equivalence", None)
    transient_compiler = update.pop("same_live_compiler_equivalence", None)
    _require(
        update_consistency == context.update_equivalence
        and update_consistency.get("schema")
        == "qwen36-v73e-content-free-update-consistency-v1"
        and isinstance(transient_compiler, dict)
        and transient_compiler.get("whole_result_mapping_exact") is True,
        "V73E update consistency receipt changed",
    )
    compiler_consistency = {
        "schema": "qwen36-v73e-current-run-dual-compiler-consistency-v1",
        "current_run_signed_reward_sha256": transient_compiler[
            "live_signed_reward_sha256"
        ],
        "coefficient_sha256": transient_compiler["coefficient_sha256"],
        "direction_count": transient_compiler["direction_count"],
        "canonical_and_independent_compiler_whole_mapping_exact": True,
        "historical_reward_or_update_authority_inherited": False,
        "floating_reward_or_derived_update_cross_rerun_equality_required": False,
    }
    compiler_consistency["consistency_sha256"] = builder.canonical_sha256(
        compiler_consistency
    )
    update.update({
        "schema": "v73e-v71-v72-qwen36-update-evidence",
        "profile_arm": arm,
        "population_content_sha256": population[
            "content_sha256_before_self_field"
        ],
        "content_free_update_consistency": update_consistency,
        "content_free_current_run_dual_compiler_consistency": (
            compiler_consistency
        ),
        "historical_semantic_reward_update_or_equivalence_authority": False,
    })
    update = _atomic_json(update_path, update)
    equivalence_path = Path(paths["equivalence"])
    transient_equivalence = _load_json(equivalence_path)
    actor_work = transient_equivalence.get("actor_work", {})
    _require(
        actor_work.get("schema")
        == "qwen36-v73e-content-free-actor-work-consistency-v1",
        "V73E actor work consistency receipt changed",
    )
    equivalence = _atomic_json(equivalence_path, {
        "schema": "qwen36-v73e-content-free-systems-consistency-v1",
        "profile_arm": arm,
        "content_free_input": copy.deepcopy(_CONTENT_FREE_INPUT_EVIDENCE),
        "population": population_consistency,
        "update": update_consistency,
        "current_run_dual_compiler": compiler_consistency,
        "actor_work": actor_work,
        "stable_cross_rerun_identity_policy": {
            "token_panel_evaluation_contract_and_plan_exact": True,
            "candidate_perturbation_inventory_and_restore_exact": True,
            "floating_reward_coefficient_manifest_and_update_candidate": (
                "content_addressed_diagnostic_only"
            ),
        },
        "accepted_historical_control_or_equivalence_present": False,
        "historical_semantic_reward_population_update_or_actor_authority": False,
        "master_committed": False,
    })
    report_path = Path(paths["report"])
    report = _load_json(report_path)
    gpu_time_accounting = gpu_time_accounting_v73e(
        report.get("compute_ledger", {})
    )
    report.update({
        "schema": "v73e-exact-phase-qwen36-calibration-report",
        "status": (
            "complete_content_free_self_consistent_no_commit_"
            "awaiting_parent_trace_analysis"
        ),
        "beads": [
            "specialist-0j5.32",
            "specialist-nen.33",
            "specialist-nen.34",
            "specialist-nen.35",
        ],
        "profile_arm": arm,
        "expanded_profiler_command_sha256": command_sha256,
        "population": _reference(population_path, population),
        "nonzero_update": _reference(update_path, update),
        "content_free_systems_consistency": _reference(
            equivalence_path, equivalence
        ),
        "content_free_input": copy.deepcopy(_CONTENT_FREE_INPUT_EVIDENCE),
        "exact_phase_receipt": phase_reference,
        "systems_only_path_guard": _reference(guard_path, guard),
        "ray_actor_guard_bootstrap": copy.deepcopy(
            _RAY_BOOTSTRAP_EVIDENCE
        ),
        "staged_only_adapter_bootstrap": copy.deepcopy(
            _STAGED_ADAPTER_EVIDENCE
        ),
        "immutable_v73d_attempt_1_predecessor": preregistration[
            "immutable_v73d_attempt_1_predecessor"
        ],
        "gpu_time_accounting_policy": {
            "reserved_wall_gpu_seconds_reported_separately": True,
            "model_resident_gpu_seconds_require_direct_evidence": True,
            "useful_gpu_seconds_require_direct_evidence": True,
            "promotion_charged_gpu_seconds": 0,
        },
        "gpu_time_accounting": gpu_time_accounting,
        "parent_trace_output_finalizes_after_target_process_exit": True,
        "semantic_quality_selection_or_hpo_performed": False,
        "raw_prompts_questions_answers_or_outputs_persisted": False,
        "checkpoint_snapshot_or_promotion_performed": False,
        "protected_dev_ood_or_holdout_opened": False,
        "successful_protected_path_opens_or_resolves": 0,
        "successful_protected_open_resolve_metadata_or_enumeration": 0,
        "quality_hpo_promotion_or_lineage_rehabilitation_performed": False,
        "historical_semantic_reward_population_update_or_equivalence_authority": (
            False
        ),
        "cross_rerun_floating_reward_or_derived_update_identity_gate": False,
    })
    for stale_key in (
        "same_live_equivalence",
        "historical_reward_floats_used_as_acceptance_gate",
        "same_live_reward_vector_used_by_both_compilers",
        "same_live_compiler_output_whole_mapping_exact",
        "accepted_v66d_equivalence",
    ):
        report.pop(stale_key, None)
    report = _atomic_json(report_path, report)

    attempt_path = Path(paths["application_attempt"])
    if attempt_path.is_file():
        attempt = _load_json(attempt_path)
        attempt.update({
            "schema": "v73e-exact-phase-qwen36-calibration-attempt",
            "status": "target_complete_parent_trace_analysis_pending",
            "profile_arm": arm,
            "expanded_profiler_command_sha256": command_sha256,
            "ray_actor_guard_bootstrap": copy.deepcopy(
                _RAY_BOOTSTRAP_EVIDENCE
            ),
            "staged_only_adapter_bootstrap": copy.deepcopy(
                _STAGED_ADAPTER_EVIDENCE
            ),
            "content_free_input": copy.deepcopy(
                _CONTENT_FREE_INPUT_EVIDENCE
            ),
            "historical_semantic_or_equivalence_authority": False,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_or_promotion_authorized": False,
        })
        attempt.pop("accepted_v66d_control", None)
        _atomic_json(attempt_path, attempt)
    _atomic_json(phase_path, phase)
    return report


def finalize_failure_v73e(
    preregistration: Mapping[str, Any],
    arm: str,
    command_sha256: str,
) -> None:
    paths = preregistration["arms"][arm]["artifacts"]
    run = Path(paths["run_directory"])
    if run.is_dir():
        phase_path = Path(paths["phase_receipt"])
        if not phase_path.exists():
            _atomic_json(phase_path, phase_receipt_v73e(arm, complete=False))
        else:
            observed_phase = _load_json(phase_path)
            if observed_phase.get("complete") is True:
                _atomic_json(
                    phase_path, phase_receipt_v73e(arm, complete=False)
                )
        guard_path = Path(paths["path_guard_receipt"])
        if not guard_path.exists() and isinstance(_GUARD_EVIDENCE, dict):
            _atomic_json(guard_path, _GUARD_EVIDENCE)
    failure_path = Path(paths["failure"])
    if failure_path.is_file():
        failure = _load_json(failure_path)
        legacy_compute = failure.get("compute_ledger", {})
        failure.update({
            "schema": "v73e-exact-phase-qwen36-calibration-failure",
            "profile_arm": arm,
            "expanded_profiler_command_sha256": command_sha256,
            "content_free_self_consistency_required": True,
            "historical_semantic_reward_population_update_or_equivalence_authority": (
                False
            ),
            "cross_rerun_floating_reward_or_derived_update_identity_gate": False,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_or_promotion_performed": False,
            "immutable_v73d_attempt_1_predecessor": preregistration[
                "immutable_v73d_attempt_1_predecessor"
            ],
            "ray_actor_guard_bootstrap": copy.deepcopy(
                _RAY_BOOTSTRAP_EVIDENCE
            ),
            "staged_only_adapter_bootstrap": copy.deepcopy(
                _STAGED_ADAPTER_EVIDENCE
            ),
            "gpu_time_accounting": gpu_time_accounting_v73e(legacy_compute),
        })
        if isinstance(_CONTENT_FREE_INPUT_EVIDENCE, dict):
            failure["content_free_input"] = copy.deepcopy(
                _CONTENT_FREE_INPUT_EVIDENCE
            )
        for stale_key in (
            "accepted_v66d_control",
            "immutable_v73c_attempt_1_predecessor",
            "same_live_equivalence_required",
            "historical_reward_floats_used_as_acceptance_gate",
        ):
            failure.pop(stale_key, None)
        if Path(paths["phase_receipt"]).is_file():
            phase = _load_json(Path(paths["phase_receipt"]))
            failure["exact_phase_receipt"] = _reference(
                Path(paths["phase_receipt"]), phase
            )
        _atomic_json(failure_path, failure)


def parser_v73e() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(builder.OUTPUT))
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--arm", choices=("timeline", "hbm_metrics"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv=None) -> int:
    args = parser_v73e().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("V73E target requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v73e(args)
    arm = preregistration["arms"][args.arm]
    expected_command = builder.expand_command_v73e(
        args.arm,
        args.preregistration_sha256,
        args.preregistration_content_sha256,
    )
    command_sha256 = builder.canonical_sha256(expected_command)
    if args.dry_run:
        print(json.dumps({
            "schema": preregistration["schema"],
            "arm": args.arm,
            "arm_status": arm["status"],
            "expected_artifacts": arm["artifacts"],
            "expanded_command_sha256": command_sha256,
            "phase_domain": builder.PHASE_DOMAIN,
            "phase_count": len(builder.PHASES),
            "train_semantics_model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "protected_dev_ood_or_holdout_opened": False,
        }, sort_keys=True))
        return 0
    _require(
        arm["launch_authorized_by_this_file_after_identity_checks"] is True,
        "V73E HBM metrics arm is blocked before model/Ray/GPU work",
    )
    _require(
        Path(sys.executable).absolute() == builder.REQUIRED_PYTHON,
        f"V73E target requires {builder.REQUIRED_PYTHON}",
    )
    _require(
        os.environ.get(COMMAND_ATTESTATION_ENV) == command_sha256,
        "V73E expanded profiler command attestation changed",
    )
    silence_target_streams_v73e()
    _path_guard_v73e()
    workload = content_free_workload_v73e(preregistration)
    control = content_free_control_v73e()
    try:
        with patched_live_v73e(
            preregistration, workload, copy.deepcopy(control), args.arm
        ) as context:
            result = v66.execute_v66(workload, args)
            _require(
                _GUARD_FAILURE is None and isinstance(_GUARD_EVIDENCE, dict),
                "V73E systems-only path guard receipt was not captured",
            )
            if _PHASE_INSTANCE is not None:
                _PHASE_INSTANCE.close_v73e_range()
            finalize_success_v73e(
                context, preregistration, args.arm, command_sha256
            )
            return result
    except BaseException:
        finalize_failure_v73e(preregistration, args.arm, command_sha256)
        raise
    finally:
        globals()["_PHASE_INSTANCE"] = None
        globals()["_GUARD_EVIDENCE"] = None
        globals()["_GUARD_FAILURE"] = None
        globals()["_RAY_BOOTSTRAP_EVIDENCE"] = None
        globals()["_STAGED_ADAPTER_EVIDENCE"] = None
        globals()["_CONTENT_FREE_INPUT_EVIDENCE"] = None


if __name__ == "__main__":
    raise SystemExit(main())
