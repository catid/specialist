#!/usr/bin/env python3
"""Pure numeric contracts for the exact-64 V65A alpha-zero calibration."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

import lora_es_pre_hpo_alpha_zero_calibration_v62b as v62b
import lora_es_robust_sampling_population_v65 as population65


ROWS_V65A = 64
ACTORS_V65A = 4
WARMUP_PERIODS_V65A = 4
SCORED_PERIODS_V65A = 4
PAIR_PERIODS_V65A = ((0, 1), (2, 3))
PAIRS_PER_ACTOR_V65A = 2
REPLICAS_PER_UNIT_V65A = ACTORS_V65A * PAIRS_PER_ACTOR_V65A
LABEL_PLAN_V65A = {
    str(actor): ["candidate", "reference", "reference", "candidate"]
    for actor in range(ACTORS_V65A)
}
COMMON_GENERATION_SEED_V65A = v62b.COMMON_GENERATION_SEED_V62B
GENERATION_PARAMS_WITHOUT_SEED_V65A = dict(
    v62b.GENERATION_PARAMS_WITHOUT_SEED_V62B
)
ENGINE_CONTROLS_V65A = {
    "tensor_parallel_size": 1,
    "dtype": "torch.bfloat16",
    "max_model_len": 2048,
    "gpu_memory_utilization": 0.82,
    "max_loras": 1,
    "max_cpu_loras": 2,
    "max_lora_rank": 32,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": False,
    "enforce_eager": True,
    "async_scheduling": False,
    "max_num_seqs": 64,
    "max_num_batched_tokens": 8192,
    "scheduling_policy": "fcfs",
    "VLLM_BATCH_INVARIANT": False,
}
METRIC_ORDER_V65A = ("f1", "exact", "nonzero")
COMPOSITE_WEIGHTS_V65A = {
    "f1_delta": 0.80,
    "nonzero_delta": 0.20,
    "stability_improvement": 0.25,
}
FUTURE_V65_NULL_BOUND_TRANSFER_V65A = {
    "B_F": "primary_cluster_bootstrap.intervals.generated_f1_delta.null_radius",
    "B_C": "primary_cluster_bootstrap.intervals.joint_composite.null_radius",
    "B_S": "primary_cluster_bootstrap.intervals.stability_improvement.null_radius",
    "B_C_pass": "primary_cluster_bootstrap.B_C_pass",
}
WARMUP_GENERATION_COMPLETIONS_V65A = (
    ROWS_V65A * ACTORS_V65A * WARMUP_PERIODS_V65A
)
SCORED_GENERATION_COMPLETIONS_V65A = (
    ROWS_V65A * ACTORS_V65A * SCORED_PERIODS_V65A
)
TOTAL_GENERATION_COMPLETIONS_V65A = (
    WARMUP_GENERATION_COMPLETIONS_V65A
    + SCORED_GENERATION_COMPLETIONS_V65A
)
BOOTSTRAP_REPLICATES_V65A = v62b.BOOTSTRAP_REPLICATES_V62B
BOOTSTRAP_SEED_V65A = v62b.BOOTSTRAP_SEED_V62B
ONE_SIDED_ALPHA_V65A = v62b.ONE_SIDED_ALPHA_V62B
MAX_PRIMARY_CI_HALFWIDTH_V65A = v62b.MAX_PRIMARY_CI_HALFWIDTH_V62B
MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65A = (
    v62b.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B
)
RANKING_PREFIX_BYTES_V65A = 136_848
RANKING_SOURCE_FILE_SIZE_BYTES_V65A = 144_481
RANKING_PREFIX_SHA256_V65A = (
    "8259894003268a2fafed6a9a66ce3e604d5eb76cdf19a1c1c759e5ffc5916c70"
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
RUNTIME_MODULE_MANIFEST_SHA256_V65A = (
    "f09f656d7890c8776170bcc65e9273fbafefad2651a9ab6bc2ef805dfae6eeca"
)
REGISTERED_SLOT_RECORDS_SHA256_V65A = (
    "c7a5ce898287b80765330f1d5c7616f1baf4c9eaab971778b0ec817edb0ce8d8"
)


def canonical_sha256_v65a(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def self_content_sha256_v65a(value: dict) -> str:
    return canonical_sha256_v65a({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def _metric_v65a(value: object) -> tuple[float, int, int]:
    expected_keys = {
        "request_index", "row_sha256", "unit_identity_sha256",
        *METRIC_ORDER_V65A,
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("v65a generation metric schema changed")
    f1 = value.get("f1")
    exact = value.get("exact")
    nonzero = value.get("nonzero")
    if (
        isinstance(f1, bool)
        or not isinstance(f1, (int, float))
        or not math.isfinite(float(f1))
        or not 0.0 <= float(f1) <= 1.0
        or isinstance(exact, bool) or not isinstance(exact, int)
        or isinstance(nonzero, bool) or not isinstance(nonzero, int)
        or exact not in (0, 1)
        or nonzero not in (0, 1)
        or exact > nonzero
        or nonzero != int(float(f1) > 0.0)
        or (exact == 1 and float(f1) != 1.0)
    ):
        raise ValueError("v65a generation metric changed")
    return float(f1), int(exact), int(nonzero)


def validate_scored_periods_v65a(scored_periods: object) -> np.ndarray:
    """Return metrics as [unit, period, actor, metric] after strict checks."""
    if not isinstance(scored_periods, list) or len(scored_periods) != 4:
        raise ValueError("v65a scored-period coverage changed")
    result = np.empty((64, 4, 4, 3), dtype=np.float64)
    identities: list[tuple[str, str]] | None = None
    for period_index, actor_batches in enumerate(scored_periods):
        if not isinstance(actor_batches, list) or len(actor_batches) != 4:
            raise ValueError("v65a actor coverage changed")
        for actor_rank, metrics in enumerate(actor_batches):
            if not isinstance(metrics, list) or len(metrics) != 64:
                raise ValueError("v65a ranking-unit coverage changed")
            observed_identities = []
            numeric = []
            for request_index, metric in enumerate(metrics):
                if not isinstance(metric, dict):
                    raise ValueError("v65a generation metric changed")
                row_sha = metric.get("row_sha256")
                unit_sha = metric.get("unit_identity_sha256")
                if (
                    isinstance(metric.get("request_index"), bool)
                    or not isinstance(metric.get("request_index"), int)
                    or metric.get("request_index") != request_index
                    or not isinstance(row_sha, str) or len(row_sha) != 64
                    or not isinstance(unit_sha, str) or len(unit_sha) != 64
                    or any(character not in "0123456789abcdef"
                           for character in row_sha + unit_sha)
                ):
                    raise ValueError("v65a metric identity changed")
                observed_identities.append((row_sha, unit_sha))
                numeric.append(_metric_v65a(metric))
            if identities is None:
                identities = observed_identities
            elif observed_identities != identities:
                raise ValueError("v65a metric panel order changed")
            result[:, period_index, actor_rank] = np.asarray(
                numeric, dtype=np.float64,
            )
    return result


def validate_state_receipts_v65a(
    warmup_receipts: object,
    scored_receipts: object,
    expected_master_receipt: dict,
) -> tuple[list[dict], list[dict]]:
    """Require one unchanged exact V434 receipt around every period."""
    if (
        not isinstance(expected_master_receipt, dict)
        or expected_master_receipt.get("canonical_fp32_master_sha256")
        != population65.design52.MASTER_SHA256_V52
        or expected_master_receipt.get("bf16_runtime_values_sha256")
        != population65.design52.MASTER_RUNTIME_SHA256_V52
    ):
        raise ValueError("v65a expected aggregate master receipt changed")

    def validate(value: object, kind: str, count: int) -> list[dict]:
        if not isinstance(value, list) or len(value) != count:
            raise ValueError(f"v65a {kind} receipt coverage changed")
        for period_index, receipt in enumerate(value):
            if (
                not isinstance(receipt, dict)
                or set(receipt) != {
                    "period_kind", "period_index", "before", "after",
                    "identical_v434_state",
                }
                or receipt.get("period_kind") != kind
                or receipt.get("period_index") != period_index
                or receipt.get("before") != expected_master_receipt
                or receipt.get("after") != expected_master_receipt
                or receipt.get("before") != receipt.get("after")
                or receipt.get("identical_v434_state") is not True
            ):
                raise ValueError(f"v65a {kind} state receipt changed")
        return value

    return (
        validate(warmup_receipts, "unscored_warmup", WARMUP_PERIODS_V65A),
        validate(scored_receipts, "scored", SCORED_PERIODS_V65A),
    )


def paired_state_metrics_v65a(
    scored: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reference and candidate [unit, actor, pair, metric]."""
    array = (
        validate_scored_periods_v65a(scored)
        if not isinstance(scored, np.ndarray)
        else np.asarray(scored, dtype=np.float64)
    )
    if array.shape != (64, 4, 4, 3) or not np.isfinite(array).all():
        raise ValueError("v65a scored metric tensor changed")
    reference_result = np.empty((64, 4, 2, 3), dtype=np.float64)
    candidate_result = np.empty((64, 4, 2, 3), dtype=np.float64)
    for actor_rank in range(4):
        labels = LABEL_PLAN_V65A[str(actor_rank)]
        for pair_index, (left, right) in enumerate(PAIR_PERIODS_V65A):
            if {labels[left], labels[right]} != {"reference", "candidate"}:
                raise RuntimeError("v65a counterbalance plan changed")
            candidate = left if labels[left] == "candidate" else right
            reference = left if labels[left] == "reference" else right
            reference_result[:, actor_rank, pair_index] = (
                array[:, reference, actor_rank]
            )
            candidate_result[:, actor_rank, pair_index] = (
                array[:, candidate, actor_rank]
            )
    return reference_result, candidate_result


def paired_replicas_v65a(scored: object) -> np.ndarray:
    """Return candidate-minus-reference [unit, actor, pair, metric]."""
    reference, candidate = paired_state_metrics_v65a(scored)
    return candidate - reference


def frozen_bootstrap_indices_v65a() -> np.ndarray:
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED_V65A))
    return rng.integers(
        0, ROWS_V65A,
        size=(BOOTSTRAP_REPLICATES_V65A, ROWS_V65A),
        dtype=np.int64,
    )


def _ordered_quantile_v65a(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    index = int(math.floor(float(probability) * (len(ordered) - 1)))
    return float(ordered[index])


def _unit_instability_v65a(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.shape[0] != ROWS_V65A
        or array.shape[-1] != len(METRIC_ORDER_V65A)
        or int(np.prod(array.shape[1:-1])) not in (4, 8)
        or not np.isfinite(array).all()
    ):
        raise ValueError("v65a instability replica tensor changed")
    array = array.reshape(ROWS_V65A, -1, len(METRIC_ORDER_V65A))
    replicas = array.shape[1]
    pairwise = [
        np.abs(array[:, left, 0] - array[:, right, 0])
        for left in range(replicas)
        for right in range(left + 1, replicas)
    ]
    mean_pairwise = np.mean(np.stack(pairwise, axis=1), axis=1)
    exact_disagreement = (
        np.max(array[..., 1], axis=1) != np.min(array[..., 1], axis=1)
    ).astype(np.float64)
    nonzero_disagreement = (
        np.max(array[..., 2], axis=1) != np.min(array[..., 2], axis=1)
    ).astype(np.float64)
    weights = population65.design61.STABILITY_WEIGHTS_V61
    return (
        weights["mean_pairwise_absolute_f1_delta"] * mean_pairwise
        + weights["exact_label_disagreement"] * exact_disagreement
        + weights["nonzero_label_disagreement"] * nonzero_disagreement
    )


def _bootstrap_interval_v65a(
    unit_values: np.ndarray, indices: np.ndarray,
) -> dict:
    values = np.asarray(unit_values, dtype=np.float64)
    if values.shape != (ROWS_V65A,) or not np.isfinite(values).all():
        raise ValueError("v65a bootstrap unit vector changed")
    samples = np.mean(values[indices], axis=1)
    ordered = np.sort(np.asarray(samples, dtype=np.float64))
    lower_index = int(math.floor(
        ONE_SIDED_ALPHA_V65A * (len(ordered) - 1)
    ))
    upper_index = len(ordered) - 1 - lower_index
    lower = float(ordered[lower_index])
    upper = float(ordered[upper_index])
    return {
        "point": float(np.mean(values)),
        "lcb": lower,
        "ucb": upper,
        "halfwidth": 0.5 * (upper - lower),
        "contains_zero": lower <= 0.0 <= upper,
        "null_radius": max(abs(lower), abs(upper)),
    }


def future_v65_null_bounds_v65a(primary: dict, gate: dict) -> dict:
    intervals = primary.get("intervals", {})
    bounds = {
        "B_F": intervals.get("generated_f1_delta", {}).get("null_radius"),
        "B_C": intervals.get("joint_composite", {}).get("null_radius"),
        "B_S": intervals.get("stability_improvement", {}).get("null_radius"),
        "B_C_pass": primary.get("B_C_pass"),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(float(value)) or float(value) < 0.0
        for value in bounds.values()
    ):
        raise ValueError("v65a future V65 null-bound transfer changed")
    return {
        "schema": "v65a-to-v65-null-bound-transfer",
        "outcome_independent_field_mapping": dict(
            FUTURE_V65_NULL_BOUND_TRANSFER_V65A
        ),
        "bounds": {key: float(value) for key, value in bounds.items()},
        "required_future_v65_spread_gates": {
            "pooled_joint_composite_direction_spread_strictly_greater_than": (
                "2*B_C"
            ),
            "each_pass_joint_composite_direction_spread_strictly_greater_than": (
                "2*B_C_pass"
            ),
            "generated_f1_direction_spread_when_used_strictly_greater_than": (
                "2*B_F"
            ),
            "stability_coefficient_enabled_only_when_spread_strictly_greater_than": (
                "2*B_S"
            ),
            "stability_coefficient_when_gate_not_met": 0.0,
            "stability_gate_not_met_causes_population_failure": False,
        },
        "calibration_outcome_may_not_change_mapping_or_gates": True,
        "rebind_or_launch_eligible_only_if_required_alpha_zero_gate_passed": (
            gate.get("passed") is True
        ),
        "failed_required_alpha_zero_gate_forbids_bound_rebinding_and_v65_launch": (
            gate.get("passed") is not True
        ),
    }


def _joint_composite_units_v65a(
    reference: np.ndarray, candidate: np.ndarray,
) -> np.ndarray:
    paired = candidate - reference
    unit_metric_means = np.mean(
        paired, axis=tuple(range(1, paired.ndim - 1)),
    )
    stability = _unit_instability_v65a(reference) - _unit_instability_v65a(
        candidate
    )
    weights = COMPOSITE_WEIGHTS_V65A
    return (
        weights["f1_delta"] * unit_metric_means[:, 0]
        + weights["nonzero_delta"] * unit_metric_means[:, 2]
        + weights["stability_improvement"] * stability
    )


def cluster_bootstrap_v65a(
    reference_replicas: object,
    candidate_replicas: object,
    *,
    bootstrap_indices: np.ndarray | None = None,
) -> dict:
    """Average all eight paired replicas, then resample conflict units only."""
    reference = np.asarray(reference_replicas, dtype=np.float64)
    candidate = np.asarray(candidate_replicas, dtype=np.float64)
    if (
        reference.shape != (64, 4, 2, 3)
        or candidate.shape != reference.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
    ):
        raise ValueError("v65a paired-state tensor changed")
    indices = (
        frozen_bootstrap_indices_v65a()
        if bootstrap_indices is None
        else np.asarray(bootstrap_indices, dtype=np.int64)
    )
    if (
        indices.shape != (BOOTSTRAP_REPLICATES_V65A, ROWS_V65A)
        or (indices < 0).any() or (indices >= ROWS_V65A).any()
    ):
        raise ValueError("v65a bootstrap index matrix changed")
    paired = candidate - reference
    unit_metric_means = np.mean(paired, axis=(1, 2))
    stability = _unit_instability_v65a(reference) - _unit_instability_v65a(
        candidate
    )
    composite = _joint_composite_units_v65a(reference, candidate)
    component_units = {
        "generated_f1_delta": unit_metric_means[:, 0],
        "generated_nonzero_delta": unit_metric_means[:, 2],
        "stability_improvement": stability,
        "joint_composite": composite,
        "generated_exact_delta_diagnostic": unit_metric_means[:, 1],
    }
    intervals = {}
    for metric, unit_values in component_units.items():
        intervals[metric] = _bootstrap_interval_v65a(unit_values, indices)
    temporal = {}
    for pair_index in range(PAIRS_PER_ACTOR_V65A):
        pair_composite = _joint_composite_units_v65a(
            reference[:, :, pair_index, :],
            candidate[:, :, pair_index, :],
        )
        temporal[f"pair_{pair_index}"] = _bootstrap_interval_v65a(
            pair_composite, indices,
        )
    b_c_pass = max(
        interval["null_radius"] for interval in temporal.values()
    )
    return {
        "schema": "v65a-ranking64-paired-cluster-bootstrap",
        "units": 64,
        "actors": 4,
        "pairs_per_actor": 2,
        "paired_replicas_per_unit_preserved_and_averaged": 8,
        "resampled_axis": "conflict_unit_only",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES_V65A,
        "bootstrap_seed": BOOTSTRAP_SEED_V65A,
        "bootstrap_index_matrix_sha256": hashlib.sha256(
            indices.astype("<i8", copy=False).tobytes(order="C")
        ).hexdigest(),
        "one_sided_alpha": ONE_SIDED_ALPHA_V65A,
        "joint_composite_weights": dict(COMPOSITE_WEIGHTS_V65A),
        "joint_distribution_bootstrapped_before_quantiles": True,
        "intervals": intervals,
        "temporal_pair_joint_composite_intervals": temporal,
        "B_C_pass": float(b_c_pass),
        "B_C_pass_definition": (
            "max(pair_0.null_radius,pair_1.null_radius)"
        ),
        "later_v65_pass_specific_spread_gate": (
            "each pass-specific direction spread must be strictly greater "
            "than 2*B_C_pass"
        ),
        "exact_is_numeric_diagnostic_not_gate": True,
    }


def actor_influence_v65a(paired_replicas: object) -> dict:
    paired = np.asarray(paired_replicas, dtype=np.float64)
    if paired.shape != (64, 4, 2, 3) or not np.isfinite(paired).all():
        raise ValueError("v65a actor-influence tensor changed")
    full = float(np.mean(paired[..., 0]))
    leave_one_out = []
    for actor_rank in range(4):
        keep = [index for index in range(4) if index != actor_rank]
        leave_one_out.append(float(np.mean(paired[:, keep, :, 0])))
    maximum = max(abs(value - full) for value in leave_one_out)
    return {
        "schema": "v65a-ranking64-actor-influence",
        "full_four_actor_f1_point": full,
        "leave_one_actor_out_f1_points": leave_one_out,
        "maximum_absolute_leave_one_actor_out_shift": float(maximum),
    }


def gate_v65a(primary: dict, actor_influence: dict) -> dict:
    f1 = primary.get("intervals", {}).get("generated_f1_delta", {})
    composite = primary.get("intervals", {}).get("joint_composite", {})
    stability = primary.get("intervals", {}).get(
        "stability_improvement", {}
    )
    checks = {
        "generated_f1_primary_interval_contains_zero": (
            f1.get("contains_zero") is True
        ),
        "joint_composite_interval_contains_zero": (
            composite.get("contains_zero") is True
        ),
        "stability_improvement_interval_contains_zero": (
            stability.get("contains_zero") is True
        ),
        "generated_f1_primary_ci_halfwidth_within_v62b_limit": (
            isinstance(f1.get("halfwidth"), (int, float))
            and math.isfinite(float(f1["halfwidth"]))
            and float(f1["halfwidth"]) <= MAX_PRIMARY_CI_HALFWIDTH_V65A
        ),
        "actor_leave_one_out_shift_within_v62b_limit": (
            isinstance(
                actor_influence.get(
                    "maximum_absolute_leave_one_actor_out_shift"
                ),
                (int, float),
            )
            and math.isfinite(float(actor_influence[
                "maximum_absolute_leave_one_actor_out_shift"
            ]))
            and float(actor_influence[
                "maximum_absolute_leave_one_actor_out_shift"
            ]) <= MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65A
        ),
    }
    return {
        "schema": "v65a-ranking64-alpha-zero-gate",
        "maximum_primary_ci_halfwidth": MAX_PRIMARY_CI_HALFWIDTH_V65A,
        "maximum_actor_leave_one_out_shift": (
            MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65A
        ),
        "checks": checks,
        "passed": all(checks.values()),
        "exact_or_sentinel_gate_applied": False,
    }


def analyze_scored_periods_v65a(scored_periods: object) -> dict:
    scored = validate_scored_periods_v65a(scored_periods)
    reference, candidate = paired_state_metrics_v65a(scored)
    paired = candidate - reference
    primary = cluster_bootstrap_v65a(reference, candidate)
    actor = actor_influence_v65a(paired)
    gate = gate_v65a(primary, actor)
    result = {
        "schema": "v65a-ranking64-alpha-zero-analysis",
        "status": "complete_numeric_only_calibration",
        "rows": 64,
        "actors": 4,
        "warmup_periods_excluded_from_every_metric": 4,
        "scored_periods": 4,
        "pairs_per_actor": 2,
        "paired_replicas_per_unit": 8,
        "primary_cluster_bootstrap": primary,
        "actor_influence": actor,
        "future_v65_null_bound_transfer": future_v65_null_bounds_v65a(
            primary, gate,
        ),
        "required_alpha_zero_gate": gate,
        "exact_sentinel_logic_present": False,
        "adapter_update_candidate_hpo_or_promotion_performed": False,
        "row_64_or_later_opened": False,
        "raw_question_answer_prompt_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v65a(result)
    return result


FORBIDDEN_TEXT_KEYS_V65A = frozenset({
    "answer", "completion", "completion_text", "generated_text",
    "output_text", "outputs", "prediction", "prompt", "prompt_token_ids",
    "question", "raw_text", "response", "text", "token_ids",
})


def forbidden_text_keys_v65a(value: object) -> list[str]:
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_TEXT_KEYS_V65A:
                found.append(key)
            found.extend(forbidden_text_keys_v65a(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(forbidden_text_keys_v65a(item))
    return found
