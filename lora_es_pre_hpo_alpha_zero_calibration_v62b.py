#!/usr/bin/env python3
"""Pure numeric analysis for the outcome-agnostic V62B alpha-zero gate."""

from __future__ import annotations

import math

import numpy as np

import lora_es_pre_hpo_alpha_zero_calibration_v62a as v62a


ROWS_V62B = v62a.ROWS_V62A
RANKING_UNITS_V62B = v62a.RANKING_UNITS_V62A
EXACT_SENTINEL_UNITS_V62B = v62a.EXACT_SENTINEL_UNITS_V62A
ACTORS_V62B = v62a.ACTORS_V62A
WARMUP_PERIODS_V62B = 4
SCORED_BLOCKS_V62B = 6
PERIODS_PER_BLOCK_V62B = 4
SCORED_PERIODS_V62B = SCORED_BLOCKS_V62B * PERIODS_PER_BLOCK_V62B
TOTAL_PERIODS_V62B = WARMUP_PERIODS_V62B + SCORED_PERIODS_V62B
PAIRS_PER_ACTOR_V62B = SCORED_BLOCKS_V62B * 2
REPLICAS_PER_UNIT_V62B = ACTORS_V62B * PAIRS_PER_ACTOR_V62B
WARMUP_GENERATION_COMPLETIONS_V62B = (
    ROWS_V62B * ACTORS_V62B * WARMUP_PERIODS_V62B
)
SCORED_GENERATION_COMPLETIONS_V62B = (
    ROWS_V62B * ACTORS_V62B * SCORED_PERIODS_V62B
)
TOTAL_GENERATION_COMPLETIONS_V62B = (
    WARMUP_GENERATION_COMPLETIONS_V62B
    + SCORED_GENERATION_COMPLETIONS_V62B
)
COMMON_GENERATION_SEED_V62B = v62a.COMMON_GENERATION_SEED_V62A
GENERATION_PARAMS_WITHOUT_SEED_V62B = dict(
    v62a.GENERATION_PARAMS_WITHOUT_SEED_V62A
)
BLOCK_LABEL_PLAN_V62B = {
    actor: list(labels) for actor, labels in v62a.LABEL_PLAN_V62A.items()
}
WARMUP_LABEL_PLAN_V62B = {
    actor: list(labels) for actor, labels in BLOCK_LABEL_PLAN_V62B.items()
}
LABEL_PLAN_V62B = {
    actor: labels * SCORED_BLOCKS_V62B
    for actor, labels in BLOCK_LABEL_PLAN_V62B.items()
}
PAIR_PERIODS_V62B = tuple(
    pair
    for block in range(SCORED_BLOCKS_V62B)
    for pair in (
        (4 * block, 4 * block + 1),
        (4 * block + 2, 4 * block + 3),
    )
)
RUNTIME_CONTROLS_V62B = dict(v62a.RUNTIME_CONTROLS_V62A)
BOOTSTRAP_REPLICATES_V62B = v62a.BOOTSTRAP_REPLICATES_V62A
BOOTSTRAP_SEED_V62B = v62a.BOOTSTRAP_SEED_V62A
ONE_SIDED_ALPHA_V62B = v62a.ONE_SIDED_ALPHA_V62A
MAX_PRIMARY_CI_HALFWIDTH_V62B = v62a.MAX_PRIMARY_CI_HALFWIDTH_V62A
MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B = (
    v62a.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
)
STAGED_DATASET_FILE_SHA256_V62B = v62a.STAGED_DATASET_FILE_SHA256_V62A
STAGED_PANEL_FILE_SHA256_V62B = v62a.STAGED_PANEL_FILE_SHA256_V62A
STAGED_PANEL_CONTENT_SHA256_V62B = v62a.STAGED_PANEL_CONTENT_SHA256_V62A
CANONICAL_FP32_MASTER_SHA256_V62B = v62a.CANONICAL_FP32_MASTER_SHA256_V62A
BF16_RUNTIME_VALUES_SHA256_V62B = v62a.BF16_RUNTIME_VALUES_SHA256_V62A
V62_METHOD_COMMIT = v62a.V62_METHOD_COMMIT
V62_NUMERIC_AUDIT_IDENTITIES = dict(v62a.V62_NUMERIC_AUDIT_IDENTITIES)
V62_PREREGISTRATION_IDENTITIES = dict(v62a.V62_PREREGISTRATION_IDENTITIES)
STABLE_EXACT_UNIT_SHA256_V62B = tuple(v62a.STABLE_EXACT_UNIT_SHA256_V62A)
ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62B = (
    v62a.ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62A
)
SENTINEL_ORDER_V62B = tuple(v62a.SENTINEL_ORDER_V62A)


canonical_sha256_v62b = v62a.canonical_sha256_v62a


def _generation_metric_v62b(value: dict) -> tuple[float, int, int]:
    try:
        return v62a._generation_metric_v62a(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("v62b generation metric changed") from error


def _validate_state_receipts_v62b(
    receipts: object,
    *,
    period_kind: str,
    expected_count: int,
) -> list[dict]:
    if not isinstance(receipts, list) or len(receipts) != expected_count:
        raise ValueError(f"v62b {period_kind} state receipt coverage changed")
    for period_index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise ValueError(f"v62b {period_kind} state receipt changed")
        before = receipt.get("before", {})
        after = receipt.get("after", {})
        if (
            set(receipt) != {
                "period_kind",
                "period_index",
                "before",
                "after",
                "identical_v434_state",
            }
            or receipt.get("period_kind") != period_kind
            or receipt.get("period_index") != period_index
            or receipt.get("identical_v434_state") is not True
            or before != after
            or set(before) != {
                "canonical_fp32_master_sha256",
                "canonical_master_identity_sha256",
                "four_actor_certificate_sha256",
                "bf16_runtime_values_sha256",
            }
            or before.get("canonical_fp32_master_sha256")
            != CANONICAL_FP32_MASTER_SHA256_V62B
            or before.get("bf16_runtime_values_sha256")
            != BF16_RUNTIME_VALUES_SHA256_V62B
            or any(
                not isinstance(item, str) or len(item) != 64
                for item in before.values()
            )
        ):
            raise ValueError(f"v62b {period_kind} V434 state receipt changed")
    return receipts


def validate_evidence_v62b(evidence: dict) -> list[dict]:
    if not isinstance(evidence, dict):
        raise ValueError("v62b evidence changed")
    compact = {
        key: item for key, item in evidence.items()
        if key != "content_sha256_before_self_field"
    }
    expected_keys = {
        "schema",
        "status",
        "v62_methodology_commit",
        "v62_numeric_audit_identities",
        "v62_preregistration_identities",
        "staged_dataset_file_sha256",
        "staged_panel_file_sha256",
        "staged_panel_content_sha256",
        "canonical_fp32_master_sha256",
        "bf16_runtime_values_sha256",
        "row_count",
        "ranking_units",
        "exact_sentinel_units",
        "actor_count",
        "unscored_warmup_period_count",
        "scored_period_count",
        "total_period_count",
        "scored_blocks",
        "periods_per_block",
        "pairs_per_actor",
        "replicas_per_unit",
        "warmup_label_plan",
        "scored_label_plan",
        "pair_periods",
        "common_generation_seed",
        "generation_params_without_seed",
        "runtime_determinism_controls",
        "warmup_state_receipts",
        "numeric_warmup_state_receipts_sha256",
        "scored_state_receipts",
        "numeric_scored_state_receipts_sha256",
        "rows",
        "numeric_actor_period_manifest_sha256",
        "generation_only",
        "warmup_generation_completions_discarded",
        "scored_generation_completions",
        "total_generation_completions",
        "warmup_raw_outputs_persisted",
        "warmup_generation_metrics_computed_or_persisted",
        "warmup_adaptive_retry_drop_or_reorder_performed",
        "scored_period_adaptive_retry_drop_reorder_or_early_stop_performed",
        "teacher_forced_requests",
        "alpha",
        "adapter_update_candidate_or_hpo_performed",
        "holdback_ood_shadow_or_protected_opened",
        "raw_question_answer_or_generation_text_persisted",
        "content_sha256_before_self_field",
    }
    warmup_receipts = evidence.get("warmup_state_receipts")
    scored_receipts = evidence.get("scored_state_receipts")
    rows = evidence.get("rows")
    if (
        set(evidence) != expected_keys
        or evidence.get("schema")
        != "v62b-pre-hpo-alpha-zero-generation-only-evidence"
        or evidence.get("status")
        != "complete_fixed_warmup_and_scored_alpha_zero_characterization"
        or evidence.get("content_sha256_before_self_field")
        != canonical_sha256_v62b(compact)
        or evidence.get("v62_methodology_commit") != V62_METHOD_COMMIT
        or evidence.get("v62_numeric_audit_identities")
        != V62_NUMERIC_AUDIT_IDENTITIES
        or evidence.get("v62_preregistration_identities")
        != V62_PREREGISTRATION_IDENTITIES
        or evidence.get("staged_dataset_file_sha256")
        != STAGED_DATASET_FILE_SHA256_V62B
        or evidence.get("staged_panel_file_sha256")
        != STAGED_PANEL_FILE_SHA256_V62B
        or evidence.get("staged_panel_content_sha256")
        != STAGED_PANEL_CONTENT_SHA256_V62B
        or evidence.get("canonical_fp32_master_sha256")
        != CANONICAL_FP32_MASTER_SHA256_V62B
        or evidence.get("bf16_runtime_values_sha256")
        != BF16_RUNTIME_VALUES_SHA256_V62B
        or evidence.get("row_count") != ROWS_V62B
        or evidence.get("ranking_units") != RANKING_UNITS_V62B
        or evidence.get("exact_sentinel_units") != EXACT_SENTINEL_UNITS_V62B
        or evidence.get("actor_count") != ACTORS_V62B
        or evidence.get("unscored_warmup_period_count")
        != WARMUP_PERIODS_V62B
        or evidence.get("scored_period_count") != SCORED_PERIODS_V62B
        or evidence.get("total_period_count") != TOTAL_PERIODS_V62B
        or evidence.get("scored_blocks") != SCORED_BLOCKS_V62B
        or evidence.get("periods_per_block") != PERIODS_PER_BLOCK_V62B
        or evidence.get("pairs_per_actor") != PAIRS_PER_ACTOR_V62B
        or evidence.get("replicas_per_unit") != REPLICAS_PER_UNIT_V62B
        or evidence.get("warmup_label_plan") != WARMUP_LABEL_PLAN_V62B
        or evidence.get("scored_label_plan") != LABEL_PLAN_V62B
        or evidence.get("pair_periods")
        != [list(pair) for pair in PAIR_PERIODS_V62B]
        or evidence.get("common_generation_seed")
        != COMMON_GENERATION_SEED_V62B
        or evidence.get("generation_params_without_seed")
        != GENERATION_PARAMS_WITHOUT_SEED_V62B
        or evidence.get("runtime_determinism_controls")
        != RUNTIME_CONTROLS_V62B
        or evidence.get("numeric_warmup_state_receipts_sha256")
        != canonical_sha256_v62b(warmup_receipts)
        or evidence.get("numeric_scored_state_receipts_sha256")
        != canonical_sha256_v62b(scored_receipts)
        or evidence.get("generation_only") is not True
        or evidence.get("warmup_generation_completions_discarded")
        != WARMUP_GENERATION_COMPLETIONS_V62B
        or evidence.get("scored_generation_completions")
        != SCORED_GENERATION_COMPLETIONS_V62B
        or evidence.get("total_generation_completions")
        != TOTAL_GENERATION_COMPLETIONS_V62B
        or evidence.get("warmup_raw_outputs_persisted") is not False
        or evidence.get("warmup_generation_metrics_computed_or_persisted")
        is not False
        or evidence.get("warmup_adaptive_retry_drop_or_reorder_performed")
        is not False
        or evidence.get(
            "scored_period_adaptive_retry_drop_reorder_or_early_stop_performed"
        ) is not False
        or evidence.get("teacher_forced_requests") != 0
        or evidence.get("alpha") != 0.0
        or evidence.get("adapter_update_candidate_or_hpo_performed") is not False
        or evidence.get("holdback_ood_shadow_or_protected_opened") is not False
        or evidence.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or not isinstance(rows, list)
        or len(rows) != ROWS_V62B
    ):
        raise ValueError("v62b evidence contract changed")
    _validate_state_receipts_v62b(
        warmup_receipts,
        period_kind="unscored_warmup",
        expected_count=WARMUP_PERIODS_V62B,
    )
    _validate_state_receipts_v62b(
        scored_receipts,
        period_kind="scored",
        expected_count=SCORED_PERIODS_V62B,
    )

    row_ids = []
    unit_ids = []
    for request_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("v62b row changed")
        periods = row.get("scored_periods")
        row_sha = row.get("row_sha256")
        unit_sha = row.get("unit_identity_sha256")
        if (
            set(row) != {
                "request_index",
                "row_sha256",
                "unit_identity_sha256",
                "role",
                "scored_periods",
            }
            or row.get("request_index") != request_index
            or not isinstance(row_sha, str)
            or len(row_sha) != 64
            or not isinstance(unit_sha, str)
            or len(unit_sha) != 64
            or row.get("role")
            != ("ranking" if request_index < 64 else "exact_sentinel")
            or not isinstance(periods, list)
            or len(periods) != SCORED_PERIODS_V62B
        ):
            raise ValueError("v62b row identity changed")
        for period_index, period in enumerate(periods):
            if not isinstance(period, dict):
                raise ValueError("v62b scored period changed")
            actors = period.get("actors")
            if (
                set(period) != {"period_index", "request_type", "actors"}
                or period.get("period_index") != period_index
                or period.get("request_type") != "generation"
                or not isinstance(actors, list)
                or len(actors) != ACTORS_V62B
            ):
                raise ValueError("v62b scored period coverage changed")
            for actor_rank, actor in enumerate(actors):
                if (
                    not isinstance(actor, dict)
                    or set(actor) != {"actor_rank", "label", "generation"}
                    or actor.get("actor_rank") != actor_rank
                    or actor.get("label")
                    != LABEL_PLAN_V62B[str(actor_rank)][period_index]
                ):
                    raise ValueError("v62b counterbalance label changed")
                _generation_metric_v62b(actor.get("generation"))
        row_ids.append(row_sha)
        unit_ids.append(unit_sha)
    if (
        len(set(row_ids)) != ROWS_V62B
        or len(set(unit_ids)) != ROWS_V62B
        or tuple(unit_ids[64:]) != SENTINEL_ORDER_V62B
        or evidence.get("numeric_actor_period_manifest_sha256")
        != canonical_sha256_v62b(rows)
    ):
        raise ValueError("v62b panel order or numeric manifest changed")
    return rows


def _generation_array_v62b(rows: list[dict]) -> np.ndarray:
    values = np.empty(
        (len(rows), ACTORS_V62B, SCORED_PERIODS_V62B, 3),
        dtype=np.float64,
    )
    for unit, row in enumerate(rows):
        for period, period_value in enumerate(row["scored_periods"]):
            for actor, value in enumerate(period_value["actors"]):
                values[unit, actor, period] = _generation_metric_v62b(
                    value["generation"]
                )
    return values


def paired_deltas_v62b(
    generation: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    generation = np.asarray(generation, dtype=np.float64)
    expected_tail = (ACTORS_V62B, SCORED_PERIODS_V62B, 3)
    if (
        generation.ndim != 4
        or generation.shape[1:] != expected_tail
        or not np.isfinite(generation).all()
    ):
        raise ValueError("v62b generation array changed")
    delta = np.empty(
        (generation.shape[0], ACTORS_V62B, PAIRS_PER_ACTOR_V62B, 3),
        dtype=np.float64,
    )
    receipts = []
    for actor in range(ACTORS_V62B):
        labels = LABEL_PLAN_V62B[str(actor)]
        order_counts = {True: 0, False: 0}
        for pair_index, periods in enumerate(PAIR_PERIODS_V62B):
            reference = next(
                period for period in periods if labels[period] == "reference"
            )
            candidate = next(
                period for period in periods if labels[period] == "candidate"
            )
            candidate_after_reference = candidate > reference
            order_counts[candidate_after_reference] += 1
            delta[:, actor, pair_index] = (
                generation[:, actor, candidate]
                - generation[:, actor, reference]
            )
            receipts.append({
                "actor_rank": actor,
                "pair_index": pair_index,
                "block_index": pair_index // 2,
                "reference_period": reference,
                "candidate_period": candidate,
                "candidate_after_reference": candidate_after_reference,
            })
        if order_counts != {True: 6, False: 6}:
            raise ValueError("v62b actor order balance changed")
    return delta, receipts


def _quantile_v62b(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[int(math.floor(probability * (len(ordered) - 1)))])


def primary_f1_bootstrap_v62b(f1_delta: np.ndarray) -> dict:
    f1_delta = np.asarray(f1_delta, dtype=np.float64)
    if (
        f1_delta.shape
        != (RANKING_UNITS_V62B, ACTORS_V62B, PAIRS_PER_ACTOR_V62B)
        or not np.isfinite(f1_delta).all()
    ):
        raise ValueError("v62b primary F1 replica array changed")
    unit_means = np.mean(
        f1_delta.reshape(RANKING_UNITS_V62B, REPLICAS_PER_UNIT_V62B),
        axis=1,
    )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED_V62B))
    samples = np.empty(BOOTSTRAP_REPLICATES_V62B, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES_V62B):
        selected = rng.integers(
            0,
            RANKING_UNITS_V62B,
            size=RANKING_UNITS_V62B,
        )
        samples[index] = float(np.mean(unit_means[selected]))
    lcb = _quantile_v62b(samples, ONE_SIDED_ALPHA_V62B)
    ucb = _quantile_v62b(samples, 1.0 - ONE_SIDED_ALPHA_V62B)
    return {
        "estimand": (
            "mean paired generated-F1 delta after averaging all 48 "
            "actor-pair replicas within each conflict unit"
        ),
        "point": float(np.mean(unit_means)),
        "lcb": lcb,
        "ucb": ucb,
        "halfwidth": (ucb - lcb) / 2.0,
        "contains_zero": lcb <= 0.0 <= ucb,
        "resampled_axis": "conflict_unit",
        "within_unit_actor_pair_replicas_preserved_and_averaged": (
            REPLICAS_PER_UNIT_V62B
        ),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES_V62B,
        "bootstrap_seed": BOOTSTRAP_SEED_V62B,
        "one_sided_alpha": ONE_SIDED_ALPHA_V62B,
    }


def actor_influence_v62b(f1_delta: np.ndarray) -> dict:
    f1_delta = np.asarray(f1_delta, dtype=np.float64)
    if (
        f1_delta.shape
        != (RANKING_UNITS_V62B, ACTORS_V62B, PAIRS_PER_ACTOR_V62B)
        or not np.isfinite(f1_delta).all()
    ):
        raise ValueError("v62b actor influence array changed")
    full = float(np.mean(f1_delta))
    actor_means = np.mean(f1_delta, axis=(0, 2))
    leave_one_out = np.asarray([
        np.mean(np.delete(f1_delta, actor, axis=1))
        for actor in range(ACTORS_V62B)
    ], dtype=np.float64)
    maximum = float(np.max(np.abs(leave_one_out - full)))
    return {
        "definition": (
            "maximum absolute difference between the full four-actor point "
            "and each leave-one-actor-out point across all 12 pairs per actor"
        ),
        "full_four_actor_point": full,
        "actor_mean_deltas": [float(value) for value in actor_means],
        "leave_one_actor_out_points": [
            float(value) for value in leave_one_out
        ],
        "maximum_absolute_leave_one_actor_out_shift": maximum,
    }


def gate_v62b(primary: dict, actor_influence: dict) -> dict:
    checks = {
        "null_primary_ci_contains_zero": primary["contains_zero"] is True,
        "primary_ci_halfwidth_at_most_frozen_limit_inclusive": (
            primary["halfwidth"] <= MAX_PRIMARY_CI_HALFWIDTH_V62B
        ),
        "actor_leave_one_out_shift_at_most_frozen_limit_inclusive": (
            actor_influence["maximum_absolute_leave_one_actor_out_shift"]
            <= MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "maximum_primary_ci_halfwidth_inclusive": (
            MAX_PRIMARY_CI_HALFWIDTH_V62B
        ),
        "maximum_actor_leave_one_out_shift_inclusive": (
            MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62B
        ),
        "failure_action": "fail_closed_before_hpo_population_or_update",
        "authorizes_hpo_population_or_update": False,
    }


def _exact_diagnostics_v62b(
    sentinel: np.ndarray,
    sentinel_rows: list[dict],
) -> dict:
    by_hash = {
        row["unit_identity_sha256"]: index
        for index, row in enumerate(sentinel_rows)
    }
    units = []
    paired_flips = 0
    for unit_sha in SENTINEL_ORDER_V62B:
        unit_index = by_hash[unit_sha]
        candidate = []
        reference = []
        for actor in range(ACTORS_V62B):
            labels = LABEL_PLAN_V62B[str(actor)]
            for periods in PAIR_PERIODS_V62B:
                reference_period = next(
                    period for period in periods
                    if labels[period] == "reference"
                )
                candidate_period = next(
                    period for period in periods
                    if labels[period] == "candidate"
                )
                reference.append(int(
                    sentinel[unit_index, actor, reference_period, 1]
                ))
                candidate.append(int(
                    sentinel[unit_index, actor, candidate_period, 1]
                ))
        flips = sum(
            left != right for left, right in zip(reference, candidate, strict=True)
        )
        paired_flips += flips
        units.append({
            "unit_identity_sha256": unit_sha,
            "role": (
                "actor_unstable_stress_diagnostic_only"
                if unit_sha == ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62B
                else "baseline_stable_diagnostic_only"
            ),
            "reference_exact_pass_count_of_48": sum(reference),
            "candidate_exact_pass_count_of_48": sum(candidate),
            "reference_strict_majority_at_least_25_of_48": (
                sum(reference) >= 25
            ),
            "candidate_strict_majority_at_least_25_of_48": (
                sum(candidate) >= 25
            ),
            "paired_exact_flip_count": flips,
        })
    stable = [
        item for item in units
        if item["role"] == "baseline_stable_diagnostic_only"
    ]
    unstable = next(
        item for item in units
        if item["role"] == "actor_unstable_stress_diagnostic_only"
    )
    return {
        "stable_panel": {
            "unit_count": 3,
            "units": stable,
            "reference_exact_pass_total_of_144": sum(
                item["reference_exact_pass_count_of_48"] for item in stable
            ),
            "candidate_exact_pass_total_of_144": sum(
                item["candidate_exact_pass_count_of_48"] for item in stable
            ),
            "reference_majority_consensus_unit_count_of_3": sum(
                item["reference_strict_majority_at_least_25_of_48"]
                for item in stable
            ),
            "candidate_majority_consensus_unit_count_of_3": sum(
                item["candidate_strict_majority_at_least_25_of_48"]
                for item in stable
            ),
        },
        "actor_unstable_stress_unit": unstable,
        "all_four_sentinel_paired_exact_flip_count": paired_flips,
        "used_in_alpha_zero_gate": False,
        "any_single_flip_aborts": False,
        "any_per_unit_all_replicas_failure_aborts": False,
        "actor_unstable_stress_unit_aborts": False,
    }


def build_analysis_v62b(evidence: dict) -> dict:
    rows = validate_evidence_v62b(evidence)
    generation = _generation_array_v62b(rows)
    delta, receipts = paired_deltas_v62b(generation[:RANKING_UNITS_V62B])
    primary = primary_f1_bootstrap_v62b(delta[..., 0])
    actor = actor_influence_v62b(delta[..., 0])
    gate = gate_v62b(primary, actor)
    exact = _exact_diagnostics_v62b(
        generation[RANKING_UNITS_V62B:],
        rows[RANKING_UNITS_V62B:],
    )
    value = {
        "schema": "v62b-pre-hpo-alpha-zero-generation-only-analysis",
        "status": (
            "complete_gate_passed_hpo_still_unauthorized"
            if gate["passed"] else "complete_gate_failed_closed"
        ),
        "source_evidence_content_sha256": evidence[
            "content_sha256_before_self_field"
        ],
        "v62_methodology_commit": V62_METHOD_COMMIT,
        "v62_numeric_audit_identities": dict(V62_NUMERIC_AUDIT_IDENTITIES),
        "v62_preregistration_identities": dict(
            V62_PREREGISTRATION_IDENTITIES
        ),
        "unscored_warmup_excluded_from_every_metric": True,
        "primary_generated_f1": primary,
        "actor_influence": actor,
        "required_pre_hpo_gate": gate,
        "counterbalance_pair_receipts": receipts,
        "exact_sentinel_diagnostics": exact,
        "generation_only": True,
        "teacher_forced_metric_computed": False,
        "teacher_forced_requests": 0,
        "hpo_population_update_candidate_or_promotion_authorized": False,
        "raw_question_answer_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v62b(value)
    return value
