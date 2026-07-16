#!/usr/bin/env python3
"""Pure numeric analysis for the V62A generation-only alpha-zero gate."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


ROWS_V62A = 68
RANKING_UNITS_V62A = 64
EXACT_SENTINEL_UNITS_V62A = 4
ACTORS_V62A = 4
PERIODS_V62A = 4
PAIRS_PER_ACTOR_V62A = 2
REPLICAS_PER_UNIT_V62A = 8
COMMON_GENERATION_SEED_V62A = 2_026_071_601
GENERATION_PARAMS_WITHOUT_SEED_V62A = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 64,
    "detokenize": True,
}
LABEL_PLAN_V62A = {
    "0": ["reference", "candidate", "candidate", "reference"],
    "1": ["candidate", "reference", "reference", "candidate"],
    "2": ["reference", "candidate", "candidate", "reference"],
    "3": ["candidate", "reference", "reference", "candidate"],
}
PAIR_PERIODS_V62A = ((0, 1), (2, 3))
RUNTIME_CONTROLS_V62A = {
    "VLLM_BATCH_INVARIANT": False,
    "async_scheduling": False,
    "max_num_seqs": 68,
    "scheduling_policy": "fcfs",
    "enforce_eager": True,
}
BOOTSTRAP_REPLICATES_V62A = 4096
BOOTSTRAP_SEED_V62A = 2_026_071_612
ONE_SIDED_ALPHA_V62A = 0.05
MAX_PRIMARY_CI_HALFWIDTH_V62A = 0.000773822590292528
MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A = 0.0012119648781783704
STAGED_DATASET_FILE_SHA256_V62A = (
    "9c1b7f69595cf70ef045259e2097c39546e9f1d84a6b0870fcb14e987655079a"
)
STAGED_PANEL_FILE_SHA256_V62A = (
    "92e0c6160bfc7884a00be4c34c427685dcb2bf5a6aa8c3820f5c53e225f8091c"
)
STAGED_PANEL_CONTENT_SHA256_V62A = (
    "ca0a947e6437c0d84360176087b0a9dab12b79cf6ba1be8f965b24e9f4ec7ba4"
)
CANONICAL_FP32_MASTER_SHA256_V62A = (
    "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
)
BF16_RUNTIME_VALUES_SHA256_V62A = (
    "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
)
V62_METHOD_COMMIT = "3878fb4d85ddbb1fb96d382d1af21446bc8764c0"
V62_NUMERIC_AUDIT_IDENTITIES = {
    "file_sha256": (
        "5597aef07576ff35613dbb77c5a3c5f35a441374d926b7e54ac48df8e6386785"
    ),
    "content_sha256": (
        "365280457e84abb2734cb21b872a79d97e4f4da2f4c62b7bc1ef5361cab659fe"
    ),
}
V62_PREREGISTRATION_IDENTITIES = {
    "file_sha256": (
        "45342e29ae1b9cd6041092e26339aad76ab6379048fe364d1ea176e9f399e03c"
    ),
    "content_sha256": (
        "6bd56b962b6cdc4543ac126dc098f357bd03356964c18fb9796a2c2204db6be3"
    ),
}
STABLE_EXACT_UNIT_SHA256_V62A = (
    "2ef7c0e5ca2ff81b7326ea6dc2bd8b32c2499f939c04f9574c7135be37837ab4",
    "f080d7ea1b60062d035852d3542a2664e49af07294efec841f95acc99994f68f",
    "aa2af8f5c0eaede64c3acd990852475c441587c29bfa50fab0f30b2ed0061a66",
)
ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62A = (
    "27c984ca0a7dbd4ffead6e79d7b691dc7a37856356a9a050a40603c11c6dbda7"
)
SENTINEL_ORDER_V62A = (
    STABLE_EXACT_UNIT_SHA256_V62A[0],
    ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62A,
    STABLE_EXACT_UNIT_SHA256_V62A[1],
    STABLE_EXACT_UNIT_SHA256_V62A[2],
)


def canonical_sha256_v62a(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def _generation_metric_v62a(value: dict) -> tuple[float, int, int]:
    if set(value) != {"f1", "exact", "nonzero"}:
        raise ValueError("v62a generation metric schema changed")
    f1 = value.get("f1")
    exact = value.get("exact")
    nonzero = value.get("nonzero")
    if (
        isinstance(f1, bool)
        or not isinstance(f1, (int, float))
        or not math.isfinite(float(f1))
        or not 0.0 <= float(f1) <= 1.0
        or exact not in (0, 1)
        or nonzero not in (0, 1)
        or exact > nonzero
        or nonzero != int(float(f1) > 0.0)
        or (exact == 1 and not math.isclose(float(f1), 1.0))
    ):
        raise ValueError("v62a generation metric changed")
    return float(f1), int(exact), int(nonzero)


def validate_evidence_v62a(evidence: dict) -> list[dict]:
    rows = list(evidence.get("rows", []))
    compact = {
        key: item for key, item in evidence.items()
        if key != "content_sha256_before_self_field"
    }
    state_receipts = evidence.get("state_receipts", [])
    if (
        set(evidence) != {
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
            "period_count",
            "pairs_per_actor",
            "replicas_per_unit",
            "label_plan",
            "pair_periods",
            "common_generation_seed",
            "generation_params_without_seed",
            "runtime_determinism_controls",
            "state_receipts",
            "numeric_state_receipts_sha256",
            "rows",
            "numeric_actor_period_manifest_sha256",
            "generation_only",
            "generation_completions",
            "teacher_forced_requests",
            "alpha",
            "adapter_update_candidate_or_hpo_performed",
            "holdback_ood_shadow_or_protected_opened",
            "raw_question_answer_or_generation_text_persisted",
            "content_sha256_before_self_field",
        }
        or evidence.get("schema")
        != "v62a-pre-hpo-alpha-zero-generation-only-evidence"
        or evidence.get("status")
        != "complete_alpha_zero_generation_only_characterization"
        or evidence.get("content_sha256_before_self_field")
        != canonical_sha256_v62a(compact)
        or evidence.get("v62_methodology_commit") != V62_METHOD_COMMIT
        or evidence.get("v62_numeric_audit_identities")
        != V62_NUMERIC_AUDIT_IDENTITIES
        or evidence.get("v62_preregistration_identities")
        != V62_PREREGISTRATION_IDENTITIES
        or evidence.get("staged_dataset_file_sha256")
        != STAGED_DATASET_FILE_SHA256_V62A
        or evidence.get("staged_panel_file_sha256")
        != STAGED_PANEL_FILE_SHA256_V62A
        or evidence.get("staged_panel_content_sha256")
        != STAGED_PANEL_CONTENT_SHA256_V62A
        or evidence.get("canonical_fp32_master_sha256")
        != CANONICAL_FP32_MASTER_SHA256_V62A
        or evidence.get("bf16_runtime_values_sha256")
        != BF16_RUNTIME_VALUES_SHA256_V62A
        or evidence.get("row_count") != ROWS_V62A
        or evidence.get("ranking_units") != RANKING_UNITS_V62A
        or evidence.get("exact_sentinel_units") != EXACT_SENTINEL_UNITS_V62A
        or evidence.get("actor_count") != ACTORS_V62A
        or evidence.get("period_count") != PERIODS_V62A
        or evidence.get("pairs_per_actor") != PAIRS_PER_ACTOR_V62A
        or evidence.get("replicas_per_unit") != REPLICAS_PER_UNIT_V62A
        or evidence.get("label_plan") != LABEL_PLAN_V62A
        or evidence.get("pair_periods")
        != [list(pair) for pair in PAIR_PERIODS_V62A]
        or evidence.get("common_generation_seed")
        != COMMON_GENERATION_SEED_V62A
        or evidence.get("generation_params_without_seed")
        != GENERATION_PARAMS_WITHOUT_SEED_V62A
        or evidence.get("runtime_determinism_controls")
        != RUNTIME_CONTROLS_V62A
        or evidence.get("generation_only") is not True
        or evidence.get("generation_completions") != 1088
        or evidence.get("teacher_forced_requests") != 0
        or evidence.get("alpha") != 0.0
        or evidence.get("adapter_update_candidate_or_hpo_performed") is not False
        or evidence.get("holdback_ood_shadow_or_protected_opened") is not False
        or evidence.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or len(state_receipts) != PERIODS_V62A
        or evidence.get("numeric_state_receipts_sha256")
        != canonical_sha256_v62a(state_receipts)
        or len(rows) != ROWS_V62A
    ):
        raise ValueError("v62a evidence contract changed")
    for period_index, receipt in enumerate(state_receipts):
        before = receipt.get("before", {})
        after = receipt.get("after", {})
        if (
            set(receipt) != {
                "period_index",
                "before",
                "after",
                "identical_v434_state",
            }
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
            != CANONICAL_FP32_MASTER_SHA256_V62A
            or before.get("bf16_runtime_values_sha256")
            != BF16_RUNTIME_VALUES_SHA256_V62A
            or any(
                not isinstance(value, str) or len(value) != 64
                for value in before.values()
            )
        ):
            raise ValueError("v62a V434 state receipt changed")

    row_ids = []
    unit_ids = []
    for request_index, row in enumerate(rows):
        periods = row.get("periods", [])
        row_sha = row.get("row_sha256")
        unit_sha = row.get("unit_identity_sha256")
        if (
            set(row) != {
                "request_index",
                "row_sha256",
                "unit_identity_sha256",
                "role",
                "periods",
            }
            or row.get("request_index") != request_index
            or not isinstance(row_sha, str)
            or len(row_sha) != 64
            or not isinstance(unit_sha, str)
            or len(unit_sha) != 64
            or row.get("role")
            != ("ranking" if request_index < 64 else "exact_sentinel")
            or len(periods) != PERIODS_V62A
        ):
            raise ValueError("v62a row identity changed")
        for period_index, period in enumerate(periods):
            actors = period.get("actors", [])
            if (
                set(period) != {"period_index", "request_type", "actors"}
                or period.get("period_index") != period_index
                or period.get("request_type") != "generation"
                or len(actors) != ACTORS_V62A
            ):
                raise ValueError("v62a period coverage changed")
            for actor_rank, actor in enumerate(actors):
                if (
                    set(actor) != {"actor_rank", "label", "generation"}
                    or actor.get("actor_rank") != actor_rank
                    or actor.get("label")
                    != LABEL_PLAN_V62A[str(actor_rank)][period_index]
                ):
                    raise ValueError("v62a counterbalance label changed")
                _generation_metric_v62a(actor["generation"])
        row_ids.append(row_sha)
        unit_ids.append(unit_sha)
    if (
        len(set(row_ids)) != ROWS_V62A
        or len(set(unit_ids)) != ROWS_V62A
        or tuple(unit_ids[64:]) != SENTINEL_ORDER_V62A
        or evidence.get("numeric_actor_period_manifest_sha256")
        != canonical_sha256_v62a(rows)
    ):
        raise ValueError("v62a panel order or numeric manifest changed")
    return rows


def _generation_array_v62a(rows: list[dict]) -> np.ndarray:
    values = np.empty((len(rows), 4, 4, 3), dtype=np.float64)
    for unit, row in enumerate(rows):
        for period, period_value in enumerate(row["periods"]):
            for actor, value in enumerate(period_value["actors"]):
                values[unit, actor, period] = _generation_metric_v62a(
                    value["generation"]
                )
    return values


def paired_deltas_v62a(
    generation: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    generation = np.asarray(generation, dtype=np.float64)
    if (
        generation.ndim != 4
        or generation.shape[1:] != (4, 4, 3)
        or not np.isfinite(generation).all()
    ):
        raise ValueError("v62a generation array changed")
    delta = np.empty((generation.shape[0], 4, 2, 3), dtype=np.float64)
    receipts = []
    for actor in range(4):
        labels = LABEL_PLAN_V62A[str(actor)]
        for pair_index, periods in enumerate(PAIR_PERIODS_V62A):
            reference = next(
                period for period in periods
                if labels[period] == "reference"
            )
            candidate = next(
                period for period in periods
                if labels[period] == "candidate"
            )
            delta[:, actor, pair_index] = (
                generation[:, actor, candidate]
                - generation[:, actor, reference]
            )
            receipts.append({
                "actor_rank": actor,
                "pair_index": pair_index,
                "reference_period": reference,
                "candidate_period": candidate,
                "candidate_after_reference": candidate > reference,
            })
    return delta, receipts


def _quantile_v62a(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[int(math.floor(probability * (len(ordered) - 1)))])


def primary_f1_bootstrap_v62a(f1_delta: np.ndarray) -> dict:
    f1_delta = np.asarray(f1_delta, dtype=np.float64)
    if (
        f1_delta.shape != (RANKING_UNITS_V62A, 4, 2)
        or not np.isfinite(f1_delta).all()
    ):
        raise ValueError("v62a primary F1 replica array changed")
    unit_means = np.mean(f1_delta.reshape(64, 8), axis=1)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED_V62A))
    samples = np.empty(BOOTSTRAP_REPLICATES_V62A, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES_V62A):
        selected = rng.integers(0, 64, size=64)
        samples[index] = float(np.mean(unit_means[selected]))
    lcb = _quantile_v62a(samples, ONE_SIDED_ALPHA_V62A)
    ucb = _quantile_v62a(samples, 1.0 - ONE_SIDED_ALPHA_V62A)
    return {
        "estimand": (
            "mean paired generated-F1 delta after averaging all 8 actor-pair "
            "replicas within each conflict unit"
        ),
        "point": float(np.mean(unit_means)),
        "lcb": lcb,
        "ucb": ucb,
        "halfwidth": (ucb - lcb) / 2.0,
        "contains_zero": lcb <= 0.0 <= ucb,
        "resampled_axis": "conflict_unit",
        "within_unit_actor_pair_replicas_preserved_and_averaged": 8,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES_V62A,
        "bootstrap_seed": BOOTSTRAP_SEED_V62A,
        "one_sided_alpha": ONE_SIDED_ALPHA_V62A,
    }


def actor_influence_v62a(f1_delta: np.ndarray) -> dict:
    f1_delta = np.asarray(f1_delta, dtype=np.float64)
    if (
        f1_delta.shape != (RANKING_UNITS_V62A, 4, 2)
        or not np.isfinite(f1_delta).all()
    ):
        raise ValueError("v62a actor influence array changed")
    full = float(np.mean(f1_delta))
    actor_means = np.mean(f1_delta, axis=(0, 2))
    leave_one_out = np.asarray([
        np.mean(np.delete(f1_delta, actor, axis=1))
        for actor in range(ACTORS_V62A)
    ], dtype=np.float64)
    maximum = float(np.max(np.abs(leave_one_out - full)))
    return {
        "definition": (
            "maximum absolute difference between the full four-actor point "
            "and each leave-one-actor-out point"
        ),
        "full_four_actor_point": full,
        "actor_mean_deltas": [float(value) for value in actor_means],
        "leave_one_actor_out_points": [
            float(value) for value in leave_one_out
        ],
        "maximum_absolute_leave_one_actor_out_shift": maximum,
    }


def gate_v62a(primary: dict, actor_influence: dict) -> dict:
    checks = {
        "null_primary_ci_contains_zero": primary["contains_zero"] is True,
        "primary_ci_halfwidth_at_most_frozen_limit_inclusive": (
            primary["halfwidth"] <= MAX_PRIMARY_CI_HALFWIDTH_V62A
        ),
        "actor_leave_one_out_shift_at_most_frozen_limit_inclusive": (
            actor_influence["maximum_absolute_leave_one_actor_out_shift"]
            <= MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "maximum_primary_ci_halfwidth_inclusive": (
            MAX_PRIMARY_CI_HALFWIDTH_V62A
        ),
        "maximum_actor_leave_one_out_shift_inclusive": (
            MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
        ),
        "failure_action": "fail_closed_before_hpo_population_or_update",
        "authorizes_hpo_population_or_update": False,
    }


def _exact_diagnostics_v62a(
    sentinel: np.ndarray,
    sentinel_rows: list[dict],
) -> dict:
    by_hash = {
        row["unit_identity_sha256"]: index
        for index, row in enumerate(sentinel_rows)
    }
    units = []
    paired_flips = 0
    for unit_sha in SENTINEL_ORDER_V62A:
        unit_index = by_hash[unit_sha]
        candidate = []
        reference = []
        for actor in range(4):
            labels = LABEL_PLAN_V62A[str(actor)]
            for periods in PAIR_PERIODS_V62A:
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
        paired_flips += sum(
            left != right for left, right in zip(reference, candidate, strict=True)
        )
        units.append({
            "unit_identity_sha256": unit_sha,
            "role": (
                "actor_unstable_stress_diagnostic_only"
                if unit_sha == ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62A
                else "baseline_stable_diagnostic_only"
            ),
            "reference_exact_pass_count_of_8": sum(reference),
            "candidate_exact_pass_count_of_8": sum(candidate),
            "reference_strict_majority_at_least_5_of_8": sum(reference) >= 5,
            "candidate_strict_majority_at_least_5_of_8": sum(candidate) >= 5,
            "paired_exact_flip_count": sum(
                left != right
                for left, right in zip(reference, candidate, strict=True)
            ),
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
            "reference_exact_pass_total_of_24": sum(
                item["reference_exact_pass_count_of_8"] for item in stable
            ),
            "candidate_exact_pass_total_of_24": sum(
                item["candidate_exact_pass_count_of_8"] for item in stable
            ),
            "reference_majority_consensus_unit_count_of_3": sum(
                item["reference_strict_majority_at_least_5_of_8"]
                for item in stable
            ),
            "candidate_majority_consensus_unit_count_of_3": sum(
                item["candidate_strict_majority_at_least_5_of_8"]
                for item in stable
            ),
        },
        "actor_unstable_stress_unit": unstable,
        "all_four_sentinel_paired_exact_flip_count": paired_flips,
        "used_in_alpha_zero_gate": False,
        "any_single_flip_aborts": False,
        "any_per_unit_eight_of_eight_failure_aborts": False,
        "actor_unstable_stress_unit_aborts": False,
    }


def build_analysis_v62a(evidence: dict) -> dict:
    rows = validate_evidence_v62a(evidence)
    generation = _generation_array_v62a(rows)
    delta, receipts = paired_deltas_v62a(generation[:64])
    primary = primary_f1_bootstrap_v62a(delta[..., 0])
    actor = actor_influence_v62a(delta[..., 0])
    gate = gate_v62a(primary, actor)
    exact = _exact_diagnostics_v62a(generation[64:], rows[64:])
    value = {
        "schema": "v62a-pre-hpo-alpha-zero-generation-only-analysis",
        "status": (
            "complete_gate_passed_hpo_still_unauthorized"
            if gate["passed"]
            else "complete_gate_failed_closed"
        ),
        "source_evidence_content_sha256": evidence[
            "content_sha256_before_self_field"
        ],
        "v62_methodology_commit": V62_METHOD_COMMIT,
        "v62_numeric_audit_identities": dict(V62_NUMERIC_AUDIT_IDENTITIES),
        "v62_preregistration_identities": dict(
            V62_PREREGISTRATION_IDENTITIES
        ),
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
    value["content_sha256_before_self_field"] = canonical_sha256_v62a(value)
    return value
