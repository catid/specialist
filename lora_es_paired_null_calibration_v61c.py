#!/usr/bin/env python3
"""Pure analysis for V61C identical-state paired evaluator calibration."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


ROWS_V61C = 68
RANKING_UNITS_V61C = 64
EXACT_SENTINEL_UNITS_V61C = 4
ACTORS_V61C = 4
PERIODS_V61C = 4
PAIRS_PER_ACTOR_V61C = 2
COMMON_GENERATION_SEED_V61C = 2_026_071_601
GENERATION_PARAMS_WITHOUT_SEED_V61C = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 64,
    "detokenize": True,
}
TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61C = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 1,
    "prompt_logprobs": 1,
    "detokenize": False,
}
LABEL_PLAN_V61C = {
    "0": ["reference", "candidate", "candidate", "reference"],
    "1": ["candidate", "reference", "reference", "candidate"],
    "2": ["reference", "candidate", "candidate", "reference"],
    "3": ["candidate", "reference", "reference", "candidate"],
}
REQUEST_TYPE_ORDER_V61C = {
    "0": ["generation", "teacher_forced"],
    "1": ["teacher_forced", "generation"],
    "2": ["generation", "teacher_forced"],
    "3": ["teacher_forced", "generation"],
}
PAIR_PERIODS_V61C = ((0, 1), (2, 3))
F1_INSTABILITY_BANDS_V61C = (0.01, 0.05, 0.10)
LOGPROB_INSTABILITY_BANDS_V61C = (1e-8, 1e-6, 1e-5, 1e-4, 1e-3)
BOOTSTRAP_REPLICATES_V61C = 4096
BOOTSTRAP_ALPHA_V61C = 0.05
BOOTSTRAP_SEED_V61C = 2_026_071_612
SINGLE_REPLICA_DIAGNOSTIC_BOOTSTRAP_SEED_V61C = 2_026_071_613
F1_PRACTICAL_EFFECT_SCALE_V61C = 0.01
LOGPROB_PRACTICAL_EFFECT_SCALE_V61C = 0.001
LOGPROB_PRIMARY_MAX_ABSOLUTE_POINT_NULL_V61C = 0.00025
LOGPROB_PRIMARY_MAX_CI_HALFWIDTH_V61C = 0.001
LOGPROB_PRIMARY_MAX_REPEAT_MEAN_ABSOLUTE_DELTA_V61C = 0.001


def canonical_sha256_v61c(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _generation_metric_v61c(value: dict) -> tuple[float, int, int]:
    if set(value) != {"f1", "exact", "nonzero"}:
        raise ValueError("v61c generation metric schema changed")
    f1 = value.get("f1"); exact = value.get("exact"); nonzero = value.get("nonzero")
    if (
        isinstance(f1, bool) or not isinstance(f1, (int, float))
        or not math.isfinite(float(f1)) or not 0.0 <= float(f1) <= 1.0
        or exact not in (0, 1) or nonzero not in (0, 1)
        or exact > nonzero or nonzero != int(float(f1) > 0.0)
        or (exact == 1 and not math.isclose(float(f1), 1.0))
    ):
        raise ValueError("v61c generation metric changed")
    return float(f1), int(exact), int(nonzero)


def _teacher_metric_v61c(value: dict) -> tuple[float, int]:
    if set(value) != {
        "mean_answer_token_logprob", "answer_token_count", "numeric_example_sha256",
    }:
        raise ValueError("v61c teacher metric schema changed")
    score = value.get("mean_answer_token_logprob")
    tokens = value.get("answer_token_count")
    identity = value.get("numeric_example_sha256")
    if (
        isinstance(score, bool) or not isinstance(score, (int, float))
        or not math.isfinite(float(score)) or float(score) > 1e-8
        or not isinstance(tokens, int) or tokens <= 0
        or not isinstance(identity, str) or len(identity) != 64
    ):
        raise ValueError("v61c teacher metric changed")
    return float(score), tokens


def validate_evidence_v61c(evidence: dict) -> list[dict]:
    rows = list(evidence.get("rows", []))
    if (
        evidence.get("schema") != "v61c-identical-state-paired-evaluator-evidence"
        or evidence.get("status") != "complete_alpha_zero_no_update_characterization"
        or evidence.get("row_count") != ROWS_V61C
        or evidence.get("ranking_units") != RANKING_UNITS_V61C
        or evidence.get("exact_sentinel_units") != EXACT_SENTINEL_UNITS_V61C
        or evidence.get("actor_count") != ACTORS_V61C
        or evidence.get("period_count") != PERIODS_V61C
        or evidence.get("label_plan") != LABEL_PLAN_V61C
        or evidence.get("request_type_order") != REQUEST_TYPE_ORDER_V61C
        or evidence.get("common_generation_seed") != COMMON_GENERATION_SEED_V61C
        or evidence.get("generation_params_without_seed")
        != GENERATION_PARAMS_WITHOUT_SEED_V61C
        or evidence.get("teacher_forced_params_without_seed")
        != TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61C
        or evidence.get("alpha") != 0.0
        or evidence.get("adapter_update_or_candidate_materialization_performed") is not False
        or evidence.get("holdback_semantics_opened") is not False
        or evidence.get("raw_question_answer_or_generation_text_persisted") is not False
        or evidence.get("protected_semantics_opened") is not False
        or len(rows) != ROWS_V61C
    ):
        raise ValueError("v61c evidence contract changed")
    row_ids = []
    unit_ids = []
    for request_index, row in enumerate(rows):
        row_sha = row.get("row_sha256"); unit_sha = row.get("unit_identity_sha256")
        role = row.get("role"); periods = row.get("periods", [])
        if (
            row.get("request_index") != request_index
            or not isinstance(row_sha, str) or len(row_sha) != 64
            or not isinstance(unit_sha, str) or len(unit_sha) != 64
            or role not in ("ranking", "exact_sentinel")
            or [item.get("period_index") for item in periods] != list(range(4))
        ):
            raise ValueError("v61c row/period identity changed")
        for period_index, period in enumerate(periods):
            actors = period.get("actors", [])
            if (
                set(period) != {"period_index", "request_type_order", "actors"}
                or period["request_type_order"]
                != REQUEST_TYPE_ORDER_V61C[str(period_index)]
                or [item.get("actor_rank") for item in actors] != list(range(4))
            ):
                raise ValueError("v61c period/actor coverage changed")
            for actor_rank, actor in enumerate(actors):
                if (
                    set(actor) != {
                        "actor_rank", "label", "generation", "teacher_forced",
                    }
                    or actor["actor_rank"] != actor_rank
                    or actor["label"] != LABEL_PLAN_V61C[str(actor_rank)][period_index]
                ):
                    raise ValueError("v61c counterbalance label changed")
                _generation_metric_v61c(actor["generation"])
                _teacher_metric_v61c(actor["teacher_forced"])
        row_ids.append(row_sha); unit_ids.append(unit_sha)
    if (
        len(set(row_ids)) != 68 or len(set(unit_ids)) != 68
        or [row["role"] for row in rows[:64]] != ["ranking"] * 64
        or [row["role"] for row in rows[64:]] != ["exact_sentinel"] * 4
    ):
        raise ValueError("v61c ranking/sentinel order changed")
    return rows


def _metric_arrays_v61c(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    generation = np.empty((len(rows), 4, 4, 3), dtype=np.float64)
    teacher = np.empty((len(rows), 4, 4), dtype=np.float64)
    for unit, row in enumerate(rows):
        for period, period_value in enumerate(row["periods"]):
            for actor, value in enumerate(period_value["actors"]):
                generation[unit, actor, period] = _generation_metric_v61c(
                    value["generation"]
                )
                teacher[unit, actor, period] = _teacher_metric_v61c(
                    value["teacher_forced"]
                )[0]
    return generation, teacher


def _paired_deltas_v61c(
    generation: np.ndarray, teacher: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    units = generation.shape[0]
    gen_delta = np.empty((units, 4, 2, 3), dtype=np.float64)
    teacher_delta = np.empty((units, 4, 2), dtype=np.float64)
    pair_receipts = []
    for actor in range(4):
        labels = LABEL_PLAN_V61C[str(actor)]
        for pair_index, periods in enumerate(PAIR_PERIODS_V61C):
            reference = next(period for period in periods if labels[period] == "reference")
            candidate = next(period for period in periods if labels[period] == "candidate")
            gen_delta[:, actor, pair_index] = (
                generation[:, actor, candidate] - generation[:, actor, reference]
            )
            teacher_delta[:, actor, pair_index] = (
                teacher[:, actor, candidate] - teacher[:, actor, reference]
            )
            pair_receipts.append({
                "actor_rank": actor, "pair_index": pair_index,
                "reference_period": reference, "candidate_period": candidate,
                "candidate_after_reference": candidate > reference,
            })
    return gen_delta, teacher_delta, pair_receipts


def _quantile_v61c(values: np.ndarray, probability: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    return float(ordered[int(math.floor(probability * (len(ordered) - 1)))])


def paired_block_bootstrap_v61c(
    generation_delta: np.ndarray, teacher_delta: np.ndarray,
    *, replicates: int = BOOTSTRAP_REPLICATES_V61C,
    seed: int = BOOTSTRAP_SEED_V61C,
) -> dict:
    generation_delta = np.asarray(generation_delta, dtype=np.float64)
    teacher_delta = np.asarray(teacher_delta, dtype=np.float64)
    if (
        generation_delta.ndim != 4 or generation_delta.shape[1:] != (4, 2, 3)
        or teacher_delta.shape != generation_delta.shape[:3]
        or generation_delta.shape[0] < 4 or replicates < 100
        or not np.isfinite(generation_delta).all()
        or not np.isfinite(teacher_delta).all()
    ):
        raise ValueError("v61c paired bootstrap arrays changed")
    units = generation_delta.shape[0]
    gen = generation_delta.reshape(units, 8, 3)
    teacher = teacher_delta.reshape(units, 8)
    # The intended HPO estimator averages all four actors and both paired
    # periods for every conflict unit.  Preserve that within-unit dependence
    # in the primary cluster bootstrap by resampling unit means, not individual
    # actor/pair replicas.
    gen_unit_mean = np.mean(gen, axis=1)
    teacher_unit_mean = np.mean(teacher, axis=1)
    point = {
        "generated_f1_delta": float(np.mean(gen[..., 0])),
        "generated_exact_delta": float(np.mean(gen[..., 1])),
        "generated_nonzero_delta": float(np.mean(gen[..., 2])),
        "teacher_forced_logprob_delta": float(np.mean(teacher)),
    }
    primary_rng = np.random.Generator(np.random.PCG64(int(seed)))
    diagnostic_seed = (
        SINGLE_REPLICA_DIAGNOSTIC_BOOTSTRAP_SEED_V61C
        if int(seed) == BOOTSTRAP_SEED_V61C else int(seed) + 1
    )
    diagnostic_rng = np.random.Generator(np.random.PCG64(diagnostic_seed))
    primary_samples = {key: np.empty(replicates) for key in point}
    single_replica_samples = {key: np.empty(replicates) for key in point}
    for index in range(replicates):
        unit_index = primary_rng.integers(0, units, size=units)
        primary_gen = gen_unit_mean[unit_index]
        primary_teacher = teacher_unit_mean[unit_index]
        primary_samples["generated_f1_delta"][index] = np.mean(primary_gen[:, 0])
        primary_samples["generated_exact_delta"][index] = np.mean(primary_gen[:, 1])
        primary_samples["generated_nonzero_delta"][index] = np.mean(primary_gen[:, 2])
        primary_samples["teacher_forced_logprob_delta"][index] = np.mean(
            primary_teacher
        )

        # Secondary diagnostic for a hypothetical future evaluator that kept
        # one actor/pair replica per unit.  It is never used for V61C primary
        # eligibility or for the intended eight-replica V61 HPO estimator.
        diagnostic_unit_index = diagnostic_rng.integers(0, units, size=units)
        replica_index = diagnostic_rng.integers(0, 8, size=units)
        selected_gen = gen[diagnostic_unit_index, replica_index]
        selected_teacher = teacher[diagnostic_unit_index, replica_index]
        single_replica_samples["generated_f1_delta"][index] = np.mean(
            selected_gen[:, 0]
        )
        single_replica_samples["generated_exact_delta"][index] = np.mean(
            selected_gen[:, 1]
        )
        single_replica_samples["generated_nonzero_delta"][index] = np.mean(
            selected_gen[:, 2]
        )
        single_replica_samples["teacher_forced_logprob_delta"][index] = np.mean(
            selected_teacher
        )

    def intervals(samples: dict[str, np.ndarray]) -> dict:
        result = {key: {
            "lcb": _quantile_v61c(values, BOOTSTRAP_ALPHA_V61C),
            "ucb": _quantile_v61c(values, 1.0 - BOOTSTRAP_ALPHA_V61C),
        } for key, values in samples.items()}
        for interval in result.values():
            interval["halfwidth"] = (interval["ucb"] - interval["lcb"]) / 2.0
            interval["contains_zero"] = interval["lcb"] <= 0.0 <= interval["ucb"]
        return result

    return {
        "schema": "v61c-paired-conflict-unit-actor-pair-null-bootstrap",
        "units": units, "replicas_per_unit": 8,
        "bootstrap_replicates": replicates, "bootstrap_seed": int(seed),
        "one_sided_alpha": BOOTSTRAP_ALPHA_V61C,
        "point": point,
        "primary_conflict_unit_cluster_bootstrap": {
            "estimand": (
                "mean conflict-unit delta after averaging all four actors and "
                "both paired periods within every selected unit"
            ),
            "resampled_axis": "conflict_unit",
            "within_unit_actor_pair_replicas_preserved_and_averaged": 8,
            "used_for_v61c_logprob_primary_eligibility": True,
            "intervals": intervals(primary_samples),
        },
        "future_single_replica_noise_diagnostic": {
            "estimand": (
                "hypothetical mean using one uniformly sampled actor/pair "
                "replica per resampled conflict unit"
            ),
            "resampled_axis": "conflict_unit_then_one_actor_pair_replica",
            "bootstrap_seed": diagnostic_seed,
            "used_for_v61c_logprob_primary_eligibility": False,
            "not_the_intended_eight_replica_hpo_estimator": True,
            "intervals": intervals(single_replica_samples),
        },
    }


def _band_counts_v61c(values: np.ndarray, bands: tuple[float, ...]) -> dict:
    return {str(band): int(np.sum(values > band)) for band in bands}


def build_analysis_v61c(evidence: dict) -> dict:
    rows = validate_evidence_v61c(evidence)
    generation, teacher = _metric_arrays_v61c(rows)
    ranking_generation = generation[:64]; ranking_teacher = teacher[:64]
    sentinel_generation = generation[64:]; sentinel_teacher = teacher[64:]
    gen_delta, teacher_delta, receipts = _paired_deltas_v61c(
        ranking_generation, ranking_teacher
    )
    ranking_bootstrap = paired_block_bootstrap_v61c(gen_delta, teacher_delta)
    sentinel_gen_delta, sentinel_teacher_delta, sentinel_receipts = _paired_deltas_v61c(
        sentinel_generation, sentinel_teacher
    )
    sentinel_bootstrap = paired_block_bootstrap_v61c(
        sentinel_gen_delta, sentinel_teacher_delta
    )

    repeat = []
    max_logprob_repeat_mean = 0.0
    for actor in range(4):
        labels = LABEL_PLAN_V61C[str(actor)]
        for label in ("reference", "candidate"):
            periods = [index for index, value in enumerate(labels) if value == label]
            f1_delta = np.abs(
                ranking_generation[:, actor, periods[0], 0]
                - ranking_generation[:, actor, periods[1], 0]
            )
            logprob_delta = np.abs(
                ranking_teacher[:, actor, periods[0]]
                - ranking_teacher[:, actor, periods[1]]
            )
            logprob_mean = float(np.mean(logprob_delta))
            max_logprob_repeat_mean = max(max_logprob_repeat_mean, logprob_mean)
            repeat.append({
                "actor_rank": actor, "label": label, "periods": periods,
                "generated_f1_mean_absolute_delta": float(np.mean(f1_delta)),
                "generated_f1_maximum_absolute_delta": float(np.max(f1_delta)),
                "generated_f1_band_counts": _band_counts_v61c(
                    f1_delta, F1_INSTABILITY_BANDS_V61C
                ),
                "teacher_logprob_mean_absolute_delta": logprob_mean,
                "teacher_logprob_maximum_absolute_delta": float(np.max(logprob_delta)),
                "teacher_logprob_band_counts": _band_counts_v61c(
                    logprob_delta, LOGPROB_INSTABILITY_BANDS_V61C
                ),
            })
    primary_intervals = ranking_bootstrap[
        "primary_conflict_unit_cluster_bootstrap"
    ]["intervals"]
    gen_interval = primary_intervals["generated_f1_delta"]
    log_interval = primary_intervals["teacher_forced_logprob_delta"]
    generated_normalized = gen_interval["halfwidth"] / F1_PRACTICAL_EFFECT_SCALE_V61C
    logprob_normalized = log_interval["halfwidth"] / LOGPROB_PRACTICAL_EFFECT_SCALE_V61C
    eligibility_checks = {
        "teacher_null_interval_contains_zero": log_interval["contains_zero"],
        "teacher_absolute_point_within_limit": abs(
            ranking_bootstrap["point"]["teacher_forced_logprob_delta"]
        ) <= LOGPROB_PRIMARY_MAX_ABSOLUTE_POINT_NULL_V61C,
        "teacher_interval_halfwidth_within_limit": (
            log_interval["halfwidth"] <= LOGPROB_PRIMARY_MAX_CI_HALFWIDTH_V61C
        ),
        "teacher_repeat_mean_absolute_delta_within_limit": (
            max_logprob_repeat_mean
            <= LOGPROB_PRIMARY_MAX_REPEAT_MEAN_ABSOLUTE_DELTA_V61C
        ),
        "teacher_normalized_null_halfwidth_not_above_generated_f1": (
            logprob_normalized <= generated_normalized
        ),
    }
    sentinel_exact_delta = sentinel_gen_delta[..., 1]
    nonzero_sentinel_exact = np.abs(sentinel_exact_delta) > 0.0
    sentinel_checks = {
        "zero_every_individual_paired_exact_delta": bool(
            np.all(sentinel_exact_delta == 0.0)
        ),
        # These aggregate checks are diagnostics only: unlike the individual
        # check above, they can be fooled by +1/-1 cancellation.
        "zero_total_paired_exact_delta": float(np.sum(sentinel_exact_delta)) == 0.0,
        "zero_per_unit_paired_exact_delta": bool(
            np.all(np.sum(sentinel_exact_delta, axis=(1, 2)) == 0.0)
        ),
        "four_sparse_exact_units": sentinel_gen_delta.shape[0] == 4,
    }
    result = {
        "schema": "v61c-identical-state-paired-evaluator-null-analysis",
        "status": "complete_characterization_only",
        "ranking_bootstrap": ranking_bootstrap,
        "within_actor_same_label_repeat": repeat,
        "counterbalance_pair_receipts": receipts,
        "noise_scale_comparison": {
            "generated_f1_practical_effect_scale": F1_PRACTICAL_EFFECT_SCALE_V61C,
            "teacher_logprob_practical_effect_scale": LOGPROB_PRACTICAL_EFFECT_SCALE_V61C,
            "generated_f1_normalized_null_ci_halfwidth": generated_normalized,
            "teacher_logprob_normalized_null_ci_halfwidth": logprob_normalized,
            "teacher_forced_logprob_primary_eligible": all(
                eligibility_checks.values()
            ),
            "eligibility_checks": eligibility_checks,
            "thresholds_frozen_before_live_access": True,
        },
        "exact_sentinel": {
            "bootstrap": sentinel_bootstrap,
            "pair_receipts": sentinel_receipts,
            "checks": sentinel_checks,
            "nonzero_individual_paired_exact_delta_count": int(
                np.sum(nonzero_sentinel_exact)
            ),
            "maximum_absolute_individual_paired_exact_delta": float(
                np.max(np.abs(sentinel_exact_delta))
            ),
            "passed": (
                sentinel_checks["zero_every_individual_paired_exact_delta"]
                and sentinel_checks["four_sparse_exact_units"]
            ),
            "used_in_primary_fitness": False,
        },
        "alpha": 0.0,
        "adapter_update_or_candidate_materialization_performed": False,
        "holdback_semantics_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
        "source_evidence_content_sha256": evidence.get(
            "content_sha256_before_self_field"
        ),
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v61c(result)
    return result
