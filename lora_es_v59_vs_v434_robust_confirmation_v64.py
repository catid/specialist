#!/usr/bin/env python3
"""Pure numeric analysis for the corrected V64 V59-versus-V434 confirmation.

V64 preserves V63's frozen V62B schedule, estimator, and scientific gates,
but corrects V63's raw-candidate serving namespace.  Evidence must now include
exact applied-weight receipts tying both immutable staged adapters to all 23
live modules and 82 GPU slot views.  The actor-unstable sentinel remains a
diagnostic and can never select or reject the candidate.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

import lora_es_pre_hpo_alpha_zero_calibration_v62b as v62b


ROWS_V64 = v62b.ROWS_V62B
RANKING_UNITS_V64 = v62b.RANKING_UNITS_V62B
EXACT_SENTINEL_UNITS_V64 = v62b.EXACT_SENTINEL_UNITS_V62B
ACTORS_V64 = v62b.ACTORS_V62B
WARMUP_PERIODS_V64 = v62b.WARMUP_PERIODS_V62B
SCORED_BLOCKS_V64 = v62b.SCORED_BLOCKS_V62B
PERIODS_PER_BLOCK_V64 = v62b.PERIODS_PER_BLOCK_V62B
SCORED_PERIODS_V64 = v62b.SCORED_PERIODS_V62B
TOTAL_PERIODS_V64 = v62b.TOTAL_PERIODS_V62B
PAIRS_PER_ACTOR_V64 = v62b.PAIRS_PER_ACTOR_V62B
REPLICAS_PER_UNIT_V64 = v62b.REPLICAS_PER_UNIT_V62B
WARMUP_GENERATION_COMPLETIONS_V64 = (
    v62b.WARMUP_GENERATION_COMPLETIONS_V62B
)
SCORED_GENERATION_COMPLETIONS_V64 = (
    v62b.SCORED_GENERATION_COMPLETIONS_V62B
)
TOTAL_GENERATION_COMPLETIONS_V64 = v62b.TOTAL_GENERATION_COMPLETIONS_V62B
COMMON_GENERATION_SEED_V64 = v62b.COMMON_GENERATION_SEED_V62B
GENERATION_PARAMS_WITHOUT_SEED_V64 = dict(
    v62b.GENERATION_PARAMS_WITHOUT_SEED_V62B
)
WARMUP_LABEL_PLAN_V64 = {
    actor: list(labels)
    for actor, labels in v62b.WARMUP_LABEL_PLAN_V62B.items()
}
LABEL_PLAN_V64 = {
    actor: list(labels) for actor, labels in v62b.LABEL_PLAN_V62B.items()
}
PAIR_PERIODS_V64 = tuple(v62b.PAIR_PERIODS_V62B)
RUNTIME_CONTROLS_V64 = {
    **v62b.RUNTIME_CONTROLS_V62B,
    "max_loras": 1,
    "max_cpu_loras": 2,
}
BOOTSTRAP_REPLICATES_V64 = v62b.BOOTSTRAP_REPLICATES_V62B
BOOTSTRAP_SEED_V64 = v62b.BOOTSTRAP_SEED_V62B
ONE_SIDED_ALPHA_V64 = v62b.ONE_SIDED_ALPHA_V62B
STAGED_DATASET_FILE_SHA256_V64 = v62b.STAGED_DATASET_FILE_SHA256_V62B
STAGED_PANEL_FILE_SHA256_V64 = v62b.STAGED_PANEL_FILE_SHA256_V62B
STAGED_PANEL_CONTENT_SHA256_V64 = v62b.STAGED_PANEL_CONTENT_SHA256_V62B
STABLE_EXACT_UNIT_SHA256_V64 = tuple(v62b.STABLE_EXACT_UNIT_SHA256_V62B)
ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V64 = (
    v62b.ACTOR_UNSTABLE_EXACT_UNIT_SHA256_V62B
)
SENTINEL_ORDER_V64 = tuple(v62b.SENTINEL_ORDER_V62B)

FROZEN_NULL_WIDTH_V64 = 0.000773822590292528
MINIMUM_POINT_IMPROVEMENT_V64 = 0.001547645180585056
if MINIMUM_POINT_IMPROVEMENT_V64 != 2.0 * FROZEN_NULL_WIDTH_V64:
    raise RuntimeError("v64 point-improvement threshold arithmetic changed")

V62B_FINALIZER_COMMIT_V64 = (
    "44208cb7c7c4bd27a038f9d3ebb9041c237ecea7"
)
V62B_FINALIZED_FILE_SHA256_V64 = (
    "92b7e847ef42b06735d29d2a3f345a8c1cc8233c8408395de8d7016e9838ae72"
)
V62B_FINALIZED_CONTENT_SHA256_V64 = (
    "f05506bfcca63bf2723b10518708897ab871b1073f654ed8a221608aaacd2149"
)

REFERENCE_CANONICAL_STATE_SHA256_V64 = (
    "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
)
REFERENCE_RUNTIME_VALUES_SHA256_V64 = (
    "a1353c47bc11f02a9b67d7859d6670b07d6754c285ac4f357255878c09384f5b"
)
REFERENCE_STAGED_WEIGHTS_SHA256_V64 = (
    "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a"
)
REFERENCE_CONFIG_SHA256_V64 = (
    "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
)
CANDIDATE_WEIGHTS_SHA256_V64 = (
    "e189cfb9fcd6c55700babae91111266825522fc46ddbd53fc0574786711eccae"
)
CANDIDATE_CONFIG_SHA256_V64 = REFERENCE_CONFIG_SHA256_V64
CANDIDATE_CANONICAL_STATE_SHA256_V64 = (
    "1713987fcad93f3e6368a309415faf5de2f4230eaf3c44baf23b8e9a2edf2a3d"
)
CANDIDATE_RUNTIME_VALUES_SHA256_V64 = (
    "ad5dd995de7cad3c9d116d64deb3aa67b9db46fbdf4e3f8a6ab5ee37340b5923"
)
CANONICAL_ORDERED_KEY_SHA256_V64 = (
    "ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280"
)
RUNTIME_MODULE_MANIFEST_SHA256_V64 = (
    "f09f656d7890c8776170bcc65e9273fbafefad2651a9ab6bc2ef805dfae6eeca"
)
REFERENCE_LORA_ID_V64 = 1
CANDIDATE_LORA_ID_V64 = 2


def canonical_sha256_v64(value: Any) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _is_sha256_v64(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def expected_adapter_identities_v64() -> dict[str, dict]:
    """Return the immutable identities used by every V64 period."""
    return {
        "reference": {
            "label": "reference",
            "adapter": "V434",
            "lora_int_id": REFERENCE_LORA_ID_V64,
            "weights_file_sha256": REFERENCE_STAGED_WEIGHTS_SHA256_V64,
            "config_file_sha256": REFERENCE_CONFIG_SHA256_V64,
            "canonical_fp32_state_sha256": (
                REFERENCE_CANONICAL_STATE_SHA256_V64
            ),
            "runtime_bf16_values_sha256": (
                REFERENCE_RUNTIME_VALUES_SHA256_V64
            ),
        },
        "candidate": {
            "label": "candidate",
            "adapter": "V59",
            "lora_int_id": CANDIDATE_LORA_ID_V64,
            "weights_file_sha256": CANDIDATE_WEIGHTS_SHA256_V64,
            "config_file_sha256": CANDIDATE_CONFIG_SHA256_V64,
            "canonical_fp32_state_sha256": (
                CANDIDATE_CANONICAL_STATE_SHA256_V64
            ),
            "runtime_bf16_values_sha256": (
                CANDIDATE_RUNTIME_VALUES_SHA256_V64
            ),
        },
    }


def _validate_state_receipts_v64(
    receipts: object,
    *,
    period_kind: str,
    expected_count: int,
) -> list[dict]:
    if not isinstance(receipts, list) or len(receipts) != expected_count:
        raise ValueError(f"v64 {period_kind} identity receipt coverage changed")
    expected_identities = expected_adapter_identities_v64()
    label_plan = (
        WARMUP_LABEL_PLAN_V64
        if period_kind == "unscored_warmup"
        else LABEL_PLAN_V64
    )
    for period_index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise ValueError(f"v64 {period_kind} identity receipt changed")
        assignments = receipt.get("actor_request_assignments")
        if (
            set(receipt) != {
                "period_kind",
                "period_index",
                "before",
                "after",
                "actor_request_assignments",
                "active_adapter_receipts",
                "both_adapter_files_exact_and_unchanged",
            }
            or receipt.get("period_kind") != period_kind
            or receipt.get("period_index") != period_index
            or receipt.get("before") != expected_identities
            or receipt.get("after") != expected_identities
            or receipt.get("before") != receipt.get("after")
            or receipt.get("both_adapter_files_exact_and_unchanged") is not True
            or not isinstance(assignments, list)
            or len(assignments) != ACTORS_V64
            or not isinstance(receipt.get("active_adapter_receipts"), list)
            or len(receipt["active_adapter_receipts"]) != ACTORS_V64
        ):
            raise ValueError(f"v64 {period_kind} adapter identity changed")
        for actor_rank, assignment in enumerate(assignments):
            label = label_plan[str(actor_rank)][period_index]
            expected = expected_identities[label]
            if assignment != {
                "actor_rank": actor_rank,
                "label": label,
                "adapter": expected["adapter"],
                "lora_int_id": expected["lora_int_id"],
                "weights_file_sha256": expected["weights_file_sha256"],
                "config_file_sha256": expected["config_file_sha256"],
                "canonical_fp32_state_sha256": expected[
                    "canonical_fp32_state_sha256"
                ],
                "runtime_bf16_values_sha256": expected[
                    "runtime_bf16_values_sha256"
                ],
            }:
                raise ValueError(
                    f"v64 {period_kind} actor request identity changed"
                )
            active = receipt["active_adapter_receipts"][actor_rank]
            loaded = active.get("loaded_cpu_cache_lora_ids", [])
            if (
                set(active) != {
                    "actor_rank", "schema", "expected_lora_int_id",
                    "active_lora_ids", "active_manager_cache_lora_ids",
                    "loaded_cpu_cache_lora_ids", "active_slot_index",
                    "facade_type", "manager_type",
                    "staged_weights_file_sha256",
                    "canonical_fp32_state_sha256",
                    "canonical_ordered_key_sha256", "canonical_tensor_count",
                    "canonical_elements", "registered_lora_module_count",
                    "matched_live_lora_module_count",
                    "unmatched_registered_lora_module_count",
                    "runtime_module_manifest_sha256",
                    "source_linked_runtime_view_count",
                    "source_linked_runtime_elements",
                    "source_linked_runtime_dtype",
                    "source_linked_runtime_values_sha256",
                    "registered_slot_view_count",
                    "registered_slot_records_sha256",
                    "exact_staged_fp32_to_gpu_slot_equality",
                    "exact_registered_postpack_to_gpu_slot_equality",
                    "active_matches_expected", "max_loras", "max_cpu_loras",
                }
                or active.get("actor_rank") != actor_rank
                or active.get("schema")
                != "v64-effective-applied-lora-receipt"
                or active.get("expected_lora_int_id") != expected["lora_int_id"]
                or active.get("active_lora_ids") != [expected["lora_int_id"]]
                or active.get("active_manager_cache_lora_ids")
                != [expected["lora_int_id"]]
                or active.get("active_slot_index") != 0
                or active.get("facade_type") != "LRUCacheWorkerLoRAManager"
                or active.get("manager_type") != "LRUCacheLoRAModelManager"
                or active.get("staged_weights_file_sha256")
                != expected["weights_file_sha256"]
                or active.get("canonical_fp32_state_sha256")
                != expected["canonical_fp32_state_sha256"]
                or active.get("canonical_ordered_key_sha256")
                != CANONICAL_ORDERED_KEY_SHA256_V64
                or active.get("canonical_tensor_count") != 70
                or active.get("canonical_elements") != 4_528_128
                or active.get("registered_lora_module_count") != 23
                or active.get("matched_live_lora_module_count") != 23
                or active.get("unmatched_registered_lora_module_count") != 0
                or active.get("runtime_module_manifest_sha256")
                != RUNTIME_MODULE_MANIFEST_SHA256_V64
                or active.get("source_linked_runtime_view_count") != 82
                or active.get("source_linked_runtime_elements") != 4_921_344
                or active.get("source_linked_runtime_dtype") != "torch.bfloat16"
                or active.get("source_linked_runtime_values_sha256")
                != expected["runtime_bf16_values_sha256"]
                or active.get("registered_slot_view_count") != 82
                or not _is_sha256_v64(
                    active.get("registered_slot_records_sha256")
                )
                or active.get("exact_staged_fp32_to_gpu_slot_equality") is not True
                or active.get("exact_registered_postpack_to_gpu_slot_equality")
                is not True
                or active.get("active_matches_expected") is not True
                or active.get("max_loras") != 1
                or active.get("max_cpu_loras") != 2
                or not isinstance(loaded, list)
                or loaded != sorted(set(loaded))
                or not 1 <= len(loaded) <= 2
                or not set(loaded).issubset({1, 2})
                or expected["lora_int_id"] not in loaded
            ):
                raise ValueError(
                    f"v64 {period_kind} effective active adapter changed"
                )
    return receipts


def _generation_metric_v64(value: dict) -> tuple[float, int, int]:
    try:
        return v62b._generation_metric_v62b(value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("v64 generation metric changed") from error


def validate_evidence_v64(evidence: dict) -> list[dict]:
    if not isinstance(evidence, dict):
        raise ValueError("v64 evidence changed")
    compact = {
        key: item for key, item in evidence.items()
        if key != "content_sha256_before_self_field"
    }
    expected_keys = {
        "schema", "status", "v62b_finalized_artifact", "adapter_identities",
        "staged_dataset_file_sha256", "staged_panel_file_sha256",
        "staged_panel_content_sha256", "row_count", "ranking_units",
        "exact_sentinel_units", "actor_count", "unscored_warmup_period_count",
        "scored_period_count", "total_period_count", "scored_blocks",
        "periods_per_block", "pairs_per_actor", "replicas_per_unit",
        "warmup_label_plan", "scored_label_plan", "pair_periods",
        "common_generation_seed", "generation_params_without_seed",
        "runtime_determinism_controls", "warmup_state_receipts",
        "numeric_warmup_state_receipts_sha256", "scored_state_receipts",
        "numeric_scored_state_receipts_sha256", "rows",
        "numeric_actor_period_manifest_sha256", "generation_only",
        "warmup_generation_completions_discarded",
        "scored_generation_completions", "total_generation_completions",
        "warmup_raw_outputs_persisted",
        "warmup_generation_metrics_computed_or_persisted",
        "warmup_adaptive_retry_drop_or_reorder_performed",
        "scored_period_adaptive_retry_drop_reorder_or_early_stop_performed",
        "teacher_forced_requests", "adapter_update_hpo_or_promotion_performed",
        "holdback_ood_shadow_or_protected_opened",
        "raw_question_answer_or_generation_text_persisted",
        "content_sha256_before_self_field",
    }
    finalized = evidence.get("v62b_finalized_artifact", {})
    rows = evidence.get("rows")
    warmup_receipts = evidence.get("warmup_state_receipts")
    scored_receipts = evidence.get("scored_state_receipts")
    if (
        set(evidence) != expected_keys
        or evidence.get("schema")
        != "v64-v59-vs-v434-train-only-generation-evidence"
        or evidence.get("status") != "complete_fixed_confirmation_schedule"
        or evidence.get("content_sha256_before_self_field")
        != canonical_sha256_v64(compact)
        or finalized != {
            "commit": V62B_FINALIZER_COMMIT_V64,
            "file_sha256": V62B_FINALIZED_FILE_SHA256_V64,
            "content_sha256": V62B_FINALIZED_CONTENT_SHA256_V64,
            "eligible_for_later_separately_preregistered_work": True,
            "launch_or_update_authority": False,
        }
        or evidence.get("adapter_identities")
        != expected_adapter_identities_v64()
        or evidence.get("staged_dataset_file_sha256")
        != STAGED_DATASET_FILE_SHA256_V64
        or evidence.get("staged_panel_file_sha256")
        != STAGED_PANEL_FILE_SHA256_V64
        or evidence.get("staged_panel_content_sha256")
        != STAGED_PANEL_CONTENT_SHA256_V64
        or evidence.get("row_count") != ROWS_V64
        or evidence.get("ranking_units") != RANKING_UNITS_V64
        or evidence.get("exact_sentinel_units") != EXACT_SENTINEL_UNITS_V64
        or evidence.get("actor_count") != ACTORS_V64
        or evidence.get("unscored_warmup_period_count") != WARMUP_PERIODS_V64
        or evidence.get("scored_period_count") != SCORED_PERIODS_V64
        or evidence.get("total_period_count") != TOTAL_PERIODS_V64
        or evidence.get("scored_blocks") != SCORED_BLOCKS_V64
        or evidence.get("periods_per_block") != PERIODS_PER_BLOCK_V64
        or evidence.get("pairs_per_actor") != PAIRS_PER_ACTOR_V64
        or evidence.get("replicas_per_unit") != REPLICAS_PER_UNIT_V64
        or evidence.get("warmup_label_plan") != WARMUP_LABEL_PLAN_V64
        or evidence.get("scored_label_plan") != LABEL_PLAN_V64
        or evidence.get("pair_periods")
        != [list(pair) for pair in PAIR_PERIODS_V64]
        or evidence.get("common_generation_seed") != COMMON_GENERATION_SEED_V64
        or evidence.get("generation_params_without_seed")
        != GENERATION_PARAMS_WITHOUT_SEED_V64
        or evidence.get("runtime_determinism_controls") != RUNTIME_CONTROLS_V64
        or evidence.get("numeric_warmup_state_receipts_sha256")
        != canonical_sha256_v64(warmup_receipts)
        or evidence.get("numeric_scored_state_receipts_sha256")
        != canonical_sha256_v64(scored_receipts)
        or evidence.get("generation_only") is not True
        or evidence.get("warmup_generation_completions_discarded")
        != WARMUP_GENERATION_COMPLETIONS_V64
        or evidence.get("scored_generation_completions")
        != SCORED_GENERATION_COMPLETIONS_V64
        or evidence.get("total_generation_completions")
        != TOTAL_GENERATION_COMPLETIONS_V64
        or evidence.get("warmup_raw_outputs_persisted") is not False
        or evidence.get("warmup_generation_metrics_computed_or_persisted")
        is not False
        or evidence.get("warmup_adaptive_retry_drop_or_reorder_performed")
        is not False
        or evidence.get(
            "scored_period_adaptive_retry_drop_reorder_or_early_stop_performed"
        ) is not False
        or evidence.get("teacher_forced_requests") != 0
        or evidence.get("adapter_update_hpo_or_promotion_performed") is not False
        or evidence.get("holdback_ood_shadow_or_protected_opened") is not False
        or evidence.get("raw_question_answer_or_generation_text_persisted")
        is not False
        or not isinstance(rows, list)
        or len(rows) != ROWS_V64
    ):
        raise ValueError("v64 evidence contract changed")
    _validate_state_receipts_v64(
        warmup_receipts,
        period_kind="unscored_warmup",
        expected_count=WARMUP_PERIODS_V64,
    )
    _validate_state_receipts_v64(
        scored_receipts,
        period_kind="scored",
        expected_count=SCORED_PERIODS_V64,
    )

    row_ids = []
    unit_ids = []
    for request_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError("v64 row changed")
        periods = row.get("scored_periods")
        row_sha = row.get("row_sha256")
        unit_sha = row.get("unit_identity_sha256")
        if (
            set(row) != {
                "request_index", "row_sha256", "unit_identity_sha256",
                "role", "scored_periods",
            }
            or row.get("request_index") != request_index
            or not _is_sha256_v64(row_sha)
            or not _is_sha256_v64(unit_sha)
            or row.get("role")
            != ("ranking" if request_index < RANKING_UNITS_V64
                else "exact_sentinel")
            or not isinstance(periods, list)
            or len(periods) != SCORED_PERIODS_V64
        ):
            raise ValueError("v64 row identity changed")
        for period_index, period in enumerate(periods):
            if not isinstance(period, dict):
                raise ValueError("v64 scored period changed")
            actors = period.get("actors")
            if (
                set(period) != {"period_index", "request_type", "actors"}
                or period.get("period_index") != period_index
                or period.get("request_type") != "generation"
                or not isinstance(actors, list)
                or len(actors) != ACTORS_V64
            ):
                raise ValueError("v64 scored period coverage changed")
            for actor_rank, actor in enumerate(actors):
                label = LABEL_PLAN_V64[str(actor_rank)][period_index]
                if (
                    not isinstance(actor, dict)
                    or set(actor) != {"actor_rank", "label", "generation"}
                    or actor.get("actor_rank") != actor_rank
                    or actor.get("label") != label
                ):
                    raise ValueError("v64 counterbalance label changed")
                _generation_metric_v64(actor.get("generation"))
        row_ids.append(row_sha)
        unit_ids.append(unit_sha)
    if (
        len(set(row_ids)) != ROWS_V64
        or len(set(unit_ids)) != ROWS_V64
        or tuple(unit_ids[RANKING_UNITS_V64:]) != SENTINEL_ORDER_V64
        or evidence.get("numeric_actor_period_manifest_sha256")
        != canonical_sha256_v64(rows)
    ):
        raise ValueError("v64 panel order or numeric manifest changed")
    return rows


def _generation_array_v64(rows: list[dict]) -> np.ndarray:
    values = np.empty(
        (len(rows), ACTORS_V64, SCORED_PERIODS_V64, 3),
        dtype=np.float64,
    )
    for unit, row in enumerate(rows):
        for period, period_value in enumerate(row["scored_periods"]):
            for actor, value in enumerate(period_value["actors"]):
                values[unit, actor, period] = _generation_metric_v64(
                    value["generation"]
                )
    return values


def paired_deltas_v64(
    generation: np.ndarray,
) -> tuple[np.ndarray, list[dict]]:
    return v62b.paired_deltas_v62b(generation)


def primary_f1_bootstrap_v64(f1_delta: np.ndarray) -> dict:
    return v62b.primary_f1_bootstrap_v62b(f1_delta)


def actor_influence_v64(f1_delta: np.ndarray) -> dict:
    return v62b.actor_influence_v62b(f1_delta)


def exact_diagnostics_v64(
    sentinel: np.ndarray,
    sentinel_rows: list[dict],
) -> dict:
    value = v62b._exact_diagnostics_v62b(sentinel, sentinel_rows)
    by_hash = {
        row["unit_identity_sha256"]: index
        for index, row in enumerate(sentinel_rows)
    }
    units = {
        item["unit_identity_sha256"]: item
        for item in (
            value["stable_panel"]["units"]
            + [value["actor_unstable_stress_unit"]]
        )
    }
    for unit_sha in SENTINEL_ORDER_V64:
        unit_index = by_hash[unit_sha]
        reference_nonzero = []
        candidate_nonzero = []
        for actor in range(ACTORS_V64):
            labels = LABEL_PLAN_V64[str(actor)]
            for periods in PAIR_PERIODS_V64:
                reference_period = next(
                    period for period in periods
                    if labels[period] == "reference"
                )
                candidate_period = next(
                    period for period in periods
                    if labels[period] == "candidate"
                )
                reference_nonzero.append(int(
                    sentinel[unit_index, actor, reference_period, 2]
                ))
                candidate_nonzero.append(int(
                    sentinel[unit_index, actor, candidate_period, 2]
                ))
        units[unit_sha].update({
            "reference_nonzero_pass_count_of_48": sum(reference_nonzero),
            "candidate_nonzero_pass_count_of_48": sum(candidate_nonzero),
        })
    stable = value["stable_panel"]
    stable["reference_nonzero_pass_total_of_144"] = sum(
        item["reference_nonzero_pass_count_of_48"]
        for item in stable["units"]
    )
    stable["candidate_nonzero_pass_total_of_144"] = sum(
        item["candidate_nonzero_pass_count_of_48"]
        for item in stable["units"]
    )
    value.pop("used_in_alpha_zero_gate")
    value["stable_panel_used_in_required_gate"] = True
    value["actor_unstable_stress_unit_used_in_required_gate"] = False
    value["majority_or_consensus_selection_performed"] = False
    return value


def robust_gate_v64(primary: dict, actor: dict, exact: dict) -> dict:
    maximum_shift = actor["maximum_absolute_leave_one_actor_out_shift"]
    robust_fitness = primary["lcb"] - maximum_shift
    stable = exact["stable_panel"]
    units = stable["units"]
    per_unit = [
        item["candidate_exact_pass_count_of_48"]
        >= item["reference_exact_pass_count_of_48"]
        for item in units
    ]
    checks = {
        "robust_fitness_lcb_minus_max_actor_loo_shift_strictly_positive": (
            robust_fitness > 0.0
        ),
        "point_improvement_at_least_twice_frozen_null_width_inclusive": (
            primary["point"] >= MINIMUM_POINT_IMPROVEMENT_V64
        ),
        "stable_exact_aggregate_candidate_noninferior_inclusive": (
            stable["candidate_exact_pass_total_of_144"]
            >= stable["reference_exact_pass_total_of_144"]
        ),
        "stable_exact_every_unit_candidate_noninferior_inclusive": all(per_unit),
        "stable_nonzero_aggregate_candidate_noninferior_inclusive": (
            stable["candidate_nonzero_pass_total_of_144"]
            >= stable["reference_nonzero_pass_total_of_144"]
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "robust_fitness": robust_fitness,
        "robust_fitness_definition": (
            "paired generated-F1 one-sided LCB minus maximum absolute "
            "actor leave-one-out point shift"
        ),
        "minimum_point_improvement_inclusive": MINIMUM_POINT_IMPROVEMENT_V64,
        "frozen_null_width": FROZEN_NULL_WIDTH_V64,
        "stable_per_unit_noninferiority": [
            {
                "unit_identity_sha256": item["unit_identity_sha256"],
                "candidate_at_least_reference": passed,
            }
            for item, passed in zip(units, per_unit, strict=True)
        ],
        "actor_unstable_stress_unit_is_diagnostic_only": True,
        "median_consensus_or_best_of_selection_performed": False,
        "failure_action": "fail_closed_without_update_hpo_or_promotion",
        "authorizes_update_hpo_promotion_or_protected_access": False,
    }


def build_analysis_v64(evidence: dict) -> dict:
    rows = validate_evidence_v64(evidence)
    generation = _generation_array_v64(rows)
    delta, receipts = paired_deltas_v64(generation[:RANKING_UNITS_V64])
    primary = primary_f1_bootstrap_v64(delta[..., 0])
    actor = actor_influence_v64(delta[..., 0])
    exact = exact_diagnostics_v64(
        generation[RANKING_UNITS_V64:],
        rows[RANKING_UNITS_V64:],
    )
    gate = robust_gate_v64(primary, actor, exact)
    value = {
        "schema": "v64-v59-vs-v434-train-only-robust-confirmation-analysis",
        "status": (
            "complete_gate_passed_without_promotion_authority"
            if gate["passed"] else "complete_gate_failed_closed"
        ),
        "source_evidence_content_sha256": evidence[
            "content_sha256_before_self_field"
        ],
        "unscored_warmup_excluded_from_every_metric": True,
        "primary_generated_f1": primary,
        "actor_influence": actor,
        "required_confirmation_gate": gate,
        "counterbalance_pair_receipts": receipts,
        "exact_sentinel_diagnostics": exact,
        "generation_only": True,
        "teacher_forced_metric_computed": False,
        "teacher_forced_requests": 0,
        "arithmetic_mean_all_48_replicas_per_unit": True,
        "median_consensus_or_best_of_selection_performed": False,
        "update_hpo_candidate_promotion_or_protected_access_authorized": False,
        "raw_question_answer_or_generation_text_persisted": False,
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v64(value)
    return value
