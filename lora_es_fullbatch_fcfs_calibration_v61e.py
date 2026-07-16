#!/usr/bin/env python3
"""V61E schema wrapper over the unchanged V61C numeric contract."""

from __future__ import annotations

import copy

import lora_es_paired_null_calibration_v61c as v61c


# V61E changes only explicit scheduler controls.  The panel, labels, periods,
# metrics, bootstrap, and every frozen threshold remain the V61C contract.
ROWS_V61E = v61c.ROWS_V61C
RANKING_UNITS_V61E = v61c.RANKING_UNITS_V61C
EXACT_SENTINEL_UNITS_V61E = v61c.EXACT_SENTINEL_UNITS_V61C
ACTORS_V61E = v61c.ACTORS_V61C
PERIODS_V61E = v61c.PERIODS_V61C
LABEL_PLAN_V61E = copy.deepcopy(v61c.LABEL_PLAN_V61C)
REQUEST_TYPE_ORDER_V61E = copy.deepcopy(v61c.REQUEST_TYPE_ORDER_V61C)
PAIR_PERIODS_V61E = tuple(v61c.PAIR_PERIODS_V61C)
COMMON_GENERATION_SEED_V61E = v61c.COMMON_GENERATION_SEED_V61C
GENERATION_PARAMS_WITHOUT_SEED_V61E = copy.deepcopy(
    v61c.GENERATION_PARAMS_WITHOUT_SEED_V61C
)
TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61E = copy.deepcopy(
    v61c.TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61C
)
F1_INSTABILITY_BANDS_V61E = tuple(v61c.F1_INSTABILITY_BANDS_V61C)
LOGPROB_INSTABILITY_BANDS_V61E = tuple(v61c.LOGPROB_INSTABILITY_BANDS_V61C)
BOOTSTRAP_REPLICATES_V61E = v61c.BOOTSTRAP_REPLICATES_V61C
BOOTSTRAP_ALPHA_V61E = v61c.BOOTSTRAP_ALPHA_V61C
BOOTSTRAP_SEED_V61E = v61c.BOOTSTRAP_SEED_V61C
SINGLE_REPLICA_DIAGNOSTIC_BOOTSTRAP_SEED_V61E = (
    v61c.SINGLE_REPLICA_DIAGNOSTIC_BOOTSTRAP_SEED_V61C
)
F1_PRACTICAL_EFFECT_SCALE_V61E = v61c.F1_PRACTICAL_EFFECT_SCALE_V61C
LOGPROB_PRACTICAL_EFFECT_SCALE_V61E = v61c.LOGPROB_PRACTICAL_EFFECT_SCALE_V61C
LOGPROB_PRIMARY_MAX_ABSOLUTE_POINT_NULL_V61E = (
    v61c.LOGPROB_PRIMARY_MAX_ABSOLUTE_POINT_NULL_V61C
)
LOGPROB_PRIMARY_MAX_CI_HALFWIDTH_V61E = (
    v61c.LOGPROB_PRIMARY_MAX_CI_HALFWIDTH_V61C
)
LOGPROB_PRIMARY_MAX_REPEAT_MEAN_ABSOLUTE_DELTA_V61E = (
    v61c.LOGPROB_PRIMARY_MAX_REPEAT_MEAN_ABSOLUTE_DELTA_V61C
)
RUNTIME_CONTROLS_V61E = {
    "VLLM_BATCH_INVARIANT": False,
    "async_scheduling": False,
    "max_num_seqs": 68,
    "scheduling_policy": "fcfs",
    "enforce_eager": True,
}

canonical_sha256_v61e = v61c.canonical_sha256_v61c
paired_block_bootstrap_v61e = v61c.paired_block_bootstrap_v61c


def _as_v61c_evidence(value: dict) -> dict:
    converted = copy.deepcopy(value)
    if (
        converted.get("schema")
        != "v61e-fullbatch-fcfs-identical-state-paired-evaluator-evidence"
        or converted.get("status")
        != "complete_matched_alpha_zero_no_update_characterization"
        or converted.get("runtime_determinism_controls") != RUNTIME_CONTROLS_V61E
        or converted.get("matched_v61c_panel_labels_metrics_bootstrap_thresholds")
        is not True
        or converted.get("v61c_effective_request_batch_size") != 68
    ):
        raise ValueError("v61e evidence scheduling provenance changed")
    converted["schema"] = "v61c-identical-state-paired-evaluator-evidence"
    converted["status"] = "complete_alpha_zero_no_update_characterization"
    return converted


def validate_evidence_v61e(evidence: dict) -> list[dict]:
    return v61c.validate_evidence_v61c(_as_v61c_evidence(evidence))


def build_analysis_v61e(evidence: dict) -> dict:
    converted = _as_v61c_evidence(evidence)
    result = v61c.build_analysis_v61c(converted)
    result["schema"] = "v61e-fullbatch-fcfs-identical-state-null-analysis"
    result["status"] = "complete_matched_characterization_only"
    result["runtime_determinism_controls"] = copy.deepcopy(
        RUNTIME_CONTROLS_V61E
    )
    result["matched_v61c_panel_labels_metrics_bootstrap_thresholds"] = True
    result["v61c_effective_request_batch_size"] = 68
    result["v61c_thresholds_relaxed_or_changed"] = False
    result["source_evidence_content_sha256"] = evidence.get(
        "content_sha256_before_self_field"
    )
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = canonical_sha256_v61e(result)
    return result
