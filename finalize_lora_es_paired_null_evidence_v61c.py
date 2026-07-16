#!/usr/bin/env python3
"""Independently verify and summarize sealed numeric-only V61C evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

import lora_es_nested_population_v52 as design_v52
import lora_es_paired_null_calibration_v61c as analysis


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v61c_v434_identical_state_paired_evaluator_calibration"
).resolve()
EVIDENCE = (RUN_DIR / "paired_null_evidence_v61c.json").resolve()
ANALYSIS = (RUN_DIR / "paired_null_analysis_v61c.json").resolve()
REPORT = (RUN_DIR / "paired_null_report_v61c.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v61c.jsonl").resolve()
OUTPUT = (RUN_DIR / "paired_null_finalized_v61c.json").resolve()

EVIDENCE_FILE_SHA256 = (
    "5be0a46ef0051c760b89d535cc252eeb1c9a6b2c700c209799049191615fa3dc"
)
EVIDENCE_CONTENT_SHA256 = (
    "15b7d74ea9b003d03ad4ba7667936ac80fac121cbbc28e4ced2c1cd9f57c7fa8"
)
ANALYSIS_FILE_SHA256 = (
    "b7588ccb58ac9ae6a196ce2605cc0b637b962170832470edc5b3095f07fafaeb"
)
ANALYSIS_CONTENT_SHA256 = (
    "93732923303da1201949c4619690b72ec3e34482ddc1c00da740cec1d0254563"
)
REPORT_FILE_SHA256 = (
    "f50b665fde835a29f3ee928d808df7a83237fbac09050d1730b2974f9dca44a9"
)
REPORT_CONTENT_SHA256 = (
    "c2e2cfbf8d1cefbe15fbb122d5e5f6fb06b73b7b312b4e0ae6d36c4e8a16378b"
)
GPU_LOG_FILE_SHA256 = (
    "16f2d695968af64ca531997ebc3ca3f2af46a3adbc3f5e1b3d02a68967eb6e83"
)
PREREGISTRATION_FILE_SHA256 = (
    "71837f96abf7a578bd2da542b30c2048065a901c69bf90086f2ecd0d730bba8b"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "cdb7d82b3073b272d97e7f619e013796ebd2f768ba5b46dcdf6b9299fba5dac0"
)
PANEL_FILE_SHA256 = (
    "92e0c6160bfc7884a00be4c34c427685dcb2bf5a6aa8c3820f5c53e225f8091c"
)
PANEL_CONTENT_SHA256 = (
    "ca0a947e6437c0d84360176087b0a9dab12b79cf6ba1be8f965b24e9f4ec7ba4"
)
TESTS = (ROOT / "test_finalize_lora_es_paired_null_evidence_v61c.py").resolve()


def file_sha256_v61c(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_self_hashed_v61c(path: Path, file_sha: str, content_sha: str) -> dict:
    if file_sha256_v61c(path) != file_sha:
        raise RuntimeError(f"v61c finalizer input file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or analysis.canonical_sha256_v61c(compact) != content_sha
    ):
        raise RuntimeError(f"v61c finalizer input content changed: {path}")
    return value


def _metric_summary_v61c(values, bands=()) -> dict:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("v61c finalizer metric values changed")
    return {
        "comparisons": int(array.size),
        "mean_delta": float(np.mean(array)),
        "mean_absolute_delta": float(np.mean(np.abs(array))),
        "maximum_absolute_delta": float(np.max(np.abs(array))),
        "positive_delta_count": int(np.sum(array > 0.0)),
        "negative_delta_count": int(np.sum(array < 0.0)),
        "zero_delta_count": int(np.sum(array == 0.0)),
        "absolute_delta_gt_counts": {
            str(band): int(np.sum(np.abs(array) > band)) for band in bands
        },
    }


def _gpu_summary_v61c(report: dict) -> dict:
    if file_sha256_v61c(GPU_LOG) != GPU_LOG_FILE_SHA256:
        raise RuntimeError("v61c GPU log changed")
    rows = [
        json.loads(line) for line in GPU_LOG.read_text(encoding="utf-8").splitlines()
        if line
    ]
    actor_pids = {
        int(item["physical_gpu_id"]): int(item["pid"])
        for item in report.get("actor_identities", [])
    }
    if set(actor_pids) != {0, 1, 2, 3} or len(set(actor_pids.values())) != 4:
        raise RuntimeError("v61c actor/GPU identity coverage changed")
    by_gpu = {}
    for gpu in range(4):
        selected = [item for item in rows if item.get("gpu") == gpu]
        expected = actor_pids[gpu]
        if (
            not selected
            or any(item.get("expected_pid") != expected for item in selected)
            or any(item.get("foreign_compute_pids") != [] for item in selected)
            or any(
                any(pid != expected for pid in item.get("compute_pids", []))
                for item in selected
            )
        ):
            raise RuntimeError("v61c foreign or mismatched GPU process observed")
        resident = [item for item in selected if expected in item["compute_pids"]]
        if not resident or not any(item["utilization_percent"] > 0 for item in resident):
            raise RuntimeError("v61c GPU lacked attributed positive activity")
        by_gpu[str(gpu)] = {
            "expected_pid": expected,
            "samples": len(selected),
            "resident_samples": len(resident),
            "positive_samples": sum(
                item["utilization_percent"] > 0 for item in resident
            ),
            "mean_resident_utilization_percent": math.fsum(
                item["utilization_percent"] for item in resident
            ) / len(resident),
            "peak_utilization_percent": max(
                item["utilization_percent"] for item in resident
            ),
            "peak_memory_used_mib": max(item["memory_used_mib"] for item in resident),
        }
    rebuilt = {"all_four_attributed_positive": True, "by_gpu": by_gpu}
    if report.get("gpu_activity") != rebuilt:
        raise RuntimeError("v61c reported GPU summary differs from numeric log")
    return {
        **rebuilt,
        "foreign_compute_process_observations": 0,
        "gpu_log_rows": len(rows),
    }


def _unit_summary_v61c(rows, generation_delta, teacher_delta) -> dict:
    f1_unit = np.mean(generation_delta[..., 0], axis=(1, 2))
    teacher_unit = np.mean(teacher_delta, axis=(1, 2))
    items = []
    for index, row in enumerate(rows):
        items.append({
            "role_index": index,
            "unit_identity_sha256": row["unit_identity_sha256"],
            "mean_paired_f1_delta": float(f1_unit[index]),
            "maximum_absolute_replica_f1_delta": float(
                np.max(np.abs(generation_delta[index, ..., 0]))
            ),
            "mean_paired_teacher_logprob_delta": float(teacher_unit[index]),
            "maximum_absolute_replica_teacher_logprob_delta": float(
                np.max(np.abs(teacher_delta[index]))
            ),
        })
    return {
        "units": len(items),
        "units_with_nonzero_mean_paired_f1_delta": int(np.sum(f1_unit != 0.0)),
        "units_with_any_nonzero_replica_f1_delta": int(np.sum(
            np.any(generation_delta[..., 0] != 0.0, axis=(1, 2))
        )),
        "units_with_nonzero_mean_paired_teacher_logprob_delta": int(np.sum(
            teacher_unit != 0.0
        )),
        "units_with_any_nonzero_replica_teacher_logprob_delta": int(np.sum(
            np.any(teacher_delta != 0.0, axis=(1, 2))
        )),
        "five_largest_absolute_mean_f1_units": sorted(
            items,
            key=lambda item: (
                -abs(item["mean_paired_f1_delta"]),
                item["unit_identity_sha256"],
            ),
        )[:5],
        "five_largest_absolute_mean_teacher_logprob_units": sorted(
            items,
            key=lambda item: (
                -abs(item["mean_paired_teacher_logprob_delta"]),
                item["unit_identity_sha256"],
            ),
        )[:5],
    }


def build_finalized_v61c() -> dict:
    evidence = _read_self_hashed_v61c(
        EVIDENCE, EVIDENCE_FILE_SHA256, EVIDENCE_CONTENT_SHA256,
    )
    stored_analysis = _read_self_hashed_v61c(
        ANALYSIS, ANALYSIS_FILE_SHA256, ANALYSIS_CONTENT_SHA256,
    )
    report = _read_self_hashed_v61c(
        REPORT, REPORT_FILE_SHA256, REPORT_CONTENT_SHA256,
    )
    rebuilt_analysis = analysis.build_analysis_v61c(evidence)
    if rebuilt_analysis != stored_analysis:
        raise RuntimeError("v61c stored analysis differs from independent rebuild")
    state_receipts = evidence.get("state_receipts", [])
    if (
        len(state_receipts) != 4
        or [item.get("period_index") for item in state_receipts] != list(range(4))
        or any(item.get("identical_v434_state") is not True
               or item.get("before") != item.get("after")
               or item.get("before", {}).get("canonical_fp32_master_sha256")
               != design_v52.MASTER_SHA256_V52
               or item.get("before", {}).get("bf16_runtime_values_sha256")
               != design_v52.MASTER_RUNTIME_SHA256_V52
               for item in state_receipts)
        or evidence.get("numeric_state_receipts_sha256")
        != analysis.canonical_sha256_v61c(state_receipts)
        or report.get("state_receipts_sha256")
        != evidence.get("numeric_state_receipts_sha256")
    ):
        raise RuntimeError("v61c identical V434 period state receipts changed")
    if (
        report.get("schema") != "v61c-identical-state-paired-evaluator-report"
        or report.get("status")
        != "complete_content_free_alpha_zero_characterization_sealed"
        or report.get("preregistration_file_sha256")
        != PREREGISTRATION_FILE_SHA256
        or report.get("preregistration_content_sha256")
        != PREREGISTRATION_CONTENT_SHA256
        or report.get("panel_file_sha256") != PANEL_FILE_SHA256
        or report.get("panel_content_sha256") != PANEL_CONTENT_SHA256
        or report.get("evidence", {}).get("file_sha256")
        != EVIDENCE_FILE_SHA256
        or report.get("evidence", {}).get("content_sha256")
        != EVIDENCE_CONTENT_SHA256
        or report.get("analysis", {}).get("file_sha256")
        != ANALYSIS_FILE_SHA256
        or report.get("analysis", {}).get("content_sha256")
        != ANALYSIS_CONTENT_SHA256
        or report.get("gpu_log_file_sha256") != GPU_LOG_FILE_SHA256
        or report.get("master_state_receipt", {}).get(
            "canonical_fp32_master_sha256"
        ) != design_v52.MASTER_SHA256_V52
        or report.get("master_state_receipt", {}).get(
            "bf16_runtime_values_sha256"
        ) != design_v52.MASTER_RUNTIME_SHA256_V52
        or report.get("cleanup", {}).get("engine_kill_count") != 4
        or report.get("cleanup", {}).get("placement_group_remove_count") != 4
        or report.get("cleanup", {}).get("all_four_gcs_states_removed") is not True
        or report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or report.get("alpha") != 0.0
        or report.get(
            "adapter_update_or_candidate_materialization_performed"
        ) is not False
        or report.get("full_v52_train_membership_or_holdback_opened") is not False
        or report.get("raw_question_answer_or_generation_text_persisted") is not False
        or report.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v61c sealed report contract changed")
    gpu = _gpu_summary_v61c(report)
    rows = analysis.validate_evidence_v61c(evidence)
    generation, teacher = analysis._metric_arrays_v61c(rows)
    ranking_generation = generation[:64]
    ranking_teacher = teacher[:64]
    sentinel_generation = generation[64:]
    sentinel_teacher = teacher[64:]
    gen_delta, teacher_delta, receipts = analysis._paired_deltas_v61c(
        ranking_generation, ranking_teacher
    )
    sentinel_gen_delta, sentinel_teacher_delta, sentinel_receipts = (
        analysis._paired_deltas_v61c(sentinel_generation, sentinel_teacher)
    )

    actors = []
    for actor in range(4):
        actors.append({
            "actor_rank": actor,
            "generated_f1": _metric_summary_v61c(
                gen_delta[:, actor, :, 0], analysis.F1_INSTABILITY_BANDS_V61C
            ),
            "generated_exact_all_zero": bool(
                np.all(gen_delta[:, actor, :, 1] == 0.0)
            ),
            "generated_nonzero_all_zero": bool(
                np.all(gen_delta[:, actor, :, 2] == 0.0)
            ),
            "teacher_logprob": _metric_summary_v61c(
                teacher_delta[:, actor, :],
                analysis.LOGPROB_INSTABILITY_BANDS_V61C,
            ),
            "pairs": [{
                "pair_index": pair,
                "reference_period": receipts[actor * 2 + pair][
                    "reference_period"
                ],
                "candidate_period": receipts[actor * 2 + pair][
                    "candidate_period"
                ],
                "candidate_after_reference": receipts[actor * 2 + pair][
                    "candidate_after_reference"
                ],
                "generated_f1": _metric_summary_v61c(
                    gen_delta[:, actor, pair, 0],
                    analysis.F1_INSTABILITY_BANDS_V61C,
                ),
                "teacher_logprob": _metric_summary_v61c(
                    teacher_delta[:, actor, pair],
                    analysis.LOGPROB_INSTABILITY_BANDS_V61C,
                ),
            } for pair in range(2)],
        })

    pairs = [{
        "pair_index": pair,
        "periods": list(analysis.PAIR_PERIODS_V61C[pair]),
        "generated_f1": _metric_summary_v61c(
            gen_delta[:, :, pair, 0], analysis.F1_INSTABILITY_BANDS_V61C,
        ),
        "teacher_logprob": _metric_summary_v61c(
            teacher_delta[:, :, pair], analysis.LOGPROB_INSTABILITY_BANDS_V61C,
        ),
    } for pair in range(2)]

    temporal_order = []
    for candidate_after in (False, True):
        f1_values = []
        teacher_values = []
        for actor in range(4):
            for pair in range(2):
                if receipts[actor * 2 + pair][
                    "candidate_after_reference"
                ] == candidate_after:
                    f1_values.append(gen_delta[:, actor, pair, 0])
                    teacher_values.append(teacher_delta[:, actor, pair])
        temporal_order.append({
            "candidate_after_reference": candidate_after,
            "generated_f1": _metric_summary_v61c(
                np.stack(f1_values, axis=1), analysis.F1_INSTABILITY_BANDS_V61C,
            ),
            "teacher_logprob": _metric_summary_v61c(
                np.stack(teacher_values, axis=1),
                analysis.LOGPROB_INSTABILITY_BANDS_V61C,
            ),
        })

    periods = [{
        "period_index": period,
        "request_type_order": analysis.REQUEST_TYPE_ORDER_V61C[str(period)],
        "generated_f1_mean": float(np.mean(ranking_generation[:, :, period, 0])),
        "generated_exact_count": int(np.sum(ranking_generation[:, :, period, 1])),
        "generated_nonzero_count": int(np.sum(ranking_generation[:, :, period, 2])),
        "teacher_logprob_mean": float(np.mean(ranking_teacher[:, :, period])),
        "actors": [{
            "actor_rank": actor,
            "label": analysis.LABEL_PLAN_V61C[str(actor)][period],
            "generated_f1_mean": float(np.mean(
                ranking_generation[:, actor, period, 0]
            )),
            "generated_exact_count": int(np.sum(
                ranking_generation[:, actor, period, 1]
            )),
            "generated_nonzero_count": int(np.sum(
                ranking_generation[:, actor, period, 2]
            )),
            "teacher_logprob_mean": float(np.mean(
                ranking_teacher[:, actor, period]
            )),
        } for actor in range(4)],
    } for period in range(4)]

    flips = []
    for unit_index, row in enumerate(rows[64:]):
        for actor in range(4):
            for pair in range(2):
                exact_delta = float(sentinel_gen_delta[unit_index, actor, pair, 1])
                if exact_delta == 0.0:
                    continue
                receipt = sentinel_receipts[actor * 2 + pair]
                flips.append({
                    "sentinel_role_index": unit_index,
                    "unit_identity_sha256": row["unit_identity_sha256"],
                    "actor_rank": actor,
                    "pair_index": pair,
                    "reference_period": receipt["reference_period"],
                    "candidate_period": receipt["candidate_period"],
                    "candidate_after_reference": receipt[
                        "candidate_after_reference"
                    ],
                    "exact_delta": exact_delta,
                    "f1_delta": float(
                        sentinel_gen_delta[unit_index, actor, pair, 0]
                    ),
                    "nonzero_delta": float(
                        sentinel_gen_delta[unit_index, actor, pair, 2]
                    ),
                    "teacher_logprob_delta": float(
                        sentinel_teacher_delta[unit_index, actor, pair]
                    ),
                })
    flips_by_unit = Counter(item["unit_identity_sha256"] for item in flips)
    if sorted(flips_by_unit.values()) != [1, 4]:
        raise RuntimeError("v61c sentinel flip concentration changed")

    primary = rebuilt_analysis["ranking_bootstrap"][
        "primary_conflict_unit_cluster_bootstrap"
    ]["intervals"]
    noise = rebuilt_analysis["noise_scale_comparison"]
    value = {
        "schema": "v61c-paired-null-independent-finalizer",
        "status": "complete_numeric_only_evidence_verified_hpo_unauthorized",
        "source_hashes": {
            "evidence": {
                "file_sha256": EVIDENCE_FILE_SHA256,
                "content_sha256": EVIDENCE_CONTENT_SHA256,
            },
            "analysis": {
                "file_sha256": ANALYSIS_FILE_SHA256,
                "content_sha256": ANALYSIS_CONTENT_SHA256,
            },
            "report": {
                "file_sha256": REPORT_FILE_SHA256,
                "content_sha256": REPORT_CONTENT_SHA256,
            },
            "gpu_log_file_sha256": GPU_LOG_FILE_SHA256,
            "preregistration": {
                "file_sha256": PREREGISTRATION_FILE_SHA256,
                "content_sha256": PREREGISTRATION_CONTENT_SHA256,
            },
            "panel": {
                "file_sha256": PANEL_FILE_SHA256,
                "content_sha256": PANEL_CONTENT_SHA256,
            },
        },
        "verification": {
            "all_source_file_and_self_hashes_verified": True,
            "stored_analysis_exactly_equals_independent_numeric_rebuild": True,
            "same_exact_v434_state_before_and_after_all_periods": True,
            "gpu_activity_recomputed_from_numeric_log": gpu,
            "four_engine_and_placement_group_cleanup_verified": True,
            "final_all_four_gpu_compute_process_lists_empty": True,
            "alpha": 0.0,
            "adapter_update_or_candidate_materialization_performed": False,
            "full_train_membership_holdback_or_protected_opened": False,
            "raw_question_answer_prediction_or_generation_text_opened_or_persisted": False,
        },
        "ranking_primary_null": {
            "point": rebuilt_analysis["ranking_bootstrap"]["point"],
            "raw_primary_cluster_intervals": primary,
            "raw_ci_halfwidths": {
                "generated_f1": primary["generated_f1_delta"]["halfwidth"],
                "teacher_logprob": primary[
                    "teacher_forced_logprob_delta"
                ]["halfwidth"],
            },
            "frozen_practical_effect_scales": {
                "generated_f1": analysis.F1_PRACTICAL_EFFECT_SCALE_V61C,
                "teacher_logprob": analysis.LOGPROB_PRACTICAL_EFFECT_SCALE_V61C,
            },
            "normalized_ci_halfwidths": {
                "generated_f1": noise[
                    "generated_f1_normalized_null_ci_halfwidth"
                ],
                "teacher_logprob": noise[
                    "teacher_logprob_normalized_null_ci_halfwidth"
                ],
            },
            "teacher_forced_logprob_primary_eligible": False,
            "teacher_eligibility_checks": noise["eligibility_checks"],
            "raw_width_and_normalized_scale_are_not_interchangeable": True,
        },
        "ranking_metric_breakdown": {
            "all_actors_pairs": {
                "generated_f1": _metric_summary_v61c(
                    gen_delta[..., 0], analysis.F1_INSTABILITY_BANDS_V61C,
                ),
                "generated_exact": _metric_summary_v61c(gen_delta[..., 1]),
                "generated_nonzero": _metric_summary_v61c(gen_delta[..., 2]),
                "teacher_logprob": _metric_summary_v61c(
                    teacher_delta, analysis.LOGPROB_INSTABILITY_BANDS_V61C,
                ),
                "all_512_generated_exact_deltas_zero": bool(
                    np.all(gen_delta[..., 1] == 0.0)
                ),
                "all_512_generated_nonzero_deltas_zero": bool(
                    np.all(gen_delta[..., 2] == 0.0)
                ),
            },
            "by_actor": actors,
            "by_pair": pairs,
            "by_candidate_temporal_order": temporal_order,
            "raw_period_actor_means": periods,
            "within_actor_same_label_repeat": rebuilt_analysis[
                "within_actor_same_label_repeat"
            ],
            "unit_concentration": _unit_summary_v61c(
                rows[:64], gen_delta, teacher_delta
            ),
        },
        "exact_sentinel": {
            "passed": False,
            "individual_exact_flip_count": len(flips),
            "units_with_any_exact_flip": len(flips_by_unit),
            "total_sentinel_units": 4,
            "flip_count_histogram_over_affected_units": dict(sorted(Counter(
                flips_by_unit.values()
            ).items())),
            "flip_count_by_unit": dict(sorted(flips_by_unit.items())),
            "maximum_absolute_individual_exact_delta": float(np.max(
                np.abs(sentinel_gen_delta[..., 1])
            )),
            "all_individual_exact_deltas_zero": False,
            "flips": flips,
        },
        "conclusion": {
            "teacher_logprob_rejected_as_v61_primary_fitness": True,
            "exact_sentinel_gate_failed": True,
            "v61_hpo_update_selection_or_promotion_authorized": False,
            "thresholds_changed_after_outcomes": False,
            "causal_source_of_nondeterminism_claimed": False,
            "holdback_ood_shadow_or_terminal_access_authorized": False,
        },
        "matched_followup_design": {
            "status": "proposal_only_requires_separate_preregistration",
            "identifier": "v61d_batch_invariant_identical_panel_null_calibration",
            "same_rows_labels_periods_metrics_bootstrap_and_frozen_thresholds": True,
            "only_scientific_change": (
                "enable installed-vLLM batch-invariant execution and disable "
                "async scheduling"
            ),
            "required_runtime_controls": {
                "VLLM_BATCH_INVARIANT": "1",
                "async_scheduling": False,
                "actor_identity_must_verify_both": True,
                "fail_closed_if_qwen_gdn_or_mamba_backend_unsupported": True,
                "base_moe_and_lora_tuned_configs_bypassed": True,
                "batch_invariant_moe_default": {
                    "BLOCK_SIZE_M": 64,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "SPLIT_K": 1,
                },
                "lora_split_k": 1,
                "v27c_tuned_table_claimed_or_used": False,
                "expected_speed_tradeoff_not_a_correctness_error": True,
            },
            "alpha_zero_no_update_characterization_only": True,
            "hpo_or_protected_access_authorized": False,
            "not_a_threshold_relaxation": True,
        },
        "implementation_bindings": {
            "finalizer_file_sha256": file_sha256_v61c(Path(__file__).resolve()),
            "tests_file_sha256": file_sha256_v61c(TESTS),
        },
        "raw_question_answer_prediction_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = analysis.canonical_sha256_v61c(
        value
    )
    return value


def _exclusive_write_v61c(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_finalized_v61c()
    _exclusive_write_v61c(
        output, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v61c(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "teacher_logprob_primary_eligible": False,
        "exact_sentinel_passed": False,
        "v61_hpo_authorized": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
