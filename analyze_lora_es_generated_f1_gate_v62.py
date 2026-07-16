#!/usr/bin/env python3
"""Numeric-only C/D/E estimator alternatives and actor-influence audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

import lora_es_fullbatch_fcfs_calibration_v61e as v61e
import lora_es_paired_null_calibration_v61c as v61c
import lora_es_singleton_fcfs_calibration_v61d as v61d


ROOT = Path(__file__).resolve().parent
RUNS = (ROOT / "experiments/eggroll_es_hpo/runs").resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "generated_f1_robust_gate_numeric_audit_v62.json"
).resolve()
SOURCES = {
    "v61c": {
        "evidence": (
            RUNS / "v61c_v434_identical_state_paired_evaluator_calibration/"
            "paired_null_evidence_v61c.json"
        ).resolve(),
        "finalizer": (
            RUNS / "v61c_v434_identical_state_paired_evaluator_calibration/"
            "paired_null_finalized_v61c.json"
        ).resolve(),
        "evidence_file_sha256": (
            "5be0a46ef0051c760b89d535cc252eeb1c9a6b2c700c209799049191615fa3dc"
        ),
        "evidence_content_sha256": (
            "15b7d74ea9b003d03ad4ba7667936ac80fac121cbbc28e4ced2c1cd9f57c7fa8"
        ),
        "finalizer_file_sha256": (
            "d3d5eabf1e5d9b0bed2dfd2a355ed5eb839a22cb4bcdea58af0ab84231042d46"
        ),
        "finalizer_content_sha256": (
            "7bc9735dea87ae8bf2374bcefb7c290b7bb273f2394b44d54dc1fa69e8e851c0"
        ),
    },
    "v61d": {
        "evidence": (
            RUNS / "v61d_v434_singleton_fcfs_paired_evaluator_calibration/"
            "singleton_fcfs_null_evidence_v61d.json"
        ).resolve(),
        "finalizer": (
            RUNS / "v61d_v434_singleton_fcfs_paired_evaluator_calibration/"
            "singleton_fcfs_finalized_v61d.json"
        ).resolve(),
        "evidence_file_sha256": (
            "49be43e8a2e02093952bec7a0186f900fd64e3ec00057ece31e290a540c7044e"
        ),
        "evidence_content_sha256": (
            "f07a24fcd5ae0cedf1703f1bf25a7e9b6ca3db900d4bd58cc7351a68ec795048"
        ),
        "finalizer_file_sha256": (
            "98da3f65e5d6a3801d1b56a143b7d9a44d95b971d290523372013525fad814fd"
        ),
        "finalizer_content_sha256": (
            "58f15e71f7bdf2b7e3804479627bd14e782303004728079d18b2e8fbe09c657a"
        ),
    },
    "v61e": {
        "evidence": (
            RUNS / "v61e_v434_fullbatch_fcfs_paired_evaluator_calibration/"
            "fullbatch_fcfs_null_evidence_v61e.json"
        ).resolve(),
        "finalizer": (
            RUNS / "v61e_v434_fullbatch_fcfs_paired_evaluator_calibration/"
            "fullbatch_fcfs_finalized_v61e.json"
        ).resolve(),
        "evidence_file_sha256": (
            "8cd25a3f2f94175dba174199fca665209cbc7d98959ce26df7856cdd4e79507f"
        ),
        "evidence_content_sha256": (
            "db145f887a90cd9383bef0d7caaeb22839e34af946c06c547ddc7c79fb564a66"
        ),
        "finalizer_file_sha256": (
            "fcd0a8777839b06ac55323d05a7107bcb6c3b6f0f8c5e34e4ae3036c55ac9e3d"
        ),
        "finalizer_content_sha256": (
            "675d70cdb95ae1a4c0d326988df0cd4555633d4c9195712ffb49c221513802cf"
        ),
    },
}
VALIDATORS = {
    "v61c": v61c.validate_evidence_v61c,
    "v61d": v61d.validate_evidence_v61d,
    "v61e": v61e.validate_evidence_v61e,
}
FINALIZER_SCHEMAS = {
    "v61c": "v61c-paired-null-independent-finalizer",
    "v61d": "v61d-singleton-fcfs-independent-finalizer",
    "v61e": "v61e-fullbatch-fcfs-independent-finalizer",
}
BOOTSTRAP_REPLICATES_V62 = 4096
BOOTSTRAP_SEED_V62 = 2026071612
ONE_SIDED_ALPHA_V62 = 0.05
V53_SIGNAL_BENCHMARK = {
    "sigma": 0.0048,
    "estimated_signal_standard_deviation": 0.001547645180585056,
    "reliability": 0.9937167282924395,
    "split_half_spearman": 0.9941176470588236,
    "fresh_calibration_observed_maximum_actor_spread": 0.0008587933528567682,
    "actor_spread_definition_proven_equivalent_to_v62_leave_one_out_shift": False,
    "actor_spread_used_as_v62_leave_one_out_threshold": False,
    "source_file_sha256": (
        "02412f0440042ed970c4efe8a908fae59fef9c99d47a8f2771c6584da7edde47"
    ),
    "source_content_sha256": (
        "0112032dd2f2dabb777a9cd3f5dcb3f30975e59185ce5ece7c87d3258f499151"
    ),
    "source_reopened_by_v62_audit": False,
}


def file_sha256_v62(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256_v62(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")).hexdigest()


def _read_self_hashed_v62(path: Path, file_sha: str, content_sha: str) -> dict:
    if file_sha256_v62(path) != file_sha:
        raise RuntimeError(f"v62 numeric source file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or canonical_sha256_v62(compact) != content_sha
    ):
        raise RuntimeError(f"v62 numeric source content changed: {path}")
    return value


def _interval_v62(unit_values: np.ndarray) -> dict:
    values = np.asarray(unit_values, dtype=np.float64).reshape(-1)
    if values.shape != (64,) or not np.isfinite(values).all():
        raise ValueError("v62 unit estimator values changed")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED_V62))
    samples = np.empty(BOOTSTRAP_REPLICATES_V62, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES_V62):
        selected = rng.integers(0, 64, size=64)
        samples[index] = float(np.mean(values[selected]))
    ordered = np.sort(samples)
    lower = float(ordered[int(math.floor(
        ONE_SIDED_ALPHA_V62 * (len(ordered) - 1)
    ))])
    upper = float(ordered[int(math.floor(
        (1.0 - ONE_SIDED_ALPHA_V62) * (len(ordered) - 1)
    ))])
    return {
        "point": float(np.mean(values)),
        "lcb": lower,
        "ucb": upper,
        "halfwidth": (upper - lower) / 2.0,
        "contains_zero": lower <= 0.0 <= upper,
    }


def _run_audit_v62(name: str, evidence: dict, finalized: dict) -> dict:
    rows = VALIDATORS[name](evidence)
    generation, teacher = v61c._metric_arrays_v61c(rows)
    delta, _, _ = v61c._paired_deltas_v61c(generation[:64], teacher[:64])
    f1 = delta[..., 0]
    replicas = f1.reshape(64, 8)
    ordered = np.sort(replicas, axis=1)
    alternatives = {
        "arithmetic_mean_8": np.mean(replicas, axis=1),
        "median_8": np.median(replicas, axis=1),
        "trim_one_each_tail_mean_6": np.mean(ordered[:, 1:-1], axis=1),
        "winsor_one_each_tail_mean_8": np.mean(np.concatenate((
            ordered[:, 1:2], ordered[:, 1:-1], ordered[:, -2:-1],
        ), axis=1), axis=1),
        "median_of_four_actor_pair_means": np.median(
            np.mean(f1, axis=2), axis=1
        ),
    }
    arithmetic_nonzero = alternatives["arithmetic_mean_8"] != 0.0
    summaries = {}
    for key, values in alternatives.items():
        interval = _interval_v62(values)
        interval.update({
            "nonzero_unit_count": int(np.sum(values != 0.0)),
            "arithmetic_nonzero_units_erased_to_zero": int(np.sum(
                arithmetic_nonzero & (values == 0.0)
            )),
            "run_level_sign_matches_arithmetic": bool(
                np.sign(np.mean(values))
                == np.sign(np.mean(alternatives["arithmetic_mean_8"]))
            ),
        })
        summaries[key] = interval
    final_intervals = finalized[
        "ranking_point_and_primary_ci_halfwidth_comparison"
    ]["generated_f1_delta"] if name != "v61c" else None
    if name == "v61c":
        expected_point = finalized["ranking_primary_null"]["point"][
            "generated_f1_delta"
        ]
        expected_halfwidth = finalized["ranking_primary_null"][
            "raw_primary_cluster_intervals"
        ]["generated_f1_delta"]["halfwidth"]
    else:
        expected_point = final_intervals["point"][name]
        expected_halfwidth = final_intervals["primary_ci_halfwidth"][name]
    arithmetic = summaries["arithmetic_mean_8"]
    if (
        arithmetic["point"] != expected_point
        or arithmetic["halfwidth"] != expected_halfwidth
    ):
        raise RuntimeError("v62 arithmetic estimator differs from finalizer")

    actor_means = np.mean(f1, axis=(0, 2))
    full = float(np.mean(f1))
    leave_one_out = np.asarray([
        np.mean(np.delete(f1, actor, axis=1)) for actor in range(4)
    ], dtype=np.float64)
    actor_pair_means = np.mean(f1, axis=2)
    positive = np.sum(actor_pair_means > 0.0, axis=1)
    negative = np.sum(actor_pair_means < 0.0, axis=1)
    units_any = np.any(replicas != 0.0, axis=1)
    max_shift = float(np.max(np.abs(leave_one_out - full)))
    return {
        "source": name,
        "ranking_units": 64,
        "replicas_per_unit": 8,
        "individual_nonzero_replica_count": int(np.sum(replicas != 0.0)),
        "units_with_any_nonzero_replica": int(np.sum(units_any)),
        "estimator_alternatives": summaries,
        "actor_influence": {
            "full_arithmetic_point": full,
            "actor_mean_deltas": [float(value) for value in actor_means],
            "actor_mean_range": float(np.max(actor_means) - np.min(actor_means)),
            "leave_one_actor_out_points": [
                float(value) for value in leave_one_out
            ],
            "maximum_absolute_leave_one_actor_out_shift": max_shift,
            "leave_one_actor_out_sign_flip_count": int(np.sum(
                np.sign(leave_one_out) != np.sign(full)
            )),
            "influence_over_v53_signal_standard_deviation": (
                max_shift
                / V53_SIGNAL_BENCHMARK["estimated_signal_standard_deviation"]
            ),
        },
        "actor_sign_consensus": {
            "units_with_any_nonzero_replica": int(np.sum(units_any)),
            "units_with_four_of_four_same_nonzero_actor_sign": int(np.sum(
                (positive == 4) | (negative == 4)
            )),
            "units_with_at_least_three_of_four_same_nonzero_actor_sign": int(
                np.sum((positive >= 3) | (negative >= 3))
            ),
            "units_with_both_positive_and_negative_actor_means": int(np.sum(
                (positive > 0) & (negative > 0)
            )),
        },
        "proposed_population_fitness_at_null": {
            "formula": "arithmetic_mean_8_cluster_bootstrap_lcb_minus_actor_influence",
            "cluster_bootstrap_lcb": arithmetic["lcb"],
            "maximum_absolute_leave_one_actor_out_shift": max_shift,
            "value": arithmetic["lcb"] - max_shift,
        },
    }


def build_audit_v62() -> dict:
    runs = {}
    source_hashes = {}
    for name, source in SOURCES.items():
        evidence = _read_self_hashed_v62(
            source["evidence"],
            source["evidence_file_sha256"],
            source["evidence_content_sha256"],
        )
        finalized = _read_self_hashed_v62(
            source["finalizer"],
            source["finalizer_file_sha256"],
            source["finalizer_content_sha256"],
        )
        if (
            finalized.get("schema") != FINALIZER_SCHEMAS[name]
            or finalized.get("source_hashes", {}).get("evidence") != {
                "file_sha256": source["evidence_file_sha256"],
                "content_sha256": source["evidence_content_sha256"],
            }
            or finalized.get("protected_semantics_opened") is not False
        ):
            raise RuntimeError("v62 finalized source chain changed")
        runs[name] = _run_audit_v62(name, evidence, finalized)
        source_hashes[name] = {
            key: source[key] for key in (
                "evidence_file_sha256",
                "evidence_content_sha256",
                "finalizer_file_sha256",
                "finalizer_content_sha256",
            )
        }

    arithmetic_nonzero = sum(
        item["estimator_alternatives"]["arithmetic_mean_8"][
            "nonzero_unit_count"
        ] for item in runs.values()
    )
    units_any = sum(
        item["units_with_any_nonzero_replica"] for item in runs.values()
    )
    median_erased = sum(
        item["estimator_alternatives"]["median_8"][
            "arithmetic_nonzero_units_erased_to_zero"
        ] for item in runs.values()
    )
    trim_erased = sum(
        item["estimator_alternatives"]["trim_one_each_tail_mean_6"][
            "arithmetic_nonzero_units_erased_to_zero"
        ] for item in runs.values()
    )
    consensus_three = sum(
        item["actor_sign_consensus"][
            "units_with_at_least_three_of_four_same_nonzero_actor_sign"
        ] for item in runs.values()
    )
    maximum_halfwidth = max(
        item["estimator_alternatives"]["arithmetic_mean_8"]["halfwidth"]
        for item in runs.values()
    )
    maximum_actor_influence = max(
        item["actor_influence"][
            "maximum_absolute_leave_one_actor_out_shift"
        ] for item in runs.values()
    )
    signal = V53_SIGNAL_BENCHMARK["estimated_signal_standard_deviation"]
    value = {
        "schema": "v62-generated-f1-robust-gate-numeric-audit",
        "status": "complete_numeric_only_method_design_hpo_unauthorized",
        "source_hashes": source_hashes,
        "v53_frozen_signal_benchmark_without_reopening": V53_SIGNAL_BENCHMARK,
        "runs": runs,
        "pooled_alternative_quantification": {
            "runs": 3,
            "ranking_units_per_run": 64,
            "arithmetic_nonzero_unit_outcomes": arithmetic_nonzero,
            "units_with_any_nonzero_replica": units_any,
            "median_8_arithmetic_nonzero_units_erased": median_erased,
            "median_8_erasure_fraction": median_erased / arithmetic_nonzero,
            "trim_mean_6_arithmetic_nonzero_units_erased": trim_erased,
            "trim_mean_6_erasure_fraction": trim_erased / arithmetic_nonzero,
            "three_of_four_actor_sign_consensus_units": consensus_three,
            "three_of_four_consensus_fraction_of_any_nonzero_units": (
                consensus_three / units_any
            ),
            "four_of_four_actor_sign_consensus_units": sum(
                item["actor_sign_consensus"][
                    "units_with_four_of_four_same_nonzero_actor_sign"
                ] for item in runs.values()
            ),
            "winsorized_run_level_sign_mismatch_count": sum(
                not item["estimator_alternatives"][
                    "winsor_one_each_tail_mean_8"
                ]["run_level_sign_matches_arithmetic"]
                for item in runs.values()
            ),
        },
        "calibrated_noise_and_signal": {
            "all_three_arithmetic_primary_intervals_contain_zero": all(
                item["estimator_alternatives"]["arithmetic_mean_8"][
                    "contains_zero"
                ] for item in runs.values()
            ),
            "maximum_arithmetic_primary_ci_halfwidth": maximum_halfwidth,
            "maximum_cde_null_actor_leave_one_out_shift": maximum_actor_influence,
            "v53_signal_standard_deviation": signal,
            "signal_over_maximum_ci_halfwidth": signal / maximum_halfwidth,
            "maximum_actor_influence_over_signal": (
                maximum_actor_influence / signal
            ),
            "future_pre_hpo_max_ci_halfwidth": signal / 2.0,
            "future_pre_hpo_max_actor_leave_one_out_shift": (
                maximum_actor_influence
            ),
            "future_actor_influence_threshold_calibration": (
                "maximum_like_for_like_finalized_v61c_v61d_v61e_null_"
                "leave_one_actor_out_shift"
            ),
            "future_actor_influence_threshold_rule": (
                "conservative_observed_maximum_no_distributional_quantile_claim"
            ),
            "v53_maximum_actor_spread_descriptive_non_equivalent": (
                V53_SIGNAL_BENCHMARK[
                    "fresh_calibration_observed_maximum_actor_spread"
                ]
            ),
            "v53_actor_spread_definition_proven_equivalent_to_v62_loo": False,
            "v53_actor_spread_used_as_v62_loo_threshold": False,
            "v61e_passes_future_ci_width_gate": (
                runs["v61e"]["estimator_alternatives"][
                    "arithmetic_mean_8"
                ]["halfwidth"] <= signal / 2.0
            ),
            "v61e_passes_future_actor_influence_gate": (
                runs["v61e"]["actor_influence"][
                    "maximum_absolute_leave_one_actor_out_shift"
                ] <= maximum_actor_influence
            ),
        },
        "method_selection": {
            "primary_estimator": "arithmetic_mean_8_conflict_unit_cluster",
            "primary_bootstrap": "conflict_unit_resampling_preserving_8_replicas",
            "population_fitness": (
                "primary_generated_f1_cluster_bootstrap_lcb_minus_maximum_"
                "absolute_leave_one_actor_out_shift"
            ),
            "median_8_rejected_as_primary_due_sparse_signal_erasure": True,
            "three_of_four_consensus_rejected_as_primary_due_sparse_erasure": True,
            "trimmed_and_winsorized_estimators": "diagnostic_sensitivity_only",
            "exact_metric": "population_diagnostic_then_full_census_precommit_gate",
            "teacher_logprob": "diagnostic_only",
            "single_exact_flip_aborts_population": False,
            "thresholds_retroactively_changed_for_v61c_v61d_or_v61e": False,
        },
        "authorization": {
            "hpo_launch_authorized": False,
            "candidate_materialization_or_master_commit_authorized": False,
            "holdback_ood_shadow_terminal_or_protected_access_authorized": False,
        },
        "raw_question_answer_prediction_or_generation_text_opened_or_persisted": False,
        "protected_semantics_opened": False,
        "implementation_bindings": {
            "audit_file_sha256": file_sha256_v62(Path(__file__).resolve()),
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256_v62(value)
    return value


def _exclusive_write_v62(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_audit_v62()
    _exclusive_write_v62(
        output,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v62(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "hpo_launch_authorized": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
