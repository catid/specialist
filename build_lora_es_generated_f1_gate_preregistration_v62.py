#!/usr/bin/env python3
"""Preregister the V62 generated-F1 evaluator methodology without launching."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import analyze_lora_es_generated_f1_gate_v62 as audit


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_generated_f1_robust_gate_v62.json"
).resolve()
AUDIT = audit.OUTPUT
AUDIT_FILE_SHA256 = (
    "5597aef07576ff35613dbb77c5a3c5f35a441374d926b7e54ac48df8e6386785"
)
AUDIT_CONTENT_SHA256 = (
    "365280457e84abb2734cb21b872a79d97e4f4da2f4c62b7bc1ef5361cab659fe"
)
EXACT_PANEL_PREVIEW = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_paired_block_bootstrap_v61_preview.json"
).resolve()
EXACT_PANEL_PREVIEW_FILE_SHA256 = (
    "a9ce060ce81df5b1fbddcc40db572fe56974ea6dfb6ef2e6ebf3e81925a400e2"
)
EXACT_PANEL_PREVIEW_CONTENT_SHA256 = (
    "1b25f3c667fc0e9eeddc19f1d20aebc70c2a0127db0c3eafe11c2f19fb35a0f0"
)
STABLE_EXACT_UNIT_SHA256 = (
    "2ef7c0e5ca2ff81b7326ea6dc2bd8b32c2499f939c04f9574c7135be37837ab4",
    "f080d7ea1b60062d035852d3542a2664e49af07294efec841f95acc99994f68f",
    "aa2af8f5c0eaede64c3acd990852475c441587c29bfa50fab0f30b2ed0061a66",
)
ACTOR_UNSTABLE_EXACT_UNIT_SHA256 = (
    "27c984ca0a7dbd4ffead6e79d7b691dc7a37856356a9a050a40603c11c6dbda7"
)


def _read_audit_v62() -> dict:
    if audit.file_sha256_v62(AUDIT) != AUDIT_FILE_SHA256:
        raise RuntimeError("v62 numeric audit file changed")
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != AUDIT_CONTENT_SHA256
        or audit.canonical_sha256_v62(compact) != AUDIT_CONTENT_SHA256
        or value.get("schema") != "v62-generated-f1-robust-gate-numeric-audit"
        or value.get("status")
        != "complete_numeric_only_method_design_hpo_unauthorized"
        or value.get("authorization", {}).get("hpo_launch_authorized") is not False
        or value.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v62 numeric audit content changed")
    return value


def _read_exact_panel_v62() -> dict:
    if audit.file_sha256_v62(EXACT_PANEL_PREVIEW) != (
        EXACT_PANEL_PREVIEW_FILE_SHA256
    ):
        raise RuntimeError("v62 exact panel preview file changed")
    value = json.loads(EXACT_PANEL_PREVIEW.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != EXACT_PANEL_PREVIEW_CONTENT_SHA256
        or audit.canonical_sha256_v62(compact)
        != EXACT_PANEL_PREVIEW_CONTENT_SHA256
        or value.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v62 exact panel preview content changed")
    exact = value.get("panels", {}).get("exact_sentinel", [])
    stable = tuple(
        item.get("unit_identity_sha256")
        for item in exact if item.get("stratum") == "stable_exact"
    )
    unstable = tuple(
        item.get("unit_identity_sha256")
        for item in exact if item.get("stratum") == "actor_unstable"
    )
    if (
        stable != STABLE_EXACT_UNIT_SHA256
        or unstable != (ACTOR_UNSTABLE_EXACT_UNIT_SHA256,)
        or any(
            item.get("base_exact_actor_count") != 4
            for item in exact if item.get("stratum") == "stable_exact"
        )
        or any(
            item.get("base_exact_actor_count") != 2
            for item in exact if item.get("stratum") == "actor_unstable"
        )
    ):
        raise RuntimeError("v62 exact panel roles changed")
    return {
        "stable_unit_identity_sha256": list(stable),
        "actor_unstable_unit_identity_sha256": unstable[0],
    }


def implementation_bindings_v62() -> dict:
    paths = {
        "numeric_audit_builder": Path(audit.__file__).resolve(),
        "preregistration_builder": Path(__file__).resolve(),
        "tests": ROOT / "test_lora_es_generated_f1_gate_v62.py",
        "analysis_v61c": ROOT / "lora_es_paired_null_calibration_v61c.py",
        "analysis_v61d": ROOT / "lora_es_singleton_fcfs_calibration_v61d.py",
        "analysis_v61e": ROOT / "lora_es_fullbatch_fcfs_calibration_v61e.py",
    }
    return {
        "code_file_sha256": {
            key: audit.file_sha256_v62(path) for key, path in paths.items()
        },
        "numeric_audit": {
            "file_sha256": AUDIT_FILE_SHA256,
            "content_sha256": AUDIT_CONTENT_SHA256,
        },
        "exact_panel_preview": {
            "file_sha256": EXACT_PANEL_PREVIEW_FILE_SHA256,
            "content_sha256": EXACT_PANEL_PREVIEW_CONTENT_SHA256,
        },
        "raw_train_qa_model_gpu_ood_or_protected_paths_opened": False,
    }


def build_preregistration_v62() -> dict:
    numeric = _read_audit_v62()
    exact_panel = _read_exact_panel_v62()
    noise = numeric["calibrated_noise_and_signal"]
    value = {
        "schema": "v62-generated-f1-robust-evaluator-hpo-gate-preregistration",
        "status": "methodology_preregistered_runtime_and_hpo_launch_unauthorized",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Freeze a train-only generated-F1 population estimator that preserves "
            "all eight actor/pair replicas, penalizes bootstrap uncertainty and "
            "actor influence, and moves exact-match to an aggregate full-census "
            "precommit gate."
        ),
        "numeric_design_evidence": {
            "audit_path": str(AUDIT),
            "audit_file_sha256": AUDIT_FILE_SHA256,
            "audit_content_sha256": AUDIT_CONTENT_SHA256,
            "source_hashes": numeric["source_hashes"],
            "only_finalized_numeric_v61c_v61d_v61e_evidence_opened": True,
            "raw_question_answer_prediction_or_generation_text_opened": False,
            "v53_signal_benchmark_reopened": False,
            "exact_panel_preview": {
                "path": str(EXACT_PANEL_PREVIEW),
                "file_sha256": EXACT_PANEL_PREVIEW_FILE_SHA256,
                "content_sha256": EXACT_PANEL_PREVIEW_CONTENT_SHA256,
                "raw_train_semantics_opened": False,
            },
        },
        "estimator_selection": {
            "primary_metric": "generated_f1_delta",
            "unit": "frozen_train_conflict_unit",
            "actors": 4,
            "counterbalanced_pairs_per_actor": 2,
            "replicas_per_unit": 8,
            "within_unit_aggregation": "arithmetic_mean_all_8_replicas",
            "across_unit_aggregation": "arithmetic_mean_conflict_units",
            "bootstrap": {
                "resampled_axis": "conflict_unit",
                "within_unit_replicas_preserved": 8,
                "replicates": audit.BOOTSTRAP_REPLICATES_V62,
                "seed": audit.BOOTSTRAP_SEED_V62,
                "one_sided_alpha": audit.ONE_SIDED_ALPHA_V62,
                "fitness_uses": "lower_confidence_bound",
            },
            "actor_influence": {
                "definition": (
                    "maximum absolute difference between full four-actor point "
                    "and each leave-one-actor-out point"
                ),
                "penalty_weight": 1.0,
            },
            "population_fitness_formula": (
                "generated_f1_primary_cluster_bootstrap_lcb_minus_1.0_times_"
                "maximum_absolute_leave_one_actor_out_shift"
            ),
            "larger_is_better": True,
            "candidate_advancement_requires_fitness_strictly_above_zero": True,
        },
        "alternative_estimators": {
            "median_8": {
                "role": "diagnostic_only",
                "rejected_as_primary": True,
                "arithmetic_nonzero_unit_erasure_fraction": numeric[
                    "pooled_alternative_quantification"
                ]["median_8_erasure_fraction"],
            },
            "trim_one_each_tail_mean_6": {
                "role": "diagnostic_sensitivity_only",
                "rejected_as_primary": True,
                "arithmetic_nonzero_unit_erasure_fraction": numeric[
                    "pooled_alternative_quantification"
                ]["trim_mean_6_erasure_fraction"],
            },
            "winsor_one_each_tail_mean_8": {
                "role": "diagnostic_sensitivity_only",
                "rejected_as_primary": True,
                "run_level_sign_mismatch_count": numeric[
                    "pooled_alternative_quantification"
                ]["winsorized_run_level_sign_mismatch_count"],
            },
            "three_of_four_actor_sign_consensus": {
                "role": "diagnostic_only",
                "rejected_as_primary": True,
                "fraction_of_any_nonzero_units_retained": numeric[
                    "pooled_alternative_quantification"
                ]["three_of_four_consensus_fraction_of_any_nonzero_units"],
            },
        },
        "fresh_pre_hpo_alpha_zero_calibration_gate": {
            "required_under_exact_future_hpo_runtime": True,
            "same_panel_labels_periods_metrics_and_eight_replica_estimator": True,
            "generated_f1_primary_interval_must_contain_zero": True,
            "maximum_primary_ci_halfwidth": noise[
                "future_pre_hpo_max_ci_halfwidth"
            ],
            "maximum_actor_leave_one_out_shift": noise[
                "future_pre_hpo_max_actor_leave_one_out_shift"
            ],
            "actor_threshold_calibration": noise[
                "future_actor_influence_threshold_calibration"
            ],
            "actor_threshold_rule": noise[
                "future_actor_influence_threshold_rule"
            ],
            "like_for_like_actor_statistic_required": True,
            "v53_actor_spread_is_descriptive_non_equivalent": True,
            "v53_actor_spread_used_as_leave_one_out_threshold": False,
            "signal_over_allowed_ci_halfwidth_minimum": 2.0,
            "v61e_historical_width_would_pass": noise[
                "v61e_passes_future_ci_width_gate"
            ],
            "v61e_historical_actor_influence_would_pass": noise[
                "v61e_passes_future_actor_influence_gate"
            ],
            "historical_outcomes_do_not_authorize_future_runtime": True,
            "gate_failure_aborts_before_population_or_update": True,
        },
        "population_reliability_gate": {
            "minimum_reliability": 0.8,
            "minimum_split_half_spearman": 0.7,
            "estimated_signal_standard_deviation_must_exceed_fresh_null_width": True,
            "frozen_v53_signal_standard_deviation": audit.V53_SIGNAL_BENCHMARK[
                "estimated_signal_standard_deviation"
            ],
            "fresh_actor_influence_is_subtracted_from_each_population_fitness": True,
        },
        "population_diagnostics_not_abort_gates": {
            "generated_exact": {
                "persist_aggregate_counts_and_individual_flip_count": True,
                "any_single_flip_aborts_population": False,
                "any_per_unit_eight_of_eight_equality_failure_aborts_population": False,
                "used_in_population_fitness": False,
            },
            "frozen_exact_sentinel": {
                "replicas_per_unit": 8,
                "baseline_stable": {
                    "unit_identity_sha256": exact_panel[
                        "stable_unit_identity_sha256"
                    ],
                    "unit_count": 3,
                    "per_unit_consensus_statistic": (
                        "strict_majority_exact_pass_count_at_least_5_of_8"
                    ),
                    "aggregate_statistic": (
                        "total_exact_pass_count_across_all_24_stable_"
                        "unit_actor_pair_replicas"
                    ),
                    "report_reference_candidate_totals_and_consensus_units": True,
                    "used_as_population_abort_gate": False,
                },
                "actor_unstable_stress": {
                    "unit_identity_sha256": exact_panel[
                        "actor_unstable_unit_identity_sha256"
                    ],
                    "unit_count": 1,
                    "role": "diagnostic_stress_unit_only",
                    "excluded_from_stable_consensus_aggregate": True,
                    "used_as_population_abort_gate": False,
                },
                "used_in_population_fitness": False,
            },
            "teacher_forced_logprob": {
                "persist_numeric_summary": True,
                "used_in_population_fitness": False,
                "used_as_population_abort_gate": False,
            },
            "median_trim_winsor_and_actor_sign_consensus": {
                "persist_numeric_sensitivity_summary": True,
                "used_in_population_fitness": False,
            },
        },
        "pre_materialization_master_commit_gate": {
            "runs_only_after_generated_f1_candidate_advancement": True,
            "eligible_train_full_census_required": True,
            "reference_and_candidate_use_identical_full_census_and_four_actors": True,
            "counterbalanced_pairs_per_actor": 2,
            "aggregate_exact_statistic": (
                "total_exact_passes_over_all_eligible_train_full_census_"
                "unit_actor_pair_replicas"
            ),
            "aggregate_exact_noninferiority_margin": 0,
            "aggregate_exact_candidate_total_must_be_at_least_reference_total": True,
            "individual_exact_flips_are_reported_but_do_not_individually_abort": True,
            "generated_f1_robust_fitness_must_remain_strictly_above_zero": True,
            "ephemeral_in_memory_candidate_state_required_for_scoring": True,
            "persistent_candidate_artifact_or_snapshot_before_gate": False,
            "failure_discards_ephemeral_candidate_before_any_persistent_"
            "artifact_or_master_commit": True,
            "master_commit_before_gate": False,
        },
        "non_reinterpretation": {
            "v61c_thresholds_changed": False,
            "v61d_thresholds_changed": False,
            "v61e_thresholds_changed": False,
            "v61e_exact_sentinel_failure_retroactively_relaxed": False,
            "new_exact_policy_applies_only_to_separately_preregistered_future_runs": True,
        },
        "access_and_authorization": {
            "train_only_methodology": True,
            "gpu_launch_authorized": False,
            "hpo_population_launch_authorized": False,
            "adapter_update_authorized": False,
            "candidate_materialization_or_master_commit_authorized": False,
            "holdback_ood_shadow_terminal_or_protected_access_authorized": False,
            "separate_runtime_preregistration_required_before_any_launch": True,
            "separate_protected_evaluation_authorization_required_later": True,
        },
        "implementation_bindings": implementation_bindings_v62(),
        "raw_question_answer_prediction_or_generation_text_persisted": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = audit.canonical_sha256_v62(value)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_preregistration_v62()
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    audit._exclusive_write_v62(output, payload)
    print(json.dumps({
        "path": str(output),
        "file_sha256": audit.file_sha256_v62(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launch_authorized": False,
        "hpo_population_launch_authorized": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
