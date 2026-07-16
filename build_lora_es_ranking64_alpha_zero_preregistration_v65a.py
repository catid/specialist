#!/usr/bin/env python3
"""Seal the exact-64 V65A alpha-zero prerequisite without semantic access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import build_lora_es_robust_sampling_preregistration_v65 as common65
import lora_es_nested_population_v52 as design52
import lora_es_ranking64_alpha_zero_calibration_v65a as design65a
import lora_es_robust_sampling_population_v65 as population65
import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime64


ROOT = Path(__file__).resolve().parent
RANKING_PANEL_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v65a_ranking64_alpha_zero_panel.json"
).resolve()
PREREGISTRATION_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "ranking64_alpha_zero_calibration_v65a.json"
).resolve()
EXPERIMENT = "v65a_ranking64_alpha_zero_calibration"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()

V64_RUNTIME_CRITICAL_ENTRY_PATHS_V65A = {
    f"v64_runtime__{name}": Path(path).resolve()
    for name, path in sorted(runtime64.WORKER_EXECUTION_PATHS_V64.items())
}
DEFAULT_IMPLEMENTATION_ENTRY_PATHS_V65A = {
    "runtime_v65a": (
        ROOT / "run_lora_es_ranking64_alpha_zero_calibration_v65a.py"
    ),
    "numeric_analysis_v65a": Path(design65a.__file__).resolve(),
    "preregistration_builder_v65a": Path(__file__).resolve(),
    "tests_v65a": (
        ROOT / "test_lora_es_ranking64_alpha_zero_calibration_v65a.py"
    ),
    "hash_only_panel_design_v65": Path(population65.__file__).resolve(),
    "numeric_cpu_scorer_v65": (
        ROOT / "eggroll_es_worker_robust_sampling_v65.py"
    ),
    "exact_master_slot_write_worker_v65a": (
        ROOT / "eggroll_es_worker_lora_v65a.py"
    ),
    "base_model_byte_receipt_runtime_v64": (
        ROOT / "run_lora_es_v59_vs_v434_robust_confirmation_v64.py"
    ),
    "calibrated_runtime_v62b": (
        ROOT / "run_lora_es_pre_hpo_alpha_zero_calibration_v62b.py"
    ),
    "population_runtime_v52": ROOT / "run_lora_es_nested_population_v52.py",
    "population_worker_v52": ROOT / "eggroll_es_worker_lora_v52.py",
    "canonical_state_worker_v41a": ROOT / "eggroll_es_worker_lora_v41a.py",
    **V64_RUNTIME_CRITICAL_ENTRY_PATHS_V65A,
}
REQUIRED_IMPLEMENTATION_BINDING_KEYS_V65A = frozenset(
    f"entry__{name}" for name in DEFAULT_IMPLEMENTATION_ENTRY_PATHS_V65A
)


def json_payload_v65a(value: dict) -> bytes:
    return (json.dumps(
        value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode("ascii")


def payload_sha256_v65a(value: dict) -> str:
    return hashlib.sha256(json_payload_v65a(value)).hexdigest()


def implementation_bindings_v65a(
    entry_paths: dict[str, Path] | None = None,
) -> dict:
    return common65.implementation_bindings_v65(
        entry_paths or DEFAULT_IMPLEMENTATION_ENTRY_PATHS_V65A
    )


def _implementation_entry_paths_v65a(
    entry_paths: dict[str, Path] | None,
    *,
    test_only_allow_nonproduction_entry_paths: bool,
) -> dict[str, Path]:
    if entry_paths is None:
        return DEFAULT_IMPLEMENTATION_ENTRY_PATHS_V65A
    if test_only_allow_nonproduction_entry_paths is not True:
        raise RuntimeError(
            "v65a nonproduction implementation roots require test-only opt-in"
        )
    return entry_paths


def sealed_source_bindings_v65a() -> tuple[dict, dict]:
    preview = population65.read_exact_self_hashed_v65(
        population65.PREVIEW_V61,
        population65.PREVIEW_V61_FILE_SHA256,
        population65.PREVIEW_V61_CONTENT_SHA256,
    )
    panel = population65.build_ranking_panel_v65(preview)
    panel61c = population65.read_exact_self_hashed_v65(
        population65.V61C_PANEL,
        population65.V61C_PANEL_FILE_SHA256,
        population65.V61C_PANEL_CONTENT_SHA256,
    )
    first64 = panel61c.get("items", [])[:64]
    if (
        len(panel61c.get("items", [])) != 68
        or panel61c.get("ranking_units") != 64
        or panel61c.get("exact_sentinel_units") != 4
        or any(item.get("role") != "ranking" for item in first64)
        or [item.get("row_sha256") for item in first64]
        != [item["row_sha256"] for item in panel["items"]]
        or [item.get("unit_identity_sha256") for item in first64]
        != [item["unit_identity_sha256"] for item in panel["items"]]
        or any(
            item.get("role") != "exact_sentinel"
            for item in panel61c["items"][64:]
        )
        or panel61c.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v65a exact V61C ranking prefix changed")

    v62b = population65.read_exact_self_hashed_v65(
        common65.V62B_FINALIZED,
        common65.V62B_FINALIZED_FILE_SHA256,
        common65.V62B_FINALIZED_CONTENT_SHA256,
    )
    eligibility = v62b.get("calibration_eligibility_observation", {})
    state = v62b.get("verification", {}).get("v434_state_receipts", {})
    if (
        v62b.get("schema")
        != "v62b-pre-hpo-alpha-zero-independent-finalizer"
        or v62b.get("status")
        != "complete_numeric_only_eligibility_observed_hpo_unauthorized"
        or eligibility.get("failed_gate_count") != 0
        or eligibility.get("passed_gate_count") != 3
        or eligibility.get("hpo_population_launch_or_update_authorized")
        is not False
        or v62b.get("protected_semantics_opened") is not False
        or state.get("canonical_fp32_master_sha256")
        != design52.MASTER_SHA256_V52
        or state.get("bf16_runtime_values_sha256")
        != design52.MASTER_RUNTIME_SHA256_V52
    ):
        raise RuntimeError("v65a V62B methodology source changed")

    model_seal = population65.read_exact_self_hashed_v65(
        common65.BASE_MODEL_SEAL,
        common65.BASE_MODEL_SEAL_FILE_SHA256,
        common65.BASE_MODEL_SEAL_CONTENT_SHA256,
    )
    base = model_seal.get("arms", {}).get("base_middle_late", {})
    if (
        base.get("path") != str(common65.BASE_MODEL)
        or base.get("shard_count") != 26
        or base.get("all_files_fingerprint_sha256")
        != common65.BASE_MODEL_ALL_FILES_FINGERPRINT_SHA256
    ):
        raise RuntimeError("v65a base-model seal changed")

    sources = {
        "v61_hash_only_ranking_preview": {
            "path": str(population65.PREVIEW_V61),
            "file_sha256": population65.PREVIEW_V61_FILE_SHA256,
            "content_sha256": population65.PREVIEW_V61_CONTENT_SHA256,
            "launch_authorized_by_source": False,
            "future_candidate_outcomes_used_for_selection": False,
        },
        "v61c_hash_numeric_panel": {
            "path": str(population65.V61C_PANEL),
            "file_sha256": population65.V61C_PANEL_FILE_SHA256,
            "content_sha256": population65.V61C_PANEL_CONTENT_SHA256,
            "first_64_items_are_exact_ranking_projection": True,
            "last_4_semantic_rows_authorized": False,
        },
        "v62b_batch68_methodology_and_threshold_source": {
            "path": str(common65.V62B_FINALIZED),
            "finalizer_commit": common65.V62B_FINALIZER_COMMIT,
            "file_sha256": common65.V62B_FINALIZED_FILE_SHA256,
            "content_sha256": common65.V62B_FINALIZED_CONTENT_SHA256,
            "v62_methodology_commit": design65a.v62b.V62_METHOD_COMMIT,
            "numeric_audit_identities": dict(
                design65a.v62b.V62_NUMERIC_AUDIT_IDENTITIES
            ),
            "preregistration_identities": dict(
                design65a.v62b.V62_PREREGISTRATION_IDENTITIES
            ),
            "batch_size": 68,
            "VLLM_BATCH_INVARIANT": False,
            "authorizes_exact_64_calibration_or_population": False,
        },
    }
    return panel, sources


def artifacts_v65a() -> dict:
    return {
        "run_directory": str(RUN_DIR),
        "attempt": str(RUN_DIR.parent / f".{EXPERIMENT}.attempt.json"),
        "gpu_log": str(RUN_DIR / "gpu_activity_v65a.jsonl"),
        "evidence": str(RUN_DIR / "ranking64_alpha_zero_evidence_v65a.json"),
        "analysis": str(RUN_DIR / "ranking64_alpha_zero_analysis_v65a.json"),
        "report": str(RUN_DIR / "ranking64_alpha_zero_report_v65a.json"),
        "failure": str(RUN_DIR / "failure_v65a.json"),
    }


def build_preregistration_v65a(
    panel: dict,
    sources: dict,
    implementation_bindings: dict,
    *,
    ranking_panel_output: Path = RANKING_PANEL_OUTPUT,
    implementation_entry_paths: dict[str, Path] | None = None,
    _test_only_allow_nonproduction_entry_paths: bool = False,
) -> dict:
    entry_paths = _implementation_entry_paths_v65a(
        implementation_entry_paths,
        test_only_allow_nonproduction_entry_paths=(
            _test_only_allow_nonproduction_entry_paths
        ),
    )
    expected_implementation = implementation_bindings_v65a(
        entry_paths
    )
    if (
        panel.get("schema") != "v65-robust-sampling-ranking-panel"
        or panel.get("ranking_units") != 64
        or panel.get("question_answer_or_generation_text_persisted") is not False
        or panel.get("protected_semantics_opened") is not False
        or set(sources) != {
            "v61_hash_only_ranking_preview", "v61c_hash_numeric_panel",
            "v62b_batch68_methodology_and_threshold_source",
        }
        or not implementation_bindings
        or implementation_bindings != expected_implementation
        or any(
            not isinstance(binding, dict)
            or set(binding) != {"path", "file_sha256"}
            for binding in implementation_bindings.values()
        )
    ):
        raise RuntimeError("v65a preregistration inputs changed")
    panel_path = Path(ranking_panel_output).resolve()
    bootstrap_indices = design65a.frozen_bootstrap_indices_v65a()
    value = {
        "schema": "v65a-ranking64-alpha-zero-calibration-preregistration",
        "status": (
            "sealed_before_v65a_train_semantics_model_ray_or_gpu_access"
        ),
        "specific_v65a_exact64_alpha_zero_gpu_launch_authorized": True,
        "prior_batch68_calibration_alone_authorizes_exact64_launch": False,
        "purpose": (
            "Calibrate the exact 64-request V65 population evaluator at alpha "
            "zero using four discarded all-actor warmups and one fixed "
            "four-period counterbalanced V434-versus-identical-V434 block. "
            "This prerequisite persists only numeric/hash evidence and cannot "
            "project, update, select, promote, or open row 64 or later."
        ),
        "authorization": {
            "authority_origin": "this_specific_v65a_preregistration_only",
            "gpu_launch": True,
            "alpha_zero_calibration": True,
            "physical_gpu_ids": [0, 1, 2, 3],
            "actors": 4,
            "tensor_parallel_size_per_actor": 1,
            "projection": False,
            "optimizer_update": False,
            "adapter_update": False,
            "candidate": False,
            "candidate_snapshot": False,
            "hpo_population": False,
            "train_holdback": False,
            "exact_sentinel": False,
            "unused_reserve": False,
            "ood_shadow": False,
            "protected_semantics": False,
            "terminal_holdout": False,
            "promotion": False,
        },
        "source_evidence": sources,
        "ranking_panel": {
            "path": str(panel_path),
            "file_sha256": payload_sha256_v65a(panel),
            "content_sha256": panel["content_sha256_before_self_field"],
            "units": 64,
            "request_order_sha256": panel["request_order_sha256"],
            "unit_order_sha256": panel["unit_order_sha256"],
            "hash_only": True,
        },
        "access_contract": {
            "builder_or_dry_run_reads_raw_dataset_or_semantic_rows": False,
            "builder_or_dry_run_reads_base_model_directory_bytes": False,
            "live_semantic_dataset_path": str(population65.V61C_ROWS),
            "decode_exactly_first_64_v61c_ranking_rows": True,
            "decode_v61c_row_64_or_later": False,
            "ranking_prefix_bytes": design65a.RANKING_PREFIX_BYTES_V65A,
            "ranking_prefix_sha256": design65a.RANKING_PREFIX_SHA256_V65A,
            "source_file_size_metadata_bytes": (
                design65a.RANKING_SOURCE_FILE_SIZE_BYTES_V65A
            ),
            "live_authorized_prefix_pread_count": 2,
            "live_read_primitive": (
                "one decoded exact-prefix os.pread before generation and one "
                "hash-only exact-prefix os.pread after cleanup"
            ),
            "postrun_prefix_integrity_pread_decodes_semantics": False,
            "full_jsonl_hash_verification_or_full_file_read_live": False,
            "v61c_hash_numeric_panel_may_open_live": True,
            "full_train_membership_holdback_sentinel_ood_protected_or_"
            "terminal_may_open": False,
            "raw_question_answer_prompt_or_generation_text_may_be_persisted": False,
            "numeric_hash_only_evidence_required": True,
        },
        "fixed_calibration_recipe": {
            "base_model": common65.base_model_binding_v65(),
            "v434_adapter": common65.v434_binding_v65(),
            "lora_request": {
                "name": "v434_ranking64_alpha_zero_v65a",
                "integer_id": 1,
                "path": str(design52.STAGED_V52),
            },
            "reference_and_candidate_are_identical_v434_aliases": True,
            "alpha": 0.0,
            "sigma_or_direction": None,
            "rows_per_actor_call": 64,
            "ranking_units": 64,
            "exact_sentinel_units": 0,
            "physical_gpu_ids": [0, 1, 2, 3],
            "actors": 4,
            "tensor_parallel_size_per_actor": 1,
            "unscored_warmup_periods": design65a.WARMUP_PERIODS_V65A,
            "warmup_state": "exact unchanged pinned V434 master",
            "warmup_generation_completions_discarded": (
                design65a.WARMUP_GENERATION_COMPLETIONS_V65A
            ),
            "warmup_outputs_scored_or_persisted": False,
            "warmup_generation_metrics_computed_or_persisted": False,
            "exact_master_rematerialization": {
                "worker_extension": (
                    "eggroll_es_worker_lora_v65a."
                    "LoRAAdapterStateWorkerExtensionV65A"
                ),
                "rpc": "rematerialize_exact_master_v65a",
                "required_before_every_warmup_and_scored_period": True,
                "period_slot_write_receipts_required": 8,
                "read_only_live_slot_receipts_required": 16,
                "read_only_edges_per_period": [
                    "before_generation", "after_generation",
                ],
                "after_generation_receipt_may_write_or_reset_slot": False,
                "host_copy_for_read_only_hashing_permitted": True,
                "source_is_unchanged_pinned_fp32_master": True,
                "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
                "bf16_runtime_values_sha256": (
                    design52.MASTER_RUNTIME_SHA256_V52
                ),
                "candidate_or_perturbation_materialized": False,
            },
            "exact_master_receipt_before_and_after_every_warmup": True,
            "scored_periods": design65a.SCORED_PERIODS_V65A,
            "scored_label_plan": dict(design65a.LABEL_PLAN_V65A),
            "pair_periods": [list(pair) for pair in design65a.PAIR_PERIODS_V65A],
            "pairs_per_actor": design65a.PAIRS_PER_ACTOR_V65A,
            "candidate_before_reference_pairs_per_actor": 1,
            "candidate_after_reference_pairs_per_actor": 1,
            "label_order_matches_v65_plus0_minus0_minus1_plus1": True,
            "paired_replicas_per_unit": design65a.REPLICAS_PER_UNIT_V65A,
            "exact_master_receipt_before_and_after_every_scored_period": True,
            "scored_generation_completions": (
                design65a.SCORED_GENERATION_COMPLETIONS_V65A
            ),
            "total_generation_completions": (
                design65a.TOTAL_GENERATION_COMPLETIONS_V65A
            ),
            "common_generation_seed": design65a.COMMON_GENERATION_SEED_V65A,
            "generation_params_without_seed": dict(
                design65a.GENERATION_PARAMS_WITHOUT_SEED_V65A
            ),
            "runtime_determinism_controls": dict(
                design65a.ENGINE_CONTROLS_V65A
            ),
            "sanitized_live_engine_and_cache_receipt": {
                "required_from_every_actor": True,
                "unique_actor_pid_and_physical_gpu": True,
                "engine_fields_must_equal_runtime_determinism_controls": True,
                "active_lora_ids_exactly": [1],
                "manager_and_cpu_cache_lora_ids_exactly": [1],
                "registered_and_gpu_slot_match_exact_staged_v434_bytes": True,
                "staged_v434_weights_file_sha256": (
                    design52.STAGED_WEIGHTS_SHA256_V52
                ),
                "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
                "bf16_runtime_values_sha256": (
                    design52.MASTER_RUNTIME_SHA256_V52
                ),
                "extra_or_candidate_adapter_loaded": False,
                "raw_model_config_semantics_or_text_persisted": False,
            },
            "adaptive_retry_drop_reorder_or_early_stop": False,
            "adapter_update_candidate_hpo_or_promotion_performed": False,
        },
        "numeric_analysis_contract": {
            "primary_components": [
                "paired_generated_f1_candidate_minus_reference",
                "paired_generated_nonzero_candidate_minus_reference",
                "reference_instability_minus_candidate_instability",
            ],
            "pairing_keys": [
                "unit_identity_sha256", "actor_rank", "pair_index",
            ],
            "within_unit_actor_pair_replicas_preserved_and_averaged": 8,
            "resampled_axis": "conflict_unit_only",
            "single_replica_per_resampled_unit_sampling": False,
            "bootstrap_replicates": design65a.BOOTSTRAP_REPLICATES_V65A,
            "bootstrap_seed": design65a.BOOTSTRAP_SEED_V65A,
            "bootstrap_index_matrix_sha256": hashlib.sha256(
                bootstrap_indices.astype("<i8", copy=False).tobytes(order="C")
            ).hexdigest(),
            "one_sided_alpha": design65a.ONE_SIDED_ALPHA_V65A,
            "joint_composite_weights": dict(
                design65a.COMPOSITE_WEIGHTS_V65A
            ),
            "joint_composite_distribution_bootstrapped_before_quantiles": True,
            "temporal_pair_joint_composite_intervals": {
                "pair_0": {
                    "periods": [0, 1],
                    "actor_replicas_preserved_and_averaged": 4,
                    "sealed_fields": [
                        "point", "lcb", "ucb", "halfwidth",
                        "contains_zero", "null_radius",
                    ],
                },
                "pair_1": {
                    "periods": [2, 3],
                    "actor_replicas_preserved_and_averaged": 4,
                    "sealed_fields": [
                        "point", "lcb", "ucb", "halfwidth",
                        "contains_zero", "null_radius",
                    ],
                },
            },
            "B_C_pass_definition": (
                "max(pair_0.null_radius,pair_1.null_radius)"
            ),
            "future_v65_null_bound_transfer": {
                "outcome_independent_field_mapping": dict(
                    design65a.FUTURE_V65_NULL_BOUND_TRANSFER_V65A
                ),
                "required_spread_gates": {
                    "pooled_joint_composite": "spread_strictly_greater_than_2*B_C",
                    "each_pass_joint_composite": (
                        "spread_strictly_greater_than_2*B_C_pass"
                    ),
                    "generated_f1_when_used": (
                        "spread_strictly_greater_than_2*B_F"
                    ),
                    "stability_when_used": (
                        "spread_strictly_greater_than_2*B_S"
                    ),
                    "stability_coefficient_when_gate_not_met": 0.0,
                    "stability_gate_not_met_causes_population_failure": False,
                },
                "mapping_or_gates_may_change_after_observing_v65a": False,
                "rebind_or_launch_requires_required_alpha_zero_gate_passed": True,
                "failed_required_alpha_zero_gate_forbids_bound_rebinding_"
                "and_v65_launch": True,
            },
            "later_v65_pass_specific_spread_gate": (
                "each pass-specific direction spread must be strictly greater "
                "than 2*B_C_pass"
            ),
            "each_component_and_joint_composite_seals": [
                "point", "lcb", "ucb", "halfwidth", "contains_zero",
                "null_radius",
            ],
            "required_gates": {
                "generated_f1_primary_interval_contains_zero": True,
                "joint_composite_interval_contains_zero": True,
                "stability_improvement_interval_contains_zero": True,
                "maximum_primary_ci_halfwidth_inclusive": (
                    design65a.MAX_PRIMARY_CI_HALFWIDTH_V65A
                ),
                "maximum_actor_leave_one_out_shift_inclusive": (
                    design65a.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V65A
                ),
            },
            "exact_is_numeric_diagnostic_not_gate": True,
            "exact_sentinel_logic_present": False,
            "success_authorizes_population_update_or_promotion": False,
        },
        "runtime": dict(design52.RUNTIME_V52),
        "required_python": str(design52.REQUIRED_PYTHON_V52),
        "implementation_bindings": implementation_bindings,
        "implementation_closure_manifest_sha256": (
            design65a.canonical_sha256_v65a({
                key: binding["file_sha256"]
                for key, binding in sorted(implementation_bindings.items())
            })
        ),
        "artifacts": artifacts_v65a(),
        "required_integrity_gates": {
            "exact_idle_four_gpu_preflight": True,
            "exact_base_model_and_v434_byte_identities": True,
            "exact_first64_prefix_and_panel_order": True,
            "four_unscored_master_warmups_before_scored_block": True,
            "all_warmup_outputs_discarded_without_scoring_or_persistence": True,
            "all_four_engine_and_cache_receipts_exact": True,
            "all_four_scored_periods_complete_without_schedule_change": True,
            "all_periods_have_read_only_exact_slot_hash_before_and_after": True,
            "all_eight_paired_replicas_preserved_before_unit_bootstrap": True,
            "all_four_gpus_attributed_positive_each_generation_phase": True,
            "strict_four_engine_cleanup_and_final_idle": True,
            "adapter_source_and_stage_contract_reverified_unchanged_"
            "postcleanup": True,
            "numeric_hash_only_evidence": True,
            "row64plus_update_hpo_holdback_sentinel_ood_protected_terminal_"
            "access_zero": True,
        },
        "raw_question_answer_prompt_or_generation_text_may_be_persisted": False,
        "row_64_or_later_opened": False,
        "adapter_update_candidate_hpo_or_promotion_performed": False,
        "protected_semantics_opened": False,
    }
    value["content_sha256_before_self_field"] = (
        design65a.canonical_sha256_v65a(value)
    )
    return value


def build_v65a(
    *,
    ranking_panel_output: Path = RANKING_PANEL_OUTPUT,
    implementation_entry_paths: dict[str, Path] | None = None,
    _test_only_allow_nonproduction_entry_paths: bool = False,
) -> tuple[dict, dict]:
    panel, sources = sealed_source_bindings_v65a()
    entry_paths = _implementation_entry_paths_v65a(
        implementation_entry_paths,
        test_only_allow_nonproduction_entry_paths=(
            _test_only_allow_nonproduction_entry_paths
        ),
    )
    implementation = implementation_bindings_v65a(
        entry_paths
    )
    return panel, build_preregistration_v65a(
        panel, sources, implementation,
        ranking_panel_output=ranking_panel_output,
        implementation_entry_paths=(
            entry_paths
            if _test_only_allow_nonproduction_entry_paths is True
            else None
        ),
        _test_only_allow_nonproduction_entry_paths=(
            _test_only_allow_nonproduction_entry_paths
        ),
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking-panel-output", default=str(RANKING_PANEL_OUTPUT))
    parser.add_argument(
        "--preregistration-output", default=str(PREREGISTRATION_OUTPUT),
    )
    args = parser.parse_args(argv)
    panel_path = Path(args.ranking_panel_output).resolve()
    preregistration_path = Path(args.preregistration_output).resolve()
    panel, preregistration = build_v65a(ranking_panel_output=panel_path)
    common65._exclusive_write_pair_v65(
        panel_path, json_payload_v65a(panel),
        preregistration_path, json_payload_v65a(preregistration),
    )
    print(json.dumps({
        "ranking_panel": str(panel_path),
        "ranking_panel_file_sha256": population65.file_sha256_v65(panel_path),
        "ranking_panel_content_sha256": panel[
            "content_sha256_before_self_field"
        ],
        "preregistration": str(preregistration_path),
        "preregistration_file_sha256": population65.file_sha256_v65(
            preregistration_path
        ),
        "preregistration_content_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "specific_v65a_exact64_alpha_zero_gpu_launch_authorized": True,
        "builder_raw_semantics_model_ray_gpu_or_protected_accessed": False,
        "update_candidate_hpo_holdback_sentinel_ood_or_terminal_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
