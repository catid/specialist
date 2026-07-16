#!/usr/bin/env python3
"""Seal the nested P8-vs-P16 LoRA-ES V52 experiment before launch."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as design


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_nested_p8_vs_p16_v52_retry5.json"
).resolve()
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v52_matched_lora_es_nested_p8_vs_p16_retry5"
).resolve()
TRAIN_PANEL_SELECTION_SEED_V52 = "v52-v434-content-free-generation-panel-20260716"


def _json_bytes_v52(value: dict) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=True, indent=2, sort_keys=True,
            allow_nan=False,
        ) + "\n"
    ).encode("ascii")


def _row_sha256_v52(row: dict) -> str:
    payload = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def seal_v434_train_contract_v52() -> dict:
    """Persist only content-free commitments derived from exact v434 train."""
    import build_train_shadow_folds_v37a as folds
    import v49a_train_only_source_temperature_weighting as v49a
    from qa_quality import qa_pair_from_record

    if design.file_sha256_v52(design.TRAIN_DATASET_V52) != design.DATASET_SHA256_V52:
        raise RuntimeError("v52 v434 train payload changed before commitment seal")
    if design.TRAIN_MEMBERSHIP_V52.exists() or design.TRAIN_GENERATION_PANEL_V52.exists():
        raise FileExistsError("v52 train commitment outputs must both be fresh")
    rows, replay = v49a.replay_v434_train_only()
    units = folds.build_conflict_units(rows)
    if len(rows) != 448 or len(units) != 208:
        raise RuntimeError("v52 v434 train membership coverage changed")
    by_row = {}
    for unit in units:
        for row_index in unit["indices"]:
            if row_index in by_row:
                raise RuntimeError("v52 v434 row belongs to multiple units")
            by_row[row_index] = {
                "unit_identity_sha256": unit["identity_sha256"],
                "row_count": unit["rows"],
            }
    row_sha = [_row_sha256_v52(row) for row in rows]
    items = [{
        "row_index": index,
        "row_sha256": row_sha[index],
        **by_row[index],
    } for index in range(len(rows))]
    membership = {
        "schema": "v52-v434-train-row-conflict-unit-membership",
        "status": "complete_content_free_projection_before_v52_preregistration",
        "source": {
            "train_dataset_path": str(design.TRAIN_DATASET_V52),
            "train_dataset_file_sha256": design.DATASET_SHA256_V52,
            "root_membership_sha256": replay["root_membership_sha256"],
            "root_membership_exactly_frozen_v412_fold3_train": True,
            "v434_replay": replay["v434_replay"],
        },
        "rows": 448,
        "conflict_units": 208,
        "items": items,
        "ordered_row_sha256": design.canonical_sha256_v52(row_sha),
        "ordered_membership_sha256": design.canonical_sha256_v52(items),
        "question_answer_evidence_or_text_persisted": False,
        "model_outcomes_used_for_selection": False,
        "nontrain_semantics_opened": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_semantics_opened": False,
    }
    membership["content_sha256_before_self_field"] = (
        design.canonical_sha256_v52(membership)
    )
    membership_payload = _json_bytes_v52(membership)
    membership_file_sha = hashlib.sha256(membership_payload).hexdigest()

    qa_pairs = [qa_pair_from_record(row) for row in rows]
    if any(pair is None or not pair[0] or not pair[1] for pair in qa_pairs):
        raise RuntimeError("v52 v434 train row is not a valid QA pair")
    weights = [1.0 / (208 * item["row_count"]) for item in items]
    train_bundle = {
        "schema": "eggroll-es-equal-unit-train-bundle-v52-v434",
        "dataset": {
            "path": str(design.TRAIN_DATASET_V52),
            "file_sha256": design.DATASET_SHA256_V52,
            "rows": 448,
            "ordered_row_sha256": membership["ordered_row_sha256"],
        },
        "train_membership": {
            "path": str(design.TRAIN_MEMBERSHIP_V52),
            "file_sha256": membership_file_sha,
            "content_sha256": membership[
                "content_sha256_before_self_field"
            ],
            "ordered_membership_sha256": membership[
                "ordered_membership_sha256"
            ],
        },
        "questions": [pair[0] for pair in qa_pairs],
        "answers": [pair[1] for pair in qa_pairs],
        "weights": weights,
        "row_sha256": row_sha,
        "conflict_units": 208,
        "weight_identity_sha256": design.canonical_sha256_v52([{
            "row_sha256": item["row_sha256"],
            "unit_identity_sha256": item["unit_identity_sha256"],
            "unit_rows": item["row_count"],
        } for item in items]),
        "unit_membership_v48b": [{
            "row_sha256": item["row_sha256"],
            "unit_identity_sha256": item["unit_identity_sha256"],
            "row_count": item["row_count"],
        } for item in items],
    }
    train_bundle_content_sha = design.canonical_sha256_v52(train_bundle)

    representatives = []
    for unit in units:
        candidates = [items[index] for index in unit["indices"]]
        representative = min(candidates, key=lambda item: hashlib.sha256((
            TRAIN_PANEL_SELECTION_SEED_V52 + "\0row\0"
            + item["unit_identity_sha256"] + "\0" + item["row_sha256"]
        ).encode("ascii")).hexdigest())
        representatives.append({
            **representative,
            "representative_tie_break_sha256": hashlib.sha256((
                TRAIN_PANEL_SELECTION_SEED_V52 + "\0row\0"
                + representative["unit_identity_sha256"] + "\0"
                + representative["row_sha256"]
            ).encode("ascii")).hexdigest(),
            "unit_selection_sha256": hashlib.sha256((
                TRAIN_PANEL_SELECTION_SEED_V52 + "\0unit\0"
                + representative["unit_identity_sha256"]
            ).encode("ascii")).hexdigest(),
        })
    selected = sorted(
        representatives,
        key=lambda item: (
            item["unit_selection_sha256"],
            item["unit_identity_sha256"],
            item["row_sha256"],
        ),
    )[:64]
    panel_items = [{
        "request_index": index,
        "row_sha256": item["row_sha256"],
        "unit_identity_sha256": item["unit_identity_sha256"],
        "unit_row_count": item["row_count"],
        "representative_tie_break_sha256": item[
            "representative_tie_break_sha256"
        ],
        "unit_selection_sha256": item["unit_selection_sha256"],
        "equal_conflict_unit_weight": 1.0 / 64.0,
    } for index, item in enumerate(selected)]
    request_rows = [item["row_sha256"] for item in panel_items]
    subset = {
        "schema": "v52-v434-train-generation-panel",
        "status": "selected_content_free_before_v52_preregistration",
        "selection_seed": TRAIN_PANEL_SELECTION_SEED_V52,
        "selection_method": (
            "one deterministic representative per v434 conflict unit; "
            "64 units with smallest seeded unit hashes"
        ),
        "source_train_rows": 448,
        "source_conflict_units": 208,
        "selected_rows": 64,
        "selected_conflict_units": 64,
        "items": panel_items,
        "request_order_row_sha256": request_rows,
        "request_order_sha256": design.canonical_sha256_v52(request_rows),
        "common_random_generation_params": {
            "n": 1,
            "seed": 2026071543,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 64,
            "detokenize": True,
        },
        "teacher_forced_domain_sampling_changed": False,
        "rows_duplicated_or_oversampled_in_domain_objective": False,
        "model_outcomes_used_for_selection": False,
        "protected_semantics_persisted": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    subset["content_sha256_before_self_field"] = (
        design.canonical_sha256_v52(subset)
    )
    panel = {
        "schema": "sealed-v434-train-generation-panel-v52",
        "status": "complete_before_v52_preregistration",
        "source": {
            "train_dataset_path": str(design.TRAIN_DATASET_V52),
            "train_dataset_file_sha256": design.DATASET_SHA256_V52,
            "membership_path": str(design.TRAIN_MEMBERSHIP_V52),
            "membership_file_sha256": membership_file_sha,
            "membership_content_sha256": membership[
                "content_sha256_before_self_field"
            ],
            "train_bundle_content_sha256": train_bundle_content_sha,
        },
        "subset": subset,
        "selected_rows": 64,
        "selected_conflict_units": 64,
        "request_order_sha256": subset["request_order_sha256"],
        "common_random_generation_params": dict(
            subset["common_random_generation_params"]
        ),
        "question_answer_or_generation_text_persisted": False,
        "model_outcomes_used_for_selection": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
        "gpu_or_model_accessed": False,
    }
    panel["content_sha256_before_self_field"] = design.canonical_sha256_v52(
        panel
    )
    panel_payload = _json_bytes_v52(panel)
    design.TRAIN_MEMBERSHIP_V52.parent.mkdir(parents=True, exist_ok=True)
    with design.TRAIN_MEMBERSHIP_V52.open("xb") as handle:
        handle.write(membership_payload)
    try:
        with design.TRAIN_GENERATION_PANEL_V52.open("xb") as handle:
            handle.write(panel_payload)
    except BaseException:
        design.TRAIN_MEMBERSHIP_V52.unlink(missing_ok=True)
        raise
    return {
        "membership_path": str(design.TRAIN_MEMBERSHIP_V52),
        "membership_file_sha256": membership_file_sha,
        "membership_content_sha256": membership[
            "content_sha256_before_self_field"
        ],
        "train_bundle_content_sha256": train_bundle_content_sha,
        "panel_path": str(design.TRAIN_GENERATION_PANEL_V52),
        "panel_file_sha256": hashlib.sha256(panel_payload).hexdigest(),
        "panel_content_sha256": panel["content_sha256_before_self_field"],
        "request_order_sha256": panel["request_order_sha256"],
        "rows": 448,
        "conflict_units": 208,
        "selected_conflict_units": 64,
        "protected_semantics_opened": False,
        "nontrain_semantics_opened": False,
        "gpu_or_model_accessed": False,
    }


def implementation_bindings_v52() -> dict:
    paths = {
        "design_v52": ROOT / "lora_es_nested_population_v52.py",
        "worker_v52": ROOT / "eggroll_es_worker_lora_v52.py",
        "builder_v52": Path(__file__).resolve(),
        "runner_v52": ROOT / "run_lora_es_nested_population_v52.py",
        "tests_v52": ROOT / "test_lora_es_nested_population_v52.py",
        "v51_pipeline": ROOT / "lora_es_direct_master_pipeline_v51.py",
        "v51_worker": ROOT / "eggroll_es_worker_lora_v51.py",
        "v48b_runtime": ROOT / "run_lora_es_generation_boundary_v48b.py",
        "v48e_projection": ROOT / "lora_es_generation_boundary_reprojection_v48e.py",
        "source_weights_v434_equal": design.SOURCE_WEIGHTS_V52,
        "source_config_v434_equal": design.SOURCE_CONFIG_V52,
        "staged_weights_v434_equal": design.STAGED_WEIGHTS_V52,
        "staged_config_v434_equal": design.STAGED_CONFIG_V52,
        "staged_manifest_v434_equal": design.STAGED_MANIFEST_V52,
        "train_dataset_v434_fold3": design.TRAIN_DATASET_V52,
        "train_membership_v52": design.TRAIN_MEMBERSHIP_V52,
        "train_generation_panel_v52": design.TRAIN_GENERATION_PANEL_V52,
    }
    return {
        label: design.file_sha256_v52(path) for label, path in paths.items()
    }


def _sealed_parent_receipts_v52() -> dict:
    values = {
        label: design.sealed_json_v52(label)
        for label in design.SEALED_NUMERIC_PARENTS_V52
    }
    if (
        values["v48b_preregistration"].get("schema")
        != "matched-lora-es-generation-boundary-preregistration-v48b"
        or values["v48b_preregistration"].get("runtime")
        != design.RUNTIME_V52
        or values["v48b_preregistration"].get(
            "protected_semantic_access_authorized"
        ) is not False
        or values["v48e_report"].get("status")
        != "complete_no_scale_passed_all_candidates_exactly_restored"
        or values["v48e_report"].get("protected_semantics_opened") is not False
        or values["v48e_report"].get(
            "shadow_ood_holdout_or_benchmark_opened"
        ) is not False
        or values["v51_acceptance_audit"].get("status")
        != "accepted_for_future_lora_es_population_transitions"
        or values["v51_acceptance_audit"].get("decision", {}).get("scope")
        != "population transition implementation only"
        or values["v51_acceptance_audit"].get("decision", {}).get(
            "quality_or_checkpoint_promotion_authorized"
        ) is not False
        or values["v49d_equal_train_report"].get("status")
        != "complete_train_only_state_sealed_non_train_unopened"
        or values["v49d_stage_manifest"].get("status")
        != "complete_cpu_only_key_transform"
        or values["v49e_replicated_shadow_report"].get("status")
        != "complete_shadow_only_holdout_unopened"
        or values["v49e_replicated_shadow_report"].get(
            "ood_eligibility_proof", {}
        ).get("both_equal_replicas_independently_ood_eligible") is not True
        or values["v49e_replicated_shadow_report"].get(
            "replicated_equal_vs_base_decision", {}
        ).get("replicated_equal_vs_base_decision_passed") is not True
        or values["v49e_replicated_shadow_report"].get(
            "heldout_or_holdout_opened"
        ) is not False
        or values["v52_original_preregistration"].get("schema")
        != "matched-lora-es-nested-p8-vs-p16-preregistration-v52"
        or values["v52_original_preregistration"].get("sealed_holdout_opened")
        is not False
        or values["v52_pre_model_attempt"].get("status")
        != "launching_train_only"
        or values["v52_pre_model_attempt"].get("preflight", {}).get(
            "compute_process_query_empty"
        ) is not True
        or values["v52_pre_model_failure"].get("type")
        != "ModuleNotFoundError"
        or values["v52_pre_model_failure"].get("message")
        != "No module named 'vllm'"
        or values["v52_pre_model_failure"].get("protected_semantics_opened")
        is not False
        or values["v52_pre_model_failure"].get("sealed_holdout_opened")
        is not False
        or values["v52_retry1_preregistration"].get("retry_revision")
        != "retry1_launcher_environment_only"
        or values["v52_retry1_attempt"].get("launcher", {}).get("matched")
        is not True
        or values["v52_retry1_failure"].get("type") != "RuntimeError"
        or values["v52_retry1_failure"].get("message")
        != "v43i zero-effect base score differs across actors"
        or values["v52_retry1_failure"].get("strict_cleanup", {}).get(
            "all_four_gcs_states_removed"
        ) is not True
        or values["v52_retry1_failure"].get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or values["v52_retry1_failure"].get("protected_semantics_opened")
        is not False
        or design.file_sha256_v52(design.RETRY1_GPU_LOG_V52)
        != design.RETRY1_GPU_LOG_SHA256_V52
        or values["v52_retry2_preregistration"].get("retry_revision")
        != "retry2_actor_indexed_nondeterminism_measurement"
        or values["v52_retry2_attempt"].get("launcher", {}).get("matched")
        is not True
        or values["v52_retry2_preinstall_baseline"].get(
            "baseline", {}
        ).get("cross_actor_exact_score_consensus") is not False
        or values["v52_retry2_failure"].get("type") != "RuntimeError"
        or values["v52_retry2_failure"].get("message")
        != "v52 postinstall score/output manifest changed within an actor"
        or values["v52_retry2_failure"].get("strict_cleanup", {}).get(
            "all_four_gcs_states_removed"
        ) is not True
        or values["v52_retry2_failure"].get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or values["v52_retry2_failure"].get("protected_semantics_opened")
        is not False
        or values["v52_retry2_failure"].get("sealed_holdout_opened")
        is not False
        or len(values["v52_retry2_failure"].get("certificates", [])) != 4
        or {
            item.get("current_identity", {}).get("sha256")
            for item in values["v52_retry2_failure"].get("certificates", [])
        } != {design.MASTER_SHA256_V52}
        or {
            item.get("materialization", {}).get("runtime_values_sha256")
            for item in values["v52_retry2_failure"].get("certificates", [])
        } != {design.MASTER_RUNTIME_SHA256_V52}
        or design.file_sha256_v52(design.RETRY2_GPU_LOG_V52)
        != design.RETRY2_GPU_LOG_SHA256_V52
        or values["v52_retry3_preregistration"].get("retry_revision")
        != "retry3_state_certificate_repair_and_shared_reliability_tightening"
        or values["v52_retry3_attempt"].get("launcher", {}).get("matched")
        is not True
        or values["v52_retry3_master_identity_audit"].get(
            "postinstall_exact_adapter_state_gate", {}
        ).get("all_four_canonical_fp32_master_identities_exact") is not True
        or values["v52_retry3_numeric_calibration"].get("schema")
        != "matched-lora-es-numeric-calibration-v43i"
        or set(values["v52_retry3_numeric_calibration"].get(
            "bootstrap", {}
        ).get("bounds", {})) != {"equal_unit_mean", "unweighted_row_mean"}
        or values["v52_retry3_failure"].get("type") != "KeyError"
        or values["v52_retry3_failure"].get("message")
        != "'equal_unit_mean'"
        or design.canonical_sha256_v52(
            values["v52_retry3_failure"].get("emergency_exact_restore")
        ) != "2054f89691154f02faf3017e08110ee6b1e49fe612dea2fc36f44bc3a5a3d710"
        or design.canonical_sha256_v52(
            values["v52_retry3_failure"].get("strict_cleanup")
        ) != "59157ed14bfbfefeee3a45478e5699fa9175da5e3b23750f1c4c505727157920"
        or design.canonical_sha256_v52(
            values["v52_retry3_failure"].get("final_gpu_idle")
        ) != "7ff0611248500ff79f380d163e62a045f38a3bf9902d03c715555f6fc56eb48a"
        or values["v52_retry3_failure"].get("protected_semantics_opened")
        is not False
        or values["v52_retry3_failure"].get("sealed_holdout_opened")
        is not False
        or design.file_sha256_v52(design.RETRY3_GPU_LOG_V52)
        != design.RETRY3_GPU_LOG_SHA256_V52
        or values["v52_retry4_preregistration"].get("retry_revision")
        != "retry4_calibration_bounds_schema_path_repair"
        or values["v52_retry4_attempt"].get("launcher", {}).get("matched")
        is not True
        or values["v52_retry4_anchor_calibration"].get(
            "calibrated_margins", {}
        ).get("passed") is not True
        or values["v52_retry4_failure"].get("message")
        != "v52 p8 population reliability failed"
        or values["v52_retry4_failure"].get("p8_train_gate") is not None
        or values["v52_retry4_failure"].get("p16_train_gate") is not None
        or values["v52_retry4_failure"].get("emergency_exact_abort")
        is not None
        or design.canonical_sha256_v52(
            values["v52_retry4_failure"].get("emergency_exact_restore")
        ) != "3ad620d05878c3f0894b6cb65b801516c4a0dbc4d66a81c82009eed3b05c3a15"
        or design.canonical_sha256_v52(
            values["v52_retry4_failure"].get("strict_cleanup")
        ) != "080627e271051aca1f70c18e3a128657c57d40e2dbcab6ff7719028f3c6f6cbb"
        or design.canonical_sha256_v52(
            values["v52_retry4_failure"].get("final_gpu_idle")
        ) != "7ff0611248500ff79f380d163e62a045f38a3bf9902d03c715555f6fc56eb48a"
        or values["v52_retry4_failure"].get("protected_semantics_opened")
        is not False
        or values["v52_retry4_failure"].get("sealed_holdout_opened")
        is not False
        or design.file_sha256_v52(design.RETRY4_GPU_LOG_V52)
        != design.RETRY4_GPU_LOG_SHA256_V52
    ):
        raise RuntimeError("v52 sealed V48E/V51 evidence contract changed")
    return {
        label: {
            "path": str(sealed["path"]),
            "file_sha256": sealed["file_sha256"],
            "content_sha256": sealed["content_sha256"],
        }
        for label, sealed in design.SEALED_NUMERIC_PARENTS_V52.items()
    }


def build_v52() -> dict:
    parents = _sealed_parent_receipts_v52()
    arms = design.scientific_arms_v52()
    variable = design.assert_one_scientific_variable_v52(arms)
    states = design.state_derivations_v52()
    compute = design.compute_plan_v52()
    artifacts = {
        "attempt": str(
            RUN_DIR.parent
            / ".v52_matched_lora_es_nested_p8_vs_p16_retry5.attempt.json"
        ),
        "run_directory": str(RUN_DIR),
        "population": str(RUN_DIR / "nested_population_v52.json"),
        "p8_train_gate": str(RUN_DIR / "p8_train_gate_v52.json"),
        "p16_train_gate": str(RUN_DIR / "p16_train_gate_v52.json"),
        "p8_candidate_snapshot": str(RUN_DIR / "p8_candidate_v52"),
        "p16_candidate_snapshot": str(RUN_DIR / "p16_candidate_v52"),
        "preinstall_actor_baseline": str(
            RUN_DIR / "preinstall_actor_baseline_v52.json"
        ),
        "master_identity_audit": str(
            RUN_DIR / "master_identity_audit_v52.json"
        ),
        "numeric_calibration": str(RUN_DIR / "numeric_calibration_v52.json"),
        "anchor_calibration": str(RUN_DIR / "anchor_calibration_v52.json"),
        "ood_aggregate": str(RUN_DIR / "ood_first_aggregate_v52.json"),
        "shadow_aggregate": str(RUN_DIR / "document_disjoint_shadow_v52.json"),
        "report": str(RUN_DIR / "nested_population_report_v52.json"),
        "gpu_log": str(RUN_DIR / "gpu_activity_v52.jsonl"),
        "failure": str(RUN_DIR / "failure_v52.json"),
        "sealed_holdout_artifact": None,
    }
    result = {
        "schema": "matched-lora-es-nested-p8-vs-p16-preregistration-v52",
        "retry_revision": design.RETRY_REVISION_V52,
        "status": (
            "preregistered_after_content_free_v434_train_contract_and_before_"
            "v52_model_gpu_or_nontrain_semantic_access"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Test whether doubling antithetic LoRA-ES directions from eight "
            "to sixteen yields a train-gate-feasible, OOD-safe, "
            "document-disjoint shadow improvement."
        ),
        "gpu_launch_authorized": True,
        "optimizer_update_authorized": True,
        "quality_selection_or_promotion_authorized_only_after_all_gates": True,
        "protected_semantic_access_authorized": False,
        "sealed_holdout_access_authorized": False,
        "current_fixed_holdout_cycle_eligible": False,
        "single_scientific_variable": variable,
        "launcher_fix": {
            "required_python": str(design.REQUIRED_PYTHON_V52),
            "change_scope": (
                "fresh_retry5_artifact_paths_plus_both_arm_reliability_"
                "measurement_persistence_only"
            ),
            "original_preregistration_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_original_preregistration"
                ]["file_sha256"]
            ),
            "original_preregistration_content_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_original_preregistration"
                ]["content_sha256"]
            ),
            "failed_attempt_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_pre_model_attempt"
                ]["file_sha256"]
            ),
            "failure_file_sha256": design.SEALED_NUMERIC_PARENTS_V52[
                "v52_pre_model_failure"
            ]["file_sha256"],
            "failure_type": "ModuleNotFoundError",
            "failure_before_model_or_gpu_actor_creation": True,
            "retry1_failure_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry1_failure"
                ]["file_sha256"]
            ),
            "retry1_gpu_log_file_sha256": design.RETRY1_GPU_LOG_SHA256_V52,
            "retry2_attempt_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry2_attempt"
                ]["file_sha256"]
            ),
            "retry2_preinstall_baseline_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry2_preinstall_baseline"
                ]["file_sha256"]
            ),
            "retry2_failure_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry2_failure"
                ]["file_sha256"]
            ),
            "retry2_gpu_log_file_sha256": design.RETRY2_GPU_LOG_SHA256_V52,
            "retry2_failure_after_four_actor_model_load": True,
            "treatment_seeds_sigma_alpha_panels_order_backtracking_changed": (
                False
            ),
            "candidate_repeat_consensus_gate_changed": False,
            "population_reliability_gate_stricter": True,
            "retry3_preregistration_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry3_preregistration"
                ]["file_sha256"]
            ),
            "retry3_attempt_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry3_attempt"
                ]["file_sha256"]
            ),
            "retry3_master_identity_audit_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry3_master_identity_audit"
                ]["file_sha256"]
            ),
            "retry3_numeric_calibration_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry3_numeric_calibration"
                ]["file_sha256"]
            ),
            "retry3_failure_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry3_failure"
                ]["file_sha256"]
            ),
            "retry3_gpu_log_file_sha256": design.RETRY3_GPU_LOG_SHA256_V52,
            "retry3_emergency_exact_restore_content_sha256": (
                "2054f89691154f02faf3017e08110ee6b1e49fe612dea2fc36f44bc3a5a3d710"
            ),
            "retry3_strict_cleanup_content_sha256": (
                "59157ed14bfbfefeee3a45478e5699fa9175da5e3b23750f1c4c505727157920"
            ),
            "retry3_final_gpu_idle_content_sha256": (
                "7ff0611248500ff79f380d163e62a045f38a3bf9902d03c715555f6fc56eb48a"
            ),
            "retry3_treatment_and_all_gates_changed": False,
            "numeric_calibration_extraction_change": (
                "bootstrap.equal_unit_mean -> bootstrap.bounds.equal_unit_mean"
            ),
            "retry4_preregistration_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry4_preregistration"
                ]["file_sha256"]
            ),
            "retry4_attempt_file_sha256": design.SEALED_NUMERIC_PARENTS_V52[
                "v52_retry4_attempt"
            ]["file_sha256"],
            "retry4_master_identity_audit_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry4_master_identity_audit"
                ]["file_sha256"]
            ),
            "retry4_numeric_calibration_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry4_numeric_calibration"
                ]["file_sha256"]
            ),
            "retry4_anchor_calibration_file_sha256": (
                design.SEALED_NUMERIC_PARENTS_V52[
                    "v52_retry4_anchor_calibration"
                ]["file_sha256"]
            ),
            "retry4_failure_file_sha256": design.SEALED_NUMERIC_PARENTS_V52[
                "v52_retry4_failure"
            ]["file_sha256"],
            "retry4_gpu_log_file_sha256": design.RETRY4_GPU_LOG_SHA256_V52,
            "retry4_emergency_exact_restore_content_sha256": (
                "3ad620d05878c3f0894b6cb65b801516c4a0dbc4d66a81c82009eed3b05c3a15"
            ),
            "retry4_strict_cleanup_content_sha256": (
                "080627e271051aca1f70c18e3a128657c57d40e2dbcab6ff7719028f3c6f6cbb"
            ),
            "retry4_no_update_or_protected_access": True,
            "retry4_treatment_reliability_thresholds_and_stop_changed": False,
        },
        "measurement_contract": {
            "reason": (
                "retry2 proved actor-local full-domain score replays are not "
                "bit-repeatable despite exact FP32 and BF16 adapter identity"
            ),
            "cross_actor_score_bit_equality_required": False,
            "four_actor_master_identity_consensus_required": True,
            "four_actor_runtime_identity_consensus_required": True,
            "preinstall_actor_indexed_score_and_output_manifests_persisted": True,
            "postinstall_actor_indexed_replay_and_exact_deltas_persisted": True,
            "postinstall_score_replay_is_descriptive_nonbinding": True,
            "score_or_output_replay_used_as_restoration_gate": False,
            "postnumeric_full_domain_replay_eliminated": True,
            "postanchor_full_domain_replay_eliminated": True,
            "postpopulation_full_domain_replay_eliminated": True,
            "final_full_domain_replay_eliminated": True,
            "restoration_gate": (
                "exact four-actor canonical FP32 master, BF16 runtime, fresh "
                "reference generation, update sequence, and transaction "
                "quiescence certificates"
            ),
            "restoration_phases": [
                "postinstall", "post_numeric_calibration",
                "post_anchor_calibration", "postpopulation",
                "final_restored",
            ],
            "population_actor_reducer": (
                "fixed arithmetic mean in ascending actor-rank order"
            ),
            "population_actor_count": 4,
            "missing_or_reordered_actor": "fail_closed",
            "numeric_tolerance_used_for_identity": False,
            "fresh_numeric_retained_repeats_authoritative": True,
            "fresh_maximum_repeat_actor_spread_authoritative": True,
            "candidate_gate_remains_repeat_consensus_based": True,
            "both_arm_reliability_receipts_persist_before_global_stop": True,
            "signed_scores_and_receipt_hashes_persist_before_global_stop": True,
            "projection_authorized_only_if_both_arms_reliable": True,
            "unreliable_arm_projection_forbidden": True,
            "global_stop_decision_changed_from_retry4": False,
            "raw_questions_answers_or_outputs_persisted": False,
        },
        "retry3_protocol_delta": {
            "sealed_before_any_v52_population_observation": True,
            "shared_identically_by_p8_and_p16": True,
            "estimated_population_signal_scale": (
                "sqrt(max(0, observed actor-mean central response variance "
                "minus mean-of-four actor noise variance))"
            ),
            "fresh_numeric_noise_scale": (
                "maximum retained-repeat actor spread from the shared fresh "
                "equal-unit calibration"
            ),
            "strict_clearance_required": True,
            "comparison": "estimated_population_signal_scale > fresh_numeric_noise_scale",
            "reliability_minimum_unchanged": 0.8,
            "split_half_spearman_minimum_unchanged": 0.7,
            "historical_calibration_ceiling_unchanged": 0.001802103667415178,
            "population_reliability_eligibility_stricter_than_retry2": True,
        },
        "arms": arms,
        "fixed_recipe": {
            "model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "matched_initialization": str(design.SOURCE_V52),
            "staged_initialization": str(design.STAGED_V52),
            "master_sha256": design.MASTER_SHA256_V52,
            "master_runtime_values_sha256": design.MASTER_RUNTIME_SHA256_V52,
            "master_ordered_key_sha256": design.MASTER_ORDERED_KEY_SHA256_V52,
            "master_tensor_count": 70,
            "master_element_count": 4_528_128,
            "master_bytes": 18_112_512,
            "runtime_view_count": 82,
            "runtime_element_count": 4_921_344,
            "runtime_assignment_sha256": design.RUNTIME_ASSIGNMENT_SHA256_V52,
            "source_weights_sha256": design.SOURCE_WEIGHTS_SHA256_V52,
            "source_config_sha256": design.SOURCE_CONFIG_SHA256_V52,
            "staged_weights_sha256": design.STAGED_WEIGHTS_SHA256_V52,
            "staged_config_sha256": design.STAGED_CONFIG_SHA256_V52,
            "staged_manifest_file_sha256": (
                design.STAGED_MANIFEST_FILE_SHA256_V52
            ),
            "staged_manifest_content_sha256": (
                design.STAGED_MANIFEST_CONTENT_SHA256_V52
            ),
            "staged_transformed_identity_sha256": (
                design.STAGED_TRANSFORMED_IDENTITY_SHA256_V52
            ),
            "staged_ordered_values_sha256": (
                design.STAGED_ORDERED_VALUES_SHA256_V52
            ),
            "dataset": str(design.TRAIN_DATASET_V52),
            "dataset_rows": 448,
            "dataset_conflict_units": 208,
            "dataset_sha256": design.DATASET_SHA256_V52,
            "train_bundle_content_sha256": (
                design.TRAIN_BUNDLE_CONTENT_SHA256_V52
            ),
            "membership": str(design.TRAIN_MEMBERSHIP_V52),
            "membership_file_sha256": design.MEMBERSHIP_SHA256_V52,
            "membership_content_sha256": (
                design.MEMBERSHIP_CONTENT_SHA256_V52
            ),
            "generation_panel": str(design.TRAIN_GENERATION_PANEL_V52),
            "generation_panel_file_sha256": design.SUBSET_FILE_SHA256_V52,
            "generation_panel_content_sha256": design.SUBSET_CONTENT_SHA256_V52,
            "generation_panel_conflict_units": 64,
            "generation_panel_selection": (
                "content-free deterministic v434 conflict-unit panel; no "
                "model outcomes used"
            ),
            "request_order_sha256": design.REQUEST_ORDER_SHA256_V52,
            "full_candidate_requests_per_actor": 896,
            "population_requests_per_actor_state": 608,
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
            "physical_gpu_ids": [0, 1, 2, 3],
            "worker_extension": (
                "eggroll_es_worker_lora_v52."
                "LoRAAdapterStateWorkerExtensionV52"
            ),
        },
        "runtime": dict(design.RUNTIME_V52),
        "state_derivations": states,
        "state_derivation_inventory_sha256": design.canonical_sha256_v52(states),
        "transition_schedule": {
            "accepted_source": "V51 direct pinned-master single-slot transition",
            "scientific_variable": False,
            "state_order": "direction ascending; plus then minus",
            "states": 32,
            "runtime_slots": 1,
            "each_candidate_derived_from_immutable_fp32_master": True,
            "candidate_derived_from_prior_candidate": False,
            "intermediate_exact_master_restores_eliminated": 31,
            "final_exact_master_restore_required": True,
            "emergency_exact_master_restore_required": True,
            "candidate_and_runtime_hash_policy": (
                "four GPU actors must produce one identical hash pair before "
                "the state may be scored; CPU-derived hashes are forbidden"
            ),
            "all_32x4x5_phase_receipts_required": True,
        },
        "train_only_selection": {
            "reliability_minimum": 0.8,
            "split_half_spearman_minimum": 0.7,
            "estimated_signal_standard_deviation_must_strictly_clear_fresh_"
            "maximum_repeat_actor_spread": True,
            "shared_reliability_tightening_preregistered_before_population": (
                True
            ),
            "five_halfspaces": list(design.OBJECTIVE_PATHS_V52),
            "scale_order": list(design.SCALE_ORDER_V52),
            "largest_strictly_passing_scale_per_arm": True,
            "required_checks": list(design.TRAIN_GATE_NAMES_V52),
            "all_nine_checks_required": True,
            "candidate_actor_consensus_required": True,
            "every_rejection_exactly_aborted_before_next_scale": True,
            "neither_arm_passes": "stop_no_update_no_ood_or_shadow_access",
            "shared_fresh_calibration_before_population": True,
            "calibration_master_sha256": design.MASTER_SHA256_V52,
            "calibration_shared_by_both_arms": True,
            "historical_v48e_calibration_values_reused": False,
        },
        "ood_first_gate": {
            "evaluated_before_shadow_ranking": True,
            "candidate_replicas": 2,
            "both_replicas_independently_required": True,
            "qa_items": 24,
            "qa_exact_count_delta_minimum": 0,
            "qa_mean_reward_delta_minimum": 0.0,
            "prose_documents": 16,
            "prose_point_delta_minimum": 0.0,
            "prose_paired_document_bootstrap_95_lcb_minimum": 0.0,
            "bootstrap_samples": 20000,
            "bootstrap_seed": 20260715,
            "protocol_or_leak_counter_increase_maximum": 0,
            "raw_questions_answers_or_generations_persisted": False,
        },
        "document_disjoint_shadow_gate": {
            "opened_only_for_ood_eligible_arms": True,
            "rows": 83,
            "conflict_units": 51,
            "split_manifest_file_sha256": (
                design.SPLIT_MANIFEST_FILE_SHA256_V52
            ),
            "split_manifest_content_sha256": (
                design.SPLIT_MANIFEST_CONTENT_SHA256_V52
            ),
            "required_zero_intersection_keys": list(
                design.EDGE_IDENTITY_KEYS_V52
            ),
            "rank": (
                "mean replicated generated equal-unit reward, exact count, "
                "nonzero count, teacher-forced equal-unit answer logprob"
            ),
            "candidate_must_beat_exact_master": True,
        },
        "treatment_success_rule": (
            "P16 must be OOD eligible and beat exact master on document-disjoint "
            "shadow; if P8 is OOD eligible, P16 must also beat P8"
        ),
        "stop_go_gates": {
            "stop_on_population_identity_or_reliability_failure": True,
            "stop_if_neither_arm_passes_all_train_checks": True,
            "stop_arm_on_any_ood_degradation": True,
            "go_to_shadow_only_for_ood_eligible_arm": True,
            "promote_only_if_treatment_success_rule_passes": True,
            "never_open_sealed_holdout": True,
        },
        "access_contract": {
            "dry_run_reads_only_preregistration_and_code_hashes": True,
            "dry_run_filesystem_writes": False,
            "dry_run_imports_torch_ray_or_model_runtime": False,
            "dry_run_reads_train_semantics": False,
            "dry_run_queries_gpu": False,
            "runtime_train_paths_are_hash_pinned": True,
            "raw_protected_examples_may_not_be_persisted": True,
            "aggregate_only_ood_and_shadow_artifacts": True,
            "sealed_holdout_path_may_not_be_opened": True,
            "forbidden_path_tokens": [
                "holdout", "heldout", "eval_qa_v3", "ood_qa_v3",
                "ood_prose_v3", "shadow_dev",
            ],
        },
        "v49d_isolation": {
            "new_v52_files_and_run_directory_only": True,
            "v49d_artifact_read_allowlist": [
                str(design.TRAIN_DATASET_V52),
                str(design.SOURCE_WEIGHTS_V52),
                str(design.SOURCE_CONFIG_V52),
                str(design.SEALED_NUMERIC_PARENTS_V52[
                    "v49d_equal_train_report"
                ]["path"]),
            ],
            "v49d_artifacts_may_not_be_written_or_imported": True,
            "v49d_protected_inputs_may_not_be_opened": True,
            "launch_requires_no_foreign_gpu_compute_processes": True,
            "launch_not_performed_by_builder_or_tests": True,
        },
        "compute_plan": compute,
        "sealed_numeric_parents": parents,
        "artifacts": artifacts,
        "implementation_bindings": implementation_bindings_v52(),
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
        "sealed_holdout_opened": False,
    }
    retry4_science_and_gate_keys = (
        "single_scientific_variable", "arms", "fixed_recipe", "runtime",
        "state_derivations", "state_derivation_inventory_sha256",
        "transition_schedule", "train_only_selection", "ood_first_gate",
        "document_disjoint_shadow_gate", "treatment_success_rule",
        "stop_go_gates", "compute_plan",
    )
    retry4 = design.sealed_json_v52("v52_retry4_preregistration")
    retry4_science_and_gates = {
        key: retry4[key] for key in retry4_science_and_gate_keys
    }
    retry5_science_and_gates = {
        key: result[key] for key in retry4_science_and_gate_keys
    }
    if retry5_science_and_gates != retry4_science_and_gates:
        raise RuntimeError("v52 retry5 changed Retry4 science or gates")
    result["retry5_equivalence_and_measurement_persistence"] = {
        "retry4_science_and_gate_keys": list(retry4_science_and_gate_keys),
        "retry4_science_and_gates_content_sha256": design.canonical_sha256_v52(
            retry4_science_and_gates
        ),
        "retry5_science_and_gates_content_sha256": design.canonical_sha256_v52(
            retry5_science_and_gates
        ),
        "retry4_science_and_gates_byte_equivalent": True,
        "retry4_shared_stricter_reliability_gate_preserved": True,
        "sole_implementation_change": (
            "persist both reliability receipts and signed-score hashes before "
            "enforcing the unchanged global reliability stop"
        ),
        "original_to_retry3_whole_science_equivalence_claimed": False,
    }
    result["content_sha256_before_self_field"] = design.canonical_sha256_v52(
        result
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--output", type=Path, default=PREREGISTRATION)
    value.add_argument("--seal-v434-train-contract", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.seal_v434_train_contract:
        print(json.dumps(seal_v434_train_contract_v52(), sort_keys=True))
        return 0
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v52()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": design.file_sha256_v52(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "protected_semantics_opened": False,
        "sealed_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
