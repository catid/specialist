#!/usr/bin/env python3
"""Seal the holdout-blind V48A generation-boundary design on CPU."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_f1_anchor_projection_v43m as v43m
import lora_es_generation_boundary_sampling_v48a as boundary
import run_lora_es_multi_anchor_v43i as v43i


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_generation_boundary_design_v48a.json"
).resolve()
V43M_NOOP = (
    ROOT / "experiments/eggroll_es_hpo/projections/"
    "lora_es_three_anchor_identity_noop_v43m.json"
).resolve()
V43M_NOOP_FILE_SHA256 = (
    "f08d9730e6f5b8294fe1d9486bbf17ed5c1d193cf6913e8227811c937f20f8e3"
)
V43M_NOOP_CONTENT_SHA256 = (
    "b7d89edda779846efdb3ee9c8b0ac3dcb00953b751fe8de9654ad0b977ec9f91"
)


def _read_v43m_noop() -> dict:
    if v43i.v40a.file_sha256(V43M_NOOP) != V43M_NOOP_FILE_SHA256:
        raise RuntimeError("v48a V43M no-op evidence file changed")
    value = json.loads(V43M_NOOP.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != V43M_NOOP_CONTENT_SHA256
        or v43i.v40a.canonical_sha256(compact) != V43M_NOOP_CONTENT_SHA256
        or value.get("status")
        != "cpu_only_noop_identical_to_rejected_v43i_candidate"
        or value.get("three_anchor_result", {}).get(
            "f1_halfspace_nonbinding"
        ) is not True
        or value.get("three_anchor_result", {}).get(
            "projection_coefficients_changed_from_v43i"
        ) is not False
        or value.get("gpu_model_or_dataset_accessed") is not False
        or value.get("shadow_ood_holdout_or_heldout_opened") is not False
        or value.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v48a V43M no-op evidence content changed")
    return value


def implementation_bindings_v48a() -> dict:
    paths = {
        "design_builder": Path(__file__).resolve(),
        "generation_boundary_runtime": Path(boundary.__file__).resolve(),
        "generation_boundary_tests": (
            ROOT / "test_lora_es_generation_boundary_sampling_v48a.py"
        ),
        "design_tests": (
            ROOT / "test_build_lora_es_generation_boundary_design_v48a.py"
        ),
        "v43i_runtime": Path(v43i.__file__).resolve(),
        "v43i_fused_anchor_runtime": Path(v43i.fused.__file__).resolve(),
        "v43i_multi_anchor_projection": Path(v43i.multi_anchor.__file__).resolve(),
        "v43m_f1_projection_runtime": Path(v43m.__file__).resolve(),
        "v43m_identity_noop_evidence": V43M_NOOP,
        "train_dataset": v43i.DATASET,
        "matched_initial_weights": v43i.SOURCE_WEIGHTS,
        "matched_initial_config": v43i.SOURCE_CONFIG,
        "matched_initial_manifest": v43i.SOURCE_MANIFEST,
    }
    return {
        label: v43i.v40a.file_sha256(path) for label, path in paths.items()
    }


def build_design_v48a() -> dict:
    noop = _read_v43m_noop()
    current_requests = 544
    new_requests = current_requests + boundary.FRAGILE_SUBSET_UNITS_V48A
    signed_actor_states = (
        v43i.POPULATION_SIZE * 2 * boundary.ACTORS_V48A
    )
    result = {
        "schema": "matched-lora-es-generation-boundary-design-v48a",
        "status": "sealed_cpu_design_pending_train_only_base_evidence",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": False,
        "why_not_launchable_yet": [
            "train-only four-actor greedy base evidence has not been generated",
            "content-free train conflict-unit membership artifact is not sealed",
            "the resulting 64-unit request-order artifact is not sealed",
            "the integrated V43I-derived GPU runtime is not yet hash-bound",
        ],
        "problem_audit": {
            "v43i_population_train_rows_teacher_forced": 448,
            "v43i_population_proxy_generation_documents": 32,
            "v43i_fused_requests_per_signed_actor_state": 544,
            "v43i_population_coefficient_objectives": [
                "domain_equal_unit_answer_logprob",
                "prose_lm",
                "qa_answer_logprob",
            ],
            "v43i_greedy_generation_used_for_population_coefficients": False,
            "v43i_greedy_generation_used_only_for_candidate_gate": True,
            "v43m_f1_halfspace_nonbinding": True,
            "v43m_coefficients_changed": False,
            "v43m_noop_file_sha256": V43M_NOOP_FILE_SHA256,
            "v43m_noop_content_sha256": V43M_NOOP_CONTENT_SHA256,
            "rejected_candidate_identity_sha256": noop[
                "already_evaluated_state"
            ]["candidate_state_sha256"],
        },
        "train_only_base_evidence_protocol": {
            "state": "exact matched V41A initialization before population",
            "matched_master_sha256": v43m.V43I_RESTORED_MASTER_SHA256,
            "rows": boundary.TRAIN_ROWS_V48A,
            "actors": boundary.ACTORS_V48A,
            "greedy_completions": (
                boundary.TRAIN_ROWS_V48A * boundary.ACTORS_V48A
            ),
            "generation_params": dict(boundary.GENERATION_PARAMS_V48A),
            "same_prompt_order_and_sampling_on_all_actors": True,
            "retained_fields": [
                "row_sha256", "actor_rank", "prediction_sha256", "f1",
                "exact", "nonzero",
            ],
            "question_answer_or_generation_text_persisted": False,
            "selection_occurs_once_before_any_population_state": True,
            "selection_is_never_recomputed_per_member_or_sign": True,
        },
        "fragile_subset_protocol": {
            "source_train_rows": boundary.TRAIN_ROWS_V48A,
            "source_conflict_units": boundary.TRAIN_CONFLICT_UNITS_V48A,
            "selected_rows": boundary.FRAGILE_SUBSET_UNITS_V48A,
            "selected_conflict_units": boundary.FRAGILE_SUBSET_UNITS_V48A,
            "maximum_rows_per_selected_conflict_unit": 1,
            "equal_weight_per_selected_conflict_unit": (
                1.0 / boundary.FRAGILE_SUBSET_UNITS_V48A
            ),
            "selection_seed": boundary.SELECTION_SEED_V48A,
            "stratum_cycle": list(boundary.STRATUM_CYCLE_V48A),
            "priority_meaning": {
                "unstable": "actor prediction or numeric generation disagreement",
                "partial": "stable nonzero but non-exact generated answer",
                "exact": "stable exact answer vulnerable to loss",
                "zero": "stable zero-F1 answer with improvement headroom",
            },
            "teacher_forced_domain_rows_duplicated": False,
            "teacher_forced_domain_weights_changed": False,
            "naive_row_oversampling": False,
        },
        "population_protocol": {
            "population_size": v43i.POPULATION_SIZE,
            "signs": ["plus", "minus"],
            "actors_per_signed_state": boundary.ACTORS_V48A,
            "signed_actor_states": signed_actor_states,
            "same_selected_items_order_and_sampling_for_all_states": True,
            "common_random_plan_receipts_required": signed_actor_states,
            "existing_requests_per_actor_state": current_requests,
            "new_requests_per_actor_state": new_requests,
            "request_count_increase_fraction": (
                new_requests / current_requests - 1.0
            ),
            "extra_fragile_generation_completions": (
                boundary.FRAGILE_SUBSET_UNITS_V48A * signed_actor_states
            ),
            "extra_worst_case_decode_tokens": (
                boundary.FRAGILE_SUBSET_UNITS_V48A
                * signed_actor_states
                * boundary.GENERATION_PARAMS_V48A["max_tokens"]
            ),
            "existing_proxy_generation_completions": 32 * signed_actor_states,
            "generation_decode_work_multiplier_vs_v43i": 3.0,
        },
        "objective_protocol": {
            "primary": (
                "equal average of unit-norm domain and fragile generated-F1 "
                "centered-rank coefficient vectors"
            ),
            "fragile_generated_f1_is_direct_primary_objective": True,
            "fragile_generated_f1_is_not_only_a_halfspace": True,
            "fail_closed_if_fragile_f1_population_spread_is_zero": True,
            "simultaneous_projection_anchors": [
                "domain", "fragile_generation_f1", "prose_lm",
                "qa_answer_logprob",
            ],
            "trust_region_norm_ratio": (
                v43i.multi_anchor.TRUST_REGION_NORM_RATIO_V43H
            ),
            "existing_full_candidate_generation_gate_retained": True,
        },
        "variance_and_bias": {
            "common_random_items_cancel_item_difficulty_in_antithetic_comparisons": True,
            "equal_conflict_unit_weight_prevents_large_documents_dominating": True,
            "base_only_preselection_avoids_member_adaptive_selection_bias": True,
            "fragility_selection_changes_objective_target_and_is_not_an_unbiased_full_train_f1_estimator": True,
            "downstream_full_candidate_gate_remains_required": True,
        },
        "access_contract": {
            "train_dataset_file_sha256": v43i.DATASET_SHA256,
            "runtime_paths_sealed_now": [],
            "future_runtime_may_open_only": [
                "content-addressed train rows",
                "content-free train-only conflict-unit membership",
                "content-free four-actor base generation evidence",
                "sealed train-only prose and QA proxy anchors already used by V43I",
            ],
            "shadow_path_authorized": False,
            "ood_path_authorized": False,
            "holdout_or_heldout_path_authorized": False,
            "benchmark_path_authorized": False,
            "protected_semantics_opened_during_design": False,
            "v46d_or_any_holdout_artifact_bound": False,
        },
        "future_seal_requirements": [
            "prove 448-row/208-unit train membership and row identities",
            "prove all four base-evidence actor slots per row and seal numeric hashes",
            "build and seal the deterministic 64-unit subset",
            "bind integrated runtime, worker, source, dataset, membership, evidence, and subset hashes",
            "dry-run with zero protected accesses",
            "retain V43I exact restoration, reliability, precommit, and candidate gates",
        ],
        "implementation_bindings": implementation_bindings_v48a(),
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    }
    result["content_sha256_before_self_field"] = v43i.v40a.canonical_sha256(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_design_v48a()
    v43i.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": v43i.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launch_authorized": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
