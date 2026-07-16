#!/usr/bin/env python3
"""Seal deterministic V43K backtracking from exact V43I train-only evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_backtracking_v43k as backtracking
import run_lora_es_backtracking_v43k as runtime


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = runtime.PREREGISTRATION
V43I_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_multi_anchor_v43i.json"
).resolve()
V43I_PREREGISTRATION_FILE_SHA256 = (
    "00c545926b217a64acabbc541f3e92e071a1a199dbabef121383c788f574272e"
)
V43I_PREREGISTRATION_CONTENT_SHA256 = (
    "086d94f1b69732a9a0d7913c8bab7789b15f64131f125ba4381eea3bcc228c5a"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def _v43i_preregistration() -> dict:
    if runtime.v40a.file_sha256(V43I_PREREGISTRATION) != (
        V43I_PREREGISTRATION_FILE_SHA256
    ):
        raise RuntimeError("v43k V43I preregistration file changed")
    value = json.loads(V43I_PREREGISTRATION.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != V43I_PREREGISTRATION_CONTENT_SHA256
        or runtime.v40a.canonical_sha256(compact)
        != V43I_PREREGISTRATION_CONTENT_SHA256
        or value.get("sealed_holdout_opened") is not False
        or value.get("recipe", {}).get("seeds") != runtime.prior.SEEDS
        or value.get("recipe", {}).get("alpha") != runtime.prior.ALPHA
    ):
        raise RuntimeError("v43k V43I preregistration content changed")
    return value


def build_v43k() -> dict:
    parent = _v43i_preregistration()
    evidence = backtracking.load_v43i_evidence_v43k(
        runtime.V43I_EVIDENCE_PATHS
    )
    predecessor = runtime.v43j_untried_scale_evidence_v43k()
    if (
        evidence["restored_master_identity"]["sha256"]
        != backtracking.RESTORED_MASTER_SHA256_V43K
        or evidence["restored_runtime_values_sha256"]
        != backtracking.RESTORED_RUNTIME_SHA256_V43K
        or [item["target_norm_ratio"] for item in evidence["scale_plans"]]
        != list(backtracking.TARGET_NORM_RATIOS_V43K)
    ):
        raise RuntimeError("v43k exact restored state/scale derivation changed")
    value = {
        "schema": "matched-lora-es-backtracking-preregistration-v43k",
        "status": "preregistered_before_train_only_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "sealed_holdout_opened": False,
        "shadow_dev_eval_ood_or_holdout_authorized": False,
        "protected_semantic_access": False,
        "quality_selection_or_promotion_authorized": False,
        "current_v42i_holdout_cycle_eligible": False,
        "current_fixed_holdout_cycle_result_may_be_used_for_tuning": False,
        "purpose": (
            "Characterize the untried 0.125 V43I projection scale and open "
            "0.0625 only if 0.125 fails the same six train-only gates, without "
            "reopening, resampling, or using any protected evaluation result."
        ),
        "parent_v43i_preregistration": {
            "path": str(V43I_PREREGISTRATION),
            "file_sha256": V43I_PREREGISTRATION_FILE_SHA256,
            "content_sha256": parent["content_sha256_before_self_field"],
        },
        "v43i_evidence": evidence,
        "v43j_untried_scale_evidence": predecessor,
        "negative_evidence_interpretation": {
            "v43i_target_norm_ratio": 0.5,
            "v43i_domain_median_paired_delta": -0.00038767746417822657,
            "v43i_greedy_qa_f1_median_paired_delta": -0.0014460729512779102,
            "v43i_domain_gate_passed": False,
            "v43i_greedy_qa_f1_gate_passed": False,
            "v43i_other_preservation_gates_passed": True,
            "v43i_candidate_committed": False,
            "v43i_exact_abort_readback_passed": True,
        },
        "access_contract": {
            "runtime_train_semantic_paths": [
                str(runtime.prior.DATASET), str(runtime.prior.PROSE_ANCHOR),
                str(runtime.prior.QA_ANCHOR),
            ],
            "train_only_numeric_evidence_paths": [
                str(path) for path in runtime.V43I_EVIDENCE_PATHS.values()
            ],
            "direct_benchmark_source_opened": False,
            "protected_eval_ood_holdout_or_heldout_paths_opened": False,
            "current_fixed_holdout_cycle_report_opened_or_hashed": False,
            "current_fixed_holdout_cycle_result_bound": False,
            "current_fixed_holdout_cycle_result_influenced_design": False,
            "builder_reads_train_examples": False,
            "builder_reads_only_sealed_train_numeric_evidence": True,
            "dry_run_loads_model_or_train_examples": False,
            "dry_run_launches_gpu": False,
        },
        "recipe": {
            "model": str(runtime.v40a.MODEL),
            "dataset": str(runtime.prior.DATASET),
            "dataset_sha256": runtime.prior.DATASET_SHA256,
            "train_bundle_content_sha256": runtime.prior.TRAIN_BUNDLE_SHA256,
            "matched_initialization": str(runtime.prior.SOURCE),
            "staged_initialization": str(runtime.prior.STAGED),
            "worker_extension": runtime.prior.WORKER_EXTENSION,
            "population_size": runtime.prior.POPULATION_SIZE,
            "seeds": runtime.prior.SEEDS,
            "alpha": runtime.prior.ALPHA,
            "v43i_sigma_used_only_for_frozen_population_evidence": (
                runtime.prior.SIGMA
            ),
            "resample_population": False,
            "recompute_population_scores": False,
            "recompute_projection": False,
            "source_projection_coefficients": evidence[
                "v43i_projected_coefficients"
            ],
            "source_projection_content_sha256": evidence[
                "v43i_projection_content_sha256"
            ],
            "source_projection_norm_ratio": 0.5,
            "scale_order": list(backtracking.TARGET_NORM_RATIOS_V43K),
            "scale_plans": evidence["scale_plans"],
            "scale_policy": (
                "evaluate 0.125 first; evaluate 0.0625 only after an exact "
                "abort caused by failure of at least one of the six train-only "
                "gates; a post-gate consensus failure does not authorize 0.0625"
            ),
            "full_anchor_documents": runtime.fused.FULL_SIZE_V43I,
            "all_four_actors_score_every_candidate": True,
        },
        "candidate_gate": {
            "reference_state": "exact V43I restored matched initialization",
            "score_surfaces": [
                "specialist full train equal-conflict-unit answer logprob",
                "approved full prose-anchor selected-token logprob",
                "full general-QA proxy teacher-forced answer logprob",
                "full general-QA proxy greedy normalized-token F1",
                "full general-QA proxy greedy exact count",
                "full general-QA proxy greedy nonzero count",
            ],
            "domain_median_paired_actor_delta_strictly_positive": True,
            "preservation_noninferiority_uses_v43i_prefit_margins": True,
            "candidate_actor_consensus_required_before_commit": True,
            "score_candidate_while_uncommitted": True,
        },
        "transaction_protocol": {
            "every_scale_starts_from_exact_restored_master": True,
            "failed_scale_aborted_before_next_scale": True,
            "ratio_0p0625_requires_ratio_0p125_six_gate_failure": True,
            "ratio_0p125_consensus_failure_authorizes_ratio_0p0625": False,
            "abort_requires_all_four_exact_master_and_runtime_readback": True,
            "passing_scale_committed_only_after_every_gate": True,
            "accepted_rollback_retained_until_snapshot_readback": True,
            "no_passing_scale": "seal a no-update exact-restoration report",
        },
        "runtime": {
            "physical_gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
            "all_four_gpus_attributed_positive_activity_required": True,
            "tuned_folder": parent["runtime"]["tuned_folder"],
            "tuned_table_content_sha256": parent["runtime"][
                "tuned_table_content_sha256"
            ],
        },
        "required_gates": {
            "initial_master_equals_v43i_exact_restored_master": True,
            "initial_runtime_hash_equals_v43i_exact_restored_runtime": True,
            "exact_v43i_seeds_and_projection_coefficients_reused": True,
            "no_population_or_projection_recomputation": True,
            "scales_evaluated_in_descending_preregistered_order": True,
            "failed_scale_exactly_aborted_before_continuing": True,
            "domain_point_improvement_required": True,
            "every_prose_qa_and_greedy_preservation_gate_required": True,
            "six_gate_continuation_policy_required": True,
            "rank_zero_snapshot_exact_readback_if_accepted": True,
            "final_all_gpu_idle_cleanup": True,
            "current_fixed_holdout_cycle_eligibility": False,
        },
        "implementation_bindings": runtime.implementation_bindings_v43k(),
        "artifacts": {
            "attempt": str(runtime.ATTEMPT),
            "run_directory": str(runtime.RUN_DIR),
            "per_scale": {
                key: str(path) for key, path in runtime.SCALE_ARTIFACTS.items()
            },
            "snapshot": str(runtime.SNAPSHOT),
            "gpu_log": str(runtime.GPU_LOG),
            "report": str(runtime.REPORT),
            "failure": str(runtime.FAILURE),
        },
    }
    return runtime.v40a.self_hashed(value)


def main(argv: list[str] | None = None) -> int:
    output = Path(parser().parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v43k()
    runtime.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launch_authorized": True,
        "protected_semantics_opened": False,
        "sealed_holdout_opened": False,
        "current_v42i_holdout_cycle_eligible": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
