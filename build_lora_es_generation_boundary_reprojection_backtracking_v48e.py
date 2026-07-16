#!/usr/bin/env python3
"""Seal launchable V48E from exact train-only V48B evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_generation_boundary_reprojection_v48e as planning
import run_lora_es_generation_boundary_reprojection_backtracking_v48e as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = runtime.PREREGISTRATION


def build_v48e() -> dict:
    evidence = runtime.load_evidence_v48e()
    if (
        evidence["restored_master_identity"]["sha256"]
        != planning.RESTORED_MASTER_SHA256_V48E
        or evidence["restored_runtime_values_sha256"]
        != planning.RESTORED_RUNTIME_SHA256_V48E
        or [item["target_norm_ratio"] for item in evidence["scale_plans"]]
        != list(planning.TARGET_NORM_RATIOS_V48E)
    ):
        raise RuntimeError("v48e exact source state or scale plans changed")
    value = {
        "schema": (
            "matched-lora-es-generation-boundary-reprojection-backtracking-"
            "preregistration-v48e"
        ),
        "status": "preregistered_before_train_only_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "purpose": (
            "CPU-reproject the frozen V48B signed population with existing "
            "panel QA-generation F1 as a fifth halfspace, then commit only "
            "the largest descending ratio that passes every train-only gate."
        ),
        "protected_semantic_access_authorized": False,
        "shadow_ood_holdout_or_benchmark_authorized": False,
        "quality_selection_or_promotion_authorized": False,
        "v48b_evidence": evidence,
        "access_contract": {
            "runtime_train_semantic_paths": [
                str(runtime.v48b.evidence_runtime.TRAIN_DATASET),
                str(runtime.v48b.evidence_runtime.MEMBERSHIP),
                str(runtime.SUBSET),
                str(runtime.prior.PROSE_ANCHOR),
                str(runtime.prior.QA_ANCHOR),
            ],
            "train_only_numeric_evidence_paths": [
                str(path) for path in runtime.V48B_EVIDENCE_PATHS.values()
            ],
            "v48b_population_reopened_as_numeric_evidence_only": True,
            "base_generation_evidence_reopened": False,
            "protected_eval_ood_holdout_or_benchmark_path_opened": False,
            "builder_reads_train_semantics": False,
            "dry_run_reads_train_semantics": False,
            "dry_run_launches_model_or_gpu": False,
        },
        "recipe": {
            "model": str(runtime.v40a.MODEL),
            "dataset": str(runtime.v48b.evidence_runtime.TRAIN_DATASET),
            "dataset_sha256": runtime.v48b.evidence_runtime.EXPECTED_TRAIN_SHA256,
            "train_bundle_content_sha256": (
                runtime.v48b.EXPECTED_TRAIN_BUNDLE_CONTENT_SHA256_V48B
            ),
            "membership": str(runtime.v48b.evidence_runtime.MEMBERSHIP),
            "subset": str(runtime.SUBSET),
            "subset_file_sha256": planning.EXPECTED_SUBSET_FILE_SHA256_V48E,
            "subset_content_sha256": (
                planning.EXPECTED_SUBSET_CONTENT_SHA256_V48E
            ),
            "request_order_sha256": planning.REQUEST_ORDER_SHA256_V48E,
            "matched_initialization": str(runtime.prior.SOURCE),
            "staged_initialization": str(runtime.prior.STAGED),
            "restored_master_sha256": planning.RESTORED_MASTER_SHA256_V48E,
            "restored_runtime_values_sha256": (
                planning.RESTORED_RUNTIME_SHA256_V48E
            ),
            "worker_extension": runtime.prior.WORKER_EXTENSION,
            "population_size": runtime.prior.POPULATION_SIZE,
            "seeds": runtime.prior.SEEDS,
            "alpha": runtime.prior.ALPHA,
            "sigma_used_only_by_frozen_v48b_population": runtime.prior.SIGMA,
            "resample_population": False,
            "recompute_population_scores": False,
            "recompute_projection_from_frozen_signed_scores": True,
            "new_population_generation": False,
            "fifth_halfspace": "qa_generation_f1",
            "signed_score_inventory_sha256": (
                planning.EXPECTED_FULL_SIGNED_SCORE_SHA256_V48E
            ),
            "source_projection_coefficients": evidence[
                "reprojection"
            ]["coefficients"],
            "source_projection_content_sha256": evidence[
                "v48e_projection_content_sha256"
            ],
            "source_projection_norm_ratio": 0.5,
            "geometry_vs_v48b": evidence[
                "reprojection"
            ]["geometry_vs_v48b"],
            "scale_order": list(planning.TARGET_NORM_RATIOS_V48E),
            "scale_plans": evidence["scale_plans"],
            "scale_policy": (
                "descending target norm ratios; stop and commit at the first "
                "strict all-gate pass; exact-abort every rejection first"
            ),
            "full_candidate_requests_per_actor": 896,
            "all_four_actors_score_every_candidate": True,
        },
        "candidate_gate": {
            "implementation": "V48B generation-boundary candidate gate",
            "required_checks": [
                "domain_point_improvement",
                "prose_lm_noninferiority",
                "qa_logprob_noninferiority",
                "qa_generation_f1_noninferiority",
                "qa_generation_exact_noninferiority",
                "qa_generation_nonzero_noninferiority",
                "fragile_generation_f1_noninferiority",
                "fragile_generation_exact_noninferiority",
                "fragile_generation_nonzero_noninferiority",
            ],
            "fragile_subset_request_order_sha256": (
                planning.REQUEST_ORDER_SHA256_V48E
            ),
            "all_checks_strictly_required": True,
            "candidate_actor_consensus_required_before_commit": True,
        },
        "transaction_protocol": {
            "every_scale_starts_from_exact_restored_master": True,
            "failed_scale_aborted_before_next_scale": True,
            "abort_requires_all_four_exact_master_and_runtime_readback": True,
            "passing_scale_committed_only_after_all_nine_gates_and_consensus": True,
            "accepted_rollback_retained_until_snapshot_readback": True,
            "no_passing_scale": "seal no-update with all candidates restored",
        },
        "runtime": {
            "physical_gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
            "all_four_gpus_attributed_positive_activity_required": True,
            "tuned_table_content_sha256": (
                "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
            ),
        },
        "required_gates": {
            "initial_state_equals_v48b_exact_abort_readback": True,
            "frozen_v48b_signed_population_reused": True,
            "deterministic_five_halfspace_cpu_reprojection_bound": True,
            "no_new_population_generation": True,
            "sealed_subset_and_request_order_reused_at_every_scale": True,
            "all_nine_generation_boundary_domain_prose_qa_checks_pass": True,
            "largest_strictly_passing_ratio_only": True,
            "every_rejection_exactly_aborted_before_continuing": True,
            "rank_zero_snapshot_exact_readback_if_accepted": True,
            "strict_four_gpu_cleanup_and_idle": True,
        },
        "implementation_bindings": runtime.implementation_bindings_v48e(),
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
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    return runtime.v40a.self_hashed(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v48e()
    runtime.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launch_authorized": True,
        "population_resampled": False,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
