#!/usr/bin/env python3
"""Seal the V55B strict-interior fragile-protective train-only experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_fragile_maximin_projection_v55b as maximin
import lora_es_nested_population_v52 as design


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments/eggroll_es_hpo/preregistrations/lora_es_fragile_maximin_backtracking_v55b.json"
RUN_DIR = ROOT / "experiments/eggroll_es_hpo/runs/v55b_lora_es_fragile_maximin_backtracking"
V54_DIR = ROOT / "experiments/eggroll_es_hpo/runs/v54_lora_es_selected_score_backtracking"
V54_GATE = V54_DIR / "p16_train_gate_v52.json"
V54_REPORT = V54_DIR / "nested_population_report_v52.json"
V54_EVIDENCE = V54_DIR / "v54_evidence_manifest.json"
V54_TELEMETRY = V54_DIR / "gpu_activity_v54.jsonl"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha(path: Path) -> str | None:
    return json.loads(path.read_text(encoding="utf-8")).get(
        "content_sha256_before_self_field"
    )


def binding(name: str) -> dict:
    path = ROOT / name
    return {"path": str(path), "file_sha256": file_sha256(path)}


def artifacts_v55b() -> dict:
    return {
        "attempt": str(RUN_DIR.parent / ".v55b_lora_es_fragile_maximin_backtracking.attempt.json"),
        "run_directory": str(RUN_DIR),
        "projection": str(RUN_DIR / "nested_population_v52.json"),
        "no_p8_evaluation_receipt": str(RUN_DIR / "p8_train_gate_v52.json"),
        "p16_train_gate": str(RUN_DIR / "p16_train_gate_v52.json"),
        "passing_candidate_snapshot": str(RUN_DIR / "selected_candidate_v55b"),
        "internal_report": str(RUN_DIR / "nested_population_report_v52.json"),
        "report": str(RUN_DIR / "fragile_maximin_backtracking_report_v55b.json"),
        "failure": str(RUN_DIR / "failure_v52.json"),
        "numeric_calibration": str(RUN_DIR / "numeric_calibration_v55b.json"),
        "anchor_calibration": str(RUN_DIR / "anchor_calibration_v55b.json"),
        "gpu_log": str(RUN_DIR / "gpu_activity_v55b.jsonl"),
        "coordinator_log": str(RUN_DIR.parent / "v55b_lora_es_fragile_maximin_backtracking.coordinator.log"),
        "sealed_holdout_artifact": None,
    }


def build_v55b() -> dict:
    projection = maximin.maximin_projection_v55b()
    plans = maximin.scale_plans_v55b(projection)
    gate = json.loads(V54_GATE.read_text(encoding="utf-8"))
    report = json.loads(V54_REPORT.read_text(encoding="utf-8"))
    evidence = json.loads(V54_EVIDENCE.read_text(encoding="utf-8"))
    results = gate.get("scale_results", [])
    domain_pattern = [
        item.get("checks", {}).get("domain_point_improvement")
        for item in results
    ]
    always_true = set(design.TRAIN_GATE_NAMES_V52) - {
        "domain_point_improvement",
        "fragile_generation_f1_noninferiority",
        "fragile_generation_nonzero_noninferiority",
    }
    if (
        gate.get("selected_target_norm_ratio") is not None
        or len(results) != 6
        or any(item.get("passed") for item in results)
        or not all(
            item["checks"]["fragile_generation_f1_noninferiority"]
            is False
            and item["checks"][
                "fragile_generation_nonzero_noninferiority"
            ] is False
            for item in results
        )
        or domain_pattern != [True, True, False, False, False, True]
        or not all(
            set(item.get("checks", {})) == set(design.TRAIN_GATE_NAMES_V52)
            and all(item["checks"][name] is True for name in always_true)
            and item.get("exact_abort_readback_passed") is True
            and item.get("snapshot") is None
            for item in results
        )
        or gate.get("all_candidates_exactly_aborted_to_common_master")
        is not True
        or gate.get("master_committed") is not False
        or gate.get("protected_semantics_opened") is not False
        or gate.get("sealed_holdout_opened") is not False
        or report.get("optimizer_master_committed") is not False
        or report.get("all_candidates_exactly_aborted_to_common_master")
        is not True
        or report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or report.get("protected_semantics_opened") is not False
        or report.get("sealed_holdout_opened") is not False
        or evidence.get("state_and_cleanup", {}).get(
            "four_actor_final_state_and_quiescence_exact"
        ) is not True
        or evidence.get("postcleanup_wrapper_issue", {}).get(
            "science_success_claimed"
        ) is not False
    ):
        raise RuntimeError("v55b V54 failure diagnosis changed")
    result = {
        "schema": "lora-es-fragile-maximin-backtracking-preregistration-v55b",
        "status": "sealed_before_v55b_model_ray_gpu_or_train_gate_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Change only the persisted-score projection direction from the "
            "V54 boundary solution to a six-objective strict-interior maximin "
            "direction that explicitly protects fragile F1 and nonzero rate."
        ),
        "v54_failure_evidence": {
            "p16_train_gate": {"path": str(V54_GATE), "file_sha256": file_sha256(V54_GATE), "content_sha256": content_sha(V54_GATE)},
            "internal_report": {"path": str(V54_REPORT), "file_sha256": file_sha256(V54_REPORT), "content_sha256": content_sha(V54_REPORT)},
            "evidence_manifest": {"path": str(V54_EVIDENCE), "file_sha256": file_sha256(V54_EVIDENCE)},
            "telemetry": {"path": str(V54_TELEMETRY), "file_sha256": file_sha256(V54_TELEMETRY)},
            "no_passing_candidate": True,
            "failed_only_fragile_f1_and_nonzero_at_all_scales_except_domain_at_three_scales": True,
            "postcleanup_wrapper_assertion_is_not_science_success": True,
            "domain_point_improvement_pattern": domain_pattern,
            "all_other_six_endpoint_gates_true_at_all_scales": True,
            "all_six_candidate_transactions_exactly_aborted": True,
            "final_exact_master_quiescence_and_cleanup_passed": True,
        },
        "selected_v53_input": {
            "path": str(maximin.V53_SELECTED_ARM),
            "file_sha256": maximin.V53_SELECTED_ARM_FILE_SHA256,
            "content_sha256": maximin.V53_SELECTED_ARM_CONTENT_SHA256,
            "sigma": 0.0048,
            "population_size": 16,
            "p16_seeds": list(design.P16_SEEDS_V52),
            "optimizer_update_alpha": design.ALPHA_V52,
            "population_rerun_authorized": False,
        },
        "maximin_projection": projection,
        "maximin_projection_content_sha256": projection["content_sha256"],
        "scale_plans": plans,
        "scale_plans_content_sha256": design.canonical_sha256_v52(plans),
        "single_scientific_change": {
            "variable": "projection_direction_only",
            "v54_direction_cosine": projection["direction_cosine_vs_v54"],
            "all_six_v55b_normalized_objective_margins": projection[
                "objective_margins"
            ],
            "strict_equal_maximin_margin": projection["maximin_margin"],
            "same_signed_scores": True,
            "same_sigma_seeds_actor_reducer_alpha_scale_order": True,
            "every_scale_coefficient_l2_norm_exactly_matches_v54": True,
            "per_ratio_v55b_and_v54_coefficient_l2_norm_receipts": [{
                "target_norm_ratio": item["target_norm_ratio"],
                "v55b_coefficient_l2_norm": item["coefficient_l2_norm"],
                "v54_coefficient_l2_norm": item[
                    "v54_reference_coefficient_l2_norm"
                ],
                "v54_coefficient_sha256": item[
                    "v54_reference_coefficient_sha256"
                ],
                "exact_match": item[
                    "coefficient_l2_norm_exactly_matches_v54_same_ratio"
                ],
            } for item in plans],
            "same_nine_endpoint_gates": True,
            "same_exact_master_numeric_anchor_and_consensus_gates": True,
        },
        "spread_contract": {
            "projection_objectives": list(maximin.OBJECTIVE_PATHS_V55B),
            "zero_spread_endpoint_only_metrics": list(
                maximin.ZERO_SPREAD_ENDPOINT_PATHS_V55B
            ),
            "zero_spread_metrics_are_not_fabricated_projection_constraints": True,
            "zero_spread_metrics_remain_required_endpoint_gates": True,
        },
        "train_only_gate": {
            "scale_order": list(maximin.SCALE_ORDER_V55B),
            "required_checks": list(design.TRAIN_GATE_NAMES_V52),
            "all_nine_checks_required_without_weakening": True,
            "fresh_numeric_and_anchor_calibration_against_exact_master": True,
            "candidate_actor_consensus_required": True,
            "save_pending_candidate_only_for_passing_scale": True,
            "every_candidate_exactly_aborted_to_master": True,
            "largest_strictly_passing_scale": True,
        },
        "authorization": {
            "gpu_launch": True,
            "reuse_persisted_signed_scores": True,
            "population_generation_or_scoring": False,
            "in_memory_candidate_transaction": True,
            "optimizer_master_commit": False,
            "p8_evaluation": False,
            "ood_shadow_benchmark_or_holdout": False,
            "protected_semantics": False,
            "sealed_holdout": False,
        },
        "fixed_recipe": {
            "model": "/home/catid/specialist/models/Qwen3.6-35B-A3B",
            "source": str(design.SOURCE_V52),
            "staged": str(design.STAGED_V52),
            "master_sha256": design.MASTER_SHA256_V52,
            "master_runtime_values_sha256": design.MASTER_RUNTIME_SHA256_V52,
            "dataset": str(design.TRAIN_DATASET_V52),
            "dataset_sha256": design.DATASET_SHA256_V52,
            "train_bundle_content_sha256": design.TRAIN_BUNDLE_CONTENT_SHA256_V52,
            "membership_file_sha256": design.MEMBERSHIP_SHA256_V52,
            "generation_panel_file_sha256": design.SUBSET_FILE_SHA256_V52,
            "request_order_sha256": design.REQUEST_ORDER_SHA256_V52,
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
            "physical_gpu_ids": [0, 1, 2, 3],
            "worker_extension": "eggroll_es_worker_lora_v52.LoRAAdapterStateWorkerExtensionV52",
        },
        "runtime": dict(design.RUNTIME_V52),
        "telemetry": {
            "gpu_log": str(RUN_DIR / "gpu_activity_v55b.jsonl"),
            "projection_phase": "v55b_strict_interior_maximin_projection_no_population_rerun",
            "minimum_projection_phase_barrier_seconds": 1.25,
            "inherited_v52_filename_or_population_phase_allowed": False,
        },
        "artifacts": artifacts_v55b(),
        "implementation_bindings": {
            "builder": binding("build_lora_es_fragile_maximin_backtracking_preregistration_v55b.py"),
            "runner": binding("run_lora_es_fragile_maximin_backtracking_v55b.py"),
            "maximin": binding("lora_es_fragile_maximin_projection_v55b.py"),
            "v52_design": binding("lora_es_nested_population_v52.py"),
            "v52_executor": binding("run_lora_es_nested_population_v52.py"),
            "v52_worker": binding("eggroll_es_worker_lora_v52.py"),
        },
        "protected_semantics_opened": False,
        "sealed_holdout_opened": False,
    }
    result["content_sha256_before_self_field"] = design.canonical_sha256_v52(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    value = build_v55b()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"path": str(args.output), "file_sha256": file_sha256(args.output), "content_sha256": value["content_sha256_before_self_field"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
