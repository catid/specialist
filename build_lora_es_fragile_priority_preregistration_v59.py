#!/usr/bin/env python3
"""Seal V59 fragile-priority actor projection train-only HPO."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_fragile_priority_projection_v59 as projection59
import lora_es_nested_population_v52 as design52


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments/eggroll_es_hpo/preregistrations/lora_es_fragile_priority_v59.json"
RUN_DIR = ROOT / "experiments/eggroll_es_hpo/runs/v59_lora_es_fragile_priority"
V58_EVIDENCE = ROOT / "experiments/eggroll_es_hpo/runs/v58_lora_es_actor_maximin/v58_evidence_manifest.json"
V58_REPORT = ROOT / "experiments/eggroll_es_hpo/runs/v58_lora_es_actor_maximin/actor_maximin_report_v58.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(name: str) -> dict:
    path = (ROOT / name).resolve()
    return {"path": str(path), "file_sha256": file_sha256(path)}


def artifacts_v59() -> dict:
    return {
        "attempt": str(RUN_DIR.parent / ".v59_lora_es_fragile_priority.attempt.json"),
        "run_directory": str(RUN_DIR),
        "projection": str(RUN_DIR / "nested_population_v52.json"),
        "no_p8_evaluation_receipt": str(RUN_DIR / "p8_train_gate_v52.json"),
        "p16_train_gate": str(RUN_DIR / "p16_train_gate_v52.json"),
        "passing_candidate_snapshot": str(RUN_DIR / "selected_candidate_v59"),
        "internal_report": str(RUN_DIR / "nested_population_report_v52.json"),
        "compatibility_report": str(RUN_DIR / "compatibility_report_v55b.json"),
        "report": str(RUN_DIR / "fragile_priority_report_v59.json"),
        "failure": str(RUN_DIR / "failure_v52.json"),
        "numeric_calibration": str(RUN_DIR / "numeric_calibration_v59.json"),
        "anchor_calibration": str(RUN_DIR / "anchor_calibration_v59.json"),
        "gpu_log": str(RUN_DIR / "gpu_activity_v59.jsonl"),
        "coordinator_log": str(RUN_DIR.parent / "v59_lora_es_fragile_priority.coordinator.log"),
        "ood_shadow_benchmark_or_holdout": None,
    }


def build_v59() -> dict:
    projection = projection59.fragile_priority_projection_v59()
    plans = projection59.scale_plans_v59(projection)
    evidence = json.loads(V58_EVIDENCE.read_text(encoding="utf-8"))
    if (
        file_sha256(V58_EVIDENCE)
        != "3dc7b6bdafa36cbfa7fe5c2d584ceff4d5ada857df4e59ce9c6ef8e5afc905ff"
        or file_sha256(V58_REPORT)
        != "2b7e541e0a806fea41e6d91dc718d89c42cd2920a06499e78c8c3e3001f09436"
        or evidence.get("selected_target_norm_ratio") is not None
        or evidence.get("train_finding", {}).get(
            "fragile_generation_f1_failed_at_every_ratio"
        ) is not True
        or evidence.get("train_finding", {}).get(
            "ratios_0p375_0p25_0p1875_failed_only_fragile_generation_f1"
        ) is not True
        or evidence.get("all_six_candidates_exactly_aborted_to_master") is not True
        or evidence.get("protected_semantics_opened") is not False
        or evidence.get("ood_shadow_opened") is not False
        or evidence.get("terminal_holdout_opened") is not False
    ):
        raise RuntimeError("v59 V58 train-only diagnosis changed")
    result = {
        "schema": "lora-es-fragile-priority-preregistration-v59",
        "status": "sealed_before_v59_model_ray_gpu_or_train_gate_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Target V58's sole repeat bottleneck by maximizing the four "
            "fragile-generation-F1 actor margins while requiring every other "
            "actor-objective margin to remain at least 0.15."
        ),
        "v58_train_only_negative_evidence": {
            "manifest": {"path": str(V58_EVIDENCE), "file_sha256": file_sha256(V58_EVIDENCE)},
            "aggregate_report": {"path": str(V58_REPORT), "file_sha256": file_sha256(V58_REPORT)},
            "fragile_f1_failed_all_six_ratios": True,
            "middle_three_ratios_failed_only_fragile_f1": True,
            "ood_shadow_benchmark_holdout_or_protected_semantics_used": False,
        },
        "fragile_priority_projection": projection,
        "fragile_priority_projection_content_sha256": projection["content_sha256"],
        "scale_plans": plans,
        "scale_plans_content_sha256": design52.canonical_sha256_v52(plans),
        "single_scientific_change": {
            "variable": "projection_direction_only",
            "old_direction": "V58 equal maximin over 24 actor-objective rows",
            "new_direction": "fragile-F1 priority with fixed nonfragile floor",
            "nonfragile_floor": projection59.NONFRAGILE_MARGIN_FLOOR_V59,
            "nonfragile_floor_preregistered_derivation": "0.5 * V55B_MINIMUM_MAXIMIN_MARGIN_0p3",
            "same_exact_master_dataset_sigma_population_seeds_alpha": True,
            "same_persisted_signed_scores": True,
            "same_original_nine_calibrated_endpoint_gates": True,
            "same_candidate_consensus": True,
            "no_raw_all_actor_sign_selector": True,
            "no_ood_output_or_semantic_informed_projection_scale_or_gate": True,
        },
        "train_only_gate": {
            "scale_order": list(projection59.SCALE_ORDER_V59),
            "required_checks": list(design52.TRAIN_GATE_NAMES_V52),
            "all_nine_original_calibrated_checks_required_without_weakening": True,
            "additional_raw_actor_sign_checks": False,
            "candidate_actor_consensus_required": True,
            "largest_strictly_passing_scale": True,
            "every_candidate_exactly_aborted_to_master": True,
            "save_pending_candidate_only_for_passing_scale": True,
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
            "source": str(design52.SOURCE_V52), "staged": str(design52.STAGED_V52),
            "master_sha256": design52.MASTER_SHA256_V52,
            "master_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
            "dataset": str(design52.TRAIN_DATASET_V52),
            "dataset_sha256": design52.DATASET_SHA256_V52,
            "train_bundle_content_sha256": design52.TRAIN_BUNDLE_CONTENT_SHA256_V52,
            "engine_count": 4, "physical_gpu_ids": [0, 1, 2, 3],
            "tensor_parallel_size_per_engine": 1,
            "worker_extension": "eggroll_es_worker_lora_v52.LoRAAdapterStateWorkerExtensionV52",
        },
        "runtime": design52.RUNTIME_V52,
        "telemetry": {
            "gpu_log": str(RUN_DIR / "gpu_activity_v59.jsonl"),
            "projection_phase": "v59_fragile_priority_projection_no_population_rerun",
            "minimum_projection_phase_barrier_seconds": 1.25,
            "all_four_physical_gpus_and_expected_actor_pids_required": True,
            "p16_train_gate_activity_receipt_required": True,
            "foreign_compute_processes_forbidden": True,
        },
        "artifacts": artifacts_v59(),
        "implementation_bindings": {
            "projection_v59": binding("lora_es_fragile_priority_projection_v59.py"),
            "runner_v59": binding("run_lora_es_fragile_priority_v59.py"),
            "v58_shell": binding("run_lora_es_actor_maximin_v58.py"),
            "compatibility_runner_v55b": binding("run_lora_es_fragile_maximin_backtracking_v55b.py"),
            "v52_design": binding("lora_es_nested_population_v52.py"),
            "v52_executor": binding("run_lora_es_nested_population_v52.py"),
            "v52_worker": binding("eggroll_es_worker_lora_v52.py"),
        },
        "document_disjoint_evaluation_doctrine_preserved": True,
        "protected_semantics_opened": False,
        "ood_shadow_opened": False,
        "terminal_holdout_opened": False,
    }
    result["content_sha256_before_self_field"] = design52.canonical_sha256_v52(result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    value = build_v59()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "path": str(args.output.resolve()), "file_sha256": file_sha256(args.output),
        "content_sha256": value["content_sha256_before_self_field"],
        "direction_sha256": value["fragile_priority_projection"]["direction_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
