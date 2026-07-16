#!/usr/bin/env python3
"""Seal V58 actor-aware projection with original calibrated train gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_actor_maximin_projection_v58 as projection58
import lora_es_nested_population_v52 as design52


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments/eggroll_es_hpo/preregistrations/lora_es_actor_maximin_v58.json"
RUN_DIR = ROOT / "experiments/eggroll_es_hpo/runs/v58_lora_es_actor_maximin"
V57_EVIDENCE = ROOT / "experiments/eggroll_es_hpo/runs/v57_lora_es_conservative_actor_robust/v57_evidence_manifest.json"
V57_REPORT = ROOT / "experiments/eggroll_es_hpo/runs/v57_lora_es_conservative_actor_robust/conservative_actor_robust_report_v57.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(name: str) -> dict:
    path = (ROOT / name).resolve()
    return {"path": str(path), "file_sha256": file_sha256(path)}


def artifacts_v58() -> dict:
    return {
        "attempt": str(RUN_DIR.parent / ".v58_lora_es_actor_maximin.attempt.json"),
        "run_directory": str(RUN_DIR),
        "projection": str(RUN_DIR / "nested_population_v52.json"),
        "no_p8_evaluation_receipt": str(RUN_DIR / "p8_train_gate_v52.json"),
        "p16_train_gate": str(RUN_DIR / "p16_train_gate_v52.json"),
        "passing_candidate_snapshot": str(RUN_DIR / "selected_candidate_v58"),
        "internal_report": str(RUN_DIR / "nested_population_report_v52.json"),
        "compatibility_report": str(RUN_DIR / "compatibility_report_v55b.json"),
        "report": str(RUN_DIR / "actor_maximin_report_v58.json"),
        "failure": str(RUN_DIR / "failure_v52.json"),
        "numeric_calibration": str(RUN_DIR / "numeric_calibration_v58.json"),
        "anchor_calibration": str(RUN_DIR / "anchor_calibration_v58.json"),
        "gpu_log": str(RUN_DIR / "gpu_activity_v58.jsonl"),
        "coordinator_log": str(RUN_DIR.parent / "v58_lora_es_actor_maximin.coordinator.log"),
        "ood_shadow_benchmark_or_holdout": None,
    }


def build_v58() -> dict:
    projection = projection58.actor_maximin_projection_v58()
    plans = projection58.scale_plans_v58(projection)
    evidence = json.loads(V57_EVIDENCE.read_text(encoding="utf-8"))
    if (
        file_sha256(V57_EVIDENCE)
        != "6a3e23dc62647afbd63b80ea05342c693d039c02fb92afe650b1065af8605cf8"
        or file_sha256(V57_REPORT)
        != "e1683345646e76193bc709ebca487fd3b6fd1915246697bf4c0ddc07966b89ee"
        or evidence.get("selected_target_norm_ratio") is not None
        or evidence.get("all_six_candidates_exactly_aborted_to_master") is not True
        or evidence.get("train_finding", {}).get(
            "raw_all_actor_sign_rule_is_not_recommended_for_reuse"
        ) is not True
        or evidence.get("protected_semantics_opened") is not False
        or evidence.get("ood_shadow_opened") is not False
        or evidence.get("terminal_holdout_opened") is not False
    ):
        raise RuntimeError("v58 V57 train-only diagnosis changed")
    result = {
        "schema": "lora-es-actor-maximin-preregistration-v58",
        "status": "sealed_before_v58_model_ray_gpu_or_train_gate_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Replace only the V55B aggregate projection direction with a "
            "24-row actor-aware maximin direction computed from the same "
            "persisted V53 signed scores; restore and retain the original "
            "nine calibrated endpoint gates and candidate consensus."
        ),
        "v57_train_only_negative_evidence": {
            "manifest": {
                "path": str(V57_EVIDENCE),
                "file_sha256": file_sha256(V57_EVIDENCE),
            },
            "aggregate_report": {
                "path": str(V57_REPORT),
                "file_sha256": file_sha256(V57_REPORT),
            },
            "all_six_same_direction_scales_rejected": True,
            "raw_all_actor_sign_rule_not_reused": True,
            "ood_shadow_benchmark_holdout_or_protected_semantics_used": False,
        },
        "actor_maximin_projection": projection,
        "actor_maximin_projection_content_sha256": projection["content_sha256"],
        "scale_plans": plans,
        "scale_plans_content_sha256": design52.canonical_sha256_v52(plans),
        "single_scientific_change": {
            "variable": "projection_direction_only",
            "old_direction": "V55B six aggregate objective maximin",
            "new_direction": "V58 six objectives by four actors maximin",
            "same_exact_master_dataset_sigma_population_seeds_alpha": True,
            "same_persisted_signed_scores": True,
            "same_original_nine_calibrated_endpoint_gates": True,
            "same_candidate_consensus": True,
            "no_raw_all_actor_sign_selector": True,
            "no_ood_output_or_semantic_informed_projection_scale_or_gate": True,
        },
        "train_only_gate": {
            "scale_order": list(projection58.SCALE_ORDER_V58),
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
            "source": str(design52.SOURCE_V52),
            "staged": str(design52.STAGED_V52),
            "master_sha256": design52.MASTER_SHA256_V52,
            "master_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
            "dataset": str(design52.TRAIN_DATASET_V52),
            "dataset_sha256": design52.DATASET_SHA256_V52,
            "train_bundle_content_sha256": design52.TRAIN_BUNDLE_CONTENT_SHA256_V52,
            "engine_count": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "tensor_parallel_size_per_engine": 1,
            "worker_extension": "eggroll_es_worker_lora_v52.LoRAAdapterStateWorkerExtensionV52",
        },
        "runtime": design52.RUNTIME_V52,
        "telemetry": {
            "gpu_log": str(RUN_DIR / "gpu_activity_v58.jsonl"),
            "projection_phase": "v58_actor_maximin_projection_no_population_rerun",
            "minimum_projection_phase_barrier_seconds": 1.25,
            "all_four_physical_gpus_and_expected_actor_pids_required": True,
            "p16_train_gate_activity_receipt_required": True,
            "foreign_compute_processes_forbidden": True,
        },
        "artifacts": artifacts_v58(),
        "implementation_bindings": {
            "projection_v58": binding("lora_es_actor_maximin_projection_v58.py"),
            "runner_v58": binding("run_lora_es_actor_maximin_v58.py"),
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
    value = build_v58()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "path": str(args.output.resolve()),
        "file_sha256": file_sha256(args.output),
        "content_sha256": value["content_sha256_before_self_field"],
        "direction_sha256": value["actor_maximin_projection"]["direction_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
