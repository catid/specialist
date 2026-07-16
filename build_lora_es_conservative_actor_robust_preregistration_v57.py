#!/usr/bin/env python3
"""Seal the V57 conservative actor-robust train-only experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_conservative_actor_robust_v57 as design57
import lora_es_fragile_maximin_projection_v55b as maximin
import lora_es_nested_population_v52 as design52


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments/eggroll_es_hpo/preregistrations/lora_es_conservative_actor_robust_v57.json"
RUN_DIR = ROOT / "experiments/eggroll_es_hpo/runs/v57_lora_es_conservative_actor_robust"
V55_GATE = ROOT / "experiments/eggroll_es_hpo/runs/v55b_lora_es_fragile_maximin_backtracking/p16_train_gate_v52.json"
V55_REPORT = ROOT / "experiments/eggroll_es_hpo/runs/v55b_lora_es_fragile_maximin_backtracking/fragile_maximin_backtracking_report_v55b.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding(name: str) -> dict:
    path = (ROOT / name).resolve()
    return {"path": str(path), "file_sha256": file_sha256(path)}


def artifacts_v57() -> dict:
    return {
        "attempt": str(RUN_DIR.parent / ".v57_lora_es_conservative_actor_robust.attempt.json"),
        "run_directory": str(RUN_DIR),
        "projection": str(RUN_DIR / "nested_population_v52.json"),
        "no_p8_evaluation_receipt": str(RUN_DIR / "p8_train_gate_v52.json"),
        "p16_train_gate": str(RUN_DIR / "p16_train_gate_v52.json"),
        "passing_candidate_snapshot": str(RUN_DIR / "selected_candidate_v57"),
        "internal_report": str(RUN_DIR / "nested_population_report_v52.json"),
        "compatibility_report": str(RUN_DIR / "compatibility_report_v55b.json"),
        "report": str(RUN_DIR / "conservative_actor_robust_report_v57.json"),
        "failure": str(RUN_DIR / "failure_v52.json"),
        "numeric_calibration": str(RUN_DIR / "numeric_calibration_v57.json"),
        "anchor_calibration": str(RUN_DIR / "anchor_calibration_v57.json"),
        "gpu_log": str(RUN_DIR / "gpu_activity_v57.jsonl"),
        "coordinator_log": str(RUN_DIR.parent / "v57_lora_es_conservative_actor_robust.coordinator.log"),
        "ood_shadow_benchmark_or_holdout": None,
    }


def _v55_train_diagnosis() -> dict:
    gate = json.loads(V55_GATE.read_text(encoding="utf-8"))
    results = gate.get("scale_results", [])
    if (
        [item.get("target_norm_ratio") for item in results] != [0.5, 0.25]
        or results[0].get("passed") is not False
        or results[1].get("passed") is not True
        or gate.get("selected_target_norm_ratio") != 0.25
        or gate.get("protected_semantics_opened") is not False
        or gate.get("sealed_holdout_opened") is not False
    ):
        raise RuntimeError("v57 V55B train-only diagnosis changed")
    metrics = results[1]["gate"]["metrics"]
    return {
        "p16_train_gate": {
            "path": str(V55_GATE), "file_sha256": file_sha256(V55_GATE),
            "content_sha256": gate["content_sha256_before_self_field"],
        },
        "train_only_report": {
            "path": str(V55_REPORT), "file_sha256": file_sha256(V55_REPORT),
        },
        "evaluated_ratios": [0.5, 0.25],
        "first_strict_pass_ratio": 0.25,
        "v55b_ratio_0p25_paired_actor_signs": {
            name: [value >= 0.0 for value in metrics[name]["paired_actor_deltas"]]
            for name in (
                "domain", "prose_lm", "qa_answer_logprob",
                "qa_generation_f1", "qa_generation_exact",
                "qa_generation_nonzero",
            )
        },
        "qa_answer_logprob_not_all_four_nonnegative": not all(
            value >= 0.0 for value in metrics["qa_answer_logprob"]["paired_actor_deltas"]
        ),
        "qa_generation_f1_not_all_four_nonnegative": not all(
            value >= 0.0 for value in metrics["qa_generation_f1"]["paired_actor_deltas"]
        ),
        "ood_shadow_benchmark_holdout_or_protected_semantics_used": False,
    }


def build_v57() -> dict:
    projection = maximin.maximin_projection_v55b()
    plans = design57.scale_plans_v57(projection)
    result = {
        "schema": "lora-es-conservative-actor-robust-preregistration-v57",
        "status": "sealed_before_v57_model_ray_gpu_or_train_gate_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "From V55B train-only evidence, search the previously untested "
            "sub-.25 scale interval while requiring every actor's paired "
            "train delta to clear zero on six observable gate metrics."
        ),
        "v55b_train_only_evidence": _v55_train_diagnosis(),
        "single_scientific_change": {
            "variables": ["sub_0p25_scale", "actor_robust_train_gate"],
            "same_exact_master_dataset_sigma_population_seeds_direction_alpha": True,
            "same_nine_inherited_endpoint_gates_without_weakening": True,
            "additional_actor_robustness_is_stricter_only": True,
            "no_ood_output_or_semantic_informed_scale_or_gate": True,
        },
        "maximin_projection": projection,
        "maximin_projection_content_sha256": projection["content_sha256"],
        "scale_plans": plans,
        "scale_plans_content_sha256": design52.canonical_sha256_v52(plans),
        "train_only_gate": {
            "scale_order": list(design57.SCALE_ORDER_V57),
            "required_checks": list(design52.TRAIN_GATE_NAMES_V52),
            "actor_robust_metrics": design57.ACTOR_ROBUST_METRICS_V57,
            "domain_comparison": "all_four_paired_actor_deltas_gt_zero",
            "noninferiority_comparison": "all_four_paired_actor_deltas_gte_zero",
            "largest_strictly_passing_scale": True,
            "candidate_actor_consensus_required": True,
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
            "gpu_log": str(RUN_DIR / "gpu_activity_v57.jsonl"),
            "projection_phase": "v57_conservative_actor_robust_projection_no_population_rerun",
            "minimum_projection_phase_barrier_seconds": 1.25,
            "all_four_physical_gpus_and_expected_actor_pids_required": True,
            "foreign_compute_processes_forbidden": True,
        },
        "artifacts": artifacts_v57(),
        "implementation_bindings": {
            "design_v57": binding("lora_es_conservative_actor_robust_v57.py"),
            "runner_v57": binding("run_lora_es_conservative_actor_robust_v57.py"),
            "maximin_v55b": binding("lora_es_fragile_maximin_projection_v55b.py"),
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
    value = build_v57()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    print(json.dumps({
        "path": str(args.output.resolve()),
        "file_sha256": file_sha256(args.output),
        "content_sha256": value["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
