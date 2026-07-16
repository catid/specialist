#!/usr/bin/env python3
"""Seal V54 selected-score projection/backtracking before GPU access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as design


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "experiments/eggroll_es_hpo/preregistrations/lora_es_selected_score_backtracking_v54.json"
RUN_DIR = ROOT / "experiments/eggroll_es_hpo/runs/v54_lora_es_selected_score_backtracking"
V53_DIR = ROOT / "experiments/eggroll_es_hpo/runs/v53_lora_es_sigma_discrimination"
V53_REPORT = V53_DIR / "sigma_discrimination_report_v53.json"
V53_POPULATION = V53_DIR / "nested_population_v52.json"
V53_SELECTED_ARM = V53_DIR / "sigma_0p0048_arm_v53.json"
V53_EVIDENCE = V53_DIR / "v53_evidence_manifest.json"
V53_TELEMETRY = V53_DIR / "gpu_activity_v52.jsonl"
V53_DEVIATION = ROOT / "experiments/eggroll_es_hpo/preregistrations/lora_es_sigma_discrimination_v53_telemetry_deviation.json"


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


def selected_projection_v54() -> tuple[dict, list[dict]]:
    arm = json.loads(V53_SELECTED_ARM.read_text(encoding="utf-8"))
    integrity = arm.get("perturbation_integrity", {})
    state = arm.get("post_arm_exact_master_state", {})
    actors = state.get("actors", [])
    evidence = json.loads(V53_EVIDENCE.read_text(encoding="utf-8"))
    if (
        arm.get("sigma") != 0.0048
        or arm.get("population_size") != 16
        or arm.get("passed") is not True
        or arm.get("reliability", {}).get("passed") is not True
        or integrity.get("states") != 32
        or integrity.get("unique_nonmaster_fp32_candidates") != 32
        or integrity.get("unique_nonmaster_bf16_candidates") != 32
        or integrity.get("four_actor_consensus_before_scoring") is not True
        or integrity.get("direct_from_pinned_master") is not True
        or integrity.get("exact_antithetic_seed_sign_coverage") is not True
        or len(actors) != 4
        or {item.get("canonical_fp32_master_sha256") for item in actors}
        != {design.MASTER_SHA256_V52}
        or {item.get("bf16_runtime_values_sha256") for item in actors}
        != {design.MASTER_RUNTIME_SHA256_V52}
        or {item.get("reference_identity_sha256") for item in actors}
        != {design.MASTER_SHA256_V52}
        or {item.get("update_sequence") for item in actors} != {0}
        or state.get("worker_transactions_quiescent") is not True
        or state.get("controller_transaction_quiescent") is not True
        or evidence.get("selected_arm", {}).get("file_sha256")
        != file_sha256(V53_SELECTED_ARM)
        or evidence.get("selected_arm", {}).get("content_sha256")
        != arm.get("content_sha256_before_self_field")
        or evidence.get("selected_arm", {}).get(
            "passed_every_preregistered_scientific_gate"
        ) is not True
        or evidence.get("selected_arm", {}).get(
            "passed_every_preregistered_state_integrity_gate"
        ) is not True
        or arm.get("projection_performed") is not False
        or arm.get("optimizer_update_or_train_gate_opened") is not False
        or arm.get("protected_semantics_opened") is not False
        or arm.get("sealed_holdout_opened") is not False
    ):
        raise RuntimeError("v54 selected V53 arm contract changed")
    projection = design.project_arm_v52(arm["signed_scores"], 16)
    plans = design.scale_plans_v52(projection)
    return projection, plans


def artifacts_v54() -> dict:
    return {
        "attempt": str(RUN_DIR.parent / ".v54_lora_es_selected_score_backtracking.attempt.json"),
        "run_directory": str(RUN_DIR),
        "selected_score_projection": str(RUN_DIR / "nested_population_v52.json"),
        "no_p8_evaluation_receipt": str(RUN_DIR / "p8_train_gate_v52.json"),
        "p16_train_gate": str(RUN_DIR / "p16_train_gate_v52.json"),
        "passing_candidate_snapshot": str(RUN_DIR / "selected_candidate_v54"),
        "internal_report": str(RUN_DIR / "nested_population_report_v52.json"),
        "report": str(RUN_DIR / "selected_score_backtracking_report_v54.json"),
        "failure": str(RUN_DIR / "failure_v52.json"),
        "numeric_calibration": str(RUN_DIR / "numeric_calibration_v54.json"),
        "anchor_calibration": str(RUN_DIR / "anchor_calibration_v54.json"),
        "gpu_log": str(RUN_DIR / "gpu_activity_v54.jsonl"),
        "coordinator_log": str(RUN_DIR.parent / "v54_lora_es_selected_score_backtracking.coordinator.log"),
        "sealed_holdout_artifact": None,
    }


def build_v54() -> dict:
    projection, plans = selected_projection_v54()
    report = json.loads(V53_REPORT.read_text(encoding="utf-8"))
    if report.get("selected_smallest_passing_sigma") != 0.0048:
        raise RuntimeError("v54 V53 report selection changed")
    result = {
        "schema": "lora-es-selected-score-backtracking-preregistration-v54",
        "status": "sealed_before_v54_model_ray_gpu_or_train_gate_access",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Reuse the selected V53 P16 signed scores exactly once to compute "
            "the existing five-halfspace projection and evaluate train-only "
            "backtracking scales against the exact v434 master."
        ),
        "selected_v53_input": {
            "sigma": 0.0048,
            "population_size": 16,
            "p16_seeds": list(design.P16_SEEDS_V52),
            "optimizer_update_alpha": design.ALPHA_V52,
            "population_rerun_authorized": False,
            "report": {"path": str(V53_REPORT), "file_sha256": file_sha256(V53_REPORT), "content_sha256": content_sha(V53_REPORT)},
            "population": {"path": str(V53_POPULATION), "file_sha256": file_sha256(V53_POPULATION), "content_sha256": content_sha(V53_POPULATION)},
            "selected_arm": {"path": str(V53_SELECTED_ARM), "file_sha256": file_sha256(V53_SELECTED_ARM), "content_sha256": content_sha(V53_SELECTED_ARM)},
            "evidence_manifest": {"path": str(V53_EVIDENCE), "file_sha256": file_sha256(V53_EVIDENCE)},
            "actual_telemetry": {"path": str(V53_TELEMETRY), "file_sha256": file_sha256(V53_TELEMETRY)},
            "telemetry_deviation": {"path": str(V53_DEVIATION), "file_sha256": file_sha256(V53_DEVIATION), "declaration_commit": "8492a77b35479ad8b2329d8126b2efb6175eba63"},
        },
        "projection": projection,
        "projection_content_sha256": design.canonical_sha256_v52(projection),
        "scale_plans": plans,
        "scale_plans_content_sha256": design.canonical_sha256_v52(plans),
        "train_only_gate": {
            "five_halfspaces": list(design.OBJECTIVE_PATHS_V52),
            "scale_order": list(design.SCALE_ORDER_V52),
            "optimizer_update_alpha": design.ALPHA_V52,
            "p16_seeds": list(design.P16_SEEDS_V52),
            "largest_strictly_passing_scale": True,
            "required_checks": list(design.TRAIN_GATE_NAMES_V52),
            "all_nine_checks_required": True,
            "fresh_numeric_and_anchor_calibration_against_exact_master": True,
            "candidate_actor_consensus_required": True,
            "save_pending_candidate_only_for_passing_scale": True,
            "every_in_memory_candidate_exactly_aborted_to_master": True,
            "p8_population_or_train_arm_evaluated": False,
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
            "gpu_log": str(RUN_DIR / "gpu_activity_v54.jsonl"),
            "projection_phase": "v54_selected_v53_p16_score_projection_no_population_rerun",
            "inherited_v52_filename_or_population_phase_allowed": False,
        },
        "artifacts": artifacts_v54(),
        "implementation_bindings": {
            "builder": binding("build_lora_es_selected_score_backtracking_preregistration_v54.py"),
            "runner": binding("run_lora_es_selected_score_backtracking_v54.py"),
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
    value = build_v54()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"path": str(args.output), "file_sha256": file_sha256(args.output), "content_sha256": value["content_sha256_before_self_field"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
