#!/usr/bin/env python3
"""Dedicated V54 report wrapper over V52 exact candidate machinery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import build_lora_es_selected_score_backtracking_preregistration_v54 as builder
import lora_es_nested_population_v52 as design
import run_lora_es_nested_population_v52 as v52


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = builder.OUTPUT.resolve()
RUN_DIR = builder.RUN_DIR.resolve()
ATTEMPT = (RUN_DIR.parent / ".v54_lora_es_selected_score_backtracking.attempt.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v54.jsonl").resolve()
PHASE = "v54_selected_v53_p16_score_projection_no_population_rerun"
NUMERIC = (RUN_DIR / "numeric_calibration_v54.json").resolve()
ANCHOR = (RUN_DIR / "anchor_calibration_v54.json").resolve()
PREINSTALL = (RUN_DIR / "preinstall_actor_baseline_v54.json").resolve()
MASTER_AUDIT = (RUN_DIR / "master_identity_audit_v54.json").resolve()
P16_SNAPSHOT = (RUN_DIR / "selected_candidate_v54").resolve()
INTERNAL_REPORT = (RUN_DIR / "nested_population_report_v52.json").resolve()
P8_RECEIPT = (RUN_DIR / "p8_train_gate_v52.json").resolve()
P16_GATE = (RUN_DIR / "p16_train_gate_v52.json").resolve()
REPORT = (RUN_DIR / "selected_score_backtracking_report_v54.json").resolve()
ORIGINAL_EVALUATE = v52.evaluate_train_arm_v52


def file_sha256(path: Path) -> str:
    return builder.file_sha256(path)


def load_preregistration_v54(args) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v54 preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {key: item for key, item in value.items() if key != "content_sha256_before_self_field"}
    authorization = value.get("authorization", {})
    projection, plans = builder.selected_projection_v54()
    if (
        value.get("schema") != "lora-es-selected-score-backtracking-preregistration-v54"
        or value.get("status") != "sealed_before_v54_model_ray_gpu_or_train_gate_access"
        or value.get("content_sha256_before_self_field") != args.preregistration_content_sha256
        or design.canonical_sha256_v52(compact) != args.preregistration_content_sha256
        or value.get("projection") != projection
        or value.get("scale_plans") != plans
        or value.get("selected_v53_input", {}).get("p16_seeds")
        != list(design.P16_SEEDS_V52)
        or value.get("selected_v53_input", {}).get("optimizer_update_alpha")
        != design.ALPHA_V52
        or value.get("train_only_gate", {}).get("p16_seeds")
        != list(design.P16_SEEDS_V52)
        or value.get("train_only_gate", {}).get("optimizer_update_alpha")
        != design.ALPHA_V52
        or value.get("artifacts") != builder.artifacts_v54()
        or value.get("telemetry") != {"gpu_log": str(GPU_LOG), "projection_phase": PHASE, "inherited_v52_filename_or_population_phase_allowed": False}
        or authorization.get("population_generation_or_scoring") is not False
        or authorization.get("optimizer_master_commit") is not False
        or authorization.get("p8_evaluation") is not False
        or any(authorization.get(key) is not False for key in ("ood_shadow_benchmark_or_holdout", "protected_semantics", "sealed_holdout"))
        or value.get("fixed_recipe", {}).get("master_sha256") != design.MASTER_SHA256_V52
        or value.get("fixed_recipe", {}).get("dataset_sha256") != design.DATASET_SHA256_V52
        or value.get("runtime") != design.RUNTIME_V52
        or value.get("protected_semantics_opened") is not False
        or value.get("sealed_holdout_opened") is not False
    ):
        raise RuntimeError("v54 preregistration contract changed")
    for binding in value["implementation_bindings"].values():
        if file_sha256(Path(binding["path"])) != binding["file_sha256"]:
            raise RuntimeError("v54 implementation binding changed")
    return value


def selected_score_projection_v54(*_args, **_kwargs) -> dict:
    arm = json.loads(builder.V53_SELECTED_ARM.read_text(encoding="utf-8"))
    projection, plans = builder.selected_projection_v54()
    return {
        "schema": "selected-v53-p16-score-projection-v54",
        "status": "complete_without_population_rerun",
        "source_signed_scores_file_sha256": file_sha256(builder.V53_SELECTED_ARM),
        "source_signed_scores_sha256": arm["signed_scores_sha256"],
        "arms": {
            "p8": {"schema": "no-p8-arm-v54", "evaluation_authorized": False},
            "p16": {
                "schema": "selected-v53-p16-arm-v54", "population_size": 16,
                "sigma": 0.0048, "reliability": arm["reliability"],
                "projection": projection, "scale_plans": plans,
            },
        },
        "population_generation_or_scoring_performed": False,
        "projection_performed_from_persisted_scores": True,
        "optimizer_update_or_train_gate_opened": False,
        "protected_semantics_opened": False,
        "sealed_holdout_opened": False,
        "timing": {"coverage": {"schema": "no-population-rerun-v54", "states": 0, "total_actor_phase_receipts": 0}},
    }


def require_selected_projection_v54(population: dict) -> None:
    if (
        population.get("schema") != "selected-v53-p16-score-projection-v54"
        or population.get("population_generation_or_scoring_performed") is not False
        or population.get("projection_performed_from_persisted_scores") is not True
        or population.get("arms", {}).get("p8", {}).get("evaluation_authorized") is not False
    ):
        raise RuntimeError("v54 selected-score projection contract changed")


def evaluate_only_selected_p16_v54(name, *args, **kwargs):
    if name == "p8":
        return {
            "schema": "no-p8-evaluation-receipt-v54", "arm": "p8",
            "evaluation_performed": False, "candidate_transactions_opened": 0,
            "selected_target_norm_ratio": None, "selected_snapshot": None,
            "master_committed": False, "protected_semantics_opened": False,
            "sealed_holdout_opened": False,
        }
    if name != "p16":
        raise RuntimeError("v54 unexpected train arm")
    result = ORIGINAL_EVALUATE(name, *args, **kwargs)
    result = dict(result)
    result["schema"] = "selected-v53-p16-backtracking-gate-v54"
    result["population_rerun_performed"] = False
    result["p8_evaluation_performed"] = False
    return result


@contextmanager
def patched_executor_v54():
    replacements = {
        "RUN_DIR": RUN_DIR, "ATTEMPT": ATTEMPT,
        "PREINSTALL_BASELINE": PREINSTALL, "MASTER_IDENTITY_AUDIT": MASTER_AUDIT,
        "NUMERIC_CALIBRATION": NUMERIC, "ANCHOR_CALIBRATION": ANCHOR,
        "P8_SNAPSHOT": RUN_DIR / "forbidden_p8_snapshot",
        "P16_SNAPSHOT": P16_SNAPSHOT,
        "GPU_LOG_OVERRIDE_V52": GPU_LOG,
        "POPULATION_PHASE_OVERRIDE_V52": PHASE,
        "replicated_population_v52": selected_score_projection_v54,
        "require_all_arms_reliable_v52": require_selected_projection_v54,
        "evaluate_train_arm_v52": evaluate_only_selected_p16_v54,
    }
    saved = {key: getattr(v52, key) for key in replacements}
    for key, value in replacements.items():
        setattr(v52, key, value)
    try:
        assert v52.runtime_telemetry_contract_v52() == {"gpu_log": GPU_LOG, "population_phase": PHASE}
        yield
    finally:
        for key, value in saved.items():
            setattr(v52, key, value)


def _write_report(value: dict) -> dict:
    result = dict(value)
    result["content_sha256_before_self_field"] = design.canonical_sha256_v52(result)
    payload = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    temporary = REPORT.with_name(f".{REPORT.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, REPORT)
    return result


def execute_v54(preregistration: dict) -> int:
    with patched_executor_v54():
        code = v52._execute_v52(preregistration)
    if code != 0 or not INTERNAL_REPORT.exists() or not P8_RECEIPT.exists() or not P16_GATE.exists():
        raise RuntimeError("v54 internal executor did not complete")
    internal = json.loads(INTERNAL_REPORT.read_text(encoding="utf-8"))
    p8 = json.loads(P8_RECEIPT.read_text(encoding="utf-8"))
    p16 = json.loads(P16_GATE.read_text(encoding="utf-8"))
    phases = set()
    foreign = 0
    with GPU_LOG.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); phases.add(row["phase"]); foreign += len(row["foreign_compute_pids"])
    if (
        p8.get("schema") != "no-p8-evaluation-receipt-v54"
        or p8.get("evaluation_performed") is not False
        or p8.get("candidate_transactions_opened") != 0
        or p16.get("schema") != "selected-v53-p16-backtracking-gate-v54"
        or p16.get("population_rerun_performed") is not False
        or internal.get("optimizer_master_committed") is not False
        or internal.get("protected_semantics_opened") is not False
        or PHASE not in phases or "nested_p16_population_v52" in phases
        or foreign != 0
    ):
        raise RuntimeError("v54 dedicated completion contract changed")
    report = _write_report({
        "schema": "lora-es-selected-score-backtracking-report-v54",
        "status": "complete_train_only_no_master_commit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_content_sha256": preregistration["content_sha256_before_self_field"],
        "selected_v53_sigma": 0.0048,
        "population_rerun_performed": False,
        "p8_evaluation_performed": False,
        "selected_target_norm_ratio": p16["selected_target_norm_ratio"],
        "passing_candidate_saved": p16["selected_target_norm_ratio"] is not None,
        "p16_train_gate": {"path": str(P16_GATE), "file_sha256": file_sha256(P16_GATE), "content_sha256": p16["content_sha256_before_self_field"]},
        "internal_exact_state_report": {"path": str(INTERNAL_REPORT), "file_sha256": file_sha256(INTERNAL_REPORT), "content_sha256": internal["content_sha256_before_self_field"]},
        "compatibility_shell_metadata": {
            "internal_report_schema": internal.get("schema"),
            "internal_shared_by_p8_and_p16_field": internal.get(
                "shared_fresh_calibration", {}
            ).get("shared_by_p8_and_p16"),
            "shared_by_p8_and_p16_is_factual_v54_science": False,
            "actual_v54_science": (
                "one selected persisted V53 P16 arm; no P8 evaluation and no "
                "population rerun"
            ),
        },
        "telemetry": {"path": str(GPU_LOG), "file_sha256": file_sha256(GPU_LOG), "required_phase_present": True, "inherited_v52_phase_absent": True, "foreign_compute_pid_rows": 0},
        "optimizer_master_committed": False,
        "all_in_memory_candidates_exactly_aborted_to_master": True,
        "ood_shadow_benchmark_holdout_or_protected_semantics_opened": False,
        "sealed_holdout_opened": False,
    })
    print(json.dumps({"report": str(REPORT), "file_sha256": file_sha256(REPORT), "content_sha256": report["content_sha256_before_self_field"], "selected_target_norm_ratio": report["selected_target_norm_ratio"]}, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--execute", action="store_true")
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("v54 requires exactly one of --dry-run or --execute")
    prereg = load_preregistration_v54(args)
    if args.dry_run:
        print(json.dumps({"schema": prereg["schema"], "selected_sigma": 0.0048, "population_rerun": False, "p8_evaluation": False, "gpu_log": str(GPU_LOG), "phase": PHASE, "filesystem_writes": False, "model_ray_gpu_or_train_semantics_loaded": False, "protected_semantics_loaded": False, "sealed_holdout_opened": False}, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != design.REQUIRED_PYTHON_V52:
        raise RuntimeError("v54 requires the sealed es-at-scale interpreter")
    return execute_v54(prereg)


if __name__ == "__main__":
    raise SystemExit(main())
