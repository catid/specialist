#!/usr/bin/env python3
"""Run V59 fragile-F1-priority LoRA EGGROLL-ES train-only HPO."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import build_lora_es_fragile_priority_preregistration_v59 as builder
import lora_es_fragile_priority_projection_v59 as design59
import lora_es_nested_population_v52 as design52
import run_lora_es_actor_maximin_v58 as shell58
import run_lora_es_fragile_maximin_backtracking_v55b as compat


ROOT = Path(__file__).resolve().parent
RUN_DIR = builder.RUN_DIR.resolve()
ATTEMPT = (
    RUN_DIR.parent / ".v59_lora_es_fragile_priority.attempt.json"
).resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v59.jsonl").resolve()
PHASE = "v59_fragile_priority_projection_no_population_rerun"
NUMERIC = (RUN_DIR / "numeric_calibration_v59.json").resolve()
ANCHOR = (RUN_DIR / "anchor_calibration_v59.json").resolve()
PREINSTALL = (RUN_DIR / "preinstall_actor_baseline_v59.json").resolve()
MASTER_AUDIT = (RUN_DIR / "master_identity_audit_v59.json").resolve()
SNAPSHOT = (RUN_DIR / "selected_candidate_v59").resolve()
INTERNAL_REPORT = (RUN_DIR / "nested_population_report_v52.json").resolve()
P8_RECEIPT = (RUN_DIR / "p8_train_gate_v52.json").resolve()
P16_GATE = (RUN_DIR / "p16_train_gate_v52.json").resolve()
COMPAT_REPORT = (RUN_DIR / "compatibility_report_v55b.json").resolve()
REPORT = (RUN_DIR / "fragile_priority_report_v59.json").resolve()


def _compact_hash(value: dict) -> str:
    return design52.canonical_sha256_v52({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def load_preregistration_v59(args) -> dict:
    path = Path(args.preregistration).resolve()
    if builder.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v59 preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    projection = design59.fragile_priority_projection_v59()
    plans = design59.scale_plans_v59(projection)
    authorization = value.get("authorization", {})
    if (
        value.get("schema")
        != "lora-es-fragile-priority-preregistration-v59"
        or value.get("status")
        != "sealed_before_v59_model_ray_gpu_or_train_gate_access"
        or value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or _compact_hash(value) != args.preregistration_content_sha256
        or value.get("fragile_priority_projection") != projection
        or value.get("scale_plans") != plans
        or value.get("scale_plans_content_sha256")
        != design52.canonical_sha256_v52(plans)
        or value.get("train_only_gate", {}).get("scale_order")
        != list(design59.SCALE_ORDER_V59)
        or value.get("train_only_gate", {}).get("required_checks")
        != list(design52.TRAIN_GATE_NAMES_V52)
        or value.get("train_only_gate", {}).get(
            "all_nine_original_calibrated_checks_required_without_weakening"
        ) is not True
        or value.get("train_only_gate", {}).get(
            "additional_raw_actor_sign_checks"
        ) is not False
        or value.get("single_scientific_change", {}).get("variable")
        != "projection_direction_only"
        or value.get("single_scientific_change", {}).get(
            "nonfragile_floor"
        ) != 0.15
        or value.get("artifacts") != builder.artifacts_v59()
        or value.get("fixed_recipe", {}).get("master_sha256")
        != design52.MASTER_SHA256_V52
        or value.get("fixed_recipe", {}).get("dataset_sha256")
        != design52.DATASET_SHA256_V52
        or value.get("runtime") != design52.RUNTIME_V52
        or authorization.get("gpu_launch") is not True
        or authorization.get("reuse_persisted_signed_scores") is not True
        or authorization.get("population_generation_or_scoring") is not False
        or any(authorization.get(key) is not False for key in (
            "optimizer_master_commit", "p8_evaluation",
            "ood_shadow_benchmark_or_holdout", "protected_semantics",
            "sealed_holdout",
        ))
        or value.get("protected_semantics_opened") is not False
        or value.get("ood_shadow_opened") is not False
        or value.get("terminal_holdout_opened") is not False
    ):
        raise RuntimeError("v59 preregistration contract changed")
    for binding in value["implementation_bindings"].values():
        if builder.file_sha256(Path(binding["path"])) != binding["file_sha256"]:
            raise RuntimeError("v59 implementation binding changed")
    return value


def _configure_v58_shell() -> None:
    replacements = {
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "GPU_LOG": GPU_LOG,
        "PHASE": PHASE,
        "NUMERIC": NUMERIC,
        "ANCHOR": ANCHOR,
        "PREINSTALL": PREINSTALL,
        "MASTER_AUDIT": MASTER_AUDIT,
        "SNAPSHOT": SNAPSHOT,
        "INTERNAL_REPORT": INTERNAL_REPORT,
        "P8_RECEIPT": P8_RECEIPT,
        "P16_GATE": P16_GATE,
        "COMPAT_REPORT": COMPAT_REPORT,
        "REPORT": REPORT,
    }
    for key, value in replacements.items():
        setattr(shell58, key, value)


def _validated_scale_path_v59(
    scale_results: list[dict], selected_target_norm_ratio: float | None,
) -> list[float]:
    """Require the executor's full-failure grid or first-pass exact prefix."""
    expected = list(design59.SCALE_ORDER_V59)
    ratios = [item.get("target_norm_ratio") for item in scale_results]
    if not ratios or ratios != expected[:len(ratios)]:
        raise RuntimeError("v59 evaluated scale path changed")
    if selected_target_norm_ratio is None:
        if ratios != expected:
            raise RuntimeError("v59 all-fail scale path ended early")
    elif selected_target_norm_ratio != ratios[-1]:
        raise RuntimeError("v59 selected scale is not final evaluated prefix item")
    return ratios


def execute_v59(preregistration: dict) -> int:
    projection = preregistration["fragile_priority_projection"]
    plans = preregistration["scale_plans"]
    compatibility_preregistration = dict(preregistration)
    compatibility_preregistration["maximin_projection"] = projection
    _configure_v58_shell()
    with shell58.patched_v58(projection, plans):
        code = compat.execute_v55b(compatibility_preregistration)
    if code != 0:
        raise RuntimeError("v59 compatibility executor failed")

    p16 = json.loads(P16_GATE.read_text(encoding="utf-8"))
    internal = json.loads(INTERNAL_REPORT.read_text(encoding="utf-8"))
    compatibility_report = json.loads(
        COMPAT_REPORT.read_text(encoding="utf-8")
    )
    scale_results = p16.get("scale_results", [])
    expected_checks = set(design52.TRAIN_GATE_NAMES_V52)
    evaluated_ratios = _validated_scale_path_v59(
        scale_results, p16.get("selected_target_norm_ratio"),
    )
    if (
        p16.get("schema") != "fragile-maximin-p16-backtracking-gate-v55b"
        or not scale_results
        or any(set(item.get("checks", {})) != expected_checks
               for item in scale_results)
        or any("actor_robustness_v57" in item.get("gate", {})
               for item in scale_results)
        or p16.get("protected_semantics_opened") is not False
        or p16.get("sealed_holdout_opened") is not False
        or internal.get("optimizer_master_committed") is not False
        or internal.get("protected_semantics_opened") is not False
        or internal.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
    ):
        raise RuntimeError("v59 completion contract changed")

    telemetry = {
        "projection": shell58._phase_summary(PHASE),
        "p16_train_gate": shell58._phase_summary(
            "p16_backtracking_train_gate"
        ),
        "gpu_log_file_sha256": builder.file_sha256(GPU_LOG),
        "foreign_compute_pid_rows": 0,
        "all_four_positive_each_required_phase": True,
    }
    per_ratio_train_gate = [
        {
            "target_norm_ratio": item["target_norm_ratio"],
            "passed": item["gate"]["passed"],
            "failed_checks": sorted(
                name for name, passed in item["checks"].items()
                if passed is not True
            ),
        }
        for item in scale_results
    ]
    report = shell58._atomic_report({
        "schema": "lora-es-fragile-priority-report-v59",
        "status": "complete_train_only_no_master_commit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_content_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "projection_direction_sha256": projection["direction_sha256"],
        "minimum_fragile_f1_actor_margin": projection[
            "minimum_fragile_margin"
        ],
        "minimum_other_actor_objective_margin": projection[
            "minimum_other_margin"
        ],
        "nonfragile_margin_floor": projection["solution"][
            "required_other_margin_floor"
        ],
        "v58_direction_cosine": projection["v58_direction_cosine"],
        "maximum_kkt_stationarity_residual": projection["solution"][
            "maximum_stationarity_residual"
        ],
        "maximum_kkt_complementarity_residual": projection["solution"][
            "maximum_complementarity_residual"
        ],
        "per_objective_minimum_projected_margins": projection["solution"][
            "per_objective_minimum_margins"
        ],
        "evaluated_ratios": evaluated_ratios,
        "per_ratio_train_gate": per_ratio_train_gate,
        "selected_target_norm_ratio": p16["selected_target_norm_ratio"],
        "passing_candidate_saved": p16["selected_target_norm_ratio"] is not None,
        "p16_train_gate": {
            "path": str(P16_GATE),
            "file_sha256": builder.file_sha256(P16_GATE),
            "content_sha256": p16["content_sha256_before_self_field"],
        },
        "internal_exact_state_report": {
            "path": str(INTERNAL_REPORT),
            "file_sha256": builder.file_sha256(INTERNAL_REPORT),
            "content_sha256": internal["content_sha256_before_self_field"],
        },
        "compatibility_report": {
            "path": str(COMPAT_REPORT),
            "file_sha256": builder.file_sha256(COMPAT_REPORT),
            "content_sha256": compatibility_report[
                "content_sha256_before_self_field"
            ],
        },
        "telemetry": telemetry,
        "original_nine_calibrated_endpoint_gates_retained": True,
        "additional_raw_actor_sign_checks_used": False,
        "optimizer_master_committed": False,
        "all_candidates_exactly_aborted_to_master": True,
        "document_disjoint_evaluation_doctrine_preserved": True,
        "ood_shadow_benchmark_holdout_or_protected_semantics_opened": False,
        "terminal_holdout_opened": False,
    })
    print(json.dumps({
        "report": str(REPORT),
        "file_sha256": builder.file_sha256(REPORT),
        "content_sha256": report["content_sha256_before_self_field"],
        "selected_target_norm_ratio": report["selected_target_norm_ratio"],
    }, sort_keys=True))
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
        raise ValueError("v59 requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v59(args)
    if args.dry_run:
        projection = preregistration["fragile_priority_projection"]
        print(json.dumps({
            "schema": preregistration["schema"],
            "direction_sha256": projection["direction_sha256"],
            "minimum_fragile_f1_actor_margin": projection[
                "minimum_fragile_margin"
            ],
            "minimum_other_actor_objective_margin": projection[
                "minimum_other_margin"
            ],
            "nonfragile_margin_floor": projection["solution"][
                "required_other_margin_floor"
            ],
            "scale_order": list(design59.SCALE_ORDER_V59),
            "original_nine_calibrated_gates": True,
            "additional_raw_actor_sign_checks": False,
            "gpu_log": str(GPU_LOG),
            "phase": PHASE,
            "filesystem_writes": False,
            "model_ray_gpu_or_train_semantics_loaded": False,
            "protected_semantics_loaded": False,
            "ood_shadow_opened": False,
            "terminal_holdout_opened": False,
        }, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != design52.REQUIRED_PYTHON_V52:
        raise RuntimeError("v59 requires the sealed es-at-scale interpreter")
    return execute_v59(preregistration)


if __name__ == "__main__":
    raise SystemExit(main())
