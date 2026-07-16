#!/usr/bin/env python3
"""Run V57 conservative actor-robust train-only LoRA EGGROLL-ES HPO."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import build_lora_es_conservative_actor_robust_preregistration_v57 as builder
import lora_es_conservative_actor_robust_v57 as design57
import lora_es_fragile_maximin_projection_v55b as maximin
import lora_es_nested_population_v52 as design52
import run_lora_es_fragile_maximin_backtracking_v55b as compat
import run_lora_es_nested_population_v52 as executor52


ROOT = Path(__file__).resolve().parent
RUN_DIR = builder.RUN_DIR.resolve()
ATTEMPT = (RUN_DIR.parent / ".v57_lora_es_conservative_actor_robust.attempt.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v57.jsonl").resolve()
PHASE = "v57_conservative_actor_robust_projection_no_population_rerun"
NUMERIC = (RUN_DIR / "numeric_calibration_v57.json").resolve()
ANCHOR = (RUN_DIR / "anchor_calibration_v57.json").resolve()
PREINSTALL = (RUN_DIR / "preinstall_actor_baseline_v57.json").resolve()
MASTER_AUDIT = (RUN_DIR / "master_identity_audit_v57.json").resolve()
SNAPSHOT = (RUN_DIR / "selected_candidate_v57").resolve()
INTERNAL_REPORT = (RUN_DIR / "nested_population_report_v52.json").resolve()
P8_RECEIPT = (RUN_DIR / "p8_train_gate_v52.json").resolve()
P16_GATE = (RUN_DIR / "p16_train_gate_v52.json").resolve()
COMPAT_REPORT = (RUN_DIR / "compatibility_report_v55b.json").resolve()
REPORT = (RUN_DIR / "conservative_actor_robust_report_v57.json").resolve()
ORIGINAL_EVALUATE = compat.ORIGINAL_EVALUATE


def _compact_hash(value: dict) -> str:
    return design52.canonical_sha256_v52({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def load_preregistration_v57(args) -> dict:
    path = Path(args.preregistration).resolve()
    if builder.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v57 preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    projection = maximin.maximin_projection_v55b()
    plans = design57.scale_plans_v57(projection)
    if (
        value.get("schema") != "lora-es-conservative-actor-robust-preregistration-v57"
        or value.get("status") != "sealed_before_v57_model_ray_gpu_or_train_gate_access"
        or value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or _compact_hash(value) != args.preregistration_content_sha256
        or value.get("maximin_projection") != projection
        or value.get("scale_plans") != plans
        or value.get("scale_plans_content_sha256")
        != design52.canonical_sha256_v52(plans)
        or value.get("train_only_gate", {}).get("scale_order")
        != list(design57.SCALE_ORDER_V57)
        or value.get("train_only_gate", {}).get("required_checks")
        != list(design52.TRAIN_GATE_NAMES_V52)
        or value.get("artifacts") != builder.artifacts_v57()
        or value.get("fixed_recipe", {}).get("master_sha256")
        != design52.MASTER_SHA256_V52
        or value.get("fixed_recipe", {}).get("dataset_sha256")
        != design52.DATASET_SHA256_V52
        or value.get("runtime") != design52.RUNTIME_V52
        or value.get("authorization", {}).get("gpu_launch") is not True
        or value.get("authorization", {}).get("reuse_persisted_signed_scores") is not True
        or value.get("authorization", {}).get("population_generation_or_scoring") is not False
        or any(value.get("authorization", {}).get(key) is not False for key in (
            "optimizer_master_commit", "p8_evaluation",
            "ood_shadow_benchmark_or_holdout", "protected_semantics",
            "sealed_holdout",
        ))
        or value.get("protected_semantics_opened") is not False
        or value.get("ood_shadow_opened") is not False
        or value.get("terminal_holdout_opened") is not False
    ):
        raise RuntimeError("v57 preregistration contract changed")
    for binding in value["implementation_bindings"].values():
        if builder.file_sha256(Path(binding["path"])) != binding["file_sha256"]:
            raise RuntimeError("v57 implementation binding changed")
    return value


def evaluate_actor_robust_v57(arm_name, trainer, prior, *args, **kwargs):
    original_gate = prior.fused.candidate_gate_v43i

    def tightened(*gate_args, **gate_kwargs):
        return design57.tighten_gate_v57(
            original_gate, *gate_args, **gate_kwargs,
        )

    prior.fused.candidate_gate_v43i = tightened
    try:
        return ORIGINAL_EVALUATE(
            arm_name, trainer, prior, *args, **kwargs,
        )
    finally:
        prior.fused.candidate_gate_v43i = original_gate


@contextmanager
def patched_v57():
    compat_values = {
        "RUN_DIR": RUN_DIR, "ATTEMPT": ATTEMPT, "GPU_LOG": GPU_LOG,
        "PHASE": PHASE, "NUMERIC": NUMERIC, "ANCHOR": ANCHOR,
        "PREINSTALL": PREINSTALL, "MASTER_AUDIT": MASTER_AUDIT,
        "SNAPSHOT": SNAPSHOT, "INTERNAL_REPORT": INTERNAL_REPORT,
        "P8_RECEIPT": P8_RECEIPT, "P16_GATE": P16_GATE,
        "REPORT": COMPAT_REPORT,
        "ORIGINAL_EVALUATE": evaluate_actor_robust_v57,
    }
    saved_compat = {key: getattr(compat, key) for key in compat_values}
    saved_scale_fn = maximin.scale_plans_v55b
    saved_scale_order = design52.SCALE_ORDER_V52
    for key, value in compat_values.items():
        setattr(compat, key, value)
    maximin.scale_plans_v55b = design57.scale_plans_v57
    design52.SCALE_ORDER_V52 = design57.SCALE_ORDER_V57
    try:
        yield
    finally:
        design52.SCALE_ORDER_V52 = saved_scale_order
        maximin.scale_plans_v55b = saved_scale_fn
        for key, value in saved_compat.items():
            setattr(compat, key, value)


def _atomic_report(value: dict) -> dict:
    result = dict(value)
    result["content_sha256_before_self_field"] = design52.canonical_sha256_v52(result)
    payload = (json.dumps(
        result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False,
    ) + "\n").encode("ascii")
    temporary = REPORT.with_name(f".{REPORT.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, REPORT)
    return result


def _telemetry_summary() -> dict:
    by_gpu = {}
    foreign = 0
    phases = set()
    with GPU_LOG.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            phases.add(row["phase"])
            foreign += len(row["foreign_compute_pids"])
            if row["phase"] != PHASE:
                continue
            gpu = str(row["gpu"])
            item = by_gpu.setdefault(gpu, {
                "expected_pid": row["expected_pid"], "samples": 0,
                "positive_utilization_samples": 0,
                "peak_utilization_percent": 0,
                "peak_memory_used_mib": 0,
            })
            if item["expected_pid"] != row["expected_pid"]:
                raise RuntimeError("v57 expected PID changed within phase")
            item["samples"] += 1
            utilization = row["utilization_percent"]
            item["positive_utilization_samples"] += int(utilization > 0)
            item["peak_utilization_percent"] = max(
                item["peak_utilization_percent"], utilization,
            )
            item["peak_memory_used_mib"] = max(
                item["peak_memory_used_mib"], row["memory_used_mib"],
            )
            if row["expected_pid"] not in row["compute_pids"]:
                raise RuntimeError("v57 expected actor PID absent")
    if (
        set(by_gpu) != {"0", "1", "2", "3"}
        or any(item["positive_utilization_samples"] < 1 for item in by_gpu.values())
        or PHASE not in phases or foreign != 0
    ):
        raise RuntimeError("v57 four-GPU telemetry contract failed")
    return {
        "phase": PHASE, "by_gpu": by_gpu,
        "all_four_expected_actor_pids_resident": True,
        "all_four_positive_utilization": True,
        "foreign_compute_pid_rows": foreign,
        "file_sha256": builder.file_sha256(GPU_LOG),
    }


def execute_v57(preregistration: dict) -> int:
    with patched_v57():
        code = compat.execute_v55b(preregistration)
    if code != 0:
        raise RuntimeError("v57 compatibility executor failed")
    p16 = json.loads(P16_GATE.read_text(encoding="utf-8"))
    internal = json.loads(INTERNAL_REPORT.read_text(encoding="utf-8"))
    compat_report = json.loads(COMPAT_REPORT.read_text(encoding="utf-8"))
    scale_results = p16.get("scale_results", [])
    if (
        p16.get("schema") != "fragile-maximin-p16-backtracking-gate-v55b"
        or not scale_results
        or any(
            item.get("gate", {}).get("actor_robustness_v57", {}).get("schema")
            != "four-actor-paired-train-robustness-v57"
            for item in scale_results
        )
        or p16.get("protected_semantics_opened") is not False
        or p16.get("sealed_holdout_opened") is not False
        or internal.get("optimizer_master_committed") is not False
        or internal.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
    ):
        raise RuntimeError("v57 completion contract changed")
    telemetry = _telemetry_summary()
    report = _atomic_report({
        "schema": "lora-es-conservative-actor-robust-report-v57",
        "status": "complete_train_only_no_master_commit",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_content_sha256": preregistration[
            "content_sha256_before_self_field"
        ],
        "scientific_basis": "V55B train-only actor-sign fragility",
        "evaluated_ratios": [item["target_norm_ratio"] for item in scale_results],
        "selected_target_norm_ratio": p16["selected_target_norm_ratio"],
        "passing_candidate_saved": p16["selected_target_norm_ratio"] is not None,
        "p16_train_gate": {
            "path": str(P16_GATE), "file_sha256": builder.file_sha256(P16_GATE),
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
            "content_sha256": compat_report["content_sha256_before_self_field"],
        },
        "telemetry": telemetry,
        "optimizer_master_committed": False,
        "all_candidates_exactly_aborted_to_master": True,
        "document_disjoint_evaluation_doctrine_preserved": True,
        "ood_shadow_benchmark_holdout_or_protected_semantics_opened": False,
        "terminal_holdout_opened": False,
    })
    print(json.dumps({
        "report": str(REPORT), "file_sha256": builder.file_sha256(REPORT),
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
        raise ValueError("v57 requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v57(args)
    if args.dry_run:
        print(json.dumps({
            "schema": preregistration["schema"],
            "scale_order": list(design57.SCALE_ORDER_V57),
            "actor_robust_gate": True, "gpu_log": str(GPU_LOG),
            "phase": PHASE, "filesystem_writes": False,
            "model_ray_gpu_or_train_semantics_loaded": False,
            "protected_semantics_loaded": False,
            "ood_shadow_opened": False, "terminal_holdout_opened": False,
        }, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != design52.REQUIRED_PYTHON_V52:
        raise RuntimeError("v57 requires the sealed es-at-scale interpreter")
    return execute_v57(preregistration)


if __name__ == "__main__":
    raise SystemExit(main())
