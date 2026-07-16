#!/usr/bin/env python3
"""Thin fail-closed V53 coordinator reusing the proven V52 actor machinery."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import lora_es_nested_population_v52 as base_design
import lora_es_sigma_discrimination_v53 as design
import run_lora_es_nested_population_v52 as v52


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = ROOT / "experiments/eggroll_es_hpo/preregistrations/lora_es_sigma_discrimination_v53.json"
RUN_DIR = ROOT / "experiments/eggroll_es_hpo/runs/v53_lora_es_sigma_discrimination"
ATTEMPT = RUN_DIR.parent / ".v53_lora_es_sigma_discrimination.attempt.json"
GPU_LOG = RUN_DIR / "gpu_activity_v53.jsonl"
NUMERIC = RUN_DIR / "numeric_calibration_v53.json"
ANCHOR = RUN_DIR / "anchor_calibration_v53.json"
PREINSTALL = RUN_DIR / "preinstall_actor_baseline_v53.json"
MASTER_AUDIT = RUN_DIR / "master_identity_audit_v53.json"
REPORT = RUN_DIR / "sigma_discrimination_report_v53.json"
POPULATION = RUN_DIR / "nested_population_v52.json"
FAILURE = RUN_DIR / "failure_v52.json"


class V53MeasurementComplete(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_content(value: dict) -> str:
    return design.canonical_sha256_v53({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def load_preregistration(args) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v53 preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    fixed = value.get("fixed_gates", {})
    authorization = value.get("authorization", {})
    if (
        value.get("schema") != "lora-es-sigma-discrimination-preregistration-v53"
        or value.get("status") != "sealed_before_v53_model_ray_gpu_or_protected_access"
        or value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or _compact_content(value) != args.preregistration_content_sha256
        or value.get("design", {}).get("arms") != design.arm_contracts_v53()
        or value.get("design", {}).get("fixed_seeds") != list(design.SEEDS_V53)
        or value.get("design", {}).get("adaptive_stop")
        != "stop_after_first_arm_passing_every_fixed_gate"
        or fixed.get("reliability_minimum") != design.MINIMUM_RELIABILITY_V53
        or fixed.get("split_half_spearman_minimum")
        != design.MINIMUM_SPLIT_HALF_SPEARMAN_V53
        or fixed.get("estimated_signal_std_strictly_greater_than_fresh_max_actor_spread") is not True
        or fixed.get("no_retroactive_threshold_grid_seed_or_order_changes") is not True
        or fixed.get("exact_master_restore_and_quiescence_after_each_arm") is not True
        or any(authorization.get(key) is not False for key in (
            "optimizer_update", "projection", "train_gate", "candidate_snapshot",
            "ood_shadow_benchmark_or_holdout", "protected_semantics", "sealed_holdout",
        ))
        or authorization.get("gpu_launch") is not True
        or value.get("runtime") != base_design.RUNTIME_V52
        or value.get("fixed_recipe", {}).get("master_sha256")
        != base_design.MASTER_SHA256_V52
        or value.get("fixed_recipe", {}).get("dataset_sha256")
        != base_design.DATASET_SHA256_V52
        or value.get("protected_semantics_opened") is not False
        or value.get("sealed_holdout_opened") is not False
    ):
        raise RuntimeError("v53 preregistration contract changed")
    for binding in value.get("implementation_bindings", {}).values():
        if file_sha256(Path(binding["path"])) != binding["file_sha256"]:
            raise RuntimeError("v53 bound implementation changed")
    return value


def _write_self_hashed(path: Path, value: dict) -> dict:
    result = dict(value)
    result["content_sha256_before_self_field"] = design.canonical_sha256_v53(result)
    payload = (json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return result


def perturbation_integrity_v53(completed: list[dict], sigma: float) -> dict:
    expected = design.state_derivations_v53(sigma)
    if [item["state"] for item in completed] != [
        v52._runtime_state_v52(row, item["transition"])
        for row, item in zip(expected, completed, strict=True)
    ]:
        raise RuntimeError("v53 runtime state order changed")
    candidates = [item["transition"]["consensus"]["candidate_identity_sha256"] for item in completed]
    runtimes = [item["transition"]["consensus"]["runtime_values_sha256"] for item in completed]
    if (
        len(completed) != 32 or len(set(candidates)) != 32 or len(set(runtimes)) != 32
        or base_design.MASTER_SHA256_V52 in candidates
        or base_design.MASTER_RUNTIME_SHA256_V52 in runtimes
        or [(item["state"]["direction"], item["state"]["sign"]) for item in completed]
        != [(direction, sign) for direction in range(16) for sign in (1, -1)]
    ):
        raise RuntimeError("v53 perturbation identity safeguard failed")
    return {
        "schema": "sigma-perturbation-integrity-v53",
        "sigma": sigma,
        "states": 32,
        "unique_nonmaster_fp32_candidates": 32,
        "unique_nonmaster_bf16_candidates": 32,
        "four_actor_consensus_before_scoring": True,
        "direct_from_pinned_master": True,
        "exact_antithetic_seed_sign_coverage": True,
        "candidate_identity_inventory_sha256": design.canonical_sha256_v53(candidates),
        "runtime_identity_inventory_sha256": design.canonical_sha256_v53(runtimes),
    }


def _arm_from_completed(completed, final_restore, sigma, fresh_maximum, *, prior, v48b):
    scores = {label: [[None] * 4 for _ in range(16)] for label in ("plus", "minus")}
    receipts = []
    timings = []
    for item in completed:
        state = item["state"]
        for rank, score in enumerate(item["actor_scores"]):
            scores[state["label"]][state["direction"]][rank] = score
            receipts.append({
                "state_index": state["state_index"], "direction": state["direction"],
                "sign": state["label"], "actor_rank": rank, "sigma": sigma,
                "score_sha256": design.canonical_sha256_v53(score),
                "subset_content_sha256": v48b._SEALED_SUBSET["subset"]["content_sha256_before_self_field"],
                "request_order_sha256": v48b._SEALED_SUBSET["request_order_sha256"],
                "generation_params": v48b._SEALED_SUBSET["common_random_generation_params"],
            })
        timings.append({
            "state_index": state["state_index"], "state": state,
            "materialize": item["transition"], "generate": item["generation"],
            "score": item["score_timing"], "restore": item["restore_timing"],
            "drain": item["drain_timing"],
        })
    if any(value is None for sign in scores.values() for direction in sign for value in direction):
        raise RuntimeError("v53 signed score matrix incomplete")
    sign_scores = base_design.extract_arm_sign_scores_v52(scores, 16)
    reliability = design.reliability_gate_v53(
        prior._central_replicates(sign_scores["domain"]), fresh_maximum,
    )
    return {
        "schema": "sigma-discrimination-arm-v53", "sigma": sigma,
        "population_size": 16, "passed": reliability["passed"],
        "signed_scores": scores, "signed_scores_sha256": design.canonical_sha256_v53(scores),
        "signed_score_receipts": receipts,
        "signed_score_receipts_sha256": design.canonical_sha256_v53(receipts),
        "reliability": reliability,
        "perturbation_integrity": perturbation_integrity_v53(completed, sigma),
        "common_random_plan": v52._common_random_plan_v52(receipts, v48b._SEALED_SUBSET["subset"]),
        "timing": {"coverage": v52.validate_timing_coverage_v52(timings), "states": timings, "final_restore": final_restore},
        "projection_performed": False, "optimizer_update_or_train_gate_opened": False,
        "protected_semantics_opened": False, "sealed_holdout_opened": False,
    }


def replicated_sigma_discrimination_v53(
    trainer, bundle, dense_items, requests, anchors, master_sha, master_runtime_sha,
    fresh_calibration_observed_maximum, *, v51, v48b, prior, v40a,
):
    runtime = v52.RayPopulationOperationsV52(
        trainer, bundle, dense_items, requests, anchors, master_sha, master_runtime_sha,
        v51=v51, v48b=v48b, prior=prior, v40a=v40a,
    )
    arms = []
    try:
        for sigma in design.SIGMAS_V53:
            states = design.state_derivations_v53(sigma)
            original = base_design.state_derivations_v52
            base_design.state_derivations_v52 = lambda states=states: states
            try:
                completed, final_restore = v52.run_direct_master_pipeline_v52(states, runtime)
            finally:
                base_design.state_derivations_v52 = original
            arm = _arm_from_completed(
                completed, final_restore, sigma, fresh_calibration_observed_maximum,
                prior=prior, v48b=v48b,
            )
            certificates = v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            master = certificates[0]["current_identity"]
            arm["post_arm_exact_master_state"] = v52.validate_exact_master_state_certificates_v52(
                certificates, master, master_runtime_sha, phase=f"post_sigma_{sigma}",
                reference_generation=certificates[0]["reference_generation"],
                update_sequence=0, controller_transaction_quiescent=True,
            )
            artifact_path = RUN_DIR / f"sigma_{str(sigma).replace('.', 'p')}_arm_v53.json"
            persisted = prior._persist_phase(artifact_path, arm)
            arms.append({
                "sigma": sigma, "population_size": 16, "passed": arm["passed"],
                "reliability": arm["reliability"], "perturbation_integrity": arm["perturbation_integrity"],
                "artifact": {"path": str(artifact_path), "file_sha256": file_sha256(artifact_path), "content_sha256": persisted["content_sha256_before_self_field"]},
            })
            if arm["passed"]:
                break
    finally:
        runtime.close()
    selected = design.select_smallest_passing_sigma_v53(arms)
    return {
        "schema": "adaptive-sigma-discrimination-population-v53",
        "status": "complete_no_update_measurement",
        "sigma_grid": list(design.SIGMAS_V53), "evaluated_arms": arms,
        "selected_smallest_passing_sigma": selected,
        "stopped_after_first_pass": selected is not None,
        "all_completed_arms_persisted_before_stop": True,
        "fresh_calibration_observed_maximum_actor_spread": fresh_calibration_observed_maximum,
        "projection_performed": False, "optimizer_update_or_train_gate_opened": False,
        "protected_semantics_opened": False, "sealed_holdout_opened": False,
    }


def require_measurement_stop(population: dict) -> None:
    if (
        population.get("schema") != "adaptive-sigma-discrimination-population-v53"
        or population.get("all_completed_arms_persisted_before_stop") is not True
        or population.get("projection_performed") is not False
        or population.get("optimizer_update_or_train_gate_opened") is not False
    ):
        raise RuntimeError("v53 population persistence contract failed")
    raise V53MeasurementComplete("v53 no-update stop after persisted sigma discrimination")


@contextmanager
def patched_executor_v53():
    replacements = {
        "RUN_DIR": RUN_DIR, "ATTEMPT": ATTEMPT, "PREINSTALL_BASELINE": PREINSTALL,
        "MASTER_IDENTITY_AUDIT": MASTER_AUDIT, "NUMERIC_CALIBRATION": NUMERIC,
        "ANCHOR_CALIBRATION": ANCHOR, "P8_SNAPSHOT": RUN_DIR / "forbidden_p8_snapshot",
        "P16_SNAPSHOT": RUN_DIR / "forbidden_p16_snapshot",
        "replicated_population_v52": replicated_sigma_discrimination_v53,
        "require_all_arms_reliable_v52": require_measurement_stop,
    }
    saved = {key: getattr(v52, key) for key in replacements}
    for key, value in replacements.items():
        setattr(v52, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(v52, key, value)


def execute(preregistration: dict) -> int:
    try:
        with patched_executor_v53():
            v52._execute_v52(preregistration)
    except V53MeasurementComplete as error:
        failure = json.loads(FAILURE.read_text(encoding="utf-8"))
        population = json.loads(POPULATION.read_text(encoding="utf-8"))
        if (
            failure.get("message") != str(error)
            or failure.get("p8_train_gate") is not None
            or failure.get("p16_train_gate") is not None
            or failure.get("protected_semantics_opened") is not False
            or failure.get("sealed_holdout_opened") is not False
            or failure.get("final_gpu_idle", {}).get("all_four_compute_process_lists_empty") is not True
            or population.get("projection_performed") is not False
            or population.get("optimizer_update_or_train_gate_opened") is not False
        ):
            raise RuntimeError("v53 expected no-update cleanup receipt changed")
        report = _write_self_hashed(REPORT, {
            "schema": "lora-es-sigma-discrimination-report-v53",
            "status": "complete_no_update_no_protected_access",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "preregistration_content_sha256": preregistration["content_sha256_before_self_field"],
            "population": {"path": str(POPULATION), "file_sha256": file_sha256(POPULATION), "content_sha256": population["content_sha256_before_self_field"]},
            "internal_fail_closed_receipt": {"path": str(FAILURE), "file_sha256": file_sha256(FAILURE), "content_sha256": failure["content_sha256_before_self_field"]},
            "selected_smallest_passing_sigma": population["selected_smallest_passing_sigma"],
            "evaluated_arms": population["evaluated_arms"],
            "optimizer_update_projection_or_train_gate_opened": False,
            "protected_semantics_opened": False, "sealed_holdout_opened": False,
            "exact_cleanup_verified": True,
        })
        print(json.dumps({"report": str(REPORT), "file_sha256": file_sha256(REPORT), "content_sha256": report["content_sha256_before_self_field"], "selected_smallest_passing_sigma": report["selected_smallest_passing_sigma"]}, sort_keys=True))
        return 0
    raise RuntimeError("v53 executor returned without mandatory no-update stop")


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
        raise ValueError("v53 requires exactly one of --dry-run or --execute")
    prereg = load_preregistration(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"], "sigma_grid": list(design.SIGMAS_V53),
            "population_size": 16, "maximum_states": 96,
            "model_ray_gpu_or_train_semantics_loaded": False,
            "filesystem_writes": False, "optimizer_update_authorized": False,
            "protected_semantics_loaded": False, "sealed_holdout_opened": False,
        }, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != base_design.REQUIRED_PYTHON_V52:
        raise RuntimeError("v53 requires the sealed es-at-scale interpreter")
    return execute(prereg)


if __name__ == "__main__":
    raise SystemExit(main())
