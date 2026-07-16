#!/usr/bin/env python3
"""Fail-closed aggregate-only finalizer for the completed V59 train run."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

from safetensors import safe_open

import build_lora_es_fragile_priority_preregistration_v59 as builder
import lora_es_nested_population_v52 as design52


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "experiments/eggroll_es_hpo/runs/v59_lora_es_fragile_priority"
ATTEMPT = ROOT / "experiments/eggroll_es_hpo/runs/.v59_lora_es_fragile_priority.attempt.json"
PREREG = ROOT / "experiments/eggroll_es_hpo/preregistrations/lora_es_fragile_priority_v59.json"
P16 = RUN / "p16_train_gate_v52.json"
INTERNAL = RUN / "nested_population_report_v52.json"
COMPAT = RUN / "compatibility_report_v55b.json"
GPU_LOG = RUN / "gpu_activity_v59.jsonl"
SNAPSHOT = RUN / "selected_candidate_v59"
OUTPUT = RUN / "v59_evidence_manifest.json"
RUNNER = ROOT / "run_lora_es_fragile_priority_v59.py"
DEDICATED_REPORT = RUN / "fragile_priority_report_v59.json"
EXPECTED_PREREG_FILE = "b299d8cb0d6f16a6a00cd00c66cb70e458a8a3c734191a5674383f49be08d997"
EXPECTED_PREREG_CONTENT = "e21bb2dd54b4e557d04b9989748f3475fbbdf9c498611a193a1d407bde34c49b"
EXPECTED_ATTEMPT_FILE = "a4478cb7ea779284e7e55e35b6c5c914bf0a815e7642852018fb957dbf60b0c2"
EXPECTED_ATTEMPT_CONTENT = "b3cc8d56633f0e41fd4deaf1f8cd3399b98f630e7edcc7ff192f7cafa6ef35dd"
EXPECTED_RUNNER_FILE = "d5d288a8b541d4a7c6d7b98b284839b5e70ddf3a04f62f79e709a768245658ad"
EXPECTED_DIRECTION = "f80db1bde940053c93e559c9305dc750e82f5231d8df4f52fcf1b85c06dc0522"
EXPECTED_WEIGHTS = "c2665b60928b16120a2b98fdf137fafd250644852c86a02d797689f02105c6c8"
EXPECTED_CONFIG = "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5"
EXPECTED_CANDIDATE = "1713987fcad93f3e6368a309415faf5de2f4230eaf3c44baf23b8e9a2edf2a3d"
EXPECTED_RUNTIME = "ad5dd995de7cad3c9d116d64deb3aa67b9db46fbdf4e3f8a6ab5ee37340b5923"
EXPECTED_MANIFEST = "e477cac0ed5fbed2cf106d0a5640648a9a533e683e94c06d4571e57cef5c85d7"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash_valid(value: dict) -> bool:
    content = value.get("content_sha256_before_self_field")
    compact = {key: item for key, item in value.items()
               if key != "content_sha256_before_self_field"}
    return content == design52.canonical_sha256_v52(compact)


def _phase_summary(rows: list[dict], phase: str, *, require_positive: bool) -> dict:
    by_gpu = {}
    for gpu in range(4):
        selected = [row for row in rows if row["phase"] == phase and row["gpu"] == gpu]
        if not selected:
            raise RuntimeError(f"V59 missing GPU {gpu} phase {phase}")
        pids = {row["expected_pid"] for row in selected}
        if len(pids) != 1:
            raise RuntimeError("V59 expected PID changed")
        pid = next(iter(pids))
        resident = all(pid in row["compute_pids"] for row in selected)
        positive = sum(row["utilization_percent"] > 0 for row in selected)
        if not resident or (require_positive and positive == 0):
            raise RuntimeError(f"V59 GPU activity contract failed: {gpu} {phase}")
        by_gpu[str(gpu)] = {
            "expected_pid": pid,
            "samples": len(selected),
            "positive_utilization_samples": positive,
            "peak_utilization_percent": max(row["utilization_percent"] for row in selected),
            "mean_utilization_percent": math.fsum(
                row["utilization_percent"] for row in selected
            ) / len(selected),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in selected),
            "expected_pid_resident_all_samples": True,
        }
    return {"phase": phase, "by_gpu": by_gpu}


def _live_idle() -> dict:
    query = subprocess.run([
        "nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True).stdout.splitlines()
    memory = {line.split(",")[0].strip(): int(line.split(",")[1]) for line in query}
    compute = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], check=True, capture_output=True, text=True).stdout.strip()
    if memory != {"0": 4, "1": 4, "2": 4, "3": 4} or compute:
        raise RuntimeError("V59 final GPU cleanup changed")
    return {"memory_used_mib": memory, "all_four_compute_process_lists_empty": True}


def build_evidence_v59() -> dict:
    if DEDICATED_REPORT.exists():
        raise RuntimeError("V59 dedicated wrapper report unexpectedly exists")
    prereg, attempt, p16, internal, compat = map(
        _read, (PREREG, ATTEMPT, P16, INTERNAL, COMPAT),
    )
    if (
        builder.file_sha256(PREREG) != EXPECTED_PREREG_FILE
        or prereg.get("content_sha256_before_self_field") != EXPECTED_PREREG_CONTENT
        or not _self_hash_valid(prereg)
        or builder.file_sha256(ATTEMPT) != EXPECTED_ATTEMPT_FILE
        or attempt.get("content_sha256_before_self_field") != EXPECTED_ATTEMPT_CONTENT
        or not _self_hash_valid(attempt)
        or attempt.get("schema") != "nested-population-attempt-v52"
        or attempt.get("status") != "launching_train_only"
        or attempt.get("phase") != "before_train_semantics_model_or_gpu_load"
        or attempt.get("preregistration_content_sha256") != EXPECTED_PREREG_CONTENT
        or attempt.get("protected_semantics_opened") is not False
        or attempt.get("sealed_holdout_opened") is not False
        or builder.file_sha256(RUNNER) != EXPECTED_RUNNER_FILE
        or prereg.get("fragile_priority_projection", {}).get("direction_sha256")
        != EXPECTED_DIRECTION
        or p16.get("schema") != "fragile-maximin-p16-backtracking-gate-v55b"
        or internal.get("schema") != "nested-population-train-only-report-v52"
        or compat.get("schema") != "lora-es-fragile-maximin-backtracking-report-v55b"
        or not all(_self_hash_valid(value) for value in (p16, internal, compat))
    ):
        raise RuntimeError("V59 sealed artifact identity changed")
    results = p16["scale_results"]
    selected = results[-1]
    checks = selected["checks"]
    metrics = selected["gate"]["metrics"]
    projection_spec = prereg["fragile_priority_projection"]
    if (
        [row["target_norm_ratio"] for row in results] != [0.5, 0.375, 0.25]
        or [row["passed"] for row in results] != [False, False, True]
        or p16.get("selected_target_norm_ratio") != 0.25
        or set(checks) != set(design52.TRAIN_GATE_NAMES_V52)
        or set(checks.values()) != {True}
        or selected.get("candidate_consensus_passed") is not True
        or any(row.get("exact_abort_readback_passed") is not True for row in results)
        or p16.get("all_candidates_exactly_aborted_to_common_master") is not True
        or p16.get("master_committed") is not False
        or p16.get("population_rerun_performed") is not False
        or p16.get("p8_evaluation_performed") is not False
        or p16.get("protected_semantics_opened") is not False
        or p16.get("sealed_holdout_opened") is not False
    ):
        raise RuntimeError("V59 scientific PASS contract changed")
    snapshot = p16["selected_snapshot"]
    workers = snapshot["workers"]
    written = [item for item in workers if item["written"] is True]
    weights = SNAPSHOT / "adapter_model.safetensors"
    config = SNAPSHOT / "adapter_config.json"
    with safe_open(weights, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if (
        snapshot.get("schema") != "four-actor-uncommitted-snapshot-consensus-v52"
        or snapshot.get("readback_verified") is not True
        or snapshot.get("master_committed") is not False
        or snapshot.get("runtime_values_sha256") != EXPECTED_RUNTIME
        or snapshot.get("candidate_identity", {}).get("sha256") != EXPECTED_CANDIDATE
        or snapshot.get("candidate_identity", {}).get("tensor_count") != 70
        or snapshot.get("candidate_identity", {}).get("elements") != 4_528_128
        or len(workers) != 4 or len(written) != 1 or written[0]["rank"] != 0
        or any(item.get("manifest_sha256") != EXPECTED_MANIFEST for item in workers)
        or any(item.get("exact_abort_required_after_snapshot") is not True for item in workers)
        or any(item.get("master_committed") is not False for item in workers)
        or builder.file_sha256(weights) != EXPECTED_WEIGHTS
        or builder.file_sha256(config) != EXPECTED_CONFIG
        or written[0].get("weights_sha256") != EXPECTED_WEIGHTS
        or written[0].get("config_sha256") != EXPECTED_CONFIG
        or metadata.get("schema") != "uncommitted-canonical-peft-fp32-v52"
        or metadata.get("candidate_sha256") != EXPECTED_CANDIDATE
        or metadata.get("manifest_sha256") != EXPECTED_MANIFEST
    ):
        raise RuntimeError("V59 snapshot provenance changed")
    if (
        internal.get("optimizer_master_committed") is not False
        or internal.get("protected_semantics_opened") is not False
        or internal.get("final_gpu_idle", {}).get("all_four_compute_process_lists_empty") is not True
        or compat.get("selected_target_norm_ratio") != 0.25
        or compat.get("passing_candidate_saved") is not True
        or compat.get("all_candidates_exactly_aborted_to_master") is not True
        or compat.get("ood_shadow_benchmark_holdout_or_protected_semantics_opened") is not False
        or compat.get("sealed_holdout_opened") is not False
    ):
        raise RuntimeError("V59 exact-state cleanup contract changed")
    rows = [json.loads(line) for line in GPU_LOG.read_text(encoding="utf-8").splitlines()]
    if any(row["foreign_compute_pids"] for row in rows):
        raise RuntimeError("V59 foreign GPU process observed")
    projection = _phase_summary(
        rows, "v59_fragile_priority_projection_no_population_rerun",
        require_positive=False,
    )
    p16_gpu = _phase_summary(rows, "p16_backtracking_train_gate", require_positive=True)
    failed_checks = {
        str(item["target_norm_ratio"]): sorted(
            name for name, passed in item["checks"].items() if passed is not True
        )
        for item in results
    }
    paired_actor_median_deltas = {
        "domain": metrics["domain"]["median_paired_delta"],
        "prose_lm": metrics["prose_lm"]["median_paired_delta"],
        "qa_answer_logprob": metrics["qa_answer_logprob"]["median_paired_delta"],
        "qa_generation_f1": metrics["qa_generation_f1"]["median_paired_delta"],
        "qa_generation_exact_count": metrics["qa_generation_exact"]["median_paired_delta"],
        "qa_generation_nonzero_count": metrics["qa_generation_nonzero"]["median_paired_delta"],
    }
    result = {
        "schema": "lora-es-fragile-priority-evidence-v59",
        "status": "complete_scientific_train_pass_candidate_saved_wrapper_telemetry_false_negative",
        "completed_at_utc": internal["completed_at_utc"],
        "preregistration": {"file_sha256": EXPECTED_PREREG_FILE, "content_sha256": EXPECTED_PREREG_CONTENT},
        "attempt": {"file_sha256": EXPECTED_ATTEMPT_FILE, "content_sha256": EXPECTED_ATTEMPT_CONTENT},
        "projection": {
            "direction_sha256": EXPECTED_DIRECTION,
            "minimum_fragile_f1_actor_margin": projection_spec["minimum_fragile_margin"],
            "minimum_other_actor_objective_margin": projection_spec["minimum_other_margin"],
            "required_other_margin_floor": projection_spec["solution"]["required_other_margin_floor"],
            "v58_direction_cosine": projection_spec["v58_direction_cosine"],
            "maximum_kkt_stationarity_residual": projection_spec["solution"]["maximum_stationarity_residual"],
            "maximum_kkt_complementarity_residual": projection_spec["solution"]["maximum_complementarity_residual"],
            "per_objective_minimum_projected_margins": projection_spec["solution"]["per_objective_minimum_margins"],
        },
        "scientific_result": {
            "selected_target_norm_ratio": 0.25,
            "evaluated_ratio_prefix": [0.5, 0.375, 0.25],
            "per_ratio_failed_checks": failed_checks,
            "ratio_0p25_gate_checks": checks,
            "ratio_0p25_paired_actor_median_deltas": paired_actor_median_deltas,
            "all_nine_calibrated_endpoint_gates_passed": True,
            "four_actor_candidate_consensus_passed": True,
            "train_only_pass": True,
        },
        "candidate_snapshot": {
            "directory": str(SNAPSHOT),
            "adapter_model_file_sha256": EXPECTED_WEIGHTS,
            "adapter_config_file_sha256": EXPECTED_CONFIG,
            "canonical_fp32_candidate_sha256": EXPECTED_CANDIDATE,
            "runtime_bf16_values_sha256": EXPECTED_RUNTIME,
            "snapshot_manifest_sha256": EXPECTED_MANIFEST,
            "tensor_count": 70, "elements": 4_528_128,
            "worker_consensus_count": 4, "readback_verified": True,
            "candidate_exactly_aborted_after_snapshot": True,
            "optimizer_master_committed": False,
        },
        "instrumentation_status": {
            "runner_exit_code": 1,
            "scientific_executor_and_snapshot_completed_before_wrapper_error": True,
            "error": "projection no-rerun sleep incorrectly required positive utilization on every GPU",
            "runner_file_sha256": EXPECTED_RUNNER_FILE,
            "dedicated_wrapper_report_written": False,
            "projection_phase_correct_requirement": "all four expected actor PIDs resident; no foreign compute",
            "p16_phase_correct_requirement": "all four expected actor PIDs resident and positive",
            "selection_gate_or_candidate_bytes_changed_by_finalizer": False,
        },
        "telemetry": {
            "gpu_log_file_sha256": builder.file_sha256(GPU_LOG),
            "projection_no_work_barrier": projection,
            "p16_train_gate": p16_gpu,
            "foreign_compute_pid_rows": 0,
        },
        "artifacts": {
            "p16_train_gate": {"file_sha256": builder.file_sha256(P16), "content_sha256": p16["content_sha256_before_self_field"]},
            "internal_report": {"file_sha256": builder.file_sha256(INTERNAL), "content_sha256": internal["content_sha256_before_self_field"]},
            "compatibility_report": {"file_sha256": builder.file_sha256(COMPAT), "content_sha256": compat["content_sha256_before_self_field"]},
        },
        "final_gpu_idle": _live_idle(),
        "optimizer_master_committed": False,
        "population_rerun_performed": False,
        "p8_evaluation_performed": False,
        "protected_semantics_opened": False,
        "ood_shadow_opened": False,
        "terminal_holdout_opened": False,
        "document_disjoint_evaluation_doctrine_preserved": True,
    }
    result["content_sha256_before_self_field"] = design52.canonical_sha256_v52(result)
    return result


def main() -> int:
    value = build_evidence_v59()
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, OUTPUT)
    print(json.dumps({"path": str(OUTPUT), "file_sha256": builder.file_sha256(OUTPUT), "content_sha256": value["content_sha256_before_self_field"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
