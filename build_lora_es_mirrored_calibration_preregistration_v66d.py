#!/usr/bin/env python3
"""Seal the fresh V66d telemetry-race repair calibration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import build_lora_es_mirrored_calibration_preregistration_v66 as v66
import build_lora_es_mirrored_calibration_preregistration_v66c as v66c


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_mirrored_calibration_v66d.json"
).resolve()
FAILED_V66C_RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v66c_lora_es_mirrored_crn_qwen36_calibration"
).resolve()
FAILED_V66C_ATTEMPT = (
    FAILED_V66C_RUN.parent
    / ".v66c_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
).resolve()
FAILED_V66C_RECEIPT = (FAILED_V66C_RUN / "failure_v66c.json").resolve()
FAILED_V66C_GPU_LOG = (FAILED_V66C_RUN / "gpu_activity_v66c.jsonl").resolve()
FAILED_V66C_POPULATION = (
    FAILED_V66C_RUN / "mirrored_population_v66c.json"
).resolve()
FAILED_V66C_UPDATE = (
    FAILED_V66C_RUN / "pair_difference_update_v66c.json"
).resolve()
RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v66d_lora_es_mirrored_crn_qwen36_calibration"
).resolve()


def artifacts_v66d():
    return {
        "attempt": str(
            RUN.parent
            / ".v66d_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
        ),
        "run_directory": str(RUN),
        "gpu_log": str(RUN / "gpu_activity_v66d.jsonl"),
        "actor_cuda_work_log": str(
            RUN / "actor_cuda_work_receipts_v66d.jsonl"
        ),
        "population": str(RUN / "mirrored_population_v66d.json"),
        "update": str(RUN / "pair_difference_update_v66d.json"),
        "report": str(RUN / "mirrored_calibration_report_v66d.json"),
        "failure": str(RUN / "failure_v66d.json"),
    }


def _failed_v66c_receipt_v66d():
    required = (
        FAILED_V66C_ATTEMPT,
        FAILED_V66C_RECEIPT,
        FAILED_V66C_GPU_LOG,
        FAILED_V66C_POPULATION,
        FAILED_V66C_UPDATE,
    )
    if any(not path.is_file() for path in required):
        raise RuntimeError("v66d requires immutable substantive V66c artifacts")
    attempt = json.loads(FAILED_V66C_ATTEMPT.read_text(encoding="utf-8"))
    failure = json.loads(FAILED_V66C_RECEIPT.read_text(encoding="utf-8"))
    population = json.loads(FAILED_V66C_POPULATION.read_text(encoding="utf-8"))
    update = json.loads(FAILED_V66C_UPDATE.read_text(encoding="utf-8"))
    if (
        v66.file_sha256_v66(FAILED_V66C_ATTEMPT)
        != "29c34e10caf32576bdc81fc6f5676b2e1e3bd5cda0ddd142491743843c4c00d9"
        or attempt.get("content_sha256_before_self_field")
        != "a43f08fce8a8e07cfa8e39c75db0ee2860d3780d23039d676f06715cc40b6ac1"
        or v66.file_sha256_v66(FAILED_V66C_RECEIPT)
        != "6468175dea71f5793207ffb0f2ad1f891c19035f73e260de6648009dd89b8624"
        or failure.get("content_sha256_before_self_field")
        != "fd15274cc07e25dd58e37ae79569215b51b74fc18f97a3034a4936d7b014f4bb"
        or failure.get("type") != "RuntimeError"
        or failure.get("message")
        != "v66 GPU 0 lacked useful activity in mirrored wave 1"
        or failure.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or failure.get("protected_dev_ood_or_holdout_opened") is not False
        or failure.get("checkpoint_snapshot_or_promotion_performed") is not False
        or v66.file_sha256_v66(FAILED_V66C_GPU_LOG)
        != "37abe8c11a3f8936e22fce611009f0f1b10b4d54687c3604219bca8355c34064"
        or v66.file_sha256_v66(FAILED_V66C_POPULATION)
        != "5543d2887f932f90201cb2aa4c92c996eed0b02d1353b0cd370004e3acfb4ebb"
        or population.get("content_sha256_before_self_field")
        != "31e0044615851a692847bcba9c1b92d1c3625f81d36c8789816ea646ca43506d"
        or v66.file_sha256_v66(FAILED_V66C_UPDATE)
        != "d1677705419f21e00c6e9ece74145afc4a03d28e7f4cdcbf85116b47b671293a"
        or update.get("content_sha256_before_self_field")
        != "abff89fa3e456ebcfffe60577cab1e6dac950f51679b89ac589ec800b3fe347e"
        or update.get("nonzero_pair_differences") != 8
        or update.get("coefficient_l2") != 0.0042338077269223435
        or update.get("master_committed") is not False
        or update.get("all_four_abort_receipts_exact") is not True
    ):
        raise RuntimeError("substantive V66c evidence changed")
    return {
        "attempt_path": str(FAILED_V66C_ATTEMPT),
        "attempt_file_sha256": v66.file_sha256_v66(FAILED_V66C_ATTEMPT),
        "attempt_content_sha256": attempt[
            "content_sha256_before_self_field"
        ],
        "failure_path": str(FAILED_V66C_RECEIPT),
        "failure_file_sha256": v66.file_sha256_v66(FAILED_V66C_RECEIPT),
        "failure_content_sha256": failure[
            "content_sha256_before_self_field"
        ],
        "gpu_log_file_sha256": v66.file_sha256_v66(FAILED_V66C_GPU_LOG),
        "population_file_sha256": v66.file_sha256_v66(
            FAILED_V66C_POPULATION
        ),
        "population_content_sha256": population[
            "content_sha256_before_self_field"
        ],
        "update_file_sha256": v66.file_sha256_v66(FAILED_V66C_UPDATE),
        "update_content_sha256": update[
            "content_sha256_before_self_field"
        ],
        "signed_candidates_completed": 16,
        "mirrored_waves_completed": 4,
        "nonzero_pair_differences": 8,
        "coefficient_l2": update["coefficient_l2"],
        "nonzero_fp32_and_bf16_candidate_executed": True,
        "master_committed": False,
        "all_four_exact_abort_receipts": True,
        "final_gpu_idle": True,
        "charged_gpu_seconds": failure["compute_ledger"][
            "charged_gpu_seconds"
        ],
        "failure_scope": "short-phase NVML label sampling only",
        "state_or_es_protocol_failure": False,
    }


def build_preregistration_v66d():
    result = v66c.build_preregistration_v66c()
    result.pop("content_sha256_before_self_field", None)
    result["schema"] = (
        "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66d"
    )
    result["status"] = (
        "sealed_after_v66c_short_phase_telemetry_race_before_v66d_gpu_access"
    )
    result["purpose"] = (
        "Fresh V66c-equivalent mirrored calibration with a phase transition "
        "barrier acknowledged only after all four NVML rows and an actor-side "
        "CUDA-event/output-cardinality receipt for every signed candidate."
    )
    result["supersedes_failed_attempts"] = {
        **result["supersedes_failed_attempts"],
        "v66c": _failed_v66c_receipt_v66d(),
    }
    result["artifacts"] = artifacts_v66d()
    result["fixed_recipe"]["gpu_work_attribution_v66d"] = {
        "sample_interval_seconds": 0.25,
        "phase_transition_barrier": (
            "monitor must flush one row for each physical GPU before work"
        ),
        "acknowledgement_gpu_set": [0, 1, 2, 3],
        "actor_receipt_per_signed_candidate": True,
        "actor_receipt_binding": [
            "engine_rank", "worker_pid", "physical_gpu_id", "work_id",
        ],
        "cuda_event_elapsed_must_be_positive": True,
        "request_outputs_per_candidate": 64,
        "samples_per_candidate": 64,
        "generated_tokens_per_candidate": 64,
        "nvml_positive_tick_required_when_actor_receipt_valid": False,
        "nvml_residency_and_no_foreign_process_required": True,
    }
    result["runtime"].update({
        "worker_extension": (
            "eggroll_es_worker_lora_v66d."
            "LoRAAdapterStateWorkerExtensionV66D"
        ),
        "required_worker_endpoints": [
            "install_adapter_state_v41a",
            "active_lora_slot_certificate_v66",
            "materialize_mirrored_adapter_v66",
            "begin_actor_gpu_work_v66d",
            "end_actor_gpu_work_v66d",
            "restore_mirrored_adapter_v66",
        ],
    })
    result["acceptance"].update({
        "each_generation_phase_acknowledged_after_all_four_gpu_rows": True,
        "every_signed_candidate_has_actor_cuda_event_receipt": True,
        "every_receipt_bound_to_exact_rank_pid_and_physical_gpu": True,
        "every_receipt_has_positive_output_and_token_cardinality": True,
        "short_phase_may_pass_without_positive_nvml_tick": True,
        "resident_but_idle_actor_must_fail": True,
    })
    result["beads"] = [
        "specialist-0j5.2", "specialist-0j5.12", "specialist-nen.25",
    ]
    bindings = result["implementation_bindings"]
    bindings["v66c_builder"] = bindings.pop("builder")
    bindings["v66c_runner"] = bindings.pop("runner")
    paths = {
        "builder": Path(__file__).resolve(),
        "runner": ROOT / "run_lora_es_mirrored_calibration_v66d.py",
        "worker": ROOT / "eggroll_es_worker_lora_v66d.py",
        "gpu_telemetry": ROOT / "eggroll_es_gpu_telemetry_v66.py",
        "failed_v66c_attempt": FAILED_V66C_ATTEMPT,
        "failed_v66c_receipt": FAILED_V66C_RECEIPT,
        "failed_v66c_gpu_log": FAILED_V66C_GPU_LOG,
        "failed_v66c_population": FAILED_V66C_POPULATION,
        "failed_v66c_update": FAILED_V66C_UPDATE,
    }
    bindings.update({
        key: {
            "path": str(path.resolve()),
            "file_sha256": v66.file_sha256_v66(path),
        }
        for key, path in paths.items()
    })
    result["content_sha256_before_self_field"] = (
        v66.canonical_sha256_v66(result)
    )
    return result


def write_preregistration_v66d(path=OUTPUT):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    value = build_preregistration_v66d()
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
    ).encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
    return value


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    value = build_preregistration_v66d()
    if args.check:
        if (
            not output.is_file()
            or json.loads(output.read_text(encoding="utf-8")) != value
        ):
            raise RuntimeError("v66d preregistration is absent or stale")
    else:
        write_preregistration_v66d(output)
    print(json.dumps({
        "path": str(output),
        "file_sha256": (
            v66.file_sha256_v66(output) if output.is_file() else None
        ),
        "content_sha256": value["content_sha256_before_self_field"],
        "checked": bool(args.check),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
