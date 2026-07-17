#!/usr/bin/env python3
"""Seal V66c after adding the mandatory four-actor LoRA activation edge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import build_lora_es_mirrored_calibration_preregistration_v66 as v66
import build_lora_es_mirrored_calibration_preregistration_v66b as v66b


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_mirrored_calibration_v66c.json"
).resolve()
FAILED_V66B_RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v66b_lora_es_mirrored_crn_qwen36_calibration"
).resolve()
FAILED_V66B_ATTEMPT = (
    FAILED_V66B_RUN.parent
    / ".v66b_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
).resolve()
FAILED_V66B_RECEIPT = (FAILED_V66B_RUN / "failure_v66b.json").resolve()
RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v66c_lora_es_mirrored_crn_qwen36_calibration"
).resolve()


def artifacts_v66c() -> dict:
    return {
        "attempt": str(
            RUN.parent
            / ".v66c_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
        ),
        "run_directory": str(RUN),
        "gpu_log": str(RUN / "gpu_activity_v66c.jsonl"),
        "population": str(RUN / "mirrored_population_v66c.json"),
        "update": str(RUN / "pair_difference_update_v66c.json"),
        "report": str(RUN / "mirrored_calibration_report_v66c.json"),
        "failure": str(RUN / "failure_v66c.json"),
    }


def _failed_v66b_receipt_v66c() -> dict:
    if not FAILED_V66B_ATTEMPT.is_file() or not FAILED_V66B_RECEIPT.is_file():
        raise RuntimeError("v66c requires immutable failed V66b receipts")
    attempt = json.loads(FAILED_V66B_ATTEMPT.read_text(encoding="utf-8"))
    failure = json.loads(FAILED_V66B_RECEIPT.read_text(encoding="utf-8"))
    if (
        attempt.get("content_sha256_before_self_field")
        != "e1d020f08d7e605ff6728f3523686c179dfaccbee0bb8d0a6a3eda31e3561142"
        or failure.get("content_sha256_before_self_field")
        != "6d1bb225e7e5d2d6e9e99fd8230bbe4d4727cb1b065646cb77e5bede1baab563"
        or failure.get("type") != "RayTaskError(RuntimeError)"
        or failure.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        )
        is not True
        or failure.get("protected_dev_ood_or_holdout_opened") is not False
        or failure.get("checkpoint_snapshot_or_promotion_performed") is not False
    ):
        raise RuntimeError("failed V66b evidence changed")
    return {
        "attempt_path": str(FAILED_V66B_ATTEMPT),
        "attempt_file_sha256": v66.file_sha256_v66(FAILED_V66B_ATTEMPT),
        "attempt_content_sha256": attempt[
            "content_sha256_before_self_field"
        ],
        "failure_path": str(FAILED_V66B_RECEIPT),
        "failure_file_sha256": v66.file_sha256_v66(FAILED_V66B_RECEIPT),
        "failure_content_sha256": failure[
            "content_sha256_before_self_field"
        ],
        "failure_type": failure["type"],
        "charged_gpu_seconds": failure["compute_ledger"][
            "charged_gpu_seconds"
        ],
        "final_gpu_idle": True,
        "trainer_constructed_in_external_lora_mode": True,
        "adapter_activation_reached": False,
        "adapter_install_reached": True,
        "candidate_evaluation_reached": False,
        "candidate_update_reached": False,
        "failure_reason": "canonical install preceded explicit add_lora",
    }


def build_preregistration_v66c() -> dict:
    result = v66.build_preregistration_v66()
    result.pop("content_sha256_before_self_field", None)
    result["schema"] = (
        "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66c"
    )
    result["status"] = (
        "sealed_after_four_actor_lora_activation_fix_before_v66c_gpu_access"
    )
    result["purpose"] = (
        "Fresh one-shot retry of V66 mirrored pair-difference calibration "
        "with an explicit add_lora, active-slot proof, canonical-state install "
        "sequence on every actor."
    )
    result["supersedes_failed_attempts"] = {
        "v66": v66b._failed_attempt_receipt_v66b(),
        "v66b": _failed_v66b_receipt_v66c(),
    }
    result["artifacts"] = artifacts_v66c()
    result["fixed_recipe"]["lora_activation"] = {
        "request_name": "matched_lora_initialization_v41b",
        "request_id": 1,
        "request_path": str(v66.STAGED_ADAPTER),
        "four_actor_add_lora_required": True,
        "active_slot_certificate_required_before_state_write": True,
        "active_slot_index": 0,
    }
    result["runtime"].update({
        "trainer_state_mode": "external_worker",
        "dense_full_weight_master_install_authorized": False,
        "required_worker_endpoints": [
            "install_adapter_state_v41a",
            "active_lora_slot_certificate_v66",
            "materialize_mirrored_adapter_v66",
            "restore_mirrored_adapter_v66",
        ],
    })
    result["acceptance"].update({
        "trainer_constructs_without_dense_master_rpc": True,
        "all_four_add_lora_return_true_before_state_write": True,
        "all_four_active_slot_certificates_are_adapter_one": True,
        "canonical_lora_install_reached_after_activation": True,
    })
    bindings = result["implementation_bindings"]
    bindings["v66_builder"] = bindings.pop("builder")
    bindings["v66_engine"] = bindings.pop("runner")
    paths = {
        "builder": Path(__file__).resolve(),
        "runner": ROOT / "run_lora_es_mirrored_calibration_v66c.py",
        "base_trainer": ROOT / "train_eggroll_es_specialist.py",
        "failed_v66_attempt": v66b.FAILED_ATTEMPT,
        "failed_v66_receipt": v66b.FAILED_RECEIPT,
        "failed_v66b_attempt": FAILED_V66B_ATTEMPT,
        "failed_v66b_receipt": FAILED_V66B_RECEIPT,
    }
    bindings.update({
        key: {
            "path": str(path.resolve()),
            "file_sha256": v66.file_sha256_v66(path),
        }
        for key, path in paths.items()
    })
    result["content_sha256_before_self_field"] = v66.canonical_sha256_v66(
        result
    )
    return result


def write_preregistration_v66c(path: Path = OUTPUT) -> dict:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    value = build_preregistration_v66c()
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    value = build_preregistration_v66c()
    if args.check:
        if not output.is_file() or json.loads(
            output.read_text(encoding="utf-8")
        ) != value:
            raise RuntimeError("v66c preregistration is absent or stale")
    else:
        write_preregistration_v66c(output)
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
