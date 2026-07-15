#!/usr/bin/env python3
"""Freeze the V42B direct-load retry of the matched-init SFT control."""

from __future__ import annotations

import copy
from pathlib import Path

import build_sft_equal_unit_matched_init_preregistration_v42a as v42a_builder
import run_sft_equal_unit_matched_init_v42b as launcher
import run_sft_train_only_control_v36a as engine
import sft_lora_equal_unit_matched_init_v42b as sft


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/sft_controls/"
    "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load"
).resolve()
OUTPUT_DIR = RUN_DIR / "middle_late_r32_seed17_init20260715041"
PREREGISTRATION = RUN_DIR / "preregistration_v42b.json"
EXPECTED_IMPLEMENTATION = {
    "launcher_sha256": (
        "6ec5aed549381e036c3b476528f4cf1a1355e2f1d47c42d325642b4a9be2f26e"
    ),
    "sft_sha256": (
        "13c087ef1bc9074e95c1fc36d52355b48676873a0e5072e9b720a0e55c82cf80"
    ),
}


def arguments() -> object:
    args = v42a_builder.arguments()
    args.output_dir = str(OUTPUT_DIR)
    args.stdout_log = str(RUN_DIR / "stdout_v42b.log")
    args.gpu_log = str(RUN_DIR / "gpu_activity_v42b.jsonl")
    args.report = str(RUN_DIR / "runtime_report_v42b.json")
    args.attempt_report = str(RUN_DIR / "attempt_v42b.json")
    args.preregistration = str(PREREGISTRATION)
    args.preregistration_sha256 = "PENDING"
    args.preregistration_content_sha256 = "PENDING"
    return args


def build() -> dict:
    predecessor = v42a_builder.build()
    args = arguments()
    observed_implementation = {
        "launcher_sha256": engine.file_sha256(
            ROOT / "run_sft_equal_unit_matched_init_v42b.py"
        ),
        "sft_sha256": engine.file_sha256(
            ROOT / "sft_lora_equal_unit_matched_init_v42b.py"
        ),
    }
    if observed_implementation != EXPECTED_IMPLEMENTATION:
        raise RuntimeError("V42B implementation binding changed")
    result = copy.deepcopy(predecessor)
    result.pop("content_sha256_before_self_field", None)
    result.update({
        "schema": "specialist-sft-matched-init-equal-unit-preregistration-v42b",
        "status": "preregistered_retry_not_yet_run",
        "experiment_name": (
            "sft_matched_init_equal_unit_v42b_fold3_v412_"
            "middle_late_r32_seed17_init20260715041_direct_load_retry"
        ),
        "adapter_loader": sft.expected_loader_audit_v42b(),
        "cpu_model_load_smoke": launcher.validate_cpu_model_load_smoke_v42b(),
        "predecessor_failure": launcher.validate_failed_predecessor_v42b(),
        "artifacts": {
            "output_dir": str(OUTPUT_DIR),
            "stdout_log": str(RUN_DIR / "stdout_v42b.log"),
            "gpu_log": str(RUN_DIR / "gpu_activity_v42b.jsonl"),
            "report": str(RUN_DIR / "runtime_report_v42b.json"),
            "attempt_report": str(RUN_DIR / "attempt_v42b.json"),
        },
    })
    result["implementation"] = {
        **predecessor["implementation"],
        "launcher": str(
            (ROOT / "run_sft_equal_unit_matched_init_v42b.py").resolve()
        ),
        "launcher_sha256": EXPECTED_IMPLEMENTATION["launcher_sha256"],
        "sft": str(
            (ROOT / "sft_lora_equal_unit_matched_init_v42b.py").resolve()
        ),
        "sft_sha256": EXPECTED_IMPLEMENTATION["sft_sha256"],
        "v42a_source_contract": str(
            (ROOT / "sft_lora_equal_unit_matched_init_v42a.py").resolve()
        ),
        "v42a_source_contract_sha256": predecessor["implementation"][
            "sft_sha256"
        ],
    }
    result["recipe"]["command"] = launcher.build_train_command(args)
    result["recipe"]["adapter_loader"] = result["adapter_loader"]
    result["recipe"]["only_change_from_v42a"] = (
        "replace PeftModel.from_pretrained conversion with get_peft_model plus "
        "exact in-place canonical FP32 state copy and immediate readback"
    )
    result["selection_firewall"]["this_arm_authorizes"] = (
        "retry training state and runtime evidence only"
    )
    result["content_sha256_before_self_field"] = engine.canonical_sha256(result)
    return result


def main() -> None:
    result = build()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if PREREGISTRATION.exists():
        raise RuntimeError("V42B preregistration already exists")
    engine.atomic_write_json(PREREGISTRATION, result)
    print(PREREGISTRATION)
    print(result["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
