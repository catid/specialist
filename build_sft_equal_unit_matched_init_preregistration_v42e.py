#!/usr/bin/env python3
"""Freeze the 3e-4 V42E matched-initialization SFT HPO arm."""

from __future__ import annotations

import copy
from pathlib import Path

import build_sft_equal_unit_matched_init_preregistration_v42b as v42b
import run_sft_equal_unit_matched_init_v42e as launcher


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/sft_controls/"
    "v42e_matched_init_equal_unit_fold3_v412_lr3e4"
).resolve()
OUTPUT_DIR = RUN_DIR / "middle_late_r32_seed17_init20260715041"
PREREGISTRATION = RUN_DIR / "preregistration_v42e.json"
LEARNING_RATE = 3e-4


def arguments():
    args = v42b.arguments()
    args.output_dir = str(OUTPUT_DIR)
    args.stdout_log = str(RUN_DIR / "stdout_v42e.log")
    args.gpu_log = str(RUN_DIR / "gpu_activity_v42e.jsonl")
    args.report = str(RUN_DIR / "runtime_report_v42e.json")
    args.attempt_report = str(RUN_DIR / "attempt_v42e.json")
    args.preregistration = str(PREREGISTRATION)
    args.preregistration_sha256 = "PENDING"
    args.preregistration_content_sha256 = "PENDING"
    args.learning_rate = LEARNING_RATE
    return args


def build() -> dict:
    result = copy.deepcopy(v42b.build())
    result.pop("content_sha256_before_self_field", None)
    args = arguments()
    result.update({
        "status": "preregistered_retry_not_yet_run",
        "experiment_name": (
            "sft_matched_init_equal_unit_v42e_fold3_v412_"
            "middle_late_r32_seed17_init20260715041_lr3e4"
        ),
        "artifacts": {
            "output_dir": str(OUTPUT_DIR),
            "stdout_log": str(RUN_DIR / "stdout_v42e.log"),
            "gpu_log": str(RUN_DIR / "gpu_activity_v42e.jsonl"),
            "report": str(RUN_DIR / "runtime_report_v42e.json"),
            "attempt_report": str(RUN_DIR / "attempt_v42e.json"),
        },
    })
    result["recipe"].update({
        "command": launcher.build_train_command(args),
        "learning_rate": LEARNING_RATE,
        "hpo_parent": "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load",
        "only_change_from_v42b": "peak cosine-schedule learning rate 1e-4 -> 3e-4",
    })
    result["implementation"]["hpo_launcher_v42e"] = {
        "path": str((ROOT / "run_sft_equal_unit_matched_init_v42e.py").resolve()),
        "sha256": v42b.engine.file_sha256(
            ROOT / "run_sft_equal_unit_matched_init_v42e.py"
        ),
    }
    result["selection_firewall"]["this_arm_authorizes"] = (
        "train-only 3e-4 HPO state and runtime evidence only"
    )
    result["content_sha256_before_self_field"] = v42b.engine.canonical_sha256(result)
    return result


def main() -> None:
    result = build()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if PREREGISTRATION.exists():
        raise RuntimeError("V42E preregistration already exists")
    v42b.engine.atomic_write_json(PREREGISTRATION, result)
    print(PREREGISTRATION)
    print(result["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
