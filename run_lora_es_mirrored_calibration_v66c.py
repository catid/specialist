#!/usr/bin/env python3
"""Fresh V66c retry with explicit four-actor LoRA slot activation."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import build_lora_es_mirrored_calibration_preregistration_v66c as builder
import run_lora_es_mirrored_calibration_v66 as v66
import train_eggroll_es_specialist as base


ROOT = Path(__file__).resolve().parent
PREREGISTRATION = builder.OUTPUT
RUN_DIR = builder.RUN
ATTEMPT = (
    RUN_DIR.parent
    / ".v66c_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
).resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v66c.jsonl").resolve()
POPULATION = (RUN_DIR / "mirrored_population_v66c.json").resolve()
UPDATE = (RUN_DIR / "pair_difference_update_v66c.json").resolve()
REPORT = (RUN_DIR / "mirrored_calibration_report_v66c.json").resolve()
FAILURE = (RUN_DIR / "failure_v66c.json").resolve()
REQUIRED_WORKER_ENDPOINTS_V66C = (
    "install_adapter_state_v41a",
    "active_lora_slot_certificate_v66",
    "materialize_mirrored_adapter_v66",
    "restore_mirrored_adapter_v66",
)


def artifacts_v66c() -> dict:
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "gpu_log": str(GPU_LOG),
        "population": str(POPULATION),
        "update": str(UPDATE),
        "report": str(REPORT),
        "failure": str(FAILURE),
    }


def validate_lora_worker_contract_v66c() -> dict:
    contract = base.validate_worker_state_mode(
        base.TRAINER_STATE_MODE_EXTERNAL_WORKER,
        v66.WORKER_EXTENSION_V66,
        REQUIRED_WORKER_ENDPOINTS_V66C,
    )
    extension = contract.pop("resolved_worker_extension")
    contract["resolved_class"] = (
        f"{extension.__module__}.{extension.__qualname__}"
    )
    contract["dense_full_weight_master_install_authorized"] = False
    return contract


def load_preregistration_v66c(args) -> dict:
    path = Path(args.preregistration).resolve()
    if v66.file_sha256_v66(path) != args.preregistration_sha256:
        raise RuntimeError("v66c preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or v66.mirrored.canonical_sha256_v66(compact)
        != args.preregistration_content_sha256
        or value != builder.build_preregistration_v66c()
        or value.get("schema")
        != "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66c"
        or value.get("status")
        != "sealed_after_four_actor_lora_activation_fix_before_v66c_gpu_access"
        or value.get("artifacts") != artifacts_v66c()
    ):
        raise RuntimeError("v66c preregistration content changed")
    return value


@contextmanager
def _fresh_namespace_v66c():
    names = {
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "GPU_LOG": GPU_LOG,
        "POPULATION": POPULATION,
        "UPDATE": UPDATE,
        "REPORT": REPORT,
        "FAILURE": FAILURE,
    }
    saved = {name: getattr(v66, name) for name in names}
    for name, value in names.items():
        setattr(v66, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(v66, name, value)


def parser_v66c() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", default=str(PREREGISTRATION))
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--preregistration-content-sha256", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv=None) -> int:
    args = parser_v66c().parse_args(argv)
    if args.dry_run == args.execute:
        raise ValueError("v66c requires exactly one of --dry-run or --execute")
    preregistration = load_preregistration_v66c(args)
    worker_contract = validate_lora_worker_contract_v66c()
    if args.dry_run:
        recipe = preregistration["fixed_recipe"]
        print(json.dumps({
            "schema": preregistration["schema"],
            "model": "Qwen3.6-35B-A3B",
            "four_tp1_engines": True,
            "direction_count": len(recipe["direction_seeds"]),
            "signed_population_size": 2 * len(recipe["direction_seeds"]),
            "train_only_rows_per_candidate": 64,
            "expected_artifacts": artifacts_v66c(),
            "worker_contract": worker_contract,
            "activation_order": [
                "add_lora_all_four",
                "active_slot_certificate_all_four",
                "install_adapter_state_all_four",
                "canonical_certificate_all_four",
            ],
            "train_semantics_model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "protected_dev_ood_or_holdout_opened": False,
            "checkpoint_snapshot_or_promotion_authorized": False,
        }, sort_keys=True))
        return 0
    if Path(sys.executable).absolute() != v66.REQUIRED_PYTHON_V66:
        raise RuntimeError(
            f"v66c requires {v66.REQUIRED_PYTHON_V66}; observed {sys.executable}"
        )
    with _fresh_namespace_v66c():
        return v66.execute_v66(preregistration, args)


if __name__ == "__main__":
    raise SystemExit(main())
