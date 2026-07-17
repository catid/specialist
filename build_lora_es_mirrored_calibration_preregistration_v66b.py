#!/usr/bin/env python3
"""Seal the fresh V66b retry after the full-weight/LoRA mode fix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import build_lora_es_mirrored_calibration_preregistration_v66 as v66


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_mirrored_calibration_v66b.json"
).resolve()
FAILED_RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v66_lora_es_mirrored_crn_qwen36_calibration"
).resolve()
FAILED_ATTEMPT = (
    FAILED_RUN.parent
    / ".v66_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
).resolve()
FAILED_RECEIPT = (FAILED_RUN / "failure_v66.json").resolve()
RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v66b_lora_es_mirrored_crn_qwen36_calibration"
).resolve()


def artifacts_v66b() -> dict:
    return {
        "attempt": str(
            RUN.parent
            / ".v66b_lora_es_mirrored_crn_qwen36_calibration.attempt.json"
        ),
        "run_directory": str(RUN),
        "gpu_log": str(RUN / "gpu_activity_v66b.jsonl"),
        "population": str(RUN / "mirrored_population_v66b.json"),
        "update": str(RUN / "pair_difference_update_v66b.json"),
        "report": str(RUN / "mirrored_calibration_report_v66b.json"),
        "failure": str(RUN / "failure_v66b.json"),
    }


def _failed_attempt_receipt_v66b() -> dict:
    if not FAILED_ATTEMPT.is_file() or not FAILED_RECEIPT.is_file():
        raise RuntimeError("v66b requires the immutable failed V66 receipts")
    attempt = json.loads(FAILED_ATTEMPT.read_text(encoding="utf-8"))
    failure = json.loads(FAILED_RECEIPT.read_text(encoding="utf-8"))
    if (
        attempt.get("content_sha256_before_self_field")
        != "25645974f4d3e2f7666c59e79a2b731ad687abf8ef2f442a5712c3d3309ace10"
        or failure.get("content_sha256_before_self_field")
        != "82deabfc5f52a8038034808d9c22d166425fb3828e17f3f742865064f90e05f6"
        or failure.get("type") != "RayTaskError(NotImplementedError)"
        or failure.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        )
        is not True
        or failure.get("protected_dev_ood_or_holdout_opened") is not False
        or failure.get("checkpoint_snapshot_or_promotion_performed") is not False
    ):
        raise RuntimeError("failed V66 evidence changed")
    return {
        "attempt_path": str(FAILED_ATTEMPT),
        "attempt_file_sha256": v66.file_sha256_v66(FAILED_ATTEMPT),
        "attempt_content_sha256": attempt[
            "content_sha256_before_self_field"
        ],
        "failure_path": str(FAILED_RECEIPT),
        "failure_file_sha256": v66.file_sha256_v66(FAILED_RECEIPT),
        "failure_content_sha256": failure[
            "content_sha256_before_self_field"
        ],
        "failure_type": failure["type"],
        "charged_gpu_seconds": failure["compute_ledger"][
            "charged_gpu_seconds"
        ],
        "final_gpu_idle": True,
        "adapter_install_reached": False,
        "candidate_evaluation_reached": False,
        "candidate_update_reached": False,
    }


def build_preregistration_v66b() -> dict:
    result = v66.build_preregistration_v66()
    result.pop("content_sha256_before_self_field", None)
    result["schema"] = (
        "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66b"
    )
    result["status"] = (
        "sealed_after_explicit_lora_state_mode_fix_before_v66b_gpu_access"
    )
    result["purpose"] = (
        "Fresh one-shot retry of the unchanged V66 mirrored pair-difference "
        "calibration after explicitly separating LoRA worker state from dense "
        "full-weight state initialization."
    )
    result["supersedes_failed_v66"] = _failed_attempt_receipt_v66b()
    result["artifacts"] = artifacts_v66b()
    result["runtime"].update({
        "trainer_state_mode": "external_worker",
        "dense_full_weight_master_install_authorized": False,
        "required_worker_endpoints": [
            "install_adapter_state_v41a",
            "materialize_mirrored_adapter_v66",
            "restore_mirrored_adapter_v66",
        ],
    })
    result["acceptance"].update({
        "trainer_constructs_without_dense_master_rpc": True,
        "canonical_lora_install_reached": True,
    })
    bindings = result["implementation_bindings"]
    bindings["v66_builder"] = bindings.pop("builder")
    bindings["v66_runner"] = bindings.pop("runner")
    paths = {
        "builder": Path(__file__).resolve(),
        "runner": ROOT / "run_lora_es_mirrored_calibration_v66b.py",
        "base_trainer": ROOT / "train_eggroll_es_specialist.py",
        "failed_v66_attempt": FAILED_ATTEMPT,
        "failed_v66_receipt": FAILED_RECEIPT,
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


def write_preregistration_v66b(path: Path = OUTPUT) -> dict:
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    value = build_preregistration_v66b()
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
    value = build_preregistration_v66b()
    if args.check:
        if not output.is_file() or json.loads(
            output.read_text(encoding="utf-8")
        ) != value:
            raise RuntimeError("v66b preregistration is absent or stale")
    else:
        write_preregistration_v66b(output)
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
