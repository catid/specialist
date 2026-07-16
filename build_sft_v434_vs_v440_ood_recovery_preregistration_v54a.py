#!/usr/bin/env python3
"""Seal CPU-only recovery of the completed V54A OOD inference receipts."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import recover_sft_v434_vs_v440_ood_only_v54a as recovery


def _self_hashed_artifact(path: Path, label: str) -> dict:
    value = recovery._load_json(path)
    content = recovery._validate_self_hash(value, label)
    return {
        "path": str(path),
        "file_sha256": recovery.file_sha256(path),
        "content_sha256": content,
    }


def build() -> dict:
    source_prereg = recovery._load_json(recovery.SOURCE_PREREGISTRATION)
    source_content = recovery._validate_self_hash(
        source_prereg, "source preregistration"
    )
    if (
        source_prereg.get("schema")
        != "sft-v434-equal-vs-v440-equal-replicated-ood-only-v54a"
        or source_prereg.get("status")
        != "preregistered_before_fresh_ood_only_launch"
        or source_prereg.get("shadow_access_authorized") is not False
        or source_prereg.get("heldout_or_holdout_access_authorized") is not False
        or recovery.file_sha256(recovery.SOURCE_PREREGISTRATION)
        != recovery.SOURCE_EXPECTED["preregistration_file"]
        or source_content != recovery.SOURCE_EXPECTED["preregistration_content"]
    ):
        raise RuntimeError("V54A source preregistration scope changed")
    raw = recovery._load_json(recovery.SOURCE_RAW)
    inventory = recovery.receipt_inventory_v54a(raw)
    gpu = recovery.gpu_receipt_v54a(recovery.SOURCE_GPU_LOG)
    source_artifacts = {
        "source_preregistration": _self_hashed_artifact(
            recovery.SOURCE_PREREGISTRATION, "source preregistration"
        ),
        "attempt": _self_hashed_artifact(
            recovery.SOURCE_ATTEMPT, "source attempt"
        ),
        "failure": _self_hashed_artifact(
            recovery.SOURCE_FAILURE, "source failure"
        ),
        "raw": {
            "path": str(recovery.SOURCE_RAW),
            "file_sha256": recovery.file_sha256(recovery.SOURCE_RAW),
            "bytes": recovery.SOURCE_RAW.stat().st_size,
            "mode": oct(recovery.SOURCE_RAW.stat().st_mode & 0o777),
        },
        "gpu_log": {
            "path": str(recovery.SOURCE_GPU_LOG),
            "file_sha256": recovery.file_sha256(recovery.SOURCE_GPU_LOG),
            "bytes": recovery.SOURCE_GPU_LOG.stat().st_size,
        },
    }
    expected = recovery.SOURCE_EXPECTED
    if (
        source_artifacts["attempt"]["file_sha256"] != expected["attempt_file"]
        or source_artifacts["attempt"]["content_sha256"]
        != expected["attempt_content"]
        or source_artifacts["failure"]["file_sha256"] != expected["failure_file"]
        or source_artifacts["failure"]["content_sha256"]
        != expected["failure_content"]
        or source_artifacts["raw"]["file_sha256"] != expected["raw_file"]
        or source_artifacts["raw"]["mode"] != "0o600"
        or source_artifacts["gpu_log"]["file_sha256"]
        != expected["gpu_log_file"]
    ):
        raise RuntimeError("V54A source receipt identity changed")
    value = {
        "schema": "v54a-ood-only-offline-recovery-preregistration",
        "status": "sealed_before_offline_recovery",
        "recovery_execution_authorized": True,
        "generation_or_gpu_access_authorized": False,
        "protected_semantic_input_access_authorized": False,
        "source_artifacts": source_artifacts,
        "source_failure_semantics": {
            "generation_and_raw_receipt_write_completed": True,
            "failure_stage": "legacy_gpu_summary_after_raw_receipt_write",
            "legacy_error": "v39a GPU 0 inactive in shadow",
            "error_is_out_of_scope_phase_requirement": True,
            "semantic_inputs_may_not_be_reopened": True,
        },
        "expected_receipt_inventory": inventory,
        "expected_gpu_receipt": gpu,
        "metric_contract": {
            "qa_aggregate": "exact V39A receipt reconstruction",
            "qa_gate": "V39A base-relative reward and exact non-degradation",
            "qa_paired_bootstrap": {
                "samples": recovery.BOOTSTRAP_SAMPLES,
                "seed": recovery.BOOTSTRAP_SEED,
                "role": "informational_not_a_gate",
            },
            "prose_aggregate": "token-weighted mean logprob from item sums",
            "prose_gate": (
                "V39A point plus paired-document-bootstrap LCB non-degradation"
            ),
            "each_replica_independently_gated": True,
            "both_replicas_required_for_logical_eligibility": True,
            "direct_v440_minus_v434_reward_minimum": 0.0,
            "direct_v440_minus_v434_exact_minimum": 0.0,
            "direct_v440_minus_v434_prose_logprob_minimum": 0.0,
            "direct_point_comparison_uses_replica_means": True,
        },
        "access_contract": {
            "offline_allowed_reads": [
                str(recovery.SOURCE_PREREGISTRATION),
                str(recovery.SOURCE_ATTEMPT),
                str(recovery.SOURCE_FAILURE),
                str(recovery.SOURCE_RAW),
                str(recovery.SOURCE_GPU_LOG),
            ],
            "semantic_source_files_allowed": [],
            "semantic_source_file_hashing_allowed": False,
            "model_or_adapter_access_allowed": False,
            "gpu_access_allowed": False,
            "shadow_split_holdout_or_heldout_access_allowed": False,
        },
        "implementation_bindings": recovery.implementation_bindings_v54a(),
        "artifact_paths": {
            "preregistration": str(recovery.DEFAULT_PREREGISTRATION),
            "report": str(recovery.REPORT),
        },
        "selection_firewall": {
            "this_phase_authorizes": "offline OOD receipt recovery only",
            "shadow_ranking_authorized": False,
            "holdout_evaluation_authorized": False,
            "selection_or_promotion_authorized": False,
        },
    }
    value["content_sha256_before_self_field"] = recovery.canonical_sha256(value)
    return value


def launch_command(path: Path, file_sha: str, content_sha: str) -> list[str]:
    return [
        str(recovery.ROOT / "es-at-scale/.venv/bin/python"),
        str(Path(recovery.__file__).resolve()),
        "--preregistration", str(path.resolve()),
        "--preregistration-sha256", file_sha,
        "--preregistration-content-sha256", content_sha,
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(recovery.DEFAULT_PREREGISTRATION))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    recovery.atomic_json(output, value)
    file_sha = recovery.file_sha256(output)
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha,
        "content_sha256": value["content_sha256_before_self_field"],
        "launch_command": shlex.join(launch_command(
            output, file_sha, value["content_sha256_before_self_field"]
        )),
        "protected_semantic_input_reads": 0,
        "model_or_generation_accessed": False,
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
