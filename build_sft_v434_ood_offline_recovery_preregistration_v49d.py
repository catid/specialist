#!/usr/bin/env python3
"""Seal the CPU-only V49D retry1 receipt recovery before metric recovery."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import recover_sft_v434_sampling_midpoint_ood_only_v49d as recovery


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
    recovery._validate_self_hash(source_prereg, "source preregistration")
    if (
        source_prereg.get("schema")
        != "sft-v434-equal-vs-source50-replicated-ood-only-v49d"
        or source_prereg.get("shadow_access_authorized") is not False
        or source_prereg.get("heldout_or_holdout_access_authorized") is not False
    ):
        raise RuntimeError("V49D source preregistration scope changed")
    raw = recovery._load_json(recovery.SOURCE_RAW)
    inventory = recovery.receipt_inventory_v49d(raw)
    gpu = recovery.gpu_receipt_v49d(recovery.SOURCE_GPU_LOG)
    source_artifacts = {
        "source_preregistration": _self_hashed_artifact(
            recovery.SOURCE_PREREGISTRATION, "source preregistration"
        ),
        "attempt": _self_hashed_artifact(recovery.SOURCE_ATTEMPT, "source attempt"),
        "failure": _self_hashed_artifact(recovery.SOURCE_FAILURE, "source failure"),
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
    value = {
        "schema": "v49d-ood-only-offline-recovery-preregistration",
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
            "prose_gate": "V39A point plus paired-document-bootstrap LCB non-degradation",
            "each_replica_independently_gated": True,
            "both_replicas_required_for_logical_eligibility": True,
            "direct_source50_minus_equal_reward_minimum": 0.0,
            "direct_source50_minus_equal_exact_minimum": 0,
            "shadow_threshold_deferred_not_applied": 0.0008257591,
        },
        "access_contract": {
            "offline_allowed_reads": [
                str(recovery.SOURCE_PREREGISTRATION), str(recovery.SOURCE_ATTEMPT),
                str(recovery.SOURCE_FAILURE), str(recovery.SOURCE_RAW),
                str(recovery.SOURCE_GPU_LOG),
            ],
            "semantic_source_files_allowed": [],
            "semantic_source_file_hashing_allowed": False,
            "model_or_adapter_access_allowed": False,
            "gpu_access_allowed": False,
            "shadow_split_holdout_or_heldout_access_allowed": False,
        },
        "implementation_bindings": recovery.implementation_bindings_v49d(),
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
    # Bind the preregistration's own eventual file identity after the first
    # deterministic write.  The content hash remains the pre-self-field seal.
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
