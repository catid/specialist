#!/usr/bin/env python3
"""Seal launchable V48B only after its train-only subset is fully bound."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_lora_es_generation_boundary_v48b as runtime
import run_lora_es_multi_anchor_v43i as v43i


ROOT = Path(__file__).resolve().parent
OUTPUT = runtime.PREREGISTRATION
PARENT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_multi_anchor_v43i.json"
).resolve()
PARENT_FILE_SHA256 = (
    "00c545926b217a64acabbc541f3e92e071a1a199dbabef121383c788f574272e"
)
PARENT_CONTENT_SHA256 = (
    "086d94f1b69732a9a0d7913c8bab7789b15f64131f125ba4381eea3bcc228c5a"
)


def build_v48b(
    subset_path: Path, subset_file_sha: str, subset_content_sha: str,
) -> dict:
    subset_path = Path(subset_path).resolve()
    if subset_path != runtime.SUBSET:
        raise RuntimeError("v48b subset must use the canonical sealed path")
    subset = runtime.load_subset_v48b(
        subset_path, subset_file_sha, subset_content_sha
    )
    if v43i.v40a.file_sha256(PARENT) != PARENT_FILE_SHA256:
        raise RuntimeError("v48b V43I parent file changed")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    if (
        parent.get("content_sha256_before_self_field")
        != PARENT_CONTENT_SHA256
        or v43i.v40a.canonical_sha256({
            key: item for key, item in parent.items()
            if key != "content_sha256_before_self_field"
        }) != PARENT_CONTENT_SHA256
        or parent.get("sealed_holdout_opened") is not False
    ):
        raise RuntimeError("v48b V43I parent content changed")
    source = subset["source"]
    value = {
        "schema": "matched-lora-es-generation-boundary-preregistration-v48b",
        "status": "preregistered_before_train_only_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "purpose": (
            "Make deterministic conflict-unit-balanced train generated F1 a "
            "direct ES objective while retaining domain, F1, prose, and QA "
            "logprob halfspaces and V43I's uncommitted full candidate gate."
        ),
        "protected_semantic_access_authorized": False,
        "shadow_ood_holdout_or_benchmark_authorized": False,
        "quality_selection_or_promotion_authorized": False,
        "access_contract": {
            "only_runtime_train_paths_may_open": [
                str(runtime.evidence_runtime.TRAIN_DATASET),
                str(runtime.evidence_runtime.MEMBERSHIP),
                str(runtime.SUBSET),
                str(v43i.PROSE_ANCHOR), str(v43i.QA_ANCHOR),
            ],
            "original_split_manifest_opened_at_runtime": False,
            "base_evidence_reopened_at_population_runtime": False,
            "direct_benchmark_source_opened": False,
            "shadow_ood_holdout_or_benchmark_path_opened": False,
            "builder_reads_train_semantics": False,
            "dry_run_reads_train_semantics": False,
            "dry_run_launches_model_or_gpu": False,
        },
        "parents": {
            "v43i_runtime": {
                "file_sha256": PARENT_FILE_SHA256,
                "content_sha256": PARENT_CONTENT_SHA256,
            },
            "base_evidence": {
                "file_sha256": source["evidence_file_sha256"],
                "content_sha256": source["evidence_content_sha256"],
                "report_file_sha256": source[
                    "evidence_report_file_sha256"
                ],
                "report_content_sha256": source[
                    "evidence_report_content_sha256"
                ],
                "reopened_at_runtime": False,
            },
        },
        "recipe": {
            "model": str(v43i.v40a.MODEL),
            "matched_initialization": str(v43i.SOURCE),
            "staged_initialization": str(v43i.STAGED),
            "dataset": str(runtime.evidence_runtime.TRAIN_DATASET),
            "dataset_sha256": runtime.evidence_runtime.EXPECTED_TRAIN_SHA256,
            "dataset_rows": 448,
            "conflict_units": 208,
            "train_bundle_content_sha256": (
                runtime.EXPECTED_TRAIN_BUNDLE_CONTENT_SHA256_V48B
            ),
            "membership": str(runtime.evidence_runtime.MEMBERSHIP),
            "membership_file_sha256": (
                runtime.evidence_runtime.EXPECTED_MEMBERSHIP_SHA256
            ),
            "subset": str(runtime.SUBSET),
            "subset_file_sha256": subset_file_sha,
            "subset_content_sha256": subset_content_sha,
            "request_order_sha256": subset["request_order_sha256"],
            "fragile_generation_documents": 64,
            "population_size": v43i.POPULATION_SIZE,
            "seeds": v43i.SEEDS,
            "sigma": v43i.SIGMA,
            "alpha": v43i.ALPHA,
            "signed_replicates_per_direction": (
                v43i.numeric.SIGNED_REPLICATES_V43G
            ),
            "fused_requests_per_population_actor_state": 608,
            "worker_extension": v43i.WORKER_EXTENSION,
            "physical_gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
        },
        "generation_boundary_objective": {
            "primary": (
                "equal average of unit-norm domain and fragile generated-F1 "
                "centered-rank coefficient vectors"
            ),
            "projection_halfspaces": [
                "domain", "fragile_generation_f1", "prose_lm",
                "qa_answer_logprob",
            ],
            "fail_if_fragile_generation_f1_has_zero_population_spread": True,
            "common_random_plan_receipts": 64,
            "trust_region_norm_ratio": (
                v43i.multi_anchor.TRUST_REGION_NORM_RATIO_V43H
            ),
        },
        "uncommitted_candidate_gate": {
            "inherits_full_128_document_v43i_anchor_gate": True,
            "adds_fragile_f1_exact_and_nonzero_noninferiority": True,
            "candidate_scored_before_commit": True,
            "abort_on_any_failure": True,
        },
        "runtime": dict(parent["runtime"]),
        "implementation_bindings": runtime.implementation_bindings_v48b(
            subset_file_sha
        ),
        "artifacts": {
            "attempt": str(runtime.ATTEMPT),
            "run_directory": str(runtime.RUN_DIR),
            "report": str(runtime.REPORT),
            "gpu_log": str(runtime.GPU_LOG),
            "snapshot": str(runtime.SNAPSHOT),
            "population": str(runtime.RELIABILITY_ARTIFACT),
            "candidate_gate": str(runtime.CANDIDATE_GATE_ARTIFACT),
            "exact_abort": str(runtime.ABORT_ARTIFACT),
        },
        "required_gates": {
            "all_common_random_plan_receipts_exact": True,
            "direct_f1_population_spread_nonzero": True,
            "domain_f1_prose_qa_halfspaces_all_nonnegative": True,
            "V43I_reliability_and_exact_restore_retained": True,
            "full_candidate_gate_passes_before_commit": True,
            "all_four_gpus_attributed_positive": True,
            "strict_cleanup_and_idle": True,
        },
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    }
    value["content_sha256_before_self_field"] = v43i.v40a.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", required=True)
    parser.add_argument("--subset-sha256", required=True)
    parser.add_argument("--subset-content-sha256", required=True)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v48b(
        Path(args.subset), args.subset_sha256, args.subset_content_sha256
    )
    v43i.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": v43i.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "gpu_launch_authorized": True,
        "protected_semantics_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
