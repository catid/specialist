#!/usr/bin/env python3
"""Seal the launch-ready, train-only multi-anchor LoRA-ES V43I contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import eggroll_es_multi_anchor_v43h as projection
import lora_es_fused_anchor_runtime_v43i as fused
import lora_es_robust_consensus_v43g as numeric
import run_lora_es_multi_anchor_v43i as runtime


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_multi_anchor_v43i.json"
).resolve()
PARENT_V43G = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_robust_v43g.json"
).resolve()
PARENT_V43H = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_multi_anchor_static_v43h.json"
).resolve()
EXPECTED_V43G_FILE_SHA256 = (
    "b923bd72ed9d09936200b7aae0f6c25671f901a41d26e062e18371eab845e3a6"
)
EXPECTED_V43H_FILE_SHA256 = (
    "bbe0affa76f65f0d19b2be3492542e14a88c8bf423e06546dc618432f11aa661"
)
EXPECTED_SOURCE_WEIGHTS = (
    "29fe0beead8a491cf06e9f562a1838d9c44e94a74e6a4024549e87f10557111f"
)
EXPECTED_STAGED_WEIGHTS = (
    "9c4382048a2bb001995586715eac53f51672a161f3351080a604543b8384a09b"
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output", default=str(DEFAULT_OUTPUT))
    return result


def _sealed_parent(path: Path, expected_file_sha256: str) -> dict:
    if runtime.v40a.file_sha256(path) != expected_file_sha256:
        raise RuntimeError(f"v43i parent file identity changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if value.get("content_sha256_before_self_field") != (
        runtime.v40a.canonical_sha256(compact)
    ):
        raise RuntimeError(f"v43i parent content identity changed: {path}")
    return value


def build_v43i() -> dict:
    v43g = _sealed_parent(PARENT_V43G, EXPECTED_V43G_FILE_SHA256)
    v43h = _sealed_parent(PARENT_V43H, EXPECTED_V43H_FILE_SHA256)
    if (
        v43g.get("sealed_holdout_opened") is not False
        or v43g.get("recipe", {}).get("sigma") != runtime.SIGMA
        or v43g.get("recipe", {}).get("alpha") != runtime.ALPHA
        or v43h.get("gpu_launch_authorized") is not False
        or v43h.get("source_firewall", {}).get(
            "direct_hotpotqa_benchmark_opened"
        ) is not False
        or v43h.get("recipe", {}).get("required_gradient_anchors")
        != ["prose_lm", "qa_answer_logprob"]
    ):
        raise RuntimeError("v43i inherited train-only design contract changed")
    bindings = runtime.implementation_bindings()
    if (
        bindings["source_weights"] != EXPECTED_SOURCE_WEIGHTS
        or bindings["staged_weights"] != EXPECTED_STAGED_WEIGHTS
        or bindings["dataset"] != runtime.DATASET_SHA256
        or bindings["split_manifest"] != runtime.SPLIT_MANIFEST_SHA256
        or bindings["prose_anchor"] != fused.PROSE_SHA256_V43I
        or bindings["prose_report"] != fused.PROSE_REPORT_SHA256_V43I
        or bindings["qa_anchor"] != fused.QA_PROXY_SHA256_V43I
        or bindings["qa_report"] != fused.QA_PROXY_REPORT_SHA256_V43I
    ):
        raise RuntimeError("v43i sealed implementation/input bindings changed")
    tuned_table = json.loads(runtime.v40a.TUNED_FILE.read_text(encoding="utf-8"))
    tuned_table.pop("triton_version", None)
    tuned_content = runtime.v40a.canonical_sha256({
        int(key): item for key, item in tuned_table.items()
    })
    value = {
        "schema": "matched-lora-es-preregistration-v43i",
        "status": "preregistered_before_train_only_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "purpose": (
            "Project robust LoRA-ES domain utility against train-only prose and "
            "general-QA proxy utilities, then require full-anchor preservation "
            "while the exact FP32 candidate remains uncommitted."
        ),
        "shadow_dev_external_eval_ood_or_holdout_authorized": False,
        "sealed_holdout_opened": False,
        "quality_selection_or_promotion_authorized": False,
        "access_contract": {
            "protected_semantic_access": False,
            "only_runtime_train_paths_may_open": [
                str(runtime.DATASET), str(runtime.PROSE_ANCHOR),
                str(runtime.QA_ANCHOR),
            ],
            "direct_benchmark_source_opened": False,
            "direct_benchmark_source_authorized": False,
            "builder_reads_train_examples": False,
            "builder_reads_protected_examples": False,
            "dry_run_reads_train_examples": False,
            "dry_run_hashes_bound_input_files": True,
            "dry_run_loads_model_runtime": False,
            "dry_run_launches_gpu": False,
        },
        "parents": {
            "v43g_verified_runtime_recipe": {
                "path": str(PARENT_V43G),
                "file_sha256": EXPECTED_V43G_FILE_SHA256,
                "content_sha256": v43g["content_sha256_before_self_field"],
            },
            "v43h_static_multi_anchor_design": {
                "path": str(PARENT_V43H),
                "file_sha256": EXPECTED_V43H_FILE_SHA256,
                "content_sha256": v43h["content_sha256_before_self_field"],
            },
        },
        "recipe": {
            "model": str(runtime.v40a.MODEL),
            "matched_initialization": str(runtime.SOURCE),
            "matched_initialization_weights_sha256": EXPECTED_SOURCE_WEIGHTS,
            "staged_initialization": str(runtime.STAGED),
            "staged_initialization_weights_sha256": EXPECTED_STAGED_WEIGHTS,
            "dataset": str(runtime.DATASET),
            "dataset_sha256": runtime.DATASET_SHA256,
            "dataset_rows": 448,
            "conflict_units": numeric.EXPECTED_CONFLICT_UNITS_V43G,
            "split_manifest": str(runtime.SPLIT_MANIFEST),
            "split_manifest_sha256": runtime.SPLIT_MANIFEST_SHA256,
            "train_bundle_content_sha256": runtime.TRAIN_BUNDLE_SHA256,
            "population_size": runtime.POPULATION_SIZE,
            "seeds": runtime.SEEDS,
            "sigma": runtime.SIGMA,
            "alpha": runtime.ALPHA,
            "signed_replicates_per_direction": numeric.SIGNED_REPLICATES_V43G,
            "signed_replication_assignment": (
                "complete four-actor paired antithetic block per direction"
            ),
            "fitness_shaping": (
                "centered ranks over 16 median signed scores; coefficient=u_plus-u_minus"
            ),
            "minimum_response_reliability": numeric.RELIABILITY_MINIMUM_V43G,
            "minimum_split_half_spearman": (
                numeric.SPLIT_HALF_SPEARMAN_MINIMUM_V43G
            ),
            "worker_extension": runtime.WORKER_EXTENSION,
            "population_anchor_documents": fused.PANEL_SIZE_V43I,
            "candidate_anchor_documents": fused.FULL_SIZE_V43I,
            "fused_requests_per_population_actor_state": 544,
            "physical_gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
            "tuned_table_content_sha256": tuned_content,
        },
        "numeric_calibration": {
            "synthetic_noise_seed": numeric.CALIBRATION_NOISE_SEED_V43G,
            "synthetic_sigma": runtime.CALIBRATION_SIGMA,
            "warmup_repeats": numeric.CALIBRATION_WARMUPS_V43G,
            "retained_repeats_per_actor": numeric.CALIBRATION_REPEATS_V43G,
            "bootstrap_resamples": numeric.BOOTSTRAP_RESAMPLES_V43G,
            "bootstrap_seed": numeric.CALIBRATION_BOOTSTRAP_SEED_V43G,
            "familywise_confidence": numeric.BOOTSTRAP_CONFIDENCE_V43G,
            "historical_catastrophic_divergence_ceiling": (
                numeric.V43F_HISTORICAL_EQUAL_UNIT_BOUND
            ),
        },
        "anchor_calibration": {
            "state": "same fixed nonzero synthetic adapter state on all actors",
            "warmup_repeats": fused.CALIBRATION_WARMUPS_V43I,
            "retained_repeats": fused.CALIBRATION_REPEATS_V43I,
            "actors": 4,
            "documents": fused.FULL_SIZE_V43I,
            "logprob_margin_ceiling": fused.LOGPROB_MARGIN_CEILING_V43I,
            "generation_f1_margin_ceiling": (
                fused.GENERATION_F1_MARGIN_CEILING_V43I
            ),
            "generation_count_margin_ceiling": (
                fused.GENERATION_COUNT_MARGIN_CEILING_V43I
            ),
            "fit_before_population": True,
        },
        "multi_anchor_projection": {
            "required_gradient_anchors": ["prose_lm", "qa_answer_logprob"],
            "method": "simultaneous Euclidean active-set halfspace projection",
            "constraints": "coefficient dot each required anchor >= 0",
            "zero_spread_required_anchor": "fail closed before update",
            "trust_region_max_norm_ratio": (
                projection.TRUST_REGION_NORM_RATIO_V43H
            ),
            "restandardize_after_projection": False,
        },
        "uncommitted_candidate_gate": {
            "score_before_commit": True,
            "abort_on_any_failure": True,
            "greedy_qa_required": True,
            "full_anchor_documents": fused.FULL_SIZE_V43I,
            "all_four_actors": True,
            "domain_point_delta_must_be_positive": True,
            "anchor_noninferiority_uses_prefit_fixed_state_margins": True,
            "exact_abort_requires_master_and_runtime_hash_readback": True,
            "snapshot_before_gate_forbidden": True,
        },
        "post_update_consensus": {
            "executed_while_candidate_uncommitted": True,
            "retained_repeats_per_actor": numeric.POST_UPDATE_REPEATS_V43G,
            "bootstrap_resamples": numeric.BOOTSTRAP_RESAMPLES_V43G,
            "bootstrap_seed": numeric.POST_UPDATE_BOOTSTRAP_SEED_V43G,
            "familywise_confidence": numeric.BOOTSTRAP_CONFIDENCE_V43G,
        },
        "runtime": {
            "tuned_folder": str(runtime.v40a.TUNED_FOLDER),
            "tuned_table_content_sha256": tuned_content,
        },
        "required_gates": {
            "all_four_gpus_have_attributed_positive_activity": True,
            "all_signed_states_use_one_fused_request_batch_per_actor": True,
            "all_required_anchor_halfspaces_satisfied": True,
            "projected_norm_at_most_half_domain_norm": True,
            "candidate_executed_without_commit": True,
            "full_anchor_candidate_gate_passes_before_commit": True,
            "any_preaccept_exception_exactly_restores_master_and_runtime": True,
            "accepted_snapshot_has_exact_rank_zero_readback": True,
            "base_weights_unchanged_at_every_state_boundary": True,
            "final_all_gpu_idle_cleanup": True,
        },
        "implementation_bindings": bindings,
        "artifacts": {
            "attempt": str(runtime.ATTEMPT),
            "run_directory": str(runtime.RUN_DIR),
            "numeric_calibration": str(runtime.CALIBRATION_ARTIFACT),
            "anchor_calibration": str(runtime.ANCHOR_CALIBRATION_ARTIFACT),
            "population_reliability": str(runtime.RELIABILITY_ARTIFACT),
            "candidate_gate": str(runtime.CANDIDATE_GATE_ARTIFACT),
            "exact_abort": str(runtime.ABORT_ARTIFACT),
            "uncommitted_candidate_consensus": str(runtime.POST_UPDATE_ARTIFACT),
            "report": str(runtime.REPORT),
            "gpu_log": str(runtime.GPU_LOG),
            "snapshot": str(runtime.SNAPSHOT),
        },
    }
    return runtime.v40a.self_hashed(value)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v43i()
    runtime.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
