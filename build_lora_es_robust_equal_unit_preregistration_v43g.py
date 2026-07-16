#!/usr/bin/env python3
"""Seal the robust complete-actor-block, train-only LoRA-ES V43G contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_robust_consensus_v43g as numeric
import run_lora_es_robust_equal_unit_v43g as runtime


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_robust_v43g.json"
).resolve()
PARENT_SMOKE = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v41a_canonical_lora_es_state_smoke_retry_r2/"
    "lora_es_state_smoke_report_v41a_retry_r2.json"
).resolve()
PARENT_EQUAL_UNIT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "equal_unit_fold3_nonzero_v38a.json"
).resolve()
PARENT_V43F_FAILURE = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43f_matched_lora_es_fold3_pop8_calibrated_retry/failure_v43f.json"
).resolve()
PARENT_V43F_CALIBRATION = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43f_matched_lora_es_fold3_pop8_calibrated_retry/"
    "numeric_calibration_v43f.json"
).resolve()
PARENT_V43F_RELIABILITY = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43f_matched_lora_es_fold3_pop8_calibrated_retry/"
    "population_reliability_v43f.json"
).resolve()
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


def _self_hash_valid(path: Path) -> tuple[dict, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if content != runtime.v40a.canonical_sha256(compact):
        raise RuntimeError(f"v43g self-hashed parent changed: {path}")
    return value, content


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    smoke, smoke_content = _self_hash_valid(PARENT_SMOKE)
    equal, equal_content = _self_hash_valid(PARENT_EQUAL_UNIT)
    v43f_failure, v43f_failure_content = _self_hash_valid(PARENT_V43F_FAILURE)
    v43f_calibration, v43f_calibration_content = _self_hash_valid(
        PARENT_V43F_CALIBRATION
    )
    v43f_reliability, v43f_reliability_content = _self_hash_valid(
        PARENT_V43F_RELIABILITY
    )
    if (
        smoke.get("status") != "complete_train_only_four_gpu"
        or smoke.get("forward_probe", {}).get("plus_changed_forward") is not True
        or smoke.get("forward_probe", {}).get("minus_changed_forward") is not True
        or smoke.get("forward_probe", {}).get(
            "restored_exact_forward_after_each_sign"
        ) is not True
        or smoke.get("snapshot_results") is None
        or equal.get("recipe", {}).get("train_bundle_content_sha256")
        != runtime.TRAIN_BUNDLE_SHA256
        or equal.get("recipe", {}).get("objective")
        != (
            "teacher-forced mean answer-token logprob per row, mean rows "
            "within each conservative conflict unit, then mean 208 units"
        )
        or v43f_failure.get("schema") != "matched-lora-es-failure-v43f"
        or "reliability=0.6581937522502567"
        not in v43f_failure.get("message", "")
        or "bound_fraction=4.5349602966089995"
        not in v43f_failure.get("message", "")
        or v43f_failure.get(
            "shadow_dev_external_eval_ood_or_holdout_opened"
        ) is not False
    ):
        raise RuntimeError("v43g required train-only parent evidence changed")
    diagnostic = numeric.diagnose_v43f_artifacts_v43g(
        v43f_calibration, v43f_reliability,
    )
    if (
        diagnostic["v43f_reliability"] != 0.6581937522502567
        or diagnostic["v43f_calibration_bound_fraction"]
        != 4.5349602966089995
        or diagnostic["largest_replicate_disagreement_direction_indices"]
        != [3, 4, 7]
        or diagnostic["linear_response_projected_v43g_reliability"]
        < 0.93
    ):
        raise RuntimeError("v43g V43F diagnostic derivation changed")
    bindings = runtime.implementation_bindings()
    if (
        bindings["source_weights"] != EXPECTED_SOURCE_WEIGHTS
        or bindings["staged_weights"] != EXPECTED_STAGED_WEIGHTS
        or bindings["dataset"] != runtime.DATASET_SHA256
        or bindings["split_manifest"] != runtime.SPLIT_MANIFEST_SHA256
    ):
        raise RuntimeError("v43g sealed inputs changed")
    tuned_table = json.loads(
        runtime.v40a.TUNED_FILE.read_text(encoding="utf-8")
    )
    tuned_table.pop("triton_version", None)
    tuned_content = runtime.v40a.canonical_sha256({
        int(key): item for key, item in tuned_table.items()
    })
    value = {
        "schema": "matched-lora-es-preregistration-v43g",
        "status": "preregistered_before_train_only_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Raise train-only LoRA-ES response SNR with doubled exploration sigma, "
            "a complete four-actor antithetic block, robust median aggregation, "
            "and EGGROLL centered-rank fitness shaping while retaining exact FP32 "
            "state consensus and post-update numerical equivalence."
        ),
        "shadow_dev_external_eval_ood_or_holdout_authorized": False,
        "sealed_holdout_opened": False,
        "quality_selection_or_promotion_authorized": False,
        "access_contract": {
            "protected_semantic_access": False,
            "only_runtime_train_path_may_open": str(runtime.DATASET),
            "builder_reads_train_examples": False,
            "builder_reads_protected_examples": False,
            "dry_run_reads_train_examples": False,
            "dry_run_hashes_bound_input_files": True,
            "dry_run_loads_model_runtime": False,
            "dry_run_launches_gpu": False,
            "v43f_parent_use": "aggregate numeric diagnostics only",
        },
        "parents": {
            "canonical_state_smoke": {
                "path": str(PARENT_SMOKE),
                "file_sha256": runtime.v40a.file_sha256(PARENT_SMOKE),
                "content_sha256": smoke_content,
            },
            "equal_unit_dense_precedent": {
                "path": str(PARENT_EQUAL_UNIT),
                "file_sha256": runtime.v40a.file_sha256(PARENT_EQUAL_UNIT),
                "content_sha256": equal_content,
            },
            "v43f_failure": {
                "path": str(PARENT_V43F_FAILURE),
                "file_sha256": runtime.v40a.file_sha256(PARENT_V43F_FAILURE),
                "content_sha256": v43f_failure_content,
                "used_for": "train-only design motivation and frozen derivation",
            },
            "v43f_numeric_calibration": {
                "path": str(PARENT_V43F_CALIBRATION),
                "file_sha256": runtime.v40a.file_sha256(PARENT_V43F_CALIBRATION),
                "content_sha256": v43f_calibration_content,
                "used_for": "historical numerical ceiling and frozen diagnosis",
            },
            "v43f_population_reliability": {
                "path": str(PARENT_V43F_RELIABILITY),
                "file_sha256": runtime.v40a.file_sha256(PARENT_V43F_RELIABILITY),
                "content_sha256": v43f_reliability_content,
                "used_for": "frozen SNR projection; no acceptance threshold fit",
            },
        },
        "v43f_train_only_diagnostic_derivation": diagnostic,
        "recipe": {
            "model": str(runtime.v40a.MODEL),
            "matched_initialization": str(runtime.SOURCE),
            "matched_initialization_weights_sha256": EXPECTED_SOURCE_WEIGHTS,
            "matched_initialization_tensor_identity_sha256": (
                "2dcb9ab45ec26c7041b9782a30fe3c82b987b605b6b0bd95ab5b905b1371ae2e"
            ),
            "initialization": (
                "35 independently seeded nonzero Kaiming A tensors and "
                "35 exact-zero B tensors"
            ),
            "staged_initialization": str(runtime.STAGED),
            "staged_initialization_weights_sha256": EXPECTED_STAGED_WEIGHTS,
            "staging": "key-only Qwen3.5 outer language_model namespace transform",
            "dataset": str(runtime.DATASET),
            "dataset_sha256": runtime.DATASET_SHA256,
            "dataset_rows": 448,
            "conflict_units": numeric.EXPECTED_CONFLICT_UNITS_V43G,
            "split_manifest": str(runtime.SPLIT_MANIFEST),
            "split_manifest_sha256": runtime.SPLIT_MANIFEST_SHA256,
            "train_bundle_content_sha256": runtime.TRAIN_BUNDLE_SHA256,
            "objective": (
                "teacher-forced gold-answer token logprob; mean answer tokens "
                "within row, equal mass per conservative conflict unit"
            ),
            "dense_reward_config": runtime.anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
            "population_size": runtime.POPULATION_SIZE,
            "seeds": runtime.SEEDS,
            "seed_sha256": runtime.v40a.canonical_sha256(runtime.SEEDS),
            "antithetic_sign_order": [1, -1],
            "sigma": runtime.SIGMA,
            "alpha": runtime.ALPHA,
            "sigma_change_from_v43f": "0.0003 to 0.0006; alpha remains 0.00015",
            "fitness_shaping": (
                "centered ranks over 16 median signed scores; "
                "coefficient=u_plus-u_minus"
            ),
            "signed_score_location": "median-of-four actor scores per sign",
            "robust_central_location": "median-of-four paired central differences",
            "signed_replicates_per_direction": numeric.SIGNED_REPLICATES_V43G,
            "signed_replication_assignment": (
                "complete four-actor paired antithetic block per direction"
            ),
            "minimum_response_reliability": numeric.RELIABILITY_MINIMUM_V43G,
            "minimum_split_half_spearman": (
                numeric.SPLIT_HALF_SPEARMAN_MINIMUM_V43G
            ),
            "split_half_actor_groups": [[0, 1], [2, 3]],
            "worker_extension": runtime.WORKER_EXTENSION,
            "canonical_trainable_dtype": "float32",
            "runtime_adapter_dtype": "bfloat16",
            "canonical_tensor_count": 70,
            "canonical_elements": 4_528_128,
            "runtime_view_count": 82,
            "physical_gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
            "tuned_table_content_sha256": tuned_content,
            "signed_sequence_presentations": (
                2 * numeric.SIGNED_REPLICATES_V43G
                * runtime.POPULATION_SIZE * 448
            ),
        },
        "numeric_calibration": {
            "state": "sealed master plus one fixed Gaussian direction",
            "synthetic_noise_seed": numeric.CALIBRATION_NOISE_SEED_V43G,
            "synthetic_sigma": runtime.CALIBRATION_SIGMA,
            "synthetic_sigma_derivation": "alpha/sqrt(population_size)",
            "same_exact_state_on_all_four_actors": True,
            "warmup_repeats": numeric.CALIBRATION_WARMUPS_V43G,
            "warmups_discarded": True,
            "retained_repeats_per_actor": numeric.CALIBRATION_REPEATS_V43G,
            "persist_per_actor_repeat_conflict_unit_aggregates": True,
            "bootstrap_method": (
                "conflict-unit cluster bootstrap of maximum actor range over repeats"
            ),
            "bootstrap_rng": "numpy.random.PCG64",
            "bootstrap_resamples": numeric.BOOTSTRAP_RESAMPLES_V43G,
            "bootstrap_seed": numeric.CALIBRATION_BOOTSTRAP_SEED_V43G,
            "familywise_confidence": numeric.BOOTSTRAP_CONFIDENCE_V43G,
            "two_metric_bonferroni_adjustment": True,
            "historical_catastrophic_divergence_ceiling": (
                numeric.V43F_HISTORICAL_EQUAL_UNIT_BOUND
            ),
            "population_gate_use": (
                "fresh observed fixed-panel maximum actor spread must not exceed "
                "the V43F historical ceiling"
            ),
            "population_response_scale_use": False,
            "post_update_equivalence_margin_use": True,
            "threshold_fitted_to_v43g_population": False,
        },
        "post_update_consensus": {
            "retained_repeats_per_actor": numeric.POST_UPDATE_REPEATS_V43G,
            "persist_per_actor_repeat_conflict_unit_aggregates": True,
            "bootstrap_method": (
                "paired conflict-unit and repeat bootstrap with Bonferroni "
                "simultaneous intervals for all six actor pairs and two metrics"
            ),
            "bootstrap_rng": "numpy.random.PCG64",
            "bootstrap_resamples": numeric.BOOTSTRAP_RESAMPLES_V43G,
            "bootstrap_seed": numeric.POST_UPDATE_BOOTSTRAP_SEED_V43G,
            "familywise_confidence": numeric.BOOTSTRAP_CONFIDENCE_V43G,
            "acceptance": (
                "all pairwise intervals and both actor-mean spreads inside "
                "the pre-population calibration margins"
            ),
            "exact_score_hash_equality_required": False,
            "exact_score_hashes_retained": True,
        },
        "runtime": {
            "tuned_folder": str(runtime.v40a.TUNED_FOLDER),
            "tuned_table_content_sha256": tuned_content,
        },
        "required_gates": {
            "all_four_gpus_have_attributed_positive_activity": True,
            "canonical_install_preserves_matched_init_train_score": True,
            "fixed_nonzero_calibration_state_exact_across_ranks": True,
            "calibration_restores_exact_master": True,
            "calibration_records_persist_before_population": True,
            "all_signed_directions_have_complete_four_actor_blocks": True,
            "replicated_direction_state_hashes_exact": True,
            "population_response_reliability_at_least_0_8": True,
            "population_split_half_spearman_at_least_0_7": True,
            "fresh_calibration_inside_v43f_historical_ceiling": True,
            "centered_rank_utility_update_nonzero": True,
            "exact_restore_preserves_prepopulation_train_score": True,
            "distributed_update_uses_fp32_collective": True,
            "all_ranks_commit_identical_canonical_fp32_state": True,
            "all_runtime_materialization_hashes_exact_across_ranks": True,
            "all_ranks_finalize_identical_committed_state_before_snapshot": True,
            "rank_zero_snapshot_exact_readback": True,
            "post_update_pairwise_99pct_simultaneous_equivalence": True,
            "post_update_actor_mean_spreads_inside_calibrated_bounds": True,
            "base_weights_unchanged_at_every_state_boundary": True,
            "final_all_gpu_idle_cleanup": True,
        },
        "implementation_bindings": bindings,
        "artifacts": {
            "attempt": str(runtime.ATTEMPT),
            "run_directory": str(runtime.RUN_DIR),
            "calibration": str(runtime.CALIBRATION_ARTIFACT),
            "population_reliability": str(runtime.RELIABILITY_ARTIFACT),
            "post_update_consensus": str(runtime.POST_UPDATE_ARTIFACT),
            "report": str(runtime.REPORT),
            "gpu_log": str(runtime.GPU_LOG),
            "snapshot": str(runtime.SNAPSHOT),
        },
    }
    value = runtime.v40a.self_hashed(value)
    runtime.v40a.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": runtime.v40a.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
