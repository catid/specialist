#!/usr/bin/env python3
"""Seal the matched-initialization, train-only LoRA-ES V43A contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_lora_es_equal_unit_v43a as runtime


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_step1_v43e.json"
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
    compact = {key: item for key, item in value.items()
               if key != "content_sha256_before_self_field"}
    if content != runtime.v40a.canonical_sha256(compact):
        raise RuntimeError(f"self-hashed parent changed: {path}")
    return value, content


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    smoke, smoke_content = _self_hash_valid(PARENT_SMOKE)
    equal, equal_content = _self_hash_valid(PARENT_EQUAL_UNIT)
    if (
        smoke.get("status") != "complete_train_only_four_gpu"
        or smoke.get("forward_probe", {}).get("plus_changed_forward") is not True
        or smoke.get("forward_probe", {}).get("minus_changed_forward") is not True
        or smoke.get("forward_probe", {}).get("restored_exact_forward_after_each_sign")
        is not True
        or smoke.get("snapshot_results") is None
        or equal.get("recipe", {}).get("train_bundle_content_sha256")
        != runtime.TRAIN_BUNDLE_SHA256
        or equal.get("recipe", {}).get("objective")
        != "teacher-forced mean answer-token logprob per row, mean rows within each conservative conflict unit, then mean 208 units"
    ):
        raise RuntimeError("v43a required parent evidence changed")
    bindings = runtime.implementation_bindings()
    if (
        bindings["source_weights"] != EXPECTED_SOURCE_WEIGHTS
        or bindings["staged_weights"] != EXPECTED_STAGED_WEIGHTS
        or bindings["dataset"] != runtime.DATASET_SHA256
        or bindings["split_manifest"] != runtime.SPLIT_MANIFEST_SHA256
    ):
        raise RuntimeError("v43a sealed inputs changed")
    tuned_table = json.loads(runtime.v40a.TUNED_FILE.read_text(encoding="utf-8"))
    tuned_table.pop("triton_version", None)
    tuned_content = runtime.v40a.canonical_sha256({
        int(key): item for key, item in tuned_table.items()
    })
    value = {
        "schema": "matched-lora-es-preregistration-v43e",
        "status": "preregistered_before_train_only_launch",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": (
            "Run one nonzero LoRA EGGROLL-ES update from the exact matched "
            "nonzero-A/zero-B initialization using the frozen v412 fold-3 "
            "equal-conflict-unit train objective."
        ),
        "shadow_dev_external_eval_ood_or_holdout_authorized": False,
        "sealed_holdout_opened": False,
        "quality_selection_or_promotion_authorized": False,
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
        },
        "recipe": {
            "model": str(runtime.v40a.MODEL),
            "matched_initialization": str(runtime.SOURCE),
            "matched_initialization_weights_sha256": EXPECTED_SOURCE_WEIGHTS,
            "matched_initialization_tensor_identity_sha256": (
                "2dcb9ab45ec26c7041b9782a30fe3c82b987b605b6b0bd95ab5b905b1371ae2e"
            ),
            "initialization": "35 independently seeded nonzero Kaiming A tensors and 35 exact-zero B tensors",
            "staged_initialization": str(runtime.STAGED),
            "staged_initialization_weights_sha256": EXPECTED_STAGED_WEIGHTS,
            "staging": "key-only Qwen3.5 outer language_model namespace transform",
            "dataset": str(runtime.DATASET),
            "dataset_sha256": runtime.DATASET_SHA256,
            "dataset_rows": 448,
            "conflict_units": 208,
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
            "coefficient_standardization": "population z-score with epsilon 1e-8",
            "cross_actor_score_atol": runtime.CROSS_ACTOR_SCORE_ATOL,
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
            "signed_sequence_presentations": 2 * runtime.POPULATION_SIZE * 448,
        },
        "runtime": {
            "tuned_folder": str(runtime.v40a.TUNED_FOLDER),
            "tuned_table_content_sha256": tuned_content,
        },
        "required_gates": {
            "all_four_gpus_have_attributed_positive_activity": True,
            "canonical_install_preserves_matched_init_train_score": True,
            "all_signed_directions_complete": True,
            "population_response_has_nonzero_spread": True,
            "exact_restore_preserves_prepopulation_train_score": True,
            "distributed_update_uses_fp32_collective": True,
            "all_ranks_commit_identical_canonical_state": True,
            "all_ranks_finalize_identical_committed_state_before_snapshot": True,
            "post_update_actor_aggregate_score_spread_at_most_1e_5": True,
            "post_update_exact_actor_hashes_retained_even_when_not_bitwise_equal": True,
            "base_weights_unchanged_at_every_state_boundary": True,
            "rank_zero_snapshot_exact_readback": True,
            "final_all_gpu_idle_cleanup": True,
        },
        "implementation_bindings": bindings,
        "artifacts": {
            "attempt": str(runtime.ATTEMPT),
            "run_directory": str(runtime.RUN_DIR),
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
