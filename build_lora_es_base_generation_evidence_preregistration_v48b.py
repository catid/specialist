#!/usr/bin/env python3
"""Seal the launchable four-GPU V48B train-only base-evidence run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lora_es_generation_boundary_sampling_v48a as boundary
import run_lora_es_base_generation_evidence_v48b as runtime
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


def build_v48b() -> dict:
    if v43i.v40a.file_sha256(PARENT) != PARENT_FILE_SHA256:
        raise RuntimeError("v48b V43I parent file changed")
    parent = json.loads(PARENT.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in parent.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        parent.get("content_sha256_before_self_field")
        != PARENT_CONTENT_SHA256
        or v43i.v40a.canonical_sha256(compact) != PARENT_CONTENT_SHA256
        or parent.get("sealed_holdout_opened") is not False
        or parent.get("access_contract", {}).get("protected_semantic_access")
        is not False
    ):
        raise RuntimeError("v48b V43I parent content changed")
    value = {
        "schema": "matched-lora-es-base-generation-evidence-preregistration-v48b",
        "status": "preregistered_before_train_only_launch",
        "retry_revision": "v48c_activation_retry",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "purpose": (
            "Collect content-free four-actor greedy metrics for every exact "
            "V43I train row at the matched initial LoRA state before any "
            "population or candidate selection."
        ),
        "protected_semantic_access_authorized": False,
        "shadow_ood_holdout_or_benchmark_authorized": False,
        "quality_selection_or_promotion_authorized": False,
        "parent_v43i_runtime": {
            "file_sha256": PARENT_FILE_SHA256,
            "content_sha256": PARENT_CONTENT_SHA256,
            "only_model_runtime_and_matched_initialization_reused": True,
            "v43i_anchor_or_population_inputs_opened": False,
        },
        "access_contract": {
            "only_runtime_semantic_paths_may_open": [
                str(runtime.TRAIN_DATASET), str(runtime.MEMBERSHIP),
            ],
            "original_split_manifest_opened_at_runtime": False,
            "prose_or_qa_proxy_anchor_opened": False,
            "shadow_ood_holdout_or_benchmark_path_opened": False,
            "builder_reads_train_semantics": False,
            "dry_run_reads_train_semantics": False,
            "dry_run_launches_model_or_gpu": False,
            "raw_question_answer_or_generation_text_may_be_persisted": False,
        },
        "recipe": {
            "model": str(v43i.v40a.MODEL),
            "train_dataset": str(runtime.TRAIN_DATASET),
            "train_dataset_sha256": runtime.EXPECTED_TRAIN_SHA256,
            "train_rows": 448,
            "train_conflict_units": 208,
            "train_bundle_content_sha256": v43i.TRAIN_BUNDLE_SHA256,
            "membership": str(runtime.MEMBERSHIP),
            "membership_file_sha256": runtime.EXPECTED_MEMBERSHIP_SHA256,
            "membership_content_sha256": (
                runtime.EXPECTED_MEMBERSHIP_CONTENT_SHA256
            ),
            "matched_initialization": str(v43i.SOURCE),
            "matched_initialization_weights_sha256": v43i.v40a.file_sha256(
                v43i.SOURCE_WEIGHTS
            ),
            "staged_initialization": str(v43i.STAGED),
            "staged_initialization_weights_sha256": v43i.v40a.file_sha256(
                v43i.STAGED_WEIGHTS
            ),
            "matched_master_sha256": runtime.v43m.V43I_RESTORED_MASTER_SHA256,
            "generation_params": dict(boundary.GENERATION_PARAMS_V48A),
            "physical_gpu_ids": [0, 1, 2, 3],
            "engine_count": 4,
            "tensor_parallel_size_per_engine": 1,
            "all_four_actors_receive_identical_448_prompt_order": True,
            "adapter_slot_activation": (
                "collective_rpc.add_lora_before_canonical_install"
            ),
            "adapter_slot_activation_actors": 4,
            "adapter_slot_activation_completions": 0,
            "worker_extension": v43i.WORKER_EXTENSION,
        },
        "evidence_contract": {
            "rows": 448,
            "actors_per_row": 4,
            "greedy_completions": 1792,
            "maximum_decode_tokens": 64,
            "worst_case_decode_tokens": 114688,
            "persisted_per_actor_fields": [
                "actor_rank", "prediction_sha256", "f1", "exact", "nonzero",
            ],
            "subset_preview_must_select_64_distinct_conflict_units": True,
            "subset_is_not_persisted_or_used_for_population_by_this_run": True,
        },
        "runtime": dict(parent["runtime"]),
        "implementation_bindings": runtime.implementation_bindings_v48b(),
        "artifacts": {
            "attempt": str(runtime.ATTEMPT),
            "run_directory": str(runtime.RUN_DIR),
            "evidence": str(runtime.EVIDENCE),
            "report": str(runtime.REPORT),
            "failure": str(runtime.FAILURE),
            "gpu_log": str(runtime.GPU_LOG),
        },
        "required_gates": {
            "exclusive_idle_four_gpu_preflight": True,
            "all_four_lora_slots_activated_before_canonical_install": True,
            "exact_matched_master_same_on_all_actors": True,
            "all_four_gpus_attributed_positive": True,
            "448_completions_per_actor": True,
            "numeric_hash_only_evidence": True,
            "deterministic_subset_preview_valid": True,
            "strict_four_engine_cleanup_and_idle": True,
        },
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
        "current_fixed_holdout_cycle_eligible": False,
    }
    value["content_sha256_before_self_field"] = v43i.v40a.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v48b()
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
