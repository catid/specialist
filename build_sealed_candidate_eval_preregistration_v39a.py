#!/usr/bin/env python3
"""Freeze the aggregate-only four-arm sealed-candidate evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import run_sealed_candidate_eval_v39a as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sealed_es_sft_base_fold3_eval_v39a.json"
).resolve()


def observed_bindings() -> dict:
    paths = {
        "runtime": ROOT / "run_sealed_candidate_eval_v39a.py",
        "model_config": runtime.MODEL / "config.json",
        "model_index": runtime.MODEL / "model.safetensors.index.json",
        "sft_adapter_config": runtime.SFT_ADAPTER / "adapter_config.json",
        "sft_adapter_weights": runtime.SFT_ADAPTER / "adapter_model.safetensors",
        "sft_report": runtime.SFT_REPORT,
        "es_report": runtime.ES_REPORT,
        "es_snapshot": runtime.ES_SNAPSHOT,
        "shadow": runtime.SHADOW,
        "split_manifest": runtime.SPLIT_MANIFEST,
        "ood_qa": runtime.OOD_QA,
        "ood_prose": runtime.OOD_PROSE,
        "layer_plan": runtime.LAYER_PLAN,
        "tuned_table": runtime.TUNED_FILE,
        "worker_v38a": ROOT / "eggroll_es_worker_v38a.py",
        "trainer_v38a": ROOT / "train_eggroll_es_equal_unit_v38a.py",
        "runtime_v38a": ROOT / "run_eggroll_es_equal_unit_v38a.py",
        "qa_quality": ROOT / "qa_quality.py",
        "reward": ROOT / "train_eggroll_es_specialist.py",
    }
    result = {key: runtime.file_sha256(path) for key, path in paths.items()}
    result["model_shards_content_sha256"] = runtime.canonical_sha256(
        runtime.model_shard_manifest()
    )
    return result


def _validate_seals(bindings: dict) -> dict:
    sft = json.loads(runtime.SFT_REPORT.read_text())
    es = json.loads(runtime.ES_REPORT.read_text())
    if (
        sft.get("status") != "complete_train_only_states_sealed_shadow_unopened"
        or sft.get("validation_ood_or_holdout_opened") is not False
        or sft.get("content_sha256_before_self_field")
        != runtime.canonical_sha256({
            key: value for key, value in sft.items()
            if key != "content_sha256_before_self_field"
        })
        or sft.get("artifacts", {}).get("output_file_sha256", {}).get(
            "final/adapter_model.safetensors"
        ) != bindings["sft_adapter_weights"]
        or sft.get("artifacts", {}).get("output_file_sha256", {}).get(
            "final/adapter_config.json"
        ) != bindings["sft_adapter_config"]
    ):
        raise RuntimeError("v39a SFT seal changed")
    if (
        es.get("status") != "complete_one_nonzero_update_state_sealed"
        or es.get("shadow_dev_external_eval_ood_or_holdout_opened") is not False
        or es.get("content_sha256_before_self_field")
        != runtime.canonical_sha256({
            key: value for key, value in es.items()
            if key != "content_sha256_before_self_field"
        })
        or es.get("artifacts", {}).get("selected_runtime_snapshot_sha256")
        != bindings["es_snapshot"]
        or not all(es.get("update_gates", {}).values())
    ):
        raise RuntimeError("v39a ES seal changed")
    return {
        "sft_report_content_sha256": sft["content_sha256_before_self_field"],
        "sft_global_step": sft["output_validation"]["final_global_step"],
        "es_report_content_sha256": es["content_sha256_before_self_field"],
        "es_final_identity": es["update"]["application"]["final_identity"],
    }


def build() -> dict:
    bindings = observed_bindings()
    seals = _validate_seals(bindings)
    tuned = json.loads(runtime.TUNED_FILE.read_text())
    tuned_content = runtime.canonical_sha256(tuned)
    result = {
        "schema": "sealed-candidate-eval-preregistration-v39a",
        "status": "preregistered_before_shadow_access",
        "created_at_utc": "2026-07-15T21:12:00+00:00",
        "heldout_or_holdout_access_authorized": False,
        "raw_shadow_or_ood_content_opened_before_preregistration": False,
        "implementation_bindings": bindings,
        "sealed_inputs": seals,
        "single_access_inputs": {
            "shadow": {"path": str(runtime.SHADOW), "file_sha256": bindings["shadow"]},
            "split_manifest": {"path": str(runtime.SPLIT_MANIFEST), "file_sha256": bindings["split_manifest"]},
            "ood_qa": {"path": str(runtime.OOD_QA), "file_sha256": bindings["ood_qa"]},
            "ood_prose": {"path": str(runtime.OOD_PROSE), "file_sha256": bindings["ood_prose"]},
        },
        "arms": {
            "base_a": "untouched base duplicate A",
            "base_b": "untouched base duplicate B",
            "sft_v37a": "sealed rank-32 LoRA served by exact vLLM LoRARequest",
            "es_v38a": "sealed selected runtime snapshot loaded with V38A identity verification",
        },
        "runtime": {
            "physical_gpu_ids": [0, 1, 2, 3], "engine_count": 4,
            "tensor_parallel_size": 1, "dtype": "bfloat16",
            "max_model_len": 2048, "gpu_memory_utilization": 0.82,
            "moe_backend": "triton", "enable_lora": True,
            "max_lora_rank": 32, "placement_group_lifetime": "driver_scoped",
            "tuned_table_file_sha256": bindings["tuned_table"],
            "tuned_table_content_sha256": tuned_content,
            "tuned_table_probe": "get_moe_configs(256,512,None) on every actor",
            "layer_plan_sha256": "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9",
        },
        "shadow_protocol": {
            "rows": 83, "conflict_units": 51,
            "teacher_forced": "mean answer-token logprob per row, row mean within unit, uniform mean of 51 units",
            "generated": {
                "template": "specialist_template", "temperature": 0.0,
                "top_p": 1.0, "max_tokens": 64,
                "seed": runtime.GENERATION_SEED,
                "reward": "specialist_reward using audited answer_score",
            },
            "base_duplicate_equivalence": "exact aggregate and hashed per-item numeric/output manifest",
            "protocol_leak_counters": [
                "protocol token emission", "normalized prompt echo", "empty extracted answer",
            ],
        },
        "selection_rule": {
            "candidates": ["sft_v37a", "es_v38a"],
            "lexicographic_higher_is_better": [
                "generated equal-unit mean reward", "generated exact count",
                "generated nonzero count", "teacher-forced equal-unit mean answer logprob",
            ],
            "final_tie_preference": "es_v38a",
            "shadow_gate": "candidate tuple strictly exceeds base and no protocol/leak counter increases",
        },
        "ood_gates": {
            "qa": "selected generated mean reward >= base and exact count >= base",
            "prose_point": "selected mean token logprob delta >= 0",
            "prose_bootstrap_lcb": {
                "paired_unit": "normalized_source_url", "samples": runtime.BOOTSTRAP_SAMPLES,
                "seed": runtime.BOOTSTRAP_SEED, "lower_percentile": 0.025,
                "requirement": "lower bound >= 0",
            },
        },
        "firewall": {
            "each_bound shadow/manifest/OOD file semantically read": "exactly once after preregistration",
            "heldout_or_holdout_paths": "forbidden by path guard and never enumerated",
            "raw_items": str(runtime.RAW),
            "raw_items_git_eligible": False,
            "aggregate_report": str(runtime.REPORT),
            "aggregate_only": True,
        },
        "required_gates": {
            "exclusive_idle_preflight": True,
            "exact_four_physical_pid_attribution_no_foreign_process": True,
            "all_four_positive_activity_on_shadow_ood_qa_ood_prose": True,
            "exact_tuned_table_loaded_all_actors": True,
            "base_duplicate_equivalence_all_splits": True,
            "es_snapshot_identity_exact": True,
            "exact_four_pg_created_to_removed": True,
            "final_gpu_idle": True,
            "no_heldout_or_holdout_access": True,
        },
    }
    result["content_sha256_before_self_field"] = runtime.canonical_sha256(result)
    return result


def main() -> None:
    value = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    runtime.atomic_json(OUTPUT, value)
    print(OUTPUT)
    print(value["content_sha256_before_self_field"])


if __name__ == "__main__":
    main()
