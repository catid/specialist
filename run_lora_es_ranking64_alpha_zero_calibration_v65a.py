#!/usr/bin/env python3
"""Run the exact-64 ranking evaluator's alpha-zero calibration on four GPUs.

V65A is a prerequisite measurement, not an HPO population.  It reads only
the authorized 64-row byte prefix, discards four fixed warmup periods, then
counterbalances four scored periods while the exact V434 LoRA state remains
unchanged.  No candidate, update, projection, holdback, sentinel, OOD, or
protected semantic path is authorized here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import build_lora_es_ranking64_alpha_zero_preregistration_v65a as builder65a
import eggroll_es_worker_robust_sampling_v65 as scoring65
import lora_es_nested_population_v52 as design52
import lora_es_ranking64_alpha_zero_calibration_v65a as analysis65a
import lora_es_robust_sampling_population_v65 as design65
import run_lora_es_baseline_census_v61a as runtime61a
import run_lora_es_nested_population_v52 as runtime52
import run_lora_es_paired_null_calibration_v61c as runtime61c
import run_lora_es_robust_sampling_population_v65 as runtime65
import run_lora_es_v59_vs_v434_robust_confirmation_v64 as runtime64


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v65a_r1_ranking64_alpha_zero_calibration"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
EVIDENCE = (RUN_DIR / "ranking64_alpha_zero_evidence_v65a_r1.json").resolve()
ANALYSIS = (RUN_DIR / "ranking64_alpha_zero_analysis_v65a_r1.json").resolve()
REPORT = (RUN_DIR / "ranking64_alpha_zero_report_v65a_r1.json").resolve()
FAILURE = (RUN_DIR / "failure_v65a_r1.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v65a_r1.jsonl").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "ranking64_alpha_zero_calibration_v65a_r1.json"
).resolve()
WORKER_EXTENSION_V65A = (
    "eggroll_es_worker_lora_v65a.LoRAAdapterStateWorkerExtensionV65A"
)

RUNTIME_CONTROLS_V65A = dict(analysis65a.ENGINE_CONTROLS_V65A)


def artifacts_v65a() -> dict:
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "evidence": str(EVIDENCE),
        "analysis": str(ANALYSIS),
        "report": str(REPORT),
        "failure": str(FAILURE),
        "gpu_log": str(GPU_LOG),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--execute", action="store_true")
    return value


def _binding_exact_v65a(binding) -> bool:
    if not isinstance(binding, dict) or set(binding) != {"path", "file_sha256"}:
        return False
    path = Path(binding["path"]).resolve()
    return path.is_file() and design65.file_sha256_v65(path) == binding["file_sha256"]


def load_preregistration_v65a(args) -> dict:
    path = Path(args.preregistration).resolve()
    if design65.file_sha256_v65(path) != args.preregistration_sha256:
        raise RuntimeError("v65a preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    authorization = value.get("authorization", {})
    access = value.get("access_contract", {})
    recipe = value.get("fixed_calibration_recipe", {})
    numeric = value.get("numeric_analysis_contract", {})
    transfer = numeric.get("future_v65_null_bound_transfer", {})
    controls = recipe.get("runtime_determinism_controls", {})
    live_receipt = recipe.get("sanitized_live_engine_and_cache_receipt", {})
    integrity = value.get("required_integrity_gates", {})
    bindings = value.get("implementation_bindings", {})
    predecessor = value.get("predecessor_failed_attempt")
    expected_predecessor = (
        builder65a.predecessor_failed_attempt_binding_v65a()
    )
    required_binding_keys = (
        builder65a.REQUIRED_IMPLEMENTATION_BINDING_KEYS_V65A
    )
    expected_bindings = builder65a.implementation_bindings_v65a()
    if (
        value.get("schema")
        != "v65a-ranking64-alpha-zero-calibration-preregistration"
        or value.get("status")
        != "sealed_before_v65a_train_semantics_model_ray_or_gpu_access"
        or value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or design65.self_content_sha256_v65(value)
        != args.preregistration_content_sha256
        or value.get("artifacts") != artifacts_v65a()
        or predecessor != expected_predecessor
        or authorization.get("gpu_launch") is not True
        or authorization.get("alpha_zero_calibration") is not True
        or authorization.get("physical_gpu_ids") != [0, 1, 2, 3]
        or authorization.get("actors") != 4
        or authorization.get("tensor_parallel_size_per_actor") != 1
        or any(authorization.get(key) is not False for key in (
            "projection", "optimizer_update", "adapter_update", "candidate",
            "candidate_snapshot", "hpo_population",
            "train_holdback", "exact_sentinel", "unused_reserve", "ood_shadow",
            "protected_semantics", "terminal_holdout", "promotion",
        ))
        or access.get("decode_exactly_first_64_v61c_ranking_rows") is not True
        or access.get("decode_v61c_row_64_or_later") is not False
        or access.get("ranking_prefix_bytes") != runtime65.RANKING_PREFIX_BYTES_V65
        or access.get("ranking_prefix_sha256")
        != runtime65.RANKING_PREFIX_SHA256_V65
        or access.get("source_file_size_metadata_bytes")
        != analysis65a.RANKING_SOURCE_FILE_SIZE_BYTES_V65A
        or access.get("live_authorized_prefix_pread_count") != 2
        or access.get("postrun_prefix_integrity_pread_decodes_semantics")
        is not False
        or recipe.get("rows_per_actor_call") != 64
        or recipe.get("lora_request") != {
            "name": "v434_ranking64_alpha_zero_v65a",
            "integer_id": 1,
            "path": str(design52.STAGED_V52),
        }
        or recipe.get("actors") != 4
        or recipe.get("unscored_warmup_periods") != 4
        or recipe.get("scored_periods") != 4
        or recipe.get("paired_replicas_per_unit") != 8
        or recipe.get("scored_label_plan") != analysis65a.LABEL_PLAN_V65A
        or recipe.get("pair_periods")
        != [list(pair) for pair in analysis65a.PAIR_PERIODS_V65A]
        or recipe.get("common_generation_seed")
        != analysis65a.COMMON_GENERATION_SEED_V65A
        or recipe.get("generation_params_without_seed")
        != analysis65a.GENERATION_PARAMS_WITHOUT_SEED_V65A
        or recipe.get("exact_master_rematerialization", {}).get("rpc")
        != "rematerialize_exact_master_v65a"
        or recipe.get("exact_master_rematerialization", {}).get(
            "period_slot_write_receipts_required"
        ) != 8
        or recipe.get("exact_master_rematerialization", {}).get(
            "read_only_live_slot_receipts_required"
        ) != 16
        or recipe.get("exact_master_rematerialization", {}).get(
            "after_generation_receipt_may_write_or_reset_slot"
        ) is not False
        or recipe.get("adaptive_retry_drop_reorder_or_early_stop") is not False
        or recipe.get("warmup_generation_completions_discarded") != 1024
        or recipe.get("scored_generation_completions") != 1024
        or recipe.get("total_generation_completions") != 2048
        or controls != RUNTIME_CONTROLS_V65A
        or live_receipt.get(
            "registered_and_gpu_slot_match_exact_staged_v434_bytes"
        ) is not True
        or live_receipt.get("staged_v434_weights_file_sha256")
        != design52.STAGED_WEIGHTS_SHA256_V52
        or integrity.get(
            "adapter_source_and_stage_contract_reverified_unchanged_postcleanup"
        ) is not True
        or numeric.get("resampled_axis") != "conflict_unit_only"
        or numeric.get("within_unit_actor_pair_replicas_preserved_and_averaged")
        != 8
        or numeric.get("bootstrap_replicates")
        != analysis65a.BOOTSTRAP_REPLICATES_V65A
        or numeric.get("bootstrap_seed") != analysis65a.BOOTSTRAP_SEED_V65A
        or numeric.get("bootstrap_index_matrix_sha256")
        != hashlib.sha256(
            analysis65a.frozen_bootstrap_indices_v65a().astype(
                "<i8", copy=False,
            ).tobytes(order="C")
        ).hexdigest()
        or numeric.get("joint_composite_weights")
        != analysis65a.COMPOSITE_WEIGHTS_V65A
        or transfer.get("outcome_independent_field_mapping")
        != analysis65a.FUTURE_V65_NULL_BOUND_TRANSFER_V65A
        or transfer.get("required_spread_gates") != {
            "pooled_joint_composite": "spread_strictly_greater_than_2*B_C",
            "each_pass_joint_composite": (
                "spread_strictly_greater_than_2*B_C_pass"
            ),
            "generated_f1_when_used": "spread_strictly_greater_than_2*B_F",
            "stability_when_used": "spread_strictly_greater_than_2*B_S",
            "stability_coefficient_when_gate_not_met": 0.0,
            "stability_gate_not_met_causes_population_failure": False,
        }
        or transfer.get("mapping_or_gates_may_change_after_observing_v65a")
        is not False
        or transfer.get(
            "rebind_or_launch_requires_required_alpha_zero_gate_passed"
        ) is not True
        or transfer.get(
            "failed_required_alpha_zero_gate_forbids_bound_rebinding_"
            "and_v65_launch"
        ) is not True
        or value.get("runtime") != design52.RUNTIME_V52
        or value.get("required_python") != str(design52.REQUIRED_PYTHON_V52)
        or not isinstance(bindings, dict)
        or not required_binding_keys.issubset(bindings)
        or bindings != expected_bindings
        or value.get("implementation_closure_manifest_sha256")
        != design65.canonical_sha256_v65({
            key: binding["file_sha256"]
            for key, binding in sorted(bindings.items())
            if isinstance(binding, dict) and "file_sha256" in binding
        })
        or not all(
            _binding_exact_v65a(binding)
            for binding in bindings.values()
        )
    ):
        raise RuntimeError("v65a preregistration contract changed")
    return value


def engine_kwargs_v65a(v40a, precision="bfloat16") -> dict:
    return {
        "model": str(v40a.MODEL),
        "tensor_parallel_size": 1,
        "worker_extension_cls": WORKER_EXTENSION_V65A,
        "dtype": precision,
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
        "enforce_eager": True,
        "async_scheduling": False,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
        "scheduling_policy": "fcfs",
        "gpu_memory_utilization": 0.82,
        "max_model_len": 2048,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "mm_processor_cache_gb": 0,
        "skip_mm_profiling": True,
        "moe_backend": "triton",
        "enable_lora": True,
        "max_lora_rank": 32,
        "max_loras": 1,
        "max_cpu_loras": 2,
    }


def lora_request_v65a(prior):
    from vllm.lora.request import LoRARequest

    return LoRARequest(
        "v434_ranking64_alpha_zero_v65a",
        1,
        str(design52.STAGED_V52),
        base_model_name=str(prior.v40a.MODEL),
    )


def make_trainer_v65a(
    preregistration: dict, prior, construction_state: dict | None = None,
):
    """Construct the exact evaluator shape and expose a live config receipt."""
    if construction_state is None:
        construction_state = {}
    v40a = prior.v40a
    parent = v40a.base.load_trainer()
    expected_tuned_content = preregistration["runtime"][
        "tuned_table_content_sha256"
    ]

    class Ranking64TrainerV65A(parent):
        def launch_engines(
            self, num_engines=4, n_gpu_per_vllm_engine=1,
            model_name="unused", precision="bfloat16",
        ):
            if int(num_engines) != 4 or int(n_gpu_per_vllm_engine) != 1:
                raise RuntimeError("v65a requires four TP1 engines")
            import ray
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy,
            )

            class Ranking64LLMV65A(ESNcclLLM):
                def runtime_identity_v65a(self):
                    import torch
                    import vllm.envs as vllm_envs
                    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe

                    raw = ray.get_gpu_ids()
                    if len(raw) != 1:
                        raise RuntimeError("v65a actor does not own exactly one GPU")
                    physical = v40a.normalize_gpu_id(raw[0])
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
                    batch_env = os.environ.get("VLLM_BATCH_INVARIANT")
                    config = self.llm_engine.vllm_config
                    scheduler = config.scheduler_config
                    cache = config.cache_config
                    lora = config.lora_config
                    fused_moe.get_moe_configs.cache_clear()
                    tuned = fused_moe.get_moe_configs(256, 512, None)
                    scheduler_class = scheduler.get_scheduler_cls().__name__
                    observed = {
                        "tensor_parallel_size": (
                            config.parallel_config.tensor_parallel_size
                        ),
                        "dtype": str(config.model_config.dtype),
                        "max_model_len": config.model_config.max_model_len,
                        "gpu_memory_utilization": cache.gpu_memory_utilization,
                        "max_loras": lora.max_loras,
                        "max_cpu_loras": lora.max_cpu_loras,
                        "max_lora_rank": lora.max_lora_rank,
                        "VLLM_BATCH_INVARIANT": vllm_envs.VLLM_BATCH_INVARIANT,
                        "enable_prefix_caching": cache.enable_prefix_caching,
                        "enable_chunked_prefill": scheduler.enable_chunked_prefill,
                        "async_scheduling": scheduler.async_scheduling,
                        "max_num_seqs": scheduler.max_num_seqs,
                        "max_num_batched_tokens": scheduler.max_num_batched_tokens,
                        "scheduling_policy": scheduler.policy,
                        "enforce_eager": config.model_config.enforce_eager,
                    }
                    if (
                        visible != str(physical)
                        or torch.cuda.device_count() != 1
                        or torch.cuda.current_device() != 0
                        or folder != str(v40a.TUNED_FOLDER)
                        or vllm_envs.VLLM_TUNED_CONFIG_FOLDER
                        != str(v40a.TUNED_FOLDER)
                        or batch_env != "0"
                        or observed != RUNTIME_CONTROLS_V65A
                        or scheduler_class != "Scheduler"
                        or v40a.canonical_sha256(tuned) != expected_tuned_content
                    ):
                        raise RuntimeError("v65a live evaluator configuration changed")
                    return {
                        "schema": "ranking64-alpha-zero-actor-identity-v65a",
                        "pid": os.getpid(),
                        "physical_gpu_id": physical,
                        "cuda_visible_devices": visible,
                        "cuda_current_device": 0,
                        "runtime_determinism_controls": observed,
                        "scheduler_class": scheduler_class,
                        "tuned_folder": folder,
                        "tuned_table_content_sha256": v40a.canonical_sha256(tuned),
                        "submitted_request_batch_size": 64,
                        "generation_only": True,
                        "global_batch_invariance_claimed": False,
                    }

            groups = [
                placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK")
                for _ in range(4)
            ]
            ray.get([group.ready() for group in groups])
            strategies = [PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            ) for group in groups]
            kwargs = engine_kwargs_v65a(v40a, precision)
            engines = [ray.remote(
                num_cpus=0, num_gpus=1, scheduling_strategy=strategy,
            )(Ranking64LLMV65A).options(runtime_env={"env_vars": {
                "VLLM_TUNED_CONFIG_FOLDER": str(v40a.TUNED_FOLDER),
                "VLLM_BATCH_INVARIANT": "0",
            }}).remote(**kwargs) for strategy in strategies]
            return engines, groups

    saved = (v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION)
    construction_state["saved"] = saved
    v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = (
        EXPERIMENT, RUN_DIR, WORKER_EXTENSION_V65A,
    )
    try:
        # Retain the partially initialized object if engine/placement-group
        # construction raises.  The outer failure path can then attempt strict
        # teardown before the unconditional Ray shutdown and four-GPU idle proof.
        trainer = Ranking64TrainerV65A.__new__(Ranking64TrainerV65A)
        construction_state["trainer"] = trainer
        Ranking64TrainerV65A.__init__(
            trainer,
            model_name=str(v40a.MODEL), checkpoint=None,
            sigma=0.0, alpha=0.0, population_size=4,
            reward_shaping="z-scores", num_iterations=0,
            max_tokens=4, batch_size=1, mini_batch_size=1,
            reward_function=v40a.base.specialist_reward,
            template_function=lambda value: value,
            train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
            n_vllm_engines=4, n_gpu_per_vllm_engine=1,
            logging="none", global_seed=20_260_716,
            use_gpus="0,1,2,3", experiment_name=EXPERIMENT,
            wandb_project="none", save_best_models=False,
            reward_function_timeout=10,
            output_directory=str(RUN_DIR.parent),
        )
        import ray
        trainer._resolve = lambda handles: ray.get(handles)
    except BaseException:
        raise
    return trainer, saved


def validate_actor_identities_v65a(
    identities: object, tuned_sha256: str,
) -> dict[int, int]:
    if not isinstance(identities, list) or len(identities) != 4:
        raise RuntimeError("v65a actor identity coverage changed")
    mapping = {}
    pids = set()
    expected_keys = {
        "schema", "pid", "physical_gpu_id", "cuda_visible_devices",
        "cuda_current_device", "runtime_determinism_controls",
        "scheduler_class", "tuned_folder", "tuned_table_content_sha256",
        "submitted_request_batch_size", "generation_only",
        "global_batch_invariance_claimed",
    }
    for item in identities:
        gpu = item.get("physical_gpu_id") if isinstance(item, dict) else None
        pid = item.get("pid") if isinstance(item, dict) else None
        if (
            not isinstance(item, dict)
            or set(item) != expected_keys
            or item.get("schema") != "ranking64-alpha-zero-actor-identity-v65a"
            or gpu not in range(4)
            or isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0
            or item.get("cuda_visible_devices") != str(gpu)
            or item.get("cuda_current_device") != 0
            or item.get("runtime_determinism_controls")
            != RUNTIME_CONTROLS_V65A
            or item.get("scheduler_class") != "Scheduler"
            or item.get("tuned_folder") != str(design52.RUNTIME_V52["tuned_folder"])
            or item.get("tuned_table_content_sha256") != tuned_sha256
            or item.get("submitted_request_batch_size") != 64
            or item.get("generation_only") is not True
            or item.get("global_batch_invariance_claimed") is not False
            or gpu in mapping or pid in pids
        ):
            raise RuntimeError("v65a actor identity or live config changed")
        mapping[gpu] = pid
        pids.add(pid)
    if set(mapping) != set(range(4)) or len(pids) != 4:
        raise RuntimeError("v65a physical GPU/PID mapping incomplete")
    return mapping


def _validate_batches_v65a(batches: object, period_kind: str) -> list:
    if (
        not isinstance(batches, list) or len(batches) != 4
        or any(not isinstance(batch, list) or len(batch) != 64 for batch in batches)
    ):
        raise RuntimeError(
            f"v65a {period_kind} exact four-actor/64-request coverage changed"
        )
    return batches


def _state_receipt_v65a(
    period_kind: str, period_index: int,
    before: dict, after: dict, installed: dict,
) -> dict:
    if (
        period_kind not in ("unscored_warmup", "scored")
        or isinstance(period_index, bool) or not isinstance(period_index, int)
        or period_index < 0 or before != after
        or any(
            before.get(key) != installed.get(key)
            for key in (
                "canonical_fp32_master_sha256",
                "canonical_master_identity_sha256",
                "bf16_runtime_values_sha256",
            )
        )
    ):
        raise RuntimeError(f"v65a exact V434 state changed across {period_kind}")
    return {
        "period_kind": period_kind,
        "period_index": period_index,
        "before": before,
        "after": after,
        "identical_v434_state": True,
    }


def _score_period_v65a(
    panel_items: list[dict], answers: list[str], batches: list,
) -> list[list[dict]]:
    result = []
    for batch in batches:
        rows = scoring65.score_generation_batch_v65(
            panel_items, answers, batch,
        )
        result.append([{
            key: value for key, value in row.items()
            if key != "prediction_sha256"
        } for row in rows])
    return result


def _self_hashed_v65a(value: dict) -> dict:
    result = dict(value)
    result.pop("content_sha256_before_self_field", None)
    result["content_sha256_before_self_field"] = (
        design65.canonical_sha256_v65(result)
    )
    return result


_MASTER_CORE_KEYS_V65A = (
    "canonical_fp32_master_sha256",
    "canonical_master_identity_sha256",
    "bf16_runtime_values_sha256",
)


def _master_core_v65a(receipt: object) -> dict:
    if not isinstance(receipt, dict):
        raise RuntimeError("v65a master receipt is not a mapping")
    result = {key: receipt.get(key) for key in _MASTER_CORE_KEYS_V65A}
    identity = result["canonical_master_identity_sha256"]
    if (
        result["canonical_fp32_master_sha256"] != design52.MASTER_SHA256_V52
        or result["bf16_runtime_values_sha256"]
        != design52.MASTER_RUNTIME_SHA256_V52
        or not isinstance(identity, str) or len(identity) != 64
    ):
        raise RuntimeError("v65a exact V434 master core changed")
    return result


def _validate_active_lora_receipts_value_v65a(receipts: object) -> list[dict]:
    if not isinstance(receipts, list) or len(receipts) != 4:
        raise RuntimeError("v65a active LoRA receipt coverage changed")
    for receipt in receipts:
        applied = receipt.get("staged_v434_applied_receipt", {}) \
            if isinstance(receipt, dict) else {}
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != "v65a-effective-active-lora-receipt"
            or receipt.get("expected_lora_int_id") != 1
            or receipt.get("active_lora_ids") != [1]
            or receipt.get("active_manager_cache_lora_ids") != [1]
            or receipt.get("loaded_cpu_cache_lora_ids") != [1]
            or receipt.get("active_slot_index") != 0
            or receipt.get("max_loras") != 1
            or receipt.get("max_cpu_loras") != 2
            or receipt.get("max_lora_rank") != 32
            or receipt.get("extra_or_candidate_adapter_loaded") is not False
            or applied.get("staged_weights_file_sha256")
            != design52.STAGED_WEIGHTS_SHA256_V52
            or applied.get("canonical_fp32_state_sha256")
            != design52.MASTER_SHA256_V52
            or applied.get("canonical_ordered_key_sha256")
            != design52.MASTER_ORDERED_KEY_SHA256_V52
            or applied.get("canonical_tensor_count") != 70
            or applied.get("canonical_elements") != 4_528_128
            or applied.get("registered_lora_module_count") != 23
            or applied.get("matched_live_lora_module_count") != 23
            or applied.get("unmatched_registered_lora_module_count") != 0
            or applied.get("runtime_module_manifest_sha256")
            != analysis65a.RUNTIME_MODULE_MANIFEST_SHA256_V65A
            or applied.get("source_linked_runtime_view_count") != 82
            or applied.get("source_linked_runtime_elements") != 4_921_344
            or applied.get("source_linked_runtime_dtype") != "torch.bfloat16"
            or applied.get("source_linked_runtime_values_sha256")
            != design52.MASTER_RUNTIME_SHA256_V52
            or applied.get("registered_slot_view_count") != 82
            or applied.get("registered_slot_records_sha256")
            != analysis65a.REGISTERED_SLOT_RECORDS_SHA256_V65A
            or applied.get("exact_staged_fp32_to_gpu_slot_equality") is not True
            or applied.get("exact_registered_postpack_to_gpu_slot_equality")
            is not True
        ):
            raise RuntimeError("v65a active staged-V434 receipt changed")
    return receipts


def _validate_installations_v65a(installations: object) -> list:
    if not isinstance(installations, list) or len(installations) != 4:
        raise RuntimeError("v65a installation coverage changed")
    for receipt in installations:
        identity = receipt.get("canonical_identity", {}) \
            if isinstance(receipt, dict) else {}
        assignments = receipt.get("assignments", []) \
            if isinstance(receipt, dict) else []
        materialization = receipt.get("materialization", {}) \
            if isinstance(receipt, dict) else {}
        base = receipt.get("base_identity", {}) \
            if isinstance(receipt, dict) else {}
        origin = receipt.get("base_origin_inventory", {}) \
            if isinstance(receipt, dict) else {}
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema") != "canonical-lora-adapter-installed-v41a"
            or receipt.get("installed") is not True
            or receipt.get("adapter_id") != 1
            or receipt.get("slot") != 0
            or receipt.get("source_weights_sha256")
            != design52.SOURCE_WEIGHTS_SHA256_V52
            or receipt.get("source_config_sha256")
            != design52.SOURCE_CONFIG_SHA256_V52
            or identity.get("sha256") != design52.MASTER_SHA256_V52
            or design65.canonical_sha256_v65(identity)
            != analysis65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
            or receipt.get("assignment_count") != 82
            or receipt.get("assignment_sha256")
            != analysis65a.ASSIGNMENT_SHA256_V65A
            or not isinstance(assignments, list) or len(assignments) != 82
            or design65.canonical_sha256_v65(assignments)
            != analysis65a.ASSIGNMENT_SHA256_V65A
            or materialization.get("storage_layout_sha256")
            != analysis65a.MATERIALIZATION_STORAGE_LAYOUT_SHA256_V65A
            or materialization.get("runtime_values_sha256")
            != design52.MASTER_RUNTIME_SHA256_V52
            or base.get("inventory_sha256")
            != analysis65a.BASE_INVENTORY_SHA256_V65A
            or origin.get("inventory_sha256")
            != analysis65a.BASE_INVENTORY_SHA256_V65A
            or design65.canonical_sha256_v65(origin.get("tensors"))
            != analysis65a.BASE_INVENTORY_SHA256_V65A
            or receipt.get("zero_zero_degeneracy_guard") != {
                "all_a_zero": False,
                "all_b_zero": False,
                "simultaneous_all_zero_forbidden": True,
            }
        ):
            raise RuntimeError("v65a exact V434 installation receipt changed")
    return installations


def _valid_worker_timing_v65a(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"clock", "started_ns", "ended_ns", "elapsed_ns"}
        and value.get("clock") == "worker_monotonic_ns"
        and isinstance(value.get("started_ns"), int)
        and not isinstance(value.get("started_ns"), bool)
        and isinstance(value.get("ended_ns"), int)
        and not isinstance(value.get("ended_ns"), bool)
        and isinstance(value.get("elapsed_ns"), int)
        and not isinstance(value.get("elapsed_ns"), bool)
        and value["ended_ns"] >= value["started_ns"]
        and value["elapsed_ns"] == value["ended_ns"] - value["started_ns"]
    )


def _validate_period_receipts_v65a(
    slot_write_receipts: object, read_only_slot_receipts: object,
    installed_master: dict,
) -> None:
    installed_core = _master_core_v65a(installed_master)
    expected_periods = [
        (kind, index)
        for kind in ("unscored_warmup", "scored")
        for index in range(4)
    ]
    if not isinstance(slot_write_receipts, list) or len(slot_write_receipts) != 8:
        raise RuntimeError("v65a slot-write receipt coverage changed")
    for receipt, (kind, index) in zip(
        slot_write_receipts, expected_periods, strict=True,
    ):
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {
                "period_kind", "period_index", "pre_write_master",
                "post_write_master", "actors", "actor_receipts_sha256",
            }
            or receipt.get("period_kind") != kind
            or receipt.get("period_index") != index
            or _master_core_v65a(receipt.get("pre_write_master"))
            != installed_core
            or _master_core_v65a(receipt.get("post_write_master"))
            != installed_core
        ):
            raise RuntimeError("v65a ordered slot-write receipt changed")
        actors = receipt["actors"]
        if (
            not isinstance(actors, list) or len(actors) != 4
            or receipt["actor_receipts_sha256"]
            != design65.canonical_sha256_v65(actors)
        ):
            raise RuntimeError("v65a slot-write actor coverage changed")
        for actor in actors:
            if (
                not isinstance(actor, dict)
                or set(actor) != {
                    "schema", "period_kind", "period_index", "master_identity",
                    "materialization", "base_identity",
                    "transaction_state_quiescent", "timing",
                }
                or actor.get("schema") != "exact-master-slot-write-v65a"
                or actor.get("period_kind") != kind
                or actor.get("period_index") != index
                or actor.get("master_identity", {}).get("sha256")
                != design52.MASTER_SHA256_V52
                or design65.canonical_sha256_v65(actor.get("master_identity"))
                != analysis65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
                or actor.get("materialization", {}).get(
                    "storage_layout_sha256"
                ) != analysis65a.MATERIALIZATION_STORAGE_LAYOUT_SHA256_V65A
                or actor.get("materialization", {}).get("runtime_values_sha256")
                != design52.MASTER_RUNTIME_SHA256_V52
                or actor.get("base_identity", {}).get("inventory_sha256")
                != analysis65a.BASE_INVENTORY_SHA256_V65A
                or actor.get("transaction_state_quiescent") is not True
                or not _valid_worker_timing_v65a(actor.get("timing"))
            ):
                raise RuntimeError("v65a slot-write actor schema changed")

    expected_edges = [
        (kind, index, edge)
        for kind in ("unscored_warmup", "scored")
        for index in range(4)
        for edge in ("before_generation", "after_generation")
    ]
    if (
        not isinstance(read_only_slot_receipts, list)
        or len(read_only_slot_receipts) != 16
    ):
        raise RuntimeError("v65a read-only receipt coverage changed")
    for receipt, (kind, index, edge) in zip(
        read_only_slot_receipts, expected_edges, strict=True,
    ):
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {
                "period_kind", "period_index", "edge", "aggregate", "actors",
                "actor_receipts_sha256",
            }
            or receipt.get("period_kind") != kind
            or receipt.get("period_index") != index
            or receipt.get("edge") != edge
        ):
            raise RuntimeError("v65a ordered read-only receipt changed")
        aggregate = receipt["aggregate"]
        if (
            not isinstance(aggregate, dict)
            or set(aggregate) != {
                "schema", "canonical_fp32_master_sha256",
                "canonical_master_identity_sha256",
                "bf16_runtime_values_sha256", "runtime_view_count_per_actor",
                "runtime_elements_per_actor", "runtime_dtype",
                "base_inventory_sha256", "four_actor_exact_read_only_consensus",
            }
            or aggregate.get("schema")
            != "v65a-read-only-four-actor-master-slot-consensus"
            or _master_core_v65a(aggregate) != installed_core
            or aggregate.get("runtime_view_count_per_actor") != 82
            or aggregate.get("runtime_elements_per_actor") != 4_921_344
            or aggregate.get("runtime_dtype") != "torch.bfloat16"
            or aggregate.get("base_inventory_sha256")
            != analysis65a.BASE_INVENTORY_SHA256_V65A
            or aggregate.get("four_actor_exact_read_only_consensus") is not True
        ):
            raise RuntimeError("v65a read-only aggregate schema changed")
        actors = receipt["actors"]
        if (
            not isinstance(actors, list) or len(actors) != 4
            or receipt["actor_receipts_sha256"]
            != design65.canonical_sha256_v65(actors)
        ):
            raise RuntimeError("v65a read-only actor coverage changed")
        base_inventories = []
        master_identities = []
        for actor in actors:
            if (
                not isinstance(actor, dict)
                or set(actor) != {
                    "schema", "period_kind", "period_index", "edge",
                    "master_identity", "runtime_view_count", "runtime_elements",
                    "runtime_dtype", "runtime_values_sha256", "active_lora_ids",
                    "active_manager_cache_lora_ids", "base_identity",
                    "transaction_state_quiescent",
                    "slot_read_only_no_weight_write_or_reset", "timing",
                }
                or actor.get("schema") != "read-only-exact-master-slot-v65a"
                or actor.get("period_kind") != kind
                or actor.get("period_index") != index
                or actor.get("edge") != edge
                or actor.get("master_identity", {}).get("sha256")
                != design52.MASTER_SHA256_V52
                or design65.canonical_sha256_v65(actor.get("master_identity"))
                != analysis65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
                or actor.get("runtime_view_count") != 82
                or actor.get("runtime_elements") != 4_921_344
                or actor.get("runtime_dtype") != "torch.bfloat16"
                or actor.get("runtime_values_sha256")
                != design52.MASTER_RUNTIME_SHA256_V52
                or actor.get("active_lora_ids") != [1]
                or actor.get("active_manager_cache_lora_ids") != [1]
                or actor.get("base_identity", {}).get("inventory_sha256")
                != analysis65a.BASE_INVENTORY_SHA256_V65A
                or actor.get("transaction_state_quiescent") is not True
                or actor.get("slot_read_only_no_weight_write_or_reset") is not True
                or not _valid_worker_timing_v65a(actor.get("timing"))
            ):
                raise RuntimeError("v65a read-only actor schema changed")
            master_identities.append(actor["master_identity"])
            base_inventories.append(actor["base_identity"].get("inventory_sha256"))
        if (
            len({design65.canonical_sha256_v65(value)
                 for value in master_identities}) != 1
            or design65.canonical_sha256_v65(master_identities[0])
            != installed_core["canonical_master_identity_sha256"]
            or len(set(base_inventories)) != 1
            or base_inventories[0] != aggregate["base_inventory_sha256"]
        ):
            raise RuntimeError("v65a read-only actor consensus changed")


def build_evidence_v65a(
    *, panel: dict, input_receipt: dict, actor_identities: list[dict],
    worker_identities: list[dict], active_lora_receipts: list[dict],
    installations: list,
    installed_master: dict, warmup_state_receipts: list[dict],
    scored_state_receipts: list[dict], slot_write_receipts: list[dict],
    read_only_slot_receipts: list[dict], final_master_state: dict,
    scored_periods: list,
) -> dict:
    numeric = analysis65a.validate_scored_periods_v65a(scored_periods)
    if numeric.shape != (64, 4, 4, 3):
        raise RuntimeError("v65a evidence coverage changed")
    _validate_active_lora_receipts_value_v65a(active_lora_receipts)
    _validate_installations_v65a(installations)
    installed_core = _master_core_v65a(installed_master)
    if _master_core_v65a(final_master_state) != installed_core:
        raise RuntimeError("v65a final exact V434 master state changed")
    if (
        not isinstance(warmup_state_receipts, list)
        or len(warmup_state_receipts) != 4
        or not isinstance(warmup_state_receipts[0], dict)
        or _master_core_v65a(warmup_state_receipts[0].get("before"))
        != installed_core
    ):
        raise RuntimeError("v65a warmup master receipt changed")
    analysis65a.validate_state_receipts_v65a(
        warmup_state_receipts,
        scored_state_receipts,
        warmup_state_receipts[0]["before"],
    )
    _validate_period_receipts_v65a(
        slot_write_receipts, read_only_slot_receipts, installed_master,
    )
    value = {
        "schema": "v65a-ranking64-alpha-zero-generation-evidence",
        "status": "complete_fixed_warmup_and_scored_exact64_characterization",
        "panel_content_sha256": panel["content_sha256_before_self_field"],
        "authorized_input_receipt": input_receipt,
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
        "row_count": 64,
        "actor_count": 4,
        "unscored_warmup_period_count": 4,
        "scored_period_count": 4,
        "paired_replicas_per_unit": 8,
        "label_plan": dict(analysis65a.LABEL_PLAN_V65A),
        "pair_periods": [list(pair) for pair in analysis65a.PAIR_PERIODS_V65A],
        "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65A),
        "actor_runtime_identities": actor_identities,
        "worker_runtime_identities": worker_identities,
        "active_lora_receipts": active_lora_receipts,
        "active_lora_receipts_sha256": design65.canonical_sha256_v65(
            active_lora_receipts
        ),
        "initial_installations": installations,
        "installed_master_state": installed_master,
        "final_restored_master_state": final_master_state,
        "warmup_state_receipts": warmup_state_receipts,
        "warmup_state_receipts_sha256": design65.canonical_sha256_v65(
            warmup_state_receipts
        ),
        "scored_state_receipts": scored_state_receipts,
        "scored_state_receipts_sha256": design65.canonical_sha256_v65(
            scored_state_receipts
        ),
        "exact_master_slot_write_receipts": slot_write_receipts,
        "exact_master_slot_write_receipts_sha256": (
            design65.canonical_sha256_v65(slot_write_receipts)
        ),
        "read_only_live_slot_receipts": read_only_slot_receipts,
        "read_only_live_slot_receipts_sha256": (
            design65.canonical_sha256_v65(read_only_slot_receipts)
        ),
        "scored_periods": scored_periods,
        "numeric_scored_periods_sha256": design65.canonical_sha256_v65(
            scored_periods
        ),
        "warmup_generation_completions_discarded": 1024,
        "scored_generation_completions": 1024,
        "total_generation_completions": 2048,
        "generation_only": True,
        "warmup_raw_outputs_persisted": False,
        "warmup_generation_metrics_computed_or_persisted": False,
        "adaptive_retry_drop_reorder_or_early_stop_performed": False,
        "alpha": 0.0,
        "sigma_or_direction": None,
        "adapter_update_candidate_hpo_or_projection_performed": False,
        "holdback_sentinel_reserve_ood_terminal_or_protected_opened": False,
        "raw_question_answer_prompt_or_generation_text_persisted": False,
    }
    return _self_hashed_v65a(value)


def _authorized_prefix_postrun_receipt_v65a() -> dict:
    descriptor = os.open(design65.V61C_ROWS, os.O_RDONLY | os.O_CLOEXEC)
    try:
        stat = os.fstat(descriptor)
        payload = os.pread(descriptor, runtime65.RANKING_PREFIX_BYTES_V65, 0)
    finally:
        os.close(descriptor)
    observed = hashlib.sha256(payload).hexdigest()
    if (
        len(payload) != runtime65.RANKING_PREFIX_BYTES_V65
        or observed != runtime65.RANKING_PREFIX_SHA256_V65
        or stat.st_size != analysis65a.RANKING_SOURCE_FILE_SIZE_BYTES_V65A
    ):
        raise RuntimeError("v65a authorized ranking prefix changed postrun")
    return {
        "path": str(design65.V61C_ROWS),
        "file_size_bytes_metadata_only": stat.st_size,
        "authorized_prefix_bytes": len(payload),
        "authorized_prefix_sha256": observed,
        "decoded_postrun": False,
        "requested_byte_offset_at_or_after_prefix": False,
        "full_file_read_or_hash_performed": False,
    }


def gpu_period_phase_summary_v65a(
    path: Path, expected_pids: dict[int, int],
) -> dict:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    phases = [
        f"unscored_warmup_{index}_generation_all_actors" for index in range(4)
    ] + [
        f"scored_period_{index}_generation_all_actors" for index in range(4)
    ]
    by_phase = {}
    foreign = 0
    for phase in phases:
        by_gpu = {}
        for gpu in range(4):
            selected = [
                row for row in rows if row["phase"] == phase and row["gpu"] == gpu
            ]
            resident = [
                row for row in selected
                if expected_pids[gpu] in row["compute_pids"]
            ]
            foreign += sum(len(row["foreign_compute_pids"]) for row in selected)
            if (
                not resident
                or not any(row["utilization_percent"] > 0 for row in resident)
            ):
                raise RuntimeError(
                    f"v65a GPU {gpu} lacked positive activity in {phase}"
                )
            by_gpu[str(gpu)] = {
                "samples": len(selected),
                "resident_samples": len(resident),
                "positive_resident_samples": sum(
                    row["utilization_percent"] > 0 for row in resident
                ),
                "peak_utilization_percent": max(
                    row["utilization_percent"] for row in resident
                ),
                "peak_memory_used_mib": max(
                    row["memory_used_mib"] for row in resident
                ),
            }
        by_phase[phase] = by_gpu
    if foreign:
        raise RuntimeError("v65a observed a foreign GPU compute process")
    return {
        "schema": "v65a-per-period-four-gpu-activity",
        "generation_phases": 8,
        "all_eight_generation_phases_positive_on_all_four_gpus": True,
        "foreign_compute_process_observations": 0,
        "by_phase": by_phase,
    }


def _validate_postrun_integrity_v65a(
    gpu: object, cleanup: object, idle: object, pid_map: dict,
) -> None:
    if (
        not isinstance(gpu, dict)
        or gpu.get("all_four_attributed_positive") is not True
        or any(
            gpu.get("by_gpu", {}).get(str(index), {}).get("expected_pid")
            != pid_map[index]
            or gpu["by_gpu"][str(index)].get("positive_samples", 0) <= 0
            for index in range(4)
        )
        or not isinstance(cleanup, dict)
        or cleanup.get("schema") != "eggroll-es-placement-group-cleanup-v38a"
        or cleanup.get("engine_kill_count") != 4
        or cleanup.get("placement_group_remove_count") != 4
        or cleanup.get("all_four_gcs_states_removed") is not True
        or idle != {"all_four_compute_process_lists_empty": True}
    ):
        raise RuntimeError("v65a utilization, cleanup, or final-idle gate changed")


def _slot_write_v65a(
    trainer, v40a, period_kind: str, period_index: int,
) -> tuple[dict, dict]:
    pre_write = runtime61c._assert_v434_certificates_v61c(
        v40a._rpc_all(trainer, "adapter_state_certificate_v52")
    )
    actor_receipts = v40a._rpc_all(
        trainer, "rematerialize_exact_master_v65a", (
            period_kind, period_index,
            design52.MASTER_SHA256_V52,
            design52.MASTER_RUNTIME_SHA256_V52,
        ),
    )
    if (
        len(actor_receipts) != 4
        or any(
            receipt.get("schema") != "exact-master-slot-write-v65a"
            or receipt.get("period_kind") != period_kind
            or receipt.get("period_index") != period_index
            or receipt.get("master_identity", {}).get("sha256")
            != design52.MASTER_SHA256_V52
            or design65.canonical_sha256_v65(receipt.get("master_identity"))
            != analysis65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
            or receipt.get("materialization", {}).get(
                "storage_layout_sha256"
            ) != analysis65a.MATERIALIZATION_STORAGE_LAYOUT_SHA256_V65A
            or receipt.get("materialization", {}).get("runtime_values_sha256")
            != design52.MASTER_RUNTIME_SHA256_V52
            or receipt.get("base_identity", {}).get("inventory_sha256")
            != analysis65a.BASE_INVENTORY_SHA256_V65A
            or receipt.get("transaction_state_quiescent") is not True
            for receipt in actor_receipts
        )
    ):
        raise RuntimeError("v65a exact-master slot-write receipt changed")
    post_write = runtime61c._assert_v434_certificates_v61c(
        v40a._rpc_all(trainer, "adapter_state_certificate_v52")
    )
    for key in (
        "canonical_fp32_master_sha256",
        "canonical_master_identity_sha256",
        "bf16_runtime_values_sha256",
    ):
        if pre_write.get(key) != post_write.get(key):
            raise RuntimeError("v65a exact master identity changed during slot write")
    return post_write, {
        "period_kind": period_kind,
        "period_index": period_index,
        "pre_write_master": pre_write,
        "post_write_master": post_write,
        "actors": actor_receipts,
        "actor_receipts_sha256": design65.canonical_sha256_v65(actor_receipts),
    }


def _read_only_slot_v65a(
    trainer, v40a, period_kind: str, period_index: int, edge: str,
) -> tuple[dict, dict]:
    actors = v40a._rpc_all(
        trainer, "read_only_exact_master_slot_v65a", (
            period_kind, period_index, edge,
        ),
    )
    if (
        len(actors) != 4
        or any(
            receipt.get("schema") != "read-only-exact-master-slot-v65a"
            or receipt.get("period_kind") != period_kind
            or receipt.get("period_index") != period_index
            or receipt.get("edge") != edge
            or receipt.get("master_identity", {}).get("sha256")
            != design52.MASTER_SHA256_V52
            or design65.canonical_sha256_v65(receipt.get("master_identity"))
            != analysis65a.MASTER_IDENTITY_OBJECT_SHA256_V65A
            or receipt.get("runtime_view_count") != 82
            or receipt.get("runtime_elements") != 4_921_344
            or receipt.get("runtime_dtype") != "torch.bfloat16"
            or receipt.get("runtime_values_sha256")
            != design52.MASTER_RUNTIME_SHA256_V52
            or receipt.get("active_lora_ids") != [1]
            or receipt.get("active_manager_cache_lora_ids") != [1]
            or receipt.get("base_identity", {}).get("inventory_sha256")
            != analysis65a.BASE_INVENTORY_SHA256_V65A
            or receipt.get("transaction_state_quiescent") is not True
            or receipt.get("slot_read_only_no_weight_write_or_reset") is not True
            for receipt in actors
        )
    ):
        raise RuntimeError("v65a read-only live slot receipt changed")
    master_identities = [receipt["master_identity"] for receipt in actors]
    base_inventories = [
        receipt["base_identity"]["inventory_sha256"] for receipt in actors
    ]
    if (
        len({design65.canonical_sha256_v65(value)
             for value in master_identities}) != 1
        or len(set(base_inventories)) != 1
        or base_inventories[0] != analysis65a.BASE_INVENTORY_SHA256_V65A
    ):
        raise RuntimeError("v65a read-only actor consensus changed")
    aggregate = {
        "schema": "v65a-read-only-four-actor-master-slot-consensus",
        "canonical_fp32_master_sha256": design52.MASTER_SHA256_V52,
        "canonical_master_identity_sha256": design65.canonical_sha256_v65(
            master_identities[0]
        ),
        "bf16_runtime_values_sha256": design52.MASTER_RUNTIME_SHA256_V52,
        "runtime_view_count_per_actor": 82,
        "runtime_elements_per_actor": 4_921_344,
        "runtime_dtype": "torch.bfloat16",
        "base_inventory_sha256": base_inventories[0],
        "four_actor_exact_read_only_consensus": True,
    }
    return aggregate, {
        "period_kind": period_kind,
        "period_index": period_index,
        "edge": edge,
        "aggregate": aggregate,
        "actors": actors,
        "actor_receipts_sha256": design65.canonical_sha256_v65(actors),
    }


def _active_lora_receipts_v65a(trainer, v40a) -> list[dict]:
    receipts = v40a._rpc_all(trainer, "runtime_active_lora_v65a", (
        1,
        str(design52.STAGED_WEIGHTS_V52),
        design52.STAGED_WEIGHTS_SHA256_V52,
        design52.MASTER_SHA256_V52,
        design52.MASTER_RUNTIME_SHA256_V52,
    ))
    return _validate_active_lora_receipts_value_v65a(receipts)


def _failure_cleanup_v65a(
    v40a, trainer, ray_module=None, *, strict_already_complete: bool = False,
) -> tuple[object, object, dict, list]:
    """Attempt teardown steps independently and require an exact idle proof."""
    cleanup = idle = None
    errors = []
    strict_complete = strict_already_complete is True
    if trainer is not None and not strict_complete:
        try:
            cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
            strict_complete = True
        except BaseException as error:
            errors.append(error)
    partial_pool_cleanup = {
        "attempted_after_incomplete_strict_cleanup": not strict_complete,
        "pool_found": False,
        "terminate_succeeded": False,
        "join_succeeded": False,
    }
    if not strict_complete:
        pool = getattr(trainer, "mp_pool", None) if trainer is not None else None
        partial_pool_cleanup["pool_found"] = pool is not None
        if pool is not None:
            try:
                pool.terminate()
                partial_pool_cleanup["terminate_succeeded"] = True
            except BaseException as error:
                errors.append(error)
            try:
                pool.join()
                partial_pool_cleanup["join_succeeded"] = True
            except BaseException as error:
                errors.append(error)
    try:
        if ray_module is None:
            import ray as ray_module
        ray_module.shutdown()
    except BaseException as error:
        errors.append(error)
    try:
        idle = v40a.cleanup_v38a.wait_for_gpu_idle()
    except BaseException as error:
        errors.append(error)
    if idle != {"all_four_compute_process_lists_empty": True}:
        errors.append(RuntimeError("v65a final four-GPU idle proof failed"))
    return cleanup, idle, partial_pool_cleanup, errors


def _sanitized_errors_v65a(errors: list) -> list[dict]:
    return [{
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(
            str(error).encode("utf-8")
        ).hexdigest(),
    } for error in errors]


def execute_v65a(preregistration: dict, args) -> int:
    import run_lora_es_generation_boundary_v48b as v48b

    prior = v48b.v43i
    v40a = prior.v40a
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v65a requires fresh artifact paths")
    expectation = runtime64.base_model_artifact_expectation_v64()
    pre_model_receipt = runtime64.verify_base_model_artifacts_v64(expectation)
    preflight = v40a.gpu_preflight()
    attempt = _self_hashed_v65a({
        "schema": "v65a-ranking64-alpha-zero-attempt",
        "status": "launching_exact64_calibration_only",
        "phase": "before_authorized_train_semantics_model_ray_or_gpu_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65A),
        "base_model_artifact_receipt": pre_model_receipt,
        "preflight": preflight,
        "fixed_unscored_warmup_periods": 4,
        "fixed_scored_periods": 4,
        "fixed_paired_replicas_per_unit": 8,
        "model_loaded_or_gpu_compute_started": False,
        "candidate_hpo_update_projection_or_protected_access": False,
    })
    runtime61a.atomic_json_v61a(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    cleanup = idle = None
    construction_state = {}
    try:
        adapter_contract = runtime52.verify_adapter_contract_v52()
        panel, _panel61c = runtime65._load_hash_only_panel_v65(preregistration)
        v40a.base.set_seed(runtime65.GENERATION_SEED_V65)
        trainer, saved = make_trainer_v65a(
            preregistration, prior, construction_state,
        )
        actor_identities = trainer._resolve([
            engine.runtime_identity_v65a.remote() for engine in trainer.engines
        ])
        pid_map = validate_actor_identities_v65a(
            actor_identities,
            preregistration["runtime"]["tuned_table_content_sha256"],
        )
        worker_identities = v40a._rpc_all(
            trainer, "runtime_identity_v40a",
        )
        if v40a.validate_identities(actor_identities, worker_identities) != pid_map:
            raise RuntimeError("v65a actor/worker identity mapping changed")
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures),
            daemon=True,
        )
        monitor.start()
        requests, params, answers, input_receipt = (
            runtime65.prepare_ranking_requests_v65(trainer, prior, panel)
        )
        request = lora_request_v65a(prior)
        if (
            getattr(request, "lora_name", None)
            != "v434_ranking64_alpha_zero_v65a"
            or getattr(request, "lora_int_id", None) != 1
            or Path(getattr(request, "lora_path", "")).resolve()
            != design52.STAGED_V52
        ):
            raise RuntimeError("v65a LoRA request identity changed")
        input_receipt.update({
            "submitted_request_batch_size": 64,
            "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65A),
            "lora_adapter_request_name": "v434_ranking64_alpha_zero_v65a",
            "lora_adapter_request_id": 1,
            "lora_adapter_request_path": str(design52.STAGED_V52),
        })
        phase.value = "activate_v434_lora_slot_all_actors"
        if v40a._rpc_all(trainer, "add_lora", (request,)) != [True] * 4:
            raise RuntimeError("v65a four-actor LoRA activation failed")
        phase.value = "install_exact_v434_master_all_actors"
        installations = v40a._rpc_all(
            trainer, "install_adapter_state_v41a", (
                str(design52.SOURCE_WEIGHTS_V52),
                str(design52.SOURCE_CONFIG_V52),
                design52.SOURCE_WEIGHTS_SHA256_V52,
                design52.SOURCE_CONFIG_SHA256_V52,
            ),
        )
        _validate_installations_v65a(installations)
        installed = runtime61c._assert_v434_certificates_v61c(
            v40a._rpc_all(trainer, "adapter_state_certificate_v52")
        )
        active_lora_receipts = _active_lora_receipts_v65a(trainer, v40a)

        warmup_state_receipts = []
        scored_state_receipts = []
        slot_write_receipts = []
        read_only_slot_receipts = []
        for period_index in range(4):
            phase.value = f"unscored_warmup_{period_index}_exact_master_slot_write"
            _write_certificate, write_receipt = _slot_write_v65a(
                trainer, v40a, "unscored_warmup", period_index,
            )
            slot_write_receipts.append(write_receipt)
            before, read_receipt = _read_only_slot_v65a(
                trainer, v40a, "unscored_warmup", period_index,
                "before_generation",
            )
            read_only_slot_receipts.append(read_receipt)
            phase.value = f"unscored_warmup_{period_index}_generation_all_actors"
            warmup_batches = _validate_batches_v65a(
                trainer._resolve([
                    engine.generate.remote(
                        requests, params, use_tqdm=False, lora_request=request,
                    ) for engine in trainer.engines
                ]),
                "unscored_warmup",
            )
            phase.value = f"unscored_warmup_{period_index}_post_generation_integrity"
            after, read_receipt = _read_only_slot_v65a(
                trainer, v40a, "unscored_warmup", period_index,
                "after_generation",
            )
            read_only_slot_receipts.append(read_receipt)
            warmup_state_receipts.append(_state_receipt_v65a(
                "unscored_warmup", period_index, before, after, installed,
            ))
            del warmup_batches

        scored_periods = []
        for period_index in range(4):
            phase.value = f"scored_period_{period_index}_exact_master_slot_write"
            _write_certificate, write_receipt = _slot_write_v65a(
                trainer, v40a, "scored", period_index,
            )
            slot_write_receipts.append(write_receipt)
            before, read_receipt = _read_only_slot_v65a(
                trainer, v40a, "scored", period_index,
                "before_generation",
            )
            read_only_slot_receipts.append(read_receipt)
            phase.value = f"scored_period_{period_index}_generation_all_actors"
            batches = _validate_batches_v65a(
                trainer._resolve([
                    engine.generate.remote(
                        requests, params, use_tqdm=False, lora_request=request,
                    ) for engine in trainer.engines
                ]),
                "scored",
            )
            phase.value = f"scored_period_{period_index}_post_generation_integrity"
            after, read_receipt = _read_only_slot_v65a(
                trainer, v40a, "scored", period_index,
                "after_generation",
            )
            read_only_slot_receipts.append(read_receipt)
            phase.value = f"scored_period_{period_index}_numeric_reduction"
            scored_periods.append(_score_period_v65a(
                panel["items"], answers, batches,
            ))
            del batches
            scored_state_receipts.append(_state_receipt_v65a(
                "scored", period_index, before, after, installed,
            ))
            phase.value = f"scored_period_{period_index}_complete"

        phase.value = "final_exact_master_restoration_certificate"
        final_master_state = runtime61c._assert_v434_certificates_v61c(
            v40a._rpc_all(trainer, "adapter_state_certificate_v52")
        )

        evidence = build_evidence_v65a(
            panel=panel,
            input_receipt=input_receipt,
            actor_identities=actor_identities,
            worker_identities=worker_identities,
            active_lora_receipts=active_lora_receipts,
            installations=installations,
            installed_master=installed,
            warmup_state_receipts=warmup_state_receipts,
            scored_state_receipts=scored_state_receipts,
            slot_write_receipts=slot_write_receipts,
            read_only_slot_receipts=read_only_slot_receipts,
            final_master_state=final_master_state,
            scored_periods=scored_periods,
        )
        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v65a GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        gpu_phases = gpu_period_phase_summary_v65a(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        _validate_postrun_integrity_v65a(gpu, cleanup, idle, pid_map)

        post_prefix_receipt = _authorized_prefix_postrun_receipt_v65a()
        post_model_receipt = runtime64.verify_base_model_artifacts_v64(expectation)
        if post_model_receipt != pre_model_receipt:
            raise RuntimeError("v65a base-model bytes changed during calibration")
        post_adapter_contract = runtime52.verify_adapter_contract_v52()
        if post_adapter_contract != adapter_contract:
            raise RuntimeError("v65a adapter source/stage bytes changed during calibration")
        evidence = _self_hashed_v65a({
            **evidence,
            "adapter_artifact_contract_prelaunch": adapter_contract,
            "adapter_artifact_contract_postcleanup": post_adapter_contract,
            "adapter_artifact_contract_unchanged": True,
        })
        null_analysis = analysis65a.analyze_scored_periods_v65a(scored_periods)
        null_analysis = _self_hashed_v65a({
            **null_analysis,
            "source_evidence_content_sha256": evidence[
                "content_sha256_before_self_field"
            ],
        })
        runtime61a.atomic_json_v61a(EVIDENCE, evidence)
        runtime61a.atomic_json_v61a(ANALYSIS, null_analysis)
        gate = null_analysis.get("required_alpha_zero_gate", {})
        report = _self_hashed_v65a({
            "schema": "v65a-ranking64-alpha-zero-calibration-report",
            "status": (
                "complete_gate_passed_population_still_unauthorized"
                if gate.get("passed") is True
                else "complete_gate_failed_closed"
            ),
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "attempt": {
                "path": str(ATTEMPT),
                "file_sha256": design65.file_sha256_v65(ATTEMPT),
                "content_sha256": attempt["content_sha256_before_self_field"],
            },
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": design65.file_sha256_v65(EVIDENCE),
                "content_sha256": evidence["content_sha256_before_self_field"],
            },
            "analysis": {
                "path": str(ANALYSIS),
                "file_sha256": design65.file_sha256_v65(ANALYSIS),
                "content_sha256": null_analysis[
                    "content_sha256_before_self_field"
                ],
                "required_alpha_zero_gate": gate,
            },
            "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65A),
            "actor_runtime_identities": actor_identities,
            "base_model_prelaunch_artifact_receipt": pre_model_receipt,
            "base_model_postrun_artifact_receipt": post_model_receipt,
            "adapter_artifact_contract_prelaunch": adapter_contract,
            "adapter_artifact_contract_postcleanup": post_adapter_contract,
            "adapter_artifact_contract_unchanged": True,
            "authorized_prefix_postrun_receipt": post_prefix_receipt,
            "gpu_activity": gpu,
            "gpu_period_phases": gpu_phases,
            "gpu_log_file_sha256": design65.file_sha256_v65(GPU_LOG),
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "warmup_generation_completions_discarded": 1024,
            "scored_generation_completions": 1024,
            "total_generation_completions": 2048,
            "raw_question_answer_prompt_or_generation_text_persisted": False,
            "candidate_hpo_update_projection_or_promotion_performed": False,
            "holdback_sentinel_reserve_ood_terminal_or_protected_opened": False,
            "v65_population_launch_authorized": False,
        })
        runtime61a.atomic_json_v61a(REPORT, report)
        print(json.dumps({
            "report_file_sha256": design65.file_sha256_v65(REPORT),
            "report_content_sha256": report["content_sha256_before_self_field"],
            "evidence_file_sha256": design65.file_sha256_v65(EVIDENCE),
            "analysis_file_sha256": design65.file_sha256_v65(ANALYSIS),
            "required_alpha_zero_gate_passed": gate.get("passed") is True,
            "all_eight_generation_phases_positive_on_all_four_gpus": True,
            "v65_population_launch_authorized": False,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        if trainer is None:
            trainer = construction_state.get("trainer")
        if saved is None:
            saved = construction_state.get("saved")
        observed_cleanup, observed_idle, partial_pool_cleanup, cleanup_errors = (
            _failure_cleanup_v65a(
                v40a, trainer, strict_already_complete=cleanup is not None,
            )
        )
        if observed_cleanup is not None:
            cleanup = observed_cleanup
        if observed_idle is not None:
            idle = observed_idle
        trainer = None
        construction_state["trainer"] = None
        if not FAILURE.exists():
            runtime61a.atomic_json_v61a(FAILURE, _self_hashed_v65a({
                "schema": "v65a-ranking64-alpha-zero-calibration-failure",
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                "type": type(error).__name__,
                "message_sha256": hashlib.sha256(
                    str(error).encode("utf-8")
                ).hexdigest(),
                "cleanup": cleanup,
                "partial_constructor_pool_cleanup": partial_pool_cleanup,
                "final_gpu_idle": idle,
                "cleanup_errors": _sanitized_errors_v65a(cleanup_errors),
                "ray_shutdown_attempted_even_without_complete_trainer": True,
                "four_gpu_idle_proof_attempted": True,
                "raw_error_message_or_traceback_persisted": False,
                "runtime_determinism_controls": dict(RUNTIME_CONTROLS_V65A),
                "adaptive_retry_drop_reorder_or_early_stop_performed": False,
                "candidate_hpo_update_projection_or_promotion_performed": False,
                "holdback_sentinel_reserve_ood_terminal_or_protected_opened": False,
                "raw_question_answer_prompt_or_generation_text_persisted": False,
            }))
        if cleanup_errors:
            raise RuntimeError(
                "v65a failed and strict four-GPU cleanup also failed"
            ) from cleanup_errors[0]
        raise
    finally:
        if trainer is not None:
            try:
                v40a.base.close_trainer(trainer)
            except Exception:
                pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    preregistration = load_preregistration_v65a(args)
    if args.dry_run:
        if args.execute:
            raise RuntimeError("v65a dry-run and execute are mutually exclusive")
        print(json.dumps({
            "schema": preregistration["schema"],
            "content_sha256": preregistration[
                "content_sha256_before_self_field"
            ],
            "authorized_semantic_rows_opened": 0,
            "model_ray_or_gpu_loaded": False,
            "filesystem_writes": False,
            "unscored_warmup_periods": 4,
            "scored_periods": 4,
            "paired_replicas_per_unit": 8,
            "candidate_hpo_update_projection_or_protected_access": False,
        }, sort_keys=True))
        return 0
    if not args.execute:
        raise RuntimeError("v65a live path requires --execute")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise RuntimeError("v65a requires VLLM_BATCH_INVARIANT absent or false")
    runtime52.require_live_interpreter_v52()
    return execute_v65a(preregistration, args)


if __name__ == "__main__":
    raise SystemExit(main())
