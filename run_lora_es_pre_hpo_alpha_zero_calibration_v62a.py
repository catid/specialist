#!/usr/bin/env python3
"""Four-GPU generation-only V62A pre-HPO alpha-zero calibration.

The live path evaluates one unchanged V434 LoRA state with reference and
candidate labels acting only as counterbalanced aliases.  It implements no
perturbation, adapter update, candidate state, HPO population, promotion,
holdback, OOD, shadow, terminal, or protected-data access.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import audit_vllm_pre_hpo_alpha_zero_support_v62a as support_audit
import lora_es_nested_population_v52 as design_v52
import lora_es_pre_hpo_alpha_zero_calibration_v62a as analysis
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_nested_population_v52 as runtime_v52
import run_lora_es_paired_null_calibration_v61c as runtime_v61c


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v62a_v434_pre_hpo_alpha_zero_generation_calibration"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
EVIDENCE = (RUN_DIR / "alpha_zero_evidence_v62a.json").resolve()
ANALYSIS = (RUN_DIR / "alpha_zero_analysis_v62a.json").resolve()
REPORT = (RUN_DIR / "alpha_zero_report_v62a.json").resolve()
FAILURE = (RUN_DIR / "failure_v62a.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v62a.jsonl").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v434_pre_hpo_alpha_zero_generation_calibration_v62a.json"
).resolve()
SUPPORT_AUDIT = support_audit.OUTPUT
SUPPORT_AUDIT_FILE_SHA256 = (
    "cc979d6828a337fe7c62fb12140ac4668e6b6e3b038a4a1346156a810f0da0b7"
)
SUPPORT_AUDIT_CONTENT_SHA256 = (
    "7d6a823fdba6fc0631974ebc7312e214f5f8a8e8a5d3b43d45db5206dcb9faa2"
)
V62_NUMERIC_AUDIT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "generated_f1_robust_gate_numeric_audit_v62.json"
).resolve()
V62_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_generated_f1_robust_gate_v62.json"
).resolve()
WORKER_EXTENSION_V62A = runtime_v52.WORKER_EXTENSION_V52


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--execute", action="store_true")
    return value


def _read_self_hashed_v62a(
    path: Path,
    identities: dict,
    schema: str,
) -> dict:
    if runtime_v61a.file_sha256_v61a(path) != identities["file_sha256"]:
        raise RuntimeError(f"v62a bound file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != identities["content_sha256"]
        or analysis.canonical_sha256_v62a(compact)
        != identities["content_sha256"]
        or value.get("schema") != schema
        or value.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError(f"v62a bound content changed: {path}")
    return value


def _read_v62_methodology_v62a() -> dict:
    numeric = _read_self_hashed_v62a(
        V62_NUMERIC_AUDIT,
        analysis.V62_NUMERIC_AUDIT_IDENTITIES,
        "v62-generated-f1-robust-gate-numeric-audit",
    )
    prereg = _read_self_hashed_v62a(
        V62_PREREGISTRATION,
        analysis.V62_PREREGISTRATION_IDENTITIES,
        "v62-generated-f1-robust-evaluator-hpo-gate-preregistration",
    )
    noise = numeric.get("calibrated_noise_and_signal", {})
    frozen = prereg.get("fresh_pre_hpo_alpha_zero_calibration_gate", {})
    if (
        numeric.get("status")
        != "complete_numeric_only_method_design_hpo_unauthorized"
        or prereg.get("status")
        != "methodology_preregistered_runtime_and_hpo_launch_unauthorized"
        or prereg.get("access_and_authorization", {}).get(
            "hpo_population_launch_authorized"
        ) is not False
        or noise.get("future_pre_hpo_max_ci_halfwidth")
        != analysis.MAX_PRIMARY_CI_HALFWIDTH_V62A
        or noise.get("future_pre_hpo_max_actor_leave_one_out_shift")
        != analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
        or frozen.get("maximum_primary_ci_halfwidth")
        != analysis.MAX_PRIMARY_CI_HALFWIDTH_V62A
        or frozen.get("maximum_actor_leave_one_out_shift")
        != analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
        or frozen.get("generated_f1_primary_interval_must_contain_zero")
        is not True
        or frozen.get("like_for_like_actor_statistic_required") is not True
        or frozen.get("v53_actor_spread_used_as_leave_one_out_threshold")
        is not False
    ):
        raise RuntimeError("v62a V62 methodology gate changed")
    return {"numeric_audit": numeric, "preregistration": prereg}


def _read_support_audit_v62a() -> dict:
    if support_audit.file_sha256_v62a(SUPPORT_AUDIT) != (
        SUPPORT_AUDIT_FILE_SHA256
    ):
        raise RuntimeError("v62a installed-runtime support audit file changed")
    value = json.loads(SUPPORT_AUDIT.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field")
        != SUPPORT_AUDIT_CONTENT_SHA256
        or support_audit.canonical_sha256_v62a(compact)
        != SUPPORT_AUDIT_CONTENT_SHA256
        or value.get("schema")
        != "v62a-installed-vllm-pre-hpo-alpha-zero-support-audit"
        or value.get("status") != "supported"
        or value.get("pre_hpo_alpha_zero_runtime_supported") is not True
        or value.get("requested_runtime_controls")
        != analysis.RUNTIME_CONTROLS_V62A
        or value.get("support_audit_authorizes_gpu_launch") is not False
        or value.get("model_train_semantics_or_gpu_accessed") is not False
    ):
        raise RuntimeError("v62a installed-runtime support changed")
    observed = {
        key: support_audit.file_sha256_v62a(path)
        for key, path in support_audit.v61e.SOURCE_PATHS.items()
    }
    if observed != value.get("installed_vllm_source_file_sha256"):
        raise RuntimeError("v62a installed vLLM source identities changed")
    return value


def implementation_bindings_v62a() -> dict:
    paths = {
        "runtime_v62a": Path(__file__).resolve(),
        "preregistration_builder_v62a": (
            ROOT / "build_lora_es_pre_hpo_alpha_zero_preregistration_v62a.py"
        ),
        "analysis_v62a": Path(analysis.__file__).resolve(),
        "tests_v62a": (
            ROOT / "test_lora_es_pre_hpo_alpha_zero_calibration_v62a.py"
        ),
        "support_audit_v62a": Path(support_audit.__file__).resolve(),
        "method_numeric_audit_v62": (
            ROOT / "analyze_lora_es_generated_f1_gate_v62.py"
        ),
        "method_preregistration_builder_v62": (
            ROOT / "build_lora_es_generated_f1_gate_preregistration_v62.py"
        ),
        "method_tests_v62": ROOT / "test_lora_es_generated_f1_gate_v62.py",
        "runtime_v61c": Path(runtime_v61c.__file__).resolve(),
        "input_builder_v61c": Path(runtime_v61c.inputs.__file__).resolve(),
        "runtime_v61a": Path(runtime_v61a.__file__).resolve(),
        "design_v52": Path(design_v52.__file__).resolve(),
        "runtime_v52": Path(runtime_v52.__file__).resolve(),
        "anchor_v4": ROOT / "train_eggroll_es_specialist_anchor_v4.py",
        "worker_v52": ROOT / "eggroll_es_worker_lora_v52.py",
        "worker_v51": ROOT / "eggroll_es_worker_lora_v51.py",
        "worker_v41a": ROOT / "eggroll_es_worker_lora_v41a.py",
    }
    return {
        "code_file_sha256": {
            key: runtime_v61a.file_sha256_v61a(path)
            for key, path in paths.items()
        },
        "support_audit": {
            "file_sha256": SUPPORT_AUDIT_FILE_SHA256,
            "content_sha256": SUPPORT_AUDIT_CONTENT_SHA256,
        },
        "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
        "v62_numeric_audit": dict(analysis.V62_NUMERIC_AUDIT_IDENTITIES),
        "v62_preregistration": dict(analysis.V62_PREREGISTRATION_IDENTITIES),
        "pinned_v434_identities_without_reopening": (
            runtime_v61a.implementation_bindings_v61a()[
                "pinned_v52_artifact_identities_without_reopening"
            ]
        ),
        "staged_dataset_file_sha256": (
            runtime_v61c.STAGED_DATASET_FILE_SHA256
        ),
        "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": (
            runtime_v61c.STAGED_PANEL_CONTENT_SHA256
        ),
        "train_semantics_model_gpu_or_protected_paths_opened": False,
    }


def _artifacts_v62a() -> dict:
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "evidence": str(EVIDENCE),
        "analysis": str(ANALYSIS),
        "report": str(REPORT),
        "failure": str(FAILURE),
        "gpu_log": str(GPU_LOG),
    }


def load_preregistration_v62a(args) -> dict:
    path = Path(args.preregistration).resolve()
    if runtime_v61a.file_sha256_v61a(path) != args.preregistration_sha256:
        raise RuntimeError("v62a preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    recipe = value.get("fixed_calibration_recipe", {})
    gates = value.get("required_alpha_zero_gates", {})
    access = value.get("access_contract", {})
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or analysis.canonical_sha256_v62a(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "v62a-v434-pre-hpo-alpha-zero-generation-preregistration"
        or value.get("status")
        != "preregistered_before_train_semantics_model_or_gpu_access"
        or value.get("specific_alpha_zero_calibration_gpu_launch_authorized")
        is not True
        or value.get("hpo_population_update_or_candidate_authorized") is not False
        or value.get("ood_shadow_holdout_or_protected_access_authorized")
        is not False
        or value.get("v62_methodology_commit") != analysis.V62_METHOD_COMMIT
        or value.get("v62_numeric_audit_identities")
        != analysis.V62_NUMERIC_AUDIT_IDENTITIES
        or value.get("v62_preregistration_identities")
        != analysis.V62_PREREGISTRATION_IDENTITIES
        or access.get("only_live_semantic_paths_may_open")
        != [str(runtime_v61c.STAGED_DATASET), str(runtime_v61c.STAGED_PANEL)]
        or access.get("full_train_membership_holdback_ood_shadow_terminal_or_"
                      "protected_may_open") is not False
        or recipe.get("base_model")
        != "/home/catid/specialist/models/Qwen3.6-35B-A3B"
        or recipe.get("adapter_state") != "V434"
        or recipe.get("staged_dataset") != str(runtime_v61c.STAGED_DATASET)
        or recipe.get("staged_dataset_file_sha256")
        != runtime_v61c.STAGED_DATASET_FILE_SHA256
        or recipe.get("staged_panel") != str(runtime_v61c.STAGED_PANEL)
        or recipe.get("staged_panel_file_sha256")
        != runtime_v61c.STAGED_PANEL_FILE_SHA256
        or recipe.get("staged_panel_content_sha256")
        != runtime_v61c.STAGED_PANEL_CONTENT_SHA256
        or recipe.get("rows") != 68
        or recipe.get("ranking_units") != 64
        or recipe.get("exact_sentinel_units") != 4
        or recipe.get("same_call_ranking_plus_sentinel_rows") is not True
        or recipe.get("holdback_units") != 0
        or recipe.get("holdback_documents") != 0
        or recipe.get("physical_gpu_ids") != [0, 1, 2, 3]
        or recipe.get("actors") != 4
        or recipe.get("tensor_parallel_size_per_actor") != 1
        or recipe.get("sequential_periods") != 4
        or recipe.get("counterbalanced_pairs_per_actor") != 2
        or recipe.get("replicas_per_conflict_unit") != 8
        or recipe.get("generation_only") is not True
        or recipe.get("request_type_per_period") != "generation"
        or recipe.get("teacher_forced_requests") != 0
        or recipe.get("teacher_forced_metric_computed") is not False
        or recipe.get("label_plan") != analysis.LABEL_PLAN_V62A
        or recipe.get("pair_periods")
        != [list(pair) for pair in analysis.PAIR_PERIODS_V62A]
        or recipe.get("runtime_determinism_controls")
        != analysis.RUNTIME_CONTROLS_V62A
        or recipe.get("generation_params_without_seed")
        != analysis.GENERATION_PARAMS_WITHOUT_SEED_V62A
        or recipe.get("common_generation_seed")
        != analysis.COMMON_GENERATION_SEED_V62A
        or recipe.get("canonical_fp32_master_sha256")
        != design_v52.MASTER_SHA256_V52
        or recipe.get("bf16_runtime_values_sha256")
        != design_v52.MASTER_RUNTIME_SHA256_V52
        or recipe.get("worker_extension") != WORKER_EXTENSION_V62A
        or recipe.get("submitted_requests_per_actor_call") != 68
        or recipe.get("active_sequence_limit_per_actor") != 68
        or recipe.get("v27c_tuned_table_runtime_identity_retained") is not True
        or recipe.get("global_batch_invariance_claimed") is not False
        or recipe.get("generation_completions") != 1088
        or recipe.get("alpha") != 0.0
        or recipe.get("sigma_or_direction") is not None
        or recipe.get("adapter_update_candidate_or_hpo_performed") is not False
        or gates.get("generated_f1_primary_interval_must_contain_zero")
        is not True
        or gates.get("maximum_primary_ci_halfwidth_inclusive")
        != analysis.MAX_PRIMARY_CI_HALFWIDTH_V62A
        or gates.get("maximum_actor_leave_one_out_shift_inclusive")
        != analysis.MAX_ACTOR_LEAVE_ONE_OUT_SHIFT_V62A
        or value.get("implementation_bindings")
        != implementation_bindings_v62a()
        or value.get("artifacts") != _artifacts_v62a()
        or value.get("runtime") != design_v52.RUNTIME_V52
        or value.get("raw_question_answer_or_generation_text_may_be_persisted")
        is not False
    ):
        raise RuntimeError("v62a preregistration contract changed")
    _read_support_audit_v62a()
    _read_v62_methodology_v62a()
    return value


def engine_kwargs_v62a(v40a, precision="bfloat16") -> dict:
    return {
        "model": str(v40a.MODEL),
        "tensor_parallel_size": 1,
        "worker_extension_cls": WORKER_EXTENSION_V62A,
        "dtype": precision,
        "enable_prefix_caching": False,
        "enforce_eager": True,
        "async_scheduling": False,
        "max_num_seqs": 68,
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
        "max_cpu_loras": 1,
    }


def _make_trainer_v62a(prereg: dict, prior):
    v40a = prior.v40a
    parent = v40a.base.load_trainer()
    expected_tuned_content = prereg["runtime"]["tuned_table_content_sha256"]

    class PreHPOAlphaZeroTrainerV62A(parent):
        def launch_engines(
            self,
            num_engines=4,
            n_gpu_per_vllm_engine=1,
            model_name="unused",
            precision="bfloat16",
        ):
            if int(num_engines) != 4 or int(n_gpu_per_vllm_engine) != 1:
                raise RuntimeError("v62a requires four TP1 engines")
            import ray
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy,
            )

            class PreHPOAlphaZeroLLMV62A(ESNcclLLM):
                def runtime_identity_v62a(self):
                    import torch
                    import vllm.envs as vllm_envs
                    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe

                    raw = ray.get_gpu_ids()
                    if len(raw) != 1:
                        raise RuntimeError("v62a actor does not own exactly one GPU")
                    physical = v40a.normalize_gpu_id(raw[0])
                    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
                    folder = os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
                    batch_invariant_env = os.environ.get("VLLM_BATCH_INVARIANT")
                    config = self.llm_engine.vllm_config
                    scheduler = config.scheduler_config
                    fused_moe.get_moe_configs.cache_clear()
                    tuned = fused_moe.get_moe_configs(256, 512, None)
                    scheduler_class = scheduler.get_scheduler_cls().__name__
                    if (
                        visible != str(physical)
                        or torch.cuda.device_count() != 1
                        or torch.cuda.current_device() != 0
                        or folder != str(v40a.TUNED_FOLDER)
                        or vllm_envs.VLLM_TUNED_CONFIG_FOLDER
                        != str(v40a.TUNED_FOLDER)
                        or batch_invariant_env != "0"
                        or vllm_envs.VLLM_BATCH_INVARIANT is not False
                        or scheduler.async_scheduling is not False
                        or scheduler.max_num_seqs != 68
                        or scheduler.policy != "fcfs"
                        or scheduler_class != "Scheduler"
                        or config.model_config.enforce_eager is not True
                        or v40a.canonical_sha256(tuned) != expected_tuned_content
                    ):
                        raise RuntimeError("v62a actor runtime identity changed")
                    return {
                        "schema": "pre-hpo-alpha-zero-actor-identity-v62a",
                        "pid": os.getpid(),
                        "physical_gpu_id": physical,
                        "cuda_visible_devices": visible,
                        "cuda_current_device": 0,
                        "VLLM_BATCH_INVARIANT": False,
                        "async_scheduling": False,
                        "max_num_seqs": 68,
                        "scheduling_policy": "fcfs",
                        "scheduler_class": scheduler_class,
                        "enforce_eager": True,
                        "tuned_folder": folder,
                        "tuned_table_content_sha256": (
                            v40a.canonical_sha256(tuned)
                        ),
                        "submitted_request_batch_size": 68,
                        "generation_only": True,
                        "global_batch_invariance_claimed": False,
                    }

            pgs = [
                placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK")
                for _ in range(4)
            ]
            ray.get([group.ready() for group in pgs])
            strategies = [PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            ) for group in pgs]
            kwargs = engine_kwargs_v62a(v40a, precision)
            engines = [ray.remote(
                num_cpus=0,
                num_gpus=1,
                scheduling_strategy=strategy,
            )(PreHPOAlphaZeroLLMV62A).options(runtime_env={"env_vars": {
                "VLLM_TUNED_CONFIG_FOLDER": str(v40a.TUNED_FOLDER),
                "VLLM_BATCH_INVARIANT": "0",
            }}).remote(**kwargs) for strategy in strategies]
            return engines, pgs

    saved = (v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION)
    v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = (
        EXPERIMENT,
        RUN_DIR,
        WORKER_EXTENSION_V62A,
    )
    try:
        trainer = PreHPOAlphaZeroTrainerV62A(
            model_name=str(v40a.MODEL),
            checkpoint=None,
            sigma=0.0,
            alpha=0.0,
            population_size=4,
            reward_shaping="z-scores",
            num_iterations=0,
            max_tokens=4,
            batch_size=1,
            mini_batch_size=1,
            reward_function=v40a.base.specialist_reward,
            template_function=lambda value: value,
            train_dataloader=[],
            eval_dataloader_dict={},
            eval_freq=1,
            n_vllm_engines=4,
            n_gpu_per_vllm_engine=1,
            logging="none",
            global_seed=20_260_716,
            use_gpus="0,1,2,3",
            experiment_name=EXPERIMENT,
            wandb_project="none",
            save_best_models=False,
            reward_function_timeout=10,
            output_directory=str(RUN_DIR.parent),
        )
        import ray
        trainer._resolve = lambda handles: ray.get(handles)
    except BaseException:
        v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved
        raise
    return trainer, saved


def _generation_params_v62a():
    from vllm import SamplingParams
    return SamplingParams(
        seed=analysis.COMMON_GENERATION_SEED_V62A,
        **analysis.GENERATION_PARAMS_WITHOUT_SEED_V62A,
    )


def _lora_request_v62a(prior):
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "v434_pre_hpo_alpha_zero_evaluator_v62a",
        1,
        str(design_v52.STAGED_V52),
        base_model_name=str(prior.v40a.MODEL),
    )


def build_evidence_v62a(rows, periods, state_receipts: list[dict]) -> dict:
    if (
        len(rows) != 68
        or len(periods) != 4
        or len(state_receipts) != 4
        or any(len(period) != 4 for period in periods)
        or any(len(batch) != 68 for period in periods for batch in period)
    ):
        raise RuntimeError("v62a generation evidence coverage changed")
    evidence_rows = []
    for request_index, row in enumerate(rows):
        evidence_rows.append({
            "request_index": request_index,
            "row_sha256": row["row_sha256"],
            "unit_identity_sha256": row["unit_identity_sha256"],
            "role": row["role"],
            "periods": [{
                "period_index": period_index,
                "request_type": "generation",
                "actors": [{
                    "actor_rank": actor_rank,
                    "label": analysis.LABEL_PLAN_V62A[str(actor_rank)][
                        period_index
                    ],
                    "generation": periods[period_index][actor_rank][
                        request_index
                    ],
                } for actor_rank in range(4)],
            } for period_index in range(4)],
        })
    value = {
        "schema": "v62a-pre-hpo-alpha-zero-generation-only-evidence",
        "status": "complete_alpha_zero_generation_only_characterization",
        "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
        "v62_numeric_audit_identities": dict(
            analysis.V62_NUMERIC_AUDIT_IDENTITIES
        ),
        "v62_preregistration_identities": dict(
            analysis.V62_PREREGISTRATION_IDENTITIES
        ),
        "staged_dataset_file_sha256": (
            runtime_v61c.STAGED_DATASET_FILE_SHA256
        ),
        "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": (
            runtime_v61c.STAGED_PANEL_CONTENT_SHA256
        ),
        "canonical_fp32_master_sha256": design_v52.MASTER_SHA256_V52,
        "bf16_runtime_values_sha256": design_v52.MASTER_RUNTIME_SHA256_V52,
        "row_count": 68,
        "ranking_units": 64,
        "exact_sentinel_units": 4,
        "actor_count": 4,
        "period_count": 4,
        "pairs_per_actor": 2,
        "replicas_per_unit": 8,
        "label_plan": dict(analysis.LABEL_PLAN_V62A),
        "pair_periods": [
            list(pair) for pair in analysis.PAIR_PERIODS_V62A
        ],
        "common_generation_seed": analysis.COMMON_GENERATION_SEED_V62A,
        "generation_params_without_seed": dict(
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V62A
        ),
        "runtime_determinism_controls": dict(
            analysis.RUNTIME_CONTROLS_V62A
        ),
        "state_receipts": state_receipts,
        "numeric_state_receipts_sha256": analysis.canonical_sha256_v62a(
            state_receipts
        ),
        "rows": evidence_rows,
        "numeric_actor_period_manifest_sha256": (
            analysis.canonical_sha256_v62a(evidence_rows)
        ),
        "generation_only": True,
        "generation_completions": 1088,
        "teacher_forced_requests": 0,
        "alpha": 0.0,
        "adapter_update_candidate_or_hpo_performed": False,
        "holdback_ood_shadow_or_protected_opened": False,
        "raw_question_answer_or_generation_text_persisted": False,
    }
    value["content_sha256_before_self_field"] = (
        analysis.canonical_sha256_v62a(value)
    )
    analysis.validate_evidence_v62a(value)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v62a(args)
    if args.dry_run:
        if args.execute:
            raise RuntimeError("v62a dry-run and execute are mutually exclusive")
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "support_audit_file_sha256": SUPPORT_AUDIT_FILE_SHA256,
            "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
            "v62_numeric_audit_file_sha256": (
                analysis.V62_NUMERIC_AUDIT_IDENTITIES["file_sha256"]
            ),
            "v62_preregistration_file_sha256": (
                analysis.V62_PREREGISTRATION_IDENTITIES["file_sha256"]
            ),
            "staged_train_rows_opened": 0,
            "generation_only": True,
            "teacher_forced_requests": 0,
            "model_or_gpu_loaded": False,
            "filesystem_writes": False,
            "hpo_update_candidate_or_protected_access": False,
        }, sort_keys=True))
        return 0
    if not args.execute:
        raise RuntimeError("v62a live path requires --execute")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise RuntimeError("v62a requires batch-invariant mode absent or false")
    runtime_v52.require_live_interpreter_v52()
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v62a requires fresh artifact paths")

    import run_lora_es_generation_boundary_v48b as v48b
    prior = v48b.v43i
    v40a = prior.v40a
    preflight = v40a.gpu_preflight()
    attempt = runtime_v61a.self_hashed_v61a({
        "schema": "v62a-pre-hpo-alpha-zero-generation-attempt",
        "status": "launching_specific_calibration_only",
        "phase": (
            "after_gpu_inventory_preflight_before_staged_train_semantics_"
            "model_load_or_gpu_compute"
        ),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "support_audit_file_sha256": SUPPORT_AUDIT_FILE_SHA256,
        "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
        "runtime_determinism_controls": dict(
            analysis.RUNTIME_CONTROLS_V62A
        ),
        "preflight": preflight,
        "gpu_inventory_preflight_performed": True,
        "model_loaded_or_gpu_compute_started": False,
        "hpo_update_candidate_or_protected_access": False,
    })
    runtime_v61a.atomic_json_v61a(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    try:
        adapter_ids = runtime_v61c._verify_adapter_artifacts_v61c()
        rows, panel = runtime_v61c.load_staged_inputs_v61c()
        prompts = [v40a.base.specialist_template(row["question"]) for row in rows]
        v40a.base.set_seed(prior.GLOBAL_SEED)
        trainer, saved = _make_trainer_v62a(prereg, prior)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v62a.remote() for engine in trainer.engines
        ])
        if (
            len(actor_ids) != 4
            or sorted(item.get("physical_gpu_id") for item in actor_ids)
            != [0, 1, 2, 3]
            or any({
                key: item.get(key)
                for key in analysis.RUNTIME_CONTROLS_V62A
            } != analysis.RUNTIME_CONTROLS_V62A for item in actor_ids)
            or any(item.get("generation_only") is not True for item in actor_ids)
        ):
            raise RuntimeError("v62a four-actor runtime controls changed")
        pid_map = prior.prior._actor_pid_map(actor_ids)
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures),
            daemon=True,
        )
        monitor.start()
        request = _lora_request_v62a(prior)
        phase.value = "activate_v434_lora_slot"
        if v40a._rpc_all(trainer, "add_lora", (request,)) != [True] * 4:
            raise RuntimeError("v62a four-actor LoRA activation failed")
        phase.value = "install_exact_v434_master"
        installations = v40a._rpc_all(trainer, "install_adapter_state_v41a", (
            str(design_v52.SOURCE_WEIGHTS_V52),
            str(design_v52.SOURCE_CONFIG_V52),
            design_v52.SOURCE_WEIGHTS_SHA256_V52,
            design_v52.SOURCE_CONFIG_SHA256_V52,
        ))
        installed = runtime_v61c._assert_v434_certificates_v61c(
            v40a._rpc_all(trainer, "adapter_state_certificate_v52")
        )
        periods = []
        state_receipts = []
        for period_index in range(4):
            before = runtime_v61c._assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            phase.value = f"period_{period_index}_generation_all_actors"
            batches = trainer._resolve([
                engine.generate.remote(
                    prompts,
                    _generation_params_v62a(),
                    use_tqdm=False,
                    lora_request=request,
                ) for engine in trainer.engines
            ])
            if len(batches) != 4 or any(len(batch) != 68 for batch in batches):
                raise RuntimeError("v62a same-call 64+4 generation coverage changed")
            periods.append([
                runtime_v61c.score_generation_batch_v61c(
                    rows,
                    batch,
                    prior.fused,
                ) for batch in batches
            ])
            after = runtime_v61c._assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            if before != after or before != installed:
                raise RuntimeError("v62a adapter state changed across a period")
            state_receipts.append({
                "period_index": period_index,
                "before": before,
                "after": after,
                "identical_v434_state": True,
            })
        evidence = build_evidence_v62a(rows, periods, state_receipts)
        null_analysis = analysis.build_analysis_v62a(evidence)
        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v62a GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        runtime_v61a.atomic_json_v61a(EVIDENCE, evidence)
        runtime_v61a.atomic_json_v61a(ANALYSIS, null_analysis)
        gate = null_analysis["required_pre_hpo_gate"]
        report = runtime_v61a.self_hashed_v61a({
            "schema": "v62a-pre-hpo-alpha-zero-generation-report",
            "status": (
                "complete_gate_passed_hpo_still_unauthorized"
                if gate["passed"]
                else "complete_gate_failed_closed"
            ),
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": (
                args.preregistration_content_sha256
            ),
            "support_audit_file_sha256": SUPPORT_AUDIT_FILE_SHA256,
            "support_audit_content_sha256": SUPPORT_AUDIT_CONTENT_SHA256,
            "v62_methodology_commit": analysis.V62_METHOD_COMMIT,
            "adapter_artifact_identities": adapter_ids,
            "panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
            "panel_content_sha256": runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
            "panel_document_block_audit": panel["document_block_audit"],
            "master_state_receipt": installed,
            "installations": installations,
            "state_receipts_sha256": evidence[
                "numeric_state_receipts_sha256"
            ],
            "actor_identities": actor_ids,
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
                "content_sha256": evidence[
                    "content_sha256_before_self_field"
                ],
                "rows": 68,
                "actors": 4,
                "periods": 4,
                "generation_completions": 1088,
                "teacher_forced_requests": 0,
            },
            "analysis": {
                "path": str(ANALYSIS),
                "file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
                "content_sha256": null_analysis[
                    "content_sha256_before_self_field"
                ],
                "required_pre_hpo_gate": gate,
                "exact_sentinel_diagnostics": null_analysis[
                    "exact_sentinel_diagnostics"
                ],
            },
            "gpu_activity": gpu,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "gpu_log_file_sha256": runtime_v61a.file_sha256_v61a(GPU_LOG),
            "alpha": 0.0,
            "generation_only": True,
            "adapter_update_candidate_or_hpo_performed": False,
            "holdback_ood_shadow_or_protected_opened": False,
            "raw_question_answer_or_generation_text_persisted": False,
            "hpo_population_launch_authorized": False,
        })
        runtime_v61a.atomic_json_v61a(REPORT, report)
        print(json.dumps({
            "report_file_sha256": runtime_v61a.file_sha256_v61a(REPORT),
            "report_content_sha256": report[
                "content_sha256_before_self_field"
            ],
            "evidence_file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
            "evidence_content_sha256": evidence[
                "content_sha256_before_self_field"
            ],
            "analysis_file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
            "analysis_content_sha256": null_analysis[
                "content_sha256_before_self_field"
            ],
            "required_pre_hpo_gate_passed": gate["passed"],
            "generation_only": True,
            "teacher_forced_requests": 0,
            "all_four_gpus_attributed_positive": True,
            "hpo_population_launch_authorized": False,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        if not FAILURE.exists():
            failure = runtime_v61a._sanitize_failure_v61a(error)
            failure.update({
                "schema": "v62a-pre-hpo-alpha-zero-generation-failure",
                "runtime_determinism_controls": dict(
                    analysis.RUNTIME_CONTROLS_V62A
                ),
                "adapter_update_candidate_or_hpo_performed": False,
                "holdback_ood_shadow_or_protected_opened": False,
            })
            runtime_v61a.atomic_json_v61a(
                FAILURE,
                runtime_v61a.self_hashed_v61a(failure),
            )
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
            prior.v40a.EXPERIMENT, prior.v40a.RUN_DIR, prior.v40a.WORKER_EXTENSION = (
                saved
            )


if __name__ == "__main__":
    raise SystemExit(main())
