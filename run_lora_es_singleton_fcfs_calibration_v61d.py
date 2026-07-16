#!/usr/bin/env python3
"""Matched four-GPU V61D singleton-FCFS null calibration."""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import audit_vllm_singleton_fcfs_support_v61d as support_audit
import lora_es_nested_population_v52 as design_v52
import lora_es_singleton_fcfs_calibration_v61d as analysis
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_nested_population_v52 as runtime_v52
import run_lora_es_paired_null_calibration_v61c as runtime_v61c


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v61d_v434_singleton_fcfs_paired_evaluator_calibration"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
EVIDENCE = (RUN_DIR / "singleton_fcfs_null_evidence_v61d.json").resolve()
ANALYSIS = (RUN_DIR / "singleton_fcfs_null_analysis_v61d.json").resolve()
REPORT = (RUN_DIR / "singleton_fcfs_null_report_v61d.json").resolve()
FAILURE = (RUN_DIR / "failure_v61d.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v61d.jsonl").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v434_singleton_fcfs_paired_evaluator_calibration_v61d.json"
).resolve()
AUDIT = support_audit.OUTPUT
AUDIT_FILE_SHA256 = (
    "7a35795961fe681ad3b438d1a57618512348ec1bffc613a757b4146f2ab69724"
)
AUDIT_CONTENT_SHA256 = (
    "9eacfe65a8c7ebadddc29b295c21371cae7232ff954c7bb024d24c06be65734c"
)
WORKER_EXTENSION_V61D = runtime_v52.WORKER_EXTENSION_V52
V61C_FINALIZER_FILE_SHA256 = (
    "d3d5eabf1e5d9b0bed2dfd2a355ed5eb839a22cb4bcdea58af0ab84231042d46"
)
V61C_FINALIZER_CONTENT_SHA256 = (
    "7bc9735dea87ae8bf2374bcefb7c290b7bb273f2394b44d54dc1fa69e8e851c0"
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--execute", action="store_true")
    return value


def _read_support_audit_v61d() -> dict:
    if support_audit.file_sha256_v61d(AUDIT) != AUDIT_FILE_SHA256:
        raise RuntimeError("v61d installed-vLLM support audit file changed")
    value = json.loads(AUDIT.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != AUDIT_CONTENT_SHA256
        or support_audit.canonical_sha256_v61d(compact) != AUDIT_CONTENT_SHA256
        or value.get("status") != "supported"
        or value.get("singleton_fcfs_controls_supported") is not True
        or value.get("gpu_launch_authorized_by_audit") is not True
        or value.get("requested_runtime_controls")
        != analysis.RUNTIME_CONTROLS_V61D
        or value.get("batch_invariant_environment_resolved_false") is not True
        or value.get("global_batch_invariance_claimed") is not False
    ):
        raise RuntimeError("v61d installed-vLLM singleton-FCFS support changed")
    observed_sources = {
        key: support_audit.file_sha256_v61d(path)
        for key, path in support_audit.SOURCE_PATHS.items()
    }
    if observed_sources != value.get("source_file_sha256"):
        raise RuntimeError("v61d installed-vLLM source identities changed")
    return value


def implementation_bindings_v61d() -> dict:
    paths = {
        "runtime_v61d": Path(__file__).resolve(),
        "preregistration_builder_v61d": (
            ROOT / "build_lora_es_singleton_fcfs_preregistration_v61d.py"
        ),
        "analysis_v61d": Path(analysis.__file__).resolve(),
        "tests_v61d": ROOT / "test_lora_es_singleton_fcfs_calibration_v61d.py",
        "support_audit_v61d": Path(support_audit.__file__).resolve(),
        "runtime_v61c": Path(runtime_v61c.__file__).resolve(),
        "analysis_v61c": Path(analysis.v61c.__file__).resolve(),
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
            "file_sha256": AUDIT_FILE_SHA256,
            "content_sha256": AUDIT_CONTENT_SHA256,
        },
        "pinned_v434_identities_without_reopening": (
            runtime_v61a.implementation_bindings_v61a()[
                "pinned_v52_artifact_identities_without_reopening"
            ]
        ),
        "staged_dataset_file_sha256": runtime_v61c.STAGED_DATASET_FILE_SHA256,
        "staged_panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
        "staged_panel_content_sha256": runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
        "train_semantics_model_gpu_or_protected_paths_opened_to_build_bindings": False,
    }


def _artifacts_v61d() -> dict:
    return {
        "attempt": str(ATTEMPT), "run_directory": str(RUN_DIR),
        "evidence": str(EVIDENCE), "analysis": str(ANALYSIS),
        "report": str(REPORT), "failure": str(FAILURE),
        "gpu_log": str(GPU_LOG),
    }


def load_preregistration_v61d(args) -> dict:
    path = Path(args.preregistration).resolve()
    if runtime_v61a.file_sha256_v61a(path) != args.preregistration_sha256:
        raise RuntimeError("v61d preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    recipe = value.get("fixed_calibration_recipe", {})
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or analysis.canonical_sha256_v61d(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "v61d-v434-singleton-fcfs-paired-evaluator-preregistration"
        or value.get("status")
        != "preregistered_before_v61d_train_semantics_model_or_gpu_access"
        or value.get("gpu_launch_authorized") is not True
        or value.get("selection_hpo_update_or_promotion_authorized") is not False
        or value.get("eval_ood_shadow_or_holdout_access_authorized") is not False
        or value.get("access_contract", {}).get("only_live_semantic_paths_may_open")
        != [str(runtime_v61c.STAGED_DATASET), str(runtime_v61c.STAGED_PANEL)]
        or recipe.get("runtime_determinism_controls")
        != analysis.RUNTIME_CONTROLS_V61D
        or recipe.get("rows") != 68 or recipe.get("ranking_units") != 64
        or recipe.get("exact_sentinel_units") != 4
        or recipe.get("actors") != 4 or recipe.get("sequential_periods") != 4
        or recipe.get("label_plan") != analysis.LABEL_PLAN_V61D
        or recipe.get("request_type_order") != analysis.REQUEST_TYPE_ORDER_V61D
        or recipe.get("pair_periods")
        != [list(pair) for pair in analysis.PAIR_PERIODS_V61D]
        or recipe.get("generation_params_without_seed")
        != analysis.GENERATION_PARAMS_WITHOUT_SEED_V61D
        or recipe.get("teacher_forced_params_without_seed")
        != analysis.TEACHER_FORCED_PARAMS_WITHOUT_SEED_V61D
        or recipe.get("canonical_fp32_master_sha256")
        != design_v52.MASTER_SHA256_V52
        or recipe.get("bf16_runtime_values_sha256")
        != design_v52.MASTER_RUNTIME_SHA256_V52
        or recipe.get("worker_extension") != WORKER_EXTENSION_V61D
        or value.get("implementation_bindings") != implementation_bindings_v61d()
        or value.get("artifacts") != _artifacts_v61d()
        or value.get("runtime") != design_v52.RUNTIME_V52
        or value.get("raw_question_answer_or_generation_text_may_be_persisted")
        is not False
    ):
        raise RuntimeError("v61d preregistration contract changed")
    _read_support_audit_v61d()
    return value


def _make_trainer_v61d(prereg: dict, prior):
    v40a = prior.v40a
    parent = v40a.base.load_trainer()
    expected_tuned_content = prereg["runtime"]["tuned_table_content_sha256"]

    class SingletonFCFSTrainerV61D(parent):
        def launch_engines(
            self, num_engines=4, n_gpu_per_vllm_engine=1,
            model_name="unused", precision="bfloat16",
        ):
            if int(num_engines) != 4 or int(n_gpu_per_vllm_engine) != 1:
                raise RuntimeError("v61d requires four TP1 engines")
            import ray
            from es_at_scale.trainer.es_trainer import ESNcclLLM
            from ray.util.placement_group import placement_group
            from ray.util.scheduling_strategies import (
                PlacementGroupSchedulingStrategy,
            )

            class SingletonFCFSLLMV61D(ESNcclLLM):
                def runtime_identity_v61d(self):
                    import torch
                    import vllm.envs as vllm_envs
                    import vllm.model_executor.layers.fused_moe.fused_moe as fused_moe

                    raw = ray.get_gpu_ids()
                    if len(raw) != 1:
                        raise RuntimeError("v61d actor does not own exactly one GPU")
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
                        or scheduler.max_num_seqs != 1
                        or scheduler.policy != "fcfs"
                        or scheduler_class != "Scheduler"
                        or config.model_config.enforce_eager is not True
                        or v40a.canonical_sha256(tuned) != expected_tuned_content
                    ):
                        raise RuntimeError("v61d actor determinism identity changed")
                    return {
                        "schema": "singleton-fcfs-actor-identity-v61d",
                        "pid": os.getpid(), "physical_gpu_id": physical,
                        "cuda_visible_devices": visible, "cuda_current_device": 0,
                        "VLLM_BATCH_INVARIANT": False,
                        "async_scheduling": False, "max_num_seqs": 1,
                        "scheduling_policy": "fcfs",
                        "scheduler_class": scheduler_class,
                        "enforce_eager": True,
                        "tuned_folder": folder,
                        "tuned_table_content_sha256": v40a.canonical_sha256(tuned),
                        "singleton_active_sequence_limit": 1,
                        "global_batch_invariance_claimed": False,
                    }

            pgs = [
                placement_group([{"GPU": 1, "CPU": 0}], strategy="PACK")
                for _ in (0, 1, 2, 3)
            ]
            ray.get([group.ready() for group in pgs])
            strategies = [PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            ) for group in pgs]
            kwargs = engine_kwargs_v61d(v40a, precision)
            engines = [ray.remote(
                num_cpus=0, num_gpus=1, scheduling_strategy=strategy,
            )(SingletonFCFSLLMV61D).options(runtime_env={"env_vars": {
                "VLLM_TUNED_CONFIG_FOLDER": str(v40a.TUNED_FOLDER),
                "VLLM_BATCH_INVARIANT": "0",
            }}).remote(**kwargs) for strategy in strategies]
            return engines, pgs

    saved = (v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION)
    v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = (
        EXPERIMENT, RUN_DIR, WORKER_EXTENSION_V61D,
    )
    try:
        trainer = SingletonFCFSTrainerV61D(
            model_name=str(v40a.MODEL), checkpoint=None, sigma=0.0, alpha=0.0,
            population_size=4, reward_shaping="z-scores", num_iterations=0,
            max_tokens=4, batch_size=1, mini_batch_size=1,
            reward_function=v40a.base.specialist_reward,
            template_function=lambda value: value,
            train_dataloader=[], eval_dataloader_dict={}, eval_freq=1,
            n_vllm_engines=4, n_gpu_per_vllm_engine=1, logging="none",
            global_seed=20_260_715, use_gpus="0,1,2,3",
            experiment_name=EXPERIMENT, wandb_project="none",
            save_best_models=False, reward_function_timeout=10,
            output_directory=str(RUN_DIR.parent),
        )
        import ray
        trainer._resolve = lambda handles: ray.get(handles)
    except BaseException:
        v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved
        raise
    return trainer, saved


def engine_kwargs_v61d(v40a, precision="bfloat16") -> dict:
    return {
        "model": str(v40a.MODEL), "tensor_parallel_size": 1,
        "worker_extension_cls": WORKER_EXTENSION_V61D,
        "dtype": precision, "enable_prefix_caching": False,
        "enforce_eager": True, "async_scheduling": False,
        "max_num_seqs": 1, "scheduling_policy": "fcfs",
        "gpu_memory_utilization": 0.82, "max_model_len": 2048,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "mm_processor_cache_gb": 0, "skip_mm_profiling": True,
        "moe_backend": "triton", "enable_lora": True,
        "max_lora_rank": 32, "max_loras": 1, "max_cpu_loras": 1,
    }


def _lora_request_v61d(prior):
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "v434_singleton_fcfs_paired_evaluator_v61d", 1,
        str(design_v52.STAGED_V52), base_model_name=str(prior.v40a.MODEL),
    )


def build_evidence_v61d(rows, periods, state_receipts) -> dict:
    value = runtime_v61c.build_evidence_v61c(rows, periods, state_receipts)
    value.pop("content_sha256_before_self_field", None)
    value["schema"] = (
        "v61d-singleton-fcfs-identical-state-paired-evaluator-evidence"
    )
    value["status"] = "complete_matched_alpha_zero_no_update_characterization"
    value["runtime_determinism_controls"] = dict(
        analysis.RUNTIME_CONTROLS_V61D
    )
    value["matched_v61c_panel_labels_metrics_bootstrap_thresholds"] = True
    value["v61c_thresholds_relaxed_or_changed"] = False
    value["content_sha256_before_self_field"] = analysis.canonical_sha256_v61d(
        value
    )
    analysis.validate_evidence_v61d(value)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v61d(args)
    if args.dry_run:
        if args.execute:
            raise RuntimeError("v61d dry-run and execute are mutually exclusive")
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "support_audit_file_sha256": AUDIT_FILE_SHA256,
            "singleton_fcfs_controls_supported": True,
            "staged_train_rows_opened": 0,
            "model_or_gpu_loaded": False, "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if not args.execute:
        raise RuntimeError("v61d live path requires --execute")
    if os.environ.get("VLLM_BATCH_INVARIANT") not in (None, "0"):
        raise RuntimeError("v61d requires batch-invariant mode absent or false")
    runtime_v52.require_live_interpreter_v52()
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v61d requires fresh artifact paths")

    import run_lora_es_generation_boundary_v48b as v48b
    prior = v48b.v43i; v40a = prior.v40a
    preflight = v40a.gpu_preflight()
    attempt = runtime_v61a.self_hashed_v61a({
        "schema": "v61d-singleton-fcfs-paired-evaluator-attempt",
        "status": "launching_matched_alpha_zero_characterization_only",
        "phase": (
            "after_gpu_inventory_preflight_before_staged_train_semantics_"
            "model_load_or_gpu_compute"
        ),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "support_audit_file_sha256": AUDIT_FILE_SHA256,
        "runtime_determinism_controls": dict(analysis.RUNTIME_CONTROLS_V61D),
        "preflight": preflight, "gpu_inventory_preflight_performed": True,
        "model_loaded_or_gpu_compute_started": False,
        "holdback_or_protected_opened": False,
    })
    runtime_v61a.atomic_json_v61a(ATTEMPT, attempt); RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event(); failures: queue.Queue = queue.Queue()
    phase = v40a.Phase(); started = time.monotonic()
    try:
        adapter_ids = runtime_v61c._verify_adapter_artifacts_v61c()
        rows, panel = runtime_v61c.load_staged_inputs_v61c()
        prompts = [v40a.base.specialist_template(row["question"]) for row in rows]
        v40a.base.set_seed(prior.GLOBAL_SEED)
        trainer, saved = _make_trainer_v61d(prereg, prior)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v61d.remote() for engine in trainer.engines
        ])
        if any({
            key: item.get(key) for key in analysis.RUNTIME_CONTROLS_V61D
        } != analysis.RUNTIME_CONTROLS_V61D for item in actor_ids):
            raise RuntimeError("v61d four-actor runtime controls changed")
        pid_map = prior.prior._actor_pid_map(actor_ids)
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
        ); monitor.start()
        request = _lora_request_v61d(prior)
        phase.value = "activate_v434_lora_slot"
        if v40a._rpc_all(trainer, "add_lora", (request,)) != [True] * 4:
            raise RuntimeError("v61d four-actor LoRA activation failed")
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
        dense_items = prior.anchor_v4.prepare_gold_answer_items_v4(
            trainer.tokenizer, prompts, [row["answer"] for row in rows],
        )
        teacher_requests = [
            {"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items
        ]
        periods = []; state_receipts = []
        for period_index in range(4):
            before = runtime_v61c._assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            period = {"generation": None, "teacher_forced": None}
            for request_type in analysis.REQUEST_TYPE_ORDER_V61D[str(period_index)]:
                phase.value = f"period_{period_index}_{request_type}_all_actors"
                if request_type == "generation":
                    batches = trainer._resolve([
                        engine.generate.remote(
                            prompts, runtime_v61c._generation_params_v61c(),
                            use_tqdm=False, lora_request=request,
                        ) for engine in trainer.engines
                    ])
                    if len(batches) != 4 or any(len(batch) != 68 for batch in batches):
                        raise RuntimeError("v61d generation coverage changed")
                    period[request_type] = [
                        runtime_v61c.score_generation_batch_v61c(
                            rows, batch, prior.fused
                        ) for batch in batches
                    ]
                else:
                    batches = trainer._resolve([
                        engine.generate.remote(
                            teacher_requests, runtime_v61c._teacher_params_v61c(),
                            use_tqdm=False, lora_request=request,
                        ) for engine in trainer.engines
                    ])
                    if len(batches) != 4 or any(len(batch) != 68 for batch in batches):
                        raise RuntimeError("v61d teacher coverage changed")
                    period[request_type] = [
                        runtime_v61c.score_teacher_batch_v61c(
                            dense_items, batch, prior.anchor_v4
                        ) for batch in batches
                    ]
            after = runtime_v61c._assert_v434_certificates_v61c(
                v40a._rpc_all(trainer, "adapter_state_certificate_v52")
            )
            if before != after or before != installed:
                raise RuntimeError("v61d adapter state changed across a period")
            periods.append(period); state_receipts.append({
                "period_index": period_index, "before": before, "after": after,
                "identical_v434_state": True,
            })
        evidence = build_evidence_v61d(rows, periods, state_receipts)
        null_analysis = analysis.build_analysis_v61d(evidence)
        stop.set(); monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v61d GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown(); idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        runtime_v61a.atomic_json_v61a(EVIDENCE, evidence)
        runtime_v61a.atomic_json_v61a(ANALYSIS, null_analysis)
        report = runtime_v61a.self_hashed_v61a({
            "schema": "v61d-singleton-fcfs-paired-evaluator-report",
            "status": "complete_matched_content_free_characterization_sealed",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "support_audit_file_sha256": AUDIT_FILE_SHA256,
            "support_audit_content_sha256": AUDIT_CONTENT_SHA256,
            "runtime_determinism_controls": dict(analysis.RUNTIME_CONTROLS_V61D),
            "matched_v61c_panel_labels_metrics_bootstrap_thresholds": True,
            "v61c_thresholds_relaxed_or_changed": False,
            "adapter_artifact_identities": adapter_ids,
            "panel_file_sha256": runtime_v61c.STAGED_PANEL_FILE_SHA256,
            "panel_content_sha256": runtime_v61c.STAGED_PANEL_CONTENT_SHA256,
            "panel_document_block_audit": panel["document_block_audit"],
            "master_state_receipt": installed, "installations": installations,
            "state_receipts_sha256": evidence["numeric_state_receipts_sha256"],
            "actor_identities": actor_ids,
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
                "content_sha256": evidence["content_sha256_before_self_field"],
                "rows": 68, "actors": 4, "periods": 4,
                "generation_completions": 1088,
                "teacher_forced_requests": 1088,
            },
            "analysis": {
                "path": str(ANALYSIS),
                "file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
                "content_sha256": null_analysis[
                    "content_sha256_before_self_field"
                ],
                "teacher_forced_logprob_primary_eligible": null_analysis[
                    "noise_scale_comparison"
                ]["teacher_forced_logprob_primary_eligible"],
                "exact_sentinel_passed": null_analysis["exact_sentinel"]["passed"],
            },
            "matched_v61c_finalizer": {
                "file_sha256": V61C_FINALIZER_FILE_SHA256,
                "content_sha256": V61C_FINALIZER_CONTENT_SHA256,
            },
            "gpu_activity": gpu, "cleanup": cleanup, "final_gpu_idle": idle,
            "gpu_log_file_sha256": runtime_v61a.file_sha256_v61a(GPU_LOG),
            "alpha": 0.0,
            "adapter_update_or_candidate_materialization_performed": False,
            "holdback_ood_shadow_or_protected_opened": False,
            "raw_question_answer_or_generation_text_persisted": False,
            "hpo_selection_update_or_promotion_authorized": False,
        })
        runtime_v61a.atomic_json_v61a(REPORT, report)
        print(json.dumps({
            "report_file_sha256": runtime_v61a.file_sha256_v61a(REPORT),
            "report_content_sha256": report["content_sha256_before_self_field"],
            "evidence_file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
            "evidence_content_sha256": evidence["content_sha256_before_self_field"],
            "analysis_file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
            "analysis_content_sha256": null_analysis[
                "content_sha256_before_self_field"
            ],
            "teacher_forced_logprob_primary_eligible": null_analysis[
                "noise_scale_comparison"
            ]["teacher_forced_logprob_primary_eligible"],
            "exact_sentinel_passed": null_analysis["exact_sentinel"]["passed"],
            "all_four_gpus_attributed_positive": True,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None: monitor.join(timeout=10)
        if not FAILURE.exists():
            failure = runtime_v61a._sanitize_failure_v61a(error)
            failure.update({
                "schema": "v61d-singleton-fcfs-paired-evaluator-failure",
                "runtime_determinism_controls": dict(analysis.RUNTIME_CONTROLS_V61D),
                "holdback_ood_shadow_or_protected_opened": False,
                "adapter_update_or_candidate_materialization_performed": False,
            })
            runtime_v61a.atomic_json_v61a(
                FAILURE, runtime_v61a.self_hashed_v61a(failure)
            )
        raise
    finally:
        if trainer is not None:
            try: v40a.base.close_trainer(trainer)
            except Exception: pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            prior.v40a.EXPERIMENT, prior.v40a.RUN_DIR, prior.v40a.WORKER_EXTENSION = saved


if __name__ == "__main__":
    raise SystemExit(main())
