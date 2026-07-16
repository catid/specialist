#!/usr/bin/env python3
"""Sealed four-GPU V434 common-seed, two-pass repeat census V61B."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import lora_es_baseline_repeat_census_v61b as analysis
import lora_es_nested_population_v52 as design_v52
import run_lora_es_baseline_census_v61a as runtime_v61a
import run_lora_es_nested_population_v52 as runtime_v52


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v61b_v434_common_seed_repeat_census"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
EVIDENCE = (RUN_DIR / "common_seed_repeat_evidence_v61b.json").resolve()
ANALYSIS = (RUN_DIR / "common_seed_repeat_analysis_v61b.json").resolve()
REPORT = (RUN_DIR / "common_seed_repeat_report_v61b.json").resolve()
FAILURE = (RUN_DIR / "failure_v61b.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v61b.jsonl").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v434_common_seed_repeat_census_v61b.json"
).resolve()
WORKER_EXTENSION_V61B = runtime_v52.WORKER_EXTENSION_V52


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--preregistration", required=True)
    value.add_argument("--preregistration-sha256", required=True)
    value.add_argument("--preregistration-content-sha256", required=True)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--execute", action="store_true")
    return value


def implementation_bindings_v61b() -> dict:
    paths = {
        "runtime_v61b": Path(__file__).resolve(),
        "builder_v61b": ROOT / "build_lora_es_baseline_repeat_census_preregistration_v61b.py",
        "analysis_v61b": Path(analysis.__file__).resolve(),
        "tests_v61b": ROOT / "test_lora_es_baseline_repeat_census_v61b.py",
        "runtime_v61a": Path(runtime_v61a.__file__).resolve(),
        "design_v52": Path(design_v52.__file__).resolve(),
        "runtime_v52": Path(runtime_v52.__file__).resolve(),
        "worker_v52": ROOT / "eggroll_es_worker_lora_v52.py",
        "worker_v51": ROOT / "eggroll_es_worker_lora_v51.py",
        "worker_v41a": ROOT / "eggroll_es_worker_lora_v41a.py",
    }
    return {
        "code_file_sha256": {
            key: runtime_v61a.file_sha256_v61a(path)
            for key, path in paths.items()
        },
        "pinned_v434_identities_without_reopening": runtime_v61a.implementation_bindings_v61a()[
            "pinned_v52_artifact_identities_without_reopening"
        ],
        "bound_v61a_aggregate": dict(analysis.V61A_BOUND_AGGREGATES),
        "train_model_gpu_or_v61a_row_evidence_opened_to_build_bindings": False,
    }


def _artifacts_v61b() -> dict:
    return {
        "attempt": str(ATTEMPT), "run_directory": str(RUN_DIR),
        "evidence": str(EVIDENCE), "analysis": str(ANALYSIS),
        "report": str(REPORT), "failure": str(FAILURE),
        "gpu_log": str(GPU_LOG),
    }


def load_preregistration_v61b(args) -> dict:
    path = Path(args.preregistration).resolve()
    if runtime_v61a.file_sha256_v61a(path) != args.preregistration_sha256:
        raise RuntimeError("v61b preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {key: item for key, item in value.items()
               if key != "content_sha256_before_self_field"}
    recipe = value.get("fixed_census_recipe", {})
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or analysis.canonical_sha256_v61b(compact)
        != args.preregistration_content_sha256
        or value.get("schema") != "v61b-v434-common-seed-repeat-census-preregistration"
        or value.get("status") != "preregistered_before_v61b_train_model_or_gpu_access"
        or value.get("gpu_launch_authorized") is not True
        or value.get("selection_update_or_promotion_authorized") is not False
        or value.get("eval_ood_shadow_holdout_access_authorized") is not False
        or recipe.get("train_rows") != analysis.TRAIN_ROWS_V61B
        or recipe.get("train_conflict_units") != analysis.TRAIN_CONFLICT_UNITS_V61B
        or recipe.get("actors") != analysis.ACTORS_V61B
        or recipe.get("sequential_passes_per_actor") != analysis.PASSES_V61B
        or recipe.get("common_generation_seed") != analysis.COMMON_GENERATION_SEED_V61B
        or recipe.get("generation_params_without_seed")
        != analysis.GENERATION_PARAMS_WITHOUT_SEED_V61B
        or recipe.get("f1_absolute_delta_thresholds")
        != list(analysis.F1_ABSOLUTE_DELTA_THRESHOLDS_V61B)
        or recipe.get("physical_gpu_ids") != [0, 1, 2, 3]
        or recipe.get("total_completions") != 3584
        or recipe.get("canonical_fp32_master_sha256") != design_v52.MASTER_SHA256_V52
        or recipe.get("worker_extension") != WORKER_EXTENSION_V61B
        or value.get("bound_v61a_distinct_seed_aggregate")
        != analysis.V61A_BOUND_AGGREGATES
        or value.get("runtime") != design_v52.RUNTIME_V52
        or value.get("implementation_bindings") != implementation_bindings_v61b()
        or value.get("artifacts") != _artifacts_v61b()
        or value.get("raw_question_answer_or_generation_text_may_be_persisted") is not False
    ):
        raise RuntimeError("v61b preregistration contract changed")
    return value


def score_outputs_v61b(rows, outputs, actor_rank: int, pass_index: int, fused):
    if (
        len(rows) != 448 or len(outputs) != 448
        or actor_rank not in range(4) or pass_index not in range(2)
    ):
        raise RuntimeError("v61b actor/pass generation coverage changed")
    result = []
    for row, output in zip(rows, outputs, strict=True):
        generated = getattr(output, "outputs", None)
        if not isinstance(generated, list) or len(generated) != 1:
            raise RuntimeError("v61b completion multiplicity changed")
        prediction = fused._extract_answer(str(generated[0].text))
        f1 = float(fused._f1(prediction, row["answer"]))
        expected = fused._tokens(row["answer"])
        exact = int(bool(expected) and fused._tokens(prediction) == expected)
        metric = {
            "actor_rank": actor_rank,
            "pass_index": pass_index,
            "generation_seed": analysis.COMMON_GENERATION_SEED_V61B,
            "f1": f1, "exact": exact, "nonzero": int(f1 > 0.0),
        }
        analysis._metric_v61b(metric, actor_rank, pass_index)
        result.append(metric)
    return result


def build_evidence_v61b(rows, metrics, master: dict) -> dict:
    if (
        len(rows) != 448 or len(metrics) != 2
        or any(len(pass_metrics) != 4 for pass_metrics in metrics)
        or any(len(actor_metrics) != 448 for pass_metrics in metrics
               for actor_metrics in pass_metrics)
        or master.get("sha256") != design_v52.MASTER_SHA256_V52
    ):
        raise RuntimeError("v61b evidence coverage or master changed")
    evidence_rows = [{
        "row_index": index,
        "row_sha256": row["row_sha256"],
        "unit_identity_sha256": row["unit_identity_sha256"],
        "row_count": row["row_count"],
        "passes": [{
            "pass_index": pass_index,
            "actors": [metrics[pass_index][actor][index] for actor in range(4)],
        } for pass_index in range(2)],
    } for index, row in enumerate(rows)]
    value = {
        "schema": "v61b-v434-common-seed-repeat-census-evidence",
        "status": "complete_characterization_only",
        "train_dataset_file_sha256": design_v52.DATASET_SHA256_V52,
        "membership_file_sha256": design_v52.MEMBERSHIP_SHA256_V52,
        "membership_content_sha256": design_v52.MEMBERSHIP_CONTENT_SHA256_V52,
        "train_bundle_content_sha256": design_v52.TRAIN_BUNDLE_CONTENT_SHA256_V52,
        "canonical_fp32_master_sha256": master["sha256"],
        "row_count": 448, "conflict_unit_count": 208,
        "actor_count": 4, "pass_count": 2,
        "common_generation_seed": analysis.COMMON_GENERATION_SEED_V61B,
        "generation_params_without_seed": dict(
            analysis.GENERATION_PARAMS_WITHOUT_SEED_V61B
        ),
        "strictly_sequential_pass_order": [0, 1],
        "rows": evidence_rows,
        "numeric_actor_pass_manifest_sha256": analysis.canonical_sha256_v61b(
            evidence_rows
        ),
        "raw_question_answer_or_generation_text_persisted": False,
        "selection_update_or_promotion_performed": False,
        "eval_ood_shadow_or_holdout_opened": False,
    }
    value["content_sha256_before_self_field"] = analysis.canonical_sha256_v61b(value)
    return value


def _make_trainer_v61b(prereg, prior):
    v40a = prior.v40a
    saved = (v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION)
    v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = (
        EXPERIMENT, RUN_DIR, WORKER_EXTENSION_V61B,
    )
    try:
        trainer = prior.v40c.make_trainer_v40c(prereg)
    except BaseException:
        v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved
        raise
    return trainer, saved


def _request_v61b(prior):
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "v434_common_seed_repeat_census_v61b", 1, str(design_v52.STAGED_V52),
        base_model_name=str(prior.v40a.MODEL),
    )


def _params_v61b():
    from vllm import SamplingParams
    return SamplingParams(
        seed=analysis.COMMON_GENERATION_SEED_V61B,
        **analysis.GENERATION_PARAMS_WITHOUT_SEED_V61B,
    )


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v61b(args)
    if args.dry_run:
        if args.execute:
            raise RuntimeError("v61b dry-run and execute are mutually exclusive")
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "train_rows_opened": 0, "v61a_row_evidence_opened": False,
            "model_or_gpu_loaded": False, "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if not args.execute:
        raise RuntimeError("v61b live path requires --execute")
    runtime_v52.require_live_interpreter_v52()
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v61b requires fresh paths")

    import run_lora_es_generation_boundary_v48b as v48b
    prior = v48b.v43i
    v40a = prior.v40a
    preflight = v40a.gpu_preflight()
    attempt = runtime_v61a.self_hashed_v61a({
        "schema": "v61b-v434-common-seed-repeat-census-attempt",
        "status": "launching_characterization_only",
        "phase": "before_train_model_or_gpu_access",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "v61a_row_evidence_opened": False,
        "eval_ood_shadow_or_holdout_opened": False,
    })
    runtime_v61a.atomic_json_v61a(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event(); failures: queue.Queue = queue.Queue()
    phase = v40a.Phase(); started = time.monotonic()
    try:
        identities = runtime_v61a.verify_v434_artifacts_v61a()
        rows, bundle = runtime_v61a.load_train_inputs_v61a()
        prompts = [v40a.base.specialist_template(row["question"]) for row in rows]
        v40a.base.set_seed(prior.GLOBAL_SEED)
        trainer, saved = _make_trainer_v61b(prereg, prior)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        pid_map = prior.prior._actor_pid_map(actor_ids)
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
        ); monitor.start()
        request = _request_v61b(prior)
        phase.value = "activate_v434_lora_slot"
        if v40a._rpc_all(trainer, "add_lora", (request,)) != [True] * 4:
            raise RuntimeError("v61b four-actor LoRA activation failed")
        phase.value = "install_exact_v434_master"
        installations = v40a._rpc_all(trainer, "install_adapter_state_v41a", (
            str(design_v52.SOURCE_WEIGHTS_V52), str(design_v52.SOURCE_CONFIG_V52),
            design_v52.SOURCE_WEIGHTS_SHA256_V52,
            design_v52.SOURCE_CONFIG_SHA256_V52,
        ))
        certificates = v40a._rpc_all(trainer, "adapter_state_certificate_v41a")
        masters = [item["current_identity"] for item in certificates]
        runtime_hashes = {item["materialization"]["runtime_values_sha256"]
                          for item in certificates}
        if (
            len({analysis.canonical_sha256_v61b(item) for item in masters}) != 1
            or any(item.get("sha256") != design_v52.MASTER_SHA256_V52 for item in masters)
            or runtime_hashes != {design_v52.MASTER_RUNTIME_SHA256_V52}
        ):
            raise RuntimeError("v61b V434 master differs across actors")
        all_metrics = []
        for pass_index in range(2):
            phase.value = f"common_seed_repeat_pass_{pass_index}_all_448_rows"
            batches = trainer._resolve([
                engine.generate.remote(
                    prompts, _params_v61b(), use_tqdm=False, lora_request=request,
                ) for engine in trainer.engines
            ])
            if len(batches) != 4 or any(len(batch) != 448 for batch in batches):
                raise RuntimeError("v61b four-actor pass coverage changed")
            all_metrics.append([
                score_outputs_v61b(rows, batch, actor, pass_index, prior.fused)
                for actor, batch in enumerate(batches)
            ])
        evidence = build_evidence_v61b(rows, all_metrics, masters[0])
        repeat_analysis = analysis.build_repeat_analysis_v61b(evidence)
        stop.set(); monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v61b GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown(); idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        runtime_v61a.atomic_json_v61a(EVIDENCE, evidence)
        runtime_v61a.atomic_json_v61a(ANALYSIS, repeat_analysis)
        report = runtime_v61a.self_hashed_v61a({
            "schema": "v61b-v434-common-seed-repeat-census-report",
            "status": "complete_content_free_characterization_sealed",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "artifact_identities": identities,
            "train_bundle_content_sha256": bundle["content_sha256_before_self_field"],
            "master": masters[0], "installations": installations,
            "actor_identities": actor_ids,
            "common_generation_seed": analysis.COMMON_GENERATION_SEED_V61B,
            "strictly_sequential_passes": [0, 1],
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
                "content_sha256": evidence["content_sha256_before_self_field"],
                "rows": 448, "actors": 4, "passes": 2, "completions": 3584,
            },
            "analysis": {
                "path": str(ANALYSIS),
                "file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
                "content_sha256": repeat_analysis["content_sha256_before_self_field"],
            },
            "gpu_activity": gpu, "cleanup": cleanup, "final_gpu_idle": idle,
            "gpu_log_file_sha256": runtime_v61a.file_sha256_v61a(GPU_LOG),
            "raw_question_answer_or_generation_text_persisted": False,
            "selection_update_or_promotion_performed": False,
            "eval_ood_shadow_or_holdout_opened": False,
            "v61a_row_level_evidence_opened": False,
        })
        runtime_v61a.atomic_json_v61a(REPORT, report)
        print(json.dumps({
            "report_file_sha256": runtime_v61a.file_sha256_v61a(REPORT),
            "report_content_sha256": report["content_sha256_before_self_field"],
            "evidence_file_sha256": runtime_v61a.file_sha256_v61a(EVIDENCE),
            "evidence_content_sha256": evidence["content_sha256_before_self_field"],
            "analysis_file_sha256": runtime_v61a.file_sha256_v61a(ANALYSIS),
            "analysis_content_sha256": repeat_analysis["content_sha256_before_self_field"],
            "all_four_gpus_attributed_positive": True,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None: monitor.join(timeout=10)
        if not FAILURE.exists():
            failure = runtime_v61a._sanitize_failure_v61a(error)
            failure["schema"] = "v61b-v434-common-seed-repeat-census-failure"
            failure["v61a_row_level_evidence_opened"] = False
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
        except Exception: pass
        if saved is not None:
            prior.v40a.EXPERIMENT, prior.v40a.RUN_DIR, prior.v40a.WORKER_EXTENSION = saved


if __name__ == "__main__":
    raise SystemExit(main())
