#!/usr/bin/env python3
"""Matched-init, train-only, one-step LoRA EGGROLL-ES run (V43A).

The run uses eight antithetic directions over the frozen v412 fold-3 training
partition and the equal-conflict-unit teacher-forced answer-logprob objective.
It never reads shadow-dev, external validation, OOD, or sealed holdout data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import eggroll_es_worker_v3 as worker_v3
import run_lora_topology_probe_v40a as v40a
import run_lora_topology_probe_v40c as v40c
import train_eggroll_es_equal_unit_v38a as equal_v38
import train_eggroll_es_specialist_anchor_v4 as anchor_v4


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v43e_matched_lora_es_fold3_pop8_step1_numeric_audit_retry"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "matched_lora_es_report_v43e.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v43e.jsonl").resolve()
SNAPSHOT = (RUN_DIR / "adapter_step1_v43e").resolve()
SOURCE = (
    ROOT / "experiments/eggroll_es_hpo/initial_adapters/"
    "matched_lora_initialization_v41a_seed20260715041"
).resolve()
SOURCE_WEIGHTS = (SOURCE / "adapter_model.safetensors").resolve()
SOURCE_CONFIG = (SOURCE / "adapter_config.json").resolve()
SOURCE_MANIFEST = (SOURCE / "initialization_manifest_v41a.json").resolve()
STAGED = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "matched_lora_initialization_v41b"
).resolve()
STAGED_WEIGHTS = (STAGED / "adapter_model.safetensors").resolve()
STAGED_CONFIG = (STAGED / "adapter_config.json").resolve()
STAGED_MANIFEST = (STAGED / "stage_manifest_v41b.json").resolve()
DATASET = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/fold_3_train.jsonl"
).resolve()
SPLIT_MANIFEST = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/manifest_v37a.json"
).resolve()
DATASET_SHA256 = "97fc920ac39f67536df26977de951e8c34bf8486eb8f42fbb0a67687f025a92a"
SPLIT_MANIFEST_SHA256 = "7d2a8f2b86f9007aa2bfe8ae043be15647451cc4bbea53a18d5915085879ee9d"
TRAIN_BUNDLE_SHA256 = "ba0a951f153b60fb5729e0169a4d780277d9d860cbacb1f95fc7433afc875e19"
WORKER_EXTENSION = (
    "eggroll_es_worker_lora_v41a.LoRAAdapterStateWorkerExtensionV41A"
)
POPULATION_SIZE = 8
SEEDS = [
    140002291, 1028842752, 480373990, 1037026679,
    759861149, 227761095, 428721957, 150663570,
]
SIGMA = 0.0003
ALPHA = 0.00015
GLOBAL_SEED = 2_026_071_543
STANDARDIZATION_EPSILON = 1e-8
CROSS_ACTOR_SCORE_ATOL = 1e-5


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _paths() -> dict[str, Path]:
    return {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_equal_unit_preregistration_v43a.py",
        "initialization_builder": ROOT / "build_matched_lora_initialization_v41a.py",
        "staging_runtime": ROOT / "stage_matched_lora_initialization_vllm_v41b.py",
        "worker": ROOT / "eggroll_es_worker_lora_v41a.py",
        "worker_v3": ROOT / "eggroll_es_worker_v3.py",
        "runtime_v40a": Path(v40a.__file__).resolve(),
        "resolver_runtime_v40c": Path(v40c.__file__).resolve(),
        "equal_unit_runtime_v38": Path(equal_v38.__file__).resolve(),
        "dense_reward_runtime_v4": Path(anchor_v4.__file__).resolve(),
        "source_weights": SOURCE_WEIGHTS,
        "source_config": SOURCE_CONFIG,
        "source_manifest": SOURCE_MANIFEST,
        "staged_weights": STAGED_WEIGHTS,
        "staged_config": STAGED_CONFIG,
        "staged_manifest": STAGED_MANIFEST,
        "dataset": DATASET,
        "split_manifest": SPLIT_MANIFEST,
        "model_config": v40a.MODEL / "config.json",
        "model_index": v40a.MODEL / "model.safetensors.index.json",
        "tuned_table": v40a.TUNED_FILE,
        "base_runtime": ROOT / "train_eggroll_es_specialist.py",
        "cleanup_runtime": ROOT / "run_eggroll_es_equal_unit_v38a.py",
    }


def implementation_bindings() -> dict[str, str]:
    result = {key: v40a.file_sha256(path) for key, path in _paths().items()}
    result["model_shards_content_sha256"] = v40a.MODEL_SHARDS_CONTENT_SHA256
    return result


def load_preregistration(args: argparse.Namespace) -> dict:
    path = Path(args.preregistration).resolve()
    if v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v43a preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {key: item for key, item in value.items()
               if key != "content_sha256_before_self_field"}
    recipe = value.get("recipe", {})
    if (
        content != args.preregistration_content_sha256
        or content != v40a.canonical_sha256(compact)
        or value.get("schema") != "matched-lora-es-preregistration-v43e"
        or value.get("status") != "preregistered_before_train_only_launch"
        or value.get("shadow_dev_external_eval_ood_or_holdout_authorized") is not False
        or value.get("sealed_holdout_opened") is not False
        or value.get("implementation_bindings") != implementation_bindings()
        or recipe.get("population_size") != POPULATION_SIZE
        or recipe.get("seeds") != SEEDS
        or recipe.get("sigma") != SIGMA
        or recipe.get("alpha") != ALPHA
        or recipe.get("dataset") != str(DATASET)
        or recipe.get("dataset_sha256") != DATASET_SHA256
        or recipe.get("train_bundle_content_sha256") != TRAIN_BUNDLE_SHA256
        or recipe.get("matched_initialization") != str(SOURCE)
        or recipe.get("staged_initialization") != str(STAGED)
        or recipe.get("worker_extension") != WORKER_EXTENSION
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != recipe.get("tuned_table_content_sha256")
        or recipe.get("cross_actor_score_atol") != CROSS_ACTOR_SCORE_ATOL
    ):
        raise RuntimeError("v43a preregistration contract changed")
    forbidden = ("shadow_dev", "eval_qa", "ood_qa", "holdout", "heldout")
    for label in ("dataset", "matched_initialization", "staged_initialization"):
        if any(token in str(recipe[label]).lower() for token in forbidden):
            raise RuntimeError(f"v43a forbidden selection path in {label}")
    return value


def _make_trainer(prereg: dict):
    saved = (v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION)
    v40a.EXPERIMENT = EXPERIMENT
    v40a.RUN_DIR = RUN_DIR
    v40a.WORKER_EXTENSION = WORKER_EXTENSION
    try:
        trainer = v40c.make_trainer_v40c(prereg)
    except Exception:
        v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved
        raise
    return trainer, saved


def _actor_pid_map(values: list[dict]) -> dict[int, int]:
    result = {int(value["physical_gpu_id"]): int(value["pid"]) for value in values}
    if set(result) != {0, 1, 2, 3} or len(set(result.values())) != 4:
        raise RuntimeError("v43a actor/GPU coverage changed")
    return result


def _lora_request():
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "matched_lora_initialization_v41b", 1, str(STAGED),
        base_model_name=str(v40a.MODEL),
    )


def _sampling_params():
    from vllm import SamplingParams
    return SamplingParams(
        n=1, seed=GLOBAL_SEED, temperature=0.0, top_p=1.0,
        max_tokens=1, prompt_logprobs=1, detokenize=False,
    )


def _prepare(trainer, bundle: dict):
    prompts = [v40a.base.specialist_template(question) for question in bundle["questions"]]
    dense_items = anchor_v4.prepare_gold_answer_items_v4(
        trainer.tokenizer, prompts, bundle["answers"],
    )
    requests = [{"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items]
    return dense_items, requests


def _score_batch(bundle: dict, dense_items: list, outputs: list) -> dict:
    return equal_v38._score_equal_unit(bundle, dense_items, outputs)


def _base_score(trainer, bundle, dense_items, requests) -> dict:
    batches = trainer._resolve([
        engine.generate.remote(
            requests, _sampling_params(), use_tqdm=False,
            lora_request=_lora_request(),
        ) for engine in trainer.engines
    ])
    if len(batches) != 4 or any(len(batch) != 448 for batch in batches):
        raise RuntimeError("v43a base score coverage changed")
    scores = [_score_batch(bundle, dense_items, batch) for batch in batches]
    exact_hashes = [v40a.canonical_sha256(score) for score in scores]
    equal_values = [score["equal_unit_mean"] for score in scores]
    row_values = [score["unweighted_row_mean"] for score in scores]
    equal_spread = max(equal_values) - min(equal_values)
    row_spread = max(row_values) - min(row_values)
    if (
        len({score["scored_answer_tokens"] for score in scores}) != 1
        or equal_spread > CROSS_ACTOR_SCORE_ATOL
        or row_spread > CROSS_ACTOR_SCORE_ATOL
    ):
        raise RuntimeError(
            "v43a cross-actor train-score dispersion exceeded tolerance: "
            f"equal={equal_spread} row={row_spread}"
        )
    return {
        "consensus": scores[0], "actors": scores,
        "exact_actor_score_sha256": exact_hashes,
        "exact_bitwise_consensus": len(set(exact_hashes)) == 1,
        "cross_actor_score_atol": CROSS_ACTOR_SCORE_ATOL,
        "equal_unit_mean_spread": equal_spread,
        "unweighted_row_mean_spread": row_spread,
    }


def _standardize(values: list[float]) -> tuple[list[float], dict]:
    values = [float(value) for value in values]
    if len(values) != POPULATION_SIZE or not all(math.isfinite(value) for value in values):
        raise RuntimeError("v43a response vector is incomplete or non-finite")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    coefficients = (
        [0.0] * len(values) if std == 0.0 else
        [(value - mean) / (std + STANDARDIZATION_EPSILON) for value in values]
    )
    return coefficients, {"mean": mean, "std": std, "zero_spread": std == 0.0}


def _population(trainer, bundle, dense_items, requests, master_sha: str) -> dict:
    signed = {"plus": [], "minus": []}
    unweighted = {"plus": [], "minus": []}
    dense_hashes = {"plus": [], "minus": []}
    perturbation_certificates = {"plus": [], "minus": []}
    restoration_certificates = {"plus": [], "minus": []}
    for start in range(0, POPULATION_SIZE, 4):
        wave = SEEDS[start:start + 4]
        for label, sign in (("plus", 1), ("minus", -1)):
            values = trainer._resolve([
                trainer.engines[index].collective_rpc.remote(
                    "materialize_antithetic_adapter_v41a",
                    args=(seed, SIGMA, sign, master_sha),
                ) for index, seed in enumerate(wave)
            ])
            if len(values) != 4 or any(len(value) != 1 for value in values):
                raise RuntimeError("v43a perturbation RPC coverage changed")
            perturbation_certificates[label].extend(value[0] for value in values)
            batches = None
            try:
                batches = trainer._resolve([
                    trainer.engines[index].generate.remote(
                        requests, _sampling_params(), use_tqdm=False,
                        lora_request=_lora_request(),
                    ) for index in range(4)
                ])
            finally:
                restored = v40a._rpc_all(trainer, "restore_adapter_master_v41a")
                restoration_certificates[label].extend(restored)
            if len(batches) != 4 or any(len(batch) != 448 for batch in batches):
                raise RuntimeError("v43a signed population wave is incomplete")
            for batch in batches:
                score = _score_batch(bundle, dense_items, batch)
                signed[label].append(score["equal_unit_mean"])
                unweighted[label].append(score["unweighted_row_mean"])
                dense_hashes[label].append(score["dense_result_sha256"])
    central = [0.5 * (plus - minus)
               for plus, minus in zip(signed["plus"], signed["minus"], strict=True)]
    coefficients, standardization = _standardize(central)
    if standardization["zero_spread"]:
        raise RuntimeError("v43a population response has zero spread")
    return {
        "equal_unit_sign_scores": signed,
        "unweighted_row_sign_scores": unweighted,
        "dense_result_sha256": dense_hashes,
        "central_response": central,
        "coefficients": coefficients,
        "standardization": standardization,
        "perturbation_certificates": perturbation_certificates,
        "restoration_certificates": restoration_certificates,
    }


def _apply_update(trainer, master: dict, reference_generation: int,
                  coefficients: list[float], plan_id: str) -> dict:
    coefficient_sha = worker_v3.coefficient_sha256_v3(SEEDS, coefficients)
    prepared = v40a._rpc_all(trainer, "prepare_sharded_adapter_update_v41a", (
        SEEDS, coefficients, coefficient_sha, POPULATION_SIZE, 4, ALPHA,
        plan_id, master["sha256"], reference_generation,
    ))
    manifests = {item["manifest_sha256"] for item in prepared}
    if len(manifests) != 1 or {item["rank"] for item in prepared} != {0, 1, 2, 3}:
        raise RuntimeError("v43a prepared update consensus changed")
    manifest_sha = next(iter(manifests))
    executed = v40a._rpc_all(
        trainer, "execute_sharded_adapter_update_v41a", (manifest_sha,),
    )
    final_identities = [item["candidate_identity"] for item in executed]
    if len({v40a.canonical_sha256(item) for item in final_identities}) != 1:
        raise RuntimeError("v43a update candidate differs across ranks")
    final_identity = final_identities[0]
    committed = v40a._rpc_all(trainer, "commit_sharded_adapter_update_v41a", (
        manifest_sha, final_identity["sha256"],
    ))
    if (
        any(item["final_identity"] != final_identity for item in committed)
        or any(item.get("requires_cross_rank_finalize") is not True
               for item in committed)
    ):
        raise RuntimeError("v43a committed identity differs across ranks")
    finalized = v40a._rpc_all(trainer, "finalize_sharded_adapter_update_v41a", (
        manifest_sha, final_identity["sha256"],
    ))
    if (
        any(item.get("finalized") is not True for item in finalized)
        or any(item["final_identity"] != final_identity for item in finalized)
    ):
        raise RuntimeError("v43a finalized identity differs across ranks")
    references = v40a._rpc_all(trainer, "capture_adapter_reference_v41a")
    snapshots = v40a._rpc_all(
        trainer, "save_adapter_snapshot_v41a", (str(SNAPSHOT), final_identity["sha256"]),
    )
    written = [item for item in snapshots if item.get("written")]
    if len(written) != 1 or not written[0].get("readback_verified"):
        raise RuntimeError("v43a snapshot readback coverage changed")
    return {
        "coefficient_sha256": coefficient_sha,
        "manifest_sha256": manifest_sha,
        "prepared": prepared, "executed": executed, "committed": committed,
        "finalized": finalized,
        "final_identity": final_identity, "new_references": references,
        "snapshots": snapshots,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "train_only": True, "sealed_holdout_opened": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v43a requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "matched-lora-es-attempt-v43e", "status": "launching",
        "phase": "before_model_or_train_data_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "shadow_dev_external_eval_ood_or_holdout_opened": False,
    })
    v40a.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    try:
        bundle = equal_v38.load_equal_unit_train_bundle(
            DATASET, DATASET_SHA256, SPLIT_MANIFEST, SPLIT_MANIFEST_SHA256,
        )
        if bundle["content_sha256_before_self_field"] != TRAIN_BUNDLE_SHA256:
            raise RuntimeError("v43a frozen train bundle identity changed")
        v40a.base.set_seed(GLOBAL_SEED)
        trainer, saved = _make_trainer(prereg)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        pid_map = _actor_pid_map(actor_ids)
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
        )
        monitor.start()
        dense_items, requests = _prepare(trainer, bundle)

        phase.value = "activate_matched_initialization"
        preinstall = _base_score(trainer, bundle, dense_items, requests)
        phase.value = "install_canonical_master"
        installs = v40a._rpc_all(trainer, "install_adapter_state_v41a", (
            str(SOURCE_WEIGHTS), str(SOURCE_CONFIG),
            v40a.file_sha256(SOURCE_WEIGHTS), v40a.file_sha256(SOURCE_CONFIG),
        ))
        certificates = v40a._rpc_all(trainer, "adapter_state_certificate_v41a")
        masters = [item["current_identity"] for item in certificates]
        if len({v40a.canonical_sha256(item) for item in masters}) != 1:
            raise RuntimeError("v43a canonical master differs across ranks")
        master = masters[0]
        references = v40a._rpc_all(trainer, "capture_adapter_reference_v41a")
        reference_generations = {item["reference_generation"] for item in references}
        if len(reference_generations) != 1:
            raise RuntimeError("v43a reference generation differs across ranks")
        postinstall = _base_score(trainer, bundle, dense_items, requests)
        if preinstall["consensus"] != postinstall["consensus"]:
            raise RuntimeError("v43a canonical install changed matched-init score")

        phase.value = "population_pop8"
        population = _population(
            trainer, bundle, dense_items, requests, master["sha256"],
        )
        phase.value = "post_population_identity"
        post_population = _base_score(trainer, bundle, dense_items, requests)
        if post_population["consensus"] != postinstall["consensus"]:
            raise RuntimeError("v43a population restore changed base score")
        plan_id = v40a.canonical_sha256({
            "schema": "matched-lora-es-update-plan-v43e",
            "master_sha256": master["sha256"], "seeds": SEEDS,
            "coefficients": population["coefficients"],
            "sigma": SIGMA, "alpha": ALPHA,
            "train_bundle_content_sha256": TRAIN_BUNDLE_SHA256,
            "reference_generation": next(iter(reference_generations)),
        })
        phase.value = "distributed_fp32_update"
        update = _apply_update(
            trainer, master, next(iter(reference_generations)),
            population["coefficients"], plan_id,
        )
        phase.value = "post_update_train_score"
        post_update = _base_score(trainer, bundle, dense_items, requests)

        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v43a GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = v40a.cleanup_v38a.wait_for_gpu_idle()
        report = v40a.self_hashed({
            "schema": "matched-lora-es-train-only-report-v43e",
            "status": "complete_one_nonzero_update_state_sealed",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "train_bundle": {
                "path": str(DATASET), "sha256": DATASET_SHA256,
                "content_sha256": TRAIN_BUNDLE_SHA256,
                "rows": 448, "conflict_units": 208,
                "weight_identity_sha256": bundle["weight_identity_sha256"],
            },
            "recipe": {
                "population_size": POPULATION_SIZE, "seeds": SEEDS,
                "sigma": SIGMA, "alpha": ALPHA,
                "signed_sequence_presentations": 2 * POPULATION_SIZE * 448,
                "objective": anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                "matched_initialization_weights_sha256": v40a.file_sha256(SOURCE_WEIGHTS),
            },
            "actor_identities": actor_ids, "physical_gpu_pid_map": pid_map,
            "installations": installs, "initial_master_identity": master,
            "initial_references": references,
            "score_audits": {
                "preinstall": preinstall, "postinstall": postinstall,
                "post_population": post_population, "post_update": post_update,
                "canonical_install_identity_passed": True,
                "population_exact_restore_passed": True,
            },
            "population": population, "plan_id": plan_id, "update": update,
            "gpu_activity": gpu, "placement_group_cleanup": cleanup,
            "final_gpu_idle": idle, "preflight": preflight,
            "gpu_log": {"path": str(GPU_LOG), "file_sha256": v40a.file_sha256(GPU_LOG)},
            "shadow_dev_external_eval_ood_or_holdout_opened": False,
            "quality_or_promotion_conclusion_authorized": False,
        })
        v40a.atomic_json(REPORT, report)
        print(json.dumps({
            "report": str(REPORT), "report_sha256": v40a.file_sha256(REPORT),
            "initial_equal_unit_mean": postinstall["consensus"]["equal_unit_mean"],
            "post_update_equal_unit_mean": post_update["consensus"]["equal_unit_mean"],
            "response_std": population["standardization"]["std"],
            "snapshot": str(SNAPSHOT),
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        v40a.atomic_json(RUN_DIR / "failure_v43e.json", v40a.self_hashed({
            "schema": "matched-lora-es-failure-v43e",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "shadow_dev_external_eval_ood_or_holdout_opened": False,
        }))
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


if __name__ == "__main__":
    raise SystemExit(main())
