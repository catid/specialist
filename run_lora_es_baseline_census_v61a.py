#!/usr/bin/env python3
"""Sealed four-GPU, train-only V434 baseline census for V61A.

The live path scores the unchanged V434 LoRA state only.  It performs no ES
population work, candidate selection, update, promotion, or protected-data
access, and persists no prompt, answer, or generated text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import lora_es_baseline_census_strata_v61a as strata
import lora_es_nested_population_v52 as design_v52
import run_lora_es_nested_population_v52 as runtime_v52


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v61a_v434_train_only_baseline_census"
RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT
).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
EVIDENCE = (RUN_DIR / "baseline_census_evidence_v61a.json").resolve()
STRATA = (RUN_DIR / "baseline_census_strata_v61a.json").resolve()
REPORT = (RUN_DIR / "baseline_census_report_v61a.json").resolve()
FAILURE = (RUN_DIR / "failure_v61a.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v61a.jsonl").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "v434_train_baseline_census_v61a.json"
).resolve()
WORKER_EXTENSION_V61A = runtime_v52.WORKER_EXTENSION_V52
FORBIDDEN_RUNTIME_PATH_TOKENS_V61A = (
    "eval", "ood", "shadow", "holdout", "heldout", "benchmark",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--execute", action="store_true")
    return result


def file_sha256_v61a(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_hashed_v61a(value: dict) -> dict:
    result = dict(value)
    result["content_sha256_before_self_field"] = (
        strata.canonical_sha256_v61a(result)
    )
    return result


def atomic_json_v61a(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def implementation_bindings_v61a() -> dict:
    """Bind implementation code without opening train/model artifacts."""
    paths = {
        "runtime_v61a": Path(__file__).resolve(),
        "builder_v61a": ROOT / "build_lora_es_baseline_census_preregistration_v61a.py",
        "strata_math_v61a": Path(strata.__file__).resolve(),
        "tests_v61a": ROOT / "test_lora_es_baseline_census_v61a.py",
        "design_v52": Path(design_v52.__file__).resolve(),
        "runtime_v52": Path(runtime_v52.__file__).resolve(),
        "worker_v52": ROOT / "eggroll_es_worker_lora_v52.py",
        "worker_v51": ROOT / "eggroll_es_worker_lora_v51.py",
        "worker_v41a": ROOT / "eggroll_es_worker_lora_v41a.py",
        "topology_runtime_v40a": ROOT / "run_lora_topology_probe_v40a.py",
        "topology_runtime_v40c": ROOT / "run_lora_topology_probe_v40c.py",
    }
    return {
        "code_file_sha256": {
            label: file_sha256_v61a(path) for label, path in paths.items()
        },
        "pinned_v52_artifact_identities_without_reopening": {
            "train_dataset_file_sha256": design_v52.DATASET_SHA256_V52,
            "membership_file_sha256": design_v52.MEMBERSHIP_SHA256_V52,
            "membership_content_sha256": design_v52.MEMBERSHIP_CONTENT_SHA256_V52,
            "train_bundle_content_sha256": (
                design_v52.TRAIN_BUNDLE_CONTENT_SHA256_V52
            ),
            "source_weights_file_sha256": design_v52.SOURCE_WEIGHTS_SHA256_V52,
            "source_config_file_sha256": design_v52.SOURCE_CONFIG_SHA256_V52,
            "staged_weights_file_sha256": design_v52.STAGED_WEIGHTS_SHA256_V52,
            "staged_config_file_sha256": design_v52.STAGED_CONFIG_SHA256_V52,
            "staged_manifest_file_sha256": (
                design_v52.STAGED_MANIFEST_FILE_SHA256_V52
            ),
            "canonical_fp32_master_sha256": design_v52.MASTER_SHA256_V52,
            "bf16_runtime_values_sha256": design_v52.MASTER_RUNTIME_SHA256_V52,
            "model_shards_content_sha256": (
                "af8ea3a900c04e97d2d8e3146b8e23be5ee3e6548dea20440020b2f43ee6656e"
            ),
        },
        "train_semantics_or_model_artifacts_opened_to_build_bindings": False,
    }


def _expected_artifacts_v61a() -> dict:
    return {
        "attempt": str(ATTEMPT),
        "run_directory": str(RUN_DIR),
        "evidence": str(EVIDENCE),
        "strata": str(STRATA),
        "report": str(REPORT),
        "failure": str(FAILURE),
        "gpu_log": str(GPU_LOG),
    }


def load_preregistration_v61a(args) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256_v61a(path) != args.preregistration_sha256:
        raise RuntimeError("v61a preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    recipe = value.get("fixed_census_recipe", {})
    access = value.get("access_contract", {})
    runtime_paths = access.get("only_live_semantic_paths_may_open", [])
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or strata.canonical_sha256_v61a(compact)
        != args.preregistration_content_sha256
        or value.get("schema") != "v61a-v434-train-baseline-census-preregistration"
        or value.get("status")
        != "preregistered_before_v61a_model_gpu_or_train_row_access"
        or value.get("gpu_launch_authorized") is not True
        or value.get("train_semantic_access_authorized") is not True
        or value.get("eval_ood_shadow_holdout_access_authorized") is not False
        or value.get("candidate_selection_update_or_promotion_authorized") is not False
        or runtime_paths != [
            str(design_v52.TRAIN_DATASET_V52),
            str(design_v52.TRAIN_MEMBERSHIP_V52),
        ]
        or any(
            token in runtime_path.casefold()
            for runtime_path in runtime_paths
            for token in FORBIDDEN_RUNTIME_PATH_TOKENS_V61A
        )
        or recipe.get("train_dataset") != str(design_v52.TRAIN_DATASET_V52)
        or recipe.get("train_dataset_file_sha256") != design_v52.DATASET_SHA256_V52
        or recipe.get("train_rows") != strata.TRAIN_ROWS_V61A
        or recipe.get("train_conflict_units") != strata.TRAIN_CONFLICT_UNITS_V61A
        or recipe.get("membership") != str(design_v52.TRAIN_MEMBERSHIP_V52)
        or recipe.get("membership_file_sha256") != design_v52.MEMBERSHIP_SHA256_V52
        or recipe.get("actor_generation_seeds")
        != list(strata.ACTOR_GENERATION_SEEDS_V61A)
        or recipe.get("generation_params_without_seed")
        != strata.GENERATION_PARAMS_WITHOUT_SEED_V61A
        or recipe.get("physical_gpu_ids") != [0, 1, 2, 3]
        or recipe.get("canonical_fp32_master_sha256")
        != design_v52.MASTER_SHA256_V52
        or recipe.get("bf16_runtime_values_sha256")
        != design_v52.MASTER_RUNTIME_SHA256_V52
        or recipe.get("worker_extension") != WORKER_EXTENSION_V61A
        or value.get("runtime") != design_v52.RUNTIME_V52
        or value.get("artifacts") != _expected_artifacts_v61a()
        or value.get("implementation_bindings") != implementation_bindings_v61a()
        or value.get("raw_question_answer_or_generation_text_may_be_persisted")
        is not False
    ):
        raise RuntimeError("v61a preregistration contract changed")
    return value


def verify_v434_artifacts_v61a() -> dict:
    observed = {
        "train_dataset": file_sha256_v61a(design_v52.TRAIN_DATASET_V52),
        "membership": file_sha256_v61a(design_v52.TRAIN_MEMBERSHIP_V52),
        "source_weights": file_sha256_v61a(design_v52.SOURCE_WEIGHTS_V52),
        "source_config": file_sha256_v61a(design_v52.SOURCE_CONFIG_V52),
        "staged_weights": file_sha256_v61a(design_v52.STAGED_WEIGHTS_V52),
        "staged_config": file_sha256_v61a(design_v52.STAGED_CONFIG_V52),
        "staged_manifest": file_sha256_v61a(design_v52.STAGED_MANIFEST_V52),
    }
    expected = {
        "train_dataset": design_v52.DATASET_SHA256_V52,
        "membership": design_v52.MEMBERSHIP_SHA256_V52,
        "source_weights": design_v52.SOURCE_WEIGHTS_SHA256_V52,
        "source_config": design_v52.SOURCE_CONFIG_SHA256_V52,
        "staged_weights": design_v52.STAGED_WEIGHTS_SHA256_V52,
        "staged_config": design_v52.STAGED_CONFIG_SHA256_V52,
        "staged_manifest": design_v52.STAGED_MANIFEST_FILE_SHA256_V52,
    }
    if observed != expected:
        raise RuntimeError("v61a V434 train or adapter artifact changed")
    return observed


def load_train_inputs_v61a() -> tuple[list[dict], dict]:
    bundle = runtime_v52.load_train_bundle_v52()
    if (
        bundle.get("content_sha256_before_self_field")
        != design_v52.TRAIN_BUNDLE_CONTENT_SHA256_V52
        or len(bundle.get("questions", [])) != strata.TRAIN_ROWS_V61A
        or len(bundle.get("answers", [])) != strata.TRAIN_ROWS_V61A
        or len(bundle.get("row_sha256", [])) != strata.TRAIN_ROWS_V61A
        or len(bundle.get("unit_membership_v48b", []))
        != strata.TRAIN_ROWS_V61A
        or bundle.get("conflict_units") != strata.TRAIN_CONFLICT_UNITS_V61A
    ):
        raise RuntimeError("v61a exact V434 train bundle changed")
    prepared = []
    for index, (question, answer, row_sha, member) in enumerate(zip(
        bundle["questions"], bundle["answers"], bundle["row_sha256"],
        bundle["unit_membership_v48b"], strict=True,
    )):
        if (
            not question or not answer
            or member.get("row_sha256") != row_sha
            or not isinstance(member.get("unit_identity_sha256"), str)
            or len(member["unit_identity_sha256"]) != 64
            or not isinstance(member.get("row_count"), int)
            or member["row_count"] <= 0
        ):
            raise RuntimeError("v61a V434 row or conflict-unit identity changed")
        prepared.append({
            "row_index": index,
            "row_sha256": row_sha,
            "unit_identity_sha256": member["unit_identity_sha256"],
            "row_count": member["row_count"],
            "question": question,
            "answer": answer,
        })
    if (
        len({item["row_sha256"] for item in prepared}) != strata.TRAIN_ROWS_V61A
        or len({item["unit_identity_sha256"] for item in prepared})
        != strata.TRAIN_CONFLICT_UNITS_V61A
    ):
        raise RuntimeError("v61a V434 train coverage changed")
    return prepared, bundle


def score_actor_outputs_v61a(
    rows: list[dict], outputs: list, actor_rank: int, fused,
) -> list[dict]:
    if (
        len(rows) != strata.TRAIN_ROWS_V61A
        or len(outputs) != strata.TRAIN_ROWS_V61A
        or actor_rank not in range(strata.ACTORS_V61A)
    ):
        raise RuntimeError("v61a actor generation coverage changed")
    metrics = []
    for row, output in zip(rows, outputs, strict=True):
        generated = getattr(output, "outputs", None)
        if not isinstance(generated, list) or len(generated) != 1:
            raise RuntimeError("v61a completion multiplicity changed")
        prediction = fused._extract_answer(str(generated[0].text))
        f1 = float(fused._f1(prediction, row["answer"]))
        expected_tokens = fused._tokens(row["answer"])
        exact = int(
            bool(expected_tokens)
            and fused._tokens(prediction) == expected_tokens
        )
        metric = {
            "actor_rank": actor_rank,
            "generation_seed": strata.ACTOR_GENERATION_SEEDS_V61A[actor_rank],
            "f1": f1,
            "exact": exact,
            "nonzero": int(f1 > 0.0),
        }
        # Enforce the exact persisted schema before generated text leaves scope.
        strata._validate_actor_metric_v61a(metric, actor_rank)
        metrics.append(metric)
    return metrics


def build_evidence_v61a(
    rows: list[dict], actor_metrics: list[list[dict]], master: dict,
) -> dict:
    if (
        len(rows) != strata.TRAIN_ROWS_V61A
        or len(actor_metrics) != strata.ACTORS_V61A
        or any(len(items) != strata.TRAIN_ROWS_V61A for items in actor_metrics)
        or master.get("sha256") != design_v52.MASTER_SHA256_V52
    ):
        raise RuntimeError("v61a evidence coverage or V434 master changed")
    evidence_rows = [{
        "row_index": index,
        "row_sha256": row["row_sha256"],
        "unit_identity_sha256": row["unit_identity_sha256"],
        "row_count": row["row_count"],
        "actors": [actor_metrics[actor][index] for actor in range(4)],
    } for index, row in enumerate(rows)]
    result = {
        "schema": "v61a-v434-train-baseline-census-evidence",
        "status": "complete_characterization_only",
        "train_dataset_file_sha256": design_v52.DATASET_SHA256_V52,
        "membership_file_sha256": design_v52.MEMBERSHIP_SHA256_V52,
        "membership_content_sha256": design_v52.MEMBERSHIP_CONTENT_SHA256_V52,
        "train_bundle_content_sha256": design_v52.TRAIN_BUNDLE_CONTENT_SHA256_V52,
        "canonical_fp32_master_sha256": master["sha256"],
        "row_count": strata.TRAIN_ROWS_V61A,
        "conflict_unit_count": strata.TRAIN_CONFLICT_UNITS_V61A,
        "actor_count": strata.ACTORS_V61A,
        "actor_generation_seeds": list(strata.ACTOR_GENERATION_SEEDS_V61A),
        "generation_params_without_seed": dict(
            strata.GENERATION_PARAMS_WITHOUT_SEED_V61A
        ),
        "rows": evidence_rows,
        "numeric_row_manifest_sha256": strata.canonical_sha256_v61a(evidence_rows),
        "raw_question_answer_or_generation_text_persisted": False,
        "eval_ood_shadow_or_holdout_opened": False,
        "candidate_selection_or_promotion_performed": False,
        "adapter_update_or_master_commit_performed": False,
    }
    result["content_sha256_before_self_field"] = (
        strata.canonical_sha256_v61a(result)
    )
    return result


def _make_trainer_v61a(prereg: dict, prior):
    v40a = prior.v40a
    saved = (v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION)
    v40a.EXPERIMENT = EXPERIMENT
    v40a.RUN_DIR = RUN_DIR
    v40a.WORKER_EXTENSION = WORKER_EXTENSION_V61A
    try:
        trainer = prior.v40c.make_trainer_v40c(prereg)
    except BaseException:
        v40a.EXPERIMENT, v40a.RUN_DIR, v40a.WORKER_EXTENSION = saved
        raise
    return trainer, saved


def _lora_request_v61a(prior):
    from vllm.lora.request import LoRARequest
    return LoRARequest(
        "v434_equal_baseline_census_v61a", 1, str(design_v52.STAGED_V52),
        base_model_name=str(prior.v40a.MODEL),
    )


def _sampling_params_v61a(actor_rank: int):
    from vllm import SamplingParams
    if actor_rank not in range(strata.ACTORS_V61A):
        raise ValueError("v61a invalid actor rank")
    return SamplingParams(
        seed=strata.ACTOR_GENERATION_SEEDS_V61A[actor_rank],
        **strata.GENERATION_PARAMS_WITHOUT_SEED_V61A,
    )


def _sanitize_failure_v61a(error: BaseException) -> dict:
    # Persist only a digest of the message in case an upstream exception ever
    # embeds a prompt or completion.
    return {
        "schema": "v61a-v434-train-baseline-census-failure",
        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        "type": type(error).__name__,
        "message_sha256": hashlib.sha256(str(error).encode("utf-8")).hexdigest(),
        "raw_error_message_or_traceback_persisted": False,
        "raw_question_answer_or_generation_text_persisted": False,
        "eval_ood_shadow_or_holdout_opened": False,
        "candidate_selection_update_or_promotion_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v61a(args)
    if args.dry_run:
        if args.execute:
            raise RuntimeError("v61a dry-run and execute are mutually exclusive")
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "train_rows_opened": 0,
            "model_or_gpu_loaded": False,
            "protected_semantic_access_count": 0,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if not args.execute:
        raise RuntimeError("v61a live path requires --execute")
    runtime_v52.require_live_interpreter_v52()
    if ATTEMPT.exists() or RUN_DIR.exists() or REPORT.exists():
        raise RuntimeError("v61a requires fresh artifact paths")

    # This is the first live-only operation.  The preregistration has already
    # been fully validated above without train/model/GPU access.
    import run_lora_es_generation_boundary_v48b as v48b

    prior = v48b.v43i
    v40a = prior.v40a
    preflight = v40a.gpu_preflight()
    attempt = self_hashed_v61a({
        "schema": "v61a-v434-train-baseline-census-attempt",
        "status": "launching_characterization_only",
        "phase": "before_train_semantics_model_or_gpu_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "eval_ood_shadow_or_holdout_opened": False,
    })
    atomic_json_v61a(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    try:
        artifact_identities = verify_v434_artifacts_v61a()
        rows, bundle = load_train_inputs_v61a()
        prompts = [v40a.base.specialist_template(row["question"]) for row in rows]
        v40a.base.set_seed(prior.GLOBAL_SEED)
        trainer, saved = _make_trainer_v61a(prereg, prior)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        pid_map = prior.prior._actor_pid_map(actor_ids)
        monitor = threading.Thread(
            target=v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures),
            daemon=True,
        )
        monitor.start()
        phase.value = "activate_v434_lora_slot"
        request = _lora_request_v61a(prior)
        activation = v40a._rpc_all(trainer, "add_lora", (request,))
        if activation != [True, True, True, True]:
            raise RuntimeError("v61a four-actor LoRA activation failed")
        phase.value = "install_exact_v434_master"
        installations = v40a._rpc_all(trainer, "install_adapter_state_v41a", (
            str(design_v52.SOURCE_WEIGHTS_V52),
            str(design_v52.SOURCE_CONFIG_V52),
            design_v52.SOURCE_WEIGHTS_SHA256_V52,
            design_v52.SOURCE_CONFIG_SHA256_V52,
        ))
        certificates = v40a._rpc_all(
            trainer, "adapter_state_certificate_v41a"
        )
        masters = [item["current_identity"] for item in certificates]
        runtime_hashes = {
            item["materialization"]["runtime_values_sha256"]
            for item in certificates
        }
        if (
            len({strata.canonical_sha256_v61a(item) for item in masters}) != 1
            or any(
                item.get("sha256") != design_v52.MASTER_SHA256_V52
                for item in masters
            )
            or runtime_hashes != {design_v52.MASTER_RUNTIME_SHA256_V52}
        ):
            raise RuntimeError("v61a V434 master differs across actors")
        phase.value = "v434_baseline_census_all_448_rows_all_four_actors"
        batches = trainer._resolve([
            engine.generate.remote(
                prompts,
                _sampling_params_v61a(actor_rank),
                use_tqdm=False,
                lora_request=request,
            )
            for actor_rank, engine in enumerate(trainer.engines)
        ])
        if (
            len(batches) != strata.ACTORS_V61A
            or any(len(batch) != strata.TRAIN_ROWS_V61A for batch in batches)
        ):
            raise RuntimeError("v61a four-actor census coverage changed")
        actor_metrics = [
            score_actor_outputs_v61a(rows, batch, actor_rank, prior.fused)
            for actor_rank, batch in enumerate(batches)
        ]
        evidence = build_evidence_v61a(rows, actor_metrics, masters[0])
        frozen_strata = strata.build_stratified_census_v61a(evidence)

        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v61a GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = v40a.cleanup_v38a.wait_for_gpu_idle()

        atomic_json_v61a(EVIDENCE, evidence)
        atomic_json_v61a(STRATA, frozen_strata)
        report = self_hashed_v61a({
            "schema": "v61a-v434-train-baseline-census-report",
            "status": "complete_content_free_characterization_sealed",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "artifact_identities": artifact_identities,
            "train_bundle_content_sha256": bundle[
                "content_sha256_before_self_field"
            ],
            "master": masters[0],
            "installations": installations,
            "actor_identities": actor_ids,
            "actor_generation_seeds": list(
                strata.ACTOR_GENERATION_SEEDS_V61A
            ),
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": file_sha256_v61a(EVIDENCE),
                "content_sha256": evidence[
                    "content_sha256_before_self_field"
                ],
                "rows": strata.TRAIN_ROWS_V61A,
                "actors": strata.ACTORS_V61A,
            },
            "strata": {
                "path": str(STRATA),
                "file_sha256": file_sha256_v61a(STRATA),
                "content_sha256": frozen_strata[
                    "content_sha256_before_self_field"
                ],
                "status": frozen_strata["status"],
                "later_v61_hpo_authorized": frozen_strata[
                    "later_v61_hpo_authorized"
                ],
                "stratum_counts": frozen_strata["stratum_counts"],
            },
            "gpu_activity": gpu,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "gpu_log_file_sha256": file_sha256_v61a(GPU_LOG),
            "raw_question_answer_or_generation_text_persisted": False,
            "eval_ood_shadow_or_holdout_opened": False,
            "candidate_selection_update_or_promotion_performed": False,
            "adapter_update_or_master_commit_performed": False,
        })
        atomic_json_v61a(REPORT, report)
        print(json.dumps({
            "report": str(REPORT),
            "report_file_sha256": file_sha256_v61a(REPORT),
            "report_content_sha256": report[
                "content_sha256_before_self_field"
            ],
            "evidence_file_sha256": file_sha256_v61a(EVIDENCE),
            "evidence_content_sha256": evidence[
                "content_sha256_before_self_field"
            ],
            "strata_file_sha256": file_sha256_v61a(STRATA),
            "strata_content_sha256": frozen_strata[
                "content_sha256_before_self_field"
            ],
            "strata_status": frozen_strata["status"],
            "later_v61_hpo_authorized": frozen_strata[
                "later_v61_hpo_authorized"
            ],
            "all_four_gpus_attributed_positive": True,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        if not FAILURE.exists():
            atomic_json_v61a(FAILURE, self_hashed_v61a(
                _sanitize_failure_v61a(error)
            ))
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
            prior.v40a.EXPERIMENT, prior.v40a.RUN_DIR, prior.v40a.WORKER_EXTENSION = saved


if __name__ == "__main__":
    raise SystemExit(main())
