#!/usr/bin/env python3
"""Four-GPU train-only matched-base greedy evidence collection for V48B."""

from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from qa_quality import qa_pair_from_record

import build_v48b_train_membership as membership_builder
import lora_es_f1_anchor_projection_v43m as v43m
import lora_es_generation_boundary_sampling_v48a as boundary
import run_lora_es_multi_anchor_v43i as v43i
import stage_v48b_train_input as train_stage


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v48c_train_only_matched_base_generation_evidence_activation_retry"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
EVIDENCE = (RUN_DIR / "base_generation_evidence_v48c.json").resolve()
REPORT = (RUN_DIR / "base_generation_evidence_report_v48c.json").resolve()
FAILURE = (RUN_DIR / "failure_v48c.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v48c.jsonl").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_base_generation_evidence_v48c.json"
).resolve()
TRAIN_DATASET = train_stage.OUTPUT
TRAIN_STAGE_MANIFEST = train_stage.MANIFEST
MEMBERSHIP = membership_builder.OUTPUT
EXPECTED_TRAIN_SHA256 = v43i.DATASET_SHA256
EXPECTED_TRAIN_STAGE_MANIFEST_SHA256 = (
    "3afd441915a86267fd3882208375d3285abd7aeca87157c33fcfbb81db431a99"
)
EXPECTED_TRAIN_STAGE_MANIFEST_CONTENT_SHA256 = (
    "b86b23be060cbd0bb1026637acd9fcebdb7c727675c47fa31d4d1dc94a98cd7a"
)
EXPECTED_MEMBERSHIP_SHA256 = (
    "13f30232dd735451d4b90682b335072cbcc86de263c0984170f931721f764a9f"
)
EXPECTED_MEMBERSHIP_CONTENT_SHA256 = (
    "fd3ca8d6a8c03cac15320394dfcf5c4590dff0ab01f1492c67716c80e19b3e4a"
)
FORBIDDEN_PATH_TOKENS = (
    "shadow", "eval", "ood", "holdout", "heldout", "benchmark",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _compact(value: dict) -> dict:
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _read_self_hashed(path: Path, file_sha: str, content_sha: str) -> dict:
    if v43i.v40a.file_sha256(path) != file_sha:
        raise RuntimeError(f"v48b sealed input changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or v43i.v40a.canonical_sha256(_compact(value)) != content_sha
    ):
        raise RuntimeError(f"v48b sealed input content changed: {path}")
    return value


def implementation_bindings_v48b() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_base_generation_evidence_preregistration_v48b.py",
        "tests": ROOT / "test_lora_es_base_generation_evidence_v48b.py",
        "v43i_runtime": Path(v43i.__file__).resolve(),
        "v40a_runtime": Path(v43i.v40a.__file__).resolve(),
        "v40c_runtime": Path(v43i.v40c.__file__).resolve(),
        "worker": ROOT / "eggroll_es_worker_lora_v43i.py",
        "train_stage_runtime": Path(train_stage.__file__).resolve(),
        "train_stage_manifest": TRAIN_STAGE_MANIFEST,
        "train_dataset": TRAIN_DATASET,
        "membership_builder": Path(membership_builder.__file__).resolve(),
        "membership": MEMBERSHIP,
        "source_weights": v43i.SOURCE_WEIGHTS,
        "source_config": v43i.SOURCE_CONFIG,
        "source_manifest": v43i.SOURCE_MANIFEST,
        "staged_weights": v43i.STAGED_WEIGHTS,
        "staged_config": v43i.STAGED_CONFIG,
        "staged_manifest": v43i.STAGED_MANIFEST,
        "model_config": v43i.v40a.MODEL / "config.json",
        "model_index": v43i.v40a.MODEL / "model.safetensors.index.json",
        "tuned_table": v43i.v40a.TUNED_FILE,
        "base_runtime": ROOT / "train_eggroll_es_specialist.py",
        "cleanup_runtime": Path(v43i.v40a.cleanup_v38a.__file__).resolve(),
    }
    result = {label: v43i.v40a.file_sha256(path) for label, path in paths.items()}
    result["model_shards_content_sha256"] = v43i.v40a.MODEL_SHARDS_CONTENT_SHA256
    return result


def load_preregistration_v48b(args) -> dict:
    path = Path(args.preregistration).resolve()
    if v43i.v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v48b preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    recipe = value.get("recipe", {})
    runtime_paths = value.get("access_contract", {}).get(
        "only_runtime_semantic_paths_may_open", []
    )
    if (
        content != args.preregistration_content_sha256
        or v43i.v40a.canonical_sha256(_compact(value)) != content
        or value.get("schema")
        != "matched-lora-es-base-generation-evidence-preregistration-v48b"
        or value.get("status") != "preregistered_before_train_only_launch"
        or value.get("retry_revision") != "v48c_activation_retry"
        or value.get("gpu_launch_authorized") is not True
        or value.get("protected_semantic_access_authorized") is not False
        or value.get("shadow_ood_holdout_or_benchmark_authorized") is not False
        or runtime_paths != [str(TRAIN_DATASET), str(MEMBERSHIP)]
        or any(
            token in path.casefold()
            for path in runtime_paths for token in FORBIDDEN_PATH_TOKENS
        )
        or recipe.get("train_dataset") != str(TRAIN_DATASET)
        or recipe.get("train_dataset_sha256") != EXPECTED_TRAIN_SHA256
        or recipe.get("train_rows") != 448
        or recipe.get("train_conflict_units") != 208
        or recipe.get("membership") != str(MEMBERSHIP)
        or recipe.get("membership_file_sha256") != EXPECTED_MEMBERSHIP_SHA256
        or recipe.get("generation_params") != boundary.GENERATION_PARAMS_V48A
        or recipe.get("matched_master_sha256")
        != v43m.V43I_RESTORED_MASTER_SHA256
        or recipe.get("physical_gpu_ids") != [0, 1, 2, 3]
        or recipe.get("worker_extension") != v43i.WORKER_EXTENSION
        or recipe.get("adapter_slot_activation")
        != "collective_rpc.add_lora_before_canonical_install"
        or recipe.get("adapter_slot_activation_completions") != 0
        or value.get("implementation_bindings") != implementation_bindings_v48b()
    ):
        raise RuntimeError("v48b preregistration contract changed")
    return value


def load_train_inputs_v48b() -> tuple[list[dict], dict]:
    stage = _read_self_hashed(
        TRAIN_STAGE_MANIFEST,
        EXPECTED_TRAIN_STAGE_MANIFEST_SHA256,
        EXPECTED_TRAIN_STAGE_MANIFEST_CONTENT_SHA256,
    )
    membership = _read_self_hashed(
        MEMBERSHIP,
        EXPECTED_MEMBERSHIP_SHA256,
        EXPECTED_MEMBERSHIP_CONTENT_SHA256,
    )
    if (
        v43i.v40a.file_sha256(TRAIN_DATASET) != EXPECTED_TRAIN_SHA256
        or stage.get("artifact", {}).get("file_sha256") != EXPECTED_TRAIN_SHA256
        or stage.get("byte_exact") is not True
        or stage.get("nontrain_or_protected_input_opened") is not False
        or membership.get("schema")
        != "v43i-train-row-conflict-unit-membership-v48b"
        or membership.get("rows") != 448
        or membership.get("conflict_units") != 208
        or membership.get("runtime_requires_original_split_commitment") is not False
        or membership.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v48b staged train input or membership changed")
    rows = [
        json.loads(line) for line in TRAIN_DATASET.read_text(
            encoding="utf-8"
        ).splitlines() if line
    ]
    if len(rows) != 448:
        raise RuntimeError("v48b train row count changed")
    prepared = []
    for index, (row, member) in enumerate(zip(
        rows, membership["items"], strict=True,
    )):
        row_sha = hashlib.sha256(json.dumps(
            row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        pair = qa_pair_from_record(row)
        if (
            pair is None or not pair[0] or not pair[1]
            or member.get("row_index") != index
            or member.get("row_sha256") != row_sha
        ):
            raise RuntimeError("v48b staged train row identity or QA pair changed")
        prepared.append({
            "row_index": index,
            "row_sha256": row_sha,
            "unit_identity_sha256": member["unit_identity_sha256"],
            "row_count": member["row_count"],
            "question": pair[0],
            "answer": pair[1],
        })
    return prepared, membership


def score_actor_outputs_v48b(rows: list[dict], outputs: list, actor_rank: int) -> list[dict]:
    if len(rows) != 448 or len(outputs) != 448 or actor_rank not in range(4):
        raise RuntimeError("v48b actor generation coverage changed")
    result = []
    for row, output in zip(rows, outputs, strict=True):
        generated = getattr(output, "outputs", None)
        if not isinstance(generated, list) or len(generated) != 1:
            raise RuntimeError("v48b completion multiplicity changed")
        prediction = v43i.fused._extract_answer(str(generated[0].text))
        f1 = v43i.fused._f1(prediction, row["answer"])
        expected_tokens = v43i.fused._tokens(row["answer"])
        exact = int(
            bool(expected_tokens)
            and v43i.fused._tokens(prediction) == expected_tokens
        )
        result.append({
            "actor_rank": actor_rank,
            "prediction_sha256": hashlib.sha256(
                prediction.encode("utf-8")
            ).hexdigest(),
            "f1": f1,
            "exact": exact,
            "nonzero": int(f1 > 0.0),
        })
    return result


def build_evidence_v48b(
    rows: list[dict], membership: dict, actor_metrics: list[list[dict]],
    master: dict,
) -> dict:
    if (
        len(actor_metrics) != 4
        or any(len(items) != 448 for items in actor_metrics)
        or master.get("sha256") != v43m.V43I_RESTORED_MASTER_SHA256
    ):
        raise RuntimeError("v48b evidence coverage or matched master changed")
    evidence_rows = [{
        "row_index": index,
        "row_sha256": row["row_sha256"],
        "unit_identity_sha256": row["unit_identity_sha256"],
        "row_count": row["row_count"],
        "actors": [actor_metrics[actor][index] for actor in range(4)],
    } for index, row in enumerate(rows)]
    result = {
        "schema": "train-only-four-actor-base-generation-evidence-v48a",
        "revision": "v48b",
        "status": "complete_before_population",
        "train_bundle_content_sha256": v43i.TRAIN_BUNDLE_SHA256,
        "train_dataset_file_sha256": EXPECTED_TRAIN_SHA256,
        "membership_file_sha256": EXPECTED_MEMBERSHIP_SHA256,
        "membership_content_sha256": EXPECTED_MEMBERSHIP_CONTENT_SHA256,
        "membership_ordered_sha256": membership["ordered_membership_sha256"],
        "matched_master_sha256": master["sha256"],
        "generation_params": dict(boundary.GENERATION_PARAMS_V48A),
        "rows": evidence_rows,
        "row_count": 448,
        "actor_count": 4,
        "raw_question_answer_or_generation_text_persisted": False,
        "selection_or_population_opened": False,
        "protected_semantics_opened": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    result["content_sha256_before_self_field"] = v43i.v40a.canonical_sha256(result)
    return result


def _make_trainer_v48b(prereg: dict):
    saved = (
        v43i.v40a.EXPERIMENT, v43i.v40a.RUN_DIR,
        v43i.v40a.WORKER_EXTENSION,
    )
    v43i.v40a.EXPERIMENT = EXPERIMENT
    v43i.v40a.RUN_DIR = RUN_DIR
    v43i.v40a.WORKER_EXTENSION = v43i.WORKER_EXTENSION
    try:
        return v43i.v40c.make_trainer_v40c(prereg), saved
    except BaseException:
        (
            v43i.v40a.EXPERIMENT, v43i.v40a.RUN_DIR,
            v43i.v40a.WORKER_EXTENSION,
        ) = saved
        raise


def activate_adapter_slots_v48c(trainer) -> dict:
    """Populate vLLM's sole LoRA slot on all actors without generating text."""
    request = v43i._lora_request()
    results = v43i.v40a._rpc_all(trainer, "add_lora", (request,))
    if results != [True, True, True, True]:
        raise RuntimeError("v48c four-actor LoRA slot activation failed")
    return {
        "method": "collective_rpc.add_lora",
        "adapter_id": 1,
        "actors": 4,
        "all_actors_added": True,
        "generation_completions": 0,
    }


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v48b(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "train_semantics_loaded": False,
            "model_or_gpu_loaded": False,
            "protected_semantic_access_count": 0,
            "shadow_ood_holdout_or_benchmark_opened": False,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists() or REPORT.exists():
        raise RuntimeError("v48b base evidence requires fresh artifact paths")
    preflight = v43i.v40a.gpu_preflight()
    attempt = v43i.v40a.self_hashed({
        "schema": "matched-lora-es-base-generation-evidence-attempt-v48b",
        "status": "launching",
        "phase": "before_train_semantics_model_or_gpu_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "protected_semantics_opened": False,
    })
    v43i.v40a.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v43i.v40a.Phase()
    started = time.monotonic()
    try:
        rows, membership = load_train_inputs_v48b()
        prompts = [
            v43i.v40a.base.specialist_template(row["question"]) for row in rows
        ]
        v43i.v40a.base.set_seed(v43i.GLOBAL_SEED)
        trainer, saved = _make_trainer_v48b(prereg)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        pid_map = v43i.prior._actor_pid_map(actor_ids)
        monitor = threading.Thread(
            target=v43i.v40a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
        )
        monitor.start()
        phase.value = "activate_matched_initialization_lora_slot"
        activation = activate_adapter_slots_v48c(trainer)
        phase.value = "install_exact_matched_master"
        installs = v43i.v40a._rpc_all(trainer, "install_adapter_state_v41a", (
            str(v43i.SOURCE_WEIGHTS), str(v43i.SOURCE_CONFIG),
            v43i.v40a.file_sha256(v43i.SOURCE_WEIGHTS),
            v43i.v40a.file_sha256(v43i.SOURCE_CONFIG),
        ))
        certificates = v43i.v40a._rpc_all(
            trainer, "adapter_state_certificate_v41a"
        )
        masters = [item["current_identity"] for item in certificates]
        if (
            len({v43i.v40a.canonical_sha256(item) for item in masters}) != 1
            or masters[0].get("sha256") != v43m.V43I_RESTORED_MASTER_SHA256
            or len({
                item["materialization"]["runtime_values_sha256"]
                for item in certificates
            }) != 1
        ):
            raise RuntimeError("v48b matched master differs across actors")
        phase.value = "matched_base_greedy_generation_all_448_train_rows"
        batches = trainer._resolve([
            engine.generate.remote(
                prompts,
                v43i._generation_sampling_params(),
                use_tqdm=False,
                lora_request=v43i._lora_request(),
            ) for engine in trainer.engines
        ])
        if len(batches) != 4 or any(len(batch) != 448 for batch in batches):
            raise RuntimeError("v48b four-actor greedy evidence coverage changed")
        actor_metrics = [
            score_actor_outputs_v48b(rows, batch, actor)
            for actor, batch in enumerate(batches)
        ]
        evidence = build_evidence_v48b(
            rows, membership, actor_metrics, masters[0]
        )
        # Prove the evidence can deterministically construct a valid subset
        # before persisting it, but leave sealing to the separate V48B phase.
        selector_bundle = {
            "row_sha256": [row["row_sha256"] for row in rows],
            "unit_membership_v48a": [{
                "row_sha256": row["row_sha256"],
                "unit_identity_sha256": row["unit_identity_sha256"],
                "row_count": row["row_count"],
            } for row in rows],
            "train_bundle_content_sha256": v43i.TRAIN_BUNDLE_SHA256,
        }
        subset_preview = boundary.build_fragile_subset_v48a(
            selector_bundle, evidence
        )
        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("v48b GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v43i.v40a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = v43i.v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        import ray
        ray.shutdown()
        idle = v43i.v40a.cleanup_v38a.wait_for_gpu_idle()
        v43i.v40a.atomic_json(EVIDENCE, evidence)
        report = v43i.v40a.self_hashed({
            "schema": "matched-lora-es-base-generation-evidence-report-v48b",
            "status": "complete_train_only_evidence_sealed",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": args.preregistration_content_sha256,
            "train_dataset_file_sha256": EXPECTED_TRAIN_SHA256,
            "membership_file_sha256": EXPECTED_MEMBERSHIP_SHA256,
            "retry_revision": "v48c_activation_retry",
            "adapter_slot_activation": activation,
            "matched_master": masters[0],
            "installations": installs,
            "actor_identities": actor_ids,
            "evidence": {
                "path": str(EVIDENCE),
                "file_sha256": v43i.v40a.file_sha256(EVIDENCE),
                "content_sha256": evidence["content_sha256_before_self_field"],
                "rows": 448, "actors": 4,
            },
            "deterministic_subset_preview": {
                "selected_rows": subset_preview["selected_rows"],
                "selected_conflict_units": subset_preview[
                    "selected_conflict_units"
                ],
                "request_order_sha256": subset_preview[
                    "request_order_sha256"
                ],
                "subset_content_sha256": subset_preview[
                    "content_sha256_before_self_field"
                ],
                "persisted": False,
            },
            "gpu_activity": gpu,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "gpu_log_file_sha256": v43i.v40a.file_sha256(GPU_LOG),
            "raw_question_answer_or_generation_text_persisted": False,
            "selection_or_population_opened": False,
            "protected_semantics_opened": False,
            "shadow_ood_holdout_or_benchmark_opened": False,
        })
        v43i.v40a.atomic_json(REPORT, report)
        print(json.dumps({
            "report": str(REPORT),
            "report_sha256": v43i.v40a.file_sha256(REPORT),
            "report_content_sha256": report["content_sha256_before_self_field"],
            "evidence": str(EVIDENCE),
            "evidence_sha256": v43i.v40a.file_sha256(EVIDENCE),
            "evidence_content_sha256": evidence[
                "content_sha256_before_self_field"
            ],
            "all_four_gpus_attributed_positive": True,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        v43i.v40a.atomic_json(FAILURE, v43i.v40a.self_hashed({
            "schema": "matched-lora-es-base-generation-evidence-failure-v48b",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "raw_question_answer_or_generation_text_persisted": False,
            "protected_semantics_opened": False,
            "shadow_ood_holdout_or_benchmark_opened": False,
        }))
        raise
    finally:
        if trainer is not None:
            try:
                v43i.v40a.base.close_trainer(trainer)
            except Exception:
                pass
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass
        if saved is not None:
            (
                v43i.v40a.EXPERIMENT, v43i.v40a.RUN_DIR,
                v43i.v40a.WORKER_EXTENSION,
            ) = saved


if __name__ == "__main__":
    raise SystemExit(main())
