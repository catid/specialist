#!/usr/bin/env python3
"""Deterministic train-only backtracking recovery from sealed V43I evidence."""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import lora_es_backtracking_v43j as backtracking
import run_lora_es_multi_anchor_v43i as prior


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v43j_lora_es_v43i_projection_backtracking"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "matched_lora_es_backtracking_report_v43j.json").resolve()
FAILURE = (RUN_DIR / "failure_v43j.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v43j.jsonl").resolve()
SNAPSHOT = (RUN_DIR / "adapter_backtracked_v43j").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_projection_backtracking_v43j.json"
).resolve()
V43I_RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43i_matched_lora_es_fold3_pop8_multi_anchor"
).resolve()
V43I_EVIDENCE_PATHS = {
    "population_reliability": V43I_RUN / "population_reliability_v43i.json",
    "candidate_gate": V43I_RUN / "candidate_gate_v43i.json",
    "exact_abort": V43I_RUN / "exact_abort_v43i.json",
    "failure": V43I_RUN / "failure_v43i.json",
    "numeric_calibration": V43I_RUN / "numeric_calibration_v43i.json",
    "anchor_calibration": V43I_RUN / "anchor_calibration_v43i.json",
}
SCALE_LABELS_V43J = {
    0.25: "ratio_0p25", 0.125: "ratio_0p125", 0.0625: "ratio_0p0625",
}

v40a = prior.v40a
numeric = prior.numeric
fused = prior.fused
equal_v38 = prior.equal_v38


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _artifact_paths() -> dict[str, Path]:
    result = {}
    for ratio, label in SCALE_LABELS_V43J.items():
        result[f"candidate_gate_{label}"] = RUN_DIR / f"candidate_gate_{label}_v43j.json"
        result[f"exact_abort_{label}"] = RUN_DIR / f"exact_abort_{label}_v43j.json"
        result[f"candidate_consensus_{label}"] = (
            RUN_DIR / f"candidate_consensus_{label}_v43j.json"
        )
    return {key: value.resolve() for key, value in result.items()}


SCALE_ARTIFACTS = _artifact_paths()


def implementation_bindings_v43j() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_backtracking_preregistration_v43j.py",
        "backtracking_runtime": Path(backtracking.__file__).resolve(),
        "tests": ROOT / "test_lora_es_backtracking_v43j.py",
        "v43i_runtime": Path(prior.__file__).resolve(),
        "v43i_worker": ROOT / "eggroll_es_worker_lora_v43i.py",
        "v43i_fused_anchor_runtime": Path(fused.__file__).resolve(),
    }
    return {
        "files": {key: v40a.file_sha256(path) for key, path in paths.items()},
        "v43i_implementation_bindings": prior.implementation_bindings(),
        "v43i_evidence_files": {
            label: v40a.file_sha256(path)
            for label, path in V43I_EVIDENCE_PATHS.items()
        },
        "model_shards_content_sha256": v40a.MODEL_SHARDS_CONTENT_SHA256,
    }


def load_preregistration_v43j(args) -> tuple[dict, dict]:
    path = Path(args.preregistration).resolve()
    if v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v43j preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    evidence = backtracking.load_v43i_evidence_v43j(V43I_EVIDENCE_PATHS)
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or v40a.canonical_sha256(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "matched-lora-es-backtracking-preregistration-v43j"
        or value.get("status") != "preregistered_before_train_only_launch"
        or value.get("gpu_launch_authorized") is not True
        or value.get("sealed_holdout_opened") is not False
        or value.get("shadow_dev_eval_ood_or_holdout_authorized") is not False
        or value.get("protected_semantic_access") is not False
        or value.get("v43i_evidence") != evidence
        or value.get("implementation_bindings") != implementation_bindings_v43j()
        or value.get("recipe", {}).get("seeds") != prior.SEEDS
        or value.get("recipe", {}).get("alpha") != prior.ALPHA
        or value.get("recipe", {}).get("scale_plans") != evidence["scale_plans"]
        or value.get("recipe", {}).get("scale_order")
        != list(backtracking.TARGET_NORM_RATIOS_V43J)
        or value.get("recipe", {}).get("resample_population") is not False
        or value.get("recipe", {}).get("recompute_projection") is not False
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    ):
        raise RuntimeError("v43j preregistration contract changed")
    return value, evidence


@contextmanager
def patched_prior_paths_v43j():
    names = {
        "EXPERIMENT": EXPERIMENT,
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "REPORT": REPORT,
        "GPU_LOG": GPU_LOG,
        "SNAPSHOT": SNAPSHOT,
    }
    saved = {key: getattr(prior, key) for key in names}
    for key, value in names.items():
        setattr(prior, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(prior, key, value)


def _persist(path: Path, value: dict) -> dict:
    return prior._persist_phase(path, value)


def _candidate_consensus_v43j(
    trainer, bundle, dense_items, requests, evidence: dict, ratio: float,
) -> dict:
    records = prior._score_repeats(
        trainer, bundle, dense_items, requests,
        warmups=0, retained=numeric.POST_UPDATE_REPEATS_V43G,
    )
    result = numeric.post_update_consensus_v43g(
        records, evidence["numeric_calibration_bootstrap_bounds"],
    )
    label = SCALE_LABELS_V43J[ratio]
    return _persist(SCALE_ARTIFACTS[f"candidate_consensus_{label}"], {
        "schema": "matched-lora-es-backtracked-candidate-consensus-v43j",
        "status": "complete_while_candidate_uncommitted",
        "target_norm_ratio": ratio,
        "records": records,
        "equivalence": result,
        "v43i_numeric_calibration_content_sha256": (
            backtracking.EXPECTED_FILES_V43J["numeric_calibration"][1]
        ),
        "protected_semantics_opened": False,
    })


def _finish_runtime(trainer, monitor, stop, failures, pid_map):
    stop.set()
    monitor.join(timeout=10)
    if monitor.is_alive() or not failures.empty():
        raise RuntimeError("v43j GPU monitor failed") from (
            failures.get() if not failures.empty() else None
        )
    gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
    cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
    import ray
    ray.shutdown()
    idle = v40a.cleanup_v38a.wait_for_gpu_idle()
    return gpu, cleanup, idle


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg, evidence = load_preregistration_v43j(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "v43i_evidence_loaded": True,
            "population_resampled": False,
            "projection_recomputed": False,
            "scale_order": list(backtracking.TARGET_NORM_RATIOS_V43J),
            "model_runtime_loaded": False,
            "gpu_launched": False,
            "protected_paths_opened": [],
            "protected_semantic_access": False,
            "sealed_holdout_opened": False,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v43j requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "matched-lora-es-backtracking-attempt-v43j",
        "status": "launching",
        "phase": "before_model_or_train_data_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "protected_semantics_opened": False,
    })
    v40a.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = None
    transaction_state = None
    transaction_accepted = False
    master = master_runtime_sha = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = v40a.Phase()
    started = time.monotonic()
    scale_results = []
    accepted_ratio = None
    update = None
    try:
        bundle = equal_v38.load_equal_unit_train_bundle(
            prior.DATASET, prior.DATASET_SHA256,
            prior.SPLIT_MANIFEST, prior.SPLIT_MANIFEST_SHA256,
        )
        if bundle["content_sha256_before_self_field"] != prior.TRAIN_BUNDLE_SHA256:
            raise RuntimeError("v43j frozen train bundle identity changed")
        bundle = prior.augment_unit_membership_v43i(bundle)
        anchor_bundle = fused.load_anchor_bundle_v43i(
            prior.PROSE_ANCHOR, prior.PROSE_REPORT,
            prior.QA_ANCHOR, prior.QA_REPORT,
        )
        v40a.base.set_seed(prior.GLOBAL_SEED)
        with patched_prior_paths_v43j():
            trainer, saved = prior._make_trainer(prereg)
            actor_ids = trainer._resolve([
                engine.runtime_identity_v40a.remote() for engine in trainer.engines
            ])
            pid_map = prior.prior._actor_pid_map(actor_ids)
            monitor = threading.Thread(
                target=v40a.monitor_gpus,
                args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
            )
            monitor.start()
            dense_items, requests, _panel, full_anchors = prior._prepare(
                trainer, bundle, anchor_bundle,
            )
            phase.value = "verify_v43i_restored_initial_state"
            preinstall = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            installs = v40a._rpc_all(trainer, "install_adapter_state_v41a", (
                str(prior.SOURCE_WEIGHTS), str(prior.SOURCE_CONFIG),
                v40a.file_sha256(prior.SOURCE_WEIGHTS),
                v40a.file_sha256(prior.SOURCE_CONFIG),
            ))
            certificates = v40a._rpc_all(
                trainer, "adapter_state_certificate_v41a",
            )
            masters = [item["current_identity"] for item in certificates]
            runtime_hashes = {
                item["materialization"]["runtime_values_sha256"]
                for item in certificates
            }
            if (
                any(item != evidence["restored_master_identity"] for item in masters)
                or runtime_hashes != {evidence["restored_runtime_values_sha256"]}
            ):
                raise RuntimeError("v43j initial state differs from V43I exact restore")
            master = masters[0]
            master_runtime_sha = next(iter(runtime_hashes))
            update_sequences = {item["update_sequence"] for item in certificates}
            if update_sequences != {0}:
                raise RuntimeError("v43j restored update sequence changed")
            references = v40a._rpc_all(
                trainer, "capture_adapter_reference_v41a",
            )
            reference_generations = {
                item["reference_generation"] for item in references
            }
            if len(reference_generations) != 1:
                raise RuntimeError("v43j reference generation differs across ranks")
            postinstall = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            if preinstall["consensus"] != postinstall["consensus"]:
                raise RuntimeError("v43j canonical install changed initial score")
            full_plan = fused.fused_requests_v43i(requests, full_anchors)
            phase.value = "full_anchor_reference_score"
            reference_actors = prior._generate_fused_actor_scores(
                trainer, bundle, dense_items, full_plan, full_anchors,
            )

            for scale_plan in evidence["scale_plans"]:
                ratio = scale_plan["target_norm_ratio"]
                label = SCALE_LABELS_V43J[ratio]
                plan_id = v40a.canonical_sha256({
                    "schema": "matched-lora-es-backtracking-plan-v43j",
                    "v43i_population_reliability_content_sha256": (
                        backtracking.EXPECTED_FILES_V43J[
                            "population_reliability"
                        ][1]
                    ),
                    "v43i_projection_content_sha256": evidence[
                        "v43i_projection_content_sha256"
                    ],
                    "restored_master_sha256": master["sha256"],
                    "seeds": prior.SEEDS,
                    "scale_plan": scale_plan,
                    "alpha": prior.ALPHA,
                    "preference": "largest passing scale in descending order",
                })
                planned_manifest_sha, _ = prior._expected_update_manifest_sha(
                    master, next(iter(reference_generations)),
                    next(iter(update_sequences)), scale_plan["coefficients"], plan_id,
                )
                transaction_state = {"manifest_sha256": planned_manifest_sha}
                phase.value = f"execute_uncommitted_{label}"
                transaction_state = prior._prepare_execute_update(
                    trainer, master, next(iter(reference_generations)),
                    next(iter(update_sequences)), scale_plan["coefficients"], plan_id,
                )
                phase.value = f"score_full_domain_prose_qa_greedy_{label}"
                candidate_actors = prior._generate_fused_actor_scores(
                    trainer, bundle, dense_items, full_plan, full_anchors,
                )
                gate = fused.candidate_gate_v43i(
                    reference_actors, candidate_actors,
                    evidence["anchor_calibrated_margins"],
                )
                gate_artifact = _persist(
                    SCALE_ARTIFACTS[f"candidate_gate_{label}"], {
                        "schema": "matched-lora-es-backtracked-candidate-gate-v43j",
                        "status": "complete_before_commit_or_abort",
                        "target_norm_ratio": ratio,
                        "scale_plan": scale_plan,
                        "candidate_identity": transaction_state["candidate_identity"],
                        "reference_actors": reference_actors,
                        "candidate_actors": candidate_actors,
                        "gate": gate,
                        "protected_semantics_opened": False,
                    },
                )
                consensus = None
                acceptance_passed = gate["passed"]
                if acceptance_passed:
                    phase.value = f"uncommitted_consensus_{label}"
                    consensus = _candidate_consensus_v43j(
                        trainer, bundle, dense_items, requests, evidence, ratio,
                    )
                    acceptance_passed = consensus["equivalence"]["passed"]
                if not acceptance_passed:
                    phase.value = f"exact_abort_readback_{label}"
                    aborted = prior._exact_abort_transaction(
                        trainer, transaction_state["manifest_sha256"],
                        master, master_runtime_sha,
                    )
                    abort_artifact = _persist(
                        SCALE_ARTIFACTS[f"exact_abort_{label}"], {
                            **aborted,
                            "status": "scale_rejected_and_exactly_restored",
                            "target_norm_ratio": ratio,
                            "candidate_gate_content_sha256": gate_artifact[
                                "content_sha256_before_self_field"
                            ],
                            "reason": (
                                "candidate preservation gate failed"
                                if not gate["passed"]
                                else "candidate actor consensus failed"
                            ),
                            "protected_semantics_opened": False,
                        },
                    )
                    transaction_state = None
                    scale_results.append({
                        "target_norm_ratio": ratio,
                        "gate_passed": False,
                        "preservation_gate_passed": gate["passed"],
                        "candidate_gate_content_sha256": gate_artifact[
                            "content_sha256_before_self_field"
                        ],
                        "candidate_consensus_passed": (
                            None if consensus is None
                            else consensus["equivalence"]["passed"]
                        ),
                        "exact_abort_readback_passed": True,
                        "exact_abort_content_sha256": abort_artifact[
                            "content_sha256_before_self_field"
                        ],
                    })
                    backtracking.largest_passing_scale_v43j(scale_results)
                    continue

                phase.value = f"commit_largest_passing_{label}"
                update = prior._commit_accept_update(trainer, transaction_state)
                phase.value = f"seal_snapshot_{label}"
                update = prior._seal_accepted_update(trainer, update)
                transaction_accepted = True
                transaction_state = None
                accepted_ratio = ratio
                scale_results.append({
                    "target_norm_ratio": ratio,
                    "gate_passed": True,
                    "preservation_gate_passed": True,
                    "candidate_gate_content_sha256": gate_artifact[
                        "content_sha256_before_self_field"
                    ],
                    "candidate_consensus_passed": True,
                    "candidate_consensus_content_sha256": consensus[
                        "content_sha256_before_self_field"
                    ],
                    "exact_abort_readback_passed": False,
                })
                if backtracking.largest_passing_scale_v43j(scale_results) != ratio:
                    raise RuntimeError("v43j largest passing scale selection changed")
                break

            if accepted_ratio is None:
                if backtracking.largest_passing_scale_v43j(scale_results) is not None:
                    raise RuntimeError("v43j no-pass backtracking result changed")
            gpu, cleanup, idle = _finish_runtime(
                trainer, monitor, stop, failures, pid_map,
            )
            trainer = None

        report = v40a.self_hashed({
            "schema": "matched-lora-es-backtracking-train-only-report-v43j",
            "status": (
                "complete_largest_passing_scale_committed_and_sealed"
                if accepted_ratio is not None
                else "complete_no_scale_passed_all_candidates_exactly_restored"
            ),
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "v43i_evidence": evidence,
            "installations": installs,
            "initial_master_identity": master,
            "initial_runtime_values_sha256": master_runtime_sha,
            "initial_references": references,
            "initial_score_audits": {
                "preinstall": preinstall, "postinstall": postinstall,
                "canonical_install_preserved_score": True,
            },
            "scale_results": scale_results,
            "accepted_target_norm_ratio": accepted_ratio,
            "largest_passing_preference_enforced": True,
            "population_resampled": False,
            "projection_recomputed": False,
            "update": update,
            "snapshot": str(SNAPSHOT) if accepted_ratio is not None else None,
            "gpu_activity": gpu,
            "placement_group_cleanup": cleanup,
            "final_gpu_idle": idle,
            "preflight": preflight,
            "gpu_log": {
                "path": str(GPU_LOG), "file_sha256": v40a.file_sha256(GPU_LOG),
            },
            "shadow_dev_eval_ood_or_holdout_opened": False,
            "protected_semantics_opened": False,
            "quality_or_promotion_conclusion_authorized": False,
        })
        v40a.atomic_json(REPORT, report)
        print(json.dumps({
            "report": str(REPORT),
            "report_sha256": v40a.file_sha256(REPORT),
            "accepted_target_norm_ratio": accepted_ratio,
            "snapshot": str(SNAPSHOT) if accepted_ratio is not None else None,
            "all_failed_scales_exactly_restored": all(
                item["gate_passed"] or item["exact_abort_readback_passed"]
                for item in scale_results
            ),
        }, sort_keys=True))
        return 0
    except BaseException as error:
        emergency_abort = emergency_abort_failure = None
        if (
            trainer is not None and transaction_state is not None
            and not transaction_accepted and master is not None
            and master_runtime_sha is not None
        ):
            try:
                emergency_abort = prior._exact_abort_transaction(
                    trainer, transaction_state["manifest_sha256"],
                    master, master_runtime_sha,
                )
                transaction_state = None
            except BaseException as abort_error:
                emergency_abort_failure = {
                    "type": type(abort_error).__name__,
                    "message": str(abort_error),
                    "traceback": traceback.format_exc(),
                }
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        v40a.atomic_json(FAILURE, v40a.self_hashed({
            "schema": "matched-lora-es-backtracking-failure-v43j",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "scale_results": scale_results,
            "emergency_exact_abort": emergency_abort,
            "emergency_exact_abort_failure": emergency_abort_failure,
            "transaction_accepted_before_failure": transaction_accepted,
            "protected_semantics_opened": False,
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
