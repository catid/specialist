#!/usr/bin/env python3
"""Sealed train-only backtracking of the frozen V48B LoRA-ES direction."""

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

import lora_es_generation_boundary_reprojection_v48e as planning
import run_lora_es_generation_boundary_v48b as v48b


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v48e_generation_boundary_reprojection_backtracking"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "generation_boundary_reprojection_backtracking_report_v48e.json").resolve()
FAILURE = (RUN_DIR / "failure_v48e.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v48e.jsonl").resolve()
SNAPSHOT = (RUN_DIR / "adapter_backtracked_v48e").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_generation_boundary_reprojection_backtracking_v48e.json"
).resolve()
V48B_RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v48b_matched_lora_es_generation_boundary_pop8"
).resolve()
V48B_PREREGISTRATION = v48b.PREREGISTRATION
SUBSET = v48b.SUBSET
V48B_EVIDENCE_PATHS = {
    "population_reliability": V48B_RUN / "population_reliability_v48b.json",
    "candidate_gate": V48B_RUN / "candidate_gate_v48b.json",
    "exact_abort": V48B_RUN / "exact_abort_v48b.json",
    "failure": V48B_RUN / "failure_v43i.json",
    "numeric_calibration": V48B_RUN / "numeric_calibration_v48b.json",
    "anchor_calibration": V48B_RUN / "anchor_calibration_v48b.json",
}
SCALE_LABELS_V48E = {
    0.5: "ratio_0p5",
    0.25: "ratio_0p25",
    0.125: "ratio_0p125",
    0.0625: "ratio_0p0625",
    0.03125: "ratio_0p03125",
    0.015625: "ratio_0p015625",
}

prior = v48b.v43i
v40a = prior.v40a
numeric = prior.numeric


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _artifact_paths_v48e() -> dict[str, Path]:
    result = {}
    for _ratio, label in SCALE_LABELS_V48E.items():
        result[f"candidate_gate_{label}"] = (
            RUN_DIR / f"candidate_gate_{label}_v48e.json"
        )
        result[f"exact_abort_{label}"] = (
            RUN_DIR / f"exact_abort_{label}_v48e.json"
        )
        result[f"candidate_consensus_{label}"] = (
            RUN_DIR / f"candidate_consensus_{label}_v48e.json"
        )
    return {key: value.resolve() for key, value in result.items()}


SCALE_ARTIFACTS = _artifact_paths_v48e()


def implementation_bindings_v48e() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_generation_boundary_reprojection_backtracking_v48e.py",
        "planning": Path(planning.__file__).resolve(),
        "tests": ROOT / "test_lora_es_generation_boundary_reprojection_backtracking_v48e.py",
        "v48b_runtime": Path(v48b.__file__).resolve(),
        "v48a_boundary": Path(v48b.boundary.__file__).resolve(),
        "v43i_runtime": Path(prior.__file__).resolve(),
        "worker": ROOT / "eggroll_es_worker_lora_v43i.py",
        "train_dataset": v48b.evidence_runtime.TRAIN_DATASET,
        "train_membership": v48b.evidence_runtime.MEMBERSHIP,
        "subset": SUBSET,
        "v48b_preregistration": V48B_PREREGISTRATION,
        "source_weights": prior.SOURCE_WEIGHTS,
        "source_config": prior.SOURCE_CONFIG,
        "prose_anchor": prior.PROSE_ANCHOR,
        "prose_report": prior.PROSE_REPORT,
        "qa_anchor": prior.QA_ANCHOR,
        "qa_report": prior.QA_REPORT,
        "model_config": v40a.MODEL / "config.json",
        "model_index": v40a.MODEL / "model.safetensors.index.json",
        "tuned_table": v40a.TUNED_FILE,
    }
    return {
        "files": {key: v40a.file_sha256(path) for key, path in paths.items()},
        "v48b_evidence_files": {
            label: v40a.file_sha256(path)
            for label, path in V48B_EVIDENCE_PATHS.items()
        },
        "model_shards_content_sha256": v40a.MODEL_SHARDS_CONTENT_SHA256,
    }


def load_evidence_v48e() -> dict:
    return planning.load_v48e_design(
        V48B_EVIDENCE_PATHS, SUBSET, V48B_PREREGISTRATION,
    )


def load_preregistration_v48e(args) -> tuple[dict, dict]:
    path = Path(args.preregistration).resolve()
    if v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v48e preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    evidence = load_evidence_v48e()
    recipe = value.get("recipe", {})
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or v40a.canonical_sha256(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "matched-lora-es-generation-boundary-reprojection-backtracking-preregistration-v48e"
        or value.get("status") != "preregistered_before_train_only_launch"
        or value.get("gpu_launch_authorized") is not True
        or value.get("protected_semantic_access_authorized") is not False
        or value.get("shadow_ood_holdout_or_benchmark_authorized") is not False
        or value.get("v48b_evidence") != evidence
        or value.get("implementation_bindings") != implementation_bindings_v48e()
        or recipe.get("seeds") != prior.SEEDS
        or recipe.get("alpha") != prior.ALPHA
        or recipe.get("scale_plans") != evidence["scale_plans"]
        or recipe.get("scale_order")
        != list(planning.TARGET_NORM_RATIOS_V48E)
        or recipe.get("resample_population") is not False
        or recipe.get("recompute_population_scores") is not False
        or recipe.get("recompute_projection_from_frozen_signed_scores")
        is not True
        or recipe.get("new_population_generation") is not False
        or recipe.get("request_order_sha256")
        != planning.REQUEST_ORDER_SHA256_V48E
    ):
        raise RuntimeError("v48e preregistration contract changed")
    return value, evidence


@contextmanager
def patched_prior_paths_v48e():
    values = {
        "EXPERIMENT": EXPERIMENT,
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "REPORT": REPORT,
        "GPU_LOG": GPU_LOG,
        "SNAPSHOT": SNAPSHOT,
    }
    saved = {key: getattr(prior, key) for key in values}
    for key, value in values.items():
        setattr(prior, key, value)
    try:
        yield
    finally:
        for key, value in saved.items():
            setattr(prior, key, value)


def _persist_v48e(path: Path, value: dict) -> dict:
    return prior._persist_phase(path, value)


def _candidate_consensus_v48e(
    trainer, bundle, dense_items, requests, evidence: dict, ratio: float,
) -> dict:
    records = prior._score_repeats(
        trainer, bundle, dense_items, requests,
        warmups=0, retained=numeric.POST_UPDATE_REPEATS_V43G,
    )
    result = numeric.post_update_consensus_v43g(
        records, evidence["numeric_calibration_bootstrap_bounds"],
    )
    label = SCALE_LABELS_V48E[ratio]
    return _persist_v48e(
        SCALE_ARTIFACTS[f"candidate_consensus_{label}"], {
            "schema": "generation-boundary-candidate-consensus-v48e",
            "status": "complete_while_candidate_uncommitted",
            "target_norm_ratio": ratio,
            "records": records,
            "equivalence": result,
            "source_numeric_calibration_content_sha256": (
                planning.EXPECTED_FILES_V48E["numeric_calibration"][1]
            ),
            "protected_semantics_opened": False,
            "shadow_ood_holdout_or_benchmark_opened": False,
        },
    )


def _finish_v48e(trainer, monitor, stop, failures, pid_map):
    stop.set()
    monitor.join(timeout=10)
    if monitor.is_alive() or not failures.empty():
        raise RuntimeError("v48e GPU monitor failed") from (
            failures.get() if not failures.empty() else None
        )
    gpu = v40a.summarize_gpu(GPU_LOG, pid_map)
    cleanup = v40a.cleanup_v38a.strict_close_trainer_v38a(trainer)
    import ray
    ray.shutdown()
    idle = v40a.cleanup_v38a.wait_for_gpu_idle()
    return gpu, cleanup, idle


def _validate_abort_v48e(aborted: dict, master: dict, runtime_sha: str) -> None:
    if (
        aborted.get("all_four_ranks_exact") is not True
        or aborted.get("restored_master_identity") != master
        or aborted.get("restored_runtime_values_sha256") != runtime_sha
        or len(aborted.get("workers", [])) != 4
        or len(aborted.get("state_certificates", [])) != 4
    ):
        raise RuntimeError("v48e rejected candidate exact abort changed")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg, evidence = load_preregistration_v48e(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "v48b_numeric_evidence_loaded": True,
            "population_resampled": False,
            "population_scores_recomputed": False,
            "projection_recomputed_from_frozen_signed_scores": True,
            "new_population_generation": False,
            "scale_order": list(planning.TARGET_NORM_RATIOS_V48E),
            "train_semantics_loaded": False,
            "model_or_gpu_loaded": False,
            "protected_semantic_access_count": 0,
            "shadow_ood_holdout_or_benchmark_opened": False,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v48e requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "generation-boundary-reprojection-backtracking-attempt-v48e",
        "status": "launching",
        "phase": "before_train_semantics_model_or_gpu_load",
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
        sealed_subset = v48b.load_subset_v48b(
            SUBSET, planning.EXPECTED_SUBSET_FILE_SHA256_V48E,
            planning.EXPECTED_SUBSET_CONTENT_SHA256_V48E,
        )
        v48b._SEALED_SUBSET = sealed_subset
        with v48b.patched_v43i_v48b(), patched_prior_paths_v48e():
            bundle = v48b.load_train_bundle_v48b()
            if bundle["content_sha256_before_self_field"] != (
                v48b.EXPECTED_TRAIN_BUNDLE_CONTENT_SHA256_V48B
            ):
                raise RuntimeError("v48e frozen train bundle changed")
            bundle = v48b.augment_unit_membership_v48b(bundle)
            anchor_bundle = prior.fused.load_anchor_bundle_v43i(
                prior.PROSE_ANCHOR, prior.PROSE_REPORT,
                prior.QA_ANCHOR, prior.QA_REPORT,
            )
            v40a.base.set_seed(prior.GLOBAL_SEED)
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
            phase.value = "verify_v48b_exact_restored_initial_state"
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
                raise RuntimeError("v48e initial state differs from V48B exact abort")
            master = masters[0]
            master_runtime_sha = next(iter(runtime_hashes))
            update_sequences = {item["update_sequence"] for item in certificates}
            if update_sequences != {0}:
                raise RuntimeError("v48e restored update sequence changed")
            references = v40a._rpc_all(
                trainer, "capture_adapter_reference_v41a",
            )
            reference_generations = {
                item["reference_generation"] for item in references
            }
            if len(reference_generations) != 1:
                raise RuntimeError("v48e reference generation differs across ranks")
            postinstall = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            if preinstall["consensus"] != postinstall["consensus"]:
                raise RuntimeError("v48e canonical install changed initial score")
            full_plan = prior.fused.fused_requests_v43i(requests, full_anchors)
            if len(full_plan["requests"]) != 896:
                raise RuntimeError("v48e full candidate request coverage changed")
            phase.value = "full_generation_boundary_reference_score"
            reference_actors = prior._generate_fused_actor_scores(
                trainer, bundle, dense_items, full_plan, full_anchors,
            )

            for scale_plan in evidence["scale_plans"]:
                ratio = scale_plan["target_norm_ratio"]
                label = SCALE_LABELS_V48E[ratio]
                plan_id = v40a.canonical_sha256({
                    "schema": "generation-boundary-reprojection-backtracking-plan-v48e",
                    "v48b_population_content_sha256": evidence[
                        "v48b_population_content_sha256"
                    ],
                    "v48e_projection_content_sha256": evidence[
                        "v48e_projection_content_sha256"
                    ],
                    "sealed_subset_content_sha256": evidence[
                        "sealed_subset"
                    ]["content_sha256"],
                    "fragile_request_order_sha256": evidence[
                        "sealed_subset"
                    ]["request_order_sha256"],
                    "restored_master_sha256": master["sha256"],
                    "seeds": prior.SEEDS,
                    "scale_plan": scale_plan,
                    "alpha": prior.ALPHA,
                    "preference": "largest strictly passing ratio",
                })
                planned_manifest_sha, _ = prior._expected_update_manifest_sha(
                    master, next(iter(reference_generations)),
                    next(iter(update_sequences)), scale_plan["coefficients"],
                    plan_id,
                )
                transaction_state = {"manifest_sha256": planned_manifest_sha}
                phase.value = f"execute_uncommitted_{label}"
                transaction_state = prior._prepare_execute_update(
                    trainer, master, next(iter(reference_generations)),
                    next(iter(update_sequences)), scale_plan["coefficients"],
                    plan_id,
                )
                phase.value = f"score_full_generation_boundary_{label}"
                candidate_actors = prior._generate_fused_actor_scores(
                    trainer, bundle, dense_items, full_plan, full_anchors,
                )
                gate = prior.fused.candidate_gate_v43i(
                    reference_actors, candidate_actors,
                    evidence["anchor_calibrated_margins"],
                )
                expected_checks = {
                    "domain_point_improvement", "prose_lm_noninferiority",
                    "qa_logprob_noninferiority",
                    "qa_generation_f1_noninferiority",
                    "qa_generation_exact_noninferiority",
                    "qa_generation_nonzero_noninferiority",
                    "fragile_generation_f1_noninferiority",
                    "fragile_generation_exact_noninferiority",
                    "fragile_generation_nonzero_noninferiority",
                }
                if set(gate.get("checks", {})) != expected_checks:
                    raise RuntimeError("v48e candidate gate inventory changed")
                gate_artifact = _persist_v48e(
                    SCALE_ARTIFACTS[f"candidate_gate_{label}"], {
                        "schema": "generation-boundary-candidate-gate-v48e",
                        "status": "complete_before_commit_or_abort",
                        "target_norm_ratio": ratio,
                        "scale_plan": scale_plan,
                        "candidate_identity": transaction_state[
                            "candidate_identity"
                        ],
                        "reference_actors": reference_actors,
                        "candidate_actors": candidate_actors,
                        "gate": gate,
                        "all_nine_train_only_checks_required": True,
                        "sealed_subset_content_sha256": evidence[
                            "sealed_subset"
                        ]["content_sha256"],
                        "fragile_request_order_sha256": evidence[
                            "sealed_subset"
                        ]["request_order_sha256"],
                        "protected_semantics_opened": False,
                        "shadow_ood_holdout_or_benchmark_opened": False,
                    },
                )
                consensus = None
                acceptance_passed = (
                    gate.get("passed") is True
                    and all(gate["checks"].values())
                )
                if acceptance_passed:
                    phase.value = f"uncommitted_consensus_{label}"
                    consensus = _candidate_consensus_v48e(
                        trainer, bundle, dense_items, requests, evidence, ratio,
                    )
                    acceptance_passed = (
                        consensus["equivalence"]["passed"] is True
                    )
                if not acceptance_passed:
                    phase.value = f"exact_abort_readback_{label}"
                    aborted = prior._exact_abort_transaction(
                        trainer, transaction_state["manifest_sha256"],
                        master, master_runtime_sha,
                    )
                    _validate_abort_v48e(aborted, master, master_runtime_sha)
                    abort_artifact = _persist_v48e(
                        SCALE_ARTIFACTS[f"exact_abort_{label}"], {
                            **aborted,
                            "status": "scale_rejected_and_exactly_restored",
                            "target_norm_ratio": ratio,
                            "candidate_gate_content_sha256": gate_artifact[
                                "content_sha256_before_self_field"
                            ],
                            "reason": (
                                "one_or_more_train_only_gates_failed"
                                if not gate.get("passed")
                                else "candidate_actor_consensus_failed"
                            ),
                            "protected_semantics_opened": False,
                            "shadow_ood_holdout_or_benchmark_opened": False,
                        },
                    )
                    transaction_state = None
                    scale_results.append({
                        "target_norm_ratio": ratio,
                        "gate_passed": False,
                        "preservation_gate_passed": gate.get("passed") is True,
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
                    planning.largest_passing_scale_v48e(scale_results)
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
                if planning.largest_passing_scale_v48e(scale_results) != ratio:
                    raise RuntimeError("v48e largest passing scale changed")
                break

            if accepted_ratio is None and (
                planning.largest_passing_scale_v48e(scale_results) is not None
            ):
                raise RuntimeError("v48e no-pass result changed")
            gpu, cleanup, idle = _finish_v48e(
                trainer, monitor, stop, failures, pid_map,
            )
            trainer = None

        report = v40a.self_hashed({
            "schema": "generation-boundary-reprojection-backtracking-report-v48e",
            "status": (
                "complete_largest_strictly_passing_scale_committed_and_sealed"
                if accepted_ratio is not None
                else "complete_no_scale_passed_all_candidates_exactly_restored"
            ),
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration_file_sha256": args.preregistration_sha256,
            "preregistration_content_sha256": (
                args.preregistration_content_sha256
            ),
            "v48b_evidence": evidence,
            "installations": installs,
            "initial_master_identity": master,
            "initial_runtime_values_sha256": master_runtime_sha,
            "initial_references": references,
            "initial_score_audits": {
                "preinstall": preinstall,
                "postinstall": postinstall,
                "canonical_install_preserved_score": True,
            },
            "scale_results": scale_results,
            "accepted_target_norm_ratio": accepted_ratio,
            "largest_strictly_passing_preference_enforced": True,
            "population_resampled": False,
            "population_scores_recomputed": False,
            "projection_recomputed_from_frozen_signed_scores": True,
            "new_population_generation": False,
            "update": update,
            "snapshot": str(SNAPSHOT) if accepted_ratio is not None else None,
            "gpu_activity": gpu,
            "cleanup": cleanup,
            "final_gpu_idle": idle,
            "gpu_log": {
                "path": str(GPU_LOG),
                "file_sha256": v40a.file_sha256(GPU_LOG),
            },
            "protected_semantics_opened": False,
            "shadow_ood_holdout_or_benchmark_opened": False,
        })
        v40a.atomic_json(REPORT, report)
        print(json.dumps({
            "report": str(REPORT),
            "report_sha256": v40a.file_sha256(REPORT),
            "report_content_sha256": report[
                "content_sha256_before_self_field"
            ],
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
            "schema": "generation-boundary-reprojection-backtracking-failure-v48e",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "scale_results": scale_results,
            "emergency_exact_abort": emergency_abort,
            "emergency_exact_abort_failure": emergency_abort_failure,
            "transaction_accepted_before_failure": transaction_accepted,
            "protected_semantics_opened": False,
            "shadow_ood_holdout_or_benchmark_opened": False,
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
