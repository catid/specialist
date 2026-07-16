#!/usr/bin/env python3
"""Train-only diagnostic of the untried V43I 0.125/0.0625 scales."""

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

import lora_es_backtracking_v43k as backtracking
import run_lora_es_multi_anchor_v43i as prior


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v43k_lora_es_v43i_projection_untried_backtracking"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "matched_lora_es_backtracking_report_v43k.json").resolve()
FAILURE = (RUN_DIR / "failure_v43k.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v43k.jsonl").resolve()
SNAPSHOT = (RUN_DIR / "adapter_backtracked_v43k").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_projection_backtracking_v43k.json"
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
SCALE_LABELS_V43K = {
    0.125: "ratio_0p125", 0.0625: "ratio_0p0625",
}
V43J_RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43j_lora_es_v43i_projection_backtracking"
).resolve()
V43J_REPORT = V43J_RUN / "matched_lora_es_backtracking_report_v43j.json"
V43J_GATE = V43J_RUN / "candidate_gate_ratio_0p25_v43j.json"
V43J_EXPECTED = {
    "report": "70084cdb1752b097e737572515c5c0c2f2a9b13f58ad0cb087f157c61ed7387a",
    "report_content": "61820f950f6c85ccb69f0e798f3526e4fae658aa40e7801b7c13bd5367056c60",
    "gate": "ea631461bf78299596b68d40fe2baa6fa95aced58674adf4d9d6b44e3ea2ac92",
    "gate_content": "4792016a320ff279c75034349dda1779ee88e744c05dda39c30c38a23133f320",
}
SIX_GATE_NAMES_V43K = {
    "domain_point_improvement",
    "prose_lm_noninferiority",
    "qa_generation_exact_noninferiority",
    "qa_generation_f1_noninferiority",
    "qa_generation_nonzero_noninferiority",
    "qa_logprob_noninferiority",
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
    for ratio, label in SCALE_LABELS_V43K.items():
        result[f"candidate_gate_{label}"] = RUN_DIR / f"candidate_gate_{label}_v43k.json"
        result[f"exact_abort_{label}"] = RUN_DIR / f"exact_abort_{label}_v43k.json"
        result[f"candidate_consensus_{label}"] = (
            RUN_DIR / f"candidate_consensus_{label}_v43k.json"
        )
    return {key: value.resolve() for key, value in result.items()}


SCALE_ARTIFACTS = _artifact_paths()


def implementation_bindings_v43k() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_backtracking_preregistration_v43k.py",
        "backtracking_runtime": Path(backtracking.__file__).resolve(),
        "tests": ROOT / "test_lora_es_backtracking_v43k.py",
        "v43i_runtime": Path(prior.__file__).resolve(),
        "v43i_worker": ROOT / "eggroll_es_worker_lora_v43i.py",
        "v43i_fused_anchor_runtime": Path(fused.__file__).resolve(),
        "v43j_success_report": V43J_REPORT,
        "v43j_ratio_0p25_gate": V43J_GATE,
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


def _self_hashed_v43k(path: Path, file_sha: str, content_sha: str) -> dict:
    if v40a.file_sha256(path) != file_sha:
        raise RuntimeError(f"v43k predecessor file identity changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or v40a.canonical_sha256(compact) != content_sha
    ):
        raise RuntimeError(f"v43k predecessor content identity changed: {path}")
    return value


def v43j_untried_scale_evidence_v43k() -> dict:
    report = _self_hashed_v43k(
        V43J_REPORT, V43J_EXPECTED["report"], V43J_EXPECTED["report_content"]
    )
    gate = _self_hashed_v43k(
        V43J_GATE, V43J_EXPECTED["gate"], V43J_EXPECTED["gate_content"]
    )
    checks = gate.get("gate", {}).get("checks", {})
    results = report.get("scale_results", [])
    if (
        report.get("schema")
        != "matched-lora-es-backtracking-train-only-report-v43j"
        or report.get("status")
        != "complete_largest_passing_scale_committed_and_sealed"
        or report.get("accepted_target_norm_ratio") != 0.25
        or len(results) != 1
        or results[0].get("target_norm_ratio") != 0.25
        or results[0].get("preservation_gate_passed") is not True
        or report.get("shadow_dev_eval_ood_or_holdout_opened") is not False
        or report.get("protected_semantics_opened") is not False
        or report.get("quality_or_promotion_conclusion_authorized") is not False
        or gate.get("schema")
        != "matched-lora-es-backtracked-candidate-gate-v43j"
        or gate.get("target_norm_ratio") != 0.25
        or gate.get("gate", {}).get("passed") is not True
        or set(checks) != SIX_GATE_NAMES_V43K
        or any(value is not True for value in checks.values())
        or gate.get("protected_semantics_opened") is not False
    ):
        raise RuntimeError("v43k V43J untried-scale evidence changed")
    return {
        "schema": "sealed-v43j-untried-scale-evidence-v43k",
        "report_file_sha256": V43J_EXPECTED["report"],
        "report_content_sha256": V43J_EXPECTED["report_content"],
        "ratio_0p25_gate_file_sha256": V43J_EXPECTED["gate"],
        "ratio_0p25_gate_content_sha256": V43J_EXPECTED["gate_content"],
        "ratio_0p25_all_six_train_only_gates_passed": True,
        "ratios_evaluated_by_v43j": [0.25],
        "ratios_untried_before_v43k": [0.125, 0.0625],
        "protected_semantics_opened": False,
        "holdout_or_heldout_opened": False,
    }
def load_preregistration_v43k(args) -> tuple[dict, dict]:
    path = Path(args.preregistration).resolve()
    if v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v43k preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    evidence = backtracking.load_v43i_evidence_v43k(V43I_EVIDENCE_PATHS)
    predecessor = v43j_untried_scale_evidence_v43k()
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or v40a.canonical_sha256(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "matched-lora-es-backtracking-preregistration-v43k"
        or value.get("status") != "preregistered_before_train_only_launch"
        or value.get("gpu_launch_authorized") is not True
        or value.get("sealed_holdout_opened") is not False
        or value.get("shadow_dev_eval_ood_or_holdout_authorized") is not False
        or value.get("protected_semantic_access") is not False
        or value.get("current_v42i_holdout_cycle_eligible") is not False
        or value.get(
            "current_fixed_holdout_cycle_result_may_be_used_for_tuning"
        ) is not False
        or value.get("access_contract", {}).get(
            "current_fixed_holdout_cycle_report_opened_or_hashed"
        ) is not False
        or value.get("access_contract", {}).get(
            "current_fixed_holdout_cycle_result_bound"
        ) is not False
        or value.get("access_contract", {}).get(
            "current_fixed_holdout_cycle_result_influenced_design"
        ) is not False
        or value.get("v43j_untried_scale_evidence") != predecessor
        or value.get("v43i_evidence") != evidence
        or value.get("implementation_bindings") != implementation_bindings_v43k()
        or value.get("recipe", {}).get("seeds") != prior.SEEDS
        or value.get("recipe", {}).get("alpha") != prior.ALPHA
        or value.get("recipe", {}).get("scale_plans") != evidence["scale_plans"]
        or value.get("recipe", {}).get("scale_order")
        != list(backtracking.TARGET_NORM_RATIOS_V43K)
        or value.get("recipe", {}).get("resample_population") is not False
        or value.get("recipe", {}).get("recompute_projection") is not False
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    ):
        raise RuntimeError("v43k preregistration contract changed")
    return value, evidence


@contextmanager
def patched_prior_paths_v43k():
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


def _candidate_consensus_v43k(
    trainer, bundle, dense_items, requests, evidence: dict, ratio: float,
) -> dict:
    records = prior._score_repeats(
        trainer, bundle, dense_items, requests,
        warmups=0, retained=numeric.POST_UPDATE_REPEATS_V43G,
    )
    result = numeric.post_update_consensus_v43g(
        records, evidence["numeric_calibration_bootstrap_bounds"],
    )
    label = SCALE_LABELS_V43K[ratio]
    return _persist(SCALE_ARTIFACTS[f"candidate_consensus_{label}"], {
        "schema": "matched-lora-es-backtracked-candidate-consensus-v43k",
        "status": "complete_while_candidate_uncommitted",
        "target_norm_ratio": ratio,
        "records": records,
        "equivalence": result,
        "v43i_numeric_calibration_content_sha256": (
            backtracking.EXPECTED_FILES_V43K["numeric_calibration"][1]
        ),
        "protected_semantics_opened": False,
    })


def _finish_runtime(trainer, monitor, stop, failures, pid_map):
    stop.set()
    monitor.join(timeout=10)
    if monitor.is_alive() or not failures.empty():
        raise RuntimeError("v43k GPU monitor failed") from (
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
    prereg, evidence = load_preregistration_v43k(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "v43i_evidence_loaded": True,
            "population_resampled": False,
            "projection_recomputed": False,
            "scale_order": list(backtracking.TARGET_NORM_RATIOS_V43K),
            "model_runtime_loaded": False,
            "gpu_launched": False,
            "protected_paths_opened": [],
            "protected_semantic_access": False,
            "sealed_holdout_opened": False,
            "current_v42i_holdout_cycle_eligible": False,
            "filesystem_writes": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists():
        raise RuntimeError("v43k requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "matched-lora-es-backtracking-attempt-v43k",
        "status": "launching",
        "phase": "before_model_or_train_data_load",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "protected_semantics_opened": False,
        "current_v42i_holdout_cycle_eligible": False,
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
            raise RuntimeError("v43k frozen train bundle identity changed")
        bundle = prior.augment_unit_membership_v43i(bundle)
        anchor_bundle = fused.load_anchor_bundle_v43i(
            prior.PROSE_ANCHOR, prior.PROSE_REPORT,
            prior.QA_ANCHOR, prior.QA_REPORT,
        )
        v40a.base.set_seed(prior.GLOBAL_SEED)
        with patched_prior_paths_v43k():
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
                raise RuntimeError("v43k initial state differs from V43I exact restore")
            master = masters[0]
            master_runtime_sha = next(iter(runtime_hashes))
            update_sequences = {item["update_sequence"] for item in certificates}
            if update_sequences != {0}:
                raise RuntimeError("v43k restored update sequence changed")
            references = v40a._rpc_all(
                trainer, "capture_adapter_reference_v41a",
            )
            reference_generations = {
                item["reference_generation"] for item in references
            }
            if len(reference_generations) != 1:
                raise RuntimeError("v43k reference generation differs across ranks")
            postinstall = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            if preinstall["consensus"] != postinstall["consensus"]:
                raise RuntimeError("v43k canonical install changed initial score")
            full_plan = fused.fused_requests_v43i(requests, full_anchors)
            phase.value = "full_anchor_reference_score"
            reference_actors = prior._generate_fused_actor_scores(
                trainer, bundle, dense_items, full_plan, full_anchors,
            )

            for scale_plan in evidence["scale_plans"]:
                ratio = scale_plan["target_norm_ratio"]
                label = SCALE_LABELS_V43K[ratio]
                plan_id = v40a.canonical_sha256({
                    "schema": "matched-lora-es-backtracking-plan-v43k",
                    "v43i_population_reliability_content_sha256": (
                        backtracking.EXPECTED_FILES_V43K[
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
                    "continuation_policy": (
                        "evaluate 0.0625 only if 0.125 fails the six "
                        "train-only gates"
                    ),
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
                if (
                    set(gate.get("checks", {})) != SIX_GATE_NAMES_V43K
                    or any(
                        not isinstance(value, bool)
                        for value in gate.get("checks", {}).values()
                    )
                    or gate.get("passed") is not all(
                        gate.get("checks", {}).values()
                    )
                ):
                    raise RuntimeError("v43k six train-only gate schema changed")
                gate_artifact = _persist(
                    SCALE_ARTIFACTS[f"candidate_gate_{label}"], {
                        "schema": "matched-lora-es-backtracked-candidate-gate-v43k",
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
                six_gate_passed = gate["passed"]
                acceptance_passed = six_gate_passed
                if six_gate_passed:
                    phase.value = f"uncommitted_consensus_{label}"
                    consensus = _candidate_consensus_v43k(
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
                        "six_train_only_gates_passed": six_gate_passed,
                        "preservation_gate_passed": six_gate_passed,
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
                    backtracking.selected_diagnostic_scale_v43k(scale_results)
                    if six_gate_passed:
                        break
                    continue

                phase.value = f"commit_passing_diagnostic_{label}"
                update = prior._commit_accept_update(trainer, transaction_state)
                phase.value = f"seal_snapshot_{label}"
                update = prior._seal_accepted_update(trainer, update)
                transaction_accepted = True
                transaction_state = None
                accepted_ratio = ratio
                scale_results.append({
                    "target_norm_ratio": ratio,
                    "gate_passed": True,
                    "six_train_only_gates_passed": True,
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
                if backtracking.selected_diagnostic_scale_v43k(
                    scale_results
                ) != ratio:
                    raise RuntimeError("v43k diagnostic scale selection changed")
                break

            if accepted_ratio is None:
                if backtracking.selected_diagnostic_scale_v43k(
                    scale_results
                ) is not None:
                    raise RuntimeError("v43k no-pass backtracking result changed")
            gpu, cleanup, idle = _finish_runtime(
                trainer, monitor, stop, failures, pid_map,
            )
            if gpu.get("all_four_attributed_positive") is not True:
                raise RuntimeError("v43k all-four-GPU attribution gate failed")
            trainer = None

        report = v40a.self_hashed({
            "schema": "matched-lora-es-backtracking-train-only-report-v43k",
            "status": (
                "complete_passing_diagnostic_scale_committed_and_sealed"
                if accepted_ratio is not None
                else "complete_no_diagnostic_accepted_all_candidates_exactly_restored"
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
            "six_gate_continuation_policy_enforced": True,
            "ratio_0p0625_opened_only_after_ratio_0p125_six_gate_failure": (
                len(scale_results) < 2
                or scale_results[0]["six_train_only_gates_passed"] is False
            ),
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
            "current_v42i_holdout_cycle_eligible": False,
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
            "schema": "matched-lora-es-backtracking-failure-v43k",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "scale_results": scale_results,
            "emergency_exact_abort": emergency_abort,
            "emergency_exact_abort_failure": emergency_abort_failure,
            "transaction_accepted_before_failure": transaction_accepted,
            "protected_semantics_opened": False,
            "current_v42i_holdout_cycle_eligible": False,
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
