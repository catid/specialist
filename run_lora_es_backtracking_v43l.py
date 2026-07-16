#!/usr/bin/env python3
"""Train-only diagnostic of the untried V43I 0.03125/0.015625 scales."""

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

import lora_es_backtracking_v43l as backtracking
import run_lora_es_multi_anchor_v43i as prior


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "v43l_lora_es_v43i_projection_untried_backtracking"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
REPORT = (RUN_DIR / "matched_lora_es_backtracking_report_v43l.json").resolve()
FAILURE = (RUN_DIR / "failure_v43l.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v43l.jsonl").resolve()
SNAPSHOT = (RUN_DIR / "adapter_backtracked_v43l").resolve()
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_projection_backtracking_v43l.json"
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
SCALE_LABELS_V43L = {
    0.03125: "ratio_0p03125", 0.015625: "ratio_0p015625",
}
V43K_RUN = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43k_lora_es_v43i_projection_untried_backtracking"
).resolve()
V43K_REPORT = V43K_RUN / "matched_lora_es_backtracking_report_v43k.json"
V43K_GATE_0P125 = V43K_RUN / "candidate_gate_ratio_0p125_v43k.json"
V43K_ABORT_0P125 = V43K_RUN / "exact_abort_ratio_0p125_v43k.json"
V43K_GATE_0P0625 = V43K_RUN / "candidate_gate_ratio_0p0625_v43k.json"
V43K_ABORT_0P0625 = V43K_RUN / "exact_abort_ratio_0p0625_v43k.json"
V43K_GPU_LOG = V43K_RUN / "gpu_activity_v43k.jsonl"
V43K_EXPECTED = {
    "report": "a24319b4d111e7924c4c2428f8f47bde99a41b76e0bd8c49f3089e4009325e5d",
    "report_content": "1c167b6c1f72e95aa7330ba4496201942c7458c470db7aebec2caad53a9df772",
    "gate_0p125": "6661eb7c4409bc690e8987b958b13c3036631252e0a0de56d270981bc2850753",
    "gate_0p125_content": "51c8181d911cd2e66be254a53897e5170669180546b5f8c777411c9918ee6359",
    "abort_0p125": "139688bc12726a71aee39d71f1bba0798b0f05d3a5cd3a02950ea47c451da2c2",
    "abort_0p125_content": "d3c133b94bf516cd34be94e6770bafb4f89b3c0f90d52202d4711ce5489eb606",
    "gate_0p0625": "10c545e785c52d4d1d7471b4c1711aa2a4eb9e289638e21ce41f3365a44b7829",
    "gate_0p0625_content": "a8d189eb28670c28fb500fab429b2d406217cd6b4dadc9f916a8d5677ae4f8a4",
    "abort_0p0625": "149f3a21b1cb922719c6a2868cafdc6f38a978018a7923d6897812507df9f3a2",
    "abort_0p0625_content": "86f83658a57742ee3972164af1438cd807c5750aa959e49dcdee6f9710edf86f",
    "gpu_log": "0a3082355863b551477799e87737183aa1ba7014b86027d542ba49275fbcf6bb",
}
SIX_GATE_NAMES_V43L = {
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
    for ratio, label in SCALE_LABELS_V43L.items():
        result[f"candidate_gate_{label}"] = RUN_DIR / f"candidate_gate_{label}_v43l.json"
        result[f"exact_abort_{label}"] = RUN_DIR / f"exact_abort_{label}_v43l.json"
        result[f"candidate_consensus_{label}"] = (
            RUN_DIR / f"candidate_consensus_{label}_v43l.json"
        )
    return {key: value.resolve() for key, value in result.items()}


SCALE_ARTIFACTS = _artifact_paths()


def implementation_bindings_v43l() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_lora_es_backtracking_preregistration_v43l.py",
        "backtracking_runtime": Path(backtracking.__file__).resolve(),
        "tests": ROOT / "test_lora_es_backtracking_v43l.py",
        "v43i_runtime": Path(prior.__file__).resolve(),
        "v43i_worker": ROOT / "eggroll_es_worker_lora_v43i.py",
        "v43i_fused_anchor_runtime": Path(fused.__file__).resolve(),
        "v43k_report": V43K_REPORT,
        "v43k_ratio_0p125_gate": V43K_GATE_0P125,
        "v43k_ratio_0p125_abort": V43K_ABORT_0P125,
        "v43k_ratio_0p0625_gate": V43K_GATE_0P0625,
        "v43k_ratio_0p0625_abort": V43K_ABORT_0P0625,
        "v43k_gpu_log": V43K_GPU_LOG,
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


def _self_hashed_v43l(path: Path, file_sha: str, content_sha: str) -> dict:
    if v40a.file_sha256(path) != file_sha:
        raise RuntimeError(f"v43l predecessor file identity changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or v40a.canonical_sha256(compact) != content_sha
    ):
        raise RuntimeError(f"v43l predecessor content identity changed: {path}")
    return value


def v43k_train_only_evidence_v43l() -> dict:
    report = _self_hashed_v43l(
        V43K_REPORT, V43K_EXPECTED["report"], V43K_EXPECTED["report_content"]
    )
    artifacts = {}
    for label, ratio, gate_path, abort_path in (
        ("0p125", 0.125, V43K_GATE_0P125, V43K_ABORT_0P125),
        ("0p0625", 0.0625, V43K_GATE_0P0625, V43K_ABORT_0P0625),
    ):
        gate = _self_hashed_v43l(
            gate_path,
            V43K_EXPECTED[f"gate_{label}"],
            V43K_EXPECTED[f"gate_{label}_content"],
        )
        abort = _self_hashed_v43l(
            abort_path,
            V43K_EXPECTED[f"abort_{label}"],
            V43K_EXPECTED[f"abort_{label}_content"],
        )
        checks = gate.get("gate", {}).get("checks", {})
        if (
            gate.get("schema")
            != "matched-lora-es-backtracked-candidate-gate-v43k"
            or gate.get("status") != "complete_before_commit_or_abort"
            or gate.get("target_norm_ratio") != ratio
            or gate.get("gate", {}).get("passed") is not False
            or set(checks) != SIX_GATE_NAMES_V43L
            or checks.get("qa_generation_f1_noninferiority") is not False
            or any(
                value is not True for name, value in checks.items()
                if name != "qa_generation_f1_noninferiority"
            )
            or gate.get("protected_semantics_opened") is not False
            or abort.get("schema") != "controller-exact-abort-readback-v43i"
            or abort.get("status") != "scale_rejected_and_exactly_restored"
            or abort.get("target_norm_ratio") != ratio
            or abort.get("candidate_gate_content_sha256")
            != V43K_EXPECTED[f"gate_{label}_content"]
            or abort.get("reason") != "candidate preservation gate failed"
            or abort.get("all_four_ranks_exact") is not True
            or abort.get("restored_master_identity", {}).get("sha256")
            != backtracking.RESTORED_MASTER_SHA256_V43L
            or abort.get("restored_runtime_values_sha256")
            != backtracking.RESTORED_RUNTIME_SHA256_V43L
            or abort.get("protected_semantics_opened") is not False
        ):
            raise RuntimeError(f"v43l V43K {label} gate/abort evidence changed")
        artifacts[label] = {
            "target_norm_ratio": ratio,
            "gate_file_sha256": V43K_EXPECTED[f"gate_{label}"],
            "gate_content_sha256": V43K_EXPECTED[f"gate_{label}_content"],
            "six_train_only_gates_passed": False,
            "failed_gate": "qa_generation_f1_noninferiority",
            "abort_file_sha256": V43K_EXPECTED[f"abort_{label}"],
            "abort_content_sha256": V43K_EXPECTED[f"abort_{label}_content"],
            "exact_abort_readback_passed": True,
        }
    results = report.get("scale_results", [])
    if (
        report.get("schema")
        != "matched-lora-es-backtracking-train-only-report-v43k"
        or report.get("status")
        != "complete_no_diagnostic_accepted_all_candidates_exactly_restored"
        or report.get("accepted_target_norm_ratio") is not None
        or [item.get("target_norm_ratio") for item in results]
        != [0.125, 0.0625]
        or any(item.get("six_train_only_gates_passed") is not False
               for item in results)
        or any(item.get("exact_abort_readback_passed") is not True
               for item in results)
        or report.get("six_gate_continuation_policy_enforced") is not True
        or report.get("shadow_dev_eval_ood_or_holdout_opened") is not False
        or report.get("protected_semantics_opened") is not False
        or report.get("quality_or_promotion_conclusion_authorized") is not False
        or report.get("current_v42i_holdout_cycle_eligible") is not False
        or report.get("gpu_activity", {}).get(
            "all_four_attributed_positive"
        ) is not True
        or report.get("gpu_log", {}).get("file_sha256")
        != V43K_EXPECTED["gpu_log"]
        or v40a.file_sha256(V43K_GPU_LOG) != V43K_EXPECTED["gpu_log"]
    ):
        raise RuntimeError("v43l V43K train-only report evidence changed")
    return {
        "schema": "sealed-v43k-train-only-continuation-evidence-v43l",
        "report_file_sha256": V43K_EXPECTED["report"],
        "report_content_sha256": V43K_EXPECTED["report_content"],
        "gpu_log_file_sha256": V43K_EXPECTED["gpu_log"],
        "per_scale_gate_and_abort_seals": artifacts,
        "ratios_evaluated_by_v43k": [0.125, 0.0625],
        "ratios_untried_before_v43l": [0.03125, 0.015625],
        "all_four_gpus_attributed_positive": True,
        "protected_semantics_opened": False,
        "holdout_or_heldout_opened": False,
    }
def load_preregistration_v43l(args) -> tuple[dict, dict]:
    path = Path(args.preregistration).resolve()
    if v40a.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("v43l preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    evidence = backtracking.load_v43i_evidence_v43l(V43I_EVIDENCE_PATHS)
    predecessor = v43k_train_only_evidence_v43l()
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or v40a.canonical_sha256(compact)
        != args.preregistration_content_sha256
        or value.get("schema")
        != "matched-lora-es-backtracking-preregistration-v43l"
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
        or value.get("v43k_train_only_evidence") != predecessor
        or value.get("v43i_evidence") != evidence
        or value.get("implementation_bindings") != implementation_bindings_v43l()
        or value.get("recipe", {}).get("seeds") != prior.SEEDS
        or value.get("recipe", {}).get("alpha") != prior.ALPHA
        or value.get("recipe", {}).get("scale_plans") != evidence["scale_plans"]
        or value.get("recipe", {}).get("scale_order")
        != list(backtracking.TARGET_NORM_RATIOS_V43L)
        or value.get("recipe", {}).get("resample_population") is not False
        or value.get("recipe", {}).get("recompute_projection") is not False
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    ):
        raise RuntimeError("v43l preregistration contract changed")
    return value, evidence


@contextmanager
def patched_prior_paths_v43l():
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


def _candidate_consensus_v43l(
    trainer, bundle, dense_items, requests, evidence: dict, ratio: float,
) -> dict:
    records = prior._score_repeats(
        trainer, bundle, dense_items, requests,
        warmups=0, retained=numeric.POST_UPDATE_REPEATS_V43G,
    )
    result = numeric.post_update_consensus_v43g(
        records, evidence["numeric_calibration_bootstrap_bounds"],
    )
    label = SCALE_LABELS_V43L[ratio]
    return _persist(SCALE_ARTIFACTS[f"candidate_consensus_{label}"], {
        "schema": "matched-lora-es-backtracked-candidate-consensus-v43l",
        "status": "complete_while_candidate_uncommitted",
        "target_norm_ratio": ratio,
        "records": records,
        "equivalence": result,
        "v43i_numeric_calibration_content_sha256": (
            backtracking.EXPECTED_FILES_V43L["numeric_calibration"][1]
        ),
        "protected_semantics_opened": False,
    })


def _finish_runtime(trainer, monitor, stop, failures, pid_map):
    stop.set()
    monitor.join(timeout=10)
    if monitor.is_alive() or not failures.empty():
        raise RuntimeError("v43l GPU monitor failed") from (
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
    prereg, evidence = load_preregistration_v43l(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "v43i_evidence_loaded": True,
            "population_resampled": False,
            "projection_recomputed": False,
            "scale_order": list(backtracking.TARGET_NORM_RATIOS_V43L),
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
        raise RuntimeError("v43l requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = v40a.self_hashed({
        "schema": "matched-lora-es-backtracking-attempt-v43l",
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
            raise RuntimeError("v43l frozen train bundle identity changed")
        bundle = prior.augment_unit_membership_v43i(bundle)
        anchor_bundle = fused.load_anchor_bundle_v43i(
            prior.PROSE_ANCHOR, prior.PROSE_REPORT,
            prior.QA_ANCHOR, prior.QA_REPORT,
        )
        v40a.base.set_seed(prior.GLOBAL_SEED)
        with patched_prior_paths_v43l():
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
                raise RuntimeError("v43l initial state differs from V43I exact restore")
            master = masters[0]
            master_runtime_sha = next(iter(runtime_hashes))
            update_sequences = {item["update_sequence"] for item in certificates}
            if update_sequences != {0}:
                raise RuntimeError("v43l restored update sequence changed")
            references = v40a._rpc_all(
                trainer, "capture_adapter_reference_v41a",
            )
            reference_generations = {
                item["reference_generation"] for item in references
            }
            if len(reference_generations) != 1:
                raise RuntimeError("v43l reference generation differs across ranks")
            postinstall = prior._exact_base_score(
                trainer, bundle, dense_items, requests,
            )
            if preinstall["consensus"] != postinstall["consensus"]:
                raise RuntimeError("v43l canonical install changed initial score")
            full_plan = fused.fused_requests_v43i(requests, full_anchors)
            phase.value = "full_anchor_reference_score"
            reference_actors = prior._generate_fused_actor_scores(
                trainer, bundle, dense_items, full_plan, full_anchors,
            )

            for scale_plan in evidence["scale_plans"]:
                ratio = scale_plan["target_norm_ratio"]
                label = SCALE_LABELS_V43L[ratio]
                plan_id = v40a.canonical_sha256({
                    "schema": "matched-lora-es-backtracking-plan-v43l",
                    "v43i_population_reliability_content_sha256": (
                        backtracking.EXPECTED_FILES_V43L[
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
                        "evaluate 0.015625 only if 0.03125 fails the six "
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
                    set(gate.get("checks", {})) != SIX_GATE_NAMES_V43L
                    or any(
                        not isinstance(value, bool)
                        for value in gate.get("checks", {}).values()
                    )
                    or gate.get("passed") is not all(
                        gate.get("checks", {}).values()
                    )
                ):
                    raise RuntimeError("v43l six train-only gate schema changed")
                gate_artifact = _persist(
                    SCALE_ARTIFACTS[f"candidate_gate_{label}"], {
                        "schema": "matched-lora-es-backtracked-candidate-gate-v43l",
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
                    consensus = _candidate_consensus_v43l(
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
                    backtracking.selected_diagnostic_scale_v43l(scale_results)
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
                if backtracking.selected_diagnostic_scale_v43l(
                    scale_results
                ) != ratio:
                    raise RuntimeError("v43l diagnostic scale selection changed")
                break

            if accepted_ratio is None:
                if backtracking.selected_diagnostic_scale_v43l(
                    scale_results
                ) is not None:
                    raise RuntimeError("v43l no-pass backtracking result changed")
            gpu, cleanup, idle = _finish_runtime(
                trainer, monitor, stop, failures, pid_map,
            )
            if gpu.get("all_four_attributed_positive") is not True:
                raise RuntimeError("v43l all-four-GPU attribution gate failed")
            trainer = None

        report = v40a.self_hashed({
            "schema": "matched-lora-es-backtracking-train-only-report-v43l",
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
            "ratio_0p015625_opened_only_after_ratio_0p03125_six_gate_failure": (
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
            "schema": "matched-lora-es-backtracking-failure-v43l",
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
