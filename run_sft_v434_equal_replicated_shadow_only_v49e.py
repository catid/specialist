#!/usr/bin/env python3
"""Run the eligible-only replicated V434-equal shadow evaluation."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import run_eggroll_es_equal_unit_v38a as cleanup
import run_lora_topology_probe_v40a as topology
import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as environment
import run_sealed_candidate_eval_v39a as metrics
import run_sft_v434_sampling_midpoint_ood_only_v49d as v49d
import train_eggroll_es_specialist as base


ROOT = Path(__file__).resolve().parent
BASE_ARMS = ("base_a", "base_b")
LOGICAL_CANDIDATE = "v434_equal"
CANDIDATE_ARMS = ("v434_equal_a", "v434_equal_b")
ARMS = BASE_ARMS + CANDIDATE_ARMS
STAGED_BY_ARM = {arm: v49d.STAGED_BY_LOGICAL[LOGICAL_CANDIDATE] for arm in CANDIDATE_ARMS}
ADAPTER_IDS = {arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS)}
RANK_FIELDS = (
    "generated_equal_unit_mean_reward",
    "generated_exact_count",
    "generated_nonzero_count",
    "teacher_forced_equal_unit_mean_answer_logprob",
)
SHADOW_INPUTS = {
    "shadow": dict(core.PROTECTED_INPUTS_V44A["shadow"]),
    "split_manifest": dict(core.PROTECTED_INPUTS_V44A["split_manifest"]),
}
OOD_RECOVERY_REPORT = (
    ROOT / "experiments/eval_reports/"
    "sft_v434_equal_vs_source50_replicated_ood_only_retry1_offline_recovery_v49d.json"
).resolve()
OOD_RECOVERY_REPORT_FILE_SHA256 = (
    "6bf7f1b7740e89689e26b2a3741ee04096aad28d24deb91e7c993a574e2b1ffc"
)
OOD_RECOVERY_REPORT_CONTENT_SHA256 = (
    "e0a10ca1e0bfd26b52b34f688089d7f971f1b2c6f7acd807788243411f3df80d"
)
EXPERIMENT = "v49e_sft_v434_equal_replicated_shadow_only"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v49e.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v49e.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "sft_v434_equal_replicated_shadow_only_v49e.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v434_equal_replicated_shadow_only_v49e.json"
).resolve()


def arm_wave_plan_v49e():
    return (tuple((arm, index) for index, arm in enumerate(ARMS)),)


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def ood_recovery_proof_v49e() -> dict:
    if core.file_sha256(OOD_RECOVERY_REPORT) != OOD_RECOVERY_REPORT_FILE_SHA256:
        raise RuntimeError("V49E OOD recovery report file changed")
    value = json.loads(OOD_RECOVERY_REPORT.read_text(encoding="utf-8"))
    equal = value.get("per_logical_candidate_gate_table", {}).get("v434_equal", {})
    source = value.get("per_logical_candidate_gate_table", {}).get("v434_source50", {})
    access = value.get("access_proof", {})
    if (
        _compact_sha(value) != OOD_RECOVERY_REPORT_CONTENT_SHA256
        or value.get("content_sha256_before_self_field")
        != OOD_RECOVERY_REPORT_CONTENT_SHA256
        or value.get("status")
        != "complete_from_sealed_retry1_receipts_shadow_and_holdout_unopened"
        or equal.get("both_replicas_independently_ood_eligible") is not True
        or source.get("both_replicas_independently_ood_eligible") is not False
        or [gate.get("arm") for gate in equal.get("replica_gates", [])]
        != list(CANDIDATE_ARMS)
        or not all(gate.get("eligible") is True for gate in equal.get("replica_gates", []))
        or access.get("shadow_opened") is not False
        or access.get("heldout_or_holdout_opened") is not False
        or access.get("offline_recovery_protected_semantic_input_reads") != 0
    ):
        raise RuntimeError("V49E OOD eligibility proof changed")
    return {
        "path": str(OOD_RECOVERY_REPORT),
        "file_sha256": OOD_RECOVERY_REPORT_FILE_SHA256,
        "content_sha256": OOD_RECOVERY_REPORT_CONTENT_SHA256,
        "eligible_logical_candidates": [LOGICAL_CANDIDATE],
        "excluded_ineligible_logical_candidates": ["v434_source50"],
        "both_equal_replicas_independently_ood_eligible": True,
        "source50_excluded_before_shadow_access": True,
        "shadow_opened_in_ood_phase": False,
        "heldout_or_holdout_opened": False,
    }


def canonical_stage_binding_v49e() -> dict:
    return v49d.canonical_stage_binding_v49d(LOGICAL_CANDIDATE)


def replica_stage_bindings_v49e() -> dict:
    stage = canonical_stage_binding_v49e()
    return {
        arm: {**stage, "replica_arm": arm, "adapter_id": ADAPTER_IDS[arm]}
        for arm in CANDIDATE_ARMS
    }


def implementation_bindings_v49e() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_sft_v434_equal_replicated_shadow_preregistration_v49e.py",
        "tests": ROOT / "test_sft_v434_equal_replicated_shadow_v49e.py",
        "core_runtime": Path(core.__file__).resolve(),
        "metric_runtime": Path(metrics.__file__).resolve(),
        "environment_runtime": Path(environment.environment.__file__).resolve(),
        "topology_runtime": Path(topology.__file__).resolve(),
        "cleanup_runtime": Path(cleanup.__file__).resolve(),
        "stage_runtime": Path(v49d.stage.__file__).resolve(),
        "v49d_ood_runtime": Path(v49d.__file__).resolve(),
        "ood_recovery_report": OOD_RECOVERY_REPORT,
        "staged_weights": STAGED_BY_ARM[CANDIDATE_ARMS[0]] / "adapter_model.safetensors",
        "staged_config": STAGED_BY_ARM[CANDIDATE_ARMS[0]] / "adapter_config.json",
        "stage_manifest": STAGED_BY_ARM[CANDIDATE_ARMS[0]] / "stage_manifest_v44a.json",
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
    }
    # Protected shadow/split inputs are intentionally absent.  They are bound
    # by pre-existing committed identities and opened once only after launch.
    result = {name: core.file_sha256(path) for name, path in paths.items()}
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    result["environment"] = environment.environment.environment_bindings_v44b()
    return result


def load_preregistration_v49e(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V49E preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or _compact_sha(value) != content
        or value.get("schema") != "v49e-v434-equal-replicated-shadow-only-preregistration"
        or value.get("status") != "sealed_after_ood_eligibility_before_shadow_access"
        or value.get("evaluation_launch_authorized") is not True
        or value.get("shadow_access_authorized") is not True
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("single_access_inputs") != SHADOW_INPUTS
        or value.get("arms") != list(ARMS)
        or value.get("one_full_fixed_wave")
        != [{"arm": arm, "engine_index": engine} for arm, engine in arm_wave_plan_v49e()[0]]
        or value.get("staged_adapters") != replica_stage_bindings_v49e()
        or value.get("ood_eligibility_proof") != ood_recovery_proof_v49e()
        or value.get("implementation_bindings") != implementation_bindings_v49e()
    ):
        raise RuntimeError("V49E preregistration content changed")
    core._forbid_holdout_v44a(item["path"] for item in SHADOW_INPUTS.values())
    return value


@contextmanager
def patched_arm_globals_v49e():
    saved = (
        core.BASE_ARMS, core.CANDIDATE_ARMS, core.ARMS, core.STAGED_BY_ARM,
        core.ADAPTER_IDS_V44A, core.ENGINE_INDEX_BY_ARM_V44A,
        core.arm_wave_plan_v44a, core.EXPERIMENT, core.RUN_DIR,
    )
    core.BASE_ARMS = BASE_ARMS
    core.CANDIDATE_ARMS = CANDIDATE_ARMS
    core.ARMS = ARMS
    core.STAGED_BY_ARM = STAGED_BY_ARM
    core.ADAPTER_IDS_V44A = ADAPTER_IDS
    core.ENGINE_INDEX_BY_ARM_V44A = {
        arm: engine for arm, engine in arm_wave_plan_v49e()[0]
    }
    core.arm_wave_plan_v44a = arm_wave_plan_v49e
    core.EXPERIMENT = EXPERIMENT
    core.RUN_DIR = RUN_DIR
    try:
        yield
    finally:
        (
            core.BASE_ARMS, core.CANDIDATE_ARMS, core.ARMS, core.STAGED_BY_ARM,
            core.ADAPTER_IDS_V44A, core.ENGINE_INDEX_BY_ARM_V44A,
            core.arm_wave_plan_v44a, core.EXPERIMENT, core.RUN_DIR,
        ) = saved


def summarize_shadow_gpu_v49e(path: Path, expected_pids: dict[int, int]) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    if {row.get("phase") for row in rows} != {"shadow"}:
        raise RuntimeError("V49E GPU phase labels changed")
    result = {}
    for gpu in (0, 1, 2, 3):
        selected = [row for row in rows if row["gpu"] == gpu]
        if (
            not selected
            or any(row["foreign_compute_pids"] for row in selected)
            or any(row["expected_pid"] != expected_pids[gpu] for row in selected)
        ):
            raise RuntimeError(f"V49E GPU {gpu} attribution changed")
        resident = [row for row in selected if expected_pids[gpu] in row["compute_pids"]]
        positive = [row for row in resident if row["utilization_percent"] > 0]
        if not resident or not positive:
            raise RuntimeError(f"V49E GPU {gpu} inactive in shadow")
        result[str(gpu)] = {
            "expected_pid": expected_pids[gpu],
            "phase": "shadow",
            "samples": len(selected),
            "resident_samples": len(resident),
            "positive_samples": len(positive),
            "peak_utilization_percent": max(row["utilization_percent"] for row in resident),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in resident),
            "mean_resident_utilization_percent": math.fsum(
                row["utilization_percent"] for row in resident
            ) / len(resident),
        }
    return {
        "phase_labels_exact": ["shadow"],
        "all_four_attributed_positive_in_shadow": True,
        "by_gpu": result,
    }


def shadow_decision_v49e(shadow: dict) -> dict:
    if shadow["base_a"] != shadow["base_b"]:
        raise RuntimeError("V49E exact base duplicate gate failed")
    mean = {
        field: math.fsum(float(shadow[arm][field]) for arm in CANDIDATE_ARMS) / 2.0
        for field in RANK_FIELDS
    }
    baseline = {field: shadow["base_a"][field] for field in RANK_FIELDS}
    candidate_key = tuple(mean[field] for field in RANK_FIELDS)
    baseline_key = tuple(float(baseline[field]) for field in RANK_FIELDS)
    replica_protocol = {
        arm: all(
            shadow[arm]["protocol_leak_counters"][key]
            <= shadow["base_a"]["protocol_leak_counters"][key]
            for key in shadow["base_a"]["protocol_leak_counters"]
        ) for arm in CANDIDATE_ARMS
    }
    improved = candidate_key > baseline_key
    passed = improved and all(replica_protocol.values())
    return {
        "ood_eligible_set_frozen_before_shadow": [LOGICAL_CANDIDATE],
        "logical_candidate": LOGICAL_CANDIDATE,
        "replicas": list(CANDIDATE_ARMS),
        "mean_replicated_shadow_metrics": mean,
        "exact_base_shadow_metrics": baseline,
        "mean_reward_delta_vs_base": (
            mean["generated_equal_unit_mean_reward"]
            - float(baseline["generated_equal_unit_mean_reward"])
        ),
        "rank_fields": list(RANK_FIELDS),
        "selection_rule": "frozen lexicographic mean-replica shadow vector versus exact base",
        "both_replicas_no_protocol_or_leak_counter_increase": all(replica_protocol.values()),
        "per_replica_protocol_safe": replica_protocol,
        "shadow_improvement_gate_passed": improved,
        "replicated_equal_vs_base_decision_passed": passed,
        "selected_logical_candidate": LOGICAL_CANDIDATE if passed else None,
        "selected_arm": CANDIDATE_ARMS[0] if passed else "base_a",
        "candidate_replica_outputs_bit_exact_required": False,
        "source50_present_or_ranked": False,
        "selection_or_promotion_authorized": False,
    }


class Phase:
    value = "setup"


def parser():
    return core.parser()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v49e(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "single_access_labels": sorted(prereg["single_access_inputs"]),
            "one_full_fixed_wave": prereg["one_full_fixed_wave"],
            "protected_semantic_access_count": 0,
            "shadow_opened": False,
            "heldout_or_holdout_opened": False,
            "gpu_accessed": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists() or REPORT.exists():
        raise RuntimeError("V49E requires fresh artifact paths")
    environment.environment.environment_bindings_v44b()
    preflight = topology.gpu_preflight()
    attempt = core.self_hashed({
        "schema": "v49e-shadow-only-attempt",
        "status": "launching",
        "phase": "before_shadow_semantic_access",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "protected_semantic_access_count": 0,
        "shadow_opened": False,
        "heldout_or_holdout_opened": False,
        "preflight": preflight,
    })
    core.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = firewall = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = Phase()
    raw_sink = {"schema": "v49e-shadow-only-raw-local"}
    started = time.monotonic()
    try:
        saved_inputs = core.PROTECTED_INPUTS_V44A
        core.PROTECTED_INPUTS_V44A = SHADOW_INPUTS
        try:
            firewall = core.SingleSemanticAccessV44A(SHADOW_INPUTS)
            shadow_rows = firewall.jsonl("shadow")
            split_manifest = firewall.json("split_manifest")
        finally:
            core.PROTECTED_INPUTS_V44A = saved_inputs
        if set(firewall.receipts) != set(SHADOW_INPUTS):
            raise RuntimeError("V49E exact shadow/split access coverage changed")
        with patched_arm_globals_v49e():
            shadow_bundle, disjoint_proof = core.shadow_bundle_v44a(
                shadow_rows, split_manifest
            )
            base.set_seed(core.GENERATION_SEED)
            trainer, saved = core.make_trainer_v44a(prereg)
            actor_ids = trainer._resolve([
                engine.runtime_identity_v40a.remote() for engine in trainer.engines
            ])
            worker_ids = topology._rpc_all(trainer, "runtime_identity_v40a")
            pid_map = core.actor_pid_map_v44a(actor_ids, worker_ids)
            monitor = threading.Thread(
                target=metrics.monitor_gpus,
                args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
            )
            monitor.start()
            phase.value = "shadow"
            shadow = core.evaluate_qa_v44a(
                trainer, shadow_bundle, raw_sink, "shadow"
            )
        decision = shadow_decision_v49e(shadow)
        raw_sink["single_access_receipts"] = firewall.receipts
        core.atomic_json(RAW, raw_sink, mode=0o600)
        raw_sha = core.file_sha256(RAW)
        stop.set(); monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("V49E GPU monitor failed")
        gpu = summarize_shadow_gpu_v49e(GPU_LOG, pid_map)
        closed = cleanup.strict_close_trainer_v38a(trainer); trainer = None
        if (
            closed.get("engine_kill_count") != 4
            or closed.get("placement_group_remove_count") != 4
            or closed.get("all_four_gcs_states_removed") is not True
        ):
            raise RuntimeError("V49E exact four-engine cleanup changed")
        import ray
        ray.shutdown()
        idle = cleanup.wait_for_gpu_idle()
        report = core.self_hashed({
            "schema": "v49e-v434-equal-replicated-shadow-only-report",
            "status": "complete_shadow_only_holdout_unopened",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "arms": list(ARMS),
            "one_full_fixed_wave": prereg["one_full_fixed_wave"],
            "staged_adapters": prereg["staged_adapters"],
            "ood_eligibility_proof": prereg["ood_eligibility_proof"],
            "actor_identities": actor_ids,
            "worker_identities": worker_ids,
            "physical_gpu_pid_map": pid_map,
            "shadow_document_disjointness": disjoint_proof,
            "shadow": shadow,
            "base_duplicate_equivalence": {
                "base_a_equals_base_b_exactly": shadow["base_a"] == shadow["base_b"]
            },
            "replicated_equal_vs_base_decision": decision,
            "single_access_receipts": firewall.receipts,
            "gpu_activity": gpu,
            "placement_group_cleanup": closed,
            "final_gpu_idle": idle,
            "preflight": preflight,
            "raw_local_artifact": {
                "path": str(RAW), "file_sha256": raw_sha,
                "mode": "0600", "git_eligible": False,
            },
            "gpu_log": {"path": str(GPU_LOG), "file_sha256": core.file_sha256(GPU_LOG)},
            "protected_semantic_access_count": 2,
            "protected_semantic_access_labels": ["shadow", "split_manifest"],
            "shadow_opened": True,
            "heldout_or_holdout_opened": False,
            "source50_present_or_ranked": False,
            "selection_or_promotion_authorized": False,
        })
        core.atomic_json(REPORT, report)
        complete = dict(attempt)
        complete.pop("content_sha256_before_self_field", None)
        complete.update({
            "status": "complete", "phase": "shadow_aggregate_sealed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protected_semantic_access_count": 2,
            "shadow_opened": True,
            "report": str(REPORT), "report_sha256": core.file_sha256(REPORT),
        })
        core.atomic_json(ATTEMPT.with_suffix(".complete.json"), core.self_hashed(complete))
        print(json.dumps({
            "report": str(REPORT), "report_sha256": core.file_sha256(REPORT),
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        failure = core.self_hashed({
            "schema": "v49e-shadow-only-failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "protected_semantic_access_count": 0 if firewall is None else len(firewall.receipts),
            "protected_semantic_access_labels": [] if firewall is None else sorted(firewall.receipts),
            "shadow_opened": firewall is not None and "shadow" in firewall.receipts,
            "heldout_or_holdout_opened": False,
        })
        core.atomic_json(RUN_DIR / "failure_v49e.json", failure)
        raise
    finally:
        if trainer is not None:
            try:
                base.close_trainer(trainer)
            except Exception:
                pass
        if saved is not None:
            topology.EXPERIMENT, topology.RUN_DIR = saved
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
