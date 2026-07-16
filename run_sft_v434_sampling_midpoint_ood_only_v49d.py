#!/usr/bin/env python3
"""Run the narrow two-wave V49D OOD QA/prose comparison."""

from __future__ import annotations

import json
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
import run_matched_lora_candidate_eval_v45a as ood_first
import run_sealed_candidate_eval_v39a as metrics
import stage_v49d_adapters_vllm as stage
import train_eggroll_es_specialist as base


ROOT = Path(__file__).resolve().parent
BASE_ARMS = ("base_a", "base_b", "base_c", "base_d")
LOGICAL_REPLICAS = {
    "v434_equal": ("v434_equal_a", "v434_equal_b"),
    "v434_source50": ("v434_source50_a", "v434_source50_b"),
}
LOGICAL_CANDIDATES = tuple(LOGICAL_REPLICAS)
CANDIDATE_ARMS = tuple(
    arm for replicas in LOGICAL_REPLICAS.values() for arm in replicas
)
ARMS = BASE_ARMS + CANDIDATE_ARMS
STAGED_BY_LOGICAL = dict(stage.OUTPUTS)
STAGED_BY_ARM = {
    arm: STAGED_BY_LOGICAL[logical]
    for logical, replicas in LOGICAL_REPLICAS.items()
    for arm in replicas
}
ADAPTER_IDS = {arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS)}
STAGE_EXPECTED = {
    "v434_equal": {
        "weights": "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a",
        "config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
        "manifest_file": "e30ba44563b5db56f4a487b26f4e2310fd3755b15f8db69d9400facd8baa3813",
        "manifest_content": "ea328ada018e1c0d182d329d2a9cb81f8f0375aef93738f3b9c0a00f63c82da3",
        "transformed_identity": "f210bf05e7fe38481d0a7d9c641a7f902e575521b50e98bdc021bf11b49cb1c8",
    },
    "v434_source50": {
        "weights": "d5f0ae8b97f5ec8c4c99145d9d072432fd08ea0daabb3c830d24299ccd932acf",
        "config": "752e31a157428c91c68ff23181fe057f74a476a766263017a51dc22c7421cd53",
        "manifest_file": "96ea8ee3767d42c5e50059e65d933f7403e3efe91616a24d1cda40d114b1ac89",
        "manifest_content": "606fb1597a094aa0362063ab476367e1e0be53bb1db42f07f31470ac6825fdef",
        "transformed_identity": "04c6dbbd615b4478ee7db145de1d567f7f86ce9ffc670996d51c24a249091228",
    },
}
OOD_INPUTS = {
    "ood_qa": {
        "path": str((ROOT / "data/ood_qa_v3.jsonl").resolve()),
        "file_sha256": "25a48b9494134731e51043047afadb340291a9ae3e9cfec9d9cfd8c73ddb255d",
    },
    "ood_prose": {
        "path": str((ROOT / "data/ood_prose_v3.jsonl").resolve()),
        "file_sha256": "3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57",
    },
}
FUTURE_TEMPLATE = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v434_equal_vs_source50_replicated_ood_first_template_v49d.json"
).resolve()
FUTURE_TEMPLATE_FILE_SHA256 = (
    "30933a738f90b8796429671a668c99693873b534fa8160f404e91e0fb34ccfab"
)
FUTURE_TEMPLATE_CONTENT_SHA256 = (
    "03b1e989810c836186898f01480389d592b1357f1d46f94f4a45110f60e495de"
)
EXPERIMENT = "v49d_v434_equal_vs_source50_replicated_ood_only"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v49d.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v49d.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "sft_v434_equal_vs_source50_replicated_ood_only_v49d.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v434_equal_vs_source50_replicated_ood_only_v49d.json"
).resolve()
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_715


def arm_wave_plan_v49d():
    return (
        tuple((arm, index) for index, arm in enumerate(BASE_ARMS)),
        tuple((arm, index) for index, arm in enumerate(CANDIDATE_ARMS)),
    )


def canonical_stage_binding_v49d(logical: str) -> dict:
    directory = STAGED_BY_LOGICAL[logical]
    expected = STAGE_EXPECTED[logical]
    manifest_path = directory / "stage_manifest_v44a.json"
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {k: v for k, v in value.items() if k != "content_sha256_before_self_field"}
    observed = {
        "logical_candidate": logical,
        "directory": str(directory),
        "weights_file_sha256": core.file_sha256(directory / "adapter_model.safetensors"),
        "adapter_config_file_sha256": core.file_sha256(directory / "adapter_config.json"),
        "manifest_file_sha256": core.file_sha256(manifest_path),
        "manifest_content_sha256": content,
        "transformed_identity_sha256": value.get("transformed_identity", {}).get("sha256"),
        "tensor_count": value.get("transformed_identity", {}).get("tensor_count"),
        "elements": value.get("transformed_identity", {}).get("elements"),
        "tensor_bytes_preserved_exactly": value.get("transformed_identity", {}).get(
            "all_tensor_bytes_preserved_exactly"
        ),
    }
    if (
        content != core.canonical_sha256(compact)
        or value.get("schema") != "candidate-lora-vllm-stage-manifest-v44a"
        or value.get("arm") != logical
        or observed["weights_file_sha256"] != expected["weights"]
        or observed["adapter_config_file_sha256"] != expected["config"]
        or observed["manifest_file_sha256"] != expected["manifest_file"]
        or content != expected["manifest_content"]
        or observed["transformed_identity_sha256"] != expected["transformed_identity"]
        or observed["tensor_count"] != 70
        or observed["elements"] != 4_528_128
        or observed["tensor_bytes_preserved_exactly"] is not True
        or value.get("dataset_or_evaluation_accessed") is not False
        or value.get("shadow_ood_holdout_or_heldout_accessed") is not False
    ):
        raise RuntimeError(f"V49D {logical} staged adapter changed")
    return observed


def replica_stage_bindings_v49d() -> dict:
    logical = {
        name: canonical_stage_binding_v49d(name) for name in LOGICAL_CANDIDATES
    }
    return {
        arm: {**logical[name], "replica_arm": arm, "adapter_id": ADAPTER_IDS[arm]}
        for name, replicas in LOGICAL_REPLICAS.items() for arm in replicas
    }


def implementation_bindings_v49d() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_sft_v434_sampling_midpoint_ood_preregistration_v49d.py",
        "tests": ROOT / "test_sft_v434_sampling_midpoint_ood_v49d.py",
        "stage_runtime": Path(stage.__file__).resolve(),
        "canonical_stage_runtime": Path(stage.prior.__file__).resolve(),
        "core_runtime": Path(core.__file__).resolve(),
        "metric_runtime": Path(metrics.__file__).resolve(),
        "ood_gate_runtime": Path(ood_first.__file__).resolve(),
        "topology_runtime": Path(topology.__file__).resolve(),
        "cleanup_runtime": Path(cleanup.__file__).resolve(),
        "future_template": FUTURE_TEMPLATE,
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
    }
    for arm in LOGICAL_CANDIDATES:
        paths[f"{arm}_source_weights"] = stage.SOURCES[arm] / "adapter_model.safetensors"
        paths[f"{arm}_source_config"] = stage.SOURCES[arm] / "adapter_config.json"
        paths[f"{arm}_source_report"] = stage.REPORTS[arm]
        paths[f"{arm}_source_attempt"] = stage.ATTEMPTS[arm]
        paths[f"{arm}_source_gpu_log"] = stage.GPU_LOGS[arm]
        paths[f"{arm}_staged_weights"] = stage.OUTPUTS[arm] / "adapter_model.safetensors"
        paths[f"{arm}_staged_config"] = stage.OUTPUTS[arm] / "adapter_config.json"
        paths[f"{arm}_stage_manifest"] = stage.OUTPUTS[arm] / "stage_manifest_v44a.json"
    result = {name: core.file_sha256(path) for name, path in paths.items()}
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    result["environment"] = ood_first.environment.environment_bindings_v44b()
    result["source_seals"] = {
        arm: stage.source_seal_v49d(arm) for arm in LOGICAL_CANDIDATES
    }
    return result


def load_preregistration_v49d(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V49D OOD preregistration file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or content != core.canonical_sha256({
            k: v for k, v in value.items() if k != "content_sha256_before_self_field"
        })
        or value.get("schema")
        != "sft-v434-equal-vs-source50-replicated-ood-only-v49d"
        or value.get("status") != "preregistered_before_fresh_ood_only_evaluation"
        or value.get("evaluation_launch_authorized") is not True
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("shadow_access_authorized") is not False
        or value.get("single_access_inputs") != OOD_INPUTS
        or value.get("arms") != list(ARMS)
        or value.get("logical_candidates") != list(LOGICAL_CANDIDATES)
        or value.get("staged_adapters") != replica_stage_bindings_v49d()
        or value.get("implementation_bindings") != implementation_bindings_v49d()
        or value.get("extends_future_template", {}).get("file_sha256")
        != FUTURE_TEMPLATE_FILE_SHA256
        or value.get("extends_future_template", {}).get("content_sha256")
        != FUTURE_TEMPLATE_CONTENT_SHA256
    ):
        raise RuntimeError("V49D OOD preregistration content changed")
    core._forbid_holdout_v44a(item["path"] for item in OOD_INPUTS.values())
    return value


@contextmanager
def patched_arm_globals_v49d():
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
        arm: engine for wave in arm_wave_plan_v49d() for arm, engine in wave
    }
    core.arm_wave_plan_v44a = arm_wave_plan_v49d
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


def _assert_exact_bases(table: dict, label: str) -> dict:
    baseline = table["base_a"]
    if any(table[arm] != baseline for arm in BASE_ARMS[1:]):
        raise RuntimeError(f"V49D four-base exact equivalence failed on {label}")
    return {"label": label, "all_four_base_outputs_exact": True}


def _replica_gate(ood_qa, prose_details, raw_sink, arm):
    qa_gate = metrics.qa_ood_gate(ood_qa["base_a"], ood_qa[arm])
    qa_gate.update(ood_first.paired_qa_bootstrap_v45a(
        raw_sink["ood_qa"]["base_a"], raw_sink["ood_qa"][arm]
    ))
    prose_gate = metrics.prose_gate(prose_details["base_a"], prose_details[arm])
    counters = ood_qa[arm]["protocol_leak_counters"]
    base_counters = ood_qa["base_a"]["protocol_leak_counters"]
    protocol_safe = all(counters[key] <= base_counters[key] for key in base_counters)
    return {
        "arm": arm,
        "ood_qa": qa_gate,
        "ood_prose": prose_gate,
        "no_protocol_or_leak_counter_increase": protocol_safe,
        "eligible": qa_gate["passed"] and prose_gate["passed"] and protocol_safe,
    }


def _gate_table(ood_qa, prose_details, raw_sink):
    table = {}
    for logical, replicas in LOGICAL_REPLICAS.items():
        gates = [_replica_gate(ood_qa, prose_details, raw_sink, arm) for arm in replicas]
        table[logical] = {
            "replicas": list(replicas),
            "replica_gates": gates,
            "both_replicas_independently_ood_eligible": all(
                gate["eligible"] for gate in gates
            ),
        }
    equal = LOGICAL_REPLICAS["v434_equal"]
    source = LOGICAL_REPLICAS["v434_source50"]
    mean = lambda arms, field: sum(float(ood_qa[arm][field]) for arm in arms) / 2.0
    direct = {
        "source50_minus_equal_mean_reward": (
            mean(source, "generated_equal_unit_mean_reward")
            - mean(equal, "generated_equal_unit_mean_reward")
        ),
        "source50_minus_equal_mean_exact_count": (
            mean(source, "generated_exact_count") - mean(equal, "generated_exact_count")
        ),
    }
    direct["reward_nonnegative"] = direct[
        "source50_minus_equal_mean_reward"
    ] >= 0.0
    direct["exact_nonnegative"] = direct[
        "source50_minus_equal_mean_exact_count"
    ] >= 0.0
    direct["paired_bootstrap_ci_role"] = "informational_not_a_gate"
    return table, direct


class Phase:
    value = "setup"


def parser():
    return core.parser()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v49d(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "single_access_labels": sorted(prereg["single_access_inputs"]),
            "protected_semantic_access_count": 0,
            "shadow_opened": False,
            "heldout_or_holdout_opened": False,
            "gpu_accessed": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists() or REPORT.exists():
        raise RuntimeError("V49D OOD evaluation requires fresh artifact paths")
    ood_first.environment.environment_bindings_v44b()
    preflight = topology.gpu_preflight()
    attempt = core.self_hashed({
        "schema": "v49d-ood-only-attempt",
        "status": "launching",
        "phase": "before_model_or_ood_semantic_access",
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
    raw_sink = {"schema": "v49d-ood-only-raw-local"}
    started = time.monotonic()
    try:
        base.set_seed(core.GENERATION_SEED)
        with patched_arm_globals_v49d():
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
            saved_inputs = core.PROTECTED_INPUTS_V44A
            core.PROTECTED_INPUTS_V44A = OOD_INPUTS
            try:
                firewall = core.SingleSemanticAccessV44A(OOD_INPUTS)
                phase.value = "ood_qa"
                ood_qa = core.evaluate_qa_v44a(
                    trainer, metrics.qa_bundle(firewall.jsonl("ood_qa")),
                    raw_sink, "ood_qa",
                )
                phase.value = "ood_prose"
                ood_prose, prose_details = core.evaluate_prose_v44a(
                    trainer, firewall.jsonl("ood_prose"), raw_sink
                )
            finally:
                core.PROTECTED_INPUTS_V44A = saved_inputs
        if set(firewall.receipts) != set(OOD_INPUTS):
            raise RuntimeError("V49D OOD single-access coverage changed")
        base_equivalence = {
            "ood_qa": _assert_exact_bases(ood_qa, "ood_qa"),
            "ood_prose": _assert_exact_bases(ood_prose, "ood_prose"),
        }
        gates, direct = _gate_table(ood_qa, prose_details, raw_sink)
        raw_sink["single_access_receipts"] = firewall.receipts
        core.atomic_json(RAW, raw_sink, mode=0o600)
        raw_sha = core.file_sha256(RAW)
        stop.set(); monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("V49D OOD GPU monitor failed")
        gpu = metrics.summarize_gpu(GPU_LOG, pid_map)
        closed = cleanup.strict_close_trainer_v38a(trainer); trainer = None
        if (
            closed.get("engine_kill_count") != 4
            or closed.get("placement_group_remove_count") != 4
            or closed.get("all_four_gcs_states_removed") is not True
        ):
            raise RuntimeError("V49D exact four-engine cleanup changed")
        import ray
        ray.shutdown()
        idle = cleanup.wait_for_gpu_idle()
        report = core.self_hashed({
            "schema": "v49d-v434-equal-source50-ood-only-aggregate",
            "status": "complete_ood_only_shadow_and_holdout_unopened",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "arms": list(ARMS),
            "staged_adapters": prereg["staged_adapters"],
            "actor_identities": actor_ids,
            "worker_identities": worker_ids,
            "physical_gpu_pid_map": pid_map,
            "base_duplicate_equivalence": base_equivalence,
            "ood_qa": ood_qa,
            "ood_prose": ood_prose,
            "per_logical_candidate_gate_table": gates,
            "direct_hypothesis_ood_point_gates": direct,
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
            "shadow_opened": False,
            "heldout_or_holdout_opened": False,
            "selection_or_promotion_authorized": False,
        })
        core.atomic_json(REPORT, report)
        complete = dict(attempt)
        complete.pop("content_sha256_before_self_field", None)
        complete.update({
            "status": "complete", "phase": "ood_aggregate_sealed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protected_semantic_access_count": 2,
            "report": str(REPORT), "report_sha256": core.file_sha256(REPORT),
        })
        core.atomic_json(ATTEMPT.with_suffix(".complete.json"), core.self_hashed(complete))
        print(json.dumps({"report": str(REPORT), "report_sha256": core.file_sha256(REPORT)}, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        failure = core.self_hashed({
            "schema": "v49d-ood-only-failure",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "protected_semantic_access_count": 0 if firewall is None else len(firewall.receipts),
            "protected_semantic_access_labels": [] if firewall is None else sorted(firewall.receipts),
            "shadow_opened": False, "heldout_or_holdout_opened": False,
        })
        core.atomic_json(RUN_DIR / "failure_v49d.json", failure)
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
