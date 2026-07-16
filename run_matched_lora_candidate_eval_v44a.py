#!/usr/bin/env python3
"""Preregistered aggregate-only V42B/C/D SFT versus V43D LoRA-ES evaluation.

Four independent TP1 vLLM engines serve six logical arms in a frozen two-wave
schedule.  The four candidate arms are served only through content-addressed
LoRARequest artifacts in the proven
Qwen3.5 outer ``language_model`` namespace.  This runtime does not contain the
obsolete V39A dense layer-plan or full-weight snapshot installation path.
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

import run_eggroll_es_equal_unit_v38a as cleanup_v38a
import run_lora_topology_probe_v40a as v40a
import run_lora_topology_probe_v40c as v40c
import run_sealed_candidate_eval_v39a as v39a
import run_sft_train_only_control_v36a as hashing
import stage_candidate_adapters_vllm_v44a as staging
import train_eggroll_es_specialist as base


ROOT = Path(__file__).resolve().parent
MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
CANDIDATE_ARMS = tuple(staging.CANDIDATE_SPECS_V44A)
STAGED_BY_ARM = {
    arm: spec[2] for arm, spec in staging.CANDIDATE_SPECS_V44A.items()
}
ADAPTER_IDS_V44A = {arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS)}
ENGINE_INDEX_BY_ARM_V44A = {
    "base_a": 0, "base_b": 1,
    CANDIDATE_ARMS[0]: 2, CANDIDATE_ARMS[1]: 3,
    CANDIDATE_ARMS[2]: 2, CANDIDATE_ARMS[3]: 3,
}
SHADOW = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/"
    "fold_3_shadow_dev.jsonl"
).resolve()
SPLIT_MANIFEST = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/manifest_v37a.json"
).resolve()
OOD_QA = (ROOT / "data/ood_qa_v3.jsonl").resolve()
OOD_PROSE = (ROOT / "data/ood_prose_v3.jsonl").resolve()
TUNED_FOLDER = v40a.TUNED_FOLDER
TUNED_FILE = v40a.TUNED_FILE

EXPERIMENT = "v44a_matched_lora_sft_es_fold3_ood_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v44a.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v44a.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "matched_lora_sft_es_fold3_ood_eval_v44a.json"
).resolve()

BASE_ARMS = ("base_a", "base_b")
ARMS = BASE_ARMS + CANDIDATE_ARMS
GPU_IDS = (0, 1, 2, 3)
GENERATION_SEED = 20_260_715
BOOTSTRAP_SEED = 20_260_715
BOOTSTRAP_SAMPLES = 20_000
MODEL_SHARDS_CONTENT_SHA256 = (
    "af8ea3a900c04e97d2d8e3146b8e23be5ee3e6548dea20440020b2f43ee6656e"
)
PROTECTED_INPUTS_V44A = {
    "shadow": {
        "path": str(SHADOW),
        "file_sha256": "6d5b72f7506a752fd5275425739ec785e25f0ff486f5c03b68e91c8e99d7ebeb",
    },
    "split_manifest": {
        "path": str(SPLIT_MANIFEST),
        "file_sha256": "7d2a8f2b86f9007aa2bfe8ae043be15647451cc4bbea53a18d5915085879ee9d",
    },
    "ood_qa": {
        "path": str(OOD_QA),
        "file_sha256": "25a48b9494134731e51043047afadb340291a9ae3e9cfec9d9cfd8c73ddb255d",
    },
    "ood_prose": {
        "path": str(OOD_PROSE),
        "file_sha256": "3299457c7a23dfb0eb10408b2226b6231e291b519a52325feed607d901605e57",
    },
}
# Retry wrappers may install a preregistered, CPU-only parser preflight.  The
# hook receives the single-access firewall before any model/trainer exists and
# returns already parsed bundles so protected files are never read twice.
PRE_MODEL_PROTECTED_PREFLIGHT_V44A = None


def canonical_sha256(value: object) -> str:
    return hashing.canonical_sha256(value)


def file_sha256(path: Path) -> str:
    return hashing.file_sha256(Path(path))


def self_hashed(value: dict) -> dict:
    result = dict(value)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def atomic_json(path: Path, value: dict, mode: int | None = None) -> None:
    v39a.atomic_json(path, value, mode=mode)


def _forbid_holdout_v44a(paths) -> None:
    for path in paths:
        lowered = str(Path(path).resolve()).casefold()
        if "holdout" in lowered or "heldout" in lowered:
            raise RuntimeError("V44A holdout/heldout path access is forbidden")


def nonprotected_paths_v44a() -> dict[str, Path]:
    result = {
        "runtime": Path(__file__).resolve(),
        "staging_runtime": Path(staging.__file__).resolve(),
        "v39a_metric_runtime": Path(v39a.__file__).resolve(),
        "v40a_engine_runtime": Path(v40a.__file__).resolve(),
        "v40c_tuned_projection_runtime": Path(v40c.__file__).resolve(),
        "cleanup_runtime": Path(cleanup_v38a.__file__).resolve(),
        "qa_quality": ROOT / "qa_quality.py",
        "reward_runtime": ROOT / "train_eggroll_es_specialist.py",
        "model_config": MODEL / "config.json",
        "model_index": MODEL / "model.safetensors.index.json",
        "tuned_table": TUNED_FILE,
        "es_source_failure": staging.ES_FAILURE_V44A,
        "es_source_attempt": staging.ES_ATTEMPT_V44A,
        "es_source_preregistration": staging.ES_PREREG_V44A,
    }
    for arm, (source, report, staged) in staging.CANDIDATE_SPECS_V44A.items():
        result[f"{arm}_source_weights"] = source / "adapter_model.safetensors"
        result[f"{arm}_source_config"] = source / "adapter_config.json"
        if report is not None:
            result[f"{arm}_source_seal"] = report
        result[f"{arm}_staged_weights"] = staged / "adapter_model.safetensors"
        result[f"{arm}_staged_config"] = staged / "adapter_config.json"
        result[f"{arm}_stage_manifest"] = staged / "stage_manifest_v44a.json"
    return result


def implementation_bindings_v44a() -> dict[str, str]:
    paths = nonprotected_paths_v44a()
    _forbid_holdout_v44a(paths.values())
    result = {key: file_sha256(path) for key, path in paths.items()}
    index = json.loads((MODEL / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 26 or not all((MODEL / name).is_file() for name in shards):
        raise RuntimeError("V44A base model shard inventory changed")
    result["model_shards_content_sha256"] = MODEL_SHARDS_CONTENT_SHA256
    return result


def staged_adapter_bindings_v44a() -> dict:
    result = {}
    for arm, directory in STAGED_BY_ARM.items():
        manifest_path = directory / "stage_manifest_v44a.json"
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        content = value.get("content_sha256_before_self_field")
        compact = {key: item for key, item in value.items()
                   if key != "content_sha256_before_self_field"}
        if (
            content != canonical_sha256(compact)
            or value.get("schema") != "candidate-lora-vllm-stage-manifest-v44a"
            or value.get("status") != "complete_cpu_only_key_transform"
            or value.get("arm") != arm
            or value.get("implementation", {}).get("path")
            != str(Path(staging.__file__).resolve())
            or value.get("implementation", {}).get("file_sha256")
            != file_sha256(Path(staging.__file__).resolve())
            or value.get("transform", {}).get("target_prefix")
            != staging.TARGET_PREFIX_V44A
            or value.get("transformed_identity", {}).get("tensor_count") != 70
            or value.get("transformed_identity", {}).get("elements") != 4_528_128
            or value.get("transformed_identity", {}).get(
                "all_tensor_bytes_preserved_exactly"
            ) is not True
            or value.get("dataset_or_evaluation_accessed") is not False
            or value.get("shadow_ood_holdout_or_heldout_accessed") is not False
        ):
            raise RuntimeError(f"V44A staged adapter manifest changed: {arm}")
        artifact = value["artifact"]
        observed = {
            "directory": str(directory),
            "weights_file_sha256": file_sha256(
                directory / "adapter_model.safetensors"
            ),
            "adapter_config_file_sha256": file_sha256(
                directory / "adapter_config.json"
            ),
            "manifest_file_sha256": file_sha256(manifest_path),
            "manifest_content_sha256": content,
            "transformed_identity_sha256": value[
                "transformed_identity"
            ]["sha256"],
            "target_namespace": artifact["target_namespace"],
            "tensor_count": 70,
            "elements": 4_528_128,
            "tensor_bytes_preserved_exactly": True,
        }
        if (
            observed["weights_file_sha256"]
            != artifact["weights_file_sha256"]
            or observed["adapter_config_file_sha256"]
            != artifact["adapter_config_file_sha256"]
        ):
            raise RuntimeError(f"V44A staged adapter files changed: {arm}")
        result[arm] = observed
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def load_preregistration(args: argparse.Namespace) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V44A preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {key: item for key, item in value.items()
               if key != "content_sha256_before_self_field"}
    if (
        content != args.preregistration_content_sha256
        or content != canonical_sha256(compact)
        or value.get("schema") != "matched-lora-candidate-eval-preregistration-v44a"
        or value.get("status") != "preregistered_before_single_semantic_access"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("raw_shadow_or_ood_content_opened_before_preregistration")
        is not False
        or value.get("implementation_bindings") != implementation_bindings_v44a()
        or value.get("staged_adapters") != staged_adapter_bindings_v44a()
        or value.get("single_access_inputs") != PROTECTED_INPUTS_V44A
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    ):
        raise RuntimeError("V44A preregistration content changed")
    _forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    # Protected files are intentionally neither opened nor hashed here.
    return value


class SingleSemanticAccessV44A:
    def __init__(self, expected: dict[str, dict]):
        if expected != PROTECTED_INPUTS_V44A:
            raise RuntimeError("V44A protected input commitment changed")
        self.expected = expected
        self.receipts: dict[str, dict] = {}
        _forbid_holdout_v44a(item["path"] for item in expected.values())

    def _read(self, label: str) -> bytes:
        if label not in self.expected or label in self.receipts:
            raise RuntimeError(f"V44A single semantic access violation: {label}")
        item = self.expected[label]
        path = Path(item["path"]).resolve()
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != item["file_sha256"]:
            raise RuntimeError(f"V44A protected input identity changed: {label}")
        self.receipts[label] = {
            "path": str(path),
            "file_sha256": actual,
            "bytes": len(raw),
            "semantic_read_count": 1,
        }
        return raw

    def jsonl(self, label: str) -> list[dict]:
        raw = self._read(label)
        rows = [json.loads(line) for line in raw.decode().splitlines() if line]
        if not rows or not all(isinstance(row, dict) for row in rows):
            raise RuntimeError(f"V44A protected JSONL is invalid: {label}")
        self.receipts[label]["rows"] = len(rows)
        return rows

    def json(self, label: str) -> dict:
        value = json.loads(self._read(label).decode())
        if not isinstance(value, dict):
            raise RuntimeError(f"V44A protected JSON is invalid: {label}")
        return value


def make_trainer_v44a(prereg: dict):
    saved = (v40a.EXPERIMENT, v40a.RUN_DIR)
    v40a.EXPERIMENT = EXPERIMENT
    v40a.RUN_DIR = RUN_DIR
    try:
        trainer = v40c.make_trainer_v40c(prereg)
    except BaseException:
        v40a.EXPERIMENT, v40a.RUN_DIR = saved
        raise
    return trainer, saved


def actor_pid_map_v44a(actor_items: list[dict], worker_items: list[dict]) -> dict[int, int]:
    if len(actor_items) != 4 or len(worker_items) != 4:
        raise RuntimeError("V44A actor identity coverage changed")
    worker_pids = {int(item["pid"]): item for item in worker_items}
    result = {}
    for item in actor_items:
        gpu, pid = int(item["physical_gpu_id"]), int(item["pid"])
        if gpu in result or pid not in worker_pids:
            raise RuntimeError("V44A actor/worker identity mismatch")
        if worker_pids[pid]["cuda_visible_devices"] != str(gpu):
            raise RuntimeError("V44A actor/worker physical GPU mismatch")
        result[gpu] = pid
    if set(result) != set(GPU_IDS) or len(set(result.values())) != 4:
        raise RuntimeError("V44A physical GPU mapping incomplete")
    return result


def lora_request_v44a(arm: str):
    from vllm.lora.request import LoRARequest
    if arm not in CANDIDATE_ARMS:
        raise ValueError(f"V44A arm has no LoRARequest: {arm}")
    return LoRARequest(
        f"{arm}_vllm_v44a", ADAPTER_IDS_V44A[arm],
        str(STAGED_BY_ARM[arm]), base_model_name=str(MODEL),
    )


def arm_wave_plan_v44a() -> tuple[tuple[tuple[str, int], ...], ...]:
    """Frozen logical schedule: every GPU participates in every eval phase."""
    return (
        (("base_a", 0), ("base_b", 1),
         (CANDIDATE_ARMS[0], 2), (CANDIDATE_ARMS[1], 3)),
        ((CANDIDATE_ARMS[2], 2), (CANDIDATE_ARMS[3], 3)),
    )


def arm_requests_v44a(engines, prompts, params):
    handles = []
    if len(engines) != 4:
        raise RuntimeError("V44A requires exactly four TP1 engines")
    scheduled = tuple(item for wave in arm_wave_plan_v44a() for item in wave)
    if tuple(arm for arm, _ in scheduled) != ARMS:
        raise RuntimeError("V44A frozen arm schedule changed")
    for arm, engine_index in scheduled:
        kwargs = {"use_tqdm": False}
        if arm in CANDIDATE_ARMS:
            kwargs["lora_request"] = lora_request_v44a(arm)
        handles.append(
            engines[engine_index].generate.remote(list(prompts), params, **kwargs)
        )
    return handles


def _with_v44a_arm_globals(function, *args):
    saved_arms, saved_requests = v39a.ARMS, v39a._arm_requests
    v39a.ARMS = ARMS
    v39a._arm_requests = arm_requests_v44a
    try:
        return function(*args)
    finally:
        v39a.ARMS, v39a._arm_requests = saved_arms, saved_requests


def evaluate_qa_v44a(trainer, bundle: dict, raw_sink: dict, label: str) -> dict:
    return _with_v44a_arm_globals(
        v39a.evaluate_qa, trainer, bundle, raw_sink, label
    )


def evaluate_prose_v44a(trainer, rows: list[dict], raw_sink: dict):
    return _with_v44a_arm_globals(
        v39a.evaluate_prose, trainer, rows, raw_sink
    )


def shadow_bundle_v44a(rows: list[dict], manifest: dict) -> tuple[dict, dict]:
    fold = manifest.get("folds", [None] * 4)[3]
    intersections = fold.get("train_dev_edge_identity_intersections", {})
    if (
        not isinstance(intersections, dict)
        or not intersections
        or any(intersections.values())
        or fold.get("train", {}).get("rows") != 448
        or fold.get("shadow_dev", {}).get("rows") != 83
    ):
        raise RuntimeError("V44A shadow/train document-disjoint commitment changed")
    bundle = v39a.shadow_bundle(rows, manifest)
    proof = {
        "document_disjoint_from_fold3_train": True,
        "train_rows": 448,
        "shadow_rows": 83,
        "shadow_conflict_units": 51,
        "edge_identity_intersections": intersections,
        "split_manifest_file_sha256": PROTECTED_INPUTS_V44A[
            "split_manifest"
        ]["file_sha256"],
    }
    return bundle, proof


def selection_key_v44a(metrics: dict, arm: str) -> tuple:
    return (
        metrics["generated_equal_unit_mean_reward"],
        metrics["generated_exact_count"],
        metrics["generated_nonzero_count"],
        metrics["teacher_forced_equal_unit_mean_answer_logprob"],
        ADAPTER_IDS_V44A.get(arm, 0),
    )


def select_candidate_v44a(shadow_metrics: dict) -> dict:
    selected = max(
        CANDIDATE_ARMS,
        key=lambda arm: selection_key_v44a(shadow_metrics[arm], arm),
    )
    candidate = shadow_metrics[selected]
    baseline = shadow_metrics["base_a"]
    no_protocol_increase = all(
        candidate["protocol_leak_counters"][key]
        <= baseline["protocol_leak_counters"][key]
        for key in baseline["protocol_leak_counters"]
    )
    passed = (
        selection_key_v44a(candidate, selected)[:-1]
        > selection_key_v44a(baseline, "base_a")[:-1]
        and no_protocol_increase
    )
    return {
        "selected_arm": selected,
        "rule": (
            "lexicographic generated mean, exact, nonzero, teacher logprob; "
            "frozen candidate-order final tie (V43D highest)"
        ),
        "shadow_improvement_gate_passed": passed,
        "no_protocol_or_leak_counter_increase": no_protocol_increase,
    }


class PhaseV44A:
    value = "setup"


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "protected_semantic_access_count": 0,
            "heldout_or_holdout_opened": False,
        }, sort_keys=True))
        return 0
    if ATTEMPT.exists() or RUN_DIR.exists() or REPORT.exists():
        raise RuntimeError("V44A requires fresh artifact paths")
    preflight = v40a.gpu_preflight()
    attempt = self_hashed({
        "schema": "matched-lora-candidate-eval-attempt-v44a",
        "status": "launching",
        "phase": "before_model_or_protected_semantic_access",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "preflight": preflight,
        "protected_semantic_access_count": 0,
        "heldout_or_holdout_opened": False,
    })
    atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = monitor = saved = firewall = protected_preflight = None
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    phase = PhaseV44A()
    raw_sink = {"schema": "matched-lora-candidate-raw-local-v44a"}
    started = time.monotonic()
    try:
        if PRE_MODEL_PROTECTED_PREFLIGHT_V44A is not None:
            firewall = SingleSemanticAccessV44A(prereg["single_access_inputs"])
            protected_preflight = PRE_MODEL_PROTECTED_PREFLIGHT_V44A(
                firewall, prereg
            )
            if (
                not isinstance(protected_preflight, dict)
                or protected_preflight.get("status")
                != "complete_before_model_creation"
                or set(firewall.receipts) != set(PROTECTED_INPUTS_V44A)
            ):
                raise RuntimeError("V44A protected CPU preflight coverage changed")
        base.set_seed(GENERATION_SEED)
        trainer, saved = make_trainer_v44a(prereg)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        worker_ids = v40a._rpc_all(trainer, "runtime_identity_v40a")
        pid_map = actor_pid_map_v44a(actor_ids, worker_ids)
        monitor = threading.Thread(
            target=v39a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
        )
        monitor.start()

        if protected_preflight is None:
            firewall = SingleSemanticAccessV44A(prereg["single_access_inputs"])
            manifest = firewall.json("split_manifest")
            shadow_rows = firewall.jsonl("shadow")
            shadow_bundle, disjointness = shadow_bundle_v44a(
                shadow_rows, manifest
            )
            ood_qa_bundle = None
            ood_prose_rows = None
        else:
            shadow_bundle = protected_preflight["shadow_bundle"]
            disjointness = protected_preflight["document_disjointness"]
            ood_qa_bundle = protected_preflight["ood_qa_bundle"]
            ood_prose_rows = protected_preflight["ood_prose_rows"]
        phase.value = "shadow"
        shadow_metrics = evaluate_qa_v44a(
            trainer, shadow_bundle, raw_sink, "shadow"
        )
        selection = select_candidate_v44a(shadow_metrics)

        phase.value = "ood_qa"
        ood_qa_metrics = evaluate_qa_v44a(
            trainer,
            ood_qa_bundle or v39a.qa_bundle(firewall.jsonl("ood_qa")),
            raw_sink,
            "ood_qa",
        )
        phase.value = "ood_prose"
        ood_prose_metrics, ood_prose_details = evaluate_prose_v44a(
            trainer, ood_prose_rows or firewall.jsonl("ood_prose"), raw_sink
        )
        if set(firewall.receipts) != set(PROTECTED_INPUTS_V44A) or any(
            item["semantic_read_count"] != 1
            for item in firewall.receipts.values()
        ):
            raise RuntimeError("V44A protected semantic access coverage changed")
        selected = selection["selected_arm"]
        qa_gate = v39a.qa_ood_gate(
            ood_qa_metrics["base_a"], ood_qa_metrics[selected]
        )
        prose_gate = v39a.prose_gate(
            ood_prose_details["base_a"], ood_prose_details[selected]
        )
        final_gate = {
            "shadow_improvement": selection["shadow_improvement_gate_passed"],
            "ood_qa_no_degradation": qa_gate["passed"],
            "ood_prose_point_and_lcb_no_degradation": prose_gate["passed"],
        }
        final_gate["passed"] = all(final_gate.values())
        base_equivalence = {
            "shadow": shadow_metrics["base_a"] == shadow_metrics["base_b"],
            "ood_qa": ood_qa_metrics["base_a"] == ood_qa_metrics["base_b"],
            "ood_prose": (
                ood_prose_metrics["base_a"] == ood_prose_metrics["base_b"]
            ),
        }
        base_equivalence["all_splits"] = all(base_equivalence.values())
        if not base_equivalence["all_splits"]:
            raise RuntimeError("V44A base duplicate equivalence changed")

        raw_sink["selection"] = selection
        raw_sink["single_access_receipts"] = firewall.receipts
        atomic_json(RAW, raw_sink, mode=0o600)
        raw_sha = file_sha256(RAW)
        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("V44A GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = v39a.summarize_gpu(GPU_LOG, pid_map)
        cleanup = cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        if (
            cleanup.get("engine_kill_count") != 4
            or cleanup.get("placement_group_remove_count") != 4
            or cleanup.get("all_four_gcs_states_removed") is not True
        ):
            raise RuntimeError("V44A exact four-engine cleanup changed")
        import ray
        ray.shutdown()
        idle = cleanup_v38a.wait_for_gpu_idle()
        report = self_hashed({
            "schema": "matched-lora-candidate-eval-aggregate-v44a",
            "status": "complete_aggregate_only_no_heldout_access",
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
            "shadow": {
                "rows": 83,
                "conflict_units": 51,
                "document_disjointness": disjointness,
                "metrics": shadow_metrics,
            },
            "selection": selection,
            "ood_qa": ood_qa_metrics,
            "ood_qa_gate": qa_gate,
            "ood_prose": ood_prose_metrics,
            "ood_prose_gate": prose_gate,
            "final_gate": final_gate,
            "single_access_receipts": firewall.receipts,
            "cpu_preflight_before_model_creation": (
                None if protected_preflight is None
                else protected_preflight["aggregate_receipt"]
            ),
            "gpu_activity": gpu,
            "placement_group_cleanup": cleanup,
            "final_gpu_idle": idle,
            "preflight": preflight,
            "raw_local_artifact": {
                "path": str(RAW),
                "file_sha256": raw_sha,
                "mode": "0600",
                "git_eligible": False,
                "raw_content_in_aggregate": False,
            },
            "gpu_log": {
                "path": str(GPU_LOG),
                "file_sha256": file_sha256(GPU_LOG),
            },
            "heldout_or_holdout_opened": False,
            "raw_questions_answers_or_generations_persisted_in_aggregate": False,
            "obsolete_full_weight_layer_plan_or_snapshot_path_used": False,
        })
        atomic_json(REPORT, report)
        complete = dict(attempt)
        complete.pop("content_sha256_before_self_field", None)
        complete.update({
            "status": "complete",
            "phase": "aggregate_sealed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protected_semantic_access_count": 4,
            "report": str(REPORT),
            "report_sha256": file_sha256(REPORT),
        })
        atomic_json(ATTEMPT.with_suffix(".complete.json"), self_hashed(complete))
        print(json.dumps({
            "report": str(REPORT),
            "report_sha256": file_sha256(REPORT),
            "selected_arm": selected,
            "final_gate_passed": final_gate["passed"],
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        failure = self_hashed({
            "schema": "matched-lora-candidate-eval-failure-v44a",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "protected_semantic_access_count": (
                0 if firewall is None else len(firewall.receipts)
            ),
            "protected_semantic_access_labels": (
                [] if firewall is None else sorted(firewall.receipts)
            ),
            "heldout_or_holdout_opened": False,
        })
        atomic_json(RUN_DIR / "failure_v44a.json", failure)
        raise
    finally:
        if trainer is not None:
            try:
                base.close_trainer(trainer)
            except Exception:
                pass
        if saved is not None:
            v40a.EXPERIMENT, v40a.RUN_DIR = saved
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
