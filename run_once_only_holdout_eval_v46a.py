#!/usr/bin/env python3
"""Prepared once-only holdout evaluation of exact staged V42I versus base."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import threading
import time
import traceback
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v44c as parser_fix


ROOT = Path(__file__).resolve().parent
ARMS_V46A = ("base_a", "base_b", "base_c", "sft_v42i")
BASE_ARMS_V46A = ("base_a", "base_b", "base_c")
CANDIDATE_ARM_V46A = "sft_v42i"
STAGED_V42I = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v42i_sft_qwen35_vllm_namespace_v45d"
).resolve()
SOURCE_V42I = (
    ROOT / "experiments/sft_controls/"
    "v42i_matched_init_equal_unit_fold3_v412_lr5p5e5/"
    "middle_late_r32_seed17_init20260715041/final"
).resolve()
SOURCE_REPORT_V42I = (
    ROOT / "experiments/sft_controls/"
    "v42i_matched_init_equal_unit_fold3_v412_lr5p5e5/runtime_report_v42i.json"
).resolve()
EVAL_BUILD_REPORT = (ROOT / "data/eval_v3.report.json").resolve()
HOLDOUT_PATH = (ROOT / "data/eval_qa_v3.jsonl").resolve()
BOUNDARY_EVIDENCE = (
    ROOT / "experiments/eval_reports/"
    "sft_boundary_resolution_stability_evidence_v46a.json"
).resolve()
EXPERIMENT = "v46a_prepared_once_only_sft_v42i_sealed_holdout_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v46a.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "prepared_once_only_sft_v42i_sealed_holdout_eval_v46a.json"
).resolve()
PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "prepared_once_only_sft_v42i_sealed_holdout_eval_v46a.json"
).resolve()
EXPECTED_HOLDOUT = {
    "path": str(HOLDOUT_PATH),
    "file_sha256": "ab9a391e249910e876826dfab9c8e2f8e17a7b8695e6f018a3e515e5aa69603b",
    "bytes": 172209,
    "total_rows": 59,
    "heldout_rows": 18,
    "heldout_documents": 11,
    "validation_rows": 41,
    "validation_documents": 22,
    "heldout_sources": {
        "esinem": 3, "kinbakutoday": 8, "rope365": 6, "wikipedia": 1,
    },
    "heldout_quality_buckets": {
        "safety_relevant_grounded": 2, "standard_grounded": 16,
    },
    "split_seed": "specialist-eval-v3-s5-20260714",
    "eval_build_report_path": str(EVAL_BUILD_REPORT),
    "eval_build_report_file_sha256": (
        "245097c9fab935558b246d577c55c5fe3d64df534de8690e75256f10a8d05d9f"
    ),
}
BOUNDARY_EVIDENCE_FILE_SHA256 = (
    "7de60d7267dca4d4dea0bdb9bb5b2ed5a1fbd0fec5a714ba837460f47791caa1"
)
BOUNDARY_EVIDENCE_CONTENT_SHA256 = (
    "279eb1bb3892b47af7da7bc90bfd058e249506881e3191770ee826a714638cb6"
)
LAUNCH_INTERLOCK_RESOLVED_V46A = False
BOOTSTRAP_SAMPLES_V46A = 20_000
BOOTSTRAP_SEED_V46A = 20_260_715


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", required=True)
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def holdout_report_commitment_v46a() -> dict:
    if core.file_sha256(EVAL_BUILD_REPORT) != EXPECTED_HOLDOUT[
        "eval_build_report_file_sha256"
    ]:
        raise RuntimeError("V46A eval-v3 content-free build report changed")
    value = json.loads(EVAL_BUILD_REPORT.read_text())
    strata = value.get("strata", {})
    heldout, validation = strata.get("heldout", {}), strata.get("validation", {})
    collisions = value.get("disjointness", {}).get("collisions", {})
    output = value.get("outputs", {}).get("domain_eval", {})
    if (
        value.get("schema") != "specialist-eval-v3-build-report-v1"
        or output != {
            "bytes": EXPECTED_HOLDOUT["bytes"],
            "rows": EXPECTED_HOLDOUT["total_rows"],
            "sha256": EXPECTED_HOLDOUT["file_sha256"],
        }
        or value.get("parameters", {}).get("split_seed")
        != EXPECTED_HOLDOUT["split_seed"]
        or heldout.get("rows") != EXPECTED_HOLDOUT["heldout_rows"]
        or heldout.get("documents") != EXPECTED_HOLDOUT["heldout_documents"]
        or heldout.get("source") != EXPECTED_HOLDOUT["heldout_sources"]
        or heldout.get("quality_bucket")
        != EXPECTED_HOLDOUT["heldout_quality_buckets"]
        or validation.get("rows") != EXPECTED_HOLDOUT["validation_rows"]
        or validation.get("documents")
        != EXPECTED_HOLDOUT["validation_documents"]
        or value.get("disjointness", {}).get("passed") is not True
        or any(collisions.get(label) != [] for label in (
            "candidate_train_vs_domain", "candidate_train_vs_ood_prose",
            "domain_vs_ood_prose", "validation_vs_heldout",
        ))
        or value.get("metric_policy", {}).get("sealed_holdout", {}).get(
            "allowed_uses_per_frozen_cycle"
        ) != 1
        or value.get("metric_policy", {}).get("sealed_holdout", {}).get(
            "selection_use"
        ) != "prohibited"
    ):
        raise RuntimeError("V46A sealed holdout commitment changed")
    return {
        **EXPECTED_HOLDOUT,
        "document_disjoint_from_candidate_training_by_normalized_source_url": True,
        "validation_vs_heldout_document_intersection_empty": True,
        "content_free_commitment_only": True,
        "holdout_file_opened_or_hashed": False,
    }


def boundary_evidence_v46a() -> dict:
    if core.file_sha256(BOUNDARY_EVIDENCE) != BOUNDARY_EVIDENCE_FILE_SHA256:
        raise RuntimeError("V46A SFT boundary evidence file changed")
    value = json.loads(BOUNDARY_EVIDENCE.read_text())
    identity = value.get("current_sft_control", {})
    scope = value.get("replication_scope", {})
    if (
        value.get("content_sha256_before_self_field")
        != BOUNDARY_EVIDENCE_CONTENT_SHA256
        or _compact_sha(value) != BOUNDARY_EVIDENCE_CONTENT_SHA256
        or identity.get("arm") != CANDIDATE_ARM_V46A
        or identity.get("staged_weights_sha256")
        != "79207dd2c0b46aaef4af5933aaac9fbbaf837db91241ab9d352e652b5c53afad"
        or identity.get("transformed_identity_sha256")
        != "d185cbe52414054759188334fd38b96dbda601957bc8931256fd1e3c0fe71041"
        or scope.get("v42g_has_three_repetitions") is not True
        or scope.get("v42i_repeated_stability_claimed") is not False
        or scope.get("v43i_ood_first_resolution_required_before_holdout_launch")
        is not True
        or value.get("heldout_or_holdout_opened") is not False
    ):
        raise RuntimeError("V46A SFT boundary evidence content changed")
    return {
        "path": str(BOUNDARY_EVIDENCE),
        "file_sha256": BOUNDARY_EVIDENCE_FILE_SHA256,
        "content_sha256": BOUNDARY_EVIDENCE_CONTENT_SHA256,
        "current_sft_control": "sft_v42i",
        "v42g_repeat_stability": "3/3",
        "v42i_boundary_evaluation_count": 1,
        "v43i_ood_first_resolution_required_before_holdout_launch": True,
        "heldout_or_holdout_opened": False,
    }


def nonprotected_bindings_v46a() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "core_runtime": Path(core.__file__).resolve(),
        "source_faithful_parser_runtime": Path(parser_fix.__file__).resolve(),
        "v39a_metric_runtime": Path(core.v39a.__file__).resolve(),
        "cleanup_runtime": Path(core.cleanup_v38a.__file__).resolve(),
        "engine_runtime": Path(core.v40a.__file__).resolve(),
        "trainer_projection_runtime": Path(core.v40c.__file__).resolve(),
        "qa_quality": ROOT / "qa_quality.py",
        "reward_runtime": ROOT / "train_eggroll_es_specialist.py",
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
        "source_report": SOURCE_REPORT_V42I,
        "source_weights": SOURCE_V42I / "adapter_model.safetensors",
        "source_config": SOURCE_V42I / "adapter_config.json",
        "staged_weights": STAGED_V42I / "adapter_model.safetensors",
        "staged_config": STAGED_V42I / "adapter_config.json",
        "stage_manifest": STAGED_V42I / "stage_manifest_v44a.json",
        "eval_build_report": EVAL_BUILD_REPORT,
        "boundary_evidence": BOUNDARY_EVIDENCE,
    }
    if HOLDOUT_PATH in paths.values():
        raise RuntimeError("V46A holdout may not be a nonprotected binding")
    result = {label: core.file_sha256(path) for label, path in paths.items()}
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    return result


@contextmanager
def patched_candidate_globals_v46a():
    saved = {
        "BASE": core.BASE_ARMS, "CANDIDATE": core.CANDIDATE_ARMS,
        "ARMS": core.ARMS, "STAGED": core.STAGED_BY_ARM,
        "IDS": core.ADAPTER_IDS_V44A, "ENGINE": core.ENGINE_INDEX_BY_ARM_V44A,
        "wave": core.arm_wave_plan_v44a,
    }
    core.BASE_ARMS = BASE_ARMS_V46A
    core.CANDIDATE_ARMS = (CANDIDATE_ARM_V46A,)
    core.ARMS = ARMS_V46A
    core.STAGED_BY_ARM = {CANDIDATE_ARM_V46A: STAGED_V42I}
    core.ADAPTER_IDS_V44A = {CANDIDATE_ARM_V46A: 1}
    core.ENGINE_INDEX_BY_ARM_V44A = {
        arm: index for index, arm in enumerate(ARMS_V46A)
    }
    core.arm_wave_plan_v44a = lambda: (
        tuple((arm, index) for index, arm in enumerate(ARMS_V46A)),
    )
    try:
        yield
    finally:
        core.BASE_ARMS = saved["BASE"]
        core.CANDIDATE_ARMS = saved["CANDIDATE"]
        core.ARMS = saved["ARMS"]
        core.STAGED_BY_ARM = saved["STAGED"]
        core.ADAPTER_IDS_V44A = saved["IDS"]
        core.ENGINE_INDEX_BY_ARM_V44A = saved["ENGINE"]
        core.arm_wave_plan_v44a = saved["wave"]


def staged_candidate_binding_v46a() -> dict:
    with patched_candidate_globals_v46a():
        value = core.staged_adapter_bindings_v44a()
    if tuple(value) != (CANDIDATE_ARM_V46A,):
        raise RuntimeError("V46A fixed staged candidate changed")
    return value[CANDIDATE_ARM_V46A]


def load_preregistration_v46a(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V46A preregistration file changed")
    value = json.loads(path.read_text())
    if (
        value.get("content_sha256_before_self_field")
        != args.preregistration_content_sha256
        or _compact_sha(value) != args.preregistration_content_sha256
        or value.get("schema")
        != "prepared-once-only-fixed-sft-v42i-holdout-preregistration-v46a"
        or value.get("status")
        != "preregistered_before_once_only_holdout_access"
        or value.get("fixed_arms") != list(ARMS_V46A)
        or value.get("candidate_selection_permitted") is not False
        or value.get("post_result_tuning_or_selection_permitted") is not False
        or value.get("holdout_access_authorized_once") is not True
        or value.get("holdout_access_count_before_preregistration") != 0
        or value.get("holdout_commitment") != holdout_report_commitment_v46a()
        or value.get("boundary_evidence") != boundary_evidence_v46a()
        or value.get("launch_interlock") != {
            "resolved": False,
            "reason": "V43I OOD-first comparison is not yet resolved",
            "real_launch_permitted": False,
            "resolution_requires_new_runtime_and_preregistration_seal": True,
        }
        or value.get("fixed_staged_candidate") != staged_candidate_binding_v46a()
        or value.get("implementation_bindings") != nonprotected_bindings_v46a()
        or value.get("runtime", {}).get("tuned_table_content_sha256")
        != "4c4a0d4bbb400ea1d881bea3aae144d6865c34199fbb67889eda9e92d3a2543d"
    ):
        raise RuntimeError("V46A preregistration content changed")
    # Deliberately do not stat, hash, or open HOLDOUT_PATH here.
    return value


class SingleHoldoutAccessV46A:
    def __init__(self, expected: dict):
        if expected != holdout_report_commitment_v46a():
            raise RuntimeError("V46A holdout commitment changed")
        self.expected = expected
        self.receipt: dict | None = None

    def jsonl_once(self) -> list[dict]:
        if self.receipt is not None:
            raise RuntimeError("V46A holdout single-access violation")
        path = Path(self.expected["path"]).resolve()
        raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if actual != self.expected["file_sha256"] or len(raw) != self.expected["bytes"]:
            raise RuntimeError("V46A holdout file identity changed")
        rows = [json.loads(line) for line in raw.decode().splitlines() if line]
        if len(rows) != self.expected["total_rows"] or not all(
            isinstance(row, dict) for row in rows
        ):
            raise RuntimeError("V46A holdout container changed")
        self.receipt = {
            "path": str(path), "file_sha256": actual, "bytes": len(raw),
            "rows": len(rows), "semantic_read_count": 1,
        }
        return rows


def holdout_bundle_v46a(rows: list[dict]) -> tuple[dict, list[dict], dict]:
    required = {
        "answer", "item_id", "normalized_source_url", "quality_bucket",
        "question", "source", "split", "url",
    }
    if len(rows) != EXPECTED_HOLDOUT["total_rows"]:
        raise RuntimeError("V46A eval-v3 row count changed")
    if any(not required.issubset(row) for row in rows):
        raise RuntimeError("V46A eval-v3 row schema changed")
    validation = [row for row in rows if row.get("split") == "validation"]
    heldout = [row for row in rows if row.get("split") == "heldout"]
    if len(validation) + len(heldout) != len(rows):
        raise RuntimeError("V46A eval-v3 split labels changed")
    validation_urls = {row["normalized_source_url"] for row in validation}
    heldout_urls = {row["normalized_source_url"] for row in heldout}
    item_ids = [row["item_id"] for row in heldout]
    if (
        len(validation) != EXPECTED_HOLDOUT["validation_rows"]
        or len(validation_urls) != EXPECTED_HOLDOUT["validation_documents"]
        or len(heldout) != EXPECTED_HOLDOUT["heldout_rows"]
        or len(heldout_urls) != EXPECTED_HOLDOUT["heldout_documents"]
        or validation_urls & heldout_urls
        or len(set(item_ids)) != len(item_ids)
        or dict(sorted(Counter(row["source"] for row in heldout).items()))
        != EXPECTED_HOLDOUT["heldout_sources"]
        or dict(sorted(Counter(
            row["quality_bucket"] for row in heldout
        ).items())) != EXPECTED_HOLDOUT["heldout_quality_buckets"]
    ):
        raise RuntimeError("V46A heldout disjointness/strata changed")
    pairs = [parser_fix.qa_pair_from_record_v44c(row) for row in heldout]
    identities = [
        core.canonical_sha256({"question": question, "answer": answer})
        for question, answer in pairs
    ]
    if len(set(identities)) != len(identities):
        raise RuntimeError("V46A heldout QA identities repeated")
    bundle = {
        "questions": [pair[0] for pair in pairs],
        "answers": [pair[1] for pair in pairs],
        "item_sha256": identities,
        "weights": [1.0 / len(heldout)] * len(heldout),
        "rows": len(heldout), "units": len(heldout),
    }
    metadata = [{
        "item_sha256": identity,
        "item_id_sha256": hashlib.sha256(row["item_id"].encode()).hexdigest(),
        "document_sha256": hashlib.sha256(
            row["normalized_source_url"].encode()
        ).hexdigest(),
        "source": row["source"],
        "quality_bucket": row["quality_bucket"],
    } for row, identity in zip(heldout, bundle["item_sha256"], strict=True)]
    proof = {
        "total_container_rows": len(rows),
        "heldout_rows": len(heldout),
        "heldout_documents": len(heldout_urls),
        "validation_rows": len(validation),
        "validation_documents": len(validation_urls),
        "validation_vs_heldout_document_intersection_count": 0,
        "source_counts": EXPECTED_HOLDOUT["heldout_sources"],
        "quality_bucket_counts": EXPECTED_HOLDOUT["heldout_quality_buckets"],
        "eval_build_report_file_sha256": EXPECTED_HOLDOUT[
            "eval_build_report_file_sha256"
        ],
        "container_file_sha256": EXPECTED_HOLDOUT["file_sha256"],
    }
    return bundle, metadata, proof


def stratified_aggregate_v46a(raw_by_arm: dict, metadata: list[dict]) -> dict:
    result = {}
    for arm in ARMS_V46A:
        rows = raw_by_arm[arm]
        if len(rows) != len(metadata) or any(
            row["item_sha256"] != meta["item_sha256"]
            for row, meta in zip(rows, metadata, strict=True)
        ):
            raise RuntimeError("V46A raw/stratum item alignment changed")
        arm_result = {}
        for dimension in ("source", "quality_bucket"):
            dimension_result = {}
            for label in sorted({meta[dimension] for meta in metadata}):
                selected = [
                    row for row, meta in zip(rows, metadata, strict=True)
                    if meta[dimension] == label
                ]
                rewards = [float(row["reward"]) for row in selected]
                dimension_result[label] = {
                    "rows": len(selected),
                    "mean_reward": math.fsum(rewards) / len(rewards),
                    "exact_count": sum(row["format"] == "exact" for row in selected),
                    "exact_rate": sum(row["format"] == "exact" for row in selected) / len(selected),
                    "nonzero_count": sum(value > 0 for value in rewards),
                    "nonzero_rate": sum(value > 0 for value in rewards) / len(selected),
                }
            arm_result[dimension] = dimension_result
        result[arm] = arm_result
    return result


def paired_bootstrap_v46a(base_rows: list[dict], candidate_rows: list[dict]) -> dict:
    if len(base_rows) != len(candidate_rows) or not base_rows:
        raise RuntimeError("V46A paired rows changed")
    deltas = []
    for base, candidate in zip(base_rows, candidate_rows, strict=True):
        if base["item_sha256"] != candidate["item_sha256"]:
            raise RuntimeError("V46A paired item identities changed")
        deltas.append((
            float(candidate["reward"]) - float(base["reward"]),
            int(candidate["format"] == "exact") - int(base["format"] == "exact"),
            int(float(candidate["reward"]) > 0) - int(float(base["reward"]) > 0),
        ))
    rng = random.Random(BOOTSTRAP_SEED_V46A)
    sampled = [[], [], []]
    for _ in range(BOOTSTRAP_SAMPLES_V46A):
        indices = [rng.randrange(len(deltas)) for _ in deltas]
        for metric in range(3):
            sampled[metric].append(
                math.fsum(deltas[index][metric] for index in indices)
                / len(indices)
            )
    names = ("reward_mean_delta", "exact_rate_delta", "nonzero_rate_delta")
    return {
        f"{name}_paired_item_bootstrap_95_ci": [
            core.base.linear_percentile(values, 0.025),
            core.base.linear_percentile(values, 0.975),
        ] for name, values in zip(names, sampled, strict=True)
    } | {
        "bootstrap": {
            "unit": "sealed_holdout_item", "item_count": len(deltas),
            "samples": BOOTSTRAP_SAMPLES_V46A, "seed": BOOTSTRAP_SEED_V46A,
            "percentiles": [0.025, 0.975], "informational_only": True,
        }
    }


def fixed_gate_v46a(metrics: dict, raw_by_arm: dict) -> dict:
    baseline, candidate = metrics["base_a"], metrics[CANDIDATE_ARM_V46A]
    counters_safe = all(
        candidate["protocol_leak_counters"][key]
        <= baseline["protocol_leak_counters"][key]
        for key in baseline["protocol_leak_counters"]
    )
    result = {
        "candidate_arm": CANDIDATE_ARM_V46A,
        "baseline_arm": "base_a",
        "mean_reward_delta": (
            candidate["generated_row_mean_reward"]
            - baseline["generated_row_mean_reward"]
        ),
        "exact_count_delta": (
            candidate["generated_exact_count"] - baseline["generated_exact_count"]
        ),
        "nonzero_count_delta": (
            candidate["generated_nonzero_count"]
            - baseline["generated_nonzero_count"]
        ),
        "mean_reward_point_non_degradation": (
            candidate["generated_row_mean_reward"]
            >= baseline["generated_row_mean_reward"]
        ),
        "exact_count_point_non_degradation": (
            candidate["generated_exact_count"] >= baseline["generated_exact_count"]
        ),
        "nonzero_count_point_non_degradation": (
            candidate["generated_nonzero_count"]
            >= baseline["generated_nonzero_count"]
        ),
        "no_protocol_or_leak_counter_increase": counters_safe,
        "teacher_forced_metric_is_informational": True,
        "bootstrap_cis_are_informational": True,
        "candidate_selection_performed": False,
        **paired_bootstrap_v46a(
            raw_by_arm["base_a"], raw_by_arm[CANDIDATE_ARM_V46A]
        ),
    }
    result["passed"] = all(result[key] for key in (
        "mean_reward_point_non_degradation",
        "exact_count_point_non_degradation",
        "nonzero_count_point_non_degradation",
        "no_protocol_or_leak_counter_increase",
    ))
    return result


class PhaseV46A:
    value = "setup"


def summarize_gpu_v46a(path: Path, expected_pids: dict[int, int]) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    result = {}
    for gpu in core.GPU_IDS:
        selected = [row for row in rows if row["gpu"] == gpu]
        phase_rows = [row for row in selected if row["phase"] == "sealed_holdout"]
        resident = [
            row for row in phase_rows if expected_pids[gpu] in row["compute_pids"]
        ]
        positive = [row for row in resident if row["utilization_percent"] > 0]
        if (
            not resident or not positive
            or any(row["foreign_compute_pids"] for row in selected)
            or any(row["expected_pid"] != expected_pids[gpu] for row in selected)
        ):
            raise RuntimeError(f"V46A GPU {gpu} holdout activity failed")
        result[str(gpu)] = {
            "expected_pid": expected_pids[gpu],
            "samples": len(phase_rows), "resident_samples": len(resident),
            "positive_samples": len(positive),
            "peak_utilization_percent": max(
                row["utilization_percent"] for row in resident
            ),
            "peak_memory_used_mib": max(row["memory_used_mib"] for row in resident),
        }
    return {"all_four_attributed_positive_on_sealed_holdout": True, "by_gpu": result}


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v46a(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "protected_semantic_access_count": 0,
            "holdout_opened_or_hashed": False,
            "candidate_selection_performed": False,
            "real_launch_permitted": False,
            "launch_interlock": "V43I OOD-first comparison unresolved",
        }, sort_keys=True))
        return 0
    if not LAUNCH_INTERLOCK_RESOLVED_V46A:
        raise RuntimeError(
            "V46A real launch is sealed off until V43I OOD-first resolution; "
            "a new runtime and preregistration seal are required"
        )
    if ATTEMPT.exists() or RUN_DIR.exists() or REPORT.exists():
        raise RuntimeError("V46A once-only attempt has already been consumed")
    preflight = core.v40a.gpu_preflight()
    attempt = core.self_hashed({
        "schema": "once-only-fixed-sft-v42i-holdout-attempt-v46a",
        "status": "launching_irrevocable_attempt",
        "phase": "before_model_or_holdout_access",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "preregistration_file_sha256": args.preregistration_sha256,
        "preregistration_content_sha256": args.preregistration_content_sha256,
        "fixed_arms": list(ARMS_V46A),
        "candidate_selection_performed": False,
        "holdout_semantic_access_count": 0,
        "holdout_opened": False,
        "preflight": preflight,
    })
    core.atomic_json(ATTEMPT, attempt)
    RUN_DIR.mkdir(parents=True)
    trainer = saved = monitor = firewall = None
    phase = PhaseV46A()
    stop = threading.Event()
    failures: queue.Queue = queue.Queue()
    started = time.monotonic()
    try:
        core.base.set_seed(core.GENERATION_SEED)
        core.EXPERIMENT, core.RUN_DIR = EXPERIMENT, RUN_DIR
        trainer, saved = core.make_trainer_v44a(prereg)
        actor_ids = trainer._resolve([
            engine.runtime_identity_v40a.remote() for engine in trainer.engines
        ])
        worker_ids = core.v40a._rpc_all(trainer, "runtime_identity_v40a")
        pid_map = core.actor_pid_map_v44a(actor_ids, worker_ids)
        monitor = threading.Thread(
            target=core.v39a.monitor_gpus,
            args=(stop, phase, pid_map, GPU_LOG, failures), daemon=True,
        )
        monitor.start()

        firewall = SingleHoldoutAccessV46A(prereg["holdout_commitment"])
        rows = firewall.jsonl_once()
        bundle, metadata, disjointness = holdout_bundle_v46a(rows)
        del rows
        phase.value = "sealed_holdout"
        raw_sink: dict = {}
        with patched_candidate_globals_v46a():
            metrics = core.evaluate_qa_v44a(
                trainer, bundle, raw_sink, "sealed_holdout"
            )
        baseline = metrics["base_a"]
        if any(metrics[arm] != baseline for arm in BASE_ARMS_V46A[1:]):
            raise RuntimeError("V46A three-base exact equivalence failed")
        raw_by_arm = raw_sink.pop("sealed_holdout")
        if raw_sink:
            raise RuntimeError("V46A unexpected raw sink content")
        stratified = stratified_aggregate_v46a(raw_by_arm, metadata)
        fixed_gate = fixed_gate_v46a(metrics, raw_by_arm)
        del raw_by_arm, bundle, metadata

        stop.set()
        monitor.join(timeout=10)
        if monitor.is_alive() or not failures.empty():
            raise RuntimeError("V46A GPU monitor failed") from (
                failures.get() if not failures.empty() else None
            )
        gpu = summarize_gpu_v46a(GPU_LOG, pid_map)
        cleanup = core.cleanup_v38a.strict_close_trainer_v38a(trainer)
        trainer = None
        if (
            cleanup.get("engine_kill_count") != 4
            or cleanup.get("placement_group_remove_count") != 4
            or cleanup.get("all_four_gcs_states_removed") is not True
        ):
            raise RuntimeError("V46A exact four-engine cleanup changed")
        import ray
        ray.shutdown()
        idle = core.cleanup_v38a.wait_for_gpu_idle()
        report = core.self_hashed({
            "schema": "once-only-fixed-sft-v42i-holdout-aggregate-v46a",
            "status": "complete_once_only_aggregate_no_selection",
            "started_at_utc": attempt["started_at_utc"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_runtime_seconds": time.monotonic() - started,
            "preregistration": {
                "path": str(Path(args.preregistration).resolve()),
                "file_sha256": args.preregistration_sha256,
                "content_sha256": args.preregistration_content_sha256,
            },
            "fixed_arms": list(ARMS_V46A),
            "fixed_candidate_arm": CANDIDATE_ARM_V46A,
            "fixed_staged_candidate": prereg["fixed_staged_candidate"],
            "candidate_selection_performed": False,
            "post_result_tuning_or_selection_permitted": False,
            "base_duplicate_equivalence": {
                "base_arms": list(BASE_ARMS_V46A),
                "exact": True,
            },
            "holdout_identity": prereg["holdout_commitment"],
            "single_access_receipt": firewall.receipt,
            "document_disjointness": disjointness,
            "metrics": metrics,
            "stratified_metrics": stratified,
            "fixed_non_degradation_gate": fixed_gate,
            "actor_identities": actor_ids,
            "worker_identities": worker_ids,
            "physical_gpu_pid_map": pid_map,
            "gpu_activity": gpu,
            "placement_group_cleanup": cleanup,
            "final_gpu_idle": idle,
            "preflight": preflight,
            "gpu_log": {
                "path": str(GPU_LOG),
                "file_sha256": core.file_sha256(GPU_LOG),
            },
            "holdout_opened": True,
            "holdout_semantic_access_count": 1,
            "raw_questions_answers_or_generations_persisted": False,
        })
        core.atomic_json(REPORT, report)
        complete = dict(attempt)
        complete.pop("content_sha256_before_self_field", None)
        complete.update({
            "status": "complete_once_only_holdout_consumed",
            "phase": "aggregate_sealed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "holdout_semantic_access_count": 1,
            "holdout_opened": True,
            "report": str(REPORT),
            "report_file_sha256": core.file_sha256(REPORT),
        })
        core.atomic_json(ATTEMPT.with_suffix(".complete.json"), core.self_hashed(complete))
        print(json.dumps({
            "report": str(REPORT),
            "report_file_sha256": core.file_sha256(REPORT),
            "fixed_candidate": CANDIDATE_ARM_V46A,
            "fixed_gate_passed": fixed_gate["passed"],
            "candidate_selection_performed": False,
            "holdout_semantic_access_count": 1,
        }, sort_keys=True))
        return 0
    except BaseException as error:
        stop.set()
        if monitor is not None:
            monitor.join(timeout=10)
        failure = core.self_hashed({
            "schema": "once-only-fixed-sft-v42i-holdout-failure-v46a",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(),
            "candidate_selection_performed": False,
            "holdout_semantic_access_count": int(
                firewall is not None and firewall.receipt is not None
            ),
            "holdout_opened": bool(
                firewall is not None and firewall.receipt is not None
            ),
            "attempt_is_irrevocably_consumed": True,
        })
        core.atomic_json(RUN_DIR / "failure_v46a.json", failure)
        raise
    finally:
        if trainer is not None:
            try:
                core.base.close_trainer(trainer)
            except Exception:
                pass
        if saved is not None:
            core.v40a.EXPERIMENT, core.v40a.RUN_DIR = saved
        try:
            import ray
            ray.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
