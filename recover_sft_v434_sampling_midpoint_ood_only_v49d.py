#!/usr/bin/env python3
"""Recover the V49D OOD-only aggregate from immutable local receipts.

This runtime is deliberately CPU-only.  It never opens the OOD QA/prose
inputs, creates a model, or performs generation.  The only semantic material
it consumes is the already sealed raw receipt written by retry1 before the
legacy telemetry summarizer incorrectly required a ``shadow`` phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v45a as paired_qa
import run_sealed_candidate_eval_v39a as metric_runtime
import train_eggroll_es_specialist as reward_runtime


ROOT = Path(__file__).resolve().parent
SOURCE_EXPERIMENT = "v49d_v434_equal_vs_source50_replicated_ood_only_retry1"
SOURCE_RUN_DIR = (
    ROOT / "experiments/eggroll_es_hpo/runs" / SOURCE_EXPERIMENT
).resolve()
SOURCE_ATTEMPT = (
    SOURCE_RUN_DIR.parent / f".{SOURCE_EXPERIMENT}.attempt.json"
).resolve()
SOURCE_FAILURE = (SOURCE_RUN_DIR / "failure_v49d.json").resolve()
SOURCE_RAW = (SOURCE_RUN_DIR / "raw_items_v49d.json").resolve()
SOURCE_GPU_LOG = (SOURCE_RUN_DIR / "gpu_activity_v49d.jsonl").resolve()
SOURCE_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v434_equal_vs_source50_replicated_ood_only_retry1_v49d.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v434_equal_vs_source50_replicated_ood_only_retry1_offline_recovery_v49d.json"
).resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "sft_v434_equal_vs_source50_replicated_ood_only_retry1_offline_recovery_v49d.json"
).resolve()

BASE_ARMS = ("base_a", "base_b", "base_c", "base_d")
LOGICAL_REPLICAS = {
    "v434_equal": ("v434_equal_a", "v434_equal_b"),
    "v434_source50": ("v434_source50_a", "v434_source50_b"),
}
CANDIDATE_ARMS = tuple(
    arm for replicas in LOGICAL_REPLICAS.values() for arm in replicas
)
ARMS = BASE_ARMS + CANDIDATE_ARMS
PHASES = ("ood_qa", "ood_prose")
ROW_COUNTS = {"ood_qa": 24, "ood_prose": 16}
WAVES = (
    tuple((arm, index) for index, arm in enumerate(BASE_ARMS)),
    tuple((arm, index) for index, arm in enumerate(CANDIDATE_ARMS)),
)
GPU_IDS = (0, 1, 2, 3)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20_260_715
QA_IDENTITY_FIELDS = ("item_index", "item_sha256", "question", "answer")
PROSE_ALIGNMENT_FIELDS = (
    "item_id", "normalized_source_url", "text_sha256", "token_ids_sha256",
    "prompt_token_count", "scored_token_count",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def self_hashed(value: dict) -> dict:
    result = dict(value)
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.link(temporary, path)
    temporary.unlink()


def _load_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"V49D expected JSON object: {path}")
    return value


def _validate_self_hash(value: dict, label: str) -> str:
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if content != canonical_sha256(compact):
        raise RuntimeError(f"V49D {label} self hash changed")
    return content


def implementation_bindings_v49d() -> dict:
    paths = {
        "recovery_runtime": Path(__file__).resolve(),
        "recovery_builder": ROOT / "build_sft_v434_ood_offline_recovery_preregistration_v49d.py",
        "recovery_tests": ROOT / "test_sft_v434_ood_offline_recovery_v49d.py",
        "source_runtime": ROOT / "run_sft_v434_sampling_midpoint_ood_only_v49d.py",
        "metric_runtime": Path(metric_runtime.__file__).resolve(),
        "paired_qa_runtime": Path(paired_qa.__file__).resolve(),
        "reward_runtime": Path(reward_runtime.__file__).resolve(),
    }
    return {name: file_sha256(path) for name, path in paths.items()}


def _identity_rows(raw: dict, phase: str, arm: str) -> list[dict]:
    rows = raw[phase][arm]
    if phase == "ood_qa":
        return [{key: row[key] for key in QA_IDENTITY_FIELDS} for row in rows]
    return [{key: row[key] for key in PROSE_ALIGNMENT_FIELDS} for row in rows]


def receipt_inventory_v49d(raw: dict) -> dict:
    """Validate receipt shape and derive content-addressed arm/wave inventory."""
    if set(raw) != {"schema", "ood_qa", "ood_prose", "single_access_receipts"}:
        raise RuntimeError("V49D raw top-level inventory changed")
    if raw.get("schema") != "v49d-ood-only-raw-local":
        raise RuntimeError("V49D raw receipt schema changed")
    if set(raw.get("single_access_receipts", {})) != set(PHASES):
        raise RuntimeError("V49D raw semantic receipt labels changed")
    phase_result = {}
    total_rows = 0
    for phase in PHASES:
        table = raw.get(phase)
        if not isinstance(table, dict) or tuple(table) != ARMS:
            raise RuntimeError(f"V49D {phase} arm order/inventory changed")
        arm_receipts = {}
        shared_identity = None
        for arm in ARMS:
            rows = table[arm]
            if not isinstance(rows, list) or len(rows) != ROW_COUNTS[phase]:
                raise RuntimeError(f"V49D {phase}/{arm} row count changed")
            identities = _identity_rows(raw, phase, arm)
            if shared_identity is None:
                shared_identity = identities
            elif identities != shared_identity:
                raise RuntimeError(f"V49D {phase} identities are not aligned")
            arm_receipts[arm] = {
                "rows": len(rows),
                "identity_sha256": canonical_sha256(identities),
                "raw_rows_sha256": canonical_sha256(rows),
            }
            total_rows += len(rows)
        wave_receipts = []
        for wave_index, wave in enumerate(WAVES):
            wave_value = {
                "phase": phase,
                "wave_index": wave_index,
                "engine_arm_map": [
                    {"arm": arm, "engine_index": engine} for arm, engine in wave
                ],
                "arm_raw_rows_sha256": {
                    arm: arm_receipts[arm]["raw_rows_sha256"] for arm, _ in wave
                },
            }
            wave_receipts.append({
                **wave_value,
                "wave_receipt_sha256": canonical_sha256(wave_value),
            })
        phase_result[phase] = {
            "rows_per_arm": ROW_COUNTS[phase],
            "shared_identity_sha256": canonical_sha256(shared_identity),
            "arm_receipts": arm_receipts,
            "wave_receipts": wave_receipts,
        }
    return {
        "schema": "v49d-offline-raw-receipt-inventory",
        "arms": list(ARMS),
        "phases": list(PHASES),
        "total_arm_rows": total_rows,
        "phase_inventory": phase_result,
        "complete_inventory_sha256": canonical_sha256(phase_result),
    }


def gpu_receipt_v49d(path: Path) -> dict:
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if not rows:
        raise RuntimeError("V49D GPU receipt is empty")
    exact_keys = {
        "sampled_at_utc", "phase", "gpu", "expected_pid", "compute_pids",
        "foreign_compute_pids", "utilization_percent", "memory_used_mib",
    }
    if any(set(row) != exact_keys for row in rows):
        raise RuntimeError("V49D GPU receipt row schema changed")
    if {row["phase"] for row in rows} != set(PHASES):
        raise RuntimeError("V49D GPU phases are not exactly OOD QA and OOD prose")
    if {row["gpu"] for row in rows} != set(GPU_IDS):
        raise RuntimeError("V49D GPU identity coverage changed")
    if any(row["foreign_compute_pids"] for row in rows):
        raise RuntimeError("V49D foreign GPU process observed")
    timestamps = {}
    for row in rows:
        group = timestamps.setdefault(row["sampled_at_utc"], [])
        group.append(row)
    for sampled, group in timestamps.items():
        if (
            {row["gpu"] for row in group} != set(GPU_IDS)
            or len({row["phase"] for row in group}) != 1
            or len(group) != len(GPU_IDS)
        ):
            raise RuntimeError(f"V49D incomplete GPU sample group: {sampled}")
    pid_map = {}
    by_gpu = {}
    for gpu in GPU_IDS:
        selected = [row for row in rows if row["gpu"] == gpu]
        pids = {int(row["expected_pid"]) for row in selected}
        if len(pids) != 1:
            raise RuntimeError(f"V49D GPU {gpu} expected PID changed")
        expected_pid = pids.pop()
        pid_map[str(gpu)] = expected_pid
        phases = {}
        for phase in PHASES:
            phase_rows = [row for row in selected if row["phase"] == phase]
            resident = [
                row for row in phase_rows if expected_pid in row["compute_pids"]
            ]
            positive = [
                row for row in resident if row["utilization_percent"] > 0
            ]
            if not phase_rows or not resident or not positive:
                raise RuntimeError(f"V49D GPU {gpu} inactive in {phase}")
            phases[phase] = {
                "samples": len(phase_rows),
                "resident_samples": len(resident),
                "positive_samples": len(positive),
                "peak_utilization_percent": max(
                    row["utilization_percent"] for row in resident
                ),
                "peak_memory_used_mib": max(
                    row["memory_used_mib"] for row in resident
                ),
                "mean_resident_utilization_percent": math.fsum(
                    row["utilization_percent"] for row in resident
                ) / len(resident),
            }
        by_gpu[str(gpu)] = {"expected_pid": expected_pid, "phases": phases}
    if len(set(pid_map.values())) != 4:
        raise RuntimeError("V49D engine PID attribution is not one-to-one")
    return {
        "schema": "v49d-offline-ood-only-gpu-receipt",
        "file_sha256": file_sha256(path),
        "canonical_rows_sha256": canonical_sha256(rows),
        "row_count": len(rows),
        "sample_group_count": len(timestamps),
        "phase_labels_exact": list(PHASES),
        "expected_pid_map": pid_map,
        "all_four_attributed_positive_each_ood_phase": True,
        "by_gpu": by_gpu,
        "shadow_phase_present": False,
    }


def _qa_aggregate(rows: list[dict]) -> dict:
    records = []
    for expected_index, row in enumerate(rows):
        if set(row) != {
            "item_index", "item_sha256", "question", "answer", "response",
            "teacher", "format", "reward", "counters",
        }:
            raise RuntimeError("V49D QA raw row schema changed")
        if row["item_index"] != expected_index:
            raise RuntimeError("V49D QA item order changed")
        expected_identity = canonical_sha256({
            "question": row["question"], "answer": row["answer"],
        })
        if row["item_sha256"] != expected_identity:
            raise RuntimeError("V49D QA item identity changed")
        fmt, reward = reward_runtime.specialist_reward(
            row["response"], row["answer"]
        )
        counters = metric_runtime.protocol_counters(
            row["question"], row["response"]
        )
        if fmt != row["format"] or float(reward) != float(row["reward"]):
            raise RuntimeError("V49D QA reward receipt changed")
        if counters != row["counters"]:
            raise RuntimeError("V49D QA protocol counters changed")
        teacher = row["teacher"]
        teacher_mean = float(teacher["mean_answer_token_logprob"])
        if not math.isfinite(teacher_mean):
            raise RuntimeError("V49D QA teacher logprob is non-finite")
        records.append({
            "item_sha256": row["item_sha256"],
            "teacher_mean_answer_token_logprob": teacher_mean,
            "generated_reward": float(row["reward"]),
            "generated_format": row["format"],
            "response_sha256": hashlib.sha256(
                row["response"].encode("utf-8")
            ).hexdigest(),
            **counters,
        })
    count = len(records)
    weight = 1.0 / count
    teacher_values = [row["teacher_mean_answer_token_logprob"] for row in records]
    rewards = [row["generated_reward"] for row in records]
    counter_names = (
        "protocol_token_emission", "prompt_echo", "empty_extracted_answer"
    )
    return {
        "rows": count,
        "units": count,
        "teacher_forced_equal_unit_mean_answer_logprob": math.fsum(
            weight * value for value in teacher_values
        ),
        "generated_equal_unit_mean_reward": math.fsum(
            weight * value for value in rewards
        ),
        "generated_row_mean_reward": math.fsum(rewards) / count,
        "generated_exact_count": sum(
            row["generated_format"] == "exact" for row in records
        ),
        "generated_nonzero_count": sum(value > 0.0 for value in rewards),
        "protocol_leak_counters": {
            name: sum(row[name] for row in records) for name in counter_names
        },
        "numeric_item_manifest_sha256": canonical_sha256(records),
    }


def _prose_detail(rows: list[dict]) -> dict:
    exact_keys = {
        "item_id", "source", "title", "url", "normalized_source_url",
        "text_sha256", "token_ids_sha256", "prompt_token_count",
        "scored_token_count", "sum_token_logprob", "mean_token_logprob",
    }
    if any(set(row) != exact_keys for row in rows):
        raise RuntimeError("V49D prose raw row schema changed")
    for row in rows:
        scored = row["scored_token_count"]
        summed = float(row["sum_token_logprob"])
        mean = float(row["mean_token_logprob"])
        if (
            isinstance(scored, bool) or not isinstance(scored, int) or scored <= 0
            or not math.isfinite(summed) or not math.isfinite(mean)
            or not math.isclose(mean, summed / scored, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise RuntimeError("V49D prose numeric receipt changed")
    scored_token_count = sum(row["scored_token_count"] for row in rows)
    sum_token_logprob = math.fsum(float(row["sum_token_logprob"]) for row in rows)
    return {
        "item_count": len(rows),
        "scored_token_count": scored_token_count,
        "sum_token_logprob": sum_token_logprob,
        "mean_token_logprob": sum_token_logprob / scored_token_count,
        "items": rows,
    }


def _prose_aggregate(detail: dict) -> dict:
    compact = [{
        "text_sha256": row["text_sha256"],
        "token_ids_sha256": row["token_ids_sha256"],
        "scored_token_count": row["scored_token_count"],
        "sum_token_logprob": row["sum_token_logprob"],
    } for row in detail["items"]]
    return {
        "item_count": detail["item_count"],
        "scored_token_count": detail["scored_token_count"],
        "mean_token_logprob": detail["mean_token_logprob"],
        "numeric_item_manifest_sha256": canonical_sha256(compact),
    }


def _assert_exact_bases(table: dict, phase: str) -> dict:
    baseline = table["base_a"]
    if any(table[arm] != baseline for arm in BASE_ARMS[1:]):
        raise RuntimeError(f"V49D four-base exact equivalence failed on {phase}")
    return {"label": phase, "all_four_base_outputs_exact": True}


def _replica_gate(qa, prose_details, raw, arm: str) -> dict:
    qa_gate = metric_runtime.qa_ood_gate(qa["base_a"], qa[arm])
    qa_gate.update(paired_qa.paired_qa_bootstrap_v45a(
        raw["ood_qa"]["base_a"], raw["ood_qa"][arm],
        samples=BOOTSTRAP_SAMPLES,
    ))
    prose_gate = metric_runtime.prose_gate(prose_details["base_a"], prose_details[arm])
    counters = qa[arm]["protocol_leak_counters"]
    base_counters = qa["base_a"]["protocol_leak_counters"]
    protocol_safe = all(counters[key] <= base_counters[key] for key in base_counters)
    return {
        "arm": arm,
        "ood_qa": qa_gate,
        "ood_prose": prose_gate,
        "no_protocol_or_leak_counter_increase": protocol_safe,
        "eligible": qa_gate["passed"] and prose_gate["passed"] and protocol_safe,
    }


def gate_table_v49d(qa: dict, prose_details: dict, raw: dict) -> tuple[dict, dict]:
    table = {}
    for logical, replicas in LOGICAL_REPLICAS.items():
        gates = [_replica_gate(qa, prose_details, raw, arm) for arm in replicas]
        table[logical] = {
            "replicas": list(replicas),
            "replica_gates": gates,
            "both_replicas_independently_ood_eligible": all(
                gate["eligible"] for gate in gates
            ),
        }
    equal = LOGICAL_REPLICAS["v434_equal"]
    source = LOGICAL_REPLICAS["v434_source50"]
    def mean(arms, field):
        return sum(float(qa[arm][field]) for arm in arms) / 2.0
    direct = {
        "source50_minus_equal_mean_reward": (
            mean(source, "generated_equal_unit_mean_reward")
            - mean(equal, "generated_equal_unit_mean_reward")
        ),
        "source50_minus_equal_mean_exact_count": (
            mean(source, "generated_exact_count")
            - mean(equal, "generated_exact_count")
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


def load_preregistration_v49d(args) -> dict:
    path = Path(args.preregistration).resolve()
    if file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V49D recovery preregistration file changed")
    value = _load_json(path)
    content = _validate_self_hash(value, "recovery preregistration")
    if (
        content != args.preregistration_content_sha256
        or value.get("schema") != "v49d-ood-only-offline-recovery-preregistration"
        or value.get("status") != "sealed_before_offline_recovery"
        or value.get("generation_or_gpu_access_authorized") is not False
        or value.get("protected_semantic_input_access_authorized") is not False
        or value.get("source_artifacts", {}).get("raw", {}).get("path") != str(SOURCE_RAW)
        or value.get("implementation_bindings") != implementation_bindings_v49d()
    ):
        raise RuntimeError("V49D recovery preregistration content changed")
    return value


def _validate_source_artifact(path: Path, expected: dict, label: str) -> dict:
    if str(path) != expected.get("path") or file_sha256(path) != expected.get("file_sha256"):
        raise RuntimeError(f"V49D sealed {label} artifact changed")
    value = _load_json(path)
    content = _validate_self_hash(value, label)
    if content != expected.get("content_sha256"):
        raise RuntimeError(f"V49D sealed {label} content changed")
    return value


def recover_v49d(prereg: dict) -> dict:
    source = prereg["source_artifacts"]
    source_prereg = _validate_source_artifact(
        SOURCE_PREREGISTRATION, source["source_preregistration"], "source preregistration"
    )
    attempt = _validate_source_artifact(SOURCE_ATTEMPT, source["attempt"], "attempt")
    failure = _validate_source_artifact(SOURCE_FAILURE, source["failure"], "failure")
    if (
        source_prereg.get("schema")
        != "sft-v434-equal-vs-source50-replicated-ood-only-v49d"
        or source_prereg.get("runtime", {}).get("two_full_fixed_waves")
        != [[{"arm": arm, "engine_index": engine} for arm, engine in wave] for wave in WAVES]
        or source_prereg.get("shadow_access_authorized") is not False
        or source_prereg.get("heldout_or_holdout_access_authorized") is not False
    ):
        raise RuntimeError("V49D source seal scope changed")
    if (
        attempt.get("protected_semantic_access_count") != 0
        or attempt.get("shadow_opened") is not False
        or attempt.get("heldout_or_holdout_opened") is not False
        or failure.get("protected_semantic_access_count") != 2
        or failure.get("protected_semantic_access_labels") != ["ood_prose", "ood_qa"]
        or failure.get("shadow_opened") is not False
        or failure.get("heldout_or_holdout_opened") is not False
        or failure.get("message") != "v39a GPU 0 inactive in shadow"
    ):
        raise RuntimeError("V49D source access/failure receipt changed")
    if file_sha256(SOURCE_RAW) != source["raw"]["file_sha256"]:
        raise RuntimeError("V49D sealed raw receipt changed")
    if oct(SOURCE_RAW.stat().st_mode & 0o777) != "0o600":
        raise RuntimeError("V49D raw receipt is not mode 0600")
    raw = _load_json(SOURCE_RAW)
    inventory = receipt_inventory_v49d(raw)
    if inventory != prereg["expected_receipt_inventory"]:
        raise RuntimeError("V49D raw receipt inventory/wave hashes changed")
    receipts = raw["single_access_receipts"]
    if (
        set(receipts) != set(PHASES)
        or any(receipts[label].get("semantic_read_count") != 1 for label in PHASES)
        or any(receipts[label].get("rows") != ROW_COUNTS[label] for label in PHASES)
        or {
            label: {
                "path": receipts[label].get("path"),
                "file_sha256": receipts[label].get("file_sha256"),
            } for label in PHASES
        } != source_prereg.get("single_access_inputs")
    ):
        raise RuntimeError("V49D semantic access receipts changed")
    gpu = gpu_receipt_v49d(SOURCE_GPU_LOG)
    if gpu != prereg["expected_gpu_receipt"]:
        raise RuntimeError("V49D GPU receipt changed")

    qa = {arm: _qa_aggregate(raw["ood_qa"][arm]) for arm in ARMS}
    prose_details = {
        arm: _prose_detail(raw["ood_prose"][arm]) for arm in ARMS
    }
    prose = {arm: _prose_aggregate(prose_details[arm]) for arm in ARMS}
    base_equivalence = {
        "ood_qa": _assert_exact_bases(qa, "ood_qa"),
        "ood_prose": _assert_exact_bases(prose, "ood_prose"),
    }
    gates, direct = gate_table_v49d(qa, prose_details, raw)
    return self_hashed({
        "schema": "v49d-v434-equal-source50-ood-only-offline-recovery",
        "status": "complete_from_sealed_retry1_receipts_shadow_and_holdout_unopened",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "recovery_preregistration": {
            "path": str(Path(prereg["artifact_paths"]["preregistration"])),
            "file_sha256": file_sha256(
                Path(prereg["artifact_paths"]["preregistration"])
            ),
            "content_sha256": prereg["content_sha256_before_self_field"],
        },
        "source_generation": {
            "experiment": SOURCE_EXPERIMENT,
            "status": "generation_receipts_complete_then_legacy_telemetry_label_failure",
            "failure": source["failure"],
            "raw_local": {
                **source["raw"], "mode": "0600", "git_eligible": False,
            },
            "gpu_log": source["gpu_log"],
        },
        "receipt_inventory": inventory,
        "base_duplicate_equivalence": base_equivalence,
        "ood_qa": qa,
        "ood_prose": prose,
        "per_logical_candidate_gate_table": gates,
        "direct_hypothesis_ood_point_gates": direct,
        "gpu_activity": gpu,
        "access_proof": {
            "source_protected_semantic_access_count": 2,
            "source_protected_semantic_access_labels": ["ood_prose", "ood_qa"],
            "source_each_authorized_semantic_input_read_exactly_once": True,
            "offline_recovery_protected_semantic_input_reads": 0,
            "offline_recovery_model_or_generation_access": False,
            "offline_recovery_gpu_access": False,
            "raw_receipt_labels_exact": ["ood_qa", "ood_prose"],
            "gpu_phase_labels_exact": ["ood_qa", "ood_prose"],
            "shadow_or_split_input_bound": False,
            "shadow_opened": False,
            "heldout_or_holdout_opened": False,
        },
        "recovery_scope": {
            "metrics_and_gates_recomputed_from_sealed_raw_receipt_only": True,
            "raw_receipt_whole_file_sha256_validated_before_parse": True,
            "inventory_and_two_wave_receipt_hashes_validated": True,
            "gpu_receipt_whole_file_sha256_validated": True,
            "selection_or_promotion_authorized": False,
            "shadow_threshold_deferred_not_applied": 0.0008257591,
        },
    })


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--preregistration", default=str(DEFAULT_PREREGISTRATION))
    result.add_argument("--preregistration-sha256", required=True)
    result.add_argument("--preregistration-content-sha256", required=True)
    result.add_argument("--dry-run", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    prereg = load_preregistration_v49d(args)
    if args.dry_run:
        print(json.dumps({
            "schema": prereg["schema"],
            "content_sha256": prereg["content_sha256_before_self_field"],
            "source_raw_file_sha256": prereg["source_artifacts"]["raw"]["file_sha256"],
            "source_gpu_log_file_sha256": prereg["source_artifacts"]["gpu_log"]["file_sha256"],
            "protected_semantic_input_reads": 0,
            "model_or_generation_accessed": False,
            "gpu_accessed": False,
            "shadow_opened": False,
            "heldout_or_holdout_opened": False,
        }, sort_keys=True))
        return 0
    if REPORT.exists():
        raise FileExistsError(REPORT)
    report = recover_v49d(prereg)
    atomic_json(REPORT, report)
    print(json.dumps({
        "report": str(REPORT),
        "report_file_sha256": file_sha256(REPORT),
        "report_content_sha256": report["content_sha256_before_self_field"],
        "protected_semantic_input_reads": 0,
        "model_or_generation_accessed": False,
        "gpu_accessed": False,
        "shadow_opened": False,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
