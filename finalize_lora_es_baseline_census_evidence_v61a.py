#!/usr/bin/env python3
"""Seal compact, content-free evidence for the completed V61A census."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import lora_es_baseline_census_strata_v61a as strata


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "experiments/eggroll_es_hpo/runs/v61a_v434_train_only_baseline_census"
PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "v434_train_baseline_census_v61a.json"
)
ATTEMPT = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    ".v61a_v434_train_only_baseline_census.attempt.json"
)
EVIDENCE = RUN / "baseline_census_evidence_v61a.json"
STRATA = RUN / "baseline_census_strata_v61a.json"
REPORT = RUN / "baseline_census_report_v61a.json"
GPU_LOG = RUN / "gpu_activity_v61a.jsonl"
OUTPUT = RUN / "v61a_evidence_manifest.json"

EXPECTED = {
    "prereg_file": "b48fd9f93f203e089098a44368f8459f7b563118025524b810c82f210a753a18",
    "prereg_content": "8138e82654792899f48ea2f110de5a846a81146455e08779fb0e92df63ee30d3",
    "attempt_file": "d47cd86fd1f1a036583e8f12088c1fcf56e09d6a5508cb2d742f24852b0072cd",
    "attempt_content": "027dd0067d2c409bac33b8e315c875c4d9ecda879b090a1876e7d2bf250b4616",
    "evidence_file": "bb95aa2b99d292f0c5cff27afbd255d4ca0697097e8c38e1dbda4dfa63280640",
    "evidence_content": "92df95db709e05c2c81c94d98d755a81d15069c94e7bbffa37c9566e4a33b3b5",
    "strata_file": "23c8393555c3d7f09c95ecc7e23a04637f86df8fd20f55f67b67000ae78257f5",
    "strata_content": "d6a34b36fea22a8bdc97698a377ffb4df596bade8cf1506c41721f5db9c4185a",
    "report_file": "89aa6b70b6150cc5abafa6ebddaffebf1751fb6001c136fdd0dd40dd29ad2878",
    "report_content": "0f14376da323846e96ad16b2a3197e722b58afbb3c20765d3eeec1fbb5009547",
    "gpu_log_file": "423272ea0332ec2cd7389acbb4a5de748e4925c0affd6335770570f4a2ca8364",
}

EXPECTED_UNIT_COUNTS = {
    "actor_unstable": {"holdback": 35, "selection_pool": 105, "total": 140},
    "difficult": {"holdback": 7, "selection_pool": 23, "total": 30},
    "stable_exact": {"holdback": 0, "selection_pool": 3, "total": 3},
    "stable_partial": {"holdback": 8, "selection_pool": 27, "total": 35},
}
EXPECTED_ROW_COUNTS = {
    "actor_unstable": 249,
    "difficult": 92,
    "stable_exact": 3,
    "stable_partial": 104,
}
F1_RANGE_CUTPOINTS = (1e-12, 0.01, 0.05, 0.10, 0.25)
RAW_MARKERS = ("<|im_start|>", "<|im_end|>", "</think>", "\nQuestion:", "\nAnswer:")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _self_hash_valid(value: dict) -> bool:
    content = value.get("content_sha256_before_self_field")
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    return content == strata.canonical_sha256_v61a(compact)


def _require_identity(path: Path, expected_file: str, expected_content: str) -> dict:
    value = _read(path)
    if (
        file_sha256(path) != expected_file
        or value.get("content_sha256_before_self_field") != expected_content
        or not _self_hash_valid(value)
    ):
        raise RuntimeError(f"V61A artifact identity changed: {path.name}")
    return value


def _validate_evidence(value: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    rows = value.get("rows", [])
    if (
        value.get("schema") != "v61a-v434-train-baseline-census-evidence"
        or value.get("status") != "complete_characterization_only"
        or value.get("row_count") != 448
        or value.get("conflict_unit_count") != 208
        or value.get("actor_count") != 4
        or len(rows) != 448
        or {row.get("row_index") for row in rows} != set(range(448))
        or len({row.get("row_sha256") for row in rows}) != 448
        or value.get("numeric_row_manifest_sha256")
        != strata.canonical_sha256_v61a(rows)
    ):
        raise RuntimeError("V61A evidence coverage changed")
    expected_metric_keys = {
        "actor_rank", "generation_seed", "f1", "exact", "nonzero",
    }
    seeds = value.get("actor_generation_seeds")
    if not isinstance(seeds, list) or len(seeds) != 4:
        raise RuntimeError("V61A actor seeds changed")
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if set(row) != {
            "row_index", "row_sha256", "unit_identity_sha256", "row_count", "actors",
        } or len(row["actors"]) != 4:
            raise RuntimeError("V61A persisted row schema changed")
        by_unit[row["unit_identity_sha256"]].append(row)
        if {metric.get("actor_rank") for metric in row["actors"]} != set(range(4)):
            raise RuntimeError("V61A actor coverage changed")
        for metric in row["actors"]:
            rank = metric["actor_rank"]
            f1 = metric.get("f1")
            if (
                set(metric) != expected_metric_keys
                or metric.get("generation_seed") != seeds[rank]
                or isinstance(f1, bool)
                or not isinstance(f1, (int, float))
                or not math.isfinite(float(f1))
                or not 0.0 <= float(f1) <= 1.0
                or type(metric.get("exact")) is not int
                or metric["exact"] not in (0, 1)
                or type(metric.get("nonzero")) is not int
                or metric["nonzero"] not in (0, 1)
                or metric["nonzero"] != int(float(f1) > 0.0)
            ):
                raise RuntimeError("V61A numeric actor metric changed")
    if len(by_unit) != 208 or any(
        any(row["row_count"] != len(unit_rows) for row in unit_rows)
        for unit_rows in by_unit.values()
    ):
        raise RuntimeError("V61A conflict-unit multiplicity changed")
    for key in (
        "raw_question_answer_or_generation_text_persisted",
        "eval_ood_shadow_or_holdout_opened",
        "candidate_selection_or_promotion_performed",
        "adapter_update_or_master_commit_performed",
    ):
        if value.get(key) is not False:
            raise RuntimeError(f"V61A forbidden evidence side effect: {key}")
    if any(marker in json.dumps(value, ensure_ascii=False) for marker in RAW_MARKERS):
        raise RuntimeError("V61A raw semantic marker persisted")
    return rows, by_unit


def _validate_strata(value: dict, evidence: dict, by_unit: dict[str, list[dict]]) -> None:
    units = value.get("units", [])
    if (
        value.get("schema") != "v61a-v434-train-baseline-census-strata"
        or value.get("status") != "fail_closed_insufficient_stable_exact_support"
        or value.get("later_v61_hpo_authorized") is not False
        or value.get("source_evidence_content_sha256")
        != evidence["content_sha256_before_self_field"]
        or value.get("stratum_counts") != EXPECTED_UNIT_COUNTS
        or value.get("row_stratum_counts") != EXPECTED_ROW_COUNTS
        or len(units) != 208
        or len({unit.get("unit_identity_sha256") for unit in units}) != 208
        or value.get("unit_manifest_sha256")
        != strata.canonical_sha256_v61a(units)
    ):
        raise RuntimeError("V61A fail-closed strata changed")
    observed = defaultdict(lambda: defaultdict(int))
    for unit in units:
        name = unit.get("stratum")
        partition = unit.get("panel_partition")
        identity = unit.get("unit_identity_sha256")
        if (
            name not in EXPECTED_UNIT_COUNTS
            or partition not in {"selection_pool", "holdback"}
            or identity not in by_unit
            or unit.get("unit_rows") != len(by_unit[identity])
        ):
            raise RuntimeError("V61A unit manifest changed")
        observed[name][partition] += 1
        observed[name]["total"] += 1
    normalized_observed = {
        name: {
            "holdback": observed[name]["holdback"],
            "selection_pool": observed[name]["selection_pool"],
            "total": observed[name]["total"],
        }
        for name in EXPECTED_UNIT_COUNTS
    }
    if normalized_observed != EXPECTED_UNIT_COUNTS:
        raise RuntimeError("V61A partition counts changed")
    for key in (
        "raw_question_answer_or_generation_text_persisted",
        "eval_ood_shadow_or_holdout_opened",
        "candidate_selection_or_promotion_performed",
    ):
        if value.get(key) is not False:
            raise RuntimeError(f"V61A forbidden strata side effect: {key}")


def _numeric_summary(rows: list[dict]) -> dict:
    by_actor = {rank: [row["actors"][rank] for row in rows] for rank in range(4)}
    actor_summary = {
        str(rank): {
            "mean_f1": math.fsum(float(item["f1"]) for item in metrics) / len(metrics),
            "exact_rows": sum(item["exact"] for item in metrics),
            "nonzero_rows": sum(item["nonzero"] for item in metrics),
        }
        for rank, metrics in by_actor.items()
    }
    pairwise = {}
    for left in range(4):
        for right in range(left + 1, 4):
            pairs = list(zip(by_actor[left], by_actor[right]))
            deltas = [abs(float(a["f1"]) - float(b["f1"])) for a, b in pairs]
            pairwise[f"{left}-{right}"] = {
                "f1_different_rows": sum(delta > 1e-12 for delta in deltas),
                "mean_absolute_f1_delta": math.fsum(deltas) / len(deltas),
                "maximum_absolute_f1_delta": max(deltas),
                "exact_label_disagreements": sum(a["exact"] != b["exact"] for a, b in pairs),
                "nonzero_label_disagreements": sum(a["nonzero"] != b["nonzero"] for a, b in pairs),
            }
    ranges = [
        max(float(item["f1"]) for item in row["actors"])
        - min(float(item["f1"]) for item in row["actors"])
        for row in rows
    ]
    return {
        "per_actor": actor_summary,
        "pairwise": pairwise,
        "f1_range_rows_strictly_above": {
            format(cutpoint, ".12g"): sum(value > cutpoint for value in ranges)
            for cutpoint in F1_RANGE_CUTPOINTS
        },
        "all_four_exact_rows": sum(
            all(item["exact"] == 1 for item in row["actors"]) for row in rows
        ),
        "any_actor_exact_rows": sum(
            any(item["exact"] == 1 for item in row["actors"]) for row in rows
        ),
    }


def _validate_gpu(report: dict) -> dict:
    rows = [json.loads(line) for line in GPU_LOG.read_text(encoding="utf-8").splitlines() if line]
    if file_sha256(GPU_LOG) != EXPECTED["gpu_log_file"] or any(
        row.get("foreign_compute_pids") for row in rows
    ):
        raise RuntimeError("V61A GPU log identity or exclusivity changed")
    result = {}
    report_by_gpu = report.get("gpu_activity", {}).get("by_gpu", {})
    for gpu in range(4):
        selected = [row for row in rows if row.get("gpu") == gpu]
        sealed = report_by_gpu.get(str(gpu), {})
        pid = sealed.get("expected_pid")
        resident = [row for row in selected if pid in row.get("compute_pids", [])]
        positive = sum(row.get("utilization_percent", 0) > 0 for row in resident)
        mean = math.fsum(row["utilization_percent"] for row in resident) / len(resident)
        if (
            not selected
            or len(resident) != len(selected)
            or positive == 0
            or sealed.get("samples") != len(selected)
            or sealed.get("resident_samples") != len(resident)
            or sealed.get("positive_samples") != positive
            or sealed.get("peak_utilization_percent")
            != max(row["utilization_percent"] for row in resident)
            or sealed.get("mean_resident_utilization_percent") != mean
        ):
            raise RuntimeError(f"V61A GPU {gpu} attribution changed")
        result[str(gpu)] = {
            "expected_pid": pid,
            "samples": len(selected),
            "positive_samples": positive,
            "mean_utilization_percent": mean,
            "peak_utilization_percent": sealed["peak_utilization_percent"],
            "peak_memory_used_mib": sealed["peak_memory_used_mib"],
        }
    return {
        "gpu_log_file_sha256": EXPECTED["gpu_log_file"],
        "foreign_compute_pid_rows": 0,
        "all_four_attributed_positive": True,
        "by_gpu": result,
    }


def build_evidence_v61a() -> dict:
    prereg = _require_identity(
        PREREG, EXPECTED["prereg_file"], EXPECTED["prereg_content"],
    )
    attempt = _require_identity(
        ATTEMPT, EXPECTED["attempt_file"], EXPECTED["attempt_content"],
    )
    evidence = _require_identity(
        EVIDENCE, EXPECTED["evidence_file"], EXPECTED["evidence_content"],
    )
    frozen = _require_identity(
        STRATA, EXPECTED["strata_file"], EXPECTED["strata_content"],
    )
    report = _require_identity(
        REPORT, EXPECTED["report_file"], EXPECTED["report_content"],
    )
    if (
        prereg.get("schema") != "v61a-v434-train-baseline-census-preregistration"
        or prereg.get("status")
        != "preregistered_before_v61a_model_gpu_or_train_row_access"
        or attempt.get("schema") != "v61a-v434-train-baseline-census-attempt"
        or attempt.get("phase") != "before_train_semantics_model_or_gpu_load"
        or attempt.get("preregistration_content_sha256") != EXPECTED["prereg_content"]
        or attempt.get("eval_ood_shadow_or_holdout_opened") is not False
    ):
        raise RuntimeError("V61A preregistration or attempt contract changed")
    rows, by_unit = _validate_evidence(evidence)
    _validate_strata(frozen, evidence, by_unit)
    if (
        report.get("schema") != "v61a-v434-train-baseline-census-report"
        or report.get("status") != "complete_content_free_characterization_sealed"
        or report.get("evidence", {}).get("file_sha256") != EXPECTED["evidence_file"]
        or report.get("strata", {}).get("file_sha256") != EXPECTED["strata_file"]
        or report.get("strata", {}).get("later_v61_hpo_authorized") is not False
        or report.get("gpu_activity", {}).get("all_four_attributed_positive") is not True
        or report.get("cleanup", {}).get("engine_kill_count") != 4
        or report.get("cleanup", {}).get("placement_group_remove_count") != 4
        or report.get("cleanup", {}).get("all_four_gcs_states_removed") is not True
        or report.get("final_gpu_idle", {}).get("all_four_compute_process_lists_empty") is not True
    ):
        raise RuntimeError("V61A sealed report changed")
    for key in (
        "raw_question_answer_or_generation_text_persisted",
        "eval_ood_shadow_or_holdout_opened",
        "candidate_selection_update_or_promotion_performed",
        "adapter_update_or_master_commit_performed",
    ):
        if report.get(key) is not False:
            raise RuntimeError(f"V61A forbidden report side effect: {key}")
    result = {
        "schema": "lora-es-baseline-census-evidence-v61a",
        "status": "complete_content_free_characterization_stable_exact_support_fail_closed",
        "completed_at_utc": report["completed_at_utc"],
        "preregistration": {
            "file_sha256": EXPECTED["prereg_file"],
            "content_sha256": EXPECTED["prereg_content"],
        },
        "attempt": {
            "file_sha256": EXPECTED["attempt_file"],
            "content_sha256": EXPECTED["attempt_content"],
        },
        "artifacts": {
            "evidence": {
                "file_sha256": EXPECTED["evidence_file"],
                "content_sha256": EXPECTED["evidence_content"],
            },
            "strata": {
                "file_sha256": EXPECTED["strata_file"],
                "content_sha256": EXPECTED["strata_content"],
            },
            "report": {
                "file_sha256": EXPECTED["report_file"],
                "content_sha256": EXPECTED["report_content"],
            },
        },
        "coverage": {
            "rows": 448,
            "conflict_units": 208,
            "actors": 4,
            "numeric_actor_metrics": 1792,
        },
        "outcome_characterization": {
            "unit_stratum_counts": EXPECTED_UNIT_COUNTS,
            "row_stratum_counts": EXPECTED_ROW_COUNTS,
            **_numeric_summary(rows),
        },
        "scientific_gate": {
            "stable_exact_minima": frozen["stable_exact_fail_closed_minima"],
            "observed_stable_exact": EXPECTED_UNIT_COUNTS["stable_exact"],
            "later_v61_hpo_authorized": False,
            "threshold_or_quota_relaxed_after_outcomes": False,
        },
        "telemetry": _validate_gpu(report),
        "cleanup": {
            "engine_kill_count": 4,
            "placement_group_remove_count": 4,
            "all_four_gcs_states_removed": True,
            "sealed_final_gpu_idle": True,
        },
        "raw_question_answer_or_generation_text_persisted": False,
        "candidate_selection_update_or_promotion_performed": False,
        "adapter_update_or_master_commit_performed": False,
        "protected_semantics_opened": False,
        "ood_shadow_or_terminal_holdout_opened": False,
    }
    result["content_sha256_before_self_field"] = strata.canonical_sha256_v61a(result)
    return result


def main() -> int:
    value = build_evidence_v61a()
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("ascii")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, OUTPUT)
    print(json.dumps({
        "path": str(OUTPUT),
        "file_sha256": file_sha256(OUTPUT),
        "content_sha256": value["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
