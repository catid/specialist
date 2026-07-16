#!/usr/bin/env python3
"""Seal compact, content-free evidence for the completed V61B repeat census."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import lora_es_baseline_repeat_census_v61b as analysis_v61b


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "experiments/eggroll_es_hpo/runs/v61b_v434_common_seed_repeat_census"
PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "v434_common_seed_repeat_census_v61b.json"
)
ATTEMPT = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    ".v61b_v434_common_seed_repeat_census.attempt.json"
)
EVIDENCE = RUN / "common_seed_repeat_evidence_v61b.json"
ANALYSIS = RUN / "common_seed_repeat_analysis_v61b.json"
REPORT = RUN / "common_seed_repeat_report_v61b.json"
GPU_LOG = RUN / "gpu_activity_v61b.jsonl"
OUTPUT = RUN / "v61b_evidence_manifest.json"

EXPECTED = {
    "prereg_file": "44fece1ef9805b529cd268c52736cc6bb3f08696e55fb7612dcdf22f6407af74",
    "prereg_content": "b526acb6b728bc767ca74cdd47b0f41567703181e4cb1f59948c85fb23109bfa",
    "attempt_file": "8d629d9b82284ea879619781318f677fc4cc984bac492e6bbfb599115a5bea86",
    "attempt_content": "28c7ea22e9d7e90425ca38748d2b264f15ae4439f6d1db4597948ef8990b22ed",
    "evidence_file": "ea8ec108938ef2b17cf2572e4debca17d8986afa1f52fb2494d1fa87f54545b9",
    "evidence_content": "a2a0f8cf07510e5f1e61635d2199ccfe5de3562ef9c859c2e23a1e8a960a29c8",
    "analysis_file": "30a569fd89d38e66e95de9a55cc83c84e760007d1527bd6fa076f7ffc8c89961",
    "analysis_content": "354317a4bcca08bac4d0bc0f2d269f019b401b84c137acd202b131c616818b74",
    "report_file": "ad970297015200d5e01e116b9afffd1125d165c715aa6ebd1490b683470327cd",
    "report_content": "b50a5aefbd716a5ac47a427ea172a18a248734c6fd0df77d1f19957d22f25acf",
    "gpu_log_file": "6af7cb290fd1878caee4f3f89cedecd96479ca1829f22d0361b1274cae9f1376",
}

EXPECTED_WITHIN_COUNTS = {
    "1e-12": 654, "0.01": 543, "0.05": 229, "0.1": 96, "0.25": 7,
}
EXPECTED_CROSS_COUNTS = [
    {"1e-12": 244, "0.01": 217, "0.05": 115, "0.1": 46, "0.25": 4},
    {"1e-12": 235, "0.01": 215, "0.05": 121, "0.1": 62, "0.25": 5},
]
RAW_MARKERS = ("<|im_start|>", "<|im_end|>", "</think>", "\nQuestion:", "\nAnswer:")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _require_identity(path: Path, expected_file: str, expected_content: str) -> dict:
    value = _read(path)
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        file_sha256(path) != expected_file
        or value.get("content_sha256_before_self_field") != expected_content
        or analysis_v61b.canonical_sha256_v61b(compact) != expected_content
    ):
        raise RuntimeError(f"V61B artifact identity changed: {path.name}")
    return value


def _validate_evidence(value: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    rows = value.get("rows", [])
    if (
        value.get("schema") != "v61b-v434-common-seed-repeat-census-evidence"
        or value.get("status") != "complete_characterization_only"
        or value.get("row_count") != 448
        or value.get("conflict_unit_count") != 208
        or value.get("actor_count") != 4
        or value.get("pass_count") != 2
        or value.get("common_generation_seed") != 20_260_716_01
        or value.get("strictly_sequential_pass_order") != [0, 1]
        or len(rows) != 448
        or {row.get("row_index") for row in rows} != set(range(448))
        or len({row.get("row_sha256") for row in rows}) != 448
        or value.get("numeric_actor_pass_manifest_sha256")
        != analysis_v61b.canonical_sha256_v61b(rows)
    ):
        raise RuntimeError("V61B evidence coverage changed")
    metric_keys = {
        "actor_rank", "pass_index", "generation_seed", "f1", "exact", "nonzero",
    }
    by_unit: dict[str, list[dict]] = defaultdict(list)
    count = 0
    for row in rows:
        if (
            set(row) != {
                "row_index", "row_sha256", "unit_identity_sha256", "row_count", "passes",
            }
            or [item.get("pass_index") for item in row["passes"]] != [0, 1]
        ):
            raise RuntimeError("V61B row/pass schema changed")
        by_unit[row["unit_identity_sha256"]].append(row)
        for pass_index, pass_value in enumerate(row["passes"]):
            if set(pass_value) != {"pass_index", "actors"} or len(pass_value["actors"]) != 4:
                raise RuntimeError("V61B actor/pass coverage changed")
            for actor_rank, metric in enumerate(pass_value["actors"]):
                f1 = metric.get("f1")
                if (
                    set(metric) != metric_keys
                    or metric.get("actor_rank") != actor_rank
                    or metric.get("pass_index") != pass_index
                    or metric.get("generation_seed") != 20_260_716_01
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
                    raise RuntimeError("V61B numeric metric changed")
                count += 1
    if (
        count != 3_584
        or len(by_unit) != 208
        or any(
            any(row["row_count"] != len(unit_rows) for row in unit_rows)
            for unit_rows in by_unit.values()
        )
    ):
        raise RuntimeError("V61B row/unit multiplicity changed")
    for key in (
        "raw_question_answer_or_generation_text_persisted",
        "selection_update_or_promotion_performed",
        "eval_ood_shadow_or_holdout_opened",
    ):
        if value.get(key) is not False:
            raise RuntimeError(f"V61B forbidden evidence side effect: {key}")
    if any(marker in json.dumps(value, ensure_ascii=False) for marker in RAW_MARKERS):
        raise RuntimeError("V61B raw semantic marker persisted")
    return rows, by_unit


def _validate_analysis(value: dict, evidence: dict) -> None:
    rebuilt = analysis_v61b.build_repeat_analysis_v61b(evidence)
    within = value.get("within_actor_pass_repeat", {})
    cross = value.get("cross_actor_same_seed_by_pass", [])
    if (
        value != rebuilt
        or value.get("schema") != "v61b-v434-common-seed-repeat-census-analysis"
        or value.get("status") != "complete_characterization_only"
        or value.get("source_evidence_content_sha256") != EXPECTED["evidence_content"]
        or within.get("all_actor_row_comparisons", {}).get(
            "f1_absolute_delta_gt_counts"
        ) != EXPECTED_WITHIN_COUNTS
        or sum(
            item.get("exact_label_disagreement_rows", -1)
            for item in within.get("actors", [])
        ) != 0
        or [item.get("f1_absolute_delta_gt_counts") for item in cross]
        != EXPECTED_CROSS_COUNTS
        or [item.get("all_actor_exact_rows") for item in cross] != [3, 3]
        or [item.get("any_actor_exact_rows") for item in cross] != [4, 4]
        or [item.get("rows_with_exact_label_disagreement") for item in cross]
        != [1, 1]
        or value.get("interpretation_contract", {}).get(
            "causal_variance_source_claimed"
        ) is not False
    ):
        raise RuntimeError("V61B repeat analysis changed")
    for key in (
        "raw_question_answer_or_generation_text_persisted",
        "selection_update_or_promotion_performed",
        "eval_ood_shadow_or_holdout_opened",
    ):
        if value.get(key) is not False:
            raise RuntimeError(f"V61B forbidden analysis side effect: {key}")


def _validate_gpu(report: dict) -> dict:
    rows = [json.loads(line) for line in GPU_LOG.read_text(encoding="utf-8").splitlines() if line]
    if file_sha256(GPU_LOG) != EXPECTED["gpu_log_file"] or any(
        row.get("foreign_compute_pids") for row in rows
    ):
        raise RuntimeError("V61B GPU log identity or exclusivity changed")
    sealed_by_gpu = report.get("gpu_activity", {}).get("by_gpu", {})
    result = {}
    for gpu in range(4):
        selected = [row for row in rows if row.get("gpu") == gpu]
        sealed = sealed_by_gpu.get(str(gpu), {})
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
            or sealed.get("mean_resident_utilization_percent") != mean
        ):
            raise RuntimeError(f"V61B GPU {gpu} attribution changed")
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


def build_evidence_v61b() -> dict:
    prereg = _require_identity(
        PREREG, EXPECTED["prereg_file"], EXPECTED["prereg_content"],
    )
    attempt = _require_identity(
        ATTEMPT, EXPECTED["attempt_file"], EXPECTED["attempt_content"],
    )
    evidence = _require_identity(
        EVIDENCE, EXPECTED["evidence_file"], EXPECTED["evidence_content"],
    )
    observed_analysis = _require_identity(
        ANALYSIS, EXPECTED["analysis_file"], EXPECTED["analysis_content"],
    )
    report = _require_identity(
        REPORT, EXPECTED["report_file"], EXPECTED["report_content"],
    )
    if (
        prereg.get("schema") != "v61b-v434-common-seed-repeat-census-preregistration"
        or prereg.get("status") != "preregistered_before_v61b_train_model_or_gpu_access"
        or attempt.get("schema") != "v61b-v434-common-seed-repeat-census-attempt"
        or attempt.get("phase") != "before_train_model_or_gpu_access"
        or attempt.get("preregistration_content_sha256") != EXPECTED["prereg_content"]
        or attempt.get("v61a_row_evidence_opened") is not False
        or attempt.get("eval_ood_shadow_or_holdout_opened") is not False
    ):
        raise RuntimeError("V61B preregistration or attempt changed")
    rows, _ = _validate_evidence(evidence)
    _validate_analysis(observed_analysis, evidence)
    if (
        report.get("schema") != "v61b-v434-common-seed-repeat-census-report"
        or report.get("status") != "complete_content_free_characterization_sealed"
        or report.get("evidence", {}).get("file_sha256") != EXPECTED["evidence_file"]
        or report.get("analysis", {}).get("file_sha256") != EXPECTED["analysis_file"]
        or report.get("evidence", {}).get("completions") != 3_584
        or report.get("strictly_sequential_passes") != [0, 1]
        or report.get("gpu_activity", {}).get("all_four_attributed_positive") is not True
        or report.get("cleanup", {}).get("engine_kill_count") != 4
        or report.get("cleanup", {}).get("placement_group_remove_count") != 4
        or report.get("cleanup", {}).get("all_four_gcs_states_removed") is not True
        or report.get("final_gpu_idle", {}).get("all_four_compute_process_lists_empty") is not True
    ):
        raise RuntimeError("V61B sealed report changed")
    for key in (
        "raw_question_answer_or_generation_text_persisted",
        "selection_update_or_promotion_performed",
        "eval_ood_shadow_or_holdout_opened",
        "v61a_row_level_evidence_opened",
    ):
        if report.get(key) is not False:
            raise RuntimeError(f"V61B forbidden report side effect: {key}")
    within = observed_analysis["within_actor_pass_repeat"]
    cross = observed_analysis["cross_actor_same_seed_by_pass"]
    result = {
        "schema": "lora-es-baseline-repeat-evidence-v61b",
        "status": "complete_content_free_same_seed_repeat_characterization",
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
            "analysis": {
                "file_sha256": EXPECTED["analysis_file"],
                "content_sha256": EXPECTED["analysis_content"],
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
            "sequential_passes": 2,
            "numeric_actor_pass_metrics": len(rows) * 4 * 2,
        },
        "repeat_characterization": {
            "common_generation_seed": 20_260_716_01,
            "f1_absolute_delta_thresholds": observed_analysis[
                "f1_absolute_delta_thresholds"
            ],
            "within_actor_all_row_comparisons": within[
                "all_actor_row_comparisons"
            ],
            "within_actor_exact_label_disagreement_rows": 0,
            "within_actor_nonzero_label_disagreement_rows": sum(
                item["nonzero_label_disagreement_rows"] for item in within["actors"]
            ),
            "cross_actor_same_seed_by_pass": [{
                "pass_index": item["pass_index"],
                "f1_absolute_delta_gt_counts": item[
                    "f1_absolute_delta_gt_counts"
                ],
                "mean_absolute_f1_delta": item["mean_absolute_f1_delta"],
                "maximum_absolute_f1_delta": item["maximum_absolute_f1_delta"],
                "all_actor_exact_rows": item["all_actor_exact_rows"],
                "any_actor_exact_rows": item["any_actor_exact_rows"],
                "rows_with_exact_label_disagreement": item[
                    "rows_with_exact_label_disagreement"
                ],
                "rows_with_nonzero_label_disagreement": item[
                    "rows_with_nonzero_label_disagreement"
                ],
            } for item in cross],
            "v61a_distinct_seed_bound_aggregate": observed_analysis[
                "v61a_distinct_seed_bound_aggregate"
            ],
            "causal_variance_source_claimed": False,
        },
        "telemetry": _validate_gpu(report),
        "cleanup": {
            "engine_kill_count": 4,
            "placement_group_remove_count": 4,
            "all_four_gcs_states_removed": True,
            "sealed_final_gpu_idle": True,
        },
        "raw_question_answer_or_generation_text_persisted": False,
        "selection_update_or_promotion_performed": False,
        "protected_semantics_opened": False,
        "v61a_row_level_evidence_opened": False,
        "ood_shadow_or_terminal_holdout_opened": False,
    }
    result["content_sha256_before_self_field"] = (
        analysis_v61b.canonical_sha256_v61b(result)
    )
    return result


def main() -> int:
    value = build_evidence_v61b()
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
