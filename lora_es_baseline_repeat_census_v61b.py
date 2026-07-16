#!/usr/bin/env python3
"""Pure numeric analysis for the V61B common-seed repeat census."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import defaultdict


TRAIN_ROWS_V61B = 448
TRAIN_CONFLICT_UNITS_V61B = 208
ACTORS_V61B = 4
PASSES_V61B = 2
COMMON_GENERATION_SEED_V61B = 2_026_071_601
MAX_GENERATION_TOKENS_V61B = 64
GENERATION_PARAMS_WITHOUT_SEED_V61B = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": MAX_GENERATION_TOKENS_V61B,
    "detokenize": True,
}
F1_ABSOLUTE_DELTA_THRESHOLDS_V61B = (1e-12, 0.01, 0.05, 0.10, 0.25)

V61A_BOUND_AGGREGATES = {
    "evidence_file_sha256": (
        "bb95aa2b99d292f0c5cff27afbd255d4ca0697097e8c38e1dbda4dfa63280640"
    ),
    "evidence_content_sha256": (
        "92df95db709e05c2c81c94d98d755a81d15069c94e7bbffa37c9566e4a33b3b5"
    ),
    "strata_file_sha256": (
        "23c8393555c3d7f09c95ecc7e23a04637f86df8fd20f55f67b67000ae78257f5"
    ),
    "strata_content_sha256": (
        "d6a34b36fea22a8bdc97698a377ffb4df596bade8cf1506c41721f5db9c4185a"
    ),
    "report_file_sha256": (
        "89aa6b70b6150cc5abafa6ebddaffebf1751fb6001c136fdd0dd40dd29ad2878"
    ),
    "report_content_sha256": (
        "0f14376da323846e96ad16b2a3197e722b58afbb3c20765d3eeec1fbb5009547"
    ),
    "actor_mean_f1_range": [0.28352, 0.28747],
    "actor_exact_row_counts": [3, 4, 4, 3],
    "actor_nonzero_row_counts": [442, 442, 444, 443],
    "cross_actor_rows_f1_delta_gt": {
        "1e-12": 249, "0.01": 225, "0.05": 124, "0.1": 55, "0.25": 4,
    },
    "pairwise_mean_absolute_f1_delta_range": [0.0186, 0.0247],
    "pairwise_exact_label_disagreement_row_count_range": [0, 1],
    "pairwise_nonzero_label_disagreement_row_count_range": [0, 3],
    "all_actor_exact_rows": 3,
    "any_actor_exact_rows": 4,
    "unit_strata_counts": {
        "stable_exact": 3,
        "stable_partial": 35,
        "difficult": 30,
        "actor_unstable": 140,
    },
    "row_strata_counts": {
        "stable_exact": 3,
        "stable_partial": 104,
        "difficult": 92,
        "actor_unstable": 249,
    },
    "comparison_semantics": "four actors, one greedy pass each, distinct fixed seeds",
}


def canonical_sha256_v61b(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _threshold_key_v61b(value: float) -> str:
    return {
        1e-12: "1e-12", 0.01: "0.01", 0.05: "0.05",
        0.10: "0.1", 0.25: "0.25",
    }[value]


def _metric_v61b(value: dict, actor: int, pass_index: int) -> tuple[float, int, int]:
    expected_keys = {
        "actor_rank", "pass_index", "generation_seed", "f1", "exact", "nonzero",
    }
    f1 = value.get("f1")
    exact = value.get("exact")
    nonzero = value.get("nonzero")
    if (
        set(value) != expected_keys
        or value.get("actor_rank") != actor
        or value.get("pass_index") != pass_index
        or value.get("generation_seed") != COMMON_GENERATION_SEED_V61B
        or isinstance(f1, bool) or not isinstance(f1, (int, float))
        or not math.isfinite(float(f1)) or not 0.0 <= float(f1) <= 1.0
        or exact not in (0, 1) or nonzero not in (0, 1)
        or exact > nonzero or nonzero != int(float(f1) > 0.0)
        or (exact == 1 and not math.isclose(float(f1), 1.0))
    ):
        raise ValueError("v61b malformed content-free actor/pass metric")
    return float(f1), int(exact), int(nonzero)


def _validate_rows_v61b(evidence: dict) -> list[dict]:
    rows = list(evidence.get("rows", []))
    if (
        evidence.get("schema") != "v61b-v434-common-seed-repeat-census-evidence"
        or evidence.get("status") != "complete_characterization_only"
        or evidence.get("row_count") != TRAIN_ROWS_V61B
        or evidence.get("conflict_unit_count") != TRAIN_CONFLICT_UNITS_V61B
        or evidence.get("actor_count") != ACTORS_V61B
        or evidence.get("pass_count") != PASSES_V61B
        or evidence.get("common_generation_seed") != COMMON_GENERATION_SEED_V61B
        or evidence.get("generation_params_without_seed")
        != GENERATION_PARAMS_WITHOUT_SEED_V61B
        or evidence.get("raw_question_answer_or_generation_text_persisted") is not False
        or evidence.get("selection_update_or_promotion_performed") is not False
        or evidence.get("eval_ood_shadow_or_holdout_opened") is not False
        or len(rows) != TRAIN_ROWS_V61B
    ):
        raise ValueError("v61b evidence contract changed")
    row_ids = []
    units = defaultdict(list)
    for row in rows:
        row_sha = row.get("row_sha256")
        unit_sha = row.get("unit_identity_sha256")
        passes = sorted(row.get("passes", []), key=lambda item: item.get("pass_index", -1))
        if (
            not isinstance(row_sha, str) or len(row_sha) != 64
            or not isinstance(unit_sha, str) or len(unit_sha) != 64
            or not isinstance(row.get("row_count"), int) or row["row_count"] <= 0
            or [item.get("pass_index") for item in passes] != [0, 1]
        ):
            raise ValueError("v61b row/pass identity changed")
        for pass_index, pass_value in enumerate(passes):
            actors = sorted(
                pass_value.get("actors", []),
                key=lambda item: item.get("actor_rank", -1),
            )
            if (
                set(pass_value) != {"pass_index", "actors"}
                or [item.get("actor_rank") for item in actors] != list(range(4))
            ):
                raise ValueError("v61b actor coverage changed")
            for actor, metric in enumerate(actors):
                _metric_v61b(metric, actor, pass_index)
        row_ids.append(row_sha)
        units[unit_sha].append(row)
    if len(set(row_ids)) != 448 or len(units) != 208 or any(
        len(items) != items[0]["row_count"]
        or any(item["row_count"] != len(items) for item in items)
        for items in units.values()
    ):
        raise ValueError("v61b row/conflict-unit coverage changed")
    return rows


def _delta_summary_v61b(deltas: list[float]) -> dict:
    if not deltas or any(not math.isfinite(value) or value < 0.0 for value in deltas):
        raise ValueError("v61b invalid delta vector")
    return {
        "comparisons": len(deltas),
        "mean_absolute_f1_delta": math.fsum(deltas) / len(deltas),
        "maximum_absolute_f1_delta": max(deltas),
        "f1_absolute_delta_gt_counts": {
            _threshold_key_v61b(threshold): sum(
                value > threshold for value in deltas
            )
            for threshold in F1_ABSOLUTE_DELTA_THRESHOLDS_V61B
        },
    }


def build_repeat_analysis_v61b(evidence: dict) -> dict:
    rows = _validate_rows_v61b(evidence)

    def metric(row: dict, pass_index: int, actor: int):
        return _metric_v61b(
            row["passes"][pass_index]["actors"][actor], actor, pass_index,
        )

    within_actors = []
    within_all_deltas = []
    for actor in range(ACTORS_V61B):
        pairs = [(metric(row, 0, actor), metric(row, 1, actor)) for row in rows]
        deltas = [abs(left[0] - right[0]) for left, right in pairs]
        within_all_deltas.extend(deltas)
        within_actors.append({
            "actor_rank": actor,
            **_delta_summary_v61b(deltas),
            "exact_label_disagreement_rows": sum(left[1] != right[1] for left, right in pairs),
            "nonzero_label_disagreement_rows": sum(left[2] != right[2] for left, right in pairs),
        })
    within_unit_counts = {}
    for threshold in F1_ABSOLUTE_DELTA_THRESHOLDS_V61B:
        affected = set()
        for row in rows:
            if any(
                abs(metric(row, 0, actor)[0] - metric(row, 1, actor)[0]) > threshold
                for actor in range(ACTORS_V61B)
            ):
                affected.add(row["unit_identity_sha256"])
        within_unit_counts[_threshold_key_v61b(threshold)] = len(affected)

    cross_passes = []
    for pass_index in range(PASSES_V61B):
        row_metrics = [[metric(row, pass_index, actor) for actor in range(4)] for row in rows]
        ranges = [max(item[0] for item in values) - min(item[0] for item in values)
                  for values in row_metrics]
        pairwise = []
        for left, right in itertools.combinations(range(ACTORS_V61B), 2):
            pairs = [(values[left], values[right]) for values in row_metrics]
            pairwise.append({
                "actor_pair": [left, right],
                **_delta_summary_v61b([abs(a[0] - b[0]) for a, b in pairs]),
                "exact_label_disagreement_rows": sum(a[1] != b[1] for a, b in pairs),
                "nonzero_label_disagreement_rows": sum(a[2] != b[2] for a, b in pairs),
            })
        unit_counts = {}
        for threshold in F1_ABSOLUTE_DELTA_THRESHOLDS_V61B:
            affected = {
                row["unit_identity_sha256"]
                for row, delta in zip(rows, ranges, strict=True)
                if delta > threshold
            }
            unit_counts[_threshold_key_v61b(threshold)] = len(affected)
        cross_passes.append({
            "pass_index": pass_index,
            **_delta_summary_v61b(ranges),
            "rows_with_exact_label_disagreement": sum(
                len({item[1] for item in values}) > 1 for values in row_metrics
            ),
            "rows_with_nonzero_label_disagreement": sum(
                len({item[2] for item in values}) > 1 for values in row_metrics
            ),
            "all_actor_exact_rows": sum(all(item[1] for item in values) for values in row_metrics),
            "any_actor_exact_rows": sum(any(item[1] for item in values) for values in row_metrics),
            "conflict_units_with_any_row_f1_range_gt": unit_counts,
            "pairwise_actor_comparisons": pairwise,
            "actor_mean_f1": [
                math.fsum(values[actor][0] for values in row_metrics) / TRAIN_ROWS_V61B
                for actor in range(ACTORS_V61B)
            ],
            "actor_exact_row_counts": [
                sum(values[actor][1] for values in row_metrics) for actor in range(4)
            ],
            "actor_nonzero_row_counts": [
                sum(values[actor][2] for values in row_metrics) for actor in range(4)
            ],
        })

    v61a_counts = V61A_BOUND_AGGREGATES["cross_actor_rows_f1_delta_gt"]
    comparison = [{
        "pass_index": item["pass_index"],
        "common_seed_minus_v61a_distinct_seed_row_counts": {
            key: item["f1_absolute_delta_gt_counts"][key] - v61a_counts[key]
            for key in v61a_counts
        },
        "common_seed_row_counts": item["f1_absolute_delta_gt_counts"],
        "v61a_distinct_seed_row_counts": dict(v61a_counts),
    } for item in cross_passes]
    result = {
        "schema": "v61b-v434-common-seed-repeat-census-analysis",
        "status": "complete_characterization_only",
        "f1_absolute_delta_thresholds": list(F1_ABSOLUTE_DELTA_THRESHOLDS_V61B),
        "within_actor_pass_repeat": {
            "actors": within_actors,
            "all_actor_row_comparisons": _delta_summary_v61b(within_all_deltas),
            "conflict_units_with_any_row_actor_delta_gt": within_unit_counts,
        },
        "cross_actor_same_seed_by_pass": cross_passes,
        "v61a_distinct_seed_bound_aggregate": dict(V61A_BOUND_AGGREGATES),
        "common_seed_vs_distinct_seed_aggregate_comparison": comparison,
        "interpretation_contract": {
            "causal_variance_source_claimed": False,
            "within_actor_repeat_and_cross_actor_same_seed_reported_separately": True,
            "exact_and_nonzero_label_disagreement_not_inferred_from_f1": True,
            "continuous_instability_thresholds_not_changed_after_outcomes": True,
        },
        "source_evidence_content_sha256": evidence.get("content_sha256_before_self_field"),
        "raw_question_answer_or_generation_text_persisted": False,
        "selection_update_or_promotion_performed": False,
        "eval_ood_shadow_or_holdout_opened": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v61b(result)
    return result
