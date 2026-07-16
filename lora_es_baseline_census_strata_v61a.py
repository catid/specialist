#!/usr/bin/env python3
"""Pure, content-free contracts for the V61A V434 baseline census.

This module deliberately imports no model, GPU, dataset, evaluation, or
training runtime.  It classifies numeric generation outcomes only after the
separately preregistered census has completed.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict


TRAIN_ROWS_V61A = 448
TRAIN_CONFLICT_UNITS_V61A = 208
ACTORS_V61A = 4
MAX_GENERATION_TOKENS_V61A = 64
ACTOR_GENERATION_SEEDS_V61A = (
    2_026_071_601,
    2_026_071_602,
    2_026_071_603,
    2_026_071_604,
)
GENERATION_PARAMS_WITHOUT_SEED_V61A = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": MAX_GENERATION_TOKENS_V61A,
    "detokenize": True,
}
STRATA_V61A = (
    "stable_exact",
    "stable_partial",
    "difficult",
    "actor_unstable",
)
PARTIAL_F1_MINIMUM_V61A = 0.25
ACTOR_F1_STABILITY_ATOL_V61A = 1e-12
HOLDBACK_FRACTION_V61A = 0.25
MINIMUM_STABLE_EXACT_UNITS_V61A = 8
MINIMUM_STABLE_EXACT_SELECTION_UNITS_V61A = 4
MINIMUM_STABLE_EXACT_HOLDBACK_UNITS_V61A = 2
REPRESENTATIVE_SEED_V61A = (
    "v61a-v434-baseline-census-representative-20260716"
)
HOLDBACK_SEED_V61A = "v61a-v434-baseline-census-holdback-20260716"
SELECTION_SEED_V61A = "v61a-v434-baseline-census-selection-20260716"


def canonical_sha256_v61a(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _seeded_hash_v61a(seed: str, *identities: str) -> str:
    if not seed or any(
        not isinstance(value, str) or len(value) != 64 for value in identities
    ):
        raise ValueError("v61a seeded hash identity changed")
    return hashlib.sha256(
        (seed + "\0" + "\0".join(identities)).encode("ascii")
    ).hexdigest()


def _validate_actor_metric_v61a(metric: dict, actor_rank: int) -> tuple[float, int, int]:
    f1 = metric.get("f1")
    exact = metric.get("exact")
    nonzero = metric.get("nonzero")
    if (
        metric.get("actor_rank") != actor_rank
        or metric.get("generation_seed") != ACTOR_GENERATION_SEEDS_V61A[actor_rank]
        or isinstance(f1, bool)
        or not isinstance(f1, (int, float))
        or not math.isfinite(float(f1))
        or not 0.0 <= float(f1) <= 1.0
        or exact not in (0, 1)
        or nonzero not in (0, 1)
        or exact > nonzero
        or nonzero != int(float(f1) > 0.0)
        or (exact == 1 and not math.isclose(float(f1), 1.0))
        or set(metric) != {
            "actor_rank", "generation_seed", "f1", "exact", "nonzero",
        }
    ):
        raise ValueError("v61a malformed content-free actor metric")
    return float(f1), int(exact), int(nonzero)


def classify_row_v61a(row: dict) -> dict:
    """Classify one row using only four numeric actor outcomes."""
    row_sha = row.get("row_sha256")
    unit_sha = row.get("unit_identity_sha256")
    actors = sorted(row.get("actors", []), key=lambda item: item.get("actor_rank", -1))
    if (
        not isinstance(row_sha, str) or len(row_sha) != 64
        or not isinstance(unit_sha, str) or len(unit_sha) != 64
        or not isinstance(row.get("row_count"), int) or row["row_count"] <= 0
        or [item.get("actor_rank") for item in actors] != list(range(ACTORS_V61A))
    ):
        raise ValueError("v61a row identity or actor coverage changed")
    metrics = [
        _validate_actor_metric_v61a(metric, rank)
        for rank, metric in enumerate(actors)
    ]
    f1s = [item[0] for item in metrics]
    exacts = [item[1] for item in metrics]
    nonzeros = [item[2] for item in metrics]
    mean_f1 = math.fsum(f1s) / ACTORS_V61A
    f1_range = max(f1s) - min(f1s)
    labels = [
        "exact" if exact else "nonzero" if nonzero else "zero"
        for _f1, exact, nonzero in metrics
    ]
    if all(exacts):
        stratum = "stable_exact"
    elif len(set(labels)) > 1 or f1_range > ACTOR_F1_STABILITY_ATOL_V61A:
        stratum = "actor_unstable"
    elif all(nonzeros) and mean_f1 >= PARTIAL_F1_MINIMUM_V61A:
        stratum = "stable_partial"
    else:
        # Includes stable low-F1 nonzero and stable zero outcomes.  Selecting
        # its highest-F1 representative preferentially retains difficult,
        # nonzero examples when the unit contains one.
        stratum = "difficult"
    return {
        "row_sha256": row_sha,
        "unit_identity_sha256": unit_sha,
        "unit_row_count": row["row_count"],
        "stratum": stratum,
        "mean_f1": mean_f1,
        "f1_range": f1_range,
        "exact_actor_count": sum(exacts),
        "nonzero_actor_count": sum(nonzeros),
        "representative_tie_sha256": _seeded_hash_v61a(
            REPRESENTATIVE_SEED_V61A, unit_sha, row_sha,
        ),
    }


def _representative_key_v61a(item: dict) -> tuple:
    priority = {
        "stable_exact": 0,
        "actor_unstable": 1,
        "stable_partial": 2,
        "difficult": 3,
    }
    # Within a stratum, keep the most informative nonzero/high-F1 row.  For
    # unstable rows, larger actor spread is preferred before mean F1.
    return (
        priority[item["stratum"]],
        -item["f1_range"] if item["stratum"] == "actor_unstable" else 0.0,
        -item["nonzero_actor_count"],
        -item["mean_f1"],
        item["representative_tie_sha256"],
        item["row_sha256"],
    )


def build_stratified_census_v61a(evidence: dict) -> dict:
    """Freeze unit strata and a selection/holdback partition.

    The partition is descriptive and cannot select or promote a model.  A
    stable-exact shortfall is recorded fail-closed for the later V61 HPO.
    """
    rows = list(evidence.get("rows", []))
    if (
        evidence.get("schema") != "v61a-v434-train-baseline-census-evidence"
        or evidence.get("status") != "complete_characterization_only"
        or evidence.get("row_count") != TRAIN_ROWS_V61A
        or evidence.get("conflict_unit_count") != TRAIN_CONFLICT_UNITS_V61A
        or evidence.get("actor_count") != ACTORS_V61A
        or evidence.get("actor_generation_seeds")
        != list(ACTOR_GENERATION_SEEDS_V61A)
        or evidence.get("generation_params_without_seed")
        != GENERATION_PARAMS_WITHOUT_SEED_V61A
        or evidence.get("raw_question_answer_or_generation_text_persisted") is not False
        or evidence.get("eval_ood_shadow_or_holdout_opened") is not False
        or evidence.get("candidate_selection_or_promotion_performed") is not False
        or len(rows) != TRAIN_ROWS_V61A
    ):
        raise ValueError("v61a census evidence contract changed")
    row_ids = [item.get("row_sha256") for item in rows]
    if len(set(row_ids)) != TRAIN_ROWS_V61A:
        raise ValueError("v61a census row identity coverage changed")
    summaries = [classify_row_v61a(row) for row in rows]
    by_unit: dict[str, list[dict]] = defaultdict(list)
    for item in summaries:
        by_unit[item["unit_identity_sha256"]].append(item)
    if len(by_unit) != TRAIN_CONFLICT_UNITS_V61A:
        raise ValueError("v61a census conflict-unit coverage changed")
    representatives = []
    for unit_sha, items in sorted(by_unit.items()):
        expected = items[0]["unit_row_count"]
        if len(items) != expected or any(
            item["unit_row_count"] != expected for item in items
        ):
            raise ValueError("v61a conflict-unit row multiplicity changed")
        representative = min(items, key=_representative_key_v61a)
        representatives.append({
            **representative,
            "unit_rows": expected,
            "holdback_priority_sha256": _seeded_hash_v61a(
                HOLDBACK_SEED_V61A, unit_sha,
            ),
            "selection_priority_sha256": _seeded_hash_v61a(
                SELECTION_SEED_V61A, unit_sha,
            ),
        })

    by_stratum: dict[str, list[dict]] = {name: [] for name in STRATA_V61A}
    for item in representatives:
        by_stratum[item["stratum"]].append(item)
    partitioned = []
    counts = {}
    for stratum in STRATA_V61A:
        items = sorted(by_stratum[stratum], key=lambda item: (
            item["holdback_priority_sha256"], item["unit_identity_sha256"],
        ))
        holdback_count = int(math.floor(len(items) * HOLDBACK_FRACTION_V61A))
        holdback_ids = {
            item["unit_identity_sha256"] for item in items[:holdback_count]
        }
        for item in items:
            partitioned.append({
                **item,
                "panel_partition": (
                    "holdback" if item["unit_identity_sha256"] in holdback_ids
                    else "selection_pool"
                ),
            })
        counts[stratum] = {
            "total": len(items),
            "selection_pool": len(items) - holdback_count,
            "holdback": holdback_count,
        }
    partitioned.sort(key=lambda item: (
        STRATA_V61A.index(item["stratum"]),
        item["panel_partition"] != "selection_pool",
        item["selection_priority_sha256"],
        item["unit_identity_sha256"],
    ))
    exact = counts["stable_exact"]
    exact_support_ok = (
        exact["total"] >= MINIMUM_STABLE_EXACT_UNITS_V61A
        and exact["selection_pool"] >= MINIMUM_STABLE_EXACT_SELECTION_UNITS_V61A
        and exact["holdback"] >= MINIMUM_STABLE_EXACT_HOLDBACK_UNITS_V61A
    )
    result = {
        "schema": "v61a-v434-train-baseline-census-strata",
        "status": (
            "complete_frozen_for_later_v61_design"
            if exact_support_ok
            else "fail_closed_insufficient_stable_exact_support"
        ),
        "later_v61_hpo_authorized": exact_support_ok,
        "row_count": TRAIN_ROWS_V61A,
        "conflict_unit_count": TRAIN_CONFLICT_UNITS_V61A,
        "actor_count": ACTORS_V61A,
        "strata": list(STRATA_V61A),
        "classification": {
            "partial_f1_minimum": PARTIAL_F1_MINIMUM_V61A,
            "actor_f1_stability_atol": ACTOR_F1_STABILITY_ATOL_V61A,
            "stable_exact": "all four actors exact on representative row",
            "stable_partial": (
                "all four actors nonexact/nonzero with identical F1 within "
                "tolerance and mean F1 at least partial_f1_minimum"
            ),
            "difficult": (
                "stable nonexact row below partial threshold, including zero; "
                "representative ordering prefers nonzero/high-F1 rows"
            ),
            "actor_unstable": (
                "actor exact/nonzero labels disagree or F1 range exceeds tolerance"
            ),
        },
        "representative_seed": REPRESENTATIVE_SEED_V61A,
        "holdback_seed": HOLDBACK_SEED_V61A,
        "selection_seed": SELECTION_SEED_V61A,
        "holdback_fraction_per_stratum": HOLDBACK_FRACTION_V61A,
        "stable_exact_fail_closed_minima": {
            "total": MINIMUM_STABLE_EXACT_UNITS_V61A,
            "selection_pool": MINIMUM_STABLE_EXACT_SELECTION_UNITS_V61A,
            "holdback": MINIMUM_STABLE_EXACT_HOLDBACK_UNITS_V61A,
        },
        "stratum_counts": counts,
        "row_stratum_counts": dict(sorted(Counter(
            item["stratum"] for item in summaries
        ).items())),
        "units": partitioned,
        "unit_manifest_sha256": canonical_sha256_v61a(partitioned),
        "source_evidence_content_sha256": evidence.get(
            "content_sha256_before_self_field"
        ),
        "raw_question_answer_or_generation_text_persisted": False,
        "eval_ood_shadow_or_holdout_opened": False,
        "candidate_selection_or_promotion_performed": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v61a(result)
    return result
