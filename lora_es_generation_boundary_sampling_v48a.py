#!/usr/bin/env python3
"""CPU contracts for train-only generation-boundary LoRA EGGROLL-ES V48A.

V48A selects one row from each of 64 distinct train conflict units using a
single, pre-population four-actor greedy-generation audit of the matched
initial state.  The sealed request order is then common to every signed state
and actor.  Generated F1 is a primary ES objective, not merely a projection
halfspace checked after the update.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict

import numpy as np

import eggroll_es_multi_anchor_v43h as multi_anchor


TRAIN_ROWS_V48A = 448
TRAIN_CONFLICT_UNITS_V48A = 208
FRAGILE_SUBSET_UNITS_V48A = 64
ACTORS_V48A = 4
POPULATION_SIZE_V48A = 8
SELECTION_SEED_V48A = "v48a-train-generation-boundary-units-20260716"
STRATUM_CYCLE_V48A = (
    "unstable", "partial", "unstable", "partial", "exact", "zero",
)
GENERATION_PARAMS_V48A = {
    "n": 1,
    "seed": 2_026_071_543,
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 64,
    "detokenize": True,
}
FORBIDDEN_RUNTIME_PATH_TOKENS_V48A = (
    "shadow_dev", "eval_qa", "ood_qa", "ood_prose", "holdout",
    "heldout", "benchmark",
)


def canonical_sha256_v48a(value: object) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def validate_runtime_paths_v48a(paths: list[str]) -> None:
    if not paths:
        raise ValueError("v48a requires explicit train-only runtime paths")
    for path in paths:
        lowered = str(path).casefold()
        if any(token in lowered for token in FORBIDDEN_RUNTIME_PATH_TOKENS_V48A):
            raise ValueError(f"v48a rejects protected runtime path: {path}")


def _metric_tuple(actor: dict) -> tuple[float, int, int, str]:
    f1 = float(actor.get("f1"))
    exact = actor.get("exact")
    nonzero = actor.get("nonzero")
    prediction = actor.get("prediction_sha256")
    if (
        not math.isfinite(f1) or not 0.0 <= f1 <= 1.0
        or exact not in (0, 1) or nonzero not in (0, 1)
        or not isinstance(prediction, str) or len(prediction) != 64
        or exact > nonzero
        or nonzero != int(f1 > 0.0)
        or (exact == 1 and not math.isclose(f1, 1.0))
    ):
        raise ValueError("v48a malformed content-free generation metric")
    return f1, int(exact), int(nonzero), prediction


def _row_summary_v48a(row: dict, membership: dict) -> dict:
    actors = sorted(row.get("actors", []), key=lambda item: item.get("actor_rank", -1))
    if (
        [item.get("actor_rank") for item in actors] != list(range(ACTORS_V48A))
        or row.get("row_sha256") != membership.get("row_sha256")
    ):
        raise ValueError("v48a base evidence actor or row coverage changed")
    metrics = [_metric_tuple(actor) for actor in actors]
    f1s = [item[0] for item in metrics]
    exacts = [item[1] for item in metrics]
    nonzeros = [item[2] for item in metrics]
    predictions = [item[3] for item in metrics]
    median_f1 = float(statistics.median(f1s))
    actor_disagreement = (
        len(set((f1, exact, nonzero) for f1, exact, nonzero, _ in metrics)) > 1
        or len(set(predictions)) > 1
    )
    if actor_disagreement:
        stratum = "unstable"
        fragility = (max(f1s) - min(f1s)) + float(len(set(predictions)) > 1)
    elif any(nonzeros) and not all(exacts):
        stratum = "partial"
        fragility = 1.0 - 2.0 * abs(median_f1 - 0.5)
    elif all(exacts):
        stratum, fragility = "exact", 0.0
    elif not any(nonzeros):
        stratum, fragility = "zero", 0.0
    else:
        raise RuntimeError("v48a generation stratum is not exhaustive")
    tie = hashlib.sha256((
        SELECTION_SEED_V48A + "\0" + membership["unit_identity_sha256"]
        + "\0" + row["row_sha256"]
    ).encode("ascii")).hexdigest()
    return {
        "row_sha256": row["row_sha256"],
        "unit_identity_sha256": membership["unit_identity_sha256"],
        "unit_row_count": membership["row_count"],
        "base_median_f1": median_f1,
        "base_f1_range": max(f1s) - min(f1s),
        "base_exact_actor_count": sum(exacts),
        "base_nonzero_actor_count": sum(nonzeros),
        "base_prediction_identity_count": len(set(predictions)),
        "stratum": stratum,
        "fragility": fragility,
        "tie_break_sha256": tie,
    }


def build_fragile_subset_v48a(bundle: dict, base_evidence: dict) -> dict:
    """Select a deterministic equal-unit subset without retaining semantics."""
    rows = list(bundle.get("row_sha256", []))
    memberships = list(bundle.get("unit_membership_v48a", []))
    evidence_rows = list(base_evidence.get("rows", []))
    expected_rows = int(bundle.get("expected_rows", TRAIN_ROWS_V48A))
    expected_units = int(
        bundle.get("expected_conflict_units", TRAIN_CONFLICT_UNITS_V48A)
    )
    subset_units = int(
        bundle.get("fragile_subset_units", FRAGILE_SUBSET_UNITS_V48A)
    )
    if (
        base_evidence.get("schema")
        != "train-only-four-actor-base-generation-evidence-v48a"
        or base_evidence.get("generation_params") != GENERATION_PARAMS_V48A
        or base_evidence.get("train_bundle_content_sha256")
        != bundle.get("train_bundle_content_sha256")
        or not isinstance(base_evidence.get("matched_master_sha256"), str)
        or len(base_evidence["matched_master_sha256"]) != 64
        or base_evidence.get("protected_semantics_opened") is not False
        or base_evidence.get("shadow_ood_holdout_or_benchmark_opened") is not False
        or len(rows) != expected_rows or len(memberships) != expected_rows
        or len(evidence_rows) != expected_rows
        or not 1 <= subset_units <= expected_units
        or len(set(rows)) != expected_rows
    ):
        raise ValueError("v48a train-only base evidence contract changed")
    membership_by_row = {}
    for row_sha, membership in zip(rows, memberships, strict=True):
        if (
            membership.get("row_sha256") != row_sha
            or not isinstance(membership.get("unit_identity_sha256"), str)
            or not isinstance(membership.get("row_count"), int)
            or membership["row_count"] <= 0
            or row_sha in membership_by_row
        ):
            raise ValueError("v48a conflict-unit membership changed")
        membership_by_row[row_sha] = membership
    if len({item["unit_identity_sha256"] for item in memberships}) != expected_units:
        raise ValueError("v48a conflict-unit count changed")
    evidence_by_row = {row.get("row_sha256"): row for row in evidence_rows}
    if set(evidence_by_row) != set(rows) or len(evidence_by_row) != expected_rows:
        raise ValueError("v48a base evidence row identity changed")
    summaries = [
        _row_summary_v48a(evidence_by_row[row_sha], membership_by_row[row_sha])
        for row_sha in rows
    ]
    by_unit = defaultdict(list)
    for item in summaries:
        by_unit[item["unit_identity_sha256"]].append(item)
    if any(
        len(items) != items[0]["unit_row_count"]
        or any(item["unit_row_count"] != len(items) for item in items)
        for items in by_unit.values()
    ):
        raise ValueError("v48a per-unit row coverage changed")
    order = {name: index for index, name in enumerate(
        ("unstable", "partial", "exact", "zero")
    )}
    representatives = []
    for unit_id in sorted(by_unit):
        representatives.append(min(
            by_unit[unit_id],
            key=lambda item: (
                order[item["stratum"]], -item["fragility"],
                item["tie_break_sha256"], item["row_sha256"],
            ),
        ))
    queues = {name: [] for name in order}
    for item in representatives:
        queues[item["stratum"]].append(item)
    for queue in queues.values():
        queue.sort(key=lambda item: (
            -item["fragility"], item["tie_break_sha256"], item["row_sha256"],
        ))
    selected = []
    cursor = {name: 0 for name in queues}
    while len(selected) < subset_units:
        progressed = False
        for stratum in STRATUM_CYCLE_V48A:
            index = cursor[stratum]
            if index < len(queues[stratum]):
                selected.append(queues[stratum][index])
                cursor[stratum] += 1
                progressed = True
                if len(selected) == subset_units:
                    break
        if not progressed:
            raise RuntimeError("v48a exhausted conflict units before subset budget")
    selected_units = [item["unit_identity_sha256"] for item in selected]
    selected_rows = [item["row_sha256"] for item in selected]
    if len(set(selected_units)) != subset_units or len(set(selected_rows)) != subset_units:
        raise RuntimeError("v48a selected duplicate row or conflict unit")
    counts = {
        name: sum(item["stratum"] == name for item in selected) for name in queues
    }
    items = [{
        **item,
        "request_index": index,
        "equal_conflict_unit_weight": 1.0 / subset_units,
    } for index, item in enumerate(selected)]
    result = {
        "schema": "train-generation-boundary-subset-v48a",
        "status": "selected_once_from_prepopulation_base_evidence",
        "selection_seed": SELECTION_SEED_V48A,
        "stratum_cycle": list(STRATUM_CYCLE_V48A),
        "source_train_rows": expected_rows,
        "source_conflict_units": expected_units,
        "selected_rows": subset_units,
        "selected_conflict_units": subset_units,
        "stratum_counts": counts,
        "items": items,
        "request_order_row_sha256": selected_rows,
        "request_order_sha256": canonical_sha256_v48a(selected_rows),
        "common_random_generation_params": dict(GENERATION_PARAMS_V48A),
        "teacher_forced_domain_sampling_changed": False,
        "rows_duplicated_or_oversampled_in_domain_objective": False,
        "protected_semantics_persisted": False,
        "shadow_ood_holdout_or_benchmark_opened": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256_v48a(result)
    return result


def score_fragile_items_v48a(subset: dict, metrics: list[dict]) -> dict:
    items = subset.get("items", [])
    if len(items) != FRAGILE_SUBSET_UNITS_V48A or len(metrics) != len(items):
        raise ValueError("v48a fragile score coverage changed")
    values = []
    compact = []
    for item, metric in zip(items, metrics, strict=True):
        f1, exact, nonzero, prediction = _metric_tuple(metric)
        if metric.get("row_sha256") != item["row_sha256"]:
            raise ValueError("v48a fragile output request order changed")
        values.append(f1)
        compact.append({
            "row_sha256": item["row_sha256"], "f1": f1,
            "exact": exact, "nonzero": nonzero,
            "prediction_sha256": prediction,
        })
    return {
        "equal_conflict_unit_mean_f1": math.fsum(values) / len(values),
        "exact_count": sum(item["exact"] for item in compact),
        "nonzero_count": sum(item["nonzero"] for item in compact),
        "numeric_item_manifest_sha256": canonical_sha256_v48a(compact),
        "selected_conflict_units": len(values),
    }


def assert_common_random_plan_v48a(receipts: list[dict], subset: dict) -> dict:
    expected_grid = {
        (direction, sign, actor)
        for direction in range(POPULATION_SIZE_V48A)
        for sign in ("plus", "minus")
        for actor in range(ACTORS_V48A)
    }
    observed = {
        (item.get("direction"), item.get("sign"), item.get("actor_rank"))
        for item in receipts
    }
    expected_subset = subset["content_sha256_before_self_field"]
    expected_order = subset["request_order_sha256"]
    if (
        len(receipts) != len(expected_grid) or observed != expected_grid
        or any(item.get("subset_content_sha256") != expected_subset for item in receipts)
        or any(item.get("request_order_sha256") != expected_order for item in receipts)
        or any(item.get("generation_params") != GENERATION_PARAMS_V48A for item in receipts)
    ):
        raise RuntimeError("v48a common-random population plan changed")
    return {
        "signed_actor_state_receipts": len(receipts),
        "all_use_identical_selected_items_order_and_sampling": True,
        "subset_content_sha256": expected_subset,
        "request_order_sha256": expected_order,
    }


def direct_generation_objective_v48a(
    domain_sign_scores: dict,
    fragile_f1_sign_scores: dict,
    prose_sign_scores: dict,
    qa_logprob_sign_scores: dict,
) -> dict:
    """Make fragile generated F1 part of the primary direction.

    Domain and fragile-F1 centered-rank vectors are normalized and averaged.
    Domain is also a projection constraint, so the generation-aware direction
    cannot knowingly trade away first-order train-domain progress.
    """
    objectives = {
        name: multi_anchor.objective_coefficients_v43h(scores)
        for name, scores in {
            "domain": domain_sign_scores,
            "fragile_generation_f1": fragile_f1_sign_scores,
            "prose_lm": prose_sign_scores,
            "qa_answer_logprob": qa_logprob_sign_scores,
        }.items()
    }
    if objectives["domain"]["zero_spread"]:
        raise RuntimeError("v48a domain objective has zero population spread")
    if objectives["fragile_generation_f1"]["zero_spread"]:
        raise RuntimeError(
            "v48a direct generated-F1 objective has zero population spread"
        )
    domain = np.asarray(objectives["domain"]["coefficients"], dtype=np.float64)
    fragile = np.asarray(
        objectives["fragile_generation_f1"]["coefficients"], dtype=np.float64
    )
    primary = 0.5 * (
        domain / np.linalg.norm(domain) + fragile / np.linalg.norm(fragile)
    )
    if not np.isfinite(primary).all() or np.linalg.norm(primary) == 0.0:
        raise RuntimeError("v48a domain/F1 primary direction cancelled or diverged")
    anchors = {
        "domain": objectives["domain"]["coefficients"],
        "fragile_generation_f1": objectives[
            "fragile_generation_f1"
        ]["coefficients"],
        "prose_lm": objectives["prose_lm"]["coefficients"],
        "qa_answer_logprob": objectives["qa_answer_logprob"]["coefficients"],
    }
    projected = multi_anchor.project_multi_anchor_trust_region_v43h(
        primary.tolist(), anchors,
        max_norm_ratio=multi_anchor.TRUST_REGION_NORM_RATIO_V43H,
    )
    diagnostics = projected["diagnostics"]
    if (
        diagnostics.get("decision") != "project_and_trust_region"
        or diagnostics.get("all_anchor_halfspaces_satisfied") is not True
        or diagnostics.get("anchor_directional_derivatives_after", {}).get(
            "fragile_generation_f1", -1.0
        ) < -multi_anchor.FEASIBILITY_TOLERANCE_V43H
        or not any(value != 0.0 for value in projected["coefficients"])
    ):
        raise RuntimeError("v48a generation-boundary projection failed closed")
    result = {
        "schema": "direct-generation-boundary-objective-v48a",
        "objective_fitness": objectives,
        "primary_construction": (
            "equal average of unit-norm domain and fragile generated-F1 "
            "centered-rank coefficient vectors"
        ),
        "fragile_generated_f1_is_primary_not_halfspace_only": True,
        "projection_anchors": [
            "domain", "fragile_generation_f1", "prose_lm",
            "qa_answer_logprob",
        ],
        "projection": projected,
    }
    result["content_sha256"] = canonical_sha256_v48a(result)
    return result
