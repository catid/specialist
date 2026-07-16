#!/usr/bin/env python3
"""Train-only V49A capped source-temperature weighting design.

The design starts from exact equal-conflict-unit weights on the immutable v412
fold-3 training membership after replaying accepted edits through v434.  It
applies a source-temperature factor, capped per row, and solves one deterministic
normalizer per frozen category so category objective masses remain exact.
"""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from collections import defaultdict
from pathlib import Path

import build_curated_qa as curated
import build_train_refresh_v430_v47a as replay
import build_train_shadow_folds_v37a as frozen
import run_sft_train_only_control_v36a as engine
import sft_lora_equal_unit_matched_init_v42a as equal_unit
from qa_quality import stable_fact_id


ROOT = Path(__file__).resolve().parent
V430 = replay.PROJECTION
FROZEN_TRAIN = (
    ROOT / "experiments/sft_controls/v37a_shadow_folds_v412/fold_3_train.jsonl"
).resolve()
FROZEN_TRAIN_ROWS = 448
FROZEN_TRAIN_FILE_SHA256 = (
    "97fc920ac39f67536df26977de951e8c34bf8486eb8f42fbb0a67687f025a92a"
)
V434_PROJECTION_ROWS = 531
V434_PROJECTION_SHA256 = (
    "f86f0618b0ac87ffd58b863763fd8d6609179c13dce2b945ddf0b96d75f3c099"
)
V434_TRAIN_ROWS = 448
V434_TRAIN_CONFLICT_UNITS = 208
V434_TRAIN_SHA256 = (
    "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a"
)
SOURCE_TEMPERATURE = 0.5
MIN_ROW_MULTIPLIER = 2.0 / 3.0
MAX_ROW_MULTIPLIER = 1.5
BISECTION_STEPS = 100
REPLAY_431_434 = (
    (431, "56d40784197f9c124acc21abcd6313ca55e648a921e95287da0821ca5fade824"),
    (432, "6bb26f0a0ef094168bed42826eadd1dc39dbd891d1ff62798032f16250c8becd"),
    (433, "5d21081e044c3f5a2110212dae19e22dfdc08e4b59d483a134871cbeaa9c268c"),
    (434, V434_PROJECTION_SHA256),
)
EXPECTED_IDENTITIES = {
    "root_membership_sha256": "e0e5eca72438aebc40d915eb824e03c42441c180d6ab585e1b375f83ddec3275",
    "current_normalized_weight_sha256": "a8cbc597f865123100de870fdbc22a2529ed5ec3534531f897d27763baa4492f",
    "alternative_normalized_weight_sha256": "76dd9224cde643b2dd22123c2bd7952a830809a3ed84c977d61208da874de612",
    "source_mass_table_sha256": "3ab5d46ab944944137fabe1bd20db95000292c89804340f5d061b56bb1da77e0",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def curation_path(version: int) -> Path:
    return (
        ROOT / "data/manual_reviews" / f"context_merit_audit_v{version}"
        / f"pending_curation_context_merit_v{version}.jsonl"
    ).resolve()


def replay_v434_train_only() -> tuple[list[dict], dict]:
    if (
        file_sha256(V430) != replay.PROJECTION_SHA256
        or file_sha256(FROZEN_TRAIN) != FROZEN_TRAIN_FILE_SHA256
    ):
        raise RuntimeError("V49A frozen train input identity changed")
    records = []
    with tempfile.TemporaryDirectory(prefix=".v49a-v434-train-only-", dir=ROOT) as name:
        current = V430
        directory = Path(name)
        for version, expected in REPLAY_431_434:
            output = directory / f"train_v{version}.jsonl"
            report = directory / f"train_v{version}.report.json"
            decision = curation_path(version)
            curated.merge([current], output, report, [], [decision])
            observed = file_sha256(output)
            if observed != expected:
                raise RuntimeError(f"V49A v434 replay drift at v{version}")
            records.append({
                "version": version,
                "decision_file_sha256": file_sha256(decision),
                "projection_sha256": observed,
            })
            current = output
        projection_bytes = current.read_bytes()
    projection = [json.loads(line) for line in projection_bytes.splitlines() if line]
    if (
        len(projection) != V434_PROJECTION_ROWS
        or hashlib.sha256(projection_bytes).hexdigest() != V434_PROJECTION_SHA256
    ):
        raise RuntimeError("V49A final v434 train projection changed")

    original = read_jsonl(replay.SOURCE)
    roots = {row["fact_id"]: row["fact_id"] for row in original}
    for version in [item[0] for item in replay.REPLAY] + [431, 432, 433, 434]:
        for decision in read_jsonl(curation_path(version)):
            if decision["action"] != "edit":
                raise RuntimeError("V49A accepted replay unexpectedly adds or drops")
            root = roots.pop(decision["fact_id"])
            successor = stable_fact_id(decision["question"], decision["answer"])
            if successor in roots:
                raise RuntimeError("V49A successor fact lineage collided")
            roots[successor] = root
    frozen_train_rows = read_jsonl(FROZEN_TRAIN)
    frozen_root_membership = {row["fact_id"] for row in frozen_train_rows}
    train = [
        row for row in projection if roots[row["fact_id"]] in frozen_root_membership
    ]
    selected_roots = {roots[row["fact_id"]] for row in train}
    membership_sha = engine.canonical_sha256(sorted(selected_roots))
    payload = frozen.jsonl_bytes(train)
    units = frozen.build_conflict_units(train)
    if (
        len(train) != V434_TRAIN_ROWS
        or len(units) != V434_TRAIN_CONFLICT_UNITS
        or hashlib.sha256(payload).hexdigest() != V434_TRAIN_SHA256
        or selected_roots != frozen_root_membership
    ):
        raise RuntimeError("V49A frozen train membership or projection changed")
    expected_membership = EXPECTED_IDENTITIES["root_membership_sha256"]
    if expected_membership != "PENDING" and membership_sha != expected_membership:
        raise RuntimeError("V49A root train membership identity changed")
    return train, {
        "v434_replay": records,
        "root_membership_sha256": membership_sha,
        "root_membership_exactly_frozen_v412_fold3_train": True,
        "train_rows": len(train),
        "train_conflict_units": len(units),
        "train_jsonl_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _weight_identity(rows: list[dict], weights: list[float]) -> str:
    return engine.canonical_sha256([
        {"fact_id": row["fact_id"], "normalized_weight_hex": weight.hex()}
        for row, weight in zip(rows, weights)
    ])


def _ess(values) -> float:
    values = list(values)
    return 1.0 / math.fsum(value * value for value in values)


def _solve_category_multiplier(
    indices: list[int], current: list[float], raw: list[float], target: float,
) -> list[float]:
    lower, upper = 0.0, 1_000.0
    for _ in range(BISECTION_STEPS):
        middle = (lower + upper) / 2.0
        observed = math.fsum(
            current[index] * min(
                MAX_ROW_MULTIPLIER,
                max(MIN_ROW_MULTIPLIER, middle * raw[index]),
            )
            for index in indices
        )
        if observed < target:
            lower = middle
        else:
            upper = middle
    scale = (lower + upper) / 2.0
    return [
        min(MAX_ROW_MULTIPLIER, max(MIN_ROW_MULTIPLIER, scale * raw[index]))
        for index in indices
    ]


def analyze() -> dict:
    rows, membership = replay_v434_train_only()
    units = frozen.build_conflict_units(rows)
    ordinary, equal_audit = equal_unit.assign_equal_unit_weights(rows)
    current = [weight / len(rows) for weight in ordinary]
    categories = [None] * len(rows)
    unit_by_row = [None] * len(rows)
    for unit_index, unit in enumerate(units):
        for row_index in unit["indices"]:
            categories[row_index] = unit["stratum"]
            unit_by_row[row_index] = unit_index
    if any(item is None for item in categories + unit_by_row):
        raise RuntimeError("V49A conflict-unit coverage incomplete")

    current_source = defaultdict(float)
    current_category = defaultdict(float)
    source_rows = defaultdict(int)
    source_units = defaultdict(set)
    category_rows = defaultdict(int)
    category_units = defaultdict(set)
    for index, (row, weight, category) in enumerate(zip(rows, current, categories)):
        source = row["source"]
        current_source[source] += weight
        current_category[category] += weight
        source_rows[source] += 1
        source_units[source].add(unit_by_row[index])
        category_rows[category] += 1
        category_units[category].add(unit_by_row[index])
    positive = sorted(current_source.values())
    median_mass = positive[len(positive) // 2]
    raw_by_source = {
        source: (mass / median_mass) ** (-SOURCE_TEMPERATURE)
        for source, mass in current_source.items()
    }
    raw = [raw_by_source[row["source"]] for row in rows]
    multipliers = [None] * len(rows)
    for category, target in current_category.items():
        indices = [i for i, item in enumerate(categories) if item == category]
        solved = _solve_category_multiplier(indices, current, raw, target)
        for index, multiplier in zip(indices, solved):
            multipliers[index] = multiplier
    alternative = [
        weight * multiplier for weight, multiplier in zip(current, multipliers)
    ]
    if (
        not math.isclose(math.fsum(current), 1.0, abs_tol=1e-15, rel_tol=0)
        or not math.isclose(math.fsum(alternative), 1.0, abs_tol=1e-15, rel_tol=0)
        or min(multipliers) < MIN_ROW_MULTIPLIER - 1e-15
        or max(multipliers) > MAX_ROW_MULTIPLIER + 1e-15
    ):
        raise RuntimeError("V49A bounded normalized weighting invariant failed")

    alternative_source = defaultdict(float)
    alternative_category = defaultdict(float)
    current_unit = [0.0] * len(units)
    alternative_unit = [0.0] * len(units)
    for index, (row, before, after, category) in enumerate(
        zip(rows, current, alternative, categories)
    ):
        alternative_source[row["source"]] += after
        alternative_category[category] += after
        current_unit[unit_by_row[index]] += before
        alternative_unit[unit_by_row[index]] += after
    category_drift = {
        category: alternative_category[category] - mass
        for category, mass in current_category.items()
    }
    if max(abs(value) for value in category_drift.values()) > 1e-15:
        raise RuntimeError("V49A category mass preservation changed")

    source_table = [{
        "source": source,
        "rows": source_rows[source],
        "conflict_units_touched": len(source_units[source]),
        "current_mass": current_source[source],
        "alternative_mass": alternative_source[source],
        "mass_delta": alternative_source[source] - current_source[source],
        "effective_mass_multiplier": (
            alternative_source[source] / current_source[source]
        ),
    } for source in sorted(current_source)]
    source_table_sha = engine.canonical_sha256(source_table)
    identities = {
        **membership,
        "current_normalized_weight_sha256": _weight_identity(rows, current),
        "alternative_normalized_weight_sha256": _weight_identity(
            rows, alternative
        ),
        "source_mass_table_sha256": source_table_sha,
        "equal_unit_weighting_identity_sha256": equal_audit["identity_sha256"],
    }
    for key, expected in EXPECTED_IDENTITIES.items():
        if expected != "PENDING" and identities[key] != expected:
            raise RuntimeError(f"V49A train-only identity changed: {key}")

    top_current = sorted(current_source.items(), key=lambda item: (-item[1], item[0]))
    top_alternative = sorted(
        alternative_source.items(), key=lambda item: (-item[1], item[0])
    )
    current_row_ratio = max(current) / min(current)
    alternative_row_ratio = max(alternative) / min(alternative)
    current_unit_ratio = max(current_unit) / min(current_unit)
    alternative_unit_ratio = max(alternative_unit) / min(alternative_unit)
    diagnostics = {
        "objective_mass_sum": {
            "current": math.fsum(current),
            "alternative": math.fsum(alternative),
        },
        "effective_sample_size": {
            "row_current": _ess(current),
            "row_alternative": _ess(alternative),
            "row_retention_fraction": _ess(alternative) / _ess(current),
            "conflict_unit_current": _ess(current_unit),
            "conflict_unit_alternative": _ess(alternative_unit),
            "conflict_unit_retention_fraction": (
                _ess(alternative_unit) / _ess(current_unit)
            ),
            "source_current": _ess(current_source.values()),
            "source_alternative": _ess(alternative_source.values()),
            "source_improvement_fraction": (
                _ess(alternative_source.values()) / _ess(current_source.values()) - 1.0
            ),
        },
        "weight_bounds": {
            "current_min": min(current), "current_max": max(current),
            "current_max_to_min_ratio": current_row_ratio,
            "alternative_min": min(alternative),
            "alternative_max": max(alternative),
            "alternative_max_to_min_ratio": alternative_row_ratio,
            "min_applied_multiplier": min(multipliers),
            "max_applied_multiplier": max(multipliers),
            "configured_min_multiplier": MIN_ROW_MULTIPLIER,
            "configured_max_multiplier": MAX_ROW_MULTIPLIER,
        },
        "conflict_unit_mass_bounds": {
            "current_min": min(current_unit), "current_max": max(current_unit),
            "current_max_to_min_ratio": current_unit_ratio,
            "alternative_min": min(alternative_unit),
            "alternative_max": max(alternative_unit),
            "alternative_max_to_min_ratio": alternative_unit_ratio,
        },
        "source_concentration": {
            "current_largest_source": top_current[0][0],
            "current_largest_source_mass": top_current[0][1],
            "alternative_largest_source": top_alternative[0][0],
            "alternative_largest_source_mass": top_alternative[0][1],
            "current_top_two_sources": [item[0] for item in top_current[:2]],
            "current_top_two_mass": math.fsum(item[1] for item in top_current[:2]),
            "alternative_top_two_sources": [item[0] for item in top_alternative[:2]],
            "alternative_top_two_mass": math.fsum(
                item[1] for item in top_alternative[:2]
            ),
            "top_two_mass_delta": (
                math.fsum(item[1] for item in top_alternative[:2])
                - math.fsum(item[1] for item in top_current[:2])
            ),
        },
    }
    category_table = [{
        "category": category,
        "rows": category_rows[category],
        "conflict_units": len(category_units[category]),
        "current_mass": current_category[category],
        "alternative_mass": alternative_category[category],
        "mass_delta": category_drift[category],
    } for category in sorted(current_category)]
    merits = (
        diagnostics["effective_sample_size"]["source_improvement_fraction"] >= 0.20
        and diagnostics["effective_sample_size"]["row_retention_fraction"] >= 0.90
        and diagnostics["effective_sample_size"][
            "conflict_unit_retention_fraction"
        ] >= 0.90
        and diagnostics["source_concentration"]["top_two_mass_delta"] <= -0.08
        and alternative_row_ratio <= current_row_ratio + 1e-12
        and max(multipliers) <= MAX_ROW_MULTIPLIER
        and max(abs(value) for value in category_drift.values()) <= 1e-15
    )
    return {
        "schema": "specialist-train-only-source-temperature-weighting-v49a",
        "status": "complete_bounded_design_unlaunched",
        "inputs": {
            "v430_projection": {
                "path": str(V430), "sha256": replay.PROJECTION_SHA256,
            },
            "v434_projection": {
                "rows": V434_PROJECTION_ROWS, "sha256": V434_PROJECTION_SHA256,
                "materialization": "deterministic temporary train-only replay",
            },
            "frozen_v412_fold3_train_membership": {
                "path": str(FROZEN_TRAIN), "rows": FROZEN_TRAIN_ROWS,
                "file_sha256": FROZEN_TRAIN_FILE_SHA256,
            },
        },
        "membership_and_weight_identities": identities,
        "current_weighting": {
            "name": "equal_conflict_unit_then_equal_rows_within_unit",
            "audit": equal_audit,
        },
        "alternative_weighting": {
            "name": "category_preserving_capped_source_temperature_over_equal_unit",
            "source_temperature": SOURCE_TEMPERATURE,
            "raw_source_factor": "(current_source_mass / median_source_mass)^-0.5",
            "category_normalization": (
                "100-step deterministic bisection with per-row multiplier clipping"
            ),
            "min_row_multiplier": MIN_ROW_MULTIPLIER,
            "max_row_multiplier": MAX_ROW_MULTIPLIER,
            "category_masses_preserved_exactly": True,
            "within_source_category_conflict_unit_structure": (
                "current equal-unit row weights retained then boundedly rescaled"
            ),
        },
        "diagnostics": diagnostics,
        "per_source_mass": source_table,
        "per_category_mass": category_table,
        "recommendation": {
            "merits_one_later_preregistered_hpo_arm": merits,
            "basis_is_train_only_diagnostics": True,
            "quality_or_generalization_improvement_claimed": False,
            "reason": (
                "source ESS and top-two concentration improve materially while "
                "category mass is exact, row multipliers are bounded, the row "
                "max/min ratio does not increase, and row/unit ESS retain over 90%"
            ),
            "required_future_gate": (
                "fresh preregistered train run followed by unchanged OOD-first "
                "eligibility and document-disjoint validation; no current test "
                "authorizes promotion"
            ),
        },
        "access_firewall": {
            "train_metadata_and_train_rows_opened": True,
            "non_train_semantics_opened_received_or_inferred": False,
            "projection_filtering_is_content_free_root_membership_only": True,
            "external_metrics_used": False,
            "gpu_accessed": False,
            "training_launched": False,
            "evaluation_launched": False,
        },
    }
