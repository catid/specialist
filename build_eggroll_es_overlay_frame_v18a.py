#!/usr/bin/env python3
"""Build the aggregate-only, coverage-preserving V18A patch frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import eggroll_es_train_panel_sampler_v13 as sampler_v13
import materialize_eggroll_es_candidate_v298 as candidate_v298


ROOT = Path(__file__).resolve().parent
CANDIDATE_PATH_V18A = candidate_v298.OUTPUT_PATH_V298
CANDIDATE_MANIFEST_PATH_V18A = candidate_v298.MANIFEST_PATH_V298
PRODUCTION_PATH_V18A = (ROOT / "data/train_qa_curated_v1.jsonl").resolve()
OUTPUT_PATH_V18A = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v18a/"
    "production_patch_nested_panels_v18a.json"
).resolve()

CANDIDATE_ROWS_V18A = 519
CANDIDATE_SHA256_V18A = candidate_v298.V298_SHA256
CANDIDATE_MANIFEST_SHA256_V18A = (
    "8fd138e10217884675188538b6776dfd87a2015cf69922611c0c6100c7ff59e0"
)
PRODUCTION_ROWS_V18A = 784
PRODUCTION_SHA256_V18A = (
    "62e7ae28c86a458d4d33bf3f73f1b91b873c86e3f70ce87706a7394d1f391507"
)
SAMPLER_SHA256_V18A = (
    "81ca63c230995d44dfdb739a78b4a1a0f85d09724ba5915860ae36f78ccc3da9"
)

PANEL_NAMES_V18A = (
    "optimization_0",
    "optimization_1",
    "optimization_2",
    "train_screen_0",
    "train_screen_1",
)
ARM_ORDER_V18A = (
    "production_only",
    "patch_one_third",
    "patch_two_thirds",
    "patch_full",
)
BASE_CATEGORIES_V18A = (
    "paired_tier_1",
    "paired_tier_2",
    "paired_tier_3",
    "fallback",
)
BASE_CATEGORY_POPULATIONS_V18A = {
    "paired_tier_1": 67,
    "paired_tier_2": 68,
    "paired_tier_3": 67,
    "fallback": 70,
}
CANDIDATE_ONLY_LAYER_POPULATIONS_V18A = {1: 8, 2: 7, 3: 8}
PRODUCTION_COMPONENT_CAPACITY_V18A = {
    "safety_consent": 38,
    "technique": 87,
    "equipment_material": 22,
    "resources_general": 125,
}
CANDIDATE_COMPONENT_CAPACITY_V18A = {
    "safety_consent": 73,
    "technique": 53,
    "equipment_material": 21,
    "resources_general": 79,
}
PRODUCTION_TOPIC_QUOTAS_V18A = {
    "safety_consent": 7,
    "technique": 17,
    "equipment_material": 4,
    "resources_general": 24,
}
BASE_CATEGORY_QUOTA_V18A = 13
CANDIDATE_ONLY_LAYER_QUOTA_V18A = 1
ARM_REQUESTS_PER_PANEL_V18A = {
    "production_only": 52,
    "patch_one_third": 53,
    "patch_two_thirds": 54,
    "patch_full": 55,
}
ARM_POPULATIONS_V18A = {
    "production_only": 272,
    "patch_one_third": 280,
    "patch_two_thirds": 287,
    "patch_full": 295,
}
ELIGIBLE_PATCH_POPULATIONS_V18A = {
    "production_only": 0,
    "patch_one_third": 75,
    "patch_two_thirds": 150,
    "patch_full": 225,
}
PREFIX_SEED_V18A = 20260721
FLOW_SEED_V18A = 20260722
REPRESENTATIVE_SEED_V18A = 20260723


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity_root_sha256(values) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def normalize_url_v18a(value) -> str:
    parsed = urlsplit(str(value).strip())
    if not parsed.scheme or not parsed.netloc:
        return str(value).strip()
    query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in {"si", "fbclid", "gclid"}
    ]
    path = re.sub(r"/+", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        urlencode(sorted(query)),
        "",
    ))


def row_urls_v18a(row: dict) -> set[str]:
    scalar = (
        "url",
        "evidence_url",
        "canonical_url",
        "supplied_url",
        "original_ropetopia_url",
        "title_evidence_url",
        "url_evidence_url",
        "canonical_resource_url",
    )
    arrays = ("urls", "canonical_urls", "supplied_urls")
    values = {row[key] for key in scalar if row.get(key)}
    values.update(url for key in arrays for url in row.get(key, ()) if url)
    return {normalize_url_v18a(value) for value in values}


def row_lineages_v18a(row: dict) -> set[str]:
    lineage = row.get("source_lineage") or {}
    return {
        f"{key}:{json.dumps(lineage[key], sort_keys=True)}"
        for key in ("raw", "raw_document", "raw_successor_document")
        if lineage.get(key)
    }


class DisjointSetV18A:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def _connect_rows_v18a(
    rows: list[dict], semantic_ids: list[str], disjoint: DisjointSetV18A,
) -> None:
    first: dict[str, int] = {}

    def connect(identifier: str, index: int) -> None:
        if identifier in first:
            disjoint.union(index, first[identifier])
        else:
            first[identifier] = index

    for index, (row, semantic_id) in enumerate(zip(rows, semantic_ids)):
        connect(f"document:{row['document_sha256']}", index)
        for url in row_urls_v18a(row):
            connect(f"url:{url}", index)
        for lineage in row_lineages_v18a(row):
            connect(f"lineage:{lineage}", index)
        connect(f"semantic:{semantic_id}", index)


def build_side_units_v18a(rows: list[dict], side: str) -> tuple[list[dict], list[int]]:
    semantic_ids = sampler_v13.build_semantic_clusters(rows)
    disjoint = DisjointSetV18A(len(rows))
    _connect_rows_v18a(rows, semantic_ids, disjoint)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(rows)):
        components[disjoint.find(index)].append(index)
    unsorted = []
    for indices in components.values():
        member_sha256 = sorted(sampler_v13.row_sha256(rows[index]) for index in indices)
        unit_id = canonical_sha256({
            "schema": "eggroll-es-side-conflict-unit-v18a",
            "side": side,
            "members": member_sha256,
        })
        stratum_counts = Counter(
            sampler_v13.classify_stratum(rows[index]) for index in indices
        )
        stratum = max(
            sampler_v13.STRATA,
            key=lambda name: (
                stratum_counts[name], sampler_v13._TIE_PRIORITY[name]
            ),
        )
        eligible = [
            index for index in indices
            if sampler_v13.classify_stratum(rows[index]) == stratum
        ]
        representative = min(
            eligible,
            key=lambda index: canonical_sha256({
                "schema": "eggroll-es-fixed-side-representative-v18a",
                "seed": REPRESENTATIVE_SEED_V18A,
                "side": side,
                "unit_id": unit_id,
                "row_sha256": sampler_v13.row_sha256(rows[index]),
            }),
        )
        unsorted.append({
            "unit_id": unit_id,
            "side": side,
            "stratum": stratum,
            "member_indices": tuple(indices),
            "document_sha256s": frozenset(
                rows[index]["document_sha256"] for index in indices
            ),
            "urls": frozenset().union(
                *(row_urls_v18a(rows[index]) for index in indices)
            ),
            "lineages": frozenset().union(
                *(row_lineages_v18a(rows[index]) for index in indices)
            ),
            "representative_index": representative,
            "representative_row_sha256": sampler_v13.row_sha256(
                rows[representative]
            ),
        })
    units = sorted(unsorted, key=lambda unit: unit["unit_id"])
    row_to_unit = [-1] * len(rows)
    for unit_index, unit in enumerate(units):
        for row_index in unit["member_indices"]:
            row_to_unit[row_index] = unit_index
    if any(unit_index < 0 for unit_index in row_to_unit):
        raise RuntimeError("v18a side-unit membership is incomplete")
    return units, row_to_unit


def load_bound_rows_v18a() -> tuple[list[dict], list[dict]]:
    expected = {
        CANDIDATE_PATH_V18A: CANDIDATE_SHA256_V18A,
        CANDIDATE_MANIFEST_PATH_V18A: CANDIDATE_MANIFEST_SHA256_V18A,
        PRODUCTION_PATH_V18A: PRODUCTION_SHA256_V18A,
        Path(sampler_v13.__file__).resolve(): SAMPLER_SHA256_V18A,
    }
    if any(file_sha256(path) != digest for path, digest in expected.items()):
        raise RuntimeError("v18a frozen train input or grouping code changed")
    candidate = _read_jsonl(CANDIDATE_PATH_V18A)
    production = _read_jsonl(PRODUCTION_PATH_V18A)
    if len(candidate) != CANDIDATE_ROWS_V18A or len(production) != PRODUCTION_ROWS_V18A:
        raise RuntimeError("v18a frozen train row count changed")
    return candidate, production


def build_joint_frame_v18a(candidate: list[dict], production: list[dict]) -> dict:
    side_rows = {"candidate": candidate, "production": production}
    side_units = {}
    row_to_unit = {}
    for side, rows in side_rows.items():
        side_units[side], row_to_unit[side] = build_side_units_v18a(rows, side)
    capacity = {
        side: dict(Counter(unit["stratum"] for unit in units))
        for side, units in side_units.items()
    }
    if capacity != {
        "candidate": CANDIDATE_COMPONENT_CAPACITY_V18A,
        "production": PRODUCTION_COMPONENT_CAPACITY_V18A,
    }:
        raise RuntimeError("v18a standalone component capacity changed")

    tagged = [
        {"side": "candidate", "side_index": index, "row": row}
        for index, row in enumerate(candidate)
    ] + [
        {"side": "production", "side_index": index, "row": row}
        for index, row in enumerate(production)
    ]
    combined_rows = [item["row"] for item in tagged]
    semantic_ids = sampler_v13.build_semantic_clusters(combined_rows)
    disjoint = DisjointSetV18A(len(combined_rows))
    _connect_rows_v18a(combined_rows, semantic_ids, disjoint)
    first_side_unit = {}
    for combined_index, item in enumerate(tagged):
        key = (item["side"], row_to_unit[item["side"]][item["side_index"]])
        if key in first_side_unit:
            disjoint.union(combined_index, first_side_unit[key])
        else:
            first_side_unit[key] = combined_index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(tagged)):
        components[disjoint.find(index)].append(index)

    joint_components = []
    for members in components.values():
        unit_indices = {"candidate": set(), "production": set()}
        for combined_index in members:
            item = tagged[combined_index]
            unit_indices[item["side"]].add(
                row_to_unit[item["side"]][item["side_index"]]
            )
        if any(len(unit_indices[side]) > 1 for side in unit_indices):
            raise RuntimeError("v18a joint component merged multiple same-side units")
        candidate_unit = next(iter(unit_indices["candidate"]), None)
        production_unit = next(iter(unit_indices["production"]), None)
        identities = sorted(
            f"{side}:{side_units[side][unit_index]['unit_id']}"
            for side, unit_index in (
                ("candidate", candidate_unit),
                ("production", production_unit),
            )
            if unit_index is not None
        )
        relation = "candidate_only" if production_unit is None else (
            "production_only" if candidate_unit is None else "paired"
        )
        match_class = relation
        if relation == "paired":
            candidate_contract = side_units["candidate"][candidate_unit]
            production_contract = side_units["production"][production_unit]
            if (
                candidate_contract["document_sha256s"]
                & production_contract["document_sha256s"]
            ):
                match_class = "shared_document"
            elif candidate_contract["urls"] & production_contract["urls"]:
                match_class = "shared_url_without_shared_document"
            elif (
                candidate_contract["lineages"]
                & production_contract["lineages"]
            ):
                match_class = "shared_lineage_without_shared_document"
            else:
                match_class = "semantic_only_without_shared_document"
        joint_components.append({
            "joint_id": canonical_sha256({
                "schema": "eggroll-es-joint-component-v18a",
                "members": identities,
            }),
            "candidate_unit": candidate_unit,
            "production_unit": production_unit,
            "relation": relation,
            "match_class": match_class,
        })
    joint_components.sort(key=lambda item: item["joint_id"])
    relation_counts = Counter(item["relation"] for item in joint_components)
    match_counts = Counter(item["match_class"] for item in joint_components)
    if (
        len(joint_components) != 295
        or relation_counts
        != {"paired": 203, "candidate_only": 23, "production_only": 69}
        or match_counts
        != {
            "shared_document": 202,
            "shared_url_without_shared_document": 1,
            "candidate_only": 23,
            "production_only": 69,
        }
    ):
        raise RuntimeError("v18a joint relation frame changed")
    return {
        "side_units": side_units,
        "joint_components": joint_components,
        "capacity": capacity,
        "relation_counts": dict(relation_counts),
        "match_counts": dict(match_counts),
    }


def assign_patch_layers_v18a(frame: dict) -> dict:
    components = frame["joint_components"]
    eligible_paired = [
        index for index, item in enumerate(components)
        if item["match_class"] == "shared_document"
    ]
    candidate_only = [
        index for index, item in enumerate(components)
        if item["relation"] == "candidate_only"
    ]
    fallback = [
        index for index, item in enumerate(components)
        if item["relation"] == "production_only"
        or item["match_class"] == "shared_url_without_shared_document"
    ]

    def ordered(indices, population_class):
        return sorted(
            indices,
            key=lambda index: canonical_sha256({
                "schema": "eggroll-es-patch-prefix-v18a",
                "seed": PREFIX_SEED_V18A,
                "population_class": population_class,
                "joint_id": components[index]["joint_id"],
            }),
        )

    eligible_paired = ordered(eligible_paired, "eligible_paired")
    candidate_only = ordered(candidate_only, "candidate_only")
    paired_splits = (67, 68, 67)
    candidate_only_splits = (8, 7, 8)
    paired_tier = {}
    cursor = 0
    for tier, size in enumerate(paired_splits, 1):
        for component_index in eligible_paired[cursor:cursor + size]:
            paired_tier[component_index] = tier
        cursor += size
    candidate_only_layer = {}
    cursor = 0
    for layer, size in enumerate(candidate_only_splits, 1):
        for component_index in candidate_only[cursor:cursor + size]:
            candidate_only_layer[component_index] = layer
        cursor += size
    base_category = {
        component_index: f"paired_tier_{tier}"
        for component_index, tier in paired_tier.items()
    }
    base_category.update({component_index: "fallback" for component_index in fallback})
    if (
        Counter(base_category.values()) != BASE_CATEGORY_POPULATIONS_V18A
        or Counter(candidate_only_layer.values())
        != CANDIDATE_ONLY_LAYER_POPULATIONS_V18A
        or len(base_category) != 272
        or len(candidate_only_layer) != 23
    ):
        raise RuntimeError("v18a exact-third patch layer assignment changed")
    return {
        "base_category": base_category,
        "paired_tier": paired_tier,
        "candidate_only_layer": candidate_only_layer,
    }


def _objective_cost_v18a(value) -> float:
    digest = canonical_sha256({
        "schema": "eggroll-es-patch-flow-objective-v18a",
        "seed": FLOW_SEED_V18A,
        "value": value,
    })
    return int(digest[:12], 16) / float(16**12)


def solve_patch_flow_v18a(frame: dict, layers: dict) -> dict:
    components = frame["joint_components"]
    production_units = frame["side_units"]["production"]
    base_indices = sorted(layers["base_category"], key=lambda i: components[i]["joint_id"])
    candidate_only_indices = sorted(
        layers["candidate_only_layer"], key=lambda i: components[i]["joint_id"]
    )
    variable_index = {}
    objective = []
    for component_index in (*base_indices, *candidate_only_indices):
        for panel_index, panel in enumerate(PANEL_NAMES_V18A):
            variable_index[(component_index, panel_index)] = len(objective)
            objective.append(_objective_cost_v18a({
                "joint_id": components[component_index]["joint_id"],
                "panel": panel,
            }))
    constraints = []
    lower = []
    upper = []

    def add_constraint(items, low, high):
        constraints.append(items)
        lower.append(low)
        upper.append(high)

    for component_index in (*base_indices, *candidate_only_indices):
        add_constraint({
            variable_index[(component_index, panel_index)]: 1.0
            for panel_index in range(5)
        }, 0.0, 1.0)
    for panel_index in range(5):
        for category in BASE_CATEGORIES_V18A:
            add_constraint({
                variable_index[(component_index, panel_index)]: 1.0
                for component_index in base_indices
                if layers["base_category"][component_index] == category
            }, BASE_CATEGORY_QUOTA_V18A, BASE_CATEGORY_QUOTA_V18A)
        for stratum, quota in PRODUCTION_TOPIC_QUOTAS_V18A.items():
            add_constraint({
                variable_index[(component_index, panel_index)]: 1.0
                for component_index in base_indices
                if production_units[
                    components[component_index]["production_unit"]
                ]["stratum"] == stratum
            }, quota, quota)
        for layer in (1, 2, 3):
            add_constraint({
                variable_index[(component_index, panel_index)]: 1.0
                for component_index in candidate_only_indices
                if layers["candidate_only_layer"][component_index] == layer
            }, CANDIDATE_ONLY_LAYER_QUOTA_V18A, CANDIDATE_ONLY_LAYER_QUOTA_V18A)

    row_indices = []
    column_indices = []
    coefficients = []
    for row_index, row in enumerate(constraints):
        for column_index, coefficient in row.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            coefficients.append(coefficient)
    matrix = coo_matrix(
        (coefficients, (row_indices, column_indices)),
        shape=(len(constraints), len(objective)),
    ).tocsr()
    result = milp(
        c=np.asarray(objective),
        integrality=np.ones(len(objective)),
        bounds=Bounds(np.zeros(len(objective)), np.ones(len(objective))),
        constraints=LinearConstraint(
            matrix, np.asarray(lower), np.asarray(upper)
        ),
        options={"presolve": True, "time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.status != 0 or result.x is None:
        raise RuntimeError(f"v18a exact constrained flow infeasible: {result.status}")
    rounded = np.rint(result.x)
    if np.max(np.abs(result.x - rounded)) > 1e-7:
        raise RuntimeError("v18a exact constrained flow returned fractional values")
    selected = {panel: [] for panel in PANEL_NAMES_V18A}
    for (component_index, panel_index), column_index in variable_index.items():
        if rounded[column_index] > 0.5:
            selected[PANEL_NAMES_V18A[panel_index]].append(component_index)
    return {
        "selected": selected,
        "solver": {
            "implementation": "scipy.optimize.milp_highs_binary_constrained_flow",
            "scipy_version": scipy.__version__,
            "success": bool(result.success),
            "status": int(result.status),
            "mip_gap": float(result.mip_gap),
            "objective": float(result.fun),
            "variable_count": len(objective),
            "constraint_count": len(constraints),
            "quota_relaxation_used": False,
            "fallback_solver_used": False,
        },
    }


def _representative_for_arm_v18a(
    component: dict, arm_tier: int, paired_tier: int | None,
    candidate_only_layer: int | None, frame: dict,
) -> tuple[str, str] | None:
    if component["relation"] == "candidate_only":
        if candidate_only_layer is None or candidate_only_layer > arm_tier:
            return None
        unit = frame["side_units"]["candidate"][component["candidate_unit"]]
        return "candidate", unit["representative_row_sha256"]
    if paired_tier is not None and paired_tier <= arm_tier:
        unit = frame["side_units"]["candidate"][component["candidate_unit"]]
        return "candidate", unit["representative_row_sha256"]
    unit = frame["side_units"]["production"][component["production_unit"]]
    return "production", unit["representative_row_sha256"]


def build_certificate_v18a() -> dict:
    candidate, production = load_bound_rows_v18a()
    frame = build_joint_frame_v18a(candidate, production)
    layers = assign_patch_layers_v18a(frame)
    flow = solve_patch_flow_v18a(frame, layers)
    components = frame["joint_components"]
    production_units = frame["side_units"]["production"]
    selected_global = []
    panel_summaries = []
    arm_tiers = dict(zip(ARM_ORDER_V18A, range(4)))
    for panel in PANEL_NAMES_V18A:
        selected = flow["selected"][panel]
        base = [index for index in selected if index in layers["base_category"]]
        candidate_only = [
            index for index in selected
            if index in layers["candidate_only_layer"]
        ]
        base_categories = Counter(layers["base_category"][index] for index in base)
        topics = Counter(
            production_units[components[index]["production_unit"]]["stratum"]
            for index in base
        )
        candidate_only_layers = Counter(
            layers["candidate_only_layer"][index] for index in candidate_only
        )
        if (
            base_categories != {category: 13 for category in BASE_CATEGORIES_V18A}
            or topics != PRODUCTION_TOPIC_QUOTAS_V18A
            or candidate_only_layers != {1: 1, 2: 1, 3: 1}
        ):
            raise RuntimeError("v18a solved panel certificate failed exact quotas")
        selected_global.extend(components[index]["joint_id"] for index in selected)
        arm_contract = {}
        for arm, maximum_tier in arm_tiers.items():
            active = [
                index for index in selected
                if index in layers["base_category"]
                or layers["candidate_only_layer"].get(index, 4) <= maximum_tier
            ]
            representatives = []
            representative_sides = Counter()
            for index in active:
                representative = _representative_for_arm_v18a(
                    components[index],
                    maximum_tier,
                    layers["paired_tier"].get(index),
                    layers["candidate_only_layer"].get(index),
                    frame,
                )
                if representative is None:
                    raise RuntimeError("v18a active component lacks representative")
                side, row_sha256 = representative
                representative_sides[side] += 1
                representatives.append(f"{components[index]['joint_id']}:{side}:{row_sha256}")
            if len(active) != ARM_REQUESTS_PER_PANEL_V18A[arm]:
                raise RuntimeError("v18a arm request count changed")
            if len({components[index]["joint_id"] for index in active}) != len(active):
                raise RuntimeError("v18a arm duplicates a paired joint component")
            arm_contract[arm] = {
                "requests": len(active),
                "production_representatives": representative_sides["production"],
                "candidate_representatives": representative_sides["candidate"],
                "active_joint_component_identity_root_sha256": identity_root_sha256(
                    components[index]["joint_id"] for index in active
                ),
                "representative_assignment_root_sha256": identity_root_sha256(
                    representatives
                ),
                "same_arm_paired_duplicate_count": 0,
            }
        panel_summaries.append({
            "name": panel,
            "role": (
                "optimization" if panel.startswith("optimization")
                else "train_only_screen"
            ),
            "base_components": len(base),
            "base_category_counts": dict(base_categories),
            "production_topic_counts": dict(topics),
            "candidate_only_layer_counts": {
                str(layer): candidate_only_layers[layer] for layer in (1, 2, 3)
            },
            "selected_base_joint_component_identity_root_sha256": (
                identity_root_sha256(components[index]["joint_id"] for index in base)
            ),
            "selected_candidate_only_joint_component_identity_root_sha256": (
                identity_root_sha256(
                    components[index]["joint_id"] for index in candidate_only
                )
            ),
            "arms": arm_contract,
        })
    if len(selected_global) != len(set(selected_global)):
        raise RuntimeError("v18a panels are not globally joint-component disjoint")

    category_roots = {
        category: identity_root_sha256(
            components[index]["joint_id"]
            for index, item in layers["base_category"].items()
            if item == category
        )
        for category in BASE_CATEGORIES_V18A
    }
    candidate_only_roots = {
        str(layer): identity_root_sha256(
            components[index]["joint_id"]
            for index, item in layers["candidate_only_layer"].items()
            if item == layer
        )
        for layer in (1, 2, 3)
    }
    ambiguous = [
        item for item in components
        if item["match_class"] == "shared_url_without_shared_document"
    ]
    ambiguous_candidate = frame["side_units"]["candidate"][
        ambiguous[0]["candidate_unit"]
    ]
    ambiguous_production = production_units[ambiguous[0]["production_unit"]]
    certificate = {
        "schema": "eggroll-es-production-patch-flow-certificate-v18a",
        "status": "feasible_frozen_frame_only_no_runtime_or_update_authorization",
        "inputs": {
            "candidate": {
                "path": str(CANDIDATE_PATH_V18A),
                "rows": CANDIDATE_ROWS_V18A,
                "file_sha256": CANDIDATE_SHA256_V18A,
                "manifest_path": str(CANDIDATE_MANIFEST_PATH_V18A),
                "manifest_file_sha256": CANDIDATE_MANIFEST_SHA256_V18A,
            },
            "production": {
                "path": str(PRODUCTION_PATH_V18A),
                "rows": PRODUCTION_ROWS_V18A,
                "file_sha256": PRODUCTION_SHA256_V18A,
            },
            "grouping_implementation": {
                "path": str(Path(sampler_v13.__file__).resolve()),
                "file_sha256": SAMPLER_SHA256_V18A,
            },
        },
        "joint_frame": {
            "grouping": (
                "combined_document_normalized_url_raw_lineage_and_pinned_v13_"
                "lexical_semantic_connected_components_with_side_components_"
                "kept_intact"
            ),
            "joint_component_count": 295,
            "relation_counts": frame["relation_counts"],
            "match_class_counts": frame["match_counts"],
            "production_component_count": 272,
            "production_components_by_topic": PRODUCTION_COMPONENT_CAPACITY_V18A,
            "candidate_component_count": 226,
            "candidate_components_by_topic": CANDIDATE_COMPONENT_CAPACITY_V18A,
            "joint_component_identity_root_sha256": identity_root_sha256(
                item["joint_id"] for item in components
            ),
            "representative_rule": (
                "fixed_side_dominant_topic_representative_within_each_side_"
                "component_no_cross_side_or_topic_fallback"
            ),
        },
        "safe_patch_population": {
            "eligible_shared_document_paired_components": 202,
            "candidate_only_components": 23,
            "eligible_patch_components": 225,
            "excluded_ambiguous_paired_components": 1,
            "excluded_ambiguous_match_class": "shared_url_without_shared_document",
            "excluded_ambiguous_candidate_topic": ambiguous_candidate["stratum"],
            "excluded_ambiguous_production_topic": ambiguous_production["stratum"],
            "excluded_ambiguous_always_uses": "production_representative",
            "paired_tier_populations": {
                "1": 67, "2": 68, "3": 67,
            },
            "candidate_only_layer_populations": {
                str(key): value
                for key, value in CANDIDATE_ONLY_LAYER_POPULATIONS_V18A.items()
            },
            "cumulative_patch_population_by_arm": ELIGIBLE_PATCH_POPULATIONS_V18A,
            "cumulative_fraction_of_eligible_225_by_arm": {
                "production_only": [0, 1],
                "patch_one_third": [1, 3],
                "patch_two_thirds": [2, 3],
                "patch_full": [1, 1],
            },
            "prefix_seed": PREFIX_SEED_V18A,
            "base_category_population_identity_root_sha256": category_roots,
            "candidate_only_layer_population_identity_root_sha256": (
                candidate_only_roots
            ),
        },
        "constrained_flow": {
            "seed": FLOW_SEED_V18A,
            "solver": flow["solver"],
            "panel_count": 5,
            "base_category_quota_per_panel": {
                category: BASE_CATEGORY_QUOTA_V18A
                for category in BASE_CATEGORIES_V18A
            },
            "production_topic_quota_per_panel": PRODUCTION_TOPIC_QUOTAS_V18A,
            "candidate_only_layer_quota_per_panel": {
                "1": 1, "2": 1, "3": 1,
            },
            "selected_base_components": 260,
            "selected_candidate_only_components": 15,
            "selected_joint_component_identity_root_sha256": identity_root_sha256(
                selected_global
            ),
            "globally_panel_disjoint_joint_components": True,
            "quota_seed_or_solver_relaxation_used": False,
            "infeasible_action": (
                "abort_v18a_without_quota_seed_grouping_or_solver_fallback"
            ),
        },
        "estimand": {
            "semantics": (
                "one_representative_per_joint_component_candidate_substitutes_"
                "eligible_paired_production_candidate_only_adds_and_fallback_"
                "remains_production"
            ),
            "arm_population_denominators": ARM_POPULATIONS_V18A,
            "arm_requests_per_panel": ARM_REQUESTS_PER_PANEL_V18A,
            "base_population_ht_strata": {
                category: {
                    "population": population,
                    "per_panel_quota": 13,
                    "horvitz_thompson_weight": population / 13,
                }
                for category, population in BASE_CATEGORY_POPULATIONS_V18A.items()
            },
            "candidate_only_ht_strata": {
                str(layer): {
                    "population": population,
                    "per_panel_quota": 1,
                    "horvitz_thompson_weight": float(population),
                    "active_from_arm_tier": layer,
                }
                for layer, population in CANDIDATE_ONLY_LAYER_POPULATIONS_V18A.items()
            },
            "arm_total": (
                "sum_HT_weight_times_selected_active_component_score_over_"
                "the_arm_specific_active_base_and_candidate_only_strata"
            ),
            "arm_mean": "arm_total_divided_by_exact_arm_population_denominator",
            "candidate_ht_never_targets_all_226": True,
            "plain_request_mean_used": False,
            "shared_component_counted_once_per_arm": True,
            "same_arm_paired_duplicate_count": 0,
        },
        "panels": panel_summaries,
        "firewall": {
            "contains_question_answer_prompt_response_or_row_content": False,
            "contains_evaluation_content": False,
            "heldout_validation_ood_eval_or_benchmark_content_opened": False,
            "runtime_launch_authorized": False,
            "model_update_authorized": False,
            "dataset_promotion_authorized": False,
        },
    }
    certificate["content_sha256_before_self_field"] = canonical_sha256(certificate)
    validate_certificate_v18a(certificate)
    return certificate


def validate_certificate_v18a(value: dict) -> dict:
    without_self = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    solver = value.get("constrained_flow", {}).get("solver", {})
    if (
        value.get("schema")
        != "eggroll-es-production-patch-flow-certificate-v18a"
        or value.get("content_sha256_before_self_field")
        != canonical_sha256(without_self)
        or value.get("safe_patch_population", {}).get(
            "eligible_patch_components"
        ) != 225
        or value.get("safe_patch_population", {}).get(
            "excluded_ambiguous_candidate_topic"
        ) != "technique"
        or value.get("safe_patch_population", {}).get(
            "excluded_ambiguous_production_topic"
        ) != "resources_general"
        or value.get("estimand", {}).get("arm_population_denominators")
        != ARM_POPULATIONS_V18A
        or value.get("estimand", {}).get("arm_requests_per_panel")
        != ARM_REQUESTS_PER_PANEL_V18A
        or solver.get("status") != 0
        or solver.get("success") is not True
        or not math.isclose(solver.get("mip_gap", math.inf), 0.0, abs_tol=1e-15)
        or solver.get("quota_relaxation_used") is not False
        or solver.get("fallback_solver_used") is not False
        or value.get("constrained_flow", {}).get(
            "quota_seed_or_solver_relaxation_used"
        ) is not False
        or any(
            panel.get("base_category_counts")
            != {category: 13 for category in BASE_CATEGORIES_V18A}
            or panel.get("production_topic_counts")
            != PRODUCTION_TOPIC_QUOTAS_V18A
            or any(
                arm.get("same_arm_paired_duplicate_count") != 0
                for arm in panel.get("arms", {}).values()
            )
            for panel in value.get("panels", [])
        )
        or len(value.get("panels", [])) != 5
        or value.get("firewall", {}).get("runtime_launch_authorized") is not False
        or value.get("firewall", {}).get("model_update_authorized") is not False
        or value.get("firewall", {}).get("dataset_promotion_authorized") is not False
        or value.get("firewall", {}).get(
            "heldout_validation_ood_eval_or_benchmark_content_opened"
        ) is not False
    ):
        raise RuntimeError("v18a patch flow certificate changed")
    return value


def _exclusive_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError("immutable v18a flow certificate already exists") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V18A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V18A:
        raise ValueError("v18a flow certificate output path changed")
    certificate = build_certificate_v18a()
    _exclusive_write(OUTPUT_PATH_V18A, certificate)
    result = {
        "schema": "eggroll-es-production-patch-flow-build-v18a",
        "output": str(OUTPUT_PATH_V18A),
        "file_sha256": file_sha256(OUTPUT_PATH_V18A),
        "content_sha256": certificate["content_sha256_before_self_field"],
        "joint_component_count": certificate["joint_frame"][
            "joint_component_count"
        ],
        "solver_status": certificate["constrained_flow"]["solver"]["status"],
        "solver_mip_gap": certificate["constrained_flow"]["solver"]["mip_gap"],
        "runtime_launch_authorized": False,
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
