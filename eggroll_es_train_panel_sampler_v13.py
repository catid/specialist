#!/usr/bin/env python3
"""Deterministic train-only, document-balanced panels for EGGROLL-ES.

The sampling frame collapses any rows connected by a shared source document or
by a conservative lexical-semantic match.  Sampling one representative from a
frame unit therefore cannot put two rows from the same document or semantic
cluster in a panel.  A keyed SHA-256 ordering replaces mutable RNG state.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


SCHEMA = "eggroll-es-document-balanced-train-panels-v13"
SOURCE_ROWS = 794
SOURCE_SHA256 = (
    "f7127c38c7b540eaf9cf4349d1a1b8076e171da7f8ea43c11068ad1c311bb776"
)
SOURCE_ARROW_SHA256 = (
    "6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6"
)
DEFAULT_SOURCE = Path(
    "/tmp/specialist-s6-candidate-guarded-ead1b21/"
    "train_qa_curated_v1.jsonl"
)
PANEL_NAMES = (
    "optimization_0",
    "optimization_1",
    "optimization_2",
    "train_screen_0",
    "train_screen_1",
)
PANEL_ROLES = {
    "optimization_0": "optimization",
    "optimization_1": "optimization",
    "optimization_2": "optimization",
    "train_screen_0": "train_only_screen",
    "train_screen_1": "train_only_screen",
}
STRATA = (
    "safety_consent",
    "technique",
    "equipment_material",
    "resources_general",
)
STRATUM_QUOTAS = {
    "safety_consent": 9,
    "technique": 16,
    "equipment_material": 6,
    "resources_general": 25,
}
PANEL_SIZE = sum(STRATUM_QUOTAS.values())
MASTER_SEED = "specialist-s6-document-balanced-panels-v13-20260714"
HARD_EXAMPLE_CAP = 0.25

_PATTERNS = {
    "safety_consent": (
        r"\b(consent|consensual|negotia\w*|safety|safe\w*|risk\w*|"
        r"danger\w*|emergency|injur\w*|nerve\w*|numb\w*|circulation|"
        r"blood flow|breath\w*|communicat\w*|check[ -]?in|aftercare|"
        r"medical|trauma|shears?|quick[ -]?release|release|hygiene|"
        r"clean\w*|wash\w*|infection|allerg\w*|fire|panic|spotter|"
        r"fall|collapse|unconscious|distress|warning|harm\w*|pain|"
        r"tingl\w*)\b"
    ),
    "equipment_material": (
        r"\b(jute|hemp|nylon|cotton|synthetic|natural fiber|rope material|"
        r"diameter|millimet\w*|\d+\s*mm|length|bamboo|carabiner|hardware|"
        r"rigging plate|frame|upline|suspension point|ring|pulley|"
        r"maintenance|condition\w*|oil\w*|rope end|whipping|equipment|"
        r"gear|vendor|product|purchase|buy|store|shop|braid|fiber|fibre)\b"
    ),
    "technique": (
        r"\b(knot|tie|tying|tied|harness|column|friction|wrap|bight|loop|"
        r"cinch|munter|bowline|lark|square knot|overhand|half hitch|"
        r"single column|double column|box tie|futomomo|takate|gote|"
        r"suspension|uplines?|load|tension|dress\w*|coiling|coil|lock|"
        r"stem|cross|weav\w*|pattern|position|technique|method|step|"
        r"construct|attach|apply)\b"
    ),
}
_REGEXES = {name: re.compile(value) for name, value in _PATTERNS.items()}
_TIE_PRIORITY = {
    "safety_consent": 3,
    "equipment_material": 2,
    "technique": 1,
    "resources_general": 0,
}
_STOPWORDS = frozenset(
    "a an and are as at be been being by can could did do does for from had "
    "has have how in into is it its of on or should that the their them there "
    "these they this to was were what when where which who why will with would "
    "according author article day lesson page rope ropes text".split()
)


def canonical_sha256(value) -> str:
    raw = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _forbid_nontrain_path(path) -> None:
    lowered = str(path).lower()
    forbidden = ("heldout", "validation", "ood_qa", "eval-output")
    if any(token in lowered for token in forbidden):
        raise ValueError("v13 panel builder accepts train-only sources")


def load_frozen_train(path=DEFAULT_SOURCE, *, enforce_frozen=True):
    path = Path(path)
    _forbid_nontrain_path(path)
    raw_sha = file_sha256(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if enforce_frozen and (raw_sha != SOURCE_SHA256 or len(rows) != SOURCE_ROWS):
        raise ValueError("frozen 794-row training source identity changed")
    required = {"question", "answer", "fact_id", "document_sha256"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"invalid training row {index}")
        if not all(isinstance(row[key], str) and row[key] for key in required):
            raise ValueError(f"empty required training field in row {index}")
    return rows, raw_sha


def row_sha256(row) -> str:
    return canonical_sha256({
        "question": row["question"],
        "answer": row["answer"],
        "fact_id": row["fact_id"],
        "document_sha256": row["document_sha256"],
    })


def classify_stratum(row) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in ("question", "answer", "source", "url")
    ).lower()
    # Priority is intentional: consent/safety and equipment mentions should not
    # disappear into the much larger technique/general strata.
    for name in ("safety_consent", "equipment_material", "technique"):
        if _REGEXES[name].search(text):
            return name
    return "resources_general"


def _content_tokens(text):
    return frozenset(
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    )


def _jaccard(left, right):
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _semantic_match(left, right):
    q_left, a_left = left
    q_right, a_right = right
    q_similarity = _jaccard(q_left, q_right)
    if q_similarity >= 0.82:
        return True
    return q_similarity >= 0.66 and _jaccard(a_left, a_right) >= 0.86


class _DisjointSet:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left, right):
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            if right_root < left_root:
                left_root, right_root = right_root, left_root
            self.parent[right_root] = left_root


def build_semantic_clusters(rows):
    """Return stable lexical-semantic cluster IDs, without model inference."""
    identities = [row_sha256(row) for row in rows]
    features = [
        (_content_tokens(row["question"]), _content_tokens(row["answer"]))
        for row in rows
    ]
    order = sorted(range(len(rows)), key=lambda index: identities[index])
    representatives = []
    assignments = {}
    for index in order:
        match = None
        for representative in representatives:
            if _semantic_match(features[index], features[representative]):
                match = representative
                break
        if match is None:
            match = index
            representatives.append(index)
        assignments[index] = match
    members = defaultdict(list)
    for index, representative in assignments.items():
        members[representative].append(identities[index])
    cluster_ids = {
        representative: canonical_sha256({
            "schema": "lexical-semantic-cluster-v13",
            "members": sorted(member_ids),
        })
        for representative, member_ids in members.items()
    }
    return [cluster_ids[assignments[index]] for index in range(len(rows))]


def build_conflict_units(rows):
    """Collapse shared-document or shared-semantic rows into sampling units."""
    semantic_ids = build_semantic_clusters(rows)
    disjoint = _DisjointSet(len(rows))
    first_document = {}
    first_semantic = {}
    for index, (row, semantic_id) in enumerate(zip(rows, semantic_ids)):
        document_id = row["document_sha256"]
        if document_id in first_document:
            disjoint.union(index, first_document[document_id])
        else:
            first_document[document_id] = index
        if semantic_id in first_semantic:
            disjoint.union(index, first_semantic[semantic_id])
        else:
            first_semantic[semantic_id] = index

    components = defaultdict(list)
    for index in range(len(rows)):
        components[disjoint.find(index)].append(index)

    units = []
    for indices in components.values():
        identities = sorted(row_sha256(rows[index]) for index in indices)
        unit_id = canonical_sha256({
            "schema": "document-semantic-conflict-unit-v13",
            "members": identities,
        })
        counts = Counter(classify_stratum(rows[index]) for index in indices)
        stratum = max(
            STRATA, key=lambda name: (counts[name], _TIE_PRIORITY[name]),
        )
        candidates = [
            index for index in indices
            if classify_stratum(rows[index]) == stratum
        ]
        representative = min(
            candidates,
            key=lambda index: canonical_sha256({
                "seed": MASTER_SEED,
                "purpose": "unit-representative",
                "unit_id": unit_id,
                "row_id": row_sha256(rows[index]),
            }),
        )
        units.append({
            "unit_id": unit_id,
            "stratum": stratum,
            "representative_index": representative,
            "representative_row_sha256": row_sha256(rows[representative]),
            "representative_semantic_cluster_sha256": (
                semantic_ids[representative]
            ),
            "document_sha256s": sorted({
                rows[index]["document_sha256"] for index in indices
            }),
            "semantic_cluster_sha256s": sorted({
                semantic_ids[index] for index in indices
            }),
            "row_count": len(indices),
            "stratum_row_counts": dict(sorted(counts.items())),
        })
    return sorted(units, key=lambda unit: unit["unit_id"]), semantic_ids


def _keyed_order(unit):
    return canonical_sha256({
        "seed": MASTER_SEED,
        "purpose": "crossed-panel-order",
        "stratum": unit["stratum"],
        "unit_id": unit["unit_id"],
    })


def build_manifest(rows, source_path=DEFAULT_SOURCE, source_sha256=None):
    if len(rows) != SOURCE_ROWS:
        raise ValueError("v13 requires the frozen 794-row train snapshot")
    units, semantic_ids = build_conflict_units(rows)
    by_stratum = {
        name: sorted(
            (unit for unit in units if unit["stratum"] == name),
            key=_keyed_order,
        )
        for name in STRATA
    }
    for name in STRATA:
        needed = len(PANEL_NAMES) * STRATUM_QUOTAS[name]
        if len(by_stratum[name]) < needed:
            raise RuntimeError(
                f"insufficient globally disjoint {name} units: "
                f"need {needed}, found {len(by_stratum[name])}"
            )

    panels = []
    for panel_index, panel_name in enumerate(PANEL_NAMES):
        chosen = []
        for stratum in STRATA:
            quota = STRATUM_QUOTAS[stratum]
            start = panel_index * quota
            population = len(by_stratum[stratum])
            probability = quota / population
            for unit in by_stratum[stratum][start:start + quota]:
                row_index = unit["representative_index"]
                row = rows[row_index]
                chosen.append({
                    "row_index": row_index,
                    "fact_id": row["fact_id"],
                    "row_sha256": unit["representative_row_sha256"],
                    "document_sha256": row["document_sha256"],
                    "semantic_cluster_sha256": (
                        unit["representative_semantic_cluster_sha256"]
                    ),
                    "conflict_unit_sha256": unit["unit_id"],
                    "stratum": stratum,
                    "source": row.get("source"),
                    "url_sha256": hashlib.sha256(
                        str(row.get("url", "")).encode("utf-8")
                    ).hexdigest(),
                    "unit_selection_probability_for_this_panel": probability,
                    "horvitz_thompson_unit_weight": 1.0 / probability,
                })
        # Interleave strata deterministically so early generation microbatches
        # are not single-stratum blocks. This order is itself part of the CRN.
        chosen.sort(key=lambda item: canonical_sha256({
            "seed": MASTER_SEED,
            "purpose": "panel-row-order",
            "panel": panel_name,
            "unit_id": item["conflict_unit_sha256"],
        }))
        for position, item in enumerate(chosen):
            item["position"] = position
        ordered_identity = [item["row_sha256"] for item in chosen]
        panels.append({
            "name": panel_name,
            "role": PANEL_ROLES[panel_name],
            "rows": len(chosen),
            "stratum_counts": dict(STRATUM_QUOTAS),
            "ordered_row_identity_sha256": canonical_sha256(ordered_identity),
            "items": chosen,
        })

    source_path = Path(source_path)
    manifest = {
        "schema": SCHEMA,
        "train_only": True,
        "source": {
            "path": str(source_path),
            "jsonl_sha256": source_sha256 or file_sha256(source_path),
            "rows": len(rows),
            "arrow_sha256": SOURCE_ARROW_SHA256,
        },
        "motivation_evidence": {
            "same_basis_data43_data44_coefficient_cosine": (
                0.4276943787514416
            ),
            "v10_raw_central_domain_cosine": 0.34415939658856237,
            "interpretation": "data-panel variation is a first-order noise source",
        },
        "sampling_frame": {
            "raw_rows": len(rows),
            "source_documents": len({
                row["document_sha256"] for row in rows
            }),
            "semantic_clusters": len(set(semantic_ids)),
            "conflict_units": len(units),
            "unit_stratum_counts": {
                name: len(by_stratum[name]) for name in STRATA
            },
            "estimand": "equal-weight document-semantic conflict-unit mean",
            "representative_policy": (
                "dominant explicit stratum; keyed SHA-256 row representative"
            ),
            "semantic_rule": {
                "question_jaccard_direct": 0.82,
                "question_jaccard_with_answer": 0.66,
                "answer_jaccard": 0.86,
                "algorithm": "deterministic greedy representative, SHA order",
            },
            "classifier_patterns_sha256": canonical_sha256(_PATTERNS),
            "frame_sha256": canonical_sha256([
                {
                    key: unit[key]
                    for key in (
                        "unit_id", "stratum", "representative_index",
                        "representative_row_sha256", "document_sha256s",
                        "semantic_cluster_sha256s",
                    )
                }
                for unit in units
            ]),
        },
        "panel_design": {
            "master_seed": MASTER_SEED,
            "panel_names": list(PANEL_NAMES),
            "optimization_panels": list(PANEL_NAMES[:3]),
            "train_only_screen_panels": list(PANEL_NAMES[3:]),
            "rows_per_panel": PANEL_SIZE,
            "stratum_quotas": dict(STRATUM_QUOTAS),
            "globally_disjoint_conflict_units": True,
            "crossing": "every direction is evaluated on all optimization panels",
            "common_random_numbers": (
                "within a panel, every direction and both signs reuse the exact "
                "ordered row identity"
            ),
            "importance_weights": (
                "exact inverse inclusion weights for the conflict-unit estimand; "
                "pi(stratum,panel)=quota/stratum_units"
            ),
        },
        "hard_example_mixture": {
            "enabled": False,
            "maximum_fraction": HARD_EXAMPLE_CAP,
            "configured_fraction": 0.0,
            "reason": (
                "No independently replicated, content-free train-only per-unit "
                "difficulty artifact is pinned. Deriving hardness from validation, "
                "OOD, heldout, or these same optimization responses would create "
                "selection leakage and unauditable adaptive inclusion probabilities."
            ),
            "enablement_requirements": [
                "frozen train-only difficulty artifact bound to frame_sha256",
                "difficulty estimated out-of-fold or on independent train panels",
                "fixed hard tier before any candidate response is observed",
                "exact per-tier inclusion probabilities and inverse weights",
                "hard allocation no more than 25 percent of every panel",
            ],
        },
        "selection_firewall": {
            "allowed": ["frozen train metadata", "train optimization panels", "train screen panels"],
            "forbidden": ["validation", "OOD", "heldout", "benchmark outcomes"],
        },
        "panels": panels,
    }
    manifest["content_sha256_before_self_field"] = canonical_sha256(manifest)
    validate_manifest(manifest, rows)
    return manifest


def validate_manifest(manifest, rows=None):
    if manifest.get("schema") != SCHEMA or manifest.get("train_only") is not True:
        raise ValueError("invalid v13 train-panel schema")
    if manifest.get("content_sha256_before_self_field") != canonical_sha256({
        key: value for key, value in manifest.items()
        if key != "content_sha256_before_self_field"
    }):
        raise ValueError("v13 manifest content hash changed")
    panels = manifest.get("panels", [])
    if [panel.get("name") for panel in panels] != list(PANEL_NAMES):
        raise ValueError("v13 panel names/order changed")
    all_units, all_documents, all_semantics, all_rows = set(), set(), set(), set()
    for panel in panels:
        items = panel.get("items", [])
        if len(items) != PANEL_SIZE:
            raise ValueError("v13 panel size changed")
        if Counter(item["stratum"] for item in items) != Counter(STRATUM_QUOTAS):
            raise ValueError("v13 panel strata changed")
        if [item["position"] for item in items] != list(range(PANEL_SIZE)):
            raise ValueError("v13 panel order positions changed")
        identities = [item["row_sha256"] for item in items]
        if panel["ordered_row_identity_sha256"] != canonical_sha256(identities):
            raise ValueError("v13 ordered panel identity changed")
        for item in items:
            for seen, key in (
                (all_units, "conflict_unit_sha256"),
                (all_documents, "document_sha256"),
                (all_semantics, "semantic_cluster_sha256"),
                (all_rows, "row_sha256"),
            ):
                if item[key] in seen:
                    raise ValueError(f"v13 global panel overlap: {key}")
                seen.add(item[key])
            probability = item["unit_selection_probability_for_this_panel"]
            if not 0.0 < probability <= 1.0:
                raise ValueError("v13 invalid inclusion probability")
            if not math.isclose(
                item["horvitz_thompson_unit_weight"], 1.0 / probability,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                raise ValueError("v13 invalid importance weight")
            if rows is not None:
                row = rows[item["row_index"]]
                if row["fact_id"] != item["fact_id"] or row_sha256(row) != item["row_sha256"]:
                    raise ValueError("v13 manifest no longer addresses source row")
    if manifest["hard_example_mixture"]["enabled"] is not False:
        raise ValueError("v13 frozen hard-example mixture must remain disabled")
    if manifest["hard_example_mixture"]["configured_fraction"] > HARD_EXAMPLE_CAP:
        raise ValueError("v13 hard-example cap exceeded")
    return True


def materialize_panel(manifest, panel_name, source_path=DEFAULT_SOURCE):
    rows, source_sha = load_frozen_train(source_path)
    if source_sha != manifest["source"]["jsonl_sha256"]:
        raise ValueError("v13 materialization source identity changed")
    validate_manifest(manifest, rows)
    matches = [panel for panel in manifest["panels"] if panel["name"] == panel_name]
    if len(matches) != 1:
        raise ValueError("unknown v13 panel")
    selected = [rows[item["row_index"]] for item in matches[0]["items"]]
    return {
        "questions": [row["question"] for row in selected],
        "answers": [row["answer"] for row in selected],
        "fact_ids": [row["fact_id"] for row in selected],
        "ordered_row_identity_sha256": matches[0]["ordered_row_identity_sha256"],
    }


def common_random_number_schedule(manifest, panel_name, direction_ids):
    """Bind every direction/sign to one immutable ordered training panel."""
    validate_manifest(manifest)
    panel = next(
        (item for item in manifest["panels"] if item["name"] == panel_name),
        None,
    )
    if panel is None:
        raise ValueError("unknown v13 panel")
    if not direction_ids or len(set(direction_ids)) != len(direction_ids):
        raise ValueError("direction IDs must be nonempty and unique")
    return [
        {
            "direction_id": str(direction_id),
            "sign": sign,
            "panel": panel_name,
            "ordered_row_identity_sha256": panel["ordered_row_identity_sha256"],
        }
        for direction_id in direction_ids
        for sign in ("plus", "minus")
    ]
