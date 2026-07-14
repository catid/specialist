#!/usr/bin/env python3
"""Rotating document-first sampling for multi-step EGGROLL-ES training.

V13 freezes disjoint panels for a controlled estimator experiment.  This
module supplies the corresponding online policy: select documents first,
select one row within each document second, and freeze that exact ordered batch
across every direction and antithetic sign in one ES iteration.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

import eggroll_es_train_panel_sampler_v13 as v13


SCHEMA = "eggroll-es-hierarchical-rotating-panel-v14"
POLICY_SCHEMA = "eggroll-es-hierarchical-rotating-policy-v14"
MASTER_SEED = "specialist-s6-hierarchical-rotating-v14-20260714"
STRATA = v13.STRATA
STRATUM_QUOTAS = dict(v13.STRATUM_QUOTAS)
PANEL_SIZE = sum(STRATUM_QUOTAS.values())
HARD_REPLAY_FRACTION = 0.0
HARD_REPLAY_CAP = 0.25


def _key(*parts):
    payload = "\0".join((MASTER_SEED, *(str(part) for part in parts)))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def _dominant_stratum(counts):
    return max(
        STRATA,
        key=lambda name: (counts.get(name, 0), v13._TIE_PRIORITY[name]),
    )


def build_document_frame(rows):
    """Group every train row by document without discarding within-page rows."""
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row["document_sha256"]].append(index)
    documents = []
    for document_sha256, indices in grouped.items():
        counts = Counter(v13.classify_stratum(rows[index]) for index in indices)
        ordered = sorted(indices, key=lambda index: v13.row_sha256(rows[index]))
        documents.append({
            "document_sha256": document_sha256,
            "row_indices": ordered,
            "row_sha256s": [v13.row_sha256(rows[index]) for index in ordered],
            "row_count": len(ordered),
            "stratum": _dominant_stratum(counts),
            "row_stratum_counts": {
                name: counts.get(name, 0) for name in STRATA
            },
        })
    documents.sort(key=lambda item: item["document_sha256"])
    counts = Counter(item["stratum"] for item in documents)
    for name, quota in STRATUM_QUOTAS.items():
        if counts[name] < quota:
            raise ValueError(f"v14 stratum {name} cannot fill its quota")
    identity = [{
        "document_sha256": item["document_sha256"],
        "row_sha256s": item["row_sha256s"],
        "stratum": item["stratum"],
    } for item in documents]
    return documents, {
        "documents": len(documents),
        "rows": len(rows),
        "maximum_rows_per_document": max(item["row_count"] for item in documents),
        "stratum_document_counts": {
            name: counts[name] for name in STRATA
        },
        "frame_sha256": v13.canonical_sha256(identity),
    }


def _document_permutation(documents, stratum):
    selected = [item for item in documents if item["stratum"] == stratum]
    return sorted(
        selected,
        key=lambda item: _key(
            "document-permutation", stratum, item["document_sha256"],
        ),
    )


def _cyclic_block(values, start, count):
    if count > len(values):
        raise ValueError("v14 cyclic block exceeds its population")
    return [values[(start + offset) % len(values)] for offset in range(count)]


def _choose_row(document, rows, iteration):
    ordered = sorted(
        document["row_indices"],
        key=lambda index: _key(
            "within-document-row", iteration,
            document["document_sha256"], v13.row_sha256(rows[index]),
        ),
    )
    return ordered[0]


def build_iteration_panel(rows, iteration):
    """Build one deterministic panel shared by a complete ES iteration."""
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        raise ValueError("v14 iteration must be a nonnegative integer")
    documents, frame = build_document_frame(rows)
    items = []
    for stratum in STRATA:
        population = _document_permutation(documents, stratum)
        quota = STRATUM_QUOTAS[stratum]
        start = (iteration * quota) % len(population)
        for document in _cyclic_block(population, start, quota):
            row_index = _choose_row(document, rows, iteration)
            row = rows[row_index]
            document_probability = quota / len(population)
            row_probability = 1.0 / document["row_count"]
            items.append({
                "document_sha256": document["document_sha256"],
                "document_row_count": document["row_count"],
                "fact_id": row["fact_id"],
                "row_index": row_index,
                "row_sha256": v13.row_sha256(row),
                "source": row.get("source"),
                "stratum": stratum,
                "document_selection_probability": document_probability,
                "within_document_row_selection_probability": row_probability,
                "joint_row_selection_probability": (
                    document_probability * row_probability
                ),
                # For an equal-document estimand, the sampled row is an
                # unbiased draw of its document mean.  No row-count multiplier
                # belongs in this contribution weight.
                "equal_document_ht_weight": 1.0 / document_probability,
            })
    items.sort(key=lambda item: _key(
        "panel-order", iteration, item["document_sha256"], item["row_sha256"],
    ))
    for position, item in enumerate(items):
        item["position"] = position
    panel = {
        "schema": SCHEMA,
        "train_only": True,
        "iteration": iteration,
        "master_seed": MASTER_SEED,
        "rows": len(items),
        "frame": frame,
        "stratum_quotas": dict(STRATUM_QUOTAS),
        "ordered_row_identity_sha256": v13.canonical_sha256([
            item["row_sha256"] for item in items
        ]),
        "items": items,
        "common_random_numbers": (
            "reuse this exact ordered row identity for every direction and "
            "both antithetic signs in this iteration"
        ),
        "hard_replay_fraction": HARD_REPLAY_FRACTION,
        "validation_ood_or_heldout_used": False,
    }
    panel["content_sha256_before_self_field"] = v13.canonical_sha256(panel)
    validate_panel(panel, rows, recompute=False)
    return panel


def validate_panel(panel, rows, *, recompute=True):
    if not isinstance(panel, dict):
        raise ValueError("v14 panel must be an object")
    expected_keys = {
        "schema", "train_only", "iteration", "master_seed", "rows", "frame",
        "stratum_quotas", "ordered_row_identity_sha256", "items",
        "common_random_numbers", "hard_replay_fraction",
        "validation_ood_or_heldout_used", "content_sha256_before_self_field",
    }
    if set(panel) != expected_keys:
        raise ValueError("v14 panel shape changed")
    content = {
        key: value for key, value in panel.items()
        if key != "content_sha256_before_self_field"
    }
    if panel["content_sha256_before_self_field"] != v13.canonical_sha256(content):
        raise ValueError("v14 panel content hash changed")
    if (
        panel["schema"] != SCHEMA
        or panel["train_only"] is not True
        or panel["master_seed"] != MASTER_SEED
        or panel["rows"] != PANEL_SIZE
        or panel["stratum_quotas"] != STRATUM_QUOTAS
        or panel["hard_replay_fraction"] != 0.0
        or panel["validation_ood_or_heldout_used"] is not False
    ):
        raise ValueError("v14 frozen policy changed")
    items = panel["items"]
    if (
        len(items) != PANEL_SIZE
        or len({item["document_sha256"] for item in items}) != PANEL_SIZE
        or Counter(item["stratum"] for item in items) != Counter(STRATUM_QUOTAS)
        or [item["position"] for item in items] != list(range(PANEL_SIZE))
    ):
        raise ValueError("v14 panel allocation changed")
    for item in items:
        row = rows[item["row_index"]]
        if (
            item["row_sha256"] != v13.row_sha256(row)
            or item["fact_id"] != row["fact_id"]
            or item["document_sha256"] != row["document_sha256"]
            or not math.isclose(
                item["joint_row_selection_probability"],
                item["document_selection_probability"]
                * item["within_document_row_selection_probability"],
                rel_tol=0.0, abs_tol=1e-15,
            )
            or not math.isclose(
                item["equal_document_ht_weight"],
                1.0 / item["document_selection_probability"],
                rel_tol=0.0, abs_tol=1e-15,
            )
        ):
            raise ValueError("v14 panel item binding changed")
    if recompute and panel != build_iteration_panel(rows, panel["iteration"]):
        raise ValueError("v14 panel does not match deterministic rebuild")
    return True


def materialize_iteration(iteration, path=v13.DEFAULT_SOURCE):
    rows, source_sha256 = v13.load_frozen_train(path)
    panel = build_iteration_panel(rows, iteration)
    return {
        "panel": panel,
        "source_sha256": source_sha256,
        "questions": [rows[item["row_index"]]["question"] for item in panel["items"]],
        "answers": [rows[item["row_index"]]["answer"] for item in panel["items"]],
        "weights": [item["equal_document_ht_weight"] for item in panel["items"]],
    }


def equal_document_mean(values, panel):
    """Horvitz-Thompson/Hajek mean for the equal-document target."""
    if len(values) != len(panel["items"]):
        raise ValueError("v14 value count differs from panel")
    numeric = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("v14 panel score is non-finite")
    weights = [item["equal_document_ht_weight"] for item in panel["items"]]
    denominator = math.fsum(weights)
    if not math.isclose(
        denominator, float(panel["frame"]["documents"]),
        rel_tol=0.0, abs_tol=1e-12,
    ):
        raise RuntimeError("v14 HT weights no longer sum to the document frame")
    return math.fsum(w * value for w, value in zip(weights, numeric)) / denominator


def common_random_number_schedule(panel, direction_ids):
    validate = panel["ordered_row_identity_sha256"]
    return [{
        "direction_id": str(direction_id),
        "sign": sign,
        "iteration": panel["iteration"],
        "ordered_row_identity_sha256": validate,
    } for direction_id in direction_ids for sign in ("plus", "minus")]


def build_policy(rows):
    _, frame = build_document_frame(rows)
    policy = {
        "schema": POLICY_SCHEMA,
        "train_only": True,
        "master_seed": MASTER_SEED,
        "frame": frame,
        "panel_size": PANEL_SIZE,
        "stratum_quotas": dict(STRATUM_QUOTAS),
        "refresh": "new document block and within-document row draw each ES iteration",
        "common_random_numbers": "frozen within an iteration",
        "estimand": "equal-weight mean of within-document row means",
        "hard_replay": {
            "enabled": False,
            "fraction": HARD_REPLAY_FRACTION,
            "maximum_fraction": HARD_REPLAY_CAP,
            "enable_only_with": (
                "lagged out-of-fold train-only difficulty and exact propensities"
            ),
        },
        "selection_firewall": {
            "allowed": ["frozen train rows", "lagged train-only difficulty"],
            "forbidden": ["validation", "OOD", "heldout", "benchmark outcomes"],
        },
    }
    policy["content_sha256_before_self_field"] = v13.canonical_sha256(policy)
    return policy
