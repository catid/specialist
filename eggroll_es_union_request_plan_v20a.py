#!/usr/bin/env python3
"""In-memory exact request unioning for nested train-only attribution arms.

The compact audit returned by this module contains hashes and counts only.  The
teacher-forced token IDs and dense scoring metadata stay in the ephemeral
runtime plan and must never be persisted in a compact report.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence

from eggroll_es_disjoint_tier_attribution_preregistration_v19a import (
    canonical_sha256,
)


def _require_names(values, label):
    values = tuple(values)
    if (
        not values
        or len(values) != len(set(values))
        or any(not isinstance(value, str) or not value for value in values)
    ):
        raise ValueError(f"v20a {label} must be unique non-empty strings")
    return values


def _token_identity(item):
    if not isinstance(item, Mapping):
        raise ValueError("v20a dense request item is not a mapping")
    token_ids = item.get("prompt_token_ids")
    digest = item.get("prompt_token_ids_sha256")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
        or not isinstance(digest, str)
        or len(digest) != 64
        or canonical_sha256(token_ids) != digest
    ):
        raise ValueError("v20a dense request token identity changed")
    return digest, token_ids


def _validate_prepared(prepared, arm_order, panel_order):
    if not isinstance(prepared, Mapping) or set(prepared) != set(arm_order):
        raise ValueError("v20a prepared arm coverage changed")
    flat_by_arm = {}
    for arm in arm_order:
        arm_value = prepared[arm]
        if (
            not isinstance(arm_value, Mapping)
            or set(arm_value) != {"panels", "prompt_items"}
            or not isinstance(arm_value["panels"], Mapping)
            or set(arm_value["panels"]) != set(panel_order)
            or not isinstance(arm_value["prompt_items"], list)
        ):
            raise ValueError(f"v20a prepared structure changed for {arm}")
        rebuilt = []
        cursor = 0
        for panel in panel_order:
            value = arm_value["panels"][panel]
            if (
                not isinstance(value, Mapping)
                or set(value) != {"dense_items", "slice"}
                or not isinstance(value["dense_items"], list)
                or not value["dense_items"]
                or value["slice"] != (cursor, cursor + len(value["dense_items"]))
            ):
                raise ValueError(f"v20a panel slice changed for {arm}/{panel}")
            for item in value["dense_items"]:
                _token_identity(item)
            rebuilt.extend(value["dense_items"])
            cursor += len(value["dense_items"])
        if rebuilt != arm_value["prompt_items"]:
            raise ValueError(f"v20a flat request order changed for {arm}")
        flat_by_arm[arm] = rebuilt
    return flat_by_arm


def build_union_request_plan_v20a(prepared, arm_order, panel_order):
    """Deduplicate exact token sequences and bind lossless arm reconstruction."""
    arm_order = _require_names(arm_order, "arm order")
    panel_order = _require_names(panel_order, "panel order")
    flat_by_arm = _validate_prepared(prepared, arm_order, panel_order)

    union_items = []
    union_token_ids = []
    digest_to_index = {}
    mappings = {}
    per_panel_raw = {panel: 0 for panel in panel_order}
    per_arm_raw = {}
    per_panel_unique_sets = {panel: set() for panel in panel_order}
    for arm in arm_order:
        mappings[arm] = {}
        per_arm_raw[arm] = len(flat_by_arm[arm])
        for panel in panel_order:
            items = prepared[arm]["panels"][panel]["dense_items"]
            indices = []
            for item in items:
                digest, token_ids = _token_identity(item)
                index = digest_to_index.get(digest)
                if index is None:
                    index = len(union_items)
                    digest_to_index[digest] = index
                    union_items.append(copy.deepcopy(item))
                    union_token_ids.append(list(token_ids))
                elif union_token_ids[index] != token_ids:
                    raise RuntimeError("v20a token SHA-256 collision detected")
                indices.append(index)
                per_panel_unique_sets[panel].add(digest)
            mappings[arm][panel] = indices
            per_panel_raw[panel] += len(items)

    raw_count = sum(per_arm_raw.values())
    unique_count = len(union_items)
    if (
        unique_count <= 0
        or unique_count > raw_count
        or sorted(set(
            index
            for arm in arm_order
            for panel in panel_order
            for index in mappings[arm][panel]
        )) != list(range(unique_count))
    ):
        raise RuntimeError("v20a union request coverage changed")
    audit = {
        "schema": "eggroll-es-union-request-plan-audit-v20a",
        "arm_order": list(arm_order),
        "panel_order": list(panel_order),
        "raw_request_count": raw_count,
        "unique_request_count": unique_count,
        "eliminated_duplicate_request_count": raw_count - unique_count,
        "raw_to_unique_ratio": raw_count / unique_count,
        "per_arm_raw_request_count": per_arm_raw,
        "per_panel_raw_request_count": per_panel_raw,
        "per_panel_unique_request_count": {
            panel: len(values) for panel, values in per_panel_unique_sets.items()
        },
        "union_request_identity_sha256": canonical_sha256([
            item["prompt_token_ids_sha256"] for item in union_items
        ]),
        "reconstruction_identity_sha256": canonical_sha256(mappings),
        "exact_token_collision_checks_passed": True,
        "contains_token_ids_or_row_content": False,
    }
    audit["content_sha256_before_self_field"] = canonical_sha256(audit)
    runtime = {
        "union_prompt_items": union_items,
        "arm_panel_union_indices": mappings,
    }
    validate_union_request_plan_v20a(runtime, audit, prepared)
    return runtime, audit


def reconstruct_arm_outputs_v20a(runtime, arm, panel_order, union_outputs):
    """Gather one arm's panel outputs from a complete union output batch."""
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {"union_prompt_items", "arm_panel_union_indices"}
        or not isinstance(union_outputs, Sequence)
        or isinstance(union_outputs, (str, bytes))
        or len(union_outputs) != len(runtime["union_prompt_items"])
        or arm not in runtime["arm_panel_union_indices"]
    ):
        raise ValueError("v20a union output coverage changed")
    panel_order = _require_names(panel_order, "panel order")
    mapping = runtime["arm_panel_union_indices"][arm]
    if set(mapping) != set(panel_order):
        raise ValueError("v20a arm reconstruction panel coverage changed")
    return {
        panel: [union_outputs[index] for index in mapping[panel]]
        for panel in panel_order
    }


def validate_union_request_plan_v20a(runtime, audit, prepared):
    """Prove that union indices reconstruct every original token sequence."""
    if (
        not isinstance(runtime, Mapping)
        or set(runtime) != {"union_prompt_items", "arm_panel_union_indices"}
        or not isinstance(audit, Mapping)
        or audit.get("schema") != "eggroll-es-union-request-plan-audit-v20a"
        or audit.get("contains_token_ids_or_row_content") is not False
        or audit.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in audit.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v20a compact union audit changed")
    union = runtime["union_prompt_items"]
    mappings = runtime["arm_panel_union_indices"]
    arm_order = tuple(audit["arm_order"])
    panel_order = tuple(audit["panel_order"])
    _validate_prepared(prepared, arm_order, panel_order)
    if (
        len(union) != audit["unique_request_count"]
        or sum(audit["per_arm_raw_request_count"].values())
        != audit["raw_request_count"]
        or audit["eliminated_duplicate_request_count"]
        != audit["raw_request_count"] - audit["unique_request_count"]
        or canonical_sha256([
            item["prompt_token_ids_sha256"] for item in union
        ]) != audit["union_request_identity_sha256"]
        or canonical_sha256(mappings) != audit["reconstruction_identity_sha256"]
    ):
        raise RuntimeError("v20a union audit counts or identities changed")
    for arm in arm_order:
        for panel in panel_order:
            originals = prepared[arm]["panels"][panel]["dense_items"]
            indices = mappings[arm][panel]
            if (
                len(indices) != len(originals)
                or any(index not in range(len(union)) for index in indices)
                or any(
                    union[index]["prompt_token_ids"]
                    != original["prompt_token_ids"]
                    for original, index in zip(originals, indices)
                )
            ):
                raise RuntimeError(f"v20a lossless reconstruction failed: {arm}/{panel}")
    return audit
