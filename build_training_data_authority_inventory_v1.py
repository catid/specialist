#!/usr/bin/env python3
"""Build the metadata-only training-data authority inventory.

This builder deliberately reads only the explicitly allowlisted aggregate
reports, registries, manifests, and one safe train projection.  The projection
audit accesses only nonsemantic identity fields (``kind``, ``fact_id``, and
``document_sha256``); question, answer, text, evidence, and URL fields are
never selected or emitted.

The output is an authority ledger, not a concatenated dataset and not launch
authorization.  In particular it preserves unresolved public-license status
while recording the user's separate project-training authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "data/training_inventory/training_data_authority_v1.json"
).resolve()
SOURCE_SPLIT_AUTHORITY = (
    ROOT / "data/training_inventory/source_group_split_authority_v1.json"
).resolve()

REGISTRY = (
    ROOT / "data/site_corpora/registry/site_corpus_registry_v1.json"
).resolve()
MARKDOWN_SNAPSHOT = (
    ROOT / "data/site_corpora/training/site_markdown_cpt_v1/manifest.json"
).resolve()
PROJECT_AUTHORIZATION = (
    ROOT
    / "data/site_corpora/registry/project_training_authorization_v1.json"
).resolve()

MULTIPAGE_MANIFESTS = {
    "crash_restraint": (
        ROOT / "data/site_corpora/crash_restraint/manifest.json"
    ).resolve(),
    "rope365": (ROOT / "data/site_corpora/rope365/manifest.json").resolve(),
    "rope_topia": (
        ROOT / "data/site_corpora/rope_topia/manifest.json"
    ).resolve(),
    "shibari_atlas": (
        ROOT / "data/site_corpora/shibari_atlas/manifest.json"
    ).resolve(),
}

LEGACY_QA_REPORT = (ROOT / "data/train_qa_curated_v1.report.json").resolve()
V440_MANIFEST = (
    ROOT
    / "experiments/sft_controls/v53a_train_refresh_v440_fold3/manifest_v53a.json"
).resolve()
V440_RUNTIME_REPORT = (
    ROOT
    / "experiments/sft_controls/v53a_v440_equal_lr5p5e5/runtime_report_v53a.json"
).resolve()
V440_PROJECTION = (
    ROOT
    / "experiments/sft_controls/v53a_train_refresh_v440_fold3/train_projection_v440.jsonl"
)
V440_ENCODING_SOURCE = (ROOT / "run_sft_train_only_control_v36a.py").resolve()

PENDING_QA_REPORTS = (
    (
        ROOT
        / "data/public_training_shards/ropeconnections_partial_suspension_v1/report.json"
    ).resolve(),
    (
        ROOT
        / "data/public_training_shards/ropeconnections_rope_dyeing_v1/report.json"
    ).resolve(),
    (
        ROOT
        / "data/public_training_shards/ropeconnections_rope_ends_update_v1/report.json"
    ).resolve(),
    (
        ROOT
        / "data/public_training_shards/ropeconnections_rope_kit_v1/report.json"
    ).resolve(),
    (
        ROOT
        / "data/public_training_shards/ropeconnections_secondary_column_v1/report.json"
    ).resolve(),
)

GENERAL_PROSE_REPORT = (ROOT / "data/general_prose_anchor_v1.report.json").resolve()
GENERAL_QA_PROXY_REPORT = (
    ROOT / "data/general_qa_proxy_anchor_v43h.report.json"
).resolve()

SCHEMA = "specialist-training-data-authority-inventory-v1"
AUTHORIZATION_SCHEMA = "site-corpus-project-training-authorization-v1"
DOMAIN_TARGET_TOKENS = 1_000_000
EXPECTED_V440_ENCODING_LINES = (34, 40)
EXPECTED_OVERRIDE_RESOURCES = frozenset(
    {"crash_restraint", "rope365", "rope_topia", "shibari_atlas"}
)
EXPECTED_AUTHORIZATION_GATES = [
    "preserve attribution, provenance, and safety-transfer flags",
    "reconstruct page or section source groups before the 80/10/10 split",
    "seal source-disjoint train, development, and final assignments before derivation",
    "extend the opaque protected-source disjointness contract before training",
]
EXPECTED_AUTHORIZATION_KEYS = {
    "authorization_date",
    "authorization_rationale",
    "authorization_scope",
    "authorized_by",
    "authorized_qwen36_tokens",
    "changes_registry_rights_status",
    "content_sha256_before_self_field",
    "not_a_public_license_determination",
    "required_global_gates",
    "resources",
    "schema",
    "supersedes_public_rights_gate_for_project_use",
}
EXPECTED_AUTHORIZATION_RESOURCE_KEYS = {
    "artifact_id",
    "project_training_authorized",
    "registered_qwen36_tokens",
    "registry_promotion_gate",
    "registry_rights_status",
    "resource_id",
}
EXPECTED_SNAPSHOT_INCLUDED_KEYS = {
    "artifact_id",
    "characters",
    "chunks",
    "complete_character_coverage",
    "emitted_chunk_tokens",
    "exact_ordered_reconstruction",
    "markdown_path",
    "markdown_sha256",
    "promotion_gate",
    "resource_id",
    "rights_status",
    "source_document_group_id",
    "source_document_identity_sha256",
    "source_tokens",
}
EXPECTED_SNAPSHOT_BLOCKED_KEYS = {
    "available_tokens",
    "manifest_path",
    "markdown_path",
    "promotion_gate",
    "reason",
    "resource_id",
    "rights_status",
}
EXPECTED_POLICY_EXCLUSION_KEYS = {"manifest_path", "reason", "resource_id"}
EXPECTED_V440_MANIFEST_KEYS = {
    "access_firewall",
    "content_sha256_before_self_field",
    "equal_conflict_unit_weighting",
    "fold_3_train",
    "lineage_stability",
    "projection",
    "schema",
    "selection_firewall",
    "status",
    "step_schedule",
}
EXPECTED_V440_PROJECTION_KEYS = {
    "path",
    "repeat_replay_byte_identical",
    "replay_v431_v440",
    "rows",
    "sha256",
}
EXPECTED_V440_LINEAGE_STABILITY = {
    "accepted_edit_decisions_v413_v440": 81,
    "added_rows": 0,
    "dropped_rows": 0,
    "fold_assignment_changes": 0,
    "root_membership_exactly_preserved": True,
}
EXPECTED_V440_ACCESS_FIREWALL = {
    "accepted_train_projection_and_curations_opened": True,
    "eval_ood_holdout_or_benchmark_opened": False,
    "external_metrics_used": False,
    "frozen_fold3_train_membership_opened": True,
    "gpu_accessed": False,
    "shadow_artifact_opened": False,
    "shadow_artifact_written": False,
    "training_launched": False,
}
EXPECTED_V440_SELECTION_FIREWALL = {
    "post_training_evaluation_authorized": False,
    "shadow_ood_holdout_feedback_authorized": False,
    "training_input": "exact v440 rows in frozen v412 fold-3 train roots",
}

_SAFE_JSON_INPUTS = frozenset(
    {
        REGISTRY,
        MARKDOWN_SNAPSHOT,
        PROJECT_AUTHORIZATION,
        LEGACY_QA_REPORT,
        V440_MANIFEST,
        V440_RUNTIME_REPORT,
        GENERAL_PROSE_REPORT,
        GENERAL_QA_PROXY_REPORT,
        *PENDING_QA_REPORTS,
        *MULTIPAGE_MANIFESTS.values(),
    }
)
_SAFE_NON_JSON_INPUTS = frozenset({V440_PROJECTION, V440_ENCODING_SOURCE})
_FORBIDDEN_PATH_TOKENS = (
    "protected",
    "holdout",
    "heldout",
    "ood",
    "terminal",
    "incident",
    "manual_review",
    "manual-review",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _assert_allowlisted(path: Path, *, json_input: bool) -> Path:
    # Lexical normalization is deliberate: resolving a now-mixed pre-seal
    # projection would itself cross the post-seal no-stat boundary.
    resolved = Path(os.path.abspath(os.fspath(path)))
    allowed = _SAFE_JSON_INPUTS if json_input else _SAFE_NON_JSON_INPUTS
    if resolved not in allowed:
        raise RuntimeError(f"inventory input is not explicitly allowlisted: {resolved}")
    relative = resolved.relative_to(ROOT).as_posix().casefold()
    if any(token in relative for token in _FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"inventory input crosses a forbidden path boundary: {relative}")
    return resolved


def read_safe_object(path: Path) -> dict:
    approved = _assert_allowlisted(path, json_input=True)
    value = json.loads(approved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected an aggregate JSON object: {approved}")
    return value


def _input_receipt(path: Path, *, json_input: bool) -> dict:
    approved = _assert_allowlisted(path, json_input=json_input)
    return {
        "path": approved.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(approved),
    }


def _report_receipt(path: Path) -> dict:
    return _input_receipt(path, json_input=True)


def _non_json_receipt(path: Path) -> dict:
    return _input_receipt(path, json_input=False)


def _validate_self_addressed(value: dict, field: str) -> None:
    declared = value.get(field)
    if not isinstance(declared, str) or not re.fullmatch(r"[0-9a-f]{64}", declared):
        raise RuntimeError(f"missing or invalid {field}")
    unsigned = dict(value)
    del unsigned[field]
    if canonical_sha256(unsigned) != declared:
        raise RuntimeError(f"content address mismatch for {field}")


def _validate_v440_authority_manifest(
    value: dict,
    *,
    actual_projection_sha256: str,
) -> None:
    if set(value) != EXPECTED_V440_MANIFEST_KEYS:
        raise RuntimeError("V440 authority manifest schema changed")
    _validate_self_addressed(value, "content_sha256_before_self_field")
    if value.get("status") != "sealed_v440_projection_fold3_train_unlaunched":
        raise RuntimeError("V440 authority manifest is not sealed and unlaunched")
    if value.get("lineage_stability") != EXPECTED_V440_LINEAGE_STABILITY:
        raise RuntimeError("V440 authority lineage stability changed")
    if value.get("access_firewall") != EXPECTED_V440_ACCESS_FIREWALL:
        raise RuntimeError("V440 authority access firewall changed")
    if value.get("selection_firewall") != EXPECTED_V440_SELECTION_FIREWALL:
        raise RuntimeError("V440 authority selection firewall changed")

    projection = value.get("projection")
    if not isinstance(projection, dict) or set(projection) != (
        EXPECTED_V440_PROJECTION_KEYS
    ):
        raise RuntimeError("V440 projection schema changed")
    projection_path = projection["path"]
    if (
        not isinstance(projection_path, str)
        or Path(os.path.abspath(projection_path)) != V440_PROJECTION
    ):
        raise RuntimeError("V440 manifest projection path changed")
    declared_sha256 = projection.get("sha256")
    if (
        not isinstance(declared_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", declared_sha256)
        or declared_sha256 != actual_projection_sha256
    ):
        raise RuntimeError("V440 projection content address changed")
    if projection.get("repeat_replay_byte_identical") is not True:
        raise RuntimeError("V440 projection replay is not byte-identical")
    replay = projection.get("replay_v431_v440")
    if (
        not isinstance(replay, list)
        or any(
            not isinstance(item, dict)
            or set(item) != {
                "decision_file_sha256",
                "projection_sha256",
                "version",
            }
            for item in replay
        )
    ):
        raise RuntimeError("V440 projection replay lineage changed")
    if (
        [item["version"] for item in replay] != list(range(431, 441))
        or any(
            not isinstance(item["decision_file_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["decision_file_sha256"])
            or not isinstance(item["projection_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", item["projection_sha256"])
            for item in replay
        )
        or replay[-1]["projection_sha256"] != declared_sha256
    ):
        raise RuntimeError("V440 projection replay lineage changed")


def _snapshot_stale_relative_to_authorization(
    authorized_resources: set[str],
    included_resources: set[str],
) -> bool:
    return not authorized_resources.issubset(included_resources)


def _validate_markdown_snapshot(
    snapshot: dict,
    registry_by_resource: dict[str, dict],
    registry_policy_exclusions: list[dict],
    authorized_resources: set[str],
) -> tuple[dict[str, dict], dict[str, dict], bool]:
    _validate_self_addressed(snapshot, "content_sha256_before_self_field")
    included_rows = snapshot.get("included_documents")
    blocked_rows = snapshot.get("rights_blocked_documents")
    policy_rows = snapshot.get("policy_excluded_documents")
    if not all(isinstance(rows, list) for rows in (
        included_rows,
        blocked_rows,
        policy_rows,
    )):
        raise RuntimeError("Markdown snapshot document ledgers changed schema")
    if any(
        not isinstance(item, dict)
        or set(item) != EXPECTED_SNAPSHOT_INCLUDED_KEYS
        or not isinstance(item.get("resource_id"), str)
        or not item["resource_id"]
        for item in included_rows
    ):
        raise RuntimeError("Markdown snapshot included-document schema changed")
    if any(
        not isinstance(item, dict)
        or set(item) != EXPECTED_SNAPSHOT_BLOCKED_KEYS
        or not isinstance(item.get("resource_id"), str)
        or not item["resource_id"]
        for item in blocked_rows
    ):
        raise RuntimeError("Markdown snapshot rights-blocked schema changed")
    if any(
        not isinstance(item, dict)
        or set(item) != EXPECTED_POLICY_EXCLUSION_KEYS
        or not isinstance(item.get("resource_id"), str)
        or not item["resource_id"]
        for item in policy_rows
    ):
        raise RuntimeError("Markdown snapshot policy-exclusion schema changed")
    if (
        not isinstance(registry_policy_exclusions, list)
        or any(
            not isinstance(item, dict)
            or set(item) != EXPECTED_POLICY_EXCLUSION_KEYS
            or not isinstance(item.get("resource_id"), str)
            or not item["resource_id"]
            for item in registry_policy_exclusions
        )
    ):
        raise RuntimeError("Markdown registry policy-exclusion schema changed")

    included = {item["resource_id"]: item for item in included_rows}
    blocked = {item["resource_id"]: item for item in blocked_rows}
    policy = {item["resource_id"]: item for item in policy_rows}
    registry_policy = {
        item["resource_id"]: item for item in registry_policy_exclusions
    }
    if (
        len(included) != len(included_rows)
        or len(blocked) != len(blocked_rows)
        or len(policy) != len(policy_rows)
        or len(registry_policy) != len(registry_policy_exclusions)
    ):
        raise RuntimeError("Markdown snapshot or registry contains duplicate resources")
    included_ids = set(included)
    blocked_ids = set(blocked)
    registry_ids = set(registry_by_resource)
    if included_ids & blocked_ids:
        raise RuntimeError("Markdown snapshot includes and blocks the same resource")
    if included_ids | blocked_ids != registry_ids:
        raise RuntimeError("Markdown snapshot does not account for every registry artifact")
    if blocked_ids - authorized_resources:
        raise RuntimeError("Markdown snapshot blocks a non-override registry artifact")
    if set(policy) != set(registry_policy) or any(
        policy[resource] != registry_policy[resource] for resource in policy
    ):
        raise RuntimeError("Markdown snapshot policy exclusions changed")
    if (registry_ids | authorized_resources) & set(registry_policy):
        raise RuntimeError("policy-excluded Markdown was included or authorized")

    for resource, item in included.items():
        registered = registry_by_resource[resource]
        rights = registered["rights_basis"]
        expected_pairs = {
            "artifact_id": registered["artifact_id"],
            "markdown_path": registered["markdown_path"],
            "markdown_sha256": registered["markdown_sha256"],
            "promotion_gate": rights["promotion_gate"],
            "rights_status": rights["status"],
            "source_document_group_id": registered[
                "required_single_document_split_group"
            ]["group_id"],
            "source_document_identity_sha256": registered[
                "source_document_identity_sha256"
            ],
            "source_tokens": registered["qwen36_token_count"],
            "emitted_chunk_tokens": registered["qwen36_token_count"],
            "complete_character_coverage": True,
            "exact_ordered_reconstruction": True,
        }
        if any(item.get(key) != value for key, value in expected_pairs.items()):
            raise RuntimeError(f"Markdown snapshot identity drift for {resource}")
        if (
            type(item.get("chunks")) is not int
            or item["chunks"] <= 0
            or type(item.get("characters")) is not int
            or item["characters"] <= 0
        ):
            raise RuntimeError(f"Markdown snapshot row accounting changed for {resource}")

    for resource, item in blocked.items():
        registered = registry_by_resource[resource]
        rights = registered["rights_basis"]
        expected_pairs = {
            "available_tokens": registered["qwen36_token_count"],
            "manifest_path": registered["manifest_path"],
            "markdown_path": registered["markdown_path"],
            "promotion_gate": rights["promotion_gate"],
            "reason": "rights_promotion_gate_not_ready",
            "rights_status": rights["status"],
        }
        if any(item.get(key) != value for key, value in expected_pairs.items()):
            raise RuntimeError(f"Markdown snapshot rights-block drift for {resource}")

    included_tokens = sum(item["source_tokens"] for item in included.values())
    emitted_tokens = sum(
        item["emitted_chunk_tokens"] for item in included.values()
    )
    train_rows = sum(item["chunks"] for item in included.values())
    expected_accounting = {
        "all_eligible_artifacts_have_training_rows": True,
        "all_included_documents_exactly_reconstruct_from_chunks": True,
        "all_registry_artifacts_accounted_for": True,
        "omission_is_a_build_error": True,
        "registry_artifact_count": len(registry_ids),
        "registry_artifacts_accounted_for": len(registry_ids),
    }
    expected_snapshot_aggregates = {
        "source_documents_included": len(included),
        "source_documents_rights_blocked": len(blocked),
        "source_tokens_included": included_tokens,
        "emitted_chunk_tokens": emitted_tokens,
        "train_rows": train_rows,
        "policy_excluded_manifests": len(policy),
        "accounting": expected_accounting,
        "full_eligible_document_content_retained": True,
        "cross_document_packing": False,
        "launch_authorized_by_snapshot": False,
    }
    if any(
        snapshot.get(key) != value
        for key, value in expected_snapshot_aggregates.items()
    ):
        raise RuntimeError("Markdown snapshot aggregate accounting changed")
    stale = _snapshot_stale_relative_to_authorization(
        authorized_resources,
        included_ids,
    )
    return included, blocked, stale


def _v440_encoding_totals() -> dict:
    """Read only the explicitly approved aggregate constant at lines 34--40."""
    source = _assert_allowlisted(V440_ENCODING_SOURCE, json_input=False)
    start, stop = EXPECTED_V440_ENCODING_LINES
    selected: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line_number > stop:
                break
            if line_number >= start:
                selected.append(line)
    snippet = "".join(selected)
    fields = {}
    for key in ("train_prompt_tokens", "train_answer_tokens", "train_rows"):
        match = re.search(rf'"{key}"\s*:\s*([0-9_]+)', snippet)
        if match is None:
            raise RuntimeError(f"approved V440 encoding aggregate lost field {key}")
        fields[key] = int(match.group(1).replace("_", ""))
    fields["qwen36_sft_tokens"] = (
        fields["train_prompt_tokens"] + fields["train_answer_tokens"]
    )
    fields["source"] = {
        "path": source.relative_to(ROOT).as_posix(),
        "line_start": start,
        "line_stop": stop,
        "file_sha256": file_sha256(source),
    }
    return fields


def aggregate_nonsemantic_qa_identity_rows(lines: Any, expected_rows: int) -> dict:
    """Aggregate only safe identity fields from JSONL lines."""
    kinds: Counter[str] = Counter()
    document_ids: set[str] = set()
    selected_document_ids: set[str] = set()
    fact_ids: set[str] = set()
    selected_rows = 0
    rows = 0
    for line in lines:
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise RuntimeError("QA projection contains a non-object row")
        # Deliberately access only these three nonsemantic fields.
        kind = row.get("kind")
        fact_id = row.get("fact_id")
        document_id = row.get("document_sha256")
        if not all(isinstance(item, str) and item for item in (
            kind,
            fact_id,
            document_id,
        )):
            raise RuntimeError("QA identity metadata is incomplete")
        kinds[kind] += 1
        fact_ids.add(fact_id)
        document_ids.add(document_id)
        if kind != "qa_resource_index":
            selected_document_ids.add(document_id)
            selected_rows += 1
        rows += 1
    if rows != expected_rows or len(fact_ids) != rows:
        raise RuntimeError("V440 aggregate row or fact identity count changed")
    return {
        "rows": rows,
        "unique_fact_ids": len(fact_ids),
        "unique_source_document_groups": len(document_ids),
        "selected_non_url_index_rows": selected_rows,
        "selected_non_url_index_source_document_groups": len(
            selected_document_ids
        ),
        "kind_counts": dict(sorted(kinds.items())),
        "selected_kind_counts": {
            key: value
            for key, value in sorted(kinds.items())
            if key != "qa_resource_index"
        },
        "qa_resource_index_rows": kinds.get("qa_resource_index", 0),
        "semantic_fields_accessed_or_emitted": False,
    }


def _audit_v440_nonsemantic_fields(expected_rows: int) -> dict:
    """Audit the allowlisted V440 projection without selecting semantics."""
    projection = _assert_allowlisted(V440_PROJECTION, json_input=False)
    with projection.open("r", encoding="utf-8") as handle:
        return aggregate_nonsemantic_qa_identity_rows(handle, expected_rows)


def _assert_preseal_inventory_rebuild_allowed() -> None:
    """Forbid reopening the mixed projection once split sealing has occurred.

    ``V440_PROJECTION`` was a permitted one-time pre-seal input.  After the
    source-disjoint authority exists it can contain final-partition rows and
    must never be opened again, even for a nominally nonsemantic inventory
    audit.  Consumers must use the sealed train/development projections and
    authority receipts instead.
    """

    if SOURCE_SPLIT_AUTHORITY.exists():
        raise RuntimeError(
            "legacy training-data inventory rebuild is forbidden after the "
            "source-disjoint authority was sealed; consume "
            "source_group_split_authority_v1.json and its materialized TRAIN "
            "projection receipts"
        )


def _multipage_coverage() -> dict[str, dict]:
    values = {
        resource: read_safe_object(path)
        for resource, path in MULTIPAGE_MANIFESTS.items()
    }
    atlas = values["shibari_atlas"]["canonical_inventory"]
    crash_entries = values["crash_restraint"]["entries"]
    crash_counts = Counter(item.get("disposition") for item in crash_entries)
    rope365 = values["rope365"]["coverage"]
    ropetopia = values["rope_topia"]["inventory"]
    ropetopia_counts = Counter(item.get("corpus_status") for item in ropetopia)
    result = {
        "crash_restraint": {
            "inventory_units": len(crash_entries),
            "included_units": crash_counts.get("included", 0),
            "excluded_or_other_units": (
                len(crash_entries) - crash_counts.get("included", 0)
            ),
        },
        "rope365": {
            "inventory_units": rope365["discovered_canonical_urls"],
            "included_units": rope365["included_pages"],
            "excluded_or_other_units": rope365["excluded_pages"],
        },
        "rope_topia": {
            "inventory_units": len(ropetopia),
            "included_units": ropetopia_counts.get("included", 0),
            "excluded_or_other_units": (
                len(ropetopia) - ropetopia_counts.get("included", 0)
            ),
        },
        "shibari_atlas": {
            "inventory_units": atlas["total_urls"],
            "included_units": atlas["decision_counts"]["include"],
            "excluded_or_other_units": atlas["decision_counts"]["exclude"],
        },
    }
    expected = {
        "crash_restraint": (181, 116, 65),
        "rope365": (232, 111, 121),
        "rope_topia": (15, 8, 7),
        "shibari_atlas": (2078, 1714, 364),
    }
    for resource, (
        inventory_units,
        included_units,
        excluded_or_other_units,
    ) in expected.items():
        observed = result[resource]
        if (
            observed["inventory_units"] != inventory_units
            or observed["included_units"] != included_units
            or observed["excluded_or_other_units"] != excluded_or_other_units
            or included_units + excluded_or_other_units != inventory_units
        ):
            raise RuntimeError(f"multi-page provenance aggregate changed: {resource}")
        observed["manifest_receipt"] = _report_receipt(
            MULTIPAGE_MANIFESTS[resource]
        )
    return result


def _load_authorization(registry_by_resource: dict[str, dict]) -> dict:
    authorization = read_safe_object(PROJECT_AUTHORIZATION)
    if set(authorization) != EXPECTED_AUTHORIZATION_KEYS:
        raise RuntimeError("project authorization top-level schema changed")
    if authorization.get("schema") != AUTHORIZATION_SCHEMA:
        raise RuntimeError("unsupported project authorization schema")
    _validate_self_addressed(authorization, "content_sha256_before_self_field")
    if (
        authorization.get("authorization_date") != "2026-07-17"
        or authorization.get("authorization_scope")
        != "specialist project training only"
        or authorization.get("authorized_by") != "user"
    ):
        raise RuntimeError("project authorization actor, date, or scope changed")
    if authorization.get("not_a_public_license_determination") is not True:
        raise RuntimeError("project authorization must preserve the license caveat")
    if authorization.get("changes_registry_rights_status") is not False:
        raise RuntimeError("project authorization may not rewrite rights status")
    if authorization.get("supersedes_public_rights_gate_for_project_use") is not True:
        raise RuntimeError("project authorization override state changed")
    if authorization.get("required_global_gates") != EXPECTED_AUTHORIZATION_GATES:
        raise RuntimeError("project authorization safety gates changed")
    if authorization.get("authorization_rationale") != {
        "record_type": "user_assertion_not_legal_determination",
        "user_asserted_fair_use": True,
        "user_asserted_noncommercial_research_experiments": True,
    }:
        raise RuntimeError("project authorization rationale changed")
    resources = authorization.get("resources")
    if not isinstance(resources, list):
        raise RuntimeError("project authorization has no resource ledger")
    if (
        len(resources) != len(EXPECTED_OVERRIDE_RESOURCES)
        or any(
            not isinstance(item, dict)
            or set(item) != EXPECTED_AUTHORIZATION_RESOURCE_KEYS
            for item in resources
        )
    ):
        raise RuntimeError("project authorization resource schema changed")
    by_resource = {item.get("resource_id"): item for item in resources}
    if (
        len(by_resource) != len(resources)
        or set(by_resource) != EXPECTED_OVERRIDE_RESOURCES
    ):
        raise RuntimeError("project authorization resource set changed")
    total = 0
    for resource, item in by_resource.items():
        registered = registry_by_resource[resource]
        rights = registered["rights_basis"]
        expected_pairs = {
            "artifact_id": registered["artifact_id"],
            "registered_qwen36_tokens": registered["qwen36_token_count"],
            "registry_rights_status": rights["status"],
            "registry_promotion_gate": rights["promotion_gate"],
            "project_training_authorized": True,
        }
        if any(item.get(key) != value for key, value in expected_pairs.items()):
            raise RuntimeError(f"authorization drift for {resource}")
        total += item["registered_qwen36_tokens"]
    if total != 1_107_618 or authorization.get("authorized_qwen36_tokens") != total:
        raise RuntimeError("project authorization token total changed")
    return authorization


def _markdown_inventory() -> tuple[list[dict], list[dict], dict]:
    registry = read_safe_object(REGISTRY)
    snapshot = read_safe_object(MARKDOWN_SNAPSHOT)
    if registry.get("schema") != "site-corpus-registry-v1":
        raise RuntimeError("unsupported Markdown registry schema")
    if snapshot.get("schema") != "site-markdown-cpt-training-snapshot-v1":
        raise RuntimeError("unsupported Markdown snapshot schema")
    if snapshot.get("registry_file_sha256") != file_sha256(REGISTRY):
        raise RuntimeError("Markdown snapshot does not match registry bytes")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 33:
        raise RuntimeError("expected all 33 registered Markdown artifacts")
    registry_by_resource = {item["resource_id"]: item for item in artifacts}
    if len(registry_by_resource) != len(artifacts):
        raise RuntimeError("Markdown resource IDs are not unique")
    authorization = _load_authorization(registry_by_resource)
    authorized_resources = {
        item["resource_id"] for item in authorization["resources"]
    }
    registry_policy_exclusions = registry.get("excluded_nontraining_manifests")
    current_included, current_blocked, snapshot_stale = (
        _validate_markdown_snapshot(
            snapshot,
            registry_by_resource,
            registry_policy_exclusions,
            authorized_resources,
        )
    )

    coverage = _multipage_coverage()
    entries = []
    for resource in sorted(registry_by_resource):
        artifact = registry_by_resource[resource]
        rights = artifact["rights_basis"]
        split = artifact["required_single_document_split_group"]
        override = resource in authorized_resources
        included = current_included.get(resource)
        if included is not None:
            materialization = {
                "status": "present_in_current_snapshot",
                "materialized_chunk_rows": included["chunks"],
                "materialized_qwen36_tokens": included["emitted_chunk_tokens"],
                "verification": "exact_ordered_reconstruction_verified",
            }
        else:
            materialization = {
                "status": "pending_authorized_snapshot_rebuild",
                "materialized_chunk_rows": None,
                "materialized_qwen36_tokens": 0,
                "verification": "registry_metadata_only_markdown_not_opened_by_inventory",
            }
        entry = {
            "source_type": "raw_markdown_document",
            "resource_id": resource,
            "artifact_id": artifact["artifact_id"],
            "source_identity_sha256": artifact["source_document_identity_sha256"],
            "registered_split_group_id": split["group_id"],
            "registered_source_document_rows": 1,
            "registered_qwen36_tokens": artifact["qwen36_token_count"],
            "rights_basis": rights,
            "rights_status": rights["status"],
            "rights_promotion_gate": rights["promotion_gate"],
            "project_training_authorization_override": override,
            "authorization_does_not_establish_public_license": override,
            "safety_transfer_flags": artifact.get("safety_transfer_flags", []),
            "inventory_decision": "include_in_project_authorized_source_pool",
            "materialization": materialization,
            "required_before_training": [
                "immutable_source_disjoint_split_assignment",
                "attribution_and_safety_flags_preserved",
                "opaque_protected_source_disjoint_contract_extension",
            ],
            "lineage": {
                "registry_path": REGISTRY.relative_to(ROOT).as_posix(),
                "registry_artifact_id": artifact["artifact_id"],
            },
        }
        if override:
            if included is None:
                entry["required_before_training"].insert(
                    0,
                    "rebuild_markdown_snapshot_with_project_authorization_ledger",
                )
            entry["multipage_provenance"] = coverage[resource]
            entry["split_group_status"] = (
                "invalidly_collapsed_digest_group_page_level_groups_required"
            )
            entry["required_before_training"].insert(
                1,
                "reconstruct_page_or_section_source_groups_from_existing_provenance",
            )
        else:
            entry["split_group_status"] = "registered_single_document_group"
        entries.append(entry)

    excluded = []
    for item in registry_policy_exclusions:
        excluded.append({
            "source_type": "policy_excluded_markdown_manifest",
            "resource_id": item["resource_id"],
            "source_identity": item["manifest_path"],
            "split_group_id": None,
            "source_document_rows": None,
            "qwen36_tokens": None,
            "qwen36_token_count_status": "not_counted_policy_excluded",
            "inventory_decision": "exclude",
            "reason": item["reason"],
            "override_applied": False,
        })

    token_total = sum(item["registered_qwen36_tokens"] for item in entries)
    currently_materialized = sum(
        item["materialization"]["materialized_qwen36_tokens"] for item in entries
    )
    if token_total != 1_212_944:
        raise RuntimeError("Markdown token accounting changed")
    rights_status_tokens: Counter[str] = Counter()
    for item in entries:
        rights_status_tokens[item["rights_status"]] += item["registered_qwen36_tokens"]
    expected_rights_status_tokens = {
        "explicit_open_license": 55_288,
        "federal_text_public_domain_presumption": 45_893,
        "legacy_manifest_gap": 1_107_618,
        "public_domain_in_usa_source_with_trademark_and_jurisdiction_limits": 4_145,
    }
    if dict(rights_status_tokens) != expected_rights_status_tokens:
        raise RuntimeError("Markdown rights-status token accounting changed")
    non_legacy_gap_tokens = sum(
        tokens
        for status, tokens in rights_status_tokens.items()
        if status != "legacy_manifest_gap"
    )
    summary = {
        "registered_documents": len(entries),
        "project_authorized_documents": len(entries),
        "registered_qwen36_tokens": token_total,
        "project_authorized_qwen36_tokens": token_total,
        "registry_rights_status_qwen36_tokens": expected_rights_status_tokens,
        "current_snapshot_documents": len(current_included),
        "current_snapshot_qwen36_tokens": currently_materialized,
        "registry_non_legacy_gap_qwen36_tokens": non_legacy_gap_tokens,
        "newly_project_authorized_license_unresolved_documents": len(
            authorized_resources
        ),
        "newly_project_authorized_license_unresolved_qwen36_tokens": (
            authorization["authorized_qwen36_tokens"]
        ),
        "current_snapshot_stale_relative_to_project_authorization": snapshot_stale,
        "post_split_train_tokens": None,
        "post_split_train_token_status": (
            "unknown_until_page_level_source_groups_and_immutable_split_are_sealed"
        ),
        "policy_excluded_manifests": len(excluded),
    }
    return entries, excluded, summary


def _qa_inventory() -> tuple[list[dict], dict]:
    legacy = read_safe_object(LEGACY_QA_REPORT)
    v440 = read_safe_object(V440_MANIFEST)
    runtime = read_safe_object(V440_RUNTIME_REPORT)
    if legacy.get("schema") != "curated-training-qa-report-v1":
        raise RuntimeError("unsupported legacy QA report schema")
    if v440.get("schema") != "specialist-v440-train-only-fold3-refresh-v53a":
        raise RuntimeError("unsupported V440 authority manifest")
    if runtime.get("schema") != "specialist-sft-v440-equal-train-only-runtime-v53a":
        raise RuntimeError("unsupported V440 runtime report")
    _validate_self_addressed(runtime, "content_sha256_before_self_field")
    projection_path = _assert_allowlisted(V440_PROJECTION, json_input=False)
    _validate_v440_authority_manifest(
        v440,
        actual_projection_sha256=file_sha256(projection_path),
    )
    projection = v440["projection"]
    encoding = _v440_encoding_totals()
    if projection["rows"] != encoding["train_rows"] or projection["rows"] != 531:
        raise RuntimeError("V440 aggregate row count changed")
    metadata_audit = _audit_v440_nonsemantic_fields(projection["rows"])
    if (
        metadata_audit["qa_resource_index_rows"] != 15
        or metadata_audit["selected_non_url_index_rows"] != 516
    ):
        raise RuntimeError("V440 URL-index exclusion aggregate changed")
    runtime_audit = runtime["observed_encoding_audit"]["value"]
    if (
        runtime["dataset"]["rows"] != 448
        or runtime_audit["train_rows"] != 448
        or runtime_audit["train_prompt_tokens"]
        + runtime_audit["train_answer_tokens"]
        != 34_212
    ):
        raise RuntimeError("V440 historical fold aggregate changed")
    legacy_counts = legacy["counts"]
    legacy_url_rows = legacy_counts["by_kind"].get("qa_resource_index", 0)
    if legacy_counts["output"] != 784 or legacy_url_rows != 15:
        raise RuntimeError("legacy QA aggregate changed")

    filtered_view_identity = canonical_sha256({
        "parent_projection_sha256": projection["sha256"],
        "selection_field": "kind",
        "excluded_value": "qa_resource_index",
        "selected_rows": metadata_audit["selected_non_url_index_rows"],
    })
    selected = {
        "source_type": "domain_qa_projection",
        "authority_id": "qa-authority:v440-minus-url-index-logical-view",
        "source_identity_sha256": filtered_view_identity,
        "parent_source_identity_sha256": projection["sha256"],
        "parent_source_path": V440_PROJECTION.relative_to(ROOT).as_posix(),
        "source_path": None,
        "source_document_rows": metadata_audit[
            "selected_non_url_index_source_document_groups"
        ],
        "parent_training_example_rows": projection["rows"],
        "training_example_rows": metadata_audit["selected_non_url_index_rows"],
        "qwen36_tokens": None,
        "qwen36_token_count_status": (
            "pending_materialized_filtered_projection_and_safe_tokenizer_report"
        ),
        "parent_unfiltered_qwen36_token_accounting": {
            "total_tokens": encoding["qwen36_sft_tokens"],
            "prompt_tokens": encoding["train_prompt_tokens"],
            "assistant_answer_tokens": encoding["train_answer_tokens"],
            "prompt_mode": "es_exact",
            "eos_appended": False,
        },
        "split_group_authority": "document_sha256",
        "split_assignment_status": "pending_immutable_source_group_split",
        "lineage": {
            "manifest": V440_MANIFEST.relative_to(ROOT).as_posix(),
            "accepted_edit_decisions_v413_v440": v440["lineage_stability"][
                "accepted_edit_decisions_v413_v440"
            ],
            "root_membership_exactly_preserved": v440["lineage_stability"][
                "root_membership_exactly_preserved"
            ],
            "projection_repeat_replay_byte_identical": projection[
                "repeat_replay_byte_identical"
            ],
        },
        "verification_status": (
            "logical_authority_resolved_materialized_projection_pending"
        ),
        "url_index_audit": metadata_audit,
        "inventory_decision": "authoritative_domain_qa_lineage",
        "selection_reason": (
            "newer sealed V440 projection with stable root lineage, deterministic "
            "replay, and an explicit metadata-only exclusion of all 15 "
            "qa_resource_index rows"
        ),
        "required_before_training": [
            "materialize content-addressed 516-row filtered projection",
            "produce safe Qwen token-count report for filtered projection",
            "immutable_source_group_split",
            "opaque_source_disjoint_contract_extension",
        ],
    }
    legacy_entry = {
        "source_type": "domain_qa_projection",
        "authority_id": "qa-lineage:legacy-curated-v1",
        "source_identity_sha256": legacy["output_sha256"],
        "source_path": legacy["output"],
        "source_document_rows": None,
        "source_document_row_count_status": "not_reported_by_safe_aggregate",
        "training_example_rows": legacy_counts["output"],
        "qwen36_tokens": None,
        "qwen36_token_count_status": "not_reported_by_safe_aggregate",
        "split_group_id": None,
        "verification_status": "legacy_report_only",
        "qa_resource_index_rows": legacy_url_rows,
        "inventory_decision": "exclude",
        "reason": (
            "superseded divergent authority; contains 15 URL-index trivia rows; "
            "must not be concatenated with V440"
        ),
    }
    fold_entry = {
        "source_type": "domain_qa_subset_receipt",
        "authority_id": "qa-subset:v440-fold3-historical-train",
        "source_identity_sha256": runtime["dataset"]["sha256"],
        "source_path": str(Path(runtime["dataset"]["path"]).relative_to(ROOT)),
        "source_document_rows": runtime["dataset"]["unique_documents"],
        "training_example_rows": runtime["dataset"]["rows"],
        "qwen36_tokens": (
            runtime_audit["train_prompt_tokens"]
            + runtime_audit["train_answer_tokens"]
        ),
        "inventory_decision": "exclude_as_separate_input",
        "reason": "strict subset of authoritative V440; recorded only to prevent double counting",
    }
    return [selected, legacy_entry, fold_entry], {
        "authoritative_lineage": selected["authority_id"],
        "authoritative_rows": selected["training_example_rows"],
        "authoritative_qwen36_tokens": selected["qwen36_tokens"],
        "parent_unfiltered_qwen36_tokens": encoding["qwen36_sft_tokens"],
        "legacy_rows_excluded": legacy_entry["training_example_rows"],
        "legacy_url_index_rows_excluded": legacy_url_rows,
        "divergent_authorities_concatenated": False,
    }


def _pending_qa_inventory() -> tuple[list[dict], dict]:
    entries = []
    for path in PENDING_QA_REPORTS:
        report = read_safe_object(path)
        if report.get("schema") != "public-training-manual-shard-report-v1":
            raise RuntimeError(f"unsupported pending shard report: {path}")
        boundary = report["boundary"]
        if (
            boundary.get("promotion_status") != "pending_semantic_leakage_audit"
            or boundary.get("protected_content_read") is not False
            or boundary.get("evaluation_content_read") is not False
            or boundary.get("active_training_snapshot_modified") is not False
        ):
            raise RuntimeError(f"pending shard boundary changed: {path}")
        quality = report["quality"]
        safety_flags = sorted(
            key
            for key, value in quality.items()
            if "requires_downstream" in key and value is True
        )
        entry = {
            "source_type": "pending_public_training_qa_shard",
            "resource_id": report["shard"],
            "source_identity_sha256": report["document_sha256"],
            "split_group_id": f'source-document-v1:{report["document_sha256"]}',
            "source_document_rows": 1,
            "training_example_rows": report["review"]["output_qa_rows"],
            "qwen36_tokens": None,
            "qwen36_token_count_status": "not_reported_by_safe_aggregate",
            "qa_artifact_path": report["artifacts"]["qa"]["path"],
            "qa_artifact_sha256": report["artifacts"]["qa"]["sha256"],
            "verification_status": "manual_review_complete_promotion_pending",
            "safety_review_flags": safety_flags,
            "url_memorization_questions": quality["url_memorization_questions"],
            "inventory_decision": "pending_excluded_from_current_training",
            "reason": (
                "semantic leakage audit and safe Qwen token-count report are required"
            ),
            "report_receipt": _report_receipt(path),
        }
        entries.append(entry)
    rows = sum(item["training_example_rows"] for item in entries)
    if rows != 49 or any(item["url_memorization_questions"] != 0 for item in entries):
        raise RuntimeError("pending public shard aggregate changed")
    return entries, {
        "shards": len(entries),
        "rows": rows,
        "qwen36_tokens": None,
        "token_count_status": "not_reported_by_safe_aggregates",
        "counted_in_training_pool": False,
    }


def _replay_anchor_inventory() -> tuple[list[dict], dict]:
    prose = read_safe_object(GENERAL_PROSE_REPORT)
    qa = read_safe_object(GENERAL_QA_PROXY_REPORT)
    if prose.get("schema") != "general-prose-anchor-build-v1":
        raise RuntimeError("unsupported general prose report")
    if qa.get("schema") != "general-qa-proxy-anchor-build-v43h":
        raise RuntimeError("unsupported general QA proxy report")
    _validate_self_addressed(qa, "content_sha256_before_self_field")
    if (
        prose["output_rows"] != 128
        or qa["rows"] != 128
        or qa["parent"]["prose_sha256"] != prose["output_sha256"]
    ):
        raise RuntimeError("general anchor lineage changed")
    qa_entry = {
        "source_type": "general_behavior_replay_candidate",
        "authority_id": "replay-candidate:general-qa-proxy-v43h",
        "source_identity_sha256": qa["output_sha256"],
        "source_document_rows": qa["unique_documents"],
        "training_example_rows": qa["rows"],
        "qwen36_tokens": None,
        "qwen36_token_count_status": "not_reported_by_safe_aggregate",
        "split_group_id": "inherits_parent_document_identity",
        "lineage": qa["parent"],
        "inventory_decision": "selected_candidate_not_yet_replay_approved",
        "reason": (
            "assistant-target proxy is format-compatible, but must be tokenized and "
            "balanced against the full replay taxonomy before use"
        ),
    }
    prose_entry = {
        "source_type": "general_behavior_replay_candidate",
        "authority_id": "replay-candidate:general-prose-anchor-v1",
        "source_identity_sha256": prose["output_sha256"],
        "source_document_rows": prose["unique_documents"],
        "training_example_rows": prose["output_rows"],
        "qwen36_tokens": None,
        "qwen36_token_count_status": "not_reported_by_safe_aggregate",
        "split_group_id": "document_identity",
        "inventory_decision": "exclude_as_separate_input",
        "reason": (
            "the selected QA proxy is derived one-to-one from these same 128 "
            "documents; concatenating both would double-count one authority"
        ),
    }
    return [qa_entry, prose_entry], {
        "selected_candidate": qa_entry["authority_id"],
        "selected_candidate_rows": qa_entry["training_example_rows"],
        "training_approved": False,
        "incompatible_anchor_representations_concatenated": False,
    }


def construct() -> dict:
    _assert_preseal_inventory_rebuild_allowed()
    markdown, policy_exclusions, markdown_summary = _markdown_inventory()
    qa, qa_summary = _qa_inventory()
    pending_qa, pending_summary = _pending_qa_inventory()
    replay, replay_summary = _replay_anchor_inventory()

    authoritative_qa_tokens = qa_summary["authoritative_qwen36_tokens"]
    if authoritative_qa_tokens is not None:
        raise RuntimeError("filtered QA tokens must remain gated in inventory v1")
    materialized_tokens = markdown_summary["current_snapshot_qwen36_tokens"]
    project_authorized_known_tokens = markdown_summary[
        "project_authorized_qwen36_tokens"
    ]
    rights_status_tokens = markdown_summary[
        "registry_rights_status_qwen36_tokens"
    ]
    explicit_open_license_tokens = rights_status_tokens[
        "explicit_open_license"
    ]
    previous_snapshot_non_legacy_gap_tokens = (
        markdown_summary["registry_non_legacy_gap_qwen36_tokens"]
    )
    launch_gates = [
        "reconstruct page or section source groups for all four multi-page digests",
        "seal immutable source-disjoint train/development/final groups before derivation",
        "extend the opaque protected-source disjointness contract to new source bytes",
        "promote pending public QA only after leakage, safety, and token-count reports",
        "construct and approve category-balanced 15 percent general replay",
    ]
    if markdown_summary[
        "current_snapshot_stale_relative_to_project_authorization"
    ]:
        launch_gates.insert(
            0,
            "rebuild Markdown snapshot so all 33 project-authorized documents are materialized",
        )
    result = {
        "schema": SCHEMA,
        "status": "authority_resolved_launch_gated",
        "purpose": (
            "metadata-only inclusion/exclusion authority; never a concatenated dataset"
        ),
        "canonical_training_protocol": "plan.md",
        "qa_authority": qa_summary,
        "markdown_authority": markdown_summary,
        "pending_public_qa": pending_summary,
        "replay_authority": replay_summary,
        "token_budget": {
            "protocol_new_domain_target_qwen36_tokens": DOMAIN_TARGET_TOKENS,
            "explicit_open_license_markdown_qwen36_tokens": (
                explicit_open_license_tokens
            ),
            "explicit_open_license_markdown_shortfall_qwen36_tokens": max(
                0, DOMAIN_TARGET_TOKENS - explicit_open_license_tokens
            ),
            "previous_snapshot_non_legacy_gap_markdown_qwen36_tokens": (
                previous_snapshot_non_legacy_gap_tokens
            ),
            "previous_snapshot_non_legacy_gap_markdown_shortfall_qwen36_tokens": max(
                0, DOMAIN_TARGET_TOKENS - previous_snapshot_non_legacy_gap_tokens
            ),
            "currently_materialized_authoritative_domain_qwen36_tokens": (
                materialized_tokens
            ),
            "currently_materialized_domain_shortfall_qwen36_tokens": max(
                0, DOMAIN_TARGET_TOKENS - materialized_tokens
            ),
            "project_authorized_known_source_pool_qwen36_tokens": (
                project_authorized_known_tokens
            ),
            "project_authorized_source_pool_is_lower_bound": True,
            "additional_filtered_qa_qwen36_tokens": None,
            "additional_filtered_qa_token_status": (
                "unknown_until_516_row_authority_is_materialized_and_tokenized"
            ),
            "project_authorized_source_pool_shortfall_qwen36_tokens": max(
                0, DOMAIN_TARGET_TOKENS - project_authorized_known_tokens
            ),
            "project_authorized_known_source_pool_surplus_qwen36_tokens": max(
                0, project_authorized_known_tokens - DOMAIN_TARGET_TOKENS
            ),
            "license_unresolved_override_qwen36_tokens": (
                markdown_summary[
                    "newly_project_authorized_license_unresolved_qwen36_tokens"
                ]
            ),
            "post_source_split_train_qwen36_tokens": None,
            "post_source_split_shortfall_status": (
                "cannot_be_computed_until_page_level_groups_and_80_10_10_split_are_sealed"
            ),
            "legal_interpretation": (
                "project authorization is recorded separately and is not a public "
                "license or legal determination"
            ),
        },
        "inclusions": {
            "authoritative_qa": [qa[0]],
            "project_authorized_markdown_source_pool": markdown,
        },
        "pending": {
            "public_qa_shards": pending_qa,
            "replay_candidate": [replay[0]],
        },
        "exclusions": {
            "superseded_or_duplicate_qa": qa[1:],
            "duplicate_replay_representation": [replay[1]],
            "policy_blocked_markdown_manifests": policy_exclusions,
        },
        "launch_gates": launch_gates,
        "invariants": {
            "semantic_training_rows_emitted": False,
            "protected_holdout_ood_terminal_incident_or_manual_review_inputs_opened": False,
            "legacy_and_v440_qa_concatenated": False,
            "general_prose_and_derived_qa_proxy_concatenated": False,
            "policy_excluded_markdown_overridden": False,
            "unresolved_license_status_rewritten": False,
            "training_launch_authorized": False,
        },
        "safe_input_receipts": sorted(
            [_report_receipt(path) for path in _SAFE_JSON_INPUTS]
            + [_non_json_receipt(path) for path in _SAFE_NON_JSON_INPUTS],
            key=lambda item: item["path"],
        ),
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def render() -> bytes:
    return (
        json.dumps(construct(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def build(*, check: bool = False) -> dict:
    payload = render()
    value = json.loads(payload)
    if check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != payload:
            raise RuntimeError(f"checked-in training inventory is stale: {OUTPUT}")
        return value
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{OUTPUT.name}.", dir=OUTPUT.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUTPUT)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    value = build(check=arguments.check)
    print(json.dumps({
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "content_sha256": value["content_sha256_before_self_field"],
        "qa_authority": value["qa_authority"]["authoritative_lineage"],
        "project_authorized_known_domain_tokens": value["token_budget"][
            "project_authorized_known_source_pool_qwen36_tokens"
        ],
        "training_launch_authorized": value["invariants"][
            "training_launch_authorized"
        ],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
