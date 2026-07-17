from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import build_training_data_authority_inventory_v1 as inventory


def _reseal(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("content_sha256_before_self_field", None)
    value["content_sha256_before_self_field"] = inventory.canonical_sha256(value)
    return value


def _checked_inventory() -> dict:
    value = json.loads(inventory.OUTPUT.read_text(encoding="utf-8"))
    unsigned = dict(value)
    declared = unsigned.pop("content_sha256_before_self_field")
    assert inventory.canonical_sha256(unsigned) == declared
    return value


def test_checked_inventory_resolves_authorities_without_double_counting():
    value = _checked_inventory()
    assert value["status"] == "authority_resolved_launch_gated"
    assert value["qa_authority"] == {
        "authoritative_lineage": (
            "qa-authority:v440-minus-url-index-logical-view"
        ),
        "authoritative_qwen36_tokens": None,
        "authoritative_rows": 516,
        "divergent_authorities_concatenated": False,
        "legacy_rows_excluded": 784,
        "legacy_url_index_rows_excluded": 15,
        "parent_unfiltered_qwen36_tokens": 39818,
    }
    qa = value["inclusions"]["authoritative_qa"][0]
    assert qa["url_index_audit"]["rows"] == 531
    assert qa["url_index_audit"]["qa_resource_index_rows"] == 15
    assert qa["url_index_audit"]["selected_non_url_index_rows"] == 516
    assert "qa_resource_index" not in qa["url_index_audit"][
        "selected_kind_counts"
    ]
    assert qa["qwen36_tokens"] is None
    assert "pending_materialized" in qa["qwen36_token_count_status"]
    assert value["invariants"]["legacy_and_v440_qa_concatenated"] is False


def test_build_check_reconstructs_exact_inventory_and_all_input_receipts():
    assert inventory.build(check=True) == _checked_inventory()
    receipts = {
        item["path"]: item["file_sha256"]
        for item in _checked_inventory()["safe_input_receipts"]
    }
    all_safe_inputs = inventory._SAFE_JSON_INPUTS | inventory._SAFE_NON_JSON_INPUTS
    assert set(receipts) == {
        path.relative_to(inventory.ROOT).as_posix() for path in all_safe_inputs
    }
    for path in inventory._SAFE_NON_JSON_INPUTS:
        assert receipts[path.relative_to(inventory.ROOT).as_posix()] == (
            inventory.file_sha256(path)
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authorization_scope", "all projects and redistribution"),
        ("authorized_by", "unrelated third party"),
        ("supersedes_public_rights_gate_for_project_use", False),
        ("required_global_gates", []),
    ],
)
def test_project_authorization_rejects_resealed_scope_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
):
    registry = inventory.read_safe_object(inventory.REGISTRY)
    registry_by_resource = {
        item["resource_id"]: item for item in registry["artifacts"]
    }
    authorization = inventory.read_safe_object(inventory.PROJECT_AUTHORIZATION)
    authorization[field] = value
    authorization = _reseal(authorization)
    monkeypatch.setattr(
        inventory,
        "read_safe_object",
        lambda path: copy.deepcopy(authorization),
    )
    with pytest.raises(RuntimeError, match="authorization"):
        inventory._load_authorization(registry_by_resource)


def test_project_authorization_rejects_duplicate_resource_entries(
    monkeypatch: pytest.MonkeyPatch,
):
    registry = inventory.read_safe_object(inventory.REGISTRY)
    registry_by_resource = {
        item["resource_id"]: item for item in registry["artifacts"]
    }
    authorization = inventory.read_safe_object(inventory.PROJECT_AUTHORIZATION)
    authorization["resources"].append(copy.deepcopy(authorization["resources"][0]))
    authorization = _reseal(authorization)
    monkeypatch.setattr(
        inventory,
        "read_safe_object",
        lambda path: copy.deepcopy(authorization),
    )
    with pytest.raises(RuntimeError, match="resource schema"):
        inventory._load_authorization(registry_by_resource)


@pytest.mark.parametrize(
    "case",
    ["root_membership", "repeat_replay", "added_rows", "status", "replay_schema"],
)
def test_v440_authority_rejects_resealed_unstable_lineage(case: str):
    manifest = inventory.read_safe_object(inventory.V440_MANIFEST)
    if case == "root_membership":
        manifest["lineage_stability"]["root_membership_exactly_preserved"] = False
    elif case == "repeat_replay":
        manifest["projection"]["repeat_replay_byte_identical"] = False
    elif case == "added_rows":
        manifest["lineage_stability"]["added_rows"] = 1
    elif case == "status":
        manifest["status"] = "unsealed"
    else:
        manifest["projection"]["replay_v431_v440"][0] = "malformed"
    manifest = _reseal(manifest)
    with pytest.raises(RuntimeError, match="V440"):
        inventory._validate_v440_authority_manifest(
            manifest,
            actual_projection_sha256=manifest["projection"]["sha256"],
        )


def test_v440_authority_rejects_projection_content_address_mismatch():
    manifest = inventory.read_safe_object(inventory.V440_MANIFEST)
    with pytest.raises(RuntimeError, match="content address"):
        inventory._validate_v440_authority_manifest(
            manifest,
            actual_projection_sha256="0" * 64,
        )


def _snapshot_validation_inputs() -> tuple[dict, dict, list, set[str]]:
    registry = inventory.read_safe_object(inventory.REGISTRY)
    snapshot = inventory.read_safe_object(inventory.MARKDOWN_SNAPSHOT)
    authorization = inventory.read_safe_object(inventory.PROJECT_AUTHORIZATION)
    registry_by_resource = {
        item["resource_id"]: item for item in registry["artifacts"]
    }
    authorized_resources = {
        item["resource_id"] for item in authorization["resources"]
    }
    return (
        snapshot,
        registry_by_resource,
        registry["excluded_nontraining_manifests"],
        authorized_resources,
    )


def test_snapshot_validation_derives_staleness_from_authorized_membership():
    snapshot, registry, exclusions, authorized = _snapshot_validation_inputs()
    included, blocked, stale = inventory._validate_markdown_snapshot(
        snapshot,
        registry,
        exclusions,
        authorized,
    )
    assert set(blocked) == authorized
    assert stale is True
    assert inventory._snapshot_stale_relative_to_authorization(
        authorized,
        set(included) | authorized,
    ) is False


@pytest.mark.parametrize("case", ["reconstruction", "identity", "duplicate", "overlap"])
def test_snapshot_validation_rejects_resealed_identity_and_membership_drift(
    case: str,
):
    snapshot, registry, exclusions, authorized = _snapshot_validation_inputs()
    snapshot = copy.deepcopy(snapshot)
    if case == "reconstruction":
        snapshot["included_documents"][0]["exact_ordered_reconstruction"] = False
    elif case == "identity":
        snapshot["included_documents"][0]["source_document_identity_sha256"] = (
            "0" * 64
        )
    elif case == "duplicate":
        snapshot["included_documents"].append(
            copy.deepcopy(snapshot["included_documents"][0])
        )
    else:
        overlapping = copy.deepcopy(snapshot["included_documents"][0])
        overlapping["resource_id"] = snapshot["rights_blocked_documents"][0][
            "resource_id"
        ]
        snapshot["included_documents"].append(overlapping)
    snapshot = _reseal(snapshot)
    with pytest.raises(RuntimeError, match="Markdown snapshot"):
        inventory._validate_markdown_snapshot(
            snapshot,
            registry,
            exclusions,
            authorized,
        )


def test_all_registered_markdown_is_authorized_but_audit_trail_is_preserved():
    value = _checked_inventory()
    authorization = inventory.read_safe_object(inventory.PROJECT_AUTHORIZATION)
    assert authorization["authorization_rationale"] == {
        "record_type": "user_assertion_not_legal_determination",
        "user_asserted_fair_use": True,
        "user_asserted_noncommercial_research_experiments": True,
    }
    assert authorization["authorization_scope"] == "specialist project training only"
    assert authorization["authorized_by"] == "user"
    assert authorization["required_global_gates"] == inventory.EXPECTED_AUTHORIZATION_GATES
    markdown = value["inclusions"]["project_authorized_markdown_source_pool"]
    assert len(markdown) == 33
    assert sum(item["registered_source_document_rows"] for item in markdown) == 33
    assert sum(item["registered_qwen36_tokens"] for item in markdown) == 1_212_944
    override = {
        item["resource_id"]: item
        for item in markdown
        if item["project_training_authorization_override"]
    }
    assert set(override) == inventory.EXPECTED_OVERRIDE_RESOURCES
    assert sum(item["registered_qwen36_tokens"] for item in override.values()) == (
        1_107_618
    )
    assert all(item["rights_status"] == "legacy_manifest_gap" for item in override.values())
    assert all(item["rights_basis"]["license"] == "not_recorded" for item in override.values())
    assert all(item["rights_basis"]["attribution_required"] is True for item in override.values())
    assert all(
        item["authorization_does_not_establish_public_license"] is True
        for item in override.values()
    )
    assert all(
        item["split_group_status"]
        == "invalidly_collapsed_digest_group_page_level_groups_required"
        for item in override.values()
    )
    assert override["shibari_atlas"]["multipage_provenance"][
        "inventory_units"
    ] == 2078
    assert override["shibari_atlas"]["multipage_provenance"][
        "included_units"
    ] == 1714
    expected_multipage_counts = {
        "crash_restraint": (181, 116, 65),
        "rope365": (232, 111, 121),
        "rope_topia": (15, 8, 7),
        "shibari_atlas": (2078, 1714, 364),
    }
    for resource, expected in expected_multipage_counts.items():
        provenance = override[resource]["multipage_provenance"]
        assert (
            provenance["inventory_units"],
            provenance["included_units"],
            provenance["excluded_or_other_units"],
        ) == expected
    assert value["markdown_authority"]["current_snapshot_documents"] == 29
    assert value["markdown_authority"]["current_snapshot_qwen36_tokens"] == 105326
    assert value["markdown_authority"][
        "current_snapshot_stale_relative_to_project_authorization"
    ] is True
    assert len(value["exclusions"]["policy_blocked_markdown_manifests"]) == 3
    assert value["invariants"]["policy_excluded_markdown_overridden"] is False
    assert value["invariants"]["unresolved_license_status_rewritten"] is False


def test_pending_qa_and_replay_are_accounted_for_but_not_silently_promoted():
    value = _checked_inventory()
    pending = value["pending"]["public_qa_shards"]
    assert len(pending) == 5
    assert sum(item["training_example_rows"] for item in pending) == 49
    assert all(item["qwen36_tokens"] is None for item in pending)
    assert all(
        item["qwen36_token_count_status"] == "not_reported_by_safe_aggregate"
        for item in pending
    )
    assert all(item["url_memorization_questions"] == 0 for item in pending)
    assert value["pending_public_qa"]["counted_in_training_pool"] is False
    assert value["replay_authority"]["selected_candidate_rows"] == 128
    assert value["replay_authority"]["training_approved"] is False
    assert value["replay_authority"][
        "incompatible_anchor_representations_concatenated"
    ] is False
    assert all(
        item["qwen36_tokens"] is None
        for item in value["pending"]["replay_candidate"]
    )
    assert all(
        item["qwen36_tokens"] is None
        for item in value["exclusions"]["policy_blocked_markdown_manifests"]
    )


def test_token_shortfalls_distinguish_materialized_rights_and_user_authority():
    budget = _checked_inventory()["token_budget"]
    assert budget["protocol_new_domain_target_qwen36_tokens"] == 1_000_000
    assert budget["currently_materialized_authoritative_domain_qwen36_tokens"] == (
        105_326
    )
    assert budget["currently_materialized_domain_shortfall_qwen36_tokens"] == 894_674
    assert budget["explicit_open_license_markdown_qwen36_tokens"] == 55_288
    assert budget["explicit_open_license_markdown_shortfall_qwen36_tokens"] == 944_712
    assert budget[
        "previous_snapshot_non_legacy_gap_markdown_qwen36_tokens"
    ] == 105_326
    assert budget[
        "previous_snapshot_non_legacy_gap_markdown_shortfall_qwen36_tokens"
    ] == 894_674
    assert budget["project_authorized_known_source_pool_qwen36_tokens"] == 1_212_944
    assert budget["project_authorized_source_pool_shortfall_qwen36_tokens"] == 0
    assert budget["project_authorized_known_source_pool_surplus_qwen36_tokens"] == (
        212_944
    )
    assert budget["license_unresolved_override_qwen36_tokens"] == 1_107_618
    assert budget["post_source_split_train_qwen36_tokens"] is None


def test_nonsemantic_projection_audit_uses_synthetic_rows_only():
    rows = [
        {
            "kind": "qa_manual",
            "fact_id": "fact-a",
            "document_sha256": "a" * 64,
            "question": "must not be selected",
            "answer": "must not be selected",
        },
        {
            "kind": "qa_resource_index",
            "fact_id": "fact-b",
            "document_sha256": "b" * 64,
            "question": "must not be selected",
            "answer": "must not be selected",
        },
    ]
    lines = [json.dumps(row) + "\n" for row in rows]
    result = inventory.aggregate_nonsemantic_qa_identity_rows(lines, 2)
    assert result["kind_counts"] == {"qa_manual": 1, "qa_resource_index": 1}
    assert result["selected_kind_counts"] == {"qa_manual": 1}
    assert result["selected_non_url_index_rows"] == 1
    assert result["semantic_fields_accessed_or_emitted"] is False
    assert "question" not in json.dumps(result)
    assert "answer" not in json.dumps(result)


def test_path_firewall_rejects_unapproved_and_forbidden_inputs():
    with pytest.raises(RuntimeError, match="not explicitly allowlisted"):
        inventory.read_safe_object(Path("/tmp/synthetic.json"))
    forbidden = Path("/tmp/protected/synthetic.json")
    with pytest.raises(RuntimeError, match="not explicitly allowlisted"):
        inventory.read_safe_object(forbidden)
