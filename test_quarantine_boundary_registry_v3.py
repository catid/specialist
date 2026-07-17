"""Content-free tests for the three-incident quarantine boundary."""

import json
from pathlib import Path

import pytest

import build_quarantine_boundary_registry_v3 as boundary
import build_recursive_lookup_access_incident_v3 as recursive_incident


def test_recursive_incident_construction_never_touches_filesystem(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("content-free recursive incident construction touched filesystem")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    receipt = recursive_incident.build_receipt()
    recursive_incident.validate_receipt(receipt)
    assert receipt["scope"]["individual_file_inventory"] == (
        "not_enumerated_reopen_prohibited"
    )


def test_boundary_registry_is_opaque_and_binds_all_three_incidents():
    registry = boundary.build_registry()
    boundary.validate_registry(registry)
    raw = json.dumps(registry, sort_keys=True)
    for value in boundary.EXACT_RELATIVE_PATHS + boundary.RELATIVE_PREFIXES:
        assert value not in raw
    assert registry["exact_path_identity_count"] == 3
    assert registry["prefix_identity_count"] == 2
    assert len(registry["incident_bindings"]) == 3


def test_boundary_policy_denies_lexically_before_resolution_then_rechecks():
    policy = boundary.build_registry()["ancestor_denial_policy"]
    assert policy["lexical_deny_before_resolution_stat_hash_or_open"] is True
    assert policy["lexically_denied_candidate_is_never_resolved"] is True
    assert (
        policy["lexically_allowed_resolution_rechecked_before_metadata_or_open"]
        is True
    )
    assert policy["resolved_target_outside_repository_root_denied"] is True
