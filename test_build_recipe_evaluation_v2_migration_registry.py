"""Synthetic and code-metadata checks for the additive V2 migration registry."""

import hashlib
import subprocess

import pytest

import build_recipe_evaluation_v2_migration_registry as subject


def _synthetic_registry():
    value = {
        "schema": subject.SCHEMA,
        "status": (
            "v2_provisional_prose_only_all_incidents_quarantined_"
            "history_preserved"
        ),
        "historical_artifacts": {"rewrite_prohibited": True},
        "v2_contract": {
            "qa_hpo_or_general_quality_promotion_authorized": False,
        },
        "v2_resealed_source_reservation": {
            "required_before_any_future_dataset_write": True,
            "terminal_source_count": 9,
            "dev_source_count": 4,
            "reserved_source_count": 13,
            "historical_selected_source_count": 12,
        },
        "content_free_quarantine_boundary": {
            "exact_path_identity_count": 3,
            "prefix_identity_count": 2,
            "plaintext_boundary_paths_persisted": False,
        },
        "immutable_incident_receipts": [{}, {}, {}],
        "superseded_v2_revision": {
            "status": "immutable_superseded_nonpromotable",
        },
        "v73c_stage_a_systems_only_closure": {
            "systems_trace_only": True,
            "quality_hpo_or_promotion_authorized": False,
            "distinct_postrun_boundary_denial_receipt_required": True,
        },
        "v1_implementation_tombstone": {
            "pre_tombstone_file_sha256": (
                subject.V1_PRE_TOMBSTONE_FILE_SHA256
            ),
            "pre_tombstone_git_blob": subject.V1_PRE_TOMBSTONE_GIT_BLOB,
            "pre_tombstone_commit": subject.V1_PRE_TOMBSTONE_COMMIT,
            "pre_tombstone_definition_commit": (
                subject.V1_PRE_TOMBSTONE_DEFINITION_COMMIT
            ),
        },
    }
    value["content_sha256_before_self_field"] = subject.v2.canonical_sha256(
        value
    )
    return value


def test_synthetic_migration_registry_validates_and_tampering_fails():
    value = _synthetic_registry()
    subject.validate_registry(value)
    value["historical_artifacts"]["rewrite_prohibited"] = False
    with pytest.raises(RuntimeError, match="invalid V2 migration registry"):
        subject.validate_registry(value)


def test_historical_artifact_inventory_is_unique_and_v1_bound():
    assert len(subject.HISTORICAL_ARTIFACTS) == len(
        set(subject.HISTORICAL_ARTIFACTS)
    )
    assert any("recipe_evaluation_compute_contract_v1" in path
               for path in subject.HISTORICAL_ARTIFACTS)
    assert all(
        path.endswith((".json", ".md"))
        for path in subject.HISTORICAL_ARTIFACTS
    )


def test_pre_tombstone_implementation_binding_matches_exact_git_object():
    payload = subprocess.check_output(
        (
            "git", "show",
            f"{subject.V1_PRE_TOMBSTONE_COMMIT}:{subject.V1_IMPLEMENTATION_PATH}",
        ),
        cwd=subject.ROOT,
    )
    assert hashlib.sha256(payload).hexdigest() == (
        subject.V1_PRE_TOMBSTONE_FILE_SHA256
    )
    blob = subprocess.check_output(
        (
            "git", "rev-parse",
            f"{subject.V1_PRE_TOMBSTONE_COMMIT}:{subject.V1_IMPLEMENTATION_PATH}",
        ),
        cwd=subject.ROOT,
        text=True,
    ).strip()
    assert blob == subject.V1_PRE_TOMBSTONE_GIT_BLOB
