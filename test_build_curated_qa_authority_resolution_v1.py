from __future__ import annotations

import json

import build_curated_qa_authority_resolution_v1 as resolution


def _checked() -> dict:
    value = json.loads(resolution.OUTPUT.read_text(encoding="utf-8"))
    unsigned = dict(value)
    declared = unsigned.pop("content_sha256_before_self_field")
    assert resolution.canonical_sha256(unsigned) == declared
    return value


def test_resolution_never_claims_to_reopen_or_reauthorize_legacy_eval():
    value = _checked()
    assert value["authority"] == {
        "quarantined_legacy_evaluation_paths_resolved_statted_hashed_counted_or_opened": False,
        "protected_holdout_ood_terminal_or_manual_review_semantics_opened": False,
        "legacy_curated_qa_authorized_for_plan_training": False,
        "active_v440_train_projection_authorized_for_snapshot_construction": True,
        "training_launch_authorized": False,
    }
    assert value["legacy_curated_qa"]["role"] == "retired_reproducibility_artifact_only"
    assert value["resolution"]["opaque_legacy_boundary_not_forged_or_reconstructed"] is True


def test_active_v440_authority_is_complete_disjoint_and_final_redacted():
    value = _checked()
    active = value["active_qa"]
    assert active["v440_rows_by_partition"] == {
        "train": 382,
        "development": 74,
        "final": 60,
    }
    assert active["v440_rows_total"] == 516
    assert active["final_records_emitted"] is False
    assert all(active["disjointness"].values())
    qa = [item for item in active["materialized_safe_projections"] if item["kind"] == "v440_qa"]
    assert [(item["partition"], item["rows"]) for item in qa] == [
        ("train", 382), ("development", 74),
    ]


def test_resolution_rebuild_and_projection_hash_checks_are_exact():
    value = _checked()
    assert resolution.build() == value
    for item in value["active_qa"]["materialized_safe_projections"]:
        path = resolution.ROOT / item["path"]
        assert resolution.file_sha256(path) == item["file_sha256"]
