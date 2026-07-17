"""Incident-only check; exclude this module from ordinary test collection."""

from pathlib import Path

import pytest


def test_incident_receipt_build_never_touches_quarantined_sources(monkeypatch):
    # Keep the incident helper import local so collecting the safe synthetic
    # boundary module cannot import or initialize incident-only code.
    import build_legacy_eval_collision_incident_v2 as incident

    def forbidden(*_args, **_kwargs):
        pytest.fail("content-free incident construction touched a filesystem path")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    receipt = incident.build_receipt()
    incident.validate_receipt(receipt)
    assert receipt["scope"]["touched_source_count"] == 2
    assert receipt["scope"]["source_file_hashes"] == "unknown_not_computed"
