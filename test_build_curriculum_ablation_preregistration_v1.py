"""Synthetic-only fail-closed check for the source-overlapping curriculum."""

import pytest

import build_curriculum_ablation_preregistration_v1 as subject


def test_curriculum_rebuild_is_blocked_before_any_source_open():
    with pytest.raises(RuntimeError, match="fresh source-disjoint extension"):
        subject.build_preregistration_v1()
