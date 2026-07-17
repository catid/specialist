"""Synthetic-only tombstone check for the quarantined V46A QA runner."""

import pytest

import run_once_only_holdout_eval_v46a as subject


def test_v46a_runner_fails_before_any_protected_source_open():
    with pytest.raises(RuntimeError, match="permanently quarantined"):
        subject.main([])
