"""Synthetic-only tombstone check for V46D QA terminal receipts."""

import pytest

import build_v46d_terminal_holdout_receipt as subject


def test_v46d_terminal_receipt_builder_is_quarantined():
    with pytest.raises(RuntimeError, match="permanently quarantined"):
        subject.build()
