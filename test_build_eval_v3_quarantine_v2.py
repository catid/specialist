"""Synthetic fail-closed check for permanently quarantined Eval V3 outputs."""

import sys

import pytest

import build_eval_v3 as subject


def test_default_quarantined_outputs_reject_before_any_input_read(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_eval_v3.py"])
    monkeypatch.setattr(
        subject,
        "read_jsonl",
        lambda *_args, **_kwargs: pytest.fail("inputs must remain unopened"),
    )
    with pytest.raises(SystemExit) as error:
        subject.main()
    assert error.value.code == 2
