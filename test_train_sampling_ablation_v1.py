"""Synthetic-only tombstone checks for historical sampling V1."""

import pytest

import train_sampling_ablation_v1 as subject


def test_sampling_v1_build_is_fail_closed_without_opening_inputs(monkeypatch):
    monkeypatch.setattr(
        subject,
        "_validate_inputs",
        lambda: pytest.fail("historical inputs must not be opened"),
    )
    with pytest.raises(RuntimeError, match="historical and nonpromotable"):
        subject.build_manifest()


def test_sampling_v1_validation_and_materialization_are_fail_closed():
    with pytest.raises(RuntimeError, match="validation is quarantined"):
        subject.validate_manifest({"synthetic": True})
    with pytest.raises(RuntimeError, match="materialization is quarantined"):
        subject.materialize_variant_rows({"synthetic": True}, "synthetic")


def test_pure_classifier_remains_synthetic_only():
    row = {
        "question": "How should synthetic equipment be stored?",
        "answer": "Keep the synthetic fixture dry.",
        "text": "synthetic fixture",
        "kind": "qa_manual",
        "source": "synthetic",
    }
    assert subject.classify_category(row) in subject.CATEGORIES
