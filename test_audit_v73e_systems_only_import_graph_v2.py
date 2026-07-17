"""Synthetic-only tests for the V73E Stage-A closure auditor."""

from pathlib import Path

import pytest

import audit_v73e_systems_only_import_graph_v2 as subject


def _write(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")


def test_synthetic_runtime_closure_is_complete_and_authority_false(tmp_path):
    false_rules = "\n".join(
        f"    {key!r}: False,"
        for key in subject.REQUIRED_FALSE_AUTHORITY_KEYS
    )
    _write(
        tmp_path / "runtime_a.py",
        "import runtime_b\nPOLICY = {\n" + false_rules + "\n}\n",
    )
    _write(tmp_path / "runtime_b.py", "VALUE = 1\n")
    graph = subject.analyze_import_graph(
        tmp_path,
        ("runtime_a.py",),
        authority_guard_modules=("runtime_a.py",),
    )
    assert graph["local_closure_modules"] == 2
    assert graph["mutable_prereg_builder_reachable"] is False
    assert graph["required_false_authority_rule_count"] == 4
    assert graph["true_quality_or_promotion_authority_rule_count"] == 0


def test_zero_success_attestation_rejects_nonzero():
    counters = {name: 0 for name in subject.ZERO_SUCCESS_COUNTERS}
    counters[subject.ZERO_SUCCESS_COUNTERS[0]] = 1
    with pytest.raises(RuntimeError, match="zero-success"):
        subject.validate_zero_success_attestation({
            "zero_successful_protected_operation_attestation": {
                "scope": "byte_stable_selected_synthetic_subprocess_receipts",
                **counters,
                "actual_stage_b_gpu_run_attested": False,
            }
        })
