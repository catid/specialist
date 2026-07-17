"""Synthetic-only tests for the additive V73D Stage-A boundary auditor."""

import json
from pathlib import Path

import pytest

import audit_v73d_systems_only_import_graph_v2 as subject


def _write_module(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")


def test_synthetic_runtime_closure_is_complete_and_authority_false(tmp_path):
    false_rules = "\n".join(
        f"    {key!r}: False," for key in subject.REQUIRED_FALSE_AUTHORITY_KEYS
    )
    _write_module(
        tmp_path / "runtime_a.py",
        "import runtime_b\nPOLICY = {\n" + false_rules + "\n}\n",
    )
    _write_module(tmp_path / "runtime_b.py", "VALUE = 1\n")

    graph = subject.analyze_import_graph(tmp_path, ("runtime_a.py",))

    assert graph["local_closure_modules"] == 2
    assert graph["mutable_prereg_builder_reachable"] is False
    assert len(graph["local_import_edge_identity_sha256s"]) == 1
    assert graph["required_false_authority_rule_count"] == 4
    assert graph["true_quality_or_promotion_authority_rule_count"] == 0


def test_synthetic_boundary_references_are_opaque_and_builder_is_detected(
    tmp_path,
):
    quarantined = sorted(subject.QUARANTINED_MODULE_REFERENCES)[0]
    semantic = subject.SEMANTIC_BOUNDARY_LITERALS[0]
    builder_module = subject.MUTABLE_PREREG_BUILDER.removesuffix(".py")
    _write_module(
        tmp_path / "runtime.py",
        f"import {quarantined}\nimport {builder_module}\nBOUNDARY = {semantic!r}\n",
    )
    _write_module(tmp_path / f"{quarantined}.py", "VALUE = 1\n")
    _write_module(tmp_path / subject.MUTABLE_PREREG_BUILDER, "VALUE = 2\n")

    graph = subject.analyze_import_graph(tmp_path, ("runtime.py",))
    encoded = json.dumps(graph, sort_keys=True)

    assert graph["mutable_prereg_builder_reachable"] is True
    assert graph["quarantined_boundary_reference_count"] == 1
    assert graph["semantic_evaluation_path_reference_count"] == 1
    assert quarantined not in encoded
    assert semantic not in encoded


def test_zero_success_attestation_accepts_exact_zeroes():
    subject.validate_zero_success_attestation({
        "zero_successful_protected_operation_attestation": {
            "scope": "byte_stable_selected_synthetic_subprocess_receipts",
            **{name: 0 for name in subject.ZERO_SUCCESS_COUNTERS},
            "actual_stage_b_gpu_run_attested": False,
        }
    })


@pytest.mark.parametrize("counter", subject.ZERO_SUCCESS_COUNTERS)
def test_zero_success_attestation_rejects_nonzero(counter):
    counters = {name: 0 for name in subject.ZERO_SUCCESS_COUNTERS}
    counters[counter] = 1
    with pytest.raises(RuntimeError, match="zero-success"):
        subject.validate_zero_success_attestation({
            "zero_successful_protected_operation_attestation": {
                "scope": "byte_stable_selected_synthetic_subprocess_receipts",
                **counters,
                "actual_stage_b_gpu_run_attested": False,
            }
        })
