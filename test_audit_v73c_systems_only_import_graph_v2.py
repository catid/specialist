"""Synthetic-only tests for the V73C closure auditor."""

from pathlib import Path

import audit_v73c_systems_only_import_graph_v2 as subject


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
    assert graph["quarantined_boundary_reference_count"] == 0
    assert graph["semantic_evaluation_path_reference_count"] == 0
    assert graph["required_false_authority_rule_count"] == 4
    assert graph["true_quality_or_promotion_authority_rule_count"] == 0


def test_synthetic_forbidden_import_and_path_are_detected(tmp_path):
    _write_module(
        tmp_path / "runtime_bad.py",
        "import recipe_evaluation_contract_v1\n"
        "BOUNDARY = 'recipe_evaluation_compute_contract_v1.json'\n",
    )
    graph = subject.analyze_import_graph(tmp_path, ("runtime_bad.py",))
    assert graph["quarantined_boundary_reference_count"] == 1
    assert graph["semantic_evaluation_path_reference_count"] == 1


def test_mutable_prereg_builder_in_runtime_closure_is_detected(tmp_path):
    module_name = subject.MUTABLE_PREREG_BUILDER.removesuffix(".py")
    _write_module(tmp_path / "runtime.py", f"import {module_name}\n")
    _write_module(tmp_path / subject.MUTABLE_PREREG_BUILDER, "VALUE = 1\n")
    graph = subject.analyze_import_graph(tmp_path, ("runtime.py",))
    assert graph["mutable_prereg_builder_reachable"] is True
