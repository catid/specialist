from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import build_curriculum_ablation_preregistration_v1 as builder
import curriculum_ablation_v1 as curriculum


@pytest.fixture(scope="module")
def built():
    return builder.build_preregistration_v1()


def test_builder_output_is_canonical_and_launch_ineligible(built):
    compact = dict(built)
    observed = compact.pop("content_sha256_before_self_field")
    assert curriculum.canonical_sha256_v1(compact) == observed
    assert built["status"] == "sealed_cpu_preview_launch_ineligible"
    assert built["source_disjoint_extension"]["audit_passed"] is False
    assert built["authorization"]["gpu_launch"] is False
    assert built["authorization"]["protected_holdout_access"] is False


def test_builder_records_rights_and_url_trivia_findings(built):
    findings = built["known_cpu_findings"]
    assert findings["rights_promoted_CPT_sources"] == 29
    assert findings["verified_SFT_rows"] == 433
    assert findings["URL_trivia_rows_excluded"] == 15
    assert findings["rights_blocked_legacy_CPT_sources"] == [
        "crash_restraint",
        "rope365",
        "rope_topia",
        "shibari_atlas",
    ]
    assert findings["rights_blocked_tokens"] == 1_107_618


def test_builder_never_opens_dev_ood_or_protected_files(monkeypatch):
    forbidden = {
        (builder.ROOT / "data/eval_qa_v3.jsonl").resolve(),
        (builder.ROOT / "data/ood_qa_v3.jsonl").resolve(),
        (builder.ROOT / "data/ood_prose_v3.jsonl").resolve(),
        (
            builder.ROOT
            / "experiments/sft_controls/v37a_shadow_folds_v412/"
            "fold_3_shadow_dev.jsonl"
        ).resolve(),
    }
    opened = []
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        resolved = Path(path).resolve()
        if resolved in forbidden:
            raise AssertionError(f"opened forbidden evaluation content: {resolved}")
        opened.append(resolved)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    result = builder.build_preregistration_v1()
    assert not set(opened) & forbidden
    assert result["access_receipt"]["protected_holdout_semantics_opened"] is False
    assert result["access_receipt"]["dev_or_ood_semantics_opened"] is False


def test_validator_has_no_model_dataset_or_gpu_imports():
    tree = ast.parse(Path(curriculum.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(item.name.split(".")[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {
        "torch", "ray", "vllm", "numpy", "datasets", "transformers"
    }


def test_main_writes_reopenable_artifact(tmp_path, capsys):
    output = tmp_path / "curriculum.json"
    assert builder.main(["--output", str(output)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    value = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["path"] == str(output.resolve())
    assert receipt["file_sha256"] == builder.file_sha256_v1(output)
    assert receipt["content_sha256"] == value[
        "content_sha256_before_self_field"
    ]
    assert curriculum.validate_plan_v1(value)["status"] == (
        "sealed_cpu_preview_launch_ineligible"
    )


def test_checked_in_artifact_is_exact_builder_output(built):
    checked_in = json.loads(builder.OUTPUT.read_text(encoding="utf-8"))
    assert checked_in == built
    assert builder.file_sha256_v1(builder.OUTPUT) == (
        "d4f36022cc41d381e337b19cbfb3a4136b9c62eb31c359f64594e73628a091fb"
    )
