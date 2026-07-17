import copy
import importlib
import json
import subprocess
import sys

import pytest

import analyze_qwen36_bf16_kv_mamba_confirmation_v80b as analysis


def _frozen():
    return json.loads(analysis.OUTPUT.read_text(encoding="ascii"))


def test_v80b_frozen_aggregate_rebuilds_exactly():
    built = analysis.analyze_v80b()
    assert built == _frozen()
    assert built["content_sha256_before_self_field"] == (
        "b38f450a1ae8d74e1e368998c43d74ff5b8d1c5d31726867beedf0f4336083f0"
    )
    assert len(built["twelve_actor_receipt_matrix"]) == 12
    assert built["promotion_authorized"] is False


def test_v80b_report_rebuilds_exactly_and_keeps_quality_closed():
    built = analysis.analyze_v80b()
    report = analysis.render_report_v80b(built)
    assert report == analysis.REPORT.read_text(encoding="ascii")
    assert "- Semantic gate run: false" in report
    assert "- Protected one-shot OOD gate run: false" in report
    assert "- Promotion authorized: false" in report


def test_v80b_self_hash_rejects_mutation():
    value = _frozen()
    analysis._validate_self_hash(value, "frozen")
    changed = copy.deepcopy(value)
    changed["validated_actor_count"] = 11
    with pytest.raises(RuntimeError, match="self hash changed"):
        analysis._validate_self_hash(changed, "mutated")


def test_v80b_wrong_run_bundle_fails_closed(monkeypatch):
    changed = dict(analysis.RUN_BUNDLES)
    changed[analysis.RUNS[1]] = "0" * 64
    monkeypatch.setattr(analysis, "RUN_BUNDLES", changed)
    with pytest.raises(RuntimeError, match="bundle"):
        analysis.analyze_v80b()


def test_v80b_wrong_v83_binding_fails_closed(monkeypatch):
    monkeypatch.setattr(analysis, "V83_RESULT_CONTENT_SHA256", "0" * 64)
    with pytest.raises(RuntimeError, match="sealed JSON content changed"):
        analysis.analyze_v80b()


def test_v80b_analyzer_does_not_import_gpu_stacks():
    code = """
import json, sys
import analyze_qwen36_bf16_kv_mamba_confirmation_v80b as a
value = a.analyze_v80b()
print(json.dumps({
    "torch": "torch" in sys.modules,
    "vllm": "vllm" in sys.modules,
    "promotion": value["promotion_authorized"],
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=analysis.ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    receipt = json.loads(completed.stdout)
    assert receipt == {"torch": False, "vllm": False, "promotion": False}


def test_v80b_module_reload_has_no_output_or_gpu_side_effects(capsys):
    importlib.reload(analysis)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
