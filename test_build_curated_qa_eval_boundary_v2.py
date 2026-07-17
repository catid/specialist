"""Synthetic-only checks for the curated-QA legacy eval quarantine."""

from pathlib import Path

import pytest

import build_curated_qa as curated
import build_legacy_eval_collision_incident_v2 as incident


def test_incident_receipt_build_never_touches_quarantined_sources(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("content-free incident construction touched a filesystem path")

    monkeypatch.setattr(Path, "open", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    receipt = incident.build_receipt()
    incident.validate_receipt(receipt)
    assert receipt["scope"]["touched_source_count"] == 2
    assert receipt["scope"]["source_file_hashes"] == "unknown_not_computed"


def test_synthetic_empty_eval_never_calls_legacy_loader(monkeypatch):
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda _paths: pytest.fail("synthetic-empty mode called eval loader"),
    )
    assert curated.evaluation_facts(None, synthetic_empty=True) == []


@pytest.mark.parametrize("path", sorted(curated.QUARANTINED_LEGACY_EVAL))
def test_each_legacy_eval_path_is_rejected_before_loader(monkeypatch, path):
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda _paths: pytest.fail("quarantined path reached eval loader"),
    )
    with pytest.raises(RuntimeError, match="quarantined"):
        curated.evaluation_facts([path], synthetic_empty=False)


def test_cli_has_no_implicit_eval_default(monkeypatch):
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda _paths: pytest.fail("missing explicit choice reached eval loader"),
    )
    monkeypatch.setattr(
        curated,
        "merge",
        lambda *_args, **_kwargs: pytest.fail("invalid CLI reached merge"),
    )
    with pytest.raises(SystemExit) as error:
        curated.main([])
    assert error.value.code == 2


def test_cli_synthetic_empty_mode_passes_content_free_fact_set(monkeypatch, capsys):
    observed = {}

    def synthetic_merge(inputs, output, report, facts, curation):
        observed.update({
            "inputs": inputs,
            "output": output,
            "report": report,
            "facts": facts,
            "curation": curation,
        })
        return {"eval_fact_count": len(facts), "synthetic": True}

    monkeypatch.setattr(curated, "merge", synthetic_merge)
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda _paths: pytest.fail("synthetic-empty CLI called eval loader"),
    )
    curated.main(["--synthetic-empty-eval"])
    assert observed["facts"] == []
    assert '"eval_fact_count": 0' in capsys.readouterr().out


def test_explicit_fresh_eval_path_is_forwarded(monkeypatch, tmp_path):
    fresh = tmp_path / "synthetic-fresh-eval.jsonl"
    fresh.write_text("", encoding="utf-8")
    sentinel = [object()]
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda paths: sentinel if paths == [fresh] else pytest.fail("wrong paths"),
    )
    assert curated.evaluation_facts([fresh], synthetic_empty=False) is sentinel


def test_resolved_alias_to_legacy_eval_rejects_before_loader(monkeypatch, tmp_path):
    alias = tmp_path / "synthetic-alias.jsonl"
    monkeypatch.setattr(
        curated.Path,
        "resolve",
        lambda self, **_kwargs: next(iter(curated.QUARANTINED_LEGACY_EVAL))
        if self == alias
        else self,
    )
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda _paths: pytest.fail("resolved legacy alias reached eval loader"),
    )
    with pytest.raises(RuntimeError, match="aliases a quarantined"):
        curated.evaluation_facts([alias], synthetic_empty=False)
