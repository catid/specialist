"""Synthetic-only checks for the curated-QA legacy eval quarantine."""

from pathlib import Path

import pytest

import build_curated_qa as curated


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


def test_cli_synthetic_empty_mode_passes_content_free_fact_set(
        monkeypatch, capsys, tmp_path):
    observed = {}

    def synthetic_merge(
            inputs, output, report, facts, curation,
            collision_authorization=None):
        observed.update({
            "collision_authorization": collision_authorization,
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
    synthetic_input = tmp_path / "synthetic-input.jsonl"
    synthetic_input.write_text("", encoding="utf-8")
    curated.main([
        "--synthetic-empty-eval",
        "--inputs", str(synthetic_input),
        "--output", str(tmp_path / "synthetic-output.jsonl"),
        "--report", str(tmp_path / "synthetic-report.json"),
        "--curation",
    ])
    assert observed["facts"] == []
    assert observed["curation"] == []
    assert observed["collision_authorization"] is None
    assert '"eval_fact_count": 0' in capsys.readouterr().out


def test_cli_synthetic_empty_mode_rejects_repository_defaults(monkeypatch):
    monkeypatch.setattr(
        curated,
        "merge",
        lambda *_args, **_kwargs: pytest.fail(
            "repository synthetic mode reached merge"
        ),
    )
    with pytest.raises(RuntimeError, match="outside the repository"):
        curated.main(["--synthetic-empty-eval"])


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


def test_external_non_synthetic_eval_fixture_is_rejected_before_loader(
        monkeypatch, tmp_path):
    fixture = tmp_path / "ordinary-fixture.jsonl"
    fixture.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda _paths: pytest.fail("non-synthetic fixture reached eval loader"),
    )
    with pytest.raises(RuntimeError, match="explicitly identify a synthetic"):
        curated.evaluation_facts([fixture], synthetic_empty=False)


def test_repo_eval_v3_is_rejected_before_any_stat_or_loader(monkeypatch):
    repo_eval = curated.ROOT / "data" / "synthetic-eval-v3-never-touch.jsonl"
    monkeypatch.setattr(
        curated.os,
        "lstat",
        lambda _path: pytest.fail("repository evaluation path was statted"),
    )
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda _paths: pytest.fail("repository evaluation reached loader"),
    )
    with pytest.raises(RuntimeError, match="outside the repository"):
        curated.evaluation_facts([repo_eval], synthetic_empty=False)


def test_resolved_alias_to_legacy_eval_rejects_before_loader(monkeypatch, tmp_path):
    alias = tmp_path / "synthetic-alias.jsonl"
    alias.symlink_to(next(iter(curated.QUARANTINED_LEGACY_EVAL)))
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda _paths: pytest.fail("resolved legacy alias reached eval loader"),
    )
    with pytest.raises(RuntimeError, match="quarantined evaluation"):
        curated.evaluation_facts([alias], synthetic_empty=False)


def test_repo_eval_alias_rejects_before_target_stat(monkeypatch, tmp_path):
    target = curated.ROOT / "data" / "eval_v3_never_touch.jsonl"
    alias = tmp_path / "synthetic-eval-alias.jsonl"
    alias.symlink_to(target)
    real_lstat = curated.os.lstat

    def guarded_lstat(path):
        lexical = Path(path).absolute()
        if lexical == target:
            pytest.fail("repository evaluation symlink target was statted")
        return real_lstat(path)

    monkeypatch.setattr(curated.os, "lstat", guarded_lstat)
    monkeypatch.setattr(
        curated,
        "eval_facts",
        lambda _paths: pytest.fail("repository evaluation alias reached loader"),
    )
    with pytest.raises(RuntimeError, match="outside the repository"):
        curated.evaluation_facts([alias], synthetic_empty=False)
