"""Security checks for the exact four-source manual QA-dev allowlist."""

import json
import os
from pathlib import Path

import pytest

import build_manual_qa_dev_curation_allowlist_v1 as subject
import recipe_evaluation_contract_v2 as v2


def test_allowlist_build_never_touches_terminal_or_denied_prefixes(monkeypatch):
    allowed_dev = {
        Path(os.path.abspath(subject.ROOT / relative))
        for relative, *_rest in subject.DEV_SOURCE_BINDINGS
    }
    denied = tuple(
        Path(os.path.abspath(subject.ROOT / relative))
        for relative in subject.DENIED_RELATIVE_PREFIXES
    )
    corpus_root = Path(os.path.abspath(subject.ROOT / "data/site_corpora"))
    original_open = Path.open
    original_stat = Path.stat

    def classify(path):
        absolute = Path(os.path.abspath(os.fspath(path)))
        if any(absolute == prefix or prefix in absolute.parents for prefix in denied):
            pytest.fail("allowlist builder touched an incident-3 prefix")
        if (absolute == corpus_root or corpus_root in absolute.parents) and (
            absolute not in allowed_dev
        ):
            pytest.fail("allowlist builder touched a non-DEV corpus path")

    def guarded_open(path, *args, **kwargs):
        classify(path)
        return original_open(path, *args, **kwargs)

    def guarded_stat(path, *args, **kwargs):
        classify(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    allowlist = subject.build_allowlist()
    subject.validate_allowlist(allowlist)
    assert allowlist["allowed_dev_source_count"] == 4


def test_allowlist_persists_exact_dev_paths_and_no_terminal_paths():
    allowlist = subject.build_allowlist()
    raw = json.dumps(allowlist, sort_keys=True)
    allowed_paths = {
        item["repository_relative_path"]
        for item in allowlist["allowed_dev_sources"]
    }
    expected_allowed = {
        relative for relative, *_rest in subject.DEV_SOURCE_BINDINGS
    }
    assert allowed_paths == expected_allowed
    assert raw.count("/CORPUS.md") == 4
    for relative, _commit in v2.POST_FREEZE_POOL:
        corpus = relative + "/CORPUS.md"
        if corpus not in expected_allowed:
            assert corpus not in raw
    terminal = allowlist["terminal_boundary"]
    assert terminal["terminal_source_count"] == 9
    assert terminal["terminal_paths_persisted"] is False
    assert terminal["terminal_file_hashes_persisted"] is False
    assert terminal["terminal_read_or_resolution_authorized"] is False


def test_outputs_are_exact_outside_denied_prefixes_and_remain_dev_only():
    allowlist = subject.build_allowlist()
    assert allowlist["allowed_output_paths"] == list(subject.ALLOWED_OUTPUT_PATHS)
    assert not any(
        subject._under_denied_prefix(path)
        for path in allowlist["allowed_output_paths"]
    )
    policy = allowlist["output_policy"]
    assert policy["role"] == "dev_only_never_model_adaptation"
    assert policy["minimum_qa_items_per_source"] == 4
    assert policy["minimum_total_qa_items"] == 16
    assert allowlist["authority"][
        "qa_hpo_or_quality_promotion_authorized"
    ] is False
    assert allowlist["authority"][
        "recursive_search_glob_or_directory_scan_authorized"
    ] is False
