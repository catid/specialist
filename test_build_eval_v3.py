import hashlib
import json
from pathlib import Path

import pytest

from build_eval_v3 import (
    build_artifacts,
    candidate_review_sha256,
    normalize_source_url,
    sha256_text,
    source_urls,
)


def write_jsonl(path, rows):
    path.write_text("".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))


def rows_from_bytes(content):
    return [json.loads(line) for line in content.decode().splitlines()]


def fixture_files(tmp_path):
    train = tmp_path / "train.jsonl"
    legacy = tmp_path / "eval_v2.jsonl"
    heldout = tmp_path / "heldout.jsonl"
    ood_qa = tmp_path / "ood_qa.jsonl"
    ood_prose = tmp_path / "ood_prose.jsonl"
    review = tmp_path / "review.json"
    write_jsonl(train, [{
        "question": "q", "answer": "a",
        "url": "https://example.com/trained/?utm_source=x",
        "evidence_url": "https://evidence.example/fact#citation",
    }])
    documents = [
        {
            "source": "alpha",
            "url": "https://alpha.example/doc-{}".format(index),
            "text": f"document {index} evidence", "title": f"doc {index}",
        }
        for index in range(4)
    ] + [
        {
            "source": "beta",
            "url": "https://beta.example/doc-{}".format(index),
            "text": f"beta document {index} evidence",
            "title": f"beta {index}",
        }
        for index in range(4)
    ] + [{
        "source": "alpha", "url": "http://www.example.com/trained#old",
        "text": "trained document evidence", "title": "trained",
    }]
    write_jsonl(heldout, documents)
    eval_rows = []
    for document in documents:
        eval_rows.append({
            "split": "heldout",
            "question": "What is this? " + document["title"],
            "answer": document["title"], "excerpt": document["text"],
            "url": document["url"],
        })
    eval_rows.append({
        "split": "train", "question": "Legacy train question?",
        "answer": "answer", "excerpt": "legacy", "url":
        "https://example.com/trained",
    })
    write_jsonl(legacy, eval_rows)
    write_jsonl(ood_qa, [{"question": "Capital?", "answer": "City"}])
    write_jsonl(ood_prose, [{
        "title": "Unrelated Topic", "text": "An unrelated prose probe.",
    }])
    review_candidates = []
    for row in eval_rows[:-1]:
        normalized = normalize_source_url(row["url"])
        if normalized == normalize_source_url(
            "https://example.com/trained"
        ):
            continue
        material = "\0".join((normalized, row["question"], row["answer"]))
        review_candidates.append({
            "answer": row["answer"],
            "item_id": "evalv3-" + sha256_text(material)[:20],
            "normalized_source_url": normalized,
            "question": row["question"],
        })
    review.write_text(json.dumps({
        "drops": [],
        "reviewed_candidate_set_sha256": candidate_review_sha256(
            review_candidates
        ),
        "reviewed_rows": len(review_candidates),
        "reason_definitions": {
            "test_rejection": "Rejected by a unit-test review."
        },
        "reviewer": "unit test",
        "schema": "specialist-eval-v3-manual-review-v1",
    }))
    return train, legacy, heldout, ood_qa, ood_prose, review


def test_normalize_source_url_folds_safe_aliases_and_keeps_semantic_query():
    left = normalize_source_url(
        "HTTP://WWW.Example.COM:80/a/./b/../c/%7Ename/?b=2&utm_source=x&a=1#x"
    )
    right = normalize_source_url(
        "https://example.com/a/c/~name?a=1&b=2"
    )
    assert left == right == "web://example.com/a/c/~name?a=1&b=2"
    assert normalize_source_url(
        "https://youtu.be/OsIcEtCoKHo?si=tracking"
    ) == normalize_source_url(
        "https://www.youtube.com/watch?v=OsIcEtCoKHo&utm_medium=social"
    )


def test_source_urls_includes_scalar_and_list_provenance_fields():
    record = {
        "url": "https://one.example/a",
        "canonical_urls": ["https://two.example/b"],
        "source_lineage": {"input": "/tmp/not-a-url"},
        "answer": "https://not-provenance.example/answer",
    }
    assert source_urls(record) == [
        ("url", "https://one.example/a"),
        ("canonical_urls", "https://two.example/b"),
    ]


def test_build_is_deterministic_disjoint_and_document_partitioned(tmp_path):
    paths = fixture_files(tmp_path)
    original_hashes = [hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths]
    first, report = build_artifacts(*paths, split_seed="test-seed")
    second, second_report = build_artifacts(*paths, split_seed="test-seed")

    assert first == second
    assert report == second_report
    assert original_hashes == [hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths]
    domain = rows_from_bytes(first["domain"])
    assert len(domain) == 8
    assert all("trained" not in row["normalized_source_url"] for row in domain)
    validation_urls = {row["normalized_source_url"] for row in domain
                       if row["split"] == "validation"}
    heldout_urls = {row["normalized_source_url"] for row in domain
                    if row["split"] == "heldout"}
    assert validation_urls
    assert heldout_urls
    assert validation_urls.isdisjoint(heldout_urls)
    assert {
        row["source"] for row in domain if row["split"] == "validation"
    } == {
        "alpha", "beta"
    }
    assert {
        row["source"] for row in domain if row["split"] == "heldout"
    } == {
        "alpha", "beta"
    }
    assert report["historical_eval_audit"]["by_legacy_split"]["heldout"][
        "overlap_with_candidate_train"
    ]["rows"] == 1
    assert report["disjointness"]["passed"] is True
    assert not any(report["disjointness"]["collisions"].values())
    assert report["metric_policy"]["sealed_holdout"]["selection_use"] == (
        "prohibited"
    )


def test_build_rejects_ood_prose_contaminated_by_training(tmp_path):
    paths = list(fixture_files(tmp_path))
    write_jsonl(paths[0], [
        {
            "question": "q", "answer": "a",
            "url": "https://example.com/trained",
        },
        {
            "question": "q2", "answer": "a2",
            "url": "https://en.wikipedia.org/wiki/Unrelated_Topic",
        },
    ])
    with pytest.raises(ValueError, match="disjointness audit failed"):
        build_artifacts(*paths, split_seed="test-seed")


def test_build_rejects_legacy_heldout_row_without_document(tmp_path):
    paths = list(fixture_files(tmp_path))
    rows = [
        json.loads(line) for line in Path(paths[1]).read_text().splitlines()
    ]
    rows[0]["url"] = "https://missing.example/document"
    write_jsonl(paths[1], rows)
    with pytest.raises(ValueError, match="absent from heldout_docs"):
        build_artifacts(*paths, split_seed="test-seed")


def test_build_rejects_review_for_a_different_candidate_cohort(tmp_path):
    paths = list(fixture_files(tmp_path))
    review = json.loads(Path(paths[5]).read_text())
    review["reviewed_candidate_set_sha256"] = "0" * 64
    Path(paths[5]).write_text(json.dumps(review))
    with pytest.raises(ValueError, match="does not cover this candidate"):
        build_artifacts(*paths, split_seed="test-seed")


def test_manual_drop_is_applied_before_document_partition(tmp_path):
    paths = list(fixture_files(tmp_path))
    review = json.loads(Path(paths[5]).read_text())
    candidate_id = "evalv3-" + sha256_text("\0".join((
        normalize_source_url("https://alpha.example/doc-0"),
        "What is this? doc 0",
        "doc 0",
    )))[:20]
    review["drops"] = [{
        "item_id": candidate_id,
        "reason_code": "test_rejection",
    }]
    Path(paths[5]).write_text(json.dumps(review))

    artifacts, report = build_artifacts(*paths, split_seed="test-seed")

    assert len(rows_from_bytes(artifacts["domain"])) == 7
    assert report["manual_review"]["reviewed_rows"] == 8
    assert report["manual_review"]["kept_rows"] == 7
    assert report["manual_review"]["reason_counts"] == {
        "test_rejection": 1
    }
