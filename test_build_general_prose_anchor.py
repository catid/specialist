import json

import pytest

from build_general_prose_anchor import build_anchor


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def trajectory(topic, observation):
    return {
        "Question: sealed": [{
            "thought": "must never be copied",
            "action": f"Search[{topic}]",
            "observation": observation,
            "answer": "must never be copied",
        }]
    }


def prose(prefix, count=30):
    return " ".join(f"{prefix}{index}" for index in range(count))


def test_builder_is_deterministic_and_copies_only_observations(tmp_path):
    source = tmp_path / "source.jsonl"
    protected = tmp_path / "eval.jsonl"
    write_jsonl(source, [
        trajectory("Alpha", prose("alpha")),
        trajectory("Beta", prose("beta")),
    ])
    write_jsonl(protected, [{
        "question": "a protected question",
        "answer": "SEALED ANSWER",
        "url": "https://example.com/protected",
    }])
    first, first_report = build_anchor(
        source, [protected], max_rows=2, minimum_words=20,
        shingle_width=10,
    )
    second, second_report = build_anchor(
        source, [protected], max_rows=2, minimum_words=20,
        shingle_width=10,
    )
    assert first == second
    assert first_report == second_report
    assert b"SEALED ANSWER" not in first
    assert b"must never be copied" not in first
    assert len(first.splitlines()) == 2


def test_document_label_and_content_shingle_collisions_are_rejected(tmp_path):
    source = tmp_path / "source.jsonl"
    protected = tmp_path / "ood.jsonl"
    shared = prose("shared", 30)
    write_jsonl(source, [
        trajectory("Protected Topic", prose("identity")),
        trajectory("Different Topic", shared + " extra"),
        trajectory("Clean Topic", prose("clean")),
    ])
    write_jsonl(protected, [{
        "title": "Protected Topic",
        "text": "prefix words here " + shared,
        "answer": "not indexed",
    }])
    output, report = build_anchor(
        source, [protected], max_rows=10, minimum_words=20,
        shingle_width=10,
    )
    rows = [json.loads(line) for line in output.splitlines()]
    assert [row["title"] for row in rows] == ["Clean Topic"]
    assert report["collision_rejections"]["document_identity"] == 1
    assert report["collision_rejections"][
        "protected_content_shingle"
    ] == 1


def test_empty_result_fails_closed(tmp_path):
    source = tmp_path / "source.jsonl"
    protected = tmp_path / "ood.jsonl"
    text = prose("same")
    write_jsonl(source, [trajectory("Same", text)])
    write_jsonl(protected, [{"title": "Same", "text": text}])
    with pytest.raises(ValueError, match="every anchor"):
        build_anchor(
            source, [protected], minimum_words=20, shingle_width=10,
        )


def test_sealed_question_answer_and_text_have_zero_build_dependency(tmp_path):
    source = tmp_path / "source.jsonl"
    heldout = tmp_path / "heldout_current.jsonl"
    write_jsonl(source, [trajectory("Clean", prose("clean"))])

    def build_with_sentinel(sentinel):
        write_jsonl(heldout, [{
            "url": "https://example.com/sealed-document",
            "question": sentinel + " question",
            "answer": sentinel + " answer",
            "text": sentinel + " source text",
        }])
        return build_anchor(
            source, [heldout], minimum_words=20, shingle_width=10,
        )

    first_output, first_report = build_with_sentinel("SECRET_ONE")
    second_output, second_report = build_with_sentinel("SECRET_TWO")
    assert first_output == second_output
    assert first_report == second_report
    serialized = json.dumps(first_report)
    assert "SECRET_ONE" not in serialized
    artifact = first_report["protected_artifacts"][0]
    assert artifact["sealed_identity_only_rows"] == 1
    assert artifact["safe_text_fields_indexed"] == 0


def test_heldout_split_rows_are_identity_only_even_in_eval_file(tmp_path):
    source = tmp_path / "source.jsonl"
    evaluation = tmp_path / "eval_qa_v3.jsonl"
    write_jsonl(source, [trajectory("Clean", prose("clean"))])
    write_jsonl(evaluation, [{
        "split": "heldout",
        "normalized_source_url": "https://example.com/sealed",
        "question": "SEALED QUESTION",
        "answer": "SEALED ANSWER",
        "excerpt": "SEALED EXCERPT",
    }])
    _, report = build_anchor(
        source, [evaluation], minimum_words=20, shingle_width=10,
    )
    assert report["protected_artifacts"][0][
        "sealed_identity_only_rows"
    ] == 1
    assert report["protected_artifacts"][0][
        "safe_text_fields_indexed"
    ] == 0
