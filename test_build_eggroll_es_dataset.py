import json

import pytest

from build_eggroll_es_dataset import build


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_combines_document_domain_and_ood_eval_inputs(tmp_path):
    train = tmp_path / "train.jsonl"
    domain = tmp_path / "domain.jsonl"
    ood = tmp_path / "ood.jsonl"
    write_jsonl(train, [{
        "fact_id": "fact-a", "question": "Training question?",
        "answer": "answer",
        "text": "Question: Training question?\nAnswer: answer",
    }])
    write_jsonl(domain, [
        {"item_id": "validation-a", "split": "validation",
         "question": "Validation?", "answer": "yes"},
        {"item_id": "heldout-a", "split": "heldout",
         "question": "Heldout?", "answer": "yes"},
    ])
    write_jsonl(ood, [{
        "item_id": "ood-a", "split": "ood_qa",
        "question": "OOD?", "answer": "yes",
    }])

    manifest = build(train, [domain, ood], tmp_path / "dataset")

    assert manifest["train_rows"] == 1
    assert manifest["eval_splits"] == {
        "heldout": 1, "ood_qa": 1, "validation": 1,
    }
    assert len(manifest["eval_inputs"]) == 2


def test_build_rejects_duplicate_eval_item_across_inputs(tmp_path):
    train = tmp_path / "train.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl(train, [{
        "fact_id": "fact-a", "question": "Training question?",
        "answer": "answer",
        "text": "Question: Training question?\nAnswer: answer",
    }])
    row = {"item_id": "same", "split": "validation",
           "question": "Question?", "answer": "answer"}
    write_jsonl(first, [row])
    write_jsonl(second, [{**row, "split": "ood_qa"}])

    with pytest.raises(ValueError, match="duplicate evaluation item_id"):
        build(train, [first, second], tmp_path / "dataset")
