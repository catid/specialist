import json

import pytest

from compare_eggroll_es_eval import compare


def write_rows(path, rewards, formats=None):
    if formats is None:
        formats = ["partial" if reward else "incorrect" for reward in rewards]
    rows = [
        {
            "prompt": f"prompt {index}",
            "answer": f"answer {index}",
            "reward": reward,
            "format": format_name,
        }
        for index, (reward, format_name) in enumerate(zip(rewards, formats))
    ]
    path.write_text(json.dumps(rows))


def test_compare_reports_paired_reward_changes(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    write_rows(left, [0.0, 0.5, 1.0], ["incorrect", "partial", "exact"])
    write_rows(right, [0.5, 0.25, 1.0], ["partial", "partial", "exact"])

    report = compare(left, right, bootstrap_samples=100, seed=7)

    assert report["aligned_items"] == 3
    assert report["right_wins"] == 1
    assert report["left_wins"] == 1
    assert report["ties"] == 1
    assert report["left"]["exact"] == report["right"]["exact"] == 1
    assert report["left"]["nonzero"] == 2
    assert report["right"]["nonzero"] == 3
    assert report["right_minus_left_mean_reward"] == pytest.approx(1 / 12)
    assert report["schema"] == "eggroll-es-paired-eval-v2"
    assert len(report["paired_items"]) == 3
    assert report["paired_items"][2]["left_exact"] is True
    assert report["paired_items"][2]["right_exact"] is True
    assert len(report["paired_items"][0]["key_sha256"]) == 64


def test_compare_fails_when_rows_are_not_aligned(tmp_path):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    write_rows(left, [0.0])
    write_rows(right, [0.0])
    rows = json.loads(right.read_text())
    rows[0]["answer"] = "different"
    right.write_text(json.dumps(rows))

    with pytest.raises(ValueError, match="diverge"):
        compare(left, right, bootstrap_samples=10)
