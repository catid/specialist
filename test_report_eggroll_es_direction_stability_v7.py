import copy

import pytest

import report_eggroll_es_direction_stability_v7 as reporter


def test_coefficient_cosine_contract():
    assert reporter.coefficient_cosine([1.0] * 16, [2.0] * 16) == 1.0
    assert reporter.coefficient_cosine([1.0] * 16, [-1.0] * 16) == -1.0
    with pytest.raises(ValueError, match="16 values"):
        reporter.coefficient_cosine([1.0], [1.0])
    with pytest.raises(ValueError, match="zero vector"):
        reporter.coefficient_cosine([0.0] * 16, [1.0] * 16)


def test_report_is_exactly_four_runs_and_emits_no_benchmark_metrics(monkeypatch):
    fixtures = {}
    for arm in ("front", "middle_late"):
        for seed in (43, 44):
            path = f"/{arm}-{seed}.json"
            fixtures[path] = {
                "arm": arm, "seed": seed, "journal": path,
                "journal_file_sha256": f"file-{arm}-{seed}",
                "content_sha256": f"content-{arm}-{seed}",
                "coefficient_sha256": f"coefficient-{arm}-{seed}",
                "robust_plan_sha256": f"robust-{arm}-{seed}",
                "coefficients": [float(index + seed) for index in range(16)],
            }
    monkeypatch.setattr(
        reporter, "_load_run", lambda path: copy.deepcopy(fixtures[path]),
    )
    report = reporter.build_report(list(fixtures))
    assert report["preregistered_threshold"] == 0.5
    assert report["coverage"] == {
        "arms": ["front", "middle_late"],
        "seeds": [43, 44], "target_alphas": [0.0],
    }
    serialized = repr(report).casefold()
    assert "mean_reward" not in serialized
    assert "ood_prose" not in serialized
    assert "holdout" in report["selection_policy"]
    with pytest.raises(ValueError, match="exactly front/middle_late"):
        reporter.build_report(list(fixtures)[:3])
