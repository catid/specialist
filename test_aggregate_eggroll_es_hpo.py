import pytest

from aggregate_eggroll_es_hpo import aggregate


def journal(seed, train_delta, ood_delta, name="candidate"):
    baseline = {"validation": 0.5, "ood_qa": 0.4}
    treatment = {
        "validation": baseline["validation"] + train_delta,
        "ood_qa": baseline["ood_qa"] + ood_delta,
    }
    return {
        "seed": seed,
        "trainer_sha256": "trainer",
        "dataset": {"train_arrow_sha256": "dataset"},
        "baseline": {
            "validation_score": baseline["validation"],
            "evaluation_scores": baseline,
        },
        "results": [{
            "name": name,
            "validation_score": treatment["validation"],
            "evaluation_scores": treatment,
        }],
    }


def test_selects_positive_stable_candidate_with_ood_guard():
    result = aggregate(
        [journal(1, 0.02, 0.0), journal(2, 0.01, 0.01)],
        "validation", ["ood_qa"],
    )
    assert result["selected"] == "candidate"
    assert result["candidates"][0]["positive_seed_fraction"] == 1.0


def test_rejects_candidate_with_mean_gain_but_one_ood_regression():
    result = aggregate(
        [journal(1, 0.02, 0.0), journal(2, 0.01, -0.001)],
        "validation", ["ood_qa"],
    )
    assert result["selected"] == "baseline"


def test_rejects_seed_unstable_candidate():
    result = aggregate(
        [journal(1, 0.03, 0.0), journal(2, -0.01, 0.0)],
        "validation", ["ood_qa"], min_positive_seed_fraction=1.0,
    )
    assert result["selected"] == "baseline"


def test_refuses_mixed_dataset_hashes():
    journals = [journal(1, 0.1, 0.0), journal(2, 0.1, 0.0)]
    journals[1]["dataset"]["train_arrow_sha256"] = "changed"
    with pytest.raises(ValueError, match="dataset/trainer"):
        aggregate(journals, "validation", ["ood_qa"])


def test_refuses_mixed_guard_eval_snapshots_even_with_same_train_hash():
    journals = [journal(1, 0.1, 0.0), journal(2, 0.1, 0.0)]
    for item in journals:
        item["dataset"]["snapshot"] = {
            "train_arrow": {"sha256": "same"},
            "eval_arrows": {"validation": {"sha256": "validation"},
                            "ood_qa": {"sha256": "ood"}},
        }
    journals[1]["dataset"]["snapshot"]["eval_arrows"]["ood_qa"][
        "sha256"
    ] = "changed"
    with pytest.raises(ValueError, match="dataset/trainer"):
        aggregate(journals, "validation", ["ood_qa"])


def test_refuses_same_name_with_different_candidate_settings():
    journals = [journal(1, 0.1, 0.0), journal(2, 0.1, 0.0)]
    journals[0]["results"][0].update(
        {"sigma": 0.001, "alpha": 0.00025, "steps": 3}
    )
    journals[1]["results"][0].update(
        {"sigma": 0.002, "alpha": 0.00025, "steps": 3}
    )
    with pytest.raises(ValueError, match="candidate settings"):
        aggregate(journals, "validation", ["ood_qa"])


def test_refuses_duplicate_seed_journals():
    with pytest.raises(ValueError, match="duplicate seeds"):
        aggregate(
            [journal(1, 0.1, 0.0), journal(1, 0.1, 0.0)],
            "validation", ["ood_qa"],
        )


def test_risk_adjusted_regression_retains_baseline():
    result = aggregate(
        [journal(1, 0.03, 0.0), journal(2, -0.001, 0.0),
         journal(3, 0.001, 0.0)],
        "validation", ["ood_qa"], min_positive_seed_fraction=0.6,
        risk_penalty=1.0,
    )
    assert result["candidates"][0]["mean_selection_delta"] > 0
    assert result["candidates"][0]["robust_score"] < 0
    assert result["selected"] == "baseline"
