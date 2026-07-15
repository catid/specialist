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


def add_prose_gate(item, delta=0.0, lower=0.0, upper=0.0,
                   threshold=0.0, declared_passed=None):
    item["ood_prose_guard_enabled"] = True
    item["max_ood_prose_degradation"] = threshold
    gate_passed = lower >= -threshold
    item["results"][0]["ood_prose_gate"] = {
        "metric": "mean_token_logprob",
        "higher_is_better": True,
        "baseline": 0.0,
        "final": delta,
        "delta": delta,
        "max_degradation": threshold,
        "paired_document_bootstrap_95_ci": [lower, upper],
        "bootstrap": {
            "unit": "normalized_source_url",
            "document_count": 16,
            "samples": 20000,
            "seed": 20260714,
            "percentiles": [0.025, 0.975],
        },
        "passed": gate_passed,
    }
    item["results"][0]["ood_prose_guard_passed"] = (
        gate_passed if declared_passed is None else declared_passed
    )
    return item


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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_scores_and_policy_values(value):
    journals = [journal(1, 0.1, 0.0), journal(2, 0.1, 0.0)]
    journals[0]["results"][0]["evaluation_scores"]["validation"] = value
    with pytest.raises(ValueError, match="finite"):
        aggregate(journals, "validation", ["ood_qa"])

    journals = [journal(1, 0.1, 0.0), journal(2, 0.1, 0.0)]
    with pytest.raises(ValueError, match="finite"):
        aggregate(
            journals, "validation", ["ood_qa"],
            max_guard_degradation=value,
        )


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


def test_rejects_exact_guard_loss_across_seeds():
    journals = [journal(1, 0.1, 0.0), journal(2, 0.1, 0.0)]
    for item in journals:
        item["baseline"]["evaluation_details"] = {
            "ood_qa": {"exact": 5}
        }
        item["results"][0]["evaluation_details"] = {
            "ood_qa": {"exact": 5}
        }
    journals[1]["results"][0]["evaluation_details"]["ood_qa"]["exact"] = 4

    result = aggregate(
        journals, "validation", ["ood_qa"], max_guard_exact_loss=0,
    )

    assert result["selected"] == "baseline"


def test_required_ood_prose_gate_is_fail_closed():
    journals = [journal(1, 0.1, 0.0), journal(2, 0.1, 0.0)]
    journals[0]["results"][0]["ood_prose_guard_passed"] = True
    journals[1]["results"][0]["ood_prose_guard_passed"] = False

    result = aggregate(
        journals, "validation", ["ood_qa"],
        require_ood_prose_guard=True,
    )

    assert result["selected"] == "baseline"


def test_strict_ood_prose_gate_accepts_only_recomputed_passes():
    journals = [
        add_prose_gate(journal(1, 0.1, 0.0)),
        add_prose_gate(journal(2, 0.1, 0.0)),
    ]

    result = aggregate(
        journals, "validation", ["ood_qa"],
        require_ood_prose_guard=True,
        expected_ood_prose_max_degradation=0.0,
    )

    assert result["selected"] == "candidate"
    assert all(
        seed["ood_prose_gate"]["policy_passed"]
        for seed in result["candidates"][0]["per_seed"]
    )


def test_loose_point02_prose_runs_are_ineligible_under_strict_zero():
    journals = [
        add_prose_gate(
            journal(1, 0.1, 0.0), delta=-0.004,
            lower=-0.007, upper=-0.001, threshold=0.02,
        ),
        add_prose_gate(
            journal(2, 0.1, 0.0), delta=-0.004,
            lower=-0.007, upper=-0.001, threshold=0.02,
        ),
    ]

    result = aggregate(
        journals, "validation", ["ood_qa"],
        require_ood_prose_guard=True,
        expected_ood_prose_max_degradation=0.0,
    )

    assert result["selected"] == "baseline"
    seed = result["candidates"][0]["per_seed"][0]
    assert seed["ood_prose_journal_policy"]["valid"] is False
    assert seed["ood_prose_gate"]["valid"] is False


def test_strict_aggregate_does_not_trust_true_boolean_over_delta():
    journals = [
        add_prose_gate(
            journal(1, 0.1, 0.0), delta=-0.001,
            lower=0.001, upper=0.002, threshold=0.0,
            declared_passed=True,
        ),
        add_prose_gate(journal(2, 0.1, 0.0)),
    ]

    result = aggregate(
        journals, "validation", ["ood_qa"],
        require_ood_prose_guard=True,
    )

    assert result["selected"] == "baseline"
    gate = result["candidates"][0]["per_seed"][0]["ood_prose_gate"]
    assert gate["valid"] is True
    assert gate["recorded_passed"] is True
    assert gate["policy_passed"] is False


def test_strict_aggregate_validates_recorded_pass_against_ci():
    journals = [
        add_prose_gate(
            journal(1, 0.1, 0.0), delta=0.001,
            lower=-0.001, upper=0.002, threshold=0.0,
            declared_passed=True,
        ),
        add_prose_gate(journal(2, 0.1, 0.0)),
    ]
    # Simulate a forged/stale true decision despite a negative lower bound.
    journals[0]["results"][0]["ood_prose_gate"]["passed"] = True

    result = aggregate(
        journals, "validation", ["ood_qa"],
        require_ood_prose_guard=True,
    )

    assert result["selected"] == "baseline"
    gate = result["candidates"][0]["per_seed"][0]["ood_prose_gate"]
    assert gate["valid"] is False
    assert any("disagrees" in issue for issue in gate["issues"])
