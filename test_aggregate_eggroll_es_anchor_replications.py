import copy
import hashlib

import pytest

from aggregate_eggroll_es_anchor_replications import (
    JournalValidationError,
    aggregate_direct_confirmations,
    canonical_sha256,
    summarize_pilot,
    validate_journal,
)


def digest(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def qa_summary(label, mean, exact, nonzero, rows):
    return {
        "path": f"/safe/{label}.json",
        "sha256": digest(label),
        "rows": rows,
        "mean_reward": mean,
        "exact": exact,
        "nonzero": nonzero,
    }


def qa_gate(baseline, candidate):
    deltas = {
        "mean_reward": candidate["mean_reward"] - baseline["mean_reward"],
        "exact": candidate["exact"] - baseline["exact"],
        "nonzero": candidate["nonzero"] - baseline["nonzero"],
    }
    return {
        "schema": "eggroll-es-strict-qa-nondegradation-v1",
        "max_mean_reward_degradation": 0.0,
        "max_exact_loss": 0,
        "max_nonzero_loss": 0,
        "deltas": deltas,
        "passed": (
            deltas["mean_reward"] >= 0.0
            and deltas["exact"] >= 0
            and deltas["nonzero"] >= 0
        ),
    }


def prose_summary(label, mean):
    return {
        "results_path": f"/safe/{label}.json",
        "results_sha256": digest(label),
        "item_count": 16,
        "scored_token_count": 10926,
        "mean_token_logprob": mean,
    }


def prose_gate(baseline, candidate, interval=None):
    delta = candidate["mean_token_logprob"] - baseline["mean_token_logprob"]
    if interval is None:
        interval = [delta, delta]
    return {
        "metric": "mean_token_logprob",
        "higher_is_better": True,
        "baseline": baseline["mean_token_logprob"],
        "final": candidate["mean_token_logprob"],
        "delta": delta,
        "max_degradation": 0.0,
        "paired_document_bootstrap_95_ci": interval,
        "bootstrap": {
            "unit": "normalized_source_url",
            "document_count": 16,
            "samples": 20000,
            "seed": 20260714,
            "percentiles": [0.025, 0.975],
        },
        "passed": interval[0] >= 0.0,
    }


def identity_audit(seed):
    weight_hash = digest("base-weights")
    identity = {
        "schema": "eggroll-es-weight-state-sha256-v2",
        "sha256": weight_hash,
        "parameter_count": 100,
        "total_bytes": 1000,
    }
    check = {
        "schema": "eggroll-es-exact-reference-check-v2",
        "passed": True,
        "reference": identity,
        "current": identity,
    }
    probe = {
        "schema": "eggroll-es-train-only-identity-probe-v2",
        "domain_output_sha256": digest(f"domain-probe-{seed}"),
        "anchor_output_sha256": digest(f"anchor-probe-{seed}"),
        "domain_requests": 64,
        "anchor_requests": 16,
    }
    return {
        "schema": "eggroll-es-alpha-zero-identity-audit-v2",
        "iteration": 0,
        "status": "passed",
        "training_signal": "train_batch_and_train_only_anchor_only",
        "reference_states": [[identity] for _ in range(4)],
        "pre_probe": probe,
        "post_probe": copy.deepcopy(probe),
        "post_reference_checks": [[check] for _ in range(4)],
        "passed": True,
    }


def snapshot(seed, targets, schema_version=2):
    implementation = {
        key: digest(key)
        for key in (
            "driver", "anchor_trainer", "base_trainer", "projection",
            "upstream_trainer", "upstream_worker", "corrected_driver",
            "exact_worker",
        )
    }
    return {
        "schema": f"eggroll-es-anchor-line-search-snapshot-v{schema_version}",
        "train": {
            "rows": 794,
            "arrow_files": [{
                "path": "/safe/s6/train.arrow",
                "sha256": digest("train-arrow"),
            }],
        },
        "evaluations": {
            "validation": {
                "rows": 41,
                "arrow_files": [{
                    "path": "/safe/s6/validation.arrow",
                    "sha256": digest("validation-arrow"),
                }],
            },
            "ood_qa": {
                "rows": 24,
                "arrow_files": [{
                    "path": "/safe/s6/ood_qa.arrow",
                    "sha256": digest("ood-arrow"),
                }],
            },
        },
        "anchor": {
            "path": "/safe/anchor.jsonl",
            "sha256": digest("anchor"),
            "rows": 128,
            "report": {
                "path": "/safe/anchor.report.json",
                "sha256": digest("anchor-report"),
                "schema": "general-prose-anchor-build-v1",
                "protected_artifact_count": 4,
            },
        },
        "fixed_train_batch": {
            "rows": 64,
            "sha256": digest(f"fixed-batch-{seed}"),
        },
        "implementation": implementation,
        "recipe": {
            "model_name": "/safe/models/Qwen3.6-35B-A3B",
            "checkpoint": None,
            "sigma": 0.0003,
            "population_size": 8,
            "batch_size": 64,
            "mini_batch_size": 64,
            "max_tokens": 32,
            "seed": seed,
            "min_anchor_cosine": 0.25,
            "anchor_items_per_step": 16,
            "target_alphas": list(targets),
        },
    }


def make_journal(
    seed,
    validation_delta=0.01,
    ood_mean_delta=0.0,
    ood_exact_delta=0,
    ood_nonzero_delta=0,
    prose_delta=0.001,
    prose_interval=None,
    targets=(0.0, 0.0000125),
    schema_version=2,
):
    baseline_validation = qa_summary(
        f"validation-base-{seed}", 0.08, 2, 13, 41,
    )
    baseline_ood = qa_summary(f"ood-base-{seed}", 0.7, 16, 23, 24)
    baseline_prose = prose_summary(f"prose-base-{seed}", -1.26)
    coefficient_hash = digest(f"coefficients-{seed}")
    states = []
    previous = 0.0
    for index, target in enumerate(targets):
        fraction = 0.0 if target == 0.0 else target / targets[-1]
        validation = qa_summary(
            f"validation-{seed}-{index}",
            baseline_validation["mean_reward"] + validation_delta * fraction,
            2, 13, 41,
        )
        ood = qa_summary(
            f"ood-{seed}-{index}",
            baseline_ood["mean_reward"] + ood_mean_delta * fraction,
            16 + (ood_exact_delta if index else 0),
            23 + (ood_nonzero_delta if index else 0),
            24,
        )
        prose = prose_summary(
            f"prose-{seed}-{index}",
            baseline_prose["mean_token_logprob"] + prose_delta * fraction,
        )
        q_gate = qa_gate(baseline_ood, ood)
        interval = (
            [0.0, 0.0] if index == 0 else prose_interval
        )
        p_gate = prose_gate(baseline_prose, prose, interval)
        states.append({
            "state_index": index,
            "target_alpha": target,
            "alpha_increment": target - previous,
            "eval_iteration": index,
            "coefficient_sha256": coefficient_hash,
            "qa": {"validation": validation, "ood_qa": ood},
            "ood_qa_gate": q_gate,
            "ood_prose": prose,
            "ood_prose_gate": p_gate,
            "strict_guards_passed": q_gate["passed"] and p_gate["passed"],
        })
        previous = target
    journal = {
        "schema": f"eggroll-es-anchor-alpha-line-search-v{schema_version}",
        "status": "complete",
        "policy": {
            "alpha_order": "zero_then_strictly_increasing",
            "branching": False,
            "resume": False,
            "rollback": False,
            "selection_during_execution": False,
            "ood_qa_max_degradation": 0.0,
            "ood_prose_max_degradation": 0.0,
        },
        "targets": list(targets),
        "trainer_configuration": {
            "model_name": "/safe/models/Qwen3.6-35B-A3B",
            "sigma": 0.0003,
            "population_size": 8,
            "batch_size": 64,
            "mini_batch_size": 64,
            "max_tokens": 32,
            "global_seed": seed,
            "min_anchor_cosine": 0.25,
            "anchor_items_per_step": 16,
        },
        "snapshot": snapshot(seed, targets, schema_version),
        "coefficient_plan": {
            "coefficient_sha256": coefficient_hash,
            "journal_path": f"/safe/plan-{seed}.json",
            "projection": {"decision": "project_to_anchor_cone"},
            "seed_count": 8,
            "identity_audit": identity_audit(seed),
        },
        "in_progress": None,
        "states": states,
    }
    journal["content_sha256_before_self_field"] = canonical_sha256(journal)
    return journal


def reseal(journal):
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = canonical_sha256(journal)
    return journal


def five_journals(deltas=None, **kwargs):
    if deltas is None:
        deltas = [0.010, 0.009, 0.008, 0.007, -0.001]
    return [
        make_journal(seed, validation_delta=delta, **kwargs)
        for seed, delta in zip((42, 43, 44, 45, 46), deltas)
    ]


def test_five_direct_confirmations_pass_all_predeclared_rules():
    report = aggregate_direct_confirmations(
        five_journals(), candidate_name="items16-cos025",
    )
    assert report["direct_confirmation"] is True
    assert report["path_dependent_pilot_states_counted"] is False
    assert report["aggregate_validation"]["positive_seed_count"] == 4
    assert report["aggregate_validation"]["median_delta"] > 0
    assert report["aggregate_validation"]["risk_adjusted_score"] > 0
    assert report["eligible"] is True
    assert report["selected"] == "items16-cos025"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("mean", -0.001, "QA gate decision"),
        ("exact", -1, "QA gate decision"),
        ("nonzero", -1, "QA gate decision"),
    ],
)
def test_explicit_ood_qa_mean_exact_nonzero_failures_are_rejected(
    field, value, message,
):
    kwargs = {
        "ood_mean_delta": value if field == "mean" else 0.0,
        "ood_exact_delta": value if field == "exact" else 0,
        "ood_nonzero_delta": value if field == "nonzero" else 0,
    }
    journal = make_journal(42, **kwargs)
    # Simulate a lying producer; aggregation must recompute the decision.
    journal["states"][1]["ood_qa_gate"]["passed"] = True
    journal["states"][1]["strict_guards_passed"] = True
    reseal(journal)
    with pytest.raises(JournalValidationError, match=message):
        validate_journal(journal)


def test_negative_prose_point_delta_is_rejected_even_with_nonnegative_bound():
    journal = make_journal(
        42, prose_delta=-0.001, prose_interval=[0.0, 0.001],
    )
    # The old producer's lower-bound-only Boolean can still say true.
    journal["states"][1]["strict_guards_passed"] = True
    reseal(journal)
    with pytest.raises(JournalValidationError, match="strict guard decision"):
        validate_journal(journal)


def test_negative_prose_lower_bound_makes_candidate_ineligible():
    journals = five_journals(prose_interval=[-0.0001, 0.002])
    report = aggregate_direct_confirmations(
        journals, candidate_name="prose-failure",
    )
    assert report["all_strict_ood_guards_passed"] is False
    assert report["eligible"] is False
    assert report["selected"] == "baseline"


def test_monotonic_pilot_cannot_masquerade_as_direct_confirmation():
    journals = [
        make_journal(seed, targets=(0.0, 0.00000625, 0.0000125))
        for seed in (42, 43, 44, 45, 46)
    ]
    with pytest.raises(JournalValidationError, match="pilot states"):
        aggregate_direct_confirmations(journals, candidate_name="not-direct")
    pilot = summarize_pilot(journals[0])
    assert pilot["classification"] == "exploratory_monotonic_pilot"
    assert pilot["direct_confirmation"] is False
    assert pilot["selection_allowed"] is False


@pytest.mark.parametrize("mutation", ["train", "eval", "recipe", "implementation"])
def test_mixed_family_provenance_is_rejected(mutation):
    journals = five_journals()
    snapshot_value = journals[-1]["snapshot"]
    if mutation == "train":
        snapshot_value["train"]["arrow_files"][0]["sha256"] = digest("other-train")
    elif mutation == "eval":
        snapshot_value["evaluations"]["validation"]["arrow_files"][0]["sha256"] = digest("other-eval")
    elif mutation == "recipe":
        snapshot_value["recipe"]["sigma"] = 0.0005
        journals[-1]["trainer_configuration"]["sigma"] = 0.0005
    else:
        snapshot_value["implementation"]["exact_worker"] = digest("other-worker")
    reseal(journals[-1])
    with pytest.raises(JournalValidationError, match="identity differs"):
        aggregate_direct_confirmations(journals, candidate_name="mixed")


@pytest.mark.parametrize("status,in_progress", [("failed", None), ("complete", {"phase": "eval"})])
def test_failed_or_incomplete_journal_is_rejected(status, in_progress):
    journal = make_journal(42)
    journal["status"] = status
    journal["in_progress"] = in_progress
    reseal(journal)
    with pytest.raises(JournalValidationError, match="incomplete|in-progress"):
        validate_journal(journal)


def test_failed_exact_identity_audit_is_rejected():
    journal = make_journal(42)
    journal["coefficient_plan"]["identity_audit"]["status"] = "failed"
    journal["coefficient_plan"]["identity_audit"]["passed"] = False
    reseal(journal)
    with pytest.raises(JournalValidationError, match="did not pass"):
        validate_journal(journal)


def test_heldout_split_or_path_is_rejected_without_opening_it():
    journal = make_journal(42)
    journal["snapshot"]["evaluations"]["heldout"] = {
        "rows": 1,
        "arrow_files": [{
            "path": "/does/not/exist/sealed.arrow",
            "sha256": digest("sealed"),
        }],
    }
    reseal(journal)
    with pytest.raises(JournalValidationError, match="heldout"):
        validate_journal(journal)


def test_content_hash_tampering_is_rejected():
    journal = make_journal(42)
    journal["states"][1]["qa"]["validation"]["mean_reward"] += 1.0
    with pytest.raises(JournalValidationError, match="content hash"):
        validate_journal(journal)


def test_five_seed_positive_fraction_and_risk_rules_fail_closed():
    too_few_positive = five_journals(
        deltas=[0.010, 0.009, 0.008, -0.001, -0.001],
    )
    report = aggregate_direct_confirmations(
        too_few_positive, candidate_name="three-of-five",
    )
    assert report["aggregate_validation"]["positive_seed_count"] == 3
    assert report["eligible"] is False

    high_variance = five_journals(
        deltas=[0.001, 0.001, 0.001, 0.001, -0.02],
    )
    report = aggregate_direct_confirmations(
        high_variance, candidate_name="high-variance",
    )
    assert report["aggregate_validation"]["positive_seed_count"] == 4
    assert report["aggregate_validation"]["risk_adjusted_score"] <= 0
    assert report["eligible"] is False


def test_corrected_v3_journals_are_supported_as_a_separate_family():
    journals = [
        make_journal(seed, schema_version=3)
        for seed in (42, 43, 44, 45, 46)
    ]
    report = aggregate_direct_confirmations(journals, candidate_name="v3")
    assert report["eligible"] is True
