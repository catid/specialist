import copy
import json
import random

import pytest

from eggroll_es_robust_anchor import (
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    DOCUMENT_LCB_CONFIG_SHA256,
    LOWER_PERCENTILE,
    canonical_sha256,
    document_lcb_config,
    reference_document_summary_identity,
    validate_document_lcb_result,
    score_population_document_lcbs,
)
from train_eggroll_es_specialist import compare_ood_prose


def document(document_id, token_count, token_sum):
    return {
        "document_id": document_id,
        "scored_token_count": token_count,
        "sum_token_logprob": token_sum,
    }


def population(seed, documents):
    return {"seed": seed, "documents": documents}


def population_result(result, seed):
    return next(row for row in result["population"] if row["seed"] == seed)


def test_positive_global_mean_with_harmed_document_has_negative_lcb():
    reference = [document(f"doc-{index}", 1, 0.0) for index in range(4)]
    harmed = [
        document("doc-0", 1, 1.0),
        document("doc-1", 1, 1.0),
        document("doc-2", 1, 1.0),
        document("doc-3", 1, -1.5),
    ]
    result = score_population_document_lcbs(
        reference,
        [population(11, harmed), population(22, reference)],
    )
    row = population_result(result, 11)
    assert row["mean_delta"] == pytest.approx(0.375)
    assert row["bootstrap_lower_confidence_bound"] < 0.0


def test_bootstrap_is_token_weighted_with_uniform_document_sampling():
    reference = [
        document("short", 1, -1.0),
        document("long", 3, -3.0),
    ]
    changed = [
        document("short", 1, -0.5),
        document("long", 3, -3.6),
    ]
    result = score_population_document_lcbs(
        reference,
        [population(7, changed), population(8, reference)],
    )
    row = population_result(result, 7)
    assert row["mean_delta"] == pytest.approx(-0.025)
    assert result["bootstrap_plan"]["document_count"] == 2
    assert result["bootstrap_plan"]["draws_per_sample"] == 2


def test_ordering_is_invariant_and_global_rng_is_unchanged():
    reference = [
        document("a", 2, -2.0),
        document("b", 3, -4.0),
        document("c", 1, -0.5),
    ]
    first_population = [
        population(9, [
            document("a", 2, -1.8),
            document("b", 3, -4.2),
            document("c", 1, -0.4),
        ]),
        population(3, copy.deepcopy(reference)),
    ]
    state = random.getstate()
    first = score_population_document_lcbs(reference, first_population)
    assert random.getstate() == state
    second = score_population_document_lcbs(
        list(reversed(reference)),
        [
            {
                "seed": row["seed"],
                "documents": list(reversed(row["documents"])),
            }
            for row in reversed(first_population)
        ],
    )
    assert random.getstate() == state
    assert first == second


def test_reference_difficulty_cancels_from_robust_scores():
    reference = [
        document("a", 2, -10.0),
        document("b", 3, -6.0),
    ]
    changed = [
        document("a", 2, -9.6),
        document("b", 3, -6.3),
    ]
    shifted_reference = [
        document("a", 2, 10.0),
        document("b", 3, 24.0),
    ]
    shifted_changed = [
        document("a", 2, 10.4),
        document("b", 3, 23.7),
    ]
    first = score_population_document_lcbs(
        reference,
        [population(1, changed), population(2, reference)],
    )
    shifted = score_population_document_lcbs(
        shifted_reference,
        [
            population(1, shifted_changed),
            population(2, shifted_reference),
        ],
    )
    assert [row["score"] for row in first["robust_scores"]] == pytest.approx(
        [row["score"] for row in shifted["robust_scores"]]
    )


def test_zero_spread_fails_closed_to_zero_standardized_scores():
    reference = [document("a", 2, -2.0), document("b", 1, -0.5)]
    result = score_population_document_lcbs(
        reference,
        [population(1, reference), population(2, copy.deepcopy(reference))],
    )
    assert result["standardization"]["zero_spread"] is True
    assert result["standardization"]["standardized_scores"] == [0.0, 0.0]
    assert [row["score"] for row in result["standardized_scores"]] == [
        0.0, 0.0,
    ]


def test_config_and_result_hashes_are_canonical_and_content_sensitive():
    config = document_lcb_config()
    assert canonical_sha256(config) == DOCUMENT_LCB_CONFIG_SHA256
    changed = copy.deepcopy(config)
    changed["percentile"] = 0.05
    assert canonical_sha256(changed) != DOCUMENT_LCB_CONFIG_SHA256
    result = score_population_document_lcbs(
        [document("secret-document-id", 1, -1.0)],
        [
            population(1, [document("secret-document-id", 1, -0.9)]),
            population(2, [document("secret-document-id", 1, -1.0)]),
        ],
    )
    assert result["content_sha256_before_self_field"] == canonical_sha256({
        key: value for key, value in result.items()
        if key != "content_sha256_before_self_field"
    })
    assert "secret-document-id" not in json.dumps(result, sort_keys=True)
    assert result["config"]["bootstrap_samples"] == BOOTSTRAP_SAMPLES
    assert result["config"]["bootstrap_seed"] == BOOTSTRAP_SEED
    assert result["config"]["percentile"] == LOWER_PERCENTILE
    assert validate_document_lcb_result(result) == result
    identity = reference_document_summary_identity([
        document("secret-document-id", 1, -1.0),
    ])
    assert identity["reference_numeric_summary_sha256"] == result[
        "reference"
    ]["numeric_summary_sha256"]
    assert identity["document_manifest_sha256"] == result[
        "document_manifest_sha256"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result["bootstrap_plan"].__setitem__(
            "indices_sha256", "0" * 64,
        ),
        lambda result: result["population"][0].__setitem__(
            "bootstrap_lower_confidence_bound", 123.0,
        ),
        lambda result: result["robust_scores"][0].__setitem__("score", 123.0),
        lambda result: result["reference"]["documents"].reverse(),
    ],
)
def test_persisted_result_must_recompute_exactly(mutation):
    result = score_population_document_lcbs(
        [document("a", 1, -1.0), document("b", 2, -2.0)],
        [
            population(1, [
                document("a", 1, -0.9), document("b", 2, -2.1),
            ]),
            population(2, [
                document("a", 1, -1.1), document("b", 2, -1.9),
            ]),
        ],
    )
    mutation(result)
    result["content_sha256_before_self_field"] = canonical_sha256({
        key: value for key, value in result.items()
        if key != "content_sha256_before_self_field"
    })
    with pytest.raises(ValueError):
        validate_document_lcb_result(result)


def test_synthetic_lower_bound_agrees_with_evaluation_gate_geometry():
    reference = [
        document("doc://a", 1, -1.0),
        document("doc://b", 3, -3.0),
    ]
    changed = [
        document("doc://a", 1, -0.5),
        document("doc://b", 3, -3.6),
    ]
    result = score_population_document_lcbs(
        reference,
        [population(7, changed), population(8, reference)],
    )

    def gate_item(item_id, row):
        return {
            "item_id": item_id,
            "normalized_source_url": row["document_id"],
            "text_sha256": f"text-{item_id}",
            "token_ids_sha256": f"tokens-{item_id}",
            "prompt_token_count": row["scored_token_count"] + 1,
            "scored_token_count": row["scored_token_count"],
            "sum_token_logprob": row["sum_token_logprob"],
        }

    baseline_items = [
        gate_item(str(index), row) for index, row in enumerate(reference)
    ]
    final_items = [
        gate_item(str(index), row) for index, row in enumerate(changed)
    ]
    gate = compare_ood_prose(
        {"mean_token_logprob": -1.0, "items": baseline_items},
        {"mean_token_logprob": -1.025, "items": final_items},
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
    )
    assert population_result(result, 7)[
        "bootstrap_lower_confidence_bound"
    ] == pytest.approx(gate["paired_document_bootstrap_95_ci"][0])


@pytest.mark.parametrize(
    "reference,populations,message",
    [
        (
            [document("a", 1, 0.0), document("a", 1, 0.0)],
            [
                population(1, [document("a", 1, 0.0)]),
                population(2, [document("a", 1, 0.0)]),
            ],
            "duplicate",
        ),
        (
            [document("a", 1, 0.0)],
            [
                population(1, [document("b", 1, 0.0)]),
                population(2, [document("a", 1, 0.0)]),
            ],
            "misaligned",
        ),
        (
            [document("a", 1, 0.0)],
            [
                population(1, [document("a", 2, 0.0)]),
                population(2, [document("a", 1, 0.0)]),
            ],
            "counts drifted",
        ),
        (
            [document("a", 1, float("nan"))],
            [
                population(1, [document("a", 1, 0.0)]),
                population(2, [document("a", 1, 0.0)]),
            ],
            "non-finite",
        ),
        (
            [{"scored_token_count": 1, "sum_token_logprob": 0.0}],
            [
                population(1, [document("a", 1, 0.0)]),
                population(2, [document("a", 1, 0.0)]),
            ],
            "document_id",
        ),
        (
            [document("a", 1, 0.0)],
            [
                {"seed": 1},
                population(2, [document("a", 1, 0.0)]),
            ],
            "sequence",
        ),
    ],
)
def test_invalid_or_misaligned_numeric_summaries_fail_closed(
    reference, populations, message,
):
    with pytest.raises(ValueError, match=message):
        score_population_document_lcbs(reference, populations)
