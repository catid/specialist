#!/usr/bin/env python3
"""Train-only aggregate tests for the V18A Qwen token audit."""

import json

import audit_eggroll_es_token_lengths_v18a as audit_v18a


def test_v18a_token_audit_recomputes_exact_v298_aggregates():
    observed = audit_v18a.build_audit_v18a()
    persisted = json.loads(audit_v18a.OUTPUT_PATH_V18A.read_text())
    assert observed == persisted
    assert audit_v18a.file_sha256(audit_v18a.OUTPUT_PATH_V18A) == (
        "df1d3810e988c3ece4ef921643ffe226fa7bb7f2f91edf2895865afe78c7ee6f"
    )
    candidate = observed["sources"]["candidate_v298"]
    assert candidate[
        "combined_prompt_answer_token_quantiles_p50_p90_p95_p99_max"
    ] == [67, 91, 104, 129, 144]
    assert candidate["answer_token_quantiles_p50_p90_p95_p99_max"] == [
        19, 42, 52, 74, 86,
    ]


def test_v18a_token_audit_has_zero_boundary_or_cap_failures_and_no_eval():
    value = json.loads(audit_v18a.OUTPUT_PATH_V18A.read_text())
    assert value["quantile_method"] == "higher"
    assert set(value["sources"]) == {"candidate_v298", "production"}
    for source in value["sources"].values():
        assert source["tokenizer_boundary_mismatch_count"] == 0
        assert source["over_frozen_1024_total_token_cap_count"] == 0
    assert value["firewall"][
        "heldout_validation_ood_eval_or_benchmark_content_opened"
    ] is False
    assert value["firewall"]["gpu_launched"] is False
