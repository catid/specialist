#!/usr/bin/env python3

import build_v45_repetition_evidence_v46a as subject


def test_v45_repetition_evidence_is_three_of_three_and_holdout_free_v46a():
    value = subject.build()
    assert value["consistency"] == {
        "repetition_count": 3,
        "sft_v42g_selected_count": 3,
        "sft_v42g_ood_eligible_count": 3,
        "strict_final_gate_pass_count": 3,
        "base_duplicate_equivalence_count": 3,
        "selected_in_all_repetitions": True,
        "ood_eligible_in_all_repetitions": True,
        "strict_gate_passed_in_all_repetitions": True,
        "base_duplicates_exact_in_all_repetitions": True,
    }
    assert value["heldout_or_holdout_opened"] is False
    assert value["protected_semantics_accessed_while_building"] is False
    assert set(value["report_bindings"]) == {"v45a", "v45b", "v45c"}
    assert value["fixed_candidate_identity"][
        "all_tensor_bytes_preserved_exactly"
    ] is True
    assert value["content_sha256_before_self_field"] == subject.core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def test_v45_repetition_evidence_documents_real_numeric_spread_v46a():
    spread = subject.build()["metric_spread"]
    assert spread["shadow_generated_equal_unit_mean_reward"]["range"] > 0
    assert spread["shadow_generated_nonzero_count"]["range"] == 1
    assert spread["ood_qa_generated_exact_count"]["range"] == 0
    assert spread["ood_qa_generated_equal_unit_mean_reward"]["range"] == 0
    assert spread["ood_prose_mean_token_logprob"]["range"] > 0
