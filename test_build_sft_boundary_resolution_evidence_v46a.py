#!/usr/bin/env python3

import build_sft_boundary_resolution_evidence_v46a as subject


def test_boundary_evidence_incorporates_prior_replication_and_v45d_v46a():
    value = subject.build()
    assert value["current_sft_control"]["arm"] == "sft_v42i"
    assert value["v45d_boundary_report"]["strict_final_gate_passed"] is True
    assert value["replication_scope"]["v42g_has_three_repetitions"] is True
    assert value["replication_scope"]["v42i_repeated_stability_claimed"] is False
    assert value["replication_scope"][
        "v43i_ood_first_resolution_required_before_holdout_launch"
    ] is True
    assert value["heldout_or_holdout_opened"] is False
    assert value["content_sha256_before_self_field"] == subject.core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })
