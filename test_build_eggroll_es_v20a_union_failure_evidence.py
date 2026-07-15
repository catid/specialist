#!/usr/bin/env python3
"""Tamper tests for compact V20A union failure evidence."""

import copy
import json

import pytest

import build_eggroll_es_v20a_union_failure_evidence as evidence_v20a


def test_v20a_failure_evidence_is_deterministic_and_bound():
    first = evidence_v20a.build_evidence_v20a()
    second = evidence_v20a.build_evidence_v20a()
    assert first == second
    assert first["input_attempt"]["file_sha256"] == (
        "03f5ef75bb6ea6279fd39dc0b313fbd230743ac6d550f27937fee77c82b49c3e"
    )
    assert first["input_attempt"]["content_sha256"] == (
        "ed5a7cb507a6cc25509a0bf7f62779808db1034c7eead45fbcb33a080deb6bac"
    )
    assert first["failure_boundary"]["state"] == "exact_reference"
    assert first["failure_boundary"]["perturbed_state_opened"] is False


def test_v20a_failure_evidence_permanently_rejects_union_and_opens_no_authority():
    value = evidence_v20a.build_evidence_v20a()
    decision = value["decision"]
    assert decision["union_scoring_authorized_for_v20a"] is False
    assert decision["raw_arm_scoring_remains_authoritative"] is True
    assert decision["next_runtime"].startswith("separately_committed_raw_only")
    assert decision["model_update_authorized"] is False
    assert decision["checkpoint_write_authorized"] is False
    assert decision["evaluation_authorized"] is False
    assert decision["dataset_promotion_authorized"] is False
    assert value[
        "contains_scores_outputs_tokens_response_vectors_or_row_content"
    ] is False


def test_v20a_failure_evidence_rejects_resealed_tampering():
    value = evidence_v20a.build_evidence_v20a()
    for path, replacement in (
        (("decision", "union_scoring_authorized_for_v20a"), True),
        (("failure_boundary", "report_persisted"), True),
        (("failure_boundary", "perturbed_state_opened"), True),
    ):
        tampered = copy.deepcopy(value)
        tampered[path[0]][path[1]] = replacement
        tampered["content_sha256_before_self_field"] = (
            evidence_v20a.canonical_sha256({
                key: item for key, item in tampered.items()
                if key != "content_sha256_before_self_field"
            })
        )
        with pytest.raises(RuntimeError, match="evidence changed"):
            evidence_v20a.validate_evidence_v20a(tampered)


def test_v20a_persisted_evidence_matches_rebuild():
    persisted = json.loads(evidence_v20a.OUTPUT_PATH_V20A.read_text())
    assert persisted == evidence_v20a.build_evidence_v20a()
