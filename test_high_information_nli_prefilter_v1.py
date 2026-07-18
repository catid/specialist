from __future__ import annotations

import run_high_information_nli_prefilter_v1 as nli


def packet(*, negative: bool = False) -> dict:
    value = {
        "packet_id": "packet-synthetic",
        "candidate_example_id": "candidate-synthetic",
        "request_id": "request-synthetic",
        "source_group_id": "group-synthetic",
        "generation_mode": (
            "calibrated_hard_negative" if negative else "positive"
        ),
        "evidence_quotes": ["Synthetic evidence supports the bounded fact."],
        "answer": (
            "The source cannot establish the requested claim."
            if negative
            else "The bounded fact is supported."
        ),
    }
    return value


def test_nli_thresholds_are_fail_closed_and_truncation_never_passes():
    verdict, manual, reasons = nli.classify_probabilities(
        {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
        input_truncated=False,
    )
    assert (verdict, manual, reasons) == ("pass", False, [])

    verdict, manual, reasons = nli.classify_probabilities(
        {"entailment": 0.2, "neutral": 0.1, "contradiction": 0.7},
        input_truncated=False,
    )
    assert verdict == "fail"
    assert manual is False

    verdict, manual, reasons = nli.classify_probabilities(
        {"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
        input_truncated=True,
    )
    assert verdict == "uncertain"
    assert manual is True
    assert "nli_input_truncated" in reasons


def test_positive_result_is_content_addressed_but_training_ineligible():
    result = nli.make_result(
        packet(),
        probabilities={"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
        input_token_count=42,
        input_truncated=False,
        run_contract_sha256="a" * 64,
    )
    nli.validate_result(result, packet(), "a" * 64)
    assert result["verdict"] == "pass"
    assert result["semantic_verification_completed"] is False
    assert result["eligible_for_training"] is False
    assert result["premise_sha256"] is not None
    assert result["hypothesis_sha256"] is not None


def test_hard_negative_is_not_misclassified_as_nli_entailment():
    result = nli.make_result(
        packet(negative=True),
        probabilities=None,
        input_token_count=None,
        input_truncated=False,
        run_contract_sha256="a" * 64,
    )
    nli.validate_result(result, packet(negative=True), "a" * 64)
    assert result["verdict"] == "not_applicable"
    assert result["probabilities"] is None
    assert result["semantic_verification_completed"] is False
    assert result["eligible_for_training"] is False
    assert result["manual_review_reasons"] == [
        "hard_negative_requires_absence_or_false_premise_verifier"
    ]


def test_resume_validation_rejects_threshold_or_contract_drift():
    source = packet()
    result = nli.make_result(
        source,
        probabilities={"entailment": 0.8, "neutral": 0.15, "contradiction": 0.05},
        input_token_count=42,
        input_truncated=False,
        run_contract_sha256="a" * 64,
    )
    result["thresholds"]["entailment_minimum"] = 0.5
    result["content_sha256_before_self_field"] = nli._self_address(result)
    import pytest

    with pytest.raises(RuntimeError, match="identity changed"):
        nli.validate_result(result, source, "a" * 64)
    result["thresholds"]["entailment_minimum"] = nli.ENTAILMENT_MINIMUM
    result["content_sha256_before_self_field"] = nli._self_address(result)
    with pytest.raises(RuntimeError, match="identity changed"):
        nli.validate_result(result, source, "b" * 64)
