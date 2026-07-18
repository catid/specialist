from __future__ import annotations

import copy
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest

import high_information_semantic_facets_v1 as facets
import run_high_information_semantic_judge_shard_v1 as judge


def packet() -> dict:
    answer = "Dan Carabas established it in 2024."
    return {
        "packet_id": "packet-synthetic",
        "candidate_example_id": "candidate-synthetic",
        "request_id": "request-synthetic",
        "source_group_id": "group-synthetic",
        "task_subtype": "direct_explanation",
        "generation_mode": "positive",
        "question": "When and where did Dan Carabas establish Shibari Studio Berlin?",
        "answer": answer,
        "assistant_qwen36_token_count": 9,
        "evidence_quotes": [
            "Dan Carabas established Shibari Studio Berlin in Berlin in 2024."
        ],
        "safety_transfer_flags": [],
    }


def test_candidate_grouping_is_singleton_and_deterministic_within_request():
    rows = []
    for request_id, candidate_id in (
        ("request-b", "candidate-2"),
        ("request-a", "candidate-3"),
        ("request-a", "candidate-1"),
    ):
        value = packet()
        value["request_id"] = request_id
        value["candidate_example_id"] = candidate_id
        rows.append(value)
    groups = judge.groups_by_request(rows)
    assert [[item["candidate_example_id"] for item in group] for group in groups] == [
        ["candidate-1"],
        ["candidate-3"],
        ["candidate-2"],
    ]


def test_guided_schema_avoids_runtime_unsupported_unique_items():
    def walk(value):
        if isinstance(value, dict):
            assert "uniqueItems" not in value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(judge.GUIDED_SCHEMA)


def test_run_contract_binds_request_batch_size():
    contract = judge.build_run_contract(
        shard_index=0,
        structural_summary={"report_file_sha256": "1" * 64},
        nli_report={
            "output_sha256": "2" * 64,
            "worker_file_sha256": "3" * 64,
        },
        model_receipts={"synthetic": {"file_sha256": "4" * 64}},
        args=Namespace(
            max_model_len=16_384,
            max_tokens=3_072,
            request_batch_size=16,
        ),
        prompt_statistics={"prompts": 2},
    )
    assert contract["request_batch_size"] == 16
    unsigned = dict(contract)
    declared = unsigned.pop("content_sha256_before_self_field")
    assert declared == judge._self_address(unsigned)


def pass_result(source: dict) -> dict:
    mappings = []
    for item in facets.deterministic_facet_signals(
        source["question"], source["answer"]
    ):
        # Deliberately reproduce the unreliable-judge failure: it maps the
        # missing location to the temporal answer span and claims support.
        span = "2024" if item["kind"] in {"temporal", "location"} else source["answer"]
        mappings.append(
            {
                "facet_id": item["facet_id"],
                "status": "supported",
                "answer_span": span,
                "evidence_quote_indices": [0],
            }
        )
    gates = {}
    for gate in judge.MODEL_GATES:
        not_applicable = gate in {
            "application_correctness",
            "hard_negative_calibration",
        }
        gates[gate] = {
            "verdict": "not_applicable" if not_applicable else "pass",
            "evidence_quote_indices": [] if not_applicable else [0],
        }
    return {
        "candidate_example_id": source["candidate_example_id"],
        "facet_mappings": mappings,
        "gate_evidence": gates,
        "confidence": "high",
        "failure_codes": [],
    }


def test_code_rejects_known_missing_where_even_if_both_judges_claim_support():
    source = packet()
    first = judge.validate_pass_result(pass_result(source), source)
    second = judge.validate_pass_result(pass_result(source), source)
    result = judge.aggregate_candidate(
        source, first, second, {"verdict": "pass"}
    )
    assert result["gate_results"]["question_answer_completeness"]["verdict"] == "fail"
    assert result["judge_consensus_passed"] is False
    assert result["manual_review_required"] is True
    assert "deterministic_missing_facet:facet_01_location" in result[
        "manual_review_reasons"
    ]
    assert result["eligible_for_training"] is False


def test_judge_may_not_add_or_drop_deterministic_facets():
    source = packet()
    value = pass_result(source)
    value["facet_mappings"].append(
        {
            "facet_id": "hallucinated_who",
            "status": "supported",
            "answer_span": "Dan Carabas",
            "evidence_quote_indices": [0],
        }
    )
    with pytest.raises(RuntimeError, match="added or dropped"):
        judge.validate_pass_result(value, source)


def test_gate_disagreement_and_nli_nonpass_are_manual_review_signals():
    source = packet()
    source["answer"] = "Dan Carabas established it in Berlin in 2024."
    first_value = pass_result(source)
    second_value = copy.deepcopy(first_value)
    second_value["gate_evidence"]["source_entailment"]["verdict"] = "fail"
    first = judge.validate_pass_result(first_value, source)
    second = judge.validate_pass_result(second_value, source)
    result = judge.aggregate_candidate(
        source, first, second, {"verdict": "uncertain"}
    )
    assert result["gate_results"]["source_entailment"]["verdict"] == "fail"
    assert "judge_gate_disagreement:source_entailment" in result[
        "manual_review_reasons"
    ]
    assert "nli_nonpass:uncertain" in result["manual_review_reasons"]
    assert result["eligible_for_training"] is False


def test_negative_and_meta_pass_gates_may_have_no_quote_but_entailment_may_not():
    source = packet()
    source["answer"] = "Dan Carabas established it in Berlin in 2024."
    value = pass_result(source)
    for gate in (
        "unsupported_claim_absence",
        "safety_transfer_preservation",
        "attribution_and_scope_preservation",
        "training_value_and_nontriviality",
    ):
        value["gate_evidence"][gate]["evidence_quote_indices"] = []
    normalized = judge.validate_pass_result(value, source)
    assert all(
        normalized["gate_evidence"][gate]["verdict"] == "pass"
        for gate in (
            "unsupported_claim_absence",
            "safety_transfer_preservation",
            "attribution_and_scope_preservation",
            "training_value_and_nontriviality",
        )
    )

    value["gate_evidence"]["source_entailment"]["evidence_quote_indices"] = []
    with pytest.raises(RuntimeError, match="gate value is inconsistent"):
        judge.validate_pass_result(value, source)


def test_failure_codes_force_rejection_and_are_preserved():
    source = packet()
    source["answer"] = "Dan Carabas established it in Berlin in 2024."
    first_value = pass_result(source)
    second_value = copy.deepcopy(first_value)
    first_value["failure_codes"] = ["unsupported_claim"]
    first = judge.validate_pass_result(first_value, source)
    second = judge.validate_pass_result(second_value, source)
    result = judge.aggregate_candidate(source, first, second, {"verdict": "pass"})
    assert result["judge_consensus_passed"] is False
    assert result["failure_codes_by_pass"][judge.PASS_NAMES[0]] == [
        "unsupported_claim"
    ]
    assert "judge_reported_failure_code" in result["manual_review_reasons"]
    assert "judge_failure_code_disagreement" in result["manual_review_reasons"]


def test_request_record_persists_both_normalized_audit_passes():
    source = packet()
    source["answer"] = "Dan Carabas established it in Berlin in 2024."
    output = {"results": [pass_result(source)]}
    record = judge.make_record(
        [source],
        {name: copy.deepcopy(output) for name in judge.PASS_NAMES},
        {source["candidate_example_id"]: {"verdict": "pass"}},
        run_contract_sha256="a" * 64,
    )
    judge.validate_record(record, [source], "a" * 64)
    assert set(record["normalized_pass_outputs"]) == set(judge.PASS_NAMES)
    for value in record["normalized_pass_outputs"].values():
        decision = value[source["candidate_example_id"]]
        assert decision["facet_mappings"]
        assert decision["gate_evidence"]
        assert "failure_codes" in decision


def test_internally_inconsistent_judge_row_is_quarantined_not_shard_fatal():
    source = packet()
    source["answer"] = "Dan Carabas established it in Berlin in 2024."
    valid = pass_result(source)
    invalid = copy.deepcopy(valid)
    first_facet = invalid["facet_mappings"][0]
    first_facet["status"] = "missing"
    # Deliberately retain the claimed span/evidence, reproducing the live
    # Mistral inconsistency caught by the GPU smoke.
    assert first_facet["answer_span"] is not None
    assert first_facet["evidence_quote_indices"]
    outputs = {
        judge.PASS_NAMES[0]: {"results": [invalid]},
        judge.PASS_NAMES[1]: {"results": [valid]},
    }
    record = judge.make_record(
        [source],
        outputs,
        {source["candidate_example_id"]: {"verdict": "pass"}},
        run_contract_sha256="b" * 64,
    )
    judge.validate_record(record, [source], "b" * 64)
    error = record["pass_validation_errors"][judge.PASS_NAMES[0]][
        source["candidate_example_id"]
    ]
    assert error["code"] == "candidate_consistency_invalid"
    assert "missing facet improperly claims" in error["detail"]
    assert source["candidate_example_id"] in record[
        "invalid_raw_pass_outputs"
    ][judge.PASS_NAMES[0]]
    result = record["results"][0]
    assert result["judge_consensus_passed"] is False
    assert result["manual_review_required"] is True
    assert any(
        reason.startswith("judge_pass_validation_error:")
        for reason in result["manual_review_reasons"]
    )


@pytest.mark.parametrize(
    ("raw_output", "expected_code"),
    [
        ('{"results":[{"candidate_example_id":"cut-off"}', "invalid_json"),
        ('["valid JSON, wrong top level"]', "top_level_schema_invalid"),
    ],
)
def test_malformed_entire_pass_is_quarantined_not_shard_fatal(
    raw_output: str, expected_code: str
):
    source = packet()
    source["answer"] = "Dan Carabas established it in Berlin in 2024."
    invalid_output = judge.parse_judge_output(raw_output)
    assert invalid_output["results"] is None
    assert invalid_output["__parse_failure__"]["code"] == expected_code
    assert invalid_output["__parse_failure__"]["raw_text_sha256"] == hashlib.sha256(
        raw_output.encode("utf-8")
    ).hexdigest()
    valid_output = {"results": [pass_result(source)]}
    record = judge.make_record(
        [source],
        {
            judge.PASS_NAMES[0]: invalid_output,
            judge.PASS_NAMES[1]: valid_output,
        },
        {source["candidate_example_id"]: {"verdict": "pass"}},
        run_contract_sha256="c" * 64,
    )
    judge.validate_record(record, [source], "c" * 64)
    error = record["pass_validation_errors"][judge.PASS_NAMES[0]][
        source["candidate_example_id"]
    ]
    assert error == {
        "code": expected_code,
        "detail": invalid_output["__parse_failure__"]["detail"],
    }
    assert record["results"][0]["judge_consensus_passed"] is False
    assert record["results"][0]["manual_review_required"] is True
    assert any(
        reason == (
            "judge_pass_validation_error:"
            f"{judge.PASS_NAMES[0]}:{expected_code}"
        )
        for reason in record["results"][0]["manual_review_reasons"]
    )


def test_model_snapshot_rejects_matching_blob_name_with_tampered_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    blob = blobs / "expected-blob-id"
    blob.write_bytes(b"tampered-runtime-bytes")
    (tmp_path / "config.json").symlink_to(blob)
    monkeypatch.setattr(judge, "MODEL_DIRECTORY", tmp_path.resolve())
    monkeypatch.setattr(
        judge, "MODEL_BLOB_RECEIPTS", {"config.json": "expected-blob-id"}
    )
    monkeypatch.setattr(
        judge,
        "RUNTIME_MODEL_FILE_SHA256",
        {"config.json": hashlib.sha256(b"expected-runtime-bytes").hexdigest()},
    )
    with pytest.raises(RuntimeError, match="runtime file content changed"):
        judge.validate_model_snapshot(tmp_path)
