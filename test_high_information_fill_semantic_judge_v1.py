from __future__ import annotations

import copy
import hashlib
from argparse import Namespace
from pathlib import Path

import pytest

import high_information_semantic_facets_v1 as facets
import run_high_information_fill_nli_prefilter_v1 as fill_nli
import run_high_information_fill_semantic_judge_v1 as fill_judge
import run_high_information_nli_prefilter_v1 as primary_nli
import run_high_information_semantic_judge_shard_v1 as primary_judge


def packet() -> dict:
    answer = "Dan Carabas established it in Berlin in 2024."
    context = "Dan Carabas established Shibari Studio Berlin in Berlin in 2024."
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
        "evidence_quotes": [context],
        "safety_transfer_flags": [],
        "context_text": context,
        "context_text_sha256": hashlib.sha256(context.encode()).hexdigest(),
    }


def pass_result(source: dict) -> dict:
    mappings = []
    for item in facets.deterministic_facet_signals(
        source["question"], source["answer"]
    ):
        span = "2024" if item["kind"] == "temporal" else "Berlin"
        mappings.append({
            "facet_id": item["facet_id"],
            "status": "supported",
            "answer_span": span,
            "evidence_quote_indices": [0],
        })
    gates = {}
    for gate in primary_judge.MODEL_GATES:
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


def test_outputs_are_fill_only_and_do_not_overlap_primary_lanes():
    for shard in range(4):
        paths = set(fill_judge.output_paths(shard).values())
        assert not paths & set(primary_judge.output_paths(shard, smoke=False).values())
        assert not paths & set(fill_nli.output_paths(shard).values())
        assert not paths & set(primary_nli.shard_paths(shard).values())
        assert all(fill_judge.PASS_SLUG in path.name for path in paths)


def test_primary_semantics_and_singleton_grouping_are_exactly_pinned():
    receipt = fill_judge.validate_reused_judge_semantics()
    assert receipt["implementation_receipts"][0]["file_sha256"] == (
        fill_judge.PRIMARY_JUDGE_FILE_SHA256
    )
    assert receipt["protocol"]["candidate_grouping"] == (
        "singleton_sorted_within_request_v1"
    )
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
    groups = fill_judge.groups_by_request(rows)
    assert [[item["candidate_example_id"] for item in group] for group in groups] == [
        ["candidate-1"],
        ["candidate-3"],
        ["candidate-2"],
    ]


@pytest.mark.parametrize(
    ("raw_output", "expected_code"),
    [
        ('{"results":[{"candidate_example_id":"cut-off"}', "invalid_json"),
        ('["valid JSON, wrong top level"]', "top_level_schema_invalid"),
    ],
)
def test_malformed_primary_json_semantics_are_inherited_fail_closed(
    raw_output: str, expected_code: str
):
    source = packet()
    malformed = primary_judge.parse_judge_output(raw_output)
    valid = {"results": [pass_result(source)]}
    record = fill_judge.make_record(
        [source],
        {
            primary_judge.PASS_NAMES[0]: malformed,
            primary_judge.PASS_NAMES[1]: valid,
        },
        {source["candidate_example_id"]: {"verdict": "pass"}},
        run_contract_sha256="a" * 64,
        generation_pass_contract_sha256="b" * 64,
        nli_output_sha256="c" * 64,
        nli_output_receipt_self_sha256="d" * 64,
    )
    fill_judge.validate_record(
        record, [source], "a" * 64, "b" * 64, "c" * 64, "d" * 64
    )
    error = record["pass_validation_errors"][primary_judge.PASS_NAMES[0]][
        source["candidate_example_id"]
    ]
    assert error["code"] == expected_code
    assert record["results"][0]["judge_consensus_passed"] is False
    assert record["results"][0]["manual_review_required"] is True
    assert record["results"][0]["eligible_for_training"] is False
    assert malformed["__parse_failure__"]["raw_text_sha256"] == hashlib.sha256(
        raw_output.encode()
    ).hexdigest()


def test_evidence_gate_rules_are_inherited_without_overrequiring_meta_quotes():
    source = packet()
    value = pass_result(source)
    meta_gates = (
        "unsupported_claim_absence",
        "safety_transfer_preservation",
        "attribution_and_scope_preservation",
        "training_value_and_nontriviality",
    )
    for gate in meta_gates:
        value["gate_evidence"][gate]["evidence_quote_indices"] = []
    normalized = primary_judge.validate_pass_result(value, source)
    assert all(normalized["gate_evidence"][gate]["verdict"] == "pass" for gate in meta_gates)
    value["gate_evidence"]["source_entailment"]["evidence_quote_indices"] = []
    with pytest.raises(RuntimeError, match="gate value is inconsistent"):
        primary_judge.validate_pass_result(value, source)


def _mock_exact_base_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fill_judge.sys, "executable", str(fill_judge.RUNTIME_PYTHON_EXECUTABLE))
    monkeypatch.setattr(fill_judge.sys, "prefix", str(fill_judge.RUNTIME_VIRTUAL_ENV))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(fill_judge.CU13_LIBRARY_PATH))
    monkeypatch.setattr(fill_judge.fill_nli, "validate_runtime_environment", lambda: {
        "python_executable": str(fill_judge.RUNTIME_PYTHON_EXECUTABLE),
        "virtual_environment": str(fill_judge.RUNTIME_VIRTUAL_ENV),
        "ld_library_path": str(fill_judge.CU13_LIBRARY_PATH),
        "transformers_version": fill_judge.TRANSFORMERS_VERSION,
        "torch_version": fill_judge.TORCH_VERSION,
        "dtype": fill_judge.DTYPE,
    })
    versions = {
        "vllm": fill_judge.VLLM_VERSION,
        "mistral-common": fill_judge.MISTRAL_COMMON_VERSION,
    }
    monkeypatch.setattr(fill_judge.metadata, "version", lambda name: versions[name])


def test_exact_es_at_scale_runtime_is_accepted(monkeypatch: pytest.MonkeyPatch):
    _mock_exact_base_runtime(monkeypatch)
    observed = fill_judge.validate_runtime_environment()
    assert observed["python_executable"] == str(fill_judge.RUNTIME_PYTHON_EXECUTABLE)
    assert observed["virtual_environment"] == str(fill_judge.RUNTIME_VIRTUAL_ENV)
    assert observed["ld_library_path"] == str(fill_judge.CU13_LIBRARY_PATH)
    assert observed["vllm_version"] == "0.25.0"
    assert observed["mistral_common_version"] == "1.11.5"


@pytest.mark.parametrize("drift", ["python", "prefix", "ld", "vllm", "mistral"])
def test_any_es_at_scale_runtime_drift_is_rejected(
    drift: str, monkeypatch: pytest.MonkeyPatch
):
    _mock_exact_base_runtime(monkeypatch)
    if drift == "python":
        monkeypatch.setattr(fill_judge.sys, "executable", "/synthetic/wrong/python")
    elif drift == "prefix":
        monkeypatch.setattr(fill_judge.sys, "prefix", "/synthetic/wrong/venv")
    elif drift == "ld":
        monkeypatch.setenv("LD_LIBRARY_PATH", "/synthetic/wrong/cu")
    else:
        versions = {
            "vllm": "0.24.0" if drift == "vllm" else fill_judge.VLLM_VERSION,
            "mistral-common": (
                "1.11.4" if drift == "mistral" else fill_judge.MISTRAL_COMMON_VERSION
            ),
        }
        monkeypatch.setattr(fill_judge.metadata, "version", lambda name: versions[name])
    with pytest.raises(RuntimeError, match="runtime"):
        fill_judge.validate_runtime_environment()


def test_prepare_stops_at_missing_sealed_nli_before_mistral_validation(
    monkeypatch: pytest.MonkeyPatch,
):
    source = packet()
    summary = {"generation_pass_contract": {}}
    monkeypatch.setattr(fill_judge.fill_nli, "load_structural_packets", lambda _: ([source], summary))
    monkeypatch.setattr(
        fill_judge,
        "load_fill_nli_results",
        lambda *_: (_ for _ in ()).throw(RuntimeError("sealed fill NLI receipt missing")),
    )
    called = {"model": False}

    def model_should_not_run(_):
        called["model"] = True
        raise AssertionError("Mistral validation ran before NLI seal")

    monkeypatch.setattr(fill_judge, "validate_model_snapshot", model_should_not_run)
    args = Namespace(
        shard_index=0,
        model_directory=Path("/synthetic/mistral"),
        smoke=False,
        max_tokens=fill_judge.DEFAULT_MAX_TOKENS,
        max_model_len=fill_judge.DEFAULT_MAX_MODEL_LEN,
    )
    with pytest.raises(RuntimeError, match="sealed fill NLI"):
        fill_judge._prepare(args)
    assert called["model"] is False


def test_run_contract_binds_all_lineage_runtime_prompt_schema_and_batch_receipts(
    monkeypatch: pytest.MonkeyPatch,
):
    pass_contract = {
        "generation_pass_id": fill_judge.PASS_ID,
        "content_sha256_before_self_field": "1" * 64,
        "candidate_file_sha256": "2" * 64,
        "generation_report_file_sha256": "3" * 64,
        "generation_report_self_sha256": "4" * 64,
        "runtime_worker_receipts": [{"path": "fill.py", "file_sha256": "5" * 64}],
        "temperature": 0.4,
        "top_p": 0.9,
        "seed_scheme": "synthetic-seed",
        "prompt_spec_sha256": "6" * 64,
    }
    summary = {"generation_pass_contract": pass_contract}
    nli_identity = {
        "generation_pass_contract_sha256": "1" * 64,
        "output_sha256": "7" * 64,
        "receipt_self_sha256": "8" * 64,
        "training_rows_emitted": False,
    }
    runtime = {
        "python_executable": str(fill_judge.RUNTIME_PYTHON_EXECUTABLE),
        "virtual_environment": str(fill_judge.RUNTIME_VIRTUAL_ENV),
        "ld_library_path": str(fill_judge.CU13_LIBRARY_PATH),
        "transformers_version": fill_judge.TRANSFORMERS_VERSION,
        "torch_version": fill_judge.TORCH_VERSION,
        "vllm_version": fill_judge.VLLM_VERSION,
        "mistral_common_version": fill_judge.MISTRAL_COMMON_VERSION,
        "dtype": fill_judge.DTYPE,
    }
    prompts = {
        "schema": "fill-semantic-judge-rendered-prompt-receipt-root-v1",
        "candidate_grouping": fill_judge.CANDIDATE_GROUPING,
        "rendered_prompts": 2,
        "receipts_sha256": "a" * 64,
        "ordered_receipt_fields": ["messages_sha256"],
        "every_rendered_system_and_user_message_bound": True,
        "prompt_text_persisted": False,
    }
    monkeypatch.setattr(fill_judge, "structural_receipt", lambda *_: {"review_file_sha256": "b" * 64})
    args = Namespace(
        smoke=False,
        request_batch_size=16,
        max_model_len=16_384,
        max_tokens=3_072,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
    )
    contract = fill_judge.build_run_contract(
        shard_index=0,
        structural_summary=summary,
        nli_identity=nli_identity,
        model_receipts={"config.json": {"file_sha256": "c" * 64}},
        runtime_environment=runtime,
        paths=fill_judge.output_paths(0),
        args=args,
        rendered_prompts=prompts,
        prompt_statistics={"prompts": 2, "maximum_prompt_plus_output_budget": 4_000},
    )
    assert contract["generation_pass"]["contract_sha256"] == "1" * 64
    assert contract["sealed_fill_nli"] == nli_identity
    assert contract["runtime"] == runtime
    assert contract["judge_protocol"]["candidate_grouping"] == fill_judge.CANDIDATE_GROUPING
    assert contract["judge_protocol"]["rendered_prompt_receipts"] == prompts
    assert contract["guided_schema"]["sha256"] == fill_judge.GUIDED_SCHEMA_SHA256
    assert contract["inference"]["request_batch_size"] == 16
    assert contract["planned_outputs"]["receipt"].endswith(".receipt.json")
    receipts = {item["path"]: item["file_sha256"] for item in contract["implementation_receipts"]}
    assert receipts["run_high_information_semantic_judge_shard_v1.py"] == fill_judge.PRIMARY_JUDGE_FILE_SHA256
    assert receipts["run_high_information_fill_nli_prefilter_v1.py"] == fill_judge.FILL_NLI_FILE_SHA256
    assert receipts["run_high_information_nli_prefilter_v1.py"] == fill_judge.PRIMARY_NLI_FILE_SHA256
    assert contract["training_rows_emitted"] is False
    assert contract["content_sha256_before_self_field"] == fill_judge._self_address(contract)


def test_output_receipt_binds_output_and_report_and_never_authorizes_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(fill_judge.corpus, "ROOT", tmp_path.resolve())
    paths = {name: tmp_path / f"synthetic.{name}" for name in ("partial", "output", "report", "receipt", "telemetry")}
    report = {
        "schema": fill_judge.REPORT_SCHEMA,
        "run_contract": {"content_sha256_before_self_field": "a" * 64},
        "training_rows_emitted": False,
    }
    report["content_sha256_before_self_field"] = fill_judge._self_address(report)
    output_payload = b'{"synthetic":true}\n'
    report_payload = (fill_judge.json.dumps(report, indent=2, sort_keys=True) + "\n").encode()
    receipt = fill_judge.build_output_receipt(
        shard_index=0,
        paths=paths,
        output_payload=output_payload,
        report_payload=report_payload,
        report=report,
    )
    assert receipt["output_sha256"] == hashlib.sha256(output_payload).hexdigest()
    assert receipt["report_file_sha256"] == hashlib.sha256(report_payload).hexdigest()
    assert receipt["semantic_verification_completed"] is False
    assert receipt["training_rows_emitted"] is False
    assert receipt["content_sha256_before_self_field"] == fill_judge._self_address(receipt)
