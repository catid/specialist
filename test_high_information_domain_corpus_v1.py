from __future__ import annotations

import json
import re
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

import build_high_information_domain_corpus_v1 as corpus
import run_high_information_candidate_shard_v1 as generator
import verify_high_information_candidates_v1 as verifier
import verify_high_information_semantic_decisions_v1 as semantic


class _Encoding:
    def __init__(self, count: int):
        self.ids = list(range(count))


class FakeTokenizer:
    _roles = {"system": 20, "user": 21, "assistant": 22, "tool": 23}

    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        if text == "\n":
            return [3]
        return [1000 + index for index, _ in enumerate(re.findall(r"\S+", text))]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|im_start|>": 1, "<|im_end|>": 2}[token]

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        assert skip_special_tokens is False
        inverse = {value: key for key, value in self._roles.items()}
        return "".join(inverse.get(value, "") for value in ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ):
        assert enable_thinking is False
        if tokenize is False:
            return "official rendered prompt"
        assert tokenize is True
        ids = []
        for message in messages:
            content_ids = [
                1000 + (sum(word.encode("utf-8")) % 100_000)
                for word in message["content"].split()
            ]
            ids.extend(
                [1, self._roles[message["role"]], 3, *content_ids, 2, 3]
            )
        if add_generation_prompt:
            ids.extend([1, self._roles["assistant"], 3])
        return ids


def _context(index: int, tokens: int = 300) -> dict:
    text = " ".join(f"fact{index}_{part}" for part in range(tokens)) + "\n"
    return {
        "schema": "high-information-source-context-v1",
        "context_id": f"context-{index}",
        "source_group_id": f"group-{index}",
        "qwen36_token_count": tokens,
        "text": text,
        "text_sha256": corpus.sha256_bytes(text.encode()),
        "rights_basis": {"status": "synthetic"},
        "safety_transfer_flags": [f"scope-{index}"],
        "lineage": {"synthetic": True},
    }


def test_source_sanitization_removes_urls_domains_and_provenance_lines():
    source = (
        "# Topic\n\n"
        "Source URL: https://example.com/page\n\n"
        "Read [the descriptive study](https://example.org/study) for the fact.\n"
        "A second pointer is https://sample.net/x.\n"
    )
    cleaned = corpus.sanitize_source_text(source)
    assert "the descriptive study" in cleaned
    assert "Source URL" not in cleaned
    assert not corpus.URL_RE.search(cleaned)
    assert not corpus.DOMAIN_RE.search(cleaned)


@pytest.mark.parametrize(
    ("kind", "question", "answer", "expected"),
    [
        ("qa_resource_direct", "A factual question?", "A fact.", "resource_lookup_kind"),
        ("qa_verified", "What is the canonical URL?", "No URL here.", "url_or_web_lookup_surface"),
        ("qa_verified", "What happened?", "<think>private</think> Fact.", "hidden_reasoning_or_protocol_surface"),
        ("qa_verified", "What happened?", "A supported fact.", None),
    ],
)
def test_seed_qa_filter_excludes_url_memorization_and_hidden_reasoning(
    kind: str, question: str, answer: str, expected: str | None
):
    assert corpus._seed_qa_exclusion_reason(
        {"kind": kind, "question": question, "answer": answer}
    ) == expected


def test_subset_fill_hits_exact_target_without_padding():
    candidates = [
        {"candidate_id": f"candidate-{index}", "qwen36_token_count": value}
        for index, value in enumerate((7, 11, 13, 17, 23))
    ]
    selected, achieved = corpus._subset_fill(candidates, 41)
    assert achieved == 41
    assert sum(item["qwen36_token_count"] for item in selected) == 41
    selected, achieved = corpus._subset_fill(candidates, 2)
    assert selected == []
    assert achieved == 0


def test_source_balanced_prefix_covers_every_group_before_filling():
    candidates = [
        {
            "candidate_id": f"candidate-{group}-{index}",
            "source_group_id": f"group-{group}",
            "qwen36_token_count": count,
        }
        for group, counts in enumerate(((9, 40), (11, 30), (13, 20)))
        for index, count in enumerate(counts)
    ]
    selected, remaining = corpus._source_balanced_prefix(candidates, 100)
    assert {item["source_group_id"] for item in selected} == {
        "group-0",
        "group-1",
        "group-2",
    }
    assert sum(item["qwen36_token_count"] for item in selected) <= 100
    assert len(selected) + len(remaining) == len(candidates)


def test_check_mode_is_read_only_and_checks_every_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    artifact = tmp_path / "artifact.jsonl"
    manifest = tmp_path / "manifest.json"
    value = {"schema": "synthetic"}
    artifact.write_bytes(b'{"synthetic":true}\n')
    manifest.write_bytes(corpus.render_manifest(value))
    monkeypatch.setattr(corpus, "MANIFEST", manifest)
    monkeypatch.setattr(
        corpus,
        "construct",
        lambda: (value, {artifact: b'{"synthetic":true}\n'}),
    )
    monkeypatch.setattr(
        corpus,
        "atomic_write",
        lambda *_: pytest.fail("check mode attempted a write"),
    )
    assert corpus.build(check=True) == value
    artifact.write_bytes(b"stale\n")
    with pytest.raises(RuntimeError, match="artifacts are stale"):
        corpus.build(check=True)


def test_budget_allocator_is_exact_deterministic_and_capacity_bounded():
    contexts = [_context(index, tokens=200 + index * 10) for index in range(12)]
    first = corpus._allocate_budget(
        contexts, target=2400, family="closed_book_application"
    )
    second = corpus._allocate_budget(
        list(reversed(contexts)), target=2400, family="closed_book_application"
    )
    assert first == second
    assert sum(first.values()) == 2400
    assert all(value >= 64 for value in first.values())
    with pytest.raises(RuntimeError, match="exceeds non-padding source capacity"):
        corpus._allocate_budget(
            contexts, target=100_000, family="grounded_synthesis"
        )


def test_generation_requests_cover_sources_hit_targets_and_balance_four_shards(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(corpus, "CLOSED_BOOK_TARGET_ASSISTANT_TOKENS", 2400)
    monkeypatch.setattr(corpus, "GROUNDED_TARGET_ASSISTANT_TOKENS", 1200)
    monkeypatch.setattr(corpus, "HARD_NEGATIVE_NUMERATOR", 100)
    monkeypatch.setattr(corpus, "HARD_NEGATIVE_DENOMINATOR", 1000)
    contexts = [_context(index, tokens=300) for index in range(12)]
    spec = corpus.build_prompt_spec()
    requests = corpus.build_generation_requests(
        contexts, seed_qa_assistant_tokens=120, prompt_spec=spec
    )
    assert {row["source_group_id"] for row in requests} == {
        row["source_group_id"] for row in contexts
    }
    assert sum(row["target_verified_assistant_tokens"] for row in requests) == 3480
    assert sum(
        row["target_verified_assistant_tokens"]
        for row in requests
        if row["generation_mode"] == "calibrated_hard_negative"
    ) == 360
    assert min(
        row["target_verified_assistant_tokens"]
        for row in requests
        if row["generation_mode"] == "calibrated_hard_negative"
    ) >= 48
    shards, loads = corpus.shard_requests(requests)
    assert sum(map(len, shards)) == len(requests)
    assert max(loads) - min(loads) <= max(
        row["target_verified_assistant_tokens"] for row in requests
    )
    again, again_loads = corpus.shard_requests(list(reversed(requests)))
    assert shards == again
    assert loads == again_loads


def _request(*, negative: bool = False) -> dict:
    return {
        "request_id": "request-1",
        "source_context_id": "context-1",
        "source_group_id": "group-1",
        "task_family": "closed_book_application",
        "task_subtype": "application_scenario",
        "generation_mode": (
            "calibrated_hard_negative" if negative else "positive"
        ),
        "target_verified_assistant_tokens": 64,
        "candidate_count": 2,
        "gpu_shard": 0,
    }


def _example(**updates) -> dict:
    value = {
        "example_type": "application_scenario",
        "question": "What does the source establish about the synthetic fact?",
        "answer": "It establishes the synthetic fact with an explicit limit.",
        "evidence_quotes": ["synthetic evidence phrase"],
        "negative_type": None,
    }
    value.update(updates)
    return value


def test_candidate_verifier_keeps_structural_pass_semantically_pending():
    context = _context(1, 50)
    context["text"] += "synthetic evidence phrase\n"
    request = _request()
    verified, errors = verifier.verify_example(
        _example(), request=request, context=context, tokenizer=FakeTokenizer()
    )
    assert errors == []
    assert verified is not None
    assert verified["deterministic_structure_status"] == "passed"
    assert verified["semantic_verification_status"] == "pending"
    assert verified["eligible_for_training"] is False
    assert verified["rights_basis"] == context["rights_basis"]
    assert verified["safety_transfer_flags"] == context["safety_transfer_flags"]
    assert verified["assistant_qwen36_token_count"] > 0
    assert verified["assistant_token_mask_method"] == corpus.ASSISTANT_MASK_METHOD


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"question": "Which website is https://example.com?"}, "url_or_domain_surface"),
        ({"answer": "<think>secret</think> A fact."}, "hidden_reasoning_or_protocol_surface"),
        ({"evidence_quotes": ["not in context"]}, "evidence_quote_not_in_source_context"),
        ({"example_type": "direct_explanation"}, "example_type_does_not_match_request"),
    ],
)
def test_candidate_verifier_rejects_url_reasoning_and_evidence_leakage(
    updates: dict, error: str
):
    context = _context(1, 50)
    context["text"] += "synthetic evidence phrase\n"
    verified, errors = verifier.verify_example(
        _example(**updates),
        request=_request(),
        context=context,
        tokenizer=FakeTokenizer(),
    )
    assert verified is None
    assert error in errors


def test_hard_negative_requires_valid_type_and_calibrated_answer():
    context = _context(1, 50)
    context["text"] += "synthetic evidence phrase\n"
    request = _request(negative=True)
    bad, errors = verifier.verify_example(
        _example(
            example_type="calibrated_hard_negative",
            negative_type="false_premise",
        ),
        request=request,
        context=context,
        tokenizer=FakeTokenizer(),
    )
    assert bad is None
    assert "hard_negative_answer_is_not_calibrated" in errors
    good, errors = verifier.verify_example(
        _example(
            example_type="calibrated_hard_negative",
            answer="The source does not establish that claim; evidence is insufficient.",
            negative_type="false_premise",
        ),
        request=request,
        context=context,
        tokenizer=FakeTokenizer(),
    )
    assert errors == []
    assert good is not None
    assert good["eligible_for_training"] is False


@pytest.mark.parametrize(
    "answer",
    [
        "No information in the context establishes that claim.",
        "The claim cannot be confirmed from the supplied evidence.",
        "The source lacks evidence for that precision.",
        "That conclusion is not supported by the context.",
    ],
)
def test_hard_negative_calibration_accepts_clear_non_formulaic_language(answer: str):
    assert verifier.CALIBRATION_RE.search(answer)


def test_structural_review_rows_are_content_addressed():
    row = verifier._address_review_row(
        {"request_id": "synthetic", "status": "rejected", "errors": []}
    )
    unsigned = dict(row)
    declared = unsigned.pop("content_sha256_before_self_field")
    assert declared == corpus.canonical_sha256(unsigned)


def test_candidate_record_rejects_duplicate_questions():
    context = _context(1, 50)
    context["text"] += "synthetic evidence phrase\n"
    request = _request()
    result = verifier.verify_candidate_record(
        {"request_id": "request-1", "examples": [_example(), _example()]},
        requests={"request-1": request},
        contexts={"context-1": context},
        tokenizer=FakeTokenizer(),
    )
    assert len(result["structurally_valid_examples"]) == 1
    assert result["errors"] == [
        {"example_index": 1, "errors": ["duplicate_question_within_candidate"]}
    ]
    assert result["training_rows_emitted"] is False


def test_generation_worker_parses_and_content_addresses_structured_output():
    request = _request()
    request["candidate_count"] = 4
    examples = [_example() for _ in range(4)]
    text = "```json\n" + json.dumps({"examples": examples}) + "\n```"
    record = generator.make_candidate_record(
        request,
        text,
        finish_reason="stop",
        generated_token_count=100,
    )
    generator.validate_candidate_record(record, request)
    assert record["parse_status"] == "parsed_unverified"
    assert record["semantic_verification_completed"] is False
    assert record["eligible_for_training"] is False
    assert verifier.validate_generated_candidate_envelope(record, request) == []


def test_generation_prompt_uses_safe_context_without_lineage_paths():
    request = _request()
    context = _context(1, 50)
    context["lineage"] = {"forbidden_mixed_path": "/do/not/open/source.md"}
    messages = generator.generation_messages(request, context)
    rendered = "\n".join(message["content"] for message in messages)
    assert context["text"] in rendered
    assert "/do/not/open/source.md" not in rendered
    assert "exactly four" in rendered
    assert "URL" in rendered


def test_generated_candidate_envelope_tamper_fails_closed():
    request = _request()
    request["candidate_count"] = 4
    record = generator.make_candidate_record(
        request,
        json.dumps({"examples": [_example() for _ in range(4)]}),
        finish_reason="stop",
        generated_token_count=50,
    )
    record["source_group_id"] = "tampered"
    errors = verifier.validate_generated_candidate_envelope(record, request)
    assert "generated_candidate_source_group_changed" in errors
    assert "generated_candidate_content_address_changed" in errors


def test_generation_runner_import_does_not_import_vllm_or_torch():
    command = (
        "import json,sys; import run_high_information_candidate_shard_v1; "
        "print(json.dumps({'torch':'torch' in sys.modules,'vllm':'vllm' in sys.modules}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=corpus.ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {"torch": False, "vllm": False}


def test_generation_engine_and_sampling_policy_are_exact():
    args = Namespace(
        model_directory=generator.MODEL_DIRECTORY,
        gpu_memory_utilization=0.9,
        max_model_len=16_384,
        enforce_eager=False,
    )
    assert generator.engine_kwargs(args) == {
        "model": str(generator.MODEL_DIRECTORY),
        "tokenizer": str(generator.MODEL_DIRECTORY),
        "runner": "generate",
        "trust_remote_code": False,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "seed": 0,
        "gpu_memory_utilization": 0.9,
        "max_model_len": 16_384,
        "enable_prefix_caching": False,
        "max_num_seqs": 64,
        "enforce_eager": False,
        "generation_config": "vllm",
    }
    request = _request()
    assert generator.sampling_kwargs(request) == {
        "n": 1,
        "temperature": 0.3,
        "top_p": 0.9,
        "seed": generator.request_seed("request-1"),
        "max_tokens": 768,
    }
    assert generator.VLLM_VERSION == "0.25.0"
    assert generator.MODEL_FILE_SHA256["chat_template.jinja"] == (
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    )


def test_generation_prompt_uses_official_template_contract():
    prompt, count = generator.render_prompt(
        FakeTokenizer(), _request(), _context(1, 20)
    )
    assert prompt == "official rendered prompt"
    assert count == 3


def test_generation_requests_reject_hidden_verifier_targets():
    assert not generator._contains_key(_request(), generator.HIDDEN_VERIFIER_KEYS)
    poisoned = {**_request(), "nested": {"reference_answer": "hidden"}}
    assert generator._contains_key(poisoned, generator.HIDDEN_VERIFIER_KEYS)


def test_partial_resume_is_ordered_and_content_addressed(tmp_path: Path):
    requests = [
        {**_request(), "request_id": "request-a"},
        {**_request(), "request_id": "request-b"},
    ]
    rows = [
        generator.make_candidate_record(
            request,
            json.dumps({"examples": [_example() for _ in range(4)]}),
            finish_reason="stop",
            generated_token_count=50,
        )
        for request in requests
    ]
    path = tmp_path / "partial.jsonl"
    path.write_bytes(corpus.jsonl_payload(rows))
    assert generator.load_partial(path, requests) == rows
    path.write_bytes(corpus.jsonl_payload(list(reversed(rows))))
    with pytest.raises(RuntimeError, match="identity changed"):
        generator.load_partial(path, requests)
    rows[0]["generated_token_count"] = 51
    path.write_bytes(corpus.jsonl_payload(rows))
    with pytest.raises(RuntimeError, match="identity changed"):
        generator.load_partial(path, requests)


def test_complete_output_report_pair_fails_closed_and_reopens_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(generator.corpus, "relative", lambda path: path.name)
    request = _request()
    request["candidate_count"] = 4
    row = generator.make_candidate_record(
        request,
        json.dumps({"examples": [_example() for _ in range(4)]}),
        finish_reason="stop",
        generated_token_count=50,
    )
    paths = {
        "output": tmp_path / "candidates.jsonl",
        "report": tmp_path / "candidates.report.json",
    }
    output_payload = corpus.jsonl_payload([row])
    paths["output"].write_bytes(output_payload)
    with pytest.raises(RuntimeError, match="pair is incomplete"):
        generator.load_complete(paths, [request])
    report = {
        "schema": "high-information-candidate-shard-report-v1",
        "status": "complete_unverified",
        "gpu_shard": 0,
        "requests": 1,
        "candidate_records": 1,
        "output": paths["output"].name,
        "output_sha256": corpus.sha256_bytes(output_payload),
        "worker_file_sha256": corpus.file_sha256(Path(generator.__file__)),
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    report["content_sha256_before_self_field"] = corpus.canonical_sha256(report)
    paths["report"].write_text(json.dumps(report), encoding="utf-8")
    assert generator.load_complete(paths, [request]) == [row]
    report["requests"] = 2
    paths["report"].write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(RuntimeError, match="content address changed"):
        generator.load_complete(paths, [request])


def _semantic_fixture(*, negative: bool = False):
    context = _context(1, 50)
    context["text"] += "synthetic evidence phrase\n"
    context["text_sha256"] = corpus.sha256_bytes(context["text"].encode())
    request = _request(negative=negative)
    example = _example(
        example_type=(
            "calibrated_hard_negative" if negative else "application_scenario"
        ),
        answer=(
            "The source does not establish that claim; evidence is insufficient."
            if negative else "It establishes the synthetic fact with an explicit limit."
        ),
        negative_type="false_premise" if negative else None,
    )
    candidate, errors = verifier.verify_example(
        example,
        request=request,
        context=context,
        tokenizer=FakeTokenizer(),
    )
    assert errors == [] and candidate is not None
    packet = semantic.build_packet(candidate, request, context)
    contract = corpus.build_semantic_verifier_contract()
    authority = {
        "schema": semantic.AUTHORITY_SCHEMA,
        "authority_kind": "pinned_independent_entailment_pipeline",
        "verifier_identity_sha256": "a" * 64,
        "generator_model_identity_sha256": corpus.GENERATOR_MODEL_IDENTITY_SHA256,
        "independent_of_generator": True,
        "implementation_receipts": [
            {"path": "synthetic_verifier.py", "file_sha256": "b" * 64}
        ],
        "prompt_or_protocol_sha256": "c" * 64,
    }
    authority["content_sha256_before_self_field"] = semantic._self_address(authority)
    contract["pinned_verifier_authority_sha256"] = authority[
        "content_sha256_before_self_field"
    ]
    expected = semantic.expected_gate_verdicts(packet, contract)
    gate_results = {
        gate: {
            "verdict": verdict,
            "method": contract["required_gates"][gate]["method"],
            "evidence_sha256s": (
                packet["evidence_quote_sha256s"] if verdict == "pass" else []
            ),
        }
        for gate, verdict in expected.items()
    }
    decision = {
        "schema": semantic.DECISION_SCHEMA,
        "packet_id": packet["packet_id"],
        "candidate_example_id": packet["candidate_example_id"],
        "verifier_authority_sha256": authority[
            "content_sha256_before_self_field"
        ],
        "gate_results": gate_results,
        "evidence_quote_sha256s": packet["evidence_quote_sha256s"],
    }
    decision["content_sha256_before_self_field"] = semantic._self_address(decision)
    return contract, authority, packet, decision


def test_semantic_contract_rejects_structural_only_and_unpinned_authority():
    contract, authority, packet, decision = _semantic_fixture()
    contract["pinned_verifier_authority_sha256"] = None
    with pytest.raises(RuntimeError, match="no independent semantic verifier"):
        semantic.validate_decision(packet, decision, authority, contract)
    contract["pinned_verifier_authority_sha256"] = authority[
        "content_sha256_before_self_field"
    ]
    with pytest.raises(RuntimeError, match="decision schema"):
        semantic.validate_decision(packet, None, authority, contract)


def test_semantic_decision_requires_every_objective_gate():
    contract, authority, packet, decision = _semantic_fixture()
    result = semantic.validate_decision(packet, decision, authority, contract)
    assert result["semantic_verification_status"] == "passed"
    assert result["eligible_for_training"] is True
    decision["gate_results"]["source_entailment"]["verdict"] = "fail"
    decision["content_sha256_before_self_field"] = semantic._self_address(decision)
    result = semantic.validate_decision(packet, decision, authority, contract)
    assert result["semantic_verification_status"] == "rejected"
    assert result["eligible_for_training"] is False
    assert result["failed_gates"] == ["source_entailment"]


def test_semantic_decision_rejects_partial_question_answer_completeness():
    contract, authority, packet, decision = _semantic_fixture()
    assert "question_answer_completeness" in decision["gate_results"]
    decision["gate_results"]["question_answer_completeness"]["verdict"] = "fail"
    decision["content_sha256_before_self_field"] = semantic._self_address(decision)
    result = semantic.validate_decision(packet, decision, authority, contract)
    assert result["eligible_for_training"] is False
    assert result["failed_gates"] == ["question_answer_completeness"]


def test_semantic_contract_separates_global_dedup_and_manual_review_gates():
    contract = corpus.build_semantic_verifier_contract()
    selection = contract["post_verification_selection_gates"]
    assert selection["exact_and_normalized_duplicate_control"][
        "per_row_judge_may_not_self_certify"
    ] is True
    assert selection["near_duplicate_and_information_preference"][
        "isolated_trivia_deprioritized"
    ] is True
    assert selection["manual_review"]["all_uncertain_rows_required"] is True
    assert selection["manual_review"][
        "deterministic_stratified_high_risk_sample_required"
    ] is True
    assert selection["manual_review"]["unresolved_rows_training_eligible"] is False


def test_semantic_negative_requires_calibration_and_application_is_conditional():
    contract, authority, packet, decision = _semantic_fixture(negative=True)
    expected = semantic.expected_gate_verdicts(packet, contract)
    assert expected["hard_negative_calibration"] == "pass"
    assert expected["application_correctness"] == "pass"
    decision["gate_results"]["hard_negative_calibration"]["verdict"] = "fail"
    decision["content_sha256_before_self_field"] = semantic._self_address(decision)
    result = semantic.validate_decision(packet, decision, authority, contract)
    assert result["eligible_for_training"] is False
    assert "hard_negative_calibration" in result["failed_gates"]


def test_semantic_receipt_requires_complete_one_to_one_decisions():
    contract, authority, packet, decision = _semantic_fixture()
    with pytest.raises(RuntimeError, match="one decision per packet"):
        semantic.build_receipt(
            [packet],
            [],
            authority,
            contract,
            plan_manifest_sha256="1" * 64,
            structural_review_sha256="2" * 64,
            semantic_packet_sha256="3" * 64,
            decision_file_sha256="4" * 64,
        )
    receipt, results = semantic.build_receipt(
        [packet],
        [decision],
        authority,
        contract,
        plan_manifest_sha256="1" * 64,
        structural_review_sha256="2" * 64,
        semantic_packet_sha256="3" * 64,
        decision_file_sha256="4" * 64,
    )
    assert results[0]["eligible_for_training"] is True
    assert receipt["semantic_verification_completed"] is True
    assert receipt["structural_only_pass_accepted"] is False
    assert receipt["training_launch_authorized"] is False
    assert receipt["per_gate_counts"]["source_entailment:pass"] == 1


def test_semantic_decision_rejects_extra_reasoning_fields():
    contract, authority, packet, decision = _semantic_fixture()
    decision["chain_of_thought"] = "synthetic hidden rationale"
    decision["content_sha256_before_self_field"] = semantic._self_address(decision)
    with pytest.raises(RuntimeError, match="decision schema"):
        semantic.validate_decision(packet, decision, authority, contract)
