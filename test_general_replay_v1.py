from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

import build_general_replay_corpus_v1 as retired_corpus_builder
import build_general_replay_prompt_specs_v1 as prompt_builder
import general_replay_v1 as replay
import prepare_general_replay_candidate_shards_v1 as shard_builder


MODEL_SHA256 = "a" * 64


class SyntheticQwenTokenizer:
    """Synthetic contract double for the Qwen chat-template mask interface."""

    def __init__(self):
        self.calls = []

    def convert_tokens_to_ids(self, token):
        return {"<|im_start|>": 1, "<|im_end|>": 6}[token]

    def encode(self, value, add_special_tokens=False):
        assert value == "\n" and add_special_tokens is False
        return [5]

    def decode(self, ids, skip_special_tokens=False):
        return {2: "system", 3: "user", 4: "assistant"}.get(
            ids[0], "synthetic"
        )

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] is True
        assert kwargs["enable_thinking"] is False
        self.calls.append({"messages": messages, "kwargs": kwargs})
        input_ids = []
        role_ids = {"system": 2, "user": 3, "assistant": 4}
        for index, message in enumerate(messages):
            input_ids.extend([1, role_ids[message["role"]], 5])
            if message["role"] == "assistant":
                if index == len(messages) - 1:
                    input_ids.extend([8, 9])
                input_ids.extend([30] * 64)
            else:
                input_ids.extend([20] * 8)
            input_ids.extend([6, 7])
        if kwargs["add_generation_prompt"]:
            input_ids.extend([1, 4, 5, 8, 9])
        return input_ids


def _spec(category: str, *, scale: int = 1) -> dict:
    specs, _ = replay.build_prompt_specs(spec_scale=scale)
    return next(item for item in specs if item["category"] == category)


def _flatten(shards: list[list[dict]]) -> list[dict]:
    return [item for shard in shards for item in shard]


def _requests(specs: list[dict]) -> list[dict]:
    return _flatten(replay.build_candidate_requests(
        specs,
        model_name="synthetic-qwen-base",
        model_revision="synthetic-revision-1",
        model_identity_sha256=MODEL_SHA256,
    ))


def _passing_message(spec: dict) -> dict:
    verifier = spec["verifier"]
    config = verifier["config"]
    if verifier["type"] in {"exact_text_v1", "normalized_exact_text_v1"}:
        return {"role": "assistant", "content": config["expected"]}
    if verifier["type"] == "json_exact_v1":
        return {
            "role": "assistant",
            "content": json.dumps(config["expected"], sort_keys=True),
        }
    if verifier["type"] == "tool_call_exact_v1":
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": config["expected"],
        }
    if verifier["type"] == "python_function_cases_v1":
        return {"role": "assistant", "content": config["reference_answer"]}
    raise AssertionError("subjective verifier has no objective passing message")


def _response(request: dict, assistant_message: dict, index: int = 0) -> dict:
    response = {
        "schema": replay.CANDIDATE_RESPONSE_SCHEMA,
        "request_id": request["request_id"],
        "spec_id": request["spec_id"],
        "shard_index": request["shard_index"],
        "candidate_index": index,
        "assistant_message": assistant_message,
        "finish_reason": (
            "tool_calls" if "tool_calls" in assistant_message else "stop"
        ),
        "generator": {
            "name": request["model"]["name"],
            "revision": request["model"]["revision"],
            "identity_sha256": request["model"]["identity_sha256"],
            "seed": request["generation"]["seed"] + index,
        },
        "response_sha256": "0" * 64,
    }
    response["response_sha256"] = replay.candidate_response_sha256(response)
    return response


def _approval(spec: dict, response: dict, *, status: str = "approved") -> dict:
    return {
        "schema": "general-replay-candidate-approval-v1",
        "spec_id": spec["spec_id"],
        "response_sha256": response["response_sha256"],
        "status": status,
        "rubric_id": spec["verifier"]["config"]["rubric_id"],
        "reviewer": "synthetic-reviewer",
        "reviewed_at": "2026-01-01T00:00:00Z",
        "reason": "synthetic rubric decision",
    }


def _has_key(value, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return bool(set(value) & forbidden) or any(
            _has_key(child, forbidden) for child in value.values()
        )
    if isinstance(value, list):
        return any(_has_key(child, forbidden) for child in value)
    return False


def _install_synthetic_seed(monkeypatch, tmp_path: Path) -> None:
    seed_path = tmp_path / "synthetic_proxy_anchor.jsonl"
    report_path = tmp_path / "synthetic_proxy_anchor.report.json"
    rows = []
    for index in range(128):
        instruction = f"Synthetic instruction {index}"
        answer = f"Synthetic answer {index}"
        rows.append({
            "answer": answer,
            "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
            "document_id": f"synthetic-document-{index}",
            "instruction": instruction,
            "instruction_sha256": hashlib.sha256(
                instruction.encode()
            ).hexdigest(),
            "item_id": f"synthetic-item-{index}",
            "parent_item_id": f"synthetic-parent-{index}",
            "parent_text_sha256": hashlib.sha256(
                f"synthetic-parent-text-{index}".encode()
            ).hexdigest(),
            "quality_bucket": "train_only_general_knowledge_instruction_proxy",
            "split": "anchor_general_qa_proxy",
        })
    seed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    seed_sha256 = replay.file_sha256(seed_path)
    report = {
        "schema": "general-qa-proxy-anchor-build-v43h",
        "output_sha256": seed_sha256,
        "rows": 128,
        "direct_benchmark_source": {
            "opened": False,
            "authorized_for_qa_semantics": False,
        },
        "policy": {
            "protected_eval_ood_heldout_or_benchmark_semantics_opened": False,
        },
    }
    report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    monkeypatch.setattr(replay, "SAFE_SEED", seed_path.resolve())
    monkeypatch.setattr(replay, "SAFE_SEED_REPORT", report_path.resolve())
    monkeypatch.setattr(replay, "ROOT", tmp_path.resolve())
    monkeypatch.setattr(replay, "SAFE_SEED_SHA256", seed_sha256)
    monkeypatch.setattr(
        replay, "SAFE_SEED_REPORT_SHA256", replay.file_sha256(report_path)
    )


def test_exact_token_budgets_match_plan_proportions():
    budgets = replay.token_budgets(120_000)
    assert budgets == {
        "coding_debugging": 24_000,
        "mathematical_reasoning": 18_000,
        "json_structured_data": 12_000,
        "tool_use_function_calling": 12_000,
        "instruction_following": 12_000,
        "ordinary_conversation": 12_000,
        "multilingual": 12_000,
        "safety_refusal": 6_000,
        "uncertainty_hallucination_resistance": 6_000,
        "long_context": 6_000,
    }
    assert sum(budgets.values()) == 120_000
    for invalid in (99_980, 150_020, 120_001, True, 120_000.0):
        with pytest.raises(ValueError):
            replay.token_budgets(invalid)


def test_candidate_response_only_corpus_builder_is_retired_fail_closed():
    with pytest.raises(RuntimeError, match="candidate-response-only assembly"):
        retired_corpus_builder.main([])


def test_initial_prompt_specs_are_deterministic_schema_exact_and_token_governed():
    first_specs, first_report = replay.build_prompt_specs()
    second_specs, second_report = replay.build_prompt_specs()
    assert first_specs == second_specs
    assert first_report == second_report
    assert len(first_specs) == 80
    assert first_report["counts_by_category"] == replay.INITIAL_SPEC_COUNTS
    assert first_report["prompt_count_is_token_budget_authority"] is False
    assert first_report["target_assistant_tokens_by_category"] == (
        replay.token_budgets(150_000)
    )
    assert len({item["spec_id"] for item in first_specs}) == 80
    assert len({item["source_group_id"] for item in first_specs}) == 80
    assert len({item["prompt_identity_sha256"] for item in first_specs}) == 80
    assert all(item["lineage"] == {
        "source_kind": "deterministic_synthetic_v1",
        "direct_benchmark_prompt": False,
        "protected_source_access": False,
        "parent_artifacts": [],
    } for item in first_specs)


def test_subjective_categories_and_non_exact_multilingual_slots_are_gated():
    specs, _ = replay.build_prompt_specs(spec_scale=2)
    always_gated = {
        "ordinary_conversation",
        "safety_refusal",
        "uncertainty_hallucination_resistance",
    }
    for spec in specs:
        if spec["category"] in always_gated:
            assert spec["verifier"]["type"] == "approval_required_v1"
            assert spec["verifier"]["status"] == "candidate_gate_required"
    multilingual = [
        item for item in specs if item["category"] == "multilingual"
    ]
    assert [item["verifier"]["type"] for item in multilingual[:8]] == (
        ["normalized_exact_text_v1"] * 8
    )
    assert all(
        item["verifier"]["type"] == "approval_required_v1"
        for item in multilingual[8:]
    )


def test_prompt_validator_rejects_duplication_and_forged_lineage():
    spec = _spec("mathematical_reasoning")
    with pytest.raises(ValueError, match="duplicate"):
        replay.validate_prompt_specs([spec, deepcopy(spec)])

    forged = deepcopy(spec)
    forged["lineage"]["direct_benchmark_prompt"] = True
    with pytest.raises(ValueError, match="forbidden prompt lineage"):
        replay.validate_prompt_specs([forged])

    forged = deepcopy(spec)
    forged["source_group_id"] = "synthetic-replay-v1:mathematical_reasoning:9999"
    with pytest.raises(ValueError, match="generator lineage|spec identity"):
        replay.validate_prompt_specs([forged])

    forged = deepcopy(spec)
    forged["template_policy"] = "custom-template"
    with pytest.raises(ValueError, match="official Qwen template"):
        replay.validate_prompt_specs([forged])


def test_four_candidate_request_shards_are_balanced_and_hide_verifier_targets():
    specs, _ = replay.build_prompt_specs()
    shards = replay.build_candidate_requests(
        specs,
        model_name="synthetic-qwen-base",
        model_revision="synthetic-revision-1",
        model_identity_sha256=MODEL_SHA256,
    )
    assert len(shards) == 4
    assert [len(shard) for shard in shards] == [20, 20, 20, 20]
    requests = _flatten(shards)
    assert len({item["request_id"] for item in requests}) == 80
    assert all(item["generation"]["candidates"] == 4 for item in requests)
    assert all(len(set(item["generation"]["candidate_seeds"])) == 4 for item in requests)
    assert all(item["template_parameters"] == {
        "add_generation_prompt": True,
        "enable_thinking": False,
    } for item in requests)
    assert all(item["engine_policy"] == {
        "backend": "vllm",
        "version": "0.25.0",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_num_seqs": 64,
        "enable_prefix_caching": False,
    } for item in requests)
    assert all(item["template_policy"] == (
        "official_qwen_apply_chat_template_v1"
    ) for item in requests)
    assert not _has_key(
        requests, {"verifier", "config", "expected", "reference_answer", "rubric_id"}
    )
    with pytest.raises(ValueError, match="SHA-256"):
        replay.build_candidate_requests(
            specs,
            model_name="synthetic-qwen-base",
            model_revision="synthetic-revision-1",
            model_identity_sha256="unpinned",
        )

    insufficient = replay.candidate_capacity_proof(specs, 120_000)
    assert insufficient["max_selectable_assistant_tokens"] == 17_664
    assert insufficient["categories"]["instruction_following"][
        "max_selectable_assistant_tokens"
    ] == 768
    assert insufficient["candidate_generation_launch_eligible"] is False
    with pytest.raises(RuntimeError, match="2x capacity"):
        shard_builder.build_shard_artifacts(
            specs,
            model_name="synthetic-qwen-base",
            model_revision="synthetic-revision-1",
            model_identity_sha256=MODEL_SHA256,
            total_assistant_tokens=120_000,
        )

    scaled_specs, _ = replay.build_prompt_specs(spec_scale=32)
    shard_bytes, report_bytes, report = shard_builder.build_shard_artifacts(
        scaled_specs,
        model_name="synthetic-qwen-base",
        model_revision="synthetic-revision-1",
        model_identity_sha256=MODEL_SHA256,
        total_assistant_tokens=120_000,
    )
    assert len(shard_bytes) == 4
    assert report["generation_launched"] is False
    assert report["verifier_targets_in_requests"] is False
    assert report["engine_policy"]["max_num_seqs"] == 64
    assert report["engine_policy"]["enable_prefix_caching"] is False
    assert [item["rows"] for item in report["shards"]] == [640, 640, 640, 640]
    assert report["max_selectable_assistant_tokens"] == 565_248
    assert all(
        item["meets_minimum_headroom"]
        for item in report["candidate_capacity_by_category"].values()
    )
    assert json.loads(report_bytes)["content_sha256_before_self_field"] == (
        report["content_sha256_before_self_field"]
    )


def test_candidate_request_validator_rejects_forged_or_ambiguous_lineage():
    specs, _ = replay.build_prompt_specs()
    requests = _flatten(replay.build_candidate_requests(
        specs,
        model_name="synthetic-qwen-base",
        model_revision="synthetic-revision-1",
        model_identity_sha256=MODEL_SHA256,
    ))
    replay.validate_candidate_requests(specs, requests)

    forged = deepcopy(requests)
    forged[0]["messages"][0]["content"] += " forged"
    with pytest.raises(ValueError, match="request lineage or policy changed"):
        replay.validate_candidate_requests(specs, forged)

    forged = deepcopy(requests)
    forged[0]["generation"]["candidate_seeds"][0] += 1
    with pytest.raises(ValueError, match="request lineage or policy changed"):
        replay.validate_candidate_requests(specs, forged)

    duplicated = [*requests, deepcopy(requests[0])]
    with pytest.raises(ValueError, match="duplicate request"):
        replay.validate_candidate_requests(specs, duplicated)

    with pytest.raises(ValueError, match="coverage is incomplete"):
        replay.validate_candidate_requests(specs, requests[:-1])


def test_objective_verifiers_pass_references_and_reject_wrong_candidates():
    for category in (
        "mathematical_reasoning",
        "json_structured_data",
        "tool_use_function_calling",
        "instruction_following",
        "multilingual",
        "long_context",
    ):
        spec = _spec(category)
        assert replay.verify_candidate(spec, _passing_message(spec))["passed"]
        wrong = {"role": "assistant", "content": "synthetic wrong answer"}
        assert not replay.verify_candidate(spec, wrong)["passed"]

    coding = _spec("coding_debugging")
    assert replay.verify_candidate(coding, _passing_message(coding))["passed"]
    wrong_code = {
        "role": "assistant",
        "content": "```python\ndef wrong_name(values):\n    return values\n```",
    }
    assert not replay.verify_candidate(coding, wrong_code)["passed"]

    subjective = _spec("ordinary_conversation")
    result = replay.verify_candidate(
        subjective, {"role": "assistant", "content": "Synthetic response."}
    )
    assert result["status"] == "pending_approval"
    assert result["passed"] is False


def test_reference_compiler_uses_targets_and_never_compiles_subjective_rows():
    specs, _ = replay.build_prompt_specs(spec_scale=2)
    tokenizer = SyntheticQwenTokenizer()
    rows, audit = replay.build_reference_compiler_rows(specs, tokenizer)
    objective_specs = [
        spec for spec in specs
        if spec["verifier"]["type"] != "approval_required_v1"
    ]
    assert len(rows) == len(objective_specs)
    assert audit["compiled_rows"] == len(objective_specs)
    assert audit["subjective_gated_specs"] == len(specs) - len(objective_specs)
    assert all(
        row["lineage"]["source_kind"]
        == replay.REFERENCE_COMPILER_NAME
        for row in rows
    )
    assert all(row["verifier"]["status"] == "passed" for row in rows)
    assert not any(
        row["category"] in {
            "ordinary_conversation",
            "safety_refusal",
            "uncertainty_hallucination_resistance",
        }
        for row in rows
    )


def test_every_scaled_coding_reference_passes_its_own_cases():
    specs, _ = replay.build_prompt_specs(spec_scale=32)
    coding = [spec for spec in specs if spec["category"] == "coding_debugging"]
    assert len(coding) == 512
    assert all(
        replay.verify_candidate(
            spec, replay.compile_objective_reference_message(spec)
        )["passed"]
        for spec in coding
    )


def test_candidate_lineage_digest_verification_and_assistant_only_row():
    spec = _spec("mathematical_reasoning")
    request = _requests([spec])[0]
    responses = [
        _response(request, _passing_message(spec), index)
        for index in range(4)
    ]
    response = responses[0]
    replay.validate_candidate_responses(responses, [request])
    tokenizer = SyntheticQwenTokenizer()
    rows, audit = replay.build_verified_candidate_rows(
        [spec], [request], responses, [], tokenizer
    )
    assert len(rows) == 1
    assert audit["verified_rows"] == 1
    assert rows[0]["assistant_token_count"] == 65
    assert rows[0]["assistant_mask"] == {
        "policy": "assistant_only_v1",
        "assistant_message_indices": [1],
        "system_tokens": False,
        "user_tokens": False,
        "tool_result_tokens": False,
    }
    assert rows[0]["lineage"]["response_sha256"] == (
        response["response_sha256"]
    )

    forged = deepcopy(response)
    forged["generator"]["seed"] += 1
    forged["response_sha256"] = replay.candidate_response_sha256(forged)
    with pytest.raises(ValueError, match="generator lineage"):
        replay.validate_candidate_responses([forged, *responses[1:]], [request])

    stale = deepcopy(response)
    stale["assistant_message"]["content"] += " changed"
    with pytest.raises(ValueError, match="stale response digest"):
        replay.validate_candidate_responses([stale, *responses[1:]], [request])

    with pytest.raises(ValueError, match="coverage is incomplete"):
        replay.validate_candidate_responses(responses[:3], [request])


def test_exhaustive_verification_retains_alternatives_for_group_selection():
    spec = _spec("mathematical_reasoning")
    request = _requests([spec])[0]
    responses = [
        _response(request, _passing_message(spec), index)
        for index in range(4)
    ]
    rows, audit = replay.build_all_verified_candidate_rows(
        [spec], [request], responses, [], SyntheticQwenTokenizer()
    )
    assert len(rows) == 4
    assert len({row["source_group_id"] for row in rows}) == 1
    assert audit["objective_passed"] == 4
    assert audit["eligible_candidate_rows"] == 4
    assert audit["eligible_source_groups"] == 1
    assert audit["by_category"]["mathematical_reasoning"] == {
        "specs": 1,
        "responses": 4,
        "length_truncated": 0,
        "token_bound_failures": 0,
        "objective_passed": 4,
        "objective_failed": 0,
        "approval_approved": 0,
        "approval_gated": 0,
        "approval_rejected": 0,
        "eligible_candidate_rows": 4,
        "eligible_source_groups": 1,
    }


def test_subjective_candidate_cannot_enter_without_matching_approval():
    spec = _spec("ordinary_conversation")
    request = _requests([spec])[0]
    responses = [
        _response(request, {
            "role": "assistant",
            "content": " ".join(["synthetic"] * 60),
        }, index)
        for index in range(4)
    ]
    response = responses[0]
    tokenizer = SyntheticQwenTokenizer()
    rows, audit = replay.build_verified_candidate_rows(
        [spec], [request], responses, [], tokenizer
    )
    assert rows == []
    assert audit["approval_gated"] == 4

    wrong = _approval(spec, response)
    wrong["rubric_id"] = "wrong-rubric"
    rows, audit = replay.build_verified_candidate_rows(
        [spec], [request], responses, [wrong], tokenizer
    )
    assert rows == []
    assert audit["approval_gated"] == 4

    rows, audit = replay.build_verified_candidate_rows(
        [spec], [request], responses, [_approval(spec, response)], tokenizer
    )
    assert len(rows) == 1
    assert rows[0]["verifier"]["status"] == "passed"
    assert audit["approval_gated"] == 0


def test_assistant_mask_template_contract_and_tool_forwarding():
    tokenizer = SyntheticQwenTokenizer()
    count = replay.assistant_token_count(
        tokenizer,
        [
            {"role": "user", "content": "Synthetic prompt"},
            {"role": "assistant", "content": "Synthetic answer"},
        ],
        [{"type": "function", "function": {"name": "synthetic"}}],
    )
    assert count == 65
    assert "tools" in tokenizer.calls[0]["kwargs"]

    multi_turn = [
        {"role": "system", "content": "Synthetic system."},
        {"role": "user", "content": "Synthetic user one."},
        {"role": "assistant", "content": "Synthetic assistant one."},
        {"role": "user", "content": "Synthetic user two."},
        {"role": "assistant", "content": "Synthetic assistant two."},
    ]
    multi_mask = replay.assistant_token_mask(tokenizer, multi_turn, [])
    assert sum(multi_mask) == 130
    assert multi_mask[0] == 0 and multi_mask[-1] == 0
    starts = [
        index for index, value in enumerate(multi_mask)
        if value == 1 and (index == 0 or multi_mask[index - 1] == 0)
    ]
    assert len(starts) == 2

    class PrefixMismatchTokenizer:
        def convert_tokens_to_ids(self, token):
            return {"<|im_start|>": 1, "<|im_end|>": 4}[token]

        def encode(self, value, add_special_tokens=False):
            return [3]

        def decode(self, ids, skip_special_tokens=False):
            return "user" if ids == [2] else "assistant"

        def apply_chat_template(self, messages, **kwargs):
            if kwargs["add_generation_prompt"]:
                return [1, 2, 3, 4, 1, 5, 3, 99]
            return [1, 2, 3, 4, 1, 5, 3, 6, 4]

    with pytest.raises(ValueError, match="generation prefix differs"):
        replay.assistant_token_count(
            PrefixMismatchTokenizer(),
            [
                {"role": "user", "content": "Synthetic prompt"},
                {"role": "assistant", "content": "Synthetic answer"},
            ],
            [],
        )


def test_real_qwen_template_prefix_mask_contract():
    from transformers import AutoTokenizer
    from qwen_chat_masking_v1 import encode_chat_assistant_only

    model_path = replay.ROOT / "models/Qwen3.6-35B-A3B"
    assert replay.file_sha256(model_path / "chat_template.jinja") == (
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    messages = [
        {"role": "user", "content": "Synthetic prefix-mask prompt."},
        {"role": "assistant", "content": "Synthetic prefix-mask answer."},
    ]
    mask = replay.assistant_token_mask(tokenizer, messages, [])
    encoded = encode_chat_assistant_only(tokenizer, messages)
    assert mask == encoded["assistant_mask"]
    assert len(mask) == len(encoded["input_ids"])
    assert 0 < sum(mask) < len(mask)
    first_assistant = mask.index(1)
    assert all(value == 0 for value in mask[:first_assistant])
    assert all(value == 1 for value in mask[first_assistant:-1])
    assert mask[-1] == 0


def test_exact_subset_is_deterministic_without_duplication_or_padding():
    rows = [
        {"row_id": "row-c", "assistant_token_count": 5},
        {"row_id": "row-a", "assistant_token_count": 3},
        {"row_id": "row-b", "assistant_token_count": 4},
    ]
    chosen = replay.exact_token_subset(rows, 7)
    assert [item["row_id"] for item in chosen] == ["row-a", "row-b"]
    assert sum(item["assistant_token_count"] for item in chosen) == 7
    assert replay.exact_token_subset(rows, 0) == []
    with pytest.raises(RuntimeError, match="cannot satisfy"):
        replay.exact_token_subset(rows, 2)
    with pytest.raises(ValueError, match="must be unique"):
        replay.exact_token_subset([rows[0], deepcopy(rows[0])], 10)


def test_exact_group_subset_uses_one_alternative_per_source_group():
    rows = [
        {
            "row_id": "row-a-short",
            "source_group_id": "group-a",
            "assistant_token_count": 3,
        },
        {
            "row_id": "row-a-long",
            "source_group_id": "group-a",
            "assistant_token_count": 5,
        },
        {
            "row_id": "row-b",
            "source_group_id": "group-b",
            "assistant_token_count": 4,
        },
    ]
    chosen = replay.exact_token_group_subset(rows, 9)
    assert [row["row_id"] for row in chosen] == ["row-a-long", "row-b"]
    assert len({row["source_group_id"] for row in chosen}) == 2
    with pytest.raises(RuntimeError, match="unique source groups"):
        replay.exact_token_group_subset(rows, 8)


def test_safe_input_firewall_rejects_lexical_alias_and_hardlink_paths(tmp_path):
    safe = tmp_path / "synthetic_input.jsonl"
    safe.write_text("{}\n", encoding="utf-8")
    assert replay.safe_regular_input(safe, "synthetic input") == safe.resolve()

    forbidden = tmp_path / "synthetic_evaluation_input.jsonl"
    with pytest.raises(RuntimeError, match="forbidden source path"):
        replay.safe_regular_input(forbidden, "synthetic forbidden input")

    for directory in ("development", "final", "manual-review"):
        forbidden = tmp_path / directory / "synthetic_input.jsonl"
        with pytest.raises(RuntimeError, match="forbidden source path"):
            replay.safe_regular_input(forbidden, "synthetic forbidden input")

    alias = tmp_path / "synthetic_alias.jsonl"
    alias.symlink_to(safe)
    with pytest.raises(RuntimeError, match="symlink"):
        replay.safe_regular_input(alias, "synthetic alias")

    hardlink = tmp_path / "synthetic_hardlink.jsonl"
    os.link(safe, hardlink)
    with pytest.raises(RuntimeError, match="hard-link"):
        replay.safe_regular_input(hardlink, "synthetic hardlink")


def test_seed_authority_and_prompt_artifact_build_use_synthetic_fixture(
        monkeypatch, tmp_path):
    _install_synthetic_seed(monkeypatch, tmp_path)
    rows, _ = replay.validate_seed_authority()
    assert len(rows) == 128
    assert len({item["source_group_id"] for item in rows}) == 128
    assert all(item["assistant_mask"]["policy"] == "assistant_only_v1" for item in rows)
    assert all(item["lineage"]["direct_benchmark_prompt"] is False for item in rows)

    monkeypatch.setattr(prompt_builder, "validate_seed_authority", lambda: (rows, {}))
    output_one, report_one, parsed_one = prompt_builder.build_artifacts(
        build_seed=replay.DEFAULT_BUILD_SEED,
        total_assistant_tokens=120_000,
        spec_scale=1,
    )
    output_two, report_two, parsed_two = prompt_builder.build_artifacts(
        build_seed=replay.DEFAULT_BUILD_SEED,
        total_assistant_tokens=120_000,
        spec_scale=1,
    )
    assert output_one == output_two
    assert report_one == report_two
    assert parsed_one == parsed_two
    assert hashlib.sha256(output_one).hexdigest() == parsed_one["output_sha256"]


def test_final_assembly_hits_every_exact_token_quota_without_padding(
        monkeypatch, tmp_path):
    _install_synthetic_seed(monkeypatch, tmp_path)
    seed_rows, _ = replay.validate_seed_authority()
    budgets = replay.token_budgets(120_000)
    seed_instruction_tokens = len(seed_rows) * 65
    candidate_rows = []
    for category, target in budgets.items():
        remaining = target
        if category == "instruction_following":
            remaining -= seed_instruction_tokens
        index = 0
        while remaining:
            count = min(64, remaining)
            candidate_rows.append({
                "row_id": f"synthetic-final-{category}-{index:04d}",
                "category": category,
                "source_group_id": f"synthetic-final-group-{category}-{index:04d}",
                "assistant_token_count": count,
            })
            remaining -= count
            index += 1
    rows, report = replay.build_final_replay_rows(
        seed_rows,
        candidate_rows,
        SyntheticQwenTokenizer(),
        total_assistant_tokens=120_000,
    )
    assert report["assistant_tokens_by_category"] == budgets
    assert report["duplicate_or_padding_fill_used"] is False
    assert len({row["row_id"] for row in rows}) == len(rows)
    assert len({row["source_group_id"] for row in rows}) == len(rows)
