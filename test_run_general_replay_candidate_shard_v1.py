from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import general_replay_v1 as replay
import run_general_replay_candidate_shard_v1 as runner


def _request() -> dict:
    specs, _ = replay.build_prompt_specs()
    shards = replay.build_candidate_requests(
        [specs[0]],
        model_name=runner.MODEL_NAME,
        model_revision=runner.MODEL_REVISION,
        model_identity_sha256=runner.MODEL_IDENTITY_SHA256,
    )
    return next(item for shard in shards for item in shard)


def _responses(request: dict) -> list[dict]:
    return [
        runner.make_response(request, index, f"Synthetic output {index}.", "stop")
        for index in range(4)
    ]


def test_request_sealing_and_hidden_target_firewall():
    request = _request()
    runner.validate_request(request, request["shard_index"])
    assert request["engine_policy"]["max_num_seqs"] == 64
    assert request["engine_policy"]["enable_prefix_caching"] is False

    forged = deepcopy(request)
    forged["engine_policy"]["max_num_seqs"] = 1_024
    with pytest.raises(ValueError, match="engine policy"):
        runner.validate_request(forged, forged["shard_index"])

    leaked = deepcopy(request)
    leaked["messages"][0]["verifier"] = {"expected": "hidden"}
    with pytest.raises(RuntimeError, match="hidden verifier target"):
        runner.validate_request(leaked, leaked["shard_index"])


def test_synthetic_model_file_sealing(monkeypatch, tmp_path: Path):
    model_directory = tmp_path / "synthetic_model"
    model_directory.mkdir()
    hashes = {}
    for name in runner.MODEL_FILE_SHA256:
        path = model_directory / name
        path.write_text(f"synthetic {name}\n", encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "MODEL_DIRECTORY", model_directory.resolve())
    monkeypatch.setattr(runner, "MODEL_FILE_SHA256", hashes)
    assert runner.validate_model_files(model_directory.resolve()) == hashes

    changed = model_directory / "config.json"
    changed.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="config.json changed"):
        runner.validate_model_files(model_directory.resolve())


def test_tool_call_parser_and_response_addressing():
    request = _request()
    message = runner.parse_assistant_message(
        "<tool_call>\n"
        "<function=calculate_cart_total>\n"
        "<parameter=prices_cents>\n[125, 240]\n</parameter>\n"
        "<parameter=quantities>\n[2, 1]\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert message == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "type": "function",
            "function": {
                "name": "calculate_cart_total",
                "arguments": {
                    "prices_cents": [125, 240],
                    "quantities": [2, 1],
                },
            },
        }],
    }
    response = runner.make_response(request, 2, "Synthetic output.", "stop")
    assert response["generator"]["seed"] == (
        request["generation"]["candidate_seeds"][2]
    )
    assert response["response_sha256"] == replay.candidate_response_sha256(
        response
    )


def test_partial_resume_requires_an_ordered_complete_request_prefix(tmp_path):
    request = _request()
    responses = _responses(request)
    partial = tmp_path / "synthetic.responses.partial.jsonl"
    runner._atomic_write(partial, runner._response_bytes(responses))
    assert runner._load_partial(partial, [request]) == responses

    runner._atomic_write(
        partial,
        runner._response_bytes([responses[1], responses[0], *responses[2:]]),
    )
    with pytest.raises(RuntimeError, match="complete request prefix"):
        runner._load_partial(partial, [request])


def test_partial_resume_fails_closed_on_stale_response_digest(tmp_path):
    request = _request()
    responses = _responses(request)
    responses[0]["assistant_message"]["content"] = "tampered"
    partial = tmp_path / "synthetic.responses.partial.jsonl"
    runner._atomic_write(partial, runner._response_bytes(responses))
    with pytest.raises(ValueError, match="stale response digest"):
        runner._load_partial(partial, [request])


def test_vllm_engine_kwargs_are_memory_safe_and_bf16():
    args = SimpleNamespace(
        model_directory=runner.MODEL_DIRECTORY,
        gpu_memory_utilization=0.9,
        max_model_len=8_192,
        enforce_eager=False,
    )
    kwargs = runner.vllm_engine_kwargs(args)
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["tensor_parallel_size"] == 1
    assert kwargs["max_num_seqs"] == 64
    assert kwargs["enable_prefix_caching"] is False
