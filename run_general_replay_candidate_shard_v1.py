#!/usr/bin/env python3
"""Generate one verified-schema replay candidate shard on one visible GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any

from general_replay_v1 import (
    CANDIDATE_REQUEST_SCHEMA,
    CANDIDATE_RESPONSE_SCHEMA,
    ROOT,
    SHARD_COUNT,
    candidate_response_sha256,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    safe_regular_input,
    validate_candidate_responses,
)


MODEL_NAME = "Qwen3.6-35B-A3B"
MODEL_REVISION = "local-checkpoint-architecture-contract-4a4960ba"
MODEL_IDENTITY_SHA256 = (
    "4a4960ba80c4e6532f5225984310af35a2a79cd50ffb642ef1c5a54bbe5fba3c"
)
MODEL_DIRECTORY = ROOT / "models/Qwen3.6-35B-A3B"
MODEL_FILE_SHA256 = {
    "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "model.safetensors.index.json": (
        "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": (
        "5186f0defcd7f232382c7f0aebcd2252d073bb921ab240e407b7ae8745d2b29b"
    ),
    "chat_template.jinja": (
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    ),
}
VLLM_VERSION = "0.25.0"
REQUEST_DIRECTORY = ROOT / "data/general_replay_v1/candidate_requests_v1_scale32"
REQUEST_REPORT = REQUEST_DIRECTORY / "request_shards.report.json"
OUTPUT_DIRECTORY = ROOT / "data/general_replay_v1/candidate_responses_v1_scale32"
HIDDEN_TARGET_KEYS = frozenset({
    "config", "expected", "reference_answer", "rubric_id", "verifier",
})
TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
TOOL_PARAMETER = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _contains_key(value: Any, keys: frozenset[str]) -> bool:
    if isinstance(value, dict):
        return bool(set(value) & keys) or any(
            _contains_key(child, keys) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, keys) for child in value)
    return False


def _safe_exact_directory(path: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    if lexical != MODEL_DIRECTORY:
        raise RuntimeError("candidate model must use the sealed checkpoint path")
    current = Path(lexical.anchor)
    metadata = None
    for component in lexical.parts[1:]:
        current /= component
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise RuntimeError("candidate model path cannot use symlink aliases")
    if metadata is None or not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("candidate model path must be a directory")
    return lexical


def validate_model_files(model_directory: Path) -> dict[str, str]:
    model_directory = _safe_exact_directory(model_directory)
    actual = {}
    for name, expected in MODEL_FILE_SHA256.items():
        path = safe_regular_input(
            model_directory / name, f"sealed model file {name}"
        )
        digest = file_sha256(path)
        if digest != expected:
            raise RuntimeError(f"sealed model file {name} changed")
        actual[name] = digest
    return actual


def _exact_keys(value: Any, expected: set[str], location: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{location}: invalid fields")
    return value


def validate_request(request: dict, shard_index: int) -> None:
    item = _exact_keys(request, {
        "assistant_mask_policy", "category", "expected_response_format",
        "engine_policy", "generation", "messages", "model",
        "prompt_identity_sha256",
        "request_id", "schema", "shard_index", "source_group_id", "spec_id",
        "template_parameters", "template_policy", "tools",
    }, "candidate request")
    if item["schema"] != CANDIDATE_REQUEST_SCHEMA:
        raise ValueError("candidate request schema changed")
    if item["shard_index"] != shard_index:
        raise ValueError("candidate request shard lineage changed")
    if item["model"] != {
        "name": MODEL_NAME,
        "revision": MODEL_REVISION,
        "identity_sha256": MODEL_IDENTITY_SHA256,
    }:
        raise ValueError("candidate request model identity changed")
    if item["template_policy"] != "official_qwen_apply_chat_template_v1":
        raise ValueError("candidate request template policy changed")
    if item["template_parameters"] != {
        "add_generation_prompt": True,
        "enable_thinking": False,
    }:
        raise ValueError("candidate request template parameters changed")
    if item["assistant_mask_policy"] != "assistant_only_v1":
        raise ValueError("candidate request assistant mask policy changed")
    if item["engine_policy"] != {
        "backend": "vllm",
        "version": VLLM_VERSION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_num_seqs": 64,
        "enable_prefix_caching": False,
    }:
        raise ValueError("candidate request engine policy changed")
    if _contains_key(item, HIDDEN_TARGET_KEYS):
        raise RuntimeError("candidate request contains a hidden verifier target")
    messages = item["messages"]
    if (
        not isinstance(messages, list)
        or not messages
        or messages[-1].get("role") != "user"
        or any(message.get("role") not in {"system", "user"} for message in messages)
    ):
        raise ValueError("candidate request prompt roles changed")
    generation = _exact_keys(item["generation"], {
        "candidate_seeds", "candidates", "max_new_tokens", "seed",
        "temperature", "top_p",
    }, "candidate request generation")
    if (
        generation["candidates"] != 4
        or generation["candidate_seeds"] != [
            generation["seed"] + index for index in range(4)
        ]
        or len(set(generation["candidate_seeds"])) != 4
        or generation["temperature"] != 0.3
        or generation["top_p"] != 0.9
        or not isinstance(generation["max_new_tokens"], int)
        or not 0 < generation["max_new_tokens"] <= 384
    ):
        raise ValueError("candidate request generation contract changed")


def load_request_shard(
        request_directory: Path, report_path: Path,
        shard_index: int) -> tuple[list[dict], dict, str]:
    if not 0 <= shard_index < SHARD_COUNT:
        raise ValueError("candidate shard index must be 0-3")
    report_path = safe_regular_input(report_path, "candidate request report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema")
        != "general-replay-candidate-request-shards-report-v1"
        or report.get("generation_launched") is not False
        or report.get("shard_count") != SHARD_COUNT
        or report.get("requests") != 2_560
        or report.get("model") != {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "identity_sha256": MODEL_IDENTITY_SHA256,
        }
        or report.get("template_policy")
        != "official_qwen_apply_chat_template_v1"
        or report.get("verifier_targets_in_requests") is not False
        or report.get("engine_policy") != {
            "backend": "vllm",
            "version": VLLM_VERSION,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_num_seqs": 64,
            "enable_prefix_caching": False,
        }
    ):
        raise RuntimeError("candidate request report changed")
    shard_report = next(
        (item for item in report.get("shards", [])
         if item.get("index") == shard_index),
        None,
    )
    if shard_report is None or shard_report.get("rows") != 640:
        raise RuntimeError("candidate request shard report changed")
    shard_path = safe_regular_input(
        request_directory / f"shard-{shard_index:02d}.requests.jsonl",
        f"candidate request shard {shard_index}",
    )
    raw = shard_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != shard_report.get("sha256"):
        raise RuntimeError("candidate request shard identity changed")
    requests = [
        json.loads(line) for line in raw.decode("utf-8").splitlines()
        if line.strip()
    ]
    if len(requests) != 640:
        raise RuntimeError("candidate request shard row count changed")
    for request in requests:
        validate_request(request, shard_index)
    for field in ("request_id", "spec_id", "source_group_id"):
        if len({item[field] for item in requests}) != len(requests):
            raise RuntimeError(f"candidate request shard duplicates {field}")
    return requests, report, digest


def load_tokenizer(model_directory: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        local_files_only=True,
        trust_remote_code=False,
    )
    template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(template, str)
        or hashlib.sha256(template.encode("utf-8")).hexdigest()
        != MODEL_FILE_SHA256["chat_template.jinja"]
    ):
        raise RuntimeError("loaded official chat template changed")
    return tokenizer


def render_request(tokenizer, request: dict) -> tuple[str, int]:
    kwargs = dict(request["template_parameters"])
    if request["tools"]:
        kwargs["tools"] = request["tools"]
    text = tokenizer.apply_chat_template(
        request["messages"], tokenize=False, **kwargs
    )
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if not isinstance(text, str) or not text:
        raise RuntimeError("official template rendered an empty candidate prompt")
    if not isinstance(tokenized, list) or not tokenized:
        raise RuntimeError("official template tokenization failed")
    return text, len(tokenized)


def _parameter_value(text: str) -> Any:
    value = text.strip()
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_assistant_message(text: str) -> dict:
    matches = list(TOOL_CALL.finditer(text))
    if not matches:
        return {"role": "assistant", "content": text.strip()}
    tool_calls = []
    for match in matches:
        body = match.group(2)
        arguments = {}
        parameters = list(TOOL_PARAMETER.finditer(body))
        residual = TOOL_PARAMETER.sub("", body).strip()
        if residual or not parameters:
            return {"role": "assistant", "content": text.strip()}
        for parameter in parameters:
            name = parameter.group(1).strip()
            if not name or name in arguments:
                return {"role": "assistant", "content": text.strip()}
            arguments[name] = _parameter_value(parameter.group(2))
        tool_calls.append({
            "type": "function",
            "function": {
                "name": match.group(1).strip(),
                "arguments": arguments,
            },
        })
    content = TOOL_CALL.sub("", text).strip()
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }


def make_response(
        request: dict, candidate_index: int, text: str,
        engine_finish_reason: str | None) -> dict:
    assistant_message = parse_assistant_message(text)
    if "tool_calls" in assistant_message:
        finish_reason = "tool_calls"
    elif engine_finish_reason in {"length", "stop"}:
        finish_reason = engine_finish_reason
    else:
        raise RuntimeError(
            f"vLLM returned unsupported finish reason {engine_finish_reason!r}"
        )
    response = {
        "schema": CANDIDATE_RESPONSE_SCHEMA,
        "request_id": request["request_id"],
        "spec_id": request["spec_id"],
        "shard_index": request["shard_index"],
        "candidate_index": candidate_index,
        "assistant_message": assistant_message,
        "finish_reason": finish_reason,
        "generator": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "identity_sha256": MODEL_IDENTITY_SHA256,
            "seed": request["generation"]["candidate_seeds"][candidate_index],
        },
        "response_sha256": "0" * 64,
    }
    response["response_sha256"] = candidate_response_sha256(response)
    return response


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _response_bytes(responses: list[dict]) -> bytes:
    return b"".join(canonical_bytes(item) for item in responses)


def _load_partial(path: Path, requests: list[dict]) -> list[dict]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("candidate partial output is not a safe regular file")
    responses = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_candidate_responses(
        responses, requests, require_complete=False
    )
    expected_slots = [
        (request["request_id"], candidate_index)
        for request in requests
        for candidate_index in range(4)
    ]
    actual_slots = [
        (response["request_id"], response["candidate_index"])
        for response in responses
    ]
    if actual_slots != expected_slots[:len(actual_slots)] or len(responses) % 4:
        raise RuntimeError("candidate partial output is not a complete request prefix")
    return responses


def _paths(
        output_directory: Path, shard_index: int,
        smoke: bool) -> dict[str, Path]:
    directory = output_directory / "smoke_v1" if smoke else output_directory
    stem = f"shard-{shard_index:02d}"
    return {
        "partial": directory / f"{stem}.responses.partial.jsonl",
        "output": directory / f"{stem}.responses.jsonl",
        "report": directory / f"{stem}.responses.report.json",
        "telemetry": directory / f"{stem}.telemetry.json",
    }


def _telemetry(
        *, shard_index: int, gpu_index: int, started_at: str,
        selected_requests: int, completed_responses: int,
        prompt_tokens: int, generated_tokens: int,
        status: str) -> dict:
    return {
        "schema": "general-replay-candidate-worker-telemetry-v1",
        "status": status,
        "pid": os.getpid(),
        "shard_index": shard_index,
        "physical_gpu_index": gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": started_at,
        "updated_at": utc_now(),
        "requests": selected_requests,
        "completed_requests": completed_responses // 4,
        "completed_responses": completed_responses,
        "prompt_tokens_rendered": prompt_tokens,
        "generated_tokens": generated_tokens,
        "model_dtype": "bfloat16",
        "engine": {"name": "vllm", "version": VLLM_VERSION},
        "engine_policy": {
            "max_num_seqs": 64,
            "enable_prefix_caching": False,
        },
    }


def vllm_engine_kwargs(args: argparse.Namespace) -> dict:
    return {
        "model": str(args.model_directory),
        "tokenizer": str(args.model_directory),
        "runner": "generate",
        "trust_remote_code": False,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "seed": 0,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": 64,
        "enable_prefix_caching": False,
        "enforce_eager": args.enforce_eager,
    }


def run(args: argparse.Namespace) -> dict:
    started_at = utc_now()
    model_hashes = validate_model_files(args.model_directory)
    requests, request_report, request_shard_sha256 = load_request_shard(
        args.request_directory, args.request_report, args.shard_index
    )
    if args.smoke:
        requests = requests[:1]
    elif args.request_limit is not None:
        raise ValueError("request limits are permitted only for smoke runs")
    paths = _paths(args.output_directory, args.shard_index, args.smoke)
    if paths["output"].exists():
        existing = [
            json.loads(line)
            for line in paths["output"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        validate_candidate_responses(existing, requests)
        return {
            "status": "already_complete",
            "output": str(paths["output"]),
            "responses": len(existing),
        }

    tokenizer = load_tokenizer(args.model_directory)
    rendered = []
    prompt_token_counts = []
    for request in requests:
        text, token_count = render_request(tokenizer, request)
        if token_count + request["generation"]["max_new_tokens"] > args.max_model_len:
            raise RuntimeError("candidate request exceeds sealed model context length")
        rendered.append(text)
        prompt_token_counts.append(token_count)

    from vllm import LLM, SamplingParams, __version__ as vllm_version

    if vllm_version != VLLM_VERSION:
        raise RuntimeError("candidate generator requires vLLM 0.25.0 exactly")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.gpu_index):
        raise RuntimeError("worker requires exactly one sealed physical GPU")
    engine = LLM(**vllm_engine_kwargs(args))

    responses = _load_partial(paths["partial"], requests)
    completed_requests = len(responses) // 4
    generated_tokens = 0
    prompt_tokens = sum(prompt_token_counts[:completed_requests]) * 4
    _atomic_write(paths["telemetry"], canonical_bytes(_telemetry(
        shard_index=args.shard_index,
        gpu_index=args.gpu_index,
        started_at=started_at,
        selected_requests=len(requests),
        completed_responses=len(responses),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        status="running",
    )))

    for batch_start in range(
            completed_requests, len(requests), args.request_batch_size):
        batch_end = min(batch_start + args.request_batch_size, len(requests))
        prompts = []
        parameters = []
        slots = []
        for request_index in range(batch_start, batch_end):
            request = requests[request_index]
            for candidate_index in range(4):
                prompts.append(rendered[request_index])
                slots.append((request_index, candidate_index))
                parameters.append(SamplingParams(
                    n=1,
                    temperature=request["generation"]["temperature"],
                    top_p=request["generation"]["top_p"],
                    seed=request["generation"]["candidate_seeds"][
                        candidate_index
                    ],
                    max_tokens=request["generation"]["max_new_tokens"],
                ))
        started_batch = time.monotonic()
        outputs = engine.generate(
            prompts, sampling_params=parameters, use_tqdm=False
        )
        if len(outputs) != len(slots):
            raise RuntimeError("vLLM candidate output count changed")
        for output, (request_index, candidate_index) in zip(outputs, slots):
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM candidate multiplicity changed")
            candidate = output.outputs[0]
            generated_tokens += len(candidate.token_ids)
            responses.append(make_response(
                requests[request_index],
                candidate_index,
                candidate.text,
                candidate.finish_reason,
            ))
        validate_candidate_responses(
            responses, requests, require_complete=False
        )
        _atomic_write(paths["partial"], _response_bytes(responses))
        prompt_tokens += sum(prompt_token_counts[batch_start:batch_end]) * 4
        telemetry = _telemetry(
            shard_index=args.shard_index,
            gpu_index=args.gpu_index,
            started_at=started_at,
            selected_requests=len(requests),
            completed_responses=len(responses),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            status="running",
        )
        telemetry["last_batch_seconds"] = time.monotonic() - started_batch
        _atomic_write(paths["telemetry"], canonical_bytes(telemetry))
        print(json.dumps({
            "event": "candidate_batch_complete",
            "shard": args.shard_index,
            "completed_requests": len(responses) // 4,
            "requests": len(requests),
            "generated_tokens_since_resume": generated_tokens,
            "at": utc_now(),
        }, sort_keys=True), flush=True)

    validate_candidate_responses(responses, requests)
    output_bytes = _response_bytes(responses)
    _atomic_write(paths["output"], output_bytes)
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    report = {
        "schema": "general-replay-candidate-response-shard-report-v1",
        "status": "complete",
        "smoke": args.smoke,
        "shard_index": args.shard_index,
        "physical_gpu_index": args.gpu_index,
        "pid": os.getpid(),
        "started_at": started_at,
        "completed_at": utc_now(),
        "requests": len(requests),
        "candidates_per_request": 4,
        "responses": len(responses),
        "request_shard_sha256": request_shard_sha256,
        "response_output": str(paths["output"]),
        "response_output_sha256": output_sha256,
        "model": request_report["model"],
        "model_files": model_hashes,
        "engine": {"name": "vllm", "version": VLLM_VERSION},
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_num_seqs": 64,
        "enable_prefix_caching": False,
        "official_chat_template_sha256": MODEL_FILE_SHA256[
            "chat_template.jinja"
        ],
        "template_parameters": {
            "add_generation_prompt": True,
            "enable_thinking": False,
        },
        "sampling_seeds_preserved": True,
        "hidden_verifier_targets_loaded": False,
        "response_schema": CANDIDATE_RESPONSE_SCHEMA,
        "resume_checkpoint": str(paths["partial"]),
        "atomic_final_output": True,
    }
    report["content_sha256_before_self_field"] = canonical_sha256(report)
    _atomic_write(paths["report"], (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"))
    _atomic_write(paths["telemetry"], canonical_bytes(_telemetry(
        shard_index=args.shard_index,
        gpu_index=args.gpu_index,
        started_at=started_at,
        selected_requests=len(requests),
        completed_responses=len(responses),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        status="complete",
    )))
    return {
        "status": "complete",
        "output": str(paths["output"]),
        "output_sha256": output_sha256,
        "report": str(paths["report"]),
        "requests": len(requests),
        "responses": len(responses),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--shard-index", required=True, type=int)
    result.add_argument("--gpu-index", required=True, type=int)
    result.add_argument("--model-directory", type=Path, default=MODEL_DIRECTORY)
    result.add_argument("--request-directory", type=Path, default=REQUEST_DIRECTORY)
    result.add_argument("--request-report", type=Path, default=REQUEST_REPORT)
    result.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    result.add_argument("--request-batch-size", type=int, default=8)
    result.add_argument("--max-model-len", type=int, default=8_192)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    result.add_argument("--enforce-eager", action="store_true")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--request-limit", type=int)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not 1 <= args.request_batch_size <= 64:
        raise ValueError("request batch size must be 1-64")
    if not 1_024 <= args.max_model_len <= 32_768:
        raise ValueError("sealed candidate model length must be 1024-32768")
    if not 0.5 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("GPU memory utilization must be 0.5-0.95")
    if args.smoke and args.request_limit not in {None, 1}:
        raise ValueError("candidate smoke is exactly one request")
    if args.gpu_index != args.shard_index:
        raise ValueError("one candidate shard is pinned to its matching GPU index")
    print(json.dumps(run(args), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
