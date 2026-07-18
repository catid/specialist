#!/usr/bin/env python3
"""Generate one train-only high-information candidate shard on one GPU.

The worker consumes only the content-addressed plan validated by
``verify_high_information_candidates_v1.load_plan``.  It does not open source
lineage paths or any development/final/protected dataset.  Parsed candidates
remain untrusted and ineligible for training until the separate structural and
semantic verifier stages complete.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Sequence

import build_high_information_domain_corpus_v1 as corpus
import verify_high_information_candidates_v1 as verifier


MODEL_DIRECTORY = corpus.ROOT / "models/Qwen3.6-35B-A3B"
MODEL_FILE_SHA256 = {
    "config.json": "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
    "model.safetensors.index.json": (
        "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
    ),
    "tokenizer.json": corpus.TOKENIZER_JSON_SHA256,
    "tokenizer_config.json": corpus.TOKENIZER_CONFIG_SHA256,
    "chat_template.jinja": (
        "e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259"
    ),
}
VLLM_VERSION = "0.25.0"
CANDIDATE_SCHEMA = "high-information-generated-candidate-v1"
HIDDEN_VERIFIER_KEYS = frozenset(
    {
        "expected_answer",
        "reference_answer",
        "semantic_verdict",
        "verifier_decision",
        "accepted_for_training",
    }
)
FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.IGNORECASE | re.DOTALL,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _payload(rows: Sequence[dict]) -> bytes:
    return corpus.jsonl_payload(rows)


def _contains_key(value: Any, keys: frozenset[str]) -> bool:
    if isinstance(value, dict):
        return bool(set(value) & keys) or any(
            _contains_key(child, keys) for child in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(child, keys) for child in value)
    return False


def validate_model_files(model_directory: Path) -> dict[str, str]:
    if model_directory.expanduser().resolve() != MODEL_DIRECTORY.resolve():
        raise RuntimeError("candidate worker requires the sealed local Qwen checkpoint")
    observed = {}
    for name, expected in MODEL_FILE_SHA256.items():
        path = model_directory / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"sealed model file is missing or aliased: {name}")
        digest = corpus.file_sha256(path)
        if digest != expected:
            raise RuntimeError(f"sealed model file changed: {name}")
        observed[name] = digest
    return observed


def request_seed(request_id: str) -> int:
    return int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16)


def max_new_tokens(request: dict) -> int:
    target = request["target_verified_assistant_tokens"]
    if not isinstance(target, int) or target <= 0:
        raise RuntimeError("generation request has an invalid token target")
    # JSON, questions, and exact evidence quotes are generation overhead and do
    # not count toward the eventual assistant-answer budget.
    return min(4096, max(768, target * 3 + 512))


def engine_kwargs(args: argparse.Namespace) -> dict[str, Any]:
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
        "enable_prefix_caching": False,
        "max_num_seqs": 64,
        "enforce_eager": args.enforce_eager,
        "generation_config": "vllm",
    }


def sampling_kwargs(request: dict) -> dict[str, Any]:
    return {
        "n": 1,
        "temperature": 0.3,
        "top_p": 0.9,
        "seed": request_seed(request["request_id"]),
        "max_tokens": max_new_tokens(request),
    }


def generation_messages(request: dict, context: dict) -> list[dict[str, str]]:
    negative = request["generation_mode"] == "calibrated_hard_negative"
    system = (
        "Create high-information supervised domain examples from the supplied "
        "train-only context. Return exactly one JSON object and no prose, code "
        "fence, analysis, scratchpad, or chain of thought. The object must contain "
        "an 'examples' array of exactly four objects. Every example object must "
        "have exactly these fields: example_type, question, answer, "
        "evidence_quotes, negative_type. Evidence quotes must be short exact "
        "substrings of the context. Never ask for or reproduce URLs, websites, "
        "page locations, or domain names. Preserve uncertainty, attribution, "
        "rights scope, and safety limitations. Do not infer procedures hidden in "
        "images or omitted steps. Make each question independently useful."
    )
    if negative:
        mode = (
            "All four examples must be calibrated hard negatives. Set example_type "
            "to 'calibrated_hard_negative'; choose negative_type from: "
            "answer_absent_from_context, false_premise, "
            "unsupported_precision_or_threshold, "
            "source_scope_or_authority_mismatch, "
            "conflicting_or_insufficient_evidence. Answers must directly state "
            "what cannot be established and must not invent a correction."
        )
    else:
        mode = (
            f"All four examples must be positive '{request['task_subtype']}' "
            "examples. Set example_type to that exact value and negative_type to "
            "null. Answers must be entailed by the context."
        )
    eventual = (
        "The eventual training prompt will include the context."
        if request["task_family"] == "grounded_synthesis"
        else "The eventual training prompt will not include the context."
    )
    user = (
        f"Task family: {request['task_family']}\n"
        f"Task subtype: {request['task_subtype']}\n"
        f"Mode: {request['generation_mode']}\n"
        f"Combined verified assistant-answer budget: approximately "
        f"{request['target_verified_assistant_tokens']} Qwen tokens.\n"
        f"{eventual}\n{mode}\n\n"
        "TRAIN-ONLY DOMAIN CONTEXT\n"
        "---\n"
        f"{context['text']}"
        "---\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def render_prompt(tokenizer: Any, request: dict, context: dict) -> tuple[str, int]:
    messages = generation_messages(request, context)
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("official template rendered an empty generation prompt")
    encoded = corpus.token_ids(tokenizer.encode(rendered, add_special_tokens=False))
    if not encoded:
        raise RuntimeError("official generation prompt has no tokens")
    return rendered, len(encoded)


def parse_examples(text: str) -> tuple[list[Any], str | None]:
    candidate = text.strip()
    match = FENCE_RE.fullmatch(candidate)
    if match:
        candidate = match.group(1).strip()
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError as error:
        return [], f"invalid_json:{error.msg}"
    if not isinstance(decoded, dict) or set(decoded) != {"examples"}:
        return [], "top_level_schema_changed"
    if not isinstance(decoded["examples"], list):
        return [], "examples_is_not_a_list"
    return decoded["examples"], None


def candidate_content_sha256(record: dict) -> str:
    unsigned = dict(record)
    unsigned.pop("content_sha256_before_self_field", None)
    return corpus.canonical_sha256(unsigned)


def make_candidate_record(
    request: dict,
    text: str,
    *,
    finish_reason: str | None,
    generated_token_count: int,
) -> dict:
    examples, parse_error = parse_examples(text)
    record = {
        "schema": CANDIDATE_SCHEMA,
        "request_id": request["request_id"],
        "source_group_id": request["source_group_id"],
        "gpu_shard": request["gpu_shard"],
        "examples": examples,
        "parse_status": "parsed_unverified" if parse_error is None else "rejected_parse",
        "parse_error": parse_error,
        "raw_completion_sha256": corpus.sha256_bytes(text.encode("utf-8")),
        "finish_reason": finish_reason,
        "generated_token_count": generated_token_count,
        "generator": {
            "model": "Qwen3.6-35B-A3B",
            "checkpoint": "sealed_local_base",
            "engine": "vllm-0.25.0",
            "dtype": "bfloat16",
            "temperature": 0.3,
            "top_p": 0.9,
            "seed": request_seed(request["request_id"]),
            "enable_thinking": False,
        },
        "semantic_verification_completed": False,
        "eligible_for_training": False,
    }
    record["content_sha256_before_self_field"] = candidate_content_sha256(record)
    return record


def validate_candidate_record(record: dict, request: dict) -> None:
    if (
        not isinstance(record, dict)
        or record.get("schema") != CANDIDATE_SCHEMA
        or record.get("request_id") != request["request_id"]
        or record.get("source_group_id") != request["source_group_id"]
        or record.get("gpu_shard") != request["gpu_shard"]
        or record.get("semantic_verification_completed") is not False
        or record.get("eligible_for_training") is not False
        or record.get("content_sha256_before_self_field")
        != candidate_content_sha256(record)
    ):
        raise RuntimeError("generated candidate record identity changed")


def shard_paths(shard_index: int, *, smoke: bool) -> dict[str, Path]:
    suffix = ".smoke" if smoke else ""
    stem = f"generation_candidates_gpu{shard_index}{suffix}"
    return {
        "partial": corpus.OUTPUT_DIR / f"{stem}.partial.jsonl",
        "output": corpus.OUTPUT_DIR / f"{stem}.jsonl",
        "report": corpus.OUTPUT_DIR / f"{stem}.report.json",
        "telemetry": corpus.OUTPUT_DIR / f"{stem}.telemetry.json",
    }


def load_partial(path: Path, requests: Sequence[dict]) -> list[dict]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise RuntimeError("candidate partial output is not a regular file")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) > len(requests):
        raise RuntimeError("candidate partial output exceeds its request shard")
    for row, request in zip(rows, requests, strict=False):
        validate_candidate_record(row, request)
    return rows


def _validate_self_address(value: dict, *, label: str) -> None:
    unsigned = dict(value)
    declared = unsigned.pop("content_sha256_before_self_field", None)
    if corpus.canonical_sha256(unsigned) != declared:
        raise RuntimeError(f"{label} content address changed")


def load_complete(paths: dict[str, Path], requests: Sequence[dict]) -> list[dict] | None:
    output_exists = paths["output"].exists()
    report_exists = paths["report"].exists()
    if not output_exists and not report_exists:
        return None
    if output_exists != report_exists:
        raise RuntimeError("candidate final output/report pair is incomplete")
    if any(path.is_symlink() or not path.is_file() for path in (paths["output"], paths["report"])):
        raise RuntimeError("candidate final output/report is not a regular file")
    rows = load_partial(paths["output"], requests)
    if len(rows) != len(requests):
        raise RuntimeError("candidate final output is incomplete")
    payload = paths["output"].read_bytes()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    _validate_self_address(report, label="candidate shard report")
    if (
        report.get("schema") != "high-information-candidate-shard-report-v1"
        or report.get("status") != "complete_unverified"
        or report.get("gpu_shard") != requests[0]["gpu_shard"]
        or report.get("requests") != len(requests)
        or report.get("candidate_records") != len(rows)
        or report.get("output") != corpus.relative(paths["output"])
        or report.get("output_sha256") != corpus.sha256_bytes(payload)
        or report.get("worker_file_sha256")
        != corpus.file_sha256(Path(__file__).resolve())
        or report.get("semantic_verification_completed") is not False
        or report.get("training_rows_emitted") is not False
    ):
        raise RuntimeError("candidate complete output/report contract changed")
    return rows


def _telemetry(
    *, shard_index: int, gpu_index: int, status: str, requests: int,
    completed: int, prompt_tokens: int, generated_tokens: int,
) -> dict:
    return {
        "schema": "high-information-candidate-worker-telemetry-v1",
        "updated_at": utc_now(),
        "status": status,
        "pid": os.getpid(),
        "gpu_shard": shard_index,
        "physical_gpu_index": gpu_index,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "requests": requests,
        "completed_requests": completed,
        "prompt_tokens": prompt_tokens,
        "generated_tokens_since_resume": generated_tokens,
    }


def load_shard_plan(shard_index: int) -> tuple[dict[str, dict], list[dict]]:
    _, contexts, request_index = verifier.load_plan()
    requests = sorted(
        (
            row for row in request_index.values()
            if row["gpu_shard"] == shard_index
        ),
        key=lambda row: row["request_id"],
    )
    if not requests or any(row["gpu_shard"] != shard_index for row in requests):
        raise RuntimeError("generation shard is empty or has crossed GPU lineage")
    if any(_contains_key(row, HIDDEN_VERIFIER_KEYS) for row in requests):
        raise RuntimeError("generation request shard contains a hidden verifier target")
    return contexts, requests


def preflight(args: argparse.Namespace) -> dict:
    validate_model_files(args.model_directory)
    contexts, requests = load_shard_plan(args.shard_index)
    tokenizer = corpus.load_tokenizer()
    prompt_counts = []
    total_budgets = []
    for request in requests:
        _, prompt_count = render_prompt(
            tokenizer,
            request,
            contexts[request["source_context_id"]],
        )
        budget = max_new_tokens(request)
        if prompt_count + budget > args.max_model_len:
            raise RuntimeError(
                f"generation request exceeds max model length: {request['request_id']}"
            )
        prompt_counts.append(prompt_count)
        total_budgets.append(prompt_count + budget)
    return {
        "status": "preflight_complete_no_gpu_launch",
        "gpu_shard": args.shard_index,
        "requests": len(requests),
        "max_prompt_tokens": max(prompt_counts),
        "max_prompt_plus_generation_budget": max(total_budgets),
        "max_model_len": args.max_model_len,
        "vllm_imported": False,
        "generation_launched": False,
    }


def run(args: argparse.Namespace) -> dict:
    started_at = utc_now()
    model_hashes = validate_model_files(args.model_directory)
    contexts, requests = load_shard_plan(args.shard_index)
    if args.smoke:
        requests = requests[:1]
    paths = shard_paths(args.shard_index, smoke=args.smoke)
    complete = load_complete(paths, requests)
    if complete is not None:
        return {
            "status": "already_complete",
            "rows": len(complete),
            "output": str(paths["output"]),
        }

    tokenizer = corpus.load_tokenizer()
    prompts = []
    prompt_token_counts = []
    for request in requests:
        prompt, count = render_prompt(
            tokenizer,
            request,
            contexts[request["source_context_id"]],
        )
        if count + max_new_tokens(request) > args.max_model_len:
            raise RuntimeError(
                f"generation request exceeds max model length: {request['request_id']}"
            )
        prompts.append(prompt)
        prompt_token_counts.append(count)

    from vllm import LLM, SamplingParams, __version__ as vllm_version

    if vllm_version != VLLM_VERSION:
        raise RuntimeError("candidate worker requires vLLM 0.25.0 exactly")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_index):
        raise RuntimeError("candidate worker requires exactly its assigned physical GPU")
    engine = LLM(**engine_kwargs(args))
    rows = load_partial(paths["partial"], requests)
    completed = len(rows)
    generated_tokens = 0
    prompt_tokens = sum(prompt_token_counts[:completed])
    for batch_start in range(completed, len(requests), args.request_batch_size):
        batch_end = min(batch_start + args.request_batch_size, len(requests))
        parameters = [
            SamplingParams(**sampling_kwargs(request))
            for request in requests[batch_start:batch_end]
        ]
        batch_started = time.monotonic()
        outputs = engine.generate(
            prompts[batch_start:batch_end],
            sampling_params=parameters,
            use_tqdm=False,
        )
        if len(outputs) != batch_end - batch_start:
            raise RuntimeError("vLLM candidate output count changed")
        for output, request in zip(outputs, requests[batch_start:batch_end], strict=True):
            if len(output.outputs) != 1:
                raise RuntimeError("vLLM candidate multiplicity changed")
            generated = output.outputs[0]
            generated_tokens += len(generated.token_ids)
            row = make_candidate_record(
                request,
                generated.text,
                finish_reason=generated.finish_reason,
                generated_token_count=len(generated.token_ids),
            )
            validate_candidate_record(row, request)
            rows.append(row)
        _atomic_write(paths["partial"], _payload(rows))
        prompt_tokens += sum(prompt_token_counts[batch_start:batch_end])
        telemetry = _telemetry(
            shard_index=args.shard_index,
            gpu_index=args.gpu_index,
            status="running",
            requests=len(requests),
            completed=len(rows),
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )
        telemetry["last_batch_seconds"] = time.monotonic() - batch_started
        _atomic_write(paths["telemetry"], corpus.canonical_bytes(telemetry))
        print(
            json.dumps(
                {
                    "event": "domain_candidate_batch_complete",
                    "gpu_shard": args.shard_index,
                    "completed_requests": len(rows),
                    "requests": len(requests),
                    "at": utc_now(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    output_payload = _payload(rows)
    _atomic_write(paths["output"], output_payload)
    report = {
        "schema": "high-information-candidate-shard-report-v1",
        "status": "complete_unverified",
        "started_at": started_at,
        "completed_at": utc_now(),
        "gpu_shard": args.shard_index,
        "physical_gpu_index": args.gpu_index,
        "requests": len(requests),
        "candidate_records": len(rows),
        "parsed_records": sum(row["parse_error"] is None for row in rows),
        "output": corpus.relative(paths["output"]),
        "output_sha256": corpus.sha256_bytes(output_payload),
        "model_files": model_hashes,
        "vllm_version": VLLM_VERSION,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "worker_file_sha256": corpus.file_sha256(Path(__file__).resolve()),
        "generation_config": "vllm",
        "semantic_verification_completed": False,
        "training_rows_emitted": False,
    }
    report["content_sha256_before_self_field"] = corpus.canonical_sha256(report)
    _atomic_write(paths["report"], (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8"))
    _atomic_write(paths["telemetry"], corpus.canonical_bytes(_telemetry(
        shard_index=args.shard_index,
        gpu_index=args.gpu_index,
        status="complete_unverified",
        requests=len(requests),
        completed=len(rows),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
    )))
    return report


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--shard-index", required=True, type=int)
    result.add_argument("--gpu-index", required=True, type=int)
    result.add_argument("--model-directory", type=Path, default=MODEL_DIRECTORY)
    result.add_argument("--request-batch-size", type=int, default=8)
    result.add_argument("--max-model-len", type=int, default=16_384)
    result.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    result.add_argument("--enforce-eager", action="store_true")
    result.add_argument("--smoke", action="store_true")
    result.add_argument("--check-plan", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.shard_index not in range(4) or args.gpu_index != args.shard_index:
        raise ValueError("candidate shard 0-3 must be pinned to the matching GPU")
    if not 1 <= args.request_batch_size <= 32:
        raise ValueError("request batch size must be 1-32")
    if not 12_288 <= args.max_model_len <= 32_768:
        raise ValueError("max model length must be 12288-32768")
    if not 0.5 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("GPU memory utilization must be 0.5-0.95")
    result = preflight(args) if args.check_plan else run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
