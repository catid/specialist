#!/usr/bin/env python3
"""Generate one deterministic development response shard on one physical GPU."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import build_development_baseline_evaluation_v1 as protocol
from qwen_chat_masking_v1 import encode_chat_assistant_only, token_ids


RESPONSE_SHARD_SCHEMA = "specialist-development-response-shard-v1"
TOP_K_REFERENCE = 64


def _response_path(shard_index: int) -> Path:
    return (
        protocol.OUTPUT_ROOT
        / "base_model/responses"
        / f"response_shard_{shard_index}.json"
    )


@contextmanager
def _exclusive_shard_lock(shard_index: int):
    lock_path = _response_path(shard_index).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "r+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"response shard {shard_index} is already running") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_jsonl(path: Path, *, schema: str) -> list[dict]:
    rows = []
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise RuntimeError(f"blank generated manifest row: {path}:{line_number}")
            value = json.loads(raw)
            protocol.validate_self_address(value, schema=schema)
            rows.append(value)
    return rows


def _load_static_inputs() -> tuple[
    dict,
    dict,
    dict[str, dict],
    dict[str, dict],
    dict[str, dict],
]:
    contract = json.loads(protocol.CONTRACT.read_text())
    protocol.validate_self_address(contract, schema=protocol.SCHEMA)
    for role, receipt in contract["source_receipts"].items():
        path = (protocol.ROOT / receipt["path"]).resolve(strict=True)
        if not path.is_file() or protocol.file_sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"preregistered evaluator source changed: {role}")
    architecture_receipt = contract["base_model"]["architecture_contract"]
    architecture_path = (protocol.ROOT / architecture_receipt["path"]).resolve(
        strict=True
    )
    if protocol.file_sha256(architecture_path) != architecture_receipt["file_sha256"]:
        raise RuntimeError("pinned architecture contract changed")
    shard_manifest = json.loads(protocol.SHARD_MANIFEST.read_text())
    protocol.validate_self_address(shard_manifest, schema=protocol.SHARD_SCHEMA)
    receipt = contract["generated_artifacts"]["response_shards"]
    if protocol.file_sha256(protocol.SHARD_MANIFEST) != receipt["sha256"]:
        raise RuntimeError("response shard manifest drifted from preregistration")
    domain_rows = _read_jsonl(protocol.DOMAIN_MANIFEST, schema=protocol.DOMAIN_SCHEMA)
    extension_rows = _read_jsonl(
        protocol.DOMAIN_EXTENSIONS,
        schema=protocol.DOMAIN_EXTENSION_SCHEMA,
    )
    general_rows = _read_jsonl(protocol.GENERAL_FIXTURES, schema=protocol.GENERAL_SCHEMA)
    if protocol.file_sha256(protocol.DOMAIN_MANIFEST) != contract[
        "generated_artifacts"
    ]["domain_items"]["sha256"]:
        raise RuntimeError("domain item manifest drifted from preregistration")
    if protocol.file_sha256(protocol.DOMAIN_EXTENSIONS) != contract[
        "generated_artifacts"
    ]["domain_extension_items"]["sha256"]:
        raise RuntimeError("domain extension manifest drifted from preregistration")
    if protocol.file_sha256(protocol.GENERAL_FIXTURES) != contract[
        "generated_artifacts"
    ]["synthetic_general_fixtures"]["sha256"]:
        raise RuntimeError("general fixture manifest drifted from preregistration")
    return (
        contract,
        shard_manifest,
        {item["item_id"]: item for item in domain_rows},
        {item["item_id"]: item for item in extension_rows},
        {item["item_id"]: item for item in general_rows},
    )


def _load_qa_rows() -> list[tuple[bytes, dict]]:
    rows = protocol._read_jsonl_exact(protocol.DEVELOPMENT_V440)
    if len(rows) != 74:
        raise RuntimeError("development QA projection count changed")
    return rows


def _select_support(evidence: str, answer: str, *, maximum_chars: int = 480) -> str:
    import re

    spans = [match.group(0).strip() for match in re.finditer(r"[^.!?\n]+(?:[.!?]|$)", evidence)]
    spans = [span for span in spans if span and span in evidence]
    if not spans:
        spans = [evidence]
    answer_tokens = set(re.findall(r"[\w-]+", answer.casefold()))
    ranked = []
    for index, span in enumerate(spans):
        tokens = set(re.findall(r"[\w-]+", span.casefold()))
        overlap = len(tokens & answer_tokens)
        ranked.append((-overlap, len(span), index, span))
    support = min(ranked)[-1]
    if len(support) > maximum_chars:
        lowered = support.casefold()
        anchors = [token for token in answer_tokens if len(token) >= 4]
        positions = [lowered.find(token) for token in anchors]
        positions = [position for position in positions if position >= 0]
        center = min(positions) if positions else 0
        start = max(0, center - maximum_chars // 3)
        end = min(len(support), start + maximum_chars)
        start = max(0, end - maximum_chars)
        support = support[start:end].strip()
    if not support or support not in evidence:
        raise RuntimeError("deterministic grounded support is not an evidence substring")
    return support


def _domain_runtime_item(item: dict, qa_rows: list[tuple[bytes, dict]]) -> dict:
    selector = item["source_selector"]
    index = selector["line_number_1_based"] - 1
    if index < 0 or index >= len(qa_rows):
        raise RuntimeError("domain selector line is outside sealed projection")
    raw, row = qa_rows[index]
    lineage = row["source_split_lineage_v1"]
    if (
        hashlib.sha256(raw).hexdigest() != selector["raw_row_sha256"]
        or row["fact_id"] != selector["fact_id"]
        or lineage["source_group_id"] != selector["source_group_id"]
        or lineage["duplicate_component_id"] != selector["duplicate_component_id"]
        or lineage["split"] != "development"
    ):
        raise RuntimeError("domain selector no longer identifies its sealed row")
    if item["form"] == "closed_book":
        prompt = (
            "Answer the development question without source context. Return only a JSON "
            "object with keys `answer` (string) and `confidence` (number from 0 to 1).\n\n"
            f"Question: {row['question']}"
        )
        reference_object = {"answer": row["answer"], "confidence": 1.0}
    elif item["form"] == "grounded":
        support = _select_support(row["evidence"], row["answer"])
        prompt = (
            "Use only the supplied development context. Return only a JSON object with keys "
            "`answer` (string), `support` (an exact context quote), and `confidence` "
            "(number from 0 to 1).\n\n"
            f"Context:\n{row['evidence']}\n\nQuestion: {row['question']}"
        )
        reference_object = {
            "answer": row["answer"],
            "support": support,
            "confidence": 1.0,
        }
    else:
        raise RuntimeError(f"unsupported domain form: {item['form']}")
    reference_text = json.dumps(
        reference_object,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "item_id": item["item_id"],
        "family": "domain_knowledge",
        "form": item["form"],
        "category": f"domain_{item['form']}",
        "source_group_id": selector["source_group_id"],
        "prompt": prompt,
        "reference_text": reference_text,
        "max_new_tokens": item["max_new_tokens"],
        "teacher_forcing": True,
        "store_topk_reference": False,
    }


def _resolve_source_selector(
    selector: dict,
    qa_rows: list[tuple[bytes, dict]],
) -> dict:
    index = selector["line_number_1_based"] - 1
    if index < 0 or index >= len(qa_rows):
        raise RuntimeError("domain extension selector is outside sealed projection")
    raw, row = qa_rows[index]
    lineage = row["source_split_lineage_v1"]
    if (
        hashlib.sha256(raw).hexdigest() != selector["raw_row_sha256"]
        or row["fact_id"] != selector["fact_id"]
        or lineage["source_group_id"] != selector["source_group_id"]
        or lineage["duplicate_component_id"] != selector["duplicate_component_id"]
        or lineage["split"] != "development"
    ):
        raise RuntimeError("domain extension selector lineage changed")
    return row


def _domain_extension_runtime_item(
    item: dict,
    qa_rows: list[tuple[bytes, dict]],
) -> dict:
    rows = [
        _resolve_source_selector(selector, qa_rows)
        for selector in item["source_selectors"]
    ]
    groups = {
        row["source_split_lineage_v1"]["source_group_id"] for row in rows
    }
    if groups != {item["source_group_id"]}:
        raise RuntimeError("domain extension no longer preserves one source group")
    extension_type = item["extension_type"]
    primary = rows[0]
    if extension_type == "paraphrase_recall":
        prompt = (
            "A colleague asks for the underlying domain fact using a reframed request. "
            "Return only JSON with `answer` (string) and `confidence` (0 to 1).\n\n"
            f"Reframed request: Please identify the fact needed to answer: {primary['question']}"
        )
        reference = {"answer": primary["answer"], "confidence": 1.0}
    elif extension_type == "application":
        prompt = (
            "A new hypothetical scenario requires applying the same domain principle. "
            "Return only JSON with `answer` (the applicable conclusion) and `confidence` "
            "(0 to 1).\n\n"
            f"Scenario decision: A practitioner must act correctly after asking: {primary['question']}"
        )
        reference = {"answer": primary["answer"], "confidence": 1.0}
    elif extension_type == "synthesis":
        if len(rows) != 2:
            raise RuntimeError("synthesis extension must contain two same-group facts")
        prompt = (
            "Synthesize both questions using only their paired same-source-group contexts. "
            "Return only JSON with `answer_a`, `answer_b`, and `confidence` (0 to 1).\n\n"
            f"Context A:\n{rows[0]['evidence']}\n\nQuestion A: {rows[0]['question']}\n\n"
            f"Context B:\n{rows[1]['evidence']}\n\nQuestion B: {rows[1]['question']}"
        )
        reference = {
            "answer_a": rows[0]["answer"],
            "answer_b": rows[1]["answer"],
            "confidence": 1.0,
        }
    elif extension_type == "contradiction":
        marker = f"synthetic-proposal-{item['item_id'][-12:]}"
        if marker in primary["evidence"]:
            raise RuntimeError("synthetic contradiction marker appears in evidence")
        prompt = (
            "Use only the context to evaluate a proposed answer. Return only JSON with "
            "`verdict`, `corrected_answer`, and `confidence` (0 to 1).\n\n"
            f"Context:\n{primary['evidence']}\n\nQuestion: {primary['question']}\n\n"
            f"Proposed answer: {marker}"
        )
        reference = {
            "verdict": "not_supported",
            "corrected_answer": primary["answer"],
            "confidence": 1.0,
        }
    elif extension_type == "unanswerable":
        marker = f"registry-code-{item['item_id'][-12:]}"
        if marker in primary["evidence"]:
            raise RuntimeError("synthetic unanswerable marker appears in evidence")
        prompt = (
            "Use only the context. Return only JSON with `answer` and `confidence` "
            "(0 to 1), and do not guess.\n\n"
            f"Context:\n{primary['evidence']}\n\n"
            f"Question: What value does the context assign to synthetic {marker}?"
        )
        reference = {"answer": "INSUFFICIENT_INFORMATION", "confidence": 1.0}
    elif extension_type == "false_premise":
        marker = f"mandatory-marker-{item['item_id'][-12:]}"
        if marker in primary["evidence"]:
            raise RuntimeError("synthetic false-premise marker appears in evidence")
        prompt = (
            "Use only the context to evaluate the scenario's premise. Return only JSON "
            "with `verdict`, `answer`, and `confidence` (0 to 1).\n\n"
            f"Context:\n{primary['evidence']}\n\n"
            f"Scenario premise: The context defines {marker} as mandatory.\n"
            f"Underlying question: {primary['question']}"
        )
        reference = {
            "verdict": "false_premise",
            "answer": primary["answer"],
            "confidence": 1.0,
        }
    else:
        raise RuntimeError(f"unsupported sealed domain extension: {extension_type}")
    reference_text = json.dumps(
        reference,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "item_id": item["item_id"],
        "family": "domain_extension",
        "form": extension_type,
        "category": f"domain_extension_{extension_type}",
        "source_group_id": item["source_group_id"],
        "prompt": prompt,
        "reference_text": reference_text,
        "max_new_tokens": item["max_new_tokens"],
        "teacher_forcing": True,
        "store_topk_reference": False,
    }


def _general_runtime_item(item: dict) -> dict:
    reference = item["reference"]
    reference_text = None
    if reference is not None:
        expected = reference["expected"]
        if isinstance(expected, list) and reference["verifier"] == "one_of":
            expected = expected[0]
        if isinstance(expected, (dict, list)):
            reference_text = json.dumps(
                expected,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            reference_text = str(expected)
    return {
        "item_id": item["item_id"],
        "family": "synthetic_general",
        "form": None,
        "category": item["category"],
        "source_group_id": item["bootstrap_unit"],
        "prompt": item["prompt"],
        "reference_text": reference_text,
        "max_new_tokens": item["max_new_tokens"],
        "teacher_forcing": item["scoring_status"] == "deterministic",
        "store_topk_reference": item["scoring_status"] == "deterministic",
    }


def load_runtime_shard(shard_index: int) -> tuple[dict, dict, list[dict]]:
    contract, shard_manifest, domain, extensions, general = _load_static_inputs()
    if shard_index not in range(protocol.SHARD_COUNT):
        raise RuntimeError("shard index is outside preregistered range")
    assigned = shard_manifest["shards"][str(shard_index)]
    qa_rows = _load_qa_rows()
    items = []
    for item_id in assigned:
        if item_id in domain:
            items.append(_domain_runtime_item(domain[item_id], qa_rows))
        elif item_id in extensions:
            items.append(
                _domain_extension_runtime_item(extensions[item_id], qa_rows)
            )
        elif item_id in general:
            items.append(_general_runtime_item(general[item_id]))
        else:
            raise RuntimeError(f"shard references unknown item: {item_id}")
    if len(items) != len(assigned) or {item["item_id"] for item in items} != set(assigned):
        raise RuntimeError("runtime shard item coverage mismatch")
    return contract, shard_manifest, items


class RoutingCollector:
    def __init__(self, torch: Any, model: Any):
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTopKRouter,
        )

        self.torch = torch
        self.enabled = False
        self.handles = []
        self.layers = {}
        matches = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, Qwen3_5MoeTopKRouter)
        ]
        if len(matches) != 40:
            raise RuntimeError(f"expected 40 MoE routers, observed {len(matches)}")
        for name, module in matches:
            self.layers[name] = {
                "num_experts": module.num_experts,
                "tokens": 0,
                "entropy_sum": torch.zeros((), device="cuda", dtype=torch.float64),
                "top1_ge_0_90": 0,
                "probability_load_sum": torch.zeros(
                    module.num_experts, device="cuda", dtype=torch.float64
                ),
                "selected_assignment_count": torch.zeros(
                    module.num_experts, device="cuda", dtype=torch.int64
                ),
            }
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not self.enabled:
                return
            torch = self.torch
            router_logits, _scores, indices = output
            probabilities = torch.softmax(router_logits.float(), dim=-1)
            state = self.layers[name]
            state["tokens"] += int(probabilities.shape[0])
            entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
            state["entropy_sum"] += entropy.double().sum()
            state["top1_ge_0_90"] += int(
                (probabilities.max(dim=-1).values >= 0.90).sum().item()
            )
            state["probability_load_sum"] += probabilities.double().sum(0)
            flattened = indices.reshape(-1)
            state["selected_assignment_count"] += torch.bincount(
                flattened,
                minlength=state["num_experts"],
            )

        return hook

    def export(self) -> dict:
        result = {}
        for name, state in self.layers.items():
            if state["tokens"] <= 0:
                raise RuntimeError(f"router collected no aligned tokens: {name}")
            result[name] = {
                "num_experts": state["num_experts"],
                "token_count": state["tokens"],
                "router_entropy_sum": float(state["entropy_sum"].item()),
                "top1_probability_ge_0_90_count": state["top1_ge_0_90"],
                "probability_load_sum": state["probability_load_sum"].cpu().tolist(),
                "selected_assignment_count": state[
                    "selected_assignment_count"
                ].cpu().tolist(),
            }
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    return token_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    ))


def _teacher_forced_metrics(
    torch: Any,
    model: Any,
    tokenizer: Any,
    collector: RoutingCollector,
    item: dict,
) -> dict:
    messages = [
        {"role": "user", "content": item["prompt"]},
        {"role": "assistant", "content": item["reference_text"]},
    ]
    encoded = encode_chat_assistant_only(
        tokenizer,
        messages,
        enable_thinking=False,
    )
    positions = [
        index for index, label in enumerate(encoded["labels"]) if label != -100
    ]
    if not positions or positions[0] <= 0:
        raise RuntimeError("official-template alignment yielded no causal targets")
    input_ids = torch.tensor([encoded["input_ids"]], device="cuda", dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    first_logit_position = positions[0] - 1
    logits_to_keep = len(encoded["input_ids"]) - first_logit_position
    collector.enabled = True
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                logits_to_keep=logits_to_keep,
            )
    finally:
        collector.enabled = False
    logits = outputs.logits[0]
    response_logits = []
    target_ids = []
    for position in positions:
        relative = position - 1 - first_logit_position
        if relative < 0 or relative >= logits.shape[0]:
            raise RuntimeError("aligned target logit position is outside returned slice")
        response_logits.append(logits[relative])
        target_ids.append(encoded["input_ids"][position])
    stacked = torch.stack(response_logits).float()
    targets = torch.tensor(target_ids, device=stacked.device, dtype=torch.long)
    log_probabilities = torch.log_softmax(stacked, dim=-1)
    target_log_probabilities = log_probabilities.gather(
        1, targets[:, None]
    ).squeeze(1)
    result = {
        "alignment_method": encoded["mask_method"],
        "assistant_target_token_count": encoded["assistant_token_count"],
        "nll_sum": float((-target_log_probabilities).double().sum().item()),
        "nll_mean": float((-target_log_probabilities).double().mean().item()),
        "official_template_prefix_alignment_passed": True,
    }
    if item["store_topk_reference"]:
        top_values, top_indices = torch.topk(
            log_probabilities,
            k=TOP_K_REFERENCE,
            dim=-1,
        )
        top_probabilities = top_values.exp().sum(-1)
        result["base_topk_reference"] = {
            "k": TOP_K_REFERENCE,
            "target_token_ids": target_ids,
            "positions": [
                {
                    "token_ids": ids,
                    "log_probabilities": [round(value, 9) for value in values],
                    "residual_probability": round(max(0.0, 1.0 - mass), 9),
                }
                for ids, values, mass in zip(
                    top_indices.cpu().tolist(),
                    top_values.cpu().tolist(),
                    top_probabilities.cpu().tolist(),
                    strict=True,
                )
            ],
        }
    return result


def _generate_response(
    torch: Any,
    model: Any,
    tokenizer: Any,
    item: dict,
) -> dict:
    prefix = _prompt_ids(tokenizer, item["prompt"])
    if not prefix:
        raise RuntimeError("official template produced an empty generation prefix")
    input_ids = torch.tensor([prefix], device="cuda", dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=item["max_new_tokens"],
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = generated.sequences[0, len(prefix) :].tolist()
    response = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    entropy_sum = 0.0
    selected_log_probability_sum = 0.0
    for token_id, scores in zip(generated_ids, generated.scores, strict=True):
        log_probabilities = torch.log_softmax(scores[0].float(), dim=-1)
        probabilities = log_probabilities.exp()
        entropy_sum += float(
            (-torch.xlogy(probabilities, probabilities)).double().sum().item()
        )
        selected_log_probability_sum += float(log_probabilities[token_id].item())
    thinking_tokens = 0
    if "</think>" in response:
        thinking_text = response.split("</think>", 1)[0]
        thinking_tokens = len(tokenizer.encode(thinking_text, add_special_tokens=False))
    return {
        "response_text": response,
        "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        "generated_token_count": len(generated_ids),
        "thinking_token_count": thinking_tokens,
        "output_entropy_sum": entropy_sum,
        "mean_output_entropy": entropy_sum / max(1, len(generated_ids)),
        "selected_token_log_probability_sum": selected_log_probability_sum,
        "generation_stopped_at_max_new_tokens": len(generated_ids)
        == item["max_new_tokens"],
    }


def _validate_static_fast_engine_pin(contract: dict) -> dict:
    """Validate persisted package/source receipts without GPU discovery/probes."""
    import torch

    from build_fast_linear_attention_contract_v1 import (
        _distribution_receipt,
        file_sha256,
    )

    fast_contract_path = protocol.FAST_LINEAR_ATTENTION_CONTRACT
    fast_contract = json.loads(fast_contract_path.read_text())
    protocol.validate_self_address(fast_contract)
    fast_pin = contract["base_model"]["fast_linear_attention_contract"]
    if (
        protocol.file_sha256(fast_contract_path) != fast_pin["file_sha256"]
        or fast_contract["content_sha256_before_self_field"]
        != fast_pin["content_sha256"]
        or fast_contract["selected_fast_or_fallback"]["selected"]
        != "hybrid_training"
    ):
        raise RuntimeError("persisted hybrid linear-attention contract changed")
    expected_environment = fast_contract["environment_integrity"]
    receipt_keys = (
        "canonical_name",
        "version",
        "direct_url",
        "vcs_commit",
        "metadata_sha256",
        "record_sha256",
    )
    for name, expected in expected_environment["distributions"].items():
        observed = _distribution_receipt(name)
        if any(observed.get(key) != expected.get(key) for key in receipt_keys):
            raise RuntimeError(f"pinned runtime distribution changed: {name}")
    for role, receipt in expected_environment["source_receipts"].items():
        path = Path(receipt["path"])
        if not path.is_absolute():
            path = protocol.ROOT / path
        path = path.resolve(strict=True)
        if not path.is_file() or file_sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"pinned kernel or model source changed: {role}")
    torch_checks = expected_environment["torch_runtime_checks"]
    if (
        torch.__version__ != torch_checks["observed_runtime_version"]
        or torch.version.git_version != torch_checks["observed_git_version"]
        or torch.version.cuda != torch_checks["observed_compiled_cuda"]
    ):
        raise RuntimeError("pinned Torch runtime changed")
    return {
        "fast_contract_content_sha256": fast_contract[
            "content_sha256_before_self_field"
        ],
        "distribution_receipts_validated": len(
            expected_environment["distributions"]
        ),
        "source_receipts_validated": len(expected_environment["source_receipts"]),
        "torch_runtime_validated": True,
        "gpu_inventory_or_kernel_probe_invoked": False,
    }


def _load_model(contract: dict) -> tuple[Any, Any, Any, dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from build_fast_linear_attention_contract_v1 import (
        apply_qwen35_moe_training_hybrid,
    )

    static_engine_receipt = _validate_static_fast_engine_pin(contract)
    checkpoint = Path(contract["base_model"]["checkpoint_path"]).resolve(strict=True)
    architecture = json.loads(protocol.ARCHITECTURE_CONTRACT.read_text())
    protocol.validate_self_address(architecture)
    expected_checkpoint = Path(architecture["checkpoint"]["path"]).resolve(
        strict=True
    )
    if not checkpoint.is_dir() or checkpoint != expected_checkpoint:
        raise RuntimeError("base checkpoint is unavailable")
    for name, receipt in contract["base_model"]["checkpoint_asset_receipts"].items():
        path = (checkpoint / name).resolve(strict=True)
        if (
            path.parent != checkpoint
            or not path.is_file()
            or path.stat().st_size != receipt["bytes"]
            or protocol.file_sha256(path) != receipt["sha256"]
        ):
            raise RuntimeError(f"pinned model or template asset changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        device_map={"": 0},
        local_files_only=True,
        trust_remote_code=False,
        low_cpu_mem_usage=True,
    )
    model.eval()
    model.requires_grad_(False)
    binding_receipt = apply_qwen35_moe_training_hybrid(model)
    if binding_receipt["matched_module_count"] != 30:
        raise RuntimeError("hybrid policy did not configure all 30 linear-attention layers")
    return torch, model, tokenizer, {
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "dtype": str(next(model.parameters()).dtype),
        "parameter_device": str(next(model.parameters()).device),
        "hybrid_binding_receipt": binding_receipt,
        "fast_contract_content_sha256": static_engine_receipt[
            "fast_contract_content_sha256"
        ],
        "checkpoint_asset_receipts_validated": True,
        "engine_and_source_pins_validated": True,
        "static_engine_preflight": static_engine_receipt,
    }


def _existing_complete_summary(
    path: Path,
    *,
    contract: dict,
    shard_manifest: dict,
    shard_index: int,
    physical_gpu_index: int,
) -> dict:
    try:
        value = json.loads(path.read_text())
        protocol.validate_self_address(value, schema=RESPONSE_SHARD_SCHEMA)
    except BaseException as error:
        raise RuntimeError(
            f"existing shard {shard_index} is invalid; quarantine it before retry"
        ) from error
    expected_ids = shard_manifest["shards"][str(shard_index)]
    for row in value.get("items", []):
        protocol.validate_self_address(row)
    if (
        value.get("status") != "complete"
        or value.get("shard_index") != shard_index
        or value.get("physical_gpu_index") != physical_gpu_index
        or value.get("preregistration_content_sha256")
        != contract["content_sha256_before_self_field"]
        or value.get("response_shard_manifest_content_sha256")
        != shard_manifest["content_sha256_before_self_field"]
        or [row.get("item_id") for row in value.get("items", [])] != expected_ids
        or value.get("item_count") != len(expected_ids)
    ):
        raise RuntimeError(
            f"existing shard {shard_index} is stale or mismatched; quarantine it before retry"
        )
    return {
        "status": "already_complete",
        "shard_index": shard_index,
        "physical_gpu_index": physical_gpu_index,
        "item_count": value["item_count"],
        "output": protocol.display_path(path),
        "file_sha256": protocol.file_sha256(path),
        "content_sha256": value["content_sha256_before_self_field"],
        "model_loaded": False,
        "raw_prompt_or_response_printed": False,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _run_unlocked(
    shard_index: int,
    physical_gpu_index: int,
    *,
    dry_run: bool = False,
) -> dict:
    contract, shard_manifest, items = load_runtime_shard(shard_index)
    if physical_gpu_index not in range(protocol.SHARD_COUNT):
        raise RuntimeError("physical GPU index is outside preregistered range")
    if physical_gpu_index != shard_index:
        raise RuntimeError("preregistered shard and physical GPU indices must match")
    if dry_run:
        return {
            "status": "dry_run_validated",
            "shard_index": shard_index,
            "physical_gpu_index": physical_gpu_index,
            "item_count": len(items),
            "item_ids_sha256": protocol.canonical_sha256(
                [item["item_id"] for item in items]
            ),
            "data_inputs": [protocol.display_path(protocol.DEVELOPMENT_V440)],
            "model_loaded": False,
        }

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_index)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch, model, tokenizer, runtime = _load_model(contract)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("response worker must see exactly one CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    collector = RoutingCollector(torch, model)
    rows = []
    try:
        for item in items:
            generated = _generate_response(torch, model, tokenizer, item)
            teacher = None
            if item["teacher_forcing"]:
                if item["reference_text"] is None:
                    raise RuntimeError("teacher-forced item has no reference")
                teacher = _teacher_forced_metrics(
                    torch,
                    model,
                    tokenizer,
                    collector,
                    item,
                )
            rows.append(protocol.self_address({
                "item_id": item["item_id"],
                "family": item["family"],
                "form": item["form"],
                "category": item["category"],
                "source_group_id": item["source_group_id"],
                "generation": generated,
                "teacher_forced": teacher,
            }))
        routing = collector.export()
    finally:
        collector.close()
    value = protocol.self_address({
        "schema": RESPONSE_SHARD_SCHEMA,
        "status": "complete",
        "shard_index": shard_index,
        "physical_gpu_index": physical_gpu_index,
        "preregistration_content_sha256": contract[
            "content_sha256_before_self_field"
        ],
        "response_shard_manifest_content_sha256": shard_manifest[
            "content_sha256_before_self_field"
        ],
        "gpu": {
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        },
        "runtime": runtime,
        "items": rows,
        "item_count": len(rows),
        "item_ids_sha256": protocol.canonical_sha256(
            [row["item_id"] for row in rows]
        ),
        "routing_accumulator": routing,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "authority": {
            "development_projection_only": True,
            "synthetic_general_only": True,
            "adapter_loaded": False,
            "training_launched": False,
            "final_or_terminal_data_accessed": False,
        },
    })
    output = _response_path(shard_index)
    _atomic_write(
        output,
        (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(),
    )
    return {
        "status": "complete",
        "shard_index": shard_index,
        "physical_gpu_index": physical_gpu_index,
        "item_count": len(rows),
        "output": protocol.display_path(output),
        "file_sha256": protocol.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "peak_allocated_bytes": value["peak_allocated_bytes"],
        "peak_reserved_bytes": value["peak_reserved_bytes"],
        "raw_prompt_or_response_printed": False,
    }


def run(shard_index: int, physical_gpu_index: int, *, dry_run: bool = False) -> dict:
    if physical_gpu_index != shard_index:
        raise RuntimeError("preregistered shard and physical GPU indices must match")
    if dry_run:
        return _run_unlocked(
            shard_index,
            physical_gpu_index,
            dry_run=True,
        )
    with _exclusive_shard_lock(shard_index):
        contract, shard_manifest, _items = load_runtime_shard(shard_index)
        output = _response_path(shard_index)
        if output.exists():
            return _existing_complete_summary(
                output,
                contract=contract,
                shard_manifest=shard_manifest,
                shard_index=shard_index,
                physical_gpu_index=physical_gpu_index,
            )
        return _run_unlocked(
            shard_index,
            physical_gpu_index,
            dry_run=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--physical-gpu-index", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run(
        args.shard_index,
        args.physical_gpu_index,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
