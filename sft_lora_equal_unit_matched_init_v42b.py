#!/usr/bin/env python3
"""V42B retry: exact matched LoRA initialization without PEFT load conversion.

PEFT 0.19.1's Transformers-v5 adapter-load conversion is incompatible with
the installed Transformers ``WeightConverter`` constructor for Qwen3.5 MoE.
V42B therefore constructs the already-audited V37A LoRA surface with
``get_peft_model`` and copies the canonical 70-tensor FP32 state directly into
the canonical adapter state views.  An immediate readback must match every
source tensor before the V42A train-only implementation may continue.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file

import sft_lora_equal_unit_matched_init_v42a as v42a


def _canonical_adapter_state_v42b(model) -> dict[str, torch.Tensor]:
    return get_peft_model_state_dict(
        model, adapter_name="default", save_embedding_layers=False
    )


def _loader_audit_from_records_v42b(records: list[dict]) -> dict:
    return {
        "schema": "specialist-direct-canonical-lora-load-v42b",
        "verified": True,
        "loader": "get_peft_model_plus_inplace_canonical_state_copy",
        "peft_pretrained_conversion_invoked": False,
        "set_peft_model_state_dict_invoked": False,
        "tensor_count": len(records),
        "elements": sum(item["elements"] for item in records),
        "dtype": "torch.float32",
        "source_weights_file_sha256": v42a.INITIAL_WEIGHTS_SHA256_V42A,
        "source_tensor_identity_sha256": (
            v42a.INITIAL_TENSOR_IDENTITY_SHA256_V42A
        ),
        "readback_records_sha256": v42a.canonical_sha256_v42a(records),
        "peft_version": peft.__version__,
        "transformers_version": transformers.__version__,
        "compatibility_reason": (
            "bypass PEFT Transformers-v5 WeightConverter construction because "
            "the installed constructor rejects distributed_operation"
        ),
    }


def expected_loader_audit_v42b() -> dict:
    v42a.validate_initialization_artifact_v42a(v42a.INITIAL_ADAPTER_V42A)
    source = load_file(
        str(v42a.INITIAL_ADAPTER_V42A / "adapter_model.safetensors"),
        device="cpu",
    )
    records = [{
        "key": key,
        "shape": list(source[key].shape),
        "dtype": str(source[key].dtype),
        "elements": source[key].numel(),
        "tensor_sha256": v42a.tensor_sha256_v42a(source[key]),
    } for key in sorted(source)]
    result = _loader_audit_from_records_v42b(records)
    if result["tensor_count"] != 70 or result["elements"] != 4_528_128:
        raise RuntimeError("V42B expected canonical adapter aggregate changed")
    return result


def copy_exact_canonical_state_v42b(
    model,
    source_state: Mapping[str, torch.Tensor],
) -> dict:
    """Copy canonical state views in-place and fail unless readback is exact."""
    runtime = _canonical_adapter_state_v42b(model)
    if set(runtime) != set(source_state):
        missing = sorted(set(source_state) - set(runtime))
        unexpected = sorted(set(runtime) - set(source_state))
        raise RuntimeError(
            "V42B canonical adapter inventory mismatch: "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    pointers = {}
    elements = 0
    with torch.no_grad():
        for key in sorted(source_state):
            source = source_state[key]
            target = runtime[key]
            if (
                source.dtype != torch.float32
                or target.dtype != torch.float32
                or source.shape != target.shape
                or not target.is_contiguous()
            ):
                raise RuntimeError(f"V42B canonical adapter metadata mismatch: {key}")
            pointers[key] = target.data_ptr()
            target.copy_(source.to(device=target.device), non_blocking=False)
            elements += target.numel()

    readback = _canonical_adapter_state_v42b(model)
    if set(readback) != set(source_state):
        raise RuntimeError("V42B canonical adapter inventory changed on readback")
    records = []
    for key in sorted(source_state):
        actual = readback[key]
        expected = source_state[key]
        if (
            actual.data_ptr() != pointers[key]
            or actual.dtype != torch.float32
            or actual.shape != expected.shape
            or not torch.equal(actual.detach().to(device="cpu"), expected)
        ):
            raise RuntimeError(f"V42B canonical adapter readback mismatch: {key}")
        records.append({
            "key": key,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "elements": actual.numel(),
            "tensor_sha256": v42a.tensor_sha256_v42a(actual),
        })
    if len(records) != 70 or elements != 4_528_128:
        raise RuntimeError("V42B canonical adapter aggregate changed")
    result = _loader_audit_from_records_v42b(records)
    if result != expected_loader_audit_v42b():
        raise RuntimeError("V42B runtime/source loader audit consensus changed")
    return result


class ExactCanonicalAdapterLoaderV42B:
    """Drop-in replacement for the one V42A ``from_pretrained`` call."""

    @classmethod
    def from_pretrained(
        cls,
        base_model,
        model_id,
        adapter_name: str = "default",
        is_trainable: bool = False,
        autocast_adapter_dtype: bool = True,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(f"V42B refuses unregistered PEFT load options: {kwargs}")
        directory = Path(model_id).resolve()
        if (
            directory != v42a.INITIAL_ADAPTER_V42A
            or adapter_name != "default"
            or is_trainable is not True
            or autocast_adapter_dtype is not True
        ):
            raise ValueError("V42B exact canonical adapter load contract changed")
        source_audit = v42a.validate_initialization_artifact_v42a(directory)
        config = LoraConfig.from_pretrained(directory)
        config.inference_mode = False
        model = get_peft_model(
            base_model,
            config,
            adapter_name=adapter_name,
            autocast_adapter_dtype=True,
        )
        source_state = load_file(
            str(directory / "adapter_model.safetensors"), device="cpu"
        )
        loader_audit = copy_exact_canonical_state_v42b(model, source_state)
        if (
            loader_audit["source_weights_file_sha256"]
            != source_audit["weights_file_sha256"]
            or loader_audit["source_tensor_identity_sha256"]
            != source_audit["tensor_identity_sha256"]
        ):
            raise RuntimeError("V42B loader/source identity consensus changed")
        print(
            json.dumps(
                {"initialization_loader_audit_v42b": loader_audit},
                sort_keys=True,
            ),
            flush=True,
        )
        return model


def main(argv: list[str] | None = None) -> None:
    original_loader = v42a.PeftModel
    v42a.PeftModel = ExactCanonicalAdapterLoaderV42B
    try:
        v42a.main(argv)
    finally:
        v42a.PeftModel = original_loader


if __name__ == "__main__":
    main()
