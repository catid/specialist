#!/usr/bin/env python3
"""Reproduce the V63 candidate namespace failure without opening any data.

This audit is deliberately limited to immutable adapter metadata, safetensor
key names, the installed vLLM name mapper/activation implementation, and the
already-finalized V63 aggregate artifacts.  It never opens prompts, answers,
generations, train rows, or any protected evaluation material.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parent
RAW_CANDIDATE = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/v59_lora_es_fragile_priority/"
    "selected_candidate_v59"
).resolve()
STAGED_CANDIDATE = (
    ROOT
    / "experiments/eggroll_es_hpo/staged_adapters/"
    "v59_fragile_priority_ratio0p25_lora_es_qwen35_vllm_namespace_v60"
).resolve()
REFERENCE = (
    ROOT
    / "experiments/eggroll_es_hpo/staged_adapters/"
    "v434_equal_sft_qwen35_vllm_namespace_v49d"
).resolve()
V63_RUN = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/"
    "v63_v59_vs_v434_train_only_robust_confirmation"
).resolve()
OUTPUT = (
    ROOT / "experiments/eval_reports/v63_candidate_namespace_noop_diagnosis_v64.json"
).resolve()

EXPECTED_FILES = {
    "raw_candidate_weights": (
        RAW_CANDIDATE / "adapter_model.safetensors",
        "c2665b60928b16120a2b98fdf137fafd250644852c86a02d797689f02105c6c8",
    ),
    "raw_candidate_config": (
        RAW_CANDIDATE / "adapter_config.json",
        "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    ),
    "staged_candidate_weights": (
        STAGED_CANDIDATE / "adapter_model.safetensors",
        "e189cfb9fcd6c55700babae91111266825522fc46ddbd53fc0574786711eccae",
    ),
    "staged_candidate_config": (
        STAGED_CANDIDATE / "adapter_config.json",
        "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    ),
    "staged_candidate_manifest": (
        STAGED_CANDIDATE / "stage_manifest_v44a.json",
        "c0e8c322f8c373dadd66506a2ad3859b7f9f18e56e56ff7b387999fb060f3d0e",
    ),
    "reference_weights": (
        REFERENCE / "adapter_model.safetensors",
        "7a41d921c6988dc62dca092230ed5ccfd5d6568a600503c87ff086cb2763485a",
    ),
    "reference_config": (
        REFERENCE / "adapter_config.json",
        "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    ),
    "v63_preregistration": (
        ROOT
        / "experiments/eggroll_es_hpo/preregistrations/"
        "v59_vs_v434_train_only_robust_confirmation_v63.json",
        "79327aae4fa42ff8741d988f2ad07252eb66911fdf5fee433a3188fb2e6f7e7b",
    ),
    "v63_evidence": (
        V63_RUN / "confirmation_evidence_v63.json",
        "16fde944611c3d2b862a78549a23b10f9780109376716b5ea91511d05b60f17a",
    ),
    "v63_analysis": (
        V63_RUN / "confirmation_analysis_v63.json",
        "6c9a566c5f6a3f20c24078165609d555ff5c899089a9942823fe5ebcc68d2f15",
    ),
    "v63_report": (
        V63_RUN / "confirmation_report_v63.json",
        "7a668e8ba9da84ab531b8f58bfacbeb96e27261ae6012219788699d2c1695d78",
    ),
    "v63_finalized": (
        V63_RUN / "confirmation_finalized_v63.json",
        "6c165dcbc043c9ce5a510369df0b6511e8ab57b51615278b1e6e993c64f8faea",
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _mapped_modules(weights: Path, mapper) -> tuple[list[str], int]:
    from vllm.lora.utils import parse_fine_tuned_lora_name

    with safe_open(weights, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    modules = sorted(
        {
            parse_fine_tuned_lora_name(key, mapper)[0]
            for key in keys
        }
    )
    return modules, len(keys)


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_diagnosis() -> dict:
    observed_files = {
        name: file_sha256(path)
        for name, (path, _) in EXPECTED_FILES.items()
    }
    expected_files = {
        name: digest for name, (_, digest) in EXPECTED_FILES.items()
    }
    if observed_files != expected_files:
        raise RuntimeError("V63 diagnosis input bytes changed")

    from vllm.lora.model_manager import LoRAModelManager
    from vllm.lora.worker_manager import WorkerLoRAManager
    from vllm.model_executor.models.qwen3_5 import (
        Qwen3_5MoeForConditionalGeneration,
    )

    mapper = (
        Qwen3_5MoeForConditionalGeneration.hf_to_vllm_mapper
        .get_unstacked_mapper()
    )
    expected_prefix_map = {
        "model.visual.": "visual.",
        "lm_head.": "language_model.lm_head.",
        "model.language_model.": "language_model.model.",
    }
    if (
        mapper.orig_to_new_prefix != expected_prefix_map
        or mapper.orig_to_new_regex
        or mapper.orig_to_new_substr
        or mapper.orig_to_new_stacked
        or mapper.orig_to_new_suffix
    ):
        raise RuntimeError("installed Qwen3.5 unstacked mapper changed")

    reference, reference_tensor_count = _mapped_modules(
        EXPECTED_FILES["reference_weights"][0], mapper
    )
    raw, raw_tensor_count = _mapped_modules(
        EXPECTED_FILES["raw_candidate_weights"][0], mapper
    )
    staged, staged_tensor_count = _mapped_modules(
        EXPECTED_FILES["staged_candidate_weights"][0], mapper
    )
    reference_set, raw_set, staged_set = set(reference), set(raw), set(staged)
    if not (
        reference_tensor_count == raw_tensor_count == staged_tensor_count == 70
        and len(reference) == len(raw) == len(staged) == 35
        and all(name.startswith("language_model.model.layers.") for name in reference)
        and all(name.startswith("model.layers.") for name in raw)
        and not (reference_set & raw_set)
        and staged_set == reference_set
    ):
        raise RuntimeError("V63 adapter namespace diagnosis no longer reproduces")

    activation_source = inspect.getsource(LoRAModelManager.activate_adapter)
    loader_source = inspect.getsource(WorkerLoRAManager._load_adapter)
    if not all(
        token in activation_source
        for token in (
            "for module_name, module in self.modules.items()",
            "module_lora = self._get_lora_layer_weights",
            "module.reset_lora(index)",
            "module.set_lora",
        )
    ) or not all(
        token in loader_source
        for token in (
            "hf_to_vllm_mapper.get_unstacked_mapper()",
            "weights_mapper=hf_to_vllm_mapper",
        )
    ):
        raise RuntimeError("installed vLLM load/activation mechanism changed")

    stage_manifest = json.loads(
        EXPECTED_FILES["staged_candidate_manifest"][0].read_text(encoding="utf-8")
    )
    if (
        stage_manifest.get("content_sha256_before_self_field")
        != "4b56da407f9a764157e6c9970ad1cc0b3695ad8e6ea9ee691cf73e274ba26bf6"
        or stage_manifest.get("transformed_identity", {}).get("sha256")
        != "17426568952a5d23490886db52f9b2f8e648ebdf267b98789616dceb31d1a6c6"
        or stage_manifest.get("transformed_identity", {}).get(
            "all_tensor_bytes_preserved_exactly"
        ) is not True
    ):
        raise RuntimeError("V59 staged transform provenance changed")

    installed_sources = {}
    for name, obj in {
        "vllm_lora_model_manager": LoRAModelManager,
        "vllm_lora_worker_manager": WorkerLoRAManager,
        "vllm_qwen35_model": Qwen3_5MoeForConditionalGeneration,
    }.items():
        path = Path(inspect.getsourcefile(obj) or "").resolve()
        installed_sources[name] = {
            "path": str(path),
            "file_sha256": file_sha256(path),
        }

    result = {
        "schema": "v63-candidate-namespace-noop-diagnosis-v64",
        "status": "reproduced_exact_serving_namespace_failure",
        "scope": {
            "sealed_v63_files_modified": False,
            "semantic_dataset_prompt_answer_or_generation_opened": False,
            "protected_ood_shadow_holdout_or_terminal_material_opened": False,
            "base_model_loaded": False,
            "cuda_compute_initialized": bool(torch.cuda.is_initialized()),
            "gpu_accessed_for_compute": False,
        },
        "exact_input_file_sha256": observed_files,
        "installed_implementation": installed_sources,
        "mapper": {
            "unstacked_prefix_map": expected_prefix_map,
            "raw_candidate_tensor_count": raw_tensor_count,
            "raw_candidate_mapped_module_count": len(raw),
            "raw_candidate_mapped_module_manifest_sha256": canonical_sha256(raw),
            "reference_tensor_count": reference_tensor_count,
            "reference_mapped_module_count": len(reference),
            "reference_mapped_module_manifest_sha256": canonical_sha256(reference),
            "staged_candidate_tensor_count": staged_tensor_count,
            "staged_candidate_mapped_module_count": len(staged),
            "staged_candidate_mapped_module_manifest_sha256": canonical_sha256(staged),
            "raw_candidate_reference_module_intersection_count": len(
                raw_set & reference_set
            ),
            "staged_candidate_reference_module_intersection_count": len(
                staged_set & reference_set
            ),
            "staged_candidate_module_set_equals_reference": staged_set == reference_set,
        },
        "mechanism": {
            "v63_requested_raw_candidate_directory": str(RAW_CANDIDATE),
            "correct_staged_candidate_directory": str(STAGED_CANDIDATE),
            "activation_records_id_before_module_weight_lookup": True,
            "activation_resets_live_module_slot_when_lookup_returns_none": True,
            "v63_active_id_receipt_was_insufficient_to_prove_weight_application": True,
            "raw_candidate_had_zero_expected_live_module_name_overlap": True,
            "correct_stage_has_exact_35_of_35_expected_module_name_overlap": True,
        },
        "interpretation": {
            "v63_negative_result_is_valid_as_a_serving_export_failure_finding": True,
            "v63_negative_result_is_not_evidence_against_the_v59_eggroll_es_weights": True,
            "corrected_confirmation_requires_new_preregistration": True,
            "corrected_confirmation_must_require_live_applied_weight_coverage_receipts": True,
        },
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def main() -> int:
    result = build_diagnosis()
    _atomic_json(OUTPUT, result)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "file_sha256": file_sha256(OUTPUT),
                "content_sha256": result["content_sha256_before_self_field"],
                "status": result["status"],
                "cuda_compute_initialized": result["scope"][
                    "cuda_compute_initialized"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
