#!/usr/bin/env python3
"""Stage the sealed V42B/C/D SFT and V43D LoRA-ES adapters for vLLM.

Only canonical PEFT tensor keys are changed, from Qwen3.5's inner
``model.layers`` namespace to vLLM's outer ``model.language_model.layers``
namespace.  All 70 FP32 tensor value byte sequences and each adapter config
are preserved exactly.  No model, tokenizer, dataset, evaluation file, or GPU
is opened by this CPU-only utility.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import run_sft_train_only_control_v36a as hashing


ROOT = Path(__file__).resolve().parent
SOURCE_PREFIX_V44A = "base_model.model.model.layers."
TARGET_PREFIX_V44A = "base_model.model.model.language_model.layers."
EXPECTED_TENSOR_COUNT_V44A = 70
EXPECTED_ELEMENTS_V44A = 4_528_128

SFT_SOURCE_V44A = (
    ROOT / "experiments/sft_controls/"
    "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load/"
    "middle_late_r32_seed17_init20260715041/final"
).resolve()
SFT_REPORT_V44A = (
    ROOT / "experiments/sft_controls/"
    "v42b_matched_init_equal_unit_fold3_v412_retry_direct_load/"
    "runtime_report_v42b.json"
).resolve()
SFT_OUTPUT_V44A = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v42b_sft_qwen35_vllm_namespace_v44a_all_candidates"
).resolve()

SFTC_SOURCE_V44A = (
    ROOT / "experiments/sft_controls/"
    "v42c_matched_init_equal_unit_fold3_v412_lr3e5/"
    "middle_late_r32_seed17_init20260715041/final"
).resolve()
SFTC_REPORT_V44A = (
    ROOT / "experiments/sft_controls/"
    "v42c_matched_init_equal_unit_fold3_v412_lr3e5/runtime_report_v42c.json"
).resolve()
SFTC_OUTPUT_V44A = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v42c_sft_qwen35_vllm_namespace_v44a_all_candidates"
).resolve()

SFTD_SOURCE_V44A = (
    ROOT / "experiments/sft_controls/"
    "v42d_matched_init_equal_unit_fold3_v412_lr1e5/"
    "middle_late_r32_seed17_init20260715041/final"
).resolve()
SFTD_REPORT_V44A = (
    ROOT / "experiments/sft_controls/"
    "v42d_matched_init_equal_unit_fold3_v412_lr1e5/runtime_report_v42d.json"
).resolve()
SFTD_OUTPUT_V44A = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v42d_sft_qwen35_vllm_namespace_v44a_all_candidates"
).resolve()

ES_SOURCE_V44A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43d_matched_lora_es_fold3_pop8_step1_finalize_retry/adapter_step1_v43d"
).resolve()
ES_FAILURE_V44A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43d_matched_lora_es_fold3_pop8_step1_finalize_retry/failure_v43d.json"
).resolve()
ES_ATTEMPT_V44A = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".v43d_matched_lora_es_fold3_pop8_step1_finalize_retry.attempt.json"
).resolve()
ES_PREREG_V44A = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_step1_v43d.json"
).resolve()
ES_OUTPUT_V44A = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v43d_lora_es_qwen35_vllm_namespace_v44a_all_candidates"
).resolve()

EXPECTED_V44A = {
    "sft_v42b": {
        "weights": "08595565b891254b18b57522819d6b029509f483661c8a06ec267890975fffd8",
        "config": "60bd216e895cba461a7cda482dc72635ac4037f301ee0f0c295e00d5a1247f5d",
        "seal": "303a9ecfdcb32ad49f16a990e47ee1ab85f5e0d026427c91ea3fadc16567e7e7",
        "seal_content": "beeb062797d7ea387857566bb955d638ab672920c64af3a44d7a49b607dbef13",
    },
    "sft_v42c": {
        "weights": "3ed4ca7e2acb7940dcb9ae3db202e2d406b8abbd57bbc4577097ef3ebd511021",
        "config": "8cf1f688544215124a8ec6570c9e734899de1be2e2ff6e6bc7973d484ad93a17",
        "seal": "a328ac874ba32e27326d8037b493b7282498dcaa3a9d55f5239273bcc5416bd7",
        "seal_content": "20119e9890806ca17c887d12b76eece198862d59bc14a5cdd843594f72861ef4",
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42c",
        "status": "complete_train_only_lr3e5_state_sealed_shadow_unopened",
    },
    "sft_v42d": {
        "weights": "3cae0bd835631b7374c807cbfc1aa3a1d187fb9b08e8a566189647cac9c592db",
        "config": "00229a1011c8f2bd28e10df8af3c0942cdf9b04dc0aedc4e5981899802bfaed6",
        "seal": "2768f91ae537cd47e399de2030c9361e28c971506224c4e09a41783b530ce0a1",
        "seal_content": "1017756945ef97d1b2fbc52bb8dc092350324eb61d398adcdbc724e0f5873f2a",
        "schema": "specialist-sft-matched-init-equal-unit-runtime-v42d",
        "status": "complete_train_only_lr1e5_state_sealed_shadow_unopened",
    },
    "lora_es_v43d": {
        "weights": "75de1765d9fbdff4964c446a49dc9d7fd72a0195a89f1b71058bc77cb4253158",
        "config": "ede582c12e82fb50eb97ac934ff08eb553a79d2c2d999235abcd8b29795b1d52",
        "failure": "cf359c0ee8d89e98e3f9b3624ec57b1830de7275c99c5463110b31d162c8f820",
        "failure_content": "90b83e972861bfda9282a204692cdfa2d7e508b51430325eff86dbe25aacf35c",
        "attempt": "af72c78b5cd15a974112ab0feeaf623bd5f31c98f3382e2950d64f8f140b38a1",
        "attempt_content": "f6cd70ca440e3a3aa2457d5af496a18feb5df1a7400d30f4e8c81b4b0a753806",
        "prereg": "928ae4930d0b52f7c51159b8d8644bafc9fa0bc0049aaf88ac2f68a1a9c36415",
        "prereg_content": "756f09afefb3441976016b516176c66c48fe44c15bd928143cf47622e36d7865",
        "master_sha256": "4f6f8ee4a892ab00afc16c13182e6d04877a907d79e1bbd85ec8a207aac3c0c7",
    },
}

CANDIDATE_SPECS_V44A = {
    "sft_v42b": (SFT_SOURCE_V44A, SFT_REPORT_V44A, SFT_OUTPUT_V44A),
    "sft_v42c": (SFTC_SOURCE_V44A, SFTC_REPORT_V44A, SFTC_OUTPUT_V44A),
    "sft_v42d": (SFTD_SOURCE_V44A, SFTD_REPORT_V44A, SFTD_OUTPUT_V44A),
    "lora_es_v43d": (ES_SOURCE_V44A, None, ES_OUTPUT_V44A),
}


def canonical_sha256_v44a(value: object) -> str:
    return hashing.canonical_sha256(value)


def file_sha256_v44a(path: Path) -> str:
    return hashing.file_sha256(Path(path))


def tensor_sha256_v44a(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def transform_key_v44a(key: str) -> str:
    if key.startswith(TARGET_PREFIX_V44A):
        raise ValueError("V44A refuses an already transformed key")
    if not key.startswith(SOURCE_PREFIX_V44A):
        raise ValueError("V44A source key is outside the canonical namespace")
    suffix = key[len(SOURCE_PREFIX_V44A):]
    if not suffix or suffix.startswith("language_model."):
        raise ValueError("V44A key cannot be transformed exactly once")
    return TARGET_PREFIX_V44A + suffix


def _read_self_hashed_v44a(path: Path, expected_file: str,
                            expected_content: str) -> dict:
    if file_sha256_v44a(path) != expected_file:
        raise RuntimeError(f"V44A seal file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    compact = {key: item for key, item in value.items()
               if key != "content_sha256_before_self_field"}
    if content != expected_content or canonical_sha256_v44a(compact) != content:
        raise RuntimeError(f"V44A seal self-hash changed: {path}")
    return value


def _source_seal_v44a(arm: str) -> dict:
    if arm.startswith("sft_"):
        report_path = CANDIDATE_SPECS_V44A[arm][1]
        expected = EXPECTED_V44A[arm]
        report = _read_self_hashed_v44a(
            report_path, expected["seal"], expected["seal_content"],
        )
        expected_schema = expected.get(
            "schema", "specialist-sft-matched-init-equal-unit-runtime-v42b"
        )
        expected_status = expected.get(
            "status", "complete_train_only_retry_state_sealed_shadow_unopened"
        )
        if (
            report.get("schema") != expected_schema
            or report.get("status") != expected_status
            or report.get("validation_ood_or_holdout_opened") is not False
            or report.get("artifacts", {}).get("output_file_sha256", {}).get(
                "final/adapter_model.safetensors"
            ) != expected["weights"]
            or report.get("artifacts", {}).get("output_file_sha256", {}).get(
                "final/adapter_config.json"
            ) != expected["config"]
        ):
            raise RuntimeError(f"V44A {arm} SFT source seal changed")
        return {
            "schema": "matched-sft-success-seal-v44a",
            "report": str(report_path),
            "report_file_sha256": expected["seal"],
            "report_content_sha256": expected["seal_content"],
            "state_complete": True,
            "selection_data_opened": False,
        }
    if arm != "lora_es_v43d":
        raise ValueError(f"unknown V44A adapter arm: {arm}")
    expected = EXPECTED_V44A[arm]
    failure = _read_self_hashed_v44a(
        ES_FAILURE_V44A, expected["failure"], expected["failure_content"]
    )
    attempt = _read_self_hashed_v44a(
        ES_ATTEMPT_V44A, expected["attempt"], expected["attempt_content"]
    )
    prereg = _read_self_hashed_v44a(
        ES_PREREG_V44A, expected["prereg"], expected["prereg_content"]
    )
    if (
        failure.get("schema") != "matched-lora-es-failure-v43d"
        or failure.get("message") != "v43a base score differs across actors"
        or "post_update = _base_score" not in failure.get("traceback", "")
        or failure.get("shadow_dev_external_eval_ood_or_holdout_opened") is not False
        or attempt.get("status") != "launching"
        or prereg.get("status") != "preregistered_before_train_only_launch"
        or prereg.get("sealed_holdout_opened") is not False
        or prereg.get("shadow_dev_external_eval_ood_or_holdout_authorized") is not False
    ):
        raise RuntimeError("V44A V43D post-update snapshot provenance changed")
    return {
        "schema": "v43d-post-update-snapshot-provenance-v44a",
        "preregistration_file_sha256": expected["prereg"],
        "preregistration_content_sha256": expected["prereg_content"],
        "attempt_file_sha256": expected["attempt"],
        "attempt_content_sha256": expected["attempt_content"],
        "failure_file_sha256": expected["failure"],
        "failure_content_sha256": expected["failure_content"],
        "snapshot_preceded_post_update_score_failure": True,
        "snapshot_master_sha256": expected["master_sha256"],
        "selection_data_opened": False,
        "quality_conclusion_from_failed_train_run": False,
    }


def source_spec_v44a(arm: str) -> dict:
    if arm not in CANDIDATE_SPECS_V44A:
        raise ValueError(f"unknown V44A adapter arm: {arm}")
    directory, _, output = CANDIDATE_SPECS_V44A[arm]
    return {
        "arm": arm,
        "directory": directory,
        "weights": directory / "adapter_model.safetensors",
        "config": directory / "adapter_config.json",
        "output": output,
        "expected": EXPECTED_V44A[arm],
    }


def audit_source_v44a(arm: str) -> dict:
    spec = source_spec_v44a(arm)
    if (
        file_sha256_v44a(spec["weights"]) != spec["expected"]["weights"]
        or file_sha256_v44a(spec["config"]) != spec["expected"]["config"]
    ):
        raise RuntimeError(f"V44A source artifact changed: {arm}")
    seal = _source_seal_v44a(arm)
    config_bytes = spec["config"].read_bytes()
    config = json.loads(config_bytes)
    if (
        config.get("r") != 32
        or config.get("lora_alpha") != 64
        or config.get("layers_to_transform") != [20, 21, 22, 23]
        or config.get("bias") != "none"
        or config.get("lora_dropout") != 0.0
    ):
        raise RuntimeError(f"V44A source config changed: {arm}")
    with safe_open(spec["weights"], framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        source_metadata = handle.metadata() or {}
    tensors = load_file(spec["weights"], device="cpu")
    records = []
    for key in keys:
        tensor = tensors[key]
        if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise RuntimeError(f"V44A source tensor metadata changed: {key}")
        target = transform_key_v44a(key)
        records.append({
            "source_key": key,
            "target_key": target,
            "shape": list(tensor.shape),
            "dtype": "torch.float32",
            "elements": tensor.numel(),
            "tensor_sha256": tensor_sha256_v44a(tensor),
        })
    if (
        keys != sorted(keys)
        or len(records) != EXPECTED_TENSOR_COUNT_V44A
        or len({item["target_key"] for item in records}) != len(records)
        or sum(item["elements"] for item in records) != EXPECTED_ELEMENTS_V44A
    ):
        raise RuntimeError(f"V44A source inventory changed: {arm}")
    if arm == "lora_es_v43d" and (
        source_metadata.get("schema") != "canonical-peft-fp32-v41a"
        or source_metadata.get("master_sha256")
        != EXPECTED_V44A[arm]["master_sha256"]
    ):
        raise RuntimeError("V44A V43D canonical snapshot metadata changed")
    return {
        **spec,
        "seal": seal,
        "config_bytes": config_bytes,
        "source_metadata": source_metadata,
        "tensors": tensors,
        "records": records,
    }


def _atomic_bytes_v44a(path: Path, raw: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if path.exists() or temporary.exists():
        raise FileExistsError(path)
    temporary.write_bytes(raw)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json_v44a(path: Path, value: dict) -> None:
    _atomic_bytes_v44a(
        path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    )


def stage_one_v44a(arm: str, output: Path | None = None) -> dict:
    source = audit_source_v44a(arm)
    directory = Path(output or source["output"]).resolve()
    if directory.exists():
        raise FileExistsError(directory)
    target_tensors = {
        item["target_key"]: source["tensors"][item["source_key"]]
        for item in source["records"]
    }
    directory.mkdir(parents=True)
    weights = directory / "adapter_model.safetensors"
    config = directory / "adapter_config.json"
    manifest_path = directory / "stage_manifest_v44a.json"
    temporary = directory / f".adapter_model.safetensors.tmp-{os.getpid()}"
    linked = False
    try:
        save_file(target_tensors, temporary, metadata={"format": "pt"})
        os.link(temporary, weights)
        linked = True
        _atomic_bytes_v44a(config, source["config_bytes"])
        reopened = load_file(weights, device="cpu")
        mappings = []
        for item in source["records"]:
            target = reopened[item["target_key"]]
            if (
                target.dtype != torch.float32
                or not torch.equal(target, source["tensors"][item["source_key"]])
                or tensor_sha256_v44a(target) != item["tensor_sha256"]
            ):
                raise RuntimeError(f"V44A staged tensor changed: {item['target_key']}")
            mappings.append({
                **item,
                "target_tensor_sha256": tensor_sha256_v44a(target),
                "tensor_bytes_preserved_exactly": True,
            })
        if config.read_bytes() != source["config_bytes"]:
            raise RuntimeError("V44A staged config bytes changed")
        identity = {
            "schema": "vllm-language-model-lora-stage-identity-v44a",
            "sha256": canonical_sha256_v44a(mappings),
            "tensor_count": len(mappings),
            "elements": sum(item["elements"] for item in mappings),
            "source_key_inventory_sha256": canonical_sha256_v44a(
                [item["source_key"] for item in mappings]
            ),
            "target_key_inventory_sha256": canonical_sha256_v44a(
                [item["target_key"] for item in mappings]
            ),
            "ordered_value_sequence_sha256": canonical_sha256_v44a(
                [item["tensor_sha256"] for item in mappings]
            ),
            "all_tensor_bytes_preserved_exactly": True,
        }
        manifest = {
            "schema": "candidate-lora-vllm-stage-manifest-v44a",
            "status": "complete_cpu_only_key_transform",
            "arm": arm,
            "source": {
                "directory": str(source["directory"]),
                "weights_file_sha256": source["expected"]["weights"],
                "adapter_config_file_sha256": source["expected"]["config"],
                "safetensors_metadata": source["source_metadata"],
                "seal": source["seal"],
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "file_sha256": file_sha256_v44a(Path(__file__).resolve()),
                "torch_version": torch.__version__,
            },
            "transform": {
                "operation": "key_prefix_replacement_only",
                "source_prefix": SOURCE_PREFIX_V44A,
                "target_prefix": TARGET_PREFIX_V44A,
                "tensor_arithmetic_performed": False,
                "tensor_cast_performed": False,
                "tensor_bytes_preserved_exactly": True,
                "adapter_config_copied_byte_exact": True,
            },
            "artifact": {
                "directory": str(directory),
                "weights_file_sha256": file_sha256_v44a(weights),
                "adapter_config_file_sha256": file_sha256_v44a(config),
                "target_namespace": TARGET_PREFIX_V44A + "*",
            },
            "transformed_identity": identity,
            "tensor_mapping_records": mappings,
            "dataset_or_evaluation_accessed": False,
            "shadow_ood_holdout_or_heldout_accessed": False,
            "gpu_accessed": False,
        }
        manifest["content_sha256_before_self_field"] = canonical_sha256_v44a(manifest)
        _atomic_json_v44a(manifest_path, manifest)
        reopened_manifest = json.loads(manifest_path.read_text())
        content = reopened_manifest.pop("content_sha256_before_self_field")
        if content != canonical_sha256_v44a(reopened_manifest):
            raise RuntimeError("V44A stage manifest self-hash changed")
        return manifest
    except BaseException:
        manifest_path.unlink(missing_ok=True)
        config.unlink(missing_ok=True)
        if linked:
            weights.unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            pass
        raise
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    if any(spec[2].exists() for spec in CANDIDATE_SPECS_V44A.values()):
        raise FileExistsError("V44A staged adapter output already exists")
    created = []
    try:
        manifests = {}
        for arm in CANDIDATE_SPECS_V44A:
            manifest = stage_one_v44a(arm)
            created.append(source_spec_v44a(arm)["output"])
            manifests[arm] = {
                "directory": manifest["artifact"]["directory"],
                "weights_sha256": manifest["artifact"]["weights_file_sha256"],
                "manifest_content_sha256": manifest[
                    "content_sha256_before_self_field"
                ],
                "transformed_identity_sha256": manifest[
                    "transformed_identity"
                ]["sha256"],
            }
        print(json.dumps(manifests, sort_keys=True))
        return 0
    except BaseException:
        for directory in reversed(created):
            for name in (
                "stage_manifest_v44a.json", "adapter_config.json",
                "adapter_model.safetensors",
            ):
                (directory / name).unlink(missing_ok=True)
            try:
                directory.rmdir()
            except OSError:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
