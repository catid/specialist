#!/usr/bin/env python3
"""V47B future-cycle replicated V42I versus V47A OOD-first evaluation."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_lora_es_vs_sft_ood_first_v46c as trusted
import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as ood_first
import stage_candidate_adapters_vllm_v44a as canonical_stage
import stage_v42i_adapter_vllm_v45d as v42i_stage
import stage_v47a_adapter_vllm_v47b as v47a_stage


ROOT = Path(__file__).resolve().parent
BASE_ARMS_V47B = ("base_a", "base_b", "base_c", "base_d")
LOGICAL_REPLICAS_V47B = {
    "sft_v42i": ("sft_v42i_a", "sft_v42i_b"),
    "sft_v47a": ("sft_v47a_a", "sft_v47a_b"),
}
LOGICAL_CANDIDATES_V47B = tuple(LOGICAL_REPLICAS_V47B)
CANDIDATE_ARMS_V47B = tuple(
    arm for replicas in LOGICAL_REPLICAS_V47B.values() for arm in replicas
)
ARMS_V47B = BASE_ARMS_V47B + CANDIDATE_ARMS_V47B
STAGED_BY_LOGICAL_V47B = {
    "sft_v42i": v42i_stage.OUTPUT,
    "sft_v47a": v47a_stage.OUTPUT,
}
STAGED_BY_ARM_V47B = {
    arm: STAGED_BY_LOGICAL_V47B[logical]
    for logical, replicas in LOGICAL_REPLICAS_V47B.items()
    for arm in replicas
}
ADAPTER_IDS_V47B = {
    arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS_V47B)
}
STAGE_EXPECTED_V47B = {
    "sft_v42i": dict(trusted.STAGE_EXPECTED_V46C["sft_v42i"]),
    "sft_v47a": {
        "arm": "sft_v47a",
        "weights": "af6504f07e549887b97dcf27207a16761b735e724df8ef3d0bc7d3005723a9ad",
        "config": "9ec17a6b05b1157d0f75fd6dc5ef4a2e5fc2b716e8bc14b99a33f5cc4f68b326",
        "manifest_file": "c8a57c8f2c757ff7aecef88c112f9c366663ef42af105ef90ebeca31fc8eb52d",
        "manifest_content": "eb606667f28bd7b929593723cbc21338fff87f1251cf67e38f181788a07679ce",
        "transformed_identity": "0e299e53c0f3bc674ee1c27b07f78deeec4e8a82e271624fb9b5c8258749caf7",
    },
}
EXPERIMENT = "v47b_sft_v42i_vs_v47a_replicated_ood_first_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v47b.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v47b.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "sft_v42i_vs_v47a_replicated_ood_first_eval_v47b.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v42i_vs_v47a_replicated_ood_first_eval_v47b.json"
).resolve()
RANK_FIELDS_V47B = trusted.RANK_FIELDS_V46C
_ORIGINAL_CANONICAL_STAGE = trusted.canonical_stage_binding_v46c


def arm_wave_plan_v47b():
    return (
        tuple((arm, index) for index, arm in enumerate(BASE_ARMS_V47B)),
        tuple((arm, index) for index, arm in enumerate(CANDIDATE_ARMS_V47B)),
    )


@contextmanager
def patched_trusted_v47b(*, runtime_functions: bool = False):
    names = {
        "BASE_ARMS_V46C": BASE_ARMS_V47B,
        "LOGICAL_REPLICAS_V46C": LOGICAL_REPLICAS_V47B,
        "LOGICAL_CANDIDATES_V46C": LOGICAL_CANDIDATES_V47B,
        "CANDIDATE_ARMS_V46C": CANDIDATE_ARMS_V47B,
        "ARMS_V46C": ARMS_V47B,
        "STAGED_BY_LOGICAL_V46C": STAGED_BY_LOGICAL_V47B,
        "STAGED_BY_ARM_V46C": STAGED_BY_ARM_V47B,
        "ADAPTER_IDS_V46C": ADAPTER_IDS_V47B,
        "STAGE_EXPECTED_V46C": STAGE_EXPECTED_V47B,
        "RANK_FIELDS_V46C": RANK_FIELDS_V47B,
        "EXPERIMENT": EXPERIMENT,
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "RAW": RAW,
        "GPU_LOG": GPU_LOG,
        "REPORT": REPORT,
        "DEFAULT_PREREGISTRATION": DEFAULT_PREREGISTRATION,
        "arm_wave_plan_v46c": arm_wave_plan_v47b,
    }
    if runtime_functions:
        names.update({
            "load_preregistration_v46c": load_preregistration_v47b,
            "implementation_bindings_v46c": implementation_bindings_v47b,
        })
    saved = {name: getattr(trusted, name) for name in names}
    for name, value in names.items():
        setattr(trusted, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(trusted, name, value)


def canonical_stage_binding_v47b(logical: str) -> dict:
    with patched_trusted_v47b():
        return _ORIGINAL_CANONICAL_STAGE(logical)


def replica_stage_bindings_v47b() -> dict:
    logical = {
        name: canonical_stage_binding_v47b(name)
        for name in LOGICAL_CANDIDATES_V47B
    }
    return {
        arm: {
            **logical[name], "replica_arm": arm,
            "adapter_id": ADAPTER_IDS_V47B[arm],
        }
        for name, replicas in LOGICAL_REPLICAS_V47B.items()
        for arm in replicas
    }


def implementation_bindings_v47b() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_sft_v42i_vs_v47a_ood_first_preregistration_v47b.py",
        "tests": ROOT / "test_sft_v42i_vs_v47a_ood_first_v47b.py",
        "trusted_v46c_runtime": Path(trusted.__file__).resolve(),
        "core_runtime": Path(core.__file__).resolve(),
        "ood_first_runtime": Path(ood_first.__file__).resolve(),
        "source_faithful_parser_runtime": Path(
            ood_first.parser_fix.__file__
        ).resolve(),
        "environment_runtime": Path(ood_first.environment.__file__).resolve(),
        "canonical_staging_runtime": Path(canonical_stage.__file__).resolve(),
        "sft_v42i_staging_runtime": Path(v42i_stage.__file__).resolve(),
        "sft_v47a_staging_runtime": Path(v47a_stage.__file__).resolve(),
        "sft_v42i_source_weights": v42i_stage.SOURCE / "adapter_model.safetensors",
        "sft_v42i_source_config": v42i_stage.SOURCE / "adapter_config.json",
        "sft_v42i_source_report": v42i_stage.REPORT,
        "sft_v47a_source_weights": v47a_stage.SOURCE / "adapter_model.safetensors",
        "sft_v47a_source_config": v47a_stage.SOURCE / "adapter_config.json",
        "sft_v47a_source_report": v47a_stage.REPORT,
        "sft_v47a_source_preregistration": v47a_stage.PREREGISTRATION,
        "sft_v47a_source_attempt": v47a_stage.ATTEMPT,
        "sft_v47a_source_gpu_log": v47a_stage.GPU_LOG,
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
        "parent_v45e_preregistration": trusted.PARENT_PREREGISTRATION,
    }
    for logical, directory in STAGED_BY_LOGICAL_V47B.items():
        paths[f"{logical}_staged_weights"] = directory / "adapter_model.safetensors"
        paths[f"{logical}_staged_config"] = directory / "adapter_config.json"
        paths[f"{logical}_stage_manifest"] = directory / "stage_manifest_v44a.json"
    result = {label: core.file_sha256(path) for label, path in paths.items()}
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    result["environment"] = ood_first.environment.environment_bindings_v44b()
    result["sft_v42i_source_audited"] = (
        len(v42i_stage.audit_source_v45d()["records"]) == 70
    )
    result["sft_v47a_source_seal"] = v47a_stage.source_seal_v47b()
    return result


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def load_preregistration_v47b(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V47B preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or _compact_sha(value) != content
        or value.get("schema")
        != "sft-v42i-vs-v47a-replicated-ood-first-preregistration-v47b"
        or value.get("status")
        != "preregistered_before_fresh_replicated_ood_first_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("current_fixed_holdout_cycle_eligible") is not False
        or value.get("protected_semantics_inspected_during_v47b_revision")
        is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V47B)
        or value.get("logical_candidates") != list(LOGICAL_CANDIDATES_V47B)
        or value.get("replica_staged_adapters") != replica_stage_bindings_v47b()
        or value.get("staged_adapters") != replica_stage_bindings_v47b()
        or value.get("implementation_bindings") != implementation_bindings_v47b()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
    ):
        raise RuntimeError("V47B preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    return value


def main(argv: list[str] | None = None) -> int:
    with patched_trusted_v47b(runtime_functions=True):
        return trusted.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
