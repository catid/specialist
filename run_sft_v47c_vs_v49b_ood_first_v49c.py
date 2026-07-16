#!/usr/bin/env python3
"""Strict replicated V47C versus source-balanced V49B OOD-first evaluation."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_sft_v42i_vs_v47c_ood_first_v47d as trusted
import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as ood_first
import stage_candidate_adapters_vllm_v44a as canonical_stage
import stage_v47c_adapter_vllm_v47d as v47c_stage
import stage_v49b_adapter_vllm_v49c as v49b_stage


ROOT = Path(__file__).resolve().parent
BASE_ARMS_V49C = ("base_a", "base_b", "base_c", "base_d")
LOGICAL_REPLICAS_V49C = {
    "sft_v47c": ("sft_v47c_a", "sft_v47c_b"),
    "sft_v49b": ("sft_v49b_a", "sft_v49b_b"),
}
LOGICAL_CANDIDATES_V49C = tuple(LOGICAL_REPLICAS_V49C)
CANDIDATE_ARMS_V49C = tuple(
    arm for replicas in LOGICAL_REPLICAS_V49C.values() for arm in replicas
)
ARMS_V49C = BASE_ARMS_V49C + CANDIDATE_ARMS_V49C
STAGED_BY_LOGICAL_V49C = {
    "sft_v47c": v47c_stage.OUTPUT,
    "sft_v49b": v49b_stage.OUTPUT,
}
STAGED_BY_ARM_V49C = {
    arm: STAGED_BY_LOGICAL_V49C[logical]
    for logical, replicas in LOGICAL_REPLICAS_V49C.items()
    for arm in replicas
}
ADAPTER_IDS_V49C = {
    arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS_V49C)
}
STAGE_EXPECTED_V49C = {
    "sft_v47c": dict(trusted.STAGE_EXPECTED_V47D["sft_v47c"]),
    "sft_v49b": {
        "arm": "sft_v49b",
        "weights": "b1cb8f132e271f2df3723909297f1c4042247339d551f09ee42d91caf69e80c8",
        "config": "941656d9035478ccceefc5693f1f022e8dc4f4366d8d4e8401ca4d7e5f72493f",
        "manifest_file": "1ebcc9512e6f3bb868cb3362662b9fad238a2b808ac642e8e199bf34b1c0b285",
        "manifest_content": "b132b92789af7977180e80e8df66f6b530a90389e03e198550732530e6fdf1a6",
        "transformed_identity": "3fbdba5bf41e0a4593aea674b1f0739414a989be387d1dd991deb4216504d214",
    },
}
EXPERIMENT = "v49c_sft_v47c_vs_v49b_replicated_ood_first_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v49c.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v49c.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "sft_v47c_vs_v49b_replicated_ood_first_eval_v49c.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v47c_vs_v49b_replicated_ood_first_eval_v49c.json"
).resolve()
PARENT_PROTOCOL = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v42i_vs_v47c_replicated_ood_first_eval_v47d.json"
).resolve()
PARENT_PROTOCOL_FILE_SHA256 = (
    "1f1e90f976c4f6399a703b58a5d2dfaadae30107d2a2fa3c758724626888d2f8"
)
PARENT_PROTOCOL_CONTENT_SHA256 = (
    "97611a30fabccc6c1f68c3ba164a1ce292d35a39e03ff6c84baaae754e9d7a36"
)
RANK_FIELDS_V49C = trusted.RANK_FIELDS_V47D


def arm_wave_plan_v49c():
    """Preserve V47D's two complete four-GPU waves exactly."""
    return (
        tuple((arm, index) for index, arm in enumerate(BASE_ARMS_V49C)),
        tuple((arm, index) for index, arm in enumerate(CANDIDATE_ARMS_V49C)),
    )


@contextmanager
def patched_trusted_v49c(*, runtime_functions: bool = False):
    names = {
        "BASE_ARMS_V47D": BASE_ARMS_V49C,
        "LOGICAL_REPLICAS_V47D": LOGICAL_REPLICAS_V49C,
        "LOGICAL_CANDIDATES_V47D": LOGICAL_CANDIDATES_V49C,
        "CANDIDATE_ARMS_V47D": CANDIDATE_ARMS_V49C,
        "ARMS_V47D": ARMS_V49C,
        "STAGED_BY_LOGICAL_V47D": STAGED_BY_LOGICAL_V49C,
        "STAGED_BY_ARM_V47D": STAGED_BY_ARM_V49C,
        "ADAPTER_IDS_V47D": ADAPTER_IDS_V49C,
        "STAGE_EXPECTED_V47D": STAGE_EXPECTED_V49C,
        "RANK_FIELDS_V47D": RANK_FIELDS_V49C,
        "EXPERIMENT": EXPERIMENT,
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "RAW": RAW,
        "GPU_LOG": GPU_LOG,
        "REPORT": REPORT,
        "DEFAULT_PREREGISTRATION": DEFAULT_PREREGISTRATION,
        "arm_wave_plan_v47d": arm_wave_plan_v49c,
    }
    if runtime_functions:
        names.update({
            "load_preregistration_v47d": load_preregistration_v49c,
            "implementation_bindings_v47d": implementation_bindings_v49c,
        })
    saved = {name: getattr(trusted, name) for name in names}
    for name, value in names.items():
        setattr(trusted, name, value)
    try:
        with trusted.patched_trusted_v47d(runtime_functions=runtime_functions):
            yield
    finally:
        for name, value in saved.items():
            setattr(trusted, name, value)


def _stage_expected_sealed_v49c() -> bool:
    return all(
        isinstance(value, str) and len(value) == 64 and "PENDING" not in value
        for value in STAGE_EXPECTED_V49C["sft_v49b"].values()
        if value != "sft_v49b"
    )


def canonical_stage_binding_v49c(logical: str) -> dict:
    if logical == "sft_v49b" and not _stage_expected_sealed_v49c():
        raise RuntimeError("V49C V49B stage identity is not sealed")
    with patched_trusted_v49c():
        return trusted.canonical_stage_binding_v47d(logical)


def replica_stage_bindings_v49c() -> dict:
    logical = {
        name: canonical_stage_binding_v49c(name)
        for name in LOGICAL_CANDIDATES_V49C
    }
    return {
        arm: {
            **logical[name], "replica_arm": arm,
            "adapter_id": ADAPTER_IDS_V49C[arm],
        }
        for name, replicas in LOGICAL_REPLICAS_V49C.items()
        for arm in replicas
    }


def implementation_bindings_v49c() -> dict:
    if not _stage_expected_sealed_v49c():
        raise RuntimeError("V49C implementation seal awaits completed V49B stage")
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_sft_v47c_vs_v49b_ood_first_preregistration_v49c.py",
        "tests": ROOT / "test_sft_v47c_vs_v49b_ood_first_v49c.py",
        "stage_tests": ROOT / "test_stage_v49b_adapter_vllm_v49c.py",
        "trusted_v47d_protocol_runtime": Path(trusted.__file__).resolve(),
        "trusted_v47d_protocol_builder": (
            ROOT / "build_sft_v42i_vs_v47c_ood_first_preregistration_v47d.py"
        ),
        "trusted_v47d_protocol_tests": (
            ROOT / "test_sft_v42i_vs_v47c_ood_first_v47d.py"
        ),
        "trusted_v46c_gate_runtime": Path(trusted.trusted.__file__).resolve(),
        "core_runtime": Path(core.__file__).resolve(),
        "ood_first_runtime": Path(ood_first.__file__).resolve(),
        "source_faithful_parser_runtime": Path(ood_first.parser_fix.__file__).resolve(),
        "environment_runtime": Path(ood_first.environment.__file__).resolve(),
        "canonical_staging_runtime": Path(canonical_stage.__file__).resolve(),
        "sft_v47c_staging_runtime": Path(v47c_stage.__file__).resolve(),
        "sft_v49b_staging_runtime": Path(v49b_stage.__file__).resolve(),
        "sft_v47c_source_weights": v47c_stage.SOURCE / "adapter_model.safetensors",
        "sft_v47c_source_config": v47c_stage.SOURCE / "adapter_config.json",
        "sft_v47c_source_report": v47c_stage.REPORT,
        "sft_v49b_source_weights": v49b_stage.SOURCE / "adapter_model.safetensors",
        "sft_v49b_source_config": v49b_stage.SOURCE / "adapter_config.json",
        "sft_v49b_source_report": v49b_stage.REPORT,
        "sft_v49b_source_preregistration": v49b_stage.PREREGISTRATION,
        "sft_v49b_source_attempt": v49b_stage.ATTEMPT,
        "sft_v49b_source_gpu_log": v49b_stage.GPU_LOG,
        "v47d_parent_protocol": PARENT_PROTOCOL,
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
        "parent_v45e_preregistration": trusted.trusted.trusted.PARENT_PREREGISTRATION,
    }
    for logical, directory in STAGED_BY_LOGICAL_V49C.items():
        paths[f"{logical}_staged_weights"] = directory / "adapter_model.safetensors"
        paths[f"{logical}_staged_config"] = directory / "adapter_config.json"
        paths[f"{logical}_stage_manifest"] = directory / "stage_manifest_v44a.json"
    result = {label: core.file_sha256(path) for label, path in paths.items()}
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    result["environment"] = ood_first.environment.environment_bindings_v44b()
    result["sft_v47c_source_seal"] = v47c_stage.source_seal_v47d()
    result["sft_v49b_source_seal"] = v49b_stage.source_seal_v49c()
    return result


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def parent_protocol_v47d_v49c() -> dict:
    """Load only the sealed V47D protocol metadata, never protected inputs."""
    if core.file_sha256(PARENT_PROTOCOL) != PARENT_PROTOCOL_FILE_SHA256:
        raise RuntimeError("V49C V47D parent protocol file changed")
    value = json.loads(PARENT_PROTOCOL.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256_before_self_field")
        != PARENT_PROTOCOL_CONTENT_SHA256
        or _compact_sha(value) != PARENT_PROTOCOL_CONTENT_SHA256
        or value.get("schema")
        != "sft-v42i-vs-v47c-replicated-ood-first-preregistration-v47d"
        or value.get("status")
        != "preregistered_before_fresh_replicated_ood_first_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("protected_semantics_inspected_during_revision") is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("replica_consensus_protocol_v47d", {}).get(
            "replicas_per_logical_candidate"
        ) != 2
        or value.get("selection_protocol_v47d", {}).get(
            "ood_eligible_set_constructed_before_shadow_ranking"
        ) is not True
        or value.get("runtime", {}).get("engine_count") != 4
        or value.get("runtime", {}).get("all_four_gpus_busy_in_every_evaluation_wave")
        is not True
    ):
        raise RuntimeError("V49C V47D parent protocol content changed")
    return value


def load_preregistration_v49c(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V49C preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or _compact_sha(value) != content
        or value.get("schema")
        != "sft-v47c-vs-v49b-replicated-ood-first-preregistration-v49c"
        or value.get("status")
        != "preregistered_after_v49b_train_completion_before_ood_first_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("current_fixed_holdout_cycle_eligible") is not False
        or value.get("protected_semantics_inspected_during_revision") is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V49C)
        or value.get("logical_candidates") != list(LOGICAL_CANDIDATES_V49C)
        or value.get("replica_staged_adapters") != replica_stage_bindings_v49c()
        or value.get("staged_adapters") != replica_stage_bindings_v49c()
        or value.get("implementation_bindings") != implementation_bindings_v49c()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
        or value.get("extends_v47d_protocol", {}).get("file_sha256")
        != PARENT_PROTOCOL_FILE_SHA256
        or value.get("extends_v47d_protocol", {}).get("content_sha256")
        != PARENT_PROTOCOL_CONTENT_SHA256
        or value.get("matched_comparison", {}).get("only_training_change")
        != "per-row example weights"
        or value.get("matched_comparison", {}).get("v42i_role")
        != "historical anchor through frozen V47D only"
    ):
        raise RuntimeError("V49C preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    return value


def main(argv: list[str] | None = None) -> int:
    with patched_trusted_v49c(runtime_functions=True):
        return trusted.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
