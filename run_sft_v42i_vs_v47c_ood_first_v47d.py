#!/usr/bin/env python3
"""Strict replicated V42I versus lineage-stable V47C OOD-first evaluation."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import run_sft_v42i_vs_v47a_ood_first_v47b as trusted
import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45a as ood_first
import stage_candidate_adapters_vllm_v44a as canonical_stage
import stage_v42i_adapter_vllm_v45d as v42i_stage
import stage_v47c_adapter_vllm_v47d as v47c_stage


ROOT = Path(__file__).resolve().parent
BASE_ARMS_V47D = ("base_a", "base_b", "base_c", "base_d")
LOGICAL_REPLICAS_V47D = {
    "sft_v42i": ("sft_v42i_a", "sft_v42i_b"),
    "sft_v47c": ("sft_v47c_a", "sft_v47c_b"),
}
LOGICAL_CANDIDATES_V47D = tuple(LOGICAL_REPLICAS_V47D)
CANDIDATE_ARMS_V47D = tuple(
    arm for replicas in LOGICAL_REPLICAS_V47D.values() for arm in replicas
)
ARMS_V47D = BASE_ARMS_V47D + CANDIDATE_ARMS_V47D
STAGED_BY_LOGICAL_V47D = {
    "sft_v42i": v42i_stage.OUTPUT,
    "sft_v47c": v47c_stage.OUTPUT,
}
STAGED_BY_ARM_V47D = {
    arm: STAGED_BY_LOGICAL_V47D[logical]
    for logical, replicas in LOGICAL_REPLICAS_V47D.items()
    for arm in replicas
}
ADAPTER_IDS_V47D = {
    arm: index + 1 for index, arm in enumerate(CANDIDATE_ARMS_V47D)
}
STAGE_EXPECTED_V47D = {
    "sft_v42i": dict(trusted.STAGE_EXPECTED_V47B["sft_v42i"]),
    "sft_v47c": {
        "arm": "sft_v47c",
        "weights": "46fd6d680a06e414d834c43dd4346e495f3d5488257ffad493c82d7cfde29930",
        "config": "6848af56fe7cbc9beb7806083c573747391b79095f66ee9bd322e76413bb4e3d",
        "manifest_file": "50ce1191d36c176ddec4cb557aa7f56f1acfb4beb98ba09c8260e750bb67e5a8",
        "manifest_content": "4809ce6700e775d0903deba1f3524f4ea142a472f233f07c457cb272ee9b463b",
        "transformed_identity": "c95a16fc670e15a04c68c77c324d45c14fe42c553c7b54a8f489c0eb04eb02e9",
    },
}
EXPERIMENT = "v47d_sft_v42i_vs_v47c_replicated_ood_first_eval"
RUN_DIR = (ROOT / "experiments/eggroll_es_hpo/runs" / EXPERIMENT).resolve()
ATTEMPT = (RUN_DIR.parent / f".{EXPERIMENT}.attempt.json").resolve()
RAW = (RUN_DIR / "raw_items_v47d.json").resolve()
GPU_LOG = (RUN_DIR / "gpu_activity_v47d.jsonl").resolve()
REPORT = (
    ROOT / "experiments/eval_reports/"
    "sft_v42i_vs_v47c_replicated_ood_first_eval_v47d.json"
).resolve()
DEFAULT_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v42i_vs_v47c_replicated_ood_first_eval_v47d.json"
).resolve()
RANK_FIELDS_V47D = trusted.RANK_FIELDS_V47B
EDIT_SCOPE_AUDIT = (
    ROOT / "experiments/sft_controls/"
    "v47c_lineage_stable_train_refresh_v430/edit_scope_audit_v47c.json"
).resolve()
EDIT_SCOPE_AUDIT_FILE_SHA256 = (
    "b6e9fb9fb7f95780c443405415bdff0dde6cf27ab0fea9e8daf049f0e3acb1ba"
)
EDIT_SCOPE_AUDIT_CONTENT_SHA256 = (
    "40e9723c6de9d7e8ba3b69649aae9d5727229a043be1d9f6bd6c5c6d1a6a6632"
)


def arm_wave_plan_v47d():
    return (
        tuple((arm, index) for index, arm in enumerate(BASE_ARMS_V47D)),
        tuple((arm, index) for index, arm in enumerate(CANDIDATE_ARMS_V47D)),
    )


@contextmanager
def patched_trusted_v47d(*, runtime_functions: bool = False):
    names = {
        "BASE_ARMS_V47B": BASE_ARMS_V47D,
        "LOGICAL_REPLICAS_V47B": LOGICAL_REPLICAS_V47D,
        "LOGICAL_CANDIDATES_V47B": LOGICAL_CANDIDATES_V47D,
        "CANDIDATE_ARMS_V47B": CANDIDATE_ARMS_V47D,
        "ARMS_V47B": ARMS_V47D,
        "STAGED_BY_LOGICAL_V47B": STAGED_BY_LOGICAL_V47D,
        "STAGED_BY_ARM_V47B": STAGED_BY_ARM_V47D,
        "ADAPTER_IDS_V47B": ADAPTER_IDS_V47D,
        "STAGE_EXPECTED_V47B": STAGE_EXPECTED_V47D,
        "RANK_FIELDS_V47B": RANK_FIELDS_V47D,
        "EXPERIMENT": EXPERIMENT,
        "RUN_DIR": RUN_DIR,
        "ATTEMPT": ATTEMPT,
        "RAW": RAW,
        "GPU_LOG": GPU_LOG,
        "REPORT": REPORT,
        "DEFAULT_PREREGISTRATION": DEFAULT_PREREGISTRATION,
        "arm_wave_plan_v47b": arm_wave_plan_v47d,
    }
    if runtime_functions:
        names.update({
            "load_preregistration_v47b": load_preregistration_v47d,
            "implementation_bindings_v47b": implementation_bindings_v47d,
        })
    saved = {name: getattr(trusted, name) for name in names}
    for name, value in names.items():
        setattr(trusted, name, value)
    try:
        # V47B is itself a wrapper over the V46C gate runtime.  Enter its
        # patch context as well so direct gate tests and runtime execution see
        # the V47D arm inventory at both wrapper layers.
        with trusted.patched_trusted_v47b(
            runtime_functions=runtime_functions
        ):
            yield
    finally:
        for name, value in saved.items():
            setattr(trusted, name, value)


def canonical_stage_binding_v47d(logical: str) -> dict:
    with patched_trusted_v47d():
        return trusted.canonical_stage_binding_v47b(logical)


def replica_stage_bindings_v47d() -> dict:
    logical = {
        name: canonical_stage_binding_v47d(name)
        for name in LOGICAL_CANDIDATES_V47D
    }
    return {
        arm: {
            **logical[name], "replica_arm": arm,
            "adapter_id": ADAPTER_IDS_V47D[arm],
        }
        for name, replicas in LOGICAL_REPLICAS_V47D.items()
        for arm in replicas
    }


def implementation_bindings_v47d() -> dict:
    paths = {
        "runtime": Path(__file__).resolve(),
        "builder": ROOT / "build_sft_v42i_vs_v47c_ood_first_preregistration_v47d.py",
        "tests": ROOT / "test_sft_v42i_vs_v47c_ood_first_v47d.py",
        "trusted_v47b_protocol_runtime": Path(trusted.__file__).resolve(),
        "trusted_v46c_gate_runtime": Path(trusted.trusted.__file__).resolve(),
        "core_runtime": Path(core.__file__).resolve(),
        "ood_first_runtime": Path(ood_first.__file__).resolve(),
        "source_faithful_parser_runtime": Path(ood_first.parser_fix.__file__).resolve(),
        "environment_runtime": Path(ood_first.environment.__file__).resolve(),
        "canonical_staging_runtime": Path(canonical_stage.__file__).resolve(),
        "sft_v42i_staging_runtime": Path(v42i_stage.__file__).resolve(),
        "sft_v47c_staging_runtime": Path(v47c_stage.__file__).resolve(),
        "sft_v42i_source_weights": v42i_stage.SOURCE / "adapter_model.safetensors",
        "sft_v42i_source_config": v42i_stage.SOURCE / "adapter_config.json",
        "sft_v42i_source_report": v42i_stage.REPORT,
        "sft_v47c_source_weights": v47c_stage.SOURCE / "adapter_model.safetensors",
        "sft_v47c_source_config": v47c_stage.SOURCE / "adapter_config.json",
        "sft_v47c_source_report": v47c_stage.REPORT,
        "sft_v47c_source_preregistration": v47c_stage.PREREGISTRATION,
        "sft_v47c_source_attempt": v47c_stage.ATTEMPT,
        "sft_v47c_source_gpu_log": v47c_stage.GPU_LOG,
        "sft_v47c_lineage_manifest": (
            ROOT / "experiments/sft_controls/"
            "v47c_lineage_stable_train_refresh_v430/manifest_v47c.json"
        ).resolve(),
        "sft_v47c_edit_scope_audit": EDIT_SCOPE_AUDIT,
        "sft_v47c_edit_scope_audit_builder": ROOT / "build_v47c_edit_scope_audit.py",
        "sft_v47c_edit_scope_audit_tests": ROOT / "test_build_v47c_edit_scope_audit.py",
        "model_config": core.MODEL / "config.json",
        "model_index": core.MODEL / "model.safetensors.index.json",
        "tuned_table": core.TUNED_FILE,
        "parent_v45e_preregistration": trusted.trusted.PARENT_PREREGISTRATION,
    }
    for logical, directory in STAGED_BY_LOGICAL_V47D.items():
        paths[f"{logical}_staged_weights"] = directory / "adapter_model.safetensors"
        paths[f"{logical}_staged_config"] = directory / "adapter_config.json"
        paths[f"{logical}_stage_manifest"] = directory / "stage_manifest_v44a.json"
    result = {label: core.file_sha256(path) for label, path in paths.items()}
    result["model_shards_content_sha256"] = core.MODEL_SHARDS_CONTENT_SHA256
    result["environment"] = ood_first.environment.environment_bindings_v44b()
    result["sft_v42i_source_audited"] = (
        len(v42i_stage.audit_source_v45d()["records"]) == 70
    )
    result["sft_v47c_source_seal"] = v47c_stage.source_seal_v47d()
    return result


def _compact_sha(value: dict) -> str:
    return core.canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    })


def load_preregistration_v47d(args) -> dict:
    path = Path(args.preregistration).resolve()
    if core.file_sha256(path) != args.preregistration_sha256:
        raise RuntimeError("V47D preregistration file identity changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        content != args.preregistration_content_sha256
        or _compact_sha(value) != content
        or value.get("schema")
        != "sft-v42i-vs-v47c-replicated-ood-first-preregistration-v47d"
        or value.get("status")
        != "preregistered_before_fresh_replicated_ood_first_evaluation"
        or value.get("heldout_or_holdout_access_authorized") is not False
        or value.get("current_fixed_holdout_cycle_eligible") is not False
        or value.get("protected_semantics_inspected_during_revision") is not False
        or value.get("single_access_inputs") != core.PROTECTED_INPUTS_V44A
        or value.get("arms") != list(ARMS_V47D)
        or value.get("logical_candidates") != list(LOGICAL_CANDIDATES_V47D)
        or value.get("replica_staged_adapters") != replica_stage_bindings_v47d()
        or value.get("staged_adapters") != replica_stage_bindings_v47d()
        or value.get("implementation_bindings") != implementation_bindings_v47d()
        or not isinstance(value.get("cpu_preflight_expected"), dict)
        or value.get("validation_label_policy", {}).get(
            "evaluation_shadow_file_sha256"
        ) != core.PROTECTED_INPUTS_V44A["shadow"]["file_sha256"]
        or value.get("validation_label_policy", {}).get(
            "refreshed_shadow_used_as_evaluation_input"
        ) is not False
        or value.get("edit_scope_audit_binding", {}).get("file_sha256")
        != EDIT_SCOPE_AUDIT_FILE_SHA256
        or value.get("edit_scope_audit_binding", {}).get("content_sha256")
        != EDIT_SCOPE_AUDIT_CONTENT_SHA256
        or value.get("edit_scope_audit_binding", {}).get(
            "edited_rows_entering_train"
        ) != 42
        or value.get("edit_scope_audit_binding", {}).get(
            "edited_rows_excluded_in_shadow"
        ) != 9
    ):
        raise RuntimeError("V47D preregistration content changed")
    core._forbid_holdout_v44a(
        item["path"] for item in value["single_access_inputs"].values()
    )
    return value


def main(argv: list[str] | None = None) -> int:
    with patched_trusted_v47d(runtime_functions=True):
        return trusted.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
