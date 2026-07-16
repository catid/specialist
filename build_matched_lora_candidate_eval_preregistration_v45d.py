#!/usr/bin/env python3
"""Seal compact V45D from prior commitments without protected-data access."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_matched_lora_candidate_eval_v44a as core
import run_matched_lora_candidate_eval_v45d as runtime


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_sft_boundary_ood_eligible_eval_v45d.json"
).resolve()


def candidate_definition_v45d(arm: str) -> dict:
    if arm == "sft_v42g":
        spec = runtime.prior.staging.SOURCE_SPECS_V45A[arm]
    elif arm in runtime.h_stage.SOURCE_SPECS_V45D:
        spec = runtime.h_stage.SOURCE_SPECS_V45D[arm]
    elif arm == "sft_v42i":
        spec = runtime.i_stage.SPEC
    else:
        raise ValueError(arm)
    value = {
        "arm": arm,
        "training_family": "matched SFT",
        "learning_rate": spec["learning_rate"],
        "completed_steps": spec["completed_steps"],
        "source_artifact_prefix": spec["artifact_prefix"],
        "source_weights_sha256": spec["weights"],
        "source_config_sha256": spec["config"],
        "source_report_file_sha256": spec["seal"],
        "source_report_content_sha256": spec["seal_content"],
        "staged_directory": str(runtime.STAGED_BY_ARM_V45D[arm]),
        "adapter_id": runtime.ADAPTER_IDS_V45D[arm],
    }
    if "trainer_state_sha256" in spec:
        value["trainer_state"] = {
            "file_sha256": spec["trainer_state_sha256"],
            "global_step": spec["completed_steps"],
            "epoch": spec["expected_epoch"],
            "max_steps": 48,
            "num_train_epochs": 3,
        }
    return value


def build() -> dict:
    prior = runtime.prior_preregistration_v45d()
    value = {
        key: item for key, item in prior.items()
        if key != "content_sha256_before_self_field"
    }
    value.update({
        "schema": "matched-lora-sft-boundary-ood-eligible-preregistration-v45d",
        "status": "preregistered_before_fresh_sft_boundary_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(runtime.ARMS_V45D),
        "base_duplicate_arms": list(runtime.BASE_ARMS_V45D),
        "padding_base_arms": [],
        "candidate_arms": list(runtime.CANDIDATE_ARMS_V45D),
        "candidate_definitions": [
            candidate_definition_v45d(arm)
            for arm in runtime.CANDIDATE_ARMS_V45D
        ],
        "implementation_bindings": runtime.implementation_bindings_v45d(),
        "staged_adapters": runtime.staged_adapter_bindings_v45d(),
        "extends_preregistration": {
            "path": str(runtime.V45C_PREREG),
            "file_sha256": runtime.V45C_PREREG_FILE_SHA256,
            "content_sha256": runtime.V45C_PREREG_CONTENT_SHA256,
            "protected_commitments_reused_without_reopening": True,
        },
        "v45c_aggregate_observed_before_v45d_preregistration": (
            runtime.prior_result_v45d()
        ),
        "rationale": (
            "V42H final had the best observed shadow mean but missed OOD QA "
            "exact non-degradation by one item; compare its epoch-1 and epoch-2 "
            "checkpoints against the repeat-stable V42G final and midpoint-LR "
            "V42I final under unchanged OOD-first filtering"
        ),
        "protected_semantics_inspected_during_v45d_revision": False,
        "heldout_or_holdout_access_authorized": False,
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
    })
    value["runtime"] = dict(value["runtime"])
    value["runtime"].update({
        "two_full_fixed_waves": [
            [{"arm": arm, "engine_index": engine} for arm, engine in wave]
            for wave in runtime.arm_wave_plan_v45d()
        ],
        "every_gpu_receives_one_request_per_wave": True,
        "all_four_gpus_busy_in_every_evaluation_wave": True,
        "identical_prompts_sampling_params_and_seed_for_all_arms": True,
        "four_base_replicates_exact_equivalence_required": True,
        "padding_base_replicates": 0,
    })
    value["shadow_protocol"] = dict(value["shadow_protocol"])
    value["shadow_protocol"].update({
        "selection_candidates": list(runtime.CANDIDATE_ARMS_V45D),
        "four_base_duplicate_exact_equivalence_required": True,
        "padding_base_arms_excluded_from_eligibility_and_ranking": [],
    })
    value["selection_protocol_v45a"] = dict(value["selection_protocol_v45a"])
    value["selection_protocol_v45a"].update({
        "candidate_set": list(runtime.CANDIDATE_ARMS_V45D),
        "every_candidate_ood_gated_before_shadow_ranking": True,
        "padding_base_arms_affect_candidate_eligibility_or_ranking": False,
    })
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build()
    core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "cpu_preflight_content_sha256": value[
            "cpu_preflight_expected"
        ]["content_sha256_before_self_field"],
        "protected_semantics_inspected_during_revision": False,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
