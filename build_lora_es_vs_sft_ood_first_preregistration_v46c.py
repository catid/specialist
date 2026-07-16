#!/usr/bin/env python3
"""Seal launchable V46C without opening protected evaluation semantics."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import run_lora_es_vs_sft_ood_first_v46c as runtime
import run_matched_lora_candidate_eval_v44a as core


OUTPUT = runtime.DEFAULT_PREREGISTRATION


def candidate_definition_v46c(logical: str) -> dict:
    stage = runtime.canonical_stage_binding_v46c(logical)
    if logical == "sft_v42i":
        source = {
            "training_family": "matched SFT",
            "learning_rate": runtime.sft_stage.SPEC["learning_rate"],
            "completed_steps": runtime.sft_stage.SPEC["completed_steps"],
            "source_weights_sha256": runtime.sft_stage.SPEC["weights"],
            "source_config_sha256": runtime.sft_stage.SPEC["config"],
            "source_report_file_sha256": runtime.sft_stage.SPEC["seal"],
            "source_report_content_sha256": runtime.sft_stage.SPEC[
                "seal_content"
            ],
        }
    elif logical == "lora_es_v43j":
        source = {
            "training_family": "matched LoRA EGGROLL-ES",
            "accepted_target_norm_ratio": 0.25,
            "source_weights_sha256": runtime.es_stage.EXPECTED["weights"],
            "source_config_sha256": runtime.es_stage.EXPECTED["config"],
            "canonical_adapter_identity_sha256": runtime.es_stage.EXPECTED[
                "master_sha256"
            ],
            "source_report_file_sha256": runtime.es_stage.EXPECTED["report"],
            "source_report_content_sha256": runtime.es_stage.EXPECTED[
                "report_content"
            ],
            "candidate_gate_file_sha256": runtime.es_stage.EXPECTED[
                "candidate_gate"
            ],
            "candidate_gate_content_sha256": runtime.es_stage.EXPECTED[
                "candidate_gate_content"
            ],
            "all_six_train_only_candidate_gates_passed": True,
            "all_four_training_gpus_attributed_positive": True,
        }
    else:
        raise ValueError(logical)
    return {
        "logical_candidate": logical,
        "replica_arms": list(runtime.LOGICAL_REPLICAS_V46C[logical]),
        **source,
        "staged_directory": stage["directory"],
        "staged_weights_sha256": stage["weights_file_sha256"],
        "staged_config_sha256": stage["adapter_config_file_sha256"],
        "stage_manifest_file_sha256": stage["manifest_file_sha256"],
        "stage_manifest_content_sha256": stage["manifest_content_sha256"],
        "transformed_identity_sha256": stage[
            "transformed_identity_sha256"
        ],
        "replicas_use_identical_staged_bytes": True,
        "replica_outputs_required_bit_exact": False,
        "replica_adapter_ids": {
            arm: runtime.ADAPTER_IDS_V46C[arm]
            for arm in runtime.LOGICAL_REPLICAS_V46C[logical]
        },
    }


def build_v46c() -> dict:
    parent = runtime.parent_preregistration_v46c()
    value = {
        key: item for key, item in parent.items()
        if key != "content_sha256_before_self_field"
    }
    stages = runtime.replica_stage_bindings_v46c()
    value.update({
        "schema": (
            "lora-es-v43j-vs-sft-v42i-replicated-ood-first-"
            "preregistration-v46c"
        ),
        "status": "preregistered_before_fresh_replicated_ood_first_evaluation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_launch_authorized": True,
        "evaluation_launch_authorized": True,
        "heldout_or_holdout_access_authorized": False,
        "protected_semantics_inspected_during_v46c_revision": False,
        "purpose": (
            "Compare the strict V42I matched-SFT control with the accepted "
            "0.25-norm V43J LoRA-ES state under independent replicated OOD "
            "gates and mean-replicated shadow ranking."
        ),
        "arms": list(runtime.ARMS_V46C),
        "base_duplicate_arms": list(runtime.BASE_ARMS_V46C),
        "logical_candidates": list(runtime.LOGICAL_CANDIDATES_V46C),
        "logical_candidate_replicas": {
            logical: list(replicas)
            for logical, replicas in runtime.LOGICAL_REPLICAS_V46C.items()
        },
        "candidate_arms": list(runtime.CANDIDATE_ARMS_V46C),
        "candidate_definitions": [
            candidate_definition_v46c(logical)
            for logical in runtime.LOGICAL_CANDIDATES_V46C
        ],
        "replica_staged_adapters": stages,
        "staged_adapters": stages,
        "implementation_bindings": runtime.implementation_bindings_v46c(),
        "extends_preregistration": {
            "path": str(runtime.PARENT_PREREGISTRATION),
            "file_sha256": runtime.PARENT_PREREGISTRATION_FILE_SHA256,
            "content_sha256": runtime.PARENT_PREREGISTRATION_CONTENT_SHA256,
            "protected_commitments_reused_without_reopening": True,
        },
        "protected_semantics_opened_by_v46c_builder": False,
        "fresh_artifacts": {
            "run_directory": str(runtime.RUN_DIR),
            "attempt": str(runtime.ATTEMPT),
            "gpu_log": str(runtime.GPU_LOG),
            "raw_local": str(runtime.RAW),
            "report": str(runtime.REPORT),
        },
        "replica_consensus_protocol_v46c": {
            "replicas_per_logical_candidate": 2,
            "replicas_share_exact_staged_adapter_bytes": True,
            "replicas_have_distinct_request_names_and_adapter_ids": True,
            "candidate_metric_or_generation_bit_equality_required": False,
            "each_replica_independently_ood_qa_gated": True,
            "each_replica_independently_ood_prose_gated": True,
            "each_replica_independently_protocol_gated": True,
            "logical_candidate_eligible_only_if_both_replicas_pass": True,
            "shadow_rank_uses_mean_of_two_replica_metric_vectors": True,
            "four_base_replicas_remain_exact_on_every_surface": True,
        },
        "selection_protocol_v46c": {
            "order": [
                "require exact four-base agreement on every surface",
                "independently OOD-gate both replicas of each logical candidate",
                "exclude a logical candidate if either replica is ineligible",
                "mean the two shadow metric vectors for each eligible candidate",
                "rank only the immutable OOD-eligible set by the mean vector",
                "require the selected mean vector to improve over exact base",
            ],
            "ood_eligible_set_constructed_before_shadow_ranking": True,
            "candidate_replica_disagreement_is_sampling_evidence_not_failure": True,
            "rank_fields": list(runtime.RANK_FIELDS_V46C),
            "exact_tie_order": list(runtime.LOGICAL_CANDIDATES_V46C),
            "fallback_when_no_logical_candidate_is_eligible": "base_a",
        },
    })
    value["runtime"] = dict(value["runtime"])
    value["runtime"].update({
        "engine_count": 4,
        "physical_gpu_ids": [0, 1, 2, 3],
        "tensor_parallel_size_per_engine": 1,
        "two_full_fixed_waves": [
            [{"arm": arm, "engine_index": engine} for arm, engine in wave]
            for wave in runtime.arm_wave_plan_v46c()
        ],
        "every_gpu_receives_one_request_per_wave": True,
        "all_four_gpus_busy_in_every_evaluation_wave": True,
        "identical_prompts_sampling_params_and_seed_for_all_arms": True,
        "four_base_replicates_exact_required": True,
        "candidate_replicate_outputs_exact_required": False,
    })
    value["shadow_protocol"] = dict(value["shadow_protocol"])
    value["shadow_protocol"].update({
        "selection_candidates": list(runtime.LOGICAL_CANDIDATES_V46C),
        "four_base_outputs_exact_equivalence_required": True,
        "candidate_replica_outputs_exact_equivalence_required": False,
        "logical_candidate_mean_replica_ranking": True,
    })
    value["content_sha256_before_self_field"] = core.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    output = Path(parser.parse_args(argv).output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build_v46c()
    core.atomic_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": core.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "cpu_preflight_content_sha256": value[
            "cpu_preflight_expected"
        ]["content_sha256_before_self_field"],
        "logical_candidates": value["logical_candidates"],
        "gpu_launch_authorized": True,
        "protected_semantics_inspected_during_revision": False,
        "heldout_or_holdout_opened": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
