#!/usr/bin/env python3
"""Seal the deferred, replicated OOD-first V49D comparison protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import run_sft_train_only_control_v36a as engine


ROOT = Path(__file__).resolve().parent
RUN_DIR = (
    ROOT / "experiments/sft_controls/v49d_v434_sampling_midpoint_lr5p5e5"
).resolve()
TRAINING_PREREGISTRATION = (RUN_DIR / "preregistration_v49d.json").resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "sft_v434_equal_vs_source50_replicated_ood_first_template_v49d.json"
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
SHADOW_MEAN_REWARD_DELTA_THRESHOLD = 0.0008257591
ARMS = (
    "base_a", "base_b", "base_c", "base_d",
    "v434_equal_a", "v434_equal_b",
    "v434_source50_a", "v434_source50_b",
)


def _validate_training_preregistration(path: Path) -> tuple[dict, str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    content = value.get("content_sha256_before_self_field")
    if (
        value.get("schema")
        != "specialist-v434-sampling-midpoint-preregistration-v49d"
        or value.get("status") != "sealed_unlaunched_train_only"
        or value.get("training_launch_authorized") is not True
        or value.get("evaluation_launch_authorized") is not False
        or tuple(value.get("training_arms", {}))
        != ("v434_equal", "v434_source50")
        or content != engine.canonical_sha256({
            key: item for key, item in value.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("V49D training preregistration changed")
    return value, engine.file_sha256(path), content


def build(training_preregistration: Path = TRAINING_PREREGISTRATION) -> dict:
    training, training_file_sha, training_content_sha = (
        _validate_training_preregistration(training_preregistration.resolve())
    )
    if engine.file_sha256(PARENT_PROTOCOL) != PARENT_PROTOCOL_FILE_SHA256:
        raise RuntimeError("V49D inherited V47D protocol file changed")
    # Do not parse the protected protocol or any protected data here. Its
    # previously sealed file/content identities are the inheritance boundary.
    value = {
        "schema": "specialist-v434-equal-source50-future-eval-v49d",
        "status": "sealed_deferred_pending_both_training_completion_receipts",
        "artifact_role": "future_protocol_only_not_a_launch_manifest",
        "gpu_launch_authorized": False,
        "evaluation_launch_authorized": False,
        "heldout_or_holdout_access_authorized": False,
        "contains_protected_semantic_content": False,
        "training_preregistration": {
            "path": str(training_preregistration.resolve()),
            "file_sha256": training_file_sha,
            "content_sha256": training_content_sha,
            "both_training_arms_preregistered": True,
            "training_recipe_identity": training["matched_control_contract"][
                "recipe_identity_sha256"
            ],
        },
        "activation_prerequisites": {
            "both_logical_candidates_must_have": [
                "complete self-hashed runtime report",
                "complete self-hashed attempt receipt",
                "exactly 48 optimizer steps and 3 complete row-equivalent passes",
                "positive attributed activity and residency on GPUs 0,1,2,3",
                "source adapter_model.safetensors SHA-256",
                "source adapter_config.json SHA-256",
                "unchanged initialization, loader, schedule, and weighting audit",
                "training shadow/OOD/holdout access remains false",
            ],
            "candidate_stage_manifests_must_then_be_content_addressed": True,
            "replicas_of_each_candidate_must_share_identical_staged_bytes": True,
            "new_runnable_preregistration_required_after_receipts_exist": True,
            "this_template_cannot_be_mutated_into_a_launch": True,
        },
        "logical_candidates": ["v434_equal", "v434_source50"],
        "arms": list(ARMS),
        "base_duplicate_arms": ["base_a", "base_b", "base_c", "base_d"],
        "logical_candidate_replicas": {
            "v434_equal": ["v434_equal_a", "v434_equal_b"],
            "v434_source50": ["v434_source50_a", "v434_source50_b"],
        },
        "runtime_shape": {
            "engine_count": 4,
            "physical_gpu_ids": [0, 1, 2, 3],
            "tensor_parallel_size_per_engine": 1,
            "two_full_fixed_waves": [
                [
                    {"arm": "base_a", "engine_index": 0},
                    {"arm": "base_b", "engine_index": 1},
                    {"arm": "base_c", "engine_index": 2},
                    {"arm": "base_d", "engine_index": 3},
                ],
                [
                    {"arm": "v434_equal_a", "engine_index": 0},
                    {"arm": "v434_equal_b", "engine_index": 1},
                    {"arm": "v434_source50_a", "engine_index": 2},
                    {"arm": "v434_source50_b", "engine_index": 3},
                ],
            ],
            "every_gpu_receives_one_request_per_wave": True,
            "all_four_gpus_busy_in_every_wave_required": True,
            "no_partial_or_third_wave_authorized": True,
            "identical_prompts_sampling_parameters_and_seed_for_all_arms": True,
            "four_base_outputs_exact_on_every_surface_required": True,
            "candidate_replica_outputs_bit_exact_required": False,
        },
        "inherited_protocol": {
            "path": str(PARENT_PROTOCOL),
            "file_sha256": PARENT_PROTOCOL_FILE_SHA256,
            "content_sha256": PARENT_PROTOCOL_CONTENT_SHA256,
            "inheritance_method": "identity_only_without_reopening_semantics",
            "bundles_prompts_sampling_seed_metrics_and_protocol_gates_unchanged": True,
            "protected_semantics_opened_by_builder": False,
            "protected_files_read_by_builder": False,
        },
        "ood_first_eligibility_gates": {
            "applied_independently_to_each_of_four_candidate_replicas": True,
            "both_replicas_of_each_logical_candidate_must_pass": True,
            "base_relative_ood_qa_mean_reward_delta_minimum": 0.0,
            "base_relative_ood_qa_exact_count_delta_minimum": 0,
            "base_relative_ood_prose_point_non_degradation_required": True,
            "base_relative_ood_prose_paired_document_bootstrap_lcb_minimum": 0.0,
            "all_inherited_protocol_and_leak_counters_must_not_increase": True,
            "all_inherited_generation_and_parser_protocol_gates_must_pass": True,
            "eligibility_set_frozen_before_any_shadow_ranking": True,
            "shadow_not_opened_for_an_ineligible_logical_candidate": True,
        },
        "direct_hypothesis_gates": {
            "comparison": "mean(v434_source50 replicas)-mean(v434_equal replicas)",
            "mean_replicated_ood_qa_reward_delta_minimum": 0.0,
            "mean_replicated_ood_qa_exact_count_delta_minimum": 0,
            "mean_replicated_shadow_reward_delta_minimum": (
                SHADOW_MEAN_REWARD_DELTA_THRESHOLD
            ),
            "shadow_threshold_origin": (
                "one_half_of_observed_V49B_minus_V47C_shadow_mean_reward_delta"
            ),
            "paired_ood_qa_bootstrap_ci_role": "informational_not_a_gate",
            "direct_gate_evaluated_only_after_both_logical_candidates_are_ood_eligible": True,
            "all_three_directional_thresholds_must_pass": True,
        },
        "selection_and_scope": {
            "primary_scientific_question": (
                "Does lambda=0.5 source balancing beat a fresh content-matched "
                "equal control without OOD degradation?"
            ),
            "lambda_values_authorized": [0.0, 0.5],
            "additional_hpo_or_lambda_sweep_authorized": False,
            "holdout_evaluation_authorized": False,
            "promotion_authorized": False,
            "comparison_is_future_cycle_diagnostic_only": True,
        },
        "access_firewall": {
            "training_rows_opened_by_builder": False,
            "shadow_semantics_opened": False,
            "ood_qa_semantics_opened": False,
            "ood_prose_semantics_opened": False,
            "heldout_or_holdout_opened": False,
            "gpu_accessed": False,
            "training_launched": False,
            "evaluation_launched": False,
        },
    }
    value["content_sha256_before_self_field"] = engine.canonical_sha256(value)
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-preregistration", default=str(
        TRAINING_PREREGISTRATION
    ))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    value = build(Path(args.training_preregistration))
    engine.atomic_write_json(output, value)
    print(json.dumps({
        "path": str(output),
        "file_sha256": engine.file_sha256(output),
        "content_sha256": value["content_sha256_before_self_field"],
        "evaluation_launch_authorized": False,
        "gpu_accessed": False,
        "protected_semantic_access_count": 0,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
