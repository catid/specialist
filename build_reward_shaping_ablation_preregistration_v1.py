#!/usr/bin/env python3
"""Build the sealed CPU contract for the reward-shaping comparison."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import reward_shaping_ablation_v1 as shaping


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "reward_shaping_ablation_v1.json"
).resolve()
EVALUATION_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json"
).resolve()
SAMPLING_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "recipe_sampling_ablation_v1.json"
).resolve()
MIRRORED_V66 = (ROOT / "eggroll_es_mirrored_v66.py").resolve()
CENTERED_RANK_V43G = (ROOT / "lora_es_robust_consensus_v43g.py").resolve()
LEGACY_ZSCORE = (
    ROOT / "es-at-scale/es_at_scale/utils/reward_shaping.py"
).resolve()

EXPECTED_EVALUATION_FILE_SHA256 = (
    "04af81499067e2feb0186c0a61e4c1af10f838a8eb7deec6dd41cd192748cacf"
)
EXPECTED_EVALUATION_CONTENT_SHA256 = (
    "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
)
EXPECTED_SAMPLING_FILE_SHA256 = (
    "8fe9005c532dbdb286edfd359cbbaf689c46f529b3debb3edc74f48e5787b301"
)
EXPECTED_SAMPLING_CONTENT_SHA256 = (
    "18cab815193d05de6e7416b17c1ffeae334a6a613f3899faa459cc719144e97f"
)
DIRECTION_SEEDS_V1 = (
    140002291,
    1028842752,
    480373990,
    1037026679,
    759861149,
    227761095,
    428721957,
    150663570,
)


def file_sha256_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sealed_json_v1(
    path: Path,
    expected_file_sha256: str,
    expected_content_sha256: str,
) -> dict:
    if file_sha256_v1(path) != expected_file_sha256:
        raise RuntimeError(f"reward-shaping sealed file changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = dict(value)
    observed_content_sha = compact.pop("content_sha256_before_self_field", None)
    if (
        observed_content_sha != expected_content_sha256
        or shaping.canonical_sha256_v1(compact) != observed_content_sha
    ):
        raise RuntimeError(f"reward-shaping sealed content changed: {path}")
    return value


def _contracts_v1() -> tuple[dict, dict]:
    evaluation = _sealed_json_v1(
        EVALUATION_CONTRACT,
        EXPECTED_EVALUATION_FILE_SHA256,
        EXPECTED_EVALUATION_CONTENT_SHA256,
    )
    sampling = _sealed_json_v1(
        SAMPLING_CONTRACT,
        EXPECTED_SAMPLING_FILE_SHA256,
        EXPECTED_SAMPLING_CONTENT_SHA256,
    )
    if (
        evaluation.get("schema")
        != "specialist-recipe-evaluation-compute-contract-v1"
        or evaluation.get("status")
        != "sealed_before_recipe_hpo_protected_holdout_unopened_by_hpo"
        or evaluation.get("disjointness", {}).get("passed") is not True
        or evaluation.get("roles", {}).get("protected_holdout", {}).get(
            "access_authorized_by_this_contract"
        ) is not False
    ):
        raise RuntimeError("reward-shaping evaluation boundary changed")
    compute = sampling.get("compute_match", {})
    if (
        sampling.get("schema")
        != "specialist-category-prioritized-sampling-ablation-v1"
        or sampling.get("status")
        != "sealed_cpu_only_before_sampling_ablation_launch"
        or compute.get("mode") != "estimator_control"
        or compute.get("contract_rollout_target") != 2048
        or compute.get("mirrored_population_per_generation") != 16
        or any(item.get("rows") != 64 for item in sampling.get("variants", []))
    ):
        raise RuntimeError("reward-shaping sampling/compute boundary changed")
    return evaluation, sampling


def _synthetic_records_v1(training_seed: int = 1701) -> list[dict]:
    records = []
    for prompt_index in range(3):
        for direction_index, direction_seed in enumerate(DIRECTION_SEEDS_V1):
            for sign in (1, -1):
                records.append({
                    "dataset_role": "train",
                    "training_seed": training_seed,
                    "generation_index": 0,
                    "population_id": f"synthetic-{training_seed}",
                    "evaluation_contract_sha256": (
                        EXPECTED_EVALUATION_CONTENT_SHA256
                    ),
                    "prompt_group_id": f"opaque-prompt-{prompt_index}",
                    "repeat_index": 0,
                    "evaluation_seed": 20260717300 + prompt_index,
                    "direction_index": direction_index,
                    "direction_seed": direction_seed,
                    "sign": sign,
                    "reward": (
                        (prompt_index + 1) * 0.7
                        + direction_index * 0.11
                        + sign * (direction_index - 2.5) * 0.017
                    ),
                })
    return records


def _cpu_diagnostic_v1() -> dict:
    clean = _synthetic_records_v1()
    contaminated = copy.deepcopy(clean)
    contaminated[0]["reward"] = 1e12
    comparison = shaping.compare_reward_shaping_v1(
        clean, DIRECTION_SEEDS_V1, EXPECTED_EVALUATION_CONTENT_SHA256
    )
    outlier = shaping.outlier_sensitivity_v1(
        clean,
        contaminated,
        DIRECTION_SEEDS_V1,
        EXPECTED_EVALUATION_CONTENT_SHA256,
    )
    return {
        "schema": "reward-shaping-synthetic-cpu-diagnostic-v1",
        "purpose": "mechanics and adversarial sensitivity only; not model evidence",
        "synthetic_prompt_groups": 3,
        "synthetic_signed_candidates": len(clean),
        "direction_coefficients": {
            method: comparison["arms"][method]["direction_coefficients"]
            for method in shaping.METHODS_V1
        },
        "pairwise_direction_cosines": comparison[
            "pairwise_direction_cosines"
        ],
        "single_extreme_outlier": outlier["method_diagnostics"],
        "raw_and_direct_pair_coefficients_exactly_equal": comparison[
            "raw_and_direct_pair_coefficients_exactly_equal"
        ],
        "every_noncontaminated_prompt_group_bitwise_unchanged": all(
            item["only_contaminated_prompt_group_changed"]
            for item in outlier["method_diagnostics"].values()
        ),
        "model_reward_improvement_claimed": False,
        "method_selected_from_synthetic_evidence": False,
    }


def _implementation_bindings_v1() -> dict:
    paths = {
        "builder": Path(__file__).resolve(),
        "reward_shaping": ROOT / "reward_shaping_ablation_v1.py",
        "mirrored_pair_difference_v66": MIRRORED_V66,
        "centered_midrank_reference_v43g": CENTERED_RANK_V43G,
        "legacy_zscore_reference": LEGACY_ZSCORE,
    }
    return {
        name: {
            "path": str(path.resolve()),
            "file_sha256": file_sha256_v1(path),
        }
        for name, path in paths.items()
    }


def build_preregistration_v1() -> dict:
    raise RuntimeError(
        "reward-shaping V1 preregistration is historical and bound to "
        "quarantined evaluation V1; create a V2 successor"
    )
    evaluation, sampling = _contracts_v1()
    seeds = evaluation["seeds"]
    compute = evaluation["compute_accounting"]["budget_modes"][
        "estimator_control"
    ]
    panels = {
        item["name"]: {
            "rows": item["rows"],
            "ordered_unit_identity_sha256": item[
                "ordered_unit_identity_sha256"
            ],
        }
        for item in sampling["variants"]
    }
    result = {
        "schema": "specialist-reward-shaping-ablation-preregistration-v1",
        "status": "sealed_cpu_only_before_reward_population_or_model_access",
        "purpose": (
            "Compare prompt-local z-scores and exact-tie centered ranks with "
            "raw mirrored rewards/direct pair reduction using identical reward "
            "tensors and rollout budgets, then select without protected-holdout "
            "adaptation."
        ),
        "authorization": {
            "this_builder_launches_gpu": False,
            "future_four_gpu_comparison_after_v66_calibration": True,
            "train_reward_tensor_access_by_future_runner": True,
            "dev_access_at_fixed_rung_only": True,
            "ood_access_for_noninferiority_gate_only": True,
            "protected_holdout_access": False,
            "candidate_promotion_by_this_artifact": False,
        },
        "parents": {
            "evaluation_contract": {
                "path": str(EVALUATION_CONTRACT),
                "file_sha256": EXPECTED_EVALUATION_FILE_SHA256,
                "content_sha256": EXPECTED_EVALUATION_CONTENT_SHA256,
                "disjointness_passed": True,
                "protected_access_authorized": False,
            },
            "sampling_contract": {
                "path": str(SAMPLING_CONTRACT),
                "file_sha256": EXPECTED_SAMPLING_FILE_SHA256,
                "content_sha256": EXPECTED_SAMPLING_CONTENT_SHA256,
                "panels": panels,
            },
        },
        "audited_current_recipe": {
            "legacy_zscore": {
                "scope": "caller-supplied population mean and standard deviation",
                "formula": "(reward-mean)/(std+1e-8)",
                "prompt_locality_enforced_in_helper": False,
                "nonfinite_rejected_in_helper": False,
            },
            "v43g_centered_rank": {
                "scope": "complete signed population",
                "range": [-0.5, 0.5],
                "exact_ties": "average midrank",
                "nonfinite": "rejected",
            },
            "v66_pair_difference": {
                "coefficient": "reward_plus-reward_minus",
                "estimator": (
                    "learning_rate/(2*N*sigma)*sum(pair_difference*epsilon)"
                ),
                "same_prompt_decode_judge_for_both_signs": True,
            },
            "raw_mirrored_equivalence": (
                "sum(sign*raw_reward*epsilon) and "
                "sum((reward_plus-reward_minus)*epsilon) are algebraically "
                "identical for a complete exact-antithetic population"
            ),
        },
        "reward_tensor_contract": {
            "dataset_role": "train",
            "one_transform_call_contains_exactly_one": [
                "training_seed",
                "generation_index",
                "population_id",
                "evaluation_contract_sha256",
            ],
            "prompt_local_group_fields": list(shaping.PROMPT_GROUP_FIELDS_V1),
            "complete_candidates_per_group": 16,
            "direction_seeds": list(DIRECTION_SEEDS_V1),
            "both_signs_share_prompt_repeat_and_evaluation_seed": True,
            "records_with_extra_semantic_or_holdout_fields": "reject",
            "duplicate_missing_nonfinite_or_wrong_identity": "reject whole population",
            "cross_role_seed_generation_or_population_mixture": "reject whole call",
            "input_order_affects_result": False,
        },
        "methods": [
            {
                "id": "raw_rewards",
                "candidate_utility": "finite raw reward",
                "direction_coefficient": "mean_prompt(Rplus-Rminus)",
                "unique_gradient_arm": True,
                "production_reduction": "antithetic_pair_difference",
            },
            {
                "id": "within_prompt_centered_zscore",
                "candidate_utility": (
                    "(reward-prompt_population_mean)/"
                    "(prompt_population_std+1e-8)"
                ),
                "variance_denominator": "population",
                "zero_spread": "all utilities exactly zero",
                "unique_gradient_arm": True,
            },
            {
                "id": "within_prompt_centered_rank",
                "candidate_utility": "exact-tie midrank/(2N-1)-0.5",
                "ties": "identical rewards receive identical average rank",
                "zero_spread": "all utilities exactly zero",
                "unique_gradient_arm": True,
            },
            {
                "id": "antithetic_pair_difference",
                "candidate_utility": "not materialized",
                "direction_coefficient": "mean_prompt(Rplus-Rminus)",
                "unique_gradient_arm": False,
                "reason": "exactly the raw mirrored gradient; retained as an algebra/control and systems implementation",
            },
        ],
        "compute_match": {
            "mode": "estimator_control",
            "signed_population": 16,
            "directions": 8,
            "prompt_groups_per_repeat": 64,
            "frozen_population_repeats_before_arm_specific_update": 2,
            "optimization_rollouts_per_method_per_seed": 2048,
            "rollout_identity_formula": "64 prompts * 16 signs/directions * 2 repeats",
            "same_reward_tensor_reused_for_all_four_statistical_views": True,
            "three_unique_parameter_updates_not_four": True,
            "duplicate_raw_vs_pair_gpu_training_run": "prohibited pseudo-replication",
            "screen_gpu_second_ceiling_per_arm": compute[
                "screen_gpu_second_ceiling_per_arm"
            ],
            "confirmation_gpu_second_ceiling_per_seed": compute[
                "confirmation_gpu_second_ceiling_per_seed"
            ],
            "failed_arm_budget_reallocated": False,
            "all_four_physical_gpus_must_have_useful_activity": [0, 1, 2, 3],
        },
        "step_scale_control": {
            "purpose": "isolate shaping direction from arbitrary coefficient units",
            "reference": "current within-prompt z-score update FP32 LoRA delta L2",
            "procedure": (
                "materialize each method's deterministic FP32 LoRA delta from "
                "the shared rewards/noise, then apply one positive scalar so "
                "its exact parameter-space L2 equals the z-score reference"
            ),
            "extra_model_forward_or_reward_rollout": False,
            "zero_or_nonfinite_reference": "fail seed before update",
            "trust_region_and_existing_safety_caps_still_apply": True,
            "unscaled_and_scaled_coefficient_hashes_both_required": True,
        },
        "registered_training_seeds": {
            "screen": seeds["screen_training_seeds"],
            "confirmation": seeds["confirmation_training_seeds"],
            "unregistered_retry": "prohibited",
        },
        "required_report": {
            "gradient_direction_stability": [
                "within-seed prompt split-half coefficient cosine on shared epsilon basis",
                "exact FP32 LoRA update cosine across methods within seed",
                "exact FP32 LoRA update cosine across all three registered seeds",
                "minimum, median, and every pairwise value; zero vectors explicit",
            ],
            "reward_improvement": [
                "dev equal-conflict-unit primary delta versus frozen baseline",
                "paired 95% interval and per-seed sign",
                "dev secondary tuple in global contract order",
            ],
            "outlier_sensitivity": [
                "leave-one-prompt-group-out maximum update-angle change",
                "single-reward deterministic extreme contamination update-angle change",
                "maximum coefficient influence and affected prompt-group count",
            ],
            "ood_deltas": [
                "prose mean-token-logprob paired 95% LCB",
                "QA mean-reward paired 95% LCB",
                "QA exact-count delta",
            ],
            "compute_and_systems": [
                "optimization and evaluation rollouts",
                "charged GPU seconds per physical GPU",
                "generated and teacher-forced tokens",
                "peak VRAM and useful-activity evidence on GPUs 0-3",
                "reward-reduction CPU time and bytes communicated",
            ],
        },
        "selection_rule": {
            "protected_holdout_visible": False,
            "all_three_confirmation_seeds_required": True,
            "all_global_ood_noninferiority_conditions_required": True,
            "pooled_dev_primary_paired_95_lcb_minimum": 0.0,
            "positive_dev_primary_seeds_minimum": 2,
            "primary": "largest pooled dev equal-unit primary delta",
            "tie_breakers": [
                "global dev secondary tuple",
                "larger minimum split-half gradient cosine",
                "smaller maximum outlier-induced update angle",
                "lower charged GPU seconds",
                "lexicographic method id",
            ],
            "raw_winner_is_named": "raw_rewards_with_direct_pair_reduction",
            "no_method_passes": "retain frozen baseline; do not access protected holdout",
            "selection_receipt_persisted_before_terminal_access": True,
            "protected_result_can_change_method_or_recipe": False,
        },
        "cpu_synthetic_diagnostic": _cpu_diagnostic_v1(),
        "implementation_bindings": _implementation_bindings_v1(),
        "access_receipt": {
            "train_semantics_opened": False,
            "dev_semantics_opened": False,
            "ood_semantics_opened": False,
            "protected_holdout_semantics_opened": False,
            "model_or_adapter_loaded": False,
            "gpu_launched": False,
            "live_run_directory_touched": False,
        },
    }
    result["content_sha256_before_self_field"] = shaping.canonical_sha256_v1(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args(argv)
    result = build_preregistration_v1()
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v1(output),
        "content_sha256": result["content_sha256_before_self_field"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
