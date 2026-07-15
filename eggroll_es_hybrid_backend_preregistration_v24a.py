#!/usr/bin/env python3
"""Preregister the train-only V24A BF16-versus-hybrid backend comparison."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CANONICAL_ROOT = Path("/home/catid/specialist")
PREREGISTRATION_PATH_V24A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V24A_HYBRID_BACKEND_COMPATIBILITY_PREREGISTRATION.json"
)
AUDIT_PATH_V24A = (
    ROOT / "experiments/eggroll_es_hpo/S6_V24_HYBRID_CHECKPOINT_AUDIT.json"
)
MODEL_SEAL_PATH_V24A = (
    ROOT / "experiments/eggroll_es_hpo/S6_V23A_INSERTION_MODEL_SEAL.json"
)
PANEL_MANIFEST_PATH_V24A = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v13/"
    "document_balanced_train_panels_v13.json"
)
LAYER_PLAN_PATH_V24A = (
    ROOT / "experiments/layer_plans/v23a_base_middle_late_dense.json"
)

SCHEMA_V24A = "eggroll-es-hybrid-backend-compatibility-preregistration-v24a"
AUDIT_FILE_SHA256_V24A = "07eee932313b74c1b00a250b3589b11dfef6a20268ef37c7cc3365233af48a43"
AUDIT_CONTENT_SHA256_V24A = "add4ee407561e9da6601babb6c1fc9ca8bc2ce00bb43f9c0d16efe11fefeb029"
MODEL_SEAL_FILE_SHA256_V24A = "96eeb236ea94678f57a530a27a471467d4b3d413d2e7be397e293b695cd4c440"
MODEL_SEAL_CONTENT_SHA256_V24A = "d4cf795408967aefbc77f841c47e6fe2fbe3cefc14a4a0fdb4bf73b2701326f9"
PANEL_MANIFEST_FILE_SHA256_V24A = "e555d9d6746cde6297cd3ab523b16dd7d78d81e2674447ee46d754ebfac52da7"
PANEL_MANIFEST_CONTENT_SHA256_V24A = "46cc98b694c98c1ee1c5456b855fb3b1db4534b3df2dcda69fc690a2d8a61bf5"
PANEL_BUNDLE_CONTENT_SHA256_V24A = "cc176a9b86c6447dcde8a11fd28d68c837d2119715126c57a3f37293fb0d492b"
TRAIN_SOURCE_FILE_SHA256_V24A = "f7127c38c7b540eaf9cf4349d1a1b8076e171da7f8ea43c11068ad1c311bb776"
TRAIN_SOURCE_ARROW_SHA256_V24A = "6b6fdfdd082f1de2bf1b4c78bd0a4154af5c709b26e46b0677dcde695d3b4cb6"
LAYER_PLAN_FILE_SHA256_V24A = "4dca69212f2eee1c5882a7f3de944e3dadd8de94b42e58e7d7495547e8b1c747"
LAYER_PLAN_SHA256_V24A = "07a155d1217b27ba1bf30e057024247236a812841c52bab401d465c9fdb5273f"
LOGICAL_SHAPE_ORDER_SHA256_V24A = "50f421bd54a01bb1a7399743d57b09d816bad2d729ec5d5971d80849d3d572dc"

BF16_CONFIG_SHA256_V24A = "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
BF16_INDEX_SHA256_V24A = "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83"
HYBRID_CONFIG_SHA256_V24A = "f65210af07523dcf66e33244ad16e1915ebf551a77a4d05375fdfdd5c79c8999"
HYBRID_INDEX_SHA256_V24A = "2a1b2447089e086ee12d669405aafdf9af9edf865f75a725dcab9f6211ebbb6c"
HYBRID_PROVENANCE_FILE_SHA256_V24A = "fd87e20bc312969c830226ce160e4ae2f734e58937f313675e9658a23faee49a"
HYBRID_PROVENANCE_CONTENT_SHA256_V24A = "d1be553015e76a0e546a9c142cfcfd42595f3989cf91204d96331eef7529176e"
HYBRID_OVERLAY_SHA256_V24A = "219c4138b8784d80b22e4e03bae39fb3c25f0a83a4059825cd8bd06a228ac927"

PERTURBATION_BASIS_SEED_V24A = 20260830
BOOTSTRAP_SEED_V24A = 20260831
CONFIRMATION_BASIS_SEED_V24A = 20260901
CONFIRMATION_BOOTSTRAP_SEED_V24A = 20260902
POPULATION_SIZE_V24A = 32
SIGMA_V24A = 0.0003
BOOTSTRAP_REPETITIONS_V24A = 50_000
FAMILY_HYPOTHESIS_COUNT_V24A = 34
PAIR_ORDER_V24A = ("pair_a", "pair_b")
ARM_ORDER_V24A = ("bf16_a", "hybrid_a", "bf16_b", "hybrid_b")
PANEL_NAMES_V24A = (
    "optimization_0", "optimization_1", "optimization_2",
    "train_screen_0", "train_screen_1",
)

QUALITY_ENDPOINT_THRESHOLDS_V24A = {
    "optimization_panel_cosine_median": 0.95,
    "optimization_panel_cosine_worst": 0.90,
    "optimization_panel_sign_agreement_median": 0.80,
    "optimization_panel_sign_agreement_worst": 0.75,
    "train_screen_cosine_median": 0.95,
    "train_screen_cosine_worst": 0.90,
    "train_screen_sign_agreement_median": 0.80,
    "train_screen_sign_agreement_worst": 0.75,
    "all_panel_cosine_median": 0.95,
    "all_panel_cosine_worst": 0.90,
    "all_panel_sign_agreement_median": 0.80,
    "all_panel_sign_agreement_worst": 0.75,
    "unperturbed_reward_delta_median": -0.01,
    "unperturbed_reward_delta_worst": -0.02,
    "unperturbed_row_reward_correlation": 0.995,
    "unperturbed_row_reward_mae_headroom": 0.0,
}
PRIOR_BASIS_CONTENT_SHA256_V24A = {
    "v13_v14_v16_v17_v18": "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11",
    "v15a": "6c358060c5f9a0a7b00e953bd230b18f915950f0233f38321e0e048a67ea05e7",
    "v15b": "97e9c5687677bd02365f77671141031ba2739018ed07ccd1bbb3eaabbc0a94f8",
    "v19a": "d4e46d7d51d5c82cfc981dad3b33db8a1766c70ad570ef931b12550d1bc7bf6c",
    "v20a": "b6d667c2f125f9d0be4d74ef536af03546fecb6c03f2838679f5a315a1ec9852",
    "v21a": "65970861cd06b53e52cf848b2c8b8961160bf9c68f6b1b9f4935a88ba8d314d2",
    "v22a": "f68624388ac0549ac82ba3d1e64a317233c42f900502a6f5c6d6f07071b4c60e",
    "v23a": "aad4ac2e82b55b13fc7a1019b89425d164e7ac8d0e6a8e4fd23c4bcc3f0757eb",
}


def canonical_sha256(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def perturbation_seeds_v24a():
    generator = np.random.default_rng(PERTURBATION_BASIS_SEED_V24A)
    seeds = generator.integers(
        0, 2**30, size=POPULATION_SIZE_V24A, dtype=np.int64,
    ).tolist()
    if len(seeds) != len(set(seeds)):
        raise RuntimeError("v24a perturbation basis contains duplicate seeds")
    return [int(value) for value in seeds]


def signed_wave_schedule_v24a():
    schedule = []
    for direction, seed in enumerate(perturbation_seeds_v24a()):
        signs = ("plus", "minus") if direction % 2 == 0 else ("minus", "plus")
        for sign in signs:
            schedule.append({
                "signed_wave_index": len(schedule),
                "direction_index": direction,
                "direction_seed": seed,
                "sign": sign,
                "negate": sign == "minus",
            })
    return schedule


def _validate_bound_sources_v24a():
    if file_sha256(AUDIT_PATH_V24A) != AUDIT_FILE_SHA256_V24A:
        raise RuntimeError("v24a hybrid audit file identity changed")
    audit = _load_json(AUDIT_PATH_V24A)
    if (
        audit.get("content_sha256_before_self_field") != AUDIT_CONTENT_SHA256_V24A
        or audit.get("selected_unit_count") != 35
        or audit.get("selected_scale_count_removed") != 25
        or audit.get("selected_element_count") != 142_999_552
        or audit.get("selected_byte_count") != 285_999_104
        or audit.get("rewritten_fp8_shard_count") != 4
        or audit.get("retained_tensor_count_in_rewritten_shards") != 6_170
        or audit.get("unaffected_hardlink_count") != 50
        or audit.get("all_selected_tensors_exact_bf16") is not True
        or audit.get("all_retained_rewritten_shard_tensors_exact_fp8") is not True
        or audit.get("all_unaffected_files_exact_hardlinks") is not True
        or audit.get("contains_dataset_or_evaluation_content") is not False
    ):
        raise RuntimeError("v24a hybrid audit content changed")
    if file_sha256(MODEL_SEAL_PATH_V24A) != MODEL_SEAL_FILE_SHA256_V24A:
        raise RuntimeError("v24a model seal file identity changed")
    model_seal = _load_json(MODEL_SEAL_PATH_V24A)
    if model_seal.get("content_sha256_before_self_field") != MODEL_SEAL_CONTENT_SHA256_V24A:
        raise RuntimeError("v24a model seal content changed")
    if file_sha256(PANEL_MANIFEST_PATH_V24A) != PANEL_MANIFEST_FILE_SHA256_V24A:
        raise RuntimeError("v24a train panel manifest changed")
    panel_manifest = _load_json(PANEL_MANIFEST_PATH_V24A)
    if panel_manifest.get("content_sha256_before_self_field") != PANEL_MANIFEST_CONTENT_SHA256_V24A:
        raise RuntimeError("v24a train panel content identity changed")
    if file_sha256(LAYER_PLAN_PATH_V24A) != LAYER_PLAN_FILE_SHA256_V24A:
        raise RuntimeError("v24a layer plan file identity changed")
    layer_plan = _load_json(LAYER_PLAN_PATH_V24A)
    if (
        layer_plan.get("plan_sha256") != LAYER_PLAN_SHA256_V24A
        or layer_plan.get("num_units") != 35
        or layer_plan.get("layers") != [20, 21, 22, 23]
        or layer_plan.get("model_config_sha256") != BF16_CONFIG_SHA256_V24A
        or not isinstance(layer_plan.get("units"), list)
        or len(layer_plan["units"]) != 35
        or len(set(layer_plan["units"])) != 35
    ):
        raise RuntimeError("v24a layer plan content changed")
    bf16 = CANONICAL_ROOT / "models/Qwen3.6-35B-A3B"
    hybrid = CANONICAL_ROOT / "models/Qwen3.6-35B-A3B-FP8-middle-late-BF16-v24"
    if (
        file_sha256(bf16 / "config.json") != BF16_CONFIG_SHA256_V24A
        or file_sha256(bf16 / "model.safetensors.index.json")
        != BF16_INDEX_SHA256_V24A
        or file_sha256(hybrid / "config.json") != HYBRID_CONFIG_SHA256_V24A
        or file_sha256(hybrid / "model.safetensors.index.json")
        != HYBRID_INDEX_SHA256_V24A
        or file_sha256(hybrid / "hybrid_selected_bf16_manifest.json")
        != HYBRID_PROVENANCE_FILE_SHA256_V24A
    ):
        raise RuntimeError("v24a runtime checkpoint identity changed")
    hybrid_provenance = _load_json(
        hybrid / "hybrid_selected_bf16_manifest.json"
    )
    if (
        hybrid_provenance.get("content_sha256_before_self_field")
        != HYBRID_PROVENANCE_CONTENT_SHA256_V24A
        or hybrid_provenance.get("overlay_file_sha256")
        != HYBRID_OVERLAY_SHA256_V24A
        or hybrid_provenance.get("layer_plan", {}).get("units")
        != layer_plan["units"]
    ):
        raise RuntimeError("v24a selected partition binding changed")
    return audit, model_seal, layer_plan


def build_preregistration_v24a():
    audit, _model_seal, layer_plan = _validate_bound_sources_v24a()
    seeds = perturbation_seeds_v24a()
    schedule = signed_wave_schedule_v24a()
    basis = {
        "basis_seed": PERTURBATION_BASIS_SEED_V24A,
        "population_size": POPULATION_SIZE_V24A,
        "sigma": SIGMA_V24A,
        "direction_seed_list_sha256": canonical_sha256(seeds),
        "signed_wave_count": len(schedule),
        "signed_wave_schedule_sha256": canonical_sha256(schedule),
        "generator": "numpy.default_rng(basis_seed).integers(0,2**30,size=32,dtype=int64)",
        "prior_basis_content_sha256": PRIOR_BASIS_CONTENT_SHA256_V24A,
        "distinct_from_every_listed_prior_basis": True,
    }
    basis["basis_content_sha256"] = canonical_sha256(basis)
    if basis["basis_content_sha256"] in set(PRIOR_BASIS_CONTENT_SHA256_V24A.values()):
        raise RuntimeError("v24a perturbation basis reused a prior basis")

    bf16_model = str(CANONICAL_ROOT / "models/Qwen3.6-35B-A3B")
    hybrid_model = str(
        CANONICAL_ROOT / "models/Qwen3.6-35B-A3B-FP8-middle-late-BF16-v24"
    )
    plan = {
        "path": str(CANONICAL_ROOT / "experiments/layer_plans/v23a_base_middle_late_dense.json"),
        "file_sha256": LAYER_PLAN_FILE_SHA256_V24A,
        "plan_sha256": LAYER_PLAN_SHA256_V24A,
        "layers": [20, 21, 22, 23],
        "num_units": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
        "selected_byte_count": 285_999_104,
        "logical_shape_order_sha256": LOGICAL_SHAPE_ORDER_SHA256_V24A,
    }
    arms = {
        "bf16_a": {"engine_rank": 0, "expected_gpu_id": 0, "pair": "pair_a", "backend": "bf16", "model_path": bf16_model},
        "hybrid_a": {"engine_rank": 1, "expected_gpu_id": 1, "pair": "pair_a", "backend": "hybrid_fp8_frozen_bf16_selected", "model_path": hybrid_model},
        "bf16_b": {"engine_rank": 2, "expected_gpu_id": 2, "pair": "pair_b", "backend": "bf16", "model_path": bf16_model},
        "hybrid_b": {"engine_rank": 3, "expected_gpu_id": 3, "pair": "pair_b", "backend": "hybrid_fp8_frozen_bf16_selected", "model_path": hybrid_model},
    }
    for arm in arms.values():
        arm["layer_plan"] = plan

    value = {
        "schema": SCHEMA_V24A,
        "experiment_name": "s6_v24a_hybrid_backend_train_only_compatibility",
        "preregistered_before_any_v24a_train_reward_scoring": True,
        "strict_train_only": True,
        "hybrid_checkpoint_audit": {
            "path": str(CANONICAL_ROOT / "experiments/eggroll_es_hpo/S6_V24_HYBRID_CHECKPOINT_AUDIT.json"),
            "file_sha256": AUDIT_FILE_SHA256_V24A,
            "content_sha256": AUDIT_CONTENT_SHA256_V24A,
            "selected_partition_exact_bf16": True,
            "retained_partition_exact_fp8": True,
            "unaffected_files_exact_hardlinks": True,
            "selected_unit_count": audit["selected_unit_count"],
            "selected_element_count": audit["selected_element_count"],
        },
        "model_contract": {
            "bf16": {
                "path": bf16_model,
                "config_sha256": BF16_CONFIG_SHA256_V24A,
                "index_sha256": BF16_INDEX_SHA256_V24A,
                "full_model_seal_file_sha256": MODEL_SEAL_FILE_SHA256_V24A,
                "full_model_seal_content_sha256": MODEL_SEAL_CONTENT_SHA256_V24A,
            },
            "hybrid": {
                "path": hybrid_model,
                "config_sha256": HYBRID_CONFIG_SHA256_V24A,
                "index_sha256": HYBRID_INDEX_SHA256_V24A,
                "provenance_file_sha256": HYBRID_PROVENANCE_FILE_SHA256_V24A,
                "provenance_content_sha256": HYBRID_PROVENANCE_CONTENT_SHA256_V24A,
                "overlay_sha256": HYBRID_OVERLAY_SHA256_V24A,
            },
            "same_exact_35_unit_bf16_selected_partition": True,
            "only_frozen_complement_backend_differs": True,
        },
        "arms": arms,
        "pairing": {
            "pair_order": list(PAIR_ORDER_V24A),
            "pairs": {
                "pair_a": {"bf16": "bf16_a", "hybrid": "hybrid_a"},
                "pair_b": {"bf16": "bf16_b", "hybrid": "hybrid_b"},
            },
            "same_direction_sign_rows_and_sampling_all_four_arms": True,
        },
        "fresh_basis": basis,
        "panel_contract": {
            "manifest_file_sha256": PANEL_MANIFEST_FILE_SHA256_V24A,
            "manifest_content_sha256": PANEL_MANIFEST_CONTENT_SHA256_V24A,
            "panel_bundle_content_sha256": PANEL_BUNDLE_CONTENT_SHA256_V24A,
            "train_source_file_sha256": TRAIN_SOURCE_FILE_SHA256_V24A,
            "train_source_arrow_sha256": TRAIN_SOURCE_ARROW_SHA256_V24A,
            "panel_names": list(PANEL_NAMES_V24A),
            "optimization_panels": list(PANEL_NAMES_V24A[:3]),
            "untouched_train_screen_panels": list(PANEL_NAMES_V24A[3:]),
            "panel_count": 5,
            "rows_per_panel": 56,
            "requests_per_arm_per_signed_wave": 280,
            "all_panels_pinned_before_results": True,
            "train_only": True,
        },
        "runtime": {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "engine_arm_mapping": {str(index): arm for index, arm in enumerate(ARM_ORDER_V24A)},
            "one_model_resident_per_gpu": True,
            "all_four_gpus_score_every_signed_wave": True,
            "sigma": SIGMA_V24A,
            "alpha": 0.0,
            "signed_wave_count": 64,
            "requests_per_engine_per_signed_wave": 280,
            "requests_all_engines_per_signed_wave": 1_120,
            "perturbed_requests_per_engine": 17_920,
            "perturbed_requests_all_engines": 71_680,
            "unperturbed_pre_post_requests_per_engine": 560,
            "unperturbed_pre_post_requests_all_engines": 2_240,
            "total_generation_requests": 73_920,
            "selected_reference_identity_equal_all_four_arms_before_scoring": True,
            "exact_selected_restore_every_arm_every_signed_wave": True,
            "unselected_partition_identity_unchanged_every_wave": True,
            "pre_post_unperturbed_dense_result_hash_equal_per_arm": True,
            "generation_timing": "per-arm synchronized monotonic wall time around generate only; exclude perturb restore bootstrap and controller",
            "resident_memory": "per-GPU NVML used bytes after model load and warmup before first score",
        },
        "analysis": {
            "pair_count": 2,
            "quality_endpoint_count_per_pair": 16,
            "quality_endpoint_thresholds": QUALITY_ENDPOINT_THRESHOLDS_V24A,
            "unperturbed_row_reward_mae_ceiling": 0.02,
            "familywise_endpoint_transforms": {
                "cosine_sign_correlation_and_reward_delta": "observed metric minus its named threshold",
                "unperturbed_row_reward_mae_headroom": "0.02 minus observed mean absolute error",
                "speed": "paired BF16_seconds / hybrid_seconds minus 1.05",
                "pass": "one-sided 0.05/34 bootstrap lower confidence bound is >= 0",
            },
            "speed_endpoint": "median paired signed-wave BF16_seconds / hybrid_seconds",
            "speedup_threshold_per_pair": 1.05,
            "speed_endpoint_count": 2,
            "memory_reduction_threshold_per_pair": 0.40,
            "memory_reduction_definition": "1 - hybrid_resident_bytes / bf16_resident_bytes",
            "family_hypothesis_count": FAMILY_HYPOTHESIS_COUNT_V24A,
            "bootstrap_seed": BOOTSTRAP_SEED_V24A,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS_V24A,
            "one_sided_familywise_quantile": 0.05 / FAMILY_HYPOTHESIS_COUNT_V24A,
            "quantile_method": "linear",
            "paired_direction_and_stratified_train_row_bootstrap": True,
            "multiplicity_covers_all_quality_and_speed_endpoints_both_pairs": True,
            "raw_scores_bootstrap_draws_and_replicates_memory_only": True,
        },
        "gate": {
            "pair_pass": "all 16 quality and own speed familywise LCBs meet thresholds, memory reduction >= 0.40, and every runtime integrity audit is true",
            "global_pass": "both pair_a and pair_b pass",
            "pass_authority": "authorize_only_separate_fresh_basis_train_only_mapping_reversed_confirmation",
            "no_pass_decision": "retain_full_bf16_backend",
            "confirmation_engine_backend_mapping": {"0": "hybrid", "1": "bf16", "2": "hybrid", "3": "bf16"},
            "confirmation_basis_seed": CONFIRMATION_BASIS_SEED_V24A,
            "confirmation_bootstrap_seed": CONFIRMATION_BOOTSTRAP_SEED_V24A,
            "direct_backend_substitution_authorized": False,
            "model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
        "excluded_inputs": {
            "heldout_validation_ood_or_benchmarks": True,
            "evaluation_reports_or_rows": True,
            "v23_results": True,
            "v349_or_later_curator_snapshots": True,
            "persisted_raw_reward_or_response_vectors": True,
        },
        "authority": {
            "train_only_raw_scoring": True,
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_allowed": False,
            "dataset_promotion_allowed": False,
            "backend_adoption_allowed": False,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return value


def validate_preregistration_v24a(value):
    expected = build_preregistration_v24a()
    if value != expected:
        raise RuntimeError("v24a preregistration content changed")
    if value.get("content_sha256_before_self_field") != canonical_sha256({
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }):
        raise RuntimeError("v24a preregistration self identity changed")
    return value


def main():
    value = build_preregistration_v24a()
    PREREGISTRATION_PATH_V24A.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(value, indent=2, sort_keys=True))
    return value


if __name__ == "__main__":
    main()
