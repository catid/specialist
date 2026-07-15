#!/usr/bin/env python3
"""Fresh strictly train-only V23A insertion-location preregistration."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path

import es_layer_plan
import build_eggroll_es_insertion_model_seal_v23a as model_seal_v23a
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
OUTPUT_PATH_V23A = (
    ROOT / "experiments/eggroll_es_hpo/"
    "S6_V23A_INSERTION_LOCATION_STABILITY_PREREGISTRATION.json"
).resolve()
PLAN_DIRECTORY_V23A = (ROOT / "experiments/layer_plans").resolve()
MODEL_SEAL_PATH_V23A = model_seal_v23a.OUTPUT_PATH_V23A
MODEL_SEAL_COMMIT_V23A = "00471e1c44b78813d02f0a8895c8e014a75dcc4e"
MODEL_SEAL_FILE_SHA256_V23A = (
    "96eeb236ea94678f57a530a27a471467d4b3d413d2e7be397e293b695cd4c440"
)
MODEL_SEAL_CONTENT_SHA256_V23A = (
    "d4cf795408967aefbc77f841c47e6fe2fbe3cefc14a4a0fdb4bf73b2701326f9"
)
MODEL_SEAL_SOURCE_HASHES_V23A = {
    "build_eggroll_es_insertion_model_seal_v23a.py": (
        "7137b944c9afd8955d0a66bc8a1813393ab05d7815094152c0d8176af028a9de"
    ),
    "test_build_eggroll_es_insertion_model_seal_v23a.py": (
        "452cfffa02e29b1fcb6f4958a1f7c4825a068851235d5d79de3ca2b7b85951aa"
    ),
    "experiments/eggroll_es_hpo/S6_V23A_INSERTION_MODEL_SEAL.json": (
        MODEL_SEAL_FILE_SHA256_V23A
    ),
}

EVIDENCE_BINDINGS_V23A = {
    "v13_train_panel_aggregate": {
        "path": ROOT / "experiments/eggroll_es_hpo/"
        "S6_V13B_TRAIN_PANEL_AGGREGATE_EVIDENCE_V14A.json",
        "file_sha256": (
            "d367c9c4de1e1f3526ddb3dfba2f5bf24efc77cbccf951f7359eb1969fcd7b54"
        ),
        "content_sha256": (
            "06f662574013345a6c777af8688a38f3941286d9e11a427ed3342de53451b1e3"
        ),
    },
    "v15a_back_positive_aggregate": {
        "path": ROOT / "experiments/eggroll_es_hpo/"
        "S6_V15A_BACK_PLAN_POSITIVE_AGGREGATE_EVIDENCE_V15B.json",
        "file_sha256": (
            "1e14abee9e1514915bc241c8f6caacbe1bb7103e1c69a9afdde1f9ce13661ae1"
        ),
        "content_sha256": (
            "c9ab854c8417c4f6c74e5fe54de29e2f6a8b222b4c7fc454b5f22b183c3b08b2"
        ),
    },
    "v15b_back_negative_aggregate": {
        "path": ROOT / "experiments/eggroll_es_hpo/"
        "S6_V15B_BACK_PLAN_CONFIRMATION_NEGATIVE_AGGREGATE_EVIDENCE_V16.json",
        "file_sha256": (
            "1d5dac57a8fdf9e117d4c0c0f90a5eb98f4d08eea9662df92849fdbcd18c9099"
        ),
        "content_sha256": (
            "5e96588eb9fcacebfac48211a63d6ee9a106033a696ee6d21afc6d07b231bd28"
        ),
    },
    "v22a_dataset_negative_aggregate": {
        "path": ROOT / "experiments/eggroll_es_hpo/"
        "S6_V22A_V341_MATCHED_REPLACEMENT_NEGATIVE_EVIDENCE.json",
        "file_sha256": (
            "408d48a8eb9a63da9f12062d94b5249c88cc04d254eb553230e23cae516a1955"
        ),
        "content_sha256": (
            "67eb3c39aa8d13a8924a2880c04f021f8ad27b9fcfb82eba9731ab195f5d3318"
        ),
    },
}
EVIDENCE_SCHEMAS_V23A = {
    "v13_train_panel_aggregate": "eggroll-es-v13b-train-panel-aggregate-evidence-v14a",
    "v15a_back_positive_aggregate": "eggroll-es-v15a-back-plan-positive-aggregate-evidence-v15b",
    "v15b_back_negative_aggregate": "eggroll-es-v15b-back-plan-confirmation-negative-aggregate-evidence-v16",
    "v22a_dataset_negative_aggregate": "eggroll-es-v341-matched-replacement-negative-evidence-v22a",
}

ARM_ORDER_V23A = model_seal_v23a.ARM_ORDER_V23A
CANDIDATE_ARMS_V23A = ARM_ORDER_V23A[1:]
TARGET_LAYERS_V23A = {
    arm: tuple(spec["target_layers"])
    for arm, spec in model_seal_v23a.MODEL_SPECS_V23A.items()
}
ENGINE_ARM_MAPPING_V23A = {
    "0": "base_middle_late", "1": "insert_front_e005",
    "2": "insert_middle_e005", "3": "insert_back_e005",
}
CONFIRMATION_ENGINE_ARM_MAPPING_V23A = {
    "0": "insert_back_e005", "1": "base_middle_late",
    "2": "insert_front_e005", "3": "insert_middle_e005",
}
GRADIENT_ENDPOINTS_V23A = (
    "optimization_pairwise_cosine_median",
    "optimization_pairwise_cosine_worst",
    "optimization_pairwise_sign_agreement_median",
    "optimization_pairwise_sign_agreement_worst",
    "aggregate_to_optimization_cosine_median",
    "aggregate_to_optimization_cosine_worst",
    "aggregate_to_optimization_sign_agreement_median",
    "aggregate_to_optimization_sign_agreement_worst",
    "train_screen_cosine_median",
    "train_screen_cosine_worst",
    "train_screen_sign_agreement_median",
    "train_screen_sign_agreement_worst",
)
REFERENCE_ENDPOINTS_V23A = (
    "unperturbed_reward_delta_median",
    "unperturbed_reward_delta_worst",
    "unperturbed_loss_compatibility_median",
    "unperturbed_loss_compatibility_worst",
)
ALL_ENDPOINTS_V23A = GRADIENT_ENDPOINTS_V23A + REFERENCE_ENDPOINTS_V23A
FAMILY_HYPOTHESIS_COUNT_V23A = len(CANDIDATE_ARMS_V23A) * len(ALL_ENDPOINTS_V23A)
PERTURBATION_BASIS_SEED_V23A = 20260826
BOOTSTRAP_SEED_V23A = 20260827
BOOTSTRAP_REPETITIONS_V23A = 50_000
POPULATION_SIZE_V23A = 32
SIGMA_V23A = 0.0003
ALPHA_V23A = 0.0
PRIOR_BASIS_CONTENT_SHA256_V23A = {
    "v13_v14_v16_v17_v18": (
        "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11"
    ),
    "v15a": "6c358060c5f9a0a7b00e953bd230b18f915950f0233f38321e0e048a67ea05e7",
    "v15b": "97e9c5687677bd02365f77671141031ba2739018ed07ccd1bbb3eaabbc0a94f8",
    "v19a": "d4e46d7d51d5c82cfc981dad3b33db8a1766c70ad570ef931b12550d1bc7bf6c",
    "v20a": "b6d667c2f125f9d0be4d74ef536af03546fecb6c03f2838679f5a315a1ec9852",
    "v21a": "65970861cd06b53e52cf848b2c8b8961160bf9c68f6b1b9f4935a88ba8d314d2",
    "v22a": "f68624388ac0549ac82ba3d1e64a317233c42f900502a6f5c6d6f07071b4c60e",
}
PERTURBATION_BASIS_SHA256_V23A = (
    "aad4ac2e82b55b13fc7a1019b89425d164e7ac8d0e6a8e4fd23c4bcc3f0757eb"
)
PERTURBATION_SEED_LIST_SHA256_V23A = (
    "03aeeb630f27c865cb222c04d4e8ee4d3de3385f426634e660454b6922cad178"
)
SIGNED_WAVE_SCHEDULE_SHA256_V23A = (
    "c1875a188c91b81623d8242cdfc3b3737329b77329756e2b892acdb47b0f4ede"
)
FORBIDDEN_CONTENT_KEYS_V23A = {
    "question", "questions", "answer", "answers", "prompt", "prompts",
    "prompt_token_ids", "token_ids", "tokens", "row_content", "responses",
    "unit_scores", "bootstrap_draws", "bootstrap_replicates", "coefficients",
    "heldout", "holdout", "validation", "ood", "benchmark", "eval",
}


def canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def file_sha256(path: Path) -> str:
    return model_seal_v23a.file_sha256(path)


def _without_self(value):
    return {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _recursive_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key).lower()
            yield from _recursive_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _recursive_keys(item)


def perturbation_basis_v23a():
    seeds = []
    for index in range(POPULATION_SIZE_V23A):
        digest = hashlib.sha256(
            f"eggroll-es-v23a-insertion-location|{PERTURBATION_BASIS_SEED_V23A}|{index}".encode()
        ).digest()
        seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
        if seed == 0 or seed in seeds:
            raise RuntimeError("v23a fresh basis seed collision")
        seeds.append(seed)
    value = {
        "schema": "eggroll-es-insertion-location-basis-v23a",
        "basis_seed": PERTURBATION_BASIS_SEED_V23A,
        "population_size": POPULATION_SIZE_V23A,
        "direction_seeds": seeds,
    }
    _require(
        canonical_sha256(value) == PERTURBATION_BASIS_SHA256_V23A
        and canonical_sha256(seeds) == PERTURBATION_SEED_LIST_SHA256_V23A
        and canonical_sha256(value) not in set(PRIOR_BASIS_CONTENT_SHA256_V23A.values()),
        "v23a perturbation basis reused a prior basis",
    )
    return value


def signed_wave_schedule_v23a():
    schedule = []
    for direction_index, seed in enumerate(perturbation_basis_v23a()["direction_seeds"]):
        signs = ("plus", "minus") if direction_index % 2 == 0 else ("minus", "plus")
        for sign in signs:
            schedule.append({
                "signed_wave_index": len(schedule),
                "direction_index": direction_index,
                "direction_seed": seed,
                "sign": sign,
                "negate": sign == "minus",
                "same_direction_seed_all_four_arms": True,
            })
    _require(
        len(schedule) == 64
        and [item["signed_wave_index"] for item in schedule] == list(range(64)),
        "v23a signed wave schedule changed",
    )
    return schedule


def _verify_model_seal_v23a():
    for relative, digest in MODEL_SEAL_SOURCE_HASHES_V23A.items():
        committed = subprocess.check_output(
            ["git", "show", f"{MODEL_SEAL_COMMIT_V23A}:{relative}"], cwd=ROOT
        )
        _require(
            hashlib.sha256(committed).hexdigest() == digest
            and file_sha256(ROOT / relative) == digest,
            "v23a committed model seal input changed",
        )
    value = json.loads(MODEL_SEAL_PATH_V23A.read_text(encoding="utf-8"))
    model_seal_v23a.validate_model_seal_v23a(value)
    _require(
        file_sha256(MODEL_SEAL_PATH_V23A) == MODEL_SEAL_FILE_SHA256_V23A
        and value["content_sha256_before_self_field"]
        == MODEL_SEAL_CONTENT_SHA256_V23A,
        "v23a frozen model seal identity changed",
    )
    return value


def _load_aggregate_evidence_v23a():
    result = {}
    for name, binding in EVIDENCE_BINDINGS_V23A.items():
        path = binding["path"].resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        _require(
            file_sha256(path) == binding["file_sha256"]
            and value.get("schema") == EVIDENCE_SCHEMAS_V23A[name]
            and value.get("content_sha256_before_self_field")
            == binding["content_sha256"]
            and value.get("content_sha256_before_self_field")
            == canonical_sha256(_without_self(value))
            and not (
                FORBIDDEN_CONTENT_KEYS_V23A & set(_recursive_keys(value))
            ),
            "v23a aggregate-only evidence binding changed",
        )
        result[name] = {
            "path": str(path), "file_sha256": binding["file_sha256"],
            "content_sha256": binding["content_sha256"],
            "schema": value["schema"],
        }
    return result


def _plan_v23a(arm, model_path, layers):
    value = es_layer_plan.plan_manifest(
        model_path, "custom", ["dense"], custom_layers=list(layers)
    )
    value["plan"] = f"v23a_{arm}"
    value.pop("plan_sha256", None)
    value["plan_sha256"] = canonical_sha256(value)
    return value


def _plan_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _load_selected_tensor_descriptor(model_path, unit, index):
    shard = index["weight_map"][unit]
    with model_seal_v23a.safe_open(model_path / shard, framework="pt") as source:
        tensor = source.get_tensor(unit)
    return {
        "shape": list(tensor.shape), "dtype": str(tensor.dtype),
        "elements": tensor.numel(),
    }


def layer_plans_v23a(model_seal):
    plans = {}
    logical_descriptors = {}
    for arm in ARM_ORDER_V23A:
        item = model_seal["arms"][arm]
        model_path = Path(item["path"]).resolve()
        layers = TARGET_LAYERS_V23A[arm]
        plan = _plan_v23a(arm, model_path, layers)
        index = json.loads(
            (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        normalized = []
        for unit in plan["units"]:
            layer = int(unit.split(".layers.", 1)[1].split(".", 1)[0])
            motif_position = layers.index(layer)
            suffix = unit.split(f".layers.{layer}.", 1)[1]
            normalized.append({
                "motif_position": motif_position, "suffix": suffix,
                **_load_selected_tensor_descriptor(model_path, unit, index),
            })
        _require(
            len(normalized) == 35
            and sum(item["elements"] for item in normalized) == 142_999_552,
            "v23a selected motif capacity changed",
        )
        logical_descriptors[arm] = normalized
        path = (PLAN_DIRECTORY_V23A / f"v23a_{arm}_dense.json").resolve()
        plans[arm] = {
            "path": str(path), "file_sha256": hashlib.sha256(_plan_bytes(plan)).hexdigest(),
            "plan_sha256": plan["plan_sha256"],
            "model_config_sha256": plan["model_config_sha256"],
            "layers": list(layers), "num_units": 35,
            "selected_element_count": 142_999_552,
            "logical_shape_order_sha256": canonical_sha256(normalized),
            "manifest": plan,
        }
    descriptor_hashes = {
        canonical_sha256(value) for value in logical_descriptors.values()
    }
    _require(
        len(descriptor_hashes) == 1,
        "v23a logical perturbation tensor shapes or order differ across arms",
    )
    return plans, descriptor_hashes.pop()


def _panel_contract_v23a():
    bundle = anchor_v13.load_panel_bundle_v13()
    _require(
        bundle["content_sha256_before_self_field"]
        == anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
        and tuple(bundle["panels"]) == anchor_v13.PANEL_NAMES_V13,
        "v23a V13 train-only panel identity changed",
    )
    return {
        "panel_bundle_content_sha256": anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13,
        "manifest_file_sha256": anchor_v13.PANEL_MANIFEST_FILE_SHA256_V13,
        "manifest_content_sha256": anchor_v13.PANEL_MANIFEST_CONTENT_SHA256_V13,
        "train_source_file_sha256": anchor_v13.TRAIN_SOURCE_FILE_SHA256_V13,
        "train_source_arrow_sha256": anchor_v13.TRAIN_ARROW_FILE_SHA256_V13,
        "panel_names": list(anchor_v13.PANEL_NAMES_V13),
        "optimization_panels": list(anchor_v13.OPTIMIZATION_PANELS_V13),
        "untouched_train_screen_panels": list(anchor_v13.TRAIN_SCREENS_V13),
        "rows_per_panel": 56, "panel_count": 5,
        "requests_per_arm_per_signed_wave": 280,
        "all_panels_pinned_before_results": True,
        "train_only": True,
    }


def build_preregistration_v23a():
    model_seal = _verify_model_seal_v23a()
    evidence = _load_aggregate_evidence_v23a()
    plans, shape_order_sha = layer_plans_v23a(model_seal)
    panel = _panel_contract_v23a()
    basis = perturbation_basis_v23a()
    schedule = signed_wave_schedule_v23a()
    value = {
        "schema": "eggroll-es-insertion-location-stability-preregistration-v23a",
        "experiment_name": "insertion_location_stability_v23a_authoritative_raw",
        "preregistered_before_results": True,
        "strict_train_only": True,
        "model_seal": {
            "commit": MODEL_SEAL_COMMIT_V23A,
            "path": str(MODEL_SEAL_PATH_V23A),
            "file_sha256": MODEL_SEAL_FILE_SHA256_V23A,
            "content_sha256": MODEL_SEAL_CONTENT_SHA256_V23A,
        },
        "aggregate_evidence": evidence,
        "arms": {
            arm: {
                "engine_rank": int(next(
                    rank for rank, mapped in ENGINE_ARM_MAPPING_V23A.items()
                    if mapped == arm
                )),
                "expected_gpu_id": int(next(
                    rank for rank, mapped in ENGINE_ARM_MAPPING_V23A.items()
                    if mapped == arm
                )),
                "model_path": model_seal["arms"][arm]["path"],
                "model_directory_fingerprint_sha256": model_seal["arms"][arm][
                    "all_files_fingerprint_sha256"
                ],
                "target_layers": list(TARGET_LAYERS_V23A[arm]),
                "epsilon": model_seal["arms"][arm]["epsilon"],
                "layer_plan": {
                    key: plans[arm][key] for key in (
                        "path", "file_sha256", "plan_sha256",
                        "model_config_sha256", "layers", "num_units",
                        "selected_element_count", "logical_shape_order_sha256",
                    )
                },
            } for arm in ARM_ORDER_V23A
        },
        "cross_arm_perturbation_contract": {
            "logical_shape_order_sha256": shape_order_sha,
            "source_unit_count_per_arm": 35,
            "runtime_selected_parameter_count_per_arm": 23,
            "selected_element_count_per_arm": 142_999_552,
            "same_logical_tensor_shape_and_order_all_arms": True,
            "same_direction_seed_all_arms_every_signed_wave": True,
        },
        "panel_contract": panel,
        "fresh_basis": {
            "basis_seed": PERTURBATION_BASIS_SEED_V23A,
            "population_size": POPULATION_SIZE_V23A,
            "direction_seed_list_sha256": canonical_sha256(basis["direction_seeds"]),
            "basis_content_sha256": canonical_sha256(basis),
            "prior_basis_content_sha256": copy.deepcopy(
                PRIOR_BASIS_CONTENT_SHA256_V23A
            ),
            "distinct_from_every_listed_prior_basis": True,
            "signed_wave_count": 64,
            "signed_wave_schedule_sha256": canonical_sha256(schedule),
        },
        "runtime": {
            "sigma": SIGMA_V23A, "alpha": ALPHA_V23A,
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "engine_arm_mapping": copy.deepcopy(ENGINE_ARM_MAPPING_V23A),
            "one_distinct_model_arm_resident_per_gpu": True,
            "all_four_gpus_score_every_signed_wave": True,
            "requests_per_engine_per_signed_wave": 280,
            "requests_all_engines_per_signed_wave": 1_120,
            "requests_per_engine_all_signed_waves": 17_920,
            "requests_all_engines_all_signed_waves": 71_680,
            "unperturbed_reference_requests_per_engine": 280,
            "unperturbed_reference_requests_all_engines": 1_120,
            "same_fixed_requests_all_arms_and_waves": True,
            "restore_once_per_arm_per_signed_wave": True,
            "exact_restore_and_unselected_identity_required_per_arm": True,
            "pre_post_unperturbed_reference_probe_equal_per_arm": True,
        },
        "reference_compatibility": {
            "teacher_forced_gold_answer_logprobs": True,
            "temperature": 0.0, "top_p": 1.0, "max_tokens": 1,
            "prompt_logprobs": 1, "detokenize": False,
            "reward": "weighted_panel_mean_gold_answer_token_logprob",
            "loss": "negative_weighted_panel_mean_gold_answer_token_logprob",
            "paired_candidate_minus_base_same_train_requests": True,
            "endpoints": list(REFERENCE_ENDPOINTS_V23A),
            "zero_noninferiority_margin": True,
            "location_cannot_advance_if_any_reference_endpoint_fails": True,
        },
        "analysis": {
            "candidate_location_count": 3,
            "gradient_endpoint_count_per_location": 12,
            "reference_endpoint_count_per_location": 4,
            "endpoint_count_per_location": 16,
            "family_hypothesis_count": FAMILY_HYPOTHESIS_COUNT_V23A,
            "gradient_endpoints": list(GRADIENT_ENDPOINTS_V23A),
            "reference_endpoints": list(REFERENCE_ENDPOINTS_V23A),
            "bootstrap_seed": BOOTSTRAP_SEED_V23A,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS_V23A,
            "paired_same_direction_and_train_row_draws_all_arms": True,
            "one_sided_familywise_quantile": 0.05 / FAMILY_HYPOTHESIS_COUNT_V23A,
            "noninferiority_margin": 0.0,
            "multiplicity_covers_all_three_locations_and_all_endpoints": True,
        },
        "gate": {
            "per_location_pass": (
                "all 12 gradient and all 4 unperturbed reference familywise LCBs "
                "are >= 0 with every runtime integrity audit true"
            ),
            "eligible_location_selection": (
                "among passing locations maximize minimum of 16 familywise LCBs; "
                "then mean LCB; then fixed front,middle,back order"
            ),
            "no_location_pass_decision": "retain_v13_base_middle_late_recipe",
            "pass_authority": (
                "authorize_only_separate_fresh_basis_train_only_confirmation"
            ),
            "confirmation_engine_arm_mapping": copy.deepcopy(
                CONFIRMATION_ENGINE_ARM_MAPPING_V23A
            ),
            "confirmation_requires_permuted_arm_gpu_mapping": True,
            "direct_model_update_authorized": False,
            "checkpoint_write_authorized": False,
            "evaluation_authorized": False,
            "dataset_promotion_authorized": False,
        },
        "authority": {
            "train_only_raw_scoring": True,
            "model_update_allowed": False,
            "checkpoint_write_allowed": False,
            "evaluation_allowed": False,
            "dataset_promotion_allowed": False,
        },
        "excluded_inputs": {
            "old_insert_training_journals": True,
            "old_insert_probe_rows_or_details": True,
            "old_eval_reports": True,
            "gen20_or_gen30_fp32_masters": True,
            "heldout_validation_ood_or_benchmarks": True,
            "persisted_response_vectors_or_unit_scores": True,
        },
    }
    value["content_sha256_before_self_field"] = canonical_sha256(value)
    return validate_preregistration_v23a(value)


def validate_preregistration_v23a(value):
    basis_hash = value.get("fresh_basis", {}).get("basis_content_sha256")
    _require(
        isinstance(value, dict)
        and set(value) == {
            "schema", "experiment_name", "preregistered_before_results",
            "strict_train_only", "model_seal", "aggregate_evidence", "arms",
            "cross_arm_perturbation_contract", "panel_contract", "fresh_basis",
            "runtime", "reference_compatibility", "analysis", "gate",
            "authority", "excluded_inputs", "content_sha256_before_self_field",
        }
        and value.get("schema")
        == "eggroll-es-insertion-location-stability-preregistration-v23a"
        and value.get("content_sha256_before_self_field")
        == canonical_sha256(_without_self(value))
        and value.get("preregistered_before_results") is True
        and value.get("strict_train_only") is True
        and list(value.get("arms", {})) == list(ARM_ORDER_V23A)
        and value.get("cross_arm_perturbation_contract", {}).get(
            "same_logical_tensor_shape_and_order_all_arms"
        ) is True
        and value.get("panel_contract", {}).get("panel_count") == 5
        and value.get("panel_contract", {}).get("rows_per_panel") == 56
        and value.get("panel_contract", {}).get("train_only") is True
        and basis_hash == PERTURBATION_BASIS_SHA256_V23A
        and basis_hash not in set(PRIOR_BASIS_CONTENT_SHA256_V23A.values())
        and value.get("runtime", {}).get("engine_arm_mapping")
        == ENGINE_ARM_MAPPING_V23A
        and value.get("runtime", {}).get("requests_all_engines_all_signed_waves")
        == 71_680
        and value.get("fresh_basis", {}).get("signed_wave_schedule_sha256")
        == SIGNED_WAVE_SCHEDULE_SHA256_V23A
        and value.get("runtime", {}).get("all_four_gpus_score_every_signed_wave")
        is True
        and value.get("reference_compatibility", {}).get(
            "location_cannot_advance_if_any_reference_endpoint_fails"
        ) is True
        and value.get("analysis", {}).get("family_hypothesis_count") == 48
        and value.get("analysis", {}).get("one_sided_familywise_quantile")
        == 0.05 / 48
        and value.get("analysis", {}).get(
            "multiplicity_covers_all_three_locations_and_all_endpoints"
        ) is True
        and value.get("gate", {}).get("confirmation_engine_arm_mapping")
        == CONFIRMATION_ENGINE_ARM_MAPPING_V23A
        and value.get("gate", {}).get(
            "confirmation_requires_permuted_arm_gpu_mapping"
        ) is True
        and all(value.get("authority", {}).get(key) is False for key in (
            "model_update_allowed", "checkpoint_write_allowed",
            "evaluation_allowed", "dataset_promotion_allowed",
        ))
        and all(value.get("excluded_inputs", {}).values()),
        "v23a insertion stability preregistration changed",
    )
    return value


def _exclusive_write(path, raw):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"immutable v23a output already exists: {path.name}") from error
    with os.fdopen(descriptor, "wb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_PATH_V23A))
    args = parser.parse_args(argv)
    if Path(args.output).resolve() != OUTPUT_PATH_V23A:
        raise ValueError("v23a preregistration output path changed")
    value = build_preregistration_v23a()
    model_seal = json.loads(MODEL_SEAL_PATH_V23A.read_text(encoding="utf-8"))
    plans, _shape = layer_plans_v23a(model_seal)
    for item in plans.values():
        _exclusive_write(Path(item["path"]), _plan_bytes(item["manifest"]))
    _exclusive_write(
        OUTPUT_PATH_V23A,
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(),
    )
    result = {
        "schema": "eggroll-es-insertion-location-preregistration-build-v23a",
        "path": str(OUTPUT_PATH_V23A),
        "file_sha256": file_sha256(OUTPUT_PATH_V23A),
        "content_sha256": value["content_sha256_before_self_field"],
        "layer_plan_file_sha256": {
            arm: item["file_sha256"] for arm, item in plans.items()
        },
        "gpu_launched": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
