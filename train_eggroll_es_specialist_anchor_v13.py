#!/usr/bin/env python3
"""Five-panel, train-only resident-sign diagnostic for V13."""

from __future__ import annotations

import copy
import json
import math
import os
import statistics
import sys
from pathlib import Path

import eggroll_es_train_panel_sampler_v13 as panel_sampler
import eggroll_es_worker_v13 as worker_v13
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v11 as anchor_v11
import train_eggroll_es_specialist_anchor_v11c as anchor_v11c


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = (
    "eggroll_es_worker_v13.TrainPanelDiagnosticWorkerExtensionV13"
)
PANEL_MANIFEST_PATH_V13 = (
    ROOT / "experiments/eggroll_es_hpo/train_panel_sampling_v13/"
    "document_balanced_train_panels_v13.json"
).resolve()
PANEL_MANIFEST_FILE_SHA256_V13 = (
    "e555d9d6746cde6297cd3ab523b16dd7d78d81e2674447ee46d754ebfac52da7"
)
PANEL_MANIFEST_CONTENT_SHA256_V13 = (
    "46cc98b694c98c1ee1c5456b855fb3b1db4534b3df2dcda69fc690a2d8a61bf5"
)
TRAIN_SOURCE_PATH_V13 = panel_sampler.DEFAULT_SOURCE.resolve()
TRAIN_SOURCE_FILE_SHA256_V13 = panel_sampler.SOURCE_SHA256
TRAIN_ARROW_FILE_SHA256_V13 = panel_sampler.SOURCE_ARROW_SHA256
PANEL_NAMES_V13 = tuple(panel_sampler.PANEL_NAMES)
OPTIMIZATION_PANELS_V13 = PANEL_NAMES_V13[:3]
TRAIN_SCREENS_V13 = PANEL_NAMES_V13[3:]
PANEL_RAW_QA_SHA256_V13 = {
    "optimization_0": "2aba1e40e786249c844b7a0a30fe2d12e4b16c3aa3ce5a58199b4d95488b6044",
    "optimization_1": "3d7017dd0b82179c21ad1d0d245c4ee5610cbb9d6558602f4996df69524c1f33",
    "optimization_2": "bb5ee713b0e81b21736a1f6aa8ef8e83ddea469db3eee99a0c4e231f2f164ec9",
    "train_screen_0": "ed97a18f25ad03ea8e268495fdfb24d97737bfa0d13dc24e0c0851b8eba04ba9",
    "train_screen_1": "39303781498cec0e013119fdccb37c19a2e8d919b8b0ade1e78190ce5c3e078b",
}
PANEL_ORDERED_ROW_SHA256_V13 = {
    "optimization_0": "680f93cde737ba1a18539d5dd58c756ff6f3436bdffa247cb05174926f944c1d",
    "optimization_1": "84adafc1bcf9ac55fd465c1d85f344349ba2b12d22b2d1c3515a7ec3f59a55f7",
    "optimization_2": "d54b92cf9102276ad80345f2d182267c18664a86fda24dc8e159396b2460b230",
    "train_screen_0": "b0fda37f28f3f3d0d2a043ba5ad94e32326e1a35167cf5839a269dd910325473",
    "train_screen_1": "9ee454ef6a13968c7476f6af9c1847a1b5cb7f50d0cc67388372a8f71c8e3c33",
}
PANEL_TEMPLATED_QA_SHA256_V13 = {
    "optimization_0": "654cfb3d7cd6cfa6b7d8ca94c6cb6e0579bad514bfea810087c061362fb3cfda",
    "optimization_1": "c6ad880dcbb25a2ae763fe1609deadac492b9cc21aa2b1a8071a74ba6092d575",
    "optimization_2": "8be919557fe866608689dd8c0c7fe94ca1fb1751d1a7859217c2a0f85ea1d44a",
    "train_screen_0": "586a948d5671d14a132ca2a92015d49049f621276a3f91c1316ac4e7ad5e18e3",
    "train_screen_1": "94fe9e5dd35dfa97186a7a95f89153b83f9e54ed62428581716c8aef859773bb",
}
STRATUM_POPULATION_V13 = {
    "safety_consent": 48,
    "technique": 94,
    "equipment_material": 39,
    "resources_general": 129,
}
PANEL_BUNDLE_CONTENT_SHA256_V13 = (
    "cc176a9b86c6447dcde8a11fd28d68c837d2119715126c57a3f37293fb0d492b"
)
PERTURBATION_SEEDS_V13 = list(anchor_v11.PERTURBATION_SEEDS_V11)
PERTURBATION_BASIS_SHA256_V13 = (
    "29e7ceb1753c39b310a176d827e222b9a5b2c85edf9f2fef5c68b630b8fabc11"
)
PERTURBATION_BASIS_SEED_V13 = 20260714
POPULATION_SIZE_V13 = 32
REQUIRED_ENGINE_COUNT = 4
SIGNS_V13 = ("plus", "minus")
STANDARDIZATION_EPSILON_V13 = 1e-8
_DEFAULT_LAYER_PLAN_BUNDLE = None

canonical_sha256 = anchor_v11c.canonical_sha256
coefficient_sha256 = anchor_v11c.coefficient_sha256
file_sha256 = anchor_v11c.file_sha256
anchor_v4 = anchor_v11c.anchor_v11b.anchor_v11.anchor_v4
anchor_v10 = anchor_v11c.anchor_v11b.anchor_v11.anchor_v10
FROZEN_RUNTIME_EXPECTATIONS_V13 = copy.deepcopy(
    anchor_v10.anchor_v8.anchor_v7.anchor_v6.FROZEN_RUNTIME_EXPECTATIONS_V6
)
validate_layer_plan_installations_v13 = anchor_v10._clone_with_globals(
    anchor_v4.validate_layer_plan_installations_v4,
    {"FROZEN_RUNTIME_EXPECTATIONS_V4": FROZEN_RUNTIME_EXPECTATIONS_V13},
    "validate_layer_plan_installations_v13",
)


def validate_frozen_layer_plan_bundle_v13(bundle):
    return anchor_v11c.validate_frozen_layer_plan_bundle_v11c(bundle)


def load_frozen_layer_plan_v13(*args, **kwargs):
    bundle = anchor_v11c.load_frozen_layer_plan_v11c(*args, **kwargs)
    validate_frozen_layer_plan_bundle_v13(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v13(argv):
    bundle, remaining = anchor_v11c.parse_frozen_layer_plan_cli_v11c(argv)
    validate_frozen_layer_plan_bundle_v13(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v13(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v13(bundle)
    anchor_v11c.set_default_layer_plan_bundle_v11c(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def load_panel_bundle_v13(
    manifest_path=PANEL_MANIFEST_PATH_V13,
    source_path=TRAIN_SOURCE_PATH_V13,
):
    manifest_path = Path(manifest_path).resolve()
    source_path = Path(source_path).resolve()
    if (
        manifest_path != PANEL_MANIFEST_PATH_V13
        or source_path != TRAIN_SOURCE_PATH_V13
        or file_sha256(manifest_path) != PANEL_MANIFEST_FILE_SHA256_V13
        or file_sha256(source_path) != TRAIN_SOURCE_FILE_SHA256_V13
    ):
        raise RuntimeError("v13 frozen train-panel file identity changed")
    rows, source_sha = panel_sampler.load_frozen_train(source_path)
    manifest = json.loads(manifest_path.read_text())
    if (
        source_sha != TRAIN_SOURCE_FILE_SHA256_V13
        or manifest.get("content_sha256_before_self_field")
        != PANEL_MANIFEST_CONTENT_SHA256_V13
        or manifest.get("source", {}).get("arrow_sha256")
        != TRAIN_ARROW_FILE_SHA256_V13
        or panel_sampler.validate_manifest(manifest, rows) is not True
    ):
        raise RuntimeError("v13 frozen train-panel semantic identity changed")
    panels = {}
    for panel in manifest["panels"]:
        selected = [rows[item["row_index"]] for item in panel["items"]]
        panels[panel["name"]] = {
            "name": panel["name"],
            "role": panel["role"],
            "ordered_row_identity_sha256": panel[
                "ordered_row_identity_sha256"
            ],
            "questions": [row["question"] for row in selected],
            "answers": [row["answer"] for row in selected],
            "fact_ids": [row["fact_id"] for row in selected],
            "row_sha256": [item["row_sha256"] for item in panel["items"]],
            "strata": [item["stratum"] for item in panel["items"]],
            "weights": [
                item["horvitz_thompson_unit_weight"]
                for item in panel["items"]
            ],
        }
    result = {
        "schema": "eggroll-es-materialized-train-panels-v13",
        "manifest": {
            "path": str(manifest_path),
            "file_sha256": PANEL_MANIFEST_FILE_SHA256_V13,
            "content_sha256": PANEL_MANIFEST_CONTENT_SHA256_V13,
            "frame_sha256": manifest["sampling_frame"]["frame_sha256"],
        },
        "source": {
            "path": str(source_path),
            "file_sha256": TRAIN_SOURCE_FILE_SHA256_V13,
            "arrow_sha256": TRAIN_ARROW_FILE_SHA256_V13,
            "rows": panel_sampler.SOURCE_ROWS,
        },
        "panels": panels,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    validate_panel_bundle_v13(result)
    return result


def validate_panel_bundle_v13(bundle):
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {
            "schema", "manifest", "source", "panels",
            "content_sha256_before_self_field",
        }
        or bundle.get("schema") != "eggroll-es-materialized-train-panels-v13"
        or bundle.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in bundle.items()
            if key != "content_sha256_before_self_field"
        })
        or bundle.get("manifest", {}).get("file_sha256")
        != PANEL_MANIFEST_FILE_SHA256_V13
        or bundle.get("manifest", {}).get("content_sha256")
        != PANEL_MANIFEST_CONTENT_SHA256_V13
        or bundle.get("source", {}).get("file_sha256")
        != TRAIN_SOURCE_FILE_SHA256_V13
        or bundle.get("source", {}).get("arrow_sha256")
        != TRAIN_ARROW_FILE_SHA256_V13
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V13
        or tuple(bundle.get("panels", {})) != PANEL_NAMES_V13
    ):
        raise RuntimeError("v13 materialized panel bundle changed")
    for name in PANEL_NAMES_V13:
        panel = bundle["panels"][name]
        expected_role = panel_sampler.PANEL_ROLES[name]
        lengths = [
            len(panel.get(key, []))
            for key in (
                "questions", "answers", "fact_ids", "row_sha256", "strata",
                "weights",
            )
        ]
        if (
            set(panel) != {
                "name", "role", "ordered_row_identity_sha256", "questions",
                "answers", "fact_ids", "row_sha256", "strata", "weights",
            }
            or panel.get("name") != name
            or panel.get("role") != expected_role
            or lengths != [panel_sampler.PANEL_SIZE] * len(lengths)
            or canonical_sha256(panel["row_sha256"])
            != panel["ordered_row_identity_sha256"]
            or panel["ordered_row_identity_sha256"]
            != PANEL_ORDERED_ROW_SHA256_V13[name]
            or canonical_sha256({
                "questions": panel["questions"], "answers": panel["answers"],
            }) != PANEL_RAW_QA_SHA256_V13[name]
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) > 0.0
                for value in panel["weights"]
            )
        ):
            raise RuntimeError(f"v13 materialized {name} contract changed")
    return bundle


def _standardize_v13(values):
    values = [float(value) for value in values]
    if len(values) != POPULATION_SIZE_V13 or not all(
        math.isfinite(value) for value in values
    ):
        raise RuntimeError("v13 response vector is incomplete or non-finite")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    coefficients = (
        [0.0] * len(values)
        if std == 0.0
        else [(value - mean) / (std + STANDARDIZATION_EPSILON_V13)
              for value in values]
    )
    return coefficients, {"mean": mean, "std": std, "zero_spread": std == 0.0}


def _cosine_v13(left, right):
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return dot / (left_norm * right_norm)


def _sign_agreement_v13(left, right):
    def sign(value):
        return 1 if value > 0.0 else (-1 if value < 0.0 else 0)

    pairs = [(sign(a), sign(b)) for a, b in zip(left, right)]
    nonzero = [(a, b) for a, b in pairs if a != 0 and b != 0]
    return {
        "all_coordinate_fraction": sum(a == b for a, b in pairs) / len(pairs),
        "nonzero_overlap_count": len(nonzero),
        "nonzero_agreement_fraction": (
            sum(a == b for a, b in nonzero) / len(nonzero)
            if nonzero else None
        ),
    }


def analyze_panel_responses_v13(responses):
    validate_responses_v13(responses)
    panel_analysis = {}
    for name in PANEL_NAMES_V13:
        response = responses[name]
        weighted = response.get("weighted_sign_scores", {})
        unweighted = response.get("unweighted_sign_scores", {})
        if set(weighted) != set(SIGNS_V13) or set(unweighted) != set(SIGNS_V13):
            raise RuntimeError("v13 response sign coverage changed")
        weighted_central = [
            0.5 * (plus - minus)
            for plus, minus in zip(weighted["plus"], weighted["minus"])
        ]
        unweighted_central = [
            0.5 * (plus - minus)
            for plus, minus in zip(unweighted["plus"], unweighted["minus"])
        ]
        coefficients, standardization = _standardize_v13(weighted_central)
        panel_analysis[name] = {
            "role": panel_sampler.PANEL_ROLES[name],
            "weighted_central_response": weighted_central,
            "unweighted_central_response": unweighted_central,
            "coefficients": coefficients,
            "coefficient_sha256": coefficient_sha256(
                PERTURBATION_SEEDS_V13, coefficients,
            ),
            "standardization": standardization,
            "coefficient_l2_norm": math.sqrt(math.fsum(
                value * value for value in coefficients
            )),
        }
    pairwise = {}
    for left_index, left in enumerate(OPTIMIZATION_PANELS_V13):
        for right in OPTIMIZATION_PANELS_V13[left_index + 1:]:
            key = f"{left}__{right}"
            left_values = panel_analysis[left]["coefficients"]
            right_values = panel_analysis[right]["coefficients"]
            pairwise[key] = {
                "cosine": _cosine_v13(left_values, right_values),
                "sign_agreement": _sign_agreement_v13(
                    left_values, right_values,
                ),
            }
    aggregate = [
        float(statistics.median(
            panel_analysis[name]["coefficients"][index]
            for name in OPTIMIZATION_PANELS_V13
        ))
        for index in range(POPULATION_SIZE_V13)
    ]
    aggregate_block = {
        "construction": (
            "coordinatewise_median_of_three_independently_standardized_"
            "Horvitz_Thompson_central_response_vectors"
        ),
        "input_panels": list(OPTIMIZATION_PANELS_V13),
        "excluded_panels": list(TRAIN_SCREENS_V13),
        "coefficients": aggregate,
        "coefficient_sha256": coefficient_sha256(
            PERTURBATION_SEEDS_V13, aggregate,
        ),
        "l2_norm": math.sqrt(math.fsum(value * value for value in aggregate)),
        "nonzero_coordinate_count": sum(value != 0.0 for value in aggregate),
    }
    transfers = {}
    for name in TRAIN_SCREENS_V13:
        screen = panel_analysis[name]["coefficients"]
        transfers[name] = {
            "cosine_to_frozen_optimization_aggregate": _cosine_v13(
                aggregate, screen,
            ),
            "sign_agreement_to_frozen_optimization_aggregate": (
                _sign_agreement_v13(aggregate, screen)
            ),
            "screen_coefficient_sha256": panel_analysis[name][
                "coefficient_sha256"
            ],
        }
    result = {
        "panel_analysis": panel_analysis,
        "optimization_pairwise": pairwise,
        "robust_optimization_aggregate": aggregate_block,
        "train_screen_transfer": transfers,
        "interpretation": "diagnostic_only_no_promotion_or_model_update",
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_responses_v13(responses):
    if not isinstance(responses, dict) or tuple(responses) != PANEL_NAMES_V13:
        raise RuntimeError("v13 response panel coverage changed")

    def finite_vector(value):
        return (
            isinstance(value, list)
            and len(value) == POPULATION_SIZE_V13
            and all(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item))
                for item in value
            )
        )

    for name in PANEL_NAMES_V13:
        response = responses[name]
        if not isinstance(response, dict) or set(response) != {
            "weighted_sign_scores", "unweighted_sign_scores",
            "stratum_sign_scores", "weighted_stratum_contributions",
            "dense_result_sha256",
        }:
            raise RuntimeError(f"v13 {name} response fields changed")
        weighted = response["weighted_sign_scores"]
        unweighted = response["unweighted_sign_scores"]
        hashes = response["dense_result_sha256"]
        if (
            set(weighted) != set(SIGNS_V13)
            or set(unweighted) != set(SIGNS_V13)
            or set(hashes) != set(SIGNS_V13)
            or any(not finite_vector(weighted[sign]) for sign in SIGNS_V13)
            or any(not finite_vector(unweighted[sign]) for sign in SIGNS_V13)
            or any(
                not isinstance(hashes[sign], list)
                or len(hashes[sign]) != POPULATION_SIZE_V13
                or any(
                    not isinstance(value, str) or len(value) != 64
                    for value in hashes[sign]
                )
                for sign in SIGNS_V13
            )
        ):
            raise RuntimeError(f"v13 {name} response vectors changed")
        strata = response["stratum_sign_scores"]
        contributions = response["weighted_stratum_contributions"]
        if set(strata) != set(panel_sampler.STRATA) or set(contributions) != set(
            panel_sampler.STRATA
        ):
            raise RuntimeError(f"v13 {name} stratum response coverage changed")
        for stratum in panel_sampler.STRATA:
            if (
                set(strata[stratum]) != set(SIGNS_V13)
                or set(contributions[stratum]) != set(SIGNS_V13)
                or any(
                    not finite_vector(strata[stratum][sign])
                    or not finite_vector(contributions[stratum][sign])
                    for sign in SIGNS_V13
                )
            ):
                raise RuntimeError(f"v13 {name} stratum vectors changed")
        for sign in SIGNS_V13:
            for index in range(POPULATION_SIZE_V13):
                expected_weighted = math.fsum(
                    contributions[stratum][sign][index]
                    for stratum in panel_sampler.STRATA
                )
                expected_unweighted = math.fsum(
                    panel_sampler.STRATUM_QUOTAS[stratum]
                    * strata[stratum][sign][index]
                    for stratum in panel_sampler.STRATA
                ) / panel_sampler.PANEL_SIZE
                if not math.isclose(
                    weighted[sign][index], expected_weighted,
                    rel_tol=0.0, abs_tol=1e-12,
                ) or not math.isclose(
                    unweighted[sign][index], expected_unweighted,
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    raise RuntimeError(f"v13 {name} response totals do not recompute")
    return responses


def _score_panel_outputs_v13(panel, dense_items, outputs):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = [row["mean_answer_token_logprob"] for row in dense["examples"]]
    weights = [float(value) for value in panel["weights"]]
    denominator = math.fsum(weights)
    if not math.isclose(denominator, 310.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("v13 Horvitz-Thompson weights changed total")
    strata = {}
    contributions = {}
    for stratum in panel_sampler.STRATA:
        indices = [
            index for index, value in enumerate(panel["strata"])
            if value == stratum
        ]
        strata[stratum] = math.fsum(rewards[index] for index in indices) / len(indices)
        contributions[stratum] = math.fsum(
            weights[index] * rewards[index] for index in indices
        ) / denominator
    return {
        "weighted_mean": math.fsum(
            weight * reward for weight, reward in zip(weights, rewards)
        ) / denominator,
        "unweighted_mean": math.fsum(rewards) / len(rewards),
        "stratum_unweighted_means": strata,
        "weighted_stratum_contributions": contributions,
        "scored_answer_tokens": dense["answer_token_count"],
        "dense_result_sha256": canonical_sha256(dense),
    }


class TrainPanelDiagnosticMixinV13:
    def configure_train_panels_v13(self, panel_bundle, *, frozen_layer_plan=None):
        panel_bundle = validate_panel_bundle_v13(copy.deepcopy(panel_bundle))
        layer = frozen_layer_plan or _DEFAULT_LAYER_PLAN_BUNDLE
        validate_frozen_layer_plan_bundle_v13(layer)
        if (
            len(self.engines) != REQUIRED_ENGINE_COUNT
            or int(self.n_vllm_engines) != REQUIRED_ENGINE_COUNT
            or int(self.n_gpu_per_vllm_engine) != 1
            or self.population_size != POPULATION_SIZE_V13
        ):
            raise ValueError("v13 requires four TP=1 engines and population 32")
        reports = self._rpc_all_engines_v4(
            "install_layer_plan_v4",
            (
                Path(layer["path"]).read_bytes(), layer["file_sha256"],
                layer["plan_sha256"], anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
            ),
        )
        install = validate_layer_plan_installations_v13(reports, layer)
        self._v4_layer_plan = layer
        self._v4_layer_plan_install = install
        self._v4_reward_config = dict(anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4)
        self._v4_reward_config_sha256 = (
            anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        )
        states = self._rpc_all_engines_v4("save_self_exact_reference", ())
        if len({canonical_sha256(item) for item in states}) != 1:
            raise RuntimeError("v13 workers captured different exact references")
        self._exact_reference_states = [[state] for state in states]
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4",
            (REQUIRED_ENGINE_COUNT, "exact_reference"),
        )
        summary = self._validate_worker_states_v4(inspected, require_fresh=True)
        self._set_coordinator_reference_v3(summary, fresh=True)
        self._v13_panel_bundle = panel_bundle
        return {
            "layer_plan_install": copy.deepcopy(install),
            "reference_identity": copy.deepcopy(summary["reference_identity"]),
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
        }

    def _prepared_panels_v13(self):
        bundle = validate_panel_bundle_v13(self._v13_panel_bundle)
        prepared = {}
        combined_prompts = []
        cursor = 0
        for name in PANEL_NAMES_V13:
            panel = bundle["panels"][name]
            prompts = [base.specialist_template(q) for q in panel["questions"]]
            templated_identity = canonical_sha256({
                "prompts": prompts, "answers": panel["answers"],
            })
            if templated_identity != PANEL_TEMPLATED_QA_SHA256_V13[name]:
                raise RuntimeError(
                    f"v13 templated {name} identity changed before generation"
                )
            dense_items = anchor_v4.prepare_gold_answer_items_v4(
                self.tokenizer, prompts, panel["answers"],
            )
            prompt_items = [
                {"prompt_token_ids": item["prompt_token_ids"]}
                for item in dense_items
            ]
            prepared[name] = {
                "panel": panel,
                "dense_items": dense_items,
                "slice": (cursor, cursor + len(prompt_items)),
                "templated_prompt_answer_sha256": templated_identity,
            }
            combined_prompts.extend(prompt_items)
            cursor += len(prompt_items)
        if cursor != len(PANEL_NAMES_V13) * panel_sampler.PANEL_SIZE:
            raise RuntimeError("v13 combined panel request count changed")
        return prepared, combined_prompts

    def _base_probe_v13(self, prepared, combined_prompts):
        outputs = anchor_v11.anchor_v1.dispatch_eval_batch(
            self.engines, combined_prompts, self._dense_sampling_params_v4(0),
            self._resolve,
        )
        result = {}
        for name in PANEL_NAMES_V13:
            start, end = prepared[name]["slice"]
            result[name] = _score_panel_outputs_v13(
                prepared[name]["panel"], prepared[name]["dense_items"],
                outputs[start:end],
            )["dense_result_sha256"]
        return {
            "schema": "eggroll-es-five-panel-base-probe-v13",
            "panel_dense_result_sha256": result,
            "combined_request_count": len(combined_prompts),
        }

    def estimate_train_panels_v13(self, seeds):
        seeds = [int(seed) for seed in seeds]
        if seeds != PERTURBATION_SEEDS_V13:
            raise RuntimeError("v13 fixed common perturbation basis changed")
        prepared, combined_prompts = self._prepared_panels_v13()
        pre_probe = self._base_probe_v13(prepared, combined_prompts)
        captures = {
            name: {
                "weighted_sign_scores": {sign: [] for sign in SIGNS_V13},
                "unweighted_sign_scores": {sign: [] for sign in SIGNS_V13},
                "stratum_sign_scores": {
                    stratum: {sign: [] for sign in SIGNS_V13}
                    for stratum in panel_sampler.STRATA
                },
                "weighted_stratum_contributions": {
                    stratum: {sign: [] for sign in SIGNS_V13}
                    for stratum in panel_sampler.STRATA
                },
                "dense_result_sha256": {sign: [] for sign in SIGNS_V13},
            }
            for name in PANEL_NAMES_V13
        }
        for start in range(0, len(seeds), REQUIRED_ENGINE_COUNT):
            wave = anchor_v10.validate_full_engine_wave_v10(
                seeds[start:start + REQUIRED_ENGINE_COUNT],
                REQUIRED_ENGINE_COUNT,
            )
            for sign, negate in (("plus", False), ("minus", True)):
                batches = None
                try:
                    self._resolve([
                        self.engines[index].collective_rpc.remote(
                            "perturb_self_weights",
                            args=(int(seed), self.sigma, negate),
                        )
                        for index, seed in enumerate(wave)
                    ])
                    batches = self._resolve([
                        self.engines[index].generate.remote(
                            list(combined_prompts),
                            self._dense_sampling_params_v4(0),
                            use_tqdm=False,
                        )
                        for index, _seed in enumerate(wave)
                    ])
                finally:
                    self._restore_all_engines_exact()
                if (
                    not isinstance(batches, list)
                    or len(batches) != REQUIRED_ENGINE_COUNT
                    or any(len(batch) != len(combined_prompts) for batch in batches)
                ):
                    raise RuntimeError("v13 resident generation wave is incomplete")
                for engine_index, _seed in enumerate(wave):
                    for name in PANEL_NAMES_V13:
                        panel = prepared[name]
                        slice_start, slice_end = panel["slice"]
                        score = _score_panel_outputs_v13(
                            panel["panel"], panel["dense_items"],
                            batches[engine_index][slice_start:slice_end],
                        )
                        capture = captures[name]
                        capture["weighted_sign_scores"][sign].append(
                            score["weighted_mean"]
                        )
                        capture["unweighted_sign_scores"][sign].append(
                            score["unweighted_mean"]
                        )
                        capture["dense_result_sha256"][sign].append(
                            score["dense_result_sha256"]
                        )
                        for stratum in panel_sampler.STRATA:
                            capture["stratum_sign_scores"][stratum][sign].append(
                                score["stratum_unweighted_means"][stratum]
                            )
                            capture["weighted_stratum_contributions"][stratum][sign].append(
                                score["weighted_stratum_contributions"][stratum]
                            )
        checks = self._resolve([
            engine.collective_rpc.remote("verify_self_exact_reference", args=())
            for engine in self.engines
        ])
        if not anchor_v4.anchor_v3.anchor_v2._all_collective_results(
            checks, lambda value: isinstance(value, dict) and value.get("passed") is True,
        ):
            raise RuntimeError("v13 post-population exact reference check failed")
        boundary = self._population_boundary_audit_v4(0)
        post_probe = self._base_probe_v13(prepared, combined_prompts)
        if pre_probe != post_probe:
            raise RuntimeError("v13 alpha-zero five-panel base probe drifted")
        analysis = analyze_panel_responses_v13(captures)
        panel_contract = {
            name: {
                "role": prepared[name]["panel"]["role"],
                "rows": panel_sampler.PANEL_SIZE,
                "ordered_row_identity_sha256": prepared[name]["panel"][
                    "ordered_row_identity_sha256"
                ],
                "templated_prompt_answer_sha256": prepared[name][
                    "templated_prompt_answer_sha256"
                ],
            }
            for name in PANEL_NAMES_V13
        }
        artifact = {
            "schema": "eggroll-es-five-panel-resident-sign-diagnostic-v13",
            "iteration": 0,
            "alpha": 0.0,
            "model_update_applied": False,
            "applications": [],
            "perturbation_basis": {
                "basis_seed": PERTURBATION_BASIS_SEED_V13,
                "basis_sha256": PERTURBATION_BASIS_SHA256_V13,
                "seeds": seeds,
                "seed_sha256": canonical_sha256(seeds),
                "sign_order": list(SIGNS_V13),
            },
            "panel_bundle_content_sha256": self._v13_panel_bundle[
                "content_sha256_before_self_field"
            ],
            "panel_contract": panel_contract,
            "common_random_numbers": {
                "generation_seed": 43,
                "temperature": 0.0,
                "same_order_every_direction_and_sign": True,
                "combined_panel_order": list(PANEL_NAMES_V13),
            },
            "responses": captures,
            "analysis": analysis,
            "identity_audit": {
                "pre_probe": pre_probe,
                "post_probe": post_probe,
                "exact_reference_checks": checks,
                "passed": True,
            },
            "population_boundary_audit_v4": boundary,
            "hardware_coverage": {
                "engine_count": REQUIRED_ENGINE_COUNT,
                "tp_per_engine": 1,
                "gpu_ids": [0, 1, 2, 3],
                "population_waves": len(seeds) // REQUIRED_ENGINE_COUNT,
                "signed_waves": 2 * len(seeds) // REQUIRED_ENGINE_COUNT,
                "partial_waves": 0,
                "all_engines_generate_every_signed_wave": True,
            },
            "interpretation": "diagnostic_only_no_promotion_decision",
        }
        artifact["content_sha256_before_self_field"] = canonical_sha256(artifact)
        validate_diagnostic_v13(artifact)
        return artifact

    def estimate_step_coefficients(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v13 requires the exact five-panel diagnostic entrypoint")

    def apply_seed_coefficients(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v13 alpha-zero diagnostic forbids model updates")


def validate_diagnostic_v13(artifact):
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {
            "schema", "iteration", "alpha", "model_update_applied",
            "applications", "perturbation_basis",
            "panel_bundle_content_sha256", "panel_contract",
            "common_random_numbers", "responses", "analysis",
            "identity_audit", "population_boundary_audit_v4",
            "hardware_coverage", "interpretation",
            "content_sha256_before_self_field",
        }
        or artifact.get("schema")
        != "eggroll-es-five-panel-resident-sign-diagnostic-v13"
        or artifact.get("iteration") != 0
        or artifact.get("alpha") != 0.0
        or artifact.get("model_update_applied") is not False
        or artifact.get("applications") != []
        or artifact.get("perturbation_basis", {}).get("seeds")
        != PERTURBATION_SEEDS_V13
        or artifact.get("perturbation_basis", {}).get("basis_sha256")
        != PERTURBATION_BASIS_SHA256_V13
        or artifact.get("perturbation_basis", {}).get("basis_seed")
        != PERTURBATION_BASIS_SEED_V13
        or artifact.get("perturbation_basis", {}).get("seed_sha256")
        != canonical_sha256(PERTURBATION_SEEDS_V13)
        or artifact.get("perturbation_basis", {}).get("sign_order")
        != list(SIGNS_V13)
        or artifact.get("panel_bundle_content_sha256")
        != PANEL_BUNDLE_CONTENT_SHA256_V13
        or tuple(artifact.get("panel_contract", {})) != PANEL_NAMES_V13
        or artifact.get("identity_audit", {}).get("passed") is not True
        or artifact.get("identity_audit", {}).get("pre_probe")
        != artifact.get("identity_audit", {}).get("post_probe")
        or artifact.get("population_boundary_audit_v4", {}).get("passed") is not True
        or artifact.get("population_boundary_audit_v4", {}).get("audit_sha256")
        != canonical_sha256({
            key: value
            for key, value in artifact.get("population_boundary_audit_v4", {}).items()
            if key != "audit_sha256"
        })
        or artifact.get("common_random_numbers") != {
            "generation_seed": 43, "temperature": 0.0,
            "same_order_every_direction_and_sign": True,
            "combined_panel_order": list(PANEL_NAMES_V13),
        }
        or artifact.get("hardware_coverage") != {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3], "population_waves": 8,
            "signed_waves": 16, "partial_waves": 0,
            "all_engines_generate_every_signed_wave": True,
        }
        or artifact.get("analysis")
        != analyze_panel_responses_v13(artifact.get("responses"))
        or artifact.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in artifact.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v13 diagnostic artifact changed or does not recompute")
    for name in PANEL_NAMES_V13:
        if artifact["panel_contract"][name] != {
            "role": panel_sampler.PANEL_ROLES[name],
            "rows": panel_sampler.PANEL_SIZE,
            "ordered_row_identity_sha256": PANEL_ORDERED_ROW_SHA256_V13[name],
            "templated_prompt_answer_sha256": PANEL_TEMPLATED_QA_SHA256_V13[name],
        }:
            raise RuntimeError("v13 diagnostic panel contract changed")
    return artifact


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v13(captured)
    parent = anchor_v11c.load_trainer(captured)
    parent.launch_engines = anchor_v10._clone_with_globals(
        parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v13",
    )

    class TrainPanelDiagnosticTrainerV13(TrainPanelDiagnosticMixinV13, parent):
        pass

    return TrainPanelDiagnosticTrainerV13
