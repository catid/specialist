#!/usr/bin/env python3
"""Full-frame, train-only resident-sign diagnostic for V14a."""

from __future__ import annotations

import copy
import math
import os
import statistics
import sys
from pathlib import Path

import eggroll_es_hierarchical_preregistration_v14a as prereg_v14a
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = anchor_v13.WORKER_EXTENSION
PERTURBATION_SEEDS_V14A = list(anchor_v13.PERTURBATION_SEEDS_V13)
PERTURBATION_BASIS_SHA256_V14A = prereg_v14a.PERTURBATION_BASIS_SHA256_V14A
POPULATION_SIZE_V14A = 32
REQUIRED_ENGINE_COUNT_V14A = 4
SIGNS_V14A = ("plus", "minus")
STANDARDIZATION_EPSILON_V14A = 1e-8
FULL_RAW_QA_SHA256_V14A = (
    "6b8f2640841ff212ba51b5d4e90f73afe5d358244a9f81e770a1978ed03df312"
)
FULL_TEMPLATED_QA_SHA256_V14A = (
    "6df92e1cec2147f5a9f7714eb6cac9f813843e9e3c50c4497e9ea6d9a5d04db2"
)
PANEL_BUNDLE_CONTENT_SHA256_V14A = (
    "185da7ac72ced6cdbd2d7a95bef26a0a41849a8a9b78e6fe11c5485b01218077"
)
_DEFAULT_LAYER_PLAN_BUNDLE = None

canonical_sha256 = anchor_v13.canonical_sha256
coefficient_sha256 = anchor_v13.coefficient_sha256
anchor_v4 = anchor_v13.anchor_v4
anchor_v10 = anchor_v13.anchor_v10
anchor_v11 = anchor_v13.anchor_v11


def validate_frozen_layer_plan_bundle_v14a(bundle):
    return anchor_v13.validate_frozen_layer_plan_bundle_v13(bundle)


def load_frozen_layer_plan_v14a(*args, **kwargs):
    bundle = anchor_v13.load_frozen_layer_plan_v13(*args, **kwargs)
    validate_frozen_layer_plan_bundle_v14a(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v14a(argv):
    bundle, remaining = anchor_v13.parse_frozen_layer_plan_cli_v13(argv)
    validate_frozen_layer_plan_bundle_v14a(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v14a(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v14a(bundle)
    anchor_v13.set_default_layer_plan_bundle_v13(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def load_panel_bundle_v14a():
    preregistration = prereg_v14a.build_preregistration_v14a()
    frozen = __import__("json").loads(
        prereg_v14a.PREREGISTRATION_PATH_V14A.read_text()
    )
    if frozen != preregistration:
        raise RuntimeError("v14a machine preregistration changed")
    rows, full_frame, matched = prereg_v14a.materialize_panels_v14a()
    questions = [rows[item["row_index"]]["question"] for item in full_frame["items"]]
    answers = [rows[item["row_index"]]["answer"] for item in full_frame["items"]]
    if canonical_sha256({
        "questions": questions, "answers": answers,
    }) != FULL_RAW_QA_SHA256_V14A:
        raise RuntimeError("v14a full-frame raw Q/A identity changed")
    full_position = {
        item["document_sha256"]: item["position"]
        for item in full_frame["items"]
    }
    matched_contract = {}
    for name in prereg_v14a.PANEL_NAMES_V14A:
        panel = matched[name]
        positions = [
            full_position[item["document_sha256"]] for item in panel["items"]
        ]
        matched_contract[name] = {
            "name": name,
            "role": prereg_v14a.PANEL_ROLES_V14A[name],
            "content_sha256": panel["content_sha256_before_self_field"],
            "ordered_row_identity_sha256": panel[
                "ordered_row_identity_sha256"
            ],
            "positions": positions,
            "strata": [item["stratum"] for item in panel["items"]],
            "weights": [
                item["equal_document_ht_weight"] for item in panel["items"]
            ],
        }
    result = {
        "schema": "eggroll-es-full-frame-runtime-bundle-v14a",
        "preregistration": {
            "path": str(prereg_v14a.PREREGISTRATION_PATH_V14A),
            "file_sha256": prereg_v14a._file_sha256(
                prereg_v14a.PREREGISTRATION_PATH_V14A
            ),
            "content_sha256": frozen["content_sha256_before_self_field"],
        },
        "source": {
            "path": str(prereg_v14a.sampler_v13.DEFAULT_SOURCE.resolve()),
            "file_sha256": prereg_v14a.SOURCE_SHA256_V14A,
            "arrow_sha256": prereg_v14a.SOURCE_ARROW_SHA256_V14A,
            "rows": 794,
            "documents": 310,
            "frame_sha256": prereg_v14a.FRAME_SHA256_V14A,
        },
        "full_frame": {
            "content_sha256": full_frame["content_sha256_before_self_field"],
            "ordered_row_identity_sha256": full_frame[
                "ordered_row_identity_sha256"
            ],
            "questions": questions,
            "answers": answers,
            "document_sha256s": [
                item["document_sha256"] for item in full_frame["items"]
            ],
            "strata": [item["stratum"] for item in full_frame["items"]],
        },
        "matched56": matched_contract,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    validate_panel_bundle_v14a(result)
    return result


def validate_panel_bundle_v14a(bundle):
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {
            "schema", "preregistration", "source", "full_frame", "matched56",
            "content_sha256_before_self_field",
        }
        or bundle.get("schema") != "eggroll-es-full-frame-runtime-bundle-v14a"
        or bundle.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in bundle.items()
            if key != "content_sha256_before_self_field"
        })
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V14A
        or bundle.get("preregistration", {}).get("file_sha256")
        != prereg_v14a._file_sha256(prereg_v14a.PREREGISTRATION_PATH_V14A)
        or bundle.get("preregistration", {}).get("content_sha256")
        != "e610c4bd83449b6b9cb3a0055f8e099ebae32ff6827aa64c6521d74705bda59d"
        or bundle.get("source", {}).get("file_sha256")
        != prereg_v14a.SOURCE_SHA256_V14A
        or bundle.get("source", {}).get("arrow_sha256")
        != prereg_v14a.SOURCE_ARROW_SHA256_V14A
        or bundle.get("source", {}).get("frame_sha256")
        != prereg_v14a.FRAME_SHA256_V14A
        or tuple(bundle.get("matched56", {})) != prereg_v14a.PANEL_NAMES_V14A
    ):
        raise RuntimeError("v14a materialized runtime bundle changed")
    full = bundle["full_frame"]
    if (
        full.get("content_sha256")
        != prereg_v14a.FULL_FRAME_IDENTITY_V14A["content_sha256"]
        or full.get("ordered_row_identity_sha256")
        != prereg_v14a.FULL_FRAME_IDENTITY_V14A[
            "ordered_row_identity_sha256"
        ]
        or any(len(full.get(key, [])) != 310 for key in (
            "questions", "answers", "document_sha256s", "strata",
        ))
        or len(set(full["document_sha256s"])) != 310
        or canonical_sha256({
            "questions": full["questions"], "answers": full["answers"],
        }) != FULL_RAW_QA_SHA256_V14A
    ):
        raise RuntimeError("v14a full-frame runtime contract changed")
    for name in prereg_v14a.PANEL_NAMES_V14A:
        panel = bundle["matched56"][name]
        expected = prereg_v14a.PANEL_IDENTITIES_V14A[name]
        if (
            set(panel) != {
                "name", "role", "content_sha256",
                "ordered_row_identity_sha256", "positions", "strata", "weights",
            }
            or panel["name"] != name
            or panel["role"] != prereg_v14a.PANEL_ROLES_V14A[name]
            or panel["content_sha256"] != expected["content_sha256"]
            or panel["ordered_row_identity_sha256"]
            != expected["ordered_row_identity_sha256"]
            or any(len(panel[key]) != 56 for key in (
                "positions", "strata", "weights",
            ))
            or len(set(panel["positions"])) != 56
            or any(not 0 <= position < 310 for position in panel["positions"])
            or not math.isclose(
                math.fsum(panel["weights"]), 310.0,
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise RuntimeError(f"v14a matched {name} runtime contract changed")
    return bundle


def _standardize(values):
    values = [float(value) for value in values]
    if len(values) != 32 or not all(math.isfinite(value) for value in values):
        raise RuntimeError("v14a response vector is incomplete or non-finite")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    coefficients = (
        [0.0] * len(values) if std == 0.0 else [
            (value - mean) / (std + STANDARDIZATION_EPSILON_V14A)
            for value in values
        ]
    )
    return coefficients, {"mean": mean, "std": std, "zero_spread": std == 0.0}


def _cosine(left, right):
    numerator = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return max(-1.0, min(1.0, numerator / (left_norm * right_norm)))


def _sign_agreement(left, right):
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


def _metric_summary(values):
    if not values or any(value is None or not math.isfinite(value) for value in values):
        raise RuntimeError("v14a stability metric is incomplete")
    return {
        "count": len(values),
        "median": float(statistics.median(values)),
        "worst": min(values),
    }


def _central(sign_scores):
    if set(sign_scores) != set(SIGNS_V14A) or any(
        len(sign_scores[sign]) != 32 for sign in SIGNS_V14A
    ):
        raise RuntimeError("v14a sign response coverage changed")
    return [
        0.5 * (plus - minus) for plus, minus in zip(
            sign_scores["plus"], sign_scores["minus"],
        )
    ]


def analyze_responses_v14a(responses, panel_bundle):
    validate_responses_v14a(responses)
    validate_panel_bundle_v14a(panel_bundle)
    vectors = {}
    standardization = {}
    for name, sign_scores in {
        "full_frame": responses["full_frame_sign_scores"],
        **{
            name: responses["matched56_sign_scores"][name]
            for name in prereg_v14a.PANEL_NAMES_V14A
        },
        **{
            f"complement_{name}": responses["complement_sign_scores"][name]
            for name in prereg_v14a.PANEL_NAMES_V14A[3:]
        },
    }.items():
        vectors[name], standardization[name] = _standardize(
            _central(sign_scores)
        )
    optimization = list(prereg_v14a.PANEL_NAMES_V14A[:3])
    pair_cosines = []
    pair_signs = []
    for left_index, left in enumerate(optimization):
        for right in optimization[left_index + 1:]:
            pair_cosines.append(_cosine(vectors[left], vectors[right]))
            pair_signs.append(_sign_agreement(
                vectors[left], vectors[right],
            )["all_coordinate_fraction"])
    full_cosines = [_cosine(vectors["full_frame"], vectors[name]) for name in optimization]
    full_signs = [_sign_agreement(
        vectors["full_frame"], vectors[name],
    )["all_coordinate_fraction"] for name in optimization]
    screen_cosines = []
    screen_signs = []
    for name in prereg_v14a.PANEL_NAMES_V14A[3:]:
        complement = vectors[f"complement_{name}"]
        screen_cosines.append(_cosine(complement, vectors[name]))
        screen_signs.append(_sign_agreement(
            complement, vectors[name],
        )["all_coordinate_fraction"])
    stability = {
        "matched56_pairwise_cosine": _metric_summary(pair_cosines),
        "matched56_pairwise_sign_agreement": _metric_summary(pair_signs),
        "full_to_matched56_optimization_cosine": _metric_summary(full_cosines),
        "full_to_matched56_optimization_sign_agreement": _metric_summary(full_signs),
        "crossfit_complement_to_screen_cosine": _metric_summary(screen_cosines),
        "crossfit_complement_to_screen_sign_agreement": _metric_summary(screen_signs),
    }
    full_coefficients = vectors["full_frame"]
    candidate = {
        "schema": "eggroll-es-full-frame-matched56-summary-v14a",
        "experiment_name": prereg_v14a.EXPERIMENT_NAME_V14A,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_or_heldout_used": False,
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V14A,
        "panel_identities": {
            "full_frame": prereg_v14a.FULL_FRAME_IDENTITY_V14A[
                "ordered_row_identity_sha256"
            ],
            **{
                name: prereg_v14a.PANEL_IDENTITIES_V14A[name][
                    "ordered_row_identity_sha256"
                ] for name in prereg_v14a.PANEL_NAMES_V14A
            },
        },
        "stability": stability,
        "all_panel_spreads_nonzero": all(
            not value["zero_spread"] for value in standardization.values()
        ),
        "robust_aggregate": {
            "coefficient_sha256": coefficient_sha256(
                PERTURBATION_SEEDS_V14A, full_coefficients,
            ),
            "l2_norm": math.sqrt(math.fsum(
                value * value for value in full_coefficients
            )),
            "nonzero_coordinate_count": sum(
                value != 0.0 for value in full_coefficients
            ),
        },
    }
    candidate["content_sha256_before_self_field"] = canonical_sha256(candidate)
    gate = prereg_v14a.evaluate_candidate_v14a(candidate)
    result = {
        "standardization": standardization,
        "candidate_summary": candidate,
        "promotion_gate": gate,
        "interpretation": "train_only_sampler_gate_no_model_update",
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_responses_v14a(responses):
    expected = {
        "full_frame_sign_scores", "matched56_sign_scores",
        "complement_sign_scores", "dense_result_sha256",
    }
    if not isinstance(responses, dict) or set(responses) != expected:
        raise RuntimeError("v14a response shape changed")

    def signs(value):
        return (
            isinstance(value, dict)
            and set(value) == set(SIGNS_V14A)
            and all(
                isinstance(value[sign], list)
                and len(value[sign]) == 32
                and all(
                    isinstance(item, (int, float))
                    and not isinstance(item, bool)
                    and math.isfinite(float(item))
                    for item in value[sign]
                ) for sign in SIGNS_V14A
            )
        )
    if (
        not signs(responses["full_frame_sign_scores"])
        or tuple(responses["matched56_sign_scores"])
        != prereg_v14a.PANEL_NAMES_V14A
        or any(not signs(value) for value in responses["matched56_sign_scores"].values())
        or tuple(responses["complement_sign_scores"])
        != prereg_v14a.PANEL_NAMES_V14A[3:]
        or any(not signs(value) for value in responses["complement_sign_scores"].values())
    ):
        raise RuntimeError("v14a response vectors changed")
    hashes = responses["dense_result_sha256"]
    if (
        set(hashes) != set(SIGNS_V14A)
        or any(
            not isinstance(hashes[sign], list)
            or len(hashes[sign]) != 32
            or any(not isinstance(value, str) or len(value) != 64 for value in hashes[sign])
            for sign in SIGNS_V14A
        )
    ):
        raise RuntimeError("v14a dense result identity coverage changed")
    return responses


def _weighted_mean(rewards, positions, weights):
    denominator = math.fsum(weights)
    if not math.isclose(denominator, 310.0, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError("v14a equal-document weights changed total")
    return math.fsum(
        weights[index] * rewards[position]
        for index, position in enumerate(positions)
    ) / denominator


def _score_full_outputs(dense_items, outputs, panel_bundle):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = [row["mean_answer_token_logprob"] for row in dense["examples"]]
    if len(rewards) != 310:
        raise RuntimeError("v14a full-frame output count changed")
    matched = {}
    for name in prereg_v14a.PANEL_NAMES_V14A:
        panel = panel_bundle["matched56"][name]
        matched[name] = _weighted_mean(
            rewards, panel["positions"], panel["weights"],
        )
    complements = {}
    all_positions = set(range(310))
    for name in prereg_v14a.PANEL_NAMES_V14A[3:]:
        screen_positions = set(panel_bundle["matched56"][name]["positions"])
        positions = sorted(all_positions - screen_positions)
        if len(positions) != 254:
            raise RuntimeError("v14a crossfit complement size changed")
        complements[name] = math.fsum(rewards[position] for position in positions) / 254.0
    return {
        "full_frame": math.fsum(rewards) / 310.0,
        "matched56": matched,
        "complements": complements,
        "dense_result_sha256": canonical_sha256(dense),
    }


class FullFrameDiagnosticMixinV14A:
    def configure_full_frame_v14a(self, panel_bundle, *, frozen_layer_plan=None):
        panel_bundle = validate_panel_bundle_v14a(copy.deepcopy(panel_bundle))
        layer = frozen_layer_plan or _DEFAULT_LAYER_PLAN_BUNDLE
        validate_frozen_layer_plan_bundle_v14a(layer)
        if (
            len(self.engines) != REQUIRED_ENGINE_COUNT_V14A
            or int(self.n_vllm_engines) != REQUIRED_ENGINE_COUNT_V14A
            or int(self.n_gpu_per_vllm_engine) != 1
            or self.population_size != POPULATION_SIZE_V14A
        ):
            raise ValueError("v14a requires four TP=1 engines and population 32")
        reports = self._rpc_all_engines_v4(
            "install_layer_plan_v4",
            (
                Path(layer["path"]).read_bytes(), layer["file_sha256"],
                layer["plan_sha256"], anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
            ),
        )
        install = anchor_v13.validate_layer_plan_installations_v13(reports, layer)
        self._v4_layer_plan = layer
        self._v4_layer_plan_install = install
        self._v4_reward_config = dict(anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4)
        self._v4_reward_config_sha256 = anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        states = self._rpc_all_engines_v4("save_self_exact_reference", ())
        if len({canonical_sha256(item) for item in states}) != 1:
            raise RuntimeError("v14a workers captured different exact references")
        self._exact_reference_states = [[state] for state in states]
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4",
            (REQUIRED_ENGINE_COUNT_V14A, "exact_reference"),
        )
        summary = self._validate_worker_states_v4(inspected, require_fresh=True)
        self._set_coordinator_reference_v3(summary, fresh=True)
        self._v14a_panel_bundle = panel_bundle
        return {
            "layer_plan_install": copy.deepcopy(install),
            "reference_identity": copy.deepcopy(summary["reference_identity"]),
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
        }

    def _prepared_full_frame_v14a(self):
        bundle = validate_panel_bundle_v14a(self._v14a_panel_bundle)
        full = bundle["full_frame"]
        prompts = [base.specialist_template(question) for question in full["questions"]]
        templated = canonical_sha256({"prompts": prompts, "answers": full["answers"]})
        if templated != FULL_TEMPLATED_QA_SHA256_V14A:
            raise RuntimeError("v14a templated full-frame identity changed")
        dense_items = anchor_v4.prepare_gold_answer_items_v4(
            self.tokenizer, prompts, full["answers"],
        )
        prompt_items = [
            {"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items
        ]
        if len(prompt_items) != 310:
            raise RuntimeError("v14a full-frame prompt count changed")
        return dense_items, prompt_items, templated

    def _base_probe_v14a(self, dense_items, prompt_items):
        outputs = anchor_v11.anchor_v1.dispatch_eval_batch(
            self.engines, prompt_items, self._dense_sampling_params_v4(0),
            self._resolve,
        )
        score = _score_full_outputs(dense_items, outputs, self._v14a_panel_bundle)
        return {
            "schema": "eggroll-es-full-frame-base-probe-v14a",
            "dense_result_sha256": score["dense_result_sha256"],
            "request_count": len(prompt_items),
        }

    def estimate_full_frame_v14a(self, seeds):
        seeds = [int(seed) for seed in seeds]
        if seeds != PERTURBATION_SEEDS_V14A:
            raise RuntimeError("v14a fixed common perturbation basis changed")
        dense_items, prompt_items, templated = self._prepared_full_frame_v14a()
        pre_probe = self._base_probe_v14a(dense_items, prompt_items)
        responses = {
            "full_frame_sign_scores": {sign: [] for sign in SIGNS_V14A},
            "matched56_sign_scores": {
                name: {sign: [] for sign in SIGNS_V14A}
                for name in prereg_v14a.PANEL_NAMES_V14A
            },
            "complement_sign_scores": {
                name: {sign: [] for sign in SIGNS_V14A}
                for name in prereg_v14a.PANEL_NAMES_V14A[3:]
            },
            "dense_result_sha256": {sign: [] for sign in SIGNS_V14A},
        }
        for start in range(0, len(seeds), REQUIRED_ENGINE_COUNT_V14A):
            wave = anchor_v10.validate_full_engine_wave_v10(
                seeds[start:start + REQUIRED_ENGINE_COUNT_V14A],
                REQUIRED_ENGINE_COUNT_V14A,
            )
            for sign, negate in (("plus", False), ("minus", True)):
                batches = None
                try:
                    self._resolve([
                        self.engines[index].collective_rpc.remote(
                            "perturb_self_weights", args=(int(seed), self.sigma, negate),
                        ) for index, seed in enumerate(wave)
                    ])
                    batches = self._resolve([
                        self.engines[index].generate.remote(
                            list(prompt_items), self._dense_sampling_params_v4(0),
                            use_tqdm=False,
                        ) for index, _seed in enumerate(wave)
                    ])
                finally:
                    self._restore_all_engines_exact()
                if (
                    not isinstance(batches, list)
                    or len(batches) != REQUIRED_ENGINE_COUNT_V14A
                    or any(len(batch) != 310 for batch in batches)
                ):
                    raise RuntimeError("v14a resident generation wave is incomplete")
                for engine_index, _seed in enumerate(wave):
                    score = _score_full_outputs(
                        dense_items, batches[engine_index], self._v14a_panel_bundle,
                    )
                    responses["full_frame_sign_scores"][sign].append(
                        score["full_frame"]
                    )
                    responses["dense_result_sha256"][sign].append(
                        score["dense_result_sha256"]
                    )
                    for name in prereg_v14a.PANEL_NAMES_V14A:
                        responses["matched56_sign_scores"][name][sign].append(
                            score["matched56"][name]
                        )
                    for name in prereg_v14a.PANEL_NAMES_V14A[3:]:
                        responses["complement_sign_scores"][name][sign].append(
                            score["complements"][name]
                        )
        validate_responses_v14a(responses)
        checks = self._resolve([
            engine.collective_rpc.remote("verify_self_exact_reference", args=())
            for engine in self.engines
        ])
        if not anchor_v4.anchor_v3.anchor_v2._all_collective_results(
            checks, lambda value: isinstance(value, dict) and value.get("passed") is True,
        ):
            raise RuntimeError("v14a post-population exact reference check failed")
        boundary = self._population_boundary_audit_v4(0)
        post_probe = self._base_probe_v14a(dense_items, prompt_items)
        if pre_probe != post_probe:
            raise RuntimeError("v14a alpha-zero full-frame base probe drifted")
        analysis = analyze_responses_v14a(responses, self._v14a_panel_bundle)
        artifact = {
            "schema": "eggroll-es-full-frame-resident-sign-diagnostic-v14a",
            "iteration": 0,
            "alpha": 0.0,
            "model_update_applied": False,
            "applications": [],
            "perturbation_basis": {
                "seeds": seeds,
                "basis_sha256": PERTURBATION_BASIS_SHA256_V14A,
                "sign_order": list(SIGNS_V14A),
            },
            "panel_bundle_content_sha256": self._v14a_panel_bundle[
                "content_sha256_before_self_field"
            ],
            "templated_prompt_answer_sha256": templated,
            "generation_contract": {
                "prompts_per_engine_per_sign": 310,
                "generation_calls_per_engine_per_sign": 1,
                "matched_and_crossfit_responses_derived_without_generation": True,
            },
            "responses": responses,
            "analysis": analysis,
            "identity_audit": {
                "pre_probe": pre_probe, "post_probe": post_probe,
                "exact_reference_checks": checks, "passed": True,
            },
            "population_boundary_audit_v4": boundary,
            "hardware_coverage": {
                "engine_count": 4, "tp_per_engine": 1,
                "gpu_ids": [0, 1, 2, 3], "population_waves": 8,
                "signed_waves": 16, "partial_waves": 0,
                "all_engines_generate_every_signed_wave": True,
            },
            "interpretation": "train_only_sampler_gate_no_model_update",
        }
        artifact["content_sha256_before_self_field"] = canonical_sha256(artifact)
        validate_diagnostic_v14a(artifact, self._v14a_panel_bundle)
        return artifact

    def estimate_step_coefficients(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v14a requires the exact full-frame diagnostic entrypoint")

    def apply_seed_coefficients(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v14a alpha-zero diagnostic forbids model updates")


def validate_diagnostic_v14a(artifact, panel_bundle):
    validate_panel_bundle_v14a(panel_bundle)
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {
            "schema", "iteration", "alpha", "model_update_applied",
            "applications", "perturbation_basis", "panel_bundle_content_sha256",
            "templated_prompt_answer_sha256", "generation_contract", "responses",
            "analysis", "identity_audit", "population_boundary_audit_v4",
            "hardware_coverage", "interpretation",
            "content_sha256_before_self_field",
        }
        or artifact.get("schema")
        != "eggroll-es-full-frame-resident-sign-diagnostic-v14a"
        or artifact.get("iteration") != 0
        or artifact.get("alpha") != 0.0
        or artifact.get("model_update_applied") is not False
        or artifact.get("applications") != []
        or artifact.get("perturbation_basis") != {
            "seeds": PERTURBATION_SEEDS_V14A,
            "basis_sha256": PERTURBATION_BASIS_SHA256_V14A,
            "sign_order": list(SIGNS_V14A),
        }
        or artifact.get("panel_bundle_content_sha256")
        != PANEL_BUNDLE_CONTENT_SHA256_V14A
        or artifact.get("templated_prompt_answer_sha256")
        != FULL_TEMPLATED_QA_SHA256_V14A
        or artifact.get("generation_contract") != {
            "prompts_per_engine_per_sign": 310,
            "generation_calls_per_engine_per_sign": 1,
            "matched_and_crossfit_responses_derived_without_generation": True,
        }
        or artifact.get("analysis")
        != analyze_responses_v14a(artifact.get("responses"), panel_bundle)
        or artifact.get("identity_audit", {}).get("passed") is not True
        or artifact.get("identity_audit", {}).get("pre_probe")
        != artifact.get("identity_audit", {}).get("post_probe")
        or artifact.get("population_boundary_audit_v4", {}).get("passed") is not True
        or artifact.get("population_boundary_audit_v4", {}).get("audit_sha256")
        != canonical_sha256({
            key: value for key, value in artifact.get(
                "population_boundary_audit_v4", {}
            ).items() if key != "audit_sha256"
        })
        or artifact.get("hardware_coverage") != {
            "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
            "population_waves": 8, "signed_waves": 16,
            "partial_waves": 0, "all_engines_generate_every_signed_wave": True,
        }
        or artifact.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in artifact.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v14a diagnostic artifact changed or does not recompute")
    return artifact


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v14a(captured)
    parent = anchor_v13.load_trainer(captured)

    class FullFrameDiagnosticTrainerV14A(FullFrameDiagnosticMixinV14A, parent):
        pass

    return FullFrameDiagnosticTrainerV14A
