#!/usr/bin/env python3
"""K=2 document-mean, train-only resident-sign diagnostic for V14b."""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

import eggroll_es_hierarchical_preregistration_v14b as prereg_v14b
import train_eggroll_es_specialist as base
import train_eggroll_es_specialist_anchor_v14a as anchor_v14a


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = anchor_v14a.WORKER_EXTENSION
PERTURBATION_SEEDS_V14B = list(anchor_v14a.PERTURBATION_SEEDS_V14A)
PERTURBATION_BASIS_SHA256_V14B = prereg_v14b.PERTURBATION_BASIS_SHA256_V14B
POPULATION_SIZE_V14B = 32
REQUIRED_ENGINE_COUNT_V14B = 4
SIGNS_V14B = ("plus", "minus")
FULL_RAW_QA_SHA256_V14B = (
    "aca7a98a044509594d62550ced1f1a6514c05118583c660b652a611947870bf9"
)
FULL_TEMPLATED_QA_SHA256_V14B = (
    "4d903646f3b27f91f14eb14e0179be7588a306c4806aa54c4bf6a0a05f314553"
)
PANEL_BUNDLE_CONTENT_SHA256_V14B = (
    "0c224167e1bf200dc39e0f451b4af505408bf21a1b3c4f42d8288db27844b567"
)
_DEFAULT_LAYER_PLAN_BUNDLE = None

canonical_sha256 = anchor_v14a.canonical_sha256
coefficient_sha256 = anchor_v14a.coefficient_sha256
anchor_v4 = anchor_v14a.anchor_v4
anchor_v10 = anchor_v14a.anchor_v10


def validate_frozen_layer_plan_bundle_v14b(bundle):
    return anchor_v14a.validate_frozen_layer_plan_bundle_v14a(bundle)


def load_frozen_layer_plan_v14b(*args, **kwargs):
    bundle = anchor_v14a.load_frozen_layer_plan_v14a(*args, **kwargs)
    validate_frozen_layer_plan_bundle_v14b(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v14b(argv):
    bundle, remaining = anchor_v14a.parse_frozen_layer_plan_cli_v14a(argv)
    validate_frozen_layer_plan_bundle_v14b(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v14b(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v14b(bundle)
    anchor_v14a.set_default_layer_plan_bundle_v14a(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def load_panel_bundle_v14b(*, validate=True):
    rows, full, panels, complements = prereg_v14b.materialize_sampler_v14b()
    questions = [None] * 481
    answers = [None] * 481
    document_groups = []
    for document in full["items"]:
        positions = []
        for selected in document["selected_rows"]:
            position = selected["prompt_position"]
            questions[position] = rows[selected["row_index"]]["question"]
            answers[position] = rows[selected["row_index"]]["answer"]
            positions.append(position)
        document_groups.append({
            "document_sha256": document["document_sha256"],
            "prompt_positions": positions,
        })
    by_document_position = {
        item["document_sha256"]: item["document_position"]
        for item in full["items"]
    }
    matched = {}
    for name in prereg_v14b.PANEL_NAMES_V14B:
        panel = panels[name]
        matched[name] = {
            "role": prereg_v14b.PANEL_ROLES_V14B[name],
            "content_sha256": panel["content_sha256_before_self_field"],
            "ordered_document_identity_sha256": panel[
                "ordered_document_identity_sha256"
            ],
            "ordered_prompt_identity_sha256": panel[
                "ordered_prompt_identity_sha256"
            ],
            "document_positions": [
                by_document_position[item["document_sha256"]]
                for item in panel["items"]
            ],
            "weights": [
                item["equal_document_ht_weight"] for item in panel["items"]
            ],
        }
    complement_contract = {
        name: {
            "content_sha256": complement["content_sha256_before_self_field"],
            "ordered_document_identity_sha256": complement[
                "ordered_document_identity_sha256"
            ],
            "ordered_prompt_identity_sha256": complement[
                "ordered_prompt_identity_sha256"
            ],
            "document_positions": [
                by_document_position[item["document_sha256"]]
                for item in complement["items"]
            ],
        }
        for name, complement in complements.items()
    }
    result = {
        "schema": "eggroll-es-k2-runtime-bundle-v14b",
        "preregistration": {
            "path": str(prereg_v14b.PREREGISTRATION_PATH_V14B),
            "file_sha256": prereg_v14b._file_sha256(
                prereg_v14b.PREREGISTRATION_PATH_V14B
            ),
            "content_sha256": prereg_v14b.build_preregistration_v14b()[
                "content_sha256_before_self_field"
            ],
        },
        "source": {
            "path": str(prereg_v14b.sampler_v13.DEFAULT_SOURCE.resolve()),
            "file_sha256": prereg_v14b.SOURCE_SHA256_V14B,
            "arrow_sha256": prereg_v14b.SOURCE_ARROW_SHA256_V14B,
            "frame_sha256": prereg_v14b.FRAME_SHA256_V14B,
            "rows": 794, "documents": 310, "prompts": 481,
        },
        "full_frame": {
            "content_sha256": full["content_sha256_before_self_field"],
            "ordered_document_identity_sha256": full[
                "ordered_document_identity_sha256"
            ],
            "ordered_prompt_identity_sha256": full[
                "ordered_prompt_identity_sha256"
            ],
            "questions": questions, "answers": answers,
            "document_groups": document_groups,
        },
        "matched56": matched,
        "complements": complement_contract,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    if validate:
        validate_panel_bundle_v14b(result)
    return result


def validate_panel_bundle_v14b(bundle):
    if (
        not isinstance(bundle, dict)
        or set(bundle) != {
            "schema", "preregistration", "source", "full_frame",
            "matched56", "complements", "content_sha256_before_self_field",
        }
        or bundle.get("schema") != "eggroll-es-k2-runtime-bundle-v14b"
        or bundle.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in bundle.items()
            if key != "content_sha256_before_self_field"
        })
        or bundle.get("content_sha256_before_self_field")
        != PANEL_BUNDLE_CONTENT_SHA256_V14B
        or bundle.get("preregistration", {}).get("file_sha256")
        != "dcab1a49befebc8b67bbb9a80b866e876438a1e960e37953dff8c742b4e2c8ec"
        or bundle.get("preregistration", {}).get("content_sha256")
        != "0963d1a8e18a97af949b94762292536c606279610c7c239e445d85e5be2c3216"
        or bundle.get("source") != {
            "path": str(prereg_v14b.sampler_v13.DEFAULT_SOURCE.resolve()),
            "file_sha256": prereg_v14b.SOURCE_SHA256_V14B,
            "arrow_sha256": prereg_v14b.SOURCE_ARROW_SHA256_V14B,
            "frame_sha256": prereg_v14b.FRAME_SHA256_V14B,
            "rows": 794, "documents": 310, "prompts": 481,
        }
        or tuple(bundle.get("matched56", {})) != prereg_v14b.PANEL_NAMES_V14B
        or tuple(bundle.get("complements", {}))
        != prereg_v14b.PANEL_NAMES_V14B[3:]
    ):
        raise RuntimeError("v14b materialized runtime bundle changed")
    full = bundle["full_frame"]
    groups = full.get("document_groups", [])
    if (
        full.get("content_sha256")
        != prereg_v14b.FULL_FRAME_IDENTITY_V14B["content_sha256"]
        or full.get("ordered_document_identity_sha256")
        != prereg_v14b.FULL_FRAME_IDENTITY_V14B[
            "ordered_document_identity_sha256"
        ]
        or full.get("ordered_prompt_identity_sha256")
        != prereg_v14b.FULL_FRAME_IDENTITY_V14B[
            "ordered_prompt_identity_sha256"
        ]
        or len(full.get("questions", [])) != 481
        or len(full.get("answers", [])) != 481
        or len(groups) != 310
        or len({item["document_sha256"] for item in groups}) != 310
        or sorted(
            position for item in groups for position in item["prompt_positions"]
        ) != list(range(481))
        or canonical_sha256({
            "questions": full["questions"], "answers": full["answers"],
        }) != FULL_RAW_QA_SHA256_V14B
    ):
        raise RuntimeError("v14b full-frame runtime contract changed")
    for name in prereg_v14b.PANEL_NAMES_V14B:
        panel = bundle["matched56"][name]
        expected = prereg_v14b.PANEL_IDENTITIES_V14B[name]
        if (
            panel["role"] != prereg_v14b.PANEL_ROLES_V14B[name]
            or panel["content_sha256"] != expected["content_sha256"]
            or panel["ordered_document_identity_sha256"]
            != expected["ordered_document_identity_sha256"]
            or panel["ordered_prompt_identity_sha256"]
            != expected["ordered_prompt_identity_sha256"]
            or len(panel["document_positions"]) != 56
            or len(set(panel["document_positions"])) != 56
            or len(panel["weights"]) != 56
            or not math.isclose(
                math.fsum(panel["weights"]), 310.0,
                rel_tol=0.0, abs_tol=1e-12,
            )
        ):
            raise RuntimeError(f"v14b matched56 {name} contract changed")
    for name in prereg_v14b.PANEL_NAMES_V14B[3:]:
        complement = bundle["complements"][name]
        expected = prereg_v14b.COMPLEMENT_IDENTITIES_V14B[name]
        if (
            complement["content_sha256"] != expected["content_sha256"]
            or complement["ordered_document_identity_sha256"]
            != expected["ordered_document_identity_sha256"]
            or complement["ordered_prompt_identity_sha256"]
            != expected["ordered_prompt_identity_sha256"]
            or len(complement["document_positions"]) != 254
            or len(set(complement["document_positions"])) != 254
            or set(complement["document_positions"]).intersection(
                bundle["matched56"][name]["document_positions"]
            )
        ):
            raise RuntimeError(f"v14b complement {name} contract changed")
    return bundle


def _weighted_mean(values, positions, weights):
    if not math.isclose(math.fsum(weights), 310.0, abs_tol=1e-12):
        raise RuntimeError("v14b matched weights changed")
    return math.fsum(
        weight * values[position]
        for position, weight in zip(positions, weights)
    ) / 310.0


def _score_full_outputs_v14b(dense_items, outputs, panel_bundle):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    rewards = [row["mean_answer_token_logprob"] for row in dense["examples"]]
    if len(rewards) != 481:
        raise RuntimeError("v14b prompt reward count changed")
    document_values = [
        math.fsum(rewards[position] for position in group["prompt_positions"])
        / len(group["prompt_positions"])
        for group in panel_bundle["full_frame"]["document_groups"]
    ]
    if (
        len(document_values) != 310
        or not all(math.isfinite(value) for value in document_values)
    ):
        raise RuntimeError("v14b document mean count changed")
    matched = {
        name: _weighted_mean(
            document_values, panel["document_positions"], panel["weights"],
        ) for name, panel in panel_bundle["matched56"].items()
    }
    complements = {
        name: math.fsum(
            document_values[position]
            for position in complement["document_positions"]
        ) / 254.0
        for name, complement in panel_bundle["complements"].items()
    }
    return {
        "full_frame": math.fsum(document_values) / 310.0,
        "matched56": matched,
        "complements": complements,
        "dense_result_sha256": canonical_sha256(dense),
    }


def validate_responses_v14b(responses):
    expected = {
        "full_frame_sign_scores", "matched56_sign_scores",
        "complement_sign_scores", "dense_result_sha256",
    }
    if not isinstance(responses, dict) or set(responses) != expected:
        raise RuntimeError("v14b response shape changed")

    def signs(value):
        return (
            isinstance(value, dict)
            and set(value) == set(SIGNS_V14B)
            and all(
                isinstance(value[sign], list) and len(value[sign]) == 32
                and all(
                    isinstance(item, (int, float)) and not isinstance(item, bool)
                    and math.isfinite(float(item)) for item in value[sign]
                ) for sign in SIGNS_V14B
            )
        )
    if (
        not signs(responses["full_frame_sign_scores"])
        or tuple(responses["matched56_sign_scores"])
        != prereg_v14b.PANEL_NAMES_V14B
        or any(not signs(value) for value in responses["matched56_sign_scores"].values())
        or tuple(responses["complement_sign_scores"])
        != prereg_v14b.PANEL_NAMES_V14B[3:]
        or any(not signs(value) for value in responses["complement_sign_scores"].values())
    ):
        raise RuntimeError("v14b response vectors changed")
    hashes = responses["dense_result_sha256"]
    if (
        set(hashes) != set(SIGNS_V14B)
        or any(
            len(hashes[sign]) != 32
            or any(not isinstance(value, str) or len(value) != 64 for value in hashes[sign])
            for sign in SIGNS_V14B
        )
    ):
        raise RuntimeError("v14b dense identity coverage changed")
    return responses


def build_integrity_audits_v14b(
    *, alpha, applications, model_update_applied, identity_audit,
    population_boundary_audit, hardware_coverage,
):
    exact_checks = identity_audit.get("exact_reference_checks")
    boundary = population_boundary_audit
    expected_hardware = {
        "engine_count": 4, "tp_per_engine": 1, "gpu_ids": [0, 1, 2, 3],
        "population_waves": 8, "signed_waves": 16,
        "partial_waves": 0, "all_engines_generate_every_signed_wave": True,
    }
    result = {
        "alpha_zero_no_applications": alpha == 0.0 and applications == [],
        "model_update_applied_false": model_update_applied is False,
        "exact_reference_checks_passed": (
            anchor_v4.anchor_v3.anchor_v2._all_collective_results(
                exact_checks,
                lambda value: (
                    isinstance(value, dict) and value.get("passed") is True
                ),
            )
        ),
        "pre_post_base_probe_equal": (
            identity_audit.get("passed") is True
            and identity_audit.get("pre_probe")
            == identity_audit.get("post_probe")
        ),
        "population_boundary_passed": boundary.get("passed") is True,
        "population_boundary_hash_valid": boundary.get("audit_sha256")
        == canonical_sha256({
            key: value for key, value in boundary.items()
            if key != "audit_sha256"
        }),
        "hardware_contract_passed": hardware_coverage == expected_hardware,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_integrity_audits_v14b(integrity):
    expected = {
        "alpha_zero_no_applications", "model_update_applied_false",
        "exact_reference_checks_passed", "pre_post_base_probe_equal",
        "population_boundary_passed", "population_boundary_hash_valid",
        "hardware_contract_passed", "content_sha256_before_self_field",
    }
    if (
        not isinstance(integrity, dict) or set(integrity) != expected
        or any(integrity[key] is not True for key in expected if key != "content_sha256_before_self_field")
        or integrity["content_sha256_before_self_field"]
        != canonical_sha256({
            key: value for key, value in integrity.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v14b integrity audits are incomplete or failed")
    return integrity


def analyze_responses_v14b(responses, integrity_audits):
    validate_responses_v14b(responses)
    validate_integrity_audits_v14b(integrity_audits)
    inputs = {
        "full_frame": responses["full_frame_sign_scores"],
        **{
            name: responses["matched56_sign_scores"][name]
            for name in prereg_v14b.PANEL_NAMES_V14B
        },
        **{
            f"complement_{name}": responses["complement_sign_scores"][name]
            for name in prereg_v14b.PANEL_NAMES_V14B[3:]
        },
    }
    vectors = {}
    standardization = {}
    for name, scores in inputs.items():
        vectors[name], standardization[name] = anchor_v14a._standardize(
            anchor_v14a._central(scores)
        )
    optimization = list(prereg_v14b.PANEL_NAMES_V14B[:3])
    pair_cosines = []
    pair_signs = []
    for index, left in enumerate(optimization):
        for right in optimization[index + 1:]:
            pair_cosines.append(anchor_v14a._cosine(vectors[left], vectors[right]))
            pair_signs.append(anchor_v14a._sign_agreement(
                vectors[left], vectors[right],
            )["all_coordinate_fraction"])
    full_cosines = [
        anchor_v14a._cosine(vectors["full_frame"], vectors[name])
        for name in optimization
    ]
    full_signs = [
        anchor_v14a._sign_agreement(
            vectors["full_frame"], vectors[name],
        )["all_coordinate_fraction"] for name in optimization
    ]
    screen_cosines = []
    screen_signs = []
    for name in prereg_v14b.PANEL_NAMES_V14B[3:]:
        complement = vectors[f"complement_{name}"]
        screen_cosines.append(anchor_v14a._cosine(complement, vectors[name]))
        screen_signs.append(anchor_v14a._sign_agreement(
            complement, vectors[name],
        )["all_coordinate_fraction"])
    stability = {
        "matched56_pairwise_cosine": anchor_v14a._metric_summary(pair_cosines),
        "matched56_pairwise_sign_agreement": anchor_v14a._metric_summary(pair_signs),
        "full_to_matched56_optimization_cosine": anchor_v14a._metric_summary(full_cosines),
        "full_to_matched56_optimization_sign_agreement": anchor_v14a._metric_summary(full_signs),
        "crossfit_complement_to_screen_cosine": anchor_v14a._metric_summary(screen_cosines),
        "crossfit_complement_to_screen_sign_agreement": anchor_v14a._metric_summary(screen_signs),
    }
    full = vectors["full_frame"]
    candidate = {
        "schema": "eggroll-es-paired-distinct-row-summary-v14b",
        "experiment_name": prereg_v14b.EXPERIMENT_NAME_V14B,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_or_heldout_used": False,
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V14B,
        "panel_identities": prereg_v14b.candidate_panel_identities_v14b(),
        "stability": stability,
        "all_panel_spreads_nonzero": all(
            not value["zero_spread"] for value in standardization.values()
        ),
        "robust_aggregate": {
            "coefficient_sha256": coefficient_sha256(
                PERTURBATION_SEEDS_V14B, full,
            ),
            "l2_norm": math.sqrt(math.fsum(value * value for value in full)),
            "nonzero_coordinate_count": sum(value != 0.0 for value in full),
        },
        "all_integrity_audits_passed": all(
            integrity_audits[key] is True
            for key in integrity_audits if key != "content_sha256_before_self_field"
        ),
    }
    candidate["content_sha256_before_self_field"] = canonical_sha256(candidate)
    gate = prereg_v14b.evaluate_candidate_v14b(candidate)
    result = {
        "standardization": standardization,
        "candidate_summary": candidate,
        "promotion_gate": gate,
        "interpretation": "train_only_k2_estimator_gate_no_model_update",
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


class PairedDistinctRowDiagnosticMixinV14B:
    def configure_full_frame_v14b(self, panel_bundle, *, frozen_layer_plan=None):
        panel_bundle = validate_panel_bundle_v14b(copy.deepcopy(panel_bundle))
        layer = frozen_layer_plan or _DEFAULT_LAYER_PLAN_BUNDLE
        validate_frozen_layer_plan_bundle_v14b(layer)
        if (
            len(self.engines) != 4 or int(self.n_vllm_engines) != 4
            or int(self.n_gpu_per_vllm_engine) != 1 or self.population_size != 32
        ):
            raise ValueError("v14b requires four TP=1 engines and population 32")
        reports = self._rpc_all_engines_v4(
            "install_layer_plan_v4",
            (
                Path(layer["path"]).read_bytes(), layer["file_sha256"],
                layer["plan_sha256"], anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4,
                anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4,
            ),
        )
        install = anchor_v14a.anchor_v13.validate_layer_plan_installations_v13(
            reports, layer,
        )
        self._v4_layer_plan = layer
        self._v4_layer_plan_install = install
        self._v4_reward_config = dict(anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4)
        self._v4_reward_config_sha256 = anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        states = self._rpc_all_engines_v4("save_self_exact_reference", ())
        if len({canonical_sha256(item) for item in states}) != 1:
            raise RuntimeError("v14b workers captured different exact references")
        self._exact_reference_states = [[state] for state in states]
        inspected = self._rpc_all_engines_v4(
            "inspect_cached_distributed_update_state_v4", (4, "exact_reference"),
        )
        summary = self._validate_worker_states_v4(inspected, require_fresh=True)
        self._set_coordinator_reference_v3(summary, fresh=True)
        self._v14b_panel_bundle = panel_bundle
        return {
            "layer_plan_install": copy.deepcopy(install),
            "reference_identity": copy.deepcopy(summary["reference_identity"]),
            "panel_bundle_content_sha256": panel_bundle[
                "content_sha256_before_self_field"
            ],
        }

    def _prepared_full_frame_v14b(self):
        bundle = validate_panel_bundle_v14b(self._v14b_panel_bundle)
        full = bundle["full_frame"]
        prompts = [base.specialist_template(question) for question in full["questions"]]
        templated = canonical_sha256({"prompts": prompts, "answers": full["answers"]})
        if templated != FULL_TEMPLATED_QA_SHA256_V14B:
            raise RuntimeError("v14b templated prompt identity changed")
        dense_items = anchor_v4.prepare_gold_answer_items_v4(
            self.tokenizer, prompts, full["answers"],
        )
        prompt_items = [
            {"prompt_token_ids": item["prompt_token_ids"]} for item in dense_items
        ]
        if len(prompt_items) != 481:
            raise RuntimeError("v14b prompt count changed")
        return dense_items, prompt_items, templated

    def _base_probe_v14b(self, dense_items, prompt_items):
        outputs = anchor_v14a.anchor_v11.anchor_v1.dispatch_eval_batch(
            self.engines, prompt_items, self._dense_sampling_params_v4(0),
            self._resolve,
        )
        score = _score_full_outputs_v14b(
            dense_items, outputs, self._v14b_panel_bundle,
        )
        return {
            "schema": "eggroll-es-k2-base-probe-v14b",
            "dense_result_sha256": score["dense_result_sha256"],
            "request_count": len(prompt_items),
        }

    def estimate_full_frame_v14b(self, seeds):
        seeds = [int(seed) for seed in seeds]
        if seeds != PERTURBATION_SEEDS_V14B:
            raise RuntimeError("v14b perturbation basis changed")
        dense_items, prompt_items, templated = self._prepared_full_frame_v14b()
        pre_probe = self._base_probe_v14b(dense_items, prompt_items)
        responses = {
            "full_frame_sign_scores": {sign: [] for sign in SIGNS_V14B},
            "matched56_sign_scores": {
                name: {sign: [] for sign in SIGNS_V14B}
                for name in prereg_v14b.PANEL_NAMES_V14B
            },
            "complement_sign_scores": {
                name: {sign: [] for sign in SIGNS_V14B}
                for name in prereg_v14b.PANEL_NAMES_V14B[3:]
            },
            "dense_result_sha256": {sign: [] for sign in SIGNS_V14B},
        }
        for start in range(0, len(seeds), 4):
            wave = anchor_v10.validate_full_engine_wave_v10(seeds[start:start + 4], 4)
            for sign, negate in (("plus", False), ("minus", True)):
                batches = None
                try:
                    self._resolve([
                        self.engines[index].collective_rpc.remote(
                            "perturb_self_weights",
                            args=(int(seed), self.sigma, negate),
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
                    not isinstance(batches, list) or len(batches) != 4
                    or any(len(batch) != 481 for batch in batches)
                ):
                    raise RuntimeError("v14b resident generation wave is incomplete")
                for index, _seed in enumerate(wave):
                    score = _score_full_outputs_v14b(
                        dense_items, batches[index], self._v14b_panel_bundle,
                    )
                    responses["full_frame_sign_scores"][sign].append(score["full_frame"])
                    responses["dense_result_sha256"][sign].append(
                        score["dense_result_sha256"]
                    )
                    for name in prereg_v14b.PANEL_NAMES_V14B:
                        responses["matched56_sign_scores"][name][sign].append(
                            score["matched56"][name]
                        )
                    for name in prereg_v14b.PANEL_NAMES_V14B[3:]:
                        responses["complement_sign_scores"][name][sign].append(
                            score["complements"][name]
                        )
        validate_responses_v14b(responses)
        checks = self._resolve([
            engine.collective_rpc.remote("verify_self_exact_reference", args=())
            for engine in self.engines
        ])
        if not anchor_v4.anchor_v3.anchor_v2._all_collective_results(
            checks,
            lambda value: isinstance(value, dict) and value.get("passed") is True,
        ):
            raise RuntimeError("v14b exact reference check failed")
        boundary = self._population_boundary_audit_v4(0)
        post_probe = self._base_probe_v14b(dense_items, prompt_items)
        if pre_probe != post_probe:
            raise RuntimeError("v14b alpha-zero base probe drifted")
        identity_audit = {
            "pre_probe": pre_probe, "post_probe": post_probe,
            "exact_reference_checks": checks, "passed": True,
        }
        hardware_coverage = {
            "engine_count": 4, "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3], "population_waves": 8,
            "signed_waves": 16, "partial_waves": 0,
            "all_engines_generate_every_signed_wave": True,
        }
        integrity_audits = build_integrity_audits_v14b(
            alpha=0.0, applications=[], model_update_applied=False,
            identity_audit=identity_audit,
            population_boundary_audit=boundary,
            hardware_coverage=hardware_coverage,
        )
        analysis = analyze_responses_v14b(responses, integrity_audits)
        artifact = {
            "schema": "eggroll-es-k2-resident-sign-diagnostic-v14b",
            "iteration": 0, "alpha": 0.0,
            "model_update_applied": False, "applications": [],
            "perturbation_basis": {
                "seeds": seeds, "basis_sha256": PERTURBATION_BASIS_SHA256_V14B,
                "sign_order": list(SIGNS_V14B),
            },
            "panel_bundle_content_sha256": self._v14b_panel_bundle[
                "content_sha256_before_self_field"
            ],
            "templated_prompt_answer_sha256": templated,
            "generation_contract": {
                "prompts_per_engine_per_sign": 481,
                "generation_calls_per_engine_per_sign": 1,
                "document_means_before_equal_document_aggregation": True,
                "matched_and_crossfit_derived_without_generation": True,
            },
            "responses": responses, "integrity_audits": integrity_audits,
            "analysis": analysis, "identity_audit": identity_audit,
            "population_boundary_audit_v4": boundary,
            "hardware_coverage": hardware_coverage,
            "interpretation": "train_only_k2_estimator_gate_no_model_update",
        }
        artifact["content_sha256_before_self_field"] = canonical_sha256(artifact)
        validate_diagnostic_v14b(artifact)
        return artifact

    def estimate_step_coefficients(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v14b requires the exact k2 diagnostic entrypoint")

    def apply_seed_coefficients(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v14b alpha-zero diagnostic forbids model updates")


def validate_diagnostic_v14b(artifact):
    recomputed_integrity = build_integrity_audits_v14b(
        alpha=artifact.get("alpha"),
        applications=artifact.get("applications"),
        model_update_applied=artifact.get("model_update_applied"),
        identity_audit=artifact.get("identity_audit", {}),
        population_boundary_audit=artifact.get(
            "population_boundary_audit_v4", {}
        ),
        hardware_coverage=artifact.get("hardware_coverage"),
    )
    if (
        not isinstance(artifact, dict)
        or set(artifact) != {
            "schema", "iteration", "alpha", "model_update_applied",
            "applications", "perturbation_basis",
            "panel_bundle_content_sha256", "templated_prompt_answer_sha256",
            "generation_contract", "responses", "integrity_audits",
            "analysis", "identity_audit", "population_boundary_audit_v4",
            "hardware_coverage", "interpretation",
            "content_sha256_before_self_field",
        }
        or artifact.get("schema") != "eggroll-es-k2-resident-sign-diagnostic-v14b"
        or artifact.get("iteration") != 0 or artifact.get("alpha") != 0.0
        or artifact.get("model_update_applied") is not False
        or artifact.get("applications") != []
        or artifact.get("perturbation_basis") != {
            "seeds": PERTURBATION_SEEDS_V14B,
            "basis_sha256": PERTURBATION_BASIS_SHA256_V14B,
            "sign_order": list(SIGNS_V14B),
        }
        or artifact.get("panel_bundle_content_sha256")
        != PANEL_BUNDLE_CONTENT_SHA256_V14B
        or artifact.get("templated_prompt_answer_sha256")
        != FULL_TEMPLATED_QA_SHA256_V14B
        or artifact.get("generation_contract") != {
            "prompts_per_engine_per_sign": 481,
            "generation_calls_per_engine_per_sign": 1,
            "document_means_before_equal_document_aggregation": True,
            "matched_and_crossfit_derived_without_generation": True,
        }
        or artifact.get("integrity_audits") != recomputed_integrity
        or artifact.get("analysis")
        != analyze_responses_v14b(
            artifact.get("responses"), recomputed_integrity,
        )
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
        raise RuntimeError("v14b diagnostic changed or does not recompute")
    validate_responses_v14b(artifact["responses"])
    return artifact


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v14b(captured)
    parent = anchor_v14a.load_trainer(captured)

    class PairedDistinctRowTrainerV14B(PairedDistinctRowDiagnosticMixinV14B, parent):
        pass

    return PairedDistinctRowTrainerV14B
