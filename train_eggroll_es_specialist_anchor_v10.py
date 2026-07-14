#!/usr/bin/env python3
"""Antithetic, crossed-domain/anchor diagnostic trainer v10."""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path
from types import FunctionType

import eggroll_es_worker_v10 as worker_v10
import train_eggroll_es_specialist_anchor_v8 as anchor_v8


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = (
    "eggroll_es_worker_v10.AntitheticCrossedAuditWorkerExtensionV10"
)
REQUIRED_ENGINE_COUNT = anchor_v8.REQUIRED_ENGINE_COUNT
BASE_DIRECTION_COUNT_V10 = 32
UNIQUE_SIGNED_DIRECTION_COUNT_V10 = 64
DOMAIN_MANIFEST_COUNT_V10 = 2
ACTUAL_PERTURB_RESTORE_CYCLE_COUNT_V10 = 128
DOMAIN_SIGNED_SCORE_COUNT_V10 = 128
ANCHOR_GENERATION_SEED_COUNT_V10 = 2
ANCHOR_SIGNED_RESPONSE_COUNT_V10 = 128
FROZEN_STABILITY_PLANS_V10 = anchor_v8.FROZEN_STABILITY_PLANS_V8
MODEL_CONFIG_SHA256_V10 = anchor_v8.MODEL_CONFIG_SHA256_V8
PERTURBATION_SEEDS_V10 = [
    140002291, 1028842752, 480373990, 1037026679,
    759861149, 227761095, 428721957, 150663570,
    863156398, 658045682, 947615772, 615729462,
    958574585, 1048698881, 573870406, 938961107,
    635418192, 42810614, 120988352, 643498157,
    552296232, 491294404, 607579228, 525239262,
    401755469, 1009385806, 266538765, 936309830,
    1063141744, 575175855, 1010837104, 983113236,
]
DOMAIN_MANIFESTS_V10 = {
    "D43": {
        "seed": 43, "rows": 64,
        "sha256": (
            "b864cfcc4ebcd987d8091f1067f631366c128d63d09fb7160a09561d10063a0f"
        ),
    },
    "D44": {
        "seed": 44, "rows": 64,
        "sha256": (
            "3574ff126f727a262957f34ab83fbefce6754ae9e4be790f810f42656e692bc2"
        ),
    },
}
COMBINED_DOMAIN_MANIFEST_SHA256_V10 = (
    "a1b77aed57313c0dec44195a35232818426668a771db2c5055bb6b28c304289a"
)
_DEFAULT_LAYER_PLAN_BUNDLE = None
anchor_v5 = anchor_v8.anchor_v7.anchor_v6.anchor_v5
anchor_v4 = anchor_v5.anchor_v4
anchor_v1 = anchor_v4.anchor_v3.anchor_v2.anchor_v1
robust_anchor = anchor_v5.robust_anchor
canonical_sha256 = anchor_v8.canonical_sha256
coefficient_sha256 = anchor_v8.coefficient_sha256
file_sha256 = anchor_v8.file_sha256
load_anchor_prose = anchor_v8.load_anchor_prose


def _clone_with_globals(function, replacements, name):
    namespace = dict(function.__globals__)
    namespace.update(replacements)
    clone = FunctionType(
        function.__code__, namespace, name, function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__doc__ = function.__doc__
    clone.__module__ = __name__
    clone.__qualname__ = name
    return clone


def validate_frozen_layer_plan_bundle_v10(bundle):
    return anchor_v8.validate_frozen_layer_plan_bundle_v8(bundle)


def load_frozen_layer_plan_v10(*args, **kwargs):
    bundle = anchor_v8.load_frozen_layer_plan_v8(*args, **kwargs)
    validate_frozen_layer_plan_bundle_v10(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v10(argv):
    bundle, remaining = anchor_v8.parse_frozen_layer_plan_cli_v8(argv)
    validate_frozen_layer_plan_bundle_v10(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v10(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v10(bundle)
    anchor_v8.set_default_layer_plan_bundle_v8(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def _scores(result, seeds):
    rows = result.get("robust_scores") if isinstance(result, dict) else None
    mapping = {
        row.get("seed"): row.get("score")
        for row in rows or [] if isinstance(row, dict)
    }
    if set(mapping) != set(seeds):
        raise RuntimeError("v10 robust anchor result changed its seed set")
    return [float(mapping[seed]) for seed in seeds]


def _central(plus, minus):
    if len(plus) != len(minus):
        raise RuntimeError("v10 antithetic sign vectors changed length")
    values = [0.5 * (float(p) - float(m)) for p, m in zip(plus, minus)]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("v10 antithetic central difference is non-finite")
    return values


def validate_full_engine_wave_v10(wave, engine_count):
    """Reject any wave that would leave a required V10 engine idle."""
    if int(engine_count) != REQUIRED_ENGINE_COUNT:
        raise ValueError("v10 antithetic waves require exactly four engines")
    wave = list(wave)
    if len(wave) != REQUIRED_ENGINE_COUNT:
        raise ValueError("partial v10 antithetic wave would idle a GPU")
    if len(set(wave)) != REQUIRED_ENGINE_COUNT:
        raise ValueError("v10 antithetic wave contains duplicate seeds")
    return wave


def _validate_sign_vectors_v10(mapping, labels, seeds, surface):
    if set(mapping) != set(labels):
        raise RuntimeError(f"v10 {surface} labels changed")
    for label in labels:
        sign_map = mapping[label]
        if not isinstance(sign_map, dict) or set(sign_map) != {"plus", "minus"}:
            raise RuntimeError(f"v10 {surface} sign map changed")
        for sign in ("plus", "minus"):
            values = sign_map[sign]
            if (
                not isinstance(values, list)
                or len(values) != len(seeds)
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in values
                )
            ):
                raise RuntimeError(f"v10 {surface} signed vector changed")


def _build_cross_artifact_v10(plan, captures):
    seeds = list(plan["seeds"])
    anchor_results = {
        "A43_plus": plan["document_lcb_anchor_v5"],
        **captures["anchor_results"],
    }
    anchor_sign_scores = {
        "A43": {
            "plus": _scores(anchor_results["A43_plus"], seeds),
            "minus": _scores(anchor_results["A43_minus"], seeds),
        },
        "A44": {
            "plus": _scores(anchor_results["A44_plus"], seeds),
            "minus": _scores(anchor_results["A44_minus"], seeds),
        },
    }
    domain_sign_scores = copy.deepcopy(captures["domain_sign_scores"])
    central_domain = {
        label: _central(values["plus"], values["minus"])
        for label, values in domain_sign_scores.items()
    }
    central_anchor = {
        label: _central(values["plus"], values["minus"])
        for label, values in anchor_sign_scores.items()
    }
    cells = {}
    for domain_label in ("D43", "D44"):
        for anchor_label in ("A43", "A44"):
            name = f"{domain_label}x{anchor_label}"
            projection = anchor_v1.project_anchor_safe_coefficients(
                central_domain[domain_label], central_anchor[anchor_label],
                min_anchor_cosine=0.8,
            )
            coefficients = projection["coefficients"]
            cells[name] = {
                "domain_manifest": domain_label,
                "anchor_generation_seed": int(anchor_label[1:]),
                "coefficients": coefficients,
                "coefficient_sha256": coefficient_sha256(
                    seeds, coefficients,
                ),
                "projection": projection["diagnostics"],
            }
    result_bindings = {
        key: value["content_sha256_before_self_field"]
        for key, value in anchor_results.items()
    }
    artifact = {
        "schema": "eggroll-es-antithetic-crossed-plan-v10",
        "population_size": BASE_DIRECTION_COUNT_V10,
        "base_direction_count": BASE_DIRECTION_COUNT_V10,
        "unique_signed_direction_count": UNIQUE_SIGNED_DIRECTION_COUNT_V10,
        "domain_manifest_count": DOMAIN_MANIFEST_COUNT_V10,
        "actual_perturb_restore_cycle_count": (
            ACTUAL_PERTURB_RESTORE_CYCLE_COUNT_V10
        ),
        "domain_signed_score_count": DOMAIN_SIGNED_SCORE_COUNT_V10,
        "anchor_generation_seed_count": ANCHOR_GENERATION_SEED_COUNT_V10,
        "anchor_signed_response_count": ANCHOR_SIGNED_RESPONSE_COUNT_V10,
        "base_perturbation_seeds": seeds,
        "sign_order": ["plus", "minus"],
        "sigma": 0.0003,
        "domain_manifests": copy.deepcopy(DOMAIN_MANIFESTS_V10),
        "combined_domain_manifest_sha256": (
            COMBINED_DOMAIN_MANIFEST_SHA256_V10
        ),
        "anchor_generation_seeds": [43, 44],
        "anchor_reference_identities": copy.deepcopy(
            captures["anchor_reference_identities"]
        ),
        "domain_sign_scores": domain_sign_scores,
        "anchor_sign_scores": anchor_sign_scores,
        "central_domain_scores": central_domain,
        "central_anchor_scores": central_anchor,
        "anchor_result_bindings": result_bindings,
        "additional_anchor_results": {
            key: value for key, value in anchor_results.items()
            if key != "A43_plus"
        },
        "cells": cells,
        "benchmark_treatment_applied": False,
        "selection_surface": "crossed_train_anchor_response_only",
    }
    artifact["content_sha256_before_self_field"] = canonical_sha256(artifact)
    return artifact


def validate_antithetic_cross_v10(plan, *, recompute_numeric=False):
    artifact = plan.get("antithetic_cross_v10") if isinstance(plan, dict) else None
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema")
        != "eggroll-es-antithetic-crossed-plan-v10"
        or artifact.get("population_size") != BASE_DIRECTION_COUNT_V10
        or artifact.get("base_direction_count") != BASE_DIRECTION_COUNT_V10
        or artifact.get("unique_signed_direction_count")
        != UNIQUE_SIGNED_DIRECTION_COUNT_V10
        or artifact.get("domain_manifest_count") != DOMAIN_MANIFEST_COUNT_V10
        or artifact.get("actual_perturb_restore_cycle_count")
        != ACTUAL_PERTURB_RESTORE_CYCLE_COUNT_V10
        or artifact.get("domain_signed_score_count")
        != DOMAIN_SIGNED_SCORE_COUNT_V10
        or artifact.get("anchor_generation_seed_count")
        != ANCHOR_GENERATION_SEED_COUNT_V10
        or artifact.get("anchor_signed_response_count")
        != ANCHOR_SIGNED_RESPONSE_COUNT_V10
        or artifact.get("base_perturbation_seeds") != PERTURBATION_SEEDS_V10
        or artifact.get("sign_order") != ["plus", "minus"]
        or artifact.get("sigma") != 0.0003
        or artifact.get("domain_manifests") != DOMAIN_MANIFESTS_V10
        or artifact.get("combined_domain_manifest_sha256")
        != COMBINED_DOMAIN_MANIFEST_SHA256_V10
        or artifact.get("anchor_generation_seeds") != [43, 44]
        or artifact.get("benchmark_treatment_applied") is not False
        or artifact.get("selection_surface")
        != "crossed_train_anchor_response_only"
        or artifact.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in artifact.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v10 crossed antithetic artifact identity changed")
    seeds = artifact["base_perturbation_seeds"]
    _validate_sign_vectors_v10(
        artifact.get("domain_sign_scores"), ("D43", "D44"), seeds,
        "domain",
    )
    _validate_sign_vectors_v10(
        artifact.get("anchor_sign_scores"), ("A43", "A44"), seeds,
        "anchor",
    )
    if set(artifact.get("anchor_reference_identities", {})) != {"A43", "A44"}:
        raise RuntimeError("v10 anchor reference identities changed")
    if set(artifact.get("additional_anchor_results", {})) != {
        "A43_minus", "A44_plus", "A44_minus",
    }:
        raise RuntimeError("v10 additional anchor result coverage changed")
    expected_cells = {
        "D43xA43": ("D43", 43), "D43xA44": ("D43", 44),
        "D44xA43": ("D44", 43), "D44xA44": ("D44", 44),
    }
    if set(artifact.get("cells", {})) != set(expected_cells):
        raise RuntimeError("v10 crossed cell coverage changed")
    for name, (domain_label, generation_seed) in expected_cells.items():
        cell = artifact["cells"][name]
        if (
            cell.get("domain_manifest") != domain_label
            or cell.get("anchor_generation_seed") != generation_seed
        ):
            raise RuntimeError("v10 crossed cell/reference binding changed")
    captures = {
        "domain_sign_scores": artifact["domain_sign_scores"],
        "anchor_reference_identities": artifact[
            "anchor_reference_identities"
        ],
        "anchor_results": artifact["additional_anchor_results"],
    }
    expected = _build_cross_artifact_v10(plan, captures)
    if expected != artifact:
        raise RuntimeError("v10 crossed antithetic numeric replay changed")
    if recompute_numeric:
        robust_anchor.validate_document_lcb_result(
            plan["document_lcb_anchor_v5"]
        )
        for result in artifact["additional_anchor_results"].values():
            robust_anchor.validate_document_lcb_result(result)
    return {
        "schema": artifact["schema"],
        "content_sha256": artifact["content_sha256_before_self_field"],
        "cell_coefficient_sha256": {
            key: value["coefficient_sha256"]
            for key, value in artifact["cells"].items()
        },
    }


class AntitheticCrossedContractMixinV10:
    def configure_anchor(self, *args, **kwargs):
        result = super().configure_anchor(*args, **kwargs)
        self._v10_reference_a44 = None
        self._v10_reference_identities = {}
        return result

    def _anchor_sampling_seed_v10(self, iteration, seed):
        return self._sampling_params(
            n=1, seed=int(seed) + int(iteration), temperature=0.0,
            top_p=1.0, max_tokens=1, prompt_logprobs=1, detokenize=False,
        )

    def _identity_probe(self, input_batch, domain_sampling, anchor_items, iteration):
        audit = super()._identity_probe(
            input_batch, domain_sampling, anchor_items, iteration,
        )
        dispatch = anchor_v1.dispatch_eval_batch
        outputs = dispatch(
            self.engines,
            [{"prompt_token_ids": item["prompt_token_ids"]}
             for item in anchor_items],
            self._anchor_sampling_seed_v10(iteration, 44),
            self._resolve,
        )
        summaries = anchor_v5.summarize_anchor_documents_v5(
            anchor_items, outputs,
        )
        identity = robust_anchor.reference_document_summary_identity(summaries)
        probe_count = int(self._v4_identity_probe_count)
        if probe_count == 1:
            self._v10_reference_a44 = summaries
            self._v10_reference_identities = {
                "A43": copy.deepcopy(
                    audit["document_lcb_anchor_v5"]
                ),
                "A44": {
                    "config_sha256": anchor_v5.DOCUMENT_LCB_CONFIG_SHA256_V5,
                    **identity, "generation_seed": 44,
                    "raw_document_content_persisted": False,
                },
            }
        elif probe_count == 2 and summaries != self._v10_reference_a44:
            raise RuntimeError("v10 seed44 anchor reference drifted")
        audit["crossed_anchor_reference_v10"] = copy.deepcopy(
            self._v10_reference_identities["A44"]
        )
        return audit

    def _evaluate_population_with_anchor(
        self, seeds, input_batch, target_batch, domain_sampling_params,
        anchor_items, iteration,
    ):
        del domain_sampling_params
        call_index = int(getattr(self, "_v10_population_call_index", 0))
        if call_index not in (0, 1):
            raise RuntimeError("v10 received an extra crossed domain minibatch")
        domain_label = ("D43", "D44")[call_index]
        self._v10_population_call_index = call_index + 1
        dense_items = anchor_v4.prepare_gold_answer_items_v4(
            self.tokenizer, input_batch, target_batch,
        )
        if len(dense_items) != 64:
            raise RuntimeError("v10 crossed domain manifest must contain 64 rows")
        dense_prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in dense_items
        ]
        dense_sampling = self._dense_sampling_params_v4(iteration)
        anchor_prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in anchor_items
        ]
        plus_metrics = {}
        sign_scores = {"plus": {}, "minus": {}}
        population_documents = {
            label: {sign: {} for sign in ("plus", "minus")}
            for label in ("A43", "A44")
        }
        results = []
        for start in range(0, len(seeds), len(self.engines)):
            wave = validate_full_engine_wave_v10(
                seeds[start:start + len(self.engines)], len(self.engines),
            )
            for sign, negate in (("plus", False), ("minus", True)):
                dense_batches = None
                anchor_43_batches = None
                anchor_44_batches = None
                try:
                    self._resolve([
                        self.engines[index].collective_rpc.remote(
                            "perturb_self_weights",
                            args=(int(seed), self.sigma, negate),
                        )
                        for index, seed in enumerate(wave)
                    ])
                    dense_batches = self._resolve([
                        self.engines[index].generate.remote(
                            list(dense_prompts), dense_sampling,
                            use_tqdm=False,
                        )
                        for index, _ in enumerate(wave)
                    ])
                    if anchor_items:
                        anchor_43_batches = self._resolve([
                            self.engines[index].generate.remote(
                                list(anchor_prompts),
                                self._anchor_sampling_seed_v10(iteration, 43),
                                use_tqdm=False,
                            )
                            for index, _ in enumerate(wave)
                        ])
                        anchor_44_batches = self._resolve([
                            self.engines[index].generate.remote(
                                list(anchor_prompts),
                                self._anchor_sampling_seed_v10(iteration, 44),
                                use_tqdm=False,
                            )
                            for index, _ in enumerate(wave)
                        ])
                finally:
                    self._restore_all_engines_exact()
                for index, seed in enumerate(wave):
                    dense = anchor_v4.score_gold_answer_outputs_v4(
                        dense_items, dense_batches[index],
                    )
                    metrics = {
                        "avg_reward": dense["mean_example_mean_logprob"],
                        "rewards": [
                            row["mean_answer_token_logprob"]
                            for row in dense["examples"]
                        ],
                        "results": dense["examples"],
                        "dense_gold_reward_v4": dense,
                    }
                    sign_scores[sign][int(seed)] = metrics["avg_reward"]
                    if sign == "plus":
                        plus_metrics[int(seed)] = metrics
                        results.append({
                            "seed": int(seed),
                            "avg_reward": metrics["avg_reward"],
                        })
                    if anchor_items:
                        population_documents["A43"][sign][int(seed)] = (
                            anchor_v5.summarize_anchor_documents_v5(
                                anchor_items, anchor_43_batches[index],
                            )
                        )
                        population_documents["A44"][sign][int(seed)] = (
                            anchor_v5.summarize_anchor_documents_v5(
                                anchor_items, anchor_44_batches[index],
                            )
                        )
        expected_seed_set = set(int(seed) for seed in seeds)
        if (
            set(sign_scores) != {"plus", "minus"}
            or any(set(mapping) != expected_seed_set for mapping in sign_scores.values())
        ):
            raise RuntimeError("v10 domain sign map is incomplete")
        self._v10_domain_sign_maps[domain_label] = sign_scores
        if not anchor_items:
            return plus_metrics, {}, results
        references = {
            "A43": self._v5_reference_document_summaries,
            "A44": self._v10_reference_a44,
        }
        anchor_results = {}
        for label in ("A43", "A44"):
            for sign in ("plus", "minus"):
                key = f"{label}_{sign}"
                anchor_results[key] = robust_anchor.score_population_document_lcbs(
                    references[label], [
                        {"seed": int(seed),
                         "documents": population_documents[label][sign][int(seed)]}
                        for seed in seeds
                    ],
                )
        self._v5_pending_robust_result = anchor_results["A43_plus"]
        self._v10_anchor_results = {
            key: value for key, value in anchor_results.items()
            if key != "A43_plus"
        }
        if set(self._v10_anchor_results) != {
            "A43_minus", "A44_plus", "A44_minus",
        }:
            raise RuntimeError("v10 anchor sign/reference map is incomplete")
        anchor_scores = dict(zip(
            seeds, _scores(anchor_results["A43_plus"], seeds),
        ))
        return plus_metrics, anchor_scores, results

    def estimate_step_coefficients(self, *args, **kwargs):
        self._v10_population_call_index = 0
        self._v10_domain_sign_maps = {}
        self._v10_anchor_results = {}
        plan = super().estimate_step_coefficients(*args, **kwargs)
        if self._v10_population_call_index != 2:
            raise RuntimeError("v10 did not evaluate both domain manifests")
        if set(self._v10_domain_sign_maps) != {"D43", "D44"}:
            raise RuntimeError("v10 crossed domain maps are incomplete")
        seeds = list(plan["seeds"])
        captures = {
            "domain_sign_scores": {
                label: {
                    sign: [mapping[seed] for seed in seeds]
                    for sign, mapping in sign_maps.items()
                }
                for label, sign_maps in self._v10_domain_sign_maps.items()
            },
            "anchor_reference_identities": copy.deepcopy(
                self._v10_reference_identities
            ),
            "anchor_results": copy.deepcopy(self._v10_anchor_results),
        }
        plan["antithetic_cross_v10"] = _build_cross_artifact_v10(
            plan, captures,
        )
        validate_antithetic_cross_v10(plan, recompute_numeric=True)
        self._persist_anchor_plan(plan)
        return plan

    def apply_seed_coefficients(self, plan, target_alpha):
        if float(target_alpha) != 0.0:
            raise ValueError("v10 variance diagnostic forbids nonzero alpha")
        validate_antithetic_cross_v10(plan, recompute_numeric=True)
        return super().apply_seed_coefficients(plan, target_alpha)


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v10(captured)
    parent = anchor_v8.load_trainer(captured)
    parent.launch_engines = _clone_with_globals(
        parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v10",
    )

    class AntitheticCrossedTrainerV10(
        AntitheticCrossedContractMixinV10, parent,
    ):
        pass

    return AntitheticCrossedTrainerV10
