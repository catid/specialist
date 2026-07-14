#!/usr/bin/env python3
"""Resident-sign exact-equivalence trainer v11."""

from __future__ import annotations

import copy
import math
import os
import sys
from pathlib import Path

import eggroll_es_worker_v11 as worker_v11
import train_eggroll_es_specialist_anchor_v10 as anchor_v10


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = "eggroll_es_worker_v11.ResidentSignAuditWorkerExtensionV11"
REQUIRED_ENGINE_COUNT = anchor_v10.REQUIRED_ENGINE_COUNT
FROZEN_STABILITY_PLANS_V11 = anchor_v10.FROZEN_STABILITY_PLANS_V10
MODEL_CONFIG_SHA256_V11 = anchor_v10.MODEL_CONFIG_SHA256_V10
PERTURBATION_SEEDS_V11 = list(anchor_v10.PERTURBATION_SEEDS_V10)
DOMAIN_MANIFESTS_V11 = copy.deepcopy(anchor_v10.DOMAIN_MANIFESTS_V10)
COMBINED_DOMAIN_MANIFEST_SHA256_V11 = (
    anchor_v10.COMBINED_DOMAIN_MANIFEST_SHA256_V10
)
BASE_DIRECTION_COUNT_V11 = 32
UNIQUE_SIGNED_DIRECTION_COUNT_V11 = 64
ACTUAL_PERTURB_RESTORE_CYCLE_COUNT_V11 = 64
ALL_ENGINE_SIGN_RESIDENCY_COUNT_V11 = 16
DOMAIN_SIGNED_SCORE_COUNT_V11 = 128
ANCHOR_SIGNED_RESPONSE_COUNT_V11 = 128
V10_EQUIVALENCE_TARGET_V11 = {
    "schema": "eggroll-es-v10-equivalence-target-v11",
    "journal_file_sha256": (
        "2708b563034367479da9b25f3fcd8bd556b0c2133f533b3b561fcfd46d9af5ee"
    ),
    "journal_content_sha256": (
        "3e68b1fb925378e31c9c4945de82d33c34f77c6abad585d0415ec456e78d71c7"
    ),
    "cross_artifact_content_sha256": (
        "e85cc13af630a6201c7e0f4777e439d6c5aab70fc69eb41a56d8337cb8d613b7"
    ),
    "cell_coefficient_sha256": {
        "D43xA43": (
            "72cc7196f6ac43c0ff2ded27e3b2c9b7516b35d2a0eb62941e7f4e309857c055"
        ),
        "D43xA44": (
            "72cc7196f6ac43c0ff2ded27e3b2c9b7516b35d2a0eb62941e7f4e309857c055"
        ),
        "D44xA43": (
            "002b44dcbc2544336c7d49bb9f7390f07aed03a17d55458c9774a6cbb213ae75"
        ),
        "D44xA44": (
            "002b44dcbc2544336c7d49bb9f7390f07aed03a17d55458c9774a6cbb213ae75"
        ),
    },
}
_DEFAULT_LAYER_PLAN_BUNDLE = None
anchor_v8 = anchor_v10.anchor_v8
anchor_v5 = anchor_v10.anchor_v5
anchor_v4 = anchor_v10.anchor_v4
anchor_v1 = anchor_v10.anchor_v1
robust_anchor = anchor_v10.robust_anchor
canonical_sha256 = anchor_v10.canonical_sha256
coefficient_sha256 = anchor_v10.coefficient_sha256
file_sha256 = anchor_v10.file_sha256
load_anchor_prose = anchor_v10.load_anchor_prose


def validate_frozen_layer_plan_bundle_v11(bundle):
    return anchor_v10.validate_frozen_layer_plan_bundle_v10(bundle)


def load_frozen_layer_plan_v11(*args, **kwargs):
    bundle = anchor_v10.load_frozen_layer_plan_v10(*args, **kwargs)
    validate_frozen_layer_plan_bundle_v11(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v11(argv):
    bundle, remaining = anchor_v10.parse_frozen_layer_plan_cli_v10(argv)
    validate_frozen_layer_plan_bundle_v11(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v11(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v11(bundle)
    anchor_v10.set_default_layer_plan_bundle_v10(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def _resident_artifact_v11(v10_artifact):
    artifact = copy.deepcopy(v10_artifact)
    artifact.pop("content_sha256_before_self_field", None)
    artifact.update({
        "schema": "eggroll-es-resident-sign-crossed-plan-v11",
        "actual_perturb_restore_cycle_count": (
            ACTUAL_PERTURB_RESTORE_CYCLE_COUNT_V11
        ),
        "all_engine_sign_residency_count": (
            ALL_ENGINE_SIGN_RESIDENCY_COUNT_V11
        ),
        "resident_generation_order": ["D43", "A43", "A44", "D44"],
        "second_parent_call": "validated_one_use_D44_cache_no_engine_dispatch",
        "v10_equivalence_target": copy.deepcopy(
            V10_EQUIVALENCE_TARGET_V11
        ),
    })
    artifact["content_sha256_before_self_field"] = canonical_sha256(artifact)
    return artifact


def _build_resident_artifact_v11(plan, captures):
    return _resident_artifact_v11(
        anchor_v10._build_cross_artifact_v10(plan, captures)
    )


def _finite_score_maps_v11(mapping, labels, seeds, surface):
    if not isinstance(mapping, dict) or set(mapping) != set(labels):
        raise RuntimeError(f"v11 {surface} labels changed")
    for label in labels:
        signs = mapping[label]
        if not isinstance(signs, dict) or set(signs) != {"plus", "minus"}:
            raise RuntimeError(f"v11 {surface} sign map changed")
        for values in signs.values():
            if (
                not isinstance(values, list)
                or len(values) != len(seeds)
                or not all(math.isfinite(float(value)) for value in values)
            ):
                raise RuntimeError(f"v11 {surface} signed vector changed")


def validate_resident_cross_v11(plan, *, recompute_numeric=False):
    artifact = plan.get("resident_sign_cross_v11") if isinstance(plan, dict) else None
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema") != "eggroll-es-resident-sign-crossed-plan-v11"
        or artifact.get("population_size") != BASE_DIRECTION_COUNT_V11
        or artifact.get("base_direction_count") != BASE_DIRECTION_COUNT_V11
        or artifact.get("unique_signed_direction_count")
        != UNIQUE_SIGNED_DIRECTION_COUNT_V11
        or artifact.get("actual_perturb_restore_cycle_count")
        != ACTUAL_PERTURB_RESTORE_CYCLE_COUNT_V11
        or artifact.get("all_engine_sign_residency_count")
        != ALL_ENGINE_SIGN_RESIDENCY_COUNT_V11
        or artifact.get("domain_signed_score_count")
        != DOMAIN_SIGNED_SCORE_COUNT_V11
        or artifact.get("anchor_signed_response_count")
        != ANCHOR_SIGNED_RESPONSE_COUNT_V11
        or artifact.get("base_perturbation_seeds") != PERTURBATION_SEEDS_V11
        or artifact.get("domain_manifests") != DOMAIN_MANIFESTS_V11
        or artifact.get("combined_domain_manifest_sha256")
        != COMBINED_DOMAIN_MANIFEST_SHA256_V11
        or artifact.get("anchor_generation_seeds") != [43, 44]
        or artifact.get("sign_order") != ["plus", "minus"]
        or artifact.get("resident_generation_order")
        != ["D43", "A43", "A44", "D44"]
        or artifact.get("second_parent_call")
        != "validated_one_use_D44_cache_no_engine_dispatch"
        or artifact.get("v10_equivalence_target")
        != V10_EQUIVALENCE_TARGET_V11
        or artifact.get("content_sha256_before_self_field")
        != canonical_sha256({
            key: value for key, value in artifact.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v11 resident-sign artifact identity changed")
    seeds = artifact["base_perturbation_seeds"]
    _finite_score_maps_v11(
        artifact.get("domain_sign_scores"), ("D43", "D44"), seeds,
        "domain",
    )
    _finite_score_maps_v11(
        artifact.get("anchor_sign_scores"), ("A43", "A44"), seeds,
        "anchor",
    )
    if set(artifact.get("anchor_reference_identities", {})) != {"A43", "A44"}:
        raise RuntimeError("v11 anchor reference identities changed")
    if set(artifact.get("additional_anchor_results", {})) != {
        "A43_minus", "A44_plus", "A44_minus",
    }:
        raise RuntimeError("v11 additional anchor result coverage changed")
    expected_cells = {
        "D43xA43": ("D43", 43), "D43xA44": ("D43", 44),
        "D44xA43": ("D44", 43), "D44xA44": ("D44", 44),
    }
    if set(artifact.get("cells", {})) != set(expected_cells):
        raise RuntimeError("v11 crossed cell coverage changed")
    for name, (domain, generation_seed) in expected_cells.items():
        cell = artifact["cells"][name]
        if (
            cell.get("domain_manifest") != domain
            or cell.get("anchor_generation_seed") != generation_seed
        ):
            raise RuntimeError("v11 crossed cell/reference binding changed")
    captures = {
        "domain_sign_scores": artifact["domain_sign_scores"],
        "anchor_reference_identities": artifact["anchor_reference_identities"],
        "anchor_results": artifact["additional_anchor_results"],
    }
    expected = _build_resident_artifact_v11(plan, captures)
    if expected != artifact:
        raise RuntimeError("v11 resident-sign numeric replay changed")
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


def compare_exact_v10_v11(v10_plan, v11_plan):
    """Require exact raw response and projected-plan equality with V10."""
    v10_cross = v10_plan["antithetic_cross_v10"]
    v11_cross = v11_plan["resident_sign_cross_v11"]
    cross_fields = (
        "base_perturbation_seeds", "sign_order", "sigma", "domain_manifests",
        "combined_domain_manifest_sha256", "anchor_generation_seeds",
        "anchor_reference_identities", "domain_sign_scores",
        "anchor_sign_scores", "central_domain_scores", "central_anchor_scores",
        "anchor_result_bindings", "additional_anchor_results", "cells",
        "benchmark_treatment_applied", "selection_surface",
    )
    mismatches = [
        field for field in cross_fields
        if v10_cross.get(field) != v11_cross.get(field)
    ]
    plan_fields = (
        "seeds", "coefficients", "coefficient_sha256", "domain_scores",
        "anchor_scores", "document_lcb_anchor_v5", "robust_plan_binding_v5",
    )
    mismatches.extend(
        f"plan.{field}" for field in plan_fields
        if v10_plan.get(field) != v11_plan.get(field)
    )
    if mismatches:
        raise RuntimeError(
            "v11 resident-sign execution differs from v10: "
            + ", ".join(mismatches)
        )
    binding = {
        "schema": "eggroll-es-resident-sign-exact-equivalence-v11",
        "v10_cross_content_sha256": v10_cross[
            "content_sha256_before_self_field"
        ],
        "v11_cross_content_sha256": v11_cross[
            "content_sha256_before_self_field"
        ],
        "exact_cross_fields": list(cross_fields),
        "exact_plan_fields": list(plan_fields),
        "all_exact": True,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def _score_dense_v11(dense_items, outputs):
    dense = anchor_v4.score_gold_answer_outputs_v4(dense_items, outputs)
    return {
        "avg_reward": dense["mean_example_mean_logprob"],
        "rewards": [
            row["mean_answer_token_logprob"] for row in dense["examples"]
        ],
        "results": dense["examples"],
        "dense_gold_reward_v4": dense,
    }


class ResidentSignContractMixinV11:
    def _evaluate_population_with_anchor(
        self, seeds, input_batch, target_batch, domain_sampling_params,
        anchor_items, iteration,
    ):
        del domain_sampling_params
        call_index = int(getattr(self, "_v11_population_call_index", 0))
        if call_index not in (0, 1):
            raise RuntimeError("v11 received an extra parent domain call")
        captured = self._v11_domain_batches
        label = ("D43", "D44")[call_index]
        expected_inputs, expected_targets = captured[label]
        if list(input_batch) != expected_inputs or list(target_batch) != expected_targets:
            raise RuntimeError(f"v11 parent {label} minibatch changed")
        self._v11_population_call_index = call_index + 1
        if call_index == 1:
            if anchor_items:
                raise RuntimeError("v11 cached D44 parent call received anchors")
            cache = self._v11_d44_cache
            if not isinstance(cache, dict) or self._v11_d44_cache_consumed:
                raise RuntimeError("v11 D44 cache is missing or was replayed")
            self._v11_d44_cache_consumed = True
            return (
                copy.deepcopy(cache["plus_metrics"]), {},
                copy.deepcopy(cache["results"]),
            )
        if len(anchor_items) != 128:
            raise RuntimeError("v11 D43 residency requires all 128 anchors")

        dense_items = {
            domain: anchor_v4.prepare_gold_answer_items_v4(
                self.tokenizer, *captured[domain],
            )
            for domain in ("D43", "D44")
        }
        if any(len(items) != 64 for items in dense_items.values()):
            raise RuntimeError("v11 crossed domain must contain 64 rows each")
        dense_prompts = {
            domain: [
                {"prompt_token_ids": item["prompt_token_ids"]}
                for item in dense_items[domain]
            ]
            for domain in ("D43", "D44")
        }
        dense_sampling = self._dense_sampling_params_v4(iteration)
        anchor_prompts = [
            {"prompt_token_ids": item["prompt_token_ids"]}
            for item in anchor_items
        ]
        plus_metrics = {"D43": {}, "D44": {}}
        plus_results = {"D43": [], "D44": []}
        domain_maps = {
            domain: {"plus": {}, "minus": {}}
            for domain in ("D43", "D44")
        }
        population_documents = {
            anchor: {sign: {} for sign in ("plus", "minus")}
            for anchor in ("A43", "A44")
        }
        for start in range(0, len(seeds), len(self.engines)):
            wave = anchor_v10.validate_full_engine_wave_v10(
                seeds[start:start + len(self.engines)], len(self.engines),
            )
            for sign, negate in (("plus", False), ("minus", True)):
                batches = {key: None for key in ("D43", "A43", "A44", "D44")}
                try:
                    self._resolve([
                        self.engines[index].collective_rpc.remote(
                            "perturb_self_weights",
                            args=(int(seed), self.sigma, negate),
                        )
                        for index, seed in enumerate(wave)
                    ])
                    batches["D43"] = self._resolve([
                        self.engines[index].generate.remote(
                            list(dense_prompts["D43"]), dense_sampling,
                            use_tqdm=False,
                        )
                        for index, _ in enumerate(wave)
                    ])
                    batches["A43"] = self._resolve([
                        self.engines[index].generate.remote(
                            list(anchor_prompts),
                            self._anchor_sampling_seed_v10(iteration, 43),
                            use_tqdm=False,
                        )
                        for index, _ in enumerate(wave)
                    ])
                    batches["A44"] = self._resolve([
                        self.engines[index].generate.remote(
                            list(anchor_prompts),
                            self._anchor_sampling_seed_v10(iteration, 44),
                            use_tqdm=False,
                        )
                        for index, _ in enumerate(wave)
                    ])
                    batches["D44"] = self._resolve([
                        self.engines[index].generate.remote(
                            list(dense_prompts["D44"]), dense_sampling,
                            use_tqdm=False,
                        )
                        for index, _ in enumerate(wave)
                    ])
                finally:
                    self._restore_all_engines_exact()
                if any(
                    not isinstance(batches[key], list)
                    or len(batches[key]) != len(wave)
                    for key in batches
                ):
                    raise RuntimeError("v11 resident generation wave is incomplete")
                for index, seed in enumerate(wave):
                    for domain in ("D43", "D44"):
                        metrics = _score_dense_v11(
                            dense_items[domain], batches[domain][index],
                        )
                        domain_maps[domain][sign][int(seed)] = metrics["avg_reward"]
                        if sign == "plus":
                            plus_metrics[domain][int(seed)] = metrics
                            plus_results[domain].append({
                                "seed": int(seed),
                                "avg_reward": metrics["avg_reward"],
                            })
                    population_documents["A43"][sign][int(seed)] = (
                        anchor_v5.summarize_anchor_documents_v5(
                            anchor_items, batches["A43"][index],
                        )
                    )
                    population_documents["A44"][sign][int(seed)] = (
                        anchor_v5.summarize_anchor_documents_v5(
                            anchor_items, batches["A44"][index],
                        )
                    )
        expected_seed_set = set(int(seed) for seed in seeds)
        if any(
            set(domain_maps[domain][sign]) != expected_seed_set
            for domain in domain_maps for sign in ("plus", "minus")
        ):
            raise RuntimeError("v11 domain sign map is incomplete")
        self._v11_domain_sign_maps = domain_maps
        references = {
            "A43": self._v5_reference_document_summaries,
            "A44": self._v10_reference_a44,
        }
        anchor_results = {}
        for label in ("A43", "A44"):
            for sign in ("plus", "minus"):
                anchor_results[f"{label}_{sign}"] = (
                    robust_anchor.score_population_document_lcbs(
                        references[label], [
                            {
                                "seed": int(seed),
                                "documents": population_documents[label][sign][
                                    int(seed)
                                ],
                            }
                            for seed in seeds
                        ],
                    )
                )
        self._v5_pending_robust_result = anchor_results["A43_plus"]
        self._v11_anchor_results = {
            key: value for key, value in anchor_results.items()
            if key != "A43_plus"
        }
        anchor_scores = dict(zip(
            seeds,
            anchor_v10._scores(anchor_results["A43_plus"], seeds),
        ))
        self._v11_d44_cache = {
            "plus_metrics": copy.deepcopy(plus_metrics["D44"]),
            "results": copy.deepcopy(plus_results["D44"]),
        }
        return plus_metrics["D43"], anchor_scores, plus_results["D43"]

    def estimate_step_coefficients(
        self, iteration, seeds, input_text, target_text,
    ):
        if len(input_text) != 128 or len(target_text) != 128:
            raise RuntimeError("v11 requires the exact combined 128-row batch")
        self._v11_domain_batches = {
            "D43": (list(input_text[:64]), list(target_text[:64])),
            "D44": (list(input_text[64:]), list(target_text[64:])),
        }
        for label in ("D43", "D44"):
            questions, answers = self._v11_domain_batches[label]
            identity = canonical_sha256({
                "questions": questions, "answers": answers,
            })
            if identity != DOMAIN_MANIFESTS_V11[label]["sha256"]:
                raise RuntimeError(f"v11 captured {label} manifest changed")
        self._v11_population_call_index = 0
        self._v11_domain_sign_maps = {}
        self._v11_anchor_results = {}
        self._v11_d44_cache = None
        self._v11_d44_cache_consumed = False
        try:
            plan = super(
                anchor_v10.AntitheticCrossedContractMixinV10, self,
            ).estimate_step_coefficients(
                iteration, seeds, input_text, target_text,
            )
            if (
                self._v11_population_call_index != 2
                or not self._v11_d44_cache_consumed
                or set(self._v11_domain_sign_maps) != {"D43", "D44"}
            ):
                raise RuntimeError("v11 parent-call/cache contract was incomplete")
            captures = {
                "domain_sign_scores": {
                    domain: {
                        sign: [maps[sign][int(seed)] for seed in seeds]
                        for sign in ("plus", "minus")
                    }
                    for domain, maps in self._v11_domain_sign_maps.items()
                },
                "anchor_reference_identities": copy.deepcopy(
                    self._v10_reference_identities
                ),
                "anchor_results": copy.deepcopy(self._v11_anchor_results),
            }
            plan["resident_sign_cross_v11"] = _build_resident_artifact_v11(
                plan, captures,
            )
            validate_resident_cross_v11(plan, recompute_numeric=True)
            self._persist_anchor_plan(plan)
            return plan
        finally:
            self._v11_d44_cache = None
            self._v11_domain_batches = None

    def apply_seed_coefficients(self, plan, target_alpha):
        if float(target_alpha) != 0.0:
            raise ValueError("v11 exact-equivalence diagnostic forbids nonzero alpha")
        validate_resident_cross_v11(plan, recompute_numeric=True)
        return super(
            anchor_v10.AntitheticCrossedContractMixinV10, self,
        ).apply_seed_coefficients(plan, target_alpha)


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v11(captured)
    parent = anchor_v10.load_trainer(captured)
    parent.launch_engines = anchor_v10._clone_with_globals(
        parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v11",
    )

    class ResidentSignTrainerV11(ResidentSignContractMixinV11, parent):
        pass

    return ResidentSignTrainerV11

