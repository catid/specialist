#!/usr/bin/env python3
"""V11b retry with distinct raw-question and templated-prompt identities."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import eggroll_es_worker_v11b as worker_v11b
import train_eggroll_es_specialist_anchor_v11 as anchor_v11


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = (
    "eggroll_es_worker_v11b.ResidentSignAuditWorkerExtensionV11B"
)
RAW_DOMAIN_MANIFESTS_V11B = copy.deepcopy(anchor_v11.DOMAIN_MANIFESTS_V11)
TEMPLATED_DOMAIN_MANIFESTS_V11B = {
    "D43": {
        "seed": 43, "rows": 64,
        "sha256": (
            "54f53464e479fa9dd0c80263f0e424a3d225681c1d8f15554b171f6d5b40c637"
        ),
    },
    "D44": {
        "seed": 44, "rows": 64,
        "sha256": (
            "44cc0ba38c7b2c685a2c44699be9f6dd6313c1391765e13c046812f06e280c23"
        ),
    },
}
_DEFAULT_LAYER_PLAN_BUNDLE = None

canonical_sha256 = anchor_v11.canonical_sha256
file_sha256 = anchor_v11.file_sha256


def validate_frozen_layer_plan_bundle_v11b(bundle):
    return anchor_v11.validate_frozen_layer_plan_bundle_v11(bundle)


def load_frozen_layer_plan_v11b(*args, **kwargs):
    bundle = anchor_v11.load_frozen_layer_plan_v11(*args, **kwargs)
    validate_frozen_layer_plan_bundle_v11b(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v11b(argv):
    bundle, remaining = anchor_v11.parse_frozen_layer_plan_cli_v11(argv)
    validate_frozen_layer_plan_bundle_v11b(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v11b(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v11b(bundle)
    anchor_v11.set_default_layer_plan_bundle_v11(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def validate_templated_domain_batches_v11b(domain_batches):
    if not isinstance(domain_batches, dict) or set(domain_batches) != {
        "D43", "D44",
    }:
        raise RuntimeError("v11b templated domain batch coverage changed")
    for label in ("D43", "D44"):
        prompts, answers = domain_batches[label]
        identity = canonical_sha256({
            "questions": list(prompts), "answers": list(answers),
        })
        if identity != TEMPLATED_DOMAIN_MANIFESTS_V11B[label]["sha256"]:
            raise RuntimeError(
                f"v11b captured templated {label} manifest changed"
            )
    return copy.deepcopy(TEMPLATED_DOMAIN_MANIFESTS_V11B)


class DualManifestResidentSignContractMixinV11B:
    """Bind raw loader rows separately from templated trainer prompts."""

    def estimate_step_coefficients(
        self, iteration, seeds, input_text, target_text,
    ):
        if len(input_text) != 128 or len(target_text) != 128:
            raise RuntimeError("v11b requires the exact combined 128-row batch")
        self._v11_domain_batches = {
            "D43": (list(input_text[:64]), list(target_text[:64])),
            "D44": (list(input_text[64:]), list(target_text[64:])),
        }
        validate_templated_domain_batches_v11b(self._v11_domain_batches)
        self._v11_population_call_index = 0
        self._v11_domain_sign_maps = {}
        self._v11_anchor_results = {}
        self._v11_d44_cache = None
        self._v11_d44_cache_consumed = False
        try:
            plan = super(
                anchor_v11.anchor_v10.AntitheticCrossedContractMixinV10,
                self,
            ).estimate_step_coefficients(
                iteration, seeds, input_text, target_text,
            )
            if (
                self._v11_population_call_index != 2
                or not self._v11_d44_cache_consumed
                or set(self._v11_domain_sign_maps) != {"D43", "D44"}
            ):
                raise RuntimeError("v11b parent-call/cache contract was incomplete")
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
            plan["resident_sign_cross_v11"] = (
                anchor_v11._build_resident_artifact_v11(plan, captures)
            )
            plan["dual_domain_manifest_binding_v11b"] = {
                "schema": "eggroll-es-dual-domain-manifest-binding-v11b",
                "raw_question_answer_manifests": copy.deepcopy(
                    RAW_DOMAIN_MANIFESTS_V11B
                ),
                "templated_prompt_answer_manifests": copy.deepcopy(
                    TEMPLATED_DOMAIN_MANIFESTS_V11B
                ),
                "raw_validated_at": "loader_and_snapshot",
                "templated_validated_at": "trainer_capture_before_generation",
            }
            plan["dual_domain_manifest_binding_v11b"][
                "content_sha256_before_self_field"
            ] = canonical_sha256(plan["dual_domain_manifest_binding_v11b"])
            anchor_v11.validate_resident_cross_v11(
                plan, recompute_numeric=True,
            )
            self._persist_anchor_plan(plan)
            return plan
        finally:
            self._v11_d44_cache = None
            self._v11_domain_batches = None


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v11b(captured)
    parent = anchor_v11.load_trainer(captured)
    parent.launch_engines = anchor_v11.anchor_v10._clone_with_globals(
        parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v11b",
    )

    class DualManifestResidentSignTrainerV11B(
        DualManifestResidentSignContractMixinV11B, parent,
    ):
        pass

    return DualManifestResidentSignTrainerV11B
