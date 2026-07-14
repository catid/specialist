#!/usr/bin/env python3
"""Alpha-zero, two-arm direction-stability EGGROLL-ES v7 trainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import FunctionType

import eggroll_es_worker_v7 as worker_v7
import train_eggroll_es_specialist_anchor_v6 as anchor_v6


ROOT = Path(__file__).resolve().parent
REQUIRED_ENGINE_COUNT = anchor_v6.REQUIRED_ENGINE_COUNT
WORKER_EXTENSION = (
    "eggroll_es_worker_v7.DirectionStabilityAuditWorkerExtensionV7"
)
MODEL_CONFIG_SHA256_V7 = anchor_v6.MODEL_CONFIG_SHA256_V6
FROZEN_STABILITY_PLANS_V7 = {
    key: value
    for key, value in anchor_v6.FROZEN_EDGE_SPLIT_PLANS_V6.items()
    if value["plan"] in {"front", "middle_late"}
}
_DEFAULT_LAYER_PLAN_BUNDLE = None


canonical_sha256 = anchor_v6.canonical_sha256
coefficient_sha256 = anchor_v6.coefficient_sha256
file_sha256 = anchor_v6.file_sha256
load_anchor_prose = anchor_v6.load_anchor_prose


def _spec_v7(plan_sha256):
    spec = FROZEN_STABILITY_PLANS_V7.get(plan_sha256)
    if not isinstance(spec, dict):
        raise ValueError("layer plan is outside the frozen v7 stability family")
    return spec


def validate_frozen_layer_plan_bundle_v7(bundle):
    metadata_v6 = anchor_v6.validate_frozen_layer_plan_bundle_v6(bundle)
    spec = _spec_v7(bundle.get("plan_sha256"))
    if metadata_v6.get("plan") != spec["plan"]:
        raise ValueError("v7 layer-plan arm changed")
    raw = Path(bundle["path"]).read_bytes()
    worker_v7.validate_frozen_layer_plan_v7(
        raw, bundle["file_sha256"], bundle["plan_sha256"],
    )
    return {
        "schema": "eggroll-es-direction-stability-plan-v7",
        "plan": spec["plan"],
        "layers": list(spec["layers"]),
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
    }


def load_frozen_layer_plan_v7(
    path, *, expected_file_sha256, expected_plan_sha256,
    expected_model_config_sha256,
):
    _spec_v7(expected_plan_sha256)
    bundle = anchor_v6.load_frozen_layer_plan_v6(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_plan_sha256=expected_plan_sha256,
        expected_model_config_sha256=expected_model_config_sha256,
    )
    validate_frozen_layer_plan_bundle_v7(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v7(argv):
    bundle, remaining = anchor_v6.parse_frozen_layer_plan_cli_v6(argv)
    validate_frozen_layer_plan_bundle_v7(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v7(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v7(bundle)
    anchor_v6.set_default_layer_plan_bundle_v6(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def validate_robust_plan_v7(plan, *, recompute_numeric=False):
    binding = anchor_v6.validate_robust_plan_v6(
        plan, recompute_numeric=recompute_numeric,
    )
    layer = plan.get("frozen_layer_plan_v4")
    plan_sha256 = layer.get("plan_sha256") if isinstance(layer, dict) else None
    spec = _spec_v7(plan_sha256)
    if binding.get("edge_split_arm_v6") != spec["plan"]:
        raise RuntimeError("v7 robust objective changed its frozen arm")
    return {**binding, "direction_stability_arm_v7": spec["plan"]}


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


class DirectionStabilityContractMixinV7:
    def configure_anchor(self, *args, frozen_layer_plan=None, **kwargs):
        bundle = frozen_layer_plan or self._v7_captured_layer_plan
        validate_frozen_layer_plan_bundle_v7(bundle)
        return super().configure_anchor(
            *args, frozen_layer_plan=bundle, **kwargs,
        )

    def estimate_step_coefficients(self, *args, **kwargs):
        plan = super().estimate_step_coefficients(*args, **kwargs)
        validate_robust_plan_v7(plan, recompute_numeric=True)
        return plan

    def apply_seed_coefficients(self, plan, target_alpha):
        if float(target_alpha) != 0.0:
            raise ValueError("v7 stability runs forbid nonzero alpha")
        validate_robust_plan_v7(plan, recompute_numeric=True)
        return super().apply_seed_coefficients(plan, target_alpha)


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v7(captured)
    v6_parent = anchor_v6.load_trainer(captured)
    v6_parent.launch_engines = _clone_with_globals(
        v6_parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v7",
    )

    class DirectionStabilityDocumentLCBTrainerV7(
        DirectionStabilityContractMixinV7, v6_parent,
    ):
        _v7_captured_layer_plan = captured

    return DirectionStabilityDocumentLCBTrainerV7
