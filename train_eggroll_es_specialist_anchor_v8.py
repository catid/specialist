#!/usr/bin/env python3
"""Split-seed population-32 middle-late EGGROLL-ES v8 trainer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import FunctionType

import eggroll_es_worker_v8 as worker_v8
import train_eggroll_es_specialist_anchor_v7 as anchor_v7


ROOT = Path(__file__).resolve().parent
REQUIRED_ENGINE_COUNT = anchor_v7.REQUIRED_ENGINE_COUNT
WORKER_EXTENSION = (
    "eggroll_es_worker_v8.SplitSeedPopulation32AuditWorkerExtensionV8"
)
MODEL_CONFIG_SHA256_V8 = anchor_v7.MODEL_CONFIG_SHA256_V7
FROZEN_STABILITY_PLANS_V8 = {
    key: value
    for key, value in anchor_v7.FROZEN_STABILITY_PLANS_V7.items()
    if value["plan"] == "middle_late"
}
_DEFAULT_LAYER_PLAN_BUNDLE = None


canonical_sha256 = anchor_v7.canonical_sha256
coefficient_sha256 = anchor_v7.coefficient_sha256
file_sha256 = anchor_v7.file_sha256
load_anchor_prose = anchor_v7.load_anchor_prose


def _spec_v8(plan_sha256):
    spec = FROZEN_STABILITY_PLANS_V8.get(plan_sha256)
    if not isinstance(spec, dict):
        raise ValueError("layer plan is outside the frozen v8 middle-late family")
    return spec


def validate_frozen_layer_plan_bundle_v8(bundle):
    metadata_v7 = anchor_v7.validate_frozen_layer_plan_bundle_v7(bundle)
    spec = _spec_v8(bundle.get("plan_sha256"))
    if metadata_v7.get("plan") != "middle_late":
        raise ValueError("v8 layer-plan arm changed")
    raw = Path(bundle["path"]).read_bytes()
    worker_v8.validate_frozen_layer_plan_v8(
        raw, bundle["file_sha256"], bundle["plan_sha256"],
    )
    return {
        "schema": "eggroll-es-split-seed-pop32-plan-v8",
        "plan": spec["plan"], "layers": list(spec["layers"]),
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
        "selected_byte_count": 285_999_104,
    }


def load_frozen_layer_plan_v8(
    path, *, expected_file_sha256, expected_plan_sha256,
    expected_model_config_sha256,
):
    _spec_v8(expected_plan_sha256)
    bundle = anchor_v7.load_frozen_layer_plan_v7(
        path, expected_file_sha256=expected_file_sha256,
        expected_plan_sha256=expected_plan_sha256,
        expected_model_config_sha256=expected_model_config_sha256,
    )
    validate_frozen_layer_plan_bundle_v8(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v8(argv):
    bundle, remaining = anchor_v7.parse_frozen_layer_plan_cli_v7(argv)
    validate_frozen_layer_plan_bundle_v8(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v8(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v8(bundle)
    anchor_v7.set_default_layer_plan_bundle_v7(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def validate_robust_plan_v8(plan, *, recompute_numeric=False):
    binding = anchor_v7.validate_robust_plan_v7(
        plan, recompute_numeric=recompute_numeric,
    )
    layer = plan.get("frozen_layer_plan_v4")
    plan_sha256 = layer.get("plan_sha256") if isinstance(layer, dict) else None
    _spec_v8(plan_sha256)
    if binding.get("direction_stability_arm_v7") != "middle_late":
        raise RuntimeError("v8 robust objective changed its frozen arm")
    return {**binding, "split_seed_arm_v8": "middle_late"}


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


class SplitSeedPopulation32ContractMixinV8:
    def configure_anchor(self, *args, frozen_layer_plan=None, **kwargs):
        bundle = frozen_layer_plan or self._v8_captured_layer_plan
        validate_frozen_layer_plan_bundle_v8(bundle)
        return super().configure_anchor(
            *args, frozen_layer_plan=bundle, **kwargs,
        )

    def estimate_step_coefficients(self, *args, **kwargs):
        plan = super().estimate_step_coefficients(*args, **kwargs)
        validate_robust_plan_v8(plan, recompute_numeric=True)
        return plan

    def apply_seed_coefficients(self, plan, target_alpha):
        if float(target_alpha) != 0.0:
            raise ValueError("v8 split-seed diagnostics forbid nonzero alpha")
        validate_robust_plan_v8(plan, recompute_numeric=True)
        return super().apply_seed_coefficients(plan, target_alpha)


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v8(captured)
    v7_parent = anchor_v7.load_trainer(captured)
    v7_parent.launch_engines = _clone_with_globals(
        v7_parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION}, "launch_engines_v8",
    )

    class SplitSeedPopulation32DocumentLCBTrainerV8(
        SplitSeedPopulation32ContractMixinV8, v7_parent,
    ):
        _v8_captured_layer_plan = captured

    return SplitSeedPopulation32DocumentLCBTrainerV8
