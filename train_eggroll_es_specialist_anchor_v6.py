#!/usr/bin/env python3
"""Frozen four-arm edge-split document-LCB EGGROLL-ES v6 trainer.

V6 changes only the allowed selected-parameter partition.  The complete v5
document-LCB optimization path and v4 exact distributed update implementation
are inherited; cloned coordinator methods replace their private plan allowlist
and worker class without mutating either predecessor module.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import FunctionType

import eggroll_es_worker_v6 as worker_v6
import train_eggroll_es_specialist_anchor_v4 as anchor_v4
import train_eggroll_es_specialist_anchor_v5 as anchor_v5


ROOT = Path(__file__).resolve().parent
REQUIRED_ENGINE_COUNT = anchor_v5.REQUIRED_ENGINE_COUNT
WORKER_EXTENSION = (
    "eggroll_es_worker_v6.FrozenEdgeSplitAuditWorkerExtensionV6"
)
MODEL_CONFIG_SHA256_V6 = (
    "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99"
)
EDGE_SPLIT_PAIR_V6 = {
    "front": "middle_early",
    "middle_early": "front",
    "back": "middle_late",
    "middle_late": "back",
}
FROZEN_EDGE_SPLIT_PLANS_V6 = {
    "af9dcf4e5c932aeb192ee0d195e9c4fee9d4a510467d850d7cf26f6db4c2d823": {
        "plan": "front",
        "path": ROOT / "experiments/layer_plans/front_dense.json",
        "file_sha256": (
            "02e5ce4cc2e20cf6b0910578a3e7982569d323b3458593000f77c624a8db62bf"
        ),
        "layers": [0, 1, 2, 3],
        "paired_control": "middle_early",
    },
    "d72624f2ef55b49b40aa8e52910394f079827a2d848bacc1ee42abb82c47846d": {
        "plan": "middle_early",
        "path": ROOT / "experiments/layer_plans/middle_early_dense_v6.json",
        "file_sha256": (
            "1496184e483071537cd95e10fd8cd051d7bd18c947df1b1e76d72f7d47bafab1"
        ),
        "layers": [16, 17, 18, 19],
        "paired_control": "front",
    },
    "03745c603a6b48898b41afbd4d9121aef276d7e45ca1a3ae14607ec5d1042cb9": {
        "plan": "middle_late",
        "path": ROOT / "experiments/layer_plans/middle_late_dense_v6.json",
        "file_sha256": (
            "d65d702969dcec7a56ca4fcf461d402c44642966191a57c2ef092ec339e3e3df"
        ),
        "layers": [20, 21, 22, 23],
        "paired_control": "back",
    },
    "6da92a4db760676acda1bcbcaec4a925a6dd7b641c250a58a3fe4837d97ac93a": {
        "plan": "back",
        "path": ROOT / "experiments/layer_plans/back_dense.json",
        "file_sha256": (
            "73bfc82ba057908c0071d3c5e190581fecf6147cc398f06a994231f31908187e"
        ),
        "layers": [36, 37, 38, 39],
        "paired_control": "middle_late",
    },
}
FROZEN_RUNTIME_EXPECTATIONS_V6 = {
    plan_sha256: {
        key: worker_v6.FROZEN_LAYER_PLANS_V6[plan_sha256][key]
        for key in (
            "source_unit_count", "runtime_selected_parameter_count",
            "selected_element_count", "selected_byte_count",
        )
    }
    for plan_sha256 in FROZEN_EDGE_SPLIT_PLANS_V6
}
_DEFAULT_LAYER_PLAN_BUNDLE = None


def canonical_sha256(value):
    return anchor_v5.canonical_sha256(value)


def coefficient_sha256(seeds, coefficients):
    return anchor_v5.coefficient_sha256(seeds, coefficients)


def file_sha256(path):
    return anchor_v5.file_sha256(path)


def load_anchor_prose(*args, **kwargs):
    return anchor_v5.load_anchor_prose(*args, **kwargs)


def validate_robust_plan_v5(*args, **kwargs):
    return anchor_v5.validate_robust_plan_v5(*args, **kwargs)


def _plan_spec_v6(plan_sha256):
    spec = FROZEN_EDGE_SPLIT_PLANS_V6.get(plan_sha256)
    if not isinstance(spec, dict):
        raise ValueError("layer plan is outside the frozen v6 edge-split family")
    return spec


def validate_frozen_layer_plan_bundle_v6(bundle):
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema") != "eggroll-es-frozen-layer-plan-bundle-v4"
    ):
        raise ValueError("v6 layer-plan bundle is invalid")
    spec = _plan_spec_v6(bundle.get("plan_sha256"))
    manifest = bundle.get("manifest")
    if (
        bundle.get("file_sha256") != spec["file_sha256"]
        or Path(str(bundle.get("path", ""))).resolve()
        != spec["path"].resolve()
        or bundle.get("model_config_sha256") != MODEL_CONFIG_SHA256_V6
        or not isinstance(manifest, dict)
        or manifest.get("plan") != spec["plan"]
        or manifest.get("layers") != spec["layers"]
        or manifest.get("groups") != ["dense"]
        or manifest.get("num_units") != 35
        or len(manifest.get("units", [])) != 35
    ):
        raise ValueError("v6 layer-plan identity or exact motif changed")
    frozen = worker_v6.FROZEN_LAYER_PLANS_V6[bundle["plan_sha256"]]
    if (
        frozen["runtime_selected_parameter_count"] != 23
        or frozen["selected_element_count"] != 142_999_552
        or frozen["selected_byte_count"] != 285_999_104
    ):
        raise ValueError("v6 capacity match changed")
    raw = Path(bundle["path"]).read_bytes()
    worker_v6.validate_frozen_layer_plan_v6(
        raw, bundle["file_sha256"], bundle["plan_sha256"],
    )
    expected_metadata = {
        "schema": "eggroll-es-edge-split-plan-v6",
        "plan": spec["plan"],
        "layers": list(spec["layers"]),
        "paired_control": spec["paired_control"],
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
    }
    if bundle.get("edge_split_v6") != expected_metadata:
        raise ValueError("v6 edge-split bundle metadata changed")
    return expected_metadata


def load_frozen_layer_plan_v6(
    path, *, expected_file_sha256, expected_plan_sha256,
    expected_model_config_sha256,
):
    spec = _plan_spec_v6(expected_plan_sha256)
    if (
        Path(path).resolve() != spec["path"].resolve()
        or expected_file_sha256 != spec["file_sha256"]
        or expected_model_config_sha256 != MODEL_CONFIG_SHA256_V6
    ):
        raise ValueError("supplied plan pins are not a frozen v6 arm")
    bundle = anchor_v4.load_frozen_layer_plan_v4(
        path,
        expected_file_sha256=expected_file_sha256,
        expected_plan_sha256=expected_plan_sha256,
        expected_model_config_sha256=expected_model_config_sha256,
    )
    bundle["edge_split_v6"] = {
        "schema": "eggroll-es-edge-split-plan-v6",
        "plan": spec["plan"],
        "layers": list(spec["layers"]),
        "paired_control": spec["paired_control"],
        "source_unit_count": 35,
        "runtime_selected_parameter_count": 23,
        "selected_element_count": 142_999_552,
    }
    validate_frozen_layer_plan_bundle_v6(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v6(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--layer-plan-json")
    parser.add_argument("--expected-layer-plan-file-sha256")
    parser.add_argument("--expected-layer-plan-sha256")
    parser.add_argument("--expected-model-config-sha256")
    options, remaining = parser.parse_known_args(list(argv))
    values = (
        options.layer_plan_json, options.expected_layer_plan_file_sha256,
        options.expected_layer_plan_sha256,
        options.expected_model_config_sha256,
    )
    if not all(value is not None for value in values):
        raise ValueError("v6 requires the layer-plan path and all three pins")
    bundle = load_frozen_layer_plan_v6(
        options.layer_plan_json,
        expected_file_sha256=options.expected_layer_plan_file_sha256,
        expected_plan_sha256=options.expected_layer_plan_sha256,
        expected_model_config_sha256=options.expected_model_config_sha256,
    )
    return bundle, remaining


def set_default_layer_plan_bundle_v6(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v6(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def validate_robust_plan_v6(plan, *, recompute_numeric=False):
    binding = anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=recompute_numeric,
    )
    layer = plan.get("frozen_layer_plan_v4")
    plan_sha256 = layer.get("plan_sha256") if isinstance(layer, dict) else None
    spec = _plan_spec_v6(plan_sha256)
    if binding.get("layer_plan_sha256") != plan_sha256:
        raise RuntimeError("v6 robust objective changed its frozen layer plan")
    return {**binding, "edge_split_arm_v6": spec["plan"]}


def _clone_with_globals(function, replacements, name):
    globals_v6 = dict(function.__globals__)
    globals_v6.update(replacements)
    clone = FunctionType(
        function.__code__, globals_v6, name, function.__defaults__,
        function.__closure__,
    )
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__doc__ = function.__doc__
    clone.__module__ = __name__
    clone.__qualname__ = name
    return clone


_configure_anchor_v6 = _clone_with_globals(
    anchor_v4.FrozenLayerDenseRewardMixinV4.configure_anchor,
    {"validate_layer_plan_installations_v4": _clone_with_globals(
        anchor_v4.validate_layer_plan_installations_v4,
        {"FROZEN_RUNTIME_EXPECTATIONS_V4": FROZEN_RUNTIME_EXPECTATIONS_V6},
        "validate_layer_plan_installations_v6",
    )},
    "configure_anchor_v6",
)


class EdgeSplitContractMixinV6:
    def configure_anchor(self, *args, frozen_layer_plan=None, **kwargs):
        bundle = frozen_layer_plan or self._v6_captured_layer_plan
        validate_frozen_layer_plan_bundle_v6(bundle)
        return super().configure_anchor(
            *args, frozen_layer_plan=bundle, **kwargs,
        )

    def estimate_step_coefficients(self, *args, **kwargs):
        plan = super().estimate_step_coefficients(*args, **kwargs)
        validate_robust_plan_v6(plan, recompute_numeric=True)
        return plan

    def apply_seed_coefficients(self, plan, target_alpha):
        validate_robust_plan_v6(plan, recompute_numeric=True)
        return super().apply_seed_coefficients(plan, target_alpha)


def load_trainer(layer_plan_bundle=None):
    """Load v5 objective/v4 updates with an immutable v6 arm and worker."""
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured_bundle = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v6(captured_bundle)
    v4_parent = anchor_v4.load_trainer(captured_bundle)

    def configure_anchor_v6(self, *args, frozen_layer_plan=None, **kwargs):
        return _configure_anchor_v6(
            self, *args,
            frozen_layer_plan=(frozen_layer_plan or captured_bundle),
            **kwargs,
        )

    v4_parent.configure_anchor = configure_anchor_v6
    v4_parent.launch_engines = _clone_with_globals(
        v4_parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v6",
    )

    class EdgeSplitDocumentLCBTrainerV6(
        EdgeSplitContractMixinV6,
        anchor_v5.DocumentLCBAnchoredMixinV5,
        v4_parent,
    ):
        _v6_captured_layer_plan = captured_bundle

    return EdgeSplitDocumentLCBTrainerV6

