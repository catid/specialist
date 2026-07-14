#!/usr/bin/env python3
"""Paired-architecture adapter for the exact V13 five-panel diagnostic.

V15A deliberately changes only two things relative to the V13 runtime:

* it admits the frozen V6 middle-late and back layer plans; and
* it substitutes the preregistered fresh perturbation basis.

The V13 panel materialization, scoring, standardization, aggregation,
resident signed-wave loop, exact restoration, and boundary audit are reused
from their original function code objects.  Model-update RPCs remain disabled
on both the controller and worker surfaces.
"""

from __future__ import annotations

import copy
import math
import os
import statistics
import sys
from pathlib import Path
from types import FunctionType

import eggroll_es_back_plan_preregistration_v15a as prereg_v15a
import eggroll_es_worker_v6 as worker_v6
import eggroll_es_worker_v13 as worker_v13
import train_eggroll_es_specialist_anchor_v6 as anchor_v6
import train_eggroll_es_specialist_anchor_v13 as anchor_v13


ROOT = Path(__file__).resolve().parent
PERTURBATION_SEEDS_V15A = list(prereg_v15a.PERTURBATION_SEEDS_V15A)
PERTURBATION_BASIS_SEED_V15A = prereg_v15a.PERTURBATION_BASIS_SEED_V15A
PERTURBATION_BASIS_SHA256_V15A = prereg_v15a.PERTURBATION_BASIS_SHA256_V15A
POPULATION_SIZE_V15A = prereg_v15a.POPULATION_SIZE_V15A
ARM_ORDER_V15A = prereg_v15a.ARM_ORDER_V15A
SIGNS_V15A = anchor_v13.SIGNS_V13
WORKER_EXTENSION = (
    "train_eggroll_es_specialist_anchor_v15a."
    "PairedArchitectureDiagnosticWorkerExtensionV15A"
)
_DEFAULT_LAYER_PLAN_BUNDLE = None

canonical_sha256 = anchor_v13.canonical_sha256
coefficient_sha256 = anchor_v13.coefficient_sha256
anchor_v4 = anchor_v13.anchor_v4
anchor_v10 = anchor_v13.anchor_v10


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


# Reuse the exact V6 installation bytecode and allowlist while retaining all
# four update-disabled methods inherited from the V13 worker.
_install_layer_plan_v15a = _clone_with_globals(
    worker_v6.FrozenEdgeSplitAuditWorkerExtensionV6.install_layer_plan_v4,
    {"validate_frozen_layer_plan_v4": worker_v6.validate_frozen_layer_plan_v6},
    "install_layer_plan_v4",
)


class PairedArchitectureDiagnosticWorkerExtensionV15A(
    worker_v13.TrainPanelDiagnosticWorkerExtensionV13,
):
    """V13 update firewall with the exact four-arm V6 plan installer."""

    install_layer_plan_v4 = _install_layer_plan_v15a


def validate_frozen_layer_plan_bundle_v15a(bundle):
    metadata = anchor_v6.validate_frozen_layer_plan_bundle_v6(bundle)
    matched = []
    for name in ARM_ORDER_V15A:
        spec = prereg_v15a.LAYER_PLANS_V15A[name]
        if (
            bundle.get("plan_sha256") == spec["plan_sha256"]
            and bundle.get("file_sha256") == spec["file_sha256"]
            and Path(bundle.get("path", "")).resolve()
            == Path(spec["path"]).resolve()
            and bundle.get("model_config_sha256")
            == prereg_v15a.MODEL_CONFIG_SHA256_V15A
            and metadata.get("plan") == name
        ):
            matched.append(name)
    if len(matched) != 1:
        raise ValueError("v15a requires an exact preregistered paired layer plan")
    runtime = anchor_v6.FROZEN_RUNTIME_EXPECTATIONS_V6[
        bundle["plan_sha256"]
    ]
    if runtime != prereg_v15a.CAPACITY_V15A:
        raise RuntimeError("v15a paired layer-plan capacity changed")
    return {
        "schema": "eggroll-es-paired-architecture-plan-v15a",
        "arm": matched[0],
        "layers": list(prereg_v15a.LAYER_PLANS_V15A[matched[0]]["layers"]),
        "capacity": copy.deepcopy(runtime),
    }


def set_default_layer_plan_bundle_v15a(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v15a(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


# These clones retain the exact V13 code objects.  Only the immutable globals
# describing the perturbation basis (and the recursively called clone) differ.
analyze_panel_responses_v15a = _clone_with_globals(
    anchor_v13.analyze_panel_responses_v13,
    {"PERTURBATION_SEEDS_V13": PERTURBATION_SEEDS_V15A},
    "analyze_panel_responses_v15a",
)

validate_diagnostic_v15a = _clone_with_globals(
    anchor_v13.validate_diagnostic_v13,
    {
        "PERTURBATION_SEEDS_V13": PERTURBATION_SEEDS_V15A,
        "PERTURBATION_BASIS_SHA256_V13": PERTURBATION_BASIS_SHA256_V15A,
        "PERTURBATION_BASIS_SEED_V13": PERTURBATION_BASIS_SEED_V15A,
        "analyze_panel_responses_v13": analyze_panel_responses_v15a,
    },
    "validate_diagnostic_v15a",
)

_configure_train_panels_v15a = _clone_with_globals(
    anchor_v13.TrainPanelDiagnosticMixinV13.configure_train_panels_v13,
    {"validate_frozen_layer_plan_bundle_v13": validate_frozen_layer_plan_bundle_v15a},
    "configure_train_panels_v15a",
)

_estimate_train_panels_v15a = _clone_with_globals(
    anchor_v13.TrainPanelDiagnosticMixinV13.estimate_train_panels_v13,
    {
        "PERTURBATION_SEEDS_V13": PERTURBATION_SEEDS_V15A,
        "PERTURBATION_BASIS_SHA256_V13": PERTURBATION_BASIS_SHA256_V15A,
        "PERTURBATION_BASIS_SEED_V13": PERTURBATION_BASIS_SEED_V15A,
        "analyze_panel_responses_v13": analyze_panel_responses_v15a,
        "validate_diagnostic_v13": validate_diagnostic_v15a,
    },
    "estimate_train_panels_v15a",
)


def _metric_summary_v15a(values, expected_count):
    values = [float(value) for value in values]
    if (
        len(values) != expected_count
        or not all(
            math.isfinite(value) and -1.0 - 1e-12 <= value <= 1.0 + 1e-12
            for value in values
        )
    ):
        raise RuntimeError("v15a native endpoint vector changed")
    values = [max(-1.0, min(1.0, value)) for value in values]
    return {
        "count": expected_count,
        "median": float(statistics.median(values)),
        "worst": min(values),
    }


def build_integrity_audits_v15a(diagnostic):
    identity = diagnostic.get("identity_audit", {})
    boundary = diagnostic.get("population_boundary_audit_v4", {})
    hardware = diagnostic.get("hardware_coverage")
    responses = diagnostic.get("responses", {})
    dense_hashes = [
        value
        for name in anchor_v13.PANEL_NAMES_V13
        for sign in SIGNS_V15A
        for value in responses.get(name, {}).get(
            "dense_result_sha256", {}
        ).get(sign, [])
    ]
    result = {
        "alpha_zero_no_applications": (
            diagnostic.get("alpha") == 0.0
            and diagnostic.get("applications") == []
        ),
        "model_update_applied_false": (
            diagnostic.get("model_update_applied") is False
        ),
        "exact_reference_checks_passed": (
            anchor_v4.anchor_v3.anchor_v2._all_collective_results(
                identity.get("exact_reference_checks"),
                lambda value: (
                    isinstance(value, dict) and value.get("passed") is True
                ),
            )
        ),
        "pre_post_base_probe_equal": (
            identity.get("passed") is True
            and identity.get("pre_probe") == identity.get("post_probe")
        ),
        "population_boundary_passed": boundary.get("passed") is True,
        "population_boundary_hash_valid": boundary.get("audit_sha256")
        == canonical_sha256({
            key: value for key, value in boundary.items()
            if key != "audit_sha256"
        }),
        "hardware_contract_passed": hardware == {
            "engine_count": 4,
            "tp_per_engine": 1,
            "gpu_ids": [0, 1, 2, 3],
            "population_waves": 8,
            "signed_waves": 16,
            "partial_waves": 0,
            "all_engines_generate_every_signed_wave": True,
        },
        "dense_direction_sign_hash_coverage_passed": (
            len(dense_hashes) == 5 * 2 * 32
            and all(isinstance(value, str) and len(value) == 64 for value in dense_hashes)
        ),
        "fresh_basis_bound": diagnostic.get("perturbation_basis") == {
            "basis_seed": PERTURBATION_BASIS_SEED_V15A,
            "basis_sha256": PERTURBATION_BASIS_SHA256_V15A,
            "seeds": PERTURBATION_SEEDS_V15A,
            "seed_sha256": canonical_sha256(PERTURBATION_SEEDS_V15A),
            "sign_order": list(SIGNS_V15A),
        },
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def validate_integrity_audits_v15a(integrity):
    expected = {
        "alpha_zero_no_applications", "model_update_applied_false",
        "exact_reference_checks_passed", "pre_post_base_probe_equal",
        "population_boundary_passed", "population_boundary_hash_valid",
        "hardware_contract_passed",
        "dense_direction_sign_hash_coverage_passed", "fresh_basis_bound",
        "content_sha256_before_self_field",
    }
    if (
        not isinstance(integrity, dict)
        or set(integrity) != expected
        or any(
            integrity[key] is not True
            for key in expected if key != "content_sha256_before_self_field"
        )
        or integrity["content_sha256_before_self_field"]
        != canonical_sha256({
            key: value for key, value in integrity.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v15a integrity audits are incomplete or failed")
    return integrity


def compact_arm_summary_v15a(arm_name, diagnostic):
    if arm_name not in ARM_ORDER_V15A:
        raise ValueError("v15a arm name changed")
    validate_diagnostic_v15a(diagnostic)
    integrity = build_integrity_audits_v15a(diagnostic)
    validate_integrity_audits_v15a(integrity)
    analysis = diagnostic["analysis"]
    pairwise = list(analysis["optimization_pairwise"].values())
    aggregate = analysis["robust_optimization_aggregate"]
    aggregate_values = aggregate["coefficients"]
    optimization_values = [
        analysis["panel_analysis"][name]["coefficients"]
        for name in anchor_v13.OPTIMIZATION_PANELS_V13
    ]
    screens = list(analysis["train_screen_transfer"].values())
    stability = {
        "optimization_pairwise_cosine": _metric_summary_v15a(
            [item["cosine"] for item in pairwise], 3,
        ),
        "optimization_pairwise_sign_agreement": _metric_summary_v15a(
            [
                item["sign_agreement"]["all_coordinate_fraction"]
                for item in pairwise
            ],
            3,
        ),
        "aggregate_to_optimization_cosine": _metric_summary_v15a(
            [
                anchor_v13._cosine_v13(aggregate_values, values)
                for values in optimization_values
            ],
            3,
        ),
        "aggregate_to_optimization_sign_agreement": _metric_summary_v15a(
            [
                anchor_v13._sign_agreement_v13(
                    aggregate_values, values,
                )["all_coordinate_fraction"]
                for values in optimization_values
            ],
            3,
        ),
        "train_screen_cosine": _metric_summary_v15a(
            [
                item["cosine_to_frozen_optimization_aggregate"]
                for item in screens
            ],
            2,
        ),
        "train_screen_sign_agreement": _metric_summary_v15a(
            [
                item["sign_agreement_to_frozen_optimization_aggregate"][
                    "all_coordinate_fraction"
                ]
                for item in screens
            ],
            2,
        ),
    }
    all_spreads = all(
        not item["standardization"]["zero_spread"]
        for item in analysis["panel_analysis"].values()
    )
    if (
        not all_spreads
        or aggregate["nonzero_coordinate_count"] != 32
        or not math.isfinite(float(aggregate["l2_norm"]))
        or float(aggregate["l2_norm"]) <= 0.0
    ):
        raise RuntimeError("v15a arm aggregate or panel spread failed")
    dense_manifest = {
        name: {
            sign: diagnostic["responses"][name]["dense_result_sha256"][sign]
            for sign in SIGNS_V15A
        }
        for name in anchor_v13.PANEL_NAMES_V13
    }
    result = {
        "schema": "eggroll-es-paired-architecture-arm-summary-v15a",
        "arm": arm_name,
        "plan_sha256": prereg_v15a.LAYER_PLANS_V15A[arm_name]["plan_sha256"],
        "diagnostic_content_sha256": diagnostic[
            "content_sha256_before_self_field"
        ],
        "stability": stability,
        "robust_aggregate": {
            "coefficient_sha256": aggregate["coefficient_sha256"],
            "l2_norm": aggregate["l2_norm"],
            "nonzero_coordinate_count": aggregate[
                "nonzero_coordinate_count"
            ],
        },
        "all_panel_spreads_nonzero": True,
        "integrity_audits": integrity,
        "dense_direction_sign_hash_manifest_sha256": canonical_sha256(
            dense_manifest
        ),
        "persisted_response_vectors": False,
        "persisted_row_content": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def compact_configuration_v15a(arm_name, configuration):
    if (
        arm_name not in ARM_ORDER_V15A
        or not isinstance(configuration, dict)
        or set(configuration) != {
            "layer_plan_install", "reference_identity",
            "panel_bundle_content_sha256",
        }
        or configuration.get("panel_bundle_content_sha256")
        != anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13
        or not isinstance(configuration.get("layer_plan_install"), dict)
        or not isinstance(configuration.get("reference_identity"), dict)
    ):
        raise RuntimeError("v15a arm configuration contract changed")
    result = {
        "schema": "eggroll-es-paired-architecture-configuration-binding-v15a",
        "arm": arm_name,
        "plan_sha256": prereg_v15a.LAYER_PLANS_V15A[arm_name]["plan_sha256"],
        "panel_bundle_content_sha256": anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13,
        "layer_plan_install_content_sha256": canonical_sha256(
            configuration["layer_plan_install"]
        ),
        "reference_identity_content_sha256": canonical_sha256(
            configuration["reference_identity"]
        ),
        "full_configuration_content_sha256": canonical_sha256(configuration),
        "persisted_configuration_payload": False,
    }
    result["content_sha256_before_self_field"] = canonical_sha256(result)
    return result


def build_candidate_v15a(arm_summaries):
    if tuple(arm_summaries) != ARM_ORDER_V15A:
        raise RuntimeError("v15a paired arm order changed")
    for name in ARM_ORDER_V15A:
        summary = arm_summaries[name]
        if (
            summary.get("arm") != name
            or summary.get("plan_sha256")
            != prereg_v15a.LAYER_PLANS_V15A[name]["plan_sha256"]
            or summary.get("all_panel_spreads_nonzero") is not True
        ):
            raise RuntimeError("v15a compact arm summary changed")
        validate_integrity_audits_v15a(summary.get("integrity_audits"))
    panels = anchor_v13.load_panel_bundle_v13()
    candidate = {
        "schema": "eggroll-es-back-plan-stability-summary-v15a",
        "experiment_name": prereg_v15a.EXPERIMENT_NAME_V15A,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_heldout_or_benchmark_used": False,
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V15A,
        "panel_bundle_content_sha256": anchor_v13.PANEL_BUNDLE_CONTENT_SHA256_V13,
        "panel_identities": {
            name: panel["ordered_row_identity_sha256"]
            for name, panel in panels["panels"].items()
        },
        "arm_order": list(ARM_ORDER_V15A),
        "arms": {
            name: {
                "plan_sha256": arm_summaries[name]["plan_sha256"],
                "stability": copy.deepcopy(arm_summaries[name]["stability"]),
                "robust_aggregate": copy.deepcopy(
                    arm_summaries[name]["robust_aggregate"]
                ),
            }
            for name in ARM_ORDER_V15A
        },
        "all_panel_spreads_nonzero": {
            name: arm_summaries[name]["all_panel_spreads_nonzero"]
            for name in ARM_ORDER_V15A
        },
        "all_integrity_audits_passed": True,
    }
    candidate["content_sha256_before_self_field"] = canonical_sha256(candidate)
    prereg_v15a.evaluate_candidate_v15a(candidate)
    return candidate


class PairedArchitectureArmMixinV15A(
    anchor_v13.TrainPanelDiagnosticMixinV13,
):
    configure_train_panels_v15a = _configure_train_panels_v15a
    estimate_train_panels_v15a = _estimate_train_panels_v15a

    def configure_train_panels_v13(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v15a forbids the historical-basis V13 entrypoint")

    def estimate_train_panels_v13(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v15a forbids the historical-basis V13 entrypoint")


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v15a(captured)
    parent = anchor_v6.load_trainer(captured)
    parent.launch_engines = _clone_with_globals(
        parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v15a",
    )

    class PairedArchitectureArmTrainerV15A(
        PairedArchitectureArmMixinV15A, parent,
    ):
        pass

    return PairedArchitectureArmTrainerV15A
