#!/usr/bin/env python3
"""V15B fresh-basis adapter for the exact V13 five-panel diagnostic."""

from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import eggroll_es_back_plan_confirmation_preregistration_v15b as prereg_v15b
import train_eggroll_es_specialist_anchor_v15a as anchor_v15a


ROOT = Path(__file__).resolve().parent
anchor_v13 = anchor_v15a.anchor_v13
anchor_v6 = anchor_v15a.anchor_v6
canonical_sha256 = anchor_v15a.canonical_sha256
PERTURBATION_SEEDS_V15B = list(prereg_v15b.PERTURBATION_SEEDS_V15B)
PERTURBATION_BASIS_SEED_V15B = prereg_v15b.PERTURBATION_BASIS_SEED_V15B
PERTURBATION_BASIS_SHA256_V15B = prereg_v15b.PERTURBATION_BASIS_SHA256_V15B
POPULATION_SIZE_V15B = prereg_v15b.POPULATION_SIZE_V15B
ARM_ORDER_V15B = prereg_v15b.ARM_ORDER_V15B
SIGNS_V15B = anchor_v13.SIGNS_V13
WORKER_EXTENSION = (
    "train_eggroll_es_specialist_anchor_v15b."
    "PairedConfirmationWorkerExtensionV15B"
)
_DEFAULT_LAYER_PLAN_BUNDLE = None


def _frozen_preregistration_v15b():
    return prereg_v15b.build_preregistration_v15b()


class PairedConfirmationWorkerExtensionV15B(
    anchor_v15a.PairedArchitectureDiagnosticWorkerExtensionV15A,
):
    """The V15A/V13 update firewall with the exact V6 plan installer."""


def validate_frozen_layer_plan_bundle_v15b(bundle):
    metadata = anchor_v15a.validate_frozen_layer_plan_bundle_v15a(bundle)
    plans = _frozen_preregistration_v15b()["paired_architecture"]["arms"]
    name = metadata["arm"]
    spec = plans[name]
    if (
        name not in ARM_ORDER_V15B
        or bundle.get("plan_sha256") != spec["plan_sha256"]
        or bundle.get("file_sha256") != spec["file_sha256"]
        or Path(bundle.get("path", "")).resolve() != Path(spec["path"]).resolve()
        or metadata.get("capacity") != spec["capacity"]
    ):
        raise ValueError("v15b requires an exact preregistered paired layer plan")
    return {
        "schema": "eggroll-es-paired-confirmation-plan-v15b",
        "arm": name,
        "layers": list(spec["layers"]),
        "capacity": copy.deepcopy(spec["capacity"]),
    }


def set_default_layer_plan_bundle_v15b(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v15b(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


analyze_panel_responses_v15b = anchor_v15a._clone_with_globals(
    anchor_v15a.analyze_panel_responses_v15a,
    {"PERTURBATION_SEEDS_V13": PERTURBATION_SEEDS_V15B},
    "analyze_panel_responses_v15b",
)

validate_diagnostic_v15b = anchor_v15a._clone_with_globals(
    anchor_v15a.validate_diagnostic_v15a,
    {
        "PERTURBATION_SEEDS_V13": PERTURBATION_SEEDS_V15B,
        "PERTURBATION_BASIS_SHA256_V13": PERTURBATION_BASIS_SHA256_V15B,
        "PERTURBATION_BASIS_SEED_V13": PERTURBATION_BASIS_SEED_V15B,
        "analyze_panel_responses_v13": analyze_panel_responses_v15b,
    },
    "validate_diagnostic_v15b",
)

_configure_train_panels_v15b = anchor_v15a._clone_with_globals(
    anchor_v15a._configure_train_panels_v15a,
    {"validate_frozen_layer_plan_bundle_v13": validate_frozen_layer_plan_bundle_v15b},
    "configure_train_panels_v15b",
)

_estimate_train_panels_v15b = anchor_v15a._clone_with_globals(
    anchor_v15a._estimate_train_panels_v15a,
    {
        "PERTURBATION_SEEDS_V13": PERTURBATION_SEEDS_V15B,
        "PERTURBATION_BASIS_SHA256_V13": PERTURBATION_BASIS_SHA256_V15B,
        "PERTURBATION_BASIS_SEED_V13": PERTURBATION_BASIS_SEED_V15B,
        "analyze_panel_responses_v13": analyze_panel_responses_v15b,
        "validate_diagnostic_v13": validate_diagnostic_v15b,
    },
    "estimate_train_panels_v15b",
)


def build_integrity_audits_v15b(diagnostic):
    cloned = anchor_v15a._clone_with_globals(
        anchor_v15a.build_integrity_audits_v15a,
        {
            "PERTURBATION_SEEDS_V15A": PERTURBATION_SEEDS_V15B,
            "PERTURBATION_BASIS_SHA256_V15A": PERTURBATION_BASIS_SHA256_V15B,
            "PERTURBATION_BASIS_SEED_V15A": PERTURBATION_BASIS_SEED_V15B,
            "SIGNS_V15A": SIGNS_V15B,
        },
        "build_integrity_audits_v15b_exact",
    )
    return cloned(diagnostic)


def validate_integrity_audits_v15b(integrity):
    return anchor_v15a.validate_integrity_audits_v15a(integrity)


def compact_arm_summary_v15b(arm_name, diagnostic):
    plans = _frozen_preregistration_v15b()["paired_architecture"]["arms"]

    class PreregistrationShim:
        LAYER_PLANS_V15A = plans

    cloned = anchor_v15a._clone_with_globals(
        anchor_v15a.compact_arm_summary_v15a,
        {
            "ARM_ORDER_V15A": ARM_ORDER_V15B,
            "validate_diagnostic_v15a": validate_diagnostic_v15b,
            "build_integrity_audits_v15a": build_integrity_audits_v15b,
            "validate_integrity_audits_v15a": validate_integrity_audits_v15b,
            "prereg_v15a": PreregistrationShim,
            "SIGNS_V15A": SIGNS_V15B,
        },
        "compact_arm_summary_v15b_exact",
    )
    result = cloned(arm_name, diagnostic)
    result["schema"] = "eggroll-es-paired-confirmation-arm-summary-v15b"
    result["content_sha256_before_self_field"] = canonical_sha256({
        key: item for key, item in result.items()
        if key != "content_sha256_before_self_field"
    })
    return result


def compact_configuration_v15b(arm_name, configuration):
    plans = _frozen_preregistration_v15b()["paired_architecture"]["arms"]

    class PreregistrationShim:
        LAYER_PLANS_V15A = plans

    cloned = anchor_v15a._clone_with_globals(
        anchor_v15a.compact_configuration_v15a,
        {"ARM_ORDER_V15A": ARM_ORDER_V15B, "prereg_v15a": PreregistrationShim},
        "compact_configuration_v15b_exact",
    )
    result = cloned(arm_name, configuration)
    result["schema"] = "eggroll-es-paired-confirmation-configuration-v15b"
    result["content_sha256_before_self_field"] = canonical_sha256({
        key: item for key, item in result.items()
        if key != "content_sha256_before_self_field"
    })
    return result


def build_candidate_v15b(arm_summaries):
    if tuple(arm_summaries) != ARM_ORDER_V15B:
        raise RuntimeError("v15b paired arm order changed")
    frozen = _frozen_preregistration_v15b()
    plans = frozen["paired_architecture"]["arms"]
    for name in ARM_ORDER_V15B:
        summary = arm_summaries[name]
        if (
            summary.get("arm") != name
            or summary.get("plan_sha256") != plans[name]["plan_sha256"]
            or summary.get("all_panel_spreads_nonzero") is not True
        ):
            raise RuntimeError("v15b compact arm summary changed")
        validate_integrity_audits_v15b(summary.get("integrity_audits"))
    candidate = {
        "schema": "eggroll-es-back-plan-confirmation-summary-v15b",
        "experiment_name": prereg_v15b.EXPERIMENT_NAME_V15B,
        "alpha": 0.0,
        "model_update_applied": False,
        "validation_ood_heldout_or_benchmark_used": False,
        "perturbation_basis_sha256": PERTURBATION_BASIS_SHA256_V15B,
        "panel_bundle_content_sha256": frozen["estimator"][
            "panel_bundle_content_sha256"
        ],
        "panel_identities": copy.deepcopy(
            frozen["estimator"]["ordered_panel_identities"]
        ),
        "arm_order": list(ARM_ORDER_V15B),
        "arms": {
            name: {
                "plan_sha256": arm_summaries[name]["plan_sha256"],
                "stability": copy.deepcopy(arm_summaries[name]["stability"]),
                "robust_aggregate": copy.deepcopy(
                    arm_summaries[name]["robust_aggregate"]
                ),
            }
            for name in ARM_ORDER_V15B
        },
        "all_panel_spreads_nonzero": {
            name: arm_summaries[name]["all_panel_spreads_nonzero"]
            for name in ARM_ORDER_V15B
        },
        "all_integrity_audits_passed": True,
    }
    candidate["content_sha256_before_self_field"] = canonical_sha256(candidate)
    prereg_v15b.evaluate_candidate_v15b(candidate)
    return candidate


class PairedConfirmationArmMixinV15B(
    anchor_v13.TrainPanelDiagnosticMixinV13,
):
    configure_train_panels_v15b = _configure_train_panels_v15b
    estimate_train_panels_v15b = _estimate_train_panels_v15b

    def configure_train_panels_v13(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v15b forbids the historical-basis V13 entrypoint")

    def estimate_train_panels_v13(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v15b forbids the historical-basis V13 entrypoint")

    def configure_train_panels_v15a(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v15b forbids the V15A-basis entrypoint")

    def estimate_train_panels_v15a(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("v15b forbids the V15A-basis entrypoint")


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v15b(captured)
    parent = anchor_v6.load_trainer(captured)
    parent.launch_engines = anchor_v15a._clone_with_globals(
        parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v15b",
    )

    class PairedConfirmationArmTrainerV15B(
        PairedConfirmationArmMixinV15B, parent,
    ):
        pass

    return PairedConfirmationArmTrainerV15B
