#!/usr/bin/env python3
"""V11c adapter exporting the complete driver-v1 anchor runtime API."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import train_eggroll_es_specialist_anchor_v11b as anchor_v11b


ROOT = Path(__file__).resolve().parent
WORKER_EXTENSION = (
    "eggroll_es_worker_v11c.ResidentSignAuditWorkerExtensionV11C"
)
RAW_DOMAIN_MANIFESTS_V11C = anchor_v11b.RAW_DOMAIN_MANIFESTS_V11B
TEMPLATED_DOMAIN_MANIFESTS_V11C = (
    anchor_v11b.TEMPLATED_DOMAIN_MANIFESTS_V11B
)
_DEFAULT_LAYER_PLAN_BUNDLE = None

# Complete direct access surface of run_eggroll_es_anchor_line_search.main and
# its execute_line_search helper when this module is substituted as `anchor`.
canonical_sha256 = anchor_v11b.canonical_sha256
coefficient_sha256 = anchor_v11b.anchor_v11.coefficient_sha256
file_sha256 = anchor_v11b.file_sha256
load_anchor_prose = anchor_v11b.anchor_v11.load_anchor_prose


def validate_frozen_layer_plan_bundle_v11c(bundle):
    return anchor_v11b.validate_frozen_layer_plan_bundle_v11b(bundle)


def load_frozen_layer_plan_v11c(*args, **kwargs):
    bundle = anchor_v11b.load_frozen_layer_plan_v11b(*args, **kwargs)
    validate_frozen_layer_plan_bundle_v11c(bundle)
    return bundle


def parse_frozen_layer_plan_cli_v11c(argv):
    bundle, remaining = anchor_v11b.parse_frozen_layer_plan_cli_v11b(argv)
    validate_frozen_layer_plan_bundle_v11c(bundle)
    return bundle, remaining


def set_default_layer_plan_bundle_v11c(bundle):
    global _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v11c(bundle)
    anchor_v11b.set_default_layer_plan_bundle_v11b(bundle)
    _DEFAULT_LAYER_PLAN_BUNDLE = bundle


def validate_templated_domain_batches_v11c(domain_batches):
    return anchor_v11b.validate_templated_domain_batches_v11b(domain_batches)


def load_trainer(layer_plan_bundle=None):
    pythonpath = [str(ROOT)]
    if os.environ.get("PYTHONPATH"):
        pythonpath.append(os.environ["PYTHONPATH"])
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    captured = layer_plan_bundle or _DEFAULT_LAYER_PLAN_BUNDLE
    validate_frozen_layer_plan_bundle_v11c(captured)
    parent = anchor_v11b.load_trainer(captured)
    parent.launch_engines = anchor_v11b.anchor_v11.anchor_v10._clone_with_globals(
        parent.launch_engines,
        {"WORKER_EXTENSION": WORKER_EXTENSION},
        "launch_engines_v11c",
    )

    class CompleteRuntimeAPIResidentSignTrainerV11C(parent):
        pass

    return CompleteRuntimeAPIResidentSignTrainerV11C
