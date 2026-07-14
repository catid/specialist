#!/usr/bin/env python3
"""Resident v5 line search with a document-LCB training anchor."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import eggroll_es_robust_anchor as robust_anchor
import aggregate_eggroll_es_anchor_replications as offline_audit
import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v4 as driver_v4
import train_eggroll_es_specialist_anchor_v5 as anchor_v5


ROOT = Path(__file__).resolve().parent
V5_IMPLEMENTATION_KEYS = (
    "distributed_driver_v5", "distributed_trainer_v5", "robust_anchor_v5",
)


def validate_effective_anchor_api(module=anchor_v5):
    driver_v4.validate_effective_anchor_api(driver_v4.anchor_v4)
    required = (
        "coefficient_sha256",
        "load_anchor_prose",
        "load_trainer",
        "load_frozen_layer_plan_v4",
        "validate_robust_plan_v5",
    )
    missing = [
        name for name in required if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RuntimeError(
            "v5 anchor adapter is missing callable API members: "
            + ", ".join(missing)
        )
    if module.WORKER_EXTENSION != driver_v4.anchor_v4.WORKER_EXTENSION:
        raise RuntimeError("v5 did not retain the audited v4 worker")
    if module.DOCUMENT_LCB_CONFIG_SHA256_V5 != (
        robust_anchor.DOCUMENT_LCB_CONFIG_SHA256
    ):
        raise RuntimeError("v5 document-LCB configuration identity changed")
    return required


def set_active_layer_plan_bundle_v5(bundle):
    driver_v4.set_active_layer_plan_bundle_v4(bundle)
    anchor_v5.set_default_layer_plan_bundle_v5(bundle)


def validate_frozen_execution_cli_v5(argv):
    frozen_v4 = driver_v4.validate_frozen_execution_cli_v4(argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--anchor-prose-jsonl",
        default=str(ROOT / "data/general_prose_anchor_v1.jsonl"),
    )
    parser.add_argument(
        "--anchor-prose-report",
        default=str(ROOT / "data/general_prose_anchor_v1.report.json"),
    )
    parser.add_argument("--anchor-items-per-step", type=int, default=128)
    parser.add_argument("--anchor-max-input-tokens", type=int, default=512)
    parser.add_argument("--min-anchor-cosine", type=float, default=0.8)
    parser.add_argument(
        "--ood-prose-jsonl", default=str(ROOT / "data/ood_prose_v3.jsonl"),
    )
    parser.add_argument("--ood-prose-max-input-tokens", type=int, default=1024)
    options, _ = parser.parse_known_args(list(argv))
    expected_anchor = (ROOT / "data/general_prose_anchor_v1.jsonl").resolve()
    expected_report = (
        ROOT / "data/general_prose_anchor_v1.report.json"
    ).resolve()
    expected_ood = (ROOT / "data/ood_prose_v3.jsonl").resolve()
    if Path(options.anchor_prose_jsonl).resolve() != expected_anchor:
        raise ValueError("v5 optimization requires the frozen training anchor")
    if Path(options.anchor_prose_report).resolve() != expected_report:
        raise ValueError("v5 training-anchor report identity changed")
    if Path(options.ood_prose_jsonl).resolve() != expected_ood:
        raise ValueError("v5 requires the frozen OOD-prose evaluation artifact")
    if (
        options.anchor_items_per_step != 128
        or options.anchor_max_input_tokens != 512
        or options.min_anchor_cosine != 0.8
    ):
        raise ValueError("v5 anchor recipe is frozen at 128/512/cosine-0.8")
    if options.ood_prose_max_input_tokens != 1024:
        raise ValueError("v5 OOD-prose token cap is frozen at 1024")
    return {
        **frozen_v4,
        "anchor_prose_jsonl": str(expected_anchor),
        "anchor_prose_report": str(expected_report),
        "anchor_items_per_step": 128,
        "anchor_max_input_tokens": 512,
        "min_anchor_cosine": 0.8,
        "ood_prose_jsonl": str(expected_ood),
        "ood_prose_max_input_tokens": 1024,
        "document_lcb_config_sha256": (
            anchor_v5.DOCUMENT_LCB_CONFIG_SHA256_V5
        ),
    }


def build_snapshot(*args, **kwargs):
    snapshot = driver_v4.build_snapshot(*args, **kwargs)
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v5"
    snapshot["implementation"]["distributed_driver_v5"] = (
        anchor_v5.file_sha256(Path(__file__).resolve())
    )
    snapshot["implementation"]["distributed_trainer_v5"] = (
        anchor_v5.file_sha256(Path(anchor_v5.__file__).resolve())
    )
    snapshot["implementation"]["robust_anchor_v5"] = (
        anchor_v5.file_sha256(Path(robust_anchor.__file__).resolve())
    )
    implementation_v5 = {
        key: snapshot["implementation"][key]
        for key in V5_IMPLEMENTATION_KEYS
    }
    snapshot["document_lcb_anchor_v5"] = {
        "config": robust_anchor.document_lcb_config(),
        "config_sha256": robust_anchor.DOCUMENT_LCB_CONFIG_SHA256,
        "objective_source": "frozen_train_only_anchor_prose",
        "ood_validation_heldout_as_objective": False,
        "implementation_bundle_sha256": driver_v1.canonical_sha256(
            implementation_v5
        ),
    }
    return snapshot


def _journal_plan_v5(journal):
    coefficient_plan = journal["coefficient_plan"]
    return {
        "seeds": list(journal["seeds"]),
        "coefficients": list(coefficient_plan["coefficients"]),
        "coefficient_sha256": coefficient_plan["coefficient_sha256"],
        "domain_scores": list(coefficient_plan["domain_scores_v5"]),
        "anchor_scores": list(coefficient_plan["anchor_scores_v5"]),
        "projection": dict(coefficient_plan["projection"]),
        "frozen_layer_plan_v4": dict(
            coefficient_plan["frozen_layer_plan_v4"]
        ),
        "identity_audit": coefficient_plan["identity_audit"],
        "document_lcb_anchor_v5": coefficient_plan[
            "document_lcb_anchor_v5"
        ],
        "robust_plan_binding_v5": coefficient_plan[
            "robust_plan_binding_v5"
        ],
    }


def _v4_compatibility_journal_v5(journal):
    """Strip only v5 extensions for the independent persisted-v4 audit."""
    compatible = copy.deepcopy(journal)
    compatible["schema"] = "eggroll-es-anchor-alpha-line-search-v4"
    snapshot = compatible["snapshot"]
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v4"
    snapshot.pop("document_lcb_anchor_v5", None)
    for key in (
        "document_lcb_anchor_required", "optimization_data",
        "ood_validation_heldout_as_objective",
    ):
        compatible["policy"].pop(key, None)
    coefficient_plan = compatible["coefficient_plan"]
    for key in (
        "domain_scores_v5", "anchor_scores_v5", "document_lcb_anchor_v5",
        "robust_plan_binding_v5",
    ):
        coefficient_plan.pop(key, None)
    identity_audit = coefficient_plan.get("identity_audit")
    if isinstance(identity_audit, dict):
        for probe_key in ("pre_probe", "post_probe"):
            probe = identity_audit.get(probe_key)
            if isinstance(probe, dict):
                probe["schema"] = "eggroll-es-train-only-identity-probe-v4"
                probe.pop("document_lcb_anchor_v5", None)
    compatible.pop("content_sha256_before_self_field", None)
    compatible["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(compatible)
    )
    return compatible


def validate_completed_journal_v5(journal):
    """Validate v5-only provenance after inherited v4 validation succeeded."""
    if not isinstance(journal, dict):
        raise RuntimeError("v5 journal is missing")
    if (
        journal.get("schema") != "eggroll-es-anchor-alpha-line-search-v5"
        or journal.get("status") != "complete"
        or journal.get("in_progress") is not None
    ):
        raise RuntimeError("v5 journal is incomplete or has the wrong schema")
    policy = journal.get("policy")
    if (
        not isinstance(policy, dict)
        or policy.get("document_lcb_anchor_required") is not True
        or policy.get("optimization_data") != "train_and_anchor_only"
        or policy.get("ood_validation_heldout_as_objective") is not False
    ):
        raise RuntimeError("v5 journal optimization policy changed")
    if journal.get("content_sha256_before_self_field") != (
        driver_v1.canonical_sha256({
            key: value for key, value in journal.items()
            if key != "content_sha256_before_self_field"
        })
    ):
        raise RuntimeError("v5 journal content hash changed")
    snapshot = journal.get("snapshot")
    robust_snapshot = (
        snapshot.get("document_lcb_anchor_v5")
        if isinstance(snapshot, dict) else None
    )
    implementation = (
        snapshot.get("implementation")
        if isinstance(snapshot, dict) else None
    )
    implementation_v5 = (
        {key: implementation.get(key) for key in V5_IMPLEMENTATION_KEYS}
        if isinstance(implementation, dict) else None
    )
    if (
        not isinstance(snapshot, dict)
        or snapshot.get("schema")
        != "eggroll-es-anchor-line-search-snapshot-v5"
        or not isinstance(robust_snapshot, dict)
        or set(robust_snapshot) != {
            "config", "config_sha256", "objective_source",
            "ood_validation_heldout_as_objective",
            "implementation_bundle_sha256",
        }
        or robust_snapshot.get("config")
        != robust_anchor.document_lcb_config()
        or robust_snapshot.get("config_sha256")
        != robust_anchor.DOCUMENT_LCB_CONFIG_SHA256
        or robust_snapshot.get("objective_source")
        != "frozen_train_only_anchor_prose"
        or robust_snapshot.get("ood_validation_heldout_as_objective") is not False
        or not isinstance(implementation_v5, dict)
        or any(
            not isinstance(value, str)
            or offline_audit.HASH_PATTERN.fullmatch(value) is None
            for value in implementation_v5.values()
        )
        or robust_snapshot.get("implementation_bundle_sha256")
        != driver_v1.canonical_sha256(implementation_v5)
    ):
        raise RuntimeError("v5 snapshot robust-objective provenance changed")
    plan = _journal_plan_v5(journal)
    binding = anchor_v5.validate_robust_plan_v5(
        plan, recompute_numeric=True,
    )
    try:
        v4_audit = offline_audit.validate_journal(
            _v4_compatibility_journal_v5(journal)
        )
    except offline_audit.JournalValidationError as error:
        raise RuntimeError(
            "v5 persisted journal failed its complete inherited v4 audit"
        ) from error
    return {
        "seed": v4_audit["seed"],
        "state_count": len(journal["states"]),
        "coefficient_sha256": plan["coefficient_sha256"],
        "robust_plan_sha256": binding["robust_plan_sha256"],
        "content_sha256": journal["content_sha256_before_self_field"],
        "v4_compatibility_content_sha256": v4_audit["content_sha256"],
    }


def execute_line_search(*args, **kwargs):
    journal = driver_v4.execute_line_search(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    plan = trainer._latest_anchor_plan
    try:
        binding = anchor_v5.validate_robust_plan_v5(plan)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "phase": "validating_document_lcb_v5_audit",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v5"
    journal["policy"]["document_lcb_anchor_required"] = True
    journal["policy"]["optimization_data"] = "train_and_anchor_only"
    journal["policy"]["ood_validation_heldout_as_objective"] = False
    coefficient_plan = journal["coefficient_plan"]
    coefficient_plan["domain_scores_v5"] = list(plan["domain_scores"])
    coefficient_plan["anchor_scores_v5"] = list(plan["anchor_scores"])
    coefficient_plan["document_lcb_anchor_v5"] = plan[
        "document_lcb_anchor_v5"
    ]
    coefficient_plan["robust_plan_binding_v5"] = binding
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(journal)
    )
    try:
        validate_completed_journal_v5(journal)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "phase": "validating_complete_v5_release_audit",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    validate_effective_anchor_api()
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v5.parse_frozen_layer_plan_cli_v4(argv)
    validate_frozen_execution_cli_v5(remaining)
    set_active_layer_plan_bundle_v5(bundle)
    old_argv = sys.argv
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    sys.argv = [old_argv[0], *remaining]
    driver_v1.anchor = anchor_v5
    driver_v1.build_snapshot = build_snapshot
    driver_v1.execute_line_search = execute_line_search
    try:
        driver_v1.main()
    finally:
        sys.argv = old_argv
        driver_v1.anchor = old_anchor
        driver_v1.build_snapshot = old_build
        driver_v1.execute_line_search = old_execute


if __name__ == "__main__":
    main()
