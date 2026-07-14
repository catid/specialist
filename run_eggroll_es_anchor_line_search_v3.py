#!/usr/bin/env python3
"""Resident alpha search using the four-engine anchored v3 update path."""

from pathlib import Path

import run_eggroll_es_anchor_line_search as driver_v1
import train_eggroll_es_specialist_anchor_v3 as anchor_v3


_V1_BUILD_SNAPSHOT = driver_v1.build_snapshot
_V1_EXECUTE_LINE_SEARCH = driver_v1.execute_line_search


def validate_effective_anchor_api(module=anchor_v3):
    required = (
        "coefficient_sha256",
        "load_anchor_prose",
        "load_trainer",
    )
    missing = [
        name for name in required if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RuntimeError(
            "distributed anchor adapter is missing callable API members: "
            + ", ".join(missing)
        )
    if anchor_v3.WORKER_EXTENSION != (
        "eggroll_es_worker_v3.DistributedExactAuditWorkerExtensionV3"
    ):
        raise RuntimeError("resident v3 wrapper selected the wrong worker")
    return required


def build_snapshot(*args, **kwargs):
    snapshot = _V1_BUILD_SNAPSHOT(*args, **kwargs)
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v3"
    snapshot["implementation"]["distributed_driver_v3"] = (
        anchor_v3.file_sha256(Path(__file__).resolve())
    )
    snapshot["implementation"]["distributed_trainer_v3"] = (
        anchor_v3.file_sha256(Path(anchor_v3.__file__).resolve())
    )
    snapshot["implementation"]["distributed_worker_v3"] = (
        anchor_v3.file_sha256(anchor_v3.ROOT / "eggroll_es_worker_v3.py")
    )
    snapshot["distributed_update_v3"] = {
        "engine_count": 4,
        "tp_per_engine": 1,
        "seed_sharding": "strided_by_inter_engine_rank",
        "collective_dtype": "torch.float32",
        "two_phase_commit": True,
        "final_hash_consensus_required": True,
        "reference_recapture_policy": "once_before_next_population_only",
        "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
        "direct_alpha_confirmation_required": True,
    }
    return snapshot


def bind_coefficient_values_v3(journal, plan):
    """Bind the v3 journal to the exact coefficients used by each shard."""
    seeds = plan.get("seeds")
    coefficients = plan.get("coefficients")
    if not isinstance(seeds, list) or not isinstance(coefficients, list):
        raise RuntimeError("v3 seed plan omitted canonical coefficient values")
    if len(seeds) == 0 or len(seeds) != len(coefficients):
        raise RuntimeError("v3 seed/coefficient value counts differ")
    if journal.get("seeds") != seeds:
        raise RuntimeError("v3 journal seeds differ from the distributed plan")
    coefficient_sha = anchor_v3.coefficient_sha256(seeds, coefficients)
    if (
        plan.get("coefficient_sha256") != coefficient_sha
        or journal.get("coefficient_plan", {}).get("coefficient_sha256")
        != coefficient_sha
    ):
        raise RuntimeError("v3 canonical coefficient identity changed")
    journal["coefficient_plan"]["coefficients"] = list(coefficients)
    return coefficient_sha


def execute_line_search(*args, **kwargs):
    journal = _V1_EXECUTE_LINE_SEARCH(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    try:
        plan = trainer._latest_anchor_plan
        audit = plan.get("identity_audit")
        if (
            not isinstance(audit, dict) or audit.get("passed") is not True
        ):
            raise RuntimeError(
                "v3 line search completed without a passed identity audit"
            )
        applications = plan.get("applications")
        expected_application_count = max(0, len(journal["targets"]) - 1)
        if (
            not isinstance(applications, list)
            or len(applications) != expected_application_count
        ):
            raise RuntimeError("v3 line search application count changed")
        for sequence, application in enumerate(applications, 1):
            manifest = (
                application.get("manifest")
                if isinstance(application, dict) else None
            )
            if (
                not isinstance(application, dict)
                or application.get("schema")
                != "eggroll-es-distributed-alpha-application-v3"
                or application.get("update_sequence") != sequence
                or not isinstance(manifest, dict)
                or manifest.get("schema")
                != "eggroll-es-distributed-update-manifest-v3"
                or manifest.get("update_sequence") != sequence
                or anchor_v3.canonical_sha256_v3(manifest)
                != application.get("manifest_sha256")
                or application.get("reference_recaptured") is not False
                or application.get("reference_fresh_for_population") is not False
                or len(application.get("prepared_shards", [])) != 4
                or len(application.get("executed_collectives", [])) != 4
                or len(application.get("commits", [])) != 4
                or len(application.get("post_commit_states", [])) != 4
                or any(
                    not isinstance(report, dict)
                    or report.get("manifest_sha256")
                    != application.get("manifest_sha256")
                    or report.get("update_sequence") != sequence
                    or not isinstance(report.get("allocation_preflight"), dict)
                    or report["allocation_preflight"].get("passed") is not True
                    for report in application.get("prepared_shards", [])
                )
            ):
                raise RuntimeError("v3 line search has an invalid update audit")
        bind_coefficient_values_v3(journal, plan)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "phase": "validating_distributed_v3_audit",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v3"
    journal["policy"]["bf16_alpha_semantics"] = (
        "path_dependent_monotonic_pilot"
    )
    journal["policy"]["direct_alpha_confirmation_required"] = True
    journal["coefficient_plan"]["identity_audit"] = audit
    journal["coefficient_plan"]["distributed_update_v3"] = (
        plan["distributed_update_v3"]
    )
    journal["coefficient_plan"]["applications"] = applications
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(journal)
    )
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main():
    validate_effective_anchor_api()
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    driver_v1.anchor = anchor_v3
    driver_v1.build_snapshot = build_snapshot
    driver_v1.execute_line_search = execute_line_search
    try:
        driver_v1.main()
    finally:
        driver_v1.anchor = old_anchor
        driver_v1.build_snapshot = old_build
        driver_v1.execute_line_search = old_execute


if __name__ == "__main__":
    main()
