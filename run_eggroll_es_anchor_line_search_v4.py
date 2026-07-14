#!/usr/bin/env python3
"""Resident alpha search for frozen-layer dense-reward anchored ES v4."""

import sys
from pathlib import Path

import run_eggroll_es_anchor_line_search as driver_v1
import run_eggroll_es_anchor_line_search_v3 as driver_v3
import train_eggroll_es_specialist_anchor_v4 as anchor_v4


_V1_EXECUTE_LINE_SEARCH = driver_v1.execute_line_search
_ACTIVE_LAYER_PLAN_BUNDLE = None


def validate_effective_anchor_api(module=anchor_v4):
    required = (
        "coefficient_sha256",
        "load_anchor_prose",
        "load_trainer",
        "load_frozen_layer_plan_v4",
    )
    missing = [
        name for name in required if not callable(getattr(module, name, None))
    ]
    if missing:
        raise RuntimeError(
            "v4 anchor adapter is missing callable API members: "
            + ", ".join(missing)
        )
    if anchor_v4.WORKER_EXTENSION != (
        "eggroll_es_worker_v4.FrozenLayerPlanAuditWorkerExtensionV4"
    ):
        raise RuntimeError("resident v4 wrapper selected the wrong worker")
    return required


def set_active_layer_plan_bundle_v4(bundle):
    global _ACTIVE_LAYER_PLAN_BUNDLE
    anchor_v4.set_default_layer_plan_bundle_v4(bundle)
    _ACTIVE_LAYER_PLAN_BUNDLE = bundle


def _active_bundle():
    bundle = _ACTIVE_LAYER_PLAN_BUNDLE
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema")
        != "eggroll-es-frozen-layer-plan-bundle-v4"
    ):
        raise RuntimeError("v4 line search has no validated frozen layer plan")
    return bundle


def build_snapshot(*args, **kwargs):
    """Extend, never replace, v3's complete implementation identity chain."""
    bundle = _active_bundle()
    snapshot = driver_v3.build_snapshot(*args, **kwargs)
    snapshot["schema"] = "eggroll-es-anchor-line-search-snapshot-v4"
    snapshot["implementation"]["distributed_driver_v4"] = (
        anchor_v4.file_sha256(Path(__file__).resolve())
    )
    snapshot["implementation"]["distributed_trainer_v4"] = (
        anchor_v4.file_sha256(Path(anchor_v4.__file__).resolve())
    )
    snapshot["implementation"]["distributed_worker_v4"] = (
        anchor_v4.file_sha256(anchor_v4.ROOT / "eggroll_es_worker_v4.py")
    )
    snapshot["frozen_layer_plan_v4"] = {
        key: bundle[key]
        for key in (
            "path", "file_sha256", "plan_sha256",
            "model_config_path", "model_config_sha256",
        )
    }
    snapshot["dense_gold_reward_v4"] = {
        "config": dict(anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4),
        "reward_config_sha256": (
            anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        ),
    }
    snapshot["distributed_update_v4"] = {
        "engine_count": anchor_v4.REQUIRED_ENGINE_COUNT,
        "tp_per_engine": 1,
        "seed_sharding": "strided_by_inter_engine_rank",
        "collective_dtype": "torch.float32",
        "two_phase_commit": True,
        "final_hash_consensus_required": True,
        "reference_recapture_policy": "once_before_next_population_only",
        "bf16_alpha_semantics": "path_dependent_monotonic_pilot",
        "direct_alpha_confirmation_required": True,
        "layer_plan_sha256": bundle["plan_sha256"],
        "dense_reward_sha256": (
            anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
        ),
    }
    return snapshot


def bind_coefficient_values_v4(journal, plan):
    seeds = plan.get("seeds")
    coefficients = plan.get("coefficients")
    if not isinstance(seeds, list) or not isinstance(coefficients, list):
        raise RuntimeError("v4 seed plan omitted canonical coefficient values")
    if not seeds or len(seeds) != len(coefficients):
        raise RuntimeError("v4 seed/coefficient value counts differ")
    if journal.get("seeds") != seeds:
        raise RuntimeError("v4 journal seeds differ from the distributed plan")
    coefficient_sha = anchor_v4.coefficient_sha256(seeds, coefficients)
    if (
        plan.get("coefficient_sha256") != coefficient_sha
        or journal.get("coefficient_plan", {}).get("coefficient_sha256")
        != coefficient_sha
    ):
        raise RuntimeError("v4 canonical coefficient identity changed")
    journal["coefficient_plan"]["coefficients"] = list(coefficients)
    return coefficient_sha


def _validate_v4_plan_and_applications(journal, plan):
    bundle = _active_bundle()
    reward_sha = anchor_v4.DENSE_GOLD_REWARD_CONFIG_SHA256_V4
    reward = plan.get("dense_gold_reward_v4")
    layer = plan.get("frozen_layer_plan_v4")
    distributed = plan.get("distributed_update_v4")
    boundary_audit = plan.get("population_boundary_audit_v4")
    if (
        not isinstance(reward, dict)
        or reward.get("config") != anchor_v4.DENSE_GOLD_REWARD_CONFIG_V4
        or reward.get("reward_config_sha256") != reward_sha
        or anchor_v4.canonical_sha256(reward["config"]) != reward_sha
        or not isinstance(layer, dict)
        or layer.get("file_sha256") != bundle["file_sha256"]
        or layer.get("plan_sha256") != bundle["plan_sha256"]
        or layer.get("model_config_sha256")
        != bundle["model_config_sha256"]
        or not isinstance(layer.get("runtime_mapping"), dict)
        or layer.get("runtime_mapping_sha256")
        != anchor_v4.canonical_sha256(layer.get("runtime_mapping"))
        or not isinstance(distributed, dict)
        or distributed.get("schema")
        != "eggroll-es-distributed-seed-plan-v4"
        or distributed.get("layer_plan_sha256") != bundle["plan_sha256"]
        or distributed.get("dense_reward_sha256") != reward_sha
        or distributed.get("runtime_mapping") != layer["runtime_mapping"]
        or distributed.get("runtime_mapping_sha256")
        != anchor_v4.canonical_sha256(layer["runtime_mapping"])
        or not isinstance(boundary_audit, dict)
        or boundary_audit.get("schema")
        != "eggroll-es-population-boundary-audit-v4"
        or boundary_audit.get("passed") is not True
        or boundary_audit.get("runtime_mapping")
        != layer["runtime_mapping"]
        or boundary_audit.get("engine_count")
        != anchor_v4.REQUIRED_ENGINE_COUNT
        or boundary_audit.get("audit_sha256")
        != anchor_v4.canonical_sha256({
            key: value for key, value in boundary_audit.items()
            if key != "audit_sha256"
        })
        or distributed.get("population_boundary_audit_sha256")
        != boundary_audit.get("audit_sha256")
    ):
        raise RuntimeError("v4 line search plan provenance differs")
    bindings = layer["runtime_mapping"]
    if set(bindings) != set(anchor_v4.UPDATE_BINDING_KEYS_V4):
        raise RuntimeError("v4 runtime mapping binding fields changed")
    expected_plan_id = anchor_v4.canonical_sha256({
        "schema": "eggroll-es-distributed-plan-id-v4",
        "iteration": int(plan.get("iteration")),
        "coefficient_sha256": plan.get("coefficient_sha256"),
        "reference_generation": distributed.get("reference_generation"),
        "reference_sha256": distributed.get("reference_identity", {}).get(
            "sha256"
        ),
        "layer_plan_sha256": bundle["plan_sha256"],
        "reward_config_sha256": reward_sha,
        "runtime_mapping_sha256": anchor_v4.canonical_sha256(bindings),
        "population_boundary_audit_sha256": boundary_audit[
            "audit_sha256"
        ],
    })
    if distributed.get("plan_id") != expected_plan_id:
        raise RuntimeError("v4 distributed plan identity differs")
    applications = plan.get("applications")
    expected_count = max(0, len(journal["targets"]) - 1)
    if not isinstance(applications, list) or len(applications) != expected_count:
        raise RuntimeError("v4 line search application count changed")
    previous_final_sha = distributed.get("reference_identity", {}).get("sha256")
    previous_alpha = 0.0
    for sequence, application in enumerate(applications, 1):
        manifest = application.get("manifest") if isinstance(
            application, dict,
        ) else None
        target_alpha = journal["targets"][sequence]
        expected_manifest = anchor_v4.update_manifest_v4(
            coefficient_sha256=plan["coefficient_sha256"],
            population_size=len(plan["seeds"]),
            world_size=anchor_v4.REQUIRED_ENGINE_COUNT,
            reference_generation=distributed["reference_generation"],
            plan_id=distributed["plan_id"],
            update_sequence=sequence,
            previous_alpha=previous_alpha,
            target_alpha=target_alpha,
            expected_base_sha256=previous_final_sha,
            bindings=bindings,
        )
        if (
            not isinstance(application, dict)
            or application.get("schema")
            != "eggroll-es-distributed-alpha-application-v4"
            or application.get("update_sequence") != sequence
            or application.get("target_alpha") != target_alpha
            or application.get("alpha_increment")
            != target_alpha - previous_alpha
            or application.get("coefficient_sha256")
            != plan["coefficient_sha256"]
            or application.get("runtime_mapping") != bindings
            or application.get("layer_plan_sha256") != bundle["plan_sha256"]
            or application.get("dense_reward_sha256") != reward_sha
            or not isinstance(manifest, dict)
            or manifest.get("schema")
            != "eggroll-es-layer-restricted-update-manifest-v4"
            or manifest.get("update_sequence") != sequence
            or manifest != expected_manifest
            or anchor_v4.canonical_sha256(manifest)
            != application.get("manifest_sha256")
            or application.get("reference_recaptured") is not False
            or application.get("reference_fresh_for_population") is not False
            or application.get("direct_alpha_confirmation_required") is not True
            or len(application.get("prepared_shards", []))
            != anchor_v4.REQUIRED_ENGINE_COUNT
            or len(application.get("executed_collectives", []))
            != anchor_v4.REQUIRED_ENGINE_COUNT
            or len(application.get("commits", []))
            != anchor_v4.REQUIRED_ENGINE_COUNT
            or len(application.get("post_commit_states", []))
            != anchor_v4.REQUIRED_ENGINE_COUNT
        ):
            raise RuntimeError("v4 line search has an invalid update audit")
        anchor_v4.validate_prepared_shards_v4(
            application["prepared_shards"],
            plan["seeds"],
            plan["coefficients"],
            application["manifest_sha256"],
            manifest["reference_generation"],
            manifest["expected_base_sha256"],
            sequence,
            bindings,
        )
        anchor_v4._validate_bound_v4_reports(
            application["executed_collectives"],
            application["manifest_sha256"],
            bindings,
            executed=True,
        )
        final_identity = anchor_v4.anchor_v3.validate_executed_updates_v3(
            application["executed_collectives"],
            application["manifest_sha256"],
        )
        if final_identity != application.get("final_identity"):
            raise RuntimeError("v4 application final identity differs")
        if any(
            not isinstance(commit, dict)
            or commit.get("schema")
            != "eggroll-es-layer-restricted-update-committed-v4"
            or commit.get("manifest_sha256")
            != application["manifest_sha256"]
            or commit.get("final_sha256") != final_identity["sha256"]
            or any(commit.get(key) != value for key, value in bindings.items())
            for commit in application["commits"]
        ):
            raise RuntimeError("v4 line search commit provenance differs")
        if sorted(commit.get("rank") for commit in application["commits"]) != (
            list(range(anchor_v4.REQUIRED_ENGINE_COUNT))
        ):
            raise RuntimeError("v4 line search commit ranks are incomplete")
        if any(
            not isinstance(state, dict)
            or state.get("current_identity") != final_identity
            or state.get("reference_fresh_for_population") is not False
            or any(state.get(key) != value for key, value in bindings.items())
            for state in application["post_commit_states"]
        ):
            raise RuntimeError("v4 line search post-commit state differs")
        communicator_ranks = sorted(
            state.get("communicator", {}).get("rank")
            for state in application["post_commit_states"]
        )
        if communicator_ranks != list(range(anchor_v4.REQUIRED_ENGINE_COUNT)):
            raise RuntimeError("v4 post-commit communicator ranks are incomplete")
        previous_final_sha = final_identity["sha256"]
        previous_alpha = target_alpha
    return applications, distributed, layer, reward, boundary_audit


def execute_line_search(*args, **kwargs):
    journal = _V1_EXECUTE_LINE_SEARCH(*args, **kwargs)
    trainer = args[0] if args else kwargs["trainer"]
    try:
        plan = trainer._latest_anchor_plan
        audit = plan.get("identity_audit")
        if not isinstance(audit, dict) or audit.get("passed") is not True:
            raise RuntimeError(
                "v4 line search completed without a passed identity audit"
            )
        applications, distributed, layer, reward, boundary_audit = (
            _validate_v4_plan_and_applications(journal, plan)
        )
        bind_coefficient_values_v4(journal, plan)
    except Exception as error:
        journal["status"] = "failed"
        journal["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "phase": "validating_distributed_v4_audit",
        }
        journal.pop("content_sha256_before_self_field", None)
        driver_v1.atomic_write_json(kwargs["journal_path"], journal)
        raise
    journal["schema"] = "eggroll-es-anchor-alpha-line-search-v4"
    journal["policy"]["bf16_alpha_semantics"] = (
        "path_dependent_monotonic_pilot"
    )
    journal["policy"]["direct_alpha_confirmation_required"] = True
    journal["policy"]["frozen_layer_plan_required"] = True
    journal["policy"]["dense_gold_reward_required"] = True
    journal["coefficient_plan"]["identity_audit"] = audit
    journal["coefficient_plan"]["distributed_update_v4"] = distributed
    journal["coefficient_plan"]["frozen_layer_plan_v4"] = layer
    journal["coefficient_plan"]["dense_gold_reward_v4"] = reward
    journal["coefficient_plan"]["population_boundary_audit_v4"] = (
        boundary_audit
    )
    journal["coefficient_plan"]["applications"] = applications
    journal.pop("content_sha256_before_self_field", None)
    journal["content_sha256_before_self_field"] = (
        driver_v1.canonical_sha256(journal)
    )
    driver_v1.atomic_write_json(kwargs["journal_path"], journal)
    return journal


def main(argv=None):
    validate_effective_anchor_api()
    argv = list(sys.argv[1:] if argv is None else argv)
    bundle, remaining = anchor_v4.parse_frozen_layer_plan_cli_v4(argv)
    set_active_layer_plan_bundle_v4(bundle)
    old_argv = sys.argv
    old_anchor = driver_v1.anchor
    old_build = driver_v1.build_snapshot
    old_execute = driver_v1.execute_line_search
    sys.argv = [old_argv[0], *remaining]
    driver_v1.anchor = anchor_v4
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
