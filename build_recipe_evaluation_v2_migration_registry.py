#!/usr/bin/env python3
"""Build the additive V2 migration registry without rewriting history."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import audit_v73c_systems_only_import_graph_v2 as v73c_closure
import build_quarantine_boundary_registry_v3 as quarantine_boundary
import recipe_evaluation_contract_v2 as v2


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_contract_v2_migration.json"
).resolve()
SCHEMA = "specialist-recipe-evaluation-contract-v2-migration-v1"
V1_IMPLEMENTATION_PATH = "recipe_evaluation_contract_v1.py"
V1_PRE_TOMBSTONE_FILE_SHA256 = (
    "e3f57e9290298e2510118e8c9f10c835618fa12206197462e7ae5a0b7ab68c25"
)
V1_PRE_TOMBSTONE_GIT_BLOB = "89926227ceb5655c65a8eefb6f7773ec2b32136b"
V1_PRE_TOMBSTONE_COMMIT = "f445ab33f0a06c0659caee1755267cbbb1f48a6c"
V1_PRE_TOMBSTONE_DEFINITION_COMMIT = (
    "888d8da2262671db2282527daa0fc174e6e2a80d"
)

HISTORICAL_ARTIFACTS = (
    "experiments/eggroll_es_hpo/preregistrations/recipe_evaluation_compute_contract_v1.json",
    "experiments/eggroll_es_hpo/preregistrations/cpt_sft_es_curriculum_ablation_v1.json",
    "experiments/eggroll_es_hpo/preregistrations/prepared_once_only_sft_v42i_sealed_holdout_eval_v46a.json",
    "experiments/eggroll_es_hpo/preregistrations/once_only_sft_v42i_sealed_holdout_eval_v46d.json",
    "experiments/eval_reports/v46d_once_only_sft_v42i_terminal_receipt.json",
    "experiments/eggroll_es_hpo/datasets/recipe_sampling_ablation_v1.json",
    "experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66.json",
    "experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66b.json",
    "experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66c.json",
    "experiments/eggroll_es_hpo/preregistrations/lora_es_mirrored_calibration_v66d.json",
    "experiments/eggroll_es_hpo/preregistrations/lora_es_v71_v72_live_calibration_v73.json",
    "experiments/eggroll_es_hpo/preregistrations/lora_es_v71_v72_same_live_calibration_v73b.json",
    "experiments/eggroll_es_hpo/preregistrations/multiobjective_reward_ood_trust_region_v67.json",
    "experiments/eggroll_es_hpo/preregistrations/greedy_seeded_stochastic_robustness_v68.json",
    "experiments/eggroll_es_hpo/preregistrations/qwen36_moe_lora_targeting_v69.json",
    "experiments/eggroll_es_hpo/preregistrations/qwen36_front_tail_lora_topology_v70.json",
    "experiments/eggroll_es_hpo/preregistrations/fp32_es_optimizer_module_sigma_ablation_v1.json",
    "experiments/eggroll_es_hpo/fp32_es_optimizer_module_sigma_v1_cpu_evidence_20260717.md",
    "experiments/eggroll_es_hpo/preregistrations/reward_shaping_ablation_v1.json",
    "experiments/eggroll_es_hpo/reward_shaping_ablation_v1_cpu_evidence_20260717.md",
    "experiments/eggroll_es_hpo/preregistrations/qwen36_collective_compression_v82.json",
    "experiments/eggroll_es_hpo/preregistrations/qwen36_lora_rank_surface_pareto_v1.json",
    "experiments/eggroll_es_hpo/qwen36_lora_rank_surface_pareto_v1_cpu_evidence_20260717.md",
    "experiments/eggroll_es_hpo/decisions/qwen36_production_layout_provisional_v75.json",
    "experiments/eggroll_es_hpo/qwen36_production_layout_decision_v75_20260717.md",
    (
        "experiments/eggroll_es_hpo/preregistrations/"
        "recipe_evaluation_compute_contract_v2_superseded_prefix_scan_20260717.json"
    ),
)

FAIL_CLOSED_ENTRYPOINTS = (
    ("recipe_evaluation_contract_v1.py", "all public V1 contract APIs"),
    ("build_eval_v3.py", "exact quarantined V1 output targets"),
    (
        "build_curated_qa.py",
        "explicit eval choice; exact legacy paths and implicit defaults denied",
    ),
    (
        "build_curriculum_ablation_preregistration_v1.py",
        "build_preregistration_v1",
    ),
    ("build_once_only_holdout_preregistration_v46a.py", "build"),
    ("build_once_only_holdout_preregistration_v46d.py", "build"),
    ("build_v46d_terminal_holdout_receipt.py", "build"),
    ("run_once_only_holdout_eval_v46a.py", "main"),
    ("run_once_only_holdout_eval_v46d.py", "main"),
    ("train_sampling_ablation_v1.py", "build/validate/materialize"),
    (
        "eggroll_es_multiobjective_trust_region_v67.py",
        "build/validate/evaluate/promote",
    ),
    ("eggroll_es_decode_robustness_v68.py", "build/validate/select"),
    ("eggroll_es_moe_targeting_v69.py", "build/validate/select"),
    ("eggroll_es_front_tail_topology_v70.py", "build/validate/select"),
    (
        "build_fp32_es_optimizer_sigma_preregistration_v1.py",
        "build_preregistration_v1",
    ),
    (
        "build_reward_shaping_ablation_preregistration_v1.py",
        "build_preregistration_v1",
    ),
    (
        "build_qwen36_collective_compression_preregistration_v82.py",
        "build/validate preregistration",
    ),
    (
        "build_qwen36_lora_rank_surface_preregistration_v1.py",
        "upstream/build/validate preregistration",
    ),
    (
        "build_qwen36_production_layout_decision_v75.py",
        "build/validate/promote decision",
    ),
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("migration input is not a JSON object")
    return value


def build_registry() -> dict:
    contract = _read_json(v2.CONTRACT)
    v2.validate_contract(contract)
    closure = _read_json(v73c_closure.OUTPUT)
    v73c_closure.validate_audit(closure)
    boundary = _read_json(quarantine_boundary.OUTPUT)
    quarantine_boundary.validate_registry(boundary)
    reservation = contract["roles"]["protected_terminal"][
        "future_adaptation_reservation"
    ]
    historical = []
    for relative_path in HISTORICAL_ARTIFACTS:
        path = (ROOT / relative_path).resolve()
        if not path.is_file() or not path.is_relative_to(ROOT):
            raise RuntimeError("historical migration artifact is absent")
        historical.append({
            "path": relative_path,
            "file_sha256": v2.file_sha256(path),
            "status": "immutable_historical_nonpromotable",
        })

    registry = {
        "schema": SCHEMA,
        "status": "v2_provisional_prose_only_all_incidents_quarantined_history_preserved",
        "created_at_utc": "2026-07-17T00:00:00+00:00",
        "v2_contract": {
            "path": str(v2.CONTRACT.relative_to(ROOT)),
            "file_sha256": v2.file_sha256(v2.CONTRACT),
            "content_sha256": contract["content_sha256_before_self_field"],
            "status": contract["status"],
            "qa_hpo_or_general_quality_promotion_authorized": False,
        },
        "immutable_incident_receipts": [
            {
                "incident_role": "protected_v1_eval_v3",
                "path": str(v2.INCIDENT.relative_to(ROOT)),
                "file_sha256": v2.INCIDENT_FILE_SHA256,
                "content_sha256": v2.INCIDENT_CONTENT_SHA256,
                "status": "immutable_content_free_quarantine_receipt",
            },
            {
                "incident_role": "legacy_eval_collision_family",
                "path": str(v2.LEGACY_EVAL_COLLISION_INCIDENT.relative_to(ROOT)),
                "file_sha256": v2.LEGACY_EVAL_COLLISION_INCIDENT_FILE_SHA256,
                "content_sha256": (
                    v2.LEGACY_EVAL_COLLISION_INCIDENT_CONTENT_SHA256
                ),
                "status": "immutable_content_free_quarantine_receipt",
            },
            {
                "incident_role": "recursive_prefix_lookup",
                "path": str(v2.RECURSIVE_LOOKUP_INCIDENT.relative_to(ROOT)),
                "file_sha256": v2.RECURSIVE_LOOKUP_INCIDENT_FILE_SHA256,
                "content_sha256": v2.RECURSIVE_LOOKUP_INCIDENT_CONTENT_SHA256,
                "status": "immutable_content_free_quarantine_receipt",
            },
        ],
        "superseded_v2_revision": {
            "path": str(v2.SUPERSEDED_V2_CONTRACT.relative_to(ROOT)),
            "file_sha256": v2.SUPERSEDED_V2_CONTRACT_FILE_SHA256,
            "content_sha256": v2.SUPERSEDED_V2_CONTRACT_CONTENT_SHA256,
            "status": "immutable_superseded_nonpromotable",
            "historical_terminal_sources_remain_reserved": 12,
        },
        "v2_resealed_source_reservation": {
            "terminal_source_count": reservation["terminal_source_count"],
            "dev_source_count": reservation["dev_source_count"],
            "reserved_source_count": reservation["reserved_source_count"],
            "historical_selected_source_count": reservation[
                "historical_selected_source_count"
            ],
            "historical_selected_sources_remain_reserved": reservation[
                "historical_selected_sources_remain_reserved"
            ],
            "reservation_content_sha256": reservation[
                "content_sha256_before_self_field"
            ],
            "roles": reservation["roles"],
            "terminal_eligible_unselected_sources": 0,
            "enforcement_api": "assert_future_dataset_excludes_reserved",
            "required_before_any_future_dataset_write": True,
        },
        "content_free_quarantine_boundary": {
            "path": str(quarantine_boundary.OUTPUT.relative_to(ROOT)),
            "file_sha256": v2.file_sha256(quarantine_boundary.OUTPUT),
            "content_sha256": boundary["content_sha256_before_self_field"],
            "exact_path_identity_count": boundary["exact_path_identity_count"],
            "prefix_identity_count": boundary["prefix_identity_count"],
            "plaintext_boundary_paths_persisted": False,
            "deny_before_resolution_stat_hash_or_open": True,
        },
        "v73c_stage_a_systems_only_closure": {
            "path": str(v73c_closure.OUTPUT.relative_to(ROOT)),
            "file_sha256": v2.file_sha256(v73c_closure.OUTPUT),
            "content_sha256": closure["content_sha256_before_self_field"],
            "schema": closure["schema"],
            "systems_trace_only": True,
            "quality_hpo_or_promotion_authorized": False,
            "distinct_postrun_boundary_denial_receipt_required": True,
        },
        "v1_implementation_tombstone": {
            "path": V1_IMPLEMENTATION_PATH,
            "status": "permanent_fail_closed_tombstone",
            "current_tombstone_file_sha256": v2.file_sha256(
                ROOT / V1_IMPLEMENTATION_PATH
            ),
            "pre_tombstone_file_sha256": V1_PRE_TOMBSTONE_FILE_SHA256,
            "pre_tombstone_git_blob": V1_PRE_TOMBSTONE_GIT_BLOB,
            "pre_tombstone_commit": V1_PRE_TOMBSTONE_COMMIT,
            "pre_tombstone_definition_commit": (
                V1_PRE_TOMBSTONE_DEFINITION_COMMIT
            ),
            "historical_reconstruction_is_explicit_only": True,
            "historical_reconstruction_authorizes_source_open": False,
            "historical_reconstruction_authorizes_promotion": False,
        },
        "historical_artifacts": {
            "rewrite_prohibited": True,
            "promotion_or_reinterpretation_prohibited": True,
            "artifacts": historical,
        },
        "fail_closed_v1_consumers": [
            {
                "path": path,
                "file_sha256": v2.file_sha256((ROOT / path).resolve()),
                "entrypoint": entrypoint,
                "status": "fail_closed_quarantine_tombstone",
            }
            for path, entrypoint in FAIL_CLOSED_ENTRYPOINTS
        ],
        "successor_policy": {
            "historical_v67_v69_v70_v75_v82_rank_reward_fp32_sampling": (
                "new versioned successors required; historical bytes remain fixed"
            ),
            "new_successor_must_bind_v2_contract_and_reservation": True,
            "silent_manifest_rebinding_prohibited": True,
            "post_outcome_rewrite_prohibited": True,
        },
        "v1_transitive_v66_lineage": {
            "status": "immutable_historical_nonpromotable",
            "quality_hpo_or_promotion_authorized": False,
            "in_place_rebinding_prohibited": True,
            "nodes": [
                {
                    "id": "v66",
                    "status": "historical_v1_bound_nonpromotable",
                },
                {
                    "id": "v66b",
                    "status": "historical_v1_descendant_nonpromotable",
                },
                {
                    "id": "v66c",
                    "status": "historical_v1_descendant_nonpromotable",
                },
                {
                    "id": "v66d",
                    "status": "historical_v1_descendant_nonpromotable",
                },
                {
                    "id": "v71_v72",
                    "status": (
                        "systems_implementation_only_no_quality_authority"
                    ),
                },
                {
                    "id": "v73",
                    "status": "historical_v1_descendant_nonpromotable",
                },
                {
                    "id": "v73b",
                    "status": "historical_v1_descendant_nonpromotable",
                },
                {
                    "id": "v73c",
                    "status": "sealed_stage_a_systems_only_trace_exception",
                    "requirements": {
                        "semantic_evaluation_or_quarantined_boundary_resolved": False,
                        "quality_hpo_or_promotion_authorized": False,
                        "systems_trace_only": True,
                        "lineage_rehabilitation": False,
                        "postrun_boundary_denial_receipt_required": True,
                    },
                },
            ],
            "migration": (
                "coordinate launcher tombstones and additive V2-bound successors "
                "after the V73C systems trace is sealed"
            ),
        },
        "explicit_blockers": [
            {
                "code": "fresh_source_disjoint_multi_item_qa_dev_absent",
                "resolution": (
                    "QA HPO or general quality promotion requires a separate "
                    "untouched multi-item QA dev and explicit authority; current "
                    "four-document dev authorizes prose-logprob tuning only"
                ),
            },
            {
                "code": "legacy_v1_row_level_comparison_prohibited",
                "resolution": (
                    "do not reopen V1 to compare unknown rows; V2 makes no "
                    "row-level semantic-disjointness claim against that source"
                ),
            },
            {
                "code": "fresh_curriculum_successor_absent",
                "resolution": (
                    "a curriculum successor requires fresh source-disjoint domain "
                    "material, V2 reservation exclusion, and a new sampling contract"
                ),
            },
            {
                "code": "terminal_execution_not_authorized",
                "resolution": (
                    "execution requires a frozen recipe/checkpoint selection receipt "
                    "and explicit authority; no terminal claim may be pre-created"
                ),
            },
            {
                "code": "historical_outcomes_nonpromotable",
                "resolution": (
                    "completed V1-bound outcomes remain historical and cannot be "
                    "converted into V2 evidence"
                ),
            },
            {
                "code": "v66_descendant_launcher_migration_incomplete",
                "resolution": (
                    "V66 descendant launchers and systems-only descendants need "
                    "coordinated V2 successors or launcher-level quarantine checks "
                    "before any quality promotion"
                ),
            },
            {
                "code": "v73c_stage_b_boundary_denial_receipt_pending",
                "resolution": (
                    "V73C postrun evidence is rejected unless a distinct "
                    "content-free receipt proves zero actual semantic-boundary "
                    "opens or resolves and no quality authority"
                ),
            },
        ],
        "implementation": {
            "builder_path": str(Path(__file__).resolve().relative_to(ROOT)),
            "builder_file_sha256": v2.file_sha256(Path(__file__).resolve()),
        },
    }
    registry["content_sha256_before_self_field"] = v2.canonical_sha256(registry)
    return registry


def validate_registry(registry: dict) -> None:
    compact = {
        key: value for key, value in registry.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        registry.get("schema") != SCHEMA
        or registry.get("status")
        != "v2_provisional_prose_only_all_incidents_quarantined_history_preserved"
        or registry.get("content_sha256_before_self_field")
        != v2.canonical_sha256(compact)
        or registry.get("historical_artifacts", {}).get("rewrite_prohibited")
        is not True
        or registry.get("v2_resealed_source_reservation", {}).get(
            "required_before_any_future_dataset_write"
        ) is not True
        or registry.get("v2_resealed_source_reservation", {}).get(
            "terminal_source_count"
        ) != 9
        or registry.get("v2_resealed_source_reservation", {}).get(
            "dev_source_count"
        ) != 4
        or registry.get("v2_resealed_source_reservation", {}).get(
            "reserved_source_count"
        ) != 13
        or registry.get("v2_resealed_source_reservation", {}).get(
            "historical_selected_source_count"
        ) != 12
        or registry.get("content_free_quarantine_boundary", {}).get(
            "exact_path_identity_count"
        ) != 3
        or registry.get("content_free_quarantine_boundary", {}).get(
            "prefix_identity_count"
        ) != 2
        or registry.get("content_free_quarantine_boundary", {}).get(
            "plaintext_boundary_paths_persisted"
        ) is not False
        or len(registry.get("immutable_incident_receipts", ())) != 3
        or registry.get("superseded_v2_revision", {}).get("status")
        != "immutable_superseded_nonpromotable"
        or registry.get("v2_contract", {}).get(
            "qa_hpo_or_general_quality_promotion_authorized"
        ) is not False
        or registry.get("v73c_stage_a_systems_only_closure", {}).get(
            "systems_trace_only"
        ) is not True
        or registry.get("v73c_stage_a_systems_only_closure", {}).get(
            "quality_hpo_or_promotion_authorized"
        ) is not False
        or registry.get("v73c_stage_a_systems_only_closure", {}).get(
            "distinct_postrun_boundary_denial_receipt_required"
        ) is not True
        or registry.get("v1_implementation_tombstone", {}).get(
            "pre_tombstone_file_sha256"
        ) != V1_PRE_TOMBSTONE_FILE_SHA256
        or registry.get("v1_implementation_tombstone", {}).get(
            "pre_tombstone_git_blob"
        ) != V1_PRE_TOMBSTONE_GIT_BLOB
        or registry.get("v1_implementation_tombstone", {}).get(
            "pre_tombstone_commit"
        ) != V1_PRE_TOMBSTONE_COMMIT
        or registry.get("v1_implementation_tombstone", {}).get(
            "pre_tombstone_definition_commit"
        ) != V1_PRE_TOMBSTONE_DEFINITION_COMMIT
    ):
        raise RuntimeError("invalid V2 migration registry")


def _write_exclusive(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_registry()
    validate_registry(value)
    if args.check:
        if _read_json(args.output.resolve()) != value:
            raise RuntimeError("persisted V2 migration registry differs from rebuild")
    else:
        _write_exclusive(args.output.resolve(), value)
    print(json.dumps({
        "content_sha256": value["content_sha256_before_self_field"],
        "historical_artifacts_preserved": len(HISTORICAL_ARTIFACTS),
        "terminal_sources_reserved": value["v2_resealed_source_reservation"][
            "terminal_source_count"
        ],
        "dev_sources_reserved": value["v2_resealed_source_reservation"][
            "dev_source_count"
        ],
        "total_sources_reserved": value["v2_resealed_source_reservation"][
            "reserved_source_count"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
