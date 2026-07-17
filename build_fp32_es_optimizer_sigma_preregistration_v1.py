#!/usr/bin/env python3
"""Build the CPU-only FP32 optimizer/module-sigma preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from safetensors import safe_open

import fp32_es_optimizer_ablation_v1 as contract


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "fp32_es_optimizer_module_sigma_ablation_v1.json"
).resolve()
PARENT_CONTRACT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json"
).resolve()
MIRRORED_PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_mirrored_calibration_v66.json"
).resolve()
SOURCE_ADAPTER = (
    ROOT / "experiments/sft_controls/"
    "v49d_v434_sampling_midpoint_lr5p5e5/"
    "v434_equal_r32_seed17_init20260715041/final"
).resolve()
SOURCE_WEIGHTS = (SOURCE_ADAPTER / "adapter_model.safetensors").resolve()
SOURCE_CONFIG = (SOURCE_ADAPTER / "adapter_config.json").resolve()
TRAIN_DATASET = (
    ROOT / "experiments/sft_controls/"
    "v49d_v434_sampling_midpoint_lr5p5e5/train_v434_fold3_v49d.jsonl"
).resolve()
TRAIN_PANEL = (
    ROOT / "experiments/eggroll_es_hpo/datasets/"
    "v52_v434_train_generation_panel.json"
).resolve()

EXPECTED_FILES = {
    PARENT_CONTRACT: "04af81499067e2feb0186c0a61e4c1af10f838a8eb7deec6dd41cd192748cacf",
    MIRRORED_PREREGISTRATION: "968d96af4c4f511eda317f0fbeda21c0cf4fedcc70caa07d2e2d02e3db17d411",
    SOURCE_WEIGHTS: "0d6efd4d5be626f41cdd711843f799b9b3c09e9ecf6a7a8e6e9aeeff09f6dc5b",
    SOURCE_CONFIG: "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    TRAIN_DATASET: "ae949c37de6abcd57fd8e2b9da8148b80ee072cfc16a7cf023c4ca89021b840a",
    TRAIN_PANEL: "dd2c857b75617351d64cfce29f5a8e5d79ce9da212e4db50d22f2de3795c70a1",
}
EXPECTED_PARENT_CONTENT = (
    "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
)
EXPECTED_MIRRORED_CONTENT = (
    "f706b63befbd9da93cdda6ad9e612bf8fccfeda395e573ae59ff3515f24e8eef"
)
EXPECTED_TRAIN_PANEL_CONTENT = (
    "cdfa9d10669171d5d814b55df1f674a89dfa557c5376b45c8d0073e5d1acaec7"
)
EXPECTED_CANONICAL_V41_MASTER = (
    "eea2d60e19530ba99e9ac4bc50f2806b20aa13ed30e159bad63a0144d0cb81b6"
)


def file_sha256_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_json_v1(path: Path, expected_content: str, expected_schema: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("schema") != expected_schema
        or value.get("content_sha256_before_self_field") != expected_content
        or contract.canonical_sha256_v1(compact) != expected_content
    ):
        raise RuntimeError(f"v1 sealed JSON changed: {path}")
    return value


def _load_master_v1() -> dict:
    tensors = {}
    with safe_open(SOURCE_WEIGHTS, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.dtype.is_floating_point is not True:
                raise RuntimeError("v1 source adapter contains a nonfloating tensor")
            tensors[key] = tensor.contiguous()
    contract.fp32_master_identity_v1(tensors, require_production_surface=True)
    return tensors


def _parent_contract_v1() -> tuple[dict, dict]:
    value = _checked_json_v1(
        PARENT_CONTRACT,
        EXPECTED_PARENT_CONTENT,
        "specialist-recipe-evaluation-compute-contract-v1",
    )
    roles = value["roles"]
    if (
        value.get("status")
        != "sealed_before_recipe_hpo_protected_holdout_unopened_by_hpo"
        or value.get("disjointness", {}).get("passed") is not True
        or roles["train"]["file_sha256"] != EXPECTED_FILES[TRAIN_DATASET]
        or roles["protected_holdout"]["access_authorized_by_this_contract"]
        is not False
        or value["compute_accounting"]["budget_modes"]["estimator_control"]
        ["confirmation_target_generated_rollouts_per_seed"]
        != contract.ROLLOUTS_PER_REPLICATE_V1
    ):
        raise RuntimeError("v1 parent evaluation/compute contract changed")
    minimized = {
        "path": str(PARENT_CONTRACT),
        "file_sha256": EXPECTED_FILES[PARENT_CONTRACT],
        "content_sha256": EXPECTED_PARENT_CONTENT,
        "source_disjointness_passed": True,
        "adaptation_train_file_sha256": roles["train"]["file_sha256"],
        "estimator_control_rollouts_per_seed": (
            contract.ROLLOUTS_PER_REPLICATE_V1
        ),
        "estimator_control_gpu_second_ceiling_per_seed": (
            contract.GPU_SECOND_CEILING_PER_REPLICATE_V1
        ),
        "protected_access_authorized": False,
    }
    return value, minimized


def _mirrored_dependency_v1() -> dict:
    value = _checked_json_v1(
        MIRRORED_PREREGISTRATION,
        EXPECTED_MIRRORED_CONTENT,
        "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66",
    )
    recipe = value["fixed_recipe"]
    if (
        recipe["direction_count"] != contract.DIRECTIONS_PER_UPDATE_V1
        or recipe["signed_population_size"]
        != contract.SIGNED_CANDIDATES_PER_UPDATE_V1
        or recipe["train_rows_per_candidate"]
        != contract.TRAIN_UNITS_PER_CANDIDATE_V1
        or value["adapter"]["canonical_fp32_master_sha256"]
        != EXPECTED_CANONICAL_V41_MASTER
    ):
        raise RuntimeError("v1 mirrored-ES dependency surface changed")
    return {
        "path": str(MIRRORED_PREREGISTRATION),
        "file_sha256": EXPECTED_FILES[MIRRORED_PREREGISTRATION],
        "content_sha256": EXPECTED_MIRRORED_CONTENT,
        "cpu_contract_complete": True,
        "substantive_nonzero_pair_differences_and_exact_restore_observed": True,
        "accepted_all_four_gpu_activity_receipt_complete": False,
        "status": "v66d_all_four_gpu_activity_receipt_pending",
    }


def _sigma_tables_v1(master: dict) -> list[dict]:
    tables = []
    for update_index, base_sigma in enumerate(contract.SIGMA_SCHEDULE_V1):
        for mode in contract.SCALE_MODES_V1:
            table = contract.module_sigma_table_v1(master, base_sigma, mode)
            tables.append({
                "update_index": update_index,
                "mode": mode,
                "table": table,
            })
    return tables


def _initial_checkpoints_v1(master: dict, arms: list[dict], seeds: dict) -> dict:
    result = {}
    schedule_sha = contract.canonical_sha256_v1(list(contract.SIGMA_SCHEDULE_V1))
    for arm in arms:
        optimizer = arm["optimizer"]
        state = contract.initial_optimizer_state_v1(optimizer)
        per_seed = {}
        for replicate_seed in contract.REPLICATE_SEEDS_V1:
            run_state = {
                "schema": "fp32-es-run-state-v1",
                "arm_id": arm["arm_id"],
                "replicate_seed": replicate_seed,
                "completed_updates": 0,
                "next_update_index": 0,
                "next_direction_seed_set_sha256": contract.canonical_sha256_v1(
                    seeds[str(replicate_seed)][0]
                ),
                "train_panel_cursor": 0,
                "train_panel_content_sha256": EXPECTED_TRAIN_PANEL_CONTENT,
                "sigma_schedule_sha256": schedule_sha,
            }
            identity = contract.checkpoint_identity_v1(
                master, state, optimizer, run_state
            )
            per_seed[str(replicate_seed)] = identity["checkpoint_sha256"]
        result[arm["arm_id"]] = per_seed
    return result


def _implementation_bindings_v1() -> dict:
    paths = {
        "builder": Path(__file__).resolve(),
        "cpu_contract": ROOT / "fp32_es_optimizer_ablation_v1.py",
        "mirrored_protocol_dependency": ROOT / "eggroll_es_mirrored_v66.py",
        "canonical_fp32_worker_dependency": ROOT / "eggroll_es_worker_lora_v41a.py",
    }
    return {
        key: {"path": str(path.resolve()), "file_sha256": file_sha256_v1(path)}
        for key, path in paths.items()
    }


def build_preregistration_v1() -> dict:
    for path, expected in EXPECTED_FILES.items():
        observed = file_sha256_v1(path)
        if observed != expected:
            raise RuntimeError(
                f"v1 sealed input changed: {path}: {observed} != {expected}"
            )
    parent, parent_minimized = _parent_contract_v1()
    mirrored = _mirrored_dependency_v1()
    master = _load_master_v1()
    master_identity = contract.fp32_master_identity_v1(
        master, require_production_surface=True
    )
    module_inventory = contract.logical_module_inventory_v1(
        master, require_production_surface=True
    )
    arms = contract.arm_grid_v1()
    direction_seeds = {
        str(seed): [
            contract.direction_seeds_v1(EXPECTED_PARENT_CONTENT, seed, update)
            for update in range(contract.UPDATES_PER_REPLICATE_V1)
        ]
        for seed in contract.REPLICATE_SEEDS_V1
    }
    initial_checkpoints = _initial_checkpoints_v1(
        master, arms, direction_seeds
    )
    sigma_tables = _sigma_tables_v1(master)
    module_tables = [
        item["table"] for item in sigma_tables
        if item["mode"] == "module_fp32_rms_shape_normalized"
    ]
    max_module_energy_share = max(
        record["expected_perturbation_square"]
        / table["expected_perturbation_l2_squared"]
        for table in module_tables
        for record in table["records"]
    )
    final_eval_requests = {
        "dev_generated_requests": parent["roles"]["dev"]["rows"],
        "dev_teacher_forced_requests": parent["roles"]["dev"]["rows"],
        "ood_qa_generated_requests": parent["roles"]["ood"]["qa"]["rows"],
        "ood_qa_teacher_forced_requests": parent["roles"]["ood"]["qa"]["rows"],
        "ood_prose_teacher_forced_requests": parent["roles"]["ood"]["prose"]["rows"],
    }
    run_root = (
        ROOT / "experiments/eggroll_es_hpo/runs/"
        "fp32_es_optimizer_module_sigma_v1"
    ).resolve()
    result = {
        "schema": contract.SCHEMA_V1,
        "status": "sealed_cpu_preview_runtime_dependencies_pending",
        "purpose": (
            "Nonadaptive, compute-matched comparison of host-FP32 SGD, "
            "momentum, and AdamW ES updates plus mirrored-safe sign/rank "
            "diagnostics under global versus shape-aware module-RMS sigma."
        ),
        "authorization": {
            "cpu_preview": True,
            "gpu_launch": False,
            "train_semantics": True,
            "dev_after_training": True,
            "ood_after_training": True,
            "protected_holdout": False,
            "live_run_access": False,
        },
        "dependencies": {
            "mirrored_es": mirrored,
            "memory_roofline": {
                "bead": "specialist-0j5.14",
                "existing_profile_is_diagnostic": True,
                "optimizer_phase_transfer_and_bandwidth_receipt_complete": False,
                "status": "optimizer_phase_empirical_profile_pending",
            },
            "all_runtime_dependencies_complete": False,
        },
        "parent_contract": parent_minimized,
        "inputs": {
            "train_dataset": {
                "path": str(TRAIN_DATASET),
                "file_sha256": EXPECTED_FILES[TRAIN_DATASET],
                "rows": 448,
                "use": "model updates and train-only reward/SNR",
            },
            "train_panel": {
                "path": str(TRAIN_PANEL),
                "file_sha256": EXPECTED_FILES[TRAIN_PANEL],
                "content_sha256": EXPECTED_TRAIN_PANEL_CONTENT,
                "unique_conflict_units": 64,
            },
            "source_adapter": {
                "path": str(SOURCE_ADAPTER),
                "weights_file_sha256": EXPECTED_FILES[SOURCE_WEIGHTS],
                "config_file_sha256": EXPECTED_FILES[SOURCE_CONFIG],
                "canonical_v41_master_sha256": EXPECTED_CANONICAL_V41_MASTER,
            },
        },
        "parameter_surface": {
            **master_identity,
            "dtype": "torch.float32",
            "runtime_view_elements": contract.RUNTIME_ELEMENTS_V1,
            "runtime_view_dtype": "torch.bfloat16_derived_inference_cache_only",
            "same_surface_every_arm": True,
            "module_inventory": module_inventory,
        },
        "sigma_contract": {
            "modes": list(contract.SCALE_MODES_V1),
            "schedule": list(contract.SIGMA_SCHEDULE_V1),
            "schedule_rule": "sealed_two_rung_decay_0p0006_then_0p0003",
            "rms_floor": contract.RMS_FLOOR_V1,
            "module_basis": "max(fp32_sqrt(sum_squares/elements),rms_floor)",
            "shape_normalization": (
                "sqrt(total_elements/sum(elements*raw_basis^2))"
            ),
            "candidate_scale_application_count": 1,
            "estimator_inverse_scale_application_count": 1,
            "equal_expected_perturbation_l2_across_modes": True,
            "max_module_expected_energy_share": max_module_energy_share,
            "tables": sigma_tables,
        },
        "estimator_contract": {
            "directions_per_update": contract.DIRECTIONS_PER_UPDATE_V1,
            "signed_candidates_per_update": (
                contract.SIGNED_CANDIDATES_PER_UPDATE_V1
            ),
            "registered_coefficient_modes": list(contract.COEFFICIENT_MODES_V1),
            "raw_formula": (
                "per_module sum_i((Rplus_i-Rminus_i)*unit_epsilon_i)"
                "/(2*directions*sigma_module)"
            ),
            "pair_sign_swap_must_negate_coefficient": True,
            "signed_rank_rule": (
                "average rank of absolute pair differences then restore sign"
            ),
            "independent_signed_candidate_ranking": "prohibited",
            "unpaired_reward_centering": "prohibited",
            "raw_mode_is_smoothed_objective_gradient_estimator": True,
            "sign_and_rank_modes_are_direction_heuristics_not_unbiased_gradients": True,
        },
        "optimizer_contract": {
            "configs": contract.OPTIMIZER_CONFIGS_V1,
            "master_dtype": "torch.float32",
            "slot_dtype": "torch.float32",
            "slot_location": "host",
            "state_replication": "exact_on_all_four_replicas",
            "reward_direction": "ascent",
            "adam_bias_correction_uses": "new_exact_committed_step",
            "adamw_decay_direction": "minus_weight_decay_times_fp32_master",
            "bf16_or_device_optimizer_state": "prohibited",
        },
        "update_budget_contract": {
            "ratio": contract.UPDATE_BUDGET_RATIO_V1,
            "rms_floor": contract.RMS_FLOOR_V1,
            "formula": "ratio*max(fp32_master_l2,rms_floor*sqrt(elements))",
            "relative_tolerance": contract.UPDATE_NORM_RELATIVE_TOLERANCE_V1,
            "nonzero_update_hits_exact_budget": True,
            "same_formula_and_ratio_every_arm": True,
            "projection_occurs_after_optimizer_and_weight_decay_direction": True,
        },
        "compute_contract": {
            "budget_mode": "estimator_control",
            "replicate_seeds": list(contract.REPLICATE_SEEDS_V1),
            "updates_per_replicate": contract.UPDATES_PER_REPLICATE_V1,
            "directions_per_update": contract.DIRECTIONS_PER_UPDATE_V1,
            "signed_candidates_per_update": (
                contract.SIGNED_CANDIDATES_PER_UPDATE_V1
            ),
            "train_units_per_candidate": contract.TRAIN_UNITS_PER_CANDIDATE_V1,
            "rollouts_per_update": contract.ROLLOUTS_PER_UPDATE_V1,
            "rollouts_per_replicate": contract.ROLLOUTS_PER_REPLICATE_V1,
            "direction_seeds_by_replicate_and_update": direction_seeds,
            "same_crn_direction_and_train_panel_by_seed_update_across_arms": True,
            "same_parameter_surface_all_arms": True,
            "exact_final_eval_requests_per_replicate": final_eval_requests,
            "gpu_second_ceiling_per_replicate": (
                contract.GPU_SECOND_CEILING_PER_REPLICATE_V1
            ),
            "gpu_seconds_and_optimizer_runtime_are_reported_outcomes": True,
            "failed_budget_reallocation": "prohibited",
        },
        "grid": arms,
        "memory_bandwidth_contract": contract.optimizer_memory_contract_v1(),
        "checkpoint_contract": {
            "initial_checkpoint_sha256_by_arm_and_seed": initial_checkpoints,
            "transaction": [
                "prepare_exact_master_and_optimizer_rollback_identity",
                "execute_candidate_without_overwriting_committed_state",
                "verify_four_replica_master_optimizer_and_runtime_consensus",
                "atomic_commit_or_exact_master_optimizer_rng_cursor_rollback",
            ],
            "resume_requires_exact": [
                "master state", "optimizer slots and committed step",
                "direction seed cursor", "train panel cursor", "sigma rung",
                "arm and replicate identity",
            ],
            "rollback_checkpoint_must_equal_previous_checkpoint": True,
            "candidate_checkpoint_must_equal_all_four_replica_commits": True,
            "unregistered_checkpoint_or_partial_state_resume": "prohibited",
            "logical_checkpoint_bytes_by_optimizer_per_replica": {
                optimizer: item["checkpoint_tensor_and_step_bytes_per_replica"]
                for optimizer, item in contract.optimizer_memory_contract_v1()[
                    "optimizers"
                ].items()
            },
        },
        "evaluation_gates": {
            "train_only_during_updates": True,
            "dev_and_ood_open_only_after_final_checkpoint_sealed": True,
            "no_model_update_after_dev_or_ood_open": True,
            "dev": {
                "file_sha256": parent["roles"]["dev"]["file_sha256"],
                "manifest_content_sha256": parent["roles"]["dev"]
                ["manifest_content_sha256"],
                "rows": parent["roles"]["dev"]["rows"],
                "use": "HPO fixed-rung comparison",
            },
            "ood": {
                "qa_file_sha256": parent["roles"]["ood"]["qa"]["file_sha256"],
                "qa_rows": parent["roles"]["ood"]["qa"]["rows"],
                "prose_file_sha256": parent["roles"]["ood"]["prose"]["file_sha256"],
                "prose_rows": parent["roles"]["ood"]["prose"]["rows"],
            },
            "ood_use": "noninferiority_only_not_point_optimization",
            "protected_holdout_access": False,
            "protected_identity_set_sha256": parent["roles"]
            ["protected_holdout"]["selected_identity_set_sha256"],
            "protected_paths_or_semantics_persisted": False,
        },
        "failure_policy": {
            "nonfinite": "abort_restore_exact_checkpoint_no_retry_seed",
            "zero_pair_difference_variance": (
                "skip_without_optimizer_step_or_checkpoint_change"
            ),
            "zero_optimizer_direction": (
                "skip_without_optimizer_step_or_checkpoint_change"
            ),
            "replica_or_resume_mismatch": "abort_restore_exact_checkpoint",
            "failed_arm_budget_reallocated": False,
            "skip_count_reported_and_arm_ineligible_if_any_skip": True,
        },
        "reporting": {
            "per_arm_per_seed": [
                "pair-difference mean, variance, SNR, and split-half agreement",
                "zero-variance/nonfinite/rollback counts",
                "raw and budgeted update norms",
                "optimizer and total runtime",
                "host memory traffic and bandwidth by phase",
                "D2H/H2D bytes and time by phase",
                "peak allocated/reserved VRAM on GPUs 0-3",
                "dev primary/secondary tuple",
                "OOD QA/prose noninferiority gates",
            ],
            "selection": (
                "parent contract dev tuple then OOD all-condition gate; ties by "
                "lower charged GPU seconds then lexicographic arm id"
            ),
            "protected_result_may_change_recipe": False,
            "all_24_arm_seed_receipts_required": True,
        },
        "implementation_bindings": _implementation_bindings_v1(),
        "artifacts": {
            "preregistration": str(OUTPUT),
            "fresh_run_root_reserved_not_opened": str(run_root),
            "receipt_template": str(run_root / "{arm_id}/seed_{seed}/receipt_v1.json"),
            "aggregate_report": str(run_root / "aggregate_v1.json"),
        },
    }
    result["content_sha256_before_self_field"] = contract.canonical_sha256_v1(result)
    contract.validate_preregistration_v1(result)
    return result


def write_preregistration_v1(path: Path = OUTPUT) -> dict:
    value = build_preregistration_v1()
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n").encode("ascii")
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    value = build_preregistration_v1()
    output = Path(args.output).resolve()
    if args.check:
        if (
            not output.is_file()
            or json.loads(output.read_text(encoding="utf-8")) != value
        ):
            raise RuntimeError("v1 checked-in preregistration is absent or stale")
    else:
        write_preregistration_v1(output)
    print(json.dumps({
        "path": str(output),
        "file_sha256": file_sha256_v1(output) if output.is_file() else None,
        "content_sha256": value["content_sha256_before_self_field"],
        "checked": bool(args.check),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
