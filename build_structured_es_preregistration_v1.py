#!/usr/bin/env python3
"""Build the CPU-only EGGROLL structured-perturbation preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from collections import Counter
from pathlib import Path

import fp32_es_optimizer_ablation_v1 as optimizer_contract
import structured_es_oracle_v1 as oracle


ROOT = Path(__file__).resolve().parent
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "structured_es_lora_comparison_v1.json"
).resolve()
OPTIMIZER_PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "fp32_es_optimizer_module_sigma_ablation_v1.json"
).resolve()
V66D_PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_mirrored_calibration_v66d.json"
).resolve()
V66D_EVIDENCE = (
    ROOT / "experiments/eggroll_es_hpo/"
    "mirrored_es_gpu_attribution_v66d_cpu_evidence_20260717.md"
).resolve()
PAPER_MARKDOWN = (ROOT / "references/papers/2509.24372.md").resolve()
PAPER_PDF = (ROOT / "references/papers/2509.24372.pdf").resolve()
REPLICATION_NOTES = (ROOT / "EGGROLL_ES_REPLICATION.md").resolve()
MODEL = (ROOT / "models/Qwen3.6-35B-A3B").resolve()
MODEL_INDEX = (MODEL / "model.safetensors.index.json").resolve()
MODEL_CONFIG = (MODEL / "config.json").resolve()

EXPECTED_FILES = {
    OPTIMIZER_PREREG: oracle.OPTIMIZER_CONTRACT_FILE_SHA256_V1,
    V66D_PREREG: oracle.V66D_PREREG_FILE_SHA256_V1,
    V66D_EVIDENCE: "31fe84300c8ee5c4ae03c61585f55ce5db786310e2755b5e9919dfdb8b15f759",
    PAPER_MARKDOWN: "48bb4cf20e6f44ce0fc6b57ad16ba19cdf3cbcc7c8548d36ad4c84c2eb80d081",
    PAPER_PDF: "67935fe6e7001d0e34b7c17819a17ba6477d1db4a4c35f664bbfaff04bf200fd",
    REPLICATION_NOTES: "88e14619fecca342ee8b6b020b5469118f0226cb2f48bdfce5d903118ee98015",
    MODEL_INDEX: "41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83",
    MODEL_CONFIG: "93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99",
}
EXPECTED_V66D_EVIDENCE_IDENTITIES = {
    "report_file_sha256": oracle.V66D_REPORT_FILE_SHA256_V1,
    "report_content_sha256": oracle.V66D_REPORT_CONTENT_SHA256_V1,
    "gpu_telemetry_file_sha256": oracle.V66D_TELEMETRY_FILE_SHA256_V1,
    "actor_log_file_sha256": oracle.V66D_ACTOR_LOG_FILE_SHA256_V1,
    "population_file_sha256": (
        "9d172d15f82a54c697b8b860ff3131733d59006f1e4b790b5b9b87ded679e9d4"
    ),
    "update_file_sha256": (
        "f958f90b26c5b2afa4a81b03a0ab91c12d9684c2ce236bbb658d674e7a5eeffd"
    ),
}


def file_sha256_v1(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _self_hashed_json_v1(path: Path, expected_schema: str, expected_content: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("schema") != expected_schema
        or value.get("content_sha256_before_self_field") != expected_content
        or oracle.canonical_sha256_v1(compact) != expected_content
    ):
        raise RuntimeError(f"v1 sealed JSON changed: {path}")
    return value


def _optimizer_source_v1() -> tuple[dict, dict]:
    value = _self_hashed_json_v1(
        OPTIMIZER_PREREG,
        optimizer_contract.SCHEMA_V1,
        oracle.OPTIMIZER_CONTRACT_CONTENT_SHA256_V1,
    )
    optimizer_contract.validate_preregistration_v1(value)
    if (
        value["parameter_surface"]["elements"] != oracle.LORA_ELEMENTS_V1
        or value["parameter_surface"]["tensor_count"] != oracle.LORA_TENSORS_V1
        or value["optimizer_contract"]["configs"]["sgd"]["optimizer"] != "sgd"
        or value["sigma_contract"]["schedule"] != list(oracle.SIGMA_SCHEDULE_V1)
        or value["update_budget_contract"]["ratio"]
        != oracle.UPDATE_BUDGET_RATIO_V1
    ):
        raise RuntimeError("v1 FP32 optimizer/sigma source surface changed")
    return value, {
        "path": str(OPTIMIZER_PREREG),
        "file_sha256": EXPECTED_FILES[OPTIMIZER_PREREG],
        "content_sha256": oracle.OPTIMIZER_CONTRACT_CONTENT_SHA256_V1,
        "cpu_contract_complete": True,
        "selected_fixed_optimizer_for_isolation": "sgd",
        "selected_fixed_sigma_mode_for_isolation": "global",
        "selection_used_empirical_optimizer_results": False,
    }


def _v66d_source_v1() -> tuple[dict, dict]:
    value = _self_hashed_json_v1(
        V66D_PREREG,
        "lora-es-mirrored-crn-qwen36-calibration-preregistration-v66d",
        oracle.V66D_PREREG_CONTENT_SHA256_V1,
    )
    if (
        value["fixed_recipe"]["direction_count"]
        != oracle.DIRECTIONS_PER_UPDATE_V1
        or value["fixed_recipe"]["signed_population_size"]
        != oracle.SIGNED_CANDIDATES_PER_UPDATE_V1
        or value["fixed_recipe"]["train_rows_per_candidate"]
        != oracle.TRAIN_UNITS_PER_CANDIDATE_V1
        or value["authorization"]["protected_holdout_access"] is not False
    ):
        raise RuntimeError("v1 V66d mirrored/telemetry source changed")
    return value, {
        "preregistration": {
            "path": str(V66D_PREREG),
            "file_sha256": EXPECTED_FILES[V66D_PREREG],
            "content_sha256": oracle.V66D_PREREG_CONTENT_SHA256_V1,
        },
        "accepted_telemetry_evidence": {
            "path": str(V66D_EVIDENCE),
            "file_sha256": EXPECTED_FILES[V66D_EVIDENCE],
            **EXPECTED_V66D_EVIDENCE_IDENTITIES,
            "accepted": True,
            "all_16_signed_candidates_actor_receipted": True,
            "all_four_gpu_batches_acknowledged": True,
            "peak_nvml_utilization_by_gpu": {"0": 74, "1": 100, "2": 100, "3": 100},
            "peak_memory_mib_each_gpu": 84_138,
            "charged_gpu_seconds": 492.405638704,
            "exact_abort_restore_and_final_idle": True,
            "live_artifacts_opened_by_this_builder": False,
        },
    }


def _lora_tensor_shapes_v1(optimizer: dict) -> list[dict]:
    rows = []
    for module in optimizer["parameter_surface"]["module_inventory"]:
        rows.extend((
            {"key": module["a_key"], "shape": module["a_shape"]},
            {"key": module["b_key"], "shape": module["b_shape"]},
        ))
    rows.sort(key=lambda item: item["key"])
    if (
        len(rows) != oracle.LORA_TENSORS_V1
        or sum(math.prod(item["shape"]) for item in rows) != oracle.LORA_ELEMENTS_V1
        or len({item["key"] for item in rows}) != len(rows)
    ):
        raise RuntimeError("v1 LoRA tensor-shape inventory changed")
    return rows


def _dense_metadata_inventory_v1() -> dict:
    shards = sorted(MODEL.glob("*.safetensors"))
    tensors = 0
    elements = 0
    data_bytes = 0
    maximum_elements = 0
    dtypes = Counter()
    ordered_records = []
    for path in shards:
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_size))
        for key, item in header.items():
            if key == "__metadata__":
                continue
            shape = item["shape"]
            count = math.prod(shape)
            byte_count = item["data_offsets"][1] - item["data_offsets"][0]
            tensors += 1
            elements += count
            data_bytes += byte_count
            maximum_elements = max(maximum_elements, count)
            dtypes[item["dtype"]] += count
            ordered_records.append({
                "key": key,
                "shape": shape,
                "dtype": item["dtype"],
                "elements": count,
                "data_bytes": byte_count,
                "shard": path.name,
            })
    ordered_records.sort(key=lambda item: item["key"])
    if (
        len(shards) != 26
        or tensors != oracle.FULL_MODEL_TENSORS_V1
        or elements != oracle.FULL_MODEL_ELEMENTS_V1
        or data_bytes != oracle.FULL_MODEL_BF16_BYTES_V1
        or maximum_elements != oracle.FULL_MODEL_MAX_TENSOR_ELEMENTS_V1
        or dtypes != Counter({"BF16": oracle.FULL_MODEL_ELEMENTS_V1})
    ):
        raise RuntimeError("v1 dense model metadata surface changed")
    return {
        "metadata_only_no_tensor_payload_loaded": True,
        "shard_count": len(shards),
        "tensor_count": tensors,
        "elements": elements,
        "bf16_bytes": data_bytes,
        "fp32_master_bytes": elements * 4,
        "maximum_tensor_elements": maximum_elements,
        "dtype_element_counts": dict(sorted(dtypes.items())),
        "ordered_tensor_metadata_sha256": oracle.canonical_sha256_v1(ordered_records),
    }


def _implementation_bindings_v1() -> dict:
    paths = {
        "builder": Path(__file__).resolve(),
        "cpu_oracle": ROOT / "structured_es_oracle_v1.py",
        "fp32_optimizer_sigma_contract": ROOT / "fp32_es_optimizer_ablation_v1.py",
        "v66d_worker_dependency": ROOT / "eggroll_es_worker_lora_v66d.py",
        "v66d_runner_dependency": ROOT / "run_lora_es_mirrored_calibration_v66d.py",
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
                f"v1 sealed source changed: {path}: {observed} != {expected}"
            )
    optimizer, optimizer_source = _optimizer_source_v1()
    _v66d, v66d_source = _v66d_source_v1()
    tensor_shapes = _lora_tensor_shapes_v1(optimizer)
    lora_accounting = oracle.lora_streaming_accounting_v1(tensor_shapes)
    dense = _dense_metadata_inventory_v1()
    dense_anchor_accounting = {
        "whole_surface_noise_elements_allocated": 0,
        "whole_surface_candidate_elements_allocated": 0,
        "inplace_tensor_streaming_required": True,
        "bf16_model_bytes_per_replica": oracle.FULL_MODEL_BF16_BYTES_V1,
        "fp32_master_bytes_per_replica": oracle.FULL_MODEL_FP32_MASTER_BYTES_V1,
        "four_replica_fp32_master_bytes": (
            oracle.FULL_MODEL_FP32_MASTER_BYTES_V1 * oracle.WORLD_SIZE_V1
        ),
        "maximum_single_tensor_fp32_noise_bytes": (
            oracle.FULL_MODEL_MAX_TENSOR_ELEMENTS_V1 * 4
        ),
        "weighted_update_scratch_ceiling_bytes": (
            oracle.FULL_MODEL_MAX_TENSOR_ELEMENTS_V1 * 8
        ),
        "candidate_parameter_write_bytes_per_signed_candidate": (
            oracle.FULL_MODEL_BF16_BYTES_V1
        ),
        "surface_ratio_vs_lora": (
            oracle.FULL_MODEL_ELEMENTS_V1 / oracle.LORA_ELEMENTS_V1
        ),
        "capacity_preflight_required_before_launch": True,
    }
    run_root = (
        ROOT / "experiments/eggroll_es_hpo/runs/structured_es_lora_comparison_v1"
    ).resolve()
    result = {
        "schema": oracle.SCHEMA_V1,
        "status": "sealed_cpu_correctness_runtime_dependencies_pending",
        "purpose": (
            "Define and adversarially verify an absolute-index, streamed "
            "rank-k structured perturbation oracle; preregister a matched "
            "LoRA-space comparison and an explicitly unmatched dense-system anchor."
        ),
        "authorization": {
            "cpu_correctness": True,
            "gpu_launch": False,
            "train_identity_use": True,
            "dev_or_ood": False,
            "protected_holdout": False,
            "live_run_read": False,
            "candidate_commit": False,
            "promotion": False,
        },
        "dependencies": {
            "v66d_accepted_gpu_attribution_complete": True,
            "fp32_optimizer_sigma_cpu_contract_complete": True,
            "production_streaming_worker_complete": False,
            "optimizer_phase_pcie_profile_complete": False,
            "dense_fullweight_capacity_preflight_complete": False,
            "runtime_blockers": [
                "CUDA absolute-index generator and chunk writer are not implemented",
                "direct PCIe byte counters remain absent from the phase roofline",
                "dense fullweight four-replica FP32-master capacity is not proven",
            ],
        },
        "source_contracts": {
            "fp32_optimizer_sigma": optimizer_source,
            "v66d_preregistration": v66d_source["preregistration"],
            "v66d_accepted_telemetry": v66d_source["accepted_telemetry_evidence"],
            "paper": {
                "markdown_path": str(PAPER_MARKDOWN),
                "markdown_file_sha256": EXPECTED_FILES[PAPER_MARKDOWN],
                "pdf_file_sha256": EXPECTED_FILES[PAPER_PDF],
                "reused_principles": [
                    "seed-reconstructed noise",
                    "in-place layer/tensor perturbation and exact restoration",
                    "decomposed seed-by-seed update without full noise surface",
                ],
                "structured_outer_product_is_new_ablation_not_paper_claim": True,
            },
            "replication_notes": {
                "path": str(REPLICATION_NOTES),
                "file_sha256": EXPECTED_FILES[REPLICATION_NOTES],
                "upstream_commit": "574a9d134da1ffce2a8bb812019899e5c96b588a",
            },
        },
        "rng_contract": {
            "algorithm": oracle.RNG_ALGORITHM_V1,
            "absolute_indexed": True,
            "counter_inputs": [
                "direction seed", "full tensor key", "method/factor domain",
                "absolute element or factor ordinal",
            ],
            "chunk_shard_tensor_order_independent": True,
            "left_right_tensor_and_method_domains_distinct": True,
            "box_muller_rounding": "CPU oracle rounds once to IEEE FP32",
            "structured_dot_rounding": (
                "FP32 product and FP32 addition in ascending component order"
            ),
            "cuda_acceptance": "exact counters and at most 2 ULP final update drift",
            "local_chunk_index_rng": "prohibited",
            "mutable_generator_state": "prohibited",
        },
        "structured_scale_theory": {
            "construction": "epsilon=(U@V.T)/sqrt(k), U,V iid standard normal",
            "ranks": list(oracle.STRUCTURED_RANKS_V1),
            "rank_scale": "1/sqrt(k)",
            "entry_mean": 0.0,
            "entry_variance": 1.0,
            "expected_frobenius_square": "rows*columns",
            "extra_row_or_column_normalization": "prohibited",
            "moments": [
                oracle.structured_moment_theory_v1(rank)
                for rank in oracle.STRUCTURED_RANKS_V1
            ],
            "estimator_scope": (
                "isotropic first-order central finite difference; finite-rank "
                "noise is not claimed to equal Gaussian smoothing at finite sigma"
            ),
        },
        "surfaces": {
            "lora": {
                "tensor_count": oracle.LORA_TENSORS_V1,
                "logical_module_count": oracle.LORA_MODULES_V1,
                "elements": oracle.LORA_ELEMENTS_V1,
                "fp32_bytes": oracle.LORA_FP32_BYTES_V1,
                "runtime_bf16_view_bytes": oracle.LORA_RUNTIME_BF16_BYTES_V1,
                "master_identity_sha256": optimizer["parameter_surface"]
                ["tensor_inventory_sha256"],
                "tensor_shapes": tensor_shapes,
            },
            "dense_fullweight": {
                "model": str(MODEL),
                "index_file_sha256": EXPECTED_FILES[MODEL_INDEX],
                "config_file_sha256": EXPECTED_FILES[MODEL_CONFIG],
                **dense,
                "quality_causal_comparison_to_lora": False,
                "systems_anchor_only": True,
            },
        },
        "streaming_contract": {
            "chunk_elements": oracle.CHUNK_ELEMENTS_V1,
            "absolute_contiguous_ranges": True,
            "balanced_four_shard_rule": "[floor(P*r/4),floor(P*(r+1)/4))",
            "exact_absolute_coverage_no_gap_overlap_duplicate": True,
            "fixed_ascending_rank_component_reduction": True,
            "dense_full_surface_noise_materialization": "prohibited",
            "dense_full_surface_candidate_materialization": "prohibited",
            "structured_factors_may_be_cached_one_tensor_at_a_time": True,
            "runtime_views_written_directly_from_chunks": True,
            "plus_minus_reuse_same_noise_identity": True,
            "restore_from_canonical_master_not_noise_subtraction": True,
        },
        "memory_bandwidth_contract": {
            "lora": lora_accounting,
            "dense_fullweight_system_anchor": dense_anchor_accounting,
            "common_candidate_runtime_write_bytes_per_lora_signed_candidate": (
                oracle.LORA_RUNTIME_BF16_BYTES_V1
            ),
            "structured_does_not_reduce_runtime_install_bytes_without_fusion": True,
            "measure_per_phase": [
                "RNG/factor generation bytes and elapsed time",
                "chunk scratch allocated/reserved peak",
                "candidate runtime writes and H2D bytes",
                "restore runtime writes and H2D bytes",
                "weighted update reads/writes and elapsed time",
                "checkpoint/rollback bytes",
            ],
        },
        "transaction_contract": {
            "canonical_master_is_immutable_during_candidate_evaluation": True,
            "phases": [
                "quiescent", "materializing_candidate", "candidate_active",
                "restoring_canonical", "quiescent",
            ],
            "interrupted_partial_candidate_restores_entire_surface": True,
            "candidate_restore_uses_exact_canonical_chunks": True,
            "noise_subtraction_restore": "prohibited",
            "overlap_gap_duplicate_or_identity_failure": "terminal_poison",
            "poisoned_actor_may_accept_further_work": False,
            "optimizer_commit_requires_complete_stream_and_four_replica_consensus": True,
            "reuse_fp32_optimizer_checkpoint_chain": True,
        },
        "compute_contract": {
            "systems_replicate_seed": 1701,
            "gpu_second_ceiling_per_systems_arm_seed": 14_400.0,
            "systems_updates_per_arm_seed": 1,
            "directions_per_update": oracle.DIRECTIONS_PER_UPDATE_V1,
            "signed_candidates_per_update": oracle.SIGNED_CANDIDATES_PER_UPDATE_V1,
            "train_units_per_candidate": oracle.TRAIN_UNITS_PER_CANDIDATE_V1,
            "rollouts_per_systems_arm_seed": oracle.ROLLOUTS_PER_UPDATE_V1,
            "quality_lora_updates_per_seed": len(oracle.SIGMA_SCHEDULE_V1),
            "quality_sigma_schedule": list(oracle.SIGMA_SCHEDULE_V1),
            "quality_replicate_seeds": list(oracle.REPLICATE_SEEDS_V1),
            "quality_rollouts_per_lora_arm_seed": (
                len(oracle.SIGMA_SCHEDULE_V1) * oracle.ROLLOUTS_PER_UPDATE_V1
            ),
            "fixed_optimizer": "sgd",
            "fixed_sigma_mode": "global",
            "update_budget_ratio": oracle.UPDATE_BUDGET_RATIO_V1,
            "update_norm_relative_tolerance": (
                oracle.UPDATE_NORM_RELATIVE_TOLERANCE_V1
            ),
            "same_train_panel_prompt_decode_judge_and_telemetry": True,
            "failed_work_reallocation": "prohibited",
        },
        "arms": oracle.comparison_arms_v1(),
        "benchmark_gates": {
            "cpu_before_runtime": [
                "pinned absolute RNG vectors",
                "all registered ranks match dense CPU oracle across chunkings",
                "four-shard reconstruction has exact coverage and stream digest",
                "candidate midpoint and weighted update agree with dense oracle",
                "scratch/allocation receipts pass",
                "partial-write rollback and terminal-poison tests pass",
            ],
            "systems_rung": (
                "all six arms, seed1701, one update, same 1024 train rollouts"
            ),
            "quality_rung": (
                "five matched LoRA arms, all three seeds, two sealed sigma rungs"
            ),
            "dense_anchor_may_select_recipe": False,
            "structured_quality_requires_no_reward_or_ood_degradation_vs_lora_iid": True,
            "protected_holdout_access": False,
        },
        "failure_policy": {
            "nonfinite_factor_noise_reward_or_update": (
                "abort_restore_exact_canonical_no_seed_replacement"
            ),
            "gap_overlap_duplicate_wrong_absolute_index": "terminal_poison",
            "restore_readback_mismatch": "terminal_poison",
            "scratch_or_dense_allocation_violation": "abort_ineligible",
            "partial_gpu_wave_or_telemetry_mismatch": "abort_restore_then_ineligible",
            "failed_budget_reallocation": False,
        },
        "reporting": {
            "systems": [
                "candidate materialize/restore/update throughput",
                "peak and reserved VRAM by GPU/phase",
                "scratch, host traffic, PCIe D2H/H2D bytes and bandwidth",
                "train reward mean/variance and pair-difference SNR",
                "stream-vs-oracle update max absolute and ULP error",
                "exact rollback/poison/idle receipts",
            ],
            "matched_lora_quality": [
                "reward and pair-difference SNR by arm/seed/rung",
                "update cosine/norm versus IID LoRA",
                "stability/nonfinite/rollback counts",
                "registered dev and OOD noninferiority after training seal",
            ],
            "dense_anchor_label": (
                "unmatched 7939x-larger systems/reward anchor; no causal quality claim"
            ),
            "rank_tradeoff": (
                "report scratch/RNG savings jointly with kurtosis, SNR, and quality"
            ),
        },
        "implementation_bindings": _implementation_bindings_v1(),
        "artifacts": {
            "preregistration": str(OUTPUT),
            "future_run_root_not_opened": str(run_root),
            "systems_receipt_template": str(
                run_root / "systems/{arm_id}/seed_{seed}/receipt_v1.json"
            ),
            "quality_receipt_template": str(
                run_root / "quality/{arm_id}/seed_{seed}/receipt_v1.json"
            ),
            "aggregate_report": str(run_root / "aggregate_v1.json"),
        },
    }
    result["content_sha256_before_self_field"] = oracle.canonical_sha256_v1(result)
    oracle.validate_preregistration_v1(result)
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
            raise RuntimeError("v1 checked-in structured preregistration is stale")
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
