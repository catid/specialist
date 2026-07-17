#!/usr/bin/env python3
"""Build the CPU-only Qwen3.6 collective-compression preregistration.

The builder inspects only local package/source binaries and already-sealed
data-free receipts.  It never imports torch, initializes CUDA, opens a model,
or reads any dataset/evaluation content.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import statistics
from pathlib import Path
from typing import Any

import eggroll_es_collective_compression_v82 as oracle


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_collective_compression_v82.json"
)
SCHEMA_V82 = "qwen36-collective-compression-preregistration-v82"
CREATED_AT_UTC_V82 = "2026-07-17T00:00:00+00:00"

RECIPE_CONTRACT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "recipe_evaluation_compute_contract_v1.json"
)
V71_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "v66d_phase_memory_analysis_v71.json"
)
V71_REPORT = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_v66d_phase_memory_analysis_v71_20260717.md"
)
V68_ARTIFACTS = {
    "experiments/eggroll_es_hpo/runs/"
    "v68_collective_inprocess_paired_12x64/benchmark.json": (
        "eeef875c72dc0bd1a6135e8cd4425e9a7833f5c34a77d8a2714f0077c05af5ff"
    ),
    "experiments/eggroll_es_hpo/runs/"
    "v68_collective_inprocess_paired_12x64_r2/benchmark.json": (
        "e4259651d63a676f00ca17427c135ab6d7c209ab93793855df50322b18fbfa64"
    ),
}

EXPECTED_FILE_SHA256 = {
    "recipe_contract": (
        "04af81499067e2feb0186c0a61e4c1af10f838a8eb7deec6dd41cd192748cacf"
    ),
    "v71_prereg": (
        "7aa416ae736585f3b3710066faed2ac93ea178dc3b8fa93f5430cc83d7a3f1d4"
    ),
    "v71_report": (
        "eb7b07ceccef4f6c61eedd2e2e0d92b1487b3c17cfdd31cfdde4518c710c5076"
    ),
    "torch_version_py": (
        "323d35171ef1184f1d7db3bbd1f3d3e227e0e826be8fd52200778346e17c873f"
    ),
    "torch_nccl_utils_hpp": (
        "d4639a2a83e45347d4f21331e03245b3d9a67a43ce317081f3770a22c067e546"
    ),
    "torch_cuda_nccl_h": (
        "a12040a73d2fed50d0b9242fa93766b16b53dd679ef9c09e8165fbdee388de4a"
    ),
    "libtorch_cuda_so": (
        "fd13a41b54fe3d8af91075962d9fe155a16616ca09c7b79b7251869ff6230d32"
    ),
    "nccl_header": (
        "06b934792a70f08cafabd96b8f94015b143a182fbb3bc56ac71c4c6de7deb971"
    ),
    "nccl_library": (
        "1c8618b866734cbdd5401715d6178be763ece283b7f808ecf86dedab211162c1"
    ),
}
EXPECTED_RECIPE_CONTENT_SHA256 = (
    "2442c0c2be3ac4c883612f400f8f213ce3bc82ef96e03fad1ef10ec3b7d11fad"
)
EXPECTED_V71_CONTENT_SHA256 = (
    "e8c2ecc2fafc2ded90ff7cde57f028f83720725a38692865bf1d01f8f26427de"
)
RUNTIME_VECTOR_SHA256 = (
    "7871fe2d020537cebb72053b8c3866b7b1d4296fee284833ac0b18c8c6a5c240"
)
ORACLE_FILE_SHA256_V82 = (
    "1b205a846dde09da73f5f36477f68e643f0d2f3ae89b16eb7918db46db03022d"
)
PREFLIGHT_FILE_SHA256_V82 = (
    "24a2b8d94d79845e54c7608869feca72fe2e5c0dbdf00a46110a1a45f18b3888"
)
ORACLE_PATH_V82 = ROOT / "eggroll_es_collective_compression_v82.py"
PREFLIGHT_PATH_V82 = ROOT / "benchmark_eggroll_es_collective_compression_v82.py"
FUTURE_GATE_PATH_V82 = (
    "experiments/eggroll_es_hpo/gates/"
    "qwen36_collective_residual_bottleneck_gate_v82.json"
)
FUTURE_OUTPUT_PATH_V82 = (
    "experiments/eggroll_es_hpo/runs/"
    "v82_collective_compression_data_free_preflight/preflight.json"
)


def file_sha256_v82(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_v82(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_json_v82(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise RuntimeError(f"duplicate JSON key in {path}: {key}")
            value[key] = item
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    _require_v82(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _contains_bytes_v82(path: Path, needles: tuple[bytes, ...]) -> dict[str, bool]:
    found = {needle: False for needle in needles}
    overlap = max(len(needle) for needle in needles) - 1
    tail = b""
    with path.open("rb") as source:
        while True:
            block = source.read(8 << 20)
            if not block:
                break
            surface = tail + block
            for needle in needles:
                if needle in surface:
                    found[needle] = True
            tail = surface[-overlap:] if overlap else b""
    return {needle.decode("ascii"): value for needle, value in found.items()}


def inspect_local_collective_capabilities_v82() -> dict[str, Any]:
    torch_dist = importlib.metadata.distribution("torch")
    nccl_dist = importlib.metadata.distribution("nvidia-nccl-cu13")
    site = Path(torch_dist.locate_file(""))
    paths = {
        "torch_version_py": site / "torch/version.py",
        "torch_nccl_utils_hpp": site / (
            "torch/include/torch/csrc/distributed/c10d/NCCLUtils.hpp"
        ),
        "torch_cuda_nccl_h": site / "torch/include/torch/csrc/cuda/nccl.h",
        "libtorch_cuda_so": site / "torch/lib/libtorch_cuda.so",
        "nccl_header": site / "nvidia/nccl/include/nccl.h",
        "nccl_library": site / "nvidia/nccl/lib/libnccl.so.2",
    }
    hashes = {}
    for label, path in paths.items():
        _require_v82(path.is_file() and not path.is_symlink(), f"missing local {label}")
        actual = file_sha256_v82(path)
        _require_v82(
            actual == EXPECTED_FILE_SHA256[label],
            f"local {label} identity changed",
        )
        hashes[label] = {"path": str(path), "file_sha256": actual}

    version_text = paths["torch_version_py"].read_text(encoding="utf-8")
    torch_header = paths["torch_nccl_utils_hpp"].read_text(encoding="utf-8")
    torch_cuda_nccl_header = paths["torch_cuda_nccl_h"].read_text(
        encoding="utf-8"
    )
    nccl_header = paths["nccl_header"].read_text(encoding="utf-8")
    binary_strings = _contains_bytes_v82(
        paths["libtorch_cuda_so"],
        (
            b"Unsupported Float8 type for NCCL reduction",
            b"Input tensor data type is not supported for NCCL process group",
        ),
    )
    _require_v82(
        torch_dist.version == "2.11.0"
        and nccl_dist.version == "2.28.9"
        and "__version__ = '2.11.0+cu130'" in version_text
        and "cuda: Optional[str] = '13.0'" in version_text,
        "local PyTorch/CUDA/NCCL version changed",
    )
    _require_v82(
        "{at::kBFloat16, ncclBfloat16}" in torch_header
        and "#if HAS_NCCL_BF16_DATATYPE" in torch_header
        and "NCCL_MINOR >= 10" in torch_cuda_nccl_header
        and "#ifdef NCCL_SUPPORTS_FP8" in torch_header,
        "installed PyTorch NCCL dtype mapping changed",
    )
    _require_v82(
        "#define NCCL_MAJOR 2" in nccl_header
        and "#define NCCL_MINOR 28" in nccl_header
        and "#define NCCL_PATCH 9" in nccl_header
        and "ncclBfloat16   = 9" in nccl_header
        and "ncclFloat8e4m3 = 10" in nccl_header
        and "ncclFloat8e5m2 = 11" in nccl_header
        and all(binary_strings.values()),
        "installed NCCL ABI or PyTorch runtime rejection evidence changed",
    )
    return {
        "schema": "local-pytorch-nccl-collective-capability-evidence-v82",
        "torch_distribution_version": torch_dist.version,
        "torch_runtime_version_from_bound_source": "2.11.0+cu130",
        "torch_git_version_from_bound_source": (
            "70d99e998b4955e0049d13a98d77ae1b14db1f45"
        ),
        "cuda_build_version_from_bound_source": "13.0",
        "nccl_distribution_version": nccl_dist.version,
        "bound_files": hashes,
        "bf16": {
            "pytorch_maps_bfloat16_to_nccl_bfloat16": True,
            "nccl_abi_declares_bfloat16": True,
            "ordinary_all_reduce_arm_registered": True,
            "live_four_gpu_preflight_still_required": True,
        },
        "fp8": {
            "nccl_abi_declares_e4m3_and_e5m2": True,
            "pytorch_mapping_is_compile_time_conditional": True,
            "installed_pytorch_binary_rejects_float8_reduction": True,
            "rejection_message": "Unsupported Float8 type for NCCL reduction",
            "ordinary_all_reduce_arm_registered": False,
            "launch_forbidden": True,
            "reason": (
                "An NCCL enum is not end-to-end PyTorch collective support; "
                "the installed ProcessGroupNCCL binary explicitly rejects "
                "Float8 reduction."
            ),
        },
        "torch_or_cuda_imported": False,
        "gpu_access_performed": False,
    }


def inspect_v68_layout_evidence_v82() -> dict[str, Any]:
    artifacts = []
    run_medians = []
    for relative, expected_sha in V68_ARTIFACTS.items():
        path = ROOT / relative
        actual_sha = file_sha256_v82(path)
        _require_v82(actual_sha == expected_sha, "V68 artifact identity changed")
        value = _load_json_v82(path)
        results = value.get("results")
        _require_v82(
            value.get("schema")
            == "eggroll-es-in-process-paired-collective-layout-v68"
            and value.get("backend") == "nccl"
            and value.get("world_size") == oracle.WORLD_SIZE_V82
            and value.get("elements_per_rank") == oracle.TOTAL_ELEMENTS_V82
            and value.get("bytes_per_layout_per_rank") == 571_998_208
            and value.get("both_layouts_resident_bytes_per_rank") == 1_143_996_416
            and value.get("pair_count") == 12
            and value.get("iterations_per_block") == 64
            and value.get("warmup_iterations_per_layout") == 5
            and value.get("runtime_parameter_count") == 23
            and value.get("runtime_parameter_elements_sha256")
            == RUNTIME_VECTOR_SHA256
            and value.get("data_or_model_opened") is False
            and isinstance(results, list)
            and len(results) == 4
            and {row.get("rank") for row in results} == {0, 1, 2, 3}
            and {row.get("gpu") for row in results} == {0, 1, 2, 3},
            "V68 layout evidence changed",
        )
        run_median = statistics.median(
            float(row["summary"]["native_over_flat_median_speed"])
            for row in results
        )
        run_medians.append(run_median)
        artifacts.append({"path": relative, "file_sha256": actual_sha})
    _require_v82(
        run_medians == [1.003526527791839, 0.9977076489242296],
        "V68 paired medians changed",
    )
    return {
        "schema": "v68-native-boundary-decision-v82",
        "artifacts": artifacts,
        "paired_run_native_over_flat_median_speeds": run_medians,
        "decision": "retain_native_23_tensor_boundaries",
        "flat_shadow_bytes_per_rank": 571_998_208,
        "flat_shadow_mib_per_rank": 545.5,
        "flat_shadow_forbidden_for_v82": True,
    }


def inspect_v66d_phase_evidence_v82() -> dict[str, Any]:
    recipe = _load_json_v82(RECIPE_CONTRACT)
    v71 = _load_json_v82(V71_PREREG)
    _require_v82(
        file_sha256_v82(RECIPE_CONTRACT) == EXPECTED_FILE_SHA256["recipe_contract"]
        and recipe.get("content_sha256_before_self_field")
        == EXPECTED_RECIPE_CONTENT_SHA256,
        "recipe evaluation contract identity changed",
    )
    _require_v82(
        file_sha256_v82(V71_PREREG) == EXPECTED_FILE_SHA256["v71_prereg"]
        and v71.get("content_sha256_before_self_field")
        == EXPECTED_V71_CONTENT_SHA256
        and file_sha256_v82(V71_REPORT) == EXPECTED_FILE_SHA256["v71_report"],
        "V71 phase evidence identity changed",
    )
    return {
        "schema": "v66d-phase-collective-evidence-v82",
        "artifacts": {
            "recipe_contract": {
                "path": str(RECIPE_CONTRACT.relative_to(ROOT)),
                "file_sha256": EXPECTED_FILE_SHA256["recipe_contract"],
                "content_sha256": EXPECTED_RECIPE_CONTENT_SHA256,
            },
            "v71_preregistration": {
                "path": str(V71_PREREG.relative_to(ROOT)),
                "file_sha256": EXPECTED_FILE_SHA256["v71_prereg"],
                "content_sha256": EXPECTED_V71_CONTENT_SHA256,
            },
            "v71_report": {
                "path": str(V71_REPORT.relative_to(ROOT)),
                "file_sha256": EXPECTED_FILE_SHA256["v71_report"],
            },
        },
        "immutable_v66d_observation": {
            "update_execute_observed_span_seconds": 4.202,
            "update_execute_upper_window_seconds": 4.622,
            "update_execute_sampled_pcie_rx_mib": 8_075.28,
            "update_execute_sampled_pcie_tx_mib": 1_496.05,
            "all_phase_sampled_pcie_rx_bytes": 9_421_838_461,
            "all_phase_sampled_pcie_tx_bytes": 15_099_460_768,
            "logical_reduced_fp32_return_bytes_all_actors": 72_450_048,
            "interpretation": (
                "The burst is consistent with collective traffic but is a "
                "phase-aligned NVML sampling estimate, not link-level NCCL bytes."
            ),
        },
        "historical_signal_only": True,
        "does_not_authorize_v82_launch": True,
        "required_next_evidence": (
            "Post-V72 fusion and host-copy reduction profile with exact "
            "collective time, link bytes, HBM bytes, and residual rank."
        ),
    }


def build_preregistration_v82() -> dict[str, Any]:
    raise RuntimeError(
        "V82 preregistration is historical and bound to quarantined evaluation "
        "V1; create a V2 successor"
    )
    _require_v82(
        file_sha256_v82(ORACLE_PATH_V82) == ORACLE_FILE_SHA256_V82
        and file_sha256_v82(PREFLIGHT_PATH_V82) == PREFLIGHT_FILE_SHA256_V82,
        "V82 oracle or prospective preflight identity changed",
    )
    local = inspect_local_collective_capabilities_v82()
    layout = inspect_v68_layout_evidence_v82()
    prior = inspect_v66d_phase_evidence_v82()
    byte_accounting = oracle.collective_byte_accounting_v82()
    body = {
        "schema": SCHEMA_V82,
        "status": "sealed_cpu_contract_live_launch_dependency_blocked",
        "created_at_utc": CREATED_AT_UTC_V82,
        "bead": "specialist-0j5.28",
        "purpose": (
            "Conditionally compare the untouched FP32 ES update collective "
            "with native-boundary BF16 communication plus deterministic FP32 "
            "local residual error feedback."
        ),
        "authority": {
            "gpu_launch_authorized_at_build": False,
            "dataset_or_model_opened": False,
            "protected_dev_ood_or_holdout_opened": False,
            "training_or_checkpoint_promotion_authorized": False,
            "ordinary_fp8_all_reduce_authorized": False,
        },
        "local_capabilities": local,
        "implementation_bindings": {
            "cpu_oracle": {
                "path": ORACLE_PATH_V82.name,
                "file_sha256": ORACLE_FILE_SHA256_V82,
                "imports_torch_or_cuda": False,
            },
            "prospective_data_free_preflight": {
                "path": PREFLIGHT_PATH_V82.name,
                "file_sha256": PREFLIGHT_FILE_SHA256_V82,
                "imports_torch_only_after_authorized_gate": True,
                "gate_receipt_path": FUTURE_GATE_PATH_V82,
                "output_path": FUTURE_OUTPUT_PATH_V82,
                "exact_future_command": (
                    "CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_DEBUG=INFO "
                    ".venv/bin/torchrun --standalone --nnodes=1 "
                    "--nproc-per-node=4 "
                    "benchmark_eggroll_es_collective_compression_v82.py "
                    f"--gate {FUTURE_GATE_PATH_V82} "
                    f"--output {FUTURE_OUTPUT_PATH_V82}"
                ),
                "command_authorized_at_build": False,
                "why_blocked": (
                    "The gate receipt does not exist; .14/.18/.19 have not "
                    "closed and .32 has not produced a post-optimization "
                    "top-three collective rank."
                ),
            },
        },
        "prior_evidence": {
            "layout": layout,
            "phase_profile": prior,
        },
        "fixed_update_surface": {
            "world_size": oracle.WORLD_SIZE_V82,
            "physical_gpus": [0, 1, 2, 3],
            "native_tensor_count": len(oracle.RUNTIME_PARAMETER_ELEMENTS_V82),
            "native_tensor_elements": list(oracle.RUNTIME_PARAMETER_ELEMENTS_V82),
            "native_tensor_elements_sha256": RUNTIME_VECTOR_SHA256,
            "elements_per_rank": oracle.TOTAL_ELEMENTS_V82,
            "canonical_master_dtype": "float32",
            "optimizer_dtype": "float32",
            "candidate_inference_precision_unchanged_between_arms": True,
            "antithetic_estimator": (
                "learning_rate/(2*N*sigma)*sum((Rplus-Rminus)*epsilon)"
            ),
            "strided_seed_sharding_unchanged": True,
            "native_tensor_order_unchanged": True,
            "flat_shadow_forbidden": True,
        },
        "arms": {
            "fp32_control": {
                "communication_dtype": "float32",
                "implementation": "existing_in_place_PyNccl_all_reduce",
                "residual_state": "absent_and_must_be_all_positive_zero",
                "conversion_or_rescaling_before_collective": False,
                "bit_exact_fallback_required": True,
            },
            "bf16_error_feedback": {
                "communication_dtype": "bfloat16",
                "input_and_residual_dtype": "float32",
                "rounding": "IEEE_BF16_round_to_nearest_even",
                "local_equation": (
                    "compensated=fp32(update+residual); q=bf16(compensated); "
                    "next_residual=fp32(compensated-fp32(q))"
                ),
                "collective": "ordinary_sum_all_reduce_over_native_tensor",
                "post_collective": (
                    "copy BF16 result to CPU, convert and scale in FP32, then "
                    "apply to the canonical FP32 master"
                ),
                "two_fp32_residual_banks_during_transaction": True,
                "residual_checkpointed_atomically_with_master": True,
                "unknown_or_partial_collective": "exact_restore_or_terminal_poison",
                "naive_full_fp32_decode_staging_forbidden": True,
            },
            "fp8_ordinary_all_reduce": {
                "registered": False,
                "launch_authorized": False,
                "reason": (
                    "Installed ProcessGroupNCCL contains an explicit Float8 "
                    "reduction rejection; NCCL datatype enums alone are not "
                    "end-to-end support."
                ),
            },
        },
        "byte_accounting": byte_accounting,
        "dependency_gate": {
            "schema": oracle.GATE_SCHEMA_V82,
            "required_closed_acceptance_beads": [
                "specialist-0j5.14",
                "specialist-0j5.18",
                "specialist-0j5.19",
                "specialist-0j5.32",
            ],
            "must_measure_after": ["specialist-0j5.18", "specialist-0j5.19"],
            "required_measurements": [
                "unoverlapped phase wall time",
                "collective link bytes",
                "collective time",
                "HBM bytes",
                "four-GPU attribution",
            ],
            "launch_condition": "update_collective_rank_in_top_three",
            "non_launch_condition": (
                "If update_collective ranks below third, close specialist-0j5.28 "
                "as not applicable without consuming a GPU ablation."
            ),
            "historical_v66d_signal_is_insufficient": True,
        },
        "execution_contract_after_gate": {
            "stage_1_data_free_preflight": {
                "three_registered_seeds": list(oracle.REGISTERED_SEEDS_V82),
                "counterbalanced_same_process_pairs": True,
                "same_fp32_local_accumulators": True,
                "same_native_23_tensor_order": True,
                "report": [
                    "measured link bytes", "collective time", "HBM bytes",
                    "peak residual/staging VRAM", "updates per second",
                    "all-four-GPU useful attribution", "cleanup idle",
                ],
                "advance_only_if_end_to_end_update_throughput_improves": True,
                "payload_size_alone_cannot_pass": True,
            },
            "stage_2_three_seed_recipe_confirmation": {
                "training_seeds": list(oracle.REGISTERED_SEEDS_V82),
                "common_random_numbers": True,
                "compute_mode": "estimator_control_then_compute_matched_confirmation",
                "source_disjoint_dev_required": True,
                "protected_holdout_opened": False,
                "dev_pooled_paired_95_lcb_minimum": 0.0,
                "positive_dev_seeds_minimum": 2,
                "ood_qa_reward_95_lcb_minimum": -0.02,
                "ood_qa_exact_count_delta_minimum": -1,
                "ood_prose_logprob_95_lcb_minimum": -0.02,
                "all_ood_conditions_required": True,
            },
        },
        "transaction_and_integrity": {
            "oracle_schema": oracle.SCHEMA_V82,
            "residual_state_schema": oracle.STATE_SCHEMA_V82,
            "transaction_schema": oracle.TRANSACTION_SCHEMA_V82,
            "prepare_does_not_publish_residual": True,
            "candidate_master_and_candidate_residual_commit_atomically": True,
            "resume_replays_identical_transmitted_and_residual_bits": True,
            "rollback_accepts_only_before_or_candidate_generation": True,
            "one_element_residual_or_order_corruption_fails_closed": True,
            "local_q_plus_next_residual_reconstructs_compensated_fp32_bits": True,
            "NCCL_reduction_order_not_claimed_bit_deterministic": True,
            "rank_local_residual_receipts_required": True,
            "rank_local_residuals_not_required_to_be_identical": True,
            "cross_rank_reduced_result_receipts_required": True,
        },
        "promotion_rule": {
            "systems": (
                "Measured end-to-end update throughput must improve after "
                "including fused quantization, residual HBM traffic, staging, "
                "collective, D2H, and CPU FP32 conversion."
            ),
            "capacity": (
                "Measured peak plus reserved headroom must fit on every GPU; "
                "the 1.11-GiB two-bank residual transaction cannot be hidden."
            ),
            "correctness": (
                "Finite bounded error, exact local conservation, atomic resume/"
                "rollback, exact FP32 control, and exact restore-or-poison."
            ),
            "quality": "All three-seed dev and OOD gates must pass.",
            "otherwise": "retain_fp32_control",
        },
    }
    return {
        **body,
        "content_sha256_before_self_field": oracle.canonical_sha256_v82(body),
    }


def validate_preregistration_v82(value: dict[str, Any]) -> dict[str, Any]:
    raise RuntimeError(
        "V82 historical preregistration is nonpromotable after V1 quarantine"
    )
    expected = build_preregistration_v82()
    _require_v82(value == expected, "V82 preregistration contract changed")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    built = build_preregistration_v82()
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(
            json.dumps(built, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.check:
        validate_preregistration_v82(_load_json_v82(OUTPUT))
    if not args.write and not args.check:
        print(json.dumps(built, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
