#!/usr/bin/env python3
"""Build/check the additive V81A CUDA integration preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
from pathlib import Path
from typing import Any, Mapping

import eggroll_es_adapter_transport_precision_v81 as transport_v81
import run_qwen36_adapter_transport_paired_v81a as runner_v81a


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_adapter_transport_cuda_v81a.json"
)

WORKER_V41A = ROOT / "eggroll_es_worker_lora_v41a.py"
WORKER_V66 = ROOT / "eggroll_es_worker_lora_v66.py"
WORKER_V66D = ROOT / "eggroll_es_worker_lora_v66d.py"
WORKER_V71 = ROOT / "eggroll_es_worker_lora_v71.py"
WORKER_V72 = ROOT / "eggroll_es_worker_lora_v72.py"
AUDIT_V71 = ROOT / "eggroll_es_audit_contract_v71.py"
HOST_V72 = ROOT / "eggroll_es_host_state_contract_v72.py"
V73_ADAPTER = ROOT / "run_lora_es_v71_v72_live_calibration_v73.py"
V73B_BUILDER = ROOT / "build_lora_es_v71_v72_same_live_preregistration_v73b.py"
V73B_RUNNER = ROOT / "run_lora_es_v71_v72_same_live_calibration_v73b.py"
V73B_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "lora_es_v71_v72_same_live_calibration_v73b.json"
)
V73B_POSTRUN = ROOT / (
    "experiments/eggroll_es_hpo/"
    "qwen36_v73b_lora_calibration_postrun_20260717.json"
)
V73B_RUN = ROOT / (
    "experiments/eggroll_es_hpo/runs/"
    "v73b_lora_es_same_live_qwen36_calibration"
)
V73B_REPORT = V73B_RUN / "mirrored_calibration_report_v73b.json"
V73B_ACTOR_RECEIPTS = V73B_RUN / "actor_cuda_work_receipts_v73b.jsonl"
V73B_AUDIT = V73B_RUN / "exact_audit_traffic_v73b.json"

V81_IMPLEMENTATION = ROOT / "eggroll_es_adapter_transport_precision_v81.py"
V81_BUILDER = ROOT / "build_adapter_transport_precision_preregistration_v81.py"
V81_TEST = ROOT / "test_eggroll_es_adapter_transport_precision_v81.py"
V81_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_adapter_transport_precision_v81.json"
)

IMPLEMENTATION = ROOT / "eggroll_es_worker_lora_pinned_transport_v81a.py"
RUNNER = ROOT / "run_qwen36_adapter_transport_paired_v81a.py"
WORKER_TEST = ROOT / "test_eggroll_es_worker_lora_pinned_transport_v81a.py"
RUNNER_TEST = ROOT / "test_run_qwen36_adapter_transport_paired_v81a.py"
BUILDER_TEST = ROOT / (
    "test_build_qwen36_adapter_transport_cuda_preregistration_v81a.py"
)
BUILDER = Path(__file__).resolve()

EXPECTED_SEALED_FILES = {
    WORKER_V41A: "cc40337eba30fe0748996c22dcbf3914b8c12249f6e2e47d6128aadee575494c",
    WORKER_V66: "ee976fbc56a720c5c2a5e52d86c7a02d1e8c7414ed7383952d3da03e72944a03",
    WORKER_V66D: "807af52dab4b842f0a74b33cf083f8b67a7e2f4bd0329eb1ca08c4ffb3831ed6",
    WORKER_V71: "6167ecd24332e0384e50f4bfa34893112623c2291f35e89989d8c3a527b9fcaa",
    WORKER_V72: "547d525edfd51412abb3a4980ddc4a55730ad0eb09987ec202ce2ce8f701a2c2",
    AUDIT_V71: "cc80ac0e1bf3c9db83e3275df16ea1479273d92a40240496163543643bd0eaa8",
    HOST_V72: "33bb0312136248cf01afa927c70a1ed22af64f10a26b9668cf3afd9570571e66",
    V73_ADAPTER: "b579ac417d88aa9bf9eead5635d1454045451b6ba9fbe383569f4b2e2530260b",
    V73B_BUILDER: "b72dc4a134853484e556f09bc9e5eed22cccc9c5fb7bdb2aff5ec2e73b57669f",
    V73B_RUNNER: "8e6b00adcfcdb81c81a9bb78032e993114203660af526026350a298a9a4320db",
    V73B_PREREG: "9c5ce43c36e08e038ee33e86380ab6c287ae1b4bcba4c80def623daeb00f7ed9",
    V73B_POSTRUN: "d65be30ec769e8a18ce75a3ffc0aab5624cec35f909732e8082a4836c63140c4",
    V73B_REPORT: "ba1b0a76dd0a9955b5e3f779d1ef440037b12b4689b3b1ab640d1ee1a4cff44a",
    V73B_ACTOR_RECEIPTS: "30df82d21b28c7d5c94ede785c69c10896bad9d07db52ccf065e3031181d7013",
    V73B_AUDIT: "388fb6f544254c94e0c0ae11956932757834894103784f9e43d6b76e1bb3cb20",
    V81_IMPLEMENTATION: "08e8b74cd8a79ab7615a89877e184cbea50e8ddfbd95bdbfefccc50c13f28bfc",
    V81_BUILDER: "34695c55b8ad098ac46ba02554de96b856648b8f7e2babe47c563c3b1c7b2405",
    V81_TEST: "cd40d20d45b1b4029ba034a72112dfa9b987d09c2c68858d5ded743c99da3392",
    V81_PREREG: "cb8f31d981d4d3471adbfb78d962fb7ae028e56e7364013874dcbc0d181e1c25",
}

EXPECTED_JSON = {
    V73B_PREREG: (
        "lora-es-v71-v72-qwen36-same-live-calibration-preregistration-v73b",
        "50b51ee2d71dc85d024c4a63cc57183d2b9c2d925c18924a719dacb1fb61dc94",
    ),
    V73B_POSTRUN: (
        "qwen36-v73b-lora-calibration-postrun-analysis-v1",
        "21689f75ecaaf583aedde50ad293ce3a9b5644009d62c2bc4624637280a651e7",
    ),
    V73B_REPORT: (
        "v73b-v71-v72-qwen36-calibration-report",
        "61051af180657f41459a836fcb09b48019d6adae8d63be2817181c727953f0e8",
    ),
    V73B_AUDIT: (
        "eggroll-es-four-actor-audit-traffic-v73",
        "04768ffc11cf99ea522730ca5fbf48dbee48e169482466640c3a42a622da999e",
    ),
    V81_PREREG: (
        "qwen36-adapter-transport-precision-preregistration-v81",
        "db01b11ca0fc1565cc81a00cd0a6426f80ae840f907eb0446fda5fb0908eeb80",
    ),
}


def file_sha256_v81a(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_v81a(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def _bound_v81a(path: Path) -> dict[str, str]:
    return {
        "path": _relative_v81a(path),
        "file_sha256": file_sha256_v81a(path),
    }


def _load_self_hashed_v81a(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    schema, expected_content = EXPECTED_JSON[path]
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("schema") != schema
        or value.get("content_sha256_before_self_field") != expected_content
        or transport_v81.canonical_sha256_v81(compact) != expected_content
    ):
        raise RuntimeError(f"v81a sealed JSON prerequisite changed: {path}")
    return value


def _validate_accepted_lifecycle_v81a(
    postrun: Mapping[str, Any],
    report: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> None:
    if (
        postrun.get("passed") is not True
        or postrun.get("status") != "accepted_same_live_v71_v72_lora_gate"
        or postrun.get("safety", {}).get("final_master_and_runtime_exact")
        is not True
        or postrun.get("safety", {}).get("all_four_update_aborts_exact")
        is not True
        or postrun.get("cleanup", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or postrun.get("authority", {}).get(
            "protected_dev_ood_or_holdout_opened"
        ) is not False
        or report.get("all_v71_rewards_accepted_before_update_math") is not True
        or report.get("all_v72_state_generations_one_or_two_bank_bounded")
        is not True
        or report.get("same_live_compiler_output_whole_mapping_exact") is not True
        or report.get("historical_reward_floats_used_as_acceptance_gate")
        is not False
        or report.get("checkpoint_snapshot_or_promotion_performed") is not False
        or report.get("protected_dev_ood_or_holdout_opened") is not False
        or report.get("final_gpu_idle", {}).get(
            "all_four_compute_process_lists_empty"
        ) is not True
        or audit.get("exact_match") is not True
        or len(audit.get("by_rank", [])) != 4
    ):
        raise RuntimeError("v81a accepted V71/V72/V73B lifecycle changed")


def _validate_new_sources_v81a() -> None:
    for path in (
        IMPLEMENTATION, RUNNER, WORKER_TEST, RUNNER_TEST, BUILDER_TEST
    ):
        if not path.is_file():
            raise RuntimeError(f"v81a additive source is absent: {path}")
    worker = IMPLEMENTATION.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    if (
        "class LoRAAdapterStateWorkerExtensionV81A("
        "LoRAAdapterStateWorkerExtensionV72)" not in worker
        or "pin_memory=True" not in worker
        or "view.copy_(expected[key], non_blocking=True)" not in worker
        or "consumer.wait_event(event)" not in worker
        or "event.synchronize()" not in worker
        or "expected_value.to(device=view.device)" in worker
        or ".to(device=view.device" in worker
        or "temporary_device_publication_staging_bytes\": 0" not in worker
        or "device_to_device_copy_bytes\": 0" not in worker
        or "final_transport_receipt_v81a" not in worker
        or "transport_status_receipt_v81a" not in worker
    ):
        raise RuntimeError("v81a additive direct-runtime worker surface changed")
    if (
        "import recipe_evaluation_contract_v1" in runner
        or "from recipe_evaluation_contract_v1" in runner
        or "PAIR_ORDER_V81A" not in runner
        or "execute_prospective_schedule_v81a" not in runner
        or "systems_only_nonpromotable" not in runner
        or "v2_bound_quality_successor" not in runner
    ):
        raise RuntimeError("v81a prospective runner boundary changed")


def build_preregistration_v81a() -> dict[str, Any]:
    for path, expected in EXPECTED_SEALED_FILES.items():
        observed = file_sha256_v81a(path)
        if observed != expected:
            raise RuntimeError(
                f"v81a sealed source changed: {_relative_v81a(path)}: "
                f"{observed} != {expected}"
            )
    v73b_prereg = _load_self_hashed_v81a(V73B_PREREG)
    postrun = _load_self_hashed_v81a(V73B_POSTRUN)
    report = _load_self_hashed_v81a(V73B_REPORT)
    audit = _load_self_hashed_v81a(V73B_AUDIT)
    v81 = _load_self_hashed_v81a(V81_PREREG)
    _validate_accepted_lifecycle_v81a(postrun, report, audit)
    _validate_new_sources_v81a()
    plan = v81["transport_plan"]
    if (
        plan.get("canonical_authority", {}).get("dtype") != "float32"
        or plan.get("canonical_authority", {}).get("location") != "cpu"
        or plan.get("execution_view", {}).get("resolved_dtype") != "bfloat16"
        or plan.get("execution_view", {}).get("view_count") != 82
        or plan.get("execution_view", {}).get("persistent_device_bytes")
        != 9_842_688
        or plan.get("challenger_pinned_direct", {}).get(
            "host_pinned_bank_bytes"
        ) != 9_842_688
    ):
        raise RuntimeError("v81a V81 transport geometry changed")

    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    aggregate_pinned = 4 * transport_v81.EXPECTED_RUNTIME_BYTES_V81
    if (
        soft != resource.RLIM_INFINITY
        and int(soft) < transport_v81.EXPECTED_RUNTIME_BYTES_V81
    ):
        raise RuntimeError("v81a inherited per-process memlock limit is too small")

    accepted_bindings = {
        "worker_v41a": _bound_v81a(WORKER_V41A),
        "worker_v66": _bound_v81a(WORKER_V66),
        "worker_v66d": _bound_v81a(WORKER_V66D),
        "worker_v71": _bound_v81a(WORKER_V71),
        "worker_v72": _bound_v81a(WORKER_V72),
        "audit_contract_v71": _bound_v81a(AUDIT_V71),
        "host_ownership_contract_v72": _bound_v81a(HOST_V72),
        "v73_adapter": _bound_v81a(V73_ADAPTER),
        "v73b_builder": _bound_v81a(V73B_BUILDER),
        "v73b_runner": _bound_v81a(V73B_RUNNER),
        "v73b_preregistration": {
            **_bound_v81a(V73B_PREREG),
            "content_sha256": v73b_prereg[
                "content_sha256_before_self_field"
            ],
        },
        "v73b_postrun": {
            **_bound_v81a(V73B_POSTRUN),
            "content_sha256": postrun["content_sha256_before_self_field"],
        },
        "v73b_report": {
            **_bound_v81a(V73B_REPORT),
            "content_sha256": report["content_sha256_before_self_field"],
        },
        "v73b_actor_receipts": _bound_v81a(V73B_ACTOR_RECEIPTS),
        "v73b_exact_audit": {
            **_bound_v81a(V73B_AUDIT),
            "content_sha256": audit["content_sha256_before_self_field"],
        },
        "v81_cpu_contract": {
            **_bound_v81a(V81_PREREG),
            "content_sha256": v81["content_sha256_before_self_field"],
        },
        "v81_cpu_implementation": _bound_v81a(V81_IMPLEMENTATION),
        "v81_cpu_builder": _bound_v81a(V81_BUILDER),
        "v81_cpu_test": _bound_v81a(V81_TEST),
    }
    implementation_bindings = {
        "cuda_worker_subclass": _bound_v81a(IMPLEMENTATION),
        "prospective_pair_runner": _bound_v81a(RUNNER),
        "cuda_worker_tests": _bound_v81a(WORKER_TEST),
        "pair_runner_tests": _bound_v81a(RUNNER_TEST),
        "preregistration_tests": _bound_v81a(BUILDER_TEST),
        "preregistration_builder": _bound_v81a(BUILDER),
    }
    result = {
        "schema": "qwen36-adapter-transport-cuda-preregistration-v81a",
        "status": "sealed_cpu_cuda_integration_before_live_gate",
        "bead": "specialist-0j5.26",
        "purpose": (
            "Add one pinned BF16 direct-runtime publication path beneath the "
            "accepted V71 exact-audit and V72 ownership lifecycle, and freeze a "
            "systems-only four-pair comparison that cannot launch before V73C."
        ),
        "authority": {
            "cpu_source_build_and_oracle_tests": True,
            "cuda_integration_source_created": True,
            "gpu_launch": False,
            "model_ray_or_training_launch": False,
            "dataset_or_protected_content_access": False,
            "quality_or_hpo_promotion": False,
            "checkpoint_model_update_or_recipe_promotion": False,
            "accepted_v71_v72_v73b_source_mutation": False,
            "vllm_or_site_package_mutation": False,
        },
        "bindings": {
            "accepted_lifecycle": accepted_bindings,
            "additive_implementation": implementation_bindings,
        },
        "accepted_lifecycle_contract": {
            "worker_inheritance": "V81A -> V72 -> V71 -> V66D",
            "v71_exact_runtime_base_master_audits_retained": True,
            "v72_host_bank_ownership_one_two_one_retained": True,
            "v73b_same_live_dual_compiler_and_abort_retained": True,
            "restore_unknown_or_partial_state": "exact_master_or_terminal_poison",
            "sole_resident_vllm_adapter_slot_count": 1,
            "final_cleanup_idle_required": True,
            "historical_reward_floats_are_not_cross_run_bitwise_oracles": True,
        },
        "canonical_authority": {
            "location": "cpu",
            "dtype": "float32",
            "tensor_count": 70,
            "elements": 4_528_128,
            "bytes_per_actor": 18_112_512,
            "roles": [
                "canonical_master",
                "noise_and_update_arithmetic",
                "optimizer_authority",
                "checkpoint_authority",
            ],
            "execution_transport_may_mutate_these_roles": False,
        },
        "cuda_integration": {
            "subclass": (
                "eggroll_es_worker_lora_pinned_transport_v81a."
                "LoRAAdapterStateWorkerExtensionV81A"
            ),
            "lazy_first_materialization_initialization": True,
            "pinned_host_bank_count_per_actor": 1,
            "pinned_host_bank_dtype": "torch.bfloat16",
            "pinned_host_bank_elements": 4_921_344,
            "pinned_host_bank_bytes_per_actor": 9_842_688,
            "pinned_host_bank_bytes_all_four_actors": aggregate_pinned,
            "runtime_view_count": 82,
            "runtime_device_elements": 4_921_344,
            "persistent_runtime_device_bytes_per_actor": 9_842_688,
            "h2d_bytes_per_transition": 9_842_688,
            "h2d_calls_per_transition": 82,
            "temporary_device_publication_staging_bytes": 0,
            "device_to_device_payload_bytes": 0,
            "copy_destination": "existing_82_resident_runtime_views",
            "copy_stream_count_per_actor": 1,
            "completion_events_per_transition": 1,
            "consumer_stream_wait_event_before_activation": True,
            "event_synchronize_before_v71_exact_readback": True,
            "event_fence_before_generation": True,
            "bank_reuse_after_legal_retirement_only": True,
            "actual_tensor_is_pinned_runtime_check_required": True,
            "proc_vm_pin_or_lock_receipt_required_but_not_authoritative": True,
            "double_buffering": False,
        },
        "memlock_preflight": {
            "observed_soft_bytes": (
                None if soft == resource.RLIM_INFINITY else int(soft)
            ),
            "observed_hard_bytes": (
                None if hard == resource.RLIM_INFINITY else int(hard)
            ),
            "soft_unlimited": soft == resource.RLIM_INFINITY,
            "hard_unlimited": hard == resource.RLIM_INFINITY,
            "required_bytes_per_actor_process": (
                transport_v81.EXPECTED_RUNTIME_BYTES_V81
            ),
            "four_actor_aggregate_context_bytes": aggregate_pinned,
            "each_actor_inherited_limit_capacity_check_passed": (
                soft == resource.RLIM_INFINITY
                or int(soft) >= transport_v81.EXPECTED_RUNTIME_BYTES_V81
            ),
            "four_bank_sum_below_reported_limit_context": (
                soft == resource.RLIM_INFINITY or int(soft) >= aggregate_pinned
            ),
            "limit_is_not_proof_of_successful_pinning": True,
            "live_each_actor_tensor_is_pinned_required": True,
        },
        "failure_contract": {
            "pageable_bank": "reject_before_copy",
            "partial_view_or_byte_coverage": "reject",
            "temporary_device_or_d2d_path": "not_implemented_and_source_attested",
            "missing_pending_stale_or_foreign_event": "reject_before_audit_or_generation",
            "cross_candidate_reuse": "reject_before_bank_overwrite",
            "copy_exception_or_unknown_device_state": (
                "synchronize_then_exact_master_rewrite_or_terminal_poison"
            ),
            "failed_exact_readback": "inherited_v71_exact_restore_or_poison",
            "install_failure": "synchronize_best_effort_and_release_v81a_state",
            "final_receipt": "exact_master_quiescent_event_complete_bank_released",
        },
        "prospective_pair_run": {
            "current_launch_authorized": False,
            "minimum_counterbalanced_pairs": 3,
            "frozen_pair_count": len(runner_v81a.PAIR_ORDER_V81A),
            "pair_order": [
                list(item) for item in runner_v81a.PAIR_ORDER_V81A
            ],
            "physical_gpu_ids_every_arm": [0, 1, 2, 3],
            "same_long_lived_four_tp1_actor_lifecycle_per_arm": True,
            "control_worker": (
                "eggroll_es_worker_lora_v72."
                "LoRAAdapterStateWorkerExtensionV72"
            ),
            "challenger_worker": (
                "eggroll_es_worker_lora_pinned_transport_v81a."
                "LoRAAdapterStateWorkerExtensionV81A"
            ),
            "only_arm_difference": "candidate_materialization_H2D_D2D_staging",
            "required_measurements": [
                "materialize_restore_update_abort_transition_time",
                "H2D_bytes_and_call_count",
                "publication_D2D_bytes_or_exact_memcpy_trace",
                "peak_allocated_and_reserved_VRAM_per_GPU",
                "PCIe_RX_TX_per_GPU",
                "HBM_activity_with_units_and_exactness_label",
                "all_four_actor_PID_useful_CUDA_attribution",
                "actual_pinned_bank_and_proc_pin_context",
                "exact_candidate_audit_reward_update_abort_final_receipts",
                "final_actor_Ray_GPU_cleanup_idle",
            ],
            "paired_exact_fields": [
                "population_plan_sha256",
                "candidate_projection_set_sha256",
                "final_master_sha256",
                "final_runtime_sha256",
            ],
            "arm_local_not_cross_run_bitwise": [
                "live_reward_float_vector",
                "reward_derived_update_candidate",
            ],
            "systems_only_nonpromotable": True,
        },
        "launch_gate_pending": {
            "required_schema": runner_v81a.GATE_SCHEMA_V81A,
            "required_bead": "specialist-0j5.32",
            "required_v73c_evidence": [
                "complete_nonoverlapping_exact_phase_ranges",
                "all_four_actor_PIDs_and_physical_GPUs",
                "CUDA_kernel_memcpy_NCCL_allocation_phase_attribution",
                "materialize_restore_update_abort_cleanup_coverage",
                "final_idle_and_no_protected_access",
                "profiled_timing_not_used_as_speed_claim",
            ],
            "gate_file_present_now": False,
            "execution_before_additive_sealed_gate": "reject",
        },
        "evaluation_boundary": {
            "this_runner_is_systems_only": True,
            "quality_evaluation_in_this_artifact": False,
            "quarantined_v1_resolved": False,
            "legacy_evaluation_contract_imported": False,
            "quality_or_OOD_promotion_requires": (
                "separate_additive_successor_bound_to_sealed_V2_boundary"
            ),
            "systems_report_can_only_authorize_V2_successor_preregistration": True,
        },
        "scope_exclusions": {
            "fp32_collective_coalescing": "specialist-0j5.36",
            "collective_bucket_or_dtype_change_in_v81a": False,
            "canonical_host_ownership_change": False,
            "noise_or_update_generation_change": False,
            "runtime_slot_layout_change": False,
            "execution_dtype_change": False,
            "persistent_device_VRAM_reduction_claim": False,
            "shared_v71_fused_exact_audit_buffer_change": False,
        },
    }
    result["content_sha256_before_self_field"] = (
        transport_v81.canonical_sha256_v81(result)
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--print", dest="print_output", action="store_true")
    args = parser.parse_args()
    value = build_preregistration_v81a()
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.print_output:
        print(rendered, end="")
        return 0
    if not OUTPUT.is_file() or OUTPUT.read_text(encoding="ascii") != rendered:
        raise SystemExit("v81a preregistration is missing or stale")
    print(
        "v81a preregistration check passed: "
        f"{value['content_sha256_before_self_field']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
