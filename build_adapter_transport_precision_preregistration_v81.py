#!/usr/bin/env python3
"""Build/check the CPU-only Qwen3.6 LoRA transport V81 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import eggroll_es_adapter_transport_precision_v81 as transport


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_adapter_transport_precision_v81.json"
)

V71_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_exact_audit_traffic_v71.json"
)
FUSED_V72_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_fused_structured_runtime_v72.json"
)
HOST_V72_PREREG = ROOT / (
    "experiments/eggroll_es_hpo/preregistrations/"
    "qwen36_host_state_ownership_v72.json"
)

V71_WORKER = ROOT / "eggroll_es_worker_lora_v71.py"
V72_WORKER = ROOT / "eggroll_es_worker_lora_v72.py"
FUSED_V72 = ROOT / "eggroll_es_fused_structured_runtime_v72.py"
IMPLEMENTATION = ROOT / "eggroll_es_adapter_transport_precision_v81.py"
TEST = ROOT / "test_eggroll_es_adapter_transport_precision_v81.py"
BUILDER = Path(__file__).resolve()

VLLM = ROOT / "es-at-scale/.venv/lib/python3.12/site-packages/vllm"
VLLM_VERSION = VLLM / "_version.py"
VLLM_CONFIG = VLLM / "config/lora.py"
VLLM_BASE = VLLM / "lora/layers/base_linear.py"
VLLM_FUSED = VLLM / "lora/layers/fused_moe.py"
VLLM_MANAGER = VLLM / "lora/model_manager.py"

EXPECTED_FILES = {
    V71_PREREG: "8747e9ca3c022b593bdfcf445881106d5410c3496f0135bcd2a663f07ca55240",
    FUSED_V72_PREREG: "ed7d78f2570032d8cd31229f847ce572628ea2347a2bd6b4af4fe28071c1e5df",
    HOST_V72_PREREG: "b748d9eb4b7404b753f29f3cc1ff6827e152e70d44b430aac9f17d026679a398",
    V71_WORKER: "6167ecd24332e0384e50f4bfa34893112623c2291f35e89989d8c3a527b9fcaa",
    V72_WORKER: "547d525edfd51412abb3a4980ddc4a55730ad0eb09987ec202ce2ce8f701a2c2",
    FUSED_V72: "357607f3c16b071f67d2bc3adb0317bbbd29f31f7e1db0cf1aa3030ac997df6e",
    VLLM_VERSION: "3f838c1dd6b9133d59f7895b2934e3ad321403bbeaf88a470c4fc7740c5cd98d",
    VLLM_CONFIG: "031bade79427ba8d1746be968cfa9c343e7c4dda1e7fdb5059a5775cb34874b5",
    VLLM_BASE: "040d75b4a76cb97f28c507453ab8aea68252c35b8b64d9f83ce40d32e2a3a495",
    VLLM_FUSED: "62dcf28f2af0906b420c75bba171dcf4e5a392e02dc9860c32fad88f46b95b7e",
    VLLM_MANAGER: "13201a06e17cccffb30c90bf3d268dfbf901567623b03454003afb5a922ae45a",
}
EXPECTED_CONTENT = {
    V71_PREREG: "14c7afe2fd370798a26641f6950e92592be67e5b2e2e5fabfc442b76462c2f99",
    FUSED_V72_PREREG: "d3144d7f22570951974b8eb366d10e5e4ab9a3d3cc149d44afcb4b2be5c2ee58",
    HOST_V72_PREREG: "95f29e599bfdee505035cf2a4a1182f6f8322d70a52f603a55b79a18c2896ad8",
}
EXPECTED_SCHEMAS = {
    V71_PREREG: "eggroll-es-exact-audit-traffic-contract-v71",
    FUSED_V72_PREREG: "qwen36-fused-structured-runtime-preregistration-v72",
    HOST_V72_PREREG: "eggroll-es-versioned-host-state-contract-v72",
}


def file_sha256_v81(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_v81(path: Path) -> str:
    return str(Path(path).resolve().relative_to(ROOT))


def _load_json_v81(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"v81 JSON source is not an object: {path}")
    return value


def _load_self_hashed_v81(path: Path) -> dict[str, Any]:
    value = _load_json_v81(path)
    compact = dict(value)
    claimed = compact.pop("content_sha256_before_self_field", None)
    if (
        value.get("schema") != EXPECTED_SCHEMAS[path]
        or claimed != EXPECTED_CONTENT[path]
        or transport.canonical_sha256_v81(compact) != claimed
    ):
        raise RuntimeError(f"v81 sealed prerequisite changed: {path}")
    return value


def _bound_file_v81(path: Path) -> dict[str, Any]:
    return {
        "path": _relative_v81(path),
        "file_sha256": file_sha256_v81(path),
    }


def _validate_prerequisite_authority_v81(
    audit: Mapping[str, Any],
    fused: Mapping[str, Any],
    host: Mapping[str, Any],
) -> None:
    if (
        audit.get("rules", {}).get(
            "unknown_or_partial_rpc_requires_exact_restore_or_poison"
        )
        is not True
        or audit.get("rules", {}).get("update_acceptance_after_exact_boundary")
        is not True
        or fused.get("authority", {}).get("gpu_launch") is not False
        or fused.get("authority", {}).get("candidate_or_update_promotion") is not False
        or host.get("gpu_launch_performed") is not False
        or host.get("dataset_or_protected_access_performed") is not False
        or host.get("rules", {}).get(
            "unknown_state_requires_exact_restore_or_poison"
        )
        is not True
    ):
        raise RuntimeError("v81 prerequisite authority changed")


def build_preregistration_v81() -> dict[str, Any]:
    for path, expected in EXPECTED_FILES.items():
        observed = file_sha256_v81(path)
        if observed != expected:
            raise RuntimeError(
                f"v81 sealed source changed: {_relative_v81(path)}: "
                f"{observed} != {expected}"
            )
    version_text = VLLM_VERSION.read_text(encoding="utf-8")
    if "__version__ = version = '0.25.0'" not in version_text:
        raise RuntimeError("v81 installed vLLM version changed")

    audit = _load_self_hashed_v81(V71_PREREG)
    fused = _load_self_hashed_v81(FUSED_V72_PREREG)
    host = _load_self_hashed_v81(HOST_V72_PREREG)
    _validate_prerequisite_authority_v81(audit, fused, host)

    capability = transport.attest_vllm_lora_sources_v81(
        VLLM_CONFIG.read_text(encoding="utf-8"),
        VLLM_BASE.read_text(encoding="utf-8"),
        VLLM_FUSED.read_text(encoding="utf-8"),
        VLLM_MANAGER.read_text(encoding="utf-8"),
    )
    projection = fused["production_projection"]["manifest"]
    plan = transport.build_transport_plan_v81(
        projection,
        capability,
        requested_dtype="auto",
        model_dtype="bfloat16",
    )

    v71_text = V71_WORKER.read_text(encoding="utf-8")
    if (
        "expected_value.to(device=view.device)" not in v71_text
        or "non_blocking=False" not in v71_text
        or "expected[key] = expected_value.to(device=\"cpu\")" not in v71_text
    ):
        raise RuntimeError("v81 V71 control materialization path changed")

    result = {
        "schema": "qwen36-adapter-transport-precision-preregistration-v81",
        "status": "sealed_cpu_contract_cuda_integration_and_live_pairs_pending",
        "bead": "specialist-0j5.26",
        "purpose": (
            "Keep all canonical EGGROLL/LoRA arithmetic and persistence FP32, "
            "attest the lowest supported vLLM execution dtype, and test one "
            "reused pinned BF16 bank that copies directly into the resident slot."
        ),
        "authority": {
            "cpu_source_audit_and_state_machine_tests": True,
            "gpu_launch": False,
            "dataset_or_protected_content_access": False,
            "training_or_scored_evaluation": False,
            "runtime_worker_or_site_package_mutation": False,
            "candidate_update_or_layout_promotion": False,
        },
        "dependencies": {
            "must_preserve": [
                "specialist-0j5.14",
                "specialist-0j5.16",
                "specialist-0j5.18",
                "specialist-0j5.19",
                "specialist-0j5.21",
            ],
            "production_layout_blocker": False,
            "reason_not_blocking_specialist_0j5_20": (
                "The accepted synchronous BF16 single-slot layout remains a safe "
                "default. This challenger changes optional transfer latency/HBM "
                "traffic but not persistent layout or H2D bytes."
            ),
        },
        "bindings": {
            "implementation": _bound_file_v81(IMPLEMENTATION),
            "test": _bound_file_v81(TEST),
            "builder": _bound_file_v81(BUILDER),
            "v71_control_worker": _bound_file_v81(V71_WORKER),
            "v72_host_worker": _bound_file_v81(V72_WORKER),
            "v72_fused_runtime": _bound_file_v81(FUSED_V72),
            "v71_audit_preregistration": {
                **_bound_file_v81(V71_PREREG),
                "content_sha256": EXPECTED_CONTENT[V71_PREREG],
            },
            "v72_fused_preregistration": {
                **_bound_file_v81(FUSED_V72_PREREG),
                "content_sha256": EXPECTED_CONTENT[FUSED_V72_PREREG],
            },
            "v72_host_preregistration": {
                **_bound_file_v81(HOST_V72_PREREG),
                "content_sha256": EXPECTED_CONTENT[HOST_V72_PREREG],
            },
            "installed_vllm": {
                "version": "0.25.0",
                "version_source": _bound_file_v81(VLLM_VERSION),
                "lora_config": _bound_file_v81(VLLM_CONFIG),
                "base_linear": _bound_file_v81(VLLM_BASE),
                "fused_moe": _bound_file_v81(VLLM_FUSED),
                "model_manager": _bound_file_v81(VLLM_MANAGER),
                "source_modified": False,
            },
        },
        "installed_capability": capability,
        "decision": {
            "canonical_noise_update_optimizer_checkpoint_dtype": "float32",
            "canonical_location": "cpu",
            "selected_execution_dtype": "bfloat16",
            "fp8_execution_view": "reject_unsupported_before_gpu_launch",
            "float16_execution_view": (
                "reject_as_memory_ablation_byte_neutral; separate quality/performance "
                "preregistration required"
            ),
            "pinned_bank_count": 1,
            "double_buffering": "rejected_for_serial_single_slot_protocol",
            "device_staging": "rejected",
            "async_copy_destination": "existing_82_runtime_slot_views",
            "event_boundary": "before_exact_v71_readback_and_before_generation",
        },
        "transport_plan": plan,
        "failure_contract": {
            "pageable_nonblocking_source": "reject",
            "missing_or_stale_event": "reject_before_exact_audit",
            "partial_copy_count_or_bytes": "reject",
            "bank_reuse_before_retirement": "reject",
            "runtime_identity_mismatch": "reject_reward_and_update",
            "unknown_or_partial_copy": "exact_master_restore_or_terminal_poison",
            "failed_or_wrong_master_restore": "terminal_poison",
            "final_receipt": "requires_exact_idle_and_unpoisoned",
        },
        "future_live_ablation": {
            "authorization_in_this_artifact": False,
            "minimum_counterbalanced_pairs": 3,
            "physical_gpus_per_arm": 4,
            "same_long_lived_actor_pair_order": ["AB", "BA", "AB"],
            "control": (
                "unchanged V71/V72 BF16 per-view host conversion, temporary device "
                "tensor, synchronous runtime copy"
            ),
            "challenger_only_change": (
                "one stable pinned BF16 host storage, direct nonblocking copies to "
                "existing runtime views, one completion event"
            ),
            "required_exact_equalities": [
                "candidate projected runtime identity",
                "post-generation exact V71 audit",
                "reward inputs and outputs",
                "update acceptance and FP32 master identity",
                "exact abort/final master and runtime identity",
            ],
            "required_measurements": [
                "materialize_restore_update_abort wall time",
                "H2D bytes and copy count",
                "device-to-device bytes or kernel/memcpy trace",
                "peak allocated and reserved VRAM",
                "PCIe RX/TX and HBM activity",
                "all-four-GPU PID/useful-work attribution",
                "host pinned bytes VmLck/VmPin and NUMA placement",
                "cleanup idle and no foreign process",
            ],
            "promotion_gate": {
                "h2d_bytes_must_not_increase": True,
                "device_staging_bytes": 0,
                "exact_receipt_drift_allowed": False,
                "material_transition_time_improvement_required": True,
                "semantic_and_source_disjoint_ood_noninferiority_required": True,
                "all_three_pairs_and_all_four_gpus_required": True,
            },
        },
    }
    result["content_sha256_before_self_field"] = transport.canonical_sha256_v81(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true")
    group.add_argument("--print", dest="print_output", action="store_true")
    args = parser.parse_args()
    value = build_preregistration_v81()
    rendered = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if args.print_output:
        print(rendered, end="")
        return 0
    if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
        raise SystemExit("v81 preregistration is missing or stale")
    print(
        "v81 preregistration check passed: "
        f"{value['content_sha256_before_self_field']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
