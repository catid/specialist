#!/usr/bin/env python3
"""Build/check the prospective source-only LoRA ES V83 preregistration."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

from eggroll_es_autograd_free_contract_v83 import canonical_sha256_v83


ROOT = Path(__file__).resolve().parent
PREREG_PATH = ROOT / "qwen36_lora_es_autograd_free_preregistration_v83.json"
EVIDENCE_PATH = ROOT / "qwen36_lora_es_autograd_free_source_evidence_v83.json"
PINNED_TRAINER_FILES = (
    "eggroll_es_worker_lora_v71.py",
    "eggroll_es_worker_lora_v72.py",
    "eggroll_es_worker_lora_pinned_transport_v81a.py",
)
ADDITIVE_SOURCE_FILES = (
    "eggroll_es_autograd_free_contract_v83.py",
    "eggroll_es_worker_lora_autograd_free_v83.py",
    "build_qwen36_lora_es_autograd_free_preregistration_v83.py",
    "test_eggroll_es_autograd_free_v83.py",
)
REQUIRED_V83_TRANSITIONS = (
    "install_adapter_state_v41a",
    "capture_adapter_reference_v41a",
    "_materialize_v41a",
    "materialize_mirrored_adapter_v71",
    "restore_mirrored_adapter_v71",
    "_restore_exact_master_v66",
    "_restore_update_master_v71",
    "accept_population_rewards_v71",
    "prepare_sharded_adapter_update_v72",
    "execute_sharded_adapter_update_v41a",
    "commit_sharded_adapter_update_v71",
    "abort_sharded_adapter_update_v71",
    "finalize_sharded_adapter_update_v71",
    "abort_mirrored_update_if_present_v66",
    "begin_actor_gpu_work_v66d",
    "end_actor_gpu_work_v66d",
    "save_adapter_snapshot_v41a",
    "final_transport_receipt_v81a",
)


def file_sha256_v83(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call_name_v83(node: ast.Call) -> str:
    value = node.func
    parts = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _source_ast_evidence_v83(filename: str):
    path = ROOT / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=filename)
    calls = [_call_name_v83(node) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return {
        "sha256": file_sha256_v83(path),
        "torch_no_grad_call_count": calls.count("torch.no_grad"),
        "torch_inference_mode_call_count": calls.count("torch.inference_mode"),
        "torch_enable_grad_call_count": calls.count("torch.enable_grad"),
        "tensor_clone_call_count": sum(
            name == "clone" or name.endswith(".clone") for name in calls
        ),
    }


def _v83_transition_coverage_v83():
    filename = "eggroll_es_worker_lora_autograd_free_v83.py"
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"), filename=filename)
    target = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "LoRAAdapterStateWorkerExtensionV83"
    )
    methods = {
        node.name: node
        for node in target.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = sorted(set(REQUIRED_V83_TRANSITIONS) - set(methods))
    unguarded = []
    for name in REQUIRED_V83_TRANSITIONS:
        method = methods.get(name)
        if method is None:
            continue
        calls = {
            _call_name_v83(node)
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
        }
        if not any(call.endswith("_run_transition_v83") for call in calls):
            unguarded.append(name)
    if missing or unguarded:
        raise RuntimeError(
            f"v83 transition coverage changed: missing={missing}, unguarded={unguarded}"
        )
    return {
        "required_transition_count": len(REQUIRED_V83_TRANSITIONS),
        "guarded_transition_count": len(REQUIRED_V83_TRANSITIONS),
        "required_transitions": list(REQUIRED_V83_TRANSITIONS),
        "missing_transitions": [],
        "unguarded_transitions": [],
    }


def build_source_evidence_v83():
    trainer = {
        filename: _source_ast_evidence_v83(filename)
        for filename in PINNED_TRAINER_FILES
    }
    additive = {
        filename: _source_ast_evidence_v83(filename)
        for filename in ADDITIVE_SOURCE_FILES
    }
    if any(item["torch_enable_grad_call_count"] for item in trainer.values()):
        raise RuntimeError("accepted trainer source contains an autograd enable call")
    if any(
        additive[name]["tensor_clone_call_count"]
        for name in (
            "eggroll_es_autograd_free_contract_v83.py",
            "eggroll_es_worker_lora_autograd_free_v83.py",
        )
    ):
        raise RuntimeError("v83 enforcement source gained a tensor clone call")
    coverage = _v83_transition_coverage_v83()
    report = {
        "schema": "qwen36-lora-es-autograd-free-source-evidence-v83",
        "status": "source_cpu_contract_sealed",
        "scope": "prospective_additive_worker_only",
        "accepted_trainers_immutable": True,
        "trainer_source": trainer,
        "additive_source": additive,
        "transition_coverage": coverage,
        "source_findings": {
            "v71_materialization_has_explicit_no_grad": (
                trainer["eggroll_es_worker_lora_v71.py"][
                    "torch_no_grad_call_count"
                ]
                >= 1
            ),
            "v81a_materialization_has_explicit_no_grad": (
                trainer["eggroll_es_worker_lora_pinned_transport_v81a.py"][
                    "torch_no_grad_call_count"
                ]
                >= 1
            ),
            "v72_update_relies_on_additive_outer_guard": True,
            "accepted_source_enables_autograd": False,
            "v83_guard_reads_tensor_metadata_only": True,
            "v83_guard_tensor_content_read_bytes": 0,
            "v83_guard_tensor_clone_bytes": 0,
            "v83_guard_full_model_clone_bytes": 0,
            "v83_guard_arbitrary_model_graph_traversal": False,
        },
        "synthetic_fault_matrix": [
            "requires_grad_true_tensor_rejected",
            "non_none_grad_tensor_rejected",
            "grad_enabled_tensor_operation_rejected",
            "hidden_torch_optimizer_registry_rejected",
            "optimizer_state_dict_registry_rejected",
        ],
        "execution_boundary": {
            "cpu_synthetic_only": True,
            "torch_cuda_launched": False,
            "vllm_or_ray_model_launched": False,
            "gpu_training_launched": False,
        },
        "authority": {
            "training_launch_authorized": False,
            "hpo_authorized": False,
            "quality_or_ood_claim_authorized": False,
            "promotion_authorized": False,
        },
    }
    report["report_sha256"] = canonical_sha256_v83(report)
    return report


def build_preregistration_v83():
    evidence = build_source_evidence_v83()
    prereg = {
        "schema": "qwen36-lora-es-autograd-free-preregistration-v83",
        "status": "prospective_source_cpu_only",
        "model_family": "Qwen3.6-35B-A3B",
        "worker_class": (
            "eggroll_es_worker_lora_autograd_free_v83."
            "LoRAAdapterStateWorkerExtensionV83"
        ),
        "source_evidence_sha256": evidence["report_sha256"],
        "hypothesis": (
            "Explicitly removing autograd authority from worker-owned LoRA ES "
            "surfaces avoids gradient buffers and graph retention without a "
            "full-model clone or content scan."
        ),
        "sealed_surface_roles": [
            "canonical_fp32_master",
            "pending_candidate_and_rollback",
            "committed_rollback",
            "runtime_lora_views_and_parents",
            "runtime_base_tensor_links",
            "pinned_bf16_publication_bank",
            "transition_inputs_and_registered_results",
        ],
        "required_invariants": {
            "requires_grad_false": True,
            "grad_none": True,
            "grad_fn_none": True,
            "autograd_leaf": True,
            "torch_optimizer_authority_absent": True,
            "optimizer_state_dict_authority_absent": True,
            "guard_mode": "torch.no_grad+reject_grad_enabled_tensor_ops",
            "inference_mode_required": False,
            "inference_mode_omission_reason": (
                "preserve accepted tensor version-counter invariants"
            ),
            "tensor_content_read_bytes": 0,
            "tensor_clone_bytes": 0,
            "full_model_clone_bytes": 0,
            "arbitrary_model_graph_traversal": False,
        },
        "fault_injections_required": evidence["synthetic_fault_matrix"],
        "live_acceptance_gate": {
            "dependency": "specialist-0j5.32",
            "dependency_satisfied_in_this_milestone": False,
            "exact_worker_count": 4,
            "all_worker_surface_receipts_required": True,
            "controller_aggregate_required": True,
            "live_receipt_present": False,
            "bead_must_remain_in_progress": True,
        },
        "claims_withheld": [
            "live_gpu_compatibility",
            "peak_vram_reduction",
            "memory_bandwidth_reduction",
            "wall_time_improvement",
            "training_or_hpo_promotion",
            "quality_or_ood_equivalence",
        ],
    }
    prereg["preregistration_sha256"] = canonical_sha256_v83(prereg)
    return prereg


def _load_json_v83(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_artifacts_v83():
    expected_evidence = build_source_evidence_v83()
    expected_prereg = build_preregistration_v83()
    if _load_json_v83(EVIDENCE_PATH) != expected_evidence:
        raise RuntimeError("v83 source evidence artifact is stale")
    if _load_json_v83(PREREG_PATH) != expected_prereg:
        raise RuntimeError("v83 preregistration artifact is stale")
    return expected_prereg, expected_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("preregistration", "evidence", "both"), default="both"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        prereg, evidence = check_artifacts_v83()
        print(json.dumps({
            "preregistration_sha256": prereg["preregistration_sha256"],
            "source_evidence_sha256": evidence["report_sha256"],
        }, sort_keys=True))
        return
    values = {
        "preregistration": build_preregistration_v83(),
        "evidence": build_source_evidence_v83(),
    }
    selected = values if args.kind == "both" else values[args.kind]
    print(json.dumps(selected, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
