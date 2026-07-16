#!/usr/bin/env python3
"""Audit and CPU-stage the fixed V59 LoRA-ES candidate for V60 OOD."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from safetensors import safe_open

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "v59_fragile_priority_ratio0p25"
RUN_ROOT = (
    ROOT / "experiments/eggroll_es_hpo/runs/v59_lora_es_fragile_priority"
).resolve()
SOURCE = (RUN_ROOT / "selected_candidate_v59").resolve()
EVIDENCE = (RUN_ROOT / "v59_evidence_manifest.json").resolve()
P16_GATE = (RUN_ROOT / "p16_train_gate_v52.json").resolve()
INTERNAL_REPORT = (RUN_ROOT / "nested_population_report_v52.json").resolve()
WRAPPER_REPORT = (RUN_ROOT / "compatibility_report_v55b.json").resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v59_fragile_priority_ratio0p25_lora_es_qwen35_vllm_namespace_v60"
).resolve()

# Evidence file/commit are pinned immediately after the aggregate-only
# finalizer commit.  Every other immutable train artifact is already fixed.
EXPECTED = {
    "evidence_commit": "c28ec65551af31623ee52d4042e5c8a4dcd7d9d3",
    "evidence": "136bb6d5e93a0e2a8187a670ae2e8760bed1dc08396dadeba76a6c053674d4ab",
    "weights": "c2665b60928b16120a2b98fdf137fafd250644852c86a02d797689f02105c6c8",
    "config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    "candidate": "1713987fcad93f3e6368a309415faf5de2f4230eaf3c44baf23b8e9a2edf2a3d",
    "runtime": "ad5dd995de7cad3c9d116d64deb3aa67b9db46fbdf4e3f8a6ab5ee37340b5923",
    "ordered_keys": "ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280",
    "snapshot_manifest": "e477cac0ed5fbed2cf106d0a5640648a9a533e683e94c06d4571e57cef5c85d7",
    "p16_gate": "1b894a23e0e98f3942d775d67ae87aaf952e042ef5c99501c031bec4a73c2487",
    "p16_gate_content": "d4a381a7c0713de7ab086e65d46ceb00d79798574e938c25d57d24f9b957e038",
    "internal_report": "08857732f3e3c11abe6fdff69871ca5101e4408d8f750cc7f7d5a78d6c349c62",
    "internal_report_content": "9f89ba1dfab7e4c09e77872f64cff5c709a21bb72be08794ab928e1abf0cb120",
    "wrapper_report": "dbd316f68d242b56542da8cf7d8943c9047d1aadfb1f6ae112f0d0c81a452c23",
    "wrapper_report_content": "c359859ba1e0fca05a36d0574537b61d86a75edd6527567a2c4b7ab2f5272589",
}


def source_seal_v60(arm: str = ARM) -> dict:
    if arm != ARM:
        raise ValueError(arm)
    if EXPECTED["evidence"].startswith("PENDING_"):
        raise RuntimeError("V60 evidence seal has not been pinned")
    paths = {
        "evidence": EVIDENCE,
        "weights": SOURCE / "adapter_model.safetensors",
        "config": SOURCE / "adapter_config.json",
        "p16_gate": P16_GATE,
        "internal_report": INTERNAL_REPORT,
        "wrapper_report": WRAPPER_REPORT,
    }
    observed = {
        name: prior.file_sha256_v44a(path) for name, path in paths.items()
    }
    if observed != {
        name: EXPECTED[name] for name in paths
    }:
        raise RuntimeError("V60 V59 source artifact changed")
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    scientific = evidence.get("scientific_result", {})
    snapshot = evidence.get("candidate_snapshot", {})
    instrumentation = evidence.get("instrumentation_status", {})
    artifacts = evidence.get("artifacts", {})
    with safe_open(paths["weights"], framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if (
        evidence.get("schema") != "lora-es-fragile-priority-evidence-v59"
        or evidence.get("status")
        != "complete_scientific_train_pass_candidate_saved_wrapper_telemetry_false_negative"
        or scientific.get("selected_target_norm_ratio") != 0.25
        or scientific.get("evaluated_ratio_prefix") != [0.5, 0.375, 0.25]
        or scientific.get("all_nine_calibrated_endpoint_gates_passed") is not True
        or scientific.get("four_actor_candidate_consensus_passed") is not True
        or set(scientific.get("ratio_0p25_gate_checks", {}).values()) != {True}
        or len(scientific.get("ratio_0p25_gate_checks", {})) != 9
        or snapshot.get("adapter_model_file_sha256") != EXPECTED["weights"]
        or snapshot.get("adapter_config_file_sha256") != EXPECTED["config"]
        or snapshot.get("canonical_fp32_candidate_sha256") != EXPECTED["candidate"]
        or snapshot.get("runtime_bf16_values_sha256") != EXPECTED["runtime"]
        or snapshot.get("snapshot_manifest_sha256") != EXPECTED["snapshot_manifest"]
        or snapshot.get("tensor_count") != 70
        or snapshot.get("elements") != 4_528_128
        or snapshot.get("worker_consensus_count") != 4
        or snapshot.get("readback_verified") is not True
        or snapshot.get("candidate_exactly_aborted_after_snapshot") is not True
        or snapshot.get("optimizer_master_committed") is not False
        or instrumentation.get(
            "scientific_executor_and_snapshot_completed_before_wrapper_error"
        ) is not True
        or instrumentation.get(
            "selection_gate_or_candidate_bytes_changed_by_finalizer"
        ) is not False
        or artifacts.get("p16_train_gate", {}).get("file_sha256")
        != EXPECTED["p16_gate"]
        or artifacts.get("p16_train_gate", {}).get("content_sha256")
        != EXPECTED["p16_gate_content"]
        or artifacts.get("internal_report", {}).get("file_sha256")
        != EXPECTED["internal_report"]
        or artifacts.get("internal_report", {}).get("content_sha256")
        != EXPECTED["internal_report_content"]
        or artifacts.get("compatibility_report", {}).get("file_sha256")
        != EXPECTED["wrapper_report"]
        or artifacts.get("compatibility_report", {}).get("content_sha256")
        != EXPECTED["wrapper_report_content"]
        or evidence.get("optimizer_master_committed") is not False
        or evidence.get("protected_semantics_opened") is not False
        or evidence.get("ood_shadow_opened") is not False
        or evidence.get("terminal_holdout_opened") is not False
        or metadata.get("schema") != "uncommitted-canonical-peft-fp32-v52"
        or metadata.get("candidate_sha256") != EXPECTED["candidate"]
        or metadata.get("manifest_sha256") != EXPECTED["snapshot_manifest"]
    ):
        raise RuntimeError("V60 V59 passing-candidate provenance changed")
    return {
        "schema": "v59-fragile-priority-train-only-success-provenance-v60",
        "arm": ARM,
        "evidence_commit": EXPECTED["evidence_commit"],
        "evidence_file_sha256": EXPECTED["evidence"],
        "source_weights_sha256": EXPECTED["weights"],
        "source_config_sha256": EXPECTED["config"],
        "canonical_candidate_sha256": EXPECTED["candidate"],
        "runtime_bf16_values_sha256": EXPECTED["runtime"],
        "selected_target_norm_ratio": 0.25,
        "all_nine_train_endpoint_gates_passed": True,
        "four_actor_consensus_passed": True,
        "wrapper_telemetry_false_negative_did_not_change_science": True,
        "selection_data_opened": False,
        "shadow_ood_holdout_or_heldout_opened": False,
    }


# Compatibility alias used by the inherited V56 evaluator binding.
source_seal_v56 = source_seal_v60


@contextmanager
def _injected():
    previous_candidate = prior.CANDIDATE_SPECS_V44A.get(ARM)
    previous_expected = prior.EXPECTED_V44A.get(ARM)
    previous_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[ARM] = (SOURCE, EVIDENCE, OUTPUT)
    prior.EXPECTED_V44A[ARM] = {
        "weights": EXPECTED["weights"], "config": EXPECTED["config"],
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v60(requested) if requested == ARM
        else previous_seal(requested)
    )
    try:
        yield
    finally:
        if previous_candidate is None:
            prior.CANDIDATE_SPECS_V44A.pop(ARM, None)
        else:
            prior.CANDIDATE_SPECS_V44A[ARM] = previous_candidate
        if previous_expected is None:
            prior.EXPECTED_V44A.pop(ARM, None)
        else:
            prior.EXPECTED_V44A[ARM] = previous_expected
        prior._source_seal_v44a = previous_seal


def audit_source_v60() -> dict:
    source_seal_v60()
    with _injected():
        return prior.audit_source_v44a(ARM)


def stage_v60() -> dict:
    audit_source_v60()
    with _injected():
        return prior.stage_one_v44a(ARM, OUTPUT)


def main() -> int:
    manifest = stage_v60()
    print(json.dumps({
        "directory": manifest["artifact"]["directory"],
        "weights_sha256": manifest["artifact"]["weights_file_sha256"],
        "config_sha256": manifest["artifact"]["adapter_config_file_sha256"],
        "manifest_file_sha256": prior.file_sha256_v44a(
            OUTPUT / "stage_manifest_v44a.json"
        ),
        "manifest_content_sha256": manifest["content_sha256_before_self_field"],
        "transformed_identity_sha256": manifest["transformed_identity"]["sha256"],
        "protected_semantic_access_count": 0,
        "heldout_or_holdout_opened": False,
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
