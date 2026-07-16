#!/usr/bin/env python3
"""Audit and CPU-stage the fixed V55B LoRA-ES candidate for V56.

The operation is deliberately limited to the established V44A PEFT-key
prefix transform.  It opens no dataset or evaluation input and uses no GPU.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

from safetensors import safe_open

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "v55b_maximin_ratio0p25"
RUN_ROOT = (
    ROOT
    / "experiments/eggroll_es_hpo/runs/"
    "v55b_lora_es_fragile_maximin_backtracking"
).resolve()
SOURCE = (RUN_ROOT / "selected_candidate_v55b").resolve()
EVIDENCE = (RUN_ROOT / "v55b_evidence_manifest.json").resolve()
P16_GATE = (RUN_ROOT / "p16_train_gate_v52.json").resolve()
INTERNAL_REPORT = (RUN_ROOT / "nested_population_report_v52.json").resolve()
WRAPPER_REPORT = (
    RUN_ROOT / "fragile_maximin_backtracking_report_v55b.json"
).resolve()
OUTPUT = (
    ROOT
    / "experiments/eggroll_es_hpo/staged_adapters/"
    "v55b_maximin_ratio0p25_lora_es_qwen35_vllm_namespace_v56"
).resolve()

EXPECTED = {
    "evidence_commit": "570a2cbf4fc9e86951b59b5e9d72764fb79b9bb8",
    "evidence": "3db9ad07e7d028a347dfc9e6a90a5ddcc3c87763439d13b80a1b33d86a0af96f",
    "weights": "d13a62107b29ca2a17682d2fa0d2eb424ef3eb90ad8aafc0bc0f5c5786b7bf9c",
    "config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
    "candidate": "78fa46a9f77387f7872a09202658e471b2c03969687abcbccc253f5f194980fc",
    "runtime": "3745c5578db05102f1784f2b5489f39e9910eae4bf55f5c9edaa5485c48fda4c",
    "ordered_keys": "ddee26a3a4a10683a51f089e8b7028e4a8d9607e0827dab7a314e04e3ece2280",
    "snapshot_manifest": "b8985cae3f94c8e427b1e16a33033b0f8751d06a4a719e368b324627d18b4f96",
    "p16_gate": "17bbc43d7821b6e01d02e131a88d75519720e4a22a2d036f1e551909a0f2aecc",
    "p16_gate_content": "c5b54a709afb37f45ed3bfbfe6c3109875775b437952a0f0e70c5d1a5b41e499",
    "internal_report": "955218f6f6defbcafdf1734e3fad37e3ff2cf37ee4aa5f214c15f721a1ba5629",
    "internal_report_content": "3ef62178ef3b8179758944e8da98abe47297a43b8558befdde83d1879a095297",
    "wrapper_report": "c7e68279f22d3b3b90802e90a9b18c3ccc096860a2d2008777b85d36a54bd9ac",
    "wrapper_report_content": "bf6738e97034092570c98d6d309fbaed7ebbe147a9ef7f82a0e4eea4283f4849",
}


def source_seal_v56(arm: str = ARM) -> dict:
    """Fail closed unless the completed train-only V55B evidence is exact."""
    if arm != ARM:
        raise ValueError(arm)
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
    expected_files = {
        name: EXPECTED[name]
        for name in (
            "evidence",
            "weights",
            "config",
            "p16_gate",
            "internal_report",
            "wrapper_report",
        )
    }
    if observed != expected_files:
        raise RuntimeError("V56 V55B source artifact changed")

    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    snapshot = evidence.get("candidate_snapshot", {})
    scientific = evidence.get("scientific_result", {})
    state = evidence.get("state_and_cleanup", {})
    artifacts = evidence.get("artifacts", {})
    gate_checks = scientific.get("ratio_0p25_gate_checks", {})
    with safe_open(paths["weights"], framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}

    if (
        evidence.get("schema")
        != "lora-es-fragile-maximin-backtracking-evidence-v55b"
        or evidence.get("status")
        != "complete_passing_train_candidate_saved_no_master_commit"
        or evidence.get(
            "ood_shadow_benchmark_holdout_or_protected_semantics_opened"
        )
        is not False
        or evidence.get("sealed_holdout_opened") is not False
        or scientific.get("selected_target_norm_ratio") != 0.25
        or set(gate_checks.values()) != {True}
        or len(gate_checks) != 9
        or snapshot.get("four_actor_consensus_passed") is not True
        or snapshot.get("readback_verified") is not True
        or snapshot.get("worker_consensus_count") != 4
        or snapshot.get("canonical_fp32_candidate_sha256")
        != EXPECTED["candidate"]
        or snapshot.get("runtime_bf16_values_sha256") != EXPECTED["runtime"]
        or snapshot.get("canonical_ordered_key_sha256")
        != EXPECTED["ordered_keys"]
        or snapshot.get("tensor_count") != 70
        or snapshot.get("elements") != 4_528_128
        or snapshot.get("adapter_model_file_sha256") != EXPECTED["weights"]
        or snapshot.get("adapter_config_file_sha256") != EXPECTED["config"]
        or snapshot.get("optimizer_master_committed") is not False
        or snapshot.get("candidate_exactly_aborted_after_snapshot") is not True
        or state.get("update_sequence") != 0
        or state.get("both_candidate_transactions_exactly_aborted") is not True
        or state.get("four_actor_final_state_and_quiescence_exact") is not True
        or state.get("optimizer_master_committed") is not False
        or state.get("all_gpu_compute_process_lists_empty") is not True
        or artifacts.get("p16_train_gate", {}).get("file_sha256")
        != EXPECTED["p16_gate"]
        or artifacts.get("p16_train_gate", {}).get("content_sha256")
        != EXPECTED["p16_gate_content"]
        or artifacts.get("internal_report", {}).get("file_sha256")
        != EXPECTED["internal_report"]
        or artifacts.get("internal_report", {}).get("content_sha256")
        != EXPECTED["internal_report_content"]
        or artifacts.get("dedicated_wrapper_report", {}).get("file_sha256")
        != EXPECTED["wrapper_report"]
        or artifacts.get("dedicated_wrapper_report", {}).get("content_sha256")
        != EXPECTED["wrapper_report_content"]
        or metadata.get("schema") != "uncommitted-canonical-peft-fp32-v52"
        or metadata.get("candidate_sha256") != EXPECTED["candidate"]
        or metadata.get("manifest_sha256") != EXPECTED["snapshot_manifest"]
    ):
        raise RuntimeError("V56 V55B passing-candidate provenance changed")

    return {
        "schema": "v55b-maximin-train-only-success-provenance-v56",
        "arm": ARM,
        "evidence_commit": EXPECTED["evidence_commit"],
        "evidence_file_sha256": EXPECTED["evidence"],
        "source_weights_sha256": EXPECTED["weights"],
        "source_config_sha256": EXPECTED["config"],
        "canonical_candidate_sha256": EXPECTED["candidate"],
        "runtime_bf16_values_sha256": EXPECTED["runtime"],
        "p16_gate_file_sha256": EXPECTED["p16_gate"],
        "p16_gate_content_sha256": EXPECTED["p16_gate_content"],
        "internal_report_file_sha256": EXPECTED["internal_report"],
        "internal_report_content_sha256": EXPECTED["internal_report_content"],
        "wrapper_report_file_sha256": EXPECTED["wrapper_report"],
        "wrapper_report_content_sha256": EXPECTED["wrapper_report_content"],
        "selected_target_norm_ratio": 0.25,
        "all_nine_train_endpoint_gates_passed": True,
        "four_actor_consensus_passed": True,
        "selection_data_opened": False,
        "shadow_ood_holdout_or_heldout_opened": False,
    }


@contextmanager
def _injected():
    previous_candidate = prior.CANDIDATE_SPECS_V44A.get(ARM)
    previous_expected = prior.EXPECTED_V44A.get(ARM)
    previous_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[ARM] = (SOURCE, EVIDENCE, OUTPUT)
    prior.EXPECTED_V44A[ARM] = {
        "weights": EXPECTED["weights"],
        "config": EXPECTED["config"],
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v56(requested)
        if requested == ARM
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


def audit_source_v56() -> dict:
    source_seal_v56()
    with _injected():
        return prior.audit_source_v44a(ARM)


def stage_v56() -> dict:
    audit_source_v56()
    with _injected():
        return prior.stage_one_v44a(ARM, OUTPUT)


def main() -> int:
    manifest = stage_v56()
    print(json.dumps({
        "directory": manifest["artifact"]["directory"],
        "weights_sha256": manifest["artifact"]["weights_file_sha256"],
        "config_sha256": manifest["artifact"]["adapter_config_file_sha256"],
        "manifest_file_sha256": prior.file_sha256_v44a(
            OUTPUT / "stage_manifest_v44a.json"
        ),
        "manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "transformed_identity_sha256": manifest[
            "transformed_identity"
        ]["sha256"],
        "protected_semantic_access_count": 0,
        "heldout_or_holdout_opened": False,
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
