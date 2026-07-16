#!/usr/bin/env python3
"""Audit and byte-exactly stage the accepted V43L LoRA-ES adapter."""

from __future__ import annotations

import json
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "lora_es_v43l"
RUN_ROOT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43l_lora_es_v43i_projection_untried_backtracking"
).resolve()
SOURCE = (RUN_ROOT / "adapter_backtracked_v43l").resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v43l_lora_es_qwen35_vllm_namespace_v46e"
).resolve()
REPORT = RUN_ROOT / "matched_lora_es_backtracking_report_v43l.json"
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_projection_backtracking_v43l.json"
).resolve()
ATTEMPT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".v43l_lora_es_v43i_projection_untried_backtracking.attempt.json"
).resolve()
CANDIDATE_GATE = RUN_ROOT / "candidate_gate_ratio_0p03125_v43l.json"
CANDIDATE_CONSENSUS = RUN_ROOT / "candidate_consensus_ratio_0p03125_v43l.json"
GPU_LOG = RUN_ROOT / "gpu_activity_v43l.jsonl"

EXPECTED = {
    "weights": "ea4aa01483b8946714ef3d3e4eedb45734e4606044e81c5e776b345fff93744f",
    "config": "ede582c12e82fb50eb97ac934ff08eb553a79d2c2d999235abcd8b29795b1d52",
    "master_sha256": "af7316cdebd319dfac832ae6d93c315c4fa42d463e353980a7b742079c180fa2",
    "report": "2422ca3e009ddac81dfd0c1ebf6b6be4394858451d45131309c94fb49589f6e5",
    "report_content": "0c59de5b3404d3567cd3c35bb5cb6cd6c990d48e91ee23103bbba542adbf1f65",
    "preregistration": "5279d9bb3b88cbd894713a2e4e32de16629e2ca6ffe6716c4edf34ac00cfa91f",
    "preregistration_content": "7962e0bf0070e1f73ae7d858ee057f9007c1d28e8aaf9a223da5afa9d5dd5684",
    "attempt": "8eae6462983d537786bb4b556340c81032d12706c7bda26f567be413df20efbd",
    "attempt_content": "8739771aa943d0b1db1bb0f3ec7a0dcfd1e5136e7abeee86f5422a941b488146",
    "candidate_gate": "d0253013a926ab153e1ba4192d9fc01dad5ba5c9fced148627d2d5980fe83354",
    "candidate_gate_content": "c4a0f57878293c0ecf1bdbe068e17c6f14232009d36d2840b3e58e063184c552",
    "candidate_consensus": "f65aa853539450e3e2d49b5d699e28ec8be92c17c10674407d5c399b1ed0b59e",
    "candidate_consensus_content": "645541ad9950f18892cd51d2a43a5c14700a10f1d4e604dd69fc1458aa26a6de",
    "gpu_log": "04af71fd6fe4320a993ab02c63a0b5617df236431caf895393465b6197c0a649",
}
EXPECTED_GATE_CHECKS = {
    "domain_point_improvement",
    "prose_lm_noninferiority",
    "qa_generation_exact_noninferiority",
    "qa_generation_f1_noninferiority",
    "qa_generation_nonzero_noninferiority",
    "qa_logprob_noninferiority",
}


def read_seal_v46e(path: Path, file_sha: str, content_sha: str) -> dict:
    if prior.file_sha256_v44a(path) != file_sha:
        raise RuntimeError(f"V46E V43L seal changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or prior.canonical_sha256_v44a(compact) != content_sha
    ):
        raise RuntimeError(f"V46E V43L self-hash changed: {path}")
    return value


def source_seal_v46e() -> dict:
    """Fail closed unless the successful V43L result is still byte-identical."""
    report = read_seal_v46e(
        REPORT, EXPECTED["report"], EXPECTED["report_content"]
    )
    preregistration = read_seal_v46e(
        PREREGISTRATION,
        EXPECTED["preregistration"],
        EXPECTED["preregistration_content"],
    )
    attempt = read_seal_v46e(
        ATTEMPT, EXPECTED["attempt"], EXPECTED["attempt_content"]
    )
    gate = read_seal_v46e(
        CANDIDATE_GATE,
        EXPECTED["candidate_gate"],
        EXPECTED["candidate_gate_content"],
    )
    consensus = read_seal_v46e(
        CANDIDATE_CONSENSUS,
        EXPECTED["candidate_consensus"],
        EXPECTED["candidate_consensus_content"],
    )
    snapshots = report.get("update", {}).get("snapshots", [])
    rank_zero = next((item for item in snapshots if item.get("rank") == 0), {})
    other_ranks = [item for item in snapshots if item.get("rank") != 0]
    final_identity = report.get("update", {}).get("final_identity", {})
    scale_results = report.get("scale_results", [])
    gate_checks = gate.get("gate", {}).get("checks", {})
    if (
        report.get("schema")
        != "matched-lora-es-backtracking-train-only-report-v43l"
        or report.get("status")
        != "complete_passing_diagnostic_scale_committed_and_sealed"
        or report.get("accepted_target_norm_ratio") != 0.03125
        or report.get("protected_semantics_opened") is not False
        or report.get("shadow_dev_eval_ood_or_holdout_opened") is not False
        or report.get("quality_or_promotion_conclusion_authorized") is not False
        or report.get("gpu_activity", {}).get(
            "all_four_attributed_positive"
        ) is not True
        or len(report.get("gpu_activity", {}).get("by_gpu", {})) != 4
        or len(scale_results) != 1
        or scale_results[0].get("target_norm_ratio") != 0.03125
        or scale_results[0].get("gate_passed") is not True
        or scale_results[0].get("candidate_gate_content_sha256")
        != EXPECTED["candidate_gate_content"]
        or scale_results[0].get("candidate_consensus_content_sha256")
        != EXPECTED["candidate_consensus_content"]
        or preregistration.get("schema")
        != "matched-lora-es-backtracking-preregistration-v43l"
        or preregistration.get("status")
        != "preregistered_before_train_only_launch"
        or preregistration.get("sealed_holdout_opened") is not False
        or attempt.get("schema") != "matched-lora-es-backtracking-attempt-v43l"
        or attempt.get("protected_semantics_opened") is not False
        or gate.get("schema")
        != "matched-lora-es-backtracked-candidate-gate-v43l"
        or gate.get("status") != "complete_before_commit_or_abort"
        or gate.get("target_norm_ratio") != 0.03125
        or gate.get("protected_semantics_opened") is not False
        or gate.get("gate", {}).get("passed") is not True
        or set(gate_checks) != EXPECTED_GATE_CHECKS
        or any(value is not True for value in gate_checks.values())
        or consensus.get("schema")
        != "matched-lora-es-backtracked-candidate-consensus-v43l"
        or consensus.get("status") != "complete_while_candidate_uncommitted"
        or consensus.get("target_norm_ratio") != 0.03125
        or consensus.get("protected_semantics_opened") is not False
        or consensus.get("equivalence", {}).get("actors") != 4
        or consensus.get("equivalence", {}).get("passed") is not True
        or len(snapshots) != 4
        or {item.get("rank") for item in snapshots} != {0, 1, 2, 3}
        or any(
            item.get("master_identity", {}).get("sha256")
            != EXPECTED["master_sha256"] for item in snapshots
        )
        or rank_zero.get("written") is not True
        or rank_zero.get("weights_sha256") != EXPECTED["weights"]
        or rank_zero.get("config_sha256") != EXPECTED["config"]
        or rank_zero.get("readback_verified") is not True
        or rank_zero.get("readback_identity", {}).get("sha256")
        != EXPECTED["master_sha256"]
        or any(item.get("written") is not False for item in other_ranks)
        or final_identity.get("sha256") != EXPECTED["master_sha256"]
        or final_identity.get("tensor_count") != 70
        or final_identity.get("elements") != 4_528_128
        or prior.file_sha256_v44a(GPU_LOG) != EXPECTED["gpu_log"]
        or report.get("gpu_log", {}).get("file_sha256") != EXPECTED["gpu_log"]
    ):
        raise RuntimeError("V46E V43L successful snapshot provenance changed")
    return {
        "schema": "v43l-successful-backtracked-snapshot-provenance-v46e",
        "report_file_sha256": EXPECTED["report"],
        "report_content_sha256": EXPECTED["report_content"],
        "preregistration_file_sha256": EXPECTED["preregistration"],
        "attempt_file_sha256": EXPECTED["attempt"],
        "candidate_gate_file_sha256": EXPECTED["candidate_gate"],
        "candidate_gate_content_sha256": EXPECTED["candidate_gate_content"],
        "candidate_consensus_file_sha256": EXPECTED["candidate_consensus"],
        "candidate_consensus_content_sha256": EXPECTED[
            "candidate_consensus_content"
        ],
        "gpu_log_file_sha256": EXPECTED["gpu_log"],
        "accepted_target_norm_ratio": 0.03125,
        "gate_check_count": len(EXPECTED_GATE_CHECKS),
        "snapshot_actor_count": 4,
        "snapshot_master_sha256": EXPECTED["master_sha256"],
        "state_complete": True,
        "all_four_gpus_attributed_positive": True,
        "selection_data_opened": False,
        "heldout_or_holdout_opened": False,
    }


def _inject_v46e():
    old_candidate = prior.CANDIDATE_SPECS_V44A.get(ARM)
    old_expected = prior.EXPECTED_V44A.get(ARM)
    old_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[ARM] = (SOURCE, None, OUTPUT)
    prior.EXPECTED_V44A[ARM] = {
        "weights": EXPECTED["weights"],
        "config": EXPECTED["config"],
        "master_sha256": EXPECTED["master_sha256"],
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v46e() if requested == ARM else old_seal(requested)
    )
    return old_candidate, old_expected, old_seal


def _restore_v46e(saved) -> None:
    old_candidate, old_expected, old_seal = saved
    if old_candidate is None:
        prior.CANDIDATE_SPECS_V44A.pop(ARM, None)
    else:
        prior.CANDIDATE_SPECS_V44A[ARM] = old_candidate
    if old_expected is None:
        prior.EXPECTED_V44A.pop(ARM, None)
    else:
        prior.EXPECTED_V44A[ARM] = old_expected
    prior._source_seal_v44a = old_seal


def audit_source_v46e() -> dict:
    saved = _inject_v46e()
    try:
        return prior.audit_source_v44a(ARM)
    finally:
        _restore_v46e(saved)


def stage_v46e(output: Path | None = None) -> dict:
    audit_source_v46e()
    saved = _inject_v46e()
    try:
        return prior.stage_one_v44a(
            ARM, Path(output).resolve() if output is not None else None
        )
    finally:
        _restore_v46e(saved)


def main() -> int:
    manifest = stage_v46e()
    print(json.dumps({
        "directory": manifest["artifact"]["directory"],
        "weights_sha256": manifest["artifact"]["weights_file_sha256"],
        "manifest_content_sha256": manifest[
            "content_sha256_before_self_field"
        ],
        "transformed_identity_sha256": manifest[
            "transformed_identity"
        ]["sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
