#!/usr/bin/env python3
"""Audit and byte-exactly stage the accepted V43J LoRA-ES adapter."""

from __future__ import annotations

import json
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "lora_es_v43j"
RUN_ROOT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43j_lora_es_v43i_projection_backtracking"
).resolve()
SOURCE = (RUN_ROOT / "adapter_backtracked_v43j").resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v43j_lora_es_qwen35_vllm_namespace_v46c"
).resolve()
REPORT = RUN_ROOT / "matched_lora_es_backtracking_report_v43j.json"
PREREGISTRATION = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_projection_backtracking_v43j.json"
).resolve()
ATTEMPT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".v43j_lora_es_v43i_projection_backtracking.attempt.json"
).resolve()
CANDIDATE_GATE = RUN_ROOT / "candidate_gate_ratio_0p25_v43j.json"
CANDIDATE_CONSENSUS = RUN_ROOT / "candidate_consensus_ratio_0p25_v43j.json"
GPU_LOG = RUN_ROOT / "gpu_activity_v43j.jsonl"

EXPECTED = {
    "weights": "9b8d839f46f756d57472ed7586eebbfd97b7e095cba44f9f4131099f28b1b77b",
    "config": "ede582c12e82fb50eb97ac934ff08eb553a79d2c2d999235abcd8b29795b1d52",
    "master_sha256": "2e764f0ec9804ccc9409d124ff5822521d70301eba522114b019b33c87943d2b",
    "report": "70084cdb1752b097e737572515c5c0c2f2a9b13f58ad0cb087f157c61ed7387a",
    "report_content": "61820f950f6c85ccb69f0e798f3526e4fae658aa40e7801b7c13bd5367056c60",
    "preregistration": "8997bed0d2ed054d3a97df106c039243e82426de3021ba291302e4ae59399626",
    "preregistration_content": "1046bdd4c328b51d4da6bfb50723df8ac2efccf25fc0cd6f4a58d1520a872b8d",
    "attempt": "8a51905aaa67bc9323518c2d3420ec338a20ca3306a8ece0a61414c3195a2885",
    "attempt_content": "2dd1470443c2705e394d2120be42d5c2d4e0206c162407fc197c54415d6b15b8",
    "candidate_gate": "ea631461bf78299596b68d40fe2baa6fa95aced58674adf4d9d6b44e3ea2ac92",
    "candidate_gate_content": "4792016a320ff279c75034349dda1779ee88e744c05dda39c30c38a23133f320",
    "candidate_consensus": "d4db53079fee24216294a5efec6bdacd10b81c1ccdf1e2359f211dd33e447e90",
    "candidate_consensus_content": "9bb9273ed5cf3d0ef9e293a94aabd769701f8f2edd311e1952e3d54d79c596c2",
    "gpu_log": "e90e60dc646872b42c18cc48d7238d0e4f6fe2c05ddf09db7a9e212f6fb6e742",
}
EXPECTED_GATE_CHECKS = {
    "domain_point_improvement",
    "prose_lm_noninferiority",
    "qa_generation_exact_noninferiority",
    "qa_generation_f1_noninferiority",
    "qa_generation_nonzero_noninferiority",
    "qa_logprob_noninferiority",
}


def read_seal_v46c(path: Path, file_sha: str, content_sha: str) -> dict:
    if prior.file_sha256_v44a(path) != file_sha:
        raise RuntimeError(f"V46C V43J seal changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or prior.canonical_sha256_v44a(compact) != content_sha
    ):
        raise RuntimeError(f"V46C V43J self-hash changed: {path}")
    return value


def source_seal_v46c() -> dict:
    """Fail closed unless the successful V43J result is still byte-identical."""
    report = read_seal_v46c(
        REPORT, EXPECTED["report"], EXPECTED["report_content"]
    )
    preregistration = read_seal_v46c(
        PREREGISTRATION,
        EXPECTED["preregistration"],
        EXPECTED["preregistration_content"],
    )
    attempt = read_seal_v46c(
        ATTEMPT, EXPECTED["attempt"], EXPECTED["attempt_content"]
    )
    gate = read_seal_v46c(
        CANDIDATE_GATE,
        EXPECTED["candidate_gate"],
        EXPECTED["candidate_gate_content"],
    )
    consensus = read_seal_v46c(
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
        != "matched-lora-es-backtracking-train-only-report-v43j"
        or report.get("status")
        != "complete_largest_passing_scale_committed_and_sealed"
        or report.get("accepted_target_norm_ratio") != 0.25
        or report.get("largest_passing_preference_enforced") is not True
        or report.get("protected_semantics_opened") is not False
        or report.get("shadow_dev_eval_ood_or_holdout_opened") is not False
        or report.get("quality_or_promotion_conclusion_authorized") is not False
        or report.get("gpu_activity", {}).get(
            "all_four_attributed_positive"
        ) is not True
        or len(report.get("gpu_activity", {}).get("by_gpu", {})) != 4
        or len(scale_results) != 1
        or scale_results[0].get("target_norm_ratio") != 0.25
        or scale_results[0].get("gate_passed") is not True
        or scale_results[0].get("candidate_gate_content_sha256")
        != EXPECTED["candidate_gate_content"]
        or scale_results[0].get("candidate_consensus_content_sha256")
        != EXPECTED["candidate_consensus_content"]
        or preregistration.get("schema")
        != "matched-lora-es-backtracking-preregistration-v43j"
        or preregistration.get("status")
        != "preregistered_before_train_only_launch"
        or preregistration.get("sealed_holdout_opened") is not False
        or attempt.get("schema") != "matched-lora-es-backtracking-attempt-v43j"
        or attempt.get("protected_semantics_opened") is not False
        or gate.get("schema")
        != "matched-lora-es-backtracked-candidate-gate-v43j"
        or gate.get("status") != "complete_before_commit_or_abort"
        or gate.get("target_norm_ratio") != 0.25
        or gate.get("protected_semantics_opened") is not False
        or gate.get("gate", {}).get("passed") is not True
        or set(gate_checks) != EXPECTED_GATE_CHECKS
        or any(value is not True for value in gate_checks.values())
        or consensus.get("schema")
        != "matched-lora-es-backtracked-candidate-consensus-v43j"
        or consensus.get("status") != "complete_while_candidate_uncommitted"
        or consensus.get("target_norm_ratio") != 0.25
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
        raise RuntimeError("V46C V43J successful snapshot provenance changed")
    return {
        "schema": "v43j-successful-backtracked-snapshot-provenance-v46c",
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
        "accepted_target_norm_ratio": 0.25,
        "gate_check_count": len(EXPECTED_GATE_CHECKS),
        "snapshot_actor_count": 4,
        "snapshot_master_sha256": EXPECTED["master_sha256"],
        "state_complete": True,
        "all_four_gpus_attributed_positive": True,
        "selection_data_opened": False,
        "heldout_or_holdout_opened": False,
    }


def _inject_v46c():
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
        source_seal_v46c() if requested == ARM else old_seal(requested)
    )
    return old_candidate, old_expected, old_seal


def _restore_v46c(saved) -> None:
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


def audit_source_v46c() -> dict:
    saved = _inject_v46c()
    try:
        return prior.audit_source_v44a(ARM)
    finally:
        _restore_v46c(saved)


def stage_v46c(output: Path | None = None) -> dict:
    audit_source_v46c()
    saved = _inject_v46c()
    try:
        return prior.stage_one_v44a(
            ARM, Path(output).resolve() if output is not None else None
        )
    finally:
        _restore_v46c(saved)


def main() -> int:
    manifest = stage_v46c()
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
