#!/usr/bin/env python3
"""Audit and stage the successful canonical V43G LoRA-ES snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
ARM = "lora_es_v43g"
SOURCE = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    "v43g_matched_lora_es_fold3_pop8_robust/adapter_step1_v43g"
).resolve()
OUTPUT = (
    ROOT / "experiments/eggroll_es_hpo/staged_adapters/"
    "v43g_lora_es_qwen35_vllm_namespace_v45b"
).resolve()
RUN_ROOT = SOURCE.parent
REPORT = RUN_ROOT / "matched_lora_es_report_v43g.json"
PREREG = (
    ROOT / "experiments/eggroll_es_hpo/preregistrations/"
    "matched_lora_es_fold3_pop8_robust_v43g.json"
).resolve()
ATTEMPT = (
    ROOT / "experiments/eggroll_es_hpo/runs/"
    ".v43g_matched_lora_es_fold3_pop8_robust.attempt.json"
).resolve()
CALIBRATION = RUN_ROOT / "numeric_calibration_v43g.json"
RELIABILITY = RUN_ROOT / "population_reliability_v43g.json"
CONSENSUS = RUN_ROOT / "post_update_consensus_v43g.json"

EXPECTED = {
    "weights": "ca54ee2167707ee95c078ad6aca7e01dc963fafeacd42f6f64f0001ea89c4cda",
    "config": "ede582c12e82fb50eb97ac934ff08eb553a79d2c2d999235abcd8b29795b1d52",
    "master_sha256": "a116edad9e8fd0ea7128982660e52b808e807e62a9d7e59cba7817a19e2e3363",
    "report": "4791bf245ff90e33de24a3645c18c52d25e34c5806a8cf9b2e791ec9e9e85fbb",
    "report_content": "a74f30d8743d640739333448a35e2f9f61a144a72b3359d0571747889a58c75f",
    "prereg": "b923bd72ed9d09936200b7aae0f6c25671f901a41d26e062e18371eab845e3a6",
    "prereg_content": "33cb3c2ed71d67f6e2eaa3540f78b32a1d90014ff45808d177e3312fc4e87299",
    "attempt": "2ea921c40ddeb4556d2f27814079ff1564c63a88732c6808a810ec37f6822760",
    "attempt_content": "371c230ca9388f1faa669b8ce81f056e733cc7a990b6fb728179cbf602c3b6e9",
    "calibration": "8328063ee36a59b52abe9ef786f084d4f1337d5c2f3d6f9c1b5b823f073ee848",
    "calibration_content": "2a0b99d25fadd143ef1bb45be46fb2cacc34bf705bcdffee18b7adf494457216",
    "reliability": "b09d65ec24901858638aba9d0b33a8cb101d3d1cf3a490fa393dd24685602fde",
    "reliability_content": "884a2c34416ada0d091f23ba483e5b22029e25970fdc42efd134700d8ea1cfb4",
    "consensus": "a04fb637c0c3649e0f04a1637922b2c7c5758fc74225ac1c38a19af9a52f8b23",
    "consensus_content": "3a6ea94145824128a65e1d452d9f40e94ccc9a8be3bcc2e75746635748c16249",
}


def read_seal_v45b(path: Path, file_sha: str, content_sha: str) -> dict:
    if prior.file_sha256_v44a(path) != file_sha:
        raise RuntimeError(f"V45B V43G seal changed: {path}")
    value = json.loads(path.read_text())
    compact = {
        key: item for key, item in value.items()
        if key != "content_sha256_before_self_field"
    }
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or prior.canonical_sha256_v44a(compact) != content_sha
    ):
        raise RuntimeError(f"V45B V43G self-hash changed: {path}")
    return value


def source_seal_v45b() -> dict:
    report = read_seal_v45b(REPORT, EXPECTED["report"], EXPECTED["report_content"])
    prereg = read_seal_v45b(PREREG, EXPECTED["prereg"], EXPECTED["prereg_content"])
    attempt = read_seal_v45b(ATTEMPT, EXPECTED["attempt"], EXPECTED["attempt_content"])
    calibration = read_seal_v45b(
        CALIBRATION, EXPECTED["calibration"], EXPECTED["calibration_content"]
    )
    reliability = read_seal_v45b(
        RELIABILITY, EXPECTED["reliability"], EXPECTED["reliability_content"]
    )
    consensus = read_seal_v45b(
        CONSENSUS, EXPECTED["consensus"], EXPECTED["consensus_content"]
    )
    snapshots = report.get("update", {}).get("snapshots", [])
    final_identity = report.get("update", {}).get("final_identity", {})
    if (
        report.get("schema") != "matched-lora-es-train-only-report-v43g"
        or report.get("status")
        != "complete_robust_centered_rank_one_update_state_sealed"
        or report.get("shadow_dev_external_eval_ood_or_holdout_opened") is not False
        or prereg.get("status") != "preregistered_before_train_only_launch"
        or prereg.get("sealed_holdout_opened") is not False
        or prereg.get("shadow_dev_external_eval_ood_or_holdout_authorized") is not False
        or attempt.get("shadow_dev_external_eval_ood_or_holdout_opened") is not False
        or calibration.get("status") != "complete_before_population"
        or reliability.get("status") != "complete_before_update_decision"
        or reliability.get("reliability_gate", {}).get("passed") is not True
        or consensus.get("status") != "complete_before_acceptance_decision"
        or consensus.get("equivalence", {}).get("passed") is not True
        or len(snapshots) != 4
        or {item.get("rank") for item in snapshots} != {0, 1, 2, 3}
        or any(item.get("master_identity", {}).get("sha256")
               != EXPECTED["master_sha256"] for item in snapshots)
        or snapshots[0].get("weights_sha256") != EXPECTED["weights"]
        or snapshots[0].get("config_sha256") != EXPECTED["config"]
        or snapshots[0].get("readback_identity", {}).get("sha256")
        != EXPECTED["master_sha256"]
        or snapshots[0].get("readback_verified") is not True
        or snapshots[0].get("written") is not True
        or any(item.get("written") is not False for item in snapshots[1:])
        or final_identity.get("sha256") != EXPECTED["master_sha256"]
        or final_identity.get("tensor_count") != 70
        or final_identity.get("elements") != 4_528_128
    ):
        raise RuntimeError("V45B V43G successful snapshot provenance changed")
    return {
        "schema": "v43g-successful-canonical-snapshot-provenance-v45b",
        "report_file_sha256": EXPECTED["report"],
        "report_content_sha256": EXPECTED["report_content"],
        "preregistration_file_sha256": EXPECTED["prereg"],
        "attempt_file_sha256": EXPECTED["attempt"],
        "calibration_file_sha256": EXPECTED["calibration"],
        "population_reliability_file_sha256": EXPECTED["reliability"],
        "post_update_consensus_file_sha256": EXPECTED["consensus"],
        "snapshot_actor_count": 4,
        "snapshot_master_sha256": EXPECTED["master_sha256"],
        "state_complete": True,
        "selection_data_opened": False,
        "heldout_or_holdout_opened": False,
    }


def _inject_v45b():
    old_candidate = prior.CANDIDATE_SPECS_V44A.get(ARM)
    old_expected = prior.EXPECTED_V44A.get(ARM)
    old_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[ARM] = (SOURCE, None, OUTPUT)
    prior.EXPECTED_V44A[ARM] = {
        "weights": EXPECTED["weights"], "config": EXPECTED["config"],
        "master_sha256": EXPECTED["master_sha256"],
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v45b() if requested == ARM else old_seal(requested)
    )
    return old_candidate, old_expected, old_seal


def _restore_v45b(saved) -> None:
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


def audit_source_v45b() -> dict:
    saved = _inject_v45b()
    try:
        return prior.audit_source_v44a(ARM)
    finally:
        _restore_v45b(saved)


def stage_v45b(output: Path | None = None) -> dict:
    audit_source_v45b()
    saved = _inject_v45b()
    try:
        return prior.stage_one_v44a(
            ARM, Path(output).resolve() if output is not None else None
        )
    finally:
        _restore_v45b(saved)


def main() -> int:
    manifest = stage_v45b()
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
