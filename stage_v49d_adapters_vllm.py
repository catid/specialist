#!/usr/bin/env python3
"""Audit and CPU-stage the two completed V49D adapters for vLLM."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import stage_candidate_adapters_vllm_v44a as prior


ROOT = Path(__file__).resolve().parent
RUN_ROOT = (
    ROOT / "experiments/sft_controls/v49d_v434_sampling_midpoint_lr5p5e5"
).resolve()
ARMS = ("v434_equal", "v434_source50")
SOURCES = {
    "v434_equal": (RUN_ROOT / "v434_equal_r32_seed17_init20260715041/final").resolve(),
    "v434_source50": (
        RUN_ROOT / "v434_source50_r32_seed17_init20260715041/final"
    ).resolve(),
}
REPORTS = {
    arm: (RUN_ROOT / f"runtime_report_{arm}.json").resolve() for arm in ARMS
}
ATTEMPTS = {
    arm: (RUN_ROOT / f"attempt_{arm}.json").resolve() for arm in ARMS
}
GPU_LOGS = {
    arm: (RUN_ROOT / f"gpu_activity_{arm}.jsonl").resolve() for arm in ARMS
}
PREREGISTRATION = (RUN_ROOT / "preregistration_v49d.json").resolve()
OUTPUTS = {
    arm: (
        ROOT / "experiments/eggroll_es_hpo/staged_adapters"
        / f"{arm}_sft_qwen35_vllm_namespace_v49d"
    ).resolve()
    for arm in ARMS
}
EXPECTED = {
    "v434_equal": {
        "weights": "0d6efd4d5be626f41cdd711843f799b9b3c09e9ecf6a7a8e6e9aeeff09f6dc5b",
        "config": "b2bf4816802328893825be9ed7634b109ed400e17774f6bb98defe2e26ad06b5",
        "report": "0f669d188046849ccb6a3013938f9979214711f5e83c0d9e54290c9c20c850d8",
        "report_content": "a8de8805238335ba27db11359fc9f75ad65431d64b25f6a6c10eb5e0f3bdba0e",
        "attempt": "0ccdd6efa829cc449c70f255f48d1a369c662f936c35a161e0225572d2794688",
        "attempt_content": "23bdce550604555c76a2a8842c1862d7e31daa3eabf3b3553131ea33674ec103",
        "gpu_log": "b3c0ba4ab983487b9cc8a6c64a78d120a6ae5e5801a5b5c36845c898485d1c78",
        "weight_identity": "a8cbc597f865123100de870fdbc22a2529ed5ec3534531f897d27763baa4492f",
    },
    "v434_source50": {
        "weights": "2dafc6ca308ecf2243dbd7496100c55d7bb419ad3f5f7eb949c5da2bcce48ef8",
        "config": "752e31a157428c91c68ff23181fe057f74a476a766263017a51dc22c7421cd53",
        "report": "ed8791f89df57812904a3081abef5249f875c9b9c98db292279ae148705a5df2",
        "report_content": "17216481e5e7ecd885a87100827987dcef731c4406e4cba91b100bbd44915bb6",
        "attempt": "e3800c23fc65e3f76ab747cc802c809f63d219e9e3791eae4334a28732113af5",
        "attempt_content": "555dd80a3e150cb1e9af2326152d73b95b78109e1fa0f5df4ae9348367ce2d32",
        "gpu_log": "e4af41a4b9406969323abedb7adb5710449f05ea83058c09cdc7a472f3aa4b55",
        "weight_identity": "725e115ac0e4c4f20653e6c95f163edda0602621ff9d83f0c7d94b5f81b29890",
    },
}
PREREGISTRATION_FILE_SHA256 = (
    "4512e15987550c5b20cc0bd2a9c230a981f0c02f0fea877e5ee263ff747a0cde"
)
PREREGISTRATION_CONTENT_SHA256 = (
    "6400e2ee17922fa30e57117ed7332bb2fdf3456267e9c0b710796b69d764efba"
)


def _read_self_hashed(path: Path, file_sha: str, content_sha: str) -> dict:
    if prior.file_sha256_v44a(path) != file_sha:
        raise RuntimeError(f"V49D completion seal changed: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    compact = {k: v for k, v in value.items() if k != "content_sha256_before_self_field"}
    if (
        value.get("content_sha256_before_self_field") != content_sha
        or prior.canonical_sha256_v44a(compact) != content_sha
    ):
        raise RuntimeError(f"V49D self-hash changed: {path}")
    return value


def source_seal_v49d(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError(arm)
    expected = EXPECTED[arm]
    report = _read_self_hashed(
        REPORTS[arm], expected["report"], expected["report_content"]
    )
    attempt = _read_self_hashed(
        ATTEMPTS[arm], expected["attempt"], expected["attempt_content"]
    )
    prereg = _read_self_hashed(
        PREREGISTRATION,
        PREREGISTRATION_FILE_SHA256,
        PREREGISTRATION_CONTENT_SHA256,
    )
    artifacts = report.get("artifacts", {})
    outputs = artifacts.get("output_file_sha256", {})
    recipe = report.get("recipe", {})
    gpu = report.get("gpu_activity", {})
    observed = report.get("observed_weighting_audit", {}).get("value", {})
    prereg_arm = prereg.get("training_arms", {}).get(arm, {})
    if (
        report.get("schema") != "specialist-v434-sampling-midpoint-runtime-v49d"
        or report.get("arm") != arm
        or report.get("status")
        != "complete_train_only_state_sealed_non_train_unopened"
        or report.get("validation_ood_or_holdout_opened") is not False
        or report.get("shadow_semantics_opened") is not False
        or report.get("expected_optimizer_steps") != 48
        or report.get("output_validation", {}).get("final_global_step") != 48
        or report.get("output_validation", {}).get("final_epoch") != 3.0
        or recipe.get("learning_rate") != 5.5e-5
        or recipe.get("prompt_mode") != "es_exact"
        or recipe.get("world_size") != 4
        or recipe.get("target_layers") != "20,21,22,23"
        or recipe.get("rank") != 32
        or recipe.get("max_steps") != 48
        or observed.get("identity_sha256") != expected["weight_identity"]
        or gpu.get("physical_gpu_ids") != [0, 1, 2, 3]
        or gpu.get("all_four_model_resident") is not True
        or gpu.get("all_four_positive_activity") is not True
        or gpu.get("activity_attributed_to_torchrun_tree") is not True
        or outputs.get("final/adapter_model.safetensors") != expected["weights"]
        or outputs.get("checkpoint-48/adapter_model.safetensors")
        != expected["weights"]
        or outputs.get("final/adapter_config.json") != expected["config"]
        or artifacts.get("gpu_log_sha256") != expected["gpu_log"]
        or prior.file_sha256_v44a(GPU_LOGS[arm]) != expected["gpu_log"]
        or prior.file_sha256_v44a(SOURCES[arm] / "adapter_model.safetensors")
        != expected["weights"]
        or prior.file_sha256_v44a(SOURCES[arm] / "adapter_config.json")
        != expected["config"]
        or prereg.get("schema")
        != "specialist-v434-sampling-midpoint-preregistration-v49d"
        or prereg.get("evaluation_launch_authorized") is not False
        or prereg_arm.get("recipe", {}).get("expected_optimizer_steps") != 48
        or prereg_arm.get("weighting_audit", {}).get(
            "runtime_compact", {}
        ).get("identity_sha256") != expected["weight_identity"]
        or prereg.get("access_firewall", {}).get("eval_ood_holdout_opened")
        is not False
        or attempt.get("schema")
        != "specialist-v434-sampling-midpoint-attempt-v49d"
        or attempt.get("arm") != arm
        or attempt.get("status") != "complete"
        or attempt.get("returncode") != 0
        or attempt.get("final_report_sha256") != expected["report"]
    ):
        raise RuntimeError(f"V49D {arm} train-only provenance changed")
    return {
        "schema": "v49d-train-only-success-provenance",
        "arm": arm,
        "source_weights_sha256": expected["weights"],
        "source_config_sha256": expected["config"],
        "report_file_sha256": expected["report"],
        "report_content_sha256": expected["report_content"],
        "attempt_file_sha256": expected["attempt"],
        "attempt_content_sha256": expected["attempt_content"],
        "gpu_log_file_sha256": expected["gpu_log"],
        "weight_identity_sha256": expected["weight_identity"],
        "completed_steps": 48,
        "all_four_training_gpus_attributed_positive": True,
        "shadow_ood_holdout_or_heldout_opened": False,
    }


@contextmanager
def _injected(arm: str):
    previous_candidate = prior.CANDIDATE_SPECS_V44A.get(arm)
    previous_expected = prior.EXPECTED_V44A.get(arm)
    previous_seal = prior._source_seal_v44a
    prior.CANDIDATE_SPECS_V44A[arm] = (
        SOURCES[arm], REPORTS[arm], OUTPUTS[arm]
    )
    prior.EXPECTED_V44A[arm] = {
        "weights": EXPECTED[arm]["weights"],
        "config": EXPECTED[arm]["config"],
    }
    prior._source_seal_v44a = lambda requested: (
        source_seal_v49d(arm) if requested == arm else previous_seal(requested)
    )
    try:
        yield
    finally:
        if previous_candidate is None:
            prior.CANDIDATE_SPECS_V44A.pop(arm, None)
        else:
            prior.CANDIDATE_SPECS_V44A[arm] = previous_candidate
        if previous_expected is None:
            prior.EXPECTED_V44A.pop(arm, None)
        else:
            prior.EXPECTED_V44A[arm] = previous_expected
        prior._source_seal_v44a = previous_seal


def audit_source_v49d(arm: str) -> dict:
    source_seal_v49d(arm)
    with _injected(arm):
        return prior.audit_source_v44a(arm)


def stage_one_v49d(arm: str, output: Path | None = None) -> dict:
    audit_source_v49d(arm)
    with _injected(arm):
        return prior.stage_one_v44a(
            arm, Path(output).resolve() if output is not None else None
        )


def main() -> int:
    results = {}
    for arm in ARMS:
        manifest = stage_one_v49d(arm)
        results[arm] = {
            "directory": manifest["artifact"]["directory"],
            "weights_sha256": manifest["artifact"]["weights_file_sha256"],
            "config_sha256": manifest["artifact"]["adapter_config_file_sha256"],
            "manifest_file_sha256": prior.file_sha256_v44a(
                OUTPUTS[arm] / "stage_manifest_v44a.json"
            ),
            "manifest_content_sha256": manifest[
                "content_sha256_before_self_field"
            ],
            "transformed_identity_sha256": manifest[
                "transformed_identity"
            ]["sha256"],
        }
    print(json.dumps({
        "staged": results,
        "protected_semantic_access_count": 0,
        "heldout_or_holdout_opened": False,
        "gpu_accessed": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
